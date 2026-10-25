"""Metal PPO replay with an explicit, checkpoint-compatible Torch AdamW bridge.

Only frozen features cross frameworks per episode. Trainable policy and optimizer
moments stay on Metal throughout an update; Torch is synchronized at its boundary.
"""

import math
from copy import deepcopy

import numpy as np
import torch


def require_mlx():
    """Fail explicitly when the optional backend cannot execute on Metal."""
    try:
        import mlx.core as mx
    except ImportError as error:
        raise RuntimeError("MLX training requires the optional mlx dependency on Apple Silicon") \
            from error
    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal unavailable; training never silently falls back")
    mx.set_default_device(mx.gpu)
    return mx


def _array(tensor):
    return require_mlx().array(tensor.detach().cpu().numpy())


class MLXAdamW:
    """AdamW with Torch's bias correction and per-parameter step counters.

    The canonical optimizer owns configuration/checkpoint IDs. This bridge owns
    live Metal moments, and never calls a Torch optimizer step or patches it.
    """

    def __init__(self, agent, optimizer):
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("MLX training requires a canonical Torch AdamW optimizer")
        self.mx = require_mlx()
        self.canonical = optimizer
        names = {id(parameter): name for name, parameter in agent.named_parameters()
                 if parameter.requires_grad}
        self.parameter_names = names
        self.groups = {}
        self.state = {}
        for group in optimizer.param_groups:
            if any(group.get(option, False) for option in
                   ("amsgrad", "maximize", "capturable", "differentiable")):
                raise ValueError("MLX bridge supports ordinary bias-corrected AdamW only")
            for parameter in group["params"]:
                if id(parameter) not in names:
                    raise ValueError("optimizer parameters must be trainable policy parameters")
                name = names[id(parameter)]
                if name in self.groups:
                    raise ValueError("optimizer contains duplicate parameters")
                self.groups[name] = group
                saved = optimizer.state.get(parameter, {})
                if saved:
                    if set(saved) != {"step", "exp_avg", "exp_avg_sq"}:
                        raise ValueError("unsupported or incomplete canonical AdamW state")
                    step = float(saved["step"])
                    if not math.isfinite(step) or step < 0 or int(step) != step:
                        raise ValueError("AdamW step must be a finite nonnegative integer")
                    for key in ("exp_avg", "exp_avg_sq"):
                        value = saved[key]
                        if (value.shape != parameter.shape or value.dtype != parameter.dtype
                                or not torch.isfinite(value).all()):
                            raise ValueError(f"invalid canonical AdamW moment: {name}/{key}")
                    self.state[name] = {"step": int(step),
                                        "exp_avg": _array(saved["exp_avg"]),
                                        "exp_avg_sq": _array(saved["exp_avg_sq"])}
        if set(self.groups) != set(names.values()):
            raise ValueError("optimizer must cover all trainable policy parameters")
        self.mx.eval(self.state)

    def step(self, weights, gradients):
        mx = self.mx
        updated = dict(weights)
        pending = {}
        for name, gradient in gradients.items():
            if name not in self.groups:
                raise ValueError(f"gradient lacks canonical optimizer parameter: {name}")
            weight, group = weights[name], self.groups[name]
            if gradient.shape != weight.shape:
                raise ValueError(f"gradient shape mismatch: {name}")
            state = self.state.get(name)
            if state is None:
                state = {"step": 0, "exp_avg": mx.zeros_like(weight),
                         "exp_avg_sq": mx.zeros_like(weight)}
            step = state["step"] + 1
            beta1, beta2 = group["betas"]
            moment = state["exp_avg"] * beta1 + gradient * (1 - beta1)
            variance = state["exp_avg_sq"] * beta2 + mx.square(gradient) * (1 - beta2)
            # Torch AdamW always applies bias correction; MLX's default AdamW does not.
            denominator = mx.sqrt(variance) / math.sqrt(1 - beta2 ** step) + group["eps"]
            updated[name] = (weight * (1 - group["lr"] * group["weight_decay"])
                             - group["lr"] / (1 - beta1 ** step) * moment / denominator)
            pending[name] = {"step": step, "exp_avg": moment, "exp_avg_sq": variance}
        mx.eval(updated, pending)
        finite_values = [*updated.values(),
                         *(state[key] for state in pending.values()
                           for key in ("exp_avg", "exp_avg_sq"))]
        if not bool(mx.all(mx.stack([mx.all(mx.isfinite(value)) for value in finite_values]))):
            raise FloatingPointError("nonfinite parameters after MLX AdamW update")
        self.state.update(pending)
        return updated

    def sync_to_torch(self):
        """Export moments using the canonical optimizer's original parameter IDs."""
        saved = {"state": {},
                 "param_groups": deepcopy(self.canonical.state_dict()["param_groups"])}
        for live_group, saved_group in zip(self.canonical.param_groups,
                                           saved["param_groups"], strict=True):
            for parameter, identifier in zip(live_group["params"], saved_group["params"],
                                             strict=True):
                name = self.parameter_names[id(parameter)]
                if name in self.state:
                    state = self.state[name]
                    saved["state"][identifier] = {
                        "step": torch.tensor(float(state["step"]), dtype=torch.float32),
                        "exp_avg": torch.from_numpy(np.array(state["exp_avg"], copy=True)),
                        "exp_avg_sq": torch.from_numpy(np.array(state["exp_avg_sq"], copy=True)),
                    }
        self.canonical.load_state_dict(saved)
        for group in self.canonical.param_groups:
            for parameter in group["params"]:
                self.groups[self.parameter_names[id(parameter)]] = group


class MLXPPOBackend:
    def __init__(self, agent, optimizer, *, cache_limit_bytes=256 * 1024**2):
        self.mx = require_mlx()
        if type(cache_limit_bytes) is not int or cache_limit_bytes < 0:
            raise ValueError("MLX cache limit must be a nonnegative number of bytes")
        self.mx.set_cache_limit(cache_limit_bytes)
        from carryrl.agent.mlx_policy import MLXPolicy

        self.agent = agent
        self.policy = MLXPolicy.from_torch(agent)
        self.optimizer = MLXAdamW(agent, optimizer)

    def apply_gradients(self, gradients, max_grad_norm, active_names=None):
        mx = self.mx
        gradients = gradients["weights"]
        if active_names is not None:
            gradients = {name: gradient for name, gradient in gradients.items()
                         if name in active_names}
        if not gradients:
            raise ValueError("MLX update requires at least one active gradient")
        norm = mx.sqrt(sum(mx.sum(mx.square(gradient)) for gradient in gradients.values()))
        mx.eval(norm)
        value = float(norm)
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite MLX gradient norm")
        # Same epsilon and clipping convention as torch.nn.utils.clip_grad_norm_.
        coefficient = min(1.0, max_grad_norm / (value + 1e-6))
        clipped = {name: gradient * coefficient for name, gradient in gradients.items()}
        updated = self.optimizer.step(self.policy.parameters()["weights"], clipped)
        self.policy.update({"weights": updated})
        return value

    def sync_to_torch(self):
        self.policy.export_to_torch(self.agent)
        self.optimizer.sync_to_torch()


def _prepare_instructions(agent, episodes, instructions):
    for episode in episodes:
        instruction = episode.instruction
        if instruction not in instructions:
            feature = agent._encode_instruction(instruction) if instruction is not None else None
            instructions[instruction] = _array(feature) if feature is not None else None


def prepare_episodes(agent, episodes, advantages, returns, *, instruction_cache=None):
    """Transfer one replay batch; share immutable instruction features across batches."""
    prepared = []
    instructions = {} if instruction_cache is None else instruction_cache
    _prepare_instructions(agent, episodes, instructions)
    for episode, advantage, target in zip(episodes, advantages, returns, strict=True):
        instruction = episode.instruction
        prepared.append({"patches": _array(episode.patches), "actions": episode.actions,
                         "old_log_prob": _array(episode.old_log_prob),
                         "advantage": _array(advantage), "target": _array(target),
                         "instruction_features": instructions[instruction],
                         "task": episode.task})
    return prepared


def ppo_loss(policy, episodes, config, *, normalization_counts=None, component=None):
    """Whole-episode PPO, optionally contributing to a larger minibatch's objective.

    Microbatches use the full minibatch's task transition counts so summing
    their losses and gradients preserves both supported weighting conventions.
    """
    import mlx.core as mx

    counts = normalization_counts
    if counts is None:
        counts = {}
        for episode in episodes:
            counts[episode["task"]] = counts.get(episode["task"], 0) + len(episode["actions"])
    total = sum(counts.values())
    metrics = dict.fromkeys(("policy_loss", "value_loss", "entropy", "approx_kl",
                            "clip_fraction"), 0.0)
    metrics.update(_ratio_sum=mx.array(0.),
                   _ratio_min=mx.array(float("inf")), _ratio_max=mx.array(0.),
                   _log_ratio_abs_max=mx.array(0.))
    loss, valid = mx.array(0.0), mx.array(True)
    for episode in episodes:
        evaluated = policy.evaluate_episode(episode["patches"], episode["actions"],
                                             instruction_features=episode["instruction_features"])
        log_prob, value, entropy = (evaluated[name] for name in ("log_prob", "value", "entropy"))
        if any(tensor.shape != (len(episode["actions"]),)
               for tensor in (log_prob, value, entropy)):
            raise FloatingPointError("invalid complete-episode policy replay")
        log_ratio = log_prob - episode["old_log_prob"]
        ratio = mx.exp(log_ratio)
        for tensor in (log_prob, value, entropy, ratio):
            valid = valid & mx.all(mx.isfinite(tensor))
        advantage = episode["advantage"]
        policy_loss = -mx.minimum(ratio * advantage,
                                 mx.clip(ratio, 1 - config.clip_range, 1 + config.clip_range)
                                 * advantage)
        value_loss = 0.5 * mx.square(value - episode["target"])
        denominator = (len(counts) * counts[episode["task"]]
                       if config.ppo_task_balance else total)
        combined = {"actor": policy_loss, "value": config.value_coef * value_loss,
                    "entropy": -config.entropy_coef * entropy}
        terms = sum(combined.values()) if component is None else combined[component]
        weight = episode.get("objective_weight")
        loss = loss + (mx.sum(terms) / denominator if weight is None else mx.sum(terms) * weight)
        metrics["_ratio_sum"] += mx.sum(ratio)
        metrics["_ratio_min"] = mx.minimum(metrics["_ratio_min"], mx.min(ratio))
        metrics["_ratio_max"] = mx.maximum(metrics["_ratio_max"], mx.max(ratio))
        metrics["_log_ratio_abs_max"] = mx.maximum(
            metrics["_log_ratio_abs_max"], mx.max(mx.abs(log_ratio)))
        for name, values in (("policy_loss", policy_loss), ("value_loss", value_loss),
                             ("entropy", entropy), ("approx_kl", ratio - 1 - log_ratio),
                             ("clip_fraction", mx.abs(ratio - 1) > config.clip_range)):
            metrics[name] = metrics[name] + (mx.sum(values) / denominator if weight is None
                                              else mx.sum(values) * weight)
    return loss, metrics, valid & mx.isfinite(loss)


def _minibatch_gradients(agent, episodes, indices, advantages, returns, config, *,
                        mx, loss_and_grad, counts, instruction_cache, episode_weights=None):
    """Accumulate one objective while retaining only one episode chunk's graph."""
    size = config.ppo_microbatch_episodes or len(indices)
    gradients, diagnostics = None, None
    has_instruction = has_memory_write = False
    for start in range(0, len(indices), size):
        selected = indices[start:start + size]
        prepared = prepare_episodes(
            agent, [episodes[index] for index in selected],
            [advantages[index] for index in selected], [returns[index] for index in selected],
            instruction_cache=instruction_cache)
        if episode_weights is not None:
            for index, episode in zip(selected, prepared, strict=True):
                episode["objective_weight"] = episode_weights[index]
        has_instruction |= any(episode["instruction_features"] is not None
                               for episode in prepared)
        has_memory_write |= any(len(episode["actions"]) > config.agent.segment_length
                                for episode in prepared)
        (loss, current, valid), chunk_gradients = loss_and_grad(prepared, counts)
        mx.eval(loss, current, valid, chunk_gradients)
        if not bool(valid):
            raise FloatingPointError("nonfinite or invalid MLX PPO replay/loss")
        if gradients is None:
            gradients, diagnostics = chunk_gradients, current
        else:
            gradients = {"weights": {
                name: gradient + chunk_gradients["weights"][name]
                for name, gradient in gradients["weights"].items()}}
            diagnostics = {
                name: (mx.minimum(value, current[name]) if name == "_ratio_min" else
                       mx.maximum(value, current[name]) if name in {
                           "_ratio_max", "_log_ratio_abs_max"} else value + current[name])
                for name, value in diagnostics.items()}

        # Materialize the running sum as well as the chunk. Neither the replay
        # graph nor a chain of lazy additions may span successive microbatches.
        mx.eval(gradients, diagnostics)
        del prepared, loss, current, valid, chunk_gradients
    active = set(gradients["weights"])
    if not has_instruction:
        active.discard("instruction_projection.weight")
    if not has_memory_write:
        active.discard("temporal.write_tokens")
    return {name: float(value) for name, value in diagnostics.items()}, gradients, active


def _mlx_component_probe(agent, policy, episodes, indices, advantages, returns, weights, config,
                         instruction_cache):
    import mlx.core as mx
    import mlx.nn as nn

    from carryrl.training.ppo_diagnostics import preserve_rng

    components = {}
    combined = None
    with preserve_rng(mx=mx):
        for component in ("actor", "value", "entropy"):
            accumulated = None
            loss_and_grad = nn.value_and_grad(
                policy, lambda prepared: ppo_loss(policy, prepared, config, component=component))
            # One complete episode bounds all feature transfers and replay graphs.
            for index, weight in zip(indices, weights, strict=True):
                prepared = prepare_episodes(
                    agent, [episodes[index]], [advantages[index]], [returns[index]],
                    instruction_cache=instruction_cache)
                prepared[0]["objective_weight"] = weight
                (loss, _, valid), gradients = loss_and_grad(prepared)
                mx.eval(loss, valid, gradients)
                if not bool(valid):
                    raise FloatingPointError("nonfinite MLX PPO gradient probe")
                accumulated = gradients["weights"] if accumulated is None else {
                    name: value + gradients["weights"][name] for name, value in accumulated.items()}
                mx.eval(accumulated)
                del prepared, loss, valid, gradients
            norm = mx.sqrt(sum(mx.sum(mx.square(value)) for value in accumulated.values()))
            mx.eval(norm)
            components[component] = float(norm)
            combined = accumulated if combined is None else {
                name: value + accumulated[name] for name, value in combined.items()}
            mx.eval(combined)
        norm = mx.sqrt(sum(mx.sum(mx.square(value)) for value in combined.values()))
        mx.eval(norm)
        components["combined"] = float(norm)
    return components


def _mlx_post_update_kl(agent, policy, episodes, indices, advantages, returns, weights, config,
                        instruction_cache):
    import mlx.core as mx

    from carryrl.training.ppo_diagnostics import preserve_rng

    kl, mass = 0., 0.
    with preserve_rng(mx=mx):
        for index, weight in zip(indices, weights, strict=True):
            prepared = prepare_episodes(
                agent, [episodes[index]], [advantages[index]], [returns[index]],
                instruction_cache=instruction_cache)
            prepared[0]["objective_weight"] = weight
            loss, metrics, valid = ppo_loss(policy, prepared, config)
            mx.eval(loss, metrics, valid)
            if not bool(valid):
                raise FloatingPointError("nonfinite MLX PPO post-update probe")
            kl += float(metrics["approx_kl"])
            mass += len(episodes[index]) * weight
            del prepared, loss, metrics, valid
    return kl / mass if config.ppo_normalization == "rollout" else kl


def update_ppo_mlx(agent, optimizer, episodes, config, *, backend=None, cycle: int = 1):
    require_mlx()
    import mlx.nn as nn

    from carryrl.training.ppo_diagnostics import (
        advantage_diagnostics,
        objective_weights,
        probe_due,
        update_means,
    )
    from carryrl.training.sequence import _episode_minibatches, _ppo_targets

    total, task_totals, advantages, returns = _ppo_targets(episodes, config)
    diagnostics = advantage_diagnostics(episodes, config)
    backend = backend if backend is not None else MLXPPOBackend(agent, optimizer)
    if backend.agent is not agent or backend.optimizer.canonical is not optimizer:
        raise ValueError("MLX backend belongs to another policy or optimizer")
    mx = backend.mx
    instruction_cache = {}
    _prepare_instructions(agent, episodes, instruction_cache)
    loss_and_grad = nn.value_and_grad(backend.policy,
                                     lambda batch, counts: ppo_loss(
                                         backend.policy, batch, config,
                                         normalization_counts=counts))
    metrics, updates, probes, optimizer_steps = [], [], [], 0
    first_ratio = None
    task_weight_sums = dict.fromkeys(task_totals, 0.0)
    task_replays = dict.fromkeys(task_totals, 0)
    stopped = False
    agent.train()
    try:
        for epoch in range(config.n_epochs):
            sums = dict.fromkeys(("policy_loss", "value_loss", "entropy", "approx_kl",
                                  "clip_fraction", "gradient_norm"), 0.0)
            observed_transitions = epoch_steps = observed_batches = 0
            epoch_updates = []
            minibatches = _episode_minibatches(episodes, config.ppo_minibatch_episodes,
                                              config.ppo_task_balance)
            if config.ppo_normalization == "rollout":
                for indices in minibatches:
                    objective_weights(episodes, indices, config)
            for batch_index, indices in enumerate(minibatches):
                counts, weights, task_masses = objective_weights(episodes, indices, config)
                episode_weights = dict(zip(indices, weights, strict=True)) \
                    if config.ppo_normalization == "rollout" else None
                batch, gradients, active = _minibatch_gradients(
                    agent, episodes, indices, advantages, returns, config,
                    mx=mx, loss_and_grad=loss_and_grad, counts=counts,
                    instruction_cache=instruction_cache, episode_weights=episode_weights)
                batch_total = sum(counts.values())
                ratio_mean = batch.pop("_ratio_sum") / batch_total
                ratio_statistics = {
                    "mean": ratio_mean,
                    "min": batch.pop("_ratio_min"), "max": batch.pop("_ratio_max"),
                    "log_ratio_abs_max": batch.pop("_log_ratio_abs_max"),
                    "transitions": batch_total, "scope": "selected transitions before update",
                }
                if first_ratio is None:
                    first_ratio = ratio_statistics
                objective_mass = sum(task_masses.values())
                if config.ppo_normalization == "rollout":
                    batch["approx_kl"] /= objective_mass
                    batch["clip_fraction"] /= objective_mass
                observed_batches += 1
                observed_transitions += batch_total
                for name in batch:
                    sums[name] += batch[name] * batch_total
                # Reject before any moments, counters, decay or policy mutation.
                if config.target_kl > 0 and batch["approx_kl"] > config.target_kl:
                    stopped = True
                    del gradients
                    break
                do_probe = epoch == 0 and batch_index == 0 and probe_due(config, cycle)
                components = (_mlx_component_probe(
                    agent, backend.policy, episodes, indices, advantages, returns, weights,
                    config, instruction_cache) if do_probe else None)
                gradient_norm = backend.apply_gradients(gradients, config.max_grad_norm,
                                                        active_names=active)
                optimizer_steps += 1
                epoch_steps += 1
                for task, count in counts.items():
                    task_weight_sums[task] += task_masses[task]
                    task_replays[task] += count
                sums["gradient_norm"] += gradient_norm * batch_total
                record = {"epoch": epoch, "minibatch": batch_index, "update": optimizer_steps,
                          **batch, "gradient_norm": gradient_norm,
                          "clip_coefficient": min(
                              1., config.max_grad_norm / (gradient_norm + 1e-6)),
                          "ratio": ratio_statistics, "task_loss_weights": task_masses,
                          "objective_mass": objective_mass}
                updates.append(record)
                epoch_updates.append(record)
                if do_probe:
                    probes.append({"cycle": cycle, "update": optimizer_steps,
                                   "component_gradient_norms": components,
                                   "gradient_norm": gradient_norm,
                                   "clip_coefficient": record["clip_coefficient"],
                                   "pre_update_kl": batch["approx_kl"],
                                   "post_update_kl": _mlx_post_update_kl(
                                       agent, backend.policy, episodes, indices, advantages,
                                       returns, weights, config, instruction_cache)})
                del gradients
            metrics.append({"epoch": epoch,
                            **{name: value / observed_transitions for name, value in sums.items()},
                            **update_means(epoch_updates),
                            "optimizer_steps": epoch_steps, "minibatches": observed_batches,
                            "replayed_transitions": observed_transitions,
                            "kl_early_stop": stopped})
            if stopped:
                break
    finally:
        backend.sync_to_torch()
        agent.eval()
    return {"valid_transitions": total, "episodes": len(episodes), "epochs": metrics,
            "optimizer_steps": optimizer_steps, "kl_early_stop": stopped,
            "training_backend": "mlx", "trainable_device": "Metal GPU",
            "task_transitions": task_totals, "updates": updates,
            "first_minibatch_ratio": first_ratio, "gradient_probes": probes,
            "advantage_diagnostics": diagnostics, "ppo_normalization": config.ppo_normalization,
            "optimizer_task_loss_weights": {
                task: weight / optimizer_steps for task, weight in task_weight_sums.items()
            } if optimizer_steps else None,
            "optimizer_task_loss_weights_scope": "mean objective coefficients over applied "
            "optimizer steps; excludes rejected KL minibatches; not gradient attribution",
            "optimizer_task_transition_replays": task_replays,
            "loss_weighting": ("equal task rollout mean with episode selection correction"
                               if config.ppo_normalization == "rollout" else
                               "equal task mean per minibatch" if config.ppo_task_balance else
                               "valid transitions per minibatch")}
