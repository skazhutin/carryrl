"""Shared PPO normalization, objective weights and read-only attribution probes."""

import math
import random
from collections import Counter
from contextlib import contextmanager

import numpy as np
import torch


def raw_targets(episodes, config, *, rewards=None):
    from carryrl.training.sequence import gae

    advantages, returns = [], []
    for index, episode in enumerate(episodes):
        advantage, target = gae(
            episode.rewards if rewards is None else rewards[index], episode.old_values,
            episode.terminated, episode.truncated, episode.bootstrap, config.gamma,
            config.gae_lambda,
        )
        advantages.append(advantage)
        returns.append(target)
    return advantages, returns


def weighted_moments(episodes, tensors, config):
    """One global mean and population standard deviation, with equal task mass."""
    if config.ppo_task_balance:
        totals = Counter()
        for episode in episodes:
            totals[episode.task] += len(episode)
        pairs = list(zip(episodes, tensors, strict=True))
        mean = sum(value.sum() / totals[episode.task] for episode, value in pairs) / len(totals)
        variance = sum((value - mean).square().sum() / totals[episode.task]
                       for episode, value in pairs) / len(totals)
        std = variance.sqrt()
    else:
        combined = torch.cat(tensors)
        mean, std = combined.mean(), combined.std(unbiased=False)
    return mean, std, std.clamp_min(config.advantage_std_floor)


def objective_weights(episodes, indices, config):
    """Per-transition coefficients for this selected set of complete episodes.

    Rollout mode estimates the fixed equal-task rollout objective from uniformly
    shuffled task episodes. Episode selection correction is distinct from the
    task transition denominator and therefore handles unequal episode lengths.
    """
    counts, selected_episodes = Counter(), Counter()
    for index in indices:
        episode = episodes[index]
        counts[episode.task] += len(episode)
        selected_episodes[episode.task] += 1
    if config.ppo_normalization == "rollout":
        totals, task_episodes = Counter(), Counter(episode.task for episode in episodes)
        for episode in episodes:
            totals[episode.task] += len(episode)
        if set(counts) != set(totals):
            raise ValueError("rollout normalization requires every minibatch to contain all tasks")
        weights = [task_episodes[episodes[index].task] / selected_episodes[episodes[index].task]
                   / (len(totals) * totals[episodes[index].task]) for index in indices]
    else:
        weights = [1 / (len(counts) * counts[episodes[index].task]
                        if config.ppo_task_balance else sum(counts.values())) for index in indices]
    masses = dict.fromkeys(counts, 0.0)
    for index, weight in zip(indices, weights, strict=True):
        masses[episodes[index].task] += len(episodes[index]) * weight
    if config.ppo_normalization == "minibatch":
        masses = {task: 1 / len(counts) if config.ppo_task_balance else count / sum(counts.values())
                  for task, count in counts.items()}
    return dict(counts), weights, masses


def stats(values):
    if not values:
        return {"count": 0, "sum": None, "mean": None, "std": None,
                "rms": None, "min": None, "max": None, "abs_max": None}
    values = torch.cat([value.detach().cpu().double().reshape(-1) for value in values])
    return {"count": len(values), "sum": float(values.sum()), "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "rms": float(values.square().mean().sqrt()), "min": float(values.min()),
            "max": float(values.max()), "abs_max": float(values.abs().max())}


def advantage_diagnostics(episodes, config):
    """Exact GAE counterfactual with fixed old critic values and episode boundaries."""
    raw, _ = raw_targets(episodes, config)
    mean, std, denominator = weighted_moments(episodes, raw, config)
    normalized = [(value - mean) / denominator for value in raw]
    tasks = sorted({episode.task for episode in episodes})
    component_names = sorted({name for episode in episodes
                              for name in (episode.reward_components or {})})
    task_stats = {}
    for task in tasks:
        selected = [index for index, episode in enumerate(episodes) if episode.task == task]
        components = {}
        for name in component_names:
            values = []
            for index in selected:
                component = (episodes[index].reward_components or {}).get(name)
                if component is not None:
                    if len(component) != len(episodes[index]):
                        raise ValueError("reward component length must match the complete episode")
                    values.append(torch.tensor(component))
            components[name] = stats(values) if values else None
        task_stats[task] = {
            "episodes": len(selected),
            "transitions": sum(len(episodes[index]) for index in selected),
            "raw_advantage": stats([raw[index] for index in selected]),
            "normalized_advantage": stats([normalized[index] for index in selected]),
            "reward": stats([episodes[index].rewards for index in selected]),
            "reward_components": components,
        }
    result = {"scope": "one shared global mean/std; equal task mass" if config.ppo_task_balance
              else "one shared global mean/std; equal transition mass",
              "mean": float(mean), "std": float(std), "denominator": float(denominator),
              "std_floor": config.advantage_std_floor,
              "floor_active": bool(std < config.advantage_std_floor), "tasks": task_stats}
    missing = [index for index, episode in enumerate(episodes)
               if any(name not in (episode.reward_components or {})
                      for name in ("control_bonus", "unsupported_penalty"))]
    if missing:
        result["auxiliary_counterfactual"] = {"status": "unavailable",
                                                "missing_component_episode_indices": missing}
        return result
    rewards = []
    for episode in episodes:
        removed = torch.tensor(episode.reward_components["control_bonus"],
                               dtype=episode.rewards.dtype, device=episode.rewards.device)
        removed = removed + torch.tensor(
            episode.reward_components["unsupported_penalty"],
            dtype=episode.rewards.dtype, device=episode.rewards.device,
        )
        if removed.shape != episode.rewards.shape:
            raise ValueError("auxiliary reward component shape must match episode rewards")
        rewards.append(episode.rewards - removed)
    counterfactual, _ = raw_targets(episodes, config, rewards=rewards)
    cf_mean, cf_std, cf_denominator = weighted_moments(episodes, counterfactual, config)
    cf_normalized = [(value - cf_mean) / cf_denominator for value in counterfactual]
    direct = [((value - other) - (mean - cf_mean)) / denominator
              for value, other in zip(raw, counterfactual, strict=True)]
    rescaling = [(value - cf_mean) * (1 / denominator - 1 / cf_denominator)
                 for value in counterfactual]
    differences = [value - other for value, other in zip(normalized, cf_normalized, strict=True)]
    cf_tasks = {}
    for task in tasks:
        selected = [index for index, episode in enumerate(episodes) if episode.task == task]
        cf_tasks[task] = {
            "raw_advantage": stats([counterfactual[index] for index in selected]),
            "normalized_advantage": stats([cf_normalized[index] for index in selected]),
            "direct_centered_effect": stats([direct[index] for index in selected]),
            "denominator_rescaling_effect": stats([rescaling[index] for index in selected]),
            "normalized_difference": stats([differences[index] for index in selected]),
            "sign_changes": sum(int((torch.sign(normalized[index])
                                      != torch.sign(cf_normalized[index])).sum())
                                for index in selected),
        }
    result["auxiliary_counterfactual"] = {
        "status": "measured", "removed_signed_components": ["control_bonus", "unsupported_penalty"],
        "old_values_held_fixed": True,
        "decomposition": "full-minus-counterfactual = centered raw GAE delta / full denominator "
        "+ centered counterfactual GAE * (1/full denominator - 1/counterfactual denominator)",
        "sign_change_definition": "sign(full) != sign(counterfactual), including zero",
        "mean": float(cf_mean), "std": float(cf_std),
        "denominator": float(cf_denominator),
        "floor_active": bool(cf_std < config.advantage_std_floor),
        "denominator_ratio_full_to_counterfactual": float(denominator / cf_denominator),
        "direct_centered_effect": stats(direct),
        "denominator_rescaling_effect": stats(rescaling),
        "normalized_difference": stats(differences),
        "sign_changes": sum(row["sign_changes"] for row in cf_tasks.values()),
        "tasks": cf_tasks,
    }
    return result


def probe_due(config, cycle):
    return (config.gradient_probe_every > 0
            and (cycle == 1 or cycle % config.gradient_probe_every == 0))


@contextmanager
def preserve_rng(*, mx=None):
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    metal = [mx.array(value) for value in mx.random.state] if mx is not None else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        if mps is not None:
            torch.mps.set_rng_state(mps)
        if metal is not None:
            mx.random.state = metal


def torch_component_probe(agent, episodes, indices, advantages, returns, weights, config):
    """Use autograd.grad without touching .grad, weights, optimizer state or RNG.

    Only one complete episode's graph is retained at a time, independent of the
    training minibatch size. Component gradients are accumulated before norms.
    """
    parameters = [parameter for parameter in agent.parameters() if parameter.requires_grad]
    accumulators = {name: [None] * len(parameters) for name in ("actor", "value", "entropy")}
    with preserve_rng():
        for index, weight in zip(indices, weights, strict=True):
            episode = episodes[index]
            options = ({"instruction": episode.instruction}
                       if episode.instruction is not None else {})
            result = agent.evaluate_episode(episode.patches, episode.actions, **options)
            ratio = (result["log_prob"] - episode.old_log_prob.to(result["log_prob"].device)).exp()
            advantage = advantages[index].to(ratio.device)
            components = {
                "actor": -torch.minimum(ratio * advantage,
                                         ratio.clamp(1 - config.clip_range, 1 + config.clip_range)
                                         * advantage).sum() * weight,
                "value": .5 * (result["value"] - returns[index].to(ratio.device)).square().sum()
                * config.value_coef * weight,
                "entropy": -config.entropy_coef * result["entropy"].sum() * weight,
            }
            for name, loss in components.items():
                if not loss.requires_grad:
                    continue
                gradients = torch.autograd.grad(
                    loss, parameters, retain_graph=name != "entropy", allow_unused=True,
                )
                for position, gradient in enumerate(gradients):
                    if gradient is not None:
                        old = accumulators[name][position]
                        accumulators[name][position] = gradient.detach().clone() if old is None \
                            else old + gradient.detach()
            del result, components, ratio, loss
    norms = {name: math.sqrt(sum(float(gradient.double().square().sum())
                                 for gradient in values if gradient is not None))
             for name, values in accumulators.items()}
    combined = []
    for position in range(len(parameters)):
        gradients = [values[position] for values in accumulators.values()
                     if values[position] is not None]
        if gradients:
            combined.append(sum(gradients))
    norms["combined"] = math.sqrt(sum(float(gradient.double().square().sum())
                                       for gradient in combined))
    return norms


def update_means(updates):
    names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "gradient_norm")
    return {"mean_update_" + name: sum(row[name] for row in updates) / len(updates)
            if updates else None for name in names}


def torch_post_update_kl(agent, episodes, indices, weights, config):
    result = 0.0
    with preserve_rng(), torch.no_grad():
        for index, weight in zip(indices, weights, strict=True):
            episode = episodes[index]
            options = ({"instruction": episode.instruction}
                       if episode.instruction is not None else {})
            logp = agent.evaluate_episode(episode.patches, episode.actions, **options)["log_prob"]
            log_ratio = logp - episode.old_log_prob.to(logp.device)
            result += float((log_ratio.exp() - 1 - log_ratio).sum()) * weight
    mass = sum(len(episodes[index]) * weight
               for index, weight in zip(indices, weights, strict=True))
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite PPO post-update KL probe")
    return result / mass if config.ppo_normalization == "rollout" else result
