"""Complete-episode PPO: frozen vision cache, replayed trainable context, learned visual reward."""

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn

from carryrl.agent.actions import (
    CODEC_VERSION,
    KEY_VOCAB,
    MOUSE_BINS,
    MOUSE_BUTTONS,
    WHEEL_BINS,
    PhysicalAction,
)
from carryrl.agent.model import GeneralistAgent
from carryrl.agent.vision import DINO_ID, DINO_REVISION
from carryrl.envs.atomic_suite import SCORING_VERSION, SUITE_VERSION
from carryrl.training.run import append_json, atomic_json, provenance
from carryrl.training.sequence_config import SequenceRunConfig


@dataclass
class Episode:
    patches: torch.Tensor
    frames: np.ndarray
    actions: list[PhysicalAction]
    old_log_prob: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    terminated: list[bool]
    truncated: list[bool]
    bootstrap: float
    task: str
    seed: int
    outcome: dict
    reward_uncertain_steps: int = 0
    final_frame: np.ndarray | None = None
    provenance: dict | None = None
    instruction: str | None = None

    def __len__(self) -> int:
        return len(self.actions)


class FrozenFrameCache:
    """Bounded exact RGB -> immutable vision patches, before every learned layer."""

    def __init__(self, agent: nn.Module, capacity: int, *, max_bytes: int = 256 * 1024**2):
        if capacity < 1 or max_bytes < 1:
            raise ValueError("frozen frame cache limits must be positive")
        if any(parameter.requires_grad for parameter in agent.vision.parameters()):
            raise ValueError("feature caching requires a frozen vision backbone")
        self.agent, self.capacity, self.max_bytes = agent, capacity, max_bytes
        self.vision_hash = tensor_hash(agent.vision)
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = self.bytes = 0

    def assert_unchanged(self) -> None:
        if (any(parameter.requires_grad for parameter in self.agent.vision.parameters())
                or tensor_hash(self.agent.vision) != self.vision_hash):
            raise RuntimeError("cached frozen vision backbone changed")

    def encode(self, frame: np.ndarray) -> torch.Tensor:
        key = (frame.shape, frame.dtype.str, hashlib.sha256(frame.tobytes()).digest())
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        patch = self.agent.encode_frames(torch.as_tensor(frame)[None])[0]
        stored = patch.detach().cpu().clone()
        size = stored.numel() * stored.element_size()
        if size <= self.max_bytes:
            while self.entries and (len(self.entries) >= self.capacity
                                    or self.bytes + size > self.max_bytes):
                _, removed = self.entries.popitem(last=False)
                self.bytes -= removed.numel() * removed.element_size()
                self.evictions += 1
            self.entries[key] = stored
            self.bytes += size
        return patch

    def statistics(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "frames": len(self.entries), "bytes": self.bytes,
                "capacity_frames": self.capacity, "capacity_bytes": self.max_bytes}


def _encode_frame(agent: nn.Module, frame: np.ndarray,
                  feature_cache: FrozenFrameCache | None) -> torch.Tensor:
    return (agent.encode_frames(torch.as_tensor(frame)[None])[0] if feature_cache is None
            else feature_cache.encode(frame))


def gae(rewards: torch.Tensor, values: torch.Tensor, terminated: list[bool],
        truncated: list[bool], bootstrap: float, gamma: float, lam: float) -> tuple:
    """Time-limit bootstraps final old history; either boundary stops GAE recursion."""
    if not (len(rewards) == len(values) == len(terminated) == len(truncated)) or not len(rewards):
        raise ValueError("GAE requires nonempty matching transitions")
    if any(terminated[:-1]) or any(truncated[:-1]):
        raise ValueError("GAE accepts one complete episode, not concatenated reset histories")
    advantages = torch.zeros_like(values)
    running = torch.zeros((), dtype=values.dtype, device=values.device)
    for index in reversed(range(len(rewards))):
        next_value = values[index + 1] if index + 1 < len(values) else bootstrap
        delta = rewards[index] + gamma * (not terminated[index]) * next_value - values[index]
        continuation = not (terminated[index] or truncated[index])
        running = delta + gamma * lam * continuation * running
        advantages[index] = running
    return advantages, advantages + values


def tensor_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        cpu = tensor.detach().cpu().contiguous()
        digest.update(str(cpu.dtype).encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _env_factory(task: str, **kwargs):
    from carryrl.envs.atomic import AtomicEnv
    return AtomicEnv(task, **kwargs)


def _reward_factory(image_size: int):
    from carryrl.rewards.visual import LearnedVisualReward
    return LearnedVisualReward(image_size=image_size)


def _outcome(frame: np.ndarray) -> dict:
    from carryrl.envs.atomic import verify_visible_outcome
    return verify_visible_outcome(frame)


def _prediction(reward: nn.Module, frame: np.ndarray, threshold: float) -> tuple[dict, bool]:
    result = dict(reward.predict(frame))
    required = ("score", "success_probability", "failure_probability")
    if any(key not in result or not np.isfinite(result[key]) for key in required):
        raise FloatingPointError("invalid learned visual reward prediction")
    if any(not 0 <= result[key] <= 1 for key in required[1:]):
        raise ValueError("reward event probabilities outside [0,1]")
    score_confidence = result.get("score_confidence")
    if "score_confidence" in result and (
        type(score_confidence) not in (int, float) or not 0 <= score_confidence <= 1
    ):
        raise ValueError("reward score confidence must be a finite probability in [0,1]")
    result["_score_trusted"] = (score_confidence is None or score_confidence >= threshold)
    uncertain = (not result.get("known", True)
                 or not result["_score_trusted"]
                 or any(max(result[key], 1 - result[key]) < threshold for key in required[1:]))
    return result, uncertain


def _transition(previous: dict, current: dict) -> float:
    if any(not prediction.get("known", True)
           or not prediction.get("_score_trusted", True) for prediction in (previous, current)):
        return 0.0
    return float(current["score"] - previous["score"]
                 + current["success_probability"] - previous["success_probability"]
                 - current["failure_probability"] + previous["failure_probability"] - 0.01)


def _episode_instruction(config: SequenceRunConfig, task: str, seed: int) -> str | None:
    """Select configured prose by public episode seed, independently of environment state."""
    variants = config.episode_instructions.get(task)
    if not variants and config.agent.text_encoder != "none":
        raise ValueError("enabled text_encoder requires a configured episode instruction")
    if not variants:
        return None
    index = int.from_bytes(hashlib.sha256(str(seed).encode("ascii")).digest(), "big")
    return variants[index % len(variants)]


def _reset_episode(agent: nn.Module, instruction: str | None) -> None:
    if instruction is None:
        agent.reset_episode()
    else:
        agent.reset_episode(instruction=instruction)


def _language_metadata(config: SequenceRunConfig) -> dict:
    if getattr(config.agent, "text_encoder", "none") == "none":
        return {}
    from carryrl.agent.language import TEXT_ID, TEXT_REVISION

    return {"language_weight_id": TEXT_ID, "language_weight_revision": TEXT_REVISION}


def _language_resource_hashes(directory: Path) -> dict[str, str]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("checkpoint packaged language resources are missing or invalid")
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint packaged language resources must not contain symlinks")
        if path.is_file():
            hashes[path.relative_to(directory).as_posix()] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    if not hashes:
        raise ValueError("checkpoint packaged language resources are empty")
    return hashes


def collect_episode(agent: nn.Module, reward: nn.Module, config: SequenceRunConfig, *, task: str,
                    seed: int, env_factory: Callable = _env_factory,
                    outcome_fn: Callable = _outcome,
                    on_decision: Callable | None = None, difficulty: float = 1.0,
                    feature_cache: FrozenFrameCache | None = None) -> Episode:
    instruction = _episode_instruction(config, task, seed)
    env = env_factory(task, split="train", **_environment_options(config, difficulty=difficulty))
    frames, patches, actions, log_probs, values, rewards, terms, truncs = ([] for _ in range(8))
    uncertain_count = 0
    agent.eval()
    try:
        _reset_episode(agent, instruction)
        frame, info = env.reset(seed=seed)
        previous, uncertain = _prediction(reward, frame, config.reward_confidence_threshold)
        uncertain_count += int(uncertain)
        with torch.no_grad():
            patch = _encode_frame(agent, frame, feature_cache)
            for _ in range(config.max_episode_steps):
                decision = agent.act_from_patches(patch, deterministic=False)
                if not np.isfinite(decision.log_prob) or not np.isfinite(decision.value):
                    raise FloatingPointError("nonfinite collected joint likelihood or value")
                frames.append(frame.copy())
                patches.append(patch.detach().cpu())
                actions.append(decision.action)
                log_probs.append(decision.log_prob)
                values.append(decision.value)
                if on_decision is not None:
                    on_decision()
                next_frame, _api_reward, terminated, truncated, _ = env.step(decision.action)
                current, uncertain = _prediction(reward, next_frame,
                                                 config.reward_confidence_threshold)
                uncertain_count += int(uncertain)
                rewards.append(_transition(previous, current))
                terms.append(bool(terminated))
                truncs.append(bool(truncated))
                next_patch = _encode_frame(agent, next_frame, feature_cache)
                if terminated or truncated:
                    bootstrap = 0.0 if terminated else agent.peek_value_from_patches(next_patch)
                    break
                frame, patch, previous = next_frame, next_patch, current
            else:
                raise RuntimeError("episode did not terminate/truncate at its declared limit")
        result = Episode(torch.stack(patches), np.stack(frames), actions,
                         torch.tensor(log_probs), torch.tensor(values), torch.tensor(rewards),
                         terms, truncs, float(bootstrap), task, seed, outcome_fn(next_frame),
                         uncertain_count, next_frame.copy(),
                         {"level_id": info.get("level_id"),
                          "presentation": info.get("presentation"), "difficulty": difficulty},
                         instruction)
        if not torch.isfinite(result.rewards).all() or not np.isfinite(result.bootstrap):
            raise FloatingPointError("nonfinite reward or final-history bootstrap")
        return result
    finally:
        env.close()
        agent.reset_episode()


def _episode_minibatches(episodes: list[Episode], size: int, balanced: bool):
    """Shuffle complete episodes, never splitting/resetting their replay history."""
    if size == 0:
        return [list(range(len(episodes)))]
    if balanced:
        by_task = {}
        for index, episode in enumerate(episodes):
            by_task.setdefault(episode.task, []).append(index)
        for indices in by_task.values():
            random.shuffle(indices)
        ordered = []
        while any(by_task.values()):
            tasks = [task for task, indices in by_task.items() if indices]
            random.shuffle(tasks)
            ordered.extend(by_task[task].pop() for task in tasks)
    else:
        ordered = list(range(len(episodes)))
        random.shuffle(ordered)
    return [ordered[start:start + size] for start in range(0, len(ordered), size)]


def _ppo_targets(episodes: list[Episode], config: SequenceRunConfig) -> tuple:
    if not episodes or any(not len(episode) for episode in episodes):
        raise ValueError("PPO requires complete nonempty episodes")
    total = sum(len(episode) for episode in episodes)
    task_totals = {}
    for episode in episodes:
        task_totals[episode.task] = task_totals.get(episode.task, 0) + len(episode)
    advantages, returns = [], []
    for episode in episodes:
        advantage, target = gae(episode.rewards, episode.old_values, episode.terminated,
                               episode.truncated, episode.bootstrap, config.gamma,
                               config.gae_lambda)
        advantages.append(advantage)
        returns.append(target)
    if config.ppo_task_balance:
        pairs = list(zip(episodes, advantages, strict=True))
        mean = sum(advantage.sum() / task_totals[episode.task]
                   for episode, advantage in pairs) / len(task_totals)
        variance = sum((advantage - mean).square().sum() / task_totals[episode.task]
                       for episode, advantage in pairs) / len(task_totals)
        std = variance.sqrt().clamp_min(1e-8)
    else:
        combined = torch.cat(advantages)
        mean, std = combined.mean(), combined.std(unbiased=False).clamp_min(1e-8)
    advantages = [(advantage - mean) / std for advantage in advantages]
    return total, task_totals, advantages, returns


def update_ppo(agent: nn.Module, optimizer: torch.optim.Optimizer, episodes: list[Episode],
               config: SequenceRunConfig, *, backend=None) -> dict:
    if config.training_backend == "mlx":
        from carryrl.training.mlx import update_ppo_mlx
        return update_ppo_mlx(agent, optimizer, episodes, config, backend=backend)
    if backend is not None:
        raise ValueError("a Metal backend requires training_backend=mlx")
    total, task_totals, advantages, returns = _ppo_targets(episodes, config)
    metrics, optimizer_steps = [], 0
    task_weight_sums = dict.fromkeys(task_totals, 0.0)
    task_replays = dict.fromkeys(task_totals, 0)
    stopped = False
    agent.train()
    try:
        for epoch in range(config.n_epochs):
            sums = {name: 0.0 for name in ("policy_loss", "value_loss", "entropy", "approx_kl",
                                         "clip_fraction", "gradient_norm")}
            observed_transitions = epoch_steps = observed_batches = 0
            for indices in _episode_minibatches(episodes, config.ppo_minibatch_episodes,
                                                config.ppo_task_balance):
                optimizer.zero_grad(set_to_none=True)
                counts = {}
                for index in indices:
                    episode = episodes[index]
                    counts[episode.task] = counts.get(episode.task, 0) + len(episode)
                batch_total = sum(counts.values())
                batch = {name: 0.0 for name in sums if name != "gradient_norm"}
                for index in indices:
                    episode, advantage, target = episodes[index], advantages[index], returns[index]
                    instruction_options = ({"instruction": episode.instruction}
                                           if episode.instruction is not None else {})
                    evaluated = agent.evaluate_episode(episode.patches, episode.actions,
                                                       **instruction_options)
                    log_prob, value, entropy = (evaluated[name]
                                               for name in ("log_prob", "value", "entropy"))
                    if any(tensor.shape != (len(episode),) or not torch.isfinite(tensor).all()
                           for tensor in (log_prob, value, entropy)):
                        raise FloatingPointError("invalid complete-episode policy replay")
                    old = episode.old_log_prob.to(log_prob.device)
                    adv, target = advantage.to(log_prob.device), target.to(log_prob.device)
                    log_ratio = log_prob - old
                    ratio = log_ratio.exp()
                    if not torch.isfinite(ratio).all():
                        raise FloatingPointError("nonfinite per-decision PPO ratio")
                    policy = -torch.minimum(ratio * adv,
                                            ratio.clamp(1 - config.clip_range,
                                                        1 + config.clip_range) * adv)
                    value_loss = 0.5 * (value - target).square()
                    denominator = (len(counts) * counts[episode.task]
                                   if config.ppo_task_balance else batch_total)
                    loss = (policy + config.value_coef * value_loss
                            - config.entropy_coef * entropy).sum() / denominator
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite PPO loss")
                    loss.backward()
                    for name, values in (("policy_loss", policy), ("value_loss", value_loss),
                                         ("entropy", entropy),
                                         ("approx_kl", ratio.detach() - 1 - log_ratio.detach()),
                                         ("clip_fraction", (ratio.detach() - 1).abs()
                                          > config.clip_range)):
                        batch[name] += float(values.detach().sum()) / denominator
                observed_batches += 1
                observed_transitions += batch_total
                for name in batch:
                    sums[name] += batch[name] * batch_total
                # No further optimizer mutation after a replay already violates
                # the explicit trust-region diagnostic against collection weights.
                if config.target_kl > 0 and batch["approx_kl"] > config.target_kl:
                    optimizer.zero_grad(set_to_none=True)
                    stopped = True
                    break
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in agent.parameters() if parameter.requires_grad],
                    config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                optimizer_steps += 1
                epoch_steps += 1
                for task, count in counts.items():
                    task_weight_sums[task] += (1 / len(counts) if config.ppo_task_balance
                                              else count / batch_total)
                    task_replays[task] += count
                sums["gradient_norm"] += float(gradient_norm) * batch_total
                if any(not torch.isfinite(parameter).all() for parameter in agent.parameters()):
                    raise FloatingPointError("nonfinite parameters after PPO update")
            metrics.append({"epoch": epoch,
                            **{name: value / observed_transitions for name, value in sums.items()},
                            "optimizer_steps": epoch_steps, "minibatches": observed_batches,
                            "replayed_transitions": observed_transitions,
                            "kl_early_stop": stopped})
            if stopped:
                break
    finally:
        agent.eval()
    return {"valid_transitions": total, "episodes": len(episodes), "epochs": metrics,
            "optimizer_steps": optimizer_steps, "kl_early_stop": stopped,
            "task_transitions": task_totals,
            "optimizer_task_loss_weights": {
                task: weight / optimizer_steps for task, weight in task_weight_sums.items()
            } if optimizer_steps else None,
            "optimizer_task_loss_weights_scope": "mean objective coefficients over applied "
            "optimizer steps; excludes rejected KL minibatches; not gradient attribution",
            "optimizer_task_transition_replays": task_replays,
            "loss_weighting": "equal task mean per minibatch" if config.ppo_task_balance
                              else "valid transitions per minibatch"}


def _random_action(rng: np.random.Generator) -> PhysicalAction:
    # General physical distribution, independent of task identity and game legality.
    count = int(rng.integers(0, 5))
    keys = tuple(KEY_VOCAB[int(index)]
                 for index in rng.choice(len(KEY_VOCAB), count, replace=False))
    buttons = tuple(button for button in MOUSE_BUTTONS if rng.random() < 0.1)
    return PhysicalAction(keys, buttons, int(rng.choice(MOUSE_BINS)), int(rng.choice(MOUSE_BINS)),
                          int(rng.choice(WHEEL_BINS)))


def _environment_options(config: SequenceRunConfig, *, difficulty: float = 1.0) -> dict:
    options = {"image_size": config.env_image_size,
               "max_steps": config.max_episode_steps}
    if config.visual_config is not None:
        options["visual_config"] = config.visual_config
    if difficulty != 1.0:
        options["difficulty"] = difficulty
    if config.level_bank_seed:
        options["level_bank_seed"] = config.level_bank_seed
    return options


def sequence_evaluation_protocol(config: SequenceRunConfig, *,
                                 split: str = "validation") -> dict:
    """Describe the exact frozen evaluation cohort without running episodes."""
    if split not in {"validation", "test"}:
        raise ValueError("frozen evaluation requires a held-out split")
    evaluation_difficulty = (config.evaluation_difficulty
                             if config.evaluation_difficulty is not None else
                             config.curriculum_difficulty
                             if config.curriculum_difficulty is not None else 1.0)
    policies = ["frozen_deterministic"]
    if config.eval_stochastic:
        policies.append("frozen_stochastic")
    policies.append("generic_physical_random")
    from carryrl.envs.atomic import atomic_level_banks

    counts = {task: max(config.eval_episodes_per_task,
                       len(atomic_level_banks(task, config.level_bank_seed)[split]))
              if config.eval_all_levels else config.eval_episodes_per_task
              for task in config.tasks}
    protocol = {"split": split, "tasks": list(config.tasks), "counts": counts,
                "suite_version": SUITE_VERSION, "scoring_version": SCORING_VERSION,
                "policies": policies, "eval_seed": config.eval_seed,
                "eval_all_levels": config.eval_all_levels,
                "max_episode_steps": config.max_episode_steps,
                "difficulty": evaluation_difficulty, "env_image_size": config.env_image_size,
                "visual_config": config.to_dict()["visual_config"],
                "episode_instructions": config.episode_instructions}
    if config.level_bank_seed:
        protocol["level_bank_seed"] = config.level_bank_seed
    return protocol


def sequence_evaluation_summaries(records, *, tasks, policies) -> dict:
    """Aggregate frozen episode outcomes with the evaluator's metric semantics."""
    def summarize(rows):
        unknown = sum(not row["outcome"].get("known", False) for row in rows)
        successes = sum(row["outcome"].get("known", False)
                        and row["outcome"].get("success") is True for row in rows)
        failures = sum(row["outcome"].get("known", False)
                       and row["outcome"].get("failure") is True for row in rows)
        visible_scores = [row["outcome"]["score"] for row in rows
                          if row["outcome"].get("known", False)
                          and row["outcome"].get("score") is not None]
        perfect = sum(row["outcome"].get("known", False)
                      and row["outcome"].get("success") is True
                      and row["outcome"].get("score") == 5 for row in rows)
        return {"episodes_started": len(rows), "unknown_outcomes": unknown,
                "success_rate_lower": successes / len(rows),
                "success_rate_upper": (successes + unknown) / len(rows),
                "visible_final_score_count": len(visible_scores),
                "perfect_score_successes": perfect,
                "perfect_score_rate": perfect / len(rows),
                "mean_visible_final_score": float(np.mean(visible_scores))
                if visible_scores else None,
                "outcomes": {"success": successes, "failure": failures,
                             "unfinished": len(rows) - successes - failures - unknown,
                             "unknown": unknown},
                "time_limit_episodes": sum(row["truncated"] for row in rows),
                "distinct_level_ids": len({row["level_id"] for row in rows
                                           if row["level_id"] is not None}),
                "distinct_semantic_level_ids": len({row["semantic_level_id"] for row in rows
                                                    if row["semantic_level_id"] is not None})}

    summaries = {}
    for policy in policies:
        rows = [row for row in records if row["policy"] == policy]
        summaries[policy] = summarize(rows)
        summaries[policy]["tasks"] = {
            task: summarize([row for row in rows if row["task"] == task]) for task in tasks
        }
    return summaries


def evaluate_sequence(agent: nn.Module, reward: nn.Module, config: SequenceRunConfig, *,
                      env_factory: Callable = _env_factory, outcome_fn: Callable = _outcome,
                      split: str = "validation", feature_cache: FrozenFrameCache | None = None,
                      on_progress: Callable[[dict], None] | None = None,
                      on_playback: Callable[[dict, list, list], None] | None = None,
                      ) -> dict:
    """Frozen evaluation; optional callbacks observe real progress and bounded RGB replay.

    ``on_progress`` receives started/step/completed events. ``on_playback`` receives
    the completed deterministic episode at index zero per task, its initial and
    post-action frames, and physical actions. Capture requires at most 80 actions.
    Callbacks run inside the frozen-state/RNG restoration guard; errors propagate.
    """
    if split not in {"validation", "test"}:
        raise ValueError("frozen evaluation requires a held-out split")
    if on_playback is not None and config.max_episode_steps > 80:
        raise ValueError("frozen playback capture requires at most 80 episode steps")
    evaluation_started = time.monotonic()
    created_utc = datetime.now(UTC).isoformat()
    agent_hash, reward_hash = tensor_hash(agent), tensor_hash(reward)
    evaluation_difficulty = (config.evaluation_difficulty
                             if config.evaluation_difficulty is not None else
                             config.curriculum_difficulty
                             if config.curriculum_difficulty is not None else 1.0)
    if feature_cache is None and config.vision_cache_frames:
        feature_cache = FrozenFrameCache(agent, config.vision_cache_frames)
    if feature_cache is not None:
        if feature_cache.agent is not agent:
            raise ValueError("frozen frame cache belongs to another agent")
        feature_cache.assert_unchanged()
    rng_state, python_state, numpy_state = (torch.random.get_rng_state(), random.getstate(),
                                           np.random.get_state())
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    prior_mode = agent.training
    records = []
    legacy_random_rng = np.random.default_rng(config.eval_seed)
    protocol = sequence_evaluation_protocol(config, split=split)
    policies = protocol["policies"]
    total_episodes = sum(protocol["counts"].values()) * len(policies)
    protocol_sha256 = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    try:
        agent.eval()
        for policy in policies:
            for task_index, task in enumerate(config.tasks):
                if config.eval_all_levels:
                    from carryrl.envs.atomic import atomic_level_banks
                    level_count = len(atomic_level_banks(task, config.level_bank_seed)[split])
                else:
                    level_count = None
                count = protocol["counts"][task]
                for index in range(count):
                    seed = 1_000_000 + index
                    instruction = _episode_instruction(config, task, seed)
                    action_seed = (config.eval_seed + task_index * 100_000 + index) % 2**32
                    rng = (np.random.default_rng(action_seed)
                           if config.eval_all_levels or config.eval_stochastic
                           else legacy_random_rng)
                    if policy == "frozen_stochastic":
                        torch.manual_seed(action_seed)
                        if mps_state is not None:
                            torch.mps.manual_seed(action_seed)
                    env = env_factory(task, split=split,
                                      **_environment_options(config,
                                                             difficulty=evaluation_difficulty))
                    try:
                        _reset_episode(agent, instruction)
                        options = {"level_index": index % level_count} if level_count else None
                        frame, info = env.reset(seed=seed, **({"options": options}
                                                            if options is not None else {}))
                        initial_frame_hash = hashlib.sha256(frame.tobytes()).hexdigest()
                        capture = on_playback is not None and policy == "frozen_deterministic" \
                            and index == 0
                        frames, actions = ([frame.copy()], []) if capture else (None, None)
                        event = {"policy": policy, "task": task, "episode_index": index,
                                 "episodes_completed": len(records),
                                 "episodes_total": total_episodes}
                        if on_progress is not None:
                            on_progress({**event, "event": "episode_started", "step": 0})
                        for step in range(config.max_episode_steps):
                            if policy == "generic_physical_random":
                                action = _random_action(rng)
                            elif feature_cache is None:
                                action = agent.act(frame, deterministic=
                                                   policy == "frozen_deterministic").action
                            else:
                                action = agent.act_from_patches(
                                    _encode_frame(agent, frame, feature_cache),
                                    deterministic=policy == "frozen_deterministic").action
                            frame, _, terminated, truncated, _ = env.step(action)
                            if capture:
                                actions.append(action.to_dict())
                                frames.append(frame.copy())
                            if on_progress is not None:
                                on_progress({**event, "event": "step", "step": step + 1})
                            if terminated or truncated:
                                break
                        else:
                            raise RuntimeError("evaluation exceeded its episode budget")
                        outcome = outcome_fn(frame)
                        records.append({"task": task, "seed": seed, "steps": step + 1,
                                        "instruction": instruction,
                                        "terminated": bool(terminated),
                                        "truncated": bool(truncated),
                                        "policy": policy, "action_seed": action_seed,
                                        "episode_index": index,
                                        "difficulty": evaluation_difficulty,
                                        "level_index": options["level_index"] if options else None,
                                        "level_id": info.get("level_id"),
                                        "semantic_level_id": info.get("semantic_level_id"),
                                        "runtime_fingerprint": info.get("runtime_fingerprint"),
                                        "level_bank_seed": config.level_bank_seed,
                                        "presentation": info.get("presentation"),
                                        "initial_frame_sha256": initial_frame_hash,
                                        "outcome": outcome})
                        if on_progress is not None:
                            on_progress({**event, "event": "episode_completed", "step": step + 1,
                                         "episodes_completed": len(records)})
                        if capture:
                            on_playback(dict(records[-1]), frames, actions)
                    finally:
                        env.close()
    finally:
        agent.reset_episode()
        agent.train(prior_mode)
        torch.random.set_rng_state(rng_state)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
    if tensor_hash(agent) != agent_hash or tensor_hash(reward) != reward_hash:
        raise RuntimeError("frozen evaluation mutated policy or reward tensors")

    summaries = sequence_evaluation_summaries(
        records, tasks=config.tasks, policies=policies)
    return {"split": split, "records": records, "summaries": summaries,
            **({"created_utc": created_utc,
                "elapsed_seconds": time.monotonic() - evaluation_started}
               if on_progress is not None or on_playback is not None else {}),
            "protocol_sha256": protocol_sha256,
            "suite_version": SUITE_VERSION, "scoring_version": SCORING_VERSION,
            "evaluation_difficulty": evaluation_difficulty,
            "level_bank_seed": config.level_bank_seed,
            "policy_and_reward_frozen": True, "evaluation_seed": config.eval_seed,
            "vision_cache": feature_cache.statistics() if feature_cache is not None else None,
            "level_selection": "explicit covering level indices" if config.eval_all_levels
                               else "legacy seeded sampling",
            "random_distribution": "0..4 uniform subset keys; independent button p=.1; "
            "uniform physical mouse/wheel bins; no task mask"}


def _reward_validation(reward: nn.Module, frames: np.ndarray, labels: dict,
                       threshold: float) -> dict:
    predictions = [_prediction(reward, frame, threshold) for frame in frames]
    scores = np.array([item[0]["score"] for item in predictions])
    result = {"frames": len(frames), "score_mae": float(np.abs(scores - labels["score"]).mean()),
              "uncertain_fraction": float(np.mean([item[1] for item in predictions]))}
    combinations = np.stack([labels[name] for name in ("score", "success", "failure")], axis=1)
    result["score_mae_by_combination"] = []
    for combination in np.unique(combinations, axis=0):
        mask = (combinations == combination).all(axis=1)
        result["score_mae_by_combination"].append({
            "score": float(combination[0]), "success": bool(combination[1]),
            "failure": bool(combination[2]), "frames": int(mask.sum()),
            "score_mae": float(np.abs(scores[mask] - labels["score"][mask]).mean()),
        })
    for event in ("success", "failure"):
        probabilities = np.array([item[0][event + "_probability"] for item in predictions])
        predicted, target = probabilities >= 0.5, np.asarray(labels[event]).astype(bool)
        true_positive = int((predicted & target).sum())
        result[event] = {"accuracy": float((predicted == target).mean()),
                         "brier_score": float(((probabilities - target) ** 2).mean()),
                         "false_positive_rate": float(predicted[~target].mean())
                         if (~target).any() else None,
                         "positive_labels": int(target.sum()),
                         "predicted_positives": int(predicted.sum()),
                         "precision": true_positive / int(predicted.sum())
                         if predicted.sum() else None,
                         "recall": true_positive / int(target.sum()) if target.sum() else None}
    return result


def prepare_reward(config: SequenceRunConfig, output: Path,
                   reward_factory: Callable = _reward_factory) -> nn.Module:
    if config.reward_model_mode == "screen_v2":
        from carryrl.rewards.pretrain import (
            load_screen_reward_checkpoint,
            pretrain_screen_reward,
        )
        if config.reward_checkpoint is not None:
            reward, report = load_screen_reward_checkpoint(config.reward_checkpoint,
                                                          device=config.device)
            if not set(config.tasks).issubset(report.get("tasks", ())):
                raise ValueError("visual reward checkpoint does not cover the configured tasks")
            source = {"source_checkpoint": str(Path(config.reward_checkpoint).resolve())}
        else:
            reward, report = pretrain_screen_reward(
                output / "reward_pretraining", tasks=config.tasks,
                episodes_per_task=config.reward_bootstrap_episodes_per_task,
                validation_episodes_per_task=config.reward_validation_episodes_per_task,
                epochs=config.reward_epochs, seed=config.seed, device=config.device,
                image_size=config.env_image_size, max_steps=config.max_episode_steps,
                presentation_config=config.visual_config)
            if not report["gate"]["passed"]:
                raise RuntimeError("visual reward validation gate failed before policy training")
            source = {"source_checkpoint": str((output / "reward_pretraining"
                                                / "reward_checkpoint.pt").resolve())}
        atomic_json(output / "reward_bootstrap.json", {
            **report, **source, "state_sha256": tensor_hash(reward),
            "policy_demonstrations": False,
            "supervision": "pixel-verifier labels; supervised visual reward pretraining"})
        return reward.eval().requires_grad_(False)
    if config.reward_checkpoint is not None:
        raise ValueError("external reward checkpoints require screen_v2 reward mode")
    from carryrl.envs.atomic import collect_bootstrap_dataset
    from carryrl.rewards.visual import train_visual_reward
    frames, labels, labels_provenance = collect_bootstrap_dataset(
        episodes_per_task=config.reward_bootstrap_episodes_per_task, seed=config.seed,
        image_size=config.env_image_size, split="train", max_steps=config.max_episode_steps,
        tasks=config.tasks)
    reward = reward_factory(config.env_image_size).to(config.device)
    metrics = train_visual_reward(reward, frames, labels, epochs=config.reward_epochs,
                                  batch_size=64, seed=config.seed, learning_rate=0.001)
    validation_frames, validation_labels, validation_provenance = collect_bootstrap_dataset(
        episodes_per_task=config.reward_validation_episodes_per_task, seed=config.seed + 50000,
        image_size=config.env_image_size, split="validation", max_steps=config.max_episode_steps,
        tasks=config.tasks)
    reward.eval().requires_grad_(False)
    atomic_json(output / "reward_bootstrap.json", {
        "supervision": "pixel-verifier HUD labels from scripted physical train trajectories; "
        "not autonomous reward discovery", "fit_split": "train", "fit": metrics,
        "fit_labels_provenance": labels_provenance,
        "validation_labels_provenance": validation_provenance,
        "heldout": _reward_validation(reward, validation_frames, validation_labels,
                                       config.reward_confidence_threshold),
        "state_sha256": tensor_hash(reward), "policy_demonstrations": False,
    })
    return reward


def save_sequence_checkpoint(agent: nn.Module, optimizer: torch.optim.Optimizer, reward: nn.Module,
                             config: SequenceRunConfig, output: Path, *, rollout: int,
                             decisions: int, revision: int) -> Path:
    parent = output / "checkpoints"
    parent.mkdir(exist_ok=True)
    destination = parent / f"rollout_{rollout:04d}"
    if destination.exists():
        raise FileExistsError(f"sequence checkpoint already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=".sequence-", dir=parent))
    try:
        archive = temporary / "state.pt"
        reward_version = getattr(reward, "model_version", None)
        reward_metadata = ({"reward_model_version": reward_version}
                           if reward_version is not None else {})
        language_metadata = _language_metadata(config)
        language_resources = {}
        if language_metadata:
            from carryrl.agent.language import MultilingualInstructionEncoder

            if isinstance(getattr(agent, "language", None), MultilingualInstructionEncoder):
                agent.language.save_resources(temporary / "language")
                language_resources["language_resources_sha256"] = _language_resource_hashes(
                    temporary / "language")
        torch.save({
            "format": "carryrl.sequence.v1", "codec_version": CODEC_VERSION,
            "suite_version": SUITE_VERSION,
            "training_backend": config.training_backend,
            "config": config.to_dict(), "agent_state": agent.state_dict(),
            "optimizer_state": optimizer.state_dict(), "reward_state": reward.state_dict(),
            "rollout": rollout, "decisions": decisions, "weights_revision": revision,
            "initialization": "pinned pretrained DINO" if config.agent.pretrained else "scratch",
            "vision_weight_id": DINO_ID if config.agent.pretrained else None,
            "vision_weight_revision": DINO_REVISION if config.agent.pretrained else None,
            "resume_rule": "new episode; original seed restart; no exact trajectory replay",
            **reward_metadata,
            **language_metadata,
        }, archive)
        atomic_json(temporary / "manifest.json", {
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "agent_sha256": tensor_hash(agent), "reward_sha256": tensor_hash(reward),
            "rollout": rollout, "decisions": decisions, "weights_revision": revision,
            "codec_version": CODEC_VERSION,
            "training_backend": config.training_backend,
            **reward_metadata,
            **language_metadata,
            **language_resources,
        })
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    atomic_json(output / "latest.json", {"checkpoint": str(destination.resolve())})
    return destination


def load_sequence_checkpoint(checkpoint: Path, config: SequenceRunConfig, *,
                             agent_factory: Callable = GeneralistAgent,
                             reward_factory: Callable = _reward_factory,
                             evaluation_only: bool = False) -> tuple:
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    archive = checkpoint / "state.pt"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("checkpoint archive SHA256 mismatch")
    data = torch.load(archive, map_location=config.device, weights_only=True)
    if data.get("format") != "carryrl.sequence.v1" or data.get("codec_version") != CODEC_VERSION:
        raise ValueError("unsupported sequence checkpoint format or codec")
    stored = SequenceRunConfig.from_dict(data["config"])
    if (data.get("training_backend", "torch") != stored.training_backend
            or manifest.get("training_backend", "torch") != stored.training_backend):
        raise ValueError("checkpoint training backend metadata mismatch")
    if not evaluation_only and replace(stored, total_rollouts=config.total_rollouts,
                                       max_wall_seconds=config.max_wall_seconds,
                                       reset_optimizer_on_warm_start=
                                       config.reset_optimizer_on_warm_start) != config:
        raise ValueError("resume must preserve settings except target rollouts and wall budget")
    if evaluation_only and stored.agent != config.agent:
        raise ValueError("evaluation architecture must match its checkpoint")
    for name, expected in _language_metadata(stored).items():
        if data.get(name) != expected or manifest.get(name) != expected:
            raise ValueError("checkpoint language model identity or revision mismatch")
    resource_hashes = manifest.get("language_resources_sha256")
    if resource_hashes is not None and _language_resource_hashes(
        checkpoint / "language"
    ) != resource_hashes:
        raise ValueError("checkpoint language resource SHA256 mismatch")
    # Build an identical graph offline; loading a checkpoint never downloads weights again.
    offline_config = replace(stored.agent, pretrained=False)
    if agent_factory is GeneralistAgent and _language_metadata(stored):
        from carryrl.agent.language import MultilingualInstructionEncoder

        if resource_hashes is None:
            raise ValueError("text checkpoint is missing packaged language resource hashes")
        language = MultilingualInstructionEncoder.from_resources(
            checkpoint / "language", max_tokens=stored.agent.instruction_max_tokens)
        agent = agent_factory(offline_config, language=language).to(config.device)
    else:
        agent = agent_factory(offline_config).to(config.device)
    agent.decoder.initialize_motor_prior(stored.motor_initialization)
    agent.load_state_dict(data["agent_state"], strict=True)
    agent.config = stored.agent
    agent.reset_episode()
    if stored.reward_model_mode == "screen_v2" and reward_factory is _reward_factory:
        from carryrl.rewards.visual import ScreenVisualReward
        reward = ScreenVisualReward(stored.env_image_size).to(config.device)
    else:
        reward = reward_factory(stored.env_image_size).to(config.device)
    saved_reward_version = data.get("reward_model_version")
    if saved_reward_version != manifest.get("reward_model_version"):
        raise ValueError("checkpoint reward model versions disagree")
    if (saved_reward_version is not None
            and saved_reward_version != getattr(reward, "model_version", None)):
        raise ValueError(f"unsupported checkpoint reward model version: {saved_reward_version}")
    try:
        reward.load_state_dict(data["reward_state"], strict=True)
    except RuntimeError as error:
        raise ValueError("checkpoint reward state is incompatible with the current reward model") \
            from error
    reward.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([parameter for parameter in agent.parameters()
                                   if parameter.requires_grad], lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    optimizer.load_state_dict(data["optimizer_state"])
    if (tensor_hash(agent) != manifest["agent_sha256"]
            or tensor_hash(reward) != manifest["reward_sha256"]):
        raise ValueError("loaded checkpoint tensor hash mismatch")
    if any(data[name] != manifest[name] for name in ("rollout", "decisions", "weights_revision")):
        raise ValueError("checkpoint progress counters disagree")
    if not evaluation_only and config.total_rollouts <= data["rollout"]:
        raise ValueError("resume target must exceed checkpoint rollout count")
    return agent, optimizer, reward, data


def train_sequence(config: SequenceRunConfig, output: Path, resume: Path | None = None, *,
                   warm_start: Path | None = None,
                   agent_factory: Callable = GeneralistAgent,
                   env_factory: Callable = _env_factory, reward: nn.Module | None = None,
                   reward_factory: Callable = _reward_factory,
                   outcome_fn: Callable = _outcome) -> dict:
    if resume is not None and warm_start is not None:
        raise ValueError("resume and warm_start are mutually exclusive")
    if config.reset_optimizer_on_warm_start and warm_start is None:
        raise ValueError("reset_optimizer_on_warm_start requires warm_start and forbids resume")
    if warm_start is not None and reward is not None:
        raise ValueError("warm_start preserves checkpoint reward; injected reward forbidden")
    if warm_start is not None:
        warm_start = Path(warm_start)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    attempted_decisions = 0
    phase_started = last_status_write = started
    episodes_per_rollout = sum(
        config.episodes_per_task * config.task_episode_multipliers.get(task, 1)
        for task in config.tasks)
    minibatch_episodes = config.ppo_minibatch_episodes or episodes_per_rollout
    updates_per_rollout_upper_bound = config.n_epochs * math.ceil(
        episodes_per_rollout / minibatch_episodes)
    phase_plan = [
        {"id": "preparation", "phases": ["loading"]},
        {"id": "baseline", "phases": ["initial_evaluation"]},
        {"id": "training", "phases": ["collecting", "optimizing", "evaluation"]},
        {"id": "final_evaluation", "phases": ["final_evaluation"]},
        {"id": "saving", "phases": ["saving"]},
        {"id": "completed", "phases": ["finished"]},
    ]
    progress = {"status": "running", "rollouts": 0, "decisions": 0,
                "rollouts_total": config.total_rollouts, "current_rollout": 0, "current_cycle": 0,
                "optimizer_steps_this_process": 0,
                "optimizer_steps_upper_bound": config.total_rollouts
                * updates_per_rollout_upper_bound,
                "optimizer_steps_upper_bound_scope": "local process; KL may reduce updates",
                "total_decisions_upper_bound": config.total_rollouts * episodes_per_rollout
                * config.max_episode_steps,
                "phase_plan": phase_plan, "phase": "loading", "phase_index": 1,
                "phase_started_at": datetime.now(UTC).isoformat(),
                "phase_units_done": 0, "phase_units_total": None,
                "phase_unit_kind": "operations"}

    def write_progress(*, force=False, **values):
        nonlocal last_status_write
        progress.update(values)
        now = time.monotonic()
        if not force and now - last_status_write < 2:
            return
        progress.update(updated_at=datetime.now(UTC).isoformat(), wall_seconds=now - started,
                        elapsed_seconds=now - started, phase_elapsed_seconds=now - phase_started,
                        attempted_train_decisions_this_process=attempted_decisions)
        atomic_json(output / "status.json", progress)
        last_status_write = now

    def phase(name, *, total=None, done=0, unit="operations", **values):
        nonlocal phase_started
        phase_started = time.monotonic()
        write_progress(force=True, phase=name,
                       phase_index=next(index for index, group in enumerate(phase_plan, 1)
                                        if name in group["phases"]),
                       phase_started_at=datetime.now(UTC).isoformat(), phase_units_done=done,
                       phase_units_total=total, phase_unit_kind=unit, **values)

    def count_decision():
        nonlocal attempted_decisions
        attempted_decisions += 1
        write_progress()
    write_progress(force=True)
    metadata = {
        "kind": "sequence_training",
        "created_utc": datetime.now(UTC).isoformat(), "config": config.to_dict(),
        "provenance": provenance(), "codec_version": CODEC_VERSION,
        "suite_version": SUITE_VERSION,
        "vision": {"model_id": DINO_ID, "revision": DINO_REVISION, "frozen": True},
        "transformers_version": version("transformers"),
        "policy_inputs": "complete RGB history and previous own physical actions; no task ID",
        "vision_cache": "frozen backbone patches only; all trainable context replayed each epoch",
        "exact_rgb_cache_frames": config.vision_cache_frames,
        "reward": "separately supervised visual HUD evaluator; continuous prediction differences",
        "bootstrap_is_supervised": True, "human_demonstrations": False,
        "optimizer": "AdamW", "weight_decay": config.weight_decay,
        "training_backend": config.training_backend,
        "trainable_device": "Metal GPU" if config.training_backend == "mlx" else config.device,
        "frozen_encoders_and_rollout_device": config.device,
        "optimizer_state_format": "canonical bias-corrected Torch AdamW",
        "sampling_balance": {"minimum_episodes_per_task": config.episodes_per_task,
                             "multipliers": config.task_episode_multipliers,
                             "loss_weighting": "equal task mean per minibatch"
                             if config.ppo_task_balance else "valid transitions per minibatch"},
        "ppo_minibatch_episodes": config.ppo_minibatch_episodes,
        "target_kl": config.target_kl, "motor_initialization": config.motor_initialization,
        "curriculum": {"initial_difficulty": config.curriculum_difficulty,
                       "warmup_rollouts": config.curriculum_warmup_rollouts,
                       "schedule": "explicit linear warmup; not a mastery detector"},
        "evaluation_frequency_rollouts": config.eval_every_rollouts,
        "checkpoint_frequency_rollouts": config.checkpoint_every_rollouts,
        "minimum_free_disk_mib": config.minimum_free_disk_mib,
        "phase_plan": phase_plan,
        "checkpoint_save_rule": "initial state, scheduled completed rollouts, and actual final "
        "or graceful wall-stop state before final evaluation; errors retain last saved snapshot",
        "resume": str(resume.resolve()) if resume else None,
        "warm_start": str(warm_start.resolve()) if warm_start else None,
        "warm_start_rng_restored": False, "warm_start_exact_episode": False,
        "resume_rng_restored": False, "resume_exact_episode": False,
        "wall_stop_rule": "complete reward bootstrap/PPO cycle boundary, then evaluate latest "
        "saved policy if needed; bounded-cycle plus final-evaluation overshoot",
    }
    if _language_metadata(config):
        metadata.update(_language_metadata(config))
        metadata["policy_inputs"] = ("complete RGB history, previous own physical actions, "
                                     "and explicit episode instruction; no task ID")
        metadata["instruction_selection"] = (
            "SHA256 of ASCII decimal episode seed, big-endian integer modulo prose variants")
    atomic_json(output / "config.json", config.to_dict())
    atomic_json(output / "metadata.json", metadata)
    try:
        if config.training_backend == "mlx":
            from carryrl.training.mlx import require_mlx
            require_mlx()
        if config.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; explicit device never silently falls back")
        torch.set_num_threads(config.torch_threads)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        if warm_start is not None:
            agent, optimizer, reward, saved = load_sequence_checkpoint(
                warm_start, config, agent_factory=agent_factory, reward_factory=reward_factory,
                evaluation_only=True)
            source_config = SequenceRunConfig.from_dict(saved["config"])
            if source_config.motor_initialization != config.motor_initialization:
                raise ValueError("warm_start must preserve its checkpoint motor prior")
            if source_config.reward_model_mode != config.reward_model_mode:
                raise ValueError("warm_start must preserve its checkpoint reward model mode")
            if (config.reward_checkpoint is not None
                    and config.reward_checkpoint != source_config.reward_checkpoint):
                raise ValueError("warm_start cannot substitute an external reward checkpoint")
            previous_groups = [{"learning_rate": group["lr"],
                                "weight_decay": group["weight_decay"]}
                               for group in optimizer.param_groups]
            source_optimizer_state_entries = len(optimizer.state)
            if config.reset_optimizer_on_warm_start:
                optimizer.state.clear()
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate
                group["weight_decay"] = config.weight_decay
            parent_manifest = json.loads((warm_start / "manifest.json").read_text())
            metadata["warm_start_parent"] = {
                "checkpoint": str(warm_start.resolve()),
                "checkpoint_sha256": parent_manifest["sha256"],
                "agent_sha256": parent_manifest["agent_sha256"],
                "reward_sha256": parent_manifest["reward_sha256"],
                "reward_model_version": saved.get("reward_model_version"),
                "source_config": source_config.to_dict(),
                "source_counters": {name: saved[name] for name in
                                    ("rollout", "decisions", "weights_revision")},
                "optimizer_state_preserved": not config.reset_optimizer_on_warm_start,
                "optimizer_reset_reason": "explicit warm-start stage optimizer reset"
                if config.reset_optimizer_on_warm_start else None,
                "source_optimizer": {
                    "type": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
                    "state_entries": source_optimizer_state_entries,
                    "checkpoint_sha256": parent_manifest["sha256"],
                    "parameter_groups": [len(group["params"]) for group in optimizer.param_groups],
                },
                "previous_optimizer_hyperparameters": previous_groups,
                "new_optimizer_hyperparameters": {
                    "learning_rate": config.learning_rate, "weight_decay": config.weight_decay},
                "counter_scope": "new local rollout stage; cumulative decisions/weights_revision",
            }
            atomic_json(output / "metadata.json", metadata)
            rollout = 0
            decisions, revision = saved["decisions"], saved["weights_revision"]
        elif resume is not None:
            agent, optimizer, reward, saved = load_sequence_checkpoint(
                resume, config, agent_factory=agent_factory, reward_factory=reward_factory)
            rollout, decisions, revision = (saved[name] for name in
                                            ("rollout", "decisions", "weights_revision"))
        else:
            agent = agent_factory(config.agent).to(config.device)
            agent.decoder.initialize_motor_prior(config.motor_initialization)
            optimizer = torch.optim.AdamW([parameter for parameter in agent.parameters()
                                           if parameter.requires_grad], lr=config.learning_rate,
                                          weight_decay=config.weight_decay)
            reward = prepare_reward(config, output, reward_factory) if reward is None else reward
            reward = reward.to(config.device).eval().requires_grad_(False)
            rollout = decisions = revision = 0
        initial_rollout, initial_decisions, initial_revision = rollout, decisions, revision
        backend = None
        if config.training_backend == "mlx":
            from carryrl.training.mlx import MLXPPOBackend
            backend = MLXPPOBackend(agent, optimizer)
        progress.update({"rollouts": rollout, "decisions": decisions,
                         "weights_revision": revision, "initial_decisions": initial_decisions,
                         "current_rollout": rollout, "current_cycle": rollout,
                         "optimizer_steps_upper_bound": (config.total_rollouts - rollout)
                         * updates_per_rollout_upper_bound})
        write_progress(force=True)
        feature_cache = (FrozenFrameCache(agent, config.vision_cache_frames)
                         if config.vision_cache_frames else None)
        reward_hash = tensor_hash(reward)
        phase("saving", current_rollout=rollout, current_cycle=rollout)
        checkpoint = save_sequence_checkpoint(agent, optimizer, reward, config, output,
                                              rollout=rollout, decisions=decisions,
                                              revision=revision)
        saved_rollout, saved_decisions, saved_revision = rollout, decisions, revision

        def checkpoint_progress():
            return {"latest_checkpoint": str(checkpoint.resolve()),
                    "latest_checkpoint_rollout": saved_rollout,
                    "latest_checkpoint_decisions": saved_decisions,
                    "latest_checkpoint_weights_revision": saved_revision,
                    "current_policy_saved": saved_rollout == rollout and saved_revision == revision}

        def save_current_checkpoint():
            nonlocal checkpoint, saved_rollout, saved_decisions, saved_revision
            if saved_rollout != rollout:
                phase("saving", current_rollout=rollout, current_cycle=rollout)
                checkpoint = save_sequence_checkpoint(agent, optimizer, reward, config, output,
                                                      rollout=rollout, decisions=decisions,
                                                      revision=revision)
                saved_rollout, saved_decisions, saved_revision = rollout, decisions, revision
            progress.update(checkpoint_progress())

        progress.update(checkpoint_progress())
        write_progress(force=True)
        status = "completed"
        evaluation = None
        last_evaluated_rollout = None
        optimizer_steps_this_process = 0

        def save_evaluation(evaluation_phase="evaluation"):
            phase(evaluation_phase, unit="episodes", current_rollout=rollout, current_cycle=rollout)
            evaluation_started = time.monotonic()

            def evaluation_progress(event):
                write_progress(phase_units_done=event["episodes_completed"],
                               phase_units_total=event["episodes_total"],
                               evaluation_task=event["task"], evaluation_policy=event["policy"],
                               evaluation_step=event["step"])

            result = evaluate_sequence(agent, reward, config, env_factory=env_factory,
                                       outcome_fn=outcome_fn, feature_cache=feature_cache,
                                       on_progress=evaluation_progress)
            write_progress(force=True)
            result.update({"rollout": rollout, "decisions": decisions,
                           "weights_revision": revision,
                           **checkpoint_progress(),
                           "agent_sha256": tensor_hash(agent),
                           "reward_sha256": tensor_hash(reward),
                           "wall_seconds": time.monotonic() - evaluation_started,
                           "elapsed_run_seconds": time.monotonic() - started,
                           "wall_budget_overshoot_seconds": max(
                               0.0, time.monotonic() - started - config.max_wall_seconds)})
            if rollout == initial_rollout:
                result["baseline_kind"] = ("stage_initial" if warm_start is not None
                                           else "resume_initial" if resume is not None
                                           else "untrained_initial")
                result["baseline_policy_previously_trained"] = revision > 0
            destination = output / "evaluations" / f"rollout_{rollout:04d}.json"
            if destination.exists():
                raise FileExistsError(f"evaluation record already exists: {destination}")
            atomic_json(destination, result)
            return result

        if config.evaluate_initial:
            evaluation = save_evaluation("initial_evaluation")
            last_evaluated_rollout = rollout
            atomic_json(output / "baseline.json", evaluation)

        while rollout < config.total_rollouts:
            if time.monotonic() - started >= config.max_wall_seconds:
                status = "resource_stopped"
                progress["stop_reason"] = "max_wall_seconds"
                break
            if config.minimum_free_disk_mib:
                free_bytes = shutil.disk_usage(output).free
                minimum_free_bytes = config.minimum_free_disk_mib * 1024**2
                write_progress(free_disk_bytes=free_bytes,
                               minimum_free_disk_bytes=minimum_free_bytes)
                if free_bytes < minimum_free_bytes:
                    status = "resource_stopped"
                    progress["stop_reason"] = "minimum_free_disk_mib"
                    break
            phase("collecting", total=episodes_per_rollout, unit="episodes",
                  current_rollout=rollout + 1, current_cycle=rollout + 1)
            frozen_reward = deepcopy(reward).eval().requires_grad_(False)
            cycle_hash = tensor_hash(frozen_reward)
            episodes = []
            cycle_started = time.monotonic()
            cache_before = feature_cache.statistics() if feature_cache is not None else None
            difficulty = (1.0 if config.curriculum_difficulty is None
                          else config.curriculum_difficulty)
            if config.curriculum_warmup_rollouts:
                fraction = min(1.0, rollout / config.curriculum_warmup_rollouts)
                difficulty += (1.0 - difficulty) * fraction
            for task_index, task in enumerate(config.tasks):
                count = (config.episodes_per_task
                         * config.task_episode_multipliers.get(task, 1))
                for episode_index in range(count):
                    seed = config.seed + rollout * 10000 + task_index * 1000 + episode_index
                    episode = collect_episode(agent, frozen_reward, config, task=task, seed=seed,
                                              env_factory=env_factory, outcome_fn=outcome_fn,
                                              on_decision=count_decision, difficulty=difficulty,
                                              feature_cache=feature_cache)
                    episodes.append(episode)
                    write_progress(phase_units_done=len(episodes))
                    records = output / "trajectories" / f"rollout_{rollout + 1:04d}"
                    records.mkdir(parents=True, exist_ok=True)
                    record = records / f"{task}_{episode_index}.npz"
                    np.savez_compressed(
                        record, frames=episode.frames, final_frame=episode.final_frame,
                        old_log_prob=episode.old_log_prob.numpy(),
                        old_values=episode.old_values.numpy(), rewards=episode.rewards.numpy(),
                        terminated=np.asarray(episode.terminated),
                        truncated=np.asarray(episode.truncated), bootstrap=episode.bootstrap,
                    )
                    atomic_json(record.with_suffix(".json"), {
                        "task": task, "seed": seed, "collection_weights_revision": revision,
                        "instruction": episode.instruction,
                        "provenance": episode.provenance,
                        "reward_sha256": cycle_hash, "codec_version": CODEC_VERSION,
                        "actions": [action.to_dict() for action in episode.actions],
                        "npz_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
                    })
            # An exception inside PPO may leave partially applied unsaved weights.
            progress["current_policy_saved"] = False
            phase("optimizing", unit="optimizer_updates",
                  done=optimizer_steps_this_process, current_rollout=rollout + 1,
                  current_cycle=rollout + 1)
            stats = update_ppo(agent, optimizer, episodes, config,
                               **({"backend": backend} if backend is not None else {}))
            if feature_cache is not None:
                feature_cache.assert_unchanged()
                stats["vision_cache"] = {
                    **feature_cache.statistics(),
                    "cycle_hits": feature_cache.hits - cache_before["hits"],
                    "cycle_misses": feature_cache.misses - cache_before["misses"]}
            if tensor_hash(frozen_reward) != cycle_hash or tensor_hash(reward) != reward_hash:
                raise RuntimeError("PPO cycle mutated its frozen reward snapshot")
            rollout += 1
            decisions += stats["valid_transitions"]
            revision += stats["optimizer_steps"]
            optimizer_steps_this_process += stats["optimizer_steps"]
            progress.update({"rollouts": rollout, "decisions": decisions,
                             "weights_revision": revision,
                             "optimizer_steps_this_process": optimizer_steps_this_process,
                             "phase_units_done": optimizer_steps_this_process,
                             **checkpoint_progress()})
            append_json(output / "training.jsonl", {
                "rollout": rollout, "decisions": decisions, "weights_revision": revision,
                "cycle_wall_seconds": time.monotonic() - cycle_started,
                "complete": True,
                "curriculum_difficulty": difficulty,
                "reward_snapshot_sha256": cycle_hash, **stats,
                "episode_records": [{"task": episode.task, "seed": episode.seed,
                                     "instruction": episode.instruction,
                                     "steps": len(episode), "outcome": episode.outcome,
                                     "provenance": episode.provenance,
                                     "reward_sum": float(episode.rewards.sum()),
                                     "reward_uncertain_steps": episode.reward_uncertain_steps,
                                     "actions": [action.to_dict() for action in episode.actions]}
                                    for episode in episodes],
            })
            # The journal contains all remaining episode diagnostics. Raw RGB and
            # frozen patches are no longer needed during checkpointing/evaluation.
            del episodes, episode
            if (rollout % config.checkpoint_every_rollouts == 0
                    or rollout == config.total_rollouts
                    or time.monotonic() - started >= config.max_wall_seconds):
                save_current_checkpoint()
            write_progress(force=True, status="running", rollouts=rollout, decisions=decisions,
                           weights_revision=revision, initial_decisions=initial_decisions,
                           current_rollout=rollout, current_cycle=rollout, **checkpoint_progress())
            if rollout % config.eval_every_rollouts == 0:
                evaluation = save_evaluation()
                last_evaluated_rollout = rollout
        save_current_checkpoint()
        # A graceful budget stop still reports the latest saved policy, even when
        # its rollout falls between scheduled evaluations. This bounded final
        # evaluation can extend the wall budget; the run retains its stop status.
        if last_evaluated_rollout != rollout:
            evaluation = save_evaluation("final_evaluation")
            last_evaluated_rollout = rollout
        # A periodic measurement may precede an off-cadence final save. Keep that
        # immutable record intact and refresh only the consolidated snapshot binding.
        final_binding = checkpoint_progress()
        if any(evaluation.get(key) != value for key, value in final_binding.items()):
            evaluation = {**evaluation, **final_binding, "checkpoint_binding_refreshed": True}
        atomic_json(output / "evaluation.json", evaluation)
        phase("finished", current_rollout=rollout, current_cycle=rollout)
        elapsed = time.monotonic() - started
        summary = {**progress, "status": status, "rollouts": rollout, "decisions": decisions,
                   "initial_rollout": initial_rollout, "initial_decisions": initial_decisions,
                   "attempted_train_decisions_this_process": attempted_decisions,
                   "weights_revision": revision, "wall_seconds": elapsed,
                   **checkpoint_progress(),
                   "optimizer_steps_this_process": optimizer_steps_this_process,
                   "initial_weights_revision": initial_revision,
                   "counter_scope": "cumulative decisions/weights_revision; local stage rollouts"
                   if warm_start is not None else "continuation counters" if resume
                   else "run counters",
                   "wall_budget_overshoot_seconds": max(0.0, elapsed - config.max_wall_seconds),
                   "reward_sha256": reward_hash,
                   "policy_parameters": sum(parameter.numel() for parameter in agent.parameters()),
                   "trainable_parameters": sum(parameter.numel() for parameter in agent.parameters()
                                                if parameter.requires_grad),
                   "evaluation_summaries": evaluation["summaries"] if evaluation else None,
                   "evaluation_rollout": last_evaluated_rollout,
                   "interpretation": "bounded engineering probe; no generalization or skill claim"}
        atomic_json(output / "summary.json", summary)
        metadata["checkpoint_state"] = {
            "current_rollout": rollout, "current_weights_revision": revision,
            **checkpoint_progress()}
        atomic_json(output / "metadata.json", metadata)
        atomic_json(output / "status.json", summary)
        return summary
    except BaseException as error:
        progress.update(updated_at=datetime.now(UTC).isoformat(),
                        phase_elapsed_seconds=time.monotonic() - phase_started)
        atomic_json(output / "status.json", {
            **progress,
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "attempted_train_decisions_this_process": attempted_decisions,
            "error_type": type(error).__name__, "error": str(error),
            "wall_seconds": time.monotonic() - started,
            "latest_checkpoint_retained": (output / "latest.json").exists(),
        })
        raise
