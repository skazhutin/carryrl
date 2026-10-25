"""Matched train-only reaction rollouts with a fixed horizontal speed change."""

import json
import math
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache

import numpy as np

from carryrl.envs.atomic import (
    AtomicEnv,
    _identity,
    atomic_level_banks,
    atomic_semantic_level_id,
    verify_visible_outcome,
)
from carryrl.envs.atomic_suite import SCORING_VERSION
from carryrl.envs.runtime_identity import atomic_runtime_level_id
from carryrl.training.imitation import (
    QUALITY_SOURCE,
    REACTION_DYNAMICS_SOURCE,
    SOURCE,
    DemonstrationDataset,
    _collected_quality,
    _episode_runtime_level,
    demonstration_episode_digest,
    scripted_teacher,
)
from carryrl.training.runtime_diversity import (
    RuntimeDiversityPairs,
    checked_combined_frame_bytes,
)

ALGORITHM = "reaction_signed_horizontal_speed_2_guarded_v2"
HORIZONTAL_SPEED = 2.0
KNOWN_LEVEL_BANK_SEEDS = (0, 20261008, 20261015)


def _normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _changed_runtime(origin):
    return _normalized({**origin, "ball_dx": math.copysign(HORIZONTAL_SPEED, origin["ball_dx"])})


def _protected_bank_seeds(*seeds):
    return tuple(sorted(set((*KNOWN_LEVEL_BANK_SEEDS, *seeds))))


@lru_cache(maxsize=128)
def _derived_heldout_signatures(bank_seeds):
    """Exclude aliases created by collapsing held-out velocity magnitudes to 2.0."""
    return frozenset(atomic_runtime_level_id("reaction", _changed_runtime(level))
        for seed in bank_seeds for split in ("validation", "test")
        for level in atomic_level_banks("reaction", seed)[split])


def _validate_runtime_values(runtime, origin):
    if not isinstance(runtime, dict) or set(runtime) != {"ball_x", "ball_dx", "paddle_x"}:
        raise ValueError("reaction dynamics runtime requires only ball_x, ball_dx and paddle_x")
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in runtime.values()):
        raise ValueError("reaction dynamics runtime values must be finite numbers, not booleans")
    if runtime != _normalized(runtime) or runtime != _changed_runtime(origin):
        raise ValueError("reaction dynamics must change only signed horizontal speed to 2.0")


def validate_reaction_runtime(episode, origin):
    """Admit exactly the signed-speed override of the declared fixed TRAIN origin."""
    if (episode.task != "reaction" or type(episode.difficulty) not in (int, float)
            or episode.difficulty != 1.0 or episode.split != "train"
            or type(episode.level_index) is not int):
        raise ValueError("reaction dynamics requires a full-difficulty reaction TRAIN origin")
    bank = atomic_level_banks("reaction", episode.level_bank_seed)["train"]
    if (not 0 <= episode.level_index < len(bank) or origin != bank[episode.level_index]
            or episode.level_id != _identity(origin)
            or episode.origin_level_id != episode.level_id):
        raise ValueError("reaction dynamics must preserve its fixed TRAIN origin")
    runtime = episode.runtime_level_override
    _validate_runtime_values(runtime, origin)
    if episode.runtime_level_id != _identity(runtime):
        raise ValueError("reaction dynamics runtime hash differs from its override")
    protected_banks = _protected_bank_seeds(episode.level_bank_seed)
    if atomic_runtime_level_id("reaction", runtime) in _derived_heldout_signatures(protected_banks):
        raise ValueError("reaction dynamics overlaps a speed-transformed held-out runtime")
    return runtime


class ReactionDynamicsEnv(AtomicEnv):
    """Local speed-only override; reset preserves every other reaction cache."""

    def __init__(self, runtime, *, level_bank_seed, max_steps=80):
        super().__init__("reaction", split="train", difficulty=1.0, image_size=96,
                         level_bank_seed=level_bank_seed, max_steps=max_steps)
        self._dynamics_runtime = deepcopy(runtime)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        _validate_runtime_values(self._dynamics_runtime, self._level)
        self._level = deepcopy(self._dynamics_runtime)
        self._effective_level_id = _identity(self._level)
        self._semantic_level_id = atomic_semantic_level_id("reaction", self._level)
        self._ball_dx = self._level["ball_dx"]
        self._frame = self._draw()
        return self._frame.copy(), self._info(verify_visible_outcome(self._frame))


def _rollout(runtime, origin, max_steps):
    fixed = atomic_level_banks("reaction", origin.level_bank_seed)["train"][origin.level_index]
    if runtime == fixed:
        env = AtomicEnv("reaction", split="train", difficulty=1.0, image_size=96,
                        level_bank_seed=origin.level_bank_seed, max_steps=max_steps)
    else:
        env = ReactionDynamicsEnv(runtime, level_bank_seed=origin.level_bank_seed,
                                  max_steps=max_steps)
    teacher = scripted_teacher("reaction")
    frames, actions = [], []
    try:
        frame, _ = env.reset(seed=origin.seed, options={"level_index": origin.level_index})
        for _ in range(max_steps):
            rgb = np.frombuffer(frame.tobytes(), dtype=np.uint8).reshape(frame.shape)
            frames.append(rgb)
            action = teacher.act(rgb)
            actions.append(action)
            frame, _, terminated, truncated, info = env.step(action)
            teacher.observe_action(action)
            if terminated or truncated:
                break
        outcome = verify_visible_outcome(frame)
        if not terminated or truncated or not outcome["success"]:
            raise ValueError("reaction dynamics teacher must terminate in visible success")
        quality = _collected_quality(info, task="reaction", actions=actions,
                                     outcome=outcome, pure_teacher=True)
        return frames, tuple(actions), frame, outcome, quality
    finally:
        env.close()


def _present(frame, presentation):
    if presentation is None:
        return frame
    from carryrl.envs.presentation import PresentationProfile, present_frame

    return present_frame(frame, PresentationProfile(**presentation))


def validate_reaction_teacher_trace(dataset, episode, runtime):
    fixed = atomic_level_banks("reaction", episode.level_bank_seed)["train"][episode.level_index]
    if runtime != validate_reaction_runtime(episode, fixed):
        raise ValueError("reaction dynamics replay runtime differs from its provenance")
    frames, actions, final, outcome, quality = _rollout(runtime, episode, max(80, len(episode)))
    if actions != episode.actions or episode.executed_actions != actions:
        raise ValueError("reaction dynamics labels and history must equal actual "
                         "RGB teacher actions")
    if len(frames) != len(episode) or any(not np.array_equal(dataset.frames[index],
            _present(frame, episode.presentation))
            for index, frame in zip(episode.frame_indices, frames, strict=True)):
        raise ValueError("reaction dynamics trace RGB differs from actual runtime replay")
    if not np.array_equal(dataset.frames[episode.final_frame_index],
                          _present(final, episode.presentation)):
        raise ValueError("reaction dynamics final RGB differs from actual runtime replay")
    if (episode.outcome != outcome or episode.quality != quality
            or episode.scoring_version != SCORING_VERSION
            or episode.quality_source != QUALITY_SOURCE
            or episode.terminated is not True or episode.truncated is not False):
        raise ValueError("reaction dynamics quality or boundary differs from runtime replay")
    baseline = _rollout(fixed, episode, max(80, len(episode)))
    if actions == baseline[1] or len(actions) != len(baseline[1]):
        raise ValueError("reaction dynamics must change teacher actions with equal whole lengths")


def _costs(quality):
    return {key: quality[key] for key in
            ("active_commands", "input_units", "neutral_commands", "command_units")}


def collect_reaction_dynamics_pairs(dataset, *, max_pairs=8, seed=8176, max_steps=80,
        max_frame_bytes=256 * 1024**2, config=None):
    """Collect delta arms with equal whole lengths and genuinely new runtime mechanics."""
    if type(max_pairs) is not int or not 1 <= max_pairs <= 64:
        raise ValueError("reaction dynamics max_pairs must be in [1,64]")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("reaction dynamics seed must be uint32")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("reaction dynamics max_steps must be positive")
    if type(max_frame_bytes) is not int or max_frame_bytes < 1:
        raise ValueError("reaction dynamics max_frame_bytes must be positive")
    budget = min(dataset.max_frame_bytes, max_frame_bytes)
    if dataset.frame_bytes > budget:
        raise MemoryError("reaction dynamics base already exceeds the combined RGB byte budget")
    if config is not None and ("reaction" not in config.tasks or config.env_image_size != 96):
        raise ValueError("reaction dynamics requires reaction and canonical 96 RGB")
    parents, seen, protected = [], set(), set()
    for row in dataset.episodes:
        if row.task != "reaction":
            continue
        runtime = _episode_runtime_level(row)
        protected.add(atomic_runtime_level_id("reaction", runtime, row.difficulty))
        key = row.level_bank_seed, row.level_id
        if (row.source != SOURCE or type(row.difficulty) not in (int, float)
                or row.difficulty != 1.0 or row.presentation is not None or key in seen):
            continue
        dataset._validate_episode(row)
        if any(dataset.frames[index].shape != (96, 96, 3)
               for index in (*row.frame_indices, row.final_frame_index)):
            raise ValueError("reaction dynamics requires native canonical 96 RGB parents")
        parents.append(row)
        seen.add(key)
    protected_banks = _protected_bank_seeds(*(row.level_bank_seed for row in dataset.episodes
        if row.task == "reaction"), *((config.level_bank_seed,) if config is not None else ()))
    derived_heldout = _derived_heldout_signatures(protected_banks)
    rng = np.random.default_rng(seed)
    repeat, varied = (DemonstrationDataset(max_frame_bytes=max_frame_bytes) for _ in range(2))
    pairs, used = [], set()
    for index in rng.permutation(len(parents)):
        row = parents[int(index)]
        origin = atomic_level_banks("reaction", row.level_bank_seed)["train"][row.level_index]
        runtime = _changed_runtime(origin)
        signature = atomic_runtime_level_id("reaction", runtime)
        if signature in protected or signature in used or signature in derived_heldout:
            continue
        frames, actions, final, outcome, quality = _rollout(origin, row, max_steps)
        if (actions != row.actions or outcome != row.outcome
                or (row.executed_actions is not None and row.executed_actions != actions)
                or (row.quality is not None and row.quality != quality)
                or (row.terminated is not None and row.terminated is not True)
                or (row.truncated is not None and row.truncated is not False)
                or not np.array_equal(dataset.frames[row.final_frame_index], final)
                or len(frames) != len(row)
                or any(not np.array_equal(dataset.frames[i], frame)
                       for i, frame in zip(row.frame_indices, frames, strict=True))):
            raise ValueError("reaction dynamics parent must be an authentic "
                             "canonical teacher trace")
        new_frames, new_actions, new_final, new_outcome, new_quality = _rollout(
            runtime, row, max_steps)
        if new_actions == actions or len(new_actions) != len(actions):
            continue
        new_row = replace(row,
            frame_indices=tuple(varied.intern_frame(frame) for frame in new_frames),
            final_frame_index=varied.intern_frame(new_final), actions=new_actions,
            executed_actions=new_actions, source=REACTION_DYNAMICS_SOURCE,
            terminated=True, truncated=False, origin_level_id=row.level_id,
            runtime_level_id=_identity(runtime), runtime_level_override=runtime,
            outcome=new_outcome, quality=new_quality, quality_source=QUALITY_SOURCE,
            scoring_version=SCORING_VERSION)
        varied.add_episode(new_row)
        repeat.add_episode(replace(row,
            frame_indices=tuple(repeat.intern_frame(dataset.frames[i]) for i in row.frame_indices),
            final_frame_index=repeat.intern_frame(dataset.frames[row.final_frame_index])))
        used.add(signature)
        pairs.append({"parent_sha256": demonstration_episode_digest(dataset, row),
            "origin_level_id": row.level_id, "level_bank_seed": row.level_bank_seed,
            "level_index": row.level_index, "runtime_level_id": new_row.runtime_level_id,
            "runtime_level_override": runtime, "runtime_signature": signature,
            "decisions": len(row), "repeat_costs": _costs(quality),
            "varied_costs": _costs(new_quality)})
        if len(pairs) == max_pairs:
            break
    if len(pairs) != max_pairs:
        raise ValueError(f"reaction dynamics requested {max_pairs} pairs, found {len(pairs)}")
    arms = {"repeat": repeat, "varied": varied}
    combined_bytes = checked_combined_frame_bytes(dataset, arms, max_frame_bytes, config=config)
    return RuntimeDiversityPairs(repeat, varied, {
        "algorithm": ALGORITHM, "seed": seed, "horizontal_speed": HORIZONTAL_SPEED,
        "requested_pairs": max_pairs, "pairs_collected": len(pairs),
        "eligible_parents": len(parents), "pairs": pairs,
        "delta_episodes_per_arm": len(pairs),
        "delta_decisions_per_arm": sum(map(len, varied.episodes)),
        "delta_costs_per_arm": {mode: {key: sum(pair[f"{mode}_costs"][key] for pair in pairs)
            for key in _costs(varied.episodes[0].quality)} for mode in arms},
        "same_episode_replay_domain": "canonical_unstyled_96",
        "unique_new_runtime_configs": len(used), "base_observed_runtime_configs": len(protected),
        "derived_heldout_protected_banks": list(protected_banks),
        "derived_heldout_runtime_configs": len(derived_heldout),
        "derived_heldout_guard_passed": used.isdisjoint(derived_heldout),
        "combined_frame_bytes": combined_bytes, "combined_frame_budget_bytes": budget,
        "ordered_lengths": {mode: [len(row) for row in arm.episodes]
                            for mode, arm in arms.items()},
        "equal_ordered_lengths": [len(row) for row in repeat.episodes]
            == [len(row) for row in varied.episodes],
        "same_base_required": True, "base_episodes": len(dataset.episodes),
        "base_decisions": sum(map(len, dataset.episodes)), "test_gameplay_used": False,
    })
