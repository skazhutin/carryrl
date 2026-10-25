"""Controlled train-only navigation wall geometry, with equal-decision replay arms."""

import itertools
import json
from copy import deepcopy
from dataclasses import dataclass, replace
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
from carryrl.envs.navigation import level_banks
from carryrl.rewards.navigation import RecognitionError, shortest_distance
from carryrl.training.imitation import (
    NAVIGATION_GEOMETRY_SOURCE,
    QUALITY_SOURCE,
    SOURCE,
    DemonstrationDataset,
    _collected_quality,
    _episode_runtime_level,
    demonstration_episode_digest,
    scripted_teacher,
)

ALGORITHM = "navigation_two_wall_equal_distance_v1"


def _normalized(value):
    return json.loads(json.dumps(value))


@lru_cache(maxsize=1)
def fresh_wall_layouts():
    """All connected two-wall maps outside every published fixed wall bank."""
    fixed = {walls for bank in level_banks().values() for walls in bank}
    layouts = []
    cells = tuple(itertools.product(range(5), repeat=2))
    for pair in itertools.combinations(cells, 2):
        walls = frozenset(pair)
        if walls in fixed:
            continue
        floor = [cell for cell in cells if cell not in walls]
        try:
            for cell in floor[1:]:
                shortest_distance(floor[0], cell, walls)
        except RecognitionError:
            continue
        layouts.append(tuple(sorted(walls)))
    return tuple(layouts)


def validate_navigation_runtime(episode, origin):
    runtime = episode.runtime_level_override
    if not isinstance(runtime, dict) or set(runtime) != {"walls", "start", "goal"}:
        raise ValueError("navigation geometry runtime requires only walls, start and goal")
    if runtime != _normalized(runtime):
        raise ValueError("navigation geometry runtime requires normalized JSON coordinates")
    if not isinstance(runtime["walls"], list):
        raise ValueError("navigation geometry walls must be a list")
    for cell in [runtime["start"], runtime["goal"], *runtime["walls"]]:
        if (not isinstance(cell, list) or len(cell) != 2
                or any(type(value) is not int or not 0 <= value < 5 for value in cell)):
            raise ValueError("navigation geometry coordinates must be grid integer pairs")
    walls = runtime["walls"]
    if (len(walls) != 2 or walls != sorted(walls)
            or tuple(map(tuple, walls)) not in fresh_wall_layouts()):
        raise ValueError("navigation geometry walls must be connected and outside fixed banks")
    if (runtime["start"] != list(origin["start"]) or runtime["goal"] != list(origin["goal"])
            or runtime["start"] in walls or runtime["goal"] in walls
            or episode.origin_level_id != episode.level_id
            or episode.runtime_level_id != _identity(runtime)):
        raise ValueError("navigation geometry must preserve its training origin start and goal")
    old_distance = shortest_distance(tuple(origin["start"]), tuple(origin["goal"]),
                                    frozenset(map(tuple, origin["walls"])))
    if shortest_distance(tuple(runtime["start"]), tuple(runtime["goal"]),
                         frozenset(map(tuple, walls))) != old_distance:
        raise ValueError("navigation geometry must preserve the shortest distance")
    return runtime


class NavigationGeometryEnv(AtomicEnv):
    """Private train geometry replay; update all navigation caches after reset."""

    def __init__(self, runtime, *, level_bank_seed, max_steps=80):
        super().__init__("navigation", split="train", difficulty=1.0, image_size=96,
                         level_bank_seed=level_bank_seed, max_steps=max_steps)
        self._geometry_runtime = deepcopy(runtime)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._level = deepcopy(self._geometry_runtime)
        self._effective_level_id = _identity(self._level)
        self._semantic_level_id = atomic_semantic_level_id("navigation", self._level)
        self._walls = frozenset(map(tuple, self._level["walls"]))
        self._position, self._goal = tuple(self._level["start"]), tuple(self._level["goal"])
        self._initial_distance = shortest_distance(self._position, self._goal, self._walls)
        self._frame = self._draw()
        return self._frame.copy(), self._info(verify_visible_outcome(self._frame))


def _rollout(runtime, origin, max_steps):
    env = NavigationGeometryEnv(runtime, level_bank_seed=origin.level_bank_seed,
                                max_steps=max_steps)
    teacher = scripted_teacher("navigation")
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
            raise ValueError("navigation geometry teacher must terminate in visible success")
        quality = _collected_quality(info, task="navigation", actions=actions,
                                     outcome=outcome, pure_teacher=True)
        return frames, tuple(actions), frame, outcome, quality
    finally:
        env.close()


def validate_navigation_teacher_trace(dataset, episode, runtime):
    frames, actions, final, outcome, quality = _rollout(runtime, episode, max(80, len(episode)))
    if actions != episode.actions:
        raise ValueError("navigation geometry labels must equal actual RGB teacher actions")
    if len(frames) != len(episode) or any(not np.array_equal(dataset.frames[index],
            _present(frame, episode.presentation))
            for index, frame in zip(episode.frame_indices, frames, strict=True)):
        raise ValueError("navigation geometry trace RGB differs from actual runtime replay")
    if not np.array_equal(dataset.frames[episode.final_frame_index],
                          _present(final, episode.presentation)):
        raise ValueError("navigation geometry final RGB differs from actual runtime replay")
    if (episode.outcome != outcome or episode.quality != quality
            or episode.terminated is not True or episode.truncated is not False):
        raise ValueError("navigation geometry quality or boundary differs from runtime replay")
    origin = atomic_level_banks("navigation", episode.level_bank_seed)["train"][episode.level_index]
    if actions == _rollout(origin, episode, max(80, len(episode)))[1]:
        raise ValueError("navigation geometry must change the RGB teacher action sequence")


def _present(frame, presentation):
    if presentation is None:
        return frame
    from carryrl.envs.presentation import PresentationProfile, present_frame

    return present_frame(frame, PresentationProfile(**presentation))


def parent_digest(dataset, episode):
    return demonstration_episode_digest(dataset, episode)


@dataclass(frozen=True)
class NavigationGeometryPairs:
    repeat: DemonstrationDataset
    new_walls: DemonstrationDataset
    manifest: dict

    def arm(self, mode):
        if mode not in ("repeat", "new_walls"):
            raise ValueError("navigation geometry mode must be repeat or new_walls")
        return getattr(self, mode)


def collect_navigation_geometry_pairs(dataset, *, max_pairs=8, seed=8176, max_steps=80,
        max_frame_bytes=256 * 1024**2, config=None):
    """Return delta arms only; never edit the base or silently shrink the requested count."""
    if type(max_pairs) is not int or not 1 <= max_pairs <= 64:
        raise ValueError("navigation geometry max_pairs must be in [1,64]")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("navigation geometry seed must be uint32")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("navigation geometry max_steps must be positive")
    if type(max_frame_bytes) is not int or max_frame_bytes < 1:
        raise ValueError("navigation geometry max_frame_bytes must be positive")
    combined_budget = min(dataset.max_frame_bytes, max_frame_bytes)
    if dataset.frame_bytes > combined_budget:
        raise MemoryError("navigation geometry base already exceeds the combined RGB byte budget")
    if config is not None and ("navigation" not in config.tasks or config.env_image_size != 96):
        raise ValueError("navigation geometry requires navigation and canonical 96 RGB")
    parents, seen = [], set()
    protected_walls = set()
    for row in dataset.episodes:
        if row.task == "navigation":
            runtime = _episode_runtime_level(row)
            protected_walls.add(tuple(sorted(map(tuple, runtime["walls"]))))
        key = row.level_bank_seed, row.level_id, row.instruction
        if (row.source != SOURCE or row.task != "navigation" or row.difficulty != 1.0
                or row.presentation is not None or key in seen):
            continue
        dataset._validate_episode(row)
        if any(dataset.frames[index].shape != (96, 96, 3)
               for index in (*row.frame_indices, row.final_frame_index)):
            raise ValueError("navigation geometry requires native canonical 96 RGB parents")
        parents.append(row)
        seen.add(key)
    rng = np.random.default_rng(seed)
    layouts = fresh_wall_layouts()
    repeat, fresh = (DemonstrationDataset(max_frame_bytes=max_frame_bytes) for _ in range(2))
    pairs, used = [], set()
    for index in rng.permutation(len(parents)):
        row = parents[int(index)]
        origin = atomic_level_banks("navigation", row.level_bank_seed)["train"][row.level_index]
        frames, actions, final, outcome, quality = _rollout(origin, row, max_steps)
        if (actions != row.actions
                or (row.executed_actions is not None and row.executed_actions != actions)
                or (row.quality is not None and row.quality != quality)
                or not np.array_equal(dataset.frames[row.final_frame_index], final)
                or len(frames) != len(row)
                or any(not np.array_equal(dataset.frames[i], f)
                       for i, f in zip(row.frame_indices, frames, strict=True))):
            raise ValueError("navigation geometry parent must be an authentic "
                             "canonical teacher trace")
        for candidate in rng.permutation(len(layouts)):
            walls = layouts[int(candidate)]
            if (walls in protected_walls or walls in used
                    or tuple(origin["start"]) in walls or tuple(origin["goal"]) in walls):
                continue
            runtime = _normalized({**origin, "walls": walls})
            if shortest_distance(tuple(origin["start"]), tuple(origin["goal"]),
                    frozenset(walls)) != quality["optimal_command_units"]:
                continue
            new_frames, new_actions, new_final, new_outcome, new_quality = _rollout(
                runtime, row, max_steps)
            if (new_actions == actions or len(new_actions) != len(actions)
                    or new_quality["command_units"] != quality["command_units"]
                    or new_quality["input_units"] != quality["input_units"]):
                continue
            new_row = replace(row, frame_indices=tuple(fresh.intern_frame(f) for f in new_frames),
                final_frame_index=fresh.intern_frame(new_final), actions=new_actions,
                executed_actions=new_actions, source=NAVIGATION_GEOMETRY_SOURCE,
                terminated=True, truncated=False, origin_level_id=row.level_id,
                runtime_level_id=_identity(runtime), runtime_level_override=runtime,
                outcome=new_outcome, quality=new_quality, quality_source=QUALITY_SOURCE,
                scoring_version=SCORING_VERSION)
            fresh.add_episode(new_row)
            repeat.add_episode(replace(row,
                frame_indices=tuple(repeat.intern_frame(dataset.frames[i])
                                    for i in row.frame_indices),
                final_frame_index=repeat.intern_frame(dataset.frames[row.final_frame_index])))
            used.add(walls)
            pairs.append({"parent_sha256": parent_digest(dataset, row),
                "origin_level_id": row.level_id, "level_bank_seed": row.level_bank_seed,
                "level_index": row.level_index, "runtime_level_id": new_row.runtime_level_id,
                "runtime_level_override": runtime, "decisions": len(row),
                "shortest_distance": quality["optimal_command_units"],
                "input_units": quality["input_units"],
                "command_units": quality["command_units"]})
            break
        if len(pairs) == max_pairs:
            break
    if len(pairs) != max_pairs:
        raise ValueError(f"navigation geometry requested {max_pairs} pairs, found {len(pairs)}")
    from carryrl.training.runtime_diversity import checked_combined_frame_bytes

    combined_bytes = checked_combined_frame_bytes(dataset,
        {"repeat": repeat, "new_walls": fresh}, max_frame_bytes, config=config)
    return NavigationGeometryPairs(repeat, fresh, {
        "algorithm": ALGORITHM, "seed": seed, "requested_pairs": max_pairs,
        "pairs_collected": len(pairs), "eligible_parents": len(parents), "pairs": pairs,
        "delta_episodes_per_arm": len(pairs),
        "delta_decisions_per_arm": sum(map(len, fresh.episodes)),
        "delta_input_units_per_arm": sum(row["input_units"] for row in pairs),
        "delta_command_units_per_arm": sum(row["command_units"] for row in pairs),
        "same_episode_replay_domain": "canonical_unstyled_96",
        "unique_new_wall_layouts": len(used),
        "base_observed_wall_layouts": len(protected_walls),
        "combined_frame_bytes": combined_bytes,
        "combined_frame_budget_bytes": combined_budget,
        "equal_ordered_lengths": [len(row) for row in fresh.episodes]
            == [len(row) for row in repeat.episodes],
        "same_base_required": True, "base_episodes": len(dataset.episodes),
        "base_decisions": sum(map(len, dataset.episodes)), "test_gameplay_used": False,
    })
