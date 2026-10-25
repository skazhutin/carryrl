"""Fresh navigation maps have authenticated replay and a matched decision budget."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import _identity, atomic_level_banks
from carryrl.envs.navigation import level_banks
from carryrl.envs.presentation import PresentationProfile, present_frame
from carryrl.rewards.navigation import shortest_distance
from carryrl.training.imitation import (
    NAVIGATION_GEOMETRY_SOURCE,
    DemonstrationDataset,
    DemonstrationEpisode,
    augment_demonstrations,
    collect_scripted_demonstrations,
    imitation_replay_summary,
    validate_dataset_runtime_holdout,
)
from carryrl.training.navigation_geometry import (
    NavigationGeometryEnv,
    _rollout,
    collect_navigation_geometry_pairs,
    fresh_wall_layouts,
)
from carryrl.training.sequence_config import SequenceRunConfig


@pytest.fixture(scope="module")
def base():
    return collect_scripted_demonstrations(16, tasks=("navigation",), difficulty=1,
                                           max_frame_bytes=32 * 1024**2)


@pytest.fixture(scope="module")
def pairs(base):
    return collect_navigation_geometry_pairs(base, config=SequenceRunConfig(tasks=("navigation",)))


def restore(dataset):
    copy = DemonstrationDataset(max_frame_bytes=dataset.max_frame_bytes)
    for frame in dataset.frames:
        copy.intern_frame(frame)
    for record in json.loads(json.dumps(dataset.provenance()))["episodes"]:
        values = {key: value for key, value in record.items()
                  if key in DemonstrationEpisode.__dataclass_fields__}
        values["frame_indices"] = tuple(values["frame_indices"])
        values["actions"] = tuple(PhysicalAction.from_dict(action) for action in values["actions"])
        if "executed_actions" in values:
            values["executed_actions"] = tuple(PhysicalAction.from_dict(action)
                                                for action in values["executed_actions"])
        copy.add_episode(DemonstrationEpisode(**values))
    return copy


def test_eight_genuine_deterministic_pairs_equal_budget_and_immutable_banks(base, pairs):
    banks = deepcopy(atomic_level_banks("navigation"))
    original = base.provenance()
    duplicate = collect_navigation_geometry_pairs(base)
    assert pairs.manifest == duplicate.manifest
    assert original == base.provenance()
    assert banks == atomic_level_banks("navigation")
    assert pairs.manifest["pairs_collected"] == 8
    assert pairs.manifest["equal_ordered_lengths"]
    assert pairs.manifest["test_gameplay_used"] is False
    assert pairs.repeat.provenance() == duplicate.repeat.provenance()
    assert pairs.new_walls.provenance() == duplicate.new_walls.provenance()
    assert len(pairs.repeat.episodes) == len(pairs.new_walls.episodes) == 8
    assert imitation_replay_summary(pairs.repeat, canonical_replay_weight=3) == (
        imitation_replay_summary(pairs.new_walls, canonical_replay_weight=3))
    fixed = {layout for bank in level_banks().values() for layout in bank}
    used = set()
    for old, new in zip(pairs.repeat.episodes, pairs.new_walls.episodes, strict=True):
        runtime = new.runtime_level_override
        walls = frozenset(map(tuple, runtime["walls"]))
        assert walls not in fixed | used
        used.add(walls)
        assert old.actions != new.actions
        assert old.executed_actions is None or old.executed_actions == old.actions
        assert new.executed_actions == new.actions
        assert len(old) == len(new)
        assert old.instruction == new.instruction
        assert new.quality["excess_command_units"] == 0
        assert new.quality["completion_efficiency"] == 1
        assert new.quality["input_units"] == old.quality["input_units"]
        assert new.quality["commands"] == new.quality["optimal_command_units"]
        assert new.terminated is True and new.truncated is False
        assert old.source == "scripted_teacher" and new.source == NAVIGATION_GEOMETRY_SOURCE
        assert "runtime_level_override" not in old.to_dict()
    for dataset in (pairs.repeat, pairs.new_walls):
        rebuilt = restore(dataset)
        assert rebuilt.provenance() == dataset.provenance()
        validate_dataset_runtime_holdout(rebuilt, level_bank_seed=20261008,
                                         evaluation_difficulty=0)


def test_runtime_reset_updates_all_navigation_fields(pairs):
    episode = pairs.new_walls.episodes[0]
    runtime = episode.runtime_level_override
    env = NavigationGeometryEnv(runtime, level_bank_seed=episode.level_bank_seed)
    frame, info = env.reset(seed=episode.seed, options={"level_index": episode.level_index})
    assert np.array_equal(frame, pairs.new_walls.frames[episode.frame_indices[0]])
    assert env._walls == frozenset(map(tuple, runtime["walls"]))
    assert env._position == tuple(runtime["start"])
    assert env._goal == tuple(runtime["goal"])
    assert env._initial_distance == episode.quality["optimal_command_units"]
    assert env._effective_level_id == _identity(runtime)
    env.close()


@pytest.mark.parametrize("mutation", [
    {"task": "memory"}, {"difficulty": 0}, {"native_size": 64},
    {"executed_actions": None}, {"terminated": False}, {"truncated": True},
    {"quality": None}, {"origin_level_id": None},
])
def test_new_source_rejects_wrong_contract(pairs, mutation):
    with pytest.raises(ValueError):
        replace(pairs.new_walls.episodes[0], **mutation)


@pytest.mark.parametrize("mutation", [
    {"walls": "bad"}, {"walls": [[True, 0], [0, 1]]}, {"extra": 1},
    {"start": [999, 0]}, {"walls": [[0, 0], [0, 0]]},
    {"walls": [list(cell) for cell in sorted(level_banks()["test"][0])]},
])
def test_malformed_or_heldout_geometry_rejected(pairs, mutation):
    row = pairs.new_walls.episodes[0]
    runtime = {**deepcopy(row.runtime_level_override), **mutation}
    with pytest.raises(ValueError):
        pairs.new_walls._validate_episode(replace(row, runtime_level_override=runtime,
                                                  runtime_level_id=_identity(runtime)))


def test_route_unchanged_geometry_is_rejected(base):
    row = base.episodes[0]
    origin = atomic_level_banks("navigation")["train"][row.level_index]
    for walls in fresh_wall_layouts():
        if tuple(origin["start"]) in walls or tuple(origin["goal"]) in walls:
            continue
        if shortest_distance(tuple(origin["start"]), tuple(origin["goal"]),
                             frozenset(walls)) != len(row):
            continue
        runtime = json.loads(json.dumps({**origin, "walls": walls}))
        frames, actions, final, outcome, quality = _rollout(runtime, row, 80)
        if actions != row.actions:
            continue
        result = DemonstrationDataset()
        new = replace(row, source=NAVIGATION_GEOMETRY_SOURCE, actions=actions,
            executed_actions=actions, frame_indices=tuple(result.intern_frame(f) for f in frames),
            final_frame_index=result.intern_frame(final), origin_level_id=row.level_id,
            runtime_level_id=_identity(runtime), runtime_level_override=runtime,
            terminated=True, truncated=False, outcome=outcome, quality=quality)
        with pytest.raises(ValueError, match="must change"):
            result.add_episode(new)
        break
    else:
        pytest.fail("fixture has no unchanged-route candidate")


def test_rgb_and_action_tampering_rejected(pairs):
    dataset = restore(pairs.new_walls)
    row = dataset.episodes[0]
    rgb = dataset.frames[row.frame_indices[0]].copy()
    rgb[0, 0] ^= 1
    index = dataset.intern_frame(rgb)
    with pytest.raises(ValueError, match="trace RGB"):
        dataset.add_episode(replace(row, frame_indices=(index, *row.frame_indices[1:])))
    changed = (PhysicalAction(), *row.actions[1:])
    with pytest.raises(ValueError):
        dataset.add_episode(replace(row, actions=changed, executed_actions=changed))
    quality = {**row.quality, "unnecessary_input_units": 1}
    with pytest.raises(ValueError):
        dataset.add_episode(replace(row, quality=quality))


def test_appearance_copies_authenticate_canonical_teacher_before_public_transform(pairs):
    copies = augment_demonstrations(pairs.new_walls, seed=3)
    assert len(copies.episodes) == 8
    assert restore(copies).provenance() == copies.provenance()
    for origin, styled in zip(pairs.new_walls.episodes, copies.episodes, strict=True):
        expected = present_frame(pairs.new_walls.frames[origin.frame_indices[0]],
                                 PresentationProfile(**styled.presentation))
        np.testing.assert_array_equal(copies.frames[styled.frame_indices[0]], expected)


def test_insufficient_pairs_fail_explicitly(base):
    empty = DemonstrationDataset()
    with pytest.raises(ValueError, match="requested 8 pairs, found 0"):
        collect_navigation_geometry_pairs(empty)
    with pytest.raises(ValueError, match="requested 64 pairs"):
        collect_navigation_geometry_pairs(base, max_pairs=64)


def test_serialization_revalidates_mutable_geometry_provenance(pairs):
    row = replace(pairs.new_walls.episodes[0],
                  runtime_level_override=deepcopy(pairs.new_walls.episodes[0].runtime_level_override))
    row.runtime_level_override["walls"] = [[0, 0], [0, 0]]
    with pytest.raises(ValueError):
        row.to_dict()


def test_incremental_collection_excludes_every_geometry_observed_in_base(base, pairs):
    combined = restore(base)
    combined.extend(pairs.new_walls)
    # Include presentation copies too: the protection tracks geometry, not style.
    combined.extend(augment_demonstrations(pairs.new_walls, seed=7))
    original = combined.provenance()
    repeated = collect_navigation_geometry_pairs(combined)
    previous_walls = {tuple(map(tuple, row.runtime_level_override["walls"]))
                      for row in pairs.new_walls.episodes}
    fresh_walls = {tuple(map(tuple, row.runtime_level_override["walls"]))
                   for row in repeated.new_walls.episodes}
    assert len(fresh_walls) == 8
    assert fresh_walls.isdisjoint(previous_walls)
    assert repeated.manifest["unique_new_wall_layouts"] == 8
    assert repeated.manifest["base_observed_wall_layouts"] == 24
    assert combined.provenance() == original


@pytest.mark.parametrize("limit", ["caller", "dataset"])
def test_combined_budget_rejected_before_mutating_base(base, limit):
    dataset = restore(base)
    before = dataset.provenance()
    options = {}
    if limit == "caller":
        options["max_frame_bytes"] = dataset.frame_bytes
    else:
        dataset.max_frame_bytes = dataset.frame_bytes
        before = dataset.provenance()
    with pytest.raises(MemoryError, match="combined RGB byte budget"):
        collect_navigation_geometry_pairs(dataset, **options)
    assert dataset.provenance() == before


def test_manifest_combined_bytes_equal_actual_append_for_both_arms(base, pairs):
    for mode in ("repeat", "new_walls"):
        combined = restore(base)
        combined.extend(pairs.arm(mode))
        assert combined.frame_bytes == pairs.manifest["combined_frame_bytes"][mode]
        assert combined.frame_bytes <= pairs.manifest["combined_frame_budget_bytes"]
    assert pairs.manifest["combined_frame_bytes"]["repeat"] == base.frame_bytes
    assert pairs.manifest["combined_frame_bytes"]["new_walls"] > base.frame_bytes
