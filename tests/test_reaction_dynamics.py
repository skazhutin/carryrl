"""Reaction speed diversity authenticates whole actual RGB teacher trajectories."""

import json
import math
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv, _identity, atomic_level_banks
from carryrl.envs.presentation import PresentationProfile, present_frame
from carryrl.envs.runtime_identity import atomic_runtime_level_id
from carryrl.training.imitation import (
    REACTION_DYNAMICS_SOURCE,
    DemonstrationDataset,
    DemonstrationEpisode,
    augment_demonstrations,
    collect_scripted_demonstrations,
    imitation_replay_summary,
    scripted_teacher,
)
from carryrl.training.reaction_dynamics import (
    ReactionDynamicsEnv,
    _changed_runtime,
    _derived_heldout_signatures,
    _protected_bank_seeds,
    _rollout,
    collect_reaction_dynamics_pairs,
    validate_reaction_runtime,
)
from carryrl.training.sequence_config import SequenceRunConfig


@pytest.fixture(scope="module")
def base():
    return collect_scripted_demonstrations(16, tasks=("reaction",), difficulty=1,
                                           max_frame_bytes=64 * 1024**2)


@pytest.fixture(scope="module")
def pairs(base):
    return collect_reaction_dynamics_pairs(base,
        config=SequenceRunConfig(tasks=("reaction",), level_bank_seed=20261008))


def restore(dataset):
    rebuilt = DemonstrationDataset(max_frame_bytes=dataset.max_frame_bytes)
    for frame in dataset.frames:
        rebuilt.intern_frame(frame)
    for record in json.loads(json.dumps(dataset.provenance()))["episodes"]:
        values = {key: value for key, value in record.items()
                  if key in DemonstrationEpisode.__dataclass_fields__}
        values["frame_indices"] = tuple(values["frame_indices"])
        values["actions"] = tuple(PhysicalAction.from_dict(a) for a in values["actions"])
        if "executed_actions" in values:
            values["executed_actions"] = tuple(PhysicalAction.from_dict(a)
                                               for a in values["executed_actions"])
        rebuilt.add_episode(DemonstrationEpisode(**values))
    return rebuilt


def test_actual_eight_pairs_equal_decisions_distinct_actions_and_deterministic(base, pairs):
    before = base.provenance()
    banks = deepcopy(atomic_level_banks("reaction"))
    duplicate = collect_reaction_dynamics_pairs(base,
        config=SequenceRunConfig(tasks=("reaction",), level_bank_seed=20261008))
    assert duplicate.manifest == pairs.manifest
    assert before == base.provenance()
    assert banks == atomic_level_banks("reaction")
    assert pairs.manifest["pairs_collected"] == 8
    assert pairs.manifest["unique_new_runtime_configs"] == 8
    assert pairs.manifest["equal_ordered_lengths"] is True
    assert pairs.manifest["ordered_lengths"] == {"repeat": [63] * 8, "varied": [63] * 8}
    assert pairs.manifest["test_gameplay_used"] is False
    assert pairs.manifest["same_episode_replay_domain"] == "canonical_unstyled_96"
    assert pairs.manifest["derived_heldout_protected_banks"] == [0, 20261008, 20261015]
    assert pairs.manifest["derived_heldout_guard_passed"] is True
    assert imitation_replay_summary(pairs.repeat, canonical_replay_weight=3) == (
        imitation_replay_summary(pairs.varied, canonical_replay_weight=3))
    for old, new, entry in zip(pairs.repeat.episodes, pairs.varied.episodes,
                               pairs.manifest["pairs"], strict=True):
        assert len(old) == len(new) == 63
        assert new.actions != old.actions
        assert new.executed_actions == new.actions
        assert new.source == REACTION_DYNAMICS_SOURCE
        assert old.source == "scripted_teacher"
        assert new.outcome == old.outcome
        assert new.quality["hits"] == old.quality["hits"] == 3
        assert new.instruction == old.instruction
        assert new.terminated is True and new.truncated is False
        assert entry["repeat_costs"]["input_units"] == old.quality["input_units"]
        assert entry["varied_costs"]["input_units"] == new.quality["input_units"]
        assert math.copysign(2, new.runtime_level_override["ball_dx"]) == (
            new.runtime_level_override["ball_dx"])
        assert np.array_equal(pairs.repeat.frames[old.frame_indices[0]],
                              pairs.varied.frames[new.frame_indices[0]])
        assert any(not np.array_equal(pairs.repeat.frames[a], pairs.varied.frames[b])
                   for a, b in zip(old.frame_indices, new.frame_indices, strict=True))
    costs = pairs.manifest["delta_costs_per_arm"]
    assert costs["repeat"]["input_units"] != costs["varied"]["input_units"]
    assert costs["repeat"]["command_units"] == costs["varied"]["command_units"] == 504
    for mode in ("repeat", "varied"):
        assert restore(pairs.arm(mode)).provenance() == pairs.arm(mode).provenance()
        assert duplicate.arm(mode).provenance() == pairs.arm(mode).provenance()


def test_reset_changes_actual_velocity_cache_and_bounces_keep_speed(pairs):
    row = pairs.varied.episodes[0]
    env = ReactionDynamicsEnv(row.runtime_level_override, level_bank_seed=row.level_bank_seed)
    ordinary = AtomicEnv("reaction", split="train", difficulty=1, image_size=96)
    try:
        frame, _ = env.reset(seed=row.seed, options={"level_index": row.level_index})
        old_frame, _ = ordinary.reset(seed=row.seed, options={"level_index": row.level_index})
        assert np.array_equal(frame, old_frame)
        assert env._level is not row.runtime_level_override
        assert env._ball_dx == row.runtime_level_override["ball_dx"]
        assert env._effective_level_id == _identity(row.runtime_level_override)
        for attr in ("_ball_x", "_ball_y", "_paddle_x", "_reaction_drop",
                     "_required_hits", "_paddle_radius", "_hits"):
            assert getattr(env, attr) == getattr(ordinary, attr)
        teacher = scripted_teacher("reaction")
        bounces, round_resets = 0, 0
        for _ in range(63):
            previous_x, previous_dx, previous_hits = env._ball_x, env._ball_dx, env._hits
            action = teacher.act(frame)
            frame, _, terminated, truncated, _ = env.step(action)
            expected_x = previous_x + previous_dx
            if expected_x < 4 or expected_x > 92:
                bounces += 1
                assert env._ball_dx == -previous_dx
            else:
                assert env._ball_dx == previous_dx
            assert abs(env._ball_dx) == 2.0
            if env._hits != previous_hits and not terminated:
                round_resets += 1
                assert env._ball_x == (env._level["ball_x"] + 13 * env._hits) % 80 + 8
            elif not terminated:
                assert env._ball_x == np.clip(expected_x, 4, 92)
            if terminated or truncated:
                break
        assert bounces >= 1 and round_resets == 2
        assert env._hits == 3
    finally:
        env.close()
        ordinary.close()


@pytest.mark.parametrize("mutation", [
    {"task": "memory"}, {"difficulty": 0}, {"difficulty": True}, {"native_size": 64},
    {"executed_actions": None}, {"terminated": False}, {"truncated": True},
    {"quality": None}, {"origin_level_id": None},
])
def test_source_contract_rejects_invalid_rows(pairs, mutation):
    with pytest.raises(ValueError):
        replace(pairs.varied.episodes[0], **mutation)


@pytest.mark.parametrize("field,value", [
    ("ball_dx", 2.5), ("ball_dx", True), ("ball_dx", float("nan")),
    ("ball_x", False), ("paddle_x", float("inf")), ("paddle_x", -100), ("extra", 3),
])
def test_runtime_rejects_physics_magnitudes_and_nonfinite(pairs, field, value):
    row = pairs.varied.episodes[0]
    runtime = {**row.runtime_level_override, field: value}
    with pytest.raises(ValueError):
        pairs.varied._validate_episode(replace(row, runtime_level_override=runtime,
                                                runtime_level_id=_identity(runtime)))


def test_runtime_rejects_sign_flip_wrong_origin_and_hash(pairs):
    row = pairs.varied.episodes[0]
    runtime = {**row.runtime_level_override, "ball_dx": -row.runtime_level_override["ball_dx"]}
    for changed in (replace(row, runtime_level_override=runtime,
                            runtime_level_id=_identity(runtime)),
                    replace(row, level_index=(row.level_index + 1) % 64),
                    replace(row, runtime_level_id="bad")):
        with pytest.raises(ValueError):
            pairs.varied._validate_episode(changed)
    origin = atomic_level_banks("reaction")["train"][row.level_index]
    with pytest.raises(ValueError):
        validate_reaction_runtime(row, {**origin, "ball_x": origin["ball_x"] + 1})


def test_trace_rejects_action_history_rgb_and_quality_tampering(pairs):
    dataset = restore(pairs.varied)
    row = dataset.episodes[0]
    changed = (PhysicalAction(), *row.actions[1:])
    if changed == row.actions:
        changed = (PhysicalAction(keys=("left",)), *row.actions[1:])
    with pytest.raises(ValueError):
        dataset.add_episode(replace(row, actions=changed, executed_actions=changed))
    with pytest.raises(ValueError):
        dataset.add_episode(replace(row, executed_actions=changed))
    for final in (False, True):
        index = row.final_frame_index if final else row.frame_indices[10]
        rgb = dataset.frames[index].copy()
        rgb[0, 0] ^= 1
        index = dataset.intern_frame(rgb)
        indices = (*row.frame_indices[:10], index, *row.frame_indices[11:])
        with pytest.raises(ValueError, match="RGB"):
            dataset.add_episode(replace(row, **({"final_frame_index": index} if final else
                                                {"frame_indices": indices})))
    with pytest.raises(ValueError, match="quality or boundary"):
        dataset.add_episode(replace(row, quality={**row.quality, "hits": 4}))


def test_serialization_rechecks_mutable_runtime(pairs):
    row = replace(pairs.varied.episodes[0],
                  runtime_level_override=deepcopy(pairs.varied.episodes[0].runtime_level_override))
    row.runtime_level_override["ball_dx"] *= -1
    with pytest.raises(ValueError):
        row.to_dict()


def test_appearance_copies_keep_canonical_labels_and_actual_history(pairs):
    copies = augment_demonstrations(pairs.varied, seed=3)
    assert restore(copies).provenance() == copies.provenance()
    for row, styled in zip(pairs.varied.episodes, copies.episodes, strict=True):
        assert row.actions == styled.actions == styled.executed_actions
        assert row.quality == styled.quality
        expected = present_frame(pairs.varied.frames[row.frame_indices[0]],
                                 PresentationProfile(**styled.presentation))
        np.testing.assert_array_equal(copies.frames[styled.frame_indices[0]], expected)


def test_incremental_collection_excludes_runtime_even_in_appearance_copies(base, pairs):
    combined = restore(base)
    combined.max_frame_bytes = 256 * 1024**2
    combined.extend(pairs.varied)
    combined.extend(augment_demonstrations(pairs.varied, seed=7))
    before = combined.provenance()
    fresh = collect_reaction_dynamics_pairs(combined)
    old = {atomic_runtime_level_id("reaction", row.runtime_level_override)
           for row in pairs.varied.episodes}
    new = {atomic_runtime_level_id("reaction", row.runtime_level_override)
           for row in fresh.varied.episodes}
    assert old.isdisjoint(new)
    assert fresh.manifest["base_observed_runtime_configs"] == 24
    assert combined.provenance() == before
    combined.extend(fresh.varied)
    with pytest.raises(ValueError, match="requested 1 pairs, found 0"):
        collect_reaction_dynamics_pairs(combined, max_pairs=1)


def test_distinct_old_velocities_collapsing_to_one_new_runtime_count_once(base):
    dataset = DemonstrationDataset()
    for seed, index in ((0, 56), (20261008, 40)):
        origin = atomic_level_banks("reaction", seed)["train"][index]
        row = replace(base.episodes[0], level_bank_seed=seed, level_index=index,
                      level_id=_identity(origin))
        frames, actions, final, outcome, quality = _rollout(origin, row, 80)
        dataset.add_episode(replace(row, actions=actions, quality=quality, outcome=outcome,
            frame_indices=tuple(dataset.intern_frame(f) for f in frames),
            final_frame_index=dataset.intern_frame(final)))
    first, second = dataset.episodes
    first_origin = atomic_level_banks("reaction", first.level_bank_seed)["train"][first.level_index]
    second_origin = atomic_level_banks("reaction", second.level_bank_seed)["train"][
        second.level_index]
    assert first_origin["ball_dx"] != second_origin["ball_dx"]
    assert _changed_runtime(first_origin) == _changed_runtime(second_origin)
    pair = collect_reaction_dynamics_pairs(dataset, max_pairs=1)
    assert pair.manifest["eligible_parents"] == 2
    assert pair.manifest["unique_new_runtime_configs"] == 1
    with pytest.raises(ValueError, match="requested 2 pairs, found 1"):
        collect_reaction_dynamics_pairs(dataset, max_pairs=2)


@pytest.mark.parametrize("limit", ["caller", "dataset"])
def test_combined_budget_fails_without_base_mutation(base, limit):
    dataset = restore(base)
    options = {}
    if limit == "dataset":
        dataset.max_frame_bytes = dataset.frame_bytes
    else:
        options["max_frame_bytes"] = dataset.frame_bytes
    before = dataset.provenance()
    with pytest.raises(MemoryError, match="combined RGB byte budget"):
        collect_reaction_dynamics_pairs(dataset, **options)
    assert before == dataset.provenance()


def test_manifest_unique_byte_counts_match_actual_combination(base, pairs):
    for mode in ("repeat", "varied"):
        combined = restore(base)
        combined.extend(pairs.arm(mode))
        assert combined.frame_bytes == pairs.manifest["combined_frame_bytes"][mode]
    assert pairs.manifest["combined_frame_bytes"]["repeat"] == base.frame_bytes


def test_missing_and_inadequate_parents_fail_explicitly(base):
    with pytest.raises(ValueError, match="requested 8 pairs, found 0"):
        collect_reaction_dynamics_pairs(DemonstrationDataset())
    with pytest.raises(ValueError, match="requested 64 pairs, found 16"):
        collect_reaction_dynamics_pairs(base, max_pairs=64)


@pytest.mark.parametrize("options", [
    {"max_pairs": 0}, {"max_pairs": True}, {"max_pairs": 65}, {"seed": -1}, {"seed": True},
    {"max_steps": 0}, {"max_frame_bytes": 0},
    {"config": SequenceRunConfig(tasks=("memory",))},
    {"config": SequenceRunConfig(tasks=("reaction",), env_image_size=64)},
])
def test_collector_rejects_invalid_options_before_rollout(base, options):
    with pytest.raises(ValueError):
        collect_reaction_dynamics_pairs(base, **options)


def test_actual_teacher_parent_tampering_is_rejected(base):
    dataset = restore(base)
    row = dataset.episodes[0]
    changed = (PhysicalAction(keys=("space",)), *row.actions[1:])
    # SOURCE predates replay authentication; this control must authenticate it explicitly.
    dataset.episodes = [replace(row, actions=changed, quality=None, quality_source=None)]
    with pytest.raises(ValueError, match="authentic canonical teacher trace"):
        collect_reaction_dynamics_pairs(dataset, max_pairs=1)


def test_legacy_parent_optional_quality_and_actual_history_are_authenticated(base):
    dataset = restore(base)
    original = dataset.episodes[0]
    dataset.episodes = [replace(original, quality=None, quality_source=None,
                                executed_actions=original.actions)]
    pair = collect_reaction_dynamics_pairs(dataset, max_pairs=1)
    assert pair.repeat.episodes[0].quality is None
    assert pair.varied.episodes[0].quality["hits"] == 3
    assert pair.manifest["pairs"][0]["repeat_costs"]["input_units"] == (
        original.quality["input_units"])
    dataset.episodes = [replace(original, quality={**original.quality, "hits": 4})]
    with pytest.raises(ValueError, match="authentic canonical teacher trace"):
        collect_reaction_dynamics_pairs(dataset, max_pairs=1)


def test_short_step_budget_never_crops_or_relabels_parents(base):
    with pytest.raises(ValueError, match="terminate in visible success"):
        collect_reaction_dynamics_pairs(base, max_pairs=1, max_steps=62)


def _single_train_parent(template, bank_seed, index):
    dataset = DemonstrationDataset()
    origin = atomic_level_banks("reaction", bank_seed)["train"][index]
    row = replace(template, level_bank_seed=bank_seed, level_index=index,
                  level_id=_identity(origin))
    frames, actions, final, outcome, quality = _rollout(origin, row, 80)
    dataset.add_episode(replace(row, actions=actions, outcome=outcome, quality=quality,
        frame_indices=tuple(dataset.intern_frame(frame) for frame in frames),
        final_frame_index=dataset.intern_frame(final)))
    return dataset


def test_speed_transform_collision_with_other_known_bank_is_rejected_before_rollout(
        base, pairs, monkeypatch):
    origin = atomic_level_banks("reaction", 20261008)["train"][39]
    heldout = atomic_level_banks("reaction", 20261015)["validation"][3]
    assert _changed_runtime(origin) == _changed_runtime(heldout)
    unsafe = _single_train_parent(base.episodes[0], 20261008, 39)
    runtime = _changed_runtime(origin)
    row = replace(pairs.varied.episodes[0], level_bank_seed=20261008, level_index=39,
        level_id=_identity(origin), origin_level_id=_identity(origin),
        runtime_level_id=_identity(runtime), runtime_level_override=runtime)
    with pytest.raises(ValueError, match="speed-transformed held-out"):
        validate_reaction_runtime(row, origin)
    with pytest.raises(ValueError, match="speed-transformed held-out"):
        pairs.varied._validate_episode(row)
    with pytest.raises(ValueError, match="speed-transformed held-out"):
        row.to_dict()
    # Safe genuine pairs still pass the same import guard.
    safe = pairs.varied.episodes[0]
    safe_origin = atomic_level_banks("reaction", safe.level_bank_seed)["train"][safe.level_index]
    assert validate_reaction_runtime(safe, safe_origin) == safe.runtime_level_override

    def forbidden_rollout(*args, **kwargs):
        pytest.fail("colliding transformed candidate reached a rollout")

    monkeypatch.setattr("carryrl.training.reaction_dynamics._rollout", forbidden_rollout)
    with pytest.raises(ValueError, match="requested 1 pairs, found 0"):
        collect_reaction_dynamics_pairs(unsafe, max_pairs=1)


def test_config_endpoint_bank_extends_derived_guard_without_changing_fixed_banks(base):
    dataset = _single_train_parent(base.episodes[0], 0, 16)
    origin = atomic_level_banks("reaction")["train"][16]
    heldout = atomic_level_banks("reaction", 47)["validation"][7]
    assert origin != heldout
    assert _changed_runtime(origin) == _changed_runtime(heldout)
    signature = atomic_runtime_level_id("reaction", _changed_runtime(origin))
    assert signature not in _derived_heldout_signatures(_protected_bank_seeds())
    assert signature in _derived_heldout_signatures((47,))
    before = deepcopy(atomic_level_banks("reaction", 47))
    safe = collect_reaction_dynamics_pairs(dataset, max_pairs=1)
    assert safe.manifest["derived_heldout_guard_passed"] is True
    with pytest.raises(ValueError, match="requested 1 pairs, found 0"):
        collect_reaction_dynamics_pairs(dataset, max_pairs=1,
            config=SequenceRunConfig(tasks=("reaction",), level_bank_seed=47))
    assert before == atomic_level_banks("reaction", 47)


def test_unlisted_source_bank_own_transformed_heldout_is_protected(base, pairs):
    dataset = _single_train_parent(base.episodes[0], 8, 32)
    origin = atomic_level_banks("reaction", 8)["train"][32]
    heldout = atomic_level_banks("reaction", 8)["test"][6]
    assert origin != heldout
    runtime = _changed_runtime(origin)
    assert runtime == _changed_runtime(heldout)
    signature = atomic_runtime_level_id("reaction", runtime)
    assert signature not in _derived_heldout_signatures(_protected_bank_seeds())
    row = replace(pairs.varied.episodes[0], level_bank_seed=8, level_index=32,
        level_id=_identity(origin), origin_level_id=_identity(origin),
        runtime_level_id=_identity(runtime), runtime_level_override=runtime)
    with pytest.raises(ValueError, match="speed-transformed held-out"):
        pairs.varied._validate_episode(row)
    with pytest.raises(ValueError, match="requested 1 pairs, found 0"):
        collect_reaction_dynamics_pairs(dataset, max_pairs=1)
