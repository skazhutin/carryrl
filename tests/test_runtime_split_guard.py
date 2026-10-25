"""Admission uses actual curriculum mechanics, including defaults and runtime overrides."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from carryrl.envs.atomic import _easier_level, _identity, atomic_level_banks
from carryrl.envs.atomic_suite import TASKS
from carryrl.envs.runtime_identity import (
    allowed_runtime_train_indices,
    atomic_runtime_level_id,
    runtime_heldout_collisions,
)
from carryrl.training.imitation import (
    DemonstrationDataset,
    collect_memory_cue_counterparts,
    collect_native_scripted_demonstrations,
    collect_scripted_corrections,
    collect_scripted_demonstrations,
    validate_dataset_runtime_holdout,
)

BANK = 20261008


@pytest.fixture(autouse=True)
def clear_runtime_cache():
    from carryrl.envs.runtime_identity import _heldout_runtime_levels

    _heldout_runtime_levels.cache_clear()
    yield
    _heldout_runtime_levels.cache_clear()


def test_ui_curriculum_collapses_distinct_train_and_test_origins():
    banks = atomic_level_banks("ui", BANK)
    train, test = banks["train"][2], banks["test"][3]
    assert _identity(train) != _identity(test)
    assert atomic_runtime_level_id("ui", train, 1) != atomic_runtime_level_id("ui", test, 1)
    assert atomic_runtime_level_id("ui", train, 0) == atomic_runtime_level_id("ui", test, 0)
    assert 2 not in allowed_runtime_train_indices("ui", BANK, 0)
    matches = runtime_heldout_collisions("ui", train, 0, BANK)
    assert any(row["level_index"] == 3 and row["split"] == "test" for row in matches)


@pytest.mark.parametrize("kind", ["teacher", "correction"])
@pytest.mark.parametrize("bank_seed", [0, BANK])
@pytest.mark.parametrize("tasks", [("platforming",), ("navigation", "platforming")])
def test_platforming_collapsed_bank_fails_before_environment_creation(kind, bank_seed, tasks):
    options = dict(tasks=tasks, difficulty=0, level_bank_seed=bank_seed,
                   env_factory=lambda *_a, **_k: pytest.fail("unsafe level reached runtime"))
    with pytest.raises(ValueError, match="no runtime levels without held-out matches"):
        if kind == "teacher":
            collect_scripted_demonstrations(1, **options)
        else:
            agent = SimpleNamespace(training=True, eval=lambda: None, train=lambda _: None)
            collect_scripted_corrections(agent, 1, **options)


def test_collection_filters_deterministically_and_admission_rechecks_imports():
    allowed = allowed_runtime_train_indices("ui", BANK, 0)
    dataset = collect_scripted_demonstrations(3, tasks=("ui",), difficulty=0,
                                             level_bank_seed=BANK, seed=2)
    assert [row.level_index for row in dataset.episodes] == [allowed[i] for i in (2, 3, 4)]
    unsafe = replace(dataset.episodes[0], level_index=2,
                     level_id=_identity(atomic_level_banks("ui", BANK)["train"][2]))
    with pytest.raises(ValueError, match="held-out runtime semantic level"):
        dataset.add_episode(unsafe)
    imported = DemonstrationDataset()
    imported.frames = dataset.frames
    imported.episodes = [unsafe]
    with pytest.raises(ValueError, match="held-out runtime semantic level"):
        imported._validate_episode(imported.episodes[0])


def test_native_collection_cannot_repeat_filtered_levels(monkeypatch):
    monkeypatch.setattr("carryrl.training.imitation.allowed_runtime_train_indices",
                        lambda *_: [0])
    with pytest.raises(ValueError, match="enough unique runtime levels"):
        collect_native_scripted_demonstrations(native_train_sizes=(64,),
            native_episodes_per_task=2, tasks=("navigation",), difficulty=1,
            env_factory=lambda *_a, **_k: pytest.fail("invalid count reached runtime"))


def test_native_collection_requires_unique_mechanics_among_distinct_origins(monkeypatch):
    # Legacy UI origins can differ only by unused decoration. They must not
    # inflate the advertised number of distinct native training mechanics.
    bank = atomic_level_banks("ui", 0)["train"]
    allowed = allowed_runtime_train_indices("ui", 0, 1)
    by_runtime = {}
    for index in allowed:
        by_runtime.setdefault(atomic_runtime_level_id("ui", bank[index]), []).append(index)
    duplicate = next(indices for indices in by_runtime.values() if len(indices) > 1)
    monkeypatch.setattr("carryrl.training.imitation.allowed_runtime_train_indices",
                        lambda *_: duplicate[:2])
    with pytest.raises(ValueError, match="enough unique runtime levels"):
        collect_native_scripted_demonstrations(native_train_sizes=(64,),
            native_episodes_per_task=2, tasks=("ui",), difficulty=1,
            env_factory=lambda *_a, **_k: pytest.fail("duplicate mechanics reached runtime"))


def test_defaults_and_used_difficulty_parameters_match_runtime():
    for task, defaults in (("timing", {"required_pipes": 3}),
                           ("steering", {"required_progress": 72.0}),
                           ("goal_choice", {"goal_spread": 31.0}),
                           ("shooting", {"required_hits": 3, "aim_radius": 5})):
        level = atomic_level_banks(task, BANK)["train"][0]
        assert atomic_runtime_level_id(task, level) == atomic_runtime_level_id(
            task, {**level, **defaults})
    for task in ("reaction", "mouse"):
        level = atomic_level_banks(task, BANK)["train"][0]
        # These curricula change runtime controls even if the level dictionary doesn't.
        assert atomic_runtime_level_id(task, level, 0) != atomic_runtime_level_id(task, level, 1)
    camera = _easier_level("camera", atomic_level_banks("camera", BANK)["train"][0], 0)
    assert atomic_runtime_level_id("camera", camera, 1) != atomic_runtime_level_id(
        "camera", {**camera, "difficulty": 1}, 1)


@pytest.mark.parametrize("difficulty,other_targets", [(0, [1, 8, 9]), (.25, [1, 2, 9])])
def test_mouse_unused_target_suffix_does_not_create_new_mechanics(difficulty, other_targets):
    level = {"cursor": 0, "targets": [1, 2, 3]}
    other = {**level, "targets": other_targets}
    assert atomic_runtime_level_id("mouse", level, difficulty) == atomic_runtime_level_id(
        "mouse", other, difficulty)
    assert atomic_runtime_level_id("mouse", level, 1) != atomic_runtime_level_id("mouse", other, 1)


@pytest.mark.parametrize("task,field", [("navigation", "walls"),
    ("interaction", "walls"), ("causality", "walls"), ("key_door", "walls"),
    ("chain", "walls"), ("hazards", "traps")])
def test_reordered_static_cells_cannot_hide_heldout_mechanics(task, field):
    heldout = atomic_level_banks(task, BANK)["validation"][0]
    reordered = {**heldout, field: list(reversed(heldout[field])) + [heldout[field][0]]}
    assert set(map(tuple, heldout[field])) == set(map(tuple, reordered[field]))
    assert _identity(heldout) != _identity(reordered)
    assert atomic_runtime_level_id(task, heldout) == atomic_runtime_level_id(task, reordered)
    matches = runtime_heldout_collisions(task, reordered, 1, BANK)
    assert any(row["split"] == "validation" and row["level_index"] == 0 for row in matches)


@pytest.mark.parametrize("task,field", [("mouse", "targets"), ("hazards", "patrol")])
def test_ordered_targets_and_patrols_remain_distinct(task, field):
    level = atomic_level_banks(task, BANK)["train"][0]
    reordered = {**level, field: list(reversed(level[field]))}
    assert atomic_runtime_level_id(task, level) != atomic_runtime_level_id(task, reordered)


def test_memory_cue_override_and_other_bank_heldouts_are_checked():
    heldout = atomic_level_banks("memory", BANK)["test"][0]
    flipped = {**heldout, "cue": 1 - heldout["cue"]}
    assert runtime_heldout_collisions("memory", {**flipped, "cue": heldout["cue"]}, 1, 0)
    assert all(len(allowed_runtime_train_indices(task, BANK, 1)) == 64 for task in TASKS)


def test_malformed_memory_override_is_rejected_before_runtime_signature():
    origin = collect_scripted_demonstrations(1, tasks=("memory",), difficulty=1,
                                            level_bank_seed=BANK)
    dataset = collect_memory_cue_counterparts(origin, max_pairs=1)
    assert len(dataset.episodes) == 1
    malformed = replace(dataset.episodes[0], runtime_level_override={"cue": 0})
    with pytest.raises(ValueError, match="runtime must equal its origin"):
        dataset._validate_episode(malformed)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -.1, 1.1])
def test_extra_heldout_difficulty_requires_finite_unit_interval(value):
    level = atomic_level_banks("navigation", BANK)["train"][0]
    with pytest.raises(ValueError, match="difficulties must be finite"):
        runtime_heldout_collisions("navigation", level, 1, BANK,
                                   heldout_difficulties=(value,))


def test_final_validator_protects_different_endpoint_bank(monkeypatch):
    dataset = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=1,
                                             level_bank_seed=BANK)
    episode = dataset.episodes[0]
    level = atomic_level_banks("navigation", BANK)["train"][episode.level_index]
    real_banks = atomic_level_banks

    def banks(task, seed=0):
        result = real_banks(task, seed)
        return {**result, "test": (level,)} if seed == 20261009 else result

    monkeypatch.setattr("carryrl.envs.runtime_identity.atomic_level_banks", banks)
    assert not runtime_heldout_collisions("navigation", level, 1, BANK)
    before = dataset.provenance()
    with pytest.raises(ValueError, match="evaluation endpoint"):
        validate_dataset_runtime_holdout(dataset, level_bank_seed=20261009,
                                          evaluation_difficulty=1)
    assert dataset.provenance() == before


def test_final_validator_protects_other_endpoint_difficulty(monkeypatch):
    dataset = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=1,
                                             level_bank_seed=BANK)
    real_banks = atomic_level_banks
    heldouts = real_banks("navigation", BANK)["validation"]
    level = next(_easier_level("navigation", heldout, .25) for heldout in heldouts
                 if not runtime_heldout_collisions("navigation",
                     _easier_level("navigation", heldout, .25), 1, BANK))

    def banks(task, seed=0):
        result = real_banks(task, seed)
        return {**result, "train": (level,)} if seed == BANK else result

    monkeypatch.setattr("carryrl.envs.runtime_identity.atomic_level_banks", banks)
    monkeypatch.setattr("carryrl.training.imitation.atomic_level_banks", banks)
    dataset.episodes[0] = replace(dataset.episodes[0], level_index=0, level_id=_identity(level))
    assert not runtime_heldout_collisions("navigation", level, 1, BANK)
    with pytest.raises(ValueError, match="evaluation endpoint"):
        validate_dataset_runtime_holdout(dataset, level_bank_seed=BANK,
                                          evaluation_difficulty=.25)
