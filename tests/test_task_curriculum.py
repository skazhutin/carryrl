"""Task stage semantics, split admission, and transactional adaptive scheduling."""

import copy
import hashlib
import json

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv, _easier_level, atomic_level_banks
from carryrl.envs.atomic_suite import TASKS
from carryrl.envs.curriculum import STAGES, resolve_effective_level, runtime_parameters
from carryrl.envs.runtime_identity import atomic_runtime_level_id
from carryrl.training.task_curriculum import (
    TaskCurriculumScheduler,
    curriculum_allowed_levels,
    prepare_task_curriculum,
    validate_task_curriculum_admission,
)


@pytest.mark.parametrize("task", TASKS)
def test_full_stage_matches_original_frames_dynamics_and_semantics(task):
    source = atomic_level_banks(task, 20261008)["train"][0]
    before = copy.deepcopy(source)
    assert resolve_effective_level(task, source, curriculum_stage="full") == source
    assert atomic_runtime_level_id(task, source, curriculum_stage="full") == (
        atomic_runtime_level_id(task, source))
    plain = AtomicEnv(task, level_bank_seed=20261008)
    staged = AtomicEnv(task, level_bank_seed=20261008, curriculum_stage="full")
    a, _ = plain.reset(options={"level_index": 0})
    b, info = staged.reset(options={"level_index": 0})
    np.testing.assert_array_equal(a, b)
    assert info["runtime_fingerprint"] == atomic_runtime_level_id(task, source)
    for _ in range(8):
        a = plain.step(PhysicalAction(keys=("right",)))
        b = staged.step(PhysicalAction(keys=("right",)))
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:4] == b[1:4]
        if a[2] or a[3]:
            break
    assert source == before


@pytest.mark.parametrize("split,difficulty,stage", [
    ("validation", 1, "easy"), ("test", 1, "full"),
    ("train", 0.5, "easy"), ("train", 1, "unknown"),
])
def test_explicit_stage_rejects_wrong_split_or_scalar(split, difficulty, stage):
    with pytest.raises(ValueError):
        AtomicEnv("navigation", split=split, difficulty=difficulty, curriculum_stage=stage)


@pytest.mark.parametrize("task", TASKS)
def test_legacy_scalar_resolution_stays_exact(task):
    source = atomic_level_banks(task, 20261008)["train"][0]
    for difficulty in (0, 0.35, 0.7, 1):
        assert resolve_effective_level(task, source, difficulty=difficulty) == (
            _easier_level(task, source, difficulty))


def test_stage_parameters_preserve_obstacles_mechanisms_and_cue_geometry():
    for task in TASKS:
        source = atomic_level_banks(task, 20261008)["train"][0]
        easy = resolve_effective_level(task, source, curriculum_stage="easy")
        bridge = resolve_effective_level(task, source, curriculum_stage="bridge")
        if task == "platforming":
            assert easy["obstacles"] == [source["obstacles"][0]]
            assert easy["goal_x"] == sum(source["obstacles"][0][:2]) + 10
            assert len(bridge["obstacles"]) == 2
            assert all(height <= 8 for _, _, height in bridge["obstacles"])
        if task in ("causality", "key_door", "chain"):
            assert {k: v for k, v in easy.items() if k != "start"} == {
                k: v for k, v in source.items() if k != "start"}
        if task == "memory":
            assert easy["unlock"] == source["cue_ticks"] + 4
            assert bridge["unlock"] == 32
            for key in ("cue", "cue_ticks", "left_x", "right_x"):
                assert easy[key] == bridge[key] == source[key]
        if task == "ui":
            assert len(easy["items"]) == 4
            assert easy["items"][easy["selection"]] != easy["request"]
            assert bridge["items"][bridge["selection"]] != bridge["request"]
        if task == "reaction":
            assert (easy["required_hits"], bridge["required_hits"]) == (1, 2)
        if task == "mouse":
            assert (easy["cursor_distance"], bridge["cursor_distance"]) == (16, 32)


def test_runtime_identity_uses_actual_constructor_defaults_and_ignores_unused_data():
    source = atomic_level_banks("timing", 20261008)["train"][0]
    changed = copy.deepcopy(source)
    changed.update(required_pipes=1, gaps=[source["gaps"][0], 999, 999], seed=7, artwork="x")
    original = copy.deepcopy(changed)
    original.update(gaps=[source["gaps"][0], -999, -999], seed=99, artwork="y")
    assert atomic_runtime_level_id("timing", changed) == atomic_runtime_level_id("timing", original)
    params = runtime_parameters("timing", changed)
    assert len(params["pipes"]) == len(params["gaps"]) == 1
    env = AtomicEnv("camera", level_bank_seed=20261008, curriculum_stage="easy")
    _, info = env.reset(options={"level_index": 0})
    assert env._game._tolerances == {"yaw_tolerance": 0.16, "pitch_tolerance": 0.10}
    assert info["runtime_fingerprint"] == atomic_runtime_level_id(
        "camera", info["effective_parameters"], curriculum_stage="full")


def _scheduler(state=None):
    return TaskCurriculumScheduler(config_hash="config", admission_hash="admission",
                                   allowed_levels={task: dict.fromkeys(STAGES, list(range(16)))
                                                   for task in TASKS}, state=state)


def _outcomes(scheduler, task, count, success=True):
    tickets = []
    for _ in range(count):
        ticket = scheduler.select_episode(task)
        scheduler.record_outcome(ticket, success, f"{ticket['stage']}:{ticket['level_index']}")
        tickets.append(ticket)
    return tickets


def test_promotion_is_committed_transactionally_and_resume_is_exact():
    scheduler = _scheduler()
    _outcomes(scheduler, "navigation", 32)
    assert scheduler.state_dict()["tasks"]["navigation"]["stage"] == 0
    restored = _scheduler(json.loads(json.dumps(scheduler.state_dict())))
    assert restored.commit_cycle() == scheduler.commit_cycle()
    assert scheduler.state_dict()["tasks"]["navigation"]["stage"] == 1
    assert scheduler.state_dict()["tasks"]["mouse"]["stage"] == 0
    assert restored.select_episode("navigation") == scheduler.select_episode("navigation")
    with pytest.raises(ValueError, match="mismatch"):
        TaskCurriculumScheduler(config_hash="changed", admission_hash="admission",
                                 allowed_levels=scheduler.allowed_levels,
                                 state=scheduler.state_dict())


def test_rehearsal_every_fifth_round_robin_and_return_after_failure():
    scheduler = _scheduler()
    _outcomes(scheduler, "navigation", 32)
    scheduler.commit_cycle()
    _outcomes(scheduler, "navigation", 40)
    scheduler.commit_cycle()
    assert scheduler.state_dict()["tasks"]["navigation"]["stage"] == 2
    stages = []
    for _ in range(160):
        ticket = scheduler.select_episode("navigation")
        if ticket["rehearsal"]:
            stages.append(ticket["stage"])
        scheduler.record_outcome(ticket, not ticket["rehearsal"], str(ticket["level_index"]))
    assert stages[:4] == ["easy", "bridge", "easy", "bridge"]
    decisions = scheduler.commit_cycle()
    assert decisions[0]["reason"] == "rehearsal_failure"
    assert scheduler.state_dict()["tasks"]["navigation"]["stage"] == 0


def test_duplicate_runtime_wins_do_not_promote_and_tickets_cannot_repeat():
    scheduler = _scheduler()
    for _ in range(32):
        ticket = scheduler.select_episode("mouse")
        scheduler.record_outcome(ticket, True, "same-level")
    assert scheduler.commit_cycle() == []
    with pytest.raises(ValueError, match="already recorded"):
        scheduler.record_outcome(ticket, True, "same-level")


def test_admission_checks_actual_source_projections_uniqueness_and_staleness(tmp_path, monkeypatch):
    import carryrl.training.task_curriculum as module

    monkeypatch.setattr(module, "_probe_level", lambda *a, noop=False, **kw: {
        "success": not noop, "ticks": 3, "terminated": True, "truncated": False})
    manifest = prepare_task_curriculum(20261020)
    assert set(manifest["tasks"]) == set(TASKS)
    # A truthful insufficient-diversity result is a preflight failure.
    if not manifest["ready"]:
        assert manifest["errors"]
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(manifest))
    if manifest["ready"]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        validated = validate_task_curriculum_admission(
            path, level_bank_seed=20261020, expected_sha256=digest)
        allowed = curriculum_allowed_levels(validated)
        assert all(len(allowed[task][stage]) >= 16 for task in TASKS for stage in STAGES)
        manifest["tasks"]["navigation"]["easy"]["levels"][0]["runtime_id"] = "forged"
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="evidence"):
            validate_task_curriculum_admission(path, level_bank_seed=20261020)
    else:
        with pytest.raises(ValueError, match="unready"):
            validate_task_curriculum_admission(path, level_bank_seed=20261020)
    with pytest.raises(ValueError, match="SHA256"):
        validate_task_curriculum_admission(path, level_bank_seed=20261020,
                                           expected_sha256="changed")


def test_scheduler_thresholds_and_independent_demotion_window():
    scheduler = _scheduler()
    for index in range(32):
        ticket = scheduler.select_episode("memory")
        scheduler.record_outcome(ticket, index < 26, str(index % 16))
    assert scheduler.commit_cycle()[0]["to"] == "bridge"
    # Prior-stage rehearsal successes cannot repair the current bridge window.
    current = 0
    while current < 32:
        ticket = scheduler.select_episode("memory")
        scheduler.record_outcome(ticket, ticket["rehearsal"] or current < 8,
                                 str(ticket["level_index"]))
        current += not ticket["rehearsal"]
    decisions = scheduler.commit_cycle()
    assert decisions[0]["to"] == "easy"
    assert decisions[0]["reason"] == "current_failure"
    assert decisions[0]["wins"] == 8


def test_bridge_goal_probe_does_not_oscillate_when_player_occludes_goal_rim():
    from carryrl.training.task_curriculum import _probe_level

    for index in (0, 1, 5, 12):
        result = _probe_level("goal_choice", "bridge", index, level_bank_seed=20261020,
                              image_size=96, max_steps=80)
        assert result["success"]
        assert result["ticks"] <= 16


def test_impossible_safe_suffix_is_an_explicit_projection_failure():
    source = {"walls": [[0, 1], [1, 0]], "start": [0, 0], "goal": [4, 4]}
    with pytest.raises(ValueError, match="safe path"):
        resolve_effective_level("navigation", source, curriculum_stage="easy")
