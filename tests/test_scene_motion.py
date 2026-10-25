"""Actual RGB traces audit diagnostic motion labels, never environment reward."""

import inspect
import json
from collections import Counter

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic_motion import (
    BACKGROUND,
    GAMES,
    HUD_HEIGHT,
    PLAYER,
    TARGET,
    _Canvas,
    make_controller,
)
from carryrl.evaluation import scene_motion
from carryrl.evaluation.scene_motion import (
    EVENT_NAMES,
    GOALS,
    SUPPORTED_TASKS,
    MotionSceneAnnotator,
)


def _game(task, size=96, seed=17):
    cls = GAMES[task]
    return cls(cls.make_level(np.random.default_rng(seed)), size)


@pytest.mark.parametrize("size", (64, 96, 128))
@pytest.mark.parametrize("task", SUPPORTED_TASKS)
def test_real_teacher_traces_are_json_and_partially_valid(task, size):
    for seed in (17, 281, 902, 4401):
        game, annotator, teacher = _game(task, size, seed), MotionSceneAnnotator(task), (
            make_controller(task)
        )
        events = Counter()
        known = 0
        for _ in range(200):
            result = annotator.observe(game.render())
            json.dumps(result, allow_nan=False)
            if result["known"]:
                known += 1
                if result["potential_valid"]:
                    assert 0 <= result["potential"] <= 1
                assert all(result["event_valid"][name] for name in result["events"])
                events.update(result["events"])
                repeated = annotator.observe(game.render(), PhysicalAction(keys=("space",)))
                assert repeated["known"]
                assert repeated["events"] == []
                assert repeated["potential"] == result["potential"]
            else:
                assert task == "timing" and result["reason"] == "no visible pipe geometry"
                assert not any(result["event_valid"].values())
            if task in {"timing", "steering", "camera"}:
                assert result["terminal_success"] is None
                assert not result["terminal_success_valid"]
            if game.status != "Play":
                break
            action = teacher.act(game.render())
            game.step(action)
            teacher.observe_action(action)
        assert known > 5
        assert game.status == "Success"
        if task == "timing":
            assert events["pipe_passed"] == 3
        elif task == "platforming":
            assert events["jump_started"] == 2
            assert events["obstacle_passed"] == 2
            assert events["landed"] >= 1
        elif task == "camera":
            assert events["door_aligned"] >= 1


@pytest.mark.parametrize("task", SUPPORTED_TASKS)
def test_hud_past_actions_and_external_counters_cannot_change_targets(task):
    game = _game(task)
    original, edited = MotionSceneAnnotator(task), MotionSceneAnnotator(task)
    rng = np.random.default_rng(99)
    teacher = make_controller(task)
    for _ in range(15):
        frame = game.render()
        changed = frame.copy()
        changed[:HUD_HEIGHT] = rng.integers(0, 256, changed[:HUD_HEIGHT].shape, dtype=np.uint8)
        assert original.observe(frame) == edited.observe(changed, PhysicalAction(keys=("space",)))
        action = teacher.act(frame)
        game.step(action)
        teacher.observe_action(action)


@pytest.mark.parametrize("task", SUPPORTED_TASKS)
def test_unknown_gap_breaks_transition_labels(task):
    game, annotator = _game(task), MotionSceneAnnotator(task)
    if task == "timing":
        for _ in range(3):
            game.step(PhysicalAction())
    assert annotator.observe(game.render())["known"]
    bad = game.render()
    bad[40, 0] = (1, 2, 3)
    unknown = annotator.observe(bad)
    assert not unknown["known"]
    assert unknown["potential"] is None
    assert not any(unknown["event_valid"].values())
    action = make_controller(task).act(game.render())
    game.step(action)
    recovered = annotator.observe(game.render())
    assert recovered["known"]
    assert recovered["events"] == []
    assert not any(recovered["event_valid"].values())


def test_platform_retreat_lowers_potential_and_hold_is_not_a_jump():
    game, annotator = _game("platforming"), MotionSceneAnnotator("platforming")
    initial = annotator.observe(game.render())
    game.step(PhysicalAction(keys=("right",)))
    forward = annotator.observe(game.render())
    game.step(PhysicalAction(keys=("left",)))
    retreat = annotator.observe(game.render())
    assert forward["potential"] > initial["potential"]
    assert retreat["potential"] == initial["potential"]
    game.step(PhysicalAction(keys=("space",)))
    assert annotator.observe(game.render())["events"] == ["jump_started"]
    for _ in range(25):
        game.step(PhysicalAction(keys=("space",)))
        assert "jump_started" not in annotator.observe(game.render())["events"]


def test_camera_backing_away_lowers_geometric_potential_and_saturation_is_unknown():
    cls = GAMES["camera"]
    game = cls({"distance": 4, "bearing": 0, "yaw_error": 0, "pitch": 0}, 96)
    annotator = MotionSceneAnnotator("camera")
    before = annotator.observe(game.render())
    for _ in range(4):
        game.step(PhysicalAction(keys=("s",)))
    after = annotator.observe(game.render())
    assert after["potential"] < before["potential"]
    for distance in (0.5, 1.0, 1.5):
        close = cls({"distance": distance, "bearing": 0, "yaw_error": 0, "pitch": 0}, 96)
        result = MotionSceneAnnotator("camera").observe(close.render())
        assert result["known"]
        assert result["terminal_success"] is None
        assert not result["terminal_success_valid"]


def test_steering_alignment_deterioration_is_not_time_progress():
    game = GAMES["steering"]({"centers": [48] * 5, "width": 26, "speed": 1.2}, 96)
    annotator = MotionSceneAnnotator("steering")
    before = annotator.observe(game.render())
    for _ in range(5):
        game.step(PhysicalAction())
        assert annotator.observe(game.render())["potential"] == before["potential"]
    game.step(PhysicalAction(keys=("left",)))
    game.step(PhysicalAction(keys=("left",)))
    assert annotator.observe(game.render())["potential"] < before["potential"]


def test_generic_blobs_and_incomplete_geometry_are_not_supported_scenes():
    canvas = _Canvas(96)
    canvas.ellipse((19, 31, 25, 37), PLAYER)
    frame = canvas.frame()
    for task in SUPPORTED_TASKS:
        assert not MotionSceneAnnotator(task).observe(frame)["known"]
    canvas.rectangle((45, 29, 51, 35), TARGET)
    assert not MotionSceneAnnotator("camera").observe(canvas.frame())["known"]
    blank = np.full((96, 96, 3), BACKGROUND, dtype=np.uint8)
    assert not MotionSceneAnnotator("platforming").observe(blank)["known"]


@pytest.mark.parametrize("frame", (np.zeros((80, 80, 3), dtype=np.uint8),
                                   np.zeros((96, 96, 3), dtype=float), None))
def test_unsupported_frames_abstain(frame):
    assert not MotionSceneAnnotator("timing").observe(frame)["known"]


def test_public_contract_has_no_privileged_targets_or_controller_calls():
    assert set(GOALS) == set(SUPPORTED_TASKS)
    assert len(EVENT_NAMES) == len(set(EVENT_NAMES))
    source = inspect.getsource(scene_motion)
    for forbidden in (".score", ".status", ".step(", ".act(", "make_controller", "game.reward"):
        assert forbidden not in source
