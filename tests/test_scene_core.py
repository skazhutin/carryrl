"""Real RGB teacher traces and executed counterexamples for core diagnostics."""

import json

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv
from carryrl.evaluation.scene_core import (
    EVENT_NAMES,
    GOALS,
    SUPPORTED_TASKS,
    CoreSceneAnnotator,
)
from carryrl.rewards.visual import HUD_HEIGHT
from carryrl.training.imitation import scripted_teacher


def _episode(task, size=96, seed=17, *, fail=False):
    env = AtomicEnv(task, image_size=size, max_steps=110)
    frame, _ = env.reset(seed=seed)
    teacher = scripted_teacher(task)
    yield frame, None
    for _ in range(110):
        action = teacher.act(frame)
        if fail:
            if task == "reaction":
                rows, columns = np.where(np.all(frame == (250, 245, 218), axis=-1))
                action = PhysicalAction(keys=("left" if columns.mean() > size/2 else "right",))
            elif task == "memory" and action.keys:
                action = PhysicalAction(keys=("right" if "left" in action.keys else "left",))
            else:
                action = PhysicalAction()
        frame, _, terminated, truncated, _ = env.step(action)
        teacher.observe_action(action)
        yield frame, action
        if terminated or truncated:
            break


@pytest.mark.parametrize("size", (64, 96, 128))
@pytest.mark.parametrize("task", SUPPORTED_TASKS)
def test_real_rgb_teachers_and_hud_replacement(task, size):
    assert task in GOALS
    for seed in (17, 281, 902, 4401):
        annotator, replaced = CoreSceneAnnotator(task), CoreSceneAnnotator(task)
        events = []
        for frame, action in _episode(task, size, seed):
            result = annotator.observe(frame, action)
            assert result["known"], result
            assert 0 <= result["potential"] <= 1
            assert all(result["event_valid"][name] for name in result["events"])
            assert all(result["event_targets"][name] == (name in result["events"])
                       for name in EVENT_NAMES)
            json.dumps(result, allow_nan=False)
            modified = frame.copy()
            modified[:HUD_HEIGHT] = np.random.default_rng(7).integers(
                0, 256, modified[:HUD_HEIGHT].shape, dtype=np.uint8)
            assert replaced.observe(modified, action) == result
            events.extend(result["events"])
        expected = {"navigation": ["exit_reached"], "reaction": ["ball_caught"]*3,
                    "mouse": ["target_clicked"]*3,
                    "memory": ["cue_disappeared", "memory_unlocked", "exit_reached"]}[task]
        assert events == expected
        if task == "mouse":
            assert result["terminal_success"] is None
            assert not result["terminal_success_valid"]
        else:
            assert result["terminal_success"] is True
            assert result["potential"] == 1
        repeated = annotator.observe(frame, action)
        assert repeated["known"] and not repeated["events"]
        assert repeated["potential"] == result["potential"]


def test_navigation_geodesic_can_regress_and_goal_is_retained_under_player():
    env = AtomicEnv("navigation")
    frame, _ = env.reset(seed=17)
    annotator = CoreSceneAnnotator("navigation")
    initial = annotator.observe(frame)
    noop, *_ = env.step(PhysicalAction())
    assert annotator.observe(noop, PhysicalAction())["potential"] == initial["potential"]
    action = scripted_teacher("navigation").act(noop)
    frame, *_ = env.step(action)
    closer = annotator.observe(frame, action)
    opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
    reverse = PhysicalAction(keys=(opposite[action.keys[0]],))
    frame, *_ = env.step(reverse)
    farther = annotator.observe(frame, reverse)
    assert closer["potential"] > initial["potential"] == farther["potential"]
    teacher = scripted_teacher("navigation")
    while not farther["terminal_success"]:
        action = teacher.act(frame)
        frame, *_ = env.step(action)
        farther = annotator.observe(frame, action)
        assert farther["known"]
    assert farther["visible_state"]["goal"] == farther["visible_state"]["player"]
    # A fresh parser cannot recover an occluded destination without history.
    assert not CoreSceneAnnotator("navigation").observe(frame)["known"]


@pytest.mark.parametrize("task", ("reaction", "memory"))
@pytest.mark.parametrize("size", (64, 96, 128))
def test_executed_failure_commands_never_invent_a_win(task, size):
    annotator = CoreSceneAnnotator(task)
    events = []
    for frame, action in _episode(task, size, fail=True):
        result = annotator.observe(frame, action)
        assert result["known"], result
        events.extend(result["events"])
    assert result["terminal_success"] is False
    assert result["potential"] < 1
    assert "exit_reached" not in events and "ball_caught" not in events


def test_memory_waiting_is_flat_and_identical_suffix_needs_prior_cue():
    full, blind = CoreSceneAnnotator("memory"), CoreSceneAnnotator("memory")
    waiting = []
    blind_started = False
    for frame, action in _episode("memory"):
        result = full.observe(frame, action)
        assert result["known"]
        if result["visible_state"]["phase"] == "Wait":
            waiting.append(result["potential"])
        if not result["visible_state"]["cue_visible"]:
            blind_started = True
            unknown = blind.observe(frame, action)
            assert not unknown["known"] and unknown["potential"] is None
            assert "cue" in unknown["reason"]
    assert blind_started and len(waiting) > 60
    assert len(set(waiting)) == 1
    assert result["terminal_success"]


def test_mouse_noop_miss_hold_release_and_final_unchanged_click():
    env = AtomicEnv("mouse", max_steps=110, difficulty=.5)
    frame, _ = env.reset(seed=17)
    annotator = CoreSceneAnnotator("mouse")
    initial = annotator.observe(frame)
    noop = PhysicalAction()
    frame, *_ = env.step(noop)
    result = annotator.observe(frame, noop)
    assert result["potential"] == initial["potential"] and not result["events"]
    miss = PhysicalAction(mouse_buttons=("left",))
    frame, *_ = env.step(miss)
    result = annotator.observe(frame, miss)
    assert result["known"] and not result["events"]
    teacher = scripted_teacher("mouse")
    teacher.observe_action(miss)
    events = []
    previous_target = result["visible_state"]["target"]
    for _ in range(50):
        action = teacher.act(frame)
        frame, _, terminated, truncated, _ = env.step(action)
        result = annotator.observe(frame, action)
        assert result["known"], result
        events.extend(result["events"])
        if terminated or truncated:
            break
        previous_target = result["visible_state"]["target"]
    # At difficulty .5 two targets are required; final rendering selects the next
    # target, so terminal status cannot be inferred from completion count.
    assert events == ["target_clicked"]*2
    assert result["terminal_success"] is None
    assert previous_target != result["visible_state"]["target"]


def test_mouse_actionless_last_click_is_unknown_and_holding_does_not_repeat():
    trace = list(_episode("mouse"))
    annotator, without_actions = CoreSceneAnnotator("mouse"), CoreSceneAnnotator("mouse")
    for frame, action in trace:
        result = annotator.observe(frame, action)
        actionless = without_actions.observe(frame)
    assert result["events"] == ["target_clicked"]
    assert actionless["events"] == []
    assert not actionless["event_valid"]["target_clicked"]
    assert not annotator.observe(frame, action)["events"]
    # Omitting a command breaks held-button continuity rather than inventing a release.
    annotator.observe(frame)
    assert not annotator.observe(frame, action)["event_valid"]["target_clicked"]


@pytest.mark.parametrize("task", SUPPORTED_TASKS)
def test_corruption_and_unsupported_geometry_abstain(task):
    frame, _ = next(_episode(task))
    corrupted = frame.copy()
    corrupted[HUD_HEIGHT+1, 1] = (1, 2, 3)
    annotator = CoreSceneAnnotator(task)
    result = annotator.observe(corrupted)
    assert not result["known"] and result["potential"] is None
    assert not any(result["event_valid"].values())
    assert not annotator.observe(np.zeros((80, 80, 3), dtype=np.uint8))["known"]
    assert not annotator.observe(frame.astype(np.float32))["known"]
    assert not CoreSceneAnnotator("missing").observe(frame)["known"]


def test_mouse_final_unchanged_target_acceptance_rim_is_not_a_valid_negative():
    env = AtomicEnv("mouse", image_size=96, max_steps=110, level_bank_seed=20261008)
    frame, _ = env.reset(seed=17, options={"level_index": 0})
    teacher = scripted_teacher("mouse")
    annotator = CoreSceneAnnotator("mouse")
    result = annotator.observe(frame)
    hits = 0
    while hits < 2:
        action = teacher.act(frame)
        frame, _, terminated, truncated, _ = env.step(action)
        assert not terminated and not truncated
        result = annotator.observe(frame, action)
        assert result["known"], result
        hits += "target_clicked" in result["events"]
    # Position on the final visible target while explicitly releasing the button.
    while result["visible_state"]["distance"]:
        recommended = teacher.act(frame)
        action = PhysicalAction(dx=recommended.dx, dy=recommended.dy)
        frame, _, terminated, truncated, _ = env.step(action)
        assert not terminated and not truncated
        teacher.observe_action(action)
        result = annotator.observe(frame, action)
        assert result["known"], result
    target = result["visible_state"]["target"]
    offset = PhysicalAction(dx=8 if target[0]+8 <= 92 else -8)
    frame, *_ = env.step(offset)
    result = annotator.observe(frame, offset)
    assert result["visible_state"]["distance"] == 8
    before = frame[HUD_HEIGHT:].copy()
    click = PhysicalAction(mouse_buttons=("left",))
    frame, _, terminated, truncated, info = env.step(click)
    assert terminated and not truncated and info["success"]
    assert np.array_equal(before, frame[HUD_HEIGHT:])
    result = annotator.observe(frame, click)
    assert result["known"]
    assert result["visible_state"]["target"] == target
    # Subpixel acceptance is uncertain from RGB, so this real successful rim
    # click must never become a valid negative event training target.
    assert result["event_targets"]["target_clicked"] or not result["event_valid"]["target_clicked"]
