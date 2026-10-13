"""Behavioral contracts for the physical-input controlled game suite."""

import copy

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import (
    PLAYER,
    TARGET,
    TASKS,
    AtomicEnv,
    _centroid,
    _identity,
    atomic_level_banks,
    collect_bootstrap_dataset,
)
from carryrl.rewards.visual import verify_visible_outcome


@pytest.mark.parametrize("task", TASKS)
def test_atomic_rgb_seed_split_and_lifecycle(task):
    env = AtomicEnv(task, max_steps=1)
    with pytest.raises(RuntimeError):
        env.step(PhysicalAction())
    frame, info = env.reset(seed=7)
    repeated, repeated_info = env.reset(seed=7)
    np.testing.assert_array_equal(frame, repeated)
    assert info == repeated_info and env.observation_space.contains(frame)
    assert info["known"] and not info["success"] and not info["failure"]
    with pytest.raises(TypeError):
        env.step(0)
    _, _, terminated, truncated, info = env.step(PhysicalAction())
    assert not terminated and truncated and info["elapsed_ticks"] == 1
    with pytest.raises(RuntimeError):
        env.step(PhysicalAction())
    env.reset(seed=7)
    env.close()
    with pytest.raises(RuntimeError):
        env.step(PhysicalAction())
    banks = atomic_level_banks(task)
    identifiers = {split: {_identity(level) for level in bank} for split, bank in banks.items()}
    assert [len(identifiers[split]) for split in ("train", "validation", "test")] == [64, 16, 16]
    assert identifiers["train"].isdisjoint(identifiers["validation"])
    assert identifiers["train"].isdisjoint(identifiers["test"])
    assert identifiers["validation"].isdisjoint(identifiers["test"])


def test_navigation_physical_keys_cancel_and_mouse_does_not_move_player():
    env = AtomicEnv("navigation")
    initial, _ = env.reset(seed=8)
    unchanged, _, _, _, _ = env.step(PhysicalAction(keys=("left", "right"), dx=64))
    np.testing.assert_array_equal(initial, unchanged)
    # An actual arrow is a one-cell physical movement, not move-toward-target.
    position = _centroid(initial, PLAYER)
    moved = False
    for key in ("up", "down", "left", "right"):
        env.reset(seed=8)
        frame, _, _, _, _ = env.step(PhysicalAction(keys=(key,)))
        next_position = _centroid(frame, PLAYER)
        if position != next_position:
            moved = True
            assert (position[0] == next_position[0]) != (position[1] == next_position[1])
    assert moved


def test_mouse_relative_motion_clamp_click_edge_and_failed_final_frame():
    env = AtomicEnv("mouse")
    frame, _ = env.reset(seed=4)
    cursor = _centroid(frame, PLAYER)
    frame, _, terminated, _, _ = env.step(PhysicalAction(dx=2, dy=-1))
    next_cursor = _centroid(frame, PLAYER)
    assert not terminated
    assert next_cursor[0] == pytest.approx(cursor[0] + 2)
    assert next_cursor[1] == pytest.approx(cursor[1] - 1)
    frame, _, _, _, _ = env.step(PhysicalAction(dx=-64, dy=-64))
    frame, _, _, _, _ = env.step(PhysicalAction(dx=-64, dy=-64))
    assert _centroid(frame, PLAYER)[0] <= 4
    final, _, terminated, truncated, info = env.step(PhysicalAction(mouse_buttons=("left",)))
    assert terminated and not truncated and info["failure"]
    assert verify_visible_outcome(final)["failure"]
    with pytest.raises(RuntimeError):
        env.step(PhysicalAction())
    # A held button cannot generate multiple click edges.
    env.reset(seed=4)
    env._cursor = list(env._mouse_location(env._level["targets"][0]))
    env.step(PhysicalAction(mouse_buttons=("left",)))
    env._cursor = list(env._mouse_location(env._level["targets"][1]))
    env.step(PhysicalAction(mouse_buttons=("left",)))
    assert env._score == 2  # First hit is retained; holding cannot hit the next target.
    env.step(PhysicalAction())
    env.step(PhysicalAction(mouse_buttons=("left",)))
    assert env._score == 3


def test_memory_cue_disappears_and_choice_requires_history():
    left, right = AtomicEnv("memory"), AtomicEnv("memory")
    left_frame, _ = left.reset(seed=0)
    right.reset(seed=0)
    right._level = copy.deepcopy(right._level)
    right._level["cue"] = 1 - left._level["cue"]
    right._frame = right._draw()
    assert not np.array_equal(left_frame, right.render())
    assert _centroid(left_frame, TARGET) is not None
    for _ in range(5):
        left_frame, *_ = left.step(PhysicalAction())
        right_frame, *_ = right.step(PhysicalAction())
    np.testing.assert_array_equal(left_frame, right_frame)
    assert _centroid(left_frame, TARGET) is None
    for _ in range(80):
        _, _, done_left, _, info_left = left.step(PhysicalAction(keys=("left",)))
        _, _, done_right, _, info_right = right.step(PhysicalAction(keys=("left",)))
        if done_left:
            break
    assert done_left and done_right
    assert info_left["elapsed_ticks"] > 64
    assert info_left["success"] != info_right["success"]


def test_diagnostic_reward_is_visible_and_unknown_hud_aborts(monkeypatch):
    env = AtomicEnv("mouse")
    initial, _ = env.reset(seed=4)
    env._cursor = list(env._mouse_location(env._level["targets"][0]))
    monkeypatch.setattr(env, "_draw", lambda: initial.copy())
    _, reward, terminated, _, info = env.step(PhysicalAction(mouse_buttons=("left",)))
    assert env._score == 2  # Hidden engine changed; visible HUD did not.
    assert reward == -0.01 and info["score"] == 0 and not terminated
    monkeypatch.setattr(env, "_draw", lambda: np.zeros_like(initial))
    with pytest.raises(RuntimeError, match="unknown"):
        env.step(PhysicalAction())
    with pytest.raises(RuntimeError):
        env.step(PhysicalAction())


def test_bootstrap_whole_episodes_labels_provenance_and_validation_separation():
    train_frames, train_labels, train = collect_bootstrap_dataset(episodes_per_task=2)
    validation_frames, _, validation = collect_bootstrap_dataset(
        episodes_per_task=2, split="validation"
    )
    assert train_frames.shape[1:] == validation_frames.shape[1:] == (96, 96, 3)
    assert set(train["frame_episode_ids"]).isdisjoint(validation["frame_episode_ids"])
    assert {episode["level_id"] for episode in train["episodes"]}.isdisjoint(
        {episode["level_id"] for episode in validation["episodes"]}
    )
    assert train["label_counts"]["success"] >= 4
    assert train["label_counts"]["failure"] >= 2
    assert train["score_counts"]["1"] > 0 and train["score_counts"]["2"] > 0
    for frame, index in zip(train_frames, range(len(train_frames)), strict=True):
        visible = verify_visible_outcome(frame)
        assert visible["known"]
        for name in train_labels:
            assert train_labels[name][index] == float(visible[name])
    assert train["not_autonomous_reward_discovery"]
