"""Tests of the visual contract, lifecycle, splits and reward trust boundary."""

from collections import deque

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from carryrl.envs.navigation import ACTION_DELTAS, NavigationEnv, level_banks
from carryrl.rewards.navigation import (
    FLOOR,
    GOAL,
    PLAYER,
    WALL,
    WINNER,
    RecognitionError,
    cell_bounds,
    navigation_reward,
    recognize_navigation,
    shortest_distance,
)


def visible_path(frame):
    state = recognize_navigation(frame)
    queue = deque([(state.player, [])])
    visited = {state.player}
    while queue:
        position, actions = queue.popleft()
        if position == state.goal:
            return actions
        for action, (dr, dc) in enumerate(ACTION_DELTAS[1:], start=1):
            destination = position[0] + dr, position[1] + dc
            if (
                0 <= destination[0] < 5
                and 0 <= destination[1] < 5
                and destination not in state.walls
                and destination not in visited
            ):
                visited.add(destination)
                queue.append((destination, actions + [action]))
    raise AssertionError("Visible goal not reachable")


def test_gymnasium_contract_and_repeatable_seed():
    env = NavigationEnv(render_mode="rgb_array")
    check_env(env)
    first, first_info = env.reset(seed=42)
    second, second_info = env.reset(seed=42)
    assert np.array_equal(first, second)
    assert first_info == second_info
    assert first.shape == (64, 64, 3) and first.dtype == np.uint8
    assert set(first_info) == {
        "success",
        "distance",
        "map_id",
        "level_id",
        "split",
        "elapsed_ticks",
        "action_ticks",
        "tick_seconds",
        "recognition",
    }
    assert not first_info["success"]
    before = env.render()
    before[:] = 0
    assert np.array_equal(
        env.render(), first
    )  # Rendering/returned arrays cannot change simulation.
    np.testing.assert_array_equal(env.render(), env.render())
    assert env._elapsed_ticks == 0  # Rendering never advances simulation.


def test_disjoint_deterministic_connected_map_banks():
    banks = level_banks()
    assert [len(banks[split]) for split in ("train", "validation", "test")] == [64, 16, 16]
    all_maps = [walls for bank in banks.values() for walls in bank]
    assert len(set(all_maps)) == 96
    for walls in all_maps:
        cells = [(r, c) for r in range(5) for c in range(5) if (r, c) not in walls]
        assert all(shortest_distance(cells[0], cell, walls) >= 0 for cell in cells)
    map_ids, level_ids = {}, {}
    for split in banks:
        env = NavigationEnv(split=split)
        infos = [env.reset(seed=seed)[1] for seed in range(100)]
        map_ids[split] = {info["map_id"] for info in infos}
        level_ids[split] = {info["level_id"] for info in infos}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert map_ids[left].isdisjoint(map_ids[right])
        assert level_ids[left].isdisjoint(level_ids[right])


def test_screen_derived_rewards_and_terminal_outcome():
    env = NavigationEnv()
    frame, _ = env.reset(seed=2)
    actions = visible_path(frame)
    initial_distance = recognize_navigation(frame).distance
    reward_total = 0.0
    for index, action in enumerate(actions):
        next_frame, reward, terminated, truncated, info = env.step(action)
        expected_reward, visible = navigation_reward(frame, next_frame)
        assert reward == expected_reward
        assert info["success"] == visible.success
        assert info["distance"] == visible.distance
        assert not truncated
        assert terminated == (index == len(actions) - 1)
        reward_total += reward
        frame = next_frame
    assert reward_total == pytest.approx(1 + (0.04 - 0.01) * initial_distance)
    assert info["success"] and info["distance"] == 0
    with pytest.raises(RuntimeError, match="Episode ended"):
        env.step(0)


def test_reward_uses_pixels_even_if_simulation_and_screen_disagree(monkeypatch):
    env = NavigationEnv()
    frame, _ = env.reset(seed=3)
    action = visible_path(frame)[0]
    prior_player = env._player
    monkeypatch.setattr(env, "_draw_frame", lambda: frame.copy())
    _, reward, terminated, truncated, info = env.step(action)
    assert env._player != prior_player  # The simulator moved, but the screenshot did not.
    assert reward == -0.01
    assert info["distance"] == recognize_navigation(frame).distance
    assert not terminated and not truncated


def test_corrupt_unknown_and_ambiguous_frames_fail():
    frame, _ = NavigationEnv().reset(seed=5)
    state = recognize_navigation(frame)
    corrupt = frame.copy()
    y0, _, x0, _ = cell_bounds(64, 0, 0)
    corrupt[y0 + 1, x0 + 1] = (1, 2, 3)
    with pytest.raises(RecognitionError):
        recognize_navigation(corrupt)
    ambiguous = frame.copy()
    empty = next(
        (r, c)
        for r in range(5)
        for c in range(5)
        if (r, c) not in state.walls and (r, c) not in (state.player, state.goal)
    )
    y0, y1, x0, x1 = cell_bounds(64, *empty)
    ambiguous[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = PLAYER
    with pytest.raises(RecognitionError, match="one visible player"):
        recognize_navigation(ambiguous)
    for malformed in (frame.astype(np.float32), frame[:, :, 0], np.zeros_like(frame)):
        with pytest.raises(RecognitionError):
            recognize_navigation(malformed)
    distant = next(
        (r, c)
        for r in range(5)
        for c in range(5)
        if (r, c) not in state.walls
        and (r, c) != state.goal
        and abs(r - state.player[0]) + abs(c - state.player[1]) > 1
    )
    teleported = frame.copy()
    for position, color in ((state.player, FLOOR), (distant, PLAYER)):
        y0, y1, x0, x1 = cell_bounds(64, *position)
        teleported[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = color
    with pytest.raises(RecognitionError, match="farther than one"):
        navigation_reward(frame, teleported)


def test_recognition_failure_aborts_episode(monkeypatch):
    env = NavigationEnv()
    env.reset(seed=7)
    monkeypatch.setattr(env, "_draw_frame", lambda: np.zeros((64, 64, 3), np.uint8))
    with pytest.raises(RecognitionError):
        env.step(0)
    with pytest.raises(RuntimeError, match="Episode ended"):
        env.step(0)


def test_unreachable_or_ambiguous_victory_frames_fail():
    frame, _ = NavigationEnv().reset(seed=5)
    positions = {(0, 1): WALL, (1, 0): WALL, (0, 0): PLAYER, (4, 4): GOAL}
    for row in range(5):
        for column in range(5):
            y0, y1, x0, x1 = cell_bounds(64, row, column)
            frame[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = positions.get((row, column), FLOOR)
    with pytest.raises(RecognitionError, match="unreachable"):
        recognize_navigation(frame)
    y0, y1, x0, x1 = cell_bounds(64, 0, 0)
    frame[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = WINNER
    with pytest.raises(RecognitionError, match="Ambiguous visible victory"):
        recognize_navigation(frame)


def test_success_at_time_limit_is_terminated_not_truncated():
    frame, _ = NavigationEnv().reset(seed=2)
    actions = visible_path(frame)
    env = NavigationEnv(max_steps=len(actions))
    env.reset(seed=2)
    for action in actions:
        _, _, terminated, truncated, info = env.step(action)
    assert info["elapsed_ticks"] == len(actions)
    assert terminated and not truncated and info["success"]


def test_truncation_action_validation_and_reset_lifecycle():
    env = NavigationEnv(max_steps=1)
    with pytest.raises(RuntimeError, match="reset before"):
        env.step(0)
    env.reset(seed=1)
    for invalid in (-1, 5, 1.0, True, "up", np.array([1])):
        with pytest.raises(ValueError, match="integer action"):
            env.step(invalid)
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == -0.01 and not terminated and truncated
    assert info["elapsed_ticks"] == 1 and not info["success"]
    with pytest.raises(RuntimeError, match="Episode ended"):
        env.step(0)
    _, info = env.reset(seed=1)
    assert info["elapsed_ticks"] == 0
    env.step(0)
    for kwargs in ({"size": 32}, {"max_steps": 0}, {"split": "other"}, {"render_mode": "other"}):
        with pytest.raises(ValueError):
            NavigationEnv(**kwargs)
