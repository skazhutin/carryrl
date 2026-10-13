"""Visible, cumulative milestone rewards and protection against repeated progress."""

import copy

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import TASKS, AtomicEnv, atomic_level_banks, collect_bootstrap_dataset
from carryrl.envs.atomic_combat import HazardsGame, ShootingGame
from carryrl.envs.atomic_interaction import GAMES
from carryrl.envs.atomic_motion import CameraGame, PlatformingGame, TimingGame, make_controller
from carryrl.rewards.visual import verify_visible_outcome


def _step(env, **kwargs):
    frame, reward, terminated, truncated, info = env.step(PhysicalAction(**kwargs))
    assert verify_visible_outcome(frame)["score"] == info["score"]
    return info["score"], reward, terminated or truncated


@pytest.mark.parametrize("size", (64, 96, 128))
@pytest.mark.parametrize("difficulty", (0.0, 0.15, 1.0))
def test_all_tasks_have_progress_success_and_reset_across_sizes_and_stages(size, difficulty):
    # Legacy canonical 128px reaction uses more ticks; the varied presentation
    # separately scales its ball speed to preserve the 80-tick episode budget.
    frames, labels, provenance = collect_bootstrap_dataset(
        episodes_per_task=2, image_size=size, difficulty=difficulty, max_steps=160,
    )
    assert provenance["suite_version"] == "carryrl.atomic.v3"
    ids = np.asarray(provenance["frame_episode_ids"])
    for episode in provenance["episodes"]:
        scores = labels["score"][ids == episode["episode_id"]]
        assert scores[0] == 0 and np.all(np.diff(scores) >= 0), episode["task"]
    successes = [episode for episode in provenance["episodes"] if "success probe" in
                 episode["strategy"]]
    assert {episode["task"] for episode in successes} == set(TASKS)
    for episode in successes:
        episode_frames = frames[ids == episode["episode_id"]]
        scores = labels["score"][ids == episode["episode_id"]]
        outcomes = [verify_visible_outcome(frame) for frame in episode_frames]
        assert episode["success"], episode["task"]
        assert scores[0] == 0 and np.all(np.diff(scores) >= 0), episode["task"]
        if difficulty == 1:
            assert any(0 < outcome["score"] < 5 and not outcome["success"]
                       for outcome in outcomes), episode["task"]
        env = AtomicEnv(episode["task"], image_size=size, difficulty=difficulty)
        _, info = env.reset(seed=episode["seed"])
        assert info["score"] == 0


def test_inventory_first_tab_is_visible_and_holds_reopens_and_wrong_selection_cannot_farm():
    bank = atomic_level_banks("ui")["train"]
    index = next(i for i, level in enumerate(bank)
                 if level["items"][level["selection"]] != level["request"])
    env = AtomicEnv("ui")
    env.reset(options={"level_index": index})
    assert _step(env, keys=("tab",))[:2] == pytest.approx((1, 0.99))
    for _ in range(3):
        assert _step(env, keys=("tab",))[:2] == pytest.approx((1, -0.01))
    for _ in range(3):
        _step(env)
        assert _step(env, keys=("tab",))[:2] == pytest.approx((1, -0.01))
        _step(env)
        assert _step(env, keys=("tab",))[:2] == pytest.approx((1, -0.01))
    # Reach the other wrong item first, then select the requested item.
    game = env._game
    correct = game.level["items"].index(game.level["request"])
    wrong = next(i for i in range(3) if i not in (correct, game.selection))
    move = "right" if game.level["layout"] == 0 else "down"
    reverse = "left" if game.level["layout"] == 0 else "up"
    direction = move if (game.selection + 1) % 3 == wrong else reverse
    assert _step(env, keys=(direction,))[:2] == pytest.approx((1, -0.01))
    direction = move if (wrong + 1) % 3 == correct else reverse
    assert _step(env, keys=(direction,))[:2] == pytest.approx((2, 0.99))
    for _ in range(2):
        for _ in range(3):
            assert _step(env, keys=(move,))[:2] == pytest.approx((2, -0.01))
    assert _step(env, keys=("enter",))[:2] == pytest.approx((3, 0.99))
    assert _step(env, keys=("enter",))[:2] == pytest.approx((3, -0.01))
    score, _, done = _step(env, keys=("escape",))
    assert score == 5 and done
    _, info = env.reset(options={"level_index": index})
    assert info["score"] == 0
    _step(env, keys=("tab",))
    score, reward, done = _step(env, keys=("enter",))
    assert score == 1 and reward < 0 and done


def test_memory_movement_milestone_cannot_reveal_either_answer_after_cue():
    left, right = AtomicEnv("memory"), AtomicEnv("memory")
    left.reset(seed=0)
    right.reset(seed=0)
    right._level = copy.deepcopy(right._level)
    right._level["cue"] = 1 - left._level["cue"]
    right._frame = right._draw()
    for _ in range(left._level["unlock"]):
        left_frame, _, left_done, _, left_info = left.step(PhysicalAction(keys=("right",)))
        right_frame, _, right_done, _, right_info = right.step(PhysicalAction(keys=("right",)))
        if left._ticks > left._level["cue_ticks"]:
            np.testing.assert_array_equal(left_frame, right_frame)
        assert not left_done and not right_done
    assert left_info["score"] == right_info["score"] == 1
    for env in (left, right):
        assert _step(env, keys=("left",))[:2] == pytest.approx((1, -0.01))
        assert _step(env, keys=("right",))[:2] == pytest.approx((1, -0.01))


@pytest.mark.parametrize("task", GAMES)
def test_approach_is_one_time_and_wrong_interaction_does_not_add_progress(task):
    level = {
        "start": [4, 0], "item": [0, 0], "button": [0, 0], "key": [0, 0],
        "use": [0, 4], "switch": [4, 4], "door": [2, 2], "exit": [0, 4],
        "hazard": [4, 3], "walls": [],
    }
    game = GAMES[task](level, 96)
    for _ in range(3):
        game.step(PhysicalAction(keys=("w",)))
        assert game.score == 1
        game.step(PhysicalAction(keys=("s",)))
        assert game.score == 1
        game.step(PhysicalAction(keys=("e",)))
        assert game.score == 1 and game.status == "Play"


def test_platform_and_camera_keep_best_progress_after_retreat():
    platform = PlatformingGame({"start_x": 8, "goal_x": 40, "obstacles": []}, 96)
    for _ in range(3):
        platform.step(PhysicalAction(keys=("d",)))
    assert platform.score == 1
    for _ in range(3):
        platform.step(PhysicalAction(keys=("a",)))
        assert platform.score == 1
    for _ in range(3):
        platform.step(PhysicalAction(keys=("d",)))
        assert platform.score == 1
    camera = CameraGame({"distance": 6, "bearing": 0, "yaw_error": 0,
                         "pitch": 0.2}, 96)
    for _ in range(6):
        camera.step(PhysicalAction(keys=("w",)))
    assert camera.score == 1
    for _ in range(6):
        camera.step(PhysicalAction(keys=("s",)))
        assert camera.score == 1
    controller = make_controller("camera")
    for _ in range(15):
        action = controller.act(camera.render())
        camera.step(PhysicalAction(dx=action.dx, dy=action.dy))
    assert camera.score == 2 and camera.status == "Play"


def test_navigation_and_hazards_cannot_farm_the_same_distance_reduction():
    navigation = AtomicEnv("navigation")
    navigation.reset(seed=0)
    navigation._position, navigation._goal = (4, 0), (0, 0)
    navigation._walls, navigation._initial_distance = frozenset(), 4
    navigation._frame = navigation._draw()
    assert _step(navigation, keys=("w",))[:2] == pytest.approx((1, 0.99))
    for _ in range(3):
        assert _step(navigation, keys=("s",))[:2] == pytest.approx((1, -0.01))
        assert _step(navigation, keys=("w",))[:2] == pytest.approx((1, -0.01))
    hazards = HazardsGame({"start": [4, 0], "goal": [0, 0], "traps": [[4, 4]],
                          "patrol": [[2, 3], [2, 4]], "enemy_index": 0,
                          "enemy_period": 2}, 96)
    hazards.step(PhysicalAction(keys=("w",)))
    assert hazards.score == 1
    for _ in range(3):
        hazards.step(PhysicalAction(keys=("s",)))
        assert hazards.score == 1
        hazards.step(PhysicalAction(keys=("w",)))
        assert hazards.score == 1 and hazards.status == "Play"


def test_timing_preparation_is_controlled_once_and_failure_has_no_new_progress():
    game = TimingGame({"start_y": 34, "pipe_x": 100, "spacing": 45, "gaps": [34] * 3,
                       "gap_height": 36, "speed": 3}, 96)
    game.step(PhysicalAction())
    assert game.score == 0
    game.step(PhysicalAction(keys=("space",)))
    assert game.score == 1
    for _ in range(3):
        game.step(PhysicalAction(keys=("space",)))
        assert game.score == 1
    hazard = HazardsGame({"start": [4, 0], "goal": [0, 0], "traps": [[3, 0]],
                          "patrol": [[2, 3], [2, 4]], "enemy_index": 0,
                          "enemy_period": 2}, 96)
    hazard.step(PhysicalAction(keys=("w",)))
    assert hazard.status == "Failed" and hazard.score == 0


def test_shooting_rewards_approach_aim_and_hits_without_farming_held_fire():
    game = ShootingGame({"player_x": 16, "player_y": 60, "enemy_x": 80,
                         "enemy_y": 20, "cursor_x": 16, "cursor_y": 60}, 96)
    for _ in range(4):
        game.step(PhysicalAction(keys=("w", "d")))
    assert game.score == 1 and game.hits == 0
    game.step(PhysicalAction(dx=64, dy=-32))
    game.step(PhysicalAction(dy=-8))
    assert game.score == 2 and game.hits == 0
    for _ in range(6):
        game.step(PhysicalAction(keys=("w", "d")))
    for _ in range(4):
        game.step(PhysicalAction(keys=("d",)))
    game.step(PhysicalAction(mouse_buttons=("left",)))
    assert game.score == 3 and game.hits == 1
    for _ in range(3):
        game.step(PhysicalAction(mouse_buttons=("left",)))
        assert game.score == 3 and game.hits == 1
    game.step(PhysicalAction())
    game.step(PhysicalAction(mouse_buttons=("left",)))
    assert game.score == 4 and game.hits == 2
