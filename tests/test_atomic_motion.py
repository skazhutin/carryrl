"""Behavioral checks for real timing, jumping, inertial steering, and camera control."""

import json
import math

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic_motion import (
    GAMES,
    HUD_HEIGHT,
    PLAYER,
    CameraGame,
    PlatformingGame,
    SteeringGame,
    TimingGame,
    make_controller,
)


def _run(game, controller):
    for _ in range(80):
        action = controller.act(game.render().copy())
        assert isinstance(action, PhysicalAction)
        game.step(action)
        if game.status != "Play":
            break
    return game.status


@pytest.mark.parametrize("task", GAMES)
@pytest.mark.parametrize("image_size", (64, 96))
def test_pixel_controllers_solve_varied_levels_within_eighty_ticks(task, image_size):
    rng = np.random.default_rng(1203)
    for _ in range(24):
        game = GAMES[task](GAMES[task].make_level(rng), image_size)
        frame = game.render()
        assert frame.shape == (image_size, image_size, 3)
        assert frame.dtype == np.uint8
        # Common adapter, not a game-specific renderer, owns score and outcome HUD.
        assert np.all(frame[:HUD_HEIGHT] == (31, 38, 52))
        assert _run(game, make_controller(task)) == "Success"
        assert game.score == 5


@pytest.mark.parametrize("task", GAMES)
def test_wrong_physical_controls_fail_instead_of_creating_progress(task):
    cls = GAMES[task]
    game = cls(cls.make_level(np.random.default_rng(33)), 96)
    assert _run(game, make_controller(task, fail=True)) == "Failed"
    assert game.score < 5
    ended_state = game.render()
    ended_ticks = game.ticks
    game.step(PhysicalAction(keys=("space", "w"), dx=32))
    np.testing.assert_array_equal(ended_state, game.render())
    assert ended_ticks == game.ticks


@pytest.mark.parametrize("task", GAMES)
def test_noop_does_not_finish_a_level_or_end_initial_tick(task):
    cls = GAMES[task]
    game = cls(cls.make_level(np.random.default_rng(77)), 96)
    game.step(PhysicalAction())
    assert game.status == "Play"
    for _ in range(79):
        game.step(PhysicalAction())
    assert game.status != "Success"


@pytest.mark.parametrize("task", GAMES)
def test_generator_can_supply_ninety_six_distinct_semantic_levels(task):
    rng = np.random.default_rng(42)
    levels = set()
    for _ in range(1000):
        levels.add(json.dumps(GAMES[task].make_level(rng), sort_keys=True))
        if len(levels) >= 96:
            break
    assert len(levels) >= 96


def test_flap_edges_change_velocity_and_gravity_continues_between_flaps():
    level = TimingGame.make_level(np.random.default_rng(2))
    game = TimingGame(level, 96)
    initial_y = game.y
    game.step(PhysicalAction())
    assert game.y > initial_y
    game.step(PhysicalAction(keys=("space",)))
    impulse = game.velocity
    assert impulse < 0
    game.step(PhysicalAction(keys=("space",)))
    assert game.velocity > impulse  # Holding is not another flap.
    game.step(PhysicalAction())
    game.step(PhysicalAction(keys=("space",)))
    assert game.velocity == pytest.approx(impulse)


def test_platform_jump_requires_ground_and_release_before_next_jump():
    game = PlatformingGame(PlatformingGame.make_level(np.random.default_rng(2)), 96)
    initial_x = game.x
    game.step(PhysicalAction(keys=("space", "left", "right")))
    assert game.x == initial_x
    assert game.y < 53
    for _ in range(20):
        game.step(PhysicalAction(keys=("space",)))
    assert game.y == 53
    game.step(PhysicalAction(keys=("space",)))
    assert game.y == 53
    game.step(PhysicalAction())
    game.step(PhysicalAction(keys=("space",)))
    assert game.y < 53


def test_steering_has_momentum_after_key_release_and_braking_reverses_it():
    game = SteeringGame(SteeringGame.make_level(np.random.default_rng(2)), 96)
    game.step(PhysicalAction(keys=("right",)))
    first_x, first_velocity = game.x, game.velocity
    game.step(PhysicalAction())
    assert game.x > first_x
    assert 0 < game.velocity < first_velocity
    game.step(PhysicalAction(keys=("left",)))
    assert game.velocity < 0


def test_camera_rotation_and_translation_are_independent_and_can_be_combined():
    level = CameraGame.make_level(np.random.default_rng(12))
    rotate, translate, combined = (CameraGame(level, 96) for _ in range(3))
    initial_yaw = rotate.yaw
    initial_frame = rotate.render()
    rotate.step(PhysicalAction(dx=8, dy=4))
    np.testing.assert_array_equal(rotate.position, np.zeros(2))
    assert rotate.yaw != initial_yaw
    assert not np.array_equal(initial_frame, rotate.render())
    translate.step(PhysicalAction(keys=("w",)))
    assert translate.yaw == pytest.approx(initial_yaw)
    assert np.linalg.norm(translate.position) == pytest.approx(0.25)
    combined.step(PhysicalAction(keys=("w",), dx=8, dy=4))
    assert combined.yaw == rotate.yaw
    assert combined.pitch == rotate.pitch
    assert not np.allclose(combined.position, translate.position)


def test_camera_arrival_requires_visible_alignment_and_mouse_can_supply_it():
    level = CameraGame.make_level(np.random.default_rng(12))
    game = CameraGame(level, 96)
    game.position = game.goal - np.array((0.5, 0.0))
    game.yaw, game.pitch = 0.5, 0.2
    game.step(PhysicalAction())
    assert game.status == "Play"  # Nearby doorway alone is insufficient.
    assert not np.any(np.all(game.render()[HUD_HEIGHT:] == PLAYER, axis=-1))
    game.step(PhysicalAction(dx=-32, dy=-8))
    assert np.any(np.all(game.render()[HUD_HEIGHT:] == PLAYER, axis=-1))
    assert game.status == "Success"


def test_camera_generated_levels_cannot_finish_with_any_keyboard_only_alignment():
    rng = np.random.default_rng(240)
    for _ in range(24):
        level = CameraGame.make_level(rng)
        assert abs(level["yaw_error"]) > 0.12
        assert abs(level["pitch"]) > 0.08
        game = CameraGame(level, 96)
        controller = make_controller("camera")
        for _ in range(80):
            desired = controller.act(game.render())
            game.step(PhysicalAction(keys=desired.keys))
            if game.status != "Play":
                break
        assert game.status != "Success"
        assert game.pitch == level["pitch"]
        # Even perfect keyboard navigation to a point in front of the door cannot
        # satisfy the physical camera pitch requirement without mouse movement.
        game = CameraGame(level, 96)
        game.position = game.goal - 0.5 * np.array((math.cos(game.yaw), math.sin(game.yaw)))
        game.step(PhysicalAction())
        assert game.status == "Play"


def test_mouse_alignment_alone_cannot_supply_camera_translation():
    game = CameraGame(CameraGame.make_level(np.random.default_rng(17)), 64)
    controller = make_controller("camera")
    for _ in range(80):
        desired = controller.act(game.render())
        game.step(PhysicalAction(dx=desired.dx, dy=desired.dy))
    np.testing.assert_array_equal(game.position, np.zeros(2))
    assert game.status == "Play"
    assert game.score == 1  # One-time alignment earns progress, never arrival.


def test_unknown_motion_controller_rejected():
    with pytest.raises(ValueError, match="Unknown motion task"):
        make_controller("unknown")
