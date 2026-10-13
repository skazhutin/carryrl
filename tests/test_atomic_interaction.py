"""Behavioral checks for physical interaction and ordered RGB grid tasks."""

import json

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic_interaction import (
    GAMES,
    Causality,
    Chain,
    Interaction,
    KeyDoor,
    make_controller,
)
from carryrl.rewards.visual import HUD_HEIGHT


def _key(name):
    return PhysicalAction(keys=(name,))


def _door_fixture():
    return {
        "start": [2, 1], "door": [2, 2], "key": [0, 0], "switch": [4, 0],
        "exit": [2, 4], "hazard": [4, 1],
        "walls": [[0, 2], [1, 2], [3, 2], [4, 2]],
    }


@pytest.mark.parametrize("task", GAMES)
@pytest.mark.parametrize("image_size", (64, 96))
def test_rgb_only_controller_completes_varied_games(task, image_size):
    for seed in (17, 281, 902, 4401, 10008, 76021):
        game = GAMES[task](GAMES[task].make_level(np.random.default_rng(seed)), image_size)
        controller = make_controller(task)
        visited = {game.position}
        for _ in range(80):
            frame = game.render()
            assert frame.shape == (image_size, image_size, 3) and frame.dtype == np.uint8
            assert np.unique(frame[:HUD_HEIGHT].reshape(-1, 3), axis=0).shape[0] == 1
            game.step(controller.act(frame))
            visited.add(game.position)
            if game.status != "Play":
                break
        assert game.status == "Success", (task, seed, game.position, game.level)
        assert game.score == 5 and game.ticks <= 80
        assert len(visited) >= 2


@pytest.mark.parametrize("task", GAMES)
def test_visible_failure_probe_and_ended_game_guard(task):
    game = GAMES[task](GAMES[task].make_level(np.random.default_rng(610)), 64)
    controller = make_controller(task, fail=True)
    for _ in range(80):
        game.step(controller.act(game.render()))
        if game.status != "Play":
            break
    assert game.status == "Failed" and game.position == game.hazard
    with pytest.raises(RuntimeError, match="ended"):
        game.step(PhysicalAction())


@pytest.mark.parametrize("task", GAMES)
def test_level_generator_has_real_layout_variation_and_json_values(task):
    rng = np.random.default_rng(501)
    levels = [GAMES[task].make_level(rng) for _ in range(120)]
    serialized = [json.dumps(level, sort_keys=True) for level in levels]
    assert len(set(serialized)) >= 96
    assert len({tuple(level["start"]) for level in levels}) >= 8
    if task != "interaction":
        assert len({tuple(level["door"]) for level in levels}) >= 9
        # Key/button begins away from the locked passage: a trip and return are required.
        for level in levels:
            item = level.get("key", level.get("button"))
            start = level["start"]
            assert abs(item[0] - start[0]) + abs(item[1] - start[1]) >= 2


def test_interaction_requires_pickup_proximity_and_new_e_press():
    level = {
        "start": [0, 0], "item": [0, 1], "use": [0, 3], "hazard": [4, 4], "walls": [],
    }
    game = Interaction(level, 64)
    game.step(_key("e"))
    assert game.carrying and game.score == 2
    for _ in range(2):
        game.step(PhysicalAction(keys=("d", "e")))
    assert game.position == (0, 2) and game.status == "Play" and game.score == 2
    game.step(PhysicalAction())
    game.step(_key("e"))
    assert game.status == "Success" and game.score == 5

    distant = Interaction({**level, "start": [4, 0]}, 64)
    for _ in range(4):
        distant.step(_key("e"))
        distant.step(PhysicalAction())
    assert not distant.carrying and distant.score == 0


def test_using_pad_without_item_is_ineffective():
    game = Interaction({
        "start": [0, 2], "item": [4, 0], "use": [0, 3], "hazard": [4, 4], "walls": [],
    }, 96)
    game.step(_key("e"))
    assert not game.carrying and game.status == "Play" and game.score == 0


def test_locked_door_requires_key_and_return_to_door():
    game = KeyDoor(_door_fixture(), 96)
    game.step(_key("e"))
    game.step(_key("d"))
    assert not game.door_open and game.position == (2, 1) and game.score == 0
    for key in ("w", "w"):
        game.step(_key(key))
        assert game.score == 1  # Approaching the key is distinct from collecting it.
    game.step(_key("e"))
    assert game.has_key and game.score == 2 and not game.door_open
    game.step(PhysicalAction())
    game.step(_key("e"))
    assert not game.door_open and game.score == 2
    for key in ("s", "s"):
        game.step(_key(key))
    game.step(_key("e"))
    assert game.door_open and game.score == 3
    for _ in range(3):
        game.step(_key("d"))
    assert game.status == "Success" and game.score == 5


def test_remote_button_opens_door_once_and_does_not_farm_progress():
    level = _door_fixture()
    level["button"] = level.pop("key")
    game = Causality(level, 64)
    game.step(_key("e"))
    assert not game.door_open and game.score == 0
    for _ in range(2):
        game.step(_key("w"))
    game.step(_key("e"))
    assert game.door_open and game.score == 2
    for _ in range(5):
        game.step(PhysicalAction())
        game.step(_key("e"))
    assert game.score == 2 and game.status == "Play"


def test_chain_enforces_key_then_switch_then_door_then_exit():
    game = Chain(_door_fixture(), 96)
    for key in ("a", "s"):
        game.step(_key(key))
    game.step(_key("e"))
    assert not game.switch_on and not game.has_key and game.score == 1
    for _ in range(3):
        game.step(_key("w"))
    game.step(_key("e"))
    assert game.has_key and game.score == 2
    for key in ("d", "s", "s"):
        game.step(_key(key))
    game.step(_key("e"))
    assert not game.door_open and not game.switch_on and game.score == 2
    for key in ("a", "s"):
        game.step(_key(key))
    game.step(_key("e"))
    assert game.switch_on and not game.door_open and game.score == 3
    game.step(PhysicalAction())
    game.step(_key("e"))
    assert game.score == 3 and not game.door_open
    for key in ("w", "d"):
        game.step(_key(key))
    game.step(_key("e"))
    assert game.door_open and game.score == 4
    for _ in range(3):
        game.step(_key("d"))
    assert game.status == "Success" and game.score == 5


def test_held_movement_and_opposing_directions_have_physical_semantics():
    game = KeyDoor(_door_fixture(), 64)
    game.step(PhysicalAction(keys=("w", "s", "a", "d")))
    assert game.position == (2, 1)
    game.step(_key("w"))
    game.step(_key("w"))
    assert game.position == (0, 1) and game.ticks == 3
    with pytest.raises(TypeError, match="PhysicalAction"):
        game.step(0)


def test_controller_rejects_unrecognized_pixels_and_unknown_task():
    with pytest.raises(ValueError, match="Unknown"):
        make_controller("other")
    with pytest.raises(TypeError, match="uint8"):
        make_controller("interaction").act(np.zeros((64, 64, 3)))
    with pytest.raises(ValueError, match="Unrecognized"):
        make_controller("interaction").act(np.zeros((64, 64, 3), dtype=np.uint8))
