"""Physical-control and pixel-reference checks for the combat/UI exercises."""

import json

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic_combat import (
    GAMES,
    GoalChoiceGame,
    HazardsGame,
    ShootingGame,
    UIGame,
    make_controller,
)


@pytest.mark.parametrize("task", GAMES)
@pytest.mark.parametrize("size", (64, 96))
@pytest.mark.parametrize("fail", (False, True))
def test_pixel_controllers_reach_opposite_outcomes_on_unseen_seeds(task, size, fail):
    # Controllers see only rendered RGB; independent test seeds are never fitted.
    for seed in (12_807, 43_021, 67_591, 91_837, 150_091, 290_011, 920_075, 920_858):
        kind = GAMES[task]
        game = kind(kind.make_level(np.random.default_rng(seed)), size)
        controller = make_controller(task, fail=fail)
        for _ in range(80):
            frame = game.render()
            assert frame.shape == (size, size, 3) and frame.dtype == np.uint8
            game.step(controller.act(frame.copy()))
            if game.status != "Play":
                break
        assert game.status == ("Failed" if fail else "Success")
        assert 0 <= game.score <= 5


@pytest.mark.parametrize("task", GAMES)
def test_noop_does_not_solve_an_exercise_and_levels_have_real_variation(task):
    kind = GAMES[task]
    rng = np.random.default_rng(7128)
    levels = [kind.make_level(rng) for _ in range(1000)]
    assert len({json.dumps(level, sort_keys=True) for level in levels}) >= 96
    game = kind(levels[0], 64)
    before = game.render()
    game.step(PhysicalAction())
    assert game.status == "Play" and game.score == 0
    assert game.ticks == 1 and game.render().shape == before.shape
    for _ in range(20):
        game.step(PhysicalAction())
    assert game.status == "Play" and game.score == 0
    with pytest.raises(TypeError):
        game.step(1)


def _choice_level():
    return {
        "safe_side": 0, "safe_value": 2, "risky_value": 5, "goal_y": 16,
        "guard_y": 22, "guard_direction": 1, "guard_speed": 1,
    }


@pytest.mark.parametrize("safe_value", (1, 2))
@pytest.mark.parametrize("risky_value", (3, 4, 5))
def test_choice_payoffs_and_guard_risk_are_different_actual_outcomes(safe_value, risky_value):
    level = {**_choice_level(), "safe_value": safe_value, "risky_value": risky_value}
    safe = GoalChoiceGame(level, 96)
    controller = make_controller("goal_choice")
    while safe.status == "Play":
        safe.step(controller.act(safe.render()))
    assert safe.status == "Success" and safe.score == safe_value
    risky = GoalChoiceGame(level, 96)
    # A longer route around the guard can obtain the greater payoff.
    for _ in range(11):
        risky.step(PhysicalAction(keys=("d",)))
    for _ in range(12):
        risky.step(PhysicalAction(keys=("w",)))
    for _ in range(3):
        if risky.status == "Play":
            risky.step(PhysicalAction(keys=("a",)))
    assert risky.status == "Success" and risky.score == risky_value
    unsafe = GoalChoiceGame(level, 96)
    failure = make_controller("goal_choice", fail=True)
    for _ in range(80):
        unsafe.step(failure.act(unsafe.render()))
        if unsafe.status != "Play":
            break
    assert unsafe.status == "Failed" and unsafe.score == 1  # Earlier approach is retained.


def test_hazards_move_even_without_player_input_and_traps_fail_on_contact():
    level = {
        "traps": [[3, 2], [1, 1], [1, 2], [1, 3]], "start": [4, 2],
        "goal": [0, 4], "patrol": [[2, 1], [2, 2]], "enemy_index": 0, "enemy_period": 1,
    }
    waiting = HazardsGame(level, 96)
    first = waiting.render()
    waiting.step(PhysicalAction())
    assert not np.array_equal(first, waiting.render())
    assert waiting.position == (4, 2) and waiting.status == "Play"
    contact = HazardsGame(level, 96)
    contact.step(PhysicalAction(keys=("w",)))
    assert contact.status == "Failed"
    opposing = HazardsGame(level, 96)
    opposing.step(PhysicalAction(keys=("w", "s", "a", "d")))
    assert opposing.position == (4, 2)


def test_hazards_have_no_universal_safe_column_or_top_corridor():
    rng = np.random.default_rng(81293)
    levels = [HazardsGame.make_level(rng) for _ in range(200)]
    for column in range(5):
        assert any(any(cell[1] == column for cell in level["traps"]) for level in levels)
    assert any(any(cell[0] == 0 for cell in level["traps"]) for level in levels)
    level = {
        "traps": [[3, 0], [1, 1], [1, 2], [1, 3]], "start": [4, 3],
        "goal": [0, 4], "patrol": [[2, 2], [2, 3]], "enemy_index": 0, "enemy_period": 1,
    }
    fixed_corridor = HazardsGame(level, 64)
    for _ in range(3):
        fixed_corridor.step(PhysicalAction(keys=("a",)))
    fixed_corridor.step(PhysicalAction(keys=("w",)))
    assert fixed_corridor.status == "Failed"
    adaptive = HazardsGame(level, 64)
    controller = make_controller("hazards")
    for _ in range(80):
        adaptive.step(controller.act(adaptive.render()))
        if adaptive.status != "Play":
            break
    assert adaptive.status == "Success"


def _shooting_level():
    return {
        "player_x": 16, "player_y": 60, "enemy_x": 80, "enemy_y": 20,
        "cursor_x": 16, "cursor_y": 60,
    }


def test_shooting_requires_movement_aim_and_button_release_between_shots():
    game = ShootingGame(_shooting_level(), 96)
    # Aim alone cannot hit a target beyond firing distance.
    game.step(PhysicalAction(mouse_buttons=("left",), dx=64, dy=-32))
    assert game.hits == 0 and game.misses == 1
    for _ in range(3):
        game.step(PhysicalAction(mouse_buttons=("left",)))
    assert game.misses == 1  # A held button is not another button-down event.
    game.step(PhysicalAction(dy=-8))
    for _ in range(10):
        game.step(PhysicalAction(keys=("w", "d")))
    for _ in range(4):
        game.step(PhysicalAction(keys=("d",)))
    assert game.position != [16.0, 60.0] and game.hits == 0
    for hit in range(3):
        if hit:
            game.step(PhysicalAction())
        game.step(PhysicalAction(mouse_buttons=("left",)))
    assert game.status == "Success" and game.score == 5


def test_ui_enforces_open_select_confirm_close_and_key_edges():
    level = {
        "request": 2, "items": [0, 1, 2], "selection": 0,
        "layout": 0, "request_x": 48, "menu_offset": 0,
    }
    game = UIGame(level, 64)
    game.step(PhysicalAction(keys=("enter", "escape")))
    assert not game.opened and game.status == "Play"
    game.step(PhysicalAction(keys=("tab",)))
    game.step(PhysicalAction(keys=("tab",)))
    assert game.opened  # Holding Tab does not toggle the menu every tick.
    game.step(PhysicalAction(keys=("right",)))
    game.step(PhysicalAction(keys=("right",)))
    game.step(PhysicalAction(keys=("enter",)))
    assert game.confirmed and game.score == 3 and game.status == "Play"
    game.step(PhysicalAction(keys=("enter",)))
    assert game.status == "Play"
    game.step(PhysicalAction(keys=("escape",)))
    assert game.status == "Success" and game.score == 5
    wrong = UIGame(level, 64)
    wrong.step(PhysicalAction(keys=("tab",)))
    wrong.step(PhysicalAction(keys=("enter",)))
    assert wrong.status == "Failed" and wrong.score == 1  # Opening was already completed.
    early_close = UIGame(level, 64)
    early_close.step(PhysicalAction(keys=("tab",)))
    early_close.step(PhysicalAction(keys=("escape",)))
    assert early_close.status == "Play" and not early_close.opened


def test_unknown_reference_controller_is_rejected():
    with pytest.raises(ValueError):
        make_controller("missing")
