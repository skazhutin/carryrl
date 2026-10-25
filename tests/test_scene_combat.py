"""RGB diagnostic fixtures; reference controllers are not learned policies."""

import json

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic_combat import GAMES, HUD_HEIGHT, UIGame, make_controller
from carryrl.evaluation.scene_combat import EVENT_NAMES, SUPPORTED_TASKS, CombatSceneAnnotator


def _game(task, size=96, seed=17):
    return GAMES[task](GAMES[task].make_level(np.random.default_rng(seed)), size)


def _step(game, annotator, action=None):
    action = PhysicalAction() if action is None else action
    game.step(action)
    return annotator.observe(game.render(), previous_action=action)


@pytest.mark.parametrize("task", SUPPORTED_TASKS)
@pytest.mark.parametrize("size", (64, 96, 128))
def test_real_rgb_teacher_completion_transitions(task, size):
    game, annotator = _game(task, size), CombatSceneAnnotator(task)
    teacher = make_controller(task)
    initial = annotator.observe(game.render())
    assert initial["known"], initial
    assert not any(initial["event_valid"].values())
    events = []
    for _ in range(80):
        result = _step(game, annotator, teacher.act(game.render()))
        assert result["known"], result
        json.dumps(result, allow_nan=False)
        if result["potential_valid"]:
            assert 0 <= result["potential"] <= 1
        events.extend(result["events"])
        assert all(result["event_valid"][name] for name in result["events"])
        if game.status != "Play":
            break
    assert game.status == "Success"
    # RGB64 contact can straddle the seven-unit boundary after quantization.
    # Abstention is preferable to inventing a success from an ambiguous image.
    if task == "goal_choice" and size == 64:
        assert result["terminal_success"] is None
        assert not result["terminal_success_valid"]
    else:
        assert result["terminal_success"] is True
        assert result["terminal_success_valid"] and result["potential"] == 1
    if task == "shooting":
        assert events == ["enemy_damaged"] * 3
    elif task == "ui":
        assert events == ["inventory_opened", "item_selected", "item_confirmed", "inventory_closed"]
    elif task == "hazards":
        assert events == ["exit_reached"]
    repeated = annotator.observe(game.render())
    assert repeated["events"] == []
    assert repeated["potential"] == result["potential"]


@pytest.mark.parametrize("task", SUPPORTED_TASKS)
@pytest.mark.parametrize("size", (64, 96, 128))
def test_hud_invariance_and_unsupported_geometry(task, size):
    game = _game(task, size)
    clean, dirty = CombatSceneAnnotator(task), CombatSceneAnnotator(task)
    teacher, rng = make_controller(task), np.random.default_rng(19)
    for _ in range(10):
        frame = game.render()
        other = frame.copy()
        other[:HUD_HEIGHT] = rng.integers(0, 256, other[:HUD_HEIGHT].shape, dtype=np.uint8)
        assert clean.observe(frame) == dirty.observe(other)
        game.step(teacher.act(frame))
        if game.status != "Play":
            break
    result = clean.observe(np.zeros((80, 80, 3), dtype=np.uint8))
    assert not result["known"] and not result["potential_valid"]
    assert result["potential"] is None
    assert not any(result["event_valid"].values())
    assert not any(result["event_targets"].values())


@pytest.mark.parametrize("task", SUPPORTED_TASKS)
def test_corrupt_pixels_abstain_and_gap_invalidates_next_edge(task):
    game, annotator = _game(task), CombatSceneAnnotator(task)
    annotator.observe(game.render())
    damaged = game.render()
    damaged[-1, -1] = (1, 2, 3)
    assert not annotator.observe(damaged)["known"]
    game.step(make_controller(task).act(game.render()))
    resumed = annotator.observe(game.render())
    assert resumed["known"], resumed
    assert resumed["events"] == []
    assert not any(resumed["event_valid"].values())
    assert set(resumed["event_targets"]) == set(EVENT_NAMES)


def test_hazards_restores_occluded_trap_and_goal_from_history():
    game, annotator = _game("hazards"), CombatSceneAnnotator("hazards")
    initial = annotator.observe(game.render())
    fail_teacher = make_controller("hazards", fail=True)
    for _ in range(30):
        result = _step(game, annotator, fail_teacher.act(game.render()))
        assert result["known"], result
        if game.status != "Play":
            break
    assert game.status == "Failed"
    assert result["visible_state"]["failed"]
    assert result["potential"] == 0 and result["events"] == []
    assert result["visible_state"]["traps"] == initial["visible_state"]["traps"]


def test_shooting_misses_are_edges_and_holding_fire_is_not_another_shot():
    game, annotator = _game("shooting"), CombatSceneAnnotator("shooting")
    initial = annotator.observe(game.render())
    fire = PhysicalAction(mouse_buttons=("left",))
    missed = _step(game, annotator, fire)
    assert missed["events"] == ["shot_missed"]
    assert missed["potential"] == initial["potential"] == 0
    held = _step(game, annotator, fire)
    assert held["events"] == [] and held["visible_state"]["misses"] == 1
    for _ in range(5):
        _step(game, annotator)
        result = _step(game, annotator, fire)
    assert game.status == "Failed" and result["visible_state"]["misses"] == 6
    assert result["terminal_success"] is False


def test_shooting_missing_health_under_cursor_does_not_invent_damage():
    game, annotator = _game("shooting"), CombatSceneAnnotator("shooting")
    annotator.observe(game.render())
    # Rendering fixture deliberately covers one health pip with the real cursor.
    ex, ey = game._point(*game.enemy)
    game.cursor = [ex - 4, ey - max(3, game.size / 18) - 2]
    result = annotator.observe(game.render())
    assert not result["known"]
    assert result["potential"] is None and result["events"] == []


def test_ui_selection_regresses_and_tab_hold_does_not_repeat_open_bonus():
    game = UIGame({"request": 1, "items": [0, 1, 2], "selection": 0,
                   "layout": 0, "request_x": 32, "menu_offset": 0}, 96)
    annotator = CombatSceneAnnotator("ui")
    closed = annotator.observe(game.render())
    assert closed["potential"] is None and not closed["terminal_success_valid"]
    opened = _step(game, annotator, PhysicalAction(keys=("tab",)))
    assert opened["events"] == ["inventory_opened"]
    held = _step(game, annotator, PhysicalAction(keys=("tab",)))
    assert held["events"] == [] and held["potential"] == opened["potential"]
    correct = _step(game, annotator, PhysicalAction(keys=("right",)))
    assert correct["events"] == ["item_selected"]
    wrong = _step(game, annotator, PhysicalAction(keys=("left",)))
    assert wrong["potential"] < correct["potential"] and wrong["events"] == []
    failed = _step(game, annotator, PhysicalAction(keys=("enter",)))
    assert game.status == "Failed"
    # Wrong Enter has no world marker, so the parser does not fabricate failure.
    assert failed["events"] == [] and not failed["terminal_success"]


def test_ui_close_across_gap_cannot_certify_confirmation_history():
    game, annotator = _game("ui"), CombatSceneAnnotator("ui")
    teacher = make_controller("ui")
    annotator.observe(game.render())
    for _ in range(20):
        result = _step(game, annotator, teacher.act(game.render()))
        if result["visible_state"]["confirmed"]:
            break
    assert result["visible_state"]["confirmed"]
    annotator.observe(np.zeros((96, 96, 3), dtype=np.uint8))
    closed = _step(game, annotator, PhysicalAction(keys=("escape",)))
    assert closed["known"] and not closed["terminal_success_valid"]
    assert closed["potential"] is None and closed["events"] == []


def test_goal_proximity_regresses_on_retreat_without_invented_payoff_weights():
    game, annotator = _game("goal_choice"), CombatSceneAnnotator("goal_choice")
    initial = annotator.observe(game.render())
    closer = _step(game, annotator, PhysicalAction(keys=("w",)))
    retreat = _step(game, annotator, PhysicalAction(keys=("s",)))
    assert closer["potential"] > retreat["potential"] == initial["potential"]
    assert retreat["visible_state"]["utility"] == "completion_only"


@pytest.mark.parametrize("size", (64, 96, 128))
@pytest.mark.parametrize("seed", (281, 902, 4401))
def test_goal_layers_reconstruct_outline_hidden_by_target_and_guard(size, seed):
    game, annotator = _game("goal_choice", size, seed), CombatSceneAnnotator("goal_choice")
    teacher = make_controller("goal_choice")
    for _ in range(60):
        result = annotator.observe(game.render())
        assert result["known"], result
        if game.status != "Play":
            break
        game.step(teacher.act(game.render()))
    assert game.status == "Success" and result["terminal_success"] is True


@pytest.mark.parametrize("size", (64, 96, 128))
@pytest.mark.parametrize("layout", (0, 1))
def test_ui_four_item_layouts_have_visible_confirmation(size, layout):
    game = UIGame({"request": 3, "items": [2, 0, 3, 1], "selection": 0,
                   "layout": layout, "request_x": 64, "menu_offset": 2}, size)
    annotator, teacher = CombatSceneAnnotator("ui"), make_controller("ui")
    for _ in range(30):
        result = annotator.observe(game.render())
        assert result["known"], result
        if game.status != "Play":
            break
        game.step(teacher.act(game.render()))
    assert game.status == "Success" and result["terminal_success"] is True


def test_shooting_multiple_damage_cues_cannot_be_a_single_shot_edge():
    game, annotator = _game("shooting"), CombatSceneAnnotator("shooting")
    annotator.observe(game.render())
    game.hits = 2
    result = annotator.observe(game.render())
    assert not result["known"] and result["events"] == []
    # Resumption can recover visible damage, without an invented edge over gap.
    resumed = annotator.observe(game.render())
    assert resumed["known"] and resumed["events"] == []
    assert not resumed["event_valid"]["enemy_damaged"]
