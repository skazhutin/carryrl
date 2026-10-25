"""The SPEC5 diagnostic is scene RGB only, with teacher traces used just as fixtures."""

import inspect
import json

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic_interaction import GAMES, PLAYER, _box, make_controller
from carryrl.evaluation import scene_progress
from carryrl.evaluation.scene_progress import (
    EVENT_NAMES,
    GOALS,
    SUPPORTED_TASKS,
    SceneProgressAnnotator,
)
from carryrl.rewards.visual import HUD_HEIGHT


def _interaction():
    return GAMES["interaction"]({
        "start": [0, 0], "item": [0, 2], "use": [0, 4], "hazard": [4, 4], "walls": [],
    }, 96)


def _door(task="chain"):
    level = {
        "start": [2, 1], "door": [2, 2], "key": [0, 0], "switch": [4, 0],
        "exit": [2, 4], "hazard": [4, 1],
        "walls": [[0, 2], [1, 2], [3, 2], [4, 2]],
    }
    if task == "causality":
        level["button"] = level.pop("key")
    return GAMES[task](level, 96)


def _observe_step(game, annotator, *keys):
    game.step(PhysicalAction(keys=keys))
    result = annotator.observe(game.render())
    assert result["known"], result["reason"]
    return result


@pytest.mark.parametrize("task", SUPPORTED_TASKS)
@pytest.mark.parametrize("size", (64, 96, 128))
def test_real_teacher_traces_have_ordered_once_only_visible_milestones(task, size):
    for seed in (17, 281, 902, 4401):
        game = GAMES[task](GAMES[task].make_level(np.random.default_rng(seed)), size)
        teacher = make_controller(task)
        annotator = SceneProgressAnnotator(task)
        first = annotator.observe(game.render())
        assert first["known"], first
        assert not any(first["event_valid"].values())
        events, potentials = [], [first["potential"]]
        for _ in range(80):
            result = _observe_step(game, annotator, *teacher.act(game.render()).keys)
            json.dumps(result, allow_nan=False)
            events.extend(result["events"])
            potentials.append(result["potential"])
            assert all(result["event_valid"][event] for event in result["events"])
            assert all(not result["event_targets"][event] or result["event_valid"][event]
                       for event in EVENT_NAMES)
            if game.status != "Play":
                break
        assert game.status == "Success"
        expected = {
            "interaction": ["item_collected"],
            "causality": ["button_activated", "door_opened", "exit_reached"],
            "key_door": ["key_collected", "door_opened", "exit_reached"],
            "chain": ["key_collected", "switch_activated", "door_opened", "exit_reached"],
        }[task]
        assert events == expected
        assert len(events) == len(set(events))
        assert all(0 <= potential <= 1 for potential in potentials)
        if task == "interaction":
            assert result["terminal_success"] is None and not result["terminal_success_valid"]
            assert max(potentials) < 1
        else:
            assert result["terminal_success"] is True and result["terminal_success_valid"]
            assert result["potential"] == 1
        repeated = annotator.observe(game.render())
        assert repeated["known"] and repeated["events"] == []
        assert repeated["potential"] == result["potential"]


def test_hud_is_excluded_even_when_its_pixels_are_arbitrary():
    game = _interaction()
    original = SceneProgressAnnotator("interaction")
    modified = SceneProgressAnnotator("interaction")
    rng = np.random.default_rng(12)
    for keys in ((), ("d",), ("e",), (), ("d",), ("d",), ("e",)):
        game.step(PhysicalAction(keys=keys))
        frame = game.render()
        changed = frame.copy()
        changed[:HUD_HEIGHT] = rng.integers(0, 256, changed[:HUD_HEIGHT].shape, dtype=np.uint8)
        assert original.observe(frame) == modified.observe(changed)


def test_noop_retreat_cycles_and_interaction_invisible_use():
    game, annotator = _interaction(), SceneProgressAnnotator("interaction")
    initial = annotator.observe(game.render())
    assert _observe_step(game, annotator)["potential"] == initial["potential"]
    closer = _observe_step(game, annotator, "d")
    retreat = _observe_step(game, annotator, "a")
    assert closer["potential"] > retreat["potential"] == initial["potential"]
    assert _observe_step(game, annotator, "d")["potential"] == closer["potential"]
    collected = _observe_step(game, annotator, "e")
    assert collected["events"] == ["item_collected"]
    assert collected["visible_state"]["objects"]["item"] == [0, 2]
    assert _observe_step(game, annotator, "e")["events"] == []
    assert _observe_step(game, annotator, "d", "e")["events"] == []
    before_use = _observe_step(game, annotator, "d", "e")
    _observe_step(game, annotator)
    used = _observe_step(game, annotator, "e")
    assert game.status == "Success"
    assert used["potential"] == before_use["potential"] < 1
    assert used["terminal_success"] is None and not used["terminal_success_valid"]
    for name in EVENT_NAMES[1:]:
        assert not used["event_valid"][name]


def test_player_over_object_preserves_rim_and_does_not_collect():
    game, annotator = _interaction(), SceneProgressAnnotator("interaction")
    annotator.observe(game.render())
    _observe_step(game, annotator, "d")
    occupied = _observe_step(game, annotator, "d")
    assert occupied["events"] == []
    assert not occupied["visible_state"]["completed"]["item"]
    assert occupied["visible_state"]["objects"]["item"] == [0, 2]
    collected = _observe_step(game, annotator, "e")
    assert collected["events"] == ["item_collected"]
    assert collected["visible_state"]["objects"]["item"] == [0, 2]


@pytest.mark.parametrize("task", ("unknown", "navigation", "memory", "reaction"))
def test_unsupported_task_abstains_with_invalid_targets(task):
    result = SceneProgressAnnotator(task).observe(_interaction().render())
    assert not result["known"] and not result["potential_valid"]
    assert result["potential"] is None and result["terminal_success"] is None
    assert not any(result["event_valid"].values())
    assert not any(result["event_targets"].values())


@pytest.mark.parametrize("damage", ("pixel", "rim", "player", "duplicate_player", "background"))
def test_full_grid_pattern_validation_rejects_corruption(damage):
    frame = _interaction().render()
    if damage == "pixel":
        x0, y0, _, _ = _box(96, (3, 3))
        frame[y0 + 1, x0 + 1] = (1, 2, 3)
    elif damage == "rim":
        x0, y0, _, _ = _box(96, (0, 0))
        frame[y0, x0] = PLAYER
    elif damage == "player":
        x0, y0, x1, y1 = _box(96, (0, 0))
        frame[y0:y1, x0:x1] = frame[y0, x0]
    elif damage == "duplicate_player":
        x0, y0, x1, y1 = _box(96, (3, 3))
        frame[y0 + 1:y1 - 1, x0 + 1:x1 - 1] = PLAYER
    else:
        frame[-1, -1] = (1, 2, 3)
    annotator = SceneProgressAnnotator("interaction")
    result = annotator.observe(frame)
    assert not result["known"] and result["potential"] is None
    assert not any(result["event_valid"].values())
    assert annotator.observe(_interaction().render())["known"]


def test_unknown_gap_never_creates_an_event_on_resume():
    game, annotator = _interaction(), SceneProgressAnnotator("interaction")
    annotator.observe(game.render())
    _observe_step(game, annotator, "d")
    assert not annotator.observe(np.zeros((96, 96, 3), dtype=np.uint8))["known"]
    game.step(PhysicalAction(keys=("e",)))
    resumed = annotator.observe(game.render())
    assert resumed["known"] and resumed["visible_state"]["completed"]["item"]
    assert resumed["events"] == [] and not any(resumed["event_valid"].values())
    assert _observe_step(game, annotator)["event_valid"]["item_collected"]
    assert _observe_step(game, annotator)["events"] == []


def test_history_rejects_reverse_edges_and_topology_changes_even_after_gap():
    game, annotator = _interaction(), SceneProgressAnnotator("interaction")
    initial = game.render()
    annotator.observe(initial)
    _observe_step(game, annotator, "d")
    _observe_step(game, annotator, "e")
    assert not annotator.observe(initial)["known"]
    assert not annotator.observe(initial)["known"]
    assert annotator.observe(game.render())["known"]
    other = _interaction().render()
    x0, y0, x1, y1 = _box(96, (3, 3))
    other[y0:y1, x0:x1] = (58, 67, 80)
    topology = SceneProgressAnnotator("interaction")
    topology.observe(initial)
    assert not topology.observe(other)["known"]


def test_visible_change_far_from_required_target_is_invalid_not_negative():
    game, annotator = _interaction(), SceneProgressAnnotator("interaction")
    annotator.observe(game.render())
    changed = game.render()
    x0, y0, x1, y1 = _box(96, (0, 2))
    changed[y0:y1, x0:x1] = (187, 198, 212)
    result = annotator.observe(changed)
    assert not result["known"]
    assert not result["event_valid"]["item_collected"]


def test_missing_initial_object_cannot_be_inferred_from_player_or_floor():
    game = _interaction()
    game.step(PhysicalAction(keys=("d",)))
    game.step(PhysicalAction(keys=("e",)))
    result = SceneProgressAnnotator("interaction").observe(game.render())
    assert not result["known"] and "initial" in result["reason"]


def test_potential_uses_geodesic_path_around_walls_and_closed_door():
    game, annotator = _door("key_door"), SceneProgressAnnotator("key_door")
    first = annotator.observe(game.render())
    assert first["known"] and first["visible_state"]["distance"] == 2
    _observe_step(game, annotator, "w")
    _observe_step(game, annotator, "w")
    collected = _observe_step(game, annotator, "e")
    assert collected["visible_state"]["next_target"] == "door"
    assert collected["visible_state"]["distance"] == 2
    _observe_step(game, annotator, "s")
    adjacent = _observe_step(game, annotator, "s")
    assert adjacent["visible_state"]["distance"] == 0
    opened = _observe_step(game, annotator, "e")
    assert opened["events"] == ["door_opened"]
    assert opened["visible_state"]["distance"] == 3


def test_distance_follows_detour_instead_of_manhattan_shortcut():
    game = GAMES["interaction"]({
        "start": [0, 0], "item": [0, 4], "use": [4, 2], "hazard": [4, 4],
        "walls": [[0, 1], [1, 1], [2, 1]],
    }, 96)
    result = SceneProgressAnnotator("interaction").observe(game.render())
    assert result["known"]
    assert result["visible_state"]["distance"] == 9
    assert result["potential"] == (24 - 9) / 25 / 2


def test_unreachable_initial_layout_does_not_poison_next_valid_scene():
    game = GAMES["interaction"]({
        "start": [0, 0], "item": [4, 4], "use": [3, 4], "hazard": [4, 3],
        "walls": [[0, 1], [1, 0]],
    }, 96)
    annotator = SceneProgressAnnotator("interaction")
    result = annotator.observe(game.render())
    assert not result["known"] and result["potential"] is None
    assert annotator.observe(_interaction().render())["known"]


@pytest.mark.parametrize("size,dtype", ((80, np.uint8), (96, np.float32)))
def test_unsupported_geometry_and_non_uint8_frames_abstain(size, dtype):
    result = SceneProgressAnnotator("interaction").observe(np.zeros((size, size, 3), dtype=dtype))
    assert not result["known"] and not any(result["event_valid"].values())


@pytest.mark.parametrize("kind", ("switch", "door"))
def test_prerequisite_inconsistency_is_unknown_even_across_gap(kind):
    game = _door()
    annotator = SceneProgressAnnotator("chain")
    annotator.observe(game.render())
    annotator.observe(np.zeros((96, 96, 3), dtype=np.uint8))
    frame = game.render()
    x0, y0, x1, y1 = _box(96, tuple(game.level[kind]))
    frame[y0:y1, x0:x1] = (113, 147, 106) if kind == "switch" else (123, 165, 184)
    result = annotator.observe(frame)
    assert not result["known"] and result["events"] == []
    assert not any(result["event_valid"].values())


def test_diagnostic_source_scope_has_no_controller_actions_game_state_or_hud_verifier():
    source = inspect.getsource(scene_progress)
    for forbidden in ("make_controller", "PhysicalAction", "verify_visible_outcome", ".status",
                      ".level", ".has_key", ".carrying", ".score", "future_length"):
        assert forbidden not in source
    assert set(GOALS) == set(SUPPORTED_TASKS)
    assert len(EVENT_NAMES) == 6
