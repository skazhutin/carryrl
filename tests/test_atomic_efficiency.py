"""Perfect visible scores require useful physical inputs, independently of wins."""

import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs import atomic
from carryrl.envs.atomic import AtomicEnv, _unnecessary_input_units
from carryrl.envs.atomic_suite import SCORING_VERSION, TASKS
from carryrl.rewards.visual import verify_visible_outcome


def _maze(monkeypatch):
    # The direct four-cell route is blocked; every shortest route takes six
    # cardinal moves. This also checks the baseline uses this level's walls.
    level = {"start": (4, 0), "goal": (0, 0), "walls": [(3, 0), (2, 2)]}
    monkeypatch.setattr(atomic, "atomic_level_banks", lambda *args: {"train": (level,)})
    env = AtomicEnv("navigation")
    env.reset(options={"level_index": 0})
    return env


def _complete(env, actions):
    total_reward = 0
    for action in actions:
        frame, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    assert terminated and not truncated and info["success"]
    assert verify_visible_outcome(frame)["score"] == info["score"]
    return info, total_reward


_ROUTE = tuple(PhysicalAction(keys=(key,)) for key in
               ("right", "up", "up", "up", "up", "left"))


def test_only_shortest_route_gets_perfect_score_and_reset_clears_cost(monkeypatch):
    env = _maze(monkeypatch)
    info, clean_reward = _complete(env, _ROUTE)
    assert info["score"] == 5 and info["completion_efficiency"] == 1
    assert info["optimal_action_units"] == info["action_units"] == 6
    assert info["quality"]["excess_command_units"] == 0
    assert info["quality"]["command_units"] == 6
    assert info["quality"]["game_seconds"] == pytest.approx(0.6)
    assert info["scoring_version"] == SCORING_VERSION
    env.reset(options={"level_index": 0})
    detour = (PhysicalAction(keys=("right",)), PhysicalAction(keys=("left",)))
    info, detour_reward = _complete(env, detour + _ROUTE)
    assert info["score"] == 4 and info["completion_efficiency"] == pytest.approx(6 / 8)
    assert info["quality"]["excess_command_units"] == 2
    assert detour_reward < clean_reward
    _, reset_info = env.reset(options={"level_index": 0})
    assert reset_info["action_units"] == reset_info["unnecessary_input_units"] == 0
    assert reset_info["completion_efficiency"] is None
    assert reset_info["quality"]["commands"] == reset_info["quality"]["input_units"] == 0
    assert _complete(env, _ROUTE)[0]["score"] == 5


@pytest.mark.parametrize("extra", (
    PhysicalAction(),                         # Unnecessary idle tick.
    PhysicalAction(keys=("up",)),             # Blocked by the visible wall.
    PhysicalAction(keys=("left", "right")),  # Opposing directions cancel.
    PhysicalAction(keys=("tab",)),
    PhysicalAction(mouse_buttons=("left",)),
    PhysicalAction(dx=1),
    PhysicalAction(dy=1),
    PhysicalAction(wheel=1),
))
def test_wasted_actions_keep_success_but_cannot_get_perfect_score(monkeypatch, extra):
    env = _maze(monkeypatch)
    info, _ = _complete(env, (extra,) + _ROUTE)
    assert info["score"] == 4 and info["completion_efficiency"] < 1


@pytest.mark.parametrize("first", (
    PhysicalAction(keys=("right", "d")),
    PhysicalAction(keys=("right", "space")),
    PhysicalAction(keys=("right",), dx=1),
))
def test_extra_input_on_useful_movement_is_still_unnecessary(monkeypatch, first):
    info, _ = _complete(_maze(monkeypatch), (first,) + _ROUTE[1:])
    assert info["score"] == 4
    assert info["unnecessary_input_units"] == 1
    assert info["completion_efficiency"] == pytest.approx(6 / 7)


def test_diagonal_ticks_are_compared_in_cardinal_movement_units(monkeypatch):
    level = {"start": (2, 0), "goal": (0, 2), "walls": []}
    monkeypatch.setattr(atomic, "atomic_level_banks", lambda *args: {"train": (level,)})
    env = AtomicEnv("navigation")
    env.reset()
    info, _ = _complete(env, (PhysicalAction(keys=("up", "right")),) * 2)
    assert info["score"] == 5 and info["elapsed_ticks"] == 2
    assert info["action_units"] == info["optimal_action_units"] == 4


@pytest.mark.parametrize("extra", (False, True))
def test_required_memory_waits_preserve_perfect_score_but_irrelevant_keys_do_not(extra):
    env = AtomicEnv("memory", difficulty=0)
    env.reset(seed=4)
    for _ in range(env._level["unlock"] - 1):
        _, _, terminated, _, info = env.step(PhysicalAction())
        assert not terminated and info["unnecessary_input_units"] == 0
    key = "left" if env._level["cue"] == 0 else "right"
    actions = [PhysicalAction(keys=(key, "tab") if extra else (key,))]
    frame, _, terminated, _, info = env.step(actions[0])
    while not terminated:
        frame, _, terminated, _, info = env.step(PhysicalAction(keys=(key,)))
    assert info["success"] and info["score"] == (4 if extra else 5)
    assert verify_visible_outcome(frame)["score"] == info["score"]
    assert info["completion_efficiency"] is None
    assert info["quality"]["optimal_command_units"] is None
    assert info["quality"]["completion_efficiency"] is None
    assert info["quality"]["neutral_commands"] == env._level["unlock"] - 1
    assert info["quality"]["commands"] == info["elapsed_ticks"]


@pytest.mark.parametrize("task", TASKS)
def test_task_input_audit_allows_release_wait_and_rejects_unused_wheel(task):
    assert _unnecessary_input_units(task, PhysicalAction(), 0) == 0
    assert _unnecessary_input_units(task, PhysicalAction(wheel=1), 0) == 1


@pytest.mark.parametrize("task, action, layout", (
    ("timing", PhysicalAction(keys=("space",)), None),
    ("platforming", PhysicalAction(keys=("right", "space")), None),
    ("camera", PhysicalAction(keys=("w",), dx=2, dy=-1), None),
    ("shooting", PhysicalAction(mouse_buttons=("left",), dx=2), None),
    ("mouse", PhysicalAction(mouse_buttons=("left",), dy=-1), None),
    ("interaction", PhysicalAction(keys=("e",)), None),
    ("ui", PhysicalAction(keys=("tab", "right", "enter", "escape")), 0),
    ("ui", PhysicalAction(keys=("down",)), 1),
))
def test_required_controls_are_not_classified_as_irrelevant(task, action, layout):
    assert _unnecessary_input_units(task, action, layout) == 0


def test_goal_payoff_is_separate_from_irrelevant_input_score_cap():
    env = AtomicEnv("goal_choice", level_bank_seed=20261008)
    env.reset(seed=1)
    # Place the actor just before claiming a five-unit goal. This isolates the
    # scoring wrapper from route planning and the moving guard's risk.
    env._game.level["risky_value"] = 5
    env._game.position = [env._game._risky_x, float(env._game.level["goal_y"])]
    _, _, terminated, truncated, info = env.step(PhysicalAction(wheel=1))
    assert terminated and not truncated and info["success"]
    assert info["score"] == 4
    assert info["quality"]["terminal_payoff"] == 5
    assert info["quality"]["unnecessary_input_units"] == 1
    assert info["quality"]["optimal_command_units"] is None


@pytest.mark.parametrize("task", TASKS)
def test_quality_counters_do_not_classify_neutral_commands_as_errors(task):
    env = AtomicEnv(task, max_steps=1, level_bank_seed=20261008)
    _, reset = env.reset(seed=0)
    assert reset["quality"]["terminal_payoff"] is None
    _, _, _, _, info = env.step(PhysicalAction())
    quality = info["quality"]
    assert quality["commands"] == quality["neutral_commands"] == 1
    assert quality["active_commands"] == quality["input_units"] == 0
    assert quality["command_units"] == 1
    assert quality["unnecessary_input_units"] == 0
    assert quality["game_seconds"] == pytest.approx(0.1)
    assert quality["completion_efficiency"] is None
    assert quality["terminal_payoff"] is None
