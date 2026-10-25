"""Pure quality contracts distinguish task outcome from input accounting."""

import json

import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic_suite import TASKS
from carryrl.evaluation.quality import (
    QUALITY_VERSION,
    command_input_units,
    summarize_episode_quality,
    task_quality_contract,
)


@pytest.mark.parametrize("task", TASKS)
def test_every_task_has_a_json_contract_without_fabricated_results(task):
    contract = task_quality_contract(task)
    assert json.loads(json.dumps(contract)) == contract
    assert contract["quality_version"] == QUALITY_VERSION == "carryrl.atomic.quality.v1"
    assert contract["task"] == task
    assert contract["certified_optimum"] == (
        "shortest_cardinal_input_units" if task == "navigation" else None
    )
    assert contract["best_known_result"] is None
    for field in ("primary_success", "visible_score", "necessary_waits", "necessary_releases",
                  "necessary_backtracking"):
        assert isinstance(contract[field], str) and contract[field]
    assert contract["risk"]["failure_events"]
    assert contract["risk"]["tradeoff"]


def test_contract_mutation_cannot_corrupt_future_or_other_task_results():
    original = task_quality_contract("memory")
    changed = task_quality_contract("memory")
    changed["primary_success"] = "corrupt"
    changed["risk"]["failure_events"] = "corrupt"
    changed["metric_semantics"]["command_input_units"] = "corrupt"
    assert task_quality_contract("memory") == original
    assert task_quality_contract("mouse")["metric_semantics"] == original["metric_semantics"]


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError, match="Unknown atomic task"):
        task_quality_contract("unknown")


@pytest.mark.parametrize(("action", "expected"), (
    (PhysicalAction(), 0),
    (PhysicalAction(keys=("up", "down")), 2),
    (PhysicalAction(keys=("right", "d")), 2),
    (PhysicalAction(dx=-64), 1),
    (PhysicalAction(dx=1, dy=-1), 2),
    (PhysicalAction(keys=("w", "space"), mouse_buttons=("left", "right"),
                    dx=-64, dy=2, wheel=-1), 7),
))
def test_command_units_count_components_not_effect_or_magnitude(action, expected):
    assert command_input_units(action) == expected


def test_command_accounting_requires_physical_action():
    with pytest.raises(TypeError, match="PhysicalAction"):
        command_input_units({"keys": []})


def test_necessary_waits_releases_and_backtracking_are_explicit():
    assert "locked interval" in task_quality_contract("memory")["necessary_waits"]
    assert "gravity" in task_quality_contract("timing")["necessary_waits"]
    assert "inertia" in task_quality_contract("steering")["necessary_releases"]
    assert "patrol timing" in task_quality_contract("hazards")["necessary_waits"]
    for task in ("mouse", "shooting"):
        assert "Release the left mouse button" in task_quality_contract(task)["necessary_releases"]
    for task in ("interaction", "key_door", "chain"):
        assert "Release E" in task_quality_contract(task)["necessary_releases"]
    assert "Return from the key" in task_quality_contract("key_door")["necessary_backtracking"]
    assert "revisiting" in task_quality_contract("chain")["necessary_backtracking"]


def test_goal_choice_does_not_impose_an_unspecified_risk_preference():
    contract = task_quality_contract("goal_choice")
    assert "either payoff goal" in contract["primary_success"]
    assert "safe 1..2, risky 3..5" in contract["visible_score"]
    assert "5 is not guaranteed" in contract["visible_score"]
    assert "reduces a payoff of 5 to 4" in contract["visible_score"]
    assert "weights are unspecified" in contract["risk"]["tradeoff"]
    assert contract["certified_optimum"] is None
    assert contract["best_known_result"] is None


def _row(task="navigation", success=True, **quality):
    return {
        "task": task,
        "outcome": {"known": True, "success": success, "failure": not success},
        "quality": {"quality_version": QUALITY_VERSION, "task": task, **quality},
    }


def test_summary_old_and_mismatched_records_have_no_inferred_metrics():
    wrong_version = _row(game_seconds=10)
    wrong_version["quality"]["quality_version"] = "old"
    wrong_task = _row(game_seconds=10)
    wrong_task["quality"]["task"] = "memory"
    rows = [{"task": "navigation", "steps": 9, "outcome": {"success": True}},
            wrong_version, wrong_task, {"task": "memory", "quality": None}]
    result = summarize_episode_quality(rows)
    assert result["quality_version"] == QUALITY_VERSION
    assert result["episodes_measured"] == 0
    assert all(metric == {"count": 0, "mean": None}
               for section in ("metrics", "successful_metrics")
               for metric in result[section].values())


def test_summary_means_have_per_metric_coverage_and_separate_successes():
    result = summarize_episode_quality([
        _row(game_seconds=10, commands=20, completion_efficiency=0.5,
             optimal_command_units=4, excess_command_units=4),
        _row(success=False, game_seconds=2, commands=4, completion_efficiency=1,
             optimal_command_units=4, excess_command_units=0),
        _row("memory", game_seconds=6, input_units=3, completion_efficiency=1,
             optimal_command_units=8, excess_command_units=0),
    ])
    assert result["episodes_measured"] == 3
    assert result["metrics"]["game_seconds"] == {"count": 3, "mean": 6}
    assert result["successful_metrics"]["game_seconds"] == {"count": 2, "mean": 8}
    assert result["metrics"]["commands"] == {"count": 2, "mean": 12}
    assert result["successful_metrics"]["commands"] == {"count": 1, "mean": 20}
    assert result["metrics"]["input_units"] == {"count": 1, "mean": 3}
    assert result["metrics"]["optimal_command_units"] == {"count": 2, "mean": 4}
    assert result["metrics"]["excess_command_units"] == {"count": 1, "mean": 4}
    assert result["metrics"]["completion_efficiency"] == {"count": 1, "mean": 0.5}


@pytest.mark.parametrize("outcome", (
    None,
    {"known": False, "success": True, "failure": False},
    {"success": True, "failure": False},
    {"known": True, "success": True, "failure": True},
    {"known": True, "success": False, "failure": False},
))
def test_uncertain_outcome_never_certifies_efficiency_or_successful_metrics(outcome):
    row = _row(game_seconds=2, completion_efficiency=1, excess_command_units=0)
    row["outcome"] = outcome
    result = summarize_episode_quality([row])
    assert result["metrics"]["game_seconds"] == {"count": 1, "mean": 2}
    assert result["metrics"]["completion_efficiency"]["count"] == 0
    assert result["metrics"]["excess_command_units"]["count"] == 0
    assert result["successful_metrics"]["game_seconds"]["count"] == 0


def test_goal_payoff_requires_explicit_diagnostic_and_known_terminal_outcome():
    unknown = _row("goal_choice", terminal_payoff=5)
    unknown["outcome"]["known"] = False
    unfinished = _row("goal_choice", terminal_payoff=1)
    unfinished["outcome"] = {"known": True, "success": False, "failure": False, "score": 1}
    partial = _row("goal_choice", success=False)
    partial["outcome"]["score"] = 1
    result = summarize_episode_quality([
        _row("goal_choice", terminal_payoff=2),
        _row("goal_choice", terminal_payoff=5),
        _row("goal_choice", success=False, terminal_payoff=0),
        _row("navigation", terminal_payoff=5),
        _row("goal_choice", success=False, terminal_payoff=1),
        unknown, unfinished, partial,
    ])
    assert result["metrics"]["terminal_payoff"] == {"count": 3, "mean": pytest.approx(7 / 3)}
    assert result["successful_metrics"]["terminal_payoff"] == {"count": 2, "mean": 3.5}


@pytest.mark.parametrize("value", (True, False, -1, float("inf"), float("nan"),
                                   "2", None, [], {}, 10**1000))
def test_malformed_values_are_skipped_without_dropping_other_metrics(value):
    result = summarize_episode_quality([_row(game_seconds=value, commands=2)])
    assert result["metrics"]["game_seconds"] == {"count": 0, "mean": None}
    assert result["metrics"]["commands"] == {"count": 1, "mean": 2}


def test_summary_rejects_efficiency_above_one_and_handles_large_finite_means():
    result = summarize_episode_quality([
        _row(game_seconds=1e308, completion_efficiency=1.1),
        _row(game_seconds=1e308),
    ])
    assert result["metrics"]["game_seconds"] == {"count": 2, "mean": 1e308}
    assert result["metrics"]["completion_efficiency"]["count"] == 0


def test_summary_does_not_mutate_input():
    rows = [_row(game_seconds=2)]
    snapshot = json.dumps(rows)
    summarize_episode_quality(rows)
    assert json.dumps(rows) == snapshot
