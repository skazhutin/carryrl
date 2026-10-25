"""Versioned diagnostic quality definitions; never policy inputs or reward weights."""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from math import isfinite
from typing import Any

from carryrl.agent.actions import PhysicalAction

QUALITY_VERSION = "carryrl.atomic.quality.v1"

# These are mechanic definitions, not claims about a controller's performance.
_TASK_CONTRACTS = {
    "navigation": {
        "primary_success": "Reach the exit cell through the maze.",
        "necessary_waits": "No mechanic requires waiting; neutral input still costs game time.",
        "necessary_releases": "Movement is held input; no release edge is required.",
        "necessary_backtracking": "A valid route may move away from the exit around walls.",
        "failure_events": "No lethal cells; failure to finish can exhaust the episode budget.",
    },
    "reaction": {
        "primary_success": "Catch the required number of falling balls with the paddle.",
        "necessary_waits": "Wait for the ball to reach a correctly positioned paddle.",
        "necessary_releases": "Release movement to keep a correctly positioned paddle still.",
        "necessary_backtracking": "Paddle reversals can be required for successive balls.",
        "failure_events": "A missed catch fails the task.",
    },
    "mouse": {
        "primary_success": "Click the required targets in their displayed order.",
        "necessary_waits": "A release interval can be needed between clicks.",
        "necessary_releases": "Release the left mouse button before another click edge.",
        "necessary_backtracking": "Successive targets may require reversing cursor motion.",
        "failure_events": "Missed clicks consume a difficulty-dependent miss allowance.",
    },
    "memory": {
        "primary_success": "After unlocking, reach the side identified by the earlier cue.",
        "necessary_waits": "Wait through the cue and locked interval before movement works.",
        "necessary_releases": "No release edge is required.",
        "necessary_backtracking": "Corrections before selecting a side are not certified waste.",
        "failure_events": "Reaching the wrong side fails the task.",
    },
    "timing": {
        "primary_success": "Pass all required pipes without collision or leaving vertical bounds.",
        "necessary_waits": "Let gravity and moving pipes advance between timed flaps.",
        "necessary_releases": "Release space before every subsequent flap edge.",
        "necessary_backtracking": "Vertical reversals are part of flight control.",
        "failure_events": "Pipe collision or leaving vertical bounds fails the task.",
    },
    "platforming": {
        "primary_success": "Reach the horizontal goal while jumping over solid hazards.",
        "necessary_waits": "Allow the jump arc and landing to complete when needed.",
        "necessary_releases": "Release space before the next grounded jump edge.",
        "necessary_backtracking": "Position corrections are not automatically unnecessary.",
        "failure_events": "Contact with an obstacle fails the task.",
    },
    "steering": {
        "primary_success": "Complete the required course progress while staying on the road.",
        "necessary_waits": "The course advances automatically; neutral steering can be correct.",
        "necessary_releases": "Release steering when inertia already provides the needed motion.",
        "necessary_backtracking": "Lateral reversals follow road bends and counter inertia.",
        "failure_events": "Leaving the road fails the task.",
    },
    "camera": {
        "primary_success": "Approach the doorway and align both camera bearing and pitch.",
        "necessary_waits": "No mechanic requires waiting; observation can inform corrections.",
        "necessary_releases": "Release translation while adjusting the camera when needed.",
        "necessary_backtracking": "Translation and camera corrections can require reversals.",
        "failure_events": "Moving beyond the allowed distance from the start fails the task.",
    },
    "interaction": {
        "primary_success": "Pick up the item with E nearby, then use it at the use location.",
        "necessary_waits": "A release interval can be needed between interactions.",
        "necessary_releases": "Release E before each subsequent interaction edge.",
        "necessary_backtracking": "Collecting the item before use can require revisiting cells.",
        "failure_events": "Entering the hazard cell fails the task.",
    },
    "causality": {
        "primary_success": "Activate the remote button, then pass the opened door to the exit.",
        "necessary_waits": "No mechanic requires a delay after the button opens the door.",
        "necessary_releases": "E activation uses a press edge.",
        "necessary_backtracking": "Visit the remote button before returning to the passage.",
        "failure_events": "Entering the hazard cell fails the task.",
    },
    "key_door": {
        "primary_success": "Collect the key, unlock the door with E, then reach the exit.",
        "necessary_waits": "A release interval can be needed between interactions.",
        "necessary_releases": "Release E between collecting the key and unlocking the door.",
        "necessary_backtracking": "Return from the key location to the locked passage.",
        "failure_events": "Entering the hazard cell fails the task.",
    },
    "goal_choice": {
        "primary_success": "Reach either payoff goal without colliding with the guard.",
        "necessary_waits": "Waiting can change guard position before approaching a risky goal.",
        "necessary_releases": "Release movement to wait or stop; no press edge is required.",
        "necessary_backtracking": "Avoiding the moving guard can require a detour or retreat.",
        "failure_events": "Contact with the moving guard at the risky side fails the task.",
    },
    "hazards": {
        "primary_success": "Reach the goal without touching traps or the moving patrol.",
        "necessary_waits": "Waiting can synchronize movement with patrol timing.",
        "necessary_releases": "Release movement to wait for a safe patrol phase.",
        "necessary_backtracking": "Safe routes may detour or retreat around traps and patrol.",
        "failure_events": "Trap contact or patrol contact before or after its move fails.",
    },
    "shooting": {
        "primary_success": "Land the required aimed shots while within firing range.",
        "necessary_waits": "A release interval can be needed between shots.",
        "necessary_releases": "Release the left mouse button before every subsequent shot.",
        "necessary_backtracking": "Position or aim corrections are not certified waste.",
        "failure_events": "Six missed shots exhaust the firing allowance and fail the task.",
    },
    "ui": {
        "primary_success": "Open the menu, select the requested item, confirm, then close it.",
        "necessary_waits": "A release interval can be needed when repeating an edge key.",
        "necessary_releases": "Tab, enter and escape use press edges; release before reuse.",
        "necessary_backtracking": "Menu selection may wrap or reverse to reach the requested item.",
        "failure_events": "Confirming the wrong item fails the task.",
    },
    "chain": {
        "primary_success": "Collect the key, activate the switch, unlock the door, reach the exit.",
        "necessary_waits": "Release intervals can be needed between ordered interactions.",
        "necessary_releases": "Release E before every subsequent interaction edge.",
        "necessary_backtracking": "The key, switch and door order can require revisiting cells.",
        "failure_events": "Entering the hazard cell fails the task.",
    },
}

_METRIC_SEMANTICS = {
    "success": "Task completion is separate from payoff, visible score and command cost.",
    "game_time": "Count elapsed ticks and tick duration, including necessary neutral intervals.",
    "command_input_units": (
        "Count held keys, held mouse buttons and each nonzero motion or wheel component per tick; "
        "neutral input costs zero command units. This is an accounting measure, not a penalty."
    ),
    "unnecessary_input_units": (
        "Count only inputs proven unnecessary by the implemented mechanic audit; "
        "do not label waits, releases, backtracking or exploration as errors by default."
    ),
    "risk": "Describe failure exposure separately; no probability or scalar utility is certified.",
}


def task_quality_contract(task: str) -> dict[str, Any]:
    """Return an independent, JSON-compatible diagnostic contract for one game."""
    if task not in _TASK_CONTRACTS:
        raise ValueError(f"Unknown atomic task: {task!r}")
    contract = deepcopy(_TASK_CONTRACTS[task])
    failure_events = contract.pop("failure_events")
    contract.update({
        "quality_version": QUALITY_VERSION,
        "task": task,
        "visible_score": (
            "Progress is at most 4 before success. A successful shortest cardinal route scores 5 "
            "only without audited unnecessary inputs; longer routes score 4. "
            "This certifies cardinal input units, not minimum ticks."
            if task == "navigation" else
            "Success pays the chosen goal value (safe 1..2, risky 3..5); 5 is not guaranteed. "
            "Partial progress is at most 1. The wrapper reduces a payoff of 5 to 4 when audited "
            "unnecessary inputs occurred. Visible score and nominal payoff are distinct; "
            "neither certifies an optimal risk policy."
            if task == "goal_choice" else
            "Score 0..4 records mechanic-specific progress; success normally scores 5, reduced "
            "to 4 by audited unnecessary inputs. Score 5 does not certify minimum time, "
            "minimum commands or an optimal trajectory."
        ),
        "certified_optimum": "shortest_cardinal_input_units" if task == "navigation" else None,
        "best_known_result": None,
        "risk": {
            "failure_events": failure_events,
            "tradeoff": (
                "Compare success rate and nominal goal payoff separately. Risk tolerance and "
                "payoff weights are unspecified; the safe diagnostic teacher is not an "
                "optimal-policy certificate."
                if task == "goal_choice" else
                "Report observed failure and timeouts separately; no risk utility is specified."
            ),
        },
        "metric_semantics": deepcopy(_METRIC_SEMANTICS),
    })
    return contract


def command_input_units(action: PhysicalAction) -> int:
    """Count active input components without classifying their necessity or quality."""
    if not isinstance(action, PhysicalAction):
        raise TypeError("Expected PhysicalAction")
    return (len(action.keys) + len(action.mouse_buttons)
            + int(bool(action.dx)) + int(bool(action.dy)) + int(bool(action.wheel)))


_METRICS = (
    "game_seconds", "commands", "active_commands", "neutral_commands", "input_units",
    "command_units", "unnecessary_input_units", "optimal_command_units",
    "excess_command_units", "completion_efficiency", "terminal_payoff",
)
_NAVIGATION_METRICS = {
    "optimal_command_units", "excess_command_units", "completion_efficiency",
}


def summarize_episode_quality(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Average eligible diagnostics with explicit per-metric coverage.

    Episodes measured counts matching version/task diagnostic records, even if a
    particular metric is absent or malformed. Old records are never reconstructed.
    Successful means require an independently known successful visual outcome.
    """
    metrics = {name: {"count": 0, "mean": None} for name in _METRICS}
    successful_metrics = deepcopy(metrics)
    episodes_measured = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        quality = row.get("quality")
        task = row.get("task")
        if (not isinstance(task, str) or task not in _TASK_CONTRACTS
                or not isinstance(quality, Mapping)
                or quality.get("quality_version") != QUALITY_VERSION
                or quality.get("task") != task):
            continue
        episodes_measured += 1
        outcome = row.get("outcome")
        known = isinstance(outcome, Mapping) and outcome.get("known") is True
        success = (known and outcome.get("success") is True
                   and outcome.get("failure") is False)
        failure = (known and outcome.get("failure") is True
                   and outcome.get("success") is False)
        for name in _METRICS:
            value = quality.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            # Converting arbitrarily large Python integers must not crash a report.
            try:
                value = float(value)
            except OverflowError:
                continue
            if not isfinite(value) or value < 0:
                continue
            if name in _NAVIGATION_METRICS and task != "navigation":
                continue
            if name in {"excess_command_units", "completion_efficiency"} and not success:
                continue
            if name == "completion_efficiency" and value > 1:
                continue
            if name == "terminal_payoff":
                if task != "goal_choice" or not (success or failure):
                    continue
                if (failure and value != 0) or (success and not 1 <= value <= 5):
                    continue
            for target in (metrics, successful_metrics) if success else (metrics,):
                entry = target[name]
                entry["count"] += 1
                # Running means avoid overflow when finite values have a large sum.
                previous_mean = entry["mean"] if entry["mean"] is not None else 0.0
                entry["mean"] = previous_mean + (value - previous_mean) / entry["count"]
    return {
        "quality_version": QUALITY_VERSION,
        "episodes_measured": episodes_measured,
        "metrics": metrics,
        "successful_metrics": successful_metrics,
    }
