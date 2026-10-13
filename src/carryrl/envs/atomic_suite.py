"""The complete first-stage suite and lazy dispatch for its extra mechanics.

The first four IDs keep their ordering and current level-bank seeds. Existing
explicit four-task configurations remain compatible with the same policy graph.
"""

from functools import lru_cache
from importlib import import_module

SUITE_VERSION = "carryrl.atomic.v3"
SCORING_VERSION = "carryrl.atomic.efficient.v1"
TASKS = (
    "navigation", "reaction", "mouse", "memory",
    "timing", "platforming", "steering", "camera",
    "interaction", "causality", "key_door", "goal_choice",
    "hazards", "shooting", "ui", "chain",
)
CORE_TASKS = TASKS[:4]
EXTRA_TASKS = TASKS[4:]
_GROUPS = {
    "atomic_motion": ("timing", "platforming", "steering", "camera"),
    "atomic_interaction": ("interaction", "causality", "key_door", "chain"),
    "atomic_combat": ("goal_choice", "hazards", "shooting", "ui"),
}


@lru_cache(maxsize=3)
def _module(name: str):
    module = import_module(f"carryrl.envs.{name}")
    if set(module.GAMES) != set(_GROUPS[name]):
        raise ValueError(f"{name} must implement exactly its declared suite mechanics")
    return module


def _group(task: str) -> str:
    for name, tasks in _GROUPS.items():
        if task in tasks:
            return name
    raise ValueError(f"Not an extra atomic mechanic: {task}")


def get_extra_game(task: str):
    return _module(_group(task)).GAMES[task]


def get_extra_controller(task: str, fail: bool = False):
    """Diagnostic pixel controller, never an action mask or input to the policy."""
    return _module(_group(task)).make_controller(task, fail=fail)
