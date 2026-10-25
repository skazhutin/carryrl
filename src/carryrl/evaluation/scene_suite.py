"""Versioned registry of diagnostic scene references; never a primary reward."""

from copy import deepcopy

from carryrl.envs.atomic_suite import TASKS
from carryrl.evaluation import scene_combat, scene_core, scene_motion, scene_progress

SCENE_SUITE_VERSION = "carryrl.scene_suite.v1"
SUPPORTED_TASKS = TASKS
_MODULES = (scene_progress, scene_core, scene_motion, scene_combat)
GOALS = {task: goal for module in _MODULES for task, goal in module.GOALS.items()}
EVENT_NAMES = tuple(dict.fromkeys(name for module in _MODULES for name in module.EVENT_NAMES))
_FACTORIES = ((scene_progress.SUPPORTED_TASKS, scene_progress.SceneProgressAnnotator),
              (scene_core.SUPPORTED_TASKS, scene_core.CoreSceneAnnotator),
              (scene_motion.SUPPORTED_TASKS, scene_motion.MotionSceneAnnotator),
              (scene_combat.SUPPORTED_TASKS, scene_combat.CombatSceneAnnotator))


class SuiteSceneAnnotator:
    """Union fixed target names while retaining explicit per-target unknown masks.

    Goals and supervision follow the disclosed atomic mechanics. Completion-only
    goal_choice is an explicit experiment, not a choice of risk/payoff preference.
    Each underlying reference is causal, and may abstain on individual targets.
    """

    def __init__(self, task, *, goal_choice_utility=None):
        if task not in TASKS:
            raise ValueError("unsupported atomic scene task")
        if task == "goal_choice" and goal_choice_utility != "completion_only":
            raise ValueError("goal_choice requires explicit completion_only scene objective")
        self.task = task
        self.reference = next(factory(task) for tasks, factory in _FACTORIES if task in tasks)

    def observe(self, frame, previous_action=None):
        if self.task in scene_progress.SUPPORTED_TASKS:
            row = self.reference.observe(frame)
        else:
            row = self.reference.observe(frame, previous_action=previous_action)
        row = deepcopy(row)
        for field in ("event_targets", "event_valid"):
            if set(row[field]) - set(EVENT_NAMES):
                raise ValueError("scene reference returned an undeclared event")
            row[field] = {name: row[field].get(name, False) for name in EVENT_NAMES}
        if (set(row["events"]) - set(EVENT_NAMES)
                or any(type(value) is not bool for name in ("event_targets", "event_valid")
                       for value in row[name].values())):
            raise ValueError("invalid scene event labels")
        return row
