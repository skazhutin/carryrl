"""Explicit validation-only reaction speed interventions for frozen evaluation."""

import math
from copy import deepcopy

from carryrl.envs.atomic import (
    AtomicEnv,
    _identity,
    atomic_level_banks,
    atomic_semantic_level_id,
    verify_visible_outcome,
)
from carryrl.envs.runtime_identity import atomic_runtime_level_id

REACTION_DYNAMICS_VERSION = "carryrl.reaction_speed_evaluation.v1"
REACTION_SPEED_PROFILES = {
    "canonical96": None,
    "reaction96_speed2": 2.0,
    "reaction96_speed2_5": 2.5,
}


def reaction_speed_condition(profile):
    return {"version": REACTION_DYNAMICS_VERSION, "task": "reaction",
            "axis": "horizontal_speed_magnitude",
            "speed_magnitude": REACTION_SPEED_PROFILES[profile]}


def reaction_speed_level(origin, magnitude):
    level = deepcopy(origin)
    if magnitude is not None:
        level["ball_dx"] = math.copysign(magnitude, origin["ball_dx"])
    return level


def validate_reaction_speed_validation_bank(bank_seed):
    """Reject derived validation aliases conservatively before any frozen rollout.

    Replacing the old speed can merge formerly different levels. The fixed 2.0
    training intervention therefore also defines the protected TRAIN signatures.
    """
    banks = dict.fromkeys((0, 20261008, 20261015, bank_seed))
    protected = {atomic_runtime_level_id("reaction", reaction_speed_level(level, 2.0))
                 for seed in banks for level in atomic_level_banks("reaction", seed)["train"]}
    aliases = [index for index, level in enumerate(
        atomic_level_banks("reaction", bank_seed)["validation"])
        if atomic_runtime_level_id("reaction", reaction_speed_level(level, 2.0)) in protected]
    if aliases:
        raise ValueError("reaction transformed validation overlaps protected TRAIN: "
                         f"bank {bank_seed}, indices {aliases}")


class ReactionSpeedEvaluationEnv(AtomicEnv):
    """Keep origin positions and all other physics; never alter bank dictionaries."""

    def __init__(self, task, *, split="validation", speed_magnitude=None, **kwargs):
        if (task != "reaction" or split != "validation"
                or (speed_magnitude is not None and (
                    type(speed_magnitude) not in (int, float)
                    or speed_magnitude not in (2.0, 2.5)))
                or type(kwargs.get("difficulty", 1.0)) not in (int, float)
                or kwargs.get("difficulty", 1.0) != 1.0
                or kwargs.get("image_size", 96) != 96):
            raise ValueError("reaction speed evaluation requires validation, native96, "
                             "difficulty1 and original/2.0/2.5 speed")
        visual = kwargs.get("visual_config")
        if visual is not None and getattr(visual, "mode", None) != "canonical":
            raise ValueError("reaction speed evaluation requires canonical presentation")
        super().__init__(task, split=split, **kwargs)
        self.speed_magnitude = speed_magnitude

    def reset(self, **kwargs):
        frame, info = super().reset(**kwargs)
        if self.speed_magnitude is None:
            return frame, info
        self._level = reaction_speed_level(self._level, self.speed_magnitude)
        self._effective_level_id = _identity(self._level)
        self._semantic_level_id = atomic_semantic_level_id("reaction", self._level)
        self._ball_dx = self._level["ball_dx"]
        self._frame = self._draw()
        return self._frame.copy(), self._info(verify_visible_outcome(self._frame))
