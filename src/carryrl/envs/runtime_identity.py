"""Train-admission signatures for effective atomic mechanics, independent of pixels.

These supplement, and do not change, the published full-difficulty bank IDs.
The field lists/defaults follow AtomicEnv and the three atomic game modules.
They detect exact effective-parameter matches, not arbitrary behavioural equivalence.
"""

import math
from functools import lru_cache

from carryrl.envs.atomic import _identity, atomic_level_banks


def atomic_runtime_level_id(task: str, original: dict, difficulty: float = 1.0, *,
                            curriculum_stage: str | None = None) -> str:
    """Hash canonical used mechanics after scalar or explicit task resolution."""
    from carryrl.envs.curriculum import resolve_effective_level, runtime_parameters

    level = resolve_effective_level(task, original, difficulty=difficulty,
                                    curriculum_stage=curriculum_stage)
    used = runtime_parameters(task, level, difficulty)

    def normalize(value):
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return value

    return _identity({"task": task, "runtime": normalize(used)})


def _protected_difficulties(difficulty, heldout_difficulties):
    values = (1.0, difficulty, *heldout_difficulties)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("runtime held-out difficulties must be finite numbers in [0,1]")
    return tuple(dict.fromkeys(values))


@lru_cache(maxsize=256)
def _heldout_runtime_levels(task: str, level_bank_seed: int, difficulty: float,
                            *, heldout_difficulties: tuple = ()):
    difficulties = _protected_difficulties(difficulty, heldout_difficulties)
    matches = {}
    for bank_seed in dict.fromkeys((0, 20261008, level_bank_seed)):
        for split in ("validation", "test"):
            for index, level in enumerate(atomic_level_banks(task, bank_seed)[split]):
                for effective_difficulty in difficulties:
                    identity = atomic_runtime_level_id(task, level, effective_difficulty)
                    matches.setdefault(identity, []).append({
                        "level_bank_seed": bank_seed, "split": split, "level_index": index,
                        "difficulty": effective_difficulty, "level_id": _identity(level)})
    return matches


def runtime_heldout_collisions(task: str, level: dict, difficulty: float,
                               level_bank_seed: int, *,
                               heldout_difficulties: tuple = ()) -> list[dict]:
    difficulties = _protected_difficulties(difficulty, heldout_difficulties)
    identity = atomic_runtime_level_id(task, level, difficulty)
    return list(_heldout_runtime_levels(task, level_bank_seed, difficulty,
        heldout_difficulties=difficulties).get(identity, ()))


def allowed_runtime_train_indices(task: str, level_bank_seed: int,
                                  difficulty: float, *,
                                  heldout_difficulties: tuple = ()) -> list[int]:
    bank = atomic_level_banks(task, level_bank_seed)["train"]
    allowed = [index for index, level in enumerate(bank)
               if not runtime_heldout_collisions(task, level, difficulty, level_bank_seed,
                                                heldout_difficulties=heldout_difficulties)]
    if not allowed:
        raise ValueError(f"{task} training bank has no runtime levels without held-out "
                         f"matches at difficulty {difficulty}")
    return allowed
