"""Admitted TRAIN-only task curriculum and transactional per-task scheduling.

Technical controllers below are reachability probes, never PPO actions or stored
demonstrations. A scheduler changes stage only after commit_cycle is called by a
successfully accepted PPO update.
"""

import copy
import hashlib
import json
from pathlib import Path

from carryrl.envs.atomic_suite import TASKS
from carryrl.envs.curriculum import CURRICULUM_VERSION, STAGES

SCHEDULER_VERSION = "carryrl.task-curriculum-scheduler.v1"
STAGE_SPEC_SHA256 = hashlib.sha256(
    (Path(__file__).parents[1] / "envs" / "curriculum.py").read_bytes()).hexdigest()


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


class TaskCurriculumScheduler:
    """One independent level/stage/window history per task, fully serializable."""

    def __init__(self, *, config_hash: str, admission_hash: str, allowed_levels: dict,
                 state: dict | None = None):
        self.config_hash, self.admission_hash = config_hash, admission_hash
        self.allowed_levels = copy.deepcopy(allowed_levels)
        if set(self.allowed_levels) != set(TASKS):
            raise ValueError("Task curriculum requires all sixteen tasks")
        for task in TASKS:
            if set(allowed_levels[task]) != set(STAGES):
                raise ValueError(f"Missing task stages: {task}")
            for stage in STAGES:
                levels = allowed_levels[task][stage]
                if (not isinstance(levels, list) or len(levels) < 16
                        or len(set(levels)) != len(levels)
                        or any(type(x) is not int or not 0 <= x < 64 for x in levels)):
                    raise ValueError(f"Expected >=16 distinct admitted indices: {task}/{stage}")
        self._bindings = {"version": SCHEDULER_VERSION, "config_hash": config_hash,
                          "admission_hash": admission_hash,
                          "stage_spec_sha256": STAGE_SPEC_SHA256,
                          "allowed_levels_sha256": _digest(self.allowed_levels)}
        self._state = {**self._bindings, "serial": 0, "cycle": 0, "pending": [],
                       "issued": {}, "tasks": {task: {
                           "stage": 0, "episodes": 0, "rehearsal_cursor": 0,
                           "level_cursors": {stage: 0 for stage in STAGES},
                           "windows": {stage: [] for stage in STAGES},
                           "rehearsal_windows": {stage: [] for stage in STAGES},
                       } for task in TASKS}}
        if state is not None:
            if any(state.get(key) != value for key, value in self._bindings.items()):
                raise ValueError("Curriculum checkpoint/config/admission mismatch")
            if set(state.get("tasks", {})) != set(TASKS):
                raise ValueError("Invalid curriculum checkpoint task set")
            for task, item in state["tasks"].items():
                if type(item.get("stage")) is not int or not 0 <= item["stage"] < len(STAGES):
                    raise ValueError(f"Invalid checkpoint stage: {task}")
                for field, limit in (("windows", 32), ("rehearsal_windows", 16)):
                    windows = item.get(field, {})
                    if set(windows) != set(STAGES) or any(
                        not isinstance(window, list) or len(window) > limit
                        or any(type(row.get("success")) is not bool
                               or not isinstance(row.get("runtime_id"), str) for row in window)
                        for window in windows.values()
                    ):
                        raise ValueError(f"Invalid checkpoint windows: {task}")
                if (set(item.get("level_cursors", {})) != set(STAGES)
                        or any(type(value) is not int or value < 0
                               for value in item["level_cursors"].values())
                        or any(type(item.get(key)) is not int or item[key] < 0
                               for key in ("episodes", "rehearsal_cursor"))):
                    raise ValueError(f"Invalid checkpoint cursors: {task}")
            self._state = copy.deepcopy(state)

    def select_episode(self, task: str) -> dict:
        item = self._state["tasks"][task]
        item["episodes"] += 1
        rehearsal = item["stage"] > 0 and item["episodes"] % 5 == 0
        if rehearsal:
            stage_index = item["rehearsal_cursor"] % item["stage"]
            item["rehearsal_cursor"] += 1
        else:
            stage_index = item["stage"]
        stage = STAGES[stage_index]
        levels = self.allowed_levels[task][stage]
        cursor = item["level_cursors"][stage]
        item["level_cursors"][stage] += 1
        self._state["serial"] += 1
        ticket = {"id": self._state["serial"], "task": task, "stage": stage,
                  "level_index": levels[cursor % len(levels)], "rehearsal": rehearsal}
        self._state["issued"][str(ticket["id"])] = copy.deepcopy(ticket)
        return ticket

    def record_outcome(self, ticket: dict, success: bool, runtime_id: str):
        if type(success) is not bool or not isinstance(runtime_id, str) or not runtime_id:
            raise ValueError("Expected boolean success and effective runtime ID")
        key = str(ticket.get("id"))
        if self._state["issued"].get(key) != ticket:
            raise ValueError("Unknown, changed, or already recorded episode ticket")
        del self._state["issued"][key]
        self._state["pending"].append({**ticket, "success": success, "runtime_id": runtime_id})

    def commit_cycle(self) -> list[dict]:
        """Commit observations only after the caller accepts the PPO update."""
        decisions = []
        touched = set()
        for result in self._state["pending"]:
            item = self._state["tasks"][result["task"]]
            field, length = ("rehearsal_windows", 16) if result["rehearsal"] else ("windows", 32)
            window = item[field][result["stage"]]
            window.append({"success": result["success"], "runtime_id": result["runtime_id"]})
            del window[:-length]
            touched.add(result["task"])
        self._state["pending"] = []
        for task in TASKS:
            if task not in touched:
                continue
            item = self._state["tasks"][task]
            old = item["stage"]
            # A failed previous stage takes priority over current-stage promotion.
            failed = [index for index in range(old)
                      if len(item["rehearsal_windows"][STAGES[index]]) == 16
                      and sum(x["success"] for x in
                              item["rehearsal_windows"][STAGES[index]]) <= 8]
            window = item["windows"][STAGES[old]]
            wins = sum(row["success"] for row in window)
            unique = len({row["runtime_id"] for row in window})
            reason = None
            if failed:
                item["stage"], reason = min(failed), "rehearsal_failure"
            elif len(window) == 32 and wins <= 8 and old > 0:
                item["stage"], reason = old - 1, "current_failure"
            elif len(window) == 32 and wins >= 26 and unique >= 16 and old < 2:
                item["stage"], reason = old + 1, "current_success"
            if reason:
                decisions.append({"task": task, "from": STAGES[old],
                                  "to": STAGES[item["stage"]], "reason": reason,
                                  "wins": wins, "episodes": len(window), "unique_levels": unique})
                # Entry starts a fresh decision window; other stages keep their histories.
                item["windows"][STAGES[item["stage"]]] = []
                if reason == "rehearsal_failure":
                    item["rehearsal_windows"][STAGES[item["stage"]]] = []
        self._state["cycle"] += 1
        return decisions

    def state_dict(self):
        return copy.deepcopy(self._state)


def curriculum_allowed_levels(manifest: dict) -> dict:
    return {task: {stage: list(manifest["tasks"][task][stage]["allowed_level_indices"])
                   for stage in STAGES} for task in TASKS}


def _probe_level(task, stage, index, *, level_bank_seed, image_size, max_steps, noop=False):
    """Bounded RGB controller reachability; returns metadata, never trajectories."""
    from carryrl.agent.actions import MOUSE_BINS, PhysicalAction
    from carryrl.envs.atomic import (
        BALL,
        PLAYER,
        TARGET,
        AtomicEnv,
        _centroid,
        _visible_navigation_action,
    )
    from carryrl.envs.atomic_suite import EXTRA_TASKS, get_extra_controller

    env = AtomicEnv(task, image_size=image_size, max_steps=max_steps,
                    level_bank_seed=level_bank_seed, curriculum_stage=stage)
    frame, info = env.reset(seed=0, options={"level_index": index})
    controller = get_extra_controller(task) if task in EXTRA_TASKS and not noop else None
    previous_buttons, memory_choice = (), None
    for tick in range(max_steps):
        if noop:
            action = PhysicalAction()
        elif task == "navigation":
            action = _visible_navigation_action(frame, False)
        elif task == "reaction":
            ball, paddle = _centroid(frame, BALL), _centroid(frame, PLAYER)
            difference = ball[0] - paddle[0]
            action = PhysicalAction(keys=() if abs(difference) < 3 else
                                    ("right" if difference > 0 else "left",))
        elif task == "mouse":
            cursor, target = _centroid(frame, PLAYER), _centroid(frame, TARGET)
            dx, dy = target[0] - cursor[0], target[1] - cursor[1]
            buttons = (("left",) if abs(dx) <= 3 and abs(dy) <= 3
                       and "left" not in previous_buttons else ())
            action = PhysicalAction(mouse_buttons=buttons,
                                    dx=min(MOUSE_BINS, key=lambda value: abs(value - dx)),
                                    dy=min(MOUSE_BINS, key=lambda value: abs(value - dy)))
        elif task == "goal_choice":
            from carryrl.envs.atomic_combat import SAFE, _center, _keys_towards

            player, target = _center(frame, PLAYER), _center(frame, SAFE)
            # A player overlapping the goal rim shifts its visible centroid.
            # Use a one-movement tolerance to avoid oscillating between the
            # two reachable columns while still approaching the safe objective.
            tolerance = 4 * image_size / 96
            if player is None or target is None:
                action = PhysicalAction()
            elif abs(player[0] - target[0]) > tolerance:
                action = PhysicalAction(keys=("d" if target[0] > player[0] else "a",))
            else:
                action = PhysicalAction(keys=_keys_towards(player, target, tolerance))
        elif task == "memory":
            cue = _centroid(frame, TARGET)
            if cue is not None:
                memory_choice = "left" if cue[0] < image_size / 2 else "right"
            if memory_choice is None:
                raise RuntimeError("Technical memory probe did not observe initial cue")
            action = PhysicalAction(keys=(memory_choice,))
        else:
            action = controller.act(frame)
        previous_buttons = action.mouse_buttons
        frame, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return {"success": info.get("success") is True, "ticks": tick + 1,
                    "terminated": terminated, "truncated": truncated}
    raise RuntimeError("Technical probe exceeded declared tick budget")


def _protected_stage_runtime_ids(task, seed):
    from carryrl.envs.atomic import atomic_level_banks
    from carryrl.envs.runtime_identity import atomic_runtime_level_id

    protected = {}
    for bank_seed in dict.fromkeys((0, 20261008, seed)):
        for split in ("validation", "test"):
            for index, original in enumerate(atomic_level_banks(task, bank_seed)[split]):
                for stage in STAGES:
                    try:
                        identity = atomic_runtime_level_id(task, original, curriculum_stage=stage)
                    except ValueError:
                        # Legacy UI has fewer than four choices and cannot be a
                        # curriculum source. Its actual full mechanics remain protected.
                        if task == "ui" and stage != "full" and len(original["items"]) != 4:
                            continue
                        raise
                    protected.setdefault(identity, []).append({
                        "level_bank_seed": bank_seed, "split": split,
                        "level_index": index, "projection": stage})
    return protected


def prepare_task_curriculum(level_bank_seed: int, *, image_size=96, max_steps=80,
                            min_unique=16) -> dict:
    """Prepare admission evidence. Never train or weaken failed stage readiness."""
    from carryrl.envs.atomic import _identity, atomic_level_banks
    from carryrl.envs.curriculum import resolve_effective_level, runtime_parameters
    from carryrl.envs.runtime_identity import atomic_runtime_level_id

    if type(min_unique) is not int or min_unique < 16:
        raise ValueError("Task curriculum requires at least sixteen distinct levels")
    if max_steps != 80:
        raise ValueError("Task curriculum readiness protocol requires an 80-tick budget")
    manifest = {"schema_version": CURRICULUM_VERSION, "stage_spec_sha256": STAGE_SPEC_SHA256,
                "level_bank_seed": level_bank_seed, "image_size": image_size,
                "max_steps": max_steps, "min_unique": min_unique, "ready": True,
                "protocol": "TRAIN exact-runtime exclusion, noop rejection, RGB solver <=80 ticks",
                "solver_role": "technical reachability only; no demonstrations or quality claim",
                "tasks": {}, "errors": []}
    for task in TASKS:
        protected = _protected_stage_runtime_ids(task, level_bank_seed)
        bank = atomic_level_banks(task, level_bank_seed)["train"]
        manifest["tasks"][task] = {}
        for stage in STAGES:
            levels, excluded, identities = [], [], set()
            for index, original in enumerate(bank):
                try:
                    effective = resolve_effective_level(task, original, curriculum_stage=stage)
                    identity = atomic_runtime_level_id(task, original, curriculum_stage=stage)
                    if identity in protected:
                        excluded.append({"level_index": index, "reason": "heldout_runtime_match",
                                         "matches": protected[identity]})
                        continue
                    if identity in identities:
                        excluded.append({"level_index": index, "reason": "duplicate_runtime"})
                        continue
                    if task == "platforming" and stage == "easy":
                        if effective["obstacles"] != [original["obstacles"][0]]:
                            raise ValueError("Forbidden fixed-obstacle stage collapse")
                    probe_args = {"level_bank_seed": level_bank_seed,
                                  "image_size": image_size, "max_steps": max_steps}
                    noop = _probe_level(task, stage, index, noop=True, **probe_args)
                    if noop["success"]:
                        excluded.append({"level_index": index, "reason": "noop_win", "probe": noop})
                        continue
                    solver = _probe_level(task, stage, index, **probe_args)
                    if not solver["success"]:
                        excluded.append({"level_index": index, "reason": "solver_unverified",
                                         "probe": solver})
                        continue
                    identities.add(identity)
                    levels.append({"level_index": index, "origin_level_id": _identity(original),
                                   "runtime_id": identity, "effective_parameters": effective,
                                   "runtime_parameters": runtime_parameters(task, effective),
                                   "solver": solver, "noop": noop})
                except (ValueError, RuntimeError, TypeError) as error:
                    excluded.append({"level_index": index, "reason": "probe_error",
                                     "error": str(error)})
            # Deterministic cue alternation gives balanced memory sampling as
            # far as the admitted source bank permits, without duplicating data.
            if task == "memory":
                groups = [[row for row in levels if row["effective_parameters"]["cue"] == cue]
                          for cue in (0, 1)]
                balanced_count = min(map(len, groups))
                for cue in (0, 1):
                    excluded.extend({"level_index": row["level_index"],
                                     "reason": "balance_memory_cues"}
                                    for row in groups[cue][balanced_count:])
                levels = [groups[cue][index] for index in range(balanced_count)
                          for cue in (0, 1)]
                if balanced_count < 8:
                    manifest["errors"].append(f"{task}/{stage}: fewer than eight levels per cue")
            ready = len(levels) >= min_unique
            manifest["tasks"][task][stage] = {
                "ready": ready, "unique_levels": len(levels), "levels": levels,
                "allowed_level_indices": [row["level_index"] for row in levels],
                "excluded": excluded}
            if not ready:
                manifest["errors"].append(
                    f"{task}/{stage}: {len(levels)}/{min_unique} unique levels")
    manifest["ready"] = not manifest["errors"]
    return manifest


def validate_task_curriculum_admission(path, *, level_bank_seed, image_size=96, max_steps=80,
                                       expected_sha256=None) -> dict:
    """Fail closed on unready, stale, altered, or wrong-source admission evidence."""
    from carryrl.envs.atomic import _identity, atomic_level_banks
    from carryrl.envs.curriculum import resolve_effective_level, runtime_parameters
    from carryrl.envs.runtime_identity import atomic_runtime_level_id

    raw = Path(path).read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and expected_sha256 != actual_sha256:
        raise ValueError("Task curriculum admission SHA256 mismatch")
    manifest = json.loads(raw)
    if (manifest.get("schema_version") != CURRICULUM_VERSION
            or manifest.get("stage_spec_sha256") != STAGE_SPEC_SHA256
            or manifest.get("level_bank_seed") != level_bank_seed
            or manifest.get("image_size") != image_size or manifest.get("max_steps") != max_steps
            or manifest.get("ready") is not True or manifest.get("errors")
            or manifest.get("min_unique", 0) < 16):
        raise ValueError("Task curriculum admission is unready, stale, or incompatible")
    if set(manifest.get("tasks", {})) != set(TASKS):
        raise ValueError("Admission must contain all sixteen tasks")
    for task in TASKS:
        bank = atomic_level_banks(task, level_bank_seed)["train"]
        protected = _protected_stage_runtime_ids(task, level_bank_seed)
        if set(manifest["tasks"][task]) != set(STAGES):
            raise ValueError(f"Admission missing stages: {task}")
        for stage in STAGES:
            evidence = manifest["tasks"][task][stage]
            indices, identities = [], set()
            for row in evidence.get("levels", []):
                index = row.get("level_index")
                if type(index) is not int or not 0 <= index < len(bank):
                    raise ValueError("Admission has invalid TRAIN source index")
                original = bank[index]
                effective = resolve_effective_level(task, original, curriculum_stage=stage)
                identity = atomic_runtime_level_id(task, original, curriculum_stage=stage)
                if (row.get("origin_level_id") != _identity(original)
                        or row.get("effective_parameters") != json.loads(json.dumps(effective))
                        or row.get("runtime_parameters") != json.loads(json.dumps(
                            runtime_parameters(task, effective)))
                        or row.get("runtime_id") != identity or identity in protected
                        or identity in identities or row.get("noop", {}).get("success") is not False
                        or row.get("solver", {}).get("success") is not True
                        or type(row.get("solver", {}).get("ticks")) is not int
                        or not 1 <= row["solver"]["ticks"] <= 80):
                    raise ValueError(
                        f"Invalid curriculum admission evidence: {task}/{stage}/{index}")
                indices.append(index)
                identities.add(identity)
            if (len(indices) < manifest["min_unique"] or len(set(indices)) != len(indices)
                    or evidence.get("ready") is not True
                    or evidence.get("unique_levels") != len(identities)
                    or evidence.get("allowed_level_indices") != indices):
                raise ValueError(f"Insufficient or inconsistent admission: {task}/{stage}")
            if task == "memory":
                cues = [bank[index]["cue"] for index in indices]
                if min(cues.count(0), cues.count(1)) < 8:
                    raise ValueError("Unbalanced admitted memory cue support")
    return manifest
