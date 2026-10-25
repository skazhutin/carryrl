"""Concrete fresh-policy jobs and primary-artifact adapters for the bounded cycle."""

import hashlib
import json
import os
import signal
from dataclasses import replace
from pathlib import Path

from carryrl.envs.atomic_suite import TASKS
from carryrl.training.run import atomic_json, file_sha256


def _read(path):
    return json.loads(Path(path).read_text())


def _state_hash(state, *, frozen):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        is_frozen = name.startswith(("vision.", "language."))
        if is_frozen != frozen:
            continue
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def inspect_initial_checkpoint(checkpoint, *, expected=None):
    """Authenticate a real CP0 before the first PPO update, never import its weights."""
    import torch

    checkpoint = Path(checkpoint)
    manifest = _read(checkpoint / "manifest.json")
    archive = checkpoint / "state.pt"
    if file_sha256(archive) != manifest["sha256"]:
        raise ValueError("initial checkpoint hash mismatch")
    data = torch.load(archive, map_location="cpu", mmap=True, weights_only=True)
    if any(data[name] != 0 for name in ("rollout", "decisions", "weights_revision")):
        raise ValueError("fresh policy counters must start at zero")
    if data["optimizer_state"]["state"]:
        raise ValueError("fresh policy optimizer must be empty")
    if any(bool(data["agent_state"][name].count_nonzero()) for name in (
        "value_head.weight", "value_head.bias"
    )):
        raise ValueError("fresh critic must start at zero")
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": manifest["sha256"],
        "initial_trainable_sha256": _state_hash(data["agent_state"], frozen=False),
        "frozen_sha256": _state_hash(data["agent_state"], frozen=True),
        "optimizer_empty": True, "counters_zero": True, "critic_zero": True,
    }
    if expected is not None:
        for key in ("initial_trainable_sha256", "frozen_sha256"):
            if result[key] != expected[key]:
                raise ValueError(f"paired CP0 mismatch: {key}")
    return result


def normalize_evaluation(path):
    """Retain complete per-game denominators; reject unknown or partial cohorts."""
    report = _read(path)
    if report.get("split", "validation") != "validation":
        raise ValueError("pilot selection requires validation")
    result = {}
    for task in TASKS:
        rows = [row for row in report["records"]
                if row["policy"] == "frozen_stochastic" and row["task"] == task]
        if len(rows) != 16 or any(not row["outcome"].get("known") for row in rows):
            raise ValueError(f"incomplete stochastic validation cohort: {task}")
        result[task] = {
            "episodes": 16,
            "successes": sum(row["outcome"].get("success") is True for row in rows),
        }
    return {"episodes": 256, "successes": sum(v["successes"] for v in result.values()),
            "games": result}


def policy_selection_report(run, *, rollout, initial, seed):
    run = Path(run)
    games = {task: {"successful_effective_levels": []} for task in TASKS}
    unique = {task: set() for task in TASKS}
    with (run / "training.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["rollout"] > rollout:
                continue
            for episode in row["episode_records"]:
                if episode["outcome"].get("success") is True:
                    provenance = episode["provenance"]
                    identity = provenance.get("curriculum", {}).get("runtime_id")
                    if not identity:
                        raise ValueError("successful curriculum episode lacks runtime identity")
                    unique[episode["task"]].add(identity)
    for task in TASKS:
        games[task]["successful_effective_levels"] = sorted(unique[task])
    return {
        "split": "validation", "mode": "stochastic", "seed": seed, "rollouts": rollout,
        "initial_trainable_sha256": initial["initial_trainable_sha256"],
        "styles": {
            name: normalize_evaluation(run / folder / f"rollout_{rollout:04d}.json")
            for name, folder in (("canonical", "evaluations"), ("pixel_art", "evaluations_pixel"))
        },
        "train": {"games": games},
    }


def _no_progress_at_two_evaluations(run):
    """Only stop after the registered full CP64/128 checkpoints both exist."""
    run = Path(run)
    for folder in ("evaluations", "evaluations_pixel"):
        files = [run / folder / f"rollout_{n:04d}.json" for n in (0, 64, 128)]
        if not all(path.exists() for path in files):
            return False
        scores = [normalize_evaluation(path)["successes"] for path in files]
        if any(value > scores[0] for value in scores[1:]):
            return False
    with (run / "training.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            state = row.get("task_curriculum", {})
            if any(task.get("stage", 0) > 0
                   for task in state.get("tasks", {}).values()):
                return False
    return True


def run_policy_job(config, output, *, expected_initial=None, full=False):
    """Fresh run with graceful signals and the registered all-game stall rule."""
    from carryrl.training.sequence import train_sequence

    output = Path(output)
    stage_limit = os.environ.get("CARRYRL_STAGE_MAX_SECONDS")
    if stage_limit is not None:
        limit = float(stage_limit)
        reserve = 1200 if config.eval_all_levels else 300
        if limit <= reserve:
            raise ValueError("policy stage has no budget after save/evaluation reserve")
        config = replace(config, max_wall_seconds=min(config.max_wall_seconds, limit - reserve))
    requested = False
    checked = None
    stop_reason = None

    def request_stop(_signum, _frame):
        nonlocal requested, stop_reason
        requested, stop_reason = True, "supervisor_requested_stop"

    def stop():
        nonlocal checked, stop_reason
        if checked is None:
            checked = inspect_initial_checkpoint(
                output / "checkpoints/rollout_0000", expected=expected_initial)
            atomic_json(output / "initial_identity.json", checked)
        if full and _no_progress_at_two_evaluations(output):
            stop_reason = "registered_no_game_or_stage_progress_at_64_and_128"
            return True
        return requested

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        summary = train_sequence(config, output, stop_requested=stop)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    # The callback is not reached if the entire wall budget is spent on CP0 evaluation.
    if checked is None:
        checked = inspect_initial_checkpoint(output / "checkpoints/rollout_0000",
                                             expected=expected_initial)
        atomic_json(output / "initial_identity.json", checked)
    final_path = Path(_read(output / "latest.json")["checkpoint"])
    import torch

    final = torch.load(final_path / "state.pt", map_location="cpu", mmap=True, weights_only=True)
    frozen_unchanged = _state_hash(final["agent_state"], frozen=True) == checked["frozen_sha256"]
    trainable_changed = (_state_hash(final["agent_state"], frozen=False)
                         != checked["initial_trainable_sha256"])
    if not frozen_unchanged:
        raise RuntimeError("policy training modified frozen encoder weights")
    planned_stop = (full and summary["rollouts"] >= 128 and stop_reason ==
                    "registered_no_game_or_stage_progress_at_64_and_128")
    report = {"status": "completed" if planned_stop else summary["status"],
              "trainer_status": summary["status"], "registered_early_stop": planned_stop,
              "rollouts": summary["rollouts"],
              "stop_reason": stop_reason, "initial": checked,
              "frozen_unchanged": frozen_unchanged, "trainable_changed": trainable_changed,
              "final_checkpoint": str(final_path), "final_checkpoint_exists": True,
              "optimizer_steps": summary["optimizer_steps_this_process"]}
    atomic_json(output / "cycle_job.json", report)
    if config.eval_all_levels and summary["rollouts"]:
        atomic_json(output / "selection_report.json", policy_selection_report(
            output, rollout=summary["rollouts"], initial=checked, seed=config.seed))
        baseline = policy_selection_report(output, rollout=0, initial=checked, seed=config.seed)
        atomic_json(output / "initial_selection_report.json", baseline)
    return report
