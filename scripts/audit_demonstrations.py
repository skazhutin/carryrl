"""Audit train-only RGB teachers and admission rules without loading a policy model."""

import argparse
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from carryrl.envs.atomic import atomic_level_banks
from carryrl.envs.atomic_suite import SCORING_VERSION, SUITE_VERSION, TASKS
from carryrl.envs.runtime_identity import allowed_runtime_train_indices, atomic_runtime_level_id
from carryrl.evaluation.quality import summarize_episode_quality
from carryrl.training.imitation import (
    collect_scripted_demonstrations,
    demonstration_episode_digest,
)
from carryrl.training.run import atomic_json, provenance


def audit_demonstrations(*, tasks=TASKS, level_bank_seed=20261008, difficulty=1.0,
                         seed=8176, levels_per_task=None, max_steps=80,
                         max_frame_bytes=256 * 1024**2, on_progress=None):
    """Audit every allowed train index by default; retain RGB for only one game.

    A game whose collector rejects an episode is reported as failed. Partial
    results from that game are not relabelled as an accepted corpus. The report
    is diagnostic metadata and cannot be loaded as a demonstration dataset.
    """
    tasks = tuple(tasks)
    if (not tasks or len(set(tasks)) != len(tasks) or any(task not in TASKS for task in tasks)
            or type(level_bank_seed) is not int or not 0 <= level_bank_seed < 2**32
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(max_steps) is not int or not 1 <= max_steps <= 80
            or type(max_frame_bytes) is not int or max_frame_bytes < 1
            or isinstance(difficulty, bool) or not isinstance(difficulty, (int, float))
            or not math.isfinite(difficulty) or not 0 <= difficulty <= 1
            or levels_per_task is not None and (
                type(levels_per_task) is not int or levels_per_task < 1)):
        raise ValueError("invalid teacher audit configuration")
    allowed = {task: allowed_runtime_train_indices(task, level_bank_seed, difficulty)
               for task in tasks}
    counts = {task: len(indices) if levels_per_task is None else levels_per_task
              for task, indices in allowed.items()}
    if any(counts[task] > len(indices) for task, indices in allowed.items()):
        raise ValueError("audit cannot claim more distinct train indices than the allowed bank")
    started, reports, records, peak_rgb_bytes = time.monotonic(), {}, [], 0
    for task in tasks:
        if on_progress:
            on_progress({"event": "task_started", "task": task, "episodes_requested": counts[task]})
        dataset = None
        try:
            dataset = collect_scripted_demonstrations(
                tasks=(task,), episodes_per_task=counts[task], seed=seed,
                difficulty=difficulty, image_size=96, max_steps=max_steps,
                level_bank_seed=level_bank_seed, max_frame_bytes=max_frame_bytes)
            bank = atomic_level_banks(task, level_bank_seed)["train"]
            rows = []
            for episode in dataset.episodes:
                if (episode.quality is None
                        or episode.quality_source != "actual_executed_trajectory"):
                    raise ValueError("audit requires actually measured trajectory quality")
                rows.append({"task": task, "source": episode.source, "split": episode.split,
                             "seed": episode.seed, "level_index": episode.level_index,
                             "level_id": episode.level_id,
                             "runtime_level_id": atomic_runtime_level_id(
                                 task, bank[episode.level_index], difficulty),
                             "steps": len(episode), "outcome": dict(episode.outcome),
                             "quality": dict(episode.quality),
                             "quality_source": episode.quality_source,
                             "trace_sha256": demonstration_episode_digest(dataset, episode)})
            if (len(rows) != counts[task]
                    or len({row["level_index"] for row in rows}) != len(rows)):
                raise ValueError("audit collector did not cover the declared distinct indices")
            rgb_bytes = dataset.frame_bytes
            peak_rgb_bytes = max(peak_rgb_bytes, rgb_bytes)
            reports[task] = {"status": "passed", "episodes_requested": counts[task],
                             "episodes_accepted": len(rows),
                             "distinct_level_ids": len({row["level_id"] for row in rows}),
                             "distinct_runtime_levels": len({row["runtime_level_id"]
                                                             for row in rows}),
                             "rgb_payload_bytes": rgb_bytes,
                             "quality": summarize_episode_quality(rows)}
            records.extend(rows)
        except (RuntimeError, ValueError, TypeError, MemoryError) as error:
            reports[task] = {"status": "failed", "episodes_requested": counts[task],
                             "error_type": type(error).__name__, "error": str(error)}
        finally:
            del dataset
        if on_progress:
            on_progress({"event": "task_completed", "task": task,
                         "status": reports[task]["status"]})
    failed = [task for task, report in reports.items() if report["status"] == "failed"]
    return {"kind": "train_rgb_teacher_quality_audit", "split": "train",
            "status": "failed" if failed else "completed", "failed_tasks": failed,
            "policy_model_loaded": False, "updates": 0, "audit_only": True,
            "rgb_saved": False, "training_corpus_saved": False,
            "suite_version": SUITE_VERSION, "scoring_version": SCORING_VERSION,
            "created_utc": datetime.now(UTC).isoformat(), "seconds": time.monotonic() - started,
            "config": {"tasks": list(tasks), "seed": seed, "difficulty": difficulty,
                       "level_bank_seed": level_bank_seed, "max_steps": max_steps,
                       "image_size": 96, "levels_per_task": levels_per_task,
                       "max_frame_bytes_per_game": max_frame_bytes},
            "episodes_requested": sum(counts.values()),
            "accepted_episodes_in_completed_games": len(records),
            "peak_retained_rgb_payload_bytes": peak_rgb_bytes,
            "memory_scope": "RGB payload of one completed game; not process RSS or whole Mac RAM",
            "interpretation": "Teacher admission and trajectory quality, not learned-policy wins. "
                              "Zero audited waste does not certify an optimum outside navigation.",
            "tasks": reports, "records": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--level-bank-seed", type=int, default=20261008)
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=8176)
    parser.add_argument("--levels-per-task", type=int,
                        help="default: every allowed train-bank index, with no repetition")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--frame-budget-mib", type=int, default=256)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "metadata.json", {"command": "audit_teacher", "updates": 0,
                "policy_model_loaded": False, **provenance()})

    def progress(event):
        state = {**event, "status": "running"}
        if "status" in event:
            state["task_status"] = event["status"]
        atomic_json(output / "status.json", state)
        print(f"{event['task']}: {event['event']} {event.get('status', '')}", flush=True)

    try:
        report = audit_demonstrations(
            tasks=args.tasks, level_bank_seed=args.level_bank_seed, difficulty=args.difficulty,
            seed=args.seed, levels_per_task=args.levels_per_task, max_steps=args.max_steps,
            max_frame_bytes=args.frame_budget_mib * 1024**2, on_progress=progress)
        atomic_json(output / "audit.json", report)
        atomic_json(output / "status.json", {"status": report["status"], "updates": 0,
                    "failed_tasks": report["failed_tasks"]})
        return 0 if report["status"] == "completed" else 1
    except BaseException as error:
        atomic_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
