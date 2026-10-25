"""Collect and replay TRAIN/validation scene probes without any model or optimizer."""

import argparse
import time
from pathlib import Path

from carryrl.envs.atomic_suite import TASKS
from carryrl.rewards.history_suite_data import collect_history_suite, load_history_suite
from carryrl.training.run import atomic_json, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--levels-per-split", type=int, default=2)
    parser.add_argument("--level-bank-seed", type=int, default=20261008)
    parser.add_argument("--seed", type=int, default=8176)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--goal-choice-utility", choices=("completion_only",))
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("scene suite output already exists")
    output, started, sources = args.output.resolve(), time.monotonic(), provenance()

    def progress(event):
        atomic_json(output / "status.json", {"status": "collecting", "updates": 0, **event})
        print(
            f"{event['task']}/{event['split']}/{event['strategy']}: {event['episodes']} episodes",
            flush=True,
        )

    try:
        report = collect_history_suite(
            output,
            tasks=args.tasks,
            levels_per_split=args.levels_per_split,
            bank_seed=args.level_bank_seed,
            seed=args.seed,
            max_steps=args.max_steps,
            goal_choice_utility=args.goal_choice_utility,
            on_progress=progress,
        )
        atomic_json(output / "status.json", {"status": "replaying", "updates": 0})
        corpus = load_history_suite(output, expected_sha256=report["corpus_sha256"])
        # Bounded first/full prefix checks complement full frame/action/label replay.
        for index, episode in enumerate(corpus.episodes):
            for step in (0, len(episode["frame_indices"]) - 1):
                inputs = corpus.prefix(index, step)
                if (
                    len(inputs["frames"]) != step + 1
                    or len(inputs["executed_previous_actions"]) != step + 1
                    or set(inputs) != {"frames", "executed_previous_actions", "goal"}
                ):
                    raise ValueError("invalid scene suite prefix fields or time range")
        report.update(loader_replay_verified=True, seconds=time.monotonic() - started)
        atomic_json(output / "audit.json", report)
        atomic_json(
            output / "metadata.json",
            {
                **sources,
                "model_loaded": False,
                "updates": 0,
                "corpus_sha256": report["corpus_sha256"],
            },
        )
        atomic_json(
            output / "status.json",
            {
                "status": "completed",
                "episodes": report["episodes"],
                "updates": 0,
                "corpus_sha256": report["corpus_sha256"],
                "ppo_eligible": False,
            },
        )
        return 0
    except BaseException as error:
        if output.exists():
            atomic_json(
                output / "status.json",
                {
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "updates": 0,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
