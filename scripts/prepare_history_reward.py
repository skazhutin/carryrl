"""Prepare and audit causal scene targets from an explicitly pinned TRAIN RGB bundle."""

import argparse
import time
from pathlib import Path

import torch

from carryrl.agent.actions import PhysicalAction, encode_actions
from carryrl.rewards.history_data import (
    export_history_reward_corpus,
    load_history_reward_corpus,
)
from carryrl.training.run import atomic_json, provenance
from carryrl.training.sequence_config import SequenceRunConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("scene history output already exists")
    output = args.output.resolve()
    config = SequenceRunConfig.load(args.source_config)
    started, sources = time.monotonic(), provenance()

    def progress(event):
        atomic_json(output / "status.json", {"status": "running", "updates": 0, **event})
        print(f"{event['task']}: {event['episodes_exported']} episodes", flush=True)

    try:
        report = export_history_reward_corpus(args.bundle, expected_sha256=args.bundle_sha256,
            source_config=config, output=output, on_progress=progress)
        corpus = load_history_reward_corpus(output, expected_sha256=report["corpus_sha256"])
        prefixes = 0
        for index, episode in enumerate(corpus.episodes):
            # Every prefix is independently bounded and receives only actual past
            # commands, even when the source teacher would have acted differently.
            executed = [PhysicalAction.from_dict(row) for row in episode["executed_actions"]]
            for step in range(len(episode["frame_indices"])):
                inputs = corpus.prefix(index, step)
                if (set(inputs) != {"frames", "executed_previous_actions", "goal"}
                        or len(inputs["frames"]) != step + 1
                        or not torch.equal(inputs["executed_previous_actions"],
                            encode_actions([PhysicalAction(), *executed[:step]]))):
                    raise ValueError("scene prefix input alignment mismatch")
                prefixes += 1
        report.update(loader_replay_verified=True, causal_prefixes_checked=prefixes,
                      seconds=time.monotonic() - started)
        atomic_json(output / "audit.json", report)
        atomic_json(output / "metadata.json", {**sources, "command": "prepare_history_reward",
            "model_loaded": False, "updates": 0, "source_config": str(args.source_config.resolve()),
            "source_bundle_sha256": args.bundle_sha256,
            "corpus_sha256": report["corpus_sha256"]})
        atomic_json(output / "status.json", {"status": "completed", "updates": 0,
            "episodes": report["episodes"], "causal_prefixes_checked": prefixes,
            "corpus_sha256": report["corpus_sha256"], "ppo_eligible": False})
        return 0
    except BaseException as error:
        if output.exists():
            atomic_json(output / "status.json", {"status": "interrupted"
                if isinstance(error, KeyboardInterrupt) else "failed", "updates": 0,
                "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
