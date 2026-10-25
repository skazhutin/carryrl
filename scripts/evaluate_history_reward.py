"""Evaluate a pinned fitted auxiliary history head on frozen TRAIN/validation references."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from carryrl.evaluation.history_reward import DEFAULT_CONTROLS, evaluate_history_reward
from carryrl.rewards.history_features import load_history_features
from carryrl.rewards.history_suite_data import load_history_suite
from carryrl.training.history_reward import load_fitted_history_reward_checkpoint
from carryrl.training.run import atomic_json, file_sha256, provenance


def _sha256(value):
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("expected a lowercase SHA256 digest")
    return value


def _threads(value):
    number = int(value)
    if not 1 <= number <= 8:
        raise argparse.ArgumentTypeError("torch threads must be between 1 and 8")
    return number


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-sha256", type=_sha256, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", type=_sha256, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", type=_sha256, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--torch-threads", type=_threads, default=2)
    return parser


def _without_symlinks(path):
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("evaluation paths must not contain symlinks")
    return path.resolve()


def _prepare_paths(args):
    paths = {
        name: _without_symlinks(getattr(args, name))
        for name in ("corpus", "features", "checkpoint", "output")
    }
    for name in ("corpus", "features"):
        if not paths[name].is_dir():
            raise NotADirectoryError(f"{name} source must be an existing directory")
    if not paths["checkpoint"].is_file():
        raise FileNotFoundError("checkpoint source must be an existing file")
    output = paths["output"]
    # A checkpoint's containing directory is part of its immutable source run.
    sources = (paths["corpus"], paths["features"], paths["checkpoint"].parent)
    if any(
        output == source or source in output.parents or output in source.parents
        for source in sources
    ):
        raise ValueError("evaluation output must be outside source directories and their ancestors")
    if output.exists() or output.is_symlink():
        raise FileExistsError("evaluation output already exists")
    output.mkdir(parents=True, exist_ok=False)
    return paths


def main(argv=None):
    args = _parser().parse_args(argv)
    paths = _prepare_paths(args)
    output = paths["output"]
    threads, started = torch.get_num_threads(), time.monotonic()
    fixed = {
        "updates": 0,
        "optimizer_loaded": False,
        "fit_performed": False,
        "ppo_eligible": False,
        "device": "cpu",
        "split": args.split,
        "test_levels_played": False,
        "test_outcomes_read": False,
    }
    sources = {
        name: {"path": str(paths[name]), "sha256": getattr(args, f"{name}_sha256")}
        for name in ("corpus", "features", "checkpoint")
    }

    def progress(event):
        atomic_json(output / "status.json", {**event, **fixed, "status": "running"})

    try:
        atomic_json(output / "status.json", {**fixed, "status": "running"})
        metadata = {
            **provenance(),
            **fixed,
            "command": "evaluate_history_reward",
            "created_utc": datetime.now(UTC).isoformat(),
            "source_pins": sources,
            "torch_threads": args.torch_threads,
            "controls": list(DEFAULT_CONTROLS),
            "thresholds": {"events": 0.5, "terminal": 0.5, "validity": 0.5},
        }
        atomic_json(output / "metadata.json", metadata)
        torch.set_num_threads(args.torch_threads)
        corpus = load_history_suite(paths["corpus"], expected_sha256=args.corpus_sha256)
        features = load_history_features(
            paths["features"], corpus, expected_sha256=args.features_sha256
        )
        head, checkpoint_metadata = load_fitted_history_reward_checkpoint(
            paths["checkpoint"],
            expected_sha256=args.checkpoint_sha256,
            corpus_identity={"corpus_sha256": corpus.corpus_sha256},
            feature_identity=features.identity,
            event_names=corpus.event_names,
            device="cpu",
        )
        if checkpoint_metadata.get("checkpoint_sha256") != args.checkpoint_sha256:
            raise ValueError("loaded checkpoint identity differs from requested pin")
        fixed["checkpoint_fitted"] = True
        metadata.update(
            checkpoint_metadata=checkpoint_metadata,
            feature_identity=features.identity,
            checkpoint_fitted=True,
        )
        atomic_json(output / "metadata.json", metadata)
        report = evaluate_history_reward(
            head,
            corpus,
            features,
            split=args.split,
            controls=DEFAULT_CONTROLS,
            event_threshold=0.5,
            terminal_threshold=0.5,
            validity_threshold=0.5,
            on_progress=progress,
        )
        if file_sha256(paths["checkpoint"]) != args.checkpoint_sha256:
            raise RuntimeError("evaluation modified the pinned checkpoint file")
        report.update(
            checkpoint_sha256=args.checkpoint_sha256,
            source_pins=sources,
            optimizer_loaded=False,
            fit_performed=False,
            checkpoint_fitted=True,
        )
        atomic_json(output / "evaluation.json", report)
        atomic_json(
            output / "status.json",
            {
                **fixed,
                "status": "completed",
                "seconds": time.monotonic() - started,
                "checkpoint_file_unchanged": True,
                "source_sha256": report["source_sha256"],
                "cohort_sha256": report["cohort_sha256"],
            },
        )
    except BaseException as error:
        atomic_json(
            output / "status.json",
            {
                **fixed,
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        torch.set_num_threads(threads)
    print(json.dumps({"output": str(output), "status": "completed", **fixed}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
