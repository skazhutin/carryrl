"""Audit a pinned frozen TRAIN correction bundle without loading learned models."""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from carryrl.agent.actions import CODEC_VERSION
from carryrl.evaluation.correction_coverage import (
    describe_correction_coverage,
    summarize_correction_coverage,
)
from carryrl.evaluation.rescue import diagnose_correction_rescue
from carryrl.training.checkpoint_storage import checkpoint_read_lock
from carryrl.training.imitation import load_correction_replacement_bundle
from carryrl.training.run import atomic_json, file_sha256, provenance
from carryrl.training.sequence_config import SequenceRunConfig


@dataclass(frozen=True)
class CorrectionSource:
    directory: Path
    checkpoint: Path
    config: SequenceRunConfig
    datasets: dict
    authentication: dict
    file_hashes: dict[str, str]

    def assert_unchanged(self):
        with checkpoint_read_lock(self.checkpoint):
            if any(file_sha256(Path(path)) != expected
                   for path, expected in self.file_hashes.items()):
                raise RuntimeError("correction audit source or checkpoint changed")


def prepare_source(directory, checkpoint, expected_sha256):
    """Replay both correction and supporting teacher histories before publication."""
    directory = Path(directory).resolve(strict=True)
    checkpoint = Path(checkpoint).resolve(strict=True)
    pins = {}

    def pin(path):
        pins[str(path)] = file_sha256(path)
        return path

    metadata = json.loads(pin(directory / "metadata.json").read_text())
    status = json.loads(pin(directory / "status.json").read_text())
    if (metadata.get("kind") != "frozen_train_correction_collection"
            or metadata.get("split") != "train" or metadata.get("training_updates") != 0
            or metadata.get("optimizer_steps") != 0
            or metadata.get("policy_and_reward_frozen") is not True
            or status.get("status") != "completed" or status.get("training_updates") != 0
            or status.get("optimizer_steps") != 0
            or status.get("checkpoint_files_unchanged") is not True
            or status.get("policy_and_reward_frozen") is not True
            or status.get("bundle_sha256") != expected_sha256
            or metadata.get("bundle_sha256") != expected_sha256
            or Path(metadata["checkpoint"]).resolve() != checkpoint):
        raise ValueError("correction audit requires a completed frozen TRAIN source")
    checkpoint_hashes = metadata["checkpoint_files_sha256_before"]
    if (checkpoint_hashes != metadata["checkpoint_files_sha256_after"]
            or not {"manifest.json", "state.pt"}.issubset(checkpoint_hashes)):
        raise ValueError("correction audit requires unchanged pinned state and manifest")
    with checkpoint_read_lock(checkpoint):
        for name, expected in checkpoint_hashes.items():
            path = (checkpoint / name).resolve(strict=True)
            if not path.is_relative_to(checkpoint) or file_sha256(path) != expected:
                raise ValueError("correction audit checkpoint SHA256 differs")
            pins[str(path)] = expected
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        if (manifest.get("sha256") != checkpoint_hashes["state.pt"]
                or manifest.get("codec_version") != CODEC_VERSION
                or manifest != metadata["checkpoint_identity"]):
            raise ValueError("correction audit checkpoint identity differs")
    config = SequenceRunConfig.from_dict(metadata["config"])
    settings = metadata["settings"]
    if (config.env_image_size != 96 or not 1 <= config.max_episode_steps <= 80
            or config.visual_config is not None and config.visual_config.mode != "canonical"
            or settings["max_steps"] != config.max_episode_steps
            or settings["difficulty"] != 1.0 or settings["image_size"] != 96
            or not set(settings["tasks"]).issubset(config.tasks)):
        raise ValueError("correction audit requires canonical96 full1 bounded ordinary data")
    for name in ("coverage.json", "trajectories.json", "frames.npy"):
        pin(directory / name)
    datasets, authentication = load_correction_replacement_bundle(directory,
        expected_sha256=expected_sha256, correction_tasks=tuple(settings["tasks"]),
        episodes_per_task=settings["episodes_per_task"], seed=settings["seed"], difficulty=1.0,
        max_steps=config.max_episode_steps, level_bank_seed=config.level_bank_seed,
        episode_instructions=config.episode_instructions, source_identity=manifest,
        initial_agent_sha256=manifest["agent_sha256"], reward_sha256=manifest["reward_sha256"])
    source = CorrectionSource(directory, checkpoint, config, datasets, authentication, pins)
    source.assert_unchanged()
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchors", nargs="+", type=int, default=[0, 20, 40, 60])
    parser.add_argument("--stall-window", type=int, default=8)
    parser.add_argument("--early-steps", type=int, default=20)
    args = parser.parse_args(argv)
    if (0 not in args.anchors or len(set(args.anchors)) != len(args.anchors)
            or any(not 0 <= anchor < 80 for anchor in args.anchors)
            or not 1 <= args.stall_window <= 80 or not 1 <= args.early_steps <= 80):
        parser.error("anchors need unique steps0..79 including0; windows need1..80")
    output = args.output.resolve()
    if output.exists() or args.output.is_symlink():
        raise FileExistsError("correction audit output already exists")
    roots = (args.bundle.resolve(strict=True), args.checkpoint.resolve(strict=True))
    if any(output.is_relative_to(root) for root in roots):
        raise ValueError("correction audit output must be outside source and checkpoint")
    source = prepare_source(args.bundle, args.checkpoint, args.bundle_sha256)
    if any(anchor >= source.config.max_episode_steps for anchor in args.anchors):
        parser.error("anchor must precede the source deadline")
    entries = [(dataset, episode) for _, dataset in sorted(source.datasets.items())
               for episode in dataset.episodes]
    if not entries:
        raise ValueError("correction audit has no authenticated corrections")
    output.mkdir(parents=True, exist_ok=False)
    producer = provenance()
    metadata = {"kind": "train_correction_coverage_and_rescue", "training_updates": 0,
        "source_bundle": str(source.directory), "bundle_sha256": args.bundle_sha256,
        "source_checkpoint": str(source.checkpoint), "source_files_sha256": source.file_hashes,
        "authentication": source.authentication, "source_config": source.config.to_dict(),
        "coverage": "all_authenticated_corrections_in_bundle", "episodes": len(entries),
        "anchors": args.anchors, "stall_window": args.stall_window,
        "early_steps": args.early_steps, "policy_loaded": False,
        "reward_model_loaded": False, "optimizer_constructed": False,
        "interpretation": "Fixed TRAIN cohort observations and controller continuations. "
            "Not representative error rates, trained policy quality, or proof of impossibility. "
            "Original correction labels and data remain unchanged; no continuations imported.",
        **producer}
    atomic_json(output / "metadata.json", metadata)
    totals, games, published, coverage_rows = defaultdict(Counter), {}, [], []
    errors = 0
    try:
        for index, (dataset, episode) in enumerate(entries):
            atomic_json(output / "status.json", {"status": "running", "training_updates": 0,
                "completed": index, "total": len(entries), "task": episode.task})
            coverage = describe_correction_coverage(dataset, episode,
                stall_window=args.stall_window, early_steps=args.early_steps)
            result = diagnose_correction_rescue(source.config, dataset, episode,
                anchors=tuple(args.anchors), stall_window=args.stall_window)
            if result.report["source_demonstration_sha256"] != coverage["episode_content_sha256"]:
                raise ValueError("coverage and rescue authenticated different corrections")
            destination = output / "episodes" / f"{episode.task}_seed_{episode.seed}"
            destination.mkdir(parents=True)
            frames_path = destination / "suffix_frames.npz"
            payload = {f"{anchor}_frame_{number:03d}": frame
                       for anchor, frames in result.frames_by_anchor.items()
                       for number, frame in enumerate(frames)}
            np.savez_compressed(frames_path, **payload)
            report_path = destination / "report.json"
            atomic_json(report_path, {"coverage": coverage, "rescue": result.report,
                "suffix_frames_sha256": file_sha256(frames_path)})
            published.append({"episode_content_sha256": coverage["episode_content_sha256"],
                "report": str(report_path.resolve()), "report_sha256": file_sha256(report_path)})
            coverage_rows.append(coverage)
            for attempt in result.report["attempts"]:
                anchor, state = attempt["anchor_id"], attempt["status"]
                totals[anchor][state] += 1
                games.setdefault(episode.task, defaultdict(Counter))[anchor][state] += 1
                errors += int(state == "teacher_error")
            del payload, result
        source.assert_unchanged()
        if provenance()["source_and_lock_sha256"] != producer["source_and_lock_sha256"]:
            raise RuntimeError("correction audit producer source changed")
        report = {"status": "completed_with_teacher_errors" if errors else "completed",
            "training_updates": 0, "source_files_unchanged": True, "episodes": len(entries),
            "teacher_errors": errors, "by_anchor": dict(totals), "by_game": games,
            "coverage": summarize_correction_coverage(coverage_rows), "published": published,
            "interpretation": metadata["interpretation"]}
        atomic_json(output / "aggregate.json", report)
        atomic_json(output / "status.json", {key: report[key] for key in
            ("status", "training_updates", "episodes", "source_files_unchanged", "teacher_errors")})
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "interrupted" if isinstance(
            error, KeyboardInterrupt) else "failed", "training_updates": 0,
            "completed": len(published), "total": len(entries),
            "error_type": type(error).__name__, "error": str(error)})
        raise
    print(json.dumps({key: report[key] for key in
        ("status", "episodes", "training_updates", "teacher_errors", "by_anchor")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
