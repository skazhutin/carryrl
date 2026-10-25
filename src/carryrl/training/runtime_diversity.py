"""Shared integration for the two explicit train-only runtime diversity axes."""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from carryrl.training.imitation import (
    DemonstrationDataset,
    SavedDemonstrationReuse,
    validate_dataset_runtime_holdout,
)
from carryrl.training.run import atomic_json, provenance
from carryrl.training.sequence_config import SequenceRunConfig

AXES = ("navigation_geometry", "reaction_dynamics")


@dataclass(frozen=True)
class RuntimeDiversityPairs:
    repeat: DemonstrationDataset
    varied: DemonstrationDataset
    manifest: dict

    def arm(self, mode):
        if mode not in ("repeat", "varied"):
            raise ValueError("runtime diversity arm must be repeat or varied")
        return getattr(self, mode)


def checked_combined_frame_bytes(base, arms, max_frame_bytes, config=None):
    """Count unique combined RGB while hashing only delta frames, before publication."""
    if type(max_frame_bytes) is not int or max_frame_bytes < 1:
        raise ValueError("runtime diversity max_frame_bytes must be positive")
    budget = min(base.max_frame_bytes, max_frame_bytes)
    if base.frame_bytes > budget:
        raise MemoryError("runtime diversity base already exceeds the combined RGB byte budget")
    combined = {}
    for mode, arm in arms.items():
        extra = {}
        for frame in arm.frames:
            key = frame.shape, hashlib.sha256(frame.tobytes()).digest()
            if key not in base._index:
                extra[key] = frame.nbytes
        combined[mode] = base.frame_bytes + sum(extra.values())
        if combined[mode] > budget:
            raise MemoryError("runtime diversity combined RGB byte budget exceeded: "
                              f"{mode} needs {combined[mode]} bytes, budget {budget}")
        if config is not None:
            from carryrl.training.sequence import sequence_evaluation_protocol

            validate_dataset_runtime_holdout(arm, level_bank_seed=config.level_bank_seed,
                evaluation_difficulty=sequence_evaluation_protocol(config)["difficulty"])
    return combined


def collect_runtime_diversity_pairs(axis, dataset, **kwargs):
    """Select an explicit supported axis, preserving the navigation manifest verbatim."""
    if axis == "navigation_geometry":
        from carryrl.training.navigation_geometry import collect_navigation_geometry_pairs

        pairs = collect_navigation_geometry_pairs(dataset, **kwargs)
        return RuntimeDiversityPairs(pairs.repeat, pairs.new_walls, pairs.manifest)
    if axis == "reaction_dynamics":
        from carryrl.training.reaction_dynamics import collect_reaction_dynamics_pairs

        return collect_reaction_dynamics_pairs(dataset, **kwargs)
    raise ValueError(f"runtime diversity axis must be one of {AXES}")


def prepare_saved_runtime_diversity(source_run, config, output, *, axis, pairs=8, seed=8176,
        max_frame_bytes=4096 * 1024**2, legacy_navigation=False):
    """Authenticate a saved corpus and publish only model-free delta arms."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("runtime diversity output already exists")
    if legacy_navigation and axis != "navigation_geometry":
        raise ValueError("legacy navigation publication requires navigation_geometry")
    source = SavedDemonstrationReuse.prepare(source_run, config=config,
                                             max_frame_bytes=max_frame_bytes)
    collected = collect_runtime_diversity_pairs(axis, source.dataset, max_pairs=pairs,
        seed=seed, max_steps=config.max_episode_steps, max_frame_bytes=max_frame_bytes,
        config=config)
    source.assert_unchanged()
    output.mkdir(parents=True, exist_ok=False)
    try:
        for arm in ("repeat", "varied"):
            dataset = collected.arm(arm)
            folder = "new_walls" if legacy_navigation and arm == "varied" else arm
            destination = output / folder
            destination.mkdir()
            atomic_json(destination / "demonstrations.json", dataset.provenance())
            np.savez_compressed(destination / "demonstration_frames.npz",
                **{f"frame_{index:06}": frame for index, frame in enumerate(dataset.frames)})
        report = {**collected.manifest, "status": "prepared_not_trained", "updates": 0,
            "source_run": str(source.directory), "source_files": source.files,
            "source_corpus_preserved": True, "config": config.to_dict(),
            "training_integration": "regenerate delta from same source, seed and pair count"}
        if not legacy_navigation:
            report["axis"] = axis
        # Producer code/dependencies remain separate from the legacy manifest contract.
        atomic_json(output / "metadata.json", {"kind": "runtime_diversity_preparation",
                                               "axis": axis, **provenance()})
        source.assert_unchanged()
        atomic_json(output / "manifest.json", report)
        atomic_json(output / "status.json", {"status": "prepared_not_trained", "updates": 0,
            "pairs": len(collected.varied.episodes), "source_corpus_preserved": True})
    except Exception as error:
        atomic_json(output / "status.json", {"status": "failed", "updates": 0,
            "error_type": type(error).__name__, "error": str(error)})
        raise
    return report


def preparation_main(argv=None, *, legacy_navigation=False):
    parser = argparse.ArgumentParser(description="Prepare runtime diversity deltas without a model")
    if not legacy_navigation:
        parser.add_argument("--axis", choices=AXES, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8176)
    parser.add_argument("--frame-budget-mib", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.frame_budget_mib < 1:
        parser.error("frame budget must be positive")
    report = prepare_saved_runtime_diversity(args.source_run, SequenceRunConfig.load(args.config),
        args.output, axis="navigation_geometry" if legacy_navigation else args.axis,
        pairs=args.pairs, seed=args.seed, max_frame_bytes=args.frame_budget_mib * 1024**2,
        legacy_navigation=legacy_navigation)
    print(f"prepared_not_trained: {report['pairs_collected']} equal-budget pairs", flush=True)
    return 0
