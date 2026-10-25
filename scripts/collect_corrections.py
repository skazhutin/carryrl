"""Collect canonical TRAIN corrections from an explicitly selected frozen checkpoint."""

import argparse
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from carryrl.envs.atomic import atomic_level_banks
from carryrl.envs.atomic_suite import TASKS
from carryrl.envs.runtime_identity import (
    allowed_runtime_train_indices,
    atomic_runtime_level_id,
    runtime_heldout_collisions,
)
from carryrl.training.imitation import (
    CORRECTION_SOURCE,
    collect_scripted_corrections,
    collect_scripted_demonstrations,
    demonstration_episode_digest,
    load_correction_replacement_bundle,
    validate_dataset_runtime_holdout,
)
from carryrl.training.run import atomic_json, file_sha256, provenance
from carryrl.training.sequence import FrozenFrameCache, load_sequence_checkpoint, tensor_hash
from carryrl.training.sequence_config import SequenceRunConfig


def _checkpoint_hashes(checkpoint):
    return {str(path.relative_to(checkpoint)): file_sha256(path)
            for path in sorted(checkpoint.rglob("*")) if path.is_file()}


def _prepare_output(output, checkpoint):
    requested_output = output
    output, checkpoint = output.resolve(), checkpoint.resolve(strict=True)
    roots = [parent.parent for parent in checkpoint.parents
             if parent.name in {"checkpoints", "checkpoints_by_epoch"}]
    source = roots[-1] if roots else checkpoint
    if output == source or source in output.parents or output in checkpoint.parents:
        raise ValueError("collection output must be outside the source run/checkpoint")
    if requested_output.is_symlink():
        raise FileExistsError("collection output must not be an existing symlink")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _preflight(config, tasks, episodes, seed, fresh_seed, original_seeds):
    """Refuse an index schedule the existing replacement importer would reinterpret."""
    selected = []
    for task in tasks:
        allowed = allowed_runtime_train_indices(task, fresh_seed, 1.0)
        actual = [allowed[(seed + index) % len(allowed)] for index in range(episodes)]
        expected = [(seed + index) % 64 for index in range(episodes)]
        if actual != expected:
            raise ValueError(f"{task}: safe collector indices {actual} differ from the "
                             f"replacement loader schedule {expected}; no policy loaded")
        bank = atomic_level_banks(task, fresh_seed)["train"]
        original_train = {}
        for original_seed in original_seeds:
            for index, level in enumerate(atomic_level_banks(task, original_seed)["train"]):
                runtime_id = atomic_runtime_level_id(task, level, 1.0)
                original_train.setdefault(runtime_id, []).append({
                    "level_bank_seed": original_seed, "level_index": index})
        for index, level_index in enumerate(actual):
            for original_seed in (*original_seeds, fresh_seed):
                matches = runtime_heldout_collisions(task, bank[level_index], 1.0,
                    original_seed, heldout_difficulties=(config.evaluation_difficulty
                        if config.evaluation_difficulty is not None else 1.0,))
                if matches:
                    raise ValueError(f"{task} level {level_index} overlaps held-out mechanics; "
                                     "no policy loaded")
            runtime_id = atomic_runtime_level_id(task, bank[level_index], 1.0)
            selected.append({"task": task, "level_index": level_index,
                             "seed": seed + 10000 * TASKS.index(task) + index,
                             "runtime_level_id": runtime_id,
                             "original_train_matches": original_train.get(runtime_id, [])})
    return selected


def _train_novelty(selected):
    distinct = {row["runtime_level_id"] for row in selected}
    repeated = {row["runtime_level_id"] for row in selected if row["original_train_matches"]}
    repeated_episodes = sum(bool(row["original_train_matches"]) for row in selected)
    return {"selected_episodes": len(selected), "distinct_selected_runtime_levels": len(distinct),
            "distinct_new_runtime_levels_vs_original_train": len(distinct - repeated),
            "distinct_repeated_original_train_runtime_levels": len(repeated),
            "new_runtime_episodes_vs_original_train": len(selected) - repeated_episodes,
            "repeated_original_train_episodes": repeated_episodes,
            "scope": "full difficulty selected mechanics versus declared original TRAIN banks",
            "interpretation": "new bank seed does not imply novel mechanics; TRAIN repeats allowed"}


def _write_frames(output, dataset):
    """Write the public bundle RGB array without a stacked allocation."""
    frames = np.lib.format.open_memmap(output / "frames.npy", mode="w+", dtype=np.uint8,
                                      shape=(len(dataset.frames), 96, 96, 3))
    try:
        for index, frame in enumerate(dataset.frames):
            frames[index] = frame
        frames.flush()
    finally:
        del frames


def _verify_serialization(output, dataset):
    rows = json.loads((output / "trajectories.json").read_text())
    expected = [episode.to_dict() for episode in dataset.episodes]
    if rows["episodes"] != expected:
        raise ValueError("exported episode JSON differs from collected trajectories")
    frames = np.load(output / "frames.npy", mmap_mode="r", allow_pickle=False)
    if frames.dtype != np.uint8 or frames.shape != (len(dataset.frames), 96, 96, 3):
        raise ValueError("exported RGB shape or dtype differs")
    for index, original in enumerate(dataset.frames):
        if not np.array_equal(frames[index], original):
            raise ValueError("exported RGB differs from collected frames")


def _bundle_identity(output):
    files = {name: {"bytes": (output / name).stat().st_size,
                    "sha256": file_sha256(output / name)}
             for name in ("coverage.json", "frames.npy", "trajectories.json")}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    return digest, files


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True,
                        help="Explicit configured task subset; no automatic priorities")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--episodes-per-task", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8176)
    parser.add_argument("--expert-probability", type=float, default=0.0,
                        help="Must be 0: importer admits deterministic learner-only corrections")
    parser.add_argument("--learner-prefix-steps", type=int, default=0,
                        help="Must be 0 for the current public replacement importer")
    parser.add_argument("--fresh-train-bank-seed", type=int, required=True)
    parser.add_argument("--original-bank-seeds", nargs="*", type=int, default=[])
    parser.add_argument("--max-frame-mib", type=int, default=256)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    config = replace(SequenceRunConfig.load(args.config), device=args.device)
    tasks = tuple(args.tasks)
    if not tasks or len(set(tasks)) != len(tasks) or not set(tasks).issubset(config.tasks):
        raise ValueError("tasks must be a unique explicit subset of configured tasks")
    if (not 1 <= args.episodes_per_task <= 64 or not 1 <= args.max_frame_mib <= 256
            or not 0 <= args.seed < 2**32 - 160000 - args.episodes_per_task
            or not math.isfinite(args.expert_probability)
            or not 0 <= args.expert_probability <= 1
            or not 0 <= args.learner_prefix_steps <= config.max_episode_steps):
        raise ValueError("invalid correction count, seed, mixture, prefix or RGB budget")
    if (args.expert_probability != 0.0 or args.learner_prefix_steps != 0):
        raise ValueError("public replacement importer requires expert probability 0 and prefix 0")
    if (not 0 < args.fresh_train_bank_seed < 2**32
            or args.fresh_train_bank_seed == config.level_bank_seed
            or any(not 0 <= seed < 2**32 for seed in args.original_bank_seeds)):
        raise ValueError("fresh bank must be a distinct positive uint32; original banks uint32")
    original_seeds = list(dict.fromkeys((0, 20261008, config.level_bank_seed,
                                        *args.original_bank_seeds)))
    if args.fresh_train_bank_seed in original_seeds:
        raise ValueError("fresh bank must differ from all original banks")
    if config.env_image_size != 96:
        raise ValueError("canonical correction collection requires configured 96x96 RGB")
    if args.device == "mps" and (not torch.backends.mps.is_available()
            or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"):
        raise RuntimeError("MPS must be available with implicit CPU fallback disabled")
    selected = _preflight(config, tasks, args.episodes_per_task, args.seed,
                          args.fresh_train_bank_seed, original_seeds)
    checkpoint = args.checkpoint.resolve(strict=True)
    before = _checkpoint_hashes(checkpoint)
    identity = json.loads((checkpoint / "manifest.json").read_text())
    output = _prepare_output(args.output, checkpoint)
    frame_budget = args.max_frame_mib * 1024**2
    settings = {"tasks": list(tasks), "episodes_per_task": args.episodes_per_task,
                "seed": args.seed, "expert_probability": args.expert_probability,
                "learner_prefix_steps": args.learner_prefix_steps,
                "difficulty": 1.0, "level_bank_seed": args.fresh_train_bank_seed,
                "original_bank_seeds": original_seeds,
                "max_steps": config.max_episode_steps, "image_size": 96,
                "max_frame_bytes": frame_budget, "selected_indices": selected}
    metadata = {"kind": "frozen_train_correction_collection", "split": "train",
                "training_updates": 0, "optimizer_steps": 0, "checkpoint": str(checkpoint),
                "checkpoint_identity": identity, "checkpoint_files_sha256_before": before,
                "config": config.to_dict(), "settings": settings,
                "train_level_novelty": _train_novelty(selected),
                "settings_status": "actual collection choices, not project requirements",
                "created_utc": datetime.now(UTC).isoformat(), **provenance()}
    atomic_json(output / "metadata.json", metadata)
    threads, started = torch.get_num_threads(), time.monotonic()
    agent = reward = None
    modes = None
    try:
        torch.set_num_threads(config.torch_threads)
        agent, optimizer, reward, state = load_sequence_checkpoint(
            checkpoint, config, evaluation_only=True, load_optimizer=False)
        if optimizer is not None:
            raise RuntimeError("frozen collection unexpectedly loaded an optimizer")
        revision = state["weights_revision"]
        del optimizer, state
        modes = agent.training, reward.training
        agent.eval()
        reward.eval()
        hashes = {"actor": tensor_hash(agent), "reward": tensor_hash(reward)}
        if (hashes["actor"] != identity["agent_sha256"]
                or hashes["reward"] != identity["reward_sha256"]
                or revision != identity["weights_revision"]):
            raise ValueError("loaded tensors/revision differ from selected checkpoint")
        cache = (FrozenFrameCache(agent, config.vision_cache_frames)
                 if config.vision_cache_frames else None)

        def progress(event):
            atomic_json(output / "status.json", {"status": "collecting", **event})
            print(json.dumps(event), flush=True)

        with torch.no_grad():
            corrections = collect_scripted_corrections(
                agent, args.episodes_per_task, tasks=tasks, seed=args.seed, difficulty=1.0,
                image_size=96, max_steps=config.max_episode_steps,
                expert_probability=args.expert_probability,
                learner_prefix_steps=args.learner_prefix_steps,
                level_bank_seed=args.fresh_train_bank_seed,
                episode_instructions=config.episode_instructions,
                max_frame_bytes=frame_budget, feature_cache=cache, on_episode=progress)
            dataset = collect_scripted_demonstrations(args.episodes_per_task,
                tasks=tasks, seed=args.seed, difficulty=1.0, image_size=96,
                max_steps=config.max_episode_steps, level_bank_seed=args.fresh_train_bank_seed,
                episode_instructions=config.episode_instructions,
                max_frame_bytes=frame_budget, on_episode=progress)
            dataset.extend(corrections)
            del corrections
        validate_dataset_runtime_holdout(dataset, level_bank_seed=config.level_bank_seed,
            evaluation_difficulty=(config.evaluation_difficulty
                                   if config.evaluation_difficulty is not None else 1.0))
        if len(dataset.episodes) != 2 * len(tasks) * args.episodes_per_task:
            raise ValueError("collector returned an incomplete correction cohort")
        _write_frames(output, dataset)
        atomic_json(output / "trajectories.json", {
            "frame_array": "frames.npy", "frame_shape": [96, 96, 3],
            "frame_count": len(dataset.frames), "raw_frame_bytes": dataset.frame_bytes,
            "episodes": [episode.to_dict() for episode in dataset.episodes]})
        _verify_serialization(output, dataset)
        after_hashes = {"actor": tensor_hash(agent), "reward": tensor_hash(reward)}
        after = _checkpoint_hashes(checkpoint)
        if after_hashes != hashes or after != before:
            raise RuntimeError("collection modified source checkpoint or policy/reward tensors")
        if cache is not None:
            cache.assert_unchanged()
        coverage = {"status": "complete", "episodes": [episode.to_dict()
                    for episode in dataset.episodes if episode.source == CORRECTION_SOURCE],
                    "policy_and_reward_frozen": True,
                    "frozen_hashes_before": hashes, "frozen_hashes_after": after_hashes,
                    "loaded_tensors_match_preregistered_identity": True,
                    "overlap_guard": {"passed": True, "collisions": [],
                        "selected": selected, "original_bank_seeds": original_seeds},
                    "preregistration": {"expected_checkpoint_identity": identity,
                        "checkpoint": str(checkpoint), "fresh_train_bank_seed":
                        args.fresh_train_bank_seed, "correction_collection_seed": args.seed,
                        "episodes_per_task": args.episodes_per_task,
                        "max_steps_per_episode": config.max_episode_steps,
                        "difficulty": 1.0, "native_rgb_size": 96,
                        "expert_probability": args.expert_probability,
                        "learner_prefix_steps": args.learner_prefix_steps,
                        "training_updates": 0, "optimizer_steps": 0,
                        "heldout_policy_rollouts": 0, "test_outcomes_read": False,
                        "tasks": list(tasks)}}
        atomic_json(output / "coverage.json", coverage)
        digest, files = _bundle_identity(output)
        expected_digests = {task: [demonstration_episode_digest(dataset, episode)
            for episode in dataset.episodes
            if episode.task == task and episode.source == CORRECTION_SOURCE] for task in tasks}
        rgb_bytes, episode_count = dataset.frame_bytes, len(dataset.episodes)
        cache_statistics = cache.statistics() if cache is not None else None
        # Replay authenticates RGB without a policy. Release the large actor and original
        # corpus before the importer reconstructs its own bounded dataset.
        agent.train(modes[0])
        reward.train(modes[1])
        modes = None
        del dataset, cache
        agent = reward = None
        gc.collect()
        imported, import_report = load_correction_replacement_bundle(
            output, expected_sha256=digest, correction_tasks=tasks,
            episodes_per_task=args.episodes_per_task, seed=args.seed, difficulty=1.0,
            max_steps=config.max_episode_steps, level_bank_seed=config.level_bank_seed,
            episode_instructions=config.episode_instructions, source_identity=identity,
            initial_agent_sha256=hashes["actor"], reward_sha256=hashes["reward"],
            max_frame_bytes=frame_budget)
        for task in tasks:
            actual = [demonstration_episode_digest(imported[task], episode)
                      for episode in imported[task].episodes]
            if expected_digests[task] != actual:
                raise ValueError("public importer changed correction content or provenance")
        replacement = {"validated": True, "report": import_report,
            "corpus_slot_replacement_validated": False,
            "scope": "bundle import only; existing corpus counts/producer/seed not audited"}
        if (_checkpoint_hashes(checkpoint) != before
                or _bundle_identity(output)[0] != digest):
            raise RuntimeError("source or bundle changed during import validation")
        metadata.update(checkpoint_files_sha256_after=after, frozen_hashes_before=hashes,
            frozen_hashes_after=after_hashes, weights_revision_before=revision,
            weights_revision_after=revision, policy_and_reward_frozen=True,
            bundle_sha256=digest, bundle_files=files, replacement_import=replacement,
            serialization_verified=True, rgb_bytes=rgb_bytes,
            feature_cache=cache_statistics,
            memory_accounting="RGB and cache bytes are array storage; process RSS not measured",
            disk_artifact_bytes={path.name: path.stat().st_size for path in output.iterdir()
                                 if path.is_file() and path.name not in {"metadata.json",
                                                                       "status.json"}})
        atomic_json(output / "metadata.json", metadata)
        atomic_json(output / "status.json", {"status": "completed", "training_updates": 0,
            "optimizer_steps": 0, "weights_revision": revision, "bundle_sha256": digest,
            "checkpoint_files_unchanged": True, "policy_and_reward_frozen": True,
            "episodes": episode_count,
            "correction_episodes": len(tasks) * args.episodes_per_task,
            "supporting_teacher_episodes": len(tasks) * args.episodes_per_task,
            "seconds": time.monotonic() - started,
            "replacement_import_validated": replacement["validated"]})
    except BaseException as error:
        atomic_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        if modes is not None:
            agent.train(modes[0])
            reward.train(modes[1])
        torch.set_num_threads(threads)
    print(json.dumps({"output": str(output), "bundle_sha256": digest,
                      "replacement_import_validated": replacement["validated"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
