"""Train-only RGB/action warmup, with frozen closed-loop validation and checkpoints."""

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from carryrl.dashboard.frozen_playback import link_matching_playbacks, write_frozen_playback
from carryrl.envs.atomic_suite import SUITE_VERSION, TASKS
from carryrl.envs.presentation import HUD_LAYOUTS, THEME_SPLITS, PresentationConfig
from carryrl.training.feature_store import (
    FrozenFeatureReuse,
    FrozenFeatureStore,
    FrozenFeatureWriter,
    feature_cache_identity,
)
from carryrl.training.imitation import (
    ImitationScreenText,
    SavedDemonstrationReuse,
    audit_correction_replacement,
    augment_demonstrations,
    collect_memory_cue_counterparts,
    collect_native_scripted_demonstrations,
    collect_scripted_corrections,
    collect_scripted_demonstrations,
    imitation_replay_summary,
    load_correction_replacement_bundle,
    train_imitation,
    validate_dataset_runtime_holdout,
)
from carryrl.training.progress import TrainingProgress
from carryrl.training.retention import validate_initial_evaluation as _reuse_initial_evaluation
from carryrl.training.run import append_json, atomic_json, file_sha256, provenance
from carryrl.training.sequence import (
    _reward_composition_identity,
    evaluate_sequence,
    load_sequence_checkpoint,
    save_sequence_checkpoint,
    sequence_evaluation_protocol,
    tensor_hash,
)
from carryrl.training.sequence_config import SequenceRunConfig


def _publish_combined_frames(output, frames):
    """Publish new compressed RGB serially without touching a reused source archive."""
    output = Path(output)
    destination = output / "demonstration_frames.npz"
    partial = output / ".demonstration_frames.partial.npz"
    if (destination.exists() or destination.is_symlink()
            or partial.exists() or partial.is_symlink()):
        raise FileExistsError("combined RGB publication requires fresh output files")
    try:
        with ZipFile(partial, "x", compression=ZIP_DEFLATED, allowZip64=True) as archive:
            for index, frame in enumerate(frames):
                if index % 128 == 0:
                    if shutil.disk_usage(output).free < 2 * 1024**3:
                        raise OSError("combined RGB publication requires a 2 GiB free disk floor")
                    if partial.stat().st_size > 6 * 1024**3:
                        raise OSError("combined RGB archive exceeds its 6 GiB disk budget")
                with archive.open(f"frame_{index:06}.npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, frame, allow_pickle=False)
        if shutil.disk_usage(output).free < 2 * 1024**3:
            raise OSError("combined RGB publication requires a 2 GiB free disk floor")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _cache_feature_batch(encoded, *, dtype, backend, remaining_bytes):
    """Store only detached frozen features at the requested cache precision."""
    encoded = encoded.detach().to(device="cpu", dtype=getattr(torch, dtype))
    stored_bytes = encoded.numel() * encoded.element_size()
    if stored_bytes > remaining_bytes:
        raise MemoryError("frozen demonstration features exceed --feature-budget-mib")
    values = encoded.unbind()
    if backend == "mlx":
        from carryrl.training.mlx import require_mlx

        # to_mlx intentionally promotes inputs for policy computation; cache storage
        # must preserve its smaller dtype until a training episode is loaded.
        mx = require_mlx()
        values = [mx.array(value.numpy()) for value in values]
        mx.eval(values)
    return values, stored_bytes


def _training_feature(value, *, backend, dtype):
    return value.to(dtype=dtype) if backend == "torch" else value


def _add_memory_cue_pairs(dataset, *, max_pairs, max_steps, frame_budget, progress):
    """Collect optional safe counterparts and account for real accepted/skipped origins."""
    base = len(dataset.episodes)
    skipped = 0
    progress.update(force=True, phase_units_total=None, collecting_task="memory",
                    collecting_source="memory_cue_counterpart", memory_pairs_requested=max_pairs)

    def collected(row):
        nonlocal skipped
        skipped += int(row["status"] == "skipped")
        progress.update(phase_units_done=base + row["pairs_collected"],
            demonstration_episodes=base + row["pairs_collected"],
            memory_pairs_collected=row["pairs_collected"], memory_pairs_skipped=skipped,
            memory_origins_attempted=row["origins_attempted"],
            memory_origins_eligible=row["eligible_origins"])

    counterparts = collect_memory_cue_counterparts(dataset, max_pairs=max_pairs,
        max_steps=max_steps, max_frame_bytes=frame_budget, on_episode=collected)
    # Count exact unique RGB before extending, enforcing the combined byte budget.
    additional_bytes = sum(frame.nbytes for frame in counterparts.frames
        if (frame.shape, hashlib.sha256(frame.tobytes()).digest()) not in dataset._index)
    if dataset.frame_bytes + additional_bytes > frame_budget:
        raise MemoryError("memory cue counterparts exceed the aggregate RGB frame budget")
    dataset.extend(counterparts)
    report = counterparts.memory_cue_counterpart_reports[0]
    progress.update(force=True, phase_units_done=len(dataset.episodes),
        phase_units_total=len(dataset.episodes), demonstration_episodes=len(dataset.episodes),
        memory_pairs_collected=report["pairs_collected"],
        memory_pairs_skipped=len(report["skipped_origins"]),
        memory_origins_attempted=report["origins_attempted"],
        memory_origins_eligible=report["eligible_origins"],
        memory_origins_unattempted=report["unattempted_origins"])
    return dataset.provenance()["memory_cue_counterparts"]


def _record_training_failure(error, *, progress, sync_to_torch=None, save_checkpoint=None,
                             close_writer=None):
    """Persist the original failure before attempting a bounded recovery snapshot."""
    failed_while_saving = (progress.state["phase"] == "saving"
                           or progress.state.get("training_action") == "saving_checkpoint")
    recovery_errors = []

    def record_recovery_error(stage, recovery_error):
        recovery_errors.append({"stage": stage, "error_type": type(recovery_error).__name__,
                                "error": str(recovery_error)})

    try:
        notes = getattr(error, "__notes__", None)
        progress.update(force=True,
                        status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error_type=type(error).__name__, error=str(error),
                        **({"error_notes": list(notes)} if notes else {}))
    except BaseException as status_error:
        record_recovery_error("failure_status", status_error)
    recovery_checkpoint = None
    if not failed_while_saving and save_checkpoint is not None:
        stage = "sync_to_torch"
        try:
            if sync_to_torch is not None:
                sync_to_torch()
            stage = "checkpoint"
            recovery_checkpoint = save_checkpoint()
        except BaseException as recovery_error:
            record_recovery_error(stage, recovery_error)
    if close_writer is not None:
        try:
            close_writer()
        except BaseException as recovery_error:
            record_recovery_error("writer_close", recovery_error)
    if recovery_errors or recovery_checkpoint is not None:
        values = {"recovery_errors": recovery_errors}
        if recovery_checkpoint is not None:
            values["recovery_checkpoint"] = str(recovery_checkpoint)
        try:
            progress.update(force=True, **values)
        except BaseException as status_error:
            record_recovery_error("recovery_status", status_error)
            progress.state["recovery_errors"] = recovery_errors


def _save_epoch_snapshot(row, *, every, max_epochs, output, progress, agent, optimizer,
                         reward, config, saved, metadata, engine=None):
    """Save completed intervals; the requested final epoch uses the normal final save."""
    if (not every or not row["complete"] or row["epoch"] % every
            or row["epoch"] == max_epochs):
        return None
    epoch = row["epoch"]
    progress.update(force=True, training_action="saving_checkpoint", checkpoint_epoch=epoch)
    if engine is not None:
        engine.sync_to_torch()
    destination = output / "checkpoints_by_epoch" / f"epoch_{epoch:04d}"
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = save_sequence_checkpoint(
        agent, optimizer, reward, config, destination, rollout=0,
        decisions=saved["decisions"],
        revision=saved["weights_revision"] + row["optimizer_steps"],
    )
    # The archive already exists even if provenance or evaluation linking fails next.
    progress.update(force=True, last_epoch_checkpoint=str(checkpoint),
                    last_checkpoint_epoch=epoch, last_checkpoint_evaluation_match="pending")
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(kind="imitation_epoch_snapshot", epoch=epoch,
                    epochs_completed=progress.state["epochs_completed"],
                    optimizer_steps=row["optimizer_steps"],
                    source_checkpoint=metadata["source_checkpoint"],
                    source_checkpoint_sha256=metadata["source_checkpoint_sha256"],
                    initial_agent_sha256=metadata["initial_agent_sha256"])
    atomic_json(manifest_path, manifest)
    evaluation_path = output / "evaluations" / f"epoch_{epoch:04d}.json"
    if evaluation_path.exists():
        evaluated = json.loads(evaluation_path.read_text())
        if any(evaluated.get(key) != manifest[key]
               for key in ("weights_revision", "agent_sha256", "reward_sha256")):
            raise RuntimeError("epoch snapshot does not match its frozen evaluation")
        manifest["evaluation_reference"] = str(evaluation_path.resolve())
        atomic_json(manifest_path, manifest)
    likelihood_path = output / "validation_likelihood" / f"epoch_{epoch:04d}.json"
    if likelihood_path.exists():
        likelihood = json.loads(likelihood_path.read_text())
        if (likelihood.get("checkpoint_weights_revision") != manifest["weights_revision"]
                or any(likelihood.get(key) != manifest[key]
                       for key in ("agent_sha256", "reward_sha256"))):
            raise RuntimeError("epoch snapshot does not match its frozen validation likelihood")
        manifest["validation_likelihood_reference"] = str(likelihood_path.resolve())
        atomic_json(manifest_path, manifest)
    progress.update(force=True, training_action="learning", last_epoch_checkpoint=str(checkpoint),
                    last_checkpoint_epoch=epoch,
                    last_checkpoint_evaluation_match=("matched" if evaluation_path.exists()
                                                      else "not_evaluated"))
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--allow-reward-composition-change", action="store_true",
                        help="explicitly transfer to a different reward composition")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-backend", choices=("torch", "mlx"), default="mlx")
    parser.add_argument("--screen-text-language", default=None,
                        help="opt in to local RGB OCR using explicit Tesseract languages, e.g. eng")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--episodes-per-task", type=int, default=16)
    parser.add_argument("--difficulty", type=float, nargs="+", default=[1.0],
                        help="training difficulties (default: 1); runtime train levels must "
                             "remain separate from held-out levels after simplification")
    parser.add_argument("--evaluation-difficulty", type=float)
    parser.add_argument("--level-bank-seed", type=int)
    parser.add_argument("--correction-episodes-per-task", type=int, default=0)
    parser.add_argument("--correction-tasks", nargs="+", choices=TASKS)
    parser.add_argument("--correction-replacement-bundle", type=Path,
                        help="replace selected corrections with a validated fresh train bundle")
    parser.add_argument("--correction-replacement-sha256",
                        help="previously audited combined SHA256 of the replacement bundle files")
    parser.add_argument("--replacement-control-run", type=Path,
                        help="require a complete content audit against this original control run")
    parser.add_argument("--expert-probability", type=float, default=0.5)
    parser.add_argument("--learner-prefix-steps", type=int, default=0)
    parser.add_argument("--styled-correction-episodes-per-task", type=int, default=0,
                        help="append up to 8 live train styled learner corrections per task")
    parser.add_argument("--styled-correction-expert-probability", type=float, default=0.35)
    parser.add_argument("--styled-correction-prefix-steps", type=int, default=4)
    parser.add_argument("--presentation-copies-per-episode", type=int, default=0)
    parser.add_argument("--augmentation-themes", nargs="+", choices=THEME_SPLITS["train"],
                        default=None, help="train themes for posthoc appearance copies")
    parser.add_argument("--augmentation-hud-layouts", nargs="+", choices=HUD_LAYOUTS,
                        default=None, help="HUD layouts for posthoc appearance copies")
    parser.add_argument("--augmentation-textures", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--augmentation-canvas-padding", type=int, default=8)
    parser.add_argument("--navigation-geometry-pairs", type=int, default=0)
    parser.add_argument("--navigation-geometry-mode", choices=("repeat", "new_walls"))
    parser.add_argument("--runtime-diversity-axis",
                        choices=("navigation_geometry", "reaction_dynamics"))
    parser.add_argument("--runtime-diversity-pairs", type=int, default=0)
    parser.add_argument("--runtime-diversity-arm", choices=("repeat", "varied"))
    parser.add_argument("--memory-cue-pairs", type=int, default=0)
    parser.add_argument("--native-train-sizes", type=int, nargs="+", choices=(64, 128), default=[])
    parser.add_argument("--native-episodes-per-task", type=int, default=0)
    parser.add_argument("--dataset-seed", type=int, default=8176)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--action-balance", choices=("uniform", "active_idle"), default="uniform")
    parser.add_argument("--canonical-replay-weight", type=float, default=1.0,
                        help="loss-only episode weight for unstyled physical 96x96 originals; "
                             "appearance copies and native RGB episodes keep weight 1")
    parser.add_argument("--canonical-anchor-coefficient", type=float, default=0.0,
                        help="teacher-prefix conditional KL to the frozen initial policy on "
                             "canonical unstyled 96 RGB episodes (0 disables)")
    parser.add_argument("--canonical-anchor-exclude-corrections", action="store_true",
                        default=argparse.SUPPRESS,
                        help="zero correction-episode anchor KL with the same canonical-episode "
                             "denominator; BC and replay weights are unchanged")
    parser.add_argument("--minibatch-episodes", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-all-levels", action="store_true", default=argparse.SUPPRESS,
                        help="use every explicit validation level at --eval-every intervals "
                             "(default: short evaluations)")
    parser.add_argument("--validation-nll", action="store_true",
                        help="measure frozen held-out teacher likelihood at gameplay evaluations; "
                             "separate from training NLL and autonomous game wins")
    parser.add_argument("--validation-nll-episodes-per-task", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="save completed epoch intervals (0 disables); the final requested "
                             "epoch uses the normal final checkpoint")
    parser.add_argument("--max-wall-seconds", type=float, default=1200)
    parser.add_argument("--feature-budget-mib", type=int, default=2048)
    parser.add_argument("--frame-budget-mib", type=int, default=256)
    parser.add_argument("--feature-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--feature-reuse-run", type=Path,
                        help="reuse compatible frozen RGB features; encode missing frames only")
    parser.add_argument("--demonstration-reuse-run", type=Path,
                        help="use the exact complete saved RGB/action corpus; skip all collection")
    parser.add_argument("--appearance-bundle", type=Path,
                        help="use a validated prepared appearance arm; skip all collection")
    parser.add_argument("--appearance-bundle-sha256")
    parser.add_argument("--appearance-arm", choices=("control", "theme_shift"))
    parser.add_argument("--final-all-levels", action="store_true")
    parser.add_argument("--reuse-initial-evaluation", type=Path)
    parser.add_argument("--tensorboard", type=Path)
    args = parser.parse_args()
    # Keep historical prepared manifests' exact settings when the option is absent.
    eval_all_levels = getattr(args, "eval_all_levels", False)
    canonical_anchor_exclude_corrections = getattr(args, "canonical_anchor_exclude_corrections",
                                                   False)
    augmentation_flags = {"--augmentation-themes", "--augmentation-hud-layouts",
                          "--augmentation-textures", "--no-augmentation-textures",
                          "--augmentation-canvas-padding"}
    collection_flags = {
        "--episodes-per-task", "--difficulty", "--correction-episodes-per-task",
        "--correction-tasks", "--expert-probability", "--learner-prefix-steps",
        "--presentation-copies-per-episode", "--memory-cue-pairs", "--native-train-sizes",
        "--native-episodes-per-task", "--correction-replacement-bundle",
        "--correction-replacement-sha256", "--replacement-control-run",
    } | augmentation_flags
    supplied = {value.split("=", 1)[0] for value in sys.argv[1:]}
    legacy_runtime_flags = {"--navigation-geometry-pairs", "--navigation-geometry-mode"}
    runtime_flags = {"--runtime-diversity-axis", "--runtime-diversity-pairs",
                     "--runtime-diversity-arm"}
    if legacy_runtime_flags & supplied and runtime_flags & supplied:
        parser.error("legacy navigation geometry and runtime diversity flags cannot be combined")
    if (not 0 <= args.navigation_geometry_pairs <= 64
            or bool(args.navigation_geometry_pairs) != bool(args.navigation_geometry_mode)):
        parser.error("navigation geometry requires pairs in [1,64] and mode together")
    if (not 0 <= args.runtime_diversity_pairs <= 64
            or (bool(runtime_flags & supplied) and not (
                args.runtime_diversity_axis and args.runtime_diversity_pairs
                and args.runtime_diversity_arm))):
        parser.error("runtime diversity requires axis, pairs in [1,64], and arm together")
    diversity_axis = args.runtime_diversity_axis
    diversity_pairs_count = args.runtime_diversity_pairs
    diversity_arm = args.runtime_diversity_arm
    legacy_geometry = bool(args.navigation_geometry_pairs)
    if legacy_geometry:
        diversity_axis = "navigation_geometry"
        diversity_pairs_count = args.navigation_geometry_pairs
        diversity_arm = "repeat" if args.navigation_geometry_mode == "repeat" else "varied"
    if diversity_pairs_count:
        incompatible = sorted(({
            "--appearance-bundle", "--appearance-bundle-sha256", "--appearance-arm",
            "--styled-correction-episodes-per-task", "--styled-correction-expert-probability",
            "--styled-correction-prefix-steps", "--presentation-copies-per-episode",
            "--correction-replacement-bundle", "--correction-replacement-sha256",
            "--replacement-control-run", "--memory-cue-pairs", "--native-train-sizes",
            "--native-episodes-per-task",
        } | augmentation_flags) & supplied)
        if incompatible:
            parser.error("runtime diversity excludes other experimental collection controls: "
                         + ", ".join(incompatible))
    appearance_inputs = (args.appearance_bundle, args.appearance_bundle_sha256, args.appearance_arm)
    if any(value is not None for value in appearance_inputs):
        if not all(value is not None for value in appearance_inputs):
            parser.error("appearance reuse requires bundle, trusted SHA256, and arm together")
        incompatible = sorted((collection_flags | {
            "--demonstration-reuse-run", "--feature-reuse-run",
            *legacy_runtime_flags, *runtime_flags,
            "--styled-correction-episodes-per-task", "--styled-correction-expert-probability",
            "--styled-correction-prefix-steps",
        }) & supplied)
        if incompatible:
            parser.error("appearance reuse excludes collection and saved reuse flags: "
                         + ", ".join(incompatible))
    if args.demonstration_reuse_run is not None:
        incompatible = sorted(collection_flags & supplied)
        if incompatible:
            parser.error("demonstration reuse excludes collection flags: "
                         + ", ".join(incompatible))
    if bool(args.correction_replacement_bundle) != bool(args.correction_replacement_sha256):
        parser.error("correction replacement requires both bundle and trusted SHA256")
    if args.replacement_control_run is not None and args.correction_replacement_bundle is None:
        parser.error("replacement control audit requires a correction replacement bundle")
    if args.correction_replacement_bundle and (
            args.difficulty != [1.0] or args.correction_episodes_per_task < 1):
        parser.error("correction replacement requires one full-difficulty stage and corrections")
    if args.feature_reuse_run is not None and args.feature_cache_dir is None:
        parser.error("feature reuse requires a feature cache directory for missing RGB rows")
    if not math.isfinite(args.canonical_replay_weight) or args.canonical_replay_weight <= 0:
        parser.error("canonical replay weight must be positive and finite")
    if (not math.isfinite(args.canonical_anchor_coefficient)
            or args.canonical_anchor_coefficient < 0):
        parser.error("canonical anchor coefficient must be nonnegative and finite")
    if args.checkpoint_every < 0:
        parser.error("checkpoint frequency must be nonnegative (0 disables)")
    if args.validation_nll_episodes_per_task < 1:
        parser.error("validation NLL episodes per task must be positive")
    if args.eval_every < 1 or args.feature_budget_mib < 1:
        parser.error("evaluation frequency and feature budget must be positive")
    if args.frame_budget_mib < 1:
        parser.error("frame budget must be positive")
    if args.correction_episodes_per_task < 0 or not 0 <= args.expert_probability <= 1:
        parser.error("correction episodes must be nonnegative and mixture probability in [0,1]")
    if args.learner_prefix_steps < 0:
        parser.error("learner prefix steps must be nonnegative")
    if (not 0 <= args.styled_correction_episodes_per_task <= 8
            or not math.isfinite(args.styled_correction_expert_probability)
            or not 0 <= args.styled_correction_expert_probability <= 1
            or args.styled_correction_prefix_steps < 0):
        parser.error("styled corrections require 0..8 episodes, finite probability in [0,1], "
                     "and a nonnegative prefix")
    if args.presentation_copies_per_episode < 0:
        parser.error("presentation copies must be nonnegative")
    if augmentation_flags & supplied and not args.presentation_copies_per_episode:
        parser.error("augmentation controls require positive --presentation-copies-per-episode")
    for name in ("augmentation_themes", "augmentation_hud_layouts"):
        selection = getattr(args, name)
        if selection is not None and len(set(selection)) != len(selection):
            parser.error(name.replace("_", " ") + " must be unique")
    if not 4 <= args.augmentation_canvas_padding <= 64:
        parser.error("augmentation canvas padding must be in 4..64")
    if not 0 <= args.memory_cue_pairs <= 64:
        parser.error("memory cue pairs must be in [0,64]")
    if (not 0 <= args.native_episodes_per_task <= 64
            or bool(args.native_train_sizes) != bool(args.native_episodes_per_task)
            or len(set(args.native_train_sizes)) != len(args.native_train_sizes)):
        parser.error("native sizes must be unique and enabled with native episodes in [1,64]")
    if args.evaluation_difficulty is not None and not 0 <= args.evaluation_difficulty <= 1:
        parser.error("evaluation difficulty must be in [0,1]")
    source_config = SequenceRunConfig.load(args.config)
    if args.learner_prefix_steps > source_config.max_episode_steps:
        parser.error("learner prefix steps must not exceed the configured maximum episode steps")
    if (args.styled_correction_episodes_per_task
            and args.styled_correction_prefix_steps > source_config.max_episode_steps):
        parser.error("styled correction prefix must not exceed maximum episode steps")
    frame_budget = args.frame_budget_mib * 1024**2
    level_bank_seed = (source_config.level_bank_seed if args.level_bank_seed is None
                       else args.level_bank_seed)
    config = replace(
        source_config, device="cpu", training_backend=args.training_backend,
        screen_text_language=(args.screen_text_language if args.screen_text_language is not None
                              else source_config.screen_text_language),
        learning_rate=args.learning_rate,
        curriculum_difficulty=(args.difficulty[0] if args.evaluation_difficulty is None
                               else args.evaluation_difficulty),
        evaluation_difficulty=(args.difficulty[0] if args.evaluation_difficulty is None
                               else args.evaluation_difficulty),
        level_bank_seed=level_bank_seed,
        curriculum_warmup_rollouts=0, eval_all_levels=False,
    )
    corpus_reuse = appearance_reuse = None
    if args.demonstration_reuse_run is not None:
        # Validate before allocating a model or creating the optimizer continuation run.
        corpus_reuse = SavedDemonstrationReuse.prepare(
            args.demonstration_reuse_run, config=config, max_frame_bytes=frame_budget)
    elif args.appearance_bundle is not None:
        from carryrl.training.appearance import PreparedAppearanceReuse

        appearance_reuse = PreparedAppearanceReuse.prepare(
            args.appearance_bundle, expected_sha256=args.appearance_bundle_sha256,
            arm=args.appearance_arm, config=config, max_frame_bytes=frame_budget)
        corpus_reuse = appearance_reuse
    if corpus_reuse is not None:
        recorded_difficulty = corpus_reuse.summary["source_difficulty"]
        config = replace(config,
            curriculum_difficulty=recorded_difficulty["curriculum_difficulty"],
            evaluation_difficulty=(recorded_difficulty["evaluation_difficulty"]
                if args.evaluation_difficulty is None else args.evaluation_difficulty))
        if args.feature_reuse_run is not None and (
                args.feature_reuse_run.resolve() != corpus_reuse.directory):
            parser.error("exact demonstration and feature reuse must refer to the same source run")
    diversity_pairs = None
    if diversity_pairs_count:
        required_task = {"navigation_geometry": "navigation", "reaction_dynamics": "reaction"}[
            diversity_axis]
        if required_task not in config.tasks or config.env_image_size != 96:
            parser.error(f"{diversity_axis} requires {required_task} and canonical 96 RGB")
        if corpus_reuse is not None:
            from carryrl.training.runtime_diversity import collect_runtime_diversity_pairs

            diversity_pairs = collect_runtime_diversity_pairs(diversity_axis, corpus_reuse.dataset,
                max_pairs=diversity_pairs_count, seed=args.dataset_seed,
                max_steps=config.max_episode_steps, max_frame_bytes=frame_budget, config=config)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    progress = TrainingProgress(output / "status.json", epochs_total=args.epochs)
    status = progress.state
    atomic_json(output / "config.json", config.to_dict())
    writer = None
    agent = engine = optimizer = reward = saved = None
    try:
        if args.training_backend == "mlx":
            from carryrl.training.mlx import require_mlx

            require_mlx().set_cache_limit(256 * 1024**2)
        torch.set_num_threads(config.torch_threads)
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        agent, optimizer, reward, saved = load_sequence_checkpoint(
            args.warm_start, config, evaluation_only=True,
            allow_reward_composition_change=args.allow_reward_composition_change,
        )
        checkpoint_config = SequenceRunConfig.from_dict(saved["config"])
        source_composition = _reward_composition_identity(checkpoint_config)
        target_composition = _reward_composition_identity(config)
        if args.allow_reward_composition_change and source_composition == target_composition:
            raise ValueError("--allow-reward-composition-change requires an actual "
                             "objective change")
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate
            group["weight_decay"] = config.weight_decay
        reward_hash = tensor_hash(reward)
        source_manifest = json.loads((args.warm_start / "manifest.json").read_text())
        correction_tasks = tuple(args.correction_tasks or config.tasks)
        if any(task not in config.tasks for task in correction_tasks):
            raise ValueError("correction tasks must belong to this run's task selection")
        replacement_datasets, replacement_summary = {}, None
        if args.correction_replacement_bundle is not None:
            replacement_datasets, replacement_summary = load_correction_replacement_bundle(
                args.correction_replacement_bundle,
                expected_sha256=args.correction_replacement_sha256,
                correction_tasks=correction_tasks,
                episodes_per_task=args.correction_episodes_per_task, seed=args.dataset_seed + 5000,
                difficulty=1.0, max_steps=config.max_episode_steps, level_bank_seed=level_bank_seed,
                episode_instructions=config.episode_instructions, source_identity=source_manifest,
                initial_agent_sha256=tensor_hash(agent), reward_sha256=reward_hash,
                max_frame_bytes=frame_budget)
        reused_initial = None
        if args.reuse_initial_evaluation is not None:
            reused_initial = _reuse_initial_evaluation(
                args.reuse_initial_evaluation, checkpoint=args.warm_start,
                config=replace(config, eval_all_levels=args.final_all_levels),
                agent_sha256=tensor_hash(agent), reward_sha256=reward_hash,
                weights_revision=saved["weights_revision"])
        metadata = {
            "kind": "imitation_training", "phase": "scripted_teacher_imitation",
            "suite_version": SUITE_VERSION,
            "config": config.to_dict(), "settings": vars(args), "provenance": provenance(),
            "source_checkpoint": str(args.warm_start.resolve()),
            "source_checkpoint_sha256": source_manifest["sha256"],
            "initial_agent_sha256": tensor_hash(agent), "reward_sha256": reward_hash,
            "optimizer_state_preserved": True, "human_demonstrations": False,
            "reward_composition_transfer": {
                "source": source_composition, "target": target_composition,
                "changed": source_composition != target_composition,
                "explicitly_allowed": args.allow_reward_composition_change,
            },
            "teacher": "scripted RGB controller; expert labels are not learned-agent results",
            "collection": "train-only; successful teacher trajectories and optional "
                          "learner-visited correction trajectories with actual action history",
            "evaluation": "frozen policy on validation levels; test remains unused",
            "eval_all_levels": eval_all_levels,
            "wall_budget_scope": "training loop including scheduled validation; excludes "
                                 "preparation, initial and final full-bank evaluation",
            "counter_scope": "checkpoint rollout=0; agent decisions inherited; imitation "
                             "updates added to weights_revision; epochs separately recorded",
        }
        metadata["settings"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "canonical_anchor_exclude_corrections" or value
        }
        metadata["canonical_anchor"] = {
            "coefficient": args.canonical_anchor_coefficient,
            "exclude_corrections": canonical_anchor_exclude_corrections,
            "parent_checkpoint": str(args.warm_start.resolve()),
            "parent_checkpoint_sha256": source_manifest["sha256"],
            "parent_agent_sha256": metadata["initial_agent_sha256"],
            "objective": (
                "teacher-prefix conditional KL(parent||student); mean physical decisions, "
                "sum eligible non-correction canonical episodes / pre-mask canonical episodes"
                if canonical_anchor_exclude_corrections else
                "teacher-prefix conditional KL(parent||student); separately mean "
                "eligible canonical unstyled 96 episodes and physical decisions"),
        }
        if appearance_reuse is not None:
            metadata["appearance_bundle"] = {
                "source": str(appearance_reuse.directory),
                "bundle_sha256": args.appearance_bundle_sha256, "arm": args.appearance_arm,
            }
        if replacement_summary is not None:
            metadata["correction_replacement"] = {
                **replacement_summary, "mode": "replace selected correction slots",
                "original_control_collection": {
                    "expert_probability": args.expert_probability,
                    "learner_prefix_steps": args.learner_prefix_steps,
                    "level_bank_seed": level_bank_seed, "seed": args.dataset_seed + 5000,
                    "correction_tasks": list(correction_tasks),
                },
            }
        atomic_json(output / "metadata.json", metadata)
        if reused_initial is not None:
            atomic_json(output / "evaluations" / "epoch_0000.json", reused_initial)
        if corpus_reuse is not None:
            dataset = corpus_reuse.dataset
            metadata["demonstration_reuse"] = corpus_reuse.summary
            metadata["collection"] = (
                "prepared paired train appearance corpus; collection skipped"
                if appearance_reuse is not None else
                "exact complete saved train-only RGB/action corpus; collection skipped")
            progress.phase("collecting", total=len(dataset.episodes), unit="episodes")
            progress.update(force=True, phase_units_done=len(dataset.episodes),
                            demonstration_episodes=len(dataset.episodes), collection_skipped=True)
        else:
            collection_total = len(args.difficulty) * (
                len(config.tasks) * args.episodes_per_task
                + len(correction_tasks) * args.correction_episodes_per_task)
            progress.phase("collecting", total=collection_total, unit="episodes")
            dataset = None
            for difficulty in args.difficulty:
                for task in config.tasks:
                    collected_base = len(dataset.episodes) if dataset is not None else 0

                    def collection_progress(row):
                        completed = collected_base + row["episodes_completed"]
                        progress.update(phase_units_done=completed,
                                        collecting_task=row["task"],
                                        demonstration_episodes=completed)

                    collected = collect_scripted_demonstrations(
                        args.episodes_per_task, tasks=(task,), seed=args.dataset_seed,
                        difficulty=difficulty, max_steps=config.max_episode_steps,
                        level_bank_seed=level_bank_seed,
                        episode_instructions=config.episode_instructions,
                        max_frame_bytes=frame_budget,
                        on_episode=collection_progress,
                    )
                    if dataset is None:
                        dataset = collected
                    else:
                        dataset.extend(collected)
                    del collected
                    progress.update(phase_units_done=len(dataset.episodes), collecting_task=task,
                                    demonstration_episodes=len(dataset.episodes))
                for task in correction_tasks if args.correction_episodes_per_task else ():
                    collected_base = len(dataset.episodes)

                    def correction_progress(row):
                        progress.update(phase_units_done=collected_base + row["episodes_completed"],
                                        collecting_task=row["task"], collecting_source="correction")

                    if task in replacement_datasets:
                        collected = replacement_datasets[task]
                    else:
                        collected = collect_scripted_corrections(
                            agent, args.correction_episodes_per_task, tasks=(task,),
                            seed=args.dataset_seed + 5000, difficulty=difficulty,
                            level_bank_seed=level_bank_seed, max_steps=config.max_episode_steps,
                            expert_probability=args.expert_probability,
                            learner_prefix_steps=args.learner_prefix_steps,
                            episode_instructions=config.episode_instructions,
                            max_frame_bytes=frame_budget,
                            on_episode=correction_progress,
                        )
                    dataset.extend(collected)
                    del collected
                    progress.update(phase_units_done=len(dataset.episodes), collecting_task=task,
                                    demonstration_episodes=len(dataset.episodes))
            if args.memory_cue_pairs:
                metadata["memory_cue_counterparts"] = _add_memory_cue_pairs(dataset,
                    max_pairs=args.memory_cue_pairs, max_steps=config.max_episode_steps,
                    frame_budget=frame_budget, progress=progress)
                atomic_json(output / "metadata.json", metadata)
            if args.native_train_sizes:
                # Native passes append after canonical collection and its memory peers.
                # They always use full difficulty, independently of canonical stages.
                native_base = len(dataset.episodes)
                native_done = 0

                def native_progress(row):
                    nonlocal native_done
                    native_done += 1
                    progress.update(phase_units_done=native_base + native_done,
                                    phase_units_total=native_base + len(args.native_train_sizes)
                                    * len(config.tasks) * args.native_episodes_per_task,
                                    demonstration_episodes=native_base + native_done,
                                    collecting_task=row["task"], collecting_source="native_teacher")

                native = collect_native_scripted_demonstrations(
                    native_train_sizes=args.native_train_sizes,
                    native_episodes_per_task=args.native_episodes_per_task,
                    tasks=config.tasks, seed=args.dataset_seed, difficulty=1.0,
                    level_bank_seed=level_bank_seed, max_steps=config.max_episode_steps,
                    episode_instructions=config.episode_instructions,
                    max_frame_bytes=frame_budget, on_episode=native_progress)
                dataset.extend(native)
                del native
            if args.presentation_copies_per_episode:
                copies = augment_demonstrations(
                    dataset, seed=args.dataset_seed,
                    copies_per_episode=args.presentation_copies_per_episode,
                    max_frame_bytes=frame_budget,
                    themes=args.augmentation_themes, hud_layouts=args.augmentation_hud_layouts,
                    textures=args.augmentation_textures,
                    canvas_padding=args.augmentation_canvas_padding)
                dataset.extend(copies)
                del copies
                progress.update(force=True, phase_units_done=len(dataset.episodes),
                                phase_units_total=len(dataset.episodes),
                                demonstration_episodes=len(dataset.episodes))
            if args.replacement_control_run is not None:
                try:
                    import_audit = audit_correction_replacement(
                        dataset, args.replacement_control_run,
                        replacement_datasets, replacement_summary)
                except Exception as error:
                    atomic_json(output / "replacement_import_audit.json", {
                        "passed": False, "before_encoding": True,
                        "control_run": str(args.replacement_control_run.resolve()),
                        "bundle_sha256": replacement_summary["bundle_sha256"],
                        "error_type": type(error).__name__, "error": str(error),
                    })
                    raise
                atomic_json(output / "replacement_import_audit.json", import_audit)
                metadata["correction_replacement"]["import_audit"] = import_audit
        if diversity_pairs_count:
            from carryrl.training.runtime_diversity import collect_runtime_diversity_pairs

            if diversity_pairs is None:
                diversity_pairs = collect_runtime_diversity_pairs(diversity_axis, dataset,
                    max_pairs=diversity_pairs_count, seed=args.dataset_seed,
                    max_steps=config.max_episode_steps, max_frame_bytes=frame_budget, config=config)
            dataset.extend(diversity_pairs.arm(diversity_arm))
            delta_metadata = {
                **diversity_pairs.manifest,
                "operation": "append delta to exact unchanged base",
                "replay_weight_per_delta_episode": args.canonical_replay_weight,
            }
            if legacy_geometry:
                metadata["navigation_geometry"] = {
                    **delta_metadata, "mode": args.navigation_geometry_mode}
            else:
                metadata["runtime_diversity"] = {
                    **delta_metadata, "axis": diversity_axis, "arm": diversity_arm}
            metadata["collection"] = (
                "unchanged base plus controlled navigation geometry delta" if legacy_geometry
                else f"unchanged base plus controlled {diversity_axis} delta")
            if corpus_reuse is not None:
                corpus_reuse.assert_unchanged()
                metadata["demonstration_reuse"] = {**corpus_reuse.summary,
                    "collection_skipped": False, "source_corpus_preserved": True,
                    "storage": "new combined compressed RGB archive; original feature references",
                    "appended_episodes": diversity_pairs_count}
            progress.update(force=True, demonstration_episodes=len(dataset.episodes),
                            collection_skipped=False)
        if args.styled_correction_episodes_per_task:
            base_episodes = len(dataset.episodes)
            progress.phase("collecting", total=base_episodes + len(config.tasks)
                * args.styled_correction_episodes_per_task, unit="episodes")
            presentation = PresentationConfig(mode="varied", native_sizes=(96,),
                themes=THEME_SPLITS["train"], style_split="train", seed=args.dataset_seed)
            producer_before = tensor_hash(agent)
            for task in config.tasks:
                collected_base = len(dataset.episodes)

                def styled_progress(row):
                    progress.update(phase_units_done=collected_base + row["episodes_completed"],
                        collecting_task=row["task"], collecting_source="live_styled",
                        demonstration_episodes=collected_base + row["episodes_completed"])

                styled = collect_scripted_corrections(agent,
                    args.styled_correction_episodes_per_task, tasks=(task,),
                    seed=args.dataset_seed + 10000, difficulty=1.0, image_size=96,
                    level_bank_seed=level_bank_seed, max_steps=config.max_episode_steps,
                    expert_probability=args.styled_correction_expert_probability,
                    learner_prefix_steps=args.styled_correction_prefix_steps,
                    presentation_config=presentation,
                    episode_instructions=config.episode_instructions,
                    max_frame_bytes=frame_budget, on_episode=styled_progress)
                dataset.extend(styled)
                del styled
            if tensor_hash(agent) != producer_before or tensor_hash(reward) != reward_hash:
                raise RuntimeError("styled correction collection changed frozen producer tensors")
            metadata["live_styled_corrections"] = {
                "episodes": len(dataset.episodes) - base_episodes, "difficulty": 1.0,
                "level_bank_seed": level_bank_seed, "seed": args.dataset_seed + 10000,
                "presentation": presentation.__dict__, "producer_frozen": True,
                "producer_agent_sha256": producer_before, "producer_reward_sha256": reward_hash,
                "expert_probability": args.styled_correction_expert_probability,
                "learner_prefix_steps": args.styled_correction_prefix_steps,
                "heldout_bank_seeds": sorted({0, 20261008, level_bank_seed}),
            }
            metadata["collection"] = "saved corpus plus live train-only styled learner corrections"
            if corpus_reuse is not None:
                corpus_reuse.assert_unchanged()
                metadata["demonstration_reuse"] = {**corpus_reuse.summary,
                    "collection_skipped": False, "source_corpus_preserved": True,
                    "storage": "new combined compressed RGB archive; original feature references",
                    "appended_episodes": len(dataset.episodes) - base_episodes}
            progress.update(force=True, phase_units_done=len(dataset.episodes),
                demonstration_episodes=len(dataset.episodes), collection_skipped=False)
        validate_dataset_runtime_holdout(dataset, level_bank_seed=config.level_bank_seed,
            evaluation_difficulty=sequence_evaluation_protocol(config)["difficulty"])
        metadata["replay_weighting"] = imitation_replay_summary(
            dataset, canonical_replay_weight=args.canonical_replay_weight)
        screen_text_provider = None
        if config.screen_text_language is not None:
            screen_text_provider = ImitationScreenText(
                dataset, agent.screen_text_reader, output / "screen_text_observations.jsonl")
            metadata["screen_text"] = {
                "language": config.screen_text_language,
                "artifact": screen_text_provider.path.name,
                "sha256": file_sha256(screen_text_provider.path),
                "observations": len(screen_text_provider.offsets),
                "source": "actual saved RGB at each decision; untrusted OCR text",
                "feature_storage": "frozen E5 tokens computed per minibatch; no corpus token cache",
            }
        atomic_json(output / "metadata.json", metadata)
        if (args.demonstration_reuse_run is not None
                and not args.styled_correction_episodes_per_task
                and not diversity_pairs_count):
            corpus_reuse.link_corpus(output)
        else:
            atomic_json(output / "demonstrations.json", dataset.provenance())
            if args.styled_correction_episodes_per_task or diversity_pairs_count:
                _publish_combined_frames(output, dataset.frames)
            elif len({frame.shape for frame in dataset.frames}) == 1:
                np.savez_compressed(output / "demonstration_frames.npz",
                                    frames=np.stack(dataset.frames))
            else:
                np.savez_compressed(output / "demonstration_frames.npz",
                                    **{f"frame_{index:06}": frame
                                       for index, frame in enumerate(dataset.frames)})
        features, feature_bytes = {}, 0
        used = sorted({index for episode in dataset.episodes for index in episode.frame_indices})
        batches_per_epoch = math.ceil(len(dataset.episodes) / args.minibatch_episodes)
        progress.phase("encoding", total=len(used), unit="frames", encoded_frames=0,
                       feature_dtype=args.feature_dtype, feature_bytes=0,
                       unique_frames=len(dataset.frames),
                       demonstration_episodes=len(dataset.episodes),
                       batches_per_epoch=batches_per_epoch,
                       optimizer_steps_total=args.epochs * batches_per_epoch)
        budget = args.feature_budget_mib * 1024**2
        cache_dtype = getattr(torch, args.feature_dtype)
        feature_identity, cache_hit, reuse = None, False, None
        encoding_indices, cached_reuse_frames = used, 0
        if args.feature_cache_dir is not None:
            vision = agent.vision
            vision_implementation = f"{type(vision).__module__}.{type(vision).__name__}"
            feature_identity = feature_cache_identity(
                dataset.frames, used, vision_sha256=tensor_hash(vision),
                preprocessing={"implementation": vision_implementation,
                               "preprocess": "letterbox_bicubic_antialias_imagenet_v1",
                               "image_size": vision.image_size, "patch_size": vision.patch_size,
                               "patch_grid": list(vision.patch_grid)},
                feature_shape=(math.prod(vision.patch_grid), vision.hidden_size),
                dtype=args.feature_dtype)
            if args.feature_reuse_run is not None:
                reuse = FrozenFeatureReuse.prepare(
                    args.feature_reuse_run, dataset.frames, used, feature_identity,
                    cache_root=args.feature_cache_dir, max_bytes=budget)
                features, encoding_indices = reuse, reuse.missing_indices
                frame_feature_bytes = (math.prod(vision.patch_grid) * vision.hidden_size
                                       * np.dtype(args.feature_dtype).itemsize)
                feature_bytes = reuse.reused_count * frame_feature_bytes
                if encoding_indices:
                    shard = FrozenFeatureStore.load(args.feature_cache_dir, reuse.shard_identity,
                                                    max_bytes=budget)
                    if shard is not None:
                        reuse.bind_shard(shard)
                        cached_reuse_frames = len(encoding_indices)
                        feature_bytes = reuse.logical_bytes
                        encoding_indices = []
                progress.update(force=True, feature_cache_hit=False, partial_feature_reuse=True,
                                encoding_source="verified_layered_rgb_reuse",
                                reused_frames=reuse.reused_count,
                                cached_shard_frames=cached_reuse_frames,
                                phase_units_done=reuse.reused_count + cached_reuse_frames,
                                encoded_frames=0, encoded_frames_total=len(encoding_indices),
                                logical_feature_bytes=reuse.logical_bytes,
                                feature_bytes=feature_bytes,
                                encoding_skipped=not encoding_indices)
                print(f"reused {reuse.reused_count} frozen RGB features; "
                      f"{len(encoding_indices)} frames require encoding", flush=True)
            else:
                features = FrozenFeatureStore.load(args.feature_cache_dir, feature_identity,
                                                   max_bytes=budget)
                cache_hit = features is not None
                progress.update(force=True, feature_cache_hit=cache_hit,
                                encoding_source="verified_disk_cache" if cache_hit
                                else "computed_for_disk_cache")
            if cache_hit:
                encoding_indices = []
                feature_bytes = features.manifest["bytes"]
                progress.update(force=True, phase_units_done=len(features),
                                phase_unit_kind="cached_frames", cached_frames=len(features),
                                encoded_frames=0, encoded_frames_total=0, encoding_skipped=True,
                                feature_bytes=feature_bytes,
                                feature_cache_created_utc=features.manifest["created_utc"])
                print(f"reused verified frozen features for {len(features)} RGB frames", flush=True)
            elif reuse is None:
                features = {}
        frame_groups = defaultdict(list)
        for index in encoding_indices:
            frame_groups[dataset.frames[index].shape].append(index)
        encoding_batches = [group[offset:offset + 16] for group in frame_groups.values()
                            for offset in range(0, len(group), 16)]
        writer_identity = reuse.shard_identity if reuse is not None else feature_identity
        writer_context = (FrozenFeatureWriter(args.feature_cache_dir, writer_identity,
                                             max_bytes=budget)
                          if writer_identity is not None and encoding_indices else nullcontext())
        encoded_frames = 0
        with writer_context as feature_writer:
            for batch_index, indices in enumerate(encoding_batches):
                batch = torch.from_numpy(np.stack([dataset.frames[index] for index in indices]))
                encoded = agent.encode_frames(batch)
                if feature_writer is not None:
                    encoded = encoded.detach().to(device="cpu", dtype=cache_dtype)
                    feature_writer.write(indices, encoded.numpy())
                    stored_bytes = encoded.numel() * encoded.element_size()
                else:
                    values, stored_bytes = _cache_feature_batch(
                        encoded, dtype=args.feature_dtype, backend=args.training_backend,
                        remaining_bytes=budget - feature_bytes)
                    features.update(zip(indices, values, strict=True))
                feature_bytes += stored_bytes
                encoded_frames += len(indices)
                done = encoded_frames + (reuse.reused_count + cached_reuse_frames
                                         if reuse is not None else 0)
                progress.update(phase_units_done=done, encoded_frames=encoded_frames,
                                encoded_frames_total=len(encoding_indices),
                                feature_bytes=feature_bytes)
                if batch_index % 16 == 0:
                    print(f"encoded {encoded_frames}/{len(encoding_indices)} RGB frames",
                          flush=True)
            if feature_writer is not None:
                shard = feature_writer.publish()
                features = reuse.bind_shard(shard) if reuse is not None else shard
        disk_cache_metadata = {}
        if isinstance(features, FrozenFeatureReuse):
            reuse_manifest = features.manifest
            disk_cache_metadata = {
                "feature_cache_hit": False, "partial_feature_reuse": True,
                "feature_reuse": reuse_manifest,
                "reused_frames": features.reused_count,
                "encoded_frames": encoded_frames, "cached_shard_frames": cached_reuse_frames,
                "logical_feature_bytes": features.logical_bytes,
                "storage": reuse_manifest["storage"],
            }
            compact_reuse = {
                key: value for key, value in disk_cache_metadata.items() if key != "feature_reuse"}
            compact_reuse["feature_reuse_mapping_sha256"] = reuse_manifest["mapping_sha256"]
            for name in ("source_pack", "shard_pack"):
                pack = reuse_manifest[name]
                compact_reuse[f"feature_reuse_{name}"] = (
                    {key: pack[key] for key in ("directory", "sha256", "created_utc")}
                    if pack is not None else None)
            progress.update(force=True, **compact_reuse)
            metadata["feature_reuse"] = disk_cache_metadata["feature_reuse"]
            atomic_json(output / "metadata.json", metadata)
        elif isinstance(features, FrozenFeatureStore):
            disk_cache_metadata = {
                "feature_cache_hit": cache_hit,
                "feature_cache_dir": str(features.directory.resolve()),
                "feature_cache_created_utc": features.manifest["created_utc"],
                "feature_cache_sha256": features.manifest["sha256"],
                "storage": "read-only NumPy memmap; copied per requested frame",
            }
            progress.update(force=True, **disk_cache_metadata)
        atomic_json(output / "feature_manifest.json", {
            "source": "frozen DINO before all trainable layers", "dtype": args.feature_dtype,
            "bytes": feature_bytes, "budget_bytes": budget, "frames": len(features),
            "vision_sha256": tensor_hash(agent.vision),
            "demonstrations_sha256": hashlib.sha256(
                (output / "demonstrations.json").read_bytes()).hexdigest(),
            **disk_cache_metadata,
        })
        if args.training_backend == "mlx":
            from carryrl.training.mlx import MLXPPOBackend

            engine = MLXPPOBackend(agent, optimizer)
        if args.tensorboard is not None:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(args.tensorboard / output.name))

        def validation_likelihood(epoch):
            if not args.validation_nll:
                return
            from carryrl.evaluation.imitation import evaluate_imitation_validation

            progress.phase("evaluation", unit="episodes", evaluation_episodes_completed=0)
            nll_config = replace(config, eval_all_levels=False,
                                 eval_episodes_per_task=args.validation_nll_episodes_per_task)

            def likelihood_progress(event):
                progress.update(phase_units_done=event["episodes_completed"],
                                phase_units_total=event["episodes_total"],
                                evaluation_episodes_completed=event["episodes_completed"],
                                evaluation_episodes_total=event["episodes_total"],
                                evaluation_policy=event["policy"], evaluation_task=event["task"],
                                evaluation_step=event["step"],
                                force=event["event"] == "episode_completed")

            likelihood = evaluate_imitation_validation(
                agent, nll_config, reward=reward,
                on_progress=likelihood_progress,
                checkpoint_weights_revision=saved["weights_revision"] + status["optimizer_steps"])
            likelihood["epoch"] = epoch
            atomic_json(output / "validation_likelihood" / f"epoch_{epoch:04d}.json", likelihood)

        last_full_evaluation = None

        def evaluate(epoch, *, all_levels=False, phase="evaluation"):
            nonlocal last_full_evaluation
            progress.phase(phase, unit="episodes", evaluation_episodes_completed=0)
            revision = saved["weights_revision"] + status["optimizer_steps"]
            agent_hash = tensor_hash(agent)
            reward_evaluation_hash = tensor_hash(reward)
            suffix = "_final" if phase == "final_evaluation" else ""
            evaluation_path = output / "evaluations" / f"epoch_{epoch:04d}{suffix}.json"
            evaluation_config = replace(config, eval_all_levels=all_levels)
            if eval_all_levels and all_levels:
                protocol = sequence_evaluation_protocol(evaluation_config)
                protocol_hash = hashlib.sha256(
                    json.dumps(protocol, sort_keys=True).encode()).hexdigest()
                identity = (epoch, revision, agent_hash, reward_evaluation_hash, protocol_hash)
                if (phase == "final_evaluation" and last_full_evaluation is not None
                        and last_full_evaluation[0] == identity):
                    _, source_path, measured = last_full_evaluation
                    atomic_json(evaluation_path, {**measured, "final_evaluation_reuse": {
                        "source": str(source_path.resolve()),
                        "source_sha256": file_sha256(source_path),
                        "measurement_reused": True,
                    }})
                    count = len(measured["records"])
                    progress.update(force=True, phase_units_done=count, phase_units_total=count,
                                    evaluation_episodes_completed=count,
                                    evaluation_episodes_total=count)
                    return

            def evaluation_progress(event):
                progress.update(phase_units_done=event["episodes_completed"],
                                phase_units_total=event["episodes_total"],
                                evaluation_episodes_completed=event["episodes_completed"],
                                evaluation_episodes_total=event["episodes_total"],
                                evaluation_policy=event["policy"], evaluation_task=event["task"],
                                evaluation_step=event["step"],
                                force=event["event"] == "episode_completed")

            def playback(record, frames, actions):
                record["split"] = "validation"
                level = record["level_index"]
                label = f"level{level}" if level is not None else "seeded"
                destination = output / "playbacks" / (
                    f"epoch_{epoch:04d}_{record['task']}_validation_{label}_greedy.json")
                if not destination.exists():
                    write_frozen_playback(output, record, frames, actions, epoch=epoch,
                                          agent_sha256=agent_hash, weights_revision=revision,
                                          evaluation_reference=evaluation_path)

            evaluated = evaluate_sequence(agent, reward, evaluation_config,
                                          on_progress=evaluation_progress, on_playback=playback)
            evaluated.update(epoch=epoch, weights_revision=revision, agent_sha256=agent_hash,
                             reward_sha256=reward_evaluation_hash)
            atomic_json(evaluation_path, evaluated)
            summary = evaluated["summaries"]["frozen_deterministic"]
            print(f"epoch {epoch}: greedy wins {summary['outcomes']['success']}/"
                  f"{summary['episodes_started']}", flush=True)
            if writer is not None:
                for policy, values in evaluated["summaries"].items():
                    for task, values in values["tasks"].items():
                        writer.add_scalar(f"imitation_eval/{policy}/{task}/success_rate",
                                          values["success_rate_lower"], epoch)
                writer.flush()
            validation_likelihood(epoch)
            if (eval_all_levels and all_levels and phase == "evaluation"
                    and evaluated.get("policy_and_reward_frozen") is True
                    and evaluated.get("protocol_sha256") == protocol_hash):
                last_full_evaluation = (identity, evaluation_path, evaluated)
        if reused_initial is None:
            evaluate(0, all_levels=args.final_all_levels, phase="initial_evaluation")
        else:
            validation_likelihood(0)
            print("epoch 0: reused validated frozen measurement from "
                  f"{args.reuse_initial_evaluation}", flush=True)
        progress.phase("imitation", total=args.epochs * batches_per_epoch,
                       unit="optimizer_steps")

        def epoch_done(row):
            if args.training_backend == "mlx":
                import mlx.core as mx

                row["mlx_memory_bytes"] = {"active": mx.get_active_memory(),
                                           "cache": mx.get_cache_memory(),
                                           "peak": mx.get_peak_memory()}
            # A visited wall-limited epoch is not a completed epoch.
            completed = status["epochs_completed"] + int(row["complete"])
            progress.update(force=True, epoch=row["epoch"], epochs=completed,
                            epochs_completed=completed, optimizer_steps=row["optimizer_steps"],
                            nll=row["nll"], training_budget_seconds=row["seconds"],
                            epoch_training_seconds=row["epoch_training_seconds"],
                            phase_units_done=row["optimizer_steps"],
                            batch=math.ceil(row["episodes"] / args.minibatch_episodes))
            append_json(output / "training.jsonl", row)
            nll_text = f"{row['nll']:.4f}" if row["nll"] is not None else "no batches"
            objective_text = ""
            if ((args.action_balance == "active_idle" or args.canonical_replay_weight != 1.0)
                    and row.get("objective") is not None):
                objective_text = f"; optimization objective {row['objective']:.4f}"
            for label in ("active", "idle"):
                if row.get(f"{label}_nll") is not None:
                    objective_text += f"; {label} NLL {row[f'{label}_nll']:.4f}"
            print(f"epoch {row['epoch']}: NLL {nll_text}; "
                  f"updates {row['optimizer_steps']}{objective_text}", flush=True)
            if writer is not None and row["nll"] is not None:
                writer.add_scalar("imitation/joint_action_nll", row["nll"], row["epoch"])
                for task, nll in row["per_task_nll"].items():
                    writer.add_scalar(f"imitation/{task}/joint_action_nll", nll, row["epoch"])
                if args.action_balance == "active_idle" and row.get("objective") is not None:
                    writer.add_scalar("imitation/balanced_objective", row["unweighted_objective"],
                                      row["epoch"])
                    for task, value in row.get("per_task_unweighted_objective", {}).items():
                        writer.add_scalar(f"imitation/{task}/balanced_objective", value,
                                          row["epoch"])
                if row.get("objective") is not None:
                    writer.add_scalar("imitation/optimization_objective", row["objective"],
                                      row["epoch"])
                    writer.add_scalar("imitation/mean_batch_objective", row["mean_batch_objective"],
                                      row["epoch"])
                    for domain, values in row["replay_weighting"]["domains"].items():
                        writer.add_scalar(f"imitation/replay/{domain}/effective_mass_fraction",
                                          values["effective_mass_fraction"], row["epoch"])
                for label in ("active", "idle"):
                    if row.get(f"{label}_nll") is not None:
                        writer.add_scalar(f"imitation/{label}_nll", row[f"{label}_nll"],
                                          row["epoch"])
                writer.flush()
            if row["complete"] and row["epoch"] % args.eval_every == 0:
                evaluate(row["epoch"], all_levels=eval_all_levels)
                progress.phase("imitation", total=args.epochs * batches_per_epoch,
                               done=status["optimizer_steps"], unit="optimizer_steps")
            _save_epoch_snapshot(
                row, every=args.checkpoint_every, max_epochs=args.epochs, output=output,
                progress=progress, agent=agent, optimizer=optimizer, reward=reward,
                config=config, saved=saved, metadata=metadata, engine=engine)

        def update_done(row):
            progress.update(optimizer_update=True, epoch=row["epoch"],
                            optimizer_steps=row["optimizer_steps"],
                            batch=(row["optimizer_steps"] - 1) % batches_per_epoch + 1,
                            phase_units_done=row["optimizer_steps"], nll=row["nll"],
                            gradient_norm=row["gradient_norm"],
                            training_budget_seconds=row["seconds"])

        def epoch_started(row):
            progress.update(force=True, epoch=row["epoch"], batch=0,
                            training_budget_seconds=row["seconds"])

        if corpus_reuse is not None:
            corpus_reuse.assert_unchanged()
        training_started = time.monotonic()

        def feature_provider(index):
            return _training_feature(features[index], backend=args.training_backend,
                                     dtype=agent.summary_query.dtype)

        result = train_imitation(
            agent, dataset, epochs=args.epochs, backend=args.training_backend,
            optimizer=optimizer, mlx_backend=engine, learning_rate=config.learning_rate,
            weight_decay=config.weight_decay, minibatch_episodes=args.minibatch_episodes,
            max_grad_norm=config.max_grad_norm,
            action_balance=args.action_balance,
            canonical_replay_weight=args.canonical_replay_weight,
            canonical_anchor_coefficient=args.canonical_anchor_coefficient,
            canonical_anchor_exclude_corrections=canonical_anchor_exclude_corrections,
            screen_text_provider=screen_text_provider,
            seed=args.dataset_seed, feature_provider=feature_provider,
            on_epoch=epoch_done, on_update=update_done, on_epoch_start=epoch_started,
            max_wall_seconds=args.max_wall_seconds,
        )
        training_budget_seconds = time.monotonic() - training_started
        metadata["canonical_anchor"].update(result.get("canonical_anchor", {}))
        atomic_json(output / "metadata.json", metadata)
        if corpus_reuse is not None:
            corpus_reuse.assert_unchanged()
        evaluate(status["epoch"], all_levels=args.final_all_levels, phase="final_evaluation")
        progress.phase("saving", total=1, unit="checkpoints")
        if tensor_hash(reward) != reward_hash:
            raise RuntimeError("imitation unexpectedly changed the frozen reward model")
        checkpoint = save_sequence_checkpoint(
            agent, optimizer, reward, config, output, rollout=0,
            decisions=saved["decisions"],
            revision=saved["weights_revision"] + result["optimizer_steps"],
        )
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        link_matching_playbacks(output, checkpoint, manifest)
        result.update(checkpoint=str(checkpoint), initial_agent_sha256=metadata[
            "initial_agent_sha256"], final_agent_sha256=tensor_hash(agent),
            reward_sha256=reward_hash, demonstration_decisions=sum(map(len, dataset.episodes)),
            wall_seconds=time.monotonic() - started,
            training_budget_seconds=training_budget_seconds)
        atomic_json(output / "summary.json", result)
        progress.update(force=True, phase_units_done=1, checkpoint=str(checkpoint))
        if writer is not None:
            completed_writer, writer = writer, None
            completed_writer.close()
        progress.phase("finished", status="completed", checkpoint=str(checkpoint),
                       stop_reason=result["stop_reason"],
                       training_budget_seconds=training_budget_seconds)
        from carryrl.training.storage_maintenance import maybe_launch

        storage_root = next((parent.parent for parent in output.resolve().parents
                             if parent.name == "runs"), None)
        if storage_root is not None:
            maybe_launch(storage_root)
    except BaseException as error:
        recovery_save = None
        if agent is not None and status["optimizer_steps"]:
            def recovery_save():
                if (output / "latest.json").exists():
                    return None
                return save_sequence_checkpoint(
                    agent, optimizer, reward, config, output, rollout=0,
                    decisions=saved["decisions"],
                    revision=saved["weights_revision"] + status["optimizer_steps"],
                )

        failed_writer, writer = writer, None
        _record_training_failure(
            error, progress=progress,
            sync_to_torch=engine.sync_to_torch if engine is not None else None,
            save_checkpoint=recovery_save,
            close_writer=failed_writer.close if failed_writer is not None else None)
        raise
    finally:
        if writer is not None:
            remaining_writer, writer = writer, None
            try:
                remaining_writer.close()
            except BaseException as error:
                _record_training_failure(error, progress=progress)
                raise


if __name__ == "__main__":
    main()
