"""Bounded data-only preparation of the fixed ordinary O1 and O2 comparisons."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import resource
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from carryrl.envs.atomic_suite import SUITE_VERSION, TASKS
from carryrl.evaluation.imitation import imitation_validation_protocol
from carryrl.training.feature_store import (
    FrozenFeatureReuse,
    FrozenFeatureStore,
    FrozenFeatureWriter,
    feature_cache_identity,
)
from carryrl.training.imitation import (
    DemonstrationDataset,
    SavedDemonstrationReuse,
    audit_correction_replacement,
    collect_scripted_corrections,
    collect_scripted_demonstrations,
    demonstration_episode_digest,
    load_correction_replacement_bundle,
    validate_dataset_runtime_holdout,
)
from carryrl.training.retention import _argv, _checkpoint, _file_pin, _settings, json_sha256
from carryrl.training.run import atomic_json, provenance
from carryrl.training.sequence import (
    load_sequence_checkpoint,
    sequence_evaluation_protocol,
    tensor_hash,
)
from carryrl.training.sequence_config import SequenceRunConfig

FORMAT = "carryrl.ordinary_preparation.v1"
GIB = 1024**3


@dataclass(frozen=True)
class _Schedule:
    correction_tasks: tuple[str, ...]
    corrections_per_task: int

    @property
    def teachers(self):
        return 64 * len(TASKS)

    @property
    def corrections(self):
        return self.corrections_per_task * len(self.correction_tasks)

    @property
    def episodes(self):
        return self.teachers + self.corrections

    @property
    def batches(self):
        return math.ceil(self.episodes / 8)

    def collection(self):
        return {"teacher_episodes_per_task": 64, "teacher_seed": 3176,
            "teacher_tasks": list(TASKS), "correction_tasks": list(self.correction_tasks),
            "correction_episodes_per_task": self.corrections_per_task, "correction_seed": 8176,
            "expert_probability": 0., "learner_prefix_steps": 0,
            "control_level_bank_seed": 20261008, "difficulty": 1., "max_steps": 80}


def _schedule(experiment):
    if experiment == "O1":
        return _Schedule(("navigation",), 8)
    if experiment == "O2":
        return _Schedule(("camera", "interaction", "causality", "key_door", "hazards", "chain"), 4)
    raise ValueError("ordinary experiment must be O1 or O2")


def _anchor_coefficient(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("ordinary canonical anchor coefficient must be finite and nonnegative")
    return float(value)


def _epoch_orders(schedule):
    rng, orders = random.Random(3176), []
    for _ in range(2):
        order = list(range(schedule.episodes))
        rng.shuffle(order)
        orders.append(order)
    return orders


def _fit_common(config, checkpoint, corpus, cache, frame_budget, feature_budget, experiment):
    common = dict(config=str(config), warm_start=str(checkpoint), training_backend="mlx", epochs=2,
        evaluation_difficulty=1., level_bank_seed=20261008, dataset_seed=3176,
        learning_rate=3e-5, action_balance="active_idle", canonical_replay_weight=1.,
        minibatch_episodes=8, eval_every=1, checkpoint_every=1, validation_nll=True,
        validation_nll_episodes_per_task=2, max_wall_seconds=1800.,
        feature_budget_mib=feature_budget, frame_budget_mib=frame_budget,
        feature_dtype="float16", feature_cache_dir=str(cache),
        feature_reuse_run=str(corpus), demonstration_reuse_run=str(corpus),
        final_all_levels=True, screen_text_language=None)
    if experiment == "O2":
        common["eval_all_levels"] = True
    return common


def _rss():
    # macOS reports bytes; Linux reports KiB. This is an observed process peak, not RAM sum.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _clone(dataset):
    """Share immutable RGB backing and immutable episodes until appending new rows."""
    other = DemonstrationDataset(max_frame_bytes=dataset.max_frame_bytes)
    other.frames = list(dataset.frames)
    other.episodes = list(dataset.episodes)
    other._index = dict(dataset._index)
    other.frame_bytes = dataset.frame_bytes
    return other


def _publish_rgb(directory, dataset):
    destination = directory / "demonstration_frames.npz"
    with ZipFile(destination, "x", compression=ZIP_DEFLATED, allowZip64=True) as archive:
        for index, frame in enumerate(dataset.frames):
            if index % 128 == 0 and shutil.disk_usage(directory).free < 8 * GIB:
                raise OSError("ordinary preparation reached its 8 GiB free disk floor")
            with archive.open(f"frame_{index:06}.npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, frame, allow_pickle=False)
    if destination.stat().st_size > 4 * GIB:
        raise OSError("ordinary RGB archive exceeds the public loader's 4 GiB limit")


def _feature_identity(dataset, agent):
    vision = agent.vision
    used = sorted({index for row in dataset.episodes for index in row.frame_indices})
    return feature_cache_identity(dataset.frames, used, vision_sha256=tensor_hash(vision),
        preprocessing={"implementation": f"{type(vision).__module__}.{type(vision).__name__}",
            "preprocess": "letterbox_bicubic_antialias_imagenet_v1",
            "image_size": vision.image_size, "patch_size": vision.patch_size,
            "patch_grid": list(vision.patch_grid)},
        feature_shape=(math.prod(vision.patch_grid), vision.hidden_size), dtype="float16")


def _feature_source_pins(source_run):
    """Pin every RGB archive and feature manifest needed by layered public replay."""
    pins, seen = {}, set()
    directory = Path(source_run).resolve(strict=True)
    while True:
        if directory in seen or len(seen) >= 16:
            raise ValueError("ordinary feature ancestry has a cycle or exceeds 16 runs")
        seen.add(directory)
        for name in ("feature_manifest.json", "demonstration_frames.npz"):
            pins[str(directory / name)] = _file_pin(directory / name)
        record = json.loads((directory / "feature_manifest.json").read_text())
        if "feature_reuse" not in record:
            return pins
        directory = Path(record["feature_reuse"]["source_run"]).resolve(strict=True)


def _check_pins(pins):
    for filename, pin in pins.items():
        if _file_pin(filename) != pin:
            raise ValueError(f"ordinary pinned file changed: {filename}")


def _freeze(dataset, agent, source_run, cache_root, budget, *, batch_size=8):
    """Reuse only authenticated RGB identities; stream missing raw patches to disk."""
    identity = _feature_identity(dataset, agent)
    reuse = FrozenFeatureReuse.prepare(source_run, dataset.frames, identity["indices"], identity,
                                      cache_root=cache_root, max_bytes=budget)
    missing = reuse.missing_indices
    encoded_count = 0
    if missing:
        shard = FrozenFeatureStore.load(cache_root, reuse.shard_identity, max_bytes=budget)
        if shard is None:
            with FrozenFeatureWriter(cache_root, reuse.shard_identity, max_bytes=budget) as writer:
                with torch.inference_mode():
                    for start in range(0, len(missing), batch_size):
                        indices = missing[start:start + batch_size]
                        batch = torch.from_numpy(np.stack([dataset.frames[i] for i in indices]))
                        encoded = agent.encode_frames(batch).detach().to("cpu", dtype=torch.float16)
                        writer.write(indices, encoded.numpy())
                        encoded_count += len(indices)
                        del batch, encoded
                        if _rss() > 10 * GIB:
                            raise MemoryError("ordinary preparation peak RSS exceeded 10 GiB")
                shard = writer.publish()
        reuse.bind_shard(shard)
    manifest = reuse.manifest
    stores = {Path(pack["directory"]).resolve() for key in ("source_pack", "shard_pack")
              if (pack := manifest[key]) is not None}
    # Layered source manifests can reference several leaves; preserve authoritative mapping.
    if isinstance(reuse.original, FrozenFeatureStore):
        stores.add(reuse.original.directory.resolve())
    else:
        stores.update(store.directory.resolve() for store, _ in reuse.original.rows.values())
    pins = {str(directory / name): _file_pin(directory / name)
            for directory in sorted(stores) for name in ("manifest.json", "features.npy")}
    pins.update(_feature_source_pins(source_run))
    return {"source": "frozen DINO before all trainable layers", "dtype": "float16",
        "bytes": reuse.logical_bytes, "budget_bytes": budget, "frames": len(identity["indices"]),
        "vision_sha256": identity["vision_sha256"], "feature_reuse": manifest,
        "newly_encoded_frames": encoded_count, "reused_frames": reuse.reused_count,
        "storage": "verified read-only NumPy memmaps; bounded encoding batches"}, pins


def _write_corpus(directory, dataset, config, checkpoint, features, *, replacement=None):
    directory.mkdir()
    atomic_json(directory / "config.json", config.to_dict())
    metadata = {"kind": "imitation_training",  # compatibility schema, not an execution claim
        "phase": "ordinary_data_preparation", "suite_version": SUITE_VERSION,
        "config": config.to_dict(), "source_checkpoint": checkpoint["path"],
        "source_checkpoint_sha256": checkpoint["sha256"],
        "initial_agent_sha256": checkpoint["agent_sha256"],
        "reward_sha256": checkpoint["reward_sha256"],
        "preparation_only": True, "status": "prepared_not_executed",
        "training_executed": False, "optimizer_allocated": False, "updates": 0,
        "optimizer_state_preserved_in_source_checkpoint": True,
        "collection": {"episodes": len(dataset.episodes),
            "per_task": {task: sum(row.task == task for row in dataset.episodes)
                         for task in config.tasks},
            "bank_seeds": sorted({row.level_bank_seed for row in dataset.episodes}),
            "split": "train", "max_steps": config.max_episode_steps}}
    if replacement is not None:
        metadata["correction_replacement"] = replacement
    atomic_json(directory / "metadata.json", metadata)
    atomic_json(directory / "demonstrations.json", dataset.provenance())
    _publish_rgb(directory, dataset)
    atomic_json(directory / "feature_manifest.json", {
        **features, "demonstrations_sha256": _file_pin(
            directory / "demonstrations.json")["sha256"]})
    reuse = SavedDemonstrationReuse.prepare(directory, config=config,
                                           max_frame_bytes=dataset.max_frame_bytes)
    before = [demonstration_episode_digest(dataset, row) for row in dataset.episodes]
    after = [demonstration_episode_digest(reuse.dataset, row) for row in reuse.dataset.episodes]
    if before != after or len(dataset.frames) != len(reuse.dataset.frames):
        raise ValueError("ordinary exported corpus RGB/targets/history round-trip mismatch")
    summary = {**reuse.summary, "ordered_episode_sha256": before,
               "episode_order_sha256": json_sha256(before), "roundtrip_verified": True}
    del reuse
    gc.collect()
    return summary


def _forecast(control, treatment, agent, source_run, cache_root, checkpoint, output, budget):
    """Conservative forecast includes both RGB archives, missing features, and future fit."""
    identity = _feature_identity(control, agent)
    reuse = FrozenFeatureReuse.prepare(source_run, control.frames, identity["indices"], identity,
                                      cache_root=cache_root, max_bytes=budget)
    # All treatment decisions absent from control's content need at most one extra feature.
    control_rgb = {hashlib.sha256(control.frames[i].tobytes()).digest()
                   for i in identity["indices"]}
    added = {hashlib.sha256(treatment.frames[i].tobytes()).digest()
             for row in treatment.episodes for i in row.frame_indices} - control_rgb
    row_bytes = math.prod(identity["feature_shape"]) * 2
    feature_bytes = (len(reuse.missing_indices) + len(added)) * row_bytes
    rgb_bytes = control.frame_bytes + treatment.frame_bytes
    # Four future checkpoint archives, plus duplicated packaged language resources.
    checkpoint_bytes = 4 * sum(pin["bytes"] for pin in checkpoint["files"].values())
    reserved_misc = 512 * 1024**2  # JSON, evaluation traces, archive headers and partial writes
    total = rgb_bytes + feature_bytes + checkpoint_bytes + reserved_misc
    free = shutil.disk_usage(output.parent).free
    if total > 12 * GIB or free - total < 8 * GIB:
        raise OSError(f"ordinary pair disk forecast rejected: {total} new bytes, {free} free")
    return {"forecast_new_bytes": total, "rgb_uncompressed_upper_bound": rgb_bytes,
        "new_features_upper_bound": feature_bytes, "future_four_checkpoints": checkpoint_bytes,
        "misc_reservation": reserved_misc, "free_before": free,
        "free_after_forecast": free - total, "new_file_budget_bytes": 12 * GIB,
        "minimum_free_bytes": 8 * GIB}


def _future_outputs(output, control_output, treatment_output):
    future = {"control": Path(control_output or output / "control_training").resolve(),
              "treatment": Path(treatment_output or output / "treatment_training").resolve()}
    protected = [output / name for name in ("control_corpus", "treatment_corpus",
        "shared_base_config.json", "manifest.json", "control_manifest.json",
        "treatment_manifest.json")]
    if (len(set(future.values())) != 2
            or any(path.exists() or path.is_symlink() for path in future.values())
            or any(left in right.parents for left in future.values() for right in future.values())):
        raise ValueError("future training output paths must be separate, absent directories")
    if any(path == item or path in item.parents or item in path.parents
           for path in future.values() for item in protected):
        raise ValueError("future training outputs overlap preparation artifacts")
    return future


def prepare_ordinary(source_checkpoint, expected_checkpoint_sha256, replacement_bundle,
                        expected_bundle_sha256, output, *, feature_cache_dir,
                        source_config=None, feature_reuse_run=None, expected_actor_sha256=None,
                        control_output=None, treatment_output=None,
                        frame_budget_mib=1536, feature_budget_mib=14336, on_episode=None,
                        experiment="O1", canonical_anchor_coefficient=0.):
    """Prepare the chosen fixed corpus pair without constructing an optimizer or fitting."""
    schedule = _schedule(experiment)
    coefficient = _anchor_coefficient(canonical_anchor_coefficient)
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("ordinary preparation output must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_checkpoint = Path(source_checkpoint).resolve(strict=True)
    source_run = source_checkpoint.parents[1]
    feature_reuse_run = Path(feature_reuse_run or source_run).resolve(strict=True)
    source_config = Path(source_config or source_run / "config.json").resolve(strict=True)
    feature_cache_dir = Path(feature_cache_dir).resolve()
    checkpoint = _checkpoint(source_checkpoint, expected_checkpoint_sha256)
    if expected_actor_sha256 is not None and checkpoint["agent_sha256"] != expected_actor_sha256:
        raise ValueError("ordinary source actor differs from expected SHA256")
    original = SequenceRunConfig.load(source_config)
    if (original.tasks != TASKS or original.max_episode_steps != 80
            or original.env_image_size != 96 or original.visual_config is not None
            or original.screen_text_language is not None
            or any(not original.episode_instructions.get(task) for task in TASKS)):
        raise ValueError("ordinary requires all 16 tasks, canonical96/max80, "
                         "explicit instructions, no OCR")
    if type(frame_budget_mib) is not int or not 1 <= frame_budget_mib <= 4096:
        raise ValueError("invalid bounded ordinary RGB budget")
    if type(feature_budget_mib) is not int or not 1 <= feature_budget_mib <= 16384:
        raise ValueError("invalid bounded ordinary feature budget")
    config = replace(original, device="cpu", training_backend="mlx", learning_rate=3e-5,
        level_bank_seed=20261008, curriculum_difficulty=1., evaluation_difficulty=1.,
        curriculum_warmup_rollouts=0, eval_all_levels=False, vision_cache_frames=0,
        reset_optimizer_on_warm_start=False)
    future = _future_outputs(output, control_output, treatment_output)
    initial_cache_files = {path.resolve() for path in feature_cache_dir.rglob("*")
                           if path.is_file()}
    initial_provenance = provenance()
    torch.set_num_threads(config.torch_threads)
    agent, optimizer, reward, saved = load_sequence_checkpoint(
        source_checkpoint, config, evaluation_only=True, load_optimizer=False)
    if optimizer is not None:
        raise AssertionError("data-only loader constructed an optimizer")
    del saved  # release loaded archive/mmap references, including irrelevant tensors
    agent.eval().requires_grad_(False)
    if tensor_hash(agent) != checkpoint["agent_sha256"] or tensor_hash(reward) != checkpoint[
            "reward_sha256"]:
        raise ValueError("loaded ordinary producer differs from pinned tensors")
    del reward
    gc.collect()
    replacements, replacement = load_correction_replacement_bundle(replacement_bundle,
        expected_sha256=expected_bundle_sha256, correction_tasks=schedule.correction_tasks,
        episodes_per_task=schedule.corrections_per_task, seed=8176, difficulty=1.,
        max_steps=80, level_bank_seed=20261008,
        episode_instructions=config.episode_instructions,
        source_identity=json.loads((source_checkpoint / "manifest.json").read_text()),
        initial_agent_sha256=checkpoint["agent_sha256"], reward_sha256=checkpoint["reward_sha256"],
        max_frame_bytes=frame_budget_mib * 1024**2)
    options = dict(seed=3176, difficulty=1., image_size=96, max_steps=80,
        level_bank_seed=20261008, episode_instructions=config.episode_instructions,
        max_frame_bytes=frame_budget_mib * 1024**2, on_episode=on_episode)
    control = collect_scripted_demonstrations(64, tasks=TASKS, **options)
    treatment = _clone(control)
    teacher_hashes = [demonstration_episode_digest(control, row) for row in control.episodes]
    if set(replacements) != set(schedule.correction_tasks):
        raise ValueError("ordinary replacement tasks differ from the fixed schedule")
    corrections = collect_scripted_corrections(agent, schedule.corrections_per_task,
        tasks=schedule.correction_tasks,
        expert_probability=0., learner_prefix_steps=0, **{**options, "seed": 8176})
    control.extend(corrections)
    del corrections
    for task in schedule.correction_tasks:
        treatment.extend(replacements[task])
    if len(teacher_hashes) != schedule.teachers or any(
            len(dataset.episodes) != schedule.episodes for dataset in (control, treatment)):
        raise ValueError("ordinary corpus counts differ from the fixed schedule")
    for dataset in (control, treatment):
        validate_dataset_runtime_holdout(dataset, level_bank_seed=20261008,
                                         evaluation_difficulty=1.)
    resources = _forecast(control, treatment, agent, feature_reuse_run, feature_cache_dir,
                          checkpoint, output, feature_budget_mib * 1024**2)
    output.mkdir()
    atomic_json(output / "shared_base_config.json", config.to_dict())
    arms, cache_pins = {}, {}
    for name, dataset, reuse_run in (("control", control, feature_reuse_run),
                                     ("treatment", treatment, output / "control_corpus")):
        features, pins = _freeze(dataset, agent, reuse_run, feature_cache_dir,
                                  feature_budget_mib * 1024**2)
        cache_pins.update(pins)
        corpus_path = output / f"{name}_corpus"
        corpus = _write_corpus(corpus_path, dataset, config, checkpoint, features,
                               replacement=replacement if name == "treatment" else None)
        common = _fit_common(output / "shared_base_config.json", source_checkpoint,
            corpus_path, feature_cache_dir, frame_budget_mib, feature_budget_mib, experiment)
        settings = _settings(common, coefficient=coefficient, output=future[name])
        arms[name] = {"output": str(future[name]), "settings": settings,
                      "argv": _argv(settings), "corpus": corpus, "features": features}
    for name, arm in arms.items():
        atomic_json(output / f"{name}_manifest.json", {**arm,
            "status": "prepared_not_executed", "updates": 0,
            "expected_optimizer_updates": 2 * schedule.batches, "training_executed": False})
    audit = audit_correction_replacement(treatment, output / "control_corpus",
                                         replacements, replacement)
    # Public loader round-trip precedes a second independent replacement audit.
    imported = SavedDemonstrationReuse.prepare(output / "treatment_corpus", config=config,
                                                max_frame_bytes=treatment.max_frame_bytes)
    roundtrip_audit = audit_correction_replacement(imported.dataset, output / "control_corpus",
                                                   replacements, replacement)
    if audit != roundtrip_audit or audit["changed_slots"] != schedule.corrections:
        raise ValueError("ordinary replacement audit differs after public-loader round-trip")
    if tensor_hash(agent) != checkpoint["agent_sha256"]:
        raise ValueError("frozen producer actor changed during preparation")
    del imported, agent
    gc.collect()
    if _checkpoint(source_checkpoint, expected_checkpoint_sha256) != checkpoint:
        raise ValueError("source checkpoint changed during ordinary preparation")
    if provenance() != initial_provenance:
        raise ValueError("source/lock changed during ordinary preparation")
    orders = _epoch_orders(schedule)
    evaluation = sequence_evaluation_protocol(replace(config, eval_all_levels=True))
    likelihood = imitation_validation_protocol(replace(config, eval_all_levels=False,
                                                       eval_episodes_per_task=2))
    new_cache_bytes = sum(path.stat().st_size for path in feature_cache_dir.rglob("*")
                          if path.is_file() and path.resolve() not in initial_cache_files)
    new_corpus_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    actual_new_bytes = new_cache_bytes + new_corpus_bytes
    if (actual_new_bytes + resources["future_four_checkpoints"]
            + resources["misc_reservation"] > 12 * GIB
            or shutil.disk_usage(output).free - resources["future_four_checkpoints"]
            - resources["misc_reservation"] < 8 * GIB or _rss() > 10 * GIB):
        raise OSError("ordinary preparation exceeded its disk or process RSS limit")
    plan = {"format": FORMAT, "experiment": experiment, "status": "prepared_not_executed",
        "training_executed": False, "optimizer_allocated": False, "updates": 0,
        "checkpoint_created": False, "model_loaded": True, "heldout_policy_rollouts": 0,
        "test_outcomes_read": False, "source_checkpoint": checkpoint,
        "source_config": {"path": str(source_config), **_file_pin(source_config)},
        "source_provenance": initial_provenance, "effective_config": config.to_dict(),
        "config_file": {"path": str(output / "shared_base_config.json"),
                        **_file_pin(output / "shared_base_config.json")},
        "replacement_bundle": replacement, "replacement_audit": audit,
        "collection_schedule": schedule.collection(),
        "teacher_ordered_episode_sha256": teacher_hashes,
        "teacher_order_sha256": json_sha256(teacher_hashes), "arms": arms,
        "feature_files": cache_pins, "epoch_orders": orders,
        "epoch_order_sha256": [json_sha256(order) for order in orders],
        "expected_epochs": 2, "batches_per_epoch": schedule.batches,
        "expected_optimizer_updates": 2 * schedule.batches,
        "evaluation_protocol": evaluation, "evaluation_protocol_sha256": json_sha256(evaluation),
        "validation_likelihood_protocol": likelihood,
        "validation_likelihood_protocol_sha256": json_sha256(likelihood),
        "resources": {**resources, "actual_new_preparation_bytes": actual_new_bytes,
                      "actual_new_cache_bytes": new_cache_bytes,
                      "actual_preparation_directory_bytes_before_manifest": new_corpus_bytes,
                      "observed_process_peak_rss_bytes": _rss(),
                      "free_after_preparation": shutil.disk_usage(output).free},
        "selection": {"epoch": 2, "require_complete_epochs": True,
                      "reject_partial_or_wall_limited_runs": True, "test_used": False}}
    atomic_json(output / "manifest.json", plan)
    return plan


def prepare_ordinary_o1(*args, **kwargs):
    """Backward-compatible entry point for the fixed navigation comparison."""
    return prepare_ordinary(*args, **kwargs, experiment="O1")


def load_ordinary_preparation(path, *, expected_sha256=None):
    """Recheck preparation pins without loading RGB, policies, or optimizer tensors."""
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    if expected_sha256 is not None and _file_pin(path)["sha256"] != expected_sha256:
        raise ValueError("ordinary manifest SHA256 mismatch")
    plan = json.loads(path.read_text())
    if (plan.get("format") != FORMAT or plan.get("status") != "prepared_not_executed"
            or plan.get("updates") != 0 or plan.get("optimizer_allocated") is not False
            or plan.get("training_executed") is not False):
        raise ValueError("unsupported or executed ordinary preparation")
    current = provenance()
    for key in ("source_and_lock_sha256", "source_identity_version", "dependencies"):
        if current[key] != plan["source_provenance"][key]:
            raise ValueError("ordinary pinned source/lock or dependency identity changed")
    schedule = _schedule(plan.get("experiment"))
    if plan.get("collection_schedule") != schedule.collection():
        raise ValueError("ordinary collection schedule differs from the fixed experiment")
    if (plan.get("expected_epochs") != 2 or plan.get("batches_per_epoch") != schedule.batches
            or plan.get("expected_optimizer_updates") != 2 * schedule.batches):
        raise ValueError("ordinary planned budget differs from the fixed schedule")
    teachers = plan["teacher_ordered_episode_sha256"]
    if len(teachers) != schedule.teachers or json_sha256(teachers) != plan["teacher_order_sha256"]:
        raise ValueError("ordinary teacher count or hash mismatch")
    if (plan["epoch_orders"] != _epoch_orders(schedule)
            or any(any(type(index) is not int for index in order)
                   for order in plan["epoch_orders"])):
        raise ValueError("ordinary epoch orders differ from the fixed shuffle permutations")
    if set(plan["arms"]) != {"control", "treatment"}:
        raise ValueError("ordinary preparation requires control and treatment arms")
    checkpoint = plan["source_checkpoint"]
    if _checkpoint(checkpoint["path"], checkpoint["sha256"]) != checkpoint:
        raise ValueError("ordinary pinned checkpoint changed")
    pins = {record["path"]: {key: record[key] for key in ("bytes", "sha256")}
            for record in (plan["source_config"], plan["config_file"])}
    pins.update(plan["feature_files"])
    bundle = plan["replacement_bundle"]
    if (set(bundle["selected_tasks"]) != set(schedule.correction_tasks)
            or bundle["episodes_per_task"] != schedule.corrections_per_task
            or bundle["replacement_episodes"] != schedule.corrections):
        raise ValueError("ordinary replacement bundle differs from the fixed schedule")
    pins.update({str(Path(bundle["bundle_directory"]) / name): pin
                 for name, pin in bundle["files"].items()})
    control_settings = plan["arms"]["control"]["settings"]
    coefficient = _anchor_coefficient(control_settings["canonical_anchor_coefficient"])
    for field, maximum in (("frame_budget_mib", 4096), ("feature_budget_mib", 16384)):
        if type(control_settings[field]) is not int or not 1 <= control_settings[field] <= maximum:
            raise ValueError("ordinary fit byte budget is invalid")
    for arm in plan["arms"].values():
        corpus = arm["corpus"]
        common = _fit_common(plan["config_file"]["path"], checkpoint["path"],
            corpus["source_run"], control_settings["feature_cache_dir"],
            control_settings["frame_budget_mib"], control_settings["feature_budget_mib"],
            plan["experiment"])
        if (plan["experiment"] == "O1" and "eval_all_levels" in control_settings
                and control_settings["eval_all_levels"] is True):
            common["eval_all_levels"] = True
        if (arm["settings"] != _settings(common, coefficient=coefficient, output=arm["output"])
                or arm["argv"] != _argv(arm["settings"])):
            raise ValueError("ordinary argv/settings mismatch")
        pins.update({str(Path(corpus["source_run"]) / name): pin
                     for name, pin in corpus["files"].items()})
        if (corpus["episodes"] != schedule.episodes
                or len(corpus["ordered_episode_sha256"]) != schedule.episodes
                or json_sha256(corpus["ordered_episode_sha256"]) != corpus["episode_order_sha256"]):
            raise ValueError("ordinary episode order mismatch")
        if corpus["ordered_episode_sha256"][:schedule.teachers] != teachers:
            raise ValueError("ordinary teachers differ between arms")
    audit = plan["replacement_audit"]
    if (audit["passed"] is not True or audit["changed_slots"] != schedule.corrections
            or audit["unchanged_slots"] != schedule.teachers
            or audit["episodes"] != schedule.episodes
            or audit["all_other_slots_identical"] is not True
            or audit["changed_positions"] != list(range(schedule.teachers, schedule.episodes))):
        raise ValueError("ordinary replacement audit differs from the fixed schedule")
    replacement_hashes = bundle["replacement_episode_content_sha256"]
    if (set(replacement_hashes) != set(schedule.correction_tasks)
            or any(len(replacement_hashes[task]) != schedule.corrections_per_task
                   for task in schedule.correction_tasks)
            or plan["arms"]["treatment"]["corpus"]["ordered_episode_sha256"][schedule.teachers:]
            != [digest for task in schedule.correction_tasks
                for digest in replacement_hashes[task]]):
        raise ValueError("ordinary replacement episode order differs from the fixed schedule")
    _check_pins(pins)
    config = SequenceRunConfig.from_dict(plan["effective_config"])
    if SequenceRunConfig.load(plan["config_file"]["path"]).to_dict() != config.to_dict():
        raise ValueError("ordinary shared effective config differs")
    for key in ("evaluation_protocol", "validation_likelihood_protocol"):
        if json_sha256(plan[key]) != plan[key + "_sha256"]:
            raise ValueError("ordinary protocol hash mismatch")
    if (plan["evaluation_protocol_sha256"] != json_sha256(sequence_evaluation_protocol(
            replace(config, eval_all_levels=True)))
            or plan["validation_likelihood_protocol_sha256"] != json_sha256(
                imitation_validation_protocol(replace(config, eval_all_levels=False,
                                                        eval_episodes_per_task=2)))):
        raise ValueError("ordinary protocol differs from the effective configuration")
    if [json_sha256(order) for order in plan["epoch_orders"]] != plan["epoch_order_sha256"]:
        raise ValueError("ordinary shuffle order mismatch")
    return plan
