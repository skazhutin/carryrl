"""Prepare an immutable, unexecuted paired canonical-retention experiment."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from carryrl.evaluation.imitation import imitation_validation_protocol
from carryrl.training.checkpoint_storage import checkpoint_read_lock
from carryrl.training.feature_store import FrozenFeatureReuse, FrozenFeatureStore
from carryrl.training.imitation import (
    SavedDemonstrationReuse,
    demonstration_episode_digest,
    imitation_replay_summary,
    validate_dataset_runtime_holdout,
)
from carryrl.training.run import atomic_json, provenance
from carryrl.training.sequence import (
    _episode_instruction,
    sequence_evaluation_protocol,
    sequence_evaluation_summaries,
)
from carryrl.training.sequence_config import SequenceRunConfig

FORMAT = "carryrl.retention_plan.v1"


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def _file_pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"pinned file must be a regular non-symlink file: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _positive(value, name, *, integer=False):
    if (isinstance(value, bool) or not isinstance(value, int if integer else (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be positive and finite")


def _checkpoint(path, expected_sha256):
    path = Path(path).resolve(strict=True)
    if not path.is_dir() or not _digest(expected_sha256):
        raise ValueError("source checkpoint requires a trusted archive SHA256")
    with checkpoint_read_lock(path):
        return _checkpoint_identity(path, expected_sha256)


def _checkpoint_identity(path, expected_sha256):
    files = {}
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ValueError("checkpoint must not contain symlinks")
        if child.is_file():
            files[str(child.relative_to(path))] = _file_pin(child)
    manifest = json.loads((path / "manifest.json").read_text())
    if (files.get("state.pt", {}).get("sha256") != expected_sha256
            or manifest.get("sha256") != expected_sha256
            or any(not _digest(manifest.get(key)) for key in ("agent_sha256", "reward_sha256"))
            or type(manifest.get("weights_revision")) is not int
            or manifest["weights_revision"] < 0):
        raise ValueError("source checkpoint identity differs from the trusted archive")
    return {"path": str(path), "files": files,
            **{key: manifest[key] for key in
               ("sha256", "agent_sha256", "reward_sha256", "weights_revision")},
            "identity_validation": "archive bytes verified; tensor identities from pinned manifest"}


def validate_initial_evaluation(source, *, checkpoint, config, agent_sha256, reward_sha256,
                              weights_revision):
    """Validate a prior measurement against this checkpoint and complete cohort."""
    checkpoint = checkpoint.resolve(strict=True)
    source = source.resolve(strict=True)
    if (checkpoint.parent.name != "checkpoints"
            or source.parent != (checkpoint.parent.parent / "evaluations").resolve()
            or not source.is_file()):
        raise ValueError("reused evaluation must belong to the warm-start run's evaluations")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if (manifest.get("agent_sha256") != agent_sha256
            or manifest.get("reward_sha256") != reward_sha256
            or manifest.get("weights_revision") != weights_revision):
        raise ValueError("reused evaluation checkpoint manifest does not match loaded weights")
    source_bytes = source.read_bytes()
    evaluated = json.loads(source_bytes)
    protocol = sequence_evaluation_protocol(config)
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (evaluated.get("split") != "validation"
            or evaluated.get("policy_and_reward_frozen") is not True
            or evaluated.get("agent_sha256") != agent_sha256
            or evaluated.get("reward_sha256", reward_sha256) != reward_sha256
            or evaluated.get("weights_revision") != weights_revision
            or evaluated.get("evaluation_seed") != config.eval_seed
            or evaluated.get("evaluation_difficulty") != protocol["difficulty"]
            or evaluated.get("level_bank_seed") != config.level_bank_seed
            or evaluated.get("protocol_sha256") != protocol_hash):
        raise ValueError("reused evaluation policy or protocol does not match this run")
    try:
        measured_at = evaluated["created_utc"]
        if datetime.fromisoformat(measured_at).tzinfo is None:
            raise ValueError("evaluation timestamp must include its timezone")
        records = evaluated["records"]
        expected = {(policy, task, index)
                    for policy in protocol["policies"] for task, count in protocol["counts"].items()
                    for index in range(count)}
        actual = [(row["policy"], row["task"], row["episode_index"]) for row in records]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("reused evaluation has incomplete or duplicate episode records")
        from carryrl.envs.atomic import _identity, atomic_level_banks, atomic_semantic_level_id

        for row in records:
            task, index = row["task"], row["episode_index"]
            action_seed = (config.eval_seed + config.tasks.index(task) * 100_000 + index) % 2**32
            if (row["seed"] != 1_000_000 + index or row["action_seed"] != action_seed
                    or row["difficulty"] != protocol["difficulty"]
                    or row["level_bank_seed"] != config.level_bank_seed
                    or row["instruction"] != _episode_instruction(config, task, row["seed"])
                    or not 1 <= row["steps"] <= config.max_episode_steps
                    or not (row["terminated"] or row["truncated"])):
                raise ValueError("reused evaluation episode metadata does not match its cohort")
            if config.eval_all_levels:
                bank = atomic_level_banks(task, config.level_bank_seed)["validation"]
                level = bank[index % len(bank)]
                if (row["level_index"] != index % len(bank)
                        or row["level_id"] != _identity(level)
                        or row["semantic_level_id"] != atomic_semantic_level_id(task, level)):
                    raise ValueError("reused evaluation records do not cover the expected bank")
        summaries = sequence_evaluation_summaries(
            records, tasks=config.tasks, policies=protocol["policies"])
        if evaluated["summaries"] != summaries:
            raise ValueError("reused evaluation summaries are inconsistent with its records")
    except (KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        raise ValueError("reused evaluation is malformed or incomplete") from error
    evaluated.update(epoch=0, weights_revision=weights_revision,
                     initial_evaluation_reuse={
                         "source": str(source),
                         "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                         "source_created_utc": measured_at,
                         "source_epoch": evaluated.get("epoch"),
                         "source_weights_revision": evaluated["weights_revision"],
                         "measurement_reused": True,
                     })
    return evaluated


def _feature_dependency_pins(source_run):
    """Pin RGB and manifest inputs of every public frozen-feature ancestor."""
    pins, seen = {}, set()
    directory = Path(source_run).resolve(strict=True)
    while True:
        if directory in seen or len(seen) >= 16:
            raise ValueError("retention feature ancestry has a cycle or exceeds 16 runs")
        seen.add(directory)
        for name in ("feature_manifest.json", "demonstration_frames.npz"):
            pins[str(directory / name)] = _file_pin(directory / name)
        manifest = json.loads((directory / "feature_manifest.json").read_text())
        reuse = manifest.get("feature_reuse")
        packs = ([Path(manifest["feature_cache_dir"])] if reuse is None else
                 [Path(reuse[key]["directory"]) for key in ("source_pack", "shard_pack")
                  if reuse.get(key) is not None])
        for pack in packs:
            for name in ("manifest.json", "features.npy"):
                pins[str(pack / name)] = _file_pin(pack / name)
        if reuse is None:
            return pins
        directory = Path(reuse["source_run"]).resolve(strict=True)


def _feature_source(source_run, *, cache_root, dataset, max_bytes, dtype):
    manifest = json.loads((source_run / "feature_manifest.json").read_text())
    identity = manifest.get("feature_reuse", {}).get("identity")
    if identity is None:
        pack = Path(manifest["feature_cache_dir"])
        identity = json.loads((pack / "manifest.json").read_text())["identity"]
    if identity.get("dtype") != dtype:
        raise ValueError("requested feature dtype differs from saved frozen features")
    used = sorted({index for row in dataset.episodes for index in row.frame_indices})
    reuse = FrozenFeatureReuse.prepare(source_run, dataset.frames, used, identity,
                                      cache_root=cache_root, max_bytes=max_bytes)
    if reuse.missing_indices:
        raise ValueError("retention preparation requires complete existing frozen features")
    stores = ([reuse.original] if isinstance(reuse.original, FrozenFeatureStore)
              else [row[0] for row in reuse.original.rows.values()])
    directories = sorted({store.directory.resolve() for store in stores})
    pins = {str(directory / name): _file_pin(directory / name)
            for directory in directories for name in ("manifest.json", "features.npy")}
    pins.update(_feature_dependency_pins(source_run))
    return {"identity": identity, "identity_sha256": json_sha256(identity),
            "logical_bytes": reuse.logical_bytes, "files": pins,
            "storage": "verified read-only NumPy memmaps; logical budget is not an RSS cap",
            "new_features": 0}


def _settings(common, *, coefficient, output):
    # JSON-normalized vars(args) of train_imitation.py, including inert collection defaults.
    settings = dict(config=common["config"], warm_start=common["warm_start"],
        allow_reward_composition_change=False, output=str(output), training_backend="mlx",
        epochs=40, episodes_per_task=16, difficulty=[1.0], evaluation_difficulty=None,
        level_bank_seed=None, correction_episodes_per_task=0, correction_tasks=None,
        correction_replacement_bundle=None, correction_replacement_sha256=None,
        replacement_control_run=None, expert_probability=0.5, learner_prefix_steps=0,
        styled_correction_episodes_per_task=0, styled_correction_expert_probability=0.35,
        styled_correction_prefix_steps=4, presentation_copies_per_episode=0, memory_cue_pairs=0,
        augmentation_themes=None, augmentation_hud_layouts=None, augmentation_textures=True,
        augmentation_canvas_padding=8,
        native_train_sizes=[], native_episodes_per_task=0, dataset_seed=8176, learning_rate=2e-4,
        action_balance="uniform", canonical_replay_weight=1.0,
        canonical_anchor_coefficient=coefficient, minibatch_episodes=8, eval_every=10,
        validation_nll=False, validation_nll_episodes_per_task=2, checkpoint_every=0,
        max_wall_seconds=1200, feature_budget_mib=2048, frame_budget_mib=256,
        feature_dtype="float32", feature_cache_dir=None, feature_reuse_run=None,
        appearance_bundle=None, appearance_bundle_sha256=None, appearance_arm=None,
        navigation_geometry_pairs=0, navigation_geometry_mode=None,
        runtime_diversity_axis=None, runtime_diversity_pairs=0, runtime_diversity_arm=None,
        demonstration_reuse_run=None, final_all_levels=False, reuse_initial_evaluation=None,
        tensorboard=None, screen_text_language=None)
    settings.update(common)
    return settings


_COMMAND_FIELDS = ("config", "warm_start", "output", "training_backend", "epochs",
    "evaluation_difficulty", "level_bank_seed", "dataset_seed", "learning_rate", "action_balance",
    "canonical_replay_weight", "canonical_anchor_coefficient", "minibatch_episodes", "eval_every",
    "validation_nll_episodes_per_task", "checkpoint_every", "max_wall_seconds",
    "feature_budget_mib",
    "frame_budget_mib", "feature_dtype", "feature_cache_dir", "feature_reuse_run",
    "demonstration_reuse_run", "screen_text_language")


def _argv(settings):
    script = Path(__file__).resolve().parents[3] / "scripts/train_imitation.py"
    argv = ["uv", "run", "--locked", "--extra", "mlx", "--extra", "monitor", "python", str(script)]
    for field in _COMMAND_FIELDS:
        if field == "screen_text_language" and settings.get(field) is None:
            continue
        argv.extend(["--" + field.replace("_", "-"), str(settings[field])])
    if settings.get("eval_all_levels", False):
        argv.append("--eval-all-levels")
    if settings.get("canonical_anchor_exclude_corrections", False):
        argv.append("--canonical-anchor-exclude-corrections")
    if settings.get("reuse_initial_evaluation") is not None:
        argv.extend(["--reuse-initial-evaluation", settings["reuse_initial_evaluation"]])
    return [*argv, "--validation-nll", "--final-all-levels"]


def prepare_retention_plan(source_run, source_checkpoint, expected_checkpoint_sha256, output, *,
                           positive_anchor_coefficient, epochs, learning_rate, dataset_seed,
                           max_wall_seconds, config_path=None, feature_cache_dir=None,
                           backend="mlx", minibatch_episodes=8, action_balance="active_idle",
                           canonical_replay_weight=1.0, evaluation_difficulty=1.0,
                           level_bank_seed=20261008, feature_dtype="float16",
                           feature_budget_mib=14336, frame_budget_mib=4096, eval_every=1,
                           checkpoint_every=1, validation_nll_episodes_per_task=2,
                           eval_all_levels=False, reuse_initial_evaluation=None):
    """Authenticate source artifacts and write JSON commands; never allocate a policy/optimizer."""
    for value, name in ((positive_anchor_coefficient, "positive anchor coefficient"),
                        (learning_rate, "learning rate"), (max_wall_seconds, "wall budget"),
                        (canonical_replay_weight, "canonical replay weight")):
        _positive(value, name)
    for value, name in ((epochs, "epochs"), (minibatch_episodes, "minibatch episodes"),
                        (feature_budget_mib, "feature budget"), (frame_budget_mib, "frame budget"),
                        (eval_every, "evaluation interval"),
                        (checkpoint_every, "checkpoint interval"),
                        (validation_nll_episodes_per_task, "validation NLL episodes")):
        _positive(value, name, integer=True)
    if (type(dataset_seed) is not int or not 0 <= dataset_seed < 2**32
            or backend not in {"mlx", "torch"} or action_balance not in {"active_idle", "uniform"}
            or feature_dtype not in {"float16", "float32"}):
        raise ValueError("invalid retention seed, backend, action balance or feature dtype")
    if type(eval_all_levels) is not bool:
        raise ValueError("eval_all_levels must be a boolean")
    source_run = Path(source_run).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("retention output must be new")
    output = output.resolve()
    checkpoint_path = Path(source_checkpoint).resolve(strict=True)
    if (output == source_run or source_run in output.parents
            or output == checkpoint_path or checkpoint_path in output.parents):
        raise ValueError("retention output must be separate from the source corpus")
    config_path = Path(config_path or source_run / "config.json").resolve(strict=True)
    source_config = SequenceRunConfig.load(config_path)
    config = replace(source_config, device="cpu", training_backend=backend,
        learning_rate=learning_rate, level_bank_seed=level_bank_seed,
        evaluation_difficulty=evaluation_difficulty, curriculum_warmup_rollouts=0,
        eval_all_levels=False)
    checkpoint = _checkpoint(source_checkpoint, expected_checkpoint_sha256)
    initial_evaluation = None
    if reuse_initial_evaluation is not None:
        source = Path(reuse_initial_evaluation).resolve(strict=True)
        source_pin = _file_pin(source)
        measured = validate_initial_evaluation(source, checkpoint=checkpoint_path,
            config=replace(config, eval_all_levels=True),
            **{key: checkpoint[key] for key in
               ("agent_sha256", "reward_sha256", "weights_revision")})
        initial_evaluation = {"path": str(source), **source_pin,
            "agent_sha256": checkpoint["agent_sha256"],
            "reward_sha256": checkpoint["reward_sha256"],
            "weights_revision": checkpoint["weights_revision"],
            "protocol_sha256": measured["protocol_sha256"],
            "records": len(measured["records"]),
            "validation": "complete authenticated frozen checkpoint and semantic cohort"}
    reuse = SavedDemonstrationReuse.prepare(source_run, config=config,
                                           max_frame_bytes=frame_budget_mib * 1024**2)
    config = replace(config, curriculum_difficulty=reuse.summary["source_difficulty"][
        "curriculum_difficulty"])
    validate_dataset_runtime_holdout(reuse.dataset, level_bank_seed=level_bank_seed,
                                     evaluation_difficulty=evaluation_difficulty)
    feature_record = json.loads((source_run / "feature_manifest.json").read_text())
    if feature_cache_dir is None:
        direct_pack = feature_record.get("feature_cache_dir")
        if direct_pack is None:
            raise ValueError("layered frozen features require an explicit feature cache root")
        feature_cache_dir = Path(direct_pack).parent
    feature_cache_dir = Path(feature_cache_dir).resolve(strict=True)
    features = _feature_source(source_run, cache_root=feature_cache_dir, dataset=reuse.dataset,
        max_bytes=feature_budget_mib * 1024**2, dtype=feature_dtype)
    replay = imitation_replay_summary(reuse.dataset,
                                      canonical_replay_weight=canonical_replay_weight)
    if not replay["domains"].get("canonical_unstyled_96", {}).get("episodes"):
        raise ValueError("positive anchor requires eligible canonical episodes")
    episodes = len(reuse.dataset.episodes)
    order = [demonstration_episode_digest(reuse.dataset, row) for row in reuse.dataset.episodes]
    rng = random.Random(dataset_seed)
    epoch_orders = []
    for _ in range(epochs):
        indices = list(range(episodes))
        rng.shuffle(indices)
        epoch_orders.append(json_sha256(indices))
    common = dict(config=str(output / "shared_base_config.json"), warm_start=checkpoint["path"],
        training_backend=backend, epochs=epochs, evaluation_difficulty=evaluation_difficulty,
        level_bank_seed=level_bank_seed, dataset_seed=dataset_seed, learning_rate=learning_rate,
        action_balance=action_balance, canonical_replay_weight=canonical_replay_weight,
        minibatch_episodes=minibatch_episodes, eval_every=eval_every,
        checkpoint_every=checkpoint_every,
        validation_nll=True, validation_nll_episodes_per_task=validation_nll_episodes_per_task,
        max_wall_seconds=max_wall_seconds, feature_budget_mib=feature_budget_mib,
        frame_budget_mib=frame_budget_mib, feature_dtype=feature_dtype,
        feature_cache_dir=str(feature_cache_dir), feature_reuse_run=str(source_run),
        demonstration_reuse_run=str(source_run), final_all_levels=True,
        screen_text_language=config.screen_text_language)
    if eval_all_levels:
        common["eval_all_levels"] = True
    if initial_evaluation is not None:
        common["reuse_initial_evaluation"] = initial_evaluation["path"]
    arms = {}
    for name, coefficient in (("anchor_off", 0.0), ("anchor_on", positive_anchor_coefficient)):
        settings = _settings(common, coefficient=coefficient, output=output / name)
        arms[name] = {"output": settings["output"], "settings": settings, "argv": _argv(settings)}
    evaluation = sequence_evaluation_protocol(replace(config, eval_all_levels=True))
    likelihood = imitation_validation_protocol(replace(config, eval_all_levels=False,
        eval_episodes_per_task=validation_nll_episodes_per_task))
    scheduled = sequence_evaluation_protocol(replace(config, eval_all_levels=eval_all_levels))
    plan = dict(format=FORMAT, preparation_revision=2, status="prepared_not_executed",
        model_loaded=False,
        optimizer_allocated=False, training_executed=False, checkpoint_created=False,
        updates=0, source_provenance=provenance(), source_checkpoint=checkpoint,
        source_config={"path": str(config_path), **_file_pin(config_path)},
        effective_config=config.to_dict(), source_corpus={**reuse.summary,
            "provenance": reuse.dataset.provenance(), "ordered_episode_sha256": order,
            "episode_order_sha256": json_sha256(order), "runtime_holdout_guard_passed": True},
        frozen_features=features, replay=replay, arms=arms,
        expected_epochs=epochs,
        expected_optimizer_updates=epochs * math.ceil(episodes / minibatch_episodes),
        batches_per_epoch=math.ceil(episodes / minibatch_episodes), epoch_order_sha256=epoch_orders,
        evaluation_protocol=evaluation, evaluation_protocol_sha256=json_sha256(evaluation),
        scheduled_evaluation_protocol=scheduled,
        scheduled_evaluation_protocol_sha256=json_sha256(scheduled),
        validation_likelihood_protocol=likelihood,
        validation_likelihood_protocol_sha256=json_sha256(likelihood),
        selection={"rule": "final_requested_epoch_only", "epoch": epochs,
                   "require_complete_epochs": True, "require_equal_optimizer_updates": True,
                   "reject_partial_or_wall_limited_runs": True, "test_used": False},
        candidate_parameters={"positive_anchor_coefficient": positive_anchor_coefficient,
            "claim": "explicit candidate; no efficacy established"},
        limitations=["Equal wall budgets do not guarantee equal completed optimizer updates.",
            "Frozen feature logical byte budget does not cap process RSS.",
            "No efficacy claim until complete paired runs and frozen validation comparison."])
    if initial_evaluation is not None:
        plan["initial_evaluation"] = initial_evaluation
    plan["wall_budget"] = {"seconds": max_wall_seconds,
        "includes": ["optimizer_training", "scheduled_validation"],
        "excludes": ["data_preparation", "initial_validation", "final_validation"]}
    plan["resources"] = {"max_process_rss_bytes": 12 * 1024**3,
        "minimum_free_disk_bytes": 8 * 1024**3,
        "enforcement": "external process guard required at execution",
        "model_loaded_during_preparation": False, "optimizer_updates": 0,
        "feature_logical_bytes": features["logical_bytes"]}
    reuse.assert_unchanged()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "shared_base_config.json", config.to_dict())
    plan["config_file"] = {"path": common["config"], **_file_pin(common["config"])}
    atomic_json(output / "manifest.json", plan)
    return plan


def load_retention_plan(path, expected_sha256=None):
    """Load a prepared plan and recheck its pinned files without loading tensors/RGB."""
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    pin = _file_pin(path)
    if expected_sha256 is not None and pin["sha256"] != expected_sha256:
        raise ValueError("retention manifest SHA256 mismatch")
    plan = json.loads(path.read_text())
    if (not isinstance(plan, dict) or plan.get("format") != FORMAT
            or plan.get("status") != "prepared_not_executed"
            or any(plan.get(field) is not False for field in
                   ("model_loaded", "optimizer_allocated", "training_executed",
                    "checkpoint_created"))
            or plan.get("updates") != 0):
        raise ValueError("unsupported or executed retention plan")
    try:
        revision = plan.get("preparation_revision", 1)
        if type(revision) is not int or revision not in (1, 2):
            raise ValueError("unsupported retention preparation revision")
        arms = plan["arms"]
        if set(arms) != {"anchor_off", "anchor_on"}:
            raise ValueError("retention plan requires two arms")
        off, on = (arms[name]["settings"] for name in ("anchor_off", "anchor_on"))
        differences = {key for key in off.keys() | on.keys() if off.get(key) != on.get(key)}
        if (differences != {"canonical_anchor_coefficient", "output"}
                or off["canonical_anchor_coefficient"] != 0
                or on["canonical_anchor_coefficient"] <= 0):
            raise ValueError("retention arm settings differ outside the controlled coefficient")
        optional_settings = {}
        if "eval_all_levels" in off:
            if type(off["eval_all_levels"]) is not bool or off["eval_all_levels"] is not True:
                raise ValueError("retention explicit eval_all_levels must be true")
            optional_settings["eval_all_levels"] = True
        if off.get("reuse_initial_evaluation") is not None:
            optional_settings["reuse_initial_evaluation"] = off["reuse_initial_evaluation"]
        for name, arm in arms.items():
            if (arm["output"] != arm["settings"]["output"]
                    or arm["output"] != str(path.resolve().parent / name)
                    or arm["argv"] != _argv(arm["settings"])
                    or arm["settings"] != _settings(
                        {key: off[key] for key in _COMMAND_FIELDS
                         if key not in {"output", "canonical_anchor_coefficient"}}
                        | {"validation_nll": True, "final_all_levels": True} | optional_settings,
                        coefficient=arm["settings"]["canonical_anchor_coefficient"],
                        output=arm["output"])):
                raise ValueError("retention argv/settings mismatch")
        config = SequenceRunConfig.from_dict(plan["effective_config"])
        config_file = plan["config_file"]
        if (SequenceRunConfig.load(config_file["path"]).to_dict() != config.to_dict()
                or off["config"] != config_file["path"]):
            raise ValueError("shared effective config differs")
        checkpoint = plan["source_checkpoint"]
        if _checkpoint(checkpoint["path"], checkpoint["sha256"]) != checkpoint:
            raise ValueError("pinned checkpoint changed")
        for record in (plan["source_config"], config_file):
            if _file_pin(record["path"]) != {key: record[key] for key in ("bytes", "sha256")}:
                raise ValueError("pinned configuration changed")
        initial = plan.get("initial_evaluation")
        if initial is None:
            if off.get("reuse_initial_evaluation") is not None:
                raise ValueError("retention reused initial evaluation has no pin")
        else:
            if (off.get("reuse_initial_evaluation") != initial["path"]
                    or _file_pin(initial["path"]) != {key: initial[key]
                                                     for key in ("bytes", "sha256")}):
                raise ValueError("retention pinned initial evaluation changed")
            measured = validate_initial_evaluation(Path(initial["path"]),
                checkpoint=Path(checkpoint["path"]),
                config=replace(config, eval_all_levels=True),
                **{key: checkpoint[key] for key in
                   ("agent_sha256", "reward_sha256", "weights_revision")})
            if any(initial[key] != checkpoint[key] for key in
                   ("agent_sha256", "reward_sha256", "weights_revision")) or (
                    initial["protocol_sha256"] != measured["protocol_sha256"]
                    or initial["records"] != len(measured["records"])):
                raise ValueError("retention initial evaluation identity differs")
        corpus = plan["source_corpus"]
        for name, identity in corpus["files"].items():
            if _file_pin(Path(corpus["source_run"]) / name) != identity:
                raise ValueError("pinned source corpus changed")
        # Old v1 plans pinned only leaf feature packs; new plans cover the whole graph.
        if revision >= 2:
            dependency_pins = _feature_dependency_pins(Path(corpus["source_run"]))
            if any(plan["frozen_features"]["files"].get(filename) != identity
                   for filename, identity in dependency_pins.items()):
                raise ValueError("retention frozen feature dependency graph differs")
        for filename, identity in plan["frozen_features"]["files"].items():
            if _file_pin(filename) != identity:
                raise ValueError("pinned frozen feature pack changed")
        for key in ("evaluation_protocol", "validation_likelihood_protocol"):
            if json_sha256(plan[key]) != plan[key + "_sha256"]:
                raise ValueError("retention evaluation protocol hash mismatch")
        if (off["warm_start"] != checkpoint["path"]
                or off["demonstration_reuse_run"] != corpus["source_run"]
                or off["feature_reuse_run"] != corpus["source_run"]
                or off["training_backend"] != config.training_backend
                or off["learning_rate"] != config.learning_rate
                or off["level_bank_seed"] != config.level_bank_seed
                or off["evaluation_difficulty"] != config.evaluation_difficulty):
            raise ValueError("retention settings differ from pinned source/config")
        if (plan["evaluation_protocol"] != json.loads(json.dumps(
                sequence_evaluation_protocol(replace(config, eval_all_levels=True))))
                or plan["validation_likelihood_protocol"] != json.loads(json.dumps(
                    imitation_validation_protocol(replace(config, eval_all_levels=False,
                        eval_episodes_per_task=off["validation_nll_episodes_per_task"]))))):
            raise ValueError("retention protocol differs from effective configuration")
        _positive(off["epochs"], "epochs", integer=True)
        _positive(off["minibatch_episodes"], "batch size", integer=True)
        _positive(on["canonical_anchor_coefficient"], "positive anchor coefficient")
        rng = random.Random(off["dataset_seed"])
        orders = []
        for _ in range(off["epochs"]):
            indices = list(range(corpus["episodes"]))
            rng.shuffle(indices)
            orders.append(json_sha256(indices))
        if plan["epoch_order_sha256"] != orders:
            raise ValueError("retention epoch order differs from dataset seed/cohort")
        if plan["selection"] != {
                "rule": "final_requested_epoch_only", "epoch": off["epochs"],
                "require_complete_epochs": True, "require_equal_optimizer_updates": True,
                "reject_partial_or_wall_limited_runs": True, "test_used": False}:
            raise ValueError("retention final selection rule changed")
        if revision >= 2 and (plan["scheduled_evaluation_protocol"] != json.loads(json.dumps(
                sequence_evaluation_protocol(replace(config, eval_all_levels=
                    off.get("eval_all_levels", False)))))
                or plan["scheduled_evaluation_protocol_sha256"]
                != json_sha256(plan["scheduled_evaluation_protocol"])):
            raise ValueError("retention scheduled evaluation protocol differs")
        if revision >= 2 and plan["wall_budget"] != {
                "seconds": off["max_wall_seconds"],
                "includes": ["optimizer_training", "scheduled_validation"],
                "excludes": ["data_preparation", "initial_validation", "final_validation"]}:
            raise ValueError("retention wall budget scope differs")
        if revision >= 2 and plan["resources"] != {
                "max_process_rss_bytes": 12 * 1024**3,
                "minimum_free_disk_bytes": 8 * 1024**3,
                "enforcement": "external process guard required at execution",
                "model_loaded_during_preparation": False, "optimizer_updates": 0,
                "feature_logical_bytes": plan["frozen_features"]["logical_bytes"]}:
            raise ValueError("retention resource limits differ")
        expected = off["epochs"] * math.ceil(corpus["episodes"] / off["minibatch_episodes"])
        if (plan["expected_epochs"] != off["epochs"]
                or plan["expected_optimizer_updates"] != expected
                or corpus["episode_order_sha256"] != json_sha256(corpus["ordered_episode_sha256"])):
            raise ValueError("retention budget or corpus order mismatch")
    except (KeyError, TypeError, ZeroDivisionError) as error:
        raise ValueError("retention plan is incomplete or malformed") from error
    return plan
