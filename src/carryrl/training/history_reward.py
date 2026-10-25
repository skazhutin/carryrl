"""Future, bounded fitting of the auxiliary causal history readout only.

No encoder or actor is loaded or trained here. A completed fit remains an
auxiliary scene readout and cannot authorize PPO or establish policy success.
"""

import hashlib
import io
import json
import math
import os
import random
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from carryrl.agent.actions import ACTION_FEATURE_SIZE
from carryrl.evaluation.scene_suite import SCENE_SUITE_VERSION
from carryrl.rewards.history import MASK_VERSION, MAX_HISTORY_STEPS, SceneHistoryHead

FITTED_HISTORY_REWARD_VERSION = "carryrl.fitted_scene_history_auxiliary.v1"
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
MAX_PARAMETERS = 5_000_000
_TARGET_KEYS = (
    "potential", "potential_valid", "events", "event_valid", "terminal_success",
    "terminal_success_valid",
)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _corpus_identity(value):
    if isinstance(value, str):
        value = {"corpus_sha256": value}
    if not isinstance(value, dict) or not _valid_sha(value.get("corpus_sha256")):
        raise ValueError("corpus identity requires a pinned corpus_sha256")
    return json.loads(_json_bytes(value))


def _feature_identity(value, corpus_identity):
    if (not isinstance(value, dict) or value.get("mask_version") != MASK_VERSION
            or value.get("version") != "carryrl.scene_history_features.v1"
            or value.get("annotation_version") != SCENE_SUITE_VERSION
            or value.get("pooling") != "mean_patch.v1"
            or value.get("corpus_sha256") != corpus_identity["corpus_sha256"]
            or any(not _valid_sha(value.get(name)) for name in (
                "vision_sha256", "goal_encoder_sha256", "frames_sha256", "goals_sha256",
                "goal_texts_sha256",
            ))):
        raise ValueError("feature identity must bind this corpus and full-HUD mask version")
    # JSON round-tripping detaches the caller's mutable identity.
    return json.loads(_json_bytes(value))


def _bind_features(config, identity):
    if (identity.get("feature_width") != config["scene_feature_size"]
            or identity.get("goal_width") != config["goal_feature_size"]
            or identity.get("event_names") != list(config["event_names"])):
        raise ValueError("feature dimensions/exact event vocabulary differ from the head")


def _config(head):
    return {
        "scene_feature_size": head.scene_feature_size,
        "goal_feature_size": head.goal_feature_size,
        "hidden_size": head.history.hidden_size,
        "max_steps": head.max_steps,
        "event_names": list(head.event_names),
    }


def _validate_config(value):
    if not isinstance(value, dict) or set(value) != {
        "scene_feature_size", "goal_feature_size", "hidden_size", "max_steps", "event_names",
    }:
        raise ValueError("invalid strict history head configuration")
    for key in ("scene_feature_size", "goal_feature_size", "hidden_size", "max_steps"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError("invalid history head dimensions")
    if (value["max_steps"] > MAX_HISTORY_STEPS or value["hidden_size"] > 1024
            or max(value["scene_feature_size"], value["goal_feature_size"]) > 8192):
        raise ValueError("history head exceeds resource limits")
    names = value["event_names"]
    if (not isinstance(names, list) or not 1 <= len(names) <= 64
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("invalid exact event vocabulary")
    hidden, events = value["hidden_size"], len(names)
    parameters = (hidden * (value["scene_feature_size"] + value["goal_feature_size"]
                           + ACTION_FEATURE_SIZE) + 12 * hidden * hidden + 9 * hidden
                  + (hidden + 1) * (2 * events + 4))
    if parameters > MAX_PARAMETERS:
        raise ValueError("history head exceeds parameter budget")
    return {**value, "event_names": tuple(names)}


def _tensor_sha(state):
    digest = hashlib.sha256()
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint must contain head tensors")
    for name in sorted(state):
        value = state[name]
        if (not isinstance(name, str) or not isinstance(value, torch.Tensor)
                or value.layout != torch.strided or not value.is_floating_point()
                or not torch.isfinite(value).all()):
            raise ValueError("checkpoint head tensors must be finite floating point")
        value = value.detach().cpu().contiguous()
        digest.update(_json_bytes([name, str(value.dtype), list(value.shape)]))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _fit_report(value):
    if not isinstance(value, dict):
        raise ValueError("missing actual fit report")
    required = ("optimizer_updates", "planned_optimizer_updates", "epochs_completed", "epochs",
                "train_episodes", "validation_episodes")
    if (any(type(value.get(key)) is not int or value[key] < 1 for key in required)
            or value.get("status") != "completed" or value.get("ppo_eligible") is not False
            or value.get("test_used") is not False
            or value["optimizer_updates"] != value["planned_optimizer_updates"]
            or value["epochs_completed"] != value["epochs"]
            or value["optimizer_updates"] != value["epochs"] * value["train_episodes"]):
        raise ValueError("checkpoint requires a completed, nonzero TRAIN-only fit report")
    return json.loads(_json_bytes(value))


def _atomic_json(path, value):
    data = _json_bytes(value)
    descriptor, temporary = tempfile.mkstemp(prefix=".history-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_fitted_history_reward_checkpoint(
    path, head, *, corpus_identity, feature_identity, fit_report,
):
    """Write a new fitted-head archive and required archive-integrity sidecar.

    Optimizer state is deliberately absent: resuming a fit is unsupported. The
    sidecar records archive integrity; loaders require an external pinned SHA.
    """
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if path == sidecar or path.exists() or sidecar.exists():
        raise FileExistsError("checkpoint and integrity sidecar must be new paths")
    if not isinstance(head, SceneHistoryHead):
        raise ValueError("only SceneHistoryHead can be persisted")
    config = _config(head)
    _validate_config(config)
    corpus_identity = _corpus_identity(corpus_identity)
    feature_identity = _feature_identity(feature_identity, corpus_identity)
    _bind_features(config, feature_identity)
    report = _fit_report(fit_report)
    state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    metadata = {
        "version": FITTED_HISTORY_REWARD_VERSION, "mask_version": MASK_VERSION,
        "head_config": config, "corpus_identity": corpus_identity,
        "feature_identity": feature_identity, "feature_identity_sha256": _sha(feature_identity),
        "corpus_identity_sha256": _sha(corpus_identity), "fit_report": report,
        "ppo_eligible": False, "resume_supported": False, "tensor_sha256": _tensor_sha(state),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".history-", dir=path.parent)
    os.close(descriptor)
    linked = False
    try:
        with open(temporary, "wb") as stream:
            torch.save({"metadata": metadata, "state_dict": state}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        archive = Path(temporary).read_bytes()
        if len(archive) > MAX_CHECKPOINT_BYTES:
            raise ValueError("history checkpoint exceeds byte budget")
        archive_sha = hashlib.sha256(archive).hexdigest()
        # Linking a private completed file is atomic and cannot replace a source artifact.
        os.link(temporary, path)
        linked = True
        with sidecar.open("x", encoding="utf-8") as stream:
            json.dump({"version": FITTED_HISTORY_REWARD_VERSION,
                       "checkpoint_sha256": archive_sha, "bytes": len(archive)}, stream)
        return {"checkpoint_sha256": archive_sha, "path": str(path), "metadata": metadata}
    except BaseException:
        if linked:
            path.unlink(missing_ok=True)
        raise
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_fitted_history_reward_checkpoint(
    path, *, expected_sha256, corpus_identity=None, feature_identity=None,
    event_names=None, device="cpu",
):
    """Load a strictly validated fitted readout, always frozen and PPO-ineligible."""
    path = Path(path)
    if path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("history checkpoint exceeds byte budget")
    archive = path.read_bytes()
    archive_sha = hashlib.sha256(archive).hexdigest()
    if not _valid_sha(expected_sha256) or archive_sha != expected_sha256:
        raise ValueError("checkpoint archive SHA mismatch")
    # Deserialize the bytes just hashed rather than reopening a mutable path.
    payload = torch.load(io.BytesIO(archive), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"metadata", "state_dict"}:
        raise ValueError("invalid history checkpoint payload")
    metadata = payload["metadata"]
    expected_keys = {
        "version", "mask_version", "head_config", "corpus_identity", "feature_identity",
        "feature_identity_sha256", "corpus_identity_sha256", "fit_report", "ppo_eligible",
        "resume_supported", "tensor_sha256",
    }
    if (not isinstance(metadata, dict) or set(metadata) != expected_keys
            or metadata.get("version") != FITTED_HISTORY_REWARD_VERSION
            or metadata.get("mask_version") != MASK_VERSION
            or metadata.get("ppo_eligible") is not False
            or metadata.get("resume_supported") is not False):
        raise ValueError("invalid fitted history checkpoint contract")
    config = _validate_config(metadata["head_config"])
    stored_corpus = _corpus_identity(metadata["corpus_identity"])
    stored_features = _feature_identity(metadata["feature_identity"], stored_corpus)
    _bind_features(config, stored_features)
    if (metadata["feature_identity_sha256"] != _sha(stored_features)
            or metadata["corpus_identity_sha256"] != _sha(stored_corpus)):
        raise ValueError("checkpoint feature/corpus identity SHA mismatch")
    if corpus_identity is not None and _corpus_identity(corpus_identity) != stored_corpus:
        raise ValueError("checkpoint corpus identity mismatch")
    if feature_identity is not None and feature_identity != stored_features:
        raise ValueError("checkpoint feature identity mismatch")
    if event_names is not None and tuple(event_names) != config["event_names"]:
        raise ValueError("checkpoint exact event vocabulary mismatch")
    _fit_report(metadata["fit_report"])
    if _tensor_sha(payload["state_dict"]) != metadata["tensor_sha256"]:
        raise ValueError("checkpoint tensor SHA mismatch")
    with torch.random.fork_rng(devices=[]):
        head = SceneHistoryHead(**config)
    if any(value.dtype != torch.float32 for value in payload["state_dict"].values()):
        raise ValueError("checkpoint head dtype must be float32")
    head.load_state_dict(payload["state_dict"], strict=True)
    head.to(device).eval().requires_grad_(False)
    head.ppo_eligible = False
    return head, {**metadata, "checkpoint_sha256": archive_sha}


def _training_indices(corpus):
    groups = {}
    indices = defaultdict(list)
    for index, episode in enumerate(corpus.episodes):
        split = episode.get("split")
        task, group = episode.get("task"), episode.get("runtime_group")
        if (split not in {"train", "validation"}
                or not isinstance(task, str) or not task
                or not isinstance(group, str) or not group):
            raise ValueError("episodes require task, split and pinned runtime_group")
        # Runtime identity already includes task identity; do not weaken it with metadata.
        if group in groups and groups[group] != split:
            raise ValueError("runtime groups overlap between corpus splits")
        groups[group] = split
        indices[split].append(index)
    if not indices["train"] or not indices["validation"]:
        raise ValueError("fit requires pre-separated TRAIN and validation episodes")
    return indices


def _inputs(features, corpus, index):
    source = features.model_inputs(corpus, index, device="cpu")
    if not isinstance(source, Mapping) or set(source) - {
        "scene_features", "executed_previous_actions", "explicit_goal_features", "lengths",
    }:
        raise ValueError("feature provider returned unauthorized model inputs")
    result = {}
    for name, value in source.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError("model inputs must be tensors")
        result[name] = value.detach().clone()
    if any(name not in result for name in (
        "scene_features", "executed_previous_actions", "explicit_goal_features",
    )) or result["scene_features"].shape[0] != 1:
        raise ValueError("fit expects exactly one complete episode per input")
    return result


def _targets(corpus, index):
    source = corpus.targets(index)
    if source.get("event_names") != corpus.event_names:
        raise ValueError("source target event vocabulary mismatch")
    result = {"event_names": corpus.event_names}
    for key in _TARGET_KEYS:
        array = np.array(source[key], copy=True)
        dtype = torch.bool if key.endswith("valid") else torch.float32
        if dtype == torch.bool and array.dtype != np.bool_:
            raise ValueError("target validity must be boolean")
        result[key] = torch.as_tensor(array, dtype=dtype).unsqueeze(0)
    return result


def fit_history_reward(
    corpus, features, output, *, epochs, learning_rate, seed, hidden_size=128,
    max_wall_seconds=60.0, max_updates=100_000, evaluator=None, device="cpu",
):
    """Fit only TRAIN head parameters; measure validation with a frozen head.

    Every complete episode receives one update per epoch. The head averages
    losses within each episode, so long episodes do not acquire extra weight.
    Task/group episode counts are reported; arbitrary unequal cohorts do not
    receive a claim of equal task/group weighting.
    The CPU-only implementation bounds model size and restores caller RNG/thread
    settings. Incomplete fits raise and leave an atomic failed status, never a
    loadable partial checkpoint. Test episodes are never evaluated here.
    """
    output = Path(output)
    resolved_output = output.resolve()
    for source in (getattr(corpus, "directory", None), getattr(features, "directory", None)):
        if source is not None:
            resolved_source = Path(source).resolve()
            if (resolved_output == resolved_source or resolved_source in resolved_output.parents
                    or resolved_output in resolved_source.parents):
                raise ValueError("fit output must be outside corpus and feature source trees")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    updates = 0
    epochs_completed = 0
    checkpoints = (output / "checkpoint.pt", output / "checkpoint.json")
    old_threads = torch.get_num_threads()
    old_random, old_numpy = random.getstate(), np.random.get_state()
    try:
        _atomic_json(output / "status.json", {"status": "running", "ppo_eligible": False,
                                              "optimizer_updates": 0})
        if (type(epochs) is not int or not 1 <= epochs <= 10_000
                or type(seed) is not int or not 0 <= seed < 2**32
                or type(max_updates) is not int or not 1 <= max_updates <= 100_000
                or not isinstance(learning_rate, (float, int)) or isinstance(learning_rate, bool)
                or not math.isfinite(learning_rate) or learning_rate <= 0
                or not isinstance(max_wall_seconds, (float, int))
                or isinstance(max_wall_seconds, bool) or not math.isfinite(max_wall_seconds)
                or max_wall_seconds <= 0 or str(device) != "cpu"):
            raise ValueError("invalid bounded CPU history-fit settings")
        indices = _training_indices(corpus)
        planned_updates = epochs * len(indices["train"])
        if planned_updates > max_updates:
            raise ValueError("planned optimizer updates exceed the explicit update budget")
        corpus_identity = _corpus_identity(corpus.corpus_sha256)
        feature_identity = _feature_identity(features.identity, corpus_identity)
        config = _validate_config({
            "scene_feature_size": features.feature_width,
            "goal_feature_size": features.goal_width, "hidden_size": hidden_size,
            "max_steps": MAX_HISTORY_STEPS, "event_names": list(corpus.event_names),
        })
        _bind_features(config, feature_identity)
        group_counts = Counter(corpus.episodes[i]["runtime_group"] for i in indices["train"])
        task_counts = Counter(corpus.episodes[i]["task"] for i in indices["train"])
        if evaluator is None:
            from carryrl.evaluation.history_reward import evaluate_history_reward

            evaluator = evaluate_history_reward

        def check_deadline():
            if time.monotonic() - started >= max_wall_seconds:
                raise TimeoutError("history fit exceeded max_wall_seconds before completion")

        torch.set_num_threads(1)
        with torch.inference_mode(False), torch.enable_grad(), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            head = SceneHistoryHead(**config)
            head.ppo_eligible = False
            optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
            rng = np.random.default_rng(seed)
            losses = []
            for epoch in range(epochs):
                head.train().requires_grad_(True)
                total = 0.0
                for index in rng.permutation(indices["train"]).tolist():
                    check_deadline()
                    inputs = _inputs(features, corpus, index)
                    target = _targets(corpus, index)
                    optimizer.zero_grad(set_to_none=True)
                    loss = head.supervised_loss(head(**inputs), target, inputs.get("lengths"))
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite history training loss")
                    loss.backward()
                    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                           for parameter in head.parameters()):
                        raise ValueError("nonfinite history gradients")
                    optimizer.step()
                    updates += 1
                    total += float(loss.detach())
                    _tensor_sha(head.state_dict())
                epochs_completed = epoch + 1
                losses.append(total / len(indices["train"]))
                _atomic_json(output / "status.json", {
                    "status": "running", "ppo_eligible": False, "optimizer_updates": updates,
                    "epochs_completed": epochs_completed,
                })
            optimizer.zero_grad(set_to_none=True)
            head.eval().requires_grad_(False)
            check_deadline()
            before_validation = _tensor_sha(head.state_dict())
            with torch.no_grad():
                validation = evaluator(head, corpus, features, split="validation")
            if (_tensor_sha(head.state_dict()) != before_validation
                    or any(parameter.requires_grad for parameter in head.parameters())
                    or any(module.training for module in head.modules())
                    or head.ppo_eligible is not False):
                raise ValueError("validation mutated the frozen history head")
            check_deadline()
            report = {
                "status": "completed", "ppo_eligible": False, "test_used": False,
                "optimizer_updates": updates, "planned_optimizer_updates": planned_updates,
                "epochs": epochs, "epochs_completed": epochs_completed,
                "train_episodes": len(indices["train"]),
                "validation_episodes": len(indices["validation"]),
                "train_runtime_groups": len(group_counts), "train_tasks": len(task_counts),
                "train_task_episode_counts": dict(task_counts),
                "train_runtime_group_episode_counts": dict(group_counts),
                "weighting": "equal_episode",
                "train_loss_by_epoch": losses, "validation": validation,
                "learning_rate": learning_rate, "seed": seed,
                "max_wall_seconds": max_wall_seconds, "max_updates": max_updates,
                "elapsed_seconds": time.monotonic() - started, "device": "cpu",
                "resume_supported": False,
            }
            result = save_fitted_history_reward_checkpoint(
                checkpoints[0], head, corpus_identity=corpus_identity,
                feature_identity=feature_identity, fit_report=report,
            )
            check_deadline()
            report["checkpoint_sha256"] = result["checkpoint_sha256"]
            _atomic_json(output / "fit_report.json", report)
            _atomic_json(output / "status.json", report)
            return report
    except BaseException as error:
        for checkpoint in checkpoints:
            checkpoint.unlink(missing_ok=True)
        _atomic_json(output / "status.json", {
            "status": "failed", "ppo_eligible": False, "test_used": False,
            "optimizer_updates": updates, "epochs_completed": epochs_completed,
            "error_type": type(error).__name__, "error": str(error),
        })
        raise
    finally:
        random.setstate(old_random)
        np.random.set_state(old_numpy)
        torch.set_num_threads(old_threads)
