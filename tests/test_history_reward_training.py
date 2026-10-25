"""Only tiny synthetic features; no corpus fitting, downloads or RGB rollouts."""

import copy
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from carryrl.agent.actions import ACTION_FEATURE_SIZE
from carryrl.evaluation.scene_suite import SCENE_SUITE_VERSION
from carryrl.rewards.history import MASK_VERSION, SceneHistoryHead
from carryrl.training.history_reward import (
    fit_history_reward,
    load_fitted_history_reward_checkpoint,
    save_fitted_history_reward_checkpoint,
)


class TinyCorpus:
    corpus_sha256 = "a" * 64
    event_names = ("collected", "exit_reached")

    def __init__(self):
        self.episodes = [
            {"task": "one", "split": "train", "runtime_group": "one/train",
             "strategy": "synthetic"},
            {"task": "two", "split": "train", "runtime_group": "two/train",
             "strategy": "synthetic"},
            {"task": "one", "split": "validation", "runtime_group": "one/validation",
             "strategy": "synthetic"},
        ]
        self.target_reads = []

    def targets(self, index):
        self.target_reads.append(index)
        assert index < 3, "Test targets must never be read"
        return {
            "potential": np.array([0.0, 1.0], np.float32),
            "potential_valid": np.ones(2, dtype=bool),
            "events": np.array([[0, 0], [1, 1]], np.float32),
            "event_valid": np.ones((2, 2), dtype=bool),
            "terminal_success": np.array([0, 1], np.float32),
            "terminal_success_valid": np.ones(2, dtype=bool),
            "event_names": self.event_names,
        }


class TinyFeatures:
    feature_width = 3
    goal_width = 2

    def __init__(self, corpus):
        self.identity = {
            "version": "carryrl.scene_history_features.v1",
            "mask_version": MASK_VERSION, "corpus_sha256": corpus.corpus_sha256,
            "annotation_version": SCENE_SUITE_VERSION, "pooling": "mean_patch.v1",
            "vision_sha256": "b" * 64, "goal_encoder_sha256": "c" * 64,
            "frames_sha256": "d" * 64, "goals_sha256": "e" * 64,
            "goal_texts_sha256": "f" * 64, "feature_width": 3, "goal_width": 2,
            "event_names": list(corpus.event_names),
        }
        self.scenes = np.arange(18, dtype=np.float32).reshape(3, 2, 3) / 20
        self.goals = np.ones((3, 2), dtype=np.float32)
        self.reads = []

    def model_inputs(self, corpus, index, device="cpu"):
        self.reads.append(index)
        assert index < 3, "Test features must never be read"
        return {
            "scene_features": torch.as_tensor(self.scenes[index:index + 1], device=device),
            "executed_previous_actions": torch.zeros((1, 2, ACTION_FEATURE_SIZE), device=device),
            "explicit_goal_features": torch.as_tensor(self.goals[index:index + 1], device=device),
        }


def _validation(head, corpus, features, *, split):
    assert split == "validation" and not torch.is_grad_enabled()
    assert not head.training and not any(p.requires_grad for p in head.parameters())
    assert all(p.grad is None for p in head.parameters())
    prediction = head(**features.model_inputs(corpus, 2))
    assert not prediction["potential"].requires_grad
    corpus.targets(2)
    return {"split": "validation", "episodes": 1, "synthetic": True}


def _fit(directory, *, corpus=None, features=None, **kwargs):
    corpus = corpus or TinyCorpus()
    features = features or TinyFeatures(corpus)
    report = fit_history_reward(
        corpus, features, directory, epochs=1, learning_rate=0.005, seed=7,
        hidden_size=4, evaluator=kwargs.pop("evaluator", _validation),
        max_wall_seconds=kwargs.pop("max_wall_seconds", 30), **kwargs,
    )
    return corpus, features, report


def _load(directory, report, **kwargs):
    return load_fitted_history_reward_checkpoint(
        directory / "checkpoint.pt", expected_sha256=report["checkpoint_sha256"], **kwargs,
    )


def test_two_actual_train_updates_frozen_validation_arrays_and_caller_state(tmp_path):
    corpus = TinyCorpus()
    features = TinyFeatures(corpus)
    scenes, goals = features.scenes.copy(), features.goals.copy()
    identity = copy.deepcopy(features.identity)
    torch_state = torch.random.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    threads = torch.get_num_threads()
    _, _, report = _fit(tmp_path / "fit", corpus=corpus, features=features)
    assert report["optimizer_updates"] == report["planned_optimizer_updates"] == 2
    assert report["epochs_completed"] == 1
    assert report["train_episodes"] == 2 and report["validation_episodes"] == 1
    assert report["ppo_eligible"] is False and report["test_used"] is False
    assert sorted(corpus.target_reads[:2]) == [0, 1] and corpus.target_reads[2:] == [2]
    assert sorted(features.reads[:2]) == [0, 1] and features.reads[2:] == [2]
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert python_state == random.getstate()
    current_numpy = np.random.get_state()
    assert numpy_state[0] == current_numpy[0] and numpy_state[2:] == current_numpy[2:]
    np.testing.assert_array_equal(numpy_state[1], current_numpy[1])
    assert torch.get_num_threads() == threads
    np.testing.assert_array_equal(features.scenes, scenes)
    np.testing.assert_array_equal(features.goals, goals)
    assert features.identity == identity
    head, metadata = _load(
        tmp_path / "fit", report, corpus_identity={"corpus_sha256": corpus.corpus_sha256},
        feature_identity=features.identity, event_names=corpus.event_names,
    )
    assert head.event_names == corpus.event_names and head.ppo_eligible is False
    assert not head.training and all(not p.requires_grad for p in head.parameters())
    assert metadata["resume_supported"] is False and "optimizer" not in metadata
    # Independently reproduce the two specified AdamW updates to prove that the
    # final snapshot contains the actual trained weights, with no validation step.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        expected = SceneHistoryHead(3, 2, hidden_size=4, event_names=corpus.event_names)
    initial = {name: value.clone() for name, value in expected.state_dict().items()}
    optimizer = torch.optim.AdamW(expected.parameters(), lr=0.005)
    for index in np.random.default_rng(7).permutation([0, 1]).tolist():
        inputs = features.model_inputs(corpus, index)
        raw = corpus.targets(index)
        targets = {key: torch.as_tensor(value).unsqueeze(0)
                   for key, value in raw.items() if key != "event_names"}
        targets["event_names"] = corpus.event_names
        optimizer.zero_grad(set_to_none=True)
        expected.supervised_loss(expected(**inputs), targets).backward()
        optimizer.step()
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(head.state_dict()[name], value)
    assert any(not torch.equal(value, initial[name]) for name, value in head.state_dict().items())
    assert json.loads((tmp_path / "fit/status.json").read_text())["status"] == "completed"


@pytest.fixture
def fitted(tmp_path):
    _, features, report = _fit(tmp_path / "source")
    return tmp_path / "source", features, report


def test_checkpoint_requires_external_sha_and_exact_bindings(fitted):
    directory, features, report = fitted
    with pytest.raises(TypeError, match="expected_sha256"):
        load_fitted_history_reward_checkpoint(directory / "checkpoint.pt")
    with pytest.raises(ValueError, match="archive SHA"):
        load_fitted_history_reward_checkpoint(directory / "checkpoint.pt", expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="corpus identity"):
        _load(directory, report, corpus_identity="0" * 64)
    with pytest.raises(ValueError, match="feature identity"):
        _load(directory, report, feature_identity={**features.identity, "pooling": "changed"})
    with pytest.raises(ValueError, match="event vocabulary"):
        _load(directory, report, event_names=("exit_reached", "collected"))


def test_tiny_final_snapshot_integrates_real_frozen_evaluator(tmp_path):
    _, _, report = _fit(tmp_path / "integrated", evaluator=None)
    validation = report["validation"]
    assert validation["split"] == "validation"
    assert validation["updates"] == 0 and validation["ppo_eligible"] is False
    assert validation["head_sha256_before"] == validation["head_sha256_after"]
    assert {row["episode_index"] for row in validation["cohort"]} == {2}
    assert set(validation["controls"]) == {
        "full", "no_scene", "no_actions", "constant_goal", "single_frame", "time_only",
    }
    _, metadata = _load(tmp_path / "integrated", report)
    assert metadata["fit_report"]["validation"] == validation


def test_fit_restores_callers_disabled_gradient_mode(tmp_path):
    with torch.no_grad():
        _fit(tmp_path / "caller_no_grad")
        assert not torch.is_grad_enabled()


def test_test_split_is_rejected_without_reading_its_features_or_targets(tmp_path):
    corpus = TinyCorpus()
    corpus.episodes.append({"task": "one", "split": "test", "runtime_group": "one/test"})
    features = TinyFeatures(corpus)
    with pytest.raises(ValueError, match="episodes require"):
        _fit(tmp_path / "test_rejected", corpus=corpus, features=features)
    assert corpus.target_reads == [] and features.reads == []


def test_validation_mutation_cannot_be_persisted_as_success(tmp_path):
    def mutate(head, *args, **kwargs):
        next(head.parameters()).add_(1)
        return {"synthetic": True}

    with pytest.raises(ValueError, match="validation mutated"):
        _fit(tmp_path / "mutated", evaluator=mutate)
    status = json.loads((tmp_path / "mutated/status.json").read_text())
    assert status["status"] == "failed" and status["optimizer_updates"] == 2
    assert not (tmp_path / "mutated/checkpoint.pt").exists()


@pytest.mark.parametrize("tamper", (
    "version", "mask", "feature_identity", "feature_sha", "corpus_identity", "vocabulary",
    "config_fields", "width", "zero_updates", "partial_updates", "ppo", "missing_report",
    "weights", "nonfinite", "missing_tensor", "dtype", "annotation_version", "pooling",
))
def test_checkpoint_rejects_tampered_contract_even_with_recomputed_archive_sha(
    fitted, tmp_path, tamper,
):
    directory, _, _ = fitted
    payload = torch.load(directory / "checkpoint.pt", weights_only=True)
    metadata, state = payload["metadata"], payload["state_dict"]
    if tamper == "version":
        metadata["version"] = "changed"
    elif tamper == "mask":
        metadata["mask_version"] = "changed"
    elif tamper == "feature_identity":
        metadata["feature_identity"]["vision_sha256"] = "0" * 64
    elif tamper == "feature_sha":
        metadata["feature_identity_sha256"] = "0" * 64
    elif tamper in {"annotation_version", "pooling"}:
        metadata["feature_identity"][tamper] = "incompatible.v1"
    elif tamper == "corpus_identity":
        metadata["corpus_identity"]["corpus_sha256"] = "0" * 64
    elif tamper == "vocabulary":
        metadata["head_config"]["event_names"].reverse()
    elif tamper == "config_fields":
        metadata["head_config"]["extra"] = 1
    elif tamper == "width":
        metadata["head_config"]["scene_feature_size"] += 1
    elif tamper == "zero_updates":
        metadata["fit_report"]["optimizer_updates"] = 0
    elif tamper == "partial_updates":
        metadata["fit_report"]["optimizer_updates"] = 1
    elif tamper == "ppo":
        metadata["ppo_eligible"] = True
    elif tamper == "missing_report":
        del metadata["fit_report"]
    elif tamper == "weights":
        state["scene_projection.weight"][0, 0] += 1
    elif tamper == "nonfinite":
        state["scene_projection.weight"][0, 0] = float("nan")
    elif tamper == "missing_tensor":
        del state["scene_projection.weight"]
    elif tamper == "dtype":
        state["scene_projection.weight"] = state["scene_projection.weight"].double()
    destination = tmp_path / "tampered.pt"
    torch.save(payload, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    with pytest.raises((ValueError, RuntimeError)):
        load_fitted_history_reward_checkpoint(destination, expected_sha256=digest)


@pytest.mark.parametrize("problem", ("overlap", "identity", "event_names", "timeout", "evaluation"))
def test_failures_leave_atomic_failed_status_no_loadable_partial_fit(tmp_path, problem):
    corpus = TinyCorpus()
    features = TinyFeatures(corpus)
    kwargs = {}
    if problem == "overlap":
        corpus.episodes[2]["runtime_group"] = corpus.episodes[0]["runtime_group"]
    elif problem == "identity":
        features.identity["corpus_sha256"] = "0" * 64
    elif problem == "event_names":
        features.identity["event_names"].reverse()
    elif problem == "timeout":
        kwargs["max_wall_seconds"] = 1e-12
    else:
        def fail(*args, **kwargs):
            raise RuntimeError("synthetic validation failed")
        kwargs["evaluator"] = fail
    state = torch.random.get_rng_state().clone()
    with pytest.raises((ValueError, TimeoutError, RuntimeError)):
        _fit(tmp_path / "failed", corpus=corpus, features=features, **kwargs)
    status = json.loads((tmp_path / "failed/status.json").read_text())
    assert status["status"] == "failed" and status["ppo_eligible"] is False
    assert status["optimizer_updates"] == (2 if problem == "evaluation" else 0)
    assert not (tmp_path / "failed/checkpoint.pt").exists()
    assert not (tmp_path / "failed/checkpoint.json").exists()
    assert torch.equal(state, torch.random.get_rng_state())


def test_no_overwrites_and_untrained_head_cannot_be_marked_fitted(fitted):
    directory, features, report = fitted
    before = (directory / "checkpoint.pt").read_bytes()
    with pytest.raises(FileExistsError):
        _fit(directory)
    head, _ = _load(directory, report)
    with pytest.raises(FileExistsError):
        save_fitted_history_reward_checkpoint(
            directory / "checkpoint.pt", head, corpus_identity="a" * 64,
            feature_identity=features.identity, fit_report=report,
        )
    with pytest.raises(ValueError, match="nonzero TRAIN-only"):
        save_fitted_history_reward_checkpoint(
            directory / "untrained.pt", head, corpus_identity="a" * 64,
            feature_identity=features.identity, fit_report={**report, "optimizer_updates": 0},
        )
    assert (directory / "checkpoint.pt").read_bytes() == before
    assert not hasattr(head, "mark_fitted")


@pytest.mark.parametrize("source_name", ("corpus", "features"))
@pytest.mark.parametrize("relationship", ("same", "child", "parent", "symlink"))
def test_fit_output_cannot_mutate_source_trees(tmp_path, source_name, relationship):
    corpus = TinyCorpus()
    features = TinyFeatures(corpus)
    source = tmp_path / "protected" / "source"
    source.mkdir(parents=True)
    (source / "sentinel").write_text("preserve source")
    owner = corpus if source_name == "corpus" else features
    owner.directory = source
    if relationship == "same":
        output = source
    elif relationship == "child":
        output = source / "new-fit"
    elif relationship == "parent":
        output = source.parent
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        output = alias / "new-fit"
    with pytest.raises(ValueError, match="outside corpus and feature source trees"):
        _fit(output, corpus=corpus, features=features)
    assert (source / "sentinel").read_text() == "preserve source"
    assert sorted(path.name for path in source.iterdir()) == ["sentinel"]
