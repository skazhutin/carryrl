"""Pinned evaluation CLI wiring, failure recording and source path protection."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carryrl.evaluation.history_reward import DEFAULT_CONTROLS
from carryrl.rewards.history import SceneHistoryHead

spec = importlib.util.spec_from_file_location(
    "evaluate_history_reward_script",
    Path(__file__).parents[1] / "scripts/evaluate_history_reward.py",
)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    paths = {name: tmp_path / name for name in ("corpus", "features", "fitted_source")}
    for directory in paths.values():
        directory.mkdir()
        (directory / "source.txt").write_text("unchanged source fixture")
    checkpoint = paths["fitted_source"] / "fixture.pt"
    checkpoint.write_bytes(b"test fixture: no training and no real fitted candidate")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    corpus_sha, feature_sha = "a" * 64, "b" * 64
    labels = {
        "event_names": ("pickup",),
        "potential": np.asarray([0.0, 1.0], np.float32),
        "potential_valid": np.ones(2, bool),
        "events": np.asarray([[0.0], [1.0]], np.float32),
        "event_valid": np.ones((2, 1), bool),
        "terminal_success": np.asarray([0.0, 1.0], np.float32),
        "terminal_success_valid": np.ones(2, bool),
    }
    corpus = SimpleNamespace(
        corpus_sha256=corpus_sha,
        event_names=("pickup",),
        episodes=[
            {
                "split": "train",
                "task": "navigation",
                "strategy": "teacher",
                "runtime_group": "train-level",
            },
            {
                "split": "validation",
                "task": "navigation",
                "strategy": "random",
                "runtime_group": "validation-level",
            },
        ],
        targets=lambda index: labels,
    )
    identity = {
        "corpus_sha256": corpus_sha,
        "event_names": ["pickup"],
        "fixture_identity": "tiny frozen tensors",
    }

    def inputs(actual_corpus, index, *, device="cpu"):
        assert actual_corpus is corpus and str(device) == "cpu"
        return {
            "scene_features": torch.ones(1, 2, 2),
            "executed_previous_actions": torch.zeros(1, 2, 33),
            "explicit_goal_features": torch.ones(1, 3),
        }

    features = SimpleNamespace(
        identity=identity, feature_width=2, goal_width=3, model_inputs=inputs
    )
    head = (
        SceneHistoryHead(
            scene_feature_size=2, goal_feature_size=3, hidden_size=4, event_names=corpus.event_names
        )
        .eval()
        .requires_grad_(False)
    )
    calls = []

    def load_corpus(path, *, expected_sha256):
        assert path == paths["corpus"] and expected_sha256 == corpus_sha
        calls.append("corpus")
        return corpus

    def load_features(path, actual_corpus, *, expected_sha256):
        assert path == paths["features"] and actual_corpus is corpus
        assert expected_sha256 == feature_sha
        calls.append("features")
        return features

    def load_checkpoint(path, **kwargs):
        assert path == checkpoint
        assert kwargs == {
            "expected_sha256": checkpoint_sha,
            "corpus_identity": {"corpus_sha256": corpus_sha},
            "feature_identity": identity,
            "event_names": corpus.event_names,
            "device": "cpu",
        }
        pending_status = json.loads((output / "status.json").read_text())
        pending_metadata = json.loads((output / "metadata.json").read_text())
        assert "checkpoint_fitted" not in pending_status
        assert "checkpoint_fitted" not in pending_metadata
        calls.append("checkpoint")
        return head, {"checkpoint_sha256": checkpoint_sha, "fit_report": {"fixture": True}}

    monkeypatch.setattr(script, "load_history_suite", load_corpus)
    monkeypatch.setattr(script, "load_history_features", load_features)
    monkeypatch.setattr(script, "load_fitted_history_reward_checkpoint", load_checkpoint)
    monkeypatch.setattr(script, "provenance", lambda: {"fixture_provenance": "recorded"})
    output = tmp_path / "evaluation"
    argv = [
        "--corpus",
        str(paths["corpus"]),
        "--corpus-sha256",
        corpus_sha,
        "--features",
        str(paths["features"]),
        "--features-sha256",
        feature_sha,
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha256",
        checkpoint_sha,
        "--output",
        str(output),
    ]
    return SimpleNamespace(
        paths=paths,
        checkpoint=checkpoint,
        checkpoint_sha=checkpoint_sha,
        argv=argv,
        output=output,
        corpus=corpus,
        features=features,
        calls=calls,
        head=head,
    )


@pytest.mark.parametrize("split", ["validation", "train"])
def test_cli_runs_all_controls_frozen_with_pins_and_restores_threads(tiny, split):
    before = {
        path: path.read_bytes() for directory in tiny.paths.values() for path in directory.iterdir()
    }
    threads = torch.get_num_threads()
    args = tiny.argv + (["--split", "train"] if split == "train" else [])
    assert script.main(args) == 0
    assert torch.get_num_threads() == threads
    assert tiny.calls == ["corpus", "features", "checkpoint"]
    assert all(path.read_bytes() == value for path, value in before.items())
    report = json.loads((tiny.output / "evaluation.json").read_text())
    assert tuple(report["controls"]) == DEFAULT_CONTROLS
    assert report["split"] == split
    assert report["checkpoint_sha256"] == tiny.checkpoint_sha
    assert report["head_sha256_before"] == report["head_sha256_after"]
    assert report["checkpoint_fitted"] is True and "model_fit" not in report
    assert report["updates"] == 0 and not report["fit_performed"] and not report["optimizer_loaded"]
    metadata = json.loads((tiny.output / "metadata.json").read_text())
    assert metadata["fixture_provenance"] == "recorded" and metadata["torch_threads"] == 2
    assert metadata["source_pins"]["checkpoint"]["sha256"] == tiny.checkpoint_sha
    assert metadata["feature_identity"] == tiny.features.identity
    assert metadata["checkpoint_fitted"] is True and not metadata["fit_performed"]
    assert "model_fit" not in metadata
    assert metadata["checkpoint_metadata"]["fit_report"] == {"fixture": True}
    status = json.loads((tiny.output / "status.json").read_text())
    assert status["status"] == "completed" and status["checkpoint_file_unchanged"]
    assert status["device"] == "cpu" and status["updates"] == 0
    assert status["checkpoint_fitted"] is True and not status["fit_performed"]
    assert "model_fit" not in status
    assert not status["test_levels_played"] and not status["test_outcomes_read"]
    assert not tiny.head.training and not any(p.requires_grad for p in tiny.head.parameters())


@pytest.mark.parametrize(
    "stage",
    [
        "load_history_suite",
        "load_history_features",
        "load_fitted_history_reward_checkpoint",
        "evaluate_history_reward",
    ],
)
def test_cli_failure_is_atomic_visible_and_restores_threads(tiny, monkeypatch, stage):
    threads = torch.get_num_threads()

    def fail(*args, **kwargs):
        assert torch.get_num_threads() == 1
        raise ValueError("fixture rejection")

    monkeypatch.setattr(script, stage, fail)
    with pytest.raises(ValueError, match="fixture rejection"):
        script.main([*tiny.argv, "--torch-threads", "1"])
    assert torch.get_num_threads() == threads
    status = json.loads((tiny.output / "status.json").read_text())
    assert status["status"] == "failed" and status["error_type"] == "ValueError"
    assert status["updates"] == 0 and not status["fit_performed"]
    if stage == "evaluate_history_reward":
        assert status["checkpoint_fitted"] is True
    else:
        assert "checkpoint_fitted" not in status
    assert not (tiny.output / "evaluation.json").exists()
    assert not list(tiny.output.glob("*.tmp"))


def test_interrupt_is_distinguished_from_failure(tiny, monkeypatch):
    threads = torch.get_num_threads()

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(script, "evaluate_history_reward", interrupt)
    with pytest.raises(KeyboardInterrupt):
        script.main(tiny.argv)
    assert torch.get_num_threads() == threads
    status = json.loads((tiny.output / "status.json").read_text())
    assert status["status"] == "interrupted"


@pytest.mark.parametrize("location", ["corpus", "features", "fitted_source", "ancestor"])
def test_output_must_not_overlap_any_source_or_ancestor(tiny, location):
    output = tiny.output.parent if location == "ancestor" else tiny.paths[location] / "nested"
    arguments = tiny.argv.copy()
    arguments[arguments.index("--output") + 1] = str(output)
    with pytest.raises(ValueError, match="outside source"):
        script.main(arguments)
    assert tiny.calls == []
    if location != "ancestor":
        assert not output.exists()


def test_existing_output_is_preserved(tiny):
    tiny.output.mkdir()
    marker = tiny.output / "user.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError, match="already exists"):
        script.main(tiny.argv)
    assert marker.read_text() == "keep" and tiny.calls == []
    assert not (tiny.output / "status.json").exists()


@pytest.mark.parametrize("kind", ["output", "output_parent", "source"])
def test_symlinks_are_rejected_before_writes(tiny, kind):
    arguments = tiny.argv.copy()
    link = tiny.output.parent / "link"
    if kind == "source":
        link.symlink_to(tiny.paths["corpus"], target_is_directory=True)
        arguments[arguments.index("--corpus") + 1] = str(link)
    elif kind == "output_parent":
        link.symlink_to(tiny.output.parent, target_is_directory=True)
        arguments[arguments.index("--output") + 1] = str(link / "child")
    else:
        link.symlink_to(tiny.output, target_is_directory=True)
        arguments[arguments.index("--output") + 1] = str(link)
    with pytest.raises(ValueError, match="symlinks"):
        script.main(arguments)
    assert tiny.calls == [] and not tiny.output.exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--split", "test"],
        ["--event-threshold", "0.2"],
        ["--torch-threads", "0"],
        ["--torch-threads", "9"],
        ["--checkpoint-sha256", "untrusted"],
    ],
)
def test_cli_rejects_test_tuning_unbounded_threads_and_missing_pins(tiny, extra):
    with pytest.raises(SystemExit):
        script.main([*tiny.argv, *extra])
    assert tiny.calls == [] and not tiny.output.exists()


def test_checkpoint_file_mutation_causes_failure_without_completed_report(tiny, monkeypatch):
    original = script.evaluate_history_reward

    def mutate(*args, **kwargs):
        report = original(*args, **kwargs)
        tiny.checkpoint.write_bytes(b"mutated fixture")
        return report

    monkeypatch.setattr(script, "evaluate_history_reward", mutate)
    with pytest.raises(RuntimeError, match="modified the pinned checkpoint"):
        script.main(tiny.argv)
    status = json.loads((tiny.output / "status.json").read_text())
    assert status["status"] == "failed" and not (tiny.output / "evaluation.json").exists()
