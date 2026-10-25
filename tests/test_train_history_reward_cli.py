"""CLI wiring only: no real fitting in these tests."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def cli(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("train_history_cli",
        Path(__file__).parents[1] / "scripts/train_history_reward.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    corpus, features = tmp_path / "corpus", tmp_path / "features"
    corpus.mkdir()
    features.mkdir()
    calls = []
    corpus_object = SimpleNamespace(directory=corpus)
    feature_object = SimpleNamespace(directory=features)
    monkeypatch.setattr(module, "provenance", lambda: {"revision": "fixture"})

    def corpus_loader(path, *, expected_sha256):
        assert path == corpus and expected_sha256 == "a" * 64
        return corpus_object

    def feature_loader(path, supplied, *, expected_sha256):
        assert path == features and supplied is corpus_object and expected_sha256 == "b" * 64
        return feature_object

    def fit(supplied_corpus, supplied_features, output, **kwargs):
        assert supplied_corpus is corpus_object and supplied_features is feature_object
        output.mkdir()
        calls.append(kwargs)

    monkeypatch.setattr(module, "load_history_suite", corpus_loader)
    monkeypatch.setattr(module, "load_history_features", feature_loader)
    monkeypatch.setattr(module, "fit_history_reward", fit)
    args = ["--corpus", str(corpus), "--corpus-sha256", "a" * 64,
            "--features", str(features), "--features-sha256", "b" * 64,
            "--epochs", "3", "--learning-rate", "0.001", "--seed", "7",
            "--max-wall-seconds", "60"]
    return module, args, calls


def test_explicit_settings_and_source_pins_reach_fit(cli, tmp_path):
    module, args, calls = cli
    output = tmp_path / "fit"
    assert module.main([*args, "--output", str(output)]) == 0
    assert calls == [{"epochs": 3, "learning_rate": 0.001, "seed": 7,
                     "hidden_size": 128, "max_wall_seconds": 60.0, "max_updates": 100_000}]
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["policy_updates"] == 0 and not metadata["ppo_eligible"]
    with pytest.raises(FileExistsError):
        module.main([*args, "--output", str(output)])
    assert len(calls) == 1


@pytest.mark.parametrize("destination", ["corpus/derived", "features/derived", "."])
def test_output_cannot_modify_source_trees(cli, tmp_path, destination):
    module, args, calls = cli
    with pytest.raises((ValueError, FileExistsError)):
        module.main([*args, "--output", str(tmp_path / destination)])
    assert calls == []


def test_source_failure_does_not_claim_completed_fit(cli, tmp_path, monkeypatch):
    module, args, calls = cli

    def fail(*args, **kwargs):
        raise ValueError("bad corpus pin")

    monkeypatch.setattr(module, "load_history_suite", fail)
    with pytest.raises(ValueError, match="bad corpus pin"):
        module.main([*args, "--output", str(tmp_path / "fit")])
    assert not (tmp_path / "fit").exists() and calls == []
