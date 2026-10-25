"""Prepared appearance arms enter training through validated immutable RGB reuse."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carryrl.training.appearance import PreparedAppearanceReuse, prepare_appearance_bundle
from carryrl.training.imitation import SavedDemonstrationReuse, collect_scripted_demonstrations
from carryrl.training.sequence_config import SequenceRunConfig

spec = importlib.util.spec_from_file_location(
    "appearance_bundle_training_script", Path(__file__).resolve().parents[1]
    / "scripts/train_imitation.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def arguments(*flags):
    return ["train_imitation.py", "--config", "unused", "--warm-start", "unused",
            "--output", "unused", *flags]


def appearance_arguments():
    return ["--appearance-bundle", "unused", "--appearance-bundle-sha256", "a" * 64,
            "--appearance-arm", "control"]


@pytest.mark.parametrize("flags", [
    ["--appearance-bundle", "unused"],
    ["--appearance-bundle-sha256", "a" * 64],
    ["--appearance-arm", "control"],
    ["--appearance-bundle", "unused", "--appearance-bundle-sha256", "a" * 64],
    ["--appearance-bundle", "unused", "--appearance-arm", "control"],
    ["--appearance-bundle-sha256", "a" * 64, "--appearance-arm", "control"],
])
def test_all_appearance_identity_arguments_are_required_before_config(monkeypatch, capsys, flags):
    monkeypatch.setattr(sys, "argv", arguments(*flags))
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
    assert "requires bundle, trusted SHA256, and arm together" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [
    ["--demonstration-reuse-run", "unused"], ["--feature-reuse-run", "unused"],
    ["--episodes-per-task=16"], ["--difficulty", "1"],
    ["--correction-episodes-per-task", "0"], ["--correction-tasks", "navigation"],
    ["--expert-probability", "0.5"], ["--learner-prefix-steps", "0"],
    ["--presentation-copies-per-episode", "0"], ["--memory-cue-pairs", "0"],
    ["--native-train-sizes", "64"], ["--native-episodes-per-task", "0"],
    ["--correction-replacement-bundle", "unused"],
    ["--correction-replacement-sha256", "a" * 64], ["--replacement-control-run", "unused"],
    ["--styled-correction-episodes-per-task", "0"],
    ["--styled-correction-expert-probability", "0.35"], ["--styled-correction-prefix-steps", "4"],
    ["--augmentation-themes", "slate"], ["--augmentation-hud-layouts", "left"],
    ["--augmentation-textures"], ["--no-augmentation-textures"],
    ["--augmentation-canvas-padding=8"],
])
def test_appearance_excludes_collection_and_saved_reuse_before_config(monkeypatch, capsys, flags):
    monkeypatch.setattr(sys, "argv", arguments(*appearance_arguments(), *flags))
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "appearance reuse excludes collection and saved reuse flags" in message
    assert flags[0].split("=", 1)[0] in message


def test_invalid_prepared_bundle_fails_before_output_or_model(monkeypatch, tmp_path):
    output = tmp_path / "output"
    config = SequenceRunConfig(tasks=("navigation",))
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(script, "load_sequence_checkpoint",
                        lambda *a, **k: pytest.fail("invalid bundle allocated model"))

    def invalid(directory, **options):
        assert not output.exists()
        raise ValueError("appearance bundle SHA256 mismatch")

    monkeypatch.setattr(PreparedAppearanceReuse, "prepare", invalid)
    monkeypatch.setattr(sys, "argv", arguments(*appearance_arguments(), "--output", str(output),
                                              "--training-backend", "torch"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        script.main()
    assert not output.exists()


@pytest.fixture
def prepared(tmp_path):
    config = SequenceRunConfig(tasks=("navigation",), curriculum_difficulty=1,
                               evaluation_difficulty=1)
    dataset = collect_scripted_demonstrations(1, tasks=config.tasks, difficulty=1)
    bundle = tmp_path.resolve() / "bundle"
    report = prepare_appearance_bundle(dataset, bundle, config=config)
    return bundle, report["source_bundle_sha256"], config


@pytest.mark.parametrize("arm", ["control", "theme_shift"])
def test_prepared_arm_is_validated_before_model_then_published_as_standard_corpus(
        monkeypatch, tmp_path, prepared, arm):
    bundle, digest, config = prepared
    output = tmp_path / "output"
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "manifest.json").write_text(json.dumps({"sha256": "5" * 64}))
    source_bytes = {name: (bundle / name).read_bytes() for name in ("bundle.json", "frames.npz")}
    calls = []
    original_prepare = PreparedAppearanceReuse.prepare

    def prepare(directory, **options):
        assert not output.exists()
        assert options["expected_sha256"] == digest
        assert options["arm"] == arm
        assert options["max_frame_bytes"] == 256 * 1024**2
        reuse = original_prepare(directory, **options)
        calls.append(reuse)
        return reuse

    monkeypatch.setattr(PreparedAppearanceReuse, "prepare", prepare)
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(script, "provenance", lambda: {})
    monkeypatch.setattr(script, "tensor_hash", lambda _: "6" * 64)
    vision = SimpleNamespace(image_size=96, patch_size=96, patch_grid=(1, 1), hidden_size=384)
    agent = SimpleNamespace(vision=vision)
    agent.encode_frames = lambda frames: torch.zeros((len(frames), 1, 384))

    def load(*args, **kwargs):
        assert len(calls) == 1 and output.exists()
        assert args[1].curriculum_difficulty == args[1].evaluation_difficulty == 1
        return (agent, SimpleNamespace(param_groups=[{}]), object(),
                {"decisions": 0, "weights_revision": 0, "config": config.to_dict()})

    monkeypatch.setattr(script, "load_sequence_checkpoint", load)
    for name in ("collect_scripted_demonstrations", "collect_scripted_corrections",
                 "collect_native_scripted_demonstrations", "collect_memory_cue_counterparts",
                 "augment_demonstrations"):
        monkeypatch.setattr(script, name,
                            lambda *a, **k: pytest.fail("prepared arm recollected RGB"))

    class Prepared(Exception):
        pass

    def stop_before_evaluation(*args, **kwargs):
        raise Prepared

    monkeypatch.setattr(script, "evaluate_sequence", stop_before_evaluation)
    cache = tmp_path / "cache"
    monkeypatch.setattr(sys, "argv", arguments("--appearance-bundle", str(bundle),
        "--appearance-bundle-sha256", digest, "--appearance-arm", arm,
        "--output", str(output), "--warm-start", str(parent), "--training-backend", "torch",
        "--feature-cache-dir", str(cache)))
    with pytest.raises(Prepared):
        script.main()
    dataset = calls[0].dataset
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["appearance_bundle"] == {
        "source": str(bundle), "bundle_sha256": digest, "arm": arm}
    assert metadata["demonstration_reuse"] == calls[0].summary
    assert metadata["source_checkpoint"] == str(parent.resolve())
    assert "collection skipped" in metadata["collection"]
    assert json.loads((output / "demonstrations.json").read_text()) == dataset.provenance()
    assert not (output / "demonstration_frames.npz").is_symlink()
    with np.load(output / "demonstration_frames.npz", allow_pickle=False) as frames:
        assert len(frames.files) == len(dataset.frames)
        for index, frame in enumerate(dataset.frames):
            np.testing.assert_array_equal(frames[f"frame_{index:06}"], frame)
    manifest = json.loads((output / "feature_manifest.json").read_text())
    assert manifest["feature_cache_hit"] is False
    assert manifest["feature_cache_dir"]
    assert "feature_reuse" not in manifest
    assert source_bytes == {name: (bundle / name).read_bytes() for name in source_bytes}
    calls[0].assert_unchanged()
    future_reuse = SavedDemonstrationReuse.prepare(output, config=config,
                                                  max_frame_bytes=256 * 1024**2)
    assert future_reuse.dataset.provenance() == dataset.provenance()
    assert future_reuse.summary["producer_lineage"]["demonstration_reuse"] == calls[0].summary
