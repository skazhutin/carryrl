"""Geometry arm append reaches standard publication before any optimizer update."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carryrl.envs.atomic_suite import SUITE_VERSION
from carryrl.training.feature_store import FrozenFeatureWriter, feature_cache_identity
from carryrl.training.imitation import SavedDemonstrationReuse, collect_scripted_demonstrations
from carryrl.training.sequence_config import SequenceRunConfig


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1]
                                                 / "scripts" / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


script = module("navigation_geometry_training", "train_imitation.py")
prepare = module("navigation_geometry_preparation", "prepare_navigation_geometry.py")


def arguments(*flags):
    return ["train_imitation.py", "--config", "unused", "--warm-start", "unused",
            "--output", "unused", *flags]


@pytest.mark.parametrize("flags", [
    ["--navigation-geometry-pairs", "1"], ["--navigation-geometry-mode", "repeat"],
    ["--navigation-geometry-pairs", "65", "--navigation-geometry-mode", "new_walls"],
    *[["--navigation-geometry-pairs", "1", "--navigation-geometry-mode", "new_walls", *extra]
      for extra in (["--appearance-bundle", "unused"],
                    ["--styled-correction-episodes-per-task", "1"],
                    ["--presentation-copies-per-episode", "1"],
                    ["--replacement-control-run", "unused"],
                    ["--augmentation-themes", "slate"], ["--native-train-sizes", "64"])],
])
def test_invalid_combinations_fail_before_config_or_model(monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", arguments(*flags))
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: pytest.fail("loaded config"))
    monkeypatch.setattr(script, "load_sequence_checkpoint",
                        lambda *a, **k: pytest.fail("allocated model"))
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2


@pytest.fixture
def source(tmp_path):
    run = tmp_path / "source"
    run.mkdir()
    config = SequenceRunConfig(tasks=("navigation",), curriculum_difficulty=1,
                               evaluation_difficulty=1)
    dataset = collect_scripted_demonstrations(16, tasks=config.tasks, difficulty=1)
    for filename, value in {
        "config.json": config.to_dict(), "demonstrations.json": dataset.provenance(),
        "metadata.json": {"kind": "imitation_training", "suite_version": SUITE_VERSION,
            "config": config.to_dict(), "initial_agent_sha256": "1" * 64,
            "reward_sha256": "2" * 64, "source_checkpoint_sha256": "3" * 64},
    }.items():
        (run / filename).write_text(json.dumps(value))
    np.savez_compressed(run / "demonstration_frames.npz", **{
        f"frame_{i:06}": frame for i, frame in enumerate(dataset.frames)})
    used = sorted({index for row in dataset.episodes for index in row.frame_indices})
    identity = feature_cache_identity(dataset.frames, used, vision_sha256="6" * 64,
        preprocessing={"implementation": "types.SimpleNamespace",
            "preprocess": "letterbox_bicubic_antialias_imagenet_v1", "image_size": 96,
            "patch_size": 96, "patch_grid": [1, 1]}, feature_shape=(1, 384), dtype="float32")
    cache = tmp_path / "source_features"
    with FrozenFeatureWriter(cache, identity, max_bytes=1024**2) as writer:
        writer.write(used, np.ones((len(used), 1, 384), np.float32))
        store = writer.publish()
    (run / "feature_manifest.json").write_text(json.dumps({
        "feature_cache_dir": str(store.directory.resolve()), "frames": len(used),
        "dtype": "float32", "vision_sha256": "6" * 64,
        "demonstrations_sha256": hashlib.sha256((run / "demonstrations.json").read_bytes())
            .hexdigest()}))
    return run, dataset, config


@pytest.mark.parametrize("mode", ["repeat", "new_walls"])
@pytest.mark.parametrize("reuse_source", [True, False])
def test_selected_arm_appended_exact_base_and_saved_reuse_roundtrip(
        monkeypatch, tmp_path, source, mode, reuse_source):
    run, base, config = source
    output = tmp_path / "combined"
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "manifest.json").write_text(json.dumps({"sha256": "5" * 64}))
    source_bytes = {path.name: path.read_bytes() for path in run.iterdir()}
    original_records = base.provenance()["episodes"]
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(script, "provenance", lambda: {})
    monkeypatch.setattr(script, "tensor_hash", lambda _: "6" * 64)
    vision = SimpleNamespace(image_size=96, patch_size=96, patch_grid=(1, 1), hidden_size=384)
    agent = SimpleNamespace(vision=vision,
        encode_frames=lambda frames: torch.zeros((len(frames), 1, 384)))

    def load(*args, **kwargs):
        assert output.exists()
        return (agent, SimpleNamespace(param_groups=[{}]), object(),
                {"decisions": 0, "weights_revision": 0, "config": config.to_dict()})

    monkeypatch.setattr(script, "load_sequence_checkpoint", load)
    if reuse_source:
        monkeypatch.setattr(script, "collect_scripted_demonstrations",
                            lambda *a, **k: pytest.fail("recollected base"))
    else:
        monkeypatch.setattr(script, "collect_scripted_demonstrations", lambda *a, **k: base)

    class Prepared(Exception):
        pass

    monkeypatch.setattr(script, "evaluate_sequence",
                        lambda *a, **k: (_ for _ in ()).throw(Prepared))
    monkeypatch.setattr(script, "train_imitation", lambda *a, **k: pytest.fail("trained"))
    flags = (["--demonstration-reuse-run", str(run)] if reuse_source else
             ["--episodes-per-task", "16", "--correction-episodes-per-task", "0"])
    monkeypatch.setattr(sys, "argv", arguments("--navigation-geometry-pairs", "8",
        "--navigation-geometry-mode", mode, "--output", str(output), "--warm-start", str(parent),
        "--training-backend", "torch", "--feature-cache-dir", str(tmp_path / "features"), *flags))
    with pytest.raises(Prepared):
        script.main()
    metadata = json.loads((output / "metadata.json").read_text())
    published = json.loads((output / "demonstrations.json").read_text())
    assert published["episodes"][:16] == original_records
    assert len(published["episodes"]) == 24
    assert metadata["navigation_geometry"]["mode"] == mode
    assert metadata["navigation_geometry"]["pairs_collected"] == 8
    assert metadata["navigation_geometry"]["equal_ordered_lengths"]
    assert not (output / "demonstration_frames.npz").is_symlink()
    assert {path.name: path.read_bytes() for path in run.iterdir()} == source_bytes
    if reuse_source:
        assert metadata["demonstration_reuse"]["collection_skipped"] is False
        assert metadata["demonstration_reuse"]["source_corpus_preserved"] is True
        assert metadata["demonstration_reuse"]["appended_episodes"] == 8
    future = SavedDemonstrationReuse.prepare(output, config=config, max_frame_bytes=256 * 1024**2)
    assert future.dataset.provenance() == published
    assert future.summary["producer_lineage"]["navigation_geometry"] == (
        metadata["navigation_geometry"])


def test_prepare_real_source_emits_delta_only_and_never_loads_model(monkeypatch, tmp_path, source):
    run, base, config = source
    config_path = run / "config.json"
    output = tmp_path / "prepared"
    before = {path.name: path.read_bytes() for path in run.iterdir()}
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k: pytest.fail("model"))
    assert prepare.main(["--source-run", str(run), "--config", str(config_path),
                         "--output", str(output)]) == 0
    report = json.loads((output / "manifest.json").read_text())
    assert report["updates"] == 0 and report["status"] == "prepared_not_trained"
    assert report["source_corpus_preserved"] and report["equal_ordered_lengths"]
    assert report["base_episodes"] == 16
    assert report["delta_episodes_per_arm"] == 8
    assert len(report["source_files"]) == 5
    assert len({row["parent_sha256"] for row in report["pairs"]}) == 8
    for mode in ("repeat", "new_walls"):
        provenance = json.loads((output / mode / "demonstrations.json").read_text())
        assert len(provenance["episodes"]) == 8
        assert provenance["frame_bytes"] < base.frame_bytes
    assert before == {path.name: path.read_bytes() for path in run.iterdir()}


def test_unsatisfiable_reuse_fails_before_output_model(monkeypatch, tmp_path, source):
    run, _, config = source
    output = tmp_path / "failed"
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k: pytest.fail("model"))
    monkeypatch.setattr(sys, "argv", arguments("--demonstration-reuse-run", str(run),
        "--navigation-geometry-pairs", "64", "--navigation-geometry-mode", "new_walls",
        "--output", str(output)))
    with pytest.raises(ValueError, match="requested 64 pairs"):
        script.main()
    assert not output.exists()


def test_combined_budget_preflight_rejects_saved_append_before_output_model(
        monkeypatch, tmp_path, source):
    run, base, config = source
    output = tmp_path / "failed_budget"
    original_prepare = SavedDemonstrationReuse.prepare

    def tight_budget(directory, **options):
        reused = original_prepare(directory, **options)
        reused.dataset.max_frame_bytes = reused.dataset.frame_bytes
        return reused

    monkeypatch.setattr(SavedDemonstrationReuse, "prepare", tight_budget)
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k: pytest.fail("model"))
    monkeypatch.setattr(sys, "argv", arguments("--demonstration-reuse-run", str(run),
        "--navigation-geometry-pairs", "8", "--navigation-geometry-mode", "new_walls",
        "--output", str(output)))
    with pytest.raises(MemoryError, match="combined RGB byte budget"):
        script.main()
    assert not output.exists()
