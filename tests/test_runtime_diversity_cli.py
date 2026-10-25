"""Runtime axes share preflight, append, publication and model-free preparation."""

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
from carryrl.training import runtime_diversity
from carryrl.training.feature_store import FrozenFeatureWriter, feature_cache_identity
from carryrl.training.imitation import (
    DemonstrationDataset,
    SavedDemonstrationReuse,
    collect_scripted_demonstrations,
)
from carryrl.training.runtime_diversity import (
    RuntimeDiversityPairs,
    checked_combined_frame_bytes,
    collect_runtime_diversity_pairs,
)
from carryrl.training.sequence_config import SequenceRunConfig


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1]
                                                 / "scripts" / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


script = module("runtime_diversity_training", "train_imitation.py")
prepare = module("runtime_diversity_preparation", "prepare_runtime_diversity.py")


def arguments(*flags):
    return ["train_imitation.py", "--config", "unused", "--warm-start", "unused",
            "--output", "unused", *flags]


@pytest.mark.parametrize("flags", [
    ["--runtime-diversity-axis", "reaction_dynamics"],
    ["--runtime-diversity-arm", "repeat"],
    ["--runtime-diversity-pairs", "0"],
    ["--runtime-diversity-axis", "reaction_dynamics", "--runtime-diversity-pairs", "65",
     "--runtime-diversity-arm", "varied"],
    ["--runtime-diversity-axis", "reaction_dynamics", "--runtime-diversity-pairs", "1",
     "--runtime-diversity-arm", "varied", "--navigation-geometry-pairs", "0"],
    *[["--runtime-diversity-axis", "reaction_dynamics", "--runtime-diversity-pairs", "1",
       "--runtime-diversity-arm", "varied", *extra] for extra in (
          ["--appearance-bundle", "unused"], ["--memory-cue-pairs", "0"],
          ["--styled-correction-episodes-per-task", "0"],
          ["--presentation-copies-per-episode", "0"],
          ["--replacement-control-run", "unused"], ["--native-train-sizes", "64"],
          ["--augmentation-themes", "slate"])],
])
def test_invalid_flags_precede_config_and_model(monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", arguments(*flags))
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: pytest.fail("loaded config"))
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k: pytest.fail("model"))
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2


@pytest.fixture(params=["navigation", "reaction"])
def source(request, tmp_path):
    run = tmp_path / "source"
    run.mkdir()
    config = SequenceRunConfig(tasks=(request.param,), curriculum_difficulty=1,
                               evaluation_difficulty=1)
    dataset = collect_scripted_demonstrations(16, tasks=config.tasks, difficulty=1)
    for name, value in {
        "config.json": config.to_dict(), "demonstrations.json": dataset.provenance(),
        "metadata.json": {"kind": "imitation_training", "suite_version": SUITE_VERSION,
            "config": config.to_dict(), "initial_agent_sha256": "1" * 64,
            "reward_sha256": "2" * 64, "source_checkpoint_sha256": "3" * 64},
    }.items():
        (run / name).write_text(json.dumps(value))
    np.savez_compressed(run / "demonstration_frames.npz", **{
        f"frame_{i:06}": frame for i, frame in enumerate(dataset.frames)})
    used = sorted({index for row in dataset.episodes for index in row.frame_indices})
    identity = feature_cache_identity(dataset.frames, used, vision_sha256="6" * 64,
        preprocessing={"implementation": "types.SimpleNamespace",
            "preprocess": "letterbox_bicubic_antialias_imagenet_v1", "image_size": 96,
            "patch_size": 96, "patch_grid": [1, 1]}, feature_shape=(1, 384), dtype="float32")
    with FrozenFeatureWriter(tmp_path / "features", identity,
                             max_bytes=8 * 1024**2) as writer:
        writer.write(used, np.ones((len(used), 1, 384), np.float32))
        store = writer.publish()
    (run / "feature_manifest.json").write_text(json.dumps({
        "feature_cache_dir": str(store.directory.resolve()), "frames": len(used),
        "feature_cache_sha256": store.manifest["sha256"],
        "feature_cache_created_utc": store.manifest["created_utc"],
        "bytes": store.manifest["bytes"],
        "dtype": "float32", "vision_sha256": "6" * 64,
        "demonstrations_sha256": hashlib.sha256((run / "demonstrations.json").read_bytes())
            .hexdigest()}))
    return run, dataset, config


def axis_for(config):
    return "navigation_geometry" if config.tasks == ("navigation",) else "reaction_dynamics"


@pytest.mark.parametrize("arm", ["repeat", "varied"])
@pytest.mark.parametrize("saved_reuse", [True, False])
def test_new_axes_append_publish_and_saved_reuse_roundtrip(
        monkeypatch, tmp_path, source, arm, saved_reuse):
    run, base, config = source
    output = tmp_path / "combined"
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "manifest.json").write_text(json.dumps({"sha256": "5" * 64}))
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    original_records = base.provenance()["episodes"]
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(script, "provenance", lambda: {})
    monkeypatch.setattr(script, "tensor_hash", lambda _: "6" * 64)
    vision = SimpleNamespace(image_size=96, patch_size=96, patch_grid=(1, 1), hidden_size=384)
    agent = SimpleNamespace(vision=vision,
        encode_frames=lambda frames: torch.zeros((len(frames), 1, 384)))
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k: (
        agent, SimpleNamespace(param_groups=[{}]), object(),
        {"decisions": 0, "weights_revision": 0, "config": config.to_dict()}))
    if saved_reuse:
        monkeypatch.setattr(script, "collect_scripted_demonstrations",
                            lambda *a, **k: pytest.fail("recollected base"))
    else:
        monkeypatch.setattr(script, "collect_scripted_demonstrations", lambda *a, **k: base)

    class Prepared(Exception):
        pass

    monkeypatch.setattr(script, "evaluate_sequence",
                        lambda *a, **k: (_ for _ in ()).throw(Prepared))
    monkeypatch.setattr(script, "train_imitation", lambda *a, **k: pytest.fail("trained"))
    flags = (["--demonstration-reuse-run", str(run), "--feature-reuse-run", str(run)]
             if saved_reuse else ["--episodes-per-task", "16"])
    monkeypatch.setattr(sys, "argv", arguments("--runtime-diversity-axis", axis_for(config),
        "--runtime-diversity-pairs", "2", "--runtime-diversity-arm", arm,
        "--output", str(output), "--warm-start", str(parent), "--training-backend", "torch",
        "--feature-cache-dir", str(tmp_path / "features"), *flags))
    with pytest.raises(Prepared):
        script.main()
    metadata = json.loads((output / "metadata.json").read_text())
    published = json.loads((output / "demonstrations.json").read_text())
    assert published["episodes"][:16] == original_records
    assert len(published["episodes"]) == 18
    assert metadata["runtime_diversity"]["axis"] == axis_for(config)
    assert metadata["runtime_diversity"]["arm"] == arm
    assert metadata["runtime_diversity"]["pairs_collected"] == 2
    assert metadata["runtime_diversity"]["equal_ordered_lengths"]
    assert not (output / "demonstration_frames.npz").is_symlink()
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
    if saved_reuse:
        assert metadata["demonstration_reuse"]["source_corpus_preserved"]
        assert metadata["demonstration_reuse"]["appended_episodes"] == 2
    future = SavedDemonstrationReuse.prepare(output, config=config, max_frame_bytes=256 * 1024**2)
    assert future.dataset.provenance() == published
    assert future.summary["producer_lineage"]["runtime_diversity"] == metadata["runtime_diversity"]


def test_saved_preflight_budget_fails_before_output_or_model(monkeypatch, tmp_path, source):
    run, base, config = source
    output = tmp_path / "absent"
    prepare_source = SavedDemonstrationReuse.prepare

    def tight_budget(*a, **k):
        source = prepare_source(*a, **k)
        source.dataset.max_frame_bytes = base.frame_bytes
        return source

    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(script.SavedDemonstrationReuse, "prepare", tight_budget)
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k: pytest.fail("model"))
    monkeypatch.setattr(sys, "argv", arguments("--runtime-diversity-axis", axis_for(config),
        "--runtime-diversity-pairs", "2", "--runtime-diversity-arm", "varied",
        "--demonstration-reuse-run", str(run), "--output", str(output)))
    with pytest.raises(MemoryError, match="combined RGB byte budget"):
        script.main()
    assert not output.exists()


def test_preparation_emits_only_authenticated_deltas(monkeypatch, tmp_path, source):
    run, base, config = source
    output = tmp_path / "prepared"
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k: pytest.fail("model"))
    assert prepare.main(["--axis", axis_for(config), "--source-run", str(run),
        "--config", str(run / "config.json"), "--output", str(output), "--pairs", "2"]) == 0
    report = json.loads((output / "manifest.json").read_text())
    assert report["status"] == "prepared_not_trained" and report["updates"] == 0
    assert report["axis"] == axis_for(config)
    assert report["base_episodes"] == 16 and report["delta_episodes_per_arm"] == 2
    assert report["source_corpus_preserved"] and len(report["source_files"]) == 5
    assert set(p.name for p in output.iterdir()) == {
        "repeat", "varied", "metadata.json", "manifest.json", "status.json"}
    for arm in ("repeat", "varied"):
        assert set(p.name for p in (output / arm).iterdir()) == {
            "demonstrations.json", "demonstration_frames.npz"}
        provenance = json.loads((output / arm / "demonstrations.json").read_text())
        assert len(provenance["episodes"]) == 2
        assert provenance["frame_bytes"] < base.frame_bytes
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before


def test_failed_publication_never_claims_prepared(monkeypatch, tmp_path, source):
    run, _, config = source
    output = tmp_path / "failed"
    monkeypatch.setattr(runtime_diversity.np, "savez_compressed",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk unavailable")))
    with pytest.raises(OSError, match="disk unavailable"):
        runtime_diversity.prepare_saved_runtime_diversity(run, config, output,
            axis=axis_for(config), pairs=1)
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert not (output / "manifest.json").exists()


def test_navigation_alias_preserves_exact_manifest_and_deltas():
    from carryrl.training.navigation_geometry import collect_navigation_geometry_pairs

    base = collect_scripted_demonstrations(2, tasks=("navigation",), difficulty=1)
    old = collect_navigation_geometry_pairs(base, max_pairs=1)
    new = collect_runtime_diversity_pairs("navigation_geometry", base, max_pairs=1)
    assert new.manifest == old.manifest
    assert new.repeat.provenance() == old.repeat.provenance()
    assert new.varied.provenance() == old.new_walls.provenance()
    for mode, old_arm in (("repeat", old.repeat), ("varied", old.new_walls)):
        assert all(np.array_equal(a, b) for a, b in zip(new.arm(mode).frames,
                                                      old_arm.frames, strict=True))


def test_selector_and_arm_are_closed_choices():
    empty = DemonstrationDataset()
    pairs = RuntimeDiversityPairs(empty, empty, {})
    with pytest.raises(ValueError, match="arm"):
        pairs.arm("new_walls")
    with pytest.raises(ValueError, match="axis"):
        collect_runtime_diversity_pairs("arbitrary", empty)


def test_combined_budget_hashes_only_delta_and_deduplicates():
    base, arm = DemonstrationDataset(), DemonstrationDataset()
    frame = np.zeros((2, 2, 3), np.uint8)
    base.intern_frame(frame)
    arm.intern_frame(frame)
    arm.intern_frame(frame + 1)
    # The base's validated index suffices; never scan or rehash its frames.
    base.frames = None
    assert checked_combined_frame_bytes(base, {"repeat": arm}, 24) == {"repeat": 24}
    with pytest.raises(MemoryError, match="combined RGB byte budget"):
        checked_combined_frame_bytes(base, {"repeat": arm}, 23)
    base.max_frame_bytes = 12
    with pytest.raises(MemoryError, match="combined RGB byte budget"):
        checked_combined_frame_bytes(base, {"repeat": arm}, 24)


def test_axis_task_preflight_precedes_output_and_model(monkeypatch, tmp_path):
    output = tmp_path / "absent"
    monkeypatch.setattr(script.SequenceRunConfig, "load",
                        lambda _: SequenceRunConfig(tasks=("navigation",)))
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k: pytest.fail("model"))
    monkeypatch.setattr(sys, "argv", arguments("--runtime-diversity-axis", "reaction_dynamics",
        "--runtime-diversity-pairs", "1", "--runtime-diversity-arm", "varied",
        "--output", str(output)))
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
    assert not output.exists()


def test_combined_budget_guards_all_arms_at_fixed_config_endpoint(monkeypatch):
    config = SequenceRunConfig(tasks=("reaction",), level_bank_seed=20261008,
                               evaluation_difficulty=0.8)
    base, repeat, varied = (DemonstrationDataset() for _ in range(3))
    observed = []
    monkeypatch.setattr(runtime_diversity, "validate_dataset_runtime_holdout",
                        lambda arm, **kwargs: observed.append((arm, kwargs)))
    checked_combined_frame_bytes(base, {"repeat": repeat, "varied": varied}, 1024,
                                 config=config)
    assert observed == [(arm, {"level_bank_seed": 20261008, "evaluation_difficulty": 0.8})
                        for arm in (repeat, varied)]
