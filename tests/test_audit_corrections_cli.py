"""Whole-cohort TRAIN audit authenticates data and never opens a checkpoint with torch."""

import gzip
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_imitation_correction_replacement import INSTRUCTIONS, SOURCE_IDENTITY, bundle_sha
from test_imitation_correction_replacement import bundle as bundle

from carryrl.agent.actions import CODEC_VERSION, PhysicalAction
from carryrl.agent.model import AgentConfig
from carryrl.training.imitation import CORRECTION_SOURCE
from carryrl.training.run import atomic_json, file_sha256
from carryrl.training.sequence_config import SequenceRunConfig

spec = importlib.util.spec_from_file_location(
    "audit_corrections_script", Path(__file__).parents[1] / "scripts/audit_corrections.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


@pytest.fixture
def source(bundle, tmp_path, monkeypatch):
    directory, _ = bundle
    checkpoint = tmp_path / "runs/checkpoint/checkpoints/rollout_0000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"fixture: never deserialize a checkpoint")
    manifest = {**SOURCE_IDENTITY, "sha256": file_sha256(checkpoint / "state.pt"),
                "codec_version": CODEC_VERSION}
    atomic_json(checkpoint / "manifest.json", manifest)
    coverage_path = directory / "coverage.json"
    coverage = json.loads(coverage_path.read_text())
    coverage["preregistration"]["expected_checkpoint_identity"] = manifest
    coverage["preregistration"]["checkpoint"] = str(checkpoint)
    atomic_json(coverage_path, coverage)
    digest = bundle_sha(directory)
    config = SequenceRunConfig(tasks=("interaction",), level_bank_seed=20261008,
        agent=replace(AgentConfig(), text_encoder="multilingual_e5_small"),
        episode_instructions={"interaction": INSTRUCTIONS["interaction"]})
    pins = {path.name: file_sha256(path) for path in checkpoint.iterdir()}
    metadata = {"kind": "frozen_train_correction_collection", "split": "train",
        "training_updates": 0, "optimizer_steps": 0, "policy_and_reward_frozen": True,
        "checkpoint": str(checkpoint), "checkpoint_identity": manifest,
        "checkpoint_files_sha256_before": pins, "checkpoint_files_sha256_after": pins,
        "config": config.to_dict(), "bundle_sha256": digest,
        "settings": {"tasks": ["interaction"], "episodes_per_task": 2, "seed": 13288,
                     "max_steps": 80, "difficulty": 1.0, "image_size": 96}}
    status = {"status": "completed", "training_updates": 0, "optimizer_steps": 0,
        "checkpoint_files_unchanged": True, "policy_and_reward_frozen": True,
        "bundle_sha256": digest}
    atomic_json(directory / "metadata.json", metadata)
    atomic_json(directory / "status.json", status)
    monkeypatch.setattr(script, "provenance", lambda: {"source_and_lock_sha256": "fixture"})
    return directory, checkpoint, digest


def arguments(source, output):
    directory, checkpoint, digest = source
    return ["--bundle", str(directory), "--bundle-sha256", digest,
            "--checkpoint", str(checkpoint), "--output", str(output)]


def test_authenticates_all_rows_replays_suffix_frames_and_never_loads_models(source, tmp_path,
                                                                           monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("TRAIN audit must not load a model or construct an optimizer")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    prepared = script.prepare_source(*source)
    pins = dict(prepared.file_hashes)
    output = tmp_path / "audit"
    assert script.main(arguments(source, output)) == 0
    result = json.loads((output / "aggregate.json").read_text())
    assert result["status"] == "completed" and result["training_updates"] == 0
    assert result["episodes"] == result["coverage"]["episodes"] == 2
    assert result["by_anchor"]["step_0"]["rescued_success"] == 2
    assert all(sum(counts.values()) == 2 for counts in result["by_anchor"].values())
    for entry in result["published"]:
        path = Path(entry["report"])
        assert file_sha256(path) == entry["report_sha256"]
        row = json.loads(path.read_text())
        assert row["coverage"]["source_episode"]["split"] == "train"
        assert row["rescue"]["source_kind"] == "authenticated_train_correction"
        assert row["coverage"]["episode_content_sha256"] == entry["episode_content_sha256"]
        frames_path = path.parent / "suffix_frames.npz"
        assert file_sha256(frames_path) == row["suffix_frames_sha256"]
        with np.load(frames_path, allow_pickle=False) as frames:
            expected_keys = set()
            for attempt in row["rescue"]["attempts"]:
                for number, expected in enumerate(attempt["frame_sha256"]):
                    key = f"{attempt['anchor_id']}_frame_{number:03d}"
                    expected_keys.add(key)
                    assert hashlib.sha256(frames[key].tobytes()).hexdigest() == expected
            assert set(frames.files) == expected_keys
    assert {path: file_sha256(Path(path)) for path in pins} == pins


def test_changed_bundle_digest_is_rejected_before_output(source, tmp_path):
    directory, checkpoint, _ = source
    output = tmp_path / "unused"
    with pytest.raises(ValueError, match="completed frozen TRAIN source"):
        script.main(arguments((directory, checkpoint, "0" * 64), output))
    assert not output.exists()


def test_recomputed_digest_does_not_bypass_actual_history_authentication(source, tmp_path):
    directory, checkpoint, _ = source
    path = directory / "trajectories.json"
    data = json.loads(path.read_text())
    row = next(row for row in data["episodes"] if row["source"] == CORRECTION_SOURCE)
    row["executed_actions"][0] = PhysicalAction(wheel=1).to_dict()
    atomic_json(path, data)
    digest = bundle_sha(directory)
    for name in ("metadata.json", "status.json"):
        value = json.loads((directory / name).read_text())
        value["bundle_sha256"] = digest
        atomic_json(directory / name, value)
    output = tmp_path / "unused"
    with pytest.raises(ValueError, match="replacement"):
        script.main(arguments((directory, checkpoint, digest), output))
    assert not output.exists()


def test_checkpoint_mutation_during_audit_is_failed_not_published(source, tmp_path, monkeypatch):
    original = script.describe_correction_coverage

    def change(*args, **kwargs):
        (source[1] / "state.pt").write_bytes(b"changed outside the audit")
        return original(*args, **kwargs)

    monkeypatch.setattr(script, "describe_correction_coverage", change)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="source or checkpoint changed"):
        script.main(arguments(source, output))
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert not (output / "aggregate.json").exists()


def test_publication_failure_preserves_explicit_failed_status(source, tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(script.np, "savez_compressed", fail)
    output = tmp_path / "failed"
    with pytest.raises(OSError, match="disk full"):
        script.main(arguments(source, output))
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert not (output / "aggregate.json").exists()


@pytest.mark.parametrize("extra", [["--anchors", "20"], ["--stall-window", "0"]])
def test_invalid_protocol_fails_before_reading_source(source, tmp_path, monkeypatch, extra):
    monkeypatch.setattr(script, "prepare_source", lambda *_: pytest.fail("preflight first"))
    with pytest.raises(SystemExit):
        script.main([*arguments(source, tmp_path / "unused"), *extra])


def test_existing_output_is_not_overwritten(source, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "user-owned"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        script.main(arguments(source, output))
    assert marker.read_text() == "keep"


def test_cold_checkpoint_source_and_recheck_restore_original_pins(source, monkeypatch):
    directory, checkpoint, digest = source
    monkeypatch.setattr("carryrl.training.checkpoint_storage.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=64 * 1024**3))
    monkeypatch.setattr("torch.load", lambda *a, **k: pytest.fail("must not load weights"))
    state = checkpoint / "state.pt"
    original = state.read_bytes()
    for prepared in (None, "recheck"):
        state.with_suffix(".pt.gz").write_bytes(gzip.compress(original))
        state.unlink()
        if prepared is None:
            authenticated = script.prepare_source(directory, checkpoint, digest)
        else:
            authenticated.assert_unchanged()
        assert state.read_bytes() == original
        assert not state.with_suffix(".pt.gz").exists()
