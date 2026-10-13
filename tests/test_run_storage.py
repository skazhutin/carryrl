"""Cold checkpoint storage preserves checkpoint identity and protects live inputs."""

import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "manage_run_storage", Path(__file__).parents[1] / "scripts/manage_run_storage.py"
)
storage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(storage)


def fixture(root, name="historical_smoke"):
    checkpoint = root / "runs" / name / "checkpoints" / "rollout_0000"
    checkpoint.mkdir(parents=True)
    data = b"checkpoint payload\0" * 1000
    (checkpoint / "state.pt").write_bytes(data)
    (checkpoint / "manifest.json").write_text(
        json.dumps({"sha256": hashlib.sha256(data).hexdigest()})
    )
    (checkpoint / "language_resources.json").write_text('{"keep": true}')
    return checkpoint, data


def test_roundtrip_and_default_dry_run(tmp_path):
    checkpoint, data = fixture(tmp_path)
    manifest = (checkpoint / "manifest.json").read_bytes()
    result = storage.transform(tmp_path, checkpoint)
    assert not result["applied"]
    assert not (checkpoint / "state.pt.gz").exists()
    result = storage.transform(tmp_path, checkpoint, apply=True)
    assert result["saved_bytes"] > 0
    assert not (checkpoint / "state.pt").exists()
    index = json.loads((checkpoint.parents[1] / "storage_manifest.json").read_text())
    entry = index["checkpoints"]["checkpoints/rollout_0000"]
    assert entry["requires_restore"]
    assert "restore" in entry["restore_command"] and "--apply" in entry["restore_command"]
    storage.transform(tmp_path, checkpoint, restore=True, apply=True)
    assert (checkpoint / "state.pt").read_bytes() == data
    assert not (checkpoint / "state.pt.gz").exists()
    assert (checkpoint / "manifest.json").read_bytes() == manifest
    assert (checkpoint / "language_resources.json").exists()


def test_corrupted_original_is_preserved(tmp_path):
    checkpoint, _ = fixture(tmp_path)
    (checkpoint / "state.pt").write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        storage.transform(tmp_path, checkpoint, apply=True)
    assert (checkpoint / "state.pt").read_bytes() == b"bad"
    assert not (checkpoint / "state.pt.gz").exists()
    assert not list(checkpoint.glob(".storage-*"))


def test_corrupted_archive_is_preserved(tmp_path):
    checkpoint, _ = fixture(tmp_path)
    storage.transform(tmp_path, checkpoint, apply=True)
    (checkpoint / "state.pt.gz").write_bytes(gzip.compress(b"corrupt"))
    with pytest.raises(ValueError, match="hash mismatch"):
        storage.transform(tmp_path, checkpoint, restore=True, apply=True)
    assert (checkpoint / "state.pt.gz").exists()
    assert not (checkpoint / "state.pt").exists()


def test_never_overwrites(tmp_path):
    checkpoint, data = fixture(tmp_path)
    (checkpoint / "state.pt.gz").write_bytes(b"existing")
    with pytest.raises(ValueError, match="overwrite"):
        storage.transform(tmp_path, checkpoint, apply=True)
    assert (checkpoint / "state.pt").read_bytes() == data
    assert (checkpoint / "state.pt.gz").read_bytes() == b"existing"


@pytest.mark.parametrize("reference", ["active", "warm_start", "pin", "command"])
def test_active_and_referenced_protection(tmp_path, reference):
    checkpoint, _ = fixture(tmp_path)
    cache = tmp_path / ".cache"
    cache.mkdir()
    if reference == "active":
        name, value = "active_mlx_launch.json", {"run": str(checkpoint.parents[1])}
    elif reference == "pin":
        name, value = "source_pinsets.json", {"source": str(checkpoint.parents[1])}
    elif reference == "command":
        name, value = "launch.json", {"command": f"train --warm-start {checkpoint}"}
    else:
        name, value = "launch.json", {"warm_start": str(checkpoint)}
    (cache / name).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="protected"):
        storage.transform(tmp_path, checkpoint, apply=True)
    assert not storage.inventory(tmp_path)["checkpoints"][0]["candidate"]


def test_non_smoke_latest_and_current_family_protection(tmp_path):
    checkpoint, _ = fixture(tmp_path, "historical_training")
    with pytest.raises(ValueError, match="protected"):
        storage.transform(tmp_path, checkpoint, apply=True)
    current, _ = fixture(tmp_path, "fresh_corrections12_smoke")
    with pytest.raises(ValueError, match="protected"):
        storage.transform(tmp_path, current, apply=True)


def test_symlink_and_outside_root_protection(tmp_path):
    checkpoint, _ = fixture(tmp_path)
    state = checkpoint / "state.pt"
    state.rename(checkpoint / "actual.pt")
    state.symlink_to(checkpoint / "actual.pt")
    with pytest.raises(ValueError, match="symlink"):
        storage.transform(tmp_path, checkpoint, apply=True)
    with pytest.raises(ValueError, match="inside"):
        storage.transform(tmp_path, tmp_path / "outside", apply=True)


def test_hash_reads_are_bounded():
    class Bounded(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= storage.CHUNK_BYTES
            return super().read(size)

    value = b"x" * (storage.CHUNK_BYTES * 3 + 1)
    assert storage.digest(Bounded(value)) == (hashlib.sha256(value).hexdigest(), len(value))


@pytest.mark.parametrize("suffix", ["json", "toml", "yaml"])
def test_config_references_protect_even_run_paths(tmp_path, suffix):
    checkpoint, _ = fixture(tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    source = str(checkpoint.parents[1])
    if suffix == "json":
        text = json.dumps({"warm_start": source})
    elif suffix == "toml":
        text = f'warm_start = "{source}"\n'
    else:
        text = f'warm_start: "{source}"\n'
    (configs / f"model.{suffix}").write_text(text)
    with pytest.raises(ValueError, match="protected"):
        storage.transform(tmp_path, checkpoint, apply=True)


def test_reference_to_payload_protects_checkpoint(tmp_path):
    checkpoint, _ = fixture(tmp_path)
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "launch.json").write_text(
        json.dumps({"weights": str(checkpoint / "state.pt")})
    )
    with pytest.raises(ValueError, match="protected"):
        storage.transform(tmp_path, checkpoint, apply=True)


def test_compaction_preserves_bytes_and_permissions(tmp_path, monkeypatch):
    payload = tmp_path / "staging.gz"
    value = gzip.compress(b"payload" * 100000)
    payload.write_bytes(value)
    payload.chmod(0o640)
    expected = hashlib.sha256(value).hexdigest()
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("unbounded file read"))
    storage.compact_staging(payload)
    assert storage.file_digest(payload) == (expected, len(value))
    assert payload.stat().st_mode & 0o777 == 0o640
    assert not list(tmp_path.glob(".storage-*"))


def test_invalid_config_fails_closed(tmp_path):
    checkpoint, _ = fixture(tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "bad.json").write_text("invalid")
    with pytest.raises(ValueError, match="protected"):
        storage.transform(tmp_path, checkpoint, apply=True)


@pytest.mark.parametrize("restore", [False, True])
def test_retry_after_manifest_publication_failure(tmp_path, monkeypatch, restore):
    checkpoint, data = fixture(tmp_path)
    if restore:
        storage.transform(tmp_path, checkpoint, apply=True)
    record = storage.record

    def fail_record(*args, **kwargs):
        assert (checkpoint / "state.pt").exists()
        assert (checkpoint / "state.pt.gz").exists()
        raise OSError("injected metadata write failure")

    monkeypatch.setattr(storage, "record", fail_record)
    with pytest.raises(OSError, match="injected"):
        storage.transform(tmp_path, checkpoint, restore=restore, apply=True)
    assert (checkpoint / "state.pt").read_bytes() == data
    assert gzip.decompress((checkpoint / "state.pt.gz").read_bytes()) == data
    monkeypatch.setattr(storage, "record", record)
    dry_run = storage.transform(tmp_path, checkpoint, restore=restore)
    assert dry_run["reconciliation_required"]
    assert (checkpoint / "state.pt").exists() and (checkpoint / "state.pt.gz").exists()
    result = storage.transform(tmp_path, checkpoint, restore=restore, apply=True)
    assert result["reconciled"]
    assert (checkpoint / "state.pt").exists() == restore
    assert (checkpoint / "state.pt.gz").exists() != restore
    index = json.loads((checkpoint.parents[1] / "storage_manifest.json").read_text())
    assert index["checkpoints"]["checkpoints/rollout_0000"]["state"] == (
        "hot" if restore else "archived"
    )


@pytest.mark.parametrize("restore", [False, True])
def test_interrupted_pair_with_divergence_never_removed(tmp_path, restore):
    checkpoint, data = fixture(tmp_path)
    (checkpoint / "state.pt.gz").write_bytes(gzip.compress(b"other checkpoint"))
    packed_before = (checkpoint / "state.pt.gz").read_bytes()
    with pytest.raises(ValueError, match="divergent"):
        storage.transform(tmp_path, checkpoint, restore=restore, apply=True)
    assert (checkpoint / "state.pt").read_bytes() == data
    assert (checkpoint / "state.pt.gz").read_bytes() == packed_before
    assert not (checkpoint.parents[1] / "storage_manifest.json").exists()


@pytest.mark.parametrize("restore", [False, True])
def test_retry_after_interruption_following_durable_manifest(tmp_path, monkeypatch, restore):
    checkpoint, _ = fixture(tmp_path)
    if restore:
        storage.transform(tmp_path, checkpoint, apply=True)
    source = checkpoint / ("state.pt.gz" if restore else "state.pt")
    unlink = Path.unlink

    def interrupt_unlink(path, *args, **kwargs):
        if path == source:
            raise KeyboardInterrupt("simulated stop after manifest")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupt_unlink)
    with pytest.raises(KeyboardInterrupt):
        storage.transform(tmp_path, checkpoint, restore=restore, apply=True)
    index = json.loads((checkpoint.parents[1] / "storage_manifest.json").read_text())
    assert index["checkpoints"]["checkpoints/rollout_0000"]["state"] == (
        "hot" if restore else "archived"
    )
    assert (checkpoint / "state.pt").exists() and (checkpoint / "state.pt.gz").exists()
    monkeypatch.setattr(Path, "unlink", unlink)
    assert storage.transform(tmp_path, checkpoint, restore=restore, apply=True)["reconciled"]
