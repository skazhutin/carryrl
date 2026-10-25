"""Cold chunk storage preserves the original container, including ZIP metadata."""

import hashlib
import io
import json
import os
import zipfile

import pytest

from carryrl.training import checkpoint_chunks as chunks


def checkpoint(tmp_path, name="one", data=b"checkpoint" * 100):
    path = tmp_path / "runs" / name / "checkpoints" / "step_1"
    path.mkdir(parents=True)
    source = path / "state.pt"
    source.write_bytes(data)
    return path, source, hashlib.sha256(data).hexdigest()


def archive(root, path, source, digest):
    return chunks.archive_checkpoint(root, path, source, digest, min_free_bytes=0)


def zip_bytes(prefix, tensor):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as result:
        result.writestr(f"{prefix}/data.pkl", b"metadata" + prefix.encode())
        result.writestr(f"{prefix}/data/0", tensor)
        result.writestr(f"{prefix}/version", b"3\n")
    return stream.getvalue()


@pytest.mark.parametrize("data", [
    b"raw\x00bytes" * 1000, zip_bytes("archive", b"tensor" * 1000), b"",
])
def test_exact_roundtrip(tmp_path, data):
    path, source, digest = checkpoint(tmp_path, data=data)
    descriptor = archive(tmp_path, path, source, digest)
    assert descriptor["bytes"] == len(data)
    assert source.read_bytes() == data
    target = path / "restore.tmp"
    chunks.restore_checkpoint(tmp_path, path, target, digest)
    assert target.read_bytes() == data
    with pytest.raises(FileExistsError):
        chunks.restore_checkpoint(tmp_path, path, target, digest)
    assert target.read_bytes() == data


def test_zip_tensor_dedup_ignores_container_offsets(tmp_path):
    tensor = bytes(range(256)) * 1000
    first = checkpoint(tmp_path, "one", zip_bytes("short", tensor))
    second = checkpoint(tmp_path, "two", zip_bytes("much_longer_container_name", tensor))
    descriptor1 = archive(tmp_path, *first)
    descriptor2 = archive(tmp_path, *second)
    tensor_sha = hashlib.sha256(tensor).hexdigest()
    assert any(item["sha256"] == tensor_sha for item in descriptor1["chunks"])
    assert any(item["sha256"] == tensor_sha for item in descriptor2["chunks"])
    assert len(list((tmp_path / "runs/.checkpoint_store").glob(tensor_sha + ".gz"))) == 1


@pytest.mark.parametrize("damage", ["missing", "corrupt", "checksum", "oversized"])
def test_restore_corruption_never_publishes_partial_file(tmp_path, damage):
    path, source, digest = checkpoint(tmp_path)
    descriptor = archive(tmp_path, path, source, digest)
    payload = tmp_path / "runs/.checkpoint_store" / (descriptor["chunks"][0]["sha256"] + ".gz")
    if damage == "missing":
        payload.unlink()
    elif damage == "corrupt":
        payload.write_bytes(b"corrupt")
    else:
        import gzip
        size = descriptor["chunks"][0]["bytes"] if damage == "checksum" else chunks.CHUNK_BYTES * 2
        payload.write_bytes(gzip.compress(b"x" * size))
    target = path / "restore.tmp"
    with pytest.raises((OSError, ValueError)):
        chunks.restore_checkpoint(tmp_path, path, target, digest)
    assert not target.exists()
    assert source.exists()


def test_interrupted_archive_preserves_source(tmp_path, monkeypatch):
    path, source, digest = checkpoint(tmp_path)
    original = source.read_bytes()
    def interrupted(*args, **kwargs):
        raise InterruptedError("simulated interruption")
    monkeypatch.setattr(chunks, "_publish", interrupted)
    with pytest.raises(InterruptedError):
        archive(tmp_path, path, source, digest)
    assert source.read_bytes() == original
    assert not (path / chunks.DESCRIPTOR).exists()


def test_low_disk_and_wrong_checksum_preserve_source(tmp_path, monkeypatch):
    path, source, digest = checkpoint(tmp_path)
    monkeypatch.setattr(chunks.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 0})())
    with pytest.raises(OSError, match="space"):
        archive(tmp_path, path, source, digest)
    with pytest.raises(ValueError, match="checksum"):
        archive(tmp_path, path, source, "0" * 64)
    assert source.exists()
    assert not (path / chunks.DESCRIPTOR).exists()


def test_traversal_and_symlinks_are_rejected(tmp_path):
    path, source, digest = checkpoint(tmp_path)
    archive(tmp_path, path, source, digest)
    descriptor_path = path / chunks.DESCRIPTOR
    descriptor = json.loads(descriptor_path.read_text())
    descriptor["chunks"][0]["sha256"] = "../escape"
    descriptor_path.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match="checksum"):
        chunks.restore_checkpoint(tmp_path, path, path / "restore.tmp", digest)
    with pytest.raises(ValueError, match="inside runs"):
        chunks.archive_checkpoint(tmp_path, tmp_path / "outside", source, digest)


def test_gc_retains_referenced_and_removes_only_old_orphans(tmp_path):
    path, source, digest = checkpoint(tmp_path)
    descriptor = archive(tmp_path, path, source, digest)
    store = tmp_path / "runs/.checkpoint_store"
    orphan = store / ("0" * 64 + ".gz")
    orphan.write_bytes(b"orphan")
    recent = store / ("1" * 64 + ".gz")
    recent.write_bytes(b"recent")
    os.utime(orphan, (1, 1))
    for item in descriptor["chunks"]:
        os.utime(store / (item["sha256"] + ".gz"), (1, 1))
    assert chunks.cleanup_orphan_chunks(tmp_path)["removed_chunks"] == 1
    assert recent.exists()
    chunks.restore_checkpoint(tmp_path, path, path / "restored.tmp", digest)


@pytest.mark.parametrize("damage", ["invalid", "unreadable", "symlink"])
def test_gc_fails_closed_on_bad_descriptor(tmp_path, monkeypatch, damage):
    path, source, digest = checkpoint(tmp_path)
    archive(tmp_path, path, source, digest)
    descriptor = path / chunks.DESCRIPTOR
    orphan = tmp_path / "runs/.checkpoint_store" / ("0" * 64 + ".gz")
    orphan.write_bytes(b"old")
    os.utime(orphan, (1, 1))
    if damage == "invalid":
        descriptor.write_text("{broken")
    elif damage == "symlink":
        descriptor.unlink()
        descriptor.symlink_to(source)
    else:
        original = chunks.read_descriptor
        def unreadable(root, checkpoint):
            raise PermissionError("unreadable descriptor")
        monkeypatch.setattr(chunks, "read_descriptor", unreadable)
        assert original(tmp_path, path)
    with pytest.raises((ValueError, OSError)):
        chunks.cleanup_orphan_chunks(tmp_path)
    assert orphan.exists()


def test_source_mutation_prevents_descriptor_publication(tmp_path, monkeypatch):
    path, source, digest = checkpoint(tmp_path)
    original = chunks._publish
    def mutate(store, destination, data, reserve):
        original(store, destination, data, reserve)
        source.write_bytes(b"changed source")
    monkeypatch.setattr(chunks, "_publish", mutate)
    with pytest.raises(ValueError, match="changed"):
        archive(tmp_path, path, source, digest)
    assert source.read_bytes() == b"changed source"
    assert not (path / chunks.DESCRIPTOR).exists()


def test_pre_publish_failure_preserves_source_without_descriptor(tmp_path):
    path, source, digest = checkpoint(tmp_path)
    original = source.read_bytes()
    called = []
    def abort():
        called.append(True)
        assert not (path / chunks.DESCRIPTOR).exists()
        raise RuntimeError("checkpoint became protected")
    with pytest.raises(RuntimeError, match="protected"):
        chunks.archive_checkpoint(tmp_path, path, source, digest,
                                  min_free_bytes=0, pre_publish=abort)
    assert called == [True]
    assert source.read_bytes() == original
    assert not (path / chunks.DESCRIPTOR).exists()
    assert list((tmp_path / "runs/.checkpoint_store").glob("*.gz"))


def test_source_identity_rechecked_after_pre_publish_callback(tmp_path):
    path, source, digest = checkpoint(tmp_path)
    def mutate():
        source.write_bytes(b"new checkpoint")
    with pytest.raises(ValueError, match="before descriptor publication"):
        chunks.archive_checkpoint(tmp_path, path, source, digest,
                                  min_free_bytes=0, pre_publish=mutate)
    assert source.read_bytes() == b"new checkpoint"
    assert not (path / chunks.DESCRIPTOR).exists()


def test_gc_removes_only_old_regular_staging_files(tmp_path):
    path, source, digest = checkpoint(tmp_path)
    archive(tmp_path, path, source, digest)
    store = tmp_path / "runs/.checkpoint_store"
    old = store / ".chunk-staging-old"
    new = store / ".chunk-staging-new"
    old.write_bytes(b"old staging")
    new.write_bytes(b"new staging")
    os.utime(old, (1, 1))
    result = chunks.cleanup_orphan_chunks(tmp_path)
    assert result["removed_staging"] == 1
    assert result["removed_chunks"] == 0
    assert result["removed_bytes"] == len(b"old staging")
    assert not old.exists()
    assert new.exists()


def test_gc_staging_symlink_fails_closed(tmp_path):
    path, source, digest = checkpoint(tmp_path)
    archive(tmp_path, path, source, digest)
    store = tmp_path / "runs/.checkpoint_store"
    old = store / ".chunk-staging-old"
    old.write_bytes(b"old staging")
    os.utime(old, (1, 1))
    (store / ".chunk-staging-link").symlink_to(source)
    with pytest.raises(ValueError, match="regular file"):
        chunks.cleanup_orphan_chunks(tmp_path)
    assert old.exists()
    assert source.exists()


def test_existing_descriptor_runs_pre_publish_guard(tmp_path):
    path, source, digest = checkpoint(tmp_path)
    archive(tmp_path, path, source, digest)
    def abort():
        raise RuntimeError("checkpoint became protected")
    with pytest.raises(RuntimeError, match="protected"):
        chunks.archive_checkpoint(tmp_path, path, source, digest,
                                  min_free_bytes=0, pre_publish=abort)
    assert source.exists()


def test_restore_holds_global_lock_during_descriptor_and_chunk_reads(tmp_path, monkeypatch):
    import fcntl
    path, source, digest = checkpoint(tmp_path)
    archive(tmp_path, path, source, digest)
    checked = []
    def guarded(original, *args):
        with (tmp_path / "runs/.checkpoint_store/.lock").open("a+b") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked.append(original.__name__)
        return original(*args)
    descriptor_read, chunk_read = chunks.read_descriptor, chunks._read_chunk
    monkeypatch.setattr(chunks, "read_descriptor", lambda *a: guarded(descriptor_read, *a))
    monkeypatch.setattr(chunks, "_read_chunk", lambda *a: guarded(chunk_read, *a))
    chunks.restore_checkpoint(tmp_path, path, path / "restored.tmp", digest)
    assert "read_descriptor" in checked
    assert "_read_chunk" in checked


@pytest.mark.parametrize("damage", ["dangling", "corrupt", "whole_checksum"])
def test_gc_semantically_corrupt_descriptor_preserves_every_candidate(tmp_path, damage):
    path, source, digest = checkpoint(tmp_path)
    descriptor = archive(tmp_path, path, source, digest)
    store = tmp_path / "runs/.checkpoint_store"
    original_chunk = store / (descriptor["chunks"][0]["sha256"] + ".gz")
    stage = store / ".chunk-staging-old"
    stage.write_bytes(b"old staging")
    orphan = store / ("1" * 64 + ".gz")
    orphan.write_bytes(b"old orphan")
    for candidate in (stage, orphan, original_chunk):
        os.utime(candidate, (1, 1))
    if damage == "whole_checksum":
        descriptor["sha256"] = "0" * 64
    else:
        descriptor["chunks"][0]["sha256"] = "0" * 64
        if damage == "corrupt":
            (store / ("0" * 64 + ".gz")).write_bytes(b"corrupt referenced chunk")
    (path / chunks.DESCRIPTOR).write_text(json.dumps(descriptor))
    with pytest.raises((ValueError, OSError)):
        chunks.cleanup_orphan_chunks(tmp_path)
    assert stage.exists()
    assert orphan.exists()
    assert original_chunk.exists()


@pytest.mark.parametrize("damage", [None, "expected_checksum", "corrupt_chunk"])
def test_verify_checkpoint_requires_no_staging_or_free_space(tmp_path, monkeypatch, damage):
    path, source, digest = checkpoint(tmp_path)
    descriptor = archive(tmp_path, path, source, digest)
    if damage == "corrupt_chunk":
        payload = tmp_path / "runs/.checkpoint_store" / (descriptor["chunks"][0]["sha256"] + ".gz")
        payload.write_bytes(b"corrupt")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    def no_space_check(*args, **kwargs):
        pytest.fail("verification must not need staging or extra free space")
    monkeypatch.setattr(chunks, "_space", no_space_check)
    if damage is None:
        assert chunks.verify_checkpoint(tmp_path, path, digest) == descriptor
    else:
        expected = "0" * 64 if damage == "expected_checksum" else digest
        with pytest.raises((ValueError, OSError)):
            chunks.verify_checkpoint(tmp_path, path, expected)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
