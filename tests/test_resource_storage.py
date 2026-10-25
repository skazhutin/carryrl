"""Immutable tokenizer compaction must preserve all bytes on every failure."""

import fcntl
import shutil
import subprocess
from pathlib import Path

import pytest

from carryrl.training import checkpoint_storage as storage
from carryrl.training import resource_storage as resources


@pytest.fixture
def pair(tmp_path, monkeypatch):
    monkeypatch.setattr(resources.sys, "platform", "darwin")
    paths = []
    for run in ("visual_bc_control", "other"):
        path = tmp_path / "runs" / run / "checkpoints/rollout_0000/language/tokenizer.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"immutable tokenizer")
        paths.append(path)
    return tmp_path, paths[0], paths[1]


def copy_clone(args, **kwargs):
    assert args[:2] == ["/bin/cp", "-c"]
    shutil.copy2(args[2], args[3])
    return subprocess.CompletedProcess(args, 0)


def test_apply_preserves_baseline_and_independent_bytes(pair, monkeypatch):
    root, source, target = pair
    monkeypatch.setattr(resources.subprocess, "run", copy_clone)
    source_inode, target_inode = source.stat().st_ino, target.stat().st_ino
    report = resources.compact_checkpoint_tokenizers(root, apply=True)
    assert report["status"] == "complete"
    assert (report["matched"], report["compacted"], report["logical_bytes"]) == (1, 1, 19)
    assert source.stat().st_ino == source_inode
    assert target.stat().st_ino != target_inode
    assert target.stat().st_ino != source.stat().st_ino
    assert target.read_bytes() == source.read_bytes()
    target.write_bytes(b"changed independently")
    assert source.read_bytes() == b"immutable tokenizer"
    assert not list(root.rglob(".tokenizer-clone-*"))


@pytest.mark.parametrize("failure", ["cp", "hash", "target_changed", "source_changed"])
def test_failures_do_not_replace_original(pair, monkeypatch, failure):
    root, source, target = pair
    inode = target.stat().st_ino

    def failing_clone(args, **kwargs):
        if failure == "cp":
            raise subprocess.CalledProcessError(1, args)
        copy_clone(args, **kwargs)
        if failure == "hash":
            Path(args[3]).write_bytes(b"corrupt clone")
        if failure == "target_changed":
            target.write_bytes(b"concurrent target update")
        if failure == "source_changed":
            source.write_bytes(b"concurrent source update")

    monkeypatch.setattr(resources.subprocess, "run", failing_clone)
    report = resources.compact_checkpoint_tokenizers(root, apply=True)
    assert report["status"] == "failed"
    assert report["compacted"] == report["logical_bytes"] == 0
    assert target.stat().st_ino == inode
    assert target.read_bytes() == (b"concurrent target update" if failure == "target_changed"
                                   else b"immutable tokenizer")
    assert not list(root.rglob(".tokenizer-clone-*"))


def test_dry_run_and_unsupported_never_clone(pair, monkeypatch):
    root, source, target = pair
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **k: pytest.fail("clone called"))
    report = resources.compact_checkpoint_tokenizers(root)
    assert report["matched"] == 1 and report["compacted"] == 0
    monkeypatch.setattr(resources.sys, "platform", "linux")
    assert resources.compact_checkpoint_tokenizers(root, apply=True)["status"] == "unsupported"


def test_symlink_readonly_and_noncheckpoint_are_not_replaced(pair, monkeypatch):
    root, source, target = pair
    target.unlink()
    target.symlink_to(source)
    outside = root / "runs/other/language/tokenizer.json"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(source.read_bytes())
    linked = root / "runs/linked"
    linked.symlink_to(source.parents[3], target_is_directory=True)
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **k: pytest.fail("clone called"))
    report = resources.compact_checkpoint_tokenizers(root, apply=True)
    assert report["compacted"] == 0
    assert target.is_symlink()
    target.unlink()
    target.write_bytes(source.read_bytes())
    target.chmod(0o444)
    assert resources.compact_checkpoint_tokenizers(root, apply=True)["compacted"] == 0


def test_candidate_byte_and_file_budgets(pair, monkeypatch):
    root, _, _ = pair
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **k: pytest.fail("clone called"))
    report = resources.compact_checkpoint_tokenizers(root, apply=True, max_files=1)
    assert report["bounded"] and report["examined"] == 1
    report = resources.compact_checkpoint_tokenizers(root, apply=True, max_bytes=1)
    assert report["examined"] == 0 and report["compacted"] == 0


def test_busy_run_skipped_without_waiting(pair, monkeypatch):
    root, source, target = pair
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **k: pytest.fail("clone called"))
    with (target.parents[3] / ".storage.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = resources.compact_checkpoint_tokenizers(root, apply=True)
    assert report["skipped_busy"] == 1
    assert target.read_bytes() == source.read_bytes()


def test_nonblocking_run_lock_and_reentrancy(tmp_path):
    with (tmp_path / ".storage.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError), storage.run_lock(tmp_path, blocking=False):
            pass
    with storage.run_lock(tmp_path, blocking=False), storage.run_lock(tmp_path, blocking=False):
        pass


def test_explicit_baseline_directory_is_never_replaced(pair, monkeypatch):
    root, _, target = pair
    inode = target.stat().st_ino
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **k: pytest.fail("clone called"))
    report = resources.compact_checkpoint_tokenizers(
        root, apply=True, baseline_directories=(target.parents[3],))
    assert report["matched"] == 1 and report["compacted"] == 0
    assert target.stat().st_ino == inode


def test_clone_symlink_is_rejected(pair, monkeypatch):
    root, source, target = pair
    inode = target.stat().st_ino

    def symlink_clone(args, **kwargs):
        Path(args[3]).symlink_to(source)

    monkeypatch.setattr(resources.subprocess, "run", symlink_clone)
    report = resources.compact_checkpoint_tokenizers(root, apply=True)
    assert report["status"] == "failed" and report["compacted"] == 0
    assert target.stat().st_ino == inode and not target.is_symlink()
