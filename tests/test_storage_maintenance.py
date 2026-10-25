"""Maintenance scheduling must never guess which project data may be archived."""

import fcntl
import json
import subprocess
import sys
from pathlib import Path

import pytest

from carryrl.training import storage_maintenance as maintenance


@pytest.fixture
def project(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "carryrl"\n')
    for directory in ("src/carryrl", ".cache", "runs/current", "runs/baseline"):
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "runs/current/manifest.json").write_text("{}")
    policy = {"version": 1, "enabled": True,
              "hot_manifests": ["runs/current/manifest.json"],
              "baseline_directories": ["runs/baseline"]}
    (tmp_path / maintenance.POLICY).write_text(json.dumps(policy))
    return tmp_path


def update_policy(project, **updates):
    path = project / maintenance.POLICY
    policy = json.loads(path.read_text())
    policy.update(updates)
    path.write_text(json.dumps(policy))


def test_launcher_detaches_current_interpreter(project, monkeypatch):
    calls = []
    monkeypatch.setattr(maintenance.subprocess, "Popen", lambda *args, **kwargs:
                        calls.append((args, kwargs)))
    assert maintenance.maybe_launch(project)
    args, options = calls[0]
    assert args[0] == [sys.executable, "-m", "carryrl.training.storage_maintenance",
                       "--root", str(project)]
    assert options["start_new_session"] is True
    assert options["cwd"] == project
    assert options["stdin"] == subprocess.DEVNULL
    assert options["stdout"] == subprocess.DEVNULL
    assert options["stderr"] == subprocess.DEVNULL


@pytest.mark.parametrize("updates", [
    {"enabled": False}, {"version": True}, {"hot_manifests": []},
    {"baseline_directories": []}, {"hot_manifests": ["../outside.json"]},
    {"hot_manifests": ["runs/missing.json"]}, {"max_checkpoints": 9},
    {"max_checkpoints": True}, {"timeout_seconds": 301}, {"disk_floor_gib": 1},
    {"cooldown_seconds": 0}, {"unknown": "ambiguous"},
])
def test_invalid_policy_never_launches(project, monkeypatch, updates):
    update_policy(project, **updates)
    monkeypatch.setattr(maintenance.subprocess, "Popen", lambda *a, **kw:
                        pytest.fail("unsafe policy launched a worker"))
    assert not maintenance.maybe_launch(project)
    assert not (project / maintenance.STATUS).exists()


def test_missing_policy_or_wrong_project_never_launches(project):
    (project / maintenance.POLICY).unlink()
    assert not maintenance.maybe_launch(project)
    (project / "pyproject.toml").write_text('[project]\nname = "another"\n')
    assert not maintenance.maybe_launch(project)


@pytest.mark.parametrize("name", [maintenance.STATUS, maintenance.LOCK, maintenance.LOG,
                                 maintenance.STATUS + ".tmp"])
def test_symlink_state_never_launches(project, name):
    target = project / "runs/baseline/precious.json"
    target.write_text("preserve me")
    (project / name).symlink_to(target)
    assert not maintenance.maybe_launch(project)
    assert target.read_text() == "preserve me"


def test_recent_or_corrupt_status_blocks_launch(project, monkeypatch):
    monkeypatch.setattr(maintenance.time, "time", lambda: 100000)
    monkeypatch.setattr(maintenance.subprocess, "Popen", lambda *a, **kw:
                        pytest.fail("cooldown bypassed"))
    (project / maintenance.STATUS).write_text('{"started_at": 99999}')
    assert not maintenance.maybe_launch(project)
    (project / maintenance.STATUS).write_text("{broken")
    assert not maintenance.maybe_launch(project)


def test_lock_prevents_concurrent_housekeeping(project, monkeypatch):
    monkeypatch.setattr(maintenance, "_run_checkpoint_worker", lambda *a:
                        pytest.fail("concurrent archive worker started"))
    with (project / maintenance.LOCK).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert maintenance.run_maintenance(project) == {"status": "locked"}
    assert not (project / maintenance.STATUS).exists()


def test_worker_command_preserves_current_and_baseline(project):
    policy = maintenance.load_policy(project)
    command = maintenance._command(project, policy)
    assert command[:6] == [sys.executable, "-m", "carryrl.training.checkpoint_storage",
                           "--root", str(project), "housekeep"]
    assert command[command.index("--hot-reference") + 1] == str(
        project / "runs/current/manifest.json")
    assert command[command.index("--keep") + 1] == str(project / "runs/baseline")
    assert command[command.index("--min-free-bytes") + 1] == str(2 * 1024**3)
    assert command[command.index("--max-checkpoints") + 1] == "8"
    assert command[-1] == "--apply"


def test_maintenance_records_result_without_touching_checkpoint(project, monkeypatch):
    checkpoint = project / "runs/baseline/policy.pt"
    checkpoint.write_bytes(b"unique checkpoint bytes")
    called = []
    monkeypatch.setattr(maintenance, "_run_checkpoint_worker", lambda *args:
                        called.append(args) or {"returncode": 0, "timed_out": False})
    status = maintenance.run_maintenance(project)
    assert status["status"] == "complete"
    assert status["features"]["all_packs_retained"]
    assert checkpoint.read_bytes() == b"unique checkpoint bytes"
    assert maintenance.run_maintenance(project) == {"status": "cooldown"}
    assert len(called) == 1
    assert json.loads((project / maintenance.STATUS).read_text()) == status


def test_feature_inventory_keeps_referenced_unknown_and_broken_packs(project):
    cache = project / ".cache/frozen_features"
    known, unknown = cache / "known", cache / "unknown"
    for pack in (known, unknown):
        pack.mkdir(parents=True)
        (pack / "features.npy").write_bytes(b"unique features")
    feature_manifest = project / "runs/current/feature_manifest.json"
    feature_manifest.write_text(json.dumps({"feature_reuse": {
        "packs": [{"directory": str(known)}]}}))
    (project / "runs/baseline/feature_manifest.json").write_text("{broken")
    inventory = maintenance.feature_inventory(project)
    records = {Path(record["directory"]).name: record for record in inventory["packs"]}
    assert records["known"]["references"] == ["runs/current/feature_manifest.json"]
    assert records["unknown"]["references"] == []
    assert inventory["all_packs_retained"] and not inventory["orphan_proof"]
    assert inventory["errors"]
    assert (known / "features.npy").read_bytes() == b"unique features"
    assert (unknown / "features.npy").read_bytes() == b"unique features"


def test_failed_worker_visible_in_status(project, monkeypatch):
    def fail(*args):
        raise OSError("disk unavailable")
    monkeypatch.setattr(maintenance, "_run_checkpoint_worker", fail)
    status = maintenance.run_maintenance(project)
    assert status["status"] == "failed"
    assert status["error"] == "disk unavailable"
    assert "finished_at" in status


def test_timeout_terminates_process_group_and_reports_failure(project, monkeypatch):
    calls = []

    class Child:
        pid = 43123
        returncode = -15
        waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("archive", timeout)
            return self.returncode

    monkeypatch.setattr(maintenance.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(maintenance.os, "killpg", lambda *args: calls.append(args))
    status = maintenance.run_maintenance(project)
    assert status["status"] == "timed_out"
    assert calls == [(43123, maintenance.signal.SIGTERM)]
    assert (project / maintenance.LOG).exists()


def test_resource_failure_stops_archive_worker(project, monkeypatch):
    monkeypatch.setattr(maintenance, "compact_checkpoint_tokenizers", lambda *a, **k:
                        {"status": "failed", "compacted": 0, "errors": ["clone failed"]})
    monkeypatch.setattr(maintenance, "_run_checkpoint_worker", lambda *a:
                        pytest.fail("archive ran after resource failure"))
    status = maintenance.run_maintenance(project)
    assert status["status"] == "failed"
    assert status["resources"]["errors"] == ["clone failed"]
