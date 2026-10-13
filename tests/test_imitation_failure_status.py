"""Failure status precedes recovery, and recovery cannot replace the original error."""

import json

import pytest
from test_imitation_feature_storage import script

from carryrl.training.progress import TrainingProgress


def _run_failure(error, *, progress, sync=None, save=None, close_writer=None):
    try:
        raise error
    except BaseException as original:
        script._record_training_failure(
            original, progress=progress, sync_to_torch=sync, save_checkpoint=save,
            close_writer=close_writer)
        raise


def test_primary_checkpoint_failure_records_failed_status_without_retry(tmp_path):
    progress = TrainingProgress(tmp_path / "status.json", epochs_total=1)
    progress.phase("saving")
    progress.update(force=True, optimizer_steps=266)
    attempts = []
    error = OSError("No space left on device")

    def save():
        attempts.append("save")
        raise error

    def sync():
        pytest.fail("primary saving failure must not trigger recovery synchronization")

    with pytest.raises(OSError) as raised:
        try:
            save()
        except BaseException as original:
            script._record_training_failure(
                original, progress=progress, sync_to_torch=sync, save_checkpoint=save)
            raise
    assert raised.value is error and attempts == ["save"]
    persisted = json.loads(progress.path.read_text())
    assert persisted["status"] == "failed" and persisted["phase"] == "saving"
    assert persisted["optimizer_steps"] == 266
    assert persisted["error_type"] == "OSError" and persisted["error"] == str(error)
    assert "recovery_checkpoint" not in persisted


@pytest.mark.parametrize("failed_stage", ["sync_to_torch", "checkpoint"])
def test_recovery_error_is_separate_and_original_failure_survives(tmp_path, failed_stage):
    progress = TrainingProgress(tmp_path / "status.json", epochs_total=1)
    progress.phase("imitation")
    original = RuntimeError("original training error")
    attempts = []

    def attempt(stage):
        persisted = json.loads(progress.path.read_text())
        assert persisted["status"] == "failed" and persisted["error"] == str(original)
        attempts.append(stage)
        if stage == failed_stage:
            raise OSError("recovery failed")

    with pytest.raises(RuntimeError) as raised:
        _run_failure(original, progress=progress,
                     sync=lambda: attempt("sync_to_torch"), save=lambda: attempt("checkpoint"))
    assert raised.value is original
    assert attempts == (["sync_to_torch"] if failed_stage == "sync_to_torch"
                        else ["sync_to_torch", "checkpoint"])
    persisted = json.loads(progress.path.read_text())
    assert persisted["status"] == "failed" and persisted["error"] == str(original)
    assert persisted["recovery_errors"] == [
        {"stage": failed_stage, "error_type": "OSError", "error": "recovery failed"}]
    assert "recovery_checkpoint" not in persisted


def test_interruption_persists_status_then_retains_successful_recovery_snapshot(tmp_path):
    progress = TrainingProgress(tmp_path / "status.json", epochs_total=1)
    progress.phase("imitation")
    progress.update(force=True, optimizer_steps=9)
    checkpoint = tmp_path / "recovered"
    attempts = []

    def sync():
        assert json.loads(progress.path.read_text())["status"] == "interrupted"
        attempts.append("sync")

    def save():
        assert json.loads(progress.path.read_text())["status"] == "interrupted"
        attempts.append("save")
        checkpoint.mkdir()
        return checkpoint

    original = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as raised:
        _run_failure(original, progress=progress, sync=sync, save=save)
    assert raised.value is original and attempts == ["sync", "save"]
    persisted = json.loads(progress.path.read_text())
    assert persisted["status"] == "interrupted" and persisted["optimizer_steps"] == 9
    assert persisted["error_type"] == "KeyboardInterrupt"
    assert persisted["recovery_checkpoint"] == str(checkpoint) and checkpoint.is_dir()
    assert persisted["recovery_errors"] == []


def test_unwritable_status_cannot_replace_original_exception(tmp_path, monkeypatch):
    progress = TrainingProgress(tmp_path / "status.json", epochs_total=1)
    original = RuntimeError("original")

    def fail_write(**kwargs):
        raise OSError("status filesystem full")

    monkeypatch.setattr(progress, "write", fail_write)
    with pytest.raises(RuntimeError) as raised:
        _run_failure(original, progress=progress)
    assert raised.value is original and progress.state["status"] == "failed"
    assert [row["stage"] for row in progress.state["recovery_errors"]] == [
        "failure_status", "recovery_status"]


def test_writer_cleanup_failure_cannot_replace_original_exception(tmp_path):
    progress = TrainingProgress(tmp_path / "status.json", epochs_total=1)
    original = RuntimeError("original")

    def close_writer():
        assert json.loads(progress.path.read_text())["status"] == "failed"
        raise OSError("writer flush failed")

    with pytest.raises(RuntimeError) as raised:
        _run_failure(original, progress=progress, close_writer=close_writer)
    assert raised.value is original
    persisted = json.loads(progress.path.read_text())
    assert persisted["recovery_errors"] == [
        {"stage": "writer_close", "error_type": "OSError", "error": "writer flush failed"}]
