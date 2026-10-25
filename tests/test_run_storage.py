"""Cold checkpoint storage preserves checkpoint identity and protects live inputs."""

import gzip
import hashlib
import io
import json
from pathlib import Path

import pytest

from carryrl.training import checkpoint_storage as storage


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


def hot_manifest(root, value=None):
    source = root / "runs" / "current_pair" / "manifest.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(value or {"status": "prepared_not_executed"}))
    return source


def mock_processes(monkeypatch, output=""):
    from types import SimpleNamespace

    monkeypatch.setattr(
        storage.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )


def mock_free(monkeypatch, free=100 * 1024**3):
    from types import SimpleNamespace

    monkeypatch.setattr(storage.shutil, "disk_usage", lambda *_: SimpleNamespace(free=free))


def test_housekeep_archives_historical_references_and_latest(tmp_path, monkeypatch):
    checkpoint, data = fixture(tmp_path, "historical_training")
    (checkpoint.parents[1] / "latest.json").write_text(json.dumps({"path": str(checkpoint)}))
    (checkpoint.parents[1] / "launch.json").write_text(json.dumps({"warm_start": str(checkpoint)}))
    reference = hot_manifest(tmp_path)
    mock_processes(monkeypatch)
    mock_free(monkeypatch)
    options = {"hot_references": [reference], "min_age_seconds": 0}
    dry = storage.housekeep(tmp_path, **options)
    assert dry["selected_count"] == 1
    assert not dry["checkpoints"][0]["applied"]
    assert (checkpoint / "state.pt").read_bytes() == data
    applied = storage.housekeep(tmp_path, apply=True, **options)
    assert applied["checkpoints"][0]["applied"]
    assert not (checkpoint / "state.pt").exists()
    assert storage.ensure_checkpoint_payload(checkpoint).read_bytes() == data


def test_large_prepared_manifest_and_relative_inputs_pin_dependencies(tmp_path, monkeypatch):
    baseline, _ = fixture(tmp_path, "baseline")
    cold, _ = fixture(tmp_path, "old_training")
    child = tmp_path / "runs" / "current_pair" / "arm" / "manifest.json"
    child.parent.mkdir(parents=True)
    child.write_text(json.dumps({"source_checkpoint": "../../baseline/checkpoints/rollout_0000"}))
    reference = hot_manifest(tmp_path, {
        "status": "prepared_not_executed", "padding": "x" * (5 * storage.CHUNK_BYTES),
        "input_files": [{"path": "arm/manifest.json"}],
    })
    mock_processes(monkeypatch)
    pins = storage.HousekeepingPins(tmp_path, [reference])()
    assert storage.protection_reasons(baseline, pins)
    assert not storage.protection_reasons(cold, pins)
    plan = storage.housekeep(tmp_path, hot_references=[reference], min_age_seconds=0)
    assert [row["checkpoint"] for row in plan["checkpoints"]] == [str(cold)]


def test_housekeep_active_config_pins_and_baseline_stay_hot(tmp_path, monkeypatch):
    active, _ = fixture(tmp_path, "active")
    configured, _ = fixture(tmp_path, "configured")
    baseline, _ = fixture(tmp_path, "visual_bc_control_original")
    live, _ = fixture(tmp_path, "live")
    status, _ = fixture(tmp_path, "status")
    cache = tmp_path / ".cache"
    cache.mkdir()
    (cache / "active_mlx_launch.json").write_text(json.dumps({"run": str(active.parents[1])}))
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "current.json").write_text(json.dumps({"weights": str(configured / "state.pt")}))
    (status.parents[1] / "status.json").write_text('{"status":"training"}')
    mock_processes(monkeypatch, f"12 python scripts/train_imitation.py --run {live.parents[1]}\n")
    pins = storage.HousekeepingPins(tmp_path, [hot_manifest(tmp_path)])()
    for checkpoint in (active, configured, baseline, live, status):
        assert storage.protection_reasons(checkpoint, pins)


@pytest.mark.parametrize("invalid", ["missing", "broken", "scalar"])
def test_housekeep_invalid_hot_reference_fails_closed(tmp_path, monkeypatch, invalid):
    checkpoint, _ = fixture(tmp_path)
    reference = hot_manifest(tmp_path)
    if invalid == "missing":
        reference.unlink()
    else:
        reference.write_text("broken" if invalid == "broken" else '"scalar"')
    mock_processes(monkeypatch)
    plan = storage.housekeep(tmp_path, hot_references=[reference], min_age_seconds=0)
    assert plan["selected_count"] == 0
    assert storage.protection_reasons(checkpoint, storage.HousekeepingPins(tmp_path, [reference])())


def test_housekeep_rechecks_changed_pin_before_removing_source(tmp_path, monkeypatch):
    checkpoint, data = fixture(tmp_path)
    reference = hot_manifest(tmp_path)
    mock_processes(monkeypatch)
    mock_free(monkeypatch)
    archive = storage.chunks.archive_checkpoint

    def pin_during_compression(*args, **kwargs):
        reference.write_text(json.dumps({"source_checkpoint": str(checkpoint)}))
        return archive(*args, **kwargs)

    monkeypatch.setattr(storage.chunks, "archive_checkpoint", pin_during_compression)
    result = storage.housekeep(tmp_path, hot_references=[reference], apply=True, min_age_seconds=0)
    assert "protected during chunk archiving" in result["checkpoints"][0]["skipped"]
    assert (checkpoint / "state.pt").read_bytes() == data
    assert not (checkpoint / "state.pt.gz").exists()


def test_housekeep_disk_floor_and_batch_limits(tmp_path, monkeypatch):
    checkpoint, data = fixture(tmp_path)
    other, _ = fixture(tmp_path, "other_smoke")
    reference = hot_manifest(tmp_path)
    mock_processes(monkeypatch)
    floor = 8 * 1024**3
    mock_free(monkeypatch, floor + 2 * len(data) - 1)
    result = storage.housekeep(tmp_path, hot_references=[reference], apply=True,
                               max_checkpoints=1, min_age_seconds=0, min_free_bytes=floor)
    assert result["selected_count"] == 1
    assert "insufficient free disk" in result["checkpoints"][0]["skipped"]
    assert (checkpoint / "state.pt").exists() and (other / "state.pt").exists()
    assert not list(tmp_path.rglob(".storage-*"))
    with pytest.raises(ValueError, match="2 GiB"):
        storage.housekeep(tmp_path, hot_references=[reference], min_free_bytes=1)
    with pytest.raises(ValueError, match="hot-reference"):
        storage.housekeep(tmp_path, hot_references=[])


def test_housekeep_reports_initial_chunk_store_cost_honestly(tmp_path, monkeypatch):
    import os

    checkpoint, _ = fixture(tmp_path)
    data = os.urandom(64)
    (checkpoint / "state.pt").write_bytes(data)
    (checkpoint / "manifest.json").write_text(
        json.dumps({"sha256": hashlib.sha256(data).hexdigest()})
    )
    mock_processes(monkeypatch)
    mock_free(monkeypatch)
    result = storage.housekeep(tmp_path, hot_references=[hot_manifest(tmp_path)],
                               apply=True, min_age_seconds=0, min_free_bytes=8 * 1024**3)
    assert result["checkpoints"][0]["saved_bytes"] < 0
    assert storage.ensure_checkpoint_payload(checkpoint).read_bytes() == data


def test_transparent_restore_reserves_uncompressed_staging_space(tmp_path, monkeypatch):
    checkpoint, data = fixture(tmp_path)
    storage.transform(tmp_path, checkpoint, apply=True)
    mock_free(monkeypatch, 8 * 1024**3 + 2 * len(data) - 1)
    with pytest.raises(OSError, match="insufficient free disk"):
        storage.ensure_checkpoint_payload(checkpoint)
    assert not (checkpoint / "state.pt").exists()
    assert (checkpoint / "state.pt.gz").exists()


def test_housekeep_process_inspection_failure_fails_closed(tmp_path, monkeypatch):
    checkpoint, _ = fixture(tmp_path)

    def denied(*args, **kwargs):
        raise OSError("process list unavailable")

    monkeypatch.setattr(storage.subprocess, "run", denied)
    pins = storage.HousekeepingPins(tmp_path, [hot_manifest(tmp_path)])()
    assert "cannot inspect live training processes" in storage.protection_reasons(checkpoint, pins)


def test_housekeep_unresolved_active_registry_fails_closed(tmp_path, monkeypatch):
    checkpoint, _ = fixture(tmp_path)
    cache = tmp_path / ".cache"
    cache.mkdir()
    (cache / "active_run.json").write_text('{"run":"runs/missing_training_run"}')
    mock_processes(monkeypatch)
    pins = storage.HousekeepingPins(tmp_path, [hot_manifest(tmp_path)])()
    assert storage.protection_reasons(checkpoint, pins)


def test_housekeep_rechecks_final_pin_after_metadata_publication(tmp_path, monkeypatch):
    checkpoint, data = fixture(tmp_path)
    reference = hot_manifest(tmp_path)
    mock_processes(monkeypatch)
    mock_free(monkeypatch)
    record = storage.record

    def pin_after_record(*args, **kwargs):
        record(*args, **kwargs)
        reference.write_text(json.dumps({"source_checkpoint": str(checkpoint)}))

    monkeypatch.setattr(storage, "record", pin_after_record)
    result = storage.housekeep(tmp_path, hot_references=[reference], apply=True, min_age_seconds=0)
    assert "protected during chunk archiving" in result["checkpoints"][0]["skipped"]
    assert (checkpoint / "state.pt").read_bytes() == data
    assert not (checkpoint / storage.chunks.DESCRIPTOR).exists()
    index = json.loads((checkpoint.parents[1] / "storage_manifest.json").read_text())
    assert index["checkpoints"]["checkpoints/rollout_0000"]["state"] == "hot"


def test_housekeep_starts_with_smaller_payload_that_fits_reserve(tmp_path, monkeypatch):
    large, _ = fixture(tmp_path, "large_smoke")
    small, _ = fixture(tmp_path, "small_smoke")
    data = b"small" * 100
    (small / "state.pt").write_bytes(data)
    (small / "manifest.json").write_text(
        json.dumps({"sha256": hashlib.sha256(data).hexdigest()})
    )
    mock_processes(monkeypatch)
    mock_free(monkeypatch, 8 * 1024**3 + 2 * len(data))
    result = storage.housekeep(tmp_path, hot_references=[hot_manifest(tmp_path)],
                               apply=True, min_age_seconds=0, min_free_bytes=8 * 1024**3)
    assert result["checkpoints"][0]["checkpoint"] == str(small)
    assert result["checkpoints"][0]["applied"]
    assert result["archived_count"] == 1
    assert "insufficient free disk" in result["checkpoints"][1]["skipped"]
    assert (large / "state.pt").exists()


def test_housekeep_protects_nested_running_arm_and_live_evaluation(tmp_path, monkeypatch):
    branch = tmp_path / "runs" / "paired" / "anchor_on"
    checkpoint = branch / "checkpoints" / "epoch_0001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"weights")
    (branch / "status.json").write_text('{"status":"training"}')
    evaluating, _ = fixture(tmp_path, "evaluating")
    mock_processes(
        monkeypatch, f"12 python scripts/evaluate_imitation.py --checkpoint {evaluating}\n",
    )
    pins = storage.HousekeepingPins(tmp_path, [hot_manifest(tmp_path)])()
    assert storage.protection_reasons(checkpoint, pins)
    assert storage.protection_reasons(evaluating, pins)


@pytest.mark.parametrize("corrupt", [False, True])
def test_concurrent_restore_accepts_only_verified_already_restored_payload(
    tmp_path, monkeypatch, corrupt,
):
    import contextlib

    checkpoint, data = fixture(tmp_path)
    storage.transform(tmp_path, checkpoint, apply=True)
    mock_free(monkeypatch)
    original_lock = storage.run_lock
    first = True

    @contextlib.contextmanager
    def interleaved_lock(run):
        nonlocal first
        if first:
            first = False
            storage.ensure_checkpoint_payload(checkpoint)
            if corrupt:
                (checkpoint / "state.pt").write_bytes(b"corrupt concurrent restore")
        with original_lock(run):
            yield

    monkeypatch.setattr(storage, "run_lock", interleaved_lock)
    if corrupt:
        with pytest.raises(ValueError, match="already restored checkpoint hash mismatch"):
            storage.ensure_checkpoint_payload(checkpoint)
    else:
        assert storage.ensure_checkpoint_payload(checkpoint).read_bytes() == data
    assert not (checkpoint / "state.pt.gz").exists()


def test_checkpoint_reader_holds_same_lock_as_archive_writer(tmp_path):
    checkpoint, data = fixture(tmp_path)
    with storage.checkpoint_read_lock(checkpoint) as state:
        assert state.read_bytes() == data
        with (checkpoint.parents[1] / ".storage.lock").open("a") as other:
            with pytest.raises(BlockingIOError):
                storage.fcntl.flock(other, storage.fcntl.LOCK_EX | storage.fcntl.LOCK_NB)


def test_sibling_checkpoint_readers_reenter_without_releasing_outer_lock(tmp_path):
    import subprocess
    import sys

    first, _ = fixture(tmp_path, "paired/anchor_off")
    second, _ = fixture(tmp_path, "paired/anchor_on")
    # A subprocess timeout makes the original self-deadlock a bounded failure.
    code = """
import fcntl
import sys
from pathlib import Path
from carryrl.training.checkpoint_storage import checkpoint_read_lock
first, second = map(Path, sys.argv[1:])
lock = first.parents[2] / '.storage.lock'
with checkpoint_read_lock(first):
    try:
        with checkpoint_read_lock(second):
            raise ValueError('inner reader failed')
    except ValueError:
        pass
    with lock.open('a') as other:
        try:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise AssertionError('inner exit released outer lock')
with lock.open('a') as other:
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
"""
    subprocess.run([sys.executable, "-c", code, str(first), str(second)],
                   check=True, capture_output=True, text=True, timeout=10)


def test_run_lock_does_not_treat_another_thread_as_reentrant(tmp_path):
    import threading

    attempting = threading.Event()
    acquired = threading.Event()
    failures = []

    def reader():
        try:
            attempting.set()
            with storage.run_lock(tmp_path):
                acquired.set()
        except BaseException as error:
            failures.append(error)

    with storage.run_lock(tmp_path):
        worker = threading.Thread(target=reader, daemon=True)
        worker.start()
        assert attempting.wait(5)
        assert not acquired.wait(0.1)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert acquired.is_set() and not failures


def test_checkpoint_reader_retries_archive_between_ensure_and_lock(tmp_path, monkeypatch):
    checkpoint, data = fixture(tmp_path)
    original = storage.ensure_checkpoint_payload
    first = True
    mock_free(monkeypatch)

    def ensure_then_archive(path):
        nonlocal first
        state = original(path)
        if first:
            first = False
            storage.transform(tmp_path, checkpoint, apply=True)
        return state

    monkeypatch.setattr(storage, "ensure_checkpoint_payload", ensure_then_archive)
    with storage.checkpoint_read_lock(checkpoint) as state:
        assert state.read_bytes() == data
    assert not (checkpoint / "state.pt.gz").exists()


def test_housekeep_explicit_checkpoint_manifest_itself_is_hot(tmp_path, monkeypatch):
    checkpoint, _ = fixture(tmp_path)
    mock_processes(monkeypatch)
    reference = hot_manifest(tmp_path, {"input_files": [{"path": "child.json"}]})
    (reference.parent / "child.json").write_text(
        json.dumps({"source_checkpoint": str(checkpoint / "manifest.json")})
    )
    pins = storage.HousekeepingPins(tmp_path, [reference])()
    assert storage.protection_reasons(checkpoint, pins)


@pytest.mark.parametrize("legacy_gzip", [False, True])
def test_housekeep_chunk_roundtrip_preserves_original_bytes_and_file_list(
    tmp_path, monkeypatch, legacy_gzip,
):
    checkpoint, data = fixture(tmp_path)
    original_files = {path.name: path.read_bytes() for path in checkpoint.iterdir()}
    if legacy_gzip:
        storage.transform(tmp_path, checkpoint, apply=True)
    mock_processes(monkeypatch)
    mock_free(monkeypatch)
    result = storage.housekeep(tmp_path, hot_references=[hot_manifest(tmp_path)],
                               apply=True, min_age_seconds=0)
    assert result["archived_count"] == 1
    descriptor = storage.chunks.read_descriptor(tmp_path, checkpoint)
    assert descriptor["sha256"] == hashlib.sha256(data).hexdigest()
    assert descriptor["bytes"] == len(data)
    assert not (checkpoint / "state.pt").exists()
    assert not (checkpoint / "state.pt.gz").exists()
    index = json.loads((checkpoint.parents[1] / "storage_manifest.json").read_text())
    entry = index["checkpoints"]["checkpoints/rollout_0000"]
    assert entry["state"] == "chunked" and entry["requires_restore"]
    assert entry["payload"] == storage.chunks.DESCRIPTOR
    inventory = storage.inventory(tmp_path)
    assert inventory["checkpoints"][0]["state"] == "chunked"
    assert inventory["chunk_store_bytes"] > 0
    assert storage.ensure_checkpoint_payload(checkpoint).read_bytes() == data
    assert {path.name: path.read_bytes() for path in checkpoint.iterdir()} == original_files


def test_corrupted_chunk_preserves_descriptor_and_never_publishes_plain(tmp_path, monkeypatch):
    checkpoint, _ = fixture(tmp_path)
    mock_processes(monkeypatch)
    mock_free(monkeypatch)
    storage.housekeep(tmp_path, hot_references=[hot_manifest(tmp_path)],
                      apply=True, min_age_seconds=0)
    descriptor_path = checkpoint / storage.chunks.DESCRIPTOR
    before = descriptor_path.read_bytes()
    descriptor = storage.chunks.read_descriptor(tmp_path, checkpoint)
    chunk = tmp_path / "runs" / ".checkpoint_store" / (descriptor["chunks"][0]["sha256"] + ".gz")
    chunk.write_bytes(gzip.compress(b"corrupted chunk"))
    with pytest.raises(ValueError, match="chunk size mismatch"):
        storage.ensure_checkpoint_payload(checkpoint)
    assert descriptor_path.read_bytes() == before
    assert not (checkpoint / "state.pt").exists()
    assert not list(checkpoint.glob(".storage-*"))


def test_interrupted_plain_and_chunk_descriptor_reconciles_before_input_pinning(
    tmp_path, monkeypatch,
):
    checkpoint, data = fixture(tmp_path)
    original_files = {path.name for path in checkpoint.iterdir()}
    mock_processes(monkeypatch)
    mock_free(monkeypatch)
    record = storage.record

    def fail_chunk_record(*args, **kwargs):
        if args[3] == "chunked":
            raise OSError("interrupted chunk metadata publication")
        return record(*args, **kwargs)

    monkeypatch.setattr(storage, "record", fail_chunk_record)
    result = storage.housekeep(tmp_path, hot_references=[hot_manifest(tmp_path)],
                               apply=True, min_age_seconds=0)
    assert "interrupted chunk metadata" in result["checkpoints"][0]["skipped"]
    assert (checkpoint / "state.pt").read_bytes() == data
    assert (checkpoint / storage.chunks.DESCRIPTOR).exists()
    monkeypatch.setattr(storage, "record", record)
    assert storage.ensure_checkpoint_payload(checkpoint).read_bytes() == data
    assert {path.name for path in checkpoint.iterdir()} == original_files


def test_chunk_archive_legacy_gzip_disk_guard_uses_uncompressed_size(tmp_path, monkeypatch):
    checkpoint, data = fixture(tmp_path)
    storage.transform(tmp_path, checkpoint, apply=True)
    mock_processes(monkeypatch)
    mock_free(monkeypatch, storage.HOUSEKEEP_MIN_FREE_BYTES + 2 * len(data) - 1)
    result = storage.housekeep(tmp_path, hot_references=[hot_manifest(tmp_path)],
                               apply=True, min_age_seconds=0)
    assert "insufficient free disk" in result["checkpoints"][0]["skipped"]
    assert gzip.decompress((checkpoint / "state.pt.gz").read_bytes()) == data
    assert not (checkpoint / storage.chunks.DESCRIPTOR).exists()
    assert not list(checkpoint.glob(".storage-*"))


def test_chunk_housekeep_does_not_convert_hot_legacy_archive(tmp_path, monkeypatch):
    checkpoint, data = fixture(tmp_path)
    storage.transform(tmp_path, checkpoint, apply=True)
    mock_processes(monkeypatch)
    result = storage.housekeep(
        tmp_path, hot_references=[hot_manifest(tmp_path, {"source_checkpoint": str(checkpoint)})],
        apply=True, min_age_seconds=0,
    )
    assert result["selected_count"] == 0
    assert gzip.decompress((checkpoint / "state.pt.gz").read_bytes()) == data
    assert not (checkpoint / storage.chunks.DESCRIPTOR).exists()


def test_chunk_housekeep_shares_identical_checkpoint_chunks(tmp_path, monkeypatch):
    first, _ = fixture(tmp_path, "first_smoke")
    second, _ = fixture(tmp_path, "second_smoke")
    mock_processes(monkeypatch)
    mock_free(monkeypatch)
    result = storage.housekeep(tmp_path, hot_references=[hot_manifest(tmp_path)],
                               apply=True, min_age_seconds=0)
    assert result["archived_count"] == 2
    assert len(list((tmp_path / "runs" / ".checkpoint_store").glob("*.gz"))) == 1
    assert storage.chunks.read_descriptor(tmp_path, first) == storage.chunks.read_descriptor(
        tmp_path, second,
    )


def test_chunk_descriptor_survives_late_pin_and_sync_error_after_source_removal(
    tmp_path, monkeypatch,
):
    checkpoint, data = fixture(tmp_path)
    reference = hot_manifest(tmp_path)
    mock_processes(monkeypatch)
    mock_free(monkeypatch)
    sync = storage.sync_directory

    def fail_after_source_removal(path):
        if (path == checkpoint and not (checkpoint / "state.pt").exists()
                and (checkpoint / storage.chunks.DESCRIPTOR).exists()):
            reference.write_text(json.dumps({"source_checkpoint": str(checkpoint)}))
            raise OSError("late directory sync error")
        sync(path)

    monkeypatch.setattr(storage, "sync_directory", fail_after_source_removal)
    result = storage.housekeep(tmp_path, hot_references=[reference], apply=True, min_age_seconds=0)
    assert "late directory sync error" in result["checkpoints"][0]["skipped"]
    assert (checkpoint / storage.chunks.DESCRIPTOR).exists()
    assert not (checkpoint / "state.pt").exists()
    monkeypatch.setattr(storage, "sync_directory", sync)
    assert storage.ensure_checkpoint_payload(checkpoint).read_bytes() == data


def test_chunk_dual_state_reconciliation_does_not_need_restore_staging_space(tmp_path, monkeypatch):
    checkpoint, data = fixture(tmp_path)
    original_files = {path.name for path in checkpoint.iterdir()}
    storage.chunks.archive_checkpoint(
        tmp_path, checkpoint, checkpoint / "state.pt", hashlib.sha256(data).hexdigest(),
    )
    mock_free(monkeypatch, 1)
    assert storage.ensure_checkpoint_payload(checkpoint).read_bytes() == data
    assert {path.name for path in checkpoint.iterdir()} == original_files
