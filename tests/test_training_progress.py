import json
from datetime import datetime

import pytest

from carryrl.training.progress import TrainingProgress


def test_progress_observed_work_heartbeat_phase_timings_and_exact_interrupt_counters(
    tmp_path, monkeypatch,
):
    now = [10.0]
    monkeypatch.setattr("carryrl.training.progress.time.monotonic", lambda: now[0])
    path = tmp_path / "status.json"
    progress = TrainingProgress(path, epochs_total=80)
    progress.phase("encoding", total=128, unit="frames")
    now[0] = 12.1
    progress.update(encoded_frames=16, phase_units_done=16)
    observed = json.loads(path.read_text())
    assert observed["encoded_frames"] == 16 and observed["phase_units_done"] == 16
    assert observed["optimizer_steps"] == 0 and observed["epochs_completed"] == 0
    assert observed["elapsed_seconds"] == observed["wall_seconds"] == pytest.approx(2.1)
    assert datetime.fromisoformat(observed["updated_at"]).utcoffset().total_seconds() == 0
    now[0] = 15
    progress.phase("imitation", total=160, unit="optimizer_steps", batches_per_epoch=2)
    assert progress.state["phase_history"][-1]["elapsed_seconds"] == 5
    progress.update(epoch=1, batch=0, force=True)
    for step in range(1, 9):
        progress.update(optimizer_update=True, epoch=step // 2 + 1, optimizer_steps=step,
                        phase_units_done=step)
    assert json.loads(path.read_text())["optimizer_steps"] == 8
    progress.update(optimizer_update=True, optimizer_steps=9, epoch=5)
    assert progress.state["optimizer_steps"] == 9
    progress.update(force=True, status="interrupted", epochs_completed=4)
    observed = json.loads(path.read_text())
    assert observed["optimizer_steps"] == 9 and observed["epochs_completed"] == 4
    assert observed["epoch"] == 5
