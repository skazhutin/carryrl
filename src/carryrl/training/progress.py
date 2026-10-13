"""Real-work status telemetry with UTC timestamps and monotonic phase durations."""

import time
from datetime import UTC, datetime
from pathlib import Path

from carryrl.training.run import atomic_json

PHASE_PLAN = [
    {"id": "preparation", "phases": ["loading", "collecting", "encoding"]},
    {"id": "baseline", "phases": ["initial_evaluation"]},
    {"id": "training", "phases": ["imitation", "evaluation"]},
    {"id": "final_evaluation", "phases": ["final_evaluation"]},
    {"id": "saving", "phases": ["saving"]},
]


class TrainingProgress:
    """Write on phase changes, and every two seconds or eight real updates.

    No background timer invents work: a heartbeat always follows observed work.
    Optimizer counters live in the shared state before writes, including interrupts.
    """

    def __init__(self, path: Path, *, epochs_total: int):
        self.path = path
        self.started = self.phase_started = time.monotonic()
        self.last_write = self.started
        self.updates_since_write = 0
        now = datetime.now(UTC).isoformat()
        self.state = {"status": "running", "phase": "loading", "started_at": now,
                      "updated_at": now, "phase_started_at": now, "epochs": 0,
                      "epoch": 0, "epochs_completed": 0, "epochs_total": epochs_total,
                      "batch": 0, "batches_per_epoch": None, "optimizer_steps": 0,
                      "optimizer_steps_total": None, "phase_units_done": 0,
                      "phase_units_total": None, "phase_unit_kind": "operations",
                      "phase_plan": PHASE_PLAN, "phase_history": []}
        self.write(force=True)

    def phase(self, phase: str, *, total=None, done=0, unit="operations", **values):
        now = time.monotonic()
        self.state["phase_history"].append({
            "phase": self.state["phase"], "started_at": self.state["phase_started_at"],
            "ended_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": now - self.phase_started,
            "units_done": self.state["phase_units_done"],
            "units_total": self.state["phase_units_total"],
            "unit_kind": self.state["phase_unit_kind"],
        })
        self.phase_started = now
        self.state.update(phase=phase, phase_started_at=datetime.now(UTC).isoformat(),
                          phase_units_done=done, phase_units_total=total,
                          phase_unit_kind=unit, **values)
        self.write(force=True)

    def update(self, *, force=False, optimizer_update=False, **values):
        self.state.update(values)
        if optimizer_update:
            self.updates_since_write += 1
        self.write(force=force)

    def write(self, *, force=False):
        now = time.monotonic()
        if not force and now - self.last_write < 2 and self.updates_since_write < 8:
            return
        self.state.update(updated_at=datetime.now(UTC).isoformat(),
                          elapsed_seconds=now - self.started,
                          wall_seconds=now - self.started,
                          phase_elapsed_seconds=now - self.phase_started)
        atomic_json(self.path, self.state)
        self.last_write = now
        self.updates_since_write = 0
