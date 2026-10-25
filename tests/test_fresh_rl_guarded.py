"""Elapsed deadline tests never start a trainer or real subprocess."""
import importlib.util
import json
import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_fresh_rl_guarded.py"
spec = importlib.util.spec_from_file_location("fresh_guarded", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + 400))
    monkeypatch.setattr(module.signal, "signal", Mock())
    killpg = Mock()
    monkeypatch.setattr(module.os, "killpg", killpg)
    actor = dict(pid=200, pgid=200, ppid=100, started="original", command="trainer")
    child = Mock(pid=100)
    child.poll.return_value = None
    child.wait.return_value = -9
    monkeypatch.setattr(module.subprocess, "Popen", Mock(return_value=child))
    guard = SimpleNamespace(same_process=Mock(return_value=True),
                            identity=Mock(return_value=actor),
                            guard=SimpleNamespace(group_members=lambda group: [{"pid": 200}]))

    def publish(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    runner = SimpleNamespace(utc=lambda: "now", publish=publish)
    report = dict(supervisor_pid=100, command_pid=200,
                  runtime_guard_override_receipt={"actor_identity": actor},
                  status="paused", active_wall_seconds=0, actual_exit_code=None)
    original = module.read_optional
    monkeypatch.setattr(module, "read_optional", lambda path:
                        report if path.name == "process.json" else original(path))
    return tmp_path, child, guard, runner, report, killpg, now


def test_elapsed_deadline_includes_memory_pause(harness):
    plan, child, guard, runner, _, killpg, _ = harness
    result = module.monitor(plan, {"budget_seconds": 2400}, guard, runner)
    assert result["status"] == "hard_timeout"
    assert result["soft_stop_elapsed_seconds"] == 1200
    assert result["elapsed_seconds"] == 2400
    child.send_signal.assert_called_once_with(signal.SIGTERM)
    child.kill.assert_called_once()
    killpg.assert_called_once_with(200, signal.SIGKILL)
    assert result["actual_exit_code"] == -9


def test_stale_actor_identity_never_signalled(harness):
    plan, child, guard, runner, _, killpg, _ = harness
    guard.same_process.return_value = False
    module.monitor(plan, {"budget_seconds": 2400}, guard, runner)
    killpg.assert_not_called()
    child.kill.assert_called_once()


def test_failure_cannot_be_overridden_by_completed_status(harness):
    plan, child, guard, runner, report, _, _ = harness
    child.poll.return_value = 7
    child.wait.return_value = 7
    report.update(status="completed", actual_exit_code=0)
    runner.publish(plan / "execution.json", {"status": "completed"})
    result = module.monitor(plan, {"budget_seconds": 2400}, guard, runner)
    assert result["status"] == "failed"
    assert result["actual_exit_code"] == 7


def test_requires_actual_actor_exit_code(harness):
    plan, child, guard, runner, report, _, _ = harness
    child.poll.return_value = 0
    child.wait.return_value = 0
    report.update(status="completed", actual_exit_code=None)
    runner.publish(plan / "execution.json", {"status": "completed"})
    assert module.monitor(plan, {"budget_seconds": 2400}, guard, runner)["status"] == "failed"


def test_rejects_existing_stage(harness):
    plan, child, guard, runner, _, _, _ = harness
    (plan / "stages/main").mkdir(parents=True)
    with pytest.raises(ValueError, match="duplicate"):
        module.monitor(plan, {"budget_seconds": 2400}, guard, runner)
    child.send_signal.assert_not_called()


def test_completed_requires_reaped_processes(harness):
    plan, child, guard, runner, report, killpg, _ = harness
    child.poll.return_value = 0
    child.wait.return_value = 0
    guard.same_process.return_value = False
    report["actual_exit_code"] = 0
    runner.publish(plan / "execution.json", {"status": "completed"})
    result = module.monitor(plan, {"budget_seconds": 2400}, guard, runner)
    assert result["status"] == "completed"
    killpg.assert_not_called()


def test_default_only_preflights(tmp_path, monkeypatch):
    runner = SimpleNamespace(verify_plan=Mock(return_value=({"budget_seconds": 14400},
                                                          None, None)))
    loader = Mock(return_value=runner)
    monkeypatch.setattr(module, "load", loader)
    launch = Mock()
    monkeypatch.setattr(module, "monitor", launch)
    assert module.main(["--plan", str(tmp_path)]) == 0
    launch.assert_not_called()
    assert loader.call_count == 1


@pytest.mark.parametrize("execution_status,guard_error,expected", [
    ("resource_stopped", None, "stopped"),
    ("stopped", None, "stopped"),
    ("failed", None, "failed"),
    ("resource_stopped", "guard failed", "failed"),
])
def test_orderly_bounded_stop_distinguished_from_failure(
        harness, execution_status, guard_error, expected):
    plan, child, guard, runner, report, _, _ = harness
    child.poll.return_value = 1
    child.wait.return_value = 1
    guard.same_process.return_value = False
    report.update(actual_exit_code=2, error=guard_error)
    runner.publish(plan / "execution.json", {"status": execution_status})
    result = module.monitor(plan, {"budget_seconds": 2400}, guard, runner)
    assert result["status"] == expected
    assert result["actual_exit_code"] == 1
    assert result["actor_exit_code"] == 2
