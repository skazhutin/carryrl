"""Teacher audit reports admission separately from any learned-policy performance."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "audit_demonstrations_script", Path(__file__).parents[1] / "scripts/audit_demonstrations.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def test_real_teacher_audit_accepts_shortest_route_waits_and_low_payoff():
    report = script.audit_demonstrations(tasks=("navigation", "memory", "goal_choice"),
                                          levels_per_task=1)
    assert report["status"] == "completed" and report["updates"] == 0
    assert report["policy_model_loaded"] is False and report["training_corpus_saved"] is False
    assert report["accepted_episodes_in_completed_games"] == 3
    rows = {row["task"]: row for row in report["records"]}
    assert rows["navigation"]["quality"]["excess_command_units"] == 0
    assert rows["memory"]["quality"]["neutral_commands"] > 0
    assert rows["goal_choice"]["quality"]["terminal_payoff"] in (1, 2)
    assert all(row["quality_source"] == "actual_executed_trajectory" for row in rows.values())
    assert all(row["quality"]["unnecessary_input_units"] == 0 for row in rows.values())


def test_rejected_game_does_not_become_a_partially_accepted_corpus(monkeypatch):
    collect = script.collect_scripted_demonstrations

    def sometimes_reject(**kwargs):
        if kwargs["tasks"] == ("navigation",):
            raise RuntimeError("teacher quality gate rejected a detour")
        return collect(**kwargs)

    monkeypatch.setattr(script, "collect_scripted_demonstrations", sometimes_reject)
    report = script.audit_demonstrations(tasks=("navigation", "memory"), levels_per_task=1)
    assert report["status"] == "failed" and report["failed_tasks"] == ["navigation"]
    assert report["episodes_requested"] == 2
    assert report["accepted_episodes_in_completed_games"] == 1
    assert [row["task"] for row in report["records"]] == ["memory"]
    assert "episodes_accepted" not in report["tasks"]["navigation"]
    assert report["tasks"]["navigation"]["error"] == "teacher quality gate rejected a detour"


def test_audit_rejects_repeated_index_coverage_before_collection(monkeypatch):
    monkeypatch.setattr(script, "allowed_runtime_train_indices", lambda *a: [0])
    monkeypatch.setattr(script, "collect_scripted_demonstrations", lambda **kw: pytest.fail(
        "invalid coverage must fail before environment collection"))
    with pytest.raises(ValueError, match="distinct train indices"):
        script.audit_demonstrations(tasks=("navigation",), levels_per_task=2)


def test_cli_saves_only_metadata_and_protects_existing_output(monkeypatch, tmp_path):
    monkeypatch.setattr(script, "provenance", lambda: {})
    output = tmp_path / "audit"
    args = ["--output", str(output), "--tasks", "navigation", "--levels-per-task", "1"]
    assert script.main(args) == 0
    assert {path.name for path in output.iterdir()} == {
        "audit.json", "metadata.json", "status.json"}
    assert json.loads((output / "status.json").read_text()) == {
        "status": "completed", "updates": 0, "failed_tasks": []}
    assert json.loads((output / "audit.json").read_text())["records"][0]["trace_sha256"]
    with pytest.raises(FileExistsError):
        script.main(args)


def test_task_failure_does_not_report_a_terminal_run_before_audit_returns(monkeypatch, tmp_path):
    monkeypatch.setattr(script, "provenance", lambda: {})
    writes = []
    real_write = script.atomic_json

    def write(path, value):
        if path.name == "status.json":
            writes.append(value)
        real_write(path, value)

    def audit(**kwargs):
        kwargs["on_progress"]({"task": "navigation", "event": "task_completed", "status": "failed"})
        assert writes[-1]["status"] == "running"
        assert writes[-1]["task_status"] == "failed"
        return {"status": "failed", "failed_tasks": ["navigation"]}

    monkeypatch.setattr(script, "atomic_json", write)
    monkeypatch.setattr(script, "audit_demonstrations", audit)
    assert script.main(["--output", str(tmp_path / "audit")]) == 1
    assert writes[-1]["status"] == "failed"
