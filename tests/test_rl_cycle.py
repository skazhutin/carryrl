"""Cycle gates and subprocess receipts, without constructing policies or training."""
import copy
import json
from unittest.mock import Mock

import pytest

from carryrl.envs.atomic_suite import TASKS
from carryrl.training import rl_cycle as module
from carryrl.training.rl_cycle import (
    BUDGET_SECONDS,
    VERSION,
    CycleRunner,
    CycleStopped,
    choose_full_horizon,
    choose_pilot,
    initialize_budget,
    pilot_eligibility,
    select_final,
    should_stop_full,
)


def report(successes=0, mode="stochastic"):
    games = {task: {"successes": successes, "episodes": 16} for task in TASKS}
    return {"split": "validation", "mode": mode, "seed": 3176, "rollouts": 32,
            "frozen_validation": True, "initial_trainable_sha256": "a" * 64,
            "styles": {style: {"successes": successes * 16, "episodes": 256,
                                "games": copy.deepcopy(games)} for style in module.STYLES},
            "train": {"games": {task: {"successful_effective_levels": ["one", "two"]}
                                 for task in TASKS}}}


def test_clock_starts_once_before_preparation_and_does_not_reset(tmp_path):
    first = initialize_budget(tmp_path, start=True, now=100)
    assert first["deadline_unix"] == 100 + BUDGET_SECONDS
    assert initialize_budget(tmp_path, start=True, now=9999) == first
    first["deadline_unix"] += 1
    module.publish(tmp_path / "cycle_state.json", first)
    with pytest.raises(ValueError, match="original 24-hour"):
        initialize_budget(tmp_path)


def test_pilot_uses_both_styles_breadth_and_effective_levels():
    cp0, zero, temporary = report(), report(1), report(2)
    assert choose_pilot(cp0, zero, temporary)["selected_schedule"] == "temporary"
    temporary["train"]["games"] = {
        task: {"successful_effective_levels": ["same", "same"]} for task in TASKS}
    assert choose_pilot(cp0, zero, temporary)["selected_schedule"] == "zero"
    zero["styles"]["pixel_art"] = copy.deepcopy(cp0["styles"]["pixel_art"])
    assert choose_pilot(cp0, zero, temporary)["selected_schedule"] is None


def test_temp_requires_eight_better_each_style_without_losing_breadth():
    cp0, zero, temporary = report(), report(1), report(1)
    assert choose_pilot(cp0, zero, temporary)["selected_schedule"] == "zero"
    for style in module.STYLES:
        temporary["styles"][style]["games"]["navigation"]["successes"] += 8
        temporary["styles"][style]["successes"] += 8
    assert choose_pilot(cp0, zero, temporary)["selected_schedule"] == "temporary"
    temporary["train"]["games"]["navigation"]["successful_effective_levels"] = []
    assert choose_pilot(cp0, zero, temporary)["selected_schedule"] == "zero"


def test_pilot_rejects_missing_games_wrong_denominator_and_unpaired_cp0():
    cp0, candidate = report(), report(1)
    candidate["initial_trainable_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="CP0"):
        pilot_eligibility(candidate, cp0)
    candidate = report(1)
    candidate["styles"]["canonical"]["episodes"] = 16
    with pytest.raises(ValueError, match="256"):
        pilot_eligibility(candidate, cp0)
    candidate = report(1)
    candidate["styles"]["canonical"]["games"].pop("mouse")
    with pytest.raises(ValueError, match="sixteen"):
        pilot_eligibility(candidate, cp0)


def test_train_requires_two_difficult_games_even_with_eight_other_games():
    candidate = report(1)
    for task in module.HARD_TASKS - {"mouse"}:
        candidate["train"]["games"][task]["successful_effective_levels"] = []
    assert not pilot_eligibility(candidate, report())["eligible"]


def test_full_horizon_reserves_final_and_rounds_down_conservatively():
    assert choose_full_horizon([100, 120], 24 * 3600)["full_rollouts"] == 192
    assert choose_full_horizon([100, 120], 14400 + 2700 + 150 * 150)["full_rollouts"] == 128
    assert choose_full_horizon([100, 120], 14400 + 2700 + 150 * 127)["full_rollouts"] is None
    with pytest.raises(ValueError, match="two measured"):
        choose_full_horizon([100], 100000)
    with pytest.raises(ValueError):
        choose_full_horizon([100, float("nan")], 100000)


def test_final_lexicographic_minimum_then_mean_then_earlier():
    low_min, high_min = report(4, "greedy"), report(3, "greedy")
    low_min["styles"]["canonical"]["games"]["mouse"]["successes"] = 1
    low_min["styles"]["canonical"]["successes"] -= 3
    candidates = [{"checkpoint": "cp64", "rollout": 64, "report": high_min},
                  {"checkpoint": "cp128", "rollout": 128, "report": low_min},
                  {"checkpoint": "cp192", "rollout": 192, "report": high_min}]
    result = select_final(candidates)
    assert result["checkpoint"] == "cp64"
    assert result["model_frozen"] and result["test_allowed"]
    candidates[0]["report"]["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        select_final(candidates)


def test_full_early_stop_only_after_128_and_two_plateaus_without_stage_advance():
    assert should_stop_full(128, [.3, .2, .3], False)
    assert not should_stop_full(96, [.3, .2, .3], False)
    assert not should_stop_full(128, [.3, .2, .3], True)
    assert not should_stop_full(128, [.3, .2, .4], False)


@pytest.fixture
def runner_harness(tmp_path, monkeypatch):
    output = tmp_path / "cycle"
    artifact = tmp_path / "result.json"
    artifact.write_text('{"ready":true}')
    plan = tmp_path / "plan.json"
    module.publish(plan, {"version": VERSION, "jobs": [
        {"id": "fit", "phase": "rewards", "argv": ["python", "fit.py"],
         "artifacts": [{"path": str(artifact), "json_equals": {"ready": True}}]}]})
    now = [100.]
    monkeypatch.setattr(module, "disk_preflight", lambda *a, **kw: {"free_bytes": 30 * module.GIB})
    monkeypatch.setattr(module, "resource_sample", lambda *a: {
        "available_bytes": 8 * module.GIB, "pressure": 1, "process_group_rss_bytes": 0})
    identity = {"pid": 1234, "pgid": 1234, "started": "original", "command": "python fit.py"}
    monkeypatch.setattr(module, "process_identity", Mock(side_effect=[identity, None]))
    child = Mock(pid=1234)
    child.poll.side_effect = [None, 0, 0]
    child.wait.return_value = 0
    launch = Mock(return_value=child)
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    runner = CycleRunner(plan, output, clock=lambda: now[0],
                         sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    return runner, child, launch, artifact, now


def test_compute_requires_exit_artifact_gate_and_records_hashes(runner_harness):
    runner, _, launch, artifact, _ = runner_harness
    result = runner.run()
    receipt = result["stages"]["fit"]
    assert result["status"] == "completed"
    assert receipt["actual_exit_code"] == 0 and receipt["process_complete"]
    assert receipt["artifacts"][str(artifact)] == module.sha256(artifact)
    assert receipt["log_sha256"]
    assert launch.call_args.kwargs["start_new_session"]
    assert launch.call_args.kwargs["env"]["CARRYRL_CYCLE_DEADLINE_UNIX"] == "86500.0"
    with pytest.raises(CycleStopped, match="auto-restart"):
        runner.run()


def test_exit_failure_cannot_be_overridden_by_success_artifact(runner_harness):
    runner, child, _, _, _ = runner_harness
    child.wait.return_value = 7
    with pytest.raises(CycleStopped, match="successfully"):
        runner.run()
    assert runner.state["stages"]["fit"]["actual_exit_code"] == 7
    assert runner.state["status"] == "stopped"


def test_artifact_gate_failure_stops_cycle_without_retry(runner_harness):
    runner, _, launch, artifact, _ = runner_harness
    artifact.write_text('{"ready":false}')
    with pytest.raises(CycleStopped, match="gate failed"):
        runner.run()
    assert launch.call_count == 1
    with pytest.raises(CycleStopped, match="auto-restart"):
        runner.run()
    assert launch.call_count == 1


def test_interrupted_stage_refuses_auto_restart(runner_harness):
    runner, _, launch, _, _ = runner_harness
    runner.state["stages"]["fit"] = {"status": "running", "actual_exit_code": None}
    runner.save()
    with pytest.raises(CycleStopped, match="no verified process identity"):
        runner.run()
    launch.assert_not_called()


def test_pinned_plan_and_inputs_cannot_drift(runner_harness):
    runner, _, launch, artifact, _ = runner_harness
    runner.plan["pins"] = {str(artifact): "f" * 64}
    with pytest.raises(CycleStopped, match="drift"):
        runner.run()
    launch.assert_not_called()
    runner.plan_path.write_text(json.dumps({**runner.plan, "new": True}))
    with pytest.raises(CycleStopped, match="plan changed"):
        CycleRunner(runner.plan_path, runner.output)


def test_global_deadline_includes_preparation_and_memory_pause(runner_harness):
    runner, _, launch, _, now = runner_harness
    runner.state = initialize_budget(runner.output, start=True, now=100)
    now[0] = 100 + BUDGET_SECONDS
    with pytest.raises(CycleStopped, match="24-hour"):
        runner.run()
    launch.assert_not_called()


def test_full_and_test_cannot_bypass_selection_gates(runner_harness):
    runner, _, launch, _, _ = runner_harness
    runner.plan["jobs"][0]["phase"] = "full"
    with pytest.raises(CycleStopped, match="passing pilot"):
        runner.run()
    launch.assert_not_called()


def test_import_receipt_requires_dead_process_real_exit_and_unchanged_hashes(
        tmp_path, monkeypatch):
    log = tmp_path / "log"
    log.write_text("completed")
    artifact = tmp_path / "artifact"
    artifact.write_text("weights")
    identity = {"pid": 123, "pgid": 123, "started": "same", "command": "compute"}
    receipt = {"status": "completed", "actual_exit_code": 0, "process_complete": True,
               "process_identity": identity, "log_path": str(log),
               "log_sha256": module.sha256(log),
               "artifacts": {str(artifact): module.sha256(artifact)}}
    monkeypatch.setattr(module, "process_identity", lambda pid: identity)
    with pytest.raises(CycleStopped, match="alive"):
        module.verify_receipt(receipt)
    monkeypatch.setattr(module, "process_identity", lambda pid: None)
    module.verify_receipt(receipt)
    artifact.write_text("different")
    with pytest.raises(CycleStopped, match="drift"):
        module.verify_receipt(receipt)


def test_disk_forecast_preserves_eight_gib(monkeypatch, tmp_path):
    monkeypatch.setattr(module.shutil, "disk_usage", lambda directory: Mock(free=10 * module.GIB))
    module.disk_preflight(tmp_path, 2 * module.GIB)
    with pytest.raises(CycleStopped, match="headroom"):
        module.disk_preflight(tmp_path, 2 * module.GIB + 1)


def test_memory_hysteresis_and_pressure_are_explicit():
    sample = {"available_bytes": module.GIB, "pressure": 1, "process_group_rss_bytes": 0}
    assert module.memory_reason(sample) is None
    assert module.memory_reason(sample, paused=True) == "available memory reserve"
    sample["available_bytes"] = 3 * module.GIB // 2
    assert module.memory_reason(sample, paused=True) is None
    sample["pressure"] = 4
    assert module.memory_reason(sample) == "critical memory pressure"


def test_graceful_cutoff_is_once_and_has_time_to_save(runner_harness, monkeypatch):
    runner, child, _, _, now = runner_harness
    runner.plan["jobs"][0].update(max_seconds=10, graceful_seconds=4)
    identity = {"pid": 1234, "pgid": 1234, "started": "original", "command": "python fit.py"}
    # Command is alive at t=0, t=6 and exits at t=7.
    child.poll.side_effect = [None, None, 0]
    samples = iter([identity, identity, None])
    monkeypatch.setattr(module, "process_identity", lambda pid: next(samples))
    monkeypatch.setattr(module.os, "killpg", Mock())
    runner.sleep = lambda seconds: now.__setitem__(0, now[0] + 6)
    # t=12 finish breaches the limit; use first 6s then 1s interval.
    ticks = iter([6, 1])
    runner.sleep = lambda seconds: now.__setitem__(0, now[0] + next(ticks))
    receipt = runner.run()["stages"]["fit"]
    assert receipt["graceful_stop_requested"]
    assert receipt["graceful_stop_unix"] == 106
    assert receipt["elapsed_seconds"] == 7
    module.os.killpg.assert_called_once_with(1234, module.signal.SIGTERM)


def test_test_cli_argument_cannot_bypass_freeze_metadata(runner_harness):
    runner, _, launch, _, _ = runner_harness
    runner.plan["jobs"][0]["argv"] += ["--split", "test"]
    with pytest.raises(CycleStopped, match="before model freeze"):
        runner.run()
    launch.assert_not_called()


def test_one_explicit_repair_preserves_clock_prior_plan_and_failed_receipt(runner_harness):
    runner, _, _, artifact, _ = runner_harness
    runner.state = initialize_budget(runner.output, start=True, now=100)
    failed = {"status": "stopped", "actual_exit_code": 2, "elapsed_seconds": 5,
              "phase": "rewards", "error": "real calibration gate failed"}
    runner.state.update(status="stopped", error="calibration", stages={"fit": failed})
    runner.save()
    repair_plan = runner.output / "repair_plan.json"
    original_digest = runner.state["plan_sha256"]
    new_plan = copy.deepcopy(runner.plan)
    new_plan["jobs"][0]["id"] = "fit_repair"
    new_plan["pins"] = {str(artifact): module.sha256(artifact)}
    module.publish(repair_plan, new_plan)
    repaired = module.authorize_repair(runner.output, repair_plan, "repair calibration", now=200)
    assert repaired["status"] == "prepared" and repaired["repair_count"] == 1
    assert repaired["first_compute_unix"] == 100
    assert repaired["deadline_unix"] == 100 + BUDGET_SECONDS
    assert repaired["original_plan_sha256"] == original_digest
    assert repaired["stages"]["fit"] == failed
    assert module.sha256(repaired["repair_history"][0]["previous_state"]) == (
        repaired["repair_history"][0]["previous_state_sha256"])
    repaired["status"] = "stopped"
    module.publish(runner.path, repaired)
    with pytest.raises(CycleStopped, match="Only one explicit"):
        module.authorize_repair(runner.output, repair_plan, "again", now=300)


def test_repair_cannot_reuse_failed_job_id_or_extend_deadline(runner_harness):
    runner, _, _, _, _ = runner_harness
    runner.state = initialize_budget(runner.output, start=True, now=100)
    runner.state.update(status="stopped", stages={"fit": {"status": "stopped"}})
    runner.save()
    with pytest.raises(CycleStopped, match="new job IDs"):
        module.authorize_repair(runner.output, runner.plan_path, "fix", now=200)
    with pytest.raises(CycleStopped, match="expired"):
        module.authorize_repair(runner.output, runner.plan_path, "fix", now=100 + BUDGET_SECONDS)


def test_real_lightweight_subprocess_has_reapable_receipt(tmp_path, monkeypatch):
    import sys

    artifact = tmp_path / "output.json"
    plan_path = tmp_path / "plan.json"
    code = ("import pathlib,sys,time; time.sleep(.25); "
            "pathlib.Path(sys.argv[1]).write_text('{\"ready\":true}'); print('done')")
    module.publish(plan_path, {"version": VERSION, "jobs": [
        {"id": "smoke", "phase": "probe", "argv": [sys.executable, "-c", code, str(artifact)],
         "artifacts": [{"path": str(artifact), "json_equals": {"ready": True}}]}]})
    monkeypatch.setattr(module, "resource_sample", lambda *a: {
        "available_bytes": 8 * module.GIB, "pressure": 1, "process_group_rss_bytes": 0})
    monkeypatch.setattr(module, "disk_preflight", lambda *a, **kw: {})
    runner = CycleRunner(plan_path, tmp_path / "cycle")
    result = runner.run()
    receipt = result["stages"]["smoke"]
    assert result["status"] == "completed"
    assert receipt["process_identity"]["pid"] == receipt["process_identity"]["pgid"]
    module.verify_receipt(receipt)
    assert "done" in open(receipt["log_path"]).read()


def test_freeze_recomputes_embedded_selection_and_pins_real_weights(runner_harness, tmp_path):
    runner, _, _, _, _ = runner_harness
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "state.pt"
    weights.write_bytes(b"actual weights")
    module.publish(checkpoint / "manifest.json", {"sha256": module.sha256(weights)})
    candidates = [{"checkpoint": str(checkpoint), "rollout": 128,
                   "report": report(3, mode="greedy")}]
    selection = {**select_final(candidates), "candidates": candidates,
                 "checkpoint_manifest_sha256": module.sha256(checkpoint / "manifest.json")}
    sidecar = tmp_path / "selection.json"
    module.publish(sidecar, selection)
    receipt = runner._decision({"id": "freeze", "kind": "freeze",
                                "selection_report": str(sidecar)})
    assert receipt["status"] == "completed"
    assert runner.state["decisions"]["model_frozen"]
    assert runner.state["decisions"]["frozen_checkpoint_files"][str(weights)] == (
        module.sha256(weights))
    selection["mean_success_rate"] = 1.0
    module.publish(sidecar, selection)
    with pytest.raises(CycleStopped, match="recomputed"):
        runner._decision({"id": "forged", "kind": "freeze", "selection_report": str(sidecar)})
    selection["mean_success_rate"] = 3 / 16
    selection["checkpoint_manifest_sha256"] = "f" * 64
    module.publish(sidecar, selection)
    with pytest.raises(CycleStopped, match="hash is stale"):
        runner._decision({"id": "stale", "kind": "freeze", "selection_report": str(sidecar)})


def test_unused_phase_time_carries_forward_to_final_without_exceeding_22h(runner_harness):
    runner, _, _, _, _ = runner_harness
    runner.plan["jobs"][0]["phase"] = "final"
    receipt = runner.run()["stages"]["fit"]
    assert receipt["base_phase_seconds"] == 7200
    assert receipt["max_seconds"] == 22 * 3600
    assert receipt["available_borrowed_seconds"] == 20 * 3600
    assert receipt["repair_credit_available_seconds"] == 0


def test_explicit_repair_credit_permits_reward_repair_after_base_cap(runner_harness):
    runner, _, _, _, now = runner_harness
    runner.state = initialize_budget(runner.output, start=True, now=100)
    runner.state.update(status="prepared", repair_count=1,
                        repair_history=[{"kind": "learning"}])
    runner.state["stages"]["old_fit"] = {
        "status": "stopped", "phase": "rewards", "elapsed_seconds": 10800,
        "actual_exit_code": 2, "process_complete": True}
    runner.save()
    now[0] = 10901
    receipt = runner.run()["stages"]["fit"]
    assert receipt["max_seconds"] == 7199
    assert receipt["repair_credit_available_seconds"] == 7200
    assert receipt["available_borrowed_seconds"] == 7199
    assert runner.state["deadline_unix"] == 100 + BUDGET_SECONDS


def test_no_repair_credit_can_be_silently_borrowed_by_reward_phase(runner_harness):
    runner, _, launch, _, now = runner_harness
    runner.state = initialize_budget(runner.output, start=True, now=100)
    runner.state["stages"]["old_fit"] = {
        "status": "stopped", "phase": "rewards", "elapsed_seconds": 10800,
        "actual_exit_code": 2, "process_complete": True}
    runner.save()
    now[0] = 10901
    with pytest.raises(CycleStopped, match="No finite elapsed budget"):
        runner.run()
    launch.assert_not_called()


@pytest.mark.parametrize("paused", [False, True])
def test_interrupted_launcher_cannot_resume_alongside_owned_actor(
        runner_harness, monkeypatch, paused):
    runner, _, launch, _, _ = runner_harness
    identity = {"pid": 1234, "pgid": 1234, "started": "original", "command": "policy"}
    receipt = {"phase": "pilot", "status": "running", "process_identity": identity,
               "actual_exit_code": None, "process_complete": False, "memory_paused": paused}
    runner.state = initialize_budget(runner.output, start=True, now=100)
    runner.state.update(status="running", stages={"orphan": copy.deepcopy(receipt)})
    runner.save()
    monkeypatch.setattr(module, "process_identity", lambda pid: identity)
    kill = Mock()
    monkeypatch.setattr(module.os, "killpg", kill)
    with pytest.raises(CycleStopped, match="Prior owned process remains alive"):
        runner.run()
    launch.assert_not_called()
    kill.assert_not_called()
    assert runner.state["stages"]["orphan"] == receipt
    assert runner.state["deadline_unix"] == 100 + BUDGET_SECONDS
    assert runner.state.get("repair_count", 0) == 0


def test_explicit_repair_refuses_paused_orphan_and_preserves_all_evidence(
        runner_harness, monkeypatch):
    runner, _, launch, _, _ = runner_harness
    identity = {"pid": 1234, "pgid": 1234, "started": "original", "command": "policy"}
    runner.state = initialize_budget(runner.output, start=True, now=100)
    runner.state.update(status="stopped", stages={"orphan": {
        "phase": "rewards", "status": "running", "process_identity": identity,
        "actual_exit_code": None, "process_complete": False, "memory_paused": True}})
    runner.save()
    original_state = runner.path.read_bytes()
    new_plan = copy.deepcopy(runner.plan)
    new_plan["jobs"][0]["id"] = "new_fit"
    repair = runner.output / "repair.json"
    module.publish(repair, new_plan)
    monkeypatch.setattr(module, "process_identity", lambda pid: identity)
    kill = Mock()
    monkeypatch.setattr(module.os, "killpg", kill)
    with pytest.raises(CycleStopped, match="paused"):
        module.authorize_repair(runner.output, repair, "bounded repair", now=200)
    assert runner.path.read_bytes() == original_state
    assert not (runner.output / "repair_history").exists()
    launch.assert_not_called()
    kill.assert_not_called()


def test_live_descendant_with_recorded_identity_blocks_repair_after_leader_exit(monkeypatch):
    leader = {"pid": 123, "pgid": 123, "started": "leader", "command": "policy"}
    child = {"pid": 124, "pgid": 123, "started": "child", "command": "encoder"}
    receipt = {"status": "stopped", "process_identity": leader,
               "owned_group_identities": [leader, child], "actual_exit_code": -9}
    monkeypatch.setattr(module, "process_identity", lambda pid: child if pid == 124 else None)
    with pytest.raises(CycleStopped, match="group member remains alive"):
        module.assert_receipt_quiescent(receipt, stage_id="orphan")


def test_reused_pid_is_not_the_prior_actor_and_is_never_signalled(monkeypatch):
    previous = {"pid": 123, "pgid": 123, "started": "original", "command": "policy"}
    unrelated = {**previous, "started": "new process", "pgid": 999, "command": "unrelated"}
    monkeypatch.setattr(module, "process_identity", lambda pid: unrelated)
    monkeypatch.setattr(module, "process_group_pids", lambda group: [])
    kill = Mock()
    monkeypatch.setattr(module.os, "killpg", kill)
    module.assert_receipt_quiescent({"status": "stopped", "process_identity": previous},
                                    stage_id="old")
    kill.assert_not_called()


def test_unanchored_surviving_group_refuses_repair_without_signalling(monkeypatch):
    previous = {"pid": 123, "pgid": 123, "started": "original", "command": "policy"}
    monkeypatch.setattr(module, "process_identity", lambda pid: None)
    monkeypatch.setattr(module, "process_group_pids", lambda group: [456])
    kill = Mock()
    monkeypatch.setattr(module.os, "killpg", kill)
    with pytest.raises(CycleStopped, match="ownership is ambiguous"):
        module.assert_receipt_quiescent({"status": "stopped", "process_identity": previous},
                                        stage_id="old")
    kill.assert_not_called()


@pytest.mark.parametrize("kind", ["pilot_select", "full_horizon"])
def test_failed_decision_preserves_actual_gate_results_and_stop_reason(
        runner_harness, tmp_path, kind):
    runner, _, launch, _, _ = runner_harness
    if kind == "pilot_select":
        paths = {}
        for name in ("cp0", "zero", "temporary"):
            path = tmp_path / f"{name}.json"
            module.publish(path, report())
            paths[name] = str(path)
        job = {"id": "failed_pilot", "kind": kind, **paths}
    else:
        path = tmp_path / "probe.json"
        module.publish(path, {"rollout_seconds": [1000, 1200]})
        job = {"id": "failed_horizon", "kind": kind, "probe_report": str(path)}
    runner.plan["jobs"] = [job]
    with pytest.raises(CycleStopped):
        runner.run()
    state = module.read_json(runner.path)
    receipt = state["stages"][job["id"]]
    artifact = runner.output / "stages" / job["id"] / "decision.json"
    actual = module.read_json(artifact)
    assert state["status"] == "stopped" and state["current_stage"] == job["id"]
    assert receipt["status"] == "stopped" and receipt["kind"] == kind
    assert receipt["artifacts"][str(artifact)] == module.sha256(artifact)
    assert receipt["reason"] == actual["reason"]
    if kind == "pilot_select":
        assert actual["selected_schedule"] is None
        assert not actual["branches"]["zero"]["eligible"]
    else:
        assert actual["full_rollouts"] is None
        assert actual["conservative_seconds_per_rollout"] == 1500
    launch.assert_not_called()


@pytest.fixture
def infrastructure_harness(tmp_path, monkeypatch):
    old_script = tmp_path / "source_old" / "validate_reward.py"
    new_script = tmp_path / "source_recovery" / "validate_reward.py"
    old_script.parent.mkdir()
    new_script.parent.mkdir()
    old_script.write_text("# old validator")
    new_script.write_text("# fixed validator")
    old_report, new_report = tmp_path / "failed.json", tmp_path / "validation_v2.json"
    old_report.write_text('{"passed": false}')
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"unchanged fitted reward weights")
    old_job = {"id": "gate", "phase": "rewards", "argv": ["python", str(old_script),
               "--checkpoint", str(checkpoint), "--output", str(old_report)]}
    plan_path = tmp_path / "old_plan.json"
    module.publish(plan_path, {"version": VERSION, "pins": {
        str(old_script): module.sha256(old_script), str(checkpoint): module.sha256(checkpoint)},
        "jobs": [old_job]})
    output = tmp_path / "cycle"
    runner = CycleRunner(plan_path, output)
    log = output / "failed.log"
    log.write_text("AssertionError: faulty checker assumption\n")
    identity = {"pid": 1234, "pgid": 1234, "started": "old", "command": "validator"}
    runner.state = initialize_budget(output, start=True, now=100)
    runner.state.update(status="stopped", current_stage="gate", stages={"gate": {
        **old_job, "status": "stopped", "actual_exit_code": 1, "process_complete": True,
        "process_identity": identity, "log_path": str(log), "log_sha256": module.sha256(log)}})
    runner.save()
    monkeypatch.setattr(module, "process_identity", lambda pid: None)
    monkeypatch.setattr(module, "process_group_pids", lambda group: [])
    new_plan = {"version": VERSION, "pins": {
        str(new_script): module.sha256(new_script), str(checkpoint): module.sha256(checkpoint)},
        "jobs": [{**old_job, "id": "gate_v2", "argv": ["python", str(new_script),
                  "--checkpoint", str(checkpoint), "--output", str(new_report)]}]}
    new_path = tmp_path / "new_plan.json"
    module.publish(new_path, new_plan)
    return runner, new_path, checkpoint, new_plan


def test_infrastructure_repair_preserves_failed_report_and_learning_quota(infrastructure_harness):
    runner, plan, checkpoint, _ = infrastructure_harness
    old_state = copy.deepcopy(runner.state)
    repaired = module.authorize_repair(runner.output, plan, "fix checker only", now=200,
                                       repair_kind="infrastructure")
    assert repaired["repair_count"] == repaired["infrastructure_repair_count"] == 1
    assert repaired["learning_repair_count"] == 0
    assert repaired["first_compute_unix"] == old_state["first_compute_unix"]
    assert repaired["deadline_unix"] == old_state["deadline_unix"]
    assert repaired["stages"] == old_state["stages"]
    assert checkpoint.read_bytes() == b"unchanged fitted reward weights"
    assert repaired["repair_history"][0]["evidence"]["role"] == "validation"
    assert (runner.output.parent / "failed.json").read_text() == '{"passed": false}'
    # An infrastructure recovery does not prevent the one permitted learning change.
    repaired["status"] = "stopped"
    module.publish(runner.path, repaired)
    learned = module.authorize_repair(runner.output, plan, "one directed ML change", now=300)
    assert learned["repair_count"] == 2 and learned["learning_repair_count"] == 1
    learned["status"] = "stopped"
    module.publish(runner.path, learned)
    with pytest.raises(CycleStopped, match="one explicit learning repair"):
        module.authorize_repair(runner.output, plan, "second ML change", now=400)


@pytest.mark.parametrize("violation", [
    "fit_repeat", "policy_started", "same_output", "same_source", "changed_checkpoint",
    "changed_checkpoint_bytes"])
def test_infrastructure_recovery_cannot_hide_learning_or_destroy_old_evidence(
        infrastructure_harness, violation):
    runner, plan, checkpoint, content = infrastructure_harness
    if violation == "fit_repeat":
        content["jobs"].append({"id": "new_fit", "phase": "rewards",
                                "argv": ["python", "train_text_control_reward.py"]})
    elif violation == "policy_started":
        runner.state["stages"]["policy"] = {"status": "stopped", "policy_output": "policy_run",
                                              "actual_exit_code": 2}
        runner.save()
    elif violation == "same_output":
        content["jobs"][0]["argv"][-1] = str(runner.output.parent / "failed.json")
    elif violation == "same_source":
        content["jobs"][0]["argv"][1] = runner.state["stages"]["gate"]["argv"][1]
    elif violation == "changed_checkpoint_bytes":
        checkpoint.write_bytes(b"changed weights at the same path")
        content["pins"][str(checkpoint)] = module.sha256(checkpoint)
    else:
        other = checkpoint.with_name("different_weights.pt")
        other.write_bytes(b"different weights")
        content["pins"][str(other)] = module.sha256(other)
        content["jobs"][0]["argv"][3] = str(other)
    module.publish(plan, content)
    expected_checkpoint = checkpoint.read_bytes()
    before = runner.path.read_bytes()
    with pytest.raises(CycleStopped):
        module.authorize_repair(runner.output, plan, "infrastructure", now=200,
                                repair_kind="infrastructure")
    assert runner.path.read_bytes() == before
    assert checkpoint.read_bytes() == expected_checkpoint


@pytest.mark.parametrize("tamper_repaired_fit", [False, True])
def test_legacy_cli_repair_requires_explicit_hashed_reclassification(
        infrastructure_harness, tamper_repaired_fit):
    runner, plan, _, _ = infrastructure_harness
    old = copy.deepcopy(runner.state)
    failed = old["stages"]["gate"]
    failed["argv"] = ["python", "-m", "carryrl", "train-reward", "--config", "same.json"]
    from pathlib import Path

    Path(failed["log_path"]).write_text(
        "python: No module named carryrl.__main__; 'carryrl' is a package "
        "and cannot be directly executed\n")
    failed["log_sha256"] = module.sha256(failed["log_path"])
    original_plan = module.read_json(runner.plan_path)
    original_plan["jobs"][0]["argv"] = failed["argv"]
    original_path = runner.output / "actual_original.json"
    module.publish(original_path, original_plan)
    old["plan_sha256"] = module.sha256(original_path)
    old["plan_snapshot"] = module._snapshot_plan(runner.output, original_path, old["plan_sha256"])
    runner.state = old
    runner.save()
    actual_repair = module.read_json(plan)
    actual_repair["jobs"][0]["argv"] = [
        "python", "-m", "carryrl.cli", "train-reward", "--config",
        "different.json" if tamper_repaired_fit else "same.json"]
    module.publish(plan, actual_repair)
    repaired = module.authorize_repair(runner.output, plan, "legacy entrypoint correction", now=200)
    # Legacy entries without kind conservatively consume the learning quota.
    repaired["repair_history"][0].pop("kind")
    repaired.pop("learning_repair_count")
    repaired.pop("infrastructure_repair_count")
    repaired["status"] = "stopped"
    module.publish(runner.path, repaired)
    assert module._repair_counts(repaired)["learning_repair_count"] == 1
    before_history = copy.deepcopy(repaired["repair_history"])
    if tamper_repaired_fit:
        original_state = runner.path.read_bytes()
        with pytest.raises(CycleStopped, match="cannot change fit arguments"):
            module.reclassify_repair(runner.output, 1, "CLI failed before fit", now=300)
        assert runner.path.read_bytes() == original_state
        assert not repaired.get("repair_reclassifications")
        return
    classified = module.reclassify_repair(runner.output, 1, "CLI failed before fit", now=300)
    assert classified["repair_history"] == before_history
    assert classified["repair_count"] == 1 and classified["learning_repair_count"] == 0
    assert classified["infrastructure_repair_count"] == 1
    audit = classified["repair_reclassifications"][0]
    assert audit["evidence"]["role"] == "cli_entrypoint"
    assert module.sha256(audit["previous_state"]) == audit["previous_state_sha256"]
    assert classified["deadline_unix"] == 100 + BUDGET_SECONDS
    with pytest.raises(CycleStopped, match="already has"):
        module.reclassify_repair(runner.output, 1, "again", now=400)


def test_failed_real_fit_cannot_be_reclassified_as_infrastructure(infrastructure_harness):
    runner, plan, _, _ = infrastructure_harness
    runner.state["stages"]["gate"]["argv"] = ["python", "train_text_control_reward.py"]
    runner.save()
    repaired = module.authorize_repair(
        runner.output, plan, "fitted model failed readiness", now=200)
    repaired["status"] = "stopped"
    module.publish(runner.path, repaired)
    with pytest.raises(CycleStopped, match="cannot repeat a fit"):
        module.reclassify_repair(runner.output, 1, "pretend infrastructure", now=300)


def test_infrastructure_repairs_do_not_release_the_learning_reserve(runner_harness):
    runner, _, _, _, _ = runner_harness
    runner.state.update(repair_count=1, learning_repair_count=0, infrastructure_repair_count=1,
                        repair_history=[{"kind": "infrastructure"}])
    runner.save()
    receipt = runner.run()["stages"]["fit"]
    assert receipt["repair_credit_available_seconds"] == 0


def test_full_horizon_does_not_hold_consumed_learning_reserve_twice():
    held = choose_full_horizon([120, 120], 32400, evaluation_overhead_seconds=2700)
    released = choose_full_horizon([120, 120], 32400, evaluation_overhead_seconds=2700,
                                   repair_reserve_seconds=0)
    assert held["full_rollouts"] is None and held["repair_reserve_seconds"] == 7200
    assert released["full_rollouts"] == 128 and released["repair_reserve_seconds"] == 0
    assert released["available_training_seconds"] == 22500
    with pytest.raises(ValueError, match="Repair reserve"):
        choose_full_horizon([120, 120], 32400, repair_reserve_seconds=-1)


@pytest.mark.parametrize("repair_kind,expected", [("learning", 128), ("infrastructure", None)])
def test_horizon_decision_releases_reserve_only_after_learning_repair(
        runner_harness, tmp_path, repair_kind, expected):
    runner, _, _, _, _ = runner_harness
    runner.state.update(repair_count=1, repair_history=[{"kind": repair_kind}])
    runner.remaining = lambda: 32400
    probe = tmp_path / "measured_probe.json"
    module.publish(probe, {"rollout_seconds": [120, 120]})
    job = {"id": "horizon", "kind": "full_horizon", "probe_report": str(probe)}
    if expected is None:
        with pytest.raises(CycleStopped, match="128-rollout"):
            runner._decision(job)
    else:
        assert runner._decision(job)["status"] == "completed"
    actual = module.read_json(runner.output / "stages/horizon/decision.json")
    assert actual["full_rollouts"] == expected
    assert actual["repair_reserve_seconds"] == (0 if repair_kind == "learning" else 7200)
