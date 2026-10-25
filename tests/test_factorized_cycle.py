import copy

import pytest
from test_rl_cycle import report
from test_rl_cycle import runner_harness as runner_harness

from carryrl.training import rl_cycle
from carryrl.training.factorized_cycle import (
    choose_factorized_full_horizon,
    choose_factorized_pilot,
    forecast_remaining_cycle,
)


def pilot(wins):
    value = report(wins)
    value["rollouts"] = 80
    return value


def test_three_way_pilot_prefers_no_shaping_when_outcomes_tie():
    branches = {name: pilot(1) for name in ("zero", "text", "text_penalty")}
    assert choose_factorized_pilot(pilot(0), branches)["selected_branch"] == "zero"


def test_penalty_needs_game_improvement_over_otherwise_identical_text():
    branches = {"zero": pilot(1), "text": pilot(2), "text_penalty": pilot(2)}
    assert choose_factorized_pilot(pilot(0), branches)["selected_branch"] == "text"
    branches["text_penalty"] = pilot(3)
    assert choose_factorized_pilot(pilot(0), branches)["selected_branch"] == "text_penalty"


def test_penalty_cannot_win_by_behavior_counts_with_same_game_outcomes():
    branches = {"zero": pilot(0), "text": pilot(1), "text_penalty": pilot(1)}
    branches["text_penalty"]["behavior"] = {"unsupported_steps": 0}
    assert choose_factorized_pilot(pilot(0), branches)["selected_branch"] == "text"


def test_penalty_advantage_in_only_one_style_does_not_suffice():
    branches = {"zero": pilot(1), "text": pilot(2), "text_penalty": pilot(3)}
    branches["text_penalty"]["styles"]["pixel_art"] = copy.deepcopy(
        branches["text"]["styles"]["pixel_art"])
    assert choose_factorized_pilot(pilot(0), branches)["selected_branch"] == "text"


def test_missing_or_wrong_duration_branch_rejected():
    with pytest.raises(ValueError, match="three"):
        choose_factorized_pilot(pilot(0), {"zero": pilot(1)})
    branches = {name: pilot(1) for name in ("zero", "text", "text_penalty")}
    branches["text"]["rollouts"] = 32
    with pytest.raises(ValueError, match="80 rollouts"):
        choose_factorized_pilot(pilot(0), branches)


def test_pilot_cannot_change_initial_weights_or_pass_without_breadth():
    branches = {name: pilot(0) for name in ("zero", "text", "text_penalty")}
    assert choose_factorized_pilot(pilot(0), branches)["selected_branch"] is None
    branches["text_penalty"]["initial_trainable_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="CP0"):
        choose_factorized_pilot(pilot(0), branches)


def test_forecast_requires_three_full_pilots_main128_and_final_reserve():
    result = forecast_remaining_cycle([100, 120], 600, 24 * 3600)
    assert result["pilot_seconds"] == 3 * (80 * 150 + 4 * 750)
    assert result["full_rollouts"] == 192
    minimum = result["total_seconds_by_main_horizon"][128]
    assert forecast_remaining_cycle([100, 120], 600, minimum)["full_rollouts"] == 128
    assert not forecast_remaining_cycle([100, 120], 600, minimum - 1)[
        "all_required_stages_fit"]
    with pytest.raises(ValueError, match="Measured"):
        forecast_remaining_cycle([100], 600, 100000)


def test_revised_budget_uses_actual_remaining_clock_without_reset(runner_harness):
    runner, _, launch, _, now = runner_harness
    runner.plan.update(budget_protocol="remaining_global_phase_caps.v1",
                       phase_limits_seconds=dict(rl_cycle.STAGE_LIMITS))
    runner.state.update(first_compute_unix=100., deadline_unix=86500.)
    now[0] = 100. + 8 * 3600
    receipt = runner._compute(runner.plan["jobs"][0])
    assert receipt["actual_exit_code"] == 0
    assert receipt["max_seconds"] == rl_cycle.STAGE_LIMITS["rewards"]
    assert runner.state["first_compute_unix"] == 100.
    assert runner.state["deadline_unix"] == 86500.
    assert launch.call_args.kwargs["env"]["CARRYRL_CYCLE_DEADLINE_UNIX"] == "86500.0"


def test_revised_budget_still_protects_final_reserve(runner_harness):
    runner, _, launch, _, now = runner_harness
    runner.plan.update(budget_protocol="remaining_global_phase_caps.v1",
                       phase_limits_seconds=dict(rl_cycle.STAGE_LIMITS))
    runner.state.update(first_compute_unix=100., deadline_unix=86500.)
    now[0] = 86500. - 7200
    with pytest.raises(rl_cycle.CycleStopped, match="budget"):
        runner._compute(runner.plan["jobs"][0])
    launch.assert_not_called()


def test_factorized_main_horizon_uses_measured_evaluation_time():
    # Four complete evaluations take 5000s conservatively, not the historical2700s.
    result = choose_factorized_full_horizon([100, 120], 1000, 30000)
    assert result["full_rollouts"] == 160
    assert result["evaluation_overhead_seconds"] == 5000
    assert result["control_schedule_rollouts"] == 64
    minimum = result["required_seconds_by_horizon"][128]
    assert choose_factorized_full_horizon([100, 120], 1000, minimum)["full_rollouts"] == 128
    assert choose_factorized_full_horizon([100, 120], 1000, minimum - 1)[
        "full_rollouts"] is None


def test_factorized_horizon_respects_remaining_phase_cap(runner_harness, tmp_path):
    runner, _, _, _, _ = runner_harness
    runner.plan.update(budget_protocol="remaining_global_phase_caps.v1",
                       phase_limits_seconds={**rl_cycle.STAGE_LIMITS, "full": 30000})
    runner.state.update(repair_count=1, repair_history=[{"kind": "learning"}])
    runner.state["stages"]["preparation"] = {"phase": "full", "elapsed_seconds": 5000}
    runner.remaining = lambda: 7200 + 50000
    probe, timing = tmp_path / "probe.json", tmp_path / "timing.json"
    rl_cycle.publish(probe, {"rollout_seconds": [100, 120]})
    rl_cycle.publish(timing, {"evaluation_seconds": 1000})
    runner._decision({"id": "horizon", "kind": "factorized_full_horizon",
                      "probe_report": str(probe), "timing_report": str(timing)})
    assert runner.state["decisions"]["full_rollouts"] == 128
    assert runner.state["decisions"]["available_training_seconds"] == 25000


def test_factorized_horizon_honors_actual_training_runtime_limit(runner_harness, tmp_path):
    runner, _, _, _, _ = runner_harness
    runner.plan.update(budget_protocol="remaining_global_phase_caps.v1",
                       phase_limits_seconds={**rl_cycle.STAGE_LIMITS, "full": 42000})
    runner.state.update(repair_count=1, repair_history=[{"kind": "learning"}])
    runner.remaining = lambda: 100000
    probe, timing = tmp_path / "probe.json", tmp_path / "timing.json"
    rl_cycle.publish(probe, {"rollout_seconds": [100, 120]})
    rl_cycle.publish(timing, {"evaluation_seconds": 1000})
    runner._decision({"id": "horizon", "kind": "factorized_full_horizon",
                      "probe_report": str(probe), "timing_report": str(timing),
                      "launch_overhead_seconds": 1500, "runtime_limit_seconds": 30000})
    assert runner.state["decisions"]["full_rollouts"] == 160
    assert runner.state["decisions"]["available_training_seconds"] == 30000


@pytest.mark.parametrize("kind", ["pilot_factorized_select", "factorized_full_horizon"])
def test_factorized_decision_receipts_need_no_command_process_identity(tmp_path, kind):
    artifact = tmp_path / "decision.json"
    rl_cycle.publish(artifact, {"kind": kind})
    pins = {str(artifact): rl_cycle.sha256(artifact)}
    state = {"stages": {"decision": {"kind": kind, "status": "completed",
                                      "artifacts": pins}}}
    assert rl_cycle._completed_artifacts(state) == pins
    artifact.write_text("{}")
    with pytest.raises(rl_cycle.CycleStopped, match="Pinned input drift"):
        rl_cycle._completed_artifacts(state)
