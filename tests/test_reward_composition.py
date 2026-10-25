import json
import math

import pytest

from carryrl.agent.actions import KEY_VOCAB, MOUSE_BUTTONS, PhysicalAction
from carryrl.rewards.composition import (
    LEGACY_REWARD_VERSION,
    MAX_INPUT_UNITS,
    OUTCOME_POTENTIAL_VERSION,
    OutcomePotentialConfig,
    RewardComposer,
    RewardCompositionConfig,
    legacy_transition,
    reward_composition_identity,
)


def prediction(score=0, status="play", **overrides):
    success, failure = {"play": (0.01, 0.01), "success": (0.99, 0.01),
                        "failure": (0.01, 0.99)}[status]
    return {"known": True, "score": float(score), "score_confidence": 0.99,
            "success_probability": success, "failure_probability": failure, **overrides}


def config(**parameters):
    return RewardCompositionConfig(OUTCOME_POTENTIAL_VERSION, parameters)


def composer(*, initial=None, horizon=8, gamma=0.99, **parameters):
    return RewardComposer(config(**parameters), gamma=gamma, horizon=horizon,
                          score_confidence_threshold=0.9,
                          initial_prediction=prediction() if initial is None else initial)


def discounted(steps, gamma):
    return sum(step.total * gamma ** index for index, step in enumerate(steps))


def run_episode(length, *, status="success", action=PhysicalAction(), horizon=20,
                initial_score=0, gamma=0.99):
    reward = composer(initial=prediction(initial_score), horizon=horizon, gamma=gamma)
    return [reward.step(prediction(min(index + initial_score, 5),
                                   status if index == length - 1 else "play"), action,
                        terminated=index == length - 1 and status != "play",
                        truncated=index == length - 1 and status == "play")
            for index in range(length)]


def test_legacy_default_exact_arithmetic_and_identity():
    before = prediction(1.123456789, success_probability=0.234567891,
                        failure_probability=0.345678912)
    after = prediction(3.987654321, success_probability=0.765432198,
                       failure_probability=0.654321987, score_confidence=0.0)
    expected = float(after["score"] - before["score"]
                     + after["success_probability"] - before["success_probability"]
                     - after["failure_probability"] + before["failure_probability"] - 0.01)
    assert legacy_transition(before, after) == expected
    reward = RewardComposer(RewardCompositionConfig(), gamma=0.99, horizon=8,
                            score_confidence_threshold=0.9, initial_prediction=before)
    step = reward.step(after, PhysicalAction(keys=("up",)), terminated=True, truncated=False)
    assert step.total == expected
    assert step.input_cost == 0
    assert step.time_cost == -0.01
    assert not reward.finite_horizon
    identity = reward_composition_identity(RewardCompositionConfig(), gamma=0.99, horizon=8,
                                           score_confidence_threshold=0.9)
    assert identity["version"] == LEGACY_REWARD_VERSION
    assert identity["parameters"] is None
    assert identity["semantics"]["time_cost"] == 0.01
    assert not identity["finite_horizon"]


@pytest.mark.parametrize("flag", ["known", "_score_trusted"])
@pytest.mark.parametrize("side", ["previous", "current"])
def test_legacy_unknown_or_explicit_untrusted_zeroes_entire_transition(flag, side):
    before, after = prediction(), prediction(5, "success")
    (before if side == "previous" else after)[flag] = False
    assert legacy_transition(before, after) == 0.0


def test_legacy_missing_known_and_trust_flags_default_true():
    before = {"score": 1, "success_probability": 0.1, "failure_probability": 0.1}
    after = {"score": 2, "success_probability": 0.2, "failure_probability": 0.0}
    assert legacy_transition(before, after) == 1 + 0.2 - 0.1 - 0.0 + 0.1 - 0.01


def test_shorter_equivalent_win_beats_longer_win():
    short = run_episode(2)
    long = run_episode(18)
    assert discounted(short, 0.99) > discounted(long, 0.99)
    assert sum(step.outcome != 0 for step in long) == 1


def test_maximum_cost_late_win_beats_immediate_failure_and_timeout():
    action = PhysicalAction(keys=KEY_VOCAB, mouse_buttons=MOUSE_BUTTONS, dx=64, dy=-64, wheel=1)
    assert MAX_INPUT_UNITS == 33
    win = run_episode(20, action=action, initial_score=4)
    failure = run_episode(1, status="failure", initial_score=4)
    timeout = run_episode(1, status="play", initial_score=4)
    assert discounted(win, 0.99) > discounted(failure, 0.99)
    assert discounted(win, 0.99) > discounted(timeout, 0.99)
    assert win[0].input_cost == -0.001


@pytest.mark.parametrize("gamma", [1.0, 0.99, 0.9])
def test_loops_and_unknown_spans_telescope_to_fixed_initial_potential(gamma):
    reward = composer(initial=prediction(2), horizon=8, gamma=gamma)
    frames = [prediction(5), prediction(0, known=False), prediction(1, score_confidence=0.1),
              prediction(2), prediction(5), prediction(0), prediction(1), prediction(2)]
    steps = [reward.step(frame, PhysicalAction(), terminated=False, truncated=index == 7)
             for index, frame in enumerate(frames)]
    shaping = sum(step.progress * gamma ** index for index, step in enumerate(steps))
    assert shaping == pytest.approx(-0.2 * 2 / 5, abs=1e-14)
    assert all(step.outcome == 0 for step in steps)
    assert steps[-1].valid_for_learning
    assert steps[1].time_cost == steps[2].time_cost == -0.001


def test_unknown_span_holds_last_potential_without_free_reward_or_cost_suppression():
    reward = composer(initial=prediction(4), horizon=3)
    unknown = reward.step(prediction(0, known=False), PhysicalAction(keys=("up",)),
                          terminated=False, truncated=False)
    recovered = reward.step(prediction(4), PhysicalAction(), terminated=False, truncated=False)
    final = reward.step(prediction(4), PhysicalAction(), terminated=False, truncated=True)
    assert unknown.progress == recovered.progress == pytest.approx(0.2 * (0.99 * 0.8 - 0.8))
    assert unknown.input_cost == pytest.approx(-0.001 / 33)
    assert not unknown.score_trusted
    assert final.progress == pytest.approx(-0.16)


def test_initial_unknown_or_missing_confidence_starts_with_zero_potential():
    for initial in (prediction(5, known=False), {"known": True, "score": 5}):
        reward = composer(initial=initial, horizon=1)
        step = reward.step(prediction(5), PhysicalAction(), terminated=False, truncated=True)
        assert step.progress == 0


def test_initial_diagnostics_use_new_mode_confidence_and_event_threshold():
    initial = prediction(3, success_probability=0.8)
    initial.pop("score_confidence")
    reward = composer(initial=initial)
    assert reward.initial_score_trusted is False
    assert reward.initial_event_status == "uncertain"


@pytest.mark.parametrize("overrides", [
    {"score": -1}, {"score": 6}, {"score": float("nan")}, {"score": True},
    {"score_confidence": 0.1}, {"score_confidence": True}, {"score_confidence": None},
])
def test_untrusted_score_never_changes_potential(overrides):
    reward = composer(initial=prediction(2), horizon=2)
    step = reward.step(prediction(5) | overrides, PhysicalAction(),
                       terminated=False, truncated=False)
    assert not step.score_trusted
    assert step.progress == pytest.approx(0.2 * (0.99 * 0.4 - 0.4))


def test_neutral_wait_and_release_pay_time_but_no_active_input_cost():
    reward = composer(horizon=3)
    held = reward.step(prediction(), PhysicalAction(keys=("space",)),
                       terminated=False, truncated=False)
    release = reward.step(prediction(), PhysicalAction(), terminated=False, truncated=False)
    wait = reward.step(prediction(), PhysicalAction(), terminated=False, truncated=True)
    assert held.input_cost < 0
    assert release.input_cost == wait.input_cost == 0
    assert release.total == wait.total == -0.001


def test_input_cost_counts_activity_not_mouse_magnitude():
    first, second = composer(horizon=1), composer(horizon=1)
    small = first.step(prediction(), PhysicalAction(dx=1, dy=-1, wheel=1),
                       terminated=False, truncated=True)
    large = second.step(prediction(), PhysicalAction(dx=64, dy=-64, wheel=-1),
                        terminated=False, truncated=True)
    assert small.input_cost == large.input_cost == pytest.approx(-0.001 * 3 / 33)


@pytest.mark.parametrize("terminated,truncated", [(True, False), (False, True), (True, True)])
def test_both_boundary_flags_zero_potential_and_use_only_learned_outcome(terminated, truncated):
    reward = composer(initial=prediction(4), horizon=2)
    step = reward.step(prediction(5, "success"), PhysicalAction(),
                       terminated=terminated, truncated=truncated)
    assert step.progress == pytest.approx(-0.16)
    assert step.outcome == 1
    assert step.event_status == "success"


def test_nonterminal_event_fluctuations_have_no_outcome_reward():
    reward = composer(horizon=3)
    win = reward.step(prediction(0, "success"), PhysicalAction(),
                      terminated=False, truncated=False)
    loss = reward.step(prediction(0, "failure"), PhysicalAction(),
                       terminated=False, truncated=False)
    timeout = reward.step(prediction(), PhysicalAction(), terminated=False, truncated=True)
    assert win.outcome == loss.outcome == timeout.outcome == 0
    assert win.valid_for_learning and loss.valid_for_learning and timeout.valid_for_learning


@pytest.mark.parametrize("overrides,status", [
    ({"known": False}, "unknown"),
    ({"success_probability": 0.99, "failure_probability": 0.99}, "conflict"),
    ({"success_probability": 0.5}, "uncertain"),
    ({"success_probability": float("nan")}, "uncertain"),
])
def test_unknown_or_ambiguous_boundary_is_invalid_and_never_invents_outcome(overrides, status):
    reward = composer(horizon=1)
    step = reward.step(prediction(**overrides), PhysicalAction(),
                       terminated=True, truncated=False)
    assert step.event_status == status
    assert not step.valid_for_learning
    assert step.invalid_reason == f"terminal_event_{status}"
    assert step.outcome == 0
    assert step.time_cost == -0.001


def test_terminated_play_is_invalid_but_truncated_play_is_valid_unfinished():
    terminated, truncated = composer(horizon=1), composer(horizon=1)
    invalid = terminated.step(prediction(), PhysicalAction(), terminated=True, truncated=False)
    valid = truncated.step(prediction(), PhysicalAction(), terminated=False, truncated=True)
    assert not invalid.valid_for_learning
    assert invalid.invalid_reason == "terminated_without_learned_terminal_event"
    assert valid.valid_for_learning and valid.outcome == 0


def test_composer_cannot_continue_after_end_and_new_episode_has_fresh_potential():
    reward = composer(initial=prediction(5), horizon=2)
    reward.step(prediction(5, "success"), PhysicalAction(), terminated=True, truncated=False)
    with pytest.raises(RuntimeError, match="ended"):
        reward.step(prediction(), PhysicalAction(), terminated=False, truncated=False)
    fresh = composer(horizon=1)
    step = fresh.step(prediction(), PhysicalAction(), terminated=False, truncated=True)
    assert step.progress == 0


def test_explicit_deadline_is_required_without_mutating_composer_on_error():
    reward = composer(initial=prediction(4), horizon=1)
    with pytest.raises(ValueError, match="boundary"):
        reward.step(prediction(5), PhysicalAction(), terminated=False, truncated=False)
    final = reward.step(prediction(), PhysicalAction(), terminated=False, truncated=True)
    assert final.progress == pytest.approx(-0.16)
    with pytest.raises(RuntimeError):
        reward.step(prediction(), PhysicalAction(), terminated=False, truncated=True)


def test_configs_and_identity_roundtrip_as_json_and_bind_actual_parameters():
    value = config(progress_weight=0.3, goal_choice_utility="completion_only")
    assert RewardCompositionConfig.from_dict(json.loads(json.dumps(value.to_dict()))) == value
    identity = reward_composition_identity(value, gamma=0.99, horizon=8,
                                           score_confidence_threshold=0.95)
    assert json.loads(json.dumps(identity)) == identity
    assert identity["parameters"]["progress_weight"] == 0.3
    assert identity["semantics"]["max_input_units"] == 33
    assert identity["finite_horizon"]
    assert identity["gamma"] == 0.99 and identity["horizon"] == 8


def test_legacy_identity_binds_gate_but_allows_discount_and_horizon_changes():
    def identity(gamma, horizon, threshold):
        return reward_composition_identity(RewardCompositionConfig(), gamma=gamma, horizon=horizon,
                                           score_confidence_threshold=threshold)

    assert identity(1, 2, 0.9) == identity(0.9, 80, 0.9)
    assert identity(1, 2, 0.9) != identity(1, 2, 0.95)


def test_goal_choice_requires_explicit_completion_only_without_invented_risk_utility():
    with pytest.raises(ValueError, match="completion_only"):
        config().validate_context(gamma=1, horizon=8, tasks=("goal_choice",),
                                  score_confidence_threshold=0.9)
    config(goal_choice_utility="completion_only").validate_context(
        gamma=1, horizon=8, tasks=("goal_choice",), score_confidence_threshold=0.9)
    for unsupported in ("risk_adjusted", "payoff", {}, True):
        with pytest.raises(ValueError, match="completion_only"):
            config(goal_choice_utility=unsupported)


@pytest.mark.parametrize("kwargs", [
    {"gamma": 0}, {"gamma": -0.1}, {"gamma": 1.1}, {"gamma": True},
    {"gamma": math.nan}, {"horizon": 0}, {"horizon": 1.0}, {"horizon": True},
    {"score_confidence_threshold": math.inf}, {"score_confidence_threshold": True},
    {"score_confidence_threshold": -0.1}, {"score_confidence_threshold": 1.1},
])
def test_invalid_context_rejected(kwargs):
    context = {"gamma": 0.99, "horizon": 8, "tasks": (), "score_confidence_threshold": 0.9}
    with pytest.raises(ValueError):
        config().validate_context(**(context | kwargs))


@pytest.mark.parametrize("parameters", [
    {"success_reward": 0}, {"failure_penalty": -1}, {"progress_weight": math.nan},
    {"time_cost": True}, {"input_cost": "0.01"}, {"event_confidence_threshold": 0.5},
    {"event_confidence_threshold": 1.1}, {"event_confidence_threshold": True},
    {"success_reward": math.inf}, {"unknown_field": 1},
])
def test_invalid_parameters_rejected(parameters):
    with pytest.raises(ValueError):
        config(**parameters)


def test_legacy_parameters_and_unknown_versions_or_fields_rejected():
    with pytest.raises(ValueError):
        RewardCompositionConfig(parameters={})
    with pytest.raises(ValueError):
        RewardCompositionConfig("unknown")
    with pytest.raises(ValueError):
        RewardCompositionConfig.from_dict({"extra": True})
    with pytest.raises(ValueError):
        RewardCompositionConfig(OUTCOME_POTENTIAL_VERSION, True)
    assert config().parameters == OutcomePotentialConfig()


@pytest.mark.parametrize("gamma,horizon,time_cost,input_cost", [
    (1, 10, 0.05, 0.05), (0.9, 50, 0.001, 0.001), (1, 2, 0.6, 0),
])
def test_win_dominance_bound_rejects_equal_or_excessive_costs(
    gamma, horizon, time_cost, input_cost,
):
    with pytest.raises(ValueError, match="success must exceed"):
        config(time_cost=time_cost, input_cost=input_cost).validate_context(
            gamma=gamma, horizon=horizon, tasks=(), score_confidence_threshold=0.9)


def test_gamma_one_and_zero_costs_satisfy_bound_for_any_supported_horizon():
    config(time_cost=0, input_cost=0).validate_context(
        gamma=1, horizon=1000000, tasks=(), score_confidence_threshold=0.9)
