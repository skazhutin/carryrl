import copy
import json

import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.rewards.composition import (
    TEXT_CONTROL_V2_VERSION,
    RewardComposer,
    RewardCompositionConfig,
    TextControlV2Config,
    control_schedule_coefficient,
    control_schedule_coefficient_v2,
    reward_composition_identity,
)
from carryrl.rewards.control_text import CANDIDATES
from carryrl.rewards.control_text_factorized import (
    FactorizedControlPrediction,
    FactorizedControlPredictionTable,
    FieldDecision,
    PairDecision,
)


def decision(value, *, accepted=True):
    return FieldDecision(value, .999, accepted, None if accepted else "low_confidence")


def table(spec):
    rows = []
    for candidate in CANDIDATES:
        status, mode, group = spec.get(candidate, ("unspecified", "unspecified", None))
        rows.append(FactorizedControlPrediction(
            candidate, decision(status, accepted=status != "unspecified"),
            decision(mode, accepted=mode != "unspecified"), group,
            status == "supported" and group is not None))
    pairs = tuple(PairDecision(a.candidate, b.candidate, decision(
        "same" if a.group is not None and a.group == b.group else "different"))
        for i, a in enumerate(rows) for b in rows[i + 1:])
    return FactorizedControlPredictionTable("a" * 64, "b" * 64, "c" * 64, tuple(rows), pairs)


def prediction(status="play"):
    return {"known": True, "score": 0, "score_confidence": .99,
            "success_probability": .99 if status == "success" else .01,
            "failure_probability": .99 if status == "failure" else .01}


def composer(spec, *, horizon=100, coefficient=1, alpha=.01, reward=1, schedule="temporary"):
    config = RewardCompositionConfig(TEXT_CONTROL_V2_VERSION, {
        "control_schedule": schedule, "unsupported_alpha": alpha,
        "success_reward": reward, "time_cost": 0, "input_cost": 0})
    return RewardComposer(config, gamma=1, horizon=horizon, score_confidence_threshold=.9,
                          initial_prediction=prediction(), control_table=table(spec),
                          control_coefficient=coefficient)


def step(reward, *, keys=(), buttons=(), dx=0, dy=0, wheel=0, boundary=False,
         terminated=False, status="play"):
    return reward.step(prediction(status), PhysicalAction(keys=keys, mouse_buttons=buttons,
                                                         dx=dx, dy=dy, wheel=wheel),
                       terminated=terminated, truncated=boundary)


@pytest.mark.parametrize("cycle,expected", [(1, 1), (32, 1), (33, 31 / 32),
                                           (63, 1 / 32), (64, 0), (200, 0)])
def test_absolute_schedule(cycle, expected):
    for length in (1, 16, 64, 1000):
        assert control_schedule_coefficient_v2("temporary", cycle, length) == expected
    assert control_schedule_coefficient_v2("zero", cycle) == 0
    assert control_schedule_coefficient_v2("permanent", cycle) == 1
    # Historical zero-based relative schedule retains its old arithmetic.
    assert control_schedule_coefficient("temporary", 4, 8) == 0


@pytest.mark.parametrize("cycle", [0, -1, True, 1.0])
def test_absolute_schedule_rejects_invalid_cycle(cycle):
    with pytest.raises(ValueError):
        control_schedule_coefficient_v2("temporary", cycle)


def test_config_identity_and_explicit_table_type():
    config = RewardCompositionConfig(TEXT_CONTROL_V2_VERSION, {"unsupported_alpha": 0})
    assert type(config.parameters) is TextControlV2Config
    assert RewardCompositionConfig.from_dict(config.to_dict()) == config
    identity = reward_composition_identity(config, gamma=1, horizon=20,
                                           score_confidence_threshold=.9)
    assert identity["semantics"]["text_control"]["unsupported_alpha"] == 0
    with pytest.raises(ValueError, match="factorized"):
        RewardComposer(config, gamma=1, horizon=20, score_confidence_threshold=.9,
                       initial_prediction=prediction(), control_table=object())
    for value in (-1, .02, True, float("nan")):
        with pytest.raises(ValueError):
            TextControlV2Config(unsupported_alpha=value)


@pytest.mark.parametrize("kind", ["key", "mouse"])
def test_held_requires_two_same_physical_samples_then_pays_once(kind):
    candidate = "key:a" if kind == "key" else "mouse:left"
    args = {"keys": ("a",)} if kind == "key" else {"buttons": ("left",)}
    reward = composer({candidate: ("supported", "held", "move")})
    assert step(reward, **args).control_bonus == .0005
    assert step(reward, **args).control_bonus == .0015
    assert step(reward, **args).control_bonus == 0
    assert step(reward).control_bonus == 0
    assert step(reward, **args).control_bonus == 0


def test_alias_switch_does_not_prove_held_and_group_is_paid_only_once():
    reward = composer({name: ("supported", "held", "move") for name in ("key:a", "key:left")})
    values = [step(reward, keys=keys).control_bonus
              for keys in (("a",), ("left",), ("a",), ("a",), ("left",), ("left",))]
    assert values == [.0005, 0, 0, .0015, 0, 0]


def test_alias_with_its_own_press_mode_can_finish_shared_group_award():
    reward = composer({"key:a": ("supported", "held", "move"),
                       "key:left": ("supported", "press", "move")})
    assert step(reward, keys=("a",)).control_bonus == .0005
    assert step(reward, keys=("left",)).control_bonus == 0
    assert step(reward).control_bonus == .0015
    assert step(reward, keys=("a",)).control_bonus == 0
    assert step(reward, keys=("a",)).control_bonus == 0


def test_relative_alias_can_finish_group_without_counting_as_a_held_key():
    reward = composer({"key:a": ("supported", "held", "move"),
                       "wheel:+": ("supported", "relative", "move")})
    assert step(reward, keys=("a",)).control_bonus == .0005
    assert step(reward, wheel=1).control_bonus == .0015
    assert step(reward, wheel=1).control_bonus == 0


def test_press_requires_single_sample_and_real_release_with_no_backward_edit():
    reward = composer({"key:a": ("supported", "press", "tap")})
    first = step(reward, keys=("a",))
    assert first.control_bonus == .0005
    assert step(reward).control_bonus == .0015
    assert first.control_bonus == .0005
    assert step(reward, keys=("a",)).control_bonus == 0
    assert step(reward).control_bonus == 0


def test_press_held_too_long_can_retry_fresh_tap():
    reward = composer({"key:a": ("supported", "press", "tap")})
    assert step(reward, keys=("a",)).control_bonus == .0005
    assert step(reward, keys=("a",)).control_bonus == 0
    assert step(reward).control_bonus == 0
    assert step(reward, keys=("a",)).control_bonus == 0
    assert step(reward).control_bonus == .0015


@pytest.mark.parametrize("axis,value", [("dx", 1), ("dx", -1), ("dy", 4), ("wheel", -1)])
def test_relative_nonzero_current_sample_can_pay_both_parts(axis, value):
    candidate = f"{axis}:{'+' if value > 0 else '-'}"
    reward = composer({candidate: ("supported", "relative", "relative")})
    assert step(reward).control_bonus == 0
    award = step(reward, **{axis: value})
    assert award.control_bonus == .002
    assert award.control_events[0]["base_amount"] == .0005
    assert award.control_events[0]["upgrade_amount"] == .0015
    assert step(reward, **{axis: value}).control_bonus == 0


def test_unknown_mode_pays_base_only_and_unknown_alias_pays_nothing():
    reward = composer({"key:a": ("supported", "unspecified", "base"),
                       "key:s": ("supported", "held", None)})
    assert step(reward, keys=("a",)).control_bonus == .0005
    assert step(reward, keys=("a",)).control_bonus == 0
    assert step(reward).control_bonus == 0
    assert step(reward, keys=("s",)).control_bonus == 0
    assert step(reward, keys=("s",)).control_bonus == 0


def test_unsupported_recurs_once_per_step_release_unknown_and_neutral_are_free():
    reward = composer({"key:a": ("unsupported", "unspecified", None),
                       "key:s": ("unsupported", "unspecified", None)}, horizon=10)
    for keys in (("a",), ("a",), ("a", "s"), ("s",)):
        result = step(reward, keys=keys)
        assert result.unsupported_penalty == -.001
        assert sum(e["event"] == "unsupported_step_penalty" for e in result.control_events) == 1
    assert step(reward).unsupported_penalty == 0
    assert step(reward, keys=("d",)).unsupported_penalty == 0
    assert step(reward).unsupported_penalty == 0
    assert reward.state_dict()["control"]["negative"] == .004


@pytest.mark.parametrize("alpha", [0, .01])
def test_unsupported_veto_independent_of_penalty_alpha_and_does_not_consume_base(alpha):
    reward = composer({"key:a": ("supported", "held", "move"),
                       "key:s": ("unsupported", "unspecified", None)}, alpha=alpha)
    assert step(reward, keys=("a", "s")).control_bonus == 0
    assert step(reward, keys=("a",)).control_bonus == .0005
    # Dirty start cannot become a clean held witness by simply dropping the bad key.
    assert step(reward, keys=("a",)).control_bonus == 0
    step(reward)
    step(reward, keys=("a",))
    assert step(reward, keys=("a",)).control_bonus == .0015


def test_dirty_release_vetoes_press_upgrade_but_allows_later_clean_retry():
    reward = composer({"key:a": ("supported", "press", "tap"),
                       "key:s": ("unsupported", "unspecified", None)})
    step(reward, keys=("a",))
    assert step(reward, keys=("s",)).control_bonus == 0
    step(reward, keys=("a",))
    assert step(reward).control_bonus == .0015


def test_upgrade_priority_and_stable_vocabulary_one_group_per_command():
    reward = composer({"key:a": ("supported", "held", "a"),
                       "key:s": ("supported", "held", "s"),
                       "dx:+": ("supported", "relative", "x")})
    # Relative witness wins over the earlier vocabulary candidate's base.
    first = step(reward, keys=("a", "s"), dx=1)
    assert first.control_bonus == .002
    assert [e["group"] for e in first.control_events if e["amount"] > 0] == ["x"]
    second = step(reward, keys=("s", "a"))
    assert second.control_bonus == .002
    assert [e["group"] for e in second.control_events if e["amount"] > 0] == ["a"]
    assert step(reward, keys=("s",)).control_bonus == .002


def test_caps_clip_final_event_and_record_actual_parts_without_decay_compensation():
    spec = {name: ("supported", "relative", name)
            for name in ("dx:-", "dx:+", "dy:-", "dy:+", "wheel:-", "wheel:+")}
    reward = composer(spec)
    amounts = [step(reward, **{axis: value}).control_bonus
               for axis, value in (("dx", -1), ("dx", 1), ("dy", -1), ("dy", 1),
                                   ("wheel", -1), ("wheel", 1))]
    assert amounts == pytest.approx([.002] * 5 + [0])
    assert reward.state_dict()["control"]["positive"] == .01
    reward = composer(spec, coefficient=.5)
    assert step(reward, dx=1).control_bonus == .001
    assert step(reward, dx=1).control_bonus == 0
    paid = reward.state_dict()["control"]["groups"]["dx:+"]
    assert paid["base_paid"] == .00025
    assert paid["upgrade_paid"] == .00075
    # First a half-mill base, then four .002 groups leaves .0015 for a combined event.
    spec["key:a"] = ("supported", "unspecified", "base")
    reward = composer(spec)
    step(reward, keys=("a",))
    for axis, value in (("dx", -1), ("dx", 1), ("dy", -1), ("dy", 1)):
        step(reward, **{axis: value})
    last = step(reward, wheel=1)
    assert last.control_bonus == pytest.approx(.0015)
    paid = reward.state_dict()["control"]["groups"]["wheel:+"]
    assert paid["base_paid"] == .0005
    assert paid["upgrade_paid"] == pytest.approx(.001)


@pytest.mark.parametrize("reward", [1, 3.5])
def test_scaling_and_terminal_penalty_but_no_base_upgrade_or_implicit_release(reward):
    spec = {"key:a": ("supported", "press", "tap"),
            "key:s": ("unsupported", "unspecified", None),
            "dx:+": ("supported", "relative", "x")}
    for kwargs in ({}, {"dx": 1}, {"keys": ("s",)}):
        instance = composer(spec, horizon=2, reward=reward)
        assert step(instance, keys=("a",)).control_bonus == .0005 * reward
        terminal = step(instance, boundary=True, **kwargs)
        assert terminal.control_bonus == 0
        assert terminal.unsupported_penalty == (-.005 * reward if kwargs.get("keys") else 0)
    instance = composer(spec, reward=reward)
    assert step(instance, dx=1).control_bonus == .002 * reward


@pytest.mark.parametrize("mode,first,second", [
    ("held", {"keys": ("a",)}, {"keys": ("a",)}),
    ("press", {"keys": ("a",)}, {}),
])
def test_json_restore_across_rollout_chunk_preserves_physical_witness(mode, first, second):
    spec = {"key:a": ("supported", mode, "a")}
    original = composer(spec)
    step(original, **first)
    state = json.loads(json.dumps(original.state_dict()))
    restored = composer(spec)
    restored.load_state_dict(state)
    assert step(restored, **second) == step(original, **second)
    assert restored.state_dict() == original.state_dict()
    # Defensive copies: external mutation cannot change live accounting.
    state["control"]["groups"]["a"]["base_paid"] = 42
    assert restored.state_dict()["control"]["groups"]["a"]["base_paid"] == .0005


@pytest.mark.parametrize("mutation", [
    lambda s: s["identity"].update(horizon=999),
    lambda s: s["control"].update(positive=1),
    lambda s: s["control"]["runs"]["key:a"].update(length=999),
    lambda s: s.update(steps=True),
    lambda s: s.update(unexpected=1),
])
def test_restore_rejects_mismatch_and_corruption_without_partial_mutation(mutation):
    reward = composer({"key:a": ("supported", "held", "a")})
    step(reward, keys=("a",))
    before = reward.state_dict()
    corrupt = copy.deepcopy(before)
    mutation(corrupt)
    with pytest.raises(ValueError):
        reward.load_state_dict(corrupt)
    assert reward.state_dict() == before


def test_horizon_is_enforced_and_terminal_state_can_be_restored():
    spec = {"key:a": ("unsupported", "unspecified", None)}
    reward = composer(spec, horizon=2)
    step(reward, keys=("a",))
    before = reward.state_dict()
    with pytest.raises(ValueError, match="boundary"):
        step(reward, keys=("a",))
    assert reward.state_dict() == before
    step(reward, keys=("a",), boundary=True)
    assert reward.state_dict()["control"]["negative"] == .01
    restored = composer(spec, horizon=2)
    restored.load_state_dict(reward.state_dict())
    with pytest.raises(RuntimeError, match="ended"):
        step(restored, keys=("a",))


def test_low_confidence_unsupported_abstains_without_veto_or_penalty():
    # The fixture's unspecified applicability is the same abstention decision
    # emitted when a learned unsupported logit does not clear calibration.
    reward = composer({"key:a": ("supported", "held", "a")})
    result = step(reward, keys=("a", "s"))
    assert result.control_bonus == .0005
    assert result.unsupported_penalty == 0
    abstention = next(e for e in result.control_events if e.get("candidate") == "key:s")
    assert abstention["reason"] == "low_confidence"


def test_unknown_terminal_visual_prediction_keeps_invalidity_and_text_penalty():
    reward = composer({"key:a": ("unsupported", "unspecified", None)}, horizon=1)
    current = prediction()
    current["known"] = False
    result = reward.step(current, PhysicalAction(keys=("a",)),
                         terminated=True, truncated=False)
    assert not result.valid_for_learning
    assert result.invalid_reason == "terminal_event_unknown"
    assert result.unsupported_penalty == -.01
    assert result.control_bonus == 0


def test_zero_coefficient_still_tracks_physical_state_without_consuming_awards():
    reward = composer({"key:a": ("supported", "held", "a"),
                       "key:s": ("unsupported", "unspecified", None)}, coefficient=0)
    assert step(reward, keys=("a",)).control_bonus == 0
    assert step(reward, keys=("a",)).control_bonus == 0
    assert step(reward, keys=("s",)).unsupported_penalty == 0
    state = reward.state_dict()
    assert state["control"]["positive"] == state["control"]["negative"] == 0
    assert not state["control"]["groups"]["a"]["base_awarded"]
    assert not state["control"]["groups"]["a"]["upgrade_awarded"]


def test_terminal_held_sample_does_not_pay_upgrade():
    reward = composer({"key:a": ("supported", "held", "a")}, horizon=2)
    assert step(reward, keys=("a",)).control_bonus == .0005
    assert step(reward, keys=("a",), boundary=True).control_bonus == 0
    assert not reward.state_dict()["control"]["groups"]["a"]["upgrade_awarded"]
