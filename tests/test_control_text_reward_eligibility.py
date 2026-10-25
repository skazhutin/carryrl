"""Semantic predictions never bypass unresolved alias reward admission."""

import pytest
from test_control_text import composer, step, table
from test_control_text_conditional import fixed_model

from carryrl.agent.actions import PhysicalAction
from carryrl.rewards.control_text import CANDIDATES, ControlPrediction, ControlPredictionTable


def test_unresolved_semantics_pay_zero_for_both_signs_and_do_not_mark_none_seen():
    rows = tuple(
        ControlPrediction(name, "supported" if i % 2 else "unsupported", "held", None,
                          1.0, deduplication_known=False)
        for i, name in enumerate(CANDIDATES)
    )
    reward = composer(ControlPredictionTable("0" * 64, "1" * 64, "2" * 64, rows))
    result = step(reward, PhysicalAction(keys=("w", "a"), dx=1))
    assert result.control_bonus == result.unsupported_penalty == 0
    assert {event["event"] for event in result.control_events} == {"deduplication_unknown"}
    assert not reward._control_seen
    assert reward._control_positive == reward._control_negative == 0
    assert step(reward).control_events == ()


def test_uncertain_cross_group_relation_cannot_create_positive_singletons():
    reward = composer(fixed_model(
        supported=("key:a", "key:d"), uncertain=(("key:a", "key:d"),),
    ).predict("Hold A and D"))
    assert step(reward, PhysicalAction(keys=("a",))).control_bonus == 0
    assert step(reward, PhysicalAction(keys=("d",))).control_bonus == 0
    assert not reward._control_seen


def test_certified_alias_reuse_and_unsupported_physical_once_ignore_pair_uncertainty():
    reward = composer(fixed_model(
        supported=("key:a", "key:d"), unsupported=("key:f", "key:q"),
        same=(("key:a", "key:d"),), uncertain=(("key:f", "key:q"),),
    ).predict("Hold A and D"))
    assert step(reward, PhysicalAction(keys=("a", "d"))).control_bonus == 0.001
    assert step(reward).control_bonus == 0  # release has no executed control
    assert step(reward, PhysicalAction(keys=("d",))).control_bonus == 0
    assert step(reward, PhysicalAction(keys=("f", "q"))).unsupported_penalty == -0.002
    assert step(reward, PhysicalAction(keys=("f", "q"))).unsupported_penalty == 0


def test_unresolved_unsupported_still_vetoes_and_marks_supported_once():
    baseline = table(**{"key:w": ("supported", "forward")})
    rows = tuple(
        ControlPrediction(row.candidate, "unsupported", "unknown", None, 1.0,
                          deduplication_known=False)
        if row.candidate == "key:f" else row for row in baseline.predictions
    )
    reward = composer(ControlPredictionTable("0" * 64, "1" * 64, "2" * 64, rows))
    result = step(reward, PhysicalAction(keys=("w", "f")))
    assert result.control_bonus == result.unsupported_penalty == 0
    assert step(reward, PhysicalAction(keys=("w",))).control_bonus == 0
    assert reward._control_seen == {"forward"}


def test_unresolved_groups_excluded_from_normalization_and_both_episode_caps():
    # Eleven resolved physical classes per sign give unit=0.01/11. A None
    # pseudo-group would incorrectly reduce this to 0.01/12.
    rows = []
    supported, unsupported = CANDIDATES[:11], CANDIDATES[11:22]
    for name in CANDIDATES:
        if name in supported or name in unsupported:
            status = "supported" if name in supported else "unsupported"
            rows.append(ControlPrediction(name, status, "held", name, 1.0))
        else:
            rows.append(ControlPrediction(name, "supported", "held", None, 1.0,
                                          deduplication_known=False))
    reward = composer(ControlPredictionTable("0" * 64, "1" * 64, "2" * 64, tuple(rows)))
    # Use a longer valid finite horizon for all 22 independent first uses.
    reward.horizon = 30
    positive = negative = 0.0
    for name in (*supported, *unsupported):
        assert name.startswith("key:")
        result = step(reward, PhysicalAction(keys=(name.removeprefix("key:"),)))
        if name in supported:
            assert result.control_bonus == pytest.approx(0.01 / 11)
        positive += result.control_bonus
        negative += result.unsupported_penalty
    assert positive == pytest.approx(0.01) and negative == pytest.approx(-0.01)
    assert reward._control_positive <= 0.01 and reward._control_negative <= 0.01
