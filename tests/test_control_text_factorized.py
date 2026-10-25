"""Partial independent semantics and conservative symmetric payout groups."""

import pytest
import torch

from carryrl.rewards.control_text import CANDIDATES
from carryrl.rewards.control_text_factorized import (
    FACTORIZED_DECODER_VERSION,
    FACTORIZED_HEAD_ARCHITECTURE,
    FACTORIZED_POPULATION_CONTRACT,
    FactorizedControlHead,
    checked_factorized_calibration,
    decode_factorized_predictions,
)


def calibration():
    return {
        "architecture": FACTORIZED_HEAD_ARCHITECTURE,
        "decoder_version": FACTORIZED_DECODER_VERSION,
        "population_contract": FACTORIZED_POPULATION_CONTRACT,
        **{
            f"{name}_{kind}": value
            for name in ("applicability", "mode", "pair")
            for kind, value in (("threshold", 0.99), ("temperature", 1.0))
        },
    }


def logits(supported=(), unsupported=(), modes=None, pairs=None):
    n = len(CANDIDATES)
    values = {
        "applicability": torch.full((n, 3), -30.0),
        "mode": torch.full((n, 4), -30.0),
        "pair": torch.full((n, n, 3), -30.0),
    }
    values["applicability"][:, 2] = 30
    values["mode"][:, 3] = 30
    values["pair"][:, :, 2] = 30
    for names, target in ((supported, 0), (unsupported, 1)):
        for name in names:
            values["applicability"][CANDIDATES.index(name)] = -30
            values["applicability"][CANDIDATES.index(name), target] = 30
    for name, mode in (modes or {}).items():
        i = CANDIDATES.index(name)
        values["mode"][i] = -30
        values["mode"][i, ("held", "press", "relative", "unspecified").index(mode)] = 30
    for (left, right), relation in (pairs or {}).items():
        i, j = CANDIDATES.index(left), CANDIDATES.index(right)
        k = ("same", "different", "unspecified").index(relation)
        values["pair"][i, j] = values["pair"][j, i] = -30
        values["pair"][i, j, k] = values["pair"][j, i, k] = 30
    return values


def decode(values):
    table = decode_factorized_predictions(values, calibration(), "0" * 64, "1" * 64, "2" * 64)
    return {p.candidate: p for p in table}, table


def test_partial_support_survives_unspecified_mode_and_unknown_aliases():
    rows, table = decode(logits(("key:a", "key:d"), ("key:q",)))
    assert rows["key:a"].status == "supported"
    assert rows["key:a"].mode == "unknown"
    assert rows["key:a"].mode_decision.reason == "insufficient_information"
    assert not rows["key:a"].reward_eligible and rows["key:a"].group is None
    assert rows["key:q"].reward_eligible
    assert rows["key:w"].applicability_decision.value == "unspecified"
    assert rows["key:w"].applicability_decision.reason == "insufficient_information"
    assert len(table.pair_decisions) == 630
    assert table.to_dict()["predictions"][CANDIDATES.index("key:a")]["status"] == "supported"


def test_singleton_requires_confident_different_from_every_supported_control():
    values = logits(
        ("key:a", "key:d", "key:f"),
        pairs={
            ("key:a", "key:d"): "same",
            ("key:a", "key:f"): "different",
            ("key:d", "key:f"): "different",
        },
    )
    rows, _ = decode(values)
    assert rows["key:a"].group == rows["key:d"].group == "key:a"
    assert rows["key:f"].group == "key:f"
    assert rows["key:a"].reward_eligible and not rows["key:a"].mode_reliable
    values["pair"][CANDIDATES.index("key:a"), CANDIDATES.index("key:f")] = 0
    values["pair"][CANDIDATES.index("key:f"), CANDIDATES.index("key:a")] = 0
    rows, _ = decode(values)
    assert all(not rows[name].reward_eligible for name in ("key:a", "key:d", "key:f"))


def test_nontransitive_same_blocks_bonus_preserves_support():
    values = logits(
        ("key:a", "key:d", "key:f"),
        pairs={
            ("key:a", "key:d"): "same",
            ("key:d", "key:f"): "same",
            ("key:a", "key:f"): "different",
        },
    )
    rows, table = decode(values)
    assert all(
        rows[name].status == "supported" and not rows[name].reward_eligible
        for name in ("key:a", "key:d", "key:f")
    )
    assert any(pair.decision.reason == "inconsistent" for pair in table.pair_decisions)


@pytest.mark.parametrize("other,mode", [("key:d", "press"), ("wheel:+", "relative")])
def test_functional_aliases_preserve_independent_physical_modes(other, mode):
    rows, table = decode(
        logits(
            ("key:a", other),
            modes={"key:a": "held", other: mode},
            pairs={("key:a", other): "same"},
        )
    )
    assert rows["key:a"].status == rows[other].status == "supported"
    assert rows["key:a"].reward_eligible and rows[other].reward_eligible
    assert rows["key:a"].group == rows[other].group
    assert rows["key:a"].mode == "held" and rows[other].mode == mode
    assert rows["key:a"].mode_reliable and rows[other].mode_reliable
    pair = next(pair for pair in table.pair_decisions
                if {pair.left, pair.right} == {"key:a", other})
    assert pair.decision.accepted and pair.decision.value == "same"


def test_low_confidence_is_distinct_from_semantic_unspecified():
    values = logits(("key:a",))
    values["applicability"][0] = 0
    rows, _ = decode(values)
    assert rows[CANDIDATES[0]].applicability_decision.reason == "low_confidence"
    assert rows["key:a"].reward_eligible
    assert rows["key:a"].mode_decision.reason == "insufficient_information"
    with pytest.raises(ValueError):
        checked_factorized_calibration({**calibration(), "mode_threshold": 0.98})


def test_head_is_symmetric_padding_invariant_and_differentiable():
    torch.manual_seed(18)
    head = FactorizedControlHead().eval()
    tokens = torch.randn(2, 5, 384)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    ground = torch.zeros(2, 36, 5, dtype=torch.bool)
    ground[:, 0, 2] = True
    values = head(tokens, mask, grounding=ground)
    assert {k: v.shape for k, v in values.items()} == {
        "applicability": (2, 36, 3),
        "mode": (2, 36, 4),
        "pair": (2, 36, 36, 3),
    }
    assert torch.equal(values["pair"], values["pair"].transpose(1, 2))
    padded = head(
        torch.cat((tokens, torch.randn(2, 3, 384)), 1),
        torch.cat((mask, torch.ones(2, 3, dtype=torch.bool)), 1),
        grounding=torch.cat((ground, torch.zeros(2, 36, 3, dtype=torch.bool)), -1),
    )
    for kind in values:
        torch.testing.assert_close(values[kind], padded[kind], atol=2e-6, rtol=2e-6)
    sum(v.sum() for v in values.values()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
