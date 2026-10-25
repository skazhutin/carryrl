"""Joint canonical targets, admission, masking and text-only inference."""

import copy
import math

import pytest
import torch
from test_control_text import composer, step
from test_control_text_grounding import OffsetTokenizer
from torch import nn

from carryrl.agent.actions import PhysicalAction
from carryrl.rewards.control_text import CANDIDATES
from carryrl.rewards.control_text_structured import (
    JOINT_STATE_COUNT,
    STRUCTURED_DECODER_VERSION,
    STRUCTURED_HEAD_ARCHITECTURE,
    STRUCTURED_MODEL_VERSION,
    STRUCTURED_POPULATION_CONTRACT,
    UNKNOWN_STATE,
    UNSUPPORTED_STATE,
    JointStructuredControlHead,
    StructuredTextControlReward,
    _nearest_mention_offsets,
    checked_structured_calibration,
    decode_joint_state,
    decode_structured_predictions,
    joint_state,
    structured_targets,
)


def calibration(**updates):
    return {
        "architecture": STRUCTURED_HEAD_ARCHITECTURE,
        "decoder_version": STRUCTURED_DECODER_VERSION,
        "population_contract": STRUCTURED_POPULATION_CONTRACT,
        "joint_temperature": 1.0,
        "joint_threshold": .99,
        **updates,
    }


def labeled_row():
    labels = [dict(candidate=name, status="unknown", mode="unknown", alias_group=0)
              for name in CANDIDATES]
    return {"labels": labels}


def positive(row, candidate, group, mode="held"):
    row["labels"][CANDIDATES.index(candidate)].update(
        status="supported", mode=mode, alias_group=group
    )


def confident_logits(states):
    logits = torch.full((len(CANDIDATES), JOINT_STATE_COUNT), -30.0)
    logits[:, UNKNOWN_STATE] = 30
    for candidate, state in states.items():
        i = CANDIDATES.index(candidate)
        logits[i] = -30
        logits[i, state] = 30
    return logits


def decoded(states=None, *, logits=None, config=None):
    table = decode_structured_predictions(
        confident_logits(states or {}) if logits is None else logits,
        config or calibration(), "0" * 64, "1" * 64, "2" * 64,
    )
    return {prediction.candidate: prediction for prediction in table.predictions}, table


def state(candidate, mode="held", representative=None):
    return joint_state(mode, CANDIDATES.index(representative or candidate))


def test_stable_state_encoding_and_canonical_labels_ignore_group_numbers_and_order():
    assert JOINT_STATE_COUNT == 110
    assert decode_joint_state(UNKNOWN_STATE) == ("unknown", None)
    assert decode_joint_state(UNSUPPORTED_STATE) == ("unsupported", None)
    assert joint_state("relative", 35) == 109
    row = labeled_row()
    positive(row, "key:a", 94)
    positive(row, "key:d", 94)
    positive(row, "key:f", 2, "press")
    row["labels"][0].update(status="unsupported")
    expected = structured_targets([row])
    changed = copy.deepcopy(row)
    for label in changed["labels"]:
        label["alias_group"] = {0: 0, 94: 1, 2: 300}[label["alias_group"]]
    changed["labels"].reverse()
    assert torch.equal(structured_targets([changed]), expected)
    assert expected.shape == (1, 36) and expected.dtype == torch.long
    assert expected[0, CANDIDATES.index("key:a")] == state("key:a")
    assert expected[0, CANDIDATES.index("key:d")] == state("key:d", representative="key:a")
    assert expected[0, CANDIDATES.index("key:f")] == state("key:f", "press")
    assert expected[0, 0] == UNSUPPORTED_STATE
    assert expected[0, -1] == UNKNOWN_STATE
    assert structured_targets([]).shape == (0, 36)


@pytest.mark.parametrize("change", ["duplicate", "missing", "status", "group", "negative_group",
                                    "mode", "discrete_conflict", "missing_field_extra"])
def test_invalid_labels_rejected(change):
    row = labeled_row()
    positive(row, "key:a", 1)
    if change == "duplicate":
        row["labels"][1] = copy.deepcopy(row["labels"][0])
    elif change == "missing":
        row["labels"].pop()
    elif change == "status":
        row["labels"][0]["status"] = "plausible"
    elif change == "group":
        row["labels"][0]["alias_group"] = True
    elif change == "negative_group":
        row["labels"][0]["alias_group"] = 4
    elif change == "mode":
        positive(row, "key:a", 1, "unknown")
    elif change == "discrete_conflict":
        positive(row, "key:d", 1, "press")
    else:
        del row["labels"][0]["mode"]
        row["labels"][0]["extra"] = 1
    with pytest.raises(ValueError):
        structured_targets([row])


def test_exact_alias_singleton_unsupported_unknown_and_reward_once():
    predictions, table = decoded({
        "key:a": state("key:a"), "key:d": state("key:d", representative="key:a"),
        "key:f": state("key:f", "press"), "key:q": UNSUPPORTED_STATE,
    })
    assert predictions["key:a"].group == predictions["key:d"].group == "key:a"
    assert predictions["key:f"].group == "key:f"
    assert predictions["key:q"].group == "key:q"
    assert predictions["key:w"].status == "unknown"
    reward = composer(table)
    assert step(reward, PhysicalAction(keys=("a", "d"))).control_bonus == .001
    assert step(reward, PhysicalAction(keys=("d",))).control_bonus == 0
    assert step(reward, PhysicalAction(keys=("q",))).unsupported_penalty == -.001
    assert step(reward, PhysicalAction(keys=("q",))).unsupported_penalty == 0


@pytest.mark.parametrize("representative_state", [UNKNOWN_STATE, UNSUPPORTED_STATE, "uncertain"])
def test_unavailable_or_uncertain_representative_cannot_become_paying_singleton(
    representative_state,
):
    logits = confident_logits({"key:d": state("key:d", representative="key:a")})
    if representative_state == "uncertain":
        i = CANDIDATES.index("key:a")
        logits[i] = -30
        logits[i, state("key:a")] = 0
        logits[i, UNKNOWN_STATE] = 0
    else:
        logits = confident_logits({"key:a": representative_state,
                                   "key:d": state("key:d", representative="key:a")})
    predictions, table = decoded(logits=logits)
    unresolved = predictions["key:d"]
    assert unresolved.status == "supported" and not unresolved.deduplication_known
    assert unresolved.group is None and not unresolved.reward_eligible
    assert step(composer(table), PhysicalAction(keys=("d",))).control_bonus == 0


def test_cycles_and_noncanonical_forward_pointer_fail_closed():
    predictions, _ = decoded({
        "key:a": state("key:a", representative="key:d"),
        "key:d": state("key:d", representative="key:a"),
    })
    assert not predictions["key:a"].reward_eligible
    assert not predictions["key:d"].reward_eligible
    predictions, _ = decoded({
        "key:a": state("key:a", representative="key:d"), "key:d": state("key:d"),
    })
    assert not predictions["key:a"].reward_eligible
    assert not predictions["key:d"].reward_eligible


def test_uncertain_other_group_does_not_block_confident_group():
    predictions, _ = decoded({
        "key:a": state("key:a"), "key:d": state("key:d", representative="key:a"),
        "key:q": state("key:q", representative="key:f"),
    })
    assert predictions["key:a"].reward_eligible and predictions["key:d"].reward_eligible
    assert not predictions["key:q"].reward_eligible


def test_mixed_axes_and_discrete_aliases_only_compare_discrete_modes():
    row = labeled_row()
    positive(row, "key:a", 1, "press")
    positive(row, "dx:-", 1, "unknown")
    targets = structured_targets([row])[0]
    assert targets[CANDIDATES.index("dx:-")] == state("dx:-", "relative", "key:a")
    predictions, _ = decoded({name: int(targets[i]) for i, name in enumerate(CANDIDATES)})
    assert predictions["key:a"].mode == "press"
    assert predictions["dx:-"].mode == "unknown"
    assert predictions["dx:-"].group == predictions["key:a"].group
    assert predictions["dx:-"].reward_eligible
    predictions, _ = decoded({
        "key:a": state("key:a"), "key:d": state("key:d", "press", "key:a"),
    })
    assert not predictions["key:a"].reward_eligible
    assert not predictions["key:d"].reward_eligible


def test_joint_point_989_abstains_even_with_certain_support_status_and_pointer():
    logits = confident_logits({})
    index = CANDIDATES.index("key:a")
    logits[index] = -torch.inf
    logits[index, state("key:a")] = math.log(.989)
    logits[index, state("key:a", "press")] = math.log(.011)
    predictions, _ = decoded(logits=logits)
    assert predictions["key:a"].confidence == pytest.approx(.989)
    assert predictions["key:a"].status == "unknown"
    assert not predictions["key:a"].reward_eligible


def test_head_padding_mask_full_context_and_finite_gradients():
    torch.manual_seed(4)
    head = JointStructuredControlHead()
    tokens = torch.randn(2, 40, 384)
    mask = torch.zeros(2, 40, dtype=torch.bool)
    mask[1, 37:] = True
    grounding = torch.zeros(2, 36, 40, dtype=torch.bool)
    grounding[:, CANDIDATES.index("key:a"), 2] = True
    logits = head(tokens, mask, grounding=grounding)
    assert logits.shape == (2, 36, JOINT_STATE_COUNT)
    assert torch.isfinite(logits[:, :, :2]).all()
    assert not torch.isfinite(logits[:, 0, state("dx:-", "relative")]).any()
    row = labeled_row()
    positive(row, "key:a", 1)
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1), structured_targets([row, row]).flatten()
    )
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in head.parameters())
    changed = tokens.clone()
    changed[1, 37:] = float("nan")
    changed_logits = head(changed, mask, grounding=grounding)
    assert torch.equal(changed_logits, logits)
    # A token 37 positions from the mention remains in global cross attention.
    changed = tokens.clone()
    changed[:, 39] += 20
    changed_logits = head(changed, mask, grounding=grounding)
    assert not torch.allclose(changed_logits[0, :, :2], logits[0, :, :2])
    assert torch.equal(changed_logits[1], logits[1])


def test_local_context_uses_mode_neighbors_that_mention_mean_pooling_loses():
    head = JointStructuredControlHead()
    with torch.no_grad():
        head.local_query.weight.zero_()
        head.local_query.bias.zero_()
    # Synthetic contextual features for "hold A ... press D": key embeddings
    # stay fixed while the two surrounding mode features are interchanged.
    projected = torch.zeros(1, 9, 128)
    projected[0, 0, 0], projected[0, 7, 0] = 1, -1
    projected[0, [1, 8], 1] = 1
    grounding = torch.zeros(1, 36, 9, dtype=torch.bool)
    a, d = CANDIDATES.index("key:a"), CANDIDATES.index("key:d")
    grounding[0, a, 1] = grounding[0, d, 8] = True
    mask = torch.zeros(1, 9, dtype=torch.bool)
    identities = head.queries[None]
    local = head._local_context(projected, identities, grounding, mask)
    swapped = projected.clone()
    swapped[0, [0, 7], 0] *= -1
    changed = head._local_context(swapped, identities, grounding, mask)
    assert torch.equal(torch.bmm(grounding.float(), projected),
                       torch.bmm(grounding.float(), swapped))
    assert local[0, a, 0] > 0 and local[0, d, 0] < 0
    torch.testing.assert_close(changed[:, :, 0], -local[:, :, 0])
    assert torch.equal(local[0, CANDIDATES.index("key:q")], torch.zeros(128))
    # An equally sized change closer to A contributes more than a distant one.
    near, far = projected.clone(), projected.clone()
    near[0, 2, 2] = far[0, 6, 2] = 1
    near_context = head._local_context(near, identities, grounding, mask)
    far_context = head._local_context(far, identities, grounding, mask)
    assert near_context[0, a, 2] > far_context[0, a, 2] > 0


def test_local_attention_is_candidate_conditioned_with_shared_mentions():
    head = JointStructuredControlHead()
    with torch.no_grad():
        head.local_query.weight.copy_(torch.eye(128))
        head.local_query.bias.zero_()
    projected = torch.zeros(1, 3, 128)
    projected[0, 0, 0], projected[0, 2, 0] = -1, 1
    identities = torch.zeros(1, 36, 128)
    identities[0, 0, 0], identities[0, 1, 0] = -math.sqrt(128), math.sqrt(128)
    grounding = torch.zeros(1, 36, 3, dtype=torch.bool)
    grounding[0, :2, 1] = True
    local = head._local_context(
        projected, identities, grounding, torch.zeros(1, 3, dtype=torch.bool)
    )
    assert local[0, 0, 0] < 0 < local[0, 1, 0]


def test_nearest_mention_scans_cover_repeats_ties_and_missing_candidates():
    grounding = torch.zeros(1, 36, 7, dtype=torch.bool)
    grounding[0, 0, [1, 5]] = True
    grounding[0, 2, [0, 6]] = True
    offsets = _nearest_mention_offsets(grounding, torch.zeros(1, 7, dtype=torch.bool))
    assert offsets[0, 0].tolist() == [-1, 0, 1, 2, -1, 0, 1]
    assert offsets[0, 1].tolist() == [0] * 7
    assert offsets[0, 2].tolist() == [0, 1, 2, 3, -2, -1, 0]
    head = JointStructuredControlHead()
    with torch.no_grad():
        head.local_query.weight.zero_()
        head.local_query.bias.zero_()
    projected = torch.randn(1, 7, 128)
    local = head._local_context(
        projected, head.queries[None], grounding, torch.zeros(1, 7, dtype=torch.bool)
    )
    weights = (-offsets[0, 0].abs() * .25).softmax(-1)
    torch.testing.assert_close(local[0, 0], weights @ projected[0])
    assert torch.isfinite(local).all()


def test_head_missing_mentions_and_inserted_padding_are_invariant():
    torch.manual_seed(24)
    head = JointStructuredControlHead()
    tokens = torch.randn(2, 7, 384)
    grounding = torch.zeros(2, 36, 7, dtype=torch.bool)
    grounding[0, 0, [1, 5]] = True
    expected = head(tokens, grounding=grounding)
    real_positions = [2, 3, 5, 6, 7, 8, 9]
    padded = torch.full((2, 12, 384), float("nan"))
    padded[:, real_positions] = tokens
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[:, real_positions] = False
    padded_grounding = torch.ones(2, 36, 12, dtype=torch.bool)
    padded_grounding[:, :, real_positions] = grounding
    actual = head(padded, mask, grounding=padded_grounding)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(actual[:, :, :2]).all()


@pytest.mark.parametrize("which", ["grounding", "binary", "padding", "all_padding", "empty"])
def test_head_rejects_invalid_masks(which):
    head = JointStructuredControlHead()
    tokens = torch.randn(1, 3, 384)
    grounding = torch.zeros(1, 36, 3, dtype=torch.bool)
    mask = None
    if which == "grounding":
        grounding = None
    elif which == "binary":
        grounding = torch.full((1, 36, 3), .5)
    elif which == "padding":
        mask = torch.zeros(1, 3)
    elif which == "all_padding":
        mask = torch.ones(1, 3, dtype=torch.bool)
    else:
        tokens = tokens[:, :0]
    with pytest.raises(ValueError):
        head(tokens, mask, grounding=grounding)


@pytest.mark.parametrize("updates", [
    {"joint_temperature": 0}, {"joint_temperature": float("nan")},
    {"joint_threshold": .989}, {"joint_threshold": 1.01}, {"joint_threshold": True},
    {"decoder_version": "old"}, {"architecture": "old"},
    {"population_contract": "old"}, {"joint_log_class_weights": [0] * 110},
])
def test_calibration_rejects_incompatible_or_weakened_contract(updates):
    with pytest.raises(ValueError):
        checked_structured_calibration(calibration(**updates))


class TextOnlyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.ones(1))
        self.tokenizer = OffsetTokenizer()
        self.calls = []

    def encode_instruction_tokens(self, text):
        self.calls.append(text)
        return torch.ones(len("query: " + text) + 2, 384) * self.marker


class FixedHead(nn.Module):
    architecture = STRUCTURED_HEAD_ARCHITECTURE

    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, *, grounding):
        assert grounding.shape == (1, 36, tokens.shape[1])
        return confident_logits({"key:a": state("key:a")})[None]


def test_predictor_is_text_only_frozen_cached_and_has_distinct_identity():
    encoder, head = TextOnlyEncoder(), FixedHead()
    with pytest.raises(ValueError, match="verified"):
        StructuredTextControlReward(encoder, head, model_hash="1" * 64,
                                    calibration=calibration(), calibration_hash="2" * 64)
    model = StructuredTextControlReward(encoder, head, model_hash="1" * 64,
                                       calibration=calibration(), calibration_hash="2" * 64,
                                       _verified=True)
    table = model.predict("Hold A key")
    assert model.predict("Hold A key") is table
    assert encoder.calls == ["Hold A key"]
    assert not encoder.marker.requires_grad and not head.marker.requires_grad
    assert not encoder.training and not head.training
    with pytest.raises(ValueError, match="only a loaded"):
        model.save_resources("unused-unverified-destination")
    assert model.identity["version"] == STRUCTURED_MODEL_VERSION
    assert table.model_hash == "1" * 64 and table.calibration_hash == "2" * 64
    assert table.predictions[CANDIDATES.index("key:a")].reward_eligible
    for text in (None, "", "  "):
        with pytest.raises(ValueError):
            model.predict(text)
