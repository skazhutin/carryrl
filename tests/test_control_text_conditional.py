"""Versioned conditional heads and independent semantics remain fail-closed."""

import json
from dataclasses import asdict

import pytest
import torch
from test_control_text_grounding import ResourceEncoder, calibration, passing_report
from torch import nn

from carryrl.rewards.control_text import (
    CANDIDATES,
    CONDITIONAL_HEAD_ARCHITECTURE,
    CONDITIONAL_POPULATION_CONTRACT,
    INDEPENDENT_DECODER_VERSION,
    LEGACY_HEAD_ARCHITECTURE,
    ConditionalTextControlHead,
    ControlPrediction,
    ControlPredictionTable,
    FrozenTextControlReward,
    _checked_calibration,
    calibrated_head_logits,
    content_hash,
    load_text_control_reward,
    save_text_control_artifact,
    text_control_head_for_architecture,
)


def conditional_calibration():
    return {
        **calibration(),
        "population_contract": CONDITIONAL_POPULATION_CONTRACT,
        "decoder_version": INDEPENDENT_DECODER_VERSION,
    }


class FixedConditionalHead(nn.Module):
    architecture = CONDITIONAL_HEAD_ARCHITECTURE

    def __init__(self, *, supported=(), unsupported=(), same=(), uncertain=(), modes=None):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1))
        status = torch.full((1, 36, 3), -20.0)
        status[:, :, 2] = 20
        mode = torch.full((1, 36, 2), -20.0)
        mode[:, :, 0] = 20
        pair = torch.full((1, 36, 36, 2), -20.0)
        pair[:, :, :, 1] = 20
        for names, index in ((supported, 0), (unsupported, 1)):
            for name in names:
                status[0, CANDIDATES.index(name)] = -20
                status[0, CANDIDATES.index(name), index] = 20
        for names, value in ((same, [20.0, -20]), (uncertain, [0.0, 0])):
            for left, right in names:
                i, j = CANDIDATES.index(left), CANDIDATES.index(right)
                pair[0, i, j] = pair[0, j, i] = torch.tensor(value)
        for name, value in (modes or {}).items():
            mode[0, CANDIDATES.index(name)] = torch.tensor(value)
        self.output = {"status": status, "mode": mode, "pair": pair}

    def forward(self, tokens, *, grounding):
        assert grounding.shape[1] == len(CANDIDATES)
        return self.output


def fixed_model(**kwargs):
    return FrozenTextControlReward(
        ResourceEncoder(), FixedConditionalHead(**kwargs), model_hash="0" * 64,
        calibration=conditional_calibration(), calibration_hash="1" * 64, _verified=True,
    )


def prediction_map(model):
    return {row.candidate: row for row in model.predict("Hold A and D keys").predictions}


def test_conditional_shapes_pair_symmetry_and_factory():
    head = text_control_head_for_architecture(CONDITIONAL_HEAD_ARCHITECTURE)
    assert type(head) is ConditionalTextControlHead
    output = head(torch.randn(2, 5, 384), grounding=torch.zeros(2, 36, 5, dtype=torch.bool))
    assert output["status"].shape == (2, 36, 3)
    assert output["mode"].shape == (2, 36, 2)
    assert output["pair"].shape == (2, 36, 36, 2)
    assert torch.equal(output["pair"], output["pair"].transpose(1, 2))


def test_calibration_width_and_contract_checks_preserve_legacy_three_classes():
    for config, width in ((calibration(), 3), (conditional_calibration(), 2)):
        config["mode_log_class_weights"] = [0.1] * width
        config["pair_log_class_weights"] = [0.2] * width
        assert _checked_calibration(config) is config
        for kind in ("mode", "pair"):
            assert calibrated_head_logits(torch.zeros(2, width), config, kind).shape == (2, width)
            with pytest.raises(ValueError, match="class width"):
                calibrated_head_logits(torch.zeros(2, 5 - width), config, kind)
        config["mode_log_class_weights"] = [0.0] * (5 - width)
        with pytest.raises(ValueError, match="class_weights"):
            _checked_calibration(config)
    for key in ("population_contract", "decoder_version"):
        config = conditional_calibration()
        config.pop(key)
        with pytest.raises(ValueError, match="contract"):
            _checked_calibration(config)


def test_save_load_export_and_mismatched_head_contract_fails_before_writing(tmp_path):
    encoder = ResourceEncoder()
    head = ConditionalTextControlHead()
    with pytest.raises(ValueError, match="contract mismatch"):
        save_text_control_artifact(
            tmp_path / "bad", encoder=encoder, head=head, calibration=calibration(),
            validation=passing_report(), training={"trained": True},
        )
    assert not (tmp_path / "bad").exists()
    save_text_control_artifact(
        tmp_path / "fit", encoder=encoder, head=head, calibration=conditional_calibration(),
        validation=passing_report(), training={"trained": True},
    )
    restored = load_text_control_reward(tmp_path / "fit", shared_encoder=encoder)
    assert type(restored.head) is ConditionalTextControlHead
    expected = restored.predict("Hold A")
    exported = restored.save_resources(tmp_path / "export")
    again = load_text_control_reward(exported, shared_encoder=encoder)
    assert again.model_hash == restored.model_hash
    assert again.predict("Hold A") == expected
    with pytest.raises(FileExistsError):
        restored.save_resources(exported)
    # A manifest cannot redirect an authenticated resource outside the bundle.
    manifest_path = exported / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["../outside"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="resources"):
        load_text_control_reward(exported, shared_encoder=encoder)


def test_uncertain_relation_preserves_semantics_without_claiming_alias_or_reward():
    rows = prediction_map(fixed_model(
        supported=("key:a", "key:d"), uncertain=(("key:a", "key:d"),),
    ))
    for name in ("key:a", "key:d"):
        row = rows[name]
        assert row.status == "supported" and row.mode == "held" and row.confidence > 0.99
        assert row.group is None and not row.deduplication_known and not row.reward_eligible


def test_certified_aliases_and_supported_only_pair_population():
    rows = prediction_map(fixed_model(
        supported=("key:a", "key:d"), unsupported=("key:f", "key:q"),
        same=(("key:a", "key:d"),), uncertain=(("key:f", "key:q"),),
    ))
    assert rows["key:a"].group == rows["key:d"].group
    assert rows["key:a"].reward_eligible and rows["key:d"].reward_eligible
    assert rows["key:f"].group == "key:f" and rows["key:q"].group == "key:q"
    assert rows["key:f"].reward_eligible


@pytest.mark.parametrize("same,uncertain,modes", [
    ((("key:a", "key:d"), ("key:d", "key:w")), (), {}),
    ((("key:a", "key:d"),), (("key:d", "key:w"),), {}),
    ((("key:a", "key:d"),), (), {"key:d": [-20.0, 20]}),
])
def test_no_transitive_cross_group_or_inconsistent_mode_fallback(same, uncertain, modes):
    rows = prediction_map(fixed_model(
        supported=("key:a", "key:d", "key:w"), same=same, uncertain=uncertain, modes=modes,
    ))
    assert rows["key:a"].status == "supported" and rows["key:d"].status == "supported"
    assert not rows["key:a"].reward_eligible and not rows["key:d"].reward_eligible


def test_unknown_key_mode_retains_availability_abstention():
    row = prediction_map(fixed_model(supported=("key:a",), modes={"key:a": [0.0, 0]}))["key:a"]
    assert row.status == "unknown" and row.mode == "unknown" and not row.reward_eligible


def test_old_serialized_table_identity_and_new_roundtrip():
    old_rows = tuple(ControlPrediction(name) for name in CANDIDATES)
    table = ControlPredictionTable("0" * 64, "1" * 64, "2" * 64, old_rows)
    historical = asdict(table)
    for row in historical["predictions"]:
        row.pop("deduplication_known")
    assert table.to_dict() == historical and table.identity == content_hash(historical)
    fresh = fixed_model(supported=("key:a", "key:d"), uncertain=(("key:a", "key:d"),)).predict("A")
    assert ControlPredictionTable.from_dict(json.loads(json.dumps(fresh.to_dict()))) == fresh
    assert any(row.get("deduplication_known") is False for row in fresh.to_dict()["predictions"])
    with pytest.raises(ValueError, match="deduplication group"):
        ControlPrediction("key:a", "supported")
    with pytest.raises(ValueError, match="unresolved"):
        ControlPrediction("key:a", "supported", "held", "fake", deduplication_known=False)
    with pytest.raises(ValueError, match="boolean"):
        ControlPrediction("key:a", deduplication_known=0)


def test_three_class_legacy_decoder_retains_semantics_and_historical_identity():
    class LegacyHead(FixedConditionalHead):
        architecture = LEGACY_HEAD_ARCHITECTURE

        def __init__(self):
            super().__init__(supported=("key:a", "key:d"), unsupported=("key:f",),
                             uncertain=(("key:a", "key:d"),))
            for kind in ("mode", "pair"):
                self.output[kind] = torch.nn.functional.pad(self.output[kind], (0, 1), value=-20)

        def forward(self, tokens):
            return self.output

    model = FrozenTextControlReward(
        ResourceEncoder(), LegacyHead(), model_hash="0" * 64,
        calibration=calibration(), calibration_hash="1" * 64, _verified=True,
    )
    actual = model.predict("Hold A and D keys")
    # The legacy decoder still erases status on uncertain pair relations.
    expected = ControlPredictionTable(
        actual.text_hash, "0" * 64, "1" * 64,
        tuple(ControlPrediction(name, "unsupported" if name == "key:f" else "unknown",
                                "held" if name.startswith(("key:", "mouse:")) else "unknown",
                                "key:f" if name == "key:f" else None, 1.0)
              for name in CANDIDATES),
    )
    assert actual == expected and actual.identity == expected.identity
    assert all("deduplication_known" not in row for row in actual.to_dict()["predictions"])
