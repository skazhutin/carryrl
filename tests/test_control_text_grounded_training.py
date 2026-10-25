"""Training/runtime use the same textual anchors and authenticated v4 splits."""

import math

import pytest
import torch
from torch import nn
from transformers import AutoTokenizer

from carryrl.agent.language import TEXT_ID, TEXT_REVISION
from carryrl.rewards.control_text import (
    CANDIDATES,
    GROUNDED_HEAD_ARCHITECTURE,
    LOCAL_GROUNDED_HEAD_ARCHITECTURE,
    PAIR_GEOMETRY_HEAD_ARCHITECTURE,
)
from carryrl.rewards.control_text_data_v4 import prepare_control_text_dataset
from carryrl.rewards.control_text_training import (
    _batch,
    _head_forward,
    class_balance_from_train,
    fit_text_control_head,
    read_control_dataset,
)


def test_v4_shared_grammar_keeps_semantic_and_heldout_family_guards(tmp_path):
    import hashlib
    import json

    dataset = tmp_path / "data"
    prepare_control_text_dataset(dataset, scenario_groups=100)
    manifest, rows = read_control_dataset(dataset)
    assert not manifest["template_families_disjoint"]
    assert {r["split"] for r in rows if r["template_family"] == "shared-core"} == {
        "train", "dev", "calibration", "sealed_test"
    }
    # Shared grammar never licenses sharing the effective control mapping.
    train = next(row for row in rows if row["split"] == "train")
    dev = next(row for row in rows if row["split"] == "dev")
    dev["semantic_mapping_id"] = train["semantic_mapping_id"]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    (dataset / "records.jsonl").write_bytes(payload)
    manifest["records_sha256"] = hashlib.sha256(payload).hexdigest()
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity mismatch|split leakage"):
        read_control_dataset(dataset)


def test_v5_wording_only_dataset_loads_with_semantic_guards(tmp_path):
    from carryrl.rewards.control_text_data_v5 import prepare_control_text_dataset

    dataset = tmp_path / "v5"
    prepare_control_text_dataset(dataset, scenario_groups=100)
    manifest, rows = read_control_dataset(dataset)
    assert manifest["version"] == "carryrl.control_text_data.v5"
    assert len(rows) == 400
    assert manifest["semantic_mapping_disjoint"]
    assert manifest["lexical_primitives_shared_across_splits"]
    assert manifest["template_families_disjoint"]


class TokenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1))
        self.tokenizer = AutoTokenizer.from_pretrained(
            TEXT_ID, revision=TEXT_REVISION, local_files_only=True,
        )
        self.seen = []

    def encode_instruction_tokens(self, text):
        self.seen.append(text)
        ids = self.tokenizer("query: " + text)["input_ids"]
        return torch.sin(torch.tensor(ids)[:, None] + torch.arange(384)[None] / 384)


@pytest.mark.parametrize("variant", [
    "grounded_balanced_v3", "local_grounded_balanced_v4", "pair_geometry_balanced_v5",
])
def test_grounded_fit_passes_text_anchors_and_records_train_prior_without_sealed_input(
    tmp_path, variant,
):
    encoder = TokenEncoder()
    labels = [dict(candidate=c, status="unsupported", mode="unknown", alias_group=0)
              for c in CANDIDATES]
    for c in ("key:w", "key:up"):
        labels[CANDIDATES.index(c)].update(status="supported", mode="held", alias_group=1)
    labels[CANDIDATES.index("key:space")].update(
        status="supported", mode="press", alias_group=2,
    )
    labels[CANDIDATES.index("key:q")].update(status="unknown")
    text = ("Hold the W key or the Up key. Tap Space and release. The Q key is conditional; "
            "its availability is unknown. Other inputs are unsupported.")
    rows = [dict(text=split + ": " + text, split=split, labels=labels)
            for split in ("train", "dev", "sealed_test")]
    head, training = fit_text_control_head(
        encoder, rows, epochs=1, head_variant=variant,
        failure_checkpoint=tmp_path / "partial.pt", diagnostic_updates=(1,),
    )
    expected = {"grounded_balanced_v3": GROUNDED_HEAD_ARCHITECTURE,
                "local_grounded_balanced_v4": LOCAL_GROUNDED_HEAD_ARCHITECTURE,
                "pair_geometry_balanced_v5": PAIR_GEOMETRY_HEAD_ARCHITECTURE}[variant]
    assert head.architecture == expected
    if variant == "pair_geometry_balanced_v5":
        from carryrl.rewards.control_text import LocalGroundedTextControlHead, encoder_state_hash

        torch.manual_seed(20261020)
        expected_common = encoder_state_hash(LocalGroundedTextControlHead().state_dict())
        assert training["initial_common_local_parameters_sha256"] == expected_common
    saved = torch.load(tmp_path / "diagnostic_update_1.pt", weights_only=True)
    assert saved["diagnostic_only"] and not saved["trained"]
    assert saved["updates"] == 1
    assert all(not value.startswith("sealed_test:") for value in encoder.seen)
    weights = class_balance_from_train(rows[:1])["weights"]
    for kind in weights:
        assert training["deployment_logit_prior_correction"][f"{kind}_log_class_weights"] == (
            pytest.approx([math.log(v) for v in weights[kind]])
        )
    assert training["history"][0]["train_fit_loss"] > 0
    assert training["history"][0]["train_probe_metrics"]["records"] == 1
    tokens, mask = _batch(encoder, rows[:2], "cpu")
    output = _head_forward(encoder, head, rows[:2], tokens, mask)
    assert output["status"].shape == (2, 36, 3)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_grounded_temperature_uses_runtime_pair_population_and_train_prior(monkeypatch):
    from carryrl.rewards import control_text_training as training
    from carryrl.rewards.control_text import GroundedTextControlHead

    encoder = TokenEncoder()
    labels = [dict(candidate=c, status="unsupported", mode="unknown", alias_group=0)
              for c in CANDIDATES]
    labels[0].update(status="supported", mode="held", alias_group=1)
    labels[1].update(status="supported", mode="held", alias_group=1)
    labels[2].update(status="supported", mode="press", alias_group=2)
    labels[-1].update(status="unknown")
    rows = [dict(text="Hold the Up key. Tap Left and release.", labels=labels,
                 split=split, group_id=split)
            for split in ("train", "dev", "calibration")]
    decision_counts = []
    original = training.F.cross_entropy

    def counted(logits, targets, *args, **kwargs):
        decision_counts.append(len(targets))
        return original(logits, targets, *args, **kwargs)

    monkeypatch.setattr(training.F, "cross_entropy", counted)
    # This test inspects development-temperature fitting; final admission remains
    # separately covered and must not trigger a second inference population.
    monkeypatch.setattr(training, "validate_text_control_predictions", lambda *args: {})
    calibration, _ = training.calibrate_text_control_head(encoder, GroundedTextControlHead(), rows)
    assert decision_counts.count(36 * 35 // 2) == 81
    assert 36 * 36 not in decision_counts
    assert calibration["pair_population"] == "unique_off_diagonal"
    weights = class_balance_from_train(rows[:1])["weights"]
    for kind in weights:
        assert calibration[f"{kind}_log_class_weights"] == pytest.approx(
            [math.log(v) for v in weights[kind]]
        )


def test_throughput_probe_does_not_extrapolate_cold_initialization_to_every_step():
    from carryrl.rewards.control_text_training import _head_throughput_probe

    probe = _head_throughput_probe([2.0] + [0.05] * 8,
                                   planned_batches=40000, completed_updates=9)
    assert probe["cold_step_seconds"] == 2.0
    assert probe["steady_step_seconds"] == 0.05
    assert probe["elapsed_seconds"] == pytest.approx(2.4)
    assert probe["estimated_fit_seconds"] == pytest.approx(2001.95)
    # A single-batch fit still accounts for its actual initialization cost.
    tiny = _head_throughput_probe([2.0], planned_batches=3, completed_updates=1)
    assert tiny["estimated_fit_seconds"] == 6.0


def test_development_selection_scores_raw_weighted_objective_before_prior_correction(monkeypatch):
    from carryrl.rewards import control_text_training as training

    head = nn.Linear(1, 1)
    logits = {kind: torch.tensor([[[1.0, 0.0, -1.0]]]).expand(1, 36, 3).clone()
              for kind in ("status", "mode")}
    logits["pair"] = logits["status"][:, :, None].expand(1, 36, 36, 3).clone()
    labels = [dict(candidate=c, status="unsupported", mode="unknown", alias_group=0)
              for c in CANDIDATES]
    rows = [dict(text="unused", labels=labels)]
    weights = {kind: [3.0, 0.5, 1.0] for kind in logits}
    correction = {f"{kind}_log_class_weights": [math.log(v) for v in value]
                  for kind, value in weights.items()}
    monkeypatch.setattr(training, "_batch", lambda *args: (None, None))
    monkeypatch.setattr(training, "_head_forward", lambda *args: logits)
    report = training.evaluate_head_metrics(None, head, rows, class_weights=weights,
                                           unique_pairs=True, log_class_weights=correction)
    targets = training._targets(rows, "cpu")
    expected = training.head_loss(logits, targets, class_weights=weights, unique_pairs=True)
    corrected = {kind: value - torch.tensor(correction[f"{kind}_log_class_weights"])
                 for kind, value in logits.items()}
    wrong = training.head_loss(corrected, targets, class_weights=weights, unique_pairs=True)
    assert report["selection_nll"] == pytest.approx(float(expected))
    assert report["selection_nll"] != pytest.approx(float(wrong))


def test_grounded_input_preflight_checks_supervision_before_expensive_fit_and_skips_sealed():
    from carryrl.rewards.control_text_training import verify_grounded_text_inputs

    tokenizer = TokenEncoder().tokenizer
    labels = [dict(candidate=c, status="unsupported", mode="unknown", alias_group=0)
              for c in CANDIDATES]
    labels[CANDIDATES.index("key:w")].update(status="supported", mode="held", alias_group=1)
    row = dict(split="train", group_id="manual", labels=labels,
               text="Hold the W key. Other inputs are unsupported.",
               specification={"operations": [{"candidates": ["key:w"]}], "uncertain": []})
    sealed = {**row, "split": "sealed_test", "text": "no named inputs"}
    report = verify_grounded_text_inputs(tokenizer, [row, sealed])
    assert report["records"] == 1 and report["missing_mentions"] == 0
    assert not report["sealed_inputs_used"]
    with pytest.raises(ValueError, match="lost named control mentions"):
        verify_grounded_text_inputs(tokenizer, [{**row, "text": "No named inputs."}])
    # Unknown explicitly named inputs must also be transmitted; their presence
    # must not become a positive-status label through a parser shortcut.
    uncertain = {**row, "specification": {"operations": [], "uncertain": ["key:q"]}}
    with pytest.raises(ValueError, match="key:q"):
        verify_grounded_text_inputs(tokenizer, [uncertain])
