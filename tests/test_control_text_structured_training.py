"""Joint fitting keeps training, temperature choice and admission independent."""

import copy
import hashlib
import json
import time

import pytest
import torch
from torch import nn

from carryrl.rewards import control_text_runtime as runtime
from carryrl.rewards import control_text_structured_training as training
from carryrl.rewards.control_text import CANDIDATES
from carryrl.rewards.control_text_structured import (
    JOINT_STATE_COUNT,
    checked_structured_calibration,
    structured_targets,
)


def row(split, index=0):
    labels = [
        dict(candidate=candidate, status="unsupported", mode="unknown", alias_group=0)
        for candidate in CANDIDATES
    ]
    labels[0].update(status="supported", mode="held", alias_group=1)
    labels[1].update(status="supported", mode="held", alias_group=1)
    labels[2].update(status="supported", mode="press", alias_group=2)
    return dict(
        split=split,
        group_id=f"{split}:{index}",
        semantic_mapping_id=f"{split}:{index}",
        text=f"{split}:{index}",
        labels=labels,
    )


def records():
    return [
        row(split, i) for split in ("train", "dev", "calibration", "sealed_test") for i in range(4)
    ]


class TinyHead(nn.Module):
    architecture = training.STRUCTURED_HEAD_ARCHITECTURE

    def __init__(self):
        super().__init__()
        self.values = nn.Parameter(torch.randn(36, JOINT_STATE_COUNT) * 0.01)


class FrozenFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


def fake_forward(calls):
    def forward(features, head, rows, **kwargs):
        calls.append((head.training, torch.is_grad_enabled(), [r["split"] for r in rows]))
        return head.values[None].expand(len(rows), -1, -1)

    return forward


def test_fresh_fit_updates_after_validation_and_never_trains_heldout(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(training, "JointStructuredControlHead", TinyHead)
    monkeypatch.setattr(training, "_forward", fake_forward(calls))
    features = FrozenFeatures()
    _, report = training.fit_structured_head(
        features,
        records(),
        seed=7,
        epochs=2,
        batch_size=2,
        checkpoint_directory=tmp_path,
    )
    assert report["optimizer_initial_state_entries"] == 0
    assert report["optimizer_weight_decay"] == 0.01
    assert report["loss_protocol"] == training.LOSS_PROTOCOL
    epoch = report["history"][0]
    assert epoch["train_joint_decisions"] == 4 * 36
    assert epoch["train_objective_components"]["mode"]["denominator"] == 4 * 3
    assert epoch["train_objective"] != epoch["train_joint_nll"]
    assert report["optimizer_updates"] == 4
    assert report["train_examples"] == 8
    assert report["initial_head_sha256"]
    assert report["public_used_for_selection"] is False
    assert report["selection_splits"] == ["dev"]
    assert report["gradient_splits"] == ["train"]
    assert report["class_weights"] is report["prior_correction"] is None
    assert report["diagnostic_only"] is True
    assert not features.weight.requires_grad and features.weight.grad is None
    for _, gradients, splits in calls:
        assert set(splits) <= {"train", "dev"}
        if gradients:
            assert set(splits) == {"train"}
    first = torch.load(tmp_path / "epoch_01.pt", weights_only=True)
    second = torch.load(tmp_path / "epoch_02.pt", weights_only=True)
    assert not torch.equal(first["head"]["values"], second["head"]["values"])
    assert first["optimizer"]["state"] and second["optimizer"]["state"]
    assert second["optimizer_updates"] == 4
    assert (tmp_path / "selected_head.pt").is_file()


def test_grouped_dev_halves_keep_paraphrases_together():
    rows = records()
    rows += [{**r, "text": r["text"] + " paraphrase"} for r in rows if r["split"] == "dev"]
    temperature, score = training.split_development_groups(rows, seed=9)
    identities = training.development_split_identity(temperature, score)
    assert not set(identities["temperature_groups"]) & set(identities["score_groups"])
    assert len(temperature) == len(score) == 4
    assert all(r["split"] == "dev" for r in temperature + score)


def test_temperature_is_joint_nll_only_and_rejects_other_splits():
    rows = [row("dev")]
    targets = structured_targets(rows)
    logits = torch.zeros(1, 36, JOINT_STATE_COUNT)
    logits.scatter_(-1, targets[..., None], 4)
    calibration = training.fit_joint_temperature(logits, rows)
    assert calibration["joint_temperature"] == pytest.approx(0.1)
    assert calibration["joint_threshold"] == 0.99
    assert calibration["temperature_decisions"] == 36
    checked_structured_calibration(calibration)
    for split in ("train", "calibration", "sealed_test"):
        with pytest.raises(ValueError, match="development rows only"):
            training.fit_joint_temperature(logits, [row(split)])


def test_selection_uses_feasibility_first_then_score_nll_only():
    feasible = training._selection_key({"passed": True}, 100)
    infeasible = training._selection_key({"passed": False}, 0.0001)
    assert feasible > infeasible
    assert training._selection_key({"passed": True}, 0.3) > feasible


def test_deadline_before_forward_and_partial_checkpoint_after_interrupt(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(training, "JointStructuredControlHead", TinyHead)
    monkeypatch.setattr(training, "_forward", fake_forward(calls))
    with pytest.raises(TimeoutError, match="wall-clock"):
        training.fit_structured_head(FrozenFeatures(), records(), deadline=time.monotonic() - 1)
    assert not calls

    def interrupted(progress):
        raise TimeoutError("synthetic cutoff")

    with pytest.raises(TimeoutError, match="synthetic cutoff"):
        training.fit_structured_head(
            FrozenFeatures(),
            records(),
            epochs=2,
            batch_size=2,
            checkpoint_directory=tmp_path,
            progress=interrupted,
        )
    partial = torch.load(tmp_path / "partial_head.pt", weights_only=True)
    assert partial["optimizer_updates"] == 1 and partial["train_examples"] == 2
    assert partial["optimizer"]["state"]


def passed_probe(tmp_path, monkeypatch):
    identity = {
        "instructions_sha256": "instruction",
        "controls_sha256": "control",
        "specification_hash": "spec",
    }
    monkeypatch.setattr(runtime, "runtime_control_rows", lambda *args: ([], identity))
    checkpoint = tmp_path / "selected_head.pt"
    checkpoint.write_bytes(b"head checkpoint")
    report = {
        "status": "complete",
        "config": {
            "probe_only": True,
            "architecture": training.STRUCTURED_HEAD_ARCHITECTURE,
            "dataset_sha256": "dataset",
            "seed": 7,
            "source_identity": training.source_identity(),
            "encoder_state_sha256": "encoder",
            "learning_rate": 3e-4,
            "batch_size": 32,
            "optimizer_weight_decay": training.STRUCTURED_ADAMW_WEIGHT_DECAY,
            "loss_protocol": training.LOSS_PROTOCOL,
        },
        "feasibility": {"passed": True},
        "runtime": {
            "passed": True,
            "records": 32,
            "passed_records": 32,
            "results": [{"passed": True, "errors": []} for _ in range(32)],
            **identity,
        },
        "sealed_test_used": False,
        "selected_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    path = tmp_path / "probe_report.json"
    return path, report


def write_probe(path, report):
    path.write_text(json.dumps(report))
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n"
    )


def verify(path, seed=8):
    return training.verify_passed_probe(
        path,
        dataset_sha256="dataset",
        seed=seed,
        encoder_state_sha256="encoder",
        learning_rate=3e-4,
        batch_size=32,
        runtime_instructions="instructions",
        runtime_controls="controls",
    )


def test_full_requires_passed_authenticated_probe_and_fresh_seed(tmp_path, monkeypatch):
    path, report = passed_probe(tmp_path, monkeypatch)
    write_probe(path, report)
    assert verify(path)["seed"] == 7
    with pytest.raises(ValueError, match="fresh seed"):
        verify(path, seed=7)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        verify(path)


@pytest.mark.parametrize(
    "change",
    [
        "feasibility",
        "runtime",
        "source",
        "grounding",
        "dataset",
        "encoder",
        "checkpoint",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "old_loss",
    ],
)
def test_failed_or_changed_probe_cannot_license_full(tmp_path, monkeypatch, change):
    path, report = passed_probe(tmp_path, monkeypatch)
    report = copy.deepcopy(report)
    if change == "feasibility":
        report["feasibility"]["passed"] = False
    elif change == "runtime":
        report["runtime"]["passed_records"] = 31
    elif change == "source":
        report["config"]["source_identity"] = {}
    elif change == "grounding":
        report["config"]["source_identity"]["control_text_grounding.py"] = "changed"
    elif change == "encoder":
        report["config"]["encoder_state_sha256"] = "changed"
    elif change == "learning_rate":
        report["config"]["learning_rate"] = 2e-4
    elif change == "batch_size":
        report["config"]["batch_size"] = 16
    elif change == "old_loss":
        report["config"]["loss_protocol"] = "unweighted_joint_ce_v1"
    elif change == "weight_decay":
        report["config"]["optimizer_weight_decay"] = 0
    elif change == "dataset":
        report["config"]["dataset_sha256"] = "different"
    else:
        (tmp_path / "selected_head.pt").write_bytes(b"changed")
    write_probe(path, report)
    with pytest.raises(ValueError, match="passed structured probe"):
        verify(path)


def test_full_missing_probe_blocks_before_features_or_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        training, "read_control_dataset", lambda _: ({"records_sha256": "dataset"}, records())
    )
    with pytest.raises(ValueError, match="--passed-probe"):
        training.train_structured_text_control_artifact(
            "dataset",
            tmp_path / "full",
            encoder=None,
            probe_only=False,
        )
    assert not (tmp_path / "full").exists()


def test_probe_feature_extraction_excludes_calibration_and_sealed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        training, "read_control_dataset", lambda _: ({"records_sha256": "dataset"}, records())
    )
    monkeypatch.setattr(training, "verify_grounded_text_inputs", lambda tokenizer, rows, **kw: {})

    class Encoder(FrozenFeatures):
        tokenizer = object()

    def stop_after_features(encoder, rows, directory, **kwargs):
        assert {r["split"] for r in rows} == {"train", "dev"}
        raise TimeoutError("stop after checking cache population")

    monkeypatch.setattr(training, "prepare_disk_token_features", stop_after_features)
    with pytest.raises(TimeoutError, match="cache population"):
        training.train_structured_text_control_artifact(
            "dataset",
            tmp_path / "probe",
            encoder=Encoder(),
            probe_only=True,
        )
    status = json.loads((tmp_path / "probe" / "status.json").read_text())
    assert status["ready"] is False and status["encoder_frozen_verified"] is True


def population_rows(kind):
    result = row("train")
    if kind in ("none", "axes"):
        for label in result["labels"]:
            label.update(status="unsupported", mode="unknown", alias_group=0)
    if kind in ("axes", "mixed"):
        index = next(i for i, candidate in enumerate(CANDIDATES) if candidate.startswith("dx:"))
        result["labels"][index].update(status="supported", mode="unknown", alias_group=3)
    return [result]


def factorized_logits(batch=1):
    head = training.JointStructuredControlHead()
    state = torch.randn(batch, 36, 5, requires_grad=True)
    pointer = torch.randn(batch, 36, 36, requires_grad=True)
    supported = state[..., 2:, None] + pointer[..., None, :]
    logits = torch.cat((state[..., :2], supported.flatten(-2)), -1)
    return logits.masked_fill(~head.valid_states, -torch.inf), state, pointer


@pytest.mark.parametrize("kind", ["none", "keys", "axes", "mixed"])
def test_population_loss_backward_safe_for_exact_physical_masks_and_empty_groups(kind):
    rows = population_rows(kind)
    targets = structured_targets(rows)
    logits, state, pointer = factorized_logits()
    objective, components = training.conditional_population_loss(logits, targets)
    assert torch.isneginf(logits).any() and torch.isfinite(objective)
    assert all(torch.isfinite(total) for total, _ in components.values())
    assert components["status"][1] == 36
    assert components["representative"][1] == {"none": 0, "keys": 3, "axes": 1, "mixed": 4}[kind]
    assert components["mode"][1] == (3 if kind in ("keys", "mixed") else 0)
    if kind == "none":
        assert components["mode"][0].requires_grad
        assert components["representative"][0].requires_grad
        assert components["mode"][0] == components["representative"][0] == 0
    objective.backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert pointer.grad is not None and torch.isfinite(pointer.grad).all()


def test_population_objective_reweights_sparse_conditionals_and_keeps_exact_joint_nll():
    rows = population_rows("keys")
    targets = structured_targets(rows)
    logits, _, _ = factorized_logits()
    objective, components = training.conditional_population_loss(logits, targets)
    means = {name: total / count for name, (total, count) in components.items()}
    assert torch.allclose(objective, means["status"] + means["mode"] + means["representative"])
    # For the additive head, the conditional mode/pointer sums exactly decompose
    # original joint NLL. The new objective intentionally gives each mean weight 1.
    status_sum, status_count = components["status"]
    conditional_sum = components["mode"][0] + components["representative"][0]
    joint_sum, joint_count = components["joint"]
    assert joint_count == status_count == 36
    assert torch.allclose(joint_sum, status_sum + conditional_sum, atol=1e-5)
    assert torch.allclose(
        means["joint"], torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    )
    assert objective > means["joint"]


def test_population_loss_invalid_state_columns_have_zero_finite_gradient():
    targets = structured_targets(population_rows("mixed"))
    valid = training.JointStructuredControlHead().valid_states
    raw = torch.randn(1, 36, JOINT_STATE_COUNT, requires_grad=True)
    objective, _ = training.conditional_population_loss(
        raw.masked_fill(~valid, -torch.inf), targets
    )
    objective.backward()
    assert torch.isfinite(raw.grad).all()
    assert torch.count_nonzero(raw.grad[:, ~valid]) == 0


def test_population_loss_rejects_empty_batch():
    with pytest.raises(ValueError, match="nonempty"):
        training.conditional_population_loss(
            torch.empty(0, 36, JOINT_STATE_COUNT), torch.empty(0, 36, dtype=torch.long)
        )
