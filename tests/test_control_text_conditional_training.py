"""The v6 heads use exactly the supported decision populations throughout fitting."""

import copy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from carryrl.rewards import control_text_training as training
from carryrl.rewards.control_text import CANDIDATES, CONDITIONAL_HEAD_ARCHITECTURE


def _row(split="train", *, text="example"):
    labels = [dict(candidate=c, status="unsupported", mode="held", alias_group=0)
              for c in CANDIDATES]
    for index, mode, group in ((0, "held", 1), (1, "held", 1), (2, "press", 2),
                               (3, "unknown", 3), (35, "held", 4)):
        labels[index].update(status="supported", mode=mode, alias_group=group)
    labels[4].update(status="unknown", mode="press")
    return dict(text=text, split=split, labels=labels, group_id=split + text)


def _logits(rows, *, requires_grad=False):
    batch = len(rows)
    return {
        "status": torch.zeros(batch, 36, 3, requires_grad=requires_grad),
        "mode": torch.zeros(batch, 36, 2, requires_grad=requires_grad),
        "pair": torch.zeros(batch, 36, 36, 2, requires_grad=requires_grad),
    }


class _Head(nn.Module):
    architecture = CONDITIONAL_HEAD_ARCHITECTURE

    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))


def test_conditional_target_populations_exclude_irrelevant_and_diagonal_decisions():
    rows = [_row()]
    targets = training._targets(rows, "cpu", conditional=True)
    logits = _logits(rows)
    arrays = {kind: training._loss_arrays(logits, targets, kind, False)
              for kind in logits}
    assert arrays["status"][0].shape == (36, 3)
    assert arrays["mode"][1].tolist() == [0, 0, 1]
    assert arrays["pair"][0].shape == (10, 2)
    assert arrays["pair"][1].tolist().count(0) == 1
    assert arrays["pair"][1].tolist().count(1) == 9
    assert not targets["mode_mask"][0, 3]  # Supported, but unknown mode.
    assert not targets["mode_mask"][0, 4]  # Unknown availability.
    assert not targets["mode_mask"][0, 5]  # Explicitly unsupported.
    assert not targets["mode_mask"][0, 35]  # Axis, despite a known padded mode.
    with pytest.raises(ValueError, match="target masks"):
        training._loss_arrays(logits, training._targets(rows, "cpu"), "pair", True)


def test_conditional_train_counts_and_prior_match_the_masked_populations():
    rows = [_row()]
    balance = training.class_balance_from_train(rows, conditional=True)
    assert balance["counts"] == {"status": [5, 30, 1], "mode": [2, 1], "pair": [1, 9]}
    assert balance["population_contract"] == "supported_decisions.v1"
    prior = training._prior_correction(_Head(), rows)
    for kind, counts in balance["counts"].items():
        expected = [sum(counts) / (len(counts) * count) for count in counts]
        assert balance["weights"][kind] == pytest.approx(expected)
        assert prior[f"{kind}_log_class_weights"] == pytest.approx(
            [math.log(value) for value in expected]
        )
    with pytest.raises(ValueError, match="TRAIN"):
        training.class_balance_from_train([_row("dev")], conditional=True)
    rows[0]["labels"][2]["mode"] = "held"
    with pytest.raises(ValueError, match="both conditional classes"):
        training.class_balance_from_train(rows, conditional=True)


def test_empty_conditional_tasks_have_finite_differentiable_zero_and_no_fake_labels():
    row = _row()
    for label in row["labels"]:
        label["status"] = "unknown"
    logits = _logits([row], requires_grad=True)
    targets = training._targets([row], "cpu", conditional=True)
    components = training._conditional_loss_components(logits, targets)
    assert components["mode"][1] == components["pair"][1] == 0
    assert components["mode"][0] == components["pair"][0] == 0
    loss = training.head_loss(logits, targets)
    assert torch.isfinite(loss)
    assert float(loss.detach()) == pytest.approx(math.log(3))
    loss.backward()
    assert logits["status"].grad is not None
    for kind in ("mode", "pair"):
        assert logits[kind].grad is None or not logits[kind].grad.any()
    # The helper also safely handles an empty microbatch across every head.
    empty_logits = {kind: value[:0].detach().requires_grad_() for kind, value in logits.items()}
    empty_targets = {kind: value[:0] for kind, value in targets.items()}
    empty = training.head_loss(empty_logits, empty_targets)
    assert torch.isfinite(empty) and empty == 0
    empty.backward()
    assert all(value.grad is not None for value in empty_logits.values())


def test_evaluation_aggregates_by_decision_count_instead_of_padded_batch_size(monkeypatch):
    rows = [_row(text=str(index)) for index in range(3)]
    for label in rows[1]["labels"]:
        label["status"] = "unknown"
    rows[2]["labels"][1]["status"] = "unsupported"

    def forward(_encoder, _head, batch, _tokens, _mask):
        logits = _logits(batch)
        for index, row in enumerate(batch):
            for values in logits.values():
                values[index, ..., 0] = float(row["text"]) + 0.5
        return logits

    monkeypatch.setattr(training, "_batch", lambda *args: (None, None))
    monkeypatch.setattr(training, "_head_forward", forward)
    weights = training.class_balance_from_train([rows[0]], conditional=True)["weights"]
    head = _Head()
    small = training.evaluate_head_metrics(None, head, rows, batch_size=1, class_weights=weights)
    large = training.evaluate_head_metrics(None, head, rows, batch_size=3, class_weights=weights)
    expected = training.head_loss(
        forward(None, head, rows, None, None),
        training._targets(rows, "cpu", conditional=True), class_weights=weights,
    )
    assert small["selection_nll"] == pytest.approx(float(expected))
    assert small["selection_nll"] == pytest.approx(large["selection_nll"])
    assert small["decision_counts"] == {"status": 108, "mode": 5, "pair": 16}
    assert small["slices"] == large["slices"]
    assert "unknown" not in small["slices"]["mode"]
    assert "unweighted_supported_decisions_nll" in small
    assert "unweighted_all_pairs_nll" not in small


def test_empty_dev_conditional_slices_are_explicit_zero(monkeypatch):
    row = _row("dev")
    for label in row["labels"]:
        label["status"] = "unknown"
    monkeypatch.setattr(training, "_batch", lambda *args: (None, None))
    monkeypatch.setattr(training, "_head_forward", lambda *args: _logits([row]))
    metrics = training.evaluate_head_metrics(None, _Head(), [row])
    assert metrics["decision_counts"] == {"status": 36, "mode": 0, "pair": 0}
    assert metrics["head_nll"]["pair"] == {"selection_nll": 0.0, "unweighted_nll": 0.0}
    assert all(not count for values in metrics["slices"]["mode"].values()
               for count in values.values())


def test_temperatures_use_conditional_dev_decisions_and_preserve_train_prior(monkeypatch):
    rows = [_row(split) for split in ("train", "dev", "calibration")]
    calls = []
    original = training.F.cross_entropy

    def counted(logits, targets, *args, **kwargs):
        calls.append((len(targets), logits.shape[-1]))
        return original(logits, targets, *args, **kwargs)

    monkeypatch.setattr(training.F, "cross_entropy", counted)
    monkeypatch.setattr(training, "_batch", lambda *args: (None, None))
    monkeypatch.setattr(training, "_head_forward", lambda *args: _logits([rows[1]]))
    monkeypatch.setattr(training, "FrozenTextControlReward", lambda *args, **kwargs: None)
    monkeypatch.setattr(training, "validate_text_control_predictions",
                        lambda *args: {"gated": True})
    calibration, report = training.calibrate_text_control_head(None, _Head(), rows)
    assert calls.count((36, 3)) == calls.count((3, 2)) == calls.count((10, 2)) == 81
    assert calibration["temperature_decision_counts"] == {"status": 36, "mode": 3, "pair": 10}
    assert calibration["population_contract"] == "supported_decisions.v1"
    assert calibration["decoder_version"] == "independent_semantics_dedup.v2"
    assert calibration["pair_population"] == "gold_supported_unique_off_diagonal"
    assert calibration["mode_population"] == "gold_supported_discrete_known_mode"
    assert all(calibration[f"{kind}_threshold"] == 0.99 for kind in ("status", "mode", "pair"))
    assert len(calibration["status_log_class_weights"]) == 3
    assert len(calibration["mode_log_class_weights"]) == 2
    assert len(calibration["pair_log_class_weights"]) == 2
    assert report == {"gated": True}


def _predictor(row, *, conditional=True, groups=True):
    predictions = []
    for label in row["labels"]:
        status = label["status"]
        group = str(label["alias_group"]) if groups and status == "supported" else None
        predictions.append(SimpleNamespace(
            candidate=label["candidate"], status=status, mode=label["mode"], group=group,
            reward_eligible=(status == "supported" and group is not None
                             and (not label["candidate"].startswith(("key:", "mouse:"))
                                  or label["mode"] != "unknown")),
        ))
    return SimpleNamespace(
        calibration={"population_contract": "supported_decisions.v1"} if conditional else {},
        predict=lambda text: SimpleNamespace(predictions=predictions),
    )


def test_alias_abstention_keeps_all_gold_pairs_in_denominator_without_false_split():
    row = _row("calibration")
    predictor = _predictor(row, groups=False)
    report = training.validate_text_control_predictions(predictor, [row])
    assert report["aliases"] == {"clear_count": 1, "accepted_count": 0, "correct_count": 0,
                                 "false_split_errors": 0, "false_merge_errors": 0}
    assert report["clear_accepted"] == 35
    # Group abstention cannot suppress independent status/mode decisions.
    assert report["accepted_correct"] == 35
    assert report["modes"]["held"] == {"clear_count": 2, "accepted_count": 2, "correct_count": 2}
    assert report["modes"]["press"] == {"clear_count": 1, "accepted_count": 1,
                                          "correct_count": 1}
    assert not report["gates"]["passed"]


def test_mode_errors_are_critical_but_do_not_lower_status_precision():
    row = _row("calibration")
    predictor = _predictor(row, groups=False)
    predictor.predict("").predictions[0].mode = "press"
    report = training.validate_text_control_predictions(predictor, [row])
    assert report["accepted_correct"] == report["accepted_count"] == 35
    assert report["critical_errors"] == 1
    assert report["error_scenarios"] == report["signs"]["supported"]["error_scenarios"] == 1
    assert report["modes"]["held"]["correct_count"] == 1
    # A rejected availability claim must not count as an accepted conditional mode.
    predictor.predict("").predictions[0].status = "unsupported"
    report = training.validate_text_control_predictions(predictor, [row])
    assert report["modes"]["held"]["accepted_count"] == 1
    assert report["critical_errors"] == 1


def test_confirmed_wrong_alias_merges_and_splits_still_are_critical():
    row = _row("calibration")
    predictor = _predictor(row)
    predictions = predictor.predict("").predictions
    predictions[1].group = "split"
    report = training.validate_text_control_predictions(predictor, [row])
    assert report["aliases"]["false_split_errors"] == 1
    predictions[1].group = predictions[0].group
    predictions[2].group = predictions[0].group
    report = training.validate_text_control_predictions(predictor, [row])
    assert report["aliases"]["false_merge_errors"] == 2
    predictions[4].status = "supported"
    predictions[4].group = predictions[0].group
    predictions[4].reward_eligible = True
    report = training.validate_text_control_predictions(predictor, [row])
    assert report["aliases"]["false_merge_errors"] == 5
    assert report["critical_errors"] == 6  # Cross-unknown semantic error is also counted.


def test_legacy_target_population_and_validation_semantics_remain_unchanged():
    row = _row("calibration")
    targets = training._targets([row], "cpu")
    assert set(targets) == {"status", "mode", "pair"}
    logits = {kind: torch.zeros(1, 36, 3) for kind in ("status", "mode")}
    logits["pair"] = torch.zeros(1, 36, 36, 3)
    assert training._loss_arrays(logits, targets, "mode", True)[1].numel() == 36
    assert training._loss_arrays(logits, targets, "pair", True)[1].numel() == 630
    assert training._loss_arrays(logits, targets, "pair", False)[1].numel() == 1296
    # Older reports count absent groups as alias false splits and couple mode/status correctness.
    legacy_row = copy.deepcopy(row)
    legacy_row["labels"][3]["mode"] = "held"
    predictor = _predictor(legacy_row, conditional=False, groups=False)
    predictor.predict("").predictions[0].mode = "press"
    report = training.validate_text_control_predictions(predictor, [legacy_row])
    assert report["accepted_correct"] == 34
    assert report["aliases"]["accepted_count"] == 1
    assert report["aliases"]["false_split_errors"] == 1


def test_grounded_unweighted_nll_excludes_untrained_diagonals_but_legacy_keeps_them(monkeypatch):
    from carryrl.rewards.control_text import GROUNDED_HEAD_ARCHITECTURE, LEGACY_HEAD_ARCHITECTURE

    row = _row("dev")
    logits = {kind: torch.zeros(1, 36, 3) for kind in ("status", "mode")}
    logits["pair"] = torch.zeros(1, 36, 36, 3)
    indices = torch.arange(36)
    logits["pair"][:, indices, indices, 2] = 100.0
    monkeypatch.setattr(training, "_batch", lambda *args: (None, None))
    monkeypatch.setattr(training, "_head_forward", lambda *args: logits)
    head = _Head()
    head.architecture = GROUNDED_HEAD_ARCHITECTURE
    grounded = training.evaluate_head_metrics(None, head, [row], unique_pairs=True)
    assert grounded["unweighted_unique_off_diagonal_nll"] == pytest.approx(math.log(3))
    head.architecture = LEGACY_HEAD_ARCHITECTURE
    legacy = training.evaluate_head_metrics(None, head, [row])
    expected = training.head_loss(logits, training._targets([row], "cpu"))
    assert legacy["unweighted_all_pairs_nll"] == pytest.approx(float(expected))
    assert legacy["unweighted_all_pairs_nll"] > grounded["unweighted_unique_off_diagonal_nll"]
