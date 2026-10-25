"""Independent population masking, explicit DEV ownership and composition metrics."""

import copy
import math

import pytest
import torch
from test_control_text_factorized import calibration, decode, logits

from carryrl.rewards.control_text import CANDIDATES
from carryrl.rewards.control_text_factorized_training import (
    FactorizedLogitPredictor,
    factorized_population_loss,
    factorized_required_margin_loss,
    factorized_required_pair_mask,
    factorized_targets,
    fit_factorized_temperatures,
    split_factorized_development,
    validate_factorized_predictions,
)


def row():
    labels = [
        dict(candidate=c, applicability="unspecified", mode="unspecified") for c in CANDIDATES
    ]
    labels[CANDIDATES.index("key:a")].update(applicability="supported", mode="held")
    labels[CANDIDATES.index("key:d")].update(applicability="supported")
    labels[CANDIDATES.index("key:q")].update(applicability="unsupported")
    return dict(
        text="A and D",
        group_id="g",
        semantic_mapping_id="s",
        split="train",
        factorized_labels=labels,
        pair_labels=[dict(left="key:a", right="key:d", relation="unspecified")],
    )


def test_target_populations_preserve_supported_unknown_mode_and_mask_inapplicable_pairs():
    target = factorized_targets([row()])
    assert target["applicability"].shape == (1, 36)
    assert target["mode"][0, CANDIDATES.index("key:d")] == 3
    assert target["mode"][0, CANDIDATES.index("key:q")] == -100
    assert target["pair"][0, CANDIDATES.index("key:a"), CANDIDATES.index("key:d")] == 2
    assert int((target["pair"] != -100).sum()) == 2
    bad = row()
    bad["pair_labels"] = []
    with pytest.raises(ValueError, match="applicable unordered pair"):
        factorized_targets([bad])


def test_population_loss_counts_each_pair_once_and_produces_finite_gradients():
    values = {k: v[None].clone().requires_grad_() for k, v in logits(("key:a", "key:d")).items()}
    loss, components = factorized_population_loss(values, factorized_targets([row()]))
    assert components["applicability"][1] == 36
    assert components["mode"][1] == 2
    assert components["pair"][1] == 1
    expected = sum(total / count for kind, (total, count) in components.items() if kind != "joint")
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert all(torch.isfinite(v.grad).all() for v in values.values())
    changed = {k: v.detach().clone() for k, v in values.items()}
    changed["pair"][:, 0, 1] = 1e3
    torch.testing.assert_close(
        loss.detach(), factorized_population_loss(changed, factorized_targets([row()]))[0]
    )


def test_dev_roles_are_explicit_disjoint_and_temperature_cannot_see_selection():
    records = [
        dict(
            group_id=f"{role}-{i}",
            semantic_mapping_id=f"{role}-{i}",
            split="dev",
            development_role=role,
        )
        for role in ("temperature", "selection")
        for i in range(300)
    ]
    temperature, selection = split_factorized_development(records)
    assert len(temperature) == len(selection) == 300
    records[0]["development_role"] = "selection"
    with pytest.raises(ValueError, match="300/300"):
        split_factorized_development(records)
    bad = row()
    bad.update(split="dev", development_role="selection")
    with pytest.raises(ValueError, match="temperature DEV"):
        fit_factorized_temperatures({}, [bad])


def test_actual_payable_coverage_blocks_semantically_supported_but_unresolved_pairs():
    record = row()
    values = {
        k: v[None]
        for k, v in logits(("key:a", "key:d"), ("key:q",), modes={"key:a": "held"}).items()
    }
    report = validate_factorized_predictions(
        FactorizedLogitPredictor([record], values, calibration()), [record]
    )
    assert report["signs"]["supported"]["clear_count"] == 2
    assert report["signs"]["supported"]["clear_accepted"] == 0
    assert report["signs"]["unsupported"]["clear_accepted"] == 1
    assert report["critical_errors"] == 0
    assert not report["gates"]["passed"]
    changed = copy.deepcopy(record)
    changed["pair_labels"][0]["relation"] = "same"
    values = {
        k: v[None]
        for k, v in logits(
            ("key:a", "key:d"),
            ("key:q",),
            modes={"key:a": "held"},
            pairs={("key:a", "key:d"): "different"},
        ).items()
    }
    report = validate_factorized_predictions(
        FactorizedLogitPredictor([changed], values, calibration()), [changed]
    )
    assert report["critical_errors"] > 0


def test_wrong_unpayable_field_is_reported_without_inventing_precision_denominator():
    record = row()
    values = {k: v[None] for k, v in logits(("key:a", "key:d"), modes={"key:a": "press"}).items()}
    report = validate_factorized_predictions(
        FactorizedLogitPredictor([record], values, calibration()), [record]
    )
    assert report["critical_errors"] == 1
    assert report["accepted_scenarios"] == report["error_scenarios"] == 0
    assert report["gates"]["passed"] is False


def relational_row(state="same", *, split="train"):
    value = row()
    value["split"] = split
    value["factorized_labels"][CANDIDATES.index("key:f")].update(
        applicability="supported", mode="press"
    )
    value["pair_labels"] = [
        dict(left="key:a", right="key:d", relation=state),
        dict(left="key:a", right="key:f", relation="different"),
        dict(left="key:d", right="key:f", relation="different"),
    ]
    value["relational_panel"] = {
        "version": "carryrl.relational_panel.v1",
        "focal_pair": ["key:d", "key:a"],
        "state": state,
    }
    return value


def loss_values(batch=1):
    generator = torch.Generator().manual_seed(32)
    return {
        "applicability": torch.randn(batch, 36, 3, generator=generator, dtype=torch.float64),
        "mode": torch.randn(batch, 36, 4, generator=generator, dtype=torch.float64),
        "pair": torch.randn(batch, 36, 36, 3, generator=generator, dtype=torch.float64),
    }


def natural_mean(components):
    return sum(
        total / count for name, (total, count) in components.items() if name != "joint" and count
    )


@pytest.mark.parametrize("state", ["same", "different", "unspecified"])
def test_relational_train_targets_mark_one_symmetric_supported_focal_pair(state):
    targets = factorized_targets([row(), relational_row(state)])
    focal = targets["pair_focal_mask"]
    left, right = CANDIDATES.index("key:a"), CANDIDATES.index("key:d")
    assert focal.shape == (2, 36, 36) and focal.dtype == torch.bool
    assert not focal[0].any()
    assert int(focal[1].sum()) == 2
    assert focal[1, left, right] and focal[1, right, left]
    assert torch.equal(focal, focal.transpose(-1, -2))
    assert targets["pair"][1, left, right] == ("same", "different", "unspecified").index(state)
    assert factorized_targets([])["pair_focal_mask"].shape == (0, 36, 36)
    assert factorized_targets([])["pair_focal_mask"].dtype == torch.bool


@pytest.mark.parametrize("split", ["dev", "calibration", "sealed_test", None])
def test_nontrain_relational_annotations_fail_closed(split):
    with pytest.raises(ValueError, match="TRAIN only"):
        factorized_targets([relational_row(split=split)])


@pytest.mark.parametrize(
    "change",
    [
        "null",
        "not_mapping",
        "version",
        "state",
        "state_mismatch",
        "missing_pair",
        "one_member",
        "three_members",
        "duplicate",
        "unknown_command",
        "inapplicable",
    ],
)
def test_malformed_relational_focal_annotation_fails_closed(change):
    value = relational_row()
    panel = value["relational_panel"]
    if change == "null":
        value["relational_panel"] = None
    elif change == "not_mapping":
        value["relational_panel"] = []
    elif change == "version":
        panel["version"] = "carryrl.relational_panel.v0"
    elif change == "state":
        panel["state"] = "unknown"
    elif change == "state_mismatch":
        panel["state"] = "different"
    elif change == "missing_pair":
        del panel["focal_pair"]
    elif change == "one_member":
        panel["focal_pair"] = ["key:a"]
    elif change == "three_members":
        panel["focal_pair"] = ["key:a", "key:d", "key:f"]
    elif change == "duplicate":
        panel["focal_pair"] = ["key:a", "key:a"]
    elif change == "unknown_command":
        panel["focal_pair"] = ["key:a", "key:invented"]
    else:
        value["factorized_labels"][CANDIDATES.index("key:d")].update(
            applicability="unsupported", mode="unspecified"
        )
        panel["state"] = "unspecified"
        for pair in value["pair_labels"]:
            if "key:d" in (pair["left"], pair["right"]):
                pair["relation"] = "unspecified"
    with pytest.raises(ValueError, match="relational focal"):
        factorized_targets([value])


def test_pair_rehearsal_uses_half_means_and_retains_natural_metrics():
    targets = factorized_targets([relational_row(), row()])
    values = loss_values(2)
    values = {name: value.requires_grad_() for name, value in values.items()}
    loss, components = factorized_population_loss(values, targets)
    mask = (targets["pair"] != -100) & torch.ones_like(targets["pair"], dtype=torch.bool).triu(1)
    focal = targets["pair_focal_mask"] & mask
    other = mask & ~focal
    ce = torch.nn.functional.cross_entropy
    expected_pair = 0.5 * ce(values["pair"][focal], targets["pair"][focal])
    expected_pair += 0.5 * ce(values["pair"][other], targets["pair"][other])
    expected = sum(components[name][0] / components[name][1] for name in ("applicability", "mode"))
    torch.testing.assert_close(loss, expected + expected_pair)
    legacy_targets = {key: value for key, value in targets.items() if key != "pair_focal_mask"}
    natural_loss, natural_components = factorized_population_loss(values, legacy_targets)
    assert not torch.isclose(loss, natural_loss)
    assert components.keys() == {"applicability", "mode", "pair", "joint"}
    for name in components:
        assert components[name][1] == natural_components[name][1]
        assert torch.equal(components[name][0], natural_components[name][0])
    assert torch.equal(natural_mean(components), natural_loss)
    actual_grads = torch.autograd.grad(loss, tuple(values.values()), retain_graph=True)
    expected_grads = torch.autograd.grad(expected + expected_pair, tuple(values.values()))
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, reference)
        assert torch.isfinite(actual).all()
    pair_grad = actual_grads[-1]
    assert not pair_grad.masked_select(~mask[..., None]).any()


def test_no_focal_mask_keeps_exact_legacy_loss_and_gradients():
    targets = factorized_targets([row()])
    values = {name: value.requires_grad_() for name, value in loss_values().items()}
    loss, components = factorized_population_loss(values, targets)
    legacy = {key: value for key, value in targets.items() if key != "pair_focal_mask"}
    reference, _ = factorized_population_loss(values, legacy)
    assert torch.equal(loss, reference)
    assert torch.equal(loss, natural_mean(components))
    gradients = torch.autograd.grad(loss, tuple(values.values()), retain_graph=True)
    reference_gradients = torch.autograd.grad(reference, tuple(values.values()))
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(gradients, reference_gradients, strict=True)
    )


def test_only_focal_population_uses_full_mean_without_halving():
    value = row()
    value["pair_labels"][0]["relation"] = "same"
    value["relational_panel"] = relational_row()["relational_panel"]
    targets = factorized_targets([value])
    values = loss_values()
    loss, components = factorized_population_loss(values, targets)
    assert torch.equal(loss, natural_mean(components))
    assert components["pair"][1] == 1


@pytest.mark.parametrize(
    "change", ["dtype", "shape", "asymmetric", "diagonal", "multiple", "masked"]
)
def test_invalid_direct_focal_masks_fail_closed(change):
    targets = factorized_targets([relational_row()])
    focal = targets["pair_focal_mask"]
    left, right, third = (CANDIDATES.index(candidate) for candidate in ("key:a", "key:d", "key:f"))
    if change == "dtype":
        targets["pair_focal_mask"] = focal.float()
    elif change == "shape":
        targets["pair_focal_mask"] = focal[0]
    elif change == "asymmetric":
        focal[0, right, left] = False
    elif change == "diagonal":
        focal[0, left, left] = True
    elif change == "multiple":
        focal[0, left, third] = focal[0, third, left] = True
    else:
        targets["pair"][0, left, right] = targets["pair"][0, right, left] = -100
    with pytest.raises(ValueError, match="pair focal mask"):
        factorized_population_loss(loss_values(), targets)


def test_new_protocol_pins_training_rehearsal_and_natural_development_objective():
    from carryrl.rewards.control_text_factorized_training import LOSS_PROTOCOL, protocol_identity

    protocol = protocol_identity()
    assert protocol["loss_protocol"] == LOSS_PROTOCOL
    assert LOSS_PROTOCOL != "independent_factorized_population_means.v1"
    assert protocol["focal_metadata"] == "TRAIN_relational_panel.v1_only"
    assert protocol["development_objective"] == "natural_population_means"


def graph_record(supported, relations):
    value = row()
    value["id"] = "gold-fixture"
    for label in value["factorized_labels"]:
        label.update(
            applicability="supported" if label["candidate"] in supported else "unspecified",
            mode="held" if label["candidate"] in supported else "unspecified",
        )
    relations = {frozenset(pair): relation for pair, relation in relations.items()}
    value["pair_labels"] = [
        dict(
            left=left, right=right, relation=relations.get(frozenset((left, right)), "unspecified")
        )
        for i, left in enumerate(supported)
        for right in supported[i + 1 :]
    ]
    return value


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (("same", "different", "different"), {(0, 1), (0, 2), (1, 2)}),
        (("different", "different", "different"), {(0, 1), (0, 2), (1, 2)}),
        (("same", "unspecified", "different"), set()),
        (("same", "different", "same"), set()),
    ],
)
def test_required_pair_mask_certifies_cliques_and_singletons_or_blocks_unknown(states, expected):
    supported = ("key:a", "key:d", "key:f")
    pairs = ((supported[0], supported[1]), (supported[0], supported[2]), supported[1:])
    targets = factorized_targets([graph_record(supported, dict(zip(pairs, states, strict=True)))])
    mask = factorized_required_pair_mask(targets)
    indices = [CANDIDATES.index(candidate) for candidate in supported]
    assert mask.dtype == torch.bool and mask.shape == targets["pair"].shape
    assert not mask.tril().any()
    assert set(map(tuple, mask[0].nonzero().tolist())) == {
        tuple(sorted((indices[i], indices[j]))) for i, j in expected
    }
    assert int(mask.sum()) == len(expected)
    assert not (mask & (targets["pair"] == 2)).any()


def test_required_pair_mask_handles_isolated_singleton_and_empty_batch():
    targets = factorized_targets([graph_record(("key:a",), {}), graph_record((), {})])
    assert not factorized_required_pair_mask(targets).any()
    assert factorized_required_pair_mask(factorized_targets([])).shape == (0, 36, 36)


def test_required_edges_preserve_exact_gold_payable_candidate_and_alias_ids():
    supported = ("key:a", "key:d", "key:f", "key:q")
    relations = {
        ("key:a", "key:d"): "same",
        ("key:a", "key:f"): "different",
        ("key:a", "key:q"): "different",
        ("key:d", "key:f"): "different",
        ("key:d", "key:q"): "different",
        ("key:f", "key:q"): "unspecified",
    }
    record = graph_record(supported, relations)
    targets = factorized_targets([record])
    mask = factorized_required_pair_mask(targets)[0]
    assert int(mask.sum()) == 5
    values = logits(supported, modes={candidate: "held" for candidate in supported})
    for i, j in mask.nonzero().tolist():
        label = int(targets["pair"][0, i, j])
        values["pair"][i, j] = values["pair"][j, i] = -30
        values["pair"][i, j, label] = values["pair"][j, i, label] = 30
    predictions, table = decode(values)
    candidate_ids = {
        record["id"] + ":" + candidate
        for candidate, prediction in predictions.items()
        if prediction.status == "supported" and prediction.reward_eligible
    }
    alias_ids = {
        record["id"] + ":" + "|".join(sorted((pair.left, pair.right)))
        for pair in table.pair_decisions
        if predictions[pair.left].reward_eligible
        and predictions[pair.right].reward_eligible
        and predictions[pair.left].group == predictions[pair.right].group
    }
    assert candidate_ids == {"gold-fixture:key:a", "gold-fixture:key:d"}
    assert alias_ids == {"gold-fixture:key:a|key:d"}


@pytest.mark.parametrize("change", ["asymmetric", "missing", "inapplicable", "bad_class"])
def test_required_pair_mask_rejects_invalid_direct_targets(change):
    targets = factorized_targets([relational_row()])
    i, j = CANDIDATES.index("key:a"), CANDIDATES.index("key:d")
    if change == "asymmetric":
        targets["pair"][0, i, j] = 1
    elif change == "missing":
        targets["pair"][0, i, j] = targets["pair"][0, j, i] = -100
    elif change == "inapplicable":
        targets["applicability"][0, i] = 2
    else:
        targets["pair"][0, i, j] = targets["pair"][0, j, i] = 3
    with pytest.raises(ValueError, match="required pair"):
        factorized_required_pair_mask(targets)


@pytest.mark.parametrize("true_class", [0, 1])
def test_required_margin_p99_boundary_temperature_and_gradient(true_class):
    temperature = 2.3
    targets = factorized_targets([relational_row("same" if true_class == 0 else "different")])
    i, j = sorted((CANDIDATES.index("key:a"), CANDIDATES.index("key:d")))
    mask = torch.zeros_like(targets["pair"], dtype=torch.bool)
    mask[0, i, j] = True
    values = torch.zeros((1, 36, 36, 3), dtype=torch.float64)
    values[0, i, j, true_class] = math.log(198.0) * temperature
    assert float((values[0, i, j] / temperature).softmax(-1)[true_class]) == pytest.approx(0.99)
    assert float(
        factorized_required_margin_loss(values, targets["pair"], mask, temperature=temperature)
    ) == pytest.approx(0.0, abs=1e-12)
    values[0, i, j, true_class] -= 0.2 * temperature
    values.requires_grad_()
    loss = factorized_required_margin_loss(values, targets["pair"], mask, temperature=temperature)
    assert float(loss.detach()) == pytest.approx(0.2)
    loss.backward()
    expected = torch.full((3,), 0.5 / temperature, dtype=torch.float64)
    expected[true_class] = -1.0 / temperature
    torch.testing.assert_close(values.grad[0, i, j], expected)
    assert not values.grad[~mask].any()
    above = values.detach().clone()
    above[0, i, j, true_class] += 0.4 * temperature
    assert (
        float(
            factorized_required_margin_loss(above, targets["pair"], mask, temperature=temperature)
        )
        == 0
    )


def test_required_margin_uses_worst_edge_per_row_and_keeps_empty_rows_in_mean():
    targets = factorized_targets([relational_row(), row()])
    mask = factorized_required_pair_mask(targets)
    values = torch.zeros((2, 36, 36, 3), dtype=torch.float64)
    edges = mask[0].nonzero().tolist()
    for i, j in edges:
        values[0, i, j, targets["pair"][0, i, j]] = math.log(198.0) - 0.2
    i, j = edges[0]
    values[0, i, j, targets["pair"][0, i, j]] -= 0.4
    values.requires_grad_()
    loss = factorized_required_margin_loss(values, targets["pair"], mask)
    assert float(loss.detach()) == pytest.approx(0.3)
    loss.backward()
    nonzero_edges = set(map(tuple, values.grad.abs().sum(-1).nonzero().tolist()))
    assert nonzero_edges == {(0, i, j)}
    empty = values.detach().clone().requires_grad_()
    zero = factorized_required_margin_loss(empty, targets["pair"], torch.zeros_like(mask))
    zero.backward()
    assert float(zero.detach()) == 0 and not empty.grad.any()
    no_rows = torch.empty((0, 36, 36, 3), requires_grad=True)
    no_targets = factorized_targets([])
    zero = factorized_required_margin_loss(
        no_rows, no_targets["pair"], factorized_required_pair_mask(no_targets)
    )
    zero.backward()
    assert float(zero.detach()) == 0 and no_rows.grad.shape == no_rows.shape


def test_optional_margin_leaves_base_ce_metrics_targets_and_unary_gradients_unchanged():
    targets = factorized_targets([relational_row(), row()])
    original_targets = copy.deepcopy(targets)
    values = {kind: tensor.requires_grad_() for kind, tensor in loss_values(2).items()}
    loss, components = factorized_population_loss(values, targets)
    original_gradients = torch.autograd.grad(loss, tuple(values.values()), retain_graph=True)
    margin = factorized_required_margin_loss(
        values["pair"], targets["pair"], factorized_required_pair_mask(targets)
    )
    unchanged, unchanged_components = factorized_population_loss(values, targets)
    assert torch.equal(loss, unchanged)
    for kind in targets:
        assert torch.equal(targets[kind], original_targets[kind])
    for kind, (total, count) in components.items():
        assert torch.equal(total, unchanged_components[kind][0])
        assert count == unchanged_components[kind][1]
    combined_gradients = torch.autograd.grad(loss + margin, tuple(values.values()))
    assert all(
        torch.equal(a, b)
        for a, b in zip(original_gradients[:2], combined_gradients[:2], strict=True)
    )
    assert not torch.equal(original_gradients[2], combined_gradients[2])


@pytest.mark.parametrize("change", ["lower", "diagonal", "unspecified", "dtype", "nan"])
def test_required_margin_rejects_invalid_mask_or_logits(change):
    targets = factorized_targets([row()])
    mask = torch.zeros_like(targets["pair"], dtype=torch.bool)
    i, j = sorted((CANDIDATES.index("key:a"), CANDIDATES.index("key:d")))
    values = loss_values()["pair"]
    if change == "lower":
        mask[0, j, i] = True
    elif change == "diagonal":
        mask[0, i, i] = True
    elif change == "unspecified":
        mask[0, i, j] = True
    elif change == "dtype":
        mask = mask.float()
    else:
        values[0, i, j, 0] = torch.nan
    with pytest.raises(ValueError, match="required margin"):
        factorized_required_margin_loss(values, targets["pair"], mask)


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.inf, math.nan, True])
def test_required_margin_rejects_invalid_temperature(temperature):
    targets = factorized_targets([row()])
    with pytest.raises(ValueError, match="temperature"):
        factorized_required_margin_loss(
            loss_values()["pair"],
            targets["pair"],
            factorized_required_pair_mask(targets),
            temperature=temperature,
        )
