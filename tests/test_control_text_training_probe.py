"""Development probes keep held-out inputs untouched and cannot certify readiness."""

import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from carryrl.rewards import control_text_training as training
from carryrl.rewards.control_text import CANDIDATES, ConditionalTextControlHead


def row(split, index=0, **extra):
    labels = [dict(candidate=c, status="unsupported", mode="unknown", alias_group=0)
              for c in CANDIDATES]
    for position, mode, group in ((0, "held", 1), (1, "held", 1), (2, "press", 2)):
        labels[position].update(status="supported", mode=mode, alias_group=group)
    labels[3]["status"] = "unknown"
    return dict(text=f"{split}:{index}", split=split, labels=labels,
                group_id=f"{split}:{index}", semantic_mapping_id=f"{split}:{index}", **extra)


def report(rows, *, alias_correct=True):
    n = len(rows)
    return {
        "independent_semantic_clusters": len({r["semantic_mapping_id"] for r in rows}),
        "critical_errors": 0 if alias_correct else 1,
        "signs": {name: dict(clear_count=n, clear_accepted=n)
                  for name in ("supported", "unsupported")},
        "modes": {name: dict(clear_count=2*n, correct_count=2*n)
                  for name in ("held", "press")},
        "aliases": dict(clear_count=n, correct_count=n if alias_correct else 0),
    }


def test_development_groups_keep_paraphrases_together_and_exclude_calibration():
    rows = [row(split, i) for split in ("train", "dev", "calibration", "sealed_test")
            for i in range(8)]
    rows += [{**r, "text": r["text"] + " paraphrase"} for r in rows if r["split"] == "dev"]
    temp, score = training.split_development_groups(rows, seed=7)
    identity = training.development_split_identity(temp, score)
    assert len(temp) == len(score) == 8
    assert not set(identity["temperature_groups"]) & set(identity["score_groups"])
    assert all(r["split"] == "dev" for r in temp + score)
    assert (temp, score) == training.split_development_groups(rows, seed=7)


def test_feasibility_requires_independent_mappings_decisions_coverage_and_zero_errors():
    rows = [row("dev", i) for i in range(100)]
    assert training.developmental_feasibility_gate(report(rows), rows)["passed"]
    assert not training.developmental_feasibility_gate(report(rows, alias_correct=False), rows)[
        "passed"]
    # Repeating paraphrases cannot manufacture independent evidence.
    repeated = [row("dev")] * 100
    assert not training.developmental_feasibility_gate(report(repeated), repeated)["passed"]
    low_coverage = report(rows)
    low_coverage["modes"]["press"]["correct_count"] = 159
    assert not training.developmental_feasibility_gate(low_coverage, rows)["passed"]
    assert training.developmental_feasibility_gate(report(rows), rows)["ready"] is False


def test_probe_subset_keeps_elementary_groups_whole_and_rejects_too_small_budget():
    rows = [row("train", i, elementary=i < 2, scenario_kind=str(i % 2)) for i in range(8)]
    rows += [{**r, "text": r["text"] + " alternative"} for r in rows]
    rows += [row(split) for split in ("dev", "calibration", "sealed_test")]
    subset, config = training.select_probe_records(rows, seed=7, max_train_groups=4,
                                                   max_train_texts=8)
    assert config["train_groups"] == 4 and config["train_texts"] == 8
    assert set(config["mandatory_elementary_group_ids"]) <= set(config["selected_train_group_ids"])
    assert {r["split"] for r in subset} == {"train", "dev"}
    assert (subset, config) == training.select_probe_records(rows, seed=7, max_train_groups=4,
                                                            max_train_texts=8)
    with pytest.raises(ValueError, match="mandatory"):
        training.select_probe_records(rows, seed=7, max_train_groups=1)


def test_conditional_dropout_preserves_weights_and_is_identical_during_evaluation():
    torch.manual_seed(9)
    original = ConditionalTextControlHead()
    changed = ConditionalTextControlHead(readout_dropout=0.7)
    changed.load_state_dict(original.state_dict(), strict=True)
    assert original.state_dict().keys() == changed.state_dict().keys()
    features = torch.randn(1, 36, 128)
    grounding = torch.zeros(1, 36, 5, dtype=torch.bool)
    original.eval()
    changed.eval()
    with torch.no_grad():
        expected = original._grounded_readout(features, grounding)
        actual = changed._grounded_readout(features, grounding)
    assert all(torch.equal(expected[kind], actual[kind]) for kind in expected)
    changed.train()
    changed_status = changed._grounded_readout(features, grounding)["status"]
    assert not torch.equal(changed_status, actual["status"])


class TrainHead(nn.Module):
    architecture = training.CONDITIONAL_HEAD_ARCHITECTURE

    def __init__(self, **kwargs):
        super().__init__()
        self.status = nn.Linear(1, 3)
        self.mode = nn.Linear(1, 2)
        self.pair = nn.Linear(1, 2)

    def forward(self, n):
        return {"status": self.status(torch.ones(n, 36, 1)),
                "mode": self.mode(torch.ones(n, 36, 1)),
                "pair": self.pair(torch.ones(n, 36, 36, 1))}


def test_feasible_epoch_beats_lower_nll_with_alias_failure_and_pair_lr_is_isolated(monkeypatch):
    rows = [row("train")] + [row("dev", i) for i in range(200)]
    monkeypatch.setattr(training, "ConditionalTextControlHead", TrainHead)
    monkeypatch.setattr(training, "_batch", lambda *args: (None, None))
    monkeypatch.setattr(training, "_head_forward", lambda e, h, r, *args: h(len(r)))
    values = iter([0.1, 3, 2, 3, 0.01, 3])  # dev, TRAIN diagnostic per epoch
    monkeypatch.setattr(training, "evaluate_head_metrics", lambda *a, **kw:
                        {"selection_nll": next(values)})
    temperature_inputs = []

    def temperatures(e, h, records, **kwargs):
        temperature_inputs.append(records)
        return {"marker": 1}

    monkeypatch.setattr(training, "fit_dev_temperatures", temperatures)
    def frozen(e, head, **kwargs):
        head.requires_grad_(False)  # Must never freeze the head still being fitted.
        return None
    monkeypatch.setattr(training, "FrozenTextControlReward", frozen)
    calls = iter([False, True, False])
    monkeypatch.setattr(training, "validate_text_control_predictions",
                        lambda p, records: report(records, alias_correct=next(calls)))
    monkeypatch.setattr(training, "calibrated_development_nll", lambda *args: 1.0)
    original = torch.optim.AdamW
    optimizers = []
    def optimizer(*args, **kwargs):
        result = original(*args, **kwargs)
        optimizers.append(result)
        return result
    monkeypatch.setattr(training.torch.optim, "AdamW", optimizer)
    encoder = nn.Linear(1, 1)
    _, metadata = training.fit_text_control_head(
        encoder, rows, epochs=3, head_variant="supported_conditional_v6",
        selection_protocol="development_feasibility_v1", learning_rate=1e-4,
        pair_learning_rate=3e-4,
    )
    assert metadata["selected_epoch"] == 2
    assert metadata["selected_development_feasible"]
    assert metadata["optimizer_started_empty"]
    groups = optimizers[0].param_groups
    assert [g["lr"] for g in groups] == [1e-4, 3e-4]
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == 6
    assert len(groups[1]["params"]) == 2
    temp, score = training.split_development_groups(rows, seed=20261020)
    assert all(records == [rows[0]] + temp for records in temperature_inputs)
    assert not {r["text"] for r in score} & {r["text"] for r in temperature_inputs[0]}


@pytest.mark.parametrize("runtime_fails", [False, True])
@pytest.mark.parametrize("cross_modal_train_repeats", [1, 4])
def test_probe_does_not_encode_or_predict_heldout_and_has_no_ready_artifact(
    tmp_path, monkeypatch, runtime_fails, cross_modal_train_repeats,
):
    rows = [row(split, i) for split in ("train", "dev", "calibration", "sealed_test")
            for i in range(200)]
    rows[0].update(scenario_kind="cross_modal_alias", elementary=True)
    manifest = {"records_sha256": "d" * 64}
    monkeypatch.setattr(training, "read_control_dataset", lambda _: (manifest, rows))
    monkeypatch.setattr(training, "verify_grounded_text_inputs", lambda *a, **kw: {})
    encoded, predicted = [], []
    features = SimpleNamespace()
    def prepare(e, records, *args, **kwargs):
        encoded.extend(records)
        return features
    monkeypatch.setattr(training, "prepare_disk_token_features", prepare)
    def fit(e, records, **kwargs):
        assert kwargs["epochs"] == 8
        assert kwargs["cross_modal_train_repeats"] == cross_modal_train_repeats
        assert len(records) == len({r["text"] for r in records})
        assert {r["split"] for r in records} == {"train", "dev"}
        return TrainHead(), {"trained": True, "selected_development_calibration": {"fixed": .99}}
    monkeypatch.setattr(training, "fit_text_control_head", fit)
    monkeypatch.setattr(training, "FrozenTextControlReward", lambda *a, **kw: None)
    def validate(p, records):
        predicted.extend(records)
        return report(records)
    monkeypatch.setattr(training, "validate_text_control_predictions", validate)
    monkeypatch.setattr(training, "calibrate_text_control_head", lambda *a, **kw:
                        pytest.fail("probe accessed full calibration"))
    monkeypatch.setattr(training, "save_text_control_artifact", lambda *a, **kw:
                        pytest.fail("probe attempted a runtime artifact"))
    from carryrl.rewards import control_text_runtime
    def runtime(*args, **kwargs):
        if runtime_fails:
            raise TimeoutError("runtime failed")
        return {"passed": True}
    monkeypatch.setattr(control_text_runtime, "validate_runtime_text_controls", runtime)
    output = tmp_path / "probe"
    def launch():
        return training.train_text_control_artifact(
            "unused", output, encoder=SimpleNamespace(tokenizer=None), probe_only=True, epochs=8,
            selection_protocol="development_feasibility_v1",
            head_variant="supported_conditional_v6", max_train_groups=10,
            cross_modal_train_repeats=cross_modal_train_repeats,
        )
    if runtime_fails:
        with pytest.raises(TimeoutError, match="runtime failed"):
            launch()
        workspace = tmp_path / "probe.fit"
        assert (workspace / "selected_head.pt").is_file()
        assert (workspace / "selected_training.json").is_file()
        assert (workspace / "selected_calibration.json").is_file()
        assert (workspace / "selected_validation.json").is_file()
        assert (workspace / "stage_timings.json").is_file()
        assert json.loads((tmp_path / "probe.failure.json").read_text())["ready"] is False
        return
    result = launch()
    assert result["ready"] is False
    assert not (output / "manifest.json").exists()
    assert {r["split"] for r in encoded} == {"train", "dev"}
    assert {r["split"] for r in predicted} == {"dev"}
    assert json.loads((output / "probe.json").read_text())["ready"] is False
    assert (tmp_path / "probe.fit" / "stage_timings.json").is_file()


@pytest.mark.parametrize("option,value", [("learning_rate", float("nan")),
    ("pair_learning_rate", 0), ("readout_dropout", 1), ("early_stopping_patience", 0)])
def test_invalid_recipe_parameters_fail_before_fitting(option, value):
    with pytest.raises(ValueError):
        training.fit_text_control_head(None, [], **{option: value})


@pytest.mark.parametrize("feasible", [False, True])
def test_full_admission_reuses_selected_temperatures_and_cannot_bypass_feasibility(
    tmp_path, monkeypatch, feasible,
):
    rows = [row(split, i) for split in ("train", "dev", "calibration", "sealed_test")
            for i in range(4)]
    monkeypatch.setattr(training, "read_control_dataset", lambda _: ({"records_sha256": "d"*64},
                                                                     rows))
    monkeypatch.setattr(training, "verify_grounded_text_inputs", lambda *a, **kw: {})
    features = SimpleNamespace(metadata={"actual_bytes": 1, "throughput_probe": {}})
    monkeypatch.setattr(training, "prepare_disk_token_features", lambda *a, **kw: features)
    temp, score = training.split_development_groups(rows, seed=20261020)
    selected_calibration = {"selected_temperature": 0.5}
    metadata = {"trained": True, "selected_development_calibration": selected_calibration,
                "selected_development_feasible": feasible,
                "development_group_split": training.development_split_identity(temp, score)}
    monkeypatch.setattr(training, "fit_text_control_head", lambda *a, **kw: (TrainHead(), metadata))
    monkeypatch.setattr(training, "calibrate_text_control_head", lambda *a, **kw:
                        pytest.fail("selected temperatures must not be refitted on score rows"))
    monkeypatch.setattr(training, "FrozenTextControlReward", lambda *a, **kw: None)
    predicted = []
    def validate(p, records):
        predicted.extend(records)
        return {}
    monkeypatch.setattr(training, "validate_text_control_predictions", validate)
    def save(output, **kwargs):
        assert kwargs["calibration"]["selected_temperature"] == 0.5
        assert kwargs["training"]["trained"] is False
        output.mkdir()
        return {"ready": False, "gates": {"passed": True},
                "files": {"head.pt": "a"*64, "calibration.json": "b"*64}}
    monkeypatch.setattr(training, "save_text_control_artifact", save)
    from carryrl.rewards import control_text_runtime
    monkeypatch.setattr(control_text_runtime, "validate_runtime_text_controls",
                        lambda *a, **kw: {"passed": True, "specification_hash": "c"*64})
    result = training.train_text_control_artifact(
        "unused", tmp_path / "full", encoder=SimpleNamespace(tokenizer=None),
        selection_protocol="development_feasibility_v1", head_variant="supported_conditional_v6",
    )
    assert result["ready"] is feasible
    assert {r["split"] for r in predicted} == {"calibration"}


def contrast_pair(pair_id="pair-0", *, first=10, variants=2):
    return [row("train", index, contrastive=True, contrastive_pair_id=pair_id,
                elementary=False, variant_index=variant, group_variants=variants)
            | {"text": f"train:{index}:{variant}"}
            for index in (first, first + 1) for variant in range(variants)]


def test_probe_makes_both_contrastive_members_mandatory_without_relabelling_elementary():
    rows = [row("train", 0, elementary=True)] + contrast_pair() + [row("train", i)
                                                                  for i in range(20, 24)]
    rows += [row("dev")]
    original = json.dumps(rows, sort_keys=True)
    selected, config = training.select_probe_records(rows, seed=7, max_train_groups=3,
                                                    max_train_texts=5)
    assert config["selected_train_group_ids"] == ["train:0", "train:10", "train:11"]
    assert config["mandatory_elementary_group_ids"] == ["train:0"]
    assert config["mandatory_contrastive_group_ids"] == ["train:10", "train:11"]
    assert config["mandatory_contrastive_pair_ids"] == ["pair-0"]
    assert config["mandatory_group_reasons"] == {
        "train:0": ["elementary"], "train:10": ["contrastive_pair"],
        "train:11": ["contrastive_pair"],
    }
    assert len([r for r in selected if r.get("contrastive")]) == 4
    assert all(r["elementary"] is False for r in selected if r.get("contrastive"))
    assert json.dumps(rows, sort_keys=True) == original
    assert (selected, config) == training.select_probe_records(
        rows, seed=7, max_train_groups=3, max_train_texts=5,
    )
    with pytest.raises(ValueError, match="3 total groups / 5 texts"):
        training.select_probe_records(rows, seed=7, max_train_groups=2)
    with pytest.raises(ValueError, match="3 total groups / 5 texts"):
        training.select_probe_records(rows, seed=7, max_train_texts=4)


@pytest.mark.parametrize("defect", ["missing_member", "three_members", "partial_variants",
                                   "inconsistent_id", "missing_id"])
def test_probe_rejects_incomplete_oversized_or_inconsistent_contrastive_pairs(defect):
    rows = contrast_pair()
    if defect == "missing_member":
        rows = rows[:2]
    elif defect == "three_members":
        rows += [row("train", 12, contrastive_pair_id="pair-0")]
    elif defect == "partial_variants":
        rows.pop()
    elif defect == "inconsistent_id":
        rows[0]["contrastive_pair_id"] = "different"
    else:
        for r in rows:
            r.pop("contrastive_pair_id")
    with pytest.raises(ValueError, match="contrastive|contrast"):
        training.select_probe_records(rows, seed=7)


def test_probe_requires_all_84_elementary_and_144_contrastive_groups():
    rows = [row("train", i, elementary=True) for i in range(84)]
    for pair in range(72):
        rows += contrast_pair(str(pair), first=1000 + 2 * pair)
    with pytest.raises(ValueError, match="84 elementary, 144 contrastive, 228 total groups"):
        training.select_probe_records(rows, seed=7, max_train_groups=227)
    _, config = training.select_probe_records(rows, seed=7, max_train_groups=228)
    assert len(config["mandatory_elementary_group_ids"]) == 84
    assert len(config["mandatory_contrastive_group_ids"]) == 144
    assert len(config["mandatory_contrastive_pair_ids"]) == 72
    assert config["train_groups"] == 228


def _v9_rows():
    from carryrl.rewards.control_text_data import semantic_mapping_identity

    rows = []
    for index, (split, family) in enumerate((("train", 0), ("dev", 12),
                                            ("calibration", 16), ("sealed_test", 20))):
        record = row(split)
        record["labels"][4 + index].update(status="supported", mode="held", alias_group=3)
        record["declared_candidates"] = list(CANDIDATES)
        identity = semantic_mapping_identity(record["labels"], record["declared_candidates"])
        record.update(mapping_id=identity, semantic_mapping_id=identity,
                      template_family=f"v8-binding-composition-{family:02}",
                      grammar_families=[f"v8-binding-composition-{family:02}",
                                        f"v9-relation-witness-{family:02}"])
        rows.append(record)
    return rows


def _write_v9_dataset(directory, rows, **overrides):
    import hashlib

    payload = "".join(json.dumps(r) + "\n" for r in rows).encode()
    manifest = {"version": "carryrl.control_text_data.v9", "records": len(rows),
                "records_sha256": hashlib.sha256(payload).hexdigest(),
                "template_families_disjoint": True, "internal_grammar_families_disjoint": True,
                **overrides}
    (directory / "records.jsonl").write_bytes(payload)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize("version", ["carryrl.control_text_data.v9",
                                     "carryrl.control_text_data.v10"])
def test_reader_authenticates_records_and_retains_semantic_grammar_guards(tmp_path, version):
    rows = _v9_rows()
    manifest = _write_v9_dataset(tmp_path, rows, version=version)
    actual_manifest, actual_rows = training.read_control_dataset(tmp_path)
    assert actual_manifest == manifest and actual_rows == rows
    with (tmp_path / "records.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        training.read_control_dataset(tmp_path)
    _write_v9_dataset(tmp_path, rows, version=version, template_families_disjoint=False)
    with pytest.raises(ValueError, match="split-owned"):
        training.read_control_dataset(tmp_path)
    _write_v9_dataset(tmp_path, rows, version=version, internal_grammar_families_disjoint=False)
    with pytest.raises(ValueError, match="split-owned"):
        training.read_control_dataset(tmp_path)
    rows[1]["grammar_families"].append(rows[0]["grammar_families"][1])
    _write_v9_dataset(tmp_path, rows, version=version)
    with pytest.raises(ValueError, match="split leakage"):
        training.read_control_dataset(tmp_path)
    rows = _v9_rows()
    rows[1]["semantic_mapping_id"] = rows[0]["semantic_mapping_id"]
    _write_v9_dataset(tmp_path, rows, version=version)
    with pytest.raises(ValueError, match="split leakage|identity mismatch"):
        training.read_control_dataset(tmp_path)


def test_full_gate_still_requires_299_independent_clusters_and_zero_critical_errors():
    from carryrl.rewards.control_text import _count_gate, calibration_gates

    def count(n):
        return dict(clear_count=1000, clear_accepted=1000, accepted_count=1000,
                    accepted_correct=1000, accepted_scenarios=n, error_scenarios=0)
    assert not _count_gate(count(298))["passed"]
    assert _count_gate(count(299))["passed"]
    mode_counts = dict(clear_count=1000, accepted_count=800, correct_count=800)
    validation = {**count(299), "critical_errors": 0,
                  "signs": {name: count(299) for name in ("supported", "unsupported")},
                  "modes": {name: dict(mode_counts) for name in ("held", "press")},
                  "aliases": dict(mode_counts)}
    assert calibration_gates(validation)["passed"]
    validation["critical_errors"] = 1
    assert not calibration_gates(validation)["passed"]
    validation["critical_errors"] = 0
    validation["aliases"].update(accepted_count=799, correct_count=799)
    assert not calibration_gates(validation)["passed"]


def test_cross_modal_exposure_repeats_complete_pair_views_and_preserves_other_populations():
    from collections import Counter

    paired = contrast_pair()
    for record in paired:
        record["scenario_kind"] = "contrastive_cross_modal_alias"
    normal = [row("train", 20, scenario_kind="cross_modal_alias") |
              {"text": f"normal-view-{index}"} for index in range(2)]
    other = [row("train", 0, scenario_kind="navigation")]
    heldout = [row(split, scenario_kind="cross_modal_alias")
               for split in ("dev", "calibration", "sealed_test")]
    records = paired + normal + other + heldout
    original = json.dumps(records, sort_keys=True)
    effective, exposure = training.effective_train_presentations(
        records, cross_modal_train_repeats=4,
    )
    counts = Counter(record["text"] for record in effective)
    assert all(counts[record["text"]] == 4 for record in paired + normal)
    assert counts[other[0]["text"]] == 1
    assert all(record["text"] not in counts for record in heldout)
    assert exposure["unique_train_groups"] == 4
    assert exposure["unique_train_records"] == exposure["unique_train_texts"] == 7
    assert exposure["effective_train_presentations_per_epoch"] == 25
    assert exposure["train_group_repeat_factors"] == {
        "train:0": 1, "train:10": 4, "train:11": 4, "train:20": 4,
    }
    assert json.dumps(records, sort_keys=True) == original
    unweighted, default = training.effective_train_presentations(records)
    assert unweighted == paired + normal + other
    assert default["effective_train_presentations_per_epoch"] == 7
    paired[2]["scenario_kind"] = paired[3]["scenario_kind"] = "navigation"
    with pytest.raises(ValueError, match="both complete members equally"):
        training.effective_train_presentations(records, cross_modal_train_repeats=4)


def test_fit_exposure_uses_effective_balance_prior_and_temperatures_with_same_init(monkeypatch):
    paired = contrast_pair()
    for record in paired:
        record["scenario_kind"] = "contrastive_cross_modal_alias"
        if record["group_id"] == "train:11":
            record["labels"][1]["alias_group"] = 3
    unique_train = paired + [row("train", 0, scenario_kind="navigation")]
    records = unique_train + [row("dev", i) for i in range(200)]
    monkeypatch.setattr(training, "ConditionalTextControlHead", TrainHead)
    monkeypatch.setattr(training, "_batch", lambda *args: (None, None))
    monkeypatch.setattr(training, "_head_forward", lambda e, h, r, *args: h(len(r)))
    diagnostic_train_populations = []
    def metrics(e, h, rows, **kwargs):
        if rows[0]["split"] == "train":
            diagnostic_train_populations.append(rows)
        return {"selection_nll": 1.0}
    monkeypatch.setattr(training, "evaluate_head_metrics", metrics)
    temperature_populations = []
    def temperatures(e, h, rows, **kwargs):
        temperature_populations.append(rows)
        return training._prior_correction(h, [r for r in rows if r["split"] == "train"])
    monkeypatch.setattr(training, "fit_dev_temperatures", temperatures)
    monkeypatch.setattr(training, "FrozenTextControlReward", lambda *a, **kw: None)
    monkeypatch.setattr(training, "validate_text_control_predictions", lambda p, r: report(r))
    monkeypatch.setattr(training, "calibrated_development_nll", lambda *a: 1.0)
    metadata = {}
    for repeat in (1, 4):
        _, metadata[repeat] = training.fit_text_control_head(
            nn.Linear(1, 1), records, epochs=8, head_variant="supported_conditional_v6",
            selection_protocol="development_feasibility_v1", cross_modal_train_repeats=repeat,
        )
    weighted = metadata[4]
    assert weighted["initial_trainable_sha256"] == metadata[1]["initial_trainable_sha256"]
    assert weighted["completed_epochs"] == metadata[1]["completed_epochs"] == 8
    assert weighted["class_balance"]["counts"] == {
        "status": [51, 544, 17], "mode": [34, 17], "pair": [9, 42],
    }
    expected_prior = {f"{kind}_log_class_weights": [math.log(w) for w in values]
                      for kind, values in weighted["class_balance"]["weights"].items()}
    assert weighted["deployment_logit_prior_correction"] == expected_prior
    assert all(weighted["selected_development_calibration"][k] == v
               for k, v in expected_prior.items())
    assert all(len([r for r in pop if r["split"] == "train"]) == 17
               for pop in temperature_populations[8:])
    assert all(len({r["text"] for r in pop}) == len(pop) == 5
               for pop in diagnostic_train_populations)
    assert all(entry["unique_train_texts"] == 5 and entry["effective_train_presentations"] == 17
               for entry in weighted["history"])
    assert weighted["train_probe_population"] == "unique_train_records"


@pytest.mark.parametrize("value", [0, 9, 1.5, True, None, float("nan")])
def test_invalid_cross_modal_repeats_rejected_before_fitting(value):
    with pytest.raises(ValueError, match="cross_modal_train_repeats"):
        training.fit_text_control_head(None, [], cross_modal_train_repeats=value)


@pytest.mark.parametrize("head,protocol", [("legacy_v1", "development_feasibility_v1"),
                                           ("supported_conditional_v6", "dev_nll")])
def test_cross_modal_repeats_require_conditional_feasibility_recipe(head, protocol):
    with pytest.raises(ValueError, match="conditional|feasibility"):
        training.fit_text_control_head(None, [], cross_modal_train_repeats=4,
                                       head_variant=head, selection_protocol=protocol)



def test_default_exposure_preserves_legacy_rows_without_new_group_constraints():
    records = [row("train", 0, scenario_kind="first", contrastive_pair_id="one"),
               row("train", 0, scenario_kind="second", contrastive_pair_id="two") |
               {"text": "another view"}]
    effective, metadata = training.effective_train_presentations(records)
    assert effective == records
    assert metadata["cross_modal_train_repeats"] == 1
    assert metadata["effective_train_presentations_per_epoch"] == 2


def test_full_fallback_calibration_receives_effective_train_prior_without_reencoding_duplicates(
    tmp_path, monkeypatch,
):
    paired = contrast_pair()
    for record in paired:
        record["scenario_kind"] = "contrastive_cross_modal_alias"
        if record["group_id"] == "train:11":
            record["labels"][1]["alias_group"] = 3
    records = paired + [row("train", 0)] + [row(split, i)
        for split in ("dev", "calibration", "sealed_test") for i in range(2)]
    monkeypatch.setattr(training, "read_control_dataset", lambda _: ({"records_sha256": "d"*64},
                                                                     records))
    monkeypatch.setattr(training, "verify_grounded_text_inputs", lambda *a, **kw: {})
    encoded = []
    features = SimpleNamespace(metadata={"actual_bytes": 1, "throughput_probe": {}})
    def prepare(e, rows, *args, **kwargs):
        encoded.extend(rows)
        return features
    monkeypatch.setattr(training, "prepare_disk_token_features", prepare)
    monkeypatch.setattr(training, "fit_text_control_head", lambda *a, **kw:
                        (TrainHead(), {"trained": True, "selected_development_feasible": False}))
    temperatures = []
    def fit_temperatures(e, h, rows, **kwargs):
        temperatures.extend(rows)
        return training._prior_correction(h, [r for r in rows if r["split"] == "train"])
    monkeypatch.setattr(training, "fit_dev_temperatures", fit_temperatures)
    monkeypatch.setattr(training, "FrozenTextControlReward", lambda *a, **kw: None)
    predicted = []
    def validate(p, rows):
        predicted.extend(rows)
        return {}
    monkeypatch.setattr(training, "validate_text_control_predictions", validate)
    def save(output, **kwargs):
        assert kwargs["calibration"]["pair_log_class_weights"] == pytest.approx(
            [math.log(51/(2*9)), math.log(51/(2*42))]
        )
        output.mkdir()
        return {"ready": False, "gates": {"passed": True},
                "files": {"head.pt": "a"*64, "calibration.json": "b"*64}}
    monkeypatch.setattr(training, "save_text_control_artifact", save)
    from carryrl.rewards import control_text_runtime
    monkeypatch.setattr(control_text_runtime, "validate_runtime_text_controls",
                        lambda *a, **kw: {"passed": True, "specification_hash": "c"*64})
    result = training.train_text_control_artifact(
        "unused", tmp_path / "full", encoder=SimpleNamespace(tokenizer=None),
        selection_protocol="development_feasibility_v1", head_variant="supported_conditional_v6",
        cross_modal_train_repeats=4,
    )
    assert result["ready"] is False
    assert encoded == records
    assert len([r for r in temperatures if r["split"] == "train"]) == 17
    assert len([r for r in temperatures if r["split"] != "train"]) == 6
    assert {r["split"] for r in predicted} == {"calibration"}
    config = json.loads((tmp_path / "full.fit" / "run_config.json").read_text())
    assert config["train_exposure"]["unique_train_texts"] == 5
    assert config["train_exposure"]["effective_train_presentations_per_epoch"] == 17


def test_training_cli_passes_cross_modal_repeats_and_rejects_invalid_recipe_before_encoder(
    monkeypatch,
):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "text_reward_cli_repeats", Path(__file__).resolve().parents[1] /
        "scripts/train_text_control_reward.py",
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    with pytest.raises(SystemExit):
        script.main(["--dataset", "unused", "--output", "unused",
                     "--cross-modal-train-repeats", "9"])
    monkeypatch.setattr(script, "MultilingualInstructionEncoder", lambda **kwargs:
                        pytest.fail("invalid recipe loaded an encoder"))
    with pytest.raises(SystemExit):
        script.main(["--dataset", "unused", "--output", "unused",
                     "--cross-modal-train-repeats", "4"])
    monkeypatch.setattr(script, "MultilingualInstructionEncoder", lambda **kwargs:
                        SimpleNamespace(to=lambda device: None))
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: None)
    observed = {}
    def fit(*args, **kwargs):
        observed.update(kwargs)
        return {"ready": False}
    monkeypatch.setattr(script, "train_text_control_artifact", fit)
    assert script.main(["--dataset", "unused", "--output", "unused", "--probe-only",
                        "--head-variant", "supported_conditional_v6",
                        "--cross-modal-train-repeats", "4"]) == 0
    assert observed["cross_modal_train_repeats"] == 4
    assert observed["selection_protocol"] == "development_feasibility_v1"
    assert observed["epochs"] == 8
