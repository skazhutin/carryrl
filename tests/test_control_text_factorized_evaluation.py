"""Independent public-spec oracle and real partial-field compatibility gates."""

import hashlib
import itertools
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from carryrl.rewards.control_text import CANDIDATES
from carryrl.rewards.control_text_factorized import (
    FactorizedControlPrediction,
    FactorizedControlPredictionTable,
    FieldDecision,
    PairDecision,
)
from carryrl.rewards.control_text_factorized_evaluation import (
    factorized_runtime_control_rows,
    validate_factorized_runtime_text_controls,
    verify_factorized_grounding,
)

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = ROOT / "configs/atomic_instructions.json"
CONTROLS = ROOT / "configs/atomic_controls.json"


def _oracle(rows):
    tables = {}
    for row in rows:
        relations = {frozenset((p["left"], p["right"])): p["relation"]
                     for p in row["pair_labels"]}
        legacy = {label["candidate"]: label for label in row["labels"]}
        predictions = []
        for label in row["factorized_labels"]:
            candidate, status, mode = label["candidate"], label["applicability"], label["mode"]
            group = legacy[candidate]["alias_group"]
            predictions.append(FactorizedControlPrediction(
                candidate, FieldDecision(status, 1., status != "unspecified",
                                          ("insufficient_information"
                                           if status == "unspecified" else None)),
                FieldDecision(mode, 1., mode != "unspecified",
                              "insufficient_information" if mode == "unspecified" else None),
                str(group) if group else candidate if status == "unsupported" else None,
                status != "unspecified"))
        pairs = []
        for a, b in itertools.combinations(CANDIDATES, 2):
            relation = relations.get(frozenset((a, b)), "unspecified")
            pairs.append(PairDecision(a, b, FieldDecision(
                relation, 1., relation != "unspecified",
                "insufficient_information" if relation == "unspecified" else None)))
        tables[row["text"]] = FactorizedControlPredictionTable(
            hashlib.sha256(row["text"].encode()).hexdigest(), "0" * 64, "1" * 64,
            tuple(predictions), tuple(pairs))
    return SimpleNamespace(predict=lambda text: tables[text]), tables


def test_new_runtime_oracle_preserves_all_global_ui_controls_and_relative_mode():
    rows, identity = factorized_runtime_control_rows(INSTRUCTIONS, CONTROLS)
    assert len(rows) == 32 and len(identity["specification_hash"]) == 64
    for row in rows:
        labels = {v["candidate"]: v for v in row["factorized_labels"]}
        if row["task"] == "ui":
            assert sum(v["applicability"] == "supported" for v in labels.values()) == 11
            assert labels["key:w"]["applicability"] == "supported"
            assert labels["key:w"]["mode"] == "held"
            assert "неизвестна" not in row["text"] and "unknown without" not in row["text"]
        if row["task"] in {"mouse", "camera", "shooting"}:
            assert labels["dx:-"]["mode"] == "relative"
        assert labels["wheel:+"]["applicability"] == "unsupported"
    oracle, _ = _oracle(rows)
    report = validate_factorized_runtime_text_controls(
        oracle, instructions=INSTRUCTIONS, controls=CONTROLS)
    assert report["passed"] and report["passed_records"] == 32
    assert report["precision_claim"] is False


def test_false_merge_false_split_and_total_abstention_fail():
    rows, _ = factorized_runtime_control_rows(INSTRUCTIONS, CONTROLS)
    for mutation in ("merge", "split", "abstain"):
        oracle, tables = _oracle(rows)
        key = rows[0]["text"]
        table = tables[key]
        if mutation == "merge":
            predictions = tuple(replace(p, group="merged") if p.status == "supported" else p
                                for p in table.predictions)
        elif mutation == "split":
            predictions = tuple(replace(p, group=p.candidate) if p.status == "supported" else p
                                for p in table.predictions)
        else:
            unknown = FieldDecision("unspecified", 0., False, "low_confidence")
            predictions = tuple(replace(p, applicability_decision=unknown, mode_decision=unknown,
                                        group=None, deduplication_known=False)
                                for p in table.predictions)
        tables[key] = replace(table, predictions=predictions)
        report = validate_factorized_runtime_text_controls(
            oracle, instructions=INSTRUCTIONS, controls=CONTROLS)
        assert not report["passed"]
        if mutation != "abstain":
            assert report["results"][0]["errors"]


def test_known_applicability_remains_correct_when_mode_abstains():
    rows, _ = factorized_runtime_control_rows(INSTRUCTIONS, CONTROLS)
    oracle, tables = _oracle(rows)
    key = rows[0]["text"]
    unknown = FieldDecision("unspecified", .99, False, "insufficient_information")
    tables[key] = replace(tables[key], predictions=tuple(
        replace(p, mode_decision=unknown) for p in tables[key].predictions))
    report = validate_factorized_runtime_text_controls(
        oracle, instructions=INSTRUCTIONS, controls=CONTROLS)
    first = report["results"][0]
    assert not first["passed"] and not first["errors"]
    assert first["counts"]["supported"]["correct"] == first["counts"]["supported"]["total"]
    assert first["counts"]["held"]["correct"] == 0


def test_actual_timing_first_inflected_space_mention_is_transmitted():
    from transformers import AutoTokenizer

    from carryrl.agent.language import TEXT_ID, TEXT_REVISION

    tokenizer = AutoTokenizer.from_pretrained(TEXT_ID, revision=TEXT_REVISION,
                                              local_files_only=True)
    rows, _ = factorized_runtime_control_rows(INSTRUCTIONS, CONTROLS)
    row = next(r for r in rows if r["task"] == "timing" and r["language"] == "ru")
    text = row["text"]
    start = text.index("пробела")
    # Independent expectation uses the actual public spelling and exact offset.
    witness = {"id": "actual-timing-ru", "split": "dev", "text": text,
               "expected_mentions": [{"candidate": "key:space", "start": start,
                                      "end": start + len("пробела")}]}
    report = verify_factorized_grounding(tokenizer, [witness])
    assert report["expected_control_mentions"] == 1 and report["missing_mentions"] == 0


def test_independent_witness_detects_loss_even_with_later_correct_mention(monkeypatch):
    from transformers import AutoTokenizer

    import carryrl.rewards.control_text_grounding as grounding
    from carryrl.agent.language import TEXT_ID, TEXT_REVISION

    tokenizer = AutoTokenizer.from_pretrained(TEXT_ID, revision=TEXT_REVISION,
                                              local_files_only=True)
    original = grounding._mention_spans
    text = "Короткое нажатие пробела выполняет взмах. Отпустите пробел."
    start = text.index("пробела")

    def broken(value):
        spans = original(value)
        spans["key:space"] = [(a, b) for a, b in spans["key:space"] if a != start]
        return spans

    monkeypatch.setattr(grounding, "_mention_spans", broken)
    row = {"id": "first-space-lost", "split": "train", "text": text,
           "expected_mentions": [{"candidate": "key:space", "start": start, "end": start + 7}]}
    with pytest.raises(ValueError, match="independent control mentions"):
        verify_factorized_grounding(tokenizer, [row])
