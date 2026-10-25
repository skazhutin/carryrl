"""Semantic recognition never certifies unresolved reward deduplication."""

import hashlib
from pathlib import Path

from carryrl.rewards.control_text import ControlPrediction, ControlPredictionTable
from carryrl.rewards.control_text_runtime import (
    runtime_control_rows,
    validate_runtime_text_controls,
)


def test_semantically_correct_unresolved_aliases_abstain_without_false_split():
    root = Path(__file__).resolve().parents[1]
    instructions = root / "configs/atomic_instructions.json"
    controls = root / "configs/atomic_controls.json"
    rows, _ = runtime_control_rows(instructions, controls)
    labels_by_text = {row["text"]: row["labels"] for row in rows}

    class UnresolvedAliasPredictor:
        def predict(self, text):
            predictions = []
            for label in labels_by_text[text]:
                status = label["status"]
                unresolved = status == "supported"
                predictions.append(ControlPrediction(
                    label["candidate"], status, label["mode"],
                    label["candidate"] if status == "unsupported" else None,
                    confidence=1.0,
                    deduplication_known=not unresolved,
                ))
            return ControlPredictionTable(
                hashlib.sha256(text.encode()).hexdigest(), "0" * 64, "1" * 64,
                tuple(predictions),
            )

    report = validate_runtime_text_controls(
        UnresolvedAliasPredictor(), instructions=instructions, controls=controls,
    )
    assert not report["passed"]
    assert all(not row["errors"] for row in report["results"])
    with_aliases = [row for row in report["results"] if row["counts"]["aliases"]["total"]]
    assert with_aliases
    for row in with_aliases:
        assert not row["passed"]
        assert row["counts"]["aliases"]["correct"] == 0
        for name in ("supported", "held", "press"):
            assert row["counts"][name]["correct"] == row["counts"][name]["total"]
