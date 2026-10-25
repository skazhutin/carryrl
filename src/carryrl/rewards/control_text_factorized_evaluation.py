"""Independent public-spec compatibility for the new partial-field text contract.

Offline only: this source and its condition metadata are never runtime inputs.
The 32 exact runtime strings are compatibility checks, not synthetic TRAIN data
and not 32 independent statistical examples for precision calibration.
"""

import hashlib
import json
import time
from pathlib import Path

from carryrl.rewards.control_text import CANDIDATES, content_hash
from carryrl.rewards.control_text_data_v14 import (
    PROTOCOL_VERSION,
    FactorizedSpec,
    Operation,
    annotate_factorized_spec,
    factorized_mapping_identity,
)

RUNTIME_PROTOCOL_VERSION = "carryrl.runtime_text_controls.factorized.v2"


def factorized_runtime_control_rows(instructions, controls):
    """Global support is known even if its currently executable condition is not."""
    instructions, controls = Path(instructions), Path(controls)
    texts = json.loads(instructions.read_text())
    public = json.loads(controls.read_text())
    if public.get("version") != "carryrl.atomic_public_controls.v2":
        raise ValueError("factorized runtime gate requires V2 global applicability")
    games = public["games"]
    if set(texts) != set(games) or len(games) != 16:
        raise ValueError("runtime control gate requires the same sixteen games")
    rows = []
    for task, variants in texts.items():
        if len(variants) != 2 or not all(
                isinstance(text, str) and text.strip() for text in variants):
            raise ValueError("runtime gate requires nonempty RU and EN text")
        source = games[task]
        operations = []
        for group in source["groups"]:
            if group["mode"] not in {"held", "press", "relative", "unspecified"}:
                raise ValueError("invalid public control mode")
            candidates = tuple(group["candidates"])
            relative = all(c.startswith(("dx:", "dy:", "wheel:")) for c in candidates)
            mode = "relative" if relative and group["mode"] == "held" else group["mode"]
            operations.append(Operation(candidates, mode,
                                        json.dumps(group["when"], sort_keys=True)
                                        if group.get("when") else None))
        spec = FactorizedSpec(tuple(operations), complete=source.get("complete") is True,
                              kind="runtime_public_spec")
        labels, pairs, legacy = annotate_factorized_spec(spec)
        for language, text in zip(("ru", "en"), variants, strict=True):
            rows.append({"group_id": task, "independent_group_id": task,
                         "task": task, "language": language, "text": text,
                         "factorized_labels": labels, "pair_labels": pairs, "labels": legacy,
                         "protocol_version": PROTOCOL_VERSION})
    identity = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in (("instructions_sha256", instructions),
                                   ("controls_sha256", controls))}
    identity["specification_hash"] = content_hash(
        {**identity, "protocol": RUNTIME_PROTOCOL_VERSION})
    return rows, identity


def public_runtime_mapping_exclusions(instructions, controls):
    """Reserve only exact complete public-spec mappings, not individual controls."""
    rows, identity = factorized_runtime_control_rows(instructions, controls)
    mapping_ids = sorted({factorized_mapping_identity(row["factorized_labels"], row["pair_labels"])
                          for row in rows})
    return mapping_ids, {"version": "carryrl.factorized_runtime_mapping_exclusions.v1",
                         **identity, "mapping_count": len(mapping_ids),
                         "mapping_ids": mapping_ids,
                         "policy": "exclude exact full public-spec semantics from every split",
                         "runtime_text_copied": False}


def validate_factorized_runtime_text_controls(predictor, *, instructions, controls, deadline=None):
    """All 32 texts: zero wrong assertions, >=80% of every useful field slice.

    Partial fields stay useful: known applicability is scored independently of
    unspecified timing. Reward eligibility additionally requires resolved alias
    deduplication; confident wrong group splits/merges are critical errors.
    """
    rows, identity = factorized_runtime_control_rows(instructions, controls)
    results = []
    for row in rows:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("factorized runtime compatibility budget exhausted")
        table = predictor.predict(row["text"])
        predictions = {value.candidate: value for value in table.predictions}
        if set(predictions) != set(CANDIDATES) or len(predictions) != len(table.predictions):
            raise ValueError("runtime prediction candidates differ from specification")
        counts = {name: {"total": 0, "correct": 0} for name in (
            "supported", "unsupported", "unknown", "held", "press", "relative", "aliases",
            "different_pairs", "supported_payable")}
        errors = []
        labels = {label["candidate"]: label for label in row["factorized_labels"]}
        for candidate, label in labels.items():
            prediction = predictions[candidate]
            applicability, mode = prediction.applicability_decision, prediction.mode_decision
            expected = label["applicability"]
            category = "unknown" if expected == "unspecified" else expected
            counts[category]["total"] += 1
            if expected == "unspecified":
                correct = (not applicability.accepted
                           and applicability.reason == "insufficient_information")
            else:
                correct = applicability.accepted and applicability.value == expected
            counts[category]["correct"] += int(correct)
            if applicability.accepted and applicability.value != expected:
                errors.append({"candidate": candidate, "field": "applicability",
                               "expected": expected, "actual": applicability.value})
            if expected == "supported":
                counts["supported_payable"]["total"] += 1
                counts["supported_payable"]["correct"] += int(
                    correct and prediction.reward_eligible)
                expected_mode = label["mode"]
                if expected_mode != "unspecified":
                    counts[expected_mode]["total"] += 1
                    counts[expected_mode]["correct"] += int(
                        correct and mode.accepted and mode.value == expected_mode)
                if mode.accepted and mode.value != expected_mode:
                    errors.append({"candidate": candidate, "field": "mode",
                                   "expected": expected_mode, "actual": mode.value})
        observed_pairs = {frozenset((p.left, p.right)): p.decision
                          for p in table.pair_decisions}
        for pair in row["pair_labels"]:
            a, b = predictions[pair["left"]], predictions[pair["right"]]
            decision = observed_pairs.get(frozenset((pair["left"], pair["right"])))
            if decision is None:
                raise ValueError("runtime prediction is missing a supported pair")
            expected = pair["relation"]
            category = "aliases" if expected == "same" else "different_pairs"
            counts[category]["total"] += 1
            accepted = (a.status == b.status == "supported"
                        and a.reward_eligible and b.reward_eligible)
            same_group = a.group is not None and a.group == b.group
            counts[category]["correct"] += int(
                accepted and decision.accepted and decision.value == expected
                and same_group == (expected == "same"))
            if decision.accepted and decision.value != expected:
                errors.append({"pair": [pair["left"], pair["right"]], "field": "relation",
                               "expected": expected, "actual": decision.value})
            if same_group and expected != "same":
                errors.append({"alias_false_merge": [pair["left"], pair["right"]]})
            if accepted and expected == "same" and not same_group:
                errors.append({"alias_false_split": [pair["left"], pair["right"]]})
        passed = not errors and all(not value["total"] or
            value["correct"] / value["total"] >= (1.0 if name == "unknown" else .8)
            for name, value in counts.items())
        results.append({"task": row["task"], "language": row["language"],
                        "passed": passed, "counts": counts, "errors": errors})
    return {"version": RUNTIME_PROTOCOL_VERSION, **identity, "records": len(results),
            "passed": len(results) == 32 and all(row["passed"] for row in results),
            "passed_records": sum(row["passed"] for row in results), "results": results,
            "protocol": "frozen_exact_runtime_compatibility_no_precision_claim",
            "conditional_context": "global_applicability_no_condition_resolution",
            "used_for_selection": False, "precision_claim": False}


def verify_factorized_grounding(tokenizer, records, *, deadline=None,
                                splits=("train", "dev"), include_sealed=False):
    """Check each independently authored mention against actual E5 token masks.

    This catches losing the first `пробела` even if a later `пробел` grounds the
    same candidate. Expected spans come from the generator, never the matcher.
    """
    from carryrl.rewards.control_text_grounding import build_grounding

    checked, mentions, failures = 0, 0, []
    for row in records:
        permitted = set(splits) | ({"sealed_test"} if include_sealed else set())
        if row["split"] not in permitted:
            continue
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("independent mention transmission budget exhausted")
        encoded = tokenizer("query: " + row["text"], return_offsets_mapping=True,
                            return_tensors="pt", padding=False, truncation=False)
        offsets = encoded["offset_mapping"][0, encoded["attention_mask"][0].bool()].tolist()
        mask = build_grounding(row["text"], tokenizer, expected_tokens=len(offsets))
        for witness in row["expected_mentions"]:
            start, end = witness["start"] + 7, witness["end"] + 7
            tokens = [i for i, (left, right) in enumerate(offsets)
                      if right > left and left < end and right > start]
            candidate = witness["candidate"]
            transmitted = tokens and any(
                mask[CANDIDATES.index(candidate), token] for token in tokens)
            if not transmitted and len(failures) < 12:
                failures.append({"id": row["id"], "candidate": candidate,
                                 "span": [witness["start"], witness["end"]],
                                 "spelling": row["text"][witness["start"]:witness["end"]]})
            mentions += 1
        checked += 1
    if failures:
        raise ValueError(f"independent control mentions were not transmitted: {failures}")
    return {"passed": True, "records": checked, "expected_control_mentions": mentions,
            "missing_mentions": 0, "sealed_inputs_used": include_sealed,
            "splits": sorted(permitted),
            "expectation_source": "renderer exact character spans, independent of matcher",
            "scope": "input consistency, not learned semantic quality"}
