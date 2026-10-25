"""Independent public-spec compatibility checks; never imported by learned inference."""

import hashlib
import json
import time
from pathlib import Path

from carryrl.rewards.control_text import CANDIDATES, content_hash


def runtime_control_rows(instructions, controls):
    """Label exact bilingual runtime text from separately authored control specs.

    No hidden state is supplied. Conditional bindings therefore remain unknown;
    only a complete specification licenses unsupported labels for omitted input.
    """
    instructions, controls = Path(instructions), Path(controls)
    texts = json.loads(instructions.read_text())
    games = json.loads(controls.read_text())["games"]
    if set(texts) != set(games) or len(games) != 16:
        raise ValueError("runtime control gate requires the same sixteen games")
    rows = []
    for task, variants in texts.items():
        if len(variants) != 2 or not all(
            isinstance(text, str) and text.strip() for text in variants
        ):
            raise ValueError("runtime gate requires two nonempty RU/EN instructions per game")
        spec = games[task]
        labels = {
            candidate: {
                "candidate": candidate,
                "status": "unsupported" if spec.get("complete") is True else "unknown",
                "mode": "unknown",
                "alias_group": 0,
            }
            for candidate in CANDIDATES
        }
        assigned = set()
        for number, group in enumerate(spec["groups"], 1):
            if group["mode"] not in {"held", "press"}:
                raise ValueError("runtime spec has invalid input mode")
            for candidate in group["candidates"]:
                if candidate not in labels or candidate in assigned:
                    raise ValueError("runtime spec has invalid or duplicated candidates")
                assigned.add(candidate)
                if group.get("when"):
                    labels[candidate].update(status="unknown", mode="unknown", alias_group=0)
                else:
                    labels[candidate].update(
                        status="supported", mode=group["mode"], alias_group=number
                    )
        for language, text in zip(("ru", "en"), variants, strict=True):
            rows.append(
                {
                    "group_id": task,
                    "task": task,
                    "language": language,
                    "text": text,
                    "labels": list(labels.values()),
                }
            )
    identity = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (("instructions_sha256", instructions), ("controls_sha256", controls))
    }
    return rows, {**identity, "specification_hash": content_hash(identity)}


def validate_runtime_text_controls(predictor, *, instructions, controls, deadline=None):
    """All 32 exact texts, zero wrong assertions and >=80% useful slice coverage.

    This compatibility suite does not supply independent statistical precision
    evidence and cannot replace the synthetic calibration gate.
    """
    rows, identity = runtime_control_rows(instructions, controls)
    results = []
    for row in rows:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("runtime text compatibility budget exhausted")
        table = predictor.predict(row["text"])
        predictions = {value.candidate: value for value in table.predictions}
        if set(predictions) != set(CANDIDATES):
            raise ValueError("runtime prediction candidates differ from specification")
        counts = {
            name: {"total": 0, "correct": 0}
            for name in ("supported", "unsupported", "unknown", "held", "press", "aliases")
        }
        errors = []
        for label in row["labels"]:
            candidate = label["candidate"]
            prediction = predictions[candidate]
            status = label["status"]
            correct = prediction.status == status
            counts[status]["total"] += 1
            counts[status]["correct"] += int(correct)
            mode_relevant = candidate.startswith(("key:", "mouse:"))
            if status == "supported" and mode_relevant:
                counts[label["mode"]]["total"] += 1
                counts[label["mode"]]["correct"] += int(
                    correct and prediction.mode == label["mode"]
                )
            if (prediction.status != "unknown" and not correct) or (
                correct
                and mode_relevant
                and status == "supported"
                and prediction.mode not in {label["mode"], "unknown"}
            ):
                errors.append(
                    {
                        "candidate": candidate,
                        "expected_status": status,
                        "actual_status": prediction.status,
                        "expected_mode": label["mode"],
                        "actual_mode": prediction.mode,
                    }
                )
        for index, left in enumerate(row["labels"]):
            for right in row["labels"][index + 1 :]:
                a, b = predictions[left["candidate"]], predictions[right["candidate"]]
                expected = left["alias_group"] > 0 and left["alias_group"] == right["alias_group"]
                actual = a.group is not None and a.group == b.group
                if expected:
                    counts["aliases"]["total"] += 1
                    counts["aliases"]["correct"] += int(
                        actual and a.status == b.status == "supported"
                    )
                # Accepted aliases must not merge distinct or unknown operations.
                if actual and not expected:
                    errors.append({"alias_false_merge": [left["candidate"], right["candidate"]]})
                if (
                    expected
                    and a.status == b.status == "supported"
                    and a.reward_eligible
                    and b.reward_eligible
                    and not actual
                ):
                    errors.append({"alias_false_split": [left["candidate"], right["candidate"]]})
        passed = not errors and all(
            not value["total"]
            or value["correct"] / value["total"] >= (1 if name == "unknown" else 0.8)
            for name, value in counts.items()
        )
        results.append(
            {
                "task": row["task"],
                "language": row["language"],
                "passed": passed,
                "counts": counts,
                "errors": errors,
            }
        )
    return {
        "version": "carryrl.runtime_text_controls.v1",
        **identity,
        "records": len(results),
        "passed": all(row["passed"] for row in results),
        "passed_records": sum(row["passed"] for row in results),
        "results": results,
        "protocol": "frozen_exact_runtime_compatibility_no_precision_claim",
        "conditional_context": "layout_unobserved_unknown",
        "used_for_selection": False,
    }
