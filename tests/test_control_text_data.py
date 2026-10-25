"""Offline preparation, semantic supervision and split isolation checks."""

import hashlib
import json
from collections import Counter, defaultdict

import pytest

from carryrl.agent.actions import KEY_VOCAB
from carryrl.rewards.control_text_data import (
    CONTROL_CANDIDATES,
    SCENARIO_KINDS,
    generate_control_text_records,
    prepare_control_text_dataset,
    semantic_mapping_identity,
)


def test_candidate_order_and_deterministic_grouped_splits():
    records = generate_control_text_records()
    assert len(records) == 16000
    assert records == generate_control_text_records()
    assert CONTROL_CANDIDATES[:27] == tuple(f"key:{key}" for key in KEY_VOCAB)
    assert len(CONTROL_CANDIDATES) == 36
    assert Counter(row["split"] for row in records) == {
        "train": 9600,
        "dev": 2400,
        "calibration": 1600,
        "sealed_test": 2400,
    }
    groups, mappings, families, texts, semantics = (defaultdict(set) for _ in range(5))
    for row in records:
        groups[row["group_id"]].add(row["split"])
        mappings[row["mapping_id"]].add(row["split"])
        assert row["semantic_mapping_id"] == semantic_mapping_identity(
            row["labels"], row["declared_candidates"]
        )
        semantics[row["semantic_mapping_id"]].add(row["split"])
        families[row["template_family"]].add(row["split"])
        texts[row["text"]].add(row["split"])
        assert [label["candidate"] for label in row["labels"]] == list(CONTROL_CANDIDATES)
        for label in row["labels"]:
            assert label["status"] in {"supported", "unsupported", "unknown"}
            assert label["mode"] in {"held", "press", "unknown"}
            if label["status"] != "supported":
                assert label["mode"] == "unknown"
                assert label["alias_group"] == 0
    assert len(groups) == len(mappings) == len(semantics) == 4000
    assert all(
        len(splits) == 1
        for table in (groups, mappings, families, texts, semantics)
        for splits in table.values()
    )
    for split in ("train", "dev", "calibration", "sealed_test"):
        subset = [row for row in records if row["split"] == split]
        assert {row["scenario_kind"] for row in subset} == set(SCENARIO_KINDS)
    assert all(count == 4 for count in Counter(row["group_id"] for row in records).values())


def test_partial_conflicting_negative_and_alias_annotations():
    records = generate_control_text_records(scenario_groups=100)
    by_kind = {row["scenario_kind"]: row for row in records}
    assert all(label["status"] == "unknown" for label in by_kind["unknown"]["labels"])
    assert Counter(label["status"] for label in by_kind["incomplete"]["labels"]) == {
        "supported": 2,
        "unknown": 33,
        "unsupported": 1,
    }
    assert Counter(label["status"] for label in by_kind["conflict"]["labels"]) == {
        "supported": 2,
        "unknown": 33,
        "unsupported": 1,
    }
    assert Counter(label["status"] for label in by_kind["negation"]["labels"]) == {
        "supported": 2,
        "unsupported": 34,
    }
    supported = [label for label in by_kind["aliases"]["labels"] if label["status"] == "supported"]
    assert len(supported) == 2
    assert supported[0]["alias_group"] == supported[1]["alias_group"] != 0
    assert {
        label["mode"] for label in by_kind["press"]["labels"] if label["status"] == "supported"
    } == {"press"}
    for row in records:
        assert row["group_id"] not in row["text"]
        assert row["mapping_id"] not in row["text"]
        assert row["template_family"] not in row["text"]


def test_preparation_manifest_is_unready_and_authenticated(tmp_path):
    output = tmp_path / "prepared"
    manifest = prepare_control_text_dataset(output, scenario_groups=100)
    assert manifest == json.loads((output / "manifest.json").read_text())
    assert (
        manifest["records_sha256"]
        == hashlib.sha256((output / "records.jsonl").read_bytes()).hexdigest()
    )
    assert manifest["readiness"] is False
    assert manifest["ppo_eligible"] is False
    assert manifest["model_loaded"] is False
    assert manifest["updates"] == 0
    assert sorted(path.name for path in output.iterdir()) == ["manifest.json", "records.jsonl"]
    with pytest.raises(FileExistsError):
        prepare_control_text_dataset(output, scenario_groups=100)


def test_invalid_group_budget_rejected():
    for count in (0, -100, 101, True, 50000):
        with pytest.raises(ValueError):
            generate_control_text_records(scenario_groups=count)


def test_public_offline_specs_match_implemented_input_sets():
    from pathlib import Path

    from carryrl.envs.atomic import _TASK_KEYS
    from carryrl.envs.atomic_suite import TASKS

    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "configs/atomic_controls.json").read_text())
    assert set(spec["games"]) == set(TASKS)
    for task, controls in spec["games"].items():
        assert controls["complete"] is True
        candidates = {
            candidate for group in controls["groups"] for candidate in group["candidates"]
        }
        assert candidates <= set(CONTROL_CANDIDATES)
        assert {candidate[4:] for candidate in candidates if candidate.startswith("key:")} == (
            set(_TASK_KEYS[task])
        )
        assert {candidate for candidate in candidates if candidate.startswith("mouse:")} == (
            {"mouse:left"} if task in {"mouse", "shooting"} else set()
        )
        assert {candidate for candidate in candidates if candidate.startswith(("dx:", "dy:"))} == (
            {"dx:-", "dx:+", "dy:-", "dy:+"} if task in {"mouse", "camera", "shooting"} else set()
        )
        assert not any(candidate.startswith("wheel:") for candidate in candidates)
        for group in controls["groups"]:
            assert group["mode"] in {"held", "press"}
    for layout in (0, 1):
        ui = spec["games"]["ui"]["groups"]
        keys = {
            candidate
            for group in ui
            if group.get("when", {}).get("layout", layout) == layout
            for candidate in group["candidates"]
        }
        expected = {"key:tab", "key:enter", "key:escape"}
        expected |= (
            {"key:left", "key:a", "key:right", "key:d"}
            if layout == 0
            else {"key:up", "key:w", "key:down", "key:s"}
        )
        assert keys == expected


def test_semantic_identity_ignores_inactive_binding_and_candidate_order():
    records = generate_control_text_records(scenario_groups=100)
    partial = next(row for row in records if row["scenario_kind"] == "incomplete")
    assert semantic_mapping_identity(partial["labels"], ["key:up"]) == (
        semantic_mapping_identity(list(reversed(partial["labels"])), ["key:down"])
    )
    unknown = next(row for row in records if row["scenario_kind"] == "unknown")
    assert semantic_mapping_identity(unknown["labels"], ["key:up"]) != (
        semantic_mapping_identity(unknown["labels"], ["key:down"])
    )


def test_default_calibration_has_enough_independent_clusters_for_both_signs():
    records = generate_control_text_records()
    calibration = {row["group_id"]: row for row in records if row["split"] == "calibration"}
    assert len(calibration) == 400
    assert len({row["semantic_mapping_id"] for row in calibration.values()}) == 400
    for status in ("supported", "unsupported"):
        eligible = [
            row
            for row in calibration.values()
            if any(label["status"] == status for label in row["labels"])
        ]
        assert len(eligible) >= 350
        assert len({row["semantic_mapping_id"] for row in eligible}) >= 299
    for row in calibration.values():
        if row["scenario_kind"] != "unknown":
            counts = Counter(label["status"] for label in row["labels"])
            assert counts["supported"] >= 2
            assert counts["unsupported"] >= 1


def test_dense_mixed_control_sets_and_multiple_alias_groups_are_represented():
    rows = generate_control_text_records(scenario_groups=100)
    for kind in ("dense", "mixed_mouse", "abbreviations", "conditional"):
        group = [row for row in rows if row["scenario_kind"] == kind]
        assert group
        for row in group:
            supported = [label for label in row["labels"] if label["status"] == "supported"]
            assert len(supported) >= (4 if kind == "conditional" else 8)
            assert len({label["alias_group"] for label in supported}) >= 2
            if kind == "mixed_mouse":
                assert any(label["candidate"].startswith("mouse:") for label in supported)
            if kind == "conditional":
                assert sum(label["status"] == "unknown" for label in row["labels"]) == 4


def test_semantic_identity_normalizes_local_alias_identifiers():
    row = next(
        row
        for row in generate_control_text_records(scenario_groups=100)
        if row["scenario_kind"] == "dense"
    )
    labels = [
        {**label, "alias_group": label["alias_group"] + 50 if label["alias_group"] else 0}
        for label in row["labels"]
    ]
    assert semantic_mapping_identity(labels) == semantic_mapping_identity(row["labels"])


def test_dense_references_fit_pinned_local_e5_token_budget():
    from transformers import AutoTokenizer

    from carryrl.agent.language import TEXT_ID, TEXT_REVISION

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            TEXT_ID, revision=TEXT_REVISION, local_files_only=True
        )
    except OSError:
        pytest.skip("pinned local tokenizer unavailable")
    rows = generate_control_text_records(scenario_groups=100)
    tokens = tokenizer(["query: " + row["text"] for row in rows], padding=False, truncation=False)
    assert max(map(len, tokens["input_ids"])) <= 512
