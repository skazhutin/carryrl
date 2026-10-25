"""Synthetic structural checks only: no fitting, game or sealed model evaluation."""

import hashlib
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from carryrl.rewards.control_text_data import semantic_mapping_identity
from carryrl.rewards.control_text_data_v4 import ControlSpec, OperationSpec
from carryrl.rewards.control_text_data_v6 import generate_control_text_records as generate_v6
from carryrl.rewards.control_text_data_v7 import (
    INTERNAL_FAMILIES,
    _binding,
    _sample_spec,
    annotate_spec,
    generate_control_text_records,
    prepare_control_text_dataset,
    render_spec,
    wording_manifest,
)
from carryrl.rewards.control_text_grounding import _mention_spans


@pytest.fixture(scope="module")
def records():
    return generate_control_text_records()


def test_v6_output_unchanged():
    rows = generate_v6(seed=20261022, scenario_groups=100)
    assert hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest() == ("9140992cf0df0c5fed7b7771539e8a765d731a773441e0c2322be93f009bf3b7")


def test_semantic_and_group_splits_are_unique(records):
    assert len(records) == 16000
    assert Counter(row["split"] for row in records) == {
        "train": 9600,
        "dev": 2400,
        "calibration": 1600,
        "sealed_test": 2400,
    }
    identities, groups, texts = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in records:
        identity = semantic_mapping_identity(row["labels"], row["declared_candidates"])
        assert row["semantic_mapping_id"] == identity
        identities[identity].add(row["split"])
        groups[row["group_id"]].add(identity)
        texts[row["text"]].add(identity)
        assert row["group_id"].startswith("v7-")
    assert len(identities) == len(groups) == 4000
    assert all(
        len(values) == 1 for table in (identities, groups, texts) for values in table.values()
    )
    assert set(Counter(row["group_id"] for row in records).values()) == {4}
    assert records[:240] == generate_control_text_records(scenario_groups=100)[:240]


def test_internal_families_are_held_out_and_change_actual_binding_syntax(records):
    split_families = defaultdict(set)
    for row in records:
        split_families[row["split"]].update(row["grammar_families"])
    for split, families in split_families.items():
        assert {f"v7-internal-{value}" for value in INTERNAL_FAMILIES[split]} <= families
        for other, other_families in split_families.items():
            if other != split:
                assert not families.intersection(other_families)
    operation = OperationSpec("abc", ("key:w", "key:q"), "held")
    cores = {_binding(operation, "en", family) for family in range(8)}
    assert len(cores) == 8
    # No split prefix/wrapper is involved in these different internal clauses.
    assert all("W key" in core and "Q key" in core and "hold" in core for core in cores)


def test_sparse_minimal_canonical_cores_and_modes_exist(records):
    summary = wording_manifest(records)
    for split in INTERNAL_FAMILIES:
        for kind, minimum in (("navigation", 8), ("horizontal", 4)):
            rows = [
                row
                for row in records[::4]
                if row["split"] == split and row["scenario_kind"] == kind
            ]
            assert rows
            minimal = [
                row
                for row in rows
                if sum(label["status"] == "supported" for label in row["labels"]) == minimum
            ]
            assert minimal
            assert any(row["specification"]["remainder"] == "unknown" for row in minimal)
            assert summary["canonical_supported_count_groups"][split][kind][str(minimum)]
            for candidate in ("key:a", "key:d"):
                assert {
                    label["mode"]
                    for row in rows
                    for label in row["labels"]
                    if label["candidate"] == candidate
                } == {"held", "press"}
    rng = random.Random(7)
    specs = [_sample_spec(rng, "navigation") for _ in range(100)]
    assert any(not spec.forbidden and len(spec.declared_candidates) == 8 for spec in specs)


def test_binding_modes_aliases_and_unknown_timing_are_independent():
    operations = (
        OperationSpec("aaa", ("key:w", "key:q"), "held"),
        OperationSpec("bbb", ("key:s", "key:up"), "press"),
        OperationSpec("ccc", ("key:a",), "unknown"),
        OperationSpec("ddd", ("dx:-",), "unknown"),
    )
    spec = ControlSpec(operations, ("key:d",), (), "unknown", "conditional")
    labels = {label["candidate"]: label for label in annotate_spec(spec)}
    assert labels["key:w"]["alias_group"] == labels["key:q"]["alias_group"]
    assert labels["key:s"]["alias_group"] == labels["key:up"]["alias_group"]
    assert labels["key:s"]["alias_group"] != labels["key:w"]["alias_group"]
    assert labels["key:w"]["mode"] == "held" and labels["key:s"]["mode"] == "press"
    assert labels["key:a"]["status"] == labels["key:d"]["status"] == "unknown"
    assert labels["key:a"]["alias_group"] == 0
    assert labels["dx:-"]["status"] == "supported" and labels["dx:-"]["mode"] == "unknown"
    for split in INTERNAL_FAMILIES:
        for variant, language in enumerate(("ru", "ru", "en", "en")):
            text, _ = render_spec(
                spec, language=language, variant=variant, split=split, seed=1, pair_seed="example"
            )
            assert ("удерж" if language == "ru" else "hold") in text
            assert ("отпусти" if language == "ru" else "release") in text
            assert ("неизвестн" if language == "ru" else "unknown") in text
            assert annotate_spec(spec) == list(labels.values())


def test_supported_controls_are_lexically_mentioned_without_goal_dependence(records):
    for row in records:
        spans = _mention_spans(row["text"])
        assert all(
            spans[label["candidate"]] for label in row["labels"] if label["status"] == "supported"
        ), row["text"]
    for text in (
        "Place the bronze marker inside the dotted frame.",
        "Помести бронзовую метку внутрь пунктирной рамки.",
    ):
        assert not any(_mention_spans(text).values())


def test_reservations_apply_to_every_split_and_manifest_is_authenticated(tmp_path):
    original = generate_control_text_records(scenario_groups=100)
    reserved = sorted({row["semantic_mapping_id"] for row in original})
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"mapping_ids": reserved}))
    output = tmp_path / "v7"
    manifest = prepare_control_text_dataset(
        output, scenario_groups=100, reservation_registry=registry
    )
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert not set(reserved).intersection(row["semantic_mapping_id"] for row in rows)
    assert (
        manifest["records_sha256"]
        == hashlib.sha256((output / "records.jsonl").read_bytes()).hexdigest()
    )
    assert (
        manifest["reservation_registry_sha256"] == hashlib.sha256(registry.read_bytes()).hexdigest()
    )
    assert manifest["excluded_semantic_mapping_count"] == 100
    assert manifest["version"] == "carryrl.control_text_data.v7"
    assert manifest["model_input_fields"] == ["text"]
    assert manifest["readiness"] is manifest["ppo_eligible"] is manifest["model_loaded"] is False
    assert manifest["updates"] == 0
    assert manifest["internal_grammar_families_disjoint"] is True
    with pytest.raises(FileExistsError):
        prepare_control_text_dataset(output, scenario_groups=100)


def test_cli_supports_v7(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "prepare_v7_cli",
        Path(__file__).resolve().parents[1] / "scripts/prepare_text_control_reward.py",
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert (
        script.main(
            [
                "--output",
                str(tmp_path / "cli"),
                "--corpus-version",
                "v7",
                "--scenario-groups",
                "100",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["version"] == "carryrl.control_text_data.v7"


def test_original_e5_context(records):
    from transformers import AutoTokenizer

    from carryrl.agent.language import TEXT_ID, TEXT_REVISION

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            TEXT_ID, revision=TEXT_REVISION, local_files_only=True
        )
    except OSError:
        pytest.skip("pinned original E5 tokenizer unavailable")
    lengths = list(
        map(
            len,
            tokenizer(
                ["query: " + row["text"] for row in records], truncation=False, padding=False
            )["input_ids"],
        )
    )
    assert max(lengths) < 384


@pytest.mark.parametrize("count", (0, 101, True, 50000))
def test_invalid_budget(count):
    with pytest.raises(ValueError):
        generate_control_text_records(scenario_groups=count)
