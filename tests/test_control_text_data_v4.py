"""Semantic-first v4 supervision, randomized timing and natural syntax."""

import hashlib
import importlib.util
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from carryrl.rewards.control_text_data import CONTROL_CANDIDATES, semantic_mapping_identity
from carryrl.rewards.control_text_data_v4 import (
    ControlSpec,
    OperationSpec,
    _sample_spec,
    annotate_spec,
    generate_control_text_records,
    prepare_control_text_dataset,
    render_spec,
)


@pytest.fixture(scope="module")
def records():
    return generate_control_text_records()


def test_semantics_are_unique_and_split_before_paraphrase(records):
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
        assert [label["candidate"] for label in row["labels"]] == list(CONTROL_CANDIDATES)
        assert row["group_id"] not in row["text"]
        assert row["semantic_mapping_id"] not in row["text"]
    assert len(identities) == len(groups) == 4000
    assert all(
        len(values) == 1 for table in (identities, groups, texts) for values in table.values()
    )
    assert all(count == 4 for count in Counter(row["group_id"] for row in records).values())
    assert records[:400] == generate_control_text_records()[:400]


def test_timing_is_independent_of_operation_position_and_identifier(records):
    positions, identifiers = defaultdict(Counter), defaultdict(Counter)
    for row in records[::4]:
        for index, operation in enumerate(row["specification"]["operations"]):
            if operation["mode"] == "unknown":
                continue
            positions[index][operation["mode"]] += 1
            identifiers[operation["identifier"]][operation["mode"]] += 1
    # Checks the actual sampled labels, including beyond former operation A/B/3.
    for counts in positions.values():
        assert 0.30 < counts["held"] / counts.total() < 0.70
    common = [counts for counts in identifiers.values() if counts.total() >= 20]
    assert len(common) >= 500
    assert sum(bool(counts["held"] and counts["press"]) for counts in common) == len(common)
    spec = ControlSpec(
        (OperationSpec("AA", ("key:a",), "held"), OperationSpec("BB", ("key:space",), "press")),
        (),
        ("key:q",),
        "unknown",
        "incomplete",
    )
    renamed = replace(
        spec,
        operations=tuple(
            replace(operation, identifier="ZZ" + str(index))
            for index, operation in enumerate(spec.operations)
        ),
    )
    assert annotate_spec(renamed) == annotate_spec(spec)
    reversed_spec = replace(spec, operations=spec.operations[::-1])
    assert semantic_mapping_identity(annotate_spec(reversed_spec)) == (
        semantic_mapping_identity(annotate_spec(spec))
    )


def test_natural_forms_encode_modes_and_aliases():
    spec = ControlSpec(
        (OperationSpec("QJ", ("key:a",), "held"), OperationSpec("LX", ("key:space",), "press")),
        ("key:q",),
        ("key:e",),
        "unknown",
        "incomplete",
    )
    text, _ = render_spec(spec, language="en", variant=0, split="train", seed=2)
    assert "Hold the A key" in text
    assert "Tap Space and release" in text
    assert "availability and timing are unknown" in text
    assert "Do not use the E key: unsupported" in text
    labels = {label["candidate"]: label for label in annotate_spec(spec)}
    assert labels["key:a"]["mode"] == "held"
    assert labels["key:space"]["mode"] == "press"
    assert labels["key:q"]["status"] == "unknown"
    assert labels["key:e"]["status"] == "unsupported"
    flipped = replace(
        spec, operations=(replace(spec.operations[0], mode="press"), spec.operations[1])
    )
    flipped_text, _ = render_spec(flipped, language="en", variant=0, split="train", seed=2)
    assert "Tap the A key and release" in flipped_text
    assert "Hold the A key" not in flipped_text
    assert annotate_spec(flipped) != annotate_spec(spec)
    for kind, phrase in (
        ("navigation", "WASD or arrow keys"),
        ("horizontal", "A/D or left/right arrows"),
    ):
        sampled = _sample_spec(random.Random(7), kind)
        text, _ = render_spec(sampled, language="en", variant=0, split="train", seed=3)
        assert phrase in text
        labels = {label["candidate"]: label for label in annotate_spec(sampled)}
        assert labels["key:a"]["alias_group"] == labels["key:left"]["alias_group"]
        assert labels["key:d"]["alias_group"] == labels["key:right"]["alias_group"]
        assert labels["key:a"]["alias_group"] != labels["key:d"]["alias_group"]


def test_all_modes_candidates_sizes_uncertainty_and_core_syntax_covered(records):
    for split in ("train", "dev", "calibration", "sealed_test"):
        subset = [row for row in records if row["split"] == split]
        sizes, coverage, grammars = set(), set(), set()
        for row in subset:
            sizes.add(sum(label["status"] == "supported" for label in row["labels"]))
            coverage.update(
                (label["candidate"], label["mode"])
                for label in row["labels"]
                if label["status"] == "supported"
            )
            grammars.update(row["grammar_families"])
        assert set(range(1, 14)) <= sizes
        assert {
            (candidate, mode) for candidate in CONTROL_CANDIDATES for mode in ("held", "press")
        } <= coverage
        assert {"imperative", "duration", "reference", split} <= grammars
        # The held/press grammar families occur inside TRAIN and all later splits.
        for mode in ("held", "press"):
            assert any(
                row["specification"]["operations"]
                and any(
                    operation["mode"] == mode for operation in row["specification"]["operations"]
                )
                for row in subset
                if "imperative" in row["grammar_families"]
            )
    calibration = [row for row in records[::4] if row["split"] == "calibration"]
    for status in ("supported", "unsupported"):
        assert (
            sum(any(label["status"] == status for label in row["labels"]) for row in calibration)
            >= 350
        )
    for row in records:
        for label in row["labels"]:
            if label["status"] != "supported":
                assert label["mode"] == "unknown" and label["alias_group"] == 0
    for row in records:
        for operation in row["specification"]["operations"]:
            if operation["mode"] == "unknown":
                for label in row["labels"]:
                    if label["candidate"] in operation["candidates"]:
                        assert label["status"] == "unknown"
                        assert label["mode"] == "unknown"
                        assert label["alias_group"] == 0


def test_manifest_is_authenticated_and_untrained(tmp_path):
    output = tmp_path / "v4"
    manifest = prepare_control_text_dataset(output, scenario_groups=100)
    assert manifest == json.loads((output / "manifest.json").read_text())
    assert (
        manifest["records_sha256"]
        == hashlib.sha256((output / "records.jsonl").read_bytes()).hexdigest()
    )
    assert manifest["version"] == "carryrl.control_text_data.v4"
    assert manifest["model_input_fields"] == ["text"]
    assert manifest["status"] == "prepared"
    assert manifest["updates"] == 0
    assert all(manifest[key] is False for key in ("readiness", "ppo_eligible", "model_loaded"))
    assert manifest["distributions"]["train"]["operation_modes_by_position"]
    with pytest.raises(FileExistsError):
        prepare_control_text_dataset(output, scenario_groups=100)


@pytest.mark.parametrize("flags, version", (((), "v3"), (("--corpus-version", "v4"), "v4")))
def test_cli_version_selection_preserves_historical_default(tmp_path, capsys, flags, version):
    script_spec = importlib.util.spec_from_file_location(
        "prepare_text_control_reward_script",
        Path(__file__).resolve().parents[1] / "scripts/prepare_text_control_reward.py",
    )
    script = importlib.util.module_from_spec(script_spec)
    script_spec.loader.exec_module(script)
    assert (
        script.main(["--output", str(tmp_path / version), "--scenario-groups", "100", *flags]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["version"] == f"carryrl.control_text_data.{version}"
    assert report["model_loaded"] is False


@pytest.mark.parametrize("count", (0, -100, 101, True, 50000))
def test_invalid_budget_rejected(count):
    with pytest.raises(ValueError):
        generate_control_text_records(scenario_groups=count)


def test_all_default_references_fit_local_e5_context(records):
    from transformers import AutoTokenizer

    from carryrl.agent.language import TEXT_ID, TEXT_REVISION

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            TEXT_ID, revision=TEXT_REVISION, local_files_only=True
        )
    except OSError:
        pytest.skip("pinned local tokenizer unavailable")
    tokens = tokenizer(
        ["query: " + row["text"] for row in records], padding=False, truncation=False
    )
    assert max(map(len, tokens["input_ids"])) <= 512
