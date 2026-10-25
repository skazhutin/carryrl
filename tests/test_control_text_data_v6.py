"""V6 composes grounded binding facts with scope, aliases and irrelevant context."""

import hashlib
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from carryrl.rewards.control_text_data import CONTROL_CANDIDATES, semantic_mapping_identity
from carryrl.rewards.control_text_data_v4 import ControlSpec, OperationSpec
from carryrl.rewards.control_text_data_v4 import generate_control_text_records as generate_legacy
from carryrl.rewards.control_text_data_v6 import (
    _binding,
    _sample_spec,
    annotate_spec,
    generate_control_text_records,
    prepare_control_text_dataset,
    read_reservation_registry,
    render_spec,
    wording_manifest,
)


@pytest.fixture(scope="module")
def records():
    return generate_control_text_records()


@pytest.mark.parametrize(
    "version, expected",
    (
        (4, "f3312f1da2b42c1a92c828195ddeff72fd40d9bb6b6c9a18119d45124055e680"),
        (5, "7720f1500ceae019363380d7ae3dd617623952f1723c107ddf3bea8e5a8d1c6a"),
    ),
)
def test_historical_v4_v5_outputs_unchanged(version, expected):
    # Recorded before introducing v6; covers labels, wording, order and metadata.
    rows = generate_legacy(seed=20261022, scenario_groups=100, wording_version=version)
    assert (
        hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        == expected
    )


def test_semantic_identity_splits_and_text_consistency(records):
    assert len(records) == 16000
    assert Counter(row["split"] for row in records) == {
        "train": 9600,
        "dev": 2400,
        "calibration": 1600,
        "sealed_test": 2400,
    }
    groups, semantics, texts = defaultdict(set), defaultdict(set), defaultdict(set)
    for row in records:
        identity = semantic_mapping_identity(row["labels"], row["declared_candidates"])
        assert identity == row["semantic_mapping_id"]
        groups[row["group_id"]].add(identity)
        semantics[identity].add(row["split"])
        texts[row["text"]].add(identity)
        assert [label["candidate"] for label in row["labels"]] == list(CONTROL_CANDIDATES)
    assert len(groups) == len(semantics) == 4000
    assert all(
        len(values) == 1 for table in (groups, semantics, texts) for values in table.values()
    )
    assert all(value == 4 for value in Counter(row["group_id"] for row in records).values())
    assert records[:400] == generate_control_text_records()[:400]


def test_sparse_partials_retain_positives_under_every_unknown_footer(records):
    for split in ("train", "dev", "calibration", "sealed_test"):
        sparse = [
            row
            for row in records
            if row["split"] == split and row["scenario_kind"] == "sparse_partial"
        ]
        assert sparse
        for row in sparse:
            spec = row["specification"]
            assert not spec["uncertain"] and not spec["forbidden"]
            assert spec["remainder"] == "unknown"
            counts = Counter(label["status"] for label in row["labels"])
            assert 1 <= counts["supported"] <= 3
            assert counts["unsupported"] == 0
            assert counts["unknown"] >= 33
    spec = ControlSpec(
        (OperationSpec("abc", ("key:w",), "held"),), (), (), "unknown", "sparse_partial"
    )
    labels = annotate_spec(spec)
    assert next(label for label in labels if label["candidate"] == "key:w")["status"] == "supported"
    for seed in range(32):
        for variant in range(4):
            text, _ = render_spec(
                spec, language="en", variant=variant, split="train", seed=seed, pair_seed=str(seed)
            )
            assert "W" in text
            assert annotate_spec(spec) == labels


def test_compact_or_is_scoped_with_and_between_distinct_groups():
    first = OperationSpec("abc", ("key:w", "key:up"), "held")
    second = OperationSpec("def", ("key:a", "key:left"), "press")
    assert _binding(first, "en", 0) == "Hold W or Up arrow"
    assert _binding(second, "en", 0) == "Press A or Left arrow once"
    spec = ControlSpec((first, second), (), (), "unknown", "scoped_alias")
    text, families = render_spec(
        spec, language="en", variant=0, split="train", seed=4, pair_seed="scope"
    )
    assert "; and " in text and " or " in text
    assert "scoped-or-with-and-between-operations" in families
    labels = {label["candidate"]: label for label in annotate_spec(spec)}
    assert labels["key:w"]["alias_group"] == labels["key:up"]["alias_group"]
    assert labels["key:a"]["alias_group"] == labels["key:left"]["alias_group"]
    assert labels["key:w"]["alias_group"] != labels["key:a"]["alias_group"]


@pytest.mark.parametrize(
    "mode, discrete, movement",
    (
        ("held", "Hold A", "repeat the selected input continuously"),
        ("press", "Press A once", "make one brief movement and stop"),
    ),
)
def test_mixed_aliases_state_discrete_and_relative_timing_separately(mode, discrete, movement):
    operation = OperationSpec("abc", ("key:a", "dx:-"), mode)
    text = _binding(operation, "en", 0)
    assert text.startswith("Equivalent input choices:")
    assert discrete in text and movement in text and " or " in text
    spec = ControlSpec((operation,), (), (), "unknown", "compact_alias")
    labels = {label["candidate"]: label for label in annotate_spec(spec)}
    assert labels["key:a"]["status"] == labels["dx:-"]["status"] == "supported"
    assert labels["key:a"]["alias_group"] == labels["dx:-"]["alias_group"]


def test_remote_separate_operations_have_distinct_signal_names():
    spec = ControlSpec(
        (OperationSpec("abc", ("key:a",), "press"), OperationSpec("def", ("key:d",), "press")),
        (),
        (),
        "unknown",
        "remote_press",
    )
    text, _ = render_spec(spec, language="en", variant=0, split="train", seed=1, pair_seed="x")
    assert "Differently named signals are independent operations." in text
    signals = re.findall(r"for the (\w+) signal", text)
    assert len(signals) == len(set(signals)) == 2
    assert "release it between activations" in text


def test_implicit_discrete_timing_abstains_but_relative_availability_does_not():
    candidates = ("key:w", "mouse:left", "dx:-", "dx:+", "dy:-", "dy:+", "wheel:+")
    spec = ControlSpec(
        tuple(
            OperationSpec(f"{i:03x}", (candidate,), "unknown")
            for i, candidate in enumerate(candidates)
        ),
        (),
        (),
        "unknown",
        "implicit",
    )
    labels = {label["candidate"]: label for label in annotate_spec(spec)}
    for candidate in candidates:
        label = labels[candidate]
        assert label["mode"] == "unknown"
        if candidate.startswith(("key:", "mouse:")):
            assert label["status"] == "unknown" and label["alias_group"] == 0
        else:
            assert label["status"] == "supported" and label["alias_group"] > 0
    for seed in range(20):
        implicit = _sample_spec(random.Random(seed), "implicit")
        assert all(label["status"] != "supported" for label in annotate_spec(implicit))


def test_generic_mouse_has_four_separate_signed_controls_even_without_timing(records):
    axes = ("dx:-", "dx:+", "dy:-", "dy:+")
    rows = [row for row in records if row["scenario_kind"] == "mouse_axes"]
    assert rows
    no_timing = 0
    for row in rows:
        labels = {label["candidate"]: label for label in row["labels"]}
        assert {labels[candidate]["status"] for candidate in axes} == {"supported"}
        assert len({labels[candidate]["alias_group"] for candidate in axes}) == 4
        assert all(labels[candidate]["alias_group"] > 0 for candidate in axes)
        assert all(
            not any(
                candidate in operation["candidates"] and len(operation["candidates"]) > 1
                for operation in row["specification"]["operations"]
            )
            for candidate in axes
        )
        assert {labels[candidate]["mode"] for candidate in axes} == {"unknown"}
        if (
            "timing is unspecified" in row["text"]
            or "Длительность движения не задана" in row["text"]
        ):
            no_timing += 1
    assert no_timing > 0


def test_canonical_group_summaries_keep_separate_alias_pairs_and_random_extras(records):
    summary = wording_manifest(records)
    for split in ("train", "dev", "calibration", "sealed_test"):
        assert sum(summary["canonical_group_patterns"][split].values()) == len(
            {
                row["group_id"]
                for row in records
                if row["split"] == split and row["scenario_kind"] in {"navigation", "horizontal"}
            }
        )
        for kind, pairs in (
            (
                "navigation",
                (
                    ("key:w", "key:up"),
                    ("key:s", "key:down"),
                    ("key:a", "key:left"),
                    ("key:d", "key:right"),
                ),
            ),
            ("horizontal", (("key:a", "key:left"), ("key:d", "key:right"))),
        ):
            rows = [
                row for row in records if row["split"] == split and row["scenario_kind"] == kind
            ]
            assert rows
            assert any(
                any(
                    family.startswith(f"canonical-{kind}-shared-")
                    for family in row["grammar_families"]
                )
                for row in rows
            )
            assert any(
                f"canonical-{kind}-expanded-mixed" in row["grammar_families"] for row in rows
            )
            for row in rows:
                labels = {label["candidate"]: label for label in row["labels"]}
                groups = []
                for first, second in pairs:
                    assert labels[first]["status"] == labels[second]["status"] == "supported"
                    assert labels[first]["alias_group"] == labels[second]["alias_group"]
                    groups.append(labels[first]["alias_group"])
                assert len(set(groups)) == len(pairs)
                assert sum(label["status"] == "supported" for label in row["labels"]) > 2 * len(
                    pairs
                )
                assert "WASD" in row["text"] if kind == "navigation" else "A/D" in row["text"]


def test_relative_duration_differences_share_one_runtime_semantic_identity():
    identities, vectors = set(), []
    for mode in ("held", "press", "unknown"):
        spec = ControlSpec(
            (OperationSpec("abc", ("dx:-",), mode), OperationSpec("def", ("dy:+",), mode)),
            (),
            (),
            "unknown",
            "mouse_axes",
        )
        labels = annotate_spec(spec)
        vectors.append(labels)
        identities.add(semantic_mapping_identity(labels, spec.declared_candidates))
    assert vectors[0] == vectors[1] == vectors[2]
    assert len(identities) == 1


def test_exclusions_resample_all_splits_before_rendering():
    original = generate_control_text_records(scenario_groups=100)
    excluded = {row["semantic_mapping_id"] for row in original}
    fresh = generate_control_text_records(scenario_groups=100, excluded_semantic_ids=excluded)
    assert len(fresh) == 400
    assert not excluded.intersection(row["semantic_mapping_id"] for row in fresh)
    assert Counter(row["split"] for row in original) == Counter(row["split"] for row in fresh)


def test_context_remote_release_click_mixed_and_dense_exist_in_every_split(records):
    for split in ("train", "dev", "calibration", "sealed_test"):
        rows = [row for row in records if row["split"] == split]
        families = {family for row in rows for family in row["grammar_families"]}
        assert {
            "irrelevant-goal-and-distractor-context",
            "remote-release-between",
            "guaranteed-phase-transition",
            "generic-independent-mouse-axes",
            "scoped-or-with-and-between-operations",
            "ru-singular-imperative",
        } <= families
        assert any("Click " in row["text"] for row in rows)
        assert any("release it between activations" in row["text"] for row in rows)
        assert any("Удерживай " in row["text"] for row in rows)
        assert any("Нажимай " in row["text"] for row in rows)
        sizes = {sum(label["status"] == "supported" for label in row["labels"]) for row in rows}
        assert set(range(1, 14)) <= sizes
        for row in rows:
            if row["scenario_kind"] == "mixed":
                supported = {
                    label["candidate"] for label in row["labels"] if label["status"] == "supported"
                }
                assert all(
                    any(candidate.startswith(prefix) for candidate in supported)
                    for prefix in ("key:", "mouse:", "wheel:")
                )
        phase = [row for row in rows if "guaranteed-phase-transition" in row["grammar_families"]]
        assert all(any(label["status"] == "supported" for label in row["labels"]) for row in phase)
        conditional = [row for row in rows if row["scenario_kind"] == "conditional"]
        for row in conditional:
            assert all(
                label["status"] == "unknown"
                for label in row["labels"]
                if label["candidate"] in row["specification"]["uncertain"]
            )


def test_calibration_counts_are_real_independent_scenarios(records):
    calibration = [row for row in records[::4] if row["split"] == "calibration"]
    eligible = {
        status: {
            row["semantic_mapping_id"]
            for row in calibration
            if any(label["status"] == status for label in row["labels"])
        }
        for status in ("supported", "unsupported")
    }
    assert {status: len(groups) for status, groups in eligible.items()} == {
        "supported": 360,
        "unsupported": 340,
    }
    assert wording_manifest(records)["calibration_eligible_sign_groups"] == {
        status: len(groups) for status, groups in eligible.items()
    }


def test_public_texts_not_copied_and_basic_modes_all_candidates_covered(records):
    public = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/atomic_instructions.json").read_text()
    )
    assert not {row["text"] for row in records}.intersection(
        text for variants in public.values() for text in variants
    )
    train = [row for row in records if row["split"] == "train"]
    coverage = {
        (label["candidate"], label["mode"])
        for row in train
        for label in row["labels"]
        if label["status"] == "supported"
    }
    assert {
        (candidate, mode)
        for candidate in CONTROL_CANDIDATES
        if candidate.startswith(("key:", "mouse:"))
        for mode in ("held", "press")
    } <= coverage
    assert {
        (candidate, "unknown")
        for candidate in CONTROL_CANDIDATES
        if candidate.startswith(("dx:", "dy:", "wheel:"))
    } <= coverage


def test_manifest_and_real_loader_accept_v6_without_model_loading(tmp_path):
    from carryrl.rewards.control_text_training import read_control_dataset

    output = tmp_path / "v6"
    manifest = prepare_control_text_dataset(output, scenario_groups=100)
    loaded_manifest, rows = read_control_dataset(output)
    assert loaded_manifest == manifest
    assert len(rows) == 400
    assert manifest["version"] == "carryrl.control_text_data.v6"
    assert (
        manifest["records_sha256"]
        == hashlib.sha256((output / "records.jsonl").read_bytes()).hexdigest()
    )
    assert manifest["model_input_fields"] == ["text"]
    assert manifest["readiness"] is manifest["ppo_eligible"] is manifest["model_loaded"] is False
    assert manifest["updates"] == 0
    with pytest.raises(FileExistsError):
        prepare_control_text_dataset(output, scenario_groups=100)


def test_cli_v6(tmp_path, capsys):
    script_spec = importlib.util.spec_from_file_location(
        "prepare_v6_cli",
        Path(__file__).resolve().parents[1] / "scripts/prepare_text_control_reward.py",
    )
    script = importlib.util.module_from_spec(script_spec)
    script_spec.loader.exec_module(script)
    assert (
        script.main(
            [
                "--output",
                str(tmp_path / "cli"),
                "--corpus-version",
                "v6",
                "--seed",
                "20261022",
                "--scenario-groups",
                "100",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["version"] == "carryrl.control_text_data.v6"


def test_registry_hash_counts_and_exclusions_are_authenticated(tmp_path):
    original = generate_control_text_records(scenario_groups=100)
    excluded = sorted({row["semantic_mapping_id"] for row in original})
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"mapping_ids": excluded}))
    exclusions, metadata = read_reservation_registry(registry)
    assert exclusions == frozenset(excluded)
    assert (
        metadata["reservation_registry_sha256"] == hashlib.sha256(registry.read_bytes()).hexdigest()
    )
    assert metadata["reservation_registry_mapping_count"] == len(excluded) == 100
    output = tmp_path / "fresh"
    manifest = prepare_control_text_dataset(
        output, scenario_groups=100, reservation_registry=registry
    )
    rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    assert manifest["reservation_registry_sha256"] == metadata["reservation_registry_sha256"]
    assert manifest["excluded_semantic_mapping_count"] == len(excluded)
    assert not exclusions.intersection(row["semantic_mapping_id"] for row in rows)
    assert manifest["excluded_semantic_mappings_observed"] == 0


def test_original_e5_context(records):
    from transformers import AutoTokenizer

    from carryrl.agent.language import TEXT_ID, TEXT_REVISION

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            TEXT_ID, revision=TEXT_REVISION, local_files_only=True
        )
    except OSError:
        pytest.skip("pinned original E5 tokenizer unavailable")
    tokens = tokenizer(
        ["query: " + row["text"] for row in records], truncation=False, padding=False
    )
    assert max(map(len, tokens["input_ids"])) < 384


@pytest.mark.parametrize("count", (0, 101, True, 50000))
def test_invalid_budget(count):
    with pytest.raises(ValueError):
        generate_control_text_records(scenario_groups=count)
