"""V8 corpus checks: structural partitioning, elementary coverage and wording."""

import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from carryrl.rewards.control_text_data import _semantic_split, semantic_mapping_identity
from carryrl.rewards.control_text_data_v4 import ControlSpec, OperationSpec
from carryrl.rewards.control_text_data_v8 import (
    INTERNAL_FAMILIES,
    PARTITION_REGISTRY_VERSION,
    PartitionRegistry,
    _binding,
    _identity,
    _resolved_elementary_specs,
    annotate_spec,
    elementary_specs,
    generate_control_text_records,
    prepare_control_text_dataset,
    read_partition_registry,
    render_spec,
    wording_manifest,
)
from carryrl.rewards.control_text_grounding import _mention_spans


def registry_file(tmp_path, *, hard=(), developmental=(), previous=None):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({
        "version": PARTITION_REGISTRY_VERSION,
        "hard_old_sealed_ids": list(hard),
        "developmental_ids": list(developmental),
        "previous_mapping_ids": list(previous if previous is not None else set(hard) | set(
            developmental
        )),
    }))
    return path


@pytest.fixture(scope="module")
def records():
    return generate_control_text_records(scenario_groups=200)


def test_registry_contract_authentication_and_strict_seal_priority(tmp_path):
    identity = "a" * 64
    path = registry_file(tmp_path, hard=[identity], developmental=[identity])
    registry, metadata = read_partition_registry(path)
    assert metadata["partition_registry_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["hard_seal_developmental_overlap_count"] == 1
    assert all(not registry.allows(identity, split) for split in INTERNAL_FAMILIES)
    developmental = PartitionRegistry(developmental_ids=frozenset([identity]),
                                      previous_mapping_ids=frozenset([identity]))
    assert developmental.allows(identity, "train")
    assert all(not developmental.allows(identity, split) for split in INTERNAL_FAMILIES
               if split != "train")
    with pytest.raises(ValueError, match="authenticated partition registry"):
        read_partition_registry(None)
    document = json.loads(path.read_text())
    for field, value in (("version", "old"), ("developmental_ids", "invalid"),
                         ("previous_mapping_ids", []), ("hard_old_sealed_ids", ["bad"])):
        malformed = {**document, field: value}
        path.write_text(json.dumps(malformed))
        with pytest.raises(ValueError):
            read_partition_registry(path)


def test_elementary_cores_bypass_hash_but_hard_seals_never_do():
    core = elementary_specs()[0]
    identity, _ = _identity(core)
    # A seed whose ordinary hash sends the clean core to evaluation must not
    # silently eliminate this foundational TRAIN rule.
    seed = next(seed for seed in range(100) if _semantic_split(identity, seed) != "train")
    rows = generate_control_text_records(seed=seed, scenario_groups=200)
    assert {row["split"] for row in rows if row["semantic_mapping_id"] == identity} == {"train"}
    blocked = PartitionRegistry(hard_old_sealed_ids=frozenset([identity]),
                                developmental_ids=frozenset([identity]),
                                previous_mapping_ids=frozenset([identity]))
    with pytest.raises(ValueError, match="mandatory clean TRAIN mapping blocked"):
        generate_control_text_records(scenario_groups=200, partition_registry=blocked)


def test_secondary_historical_seal_is_reported_and_partial_alternative_is_real():
    spec = next(spec for spec in elementary_specs()
                if spec.kind == "singleton_keys" and spec.operations[0].mode == "held")
    identity, _ = _identity(spec)
    registry = PartitionRegistry(hard_old_sealed_ids=frozenset([identity]),
                                 previous_mapping_ids=frozenset([identity]))
    selected, gaps = _resolved_elementary_specs(registry)
    assert len(gaps) == 1 and gaps[0]["blocked_mapping_id"] == identity
    alternate_id = gaps[0]["alternative_mapping_id"]
    assert alternate_id != identity
    alternate = next(spec for spec in selected if _identity(spec)[0] == alternate_id)
    assert alternate.remainder == "unknown"
    candidate = spec.operations[0].candidates[0]
    original_labels = {label["candidate"]: label for label in annotate_spec(spec)}
    partial_labels = {label["candidate"]: label for label in annotate_spec(alternate)}
    assert original_labels[candidate] == partial_labels[candidate]
    assert any(label["status"] == "unknown" for label in partial_labels.values())
    rows = generate_control_text_records(scenario_groups=200, partition_registry=registry)
    assert identity not in {row["semantic_mapping_id"] for row in rows}
    assert all(row["specification"]["remainder"] == "unknown" for row in rows
               if row["semantic_mapping_id"] == alternate_id)


def test_unique_effective_groups_and_variable_bilingual_variants(records):
    groups, identities, texts = defaultdict(list), defaultdict(set), defaultdict(set)
    for row in records:
        groups[row["group_id"]].append(row)
        identities[row["semantic_mapping_id"]].add(row["split"])
        texts[row["text"]].add(row["semantic_mapping_id"])
        assert row["semantic_mapping_id"] == semantic_mapping_identity(
            row["labels"], row["declared_candidates"]
        )
    assert len(groups) == len(identities) == 200
    assert all(len(splits) == 1 for splits in identities.values())
    assert all(len(values) == 1 for values in texts.values())
    assert Counter(rows[0]["split"] for rows in groups.values()) == {
        "train": 120, "dev": 30, "calibration": 20, "sealed_test": 30,
    }
    for rows in groups.values():
        expected = 8 if rows[0]["elementary"] else 6
        assert len(rows) == expected
        assert Counter(row["language"] for row in rows) == {
            "ru": expected // 2, "en": expected // 2,
        }
        assert {row["group_variants"] for row in rows} == {expected}
        assert len({row["text"] for row in rows}) == expected


def test_previous_exposure_ids_train_only_fresh_heldout_and_exact_quotas(records):
    # Treat a whole previous synthetic preparation as developmental. All three
    # new heldouts must exclude every previous effective identity.
    previous = frozenset(row["semantic_mapping_id"] for row in records)
    registry = PartitionRegistry(developmental_ids=previous, previous_mapping_ids=previous)
    fresh = generate_control_text_records(scenario_groups=200, partition_registry=registry)
    assert all(row["semantic_mapping_id"] not in previous for row in fresh
               if row["split"] != "train")
    groups = {row["group_id"]: row for row in fresh}.values()
    assert Counter(row["split"] for row in groups) == {
        "train": 120, "dev": 30, "calibration": 20, "sealed_test": 30,
    }
    assert all(row["split"] == "train" for row in fresh if row["developmental_compatibility"])


def test_twelve_train_compositions_and_actual_internal_holdouts(records):
    split_families = defaultdict(set)
    for row in records:
        split_families[row["split"]].add(row["template_family"])
    assert len(split_families["train"]) == 12
    for split, families in split_families.items():
        assert families == {f"v8-binding-composition-{value:02d}"
                            for value in INTERNAL_FAMILIES[split]}
        assert all(not families & others for other, others in split_families.items()
                   if other != split)
    operation = OperationSpec("anon", ("key:a", "key:left"), "held")
    assert len({_binding(operation, "en", family) for family in range(24)}) == 24
    assert len({_binding(operation, "ru", family) for family in range(24)}) == 24
    # Additional aggregate compositions are also owned by exactly one split.
    grammar_owners = defaultdict(set)
    for row in records:
        for family in row["grammar_families"]:
            grammar_owners[family].add(row["split"])
    assert all(len(owners) == 1 for owners in grammar_owners.values())


def test_all_supported_and_uncertain_mentions_survive_rendering(records):
    for row in records:
        spans = _mention_spans(row["text"])
        expected = {value for operation in row["specification"]["operations"]
                    for value in operation["candidates"]}
        expected.update(row["specification"]["uncertain"])
        assert all(spans[candidate] for candidate in expected), row["text"]
        assert "Для клавишу" not in row["text"]
        spec = row["specification"]
        assert len(spec["operations"]) <= 5
        assert len(spec["operations"]) + len(spec["uncertain"]) + len(spec["forbidden"]) <= 6


def test_key_modes_are_counterfactual_and_relative_labels_have_separate_contract():
    for mode in ("held", "press", "unknown"):
        spec = ControlSpec((OperationSpec("anon", ("key:w", "key:up"), mode),
                            OperationSpec("axis", ("dx:-",), "unknown")),
                           (), (), "unknown", "partial_unknown")
        labels = {label["candidate"]: label for label in annotate_spec(spec)}
        assert labels["key:w"]["mode"] == mode
        assert labels["key:w"]["status"] == ("unknown" if mode == "unknown" else "supported")
        assert labels["key:w"]["alias_group"] == labels["key:up"]["alias_group"]
        assert labels["dx:-"]["status"] == "supported"
        assert labels["dx:-"]["mode"] == "unknown"
        if mode != "unknown":
            assert labels["dx:-"]["alias_group"] != labels["key:w"]["alias_group"]


def test_main_budget_postfilter_coverage_and_calibration_signs():
    rows = generate_control_text_records(scenario_groups=4000)
    summary = wording_manifest(rows)
    assert 24000 <= len(rows) <= 32000
    assert min(summary["calibration_eligible_sign_groups"].values()) >= 300
    for split, coverage in summary["basic_coverage_post_filter"].items():
        assert coverage["held_groups"] and coverage["press_groups"]
        assert coverage["normal_direction_alias_groups"]
        quotas = summary["post_filter_group_quotas"][split]
        assert set(quotas) == set(summary["scenario_kinds"])
        assert quotas["cross_modal_alias"] / sum(quotas.values()) < 0.04
    assert summary["elementary_count"] == len(elementary_specs())
    assert summary["elementary_split_groups"] == {"train": len(elementary_specs())}


def test_preparation_authenticates_rows_registry_and_manifest(tmp_path):
    path = registry_file(tmp_path)
    output = tmp_path / "prepared"
    manifest = prepare_control_text_dataset(output, scenario_groups=200, reservation_registry=path)
    assert manifest["version"] == "carryrl.control_text_data.v8"
    records_hash = hashlib.sha256((output / "records.jsonl").read_bytes()).hexdigest()
    assert manifest["records_sha256"] == records_hash
    assert manifest["model_input_fields"] == ["text"]
    assert manifest["runtime_instructions_used"] is False
    assert manifest["template_families_disjoint"] is True
    assert manifest["historical_seal_elementary_gap_count"] == 0
    assert manifest["updates"] == 0
    assert manifest["model_loaded"] is manifest["readiness"] is manifest["ppo_eligible"] is False
    assert all(manifest["template_family_split_ownership"][family] == split
               for split, coverage in manifest["basic_coverage_post_filter"].items()
               for family in coverage["syntax_compositions"])
    from carryrl.rewards.control_text_training import read_control_dataset

    read_manifest, rows = read_control_dataset(output)
    assert read_manifest["records_sha256"] == manifest["records_sha256"]
    assert len(rows) == manifest["records"]
    with pytest.raises(FileExistsError):
        prepare_control_text_dataset(output, scenario_groups=200, reservation_registry=path)


def test_prepare_cli_v8_and_required_registry(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "prepare_v8_cli",
        Path(__file__).resolve().parents[1] / "scripts/prepare_text_control_reward.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    path = registry_file(tmp_path)
    assert script.main(["--output", str(tmp_path / "cli"), "--corpus-version", "v8",
                        "--scenario-groups", "200", "--reservation-registry", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == "carryrl.control_text_data.v8"
    with pytest.raises(ValueError, match="authenticated partition registry"):
        script.main(["--output", str(tmp_path / "missing"), "--corpus-version", "v8"])


def test_invalid_group_budget_is_rejected():
    for budget in (True, 100, 201, 40100):
        with pytest.raises(ValueError):
            generate_control_text_records(scenario_groups=budget)
    # All elementary variants remain specification labels when their kind changes.
    spec = elementary_specs()[0]
    assert annotate_spec(replace(spec, kind="different_wording")) == annotate_spec(spec)


def test_aggregate_mouse_motion_has_independent_axes_and_preserves_other_actions(records):
    axes = ("dx:-", "dx:+", "dy:-", "dy:+")
    aggregate_rows = [row for row in records if any(
        family.startswith("v8-aggregate-axes-") for family in row["grammar_families"]
    )]
    assert {row["split"] for row in aggregate_rows} >= {"train", "dev"}
    assert {row["language"] for row in aggregate_rows if row["split"] == "train"} == {"ru", "en"}
    for extra in ("key:space", "mouse:left"):
        spec = ControlSpec(tuple(OperationSpec(str(index), (axis,), "unknown")
                                 for index, axis in enumerate(axes)) + (
                                     OperationSpec("extra", (extra,), "press"),
                                 ), (), (), "unsupported", "mouse_axes")
        labels = {label["candidate"]: label for label in annotate_spec(spec)}
        assert len({labels[axis]["alias_group"] for axis in axes}) == 4
        assert all(labels[axis]["status"] == "supported" and
                   labels[axis]["mode"] == "unknown" for axis in axes)
        for split, families in INTERNAL_FAMILIES.items():
            for family in families:
                for language in ("ru", "en"):
                    text, grammars = render_spec(
                        spec, language=language, variant=family - families[0], split=split,
                        seed=5, pair_seed="0" * 64,
                    )
                    assert f"v8-aggregate-axes-{family:02d}" in grammars
                    mentions = _mention_spans(text)
                    assert all(mentions[axis] for axis in axes), text
                    assert mentions[extra], text
                    assert ("отпусти" if language == "ru" else "release") in text
                    assert annotate_spec(spec) == list(labels.values())
    incomplete = replace(spec, operations=spec.operations[1:])
    text, grammars = render_spec(incomplete, language="en", variant=0, split="train",
                                seed=5, pair_seed="0" * 64)
    assert not any(family.startswith("v8-aggregate-axes-") for family in grammars)
    assert not _mention_spans(text)[axes[0]]
