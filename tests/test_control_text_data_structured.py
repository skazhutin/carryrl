"""Concrete V12 effects preserve supervision and every nonselected record byte."""

import hashlib
import json
import random
from dataclasses import asdict

import pytest

from carryrl.rewards.control_text_data import semantic_mapping_identity
from carryrl.rewards.control_text_data_contrastive import (
    _paired_specs,
    prepare_contrastive_control_dataset,
)
from carryrl.rewards.control_text_data_natural import prepare_natural_control_dataset
from carryrl.rewards.control_text_data_structured import (
    FAMILIES,
    prepare_structured_control_dataset,
    render_structured_view,
    structured_effects,
)
from carryrl.rewards.control_text_data_transfer import prepare_transfer_control_dataset
from carryrl.rewards.control_text_data_v4 import ControlSpec, OperationSpec
from carryrl.rewards.control_text_data_v8 import (
    _NAVIGATION,
    PARTITION_REGISTRY_VERSION,
    annotate_spec,
    prepare_control_text_dataset,
)
from carryrl.rewards.control_text_data_v9 import prepare_relation_grounded_dataset
from carryrl.rewards.control_text_data_v10 import prepare_canonical_repaired_dataset
from carryrl.rewards.control_text_grounding import _mention_spans, build_grounding
from carryrl.rewards.control_text_training import read_control_dataset


def _row(spec, *, variant=0, language="ru", pair_id=None):
    labels = annotate_spec(spec)
    identity = semantic_mapping_identity(labels, spec.declared_candidates)
    row = {"group_id": "example", "split": "train", "variant_index": variant,
           "group_variants": 6, "language": language, "specification": asdict(spec),
           "labels": labels, "text": "PRIVATE OLD TEXT", "mapping_id": identity,
           "semantic_mapping_id": identity, "declared_candidates": list(spec.declared_candidates),
           "template_family": "v8-binding-composition-00",
           "grammar_families": ["v8-binding-composition-00", "v9-relation-witness-00"]}
    if pair_id:
        row["contrastive_pair_id"] = pair_id
    return row


def test_concrete_effects_stable_in_every_view_and_distinct_for_every_group():
    spec = ControlSpec((OperationSpec("one", ("key:w", "mouse:left"), "held"),
                        OperationSpec("two", ("key:q",), "press"),
                        OperationSpec("three", ("dx:-",), "unknown")),
                       (), (), "unsupported", "mixed")
    expected = structured_effects(_row(spec))
    frames = set()
    for variant in (0, 1, 3, 4):
        for language in ("ru", "en"):
            old = _row(spec, variant=variant, language=language)
            assert structured_effects(old) == expected
            new, info = render_structured_view(old)
            frames.add(info["frame"])
            assert len(set(effect[0] for _, effect in expected)) == len(expected)
            for _, effect in expected:
                assert effect[1 if language == "ru" else 2] in new["text"]
            assert "different effects" in new["text"] if language == "en" else (
                "разные эффекты" in new["text"]
            )
            assert "interchangeable" in new["text"] if language == "en" else (
                "равнозначны" in new["text"]
            )
            assert "[" not in new["text"] and "]" not in new["text"]
            assert "PRIVATE OLD TEXT" not in new["text"]
            assert new["labels"] == old["labels"]
            assert semantic_mapping_identity(new["labels"], new["declared_candidates"]) == (
                old["semantic_mapping_id"]
            )
    assert frames == {0, 1, 2, 3}


def test_canonical_and_relative_actions_do_not_collapse_different_effects():
    spec = ControlSpec(tuple(OperationSpec(str(index), group, "held")
                             for index, group in enumerate(_NAVIGATION)) + (
                                 OperationSpec("extra", ("dx:+",), "unknown"),
                             ), (), (), "unsupported", "navigation")
    effects = structured_effects(_row(spec))
    assert {effect[0] for _, effect in effects} == {
        "token_up", "token_down", "token_left", "token_right", "marker_right",
    }
    assert len({effect[1] for _, effect in effects}) == len(effects)
    assert len({effect[2] for _, effect in effects}) == len(effects)


def test_matched_contexts_keep_effect_meanings_and_change_only_gold_relation():
    for kind in ("cross_modal_alias", "explicit_status", "adjacent_timing"):
        specs, _ = _paired_specs(random.Random(21), kind)
        rows = [_row(spec, pair_id="same-context") for spec in specs]
        effects = [structured_effects(row) for row in rows]
        assert effects[0][0][1] == effects[1][0][1]
        if kind == "cross_modal_alias":
            assert len(effects[0]) == 1 and len(effects[1]) == 2
            assert effects[0][0][1] != effects[1][1][1]
        for variant in (0, 1, 3, 4):
            generated = [
                render_structured_view({**row, "variant_index": variant})[0] for row in rows
            ]
            assert generated[0]["template_family"] == generated[1]["template_family"]
            assert all(old["labels"] == new["labels"]
                       for old, new in zip(rows, generated, strict=True))


def test_unknown_availability_timing_forbidden_and_remainder_facts_remain_explicit():
    spec = ControlSpec((OperationSpec("one", ("key:w",), "unknown"),
                        OperationSpec("two", ("dx:-",), "unknown")),
                       ("key:q",), ("mouse:left",), "unknown", "conditional")
    for language in ("ru", "en"):
        old = _row(spec, language=language)
        new, _ = render_structured_view(old)
        assert new["labels"] == old["labels"]
        if language == "ru":
            assert "указан как доступный" in new["text"]
            assert "режим нажатия не описан" in new["text"]
            assert "доступность не установлена" in new["text"] and "запрещён" in new["text"]
        else:
            assert "listed as available" in new["text"]
            assert "press timing is not specified" in new["text"]
            assert "availability is not established" in new["text"] and "unsupported" in new["text"]
        spans = _mention_spans(new["text"])
        assert all(spans[candidate] for candidate in spec.declared_candidates)
        assert not _mention_spans(new["text"])["key:enter"]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    from carryrl.agent.language import TEXT_ID, TEXT_REVISION

    try:
        return AutoTokenizer.from_pretrained(TEXT_ID, revision=TEXT_REVISION, local_files_only=True)
    except OSError:
        pytest.skip("original E5 tokenizer is not cached locally")


def test_real_e5_bounds_deferred_effect_reference_and_case_safe_grounding(tokenizer):
    spec = ControlSpec((OperationSpec("one", ("key:w", "mouse:left"), "press"),
                        OperationSpec("two", ("key:q",), "held"),
                        OperationSpec("three", ("dy:+",), "unknown"),
                        OperationSpec("four", ("key:space",), "press"),
                        OperationSpec("five", ("key:up",), "held")),
                       (), (), "unsupported", "mixed")
    for language in ("ru", "en"):
        delayed = False
        for variant in (0, 1, 3, 4):
            old = _row(spec, language=language, variant=variant)
            new, info = render_structured_view(old)
            encoded = tokenizer("query: " + new["text"], return_offsets_mapping=True,
                                truncation=False)
            assert len(encoded["input_ids"]) <= 512
            grounding = build_grounding(new["text"], tokenizer)
            assert grounding.any()
            mentions = _mention_spans(new["text"])
            assert all(mentions[candidate] for candidate in spec.declared_candidates)
            assert "Для клавишу" not in new["text"]
            if info.get("timing_rule_text"):
                delayed = True
                named = info["named_input_text"]
                named_end = new["text"].index(named) + len(named)
                end = 7 + named_end
                start = 7 + new["text"].index(info["timing_rule_text"])
                distance = sum(a >= end and b <= start and b > a
                               for a, b in encoded["offset_mapping"])
                assert distance > 24
        assert delayed


@pytest.fixture(scope="module")
def mini_v11(tmp_path_factory):
    root = tmp_path_factory.mktemp("structured")
    registry = root / "registry.json"
    registry.write_text(json.dumps({"version": PARTITION_REGISTRY_VERSION,
                                    "hard_old_sealed_ids": [], "developmental_ids": [],
                                    "previous_mapping_ids": []}))
    v8, v8c, v9, natural, v10, v11 = [root / name for name in (
        "v8", "v8c", "v9", "natural", "v10", "v11"
    )]
    prepare_control_text_dataset(v8, scenario_groups=400, reservation_registry=registry)
    prepare_contrastive_control_dataset(v8, v8c, reservation_registry=registry, pairs_per_kind=1)
    prepare_relation_grounded_dataset(v8c, v9)
    prepare_natural_control_dataset(v9, natural)
    prepare_canonical_repaired_dataset(natural, v10)
    prepare_transfer_control_dataset(v10, v11, timing_pairs=1, conditionals=2, mouse_per_mode=1)
    return root, v11


def test_preparation_byte_preservation_identity_counts_and_v12_reader_guards(mini_v11):
    root, source = mini_v11
    old_manifest, old_rows = read_control_dataset(source)
    source_bytes = (source / "records.jsonl").read_bytes()
    source_manifest_bytes = (source / "manifest.json").read_bytes()
    output = root / "v12"
    manifest = prepare_structured_control_dataset(source, output)
    _, rows = read_control_dataset(output)
    output_bytes = (output / "records.jsonl").read_bytes()
    assert (source / "records.jsonl").read_bytes() == source_bytes
    assert (source / "manifest.json").read_bytes() == source_manifest_bytes
    changed = 0
    for old, new, old_line, new_line in zip(old_rows, rows, source_bytes.splitlines(keepends=True),
                                           output_bytes.splitlines(keepends=True), strict=True):
        replace = (old["split"] == "train"
                   and old["template_family"].startswith("v8-binding-composition-"))
        if replace:
            changed += 1
            immutable = old.keys() - {"text", "template_family", "grammar_families"}
            assert all(old[key] == new[key] for key in immutable)
            assert new["template_family"] in FAMILIES
        else:
            assert old_line == new_line and old == new
        if new["split"] == "train":
            assert not any(f.startswith("v9-relation-witness-") for f in new["grammar_families"])
    metadata = manifest["structured_augmentation"]
    assert metadata["replacement_records"] == changed
    assert metadata["added_records"] == metadata["new_independent_semantic_groups"] == 0
    assert metadata["base_records_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert metadata["base_manifest_sha256"] == hashlib.sha256(source_manifest_bytes).hexdigest()
    assert manifest["records_sha256"] == hashlib.sha256(output_bytes).hexdigest()
    assert manifest["records"] == old_manifest["records"]
    assert manifest["split_groups"] == old_manifest["split_groups"]
    assert manifest["version"] == "carryrl.control_text_data.v12"
    assert manifest["relation_witness_group_counts"]["train"] == 0
    assert manifest["model_input_fields"] == ["text"]
    assert manifest["readiness"] is manifest["model_loaded"] is manifest["ppo_eligible"] is False
    for mutation in ("grammar", "identity"):
        broken = [dict(row) for row in rows]
        if mutation == "grammar":
            victim = next(row for row in broken if row["split"] == "dev")
            victim["grammar_families"] = [broken[0]["template_family"]]
        else:
            broken[0]["semantic_mapping_id"] = "f" * 64
        damaged = "".join(json.dumps(row) + "\n" for row in broken).encode()
        altered_manifest = {**manifest, "records_sha256": hashlib.sha256(damaged).hexdigest()}
        (output / "records.jsonl").write_bytes(damaged)
        (output / "manifest.json").write_text(json.dumps(altered_manifest))
        with pytest.raises(ValueError, match="leakage|identity"):
            read_control_dataset(output)
    (output / "records.jsonl").write_bytes(output_bytes)
    (output / "manifest.json").write_text(json.dumps(manifest))
