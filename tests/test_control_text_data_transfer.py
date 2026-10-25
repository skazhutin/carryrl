"""Targeted V11 transfer views preserve semantics, bytes, and E5 grounding."""

import hashlib
import json
from dataclasses import asdict

import pytest

from carryrl.rewards.control_text_data import CONTROL_CANDIDATES, semantic_mapping_identity
from carryrl.rewards.control_text_data_contrastive import prepare_contrastive_control_dataset
from carryrl.rewards.control_text_data_natural import prepare_natural_control_dataset
from carryrl.rewards.control_text_data_transfer import (
    _BRIDGE,
    plan_transfer_views,
    prepare_transfer_control_dataset,
    render_transfer_view,
)
from carryrl.rewards.control_text_data_v4 import ControlSpec, OperationSpec
from carryrl.rewards.control_text_data_v8 import PARTITION_REGISTRY_VERSION, annotate_spec
from carryrl.rewards.control_text_data_v8 import prepare_control_text_dataset as prepare_v8
from carryrl.rewards.control_text_data_v9 import prepare_relation_grounded_dataset
from carryrl.rewards.control_text_data_v10 import prepare_canonical_repaired_dataset
from carryrl.rewards.control_text_grounding import _mention_spans, build_grounding
from carryrl.rewards.control_text_training import read_control_dataset


def _row(spec, language="en"):
    labels = annotate_spec(spec)
    identity = semantic_mapping_identity(labels, spec.declared_candidates)
    return {"specification": asdict(spec), "labels": labels, "text": "PRIVATE SOURCE TEXT",
            "split": "train", "language": language, "variant_index": 5 if language == "en" else 2,
            "template_family": "v9-natural-composition-01",
            "grammar_families": ["v9-natural-composition-01"], "group_id": "synthetic",
            "semantic_mapping_id": identity, "mapping_id": identity,
            "declared_candidates": list(spec.declared_candidates), "group_variants": 6}


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    from carryrl.agent.language import TEXT_ID, TEXT_REVISION

    try:
        return AutoTokenizer.from_pretrained(TEXT_ID, revision=TEXT_REVISION, local_files_only=True)
    except OSError:
        pytest.skip("original E5 tokenizer is not cached locally")


def test_deferred_press_and_held_have_actual_e5_distance_and_no_nearby_mode(tokenizer):
    for language in ("ru", "en"):
        for candidate in ("key:w", "mouse:left"):
            for mode, other in (("press", "held"), ("held", "press")):
                spec = ControlSpec((OperationSpec("one", (candidate,), mode),
                                    OperationSpec("two", ("key:q",), other)),
                                   (), (), "unsupported", "adjacent_modes")
                source = _row(spec, language)
                kind = (f"deferred_mouse_{mode}" if candidate.startswith("mouse:")
                        else "deferred_timing")
                entry = {"kind": kind, "target": candidate, "language": language}
                if candidate.startswith("mouse:"):
                    entry["intervening"] = "key:q"
                row, info = render_transfer_view(source, entry)
                text, prefix = row["text"], "query: "
                encoded = tokenizer(prefix + text, return_offsets_mapping=True, truncation=False)
                named_end = text.index(info["named_input_text"]) + len(info["named_input_text"])
                end = len(prefix) + named_end
                start = len(prefix) + text.index(info["timing_rule_text"])
                offsets = encoded["offset_mapping"]
                assert sum(a >= end and b <= start and b > a for a, b in offsets) > 24
                assert len(encoded["input_ids"]) <= 512
                between = text[named_end:]
                nearby = between[:between.index(_BRIDGE[language])]
                if candidate.startswith("mouse:"):
                    cue_start = len(prefix) + text.index(info["competing_cue_text"])
                    distance = sum(a >= end and b <= cue_start and b > a for a, b in offsets)
                    assert distance <= 8
                    assert text.index(info["competing_cue_text"]) < text.index(_BRIDGE[language])
                    nearby = nearby[:nearby.index(info["competing_cue_text"])]
                bridge = nearby
                assert not any(word in bridge.lower()
                               for word in ("tap", "hold", "press", "наж", "заж"))
                if candidate.startswith("mouse:"):
                    other = "Second:" if language == "en" else "Вторая:"
                else:
                    other = "Separate function 2" if language == "en" else "Отдельная функция 2"
                assert text.index(other) < text.index(info["timing_rule_text"])
                assert row["labels"] == source["labels"]
                assert row["semantic_mapping_id"] == source["semantic_mapping_id"]
                assert _mention_spans(text)[candidate]
                assert "PRIVATE SOURCE TEXT" not in text
                replacement_source = {**source, "text": "A DIFFERENT PRIVATE OR PUBLIC SOURCE TEXT"}
                assert render_transfer_view(replacement_source, entry)[0]["text"] == text
                neutral = build_grounding(_BRIDGE["en"], tokenizer)
                assert not neutral[CONTROL_CANDIDATES.index("key:enter")].any()


def test_conditional_known_mechanics_remain_effectively_unknown():
    spec = ControlSpec((OperationSpec("one", ("key:w",), "held"),), ("key:q",), (),
                       "unknown", "conditional")
    for language in ("ru", "en"):
        source = _row(spec, language)
        rendered = []
        for mode in ("held", "press"):
            entry = {"kind": "conditional_mechanic", "target": "key:q",
                     "language": language, "hypothetical_mode": mode}
            row, _ = render_transfer_view(source, entry)
            rendered.append(row["text"])
            label = next(label for label in row["labels"] if label["candidate"] == "key:q")
            assert label["status"] == label["mode"] == "unknown" and label["alias_group"] == 0
            assert row["specification"] == source["specification"]
            assert row["labels"] == source["labels"]
            necessary = "неизвестного условия" if language == "ru" else "necessary condition"
            assert necessary in row["text"]
            assert ("сейчас неизвестны" if language == "ru" else "currently unknown") in row["text"]
        assert rendered[0] != rendered[1]


def test_aggregate_all_four_masks_identical_without_individual_axis_witness(tokenizer):
    axes = ("dx:-", "dx:+", "dy:-", "dy:+")
    spec = ControlSpec(tuple(OperationSpec(str(index), (candidate,), "unknown")
                             for index, candidate in enumerate(axes)) + (
                                 OperationSpec("extra", ("mouse:left",), "press"),
                             ), (), (), "unsupported", "mouse_axes")
    for language in ("ru", "en"):
        source = _row(spec, language)
        row, _ = render_transfer_view(source, {"kind": "aggregate_axes", "language": language})
        grounding = build_grounding(row["text"], tokenizer)
        masks = [grounding[CONTROL_CANDIDATES.index(candidate)] for candidate in axes]
        assert all(mask.any() and mask.equal(masks[0]) for mask in masks)
        assert "v9-relation-witness" not in row["grammar_families"]
        assert "[" not in row["text"] and "]" not in row["text"]
        assert len({label["alias_group"] for label in row["labels"]
                    if label["candidate"] in axes}) == 4
        assert all(label["mode"] == "unknown" for label in row["labels"]
                   if label["candidate"] in axes)
        assert ("четыре разных" if language == "ru" else "four different") in row["text"]
        assert ("не взаимозаменяемы" if language == "ru" else "not interchangeable") in row["text"]
        assert grounding[CONTROL_CANDIDATES.index("mouse:left")].any()
        assert row["labels"] == source["labels"]
        assert not any(word in row["text"].lower() for word in ("camera", "shoot", "door", "menu"))


@pytest.fixture(scope="module")
def mini_base(tmp_path_factory):
    root = tmp_path_factory.mktemp("transfer")
    registry = root / "registry.json"
    registry.write_text(json.dumps({"version": PARTITION_REGISTRY_VERSION,
                                    "hard_old_sealed_ids": [], "developmental_ids": [],
                                    "previous_mapping_ids": []}))
    v8, v8c, v9, natural, v10 = [root / name for name in ("v8", "v8c", "v9", "natural", "v10")]
    prepare_v8(v8, scenario_groups=400, reservation_registry=registry)
    prepare_contrastive_control_dataset(v8, v8c, reservation_registry=registry, pairs_per_kind=1)
    prepare_relation_grounded_dataset(v8c, v9)
    prepare_natural_control_dataset(v9, natural)
    prepare_canonical_repaired_dataset(natural, v10)
    return root, v10


def test_preparation_plan_bytes_reader_guards_and_no_new_semantic_groups(mini_base, tokenizer):
    root, base = mini_base
    before_manifest, before = read_control_dataset(base)
    options = {"timing_pairs": 1, "conditionals": 2, "mouse_per_mode": 1}
    plan = plan_transfer_views(before, **options)
    assert len({entry["group_id"] for entry in plan}) == len(plan)
    assert sum(entry["kind"] == "deferred_timing" for entry in plan) == 2
    assert sum(entry["kind"] == "conditional_mechanic" for entry in plan) == 2
    assert {entry["kind"] for entry in plan} >= {"deferred_mouse_press", "deferred_mouse_held"}
    assert plan == plan_transfer_views(before, **options)
    old_payload = (base / "records.jsonl").read_bytes()
    old_manifest = (base / "manifest.json").read_bytes()
    output = root / "v11"
    manifest = prepare_transfer_control_dataset(base, output, **options)
    _, after = read_control_dataset(output)
    payload = (output / "records.jsonl").read_bytes()
    assert (base / "records.jsonl").read_bytes() == old_payload
    assert (base / "manifest.json").read_bytes() == old_manifest
    selected = {(entry["group_id"], entry["variant_index"]) for entry in plan}
    changed = 0
    for old, new, old_line, new_line in zip(before, after, old_payload.splitlines(keepends=True),
                                           payload.splitlines(keepends=True), strict=True):
        immutable = old.keys() - {"text", "template_family", "grammar_families"}
        assert all(old[key] == new[key] for key in immutable)
        if (old["group_id"], old["variant_index"]) in selected:
            changed += 1
            assert old["split"] == "train"
            assert len(tokenizer("query: " + new["text"], truncation=False)["input_ids"]) <= 512
            assert any(family.startswith("v11-transfer-") for family in new["grammar_families"])
        else:
            assert old_line == new_line
    assert changed == len(plan)
    metadata = manifest["transfer_augmentation"]
    assert metadata["new_independent_semantic_groups"] == metadata["added_records"] == 0
    assert metadata["replacement_existing_semantic_groups"] == changed
    assert metadata["base_records_sha256"] == hashlib.sha256(old_payload).hexdigest()
    assert manifest["records_sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["split_groups"] == before_manifest["split_groups"]
    assert manifest["calibration_eligible_sign_groups"] == before_manifest[
        "calibration_eligible_sign_groups"
    ]
    assert manifest["version"] == "carryrl.control_text_data.v11"
    assert manifest["readiness"] is manifest["model_loaded"] is manifest["ppo_eligible"] is False
    # Authenticate the new version through the existing reader, then prove that
    # recomputing the file hash cannot bypass grammar and semantic identity guards.
    source_lines = payload.splitlines()
    for mutation in ("grammar", "identity"):
        damaged = [json.loads(line) for line in source_lines]
        if mutation == "grammar":
            victim = next(row for row in damaged if row["split"] == "dev")
            victim["grammar_families"] = [damaged[0]["template_family"]]
        else:
            damaged[0]["semantic_mapping_id"] = "f" * 64
        broken_payload = "".join(json.dumps(row) + "\n" for row in damaged).encode()
        broken_manifest = {**manifest, "records_sha256": hashlib.sha256(broken_payload).hexdigest()}
        (output / "records.jsonl").write_bytes(broken_payload)
        (output / "manifest.json").write_text(json.dumps(broken_manifest))
        with pytest.raises(ValueError, match="leakage|identity"):
            read_control_dataset(output)
    (output / "records.jsonl").write_bytes(payload)
    (output / "manifest.json").write_text(json.dumps(manifest))
