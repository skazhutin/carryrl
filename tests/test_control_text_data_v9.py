"""V9 explicitly states effect relations while preserving every structural label."""

import hashlib
import json
from collections import defaultdict

import pytest

from carryrl.rewards.control_text_data_contrastive import prepare_contrastive_control_dataset
from carryrl.rewards.control_text_data_v4 import ControlSpec, OperationSpec
from carryrl.rewards.control_text_data_v8 import (
    INTERNAL_FAMILIES,
    PARTITION_REGISTRY_VERSION,
    annotate_spec,
    prepare_control_text_dataset,
    render_spec,
)
from carryrl.rewards.control_text_data_v9 import (
    _known_groups,
    _witness,
    prepare_relation_grounded_dataset,
    relation_grounded_records,
)
from carryrl.rewards.control_text_grounding import _mention_spans
from carryrl.rewards.control_text_training import read_control_dataset


def _row(spec, *, split="train", language="en", family_offset=0):
    from dataclasses import asdict

    text, grammars = render_spec(spec, language=language, variant=family_offset, split=split,
                                seed=1, pair_seed="0" * 64)
    return {"specification": asdict(spec), "labels": annotate_spec(spec), "text": text,
            "split": split, "language": language, "template_family": grammars[0],
            "grammar_families": grammars}


def test_explicit_same_and_different_relations_share_grammar():
    alias = [("key:w", "mouse:left")]
    independent = [("key:w",), ("mouse:left",)]
    for language in ("ru", "en"):
        for family in range(24):
            same = _witness(alias, language, family)
            different = _witness(independent, language, family)
            if language == "ru":
                same_names = "[клавиша W / левая кнопка мыши]"
                different_names = "[клавиша W]; [левая кнопка мыши]"
                assert "эффект" in different or "действ" in different
                assert "разн" in different or "различ" in different
            else:
                same_names = "[W key / left mouse button]"
                different_names = "[W key]; [left mouse button]"
                assert "different" in different or "differ" in different
            assert same_names in same and different_names in different
            assert same.replace(same_names, "GROUPS") == different.replace(
                different_names, "GROUPS"
            )
    assert len({_witness(alias, "en", family) for family in range(24)}) == 24


def test_unknown_unsupported_and_unknown_timing_keys_are_excluded():
    spec = ControlSpec((OperationSpec("one", ("key:w", "key:up"), "held"),
                        OperationSpec("two", ("mouse:left",), "press"),
                        OperationSpec("three", ("key:q",), "unknown"),
                        OperationSpec("four", ("dx:-",), "unknown")),
                       ("key:r",), ("key:e",), "unknown", "partial_unknown")
    row = _row(spec)
    assert _known_groups(row) == [("key:w", "key:up"), ("mouse:left",), ("dx:-",)]
    repaired, _ = relation_grounded_records([row])
    witness = repaired[0]["text"][len(row["text"]):]
    mentions = _mention_spans(witness)
    assert mentions["key:w"] and mentions["key:up"] and mentions["mouse:left"] and mentions["dx:-"]
    assert not mentions["key:q"] and not mentions["key:r"] and not mentions["key:e"]
    # Relative timing may remain unknown while the physical input is supported.
    assert next(label for label in row["labels"] if label["candidate"] == "dx:-")["mode"] == (
        "unknown"
    )
    assert repaired[0]["labels"] == row["labels"]


def test_all_supported_axes_have_different_effects_and_aliases_stay_inside_groups():
    spec = ControlSpec(tuple(OperationSpec(str(index), (candidate,), "unknown")
                             for index, candidate in enumerate(("dx:-", "dx:+", "dy:-", "dy:+"))),
                       (), (), "unsupported", "mouse_axes")
    row = _row(spec, language="ru")
    repaired, _ = relation_grounded_records([row])
    witness = repaired[0]["text"][len(row["text"]):]
    assert "[движение мыши влево]; [движение мыши вправо]; " in witness
    assert "[движение мыши вверх]; [движение мыши вниз]" in witness
    assert "разные действия" in witness and "не заменяют" in witness
    labels = repaired[0]["labels"]
    assert len({label["alias_group"] for label in labels if label["status"] == "supported"}) == 4


def test_all_unknown_row_is_preserved_without_a_relation_claim():
    spec = ControlSpec((), ("key:w", "mouse:left"), (), "unknown", "unknown")
    row = _row(spec)
    repaired, counts = relation_grounded_records([row])
    assert repaired == [row]
    assert counts == {"train|no_known_group_records": 1}


def test_relation_witness_grammars_follow_original_split_ownership():
    spec = ControlSpec((OperationSpec("one", ("key:a",), "held"),
                        OperationSpec("two", ("key:d",), "press")),
                       (), (), "unsupported", "adjacent_modes")
    owners = defaultdict(set)
    for split, families in INTERNAL_FAMILIES.items():
        for offset, family in enumerate(families):
            row = _row(spec, split=split, family_offset=offset)
            result, _ = relation_grounded_records([row])
            assert result[0]["template_family"] == row["template_family"]
            assert result[0]["grammar_families"][-1] == f"v9-relation-witness-{family:02d}"
            owners[result[0]["grammar_families"][-1]].add(split)
    assert len(owners) == 24 and all(len(value) == 1 for value in owners.values())
    malformed = _row(spec)
    malformed["template_family"] = "v8-binding-composition-13"
    with pytest.raises(ValueError, match="split-owned"):
        relation_grounded_records([malformed])


def test_repair_writes_v9_with_unchanged_labels_splits_and_authenticated_base(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "version": PARTITION_REGISTRY_VERSION, "hard_old_sealed_ids": [],
        "developmental_ids": [], "previous_mapping_ids": [],
    }))
    original = tmp_path / "original"
    prepare_control_text_dataset(original, scenario_groups=200, reservation_registry=registry)
    base = tmp_path / "base_v8c"
    prepare_contrastive_control_dataset(
        original, base, reservation_registry=registry, pairs_per_kind=1
    )
    original_payload = (base / "records.jsonl").read_bytes()
    original_manifest = (base / "manifest.json").read_bytes()
    output = tmp_path / "v9"
    manifest = prepare_relation_grounded_dataset(base, output)
    assert (base / "records.jsonl").read_bytes() == original_payload
    assert (base / "manifest.json").read_bytes() == original_manifest
    before_manifest, before = read_control_dataset(base)
    after_manifest, after = read_control_dataset(output)
    assert manifest["version"] == after_manifest["version"] == "carryrl.control_text_data.v9"
    assert manifest["records"] == len(before) == len(after)
    assert manifest["split_groups"] == before_manifest["split_groups"]
    assert manifest["calibration_eligible_sign_groups"] == before_manifest[
        "calibration_eligible_sign_groups"
    ]
    for old, new in zip(before, after, strict=True):
        immutable_keys = old.keys() - {"text", "grammar_families"}
        assert {key: old[key] for key in immutable_keys} == {
            key: new[key] for key in immutable_keys
        }
        assert new["text"].startswith(old["text"])
        if _known_groups(old):
            assert new["text"] != old["text"]
            assert new["grammar_families"][:-1] == old["grammar_families"]
        else:
            assert new == old
    repair = manifest["relation_grounding_repair"]
    assert repair["evaluation_wording_protocol_changed"] is True
    assert repair["heldout_mappings_moved_to_train"] == 0
    assert repair["base_records_sha256"] == hashlib.sha256(original_payload).hexdigest()
    assert repair["base_manifest_sha256"] == hashlib.sha256(original_manifest).hexdigest()
    output_digest = hashlib.sha256((output / "records.jsonl").read_bytes()).hexdigest()
    assert manifest["records_sha256"] == output_digest
    assert all(manifest["relation_witness_group_counts"][split] > 0 for split in INTERNAL_FAMILIES)
    assert manifest["model_input_fields"] == ["text"]
    assert manifest["readiness"] is manifest["model_loaded"] is manifest["ppo_eligible"] is False
    paired = defaultdict(list)
    for row in after:
        if row.get("contrastive_pair_id"):
            paired[(row["contrastive_pair_id"], row["variant_index"])].append(row)
    assert paired
    assert all(
        len(views) == 2 and views[0]["grammar_families"][-1] == views[1]["grammar_families"][-1]
        for views in paired.values()
    )
    with pytest.raises(FileExistsError):
        prepare_relation_grounded_dataset(base, output)


def test_structurally_inconsistent_alias_supervision_is_rejected():
    spec = ControlSpec((OperationSpec("one", ("key:w",), "held"),
                        OperationSpec("two", ("mouse:left",), "held")),
                       (), (), "unsupported", "mixed")
    row = _row(spec)
    for label in row["labels"]:
        if label["candidate"] == "mouse:left":
            label["alias_group"] = 1
    with pytest.raises(ValueError, match="supervised effect relations"):
        relation_grounded_records([row])
