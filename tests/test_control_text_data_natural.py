"""Natural TRAIN replacements keep frozen supervision and evaluation line bytes."""

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict

import pytest

from carryrl.rewards.control_text_data_contrastive import (
    _paired_specs,
    prepare_contrastive_control_dataset,
)
from carryrl.rewards.control_text_data_natural import (
    NATURAL_FAMILIES,
    _binding,
    _natural_view,
    _targets,
    prepare_natural_control_dataset,
)
from carryrl.rewards.control_text_data_v4 import ControlSpec, OperationSpec
from carryrl.rewards.control_text_data_v8 import (
    PARTITION_REGISTRY_VERSION,
    annotate_spec,
    prepare_control_text_dataset,
)
from carryrl.rewards.control_text_data_v9 import prepare_relation_grounded_dataset
from carryrl.rewards.control_text_grounding import _mention_spans
from carryrl.rewards.control_text_training import read_control_dataset


def _row(spec, *, language="en", variant=5, group_id="example", pair_id=None):
    row = {"group_id": group_id, "split": "train", "specification": asdict(spec),
           "labels": annotate_spec(spec), "language": language, "variant_index": variant,
           "group_variants": 6, "elementary": False, "text": "Original text.",
           "template_family": "v8-binding-composition-00",
           "grammar_families": ["v8-binding-composition-00", "v9-relation-witness-00"]}
    if pair_id:
        row["contrastive_pair_id"] = pair_id
    return row


def test_core_lexical_diversity_and_control_types():
    for language in ("ru", "en"):
        held = {_binding(("key:w",), "held", language, style) for style in range(12)}
        press = {_binding(("key:w",), "press", language, style) for style in range(12)}
        assert len(held) == 4 and len(press) == 3
        assert all("отпус" in text.lower() or "отпусти" in text.lower() for text in held) if (
            language == "ru"
        ) else all("release" in text or "letting go" in text for text in held)
        for style in range(12):
            key = _binding(("key:w",), "press", language, style)
            mouse = _binding(("mouse:left",), "press", language, style)
            mixed = _binding(("key:w", "mouse:left"), "press", language, style)
            relative = _binding(("dx:-",), "unknown", language, style)
            assert "click" not in key and "click" not in mixed and "Tap" not in mixed
            assert "щёлк" not in key and "щелч" not in key
            assert "щёлк" not in mixed and "щелч" not in mixed
            assert not any(word in relative.lower() for word in ("tap", "click", "hold", "held"))
            assert not any(word in relative.lower() for word in ("нажми", "щелч", "удерж", "заж"))
            if language == "en" and style % 3 < 2:
                assert "click" in mouse.lower()


def test_same_pair_styles_and_explicit_effect_relations_without_brackets():
    for kind in ("cross_modal_alias", "explicit_status", "adjacent_timing"):
        specs, targets = _paired_specs(random.Random(19), kind)
        for language, variant in (("ru", 2), ("en", 5)):
            before = [_row(spec, language=language, variant=variant, group_id=str(index),
                           pair_id="shared-pair") for index, spec in enumerate(specs)]
            after = [_natural_view(row, seed=20)[0] for row in before]
            assert after[0]["template_family"] == after[1]["template_family"]
            assert all(row["template_family"] in NATURAL_FAMILIES for row in after)
            for old, new in zip(before, after, strict=True):
                assert new["labels"] == old["labels"]
                assert "[" not in new["text"] and "]" not in new["text"]
                assert "разн" in new["text"] or "различ" in new["text"] if language == "ru" else (
                    "differ" in new["text"]
                )
                mentions = _mention_spans(new["text"])
                assert all(mentions[candidate] for candidate in targets)
            if kind == "cross_modal_alias":
                equivalent = (
                    "равнозначны" if language == "ru" else "interchangeable for this function"
                )
                assert equivalent in after[0]["text"]
                independent = "равнозначны" if language == "ru" else "choices are interchangeable"
                assert independent not in after[1]["text"]
            if kind == "explicit_status":
                assert ("запрещён" if language == "ru" else "unsupported") in after[1]["text"]


def test_deferred_release_uses_ordinal_after_an_intervening_binding():
    for mode in ("held", "press"):
        other = "press" if mode == "held" else "held"
        spec = ControlSpec((OperationSpec("one", ("key:w",), mode),
                            OperationSpec("two", ("key:q",), other)),
                           (), (), "unsupported", "adjacent_modes")
        for language in ("ru", "en"):
            row = _row(spec, language=language)
            result, deferred, _ = next(_natural_view(row, seed=seed) for seed in range(100)
                                       if _natural_view(row, seed=seed)[1])
            assert deferred
            text = result["text"]
            second = text.index("Вторая функция" if language == "ru" else "The second function")
            suffix = text[second:]
            referent = "первая функция" if mode == "held" else "первой функции"
            assert (referent if language == "ru" else "first function") in suffix.lower()
            assert ("отпус" if language == "ru" else "release") in suffix.lower()
            # Ordinal release referent is not a repeated literal W key name.
            assert "клавиша W" not in suffix and "клавишу W" not in suffix and "W key" not in suffix
            if mode == "held":
                phrase = "до отпускания" if language == "ru" else "until its input is released"
                assert phrase in suffix
            else:
                phrase = "После каждого выполнения" if language == "ru" else "After each execution"
                assert phrase in suffix


def test_unknown_inputs_abstain_and_relative_mode_is_preserved():
    spec = ControlSpec((OperationSpec("one", ("key:w",), "unknown"),
                        OperationSpec("two", ("dx:-",), "unknown")),
                       ("key:q",), ("mouse:left",), "unknown", "conditional")
    for language in ("ru", "en"):
        old = _row(spec, language=language)
        new, _, _ = _natural_view(old, seed=1)
        assert new["labels"] == old["labels"]
        labels = {label["candidate"]: label for label in new["labels"]}
        assert labels["key:w"]["status"] == labels["key:q"]["status"] == "unknown"
        assert labels["mouse:left"]["status"] == "unsupported"
        assert labels["dx:-"]["status"] == "supported" and labels["dx:-"]["mode"] == "unknown"
        unknown = (
            "исполнимость не установлена" if language == "ru" else "executable support is unknown"
        )
        assert unknown in new["text"]
        assert ("запрещён" if language == "ru" else "unsupported") in new["text"]
        mentions = _mention_spans(new["text"])
        assert all(mentions[candidate] for candidate in ("key:w", "key:q", "mouse:left", "dx:-"))


def test_target_indices_are_exact_and_invalid_groups_fail():
    spec = ControlSpec((OperationSpec("one", ("key:w",), "held"),), (), (), "unsupported", "single")
    rows = []
    for size in (6, 8):
        for variant in range(size):
            row = _row(spec, group_id=str(size), variant=variant,
                       language="ru" if variant < size // 2 else "en")
            row["group_variants"] = size
            rows.append(row)
    assert _targets(rows) == {("6", 2), ("6", 5), ("8", 3), ("8", 7)}
    with pytest.raises(ValueError, match="complete"):
        _targets(rows[:-1])
    broken = [dict(row) for row in rows]
    broken[2]["language"] = "en"
    with pytest.raises(ValueError, match="Russian then English"):
        _targets(broken)


def test_preparation_preserves_unselected_and_heldout_bytes_with_current_coverage(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"version": PARTITION_REGISTRY_VERSION,
                                    "hard_old_sealed_ids": [], "developmental_ids": [],
                                    "previous_mapping_ids": []}))
    v8, v8c, v9, natural = [tmp_path / name for name in ("v8", "v8c", "v9", "natural")]
    prepare_control_text_dataset(v8, scenario_groups=200, reservation_registry=registry)
    prepare_contrastive_control_dataset(v8, v8c, reservation_registry=registry, pairs_per_kind=1)
    prepare_relation_grounded_dataset(v8c, v9)
    old_payload = (v9 / "records.jsonl").read_bytes()
    old_manifest_bytes = (v9 / "manifest.json").read_bytes()
    before_manifest, before = read_control_dataset(v9)
    manifest = prepare_natural_control_dataset(v9, natural)
    loaded_manifest, after = read_control_dataset(natural)
    new_payload = (natural / "records.jsonl").read_bytes()
    assert (v9 / "records.jsonl").read_bytes() == old_payload
    assert (v9 / "manifest.json").read_bytes() == old_manifest_bytes
    assert manifest["version"] == loaded_manifest["version"] == "carryrl.control_text_data.v9"
    selected = _targets(before)
    changed = Counter()
    old_lines = old_payload.splitlines(keepends=True)
    new_lines = new_payload.splitlines(keepends=True)
    for old, new, old_line, new_line in zip(before, after, old_lines, new_lines, strict=True):
        target = old["split"] == "train" and (old["group_id"], old["variant_index"]) in selected
        if target:
            changed[(old["group_id"], old["language"])] += 1
            assert old["text"] != new["text"]
            assert new["template_family"] in NATURAL_FAMILIES
            immutable = old.keys() - {"text", "template_family", "grammar_families"}
            assert all(old[key] == new[key] for key in immutable)
        else:
            assert old_line == new_line and old == new
    assert all(count == 1 for count in changed.values())
    assert len(changed) == before_manifest["split_groups"]["train"] * 2
    augmentation = manifest["natural_augmentation"]
    assert augmentation["changed_records"] == len(changed)
    assert augmentation["base_records_sha256"] == hashlib.sha256(old_payload).hexdigest()
    assert augmentation["base_manifest_sha256"] == hashlib.sha256(old_manifest_bytes).hexdigest()
    for split in ("dev", "calibration", "sealed_test"):
        exact_lines = b"".join(line for row, line in zip(before, old_lines, strict=True)
                               if row["split"] == split)
        assert augmentation["heldout_records_sha256_by_split"][split] == hashlib.sha256(
            exact_lines
        ).hexdigest()
    coverage = manifest["relation_grounding_repair"]["coverage_records"]
    for split in ("train", "dev", "calibration", "sealed_test"):
        actual = sum(
            row["split"] == split and any(f.startswith("v9-relation-witness-")
                                         for f in row["grammar_families"]) for row in after
        )
        assert coverage.get(f"{split}|witness_records", 0) == actual
    assert coverage["train|witness_records"] < before_manifest["relation_grounding_repair"][
        "coverage_records"
    ]["train|witness_records"]
    assert manifest["split_groups"] == before_manifest["split_groups"]
    assert manifest["calibration_eligible_sign_groups"] == before_manifest[
        "calibration_eligible_sign_groups"
    ]
    assert manifest["model_input_fields"] == ["text"]
    assert manifest["readiness"] is manifest["model_loaded"] is manifest["ppo_eligible"] is False
    with pytest.raises(FileExistsError):
        prepare_natural_control_dataset(v9, natural)
    with pytest.raises(ValueError, match="unaugmented"):
        prepare_natural_control_dataset(natural, tmp_path / "again")
