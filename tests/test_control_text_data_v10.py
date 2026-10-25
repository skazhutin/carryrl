"""Canonical spellings render directly; historical repairs touch only bad spans."""

import hashlib
import json
from dataclasses import asdict

import pytest

from carryrl.rewards.control_text_data_natural import prepare_natural_control_dataset
from carryrl.rewards.control_text_data_v4 import ControlSpec, OperationSpec
from carryrl.rewards.control_text_data_v8 import (
    _HORIZONTAL,
    _NAVIGATION,
    INTERNAL_FAMILIES,
    PARTITION_REGISTRY_VERSION,
    _binding,
    _canonical,
    _canonical_names,
    _name,
    annotate_spec,
    prepare_control_text_dataset,
)
from carryrl.rewards.control_text_data_v9 import prepare_relation_grounded_dataset
from carryrl.rewards.control_text_data_v10 import (
    _specification,
    prepare_canonical_repaired_dataset,
    repair_canonical_view,
)
from carryrl.rewards.control_text_grounding import _mention_spans
from carryrl.rewards.control_text_training import read_control_dataset


def _spec(kind, mode, *, extra_w=False):
    groups = _NAVIGATION if kind == "navigation" else _HORIZONTAL
    operations = tuple(OperationSpec(str(index), group, mode) for index, group in enumerate(groups))
    if extra_w:
        assert kind == "horizontal"
        operations += (OperationSpec("extra", ("key:w",), "press"),)
    return ControlSpec(operations, (), (), "unsupported", kind)


def _historical_clause(spec, language, family):
    """Reproduce the old defect locally, never as a production renderer."""
    mode = _canonical(spec, language, family)[0][0].mode
    names = _canonical_names(spec.kind, language)
    example = _binding(OperationSpec("dummy", ("key:w",), mode), language, family)
    old = example.replace(_name("key:w", language, accusative=True), names).replace(
        _name("key:w", language), names
    )
    if mode == "held":
        description = (
            "удержание повторяет движение" if language == "ru" else "holding repeats movement"
        )
    else:
        description = "нажимай один раз и отпускай между шагами" if language == "ru" else (
            "press once and release between steps"
        )
    return old + ("; направления имеют обычный смысл; " if language == "ru" else (
        "; directions have their usual meanings; "
    )) + description


def _row(spec, language, family, *, historical=True):
    split = next(split for split, families in INTERNAL_FAMILIES.items() if family in families)
    clause = _historical_clause(spec, language, family) if historical else _canonical(
        spec, language, family
    )[1]
    suffix = (" " + _binding(spec.operations[-1], language, family) + ".") if (
        spec.operations[-1].identifier == "extra"
    ) else ""
    return {"specification": asdict(spec), "labels": annotate_spec(spec), "language": language,
            "split": split, "scenario_kind": spec.kind,
            "text": "Goal context. " + clause + "." + suffix,
            "template_family": f"v8-binding-composition-{family:02d}",
            "grammar_families": [f"v8-binding-composition-{family:02d}"]}


@pytest.mark.parametrize("family", range(24))
@pytest.mark.parametrize("mode", ("held", "press"))
@pytest.mark.parametrize("language", ("ru", "en"))
@pytest.mark.parametrize("kind", ("horizontal", "navigation"))
def test_all_canonical_grammars_modes_and_languages_have_no_dummy_control(
    kind, language, mode, family
):
    spec = _spec(kind, mode)
    core, text = _canonical(spec, language, family)
    assert len(core) == (2 if kind == "horizontal" else 4)
    spans = _mention_spans(text)
    assert all(spans[candidate] for operation in core for candidate in operation.candidates)
    assert "Клавиша W" not in text and "клавиша W" not in text and "W key" not in text
    if kind == "horizontal":
        assert not spans["key:w"] and not spans["key:s"]
    else:
        assert spans["key:w"] and spans["key:s"]
    legacy = _historical_clause(spec, language, family)
    expected_defect = language == "ru" and family in {1, 6, 10, 13, 22}
    assert (legacy != text) == expected_defect
    row = _row(spec, language, family)
    repaired, changed = repair_canonical_view(row)
    assert changed == expected_defect
    assert repaired["text"] == _row(spec, language, family, historical=False)["text"]
    assert repaired["labels"] == row["labels"]
    assert repair_canonical_view(repaired) == (repaired, False)


def test_real_w_in_extra_operation_is_not_globally_replaced():
    spec = _spec("horizontal", "held", extra_w=True)
    row = _row(spec, "ru", 13)
    real_binding = _binding(spec.operations[-1], "ru", 13)
    repaired, changed = repair_canonical_view(row)
    assert changed
    assert repaired["text"].endswith(real_binding + ".")
    assert _mention_spans(repaired["text"])["key:w"]
    actual_w = next(label for label in repaired["labels"] if label["candidate"] == "key:w")
    assert actual_w["status"] == "supported"
    for language in ("ru", "en"):
        for family in range(24):
            # The underlying renderer still names a real standalone W in both
            # name-first and lower-case internal positions.
            text = _binding(OperationSpec("real", ("key:w",), "held"), language, family)
            assert _mention_spans(text)["key:w"]


def test_natural_and_noncompact_views_are_not_rewritten():
    row = _row(_spec("horizontal", "press"), "ru", 13)
    row["template_family"] = "v9-natural-composition-04"
    assert repair_canonical_view(row) == (row, False)
    spec = _spec("horizontal", "held", extra_w=True)
    operations = list(spec.operations)
    operations[1] = OperationSpec(operations[1].identifier, operations[1].candidates, "press")
    mixed = ControlSpec(tuple(operations), (), (), "unsupported", "horizontal")
    row = _row(spec, "ru", 13)
    row["specification"] = asdict(mixed)
    row["labels"] = annotate_spec(mixed)
    row["text"] = _binding(mixed.operations[-1], "ru", 13)
    assert repair_canonical_view(row) == (row, False)


def test_v10_preserves_source_supervision_counts_natural_views_and_unaffected_bytes(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"version": PARTITION_REGISTRY_VERSION,
                                    "hard_old_sealed_ids": [], "developmental_ids": [],
                                    "previous_mapping_ids": []}))
    v8, v9, natural, output = [tmp_path / name for name in ("v8", "v9", "natural", "v10")]
    prepare_control_text_dataset(v8, scenario_groups=200, reservation_registry=registry)
    prepare_relation_grounded_dataset(v8, v9)
    prepare_natural_control_dataset(v9, natural)
    # Controlled synthetic fixture recreates historical bad clauses. The actual
    # immutable project preparations are never read or edited by this test.
    manifest, rows = read_control_dataset(natural)
    injected = 0
    for row in rows:
        if (row["language"] != "ru" or row["scenario_kind"] not in {"navigation", "horizontal"}
                or not row["template_family"].startswith("v8-binding-composition-")):
            continue
        spec = _specification(row)
        _, corrected = _canonical(spec, "ru", int(row["template_family"][-2:]))
        if not corrected:
            continue
        historical = _historical_clause(spec, "ru", int(row["template_family"][-2:]))
        if corrected != historical:
            row["text"] = row["text"].replace(corrected + ".", historical + ".", 1)
            injected += 1
    assert injected > 0
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()
    (natural / "records.jsonl").write_bytes(payload)
    manifest["records_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["records_bytes"] = len(payload)
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    (natural / "manifest.json").write_bytes(manifest_bytes)
    repaired_manifest = prepare_canonical_repaired_dataset(natural, output)
    _, after = read_control_dataset(output)
    assert (natural / "records.jsonl").read_bytes() == payload
    assert (natural / "manifest.json").read_bytes() == manifest_bytes
    output_payload = (output / "records.jsonl").read_bytes()
    changed = 0
    for old, new, old_line, new_line in zip(rows, after, payload.splitlines(keepends=True),
                                           output_payload.splitlines(keepends=True), strict=True):
        immutable = old.keys() - {"text"}
        assert all(old[key] == new[key] for key in immutable)
        if old["text"] == new["text"]:
            assert old_line == new_line
        else:
            changed += 1
            assert not old["template_family"].startswith("v9-natural-")
    assert changed == injected == repaired_manifest["canonical_render_repair"]["changed_records"]
    assert repaired_manifest["version"] == "carryrl.control_text_data.v10"
    assert repaired_manifest["records"] == manifest["records"]
    assert repaired_manifest["split_groups"] == manifest["split_groups"]
    assert repaired_manifest["calibration_eligible_sign_groups"] == manifest[
        "calibration_eligible_sign_groups"
    ]
    provenance = repaired_manifest["canonical_render_repair"]
    assert provenance["evaluation_wording_protocol_changed"] is True
    assert provenance["base_records_sha256"] == hashlib.sha256(payload).hexdigest()
    assert provenance["base_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert repaired_manifest["records_sha256"] == hashlib.sha256(output_payload).hexdigest()
    assert repaired_manifest["model_input_fields"] == ["text"]
    assert repaired_manifest["readiness"] is repaired_manifest["model_loaded"] is False
    with pytest.raises(FileExistsError):
        prepare_canonical_repaired_dataset(natural, output)
