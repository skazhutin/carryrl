"""V13 exact-line mixture of immutable V11 wording and its V12 derivative.

For each TRAIN group and language, restore the lowest-index ordinary V11 view.
Everything else remains an exact V12 line. There is no rendering, model call,
public-input ingestion, or new semantic group in this preparation.
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from carryrl.rewards.control_text_data_v8 import wording_manifest
from carryrl.rewards.control_text_data_v9 import _known_groups

CONTROL_TEXT_DATA_VERSION = "carryrl.control_text_data.v13"
MIXED_VERSION = "carryrl.control_text_mixed.v1"
_TEXT_FIELDS = {"text", "template_family", "grammar_families"}


def _key(row):
    return row["group_id"], row["variant_index"]


def _ordinary(row):
    return row["split"] == "train" and row["template_family"].startswith("v8-binding-composition-")


def select_restored_view_ids(records):
    """Select the minimum eligible index independently for every group/language."""
    choices, languages = {}, defaultdict(set)
    for row in records:
        if row["split"] != "train":
            continue
        pair = row["group_id"], row["language"]
        languages[row["group_id"]].add(row["language"])
        if _ordinary(row) and (pair not in choices or row["variant_index"] < choices[pair][1]):
            choices[pair] = _key(row)
    if any(values != {"ru", "en"} for values in languages.values()):
        raise ValueError("V11 TRAIN groups must contain Russian and English views")
    expected = {(group_id, language) for group_id in languages for language in ("ru", "en")}
    if choices.keys() != expected:
        raise ValueError("every V11 TRAIN group/language needs an ordinary view to restore")
    return set(choices.values())


def _load(directory, version):
    from carryrl.rewards.control_text_training import read_control_dataset

    directory = Path(directory)
    manifest, records = read_control_dataset(directory)
    if manifest.get("version") != version:
        raise ValueError(f"mixed preparation requires source version {version}")
    payload = (directory / "records.jsonl").read_bytes()
    manifest_payload = (directory / "manifest.json").read_bytes()
    if (hashlib.sha256(payload).hexdigest() != manifest["records_sha256"]
            or json.loads(manifest_payload) != manifest):
        raise ValueError("source changed during mixed preparation")
    lines = payload.splitlines(keepends=True)
    if len(lines) != len(records):
        raise ValueError("source JSONL must contain one record on each line")
    indexed, groups = {}, defaultdict(list)
    for row, line in zip(records, lines, strict=True):
        if _key(row) in indexed:
            raise ValueError("source contains duplicate view identifiers")
        indexed[_key(row)] = (row, line)
        groups[row["group_id"]].append(row)
    for rows in groups.values():
        size = rows[0]["group_variants"]
        if (size not in {6, 8} or len(rows) != size
                or any(row["group_variants"] != size for row in rows)
                or {row["variant_index"] for row in rows} != set(range(size))):
            raise ValueError("source contains an incomplete or inconsistent semantic group")
    return directory, manifest, records, payload, manifest_payload, indexed


def _actual_coverage(records):
    templates, grammars, coverage = {}, {}, Counter()
    witnesses, aggregates = defaultdict(set), defaultdict(set)
    for row in records:
        split = row["split"]
        for table, values in ((templates, [row["template_family"]]),
                              (grammars, row["grammar_families"])):
            for family in values:
                if table.setdefault(family, split) != split:
                    raise ValueError("mixed wording has grammar split leakage")
        groups = _known_groups(row)
        if any(family.startswith("v9-relation-witness-") for family in row["grammar_families"]):
            coverage[f"{split}|witness_records"] += 1
            coverage[f"{split}|multiple_known_group_records"] += len(groups) > 1
            coverage[f"{split}|within_group_alias_records"] += any(
                len(group) > 1 for group in groups
            )
            witnesses[split].add(row["group_id"])
        elif groups:
            coverage[f"{split}|known_records_without_bracket_witness"] += 1
        else:
            coverage[f"{split}|no_known_group_records"] += 1
        if any(family.startswith("v8-aggregate-axes-") for family in row["grammar_families"]):
            aggregates[split].add(row["group_id"])
    return templates, grammars, dict(coverage), witnesses, aggregates


def prepare_mixed_control_dataset(v11_source, v12_source, output):
    """Splice authenticated V11 lines into V12 without changing supervision."""
    source11 = _load(v11_source, "carryrl.control_text_data.v11")
    source12 = _load(v12_source, "carryrl.control_text_data.v12")
    path11, manifest11, records11, payload11, manifest_payload11, index11 = source11
    path12, manifest12, records12, payload12, manifest_payload12, index12 = source12
    lineage = manifest12.get("structured_augmentation", {})
    if (lineage.get("base_records_sha256") != hashlib.sha256(payload11).hexdigest()
            or lineage.get("base_manifest_sha256")
            != hashlib.sha256(manifest_payload11).hexdigest()):
        raise ValueError("V12 lineage does not authenticate the supplied V11 source")
    if [_key(row) for row in records11] != [_key(row) for row in records12]:
        raise ValueError("V11/V12 input view identifiers or ordering mismatch")
    for key, (row11, line11) in index11.items():
        row12, line12 = index12[key]
        immutable11 = {field: value for field, value in row11.items() if field not in _TEXT_FIELDS}
        immutable12 = {field: value for field, value in row12.items() if field not in _TEXT_FIELDS}
        if immutable11 != immutable12:
            raise ValueError("V11/V12 immutable input fields mismatch")
        if _ordinary(row11):
            if not row12["template_family"].startswith("v12-structured-"):
                raise ValueError("V12 input is not the complete structured V11 derivative")
        elif line11 != line12:
            raise ValueError("V11/V12 nonordinary or heldout input bytes mismatch")
    selected = select_restored_view_ids(records11)
    rows, lines, restored_by_language = [], [], Counter()
    for row12 in records12:
        key = _key(row12)
        if key in selected:
            row, line = index11[key]
            restored_by_language[row["language"]] += 1
        else:
            row, line = index12[key]
        rows.append(row)
        lines.append(line)
    payload = b"".join(lines)
    summary = wording_manifest(rows)
    templates, grammars, coverage, witnesses, aggregates = _actual_coverage(rows)
    splits = ("train", "dev", "calibration", "sealed_test")
    relation_metadata = dict(manifest12["relation_grounding_repair"])
    relation_metadata["coverage_records"] = coverage
    relation_metadata["coverage_scope"] = "Current V13 rows after exact V11 wording restoration."
    sources = {
        name: {"path": str(path.resolve()), "records_sha256": hashlib.sha256(data).hexdigest(),
               "manifest_sha256": hashlib.sha256(document).hexdigest()}
        for name, path, data, document in (("v11", path11, payload11, manifest_payload11),
                                           ("v12", path12, payload12, manifest_payload12))
    }
    train = [row for row in rows if row["split"] == "train"]
    metadata = {
        "version": MIXED_VERSION, "sources": sources,
        "selection": "Minimum V11 ordinary variant_index for each TRAIN group_id/language.",
        "restored_records": len(selected),
        "restored_existing_group_language_ids": len(selected),
        "restored_records_by_language": dict(restored_by_language),
        "restored_existing_semantic_groups": len({group_id for group_id, _ in selected}),
        "remaining_concrete_train_records": sum(row["template_family"].startswith("v12-structured-")
                                                for row in train),
        "preserved_natural_train_records": sum(row["template_family"].startswith("v9-natural-")
                                               for row in train),
        "preserved_directed_train_records": sum(row["template_family"].startswith("v11-transfer-")
                                                for row in train),
        "added_records": 0, "new_independent_semantic_groups": 0,
        "restored_lines_identical_to_v11": True, "all_other_lines_identical_to_v12": True,
        "labels_ids_splits_pair_metadata_unchanged": True,
        "evaluation_wording_protocol_changed": False,
        "heldout_records_sha256_by_split": {
            split: hashlib.sha256(b"".join(line for row, line in zip(rows, lines, strict=True)
                                           if row["split"] == split)).hexdigest()
            for split in splits if split != "train"
        },
        "rendering_or_model_forward_performed": False,
        "effectiveness_measured": False,
        "inherited_stage_metadata_scope": "Historical preparations; current composition is V13.",
    }
    updated = {
        **manifest12, **summary,
        "version": CONTROL_TEXT_DATA_VERSION, "wording_version": 13,
        "semantic_generator_version": CONTROL_TEXT_DATA_VERSION,
        "scenario_kinds": manifest12["scenario_kinds"],
        "records_sha256": hashlib.sha256(payload).hexdigest(), "records_bytes": len(payload),
        "template_family_split_ownership": templates, "grammar_family_split_ownership": grammars,
        "relation_grounding_repair": relation_metadata,
        "relation_witness_group_counts": {split: len(witnesses[split]) for split in splits},
        "relation_witness_split_ownership": {family: split for family, split in grammars.items()
                                             if family.startswith("v9-relation-witness-")},
        "aggregate_axis_group_counts": {split: len(aggregates[split]) for split in splits},
        "mixed_augmentation": metadata,
        "status": "prepared", "updates": 0, "model_loaded": False,
        "readiness": False, "ppo_eligible": False,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "records.jsonl").write_bytes(payload)
    (output / "manifest.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return updated
