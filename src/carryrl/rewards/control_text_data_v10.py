"""V10: repair historical capitalized dummy-W canonical binding clauses.

All supervision and split identities stay frozen. Only the complete defective
canonical clause is replaced, retaining goal text, other bindings, relational
witnesses, natural TRAIN views, and every unaffected JSONL line byte for byte.
The new protocol explicitly includes corrected evaluation wording.
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from carryrl.rewards.control_text_data_v4 import ControlSpec, OperationSpec
from carryrl.rewards.control_text_data_v8 import (
    INTERNAL_FAMILIES,
    _canonical,
    _canonical_names,
    annotate_spec,
)

CONTROL_TEXT_DATA_VERSION = "carryrl.control_text_data.v10"
REPAIR_VERSION = "carryrl.control_text_canonical_repair.v1"


def _specification(row):
    source = row["specification"]
    return ControlSpec(tuple(OperationSpec(operation["identifier"], tuple(operation["candidates"]),
                                           operation["mode"])
                             for operation in source["operations"]),
                       tuple(source["uncertain"]), tuple(source["forbidden"]),
                       source["remainder"], source["kind"])


def repair_canonical_view(row):
    """Fix one confirmed historical rendering defect; never replace a real W.

    Old name-first Russian grammars capitalized the dummy spelling before a
    case-sensitive substitution. The legacy signature is reconstructed from the
    full canonical clause and its grammar, rather than from a bare W mention.
    """
    if row["template_family"].startswith("v9-natural-"):
        return row, False
    if row["language"] != "ru" or row["specification"]["kind"] not in {
        "navigation", "horizontal"
    }:
        return row, False
    match = re.fullmatch(r"v8-binding-composition-(\d{2})", row["template_family"])
    if match is None or int(match[1]) not in INTERNAL_FAMILIES.get(row["split"], ()):
        raise ValueError("canonical repair requires a split-owned original binding grammar")
    spec = _specification(row)
    if annotate_spec(spec) != row["labels"]:
        raise ValueError("canonical specification disagrees with frozen supervision")
    core, corrected = _canonical(spec, row["language"], int(match[1]))
    names = _canonical_names(spec.kind, row["language"])
    if not core or not corrected.startswith(names):
        return row, False
    # This is a quarantined historical defect signature, not a rendering input.
    # Correct rendering passes real compact names directly to the binding grammar.
    legacy = "Клавиша W" + corrected[len(names):]
    old_clause, new_clause = legacy + ".", corrected + "."
    occurrences = row["text"].count(old_clause)
    if occurrences == 0:
        if new_clause not in row["text"]:
            raise ValueError("canonical view matches neither the historical nor corrected clause")
        return row, False
    if occurrences != 1:
        raise ValueError("historical canonical defect has an ambiguous text span")
    repaired = dict(row)
    repaired["text"] = row["text"].replace(old_clause, new_clause, 1)
    return repaired, True


def prepare_canonical_repaired_dataset(base_v9_natural, output):
    """Prepare V10 separately; never write to the source preparation."""
    from carryrl.rewards.control_text_training import read_control_dataset

    base, output = Path(base_v9_natural), Path(output)
    manifest, records = read_control_dataset(base)
    if (manifest.get("version") != "carryrl.control_text_data.v9"
            or not isinstance(manifest.get("natural_augmentation"), dict)):
        raise ValueError("canonical repair requires a V9 natural preparation")
    source_payload = (base / "records.jsonl").read_bytes()
    source_manifest = (base / "manifest.json").read_bytes()
    if (hashlib.sha256(source_payload).hexdigest() != manifest["records_sha256"]
            or json.loads(source_manifest) != manifest):
        raise ValueError("base preparation changed during canonical repair")
    lines = source_payload.splitlines(keepends=True)
    if len(lines) != len(records):
        raise ValueError("base JSONL must have exactly one record per line")
    repaired, output_lines, counts = [], [], Counter()
    changed_groups = set()
    for row, line in zip(records, lines, strict=True):
        current, changed = repair_canonical_view(row)
        if changed:
            line = (json.dumps(current, ensure_ascii=False, sort_keys=True) + "\n").encode()
            counts[f"{row['split']}|{row['scenario_kind']}|{row['template_family']}"] += 1
            changed_groups.add(row["group_id"])
        repaired.append(current)
        output_lines.append(line)
    payload = b"".join(output_lines)
    changed_count = sum(counts.values())
    metadata = {
        "version": REPAIR_VERSION,
        "base_dataset": str(base.resolve()), "base_version": manifest["version"],
        "base_records_sha256": hashlib.sha256(source_payload).hexdigest(),
        "base_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
        "changed_records": changed_count, "changed_semantic_groups": len(changed_groups),
        "changed_records_by_split_kind_grammar": dict(counts),
        "unchanged_record_bytes_preserved": True,
        "natural_record_bytes_preserved": True,
        "structural_labels_ids_splits_pair_metadata_unchanged": True,
        "changed_fields": ["text"],
        "evaluation_wording_protocol_changed": True,
        "heldout_mappings_moved_to_train": 0,
        "scope": "Only the exact complete historical dummy-W canonical binding clause.",
        "rendering_fix": (
            "Canonical names passed directly before capitalization; no dummy substitution."
        ),
        "inherited_natural_augmentation_metadata_scope": (
            "Historical V9-natural stage provenance; current wording protocol is V10."
        ),
        "runtime_or_model_predictions_used": False,
        "effectiveness_measured": False,
    }
    updated = {
        **manifest,
        "version": CONTROL_TEXT_DATA_VERSION, "wording_version": 10,
        "semantic_generator_version": CONTROL_TEXT_DATA_VERSION,
        "records_sha256": hashlib.sha256(payload).hexdigest(), "records_bytes": len(payload),
        "canonical_render_repair": metadata,
        "status": "prepared", "updates": 0, "model_loaded": False,
        "readiness": False, "ppo_eligible": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "records.jsonl").write_bytes(payload)
    (output / "manifest.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return updated
