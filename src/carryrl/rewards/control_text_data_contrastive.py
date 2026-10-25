"""Optional paired TRAIN augmentation of an immutable V8 preparation.

This helper prepares structural supervision only. It never reads diagnostics,
public runtime strings, model outputs, or heldout text to construct examples.
Every existing JSONL byte is retained; appended groups are admitted in pairs
before rendering against historical seals and all base evaluation identities.
"""

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from carryrl.rewards.control_text_data_v4 import ControlSpec, OperationSpec
from carryrl.rewards.control_text_data_v8 import (
    _BUTTONS,
    _KEYS,
    CONTROL_TEXT_DATA_VERSION,
    _identity,
    read_partition_registry,
    render_spec,
    wording_manifest,
)

AUGMENTATION_VERSION = "carryrl.control_text_contrastive.v1"
CONTRAST_KINDS = ("cross_modal_alias", "explicit_status", "adjacent_timing")
_MEMBERS = {
    "cross_modal_alias": ("equivalent", "independent"),
    "explicit_status": ("supported", "unsupported"),
    "adjacent_timing": ("first_held_second_press", "first_press_second_held"),
}


def _spec(operations, kind, *, forbidden=()):
    return ControlSpec(
        tuple(operations), (), tuple(forbidden), "unsupported", f"contrastive_{kind}"
    )


def _paired_specs(rng, kind):
    """Change only the requested property; keep all other candidate labels fixed."""
    if kind == "cross_modal_alias":
        key, button = rng.choice(_KEYS), rng.choice(_BUTTONS)
        mode = rng.choice(("held", "press"))
        equivalent = _spec([OperationSpec("000", (key, button), mode)], kind)
        independent = _spec([OperationSpec("000", (key,), mode),
                             OperationSpec("001", (button,), mode)], kind)
        return (equivalent, independent), (key, button)
    if kind == "explicit_status":
        target, first, second = rng.sample(_KEYS, 3)
        context = [OperationSpec("000", (first,), rng.choice(("held", "press"))),
                   OperationSpec("001", (second,), rng.choice(("held", "press")))]
        supported = _spec(context + [OperationSpec("002", (target,),
                                                   rng.choice(("held", "press")))], kind)
        unsupported = _spec(context, kind, forbidden=(target,))
        return (supported, unsupported), (target,)
    if kind == "adjacent_timing":
        first, second = rng.sample(_KEYS, 2)
        held_press = _spec([OperationSpec("000", (first,), "held"),
                            OperationSpec("001", (second,), "press")], kind)
        press_held = _spec([OperationSpec("000", (first,), "press"),
                            OperationSpec("001", (second,), "held")], kind)
        return (held_press, press_held), (first, second)
    raise ValueError(f"unknown contrast kind: {kind}")


def generate_contrastive_train_records(
    base_records, *, partition_registry, seed=20261025, pairs_per_kind=24
):
    """Return new TRAIN rows and paired provenance, without mutating base rows.

    Both members must be novel effective groups. A collision or blocked member
    rejects the entire proposal. Elementary=False keeps bounded probe selection
    possible; a probe that samples these rows should select complete pairs.
    """
    if type(pairs_per_kind) is not int or not 1 <= pairs_per_kind <= 30:
        raise ValueError("pairs_per_kind must be an integer between 1 and 30")
    protected = {row["semantic_mapping_id"] for row in base_records if row["split"] != "train"}
    seen = {row["semantic_mapping_id"] for row in base_records}
    text_identities = {hashlib.sha256(row["text"].encode()).hexdigest(): row["semantic_mapping_id"]
                       for row in base_records}
    group_ids = {row["group_id"] for row in base_records}
    rng, rows, pairs = random.Random(seed), [], []
    for kind in CONTRAST_KINDS:
        for _ in range(pairs_per_kind):
            for _attempt in range(10000):
                specs, targets = _paired_specs(rng, kind)
                annotations = [_identity(spec) for spec in specs]
                identities = [identity for identity, _ in annotations]
                if len(set(identities)) != 2 or any(
                    identity in seen or identity in protected
                    or not partition_registry.allows(identity, "train")
                    for identity in identities
                ):
                    continue
                pair_id = hashlib.sha256(json.dumps(
                    [AUGMENTATION_VERSION, seed, kind, identities], separators=(",", ":")
                ).encode()).hexdigest()
                pending = []
                for member_index, (spec, (identity, labels)) in enumerate(zip(
                    specs, annotations, strict=True
                )):
                    group_id = f"v8c-contrast-{pair_id[:20]}-{member_index}"
                    if group_id in group_ids:
                        raise ValueError("contrastive group ID collision")
                    for variant in range(6):
                        language = "ru" if variant < 3 else "en"
                        # Shared primitive composition, remainder form and RNG
                        # within a pair avoid making grammar a label shortcut.
                        render_seed = int(hashlib.sha256(
                            f"{pair_id}:{variant}".encode()
                        ).hexdigest()[:16], 16)
                        text, grammars = render_spec(
                            spec, language=language, variant=variant, split="train",
                            seed=render_seed, pair_seed=pair_id,
                        )
                        pending.append({
                            "group_id": group_id, "split": "train",
                            "mapping_id": identity, "semantic_mapping_id": identity,
                            "declared_candidates": list(spec.declared_candidates),
                            "template_family": grammars[0], "grammar_families": grammars,
                            "scenario_kind": spec.kind, "language": language, "text": text,
                            "labels": labels, "specification": asdict(spec),
                            "elementary": False, "variant_index": variant, "group_variants": 6,
                            "developmental_compatibility": identity in (
                                partition_registry.developmental_ids
                            ),
                            "contrastive": True, "contrastive_pair_id": pair_id,
                            "contrastive_kind": kind,
                            "contrastive_member": _MEMBERS[kind][member_index],
                            "contrastive_target_candidates": list(targets),
                        })
                pending_texts = {}
                conflict = False
                for row in pending:
                    text_hash = hashlib.sha256(row["text"].encode()).hexdigest()
                    identity = row["semantic_mapping_id"]
                    if (text_identities.get(text_hash, identity) != identity
                            or pending_texts.setdefault(text_hash, identity) != identity):
                        conflict = True
                        break
                if conflict:
                    continue
                seen.update(identities)
                group_ids.update(row["group_id"] for row in pending)
                text_identities.update(pending_texts)
                rows.extend(pending)
                pairs.append({
                    "pair_id": pair_id, "kind": kind, "target_candidates": list(targets),
                    "group_ids": [pending[0]["group_id"], pending[6]["group_id"]],
                    "semantic_mapping_ids": identities, "members": list(_MEMBERS[kind]),
                })
                break
            else:
                raise ValueError(f"unable to admit a fresh complete contrastive pair for {kind}")
    return rows, pairs


def _heldout_digests(records, payload):
    """Hash exact JSONL lines, preserving their ordering and serialization."""
    lines = payload.splitlines(keepends=True)
    if len(lines) != len(records):
        raise ValueError("base JSONL must contain exactly one record on each line")
    return {
        split: hashlib.sha256(b"".join(line for row, line in zip(records, lines, strict=True)
                                      if row["split"] == split)).hexdigest()
        for split in ("dev", "calibration", "sealed_test")
    }


def prepare_contrastive_control_dataset(
    base_dataset, output, *, reservation_registry, seed=20261025, pairs_per_kind=24
):
    """Prepare a separate dataset, preserving every existing row byte for byte."""
    from carryrl.rewards.control_text_training import read_control_dataset

    base_dataset, output = Path(base_dataset), Path(output)
    base_manifest, base_records = read_control_dataset(base_dataset)
    if base_manifest.get("version") != CONTROL_TEXT_DATA_VERSION:
        raise ValueError("contrastive augmentation requires a v8 base dataset")
    if base_manifest.get("contrastive_augmentation") is not None:
        raise ValueError("contrastive augmentation requires an unaugmented base dataset")
    base_payload = (base_dataset / "records.jsonl").read_bytes()
    if hashlib.sha256(base_payload).hexdigest() != base_manifest["records_sha256"]:
        raise ValueError("base records changed during contrastive preparation")
    if not base_payload.endswith(b"\n"):
        raise ValueError("base JSONL must end with a newline to preserve all existing bytes")
    registry, registry_metadata = read_partition_registry(reservation_registry)
    if any(row["semantic_mapping_id"] in registry.hard_old_sealed_ids
           for row in base_records if row["split"] == "train"):
        raise ValueError("base TRAIN contains a mapping blocked by the supplied historical seal")
    augmented, pairs = generate_contrastive_train_records(
        base_records, partition_registry=registry, seed=seed, pairs_per_kind=pairs_per_kind
    )
    records = base_records + augmented
    appended_payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                               for row in augmented).encode()
    payload = base_payload + appended_payload
    digests = _heldout_digests(base_records, base_payload)
    if _heldout_digests(records, payload) != digests:
        raise ValueError("contrastive preparation changed heldout bytes")
    summary = wording_manifest(records)
    summary["scenario_kinds"] = list(dict.fromkeys(
        list(base_manifest["scenario_kinds"]) + [f"contrastive_{kind}" for kind in CONTRAST_KINDS]
    ))
    group_counts = Counter({row["group_id"]: row["split"] for row in records}.values())
    metadata = {
        "version": AUGMENTATION_VERSION, "seed": seed, "pairs_per_kind": pairs_per_kind,
        "pairs": len(pairs), "groups": len(pairs) * 2, "records": len(augmented),
        "pair_kind_counts": dict(Counter(pair["kind"] for pair in pairs)),
        "pair_registry": pairs,
        "all_new_rows_split": "train", "split_before_paraphrase": True,
        "paired_admission": "Both identities accepted before any text is rendered.",
        "elementary": False,
        "probe_sampling": "Select complete pairs within an explicitly bounded contrastive quota.",
        "base_dataset": str(base_dataset.resolve()),
        "base_records_sha256": hashlib.sha256(base_payload).hexdigest(),
        "base_manifest_sha256": hashlib.sha256(
            (base_dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "base_records": len(base_records), "base_records_bytes": len(base_payload),
        "base_registry_sha256": base_manifest.get("partition_registry_sha256"),
        "heldout_mapping_count": len({row["semantic_mapping_id"] for row in base_records
                                      if row["split"] != "train"}),
        "heldout_records_sha256_by_split": digests,
        "base_jsonl_prefix_preserved": True, "heldout_record_bytes_preserved": True,
        "runtime_or_diagnostic_text_used": False, "effectiveness_measured": False,
    }
    manifest = {
        **base_manifest, **summary, **registry_metadata,
        "scenario_groups": sum(group_counts.values()), "records": len(records),
        "split_records": dict(Counter(row["split"] for row in records)),
        "split_groups": dict(group_counts),
        "records_sha256": hashlib.sha256(payload).hexdigest(), "records_bytes": len(payload),
        "status": "prepared", "updates": 0, "model_loaded": False,
        "readiness": False, "ppo_eligible": False,
        "contrastive_augmentation": metadata,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "records.jsonl").write_bytes(payload)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
