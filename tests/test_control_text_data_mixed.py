"""Exact V11-line restoration into V12, with authenticated paired lineage."""

import hashlib
import json
import random
import shutil
from collections import Counter

import pytest

from carryrl.rewards.control_text_data_contrastive import prepare_contrastive_control_dataset
from carryrl.rewards.control_text_data_mixed import (
    prepare_mixed_control_dataset,
    select_restored_view_ids,
)
from carryrl.rewards.control_text_data_natural import prepare_natural_control_dataset
from carryrl.rewards.control_text_data_structured import prepare_structured_control_dataset
from carryrl.rewards.control_text_data_transfer import prepare_transfer_control_dataset
from carryrl.rewards.control_text_data_v8 import (
    PARTITION_REGISTRY_VERSION,
    prepare_control_text_dataset,
)
from carryrl.rewards.control_text_data_v9 import prepare_relation_grounded_dataset
from carryrl.rewards.control_text_data_v10 import prepare_canonical_repaired_dataset
from carryrl.rewards.control_text_training import read_control_dataset


def test_selection_uses_minimum_variant_for_each_group_and_language_not_row_order():
    records = []
    for size in (6, 8):
        natural = {2, 5} if size == 6 else {3, 7}
        for variant in range(size):
            records.append({"group_id": str(size), "variant_index": variant, "split": "train",
                            "language": "ru" if variant < size // 2 else "en",
                            "template_family": "v9-natural-composition-00" if variant in natural
                            else "v8-binding-composition-00"})
    records.append({"group_id": "heldout", "variant_index": 0, "split": "dev",
                    "language": "ru", "template_family": "v8-binding-composition-12"})
    expected = {("6", 0), ("6", 3), ("8", 0), ("8", 4)}
    assert select_restored_view_ids(records) == expected
    random.Random(4).shuffle(records)
    assert select_restored_view_ids(records) == expected
    broken = [row for row in records if not (row["group_id"] == "6" and row["language"] == "en")]
    with pytest.raises(ValueError, match="Russian and English"):
        select_restored_view_ids(broken)
    broken = [dict(row) for row in records]
    for row in broken:
        if row["group_id"] == "6" and row["language"] == "en":
            row["template_family"] = "v9-natural-composition-00"
    with pytest.raises(ValueError, match="ordinary view"):
        select_restored_view_ids(broken)


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("mixed")
    registry = root / "registry.json"
    registry.write_text(json.dumps({"version": PARTITION_REGISTRY_VERSION,
                                    "hard_old_sealed_ids": [], "developmental_ids": [],
                                    "previous_mapping_ids": []}))
    v8, v8c, v9, natural, v10, v11, v12 = [root / name for name in (
        "v8", "v8c", "v9", "natural", "v10", "v11", "v12"
    )]
    prepare_control_text_dataset(v8, scenario_groups=400, reservation_registry=registry)
    prepare_contrastive_control_dataset(v8, v8c, reservation_registry=registry, pairs_per_kind=1)
    prepare_relation_grounded_dataset(v8c, v9)
    prepare_natural_control_dataset(v9, natural)
    prepare_canonical_repaired_dataset(natural, v10)
    prepare_transfer_control_dataset(v10, v11, timing_pairs=1, conditionals=2, mouse_per_mode=1)
    prepare_structured_control_dataset(v11, v12)
    return root, v11, v12


def test_exact_mixture_supervision_counts_ownership_and_source_bytes(sources):
    root, v11, v12 = sources
    manifest11, records11 = read_control_dataset(v11)
    manifest12, records12 = read_control_dataset(v12)
    bytes11, bytes12 = (v11 / "records.jsonl").read_bytes(), (v12 / "records.jsonl").read_bytes()
    doc11, doc12 = (v11 / "manifest.json").read_bytes(), (v12 / "manifest.json").read_bytes()
    selected = select_restored_view_ids(records11)
    output = root / "v13"
    manifest = prepare_mixed_control_dataset(v11, v12, output)
    _, rows = read_control_dataset(output)
    payload = (output / "records.jsonl").read_bytes()
    assert (v11 / "records.jsonl").read_bytes() == bytes11
    assert (v12 / "records.jsonl").read_bytes() == bytes12
    assert (v11 / "manifest.json").read_bytes() == doc11
    assert (v12 / "manifest.json").read_bytes() == doc12
    restored, totals_before, totals_after, owners = Counter(), Counter(), Counter(), {}
    for old11, old12, row, line11, line12, line in zip(
        records11, records12, rows, bytes11.splitlines(keepends=True),
        bytes12.splitlines(keepends=True), payload.splitlines(keepends=True), strict=True
    ):
        key = row["group_id"], row["variant_index"]
        if key in selected:
            restored[(row["group_id"], row["language"])] += 1
            assert line == line11 and row == old11
            assert row["split"] == "train"
        else:
            assert line == line12 and row == old12
        immutable = row.keys() - {"text", "template_family", "grammar_families"}
        assert all(row[field] == old11[field] == old12[field] for field in immutable)
        totals_before[(old12["group_id"], old12["language"])] += 1
        totals_after[(row["group_id"], row["language"])] += 1
        for family in row["grammar_families"]:
            assert owners.setdefault(family, row["split"]) == row["split"]
    assert totals_after == totals_before
    assert len(restored) == 2 * manifest11["split_groups"]["train"]
    assert set(restored.values()) == {1}
    metadata = manifest["mixed_augmentation"]
    assert metadata["restored_records"] == len(restored)
    concrete12 = sum(
        row["split"] == "train" and row["template_family"].startswith("v12-structured-")
        for row in records12
    )
    assert metadata["remaining_concrete_train_records"] == concrete12 - len(restored)
    for prefix, field in (("v9-natural-", "preserved_natural_train_records"),
                          ("v11-transfer-", "preserved_directed_train_records")):
        expected_count = sum(row["split"] == "train" and row["template_family"].startswith(prefix)
                             for row in records12)
        assert metadata[field] == expected_count
    assert metadata["restored_records_by_language"] == dict.fromkeys(
        ("ru", "en"), manifest11["split_groups"]["train"]
    )
    assert metadata["sources"]["v11"]["records_sha256"] == hashlib.sha256(bytes11).hexdigest()
    assert metadata["sources"]["v12"]["records_sha256"] == hashlib.sha256(bytes12).hexdigest()
    assert metadata["sources"]["v11"]["manifest_sha256"] == hashlib.sha256(doc11).hexdigest()
    assert metadata["sources"]["v12"]["manifest_sha256"] == hashlib.sha256(doc12).hexdigest()
    assert metadata["added_records"] == metadata["new_independent_semantic_groups"] == 0
    assert manifest["grammar_family_split_ownership"] == owners
    assert manifest["records"] == manifest12["records"]
    assert manifest["split_groups"] == manifest12["split_groups"]
    assert manifest["records_sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["version"] == "carryrl.control_text_data.v13"
    assert manifest["model_input_fields"] == ["text"]
    assert manifest["readiness"] is manifest["model_loaded"] is manifest["ppo_eligible"] is False
    for split in ("train", "dev", "calibration", "sealed_test"):
        actual_witness_records = sum(row["split"] == split and any(
            family.startswith("v9-relation-witness-") for family in row["grammar_families"]
        ) for row in rows)
        assert manifest["relation_grounding_repair"]["coverage_records"].get(
            f"{split}|witness_records", 0
        ) == actual_witness_records
        if split != "train":
            exact = b"".join(
                line for row, line in zip(rows, payload.splitlines(keepends=True), strict=True)
                if row["split"] == split
            )
            actual_digest = hashlib.sha256(exact).hexdigest()
            assert metadata["heldout_records_sha256_by_split"][split] == actual_digest
    with pytest.raises(FileExistsError):
        prepare_mixed_control_dataset(v11, v12, output)


@pytest.mark.parametrize("corruption", ("lineage", "immutable", "heldout", "missing_view"))
def test_lineage_and_input_mismatches_are_rejected_even_after_rehash(sources, tmp_path, corruption):
    _, v11, original12 = sources
    v12 = tmp_path / "copy12"
    shutil.copytree(original12, v12)
    manifest = json.loads((v12 / "manifest.json").read_text())
    if corruption == "lineage":
        manifest["structured_augmentation"]["base_manifest_sha256"] = "f" * 64
    else:
        rows = [json.loads(line) for line in (v12 / "records.jsonl").read_bytes().splitlines()]
        if corruption == "immutable":
            rows[0]["elementary"] = not rows[0]["elementary"]
        elif corruption == "heldout":
            next(row for row in rows if row["split"] == "dev")["text"] += " altered"
        else:
            rows.pop()
            manifest["records"] = len(rows)
        payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                          for row in rows).encode()
        (v12 / "records.jsonl").write_bytes(payload)
        manifest["records_sha256"] = hashlib.sha256(payload).hexdigest()
    (v12 / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="lineage|mismatch|incomplete"):
        prepare_mixed_control_dataset(v11, v12, tmp_path / "rejected")


def test_v13_reader_checks_semantics_and_cross_split_grammar(sources, tmp_path):
    _, v11, v12 = sources
    output = tmp_path / "v13"
    manifest = prepare_mixed_control_dataset(v11, v12, output)
    source = (output / "records.jsonl").read_bytes()
    for corruption in ("grammar", "identity"):
        rows = [json.loads(line) for line in source.splitlines()]
        if corruption == "grammar":
            victim = next(row for row in rows if row["split"] == "dev")
            victim["grammar_families"] = [rows[0]["template_family"]]
        else:
            rows[0]["semantic_mapping_id"] = "f" * 64
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (output / "records.jsonl").write_bytes(payload)
        (output / "manifest.json").write_text(json.dumps({
            **manifest, "records_sha256": hashlib.sha256(payload).hexdigest(),
        }))
        with pytest.raises(ValueError, match="leakage|identity"):
            read_control_dataset(output)
