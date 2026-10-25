"""Optional augmentation preserves evaluation bytes and provides paired targets."""

import hashlib
import json
import random
from collections import Counter, defaultdict

import pytest

from carryrl.rewards.control_text_data_contrastive import (
    CONTRAST_KINDS,
    _paired_specs,
    generate_contrastive_train_records,
    prepare_contrastive_control_dataset,
)
from carryrl.rewards.control_text_data_v8 import (
    PARTITION_REGISTRY_VERSION,
    PartitionRegistry,
    _identity,
    generate_control_text_records,
    prepare_control_text_dataset,
)
from carryrl.rewards.control_text_grounding import _mention_spans
from carryrl.rewards.control_text_training import read_control_dataset


@pytest.fixture(scope="module")
def base_records():
    return generate_control_text_records(scenario_groups=200)


@pytest.fixture(scope="module")
def augmentation(base_records):
    return generate_contrastive_train_records(
        base_records, partition_registry=PartitionRegistry(), pairs_per_kind=3
    )


def test_each_contrast_changes_only_the_requested_property(augmentation):
    rows, pairs = augmentation
    groups = {row["group_id"]: row for row in rows}
    assert len(pairs) == 9
    assert len(rows) == 9 * 2 * 6
    assert Counter(pair["kind"] for pair in pairs) == dict.fromkeys(CONTRAST_KINDS, 3)
    for pair in pairs:
        first, second = [groups[group] for group in pair["group_ids"]]
        a = {label["candidate"]: label for label in first["labels"]}
        b = {label["candidate"]: label for label in second["labels"]}
        targets = set(pair["target_candidates"])
        assert all(a[candidate] == b[candidate] for candidate in a if candidate not in targets)
        if pair["kind"] == "cross_modal_alias":
            key, mouse = pair["target_candidates"]
            assert key.startswith("key:") and mouse.startswith("mouse:")
            assert a[key]["alias_group"] == a[mouse]["alias_group"]
            assert b[key]["alias_group"] != b[mouse]["alias_group"]
            assert {a[key]["mode"], a[mouse]["mode"], b[key]["mode"], b[mouse]["mode"]} <= {
                "held", "press",
            }
            for candidate in targets:
                assert a[candidate]["status"] == b[candidate]["status"] == "supported"
                assert a[candidate]["mode"] == b[candidate]["mode"]
        elif pair["kind"] == "explicit_status":
            target, = targets
            assert a[target]["status"] == "supported"
            assert a[target]["mode"] in {"held", "press"}
            assert b[target]["status"] == "unsupported"
            assert b[target]["mode"] == "unknown" and b[target]["alias_group"] == 0
            assert second["specification"]["forbidden"] == (target,)
        else:
            first_key, second_key = pair["target_candidates"]
            assert (a[first_key]["mode"], a[second_key]["mode"]) == ("held", "press")
            assert (b[first_key]["mode"], b[second_key]["mode"]) == ("press", "held")
            for candidate in targets:
                assert a[candidate]["status"] == b[candidate]["status"] == "supported"
                assert a[candidate]["alias_group"] == b[candidate]["alias_group"]


def test_new_views_are_train_only_paired_natural_and_not_mandatory(augmentation, base_records):
    rows, pairs = augmentation
    old_ids = {row["semantic_mapping_id"] for row in base_records}
    old_texts = {row["text"] for row in base_records}
    pair_groups = defaultdict(set)
    variants = defaultdict(list)
    for row in rows:
        assert row["split"] == "train"
        assert row["semantic_mapping_id"] not in old_ids
        assert row["text"] not in old_texts
        assert row["elementary"] is False and row["contrastive"] is True
        assert row["group_variants"] == 6
        pair_groups[row["contrastive_pair_id"]].add(row["group_id"])
        variants[row["group_id"]].append(row)
        mentions = _mention_spans(row["text"])
        assert all(mentions[candidate] for candidate in row["declared_candidates"])
        assert "Для клавишу" not in row["text"]
        if row["contrastive_member"] == "unsupported":
            assert ("запрещён" if row["language"] == "ru" else "unsupported") in row["text"]
        if row["contrastive_member"] == "equivalent":
            assert ("равнозначны" if row["language"] == "ru" else "equivalent") in row["text"]
    assert all(len(groups) == 2 for groups in pair_groups.values())
    for views in variants.values():
        assert len({row["text"] for row in views}) == 6
        assert Counter(row["language"] for row in views) == {"ru": 3, "en": 3}
    for pair in pairs:
        first, second = [variants[group] for group in pair["group_ids"]]
        assert [row["template_family"] for row in first] == [
            row["template_family"] for row in second
        ]


def test_historical_seal_and_base_heldout_collision_reject_entire_pair():
    specs, _ = _paired_specs(random.Random(19), "cross_modal_alias")
    first, second = [_identity(spec)[0] for spec in specs]
    # Strict priority still applies if the blocked ID is also developmental.
    registry = PartitionRegistry(hard_old_sealed_ids=frozenset([first]),
                                 developmental_ids=frozenset([first]),
                                 previous_mapping_ids=frozenset([first]))
    rows, pairs = generate_contrastive_train_records(
        [], partition_registry=registry, seed=19, pairs_per_kind=1
    )
    identities = {row["semantic_mapping_id"] for row in rows}
    assert first not in identities and second not in identities
    protected = [{"semantic_mapping_id": first, "split": "dev", "group_id": "old-dev",
                  "text": "A private evaluation example is not reused."}]
    rows, pairs = generate_contrastive_train_records(
        protected, partition_registry=PartitionRegistry(), seed=19, pairs_per_kind=1
    )
    identities = {row["semantic_mapping_id"] for row in rows}
    assert first not in identities and second not in identities
    assert len(pairs) == 3


def test_existing_train_identity_collision_is_not_duplicated():
    specs, _ = _paired_specs(random.Random(23), "cross_modal_alias")
    identity = _identity(specs[0])[0]
    base = [{"semantic_mapping_id": identity, "split": "train", "group_id": "old-train",
             "text": "Existing training example."}]
    rows, _ = generate_contrastive_train_records(
        base, partition_registry=PartitionRegistry(), seed=23, pairs_per_kind=1
    )
    assert identity not in {row["semantic_mapping_id"] for row in rows}
    assert len({row["semantic_mapping_id"] for row in rows}) == 6


def test_prepare_keeps_every_base_line_and_heldout_record_byte_for_byte(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({
        "version": PARTITION_REGISTRY_VERSION, "hard_old_sealed_ids": [],
        "developmental_ids": [], "previous_mapping_ids": [],
    }))
    base = tmp_path / "base"
    prepare_control_text_dataset(base, scenario_groups=200, reservation_registry=registry_path)
    original_payload = (base / "records.jsonl").read_bytes()
    original_manifest = (base / "manifest.json").read_bytes()
    destination = tmp_path / "augmented"
    manifest = prepare_contrastive_control_dataset(
        base, destination, reservation_registry=registry_path, pairs_per_kind=2
    )
    payload = (destination / "records.jsonl").read_bytes()
    assert payload.startswith(original_payload)
    assert (base / "records.jsonl").read_bytes() == original_payload
    assert (base / "manifest.json").read_bytes() == original_manifest
    _, original_rows = read_control_dataset(base)
    loaded_manifest, rows = read_control_dataset(destination)
    assert rows[:len(original_rows)] == original_rows
    assert all(row["split"] == "train" for row in rows[len(original_rows):])
    for split in ("dev", "calibration", "sealed_test"):
        before = b"".join(line for line in original_payload.splitlines(keepends=True)
                          if json.loads(line)["split"] == split)
        after = b"".join(line for line in payload.splitlines(keepends=True)
                         if json.loads(line)["split"] == split)
        assert before == after
        assert manifest["contrastive_augmentation"]["heldout_records_sha256_by_split"][split] == (
            hashlib.sha256(before).hexdigest()
        )
    assert manifest["version"] == "carryrl.control_text_data.v8"
    assert manifest["scenario_groups"] == 212
    assert manifest["contrastive_augmentation"]["pairs"] == 6
    assert manifest["contrastive_augmentation"]["base_records_sha256"] == hashlib.sha256(
        original_payload
    ).hexdigest()
    assert manifest["contrastive_augmentation"]["base_manifest_sha256"] == hashlib.sha256(
        original_manifest
    ).hexdigest()
    assert manifest["records_sha256"] == loaded_manifest["records_sha256"]
    assert manifest["readiness"] is manifest["model_loaded"] is manifest["ppo_eligible"] is False
    assert manifest["contrastive_augmentation"]["effectiveness_measured"] is False
    assert manifest["model_input_fields"] == ["text"]
    with pytest.raises(FileExistsError):
        prepare_contrastive_control_dataset(
            base, destination, reservation_registry=registry_path, pairs_per_kind=1
        )
    with pytest.raises(ValueError, match="unaugmented"):
        prepare_contrastive_control_dataset(
            destination, tmp_path / "again", reservation_registry=registry_path, pairs_per_kind=1
        )


def test_default_augmentation_budget_and_determinism(base_records):
    first, pairs = generate_contrastive_train_records(
        base_records, partition_registry=PartitionRegistry()
    )
    second, repeated_pairs = generate_contrastive_train_records(
        base_records, partition_registry=PartitionRegistry()
    )
    assert first == second and pairs == repeated_pairs
    assert len(pairs) == 72
    assert len({row["semantic_mapping_id"] for row in first}) == 144
    assert len(first) == 864
    for value in (False, 0, 31, 1.5):
        with pytest.raises(ValueError, match="pairs_per_kind"):
            generate_contrastive_train_records(base_records, partition_registry=PartitionRegistry(),
                                               pairs_per_kind=value)
