"""Verified composite sources flatten exact RGB references without rewriting packs."""

import hashlib
import json

import numpy as np
import pytest
from test_feature_reuse import identity, prepare, write
from test_feature_reuse import source as source

from carryrl.training.feature_store import FrozenFeatureReuse, FrozenFeatureStore


def save_run(path, plan, frames):
    path.mkdir()
    np.savez_compressed(path / "demonstration_frames.npz", frames=np.stack(frames))
    recorded = {"feature_reuse": plan.manifest, "bytes": plan.logical_bytes,
                "frames": len(plan), "dtype": plan.current_identity["dtype"],
                "vision_sha256": plan.current_identity["vision_sha256"]}
    (path / "feature_manifest.json").write_text(json.dumps(recorded))
    return path


@pytest.fixture
def composite(source):
    run, root, old, _ = source
    frames = [old[2], old[0], old[1]]
    plan = prepare(source, frames)
    shard = write(root, plan.shard_identity, np.full((1, 4, 384), 33, np.float16))
    plan.bind_shard(shard)
    return save_run(run.parent / "layer1", plan, frames), root, frames, shard


def test_composite_remapping_missing_rows_and_nested_flattening(composite, monkeypatch):
    run, root, frames, _ = composite
    remapped = [frames[2], frames[0], frames[2].copy(), np.full((3, 4, 3), 7, np.uint8), frames[1]]
    plan = prepare(composite, remapped)
    assert plan.reused_count == 4 and plan.missing_indices == [3]
    assert plan[0][0, 0].item() == 33 and plan[1][0, 0].item() == 22
    plan[2].fill_(0)
    assert plan[2][0, 0].item() == 33
    plan.bind_shard(write(root, plan.shard_identity, np.full((1, 4, 384), 44, np.float16)))
    manifest = plan.manifest
    assert manifest["source_pack"] is None
    assert manifest["source_reuse"]["source_run"] == str(run)
    assert manifest["source_reuse"]["manifest_sha256"] == hashlib.sha256(
        (run / "feature_manifest.json").read_bytes()).hexdigest()
    layer2 = save_run(run.parent / "layer2", plan, remapped)
    verified_paths = []
    original_load = FrozenFeatureStore.load

    def load(cache, key, *, max_bytes):
        result = original_load(cache, key, max_bytes=max_bytes)
        verified_paths.append(result.directory)
        return result

    monkeypatch.setattr(FrozenFeatureStore, "load", load)
    latest_frames = [remapped[3], frames[2], frames[1]]
    latest = FrozenFeatureReuse.prepare(layer2, latest_frames, [0, 1, 2],
        identity(latest_frames, [0, 1, 2]), cache_root=root, max_bytes=1024**2)
    assert latest.missing_indices == [] and latest.reused_count == 3
    assert [latest[index][0, 0].item() for index in range(3)] == [44, 33, 11]
    assert len(verified_paths) == len(set(verified_paths)) == 3
    assert len(list(root.iterdir())) == 3  # No duplicate full pack or rewritten shard.


@pytest.mark.parametrize("damage", ["cycle", "origin", "unordered", "unknown_source", "rgb_ref",
    "mapping_digest", "counts", "identity", "shard_identity", "pack_hash", "pack_path",
    "shard_rgb", "source_rgb"])
def test_composite_malformed_maps_cycles_and_source_mutations_fail_closed(composite, damage):
    run, root, frames, shard = composite
    path = run / "feature_manifest.json"
    recorded = json.loads(path.read_text())
    reuse = recorded["feature_reuse"]
    if damage == "cycle":
        reuse["source_run"] = str(run)
    elif damage == "origin":
        reuse["mapping"][0][1] = "other"
    elif damage == "unordered":
        reuse["mapping"].reverse()
    elif damage == "unknown_source":
        reuse["mapping"][0][2] = 500
    elif damage == "rgb_ref":
        reuse["mapping"][0][2] = 0
    elif damage == "mapping_digest":
        reuse["mapping_sha256"] = "0" * 64
    elif damage == "counts":
        reuse["reused_count"] += 1
    elif damage == "identity":
        reuse["identity"]["ordered_rgb_sha256"] = "0" * 64
    elif damage == "shard_identity":
        reuse["shard_pack"]["identity"]["dtype"] = "float32"
    elif damage == "pack_hash":
        with (shard.directory / "features.npy").open("r+b") as output:
            output.seek(-1, 2)
            output.write(b"x")
    elif damage == "pack_path":
        reuse["shard_pack"]["directory"] = str(root.parent / "outside")
    elif damage == "shard_rgb":
        changed = np.stack(frames)
        changed[2, 0, 0, 0] += 1
        np.savez_compressed(run / "demonstration_frames.npz", frames=changed)
        # A coherent composite input identity alone cannot repoint a frozen shard.
        reuse["identity"] = identity(changed, [0, 1, 2])
    elif damage == "source_rgb":
        parent = run.parent / "run/demonstration_frames.npz"
        with np.load(parent) as data:
            changed = data["frames"].copy()
        changed[0, 0, 0, 0] += 1
        np.savez_compressed(parent, frames=changed)
    if damage in {"origin", "unordered", "unknown_source", "rgb_ref"}:
        reuse["mapping_sha256"] = hashlib.sha256(json.dumps(reuse["mapping"]).encode()).hexdigest()
    path.write_text(json.dumps(recorded))
    with pytest.raises(ValueError):
        prepare(composite, frames)


def test_composite_parent_manifest_pin_rejects_even_harmless_mutation(composite):
    run, _, frames, _ = composite
    plan = prepare(composite, frames)
    layer2 = save_run(run.parent / "layer2", plan, frames)
    path = run / "feature_manifest.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="parent reuse reference"):
        prepare((layer2, composite[1], frames, None), frames)
