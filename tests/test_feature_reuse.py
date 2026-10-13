"""Exact per-frame reuse across changed index spaces, without a model."""

import json

import numpy as np
import pytest

from carryrl.training.feature_store import (
    FrozenFeatureReuse,
    FrozenFeatureWriter,
    feature_cache_identity,
)

OPTIONS = dict(vision_sha256="a" * 64, preprocessing={"image_size": 28, "patch_size": 14},
               feature_shape=(4, 384), dtype="float16")


def identity(frames, used, **changes):
    return feature_cache_identity(frames, used, **(OPTIONS | changes))


def write(root, key, values):
    with FrozenFeatureWriter(root, key, max_bytes=values.nbytes) as writer:
        writer.write(key["indices"], values)
        return writer.publish()


@pytest.fixture
def source(tmp_path):
    frames = [np.full((3, 4, 3), index, np.uint8) for index in range(4)]
    used = [0, 2]  # Final-only frames deliberately have no cached features.
    values = np.stack([np.full((4, 384), value, np.float16) for value in (11, 22)])
    root = tmp_path / "cache"
    store = write(root, identity(frames, used), values)
    run = tmp_path / "run"
    run.mkdir()
    np.savez_compressed(run / "demonstration_frames.npz", frames=np.stack(frames))
    manifest = {"feature_cache_dir": str(store.directory.resolve()),
                "feature_cache_sha256": store.manifest["sha256"],
                "feature_cache_created_utc": store.manifest["created_utc"],
                "bytes": values.nbytes, "frames": 2, "dtype": "float16",
                "vision_sha256": OPTIONS["vision_sha256"]}
    (run / "feature_manifest.json").write_text(json.dumps(manifest))
    return run, root, frames, store


def prepare(source, frames, used=None, **changes):
    run, root, _, _ = source
    used = list(range(len(frames))) if used is None else used
    return FrozenFeatureReuse.prepare(run, frames, used, identity(frames, used, **changes),
                                      cache_root=root, max_bytes=1024 * 1024)


def test_remapping_duplicates_exact_values_detached_and_composite_provenance(source):
    _, root, old, store = source
    frames = [old[2], old[0], old[2].copy(), old[1], old[3]]
    plan = prepare(source, frames, [0, 1, 2, 3])
    assert plan.reused_count == 3 and plan.missing_indices == [3]
    assert plan.shard_identity == identity(frames, [3])
    assert len(plan) == 4 and plan.logical_bytes == 4 * 4 * 384 * 2
    with pytest.raises(ValueError, match="not been bound"):
        plan[3]
    with pytest.raises(ValueError, match="before binding"):
        _ = plan.manifest
    shard = write(root, plan.shard_identity, np.full((1, 4, 384), 33, np.float16))
    assert plan.bind_shard(shard) is plan
    for index, value in enumerate([22, 11, 22, 33]):
        actual = plan[index]
        assert not actual.requires_grad
        np.testing.assert_array_equal(actual.numpy(), np.full((4, 384), value, np.float16))
        actual.fill_(0)
        assert plan[index][0, 0].item() == value
    with pytest.raises(KeyError):
        plan[4]
    manifest = plan.manifest
    assert manifest["source_pack"]["sha256"] == store.manifest["sha256"]
    assert manifest["shard_pack"]["created_utc"] == shard.manifest["created_utc"]
    assert manifest["mapping"] == [[0, "source", 2], [1, "source", 0],
                                    [2, "source", 2], [3, "shard", 3]]
    assert manifest["reused_bytes"] + manifest["new_bytes"] == plan.logical_bytes
    assert len(manifest["mapping_sha256"]) == 64
    assert len(list(root.iterdir())) == 2


def test_all_reused_requires_no_shard(source):
    _, root, old, _ = source
    plan = prepare(source, [old[2], old[0]])
    assert plan.missing_indices == [] and plan.shard_identity is None
    assert plan.manifest["shard_pack"] is None and len(list(root.iterdir())) == 1
    assert plan[0][0, 0].item() == 22


def test_all_unmatched_and_final_only_frames_are_encoded(source):
    _, root, old, _ = source
    frames = [old[1], old[3], old[0].reshape(4, 3, 3)]
    plan = prepare(source, frames)
    assert plan.reused_count == 0 and plan.missing_indices == [0, 1, 2]
    plan.bind_shard(write(root, plan.shard_identity, np.full((3, 4, 384), 44, np.float16)))
    assert plan[2][0, 0].item() == 44


def test_variable_shape_archive_streams_old_used_rows(source):
    run, _, old, _ = source
    np.savez_compressed(run / "demonstration_frames.npz",
                        **{f"frame_{index:06}": frame for index, frame in enumerate(old)})
    plan = prepare(source, [old[2], old[0]])
    assert plan.reused_count == 2


@pytest.mark.parametrize("change", [dict(vision_sha256="b" * 64), dict(dtype="float32"),
    dict(preprocessing={"image_size": 56}), dict(feature_shape=(9, 384))])
def test_incompatible_encoder_is_rejected(source, change):
    with pytest.raises(ValueError, match="incompatible"):
        prepare(source, [source[2][0]], **change)


@pytest.mark.parametrize("damage", ["rgb", "truncated_rgb", "missing_rgb", "pack", "manifest",
                                    "path", "missing_pack"])
def test_corruption_and_provenance_mismatch_fail_closed(source, damage):
    run, _, old, store = source
    if damage == "rgb":
        damaged = np.stack(old)
        damaged[2, 0, 0, 0] += 1
        np.savez_compressed(run / "demonstration_frames.npz", frames=damaged)
    elif damage == "truncated_rgb":
        (run / "demonstration_frames.npz").write_bytes(b"broken")
    elif damage == "missing_rgb":
        (run / "demonstration_frames.npz").unlink()
    elif damage == "pack":
        with (store.directory / "features.npy").open("r+b") as output:
            output.seek(-1, 2)
            output.write(b"x")
    elif damage == "missing_pack":
        (store.directory / "features.npy").unlink()
    else:
        path = run / "feature_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["feature_cache_sha256" if damage == "manifest" else "feature_cache_dir"] = "bad"
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        prepare(source, [old[0]])


def test_budget_checks_current_and_old_logical_sizes(source):
    run, root, old, store = source
    with pytest.raises(MemoryError):
        FrozenFeatureReuse.prepare(run, [old[0]] * 3, [0, 1, 2], identity([old[0]] * 3, [0, 1, 2]),
                                   cache_root=root, max_bytes=store.manifest["bytes"])
    with pytest.raises(MemoryError):
        FrozenFeatureReuse.prepare(run, [old[0]], [0], identity([old[0]], [0]),
                                   cache_root=root, max_bytes=store.manifest["bytes"] - 1)


def test_current_identity_and_used_mismatch_are_rejected(source):
    run, root, old, _ = source
    key = identity(old, [0, 2])
    with pytest.raises(ValueError, match="indices differ"):
        FrozenFeatureReuse.prepare(run, old, [0], key, cache_root=root, max_bytes=100000)
    key["ordered_rgb_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="current RGB identity"):
        FrozenFeatureReuse.prepare(run, old, [0, 2], key, cache_root=root, max_bytes=100000)


def test_bad_or_missing_shard_fails_closed(source):
    _, root, old, store = source
    plan = prepare(source, [old[0], old[1]])
    with pytest.raises(ValueError, match="identity"):
        plan.bind_shard(store)
    shard = write(root, plan.shard_identity, np.full((1, 4, 384), 33, np.float16))
    (shard.directory / "features.npy").unlink()
    with pytest.raises(ValueError):
        plan.bind_shard(shard)
