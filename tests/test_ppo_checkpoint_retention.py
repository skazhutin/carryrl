"""The active PPO loop archives only its own completed older snapshots."""

import hashlib
import json

import pytest

from carryrl.training.checkpoint_storage import restore_chunked_checkpoint
from carryrl.training.ppo_checkpoint_retention import retain_completed_checkpoints


def snapshots(root, name="stage", count=5):
    output = root / "runs" / name
    paths, payloads = [], {}
    for rollout in range(count):
        path = output / "checkpoints" / f"rollout_{rollout:04d}"
        path.mkdir(parents=True)
        payload = (str(rollout) * 2048).encode()
        (path / "state.pt").write_bytes(payload)
        (path / "manifest.json").write_text(json.dumps({
            "sha256": hashlib.sha256(payload).hexdigest(), "rollout": rollout}))
        paths.append(path)
        payloads[path] = payload
    (output / "latest.json").write_text(json.dumps({"checkpoint": str(paths[-1])}))
    return output, paths, payloads


def test_retention_keeps_initial_latest_two_and_external_parent_and_restores_exact_bytes(tmp_path):
    output, paths, payloads = snapshots(tmp_path)
    _, parent, _ = snapshots(tmp_path, "parent", count=1)
    manifests = {path: (path / "manifest.json").read_bytes() for path in paths}
    results = retain_completed_checkpoints(output, paths, keep_raw=2,
                                          protected=tuple(parent), min_free_bytes=0)
    assert len(results) == 2 and all(row["applied"] for row in results)
    assert [path for path in paths if (path / "state.pt").exists()] == [paths[0], *paths[-2:]]
    assert (parent[0] / "state.pt").exists()
    receipt = json.loads((output / "storage_manifest.json").read_text())
    for path in paths[1:3]:
        row = receipt["checkpoints"][str(path.relative_to(output))]
        assert row["requires_restore"] is True
        assert row["restore_command"].endswith("--apply")
        restored = restore_chunked_checkpoint(tmp_path, path, apply=True)
        assert restored["applied"] is True
        assert (path / "state.pt").read_bytes() == payloads[path]
    assert all((path / "manifest.json").read_bytes() == manifests[path] for path in paths)


def test_retention_rejects_foreign_snapshots_and_changed_latest_pointer(tmp_path):
    output, paths, _ = snapshots(tmp_path)
    _, foreign, _ = snapshots(tmp_path, "foreign", count=1)
    with pytest.raises(ValueError, match="outside its owned run"):
        retain_completed_checkpoints(output, paths + foreign, keep_raw=2, min_free_bytes=0)
    (output / "latest.json").write_text(json.dumps({"checkpoint": str(paths[0])}))
    with pytest.raises(ValueError, match="latest pointer changed"):
        retain_completed_checkpoints(output, paths, keep_raw=2, min_free_bytes=0)
    assert all((path / "state.pt").exists() for path in paths)


def test_retention_corrupt_payload_fails_without_removing_raw_state(tmp_path):
    output, paths, _ = snapshots(tmp_path)
    (paths[1] / "state.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="original checkpoint hash mismatch"):
        retain_completed_checkpoints(output, paths, keep_raw=2, min_free_bytes=0)
    assert all((path / "state.pt").exists() for path in paths)
    assert not (output / "storage_manifest.json").exists()
