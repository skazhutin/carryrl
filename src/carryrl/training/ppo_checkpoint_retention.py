"""Reversible retention for snapshots completed by one active PPO process."""

import json
from pathlib import Path

from carryrl.training.checkpoint_storage import (
    archive_cold_checkpoint,
    checkpoint_path,
    expected_hash,
    run_lock,
)


def retain_completed_checkpoints(output: Path, owned: list[Path], *, keep_raw: int,
                                 protected: tuple[Path, ...] = (),
                                 min_free_bytes: int = 2 * 1024**3) -> list[dict]:
    """Archive only this process's older snapshots; retain initial and latest raw.

    The general housekeeper protects whole active runs. Here ownership is explicit:
    candidates must be immutable completed snapshots supplied by the saving loop,
    inside its direct run directory. Storage still verifies the entire original
    SHA256, publishes recoverable chunks and a receipt, then removes the raw file.
    """
    if not keep_raw:
        return []
    if type(keep_raw) is not int or keep_raw < 2:
        raise ValueError("checkpoint_keep_raw must be zero or at least two")
    output = Path(output).absolute()
    if output.parent.name != "runs":
        raise ValueError("checkpoint retention requires a direct project runs directory")
    root = output.parent.parent
    snapshots = list(dict.fromkeys(Path(path).absolute() for path in owned))
    for path in snapshots:
        checked, run = checkpoint_path(root, path)
        if checked.parent != output / "checkpoints" or run != output:
            raise ValueError("checkpoint retention refuses snapshots outside its owned run")
        expected_hash(path)
    if len(snapshots) <= keep_raw:
        return []
    # The first state anchors baseline comparisons and remains immediately loadable.
    protected_paths = {snapshots[0], *(Path(path).resolve() for path in protected)}
    protected_paths.update(snapshots[-keep_raw:])
    candidates = [path for path in snapshots[:-keep_raw] if path not in protected_paths]
    results = []
    with run_lock(output):
        latest = Path(json.loads((output / "latest.json").read_text())["checkpoint"]).resolve()
        if latest != snapshots[-1].resolve():
            raise ValueError("checkpoint retention latest pointer changed outside saving loop")
        latest_rollout = json.loads((latest / "manifest.json").read_text())["rollout"]
        for candidate in candidates:
            if not (candidate / "state.pt").exists():
                continue

            def protection():
                pins = {path: {"initial, recent or source checkpoint"} for path in protected_paths}
                current = Path(json.loads((output / "latest.json").read_text())["checkpoint"])
                pins[current.resolve()] = {"latest checkpoint"}
                manifest = json.loads((candidate / "manifest.json").read_text())
                if manifest["rollout"] >= latest_rollout:
                    pins[candidate] = {"snapshot is not older than latest completed state"}
                return pins

            results.append(archive_cold_checkpoint(
                root, candidate, apply=True, protection_fn=protection,
                min_free_bytes=min_free_bytes))
    return results
