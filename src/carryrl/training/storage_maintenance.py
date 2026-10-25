"""Bounded, opt-in storage housekeeping without importing training runtimes."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

from carryrl.training.resource_storage import compact_checkpoint_tokenizers

POLICY = ".cache/storage_policy.json"
STATUS = ".cache/storage_maintenance.json"
LOCK = ".cache/storage_maintenance.lock"
LOG = ".cache/storage_maintenance.log"
DEFAULT_COOLDOWN = 6 * 3600


def _json(path: Path, limit: int = 65536):
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError(f"unsafe or oversized JSON: {path}")
    return json.loads(path.read_text())


def _inside(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("policy paths must be nonempty strings")
    candidate = root / value
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root / "runs") or candidate.is_symlink():
        raise ValueError("policy references must remain inside project runs")
    return resolved


def load_policy(root: Path) -> dict:
    """Reject missing/ambiguous protection settings before any maintenance writes."""
    root = root.resolve(strict=True)
    project = root / "pyproject.toml"
    if project.stat().st_size > 65536:
        raise ValueError("oversized project configuration")
    if tomllib.loads(project.read_text()).get("project", {}).get("name") != "carryrl":
        raise ValueError("not a carryrl project")
    for name in ("src/carryrl", "runs", ".cache"):
        path = root / name
        if not path.is_dir() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"unsafe project directory: {name}")
    for name in (STATUS, LOCK, LOG, STATUS + ".tmp"):
        if (root / name).is_symlink():
            raise ValueError(f"maintenance state must not be a symlink: {name}")
    policy = _json(root / POLICY)
    allowed = {"version", "enabled", "hot_manifests", "baseline_directories",
               "cooldown_seconds", "timeout_seconds", "max_checkpoints", "disk_floor_gib"}
    if not isinstance(policy, dict) or set(policy) - allowed:
        raise ValueError("unknown policy fields")
    if type(policy.get("version")) is not int or policy["version"] != 1:
        raise ValueError("unsupported policy version")
    if policy.get("enabled") is not True:
        raise ValueError("maintenance is not explicitly enabled")
    for key in ("hot_manifests", "baseline_directories"):
        values = policy.get(key)
        if not isinstance(values, list) or not values or len(values) > 64:
            raise ValueError(f"explicit nonempty {key} required")
        policy[key] = [str(_inside(root, value)) for value in values]
    if any(not Path(path).is_file() for path in policy["hot_manifests"]):
        raise ValueError("hot manifest is not a file")
    if any(not Path(path).is_dir() for path in policy["baseline_directories"]):
        raise ValueError("baseline is not a directory")
    for key, default, lower, upper in (
        ("cooldown_seconds", DEFAULT_COOLDOWN, 60, 30 * 86400),
        ("timeout_seconds", 300, 1, 300),
        ("max_checkpoints", 8, 1, 8),
        ("disk_floor_gib", 2, 2, 1024),
    ):
        value = policy.get(key, default)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not lower <= value <= upper):
            raise ValueError(f"invalid {key}")
        if key != "disk_floor_gib" and type(value) is not int:
            raise ValueError(f"{key} must be an integer")
        policy[key] = value
    return policy


def _recent(root: Path, policy: dict, now: float) -> bool:
    try:
        status = _json(root / STATUS)
        started = status["started_at"]
        if type(started) not in (int, float) or not math.isfinite(started):
            return True
        return now - started < policy["cooldown_seconds"]
    except FileNotFoundError:
        return False
    except (OSError, ValueError, KeyError, TypeError):
        # Corrupt coordination state needs inspection; do not bypass the cooldown.
        return True


def maybe_launch(root: Path | str) -> bool:
    """Start an independent worker; failures never block tests or the dashboard."""
    try:
        root = Path(root).resolve(strict=True)
        policy = load_policy(root)
        if _recent(root, policy, time.time()):
            return False
        subprocess.Popen(
            [sys.executable, "-m", "carryrl.training.storage_maintenance", "--root", str(root)],
            cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True,
        )
        return True
    except (OSError, ValueError, TypeError):
        return False


def _write_status(root: Path, status: dict) -> None:
    target = root / STATUS
    temporary = target.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(status, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(target)


def feature_inventory(root: Path) -> dict:
    """Re-read live feature references. Retain every pack, including uncertain ones.

    A feature manifest alone cannot prove absence of references in arbitrary run
    metadata. Inventory is deliberately non-destructive until that proof exists.
    """
    cache = root / ".cache/frozen_features"
    packs = {}
    if cache.is_dir() and not cache.is_symlink():
        for entry in cache.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                packs[str(entry.resolve())] = {"directory": str(entry), "references": []}
    errors = []
    consumed = 0
    examined = 0
    for manifest in (root / "runs").rglob("feature_manifest.json"):
        examined += 1
        try:
            if examined > 20000:
                raise ValueError("feature inventory file budget exceeded")
            if not manifest.resolve().is_relative_to(root / "runs"):
                raise ValueError("feature reference escapes project runs")
            consumed += manifest.stat().st_size
            if consumed > 64 * 1024**2:
                raise ValueError("feature inventory byte budget exceeded")
            pending = [_json(manifest, 16 * 1024**2)]
            while pending:
                value = pending.pop()
                if isinstance(value, dict):
                    pending.extend(value.values())
                elif isinstance(value, list):
                    pending.extend(value)
                elif isinstance(value, str):
                    # Exact directory strings are how immutable packs are recorded.
                    if value in packs:
                        refs = packs[value]["references"]
                        relative = str(manifest.relative_to(root))
                        if relative not in refs:
                            refs.append(relative)
        except (OSError, ValueError, TypeError) as error:
            errors.append({"path": str(manifest.relative_to(root)), "error": str(error)})
            if consumed > 64 * 1024**2 or examined > 20000:
                break
    return {"mode": "inventory_only", "all_packs_retained": True,
            "orphan_proof": False, "feature_manifests_examined": examined,
            "packs": list(packs.values()), "errors": errors}


def _command(root: Path, policy: dict) -> list[str]:
    command = [sys.executable, "-m", "carryrl.training.checkpoint_storage",
               "--root", str(root), "housekeep"]
    for path in policy["hot_manifests"]:
        command.extend(["--hot-reference", path])
    for path in policy["baseline_directories"]:
        command.extend(["--keep", path])
    command.extend(["--min-free-bytes", str(int(policy["disk_floor_gib"] * 1024**3)),
                    "--max-checkpoints", str(policy["max_checkpoints"]),
                    "--max-bytes", str(8 * 1024**3), "--min-age-seconds", "3600", "--apply"])
    return command


def _run_checkpoint_worker(root: Path, policy: dict) -> dict:
    # TemporaryFile prevents child output from growing Python memory or filling a pipe.
    with tempfile.TemporaryFile() as output:
        child = subprocess.Popen(
            _command(root, policy), cwd=root, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
        )
        timed_out = False
        try:
            child.wait(timeout=policy["timeout_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        output.seek(0)
        diagnostic = output.read(1024 * 1024).decode("utf-8", errors="replace")
    (root / LOG).write_text(diagnostic)
    return {"returncode": child.returncode, "timed_out": timed_out, "log": LOG}


def run_maintenance(root: Path | str) -> dict:
    root = Path(root).resolve(strict=True)
    policy = load_policy(root)
    with (root / LOCK).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "locked"}
        started = time.time()
        if _recent(root, policy, started):
            return {"status": "cooldown"}
        status = {"status": "running", "started_at": started}
        _write_status(root, status)
        try:
            status["resources"] = compact_checkpoint_tokenizers(
                root, apply=True, baseline_directories=tuple(policy["baseline_directories"]),
                timeout_seconds=min(60, policy["timeout_seconds"]),
            )
            if status["resources"]["status"] == "failed":
                raise RuntimeError("checkpoint resource compaction failed")
            status["features"] = feature_inventory(root)
            status["checkpoints"] = _run_checkpoint_worker(root, policy)
            result = status["checkpoints"]
            status["status"] = "complete" if result["returncode"] == 0 else "failed"
            if result["timed_out"]:
                status["status"] = "timed_out"
        except Exception as error:
            status.update(status="failed", error=str(error))
        status["finished_at"] = time.time()
        _write_status(root, status)
        return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    options = parser.parse_args()
    try:
        status = run_maintenance(options.root)
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}))
        return 1
    print(json.dumps(status))
    return int(status["status"] in {"failed", "timed_out"})


if __name__ == "__main__":
    raise SystemExit(main())
