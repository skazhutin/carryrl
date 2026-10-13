#!/usr/bin/env python3
"""Inventory and reversibly archive checkpoint payloads; mutations require --apply."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import json
import os
import re
import shlex
import stat
import tempfile
import tomllib
from pathlib import Path

CHUNK_BYTES = 1024 * 1024
FORMAT = "carryrl-checkpoint-storage-v1"


def digest(stream):
    result = hashlib.sha256()
    size = 0
    while chunk := stream.read(CHUNK_BYTES):
        result.update(chunk)
        size += len(chunk)
    return result.hexdigest(), size


def file_digest(path):
    with path.open("rb") as stream:
        return digest(stream)


def read_json(path):
    return json.loads(path.read_text())


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    fd, name = tempfile.mkstemp(prefix=".storage-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def path_strings(data):
    for value in strings(data):
        try:
            words = shlex.split(value)
        except ValueError:
            words = [value]
        for word in words:
            if "runs/" in word and len(word) < 4096:
                yield word


def checkpoint_path(root, checkpoint):
    root = root.resolve()
    path = Path(os.path.abspath(checkpoint))
    runs = root / "runs"
    if not path.is_relative_to(runs) or len(path.relative_to(runs).parts) < 3:
        raise ValueError("checkpoint must be a directory inside this project's runs")
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("symlink checkpoints or ancestors are protected")
    return path, runs / path.relative_to(runs).parts[0]


def protection(root):
    """Conservatively retain active runs, configured inputs, pins and latest weights."""
    root = root.resolve()
    reasons = {}

    def protect(path, reason):
        if path.name in {"state.pt", "state.pt.gz"}:
            path = path.parent
        reasons.setdefault(path.resolve(), set()).add(reason)

    for base in (root / "configs", root / ".cache", root / "runs"):
        if not base.exists():
            continue
        for source in base.rglob("*"):
            if not source.is_file() or source.suffix not in {".json", ".toml", ".yaml", ".yml"}:
                continue
            if (
                source.is_symlink()
                or source.name
                in {
                    "storage_manifest.json",
                    "manifest.json",
                    "run_storage_inventory.json",
                    "run_storage_archive_report.json",
                }
                or "frozen_features" in source.parts
            ):
                continue
            if source.stat().st_size > 4 * CHUNK_BYTES:
                if base.name == "configs" or "pin" in source.name:
                    protect(root / "runs", f"unreadable reference source {source}")
                elif source.name in {"metadata.json", "status.json", "runtime_status.json"}:
                    protect(source.parent, f"unreadable run metadata {source}")
                continue
            try:
                if source.suffix == ".json":
                    data = read_json(source)
                elif source.suffix == ".toml":
                    data = tomllib.loads(source.read_text())
                else:
                    data = re.findall(r"[^\s\"\']*runs/[^\s\"\']+", source.read_text())
            except (OSError, ValueError):
                if (
                    base.name == "configs"
                    or "pin" in source.name
                    or source.name.startswith("active_")
                ):
                    protect(root / "runs", f"invalid reference source {source}")
                elif source.name in {"metadata.json", "status.json", "runtime_status.json"}:
                    protect(source.parent, f"invalid run metadata {source}")
                continue
            active_registry = source.name in {
                "active_mlx_launch.json",
                "active_learning_launch.json",
            }
            for value in path_strings(data):
                if "\n" in value or not value or len(value) > 4096 or "runs/" not in value:
                    continue
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = root / candidate
                if not candidate.exists() or not candidate.resolve().is_relative_to(root / "runs"):
                    continue
                # latest.json is handled separately: smoke checkpoints can all become cold.
                if source.name == "latest.json":
                    continue
                if (
                    active_registry
                    or "checkpoints" in candidate.parts
                    or "pin" in source.name
                    or base.name == "configs"
                ):
                    protect(candidate, f"referenced by {source.relative_to(root)}")
            if source.name in {"status.json", "progress.json", "runtime_status.json"}:
                if isinstance(data, dict) and data.get("status") in {
                    "running",
                    "training",
                    "active",
                }:
                    protect(source.parent, "run reports active training")
    for run in (root / "runs").iterdir():
        if not run.is_dir():
            continue
        if any(
            word in run.name for word in ("fresh_corrections12", "visual_bc_control", "paired40")
        ):
            protect(run, "protected current training/source family")
        if "smoke" not in run.name:
            states = sorted(run.rglob("state.pt"))
            if states:
                protect(states[-1].parent, "latest checkpoint retained for non-smoke run")
            latest = run / "latest.json"
            if latest.exists():
                try:
                    for value in strings(read_json(latest)):
                        candidate = Path(value)
                        if not candidate.is_absolute():
                            candidate = root / candidate
                        protect(candidate, "latest checkpoint retained for non-smoke run")
                except (OSError, ValueError):
                    protect(run, "unreadable latest checkpoint pointer")
    return reasons


def protection_reasons(path, reasons):
    return sorted(
        {
            reason
            for protected, labels in reasons.items()
            if path == protected or path.is_relative_to(protected)
            for reason in labels
        }
    )


def restore_command(root, checkpoint):
    return (
        f"python {shlex.quote(str(root / 'scripts/manage_run_storage.py'))} "
        f"--root {shlex.quote(str(root))} restore "
        f"--checkpoint {shlex.quote(str(checkpoint))} --apply"
    )


def inventory(root):
    root = root.resolve()
    reasons = protection(root)
    rows = []
    for name in ("state.pt", "state.pt.gz"):
        for state in sorted((root / "runs").rglob(name)):
            try:
                path, run = checkpoint_path(root, state.parent)
                blocked = protection_reasons(path, reasons)
                if state.is_symlink():
                    blocked.append("symlink payload")
            except ValueError as error:
                blocked = [str(error)]
                run = root / "runs" / state.relative_to(root / "runs").parts[0]
            rows.append(
                {
                    "checkpoint": str(state.parent),
                    "run": run.name,
                    "state": "archived" if name.endswith("gz") else "hot",
                    "bytes": state.stat().st_size,
                    "allocated_bytes": state.stat().st_blocks * 512,
                    "protected": blocked,
                    "candidate": not blocked and name == "state.pt",
                    "restore_command": restore_command(root, state.parent),
                }
            )
    return {
        "format": FORMAT,
        "checkpoints": rows,
        "candidate_bytes": sum(row["bytes"] for row in rows if row["candidate"]),
        "candidate_count": sum(row["candidate"] for row in rows),
    }


@contextlib.contextmanager
def run_lock(run):
    # This persistent lock inode must not be removed: concurrent callers share it.
    with (run / ".storage.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def expected_hash(checkpoint):
    value = read_json(checkpoint / "manifest.json")["sha256"]
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("missing valid original checkpoint sha256")
    return value


def record(root, run, checkpoint, state, original_bytes, compressed_bytes, sha256):
    destination = run / "storage_manifest.json"
    data = read_json(destination) if destination.exists() else {"format": FORMAT, "checkpoints": {}}
    if data.get("format") != FORMAT:
        raise ValueError("unrecognized storage manifest")
    data["checkpoints"][str(checkpoint.relative_to(run))] = {
        "state": state,
        "sha256": sha256,
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "payload": "state.pt.gz" if state == "archived" else "state.pt",
        "requires_restore": state == "archived",
        "restore_command": restore_command(root, checkpoint),
    }
    atomic_json(destination, data)


def compact_staging(path):
    """Pre-size staging to avoid APFS speculative extent allocation on growing writes."""
    before = path.stat()
    fd, name = tempfile.mkstemp(prefix=".storage-compact-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output, path.open("rb") as incoming:
            output.truncate(before.st_size)
            while chunk := incoming.read(CHUNK_BYTES):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if file_digest(temporary) != file_digest(path):
            raise ValueError("staging compaction hash mismatch")
        os.chmod(temporary, stat.S_IMODE(before.st_mode))
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def verify_existing_pair(plain, packed, sha256):
    """Only identical verified payloads qualify as an interrupted publication."""
    before = (plain.stat(), packed.stat())
    try:
        original_hash, original_bytes = file_digest(plain)
        with gzip.open(packed, "rb") as stream:
            packed_hash, unpacked_bytes = digest(stream)
    except (OSError, EOFError) as error:
        raise ValueError("refusing to overwrite: existing payloads are not a valid pair") from error
    if original_hash != sha256 or packed_hash != sha256 or unpacked_bytes != original_bytes:
        raise ValueError(
            "refusing to overwrite: existing payloads have divergent checkpoint hashes"
        )
    after = (plain.stat(), packed.stat())
    for old, new in zip(before, after, strict=True):
        if (old.st_ino, old.st_size, old.st_mtime_ns) != (new.st_ino, new.st_size, new.st_mtime_ns):
            raise ValueError("existing payload changed during reconciliation")
    return original_bytes, packed.stat().st_size


def transform(root, checkpoint, *, restore=False, apply=False):
    root = root.resolve()
    checkpoint, run = checkpoint_path(root, checkpoint)
    plain, packed = checkpoint / "state.pt", checkpoint / "state.pt.gz"
    source, destination = (packed, plain) if restore else (plain, packed)
    if source.is_symlink() or destination.is_symlink():
        raise ValueError("symlink payload is protected")
    if not source.is_file():
        raise ValueError(f"missing source: {source}")
    if not restore:
        blocked = protection_reasons(checkpoint, protection(root))
        if blocked:
            raise ValueError("protected checkpoint: " + "; ".join(blocked))
    sha256 = expected_hash(checkpoint)
    result = {
        "checkpoint": str(checkpoint),
        "action": "restore" if restore else "archive",
        "applied": False,
        "bytes": source.stat().st_size,
        "restore_command": restore_command(root, checkpoint),
    }
    if not apply:
        if destination.exists():
            verify_existing_pair(plain, packed, sha256)
            result["reconciliation_required"] = True
        return result
    with run_lock(run):
        if not restore and protection_reasons(checkpoint, protection(root)):
            raise ValueError("checkpoint became protected")
        before = source.stat()
        if destination.exists():
            if source.is_symlink() or destination.is_symlink():
                raise ValueError("symlink payload is protected")
            original_bytes, compressed_bytes = verify_existing_pair(plain, packed, sha256)
            if not restore and protection_reasons(checkpoint, protection(root)):
                raise ValueError("checkpoint became protected during reconciliation")
            # Retry metadata publication before removing either verified payload.
            record(
                root,
                run,
                checkpoint,
                "hot" if restore else "archived",
                original_bytes,
                compressed_bytes,
                sha256,
            )
            source.unlink()
            sync_directory(checkpoint)
            result.update(
                applied=True,
                reconciled=True,
                original_bytes=original_bytes,
                compressed_bytes=compressed_bytes,
                saved_bytes=original_bytes - compressed_bytes,
                allocated_bytes_before=before.st_blocks * 512,
                allocated_bytes_after=destination.stat().st_blocks * 512,
                allocated_bytes_saved=(before.st_blocks - destination.stat().st_blocks) * 512,
            )
            return result
        fd, name = tempfile.mkstemp(prefix=".storage-", dir=checkpoint)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as output:
                if restore:
                    with gzip.open(source, "rb") as incoming:
                        while chunk := incoming.read(CHUNK_BYTES):
                            output.write(chunk)
                else:
                    with source.open("rb") as incoming:
                        with gzip.GzipFile(
                            filename="", mode="wb", fileobj=output, compresslevel=6, mtime=0
                        ) as outgoing:
                            while chunk := incoming.read(CHUNK_BYTES):
                                outgoing.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            compact_staging(temporary)
            if restore:
                actual, original_bytes = file_digest(temporary)
                compressed_bytes = before.st_size
            else:
                with gzip.open(temporary, "rb") as stream:
                    actual, original_bytes = digest(stream)
                compressed_bytes = temporary.stat().st_size
                if file_digest(source) != (sha256, original_bytes):
                    raise ValueError("original checkpoint hash mismatch")
            if actual != sha256:
                raise ValueError("checkpoint hash mismatch")
            after = source.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("source changed during operation")
            if not restore and protection_reasons(checkpoint, protection(root)):
                raise ValueError("checkpoint became protected during compression")
            # Exclusive hard-link publication is atomic and never overwrites a payload.
            os.chmod(temporary, stat.S_IMODE(before.st_mode))
            os.link(temporary, destination)
            sync_directory(checkpoint)
            record(
                root,
                run,
                checkpoint,
                "hot" if restore else "archived",
                original_bytes,
                compressed_bytes,
                sha256,
            )
            source.unlink()
            sync_directory(checkpoint)
            result.update(
                applied=True,
                original_bytes=original_bytes,
                compressed_bytes=compressed_bytes,
                saved_bytes=original_bytes - compressed_bytes,
                allocated_bytes_before=before.st_blocks * 512,
                allocated_bytes_after=destination.stat().st_blocks * 512,
                allocated_bytes_saved=(before.st_blocks - destination.stat().st_blocks) * 512,
            )
        finally:
            temporary.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("command", choices=("inventory", "plan", "archive", "restore"))
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--apply", action="store_true", help="perform verified reversible mutation")
    args = parser.parse_args()
    try:
        if args.command in {"inventory", "plan"}:
            result = inventory(args.root)
            if args.command == "plan":
                result["checkpoints"] = [row for row in result["checkpoints"] if row["candidate"]]
        else:
            if not args.checkpoint:
                parser.error("archive/restore requires explicit --checkpoint selections")
            result = [
                transform(
                    args.root, checkpoint, restore=args.command == "restore", apply=args.apply
                )
                for checkpoint in args.checkpoint
            ]
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"storage operation refused: {error}\n")


if __name__ == "__main__":
    main()
