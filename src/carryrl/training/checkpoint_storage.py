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
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import tomllib
from pathlib import Path

from carryrl.training import checkpoint_chunks as chunks

CHUNK_BYTES = 1024 * 1024
FORMAT = "carryrl-checkpoint-storage-v1"
HOUSEKEEP_MIN_FREE_BYTES = 2 * 1024**3
_RUN_LOCKS = threading.local()


def digest(stream, *, max_bytes=None):
    result = hashlib.sha256()
    size = 0
    while chunk := stream.read(CHUNK_BYTES):
        result.update(chunk)
        size += len(chunk)
        if max_bytes is not None and size > max_bytes:
            raise ValueError("checkpoint exceeds bounded size limit")
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


class HousekeepingPins:
    """Only current dependencies are hot; historical references survive in cold storage.

    Explicit current manifests are mandatory. Unlike legacy archive protection,
    these manifests are read even when large and their manifest dependencies are
    followed. Parsed files are cached, but their identity and critical process/run
    pins are checked again before each destructive publication.
    """

    def __init__(self, root, hot_references, keep=()):
        self.root = root.resolve()
        if not hot_references:
            raise ValueError("housekeep requires at least one explicit --hot-reference")
        self.hot_references = [self.resolve_source(path) for path in hot_references]
        self.keep = [self.resolve_source(path) for path in keep]
        self.cache = {}

    def resolve_source(self, path):
        return (self.root / path).resolve() if not path.is_absolute() else path.resolve()

    def read(self, source):
        info = source.stat()
        identity = (info.st_ino, info.st_size, info.st_mtime_ns)
        if info.st_size > 128 * CHUNK_BYTES or source.is_symlink():
            raise ValueError(f"unsafe or oversized active reference {source}")
        cached = self.cache.get(source)
        if cached and cached[0] == identity:
            return cached[1]
        if source.suffix == ".json":
            data = read_json(source)
        elif source.suffix == ".toml":
            data = tomllib.loads(source.read_text())
        else:
            data = re.findall(r"[^\s\"\']*runs/[^\s\"\']+", source.read_text())
        after = source.stat()
        if (after.st_ino, after.st_size, after.st_mtime_ns) != identity:
            raise ValueError(f"active reference changed while reading {source}")
        self.cache[source] = (identity, data)
        return data

    def paths(self, data, source):
        """Resolve repository and source-relative paths, including missing hot payloads."""
        runs = self.root / "runs"
        for value in strings(data):
            if not value or len(value) > 4096 or "\n" in value:
                continue
            # Digests and labels cannot be supported paths. Avoid shell parsing
            # tens of thousands of these strings in prepared corpus manifests.
            if "/" not in value and "." not in value:
                continue
            try:
                values = [value, *shlex.split(value)]
            except ValueError:
                values = [value]
            for word in values:
                if word.startswith("--") and "=" in word:
                    word = word.split("=", 1)[1]
                path = Path(word)
                if (not path.is_absolute() and "/" not in word
                        and path.suffix not in {".json", ".pt", ".gz", ".toml", ".yaml", ".yml"}):
                    continue
                options = [path] if path.is_absolute() else [self.root / path, source.parent / path]
                for option in options:
                    candidate = option.resolve()
                    if not candidate.is_relative_to(runs):
                        continue
                    # A compressed checkpoint may leave state.pt absent. Pin its directory.
                    if candidate.name in {"state.pt", "state.pt.gz"}:
                        candidate = candidate.parent
                    if candidate.exists():
                        yield candidate

    def __call__(self):
        root, reasons = self.root, {}

        def protect(path, reason):
            # A dependency can refer to any checkpoint sidecar, not only state.pt.
            if "checkpoints" in path.parts:
                index = path.parts.index("checkpoints")
                if len(path.parts) > index + 1:
                    path = Path(*path.parts[: index + 2])
            reasons.setdefault(path.resolve(), set()).add(reason)

        for path in self.keep:
            if not path.exists():
                protect(root / "runs", f"missing explicit hot path {path}")
            else:
                protect(path, "explicit hot dependency")
        queue = list(self.hot_references)
        for base in (root / "configs", root / ".cache"):
            if base.exists():
                queue.extend(
                    source for source in base.rglob("*")
                    if (source.is_file() or source.is_symlink())
                    and source.suffix in {".json", ".toml", ".yaml", ".yml"}
                    and (base.name == "configs" or "pin" in source.name
                         or source.name.startswith("active_"))
                )
        # Run status is a current signal; old launch/provenance/config files are not.
        for run in (root / "runs").iterdir():
            if not run.is_dir():
                continue
            if "visual_bc_control" in run.name:
                protect(run, "accepted baseline retained hot")
            sources = []
            for directory, folders, files in os.walk(run):
                folders[:] = [name for name in folders if name not in {
                    "checkpoints", "frozen_features", "playbacks", "evaluation_playbacks",
                }]
                sources.extend(Path(directory) / name for name in files if name in {
                    "status.json", "progress.json", "runtime_status.json",
                })
            for source in sources:
                try:
                    data = self.read(source)
                    if not isinstance(data, dict):
                        raise ValueError("run status must be an object")
                    if data.get("status") in {"running", "training", "active"}:
                        protect(source.parent, "run reports active training")
                        queue.append(source)
                except (OSError, ValueError):
                    protect(source.parent, f"unreadable current run status {source}")
        seen = set()
        while queue:
            source = queue.pop()
            if source in seen:
                continue
            seen.add(source)
            if "checkpoints" in source.parts:
                protect(source, "explicit current checkpoint reference")
            if len(seen) > 4096:
                protect(root / "runs", "too many current reference dependencies")
                break
            try:
                data = self.read(source)
                if not isinstance(data, (dict, list)):
                    raise ValueError("active reference must contain an object or list")
                candidates = list(self.paths(data, source))
                if source.name.startswith("active_") and not candidates:
                    raise ValueError("active registry has no resolvable run dependency")
                for candidate in candidates:
                    protect(candidate, f"current reference {source}")
                    if (candidate.is_file()
                            and candidate.suffix in {".json", ".toml", ".yaml", ".yml"}
                            and "checkpoints" not in candidate.parts):
                        queue.append(candidate)
                    elif candidate.is_dir() and "checkpoints" not in candidate.parts:
                        manifest = candidate / "manifest.json"
                        if manifest.exists():
                            queue.append(manifest)
            except (OSError, ValueError):
                protect(root / "runs", f"unreadable active reference {source}")
        try:
            processes = subprocess.run(
                ["ps", "-axo", "pid=,command="], check=True, capture_output=True, text=True,
            ).stdout
            for line in processes.splitlines():
                pid, _, command = line.strip().partition(" ")
                if pid == str(os.getpid()):
                    continue
                try:
                    words = shlex.split(command)
                except ValueError:
                    continue
                # Maintenance's own explicit selections are not consumers of weights.
                if any(Path(word).name == "manage_run_storage.py"
                       or word == "carryrl.training.checkpoint_storage" for word in words):
                    continue
                paths = list(self.paths(words, root / "process.json"))
                training = any(
                    Path(word).name in {"train_imitation.py", "train_history_reward.py"}
                    for word in words
                )
                if training and not paths:
                    protect(root / "runs", "training process has ambiguous run path")
                for path in paths:
                    if training or "checkpoints" in path.parts:
                        protect(path, f"live checkpoint consumer {pid}")
        except (OSError, subprocess.SubprocessError):
            protect(root / "runs", "cannot inspect live training processes")
        return reasons


def housekeep(root, *, hot_references, keep=(), apply=False, max_checkpoints=8,
              max_bytes=8 * 1024**3, min_free_bytes=HOUSEKEEP_MIN_FREE_BYTES,
              min_age_seconds=3600):
    """Run one bounded pass; every cold payload remains hash-verifiable and restorable."""
    if max_checkpoints < 1 or max_bytes < 1 or min_free_bytes < HOUSEKEEP_MIN_FREE_BYTES:
        raise ValueError("positive batch limits and at least 2 GiB reserve are required")
    if min_age_seconds < 0:
        raise ValueError("minimum age cannot be negative")
    root = root.resolve()
    pins = HousekeepingPins(root, hot_references, keep)
    reasons = pins()
    selected_bytes, rows, selected = 0, [], 0
    # Small checkpoints can free enough space for larger ones near the reserve.
    states = {path.parent: path for path in (root / "runs").rglob("state.pt.gz")}
    states.update({path.parent: path for path in (root / "runs").rglob("state.pt")})
    states = sorted(states.values(),
                    key=lambda path: (path.stat().st_size, path.stat().st_mtime_ns))
    for state in states:
        checkpoint, _ = checkpoint_path(root, state.parent)
        blocked = protection_reasons(checkpoint, reasons)
        size = state.stat().st_size
        if blocked or time.time() - state.stat().st_mtime < min_age_seconds:
            continue
        if selected >= max_checkpoints or selected_bytes + size > max_bytes:
            continue
        selected += 1
        selected_bytes += size
        try:
            result = archive_cold_checkpoint(
                root, checkpoint, apply=apply, protection_fn=pins,
                min_free_bytes=min_free_bytes,
            )
        except (OSError, ValueError, KeyError) as error:
            result = {"checkpoint": str(checkpoint), "applied": False, "skipped": str(error)}
        rows.append(result)
    cleanup = {}
    if apply and (root / "runs" / ".checkpoint_store").exists():
        try:
            cleanup = chunks.cleanup_orphan_chunks(root)
        except (OSError, ValueError, KeyError) as error:
            cleanup = {"skipped": str(error)}
    return {
        "format": FORMAT, "mode": "housekeep", "applied": apply,
        "selected_bytes": selected_bytes, "selected_count": selected,
        "archived_count": sum(row.get("applied", False) for row in rows),
        "saved_bytes": sum(row.get("saved_bytes", 0) for row in rows if row.get("applied")),
        "min_free_bytes": min_free_bytes, "checkpoints": rows,
        "protected": {str(path): sorted(labels) for path, labels in reasons.items()},
        "chunk_cleanup": cleanup,
    }


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
    for name in ("state.pt", "state.pt.gz", chunks.DESCRIPTOR):
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
                    "state": ("chunked" if name == chunks.DESCRIPTOR else
                              "archived" if name.endswith("gz") else "hot"),
                    "bytes": state.stat().st_size,
                    "allocated_bytes": state.stat().st_blocks * 512,
                    "protected": blocked,
                    "candidate": not blocked and name == "state.pt",
                    "restore_command": restore_command(root, state.parent),
                }
            )
    chunk_files = list((root / "runs" / ".checkpoint_store").glob("*.gz"))
    return {
        "format": FORMAT,
        "checkpoints": rows,
        "chunk_store_bytes": sum(path.stat().st_size for path in chunk_files),
        "chunk_store_allocated_bytes": sum(path.stat().st_blocks * 512 for path in chunk_files),
        "candidate_bytes": sum(row["bytes"] for row in rows if row["candidate"]),
        "candidate_count": sum(row["candidate"] for row in rows),
    }


@contextlib.contextmanager
def run_lock(run, *, blocking=True):
    # This persistent lock inode must not be removed: concurrent callers share it.
    with (run / ".storage.lock").open("a") as stream:
        held = _RUN_LOCKS.__dict__.setdefault("held", set())
        info = os.fstat(stream.fileno())
        identity = (os.getpid(), info.st_dev, info.st_ino)
        # Sibling checkpoints share this lock. A second open/flock in the same
        # thread would wait for its own outer reader; other threads/processes
        # must still acquire the OS lock independently.
        if identity in held:
            yield
            return
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        held.add(identity)
        try:
            yield
        finally:
            held.remove(identity)


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
        "payload": (chunks.DESCRIPTOR if state == "chunked" else
                    "state.pt.gz" if state == "archived" else "state.pt"),
        "requires_restore": state != "hot",
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


def archive_cold_checkpoint(root, checkpoint, *, apply=False, protection_fn,
                            min_free_bytes=HOUSEKEEP_MIN_FREE_BYTES):
    """Replace verified plain/legacy gzip payloads with a recoverable chunk descriptor."""
    checkpoint, run = checkpoint_path(root, checkpoint)
    plain, packed = checkpoint / "state.pt", checkpoint / "state.pt.gz"
    sources = [path for path in (plain, packed) if path.exists() or path.is_symlink()]
    if not sources or any(path.is_symlink() for path in sources):
        raise ValueError("missing or symlink checkpoint payload")
    result = {"checkpoint": str(checkpoint), "action": "archive_chunks", "applied": False,
              "bytes": sum(path.stat().st_size for path in sources),
              "restore_command": restore_command(root, checkpoint)}

    def check_pins():
        if protection_reasons(checkpoint, protection_fn()):
            raise ValueError("checkpoint became protected during chunk archiving")

    check_pins()
    if not apply:
        return result
    with run_lock(run):
        check_pins()
        sha256 = expected_hash(checkpoint)
        before = {path: path.stat() for path in sources}
        if packed in sources:
            with gzip.open(packed, "rb") as stream:
                actual, original_bytes = digest(stream, max_bytes=chunks.MAX_BYTES)
            if actual != sha256:
                raise ValueError("legacy archive checkpoint hash mismatch")
        else:
            original_bytes = plain.stat().st_size
        if (shutil.disk_usage(checkpoint).free
                < min_free_bytes + 2 * original_bytes):
            result["skipped"] = "insufficient free disk for reserve and two staging copies"
            return result
        descriptor_path = checkpoint / chunks.DESCRIPTOR
        had_descriptor = descriptor_path.exists() or descriptor_path.is_symlink()
        store = root / "runs" / ".checkpoint_store"
        existing = set(store.glob("*.gz")) if store.exists() else set()
        temporary, published_record, deletion_started = None, False, False
        try:
            source = plain
            if not plain.exists():
                fd, name = tempfile.mkstemp(prefix=".storage-unpack-", dir=checkpoint)
                temporary = Path(name)
                with os.fdopen(fd, "wb") as output, gzip.open(packed, "rb") as incoming:
                    while chunk := incoming.read(CHUNK_BYTES):
                        if shutil.disk_usage(checkpoint).free < min_free_bytes + len(chunk):
                            raise OSError("insufficient free disk during legacy decompression")
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                source = temporary
            if file_digest(source) != (sha256, original_bytes):
                raise ValueError("original checkpoint hash mismatch")
            descriptor = chunks.archive_checkpoint(
                root, checkpoint, source, sha256, min_free_bytes, pre_publish=check_pins,
            )
            check_pins()
            for path, old in before.items():
                new = path.stat()
                if (old.st_ino, old.st_size, old.st_mtime_ns) != (
                        new.st_ino, new.st_size, new.st_mtime_ns):
                    raise ValueError("source changed during chunk archiving")
            if file_digest(source) != (sha256, original_bytes):
                raise ValueError("source checkpoint hash mismatch before removal")
            chunk_paths = {store / (item["sha256"] + ".gz") for item in descriptor["chunks"]}
            added_bytes = (sum(path.stat().st_size for path in chunk_paths - existing)
                           + descriptor_path.stat().st_size)
            record(root, run, checkpoint, "chunked", original_bytes, added_bytes, sha256)
            published_record = True
            check_pins()
            for path, old in before.items():
                new = path.stat()
                if (old.st_ino, old.st_size, old.st_mtime_ns) != (
                        new.st_ino, new.st_size, new.st_mtime_ns):
                    raise ValueError("source changed before chunk source removal")
            deletion_started = True
            for path in sources:
                path.unlink()
            sync_directory(checkpoint)
            result.update(applied=True, original_bytes=original_bytes,
                          compressed_bytes=added_bytes, saved_bytes=result["bytes"] - added_bytes)
        except BaseException:
            # A new active pin must leave exactly the original checkpoint file list.
            originals_intact = all(
                path.is_file() and not path.is_symlink()
                and (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
                == (old.st_ino, old.st_size, old.st_mtime_ns)
                for path, old in before.items()
            )
            if (not deletion_started and originals_intact and not had_descriptor
                    and protection_reasons(checkpoint, protection_fn())):
                descriptor_path.unlink(missing_ok=True)
                sync_directory(checkpoint)
                if published_record:
                    record(root, run, checkpoint, "hot" if plain.exists() else "archived",
                           original_bytes, before.get(packed, before[sources[0]]).st_size, sha256)
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return result


def restore_chunked_checkpoint(root, checkpoint, *, apply=False, min_free_bytes=0):
    checkpoint, run = checkpoint_path(root, checkpoint)
    plain, descriptor_path = checkpoint / "state.pt", checkpoint / chunks.DESCRIPTOR
    sha256 = expected_hash(checkpoint)
    result = {"checkpoint": str(checkpoint), "action": "restore", "applied": False}
    if not apply:
        descriptor = chunks.read_descriptor(root, checkpoint)
        if descriptor["sha256"] != sha256:
            raise ValueError("descriptor checkpoint hash mismatch")
        return result
    with run_lock(run):
        if not descriptor_path.exists():
            if plain.is_file() and not plain.is_symlink() and file_digest(plain)[0] == sha256:
                result["already_restored"] = True
                return result
            raise ValueError("missing chunk descriptor and verified plain checkpoint")
        descriptor = chunks.read_descriptor(root, checkpoint)
        if descriptor["sha256"] != sha256:
            raise ValueError("descriptor checkpoint hash mismatch")
        packed = checkpoint / "state.pt.gz"
        if packed.exists() or packed.is_symlink():
            if packed.is_symlink():
                raise ValueError("symlink payload is protected")
            with gzip.open(packed, "rb") as stream:
                if digest(stream, max_bytes=chunks.MAX_BYTES) != (sha256, descriptor["bytes"]):
                    raise ValueError("existing legacy archive checkpoint hash mismatch")
        if plain.exists() or plain.is_symlink():
            if plain.is_symlink() or file_digest(plain) != (sha256, descriptor["bytes"]):
                raise ValueError("existing plain checkpoint hash mismatch")
            chunks.verify_checkpoint(root, checkpoint, sha256)
            record(root, run, checkpoint, "hot", descriptor["bytes"], 0, sha256)
            packed.unlink(missing_ok=True)
            descriptor_path.unlink()
            sync_directory(checkpoint)
            result.update(applied=True, reconciled=True, original_bytes=descriptor["bytes"])
            return result
        if shutil.disk_usage(checkpoint).free < min_free_bytes + 2 * descriptor["bytes"]:
            result["skipped"] = "insufficient free disk for reserve and two staging copies"
            return result
        fd, name = tempfile.mkstemp(prefix=".storage-restore-", dir=checkpoint)
        os.close(fd)
        temporary = Path(name)
        temporary.unlink()
        try:
            chunks.restore_checkpoint(root, checkpoint, temporary, sha256)
            if plain.exists() or plain.is_symlink():
                if plain.is_symlink() or file_digest(plain) != (sha256, descriptor["bytes"]):
                    raise ValueError("existing plain checkpoint hash mismatch")
            else:
                os.link(temporary, plain)
                sync_directory(checkpoint)
            record(root, run, checkpoint, "hot", descriptor["bytes"], 0, sha256)
            packed.unlink(missing_ok=True)
            descriptor_path.unlink()
            sync_directory(checkpoint)
            result.update(applied=True, original_bytes=descriptor["bytes"])
        finally:
            temporary.unlink(missing_ok=True)
    return result


def transform(
    root, checkpoint, *, restore=False, apply=False, protection_fn=None, min_free_bytes=0,
    skip_unprofitable=False,
):
    root = root.resolve()
    checkpoint, run = checkpoint_path(root, checkpoint)
    if restore and (checkpoint / chunks.DESCRIPTOR).exists():
        return restore_chunked_checkpoint(
            root, checkpoint, apply=apply, min_free_bytes=min_free_bytes,
        )
    plain, packed = checkpoint / "state.pt", checkpoint / "state.pt.gz"
    source, destination = (packed, plain) if restore else (plain, packed)
    if source.is_symlink() or destination.is_symlink():
        raise ValueError("symlink payload is protected")
    if not source.is_file() and not (restore and apply):
        raise ValueError(f"missing source: {source}")
    protection_fn = protection_fn or (lambda: protection(root))
    if not restore:
        blocked = protection_reasons(checkpoint, protection_fn())
        if blocked:
            raise ValueError("protected checkpoint: " + "; ".join(blocked))
    sha256 = expected_hash(checkpoint)
    result = {
        "checkpoint": str(checkpoint),
        "action": "restore" if restore else "archive",
        "applied": False,
        "bytes": 0,
        "restore_command": restore_command(root, checkpoint),
    }
    try:
        result["bytes"] = source.stat().st_size
    except FileNotFoundError:
        if not (restore and apply):
            raise
    if not apply:
        if destination.exists():
            verify_existing_pair(plain, packed, sha256)
            result["reconciliation_required"] = True
        return result
    with run_lock(run):
        # A concurrent reader may have restored and removed the archive while
        # this caller waited for the lock. Accept only the verified hot payload.
        if restore and not source.is_file():
            if plain.is_file() and not plain.is_symlink():
                actual, original_bytes = file_digest(plain)
                if actual != sha256:
                    raise ValueError("already restored checkpoint hash mismatch")
                result.update(already_restored=True, original_bytes=original_bytes)
                return result
            raise ValueError(f"missing source: {source}")
        if not restore and protection_reasons(checkpoint, protection_fn()):
            raise ValueError("checkpoint became protected")
        before = source.stat()
        if destination.exists():
            if source.is_symlink() or destination.is_symlink():
                raise ValueError("symlink payload is protected")
            original_bytes, compressed_bytes = verify_existing_pair(plain, packed, sha256)
            if not restore and protection_reasons(checkpoint, protection_fn()):
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
            if not restore and protection_reasons(checkpoint, protection_fn()):
                raise ValueError("checkpoint became protected before source removal")
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
        # Both gzip staging and its compact copy coexist before atomic publication.
        # A plain-size upper bound avoids relying on optimistic compression ratios.
        staging_bytes = before.st_size
        if restore and min_free_bytes:
            with gzip.open(source, "rb") as stream:
                actual, staging_bytes = digest(stream)
            if actual != sha256:
                raise ValueError("checkpoint hash mismatch")
        if (min_free_bytes
                and shutil.disk_usage(checkpoint).free < min_free_bytes + 2 * staging_bytes):
            result["skipped"] = "insufficient free disk for reserve and two staging copies"
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
            if not restore and skip_unprofitable and compressed_bytes >= original_bytes:
                result["skipped"] = "compression would not save bytes"
                return result
            if actual != sha256:
                raise ValueError("checkpoint hash mismatch")
            after = source.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("source changed during operation")
            if not restore and protection_reasons(checkpoint, protection_fn()):
                raise ValueError("checkpoint became protected during compression")
            if min_free_bytes and shutil.disk_usage(checkpoint).free < min_free_bytes:
                result["skipped"] = "free disk fell below reserve during operation"
                return result
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
            if not restore and protection_reasons(checkpoint, protection_fn()):
                raise ValueError("checkpoint became protected before source removal")
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


def ensure_checkpoint_payload(checkpoint):
    """Restore a cold project checkpoint on demand before a normal model load."""
    checkpoint = Path(checkpoint).resolve()
    plain = checkpoint / "state.pt"
    chunked = (checkpoint / chunks.DESCRIPTOR).exists()
    if ((plain.is_file() and not chunked)
            or not (chunked or (checkpoint / "state.pt.gz").exists())):
        return plain
    runs = next((parent for parent in checkpoint.parents if parent.name == "runs"), None)
    if runs is None:
        raise ValueError("cold checkpoint must be inside a project runs directory")
    result = transform(
        runs.parent, checkpoint, restore=True, apply=True, min_free_bytes=8 * 1024**3,
    )
    if not result["applied"] and not result.get("already_restored"):
        raise OSError(f"cold checkpoint restore refused: {result.get('skipped', 'not restored')}")
    return plain


@contextlib.contextmanager
def checkpoint_read_lock(checkpoint):
    """Hold the archive writer lock across a model reader's hash and tensor load."""
    checkpoint = Path(checkpoint).resolve()
    runs = next((parent for parent in checkpoint.parents if parent.name == "runs"), None)
    if runs is None:
        yield ensure_checkpoint_payload(checkpoint)
        return
    _, run = checkpoint_path(runs.parent, checkpoint)
    for _ in range(3):
        plain = ensure_checkpoint_payload(checkpoint)
        with run_lock(run):
            # A writer could archive between ensure and acquiring the lock.
            if plain.is_file():
                yield plain
                return
    raise OSError("checkpoint repeatedly archived before reader could acquire its lock")


def main():
    def terminate(signum, _frame):
        # Run temporary-file cleanup on a maintenance subprocess timeout.
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("command", choices=("inventory", "plan", "archive", "restore", "housekeep"))
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--apply", action="store_true", help="perform verified reversible mutation")
    parser.add_argument("--hot-reference", type=Path, action="append", default=[],
                        help="current pair/input manifest; required for housekeep, repeatable")
    parser.add_argument("--keep", type=Path, action="append", default=[],
                        help="additional run/checkpoint to retain hot, repeatable")
    parser.add_argument("--max-checkpoints", type=int, default=8)
    parser.add_argument("--max-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--min-free-bytes", type=int, default=HOUSEKEEP_MIN_FREE_BYTES)
    parser.add_argument("--min-age-seconds", type=float, default=3600)
    args = parser.parse_args()
    try:
        if args.command == "housekeep":
            result = housekeep(
                args.root, hot_references=args.hot_reference, keep=args.keep, apply=args.apply,
                max_checkpoints=args.max_checkpoints, max_bytes=args.max_bytes,
                min_free_bytes=args.min_free_bytes, min_age_seconds=args.min_age_seconds,
            )
        elif args.command in {"inventory", "plan"}:
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
