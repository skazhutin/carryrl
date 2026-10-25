"""Bounded APFS clones for identical immutable checkpoint tokenizers only."""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from carryrl.training import checkpoint_storage as storage


def _safe(root: Path, path: Path) -> Path:
    relative = path.relative_to(root / "runs")
    parts = relative.parts
    if (len(parts) < 5 or parts[-2:] != ("language", "tokenizer.json")
            or "checkpoints" not in parts[1:-3]):
        raise ValueError("not a checkpoint language tokenizer")
    for current in (path, *path.parents):
        if current == root:
            break
        if current.is_symlink():
            raise ValueError("symlink resource or ancestor")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not info.st_mode & 0o222:
        raise ValueError("resource is not a writable regular file")
    if not path.parent.stat().st_mode & 0o222:
        raise ValueError("resource directory is read-only")
    return root / "runs" / parts[0]


def _identity(path: Path) -> tuple:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _digest(path: Path, deadline: float, max_bytes: int) -> str:
    digest = hashlib.sha256()
    consumed = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            consumed += len(chunk)
            if consumed > max_bytes:
                raise ValueError("resource grew beyond its bounded size")
            if time.monotonic() >= deadline:
                raise TimeoutError("resource compaction time budget exceeded")
            digest.update(chunk)
    return digest.hexdigest()


def compact_checkpoint_tokenizers(
    root: Path | str, *, apply: bool = False, max_files: int = 64,
    max_bytes: int = 1024**3, timeout_seconds: float = 60,
    baseline_directories: tuple[Path | str, ...] = (),
) -> dict:
    """Clone matching files without hard links, preserving independent future writes.

    Logical bytes describe successfully replaced duplicates, not physical savings.
    APFS sharing cannot be measured by summing per-file allocated blocks.
    """
    if (type(max_files) is not int or not 1 <= max_files <= 64
            or type(max_bytes) is not int or not 1 <= max_bytes <= 1024**3
            or not 0 < timeout_seconds <= 300):
        raise ValueError("invalid resource compaction budget")
    root = Path(root).resolve(strict=True)
    report = {"status": "complete", "mode": "apply" if apply else "dry_run",
              "matched": 0, "compacted": 0, "logical_bytes": 0,
              "skipped_busy": 0, "examined": 0, "examined_bytes": 0, "errors": [], "bounded": False,
              "free_bytes_before": shutil.disk_usage(root).free}
    if sys.platform != "darwin":
        report.update(status="unsupported", free_bytes_after=report["free_bytes_before"])
        return report
    runs = root / "runs"
    if runs.is_symlink() or not runs.is_dir():
        raise ValueError("unsafe project runs directory")
    baselines = [Path(os.path.abspath(root / path)) for path in baseline_directories]
    deadline = time.monotonic() + timeout_seconds
    groups = {}
    candidates = []
    visited = 0
    # Do not follow directory symlinks, including those pointing back into runs.
    for directory, directories, files in os.walk(runs, followlinks=False):
        visited += 1
        if time.monotonic() >= deadline or visited > 20000:
            report["bounded"] = True
            break
        directories[:] = sorted(name for name in directories
                                 if not (Path(directory) / name).is_symlink())
        if "tokenizer.json" not in files:
            continue
        path = Path(directory) / "tokenizer.json"
        try:
            run = _safe(root, path)
            size = path.stat().st_size
            if size > max_bytes:
                continue
            if report["examined"] >= max_files or report["examined_bytes"] + size > max_bytes:
                report["bounded"] = True
                break
            report["examined"] += 1
            report["examined_bytes"] += size
            protected = ("visual_bc_control" in run.name
                         or any(path.is_relative_to(base) for base in baselines))
            candidates.append((not protected, str(path), run, protected, size))
        except (OSError, ValueError) as error:
            report["errors"].append({"path": str(path), "error": str(error)})
    # Accepted baseline is the canonical source whenever present; never replace it.
    for _, value, run, protected, size in sorted(candidates):
        path = Path(value)
        temporary = None
        try:
            if time.monotonic() >= deadline:
                report["bounded"] = True
                break
            with storage.run_lock(run, blocking=False):
                _safe(root, path)
                identity = _identity(path)
                if identity[2] != size:
                    raise ValueError("resource size changed since inventory")
                digest = _digest(path, deadline, size)
                if identity != _identity(path):
                    raise ValueError("resource changed during digest")
            key = (identity[2], digest)
            if key not in groups:
                groups[key] = (path, run, identity)
                continue
            report["matched"] += 1
            if protected or not apply:
                continue
            source, source_run, source_identity = groups[key]
            with contextlib.ExitStack() as stack:
                for locked_run in sorted({run, source_run}):
                    stack.enter_context(storage.run_lock(
                        locked_run, blocking=False))
                for original, expected in ((path, identity), (source, source_identity)):
                    _safe(root, original)
                    if (_identity(original) != expected
                            or _digest(original, deadline, identity[2]) != digest):
                        raise ValueError("resource changed before clone")
                fd, name = tempfile.mkstemp(prefix=".tokenizer-clone-", dir=path.parent)
                os.close(fd)
                temporary = Path(name)
                # cp -c requires a fresh destination. Only our own empty staging is removed.
                temporary.unlink()
                subprocess.run(["/bin/cp", "-c", str(source), str(temporary)], check=True,
                               capture_output=True, timeout=max(0.001, deadline-time.monotonic()))
                if (temporary.is_symlink()
                        or not stat.S_ISREG(temporary.lstat().st_mode)
                        or _identity(temporary)[:2] in {identity[:2], source_identity[:2]}
                        or temporary.stat().st_size != identity[2]
                        or _digest(temporary, deadline, identity[2]) != digest):
                    raise ValueError("clone verification failed")
                shutil.copystat(path, temporary, follow_symlinks=False)
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                for original, expected in ((path, identity), (source, source_identity)):
                    _safe(root, original)
                    if (_identity(original) != expected
                            or _digest(original, deadline, identity[2]) != digest):
                        raise ValueError("resource changed while cloning")
                os.replace(temporary, path)
                storage.sync_directory(path.parent)
                report["compacted"] += 1
                report["logical_bytes"] += identity[2]
        except BlockingIOError:
            report["skipped_busy"] += 1
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report["errors"].append({"path": str(path), "error": str(error)})
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if report["errors"]:
        report["status"] = "failed"
    elif report["bounded"]:
        report["status"] = "bounded"
    report["free_bytes_after"] = shutil.disk_usage(root).free
    return report
