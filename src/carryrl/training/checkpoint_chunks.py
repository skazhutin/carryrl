"""Lossless cold checkpoint storage, with shared immutable compressed chunks."""

from __future__ import annotations

import contextlib
import fcntl
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import tempfile
import time
import zipfile
from pathlib import Path

FORMAT = "carryrl-checkpoint-chunks-v1"
DESCRIPTOR = "state.pt.chunks.json"
CHUNK_BYTES = 4 * 1024**2
MAX_BYTES = 64 * 1024**3
MAX_CHUNKS = 1_000_000
SHA = re.compile(r"[0-9a-f]{64}\Z")


def _regular(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"not a regular file: {path}")


def _paths(root, checkpoint=None):
    runs = Path(root).resolve() / "runs"
    store = runs / ".checkpoint_store"
    paths = [runs, store]
    if checkpoint is not None:
        checkpoint = Path(os.path.abspath(checkpoint))
        if not checkpoint.is_relative_to(runs) or checkpoint == runs:
            raise ValueError("checkpoint must be inside runs")
        paths.extend([checkpoint, *checkpoint.parents])
    for path in paths:
        if path == runs.parent:
            break
        if path.is_symlink():
            raise ValueError(f"symlink storage path: {path}")
    return store, checkpoint


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _sync(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def _locked(store):
    store.mkdir(parents=True, exist_ok=True)
    _sync(store.parent)
    lock = store / ".lock"
    if lock.is_symlink():
        raise ValueError("symlink chunk store lock")
    with lock.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _space(store, needed, reserve):
    if shutil.disk_usage(store).free < needed + reserve:
        raise OSError("insufficient free space for checkpoint chunks and reserve")


def _ranges(source, size):
    """Keep ZIP payloads aligned even when container metadata/offsets differ."""
    boundaries = {0, size}
    try:
        with zipfile.ZipFile(source) as archive, source.open("rb") as stream:
            for entry in archive.infolist():
                stream.seek(entry.header_offset)
                header = stream.read(30)
                if len(header) != 30 or header[:4] != b"PK\x03\x04":
                    raise ValueError("invalid local ZIP header")
                name, extra = struct.unpack_from("<HH", header, 26)
                start = entry.header_offset + 30 + name + extra
                end = start + entry.compress_size
                if not 0 <= start <= end <= size:
                    raise ValueError("invalid ZIP data range")
                boundaries.update((start, end))
    except (zipfile.BadZipFile, ValueError, OSError):
        boundaries = {0, size}
    points = sorted(boundaries)
    for begin, end in zip(points, points[1:]):
        for offset in range(begin, end, CHUNK_BYTES):
            yield offset, min(CHUNK_BYTES, end - offset)


def _read_chunk(store, item):
    path = store / (item["sha256"] + ".gz")
    _regular(path)
    # A valid gzip of a bounded chunk cannot need a huge compressed input either.
    if path.stat().st_size > CHUNK_BYTES + 65536:
        raise ValueError("oversized compressed chunk")
    with gzip.open(path, "rb") as stream:
        data = stream.read(item["bytes"] + 1)
        if len(data) != item["bytes"] or stream.read(1):
            raise ValueError("chunk size mismatch")
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError("chunk checksum mismatch")
    return data


def _validate(value):
    if not isinstance(value, dict) or value.get("format") != FORMAT:
        raise ValueError("unsupported checkpoint chunk descriptor")
    size = value.get("bytes")
    if type(size) is not int or not 0 <= size <= MAX_BYTES:
        raise ValueError("invalid checkpoint byte count")
    if not isinstance(value.get("sha256"), str) or not SHA.fullmatch(value["sha256"]):
        raise ValueError("invalid checkpoint checksum")
    chunks = value.get("chunks")
    if not isinstance(chunks, list) or len(chunks) > MAX_CHUNKS:
        raise ValueError("invalid chunk count")
    total = 0
    for item in chunks:
        if not isinstance(item, dict) or set(item) != {"sha256", "bytes"}:
            raise ValueError("invalid chunk reference")
        if not isinstance(item["sha256"], str) or not SHA.fullmatch(item["sha256"]):
            raise ValueError("invalid chunk checksum")
        if type(item["bytes"]) is not int or not 0 < item["bytes"] <= CHUNK_BYTES:
            raise ValueError("invalid chunk byte count")
        total += item["bytes"]
    if total != size:
        raise ValueError("checkpoint size mismatch")
    return value


def read_descriptor(root, checkpoint):
    """Read and validate a descriptor without trusting paths from its contents."""
    _, checkpoint = _paths(root, checkpoint)
    path = checkpoint / DESCRIPTOR
    _regular(path)
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError("oversized checkpoint chunk descriptor")
    return _validate(json.loads(path.read_text()))


def _verify(store, descriptor, output=None):
    digest = hashlib.sha256()
    for item in descriptor["chunks"]:
        data = _read_chunk(store, item)
        digest.update(data)
        if output is not None:
            output.write(data)
    if digest.hexdigest() != descriptor["sha256"]:
        raise ValueError("checkpoint checksum mismatch")


def _publish(store, destination, data, reserve):
    _space(store, len(data), reserve)
    fd, name = tempfile.mkstemp(prefix=".chunk-staging-", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Never replace an existing immutable chunk or an existing descriptor.
        os.link(temporary, destination)
        _sync(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def archive_checkpoint(root, checkpoint, source_plain, expected_sha256,
                       min_free_bytes=2 * 1024**3, *, pre_publish=None):
    """Publish verified chunks/descriptor; never delete or modify the source."""
    store, checkpoint = _paths(root, checkpoint)
    source = Path(source_plain)
    _regular(source)
    before = source.stat()
    if before.st_size > MAX_BYTES:
        raise ValueError("checkpoint exceeds size limit")
    with _locked(store):
        descriptor_path = checkpoint / DESCRIPTOR
        if descriptor_path.exists() or descriptor_path.is_symlink():
            descriptor = read_descriptor(root, checkpoint)
            if descriptor["sha256"] != expected_sha256:
                raise ValueError("existing descriptor checksum mismatch")
            _verify(store, descriptor)
            if pre_publish is not None:
                pre_publish()
            return descriptor
        plan, chunks, missing = [], [], {}
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for offset, length in _ranges(source, before.st_size):
                stream.seek(offset)
                data = stream.read(length)
                if len(data) != length:
                    raise ValueError("checkpoint changed while reading")
                digest.update(data)
                item = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": length}
                chunks.append(item)
                plan.append((offset, item))
                destination = store / (item["sha256"] + ".gz")
                if destination.exists() or destination.is_symlink():
                    _read_chunk(store, item)
                elif item["sha256"] not in missing:
                    missing[item["sha256"]] = len(gzip.compress(data, compresslevel=1, mtime=0))
        if digest.hexdigest() != expected_sha256:
            raise ValueError("source checkpoint checksum mismatch")
        descriptor = _validate({"format": FORMAT, "sha256": expected_sha256,
                                "bytes": before.st_size, "chunks": chunks})
        encoded = (json.dumps(descriptor, sort_keys=True) + "\n").encode()
        _space(store, sum(missing.values()) + len(encoded), min_free_bytes)
        with source.open("rb") as stream:
            for offset, item in plan:
                destination = store / (item["sha256"] + ".gz")
                if destination.exists():
                    continue
                stream.seek(offset)
                data = stream.read(item["bytes"])
                if hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("checkpoint changed while archiving")
                _publish(store, destination, gzip.compress(data, compresslevel=1, mtime=0),
                         min_free_bytes)
        _verify(store, descriptor)
        after = source.stat()
        if _identity(before) != _identity(after):
            raise ValueError("checkpoint changed while archiving")
        if pre_publish is not None:
            pre_publish()
        if _identity(before) != _identity(source.stat()):
            raise ValueError("checkpoint changed before descriptor publication")
        _publish(store, descriptor_path, encoded, min_free_bytes)
        return descriptor


def verify_checkpoint(root, checkpoint, expected_sha):
    """Verify archived bytes without creating a plain checkpoint or staging file."""
    store, _ = _paths(root, checkpoint)
    with _locked(store):
        descriptor = read_descriptor(root, checkpoint)
        if descriptor["sha256"] != expected_sha:
            raise ValueError("descriptor checkpoint checksum mismatch")
        _verify(store, descriptor)
        return descriptor


def restore_checkpoint(root, checkpoint, target_temporary, expected_sha):
    """Create an exclusive verified temporary file; caller owns final publication."""
    store, _ = _paths(root, checkpoint)
    with _locked(store):
        descriptor = read_descriptor(root, checkpoint)
        if descriptor["sha256"] != expected_sha:
            raise ValueError("descriptor checkpoint checksum mismatch")
        target = Path(target_temporary)
        created = False
        try:
            with target.open("xb") as stream:
                created = True
                _verify(store, descriptor, stream)
                stream.flush()
                os.fsync(stream.fileno())
            return descriptor
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise

def cleanup_orphan_chunks(root, min_age_seconds=86400):
    """Fail closed on unreadable metadata; collect only old unreferenced chunks."""
    store, _ = _paths(root)
    with _locked(store):
        referenced, descriptors = set(), []
        # os.walk with an error callback also catches unreadable directories.
        def fail(error):
            raise error
        for directory, dirs, files in os.walk(store.parent, onerror=fail):
            current = Path(directory)
            if current == store:
                dirs[:] = []
                continue
            for name in dirs:
                if (current / name).is_symlink():
                    raise ValueError("symlink directory blocks chunk collection")
            if DESCRIPTOR in files:
                descriptor = read_descriptor(root, current)
                descriptors.append(descriptor)
                referenced.update(item["sha256"] for item in descriptor["chunks"])
        candidates, staging = [], []
        cutoff = time.time() - max(0, min_age_seconds)
        for path in store.iterdir():
            if path.name == ".lock":
                continue
            if path.name.startswith(".chunk-staging-"):
                _regular(path)
                if path.stat().st_mtime < cutoff:
                    staging.append(path)
                continue
            if path.suffix != ".gz" or not SHA.fullmatch(path.stem):
                raise ValueError("unexpected chunk store entry")
            _regular(path)
            if path.stem not in referenced and path.stat().st_mtime < cutoff:
                candidates.append(path)
        if candidates or staging:
            for descriptor in descriptors:
                _verify(store, descriptor)
        removed = sum(path.stat().st_size for path in candidates + staging)
        for path in candidates + staging:
            path.unlink()
        _sync(store)
        return {"removed_chunks": len(candidates), "removed_staging": len(staging),
                "removed_bytes": removed}
