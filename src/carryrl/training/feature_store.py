"""Content-addressed, verified disk storage of detached raw frozen vision patches."""

import errno
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
import zlib
from datetime import UTC, datetime
from math import isqrt
from pathlib import Path

import numpy as np
import torch

FORMAT = "carryrl.frozen_features.v1"
REUSE_FORMAT = "carryrl.frozen_feature_reuse.v1"


def _sha256_file(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _layout(identity):
    indices = identity["indices"]
    shape = tuple(identity["feature_shape"])
    dtype = np.dtype(identity["dtype"])
    if (not indices or indices != sorted(set(indices))
            or any(type(index) is not int or index < 0 for index in indices)
            or len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape)
            or isqrt(shape[0]) ** 2 != shape[0] or shape[1] != 384
            or dtype not in (np.dtype("float16"), np.dtype("float32"))):
        raise ValueError("feature store requires ordered indices and raw square DINO patches")
    full_shape = (len(indices), *shape)
    stored_bytes = len(indices) * shape[0] * shape[1] * dtype.itemsize
    return full_shape, dtype, stored_bytes


def feature_cache_identity(frames, indices, *, vision_sha256, preprocessing, feature_shape, dtype):
    """Hash exact ordered uint8 RGB inputs, independently of labels and policy weights."""
    identity = {"format": FORMAT, "indices": list(indices), "vision_sha256": vision_sha256,
                "preprocessing": preprocessing, "feature_shape": list(feature_shape),
                "dtype": dtype}
    _layout(identity)
    digest = hashlib.sha256()
    for index in identity["indices"]:
        frame = frames[index]
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("feature cache input must be uint8 HWC RGB")
        # Shape/index delimiters prevent ambiguous concatenations of variable-size frames.
        digest.update(json.dumps([index, list(frame.shape)]).encode() + b"\n")
        digest.update(frame.tobytes())
    identity["ordered_rgb_sha256"] = digest.hexdigest()
    return identity


def _rgb_key(frame):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("feature cache input must be uint8 HWC RGB")
    return tuple(frame.shape), hashlib.sha256(frame.tobytes()).hexdigest()


def _source_rgb_keys(path, identity, *, index_keys=None, extra_identity=None):
    """Stream a stacked NPZ member, or load individual variable-size frame members."""
    indices = identity["indices"]
    wanted = set(indices)
    digest = hashlib.sha256()
    keys = {}
    extra_indices = set(extra_identity["indices"]) if extra_identity is not None else set()
    extra_digest = hashlib.sha256()
    if not extra_indices <= wanted:
        raise ValueError("source shard indices are outside the composite identity")

    def consume(index, frame):
        key = _rgb_key(frame)
        digest.update(json.dumps([index, list(frame.shape)]).encode() + b"\n")
        digest.update(frame.tobytes())
        if index_keys is not None:
            index_keys[index] = key
        if index in extra_indices:
            extra_digest.update(json.dumps([index, list(frame.shape)]).encode() + b"\n")
            extra_digest.update(frame.tobytes())
        # Duplicate old content may safely share the first verified feature row.
        keys.setdefault(key, index)

    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("source RGB archive contains duplicate members")
            if "frames.npy" in names:
                if names != ["frames.npy"]:
                    raise ValueError("source RGB archive has ambiguous frame layout")
                with archive.open("frames.npy") as source:
                    version = np.lib.format.read_magic(source)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(source)
                    else:
                        raise ValueError("unsupported source RGB NPY version")
                    if (dtype != np.dtype("uint8") or fortran or len(shape) != 4
                            or shape[-1] != 3 or any(size <= 0 for size in shape)
                            or indices[-1] >= shape[0]):
                        raise ValueError("source RGB archive layout mismatch")
                    row_bytes = int(np.prod(shape[1:]))
                    for index in range(shape[0]):
                        data = source.read(row_bytes)
                        if len(data) != row_bytes:
                            raise ValueError("source RGB archive is truncated")
                        if index in wanted:
                            consume(index, np.frombuffer(data, dtype=dtype).reshape(shape[1:]))
                    if source.read(1):
                        raise ValueError("source RGB archive has trailing data")
            else:
                with np.load(path, allow_pickle=False) as source:
                    for index in indices:
                        consume(index, source[f"frame_{index:06}"])
        if digest.hexdigest() != identity["ordered_rgb_sha256"]:
            raise ValueError("source RGB ordered digest mismatch")
        if extra_identity is not None and extra_digest.hexdigest() != extra_identity[
                "ordered_rgb_sha256"]:
            raise ValueError("source shard RGB ordered digest mismatch")
        return keys
    except (OSError, KeyError, EOFError, zipfile.BadZipFile, zlib.error) as error:
        raise ValueError("source RGB archive is missing, partial or corrupt") from error


def _pack_reference(store):
    return {"directory": str(store.directory.resolve()),
            "identity": store.manifest["identity"], "sha256": store.manifest["sha256"],
            "file_bytes": store.manifest["file_bytes"], "bytes": store.manifest["bytes"],
            "created_utc": store.manifest["created_utc"]}


def _load_reuse_pack(reference, options, cache_root, max_bytes, verified):
    recorded_path = Path(reference["directory"])
    directory = recorded_path.resolve()
    if not recorded_path.is_absolute() or directory.parent != cache_root:
        raise ValueError("source feature pack path differs from cache root")
    if recorded_path.is_symlink():
        raise ValueError("source feature pack must not be a symlink")
    identity = reference["identity"]
    if identity["format"] != FORMAT or directory.name != _key(identity):
        raise ValueError("source feature pack path or format differs from identity")
    if any(identity[name] != value for name, value in options.items()):
        raise ValueError("source frozen encoder configuration is incompatible")
    if directory not in verified:
        store = FrozenFeatureStore.load(cache_root, identity, max_bytes=max_bytes)
        if store is None:
            raise ValueError("source frozen feature pack is missing")
        verified[directory] = store
    store = verified[directory]
    if reference != _pack_reference(store):
        raise ValueError("source feature pack reference mismatch")
    return store


class _CompositeSource:
    """Flattened verified leaf routing; recursion never remains in tensor reads."""

    def __init__(self, manifest, rows, manifest_sha256):
        self.manifest, self.rows, self.manifest_sha256 = manifest, rows, manifest_sha256

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        store, leaf_index = self.rows[index]
        return store[leaf_index]


def _resolve_reuse_source(source_run, options, cache_root, max_bytes, verified, visited):
    if source_run in visited or len(visited) >= 16:
        raise ValueError("source feature reuse ancestry contains a cycle or exceeds 16 runs")
    visited = {*visited, source_run}
    manifest_path = source_run / "feature_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    recorded = json.loads(manifest_bytes)
    reuse = recorded.get("feature_reuse")
    if reuse is None:
        recorded_path = Path(recorded["feature_cache_dir"])
        directory = recorded_path.resolve()
        if (not recorded_path.is_absolute() or recorded_path.is_symlink()
                or directory.parent != cache_root):
            raise ValueError("source feature pack path differs from cache root")
        pack_manifest = json.loads((directory / "manifest.json").read_text())
        reference = {"directory": recorded["feature_cache_dir"],
                     "identity": pack_manifest["identity"],
                     "sha256": recorded["feature_cache_sha256"],
                     "created_utc": recorded["feature_cache_created_utc"],
                     "file_bytes": pack_manifest["file_bytes"], "bytes": recorded["bytes"]}
        provider = _load_reuse_pack(reference, options, cache_root, max_bytes, verified)
        identity = provider.manifest["identity"]
        index_keys = {}
        keys = _source_rgb_keys(source_run / "demonstration_frames.npz", identity,
                                index_keys=index_keys)
        rows = {index: (provider, index) for index in identity["indices"]}
    else:
        identity = reuse["identity"]
        shape, _, logical_bytes = _layout(identity)
        if logical_bytes > max_bytes:
            raise MemoryError("frozen demonstration features exceed --feature-budget-mib")
        if (reuse["format"] != REUSE_FORMAT or identity["format"] != FORMAT
                or any(identity[name] != value for name, value in options.items())):
            raise ValueError("source composite frozen encoder configuration is incompatible")
        parent_path = Path(reuse["source_run"])
        if not parent_path.is_absolute() or parent_path.is_symlink():
            raise ValueError("source feature reuse ancestor path is invalid")
        parent, _, parent_index_keys, parent_rows = _resolve_reuse_source(
            parent_path.resolve(), options, cache_root, max_bytes, verified, visited)
        if isinstance(parent, FrozenFeatureStore):
            if reuse.get("source_reuse") is not None or reuse["source_pack"] != _pack_reference(
                    parent):
                raise ValueError("source composite parent pack reference mismatch")
        else:
            expected = {"source_run": str(parent_path.resolve()),
                        "manifest_sha256": parent.manifest_sha256,
                        "identity": parent.manifest["identity"],
                        "mapping_sha256": parent.manifest["mapping_sha256"]}
            if reuse.get("source_pack") is not None or reuse.get("source_reuse") != expected:
                raise ValueError("source composite parent reuse reference mismatch")
        shard_reference = reuse["shard_pack"]
        shard = (_load_reuse_pack(shard_reference, options, cache_root, max_bytes, verified)
                 if shard_reference is not None else None)
        index_keys = {}
        keys = _source_rgb_keys(source_run / "demonstration_frames.npz", identity,
            index_keys=index_keys, extra_identity=shard.manifest["identity"] if shard else None)
        mapping = reuse["mapping"]
        if (not isinstance(mapping, list) or len(mapping) != len(identity["indices"])
                or hashlib.sha256(json.dumps(mapping).encode()).hexdigest()
                != reuse["mapping_sha256"]):
            raise ValueError("source composite mapping digest or count mismatch")
        rows, shard_indices = {}, []
        reused_count = 0
        for index, entry in zip(identity["indices"], mapping, strict=True):
            if (not isinstance(entry, list) or len(entry) != 3 or type(entry[0]) is not int
                    or entry[0] != index or type(entry[2]) is not int):
                raise ValueError("source composite mapping is malformed or unordered")
            _, origin, target = entry
            if origin == "source":
                if target not in parent_rows or index_keys[index] != parent_index_keys[target]:
                    raise ValueError("source composite mapping RGB reference mismatch")
                rows[index] = parent_rows[target]
                reused_count += 1
            elif origin == "shard":
                if shard is None or target != index or target not in shard._rows:
                    raise ValueError("source composite mapping shard reference mismatch")
                rows[index] = shard, target
                shard_indices.append(target)
            else:
                raise ValueError("source composite mapping origin is invalid")
        if shard is not None and shard_indices != shard.manifest["identity"]["indices"]:
            raise ValueError("source composite shard mapping is incomplete")
        row_bytes = logical_bytes // shape[0]
        expected = {"frames": len(rows), "bytes": logical_bytes, "logical_bytes": logical_bytes,
                    "reused_count": reused_count, "new_count": len(shard_indices),
                    "reused_bytes": reused_count * row_bytes,
                    "new_bytes": len(shard_indices) * row_bytes}
        if any(reuse[name] != value for name, value in expected.items()):
            raise ValueError("source composite counts or byte provenance mismatch")
        provider = _CompositeSource(reuse, rows, hashlib.sha256(manifest_bytes).hexdigest())
    expected = {"bytes": len(rows) * np.prod(identity["feature_shape"])
                * np.dtype(identity["dtype"]).itemsize, "frames": len(rows),
                "dtype": identity["dtype"], "vision_sha256": identity["vision_sha256"]}
    if any(recorded[name] != value for name, value in expected.items()):
        raise ValueError("source run feature provenance mismatch")
    return provider, keys, index_keys, rows


class FrozenFeatureReuse:
    """Current frame indices routed to exact RGB matches in immutable layered packs."""

    @classmethod
    def prepare(cls, source_run, frames, used, current_identity, *, cache_root, max_bytes):
        """Verify the old pack and RGBs without loading a model or copying feature data."""
        used = list(used)
        shape, _, logical_bytes = _layout(current_identity)
        if used != current_identity["indices"]:
            raise ValueError("reuse indices differ from current feature identity")
        if logical_bytes > max_bytes:
            raise MemoryError("frozen demonstration features exceed --feature-budget-mib")
        options = {name: current_identity[name] for name in
                   ("vision_sha256", "preprocessing", "feature_shape", "dtype")}
        if feature_cache_identity(frames, used, **options) != current_identity:
            raise ValueError("current RGB identity mismatch")
        source_run = Path(source_run).resolve()
        cache_root = Path(cache_root).resolve()
        try:
            original, keys, _, _ = _resolve_reuse_source(
                source_run, options, cache_root, max_bytes, {}, set())
        except (OSError, KeyError, TypeError, EOFError, AttributeError, IndexError) as error:
            raise ValueError("source feature reuse provenance is partial or malformed") from error
        result = cls()
        result.original = original
        result.source_run = source_run
        result.current_identity = current_identity
        result.logical_bytes = result.feature_bytes = logical_bytes
        result._used = set(used)
        result._reused = {}
        result.missing_indices = []
        for index in used:
            old_index = keys.get(_rgb_key(frames[index]))
            if old_index is None:
                result.missing_indices.append(index)
            else:
                result._reused[index] = old_index
        result.reused_count = len(result._reused)
        result.shard_identity = (feature_cache_identity(frames, result.missing_indices, **options)
                                 if result.missing_indices else None)
        result.shard = None
        result._row_bytes = logical_bytes // shape[0]
        return result

    def bind_shard(self, store):
        """Bind a complete verified missing-frame pack before training begins."""
        if (self.shard_identity is None or not isinstance(store, FrozenFeatureStore)
                or store.manifest["identity"] != self.shard_identity
                or len(store) != len(self.missing_indices)):
            raise ValueError("reuse shard identity or completeness mismatch")
        verified = FrozenFeatureStore.load(store.directory.parent, self.shard_identity,
                                           max_bytes=self.logical_bytes)
        if verified is None or verified.directory.resolve() != store.directory.resolve():
            raise ValueError("reuse shard pack is missing or has an invalid path")
        self.shard = verified
        return self

    def __len__(self):
        return len(self._used)

    def __getitem__(self, index):
        if index not in self._used:
            raise KeyError(index)
        if index in self._reused:
            return self.original[self._reused[index]]
        if self.shard is None:
            raise ValueError("missing frozen feature shard has not been bound")
        return self.shard[index]

    @property
    def manifest(self):
        if self.missing_indices and self.shard is None:
            raise ValueError("cannot report complete reuse provenance before binding shard")

        source_pack = (_pack_reference(self.original)
                       if isinstance(self.original, FrozenFeatureStore) else None)
        source_reuse = (None if source_pack is not None else {
            "source_run": str(self.source_run),
            "manifest_sha256": self.original.manifest_sha256,
            "identity": self.original.manifest["identity"],
            "mapping_sha256": self.original.manifest["mapping_sha256"],
        })
        mapping = [[index, "source", self._reused[index]] if index in self._reused
                   else [index, "shard", index] for index in self.current_identity["indices"]]
        return {"format": REUSE_FORMAT, "identity": self.current_identity,
                "source_run": str(self.source_run), "source_pack": source_pack,
                **({"source_reuse": source_reuse} if source_reuse is not None else {}),
                "shard_pack": _pack_reference(self.shard) if self.shard is not None else None,
                "mapping": mapping,
                "mapping_sha256": hashlib.sha256(json.dumps(mapping).encode()).hexdigest(),
                "frames": len(self), "bytes": self.logical_bytes,
                "logical_bytes": self.logical_bytes, "reused_count": self.reused_count,
                "new_count": len(self.missing_indices),
                "reused_bytes": self.reused_count * self._row_bytes,
                "new_bytes": len(self.missing_indices) * self._row_bytes,
                "storage": "composite read-only NumPy memmaps; copied per requested frame"}


class FrozenFeatureStore:
    """Read-only memmap; each requested frame is copied into a detached Torch tensor."""

    def __init__(self, directory, manifest, array):
        self.directory = directory
        self.manifest = manifest
        self.array = array
        self._rows = {index: row for row, index in enumerate(manifest["identity"]["indices"])}

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return torch.from_numpy(self.array[self._rows[index]].copy())

    @classmethod
    def load(cls, cache_dir, identity, *, max_bytes):
        """Return a verified hit or None; corrupt published entries are explicit errors."""
        directory = Path(cache_dir) / _key(identity)
        if not directory.exists():
            return None
        shape, dtype, stored_bytes = _layout(identity)
        if stored_bytes > max_bytes:
            raise MemoryError("frozen demonstration features exceed --feature-budget-mib")
        try:
            manifest = json.loads((directory / "manifest.json").read_text())
            data = directory / "features.npy"
            if (manifest["format"] != FORMAT or manifest["identity"] != identity
                    or manifest["key"] != _key(identity)
                    or manifest["shape"] != list(shape) or manifest["dtype"] != dtype.name
                    or manifest["bytes"] != stored_bytes
                    or data.stat().st_size != manifest["file_bytes"]
                    or _sha256_file(data) != manifest["sha256"]):
                raise ValueError("frozen feature cache manifest or file hash mismatch")
            array = np.load(data, mmap_mode="r", allow_pickle=False)
            if array.shape != shape or array.dtype != dtype or array.nbytes != stored_bytes:
                raise ValueError("frozen feature cache array layout mismatch")
            if not isinstance(array, np.memmap) or array.flags.writeable:
                raise ValueError("frozen feature cache must be a read-only NumPy memmap")
            return cls(directory, manifest, array)
        except (OSError, KeyError, TypeError, EOFError) as error:
            raise ValueError("frozen feature cache is partial or malformed") from error


class FrozenFeatureWriter:
    """Fill every row once, then atomically publish both the data and its manifest."""

    def __init__(self, cache_dir, identity, *, max_bytes):
        self.shape, self.dtype, self.stored_bytes = _layout(identity)
        if self.stored_bytes > max_bytes:
            raise MemoryError("frozen demonstration features exceed --feature-budget-mib")
        self.cache_dir = Path(cache_dir)
        self.identity = identity
        self.max_bytes = max_bytes
        self._rows = {index: row for row, index in enumerate(identity["indices"])}
        self._written = set()
        self.temporary = self.array = None

    def __enter__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temporary = Path(tempfile.mkdtemp(prefix=f".{_key(self.identity)}-",
                                               suffix=".partial", dir=self.cache_dir))
        try:
            self.array = np.lib.format.open_memmap(
                self.temporary / "features.npy", mode="w+", dtype=self.dtype, shape=self.shape)
        except BaseException:
            shutil.rmtree(self.temporary)
            raise
        return self

    def __exit__(self, *error):
        self.array = None
        if self.temporary is not None:
            shutil.rmtree(self.temporary, ignore_errors=True)

    def write(self, indices, encoded):
        encoded = np.asarray(encoded)
        if (encoded.shape != (len(indices), *self.shape[1:]) or encoded.dtype != self.dtype
                or not np.isfinite(encoded).all()):
            raise ValueError("feature cache batch must contain finite raw patches at storage dtype")
        if (len(set(indices)) != len(indices) or any(index not in self._rows for index in indices)
                or self._written.intersection(indices)):
            raise ValueError("feature cache batch has unknown or duplicate frame indices")
        self.array[[self._rows[index] for index in indices]] = encoded
        self._written.update(indices)

    def publish(self):
        if self._written != set(self._rows):
            raise ValueError("cannot publish an incomplete frozen feature cache")
        self.array.flush()
        self.array = None
        data = self.temporary / "features.npy"
        with data.open("rb") as source:
            os.fsync(source.fileno())
        manifest = {"format": FORMAT, "key": _key(self.identity), "identity": self.identity,
                    "shape": list(self.shape), "dtype": self.dtype.name, "bytes": self.stored_bytes,
                    "file_bytes": data.stat().st_size, "sha256": _sha256_file(data),
                    "created_utc": datetime.now(UTC).isoformat()}
        with (self.temporary / "manifest.json").open("w") as destination:
            json.dump(manifest, destination, indent=2, sort_keys=True)
            destination.flush()
            os.fsync(destination.fileno())
        directory = self.cache_dir / manifest["key"]
        try:
            os.rename(self.temporary, directory)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # Another run may publish the same immutable cache while we encode.
            # Verify that winner before accepting it; our context cleans its own partial.
            existing = FrozenFeatureStore.load(self.cache_dir, self.identity,
                                               max_bytes=self.max_bytes)
            if existing is None:
                raise
            return existing
        self.temporary = None
        # The just-written file was hashed before the atomic directory rename.
        array = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
        return FrozenFeatureStore(directory, manifest, array)
