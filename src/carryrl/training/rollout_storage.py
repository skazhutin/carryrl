"""Bounded complete-rollout arrays with optional disk-backed frozen features.

Counters describe retained array backing, excluding OS page cache, model weights,
optimizer state, Python objects and transient encoder/optimizer working memory.
Mapped tensors remain valid while their owning episodes exist; close unlinks only
this storage instance's files and never forcibly invalidates a live Torch view.
"""

import io
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

MIB = 1024**2
DISK_RESERVE_BYTES = 16 * MIB
MINIMUM_DISK_FREE_BYTES = 2048 * MIB


def _array_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    return 0


def _episode_array_bytes(episode):
    """Direct episode tensor/array fields; small Python histories are outside scope."""
    return sum(_array_bytes(value) for value in vars(episode).values())


class RolloutStorage:
    def __init__(self, output, *, mode="memory", memory_budget_mib=0,
                 disk_budget_mib=0, minimum_free_disk_mib=0):
        if mode not in ("memory", "mmap"):
            raise ValueError("rollout feature storage must be memory or mmap")
        for name, value in (("memory_budget_mib", memory_budget_mib),
                            ("disk_budget_mib", disk_budget_mib),
                            ("minimum_free_disk_mib", minimum_free_disk_mib)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.mode = mode
        self.memory_budget_bytes = memory_budget_mib * MIB
        self.disk_budget_bytes = disk_budget_mib * MIB
        self.minimum_free_disk_bytes = max(MINIMUM_DISK_FREE_BYTES,
                                           minimum_free_disk_mib * MIB)
        self.logical_array_bytes = self.mapped_feature_bytes = 0
        self.retained_memory_array_bytes = self.disk_bytes = 0
        self._episodes, self._episode_ids, self._files = [], set(), []
        self._closed = False
        self.directory = None
        if mode == "mmap":
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(prefix=".rollout-features-", dir=output))

    @property
    def remaining_memory_bytes(self):
        if not self.memory_budget_bytes:
            return None
        return max(0, self.memory_budget_bytes - self.retained_memory_array_bytes)

    def metrics(self):
        return {"mode": self.mode, "logical_array_bytes": self.logical_array_bytes,
                "mapped_feature_bytes": self.mapped_feature_bytes,
                "retained_memory_array_bytes": self.retained_memory_array_bytes,
                "disk_bytes": self.disk_bytes, "episodes": len(self._episodes),
                "scope": "retained direct episode arrays; excludes OS page cache, model, "
                         "optimizer, Python objects and transient working memory"}

    def store_episode(self, episode):
        if self._closed:
            raise RuntimeError("rollout storage is closed")
        if id(episode) in self._episode_ids:
            raise ValueError("episode already belongs to this rollout storage")
        patches = episode.patches
        if not isinstance(patches, torch.Tensor) or patches.device.type != "cpu":
            raise ValueError("rollout frozen patches must be a CPU Torch tensor")
        if patches.requires_grad:
            raise ValueError("rollout frozen patches must not require gradients")
        logical_bytes = _episode_array_bytes(episode)
        feature_bytes = _array_bytes(patches)
        retained = logical_bytes - (feature_bytes if self.mode == "mmap" else 0)
        if (self.memory_budget_bytes
                and self.retained_memory_array_bytes + retained > self.memory_budget_bytes):
            raise MemoryError("rollout retained arrays exceed the configured memory budget")
        disk_bytes = 0
        if self.mode == "mmap":
            # Numpy views share CPU tensor storage; neither conversion nor the
            # loaded copy-on-write map clones the full frozen feature array.
            array = patches.detach().numpy()
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(
                header, np.lib.format.header_data_from_array_1_0(array))
            estimated = feature_bytes + header.tell()
            if (self.disk_budget_bytes
                    and self.disk_bytes + estimated + DISK_RESERVE_BYTES > self.disk_budget_bytes):
                raise OSError("rollout mapped features exceed the configured disk budget")
            available = shutil.disk_usage(self.directory).free
            if available - estimated - DISK_RESERVE_BYTES < self.minimum_free_disk_bytes:
                raise OSError("rollout mapped features would cross the free disk floor")
            filename = self.directory / f"episode-{len(self._episodes):06d}.npy"
            created = False
            try:
                with filename.open("xb") as stream:
                    created = True
                    np.lib.format.write_array(stream, array, version=(1, 0), allow_pickle=False)
                disk_bytes = filename.stat().st_size
                if disk_bytes > estimated:
                    raise OSError("mapped feature file exceeded its preflight size bound")
                mapped = np.load(filename, mmap_mode="c", allow_pickle=False)
                tensor = torch.from_numpy(mapped)
                if tensor.shape != patches.shape or tensor.dtype != patches.dtype:
                    raise ValueError("mapped feature shape/dtype differs from the CPU episode")
                episode.patches = tensor
                self._files.append(filename)
            except BaseException:
                if created:
                    filename.unlink(missing_ok=True)
                raise
        self.logical_array_bytes += logical_bytes
        self.mapped_feature_bytes += feature_bytes if self.mode == "mmap" else 0
        self.retained_memory_array_bytes += retained
        self.disk_bytes += disk_bytes
        self._episode_ids.add(id(episode))
        self._episodes.append(episode)
        return self.metrics()

    def close(self):
        if self._closed:
            return
        for filename in self._files:
            filename.unlink(missing_ok=True)
        if self.directory is not None:
            try:
                self.directory.rmdir()
            except OSError:
                # A foreign file may have appeared after unique directory creation.
                # It is outside this instance's ownership and must survive cleanup.
                if not self.directory.is_dir() or not any(self.directory.iterdir()):
                    raise
        self._files.clear()
        self._episodes.clear()
        self._episode_ids.clear()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("rollout storage is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
