"""Rollout array parity and fail-closed resource/file ownership checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carryrl.training.rollout_storage import (
    DISK_RESERVE_BYTES,
    MIB,
    MINIMUM_DISK_FREE_BYTES,
    RolloutStorage,
)


def _episode(feature_elements=128, rgb_elements=100):
    return SimpleNamespace(
        patches=torch.arange(feature_elements, dtype=torch.float32).reshape(1, 1, -1),
        frames=np.arange(rgb_elements, dtype=np.uint8), final_frame=np.zeros(6, dtype=np.uint8),
        old_log_prob=torch.zeros(1), old_values=torch.ones(1), rewards=torch.tensor([.5]))


def _free(monkeypatch, value):
    import carryrl.training.rollout_storage as module

    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=value))


def test_mmap_parity_zero_copy_accounting_and_cleanup(tmp_path, monkeypatch):
    _free(monkeypatch, 8 * 1024**3)
    foreign = tmp_path / "existing.txt"
    foreign.write_text("user file")
    episode = _episode()
    expected, rgb = episode.patches.clone(), episode.frames
    store = RolloutStorage(tmp_path, mode="mmap", memory_budget_mib=1, disk_budget_mib=32)
    directory = store.directory
    metrics = store.store_episode(episode)
    torch.testing.assert_close(episode.patches, expected)
    assert episode.frames is rgb
    assert metrics["mapped_feature_bytes"] == expected.numel() * expected.element_size()
    assert metrics["retained_memory_array_bytes"] == 100 + 6 + 3 * 4
    assert metrics["logical_array_bytes"] == metrics["mapped_feature_bytes"] + 118
    assert metrics["disk_bytes"] > metrics["mapped_feature_bytes"]
    assert store.remaining_memory_bytes == MIB - 118
    # Torch owns the copy-on-write numpy map directly; modifications do not
    # alter the persisted frozen file and do not require a whole-array clone.
    feature_file = next(directory.glob("*.npy"))
    episode.patches[0, 0, 0] = -5
    assert np.load(feature_file)[0, 0, 0] == 0
    assert episode.patches[0, 0, 0].item() == -5
    store.close()
    assert not directory.exists()
    assert foreign.read_text() == "user file"
    # Cleanup must not invalidate views still held by an episode.
    assert episode.patches[0, 0, 1].item() == 1
    store.close()


def test_memory_mode_counts_full_features_and_creates_no_directory(tmp_path):
    output = tmp_path / "not-created"
    store = RolloutStorage(output, memory_budget_mib=1)
    episode = _episode()
    original = episode.patches
    metrics = store.store_episode(episode)
    assert episode.patches is original
    assert metrics["retained_memory_array_bytes"] == metrics["logical_array_bytes"] == 630
    assert metrics["disk_bytes"] == metrics["mapped_feature_bytes"] == 0
    assert not output.exists()
    with pytest.raises(ValueError, match="already"):
        store.store_episode(episode)
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.store_episode(_episode())


def test_mapped_features_do_not_consume_remaining_retained_array_budget(tmp_path, monkeypatch):
    _free(monkeypatch, 8 * 1024**3)
    with RolloutStorage(tmp_path, mode="mmap", memory_budget_mib=1, disk_budget_mib=32) as store:
        store.store_episode(_episode(feature_elements=300000))
        assert store.mapped_feature_bytes > MIB
        assert store.remaining_memory_bytes == MIB - 118
        store.store_episode(_episode(feature_elements=300000))
        assert store.remaining_memory_bytes == MIB - 236
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ("memory", "mmap"))
def test_memory_budget_rejects_before_write_and_before_episode_mutation(
    tmp_path, monkeypatch, mode
):
    _free(monkeypatch, 8 * 1024**3)
    episode = _episode(rgb_elements=MIB + 1)
    original = episode.patches
    with RolloutStorage(tmp_path, mode=mode, memory_budget_mib=1) as store:
        with pytest.raises(MemoryError, match="memory budget"):
            store.store_episode(episode)
        assert episode.patches is original
        assert store.logical_array_bytes == store.disk_bytes == 0
        assert not list(tmp_path.rglob("*.npy"))
    assert not list(tmp_path.iterdir())


def test_disk_budget_accounts_header_and_reserve_before_writing(tmp_path, monkeypatch):
    _free(monkeypatch, 8 * 1024**3)
    with RolloutStorage(tmp_path, mode="mmap", disk_budget_mib=16) as store:
        with pytest.raises(OSError, match="disk budget"):
            store.store_episode(_episode())
        assert store.disk_bytes == 0
        assert not list(store.directory.iterdir())
    with RolloutStorage(tmp_path, mode="mmap", disk_budget_mib=17) as store:
        store.store_episode(_episode(feature_elements=200000))
        used = store.disk_bytes
        with pytest.raises(OSError, match="disk budget"):
            store.store_episode(_episode(feature_elements=200000))
        assert store.disk_bytes == used
        assert len(list(store.directory.iterdir())) == 1


def test_free_disk_floor_is_at_least_two_gib_and_includes_write_reserve(tmp_path, monkeypatch):
    _free(monkeypatch, MINIMUM_DISK_FREE_BYTES + DISK_RESERVE_BYTES + 1)
    with RolloutStorage(tmp_path, mode="mmap", minimum_free_disk_mib=1) as store:
        with pytest.raises(OSError, match="free disk floor"):
            store.store_episode(_episode())
        assert not list(store.directory.iterdir())
    _free(monkeypatch, 3 * 1024**3)
    with RolloutStorage(tmp_path, mode="mmap", minimum_free_disk_mib=4096) as store:
        with pytest.raises(OSError, match="free disk floor"):
            store.store_episode(_episode())


def test_write_failure_cleans_partial_file_and_preserves_original_episode(tmp_path, monkeypatch):
    import carryrl.training.rollout_storage as module

    _free(monkeypatch, 8 * 1024**3)

    def fail(stream, array, **kwargs):
        stream.write(b"partial")
        raise OSError("injected disk write error")

    monkeypatch.setattr(module.np.lib.format, "write_array", fail)
    episode = _episode()
    original = episode.patches
    with RolloutStorage(tmp_path, mode="mmap") as store:
        with pytest.raises(OSError, match="injected"):
            store.store_episode(episode)
        assert episode.patches is original
        assert store.metrics()["episodes"] == 0
        assert not list(store.directory.iterdir())
    assert not list(tmp_path.iterdir())


def test_existing_file_is_never_overwritten_or_cleaned_as_owned(tmp_path, monkeypatch):
    _free(monkeypatch, 8 * 1024**3)
    store = RolloutStorage(tmp_path, mode="mmap")
    foreign = store.directory / "episode-000000.npy"
    foreign.write_bytes(b"foreign")
    with pytest.raises(FileExistsError):
        store.store_episode(_episode())
    assert foreign.read_bytes() == b"foreign"
    store.close()
    assert foreign.read_bytes() == b"foreign"
    assert store.disk_bytes == store.logical_array_bytes == 0


def test_unique_directories_unlimited_memory_and_constructor_validation(tmp_path):
    first = RolloutStorage(tmp_path, mode="mmap")
    second = RolloutStorage(tmp_path, mode="mmap")
    assert first.directory != second.directory
    assert first.remaining_memory_bytes is None
    first.close()
    assert second.directory.is_dir()
    second.close()
    with pytest.raises(ValueError, match="memory or mmap"):
        RolloutStorage(tmp_path, mode="other")
    with pytest.raises(ValueError, match="nonnegative"):
        RolloutStorage(tmp_path, memory_budget_mib=True)
