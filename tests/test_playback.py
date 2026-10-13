"""Archived gameplay preserves frame/action alignment and source integrity."""

import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from carryrl.dashboard import playback as playback_module
from carryrl.dashboard.data import RunReader
from carryrl.dashboard.playback import PlaybackCache, save_playback


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def episode(tmp_path):
    run = tmp_path / "runs" / "pilot"
    path = run / "trajectories" / "rollout_0011" / "navigation_3.npz"
    path.parent.mkdir(parents=True)
    frames = np.zeros((2, 8, 12, 3), dtype=np.uint8)
    frames[0, ..., 0] = 200
    frames[1, ..., 1] = 200
    final = np.zeros((8, 12, 3), dtype=np.uint8)
    final[..., 2] = 200
    np.savez_compressed(path, frames=frames, final_frame=final)
    actions = [{"keys": [key], "mouse_buttons": [], "dx": index, "dy": 0, "wheel": 0}
               for index, key in enumerate(("w", "d"))]
    write_json(path.with_suffix(".json"), {"task": "navigation", "actions": actions,
                                          "collection_weights_revision": 20, "seed": 9,
                                          "npz_sha256": hashlib.sha256(path.read_bytes())
                                          .hexdigest()})
    return run, path, tmp_path / "playback.gif"


def test_gif_contains_pre_action_frames_then_final_and_true_policy_metadata(episode):
    run, _, _ = episode
    reader = RunReader(run.parent)
    expected = reader.frame_metadata("pilot", "navigation")
    playback = PlaybackCache().get(reader, "pilot", "navigation", expected=expected)
    assert playback.metadata["collection_weights_revision"] == 20
    assert playback.metadata["frame_count"] == 3
    assert playback.metadata["steps"] == 2
    assert playback.metadata["policy"] == "stochastic_training"
    assert playback.metadata["frame_alignment"] == "frame t precedes action t"
    with Image.open(io.BytesIO(playback.encoded_gif)) as image:
        assert image.n_frames == 3
        colors = []
        footers = []
        for index in range(image.n_frames):
            image.seek(index)
            assert image.info["duration"] == 100
            rgb = image.convert("RGB")
            colors.append(rgb.getpixel((image.width // 2, 0)))
            footers.append(rgb.crop((0, image.height - 60, image.width, image.height)).tobytes())
    assert colors == [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
    assert len(set(footers)) == 3  # Timing/action/final labels match each distinct transition.


def test_playback_cache_avoids_reloading_archive_and_saving_preserves_source(episode, monkeypatch):
    run, path, output = episode
    original = path.read_bytes(), path.with_suffix(".json").read_bytes()
    reader, cache = RunReader(run.parent), PlaybackCache()
    first = cache.get(reader, "pilot", "navigation")
    monkeypatch.setattr(playback_module, "_rgb_array", lambda *args: pytest.fail("cache missed"))
    assert cache.get(reader, "pilot", "navigation") is first
    monkeypatch.undo()
    saved = save_playback(reader, "pilot", "navigation", output)
    assert output.read_bytes() == saved.encoded_gif
    assert (path.read_bytes(), path.with_suffix(".json").read_bytes()) == original
    with pytest.raises(FileExistsError):
        save_playback(reader, "pilot", "navigation", output)


def test_playback_rejects_unmatched_archive_checksum(episode):
    run, path, _ = episode
    path.write_bytes(path.read_bytes() + b"concurrent modification")
    with pytest.raises(ValueError, match="SHA256"):
        PlaybackCache().get(RunReader(run.parent), "pilot", "navigation")


def test_playback_rejects_frame_action_mismatch_even_with_matching_checksum(episode):
    run, path, _ = episode
    with np.load(path, allow_pickle=False) as data:
        np.savez_compressed(path, frames=data["frames"][:1], final_frame=data["final_frame"])
    sidecar = json.loads(path.with_suffix(".json").read_text())
    sidecar["npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(path.with_suffix(".json"), sidecar)
    with pytest.raises(ValueError, match="do not align"):
        PlaybackCache().get(RunReader(run.parent), "pilot", "navigation")


@pytest.mark.parametrize("limit", ["MAX_ARCHIVE_BYTES", "MAX_ARRAY_BYTES", "MAX_FRAMES"])
def test_playback_caps_bound_decoding(episode, monkeypatch, limit):
    run, _, _ = episode
    monkeypatch.setattr(playback_module, limit, 1)
    with pytest.raises(ValueError):
        PlaybackCache().get(RunReader(run.parent), "pilot", "navigation")


def test_playback_output_symlink_cannot_write_inside_source(episode):
    run, _, output = episode
    output.symlink_to(run / "intrusion.gif")
    with pytest.raises(ValueError, match="outside"):
        save_playback(RunReader(run.parent), "pilot", "navigation", output)
    assert not (run / "intrusion.gif").exists()


def test_selection_change_returns_no_mismatched_clip(episode):
    run, _, _ = episode
    reader = RunReader(run.parent)
    expected = reader.frame_metadata("pilot", "navigation") | {"step": 99}
    assert PlaybackCache().get(reader, "pilot", "navigation", expected=expected) is None
