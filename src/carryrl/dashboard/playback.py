"""Bounded RGB/action playback of saved stochastic training episodes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from carryrl.dashboard.data import RunReader

MAX_FRAMES = 128
MAX_DIMENSION = 256
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_ARRAY_BYTES = 32 * 1024 * 1024
MAX_GIF_BYTES = 8 * 1024 * 1024
FRAME_DURATION_MS = 100  # The collector's fixed 0.1 s controller/simulation interval.


@dataclass(frozen=True)
class EpisodePlayback:
    encoded_gif: bytes
    metadata: dict
    width: int
    height: int


def _rgb_array(archive, name, ndim):
    member = archive.getinfo(name)
    if member.file_size > MAX_ARRAY_BYTES:
        raise ValueError(f"Playback array {name} exceeds byte limit")
    content = archive.read(member)
    header = io.BytesIO(content)
    version = np.lib.format.read_magic(header)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(header)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(header)
    else:
        raise ValueError("Unsupported playback array format")
    if dtype != np.uint8 or len(shape) != ndim or shape[-1] != 3:
        raise ValueError("Playback requires uint8 RGB arrays")
    if min(shape[-3:-1]) < 1 or max(shape[-3:-1]) > MAX_DIMENSION:
        raise ValueError("Playback frame dimensions exceed limit")
    if ndim == 4 and not 1 <= shape[0] <= MAX_FRAMES:
        raise ValueError("Playback episode length exceeds limit")
    if math.prod(shape) > MAX_ARRAY_BYTES:
        raise ValueError("Playback array shape exceeds byte limit")
    return np.load(io.BytesIO(content), allow_pickle=False)


def _action_lines(action):
    if action is None:
        return ["FINAL: after the last action", "No next action"]
    keys = ",".join(action.get("keys", [])) or "none"
    buttons = ",".join(action.get("mouse_buttons", [])) or "none"
    return [f"keys: {keys} | buttons: {buttons}",
            f"mouse: dx {action.get('dx')} dy {action.get('dy')} | wheel {action.get('wheel')}"]


def _render_frame(rgb, metadata, index, action):
    image = Image.fromarray(rgb)
    scale = max(1, min(3, 384 // max(image.size)))
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (max(288, image.width), image.height + 60), "#141a24")
    canvas.paste(image, ((canvas.width - image.width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    lines = [f"TRAIN | rollout {metadata['rollout']} / episode {metadata['episode']}",
             f"revision {metadata['collection_weights_revision']} | t {index}/{metadata['steps']} "
             "| 10 FPS", *_action_lines(action)]
    for row, line in enumerate(lines):
        draw.text((5, image.height + 3 + row * 13), line, fill="#e3e9f0")
    return canvas


class PlaybackCache:
    """Verify an immutable archive and keep at most 16 rendered episodes in memory."""

    def __init__(self):
        self._cache = OrderedDict()

    def get(self, reader: RunReader, run_id: str, task: str, *, expected: dict | None = None):
        selected = reader.episode_source(run_id, task, expected=expected)
        if selected is None:
            return None
        metadata, path = selected
        stat = path.stat()
        actions = metadata.get("actions")
        digest = metadata.get("npz_sha256")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", digest or ""):
            raise ValueError("Playback requires an archive SHA256 in its sidecar")
        if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_FRAMES:
            raise ValueError("Playback requires one action per frame within the length limit")
        for action in actions:
            if not isinstance(action, dict) or not all(
                isinstance(action.get(field), list)
                and all(isinstance(value, str) for value in action[field])
                for field in ("keys", "mouse_buttons")
            ) or not all(isinstance(action.get(field), int) for field in ("dx", "dy", "wheel")):
                raise ValueError("Malformed physical action in playback sidecar")
        action_hash = hashlib.sha256(json.dumps(actions, sort_keys=True).encode()).hexdigest()
        metadata_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        key = (path, stat.st_mtime_ns, stat.st_size, digest, action_hash, metadata_hash)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        if stat.st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("Playback archive exceeds byte limit")
        with path.open("rb") as stream:
            data = stream.read(MAX_ARCHIVE_BYTES + 1)
        if len(data) > MAX_ARCHIVE_BYTES or hashlib.sha256(data).hexdigest() != digest.lower():
            raise ValueError("Playback archive SHA256 does not match its sidecar")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if len(archive.infolist()) > 32:
                raise ValueError("Playback archive has too many array members")
            frames = _rgb_array(archive, "frames.npy", 4)
            final = _rgb_array(archive, "final_frame.npy", 3)
        if len(frames) != len(actions) or frames.shape[1:] != final.shape:
            raise ValueError("Playback frames/actions/final frame do not align")
        # Recheck selection after reading: a newly completed episode belongs to
        # the next refresh, never to the caption of this older episode.
        confirmed = reader.episode_source(run_id, task, expected=expected)
        if confirmed is None or confirmed[1] != path or confirmed[0] != metadata:
            return None
        metadata = {key: value for key, value in metadata.items() if key != "actions"} | {
            "steps": len(actions), "frame_count": len(actions) + 1,
            "fps": 10, "frame_duration_ms": FRAME_DURATION_MS,
            "policy": "stochastic_training", "frame_alignment": "frame t precedes action t",
        }
        rendered = [_render_frame(rgb, metadata, index, actions[index])
                    for index, rgb in enumerate(frames)]
        rendered.append(_render_frame(final, metadata, len(actions), None))
        output = io.BytesIO()
        rendered[0].save(output, format="GIF", save_all=True, append_images=rendered[1:],
                         duration=FRAME_DURATION_MS, loop=0, disposal=2, optimize=False)
        gif = output.getvalue()
        if len(gif) > MAX_GIF_BYTES:
            raise ValueError("Encoded playback GIF exceeds byte limit")
        result = EpisodePlayback(gif, metadata, rendered[0].width, rendered[0].height)
        self._cache[key] = result
        if len(self._cache) > 16:
            self._cache.popitem(last=False)
        return result


def save_playback(reader: RunReader, run_id: str, task: str, output: Path | str):
    """Save a GIF outside the source tree; an existing output is never overwritten."""
    path = Path(output).resolve()
    if path == reader.root or reader.root in path.parents:
        raise ValueError("Playback output must be outside the training runs directory")
    playback = PlaybackCache().get(reader, run_id, task)
    if playback is None:
        raise ValueError("No complete saved training episode is available for this task")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(playback.encoded_gif)
    return playback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        playback = save_playback(RunReader(args.runs), args.run, args.task, args.output)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Cannot export playback: {error}\n")
    print(json.dumps({"output": str(args.output.resolve()), **playback.metadata},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
