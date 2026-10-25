"""Read screen text from RGB images or authenticated lossless frozen playbacks."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from carryrl.perception.screen_text import ScreenTextReader
from carryrl.training.run import atomic_json, file_sha256, provenance

MAX_SOURCE_PIXELS = 16_000_000


def _rgb_image(path):
    with Image.open(path) as image:
        if image.width * image.height > MAX_SOURCE_PIXELS:
            raise ValueError("screen text source exceeds the image pixel budget")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("use a lossless playback JSON instead of an animated image")
        return image.convert("RGB")


def _sprite_path(path, data):
    filename = data.get("sprite_filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("playback sprite must be a sibling filename")
    return path.parent / filename


def _source_frames(path, stride):
    """Yield actual pixels, never text fields or hidden metadata as OCR input."""
    if path.suffix.lower() != ".json":
        rgb = np.asarray(_rgb_image(path), dtype=np.uint8)
        yield 0, rgb
        return
    data = json.loads(path.read_text())
    sprite = _sprite_path(path, data)
    if file_sha256(sprite) != data.get("sprite_sha256"):
        raise ValueError("playback sprite SHA256 mismatch")
    width, height, count, columns = (data.get(key) for key in
        ("frame_width", "frame_height", "frame_count", "sprite_columns"))
    if any(type(value) is not int or value < 1 for value in (width, height, count, columns)):
        raise ValueError("playback frame dimensions/count must be positive integers")
    hashes = data.get("frame_sha256")
    if not isinstance(hashes, list) or len(hashes) != count:
        raise ValueError("playback requires a SHA256 for every lossless frame")
    image = _rgb_image(sprite)
    if image.size != (columns * width, math.ceil(count / columns) * height):
        raise ValueError("playback sprite dimensions differ from its frame declaration")
    indices = list(range(0, count, stride))
    if count - 1 not in indices:
        indices.append(count - 1)
    for index in indices:
        x, y = index % columns * width, index // columns * height
        rgb = np.asarray(image.crop((x, y, x + width, y + height)), dtype=np.uint8)
        if hashlib.sha256(rgb.tobytes()).hexdigest() != hashes[index]:
            raise ValueError("playback frame SHA256 mismatch")
        yield index, rgb


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="eng")
    parser.add_argument("--executable", default="tesseract")
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=256)
    args = parser.parse_args(argv)
    if args.stride < 1 or args.max_frames < 1:
        parser.error("stride and max-frames must be positive")
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("screen text output already exists")
    paths = [path.resolve(strict=True) for path in args.input]
    source_hashes = {str(path): file_sha256(path) for path in paths}
    expected_frames = 0
    for path in paths:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
            sprite = _sprite_path(path, data)
            source_hashes[str(sprite)] = file_sha256(sprite)
            count = data.get("frame_count")
            if type(count) is not int or count < 1:
                raise ValueError("playback frame_count must be a positive integer")
            expected_frames += (count - 1) // args.stride + 1 + int(
                (count - 1) % args.stride != 0)
        else:
            expected_frames += 1
    if expected_frames > args.max_frames:
        raise ValueError("selected inputs exceed the OCR frame budget")
    reader = ScreenTextReader(language=args.language, executable=args.executable,
        scale=args.scale, timeout_seconds=args.timeout_seconds)
    output.mkdir(parents=True, exist_ok=False)
    processed, with_text = 0, 0
    metadata = {"kind": "screen_text_diagnostics", "training_updates": 0,
        "policy_input_changed": False, "reward_input_changed": False,
        "sources_sha256": source_hashes, "language": args.language,
        "available_languages": reader.available_languages, "engine_version": reader.engine_version,
        "stride": args.stride, "max_frames": args.max_frames,
        "interpretation": "OCR from actual RGB only. Text roles, reward and key meanings "
                          "are not inferred; recognition accuracy is not established by output.",
        **provenance()}
    atomic_json(output / "metadata.json", metadata)
    atomic_json(output / "status.json", {"status": "running", "training_updates": 0})
    try:
        with (output / "observations.jsonl").open("x", encoding="utf-8") as stream:
            for path in paths:
                for index, frame in _source_frames(path, args.stride):
                    if processed == args.max_frames:
                        raise ValueError("selected inputs exceed the OCR frame budget")
                    result = reader.read(frame, step=index).to_dict()
                    row = {"source": str(path), "source_sha256": source_hashes[str(path)],
                           "frame_index": index, "screen_text": result}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    processed += 1
                    with_text += int(result["status"] == "ok")
        if any(file_sha256(Path(path)) != expected for path, expected in source_hashes.items()):
            raise RuntimeError("OCR source changed during reading")
        report = {"status": "completed", "training_updates": 0, "frames": processed,
            "frames_with_text": with_text, "frames_without_text": processed - with_text,
            "reader": reader.statistics(), "source_files_unchanged": True,
            "observations_sha256": file_sha256(output / "observations.jsonl"),
            "policy_input_changed": False, "reward_input_changed": False}
        atomic_json(output / "report.json", report)
        atomic_json(output / "status.json", report)
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "interrupted" if isinstance(
            error, KeyboardInterrupt) else "failed", "training_updates": 0,
            "frames_processed": processed, "error_type": type(error).__name__,
            "error": str(error)})
        raise
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
