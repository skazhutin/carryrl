"""OCR diagnostics authenticate source pixels and never use playback text fields."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "read_screen_text_script", Path(__file__).parents[1] / "scripts/read_screen_text.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def playback(tmp_path, *, count=4):
    frames = [np.full((8, 10, 3), index, dtype=np.uint8) for index in range(count)]
    sprite = Image.new("RGB", (20, 8 * ((count + 1) // 2)))
    for index, frame in enumerate(frames):
        sprite.paste(Image.fromarray(frame), (index % 2 * 10, index // 2 * 8))
    sprite_path = tmp_path / "frames.png"
    sprite.save(sprite_path)
    data = {"sprite_filename": sprite_path.name,
        "sprite_sha256": hashlib.sha256(sprite_path.read_bytes()).hexdigest(),
        "frame_width": 10, "frame_height": 8, "frame_count": count, "sprite_columns": 2,
        "frame_sha256": [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames],
        "instruction": "SECRET game text must not enter OCR", "outcome": {"score": 999}}
    path = tmp_path / "playback.json"
    path.write_text(json.dumps(data))
    return path, sprite_path, frames


@pytest.fixture
def fake_reader(monkeypatch):
    calls = []

    class Reader:
        available_languages = ("eng",)
        engine_version = "test"

        def __init__(self, **kwargs):
            assert kwargs["language"] == "eng"

        def read(self, frame, *, step):
            calls.append((frame.copy(), step))
            return SimpleNamespace(to_dict=lambda: {"status": "ok", "blocks": [{
                "text": str(int(frame[0, 0, 0])), "role": "unknown"}], "step": step})

        def statistics(self):
            return {"requests": len(calls)}

    monkeypatch.setattr(script, "ScreenTextReader", Reader)
    monkeypatch.setattr(script, "provenance", lambda: {})
    return calls


def test_cli_reads_authenticated_actual_frames_and_reports_separate_ocr_output(
        tmp_path, fake_reader):
    path, sprite, frames = playback(tmp_path)
    before = {p: p.read_bytes() for p in (path, sprite)}
    output = tmp_path / "ocr"
    assert script.main(["--input", str(path), "--output", str(output), "--stride", "2"]) == 0
    assert [step for _, step in fake_reader] == [0, 2, 3]
    for frame, step in fake_reader:
        np.testing.assert_array_equal(frame, frames[step])
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert [row["screen_text"]["blocks"][0]["text"] for row in rows] == ["0", "2", "3"]
    assert all("SECRET" not in str(row["screen_text"]) for row in rows)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed" and report["frames"] == 3
    assert report["training_updates"] == 0
    assert not report["policy_input_changed"] and not report["reward_input_changed"]
    assert {p: p.read_bytes() for p in before} == before


def test_cli_accepts_a_single_rgb_image(tmp_path, fake_reader):
    path = tmp_path / "image.png"
    Image.new("RGB", (12, 8), (7, 8, 9)).save(path)
    output = tmp_path / "ocr"
    assert script.main(["--input", str(path), "--output", str(output)]) == 0
    assert len(fake_reader) == 1 and fake_reader[0][1] == 0
    assert tuple(fake_reader[0][0][0, 0]) == (7, 8, 9)


@pytest.mark.parametrize("corruption", ["sprite_sha256", "frame_sha256", "sprite_filename"])
def test_corrupt_playback_is_never_reported_completed(tmp_path, fake_reader, corruption):
    path, _, _ = playback(tmp_path)
    data = json.loads(path.read_text())
    data[corruption] = (["changed"] * 4 if corruption == "frame_sha256" else
                        "../outside.png" if corruption == "sprite_filename" else "changed")
    path.write_text(json.dumps(data))
    output = tmp_path / "ocr"
    with pytest.raises(ValueError):
        script.main(["--input", str(path), "--output", str(output)])
    if output.exists():
        assert json.loads((output / "status.json").read_text())["status"] == "failed"
        assert not (output / "report.json").exists()
    assert not fake_reader


def test_frame_budget_rejects_before_starting_ocr(tmp_path, monkeypatch):
    path, _, _ = playback(tmp_path)
    monkeypatch.setattr(script, "ScreenTextReader", lambda **kw: pytest.fail("OCR started"))
    output = tmp_path / "ocr"
    with pytest.raises(ValueError, match="frame budget"):
        script.main(["--input", str(path), "--output", str(output), "--max-frames", "2"])
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    output = tmp_path / "ocr"
    output.mkdir()
    (output / "keep.txt").write_text("user data")
    monkeypatch.setattr(script, "ScreenTextReader", lambda **kw: pytest.fail("OCR started"))
    with pytest.raises(FileExistsError):
        script.main(["--input", "unused", "--output", str(output)])
    assert (output / "keep.txt").read_text() == "user data"
