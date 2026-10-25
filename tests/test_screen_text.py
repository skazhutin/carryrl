"""Local OCR is explicit about provenance, resource bounds and failures."""

import hashlib
import io
import json
import subprocess
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from PIL import Image

from carryrl.perception import (
    OCRExecutionError,
    OCRUnavailableError,
    ScreenTextReader,
    format_model_text,
)

HEADER = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
          "left\ttop\twidth\theight\tconf\ttext")
PAGE = "1\t1\t0\t0\t0\t0\t0\t0\t384\t384\t-1\t"
WORD = "5\t1\t1\t1\t1\t1\t40\t80\t24\t20\t95.5\tScore:"


def tsv(*rows):
    return ("\n".join((HEADER, *rows)) + "\n").encode()


@pytest.fixture
def engine(monkeypatch):
    state = {"calls": [], "output": tsv(PAGE, WORD), "error": None}

    def run(argv, **kwargs):
        state["calls"].append((argv, kwargs))
        assert kwargs["shell"] is False and kwargs["check"] is False
        assert kwargs["timeout"] > 0 and kwargs["capture_output"] is True
        if "--version" in argv:
            output = b"tesseract 5.5.3\n leptonica-1.87.0\n"
        elif "--list-langs" in argv:
            output = b'List of available languages in "/local/tessdata/" (3):\neng\nrus\nosd\n'
        else:
            if state["error"] is not None:
                raise state["error"]
            output = state["output"]
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr=b"")

    monkeypatch.setattr("carryrl.perception.screen_text.subprocess.run", run)
    return state


def frame(size=(96, 96)):
    return np.full((*size, 3), 140, dtype=np.uint8)


def test_structured_lines_original_boxes_numbers_unicode_and_confidence(engine):
    engine["output"] = tsv(PAGE,
        "5\t1\t1\t1\t2\t1\t12\t120\t40\t16\t88\tПривет",
        "5\t1\t1\t1\t1\t2\t80\t80\t20\t20\t81.25\t0007",
        WORD,
        '5\t1\t1\t1\t2\t2\t56\t120\t30\t16\t90\t"мир"')
    rgb = frame()
    before = rgb.tobytes()
    reader = ScreenTextReader(language="eng+rus")
    result = reader.read(rgb, step=18)
    assert result.status == "ok" and result.step == 18
    assert result.source_width == result.source_height == 96
    assert result.frame_sha256 == hashlib.sha256(before).hexdigest()
    assert result.engine == "tesseract" and result.engine_version == "5.5.3"
    assert result.language == "eng+rus"
    assert result.settings.ocr_width == result.settings.ocr_height == 384
    assert reader.available_languages == ("eng", "rus", "osd")
    assert [block.text for block in result.blocks] == ['Score: 0007', 'Привет "мир"']
    assert result.blocks[0].confidence == 0.8125
    assert result.blocks[0].bbox == (10, 20, 25, 25)
    assert result.blocks[1].bbox == (3, 30, 22, 34)
    assert all(block.role == "unknown" for block in result.blocks)
    assert rgb.tobytes() == before
    assert json.loads(json.dumps(result.to_dict(), ensure_ascii=False))["blocks"][1]["text"] == (
        'Привет "мир"')
    assert result.to_dict()["blocks"][0]["bbox"] == [10, 20, 25, 25]
    with pytest.raises(FrozenInstanceError):
        result.step = 19
    with pytest.raises(FrozenInstanceError):
        result.blocks[0].text = "changed"


def test_png_stdin_safe_argv_and_two_probes_only(engine):
    reader = ScreenTextReader(executable="/local/bin/tesseract", timeout_seconds=2)
    reader.read(frame())
    reader.read(frame(), step=2)
    assert len(engine["calls"]) == 3
    argv, kwargs = engine["calls"][-1]
    assert argv == ["/local/bin/tesseract", "stdin", "stdout", "-l", "eng",
                    "--psm", "11", "tsv"]
    with Image.open(io.BytesIO(kwargs["input"])) as image:
        assert image.format == "PNG" and image.mode == "RGB" and image.size == (384, 384)
    assert reader.statistics() == {"requests": 2, "cache_hits": 1, "ocr_calls": 1,
                                   "cache_entries": 1}


def test_cache_updates_step_and_keeps_original_result_and_serialization_isolated(engine):
    reader = ScreenTextReader()
    rgb = frame()
    first = reader.read(rgb, step=7)
    second = reader.read(rgb, step=8)
    third = reader.read(rgb)
    assert (first.step, second.step, third.step) == (7, 8, None)
    assert first.blocks == second.blocks == third.blocks
    assert first.frame_sha256 == second.frame_sha256
    mutable = first.to_dict()
    mutable["blocks"][0]["text"] = "forged"
    mutable["settings"]["ocr_width"] = 1
    assert reader.read(rgb).blocks[0].text == "Score:"
    assert reader.read(rgb).settings.ocr_width == 384
    assert reader.statistics()["ocr_calls"] == 1
    rgb[50, 50] ^= 1
    assert reader.read(rgb).frame_sha256 != first.frame_sha256
    assert reader.statistics()["ocr_calls"] == 2


def test_cache_is_lru_bounded_and_can_be_disabled(engine):
    reader = ScreenTextReader(cache_entries=2)
    first, second, third = frame(), frame(), frame()
    second[0, 0] = 1
    third[0, 0] = 2
    reader.read(first)
    reader.read(second)
    reader.read(first)  # Refresh recency, so second is evicted.
    reader.read(third)
    reader.read(first)
    reader.read(second)
    assert reader.statistics() == {"requests": 6, "cache_hits": 2, "ocr_calls": 4,
                                   "cache_entries": 2}
    uncached = ScreenTextReader(cache_entries=0)
    uncached.read(first)
    uncached.read(first)
    assert uncached.statistics() == {"requests": 2, "cache_hits": 0, "ocr_calls": 2,
                                     "cache_entries": 0}


def test_immutable_noncontiguous_rgb_is_accepted_without_mutation(engine):
    rgb = frame((96, 192))[:, ::2]
    assert not rgb.flags.c_contiguous
    rgb.setflags(write=False)
    expected = rgb.tobytes()
    result = ScreenTextReader().read(rgb)
    assert result.frame_sha256 == hashlib.sha256(expected).hexdigest()
    assert rgb.tobytes() == expected and not rgb.flags.writeable


@pytest.mark.parametrize("shape", [(96, 96), (2, 4608)])
def test_identical_bytes_with_different_geometry_never_share_cache(engine, shape):
    engine["output"] = tsv()
    reader = ScreenTextReader()
    first = reader.read(frame())
    reshaped = reader.read(frame(shape))
    assert first.frame_sha256 == reshaped.frame_sha256
    assert reader.statistics()["ocr_calls"] == (1 if shape == (96, 96) else 2)
    assert (reshaped.source_height, reshaped.source_width) == shape


def test_resize_pixel_cap_and_bbox_rounding_use_actual_canvas(engine):
    engine["output"] = tsv("5\t1\t1\t1\t1\t1\t1\t1\t31\t15\t0\t  003.50  ")
    reader = ScreenTextReader(scale=4, max_ocr_pixels=512)
    result = reader.read(frame((50, 100)))
    assert (result.settings.ocr_width, result.settings.ocr_height) == (32, 16)
    assert result.settings.ocr_width * result.settings.ocr_height <= 512
    assert result.blocks[0].bbox == (3, 3, 100, 50)
    assert result.blocks[0].text == "003.50" and result.blocks[0].confidence == 0
    assert result.blocks[0].role == "unknown"  # A numeric string has no assigned reward meaning.


def test_no_text_is_a_successful_empty_tsv_not_an_execution_failure(engine):
    engine["output"] = tsv(PAGE)
    reader = ScreenTextReader()
    result = reader.read(frame())
    assert result.status == "no_text" and result.blocks == ()
    engine["output"] = b""
    with pytest.raises(OCRExecutionError, match="header"):
        reader.read(frame() ^ 1)
    assert reader.statistics()["ocr_calls"] == 2
    assert reader.statistics()["cache_entries"] == 1


@pytest.mark.parametrize("output", [b"", b"text only\n", b"\xff", tsv("5\t1"),
    tsv(WORD + "\textra-column"), tsv(WORD.replace("95.5", "nan")),
    tsv(WORD.replace("95.5", "inf")),
    tsv(WORD.replace("95.5", "-0.5")), tsv(WORD.replace("95.5", "101")),
    tsv(WORD.replace("95.5", "-1")), tsv(WORD.replace("\t40\t80", "\t-1\t80")),
    tsv(WORD.replace("\t40\t80", "\t400\t80")),
    tsv(WORD.replace("\t24\t20", "\t0\t20")),
    tsv(WORD.replace("\t24\t20", "\t24.5\t20")),
    tsv(WORD.replace("5\t1\t1", "6\t1\t1", 1)), tsv(WORD, WORD)])
def test_malformed_tsv_never_becomes_no_text(engine, output):
    engine["output"] = output
    reader = ScreenTextReader()
    with pytest.raises(OCRExecutionError):
        reader.read(frame())
    assert reader.statistics()["cache_entries"] == 0


def test_timeout_is_explicit_and_failed_calls_are_not_cached(engine):
    engine["error"] = subprocess.TimeoutExpired("tesseract", 2)
    reader = ScreenTextReader(timeout_seconds=2)
    with pytest.raises(OCRExecutionError, match="timed out"):
        reader.read(frame())
    engine["error"] = None
    assert reader.read(frame()).status == "ok"
    assert reader.statistics()["ocr_calls"] == 2


def test_nonzero_exit_is_an_explicit_execution_error(engine, monkeypatch):
    reader = ScreenTextReader()
    monkeypatch.setattr("carryrl.perception.screen_text.subprocess.run", lambda *a, **kw:
        subprocess.CompletedProcess(a, 3, stdout=b"", stderr=b"engine diagnostic"))
    with pytest.raises(OCRExecutionError, match="status 3: engine diagnostic"):
        reader.read(frame())


def test_missing_binary_and_missing_explicit_language_fail_closed(monkeypatch, engine):
    with pytest.raises(OCRUnavailableError, match="deu.*available: eng, rus, osd"):
        ScreenTextReader(language="eng+deu")
    assert len(engine["calls"]) == 2

    def missing(*args, **kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr("carryrl.perception.screen_text.subprocess.run", missing)
    with pytest.raises(OCRUnavailableError, match="executable unavailable"):
        ScreenTextReader()


@pytest.mark.parametrize("kwargs", [{"timeout_seconds": 0}, {"timeout_seconds": float("nan")},
    {"timeout_seconds": True}, {"scale": -1}, {"scale": float("inf")},
    {"max_ocr_pixels": 0}, {"max_ocr_pixels": True}, {"cache_entries": -1},
    {"cache_entries": False}, {"language": "eng+"}, {"language": ""}, {"executable": ""}])
def test_invalid_configuration_is_rejected_before_subprocess(engine, kwargs):
    with pytest.raises(ValueError):
        ScreenTextReader(**kwargs)
    assert not engine["calls"]


@pytest.mark.parametrize("rgb", [np.zeros((2, 2, 4), np.uint8), np.zeros((2, 2), np.uint8),
    np.zeros((2, 2, 3), float), np.zeros((0, 2, 3), np.uint8), [[[1, 2, 3]]]])
def test_non_rgb_inputs_are_rejected_without_ocr(engine, rgb):
    reader = ScreenTextReader()
    with pytest.raises(ValueError, match="uint8 HWC RGB"):
        reader.read(rgb)
    assert reader.statistics()["ocr_calls"] == 0


@pytest.mark.parametrize("step", [-1, True, 1.5, "1"])
def test_invalid_source_steps_are_rejected(engine, step):
    reader = ScreenTextReader()
    with pytest.raises(ValueError, match="step"):
        reader.read(frame(), step=step)
    assert reader.statistics()["ocr_calls"] == 0


def test_model_text_is_bounded_and_retains_literal_spatial_observations(engine):
    engine["output"] = tsv(WORD, "5\t1\t1\t1\t2\t1\t40\t120\t24\t20\t90\t42")
    result = ScreenTextReader().read(frame(), step=12)
    complete = json.loads(format_model_text(result))
    assert complete["blocks"][0] == result.blocks[0].to_dict()
    assert complete["blocks"][1]["text"] == "42"
    assert complete["step"] == 12 and complete["source"] == "untrusted_screen_ocr"
    assert complete["omitted_blocks"] == 0
    bounded = format_model_text(result, max_chars=180)
    assert len(bounded) <= 180
    assert json.loads(bounded)["omitted_blocks"] > 0
    with pytest.raises(ValueError, match="max_chars"):
        format_model_text(result, max_chars=3)


@pytest.mark.parametrize("problem", ["version_identity", "version_exit", "language_inventory",
                                     "probe_timeout", "permission"])
def test_startup_probe_failure_is_unavailable_not_empty_ocr(monkeypatch, problem):
    def run(argv, **kwargs):
        if problem == "probe_timeout":
            raise subprocess.TimeoutExpired(argv, 1)
        if problem == "permission":
            raise PermissionError("binary not executable")
        if "--version" in argv:
            output = b"not tesseract\n" if problem == "version_identity" else b"tesseract 5.5.3\n"
            return subprocess.CompletedProcess(argv, int(problem == "version_exit"),
                                               stdout=output, stderr=b"probe failed")
        return subprocess.CompletedProcess(argv, 0, stdout=b"broken language inventory", stderr=b"")

    monkeypatch.setattr("carryrl.perception.screen_text.subprocess.run", run)
    with pytest.raises(OCRUnavailableError):
        ScreenTextReader()


def test_unrepresentable_aspect_ratio_budget_fails_before_ocr(engine):
    reader = ScreenTextReader(max_ocr_pixels=1)
    with pytest.raises(ValueError, match="aspect ratio"):
        reader.read(frame((1, 100)))
    assert reader.statistics()["ocr_calls"] == 0


def test_extreme_numeric_config_is_rejected_before_probe(engine):
    with pytest.raises(ValueError, match="positive finite"):
        ScreenTextReader(scale=10**1000)
    assert not engine["calls"]
