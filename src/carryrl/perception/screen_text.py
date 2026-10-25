"""Local RGB OCR with explicit provenance and uninterpreted, spatial text lines."""

import hashlib
import io
import json
import math
import re
import subprocess
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace

import numpy as np
from PIL import Image

_TSV_HEADER = (
    "level", "page_num", "block_num", "par_num", "line_num", "word_num",
    "left", "top", "width", "height", "conf", "text",
)
_PSM = 11  # Sparse screenshot text; line/word ordering remains Tesseract's output.


class OCRUnavailableError(RuntimeError):
    """The local OCR executable or explicitly requested language is unavailable."""


class OCRExecutionError(RuntimeError):
    """OCR timed out, failed, or returned invalid structured output."""


@dataclass(frozen=True)
class ScreenTextBlock:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    role: str = "unknown"

    def to_dict(self):
        return {**asdict(self), "bbox": list(self.bbox)}


@dataclass(frozen=True)
class ScreenTextSettings:
    scale: float
    max_ocr_pixels: int
    ocr_width: int
    ocr_height: int
    page_segmentation_mode: int = _PSM
    resize_filter: str = "lanczos"


@dataclass(frozen=True)
class ScreenTextResult:
    status: str
    blocks: tuple[ScreenTextBlock, ...]
    frame_sha256: str
    step: int | None
    source_width: int
    source_height: int
    engine: str
    engine_version: str
    language: str
    settings: ScreenTextSettings

    def to_dict(self):
        return {**asdict(self), "blocks": [block.to_dict() for block in self.blocks]}


def _positive_number(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _decode(output, exception, purpose):
    try:
        return output.decode("utf-8")
    except (UnicodeError, AttributeError) as error:
        raise exception(f"Tesseract {purpose} was not UTF-8 text") from error


def _parse_tsv(output, *, source_size, ocr_size):
    lines = _decode(output, OCRExecutionError, "TSV").splitlines()
    if not lines or tuple(lines[0].split("\t")) != _TSV_HEADER:
        raise OCRExecutionError("Tesseract TSV has a missing or invalid header")
    source_width, source_height = source_size
    ocr_width, ocr_height = ocr_size
    grouped, seen = {}, set()
    for row_number, line in enumerate(lines[1:], start=2):
        if not line:
            continue
        columns = line.split("\t")
        if len(columns) != len(_TSV_HEADER):
            raise OCRExecutionError(f"Tesseract TSV row {row_number} has missing or extra columns")
        try:
            level, page, block, paragraph, line_id, word, left, top, width, height = (
                int(value) for value in columns[:10])
            confidence = float(columns[10])
        except (ValueError, OverflowError) as error:
            raise OCRExecutionError(
                f"Tesseract TSV row {row_number} has invalid numbers") from error
        if (level not in range(1, 6) or page < 1
                or min(block, paragraph, line_id, word, left, top, width, height) < 0
                or not math.isfinite(confidence)
                or not (confidence == -1 or 0 <= confidence <= 100)
                or left + width > ocr_width or top + height > ocr_height):
            raise OCRExecutionError(f"Tesseract TSV row {row_number} has invalid bounds/confidence")
        if level != 5:
            continue  # Page/block/paragraph/line containers have no recognized word.
        if min(block, paragraph, line_id, word) < 1 or confidence < 0:
            raise OCRExecutionError(f"Tesseract TSV row {row_number} has invalid word metadata")
        text = columns[11].strip()
        if not text:
            continue
        if width == 0 or height == 0:
            raise OCRExecutionError(f"Tesseract TSV row {row_number} has an empty word box")
        key = page, block, paragraph, line_id
        if (*key, word) in seen:
            raise OCRExecutionError(f"Tesseract TSV row {row_number} duplicates a word identity")
        seen.add((*key, word))
        grouped.setdefault(key, []).append((word, text, confidence / 100,
                                            (left, top, left + width, top + height)))
    blocks = []
    for key in sorted(grouped):
        words = sorted(grouped[key])
        left = min(word[3][0] for word in words)
        top = min(word[3][1] for word in words)
        right = max(word[3][2] for word in words)
        bottom = max(word[3][3] for word in words)
        # Floor/ceil contain every OCR pixel, including rounding of the resized canvas.
        bbox = (left * source_width // ocr_width, top * source_height // ocr_height,
                math.ceil(right * source_width / ocr_width),
                math.ceil(bottom * source_height / ocr_height))
        blocks.append(ScreenTextBlock(" ".join(word[1] for word in words),
                                      min(word[2] for word in words), bbox))
    return tuple(blocks)


class ScreenTextReader:
    """Bounded exact-frame OCR cache; only initial startup probes inspect the engine."""

    def __init__(self, language="eng", executable="tesseract", timeout_seconds=10.0,
                 scale=4.0, max_ocr_pixels=4_000_000, cache_entries=128):
        if (not isinstance(language, str) or not language
                or any(not part or part.strip() != part for part in language.split("+"))):
            raise ValueError("language must name one or more explicit Tesseract languages")
        if not isinstance(executable, str) or not executable:
            raise ValueError("executable must be a nonempty path or command name")
        self._timeout = _positive_number(timeout_seconds, "timeout_seconds")
        self._scale = _positive_number(scale, "scale")
        if type(max_ocr_pixels) is not int or max_ocr_pixels < 1:
            raise ValueError("max_ocr_pixels must be a positive integer")
        if type(cache_entries) is not int or cache_entries < 0:
            raise ValueError("cache_entries must be a nonnegative integer")
        self._language, self._executable = language, executable
        self._max_pixels, self._cache_limit = max_ocr_pixels, cache_entries
        self._cache = OrderedDict()
        self._requests = self._cache_hits = self._ocr_calls = 0
        version = _decode(self._run(["--version"], probe=True), OCRUnavailableError, "version")
        first_line = version.splitlines()[0] if version.splitlines() else ""
        match = re.fullmatch(r"tesseract\s+(\S+)", first_line.strip())
        if match is None:
            raise OCRUnavailableError("Tesseract returned an invalid version identity")
        self._engine_version = match[1]
        inventory = _decode(self._run(["--list-langs"], probe=True),
                            OCRUnavailableError, "language inventory").splitlines()
        count = re.search(r"\((\d+)\):$", inventory[0]) if inventory else None
        languages = tuple(line.strip() for line in inventory[1:] if line.strip())
        if (not inventory or not inventory[0].startswith("List of available languages")
                or count is None or int(count[1]) != len(languages)):
            raise OCRUnavailableError("Tesseract returned an invalid language inventory")
        self._available_languages = languages
        missing = set(language.split("+")) - set(languages)
        if missing:
            raise OCRUnavailableError(
                f"Tesseract languages unavailable: {', '.join(sorted(missing))}; "
                f"available: {', '.join(languages) or '(none)'}")

    @property
    def available_languages(self):
        return self._available_languages

    @property
    def engine_version(self):
        return self._engine_version

    def _run(self, arguments, *, input_bytes=None, probe=False):
        exception = OCRUnavailableError if probe else OCRExecutionError
        try:
            result = subprocess.run([self._executable, *arguments], input=input_bytes,
                                    capture_output=True, timeout=self._timeout, check=False,
                                    shell=False)
        except FileNotFoundError as error:
            raise OCRUnavailableError(f"Tesseract executable unavailable: {self._executable}") \
                from error
        except subprocess.TimeoutExpired as error:
            raise exception(f"Tesseract timed out after {self._timeout:g} seconds") from error
        except OSError as error:
            raise exception(f"Tesseract could not execute: {error}") from error
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace").strip()[:500]
            raise exception(f"Tesseract exited with status {result.returncode}: {detail}")
        return result.stdout

    def read(self, frame, *, step=None):
        if step is not None and (type(step) is not int or step < 0):
            raise ValueError("step must be a nonnegative integer or None")
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[-1] != 3 or min(frame.shape[:2]) < 1):
            raise ValueError("screen text requires nonempty uint8 HWC RGB")
        height, width = frame.shape[:2]
        raw = frame.tobytes(order="C")  # Immutable snapshot; never alter the caller's RGB.
        frame_hash = hashlib.sha256(raw).hexdigest()
        self._requests += 1
        key = (frame_hash, width, height, self._language, self._scale, self._max_pixels,
               self._executable, self._engine_version, _PSM)
        if key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return replace(self._cache[key], step=step)
        factor = min(self._scale, math.sqrt(self._max_pixels / (width * height)))
        ocr_width, ocr_height = max(1, int(width * factor)), max(1, int(height * factor))
        if ocr_width * ocr_height > self._max_pixels:
            raise ValueError("OCR pixel budget is too small to preserve this frame's aspect ratio")
        image = Image.frombytes("RGB", (width, height), raw)
        if image.size != (ocr_width, ocr_height):
            image = image.resize((ocr_width, ocr_height), Image.Resampling.LANCZOS)
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        self._ocr_calls += 1
        tsv = self._run(["stdin", "stdout", "-l", self._language, "--psm", str(_PSM), "tsv"],
                        input_bytes=encoded.getvalue())
        blocks = _parse_tsv(tsv, source_size=(width, height), ocr_size=image.size)
        result = ScreenTextResult("ok" if blocks else "no_text", blocks, frame_hash, step,
                                  width, height, "tesseract", self._engine_version,
                                  self._language, ScreenTextSettings(self._scale, self._max_pixels,
                                                                  ocr_width, ocr_height))
        if self._cache_limit:
            self._cache[key] = replace(result, step=None)
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
        return result

    def statistics(self):
        """Frame requests/hits and OCR attempts (including failures); excludes startup probes."""
        return {"requests": self._requests, "cache_hits": self._cache_hits,
                "ocr_calls": self._ocr_calls, "cache_entries": len(self._cache)}


def format_model_text(result, max_chars=4096):
    """Bounded untrusted text observations; never interpret strings as rewards or commands."""
    if not isinstance(result, ScreenTextResult):
        raise TypeError("expected ScreenTextResult")
    if type(max_chars) is not int or max_chars < 128:
        raise ValueError("max_chars must be an integer >= 128")
    payload = {"source": "untrusted_screen_ocr", "step": result.step, "blocks": [],
               "omitted_blocks": len(result.blocks)}
    for block in result.blocks:
        candidate = {**payload, "blocks": [*payload["blocks"], block.to_dict()],
                     "omitted_blocks": payload["omitted_blocks"] - 1}
        if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))) > max_chars:
            break
        payload = candidate
    output = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(output) > max_chars:
        raise ValueError("max_chars cannot hold the observation envelope")
    return output
