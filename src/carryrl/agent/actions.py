"""Versioned physical keyboard/mouse actions shared by every game adapter."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

CODEC_VERSION = "carryrl.physical.v1"
KEY_VOCAB = (
    "up", "down", "left", "right", "w", "a", "s", "d", "space", "e", "f", "q", "r",
    "shift", "ctrl", "escape", "tab", "enter", "1", "2", "3", "4", "5", "6", "7", "8", "9",
)
MOUSE_BUTTONS = ("left", "right", "middle")
MOUSE_BINS = (-64, -32, -16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64)
WHEEL_BINS = (-1, 0, 1)
ACTION_FEATURE_SIZE = len(KEY_VOCAB) + len(MOUSE_BUTTONS) + 3


def _canonical(values: tuple[str, ...], vocabulary: tuple[str, ...], name: str):
    if not isinstance(values, tuple) or any(not isinstance(value, str) for value in values):
        raise TypeError(f"{name} must be a tuple of strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate entries")
    unknown = set(values).difference(vocabulary)
    if unknown:
        raise ValueError(f"unknown {name}: {sorted(unknown)}")
    present = set(values)
    return tuple(value for value in vocabulary if value in present)


@dataclass(frozen=True, slots=True)
class PhysicalAction:
    """Held input set and quantized relative mouse motion for one control interval.

    A subsequent action replaces the held set: absent keys/buttons must be released
    by an executor. Codec v1 does not represent variable duration or input macros.
    """

    keys: tuple[str, ...] = ()
    mouse_buttons: tuple[str, ...] = ()
    dx: int = 0
    dy: int = 0
    wheel: int = 0

    def __post_init__(self):
        object.__setattr__(self, "keys", _canonical(self.keys, KEY_VOCAB, "keys"))
        object.__setattr__(self, "mouse_buttons", _canonical(
            self.mouse_buttons, MOUSE_BUTTONS, "mouse_buttons",
        ))
        for name, bins in (("dx", MOUSE_BINS), ("dy", MOUSE_BINS), ("wheel", WHEEL_BINS)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a non-boolean integer")
            if value not in bins:
                raise ValueError(f"{name} must be one of {bins}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "codec_version": CODEC_VERSION,
            "keys": list(self.keys),
            "mouse_buttons": list(self.mouse_buttons),
            "dx": self.dx,
            "dy": self.dy,
            "wheel": self.wheel,
        }

    def serialize(self) -> dict[str, Any]:
        """Return a JSON-compatible record with an explicit codec version."""
        return self.to_dict()

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "PhysicalAction":
        if not isinstance(record, Mapping):
            raise TypeError("action record must be a mapping")
        fields = {"codec_version", "keys", "mouse_buttons", "dx", "dy", "wheel"}
        if set(record) != fields:
            raise ValueError("action record must contain exactly the codec v1 fields")
        if record["codec_version"] != CODEC_VERSION:
            raise ValueError(f"unsupported action codec version: {record['codec_version']!r}")
        for name in ("keys", "mouse_buttons"):
            if not isinstance(record[name], list):
                raise TypeError(f"serialized {name} must be a list")
        return cls(
            keys=tuple(record["keys"]), mouse_buttons=tuple(record["mouse_buttons"]),
            dx=record["dx"], dy=record["dy"], wheel=record["wheel"],
        )


def encode_actions(
    actions: Sequence[PhysicalAction], device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode prior own inputs as multi-hot keys/buttons and normalized mouse deltas."""
    rows = []
    for action in actions:
        if not isinstance(action, PhysicalAction):
            raise TypeError("encode_actions expects PhysicalAction entries")
        rows.append([
            *(float(key in action.keys) for key in KEY_VOCAB),
            *(float(button in action.mouse_buttons) for button in MOUSE_BUTTONS),
            action.dx / max(MOUSE_BINS), action.dy / max(MOUSE_BINS), float(action.wheel),
        ])
    return torch.tensor(rows, dtype=torch.float32, device=device).reshape(-1, ACTION_FEATURE_SIZE)
