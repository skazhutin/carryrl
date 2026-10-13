"""Episode-stable visual presentation, independent of game mechanics.

The interactive viewport is never rescaled. Its placement changes screen
coordinates by translation only, so physical mouse deltas keep their meaning.
Public HUD templates are ordinary readable UI, for independent pixel checks.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HUD_LAYOUTS = ("top", "bottom", "left", "right")
HUD_THEMES = {
    "slate": {"background": (20, 27, 39), "text": (237, 243, 251), "font_size": 10},
    "sand": {"background": (51, 39, 24), "text": (252, 238, 205), "font_size": 10},
    "forest": {"background": (18, 39, 31), "text": (224, 247, 219), "font_size": 11},
    "violet": {"background": (38, 26, 52), "text": (243, 224, 253), "font_size": 11},
    "ice": {"background": (18, 37, 48), "text": (222, 250, 255), "font_size": 12},
    "ember": {"background": (49, 24, 27), "text": (255, 232, 218), "font_size": 12},
}
THEME_SPLITS = {
    "train": ("slate", "sand", "forest"),
    "validation": ("violet",),
    "test": ("ice", "ember"),
}


@dataclass(frozen=True)
class PresentationConfig:
    mode: str = "canonical"
    native_sizes: tuple[int, ...] = (64, 96, 128)
    themes: tuple[str, ...] = tuple(HUD_THEMES)
    hud_layouts: tuple[str, ...] = HUD_LAYOUTS
    textures: bool = True
    seed: int = 0
    canvas_padding: int = 8
    style_split: str | None = None

    def __post_init__(self):
        for name in ("native_sizes", "themes", "hud_layouts"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.mode not in {"canonical", "varied"}:
            raise ValueError("Presentation mode must be canonical or varied")
        if not self.native_sizes or any(
            isinstance(size, bool) or not isinstance(size, int) or not 64 <= size <= 512
            for size in self.native_sizes
        ):
            raise ValueError("Native sizes must be integers in 64..512")
        if not self.themes or any(theme not in HUD_THEMES for theme in self.themes):
            raise ValueError("Unknown or empty presentation theme selection")
        if not self.hud_layouts or any(layout not in HUD_LAYOUTS for layout in self.hud_layouts):
            raise ValueError("Unknown or empty HUD layout selection")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Presentation seed must be a nonnegative integer")
        if (isinstance(self.canvas_padding, bool) or not isinstance(self.canvas_padding, int)
                or not 4 <= self.canvas_padding <= 64):
            raise ValueError("Canvas padding must be an integer in 4..64")
        if not isinstance(self.textures, bool):
            raise ValueError("textures must be a boolean")
        if self.style_split is not None and self.style_split not in THEME_SPLITS:
            raise ValueError("Unknown appearance split")


def hud_size(layout: str) -> tuple[int, int]:
    """Return public panel width and height, without game-state information."""
    if layout not in HUD_LAYOUTS:
        raise ValueError("Unknown HUD layout")
    return (96, 28) if layout in {"top", "bottom"} else (64, 44)


def hud_content_boxes(layout: str) -> tuple[tuple[int, int, int, int], ...]:
    """Two text-line boxes in panel-relative coordinates (x0,y0,x1,y1)."""
    width, height = hud_size(layout)
    return ((2, 1, width - 2, height // 2), (2, height // 2, width - 2, height - 1))


@lru_cache(maxsize=432)
def render_hud(theme: str, layout: str, score: int, status: str) -> np.ndarray:
    """Render readable score and status with an untouched background border."""
    if theme not in HUD_THEMES or score not in range(6) or status not in {
        "Play", "Success", "Failed"
    }:
        raise ValueError("Unsupported visible HUD theme/score/status")
    width, height = hud_size(layout)
    palette = HUD_THEMES[theme]
    image = Image.new("RGB", (width, height), palette["background"])
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=palette["font_size"])
    texts = (f"Score: {score}", status)
    for text, box in zip(texts, hud_content_boxes(layout), strict=True):
        text_width = draw.textlength(text, font=font)
        x = 4 if layout in {"top", "left"} else max(3, round((width - text_width) / 2))
        draw.text((x, box[1]), text, fill=palette["text"], font=font, anchor="lt")
    result = np.asarray(image).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PresentationProfile:
    native_size: int
    theme: str
    hud_layout: str
    texture_seed: int = 0
    textures: bool = True
    canvas_padding: int = 8

    def __post_init__(self):
        PresentationConfig(mode="varied", native_sizes=(self.native_size,), themes=(self.theme,),
                           hud_layouts=(self.hud_layout,), textures=self.textures,
                           seed=self.texture_seed, canvas_padding=self.canvas_padding)

    @property
    def canvas_size(self) -> tuple[int, int]:
        panel_width, panel_height = hud_size(self.hud_layout)
        padding = self.canvas_padding
        if self.hud_layout in {"top", "bottom"}:
            return max(self.native_size, panel_width) + 2 * padding, (
                self.native_size + panel_height + 3 * padding
            )
        return self.native_size + panel_width + 3 * padding, (
            max(self.native_size, panel_height) + 2 * padding
        )

    @property
    def viewport_box(self) -> tuple[int, int, int, int]:
        width, height = self.canvas_size
        panel_width, panel_height = hud_size(self.hud_layout)
        padding, size = self.canvas_padding, self.native_size
        x = (width - size) // 2 if self.hud_layout in {"top", "bottom"} else padding
        y = (height - size) // 2 if self.hud_layout in {"left", "right"} else padding
        if self.hud_layout == "top":
            y += panel_height + padding
        if self.hud_layout == "left":
            x += panel_width + padding
        return x, y, x + size, y + size

    @property
    def hud_box(self) -> tuple[int, int, int, int]:
        width, height = self.canvas_size
        panel_width, panel_height = hud_size(self.hud_layout)
        padding = self.canvas_padding
        x, y = (width - panel_width) // 2, padding
        if self.hud_layout == "bottom":
            y = height - padding - panel_height
        elif self.hud_layout in {"left", "right"}:
            x = padding if self.hud_layout == "left" else width - padding - panel_width
            y = (height - panel_height) // 2
        return x, y, x + panel_width, y + panel_height


def choose_presentation(
    config: PresentationConfig | dict | None,
    split: str,
    episode_seed: int,
    image_size: int = 96,
) -> PresentationProfile | None:
    """Choose appearance without consuming the level/mechanics RNG."""
    if config is None:
        return None
    if isinstance(config, dict):
        config = PresentationConfig(**config)
    if not isinstance(config, PresentationConfig):
        raise TypeError("Expected PresentationConfig, configuration dict, or None")
    if config.mode == "canonical":
        return None
    appearance_split = config.style_split or split
    if appearance_split not in THEME_SPLITS:
        raise ValueError("Unknown appearance split")
    themes = tuple(theme for theme in config.themes if theme in THEME_SPLITS[appearance_split])
    if not themes:
        raise ValueError("No configured themes belong to the requested appearance split")
    if isinstance(episode_seed, bool) or not isinstance(episode_seed, int) or episode_seed < 0:
        raise ValueError("Appearance episode seed must be a nonnegative integer")
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, episode_seed]))
    return PresentationProfile(
        native_size=int(rng.choice(config.native_sizes)),
        theme=str(rng.choice(themes)),
        hud_layout=str(rng.choice(config.hud_layouts)),
        texture_seed=int(rng.integers(2**31)),
        textures=config.textures,
        canvas_padding=config.canvas_padding,
    )


@lru_cache(maxsize=128)
def _material_maps(profile: PresentationProfile) -> tuple[np.ndarray, np.ndarray]:
    size = profile.native_size
    y, x = np.indices((size, size))
    rng = np.random.default_rng(profile.texture_seed)
    grain = rng.integers(-3, 4, (size, size), dtype=np.int16)
    checker = (((x // 7 + y // 7) % 2) * 2 - 1) * 4
    material = grain + checker
    material.setflags(write=False)
    diagonal = ((x + 2 * y + profile.texture_seed % 17) % 11 == 0)
    diagonal.setflags(write=False)
    return material, diagonal


def _style_viewport(frame: np.ndarray, profile: PresentationProfile) -> np.ndarray:
    # Color offsets preserve object silhouettes and collision geometry. Outlines
    # occupy existing object pixels, so the visible extent is never enlarged.
    colors = {
        "slate": (0, 0, 0), "sand": (19, 6, -21), "forest": (-13, 13, -8),
        "violet": (13, -7, 18), "ice": (-14, 10, 19), "ember": (22, -8, -10),
    }
    result = frame.astype(np.int16) + np.asarray(colors[profile.theme], dtype=np.int16)
    if profile.textures:
        material, diagonal = _material_maps(profile)
        # Texture only large flat material regions; tiny targets/cursors and UI
        # glyphs stay crisp while floor/walls acquire stable procedural grain.
        counts = np.all(frame == np.roll(frame, 1, 0), axis=-1)
        counts &= np.all(frame == np.roll(frame, 1, 1), axis=-1)
        counts &= np.all(frame == np.roll(frame, -1, 0), axis=-1)
        counts &= np.all(frame == np.roll(frame, -1, 1), axis=-1)
        result += np.where(counts, material, 0)[..., None]
        result += np.where(counts & diagonal, 3, 0)[..., None]
        edges = ~counts
        result += np.where(edges, -3, 0)[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


def present_frame(canonical_rgb: np.ndarray, profile: PresentationProfile) -> np.ndarray:
    """Transform saved canonical RGB; outcome comes only from its visible HUD.

    No renderer score/status fields are accepted. This makes the same transform
    safe for canonical diagnostic episodes used to train reward recognition.
    """
    from carryrl.rewards.visual import HUD_HEIGHT, verify_visible_outcome

    if (not isinstance(canonical_rgb, np.ndarray) or canonical_rgb.dtype != np.uint8
            or canonical_rgb.shape != (profile.native_size, profile.native_size, 3)):
        raise ValueError("Expected canonical RGB at the profile's unscaled native size")
    outcome = verify_visible_outcome(canonical_rgb)
    if not outcome["known"]:
        raise ValueError("Canonical HUD is unknown; refusing to infer presentation labels")
    status = "Success" if outcome["success"] else "Failed" if outcome["failure"] else "Play"
    viewport = _style_viewport(canonical_rgb, profile)
    # The old score panel disappears completely; no duplicate outcome is left.
    viewport[:HUD_HEIGHT] = np.array((31, 38, 52), dtype=np.uint8)
    width, height = profile.canvas_size
    canvas = np.full((height, width, 3), (11, 15, 23), dtype=np.uint8)
    vx0, vy0, vx1, vy1 = profile.viewport_box
    canvas[vy0:vy1, vx0:vx1] = viewport
    x0, y0, x1, y1 = profile.hud_box
    canvas[y0:y1, x0:x1] = render_hud(profile.theme, profile.hud_layout,
                                    int(outcome["score"]), status)
    return canvas
