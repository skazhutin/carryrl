"""Supervised visual score/event bootstrap for the atomic game's ordinary HUD.

This is a learned recognizer of an explicitly taught HUD, not autonomous reward
discovery. The independent pixel verifier provides labels and evaluation; its
outputs must never enter a policy observation or replace the frozen learned
recognizer in primary training.
"""

from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

HUD_HEIGHT = 28
HUD_BACKGROUND = (16, 20, 30)
HUD_TEXT = (235, 239, 244)
STATUSES = ("Play", "Success", "Failed")
SCREEN_REWARD_VERSION = "carryrl.screen_reward.v3"


@lru_cache(maxsize=1)
def _presented_catalogue() -> tuple:
    # The rendering catalogue is public UI vocabulary, not per-episode state.
    # Import lazily: atomic environments themselves import this module.
    from carryrl.envs.presentation import HUD_LAYOUTS, HUD_THEMES, render_hud

    return tuple(
        (theme, layout, render_hud(theme, layout, 0, "Play"))
        for theme in HUD_THEMES
        for layout in HUD_LAYOUTS
    )


def _panel_locations(frame: np.ndarray, background: tuple, height: int, width: int,
                     same: np.ndarray | None = None) -> list:
    """Locate clean panel borders by RGB alone, without reading any text labels."""
    if height > frame.shape[0] or width > frame.shape[1]:
        return []
    if same is None:
        same = np.all(frame == background, axis=-1)
    if np.count_nonzero(same) < 2 * height + 2 * width - 4:
        return []
    rows, columns = same.shape[0] - height + 1, same.shape[1] - width + 1
    corners = (
        same[:rows, :columns]
        & same[height - 1:, :columns]
        & same[:rows, width - 1:]
        & same[height - 1:, width - 1:]
    )
    candidates = np.argwhere(corners)
    # Large flat backgrounds are not HUDs; row/column prefix sums remove them
    # before looking for ordinary text in the interior.
    horizontal = np.pad(same.cumsum(axis=1), ((0, 0), (1, 0)))
    vertical = np.pad(same.cumsum(axis=0), ((1, 0), (0, 0)))
    locations = []
    for y, x in candidates:
        if (
            horizontal[y, x + width] - horizontal[y, x] == width
            and horizontal[y + height - 1, x + width]
            - horizontal[y + height - 1, x] == width
            and vertical[y + height, x] - vertical[y, x] == height
            and vertical[y + height, x + width - 1]
            - vertical[y, x + width - 1] == height
        ):
            crop = same[y:y + height, x:x + width]
            if 0 < np.count_nonzero(~crop) < height * width / 2:
                locations.append((int(y), int(x), height, width, tuple(background)))
    return locations


def locate_visible_hud(frame: np.ndarray) -> tuple | None:
    """Find one supported ordinary panel using border/color pixels only.

    No score/status templates or environment identifiers enter this localization
    step. Unsupported screens and two visible panels explicitly abstain.
    """
    if (
        not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
        or frame.ndim != 3 or frame.shape[2] != 3
        or min(frame.shape[:2]) < 64 or max(frame.shape[:2]) > 2048
    ):
        return None
    shapes = {(HUD_BACKGROUND, HUD_HEIGHT, frame.shape[1])}
    for _, _, template in _presented_catalogue():
        shapes.add((tuple(int(v) for v in template[0, 0]), *template.shape[:2]))
    found = set()
    encoded = (frame[:, :, 0].astype(np.uint32) << 16
               | frame[:, :, 1].astype(np.uint32) << 8 | frame[:, :, 2])
    masks = {}
    for background, height, width in shapes:
        if background not in masks:
            code = background[0] << 16 | background[1] << 8 | background[2]
            masks[background] = encoded == code
        found.update(_panel_locations(frame, background, height, width, masks[background]))
    return next(iter(found)) if len(found) == 1 else None


def _verify_presented_huds(frame: np.ndarray) -> list:
    from carryrl.envs.presentation import render_hud

    location = locate_visible_hud(frame)
    if location is None:
        return []
    y, x, height, width, _ = location
    crop = frame[y:y + height, x:x + width]
    matches = []
    for theme, layout, template in _presented_catalogue():
        if template.shape != crop.shape:
            continue
        for score in range(6):
            for status in STATUSES:
                if np.array_equal(crop, render_hud(theme, layout, score, status)):
                    matches.append((score, status))
    return matches


def screen_reward_readout(frame: np.ndarray) -> np.ndarray | None:
    """RGB-only localization and contrast/geometry normalization, without OCR.

    The two ordinary text lines become small monochrome images. Glyph identities
    are still learned by the CNN. Crop position, UI palette and native resolution
    are inferred from pixels and never supplied by an environment profile.
    """
    location = locate_visible_hud(frame)
    if location is None:
        return None
    y, x, height, width, background = location
    crop = frame[y:y + height, x:x + width].astype(np.int16)
    ink = np.max(np.abs(crop - np.asarray(background)), axis=-1).astype(np.float32)
    if ink.max() <= 0:
        return None
    ink /= ink.max()
    occupied = np.flatnonzero((ink > 0.02).any(axis=1))
    bands = np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1)
    if len(bands) != 2 or any(len(band) < 2 for band in bands):
        return None
    result = np.zeros((32, 64), np.uint8)
    for index, band in enumerate(bands):
        line = ink[band[0]:band[-1] + 1]
        columns = np.flatnonzero((line > 0.02).any(axis=0))
        line = line[:, columns[0]:columns[-1] + 1]
        resized = Image.fromarray((line * 255).round().astype(np.uint8)).resize(
            (60, 12), Image.Resampling.BILINEAR
        )
        result[2 + index * 16:14 + index * 16, 2:62] = np.asarray(resized)
    return np.repeat(result[:, :, None], 3, axis=2)


@lru_cache(maxsize=256)
def _hud_template(image_size: int, score: int, status: str) -> np.ndarray:
    image = Image.new("RGB", (image_size, HUD_HEIGHT), HUD_BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=10)
    draw.text((4, 1), f"Score: {score}", fill=HUD_TEXT, font=font)
    draw.text((4, 14), status, fill=HUD_TEXT, font=font)
    result = np.asarray(image).copy()
    result.setflags(write=False)
    return result


def draw_visible_hud(frame: np.ndarray, score: int, status: str) -> None:
    """Draw ordinary human-readable score and outcome, without coded markers."""
    if score not in range(6) or status not in STATUSES:
        raise ValueError("Atomic HUD score/status outside the documented range")
    frame[:HUD_HEIGHT] = _hud_template(frame.shape[1], score, status)


def verify_visible_outcome(frame: np.ndarray) -> dict:
    """Independent exact HUD verification; ambiguous frames explicitly remain unknown."""
    unknown = {"score": None, "success": None, "failure": None, "known": False}
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or min(frame.shape[:2]) < 64
    ):
        return unknown
    matches = []
    if frame.shape[0] == frame.shape[1]:
        for score in range(6):
            for status in STATUSES:
                if np.array_equal(frame[:HUD_HEIGHT], _hud_template(frame.shape[1], score, status)):
                    matches.append((score, status))
    if not matches:
        matches = list(set(_verify_presented_huds(frame)))
    elif locate_visible_hud(frame) is None:
        return unknown
    if len(matches) != 1:
        return unknown
    score, status = matches[0]
    return {
        "score": float(score),
        "success": status == "Success",
        "failure": status == "Failed",
        "known": True,
    }


class LearnedVisualReward(nn.Module):
    """Small CNN reading RGB pixels in the shared HUD, with explicit supervision.

    Restricting its view to the common HUD is an implementation choice for this
    bootstrap. It does not establish generalization to independent game UIs.
    Score predicts raw score units (projected to 0..5 at inference); two
    independent event logits expose uncertainty. A fixed 64-pixel HUD crop is
    padded to 32 rows so pooling divides exactly on CPU and Apple MPS.
    """

    def __init__(self, image_size: int = 96):
        super().__init__()
        if image_size < 64:
            raise ValueError("image_size must be >= 64")
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 16)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        if (
            frames.ndim != 4
            or frames.shape[1] != 3
            or frames.shape[2:] != (self.image_size, self.image_size)
            or not frames.is_floating_point()
        ):
            raise ValueError("Expected normalized float RGB NCHW frames of the configured size")
        if not torch.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError("Expected finite RGB values in [0, 1]")
        hud = nn.functional.pad(frames[:, :, :HUD_HEIGHT, :64], (0, 0, 0, 4))
        outputs = self.encoder(hud)
        return {
            "score": outputs[:, 0],
            "success_logits": outputs[:, 1],
            "failure_logits": outputs[:, 2],
        }

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> dict[str, float]:
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.shape != (self.image_size, self.image_size, 3)
        ):
            raise ValueError("Expected an RGB uint8 frame of the configured size")
        tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        output = self(tensor.to(next(self.parameters()).device))
        return {
            "score": float(output["score"][0].clamp(0, 5)),
            "success_probability": float(output["success_logits"][0].sigmoid()),
            "failure_probability": float(output["failure_logits"][0].sigmoid()),
        }

    def transition(self, previous: np.ndarray, current: np.ndarray) -> float:
        """Frozen learned score/event differences, with an explicit time cost."""
        before, after = self.predict(previous), self.predict(current)
        return float(
            after["score"]
            - before["score"]
            + after["success_probability"]
            - before["success_probability"]
            - after["failure_probability"]
            + before["failure_probability"]
            - 0.01
        )


class ScreenVisualReward(LearnedVisualReward):
    """Versioned learned readout for the supported varied ordinary HUDs.

    Localization is a disclosed pixel-only UI prior. The recognizer still needs
    supervised fitting; this is not reward discovery for arbitrary unseen games.
    Legacy ``LearnedVisualReward`` weights and input contract remain unchanged.
    """

    model_version = SCREEN_REWARD_VERSION

    def __init__(self, image_size: int = 96):
        nn.Module.__init__(self)
        if type(image_size) is not int or image_size < 64:
            raise ValueError("image_size must be an integer >=64")
        self.image_size = image_size

        def line_encoder(outputs):
            return nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 16)), nn.Flatten(),
                nn.Linear(16 * 2 * 16, 64), nn.ReLU(), nn.Linear(64, outputs),
            )

        # Separate pixel support prevents the status line from supplying score
        # information, including score/status combinations absent in play data.
        self.score_encoder = line_encoder(6)
        self.event_encoder = line_encoder(2)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        if (
            frames.ndim != 4 or frames.shape[1:] != (3, 32, 64)
            or not frames.is_floating_point()
        ):
            raise ValueError("Expected normalized RGB-only readouts with shape N,3,32,64")
        if not torch.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError("Expected finite readout values in [0,1]")
        score_outputs = self.score_encoder(frames[:, :, :16])
        event_outputs = self.event_encoder(frames[:, :, 16:])
        score_probabilities = score_outputs.softmax(dim=1)
        return {
            "score": score_probabilities.argmax(dim=1).float(),
            "score_logits": score_outputs,
            "score_confidence": score_probabilities.max(dim=1).values,
            "success_logits": event_outputs[:, 0],
            "failure_logits": event_outputs[:, 1],
        }

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> dict:
        if (
            not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3
        ):
            raise ValueError("Expected an RGB uint8 screen")
        readout = screen_reward_readout(frame)
        if readout is None:
            return {"score": 0.0, "success_probability": 0.5,
                    "failure_probability": 0.5, "known": False}
        tensor = torch.from_numpy(readout).permute(2, 0, 1).unsqueeze(0).float() / 255
        output = self(tensor.to(next(self.parameters()).device))
        return {
            "score": float(output["score"][0].clamp(0, 5)),
            "success_probability": float(output["success_logits"][0].sigmoid()),
            "failure_probability": float(output["failure_logits"][0].sigmoid()),
            "score_confidence": float(output["score_confidence"][0]),
            "known": True,
        }

    def transition(self, previous: np.ndarray, current: np.ndarray) -> float:
        before, after = self.predict(previous), self.predict(current)
        if not before["known"] or not after["known"]:
            return 0.0
        return float(
            after["score"] - before["score"]
            + after["success_probability"] - before["success_probability"]
            - after["failure_probability"] + before["failure_probability"] - 0.01
        )


def _dataset_tensors(frames: np.ndarray, labels: dict) -> tuple[torch.Tensor, dict]:
    if (
        not isinstance(frames, np.ndarray)
        or frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[-1] != 3
        or len(frames) == 0
    ):
        raise ValueError("Expected a nonempty RGB uint8 NHWC dataset")
    tensors = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255
    targets = {}
    for name in ("score", "success", "failure"):
        values = np.asarray(labels[name], dtype=np.float32)
        if values.shape != (len(frames),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} labels")
        if (values < 0).any() or (values > (5 if name == "score" else 1)).any():
            raise ValueError(f"Out-of-range {name} labels")
        targets[name] = torch.from_numpy(values)
    return tensors, targets


@torch.no_grad()
def evaluate_visual_reward(model: LearnedVisualReward, frames: np.ndarray, labels: dict) -> dict:
    tensors, targets = _dataset_tensors(frames, labels)
    model.eval()
    device = next(model.parameters()).device
    pieces = {name: [] for name in ("score", "success_logits", "failure_logits")}
    for batch in tensors.split(128):
        output = model(batch.to(device))
        for name in pieces:
            pieces[name].append(output[name].cpu())
    output = {name: torch.cat(values) for name, values in pieces.items()}
    event_metrics = {}
    for name in ("success", "failure"):
        truth = targets[name].bool()
        predicted = output[f"{name}_logits"] >= 0
        probabilities = output[f"{name}_logits"].sigmoid()
        event_metrics[f"{name}_brier_score"] = float(((probabilities - targets[name]) ** 2).mean())
        event_metrics[f"{name}_positive_probability_mean"] = (
            float(probabilities[truth].mean()) if truth.any() else None
        )
        event_metrics[f"{name}_negative_probability_mean"] = (
            float(probabilities[~truth].mean()) if (~truth).any() else None
        )
        event_metrics[f"{name}_positive_frames"] = int(truth.sum())
        event_metrics[f"{name}_recall"] = (
            float(predicted[truth].float().mean()) if truth.any() else None
        )
        event_metrics[f"{name}_false_positive_rate"] = (
            float(predicted[~truth].float().mean()) if (~truth).any() else None
        )
    combinations = torch.stack([targets[name] for name in ("score", "success", "failure")], dim=1)
    combination_metrics = []
    for combination in torch.unique(combinations, dim=0):
        mask = (combinations == combination).all(dim=1)
        combination_metrics.append(
            {
                "score": float(combination[0]),
                "success": bool(combination[1]),
                "failure": bool(combination[2]),
                "frames": int(mask.sum()),
                "score_mae": float((output["score"][mask] - targets["score"][mask]).abs().mean()),
            }
        )
    return {
        "score_mae": float((output["score"] - targets["score"]).abs().mean()),
        "score_mse": float(((output["score"] - targets["score"]) ** 2).mean()),
        "success_accuracy": float(
            ((output["success_logits"] >= 0) == targets["success"].bool()).float().mean()
        ),
        "failure_accuracy": float(
            ((output["failure_logits"] >= 0) == targets["failure"].bool()).float().mean()
        ),
        "frames": len(frames),
        "score_mae_by_combination": combination_metrics,
        **event_metrics,
    }


def train_visual_reward(
    model: LearnedVisualReward,
    frames: np.ndarray,
    labels: dict,
    epochs: int = 20,
    batch_size: int = 64,
    seed: int = 0,
    learning_rate: float = 1e-3,
    min_samples_per_epoch: int = 0,
) -> dict:
    """Fit only caller-supplied training frames, then freeze the recognizer.

    The caller must split whole episodes/maps before invoking this helper.
    Validation/test data are never automatically selected or fitted here.
    """
    if (epochs < 1 or batch_size < 1 or learning_rate <= 0
            or type(min_samples_per_epoch) is not int or min_samples_per_epoch < 0):
        raise ValueError("Training epochs, batch size and learning rate must be positive")
    tensors, targets = _dataset_tensors(frames, labels)
    torch.manual_seed(seed)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    # Balance the taught score/event combinations; long PLAY episodes must not
    # hide a recognizer that predicts PLAY for every rare terminal frame.
    groups = torch.stack([targets[name] for name in ("score", "success", "failure")], dim=1)
    _, inverse, counts = torch.unique(groups, dim=0, return_inverse=True, return_counts=True)
    sampling_weights = counts[inverse].float().reciprocal()
    first_loss = final_loss = None
    for _ in range(epochs):
        losses = []
        sampled = torch.multinomial(
            sampling_weights, max(len(frames), min_samples_per_epoch), replacement=True,
            generator=generator,
        )
        for indices in sampled.split(batch_size):
            output = model(tensors[indices].to(device))
            # Keep score regression in its raw units. Dividing it by the score
            # range let the easier event text dominate the shared features and
            # left small digits unresolved when success can have different scores.
            if "score_logits" in output:
                loss = nn.functional.cross_entropy(
                    output["score_logits"], targets["score"][indices].long().to(device)
                )
            else:
                loss = ((output["score"] - targets["score"][indices].to(device)) ** 2).mean()
            for name in ("success", "failure"):
                loss = loss + nn.functional.binary_cross_entropy_with_logits(
                    output[f"{name}_logits"],
                    targets[name][indices].to(device),
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        final_loss = float(np.mean(losses))
        if first_loss is None:
            first_loss = final_loss
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "first_epoch_loss": first_loss,
        "final_epoch_loss": final_loss,
        "epochs": epochs,
        "samples_per_epoch": max(len(frames), min_samples_per_epoch),
        "seed": seed,
        "sampling": "balanced training-only score/event combinations with replacement",
        "score_loss": ("categorical cross entropy for the six taught score glyphs"
                       if getattr(model, "model_version", None) == SCREEN_REWARD_VERSION
                       else "mean squared error in raw score units, weight 1.0"),
        "confidence": "event probabilities are uncalibrated diagnostic outputs",
        "supervision": "Exact pixel verification of the controlled games' shared ordinary HUD",
        "scope": "HUD recognition bootstrap, not autonomous reward discovery",
        "train_metrics": evaluate_visual_reward(model, frames, labels),
    }
