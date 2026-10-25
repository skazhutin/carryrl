"""Unfitted candidate for causal, scene-grounded reward supervision.

This module supplies computational building blocks, not an inference reward or
a selected architecture. No fitted checkpoint or PPO eligibility is claimed.
Scene features must be encoded from ``mask_reward_hud`` RGB *before* encoding;
full-frame policy feature caches cannot satisfy that contract. The encoder and
explicit-goal embedding are supplied by the caller; nothing is downloaded here.
"""

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from carryrl.agent.actions import ACTION_FEATURE_SIZE
from carryrl.rewards.visual import locate_visible_hud

HISTORY_REWARD_VERSION = "carryrl.scene_history_candidate.v1"
MASK_VERSION = "carryrl.full_hud_mask.v1"
EVENT_NAMES = (
    "item_collected", "key_collected", "button_activated", "switch_activated",
    "door_opened", "exit_reached",
)
EVENT_COUNT = len(EVENT_NAMES)
# Includes the initial frame plus the 128-command episode budget.
MAX_HISTORY_STEPS = 129
HUD_MASK_COLOR = (0, 0, 0)


def mask_reward_hud(frame: np.ndarray) -> np.ndarray | None:
    """Copy RGB and erase the whole localized panel; abstain on unknown panels.

    Localization uses public border geometry/colors, without reading score or
    status. Pixels outside the located rectangle are byte-for-byte preserved.
    Unknown or ambiguous panels must never pass through to a visual encoder.
    """
    location = locate_visible_hud(frame)
    if location is None:
        return None
    y, x, height, width, _ = location
    masked = frame.copy()
    masked[y:y + height, x:x + width] = HUD_MASK_COLOR
    return masked


def _positive_size(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _length_mask(
    lengths: torch.Tensor | None, batch: int, steps: int, device: torch.device,
) -> torch.Tensor:
    if lengths is None:
        return torch.ones((batch, steps), dtype=torch.bool, device=device)
    if (
        not isinstance(lengths, torch.Tensor) or lengths.shape != (batch,)
        or lengths.dtype not in (torch.int32, torch.int64)
        or (lengths < 1).any() or (lengths > steps).any()
    ):
        raise ValueError("lengths must be integer [B] with values in 1..T")
    return torch.arange(steps, device=device)[None] < lengths.to(device)[:, None]


class SceneHistoryHead(nn.Module):
    """Small experimental projection + causal GRU, not a fitted reward model.

    Each sequence starts at the actual episode beginning: observation t receives
    the own action executed before it; t=0 receives the zero action. Padding is
    optional, and meaningful prefixes are bounded to 129 observations. No hidden
    state, task ID, outcome, teacher target, or future prefix is a model input.
    """

    def __init__(
        self, scene_feature_size: int = 384, goal_feature_size: int = 384,
        hidden_size: int = 128, max_steps: int = MAX_HISTORY_STEPS,
        event_names: tuple[str, ...] = EVENT_NAMES,
    ):
        super().__init__()
        for name, value in (
            ("scene_feature_size", scene_feature_size),
            ("goal_feature_size", goal_feature_size), ("hidden_size", hidden_size),
            ("max_steps", max_steps),
        ):
            _positive_size(name, value)
        if max_steps > MAX_HISTORY_STEPS:
            raise ValueError("max_steps must be <= 129")
        if (not isinstance(event_names, tuple) or not 1 <= len(event_names) <= 64
                or any(not isinstance(name, str) or not name.strip() for name in event_names)
                or len(set(event_names)) != len(event_names)):
            raise ValueError("event_names must be a unique nonempty tuple of at most 64 names")
        self.event_names = event_names
        self.scene_feature_size = scene_feature_size
        self.goal_feature_size = goal_feature_size
        self.max_steps = max_steps
        self.scene_projection = nn.Linear(scene_feature_size, hidden_size)
        self.goal_projection = nn.Linear(goal_feature_size, hidden_size)
        self.action_projection = nn.Linear(ACTION_FEATURE_SIZE, hidden_size)
        self.history = nn.GRU(3 * hidden_size, hidden_size, batch_first=True)
        self.event_head = nn.Linear(hidden_size, len(event_names))
        self.potential_head = nn.Linear(hidden_size, 1)
        self.potential_valid_head = nn.Linear(hidden_size, 1)
        self.event_valid_head = nn.Linear(hidden_size, len(event_names))
        self.terminal_success_head = nn.Linear(hidden_size, 1)
        self.terminal_valid_head = nn.Linear(hidden_size, 1)

    def forward(
        self, scene_features: torch.Tensor, executed_previous_actions: torch.Tensor,
        explicit_goal_features: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the unfitted primitive on supplied masked-RGB features only."""
        if (
            not isinstance(scene_features, torch.Tensor) or scene_features.ndim != 3
            or scene_features.shape[0] < 1
            or not 1 <= scene_features.shape[1] <= self.max_steps
            or scene_features.shape[2] != self.scene_feature_size
        ):
            raise ValueError("scene_features must be [B,T,F] with B>0 and 1<=T<=max_steps")
        batch, steps, _ = scene_features.shape
        expected = (
            (scene_features, (batch, steps, self.scene_feature_size), "scene_features"),
            (executed_previous_actions, (batch, steps, ACTION_FEATURE_SIZE), "actions"),
            (explicit_goal_features, (batch, self.goal_feature_size), "explicit goals"),
        )
        for value, shape, name in expected:
            if (
                not isinstance(value, torch.Tensor) or value.shape != shape
                or not value.is_floating_point() or not torch.isfinite(value).all()
                or value.device != scene_features.device or value.dtype != scene_features.dtype
            ):
                raise ValueError(f"{name} must be finite floating {shape} on the same device/dtype")
        if (explicit_goal_features == 0).all(dim=-1).any():
            raise ValueError("Each explicit goal embedding must be nonzero")
        actions = executed_previous_actions
        if (actions.abs() > 1).any() or not ((actions[..., :-3] == 0)
                                               | (actions[..., :-3] == 1)).all():
            raise ValueError("Actions must use binary keys/buttons and normalized deltas in [-1,1]")
        if (actions[:, 0] != 0).any():
            raise ValueError("The initial observation must have a zero executed previous action")
        observed = _length_mask(lengths, batch, steps, scene_features.device)
        projected = torch.cat((
            torch.tanh(self.scene_projection(scene_features)),
            torch.tanh(self.action_projection(actions)),
            torch.tanh(self.goal_projection(explicit_goal_features))[:, None].expand(-1, steps, -1),
        ), dim=-1)
        # A unidirectional GRU never reads later observations, including padding.
        hidden, _ = self.history(projected)
        outputs = {
            "event_logits": self.event_head(hidden),
            "potential": self.potential_head(hidden).squeeze(-1).sigmoid(),
            "potential_valid_logits": self.potential_valid_head(hidden).squeeze(-1),
            "event_valid_logits": self.event_valid_head(hidden),
            "terminal_success_logits": self.terminal_success_head(hidden).squeeze(-1),
            "terminal_valid_logits": self.terminal_valid_head(hidden).squeeze(-1),
        }
        return {name: torch.where(observed[..., None] if value.ndim == 3 else observed,
                                  value, 0) for name, value in outputs.items()}

    def supervised_loss(self, outputs, targets, lengths=None):
        """Bind the target vocabulary to this head, including order, before loss."""
        return history_reward_loss(outputs, targets, lengths,
                                   event_names=self.event_names)


def _episode_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Equal episode weighting, including zero contribution for unobserved targets."""
    dimensions = tuple(range(1, values.ndim))
    totals = torch.where(mask, values, 0).sum(dim=dimensions)
    counts = mask.sum(dim=dimensions).clamp_min(1)
    return (totals / counts).mean()


def history_reward_loss(
    outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor],
    lengths: torch.Tensor | None = None,
    *, event_names: tuple[str, ...] = EVENT_NAMES,
) -> torch.Tensor:
    """Masked auxiliary supervision; unknown values never become negative labels.

    Validity classifiers learn the explicit true/false observation masks on real
    timesteps. Main heads learn only observed targets. Each component averages
    within episodes before averaging over episodes, so longer episodes do not
    dominate. Targets are scene-grounded labels, never HUD outcomes.
    """
    if targets.get("event_names", EVENT_NAMES) != event_names:
        raise ValueError("scene target event vocabulary/order differs from the history head")
    potential = outputs["potential"]
    if potential.ndim != 2 or min(potential.shape) < 1 or potential.shape[1] > MAX_HISTORY_STEPS:
        raise ValueError("potential must be [B,T] with B>0 and 1<=T<=129")
    batch, steps = potential.shape
    if lengths is None:
        lengths = targets.get("lengths")
    observed = _length_mask(lengths, batch, steps, potential.device)
    total = potential.sum() * 0
    for target_name, valid_name, output_name, validity_output, events in (
        ("potential", "potential_valid", "potential", "potential_valid_logits", False),
        ("events", "event_valid", "event_logits", "event_valid_logits", True),
        ("terminal_success", "terminal_success_valid", "terminal_success_logits",
         "terminal_valid_logits", False),
    ):
        shape = (batch, steps, len(event_names)) if events else (batch, steps)
        label, valid = targets[target_name], targets[valid_name]
        prediction, validity = outputs[output_name], outputs[validity_output]
        for value in (label, valid, prediction, validity):
            if value.shape != shape or value.device != potential.device:
                raise ValueError(f"{target_name} and associated tensors must have shape {shape}")
        if valid.dtype != torch.bool:
            raise ValueError(f"{valid_name} must be boolean")
        if not label.is_floating_point():
            raise ValueError(f"{target_name} targets must be floating point")
        if not torch.isfinite(prediction).all() or not torch.isfinite(validity).all():
            raise ValueError("Head outputs must be finite")
        real = observed[..., None].expand_as(valid) if events else observed
        mask = valid & real
        if not torch.isfinite(label[mask]).all() or ((label[mask] < 0)
                                                    | (label[mask] > 1)).any():
            raise ValueError(f"Observed {target_name} targets must be finite in [0,1]")
        safe_label = torch.where(mask, label, 0).to(prediction.dtype)
        if target_name == "potential":
            if (prediction < 0).any() or (prediction > 1).any():
                raise ValueError("Predicted potential must be in [0,1]")
            main_loss = (prediction - safe_label).square()
        else:
            main_loss = nn.functional.binary_cross_entropy_with_logits(
                prediction, safe_label, reduction="none",
            )
        valid_loss = nn.functional.binary_cross_entropy_with_logits(
            validity, valid.to(validity.dtype), reduction="none",
        )
        total = total + _episode_mean(main_loss, mask) + _episode_mean(valid_loss, real)
    return total
