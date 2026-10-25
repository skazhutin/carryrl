"""One screen-history policy with spatial vision and segmented memory.

The vision backbone is frozen in this first implementation. PPO replays the
complete recorded episode through every trainable layer from a clean state.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from carryrl.agent.actions import ACTION_FEATURE_SIZE, PhysicalAction, encode_actions
from carryrl.agent.decoder import AutoregressiveActionDecoder
from carryrl.agent.language import TEXT_WIDTH, MultilingualInstructionEncoder, validate_instruction
from carryrl.agent.memory import TemporalCore
from carryrl.agent.vision import DinoVision, SpatialResampler


@dataclass(frozen=True)
class AgentConfig:
    pretrained: bool = True
    image_size: int = 224
    width: int = 256
    layers: int = 4
    heads: int = 4
    spatial_tokens: int = 16
    memory_tokens: int = 8
    segment_length: int = 64
    text_encoder: str = "none"
    instruction_max_tokens: int = 512
    motor_version: str = "legacy"
    critic_initialization: str = "legacy"

    def __post_init__(self) -> None:
        if type(self.pretrained) is not bool:
            raise ValueError("pretrained must be a boolean")
        for name, value in asdict(self).items():
            if name in {"pretrained", "text_encoder", "instruction_max_tokens",
                        "motor_version", "critic_initialization"}:
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (not isinstance(self.motor_version, str)
                or self.motor_version not in {"legacy", "motor_v2"}):
            raise ValueError("motor_version must be legacy or motor_v2")
        if (not isinstance(self.critic_initialization, str)
                or self.critic_initialization not in {"legacy", "zero"}):
            raise ValueError("critic_initialization must be legacy or zero")
        if self.image_size % 14:
            raise ValueError("image_size must be divisible by DINO's patch size 14")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not isinstance(self.text_encoder, str) or self.text_encoder not in {
            "none", "multilingual_e5_small",
        }:
            raise ValueError("text_encoder must be none or multilingual_e5_small")
        if (type(self.instruction_max_tokens) is not int
                or not 2 <= self.instruction_max_tokens <= 512):
            raise ValueError("instruction_max_tokens must be an integer between 2 and 512")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        if not isinstance(data, dict):
            raise ValueError("agent configuration must be an object")
        try:
            return cls(**data)
        except TypeError as error:
            raise ValueError(f"invalid agent configuration: {error}") from error

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentDecision:
    action: PhysicalAction
    log_prob: float
    value: float


class GeneralistAgent(nn.Module):
    def __init__(
        self, config: AgentConfig | None = None, *, vision: nn.Module | None = None,
        language: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config or AgentConfig()
        width = self.config.width
        self.vision = vision if vision is not None else DinoVision(
            pretrained=self.config.pretrained, frozen=True, image_size=self.config.image_size,
        )
        self.vision.requires_grad_(False)
        self.vision.eval()
        self.resampler = SpatialResampler(width=width, tokens=self.config.spatial_tokens,
                                         input_width=384, heads=self.config.heads)
        self.summary_query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.summary_attention = nn.MultiheadAttention(width, self.config.heads,
                                                      dropout=0, batch_first=True)
        self.previous_action_embedding = nn.Linear(ACTION_FEATURE_SIZE, width)
        self.time_embedding = nn.Linear(1, width)
        self.summary_norm = nn.LayerNorm(width)
        self.temporal = TemporalCore(width=width, layers=self.config.layers,
                                     heads=self.config.heads,
                                     segment_length=self.config.segment_length,
                                     memory_tokens=self.config.memory_tokens)
        self.actor_attention = nn.MultiheadAttention(width, self.config.heads,
                                                    dropout=0, batch_first=True)
        self.actor_norm = nn.LayerNorm(width)
        self.decoder = AutoregressiveActionDecoder(
            width=width, motor_version=self.config.motor_version,
        )
        self.value_head = nn.Linear(width, 1)
        if self.config.critic_initialization == "zero":
            nn.init.zeros_(self.value_head.weight)
            nn.init.zeros_(self.value_head.bias)
        if self.config.text_encoder != "none":
            self.language = language if language is not None else MultilingualInstructionEncoder(
                max_tokens=self.config.instruction_max_tokens,
            )
            self.language.requires_grad_(False)
            self.language.eval()
            self.instruction_projection = nn.Linear(TEXT_WIDTH, width, bias=False)
        elif language is not None:
            raise ValueError("an injected language encoder requires text_encoder to be enabled")
        self._episode_state = None
        self._previous_action = PhysicalAction()
        self._instruction: str | None = None
        self._instruction_features: torch.Tensor | None = None
        self.screen_text_reader = None
        self.last_screen_text_result = None
        self._screen_text_step = 0

    @property
    def device(self) -> torch.device:
        return self.summary_query.device

    def train(self, mode: bool = True) -> "GeneralistAgent":
        super().train(mode)
        self.vision.eval()
        if self.config.text_encoder != "none":
            self.language.eval()
        return self

    def _invalidate_episode_features(self) -> None:
        self._episode_state = None
        self._previous_action = PhysicalAction()
        self._instruction_features = None
        self.last_screen_text_result = None
        self._screen_text_step = 0

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._invalidate_episode_features()
        return result

    def _load_from_state_dict(self, *args, **kwargs):
        self._invalidate_episode_features()
        super()._load_from_state_dict(*args, **kwargs)

    @torch.no_grad()
    def _encode_instruction(self, instruction: str | None) -> torch.Tensor | None:
        instruction = validate_instruction(instruction)
        if instruction is None:
            return None
        if self.config.text_encoder == "none":
            raise ValueError("nonempty instruction requires an enabled text_encoder")
        self.language.eval()
        feature = self.language(instruction)
        if feature.shape != (TEXT_WIDTH,) or not torch.isfinite(feature).all():
            raise RuntimeError(f"language encoder must produce a finite [{TEXT_WIDTH}] embedding")
        return feature.detach().cpu()

    @torch.no_grad()
    def encode_screen_text(self, text: str | None) -> torch.Tensor | None:
        """Encode one frame's OCR into separate frozen, contextual CPU tokens."""
        if text is None:
            return None
        if not isinstance(text, str):
            raise ValueError("screen_text must be a string or None")
        if not text.strip():
            return None
        if self.config.text_encoder == "none":
            raise ValueError("nonempty screen_text requires an enabled text_encoder")
        self.language.eval()
        features = self.language.encode_tokens(text)
        if (features.ndim != 2 or features.shape[1] != TEXT_WIDTH
                or not 0 < len(features) <= self.config.instruction_max_tokens
                or not torch.isfinite(features).all()):
            raise RuntimeError(
                f"language encoder must produce finite [tokens, {TEXT_WIDTH}] features"
            )
        return features.detach().cpu()

    def _screen_text_batch(self, texts: Sequence[str | None] | None, decisions: int):
        if texts is None:
            return None, None
        if isinstance(texts, (str, bytes)) or len(texts) != decisions:
            raise ValueError("screen_text must match every episode decision")
        features = [self.encode_screen_text(text) for text in texts]
        longest = max((len(feature) if feature is not None else 0 for feature in features),
                      default=0)
        if longest == 0:
            return None, None
        sample = next(feature for feature in features if feature is not None)
        padded = sample.new_zeros((decisions, longest, TEXT_WIDTH))
        mask = torch.zeros((decisions, self.config.spatial_tokens + longest),
                           dtype=torch.bool, device=self.device)
        mask[:, self.config.spatial_tokens:] = True
        for index, feature in enumerate(features):
            if feature is not None:
                padded[index, :len(feature)] = feature
                start = self.config.spatial_tokens
                mask[index, start:start + len(feature)] = False
        return padded, mask

    def _instruction_context(self, feature: torch.Tensor | None) -> torch.Tensor | None:
        if feature is None:
            return None
        return self.instruction_projection(feature.to(
            device=self.device, dtype=self.instruction_projection.weight.dtype,
        ))

    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must be uint8 NHWC RGB")
        self.vision.eval()
        return self.vision(frames.to(self.device))

    def reset_episode(self, *, instruction: str | None = None) -> None:
        instruction = validate_instruction(instruction)
        features = self._encode_instruction(instruction)
        self._episode_state = self.temporal.initial_state(
            device=self.device, dtype=self.summary_query.dtype,
        )
        self._previous_action = PhysicalAction()
        self._instruction = instruction
        self._instruction_features = features
        self.last_screen_text_result = None
        self._screen_text_step = 0

    def _spatial_summary(
        self, patches: torch.Tensor, previous: list[PhysicalAction],
        instruction_context: torch.Tensor | None = None,
        *, screen_text_features: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if patches.ndim != 3 or patches.shape[-1] != 384 or len(patches) != len(previous):
            raise ValueError("patches must be [decisions, patches, 384] matching actions")
        spatial = self.resampler(patches.to(self.device))
        if screen_text_features is not None:
            projected = self.instruction_projection(screen_text_features.to(
                device=self.device, dtype=self.instruction_projection.weight.dtype,
            ))
            spatial = torch.cat((spatial, projected), dim=1)
        query = self.summary_query.expand(len(patches), -1, -1)
        summary, _ = self.summary_attention(query, spatial, spatial, need_weights=False,
                                            key_padding_mask=key_padding_mask)
        action_context = encode_actions(previous, device=self.device)
        # The executor has a fixed 0.1-second interval in the initial suite.
        elapsed = torch.full((len(patches), 1), 0.1, device=self.device,
                             dtype=summary.dtype)
        summary = (summary[:, 0] + self.previous_action_embedding(action_context)
                   + self.time_embedding(elapsed))
        if instruction_context is not None:
            summary = summary + instruction_context
        summary = self.summary_norm(summary)
        return spatial, summary

    def _actor_features(
        self, states: torch.Tensor, spatial: torch.Tensor,
        instruction_context: torch.Tensor | None = None,
        *, key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.actor_attention(states[:, None], spatial, spatial, need_weights=False,
                                          key_padding_mask=key_padding_mask)
        actor = states + attended[:, 0]
        if instruction_context is not None:
            actor = actor + instruction_context
        return self.actor_norm(actor)

    def evaluate_episode(
        self, patches: torch.Tensor, actions: list[PhysicalAction],
        *, instruction: str | None = None, executed_actions: list[PhysicalAction] | None = None,
        return_decoder_trace: bool = False,
        screen_text: Sequence[str | None] | None = None,
    ) -> dict[str, torch.Tensor | list[tuple[torch.Tensor, torch.Tensor]]]:
        if not actions or len(actions) != len(patches):
            raise ValueError("a complete nonempty episode must match its recorded actions")
        executed = actions if executed_actions is None else executed_actions
        if len(executed) != len(actions) or any(
                not isinstance(action, PhysicalAction) for action in executed):
            raise ValueError("executed actions must match the complete episode")
        previous = [PhysicalAction(), *executed[:-1]]
        context = self._instruction_context(self._encode_instruction(instruction))
        tokens, mask = self._screen_text_batch(screen_text, len(actions))
        spatial, summary = self._spatial_summary(
            patches, previous, context, screen_text_features=tokens, key_padding_mask=mask,
        )
        states = self.temporal.forward_sequence(summary)
        actor_features = self._actor_features(states, spatial, context, key_padding_mask=mask)
        if return_decoder_trace:
            log_prob, entropy, trace = self.decoder.evaluate_with_trace(
                actor_features, actions, previous_actions=previous,
            )
        else:
            log_prob, entropy = self.decoder.evaluate(
                actor_features, actions, previous_actions=previous,
            )
        result = {"log_prob": log_prob, "entropy": entropy,
                  "value": self.value_head(states).squeeze(-1)}
        if return_decoder_trace:
            result["decoder_trace"] = trace
        return result

    def observe_action(self, action: PhysicalAction) -> None:
        """Replace the sampled input with the input actually applied, without a memory step."""
        if not isinstance(action, PhysicalAction):
            raise TypeError("executed input must be a PhysicalAction")
        self._previous_action = action

    def _step_features(self, patches: torch.Tensor, screen_text: str | None = None):
        if self.screen_text_reader is not None and screen_text is None:
            raise ValueError("patch-only decisions require explicit screen_text with a live reader")
        if self._episode_state is None:
            self.reset_episode(instruction=self._instruction)
        if patches.ndim != 2 or patches.shape[-1] != 384:
            raise ValueError("one frame must have [patches, 384] features")
        context = self._instruction_context(self._instruction_features)
        tokens, mask = self._screen_text_batch([screen_text], 1)
        spatial, summary = self._spatial_summary(
            patches[None], [self._previous_action], context,
            screen_text_features=tokens, key_padding_mask=mask,
        )
        state, next_episode_state = self.temporal.step(summary[0], self._episode_state)
        actor_features = self._actor_features(state[None], spatial, context, key_padding_mask=mask)
        value = self.value_head(state).squeeze(-1)
        return actor_features, value, next_episode_state

    @torch.no_grad()
    def act_from_patches(
        self, patches: torch.Tensor, deterministic: bool = False, *, screen_text: str | None = None,
    ) -> AgentDecision:
        actor, value, next_episode_state = self._step_features(patches, screen_text)
        actions, log_prob, _ = self.decoder.sample(
            actor, deterministic=deterministic, previous_actions=[self._previous_action],
        )
        self._episode_state = next_episode_state
        self._previous_action = actions[0]
        self._screen_text_step += 1
        return AgentDecision(actions[0], float(log_prob[0]), float(value))

    @torch.no_grad()
    def peek_value_from_patches(
        self, patches: torch.Tensor, *, screen_text: str | None = None,
    ) -> float:
        _, value, _ = self._step_features(patches, screen_text)
        return float(value)

    @torch.no_grad()
    def act(
        self, frame: np.ndarray | torch.Tensor, deterministic: bool = False,
        *, screen_text: str | None = None,
    ) -> AgentDecision:
        if self.screen_text_reader is not None and screen_text is not None:
            raise ValueError(
                "explicit screen_text cannot be combined with a live screen_text_reader"
            )
        if isinstance(frame, np.ndarray) and not frame.flags.writeable:
            frame = frame.copy()
        rgb = torch.as_tensor(frame)
        if rgb.dtype != torch.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("one RGB frame must have uint8 HWC shape")
        if self._episode_state is None:
            self.reset_episode(instruction=self._instruction)
        if self.screen_text_reader is not None:
            from carryrl.perception.screen_text import format_model_text

            result = self.screen_text_reader.read(rgb.detach().cpu().numpy(),
                                                   step=self._screen_text_step)
            screen_text = format_model_text(result)
            source = rgb.detach().cpu().numpy()
            if (result.frame_sha256 != hashlib.sha256(source.tobytes(order="C")).hexdigest()
                    or result.source_width != source.shape[1]
                    or result.source_height != source.shape[0]
                    or result.step != self._screen_text_step):
                raise ValueError(
                    "screen text reader result does not match the current RGB frame/step"
                )
            self.last_screen_text_result = result
        return self.act_from_patches(self.encode_frames(rgb[None])[0], deterministic,
                                     screen_text=screen_text)
