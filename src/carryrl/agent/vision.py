"""Full-frame DINOv2 patch features and a trainable spatial resampler."""

from contextlib import nullcontext
from math import isqrt
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Dinov2Config, Dinov2Model

DINO_ID = "facebook/dinov2-small"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
DINO_PARAMETER_COUNT = 22_056_576


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _small_config() -> Dinov2Config:
    """The pinned checkpoint's configuration, without contacting the Hub."""
    return Dinov2Config(
        hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
        image_size=518, patch_size=14, num_channels=3, mlp_ratio=4,
        hidden_act="gelu", hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0, drop_path_rate=0.0,
        initializer_range=0.02, layer_norm_eps=1e-6,
        layerscale_value=1.0, qkv_bias=True, use_swiglu_ffn=False,
    )


class DinoVision(nn.Module):
    """Encode uint8 RGB batches into spatial patches, excluding the CLS token.

    The default checkpoint is pinned. ``pretrained=False`` builds that same
    graph with random weights and needs no network. ``backbone`` or ``config``
    can supply a smaller offline model for tests, only with pretrained=False.
    Loading failures propagate; they never switch to random weights.
    """

    def __init__(
        self, pretrained: bool = True, frozen: bool = True, image_size: int = 224,
        *, local_files_only: bool = False, cache_dir: str | Path | None = None,
        backbone: Dinov2Model | None = None, config: Dinov2Config | None = None,
    ):
        super().__init__()
        _positive_integer("image_size", image_size)
        if backbone is not None and config is not None:
            raise ValueError("supply either backbone or config")
        if pretrained and (backbone is not None or config is not None):
            raise ValueError("an injected backbone/config requires pretrained=False")
        if pretrained:
            self.backbone = Dinov2Model.from_pretrained(
                DINO_ID, revision=DINO_REVISION, local_files_only=local_files_only,
                cache_dir=cache_dir, use_safetensors=True, attn_implementation="sdpa",
            )
        elif backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = Dinov2Model(config if config is not None else _small_config())
        patch_size = self.backbone.config.patch_size
        if not isinstance(patch_size, int) or image_size % patch_size:
            raise ValueError("image_size must be divisible by the backbone's square patch size")
        if self.backbone.config.num_channels != 3:
            raise ValueError("the backbone must accept three RGB channels")
        self.pretrained = pretrained
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_grid = (image_size // patch_size, image_size // patch_size)
        self.hidden_size = self.backbone.config.hidden_size
        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.set_frozen(frozen)

    @classmethod
    def from_cached(cls, **kwargs) -> "DinoVision":
        """Load the pinned weights locally; an absent cache is an error."""
        return cls(pretrained=True, local_files_only=True, **kwargs)

    def set_frozen(self, frozen: bool) -> None:
        self.frozen = frozen
        self.backbone.requires_grad_(not frozen)
        self.backbone.train(self.training and not frozen)

    def train(self, mode: bool = True) -> "DinoVision":
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def preprocess(self, rgb: torch.Tensor) -> torch.Tensor:
        """Letterbox the entire frame, then apply ImageNet normalization.

        Padding has the RGB mean and thus becomes zero after normalization.
        Geometry depends only on the observed frame dimensions.
        """
        if rgb.dtype != torch.uint8:
            raise TypeError("RGB input must be uint8, not pre-normalized floats")
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or min(rgb.shape[:3]) <= 0:
            raise ValueError("RGB input must have shape [N, H, W, 3] with positive dimensions")
        if rgb.device != self.pixel_mean.device:
            raise ValueError("RGB input and vision module must be on the same device")
        height, width = rgb.shape[1:3]
        scale = self.image_size / max(height, width)
        resized_height = max(1, min(self.image_size, round(height * scale)))
        resized_width = max(1, min(self.image_size, round(width * scale)))
        pixels = rgb.permute(0, 3, 1, 2).to(dtype=self.pixel_mean.dtype) / 255.0
        pixels = F.interpolate(
            pixels, size=(resized_height, resized_width), mode="bicubic",
            align_corners=False, antialias=True,
        ).clamp(0.0, 1.0)
        canvas = self.pixel_mean.expand(rgb.shape[0], 3, self.image_size, self.image_size).clone()
        top = (self.image_size - resized_height) // 2
        left = (self.image_size - resized_width) // 2
        canvas[:, :, top:top + resized_height, left:left + resized_width] = pixels
        return (canvas - self.pixel_mean) / self.pixel_std

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        pixels = self.preprocess(rgb)
        with torch.no_grad() if self.frozen else nullcontext():
            hidden = self.backbone(pixel_values=pixels, return_dict=True).last_hidden_state
        expected_tokens = self.patch_grid[0] * self.patch_grid[1] + 1
        if hidden.shape != (rgb.shape[0], expected_tokens, self.hidden_size):
            raise RuntimeError("backbone output does not match the expected CLS + spatial patches")
        return hidden[:, 1:, :]


class _ResamplingStage(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width),
        )

    def forward(self, queries: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        context = self.context_norm(patches)
        attended, _ = self.attention(self.query_norm(queries), context, context, need_weights=False)
        queries = queries + attended
        return queries + self.mlp(self.mlp_norm(queries))


class SpatialResampler(nn.Module):
    """Two cross-attention stages over patches with explicit 2D patch centers."""

    def __init__(self, width: int = 256, tokens: int = 16, input_width: int = 384, heads: int = 4):
        super().__init__()
        for name, value in (("width", width), ("tokens", tokens), ("input_width", input_width),
                            ("heads", heads)):
            _positive_integer(name, value)
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.tokens = tokens
        self.input_width = input_width
        self.patch_projection = nn.Linear(input_width, width)
        self.position_projection = nn.Linear(2, width, bias=False)
        self.queries = nn.Parameter(torch.empty(tokens, width))
        nn.init.normal_(self.queries, std=0.02)
        self.stages = nn.ModuleList([_ResamplingStage(width, heads) for _ in range(2)])
        self.output_norm = nn.LayerNorm(width)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3 or patches.shape[-1] != self.input_width or patches.shape[0] == 0:
            raise ValueError("patches must have shape [N, square_patch_count, input_width]")
        side = isqrt(patches.shape[1])
        if side == 0 or side * side != patches.shape[1]:
            raise ValueError("patch count must form the square letterboxed image grid")
        coordinates = (torch.arange(side, device=patches.device, dtype=patches.dtype) + 0.5)
        coordinates = coordinates * (2.0 / side) - 1.0
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        positions = torch.stack((xx, yy), dim=-1).reshape(side * side, 2)
        context = self.patch_projection(patches) + self.position_projection(positions)[None]
        queries = self.queries[None].expand(patches.shape[0], -1, -1)
        for stage in self.stages:
            queries = stage(queries, context)
        return self.output_norm(queries)
