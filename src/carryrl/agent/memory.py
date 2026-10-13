"""Causal segmented Transformer with read/write memory and streaming KV caches."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LayerCache:
    """Keys/values for one layer, shaped [heads, cached_tokens, head_width]."""

    keys: torch.Tensor
    values: torch.Tensor

    def detach(self) -> "LayerCache":
        return LayerCache(self.keys.detach(), self.values.detach())


@dataclass(frozen=True)
class TemporalState:
    """Episode-local state; calls return new states instead of mutating this one.

    It is valid only for the current network weights. Reconstruct it from the
    sequence after optimizer updates rather than reusing stale rollout caches.
    """

    memory: torch.Tensor
    layer_caches: tuple[LayerCache, ...]
    position: int

    def detach(self) -> "TemporalState":
        return TemporalState(self.memory.detach(), tuple(c.detach() for c in self.layer_caches),
                             self.position)


class _TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width),
        )

    def forward_tokens(
        self, tokens: torch.Tensor, past: LayerCache | None = None,
        allowed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LayerCache]:
        qkv = self.qkv(self.attention_norm(tokens)).reshape(
            tokens.shape[0], 3, self.heads, self.head_width,
        ).permute(1, 2, 0, 3)
        queries, keys, values = qkv.unbind(0)
        if past is not None:
            keys = torch.cat((past.keys, keys), dim=1)
            values = torch.cat((past.values, values), dim=1)
        attended = F.scaled_dot_product_attention(
            queries[None], keys[None], values[None], attn_mask=allowed, dropout_p=0.0,
        )[0].transpose(0, 1).reshape(tokens.shape[0], self.width)
        tokens = tokens + self.output(attended)
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        return tokens, LayerCache(keys, values)


class TemporalCore(nn.Module):
    """RMT-style memory with a bounded causal cache inside each segment.

    Read-memory tokens see only memory from the previous segment. Decision
    tokens see that memory and the causal decision prefix. Write tokens attend
    to the completed segment, and are used only in the following segment.
    The same four blocks process read, decision, and write tokens.

    ``forward_sequence`` vectorizes each segment and retains its autograd graph
    across segment boundaries. The caller chooses BPTT length. ``step`` caches
    per-layer KV tensors, and has the same causal computation during inference.
    A fresh ``initial_state`` resets all episode-local history.
    """

    def __init__(
        self, width: int = 256, layers: int = 4, heads: int = 4,
        segment_length: int = 64, memory_tokens: int = 8,
    ):
        super().__init__()
        for name, value in (("width", width), ("layers", layers), ("heads", heads),
                            ("segment_length", segment_length), ("memory_tokens", memory_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.layers = layers
        self.heads = heads
        self.segment_length = segment_length
        self.memory_tokens = memory_tokens
        self.initial_memory = nn.Parameter(torch.empty(memory_tokens, width))
        self.read_positions = nn.Parameter(torch.empty(memory_tokens, width))
        self.decision_positions = nn.Parameter(torch.empty(segment_length, width))
        self.write_tokens = nn.Parameter(torch.empty(memory_tokens, width))
        for parameter in (self.initial_memory, self.read_positions, self.decision_positions,
                          self.write_tokens):
            nn.init.normal_(parameter, std=0.02)
        self.blocks = nn.ModuleList([_TransformerBlock(width, heads) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(width)

    def _validate_summary(self, summary: torch.Tensor, *, sequence: bool) -> torch.Tensor:
        if not sequence and summary.shape == (self.width,):
            summary = summary[None]
        if summary.ndim != 2 or summary.shape[-1] != self.width:
            raise ValueError("summary must have shape [T, width] or [width] for a single step")
        if not sequence and summary.shape[0] != 1:
            raise ValueError("step accepts exactly one decision")
        if (summary.device != self.initial_memory.device
                or summary.dtype != self.initial_memory.dtype):
            raise ValueError("summary and core must have the same device and floating dtype")
        return summary

    def _start_segment(self, memory: torch.Tensor) -> TemporalState:
        tokens = memory + self.read_positions
        caches = []
        for block in self.blocks:
            # The read-memory prefix can interact bidirectionally: it has no
            # observations from the current segment.
            tokens, cache = block.forward_tokens(tokens)
            caches.append(cache)
        return TemporalState(memory, tuple(caches), 0)

    def initial_state(
        self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None,
    ) -> TemporalState:
        parameter = self.initial_memory
        if device is not None:
            requested = torch.empty(0, device=device).device
            if requested != parameter.device:
                raise ValueError("state device must match the temporal core; move the module first")
        if dtype is not None and dtype != parameter.dtype:
            raise ValueError("state dtype must match the temporal core; cast the module first")
        return self._start_segment(parameter)

    def _write_memory(self, caches: tuple[LayerCache, ...]) -> torch.Tensor:
        tokens = self.write_tokens
        for block, past in zip(self.blocks, caches, strict=True):
            # All decisions are in the past here. Write queries may interact,
            # but never become keys for a decision in the completed segment.
            tokens, _ = block.forward_tokens(tokens, past=past)
        return self.output_norm(tokens)

    def _segment(
        self, summary: torch.Tensor, memory: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerCache, ...]]:
        count = summary.shape[0]
        tokens = torch.cat((memory + self.read_positions,
                            summary + self.decision_positions[:count]), dim=0)
        total = self.memory_tokens + count
        allowed = torch.ones(total, total, device=tokens.device, dtype=torch.bool).tril()
        allowed[:self.memory_tokens, :self.memory_tokens] = True
        caches = []
        for block in self.blocks:
            tokens, cache = block.forward_tokens(tokens, allowed=allowed)
            caches.append(cache)
        return self.output_norm(tokens[self.memory_tokens:]), tuple(caches)

    def forward_sequence(self, summary: torch.Tensor) -> torch.Tensor:
        summary = self._validate_summary(summary, sequence=True)
        if summary.shape[0] == 0:
            return summary
        outputs = []
        memory = self.initial_memory
        for start in range(0, summary.shape[0], self.segment_length):
            segment = summary[start:start + self.segment_length]
            hidden, caches = self._segment(segment, memory)
            outputs.append(hidden)
            if start + self.segment_length < summary.shape[0]:
                memory = self._write_memory(caches)
        return torch.cat(outputs, dim=0)

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(summary)

    def step(
        self, summary: torch.Tensor, state: TemporalState,
    ) -> tuple[torch.Tensor, TemporalState]:
        summary = self._validate_summary(summary, sequence=False)
        if not 0 <= state.position < self.segment_length or len(state.layer_caches) != self.layers:
            raise ValueError("state does not match the temporal core")
        if state.memory.shape != self.initial_memory.shape:
            raise ValueError("state memory does not match the temporal core")
        expected_length = self.memory_tokens + state.position
        expected_shape = (self.heads, expected_length, self.width // self.heads)
        for cache in state.layer_caches:
            if cache.keys.shape != expected_shape or cache.values.shape != expected_shape:
                raise ValueError("state cache does not match the temporal core")
        tokens = summary + self.decision_positions[state.position:state.position + 1]
        caches = []
        for block, past in zip(self.blocks, state.layer_caches, strict=True):
            tokens, cache = block.forward_tokens(tokens, past=past)
            caches.append(cache)
        output = self.output_norm(tokens)[0]
        position = state.position + 1
        if position == self.segment_length:
            new_state = self._start_segment(self._write_memory(tuple(caches)))
        else:
            new_state = TemporalState(state.memory, tuple(caches), position)
        return output, new_state
