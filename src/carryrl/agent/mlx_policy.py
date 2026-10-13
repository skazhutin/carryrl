"""The complete trainable GeneralistAgent graph on MLX's Metal backend.

Frozen vision and language encoders stay in Torch. Their detached float32
features cross through NumPy once; every policy operation remains in MLX.
Parameter names and layouts match Torch exactly, including packed attention
projections and the GRU's separate input/hidden biases.
"""

from collections.abc import Sequence
from math import isqrt

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from carryrl.agent.actions import (
    ACTION_FEATURE_SIZE,
    KEY_VOCAB,
    MOUSE_BINS,
    MOUSE_BUTTONS,
    WHEEL_BINS,
    PhysicalAction,
)
from carryrl.agent.model import AgentConfig, GeneralistAgent


def to_mlx(value) -> mx.array:
    """Bridge detached Torch CPU features through NumPy with explicit float32."""
    if isinstance(value, mx.array):
        return value.astype(mx.float32)
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    return mx.array(np.asarray(value, dtype=np.float32))


def _parameter_shapes(config: AgentConfig) -> dict[str, tuple[int, ...]]:
    """Independent inventory: importing an unsupported trainable layer fails."""
    width = config.width
    shapes = {}

    def linear(name, incoming, outgoing, bias=True):
        shapes[name + ".weight"] = (outgoing, incoming)
        if bias:
            shapes[name + ".bias"] = (outgoing,)

    def norm(name):
        shapes[name + ".weight"] = (width,)
        shapes[name + ".bias"] = (width,)

    def attention(name):
        shapes[name + ".in_proj_weight"] = (3 * width, width)
        shapes[name + ".in_proj_bias"] = (3 * width,)
        linear(name + ".out_proj", width, width)

    linear("resampler.patch_projection", 384, width)
    linear("resampler.position_projection", 2, width, False)
    shapes["resampler.queries"] = (config.spatial_tokens, width)
    for index in range(2):
        prefix = f"resampler.stages.{index}"
        for name in ("query_norm", "context_norm", "mlp_norm"):
            norm(prefix + "." + name)
        attention(prefix + ".attention")
        linear(prefix + ".mlp.0", width, 4 * width)
        linear(prefix + ".mlp.2", 4 * width, width)
    norm("resampler.output_norm")
    shapes["summary_query"] = (1, 1, width)
    attention("summary_attention")
    linear("previous_action_embedding", ACTION_FEATURE_SIZE, width)
    linear("time_embedding", 1, width)
    norm("summary_norm")
    for name in ("initial_memory", "read_positions", "write_tokens"):
        shapes["temporal." + name] = (config.memory_tokens, width)
    shapes["temporal.decision_positions"] = (config.segment_length, width)
    for index in range(config.layers):
        prefix = f"temporal.blocks.{index}"
        norm(prefix + ".attention_norm")
        linear(prefix + ".qkv", width, 3 * width)
        linear(prefix + ".output", width, width)
        norm(prefix + ".mlp_norm")
        linear(prefix + ".mlp.0", width, 4 * width)
        linear(prefix + ".mlp.2", 4 * width, width)
    norm("temporal.output_norm")
    attention("actor_attention")
    norm("actor_norm")
    linear("decoder.context", width, width)
    for name in ("weight_ih", "weight_hh"):
        shapes["decoder.state_cell." + name] = (3 * width, width)
    for name in ("bias_ih", "bias_hh"):
        shapes["decoder.state_cell." + name] = (3 * width,)
    for name, count in (("key", len(KEY_VOCAB) + 1), ("mouse", 8),
                        ("dx", len(MOUSE_BINS)), ("dy", len(MOUSE_BINS))):
        shapes[f"decoder.{name}_embedding.weight"] = (count, width)
        linear(f"decoder.{name}_head", width, count)
    linear("decoder.wheel_head", width, len(WHEEL_BINS))
    linear("value_head", width, 1)
    if config.text_encoder != "none":
        linear("instruction_projection", 384, width, False)
    return shapes


class MLXPolicy(nn.Module):
    """Metal policy with a parameter tree ``{'weights': {Torch name: array}}``.

    This tree is directly supported by ``nn.value_and_grad`` and MLX optimizers.
    The independent inventory rejects missing, extra, or reshaped parameters.
    No frozen encoder arrays are duplicated into this graph.
    """

    def __init__(self, config: AgentConfig, weights: dict[str, mx.array],
                 *, motor_initialization: str = "legacy"):
        super().__init__()
        if not mx.metal.is_available():
            raise RuntimeError("the MLX policy requires an available Metal GPU")
        if mx.default_device() != mx.gpu:
            raise RuntimeError("the MLX policy requires the default MLX device to be GPU")
        self.config = config
        self.motor_initialization = motor_initialization
        if motor_initialization not in {"legacy", "balanced"}:
            raise ValueError("unsupported motor initialization")
        self._validate_weights(weights)
        self.weights = weights

    def _validate_weights(self, weights):
        expected = _parameter_shapes(self.config)
        if set(weights) != set(expected):
            raise ValueError(f"MLX trainable parameter coverage mismatch: "
                             f"missing={sorted(set(expected) - set(weights))}, "
                             f"extra={sorted(set(weights) - set(expected))}")
        for name, shape in expected.items():
            if tuple(weights[name].shape) != shape:
                raise ValueError(f"MLX parameter {name} must have shape {shape}")
            if weights[name].dtype != mx.float32:
                raise ValueError(f"MLX parameter {name} must use float32")

    @classmethod
    def from_torch(cls, agent: GeneralistAgent) -> "MLXPolicy":
        weights = {name: to_mlx(parameter) for name, parameter in agent.named_parameters()
                   if parameter.requires_grad}
        return cls(agent.config, weights, motor_initialization=agent.decoder.motor_initialization)

    def export_to_torch(self, agent: GeneralistAgent) -> None:
        """Synchronize every trainable parameter; preserve frozen encoder state."""
        if agent.config != self.config:
            raise ValueError("Torch and MLX agent configurations must match")
        self._validate_weights(self.weights)
        parameters = {name: parameter for name, parameter in agent.named_parameters()
                      if parameter.requires_grad}
        if set(parameters) != set(self.weights):
            raise ValueError("Torch and MLX trainable parameter names must match")
        for name, parameter in parameters.items():
            if tuple(parameter.shape) != self.weights[name].shape:
                raise ValueError(f"Torch and MLX parameter shapes differ for {name}")
        mx.eval(self.weights)
        # Copy rather than replacing Parameter objects, preserving optimizer references.
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(torch.from_numpy(np.array(self.weights[name])).to(
                    device=parameter.device, dtype=parameter.dtype,
                ))
        agent.decoder.motor_initialization = self.motor_initialization
        agent._invalidate_episode_features()

    def _linear(self, x, name):
        result = x @ self.weights[name + ".weight"].T
        bias = self.weights.get(name + ".bias")
        return result if bias is None else result + bias

    def _norm(self, x, name):
        return mx.fast.layer_norm(x, self.weights[name + ".weight"],
                                  self.weights[name + ".bias"], eps=1e-5)

    def _mlp(self, x, name):
        return self._linear(nn.gelu(self._linear(x, name + ".0")), name + ".2")

    def _attention(self, queries, keys, values, name):
        width, heads = self.config.width, self.config.heads
        packed = self.weights[name + ".in_proj_weight"]
        bias = self.weights[name + ".in_proj_bias"]
        projected = []
        for index, tokens in enumerate((queries, keys, values)):
            offset = index * width
            x = tokens @ packed[offset:offset + width].T + bias[offset:offset + width]
            projected.append(x.reshape(x.shape[0], x.shape[1], heads, width // heads)
                             .transpose(0, 2, 1, 3))
        attended = mx.fast.scaled_dot_product_attention(
            *projected, scale=(width // heads) ** -0.5,
        ).transpose(0, 2, 1, 3).reshape(queries.shape[0], queries.shape[1], width)
        return self._linear(attended, name + ".out_proj")

    def _spatial_summary(self, patches, previous, instruction_context):
        side = isqrt(patches.shape[1])
        if side == 0 or side * side != patches.shape[1]:
            raise ValueError("patch count must form a square image grid")
        coordinates = (mx.arange(side, dtype=mx.float32) + 0.5) * (2.0 / side) - 1.0
        yy, xx = mx.meshgrid(coordinates, coordinates, indexing="ij")
        positions = mx.stack((xx, yy), axis=-1).reshape(side * side, 2)
        context = (self._linear(patches, "resampler.patch_projection")
                   + self._linear(positions, "resampler.position_projection")[None])
        spatial = mx.broadcast_to(self.weights["resampler.queries"][None],
                                  (len(previous), self.config.spatial_tokens, self.config.width))
        for index in range(2):
            name = f"resampler.stages.{index}"
            normalized_context = self._norm(context, name + ".context_norm")
            spatial = spatial + self._attention(self._norm(spatial, name + ".query_norm"),
                                               normalized_context, normalized_context,
                                               name + ".attention")
            spatial = spatial + self._mlp(self._norm(spatial, name + ".mlp_norm"), name + ".mlp")
        spatial = self._norm(spatial, "resampler.output_norm")
        query = mx.broadcast_to(self.weights["summary_query"],
                                (len(previous), 1, self.config.width))
        summary = self._attention(query, spatial, spatial, "summary_attention")[:, 0]
        previous_features = mx.array([
            [*(float(key in action.keys) for key in KEY_VOCAB),
             *(float(button in action.mouse_buttons) for button in MOUSE_BUTTONS),
             action.dx / max(MOUSE_BINS), action.dy / max(MOUSE_BINS), float(action.wheel)]
            for action in previous
        ], dtype=mx.float32)
        summary = (summary + self._linear(previous_features, "previous_action_embedding")
                   + self._linear(mx.full((len(previous), 1), 0.1), "time_embedding"))
        if instruction_context is not None:
            summary = summary + instruction_context
        return spatial, self._norm(summary, "summary_norm")

    def _block(self, tokens, name, past=None, mask=None):
        width, heads = self.config.width, self.config.heads
        qkv = self._linear(self._norm(tokens, name + ".attention_norm"), name + ".qkv")
        qkv = qkv.reshape(tokens.shape[0], 3, heads, width // heads).transpose(1, 2, 0, 3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        if past is not None:
            keys = mx.concatenate((past[0], keys), axis=1)
            values = mx.concatenate((past[1], values), axis=1)
        attended = mx.fast.scaled_dot_product_attention(
            queries[None], keys[None], values[None], scale=(width // heads) ** -0.5,
            mask=mask,
        )[0].transpose(1, 0, 2).reshape(tokens.shape[0], width)
        tokens = tokens + self._linear(attended, name + ".output")
        tokens = tokens + self._mlp(self._norm(tokens, name + ".mlp_norm"), name + ".mlp")
        return tokens, (keys, values)

    def _temporal(self, summary):
        memory = self.weights["temporal.initial_memory"]
        outputs = []
        segment_length, memory_tokens = self.config.segment_length, self.config.memory_tokens
        for start in range(0, summary.shape[0], segment_length):
            segment = summary[start:start + segment_length]
            count = segment.shape[0]
            tokens = mx.concatenate((memory + self.weights["temporal.read_positions"],
                                     segment + self.weights["temporal.decision_positions"][:count]))
            indices = mx.arange(memory_tokens + count)
            allowed = ((indices[:, None] >= indices[None])
                       | ((indices[:, None] < memory_tokens) & (indices[None] < memory_tokens)))
            mask = mx.where(allowed, 0.0, -float("inf"))
            caches = []
            for index in range(self.config.layers):
                tokens, cache = self._block(tokens, f"temporal.blocks.{index}", mask=mask)
                caches.append(cache)
            outputs.append(self._norm(tokens[memory_tokens:], "temporal.output_norm"))
            if start + segment_length < summary.shape[0]:
                memory = self.weights["temporal.write_tokens"]
                for index, cache in enumerate(caches):
                    memory, _ = self._block(memory, f"temporal.blocks.{index}", past=cache)
                memory = self._norm(memory, "temporal.output_norm")
        return mx.concatenate(outputs)

    def _gru(self, embedded, state):
        name = "decoder.state_cell."
        incoming = (embedded @ self.weights[name + "weight_ih"].T
                    + self.weights[name + "bias_ih"])
        hidden = state @ self.weights[name + "weight_hh"].T + self.weights[name + "bias_hh"]
        ir, iz, inn = mx.split(incoming, 3, axis=-1)
        hr, hz, hn = mx.split(hidden, 3, axis=-1)
        reset, update = mx.sigmoid(ir + hr), mx.sigmoid(iz + hz)
        candidate = mx.tanh(inn + reset * hn)
        return (1 - update) * candidate + update * state

    @staticmethod
    def _categorical(logits, picked):
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        selected = mx.take_along_axis(log_probs, picked[:, None], axis=1)[:, 0]
        # Match Torch Categorical's finite clamp on excluded support. Avoid
        # 0 * -inf in both the forward entropy and its derivatives.
        safe_log_probs = mx.maximum(log_probs, mx.finfo(logits.dtype).min)
        entropy = -mx.sum(mx.exp(log_probs) * safe_log_probs, axis=-1)
        return selected, entropy

    def _decode(self, features, actions, trace=None):
        state = mx.tanh(self._linear(features, "decoder.context"))
        count = len(actions)
        logp, entropy = mx.zeros((count,)), mx.zeros((count,))
        active = mx.ones((count,), dtype=mx.bool_)
        last = mx.full((count,), -1, dtype=mx.int32)
        targets = [tuple(KEY_VOCAB.index(key) for key in action.keys) for action in actions]
        eos = len(KEY_VOCAB)
        for step in range(max(map(len, targets)) + 1):
            permitted = (mx.arange(eos)[None] > last[:, None]) & active[:, None]
            mask = mx.concatenate((permitted, mx.ones((count, 1), dtype=mx.bool_)), axis=1)
            logits = self._linear(state, "decoder.key_head")
            if self.motor_initialization == "balanced":
                prior = mx.concatenate((mx.zeros((count, eos)),
                                        ((last >= 0).astype(mx.float32) * 3.5)[:, None]), axis=1)
                logits = logits + prior
            logits = mx.where(mask, logits, -float("inf"))
            if trace is not None:
                trace.append((logits - mx.logsumexp(logits, axis=-1, keepdims=True), active))
            picked = mx.array([row[step] if step < len(row) else eos for row in targets])
            part_logp, part_entropy = self._categorical(logits, picked)
            logp, entropy = logp + part_logp * active, entropy + part_entropy * active
            following = self._gru(self.weights["decoder.key_embedding.weight"][picked], state)
            # EOS advances the GRU on its active step before stopping the prefix.
            state = mx.where(active[:, None], following, state)
            has_key = active & (picked != eos)
            last, active = mx.where(has_key, picked, last), has_key
        components = (
            ("mouse", [sum(1 << bit for bit, button in enumerate(MOUSE_BUTTONS)
                           if button in action.mouse_buttons) for action in actions]),
            ("dx", [MOUSE_BINS.index(action.dx) for action in actions]),
            ("dy", [MOUSE_BINS.index(action.dy) for action in actions]),
            ("wheel", [WHEEL_BINS.index(action.wheel) for action in actions]),
        )
        for name, indices in components:
            picked = mx.array(indices)
            logits = self._linear(state, f"decoder.{name}_head")
            if trace is not None:
                trace.append((logits - mx.logsumexp(logits, axis=-1, keepdims=True),
                              mx.ones((count,), dtype=mx.bool_)))
            part_logp, part_entropy = self._categorical(
                logits, picked,
            )
            logp, entropy = logp + part_logp, entropy + part_entropy
            if name != "wheel":
                state = self._gru(self.weights[f"decoder.{name}_embedding.weight"][picked], state)
        return logp, entropy

    def evaluate_episode(self, patches, actions: Sequence[PhysicalAction], *,
                         instruction_features=None, executed_actions=None,
                         return_decoder_trace=False) -> dict:
        """Differentiate a full episode, including every cross-segment memory write."""
        if not actions or any(not isinstance(action, PhysicalAction) for action in actions):
            raise ValueError("a nonempty episode requires PhysicalAction entries")
        patches = to_mlx(patches)
        if patches.ndim != 3 or patches.shape[-1] != 384 or len(actions) != patches.shape[0]:
            raise ValueError("patches must be [decisions, patches, 384] matching actions")
        executed = actions if executed_actions is None else executed_actions
        if len(executed) != len(actions) or any(
                not isinstance(action, PhysicalAction) for action in executed):
            raise ValueError("executed actions must match the complete episode")
        context = None
        if instruction_features is not None:
            if self.config.text_encoder == "none":
                raise ValueError("instruction features require an enabled text encoder")
            feature = to_mlx(instruction_features)
            if feature.shape != (384,):
                raise ValueError("instruction features must have shape [384]")
            context = self._linear(feature, "instruction_projection")
        spatial, summary = self._spatial_summary(
            patches, [PhysicalAction(), *executed[:-1]], context,
        )
        states = self._temporal(summary)
        actor = states + self._attention(states[:, None], spatial, spatial, "actor_attention")[:, 0]
        if context is not None:
            actor = actor + context
        trace = [] if return_decoder_trace else None
        logp, entropy = self._decode(self._norm(actor, "actor_norm"), actions, trace)
        result = {"log_prob": logp, "entropy": entropy,
                  "value": self._linear(states, "value_head")[:, 0]}
        if trace is not None:
            result["decoder_trace"] = trace
        return result

    def __call__(self, patches, actions, *, instruction_features=None, executed_actions=None):
        return self.evaluate_episode(patches, actions, instruction_features=instruction_features,
                                     executed_actions=executed_actions)
