"""Causal physical action decoder with a joint likelihood for sequence PPO."""

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from carryrl.agent.actions import KEY_VOCAB, MOUSE_BINS, MOUSE_BUTTONS, WHEEL_BINS, PhysicalAction


class AutoregressiveActionDecoder(nn.Module):
    """Predict a sorted held-key sequence including EOS, then mouse components.

    Entropy is the sum of conditional categorical entropies along the supplied or
    sampled prefix. It is a trajectory estimate, not an exact enumeration of the
    joint action entropy. PPO must clip the ratio of the returned joint log-probs.
    """

    def __init__(self, width: int = 256):
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        self.width = width
        self.eos_index = len(KEY_VOCAB)
        self.context = nn.Linear(width, width)
        self.state_cell = nn.GRUCell(width, width)
        self.key_embedding = nn.Embedding(len(KEY_VOCAB) + 1, width)
        self.key_head = nn.Linear(width, len(KEY_VOCAB) + 1)
        self.mouse_embedding = nn.Embedding(8, width)
        self.mouse_head = nn.Linear(width, 8)
        self.dx_embedding = nn.Embedding(len(MOUSE_BINS), width)
        self.dx_head = nn.Linear(width, len(MOUSE_BINS))
        self.dy_embedding = nn.Embedding(len(MOUSE_BINS), width)
        self.dy_head = nn.Linear(width, len(MOUSE_BINS))
        self.wheel_head = nn.Linear(width, len(WHEEL_BINS))
        for head in (self.key_head, self.mouse_head, self.dx_head, self.dy_head, self.wheel_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        self.motor_initialization = "legacy"
        self.initialize_motor_prior("legacy")

    @torch.no_grad()
    def initialize_motor_prior(self, mode: str) -> None:
        """Initialize a general input prior, without knowing which game is running.

        The balanced option favors short key holds, button releases and small
        mouse motion. Every physical command retains positive probability. Its
        soft prefix termination preference remains active after initialization;
        checkpoint loaders restore the mode separately from the learned tensors.
        """
        if mode not in {"legacy", "balanced"}:
            raise ValueError("motor initialization must be legacy or balanced")
        self.motor_initialization = mode
        for head in (self.key_head, self.mouse_head, self.dx_head,
                     self.dy_head, self.wheel_head):
            head.bias.zero_()
        if mode == "legacy":
            self.key_head.bias[self.eos_index] = 3.0
            return
        self.key_head.bias.fill_(-1.0)
        for key in ("up", "down", "left", "right", "w", "a", "s", "d"):
            self.key_head.bias[KEY_VOCAB.index(key)] = 1.5
        for key in ("space", "e", "enter", "escape", "tab"):
            self.key_head.bias[KEY_VOCAB.index(key)] = 0.5
        self.key_head.bias[self.eos_index] = 1.0
        button_probabilities = (0.06, 0.03, 0.02)
        for index in range(8):
            self.mouse_head.bias[index] = sum(math.log(p if index & (1 << bit) else 1 - p)
                                              for bit, p in enumerate(button_probabilities))
        for head in (self.dx_head, self.dy_head):
            for index, delta in enumerate(MOUSE_BINS):
                # A smooth prior has full support, including long camera turns.
                head.bias[index] = -math.log1p(abs(delta)) * 1.5
        for index, delta in enumerate(WHEEL_BINS):
            self.wheel_head.bias[index] = math.log(0.96 if delta == 0 else 0.02)

    def _features(self, features: torch.Tensor) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a tensor")
        if features.ndim != 2 or features.shape[1] != self.width or features.shape[0] < 1:
            raise ValueError(f"features must have shape [N, {self.width}] with N > 0")
        if not features.is_floating_point():
            raise TypeError("features must use a floating-point dtype")
        if not torch.isfinite(features).all():
            raise FloatingPointError("nonfinite decoder features")
        return torch.tanh(self.context(features))

    def _key_distribution(self, state, last, active):
        indices = torch.arange(len(KEY_VOCAB), device=state.device)
        permitted = (indices.unsqueeze(0) > last.unsqueeze(1)) & active.unsqueeze(1)
        mask = torch.cat((permitted, torch.ones_like(active).unsqueeze(1)), dim=1)
        logits = self.key_head(state)
        if self.motor_initialization == "balanced":
            prefix_prior = torch.zeros_like(logits)
            prefix_prior[:, self.eos_index] = (last >= 0).to(logits.dtype) * 3.5
            logits = logits + prefix_prior
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf))

    @staticmethod
    def _choose(distribution, deterministic):
        return distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()

    def _run(
        self, features: torch.Tensor, actions: Sequence[PhysicalAction] | None,
        deterministic: bool,
        trace: list | None = None,
    ) -> tuple[list[PhysicalAction], torch.Tensor, torch.Tensor]:
        state = self._features(features)
        count = state.shape[0]
        if actions is not None:
            if len(actions) != count:
                raise ValueError("one PhysicalAction is required per feature row")
            if any(not isinstance(action, PhysicalAction) for action in actions):
                raise TypeError("evaluate expects PhysicalAction entries")
        logp = features.new_zeros(count)
        entropy = features.new_zeros(count)
        active = torch.ones(count, dtype=torch.bool, device=state.device)
        last = torch.full((count,), -1, dtype=torch.long, device=state.device)
        picked_keys: list[list[str]] = [[] for _ in range(count)]
        key_targets = None
        if actions is not None:
            key_targets = [tuple(KEY_VOCAB.index(key) for key in action.keys) for action in actions]
        max_keys = len(KEY_VOCAB) if key_targets is None else max(map(len, key_targets))
        for step in range(max_keys + 1):
            distribution = self._key_distribution(state, last, active)
            if trace is not None:
                trace.append((distribution.logits, active))
            if key_targets is None:
                picked = self._choose(distribution, deterministic)
            else:
                picked = torch.tensor([
                    row[step] if step < len(row) else self.eos_index for row in key_targets
                ], device=state.device)
            logp = logp + distribution.log_prob(picked) * active
            entropy = entropy + distribution.entropy() * active
            following = self.state_cell(self.key_embedding(picked), state)
            state = torch.where(active.unsqueeze(1), following, state)
            has_key = active & (picked != self.eos_index)
            if key_targets is None:
                for row, index in enumerate(picked.detach().cpu().tolist()):
                    if index != self.eos_index:
                        picked_keys[row].append(KEY_VOCAB[index])
            last = torch.where(has_key, picked, last)
            active = has_key
            if key_targets is None and not bool(active.any()):
                break

        component_values = []
        components = (
            (self.mouse_head, self.mouse_embedding), (self.dx_head, self.dx_embedding),
            (self.dy_head, self.dy_embedding), (self.wheel_head, None),
        )
        for component, (head, embedding) in enumerate(components):
            distribution = Categorical(logits=head(state))
            if trace is not None:
                trace.append((distribution.logits, torch.ones_like(active)))
            if actions is None:
                picked = self._choose(distribution, deterministic)
            else:
                indices = []
                for action in actions:
                    if component == 0:
                        index = sum(1 << bit for bit, button in enumerate(MOUSE_BUTTONS)
                                    if button in action.mouse_buttons)
                    elif component == 1:
                        index = MOUSE_BINS.index(action.dx)
                    elif component == 2:
                        index = MOUSE_BINS.index(action.dy)
                    else:
                        index = WHEEL_BINS.index(action.wheel)
                    indices.append(index)
                picked = torch.tensor(indices, device=state.device)
            logp = logp + distribution.log_prob(picked)
            entropy = entropy + distribution.entropy()
            if actions is None:
                component_values.append(picked.detach().cpu().tolist())
            if embedding is not None:
                state = self.state_cell(embedding(picked), state)
        if actions is None:
            generated = []
            for row in range(count):
                buttons, dx, dy, wheel = (values[row] for values in component_values)
                generated.append(PhysicalAction(
                    keys=tuple(picked_keys[row]),
                    mouse_buttons=tuple(button for bit, button in enumerate(MOUSE_BUTTONS)
                                        if buttons & (1 << bit)),
                    dx=MOUSE_BINS[dx], dy=MOUSE_BINS[dy], wheel=WHEEL_BINS[wheel],
                ))
        else:
            generated = list(actions)
        return generated, logp, entropy

    def sample(
        self, features: torch.Tensor, deterministic: bool = False,
    ) -> tuple[list[PhysicalAction], torch.Tensor, torch.Tensor]:
        return self._run(features, None, deterministic)

    def evaluate(
        self, features: torch.Tensor, actions: Sequence[PhysicalAction],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, logp, entropy = self._run(features, actions, False)
        return logp, entropy

    def evaluate_with_trace(self, features: torch.Tensor, actions: Sequence[PhysicalAction]):
        """Conditional categorical log probabilities along expert action prefixes.

        Key steps include EOS; the mask excludes padding after each row's EOS.
        This trace is optional and adds no parameters or inference behavior.
        """
        trace = []
        _, logp, entropy = self._run(features, actions, False, trace)
        return logp, entropy, trace
