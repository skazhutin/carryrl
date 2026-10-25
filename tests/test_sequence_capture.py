"""Capture scope changes acquisition only, preserving frozen evaluation cohorts."""

import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from carryrl.agent.actions import PhysicalAction
from carryrl.evaluation.trajectory_diagnostics import diagnose_frozen_trajectory
from carryrl.training import sequence
from carryrl.training.sequence_config import SequenceRunConfig


class ConstantAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = 0

    def reset_episode(self, *, instruction=None):
        pass

    def act(self, frame, *, deterministic):
        self.calls += 1
        return SimpleNamespace(action=PhysicalAction())


class OutcomeEnv:
    """Four deliberately distinct outcome categories with two real input ticks."""

    def __init__(self, task, **kwargs):
        self.task = task

    def reset(self, *, seed, **kwargs):
        self.index = seed - 1_000_000
        self.steps = 0
        self.frame = np.zeros((96, 96, 3), dtype=np.uint8)
        self.frame[0, 0] = (self.index, 0, 0)
        return self.frame.copy(), {"level_id": f"level{self.index}",
                                  "semantic_level_id": f"semantic{self.index}",
                                  "runtime_fingerprint": f"runtime{self.index}"}

    def step(self, action):
        self.steps += 1
        self.frame[0, 1] = self.steps
        return self.frame.copy(), 0, self.steps == 2 and self.index < 2, \
            self.steps == 2 and self.index >= 2, {}

    def close(self):
        pass


def outcome(frame):
    index = int(frame[0, 0, 0])
    return {"known": index != 3, "success": index == 0,
            "failure": index == 1, "score": 5 if index == 0 else 0}


def config(**changes):
    return SequenceRunConfig(tasks=("navigation", "reaction"), eval_episodes_per_task=4,
                             max_episode_steps=2, vision_cache_frames=0, **changes)


@pytest.mark.parametrize("scope,indices", [("first_per_task", [0]),
                                          ("failed", [1, 2, 3]), ("all", [0, 1, 2, 3])])
def test_capture_selection_preserves_cohort_order_random_actions_and_rng(scope, indices):
    cfg = config()
    agent, reward = ConstantAgent(), nn.Linear(1, 1)
    before_hash = sequence.tensor_hash(agent)
    traces = []

    class TrackedEnv(OutcomeEnv):
        def reset(self, **kwargs):
            self.actions = []
            traces.append(self.actions)
            return super().reset(**kwargs)

        def step(self, action):
            self.actions.append(action)
            return super().step(action)

    baseline = sequence.evaluate_sequence(agent, reward, cfg,
        env_factory=TrackedEnv, outcome_fn=outcome)
    baseline_traces = list(traces)
    traces.clear()
    calls = agent.calls
    torch_state, python_state, numpy_state = (torch.get_rng_state(), random.getstate(),
                                             np.random.get_state())
    captures = []
    result = sequence.evaluate_sequence(agent, reward, cfg, env_factory=TrackedEnv,
        outcome_fn=outcome, capture_scope=scope,
        on_playback=lambda record, frames, actions: captures.append((record, frames, actions)))
    assert traces == baseline_traces  # Includes actual generic random physical commands.
    assert result["records"] == baseline["records"]
    assert result["summaries"] == baseline["summaries"]
    assert result["protocol_sha256"] == baseline["protocol_sha256"]
    assert agent.calls - calls == 16  # Eight learned episodes, two actions each; no recapture.
    assert [(row["task"], row["episode_index"]) for row, _, _ in captures] == [
        (task, index) for task in cfg.tasks for index in indices]
    for row, frames, actions in captures:
        assert row["policy"] == "frozen_deterministic"
        assert len(frames) == len(actions) + 1 == row["steps"] + 1
        assert all(action == PhysicalAction().to_dict() for action in actions)
        assert outcome(frames[-1]) == row["outcome"]
        assert all(not np.shares_memory(a, b) for a, b in zip(frames, frames[1:]))
    assert sequence.tensor_hash(agent) == before_hash and agent.training
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    after_numpy = np.random.get_state()
    assert numpy_state[0] == after_numpy[0]
    assert np.array_equal(numpy_state[1], after_numpy[1])
    assert numpy_state[2:] == after_numpy[2:]


def test_default_scope_keeps_only_first_deterministic_with_stochastic_enabled():
    captures = []
    sequence.evaluate_sequence(ConstantAgent(), nn.Linear(1, 1), config(eval_stochastic=True),
        env_factory=OutcomeEnv, outcome_fn=outcome,
        on_playback=lambda record, frames, actions: captures.append(record))
    assert [(row["policy"], row["episode_index"]) for row in captures] == [
        ("frozen_deterministic", 0), ("frozen_deterministic", 0)]


@pytest.mark.parametrize("scope,stochastic", [("invalid", False), ("all", True), ("failed", True)])
def test_invalid_scope_fails_before_model_inspection(scope, stochastic, monkeypatch):
    monkeypatch.setattr(sequence, "tensor_hash", lambda model: pytest.fail("inspected model"))
    with pytest.raises(ValueError, match="capture|playback"):
        sequence.evaluate_sequence(None, None, config(eval_stochastic=stochastic),
                                   capture_scope=scope)


def test_callback_failure_restores_frozen_state_and_rng():
    agent, reward = ConstantAgent(), nn.Linear(1, 1)
    rng_state = torch.get_rng_state()
    before = sequence.tensor_hash(agent)

    def broken(*args):
        torch.rand(10)
        raise OSError("capture disk failed")

    with pytest.raises(OSError, match="disk failed"):
        sequence.evaluate_sequence(agent, reward, config(), env_factory=OutcomeEnv,
            outcome_fn=outcome, capture_scope="failed", on_playback=broken)
    assert agent.training and sequence.tensor_hash(agent) == before
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_new_scope_captures_authenticatable_actual_atomic_trajectory():
    cfg = replace(config(), tasks=("navigation",), eval_episodes_per_task=2)
    captures = []
    evaluated = sequence.evaluate_sequence(ConstantAgent(), nn.Linear(1, 1), cfg,
        capture_scope="failed",
        on_playback=lambda record, frames, actions: captures.append((record, frames, actions)))
    assert len(captures) == 2
    assert len(evaluated["records"]) == 4
    for row, frames, actions in captures:
        assert row["quality"]["commands"] == 2
        assert row["truncated"] and not row["terminated"]
        diagnostic = diagnose_frozen_trajectory(cfg, row, frames, actions)
        assert diagnostic["steps"] == len(diagnostic["actual_actions"]) == 2
        assert diagnostic["replay_verified"] is True


def test_expanded_capture_never_reencodes_or_changes_feature_cache():
    class CachedAgent(ConstantAgent):
        def __init__(self):
            super().__init__()
            self.vision = nn.Identity()
            self.encodes = 0

        def encode_frames(self, frames):
            self.encodes += 1
            return frames[:, :1, :1].float()

        def act_from_patches(self, patches, *, deterministic):
            return self.act(None, deterministic=deterministic)

    agent, reward = CachedAgent(), nn.Linear(1, 1)
    cfg = replace(config(), vision_cache_frames=32)
    baseline = sequence.evaluate_sequence(agent, reward, cfg,
        env_factory=OutcomeEnv, outcome_fn=outcome)
    encodes = agent.encodes
    captures = []
    expanded = sequence.evaluate_sequence(agent, reward, cfg,
        env_factory=OutcomeEnv, outcome_fn=outcome, capture_scope="all",
        on_playback=lambda record, frames, actions: captures.append(record))
    assert expanded["records"] == baseline["records"]
    assert expanded["vision_cache"] == baseline["vision_cache"]
    assert expanded["vision_cache"]["hits"] > 0
    assert agent.encodes - encodes == encodes
    assert len(captures) == 8
