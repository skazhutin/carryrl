"""Regression checks for a separately registered frozen stochastic diagnostic."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

script_path = Path(__file__).resolve().parents[1] / "scripts/diagnose_stochastic.py"
spec = importlib.util.spec_from_file_location("diagnose_stochastic", script_path)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
aggregate = diagnostic.aggregate
assert_frozen = diagnostic.assert_frozen
frozen_state = diagnostic.frozen_state
run_episode = diagnostic.run_episode


class ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.1, 0.2, -0.1, 0.0, 0.3]))

    def set_training_mode(self, mode):
        self.train(mode)


class ToyAgent:
    def __init__(self):
        self.policy = ToyPolicy()
        self.num_timesteps = 123

    def predict(self, pixels, deterministic):
        assert isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8
        assert pixels.shape == (64, 64, 3)
        assert deterministic is False
        self.policy.set_training_mode(False)
        return torch.distributions.Categorical(logits=self.policy.logits).sample().item(), None


def test_stochastic_episode_is_reproducible_frozen_and_preserves_global_rng():
    model = ToyAgent()
    before = frozen_state(model)
    rng = torch.random.get_rng_state().clone()
    first = run_episode(model, level_seed=1000, action_seed=5000, max_steps=5)
    second = run_episode(model, level_seed=1000, action_seed=5000, max_steps=5)
    assert first["actions"] == second["actions"]
    assert first["torch_episode_seed"] == 5000 * 65536 + 1000
    assert first["level_id"] == second["level_id"]
    assert first["success"] == second["success"]
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert model.policy.training is True
    assert_frozen(model, before)


def test_random_control_uses_same_declared_rng_and_repeats_exactly():
    first = run_episode(None, level_seed=1001, action_seed=5001, max_steps=5)
    second = run_episode(None, level_seed=1001, action_seed=5001, max_steps=5)
    assert first["actions"] == second["actions"]
    assert first["policy"] == "uniform_torch_categorical"
    assert all(0 <= action < 5 for action in first["actions"])


def test_frozen_check_detects_parameter_and_counter_changes():
    model = ToyAgent()
    before = frozen_state(model)
    model.num_timesteps += 1
    with pytest.raises(AssertionError, match="training counter"):
        assert_frozen(model, before)
    model.num_timesteps -= 1
    with torch.no_grad():
        model.policy.logits[0] += 1
    with pytest.raises(AssertionError, match="policy tensors"):
        assert_frozen(model, before)


def test_unknowns_stay_in_denominator_and_prevent_favorable_return_subset():
    rows = [
        {"success": True, "return": 1.0, "observed_steps": 1},
        {"success": False, "return": -0.64, "observed_steps": 64},
        {"success": None, "return": None, "observed_steps": 3},
    ]
    summary = aggregate(rows)
    assert summary["episodes_started"] == 3
    assert summary["unknown_outcomes"] == 1
    assert summary["success_rate_lower"] == 1 / 3
    assert summary["success_rate_upper"] == 2 / 3
    assert summary["mean_return"] is None


def test_predictor_failure_retains_unknown_and_restores_rng_and_mode():
    class BrokenAgent(ToyAgent):
        def predict(self, pixels, deterministic):
            self.policy.set_training_mode(False)
            raise RuntimeError("synthetic predictor failure")

    model = BrokenAgent()
    before = frozen_state(model)
    rng = torch.random.get_rng_state().clone()
    row = run_episode(model, level_seed=1000, action_seed=5000, max_steps=5)
    assert row["success"] is None and row["return"] is None
    assert row["error"]["type"] == "RuntimeError"
    assert model.policy.training is True
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert_frozen(model, before)
    assert aggregate([row])["unknown_outcomes"] == 1
