"""Real Metal gradient accumulation preserves the complete PPO minibatch."""

import weakref
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_mlx_training import assert_agent_close, assert_optimizer_close, optimizer, setup_case
from test_sequence_training import TinyLanguage
from torch import nn

import carryrl.training.mlx as training
from carryrl.agent.model import GeneralistAgent
from carryrl.training.sequence_config import SequenceRunConfig


@pytest.fixture
def mx():
    core = pytest.importorskip("mlx.core")
    if not core.metal.is_available():
        pytest.skip("MLX Metal unavailable")
    core.set_default_device(core.gpu)
    torch.set_num_threads(1)
    return core


def case():
    config, _, base = setup_case()
    config = replace(config, training_backend="mlx",
                     episode_instructions={task: ("remember",) for task in config.tasks},
                     agent=replace(config.agent, text_encoder="multilingual_e5_small"))
    agent = GeneralistAgent(config.agent, vision=nn.Linear(3, 384),
                            language=TinyLanguage()).eval()
    episodes = []
    for index, length in enumerate((1, 2, 3, 4)):
        episode = deepcopy(base[1])
        for name in ("patches", "frames", "actions", "rewards"):
            setattr(episode, name, getattr(episode, name)[:length])
        episode.terminated = [False] * (length - 1) + [True]
        episode.truncated = [False] * length
        episode.task = "memory" if index == 3 else "navigation"
        episode.instruction = "remember" if index == 3 else None
        with torch.no_grad():
            result = agent.evaluate_episode(episode.patches, episode.actions,
                                             instruction=episode.instruction)
        episode.old_log_prob = result["log_prob"] + .02
        episode.old_values = result["value"]
        episodes.append(episode)
    return config, agent, episodes


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("size", [1, 2, 3])
def test_accumulated_gradients_weights_moments_and_metrics_match_full_batch(
        mx, monkeypatch, balanced, size):
    config, initial, episodes = case()
    config = replace(config, ppo_task_balance=balanced)
    agents = [deepcopy(initial), deepcopy(initial)]
    optimizers = [optimizer(agent, config) for agent in agents]
    observed_gradients, active_sets = [], []
    original = training.MLXPPOBackend.apply_gradients

    def capture(self, gradients, max_grad_norm, active_names=None):
        observed_gradients.append({
            name: np.array(gradient) for name, gradient in gradients["weights"].items()})
        active_sets.append(active_names)
        return original(self, gradients, max_grad_norm, active_names=active_names)

    monkeypatch.setattr(training.MLXPPOBackend, "apply_gradients", capture)
    full = training.update_ppo_mlx(agents[0], optimizers[0], episodes, config)
    accumulated = training.update_ppo_mlx(
        agents[1], optimizers[1], episodes, replace(config, ppo_microbatch_episodes=size))
    for first, second in zip(observed_gradients[:2], observed_gradients[2:], strict=True):
        for name in first:
            np.testing.assert_allclose(first[name], second[name], atol=5e-6, rtol=5e-4)
    assert all({"instruction_projection.weight", "temporal.write_tokens"} <= names
               for names in active_sets)
    assert_agent_close(*agents)
    assert_optimizer_close(*optimizers)
    for canonical in optimizers:
        assert all(int(state["step"]) == 2 for state in canonical.state.values())
    def compare(actual, expected):
        if isinstance(expected, dict):
            assert set(actual) == set(expected)
            for name in expected:
                compare(actual[name], expected[name])
        elif isinstance(expected, list):
            assert len(actual) == len(expected)
            for left, right in zip(actual, expected, strict=True):
                compare(left, right)
        elif isinstance(expected, float):
            assert actual == pytest.approx(expected, abs=5e-6, rel=5e-5)
        else:
            assert actual == expected

    for name in full.keys() - {"epochs"}:
        compare(accumulated[name], full[name])
    for expected, actual in zip(full["epochs"], accumulated["epochs"], strict=True):
        for name in expected:
            assert actual[name] == pytest.approx(expected[name], abs=5e-6, rel=5e-5)


def test_late_microbatch_kl_rejects_without_any_parameter_or_optimizer_mutation(mx):
    config, agent, episodes = case()
    config = replace(config, n_epochs=1, ppo_microbatch_episodes=1)
    canonical = optimizer(agent, config)
    backend = training.MLXPPOBackend(agent, canonical)
    training.update_ppo_mlx(agent, canonical, episodes, config, backend=backend)
    before_weights = deepcopy(agent.state_dict())
    before_optimizer = deepcopy(canonical.state_dict())
    for episode in episodes:
        with torch.no_grad():
            result = agent.evaluate_episode(episode.patches, episode.actions,
                                             instruction=episode.instruction)
        episode.old_log_prob = result["log_prob"] + .02
    episodes[-1].old_log_prob -= 3
    result = training.update_ppo_mlx(
        agent, canonical, episodes, replace(config, target_kl=.001), backend=backend)
    assert result["kl_early_stop"] and result["optimizer_steps"] == 0
    assert result["epochs"][0]["replayed_transitions"] == 10
    assert result["optimizer_task_transition_replays"] == {"navigation": 0, "memory": 0}
    for name, value in agent.state_dict().items():
        assert torch.equal(value, before_weights[name])
    assert canonical.state_dict()["param_groups"] == before_optimizer["param_groups"]
    for identifier, state in canonical.state_dict()["state"].items():
        for name, value in state.items():
            assert torch.equal(value, before_optimizer["state"][identifier][name])


def test_kl_threshold_applies_to_full_objective_not_each_microbatch(mx):
    config, agent, episodes = case()
    config = replace(config, n_epochs=1, ppo_microbatch_episodes=1, target_kl=.1,
                     ppo_task_balance=False)
    # This one-transition episode has KL ~0.29 alone, ~0.029 in the full batch.
    episodes[0].old_log_prob += .85
    result = training.update_ppo_mlx(agent, optimizer(agent, config), episodes, config)
    assert not result["kl_early_stop"] and result["optimizer_steps"] == 1
    assert 0 < result["epochs"][0]["approx_kl"] < config.target_kl


def test_microbatch_transfer_is_bounded_and_inactive_parameters_are_untouched(mx, monkeypatch):
    config, agent, episodes = case()
    episodes = episodes[:2] * 3
    config = replace(config, n_epochs=1, ppo_microbatch_episodes=2)
    canonical = optimizer(agent, config)
    untouched = {name: value.detach().clone() for name, value in agent.named_parameters()
                 if name in {"instruction_projection.weight", "temporal.write_tokens"}}
    references, sizes = [], []
    original = training.prepare_episodes

    class Batch(list):
        pass

    def prepare(*args, **kwargs):
        assert all(reference() is None for reference in references)
        prepared = Batch(original(*args, **kwargs))
        references.append(weakref.ref(prepared))
        sizes.append(len(prepared))
        return prepared

    monkeypatch.setattr(training, "prepare_episodes", prepare)
    result = training.update_ppo_mlx(agent, canonical, episodes, config)
    assert sizes == [2, 2, 2]
    assert all(reference() is None for reference in references)
    assert result["optimizer_steps"] == 1
    for name, parameter in agent.named_parameters():
        if name in untouched:
            assert torch.equal(parameter, untouched[name])
            assert parameter not in canonical.state
    assert all(int(state["step"]) == 1 for state in canonical.state.values())


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_microbatch_size_must_be_nonnegative_integer(value):
    with pytest.raises(ValueError, match="ppo_microbatch_episodes"):
        SequenceRunConfig(training_backend="mlx", ppo_microbatch_episodes=value)


def test_microbatch_torch_configuration_is_rejected():
    with pytest.raises(ValueError, match="ppo_microbatch_episodes requires training_backend=mlx"):
        SequenceRunConfig(ppo_microbatch_episodes=1)
    assert SequenceRunConfig().ppo_microbatch_episodes == 0
    config = SequenceRunConfig(training_backend="mlx", ppo_microbatch_episodes=4)
    assert SequenceRunConfig.from_dict(config.to_dict()).ppo_microbatch_episodes == 4
