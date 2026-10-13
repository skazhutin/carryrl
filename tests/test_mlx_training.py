"""Whole-episode Metal PPO, canonical AdamW updates and continuation parity."""

import builtins
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.model import AgentConfig, GeneralistAgent
from carryrl.training.sequence import Episode, _ppo_targets, update_ppo
from carryrl.training.sequence_config import SequenceRunConfig


@pytest.fixture
def mx():
    core = pytest.importorskip("mlx.core")
    if not core.metal.is_available():
        pytest.skip("MLX Metal unavailable")
    core.set_default_device(core.gpu)
    torch.set_num_threads(1)
    return core


def setup_case():
    torch.manual_seed(17)
    config = SequenceRunConfig(
        agent=AgentConfig(pretrained=False, image_size=28, width=16, layers=1,
                          heads=2, spatial_tokens=2, memory_tokens=2, segment_length=2),
        tasks=("navigation", "memory"), max_episode_steps=4, n_epochs=2,
        learning_rate=3e-4, weight_decay=.03, max_grad_norm=.7,
        ppo_task_balance=True, ppo_minibatch_episodes=0,
    )
    agent = GeneralistAgent(config.agent, vision=nn.Linear(3, 384)).eval()
    actions = [PhysicalAction(), PhysicalAction(("left",), ("left",)),
               PhysicalAction(("up",), (), 4, -4, 0), PhysicalAction(("right",))]
    episodes = []
    for task, length in zip(config.tasks, (3, 4), strict=True):
        patches = torch.randn(length, 4, 384)
        chosen = actions[:length]
        with torch.no_grad():
            output = agent.evaluate_episode(patches, chosen)
        episodes.append(Episode(
            patches=patches, frames=np.zeros((length, 1, 1, 3), dtype=np.uint8),
            actions=chosen, old_log_prob=output["log_prob"] + .02,
            old_values=output["value"], rewards=torch.linspace(-.2, .8, length),
            terminated=[False] * (length - 1) + [True], truncated=[False] * length,
            bootstrap=0., task=task, seed=17, outcome={},
        ))
    return config, agent, episodes


def optimizer(agent, config):
    return torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad],
                             lr=config.learning_rate, weight_decay=config.weight_decay)


def assert_optimizer_close(first, second):
    left, right = first.state_dict(), second.state_dict()
    assert left["param_groups"] == right["param_groups"]
    assert set(left["state"]) == set(right["state"])
    for identifier, values in left["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, right["state"][identifier][name],
                                       atol=3e-7, rtol=2e-4)


def assert_agent_close(first, second):
    for name, parameter in first.named_parameters():
        other = dict(second.named_parameters())[name]
        # Attention key bias cancels algebraically in softmax. Backend rounding
        # leaves ~1e-9 gradients; Adam's 1e-8 epsilon amplifies these null directions.
        attention_bias = name.endswith(("in_proj_bias", "qkv.bias"))
        torch.testing.assert_close(parameter, other,
                                   atol=1e-4 if attention_bias else 4e-6, rtol=3e-4)


def test_full_task_balanced_ppo_loss_and_every_gradient_match_torch(mx):
    import mlx.nn as mlx_nn

    from carryrl.training.mlx import MLXPPOBackend, ppo_loss, prepare_episodes

    config, agent, episodes = setup_case()
    backend = MLXPPOBackend(agent, optimizer(agent, config))
    _, counts, advantages, returns = _ppo_targets(episodes, config)
    prepared = prepare_episodes(agent, episodes, advantages, returns)
    (loss, metrics, valid), gradients = mlx_nn.value_and_grad(
        backend.policy, lambda: ppo_loss(backend.policy, prepared, config))()
    mx.eval(loss, metrics, valid, gradients)
    torch_loss = 0.
    for episode, advantage, target in zip(episodes, advantages, returns, strict=True):
        result = agent.evaluate_episode(episode.patches, episode.actions)
        ratio = (result["log_prob"] - episode.old_log_prob).exp()
        policy = -torch.minimum(ratio * advantage,
                                ratio.clamp(1 - config.clip_range, 1 + config.clip_range)
                                * advantage)
        value = .5 * (result["value"] - target).square()
        torch_loss += (policy + config.value_coef * value
                       - config.entropy_coef * result["entropy"]).sum() / (
                           len(counts) * counts[episode.task])
    torch_loss.backward()
    assert bool(valid)
    assert float(loss) == pytest.approx(float(torch_loss.detach()), abs=2e-6)
    for name, parameter in agent.named_parameters():
        if parameter.requires_grad:
            gradient = torch.from_numpy(np.array(gradients["weights"][name]))
            expected = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            torch.testing.assert_close(gradient, expected, atol=4e-6, rtol=5e-4)


def test_ppo_adamw_updates_canonical_moments_and_resume_match_torch(mx):
    from carryrl.training.mlx import MLXPPOBackend

    config, torch_agent, episodes = setup_case()
    mlx_agent = deepcopy(torch_agent)
    torch_optimizer, mlx_optimizer = (optimizer(agent, config)
                                      for agent in (torch_agent, mlx_agent))
    metal_config = replace(config, training_backend="mlx")
    backend = MLXPPOBackend(mlx_agent, mlx_optimizer)
    torch_metrics = update_ppo(torch_agent, torch_optimizer, episodes, config)
    metal_metrics = update_ppo(mlx_agent, mlx_optimizer, episodes, metal_config, backend=backend)
    assert metal_metrics["optimizer_steps"] == torch_metrics["optimizer_steps"] == 2
    assert metal_metrics["optimizer_task_loss_weights"] == {"navigation": .5, "memory": .5}
    assert_agent_close(torch_agent, mlx_agent)
    assert_optimizer_close(torch_optimizer, mlx_optimizer)
    # Reconstruct the Metal backend solely from canonical checkpoint weights/moments.
    resumed_agent = deepcopy(mlx_agent)
    resumed_optimizer = optimizer(resumed_agent, config)
    resumed_optimizer.load_state_dict(deepcopy(mlx_optimizer.state_dict()))
    resumed = MLXPPOBackend(resumed_agent, resumed_optimizer)
    update_ppo(mlx_agent, mlx_optimizer, episodes, metal_config, backend=backend)
    update_ppo(resumed_agent, resumed_optimizer, episodes, metal_config, backend=resumed)
    update_ppo(torch_agent, torch_optimizer, episodes, config)
    assert_agent_close(mlx_agent, resumed_agent)
    assert_optimizer_close(mlx_optimizer, resumed_optimizer)
    assert_agent_close(torch_agent, resumed_agent)
    assert_optimizer_close(torch_optimizer, resumed_optimizer)
    for episode in episodes:
        torch_result = torch_agent.evaluate_episode(episode.patches, episode.actions)
        mlx_result = resumed_agent.evaluate_episode(episode.patches, episode.actions)
        for name in ("log_prob", "entropy", "value"):
            torch.testing.assert_close(torch_result[name], mlx_result[name],
                                       atol=2e-5, rtol=1e-5)


def test_kl_rejection_preserves_weights_and_existing_adamw_moments(mx):
    from carryrl.training.mlx import MLXPPOBackend

    config, agent, episodes = setup_case()
    canonical = optimizer(agent, config)
    config = replace(config, training_backend="mlx", n_epochs=1)
    backend = MLXPPOBackend(agent, canonical)
    update_ppo(agent, canonical, episodes, config, backend=backend)
    before_agent, before_optimizer = deepcopy(agent), deepcopy(canonical.state_dict())
    for episode in episodes:
        episode.old_log_prob = episode.old_log_prob - 3
    result = update_ppo(agent, canonical, episodes, replace(config, target_kl=.001),
                        backend=backend)
    assert result["kl_early_stop"] and result["optimizer_steps"] == 0
    for name, parameter in agent.named_parameters():
        assert torch.equal(parameter, dict(before_agent.named_parameters())[name])
    assert before_optimizer["param_groups"] == canonical.state_dict()["param_groups"]
    for identifier, state in before_optimizer["state"].items():
        for name, value in state.items():
            assert torch.equal(value, canonical.state_dict()["state"][identifier][name])


def test_nonfinite_gradients_do_not_mutate_policy_or_moments(mx):
    from carryrl.training.mlx import MLXPPOBackend

    config, agent, _ = setup_case()
    canonical = optimizer(agent, config)
    backend = MLXPPOBackend(agent, canonical)
    before = dict(backend.policy.weights)
    gradients = {name: mx.full(value.shape, float("nan")) for name, value in before.items()}
    with pytest.raises(FloatingPointError, match="gradient norm"):
        backend.apply_gradients({"weights": gradients}, config.max_grad_norm)
    assert not backend.optimizer.state and not canonical.state
    assert all(value is backend.policy.weights[name] for name, value in before.items())


def test_partial_gradients_preserve_unused_parameter_and_per_parameter_step(mx):
    from carryrl.training.mlx import MLXPPOBackend

    config, agent, _ = setup_case()
    canonical = optimizer(agent, config)
    backend = MLXPPOBackend(agent, canonical)
    critic = np.array(backend.policy.weights["value_head.weight"])
    gradients = {name: mx.ones_like(value) for name, value in backend.policy.weights.items()}
    active = set(gradients) - {"value_head.weight", "value_head.bias"}
    backend.apply_gradients({"weights": gradients}, .7, active_names=active)
    backend.sync_to_torch()
    np.testing.assert_array_equal(critic, np.array(backend.policy.weights["value_head.weight"]))
    assert "value_head.weight" not in backend.optimizer.state
    assert agent.value_head.weight not in canonical.state


def test_missing_optional_mlx_raises_explicit_backend_error(monkeypatch):
    from carryrl.training.mlx import require_mlx

    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == "mlx.core":
            raise ImportError("mlx unavailable")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(RuntimeError, match="optional mlx dependency"):
        require_mlx()


@pytest.mark.parametrize("options", [{"training_backend": "auto"},
                                    {"training_backend": "mlx", "device": "mps"}])
def test_training_backend_is_explicit_and_validated(options):
    with pytest.raises(ValueError, match="training_backend|MLX training"):
        SequenceRunConfig(**options)


def test_adamw_bridge_matches_torch_with_identical_gradients_and_restored_state(mx):
    from carryrl.training.mlx import MLXAdamW

    torch.manual_seed(32)
    model = nn.Linear(7, 5)
    canonical = torch.optim.AdamW(model.parameters(), lr=.002, betas=(.8, .94),
                                 eps=1e-7, weight_decay=.08)
    reference = deepcopy(model)
    torch_optimizer = torch.optim.AdamW(reference.parameters(), lr=.002, betas=(.8, .94),
                                        eps=1e-7, weight_decay=.08)
    bridge = MLXAdamW(model, canonical)
    weights = {name: mx.array(parameter.detach().numpy())
               for name, parameter in model.named_parameters()}
    for step in range(4):
        gradients = {}
        for name, parameter in reference.named_parameters():
            gradient = torch.linspace(.01, .7, parameter.numel()).reshape(parameter.shape)
            gradient = gradient * (1 if step % 2 else -.5)
            parameter.grad = gradient
            gradients[name] = mx.array(gradient.numpy())
        weights = bridge.step(weights, gradients)
        torch_optimizer.step()
        bridge.sync_to_torch()
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(parameter, torch.from_numpy(np.array(weights[name])),
                                       atol=1e-7, rtol=1e-6)
        assert_optimizer_close(torch_optimizer, canonical)
        if step == 1:
            bridge = MLXAdamW(model, canonical)


def test_short_episode_preserves_unused_memory_write_parameter(mx):
    from carryrl.training.mlx import MLXPPOBackend

    config, agent, episodes = setup_case()
    config = replace(config, training_backend="mlx", n_epochs=1)
    episode = episodes[0]
    episode.patches = episode.patches[:2]
    episode.actions = episode.actions[:2]
    for name in ("old_log_prob", "old_values", "rewards"):
        setattr(episode, name, getattr(episode, name)[:2])
    episode.terminated, episode.truncated = [False, True], [False, False]
    canonical = optimizer(agent, config)
    before = agent.temporal.write_tokens.detach().clone()
    update_ppo(agent, canonical, [episode], config, backend=MLXPPOBackend(agent, canonical))
    assert torch.equal(before, agent.temporal.write_tokens)
    assert agent.temporal.write_tokens not in canonical.state


def test_checkpoint_canonical_optimizer_resumes_metal_without_state_reset(mx, tmp_path):
    from carryrl.training.mlx import MLXPPOBackend
    from carryrl.training.sequence import load_sequence_checkpoint, save_sequence_checkpoint

    config, agent, episodes = setup_case()
    config = replace(config, training_backend="mlx", n_epochs=1)
    canonical = optimizer(agent, config)
    backend = MLXPPOBackend(agent, canonical)
    update_ppo(agent, canonical, episodes, config, backend=backend)
    reward = nn.Linear(1, 1).eval().requires_grad_(False)
    checkpoint = save_sequence_checkpoint(agent, canonical, reward, config, tmp_path,
                                          rollout=1, decisions=7, revision=1)
    saved = torch.load(checkpoint / "state.pt", weights_only=True)
    assert saved["training_backend"] == "mlx"
    assert saved["optimizer_state"]["state"]
    resumed_config = replace(config, total_rollouts=2)
    restored, restored_optimizer, _, counters = load_sequence_checkpoint(
        checkpoint, resumed_config,
        agent_factory=lambda architecture: GeneralistAgent(architecture, vision=nn.Linear(3, 384)),
        reward_factory=lambda _: nn.Linear(1, 1))
    assert counters["weights_revision"] == 1
    assert_optimizer_close(canonical, restored_optimizer)
    for identifier, state in canonical.state_dict()["state"].items():
        for name, value in state.items():
            assert torch.equal(value, restored_optimizer.state_dict()["state"][identifier][name])
    resumed_backend = MLXPPOBackend(restored, restored_optimizer)
    update_ppo(agent, canonical, episodes, config, backend=backend)
    update_ppo(restored, restored_optimizer, episodes, resumed_config, backend=resumed_backend)
    for name, parameter in agent.named_parameters():
        assert torch.equal(parameter, dict(restored.named_parameters())[name])
    assert_optimizer_close(canonical, restored_optimizer)
