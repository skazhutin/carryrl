"""Fixed rollout objectives, exact auxiliary GAE attribution and read-only probes."""

import itertools
import random
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_mlx_training import assert_agent_close, assert_optimizer_close, optimizer, setup_case
from torch import nn

from carryrl.agent.actions import PhysicalAction
from carryrl.training.ppo_diagnostics import (
    advantage_diagnostics,
    objective_weights,
    raw_targets,
    torch_component_probe,
    weighted_moments,
)
from carryrl.training.sequence import Episode, _ppo_targets, update_ppo


def episode(task, rewards, values=None, *, auxiliary=None):
    length = len(rewards)
    return Episode(
        patches=torch.zeros(length, 1, 384), frames=np.zeros((length, 1, 1, 3), dtype=np.uint8),
        actions=[PhysicalAction()] * length, old_log_prob=torch.zeros(length),
        old_values=torch.zeros(length) if values is None else torch.tensor(values),
        rewards=torch.tensor(rewards), terminated=[False] * (length - 1) + [True],
        truncated=[False] * length, bootstrap=0., task=task, seed=0, outcome={},
        reward_components=auxiliary,
    )


class ConstantPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Parameter(torch.tensor(.0))
        self.critic = nn.Parameter(torch.tensor(.0))

    def evaluate_episode(self, patches, actions):
        return {"log_prob": self.actor.expand(len(actions)),
                "value": self.critic.expand(len(actions)),
                "entropy": (self.actor * .3).expand(len(actions))}


def test_global_task_weighted_normalization_uses_configured_floor():
    config, _, _ = setup_case()
    config = replace(config, advantage_std_floor=.2, gamma=1e-12)
    episodes = [episode("a", [.1, .1]), episode("b", [.2] * 6)]
    _, _, normalized, _ = _ppo_targets(episodes, config)
    # Equal tasks: shared mean=.15, true std=.05, floor=.2.
    torch.testing.assert_close(normalized[0], torch.tensor([-.25, -.25]))
    torch.testing.assert_close(normalized[1], torch.full((6,), .25))
    diagnostics = advantage_diagnostics(episodes, config)
    assert diagnostics["mean"] == pytest.approx(.15)
    assert diagnostics["std"] == pytest.approx(.05)
    assert diagnostics["floor_active"]
    assert diagnostics["denominator"] == pytest.approx(.2)
    assert diagnostics["auxiliary_counterfactual"]["status"] == "unavailable"


def test_counterfactual_uses_same_nonzero_values_and_exact_gae_boundaries():
    config, _, _ = setup_case()
    episodes = [episode("a", [.001, .003], [.2, -.3], auxiliary={
        "control_bonus": [.0002, .001], "unsupported_penalty": [-.0001, -.0002]}),
        episode("b", [.002, .001, .004], [-.1, .4, .2], auxiliary={
            "control_bonus": [0., .0002, .0001], "unsupported_penalty": [0., -.0001, 0.]})]
    episodes[1].terminated[-1], episodes[1].truncated[-1], episodes[1].bootstrap = False, True, .25
    result = advantage_diagnostics(episodes, config)
    actual = result["auxiliary_counterfactual"]
    rewards = [item.rewards - torch.tensor(item.reward_components["control_bonus"])
               - torch.tensor(item.reward_components["unsupported_penalty"]) for item in episodes]
    raw, _ = raw_targets(episodes, config)
    other, _ = raw_targets(episodes, config, rewards=rewards)
    mean, _, denominator = weighted_moments(episodes, raw, config)
    cf_mean, _, cf_denominator = weighted_moments(episodes, other, config)
    expected_differences, directs, rescales = [], [], []
    for before, after in zip(raw, other, strict=True):
        difference = (before - mean) / denominator - (after - cf_mean) / cf_denominator
        direct = ((before - after) - (mean - cf_mean)) / denominator
        rescale = (after - cf_mean) * (1 / denominator - 1 / cf_denominator)
        torch.testing.assert_close(difference, direct + rescale, atol=2e-7, rtol=2e-4)
        expected_differences.append(difference)
        directs.append(direct)
        rescales.append(rescale)
    for field, values in (("normalized_difference", expected_differences),
                          ("direct_centered_effect", directs),
                          ("denominator_rescaling_effect", rescales)):
        combined = torch.cat(values).double()
        assert actual[field]["rms"] == pytest.approx(float(combined.square().mean().sqrt()))
        assert actual[field]["abs_max"] == pytest.approx(float(combined.abs().max()))
    assert actual["old_values_held_fixed"] and actual["status"] == "measured"
    assert actual["mean"] == pytest.approx(float(cf_mean))
    assert actual["denominator"] == pytest.approx(float(cf_denominator))


def test_small_auxiliary_signal_and_floored_counterfactual_are_reported_without_nan():
    config, _, _ = setup_case()
    config = replace(config, gamma=1e-12, advantage_std_floor=1e-8)
    episodes = [episode("a", [0., 2e-9], auxiliary={
        "control_bonus": [0., 2e-9], "unsupported_penalty": [0., 0.]})]
    actual = advantage_diagnostics(episodes, config)
    counterfactual = actual["auxiliary_counterfactual"]
    assert actual["floor_active"] and counterfactual["floor_active"]
    assert actual["denominator"] == pytest.approx(1e-8)
    assert counterfactual["normalized_difference"]["rms"] == pytest.approx(.1)
    assert counterfactual["direct_centered_effect"]["rms"] == pytest.approx(.1)
    assert counterfactual["denominator_rescaling_effect"]["rms"] == 0
    assert counterfactual["sign_changes"] == 2
    assert counterfactual["tasks"]["a"]["raw_advantage"]["rms"] == 0


def test_rollout_episode_selection_correction_unbiased_for_unequal_lengths_and_counts():
    config, _, _ = setup_case()
    config = replace(config, ppo_normalization="rollout")
    episodes = [episode(task, [float(length)] * length)
                for task, length in (("a", 1), ("a", 3), ("b", 2), ("b", 4), ("b", 5))]
    selections = list(itertools.product((0, 1), (2, 3, 4)))
    actual = []
    for indices in selections:
        _, weights, masses = objective_weights(episodes, indices, config)
        actual.append(sum(float(episodes[index].rewards.sum()) * weight
                          for index, weight in zip(indices, weights, strict=True)))
        for index, weight in zip(indices, weights, strict=True):
            assert masses[episodes[index].task] == pytest.approx(weight * len(episodes[index]))
    expected = .5 * (10 / 4 + 45 / 11)
    assert sum(actual) / len(actual) == pytest.approx(expected)
    _, weights, _ = objective_weights(episodes, [0, 1, 2, 3, 4], config)
    assert sum(float(item.rewards.sum()) * weight
               for item, weight in zip(episodes, weights, strict=True)) == pytest.approx(expected)


def test_rollout_schedule_missing_task_rejected_before_any_update(monkeypatch):
    import carryrl.training.sequence as sequence

    config, _, _ = setup_case()
    config = replace(config, ppo_normalization="rollout", ppo_minibatch_episodes=2)
    episodes = [episode("a", [1.]), episode("b", [2.]), episode("a", [3.])]
    monkeypatch.setattr(sequence, "_episode_minibatches", lambda *args: [[0, 1], [2]])
    agent = ConstantPolicy()
    canonical = optimizer(agent, config)
    before = deepcopy(agent.state_dict())
    with pytest.raises(ValueError, match="all tasks"):
        update_ppo(agent, canonical, episodes, config)
    assert not canonical.state
    for name, value in agent.state_dict().items():
        assert torch.equal(before[name], value)


def test_rollout_updates_report_actual_coefficients_and_unweighted_update_means(monkeypatch):
    import carryrl.training.sequence as sequence

    config, _, _ = setup_case()
    config = replace(config, n_epochs=1, ppo_normalization="rollout", target_kl=0,
                     ppo_minibatch_episodes=2)
    episodes = [episode("a", [1.]), episode("a", [2.] * 3),
                episode("b", [3.] * 2), episode("b", [4.] * 6)]
    monkeypatch.setattr(sequence, "_episode_minibatches", lambda *args: [[0, 2], [1, 3]])
    agent = ConstantPolicy()
    result = update_ppo(agent, optimizer(agent, config), episodes, config)
    assert result["updates"][0]["task_loss_weights"] == {"a": .25, "b": .25}
    assert result["updates"][1]["task_loss_weights"] == {"a": .75, "b": .75}
    assert result["optimizer_task_loss_weights"] == {"a": .5, "b": .5}
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "gradient_norm"):
        expected = sum(row[name] for row in result["updates"]) / 2
        assert result["epochs"][0]["mean_update_" + name] == pytest.approx(expected)
    assert result["epochs"][0]["value_loss"] != pytest.approx(
        result["epochs"][0]["mean_update_value_loss"])


def test_rejected_kl_has_no_applied_diagnostics_or_optimizer_mutation():
    config, _, _ = setup_case()
    config = replace(config, ppo_normalization="rollout", target_kl=.001, gradient_probe_every=1)
    episodes = [episode("a", [1.]), episode("b", [2.])]
    episodes[1].old_log_prob -= 2
    agent = ConstantPolicy()
    canonical = optimizer(agent, config)
    before = deepcopy(agent.state_dict())
    result = update_ppo(agent, canonical, episodes, config)
    assert result["kl_early_stop"] and not result["updates"] and not result["gradient_probes"]
    assert result["epochs"][0]["mean_update_gradient_norm"] is None
    assert result["first_minibatch_ratio"]["max"] == pytest.approx(np.exp(2))
    assert not canonical.state
    for name, value in agent.state_dict().items():
        assert torch.equal(before[name], value)


def test_torch_probe_preserves_existing_gradients_weights_and_all_rng_states():
    config, agent, episodes = setup_case()
    _, _, advantages, returns = _ppo_targets(episodes, config)
    _, weights, _ = objective_weights(episodes, [0, 1], config)
    for parameter in agent.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.ones_like(parameter) * .314
    before = {name: (parameter.clone(), None if parameter.grad is None else parameter.grad.clone())
              for name, parameter in agent.named_parameters()}
    torch_rng, python_rng = torch.get_rng_state(), random.getstate()
    numpy_rng = np.random.get_state()
    actual = torch_component_probe(agent, episodes, [0, 1], advantages, returns, weights, config)
    assert all(value > 0 for value in actual.values())
    assert torch.equal(torch.get_rng_state(), torch_rng) and random.getstate() == python_rng
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    for name, parameter in agent.named_parameters():
        assert torch.equal(parameter, before[name][0])
        if before[name][1] is not None:
            assert torch.equal(parameter.grad, before[name][1])


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_enabled_probe_preserves_update_moments_rng_and_reports_post_update_kl(backend):
    if backend == "mlx":
        mx = pytest.importorskip("mlx.core")
        if not mx.metal.is_available():
            pytest.skip("Metal unavailable")
    config, initial, episodes = setup_case()
    config = replace(config, n_epochs=1, training_backend=backend, ppo_normalization="rollout",
                     ppo_microbatch_episodes=1 if backend == "mlx" else 0)
    agents = [deepcopy(initial), deepcopy(initial)]
    optimizers = [optimizer(agent, config) for agent in agents]
    random_state, torch_state = random.getstate(), torch.get_rng_state()
    baseline = update_ppo(agents[0], optimizers[0], episodes, config)
    after_python, after_torch = random.getstate(), torch.get_rng_state()
    metal_rng = [np.array(value) for value in mx.random.state] if backend == "mlx" else None
    random.setstate(random_state)
    torch.set_rng_state(torch_state)
    result = update_ppo(agents[1], optimizers[1], episodes,
                        replace(config, gradient_probe_every=5), cycle=1)
    assert_agent_close(*agents)
    assert_optimizer_close(*optimizers)
    assert random.getstate() == after_python
    assert torch.equal(torch.get_rng_state(), after_torch)
    if metal_rng is not None:
        for actual, expected in zip(mx.random.state, metal_rng, strict=True):
            np.testing.assert_array_equal(np.array(actual), expected)
    for name, parameter in agents[0].state_dict().items():
        assert torch.equal(parameter, agents[1].state_dict()[name])
    assert len(result["gradient_probes"]) == 1
    probe = result["gradient_probes"][0]
    assert probe["pre_update_kl"] == pytest.approx(baseline["updates"][0]["approx_kl"])
    assert probe["post_update_kl"] >= 0
    assert probe["component_gradient_norms"]["combined"] == pytest.approx(
        probe["gradient_norm"], rel=2e-5, abs=1e-6)
    result = update_ppo(agents[1], optimizers[1], episodes,
                        replace(config, gradient_probe_every=5), cycle=2)
    assert result["gradient_probes"] == []
    result = update_ppo(agents[1], optimizers[1], episodes,
                        replace(config, gradient_probe_every=5), cycle=5)
    assert len(result["gradient_probes"]) == 1


def test_probe_component_norms_and_post_update_kl_match_torch_mlx():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal unavailable")
    config, initial, episodes = setup_case()
    config = replace(config, n_epochs=1, ppo_normalization="rollout", gradient_probe_every=1)
    results = []
    for backend in ("torch", "mlx"):
        agent = deepcopy(initial)
        chosen = replace(config, training_backend=backend,
                         ppo_microbatch_episodes=1 if backend == "mlx" else 0)
        results.append(update_ppo(agent, optimizer(agent, chosen), episodes, chosen))
    expected, actual = [result["gradient_probes"][0] for result in results]
    assert actual["component_gradient_norms"] == pytest.approx(
        expected["component_gradient_norms"], rel=3e-4, abs=6e-6)
    assert actual["post_update_kl"] == pytest.approx(expected["post_update_kl"], abs=2e-6, rel=.01)
