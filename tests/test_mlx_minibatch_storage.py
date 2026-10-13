"""Batch-only frozen feature transfer preserves eager Metal PPO semantics."""

import random
import weakref
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from test_mlx_training import optimizer, setup_case

import carryrl.training.mlx as training
from carryrl.training.sequence import _ppo_targets


@pytest.fixture
def mx():
    core = pytest.importorskip("mlx.core")
    if not core.metal.is_available():
        pytest.skip("MLX Metal unavailable")
    core.set_default_device(core.gpu)
    torch.set_num_threads(1)
    return core


class TrackedBatch(list):
    pass


@pytest.mark.parametrize("target_kl", [0.0, .00001])
def test_streamed_batches_preserve_eager_ppo_and_float32_moments(mx, monkeypatch, target_kl):
    cfg, initial, base = setup_case()
    cfg = replace(cfg, training_backend="mlx", ppo_minibatch_episodes=4, target_kl=target_kl)
    episodes = [deepcopy(base[index % 2]) for index in range(8)]
    for index, episode in enumerate(episodes):
        episode.seed = index
    _, _, advantages, returns = _ppo_targets(episodes, cfg)
    real_prepare = training.prepare_episodes
    eager = real_prepare(initial, episodes, advantages, returns)
    eager_by_id = {id(episode): value for episode, value in zip(episodes, eager, strict=True)}
    reference, streamed, repeated = (deepcopy(initial) for _ in range(3))
    left, right = optimizer(reference, cfg), optimizer(streamed, cfg)
    control_optimizer = optimizer(repeated, cfg)
    traces = [[], []]

    def eager_prepare(agent, batch, *args, **kwargs):
        traces[0].append([episode.seed for episode in batch])
        return [eager_by_id[id(episode)] for episode in batch]

    monkeypatch.setattr(training, "prepare_episodes", eager_prepare)
    random.seed(191)
    expected = training.update_ppo_mlx(reference, left, episodes, cfg)
    reference_rng = random.getstate()
    random.seed(191)
    control = training.update_ppo_mlx(repeated, control_optimizer, episodes, cfg)
    assert control == expected and random.getstate() == reference_rng
    traces[0] = traces[0][:len(traces[0]) // 2]
    prior_batches = []

    def streamed_prepare(agent, batch, *args, **kwargs):
        assert all(reference() is None for reference in prior_batches)
        assert len(batch) <= 4
        traces[1].append([episode.seed for episode in batch])
        prepared = TrackedBatch(real_prepare(agent, batch, *args, **kwargs))
        prior_batches.append(weakref.ref(prepared))
        return prepared

    monkeypatch.setattr(training, "prepare_episodes", streamed_prepare)
    random.seed(191)
    actual = training.update_ppo_mlx(streamed, right, episodes, cfg)
    assert random.getstate() == reference_rng
    assert traces[0] == traces[1] and actual == expected
    assert all(reference() is None for reference in prior_batches)
    assert actual["optimizer_steps"] == (4 if not target_kl else 0)
    for name, parameter in reference.named_parameters():
        assert torch.equal(parameter, dict(streamed.named_parameters())[name])
    max_abs = max_rel = control_abs = control_rel = 0.0
    for parameter, state in left.state_dict()["state"].items():
        for name, value in state.items():
            other = right.state_dict()["state"][parameter][name]
            repeated_value = control_optimizer.state_dict()["state"][parameter][name]
            if name == "step":
                assert torch.equal(value, other) and torch.equal(value, repeated_value)
                continue
            for candidate, is_control in ((other, False), (repeated_value, True)):
                delta = (value - candidate).abs()
                scale = torch.maximum(value.abs(), candidate.abs())
                absolute = float(delta.max())
                relative = float((delta[scale > 0] / scale[scale > 0]).max()) if scale.any() else 0.
                if is_control:
                    control_abs = max(control_abs, absolute)
                    control_rel = max(control_rel, relative)
                else:
                    max_abs, max_rel = max(max_abs, absolute), max(max_rel, relative)
                # One float32 rounding bit may depend on GPU buffer placement.
                torch.testing.assert_close(value, candidate, atol=1e-14, rtol=3e-7)
    print(f"moments streaming abs={max_abs:.3g} rel={max_rel:.3g}; "
          f"eager-repeat abs={control_abs:.3g} rel={control_rel:.3g}")
    assert left.state_dict()["param_groups"] == right.state_dict()["param_groups"]


def test_64_episode_corpus_converts_only_first_four_when_kl_rejects(mx, monkeypatch):
    cfg, agent, base = setup_case()
    cfg = replace(cfg, training_backend="mlx", ppo_minibatch_episodes=4, target_kl=.00001)
    episodes = [deepcopy(base[index % 2]) for index in range(64)]
    converted = []
    original = training.prepare_episodes

    def prepare(agent, batch, *args, **kwargs):
        values = original(agent, batch, *args, **kwargs)
        converted.append((len(values), sum(value["patches"].nbytes for value in values)))
        return values

    monkeypatch.setattr(training, "prepare_episodes", prepare)
    result = training.update_ppo_mlx(agent, optimizer(agent, cfg), episodes, cfg)
    assert result["kl_early_stop"] and result["optimizer_steps"] == 0
    assert len(converted) == 1 and converted[0][0] == 4
    assert converted[0][1] <= 4 * 4 * 4 * 384 * 4
    assert converted[0][1] < sum(episode.patches.numel() * episode.patches.element_size()
                                 for episode in episodes)


def test_shared_instruction_cache_encodes_each_unique_prose_once(monkeypatch):
    cfg, agent, episodes = setup_case()
    calls = []

    def encode(instruction):
        calls.append(instruction)
        return torch.full((384,), float(len(instruction)))

    monkeypatch.setattr(agent, "_encode_instruction", encode)
    monkeypatch.setattr(training, "_array", lambda value: value.numpy().copy())
    episodes[0].instruction, episodes[1].instruction = "first", "second"
    _, _, advantages, returns = _ppo_targets(episodes, cfg)
    cache = {}
    training._prepare_instructions(agent, episodes, cache)
    for _ in range(2):
        for index in range(2):
            values = training.prepare_episodes(agent, episodes[index:index + 1],
                advantages[index:index + 1], returns[index:index + 1], instruction_cache=cache)
            assert values[0]["instruction_features"] is cache[episodes[index].instruction]
    assert calls == ["first", "second"]
