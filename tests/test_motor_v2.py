"""Versioned general motor prior, exact action likelihoods and executed-input replay."""

import io
import itertools
import math

import pytest
import torch
from torch import nn

from carryrl.agent.actions import KEY_VOCAB, MOUSE_BINS, MOUSE_BUTTONS, WHEEL_BINS, PhysicalAction
from carryrl.agent.decoder import AutoregressiveActionDecoder
from carryrl.agent.model import AgentConfig, GeneralistAgent


def _agent(**options):
    torch.manual_seed(41)
    config = AgentConfig(pretrained=False, width=16, heads=4, layers=1, spatial_tokens=2,
                         memory_tokens=2, segment_length=2, **options)
    return GeneralistAgent(config, vision=nn.Identity())


@pytest.mark.parametrize("deterministic", [False, True])
def test_v2_sample_replay_ratio_entropy_and_gradients(deterministic):
    torch.manual_seed(13)
    decoder = AutoregressiveActionDecoder(16, motor_version="motor_v2")
    features = torch.randn(64, 16, requires_grad=True)
    previous = [PhysicalAction(keys=("w", "space"))] * len(features)
    actions, logp, entropy = decoder.sample(features, deterministic, previous_actions=previous)
    replay, replay_entropy = decoder.evaluate(features, actions, previous_actions=previous)
    torch.testing.assert_close((replay - logp).exp(), torch.ones_like(logp))
    torch.testing.assert_close(replay_entropy, entropy)
    assert torch.isfinite(entropy).all()
    (-(replay.mean() + .01 * replay_entropy.mean())).backward()
    assert torch.isfinite(features.grad).all()
    for parameter in decoder.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert decoder.motion_head.weight.grad.abs().sum() > 0
    assert decoder.mouse_head.weight.grad.abs().sum() > 0


def test_v2_unique_mouse_distribution_normalizes_and_has_full_physical_support():
    decoder = AutoregressiveActionDecoder(8, motor_version="motor_v2")
    buttons = [tuple(button for bit, button in enumerate(MOUSE_BUTTONS) if mask & (1 << bit))
               for mask in range(8)]
    actions = [PhysicalAction(dx=dx, dy=dy, mouse_buttons=held, wheel=wheel)
               for dx, dy, held, wheel in itertools.product(
                   MOUSE_BINS, MOUSE_BINS, buttons, WHEEL_BINS)]
    with torch.no_grad():
        logp, _, trace = decoder.evaluate_with_trace(torch.zeros(len(actions), 8), actions)
    # The common empty key prefix is factored out; enumerate every physical mouse action.
    probabilities = (logp - trace[0][0][:, len(KEY_VOCAB)]).exp().double()
    assert torch.isfinite(logp).all() and (probabilities > 0).all()
    assert probabilities.sum().item() == pytest.approx(1, abs=1e-6)
    for gate, expected in enumerate((.7, .1, .1, .1)):
        selected = torch.tensor([int(action.dx != 0) + 2 * int(action.dy != 0) == gate
                                 for action in actions])
        assert probabilities[selected].sum().item() == pytest.approx(expected, abs=1e-6)
    extreme = PhysicalAction(keys=KEY_VOCAB, dx=-64, dy=64,
                             mouse_buttons=MOUSE_BUTTONS, wheel=1)
    assert torch.isfinite(decoder.evaluate(torch.zeros(1, 8), [extreme])[0]).all()


def test_v2_soft_movement_persistence_and_prefix_bonus_preserve_release_support():
    decoder = AutoregressiveActionDecoder(8, motor_version="motor_v2")
    state = decoder._features(torch.zeros(1, 8))
    last, active = torch.tensor([-1]), torch.tensor([True])
    baseline = decoder._key_distribution(state, last, active, [PhysicalAction()])
    previous = PhysicalAction(keys=("w", "space", "e", "shift"), mouse_buttons=("left",))
    held = decoder._key_distribution(state, last, active, [previous])
    eos, w = len(KEY_VOCAB), KEY_VOCAB.index("w")
    assert (held.logits[0, w] - held.logits[0, eos]
            - baseline.logits[0, w] + baseline.logits[0, eos]).item() == pytest.approx(1.5)
    for key in ("space", "e", "shift"):
        index = KEY_VOCAB.index(key)
        assert (held.logits[0, index] - held.logits[0, eos]).item() == pytest.approx(
            (baseline.logits[0, index] - baseline.logits[0, eos]).item())
    after = decoder._key_distribution(state, torch.tensor([0]), active, [PhysicalAction()])
    assert (after.logits[0, eos] - after.logits[0, w]
            - baseline.logits[0, eos] + baseline.logits[0, w]).item() == pytest.approx(.5)
    assert held.probs[0, eos] > 0  # release remains available even while movement was held


def test_v2_buttons_are_conditioned_on_motion_before_wheel():
    decoder = AutoregressiveActionDecoder(8, motor_version="motor_v2")
    seen = []
    handle = decoder.mouse_head.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone()),
    )
    decoder.evaluate(torch.zeros(2, 8), [PhysicalAction(dx=-64), PhysicalAction(dx=64)])
    handle.remove()
    assert not torch.allclose(seen[0][0], seen[0][1])


def test_v2_executed_history_replay_reset_and_checkpoint_round_trip():
    agent = _agent(motor_version="motor_v2", critic_initialization="zero")
    patches = torch.randn(5, 4, 384)
    labels, executed, collected = [], [], []
    for index, frame in enumerate(patches):
        decision = agent.act_from_patches(frame)
        labels.append(decision.action)
        collected.append(decision.log_prob)
        applied = PhysicalAction(keys=("w",)) if index % 2 == 0 else PhysicalAction()
        executed.append(applied)
        agent.observe_action(applied)
    replay = agent.evaluate_episode(patches, labels, executed_actions=executed)
    torch.testing.assert_close(replay["log_prob"], torch.tensor(collected), atol=2e-5, rtol=2e-5)
    assert torch.equal(replay["value"], torch.zeros(5))
    agent.reset_episode()
    first = agent.act_from_patches(patches[0], deterministic=True)
    agent.observe_action(PhysicalAction(keys=("w", "space")))
    agent.act_from_patches(patches[1], deterministic=True)
    agent.reset_episode()
    assert agent.act_from_patches(patches[0], deterministic=True) == first
    # Loading trained critic tensors never reapplies the fresh initialization option.
    with torch.no_grad():
        agent.value_head.weight.fill_(.12)
        agent.value_head.bias.fill_(.7)
    saved = io.BytesIO()
    torch.save({"config": agent.config.to_dict(), "state": agent.state_dict()}, saved)
    saved.seek(0)
    record = torch.load(saved, weights_only=True)
    restored = GeneralistAgent(AgentConfig.from_dict(record["config"]), vision=nn.Identity())
    restored.load_state_dict(record["state"])
    actual = restored.evaluate_episode(patches, labels, executed_actions=executed)
    expected = agent.evaluate_episode(patches, labels, executed_actions=executed)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert restored.value_head.bias.item() == pytest.approx(.7)


def test_legacy_config_and_initialization_stay_compatible():
    old = _agent()
    config = old.config.to_dict()
    config.pop("motor_version")
    config.pop("critic_initialization")
    restored_config = AgentConfig.from_dict(config)
    assert restored_config.motor_version == "legacy"
    assert restored_config.critic_initialization == "legacy"
    restored = GeneralistAgent(restored_config, vision=nn.Identity())
    restored.load_state_dict(old.state_dict())
    assert not any("motion_" in name for name in old.state_dict())
    patches = torch.randn(1, 4, 384)
    actions = [PhysicalAction(dx=-64, mouse_buttons=("left",))]
    for name, value in old.evaluate_episode(patches, actions).items():
        torch.testing.assert_close(restored.evaluate_episode(patches, actions)[name], value,
                                   rtol=0, atol=0)
    assert old.value_head.weight.abs().sum() > 0
    assert math.isclose(old.decoder.key_head.bias[-1].item(), 3.)


@pytest.mark.parametrize("options", [{"motor_version": "v3"}, {"critic_initialization": "yes"}])
def test_versioned_config_rejects_unknown_options(options):
    with pytest.raises(ValueError):
        AgentConfig(**options)
