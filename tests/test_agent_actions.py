"""Codec boundaries, causal teacher forcing and joint physical action likelihoods."""

import json
import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from carryrl.agent.actions import (
    ACTION_FEATURE_SIZE,
    CODEC_VERSION,
    KEY_VOCAB,
    MOUSE_BINS,
    MOUSE_BUTTONS,
    WHEEL_BINS,
    PhysicalAction,
    encode_actions,
)
from carryrl.agent.decoder import AutoregressiveActionDecoder


def test_action_canonical_round_trip_and_frozen_values():
    action = PhysicalAction(keys=("e", "up", "space"), mouse_buttons=("middle", "left"),
                            dx=-64, dy=1, wheel=1)
    assert action.keys == ("up", "space", "e")
    assert action.mouse_buttons == ("left", "middle")
    assert action.serialize()["codec_version"] == CODEC_VERSION
    assert PhysicalAction.from_dict(json.loads(json.dumps(action.to_dict()))) == action
    with pytest.raises(FrozenInstanceError):
        action.dx = 0


@pytest.mark.parametrize("kwargs", [
    {"keys": ("w", "w")}, {"keys": ("W",)}, {"keys": ["w"]}, {"keys": (1,)},
    {"mouse_buttons": ("left", "left")}, {"mouse_buttons": ("unknown",)},
    {"dx": True}, {"dx": 1.0}, {"dx": float("nan")}, {"dx": float("inf")},
    {"dy": -3}, {"dy": 65}, {"wheel": 2}, {"wheel": False},
])
def test_action_rejects_invalid_or_unrepresentable_inputs(kwargs):
    with pytest.raises((ValueError, TypeError)):
        PhysicalAction(**kwargs)


@pytest.mark.parametrize("changes", [
    {"codec_version": "future"}, {"unknown": 1}, {"keys": "up"},
    {"mouse_buttons": ()}, {"dx": False},
])
def test_serialized_action_validates_version_shape_and_fields(changes):
    with pytest.raises((ValueError, TypeError)):
        PhysicalAction.from_dict({**PhysicalAction().serialize(), **changes})
    with pytest.raises(ValueError):
        PhysicalAction.from_dict({"keys": []})


def test_encoding_preserves_simultaneous_holds_and_relative_motion():
    actions = [PhysicalAction(), PhysicalAction(keys=KEY_VOCAB, mouse_buttons=MOUSE_BUTTONS,
                                               dx=-64, dy=64, wheel=-1)]
    encoded = encode_actions(actions)
    assert encoded.shape == (2, ACTION_FEATURE_SIZE)
    assert encoded.dtype == torch.float32 and encoded.device.type == "cpu"
    assert torch.equal(encoded[0], torch.zeros(ACTION_FEATURE_SIZE))
    assert torch.equal(encoded[1, :-3], torch.ones(ACTION_FEATURE_SIZE - 3))
    assert encoded[1, -3:].tolist() == [-1.0, 1.0, -1.0]
    assert encode_actions([]).shape == (0, ACTION_FEATURE_SIZE)
    with pytest.raises(TypeError):
        encode_actions([{}])


@pytest.mark.parametrize("deterministic", [False, True])
def test_sample_and_teacher_forced_joint_likelihood_match(deterministic):
    torch.manual_seed(14)
    decoder = AutoregressiveActionDecoder(width=16)
    features = torch.randn(32, 16)
    actions, sampled_logp, sampled_entropy = decoder.sample(features, deterministic)
    evaluated_logp, evaluated_entropy = decoder.evaluate(features, actions)
    assert torch.allclose(sampled_logp, evaluated_logp, atol=1e-6)
    assert torch.allclose(sampled_entropy, evaluated_entropy, atol=1e-6)
    assert torch.isfinite(sampled_logp).all() and torch.isfinite(sampled_entropy).all()
    for action in actions:
        assert len(action.keys) == len(set(action.keys))
        assert PhysicalAction.from_dict(action.to_dict()) == action


def test_joint_log_prob_includes_eos_and_every_component_for_empty_and_full_sets():
    decoder = AutoregressiveActionDecoder(width=8)
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.zero_()
    actions = [PhysicalAction(), PhysicalAction(keys=KEY_VOCAB, mouse_buttons=MOUSE_BUTTONS,
                                               dx=MOUSE_BINS[0], dy=MOUSE_BINS[-1],
                                               wheel=WHEEL_BINS[-1])]
    logp, entropy = decoder.evaluate(torch.zeros(2, 8), actions)
    mouse_logp = -math.log(8) - 2 * math.log(len(MOUSE_BINS)) - math.log(len(WHEEL_BINS))
    empty_expected = -math.log(len(KEY_VOCAB) + 1) + mouse_logp
    full_expected = -math.lgamma(len(KEY_VOCAB) + 2) + mouse_logp
    assert logp[0].item() == pytest.approx(empty_expected, abs=2e-6)
    assert logp[1].item() == pytest.approx(full_expected, abs=2e-5)
    assert torch.allclose(entropy, -logp, atol=2e-5)


def test_eos_sparse_prior_does_not_prevent_any_key_set():
    decoder = AutoregressiveActionDecoder(width=8)
    with torch.no_grad():
        decoder.key_head.weight.zero_()
    actions, _, _ = decoder.sample(torch.zeros(1, 8), deterministic=True)
    assert actions[0].keys == ()
    logp, _ = decoder.evaluate(torch.zeros(1, 8), [PhysicalAction(keys=KEY_VOCAB)])
    assert torch.isfinite(logp).all()


def test_teacher_forcing_mouse_choices_changes_later_conditioning_and_has_gradients():
    torch.manual_seed(7)
    decoder = AutoregressiveActionDecoder(width=16)
    features = torch.randn(1, 16).repeat(2, 1).requires_grad_()
    seen = []
    handle = decoder.wheel_head.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone()),
    )
    actions = [PhysicalAction(dx=-64), PhysicalAction(dx=64)]
    logp, entropy = decoder.evaluate(features, actions)
    handle.remove()
    assert not torch.allclose(seen[0][0], seen[0][1])
    loss = -(logp + 0.01 * entropy).mean()
    loss.backward()
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
    for parameter in decoder.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert decoder.dx_embedding.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("features", [torch.zeros(2, 7), torch.zeros(0, 8),
                                         torch.zeros(1, 8, dtype=torch.int64),
                                         torch.full((1, 8), float("nan"))])
def test_decoder_rejects_invalid_features(features):
    with pytest.raises((ValueError, TypeError, FloatingPointError)):
        AutoregressiveActionDecoder(width=8).sample(features)


def test_teacher_forcing_rejects_missing_or_nonphysical_action_rows():
    decoder = AutoregressiveActionDecoder(width=8)
    with pytest.raises(ValueError):
        decoder.evaluate(torch.zeros(2, 8), [PhysicalAction()])
    with pytest.raises(TypeError):
        decoder.evaluate(torch.zeros(1, 8), [{}])


def test_balanced_motor_prior_explores_inputs_without_uniform_clicks_or_losing_support():
    torch.set_num_threads(1)
    torch.manual_seed(103)
    decoder = AutoregressiveActionDecoder(width=16)
    original_keys = set(decoder.state_dict())
    decoder.initialize_motor_prior("balanced")
    assert set(decoder.state_dict()) == original_keys
    features = torch.zeros(1024, 16)
    actions, logp, entropy = decoder.sample(features)
    replay_logp, replay_entropy = decoder.evaluate(features, actions)
    assert torch.allclose(logp, replay_logp, atol=2e-6)
    assert torch.allclose(entropy, replay_entropy, atol=2e-6)
    assert sum(bool(action.keys) for action in actions) > 800
    assert sum(not action.mouse_buttons for action in actions) > 850
    assert sum(abs(action.dx) <= 4 and abs(action.dy) <= 4 for action in actions) > 750
    greedy, _, _ = decoder.sample(features[:16], deterministic=True)
    assert all(len(action.keys) == 1 for action in greedy)
    extreme = PhysicalAction(keys=KEY_VOCAB, mouse_buttons=MOUSE_BUTTONS,
                             dx=-64, dy=64, wheel=1)
    assert torch.isfinite(decoder.evaluate(features[:1], [extreme])[0]).all()
    decoder.initialize_motor_prior("legacy")
    assert decoder.motor_initialization == "legacy"
    assert decoder.key_head.bias[-1].item() == 3.0
    with pytest.raises(ValueError, match="motor initialization"):
        decoder.initialize_motor_prior("game-specific")
