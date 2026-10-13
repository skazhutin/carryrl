"""Causal collection/replay and context contracts for the shared agent."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.model import AgentConfig, GeneralistAgent


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 384)

    def forward(self, frames):
        rgb = frames.permute(0, 3, 1, 2).float() / 255
        pixels = F.adaptive_avg_pool2d(rgb, (4, 4)).flatten(2).transpose(1, 2)
        return self.projection(pixels)


def make_agent():
    torch.manual_seed(17)
    return GeneralistAgent(AgentConfig(pretrained=False, image_size=28, width=32,
                                      heads=4, layers=2, spatial_tokens=4,
                                      memory_tokens=2, segment_length=4), vision=TinyVision())


def test_collection_replay_matches_across_two_memory_boundaries():
    agent = make_agent()
    patches = torch.randn(9, 16, 384)
    agent.reset_episode()
    decisions = [agent.act_from_patches(frame) for frame in patches]
    replay = agent.evaluate_episode(patches, [decision.action for decision in decisions])
    torch.testing.assert_close(replay["log_prob"],
                               torch.tensor([decision.log_prob for decision in decisions]),
                               atol=3e-5, rtol=2e-5)
    torch.testing.assert_close(replay["value"],
                               torch.tensor([decision.value for decision in decisions]),
                               atol=3e-5, rtol=2e-5)


def test_future_pixels_and_actions_do_not_change_earlier_likelihoods():
    agent = make_agent()
    patches = torch.randn(9, 16, 384)
    actions = [PhysicalAction(keys=("w",))] * 9
    original = agent.evaluate_episode(patches, actions)
    altered = patches.clone()
    altered[5:] += 20
    changed_actions = actions[:5] + [PhysicalAction(mouse_buttons=("left",))] * 4
    result = agent.evaluate_episode(altered, changed_actions)
    for name in ("log_prob", "value"):
        torch.testing.assert_close(result[name][:5], original[name][:5])


def test_bootstrap_peek_does_not_commit_history_and_reset_removes_prior_episode():
    agent = make_agent()
    patches = torch.randn(6, 16, 384)
    agent.reset_episode()
    first = agent.act_from_patches(patches[0], deterministic=True)
    for frame in patches[1:5]:
        agent.act_from_patches(frame, deterministic=True)
    peek = agent.peek_value_from_patches(patches[5])
    assert agent.peek_value_from_patches(patches[5]) == peek
    assert agent.act_from_patches(patches[5], deterministic=True).value == peek
    agent.reset_episode()
    repeat = agent.act_from_patches(patches[0], deterministic=True)
    assert repeat == first


def test_updates_change_policy_but_preserve_frozen_backbone():
    agent = make_agent().train()
    assert not agent.vision.training
    assert all(not parameter.requires_grad for parameter in agent.vision.parameters())
    before = {key: value.clone() for key, value in agent.vision.state_dict().items()}
    original = agent.value_head.weight.detach().clone()
    patches = torch.randn(9, 16, 384)
    result = agent.evaluate_episode(patches, [PhysicalAction()] * 9)
    optimizer = torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad), lr=1e-3)
    optimizer.zero_grad()
    (result["value"].square().mean() - result["log_prob"].mean()).backward()
    optimizer.step()
    assert not torch.equal(original, agent.value_head.weight)
    for key, value in agent.vision.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_uint8_observation_contract_and_physical_action():
    agent = make_agent()
    frame = np.zeros((24, 36, 3), dtype=np.uint8)
    decision = agent.act(frame)
    assert isinstance(decision.action, PhysicalAction)
    with pytest.raises(ValueError, match="uint8"):
        agent.encode_frames(torch.zeros(1, 24, 36, 3))


def test_valid_nondefault_attention_heads_apply_to_the_whole_policy():
    agent = GeneralistAgent(AgentConfig(pretrained=False, image_size=28, width=30,
                                      heads=3, layers=1, spatial_tokens=4,
                                      memory_tokens=2, segment_length=4), vision=TinyVision())
    result = agent.evaluate_episode(torch.randn(5, 16, 384), [PhysicalAction()] * 5)
    assert all(torch.isfinite(value).all() for value in result.values())


@pytest.mark.parametrize("data", [
    {"pretrained": 1}, {"image_size": 64}, {"layers": 0},
    {"width": 33, "heads": 4}, {"unexpected": True},
])
def test_invalid_architecture_configuration_is_rejected(data):
    with pytest.raises(ValueError):
        AgentConfig.from_dict(data)
