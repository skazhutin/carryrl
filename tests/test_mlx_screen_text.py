"""Offline OCR token fusion parity on the actual Metal policy graph."""

import numpy as np
import pytest
import torch
from torch import nn

mx = pytest.importorskip("mlx.core")
mnn = pytest.importorskip("mlx.nn")
if not mx.metal.is_available():
    pytest.skip("MLX OCR parity requires a Metal GPU", allow_module_level=True)

from carryrl.agent.actions import PhysicalAction  # noqa: E402
from carryrl.agent.mlx_policy import MLXPolicy  # noqa: E402
from carryrl.agent.model import AgentConfig, GeneralistAgent  # noqa: E402


class FrozenFeatures(nn.Module):
    def forward(self, text):
        return torch.ones(384)

    def encode_tokens(self, text):
        return torch.arange(len(text.split()) * 384, dtype=torch.float32).reshape(-1, 384) / 384


def _agent(*, text=True):
    torch.manual_seed(29)
    return GeneralistAgent(
        AgentConfig(pretrained=False, width=32, layers=2, heads=4, spatial_tokens=3,
                    memory_tokens=2, segment_length=3,
                    text_encoder="multilingual_e5_small" if text else "none"),
        vision=FrozenFeatures(), language=FrozenFeatures() if text else None,
    )


def _episode():
    return torch.randn(7, 4, 384), [PhysicalAction(keys=("up",))] * 7


def test_screen_tokens_values_and_gradients_cross_two_memory_boundaries():
    agent = _agent()
    patches, actions = _episode()
    strings = ["score next", None, "", "win", "one two three four five", None, "next stage"]
    text = [agent.encode_screen_text(value) for value in strings]
    expected = agent.evaluate_episode(patches, actions, instruction="move", screen_text=strings)
    loss = expected["log_prob"][-1] + expected["value"][-1] + .02 * expected["entropy"][-1]
    loss.backward()
    policy = MLXPolicy.from_torch(agent)
    actual = policy(patches, actions, instruction_features=torch.ones(384),
                    screen_text_features=text)
    mx.eval(actual)
    for name in expected:
        np.testing.assert_allclose(np.array(actual[name]), expected[name].detach().numpy(),
                                   rtol=4e-5, atol=3e-5)

    def objective(model):
        result = model(patches, actions, instruction_features=torch.ones(384),
                       screen_text_features=text)
        return result["log_prob"][-1] + result["value"][-1] + .02 * result["entropy"][-1]

    actual_loss, gradients = mnn.value_and_grad(policy, objective)(policy)
    mx.eval(actual_loss, gradients)
    assert float(actual_loss) == pytest.approx(float(loss.detach()), abs=3e-5)
    parameters = dict(agent.named_parameters())
    for name, gradient in gradients["weights"].items():
        value = np.array(gradient)
        assert np.isfinite(value).all(), name
        np.testing.assert_allclose(value, parameters[name].grad.numpy(),
                                   rtol=3e-3, atol=1e-5, err_msg=name)
    assert np.linalg.norm(np.array(gradients["weights"]["instruction_projection.weight"])) > 0
    assert np.linalg.norm(np.array(gradients["weights"]["temporal.write_tokens"])) > 0


@pytest.mark.parametrize("text_enabled", [False, True])
def test_empty_screen_tokens_preserve_exact_default_outputs_and_inventory(text_enabled):
    agent = _agent(text=text_enabled)
    policy = MLXPolicy.from_torch(agent)
    patches, actions = _episode()
    inventory = set(policy.weights)
    baseline = policy(patches, actions)
    for features in (None, [None] * 7, [torch.empty(0, 384)] * 7):
        actual = policy(patches, actions, screen_text_features=features)
        mx.eval(baseline, actual)
        for name in baseline:
            np.testing.assert_array_equal(np.array(actual[name]), np.array(baseline[name]))
    assert set(policy.weights) == inventory
    assert inventory == {name for name, param in agent.named_parameters() if param.requires_grad}


@pytest.mark.parametrize("name", ["summary_attention", "actor_attention"])
def test_padding_is_excluded_from_both_attention_stages(name):
    policy = MLXPolicy.from_torch(_agent())
    query = mx.random.normal((2, 1, 32))
    tokens = mx.random.normal((2, 5, 32))
    padding = mx.array([[False, False, False, True, True],
                        [False, False, False, False, True]])
    changed = mx.where(padding[..., None], 1e4, tokens)
    expected = policy._attention(query, tokens, tokens, name, padding)
    actual = policy._attention(query, changed, changed, name, padding)
    mx.eval(expected, actual)
    np.testing.assert_array_equal(np.array(expected), np.array(actual))


@pytest.mark.parametrize("features,match", [
    ([None], "complete episode"),
    ([torch.zeros(384)] * 7, "shape"),
    ([torch.zeros(2, 383)] * 7, "shape"),
    ([torch.zeros(1, 2, 384)] * 7, "shape"),
    ([torch.full((1, 384), float("nan"))] * 7, "finite"),
    ([torch.full((1, 384), float("inf"))] * 7, "finite"),
])
def test_invalid_screen_tokens_are_rejected(features, match):
    policy = MLXPolicy.from_torch(_agent())
    patches, actions = _episode()
    with pytest.raises(ValueError, match=match):
        policy(patches, actions, screen_text_features=features)


def test_nonempty_screen_tokens_require_text_projection():
    policy = MLXPolicy.from_torch(_agent(text=False))
    patches, actions = _episode()
    with pytest.raises(ValueError, match="enabled text encoder"):
        policy(patches, actions, screen_text_features=[torch.ones(1, 384)] * 7)
