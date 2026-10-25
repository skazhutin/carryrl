"""Offline numerical parity for the complete MLX policy on a real Metal GPU."""

import numpy as np
import pytest
import torch
from torch import nn

mx = pytest.importorskip("mlx.core")
mnn = pytest.importorskip("mlx.nn")
if not mx.metal.is_available():
    pytest.skip("MLX policy parity requires a Metal GPU", allow_module_level=True)

from carryrl.agent.actions import KEY_VOCAB, PhysicalAction  # noqa: E402
from carryrl.agent.mlx_policy import MLXPolicy, to_mlx  # noqa: E402
from carryrl.agent.model import AgentConfig, GeneralistAgent  # noqa: E402


class FrozenFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Parameter(torch.randn(384))

    def forward(self, instruction):
        return self.feature


def _agent(*, instruction=False, canonical=False, motor_version="legacy",
           critic_initialization="legacy"):
    torch.manual_seed(17)
    config = AgentConfig(
        pretrained=False, motor_version=motor_version, critic_initialization=critic_initialization,
        **({} if canonical else dict(width=32, layers=2, heads=4, spatial_tokens=3,
                                     memory_tokens=2, segment_length=3)),
        text_encoder="multilingual_e5_small" if instruction else "none",
    )
    return GeneralistAgent(config, vision=FrozenFeatures(),
                           language=FrozenFeatures() if instruction else None)


def _actions():
    return [
        PhysicalAction(),
        PhysicalAction(keys=("up", "w", "space"), mouse_buttons=("left", "middle"),
                       dx=-64, dy=32, wheel=-1),
        PhysicalAction(keys=("9",), mouse_buttons=("right",), dx=64, dy=-64, wheel=1),
        PhysicalAction(keys=("shift", "ctrl", "enter")),
        PhysicalAction(keys=KEY_VOCAB, mouse_buttons=("left", "right", "middle")),
        PhysicalAction(keys=("down",), dx=-1, dy=2),
        PhysicalAction(),
    ]


@pytest.mark.parametrize("mode", ["legacy", "balanced"])
@pytest.mark.parametrize("instruction", [False, True])
def test_complete_episode_parity_through_two_memory_boundaries(mode, instruction):
    agent = _agent(instruction=instruction)
    agent.decoder.initialize_motor_prior(mode)
    patches = torch.randn(7, 4, 384)
    actions = _actions()
    feature = agent.language.feature.detach() if instruction else None
    expected = agent.evaluate_episode(patches, actions, instruction="move" if instruction else None)
    policy = MLXPolicy.from_torch(agent)
    actual = policy.evaluate_episode(patches, actions, instruction_features=feature)
    mx.eval(actual)
    for name in expected:
        np.testing.assert_allclose(np.array(actual[name]), expected[name].detach().numpy(),
                                   rtol=3e-5, atol=2e-5)
        assert np.isfinite(np.array(actual[name])).all()
    assert set(policy.trainable_parameters()) == {"weights"}
    assert set(policy.trainable_parameters()["weights"]) == {
        name for name, parameter in agent.named_parameters() if parameter.requires_grad
    }


def test_gradients_include_memory_writes_instruction_and_all_policy_stages():
    agent = _agent(instruction=True)
    agent.decoder.initialize_motor_prior("balanced")
    # Nonzero trained head weights make gradient parity more informative than
    # only comparing the almost uniform default decoder initialization.
    with torch.no_grad():
        for name, parameter in agent.named_parameters():
            if name.startswith("decoder.") and name.endswith("head.weight"):
                parameter.mul_(20)
    patches, actions = torch.randn(7, 4, 384), _actions()
    feature = agent.language.feature.detach()
    expected = agent.evaluate_episode(patches, actions, instruction="move")
    # Only the final decision contributes: early memory and writes must receive
    # gradients through two boundaries rather than their own local losses.
    torch_loss = (expected["log_prob"][-1] + expected["value"][-1]
                  + 0.02 * expected["entropy"][-1])
    torch_loss.backward()
    policy = MLXPolicy.from_torch(agent)

    def loss_fn(model):
        result = model.evaluate_episode(patches, actions, instruction_features=feature)
        return result["log_prob"][-1] + result["value"][-1] + 0.02 * result["entropy"][-1]

    loss, gradients = mnn.value_and_grad(policy, loss_fn)(policy)
    mx.eval(loss, gradients)
    assert float(loss) == pytest.approx(float(torch_loss.detach()), rel=3e-5, abs=2e-5)
    selected = [
        "resampler.patch_projection.weight", "resampler.position_projection.weight",
        "resampler.queries", "resampler.stages.0.attention.in_proj_weight",
        "resampler.stages.1.mlp.0.weight", "summary_attention.in_proj_weight",
        "previous_action_embedding.weight", "time_embedding.weight",
        "temporal.initial_memory", "temporal.write_tokens", "temporal.read_positions",
        "temporal.blocks.0.qkv.weight", "temporal.blocks.1.mlp.2.weight",
        "actor_attention.in_proj_weight", "instruction_projection.weight",
        "decoder.context.weight", "decoder.state_cell.weight_ih", "decoder.state_cell.weight_hh",
        "decoder.state_cell.bias_ih", "decoder.state_cell.bias_hh",
        "decoder.key_embedding.weight", "decoder.mouse_embedding.weight",
        "decoder.dx_embedding.weight", "decoder.dy_embedding.weight",
        "decoder.key_head.weight", "decoder.mouse_head.weight", "decoder.dx_head.weight",
        "decoder.dy_head.weight", "decoder.wheel_head.weight", "value_head.weight",
    ]
    parameters = dict(agent.named_parameters())
    assert set(gradients["weights"]) == set(policy.weights)
    for name in selected:
        actual, expected_gradient = np.array(gradients["weights"][name]), parameters[name].grad
        assert np.isfinite(actual).all(), name
        assert np.linalg.norm(actual) > 0, name
        np.testing.assert_allclose(actual, expected_gradient.numpy(), rtol=2e-3, atol=8e-6,
                                   err_msg=name)


def test_export_exactly_preserves_parameter_names_shapes_and_frozen_encoders():
    agent = _agent(instruction=True)
    original = {name: value.clone() for name, value in agent.state_dict().items()}
    policy = MLXPolicy.from_torch(agent)
    policy.export_to_torch(agent)
    for name, value in agent.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    name = "temporal.blocks.1.qkv.weight"
    policy.weights[name] = policy.weights[name] + 0.125
    policy.motor_initialization = "balanced"
    policy.export_to_torch(agent)
    torch.testing.assert_close(dict(agent.named_parameters())[name],
                               torch.from_numpy(np.array(policy.weights[name])), rtol=0, atol=0)
    assert agent.decoder.motor_initialization == "balanced"
    torch.testing.assert_close(agent.vision.feature, original["vision.feature"], rtol=0, atol=0)
    torch.testing.assert_close(agent.language.feature, original["language.feature"], rtol=0, atol=0)
    assert agent._episode_state is None


def test_canonical_graph_is_not_replaced_by_a_smaller_policy():
    agent = _agent(canonical=True, instruction=True)
    policy = MLXPolicy.from_torch(agent)
    count = sum(parameter.numel() for parameter in agent.parameters() if parameter.requires_grad)
    assert count == 5_996_870
    assert sum(array.size for array in policy.weights.values()) == count
    assert policy.config.width == 256
    assert policy.config.layers == 4
    assert policy.weights["temporal.decision_positions"].shape == (64, 256)
    assert policy.weights["temporal.initial_memory"].shape == (8, 256)
    patches = torch.randn(2, 4, 384)
    actions = _actions()[:2]
    expected = agent.evaluate_episode(patches, actions, instruction="move")
    actual = policy.evaluate_episode(patches, actions, instruction_features=agent.language.feature)
    mx.eval(actual)
    for name in expected:
        np.testing.assert_allclose(np.array(actual[name]), expected[name].detach().numpy(),
                                   rtol=5e-5, atol=2e-5)


def test_parameter_inventory_rejects_extra_missing_or_reshaped_trainable_weights():
    agent = _agent()
    policy = MLXPolicy.from_torch(agent)
    invalid = dict(policy.weights)
    invalid.pop("summary_query")
    with pytest.raises(ValueError, match="coverage mismatch"):
        MLXPolicy(agent.config, invalid)
    invalid = dict(policy.weights)
    invalid["summary_query"] = mx.zeros((1, 1, 33))
    with pytest.raises(ValueError, match="shape"):
        MLXPolicy(agent.config, invalid)
    agent.register_parameter("unexpected", nn.Parameter(torch.zeros(1)))
    with pytest.raises(ValueError, match="coverage mismatch"):
        MLXPolicy.from_torch(agent)


def test_validation_and_explicit_cpu_feature_bridge():
    policy = MLXPolicy.from_torch(_agent())
    np.testing.assert_array_equal(np.array(to_mlx(torch.ones(2, dtype=torch.float64))),
                                  np.ones(2, dtype=np.float32))
    with pytest.raises(ValueError, match="nonempty"):
        policy.evaluate_episode(torch.randn(0, 4, 384), [])
    with pytest.raises(ValueError, match="square"):
        policy.evaluate_episode(torch.randn(1, 3, 384), [PhysicalAction()])
    with pytest.raises(ValueError, match="enabled text"):
        policy.evaluate_episode(torch.randn(1, 4, 384), [PhysicalAction()],
                                instruction_features=torch.randn(384))
    previous_device = mx.default_device()
    try:
        mx.set_default_device(mx.cpu)
        with pytest.raises(RuntimeError, match="device to be GPU"):
            MLXPolicy.from_torch(_agent())
    finally:
        mx.set_default_device(previous_device)


def test_executed_input_conditioning_parity_preserves_expert_targets():
    agent = _agent(instruction=True)
    patches = torch.randn(7, 4, 384)
    labels = _actions()
    executed = list(reversed(labels))
    expected = agent.evaluate_episode(patches, labels, instruction="move",
                                      executed_actions=executed)
    policy = MLXPolicy.from_torch(agent)
    actual = policy(patches, labels, instruction_features=agent.language.feature,
                    executed_actions=executed)
    mx.eval(actual)
    for name in expected:
        np.testing.assert_allclose(np.array(actual[name]), expected[name].detach().numpy(),
                                   rtol=3e-5, atol=2e-5)
    with pytest.raises(ValueError, match="executed actions"):
        policy.evaluate_episode(patches, labels, executed_actions=executed[:-1])


def test_motor_v2_complete_replay_trace_and_gradient_parity_with_executed_history():
    agent = _agent(motor_version="motor_v2", critic_initialization="zero")
    patches, actions = torch.randn(7, 4, 384), _actions()
    executed = [PhysicalAction(keys=("w", "space")), PhysicalAction(),
                PhysicalAction(keys=("a", "shift")), PhysicalAction(),
                PhysicalAction(keys=("right",)), PhysicalAction(), PhysicalAction()]
    expected = agent.evaluate_episode(patches, actions, executed_actions=executed,
                                      return_decoder_trace=True)
    policy = MLXPolicy.from_torch(agent)
    actual = policy.evaluate_episode(patches, actions, executed_actions=executed,
                                     return_decoder_trace=True)
    mx.eval(actual)
    for name in ("log_prob", "entropy", "value"):
        np.testing.assert_allclose(np.array(actual[name]), expected[name].detach().numpy(),
                                   rtol=3e-5, atol=3e-5)
    assert len(actual["decoder_trace"]) == len(expected["decoder_trace"])
    for (actual_logits, actual_mask), (expected_logits, expected_mask) in zip(
            actual["decoder_trace"], expected["decoder_trace"], strict=True):
        np.testing.assert_allclose(np.array(actual_logits), expected_logits.detach().numpy(),
                                   rtol=3e-5, atol=3e-5)
        np.testing.assert_array_equal(np.array(actual_mask), expected_mask.numpy())
    torch_loss = -(expected["log_prob"] + .02 * expected["entropy"]).mean()
    torch_loss.backward()

    def loss_fn(model):
        result = model.evaluate_episode(patches, actions, executed_actions=executed)
        return -mx.mean(result["log_prob"] + .02 * result["entropy"])

    loss, gradients = mnn.value_and_grad(policy, loss_fn)(policy)
    mx.eval(loss, gradients)
    assert float(loss) == pytest.approx(float(torch_loss.detach()), rel=3e-5, abs=2e-5)
    for name, parameter in agent.named_parameters():
        if name.startswith("decoder."):
            actual_gradient = np.array(gradients["weights"][name])
            assert np.isfinite(actual_gradient).all(), name
            assert parameter.grad is not None
            np.testing.assert_allclose(actual_gradient, parameter.grad.numpy(),
                                       rtol=3e-3, atol=8e-6, err_msg=name)
    for name in ("decoder.motion_head.weight", "decoder.motion_embedding.weight"):
        assert np.linalg.norm(np.array(gradients["weights"][name])) > 0
    policy.export_to_torch(agent)
    assert agent.decoder.motor_version == "motor_v2"
    assert agent.config.critic_initialization == "zero"
