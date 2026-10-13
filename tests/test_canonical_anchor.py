"""Optional retention uses teacher component prefixes and actual episode history."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_imitation import make_agent

from carryrl.agent.actions import KEY_VOCAB, PhysicalAction
from carryrl.training.imitation import (
    CORRECTION_SOURCE,
    _canonical_anchor_parent,
    augment_demonstrations,
    collect_native_scripted_demonstrations,
    collect_scripted_demonstrations,
    teacher_prefix_conditional_kl,
    train_imitation,
)


def targets():
    return [PhysicalAction(), PhysicalAction(keys=("up", "9"), mouse_buttons=("left",),
                                             dx=8, dy=-4, wheel=1),
            PhysicalAction(keys=("right",), mouse_buttons=("middle",), dx=-64, dy=32)]


def test_trace_matches_legacy_and_masks_sorted_keys_and_eos_padding():
    agent = make_agent()
    features = torch.randn(3, agent.config.width)
    actions = targets()
    old = agent.decoder.evaluate(features, actions)
    logp, entropy, trace = agent.decoder.evaluate_with_trace(features, actions)
    torch.testing.assert_close(logp, old[0], rtol=0, atol=0)
    torch.testing.assert_close(entropy, old[1], rtol=0, atol=0)
    assert len(trace) == 7  # three key/EOS steps and all four mouse heads
    assert [mask.tolist() for _, mask in trace[:3]] == [
        [True, True, True], [False, True, True], [False, True, False]]
    assert torch.isneginf(trace[1][0][1, 0])  # cannot repeat or reverse sorted "up"
    assert torch.isneginf(trace[2][0][1, :len(KEY_VOCAB)]).all()  # highest key -> EOS
    kl = teacher_prefix_conditional_kl(trace, trace)
    assert float(kl.detach()) == 0.0
    kl.backward()
    assert all(torch.isfinite(p.grad).all() for p in agent.decoder.parameters()
               if p.grad is not None)


def test_parent_shares_frozen_encoders_and_divergent_student_has_only_student_gradients():
    agent = make_agent(instruction=True)
    parent, digest = _canonical_anchor_parent(agent, None)
    assert len(digest) == 64
    assert parent.vision is agent.vision and parent.language is agent.language
    assert not any(p.requires_grad for p in parent.parameters())
    before = {name: p.detach().clone() for name, p in parent.named_parameters()}
    with torch.no_grad():
        agent.decoder.key_head.bias[0] += .4
        agent.decoder.mouse_head.bias[0] += .7
        agent.decoder.dx_head.bias[0] += .6
        agent.decoder.dy_head.bias[0] += .3
        agent.decoder.wheel_head.bias[0] += .8
    patches = torch.randn(3, 4, 384)
    executed = [PhysicalAction(keys=("left",))] * 3
    teacher = parent.evaluate_episode(patches, targets(), executed_actions=executed,
                                     instruction="Remember", return_decoder_trace=True)
    student = agent.evaluate_episode(patches, targets(), executed_actions=executed,
                                    instruction="Remember", return_decoder_trace=True)
    kl = teacher_prefix_conditional_kl(teacher["decoder_trace"], student["decoder_trace"])
    assert float(kl.detach()) > 0 and torch.isfinite(kl)
    kl.backward()
    assert not any(p.grad is not None for p in parent.parameters())
    assert agent.previous_action_embedding.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in agent.parameters() if p.grad is not None)
    for name, p in parent.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    # The previous-action path must not replace actual learner inputs with teacher labels.
    wrong_history = parent.evaluate_episode(patches, targets(), executed_actions=targets(),
                                           instruction="Remember", return_decoder_trace=True)
    assert not torch.allclose(wrong_history["log_prob"], teacher["log_prob"])


def data_and_features():
    dataset = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=0)
    values = {index: torch.randn(4, 384) for index in range(len(dataset.frames))}
    return dataset, values


def test_zero_coefficient_exact_parameter_and_objective_parity():
    original = make_agent()
    agents = [deepcopy(original), deepcopy(original)]
    dataset, features = data_and_features()
    results = [train_imitation(agent, dataset, epochs=2,
                              feature_provider=features.__getitem__,
                              **({"canonical_anchor_coefficient": 0.0} if index else {}))
               for index, agent in enumerate(agents)]
    for (name, p), (_, q) in zip(agents[0].named_parameters(), agents[1].named_parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0, msg=name)
    for first, second in zip(results[0]["history"], results[1]["history"]):
        assert first["objective"] == second["objective"]
        assert first["anchor_kl"] == second["anchor_kl"] == 0
        assert first["total_objective"] == first["bc_objective"]


def test_mixed_domain_anchor_separately_normalizes_eligible_episodes(monkeypatch):
    import carryrl.training.imitation as module

    dataset, features = data_and_features()
    episode = dataset.episodes[0]
    actual = tuple(PhysicalAction(keys=("left",)) for _ in episode.actions)
    dataset.episodes[0] = replace(episode, source=CORRECTION_SOURCE,
        executed_actions=actual, expert_probability=.5, terminated=True, truncated=False)
    dataset.extend(collect_native_scripted_demonstrations(
        native_train_sizes=(64, 128), native_episodes_per_task=1, tasks=("navigation",)))
    dataset.extend(augment_demonstrations(dataset, copies_per_episode=1))
    features.update({index: torch.randn(4, 384) for index in range(len(dataset.frames))})
    seen = []
    original_evaluate = module.deepcopy

    def snapshot(*args, **kwargs):
        parent = original_evaluate(*args, **kwargs)
        evaluate = parent.evaluate_episode

        def record(patches, actions, **options):
            seen.append(options["executed_actions"])
            return evaluate(patches, actions, **options)

        parent.evaluate_episode = record
        return parent

    monkeypatch.setattr(module, "deepcopy", snapshot)
    # Fixed positive KL makes the normalization independently observable.
    monkeypatch.setattr(module, "teacher_prefix_conditional_kl",
                        lambda parent, student: student[0][0].exp().sum() * 0 + 2.0)
    updates = []
    result = train_imitation(make_agent(), dataset, canonical_anchor_coefficient=3,
        minibatch_episodes=len(dataset.episodes), feature_provider=features.__getitem__,
        on_update=updates.append)
    assert seen == [actual]
    row = updates[0]
    assert row["anchor_eligible_episodes"] == 1
    assert row["anchor_kl"] == 2
    assert row["total_objective"] == pytest.approx(row["bc_objective"] + 6)
    assert result["canonical_anchor"]["eligible_episodes"] == 1


@pytest.mark.parametrize("coefficient", [-1, float("nan"), float("inf"), True])
def test_invalid_coefficients_fail_before_model_or_features(coefficient):
    dataset, _ = data_and_features()
    with pytest.raises(ValueError, match="canonical anchor coefficient"):
        train_imitation(make_agent(), dataset, canonical_anchor_coefficient=coefficient)


def test_styled_and_native_only_have_zero_anchor_and_identical_bc_update():
    native = collect_native_scripted_demonstrations(native_train_sizes=(64, 128),
        native_episodes_per_task=1, tasks=("navigation",))
    dataset, _ = data_and_features()
    styled = augment_demonstrations(dataset, copies_per_episode=1)
    native.extend(styled)
    features = {index: torch.randn(4, 384) for index in range(len(native.frames))}
    first, second = make_agent(), make_agent()
    base = train_imitation(first, native, minibatch_episodes=len(native.episodes),
                          feature_provider=features.__getitem__)
    anchored = train_imitation(second, native, minibatch_episodes=len(native.episodes),
        feature_provider=features.__getitem__, canonical_anchor_coefficient=2)
    assert anchored["canonical_anchor"]["eligible_episodes"] == 0
    assert anchored["history"][0]["anchor_kl"] == 0
    assert anchored["history"][0]["total_objective"] == base["history"][0]["objective"]
    for p, q in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_real_mlx_torch_conditional_kl_values_and_gradients_match():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("MLX anchor parity requires Metal")
    import mlx.nn as mnn

    from carryrl.agent.mlx_policy import MLXPolicy, to_mlx

    student = make_agent(instruction=True)
    parent, _ = _canonical_anchor_parent(student, None)
    # Conversion must precede freezing Torch requires_grad.
    mlx_parent = MLXPolicy.from_torch(deepcopy(student))
    mlx_parent.freeze()
    with torch.no_grad():
        student.decoder.key_head.bias[0] += .7
        student.decoder.dx_head.bias[0] += .4
        student.previous_action_embedding.weight[0, 0] += .2
    mlx_student = MLXPolicy.from_torch(student)
    patches = torch.randn(3, 4, 384)
    actions = targets()
    actual = [PhysicalAction(keys=("left",))] * 3
    instruction = "Remember the cue"
    teacher = parent.evaluate_episode(patches, actions, executed_actions=actual,
                                     instruction=instruction, return_decoder_trace=True)
    output = student.evaluate_episode(patches, actions, executed_actions=actual,
                                      instruction=instruction, return_decoder_trace=True)
    loss = teacher_prefix_conditional_kl(teacher["decoder_trace"], output["decoder_trace"])
    loss.backward()
    raw, language = to_mlx(patches), to_mlx(student._encode_instruction(instruction))
    teacher_mlx = mlx_parent.evaluate_episode(raw, actions, executed_actions=actual,
        instruction_features=language, return_decoder_trace=True)

    def mlx_loss():
        predicted = mlx_student.evaluate_episode(raw, actions, executed_actions=actual,
            instruction_features=language, return_decoder_trace=True)
        return teacher_prefix_conditional_kl(teacher_mlx["decoder_trace"],
                                             predicted["decoder_trace"])

    mlx_value, grads = mnn.value_and_grad(mlx_student, mlx_loss)()
    mx.eval(mlx_value, grads)
    assert float(mlx_value) == pytest.approx(float(loss.detach()), abs=3e-6)
    for name, p in student.named_parameters():
        if p.grad is not None:
            np.testing.assert_allclose(np.asarray(grads["weights"][name]), p.grad.numpy(),
                                       atol=5e-6, rtol=5e-4, err_msg=name)


def test_anchored_mlx_training_matches_torch_and_snapshot_stays_initial(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("MLX anchor training requires Metal")
    import carryrl.training.imitation as module
    from carryrl.training.mlx import MLXPPOBackend

    dataset, features = data_and_features()
    torch_agent, mlx_agent = make_agent(), make_agent()
    optimizers = [torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad),
                                   lr=3e-4, weight_decay=.03)
                  for agent in (torch_agent, mlx_agent)]
    engine = MLXPPOBackend(mlx_agent, optimizers[1])
    original = module._canonical_anchor_parent
    snapshots = []

    def capture(agent, engine):
        parent, digest = original(agent, engine)
        if engine is None:
            before = {name: p.detach().clone() for name, p in parent.named_parameters()}
        else:
            before = {name: np.array(value) for name, value in parent.weights.items()}
        snapshots.append((parent, before, digest))
        return parent, digest

    monkeypatch.setattr(module, "_canonical_anchor_parent", capture)
    results = []
    for agent, optimizer, backend in zip(
            (torch_agent, mlx_agent), optimizers, ("torch", "mlx"), strict=True):
        results.append(train_imitation(agent, dataset, epochs=2, optimizer=optimizer,
            backend=backend, mlx_backend=engine if backend == "mlx" else None,
            canonical_anchor_coefficient=1, action_balance="active_idle",
            feature_provider=features.__getitem__))
    assert snapshots[0][2] == snapshots[1][2]
    for result in results:
        assert result["history"][0]["anchor_kl"] == pytest.approx(0, abs=1e-6)
        assert result["history"][1]["anchor_kl"] > 0
    for first, second in zip(results[0]["history"], results[1]["history"], strict=True):
        assert first["total_objective"] == pytest.approx(second["total_objective"], abs=5e-6)
    for name, p in torch_agent.named_parameters():
        torch.testing.assert_close(p, dict(mlx_agent.named_parameters())[name],
                                   atol=6e-6, rtol=5e-4)
    for parent, before, _ in snapshots:
        if hasattr(parent, "weights"):
            from mlx.utils import tree_flatten

            assert not tree_flatten(parent.trainable_parameters())
            for name, value in parent.weights.items():
                np.testing.assert_array_equal(value, before[name])
        else:
            for name, value in parent.named_parameters():
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)


@pytest.mark.parametrize("coefficient", ["-1", "nan", "inf"])
def test_cli_rejects_invalid_anchor_before_loading_config(coefficient, monkeypatch, capsys):
    import sys

    from test_imitation_replay_weights import script

    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "missing.json",
        "--warm-start", "missing", "--output", "missing",
        "--canonical-anchor-coefficient", coefficient])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
    assert "canonical anchor coefficient must be nonnegative and finite" in capsys.readouterr().err
