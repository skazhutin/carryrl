"""Lower precision frozen caches preserve full precision policy updates."""

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from test_imitation import make_agent

from carryrl.training.imitation import collect_scripted_demonstrations, train_imitation

spec = importlib.util.spec_from_file_location(
    "train_imitation_script", Path(__file__).resolve().parents[1] / "scripts/train_imitation.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


@pytest.mark.parametrize("budget", ["0", "-1"])
def test_cli_rejects_nonpositive_frame_budget_before_loading_config(monkeypatch, capsys, budget):
    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "missing.json",
        "--warm-start", "missing", "--output", "unused", "--frame-budget-mib", budget])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
    assert "frame budget must be positive" in capsys.readouterr().err


def test_cli_rejects_oversized_learner_prefix_before_creating_output(monkeypatch, tmp_path, capsys):
    from carryrl.training.sequence_config import SequenceRunConfig

    output = tmp_path / "unused"
    config = SequenceRunConfig(max_episode_steps=2)
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "missing.json",
        "--warm-start", "missing", "--output", str(output), "--learner-prefix-steps", "3"])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
    assert "must not exceed" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_half_cache_actual_bytes_budget_and_full_precision_training(backend):
    if backend == "mlx":
        mx = pytest.importorskip("mlx.core")
        if not mx.metal.is_available():
            pytest.skip("MLX requires Metal")

    agent = make_agent()
    reference = deepcopy(agent)
    dataset = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=0)
    features, rounded = {}, {}
    for index, frame in enumerate(dataset.frames):
        encoded = agent.encode_frames(torch.tensor(frame)[None])
        size = encoded.numel() * 2
        with pytest.raises(MemoryError, match="feature-budget-mib"):
            script._cache_feature_batch(encoded, dtype="float16", backend=backend,
                                        remaining_bytes=size - 1)
        values, stored_bytes = script._cache_feature_batch(
            encoded, dtype="float16", backend=backend, remaining_bytes=size)
        assert stored_bytes == size
        feature = features[index] = values[0]
        assert feature.dtype == (torch.float16 if backend == "torch" else mx.float16)
        if backend == "torch":
            assert not feature.requires_grad
        rounded[index] = encoded.detach().half().float()[0]

    float_values, float_bytes = script._cache_feature_batch(
        encoded, dtype="float32", backend=backend, remaining_bytes=size * 2)
    assert float_bytes == size * 2
    assert float_values[0].dtype == (torch.float32 if backend == "torch" else mx.float32)
    optimizers = [torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
                  for model in (agent, reference)]
    engine = None
    if backend == "mlx":
        from carryrl.training.mlx import MLXPPOBackend

        engine = MLXPPOBackend(agent, optimizers[0])
    result = train_imitation(
        agent, dataset, optimizer=optimizers[0], backend=backend, mlx_backend=engine,
        feature_provider=lambda index: script._training_feature(
            features[index], backend=backend, dtype=agent.summary_query.dtype))
    expected = train_imitation(reference, dataset, optimizer=optimizers[1],
                               feature_provider=rounded.__getitem__)
    assert result["history"][0]["nll"] == pytest.approx(expected["history"][0]["nll"], abs=3e-6)
    for name, parameter in agent.named_parameters():
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter, dict(reference.named_parameters())[name],
                                   atol=4e-6, rtol=3e-4)
    for state in optimizers[0].state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
