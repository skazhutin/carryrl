"""Behavioral checks for checkpoint continuation, frozen evaluation, and failure records."""

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from carryrl.envs.navigation import NavigationEnv
from carryrl.evaluation.runner import evaluate
from carryrl.training.config import RunConfig
from carryrl.training.run import load_checkpoint, train


def tiny_config(**changes) -> RunConfig:
    return RunConfig.from_dict({
        "total_timesteps": 64, "n_steps": 16, "batch_size": 8, "n_epochs": 1,
        "eval_interval": 64, "eval_seeds": [1101, 1102], "max_episode_steps": 8,
        "embedding_size": 32, "max_wall_seconds": 60, **changes,
    })


@pytest.mark.parametrize("change", [
    {"surprise": 1}, {"device": "auto"}, {"seed": True}, {"seed": 2**32},
    {"n_steps": 17}, {"total_timesteps": 63}, {"learning_rate": float("nan")},
    {"image_size": 32}, {"eval_seeds": [1, 1]}, {"eval_seeds": [2**32]},
])
def test_reject_invalid_config(change):
    with pytest.raises((ValueError, TypeError)):
        tiny_config(**change)


def test_checkpoint_preserves_weights_optimizer_and_continues_on_fresh_episode(tmp_path):
    config = tiny_config()
    first = train(config, tmp_path / "first")
    assert first["status"] == "completed"
    assert first["timesteps"] == 64
    checkpoint = Path(first["latest_checkpoint"])
    with ZipFile(checkpoint / "model.zip") as archive:
        assert "policy.optimizer.pth" in archive.namelist()
        saved_optimizer = torch.load(BytesIO(archive.read("policy.optimizer.pth")),
                                     map_location="cpu", weights_only=True)
    resumed_config = replace(config, total_timesteps=128)
    env = NavigationEnv(max_steps=config.max_episode_steps)
    model, progress = load_checkpoint(checkpoint, env, resumed_config)
    try:
        before = {name: tensor.clone() for name, tensor in model.policy.state_dict().items()}
        before_steps = model.num_timesteps
        optimizer_state = model.policy.optimizer.state_dict()
        assert optimizer_state["state"]
        for index, saved_state in saved_optimizer["state"].items():
            for name, tensor in saved_state.items():
                assert torch.equal(tensor, optimizer_state["state"][index][name])
        initial_model, _ = load_checkpoint(tmp_path / "first/checkpoints/step_00000000",
                                          env, config)
        assert any(not torch.equal(before[name], tensor)
                   for name, tensor in initial_model.policy.state_dict().items())
        evaluation = evaluate(model, seeds=config.eval_seeds, max_episode_steps=8)
        assert evaluation["split"] == "validation"
        assert model.num_timesteps == before_steps
        assert all(torch.equal(before[name], tensor)
                   for name, tensor in model.policy.state_dict().items())
        assert progress["resume_rule"].startswith("weights and optimizer restored; new episode")
        assert model._last_obs is None
    finally:
        env.close()
    second = train(resumed_config, tmp_path / "second", resume=checkpoint)
    assert second["initial_timesteps"] == 64
    assert second["timesteps"] == 128
    assert second["decisions_this_run"] == 64
    assert second["optimizer_epochs_this_run"] == 4
    assert (tmp_path / "second" / "agent_fixed_seed.gif").is_file()
    records = [json.loads(line) for line in
               (tmp_path / "second" / "evaluation.jsonl").read_text().splitlines()]
    assert all(record["split"] == "validation" for record in records)
    with pytest.raises(FileExistsError):
        train(config, tmp_path / "first")


def test_resume_rejects_changed_algorithm_settings(tmp_path):
    config = tiny_config()
    first = train(config, tmp_path / "first")
    env = NavigationEnv()
    try:
        with pytest.raises(ValueError, match="preserve all settings"):
            load_checkpoint(Path(first["latest_checkpoint"]), env,
                            replace(config, total_timesteps=128, gamma=0.9))
    finally:
        env.close()


def test_numerical_failure_is_recorded_and_does_not_report_completion(tmp_path, monkeypatch):
    import carryrl.training.run as run_module

    class NaNEnv(NavigationEnv):
        def step(self, action):
            obs, _, terminated, truncated, info = super().step(action)
            return obs, float("nan"), terminated, truncated, info

    monkeypatch.setattr(run_module, "NavigationEnv", NaNEnv)
    output = tmp_path / "failed"
    with pytest.raises(FloatingPointError, match="nonfinite training reward"):
        train(tiny_config(), output)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["latest_checkpoint_retained"] is True
    assert not (output / "summary.json").exists()


def test_wall_budget_stopping_is_not_completed_training(tmp_path):
    summary = train(tiny_config(max_wall_seconds=0.000001), tmp_path / "stopped")
    assert summary["status"] == "resource_stopped"
    assert summary["timesteps"] == 0
    assert summary["decisions_this_run"] == 0
    assert summary["learning_decisions_per_second"] is None


def test_policy_interface_receives_only_rgb(tmp_path):
    class Mode:
        training = True

        def set_training_mode(self, mode):
            self.training = mode

    class ScreenOnlyPolicy:
        policy = Mode()
        frames_seen = 0

        def predict(self, pixels, deterministic):
            assert isinstance(pixels, np.ndarray)
            assert pixels.shape == (64, 64, 3) and pixels.dtype == np.uint8
            assert deterministic is True
            self.frames_seen += 1
            return 0, None

    policy = ScreenOnlyPolicy()
    result = evaluate(policy, seeds=(1001,), max_episode_steps=3)
    assert policy.frames_seen == 3
    assert result["episodes_started"] == 1
    assert result["unknown_outcomes"] == 0
    assert policy.policy.training is True


def test_cli_returns_nonzero_for_resource_stopped_run(tmp_path, monkeypatch):
    import carryrl.cli as cli

    monkeypatch.setattr(cli, "train", lambda *args: {"status": "resource_stopped"})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(tiny_config().to_dict()))
    assert cli.main(["train", "--config", str(config_path), "--output",
                     str(tmp_path / "run")]) == 1


def test_evaluation_error_restores_policy_mode_and_records_cli_failure(tmp_path, monkeypatch):
    import carryrl.cli as cli
    import carryrl.evaluation.runner as runner

    closed = []

    class TrackingEnv(NavigationEnv):
        def close(self):
            closed.append(True)
            return super().close()

    class Mode:
        training = True

        def set_training_mode(self, mode):
            self.training = mode

    class BrokenPolicy:
        policy = Mode()

        def predict(self, pixels, deterministic):
            self.policy.set_training_mode(False)
            raise RuntimeError("intentional inference error")

    monkeypatch.setattr(runner, "NavigationEnv", TrackingEnv)
    policy = BrokenPolicy()
    with pytest.raises(RuntimeError, match="intentional inference error"):
        evaluate(policy, seeds=(1001,), max_episode_steps=3)
    assert policy.policy.training is True
    assert closed == [True]
    monkeypatch.setattr(cli.PPO, "load", lambda *args, **kwargs: policy)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(tiny_config().to_dict()))
    output = tmp_path / "evaluation"
    with pytest.raises(RuntimeError, match="intentional inference error"):
        cli.main(["evaluate", "--config", str(config_path), "--output", str(output),
                  "--checkpoint", str(tmp_path / "unused.zip")])
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert not (output / "evaluation.json").exists()
