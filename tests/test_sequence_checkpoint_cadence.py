"""Sparse checkpoints preserve actual final optimizer state and stop-time provenance."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_sequence_training import TinyReward, config, factory

import carryrl.training.sequence as sequence
from carryrl.training.sequence_config import SequenceRunConfig


def read(path):
    return json.loads(path.read_text())


def test_sparse_cadence_matches_default_final_weights_optimizer_and_saved_snapshots(tmp_path):
    cfg = replace(config(), total_rollouts=3, eval_every_rollouts=1)
    results = {}
    for cadence in (1, 2):
        output = tmp_path / str(cadence)
        results[cadence] = sequence.train_sequence(
            replace(cfg, checkpoint_every_rollouts=cadence), output,
            agent_factory=factory, reward=TinyReward(), reward_factory=TinyReward)
        checkpoints = sorted((output / "checkpoints").iterdir())
        expected = [0, 1, 2, 3] if cadence == 1 else [0, 2, 3]
        assert [int(path.name.split("_")[1]) for path in checkpoints] == expected
        for path, rollout in zip(checkpoints, expected, strict=True):
            manifest = read(path / "manifest.json")
            archive = torch.load(path / "state.pt", weights_only=True)
            assert manifest["rollout"] == archive["rollout"] == rollout
            assert manifest["weights_revision"] == archive["weights_revision"] == rollout
            assert manifest["decisions"] == archive["decisions"] == rollout * 8
            if rollout:
                assert archive["optimizer_state"]["state"]
                assert all(state["step"].item() == rollout
                           for state in archive["optimizer_state"]["state"].values())
        assert read(output / "latest.json")["checkpoint"] == results[cadence]["latest_checkpoint"]
        final = read(output / "evaluation.json")
        saved = read(Path(results[cadence]["latest_checkpoint"]) / "manifest.json")
        assert final["agent_sha256"] == saved["agent_sha256"]
        assert final["current_policy_saved"] and final["latest_checkpoint_rollout"] == 3
        metadata = read(output / "metadata.json")
        assert metadata["checkpoint_frequency_rollouts"] == cadence
        assert metadata["checkpoint_state"]["current_rollout"] == 3
    sparse_periodic = read(tmp_path / "2/evaluations/rollout_0001.json")
    assert not sparse_periodic["current_policy_saved"]
    assert sparse_periodic["rollout"] == 1 and sparse_periodic["latest_checkpoint_rollout"] == 0
    first, second = [torch.load(Path(results[cadence]["latest_checkpoint"]) / "state.pt",
                               weights_only=True) for cadence in (1, 2)]
    for name, value in first["agent_state"].items():
        assert torch.equal(value, second["agent_state"][name])
    for parameter, state in first["optimizer_state"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, second["optimizer_state"]["state"][parameter][name])


def test_graceful_wall_stop_saves_off_interval_state_before_evaluation(tmp_path, monkeypatch):
    output = tmp_path / "wall"
    clock = {"now": 0.0}
    original_update, original_evaluate = sequence.update_ppo, sequence.evaluate_sequence
    evaluations = []

    def update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        clock["now"] = 11.0
        return result

    def evaluate(agent, reward, *args, **kwargs):
        latest = Path(read(output / "latest.json")["checkpoint"])
        saved = read(latest / "manifest.json")
        assert saved["rollout"] == 1
        assert saved["agent_sha256"] == sequence.tensor_hash(agent)
        archive = torch.load(latest / "state.pt", weights_only=True)
        assert archive["weights_revision"] == 1
        assert all(state["step"].item() == 1
                   for state in archive["optimizer_state"]["state"].values())
        evaluations.append(latest)
        return original_evaluate(agent, reward, *args, **kwargs)

    monkeypatch.setattr(sequence.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(sequence, "update_ppo", update)
    monkeypatch.setattr(sequence, "evaluate_sequence", evaluate)
    cfg = replace(config(), total_rollouts=5, checkpoint_every_rollouts=2,
                  eval_every_rollouts=3, max_wall_seconds=10)
    summary = sequence.train_sequence(cfg, output, agent_factory=factory,
                                      reward=TinyReward(), reward_factory=TinyReward)
    assert summary["status"] == "resource_stopped"
    assert summary["rollouts"] == summary["latest_checkpoint_rollout"] == 1
    assert summary["current_policy_saved"] and len(evaluations) == 1
    assert [path.name for path in sorted((output / "checkpoints").iterdir())] == [
        "rollout_0000", "rollout_0001"]


def test_ppo_failure_reports_only_last_saved_snapshot(tmp_path, monkeypatch):
    output = tmp_path / "failed"
    original_update = sequence.update_ppo
    updates = []

    def update(agent, *args, **kwargs):
        result = original_update(agent, *args, **kwargs)
        updates.append(sequence.tensor_hash(agent))
        if len(updates) == 2:
            raise RuntimeError("failed after applying unsaved PPO")
        return result

    monkeypatch.setattr(sequence, "update_ppo", update)
    cfg = replace(config(), total_rollouts=3, checkpoint_every_rollouts=3,
                  eval_every_rollouts=3)
    with pytest.raises(RuntimeError, match="unsaved PPO"):
        sequence.train_sequence(cfg, output, agent_factory=factory,
                                reward=TinyReward(), reward_factory=TinyReward)
    status = read(output / "status.json")
    assert status["status"] == "failed" and not status["current_policy_saved"]
    assert status["rollouts"] == 1 and status["latest_checkpoint_rollout"] == 0
    assert status["latest_checkpoint_weights_revision"] == 0
    saved = Path(read(output / "latest.json")["checkpoint"])
    assert read(saved / "manifest.json")["agent_sha256"] not in updates
    assert [path.name for path in (output / "checkpoints").iterdir()] == ["rollout_0000"]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_checkpoint_cadence_is_strict_positive_integer_and_legacy_default(value):
    with pytest.raises(ValueError, match="checkpoint_every_rollouts"):
        SequenceRunConfig(checkpoint_every_rollouts=value)
    legacy = config().to_dict()
    legacy.pop("checkpoint_every_rollouts")
    assert SequenceRunConfig.from_dict(legacy).checkpoint_every_rollouts == 1
