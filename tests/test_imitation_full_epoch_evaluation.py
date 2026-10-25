"""Exercise epoch scheduling without loading models or performing optimizer updates."""

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_imitation_feature_storage import script

from carryrl.training.run import atomic_json
from carryrl.training.sequence_config import SequenceRunConfig


@pytest.fixture
def run_stub(monkeypatch, tmp_path):
    config = SequenceRunConfig(tasks=("navigation",), eval_episodes_per_task=1)
    source, output = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    atomic_json(source / "manifest.json", {"sha256": "source archive"})
    agent = SimpleNamespace(version=0, vision=object(), summary_query=torch.zeros(1),
                            encode_frames=lambda _: torch.zeros(1, 1, 384))
    reward = SimpleNamespace(version=0)
    optimizer = SimpleNamespace(param_groups=[{}])

    class Episode:
        frame_indices = (0,)
        final_frame_index = 0
        presentation = native_size = None
        source = "scripted_teacher"

        def __len__(self):
            return 1

    dataset = SimpleNamespace(episodes=[Episode()],
        frames=[np.zeros((4, 4, 3), dtype=np.uint8)], provenance=lambda: {})
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(script, "provenance", lambda: {})
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k:
        (agent, optimizer, reward, {"decisions": 123, "weights_revision": 7,
                                   "config": config.to_dict()}))
    monkeypatch.setattr(script, "tensor_hash", lambda value:
        f"agent{value.version}" if value is agent else
        f"reward{value.version}" if value is reward else "vision")
    monkeypatch.setattr(script, "collect_scripted_demonstrations", lambda *a, **k: dataset)
    monkeypatch.setattr(script, "validate_dataset_runtime_holdout", lambda *a, **k: None)
    monkeypatch.setattr(script, "link_matching_playbacks", lambda *a: None)
    calls, likelihood_calls = [], []
    frozen = [True]
    protocol_change = [False]
    original_protocol = script.sequence_evaluation_protocol

    def protocol(config):
        result = original_protocol(config)
        if protocol_change[0]:
            result["eval_seed"] += 1
        return result

    monkeypatch.setattr(script, "sequence_evaluation_protocol", protocol)

    def evaluate(agent, reward, config, **kwargs):
        measured_protocol = protocol(config)
        calls.append((agent.version, config.eval_all_levels, measured_protocol))
        return {"split": "validation", "policy_and_reward_frozen": frozen[0],
            "protocol_sha256": hashlib.sha256(
                json.dumps(measured_protocol, sort_keys=True).encode()).hexdigest(),
            "records": [{}] * sum(measured_protocol["counts"].values()),
            "summaries": {"frozen_deterministic": {"outcomes": {"success": 0},
                "episodes_started": sum(measured_protocol["counts"].values()), "tasks": {}}}}

    monkeypatch.setattr(script, "evaluate_sequence", evaluate)

    def likelihood(agent, config, *, reward, checkpoint_weights_revision, **kwargs):
        likelihood_calls.append((agent.version, config.eval_all_levels,
                                 config.eval_episodes_per_task))
        return {"agent_sha256": f"agent{agent.version}", "reward_sha256": "reward0",
                "checkpoint_weights_revision": checkpoint_weights_revision}

    monkeypatch.setattr("carryrl.evaluation.imitation.evaluate_imitation_validation", likelihood)

    def save(agent, optimizer, reward, config, destination, **counters):
        checkpoint = destination / "checkpoints/rollout_0000"
        checkpoint.mkdir(parents=True)
        atomic_json(checkpoint / "manifest.json", {"sha256": "archive",
            "agent_sha256": f"agent{agent.version}", "reward_sha256": "reward0",
            "weights_revision": counters["revision"]})
        return checkpoint

    monkeypatch.setattr(script, "save_sequence_checkpoint", save)

    def run(*, full=True, final_full=True, complete=(True, True), interval=1, mutate=None):
        def train(*args, **kwargs):
            for epoch, completed in enumerate(complete, 1):
                agent.version = epoch
                kwargs["on_epoch"]({"epoch": epoch, "complete": completed,
                    "optimizer_steps": epoch, "nll": 1., "seconds": epoch * .01,
                    "epoch_training_seconds": .01, "episodes": 1,
                    "per_task_nll": {"navigation": 1.}})
            if mutate == "agent":
                agent.version += 1
            elif mutate == "reward":
                reward.version += 1
            elif mutate == "revision":
                kwargs["on_update"]({"epoch": len(complete), "optimizer_steps": 3,
                    "nll": 1., "gradient_norm": 0., "seconds": .03})
            elif mutate == "protocol":
                protocol_change[0] = True
            return {"optimizer_steps": 3 if mutate == "revision" else len(complete),
                    "stop_reason": "completed" if all(complete) else "wall_budget"}

        monkeypatch.setattr(script, "train_imitation", train)
        monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "unused",
            "--warm-start", str(source), "--output", str(output),
            "--training-backend", "torch", "--epochs", "2", "--eval-every", str(interval),
            "--checkpoint-every", "1", "--validation-nll",
            "--validation-nll-episodes-per-task", "1",
            *(["--eval-all-levels"] if full else []),
            *(["--final-all-levels"] if final_full else [])])
        script.main()
        return output

    return SimpleNamespace(run=run, calls=calls, likelihood_calls=likelihood_calls,
                           frozen=frozen, output=output)


def test_every_completed_epoch_uses_full_bank_and_final_reuses_exact_measurement(run_stub):
    output = run_stub.run()
    assert [(epoch, full) for epoch, full, _ in run_stub.calls] == [
        (0, True), (1, True), (2, True)]
    assert all(row[2]["counts"]["navigation"] > 1 for row in run_stub.calls)
    assert run_stub.likelihood_calls == [(0, False, 1), (1, False, 1), (2, False, 1)]
    measured = json.loads((output / "evaluations/epoch_0002.json").read_text())
    final = json.loads((output / "evaluations/epoch_0002_final.json").read_text())
    reused = final.pop("final_evaluation_reuse")
    assert final == measured and reused["measurement_reused"] is True
    assert reused["source"] == str((output / "evaluations/epoch_0002.json").resolve())
    snapshot = json.loads((output /
        "checkpoints_by_epoch/epoch_0001/checkpoints/rollout_0000/manifest.json").read_text())
    assert snapshot["evaluation_reference"].endswith("evaluations/epoch_0001.json")
    assert snapshot["agent_sha256"] == "agent1" and snapshot["weights_revision"] == 8
    assert snapshot["validation_likelihood_reference"].endswith("epoch_0001.json")
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["eval_all_levels"] is metadata["settings"]["eval_all_levels"] is True
    assert metadata["config"]["eval_all_levels"] is False
    status = json.loads((output / "status.json").read_text())
    assert status["epochs_completed"] == 2 and status["optimizer_steps"] == 2
    assert status["training_budget_seconds"] >= 0


def test_legacy_flag_keeps_short_scheduled_evaluations_and_separate_full_final(run_stub):
    output = run_stub.run(full=False)
    assert [(epoch, full) for epoch, full, _ in run_stub.calls] == [
        (0, True), (1, False), (2, False), (2, True)]
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["eval_all_levels"] is False
    assert "eval_all_levels" not in metadata["settings"]


def test_full_bank_flag_preserves_scheduling_interval(run_stub):
    run_stub.run(interval=2)
    assert [(epoch, full) for epoch, full, _ in run_stub.calls] == [(0, True), (2, True)]


def test_partial_epoch_gets_fresh_full_final_and_no_completed_snapshot(run_stub):
    output = run_stub.run(complete=(True, False))
    assert [(epoch, full) for epoch, full, _ in run_stub.calls] == [
        (0, True), (1, True), (2, True)]
    assert not (output / "evaluations/epoch_0002.json").exists()
    assert not (output / "checkpoints_by_epoch/epoch_0002").exists()
    final = json.loads((output / "evaluations/epoch_0002_final.json").read_text())
    assert final["agent_sha256"] == "agent2" and "final_evaluation_reuse" not in final
    status = json.loads((output / "status.json").read_text())
    assert status["epochs_completed"] == 1 and status["stop_reason"] == "wall_budget"


def test_stop_after_completed_epoch_reuses_its_full_measurement(run_stub):
    output = run_stub.run(complete=(True,))
    assert [(epoch, full) for epoch, full, _ in run_stub.calls] == [(0, True), (1, True)]
    assert run_stub.likelihood_calls == [(0, False, 1), (1, False, 1)]
    final = json.loads((output / "evaluations/epoch_0001_final.json").read_text())
    assert final["final_evaluation_reuse"]["source"].endswith("epoch_0001.json")


@pytest.mark.parametrize("mutate", ["agent", "reward", "revision", "protocol", "frozen"])
def test_final_reuse_requires_matching_frozen_actor_reward_revision_and_protocol(run_stub, mutate):
    if mutate == "frozen":
        run_stub.frozen[0] = False
    if mutate == "reward":
        with pytest.raises(RuntimeError, match="changed the frozen reward"):
            run_stub.run(mutate=mutate)
    else:
        run_stub.run(mutate=mutate)
    assert len(run_stub.calls) == 4
    final = json.loads((run_stub.output / "evaluations/epoch_0002_final.json").read_text())
    assert "final_evaluation_reuse" not in final


def test_full_scheduled_and_short_final_use_distinct_protocols(run_stub):
    run_stub.run(final_full=False)
    assert [(epoch, full) for epoch, full, _ in run_stub.calls] == [
        (0, False), (1, True), (2, True), (2, False)]
