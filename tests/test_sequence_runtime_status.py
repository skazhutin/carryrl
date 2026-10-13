"""Actual PPO progress, measured evaluation binding, and bounded disk stops."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_sequence_training import TinyReward, config, factory

import carryrl.training.sequence as sequence
from carryrl.training.sequence_config import SequenceRunConfig


def read(path):
    return json.loads(path.read_text())


def test_budget_expires_in_periodic_evaluation_refreshes_only_final_binding(tmp_path, monkeypatch):
    clock = {"now": 0.0}
    calls = []
    original_evaluate = sequence.evaluate_sequence

    def evaluate(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        calls.append(result)
        clock["now"] = 11.0
        return result

    monkeypatch.setattr(sequence.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(sequence, "evaluate_sequence", evaluate)
    output = tmp_path / "during_eval"
    cfg = replace(config(), total_rollouts=3, checkpoint_every_rollouts=3,
                  eval_every_rollouts=1, max_wall_seconds=10)
    summary = sequence.train_sequence(cfg, output, agent_factory=factory,
                                      reward=TinyReward(), reward_factory=TinyReward)
    assert summary["status"] == "resource_stopped" and len(calls) == 1
    periodic = read(output / "evaluations/rollout_0001.json")
    final = read(output / "evaluation.json")
    assert not periodic["current_policy_saved"] and periodic["latest_checkpoint_rollout"] == 0
    assert final["current_policy_saved"] and final["latest_checkpoint_rollout"] == 1
    assert final["checkpoint_binding_refreshed"]
    binding_keys = {*sequence_binding_keys(), "checkpoint_binding_refreshed"}
    assert {key: value for key, value in final.items() if key not in binding_keys} == {
        key: value for key, value in periodic.items() if key not in binding_keys}
    manifest = read(Path(summary["latest_checkpoint"]) / "manifest.json")
    assert final["agent_sha256"] == periodic["agent_sha256"] == manifest["agent_sha256"]


def sequence_binding_keys():
    return {"latest_checkpoint", "latest_checkpoint_rollout", "latest_checkpoint_decisions",
            "latest_checkpoint_weights_revision", "current_policy_saved"}


def test_real_episode_heartbeats_phases_and_accepted_update_counters(tmp_path, monkeypatch):
    clock = {"now": 0.0}
    snapshots = []
    original_json, original_collect = sequence.atomic_json, sequence.collect_episode

    def atomic(path, data):
        if path.name == "status.json":
            snapshots.append((clock["now"], deepcopy(data)))
        return original_json(path, data)

    def collect(*args, **kwargs):
        actual_decision = kwargs["on_decision"]

        def decision():
            clock["now"] += .6
            actual_decision()
        kwargs["on_decision"] = decision
        return original_collect(*args, **kwargs)

    monkeypatch.setattr(sequence.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(sequence, "atomic_json", atomic)
    monkeypatch.setattr(sequence, "collect_episode", collect)
    output = tmp_path / "heartbeat"
    cfg = replace(config(), evaluate_initial=True, eval_every_rollouts=3)
    summary = sequence.train_sequence(cfg, output, agent_factory=factory,
                                      reward=TinyReward(), reward_factory=TinyReward)
    phases = [row["phase"] for _, row in snapshots]
    assert {"loading", "saving", "initial_evaluation", "collecting", "optimizing",
            "final_evaluation", "finished"} <= set(phases)
    collecting = [(now, row) for now, row in snapshots if row["phase"] == "collecting"]
    assert len(collecting) >= 3
    assert all(later[0] - earlier[0] >= 2
               for earlier, later in zip(collecting, collecting[1:]))
    for _, row in collecting:
        assert row["phase_index"] == 3 and row["current_rollout"] == row["current_cycle"] == 1
        assert row["rollouts"] == 0 and row["phase_unit_kind"] == "episodes"
        assert row["phase_units_total"] == 2 and 0 <= row["phase_units_done"] <= 2
        assert row["optimizer_steps_this_process"] == 0
        assert row["phase_elapsed_seconds"] == row["wall_seconds"]
    optimizing = [row for _, row in snapshots if row["phase"] == "optimizing"]
    assert optimizing[0]["optimizer_steps_this_process"] == 0
    assert optimizing[0]["phase_units_total"] is None
    assert summary["optimizer_steps_this_process"] == 1
    assert summary["optimizer_steps_upper_bound"] == 1
    assert summary["total_decisions_upper_bound"] == 8
    assert summary["attempted_train_decisions_this_process"] == 8
    assert summary["phase"] == "finished" and summary["rollouts_total"] == 1
    evaluation_status = [row for _, row in snapshots if row["phase"] == "final_evaluation"]
    assert evaluation_status[-1]["phase_units_done"] == evaluation_status[-1]["phase_units_total"]
    assert evaluation_status[-1]["phase_units_done"] > 0
    row = json.loads((output / "training.jsonl").read_text().splitlines()[0])
    assert row["complete"] and row["cycle_wall_seconds"] == pytest.approx(4.8)


def test_disk_floor_drop_saves_actual_off_cadence_state(tmp_path, monkeypatch):
    free = {"bytes": 10 * 1024**2}
    original_update = sequence.update_ppo

    def update(*args, **kwargs):
        stats = original_update(*args, **kwargs)
        free["bytes"] = 1024**2 - 1
        return stats

    monkeypatch.setattr(sequence.shutil, "disk_usage",
                        lambda _: SimpleNamespace(free=free["bytes"]))
    monkeypatch.setattr(sequence, "update_ppo", update)
    output = tmp_path / "disk"
    cfg = replace(config(), total_rollouts=3, checkpoint_every_rollouts=32,
                  eval_every_rollouts=3, minimum_free_disk_mib=1)
    summary = sequence.train_sequence(cfg, output, agent_factory=factory,
                                      reward=TinyReward(), reward_factory=TinyReward)
    assert summary["status"] == "resource_stopped"
    assert summary["stop_reason"] == "minimum_free_disk_mib"
    assert summary["free_disk_bytes"] == 1024**2 - 1
    assert summary["minimum_free_disk_bytes"] == 1024**2
    assert summary["rollouts"] == summary["latest_checkpoint_rollout"] == 1
    assert summary["optimizer_steps_this_process"] == 1 and summary["current_policy_saved"]
    assert [path.name for path in sorted((output / "checkpoints").iterdir())] == [
        "rollout_0000", "rollout_0001"]
    final = read(output / "evaluation.json")
    manifest = read(Path(summary["latest_checkpoint"]) / "manifest.json")
    assert final["agent_sha256"] == manifest["agent_sha256"]


@pytest.mark.parametrize("value", [-1, True, 1.5, "1024", None])
def test_disk_floor_is_strict_nonnegative_integer(value):
    with pytest.raises(ValueError, match="minimum_free_disk_mib"):
        SequenceRunConfig(minimum_free_disk_mib=value)
