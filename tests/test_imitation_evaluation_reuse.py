"""Only complete measurements of the exact checkpoint and cohort may be reused."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_imitation_feature_storage import script
from test_level_bank_protocol import _FrozenPolicy
from torch import nn

from carryrl.training.sequence import evaluate_sequence, tensor_hash
from carryrl.training.sequence_config import SequenceRunConfig


@pytest.fixture
def reusable_modules():
    return _FrozenPolicy(), nn.Linear(1, 1)


@pytest.fixture
def reusable_evaluation(tmp_path, reusable_modules):
    config = SequenceRunConfig(tasks=("ui",), max_episode_steps=1, eval_all_levels=True,
                               eval_episodes_per_task=1)
    agent, reward = reusable_modules
    agent_hash, reward_hash = tensor_hash(agent), tensor_hash(reward)
    evaluated = evaluate_sequence(agent, reward, config, on_progress=lambda _: None)
    evaluated.update(epoch=8, weights_revision=3, agent_sha256=agent_hash)
    run = tmp_path / "source_run"
    checkpoint = run / "checkpoints" / "rollout_0000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text(json.dumps({
        "agent_sha256": agent_hash, "reward_sha256": reward_hash, "weights_revision": 3}))
    source = run / "evaluations" / "epoch_0008_final.json"
    source.parent.mkdir()
    source.write_text(json.dumps(evaluated))
    arguments = {"checkpoint": checkpoint, "config": config,
                 "agent_sha256": agent_hash, "reward_sha256": reward_hash, "weights_revision": 3}
    return source, evaluated, arguments


def test_reuse_preserves_measurement_records_metrics_and_timestamp(reusable_evaluation):
    source, original, arguments = reusable_evaluation
    reused = script._reuse_initial_evaluation(source, **arguments)
    assert reused["epoch"] == 0 and reused["weights_revision"] == 3
    for key in ("records", "summaries", "created_utc", "elapsed_seconds", "protocol_sha256"):
        assert reused[key] == original[key]
    assert reused["initial_evaluation_reuse"]["source"] == str(source.resolve())
    assert reused["initial_evaluation_reuse"]["source_created_utc"] == original["created_utc"]
    assert reused["initial_evaluation_reuse"]["source_epoch"] == 8
    assert json.loads(source.read_text()) == original


@pytest.mark.parametrize("mutation", ["agent", "protocol", "partial", "duplicate", "summary",
                                      "bank_record", "timestamp", "split", "unfrozen"])
def test_reuse_rejects_changed_or_incomplete_artifact(reusable_evaluation, mutation):
    source, evaluated, arguments = reusable_evaluation
    if mutation == "agent":
        evaluated["agent_sha256"] = "wrong"
    elif mutation == "protocol":
        evaluated["protocol_sha256"] = "wrong"
    elif mutation == "partial":
        evaluated["records"].pop()
    elif mutation == "duplicate":
        evaluated["records"][-1] = evaluated["records"][0]
    elif mutation == "summary":
        evaluated["summaries"]["frozen_deterministic"]["episodes_started"] += 1
    elif mutation == "bank_record":
        evaluated["records"][0]["level_id"] = "wrong"
    elif mutation == "timestamp":
        evaluated.pop("created_utc")
    elif mutation == "split":
        evaluated["split"] = "test"
    else:
        evaluated["policy_and_reward_frozen"] = False
    source.write_text(json.dumps(evaluated))
    with pytest.raises(ValueError):
        script._reuse_initial_evaluation(source, **arguments)


def test_reuse_rejects_different_current_bank(reusable_evaluation):
    source, _, arguments = reusable_evaluation
    arguments["config"] = replace(arguments["config"], level_bank_seed=7193)
    with pytest.raises(ValueError, match="protocol"):
        script._reuse_initial_evaluation(source, **arguments)


def test_reuse_requires_checkpoint_reward_identity_and_run_directory(reusable_evaluation, tmp_path):
    source, _, arguments = reusable_evaluation
    unrelated = tmp_path / "copied.json"
    unrelated.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="warm-start run"):
        script._reuse_initial_evaluation(unrelated, **arguments)
    arguments["reward_sha256"] = "changed"
    with pytest.raises(ValueError, match="manifest"):
        script._reuse_initial_evaluation(source, **arguments)


@pytest.mark.parametrize("mismatch", [False, True])
def test_main_publishes_verified_baseline_before_collection_and_encoding(
        reusable_evaluation, reusable_modules, tmp_path, monkeypatch, mismatch):
    source, original, arguments = reusable_evaluation
    original["created_utc"] = "2026-10-01T01:02:03+00:00"
    if mismatch:
        original["agent_sha256"] = "different checkpoint"
    source.write_text(json.dumps(original))
    manifest_path = arguments["checkpoint"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = "fixture source archive hash"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "new_run"
    agent, reward = reusable_modules
    agent.vision = nn.Identity()
    agent.summary_query = torch.zeros(1)
    optimizer = torch.optim.AdamW(agent.parameters())
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: arguments["config"])
    monkeypatch.setattr(script, "provenance", lambda: {})
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k:
        (agent, optimizer, reward, {"decisions": 0, "weights_revision": 3,
                                   "config": arguments["config"].to_dict()}))
    monkeypatch.setattr(script, "evaluate_sequence", lambda *a, **k:
        pytest.fail("publishing an existing measurement must not evaluate again"))
    observed = []

    def observe(phase):
        observed.append(phase)
        status = json.loads((output / "status.json").read_text())
        assert status["phase"] == phase and status["optimizer_steps"] == status["epoch"] == 0
        assert json.loads((output / "metadata.json").read_text())["suite_version"] \
            == script.SUITE_VERSION
        reused = json.loads((output / "evaluations" / "epoch_0000.json").read_text())
        assert reused == script._reuse_initial_evaluation(source, **arguments)
        assert reused["created_utc"] == "2026-10-01T01:02:03+00:00"
        assert reused["initial_evaluation_reuse"]["source_created_utc"] == reused["created_utc"]
        assert reused["initial_evaluation_reuse"]["source_epoch"] == 8
        assert reused["records"] == original["records"]

    def collect(*args, **kwargs):
        observe("collecting")
        episode = SimpleNamespace(frame_indices=(0,), final_frame_index=0,
                                  source="scripted_teacher", presentation=None, native_size=None)
        return SimpleNamespace(episodes=[episode], frames=[np.zeros((4, 4, 3), dtype=np.uint8)],
                               provenance=lambda: {"source": "stub train-only RGB"})

    def encode(batch):
        observe("encoding")
        raise RuntimeError("stop after encoding observation")

    monkeypatch.setattr(script, "collect_scripted_demonstrations", collect)
    # This test isolates publication order with a lifecycle-only dataset stub.
    monkeypatch.setattr(script, "validate_dataset_runtime_holdout", lambda *a, **k: None)
    agent.encode_frames = encode
    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "unused.json",
        "--warm-start", str(arguments["checkpoint"]), "--output", str(output),
        "--training-backend", "torch", "--evaluation-difficulty", "1", "--final-all-levels",
        "--reuse-initial-evaluation", str(source)])
    if mismatch:
        with pytest.raises(ValueError, match="policy or protocol"):
            script.main()
        assert observed == [] and not (output / "evaluations").exists()
    else:
        with pytest.raises(RuntimeError, match="stop after encoding observation"):
            script.main()
        assert observed == ["collecting", "encoding"]
    assert json.loads(source.read_text()) == original
