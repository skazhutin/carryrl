"""Explicit stage reset discards optimizer moments once while preserving learned state."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_sequence_training import TinyReward, config, factory

from carryrl.training.sequence import load_sequence_checkpoint, train_sequence
from carryrl.training.sequence_config import SequenceRunConfig


def test_warm_start_reset_preserves_parent_tensors_and_groups_but_clears_moments_once(tmp_path):
    cfg = config()
    parent = train_sequence(cfg, tmp_path / "parent", agent_factory=factory,
                            reward=TinyReward(), reward_factory=TinyReward)
    checkpoint = Path(parent["latest_checkpoint"])
    archive = checkpoint / "state.pt"
    original_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    parent_state = torch.load(archive, weights_only=True)
    assert parent_state["optimizer_state"]["state"]
    changed = replace(cfg, reset_optimizer_on_warm_start=True, total_rollouts=2,
                      checkpoint_every_rollouts=2, learning_rate=.0003, weight_decay=.03)
    output = tmp_path / "reset"
    child = train_sequence(changed, output, warm_start=checkpoint,
                           agent_factory=factory, reward_factory=TinyReward)
    initial = torch.load(output / "checkpoints/rollout_0000/state.pt", weights_only=True)
    for section in ("agent_state", "reward_state"):
        assert all(torch.equal(value, initial[section][name])
                   for name, value in parent_state[section].items())
    assert initial["optimizer_state"]["state"] == {}
    for before, after in zip(parent_state["optimizer_state"]["param_groups"],
                             initial["optimizer_state"]["param_groups"], strict=True):
        assert before["params"] == after["params"]
        assert after["lr"] == .0003 and after["weight_decay"] == .03
    final = torch.load(Path(child["latest_checkpoint"]) / "state.pt", weights_only=True)
    assert final["optimizer_state"]["state"]
    assert all(state["step"].item() == 2
               for state in final["optimizer_state"]["state"].values())
    assert final["weights_revision"] == parent_state["weights_revision"] + 2
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == original_sha
    lineage = json.loads((output / "metadata.json").read_text())["warm_start_parent"]
    assert lineage["optimizer_state_preserved"] is False
    assert lineage["optimizer_reset_reason"]
    assert lineage["source_optimizer"]["state_entries"] == len(
        parent_state["optimizer_state"]["state"])
    assert lineage["source_optimizer"]["checkpoint_sha256"] == original_sha
    # Reset is an entry action, so later continuation must preserve these new moments.
    resume_config = replace(changed, reset_optimizer_on_warm_start=False, total_rollouts=3)
    _, resumed_optimizer, _, _ = load_sequence_checkpoint(
        Path(child["latest_checkpoint"]), resume_config,
        agent_factory=factory, reward_factory=TinyReward)
    assert all(state["step"].item() == 2 for state in resumed_optimizer.state.values())


@pytest.mark.parametrize("resume", [None, Path("unused_checkpoint")])
def test_reset_without_warm_start_is_rejected_before_creating_run(tmp_path, resume):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="requires warm_start"):
        train_sequence(replace(config(), reset_optimizer_on_warm_start=True), output,
                       resume=resume, agent_factory=factory, reward=TinyReward())
    assert not output.exists()


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_optimizer_reset_flag_is_strict_boolean(value):
    with pytest.raises(ValueError, match="reset_optimizer_on_warm_start"):
        SequenceRunConfig(reset_optimizer_on_warm_start=value)
