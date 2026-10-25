"""Frozen data collection verifies its round-trip and never trains or overwrites."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from carryrl.agent.actions import PhysicalAction
from carryrl.training.imitation import CORRECTION_SOURCE, load_correction_replacement_bundle
from carryrl.training.sequence import tensor_hash
from carryrl.training.sequence_config import SequenceRunConfig

spec = importlib.util.spec_from_file_location(
    "collect_corrections_script", Path(__file__).parents[1] / "scripts/collect_corrections.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class TinyLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor([1.0]))

    def reset_episode(self, **kwargs):
        pass

    def act(self, frame, deterministic):
        assert deterministic and not torch.is_grad_enabled() and not self.training
        return SimpleNamespace(action=PhysicalAction(keys=("left",)))

    def observe_action(self, action):
        pass


@pytest.fixture
def frozen_collection(tmp_path, monkeypatch):
    checkpoint = tmp_path / "source/checkpoints_by_epoch/epoch_0002/checkpoints/rollout_0000"
    checkpoint.mkdir(parents=True)
    agent, reward = TinyLearner(), nn.Linear(1, 1)
    manifest = {"sha256": "3" * 64, "agent_sha256": tensor_hash(agent),
                "reward_sha256": tensor_hash(reward), "rollout": 0, "weights_revision": 17}
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    (checkpoint / "state.pt").write_bytes(b"source checkpoint unchanged")
    config = SequenceRunConfig(tasks=("interaction",), torch_threads=1)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    output = tmp_path / "bundle"
    flags = []

    def load(path, loaded_config, **kwargs):
        flags.append(kwargs)
        assert path == checkpoint and loaded_config.agent == config.agent
        assert kwargs == {"evaluation_only": True, "load_optimizer": False}
        return agent, None, reward, {"weights_revision": 17}

    monkeypatch.setattr(script, "load_sequence_checkpoint", load)
    monkeypatch.setattr(script, "provenance", lambda: {})
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **kw: pytest.fail("no optimizer"))
    args = ["--checkpoint", str(checkpoint), "--config", str(config_path),
            "--output", str(output), "--tasks", "interaction", "--episodes-per-task", "1",
            "--fresh-train-bank-seed", "20261015"]
    return SimpleNamespace(checkpoint=checkpoint, config=config, config_path=config_path,
        output=output, agent=agent, reward=reward, args=args, flags=flags, manifest=manifest)


def test_public_import_preserves_failed_corrections_quality_and_provenance(frozen_collection):
    case = frozen_collection
    before = script._checkpoint_hashes(case.checkpoint)
    hashes = tensor_hash(case.agent), tensor_hash(case.reward)
    threads = torch.get_num_threads()
    assert case.agent.training and case.reward.training
    assert script.main(case.args) == 0
    assert case.flags == [{"evaluation_only": True, "load_optimizer": False}]
    assert script._checkpoint_hashes(case.checkpoint) == before
    assert hashes == (tensor_hash(case.agent), tensor_hash(case.reward))
    assert case.agent.training and case.reward.training and torch.get_num_threads() == threads
    metadata = json.loads((case.output / "metadata.json").read_text())
    status = json.loads((case.output / "status.json").read_text())
    assert status["status"] == "completed" and status["optimizer_steps"] == 0
    assert metadata["weights_revision_before"] == metadata["weights_revision_after"] == 17
    assert metadata["checkpoint_files_sha256_before"] == before
    assert metadata["checkpoint_files_sha256_after"] == before
    novelty = metadata["train_level_novelty"]
    assert novelty["selected_episodes"] == novelty["distinct_selected_runtime_levels"] == 1
    assert novelty["new_runtime_episodes_vs_original_train"] == 1
    assert novelty["repeated_original_train_episodes"] == 0
    assert metadata["serialization_verified"]
    assert metadata["replacement_import"]["validated"]
    assert metadata["replacement_import"]["corpus_slot_replacement_validated"] is False
    assert metadata["settings"]["expert_probability"] == 0
    assert metadata["settings"]["learner_prefix_steps"] == 0
    assert metadata["settings"]["difficulty"] == 1
    assert metadata["settings"]["level_bank_seed"] == 20261015
    digest, files = script._bundle_identity(case.output)
    assert digest == metadata["bundle_sha256"] == status["bundle_sha256"]
    assert files == metadata["bundle_files"]
    imported, report = load_correction_replacement_bundle(case.output,
        expected_sha256=digest, correction_tasks=("interaction",), episodes_per_task=1,
        seed=8176, difficulty=1., max_steps=80, level_bank_seed=0, episode_instructions={},
        source_identity=case.manifest, initial_agent_sha256=hashes[0], reward_sha256=hashes[1])
    assert report["supporting_teacher_episodes_excluded"] == 1
    episode = imported["interaction"].episodes[0]
    assert episode.source == CORRECTION_SOURCE and episode.truncated
    assert episode.outcome["success"] is False
    assert episode.executed_actions != episode.actions
    assert episode.quality_source == "actual_executed_trajectory"
    assert episode.quality["commands"] == len(episode.executed_actions)
    assert episode.quality["game_seconds"] > 0
    corpus = json.loads((case.output / "trajectories.json").read_text())
    assert len(corpus["episodes"]) == 2
    expected_row = [row for row in corpus["episodes"] if row["source"] == CORRECTION_SOURCE][0]
    assert episode.to_dict() == expected_row
    frames = np.load(case.output / "frames.npy", mmap_mode="r", allow_pickle=False)
    assert len(frames) == corpus["frame_count"]
    assert not (case.output / "demonstrations.json").exists()
    assert not (case.output / "demonstration_frames.npz").exists()
    assert metadata["rgb_bytes"] <= 256 * 1024**2
    for name, size in metadata["disk_artifact_bytes"].items():
        assert (case.output / name).stat().st_size == size


def test_index_pattern_fails_before_load_or_output(frozen_collection, monkeypatch):
    case = frozen_collection
    monkeypatch.setattr(script, "allowed_runtime_train_indices", lambda *a: list(range(1, 64)))
    with pytest.raises(ValueError, match="differ.*schedule"):
        script.main(case.args)
    assert case.flags == [] and not case.output.exists()


def test_heldout_collision_fails_before_load_or_output(frozen_collection, monkeypatch):
    case = frozen_collection
    monkeypatch.setattr(script, "runtime_heldout_collisions", lambda *a, **kw: [{}])
    with pytest.raises(ValueError, match="overlaps held-out"):
        script.main(case.args)
    assert case.flags == [] and not case.output.exists()


def test_existing_and_source_outputs_are_protected(frozen_collection):
    case = frozen_collection
    case.output.mkdir()
    sentinel = case.output / "user-owned"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        script.main(case.args)
    assert sentinel.read_text() == "preserve" and case.flags == []
    source = case.checkpoint.parents[3]
    for output in (source, source / "fresh", case.checkpoint / "fresh", source.parent):
        with pytest.raises(ValueError, match="source run/checkpoint"):
            script._prepare_output(output, case.checkpoint)


@pytest.mark.parametrize("extra", [
    ["--expert-probability", "0.75"], ["--learner-prefix-steps", "4"],
    ["--fresh-train-bank-seed", "0"], ["--fresh-train-bank-seed", "20261008"],
    ["--tasks", "camera"], ["--episodes-per-task", "0"], ["--max-frame-mib", "257"],
])
def test_invalid_choices_fail_before_load(frozen_collection, extra):
    case = frozen_collection
    with pytest.raises(ValueError):
        script.main([*case.args, *extra])
    assert case.flags == [] and not case.output.exists()


def test_corrupt_serialization_is_failed_not_completed(frozen_collection, monkeypatch):
    case = frozen_collection
    writer = script._write_frames

    def corrupt(output, dataset):
        writer(output, dataset)
        frames = np.lib.format.open_memmap(output / "frames.npy", mode="r+")
        frames[0, 0, 0, 0] ^= 1
        frames.flush()

    monkeypatch.setattr(script, "_write_frames", corrupt)
    with pytest.raises(ValueError, match="exported RGB differs"):
        script.main(case.args)
    status = json.loads((case.output / "status.json").read_text())
    assert status["status"] == "failed"
    assert not (case.output / "coverage.json").exists()
    assert case.agent.training and case.reward.training


def test_public_import_rejection_cannot_publish_success(frozen_collection, monkeypatch):
    case = frozen_collection

    def reject(*a, **kw):
        raise ValueError("import failed")

    monkeypatch.setattr(script, "load_correction_replacement_bundle", reject)
    with pytest.raises(ValueError, match="import failed"):
        script.main(case.args)
    assert json.loads((case.output / "status.json").read_text())["status"] == "failed"
    assert case.agent.training and case.reward.training


def test_broken_output_symlink_is_existing_user_owned_path(frozen_collection):
    case = frozen_collection
    target = case.output.parent / "missing-target"
    case.output.symlink_to(target)
    with pytest.raises(FileExistsError, match="existing symlink"):
        script.main(case.args)
    assert case.output.is_symlink() and not target.exists() and case.flags == []


def test_memory_new_bank_reports_repeated_original_train_mechanics():
    config = SequenceRunConfig(tasks=("memory",), level_bank_seed=20261008)
    selected = script._preflight(config, ("memory",), 24, 8176, 20261015, (0, 20261008))
    repeated = next(row for row in selected if row["level_index"] == 1)
    assert repeated["runtime_level_id"]
    assert {"level_bank_seed": 20261008, "level_index": 61} in (
        repeated["original_train_matches"])
    summary = script._train_novelty(selected)
    assert summary["repeated_original_train_episodes"] >= 1
    assert summary["distinct_repeated_original_train_runtime_levels"] >= 1
    assert summary["distinct_selected_runtime_levels"] == summary["selected_episodes"] == 24
    assert summary["distinct_new_runtime_levels_vs_original_train"] < 24
    assert summary["new_runtime_episodes_vs_original_train"] == (
        24 - summary["repeated_original_train_episodes"])
    assert "new bank seed does not imply novel mechanics" in summary["interpretation"]
