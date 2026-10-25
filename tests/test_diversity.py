"""Count verified corpus diversity without mistaking more views for experience."""

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.model import AgentConfig
from carryrl.envs.atomic_suite import SUITE_VERSION
from carryrl.envs.presentation import PresentationConfig
from carryrl.training.diversity import audit_diversity
from carryrl.training.feature_store import feature_cache_identity
from carryrl.training.imitation import (
    DemonstrationDataset,
    augment_demonstrations,
    collect_native_scripted_demonstrations,
    collect_scripted_corrections,
    collect_scripted_demonstrations,
)
from carryrl.training.sequence_config import SequenceRunConfig


def _canonical():
    return collect_scripted_demonstrations(1, tasks=("navigation",), seed=8176,
                                         level_bank_seed=20261008)


class Learner:
    training = False

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def reset_episode(self, **kwargs):
        pass

    def act(self, frame, deterministic):
        assert deterministic
        return SimpleNamespace(action=PhysicalAction())

    def observe_action(self, action):
        pass


def test_additional_views_and_instructions_do_not_create_physical_experience():
    dataset = _canonical()
    before = audit_diversity(dataset)
    dataset.extend(augment_demonstrations(dataset, copies_per_episode=2, seed=123))
    dataset.add_episode(replace(dataset.episodes[0], instruction="Reach the goal"))
    report = audit_diversity(dataset)
    assert report["counts"]["entries"] == 4
    assert report["counts"]["canonical_record_entries"] == 2
    assert report["counts"]["posthoc_appearance_copy_entries"] == 2
    assert report["counts"]["live_styled_rollout_entries"] == 0
    assert report["counts"]["metadata_physical_origins"] == 1
    assert report["counts"]["runtime_parameter_groups"] == before["counts"][
        "runtime_parameter_groups"] == 1
    assert len(report["instructions"]["navigation"]) == 2
    assert sum(cell["entries"] for cell in report["cross_cells"]) == 4
    assert report["counts"]["rgb_payload_bytes"] == sum(f.nbytes for f in dataset.frames)
    assert report["updates"] == 0 and report["test_used"] is False
    assert json.loads(json.dumps(report))["counts"] == report["counts"]


def test_plain_texture_seed_changes_are_not_new_effective_appearances():
    dataset = _canonical()
    options = {"themes": ("slate",), "hud_layouts": ("top",), "textures": False}
    first = augment_demonstrations(dataset, seed=1, **options)
    second = augment_demonstrations(dataset, seed=2, **options)
    assert first.episodes[0].presentation != second.episodes[0].presentation
    assert len(first.frames) == len(second.frames)
    for left, right in zip(first.frames, second.frames, strict=True):
        np.testing.assert_array_equal(left, right)
    dataset.extend(first)
    bytes_before = dataset.frame_bytes
    dataset.extend(second)
    report = audit_diversity(dataset)
    assert dataset.frame_bytes == bytes_before
    assert report["counts"]["declared_profile_configs"] == 3
    assert report["counts"]["effective_profile_configs"] == 2
    assert report["counts"]["metadata_physical_origins"] == 1


def test_native_actual_rollouts_and_copies_keep_distinct_runtime_size_groups():
    dataset = collect_native_scripted_demonstrations(native_train_sizes=(64, 128),
        native_episodes_per_task=1, tasks=("navigation",), seed=8176,
        level_bank_seed=20261008)
    dataset.extend(augment_demonstrations(dataset, seed=123))
    report = audit_diversity(dataset)
    assert report["counts"]["metadata_physical_origins"] == 2
    assert report["counts"]["runtime_parameter_groups"] == 2
    assert {(row["native_size"], row["collection_kind"], row["entries"])
            for row in report["native_sizes"]} == {
        (size, kind, 1) for size in (64, 128)
        for kind in ("canonical", "posthoc_appearance_copy")}


def test_actual_correction_context_and_live_styled_provenance_are_separate():
    options = dict(tasks=("navigation",), seed=8176, level_bank_seed=20261008,
                   learner_prefix_steps=4, expert_probability=1, max_steps=20)
    canonical = collect_scripted_corrections(Learner(), 1, **options)
    original = canonical.episodes[0]
    assert original.actions != original.executed_actions
    canonical.extend(augment_demonstrations(canonical))
    canonical.add_episode(replace(original,
        actions=(PhysicalAction(),) * len(original), instruction="A different phrasing"))
    report = audit_diversity(canonical)
    assert report["counts"]["metadata_physical_origins"] == 1
    assert report["counts"]["history_from_executed_actions_entries"] == 3
    assert report["counts"]["history_from_teacher_actions_entries"] == 0
    assert report["counts"]["live_styled_rollout_entries"] == 0
    styled = collect_scripted_corrections(Learner(), 1, **options,
        presentation_config=PresentationConfig(mode="varied", native_sizes=(96,),
            themes=("slate",), hud_layouts=("left",), style_split="train", seed=123))
    report = audit_diversity(styled)
    assert report["counts"]["live_styled_rollout_entries"] == 1
    assert report["counts"]["posthoc_appearance_copy_entries"] == 0
    assert styled.episodes[0].collection_type == "live_styled"


def test_audit_revalidates_episode_and_actual_byte_accounting():
    dataset = _canonical()
    valid = dataset.episodes[0]
    object.__setattr__(valid, "final_frame_index", len(dataset.frames))
    with pytest.raises(ValueError, match="missing RGB frame"):
        audit_diversity(dataset)
    dataset = _canonical()
    dataset.frame_bytes += 1
    with pytest.raises(ValueError, match="byte accounting"):
        audit_diversity(dataset)
    with pytest.raises(TypeError, match="DemonstrationDataset"):
        audit_diversity(SimpleNamespace(episodes=[]))


def _saved_corpus(directory):
    directory.mkdir()
    dataset = _canonical()
    config = SequenceRunConfig(agent=AgentConfig(pretrained=False), tasks=("navigation",),
                               level_bank_seed=20261008)
    np.savez_compressed(directory / "demonstration_frames.npz", **{
        f"frame_{index:06}": frame for index, frame in enumerate(dataset.frames)})
    records = {"demonstrations.json": dataset.provenance(), "config.json": config.to_dict(),
               "metadata.json": {"kind": "imitation_training", "suite_version": SUITE_VERSION,
                   "config": config.to_dict(), "initial_agent_sha256": "1" * 64,
                   "reward_sha256": "2" * 64, "source_checkpoint_sha256": "3" * 64}}
    for name, value in records.items():
        (directory / name).write_text(json.dumps(value))
    used = sorted({index for row in dataset.episodes for index in row.frame_indices})
    identity = feature_cache_identity(dataset.frames, used, vision_sha256="a" * 64,
        preprocessing={"fixture": "RGB identity; no encoder allocated"},
        feature_shape=(1, 384), dtype="float16")
    (directory / "feature_manifest.json").write_text(json.dumps({
        "feature_reuse": {"identity": identity}, "frames": len(used),
        "vision_sha256": identity["vision_sha256"], "dtype": identity["dtype"],
        "demonstrations_sha256": hashlib.sha256(
            (directory / "demonstrations.json").read_bytes()).hexdigest()}))
    return dataset


def test_cli_authenticates_complete_saved_corpus_without_mutation_or_checkpoint(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/audit_diversity.py"
    spec = importlib.util.spec_from_file_location("audit_diversity_cli", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    source, output = tmp_path / "source", tmp_path / "audit"
    dataset = _saved_corpus(source)
    before = {file.name: file.read_bytes() for file in source.iterdir()}
    assert script.main(["--source-run", str(source), "--output", str(output),
                        "--frame-budget-mib", "16"]) == 0
    report = json.loads((output / "audit.json").read_text())
    assert report["counts"]["entries"] == len(dataset.episodes)
    assert report["validation"]["evaluation_endpoint_guard"] is True
    assert report["source_corpus_identity"]["exact_complete_corpus"] is True
    for name, data in before.items():
        identity = report["source_corpus_identity"]["files"][name]
        assert identity["sha256"] == hashlib.sha256(data).hexdigest()
    assert before == {file.name: file.read_bytes() for file in source.iterdir()}
    assert {file.name for file in output.iterdir()} == {"audit.json", "status.json"}
    with pytest.raises(SystemExit):
        script.main(["--source-run", str(source), "--output", str(source / "audit")])
    assert not (source / "audit").exists()


def test_empty_dataset_has_no_invented_grid_or_diversity():
    report = audit_diversity(DemonstrationDataset())
    assert report["cross_cells"] == []
    assert report["counts"]["entries"] == report["counts"]["metadata_physical_origins"] == 0
