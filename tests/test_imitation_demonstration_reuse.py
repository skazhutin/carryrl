"""Saved RGB replay preserves the entire cohort independently of the optimizer parent."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.model import AgentConfig
from carryrl.envs.atomic import _identity, atomic_level_banks
from carryrl.envs.atomic_suite import SUITE_VERSION
from carryrl.training.feature_store import FrozenFeatureWriter, feature_cache_identity
from carryrl.training.imitation import (
    SavedDemonstrationReuse,
    augment_demonstrations,
    collect_memory_cue_counterparts,
    collect_native_scripted_demonstrations,
    collect_scripted_corrections,
    collect_scripted_demonstrations,
    demonstration_episode_digest,
)
from carryrl.training.sequence_config import SequenceRunConfig

spec = importlib.util.spec_from_file_location(
    "saved_demonstration_script", Path(__file__).resolve().parents[1]
    / "scripts/train_imitation.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class Learner:
    training = False

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def reset_episode(self, **kwargs):
        pass

    def act(self, frame, deterministic):
        return SimpleNamespace(action=PhysicalAction(keys=("left",)))

    def observe_action(self, action):
        pass


def write_json(path, value):
    path.write_text(json.dumps(value))


def refresh_demonstration_hash(run):
    path = run / "feature_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["demonstrations_sha256"] = hashlib.sha256(
        (run / "demonstrations.json").read_bytes()).hexdigest()
    write_json(path, manifest)


@pytest.fixture
def corpus(tmp_path):
    run = tmp_path / "original"
    run.mkdir()
    config = SequenceRunConfig(agent=AgentConfig(pretrained=False,
                               text_encoder="multilingual_e5_small"),
                               tasks=("interaction", "memory"),
                               episode_instructions={"interaction": ("Carry the item",),
                                                     "memory": ("Remember the door",)},
                               max_episode_steps=80, level_bank_seed=20261008,
                               curriculum_difficulty=1., evaluation_difficulty=1.)
    options = dict(tasks=config.tasks, seed=8288, difficulty=1., max_steps=80,
                   level_bank_seed=config.level_bank_seed,
                   episode_instructions=config.episode_instructions)
    dataset = collect_scripted_demonstrations(1, **options)
    dataset.extend(collect_memory_cue_counterparts(dataset, max_pairs=1))
    dataset.extend(collect_scripted_corrections(Learner(), 1, expert_probability=0., **options))
    dataset.extend(collect_native_scripted_demonstrations(
        native_train_sizes=[64], native_episodes_per_task=1, **options))
    dataset.extend(augment_demonstrations(dataset, seed=123, copies_per_episode=1))
    np.savez_compressed(run / "demonstration_frames.npz", **{
        f"frame_{index:06}": frame for index, frame in enumerate(dataset.frames)})
    write_json(run / "demonstrations.json", dataset.provenance())
    write_json(run / "config.json", config.to_dict())
    write_json(run / "metadata.json", {
        "kind": "imitation_training", "suite_version": SUITE_VERSION,
        "config": config.to_dict(), "initial_agent_sha256": "1" * 64,
        "reward_sha256": "2" * 64, "source_checkpoint_sha256": "3" * 64,
        "source_checkpoint": "/original/producer/checkpoints/rollout_0000",
        "correction_replacement": {"producer_hashes": {"actor": "4" * 64}},
    })
    used = sorted({index for episode in dataset.episodes for index in episode.frame_indices})
    identity = feature_cache_identity(dataset.frames, used, vision_sha256="a" * 64,
        preprocessing={"implementation": "types.SimpleNamespace",
                       "preprocess": "letterbox_bicubic_antialias_imagenet_v1",
                       "image_size": 96, "patch_size": 96, "patch_grid": [1, 1]},
        feature_shape=(1, 384), dtype="float16")
    cache = tmp_path / "cache"
    with FrozenFeatureWriter(cache, identity, max_bytes=1024**2) as writer:
        writer.write(used, np.ones((len(used), 1, 384), np.float16))
        store = writer.publish()
    write_json(run / "feature_manifest.json", {
        "feature_cache_dir": str(store.directory.resolve()),
        "feature_cache_sha256": store.manifest["sha256"],
        "feature_cache_created_utc": store.manifest["created_utc"],
        "bytes": store.manifest["bytes"], "frames": len(used),
        "dtype": "float16", "vision_sha256": "a" * 64})
    refresh_demonstration_hash(run)
    return run, dataset, config, cache


def load(corpus, **changes):
    run, _, config, _ = corpus
    return SavedDemonstrationReuse.prepare(run, config=config,
                                           max_frame_bytes=256 * 1024**2, **changes)


def test_complete_mixed_corpus_roundtrip_and_readonly_source_references(corpus, tmp_path):
    run, original, _, _ = corpus
    reuse = load(corpus)
    assert [episode.to_dict() for episode in reuse.dataset.episodes] == [
        episode.to_dict() for episode in original.episodes]
    assert [demonstration_episode_digest(reuse.dataset, row) for row in reuse.dataset.episodes] == [
        demonstration_episode_digest(original, row) for row in original.episodes]
    assert reuse.dataset.frame_bytes == original.frame_bytes
    assert reuse.dataset.memory_cue_counterpart_reports == original.memory_cue_counterpart_reports
    assert any(row.runtime_level_override is not None for row in reuse.dataset.episodes)
    for actual, expected in zip(reuse.dataset.frames, original.frames, strict=True):
        np.testing.assert_array_equal(actual, expected)
        with pytest.raises(ValueError):
            actual.setflags(write=True)
    assert any(row.executed_actions != row.actions for row in reuse.dataset.episodes
               if row.executed_actions is not None)
    output = tmp_path / "continued"
    output.mkdir()
    reuse.link_corpus(output)
    for name in ("demonstrations.json", "demonstration_frames.npz"):
        assert (output / name).is_symlink()
        assert (output / name).resolve() == (run / name).resolve()
    reuse.assert_unchanged()
    assert reuse.summary["producer_lineage"]["initial_agent_sha256"] == "1" * 64
    assert reuse.summary["exact_complete_corpus"]


def test_unversioned_historical_episode_rows_remain_exact_and_legacy(corpus):
    run, _, _, _ = corpus
    path = run / "demonstrations.json"
    value = json.loads(path.read_text())
    value.pop("scoring_versions")
    for row in value["episodes"]:
        row.pop("scoring_version", None)
        row.pop("quality", None)
        row.pop("quality_source", None)
    write_json(path, value)
    refresh_demonstration_hash(run)
    reuse = load(corpus)
    assert [row.to_dict() for row in reuse.dataset.episodes] == value["episodes"]
    assert all(row.effective_scoring_version == "legacy.v0" for row in reuse.dataset.episodes)
    assert reuse.dataset.provenance() == value


@pytest.mark.parametrize("mutation", ["hash", "bounds", "history", "instruction", "split",
                                      "summary", "extra_field", "presentation", "config"])
def test_malformed_or_changed_saved_corpus_is_rejected(corpus, mutation):
    run, _, _, _ = corpus
    path = run / "demonstrations.json"
    value = json.loads(path.read_text())
    if mutation == "bounds":
        value["episodes"][0]["frame_indices"][0] = value["unique_frames"]
    elif mutation == "history":
        next(row for row in value["episodes"] if "executed_actions" in row)[
            "executed_actions"] = []
    elif mutation == "instruction":
        value["episodes"][0]["instruction"] = "Different goal"
    elif mutation == "split":
        value["episodes"][0]["split"] = "validation"
    elif mutation == "summary":
        value["frame_bytes"] -= 1
    elif mutation == "extra_field":
        value["episodes"][0]["hidden_reward"] = 1
    elif mutation == "presentation":
        next(row for row in value["episodes"] if "presentation" in row)[
            "presentation"]["theme"] = "unsupported"
    elif mutation == "config":
        metadata = json.loads((run / "metadata.json").read_text())
        metadata["config"]["max_episode_steps"] = 79
        write_json(run / "metadata.json", metadata)
    else:
        value["episodes"][0]["instruction"] = "Use the item"
    write_json(path, value)
    if mutation != "hash":
        refresh_demonstration_hash(run)
    with pytest.raises(ValueError):
        load(corpus)


def test_mutated_source_after_prepare_is_rejected(corpus, tmp_path):
    run, _, _, _ = corpus
    reuse = load(corpus)
    with (run / "demonstration_frames.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        reuse.assert_unchanged()
    output = tmp_path / "continued"
    output.mkdir()
    with pytest.raises(ValueError):
        reuse.link_corpus(output)
    assert not list(output.iterdir())


def test_changed_action_input_rgb_fails_original_feature_identity(corpus):
    run, dataset, _, _ = corpus
    frames = [frame.copy() for frame in dataset.frames]
    frames[0][40, 40, 0] ^= 1
    np.savez_compressed(run / "demonstration_frames.npz", **{
        f"frame_{index:06}": frame for index, frame in enumerate(frames)})
    with pytest.raises(ValueError, match="RGB identity"):
        load(corpus)


@pytest.mark.parametrize("mutation", ["dtype", "shape", "missing", "extra", "duplicate_rgb"])
def test_archive_layout_and_frame_index_integrity(corpus, mutation):
    run, dataset, _, _ = corpus
    frames = {f"frame_{index:06}": frame.copy() for index, frame in enumerate(dataset.frames)}
    if mutation == "dtype":
        frames["frame_000000"] = frames["frame_000000"].astype(np.float32)
    elif mutation == "shape":
        frames["frame_000000"] = frames["frame_000000"][:, :, :2]
    elif mutation == "missing":
        frames.pop("frame_000000")
    elif mutation == "extra":
        frames["unexpected"] = frames["frame_000000"]
    else:
        frames["frame_000001"] = frames["frame_000000"]
    np.savez_compressed(run / "demonstration_frames.npz", **frames)
    with pytest.raises(ValueError):
        load(corpus)


def test_source_file_symlink_escape_is_rejected(corpus, tmp_path):
    run, _, _, _ = corpus
    source = run / "demonstrations.json"
    external = tmp_path / "external.json"
    source.rename(external)
    source.symlink_to(external)
    with pytest.raises(ValueError, match="directly"):
        load(corpus)


@pytest.mark.parametrize("admitted", [False, True])
@pytest.mark.parametrize("inherited", [False, True])
def test_correction_bank_must_be_admitted_by_original_producer_lineage(
        corpus, admitted, inherited):
    run, _, _, _ = corpus
    path = run / "demonstrations.json"
    value = json.loads(path.read_text())
    for row in value["episodes"]:
        if row["source"] == "scripted_rgb_correction" and row["task"] == "interaction":
            row["level_bank_seed"] = 20261009
            row["level_id"] = _identity(atomic_level_banks(
                row["task"], 20261009)["train"][row["level_index"]])
    write_json(path, value)
    refresh_demonstration_hash(run)
    if admitted:
        path = run / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["correction_replacement"].update(
            actual_collection={"level_bank_seed": 20261009},
            selected_tasks=["interaction"],
            overlap_guard_passed=True, producer_frozen=True)
        if inherited:
            metadata["demonstration_reuse"] = {"producer_lineage": {
                "correction_replacement": metadata.pop("correction_replacement")}}
        write_json(path, metadata)
        assert load(corpus).summary["exact_complete_corpus"]
    else:
        with pytest.raises(ValueError, match="unexplained training bank"):
            load(corpus)


@pytest.mark.parametrize("field,value", [
    ("overlap_guard_passed", None), ("overlap_guard_passed", False),
    ("producer_frozen", None), ("producer_frozen", False),
])
def test_admitted_extra_bank_requires_original_frozen_overlap_proof(corpus, field, value):
    run, _, _, _ = corpus
    path = run / "demonstrations.json"
    source = json.loads(path.read_text())
    row = next(row for row in source["episodes"] if row["source"] == "scripted_rgb_correction")
    row["level_bank_seed"] = 20261009
    row["level_id"] = _identity(atomic_level_banks(
        row["task"], 20261009)["train"][row["level_index"]])
    write_json(path, source)
    refresh_demonstration_hash(run)
    path = run / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["correction_replacement"].update(
        actual_collection={"level_bank_seed": 20261009}, selected_tasks=[row["task"]],
        overlap_guard_passed=True, producer_frozen=True)
    metadata["correction_replacement"][field] = value
    write_json(path, metadata)
    with pytest.raises(ValueError, match="frozen overlap proof"):
        load(corpus)


@pytest.mark.parametrize("level_index", [28, 41, 42])
def test_extra_bank_rejects_current_heldout_semantic_collision_even_with_true_proof(
        corpus, level_index):
    run, dataset, config, _ = corpus
    source = json.loads((run / "demonstrations.json").read_text())
    row = next(row for row in source["episodes"] if row["source"] == "scripted_rgb_correction")
    row.update(task="ui", level_bank_seed=20261009, level_index=level_index,
               level_id=_identity(atomic_level_banks("ui", 20261009)["train"][level_index]),
               instruction="Use the item")
    row.pop("quality", None)
    row.pop("quality_source", None)
    write_json(run / "demonstrations.json", source)
    refresh_demonstration_hash(run)
    current = config.to_dict()
    current["tasks"] = [*current["tasks"], "ui"]
    current["episode_instructions"]["ui"] = ["Use the item"]
    write_json(run / "config.json", current)
    metadata = json.loads((run / "metadata.json").read_text())
    metadata["config"] = current
    metadata["correction_replacement"].update(
        actual_collection={"level_bank_seed": 20261009}, selected_tasks=["ui"],
        overlap_guard_passed=True, producer_frozen=True)
    write_json(run / "metadata.json", metadata)
    with pytest.raises(ValueError, match="overlaps held-out"):
        load((run, dataset, SequenceRunConfig(**current), corpus[3]))


@pytest.mark.parametrize("flags", [
    ["--difficulty", "1"], ["--episodes-per-task=16"],
    ["--correction-replacement-bundle", "bundle"], ["--correction-tasks", "interaction"],
    ["--native-train-sizes", "64"], ["--presentation-copies-per-episode", "0"],
])
def test_reuse_rejects_ambiguous_collection_flags_before_config_or_model(monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "unused",
        "--warm-start", "unused", "--output", "unused", "--demonstration-reuse-run", "unused",
        *flags])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2


@pytest.mark.parametrize("append_styled", [False, True])
def test_trainer_uses_exact_corpus_different_optimizer_parent_and_bounded_encoding(
        corpus, monkeypatch, tmp_path, append_styled):
    run, original, config, cache = corpus
    parent = tmp_path / "later_parent"
    parent.mkdir()
    write_json(parent / "manifest.json", {"sha256": "5" * 64})
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(script, "provenance", lambda: {})
    vision = SimpleNamespace(image_size=96, patch_size=96, patch_grid=(1, 1), hidden_size=384)
    agent = SimpleNamespace(vision=vision, summary_query=torch.zeros(1))
    reward = object()
    monkeypatch.setattr(script, "tensor_hash", lambda value:
        "a" * 64 if value is vision else "2" * 64 if value is reward else "6" * 64)
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k:
        (agent, SimpleNamespace(param_groups=[{}]), reward,
         {"decisions": 7, "weights_revision": 123, "config": config.to_dict()}))
    for name in ("collect_scripted_demonstrations", "collect_scripted_corrections",
                 "collect_native_scripted_demonstrations", "collect_memory_cue_counterparts",
                 "augment_demonstrations"):
        monkeypatch.setattr(script, name, lambda *a, **k: pytest.fail("reuse recollected corpus"))
    if append_styled:
        monkeypatch.setattr(script, "collect_scripted_corrections", lambda agent, count, **options:
            collect_scripted_corrections(Learner(), count, **options))
    monkeypatch.setattr(np, "savez_compressed", lambda *a, **k:
                        pytest.fail("reuse rewrote saved RGB archive"))
    agent.encode_frames = lambda *a, **k: pytest.fail("fully reused RGB reached encoder")
    if append_styled:
        agent.encode_frames = lambda frames: torch.zeros((len(frames), 1, 384))

    class Prepared(Exception):
        pass

    def stop_before_evaluation(*args, **kwargs):
        raise Prepared

    monkeypatch.setattr(script, "evaluate_sequence", stop_before_evaluation)
    output = tmp_path / "continued"
    flags = ["train_imitation.py", "--config", "unused",
        "--warm-start", str(parent), "--output", str(output), "--training-backend", "torch",
        "--demonstration-reuse-run", str(run), "--feature-reuse-run", str(run),
        "--feature-cache-dir", str(cache), "--feature-dtype", "float16"]
    if append_styled:
        flags += ["--styled-correction-episodes-per-task", "1"]
    monkeypatch.setattr(sys, "argv", flags)
    with pytest.raises(Prepared):
        script.main()
    assert (output / "demonstration_frames.npz").is_symlink() is not append_styled
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["source_checkpoint_sha256"] == "5" * 64
    assert metadata["initial_agent_sha256"] == "6" * 64
    assert metadata["config"]["curriculum_difficulty"] == 1.
    assert metadata["config"]["evaluation_difficulty"] == 1.
    reuse = metadata["demonstration_reuse"]
    assert reuse["producer_lineage"]["initial_agent_sha256"] == "1" * 64
    assert reuse["producer_lineage"]["correction_replacement"] == {
        "producer_hashes": {"actor": "4" * 64}}
    assert reuse["episodes"] == len(original.episodes)
    assert reuse["decisions"] == sum(map(len, original.episodes))
    features = json.loads((output / "feature_manifest.json").read_text())
    if append_styled:
        assert features["encoded_frames"] > 0
        assert features["feature_reuse"]["shard_pack"] is not None
        rows = json.loads((output / "demonstrations.json").read_text())["episodes"]
        assert rows[:len(original.episodes)] == [row.to_dict() for row in original.episodes]
        assert len(rows) == len(original.episodes) + len(config.tasks)
        assert all(row["collection_type"] == "live_styled" for row in rows[len(original.episodes):])
        assert reuse["appended_episodes"] == len(config.tasks)
        assert reuse["source_corpus_preserved"] and not reuse["collection_skipped"]
        load(corpus).assert_unchanged()
        with np.load(output / "demonstration_frames.npz", allow_pickle=False) as archive:
            for index, frame in enumerate(original.frames):
                np.testing.assert_array_equal(archive[f"frame_{index:06}"], frame)
        assert features["reused_frames"] == len({index for row in original.episodes
                                                 for index in row.frame_indices})
    else:
        assert features["encoded_frames"] == 0
        assert features["feature_reuse"]["shard_pack"] is None


def test_combined_archive_never_writes_through_existing_symlink(tmp_path):
    original = tmp_path / "original.npz"
    original.write_bytes(b"untouched archive")
    output = tmp_path / "output"
    output.mkdir()
    (output / "demonstration_frames.npz").symlink_to(original)
    with pytest.raises(FileExistsError, match="fresh output"):
        script._publish_combined_frames(output, [np.zeros((96, 96, 3), dtype=np.uint8)])
    assert original.read_bytes() == b"untouched archive"


def test_combined_archive_failure_does_not_publish_partial_files(tmp_path, monkeypatch):
    monkeypatch.setattr(script.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024))
    with pytest.raises(OSError, match="free disk floor"):
        script._publish_combined_frames(tmp_path, [np.zeros((96, 96, 3), dtype=np.uint8)])
    assert not list(tmp_path.iterdir())
