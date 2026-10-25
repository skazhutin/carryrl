"""Pinned correction replacement admits complete train RGB and own input histories only."""

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.model import AgentConfig
from carryrl.training.feature_store import (
    FrozenFeatureReuse,
    FrozenFeatureWriter,
    feature_cache_identity,
)
from carryrl.training.imitation import (
    CORRECTION_SOURCE,
    DemonstrationDataset,
    audit_correction_replacement,
    augment_demonstrations,
    collect_scripted_corrections,
    collect_scripted_demonstrations,
    demonstration_episode_digest,
    load_correction_replacement_bundle,
)
from carryrl.training.sequence_config import SequenceRunConfig

spec = importlib.util.spec_from_file_location(
    "correction_replacement_script", Path(__file__).resolve().parents[1]
    / "scripts/train_imitation.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

ACTOR_HASH, REWARD_HASH = "1" * 64, "2" * 64
SOURCE_IDENTITY = {"sha256": "3" * 64, "agent_sha256": ACTOR_HASH,
                   "reward_sha256": REWARD_HASH, "rollout": 0, "weights_revision": 13053}
INSTRUCTIONS = {"interaction": ("Carry the item", "Use the item"),
                "key_door": ("Find the key", "Open the door")}


class Learner:
    training = False

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode

    def reset_episode(self, **kwargs):
        pass

    def act(self, frame, deterministic):
        return SimpleNamespace(action=PhysicalAction(keys=("left",)))

    def observe_action(self, action):
        pass


def json_write(path, value):
    path.write_text(json.dumps(value))


def bundle_sha(directory):
    files = {}
    for name in ("coverage.json", "frames.npy", "trajectories.json"):
        raw = (directory / name).read_bytes()
        files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return hashlib.sha256(json.dumps(files, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    options = dict(tasks=("interaction",), seed=13288, difficulty=1., max_steps=80,
                   level_bank_seed=20261009, episode_instructions=INSTRUCTIONS)
    original = collect_scripted_demonstrations(2, **options)
    original.extend(collect_scripted_corrections(Learner(), 2, expert_probability=0., **options))
    np.save(directory / "frames.npy", np.stack(original.frames))
    json_write(directory / "trajectories.json", {
        "frame_array": "frames.npy", "frame_shape": [96, 96, 3],
        "frame_count": len(original.frames), "raw_frame_bytes": original.frame_bytes,
        "episodes": [episode.to_dict() for episode in original.episodes]})
    json_write(directory / "coverage.json", {
        "status": "complete", "episodes": [{}, {}],
        "overlap_guard": {"passed": True, "collisions": [], "selected": [{}, {}],
                          "original_bank_seeds": [0, 20261008]},
        "loaded_tensors_match_preregistered_identity": True, "policy_and_reward_frozen": True,
        "frozen_hashes_before": {"actor": ACTOR_HASH, "reward": REWARD_HASH},
        "frozen_hashes_after": {"actor": ACTOR_HASH, "reward": REWARD_HASH},
        "preregistration": {
            "expected_checkpoint_identity": SOURCE_IDENTITY,
            "checkpoint": "/test/source40/checkpoints/rollout_0000",
            "fresh_train_bank_seed": 20261009, "correction_collection_seed": 13288,
            "episodes_per_task": 2, "max_steps_per_episode": 80,
            "difficulty": 1., "native_rgb_size": 96, "expert_probability": 0.,
            "learner_prefix_steps": 0, "training_updates": 0, "optimizer_steps": 0,
            "heldout_policy_rollouts": 0, "test_outcomes_read": False, "tasks": ["interaction"],
        }})
    return directory, original


def load(directory, **overrides):
    options = dict(expected_sha256=bundle_sha(directory),
                   correction_tasks=("interaction", "key_door"), episodes_per_task=2, seed=13288,
                   difficulty=1., max_steps=80, level_bank_seed=20261008,
                   episode_instructions=INSTRUCTIONS, source_identity=SOURCE_IDENTITY,
                   initial_agent_sha256=ACTOR_HASH, reward_sha256=REWARD_HASH)
    options.update(overrides)
    return load_correction_replacement_bundle(directory, **options)


def test_import_excludes_teachers_and_preserves_content_histories_and_truncations(bundle):
    directory, original = bundle
    before = bundle_sha(directory)
    grouped, summary = load(directory)
    assert list(grouped) == ["interaction"]
    imported = grouped["interaction"]
    assert len(imported.episodes) == 2
    expected = [episode for episode in original.episodes if episode.source == CORRECTION_SOURCE]
    for left, right in zip(expected, imported.episodes, strict=True):
        assert demonstration_episode_digest(original, left) == demonstration_episode_digest(
            imported, right)
        assert left.executed_actions == right.executed_actions
        assert right.source == CORRECTION_SOURCE and right.level_bank_seed == 20261009
        assert right.terminated == left.terminated and right.truncated == left.truncated
    assert any(episode.truncated for episode in imported.episodes)
    assert summary["replacement_episodes"] == summary["supporting_teacher_episodes_excluded"] == 2
    assert summary["producer_hashes"] == {"actor": ACTOR_HASH, "reward": REWARD_HASH}
    assert summary["rgb_teacher_and_executed_history_replay_validated"]
    assert summary["files"]["frames.npy"]["sha256"] == hashlib.sha256(
        (directory / "frames.npy").read_bytes()).hexdigest()
    assert summary["referenced_frame_sha256"]
    assert bundle_sha(directory) == before


def test_content_digest_ignores_storage_indices_but_preserves_history_and_instruction(bundle):
    _, original = bundle
    episode = original.episodes[-1]
    imported = DemonstrationDataset()
    imported.intern_frame(np.zeros((96, 96, 3), dtype=np.uint8))
    imported.extend(original)
    remapped = imported.episodes[-1]
    assert episode.frame_indices != remapped.frame_indices
    assert demonstration_episode_digest(original, episode) == demonstration_episode_digest(
        imported, remapped)
    changed = replace(remapped, instruction="Different instruction")
    assert demonstration_episode_digest(imported, changed) != demonstration_episode_digest(
        imported, remapped)
    changed = replace(remapped, executed_actions=(PhysicalAction(),) * len(remapped),
                      quality=None, quality_source=None)
    assert demonstration_episode_digest(imported, changed) != demonstration_episode_digest(
        imported, remapped)


@pytest.mark.parametrize("filename", ["coverage.json", "trajectories.json", "frames.npy"])
def test_pinned_digest_rejects_tampering_before_any_replay(bundle, filename, monkeypatch):
    directory, _ = bundle
    trusted = bundle_sha(directory)
    path = directory / filename
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    monkeypatch.setattr("carryrl.training.imitation.AtomicEnv",
                        lambda *a, **k: pytest.fail("tampered bundle reached environment replay"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load(directory, expected_sha256=trusted)


@pytest.mark.parametrize("field,value", [
    ("source", "human"), ("split", "test"), ("level_bank_seed", 0),
    ("seed", 13289), ("difficulty", 0.), ("instruction", "Wrong goal"),
    ("expert_probability", .35), ("executed_actions", None),
    ("terminated", 1), ("frame_indices", [999999]),
])
def test_malformed_correction_provenance_is_rejected(bundle, field, value):
    directory, _ = bundle
    path = directory / "trajectories.json"
    data = json.loads(path.read_text())
    data["episodes"][2][field] = value
    json_write(path, data)
    with pytest.raises(ValueError):
        load(directory)


@pytest.mark.parametrize("mutation", ["count", "duplicate", "outcome", "history", "label"])
def test_count_final_outcome_and_actual_trace_validation_are_not_bypassed(bundle, mutation):
    directory, _ = bundle
    path = directory / "trajectories.json"
    data = json.loads(path.read_text())
    if mutation == "count":
        data["episodes"].pop()
    elif mutation == "duplicate":
        data["episodes"][3] = deepcopy(data["episodes"][2])
    elif mutation == "outcome":
        data["episodes"][2]["outcome"]["success"] = True
    elif mutation == "history":
        data["episodes"][2]["executed_actions"][0] = PhysicalAction(keys=("right",)).to_dict()
    else:
        data["episodes"][2]["actions"][0] = PhysicalAction(keys=("space",)).to_dict()
    json_write(path, data)
    with pytest.raises(ValueError):
        load(directory)


@pytest.mark.parametrize("mutation", ["status", "overlap", "actor", "after", "declared_count"])
def test_bundle_completion_overlap_and_producer_proofs_are_required(bundle, mutation):
    directory, _ = bundle
    path = directory / "coverage.json"
    data = json.loads(path.read_text())
    if mutation == "status":
        data["status"] = "failed"
    elif mutation == "overlap":
        data["overlap_guard"]["passed"] = False
    elif mutation == "actor":
        data["preregistration"]["expected_checkpoint_identity"]["agent_sha256"] = "4" * 64
    elif mutation == "after":
        data["frozen_hashes_after"]["actor"] = "4" * 64
    else:
        data["preregistration"]["episodes_per_task"] = 1
    json_write(path, data)
    with pytest.raises(ValueError):
        load(directory)


@pytest.mark.parametrize("dtype,shape", [(np.float32, (96, 96, 3)),
                                         (np.uint8, (64, 64, 3)), (object, (96, 96, 3))])
def test_frame_dtype_shape_and_nonpickle_requirement(bundle, dtype, shape):
    directory, _ = bundle
    np.save(directory / "frames.npy", np.zeros((1, *shape), dtype=dtype))
    with pytest.raises(ValueError):
        load(directory)


@pytest.mark.parametrize("options", [
    {"max_frame_bytes": 1}, {"max_json_bytes": 1}, {"episodes_per_task": 1},
    {"correction_tasks": ("key_door",)}, {"difficulty": .5},
    {"initial_agent_sha256": "5" * 64}, {"reward_sha256": "6" * 64},
])
def test_budgets_current_counts_tasks_difficulty_and_warmstart_identity(bundle, options):
    with pytest.raises(ValueError):
        load(bundle[0], **options)


@pytest.mark.parametrize("flags", [
    ["--correction-replacement-bundle", "unused"],
    ["--correction-replacement-sha256", "1" * 64],
    ["--correction-replacement-bundle", "unused", "--correction-replacement-sha256", "1" * 64],
    ["--feature-reuse-run", "unused"],
    ["--replacement-control-run", "unused"],
])
def test_cli_requires_complete_optin_configuration_before_loading(monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "missing.json",
        "--warm-start", "unused", "--output", "unused", *flags])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2


class BeforeEncoding(Exception):
    pass


def stub_script(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    json_write(source / "manifest.json", SOURCE_IDENTITY)
    config = SequenceRunConfig(agent=AgentConfig(pretrained=False,
                               text_encoder="multilingual_e5_small"),
                               tasks=("interaction", "key_door"), level_bank_seed=20261008,
                               episode_instructions=INSTRUCTIONS, max_episode_steps=80)
    reward = object()

    def encode(_):
        raise BeforeEncoding

    agent = SimpleNamespace(vision=object(), summary_query=torch.zeros(1), encode_frames=encode)
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k:
        (agent, SimpleNamespace(param_groups=[{}]), reward, {"decisions": 0,
                                                           "weights_revision": 13053,
                                                           "config": config.to_dict()}))
    monkeypatch.setattr(script, "tensor_hash", lambda value:
        REWARD_HASH if value is reward else ACTOR_HASH)
    monkeypatch.setattr(script, "provenance", lambda: {})
    return source, config


@pytest.mark.parametrize("replace_selected", [False, True])
def test_trainer_replaces_only_selected_correction_slots_without_teacher_extras(
        bundle, monkeypatch, tmp_path, replace_selected):
    directory, _ = bundle
    source, config = stub_script(monkeypatch, tmp_path)
    output = tmp_path / "run"
    collected = []

    def corrections(agent, count, **options):
        collected.append(options["tasks"][0])
        return collect_scripted_corrections(Learner(), count, **options)

    monkeypatch.setattr(script, "collect_scripted_corrections", corrections)
    flags = ["train_imitation.py", "--config", "unused.json", "--warm-start", str(source),
             "--output", str(output), "--training-backend", "torch", "--difficulty", "1",
             "--dataset-seed", "8288", "--episodes-per-task", "1",
             "--correction-episodes-per-task", "2", "--expert-probability", ".35",
             "--learner-prefix-steps", "1"]
    if replace_selected:
        flags += ["--correction-replacement-bundle", str(directory),
                  "--correction-replacement-sha256", bundle_sha(directory)]
    monkeypatch.setattr(sys, "argv", flags)
    with pytest.raises(BeforeEncoding):
        script.main()
    episodes = json.loads((output / "demonstrations.json").read_text())["episodes"]
    assert len(episodes) == 6
    assert collected == (["key_door"] if replace_selected else ["interaction", "key_door"])
    correction_rows = [row for row in episodes if row["source"] == CORRECTION_SOURCE]
    assert len(correction_rows) == 4
    assert [row["task"] for row in episodes] == [
        "interaction", "key_door", "interaction", "interaction", "key_door", "key_door"]
    for row in correction_rows:
        replaced = replace_selected and row["task"] == "interaction"
        assert row["level_bank_seed"] == (20261009 if replaced else config.level_bank_seed)
        assert row["expert_probability"] == (0. if replaced else .35)
    metadata = json.loads((output / "metadata.json").read_text())
    if replace_selected:
        replacement = metadata["correction_replacement"]
        assert replacement["supporting_teacher_episodes_excluded"] == 2
        assert replacement["original_control_collection"]["expert_probability"] == .35
        assert replacement["actual_collection"]["expert_probability"] == 0.
    else:
        assert "correction_replacement" not in metadata


def audit_cohort(bundle, tmp_path, *, copies):
    replacements, summary = load(bundle[0])
    options = dict(tasks=("interaction",), seed=13288, difficulty=1., max_steps=80,
                   level_bank_seed=20261008, episode_instructions=INSTRUCTIONS)
    teachers = collect_scripted_demonstrations(1, **options)
    control, candidate = DemonstrationDataset(), DemonstrationDataset()
    control.extend(teachers)
    candidate.extend(teachers)
    control.extend(collect_scripted_corrections(Learner(), 2, expert_probability=.35,
                                               learner_prefix_steps=1, **options))
    candidate.extend(replacements["interaction"])
    if copies:
        control.extend(augment_demonstrations(control, copies_per_episode=copies, seed=53))
        candidate.extend(augment_demonstrations(candidate, copies_per_episode=copies, seed=53))
    run = tmp_path / "control"
    run.mkdir()
    json_write(run / "demonstrations.json", control.provenance())
    json_write(run / "metadata.json", {"initial_agent_sha256": ACTOR_HASH,
        "reward_sha256": REWARD_HASH, "source_checkpoint_sha256": SOURCE_IDENTITY["sha256"]})
    if copies:
        np.savez_compressed(run / "demonstration_frames.npz",
                            **{f"frame_{index:06}": frame
                               for index, frame in enumerate(control.frames)})
    else:
        np.savez_compressed(run / "demonstration_frames.npz", frames=np.stack(control.frames))
    return candidate, run, replacements, summary


@pytest.mark.parametrize("copies", [0, 1])
def test_import_audit_proves_unchanged_content_and_exact_replacement_copies_streaming(
        bundle, tmp_path, monkeypatch, copies):
    candidate, run, replacements, summary = audit_cohort(bundle, tmp_path, copies=copies)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("audit materialized a whole NPZ"))
    result = audit_correction_replacement(candidate, run, replacements, summary)
    assert result["passed"] and result["all_other_slots_identical"]
    assert result["episodes"] == 3 * (copies + 1)
    assert result["changed_slots"] == 2 * (copies + 1)
    assert result["unchanged_slots"] == copies + 1
    assert result["replaced_unstyled_slots"] == 2
    assert result["replaced_appearance_slots"] == 2 * copies
    assert result["replacement_copy_profiles_unchanged"]
    assert result["control_ordered_episode_content_sha256"] != result[
        "current_ordered_episode_content_sha256"]


@pytest.mark.parametrize("mutation", ["unchanged", "copy_profile", "history", "order", "extra"])
def test_import_audit_rejects_uncontrolled_changes_before_encoding(bundle, tmp_path, mutation):
    candidate, run, replacements, summary = audit_cohort(bundle, tmp_path, copies=1)
    if mutation == "unchanged":
        candidate.episodes[0] = replace(candidate.episodes[0], instruction="Altered teacher goal")
    elif mutation == "copy_profile":
        profile = dict(candidate.episodes[-1].presentation)
        profile["texture_seed"] += 1
        candidate.episodes[-1] = replace(candidate.episodes[-1], presentation=profile)
    elif mutation == "history":
        row = candidate.episodes[1]
        candidate.episodes[1] = replace(row, executed_actions=(
            PhysicalAction(wheel=1), *row.executed_actions[1:]))
    elif mutation == "order":
        candidate.episodes[0], candidate.episodes[1] = candidate.episodes[1], candidate.episodes[0]
    else:
        candidate.episodes.append(candidate.episodes[0])
    with pytest.raises(ValueError):
        audit_correction_replacement(candidate, run, replacements, summary)


def test_import_audit_requires_each_admitted_unstyled_seed_exactly_once(bundle, tmp_path):
    candidate, run, replacements, summary = audit_cohort(bundle, tmp_path, copies=0)
    candidate.episodes[2] = candidate.episodes[1]
    control = json.loads((run / "demonstrations.json").read_text())
    control["episodes"][2] = control["episodes"][1]
    json_write(run / "demonstrations.json", control)
    with pytest.raises(ValueError, match="repeats an unstyled"):
        audit_correction_replacement(candidate, run, replacements, summary)


@pytest.mark.parametrize("mode", ["missing", "all_reused", "cached_shard", "unmatched"])
def test_trainer_reuse_encodes_only_missing_and_keeps_progress_compact(
        monkeypatch, tmp_path, mode):
    source, config = stub_script(monkeypatch, tmp_path)
    config = replace(config, tasks=("interaction",),
                     episode_instructions={"interaction": INSTRUCTIONS["interaction"]})
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    dataset = collect_scripted_demonstrations(
        1, tasks=config.tasks, seed=8288, difficulty=1., max_steps=80,
        level_bank_seed=config.level_bank_seed, episode_instructions=INSTRUCTIONS)
    used = sorted(set(dataset.episodes[0].frame_indices))
    vision = SimpleNamespace(image_size=96, patch_size=96, patch_grid=(1, 1), hidden_size=384)
    preprocessing = {"implementation": "types.SimpleNamespace",
                     "preprocess": "letterbox_bicubic_antialias_imagenet_v1",
                     "image_size": 96, "patch_size": 96, "patch_grid": [1, 1]}

    def identity(frames, indices):
        return feature_cache_identity(frames, indices, vision_sha256=ACTOR_HASH,
                                      preprocessing=preprocessing,
                                      feature_shape=(1, 384), dtype="float16")

    def values(frames):
        means = frames.float().mean(dim=(1, 2, 3))
        return means[:, None, None].expand(-1, 1, 384).contiguous()

    old_frames = dataset.frames
    old_used = used if mode == "all_reused" else used[:-1]
    if mode == "unmatched":
        old_frames = [np.full((96, 96, 3), 17, np.uint8)]
        old_used = [0]
    cache = tmp_path / "cache"
    with FrozenFeatureWriter(cache, identity(old_frames, old_used), max_bytes=1024**2) as writer:
        writer.write(old_used, values(torch.from_numpy(np.stack(
            [old_frames[index] for index in old_used]))).numpy().astype(np.float16))
        store = writer.publish()
    reuse_run = tmp_path / "reuse_source"
    reuse_run.mkdir()
    np.savez_compressed(reuse_run / "demonstration_frames.npz", frames=np.stack(old_frames))
    json_write(reuse_run / "feature_manifest.json", {
        "feature_cache_dir": str(store.directory.resolve()),
        "feature_cache_sha256": store.manifest["sha256"],
        "feature_cache_created_utc": store.manifest["created_utc"],
        "bytes": store.manifest["bytes"], "frames": len(old_used),
        "dtype": "float16", "vision_sha256": ACTOR_HASH})
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in store.directory.iterdir()}
    plan = FrozenFeatureReuse.prepare(reuse_run, dataset.frames, used,
                                     identity(dataset.frames, used),
                                     cache_root=cache, max_bytes=1024**2)
    missing = plan.missing_indices
    if mode == "cached_shard":
        with FrozenFeatureWriter(cache, plan.shard_identity, max_bytes=1024**2) as writer:
            writer.write(missing, values(torch.from_numpy(np.stack(
                [dataset.frames[index] for index in missing]))).numpy().astype(np.float16))
            writer.publish()
    encoded = []

    def encode(batch):
        encoded.extend(hashlib.sha256(frame.numpy().tobytes()).hexdigest() for frame in batch)
        return values(batch)

    agent = SimpleNamespace(vision=vision, summary_query=torch.zeros(1), encode_frames=encode)
    reward = object()
    monkeypatch.setattr(script, "tensor_hash", lambda value:
                        REWARD_HASH if value is reward else ACTOR_HASH)
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k:
                        (agent, SimpleNamespace(param_groups=[{}]), reward,
                         {"decisions": 0, "weights_revision": 13053, "config": config.to_dict()}))
    monkeypatch.setattr(script, "collect_scripted_demonstrations", lambda *a, **k: dataset)

    def stop_before_evaluation(*args, **kwargs):
        raise BeforeEncoding("feature encoding completed; no policy evaluation")

    monkeypatch.setattr(script, "evaluate_sequence", stop_before_evaluation)
    monkeypatch.setattr(script, "train_imitation", lambda *a, **k:
                        pytest.fail("test must not train"))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "train_imitation.py", "--config", "unused", "--warm-start", str(source),
        "--output", str(output), "--training-backend", "torch", "--difficulty", "1",
        "--episodes-per-task", "1", "--feature-cache-dir", str(cache),
        "--feature-reuse-run", str(reuse_run), "--feature-dtype", "float16"])
    with pytest.raises(BeforeEncoding):
        script.main()
    expected_encoded = [] if mode == "cached_shard" else [
        hashlib.sha256(dataset.frames[index].tobytes()).hexdigest() for index in missing]
    assert encoded == expected_encoded
    manifest = json.loads((output / "feature_manifest.json").read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    progress = json.loads((output / "status.json").read_text())
    assert manifest["feature_reuse"] == metadata["feature_reuse"]
    assert manifest["feature_reuse"]["mapping"] and manifest["feature_reuse"]["identity"]["indices"]
    assert manifest["reused_frames"] == len(used) - len(missing)
    assert manifest["encoded_frames"] == len(expected_encoded)
    assert manifest["cached_shard_frames"] == (len(missing) if mode == "cached_shard" else 0)
    assert manifest["bytes"] == manifest["logical_feature_bytes"] == len(used) * 384 * 2
    assert not progress["feature_cache_hit"] and progress["partial_feature_reuse"]
    assert "feature_reuse" not in progress and "mapping" not in progress
    assert progress["feature_reuse_mapping_sha256"] == manifest["feature_reuse"]["mapping_sha256"]
    for name in ("source_pack", "shard_pack"):
        compact = progress[f"feature_reuse_{name}"]
        assert compact is None or set(compact) == {"directory", "sha256", "created_utc"}
    assert source_hashes == {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in store.directory.iterdir()}


def test_trainer_audit_failure_is_recorded_before_encoding(bundle, monkeypatch, tmp_path):
    source, _ = stub_script(monkeypatch, tmp_path)
    monkeypatch.setattr(script, "collect_scripted_corrections", lambda agent, count, **options:
                        collect_scripted_corrections(Learner(), count, **options))

    def reject(*args):
        raise ValueError("unexpected unchanged slot mutation")

    monkeypatch.setattr(script, "audit_correction_replacement", reject)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "train_imitation.py", "--config", "unused", "--warm-start", str(source),
        "--output", str(output), "--training-backend", "torch", "--difficulty", "1",
        "--dataset-seed", "8288", "--episodes-per-task", "1",
        "--correction-episodes-per-task", "2", "--correction-replacement-bundle", str(bundle[0]),
        "--correction-replacement-sha256", bundle_sha(bundle[0]),
        "--replacement-control-run", str(tmp_path / "control")])
    with pytest.raises(ValueError, match="unchanged slot"):
        script.main()
    audit = json.loads((output / "replacement_import_audit.json").read_text())
    assert audit["before_encoding"] and not audit["passed"]
    assert audit["bundle_sha256"] == bundle_sha(bundle[0])
    assert not (output / "feature_manifest.json").exists()
