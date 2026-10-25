"""Data-only exports exercise real replay loaders, audits and frozen feature storage."""

import argparse
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_imitation_correction_replacement import (
    INSTRUCTIONS,
    SOURCE_IDENTITY,
    Learner,
    load,
)
from test_imitation_correction_replacement import (
    bundle as bundle,  # noqa: F401
)
from test_imitation_demonstration_reuse import corpus as corpus  # noqa: F401

from carryrl.agent.model import AgentConfig
from carryrl.training import ordinary_preparation as ordinary
from carryrl.training.feature_store import FrozenFeatureReuse
from carryrl.training.imitation import (
    SavedDemonstrationReuse,
    audit_correction_replacement,
    collect_scripted_corrections,
    collect_scripted_demonstrations,
)
from carryrl.training.retention import _argv, _settings
from carryrl.training.sequence_config import SequenceRunConfig


class FrozenEncoder:
    vision = SimpleNamespace(image_size=96, patch_size=96, patch_grid=(1, 1), hidden_size=384)

    def __init__(self):
        self.batches = []

    def encode_frames(self, batch):
        self.batches.append(len(batch))
        return torch.ones((len(batch), 1, 384))


def forbidden(*args, **kwargs):
    raise AssertionError("data preparation attempted an optimizer or fit")


def test_real_export_import_and_audit_with_bounded_frozen_features(
        corpus, bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(ordinary.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "step", forbidden)
    monkeypatch.setattr("carryrl.training.imitation.train_imitation", forbidden)
    monkeypatch.setattr(ordinary, "tensor_hash", lambda module: "a" * 64)
    source, _, _, cache = corpus
    config = SequenceRunConfig(agent=AgentConfig(
        pretrained=False, text_encoder="multilingual_e5_small"),
        tasks=("interaction",),
        episode_instructions={"interaction": INSTRUCTIONS["interaction"]},
        level_bank_seed=20261008, curriculum_difficulty=1., evaluation_difficulty=1.)
    options = dict(tasks=config.tasks, seed=13288, difficulty=1., max_steps=80,
        level_bank_seed=20261008, episode_instructions=config.episode_instructions)
    control = collect_scripted_demonstrations(2, **options)
    treatment = ordinary._clone(control)
    assert treatment.frames[0] is control.frames[0] and not control.frames[0].flags.writeable
    control.extend(collect_scripted_corrections(Learner(), 2, expert_probability=0., **options))
    replacements, replacement = load(bundle[0])
    treatment.extend(replacements["interaction"])
    agent = FrozenEncoder()
    checkpoint = {"path": "/producer/checkpoints/rollout_0000",
                  **{key: SOURCE_IDENTITY[key] for key in
                     ("sha256", "agent_sha256", "reward_sha256")}}
    feature_pins = {}
    for name, dataset, reused in (("control", control, source),
                                  ("treatment", treatment, tmp_path / "control")):
        features, pins = ordinary._freeze(dataset, agent, reused, cache, 1024**2, batch_size=3)
        feature_pins.update(pins)
        summary = ordinary._write_corpus(tmp_path / name, dataset, config, checkpoint, features,
                         replacement=replacement if name == "treatment" else None)
        assert summary["roundtrip_verified"]
        metadata = json.loads((tmp_path / name / "metadata.json").read_text())
        assert metadata["preparation_only"] and not metadata["training_executed"]
        assert not metadata["optimizer_allocated"] and metadata["updates"] == 0
    assert agent.batches and max(agent.batches) <= 3
    assert feature_pins
    for filename in feature_pins:
        if filename.endswith("features.npy"):
            mapped = np.load(filename, mmap_mode="r", allow_pickle=False)
            assert isinstance(mapped, np.memmap) and mapped.dtype == np.float16
    actual = SavedDemonstrationReuse.prepare(tmp_path / "treatment", config=config,
                                             max_frame_bytes=256 * 1024**2)
    audited = audit_correction_replacement(actual.dataset, tmp_path / "control",
                                           replacements, replacement)
    assert audited["changed_slots"] == 2 and audited["unchanged_slots"] == 2
    assert audited["all_other_slots_identical"]
    assert any(row.executed_actions != row.actions for row in actual.dataset.episodes
               if row.executed_actions is not None)
    feature_record = json.loads((tmp_path / "treatment/feature_manifest.json").read_text())
    identity = feature_record["feature_reuse"]["identity"]
    reused = FrozenFeatureReuse.prepare(tmp_path / "treatment", actual.dataset.frames,
        identity["indices"], identity, cache_root=cache, max_bytes=1024**2)
    assert not reused.missing_indices


def test_wrong_rgb_or_frozen_encoder_identity_is_rejected(corpus, monkeypatch):
    source, dataset, _, cache = corpus
    monkeypatch.setattr(ordinary, "tensor_hash", lambda module: "b" * 64)
    with pytest.raises(ValueError, match="incompatible|identity|configuration"):
        ordinary._freeze(dataset, FrozenEncoder(), source, cache, 1024**2)


def test_checkpoint_sha_mismatch_fails_before_model_or_optimizer(corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(ordinary, "load_sequence_checkpoint", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    checkpoint = tmp_path / "source/checkpoints/rollout_0000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"opaque checkpoint")
    (checkpoint / "manifest.json").write_text(json.dumps({
        "sha256": "1" * 64, "agent_sha256": "2" * 64,
        "reward_sha256": "3" * 64, "weights_revision": 0}))
    with pytest.raises(ValueError, match="trusted archive"):
        ordinary.prepare_ordinary_o1(checkpoint, "0" * 64, tmp_path / "bundle", "4" * 64,
            tmp_path / "prepared", source_config=corpus[0] / "config.json",
            feature_reuse_run=corpus[0], feature_cache_dir=corpus[3])
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("eval_all_levels", [False, True])
def test_ordinary_launch_argv_matches_real_parser_and_has_no_collection_flags(
        corpus, tmp_path, monkeypatch, eval_all_levels):
    path = Path(__file__).resolve().parents[1] / "scripts/train_imitation.py"
    spec = importlib.util.spec_from_file_location("ordinary_training_parser", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    common = dict(config=str(corpus[0] / "config.json"), warm_start="/source/checkpoint",
        training_backend="mlx", epochs=2, evaluation_difficulty=1., level_bank_seed=20261008,
        dataset_seed=3176, learning_rate=3e-5, action_balance="active_idle",
        canonical_replay_weight=1., minibatch_episodes=8, eval_every=1, checkpoint_every=1,
        validation_nll=True, validation_nll_episodes_per_task=2, max_wall_seconds=1800.,
        feature_budget_mib=14336, frame_budget_mib=1536, feature_dtype="float16",
        feature_cache_dir=str(corpus[3]), feature_reuse_run=str(corpus[0]),
        demonstration_reuse_run=str(corpus[0]), final_all_levels=True, screen_text_language=None)
    if eval_all_levels:
        common["eval_all_levels"] = True
    settings = _settings(common, coefficient=0., output=tmp_path / "future")
    argv = _argv(settings)
    original = argparse.ArgumentParser.parse_args
    captured = {}

    class Parsed(Exception):
        pass

    def capture(parser, *args, **kwargs):
        captured.update(json.loads(json.dumps(
            vars(original(parser, *args, **kwargs)), default=str)))
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    monkeypatch.setattr(sys, "argv", [str(path), *argv[argv.index(str(path)) + 1:]])
    with pytest.raises(Parsed):
        script.main()
    assert captured == settings
    assert "--demonstration-reuse-run" in argv and "--feature-reuse-run" in argv
    assert not any(flag in argv for flag in ("--difficulty", "--episodes-per-task",
        "--correction-episodes-per-task", "--correction-replacement-bundle"))
    assert not (tmp_path / "future").exists()


@pytest.mark.parametrize("filename", ["feature_manifest.json", "demonstration_frames.npz"])
def test_feature_ancestry_pins_reject_changed_source_dependencies(corpus, filename):
    source, _, _, _ = corpus
    pins = ordinary._feature_source_pins(source)
    assert str(source / filename) in pins
    ordinary._check_pins(pins)
    with (source / filename).open("ab") as stream:
        stream.write(b"changed source dependency")
    with pytest.raises(ValueError, match="pinned file changed"):
        ordinary._check_pins(pins)


@pytest.mark.parametrize("control,treatment", [
    ("control_corpus", "future"), ("control_corpus/fit", "future"),
    ("shared_base_config.json", "future"), ("control_manifest.json", "future"),
    ("alias/../same", "same"), ("future", "future/child")])
def test_future_outputs_cannot_alias_preparation_or_each_other(tmp_path, control, treatment):
    output = (tmp_path / "preparation").resolve()
    with pytest.raises(ValueError, match="future training output"):
        ordinary._future_outputs(output, output / control, output / treatment)
    assert not output.exists()


@pytest.mark.parametrize("field", ["source_and_lock_sha256", "dependencies"])
def test_manifest_verification_rejects_changed_source_or_dependencies(tmp_path, monkeypatch, field):
    identity = {"source_and_lock_sha256": "1" * 64, "source_identity_version": "test",
                "dependencies": {"torch": "frozen"}}
    monkeypatch.setattr(ordinary, "provenance", lambda: identity)
    stored = {**identity, field: "changed"}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"format": ordinary.FORMAT,
        "status": "prepared_not_executed", "updates": 0, "optimizer_allocated": False,
        "training_executed": False, "source_provenance": stored}))
    with pytest.raises(ValueError, match="source/lock or dependency identity changed"):
        ordinary.load_ordinary_preparation(path)


@pytest.fixture
def mocked_preparation(tmp_path, monkeypatch):
    """Exercise the full fixed schedule with tiny fake rows and no model/RGB loading."""
    checkpoint = tmp_path / "source/checkpoints/rollout_0000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text("{}")
    source_config = checkpoint.parents[1] / "config.json"
    config = SequenceRunConfig(agent=AgentConfig(
        pretrained=False, text_encoder="multilingual_e5_small"),
        episode_instructions={task: (task,) for task in ordinary.TASKS})
    source_config.write_text(json.dumps(config.to_dict()))
    identity = {"path": str(checkpoint), "sha256": "1" * 64,
                "agent_sha256": "a" * 64, "reward_sha256": "a" * 64}
    monkeypatch.setattr(ordinary, "_checkpoint", lambda *args: identity)
    monkeypatch.setattr(ordinary, "tensor_hash", lambda module: "a" * 64)
    provenance = {"source_and_lock_sha256": "1" * 64, "source_identity_version": "test",
                  "dependencies": {"torch": "frozen"}}
    monkeypatch.setattr(ordinary, "provenance", lambda: provenance)
    monkeypatch.setattr(ordinary, "_rss", lambda: 0)
    monkeypatch.setattr(ordinary.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=100 * ordinary.GIB))
    monkeypatch.setattr(ordinary, "load_sequence_checkpoint",
                        lambda *args, **kwargs: (torch.nn.Identity(), None, object(), {}))
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "step", forbidden)
    monkeypatch.setattr("carryrl.training.imitation.train_imitation", forbidden)
    calls, exported = {}, {}

    def dataset(count, tasks, label):
        result = ordinary.DemonstrationDataset()
        result.episodes = [SimpleNamespace(task=task, token=f"{label}:{task}:{index}")
                           for task in tasks for index in range(count)]
        return result

    def teachers(count, *, tasks, **kwargs):
        calls["teachers"] = (count, tasks, kwargs)
        return dataset(count, tasks, "teacher")

    def corrections(agent, count, *, tasks, **kwargs):
        calls["corrections"] = (count, tasks, kwargs)
        return dataset(count, tasks, "control_correction")

    digest = lambda data, row: ordinary.json_sha256(row.token)  # noqa: E731
    monkeypatch.setattr(ordinary, "demonstration_episode_digest", digest)
    # Only extend's ordering matters here; replay integrity has real coverage above.
    monkeypatch.setattr(ordinary.DemonstrationDataset, "extend",
                        lambda self, other: self.episodes.extend(other.episodes))
    monkeypatch.setattr(ordinary, "collect_scripted_demonstrations", teachers)
    monkeypatch.setattr(ordinary, "collect_scripted_corrections", corrections)

    def bundle(path, *, correction_tasks, episodes_per_task, **kwargs):
        calls["bundle"] = (correction_tasks, episodes_per_task, kwargs)
        # The real loader returns alphabetical keys, independently of collection order.
        replacements = {task: dataset(episodes_per_task, (task,), "replacement")
                        for task in sorted(correction_tasks)}
        count = len(correction_tasks) * episodes_per_task
        return replacements, {"bundle_directory": str(tmp_path / "bundle"), "files": {},
            "selected_tasks": sorted(correction_tasks), "episodes_per_task": episodes_per_task,
            "replacement_episodes": count, "supporting_teacher_episodes_excluded": count,
            "replacement_episode_content_sha256": {
                task: [digest(data, row) for row in data.episodes]
                for task, data in replacements.items()}}

    monkeypatch.setattr(ordinary, "load_correction_replacement_bundle", bundle)
    monkeypatch.setattr(ordinary, "validate_dataset_runtime_holdout", lambda *a, **kw: None)
    monkeypatch.setattr(ordinary, "_forecast", lambda *a: {
        "future_four_checkpoints": 0, "misc_reservation": 0})
    monkeypatch.setattr(ordinary, "_freeze", lambda *a: ({}, {}))

    def write_corpus(directory, data, *args, **kwargs):
        directory.mkdir()
        exported[directory.name] = data
        hashes = [digest(data, row) for row in data.episodes]
        return {"source_run": str(directory), "files": {}, "episodes": len(hashes),
                "ordered_episode_sha256": hashes, "episode_order_sha256": ordinary.json_sha256(
                    hashes), "roundtrip_verified": True}

    monkeypatch.setattr(ordinary, "_write_corpus", write_corpus)
    monkeypatch.setattr(ordinary.SavedDemonstrationReuse, "prepare",
        lambda path, **kw: SimpleNamespace(dataset=exported[Path(path).name]))

    def audit(data, *args):
        teacher_count = 64 * len(ordinary.TASKS)
        return {"passed": True, "episodes": len(data.episodes),
                "changed_slots": len(data.episodes) - teacher_count,
                "unchanged_slots": teacher_count, "all_other_slots_identical": True,
                "changed_positions": list(range(teacher_count, len(data.episodes)))}

    monkeypatch.setattr(ordinary, "audit_correction_replacement", audit)

    def prepare(experiment="O2", coefficient=1.):
        return ordinary.prepare_ordinary(checkpoint, "1" * 64, tmp_path / "bundle", "2" * 64,
            tmp_path / experiment, feature_cache_dir=tmp_path / "cache",
            experiment=experiment, canonical_anchor_coefficient=coefficient)

    return SimpleNamespace(prepare=prepare, calls=calls, exported=exported, tmp_path=tmp_path)


@pytest.mark.parametrize("experiment,correction_count,total,batches,updates", [
    ("O1", 8, 1032, 129, 258), ("O2", 4, 1048, 131, 262)])
def test_fixed_schedules_build_equal_arms_without_supporting_teachers(
        mocked_preparation, experiment, correction_count, total, batches, updates):
    fake = mocked_preparation
    coefficient = 0. if experiment == "O1" else 1.
    plan = fake.prepare(experiment, coefficient)
    tasks = ("navigation",) if experiment == "O1" else (
        "camera", "interaction", "causality", "key_door", "hazards", "chain")
    assert fake.calls["teachers"][:2] == (64, ordinary.TASKS)
    assert fake.calls["corrections"][:2] == (correction_count, tasks)
    assert fake.calls["corrections"][2]["seed"] == 8176
    assert fake.calls["corrections"][2]["expert_probability"] == 0
    assert fake.calls["bundle"][:2] == (tasks, correction_count)
    assert plan["batches_per_epoch"] == batches
    assert plan["expected_optimizer_updates"] == updates and plan["expected_epochs"] == 2
    assert not plan["training_executed"] and not plan["optimizer_allocated"]
    assert plan["updates"] == 0
    for name, arm in plan["arms"].items():
        assert arm["corpus"]["episodes"] == total
        assert arm["settings"]["canonical_anchor_coefficient"] == coefficient
        assert ("--eval-all-levels" in arm["argv"]) == (experiment == "O2")
        rows = fake.exported[f"{name}_corpus"].episodes
        assert all(row.token.startswith("teacher:") for row in rows[:1024])
        assert [row.task for row in rows[1024:]] == [
            task for task in tasks for _ in range(correction_count)]
        prefix = "control_correction:" if name == "control" else "replacement:"
        assert all(row.token.startswith(prefix) for row in rows[1024:])
    for order in plan["epoch_orders"]:
        assert sorted(order) == list(range(total))
    assert ordinary.load_ordinary_preparation(fake.tmp_path / experiment) == json.loads(
        json.dumps(plan))


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf"), True, "1"])
def test_invalid_anchor_fails_before_loading_model(tmp_path, monkeypatch, value):
    monkeypatch.setattr(ordinary, "load_sequence_checkpoint", forbidden)
    with pytest.raises(ValueError, match="anchor coefficient"):
        ordinary.prepare_ordinary(tmp_path, "", tmp_path, "", tmp_path,
            feature_cache_dir=tmp_path, canonical_anchor_coefficient=value)


def test_unknown_experiment_fails_before_loading_model(tmp_path, monkeypatch):
    monkeypatch.setattr(ordinary, "load_sequence_checkpoint", forbidden)
    with pytest.raises(ValueError, match="experiment must be O1 or O2"):
        ordinary.prepare_ordinary(tmp_path, "", tmp_path, "", tmp_path,
            feature_cache_dir=tmp_path, experiment="O1+O2")


def test_legacy_o1_entrypoint_preserves_default_schedule(monkeypatch):
    captured = {}

    def prepare(*args, **kwargs):
        captured.update(kwargs)
        return "prepared"

    monkeypatch.setattr(ordinary, "prepare_ordinary", prepare)
    assert ordinary.prepare_ordinary_o1("source", feature_cache_dir="cache") == "prepared"
    assert captured == {"feature_cache_dir": "cache", "experiment": "O1"}


@pytest.mark.parametrize("mutation,match", [
    ("schedule", "collection schedule"), ("budget", "planned budget"),
    ("teacher_hash", "teacher count or hash"), ("teacher_length", "teacher count or hash"),
    ("corpus_length", "episode order"), ("permutation", "shuffle permutations"),
    ("shuffle_seed", "shuffle permutations"), ("unequal_anchor", "argv/settings"),
    ("invalid_anchor", "anchor coefficient"), ("missing_eval_all", "argv/settings"),
    ("fit_budget", "byte budget"), ("replacement_order", "replacement episode order"),
])
def test_o2_verification_rejects_inconsistent_schedule_and_fit(mocked_preparation, mutation, match):
    fake = mocked_preparation
    plan = deepcopy(fake.prepare())
    if mutation == "schedule":
        plan["collection_schedule"]["correction_tasks"].reverse()
    elif mutation == "budget":
        plan["expected_optimizer_updates"] = 258
    elif mutation == "teacher_hash":
        plan["teacher_order_sha256"] = "0" * 64
    elif mutation == "teacher_length":
        plan["teacher_ordered_episode_sha256"].pop()
        plan["teacher_order_sha256"] = ordinary.json_sha256(plan["teacher_ordered_episode_sha256"])
    elif mutation == "corpus_length":
        corpus = plan["arms"]["control"]["corpus"]
        corpus["ordered_episode_sha256"].pop()
        corpus["episode_order_sha256"] = ordinary.json_sha256(corpus["ordered_episode_sha256"])
    elif mutation in ("permutation", "shuffle_seed"):
        order = plan["epoch_orders"][0]
        if mutation == "permutation":
            order[0] = order[1]
        else:
            order.reverse()
        plan["epoch_order_sha256"] = [ordinary.json_sha256(row) for row in plan["epoch_orders"]]
    elif mutation == "unequal_anchor":
        plan["arms"]["treatment"]["settings"]["canonical_anchor_coefficient"] = 0.
    elif mutation == "invalid_anchor":
        plan["arms"]["control"]["settings"]["canonical_anchor_coefficient"] = -1.
    elif mutation == "missing_eval_all":
        plan["arms"]["control"]["settings"].pop("eval_all_levels")
    elif mutation == "fit_budget":
        plan["arms"]["control"]["settings"]["feature_budget_mib"] = 0
    else:
        corpus = plan["arms"]["treatment"]["corpus"]
        corpus["ordered_episode_sha256"][1024:] = reversed(corpus["ordered_episode_sha256"][1024:])
        corpus["episode_order_sha256"] = ordinary.json_sha256(corpus["ordered_episode_sha256"])
    for arm in plan["arms"].values():
        arm["argv"] = ordinary._argv(arm["settings"])
    path = fake.tmp_path / "O2/manifest.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=match):
        ordinary.load_ordinary_preparation(path)


@pytest.mark.parametrize("experiment,coefficient", [(None, None), ("O2", "1")])
def test_preparation_cli_dispatches_fixed_schedule_and_shared_anchor(
        tmp_path, monkeypatch, experiment, coefficient):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_ordinary.py"
    spec = importlib.util.spec_from_file_location("ordinary_preparation_parser", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    captured = {}

    def prepare(**kwargs):
        captured.update(kwargs)
        return {"status": "prepared_not_executed", "expected_optimizer_updates": 262}

    monkeypatch.setattr(script, "prepare_ordinary", prepare)
    args = ["--source-checkpoint", str(tmp_path), "--source-checkpoint-sha256", "1" * 64,
            "--replacement-bundle", str(tmp_path), "--replacement-bundle-sha256", "2" * 64,
            "--output", str(tmp_path / "out"), "--feature-cache-dir", str(tmp_path / "cache")]
    if experiment is not None:
        args += ["--experiment", experiment, "--canonical-anchor-coefficient", coefficient]
    assert script.main(args) == 0
    assert captured["experiment"] == (experiment or "O1")
    assert captured["canonical_anchor_coefficient"] == (float(coefficient) if coefficient else 0.)
    assert captured["expected_checkpoint_sha256"] == "1" * 64
    assert captured["expected_bundle_sha256"] == "2" * 64
