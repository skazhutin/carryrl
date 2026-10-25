"""Preparation authenticates a real tiny mixed RGB corpus without allocating a policy."""

import argparse
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_imitation_demonstration_reuse import corpus as corpus  # noqa: F401

from carryrl.training.retention import (
    json_sha256,
    load_retention_plan,
    prepare_retention_plan,
)


@pytest.fixture
def checkpoint(tmp_path):
    directory = tmp_path / "baseline"
    directory.mkdir()
    # Intentionally not a Torch archive: preparation must only authenticate its bytes.
    (directory / "state.pt").write_bytes(b"opaque unchanged optimizer and policy state")
    digest = hashlib.sha256((directory / "state.pt").read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps({
        "sha256": digest, "agent_sha256": "b" * 64, "reward_sha256": "c" * 64,
        "weights_revision": 41}))
    return directory, digest


def prepare(corpus, checkpoint, tmp_path, **changes):
    run, _, _, cache = corpus
    directory, digest = checkpoint
    options = dict(positive_anchor_coefficient=0.1, epochs=2, learning_rate=1e-5,
        dataset_seed=1234, max_wall_seconds=60, backend="torch", feature_cache_dir=cache,
        frame_budget_mib=256, feature_budget_mib=64, minibatch_episodes=8)
    options.update(changes)
    return prepare_retention_plan(run, directory, digest, tmp_path / "plan", **options)


def test_exact_pair_preparation_never_loads_models_and_rechecks_sources(
        corpus, checkpoint, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("preparation attempted model/optimizer/archive loading")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    run, dataset, _, _ = corpus
    before = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
    plan = prepare(corpus, checkpoint, tmp_path)
    assert plan["status"] == "prepared_not_executed"
    assert plan["updates"] == 0 and not plan["checkpoint_created"]
    assert plan["expected_optimizer_updates"] == 2 * ((len(dataset.episodes) + 7) // 8)
    assert plan["source_corpus"]["episodes"] == len(dataset.episodes)
    assert plan["source_corpus"]["decisions"] == sum(map(len, dataset.episodes))
    assert plan["source_corpus"]["provenance"] == dataset.provenance()
    assert plan["source_corpus"]["files"]["demonstrations.json"]["sha256"] == (
        hashlib.sha256(before["demonstrations.json"]).hexdigest())
    assert plan["frozen_features"]["new_features"] == 0
    assert plan["source_corpus"]["runtime_holdout_guard_passed"] is True
    off, on = (plan["arms"][name]["settings"] for name in ("anchor_off", "anchor_on"))
    assert {key for key in off if off[key] != on[key]} == {
        "output", "canonical_anchor_coefficient"}
    for arm in plan["arms"].values():
        assert not Path(arm["output"]).exists()
        assert "--demonstration-reuse-run" in arm["argv"]
        assert "--feature-reuse-run" in arm["argv"]
        assert "--difficulty" not in arm["argv"]
        assert "--styled-correction-episodes-per-task" not in arm["argv"]
    assert sorted(path.name for path in (tmp_path / "plan").iterdir()) == [
        "manifest.json", "shared_base_config.json"]
    loaded = load_retention_plan(tmp_path / "plan")
    assert loaded["source_checkpoint"]["weights_revision"] == 41
    assert loaded["evaluation_protocol"]["split"] == "validation"
    assert loaded["validation_likelihood_protocol"]["split"] == "validation"
    assert before == {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}


@pytest.mark.parametrize("full_levels", [False, True])
def test_prepared_settings_equal_real_training_argparse(
        corpus, checkpoint, tmp_path, monkeypatch, full_levels):
    options = {"eval_all_levels": full_levels}
    if full_levels:
        source, checkpoint = initial_evaluation(corpus, checkpoint)
        options["reuse_initial_evaluation"] = source
    plan = prepare(corpus, checkpoint, tmp_path, **options)
    spec = importlib.util.spec_from_file_location("retention_train_parser",
        Path(__file__).resolve().parents[1] / "scripts/train_imitation.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    original_parse = argparse.ArgumentParser.parse_args
    captured = {}

    class Parsed(Exception):
        pass

    def capture(parser, *args, **kwargs):
        parsed = original_parse(parser, *args, **kwargs)
        captured.update(json.loads(json.dumps(vars(parsed), default=str)))
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    for arm in plan["arms"].values():
        captured.clear()
        start = arm["argv"].index(str(Path(script.__file__))) + 1
        monkeypatch.setattr(sys, "argv", [str(Path(script.__file__)), *arm["argv"][start:]])
        with pytest.raises(Parsed):
            script.main()
        assert captured == arm["settings"]


@pytest.mark.parametrize("mutation",
                         ["checkpoint", "corpus", "feature", "config", "arm", "protocol"])
def test_loader_rejects_changed_pins_and_uncontrolled_arms(corpus, checkpoint, tmp_path, mutation):
    prepare(corpus, checkpoint, tmp_path)
    path = tmp_path / "plan/manifest.json"
    plan = json.loads(path.read_text())
    if mutation == "checkpoint":
        (checkpoint[0] / "state.pt").write_bytes(b"changed optimizer")
    elif mutation == "corpus":
        (corpus[0] / "metadata.json").write_text("{}")
    elif mutation == "feature":
        feature = next(filename for filename in plan["frozen_features"]["files"]
                       if filename.endswith("features.npy"))
        with Path(feature).open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"x")
    elif mutation == "config":
        config = tmp_path / "plan/shared_base_config.json"
        value = json.loads(config.read_text())
        value["weight_decay"] = 0.7
        config.write_text(json.dumps(value))
    elif mutation == "arm":
        plan["arms"]["anchor_on"]["settings"]["dataset_seed"] += 1
        path.write_text(json.dumps(plan))
    else:
        plan["evaluation_protocol"]["split"] = "test"
        path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        load_retention_plan(path)


def test_trusted_manifest_hash_and_fresh_destination_required(corpus, checkpoint, tmp_path):
    prepare(corpus, checkpoint, tmp_path)
    with pytest.raises(ValueError, match="manifest SHA256"):
        load_retention_plan(tmp_path / "plan", expected_sha256="0" * 64)
    with pytest.raises(FileExistsError):
        prepare(corpus, checkpoint, tmp_path)
    assert json_sha256({"b": 2, "a": 1}) == json_sha256({"a": 1, "b": 2})


@pytest.mark.parametrize("changes", [{"positive_anchor_coefficient": 0}, {"epochs": 0},
    {"learning_rate": float("nan")}, {"dataset_seed": -1}, {"max_wall_seconds": 0}])
def test_invalid_explicit_candidate_is_rejected_before_allocation(
        corpus, checkpoint, tmp_path, changes):
    with pytest.raises(ValueError):
        prepare(corpus, checkpoint, tmp_path, **changes)
    assert not (tmp_path / "plan").exists()


def initial_evaluation(corpus, checkpoint):
    """Build a complete cohort fixture from metadata, without policy inference."""
    from dataclasses import replace

    from carryrl.envs.atomic import _identity, atomic_level_banks, atomic_semantic_level_id
    from carryrl.training.sequence import (
        _episode_instruction,
        sequence_evaluation_protocol,
        sequence_evaluation_summaries,
    )

    config = replace(corpus[2], eval_all_levels=True)
    protocol = sequence_evaluation_protocol(config)
    records = []
    for policy in protocol["policies"]:
        for task, count in protocol["counts"].items():
            bank = atomic_level_banks(task, config.level_bank_seed)["validation"]
            for index in range(count):
                level = bank[index % len(bank)]
                records.append({"policy": policy, "task": task, "episode_index": index,
                    "seed": 1_000_000 + index,
                    "action_seed": (config.eval_seed + config.tasks.index(task) * 100_000
                                    + index) % 2**32,
                    "difficulty": protocol["difficulty"], "level_bank_seed": config.level_bank_seed,
                    "instruction": _episode_instruction(config, task, 1_000_000 + index),
                    "steps": 1, "terminated": True, "truncated": False,
                    "level_index": index % len(bank), "level_id": _identity(level),
                    "semantic_level_id": atomic_semantic_level_id(task, level),
                    "outcome": {"known": True, "success": False, "failure": True, "score": 0}})
    directory, digest = checkpoint
    manifest = json.loads((directory / "manifest.json").read_text())
    evaluation = {"split": "validation", "policy_and_reward_frozen": True,
        **{key: manifest[key] for key in
           ("agent_sha256", "reward_sha256", "weights_revision")},
        "evaluation_seed": config.eval_seed, "evaluation_difficulty": protocol["difficulty"],
        "level_bank_seed": config.level_bank_seed, "protocol_sha256": json_sha256(protocol),
        "created_utc": "2026-10-01T01:02:03+00:00", "epoch": 0, "records": records,
        "summaries": sequence_evaluation_summaries(records, tasks=config.tasks,
                                                    policies=protocol["policies"])}
    destination = directory / "checkpoints" / "rollout_0000"
    destination.parent.mkdir(parents=True)
    # Keep the opaque authenticated fixture; only its run layout changes.
    destination.mkdir()
    for name in ("manifest.json", "state.pt"):
        (destination / name).write_bytes((directory / name).read_bytes())
    source = directory / "evaluations" / "baseline.json"
    source.parent.mkdir()
    source.write_text(json.dumps(evaluation))
    return source, (destination, digest)


def test_full_epoch_protocol_and_initial_evaluation_are_pinned_without_model_loading(
        corpus, checkpoint, tmp_path, monkeypatch):
    source, checkpoint = initial_evaluation(corpus, checkpoint)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("must not load models"))
    plan = prepare(corpus, checkpoint, tmp_path, eval_all_levels=True,
                   reuse_initial_evaluation=source)
    assert plan["initial_evaluation"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert plan["scheduled_evaluation_protocol"] == plan["evaluation_protocol"]
    assert plan["wall_budget"]["includes"] == ["optimizer_training", "scheduled_validation"]
    assert plan["resources"]["max_process_rss_bytes"] == 12 * 1024**3
    for arm in plan["arms"].values():
        assert arm["settings"]["eval_all_levels"] is True
        assert arm["settings"]["reuse_initial_evaluation"] == str(source)
        assert "--eval-all-levels" in arm["argv"]
        assert "--reuse-initial-evaluation" in arm["argv"]
    assert str(corpus[0] / "feature_manifest.json") in plan["frozen_features"]["files"]
    loaded = load_retention_plan(tmp_path / "plan")
    assert loaded["initial_evaluation"] == plan["initial_evaluation"]


@pytest.mark.parametrize("mutation", ["source_bytes", "semantic_cohort", "policy", "settings",
                                      "scheduled_protocol", "resource", "missing_pin"])
def test_reused_evaluation_and_scheduled_flags_reject_tampering(
        corpus, checkpoint, tmp_path, mutation):
    source, checkpoint = initial_evaluation(corpus, checkpoint)
    prepare(corpus, checkpoint, tmp_path, eval_all_levels=True, reuse_initial_evaluation=source)
    path = tmp_path / "plan/manifest.json"
    plan = json.loads(path.read_text())
    if mutation in {"source_bytes", "semantic_cohort", "policy"}:
        evaluated = json.loads(source.read_text())
        if mutation == "semantic_cohort":
            evaluated["records"][0]["semantic_level_id"] = "wrong"
        elif mutation == "policy":
            evaluated["agent_sha256"] = "0" * 64
        else:
            evaluated["created_utc"] = "2026-10-02T01:02:03+00:00"
        source.write_text(json.dumps(evaluated))
        if mutation != "source_bytes":
            # Updating the byte pin cannot bypass semantic/checkpoint checks.
            plan["initial_evaluation"].update(bytes=source.stat().st_size,
                sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    elif mutation == "settings":
        for arm in plan["arms"].values():
            arm["settings"].pop("eval_all_levels")
            arm["argv"].remove("--eval-all-levels")
    elif mutation == "scheduled_protocol":
        plan["scheduled_evaluation_protocol"]["split"] = "test"
    elif mutation == "resource":
        plan["resources"]["minimum_free_disk_bytes"] = 0
    else:
        plan.pop("initial_evaluation")
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        load_retention_plan(path)


def test_preparation_rejects_initial_evaluation_from_unrelated_directory(
        corpus, checkpoint, tmp_path):
    source, checkpoint = initial_evaluation(corpus, checkpoint)
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="warm-start run"):
        prepare(corpus, checkpoint, tmp_path, reuse_initial_evaluation=unrelated)
    assert not (tmp_path / "plan").exists()


def test_historical_v1_plan_without_new_flags_still_loads(corpus, checkpoint, tmp_path):
    plan = prepare(corpus, checkpoint, tmp_path)
    path = tmp_path / "plan/manifest.json"
    for key in ("preparation_revision", "scheduled_evaluation_protocol",
                "scheduled_evaluation_protocol_sha256", "resources", "wall_budget"):
        plan.pop(key)
    plan["frozen_features"]["files"] = {filename: pin for filename, pin in
        plan["frozen_features"]["files"].items() if not filename.startswith(str(corpus[0]))}
    assert "eval_all_levels" not in plan["arms"]["anchor_off"]["settings"]
    path.write_text(json.dumps(plan))
    assert load_retention_plan(path)["expected_epochs"] == 2


def test_feature_graph_pins_include_referenced_shards_even_when_not_used_by_rows(tmp_path):
    from carryrl.training.retention import _feature_dependency_pins

    parent, child = tmp_path / "parent", tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    packs = []
    for name in ("parent_pack", "referenced_shard"):
        pack = tmp_path / name
        pack.mkdir()
        (pack / "manifest.json").write_text("{}")
        (pack / "features.npy").write_bytes(b"opaque immutable pack")
        packs.append(pack)
    (parent / "feature_manifest.json").write_text(json.dumps({
        "feature_cache_dir": str(packs[0])}))
    (child / "feature_manifest.json").write_text(json.dumps({"feature_reuse": {
        "source_run": str(parent), "source_pack": {"directory": str(packs[0])},
        "shard_pack": {"directory": str(packs[1])}}}))
    for run in (parent, child):
        (run / "demonstration_frames.npz").write_bytes(b"opaque RGB")
    pins = _feature_dependency_pins(child)
    assert set(pins) == {str(directory / filename)
        for directory in (parent, child) for filename in
        ("feature_manifest.json", "demonstration_frames.npz")} | {
        str(directory / filename) for directory in packs for filename in
        ("manifest.json", "features.npy")}


def test_checkpoint_identity_restores_original_file_pins(checkpoint, tmp_path, monkeypatch):
    from carryrl.training.retention import _checkpoint

    source, digest = checkpoint
    directory = tmp_path / "runs/baseline/checkpoints/rollout_0000"
    shutil.copytree(source, directory)
    expected = _checkpoint(directory, digest)
    state = directory / "state.pt"
    state.with_suffix(".pt.gz").write_bytes(gzip.compress(state.read_bytes()))
    state.unlink()
    monkeypatch.setattr("carryrl.training.checkpoint_storage.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=100 * 1024**3))
    assert _checkpoint(directory, digest) == expected
    assert set(expected["files"]) == {"manifest.json", "state.pt"}
    assert state.is_file() and not state.with_suffix(".pt.gz").exists()
