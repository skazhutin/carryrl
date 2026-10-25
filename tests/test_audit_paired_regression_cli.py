"""Paired regression CLI pins sources and fails closed around all four captures."""

import copy
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from carryrl.agent.actions import CODEC_VERSION, PhysicalAction
from carryrl.envs.atomic import AtomicEnv, verify_visible_outcome
from carryrl.envs.atomic_suite import SCORING_VERSION, SUITE_VERSION
from carryrl.training.imitation import scripted_teacher
from carryrl.training.run import file_sha256
from carryrl.training.sequence import sequence_evaluation_protocol, tensor_hash
from carryrl.training.sequence_config import SequenceRunConfig

spec = importlib.util.spec_from_file_location(
    "audit_paired_regression_cli", Path(__file__).parents[1] / "scripts/audit_paired_regression.py")
script = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = script
spec.loader.exec_module(script)


class ToyActor(nn.Module):
    def __init__(self, successful):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor([1.0 if successful else 2.0]))
        self.successful = successful
        self.reset_episode()

    def reset_episode(self, **kwargs):
        self.teacher = scripted_teacher("interaction")

    def act(self, frame, deterministic=True):
        assert deterministic
        action = self.teacher.act(frame) if self.successful else PhysicalAction(keys=("left",))
        return SimpleNamespace(action=action)

    def observe_action(self, action):
        self.teacher.observe_action(action)


def _record(config, policy, index):
    env = AtomicEnv("interaction", split="validation", image_size=96,
                    max_steps=80, level_bank_seed=config.level_bank_seed, difficulty=1.0)
    try:
        frame, info = env.reset(seed=1_000_000 + index, options={"level_index": index})
        return {"task": "interaction", "seed": 1_000_000 + index, "instruction": None,
            "policy": policy, "episode_index": index, "difficulty": 1.0,
            "action_seed": config.eval_seed + index, "level_index": index,
            "level_id": info["level_id"], "semantic_level_id": info.get("semantic_level_id"),
            "runtime_fingerprint": info.get("runtime_fingerprint"),
            "presentation": info.get("presentation"), "level_bank_seed": config.level_bank_seed,
            "initial_frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "steps": 80, "terminated": False, "truncated": True,
            "outcome": {"known": True, "success": False, "failure": False, "score": 0}}
    finally:
        env.close()


def _actual_reference(config, actor, row):
    env = AtomicEnv(row["task"], split="validation", image_size=96,
                    max_steps=80, level_bank_seed=config.level_bank_seed, difficulty=1.0)
    try:
        frame, _ = env.reset(seed=row["seed"], options={"level_index": row["level_index"]})
        actor.reset_episode()
        for step in range(80):
            action = actor.act(frame).action
            frame, _, terminated, truncated, info = env.step(action)
            actor.observe_action(action)
            if terminated or truncated:
                break
        return {**row, "steps": step + 1, "terminated": terminated, "truncated": truncated,
                "outcome": verify_visible_outcome(frame), "quality": info["quality"]}
    finally:
        env.close()


@pytest.fixture
def audit(tmp_path, monkeypatch):
    config = SequenceRunConfig(tasks=("interaction",), level_bank_seed=20261008,
                               eval_all_levels=True, torch_threads=1)
    protocol = sequence_evaluation_protocol(config)
    rows = [_record(config, policy, index) for policy in protocol["policies"]
            for index in range(protocol["counts"]["interaction"])]
    reward = nn.Linear(1, 1)
    actors = {"before": ToyActor(True), "after": ToyActor(False)}
    paths, loaded, phases = {}, [], []
    for role in ("before", "after"):
        root = tmp_path / "runs" / role
        checkpoint = root / "checkpoints/rollout_0000"
        checkpoint.mkdir(parents=True)
        (checkpoint / "state.pt").write_bytes(f"{role} source state".encode())
        manifest = {"sha256": file_sha256(checkpoint / "state.pt"),
            "agent_sha256": tensor_hash(actors[role]), "reward_sha256": tensor_hash(reward),
            "weights_revision": 1 if role == "before" else 2,
            "codec_version": CODEC_VERSION}
        (checkpoint / "manifest.json").write_text(json.dumps(manifest))
        config_path, evaluation_path = root / "config.json", root / "evaluation.json"
        config_path.write_text(json.dumps(config.to_dict()))
        records = copy.deepcopy(rows)
        records[0] = _actual_reference(config, actors[role], records[0])
        evaluation = {"split": "validation", "suite_version": SUITE_VERSION,
            "scoring_version": SCORING_VERSION, "policy_and_reward_frozen": True,
            "level_bank_seed": config.level_bank_seed,
            "protocol_sha256": script._json_hash(protocol), "records": records,
            **{key: manifest[key] for key in ("agent_sha256", "reward_sha256", "weights_revision")}}
        evaluation_path.write_text(json.dumps(evaluation))
        paths[role] = SimpleNamespace(checkpoint=checkpoint, config=config_path,
                                      evaluation=evaluation_path, manifest=manifest)

    def load(checkpoint, actual_config, **kwargs):
        role = checkpoint.parents[1].name
        assert kwargs == {"evaluation_only": True, "load_optimizer": False}
        assert actual_config.eval_all_levels
        loaded.append(role)
        return actors[role], None, reward, {"weights_revision": paths[role].manifest[
            "weights_revision"]}

    monkeypatch.setattr(script, "load_sequence_checkpoint", load)
    monkeypatch.setattr(script, "provenance", lambda: {"source_and_lock_sha256": "code"})
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("no actual torch.load"))
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **kw: pytest.fail("no optimizer"))
    capture = script.capture_frozen_case

    def tracked(*args, **kwargs):
        phases.append(kwargs["role"])
        return capture(*args, **kwargs)

    monkeypatch.setattr(script, "capture_frozen_case", tracked)
    output = tmp_path / "audit"
    args = ["--output", str(output)]
    for role, files in paths.items():
        args += [f"--{role}-checkpoint", str(files.checkpoint),
                 f"--{role}-config", str(files.config),
                 f"--{role}-evaluation", str(files.evaluation)]
    return SimpleNamespace(paths=paths, output=output, args=args, actors=actors, reward=reward,
                           config=config, loaded=loaded, phases=phases)


def _mutate_json(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))


def test_all_lost_wins_capture_four_roles_with_exact_provenance(audit):
    threads = torch.get_num_threads()
    before = {role: script._checkpoint_hashes(path.checkpoint)
              for role, path in audit.paths.items()}
    assert script.main(audit.args) == 0
    assert audit.loaded == ["before", "after"]
    assert audit.phases == ["before", "candidate", "sham", "intervention"]
    assert torch.get_num_threads() == threads
    metadata = json.loads((audit.output / "metadata.json").read_text())
    report = json.loads((audit.output / "aggregate.json").read_text())
    status = json.loads((audit.output / "status.json").read_text())
    assert status["status"] == "completed" and status["optimizer_steps"] == 0
    assert metadata["before"]["legacy_weights_manifest"]
    assert metadata["after"]["legacy_weights_manifest"]
    assert metadata["selected_count"] == 1 and metadata["full_cohort_model_episodes"] == 16
    assert report["overall"] == {"cases": 1, "before_successes": 1, "candidate_successes": 0,
                                  "sham_successes": 0, "intervention_successes": 0}
    case = report["cases"][0]
    assert case["sham_verified"]
    assert case["comparison"]["first_command_difference"]["index"] == case["single_override_index"]
    for role, entry in case["roles"].items():
        trajectory = json.loads(Path(entry["trajectory"]).read_text())
        source = audit.paths["before" if role == "before" else "after"]
        assert trajectory["agent_sha256"] == source.manifest["agent_sha256"]
        assert trajectory["weights_revision"] == source.manifest["weights_revision"]
        assert trajectory["audit_role"] == role
        assert trajectory["learned_policy_alone"] == (role != "intervention")
        assert trajectory["frames_sha256"] == file_sha256(Path(entry["frames"]))
        with np.load(entry["frames"], allow_pickle=False) as frames:
            assert len(frames.files) == trajectory["steps"] + 1
        assert len(trajectory["executed_actions"]) == len(trajectory["proposed_actions"])
        if role == "intervention":
            assert trajectory["policy"] == "diagnostic_single_command_intervention"
    assert {role: script._checkpoint_hashes(path.checkpoint)
            for role, path in audit.paths.items()} == before
    assert all(row["hashes_before"] == row["hashes_after"] for row in status["phase_hashes"])


@pytest.mark.parametrize("damage", ["state", "cohort", "protocol", "revision", "config"])
def test_bad_reference_or_file_rejected_before_loading_or_output(audit, damage):
    source = audit.paths["after"]
    if damage == "state":
        (source.checkpoint / "state.pt").write_bytes(b"damaged checkpoint bytes")
    elif damage == "cohort":
        _mutate_json(source.evaluation, lambda value: value["records"].pop())
    elif damage == "protocol":
        _mutate_json(source.evaluation, lambda value: value.update(protocol_sha256="wrong"))
    elif damage == "revision":
        _mutate_json(source.evaluation, lambda value: value.update(weights_revision=999))
    else:
        _mutate_json(source.config, lambda value: value.update(eval_stochastic=True))
    with pytest.raises(ValueError):
        script.main(audit.args)
    assert not audit.loaded and not audit.output.exists()


def test_no_lost_wins_refuses_before_loading_or_output(audit):
    _mutate_json(audit.paths["before"].evaluation,
                 lambda value: value["records"][0]["outcome"].update(success=False))
    with pytest.raises(ValueError, match="no known lost wins"):
        script.main(audit.args)
    assert not audit.loaded and not audit.output.exists()


@pytest.mark.parametrize("failure", ["sham", "historical", "tensor", "source", "code"])
def test_failed_audit_preserves_partials_without_aggregate(audit, monkeypatch, failure):
    if failure == "sham":
        def reject(*args):
            raise ValueError("sham mismatch")
        monkeypatch.setattr(script, "verify_sham", reject)
    elif failure == "historical":
        _mutate_json(audit.paths["before"].evaluation,
                     lambda value: value["records"][0].update(steps=79))
    elif failure == "code":
        calls = iter(("initial", "changed"))
        monkeypatch.setattr(script, "provenance", lambda: {
            "source_and_lock_sha256": next(calls)})
    else:
        original = script.capture_frozen_case

        def mutate(*args, **kwargs):
            case = original(*args, **kwargs)
            if kwargs["role"] == "before":
                if failure == "tensor":
                    with torch.no_grad():
                        audit.actors["before"].marker.add_(1)
                else:
                    audit.paths["before"].config.write_text("changed source")
            return case

        monkeypatch.setattr(script, "capture_frozen_case", mutate)
    threads = torch.get_num_threads()
    with pytest.raises((ValueError, RuntimeError)):
        script.main(audit.args)
    assert torch.get_num_threads() == threads
    assert json.loads((audit.output / "status.json").read_text())["status"] == "failed"
    assert not (audit.output / "aggregate.json").exists()


def test_existing_and_source_output_are_protected(audit):
    audit.output.mkdir()
    sentinel = audit.output / "preserve"
    sentinel.write_text("user data")
    with pytest.raises(FileExistsError):
        script.main(audit.args)
    assert sentinel.read_text() == "user data" and not audit.loaded
    with pytest.raises(ValueError, match="outside both source"):
        script.main([*audit.args, "--output", str(audit.paths["after"].checkpoint.parent / "new")])


@pytest.mark.parametrize("field", ["suite_version", "scoring_version"])
def test_manifest_declared_wrong_environment_version_is_rejected(audit, field):
    _mutate_json(audit.paths["after"].checkpoint / "manifest.json",
                 lambda value: value.update({field: "wrong.declared.version"}))
    with pytest.raises(ValueError, match="checkpoint codec/suite/scoring"):
        script.main(audit.args)
    assert not audit.loaded and not audit.output.exists()


def test_protocol_expectation_refuses_before_loading_or_output(audit):
    with pytest.raises(ValueError, match="protocol SHA256 differs"):
        script.main([*audit.args, "--expected-protocol-sha256", "wrong"])
    assert not audit.loaded and not audit.output.exists()


def test_packaged_language_hash_checked_without_loading_weights(audit):
    checkpoint = audit.paths["after"].checkpoint
    language = checkpoint / "language"
    language.mkdir()
    (language / "tokenizer.json").write_text("changed tokenizer bytes")
    _mutate_json(checkpoint / "manifest.json", lambda value: value.update(
        language_resources_sha256={"tokenizer.json": "0" * 64}))
    with pytest.raises(ValueError, match="language SHA256 mismatch"):
        script.main(audit.args)
    assert not audit.loaded and not audit.output.exists()


def test_missing_historical_quality_does_not_invent_source_quality(audit):
    for source in audit.paths.values():
        _mutate_json(source.evaluation, lambda value: value["records"][0].pop("quality"))
    assert script.main(audit.args) == 0
    for role in script.ROLES:
        path = audit.output / "cases" / "0000" / role / "trajectory.json"
        record = json.loads(path.read_text())
        assert record["quality"]["commands"] == record["steps"]


def test_cold_checkpoint_source_and_recheck_restore_original_pins(audit, monkeypatch):
    row = audit.paths["before"]
    checkpoint = row.checkpoint
    monkeypatch.setattr("carryrl.training.checkpoint_storage.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=64 * 1024**3))
    monkeypatch.setattr("torch.load", lambda *a, **k: pytest.fail("must not load weights"))
    state = checkpoint / "state.pt"
    original = state.read_bytes()
    for prepared in (None, "recheck"):
        state.with_suffix(".pt.gz").write_bytes(gzip.compress(original))
        state.unlink()
        if prepared is None:
            authenticated = script._source(checkpoint, row.config, row.evaluation, "cpu")
        else:
            authenticated.assert_unchanged()
        assert state.read_bytes() == original
        assert not state.with_suffix(".pt.gz").exists()
