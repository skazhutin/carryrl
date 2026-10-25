"""Teacher rescue links original captures without loading or modifying a model."""

import gzip
import hashlib
import importlib.util
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_trajectory_diagnostics import capture, config

from carryrl.agent.actions import CODEC_VERSION, PhysicalAction
from carryrl.dashboard.frozen_playback import write_frozen_playback
from carryrl.envs.atomic import AtomicEnv
from carryrl.envs.atomic_suite import SCORING_VERSION, SUITE_VERSION
from carryrl.evaluation.paired import episode_descriptor
from carryrl.evaluation.trajectory_diagnostics import diagnose_frozen_trajectory
from carryrl.training.run import atomic_json, file_sha256
from carryrl.training.sequence import sequence_evaluation_protocol

spec = importlib.util.spec_from_file_location(
    "teacher_rescue_script", Path(__file__).parents[1] / "scripts/audit_teacher_rescue.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


@pytest.fixture
def source(tmp_path, monkeypatch):
    run = tmp_path / "source"
    run.mkdir()
    checkpoint = tmp_path / "runs/checkpoint/checkpoints/rollout_0000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"fixture: never deserialize a model")
    manifest = {"sha256": file_sha256(checkpoint / "state.pt"), "codec_version": CODEC_VERSION,
                "agent_sha256": "1" * 64, "reward_sha256": "2" * 64, "weights_revision": 1}
    atomic_json(checkpoint / "manifest.json", manifest)
    cfg = config(level_bank_seed=20261008)
    cfg = replace(cfg, agent=replace(cfg.agent, text_encoder="multilingual_e5_small"),
                  episode_instructions={"navigation": (
                      "Доберись до зелёного выхода.", "Reach the green exit.")})
    record, frames, actions = capture(cfg, input_change=lambda *_: PhysicalAction())
    record.update(action_seed=cfg.eval_seed, split="validation",
                  instruction=script._episode_instruction(cfg, "navigation", record["seed"]))
    diagnostic = diagnose_frozen_trajectory(cfg, record, frames, actions)
    path = write_frozen_playback(run, record, frames, actions, epoch=0,
        agent_sha256=manifest["agent_sha256"], weights_revision=1, checkpoint=checkpoint,
        manifest=manifest, evaluation_reference=run / "evaluations/frozen_full.json")
    diagnostic.update(agent_sha256=manifest["agent_sha256"], weights_revision=1)
    diagnostic_path = run / "diagnostic.json"
    atomic_json(diagnostic_path, diagnostic)
    selected = {"episode": {**episode_descriptor(record), "split": "validation",
                            "level_bank_seed": cfg.level_bank_seed},
                "outcome": record["outcome"], "playback": str(path),
                "diagnostic": str(diagnostic_path)}
    captures = {"requested_scope": "failed", "enabled": True, "captured_count": 1,
                "selected_episodes": [selected]}
    protocol = sequence_evaluation_protocol(cfg)
    # Only level0 is the real controlled failure. Other records test cohort admission;
    # they are synthetic fixture outcomes, never asserted as learned-agent results.
    records = []
    env = AtomicEnv("navigation", split="validation", level_bank_seed=cfg.level_bank_seed)
    try:
        for policy in protocol["policies"]:
            for index in range(protocol["counts"]["navigation"]):
                rgb, info = env.reset(seed=1_000_000 + index, options={"level_index": index})
                row = deepcopy(record)
                row.update(policy=policy, episode_index=index, level_index=index,
                    seed=1_000_000 + index, action_seed=cfg.eval_seed + index,
                    level_id=info["level_id"], semantic_level_id=info["semantic_level_id"],
                    runtime_fingerprint=info["runtime_fingerprint"],
                    instruction=script._episode_instruction(cfg, "navigation", 1_000_000 + index),
                    initial_frame_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                if policy != "frozen_deterministic" or index:
                    row["outcome"] = {"known": True, "success": True, "failure": False}
                records.append(row)
    finally:
        env.close()
    evaluation = {"records": records, "playback_capture": captures, "split": "validation",
        "policy_and_reward_frozen": True, "suite_version": SUITE_VERSION,
        "scoring_version": SCORING_VERSION, "protocol_sha256": script._json_hash(protocol),
        "agent_sha256": manifest["agent_sha256"], "reward_sha256": manifest["reward_sha256"],
        "weights_revision": 1}
    metadata = {"kind": "frozen_imitation_diagnostics", "updates": 0, "split": "validation",
        "config": cfg.to_dict(), "protocol": protocol,
        "protocol_sha256": script._json_hash(protocol),
        "checkpoint": str(checkpoint), "checkpoint_files_sha256": {
            p.name: file_sha256(p) for p in checkpoint.iterdir()}, "playback_capture": captures}
    status = {"status": "completed", "updates": 0, "checkpoint_files_unchanged": True,
              "policy_and_reward_frozen": True, "playback_capture": captures}
    for target, value in ((run / "metadata.json", metadata), (run / "status.json", status),
                          (run / "evaluation.json", evaluation),
                          (run / "evaluations/frozen_full.json", evaluation)):
        atomic_json(target, value)
    monkeypatch.setattr(script, "provenance", lambda: {"source_and_lock_sha256": "test"})
    return run


def test_source_and_cli_replay_actual_failure_without_loading_checkpoint(tmp_path, source):
    prepared = script.prepare_source(source)
    # Serialized lists and in-memory tuples describe the same linguistic protocol.
    protocol = sequence_evaluation_protocol(prepared.config)
    assert prepared.metadata["protocol"] != protocol
    assert script._json_hash(prepared.metadata["protocol"]) == script._json_hash(protocol)
    assert len(prepared.entries) == 1
    original = dict(prepared.file_hashes)
    output = tmp_path / "rescue"
    assert script.main(["--source-run", str(source), "--output", str(output)]) == 0
    report = json.loads((output / "aggregate.json").read_text())
    assert report["status"] == "completed" and report["training_updates"] == 0
    assert report["selected_episodes"] == 1
    assert report["by_anchor"]["step_0"]["rescued_success"] == 1
    assert report["by_anchor"]["step_40"]["rescued_success"] == 1
    episode = json.loads(Path(report["published"][0]["report"]).read_text())
    with np.load(Path(report["published"][0]["report"]).parent / "suffix_frames.npz") as saved:
        for attempt in episode["attempts"]:
            for index, expected in enumerate(attempt["frame_sha256"]):
                rgb = saved[f"{attempt['anchor_id']}_frame_{index:03d}"]
                assert hashlib.sha256(rgb.tobytes()).hexdigest() == expected
    assert {path: file_sha256(Path(path)) for path in original} == original


def test_checkpoint_mismatch_rejected_without_loading_or_output(tmp_path, source):
    checkpoint = Path(json.loads((source / "metadata.json").read_text())["checkpoint"])
    (checkpoint / "state.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        script.main(["--source-run", str(source), "--output", str(tmp_path / "unused")])
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("missing", ["manifest.json", "state.pt"])
def test_unpinned_checkpoint_identity_is_rejected(source, missing):
    path = source / "metadata.json"
    data = json.loads(path.read_text())
    del data["checkpoint_files_sha256"][missing]
    atomic_json(path, data)
    with pytest.raises(ValueError, match="pinned checkpoint state and manifest"):
        script.prepare_source(source)


def test_changed_serialized_instruction_is_not_accepted_as_same_protocol(source):
    path = source / "metadata.json"
    data = json.loads(path.read_text())
    data["protocol"]["episode_instructions"]["navigation"][0] = "Different goal."
    atomic_json(path, data)
    with pytest.raises(ValueError, match="source protocol/config differs"):
        script.prepare_source(source)


def test_partial_frozen_cohort_is_rejected(source):
    for path in (source / "evaluation.json", source / "evaluations/frozen_full.json"):
        data = json.loads(path.read_text())
        data["records"].pop()
        atomic_json(path, data)
    with pytest.raises(ValueError, match="complete ordered cohort"):
        script.prepare_source(source)


def test_corrupt_actual_actions_are_rejected_before_rescue(source):
    path = source / "diagnostic.json"
    data = json.loads(path.read_text())
    data["actual_actions"][0] = PhysicalAction(keys=("left",)).to_dict()
    atomic_json(path, data)
    prepared = script.prepare_source(source)
    with pytest.raises(ValueError, match="RGB/actions differ"):
        script.load_frames(prepared.entries[0][1])


def test_suffix_publication_failure_is_not_reported_success(tmp_path, source, monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(script.np, "savez_compressed", fail)
    output = tmp_path / "failed"
    with pytest.raises(OSError, match="disk full"):
        script.main(["--source-run", str(source), "--output", str(output)])
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert not (output / "aggregate.json").exists()


def test_cold_checkpoint_source_and_recheck_restore_original_pins(source, monkeypatch):
    checkpoint = Path(json.loads((source / "metadata.json").read_text())["checkpoint"])
    monkeypatch.setattr("carryrl.training.checkpoint_storage.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=64 * 1024**3))
    monkeypatch.setattr("torch.load", lambda *a, **k: pytest.fail("must not load weights"))
    state = checkpoint / "state.pt"
    original = state.read_bytes()
    for prepared in (None, "recheck"):
        state.with_suffix(".pt.gz").write_bytes(gzip.compress(original))
        state.unlink()
        if prepared is None:
            authenticated = script.prepare_source(source)
        else:
            authenticated.assert_unchanged()
        assert state.read_bytes() == original
        assert not state.with_suffix(".pt.gz").exists()
