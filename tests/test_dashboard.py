"""Artifact adapters preserve metric semantics and survive concurrent writers."""

import io
import json

import numpy as np
import pytest
from PIL import Image

from carryrl.dashboard import data as dashboard_data
from carryrl.dashboard.data import RunReader


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def run(tmp_path):
    path = tmp_path / "runs" / "pilot"
    path.mkdir(parents=True)
    write_json(path / "metadata.json", {
        "kind": "sequence_training",
        "created_utc": "2026-10-08T11:29:07+00:00", "config": {
            "tasks": ["navigation", "mouse"], "episodes_per_task": 2,
            "total_rollouts": 32, "max_wall_seconds": 1800, "device": "cpu", "seed": 0,
        },
    })
    write_json(path / "status.json", {"status": "running", "rollouts": 1, "decisions": 3})
    return path


def training_row():
    return {"rollout": 1, "decisions": 3, "cycle_wall_seconds": 5,
            "epochs": [{"policy_loss": -0.01, "value_loss": 4},
                       {"policy_loss": -0.02, "value_loss": 2}],
            "episode_records": [
                {"task": "navigation", "reward_sum": 9, "steps": 1,
                 "outcome": {"known": True, "success": True, "failure": False}},
                {"task": "mouse", "reward_sum": -3, "steps": 2,
                 "outcome": {"known": False, "success": True, "failure": False}},
            ]}


def archive(path, task="navigation", episode=0):
    destination = path / "trajectories" / "rollout_0002" / f"{task}_{episode}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 0] = 120
    np.savez_compressed(destination, final_frame=frame, frames=frame[None])
    write_json(destination.with_suffix(".json"), {"task": task, "actions": [
        {"keys": ["w"], "mouse_buttons": [], "dx": 1, "dy": 0, "wheel": 0}]})
    return destination


def test_incremental_partial_record_uses_last_epoch_and_equal_episode_reward(run):
    reader = RunReader(run.parent)
    log = run / "training.jsonl"
    first = json.dumps(training_row()).encode()
    second = json.dumps(training_row() | {"rollout": 2, "decisions": 7}).encode()
    log.write_bytes(first + b"\n" + second[:20])
    snapshot = reader.snapshot("pilot")
    assert len(snapshot["metrics"]) == 1
    assert snapshot["metrics"][0]["policy_loss"] == -.02
    assert snapshot["metrics"][0]["value_loss"] == 2
    assert snapshot["metrics"][0]["reward_mean"] == 3
    assert snapshot["metrics"][0]["train_successes"] == 1
    assert snapshot["metrics"][0]["entropy"] is None
    mouse = snapshot["tasks"][1]
    assert mouse["train_successes"] == 0
    assert mouse["train_unknown"] == 1
    with log.open("ab") as stream:
        stream.write(second[20:] + b"\n")
    assert len(reader.snapshot("pilot")["metrics"]) == 2
    assert len(reader.snapshot("pilot")["metrics"]) == 2  # No duplicate append.
    log.write_text(json.dumps(training_row()) + "\n")  # Truncated/restarted writer.
    assert len(reader.snapshot("pilot")["metrics"]) == 1


def test_archive_final_frame_and_action_preserve_source_without_loading_frames(run, monkeypatch):
    archive(run)
    reader = RunReader(run.parent)
    snapshot = reader.snapshot("pilot", task="navigation")
    assert snapshot["frame"]["source"] == "last_episode"
    assert snapshot["frame"]["action"]["keys"] == ["w"]
    assert snapshot["progress"]["source"] == "inferred"
    assert snapshot["progress"]["episode_step"] is None
    assert len(snapshot["tasks"]) == 2  # Selecting an image does not filter metrics.
    assert snapshot["frame"]["rollout"] == 2
    png = reader.frame_png("pilot", "navigation")
    with Image.open(io.BytesIO(png)) as image:
        assert image.getpixel((0, 0)) == (120, 0, 0)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("frame cache missed"))
    assert reader.frame_png("pilot", "navigation") == png
    assert reader.frame_png("pilot", "mouse") is None


def test_inflight_archive_is_retried_when_replaced_with_valid_zip(run):
    destination = archive(run)
    destination.write_bytes(b"PK unfinished write")
    reader = RunReader(run.parent)
    assert reader.frame_png("pilot") is None
    archive(run)
    assert reader.frame_png("pilot") is not None


@pytest.mark.parametrize("identifier", ["../pilot", "..", ".", "/tmp", "pilot/..", "p%2Fq"])
def test_run_identifier_rejects_traversal(run, identifier):
    with pytest.raises(ValueError):
        RunReader(run.parent).snapshot(identifier)


def test_symlink_directory_and_artifact_escapes_are_rejected(run, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (run.parent / "escape").symlink_to(outside, target_is_directory=True)
    reader = RunReader(run.parent)
    with pytest.raises(ValueError):
        reader.snapshot("escape")
    write_json(outside / "secret.json", {"status": "sensitive"})
    (run / "status.json").unlink()
    (run / "status.json").symlink_to(outside / "secret.json")
    assert reader.snapshot("pilot")["run"]["status"] == "unknown"
    assert [row["id"] for row in reader.list_runs()["runs"]] == ["pilot"]


def test_unknown_fields_stay_null_and_legacy_run_is_not_listed(run):
    other = run.parent / "legacy"
    write_json(other / "metadata.json", {"config": {"device": "cpu"}})
    snapshot = RunReader(run.parent).snapshot("pilot")
    assert snapshot["run"]["process_alive"] is None
    assert snapshot["reward"] is None
    assert snapshot["frame"] is None
    assert snapshot["tasks"][0]["train_mean_reward"] is None
    assert snapshot["tasks"][0]["last_eval"] is None
    assert snapshot["progress"]["stage"] == "bootstrap"


def test_evaluation_counts_are_independent_and_scores_are_not_rewards(run):
    def summary(successes, score):
        return {"episodes_started": 4, "outcomes": {"success": successes},
                "mean_visible_final_score": score}
    agent, random = summary(1, 1.5), summary(2, 2.5)
    agent["tasks"] = {"navigation": summary(1, 1.5)}
    random["tasks"] = {"navigation": summary(2, 2.5)}
    write_json(run / "evaluations" / "rollout_0001.json", {
        "rollout": 1, "decisions": 3, "summaries": {
            "frozen_deterministic": agent, "generic_physical_random": random}})
    snapshot = RunReader(run.parent).snapshot("pilot")
    evaluation = snapshot["evaluations"][0]
    assert evaluation["agent_successes"] == 1
    assert evaluation["random_successes"] == 2
    assert snapshot["tasks"][0]["last_eval"]["mean_visible_score"] == 1.5


def test_live_telemetry_beats_inference_but_task_filter_uses_archive(run):
    archive(run, "mouse")
    write_json(run / "live.json", {"phase": "collect", "task": "navigation",
                                 "rollout": 2, "episode": 1, "episode_step": 12,
                                 "decisions": 9, "updated_utc": "2026-10-08T12:00:00+00:00"})
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(run / "live_frame.png")
    reader = RunReader(run.parent)
    snapshot = reader.snapshot("pilot")
    assert snapshot["progress"]["source"] == "telemetry"
    assert snapshot["progress"]["episode_step"] == 12
    assert snapshot["run"]["decisions"] == 9
    assert snapshot["frame"]["source"] == "live"
    assert reader.snapshot("pilot", "mouse")["frame"]["source"] == "last_episode"


def test_metrics_without_epochs_have_unknown_loss(run):
    row = training_row()
    row.pop("epochs")
    (run / "training.jsonl").write_text(json.dumps(row) + "\n")
    assert RunReader(run.parent).snapshot("pilot")["metrics"][0]["policy_loss"] is None


@pytest.mark.parametrize("cap", ["_MAX_RECORDS_PER_REFRESH", "_MAX_BYTES_PER_REFRESH"])
def test_refresh_caps_continue_unchanged_file_next_poll(run, monkeypatch, cap):
    monkeypatch.setattr(dashboard_data, cap, 1)
    rows = [training_row() | {"rollout": index + 1} for index in range(3)]
    (run / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    reader = RunReader(run.parent)
    assert len(reader.snapshot("pilot")["metrics"]) == 1
    assert len(reader.snapshot("pilot")["metrics"]) == 2
    assert len(reader.snapshot("pilot")["metrics"]) == 3
    assert len(reader.snapshot("pilot")["metrics"]) == 3


def test_per_rollout_task_history_preserves_unknown_denominator(run):
    (run / "training.jsonl").write_text(json.dumps(training_row()) + "\n")
    metric = RunReader(run.parent).snapshot("pilot")["metrics"][0]
    assert metric["tasks"]["navigation"]["successes"] == 1
    assert metric["tasks"]["mouse"]["episodes"] == 1
    assert metric["tasks"]["mouse"]["successes"] == 0
    assert metric["tasks"]["mouse"]["unknown"] == 1
    assert metric["train_unknown"] == 1


def test_run_listing_excludes_standalone_evaluation_and_keeps_bootstrap_training(run):
    root = run.parent
    metadata = json.loads((run / "metadata.json").read_text())
    config = metadata["config"]
    write_json(root / "evaluation" / "metadata.json", {
        "config": config, "checkpoint": str(run / "checkpoints" / "rollout_0001"), "updates": 0})
    write_json(root / "evaluation" / "status.json", {"status": "completed"})
    write_json(root / "bootstrap" / "metadata.json", {
        "kind": "sequence_training", "created_utc": metadata["created_utc"], "config": config})
    write_json(root / "bootstrap" / "status.json", {"status": "running", "rollouts": 0,
                                                  "decisions": 0})
    write_json(root / "legacy_training" / "metadata.json", {
        "created_utc": metadata["created_utc"], "config": config, "provenance": {},
        "optimizer": "AdamW"})
    reader = RunReader(root)
    assert {row["id"] for row in reader.list_runs()["runs"]} == {
        "pilot", "bootstrap", "legacy_training"}


def legacy_final_evaluation(run):
    checkpoint = run / "checkpoints" / "rollout_0001"
    write_json(checkpoint / "manifest.json", {"rollout": 1, "decisions": 10,
                                            "weights_revision": 2, "agent_sha256": "own-policy"})
    write_json(run / "latest.json", {"checkpoint": str(checkpoint.resolve())})
    agent = {"episodes_started": 4, "outcomes": {"success": 1}, "tasks": {
        "navigation": {"episodes_started": 4, "outcomes": {"success": 1}}}}
    write_json(run / "evaluation.json", {"policy_and_reward_frozen": True,
                                        "summaries": {"frozen_deterministic": agent}})
    write_json(run / "status.json", {"status": "completed", "rollouts": 1, "decisions": 10,
                                     "weights_revision": 2,
                                     "latest_checkpoint": str(checkpoint.resolve())})
    return checkpoint


def test_legacy_final_evaluation_uses_only_own_completed_checkpoint_step(run):
    legacy_final_evaluation(run)
    evaluation = RunReader(run.parent).snapshot("pilot")["evaluations"][0]
    assert evaluation["rollout"] == 1 and evaluation["decisions"] == 10


@pytest.mark.parametrize("mismatch", ["status_decisions", "foreign_checkpoint", "policy_hash"])
def test_ambiguous_legacy_final_evaluation_is_omitted_with_one_warning(run, caplog, mismatch):
    checkpoint = legacy_final_evaluation(run)
    if mismatch == "policy_hash":
        path = run / "evaluation.json"
        value = json.loads(path.read_text()) | {"agent_sha256": "foreign-policy"}
    else:
        path = run / "status.json"
        value = json.loads(path.read_text())
        if mismatch == "status_decisions":
            value["decisions"] = 11
        else:
            value["latest_checkpoint"] = str(checkpoint.parent.parent.parent / "foreign")
    write_json(path, value)
    reader = RunReader(run.parent)
    assert reader.snapshot("pilot")["evaluations"] == []
    assert reader.snapshot("pilot")["evaluations"] == []
    assert caplog.text.count("Omitting evaluation without unambiguous") == 1


def test_episode_multipliers_are_used_for_rollout_progress(run):
    path = run / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["config"]["task_episode_multipliers"] = {"navigation": 3}
    write_json(path, metadata)
    assert RunReader(run.parent).snapshot("pilot")["progress"]["episodes_per_rollout"] == 8
