"""Periodic imitation saves retain exact measured weights without changing final paths."""

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from test_imitation_feature_storage import script

from carryrl.training.progress import TrainingProgress
from carryrl.training.run import atomic_json
from carryrl.training.sequence_config import SequenceRunConfig


def _snapshot_arguments(tmp_path):
    progress = TrainingProgress(tmp_path / "status.json", epochs_total=40)
    progress.phase("imitation", total=400, done=200, unit="optimizer_steps")
    progress.update(force=True, epoch=20, epochs_completed=20, optimizer_steps=200)
    return dict(every=10, max_epochs=40, output=tmp_path, progress=progress,
                agent={"weight": "stale"}, optimizer={"moment": "stale"}, reward=object(),
                config=object(), saved={"decisions": 123, "weights_revision": 7},
                metadata={"source_checkpoint": "/source/checkpoint",
                          "source_checkpoint_sha256": "source archive hash",
                          "initial_agent_sha256": "initial tensor hash"})


def test_snapshot_syncs_weights_and_moments_then_saves_matching_evaluation(monkeypatch, tmp_path):
    args = _snapshot_arguments(tmp_path)
    events = []

    def sync():
        events.append("sync")
        args["agent"]["weight"] = "epoch20"
        args["optimizer"]["moment"] = "epoch20 moment"

    def save(agent, optimizer, reward, config, destination, **counters):
        events.append("save")
        assert agent == {"weight": "epoch20"}
        assert optimizer == {"moment": "epoch20 moment"}
        assert counters == {"rollout": 0, "decisions": 123, "revision": 207}
        assert destination == tmp_path / "checkpoints_by_epoch" / "epoch_0020"
        persisted = json.loads(args["progress"].path.read_text())
        assert persisted["phase"] == "imitation" and persisted["status"] == "running"
        assert persisted["training_action"] == "saving_checkpoint"
        checkpoint = destination / "checkpoints" / "rollout_0000"
        checkpoint.mkdir(parents=True)
        atomic_json(checkpoint / "manifest.json", {
            "sha256": "archive hash", "agent_sha256": "epoch20 hash",
            "reward_sha256": "frozen reward hash", "weights_revision": 207, "rollout": 0})
        return checkpoint

    monkeypatch.setattr(script, "save_sequence_checkpoint", save)
    args["engine"] = SimpleNamespace(sync_to_torch=sync)
    evaluated = tmp_path / "evaluations" / "epoch_0020.json"
    evaluated.parent.mkdir()
    atomic_json(evaluated, {"agent_sha256": "epoch20 hash",
                           "reward_sha256": "frozen reward hash", "weights_revision": 207})
    checkpoint = script._save_epoch_snapshot(
        {"complete": True, "epoch": 20, "optimizer_steps": 200}, **args)
    assert events == ["sync", "save"]
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["epoch"] == manifest["epochs_completed"] == 20
    assert manifest["optimizer_steps"] == 200 and manifest["rollout"] == 0
    assert manifest["source_checkpoint_sha256"] == "source archive hash"
    assert manifest["sha256"] == "archive hash"
    assert manifest["evaluation_reference"] == str(evaluated.resolve())
    assert args["progress"].state["training_action"] == "learning"
    assert args["progress"].state["last_epoch_checkpoint"] == str(checkpoint)
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "latest.json").exists()


@pytest.mark.parametrize(("every", "epoch", "complete"), [
    (0, 20, True), (10, 21, True), (10, 20, False), (10, 40, True)])
def test_disabled_noninterval_incomplete_and_final_epochs_do_no_extra_work(
        monkeypatch, tmp_path, every, epoch, complete):
    args = _snapshot_arguments(tmp_path)
    args["every"] = every

    def unexpected(*args, **kwargs):
        pytest.fail("no snapshot synchronization or save expected")

    args["engine"] = SimpleNamespace(sync_to_torch=unexpected)
    monkeypatch.setattr(script, "save_sequence_checkpoint", unexpected)
    before = args["progress"].path.read_bytes()
    assert script._save_epoch_snapshot(
        {"epoch": epoch, "complete": complete, "optimizer_steps": 200}, **args) is None
    assert args["progress"].path.read_bytes() == before
    assert not (tmp_path / "checkpoints_by_epoch").exists()


@pytest.mark.parametrize("failed_stage", ["sync", "save"])
def test_snapshot_failure_persists_original_error_without_recovery_retry(
        monkeypatch, tmp_path, failed_stage):
    args = _snapshot_arguments(tmp_path)
    error = OSError("snapshot disk or sync failure")
    events = []

    def attempt(stage):
        events.append(stage)
        if stage == failed_stage:
            raise error

    args["engine"] = SimpleNamespace(sync_to_torch=lambda: attempt("sync"))
    monkeypatch.setattr(script, "save_sequence_checkpoint", lambda *a, **k: attempt("save"))
    with pytest.raises(OSError) as raised:
        try:
            script._save_epoch_snapshot(
                {"epoch": 20, "complete": True, "optimizer_steps": 200}, **args)
        except BaseException as original:
            script._record_training_failure(
                original, progress=args["progress"],
                sync_to_torch=lambda: pytest.fail("must not retry sync"),
                save_checkpoint=lambda: pytest.fail("must not retry save"))
            raise
    assert raised.value is error
    assert events == (["sync"] if failed_stage == "sync" else ["sync", "save"])
    persisted = json.loads(args["progress"].path.read_text())
    assert persisted["status"] == "failed" and persisted["error"] == str(error)
    assert persisted["phase"] == "imitation" and persisted["optimizer_steps"] == 200
    assert "recovery_checkpoint" not in persisted
    assert "error_notes" not in persisted


@pytest.mark.parametrize("value", ["-1", "not-an-integer"])
def test_cli_rejects_invalid_checkpoint_interval_before_config_load(monkeypatch, capsys, value):
    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "missing.json",
        "--warm-start", "missing", "--output", "unused", "--checkpoint-every", value])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
    assert "checkpoint" in capsys.readouterr().err


def test_cli_checkpoint_interval_defaults_to_disabled(monkeypatch):
    parsed = []
    parse = script.argparse.ArgumentParser.parse_args

    def capture(parser, *args, **kwargs):
        result = parse(parser, *args, **kwargs)
        parsed.append(result)
        return result

    def stop_before_loading(_):
        raise RuntimeError("stop before model loading")

    monkeypatch.setattr(script.argparse.ArgumentParser, "parse_args", capture)
    monkeypatch.setattr(script.SequenceRunConfig, "load", stop_before_loading)
    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "missing.json",
        "--warm-start", "missing", "--output", "unused"])
    with pytest.raises(RuntimeError, match="stop before model loading"):
        script.main()
    assert parsed[0].checkpoint_every == 0


@pytest.mark.parametrize("mismatched_key", ["agent_sha256", "reward_sha256", "weights_revision",
                                           "missing_reward_sha256"])
def test_snapshot_refuses_to_link_different_measured_weights(monkeypatch, tmp_path, mismatched_key):
    args = _snapshot_arguments(tmp_path)

    def save(*values, **counters):
        checkpoint = values[4] / "checkpoints" / "rollout_0000"
        checkpoint.mkdir(parents=True)
        atomic_json(checkpoint / "manifest.json", {
            "agent_sha256": "snapshot hash", "reward_sha256": "reward hash",
            "weights_revision": counters["revision"]})
        return checkpoint

    monkeypatch.setattr(script, "save_sequence_checkpoint", save)
    evaluation = tmp_path / "evaluations" / "epoch_0020.json"
    evaluation.parent.mkdir()
    evaluated = {"agent_sha256": "snapshot hash", "reward_sha256": "reward hash",
                 "weights_revision": 207}
    if mismatched_key == "missing_reward_sha256":
        del evaluated["reward_sha256"]
    else:
        evaluated[mismatched_key] = "different identity"
    atomic_json(evaluation, evaluated)
    with pytest.raises(RuntimeError, match="does not match its frozen evaluation"):
        try:
            script._save_epoch_snapshot(
                {"complete": True, "epoch": 20, "optimizer_steps": 200}, **args)
        except RuntimeError as error:
            script._record_training_failure(error, progress=args["progress"])
            raise
    persisted = json.loads(args["progress"].path.read_text())
    assert persisted["status"] == "failed"
    assert persisted["training_action"] == "saving_checkpoint"
    checkpoint = tmp_path / "checkpoints_by_epoch" / "epoch_0020" / "checkpoints" / "rollout_0000"
    assert persisted["last_epoch_checkpoint"] == str(checkpoint)
    assert persisted["last_checkpoint_evaluation_match"] == "pending"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["epoch"] == 20 and "evaluation_reference" not in manifest


def test_training_failure_persists_secondary_exception_notes(tmp_path):
    progress = TrainingProgress(tmp_path / "status.json", epochs_total=1)
    error = OSError("snapshot save failed")
    error.add_note("Final MLX synchronization failed: RuntimeError: cleanup bridge failed")
    script._record_training_failure(error, progress=progress)
    persisted = json.loads(progress.path.read_text())
    assert persisted["error"] == "snapshot save failed"
    assert persisted["error_type"] == "OSError"
    assert persisted["error_notes"] == error.__notes__


@pytest.mark.parametrize("close_fails", [False, True])
def test_main_closes_writer_once_before_reporting_completion(monkeypatch, tmp_path, close_fails):
    output, source = tmp_path / "run", tmp_path / "source"
    source.mkdir()
    atomic_json(source / "manifest.json", {"sha256": "source archive hash"})
    config = SequenceRunConfig(tasks=("navigation",))
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(script, "provenance", lambda: {})

    class Episode:
        frame_indices = (0,)
        final_frame_index = 0
        presentation = None
        native_size = None
        source = "scripted_teacher"

        def __len__(self):
            return 1

    dataset = SimpleNamespace(
        episodes=[Episode()], frames=[np.zeros((4, 4, 3), dtype=np.uint8)],
        provenance=lambda: {"source": "stub train-only RGB"})
    agent = SimpleNamespace(vision=object(), summary_query=torch.zeros(1),
                            encode_frames=lambda _: torch.zeros(1, 1, 384))
    optimizer, reward = SimpleNamespace(param_groups=[{}]), object()
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k:
        (agent, optimizer, reward, {"decisions": 123, "weights_revision": 7}))
    monkeypatch.setattr(script, "tensor_hash", lambda _: "stable tensor hash")
    monkeypatch.setattr(script, "collect_scripted_demonstrations", lambda *a, **k: dataset)
    monkeypatch.setattr(script, "evaluate_sequence", lambda *a, **k: {"summaries": {
        "frozen_deterministic": {"outcomes": {"success": 0}, "episodes_started": 1,
                                 "tasks": {"navigation": {"success_rate_lower": 0.0}}}}})
    monkeypatch.setattr(script, "link_matching_playbacks", lambda *a: None)

    def train(*args, **kwargs):
        kwargs["on_update"]({"epoch": 1, "optimizer_steps": 1, "nll": 1.0,
                              "gradient_norm": 0.0, "seconds": 0.01})
        kwargs["on_epoch"]({"epoch": 1, "complete": True, "optimizer_steps": 1,
                             "nll": 1.0, "seconds": 0.01, "epoch_training_seconds": 0.01,
                             "episodes": 1, "per_task_nll": {"navigation": 1.0}})
        return {"optimizer_steps": 1, "stop_reason": "completed"}

    monkeypatch.setattr(script, "train_imitation", train)
    save_calls = []

    def save(*args, **kwargs):
        save_calls.append("save")
        checkpoint = args[4] / "checkpoints" / "rollout_0000"
        checkpoint.mkdir(parents=True)
        (checkpoint / "state.pt").write_bytes(b"stub checkpoint retained")
        atomic_json(checkpoint / "manifest.json", {"sha256": "stub archive hash"})
        atomic_json(args[4] / "latest.json", {"checkpoint": str(checkpoint)})
        return checkpoint

    monkeypatch.setattr(script, "save_sequence_checkpoint", save)
    close_calls = []
    close_error = OSError("writer close failed")

    class Writer:
        def __init__(self, path):
            pass

        def add_scalar(self, *args):
            pass

        def flush(self):
            pass

        def close(self):
            close_calls.append("close")
            state = json.loads((output / "status.json").read_text())
            assert state["status"] == "running" and state["phase"] == "saving"
            assert not any(row["phase"] == "finished" for row in state["phase_history"])
            if close_fails:
                raise close_error

    tensorboard = ModuleType("torch.utils.tensorboard")
    tensorboard.SummaryWriter = Writer
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tensorboard)
    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "unused.json",
        "--warm-start", str(source), "--output", str(output), "--training-backend", "torch",
        "--epochs", "1", "--tensorboard", str(tmp_path / "tensorboard")])
    if close_fails:
        with pytest.raises(OSError, match="writer close failed") as raised:
            script.main()
        assert raised.value is close_error
    else:
        script.main()
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == ("failed" if close_fails else "completed")
    assert status["phase"] == ("saving" if close_fails else "finished")
    assert status["optimizer_steps"] == 1
    if close_fails:
        assert status["error"] == str(close_error) and status["error_type"] == "OSError"
        assert "recovery_checkpoint" not in status
    assert close_calls == ["close"] and save_calls == ["save"]
    checkpoint = output / "checkpoints" / "rollout_0000"
    assert status["checkpoint"] == str(checkpoint)
    assert (checkpoint / "state.pt").read_bytes() == b"stub checkpoint retained"


def test_main_real_frozen_evaluation_and_checkpoint_identities_match(monkeypatch, tmp_path):
    """Exercise report generation and archive saving together, with tiny CPU modules."""
    from test_imitation import make_agent

    from carryrl.training.sequence import tensor_hash

    agent = make_agent()
    reward = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(parameter for parameter in agent.parameters()
                                  if parameter.requires_grad)
    config = SequenceRunConfig(agent=agent.config, tasks=("navigation",),
                               max_episode_steps=1, eval_episodes_per_task=1)
    source, output = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    atomic_json(source / "manifest.json", {"sha256": "source archive hash"})
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(script, "provenance", lambda: {})
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k:
        (agent, optimizer, reward, {"decisions": 123, "weights_revision": 7}))

    class Episode:
        frame_indices = (0,)
        final_frame_index = 0
        presentation = None
        native_size = None
        source = "scripted_teacher"

        def __len__(self):
            return 1

    dataset = SimpleNamespace(
        episodes=[Episode()], frames=[np.zeros((4, 4, 3), dtype=np.uint8)],
        provenance=lambda: {"source": "stub train-only RGB"})
    monkeypatch.setattr(script, "collect_scripted_demonstrations", lambda *a, **k: dataset)
    monkeypatch.setattr(script, "write_frozen_playback", lambda *a, **k: None)
    # Keep real evaluate_sequence, tensor_hash and save_sequence_checkpoint. The training
    # stub performs real CPU optimizer updates and invokes the production epoch callbacks.
    def train(*args, **kwargs):
        for epoch in (1, 2):
            optimizer.zero_grad()
            agent.decoder.key_head.weight.square().sum().backward()
            optimizer.step()
            kwargs["on_update"]({"epoch": epoch, "optimizer_steps": epoch, "nll": 1.0,
                                  "gradient_norm": 0.0, "seconds": 0.01})
            kwargs["on_epoch"]({"epoch": epoch, "complete": True, "optimizer_steps": epoch,
                                 "nll": 1.0, "seconds": 0.01, "epoch_training_seconds": 0.01,
                                 "episodes": 1, "per_task_nll": {"navigation": 1.0}})
        return {"optimizer_steps": 2, "stop_reason": "completed"}

    monkeypatch.setattr(script, "train_imitation", train)
    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "unused.json",
        "--warm-start", str(source), "--output", str(output), "--training-backend", "torch",
        "--epochs", "2", "--eval-every", "1", "--checkpoint-every", "1"])
    script.main()
    evaluation = json.loads((output / "evaluations" / "epoch_0001.json").read_text())
    checkpoint = output / "checkpoints_by_epoch" / "epoch_0001" / "checkpoints" / "rollout_0000"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    archive = torch.load(checkpoint / "state.pt", weights_only=True)
    assert evaluation["policy_and_reward_frozen"] is True
    assert evaluation["split"] == "validation" and len(evaluation["records"]) == 2
    for key in ("agent_sha256", "reward_sha256", "weights_revision"):
        assert evaluation[key] == manifest[key]
    assert evaluation["reward_sha256"] == tensor_hash(reward)
    assert evaluation["weights_revision"] == archive["weights_revision"] == 8
    assert manifest["epoch"] == manifest["optimizer_steps"] == 1
    assert manifest["evaluation_reference"] == str(
        (output / "evaluations" / "epoch_0001.json").resolve())
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "completed" and status["last_epoch_checkpoint"] == str(checkpoint)
    assert status["last_checkpoint_evaluation_match"] == "matched"
