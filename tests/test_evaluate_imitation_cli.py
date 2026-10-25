"""Standalone diagnostics protect artifacts and compare actual frozen cohorts."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from carryrl.training.sequence_config import SequenceRunConfig

spec = importlib.util.spec_from_file_location(
    "evaluate_imitation_script", Path(__file__).parents[1] / "scripts/evaluate_imitation.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def _evaluation():
    return {"split": "validation", "protocol_sha256": "protocol", "suite_version": "suite",
            "scoring_version": "score", "reward_sha256": "reward",
            "policy_and_reward_frozen": True, "records": [{
                "task": "navigation", "seed": 1, "instruction": None,
                "policy": "frozen_deterministic", "action_seed": 2, "episode_index": 0,
                "difficulty": 1, "level_index": 0, "level_id": "level", "presentation": None,
                "initial_frame_sha256": "rgb", "outcome": {"known": True, "success": False}}]}


def test_paired_report_requires_protocol_and_actual_ordered_cohort():
    before, after = _evaluation(), _evaluation()
    after["records"][0]["outcome"]["success"] = True
    assert script._paired_report(before, after)["gained"] == 1
    after["records"][0]["initial_frame_sha256"] = "different frame"
    with pytest.raises(ValueError, match="протокол"):
        script._paired_report(before, after)


@pytest.mark.parametrize("field", ["split", "protocol_sha256", "suite_version",
                                    "scoring_version", "reward_sha256",
                                    "policy_and_reward_frozen"])
def test_paired_report_rejects_missing_metadata(field):
    before = _evaluation()
    before.pop(field)
    with pytest.raises(ValueError):
        script._paired_report(before, _evaluation())


def test_output_protects_nested_snapshot_source_run_and_existing_measurements(tmp_path):
    run = tmp_path / "source"
    checkpoint = run / "checkpoints_by_epoch/epoch_0002/checkpoints/rollout_0000"
    checkpoint.mkdir(parents=True)
    for output in (run, run / "evaluations/new", checkpoint / "diagnostic", tmp_path):
        with pytest.raises(ValueError):
            script._prepare_output(output, checkpoint)
    target = tmp_path / "diagnostic"
    assert script._prepare_output(target, checkpoint) == target
    with pytest.raises(FileExistsError):
        script._prepare_output(target, checkpoint)


def test_cli_frozen_nll_only_restores_threads_and_preserves_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "source/checkpoints/rollout_0000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text(json.dumps({"epoch": 2}))
    (checkpoint / "state.pt").write_bytes(b"unchanged source")
    config = SequenceRunConfig(tasks=("navigation",), device="cpu")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    agent = SimpleNamespace(name="agent")
    reward = SimpleNamespace(name="reward")
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **kw: (
        agent, object(), reward, {"weights_revision": 19457}))
    monkeypatch.setattr(script, "tensor_hash", lambda value: value.name)
    monkeypatch.setattr(script, "provenance", lambda: {})

    def likelihood(actual, nll_config, **kwargs):
        assert actual is agent and kwargs["reward"] is reward
        assert kwargs["checkpoint_weights_revision"] == 19457
        assert not nll_config.eval_all_levels and nll_config.eval_episodes_per_task == 2
        return {"diagnostic_only": True, "joint_nll": 1.25}

    monkeypatch.setattr(script, "evaluate_imitation_validation", likelihood)
    monkeypatch.setattr(script, "evaluate_sequence", lambda *a, **kw: pytest.fail("no gameplay"))
    before, threads = script._checkpoint_hashes(checkpoint), torch.get_num_threads()
    output = tmp_path / "diagnostics"
    assert script.main(["--checkpoint", str(checkpoint), "--config", str(config_path),
                        "--output", str(output), "--skip-gameplay", "--validation-nll"]) == 0
    assert script._checkpoint_hashes(checkpoint) == before
    assert torch.get_num_threads() == threads
    assert json.loads((output / "status.json").read_text())["updates"] == 0
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["protocol"] is None and metadata["protocol_sha256"] is None
    assert metadata["validation_likelihood_protocol"]["counts"] == {"navigation": 2}
    assert metadata["validation_likelihood_protocol"]["eval_all_levels"] is False
    assert json.loads((output / "validation_nll.json").read_text())["joint_nll"] == 1.25
    assert not (output / "evaluation.json").exists()


def test_protocol_guard_fails_before_creating_output(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(SequenceRunConfig().to_dict()))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="protocol"):
        script.main(["--checkpoint", str(tmp_path / "absent"), "--config", str(config),
                     "--output", str(output), "--expected-protocol-sha256", "wrong"])
    assert not output.exists()


def test_nll_only_checks_the_executed_likelihood_protocol(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(SequenceRunConfig().to_dict()))
    output = tmp_path / "output"
    arguments = ["--checkpoint", str(tmp_path / "absent"), "--config", str(config),
                 "--output", str(output), "--skip-gameplay", "--validation-nll"]
    with pytest.raises(ValueError, match="validation protocol"):
        script.main([*arguments, "--expected-validation-protocol-sha256", "wrong"])
    with pytest.raises(ValueError, match="gameplay protocol guard"):
        script.main([*arguments, "--expected-protocol-sha256", "gameplay only"])
    assert not output.exists()


@pytest.fixture
def frozen_cli(monkeypatch, tmp_path):
    from carryrl.agent.actions import PhysicalAction

    class NoopAgent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def reset_episode(self, *, instruction=None):
            pass

        def act(self, frame, *, deterministic):
            return SimpleNamespace(action=PhysicalAction())

    agent, reward = NoopAgent(), torch.nn.Linear(1, 1)
    checkpoint = tmp_path / "source/checkpoints/rollout_0000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"unchanged frozen checkpoint")
    (checkpoint / "manifest.json").write_text(json.dumps({
        "epoch": 2, "agent_sha256": script.tensor_hash(agent), "weights_revision": 42,
        "sha256": script.file_sha256(checkpoint / "state.pt")}))
    config = SequenceRunConfig(tasks=("navigation",), max_episode_steps=2,
                               eval_episodes_per_task=4, vision_cache_frames=0)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    output = tmp_path / "diagnostic"
    arguments = ["--checkpoint", str(checkpoint), "--config", str(config_path),
                 "--output", str(output), "--no-all-levels"]
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *args, **kwargs: (
        agent, None, reward, {"weights_revision": 42}))
    monkeypatch.setattr(script, "provenance", lambda: {})
    return arguments, checkpoint, output, config


@pytest.mark.parametrize("scope,count", [("first_per_task", 1), ("failed", 4), ("all", 4)])
def test_cli_capture_real_repeated_seeded_episodes_and_metadata(frozen_cli, scope, count):
    import hashlib
    from dataclasses import replace

    arguments, checkpoint, output, config = frozen_cli
    expected_protocol = script.sequence_evaluation_protocol(replace(config, eval_all_levels=False))
    expected_hash = hashlib.sha256(
        json.dumps(expected_protocol, sort_keys=True).encode()).hexdigest()
    before = script._checkpoint_hashes(checkpoint)
    assert script.main([*arguments, "--capture-scope", scope,
                        "--expected-protocol-sha256", expected_hash]) == 0
    metadata = json.loads((output / "metadata.json").read_text())
    evaluated = json.loads((output / "evaluation.json").read_text())
    status = json.loads((output / "status.json").read_text())
    capture = metadata["playback_capture"]
    assert capture == evaluated["playback_capture"] == status["playback_capture"]
    assert capture["requested_scope"] == scope
    assert capture["enabled"] is True and capture["policy"] == "frozen_deterministic"
    assert capture["captured_count"] == count
    assert capture["outcomes"] == {"success": 0, "failure": 0, "unfinished": count, "unknown": 0}
    assert len(capture["selected_episodes"]) == count
    assert metadata["protocol"] == expected_protocol
    assert metadata["protocol_sha256"] == evaluated["protocol_sha256"] == expected_hash
    assert "capture_scope" not in metadata["protocol"]
    assert [row["episode"]["episode_index"] for row in capture["selected_episodes"]] == (
        list(range(count)))
    paths = []
    for row in capture["selected_episodes"]:
        assert row["episode"]["level_bank_seed"] == config.level_bank_seed
        assert row["episode"]["level_index"] is None  # Legacy seeded naming would collide.
        assert row["episode"]["level_id"] and "instruction" in row["episode"]
        path = Path(row["playback"])
        paths.append(path)
        replay = json.loads(path.read_text())
        diagnostic = json.loads(Path(row["diagnostic"]).read_text())
        assert replay["observation_frame_count"] == replay["action_count"] + 1 == 3
        assert replay["quality"]["commands"] == 2
        assert replay["truncated"] and not replay["terminated"]
        assert diagnostic["replay_verified"] is True
        assert diagnostic["actual_actions"] == [entry["action"] for entry in replay["actions"]]
    assert len(set(paths)) == count
    if scope == "first_per_task":
        assert all(path.parent == output / "playbacks" for path in paths)
    else:
        assert all(output / "captured_episodes" in path.parents for path in paths)
    assert status["status"] == "completed" and status["updates"] == 0
    assert script._checkpoint_hashes(checkpoint) == before


def test_cli_capture_counts_keep_unknown_separate():
    selected = [{"outcome_category": category} for category in
                ("success", "failure", "unfinished", "unknown", "unknown")]
    report = script._capture_report("all", selected, enabled=True)
    assert report["captured_count"] == 5
    assert report["outcomes"] == {"success": 1, "failure": 1, "unfinished": 1, "unknown": 2}


def test_cli_capture_disk_failure_never_marks_evaluation_completed(frozen_cli, monkeypatch):
    arguments, checkpoint, output, _ = frozen_cli
    before = script._checkpoint_hashes(checkpoint)

    def failed_write(*args, **kwargs):
        raise OSError("playback disk write failed")

    monkeypatch.setattr(script, "write_frozen_playback", failed_write)
    with pytest.raises(OSError, match="disk write failed"):
        script.main([*arguments, "--capture-scope", "failed"])
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["error_type"] == "OSError"
    assert status["playback_capture"]["captured_count"] == 0
    assert not (output / "evaluation.json").exists()
    assert script._checkpoint_hashes(checkpoint) == before


def test_cli_authenticates_capture_before_writing_any_playback(frozen_cli, monkeypatch):
    arguments, _, output, _ = frozen_cli
    original = script.evaluate_sequence

    def corrupted_evaluation(*args, **kwargs):
        callback = kwargs["on_playback"]

        def corrupt(record, frames, actions):
            frames[1][50, 0] ^= 1
            callback(record, frames, actions)

        return original(*args, **{**kwargs, "on_playback": corrupt})

    monkeypatch.setattr(script, "evaluate_sequence", corrupted_evaluation)
    monkeypatch.setattr(script, "write_frozen_playback",
                        lambda *args, **kwargs: pytest.fail("published unauthenticated replay"))
    with pytest.raises(ValueError, match="RGB alignment"):
        script.main([*arguments, "--capture-scope", "all"])
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert not (output / "captured_episodes").exists()


@pytest.mark.parametrize("scope", ["failed", "all"])
def test_cli_rejects_stochastic_capture_scope_before_model_or_output(tmp_path, scope):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(SequenceRunConfig(eval_stochastic=True).to_dict()))
    output = tmp_path / "diagnostic"
    with pytest.raises(ValueError, match="eval_stochastic=False"):
        script.main(["--checkpoint", str(tmp_path / "absent"), "--config", str(config),
                     "--output", str(output), "--capture-scope", scope])
    assert not output.exists()


def test_cli_invalid_scope_fails_before_config_model_and_output(tmp_path):
    output = tmp_path / "diagnostic"
    with pytest.raises(SystemExit):
        script.main(["--checkpoint", str(tmp_path / "absent"),
                     "--config", str(tmp_path / "absent_config"), "--output", str(output),
                     "--capture-scope", "invalid"])
    assert not output.exists()
