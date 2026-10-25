import http.client
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from carryrl.dashboard import validation_browser as browser
from carryrl.dashboard.overview import make_server
from carryrl.dashboard.validation_browser import (
    ReplayBusy,
    ReplayResources,
    ValidationBrowser,
    write_json,
)


@pytest.fixture
def run(tmp_path):
    run = tmp_path / "runs" / "example"
    checkpoint = run / "checkpoints" / "rollout_0000"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"mock")
    write_json(checkpoint / "manifest.json", {"rollout": 0, "weights_revision": 2,
        "sha256": "checkpoint-hash", "agent_sha256": "agent-hash", "training_backend": "mlx"})
    write_json(run / "config.json", {"tasks": ["navigation"], "level_bank_seed": 0})
    for folder in ("evaluations", "evaluations_pixel"):
        destination = run / folder
        destination.mkdir()
        write_json(destination / "rollout_0000.json", {"split": "validation",
            "agent_sha256": "agent-hash", "protocol_sha256": "protocol-hash", "records": [
                {"task": "navigation", "level_index": 1, "seed": 1000001,
                 "policy": policy, "outcome": {"known": True, "success": success, "score": 5},
                 "presentation": {"pixel_art": True} if folder == "evaluations_pixel" else None,
                 "truncated": not success}
                for policy, success in [("frozen_deterministic", True),
                                        ("frozen_stochastic", False)]]})
    return run


def request_data():
    return {"run": "example", "checkpoint": "rollout_0000", "task": "navigation",
            "level_index": 1, "appearance": "canonical", "mode": "greedy"}


@pytest.mark.skipif(sys.platform not in {"darwin", "linux"}, reason="native resource probe")
def test_resource_snapshot_works_in_base_environment_without_optional_packages(tmp_path):
    snapshot = browser.resource_snapshot(tmp_path, os.getpid())
    assert snapshot["available_bytes"] > 0
    assert snapshot["free_disk_bytes"] > 0
    assert snapshot["process_tree_rss_bytes"] > 0


def test_catalog_keeps_modes_appearances_and_checkpoint_identity_separate(run):
    catalog = ValidationBrowser(run.parent).catalog(run.name)
    assert catalog["checkpoints"][0]["weights_revision"] == 2
    assert len(catalog["tasks"][0]["levels"]) == 16
    assert len(catalog["records"]) == 4
    assert {(row["appearance"], row["mode"]) for row in catalog["records"]} == {
        ("canonical", "greedy"), ("canonical", "stochastic"),
        ("pixel", "greedy"), ("pixel", "stochastic")}
    assert all(row["checkpoint_id"] == "rollout_0000" for row in catalog["records"])
    # No model load, archive extraction, evaluation, or replay occurs on inventory reads.
    assert not (run / "dashboard_replays").exists()


def test_catalog_never_links_unmatched_weights_or_test_split(run):
    write_json(run / "evaluations" / "rollout_0001.json",
               {"split": "validation", "agent_sha256": "other", "records": [
                   {"task": "navigation", "policy": "frozen_deterministic", "level_index": 1}]})
    write_json(run / "evaluations" / "test.json",
               {"split": "test", "agent_sha256": "agent-hash", "records": [
                   {"task": "navigation", "policy": "frozen_deterministic", "level_index": 1}]})
    assert len(ValidationBrowser(run.parent).catalog(run.name)["records"]) == 4


def test_catalog_classifies_actual_presentation_not_directory(run):
    path = run / "evaluations" / "rollout_0000.json"
    data = browser.read_json(path)
    data["records"][0]["presentation"] = {"pixel_art": True}
    data["records"][1]["presentation"] = {"pixel_art": False, "mode": "varied"}
    write_json(path, data)
    result = ValidationBrowser(run.parent).catalog(run.name)
    assert result["records"][0]["appearance"] == "pixel"
    assert result["records"][1]["appearance"] == "varied"
    assert result["warnings"]


def test_paths_and_symlink_run_are_rejected(run, tmp_path):
    service = ValidationBrowser(run.parent)
    with pytest.raises(ValueError):
        service.catalog("../example")
    (run.parent / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        service.catalog("escape")
    (run / "dashboard_replays").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        service.media(run.name, "a" * 32, "frame.gif")


@pytest.mark.parametrize("patch", [
    {"checkpoint": "../rollout_0000"}, {"level_index": -1}, {"level_index": True},
    {"level_index": 16}, {"task": "unknown"}, {"mode": "stochastic"},
    {"appearance": "test"}, {"split": "test"}])
def test_invalid_requests_cannot_start_process(run, patch, monkeypatch):
    monkeypatch.setattr(browser.subprocess, "Popen",
                        lambda *args, **kwargs: pytest.fail("inference must not start"))
    with pytest.raises(ValueError):
        ValidationBrowser(run.parent).start({**request_data(), **patch})


def test_resources_and_single_job_guard_precede_launch(run, monkeypatch):
    service = ValidationBrowser(run.parent)
    monkeypatch.setattr(browser, "resource_snapshot", lambda *args: {
        "available_bytes": 1, "free_disk_bytes": 100 * 1024**3, "process_tree_rss_bytes": 0})
    with pytest.raises(ReplayResources):
        service.start(request_data())
    assert not (run / "dashboard_replays").exists()
    service.active = "already-running"
    with pytest.raises(ReplayBusy):
        service.start(request_data())


def test_launch_is_explicit_and_isolated(run, monkeypatch):
    calls = []
    monkeypatch.setattr(browser, "resource_snapshot", lambda *args: {
        "available_bytes": 10 * 1024**3, "free_disk_bytes": 100 * 1024**3,
        "process_tree_rss_bytes": 0})
    monkeypatch.setattr(browser.subprocess, "Popen",
                        lambda args, **kwargs: calls.append((args, kwargs)) or object())
    monkeypatch.setattr(browser.threading.Thread, "start", lambda self: None)
    service = ValidationBrowser(run.parent)
    job = service.start(request_data())
    assert job["status"] == "running"
    args, options = calls[0]
    assert "carryrl.dashboard.validation_replay_job" in args
    assert options["start_new_session"] is True
    assert options["env"]["OMP_NUM_THREADS"] == "2"
    path = run / "dashboard_replays" / job["id"] / "request.json"
    payload = json.loads(path.read_text())
    assert payload["reference"]["seed"] == 1000001
    assert payload["checkpoint_sha256"] == "checkpoint-hash"
    assert service.status(job["id"])["request"] == request_data()


def test_supervisor_handles_exit_during_resource_stop_and_failed_status_write(run, monkeypatch):
    service = ValidationBrowser(run.parent)
    service.active = "job"
    service.jobs["job"] = {"status": "running", "request": request_data()}
    process = SimpleNamespace(pid=123456, returncode=-15, poll=lambda: None,
                              wait=lambda **kwargs: -15)
    monkeypatch.setattr(browser, "resource_snapshot", lambda *args: {
        "available_bytes": 1, "free_disk_bytes": 100 * 1024**3, "process_tree_rss_bytes": 0})

    def gone(*args):
        raise ProcessLookupError

    monkeypatch.setattr(browser.os, "killpg", gone)
    monkeypatch.setattr(browser, "write_json",
                        lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    service._watch(process, run, "job")
    assert service.active is None
    assert service.jobs["job"]["status"] == "failed"
    assert "disk full" in service.jobs["job"]["error"]


def test_supervisor_timeout_terminates_only_its_process_group(run, monkeypatch):
    service = ValidationBrowser(run.parent)
    service.active = "job"
    service.jobs["job"] = {"status": "running", "request": request_data()}
    killed = []
    process = SimpleNamespace(pid=123456, returncode=-15, poll=lambda: None,
                              wait=lambda **kwargs: -15)
    monkeypatch.setattr(browser, "resource_snapshot", lambda *args: {
        "available_bytes": 10 * 1024**3, "free_disk_bytes": 100 * 1024**3,
        "process_tree_rss_bytes": 0})
    ticks = iter([0, browser.MAX_SECONDS + 1])
    monkeypatch.setattr(browser.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(browser.os, "killpg", lambda pid, signal: killed.append(pid))
    service._watch(process, run, "job")
    assert killed == [123456]
    assert service.active is None
    assert "180" in service.jobs["job"]["error"]


def test_replay_http_write_boundary_and_errors(tmp_path):
    class Service:
        calls = []

        def start(self, request):
            self.calls.append(request)
            return {"id": "a" * 32, "status": "running"}

        def status(self, job_id):
            return None

    service = Service()
    server = make_server(SimpleNamespace(root=tmp_path), port=0, validation=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def post(body, headers=None):
        client = http.client.HTTPConnection("127.0.0.1", server.server_port)
        client.request("POST", "/api/replay", body, headers={
            "Content-Type": "application/json", **(headers or {})})
        response = client.getresponse()
        status, result = response.status, json.loads(response.read())
        client.close()
        return status, result

    try:
        assert post("{}", {"Origin": "https://evil.example"})[0] == 403
        assert post("{}", {"Host": "external.example"})[0] == 403
        assert post("{}", {"Content-Type": "text/plain"})[0] == 400
        assert post("{")[0] == 400
        assert post("x" * 4097)[0] == 400
        assert service.calls == []
        assert post(json.dumps(request_data()), {
            "Origin": f"http://127.0.0.1:{server.server_port}"})[0] == 202
        assert service.calls == [request_data()]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("appearance", ["canonical", "pixel"])
def test_mock_inference_preserves_level_conditions_and_writes_separate_artifacts(
        run, monkeypatch, appearance):
    from carryrl.agent.actions import PhysicalAction
    from carryrl.dashboard import frozen_playback, validation_replay_job
    from carryrl.envs.presentation import PresentationConfig
    from carryrl.training import sequence
    from carryrl.training.sequence_config import SequenceRunConfig

    # The active fresh run evaluates both appearances. These flags enforce
    # cross-field validation and must be normalized before a manual replay.
    config = SequenceRunConfig(tasks=("navigation",), max_episode_steps=2,
        evaluation_canonical=True, evaluation_pixel_art=True,
        visual_config=PresentationConfig(pixel_art=True))
    reset_instructions = []
    agent = SimpleNamespace(reset_episode=lambda **kwargs: reset_instructions.append(kwargs),
                            act=lambda frame, **kwargs: SimpleNamespace(action=PhysicalAction()))
    manifest = browser.read_json(run / "checkpoints" / "rollout_0000" / "manifest.json")
    monkeypatch.setattr(frozen_playback, "load_frozen_agent",
                        lambda path: (agent, config, manifest))
    monkeypatch.setattr(sequence, "tensor_hash", lambda agent: "agent-hash")
    monkeypatch.setattr("torch.set_num_interop_threads", lambda value: None)
    directory = run / "dashboard_replays" / ("a" * 32)
    directory.mkdir(parents=True)
    write_json(directory / "request.json", {**request_data(), "appearance": appearance,
        "checkpoint_sha256": "checkpoint-hash", "agent_sha256": "agent-hash", "reference": None})
    result = validation_replay_job.replay(directory)
    assert result["seed"] == 1000001
    assert result["level_index"] == 1
    assert result["split"] == "validation"
    assert result["frame_count"] == result["action_count"] + 1 == 3
    assert result["reference_match"] is None
    assert result["matches_frozen_evaluation_record"] is False
    assert result["inference_backend"] == "torch"
    assert result["checkpoint_training_backend"] == "mlx"
    assert Path(result["gif"]).is_relative_to(directory)
    assert not (run / "playbacks").exists()
    assert reset_instructions[0]["instruction"] == result["instruction"]
    if appearance == "pixel":
        assert result["presentation"]["pixel_style"] in {"frost", "circuit"}
    else:
        assert result["presentation"] is None
    service = ValidationBrowser(run.parent)
    assert service.catalog(run.name)["replays"] == []
    # Publishing status invalidates the inventory cache; a new server can recover
    # the replay without loading a checkpoint or consulting in-memory jobs.
    write_json(directory / "status.json", {"id": directory.name, "status": "completed",
        "request": {**request_data(), "appearance": appearance}, "replay": result})
    catalog = service.catalog(run.name)
    restored = catalog["replays"][0]
    assert restored["id"] == directory.name
    assert restored["url"].startswith(f"/replay-media/{run.name}/{directory.name}/")
    assert [row for row in catalog["records"] if row.get("manual")][0][
        "replay_id"] == directory.name
    assert ValidationBrowser(run.parent).catalog(run.name)["replays"] == catalog["replays"]
    manifest["sha256"] = "different-checkpoint"
    write_json(run / "checkpoints" / "rollout_0000" / "manifest.json", manifest)
    assert service.catalog(run.name)["replays"] == []


def test_active_job_is_recovered_even_when_catalog_is_cached(run):
    service = ValidationBrowser(run.parent)
    assert service.catalog(run.name)["active_job"] is None
    service.active = "job"
    service.jobs["job"] = {"id": "job", "status": "running", "request": request_data()}
    assert service.catalog(run.name)["active_job"]["id"] == "job"
    service.jobs["job"]["request"]["run"] = "another"
    assert service.catalog(run.name)["active_job"] is None
