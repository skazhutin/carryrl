"""Preparation is inert; resource limits and constructor RNG remain controlled."""

import importlib.util
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_token_comparison.py"
spec = importlib.util.spec_from_file_location("token_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_check_mode_does_not_launch_or_create_output(monkeypatch, tmp_path, capsys):
    output = tmp_path / "training"
    plan = {
        "modelinit_verified": False,
        "arms": [{"spatial_tokens": 16, "training_output": str(output)}],
    }
    monkeypatch.setattr(runner, "verify_plan", lambda _: plan)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("launch"))
    assert runner.main(["--plan", str(tmp_path), "--tokens", "16"]) == 0
    assert '"training_started": false' in capsys.readouterr().out
    assert not output.exists()
    with pytest.raises(ValueError, match="materialized"):
        runner.main(["--plan", str(tmp_path), "--tokens", "16", "--execute"])
    plan["modelinit_verified"] = True
    plan["arms"][0]["initialization_verified"] = True
    with pytest.raises(ValueError, match="calibration"):
        runner.main(["--plan", str(tmp_path), "--tokens", "16", "--execute"])


def test_resource_reserves_distinguish_graceful_and_emergency_stop():
    sample = {
        "process_group_rss_bytes": 4 * runner.GIB,
        "system_available_bytes": 2 * runner.GIB,
        "system_memory_pressure_level": 1,
        "disk_free_bytes": 9 * runner.GIB,
    }
    assert runner.resource_reason(sample) is None
    assert runner.resource_reason(sample, soft=True) == "checkpoint staging reserve"
    for key, value, expected in [
        ("disk_free_bytes", 8 * runner.GIB, "free disk reserve"),
        ("process_group_rss_bytes", 12 * runner.GIB, "RSS limit"),
        ("system_memory_pressure_level", 4, "critical or unavailable memory pressure"),
        ("system_available_bytes", runner.GIB // 2, "system available memory reserve"),
    ]:
        assert runner.resource_reason(dict(sample, **{key: value})) == expected
    assert runner.resource_reason(dict(sample, system_swap_delta_bytes=2 * runner.GIB)) is None
    assert runner.resource_reason(
        dict(sample, disk_free_bytes=20 * runner.GIB, system_swap_delta_bytes=2 * runner.GIB),
        soft=True,
    ) == "system swap growth"
    assert runner.resource_reason(
        dict(sample, system_swap_delta_bytes=4 * runner.GIB)
    ) == "system swap growth"
    assert runner.resource_reason(dict(sample, system_swap_delta_bytes="bad")) == (
        "invalid resource metrics"
    )


def test_swap_metrics_parse_available_and_fail_closed_when_malformed(monkeypatch):
    result = SimpleNamespace(returncode=0, stdout="total = 4096.00M used = 512.50M free = 3583.50M")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: result)
    assert runner.swap_used_bytes() == int(512.5 * 1024**2)
    result.stdout = "not a vm.swapusage response"
    with pytest.raises(ValueError, match="unreadable"):
        runner.swap_used_bytes()
    result.returncode = 1
    assert runner.swap_used_bytes() is None


def test_calibration_requires_explicit_execute_and_32_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("launch"))
    for arguments in (["--tokens", "32"], ["--tokens", "16", "--execute"]):
        with pytest.raises(SystemExit):
            runner.main(["--plan", str(tmp_path), "--calibrate-lr", "0.00001", *arguments])


def test_calibration_dispatch_propagates_lr_and_uses_its_config(monkeypatch, tmp_path):
    plan = {
        "modelinit_verified": True,
        "ready_to_train": False,
        "project_root": str(tmp_path),
        "arms": [{"spatial_tokens": 32, "training_output": "primary"}],
    }
    artifact = {
        "training_output": str(tmp_path / "trial"),
        "config": {"path": str(tmp_path / "calibration/config.json")},
        "initial_checkpoint": str(tmp_path / "initial/t32"),
        "initialization_verified": True,
    }
    calls = []
    monkeypatch.setattr(runner, "verify_plan", lambda _: plan)
    monkeypatch.setattr(runner, "prepare_calibration", lambda path, lr: artifact)
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=32 * runner.GIB)
    )
    monkeypatch.setattr(runner.subprocess, "Popen", lambda argv, **kw: (
        calls.append(argv) or SimpleNamespace(pid=99)
    ))
    assert runner.main([
        "--plan", str(tmp_path), "--tokens", "32", "--calibrate-lr", "0.00003", "--execute",
    ]) == 0
    assert calls[0][-2:] == ["--calibrate-lr", "3e-05"]
    assert "--supervise" in calls[0]
    observed = []
    from carryrl.training import sequence
    from carryrl.training.sequence_config import SequenceRunConfig

    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(SequenceRunConfig, "load", lambda path: observed.append(path) or "config")
    monkeypatch.setattr(sequence, "train_sequence", lambda config, output, **kwargs: (
        observed.append((config, output, kwargs["warm_start"])) or {"status": "completed"}
    ))
    assert runner.main([
        "--plan", str(tmp_path), "--tokens", "32", "--calibrate-lr", "0.00003", "--train",
    ]) == 0
    assert observed[0] == artifact["config"]["path"]
    assert observed[1] == ("config", Path(artifact["training_output"]),
                           Path(artifact["initial_checkpoint"]))


def test_supervisor_propagates_calibration_to_training_child(monkeypatch, tmp_path):
    artifact = {"training_output": str(tmp_path / "trial")}
    monkeypatch.setattr(
        runner, "execution_arm", lambda *_: ({"project_root": str(tmp_path)}, artifact)
    )
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=32 * runner.GIB)
    )
    monkeypatch.setattr(runner, "swap_used_bytes", lambda: 100)
    monkeypatch.setattr(runner, "group_members", lambda _: [])
    commands = []
    child = SimpleNamespace(pid=99, poll=lambda: 0, wait=lambda: 0)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda argv, **kw: (
        commands.append(argv) or child
    ))
    assert runner._supervise(tmp_path, 32, 0.00001) == 0
    assert commands[0][-2:] == ["--calibrate-lr", "1e-05"]
    assert "--train" in commands[0]


def test_internal_train_cannot_bypass_initial_or_readiness_gate(monkeypatch, tmp_path):
    plan = {"modelinit_verified": False,
            "arms": [{"spatial_tokens": 32, "initialization_verified": False}]}
    monkeypatch.setattr(runner, "verify_plan", lambda _: plan)
    with pytest.raises(ValueError, match="materialized"):
        runner.train_arm(tmp_path, 32)
    plan["modelinit_verified"] = plan["arms"][0]["initialization_verified"] = True
    with pytest.raises(ValueError, match="calibration"):
        runner.train_arm(tmp_path, 32)


def test_stop_waits_for_descendants_after_wrapper_exit(monkeypatch):
    alive = True
    sent = []
    clock = 0

    def monotonic():
        nonlocal clock
        clock += 20
        return clock

    def killpg(group, sig):
        nonlocal alive
        sent.append(sig)
        if sig == signal.SIGTERM:
            alive = False

    monkeypatch.setattr(runner, "group_members", lambda _: [123] if alive else [])
    monkeypatch.setattr(runner.os, "killpg", killpg)
    monkeypatch.setattr(runner.time, "monotonic", monotonic)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    child = SimpleNamespace(pid=123, poll=lambda: 0, wait=lambda: 0)
    assert runner.stop_group(child) == 0
    assert sent == [signal.SIGINT, signal.SIGTERM]
    assert not alive


def test_checkpoint_constructor_does_not_shift_rollout_rng(tmp_path):
    class Agent:
        def __init__(self, config):
            torch.rand(config.spatial_tokens, 256)

    factory = runner.make_agent_factory(tmp_path, agent_class=Agent)
    torch.manual_seed(20261019)
    initial = torch.get_rng_state().clone()
    for tokens in (16, 32, 64):
        factory(SimpleNamespace(text_encoder="none", spatial_tokens=tokens))
        assert torch.equal(initial, torch.get_rng_state())
