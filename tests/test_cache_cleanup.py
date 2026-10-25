"""Cache maintenance cannot hold up pytest or overlap an existing cleanup."""

import fcntl
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("returncode, state", [(0, "completed"), (2, "failed")])
def test_prune_records_result_and_uses_offline_bounded_command(tmp_path, monkeypatch,
                                                            returncode, state):
    helper = load_helper("cache_cleanup")
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(helper.subprocess, "run", run)
    helper.prune_cache("/bin/uv", tmp_path)
    assert calls[0][0] == ["/bin/uv", "cache", "prune", "--offline"]
    assert calls[0][1]["timeout"] == 60
    status = json.loads((tmp_path / "package_cache_cleanup.json").read_text())
    assert status["status"] == state
    assert status["returncode"] == returncode


@pytest.mark.parametrize("error, state", [
    (subprocess.TimeoutExpired("uv", 60), "timeout"),
    (OSError("unavailable"), "failed"),
])
def test_prune_records_errors(tmp_path, monkeypatch, error, state):
    helper = load_helper("cache_cleanup")

    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(helper.subprocess, "run", run)
    helper.prune_cache("uv", tmp_path)
    assert json.loads((tmp_path / "package_cache_cleanup.json").read_text())["status"] == state
    assert (tmp_path / "package_cache_cleanup.log").read_text()


def test_duplicate_job_preserves_active_status(tmp_path, monkeypatch):
    helper = load_helper("cache_cleanup")
    status = tmp_path / "package_cache_cleanup.json"
    status.write_text('{"status": "running"}')
    monkeypatch.setattr(helper.subprocess, "run", lambda *a, **kw: pytest.fail("duplicate prune"))
    with (tmp_path / "package_cache_cleanup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        helper.prune_cache("uv", tmp_path)
    assert status.read_text() == '{"status": "running"}'


def test_sessionfinish_detaches_without_waiting(monkeypatch):
    hook = load_helper("conftest")
    calls = []
    monkeypatch.setattr(hook.shutil, "which", lambda _: "/bin/uv")
    monkeypatch.setattr(hook.subprocess, "Popen", lambda *a, **kw: calls.append((a, kw)))
    hook.pytest_sessionfinish(None, 1)
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["close_fds"] is True
    for stream in ("stdin", "stdout", "stderr"):
        assert calls[0][1][stream] == subprocess.DEVNULL
    assert calls[0][0][0][1].endswith("/tests/cache_cleanup.py")


def test_sessionfinish_warns_on_launch_failure(monkeypatch):
    hook = load_helper("conftest")
    monkeypatch.setattr(hook.shutil, "which", lambda _: "/bin/uv")

    def fail(*args, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(hook.subprocess, "Popen", fail)
    with pytest.warns(pytest.PytestWarning, match="failed to start"):
        hook.pytest_sessionfinish(None, 0)
