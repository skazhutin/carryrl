"""Bound disposable test and package caches after project checks."""

import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest


def pytest_sessionfinish(session, exitstatus):
    uv = shutil.which("uv")
    if uv is None:
        warnings.warn("Package cache cleanup skipped: uv is unavailable.", pytest.PytestWarning)
        return
    tests_dir = Path(__file__).resolve().parent
    try:
        # uv run holds its cache lock until pytest exits. A detached child can wait
        # for that lock without blocking pytest or retaining the runner's pipes.
        subprocess.Popen(
            [sys.executable, str(tests_dir / "cache_cleanup.py"), uv,
             str(tests_dir.parent / ".cache")],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True,
        )
    except OSError as error:
        warnings.warn(f"Package cache cleanup failed to start: {error}", pytest.PytestWarning)
