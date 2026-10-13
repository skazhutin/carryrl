"""Own one TensorBoard subprocess and discard its cache on a source change."""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class FocusServerError(RuntimeError):
    """The owned server could not start or stop safely."""


class FocusServer:
    def __init__(self, logdir: Path, port: int):
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("TensorBoard port must be between 1 and 65535")
        self.logdir = Path(logdir).resolve()
        self.port = port
        self.log_path = self.logdir.with_name(self.logdir.name + "_server.log")
        self.process = None
        self.identity = None

    def _start(self):
        # Probe availability without inspecting or terminating another process.
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", self.port))
            except OSError as error:
                raise FocusServerError(
                    f"TensorBoard port {self.port} is unavailable: {error}. "
                    "No existing process was stopped.") from error
        if self.log_path.is_symlink():
            raise FocusServerError("TensorBoard server log must not be a symlink")
        command = [sys.executable, "-m", "tensorboard.main", "--logdir", str(self.logdir),
                   "--host", "127.0.0.1", "--port", str(self.port), "--reload_interval", "5",
                   "--samples_per_plugin", "text=1,images=4", "--load_fast", "false",
                   "--window_title", "СЕЙЧАС — MLX — 16 игр"]
        try:
            with self.log_path.open("ab") as log:
                self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        except OSError as error:
            raise FocusServerError(f"Cannot launch owned TensorBoard: {error}") from error
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise FocusServerError(f"TensorBoard exited with code {self.process.returncode}; "
                                       f"see {self.log_path}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=.2):
                    LOGGER.info("Owned TensorBoard ready at http://127.0.0.1:%s", self.port)
                    return
            except OSError:
                time.sleep(.1)
        self.close()
        raise FocusServerError(f"TensorBoard did not become ready; see {self.log_path}")

    def ensure_generation(self, identity: str):
        """Call only after events and their cursor have been successfully flushed."""
        if self.process is not None and self.process.poll() is not None:
            raise FocusServerError(f"Owned TensorBoard exited; see {self.log_path}")
        if self.process is not None and self.identity == identity:
            return
        self.close()  # Wait for the owned listener to stop before binding its port again.
        self._start()
        self.identity = identity

    def close(self):
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired as error:
                    raise FocusServerError("Owned TensorBoard did not stop; port was not reused") \
                        from error
        self.process = None
        self.identity = None
