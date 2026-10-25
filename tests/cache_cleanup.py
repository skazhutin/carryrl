"""Detached, bounded uv cache maintenance (never imported by the application)."""

import fcntl
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def prune_cache(uv: str, cache_dir: Path, timeout: float = 60) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    status_path = cache_dir / "package_cache_cleanup.json"
    with (cache_dir / "package_cache_cleanup.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # The active job owns the status file and already covers this request.
            return

        def status(state: str, **details) -> None:
            data = {"status": state, "updated_at": datetime.now(UTC).isoformat(),
                    "pid": os.getpid(), **details}
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, indent=2) + "\n")
            temporary.replace(status_path)

        with (cache_dir / "package_cache_cleanup.log").open("w") as log:
            status("running", command=[uv, "cache", "prune", "--offline"])
            try:
                result = subprocess.run(
                    [uv, "cache", "prune", "--offline"],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    timeout=timeout, check=False,
                )
            except subprocess.TimeoutExpired:
                message = f"Package cache cleanup timed out after {timeout}s."
                log.write(message + "\n")
                status("timeout", error=message)
            except OSError as error:
                log.write(f"Package cache cleanup failed: {error}\n")
                status("failed", error=str(error))
            else:
                status("completed" if result.returncode == 0 else "failed",
                       returncode=result.returncode)


if __name__ == "__main__":
    prune_cache(sys.argv[1], Path(sys.argv[2]))
    from carryrl.training.storage_maintenance import maybe_launch

    maybe_launch(Path(sys.argv[2]).resolve().parent)
