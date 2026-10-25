"""Preflight, or explicitly run fresh RL under memory and elapsed-time guards."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_optional(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def monitor(plan, manifest, guard, runner):
    """Only this Popen and actors proven to belong to it may receive signals."""
    budget = manifest["budget_seconds"]
    stage = plan / "stages" / "main"
    if stage.exists() or (plan / "supervisor.json").exists():
        raise ValueError("supervisor/stage already exists; refusing a duplicate launch")
    stage.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / ".venv/bin/python"), str(ROOT / ".cache/guard_memory_throttle.py"),
               "--directory", str(stage), "--graceful", "--wall-seconds", "14400", "--",
               str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/train_fresh_rl.py"),
               "--plan", str(plan), "--execute"]
    started = time.monotonic()
    receipt = {"status": "starting", "started_utc": runner.utc(),
               "budget_seconds": budget, "command": command, "actual_exit_code": None,
               "hard_timeout": False, "soft_stop_requested": False}
    child, actor, members = None, None, []
    requested = False

    def stop(_signum, _frame):
        nonlocal requested
        requested = True

    def refresh_members():
        nonlocal actor, members
        report = read_optional(stage / "process.json")
        candidate = report.get("runtime_guard_override_receipt", {}).get("actor_identity")
        if (actor is None and candidate and report.get("supervisor_pid") == child.pid
                and candidate.get("ppid") == child.pid
                and candidate.get("pid") == candidate.get("pgid")
                and report.get("command_pid") == candidate["pid"]
                and guard.same_process(candidate)):
            actor = candidate
            receipt["actor_identity"] = actor
        anchors = ([actor] if actor else []) + members
        if any(guard.same_process(item) for item in anchors):
            members = [identity for item in guard.guard.group_members(actor["pgid"])
                       if (identity := guard.identity(item["pid"])) is not None
                       and identity["pgid"] == actor["pgid"]]
        return report

    def kill_owned_actor():
        anchors = ([actor] if actor else []) + members
        if any(guard.same_process(item) for item in anchors):
            try:
                os.killpg(actor["pgid"], signal.SIGKILL)
                receipt["actor_cleanup_required"] = True
            except ProcessLookupError:
                pass

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (plan / "supervisor.log").open("x") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            receipt.update(status="running", guard_pid=child.pid)
            runner.publish(plan / "supervisor.json", receipt)
            while True:
                report = refresh_members()
                elapsed = time.monotonic() - started
                code = child.poll()
                if elapsed >= budget:
                    receipt["hard_timeout"] = True
                    kill_owned_actor()
                    if code is None:
                        child.kill()
                    break
                if code is not None:
                    # A wrapper exit must not leave its verified actor running.
                    kill_owned_actor()
                    break
                if (requested or elapsed >= budget - 1200) and not receipt["soft_stop_requested"]:
                    child.send_signal(signal.SIGTERM)
                    receipt.update(soft_stop_requested=True, soft_stop_elapsed_seconds=elapsed)
                    runner.publish(plan / "supervisor.json", receipt)
                time.sleep(min(1.0, max(0.0, budget - elapsed)))
            receipt["actual_exit_code"] = child.wait()
            report = read_optional(stage / "process.json")
            execution = read_optional(plan / "execution.json")
            receipt["actor_exit_code"] = report.get("actual_exit_code")
            receipt["execution_status"] = execution.get("status")
            receipt["status"] = (
                "hard_timeout" if receipt["hard_timeout"] else "completed"
                if receipt["actual_exit_code"] == 0 and receipt["actor_exit_code"] == 0
                and execution.get("status") == "completed"
                and not receipt.get("actor_cleanup_required") else "failed")
            if (receipt["status"] == "failed"
                    and receipt["actual_exit_code"] == 1
                    and receipt["actor_exit_code"] == 2
                    and receipt["execution_status"] in {"resource_stopped", "stopped"}
                    and not report.get("error")
                    and not receipt.get("actor_cleanup_required")):
                receipt["status"] = "stopped"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        if child is not None:
            try:
                refresh_members()
            finally:
                kill_owned_actor()
                if child.poll() is None:
                    child.kill()
                receipt["actual_exit_code"] = child.wait()
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        receipt.update(elapsed_seconds=time.monotonic() - started, finished_utc=runner.utc())
        runner.publish(plan / "supervisor.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = args.plan.resolve(strict=True)
    runner = load(ROOT / "scripts/train_fresh_rl.py", "fresh_rl_runner")
    manifest, _, _ = runner.verify_plan(plan)
    guard_path = ROOT / ".cache/guard_memory_throttle.py"
    if manifest.get("guard_sha256") and runner.sha(guard_path) != manifest["guard_sha256"]:
        raise ValueError("memory guard hash mismatch")
    if not args.execute:
        print(json.dumps({"verified": True, "training_started": False,
                          "elapsed_budget_seconds": manifest["budget_seconds"]}))
        return 0
    guard = load(guard_path, "fresh_rl_memory_guard")
    receipt = monitor(plan, manifest, guard, runner)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
