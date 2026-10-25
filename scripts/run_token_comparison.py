"""Check a prepared token arm; launch supervised RL only with explicit --execute."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from carryrl.training.token_calibration import (
    LEARNING_RATES,
    calibration_label,
    prepare_calibration,
)
from carryrl.training.token_comparison import verify_plan

GIB = 1024**3


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def utc():
    return datetime.now(UTC).isoformat()


def make_agent_factory(initial, *, agent_class=None, language_class=None):
    """Checkpoint graph construction must not shift policy RNG by token count."""

    def factory(config):
        import torch

        from carryrl.agent.language import MultilingualInstructionEncoder
        from carryrl.agent.model import GeneralistAgent

        with torch.random.fork_rng(devices=[]):
            kwargs = {}
            if config.text_encoder != "none":
                encoder = language_class or MultilingualInstructionEncoder
                kwargs["language"] = encoder.from_resources(
                    Path(initial) / "language", max_tokens=config.instruction_max_tokens
                )
            return (agent_class or GeneralistAgent)(config, **kwargs)

    return factory


def select_arm(plan, tokens):
    matches = [arm for arm in plan["arms"] if arm["spatial_tokens"] == tokens]
    if len(matches) != 1:
        raise ValueError("comparison must contain exactly one primary arm per token count")
    return matches[0]


def group_members(group):
    result = subprocess.run(
        ["ps", "-axo", "pid=,pgid=,rss="], check=True, capture_output=True, text=True
    )
    members = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and int(fields[1]) == group:
            members.append({"pid": int(fields[0]), "rss_bytes": int(fields[2]) * 1024})
    return members


def resources(group, root):
    members = group_members(group)
    vm = subprocess.run(["vm_stat"], check=True, capture_output=True, text=True).stdout
    page_match = re.search(r"page size of (\d+) bytes", vm)
    if page_match is None:
        raise ValueError("unreadable vm_stat page size")
    page = int(page_match.group(1))
    counts = dict(re.findall(r"^([^:\n]+):\s+(\d+)\.", vm, re.MULTILINE))
    if not all(key in counts for key in ("Pages free", "Pages inactive", "Pages speculative")):
        raise ValueError("incomplete vm_stat memory counters")
    available = page * sum(
        int(counts.get(key, 0)) for key in ("Pages free", "Pages inactive", "Pages speculative")
    )
    pressure = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        capture_output=True,
        text=True,
    )
    return {
        "utc": utc(),
        "processes": members,
        "process_group_rss_bytes": sum(item["rss_bytes"] for item in members),
        "system_available_bytes": available,
        "system_memory_pressure_level": int(pressure.stdout) if pressure.returncode == 0 else None,
        "system_swap_used_bytes": swap_used_bytes(),
        "disk_free_bytes": shutil.disk_usage(root).free,
    }


def swap_used_bytes():
    result = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "vm.swapusage"], capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    match = re.search(r"\bused\s*=\s*(\d+(?:\.\d+)?)\s*([KMGT]?)\b", result.stdout)
    if match is None:
        raise ValueError("unreadable vm.swapusage")
    return int(float(match.group(1)) * 1024 ** " KMGT".index(match.group(2) or " "))


def resource_reason(sample, *, soft=False):
    for key in ("process_group_rss_bytes", "system_available_bytes", "disk_free_bytes"):
        if type(sample.get(key)) is not int or sample[key] < 0:
            return "invalid resource metrics"
    swap = sample.get("system_swap_delta_bytes")
    if swap is not None and (type(swap) is not int or swap < 0):
        return "invalid resource metrics"
    if sample["process_group_rss_bytes"] >= 12 * GIB:
        return "RSS limit"
    if sample["system_memory_pressure_level"] not in (1, 2):
        return "critical or unavailable memory pressure"
    if sample["system_available_bytes"] < GIB:
        return "system available memory reserve"
    if swap is not None and swap >= (2 if soft else 4) * GIB:
        return "system swap growth"
    if sample["disk_free_bytes"] <= (10 if soft else 8) * GIB:
        return "checkpoint staging reserve" if soft else "free disk reserve"
    return None


def stop_group(child):
    """A dead wrapper is insufficient: every member must leave the process group."""
    for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if not group_members(child.pid):
            break
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            child.poll()
            if not group_members(child.pid):
                return child.wait()
            time.sleep(0.2)
    if group_members(child.pid):
        raise RuntimeError("training process group did not terminate")
    return child.wait()


def execution_arm(plan_path, tokens, calibration_lr=None):
    plan = verify_plan(plan_path)
    if calibration_lr is not None:
        if tokens != 32:
            raise ValueError("common learning-rate calibration requires 32 tokens")
        return plan, prepare_calibration(plan_path, calibration_lr)
    arm = select_arm(plan, tokens)
    if not plan["modelinit_verified"] or not arm.get("initialization_verified"):
        raise ValueError("initial checkpoints must be materialized and verified before execution")
    if not plan.get("ready_to_train"):
        raise ValueError("comparison prerequisites and common calibration remain pending")
    return plan, arm


def train_arm(plan_path, tokens, calibration_lr=None):
    from carryrl.training.sequence import train_sequence
    from carryrl.training.sequence_config import SequenceRunConfig

    plan, arm = execution_arm(plan_path, tokens, calibration_lr)
    stopping = False

    def requested(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGUSR1, requested)
    result = train_sequence(
        SequenceRunConfig.load(arm["config"]["path"]),
        Path(arm["training_output"]),
        warm_start=Path(arm["initial_checkpoint"]),
        agent_factory=make_agent_factory(arm["initial_checkpoint"]),
        stop_requested=lambda: stopping,
    )
    return 0 if result["status"] == "completed" else 2


def supervise(plan_path, tokens, calibration_lr=None):
    # The lock spans the entire arm, including evaluation and final checkpoint.
    with (Path(plan_path) / "execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another comparison arm is already running") from error
        return _supervise(plan_path, tokens, calibration_lr)


def _supervise(plan_path, tokens, calibration_lr=None):
    plan, arm = execution_arm(plan_path, tokens, calibration_lr)
    root, run = Path(plan["project_root"]), Path(arm["training_output"])
    if run.exists():
        raise FileExistsError("refusing to overwrite an existing arm")
    if shutil.disk_usage(root).free < 16 * GIB:
        raise RuntimeError("need at least 16 GiB free for model, staging, swap and disk reserve")
    launch = run.with_name(run.name + "_launch")
    launch.mkdir(exist_ok=False)
    report = {
        "status": "running",
        "started_utc": utc(),
        "supervisor_pid": os.getpid(),
        "maximum_rss_bytes": 12 * GIB,
        "minimum_free_bytes_limit": 8 * GIB,
        "peak_rss_bytes": 0,
        "minimum_free_bytes": None,
        "stop_reason": None,
        "actual_exit_code": None,
    }
    child = None
    started = time.monotonic()
    stop_at = None
    try:
        baseline_swap = swap_used_bytes()
        report["initial_system_swap_used_bytes"] = baseline_swap
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--plan", str(plan_path), "--tokens", str(tokens), "--train",
        ]
        if calibration_lr is not None:
            command.extend(["--calibrate-lr", str(calibration_lr)])
        with (
            (launch / "training.log").open("x") as log,
            (launch / "resources.jsonl").open("x") as journal,
        ):
            child = subprocess.Popen(
                command,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=dict(
                    os.environ,
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    TOKENIZERS_PARALLELISM="false",
                    PYTHONUNBUFFERED="1",
                ),
            )
            report["command_pid"] = child.pid
            publish(
                root / ".cache/active_learning_launch.json",
                {"run": str(run), "launch": str(launch)},
            )
            while child.poll() is None:
                sample = resources(child.pid, root)
                current_swap = sample["system_swap_used_bytes"]
                if baseline_swap is not None and current_swap is None:
                    raise ValueError("system swap metrics became unavailable")
                sample["system_swap_delta_bytes"] = (
                    max(0, current_swap - baseline_swap)
                    if current_swap is not None and baseline_swap is not None else None
                )
                report.update(
                    resource_sample=sample,
                    wall_seconds=time.monotonic() - started,
                    peak_rss_bytes=max(report["peak_rss_bytes"], sample["process_group_rss_bytes"]),
                )
                report["minimum_free_bytes"] = min(
                    report["minimum_free_bytes"] or sample["disk_free_bytes"],
                    sample["disk_free_bytes"],
                )
                journal.write(json.dumps(sample) + "\n")
                journal.flush()
                hard = resource_reason(sample)
                soft = resource_reason(sample, soft=True)
                if report["wall_seconds"] >= 12 * 3600:
                    hard = "supervisor wall budget"
                if hard:
                    report["stop_reason"] = hard
                    stop_group(child)
                    break
                if soft and stop_at is None:
                    report["stop_reason"] = soft
                    stop_at = time.monotonic()
                    os.kill(child.pid, signal.SIGUSR1)
                if stop_at is not None and time.monotonic() - stop_at > 20 * 60:
                    report["stop_reason"] = "graceful stop timed out"
                    stop_group(child)
                    break
                publish(launch / "process.json", report)
                time.sleep(2)
            report["actual_exit_code"] = child.wait()
            if group_members(child.pid):
                stop_group(child)
                raise RuntimeError("training left descendant processes")
    except BaseException as error:
        if child is not None:
            stop_group(child)
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report.update(status="exited", finished_utc=utc(), wall_seconds=time.monotonic() - started)
        publish(launch / "process.json", report)
    return 0 if report["actual_exit_code"] == 0 and not report.get("error") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tokens", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--calibrate-lr", type=float, choices=LEARNING_RATES)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--train", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.calibrate_lr is not None:
        if args.tokens != 32:
            parser.error("--calibrate-lr requires --tokens 32")
        if not (args.execute or args.train or args.supervise):
            parser.error("--calibrate-lr requires --execute")
    if args.train:
        return train_arm(args.plan, args.tokens, args.calibrate_lr)
    if args.supervise:
        return supervise(args.plan, args.tokens, args.calibrate_lr)
    plan = verify_plan(args.plan)
    arm = select_arm(plan, args.tokens)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "checked",
                    "tokens": args.tokens,
                    "initial_verified": plan["modelinit_verified"],
                    "training_started": False,
                    "output": arm["training_output"],
                }
            )
        )
        return 0
    plan, arm = execution_arm(args.plan, args.tokens, args.calibrate_lr)
    root = Path(plan["project_root"])
    if shutil.disk_usage(root).free < 16 * GIB:
        raise RuntimeError("need at least 16 GiB free before launch")
    suffix = (f"calibration_{calibration_label(args.calibrate_lr)}"
              if args.calibrate_lr is not None else f"t{args.tokens}")
    log_path = Path(args.plan) / f"supervisor_{suffix}.log"
    command = [
        sys.executable, str(Path(__file__).resolve()), "--plan", str(args.plan.resolve()),
        "--tokens", str(args.tokens), "--supervise",
    ]
    if args.calibrate_lr is not None:
        command.extend(["--calibrate-lr", str(args.calibrate_lr)])
    with log_path.open("x") as log:
        child = subprocess.Popen(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(json.dumps({"status": "dispatched", "supervisor_pid": child.pid}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
