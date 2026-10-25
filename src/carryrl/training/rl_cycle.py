"""Finite, fail-closed orchestration of the fresh all-sixteen-game RL cycle.

Jobs are explicit argv arrays. The parent process owns exit receipts; model JSON
cannot turn an unsuccessful process into a successful stage. Nothing auto-retries.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from carryrl.envs.atomic_suite import TASKS

VERSION = "carryrl.rl-cycle.v1"
GIB = 1024**3
BUDGET_SECONDS = 86400
FINAL_SECONDS = 7200
RESERVE_SECONDS = 7200
STAGE_LIMITS = {"rewards": 10800, "probe": 3600, "pilot": 14400,
                "full": 43200, "final": FINAL_SECONDS}
STYLES = ("canonical", "pixel_art")
HARD_TASKS = {"mouse", "platforming", "steering", "camera", "shooting"}


class CycleStopped(RuntimeError):
    """A gate, resource limit or immutable receipt prevents further computation."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def initialize_budget(output, *, start=False, now=None):
    """Start once before any corpus/preparation compute; subsequent calls preserve it."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "cycle_state.json"
    if path.exists():
        state = read_json(path)
        if state.get("version") != VERSION:
            raise ValueError("Unsupported cycle state")
    else:
        state = {"version": VERSION, "plan_sha256": None, "first_compute_unix": None,
                 "deadline_unix": None, "status": "prepared", "stages": {}, "decisions": {}}
    first, deadline = state.get("first_compute_unix"), state.get("deadline_unix")
    if (first is None) != (deadline is None):
        raise ValueError("Incomplete persistent cycle clock")
    if first is not None and (not math.isfinite(first) or not math.isfinite(deadline)
                              or deadline != first + BUDGET_SECONDS):
        raise ValueError("Persistent cycle clock must preserve the original 24-hour deadline")
    if start and first is None:
        state["first_compute_unix"] = time.time() if now is None else now
        state["deadline_unix"] = state["first_compute_unix"] + BUDGET_SECONDS
    publish(path, state)
    return state



def _validate_job_ids(plan):
    if plan.get("version") != VERSION:
        raise ValueError("Unsupported cycle plan")
    jobs = plan.get("jobs", [])
    ids = [job["id"] for job in jobs]
    if not jobs or len(ids) != len(set(ids)) or any(
            not isinstance(item, str) or not item.replace("_", "").replace("-", "").isalnum()
            for item in ids):
        raise ValueError("Cycle jobs need unique safe IDs")
    return set(ids)


def _snapshot_plan(output, path, digest):
    target = Path(output) / "plan_snapshots" / f"{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with target.open("xb") as stream:
            stream.write(Path(path).read_bytes())
    if sha256(target) != digest:
        raise CycleStopped("Archived cycle plan hash mismatch")
    return str(target)


def _repair_counts(state, *, verify_declared=True):
    history = state.get("repair_history", [])
    if state.get("repair_count", 0) != len(history):
        raise CycleStopped("Repair total does not match immutable repair history")
    kinds = [event.get("kind", "learning") for event in history]
    for audit in state.get("repair_reclassifications", []):
        index = audit["repair_index"] - 1
        if not 0 <= index < len(kinds) or kinds[index] != audit["from_kind"]:
            raise CycleStopped("Inconsistent repair classification history")
        kinds[index] = audit["to_kind"]
    if any(kind not in ("learning", "infrastructure") for kind in kinds):
        raise CycleStopped("Unknown repair classification")
    result = {"repair_count": len(history), "learning_repair_count": kinds.count("learning"),
              "infrastructure_repair_count": kinds.count("infrastructure")}
    if verify_declared and any(key in state and state[key] != value
                               for key, value in result.items()):
        raise CycleStopped("Repair counters disagree with their classification audit")
    return result


def _command_role(job):
    argv = job.get("argv", [])
    names = {Path(value).name for value in argv}
    if job.get("policy_output") or "train-v1" in argv or "run_cycle_policy.py" in names:
        return "policy"
    if "train-reward" in argv or any(name.startswith("train_") for name in names):
        return "fit"
    if any(name.startswith(("validate_", "evaluate_")) and name.endswith(".py") for name in names):
        return "validation"
    if any(name.startswith("prepare_") and name.endswith(".py") for name in names):
        return "preparation"
    return "unknown"


def _archived_plan(output, state):
    path = Path(state.get("plan_snapshot") or
                Path(output) / "plan_snapshots" / f"{state['plan_sha256']}.json")
    verify_pins({str(path): state["plan_sha256"]})
    return read_json(path)


def _completed_artifacts(state):
    result = {}
    for receipt in state["stages"].values():
        if receipt.get("status") != "completed":
            continue
        if receipt.get("kind") in ("pilot_select", "pilot_factorized_select", "full_horizon",
                                   "factorized_full_horizon", "freeze"):
            verify_pins(receipt["artifacts"])
        else:
            verify_receipt(receipt)
        result.update(receipt["artifacts"])
    return result


def _infrastructure_evidence(output, state):
    if any(_command_role(receipt) == "policy" for receipt in state["stages"].values()):
        raise CycleStopped("Infrastructure recovery is unavailable after any policy job started")
    stage = state.get("current_stage")
    receipt = state["stages"].get(stage, {})
    if receipt.get("status") != "stopped" or not receipt.get("argv"):
        raise CycleStopped("Infrastructure repair requires an identified failed command")
    verify_pins({receipt["log_path"]: receipt["log_sha256"]})
    role = _command_role(receipt)
    log = Path(receipt["log_path"]).read_text()
    argv = receipt["argv"]
    module = argv[argv.index("-m") + 1] if "-m" in argv else None
    # These interpreter errors prove the target module was never executed.
    bootstrap_failure = (receipt.get("actual_exit_code") not in (None, 0)
                         and module is not None and "Traceback" not in log
                         and len(log.strip().splitlines()) == 1
                         and (f"No module named {module}.__main__; '{module}' is a package "
                              "and cannot be directly executed" in log
                              or log.strip().endswith(f": No module named {module}")))
    if role not in ("validation", "preparation") and not bootstrap_failure:
        raise CycleStopped("Infrastructure repair cannot repeat a fit or change learned weights")
    return {"failed_stage": stage, "role": "cli_entrypoint" if bootstrap_failure else role,
            "failed_log": receipt["log_path"], "failed_log_sha256": receipt["log_sha256"]}


def _normalized_command(job):
    # A new immutable source snapshot may relocate scripts; all non-source
    # operands (configs, hyperparameters, datasets and output paths) stay literal.
    values = list(job.get("argv", []))
    if len(values) > 1 and values[1].endswith(".py"):
        values[1] = "script:" + Path(values[1]).name
    return values


def _check_infrastructure_plan(output, state, plan, evidence):
    old = _archived_plan(output, state)
    prior = {job["id"]: job for job in old["jobs"]}
    failed = state["stages"][evidence["failed_stage"]]
    original_failed = prior.get(evidence["failed_stage"])
    if original_failed is None or original_failed.get("argv") != failed.get("argv"):
        raise CycleStopped("Failed command does not match its immutable original plan")
    first = plan["jobs"][0]
    if evidence["role"] == "cli_entrypoint":
        # The failed -m module never ran. Retrying its fit is allowed solely as
        # an entrypoint correction, with the original config/data/output args.
        before, after = failed["argv"], first.get("argv", [])
        if "-m" not in before or "-m" not in after:
            raise CycleStopped("Entrypoint recovery must retain the explicit module invocation")
        b, a = before.index("-m"), after.index("-m")
        if (before[:b] != after[:a] or before[b + 2:] != after[a + 2:]
                or after[a + 1] not in (before[b + 1], before[b + 1] + ".cli")):
            raise CycleStopped("Entrypoint recovery cannot change fit arguments/target package")
    else:
        first_scripts = {Path(value).name for value in first.get("argv", [])
                         if value.endswith(".py")}
        failed_scripts = {Path(value).name for value in failed["argv"] if value.endswith(".py")}
        if _command_role(first) != evidence["role"] or not first_scripts & failed_scripts:
            raise CycleStopped(
                "Infrastructure recovery must first retry the failed checker/preparer")
        relocated = any(new != old and Path(new).name == Path(old).name
                        and new in plan.get("pins", {})
                        for new in first.get("argv", []) if new.endswith(".py")
                        for old in failed["argv"] if old.endswith(".py"))
        if not relocated:
            raise CycleStopped(
                "Infrastructure checker recovery requires a new pinned source snapshot")
        def operands(argv):
            values = _normalized_command({"argv": argv})
            if "--output" in values:
                values[values.index("--output") + 1] = "<new preserved-output path>"
            return values
        if operands(first["argv"]) != operands(failed["argv"]):
            raise CycleStopped("Infrastructure checker recovery cannot change validation arguments")
        if "--output" in failed["argv"]:
            old_output = failed["argv"][failed["argv"].index("--output") + 1]
            if Path(old_output).exists() and old_output in first.get("argv", []):
                raise CycleStopped("Infrastructure recovery must preserve the failed output path")
    for job in plan["jobs"][1:]:
        if _command_role(job) in ("fit", "policy"):
            original = prior.get(job["id"])
            if (original is None or job["id"] in state["stages"]
                    or _normalized_command(job) != _normalized_command(original)):
                raise CycleStopped(
                    "Infrastructure recovery cannot introduce or repeat a fit/policy")
    # Preserve original non-code input pins for checkers/preparers as well as
    # future fits: an unchanged path cannot silently introduce different weights.
    # Source changes remain explicit in the new immutable snapshot.
    for job in plan["jobs"]:
        if _command_role(job) not in ("fit", "policy", "validation", "preparation"):
            continue
        for value in job.get("argv", []):
            expected = old.get("pins", {}).get(value)
            if expected is not None and not value.endswith(".py"):
                if plan.get("pins", {}).get(value) != expected:
                    raise CycleStopped(
                        "Infrastructure recovery cannot change pinned command inputs")


def reclassify_repair(output, repair_index, reason, *, now=None):
    """Explicit, evidence-backed migration of one legacy repair to infrastructure."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("An explicit reclassification reason is required")
    output = Path(output)
    with (output / "cycle.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CycleStopped("Cannot reclassify a cycle owned by another launcher") from error
        state = initialize_budget(output)
        counts = _repair_counts(state)
        if state["status"] != "stopped":
            raise CycleStopped("Reclassification requires a stopped cycle")
        if type(repair_index) is not int or not 1 <= repair_index <= counts["repair_count"]:
            raise ValueError("Repair index must identify an existing one-based history entry")
        if any(audit["repair_index"] == repair_index
               for audit in state.get("repair_reclassifications", [])):
            raise CycleStopped("This repair already has an explicit classification audit")
        event = state["repair_history"][repair_index - 1]
        if event.get("kind", "learning") != "learning":
            raise CycleStopped("Only a learning/legacy repair can be reclassified")
        assert_prior_processes_quiescent(state)
        _completed_artifacts(state)
        verify_pins({event["previous_state"]: event["previous_state_sha256"]})
        previous = read_json(event["previous_state"])
        evidence = _infrastructure_evidence(output, previous)
        verify_pins({event["new_plan_snapshot"]: event["new_plan_sha256"]})
        _check_infrastructure_plan(
            output, previous, read_json(event["new_plan_snapshot"]), evidence)
        archive = output / "repair_history" / f"before_reclassification_{repair_index}.json"
        if archive.exists():
            raise CycleStopped("Reclassification archive already exists")
        publish(archive, state)
        audit = {"repair_index": repair_index, "from_kind": "learning",
                 "to_kind": "infrastructure", "reason": reason.strip(),
                 "requested_unix": time.time() if now is None else now,
                 "previous_state": str(archive), "previous_state_sha256": sha256(archive),
                 "evidence": evidence}
        state.setdefault("repair_reclassifications", []).append(audit)
        state.update(_repair_counts(state, verify_declared=False))
        publish(output / "cycle_state.json", state)
        return state


def authorize_repair(output, new_plan, reason, *, now=None, repair_kind="learning"):
    """Explicit recovery without resetting clocks, evidence or the learning quota."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("An explicit repair reason is required")
    if repair_kind not in ("learning", "infrastructure"):
        raise ValueError("repair_kind must be learning or infrastructure")
    output, new_plan = Path(output), Path(new_plan).resolve(strict=True)
    with (output / "cycle.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CycleStopped("Cannot repair a cycle owned by another launcher") from error
        state = initialize_budget(output)
        counts = _repair_counts(state)
        instant = time.time() if now is None else now
        if state["status"] != "stopped" or (repair_kind == "learning"
                                          and counts["learning_repair_count"] >= 1):
            raise CycleStopped("Only one explicit learning repair of a stopped cycle is allowed")
        if state["deadline_unix"] is not None and instant >= state["deadline_unix"]:
            raise CycleStopped("Repair cannot extend an expired cycle budget")
        assert_prior_processes_quiescent(state)
        plan = read_json(new_plan)
        ids = _validate_job_ids(plan)
        if ids & set(state["stages"]):
            raise CycleStopped("Repair must use new job IDs; prior receipts are immutable")
        preserved = _completed_artifacts(state)
        verify_pins(plan.get("pins", {}))
        evidence = None
        if repair_kind == "infrastructure":
            evidence = _infrastructure_evidence(output, state)
            _check_infrastructure_plan(output, state, plan, evidence)
        digest = sha256(new_plan)
        snapshot = _snapshot_plan(output, new_plan, digest)
        number = counts["repair_count"] + 1
        previous_path = output / "repair_history" / f"before_repair_{number}.json"
        if previous_path.exists():
            raise CycleStopped("Prior repair history already exists")
        publish(previous_path, state)
        event = {"kind": repair_kind, "reason": reason.strip(), "requested_unix": instant,
                 "previous_plan_sha256": state["plan_sha256"], "new_plan_sha256": digest,
                 "previous_state": str(previous_path),
                 "previous_state_sha256": sha256(previous_path), "new_plan_snapshot": snapshot}
        if evidence is not None:
            event["evidence"] = evidence
            state.setdefault("infrastructure_preserved_artifacts", {}).update(preserved)
        state.setdefault("original_plan_sha256", state["plan_sha256"])
        state.update(plan_sha256=digest, status="prepared", current_stage=None)
        state.setdefault("repair_history", []).append(event)
        state.update(_repair_counts({**state, "repair_count": number}, verify_declared=False))
        state.pop("error", None)
        state.pop("finished_unix", None)
        publish(output / "cycle_state.json", state)
        return state


def _counts(row, episodes):
    successes = row.get("successes")
    if (row.get("episodes") != episodes or type(successes) is not int
            or not 0 <= successes <= episodes):
        raise ValueError(f"Expected explicit integer successes/{episodes} episodes")
    return successes


def _styles(report, *, mode):
    if report.get("split") != "validation" or report.get("mode") != mode:
        raise ValueError("Selection requires the pinned validation split and action mode")
    styles = report.get("styles", {})
    if set(styles) != set(STYLES):
        raise ValueError("Both canonical and pixel-art styles are required")
    for style in STYLES:
        row = styles[style]
        _counts(row, 256)
        games = row.get("games", {})
        if set(games) != set(TASKS):
            raise ValueError("Validation must include all sixteen games")
        if sum(_counts(games[task], 16) for task in TASKS) != row["successes"]:
            raise ValueError("Aggregate validation counts differ from per-game counts")
    return styles


def pilot_eligibility(report, cp0, *, expected_rollouts=32):
    """Evaluate quality/breadth using outcomes, never loss or shaped reward."""
    styles, initial = _styles(report, mode="stochastic"), _styles(cp0, mode="stochastic")
    if expected_rollouts not in (32, 80):
        raise ValueError("Unsupported registered pilot duration")
    if report.get("seed") != 3176 or report.get("rollouts") != expected_rollouts:
        raise ValueError(f"Pilot must complete {expected_rollouts} rollouts at seed 3176")
    pin = report.get("initial_trainable_sha256")
    if (not isinstance(pin, str) or len(pin) != 64
            or pin != cp0.get("initial_trainable_sha256")):
        raise ValueError("Paired pilot initial trainable hash differs from fresh CP0")
    style_gates = {}
    for style in STYLES:
        games = styles[style]["games"]
        improved = [task for task in TASKS if games[task]["successes"]
                    > initial[style]["games"][task]["successes"]]
        delta = styles[style]["successes"] - initial[style]["successes"]
        style_gates[style] = {"success_delta": delta, "improved_games": improved,
                              "passed": delta >= 16 and len(improved) >= 4}
    train = report.get("train", {}).get("games", {})
    if set(train) != set(TASKS):
        raise ValueError("Pilot TRAIN outcome evidence is required for all games")
    mastered = []
    for task, row in train.items():
        levels = row.get("successful_effective_levels")
        if not isinstance(levels, list) or any(not isinstance(item, str) for item in levels):
            raise ValueError("TRAIN success must identify unique effective level IDs")
        if len(set(levels)) >= 2:
            mastered.append(task)
    hard = sorted(set(mastered) & HARD_TASKS)
    train_gate = len(mastered) >= 8 and len(hard) >= 2
    return {"eligible": all(row["passed"] for row in style_gates.values()) and train_gate,
            "styles": style_gates, "train_games": sorted(mastered), "hard_games": hard,
            "train_passed": train_gate}


def choose_pilot(cp0, zero, temporary):
    results = {"zero": pilot_eligibility(zero, cp0),
               "temporary": pilot_eligibility(temporary, cp0)}
    eligible = [name for name, value in results.items() if value["eligible"]]
    selected = eligible[0] if len(eligible) == 1 else None
    if len(eligible) == 2:
        better = all(temporary["styles"][style]["successes"]
                     >= zero["styles"][style]["successes"] + 8 for style in STYLES)
        breadth = all(len(results["temporary"]["styles"][style]["improved_games"])
                      >= len(results["zero"]["styles"][style]["improved_games"])
                      for style in STYLES)
        breadth &= len(results["temporary"]["train_games"]) >= len(results["zero"]["train_games"])
        selected = "temporary" if better and breadth else "zero"
    return {"selected_schedule": selected, "branches": results,
            "reason": "no eligible branch; full training prohibited" if selected is None
            else "eligible branch selected by frozen pilot rule"}


def choose_full_horizon(rollout_seconds, remaining_seconds, *, evaluation_overhead_seconds=2700,
                        repair_reserve_seconds=RESERVE_SECONDS):
    """Conservative observed maximum plus 25%; full training never starts below 128."""
    if (not isinstance(rollout_seconds, list) or len(rollout_seconds) < 2
            or any(type(item) not in (int, float) or not math.isfinite(item) or item <= 0
                   for item in rollout_seconds)):
        raise ValueError("At least two measured positive rollout timings are required")
    if (type(evaluation_overhead_seconds) not in (int, float)
            or not math.isfinite(evaluation_overhead_seconds) or evaluation_overhead_seconds < 0):
        raise ValueError("Evaluation/checkpoint overhead must be finite nonnegative seconds")
    if (type(repair_reserve_seconds) not in (int, float)
            or not math.isfinite(repair_reserve_seconds)
            or not 0 <= repair_reserve_seconds <= RESERVE_SECONDS):
        raise ValueError("Repair reserve must be finite seconds in 0..7200")
    conservative = max(rollout_seconds) * 1.25
    available = (min(STAGE_LIMITS["full"], remaining_seconds - FINAL_SECONDS
                     - repair_reserve_seconds) - evaluation_overhead_seconds)
    horizon = min(192, max(0, math.floor(available / conservative / 32) * 32))
    return {"full_rollouts": horizon if horizon >= 128 else None,
            "conservative_seconds_per_rollout": conservative,
            "available_training_seconds": max(0, available),
            "evaluation_overhead_seconds": evaluation_overhead_seconds,
            "repair_reserve_seconds": repair_reserve_seconds,
            "control_schedule_rollouts": 32}


def select_final(candidates):
    """Frozen validation selection: minimum greedy game/style, mean, earlier checkpoint."""
    if not candidates:
        raise ValueError("No frozen-validation candidates")
    ranked = []
    for candidate in candidates:
        report = candidate["report"]
        styles = _styles(report, mode="greedy")
        if report.get("frozen_validation") is not True:
            raise ValueError("Final selection requires frozen validation reports")
        rollout = candidate["rollout"]
        if type(rollout) is not int or rollout < 0:
            raise ValueError("Invalid selection checkpoint rollout")
        counts = [styles[style]["games"][task]["successes"]
                  for style in STYLES for task in TASKS]
        ranked.append(((min(counts) / 16, sum(counts) / 512, -rollout), candidate))
    score, chosen = max(ranked, key=lambda item: item[0])
    return {"checkpoint": chosen["checkpoint"], "rollout": chosen["rollout"],
            "minimum_success_rate": score[0], "mean_success_rate": score[1],
            "model_frozen": True, "test_allowed": True}


def should_stop_full(rollout, evaluations, train_stage_advanced):
    """Two consecutive evaluations without improvement and no TRAIN advancement."""
    if rollout < 128 or len(evaluations) < 3 or train_stage_advanced:
        return False
    if any(not math.isfinite(value) for value in evaluations):
        raise ValueError("Full evaluation scores must be finite")
    best_before = max(evaluations[:-2])
    return max(evaluations[-2:]) <= best_before


def process_identity(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "pid=,pgid=,lstart=,args="],
                            capture_output=True, text=True, check=False)
    if result.returncode or not result.stdout.strip():
        return None
    fields = result.stdout.strip().split(None, 7)
    if len(fields) != 8:
        raise CycleStopped("Cannot verify subprocess identity")
    return {"pid": int(fields[0]), "pgid": int(fields[1]),
            "started": " ".join(fields[2:7]), "command": fields[7]}



def process_group_pids(group):
    result = subprocess.run(["ps", "-axo", "pid=,pgid="], check=True,
                            capture_output=True, text=True)
    return [int(fields[0]) for line in result.stdout.splitlines()
            if len(fields := line.split()) == 2 and int(fields[1]) == group]


def assert_receipt_quiescent(receipt, *, stage_id):
    """Never start a second job alongside a surviving or unverified prior actor.

    This check sends no signals. A reused PID is never treated as the original
    actor; an unanchored surviving group requires explicit manual reconciliation.
    """
    expected = receipt.get("process_identity")
    if expected is None:
        uncertain = (receipt.get("status") in ("starting", "running") or
                     (receipt.get("phase") in STAGE_LIMITS
                      and receipt.get("actual_exit_code") is None
                      and receipt.get("process_complete") is not True))
        if uncertain:
            raise CycleStopped(f"Prior stage {stage_id} has no verified process identity/exit; "
                               "manual process reconciliation is required")
        return
    if process_identity(expected["pid"]) == expected:
        state = "paused" if receipt.get("memory_paused") else "alive"
        raise CycleStopped(f"Prior owned process remains alive ({state}): stage={stage_id}, "
                           f"identity={json.dumps(expected, sort_keys=True)}")
    anchors = receipt.get("owned_group_identities", [])
    for anchor in anchors:
        if anchor != expected and process_identity(anchor["pid"]) == anchor:
            raise CycleStopped(f"Prior owned group member remains alive: stage={stage_id}, "
                               f"identity={json.dumps(anchor, sort_keys=True)}")
    members = process_group_pids(expected["pgid"])
    if members:
        raise CycleStopped(f"Prior stage {stage_id} group {expected['pgid']} still has members "
                           f"{members}; ownership is ambiguous; manual reconciliation required")


def assert_prior_processes_quiescent(state):
    for stage_id, receipt in state["stages"].items():
        assert_receipt_quiescent(receipt, stage_id=stage_id)


def verify_pins(pins):
    if not isinstance(pins, dict):
        raise ValueError("Input pins must map paths to SHA256 hashes")
    for path, expected in pins.items():
        if len(expected) != 64 or sha256(path) != expected:
            raise CycleStopped(f"Pinned input drift: {path}")


def disk_preflight(directory, forecast_bytes=0):
    if type(forecast_bytes) is not int or forecast_bytes < 0:
        raise ValueError("Disk forecast must be nonnegative bytes")
    free = shutil.disk_usage(directory).free
    if free < 8 * GIB + forecast_bytes:
        raise CycleStopped("Disk forecast would consume the required 8 GiB headroom")
    return {"free_bytes": free, "forecast_bytes": forecast_bytes, "minimum_headroom_bytes": 8 * GIB}



def resource_sample(group=None):
    """Read system pressure and RSS without importing model runtimes."""
    if sys.platform == "darwin":
        vm = subprocess.run(["vm_stat"], check=True, capture_output=True, text=True).stdout
        page = re.search(r"page size of (\d+) bytes", vm)
        counts = dict(re.findall(r"^([^:\n]+):\s+(\d+)\.", vm, re.MULTILINE))
        names = ("Pages free", "Pages inactive", "Pages speculative")
        if page is None or not all(name in counts for name in names):
            raise CycleStopped("Unreadable memory counters")
        available = int(page[1]) * sum(int(counts[name]) for name in names)
        pressure = int(subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            check=True, capture_output=True, text=True).stdout)
    else:
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        pressure = 1
    members, rss = [], 0
    if group is not None:
        result = subprocess.run(["ps", "-axo", "pid=,pgid=,rss="], check=True,
                                capture_output=True, text=True)
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) == 3 and int(fields[1]) == group:
                members.append(int(fields[0]))
                rss += int(fields[2]) * 1024
    return {"available_bytes": available, "pressure": pressure,
            "process_group_rss_bytes": rss, "group_members": members}


def memory_reason(sample, *, paused=False):
    if (type(sample.get("available_bytes")) is not int or sample["available_bytes"] < 0
            or type(sample.get("process_group_rss_bytes")) is not int):
        return "invalid memory metrics"
    if sample.get("pressure") not in (1, 2):
        return "critical memory pressure"
    if sample["process_group_rss_bytes"] >= 12 * GIB:
        return "owned group RSS >= 12 GiB"
    if sample["available_bytes"] < (3 * GIB // 2 if paused else GIB):
        return "available memory reserve"
    return None


def _field(value, dotted):
    for part in dotted.split("."):
        value = value[part]
    return value


def check_artifacts(specs):
    """Require concrete files, record hashes, optionally check explicit JSON gates."""
    checked = {}
    if not specs:
        raise CycleStopped("A compute job requires at least one verifiable output artifact")
    for spec in specs:
        path = Path(spec["path"])
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise CycleStopped(f"Missing or empty stage artifact: {path}")
        digest = sha256(path)
        if spec.get("sha256") is not None and spec["sha256"] != digest:
            raise CycleStopped(f"Stage artifact hash mismatch: {path}")
        for field, expected in spec.get("json_equals", {}).items():
            try:
                actual = _field(read_json(path), field)
            except (KeyError, TypeError, ValueError) as error:
                raise CycleStopped(f"Unreadable artifact gate {path}:{field}") from error
            if type(actual) is not type(expected) or actual != expected:
                raise CycleStopped(f"Stage artifact gate failed: {path}:{field}")
        checked[str(path)] = digest
    return checked


def verify_receipt(receipt):
    if (receipt.get("status") != "completed" or receipt.get("actual_exit_code") != 0
            or receipt.get("process_complete") is not True
            or not receipt.get("process_identity") or not receipt.get("log_sha256")):
        raise CycleStopped("Stage lacks a verified successful process/exit/log receipt")
    assert_receipt_quiescent(receipt, stage_id="imported_or_completed")
    verify_pins({receipt["log_path"]: receipt["log_sha256"], **receipt["artifacts"]})


class CycleRunner:
    def __init__(self, plan_path, output, *, clock=time.time, sleep=time.sleep):
        self.output = Path(output).resolve()
        self.path = self.output / "cycle_state.json"
        self.state = initialize_budget(self.output)
        self.plan_path = Path(plan_path).resolve(strict=True)
        self.plan = read_json(self.plan_path)
        self.clock, self.sleep = clock, sleep
        digest = sha256(self.plan_path)
        if self.state["plan_sha256"] not in (None, digest):
            raise CycleStopped("Cycle plan changed; budget and completed stages cannot be reset")
        _validate_job_ids(self.plan)
        self.state["plan_sha256"] = digest
        if self.state.get("original_plan_sha256") is None:
            self.state["original_plan_sha256"] = digest
        self.state["plan_snapshot"] = _snapshot_plan(self.output, self.plan_path, digest)
        self.save()

    def save(self):
        publish(self.path, self.state)

    def remaining(self):
        deadline = self.state["deadline_unix"]
        return BUDGET_SECONDS if deadline is None else max(0, deadline - self.clock())

    def preflight(self):
        if sha256(self.plan_path) != self.state["plan_sha256"]:
            raise CycleStopped("Cycle plan drifted during execution")
        verify_pins(self.plan.get("pins", {}))
        verify_pins(self.state.get("infrastructure_preserved_artifacts", {}))
        disk = disk_preflight(self.output, self.plan.get("forecast_bytes", 0))
        memory = resource_sample()
        reason = memory_reason(memory)
        if reason:
            raise CycleStopped(reason)
        return {"memory": memory,"remaining_seconds": self.remaining(), "disk": disk,
                "training_started": False, "plan_sha256": self.state["plan_sha256"]}

    def _expand(self, value):
        if isinstance(value, str):
            for key, replacement in self.state["decisions"].items():
                if isinstance(replacement, (str, int, float)):
                    value = value.replace("${" + key + "}", str(replacement))
            if "${" in value:
                raise CycleStopped(f"Unresolved cycle decision in {value}")
        elif isinstance(value, list):
            value = [self._expand(item) for item in value]
        elif isinstance(value, dict):
            value = {key: self._expand(item) for key, item in value.items()}
        return value

    def _decision(self, job):
        kind = job["kind"]
        failure = None
        if kind == "pilot_select":
            result = choose_pilot(*(read_json(job[name]) for name in ("cp0", "zero", "temporary")))
            if result["selected_schedule"] is None:
                failure = "Neither pilot passed; full training prohibited"
        elif kind == "pilot_factorized_select":
            from carryrl.training.factorized_cycle import BRANCHES, choose_factorized_pilot

            result = choose_factorized_pilot(
                read_json(job["cp0"]), {name: read_json(job[name]) for name in BRANCHES})
            branch = result["selected_branch"]
            result["selected_schedule"] = (
                None if branch is None else "zero" if branch == "zero" else "temporary")
            if branch is None:
                failure = "No factorized pilot passed; full training prohibited"
        elif kind == "factorized_full_horizon":
            from carryrl.training.factorized_cycle import choose_factorized_full_horizon

            if self.plan.get("budget_protocol") != "remaining_global_phase_caps.v1":
                raise ValueError("Factorized full horizon requires explicit phase caps")
            cap = self.plan.get("phase_limits_seconds", {}).get("full")
            if (type(cap) not in (int, float) or not math.isfinite(cap)
                    or not 0 < cap <= BUDGET_SECONDS):
                raise ValueError("Factorized full horizon requires a finite full-phase cap")
            used = sum(row.get("elapsed_seconds", 0) for row in self.state["stages"].values()
                       if row.get("phase") == "full")
            repair_reserve = (0 if _repair_counts(self.state)["learning_repair_count"]
                              else RESERVE_SECONDS)
            overhead = job.get("launch_overhead_seconds", 0)
            runtime_limit = job.get("runtime_limit_seconds", cap)
            if (type(overhead) not in (int, float) or not math.isfinite(overhead) or overhead < 0
                    or type(runtime_limit) not in (int, float)
                    or not math.isfinite(runtime_limit) or runtime_limit <= 0):
                raise ValueError("Full horizon requires finite nonnegative launch overhead "
                                 "and a positive runtime limit")
            available = min(cap - used - overhead,
                            self.remaining() - FINAL_SECONDS - repair_reserve - overhead,
                            runtime_limit)
            if available <= 0:
                raise CycleStopped("No budget remains for factorized full training")
            result = choose_factorized_full_horizon(
                read_json(job["probe_report"])["rollout_seconds"],
                read_json(job["timing_report"])["evaluation_seconds"], available)
            result["repair_reserve_seconds"] = repair_reserve
            result["launch_overhead_seconds"] = overhead
            result["runtime_limit_seconds"] = runtime_limit
            if result["full_rollouts"] is None:
                failure = "Measured remaining budget cannot support a fresh 128-rollout full run"
        elif kind == "full_horizon":
            result = choose_full_horizon(read_json(job["probe_report"])["rollout_seconds"],
                                         self.remaining(), evaluation_overhead_seconds=
                                         job.get("evaluation_overhead_seconds", 2700),
                                         repair_reserve_seconds=(0 if _repair_counts(self.state)[
                                             "learning_repair_count"] else RESERVE_SECONDS))
            if result["full_rollouts"] is None:
                failure = "Remaining budget cannot support a fresh 128-rollout full run"
            if self.state["decisions"].get("selected_branch") is not None:
                result["control_schedule_rollouts"] = 64
        elif kind == "freeze":
            selection = None
            if job.get("selection_report") is not None:
                selection = read_json(job["selection_report"])
                result = select_final(selection["candidates"])
                if any(selection.get(key) != value for key, value in result.items()):
                    raise CycleStopped(
                        "Selection sidecar disagrees with recomputed validation rank")
            else:
                result = select_final([{**item, "report": read_json(item["report"])}
                                       for item in job["candidates"]])
            checkpoint = Path(result["checkpoint"])
            manifest_digest = sha256(checkpoint / "manifest.json")
            if selection is not None:
                if selection.get("checkpoint_manifest_sha256") != manifest_digest:
                    raise CycleStopped("Selection sidecar checkpoint hash is stale")
                result["selection_report_sha256"] = sha256(job["selection_report"])
            from carryrl.training.checkpoint_storage import checkpoint_read_lock

            # Cold snapshots restore through the existing verified, disk-bounded path.
            # The lock prevents archival while the selected immutable bytes are pinned.
            with checkpoint_read_lock(checkpoint) as payload:
                manifest = read_json(checkpoint / "manifest.json")
                if sha256(payload) != manifest["sha256"]:
                    raise CycleStopped("Selected checkpoint weights differ from their manifest")
                members = list(checkpoint.rglob("*"))
                if any(member.is_symlink() for member in members):
                    raise CycleStopped("Selected checkpoint cannot contain symbolic links")
                result["frozen_checkpoint_files"] = {
                    str(member): sha256(member) for member in members if member.is_file()}
                result["checkpoint_manifest_sha256"] = sha256(checkpoint / "manifest.json")
        else:
            raise ValueError(f"Unsupported decision job: {kind}")
        if failure is not None:
            result["reason"] = failure
        self.state["decisions"].update(result)
        path = self.output / "stages" / job["id"] / "decision.json"
        publish(path, result)
        receipt = {"status": "stopped" if failure else "completed", "kind": kind,
                   "artifacts": {str(path): sha256(path)}}
        if failure is not None:
            receipt["reason"] = failure
            self.state["stages"][job["id"]] = receipt
            self.state["current_stage"] = job["id"]
            self.save()
            raise CycleStopped(failure)
        return receipt

    def _stop_child(self, child, expected, *, grace=10):
        # Signals target only this Popen session after verifying its live identity.
        current = process_identity(child.pid)
        if current is not None and current == expected:
            # A paused process must run again to handle the graceful termination.
            if grace <= 0:
                os.killpg(expected["pgid"], signal.SIGKILL)
            else:
                os.killpg(expected["pgid"], signal.SIGCONT)
                os.killpg(expected["pgid"], signal.SIGTERM)
            try:
                child.wait(timeout=max(.01, grace))
            except subprocess.TimeoutExpired:
                if process_identity(child.pid) == expected:
                    os.killpg(expected["pgid"], signal.SIGKILL)
        elif child.poll() is None:
            child.kill()
        return child.wait(timeout=15)

    def _compute(self, job):
        phase = job["phase"]
        if phase not in STAGE_LIMITS:
            raise ValueError("Compute phase must have an explicit stage budget")
        if phase == "full" and (self.state["decisions"].get("selected_schedule") is None
                                or self.state["decisions"].get("full_rollouts") is None):
            raise CycleStopped("Full training requires passing pilot and timing decisions")
        argv = job.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(a, str) for a in argv):
            raise ValueError("Each command must be a nonempty explicit argv array")
        test_requested = job.get("split") == "test" or "--split=test" in argv or any(
            argv[index:index + 2] == ["--split", "test"] for index in range(len(argv)))
        if test_requested:
            if not self.state["decisions"].get("model_frozen"):
                raise CycleStopped("Test evaluation is prohibited before model freeze")
            checkpoint = Path(self.state["decisions"]["checkpoint"]) / "manifest.json"
            if sha256(checkpoint) != self.state["decisions"]["checkpoint_manifest_sha256"]:
                raise CycleStopped("Frozen selected checkpoint changed before test")
            verify_pins(self.state["decisions"]["frozen_checkpoint_files"])
        verify_pins(self.plan.get("pins", {}))
        verify_pins(job.get("pins", {}))
        verify_pins(self.state.get("infrastructure_preserved_artifacts", {}))
        preflight = disk_preflight(self.output, job.get("forecast_bytes", 0))
        phase_used = sum(row.get("elapsed_seconds", 0) for row in self.state["stages"].values()
                         if row.get("phase") == phase)
        elapsed = (0 if self.state["first_compute_unix"] is None else
                   max(0, self.clock() - self.state["first_compute_unix"]))
        elapsed = max(elapsed, sum(row.get("elapsed_seconds", 0)
                                  for row in self.state["stages"].values()))
        prefix = 0
        for name, seconds in STAGE_LIMITS.items():
            prefix += seconds
            if name == phase:
                break
        repair_credit = (RESERVE_SECONDS if _repair_counts(self.state)["learning_repair_count"]
                         else 0)
        reserve = (0 if phase == "final" else FINAL_SECONDS) + RESERVE_SECONDS - repair_credit
        prefix_available = prefix + repair_credit - elapsed
        limit = min(prefix_available, self.remaining() - reserve,
                    job.get("max_seconds", prefix_available))
        base_available = max(0, STAGE_LIMITS[phase] - phase_used)
        budget_protocol = self.plan.get("budget_protocol", "historical_prefix.v1")
        if budget_protocol == "remaining_global_phase_caps.v1":
            caps = self.plan.get("phase_limits_seconds", {})
            if (set(caps) != set(STAGE_LIMITS)
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           or not 0 < value <= BUDGET_SECONDS for value in caps.values())):
                raise ValueError("Revised cycle requires explicit finite phase caps")
            base_available = max(0, caps[phase] - phase_used)
            limit = min(base_available, self.remaining() - reserve,
                        job.get("max_seconds", base_available))
        elif budget_protocol != "historical_prefix.v1":
            raise ValueError("Unsupported cycle budget protocol")
        if not math.isfinite(limit) or limit <= 0:
            raise CycleStopped("No finite elapsed budget remains for this phase")
        grace = job.get("graceful_seconds", 0)
        if type(grace) not in (int, float) or not math.isfinite(grace) or not 0 <= grace < limit:
            raise ValueError("graceful_seconds must be finite and below the stage limit")
        if self.state["first_compute_unix"] is None:
            self.state = initialize_budget(self.output, start=True, now=self.clock())
        stage = self.output / "stages" / job["id"]
        stage.mkdir(parents=True, exist_ok=True)
        log_path = stage / "stdout.log"
        receipt = {"status": "starting", "phase": phase, "argv": argv,
                   "started_unix": self.clock(), "max_seconds": limit,
                   "actual_exit_code": None, "process_complete": False,
                   "log_path": str(log_path), "disk_preflight": preflight,
                   "policy_output": job.get("policy_output"),
                   "base_phase_seconds": (self.plan["phase_limits_seconds"][phase]
                                          if budget_protocol == "remaining_global_phase_caps.v1"
                                          else STAGE_LIMITS[phase]),
                   "budget_protocol": budget_protocol,
                   "base_phase_remaining_seconds": base_available,
                   "cumulative_prefix_seconds": prefix,
                   "cumulative_elapsed_seconds": elapsed,
                   "available_borrowed_seconds": max(0, limit - base_available),
                   "repair_credit_available_seconds": repair_credit}
        self.state["stages"][job["id"]] = receipt
        self.save()
        child, identity = None, None
        paused, next_sample = False, 0.0
        try:
            env = {**os.environ, "CARRYRL_CYCLE_DEADLINE_UNIX": str(self.state["deadline_unix"]),
                   "CARRYRL_STAGE_MAX_SECONDS": str(limit)}
            with log_path.open("xb") as log:
                child = subprocess.Popen(argv, cwd=self.plan.get("cwd"), env=env,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                identity = process_identity(child.pid)
                if identity is None or identity["pgid"] != child.pid:
                    raise CycleStopped("Could not establish owned process session")
                receipt.update(status="running", process_identity=identity,
                               owned_group_identities=[identity])
                self.save()
                while child.poll() is None:
                    if self.clock() - receipt["started_unix"] >= limit or self.remaining() <= 0:
                        raise CycleStopped("Elapsed stage/global deadline reached")
                    if (grace and not receipt.get("graceful_stop_requested")
                            and self.clock() - receipt["started_unix"] >= limit - grace):
                        if process_identity(child.pid) != identity:
                            raise CycleStopped("Lost process identity before graceful cutoff")
                        if paused:
                            os.killpg(identity["pgid"], signal.SIGCONT)
                            paused = False
                        os.killpg(identity["pgid"], signal.SIGTERM)
                        receipt.update(graceful_stop_requested=True,
                                       graceful_stop_unix=self.clock())
                        self.save()
                    disk_preflight(self.output)
                    if self.clock() >= next_sample:
                        sample = resource_sample(identity["pgid"])
                        for member in sample.get("group_members", []):
                            anchor = process_identity(member)
                            if (anchor is not None and anchor["pgid"] == identity["pgid"]
                                    and anchor not in receipt["owned_group_identities"]):
                                receipt["owned_group_identities"].append(anchor)
                        reason = memory_reason(sample, paused=paused)
                        if reason is not None and not paused:
                            if process_identity(child.pid) != identity:
                                raise CycleStopped("Lost process identity before memory pause")
                            os.killpg(identity["pgid"], signal.SIGSTOP)
                            paused = True
                        elif reason is None and paused:
                            if process_identity(child.pid) != identity:
                                raise CycleStopped("Lost process identity before memory resume")
                            os.killpg(identity["pgid"], signal.SIGCONT)
                            paused = False
                        receipt.update(memory_paused=paused, resource_sample=sample,
                                       memory_pause_reason=reason)
                        with (stage / "resources.jsonl").open("a") as journal:
                            journal.write(json.dumps({"unix": self.clock(), **sample,
                                                      "paused": paused}) + "\n")
                        self.save()
                        next_sample = self.clock() + 5
                    self.sleep(min(1, max(.01, limit)))
                receipt["actual_exit_code"] = child.wait()
            remaining_members = resource_sample(identity["pgid"]).get("group_members", [])
            receipt["process_complete"] = (process_identity(child.pid) != identity
                                           and not remaining_members)
            if receipt["actual_exit_code"] != 0 or not receipt["process_complete"]:
                raise CycleStopped("Subprocess did not complete successfully")
            if self.clock() - receipt["started_unix"] > limit or self.remaining() <= 0:
                raise CycleStopped("Stage finished beyond its elapsed deadline")
            receipt["artifacts"] = check_artifacts(job.get("artifacts", []))
            receipt["status"] = "completed"
        except BaseException as error:
            if child is None:
                receipt["process_complete"] = True
            if child is not None and child.poll() is None:
                hard = self.remaining() <= 0 or self.clock() - receipt["started_unix"] >= limit
                receipt["actual_exit_code"] = self._stop_child(
                    child, identity, grace=0 if hard else 10)
            if identity is not None:
                try:
                    remaining_members = resource_sample(identity["pgid"]).get("group_members", [])
                    anchors = [process_identity(pid) for pid in remaining_members]
                    if any(anchor is not None and anchor["pgid"] == identity["pgid"]
                           for anchor in anchors):
                        os.killpg(identity["pgid"], signal.SIGKILL)
                        receipt["orphan_cleanup_required"] = True
                except (OSError, ValueError, subprocess.SubprocessError) as cleanup_error:
                    receipt["cleanup_error"] = str(cleanup_error)
            receipt.update(status="stopped", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            receipt["elapsed_seconds"] = self.clock() - receipt["started_unix"]
            receipt["finished_unix"] = self.clock()
            receipt["borrowed_seconds"] = max(0, receipt["elapsed_seconds"] - base_available)
            receipt["repair_credit_used_seconds"] = max(
                0, min(repair_credit, elapsed + receipt["elapsed_seconds"] - prefix))
            if log_path.exists():
                receipt["log_sha256"] = sha256(log_path)
            self.save()
        return receipt

    def run(self):
        with (self.output / "cycle.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise CycleStopped("Another launcher owns this cycle") from error
            self.state = read_json(self.path)
            def stop(signum, frame):
                if signum == signal.SIGALRM:
                    raise CycleStopped("Global 24-hour deadline reached")
                raise CycleStopped(f"Cycle interrupted by signal {signum}")
            previous = {sig: signal.signal(sig, stop)
                        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
            prior_timer = signal.setitimer(signal.ITIMER_REAL, max(.001, self.remaining()))
            timer_started = time.monotonic()
            try:
                return self._run_locked()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                if prior_timer[0] > 0:
                    signal.setitimer(signal.ITIMER_REAL,
                                     max(.001, prior_timer[0] - (time.monotonic() - timer_started)),
                                     prior_timer[1])

    def _run_locked(self):
        if self.state["status"] not in ("prepared", "running"):
            raise CycleStopped("Stopped/failed/completed cycle cannot auto-restart")
        self.state["status"] = "running"
        self.save()
        try:
            assert_prior_processes_quiescent(self.state)
            self.preflight()
            for original in self.plan["jobs"]:
                job = self._expand(original)
                self.state["current_stage"] = job["id"]
                self.save()
                previous = self.state["stages"].get(job["id"])
                if previous:
                    if previous.get("kind") in (
                        "pilot_select", "pilot_factorized_select", "full_horizon",
                        "factorized_full_horizon", "freeze"
                    ):
                        if previous.get("status") != "completed":
                            raise CycleStopped("Incomplete decision cannot auto-restart")
                        verify_pins(previous["artifacts"])
                    else:
                        verify_receipt(previous)
                    continue
                if self.remaining() <= 0:
                    raise CycleStopped("Global 24-hour deadline reached")
                if job.get("existing_receipt"):
                    receipt = read_json(job["existing_receipt"])
                    verify_receipt(receipt)
                    receipt["artifacts"] = check_artifacts(job.get("artifacts", []))
                elif job.get("kind", "command") == "command":
                    receipt = self._compute(job)
                else:
                    receipt = self._decision(job)
                self.state["stages"][job["id"]] = receipt
                self.save()
            self.state["status"] = "completed"
        except BaseException as error:
            self.state.update(status="stopped", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if self.state["status"] in ("stopped", "completed"):
                self.state["finished_unix"] = self.clock()
            self.save()
        return self.state
