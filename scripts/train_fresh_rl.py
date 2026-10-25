"""Preflight or explicitly execute one pinned, fresh all-task PPO experiment."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import signal
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

FORMAT = "carryrl.fresh_rl.v1"
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def utc():
    return datetime.now(UTC).isoformat()


def source_files(root):
    return {str(path.relative_to(root)) for folder in ("src", "scripts")
            for path in (root / folder).rglob("*.py")} | {"pyproject.toml", "uv.lock"}


def verify_plan(plan, *, root=ROOT):
    """Read configuration and hashes only; never construct models or collect frames."""
    from carryrl.envs.atomic_suite import TASKS
    from carryrl.envs.presentation import THEME_SPLITS, PresentationConfig
    from carryrl.evaluation.fresh_reward import BANK
    from carryrl.rewards.composition import OUTCOME_POTENTIAL_VERSION
    from carryrl.rewards.pretrain import RewardGate, pretrain_screen_reward
    from carryrl.training.sequence_config import SequenceRunConfig

    plan, root = Path(plan).resolve(strict=True), Path(root).resolve(strict=True)
    manifest = read(plan / "manifest.json")
    if manifest.get("format") != FORMAT:
        raise ValueError("unsupported fresh RL manifest")
    pins = manifest.get("source_pins", {})
    if set(pins) != source_files(root):
        raise ValueError("source inventory differs from pinned src/scripts and project files")
    for name, digest in pins.items():
        path = root / name
        if not path.resolve().is_relative_to(root) or sha(path) != digest:
            raise ValueError(f"source hash mismatch: {name}")
    configs = manifest.get("config_pins", {})
    if set(configs) != {"policy_config.json", "reward_config.json"}:
        raise ValueError("both configuration hashes are required")
    for name, digest in configs.items():
        if sha(plan / name) != digest:
            raise ValueError(f"config hash mismatch: {name}")
    budget, reserve = manifest.get("budget_seconds"), manifest.get("reserve_seconds")
    if (type(budget) not in (int, float) or not math.isfinite(budget)
            or not 0 < budget <= 14400 or type(reserve) not in (int, float)
            or not math.isfinite(reserve) or not 1800 <= reserve < budget):
        raise ValueError("budget must be <=14400 seconds with at least 1800 seconds reserved")
    policy = SequenceRunConfig.from_dict(read(plan / "policy_config.json"))
    reward = read(plan / "reward_config.json")
    if tuple(policy.tasks) != TASKS or tuple(reward.get("tasks", ())) != TASKS:
        raise ValueError("policy and reward must include all 16 tasks in suite order")
    if (policy.reward_checkpoint is not None or policy.reset_optimizer_on_warm_start
            or policy.reward_model_mode != "screen_v2"):
        raise ValueError("fresh policy requires a new injected screen reward and optimizer")
    if policy.level_bank_seed != BANK or reward.get("level_bank_seed") != BANK:
        raise ValueError("policy/reward bank must match the fresh diagnostic bank")
    if (policy.reward_composition.version != OUTCOME_POTENTIAL_VERSION
            or policy.reward_confidence_threshold != 0.9):
        raise ValueError(
            "fresh diagnostics require outcome-potential composition and threshold 0.9")
    allowed = set(inspect.signature(pretrain_screen_reward).parameters) - {"output"}
    if set(reward) - allowed:
        raise ValueError("unexpected reward fitter arguments; checkpoint/data imports forbidden")
    presentation = PresentationConfig(**reward.get("presentation_config", {}))
    allowed_themes = set(THEME_SPLITS["train"] + THEME_SPLITS["validation"])
    if presentation.mode != "varied" or set(presentation.themes) != allowed_themes:
        raise ValueError(
            "reward fitting requires exactly train/validation HUD themes; test themes forbidden")
    RewardGate(**reward.get("gate", {}))
    for name in ("episodes_per_task", "validation_episodes_per_task", "epochs", "max_steps"):
        value = reward.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"reward {name} must be a positive integer")
    if reward.get("image_size") != policy.env_image_size:
        raise ValueError("reward and policy image_size must match")
    diagnostics = manifest.get("diagnostics", {})
    if (set(diagnostics) != {"levels", "horizon", "seed"}
            or type(diagnostics["levels"]) is not int or not 1 <= diagnostics["levels"] <= 4
            or diagnostics["horizon"] != policy.max_episode_steps
            or type(diagnostics["seed"]) is not int or diagnostics["seed"] < 0):
        raise ValueError("diagnostics require levels 1..4, policy horizon, and a nonnegative seed")
    return manifest, policy, reward


def fit_reward(output, config):
    from carryrl.rewards.pretrain import RewardGate, pretrain_screen_reward

    config = dict(config)
    if "gate" in config:
        config["gate"] = RewardGate(**config["gate"])
    return pretrain_screen_reward(output, **config)


def validate_reward(reward, policy, diagnostics):
    from carryrl.evaluation.fresh_reward import _validate_reward

    return _validate_reward(reward, tasks=policy.tasks, config=policy.reward_composition,
                            gamma=policy.gamma, **diagnostics)


def train_policy(policy, output, reward, stop_requested):
    from carryrl.training.sequence import train_sequence

    return train_sequence(policy, output, reward=reward, stop_requested=stop_requested)


def policy_output(plan, root=ROOT):
    return Path(root).resolve() / "runs" / (Path(plan).name + "_main")


def execute(plan, *, stop_requested, root=ROOT):
    plan, root = Path(plan).resolve(strict=True), Path(root).resolve(strict=True)
    run = policy_output(plan, root)
    launch = plan / "stages/main"
    for directory in (launch.parent, launch):
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise FileExistsError(f"fresh execution requires a real stage directory: {directory}")
    # The resource guard owns this directory and its process.json before launching us.
    # Only this runner's outputs must be absent; guard artifacts remain untouched.
    for path in (plan / "execution.json", plan / "reward", run,
                 launch / "launch.json", launch / "status.json"):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"fresh execution refuses existing output: {path}")
    # Exclusive creation also prevents two processes from claiming the same plan.
    with (plan / "execution.json").open("x") as handle:
        handle.write("{}\n")
    started = time.monotonic()
    receipt = {
        "format": FORMAT, "status": "running", "phase": "preflight",
        "started_utc": utc(), "command_pid": os.getpid(), "supervisor_pid": os.getpid(),
        "plan": str(plan), "policy_output": str(run), "manifest_sha256": None,
        "resume": None, "warm_start": None, "imported_checkpoints": [],
        "old_corpora": [], "policy_initialization": "random trainable parameters",
        "initial_counters": {"rollout": 0, "decisions": 0, "weights_revision": 0},
        "initial_checkpoint": None, "final_checkpoint": None,
        "interpretation": "engineering experiment; no unmeasured skill claims",
    }

    def save(**values):
        receipt.update(values, updated_utc=utc(), wall_seconds=time.monotonic() - started)
        publish(plan / "execution.json", receipt)
        publish(launch / "status.json", receipt)

    try:
        receipt["manifest_sha256"] = sha(plan / "manifest.json")
        manifest, policy, reward_config = verify_plan(plan, root=root)
        deadline = started + manifest["budget_seconds"] - manifest["reserve_seconds"]

        def stopping():
            return stop_requested() or time.monotonic() >= deadline

        save(budget_seconds=manifest["budget_seconds"],
             reserve_seconds=manifest["reserve_seconds"],
             execution_started_utc=receipt["started_utc"],
             stop_rule="finish current preparation/PPO cycle; reserve for saving/evaluation; "
             "external guard enforces hard deadline")
        registry = root / ".cache/active_learning_launch.json"
        if registry.exists():
            shutil.copy2(registry, plan / "previous_active_learning_launch.json")
        publish(launch / "launch.json", receipt)
        publish(registry, {"run": str(run), "launch": str(launch)})
        if stopping():
            save(status="stopped", stop_reason="stop requested or preparation budget exhausted")
            return receipt
        save(phase="reward_fitting")
        reward, report = fit_reward(plan / "reward", reward_config)
        if report.get("gate", {}).get("passed") is not True:
            raise RuntimeError("fresh reward fitting gate failed; policy not created")
        checkpoint = plan / "reward/reward_checkpoint.pt"
        reward_provenance = {
            "initialization": "random", "bootstrap_is_supervised": True,
            "reward_bootstrap_executed": True, "fitted_by": "fresh RL runner before PPO",
            "imported_checkpoints": [], "old_corpora": [],
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
            "state_sha256": report.get("state_sha256"),
            "report": str(plan / "reward/reward_report.json"),
        }
        publish(plan / "reward/provenance.json", reward_provenance)
        save(reward_provenance=reward_provenance)
        if stopping():
            save(status="stopped",
                 stop_reason="stop requested or reward preparation budget exhausted")
            return receipt
        save(phase="reward_validation")
        validation = validate_reward(reward, policy, manifest["diagnostics"])
        publish(plan / "reward/validation_report.json", validation)
        if (validation.get("hud_passed") is not True
                or validation.get("progress_passed") is not True
                or set(validation.get("coverage", {})) != set(policy.tasks)
                or not all(item.get("passed") is True
                           for item in validation["coverage"].values())):
            raise RuntimeError("fresh reward diagnostic gate failed; policy not created")
        if stopping():
            save(status="stopped", stop_reason="stop requested or validation budget exhausted")
            return receipt
        policy = replace(policy, max_wall_seconds=min(
            policy.max_wall_seconds, deadline - time.monotonic()))
        publish(plan / "effective_policy_config.json", policy.to_dict())
        save(phase="policy_training", policy_max_wall_seconds=policy.max_wall_seconds)
        summary = train_policy(policy, run, reward, stopping)
        publish(plan / "metrics_summary.json", summary)
        save(status=summary["status"], phase="finished", metrics_summary=summary)
    except BaseException as error:
        save(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
             error_type=type(error).__name__, error=str(error))
        raise
    finally:
        initial = run / "checkpoints/rollout_0000"
        latest = run / "latest.json"
        receipt["initial_checkpoint"] = str(initial) if initial.exists() else None
        if latest.exists():
            receipt["final_checkpoint"] = read(latest).get("checkpoint")
        receipt["final_checkpoint_exists"] = (
            Path(receipt["final_checkpoint"]).exists() if receipt["final_checkpoint"] else False)
        if run.is_dir() and "reward_provenance" in receipt:
            publish(run / "fresh_reward_provenance.json", receipt["reward_provenance"])
        save(finished_utc=utc(), stop_requested=stop_requested())
    return receipt


def main(argv=None):
    stopped = False

    def request_stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    previous = signal.signal(signal.SIGUSR1, request_stop)
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", type=Path, required=True)
        parser.add_argument("--execute", action="store_true")
        args = parser.parse_args(argv)
        if args.execute:
            result = execute(args.plan, stop_requested=lambda: stopped)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "completed" else 2
        manifest, policy, _ = verify_plan(args.plan)
        print(json.dumps({"training_started": False, "verified": True,
                          "format": manifest["format"], "tasks": list(policy.tasks)}))
        return 0
    finally:
        signal.signal(signal.SIGUSR1, previous)


if __name__ == "__main__":
    raise SystemExit(main())
