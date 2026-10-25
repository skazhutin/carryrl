"""Immutable, short common-LR trials from the same verified fresh t32 state."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from carryrl.training.token_comparison import (
    _pin,
    _read,
    _write,
    require_fresh_reward,
    verify_plan,
)

LEARNING_RATES = (0.00001, 0.00003)


def calibration_label(learning_rate):
    if type(learning_rate) not in (float, int) or learning_rate not in LEARNING_RATES:
        raise ValueError("calibration learning rate must be 0.00001 or 0.00003")
    return f"lr{learning_rate:.5f}"


def prepare_calibration(plan_path, learning_rate):
    """Prepare or verify one trial; never change primary configs or execute training."""
    label = calibration_label(learning_rate)
    plan_path = Path(plan_path).resolve()
    plan = verify_plan(plan_path)
    arms = [arm for arm in plan["arms"] if arm["spatial_tokens"] == 32]
    if (not plan.get("modelinit_verified") or len(arms) != 1
            or not arms[0].get("initialization_verified")):
        raise ValueError("materialize and verify fresh initial checkpoints before calibration")
    arm = arms[0]
    reward_pins = require_fresh_reward(plan_path, plan)
    if not reward_pins or reward_pins != plan.get("fresh_reward_pins"):
        raise ValueError("calibration requires the materialized, pinned fresh reward")
    initial = Path(arm["initial_checkpoint"])
    initial_manifest = _read(initial / "manifest.json")
    if any(initial_manifest.get(key) != 0 for key in
           ("rollout", "decisions", "weights_revision")):
        raise ValueError("calibration requires a zero-update fresh initial checkpoint")
    raw = copy.deepcopy(_read(arm["config"]["path"]))
    raw.update(
        learning_rate=learning_rate,
        total_rollouts=2,
        episodes_per_task=1,
        n_epochs=2,
        task_episode_multipliers={},
        evaluate_initial=True,
        eval_every_rollouts=2,
        eval_all_levels=False,
        eval_episodes_per_task=2,
        evaluation_canonical=True,
        evaluation_pixel_art=True,
        eval_stochastic=False,
        checkpoint_every_rollouts=2,
        reset_optimizer_on_warm_start=False,
    )
    if len(raw["tasks"]) != 16 or raw["agent"]["spatial_tokens"] != 32:
        raise ValueError("calibration requires the shared 32-token, all-16-task config")
    directory = plan_path / "calibration" / label
    config_path = directory / "config.json"
    encoded = json.dumps(raw, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output = plan_path.with_name(f"{plan_path.name}_calibration_{label}")
    artifact = {
        "format": "carryrl.token_calibration.v1",
        "spatial_tokens": 32,
        "learning_rate": learning_rate,
        "seed": raw["seed"],
        "initial_checkpoint": str(initial),
        "initial_manifest": initial_manifest,
        "initial_optimizer_state_entries": 0,
        "initialization_verified": True,
        "training_output": str(output),
        "fresh_reward_pins": reward_pins,
        "policy_supervision": "PPO only; no teacher actions or behavioral cloning",
        "selection_rule": "finite PPO updates and KL stability; no per-arm tuning",
    }
    if config_path.exists():
        if config_path.read_text() != encoded:
            raise ValueError("immutable calibration config changed")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        with config_path.open("x") as stream:
            stream.write(encoded)
    artifact["config"] = _pin(config_path)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if _read(manifest_path) != artifact:
            raise ValueError("immutable calibration lineage changed")
    else:
        with manifest_path.open("x") as stream:
            stream.write(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return artifact


def _finite_metrics(value):
    if isinstance(value, dict):
        return all(_finite_metrics(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_metrics(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _calibration_evidence(artifact):
    """Validate the actual sequence/supervisor schemas, never an asserted PASS file."""
    from carryrl.training.sequence_config import SequenceRunConfig

    run = Path(artifact["training_output"])
    paths = [run / name for name in (
        "summary.json", "metadata.json", "config.json", "training.jsonl",
        "evaluation.json", "evaluation_pixel.json",
    )]
    paths.append(run.with_name(run.name + "_launch") / "process.json")
    initial_paths = [run / name / "rollout_0000.json" for name in
                     ("evaluations", "evaluations_pixel")]
    paths.extend(initial_paths)
    pins = [_pin(path) for path in paths]
    summary, metadata, config = (_read(path) for path in paths[:3])
    rows = [json.loads(line) for line in paths[3].read_text().splitlines() if line.strip()]
    canonical, pixel, process = (_read(path) for path in paths[4:7])
    initial = artifact["initial_manifest"]
    source = metadata.get("warm_start_parent", {})
    if (process.get("status") != "exited" or process.get("actual_exit_code") != 0
            or process.get("error") or process.get("stop_reason")):
        raise ValueError("calibration supervisor did not exit successfully")
    if (summary.get("status") != "completed" or summary.get("rollouts") != 2
            or type(summary.get("optimizer_steps_this_process")) is not int
            or summary["optimizer_steps_this_process"] <= 0
            or summary.get("initial_weights_revision") != 0
            or summary.get("reward_sha256") != initial["reward_sha256"]):
        raise ValueError("calibration needs two completed rollouts and actual fresh-reward updates")
    if (Path(source.get("checkpoint", "")).resolve() != Path(artifact["initial_checkpoint"])
            or any(source.get(key) != initial[key] for key in
                   ("agent_sha256", "reward_sha256"))
            or source.get("checkpoint_sha256") != initial["sha256"]
            or source.get("source_counters") !=
            {"rollout": 0, "decisions": 0, "weights_revision": 0}
            or source.get("source_optimizer", {}).get("state_entries") != 0
            or source.get("new_optimizer_hyperparameters", {}).get("learning_rate")
            != artifact["learning_rate"]):
        raise ValueError("calibration warm-start lineage or fresh optimizer differs")
    expected = SequenceRunConfig.load(artifact["config"]["path"]).to_dict()
    if SequenceRunConfig.from_dict(config).to_dict() != expected:
        raise ValueError("actual calibration config differs from its immutable config")
    if (canonical.get("rollout") != 2 or pixel.get("rollout") != 2
            or summary.get("evaluation_rollout") != 2
            or summary.get("evaluation_pixel_rollout") != 2):
        raise ValueError("calibration final canonical and pixel validation are incomplete")
    for path in initial_paths:
        baseline = _read(path)
        if (baseline.get("rollout") != 0 or baseline.get("weights_revision") != 0
                or baseline.get("agent_sha256") != initial["agent_sha256"]
                or baseline.get("reward_sha256") != initial["reward_sha256"]):
            raise ValueError("calibration initial validation differs from the fresh policy")
    if len(rows) != 2 or [row.get("rollout") for row in rows] != [1, 2]:
        raise ValueError("calibration training journal must contain exactly two rollouts")
    epochs = []
    for row in rows:
        if (row.get("complete") is not True or type(row.get("optimizer_steps")) is not int
                or row["optimizer_steps"] <= 0
                or not isinstance(row.get("epochs"), list) or not row["epochs"]
                or not _finite_metrics(row)
                or row.get("reward_snapshot_sha256") != initial["reward_sha256"]):
            raise ValueError("calibration PPO journal has missing, nonfinite or invalid updates")
        for epoch in row["epochs"]:
            for name in ("policy_loss", "value_loss", "entropy", "approx_kl",
                         "clip_fraction", "gradient_norm"):
                if type(epoch.get(name)) not in (int, float) or not math.isfinite(epoch[name]):
                    raise ValueError("calibration PPO epoch metrics are missing or nonfinite")
            epochs.append(epoch)
        applied = sum(epoch.get("optimizer_steps", 0) for epoch in row["epochs"])
        if applied != row["optimizer_steps"]:
            raise ValueError("calibration optimizer update counters disagree")
    updates = sum(row["optimizer_steps"] for row in rows)
    if (updates != summary["optimizer_steps_this_process"]
            or summary.get("weights_revision") != updates or not _finite_metrics(summary)):
        raise ValueError("calibration summary and actual PPO updates disagree")
    maximum_kl = max(epoch["approx_kl"] for epoch in epochs)
    early_stop = any(row.get("kl_early_stop") or epoch.get("kl_early_stop")
                     for row in rows for epoch in row["epochs"])
    return {
        "learning_rate": artifact["learning_rate"],
        "run": str(run),
        "optimizer_steps": updates,
        "maximum_epoch_approx_kl": maximum_kl,
        "target_kl": expected["target_kl"],
        "kl_early_stop": early_stop,
        "initial_agent_sha256": initial["agent_sha256"],
        "reward_sha256": initial["reward_sha256"],
    }, pins


def finalize_common_calibration(plan_path):
    """Enable the original shared LR only after both real trial pipelines complete."""
    plan_path = Path(plan_path).resolve()
    artifacts = [prepare_calibration(plan_path, lr) for lr in LEARNING_RATES]
    diagnostics, evidence_pins = [], []
    for artifact in artifacts:
        observed, pins = _calibration_evidence(artifact)
        diagnostics.append(observed)
        lineage_path = Path(artifact["config"]["path"]).with_name("manifest.json")
        evidence_pins.extend([artifact["config"], _pin(lineage_path), *pins])
    conservative = diagnostics[0]
    if (conservative["kl_early_stop"]
            or conservative["maximum_epoch_approx_kl"] > conservative["target_kl"]):
        raise ValueError("original common LR1e-5 did not pass the predeclared KL stability gate")
    plan = verify_plan(plan_path)
    if any(_read(arm["config"]["path"])["learning_rate"] != LEARNING_RATES[0]
           for arm in plan["arms"]):
        raise ValueError("primary configs must already use the original shared LR1e-5")
    selection = {
        "format": "carryrl.token_calibration_selection.v1",
        "approved": True,
        "selected_learning_rate": LEARNING_RATES[0],
        "rule": "retain original common LR1e-5 after finite real updates and KL stability; "
                "LR3e-5 is diagnostic only; no wins or per-token tuning",
        "interpretation": "two-rollout engineering calibration; no proof of learning",
        "trials": diagnostics,
        "evidence_pins": evidence_pins,
    }
    path = plan_path / "calibration" / "selection.json"
    if path.exists():
        if _read(path) != selection:
            raise ValueError("immutable calibration selection changed")
    else:
        _write(path, selection)
    plan.update(
        calibration_selection_pin=_pin(path),
        calibration_evidence_pins=evidence_pins,
        common_lr_calibration="verified shared LR1e-5; both finite-update trials completed; "
                              "original LR passed predeclared KL stability gate",
        ready_to_train=True,
    )
    _write(plan_path / "manifest.json", plan)
    return selection
