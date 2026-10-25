"""Preparation and selection for the separately registered three-branch text pilot."""

import json
import math
from dataclasses import replace
from pathlib import Path

from carryrl.envs.atomic_suite import TASKS
from carryrl.training.rl_cycle import STYLES, pilot_eligibility
from carryrl.training.run import atomic_json, file_sha256
from carryrl.training.sequence_config import SequenceRunConfig

BRANCHES = ("zero", "text", "text_penalty")
PILOT_ROLLOUTS = 80


def choose_factorized_full_horizon(rollout_seconds, evaluation_seconds, available_seconds):
    """Choose the fresh main horizon after pilots using measured evaluation costs."""
    if (len(rollout_seconds) < 2
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in [*rollout_seconds, evaluation_seconds, available_seconds])):
        raise ValueError("Measured timings and a positive available full-stage budget required")
    cycle = max(rollout_seconds) * 1.25
    overhead = 4 * evaluation_seconds * 1.25
    costs = {horizon: horizon * cycle + overhead for horizon in (128, 160, 192)}
    feasible = [horizon for horizon, cost in costs.items() if cost <= available_seconds]
    return {"version": "carryrl.factorized_full_horizon.v1",
            "full_rollouts": max(feasible) if feasible else None,
            "conservative_seconds_per_rollout": cycle,
            "evaluation_overhead_seconds": overhead,
            "required_seconds_by_horizon": costs,
            "available_training_seconds": available_seconds,
            "control_schedule_rollouts": 64}


def forecast_remaining_cycle(rollout_seconds, evaluation_seconds, remaining_seconds):
    """Require all three pilots, a genuine >=128 main run and final two hours to fit."""
    if (len(rollout_seconds) < 2
            or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                   for x in [*rollout_seconds, evaluation_seconds, remaining_seconds])):
        raise ValueError("Measured full-stage timings and a positive remaining budget required")
    cycle = max(rollout_seconds) * 1.25
    evaluation = evaluation_seconds * 1.25
    pilot = len(BRANCHES) * (PILOT_ROLLOUTS * cycle + 4 * evaluation)
    estimates = {horizon: pilot + horizon * cycle + 4 * evaluation + 7200
                 for horizon in (128, 160, 192)}
    feasible = [horizon for horizon, seconds in estimates.items() if seconds <= remaining_seconds]
    return {"version": "carryrl.factorized_remaining_forecast.v1",
            "full_rollouts": max(feasible) if feasible else None,
            "all_required_stages_fit": bool(feasible),
            "pilot_seconds": pilot, "total_seconds_by_main_horizon": estimates,
            "remaining_seconds": remaining_seconds, "final_reserve_seconds": 7200,
            "conservative_cycle_seconds": cycle, "conservative_evaluation_seconds": evaluation,
            "rule": "No shortened pilot or main run; budget begins at original cycle start"}


def choose_factorized_pilot(cp0, reports):
    """Prefer less shaping unless both styles gain eight wins without lost breadth."""
    if set(reports) != set(BRANCHES):
        raise ValueError("All three registered pilot branches are required")
    gates = {name: pilot_eligibility(report, cp0, expected_rollouts=PILOT_ROLLOUTS)
             for name, report in reports.items()}
    selected = None
    for name in BRANCHES[:2]:
        if not gates[name]["eligible"]:
            continue
        if selected is None:
            selected = name
            continue
        advantage = all(
            reports[name]["styles"][style]["successes"]
            >= reports[selected]["styles"][style]["successes"] + 8
            for style in STYLES)
        breadth = (len(gates[name]["train_games"]) >= len(gates[selected]["train_games"])
                   and all(len(gates[name]["styles"][style]["improved_games"])
                           >= len(gates[selected]["styles"][style]["improved_games"])
                           for style in STYLES))
        if advantage and breadth:
            selected = name
    # The penalty specifically needs evidence against the otherwise identical text recipe.
    if gates["text_penalty"]["eligible"]:
        penalty_benefit = all(
            reports["text_penalty"]["styles"][style]["successes"]
            >= reports["text"]["styles"][style]["successes"] + 8
            for style in STYLES)
        breadth = (len(gates["text_penalty"]["train_games"])
                   >= len(gates["text"]["train_games"]))
        if selected is not None:
            penalty_benefit &= all(
                reports["text_penalty"]["styles"][style]["successes"]
                >= reports[selected]["styles"][style]["successes"] + 8
                for style in STYLES)
            breadth &= len(gates["text_penalty"]["train_games"]) >= len(
                gates[selected]["train_games"])
        if penalty_benefit and breadth:
            selected = "text_penalty"
    return {"version": "carryrl.factorized_pilot_selection.v1", "selected_branch": selected,
            "branches": gates, "rollouts": PILOT_ROLLOUTS,
            "reason": "outcome and breadth gates" if selected else "full training prohibited"}


def prepare_factorized_profiles(plan, *, text_reward, visual_validation):
    """Require admitted frozen resources before writing executable profiles; never launch."""
    from carryrl.rewards.composition import TEXT_CONTROL_V2_VERSION
    from carryrl.training.rl_improvements import inspect_text_reward_artifact
    from carryrl.training.task_curriculum import validate_task_curriculum_admission

    plan, text = Path(plan).resolve(), Path(text_reward).resolve()
    base = SequenceRunConfig.load(plan / "policy_base.json")
    visual = plan / "reward_visual/reward_checkpoint.pt"
    gate = json.loads(Path(visual_validation).read_text())
    if (not gate.get("passed") or not gate.get("coverage_passed")
            or gate.get("checkpoint_sha256") != file_sha256(visual)):
        raise ValueError("Visual reward has no matching successful current gate")
    instructions = Path(__file__).resolve().parents[3] / "configs/atomic_instructions.json"
    controls = instructions.with_name("atomic_controls.json")
    text_pin = file_sha256(text / "manifest.json")
    text_gate = inspect_text_reward_artifact(text, text_pin, instructions_path=instructions,
                                            controls_path=controls)
    from carryrl.rewards.control_text_factorized import FACTORIZED_MODEL_VERSION

    if text_gate.get("version") != FACTORIZED_MODEL_VERSION:
        raise ValueError("New reward composition requires the admitted factorized model")
    admission = plan / "admission.json"
    admission_pin = file_sha256(admission)
    validate_task_curriculum_admission(
        admission, level_bank_seed=base.level_bank_seed, image_size=base.env_image_size,
        max_steps=base.max_episode_steps, expected_sha256=admission_pin)
    settings = base.to_dict()
    settings.update(
        seed=3176, tasks=list(TASKS), total_rollouts=PILOT_ROLLOUTS, episodes_per_task=8,
        task_episode_multipliers={}, control_schedule_rollouts=64,
        curriculum_difficulty=None, curriculum_warmup_rollouts=0, evaluation_difficulty=1.,
        ppo_normalization="rollout", ppo_task_balance=True, ppo_minibatch_episodes=32,
        ppo_microbatch_episodes=1, training_backend="mlx", device="cpu",
        rollout_memory_budget_mib=2048, rollout_feature_storage="mmap",
        rollout_disk_budget_mib=4096, minimum_free_disk_mib=8192,
        motor_initialization="balanced", gradient_probe_every=8,
        advantage_std_floor=.1, reward_model_mode="screen_v2",
        reward_checkpoint=str(visual), text_reward_checkpoint=str(text),
        text_reward_sha256=text_pin, reset_optimizer_on_warm_start=False,
        episode_instructions=json.loads(instructions.read_text()), screen_text_language=None,
        task_curriculum={"admission_path": str(admission), "admission_sha256": admission_pin},
        eval_every_rollouts=32, checkpoint_every_rollouts=16, checkpoint_keep_raw=2,
        evaluate_initial=True, eval_all_levels=True, evaluation_canonical=True,
        evaluation_pixel_art=True, max_wall_seconds=14400)
    settings["agent"].update(motor_version="motor_v2", critic_initialization="zero",
                             text_encoder="multilingual_e5_small", instruction_max_tokens=512)
    output = plan / "configs_factorized"
    output.mkdir(exist_ok=False)
    for branch in BRANCHES:
        settings["reward_composition"] = {"version": TEXT_CONTROL_V2_VERSION, "parameters": {
            "success_reward": 1., "progress_weight": .2, "failure_penalty": 0.,
            "time_cost": 0., "input_cost": 0., "goal_choice_utility": "completion_only",
            "control_schedule": "zero" if branch == "zero" else "temporary",
            "unsupported_alpha": .01 if branch == "text_penalty" else 0.,
        }}
        config = SequenceRunConfig.from_dict(settings)
        atomic_json(output / f"pilot_{branch}.json", config.to_dict())
    probe = replace(config, seed=3175, total_rollouts=2, evaluate_initial=False,
                    eval_all_levels=False, eval_episodes_per_task=1, eval_every_rollouts=1000,
                    checkpoint_every_rollouts=2, gradient_probe_every=1, max_wall_seconds=3000)
    atomic_json(output / "probe_curriculum.json", probe.to_dict())
    atomic_json(output / "probe_full.json", replace(
        probe, seed=3174, task_curriculum=None, curriculum_difficulty=1.).to_dict())
    return {"version": "carryrl.factorized_cycle_profiles.v1", "text_reward": text_gate,
            "branches": list(BRANCHES), "rollouts": PILOT_ROLLOUTS, "training_started": False}


def prepare_factorized_full_config(plan, output):
    plan = Path(plan)
    decisions = json.loads((plan / "cycle_state.json").read_text())["decisions"]
    branch, horizon = decisions.get("selected_branch"), decisions.get("full_rollouts")
    if branch not in BRANCHES or horizon not in (128, 160, 192):
        raise ValueError("Full training requires an admitted recipe and feasible horizon")
    config = SequenceRunConfig.load(plan / "configs_factorized" / f"pilot_{branch}.json")
    config = replace(config, seed=3177, total_rollouts=horizon, eval_every_rollouts=64,
                     checkpoint_every_rollouts=32, control_schedule_rollouts=64,
                     max_wall_seconds=42000)
    atomic_json(output, config.to_dict())
    return {"selected_branch": branch, "total_rollouts": horizon, "seed": 3177,
            "control_schedule_rollouts": 64, "training_started": False}
