"""Fresh, bounded RGB reward diagnostics, separate from policy evaluation or fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv
from carryrl.envs.atomic_suite import TASKS
from carryrl.envs.pixel_art import style_pixel_art
from carryrl.rewards.composition import (
    MAX_INPUT_UNITS,
    OUTCOME_POTENTIAL_VERSION,
    RewardComposer,
    RewardCompositionConfig,
    classify_learned_event,
)
from carryrl.rewards.pretrain import load_screen_reward_checkpoint
from carryrl.rewards.visual import HUD_HEIGHT, verify_visible_outcome
from carryrl.training.imitation import scripted_teacher
from carryrl.training.token_comparison import verify_plan

FORMAT = "carryrl.fresh_reward_validation.v1"
STYLES = ("canonical", "frost", "circuit")
BANK = 20261008
THRESHOLD = 0.9


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _status(observed):
    return "success" if observed["success"] else "failure" if observed["failure"] else "play"


def _close(actual, expected):
    return type(actual) in (int, float) and math.isclose(actual, expected, abs_tol=1e-7)


class _Checks:
    def __init__(self):
        self.counts = {"hud": 0, "progress": 0}
        self.failures = {"hud": 0, "progress": 0}
        self.examples = []

    def check(self, category, passed, reason, **context):
        self.counts[category] += 1
        if not passed:
            self.failures[category] += 1
            if len(self.examples) < 32:
                self.examples.append({"category": category, "reason": reason, **context})
        return bool(passed)


def _random_action(rng):
    keys = ((), ("left",), ("right",), ("up",), ("down",), ("space",),
            ("e",), ("enter",), ("tab",))
    return PhysicalAction(keys=keys[int(rng.integers(len(keys)))],
                          mouse_buttons=("left",) if rng.random() < 0.2 else (),
                          dx=int(rng.choice((-16, 0, 16))), dy=int(rng.choice((-16, 0, 16))))


def _composer(config, initial, *, gamma, horizon):
    return RewardComposer(config, gamma=gamma, horizon=horizon,
                          score_confidence_threshold=THRESHOLD, initial_prediction=initial)


def _frame_checks(reward, frame, expected, checks, context):
    observed = verify_visible_outcome(frame)
    prediction = reward.predict(frame)
    checks.check("hud", observed == expected and observed["known"],
                 "independent RGB verifier mismatch", **context)
    checks.check("hud", prediction.get("known") is True,
                 "learned readout unknown", **context)
    checks.check("hud", _close(prediction.get("score"), expected["score"]),
                 "learned score differs from exact visible score", **context)
    confidence = prediction.get("score_confidence", -1)
    checks.check("hud", type(confidence) in (int, float) and THRESHOLD <= confidence <= 1,
                 "score confidence below 0.9", **context)
    checks.check("hud", classify_learned_event(prediction, THRESHOLD) == _status(expected),
                 "learned event class mismatch at 0.9", **context)
    return prediction


def _composition_checks(step, previous, observed, action, boundary, config, gamma,
                        checks, context):
    parameters = config.parameters
    potential = 0.0 if boundary else observed["score"] / 5
    expected_progress = parameters.progress_weight * (gamma * potential - previous["score"] / 5)
    units = (len(action.keys) + len(action.mouse_buttons) + int(action.dx != 0)
             + int(action.dy != 0) + int(action.wheel != 0))
    expected_outcome = (parameters.success_reward if observed["success"] else
                        -parameters.failure_penalty if observed["failure"] else 0.0)
    checks.check("progress", step.valid_for_learning and step.score_trusted,
                 "supported transition rejected or score untrusted", **context)
    checks.check("progress", _close(step.progress, expected_progress),
                 "progress differs from visible potential (including zero terminal potential)",
                 **context)
    checks.check("progress", _close(step.outcome, expected_outcome if boundary else 0.0),
                 "outcome must occur only at actual episode boundary", **context)
    checks.check("progress", _close(step.time_cost, -parameters.time_cost)
                 and _close(step.input_cost, -parameters.input_cost * units / MAX_INPUT_UNITS),
                 "executed command cost mismatch", **context)
    checks.check("progress", _close(step.total, step.outcome + step.progress
                                     + step.time_cost + step.input_cost),
                 "reward component sum mismatch", **context)


def _repeat_checks(prediction, config, gamma, checks, context):
    """Stationary recognized screens earn no positive progress or repeated outcome."""
    composer = _composer(config, prediction, gamma=gamma, horizon=4)
    discounted_progress = 0.0
    for index in range(4):
        step = composer.step(prediction, PhysicalAction(), terminated=False, truncated=index == 3)
        discounted_progress += gamma**index * step.progress
        checks.check("progress", step.progress <= 1e-7 and step.outcome == 0.0,
                     "stationary screen creates positive progress or repeated outcome", **context)
        checks.check("progress", step.time_cost == -config.parameters.time_cost,
                     "stationary command escapes time cost", **context)
    expected = -config.parameters.progress_weight * prediction["score"] / 5
    checks.check("progress", _close(discounted_progress, expected),
                 "stationary potential does not telescope to terminal zero", **context)


def _episode(reward, task, strategy, level, seed, config, gamma, horizon, checks,
             env_factory, teacher_factory, *, split="validation", stage=None,
             bank=BANK, styles=STYLES):
    options = {"split": split, "image_size": 96, "max_steps": horizon,
               "difficulty": 1.0, "level_bank_seed": bank}
    if stage is not None:
        options["curriculum_stage"] = stage
    env = env_factory(task, **options)
    context = {"task": task, "strategy": strategy, "level_index": level,
               "split": split, "stage": stage}
    episode = {**context, "seed": seed, "steps": 0, "scores": [], "intermediate_scores": [],
               "success": False, "failure": False, "timeout": False, "completed": False}
    try:
        frame, info = env.reset(seed=seed, options={"level_index": level})
        episode["level_id"] = info["level_id"]
        checks.check("progress", info.get("level_bank_seed") == bank,
                     "episode bank differs from fresh bank", **context)
        teacher = teacher_factory(task) if strategy == "diagnostic_teacher" else None
        rng = np.random.default_rng(seed)
        composers, previous, repeat_checked = {}, None, False
        terminated = truncated = False
        action = PhysicalAction()
        for tick in range(horizon + 1):
            observed = verify_visible_outcome(frame)
            if not observed["known"]:
                checks.check("hud", False, "canonical episode HUD unknown", tick=tick, **context)
                break
            episode["scores"].append(observed["score"])
            if not observed["success"] and not observed["failure"] and 0 < observed["score"] < 5:
                episode["intermediate_scores"].append(observed["score"])
            for style in styles:
                styled = frame if style == "canonical" else style_pixel_art(
                    frame, style, seed, task=task)
                frame_context = {**context, "style": style, "tick": tick}
                checks.check("hud", np.array_equal(styled[:HUD_HEIGHT], frame[:HUD_HEIGHT]),
                             "pixel presentation changed original HUD", **frame_context)
                prediction = _frame_checks(reward, styled, observed, checks, frame_context)
                if tick == 0:
                    composers[style] = _composer(config, prediction, gamma=gamma, horizon=horizon)
                else:
                    step = composers[style].step(prediction, action, terminated=terminated,
                                                 truncated=truncated)
                    _composition_checks(step, previous, observed, action, terminated or truncated,
                                        config, gamma, checks, frame_context)
                if not repeat_checked and observed["score"] > 0 and _status(observed) == "play":
                    _repeat_checks(prediction, config, gamma, checks, frame_context)
            if observed["score"] > 0 and _status(observed) == "play":
                repeat_checked = True
            if terminated or truncated:
                episode.update(success=observed["success"], failure=observed["failure"],
                               timeout=truncated, completed=True)
                break
            previous = observed
            action = (teacher.act(frame.copy()) if teacher else PhysicalAction()
                      if strategy == "neutral" else _random_action(rng))
            frame, _, terminated, truncated, _ = env.step(action)
            episode["steps"] += 1
        if not episode["completed"]:
            checks.check("progress", False, "episode has no observed boundary", **context)
    except (ValueError, RuntimeError, KeyError, TypeError) as error:
        checks.check("progress", False, f"diagnostic episode error: {error}", **context)
        episode["error"] = str(error)
    finally:
        env.close()
    episode["scores"] = sorted(set(episode["scores"]))
    episode["intermediate_scores"] = sorted(set(episode["intermediate_scores"]))
    return episode


def _validate_reward(reward, *, tasks, config, gamma=0.99, horizon=80, levels=4,
                     seed=20261029, env_factory=AtomicEnv, teacher_factory=scripted_teacher):
    """Injectable diagnostic core; only the public bundle gate certifies all tasks."""
    checks, coverage = _Checks(), {}
    config.validate_context(gamma=gamma, horizon=horizon, tasks=tasks,
                            score_confidence_threshold=THRESHOLD)
    for task_index, task in enumerate(tasks):
        episodes = []
        for strategy_index, strategy in enumerate(("diagnostic_teacher", "neutral", "random")):
            for level in range(levels):
                episode_seed = seed + task_index * 10000 + strategy_index * 1000 + level
                episode = _episode(reward, task, strategy, level, episode_seed, config, gamma,
                                   horizon, checks, env_factory, teacher_factory)
                episodes.append(episode)
                # Search for success/progress only with the diagnostic controller;
                # neutral/random controls each retain a real complete nonwin.
                if strategy == "diagnostic_teacher":
                    if episode["success"] and episode["intermediate_scores"]:
                        break
                elif episode["completed"] and not episode["success"]:
                    break
        successes = sum(episode["success"] for episode in episodes)
        nonwins = sum(episode["completed"] and not episode["success"] for episode in episodes)
        intermediate = sorted({score for episode in episodes for score in episode[
            "intermediate_scores"]})
        required = {"observed_success": successes > 0, "observed_nonwin": nonwins > 0,
                    "observed_intermediate": bool(intermediate),
                    "neutral_nonwin": any(episode["strategy"] == "neutral"
                                           and episode["completed"] and not episode["success"]
                                           for episode in episodes),
                    "random_nonwin": any(episode["strategy"] == "random"
                                          and episode["completed"] and not episode["success"]
                                          for episode in episodes)}
        for name, passed in required.items():
            checks.check("progress", passed, f"coverage missing: {name}", task=task)
        coverage[task] = {"passed": all(required.values()), "requirements": required,
                          "successes": successes, "nonwins": nonwins,
                          "intermediate_scores": intermediate, "episodes": episodes}
    _unknown_checks(reward, config, gamma, checks, seed)
    return _check_details(checks, coverage)


def _unknown_checks(reward, config, gamma, checks, seed):
    for name, frame in (("blank", np.zeros((96, 96, 3), dtype=np.uint8)),
                        ("noise", np.random.default_rng(seed).integers(
                            0, 256, (96, 96, 3), dtype=np.uint8))):
        prediction = reward.predict(frame)
        checks.check("hud", not verify_visible_outcome(frame)["known"]
                     and prediction.get("known") is False, "unknown RGB was accepted", case=name)
        for terminated in (True, False):
            composer = _composer(config, prediction, gamma=gamma, horizon=1)
            step = composer.step(prediction, PhysicalAction(), terminated=terminated,
                                 truncated=not terminated)
            checks.check("progress", not step.valid_for_learning and step.outcome == 0.0
                         and step.progress == 0.0
                         and step.time_cost == -config.parameters.time_cost,
                         "unknown terminal was valid or escaped costs", case=name,
                         terminated=terminated)


def _check_details(checks, coverage):
    return {"hud_passed": checks.counts["hud"] > 0 and checks.failures["hud"] == 0,
            "progress_passed": checks.counts["progress"] > 0 and checks.failures["progress"] == 0,
            "coverage": coverage, "checks": checks.counts, "failure_counts": checks.failures,
            "failure_examples": checks.examples}


def validate_fresh_reward(plan, *, levels=4):
    """Validate only the pinned bundle's fresh reward; never fit or load a policy."""
    if type(levels) is not int or not 1 <= levels <= 4:
        raise ValueError("levels must be an integer in [1,4]")
    plan = Path(plan).resolve(strict=True)
    if not plan.is_dir():
        raise ValueError("--plan must name the bundle directory, not its manifest file")
    destination = plan / "reward" / "validation_report.json"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"validation report already exists: {destination}")
    manifest = verify_plan(plan)
    recipe_path = plan / "reward_recipe.json"
    recipe = _read(recipe_path)
    config = _read(manifest["reward_config_pin"]["path"])
    arm = _read(manifest["arms"][0]["config"]["path"])
    if (recipe.get("format") != "carryrl.fresh_reward_recipe.v1"
            or recipe.get("test_opened") is not False
            or recipe.get("config") != manifest["reward_config_pin"]
            or tuple(config["tasks"]) != TASKS or arm["tasks"] != config["tasks"]
            or config.get("level_bank_seed") != BANK or arm.get("level_bank_seed") != BANK):
        raise ValueError("fresh reward recipe/task/bank provenance disagrees with plan")
    checkpoint, report_path = (plan / "reward" / name for name in (
        "reward_checkpoint.pt", "reward_report.json"))
    for path in (checkpoint, report_path):
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(plan):
            raise ValueError("fresh reward artifacts must remain inside their bundle")
    before = {"recipe_sha256": _sha(recipe_path), "checkpoint_sha256": _sha(checkpoint),
              "reward_report_sha256": _sha(report_path)}
    torch_threads = config.get("torch_threads", 4)
    if type(torch_threads) is not int or torch_threads < 1:
        raise ValueError("torch_threads must be a positive integer")
    torch.set_num_threads(torch_threads)
    reward, embedded_report = load_screen_reward_checkpoint(checkpoint, require_pass=True)
    # Torch preserves tuples; the sibling JSON report necessarily stores them as lists.
    embedded_report = json.loads(json.dumps(embedded_report, allow_nan=False))
    report = _read(report_path)
    if (report != embedded_report or report.get("initialization") != "random"
            or report.get("imported_checkpoints") != [] or report.get("old_corpora") != []
            or report.get("gate", {}).get("passed") is not True
            or report.get("test_levels_used") is not False
            or report.get("policy_demonstrations") is not False
            or report.get("level_bank_seed") != BANK or report.get("seed") != config["seed"]
            or report.get("tasks") != config["tasks"]):
        raise ValueError("fresh reward training report/checkpoint provenance mismatch")
    composition = RewardCompositionConfig.from_dict(arm["reward_composition"])
    if composition.version != OUTCOME_POTENTIAL_VERSION:
        raise ValueError("fresh validator requires outcome/potential reward composition")
    horizon = arm["max_episode_steps"]
    if type(horizon) is not int or not 1 <= horizon <= 80:
        raise ValueError("fresh validation requires a bounded horizon of at most 80")
    details = _validate_reward(reward, tasks=TASKS, config=composition, gamma=arm["gamma"],
                               horizon=horizon, levels=levels)
    verify_plan(plan)
    after = {"recipe_sha256": _sha(recipe_path), "checkpoint_sha256": _sha(checkpoint),
             "reward_report_sha256": _sha(report_path)}
    if before != after:
        raise ValueError("reward artifacts changed during validation")
    result = {"format": FORMAT, "passed": details["hud_passed"] and details["progress_passed"],
              "tasks": list(TASKS), "level_bank_seed": BANK, "split": "validation",
              "initialization": "random", "imported_checkpoints": [], "old_corpora": [],
              **before, **details, "styles": list(STYLES), "test_levels_used": False,
              "policy_evaluated": False, "policy_updates": 0,
              "diagnostic_controller": "RGB scripted_teacher; fixture generator only",
              "scope": "bounded fresh diagnostic episodes, not learned-policy evaluation; "
              "frost/circuit change scenery while preserving original canonical HUD",
              "bounds": {"levels_per_strategy": levels, "max_steps": horizon,
                         "seed": 20261029, "score_confidence": THRESHOLD,
                         "event_confidence": THRESHOLD}}
    with destination.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True,
                        help="token comparison bundle directory")
    parser.add_argument("--levels", type=int, default=4, help="bounded level search, 1..4")
    args = parser.parse_args(argv)
    result = validate_fresh_reward(args.plan, levels=args.levels)
    print(json.dumps({"passed": result["passed"], "hud_passed": result["hud_passed"],
                      "progress_passed": result["progress_passed"],
                      "report": str(args.plan / "reward" / "validation_report.json")}))
    return 0 if result["passed"] else 1
