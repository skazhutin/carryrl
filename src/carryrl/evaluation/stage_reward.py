"""Pinned, RGB-only reward readiness for admitted task stages and full validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv
from carryrl.envs.atomic_suite import TASKS
from carryrl.envs.curriculum import STAGES
from carryrl.envs.pixel_art import PIXEL_STYLE_SPLITS
from carryrl.envs.presentation import THEME_SPLITS, PresentationConfig
from carryrl.evaluation.fresh_reward import (
    THRESHOLD,
    _check_details,
    _Checks,
    _episode,
    _read,
    _sha,
    _unknown_checks,
)
from carryrl.rewards.composition import OUTCOME_POTENTIAL_VERSION, RewardCompositionConfig
from carryrl.rewards.pretrain import RewardGate, load_screen_reward_checkpoint
from carryrl.training.imitation import scripted_teacher
from carryrl.training.task_curriculum import (
    STAGE_SPEC_SHA256,
    validate_task_curriculum_admission,
)

FORMAT = "carryrl.fresh_stage_reward_validation.v2"
BANK = 20261020


def fresh_stage_reward_config(*, seed=20261021, device="cpu"):
    """Separate fresh HUD fit; curriculum mechanics need only pass RGB diagnostics."""
    return {
        "tasks": list(TASKS), "episodes_per_task": 8,
        "validation_episodes_per_task": 2, "epochs": 40, "seed": seed,
        "device": device, "image_size": 96, "max_steps": 80, "level_bank_seed": BANK,
        "presentation_config": asdict(PresentationConfig(
            mode="varied", themes=THEME_SPLITS["train"] + THEME_SPLITS["validation"],
            pixel_art=False, seed=seed)),
        "gate": asdict(RewardGate()),
    }


class _DiagnosticController:
    """Admission-compatible RGB reachability controller; never a policy input."""

    def __init__(self, task):
        self.task = task
        self.controller = scripted_teacher(task)

    def act(self, frame):
        if self.task != "goal_choice":
            return self.controller.act(frame)
        from carryrl.envs.atomic import PLAYER
        from carryrl.envs.atomic_combat import SAFE, _center, _keys_towards

        player, target = _center(frame, PLAYER), _center(frame, SAFE)
        tolerance = 4 * frame.shape[0] / 96
        if player is None or target is None:
            return PhysicalAction()
        if abs(player[0] - target[0]) > tolerance:
            return PhysicalAction(keys=("d" if target[0] > player[0] else "a",))
        return PhysicalAction(keys=_keys_towards(player, target, tolerance))


def _validate_cells(reward, admission, *, config, gamma, horizon, levels, seed,
                    tasks=TASKS, env_factory=AtomicEnv,
                    teacher_factory=_DiagnosticController):
    """Injectable diagnostics. Public certification always requires all 16 tasks."""
    checks, coverage = _Checks(), {}
    config.validate_context(gamma=gamma, horizon=horizon, tasks=tasks,
                            score_confidence_threshold=THRESHOLD)
    for task_index, task in enumerate(tasks):
        for split, stage in [("train", stage) for stage in STAGES] + [("validation", None)]:
            styles = ("canonical",) + PIXEL_STYLE_SPLITS[split]
            indices = (admission["tasks"][task][stage]["allowed_level_indices"]
                       if split == "train" else list(range(16)))
            cell = f"{task}/{split}/{stage or 'full'}"
            episodes = []
            cell_number = len(coverage)
            for strategy_index, strategy in enumerate(("diagnostic_teacher", "neutral", "random")):
                for search_index, level in enumerate(indices[:levels]):
                    episode = _episode(
                        reward, task, strategy, level,
                        seed + task_index * 100000 + cell_number * 1000
                        + strategy_index * 100 + search_index,
                        config, gamma, horizon, checks, env_factory, teacher_factory,
                        split=split, stage=stage, bank=BANK, styles=styles)
                    episodes.append(episode)
                    if (strategy == "diagnostic_teacher" and episode["success"]
                            and (stage not in (None, "full") or episode["intermediate_scores"])):
                        break
                    if (strategy != "diagnostic_teacher" and episode["completed"]
                            and not episode["success"]):
                        break
            requirements = {
                "observed_success": any(row["success"] for row in episodes),
                "observed_nonwin": any(row["completed"] and not row["success"]
                                       for row in episodes),
                # Random controls must be observed, but a genuine random win is
                # valid evidence and must not make reward recognition fail.
                "random_trajectory": any(row["strategy"] == "random" and row["completed"]
                                         for row in episodes),
                "all_episodes_completed": bool(episodes) and all(
                    row["completed"] for row in episodes),
            }
            for name, passed in requirements.items():
                checks.check("progress", passed, f"coverage missing: {name}", cell=cell)
            coverage[cell] = {
                "task": task, "split": split, "stage": stage or "full", "styles": list(styles),
                "passed": all(requirements.values()), "requirements": requirements,
                "successes": sum(row["success"] for row in episodes),
                "nonwins": sum(row["completed"] and not row["success"] for row in episodes),
                "intermediate_scores": sorted({score for row in episodes
                                                for score in row["intermediate_scores"]}),
                "episodes": episodes,
            }
    # Single-hit easy tasks can move directly from 0 to terminal. Intermediate
    # nonterminal progress must still be observed for each task's full mechanics.
    for task in tasks:
        observed = any(row["task"] == task and row["stage"] == "full"
                       and row["intermediate_scores"] for row in coverage.values())
        checks.check("progress", observed, "full task has no intermediate visible progress",
                     task=task)
    checks.check("progress", bool(tasks) and len(coverage) == len(tasks) * 4,
                 "empty or incomplete task/stage coverage")
    _unknown_checks(reward, config, gamma, checks, seed)
    return _check_details(checks, coverage)


def validate_stage_reward(reward, *, admission_path, admission_sha256, config,
                          gamma=0.99, horizon=80, levels=4, seed=20261021,
                          level_bank_seed=BANK):
    """Return a strict readiness report; no fitting, policy loading, or TEST rendering."""
    if (type(levels) is not int or not 1 <= levels <= 4 or horizon != 80
            or type(seed) is not int or seed < 0 or level_bank_seed != BANK):
        raise ValueError("stage reward protocol requires bank20261020, horizon80, levels1..4")
    if (not isinstance(admission_sha256, str) or len(admission_sha256) != 64
            or config.version != OUTCOME_POTENTIAL_VERSION):
        raise ValueError("pinned admission SHA256 and outcome-potential composition required")
    admission_path = Path(admission_path).resolve(strict=True)
    admission = validate_task_curriculum_admission(
        admission_path, level_bank_seed=BANK, image_size=96, max_steps=horizon,
        expected_sha256=admission_sha256)
    details = _validate_cells(reward, admission, config=config, gamma=gamma,
                              horizon=horizon, levels=levels, seed=seed)
    if _sha(admission_path) != admission_sha256:
        raise ValueError("admission changed during reward validation")
    expected = {f"{task}/{split}/{stage}" for task in TASKS
                for split, stages in (("train", STAGES), ("validation", ("full",)))
                for stage in stages}
    coverage = details["coverage"]
    expected_cells = {f"{name}/{style}" for name in expected
                      for style in ("canonical",) + PIXEL_STYLE_SPLITS[name.split("/")[1]]}
    covered_cells = [f"{name}/{style}" for name, row in coverage.items() for style in row["styles"]
                     if row["passed"]]
    coverage_passed = (set(coverage) == expected and set(covered_cells) == expected_cells
                       and len(covered_cells) == len(expected_cells)
                       and all(row["passed"] and row.get("episodes")
                               and all(episode.get("completed") for episode in row["episodes"])
                               for row in coverage.values()))
    passed = details["hud_passed"] and details["progress_passed"] and coverage_passed
    return {
        "format": FORMAT, "protocol": FORMAT, "passed": passed, "gate": {"passed": passed},
        "level_bank_seed": BANK, "tasks": list(TASKS), "train_stages": list(STAGES),
        "validation_stage": "full", "admission_sha256": admission_sha256,
        "stage_spec_sha256": STAGE_SPEC_SHA256,
        "composition": config.to_dict(), "gamma": gamma,
        "coverage_passed": coverage_passed, "covered_cells": covered_cells,
        "required_cells": len(expected_cells), "covered_cell_count": len(covered_cells),
        "train_styles": ["canonical", *PIXEL_STYLE_SPLITS["train"]],
        "validation_styles": ["canonical", *PIXEL_STYLE_SPLITS["validation"]],
        "test_levels_used": False, "test_themes_used": False,
        "policy_evaluated": False, "policy_updates": 0,
        "diagnostic_controller": "RGB reachability fixture generator only; never policy input",
        "labels": "independent exact RGB HUD verifier; environment reward/info not labels",
        "scope": "sampled admitted TRAIN stages and full validation; no policy quality claim",
        "bounds": {"levels_per_strategy": levels, "max_steps": horizon, "seed": seed,
                   "score_confidence": THRESHOLD, "event_confidence": THRESHOLD},
        **details,
    }


def validate_stage_reward_checkpoint(checkpoint, *, admission_path, admission_sha256,
                                     policy_config, reward_config, output, levels=4,
                                     seed=20261021):
    """Validate fresh fit provenance and write one immutable checkpoint-bound report."""
    paths = {"checkpoint": Path(checkpoint).resolve(strict=True),
             "policy_config": Path(policy_config).resolve(strict=True),
             "reward_config": Path(reward_config).resolve(strict=True)}
    paths["reward_report"] = paths["checkpoint"].with_name("reward_report.json")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"validation report already exists: {output}")
    hashes = {f"{name}_sha256": _sha(path) for name, path in paths.items()}
    fit_config, policy = _read(paths["reward_config"]), _read(paths["policy_config"])
    if (fit_config.get("tasks") != list(TASKS) or policy.get("tasks") != list(TASKS)
            or fit_config.get("level_bank_seed") != BANK or policy.get("level_bank_seed") != BANK
            or fit_config.get("image_size") != 96 or policy.get("env_image_size") != 96):
        raise ValueError("all16/bank20261020/96px policy and reward configuration required")
    presentation = PresentationConfig(**fit_config["presentation_config"])
    if (presentation.mode != "varied" or presentation.pixel_art
            or set(presentation.themes) != set(THEME_SPLITS["train"] + THEME_SPLITS["validation"])
            or fit_config.get("gate") != asdict(RewardGate())):
        raise ValueError("fresh varied fit requires fixed gates and train/validation HUD themes")
    torch.set_num_threads(4)
    reward, embedded = load_screen_reward_checkpoint(paths["checkpoint"], require_pass=True)
    report = _read(paths["reward_report"])
    if (json.loads(json.dumps(embedded, allow_nan=False)) != report
            or report.get("initialization") != "random" or report.get("imported_checkpoints") != []
            or report.get("old_corpora") != [] or report.get("test_levels_used") is not False
            or report.get("policy_demonstrations") is not False
            or report.get("tasks") != list(TASKS) or report.get("level_bank_seed") != BANK
            or report.get("seed") != fit_config["seed"]
            or report.get("presentation_config") != json.loads(json.dumps(asdict(presentation)))
            or report.get("gate", {}).get("tolerances") != asdict(RewardGate())
            or report.get("gate", {}).get("passed") is not True):
        raise ValueError("fresh reward training report/checkpoint provenance mismatch")
    result = validate_stage_reward(
        reward, admission_path=admission_path, admission_sha256=admission_sha256,
        config=RewardCompositionConfig.from_dict(policy["reward_composition"]),
        gamma=policy["gamma"], horizon=policy["max_episode_steps"], levels=levels, seed=seed)
    if hashes != {f"{name}_sha256": _sha(path) for name, path in paths.items()}:
        raise ValueError("pinned artifacts changed during validation")
    result.update(hashes)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "admission", "policy-config", "reward-config", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args(argv)
    report = validate_stage_reward_checkpoint(
        args.checkpoint, admission_path=args.admission, admission_sha256=args.admission_sha256,
        policy_config=args.policy_config, reward_config=args.reward_config, output=args.output,
        levels=args.levels, seed=args.seed)
    print(json.dumps({"passed": report["passed"], "covered_cells": report["covered_cell_count"],
                      "required_cells": report["required_cells"], "report": str(args.output)}))
    return 0 if report["passed"] else 1
