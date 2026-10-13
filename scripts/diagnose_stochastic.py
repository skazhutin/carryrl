"""Post-pilot frozen stochastic diagnostic; the original primary evaluation is unchanged."""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from stable_baselines3 import PPO

from carryrl.envs.navigation import NavigationEnv
from carryrl.training.run import append_json, atomic_json

LEVEL_SEEDS = tuple(range(1000, 1030))
ACTION_SEEDS = (5000, 5001, 5002)
MODEL_SEEDS = (0, 1, 2)


def episode_rng_seed(action_seed: int, level_seed: int) -> int:
    return action_seed * 65536 + level_seed


def frozen_state(model: Any) -> tuple[dict[str, torch.Tensor], int, bool]:
    return ({name: tensor.detach().cpu().clone()
             for name, tensor in model.policy.state_dict().items()},
            model.num_timesteps, model.policy.training)


def assert_frozen(model: Any, before: tuple) -> None:
    parameters, timesteps, _ = before
    assert model.num_timesteps == timesteps, "evaluation changed training counter"
    after = model.policy.state_dict()
    assert set(parameters) == set(after), "evaluation changed state dictionary keys"
    assert all(torch.equal(tensor, after[name].detach().cpu())
               for name, tensor in parameters.items()), "evaluation changed policy tensors"


def run_episode(
    model: Any | None, *, level_seed: int, action_seed: int, image_size: int = 64,
    max_steps: int = 64, gif_path: Path | None = None,
) -> dict:
    """One fixed episode; only uint8 RGB is passed to a learned policy."""
    seed = episode_rng_seed(action_seed, level_seed)
    rng_before = torch.random.get_rng_state()
    state_before = frozen_state(model) if model is not None else None
    env = NavigationEnv(size=image_size, max_steps=max_steps, split="validation")
    frames = []
    actions = []
    rewards = []
    info: dict = {}
    terminated = truncated = False
    error = None
    started = time.monotonic()
    try:
        torch.manual_seed(seed)
        pixels, info = env.reset(seed=level_seed)
        if gif_path is not None:
            frames.append(Image.fromarray(pixels))
        for _ in range(max_steps):
            if model is None:
                action = int(torch.distributions.Categorical(logits=torch.zeros(5)).sample())
            else:
                action, _ = model.predict(pixels, deterministic=False)
                action = int(np.asarray(action).item())
            actions.append(action)
            pixels, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(reward):
                raise FloatingPointError("nonfinite diagnostic reward")
            rewards.append(float(reward))
            if gif_path is not None:
                frames.append(Image.fromarray(pixels))
            if terminated or truncated:
                break
        if not terminated and not truncated:
            raise RuntimeError("environment did not end by the declared episode budget")
    except Exception as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
    finally:
        env.close()
        torch.random.set_rng_state(rng_before)
        if model is not None:
            model.policy.set_training_mode(state_before[2])
            assert_frozen(model, state_before)
    success = info.get("success")
    known = error is None and info.get("recognition") == "ok" and type(success) is bool
    if gif_path is not None and frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100,
                       loop=0, optimize=False)
    return {
        "level_seed": level_seed, "action_seed": action_seed, "torch_episode_seed": seed,
        "policy": "uniform_torch_categorical" if model is None else "frozen_stochastic",
        "map_id": info.get("map_id"), "level_id": info.get("level_id"),
        "success": success if known else None, "recognition": info.get("recognition"),
        "error": error, "actions": actions, "observed_steps": len(rewards),
        "attempted_actions": len(actions), "return": sum(rewards) if known else None,
        "terminated": bool(terminated), "truncated": bool(truncated),
        "wall_seconds": time.monotonic() - started,
    }


def aggregate(rows: list[dict]) -> dict:
    count = len(rows)
    known = sum(row["success"] is not None for row in rows)
    successes = sum(row["success"] is True for row in rows)
    unknown = count - known
    return {
        "episodes_started": count, "known_outcomes": known, "unknown_outcomes": unknown,
        "success_rate_lower": successes / count if count else None,
        "success_rate_upper": (successes + unknown) / count if count else None,
        "mean_return": float(np.mean([row["return"] for row in rows]))
        if count and unknown == 0 else None,
        "mean_observed_steps": float(np.mean([row["observed_steps"] for row in rows]))
        if count else None,
        "return_rule": "undefined when any outcome is unknown; no favorable subset",
    }


def initial_control_extension(output: Path, checkpoints: list[dict], image_size: int,
                              learned_rows: list[dict]) -> dict:
    """Evaluate original scratch networks under the separately preregistered extension."""
    rows = []
    started = time.monotonic()
    atomic_json(output / "initial_control_status.json", {"status": "running"})
    for checkpoint in checkpoints:
        model = PPO.load(checkpoint["path"], device="cpu")
        before = frozen_state(model)
        if model.num_timesteps != 0:
            raise ValueError("initial control checkpoint must precede training")
        for action_seed in ACTION_SEEDS:
            for level_seed in LEVEL_SEEDS:
                row = run_episode(model, level_seed=level_seed, action_seed=action_seed,
                                  image_size=image_size)
                row = {"model_seed": checkpoint["model_seed"], **row}
                rows.append(row)
                append_json(output / "initial_control_episodes.jsonl", row)
        assert_frozen(model, before)
        checkpoint["sha256_after"] = hashlib.sha256(
            Path(checkpoint["path"]).read_bytes()).hexdigest()
        if checkpoint["sha256_after"] != checkpoint["sha256_before"]:
            raise RuntimeError("initial diagnostic changed checkpoint archive")
    if len(rows) != 270:
        raise RuntimeError("initial control episode count differs from its registered extension")
    per_model = {str(seed): aggregate([row for row in rows if row["model_seed"] == seed])
                 for seed in MODEL_SEEDS}
    differences = {
        str(seed): aggregate([row for row in learned_rows if row["model_seed"] == seed])[
            "success_rate_lower"] - per_model[str(seed)]["success_rate_lower"]
        for seed in MODEL_SEEDS
    }
    summary = {
        "status": "completed_with_unknowns" if any(row["success"] is None for row in rows)
        else "completed", "episodes_started": len(rows),
        "wall_seconds": time.monotonic() - started, "initial": aggregate(rows),
        "models": per_model, "learned_minus_initial_success_lower": differences,
        "mean_model_difference": float(np.mean(list(differences.values()))),
        "checkpoints_unchanged": checkpoints,
        "interpretation": "separate initial-control extension registered "
        "before stochastic actions; "
        "three paired network initializations; descriptive, no statistical superiority claim",
    }
    atomic_json(output / "initial_control_summary.json", summary)
    atomic_json(output / "initial_control_status.json", {"status": summary["status"]})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-control", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoints = []
    try:
        protocol_path = args.pilot / "protocol.json"
        primary_protocol = json.loads(protocol_path.read_text())
        config = primary_protocol["config"]
        if primary_protocol["seeds"] != list(MODEL_SEEDS):
            raise ValueError("this diagnostic requires all three original model seeds")
        if config["eval_seeds"] != list(LEVEL_SEEDS) or config["max_episode_steps"] != 64:
            raise ValueError("primary validation protocol differs from the prescribed diagnostic")
        for model_seed in MODEL_SEEDS:
            run = args.pilot / f"seed_{model_seed}"
            summary = json.loads((run / "summary.json").read_text())
            if (summary["status"] != "completed"
                    or summary["timesteps"] != config["total_timesteps"]):
                raise ValueError("all original final checkpoints must exist before diagnosis")
            checkpoint = Path(summary["latest_checkpoint"]) / "model.zip"
            checkpoints.append({
                "model_seed": model_seed, "path": str(checkpoint.resolve()),
                "sha256_before": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "training_timesteps": summary["timesteps"],
            })
        atomic_json(args.output / "protocol.json", {
            "registered_utc": datetime.now(UTC).isoformat(),
            "chosen_after_midpilot": True, "changes_primary_results": False,
            "training_updates": 0,
            "checkpoint_selection": "final checkpoint for every declared seed",
            "model_seeds": MODEL_SEEDS, "level_seeds": LEVEL_SEEDS, "action_seeds": ACTION_SEEDS,
            "torch_episode_seed_rule": "action_seed * 65536 + level_seed",
            "image_size": config["image_size"], "max_steps": 64, "device": "cpu",
            "learned_episodes_planned": 270, "random_episodes_planned": 90,
            "random_control": "uniform Torch Categorical over five actions; "
            "once per action/level seed",
            "control_reuse": "same 90 episodes reused across all models, "
            "not independent repetitions",
            "unknown_rule": "all attempts retained; success bounds; undefined return with unknowns",
            "retries": 0, "temperatures": None, "checkpoints": checkpoints,
            "primary_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fixed_gif": {"model_seed": 0, "action_seed": 5000, "level_seed": 1000},
        })
        initial_checkpoints = []
        if args.initial_control:
            for model_seed in MODEL_SEEDS:
                checkpoint = args.pilot / f"seed_{model_seed}/checkpoints/step_00000000/model.zip"
                initial_checkpoints.append({
                    "model_seed": model_seed, "path": str(checkpoint.resolve()),
                    "sha256_before": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                })
            atomic_json(args.output / "initial_control_protocol.json", {
                "registered_utc": datetime.now(UTC).isoformat(),
                "registered_before_any_stochastic_actions": True,
                "chosen_after_deterministic_pilot_before_stochastic_results": True,
                "episodes_planned": 270, "training_updates": 0,
                "checkpoint_selection": "step_00000000 for all three declared model seeds",
                "model_seeds": MODEL_SEEDS, "action_seeds": ACTION_SEEDS,
                "level_seeds": LEVEL_SEEDS,
                "torch_episode_seed_rule": "action_seed * 65536 + level_seed",
                "max_steps": 64, "image_size": config["image_size"], "device": "cpu",
                "controls": "reuse original diagnostic 90 random episodes; no new random runs",
                "changes_original_360_report": False, "checkpoints": initial_checkpoints,
                "retries": 0,
            })
        atomic_json(args.output / "status.json", {"status": "running"})
        torch.set_num_threads(4)
        started = time.monotonic()
        random_rows = []
        learned_rows = []
        for action_seed in ACTION_SEEDS:
            for level_seed in LEVEL_SEEDS:
                row = run_episode(None, level_seed=level_seed, action_seed=action_seed,
                                  image_size=config["image_size"])
                random_rows.append(row)
                append_json(args.output / "episodes.jsonl", {"model_seed": None, **row})
        for checkpoint in checkpoints:
            model = PPO.load(checkpoint["path"], device="cpu")
            before = frozen_state(model)
            if model.num_timesteps != checkpoint["training_timesteps"]:
                raise ValueError("loaded model counter disagrees with original final checkpoint")
            for action_seed in ACTION_SEEDS:
                for level_seed in LEVEL_SEEDS:
                    gif = args.output / "fixed_model0_action5000_level1000.gif" if (
                        checkpoint["model_seed"] == 0 and action_seed == 5000 and level_seed == 1000
                    ) else None
                    row = run_episode(model, level_seed=level_seed, action_seed=action_seed,
                                      image_size=config["image_size"], gif_path=gif)
                    row = {"model_seed": checkpoint["model_seed"], **row}
                    learned_rows.append(row)
                    append_json(args.output / "episodes.jsonl", row)
            assert_frozen(model, before)
            checkpoint["sha256_after"] = hashlib.sha256(Path(checkpoint["path"]).read_bytes())
            checkpoint["sha256_after"] = checkpoint["sha256_after"].hexdigest()
            if checkpoint["sha256_after"] != checkpoint["sha256_before"]:
                raise RuntimeError("diagnostic changed checkpoint archive")
        if len(random_rows) != 90 or len(learned_rows) != 270:
            raise RuntimeError("diagnostic episode counts do not match the registered protocol")
        per_model = {
            str(seed): aggregate([row for row in learned_rows if row["model_seed"] == seed])
            for seed in MODEL_SEEDS
        }
        any_unknown = any(row["success"] is None for row in random_rows + learned_rows)
        summary = {
            "status": "completed_with_unknowns" if any_unknown else "completed",
            "wall_seconds": time.monotonic() - started,
            "episodes_started": len(random_rows) + len(learned_rows),
            "random": aggregate(random_rows), "learned": aggregate(learned_rows),
            "models": per_model,
            "models_by_action_seed": {
                f"model{seed}_action{action}": aggregate([
                    row for row in learned_rows if row["model_seed"] == seed
                    and row["action_seed"] == action])
                for seed in MODEL_SEEDS for action in ACTION_SEEDS
            },
            "random_by_action_seed": {
                str(action): aggregate([row for row in random_rows if row["action_seed"] == action])
                for action in ACTION_SEEDS
            },
            "mean_model_success_lower": float(np.mean([
                value["success_rate_lower"] for value in per_model.values()])),
            "checkpoints_unchanged": checkpoints,
            "interpretation": "post-pilot diagnosis, not a predeclared primary comparison; "
            "episodes do not replace independent model seeds; random controls are reused",
        }
        atomic_json(args.output / "summary.json", summary)
        atomic_json(args.output / "status.json", {"status": summary["status"]})
        print(json.dumps(summary, indent=2))
        if args.initial_control:
            try:
                extension = initial_control_extension(args.output, initial_checkpoints,
                                                      config["image_size"], learned_rows)
                print(json.dumps({"initial_control_extension": extension}, indent=2))
                if extension["status"] != "completed":
                    return 1
            except BaseException as error:
                atomic_json(args.output / "initial_control_status.json", {
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "error_type": type(error).__name__,
                    "error": str(error), "original_360_report_retained": True, "retries": 0,
                })
                return 1
        return 0 if summary["status"] == "completed" else 1
    except BaseException as error:
        atomic_json(args.output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error_type": type(error).__name__, "error": str(error), "retries": 0,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
