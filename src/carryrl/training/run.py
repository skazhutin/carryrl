"""Bounded local PPO runs with frozen evaluation and atomic resumable snapshots."""

import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from carryrl.envs.navigation import NavigationEnv
from carryrl.evaluation.runner import evaluate
from carryrl.training.config import RunConfig
from carryrl.training.model import build_model

SOURCE_IDENTITY_VERSION = "carryrl.source.v2"


def file_sha256(path: Path) -> str:
    """Hash a file with bounded reads, including large checkpoints and lock files."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_and_lock_sha256(root: Path) -> str:
    """Identify source contents and paths with unambiguous versioned framing."""
    paths = [*(root / "src" / "carryrl").rglob("*.py"),
             *(root / "scripts").rglob("*.py")]
    lock = root / "uv.lock"
    if lock.exists():
        paths.append(lock)
    digest = hashlib.sha256(SOURCE_IDENTITY_VERSION.encode() + b"\0")
    for path in sorted(paths, key=lambda path: path.relative_to(root).as_posix()):
        relative_path = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def atomic_json(path: Path, value: Any, *, indent: int | None = 2) -> None:
    """Replace one JSON record only after the new complete record is flushed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=indent, allow_nan=False,
                      separators=(",", ":") if indent is None else None)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def append_json(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    source_digest = _source_and_lock_sha256(root)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                            text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                           text=True, check=False)
    return {
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_dirty": dirty.stdout.splitlines() if dirty.returncode == 0 else None,
        "source_and_lock_sha256": source_digest,
        "source_identity_version": SOURCE_IDENTITY_VERSION,
        "dependencies": {name: version(name) for name in
                         ("torch", "stable-baselines3", "gymnasium", "numpy", "pillow")},
        "torch_mps_available": torch.backends.mps.is_available(),
        "mps_fallback_environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }


class TimingEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.step_wall_seconds = 0.0
        self.reset_wall_seconds = 0.0
        self.decisions = 0
        self.completed_episodes = 0
        self.episode_returns: list[float] = []
        self._episode_return = 0.0

    def reset(self, **kwargs):
        started = time.monotonic()
        result = self.env.reset(**kwargs)
        self.reset_wall_seconds += time.monotonic() - started
        self._episode_return = 0.0
        return result

    def step(self, action):
        started = time.monotonic()
        result = self.env.step(action)
        self.step_wall_seconds += time.monotonic() - started
        self.decisions += 1
        self._episode_return += float(result[1])
        if result[2] or result[3]:
            self.completed_episodes += 1
            self.episode_returns.append(self._episode_return)
        return result


class NumericGuard(BaseCallback):
    def _on_step(self) -> bool:
        if not np.isfinite(self.locals["rewards"]).all():
            raise FloatingPointError("nonfinite training reward")
        return True


def save_checkpoint(model: PPO, output: Path, config: RunConfig,
                    progress: dict[str, Any]) -> Path:
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    destination = checkpoints / f"step_{model.num_timesteps:08d}"
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=checkpoints))
    try:
        model.save(temporary / "model.zip")
        atomic_json(temporary / "progress.json", {
            **progress, "config": config.to_dict(), "timesteps": model.num_timesteps,
            "optimizer_state_saved": True,
            "resume_rule": "weights and optimizer restored; new episode, rollout discarded; "
            "RNG restarts from original seed",
            "resume_rng_restored": False,
        })
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    atomic_json(output / "latest.json", {"checkpoint": str(destination.resolve())})
    return destination


def load_checkpoint(checkpoint: Path, env: gym.Env, config: RunConfig) -> tuple[PPO, dict]:
    with (checkpoint / "progress.json").open() as stream:
        progress = json.load(stream)
    old_config = RunConfig.from_dict(progress["config"])
    if replace(old_config, total_timesteps=config.total_timesteps,
               max_wall_seconds=config.max_wall_seconds) != config:
        raise ValueError("resume must preserve all settings except target steps and wall budget")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    torch.set_num_threads(config.torch_threads)
    model = PPO.load(checkpoint / "model.zip", env=env, device=config.device, force_reset=True)
    if model.num_timesteps != progress["timesteps"]:
        raise ValueError("checkpoint counters disagree")
    if model.num_timesteps % config.n_steps:
        raise ValueError("resume requires a completed rollout/update boundary")
    if config.total_timesteps <= model.num_timesteps:
        raise ValueError("resume target must exceed checkpoint steps")
    return model, progress


def train(config: RunConfig, output: Path, resume: Path | None = None) -> dict[str, Any]:
    """Train one seed sequentially; the target counts decisions, not rendered ticks."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    env = None
    summary: dict[str, Any] = {"status": "running", "target_timesteps": config.total_timesteps}
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(), "config": config.to_dict(),
        "config_sha256": hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode())
        .hexdigest(), "provenance": provenance(), "candidate": "PPO_small_CNN",
        "initialization": "scratch" if resume is None else "continued_scratch_checkpoint",
        "human_demonstrations": False, "policy_inputs": "RGB frame only",
        "reward_protocol": "screen-derived distance and visible success; engineering proposal",
        "action_protocol": "5 keyboard holds, one fixed 0.1-second simulation tick per decision",
        "training_split": "train", "evaluation_split": "validation",
        "resume_from": str(resume.resolve()) if resume is not None else None,
        "resume_exact_episode": False,
        "resume_rng_restored": False,
        "resume_rng_rule": "SB3 load reinitializes original seed; no exact trajectory continuation",
        "wall_stop_rule": "at completed rollout/update boundary; one-rollout overshoot possible",
    }
    atomic_json(output / "metadata.json", metadata)
    atomic_json(output / "status.json", summary)
    try:
        if shutil.disk_usage(output).free < 200 * 1024 * 1024:
            raise OSError("less than 200 MiB free disk; run not started")
        env = TimingEnv(NavigationEnv(size=config.image_size, max_steps=config.max_episode_steps,
                                     split="train"))
        model = (build_model(config, env) if resume is None
                 else load_checkpoint(resume, env, config)[0])
        initial_steps = model.num_timesteps
        params = sum(parameter.numel() for parameter in model.policy.parameters())
        baseline = evaluate(None, seeds=config.eval_seeds, image_size=config.image_size,
                            max_episode_steps=config.max_episode_steps, random_seed=991,
                            gif_path=output / "random_fixed_seed.gif")
        append_json(output / "evaluation.jsonl", {"training_timesteps": initial_steps,
                                                 "kind": "random_baseline", **baseline})
        initial = evaluate(model, seeds=config.eval_seeds, image_size=config.image_size,
                           max_episode_steps=config.max_episode_steps)
        append_json(output / "evaluation.jsonl", {"training_timesteps": initial_steps,
                                                 "kind": "initial_policy", **initial})
        evaluation_seconds = baseline["wall_seconds"] + initial["wall_seconds"]
        learning_seconds = 0.0
        initial_updates = model._n_updates
        checkpoint = save_checkpoint(model, output, config, {"phase": "initial"})
        last_evaluation_steps = initial_steps
        final = initial
        status = "completed"
        while model.num_timesteps < config.total_timesteps:
            if time.monotonic() - started >= config.max_wall_seconds:
                status = "resource_stopped"
                break
            interval_started = time.monotonic()
            env_seconds_before = env.step_wall_seconds + env.reset_wall_seconds
            model.learn(total_timesteps=config.n_steps, reset_num_timesteps=False,
                        callback=NumericGuard(), progress_bar=False)
            interval_seconds = time.monotonic() - interval_started
            learning_seconds += interval_seconds
            for parameter in model.policy.parameters():
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError("nonfinite policy parameters after update")
            losses = {key: float(value) for key, value in model.logger.name_to_value.items()
                      if key.startswith("train/") and isinstance(value, (float, int, np.number))}
            diagnostics = []
            explained_variance = losses.get("train/explained_variance")
            if explained_variance is not None and np.isnan(explained_variance):
                losses["train/explained_variance"] = None
                diagnostics.append("explained_variance undefined for zero return variance")
            if not all(value is None or np.isfinite(value) for value in losses.values()):
                raise FloatingPointError("nonfinite PPO update metric")
            append_json(output / "training.jsonl", {
                "timesteps": model.num_timesteps, "optimizer_epochs": model._n_updates,
                "elapsed_wall_seconds": time.monotonic() - started,
                "rollout_and_update_seconds": interval_seconds,
                "env_step_and_reset_seconds": env.step_wall_seconds + env.reset_wall_seconds
                - env_seconds_before, "completed_train_episodes": env.completed_episodes,
                "mean_recent_train_return": float(np.mean(env.episode_returns[-20:]))
                if env.episode_returns else None, "metrics": losses, "diagnostics": diagnostics,
            })
            if model.num_timesteps % config.eval_interval == 0:
                final = evaluate(model, seeds=config.eval_seeds, image_size=config.image_size,
                                 max_episode_steps=config.max_episode_steps)
                evaluation_seconds += final["wall_seconds"]
                append_json(output / "evaluation.jsonl", {"training_timesteps": model.num_timesteps,
                                                         "kind": "validation", **final})
                last_evaluation_steps = model.num_timesteps
                checkpoint = save_checkpoint(model, output, config, {"phase": "trained"})
        final_video = evaluate(model, seeds=(config.eval_seeds[0],), image_size=config.image_size,
                               max_episode_steps=config.max_episode_steps,
                               gif_path=output / "agent_fixed_seed.gif")
        evaluation_seconds += final_video["wall_seconds"]
        append_json(output / "evaluation.jsonl", {"training_timesteps": model.num_timesteps,
                                                 "kind": "fixed_seed_video", **final_video})
        if last_evaluation_steps != model.num_timesteps:
            final = evaluate(model, seeds=config.eval_seeds, image_size=config.image_size,
                             max_episode_steps=config.max_episode_steps)
            evaluation_seconds += final["wall_seconds"]
            append_json(output / "evaluation.jsonl", {"training_timesteps": model.num_timesteps,
                                                     "kind": "validation", **final})
        if checkpoint.name != f"step_{model.num_timesteps:08d}":
            checkpoint = save_checkpoint(model, output, config, {"phase": "trained"})
        elapsed = time.monotonic() - started
        final_provenance = provenance()
        summary = {
            "status": status, "target_timesteps": config.total_timesteps,
            "initial_timesteps": initial_steps, "timesteps": model.num_timesteps,
            "decisions_this_run": env.decisions, "optimizer_epochs_this_run":
            model._n_updates - initial_updates, "policy_parameters": params,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "peak_rss_scope": "process lifetime; includes prior seeds in a same-process series",
            "wall_seconds": elapsed, "learning_wall_seconds": learning_seconds,
            "evaluation_wall_seconds": evaluation_seconds,
            "env_step_wall_seconds": env.step_wall_seconds,
            "env_reset_wall_seconds": env.reset_wall_seconds,
            "full_run_decisions_per_second": env.decisions / elapsed,
            "learning_decisions_per_second": env.decisions / learning_seconds
            if learning_seconds else None,
            "wall_budget_overshoot_seconds": max(0.0, elapsed - config.max_wall_seconds),
            "random_success_rate": baseline["success_rate_lower"],
            "initial_success_rate": initial["success_rate_lower"],
            "final_success_rate_lower": final["success_rate_lower"],
            "final_success_rate_upper": final["success_rate_upper"],
            "final_mean_return": final["mean_return"],
            "final_unknown_outcomes": final["unknown_outcomes"],
            "latest_checkpoint": str(checkpoint.resolve()),
            "inference_video_seed": config.eval_seeds[0],
            "final_provenance": final_provenance,
            "source_changed_during_run": final_provenance["source_and_lock_sha256"]
            != metadata["provenance"]["source_and_lock_sha256"],
            "interpretation": "development pilot; no statistical superiority claim",
        }
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "status.json", summary)
        return summary
    except BaseException as error:
        atomic_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error_type": type(error).__name__, "error": str(error),
            "wall_seconds": time.monotonic() - started,
            "latest_checkpoint_retained": (output / "latest.json").exists(),
        })
        raise
    finally:
        if env is not None:
            env.close()
