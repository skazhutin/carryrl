"""Separate evaluation environments; the policy receives only RGB pixels."""

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from carryrl.envs.navigation import NavigationEnv


def evaluate(
    model: Any | None,
    *,
    seeds: tuple[int, ...],
    split: str = "validation",
    image_size: int = 64,
    max_episode_steps: int = 64,
    random_seed: int = 991,
    gif_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate fixed episodes; a GIF always uses the first declared seed."""
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    if not seeds:
        raise ValueError("evaluation needs at least one seed")
    env = NavigationEnv(size=image_size, max_steps=max_episode_steps, split=split)
    rng = np.random.default_rng(random_seed)
    episodes = []
    frames = []
    started = time.monotonic()
    previous_mode = model.policy.training if model is not None else None
    try:
        for episode_index, seed in enumerate(seeds):
            obs, info = env.reset(seed=seed)
            episode_return = 0.0
            initial_distance = info.get("distance")
            steps = 0
            if gif_path is not None and episode_index == 0:
                frames.append(Image.fromarray(obs))
            while True:
                if model is None:
                    action = int(rng.integers(env.action_space.n))
                else:
                    # No environment info, coordinates, map IDs, or rewards enter predict.
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(np.asarray(action).item())
                obs, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(reward):
                    raise FloatingPointError("nonfinite reward during evaluation")
                episode_return += float(reward)
                steps += 1
                if gif_path is not None and episode_index == 0:
                    frames.append(Image.fromarray(obs))
                if terminated or truncated:
                    break
            success = info.get("success")
            if info.get("recognition") != "ok" or type(success) is not bool:
                success = None
            episodes.append({
                "seed": seed, "level_id": info.get("level_id"), "map_id": info.get("map_id"),
                "return": episode_return, "steps": steps, "success": success,
                "terminated": bool(terminated), "truncated": bool(truncated),
                "initial_distance": initial_distance, "final_distance": info.get("distance"),
                "recognition": info.get("recognition", "unknown"),
            })
        duration = time.monotonic() - started
        if gif_path is not None:
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100,
                           loop=0, optimize=False)
    finally:
        env.close()
        if model is not None:
            model.policy.set_training_mode(previous_mode)
    count = len(episodes)
    successes = sum(episode["success"] is True for episode in episodes)
    unknown = sum(episode["success"] is None for episode in episodes)
    return {
        "split": split, "episodes_started": count, "unknown_outcomes": unknown,
        "success_rate_lower": successes / count,
        "success_rate_upper": (successes + unknown) / count,
        "mean_return": float(np.mean([episode["return"] for episode in episodes])),
        "mean_steps": float(np.mean([episode["steps"] for episode in episodes])),
        "environment_steps": sum(episode["steps"] for episode in episodes),
        "wall_seconds": duration, "episodes": episodes,
        "policy": "random" if model is None else "frozen_deterministic",
        "gif_seed": seeds[0] if gif_path is not None else None,
    }
