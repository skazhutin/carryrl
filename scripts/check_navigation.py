"""Validate random episodes and save explicitly scripted, agent-independent evidence."""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from carryrl.envs.navigation import ACTION_DELTAS, NavigationEnv
from carryrl.rewards.navigation import navigation_reward, recognize_navigation


def scripted_visible_path(frame):
    """A screen-based BFS is a diagnostic script, never the learning agent."""
    state = recognize_navigation(frame)
    frontier = deque([(state.player, [])])
    visited = {state.player}
    while frontier:
        position, actions = frontier.popleft()
        if position == state.goal:
            return actions
        for action, (dr, dc) in enumerate(ACTION_DELTAS[1:], start=1):
            neighbor = position[0] + dr, position[1] + dc
            if (
                0 <= neighbor[0] < 5
                and 0 <= neighbor[1] < 5
                and neighbor not in state.walls
                and neighbor not in visited
            ):
                visited.add(neighbor)
                frontier.append((neighbor, actions + [action]))
    raise RuntimeError("Recognized visible goal is unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/navigation_check"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    env = NavigationEnv(render_mode="rgb_array")
    episodes = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        frame, reset_info = env.reset(seed=seed)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(rng.integers(5))
            next_frame, reward, terminated, truncated, info = env.step(action)
            expected_reward, visible = navigation_reward(frame, next_frame)
            if reward != expected_reward or info["success"] != visible.success:
                raise RuntimeError("Reward or success differs from independently recognized pixels")
            if not env.observation_space.contains(next_frame):
                raise RuntimeError("Observation violates RGB contract")
            total_reward += reward
            frame = next_frame
        episodes.append(
            {
                "seed": seed,
                "map_id": reset_info["map_id"],
                "level_id": reset_info["level_id"],
                "success": info["success"],
                "ticks": info["elapsed_ticks"],
                "return": total_reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
    elapsed = time.perf_counter() - started
    # A separate scripted success shows rendering/lifecycle, not learned performance.
    frame, info = env.reset(seed=args.seed)
    images = [Image.fromarray(frame).resize((512, 512), Image.Resampling.NEAREST)]
    images[0].save(args.output / "initial_frame.png")
    actions = scripted_visible_path(frame)
    for action in actions:
        frame, _, terminated, truncated, info = env.step(action)
        images.append(Image.fromarray(frame).resize((512, 512), Image.Resampling.NEAREST))
    images[-1].save(args.output / "success_frame.png")
    images[0].save(
        args.output / "scripted_diagnostic_episode.gif",
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0,
    )
    if not terminated or truncated or not info["success"]:
        raise RuntimeError("Scripted visual diagnostic did not terminate with visible success")
    env.close()
    ticks = sum(episode["ticks"] for episode in episodes)
    report = {
        "status": "passed",
        "strategy": "uniform random physical actions",
        "episodes": args.episodes,
        "seed": args.seed,
        "successes": sum(episode["success"] for episode in episodes),
        "ticks": ticks,
        "elapsed_seconds": elapsed,
        "checked_ticks_per_second": ticks / elapsed,
        "evidence_strategy": "scripted visible shortest path; not a learned agent",
        "evidence_actions": actions,
        "episode_results": episodes,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {key: value for key, value in report.items() if key != "episode_results"}
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
