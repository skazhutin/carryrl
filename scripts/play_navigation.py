"""Manually play: arrows hold for fixed ticks, R resets, Esc/window close quits.

This script does not record demonstrations or use OS input injection. Training
and playback use the same game and visual observation rules.
"""

import argparse
import json
import time
from pathlib import Path

import pygame

from carryrl.envs.navigation import TICK_SECONDS, NavigationEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Manual play time limit, independent of training configuration",
    )
    parser.add_argument(
        "--smoke-seconds", type=float, help="Close the actual window after this many seconds"
    )
    parser.add_argument(
        "--smoke-report",
        type=Path,
        help="Save native window lifecycle metadata; does not verify manual input",
    )
    args = parser.parse_args()
    if args.smoke_seconds is not None and args.smoke_seconds <= 0:
        parser.error("--smoke-seconds must be positive")
    env = NavigationEnv(render_mode="human", split=args.split, max_steps=args.max_steps)
    seed = args.seed
    env.reset(seed=seed)
    window_report = {
        "scope": (
            "native window initialization and shutdown; "
            "manual input not automatically verified"
        ),
        "status": "opened",
        "driver": pygame.display.get_driver(),
        "display_size": list(pygame.display.get_surface().get_size()),
        "caption": pygame.display.get_caption()[0],
    }
    print(json.dumps(window_report), flush=True)
    if args.smoke_report:
        args.smoke_report.parent.mkdir(parents=True, exist_ok=True)
        args.smoke_report.write_text(json.dumps(window_report, indent=2) + "\n")
    finished, running = False, True
    next_tick = time.monotonic() + TICK_SECONDS
    close_at = time.monotonic() + args.smoke_seconds if args.smoke_seconds else None
    clock = pygame.time.Clock()
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    seed += 1
                    env.reset(seed=seed)
                    finished = False
                    next_tick = time.monotonic() + TICK_SECONDS
            now = time.monotonic()
            if close_at is not None and now >= close_at:
                running = False
            if running and not finished and now >= next_tick:
                keys = pygame.key.get_pressed()
                action = next(
                    (
                        value
                        for key, value in (
                            (pygame.K_UP, 1),
                            (pygame.K_DOWN, 2),
                            (pygame.K_LEFT, 3),
                            (pygame.K_RIGHT, 4),
                        )
                        if keys[key]
                    ),
                    0,
                )
                _, reward, terminated, truncated, info = env.step(action)
                finished = terminated or truncated
                if finished:
                    print(
                        f"success={info['success']} ticks={info['elapsed_ticks']} "
                        f"reward={reward:.2f}; R resets"
                    )
                # Do not catch up by applying stale repeated actions after a stalled UI.
                next_tick = now + TICK_SECONDS
            clock.tick(60)
    finally:
        env.close()
        window_report["status"] = "closed_cleanly"
        if args.smoke_report:
            args.smoke_report.write_text(json.dumps(window_report, indent=2) + "\n")
        print("Navigation window closed cleanly.")


if __name__ == "__main__":
    main()
