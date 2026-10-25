"""Collect canonical TRAIN teacher trajectories and prepare two paired RGB appearance arms."""

import argparse
from dataclasses import replace
from pathlib import Path

from carryrl.envs.atomic_suite import TASKS
from carryrl.training.appearance import prepare_appearance_bundle
from carryrl.training.imitation import collect_scripted_demonstrations
from carryrl.training.sequence_config import SequenceRunConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8288)
    parser.add_argument("--frame-budget-mib", type=int, default=256)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("appearance output already exists")
    if args.frame_budget_mib <= 0:
        raise ValueError("frame budget must be positive")
    config = replace(SequenceRunConfig.load(args.config), curriculum_difficulty=1.0,
                     evaluation_difficulty=1.0)
    if config.tasks != TASKS or config.env_image_size != 96:
        raise ValueError("first appearance preparation requires the full 16-task canonical suite")
    budget = args.frame_budget_mib * 1024**2
    dataset = collect_scripted_demonstrations(args.episodes_per_task, tasks=config.tasks,
        seed=args.seed, difficulty=1.0, image_size=96, max_steps=config.max_episode_steps,
        level_bank_seed=config.level_bank_seed, episode_instructions=config.episode_instructions,
        max_frame_bytes=budget)
    report = prepare_appearance_bundle(dataset, args.output, config=config, seed=args.seed,
                                      max_frame_bytes=budget)
    print(f"prepared_not_trained: {report['source_bundle_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
