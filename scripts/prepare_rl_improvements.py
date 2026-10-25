"""Prepare future all-task PPO comparisons, without training or launching."""

import argparse
import json
from pathlib import Path

from carryrl.training.rl_improvements import prepare_rl_improvements
from carryrl.training.sequence_config import SequenceRunConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--admission-sha256")
    parser.add_argument("--text-reward-checkpoint", type=Path)
    parser.add_argument("--text-reward-sha256")
    parser.add_argument("--schedule-rollouts", type=int,
                        help="legacy shorthand for both horizons")
    parser.add_argument("--total-rollouts", type=int)
    parser.add_argument("--control-schedule-rollouts", type=int)
    parser.add_argument("--seed", type=int, default=3176)
    parser.add_argument("--episodes-per-task", type=int, default=8)
    args = parser.parse_args(argv)
    manifest = prepare_rl_improvements(
        args.output, base_config=SequenceRunConfig.load(args.config),
        admission_path=args.admission, admission_sha256=args.admission_sha256,
        schedule_rollouts=args.schedule_rollouts, total_rollouts=args.total_rollouts,
        control_schedule_rollouts=args.control_schedule_rollouts, seed=args.seed,
        episodes_per_task=args.episodes_per_task,
        text_reward_checkpoint=args.text_reward_checkpoint,
        text_reward_sha256=args.text_reward_sha256)
    print(json.dumps({"preparation_ready": manifest["preparation_ready"],
                      "training_started": False, "output": str(args.output),
                      "required_inputs": manifest["required_inputs"]}, ensure_ascii=False))
    return 0 if manifest["preparation_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
