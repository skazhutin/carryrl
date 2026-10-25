"""Prepare authenticated paired canonical-retention commands without executing training."""

import argparse
from pathlib import Path

from carryrl.training.retention import prepare_retention_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-anchor-coefficient", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--dataset-seed", type=int, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--training-backend", choices=("mlx", "torch"), default="mlx")
    parser.add_argument("--minibatch-episodes", type=int, default=8)
    parser.add_argument("--action-balance", choices=("active_idle", "uniform"),
                        default="active_idle")
    parser.add_argument("--canonical-replay-weight", type=float, default=1.0)
    parser.add_argument("--evaluation-difficulty", type=float, default=1.0)
    parser.add_argument("--level-bank-seed", type=int, default=20261008)
    parser.add_argument("--feature-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--feature-budget-mib", type=int, default=14336)
    parser.add_argument("--frame-budget-mib", type=int, default=4096)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--validation-nll-episodes-per-task", type=int, default=2)
    parser.add_argument("--eval-all-levels", action="store_true",
                        help="evaluate the complete validation bank at each scheduled evaluation")
    parser.add_argument("--reuse-initial-evaluation", type=Path,
                        help="authenticate and pin a complete existing warm-start evaluation")
    args = vars(parser.parse_args(argv))
    args["expected_checkpoint_sha256"] = args.pop("source_checkpoint_sha256")
    args["config_path"] = args.pop("source_config")
    args["backend"] = args.pop("training_backend")
    plan = prepare_retention_plan(**args)
    print(f"prepared_not_executed: {plan['expected_epochs']} epochs, "
          f"{plan['expected_optimizer_updates']} planned updates per arm; "
          f"manifest: {args['output'] / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
