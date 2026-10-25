"""Prepare synthetic control-text supervision without fitting or loading a model."""

import argparse
import json
from pathlib import Path

from carryrl.rewards.control_text_data import prepare_control_text_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/text_control_reward_preparation_v1")
    )
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--scenario-groups", type=int, default=4000)
    parser.add_argument(
        "--corpus-version", choices=("v3", "v4", "v5", "v6", "v7", "v8"), default="v3"
    )
    parser.add_argument("--reservation-registry", type=Path)
    args = parser.parse_args(argv)
    if args.reservation_registry is not None and args.corpus_version not in {"v6", "v7", "v8"}:
        parser.error("--reservation-registry requires --corpus-version v6, v7 or v8")
    prepare = prepare_control_text_dataset
    if args.corpus_version == "v4":
        from carryrl.rewards.control_text_data_v4 import prepare_control_text_dataset as prepare
    elif args.corpus_version == "v5":
        from carryrl.rewards.control_text_data_v5 import prepare_control_text_dataset as prepare
    elif args.corpus_version == "v6":
        from carryrl.rewards.control_text_data_v6 import prepare_control_text_dataset as prepare
    elif args.corpus_version == "v7":
        from carryrl.rewards.control_text_data_v7 import prepare_control_text_dataset as prepare
    elif args.corpus_version == "v8":
        from carryrl.rewards.control_text_data_v8 import prepare_control_text_dataset as prepare
    options = {}
    if args.corpus_version in {"v6", "v7", "v8"}:
        options["reservation_registry"] = args.reservation_registry
    manifest = prepare(args.output, seed=args.seed, scenario_groups=args.scenario_groups, **options)
    print(json.dumps({"output": str(args.output.resolve()), **manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
