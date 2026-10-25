"""Prepare four validation-only RGB/text/memory/executed-history diagnostic pairs."""

import argparse
import json

from carryrl.evaluation.cycle_input_pairs import prepare_cycle_input_pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--instructions", default="configs/atomic_instructions.json")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--level-bank-seed", type=int, default=20261020)
    args = parser.parse_args(argv)
    path = prepare_cycle_input_pairs(
        args.output,
        instructions=args.instructions,
        seed=args.seed,
        level_bank_seed=args.level_bank_seed,
    )
    print(json.dumps({"pairs": str(path.resolve()), "split": "validation", "cases": 4}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
