"""One explicitly authorized fresh policy job; never imports previous policy weights."""

import argparse
import json
from pathlib import Path

from carryrl.training.cycle_jobs import run_policy_job
from carryrl.training.sequence_config import SequenceRunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-initial", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    expected = json.loads(args.expected_initial.read_text()) if args.expected_initial else None
    result = run_policy_job(SequenceRunConfig.load(args.config), args.output,
                            expected_initial=expected, full=args.full)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
