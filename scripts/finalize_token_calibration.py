"""Finalize the original common LR after both supervised calibration runs finish."""

import argparse
import json
from pathlib import Path

from carryrl.training.token_calibration import finalize_common_calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    selection = finalize_common_calibration(args.plan)
    print(json.dumps({"ready_to_train": True,
                      "selected_learning_rate": selection["selected_learning_rate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
