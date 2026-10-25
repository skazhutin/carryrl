"""Evaluate frozen local text-control model; sealed test requires an explicit opt-in."""

import argparse
import json

from carryrl.rewards.control_text import load_text_control_reward
from carryrl.rewards.control_text_training import (
    read_control_dataset,
    validate_text_control_predictions,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--runtime-instructions")
    parser.add_argument("--runtime-controls")
    parser.add_argument("--output")
    parser.add_argument("--split", choices=("dev", "calibration", "sealed_test"), default="dev")
    parser.add_argument("--allow-sealed-test-after-freeze", action="store_true")
    args = parser.parse_args(argv)
    if args.split == "sealed_test" and not args.allow_sealed_test_after_freeze:
        parser.error("sealed test requires --allow-sealed-test-after-freeze")
    model = load_text_control_reward(args.model)
    if args.runtime_instructions or args.runtime_controls:
        if not args.runtime_instructions or not args.runtime_controls:
            parser.error("both runtime specification paths are required")
        from carryrl.rewards.control_text_runtime import validate_runtime_text_controls

        report = validate_runtime_text_controls(
            model, instructions=args.runtime_instructions, controls=args.runtime_controls
        )
        report["model"] = model.identity
        payload = json.dumps(report, sort_keys=True)
        if args.output:
            from pathlib import Path

            Path(args.output).write_text(payload + "\n")
        print(payload)
        return 0 if report["passed"] else 1
    if not args.dataset:
        parser.error("--dataset or both runtime specification paths are required")
    dataset, records = read_control_dataset(args.dataset)
    report = validate_text_control_predictions(
        model, [row for row in records if row["split"] == args.split]
    )
    result = {
        "model": model.identity,
        "split": args.split,
        "dataset_sha256": dataset["records_sha256"],
        "frozen_model_loaded": True,
        "sealed_test_after_freeze_attested": args.split == "sealed_test"
        and args.allow_sealed_test_after_freeze,
        **report,
    }
    payload = json.dumps(result, sort_keys=True)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(payload + "\n")
    print(payload)
    return 0 if report["gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
