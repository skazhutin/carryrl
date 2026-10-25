"""Concrete preparation/audit steps; importing this module never trains a model."""

import argparse
import json
from pathlib import Path

from carryrl.training.cycle_artifacts import (
    audit_probe,
    clean_successful_text_features,
    final_cycle_result,
    frozen_candidates,
    prepare_full_config,
    prepare_profiles,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("profiles", "probe", "full_config", "freeze",
                                         "clean_features", "result"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--curriculum-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visual-validation", type=Path)
    parser.add_argument("--text-reward", type=Path)
    args = parser.parse_args()
    if args.kind == "profiles":
        result = prepare_profiles(args.plan, visual_validation=args.visual_validation,
                                  text_reward=args.text_reward)
    elif args.kind == "probe":
        result = audit_probe(args.curriculum_run, args.run, args.output)
    elif args.kind == "full_config":
        result = prepare_full_config(args.plan, args.output)
    elif args.kind == "clean_features":
        result = clean_successful_text_features(args.plan)
    elif args.kind == "result":
        result = final_cycle_result(args.plan, args.run, args.output)
    else:
        result = frozen_candidates(args.run, args.output)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"candidates", "runs", "profiles"}}))


if __name__ == "__main__":
    main()
