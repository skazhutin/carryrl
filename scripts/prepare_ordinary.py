"""Prepare fixed O1/O2 ordinary RGB corpora and frozen features without training."""

import argparse
from pathlib import Path

from carryrl.training.ordinary_preparation import (
    load_ordinary_preparation,
    prepare_ordinary,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", type=Path, help="recheck a prepared manifest's pins")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--experiment", choices=("O1", "O2"), default="O1")
    parser.add_argument("--canonical-anchor-coefficient", type=float, default=0.)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--source-checkpoint-sha256")
    parser.add_argument("--expected-actor-sha256")
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--replacement-bundle", type=Path)
    parser.add_argument("--replacement-bundle-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--control-output", type=Path)
    parser.add_argument("--treatment-output", type=Path)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--feature-reuse-run", type=Path)
    parser.add_argument("--frame-budget-mib", type=int, default=1536)
    parser.add_argument("--feature-budget-mib", type=int, default=14336)
    args = parser.parse_args(argv)
    if args.verify_only:
        plan = load_ordinary_preparation(args.verify_only, expected_sha256=args.manifest_sha256)
    else:
        required = ("source_checkpoint", "source_checkpoint_sha256", "replacement_bundle",
                    "replacement_bundle_sha256", "output", "feature_cache_dir")
        for name in required:
            if getattr(args, name) is None:
                parser.error("preparation requires --" + name.replace("_", "-"))
        options = vars(args).copy()
        options.pop("verify_only")
        options.pop("manifest_sha256")
        options["expected_checkpoint_sha256"] = options.pop("source_checkpoint_sha256")
        options["expected_bundle_sha256"] = options.pop("replacement_bundle_sha256")

        def progress(row):
            if row.get("episodes_completed", 0) % 64 == 0:
                print(f"collected {row['task']}: {row['episodes_completed']} train episodes",
                      flush=True)

        plan = prepare_ordinary(**options, on_episode=progress)
    print(f"{plan['status']}: 0 updates executed; "
          f"{plan['expected_optimizer_updates']} planned updates per arm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
