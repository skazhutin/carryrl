"""Audit the exact saved training RGB corpus without loading a model or training."""

import argparse
import json
from pathlib import Path

from carryrl.training.diversity import audit_diversity
from carryrl.training.imitation import SavedDemonstrationReuse
from carryrl.training.run import atomic_json
from carryrl.training.sequence_config import SequenceRunConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        help="default: source run's config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-budget-mib", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.frame_budget_mib <= 0:
        parser.error("--frame-budget-mib must be positive")
    source, output = args.source_run.resolve(strict=True), args.output.resolve()
    if output == source or source in output.parents:
        parser.error("audit output must be separate from the source run")
    config = SequenceRunConfig.load(args.config or source / "config.json")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "status.json", {"status": "validating", "updates": 0})
    try:
        reuse = SavedDemonstrationReuse.prepare(source, config=config,
            max_frame_bytes=args.frame_budget_mib * 1024**2)
        report = audit_diversity(reuse.dataset, corpus_identity=reuse.summary,
                                 source_config=config)
        reuse.assert_unchanged()
        atomic_json(output / "audit.json", report)
        atomic_json(output / "status.json", {"status": "completed", "updates": 0,
                                             "test_used": False})
        print(json.dumps({"status": "completed", **report["counts"]}))
        return 0
    except BaseException as error:
        atomic_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "updates": 0, "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
