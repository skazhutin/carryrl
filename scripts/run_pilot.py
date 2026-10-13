"""Record a finite pilot protocol before sequentially launching its seed runs."""

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from carryrl.evaluation.plot import pilot_statistics, plot_pilot
from carryrl.training.config import RunConfig
from carryrl.training.run import atomic_json, provenance, train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    config = RunConfig.load(args.config)
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be unique nonnegative integers")
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "protocol.json", {
        "status": "registered_before_runs", "registered_utc": datetime.now(UTC).isoformat(),
        "seeds": args.seeds, "config": config.to_dict(), "provenance": provenance(),
        "series_kind": "development pilot, not final comparison",
        "final_test_split_used": False, "retries": 0,
        "resource_stopped_or_failed_seeds": "retained; not replaced by successful seeds",
        "execution": "one training run at a time",
    })
    results = []
    for seed in args.seeds:
        try:
            result = train(replace(config, seed=seed), args.output / f"seed_{seed}")
        except Exception as error:
            results.append({"seed": seed, "status": "failed", "error_type": type(error).__name__,
                            "error": str(error)})
            atomic_json(args.output / "series.json", {"status": "failed", "runs": results})
            raise
        results.append({"seed": seed, **result})
        atomic_json(args.output / "series.json", {"status": "running", "runs": results})
        print(json.dumps({"seed": seed, "status": result["status"],
                          "timesteps": result["timesteps"],
                          "success": result["final_success_rate_lower"],
                          "wall_seconds": result["wall_seconds"]}), flush=True)
    status = "completed" if all(run["status"] == "completed" for run in results) else "incomplete"
    plot_pilot(args.output)
    report = {"status": status, "runs": results, "statistics": pilot_statistics(args.output)}
    atomic_json(args.output / "series.json", report)
    print(json.dumps(report, indent=2))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
