"""Preflight or explicitly execute one persistent fresh all16 24-hour cycle."""
import argparse
import json
from pathlib import Path

from carryrl.training.rl_cycle import (
    CycleRunner,
    CycleStopped,
    authorize_repair,
    initialize_budget,
    reclassify_repair,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--init-budget", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repair-plan", type=Path)
    parser.add_argument("--repair-reason")
    parser.add_argument("--repair-kind", choices=("infrastructure", "learning"), default="learning")
    parser.add_argument("--reclassify-repair", type=int)
    args = parser.parse_args(argv)
    if args.init_budget:
        state = initialize_budget(args.output, start=True)
        if args.plan is None and not args.execute:
            print(json.dumps(state))
            return 0
    if args.reclassify_repair is not None:
        if (args.plan is not None or args.repair_plan is not None
                or args.execute or args.init_budget):
            parser.error("reclassification is a standalone audited action")
        try:
            result = reclassify_repair(args.output, args.reclassify_repair, args.repair_reason)
        except (CycleStopped, ValueError, OSError) as error:
            print(json.dumps({"status": "stopped", "error": str(error)}))
            return 2
        print(json.dumps(result))
        return 0
    if args.plan is None and args.repair_plan is None:
        parser.error("--plan is required for preflight/execution")
    try:
        if args.repair_plan is not None:
            authorize_repair(args.output, args.repair_plan, args.repair_reason,
                             repair_kind=args.repair_kind)
            args.plan = args.repair_plan
        runner = CycleRunner(args.plan, args.output)
        result = runner.run() if args.execute else runner.preflight()
    except (CycleStopped, ValueError, OSError) as error:
        print(json.dumps({"status": "stopped", "error": str(error)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
