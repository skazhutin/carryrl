"""Prepare mandatory TRAIN-only task curriculum admission (does not train)."""

import argparse
import hashlib
import json
from pathlib import Path

from carryrl.training.task_curriculum import prepare_task_curriculum


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level-bank-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=96)
    args = parser.parse_args(argv)
    manifest = prepare_task_curriculum(args.level_bank_seed, image_size=args.image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    args.output.write_bytes(raw)
    print(json.dumps({"ready": manifest["ready"], "path": str(args.output),
                      "sha256": hashlib.sha256(raw).hexdigest(), "errors": manifest["errors"]}))
    return 0 if manifest["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
