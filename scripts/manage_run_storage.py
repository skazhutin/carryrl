#!/usr/bin/env python3
"""Compatibility CLI for reversible checkpoint storage."""

import sys
from pathlib import Path

# Keep the original standalone, standard-library-only recovery command usable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from carryrl.training.checkpoint_storage import *  # noqa: E402,F403
from carryrl.training.checkpoint_storage import main  # noqa: E402

if __name__ == "__main__":
    main()
