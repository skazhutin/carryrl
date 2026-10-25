"""Prepare repeat/varied train-only runtime diversity deltas without allocating a model."""

from carryrl.training.runtime_diversity import preparation_main


def main(argv=None):
    return preparation_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
