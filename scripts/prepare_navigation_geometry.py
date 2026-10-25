"""Prepare equal-budget navigation geometry delta arms without allocating a model."""

from carryrl.training.runtime_diversity import preparation_main


def main(argv=None):
    return preparation_main(argv, legacy_navigation=True)


if __name__ == "__main__":
    raise SystemExit(main())
