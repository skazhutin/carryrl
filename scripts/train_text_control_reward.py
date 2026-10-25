"""Explicit future CPU head fit; importing this module performs no training."""

import argparse
import json
import math
import os
import signal
import time

from carryrl.agent.language import MultilingualInstructionEncoder
from carryrl.rewards.control_text_training import train_text_control_artifact


def bounded_fit_seconds(requested, *, now=None, environ=None):
    """Keep two minutes for failure/partial artifacts before parent cutoffs."""
    now = time.time() if now is None else now
    environ = os.environ if environ is None else environ
    if not math.isfinite(requested) or not 0 < requested <= 10800:
        raise ValueError("text fit budget must be positive and at most three hours")
    limits = [requested]
    for name in ("CARRYRL_STAGE_MAX_SECONDS", "CARRYRL_CYCLE_DEADLINE_UNIX"):
        if name not in environ:
            continue
        value = float(environ[name])
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        limits.append(value - (now if name.endswith("DEADLINE_UNIX") else 0) - 120)
    budget = min(limits)
    if budget <= 0:
        raise TimeoutError("no positive text fit budget remains before the parent cutoff")
    return budget


def _terminate(signum, frame):
    raise TimeoutError(f"text fit interrupted by signal {signum}; preserve partial artifacts")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--max-train-groups", type=int)
    parser.add_argument("--max-train-texts", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--pair-learning-rate", type=float)
    parser.add_argument("--readout-dropout", type=float, default=0.0)
    parser.add_argument("--cross-modal-train-repeats", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--selection-protocol",
                        choices=("dev_nll", "development_feasibility_v1"))
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--allow-model-download", action="store_true")
    feature_options = parser.add_mutually_exclusive_group()
    feature_options.add_argument("--feature-cache")
    feature_options.add_argument("--reuse-feature-cache")
    parser.add_argument("--fit-workspace")
    parser.add_argument("--calibration-previously-inspected", action="store_true")
    parser.add_argument(
        "--head-variant",
        choices=("legacy_v1", "residual_balanced_v2", "grounded_balanced_v3",
                 "local_grounded_balanced_v4", "pair_geometry_balanced_v5",
                 "supported_conditional_v6"),
        default="legacy_v1",
    )
    parser.add_argument("--diagnostic-update", type=int, action="append", default=[])
    parser.add_argument("--max-feature-gib", type=float, default=8)
    parser.add_argument("--max-seconds", type=float, default=10800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--runtime-instructions", default="configs/atomic_instructions.json")
    parser.add_argument("--runtime-controls", default="configs/atomic_controls.json")
    args = parser.parse_args(argv)
    import torch

    selection_protocol = args.selection_protocol or (
        "development_feasibility_v1" if args.probe_only else "dev_nll"
    )
    if args.cross_modal_train_repeats != 1 and (
        args.head_variant != "supported_conditional_v6"
        or selection_protocol != "development_feasibility_v1"
    ):
        parser.error("cross-modal repeats require the conditional head and feasibility selection")
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be in 1..16")
    fit_deadline = time.time() + bounded_fit_seconds(args.max_seconds)
    previous_term = signal.signal(signal.SIGTERM, _terminate)
    try:
        torch.set_num_threads(args.threads)
        encoder = MultilingualInstructionEncoder(
            max_tokens=512, local_files_only=not args.allow_model_download
        )
        encoder.to(args.device)
        remaining = fit_deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("text fit budget exhausted during encoder loading")
        result = train_text_control_artifact(
            args.dataset,
            args.output,
            encoder=encoder,
            seed=args.seed,
            epochs=args.epochs if args.epochs is not None else 8 if args.probe_only else 10,
            probe_only=args.probe_only,
            max_train_groups=args.max_train_groups,
            max_train_texts=args.max_train_texts,
            learning_rate=args.learning_rate, pair_learning_rate=args.pair_learning_rate,
            readout_dropout=args.readout_dropout,
            cross_modal_train_repeats=args.cross_modal_train_repeats,
            early_stopping_patience=args.early_stopping_patience,
            selection_protocol=selection_protocol,
            feature_cache=args.feature_cache,
            reuse_feature_cache=args.reuse_feature_cache,
            fit_workspace=args.fit_workspace,
            head_variant=args.head_variant,
            diagnostic_updates=args.diagnostic_update,
            calibration_previously_inspected=args.calibration_previously_inspected,
            max_feature_bytes=int(args.max_feature_gib * 1024**3),
            max_seconds=remaining,
            batch_size=args.batch_size,
            device=args.device,
            runtime_instructions=args.runtime_instructions,
            runtime_controls=args.runtime_controls,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
