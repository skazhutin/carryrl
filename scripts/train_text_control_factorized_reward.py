"""Explicit bounded fresh factorized-control head probe or gated full training."""

import argparse
import json
import signal
import time

from train_text_control_reward import _terminate, bounded_fit_seconds

from carryrl.agent.language import MultilingualInstructionEncoder
from carryrl.rewards.control_text_factorized_training import (
    LOSS_PROTOCOL,
    STAGED_TRAINING_PROTOCOL,
    train_factorized_text_control_artifact,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--passed-probe")
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--training-protocol",
        choices=(LOSS_PROTOCOL, STAGED_TRAINING_PROTOCOL),
        default=LOSS_PROTOCOL,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--max-feature-gib", type=float, default=8)
    parser.add_argument("--allow-model-download", action="store_true")
    features = parser.add_mutually_exclusive_group()
    features.add_argument("--feature-cache")
    features.add_argument("--reuse-feature-cache")
    parser.add_argument("--runtime-instructions", default="configs/atomic_instructions.json")
    parser.add_argument("--runtime-controls", default="configs/atomic_controls.json")
    parser.add_argument("--defer-sealed-test", action="store_true")
    args = parser.parse_args(argv)
    staged = args.training_protocol == STAGED_TRAINING_PROTOCOL
    epochs = args.epochs if args.epochs is not None else (8 if args.probe_only or staged else 10)
    if staged and (
        epochs != 8 or args.batch_size != 32 or args.learning_rate != 0.0003 or args.device != "cpu"
    ):
        parser.error("staged protocol requires --epochs8, --batch-size32, LR.0003 and CPU")
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be in 1..16")
    if not args.probe_only and not args.passed_probe:
        parser.error(
            "full training requires --passed-probe from a successful four/eight-epoch probe"
        )
    max_seconds = args.max_seconds or (4200 if args.probe_only else 7200)
    if max_seconds > (4200 if args.probe_only else 7200):
        parser.error("wall-clock budget exceeds factorized protocol")
    deadline = time.time() + bounded_fit_seconds(max_seconds)
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        import torch

        torch.set_num_threads(args.threads)
        encoder = MultilingualInstructionEncoder(
            max_tokens=512, local_files_only=not args.allow_model_download
        )
        encoder.to(args.device)
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("factorized fit budget exhausted during encoder loading")
        result = train_factorized_text_control_artifact(
            args.dataset,
            args.output,
            encoder=encoder,
            probe_only=args.probe_only,
            passed_probe=args.passed_probe,
            epochs=epochs,
            seed=args.seed
            if args.seed is not None
            else (20261029 if args.probe_only else 20261030),
            feature_cache=args.feature_cache,
            reuse_feature_cache=args.reuse_feature_cache,
            max_feature_bytes=int(args.max_feature_gib * 1024**3),
            max_seconds=remaining,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            runtime_instructions=args.runtime_instructions,
            runtime_controls=args.runtime_controls,
            evaluate_sealed=not args.defer_sealed_test,
            training_protocol=args.training_protocol,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "ready",
                    "feasibility",
                    "runtime",
                    "admission_gates_passed",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
