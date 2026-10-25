"""Frozen validation-only paired RGB/text/history likelihood diagnostics."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from carryrl.evaluation.input_sensitivity import (
    evaluate_input_sensitivity,
    load_sensitivity_bundle,
    prepare_mouse_position_bundle,
)
from carryrl.training.run import atomic_json, file_sha256
from carryrl.training.sequence import load_sequence_checkpoint
from carryrl.training.sequence_config import SequenceRunConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--prepare-mouse-pairs",
        type=Path,
        help="Prepare one bounded validation scene pair without loading a model",
    )
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--level-bank-seed", type=int, default=1729)
    args = parser.parse_args(argv)
    if args.prepare_mouse_pairs is not None:
        if any(
            value is not None for value in (args.checkpoint, args.config, args.pairs, args.output)
        ):
            parser.error("pair preparation is separate from frozen checkpoint evaluation")
        path = prepare_mouse_position_bundle(
            args.prepare_mouse_pairs,
            image_size=args.image_size,
            seed=args.seed,
            level_bank_seed=args.level_bank_seed,
        )
        print(
            json.dumps(
                {"prepared": str(path.resolve()), "model_loaded": False, "training_updates": 0}
            )
        )
        return
    if any(value is None for value in (args.checkpoint, args.config, args.pairs, args.output)):
        parser.error("evaluation requires --checkpoint --config --pairs --output")
    pairs, bundle_identity = load_sensitivity_bundle(args.pairs)
    config = replace(SequenceRunConfig.load(args.config), device="cpu")
    if config.screen_text_language is not None:
        parser.error("paired RGB/text/history diagnostics require screen OCR disabled")
    checkpoint = args.checkpoint.resolve(strict=True)
    if args.output.resolve().is_relative_to(checkpoint):
        parser.error("diagnostic output must be outside the immutable checkpoint")
    before = {
        str(path.relative_to(checkpoint)): file_sha256(path)
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.torch_threads)
        agent, _, _, metadata = load_sequence_checkpoint(
            checkpoint, config, evaluation_only=True, load_optimizer=False
        )
        report = evaluate_input_sensitivity(agent, pairs)
    finally:
        torch.set_num_threads(threads)
        after = {
            str(path.relative_to(checkpoint)): file_sha256(path)
            for path in checkpoint.rglob("*")
            if path.is_file()
        }
        if after != before:
            raise RuntimeError("input sensitivity diagnostic modified checkpoint files")
    report.update(
        checkpoint=str(checkpoint),
        checkpoint_sha256=before,
        weights_revision=metadata["weights_revision"],
        bundle=bundle_identity,
    )
    atomic_json(args.output, report)
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
