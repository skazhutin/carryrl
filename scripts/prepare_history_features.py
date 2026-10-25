"""Extract frozen masked-scene/explicit-goal features without fitting any model."""

import argparse
import gc
import time
from dataclasses import replace
from pathlib import Path

import torch

from carryrl.rewards.history_features import load_history_features, prepare_history_features
from carryrl.rewards.history_suite_data import load_history_suite
from carryrl.training.run import atomic_json, file_sha256, provenance
from carryrl.training.sequence import load_sequence_checkpoint
from carryrl.training.sequence_config import SequenceRunConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-batch-size", type=int, default=4, choices=range(1, 33))
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("history feature output already exists")
    output = args.output.resolve()
    checkpoint, corpus_path = args.checkpoint.resolve(strict=True), args.corpus.resolve(strict=True)
    source = next((parent.parent for parent in checkpoint.parents
                   if parent.name in {"checkpoints", "checkpoints_by_epoch"}), checkpoint)
    if any(output == path or path in output.parents or output in path.parents
           for path in (source, corpus_path)):
        raise ValueError("feature output must be outside its source run and corpus")
    config = replace(SequenceRunConfig.load(args.config), device="cpu")
    output.mkdir(parents=True)
    paths = [path for path in sorted(checkpoint.rglob("*")) if path.is_file()]
    before = {str(path.relative_to(checkpoint)): file_sha256(path) for path in paths}
    atomic_json(output / "metadata.json", {**provenance(), "command": "prepare_history_features",
        "checkpoint": str(checkpoint), "corpus_sha256": args.corpus_sha256,
        "checkpoint_files_sha256_before": before, "updates": 0, "model_fit": False})
    threads, started = torch.get_num_threads(), time.monotonic()
    try:
        atomic_json(output / "status.json", {"status": "loading_corpus", "updates": 0})
        corpus = load_history_suite(corpus_path, expected_sha256=args.corpus_sha256)
        torch.set_num_threads(config.torch_threads)
        agent, optimizer, reward, state = load_sequence_checkpoint(
            checkpoint, config, evaluation_only=True, load_optimizer=False)
        if optimizer is not None or agent.language is None:
            raise ValueError("frozen feature preparation requires explicit goals and no optimizer")
        vision, language, revision = agent.vision, agent.language, state["weights_revision"]
        del agent, optimizer, reward, state
        gc.collect()

        def progress(event):
            if event["frames_encoded"] % 64 == 0 or event["frames_encoded"] == event["frames"]:
                atomic_json(output / "status.json", {"status": "encoding", **event})
                print(f"masked RGB: {event['frames_encoded']}/{event['frames']}", flush=True)

        report = prepare_history_features(corpus, output / "features", vision_encoder=vision,
            goal_encoder=language, frame_batch_size=args.frame_batch_size, on_progress=progress)
        features = load_history_features(output / "features", corpus,
                                         expected_sha256=report["feature_sha256"])
        for index in range(len(corpus.episodes)):
            features.model_inputs(corpus, index)
        after = {str(path.relative_to(checkpoint)): file_sha256(path) for path in paths}
        if before != after:
            raise RuntimeError("source checkpoint files changed during frozen feature extraction")
        report.update(seconds=time.monotonic() - started, weights_revision=revision,
                      checkpoint_files_unchanged=True, loader_verified=True,
                      episodes_checked=len(corpus.episodes))
        atomic_json(output / "audit.json", report)
        atomic_json(output / "status.json", {"status": "completed", "updates": 0,
            "feature_sha256": report["feature_sha256"], "model_fit": False,
            "ppo_eligible": False, "frames": report["frames"]})
        return 0
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "interrupted"
            if isinstance(error, KeyboardInterrupt) else "failed", "updates": 0,
            "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        torch.set_num_threads(threads)


if __name__ == "__main__":
    raise SystemExit(main())
