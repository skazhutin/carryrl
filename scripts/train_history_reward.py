"""Explicit future fit of the auxiliary history head; no policy or PPO training."""

import argparse
from pathlib import Path

from carryrl.rewards.history_features import load_history_features
from carryrl.rewards.history_suite_data import load_history_suite
from carryrl.training.history_reward import fit_history_reward
from carryrl.training.run import atomic_json, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--max-updates", type=int, default=100_000)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("history fit output already exists")
    # The fit function checks loaded source directories again before creating output.
    output = args.output.resolve()
    for source in (args.corpus.resolve(strict=True), args.features.resolve(strict=True)):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("history fit output must be outside source corpus and features")
    sources = provenance()
    corpus = load_history_suite(args.corpus, expected_sha256=args.corpus_sha256)
    features = load_history_features(args.features, corpus, expected_sha256=args.features_sha256)
    fit_history_reward(corpus, features, output, epochs=args.epochs,
        learning_rate=args.learning_rate, seed=args.seed, hidden_size=args.hidden_size,
        max_wall_seconds=args.max_wall_seconds, max_updates=args.max_updates)
    atomic_json(output / "metadata.json", {**sources, "command": "train_history_reward",
        "corpus": str(args.corpus.resolve()), "corpus_sha256": args.corpus_sha256,
        "features": str(args.features.resolve()), "features_sha256": args.features_sha256,
        "policy_updates": 0, "ppo_eligible": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
