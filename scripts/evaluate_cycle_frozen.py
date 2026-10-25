"""Explicit frozen endpoint evaluation after the registered validation-only choice."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from carryrl.training.run import atomic_json, file_sha256
from carryrl.training.sequence import evaluate_sequence, load_sequence_checkpoint, tensor_hash
from carryrl.training.sequence_config import SequenceRunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[80001, 80002, 80003])
    args = parser.parse_args()
    state = json.loads((args.plan / "cycle_state.json").read_text())
    decision = state["decisions"]
    selected = Path(decision["checkpoint"]).resolve()
    if (decision.get("model_frozen") is not True
            or file_sha256(selected / "manifest.json")
            != decision["checkpoint_manifest_sha256"]):
        raise ValueError("endpoint evaluation requires an authenticated frozen selection")
    checkpoint = args.checkpoint.resolve(strict=True)
    if args.split == "test" and (
        checkpoint != selected or decision.get("test_allowed") is not True
    ):
        raise ValueError("only the explicitly frozen selected model may open the game test")
    protocol = json.loads((args.plan / "protocol.json").read_text())
    if not args.seeds or any(seed not in protocol["action_seeds"] for seed in args.seeds):
        raise ValueError("endpoint action seeds must be registered before training")
    source = SequenceRunConfig.load(checkpoint.parent.parent / "config.json")
    instruction_path = args.plan / "source_final/configs/atomic_instructions.json"
    instructions = json.loads(instruction_path.read_text())
    config = replace(source, device="cpu", level_bank_seed=protocol["bank"],
                     episode_instructions=instructions, eval_all_levels=True,
                     eval_episodes_per_task=16, eval_stochastic=True,
                     evaluation_difficulty=1.0, evaluation_canonical=True,
                     evaluation_pixel_art=True)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    agent, _, reward, saved = load_sequence_checkpoint(
        checkpoint, config, evaluation_only=True, load_optimizer=False)
    agent.eval().requires_grad_(False)
    reward.eval().requires_grad_(False)
    before = tensor_hash(agent)
    reward_before = tensor_hash(reward)
    evaluations = []
    for seed in args.seeds:
        seeded = replace(config, eval_seed=seed)
        for style in ("canonical", "pixel_art"):
            current = seeded if style == "canonical" else replace(
                seeded, evaluation_canonical=False, evaluation_pixel_art=False)
            report = evaluate_sequence(agent, reward, current, split=args.split)
            report.update(checkpoint=str(checkpoint),
                          checkpoint_weights_revision=saved["weights_revision"],
                          endpoint_style=style, endpoint_action_seed=seed,
                          selected_manifest_sha256=decision["checkpoint_manifest_sha256"])
            name = f"{args.split}_{style}_{seed}.json"
            atomic_json(args.output / name, report)
            evaluations.append({"path": str(args.output / name),
                                "sha256": file_sha256(args.output / name)})
    if tensor_hash(agent) != before or tensor_hash(reward) != reward_before:
        raise RuntimeError("endpoint evaluation changed policy or reward weights")
    atomic_json(args.output / "status.json", {
        "status": "completed", "split": args.split, "checkpoint": str(checkpoint),
        "agent_sha256": before, "reward_sha256": reward_before,
        "selected_checkpoint": str(selected), "training_updates": 0,
        "evaluations": evaluations, "action_seeds": args.seeds,
    })
    print(json.dumps({"status": "completed", "evaluations": len(evaluations),
                      "checkpoint": str(checkpoint)}))


if __name__ == "__main__":
    main()
