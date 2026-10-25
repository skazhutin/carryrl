"""Local experiment commands. Evaluation is separate from training."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from stable_baselines3 import PPO

from carryrl.evaluation.runner import evaluate
from carryrl.training.config import RunConfig
from carryrl.training.run import atomic_json, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="carryrl")
    commands = parser.add_subparsers(dest="command", required=True)
    trainer = commands.add_parser("train", help="run one bounded PPO engineering experiment")
    trainer.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    trainer.add_argument("--output", type=Path, required=True)
    trainer.add_argument("--seed", type=int)
    trainer.add_argument("--device", choices=("cpu", "mps"))
    trainer.add_argument("--resume", type=Path, help="checkpoint directory; starts a new episode")
    for command in ("evaluate", "random"):
        evaluator = commands.add_parser(command)
        evaluator.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
        evaluator.add_argument("--output", type=Path, required=True)
        evaluator.add_argument("--split", choices=("validation", "test"), default="validation")
        if command == "evaluate":
            evaluator.add_argument("--checkpoint", type=Path, required=True)
    sequence_train = commands.add_parser("train-v1", help="complete-episode generalist PPO probe")
    sequence_train.add_argument("--config", type=Path, default=Path("configs/agent_v1_smoke.json"))
    sequence_train.add_argument("--output", type=Path, required=True)
    sequence_train.add_argument("--seed", type=int)
    sequence_train.add_argument("--device", choices=("cpu", "mps"))
    sequence_train.add_argument(
        "--training-backend", choices=("torch", "mlx"),
        help="policy optimizer backend; MLX uses Metal with frozen Torch encoders on CPU",
    )
    sequence_start = sequence_train.add_mutually_exclusive_group()
    sequence_start.add_argument("--resume", type=Path)
    sequence_start.add_argument(
        "--warm-start", type=Path,
        help="start an explicit curriculum stage from saved policy and reward weights",
    )
    sequence_train.add_argument("--reward-checkpoint", type=Path,
                                help="previously validated screen-v2 reward checkpoint")
    sequence_train.add_argument(
        "--instructions", type=Path,
        help="JSON mapping of games to user instructions; enables the multilingual text input",
    )
    sequence_eval = commands.add_parser("evaluate-v1", help="frozen generalist evaluation")
    sequence_eval.add_argument("--config", type=Path, default=Path("configs/agent_v1_smoke.json"))
    sequence_eval.add_argument("--output", type=Path, required=True)
    sequence_eval.add_argument("--checkpoint", type=Path, required=True)
    sequence_eval.add_argument("--device", choices=("cpu", "mps"))
    sequence_eval.add_argument("--split", choices=("validation", "test"), default="validation")
    sequence_eval.add_argument("--difficulty", type=float,
                               help="explicit evaluation difficulty; 1 uses original tasks")
    sequence_eval.add_argument("--all-levels", action="store_true",
                               help="cover every level in the selected held-out bank")
    sequence_eval.add_argument("--episodes-per-task", type=int)
    sequence_eval.add_argument("--eval-seed", type=int)
    sequence_eval.add_argument(
        "--instructions", type=Path,
        help="explicit instruction variants for an instruction-conditioned checkpoint",
    )
    reward_train = commands.add_parser(
        "train-reward", help="train and validate screen reward first"
    )
    reward_train.add_argument("--config", type=Path, required=True)
    reward_train.add_argument("--output", type=Path, required=True)
    reward_train.add_argument("--seed", type=int)
    reward_train.add_argument("--device", choices=("cpu", "mps"))
    args = parser.parse_args(argv)
    if args.command == "train-reward":
        return _main_reward(args)
    if args.command in {"train-v1", "evaluate-v1"}:
        return _main_v1(args)
    config = RunConfig.load(args.config)
    if args.command == "train":
        if args.seed is not None:
            config = replace(config, seed=args.seed)
        if args.device is not None:
            config = replace(config, device=args.device)
        result = train(config, args.output, args.resume)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output / "metadata.json", {
            "command": args.command, "config": config.to_dict(), "split": args.split,
            "checkpoint": str(args.checkpoint) if args.command == "evaluate" else None,
            "policy_inputs": "RGB frame only", "policy_updates": False,
        })
        try:
            torch.set_num_threads(config.torch_threads)
            if config.device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available")
            model = None
            if args.command == "evaluate":
                checkpoint = args.checkpoint
                archive = checkpoint / "model.zip" if checkpoint.is_dir() else checkpoint
                model = PPO.load(archive, device=config.device)
            result = evaluate(
                model, seeds=config.eval_seeds, split=args.split,
                image_size=config.image_size, max_episode_steps=config.max_episode_steps,
                gif_path=args.output / "fixed_seed.gif",
            )
            atomic_json(args.output / "evaluation.json", result)
            atomic_json(args.output / "status.json", {"status": "completed"})
        except BaseException as error:
            atomic_json(args.output / "status.json", {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error_type": type(error).__name__, "error": str(error),
                "partial_evaluation_included_in_summary": False,
            })
            raise
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"},
                     indent=2, allow_nan=False))
    return 0 if result.get("status", "completed") == "completed" else 1


def _main_reward(args) -> int:
    from carryrl.rewards.pretrain import pretrain_screen_reward
    from carryrl.training.sequence_config import SequenceRunConfig

    config = SequenceRunConfig.load(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.device is not None:
        config = replace(config, device=args.device)
    if config.reward_model_mode != "screen_v2":
        raise ValueError("train-reward requires reward_model_mode=screen_v2")
    torch.set_num_threads(config.torch_threads)
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no implicit device fallback")
    _, report = pretrain_screen_reward(
        args.output, tasks=config.tasks,
        episodes_per_task=config.reward_bootstrap_episodes_per_task,
        validation_episodes_per_task=config.reward_validation_episodes_per_task,
        epochs=config.reward_epochs, seed=config.seed, device=config.device,
        image_size=config.env_image_size, max_steps=config.max_episode_steps,
        presentation_config=config.visual_config,
        level_bank_seed=config.level_bank_seed,
    )
    print(json.dumps({"model_version": report["model_version"],
                      "tasks": report["tasks"], "gate": report["gate"],
                      "checkpoint": str(args.output / "reward_checkpoint.pt"),
                      "report": str(args.output / "reward_report.json")},
                     indent=2, allow_nan=False))
    return 0 if report.get("gate", {}).get("passed", False) else 1


def _main_v1(args) -> int:
    from carryrl.training.sequence import (
        evaluate_sequence,
        load_sequence_checkpoint,
        train_sequence,
    )
    from carryrl.training.sequence_config import SequenceRunConfig

    config = SequenceRunConfig.load(args.config)
    if args.device is not None:
        config = replace(config, device=args.device)
    if args.instructions is not None:
        instructions = json.loads(args.instructions.read_text(encoding="utf-8"))
        if args.command == "evaluate-v1" and config.agent.text_encoder == "none":
            raise ValueError("instructions require a checkpoint trained with a text input")
        config = replace(
            config,
            agent=replace(config.agent, text_encoder="multilingual_e5_small"),
            episode_instructions=instructions,
        )
    if args.command == "train-v1":
        if args.training_backend is not None:
            config = replace(config, training_backend=args.training_backend)
        if args.seed is not None:
            config = replace(config, seed=args.seed)
        if args.reward_checkpoint is not None:
            config = replace(config, reward_checkpoint=str(args.reward_checkpoint.resolve()))
        start_options = {"warm_start": args.warm_start} if args.warm_start is not None else {}
        result = train_sequence(config, args.output, args.resume, **start_options)
    else:
        if args.difficulty is not None:
            config = replace(config, curriculum_difficulty=args.difficulty,
                             evaluation_difficulty=args.difficulty)
        if args.all_levels:
            config = replace(config, eval_all_levels=True)
        if args.episodes_per_task is not None:
            config = replace(config, eval_episodes_per_task=args.episodes_per_task)
        if args.eval_seed is not None:
            config = replace(config, eval_seed=args.eval_seed)
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output / "metadata.json", {
            "config": config.to_dict(), "checkpoint": str(args.checkpoint),
            "split": args.split, "updates": 0,
        })
        try:
            torch.set_num_threads(config.torch_threads)
            if config.device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS unavailable; no implicit device fallback")
            agent, _, reward, progress = load_sequence_checkpoint(
                args.checkpoint, config, evaluation_only=True)
            result = evaluate_sequence(agent, reward, config, split=args.split)
            result["checkpoint_weights_revision"] = progress["weights_revision"]
            atomic_json(args.output / "evaluation.json", result)
            atomic_json(args.output / "status.json", {"status": "completed"})
        except BaseException as error:
            atomic_json(args.output / "status.json", {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "error_type": type(error).__name__, "error": str(error),
            })
            raise
    print(json.dumps({key: value for key, value in result.items() if key != "records"},
                     indent=2, allow_nan=False))
    return 0 if result.get("status", "completed") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
