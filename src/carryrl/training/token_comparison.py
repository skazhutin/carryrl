"""Prepare a paired spatial-token experiment; preparation never runs PPO or BC.

The planning path intentionally uses only the standard library. Models and optimizer
classes are imported only by explicit initial-checkpoint materialization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

FORMAT = "carryrl.token_comparison.v1"
TOKENS = (16, 32, 64)
PRIMARY_SEED = 20261019
CONFIRMATION_SEEDS = (20261020, 20261021)
BASE_CONFIG = "configs/agent_v1_learning.json"
INSTRUCTIONS = "configs/atomic_instructions.json"
QUERY = "resampler.queries"


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}


def comparison_configs(base, *, instructions=None, seed=PRIMARY_SEED, learning_rate=1e-5):
    """Return isolated arms with equal episode budgets and held-out evaluations."""
    common = copy.deepcopy(base)
    if (
        type(learning_rate) not in (int, float)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    if common.get("reward_checkpoint") is not None:
        raise ValueError("comparison base config must not depend on a learned reward artifact")
    if instructions is not None:
        common["episode_instructions"] = copy.deepcopy(instructions)
    common["agent"].update(pretrained=True, text_encoder="multilingual_e5_small")
    common.update(
        device="cpu",
        training_backend="mlx",
        level_bank_seed=20261008,
        reward_checkpoint=None,
        reward_model_mode="screen_v2",
        seed=seed,
        total_rollouts=128,
        episodes_per_task=1,
        n_epochs=2,
        learning_rate=learning_rate,
        clip_range=0.1,
        target_kl=0.01,
        entropy_coef=0.001,
        value_coef=0.1,
        ppo_minibatch_episodes=4,
        ppo_task_balance=True,
        task_episode_multipliers={},
        curriculum_difficulty=1.0,
        evaluation_difficulty=1.0,
        curriculum_warmup_rollouts=0,
        motor_initialization="balanced",
        evaluate_initial=True,
        eval_every_rollouts=16,
        eval_all_levels=True,
        eval_episodes_per_task=16,
        evaluation_canonical=True,
        evaluation_pixel_art=True,
        eval_stochastic=False,
        screen_text_language=None,
        reset_optimizer_on_warm_start=False,
        reward_composition={
            "version": "carryrl.reward.outcome_potential_cost.v1",
            "parameters": {"goal_choice_utility": "completion_only"},
        },
        checkpoint_every_rollouts=16,
        checkpoint_keep_raw=2,
        minimum_free_disk_mib=8192,
        torch_threads=2,
        # A safety boundary, never a replacement for the episode budget.
        max_wall_seconds=10 * 3600.0,
    )
    common["visual_config"] = {
        "mode": "canonical",
        "pixel_art": True,
        "seed": 20261019,
        "style_split": None,
    }
    expected = {
        "width": 256,
        "layers": 4,
        "heads": 4,
        "memory_tokens": 8,
        "segment_length": 64,
        "text_encoder": "multilingual_e5_small",
    }
    if any(common["agent"].get(key) != value for key, value in expected.items()):
        raise ValueError("comparison requires the unchanged 256/4/4/8/64 E5 architecture")
    if (
        len(common["tasks"]) != 16
        or len(set(common["tasks"])) != 16
        or set(common["episode_instructions"]) != set(common["tasks"])
        or any(
            not rows or any(not isinstance(text, str) or not text.strip() for text in rows)
            for rows in common["episode_instructions"].values()
        )
    ):
        raise ValueError("comparison requires 16 distinct tasks with explicit instructions")
    arms = {}
    for tokens in TOKENS:
        cfg = copy.deepcopy(common)
        cfg["agent"]["spatial_tokens"] = tokens
        arms[tokens] = cfg
    return arms


def _argv(plan, tokens):
    return [
        "uv",
        "run",
        "--locked",
        "--extra",
        "mlx",
        "--extra",
        "monitor",
        "python",
        "scripts/run_token_comparison.py",
        "--plan",
        str(plan),
        "--tokens",
        str(tokens),
        "--execute",
    ]


def _source_paths(root):
    return sorted(
        [
            *root.glob("src/carryrl/**/*.py"),
            *root.glob("scripts/*.py"),
            root / "pyproject.toml",
            root / "uv.lock",
        ]
    )


def prepare_comparison(root, output, *, base_config=None, instructions=None, learning_rate=1e-5):
    """Write a fresh, model-free plan without reading any previous experiment."""
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    base_config = Path(base_config or root / BASE_CONFIG).resolve(strict=True)
    instructions = Path(instructions or root / INSTRUCTIONS).resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"comparison output already exists: {output}")
    configs = comparison_configs(
        _read(base_config), instructions=_read(instructions), learning_rate=learning_rate
    )
    source_pins = [_pin(path) for path in _source_paths(root)]
    input_pins = [_pin(base_config), _pin(instructions)]
    output.mkdir(parents=True)
    arms = []
    reward_checkpoint = output / "reward" / "reward_checkpoint.pt"
    for tokens, config in configs.items():
        config["reward_checkpoint"] = str(reward_checkpoint)
        config_path = output / "configs" / f"seed_{PRIMARY_SEED}" / f"t{tokens}.json"
        _write(config_path, config)
        initial = output / "initial" / f"seed_{PRIMARY_SEED}" / f"t{tokens}"
        checkpoint = initial / "checkpoints" / "rollout_0000"
        run = root / "runs" / f"{output.name}_s{PRIMARY_SEED}_t{tokens}"
        arms.append(
            {
                "spatial_tokens": tokens,
                "seed": PRIMARY_SEED,
                "config": _pin(config_path),
                "initial_checkpoint": str(checkpoint),
                "training_output": str(run),
                "training_executed": False,
                "future_argv": _argv(output, tokens),
            }
        )
    # The current reward fitter requires varied HUDs and separate theme banks.
    reward_config = copy.deepcopy(configs[16])
    reward_config.update(
        seed=20261022,
        reward_checkpoint=None,
        training_backend="torch",
        evaluation_pixel_art=False,
        visual_config={
            "mode": "varied",
            "pixel_art": False,
            "themes": ["slate", "sand", "forest", "violet"],
            "native_sizes": [64, 96, 128],
            "textures": True,
            "seed": 20261022,
        },
    )
    reward_config_path = output / "configs" / "fresh_reward.json"
    _write(reward_config_path, reward_config)
    recipe = {
        "format": "carryrl.fresh_reward_recipe.v1",
        "executed": False,
        "initialization": "fresh random ScreenVisualReward; no imported checkpoints",
        "training_data": "new supported HUD RGB and new train episodes; no old corpus",
        "config": _pin(reward_config_path),
        "training_argv": [
            "uv",
            "run",
            "--locked",
            "carryrl",
            "train-reward",
            "--config",
            str(reward_config_path),
            "--output",
            str(output / "reward"),
        ],
        "validation_status": "pending; the training command alone does not establish readiness",
        "bootstrap_prerequisite": "train-reward propagates the explicit level_bank_seed; "
        "the saved training report and independent validation must agree with that bank",
        "required_validation": [
            "standard reward held-out gate",
            "all 16 games' canonical HUDs",
            "all 16 games' pixel-art HUDs",
            "all 16 games' progress/outcomes",
        ],
        "required_validation_report": str(output / "reward" / "validation_report.json"),
        "validation_report_contract": {
            "format": "carryrl.fresh_reward_validation.v1",
            "passed": True,
            "tasks": configs[16]["tasks"],
            "hud_passed": True,
            "progress_passed": True,
            "initialization": "random",
            "imported_checkpoints": [],
            "old_corpora": [],
            "level_bank_seed": 20261008,
            "recipe_sha256": "SHA256 of reward_recipe.json",
            "checkpoint_sha256": "SHA256 of reward/reward_checkpoint.pt",
            "reward_report_sha256": "SHA256 of reward/reward_report.json",
        },
        "scope": "supervised reward recognizer only; no policy BC, teacher action input or OCR",
        "test_opened": False,
    }
    _write(output / "reward_recipe.json", recipe)
    for pin in source_pins:
        relative = Path(pin["path"]).relative_to(root)
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pin["path"], destination)
        if _sha(destination) != pin["sha256"]:
            raise ValueError(f"source changed while taking snapshot: {relative}")
    manifest = {
        "format": FORMAT,
        "project_root": str(root),
        "prepared": True,
        "modelinit_verified": False,
        "training_executed": False,
        "policy_updates": 0,
        "ready_to_train": False,
        "reward_setup": "pending_fresh_training_and_validation",
        "common_lr_calibration": "pending; calibrate one common schedule before comparison; "
        "no per-arm tuning",
        "hyperparameters_status": "proposed shared settings, not established optimum",
        "old_experiment_dependencies": [],
        "input_pins": input_pins,
        "source_pins": source_pins,
        "arms": arms,
        "recipe_pin": _pin(output / "reward_recipe.json"),
        "reward_config_pin": _pin(reward_config_path),
        "sampling_rng": "runner must isolate model-constructor RNG consumption from rollout RNG; "
        "raw CLI warm-start does not guarantee matched sampling seeds",
        "primary_seed": PRIMARY_SEED,
        "confirmation_seeds": [
            {"seed": seed, "materialized": False, "executed": False} for seed in CONFIRMATION_SEEDS
        ],
        "confirmation_rule": "optional only after all primary arms complete; report every seed; "
        "no automatic best-seed selection",
        "actor_initialization": "fresh shared 64-query master; equal common tensors; "
        "query prefixes",
        "frozen_initialization": "standard pinned pretrained DINOv2/E5; one newly trained reward "
        "shared by all arms; no previous project weights",
        "initial_optimizer": "fresh AdamW, no state entries; all progress counters zero",
        "pretraining_disclosure": "external pretrained DINOv2/E5 allowed; reward must be freshly "
        "trained; actor learns only with RL",
        "budget": {
            "rollouts_per_arm": 128,
            "episodes_per_rollout": 16,
            "environment_episodes_per_arm": 2048,
            "epochs": 2,
            "optimizer_steps": "may differ because target-KL early stops are enabled",
            "comparison_axis": "equal environment episodes, not equal wall time",
        },
        "evaluation": {
            "split": "validation",
            "test_opened": False,
            "episodes_per_presentation": 256,
            "presentations": ["canonical", "held_out_pixel_art"],
            "rollouts": [0, *range(16, 129, 16)],
            "selection": "fixed final rollout 128; per-game and aggregate success; "
            "report canonical and pixel separately; no reward/loss winner",
        },
        "stop_rule": "wall/resource/failed arms are incomplete; never declare a winner from them",
        "execution": "sequential arms; no automatic launches; preparation performs zero updates",
        "train_presentation": "pixel art only; canonical mechanics/layout; "
        "changing training styles",
        "configuration_validation": "model-free invariants; full parser checked at materialization",
        "verify_argv": [
            "uv",
            "run",
            "--locked",
            "python",
            "scripts/prepare_token_comparison.py",
            "--output",
            str(output),
            "--verify-plan",
        ],
    }
    _write(output / "manifest.json", manifest)
    verify_plan(output)
    return manifest


def require_fresh_reward(output, manifest):
    """Reject absent, incomplete, or reused reward provenance before importing models."""
    output = Path(output).resolve()
    directory = output / "reward"
    paths = [
        directory / name
        for name in ("reward_checkpoint.pt", "reward_report.json", "validation_report.json")
    ]
    if any(not path.is_file() for path in paths):
        raise ValueError(
            "fresh reward is pending: train and independently validate the new "
            "bundle reward before materializing policy checkpoints"
        )
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError("fresh reward artifacts must remain inside their bundle")
    checkpoint, report_path, validation_path = paths
    report, validation = _read(report_path), _read(validation_path)
    reward_config = _read(manifest["reward_config_pin"]["path"])
    expected = {
        "format": "carryrl.fresh_reward_validation.v1",
        "passed": True,
        "tasks": _read(manifest["arms"][0]["config"]["path"])["tasks"],
        "hud_passed": True,
        "progress_passed": True,
        "initialization": "random",
        "imported_checkpoints": [],
        "old_corpora": [],
        "level_bank_seed": 20261008,
        "recipe_sha256": manifest["recipe_pin"]["sha256"],
        "checkpoint_sha256": _sha(checkpoint),
        "reward_report_sha256": _sha(report_path),
    }
    if (
        any(validation.get(key) != value for key, value in expected.items())
        or report.get("gate", {}).get("passed") is not True
        or report.get("policy_demonstrations") is not False
        or report.get("test_levels_used") is not False
        or report.get("seed") != reward_config["seed"]
        or report.get("tasks") != reward_config["tasks"]
        or report.get("level_bank_seed") != reward_config["level_bank_seed"]
        or report.get("initialization") != "random"
        or report.get("imported_checkpoints") != []
        or report.get("old_corpora") != []
    ):
        raise ValueError("fresh reward validation/provenance is incomplete or disagrees with pins")
    return [_pin(path) for path in paths]


def verify_plan(output):
    """Verify live and copied bytes before any explicit materialization/launch."""
    output = Path(output)
    manifest = _read(output / "manifest.json")
    if manifest.get("format") != FORMAT:
        raise ValueError("unsupported token comparison manifest")
    root = Path(manifest["project_root"])
    if {str(path.resolve()) for path in _source_paths(root)} != {
        pin["path"] for pin in manifest["source_pins"]
    }:
        raise ValueError("pinned source file set changed")
    for pin in [
        *manifest["source_pins"],
        *manifest["input_pins"],
        manifest["recipe_pin"],
        manifest["reward_config_pin"],
        *(arm["config"] for arm in manifest["arms"]),
    ]:
        if _pin(pin["path"]) != pin:
            raise ValueError(f"pinned file changed: {pin['path']}")
    for pin in manifest["source_pins"]:
        copied = output / "source_snapshot" / Path(pin["path"]).relative_to(root)
        if _sha(copied) != pin["sha256"]:
            raise ValueError(f"source snapshot changed: {copied}")
    for arm in manifest["arms"]:
        if arm.get("initialization_verified"):
            checkpoint = Path(arm["initial_checkpoint"])
            expected = arm["initial_manifest"]
            if (
                _read(checkpoint / "manifest.json") != expected
                or _sha(checkpoint / "state.pt") != expected["sha256"]
            ):
                raise ValueError("prepared initial checkpoint identity changed")
    for pin in manifest.get("fresh_reward_pins", []):
        if _pin(pin["path"]) != pin:
            raise ValueError("validated fresh reward artifact changed")
    if manifest.get("ready_to_train"):
        selection_pin = manifest.get("calibration_selection_pin")
        evidence = manifest.get("calibration_evidence_pins")
        if (
            not manifest.get("modelinit_verified")
            or not selection_pin
            or not evidence
            or not manifest.get("fresh_reward_pins")
            or not all(arm.get("initialization_verified") for arm in manifest["arms"])
        ):
            raise ValueError("ready comparison requires verified calibration evidence")
        for pin in [selection_pin, *evidence]:
            if _pin(pin["path"]) != pin:
                raise ValueError("calibration evidence changed")
        selection = _read(selection_pin["path"])
        if (
            selection.get("format") != "carryrl.token_calibration_selection.v1"
            or selection.get("approved") is not True
            or any(
                _read(arm["config"]["path"])["learning_rate"]
                != selection.get("selected_learning_rate")
                for arm in manifest["arms"]
            )
        ):
            raise ValueError("calibration selection disagrees with primary settings")
    return manifest


def build_initial_agents(
    configs, frozen_modules=None, *, seed, agent_factory=None, optimizer_factory=None
):
    """Build identical common trainables and nested queries using shared frozen modules.

    Injected factories permit inexpensive, offline verification. The returned models
    share frozen encoders, but never share trainable parameter storage.
    """
    import torch

    from carryrl.agent.model import GeneralistAgent

    agent_factory = agent_factory or GeneralistAgent
    optimizer_factory = optimizer_factory or torch.optim.AdamW
    if tuple(sorted(configs)) != TOKENS:
        raise ValueError("exactly the 16/32/64 arms are required")
    if frozen_modules is not None:
        if any(parameter.requires_grad for parameter in frozen_modules.vision.parameters()):
            raise ValueError("injected vision must be frozen")
        language = getattr(frozen_modules, "language", None)
        if language is not None and any(p.requires_grad for p in language.parameters()):
            raise ValueError("injected language must be frozen")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        master = (
            agent_factory(configs[64].agent)
            if frozen_modules is None
            else agent_factory(
                configs[64].agent,
                vision=frozen_modules.vision,
                language=getattr(frozen_modules, "language", None),
            )
        )
        vision, language = master.vision, getattr(master, "language", None)
        master.decoder.initialize_motor_prior("balanced")
        master_state = master.state_dict()
        result = {}
        for tokens in TOKENS:
            with torch.random.fork_rng(devices=[]):
                agent = (
                    master
                    if tokens == 64
                    else agent_factory(configs[tokens].agent, vision=vision, language=language)
                )
            if tokens != 64:
                state = {
                    name: (value[:tokens] if name == QUERY else value)
                    for name, value in master_state.items()
                }
                agent.load_state_dict(state, strict=True)
            optimizer = optimizer_factory(
                [p for p in agent.parameters() if p.requires_grad],
                lr=configs[tokens].learning_rate,
                weight_decay=configs[tokens].weight_decay,
            )
            if optimizer.state:
                raise ValueError("initial optimizer must have empty state")
            result[tokens] = (agent, optimizer)
        for tokens, (agent, _) in result.items():
            if agent.vision is not vision or (
                language is not None and agent.language is not language
            ):
                raise ValueError("initial arms must share their frozen encoders")
            for name, value in agent.state_dict().items():
                expected = master_state[name][:tokens] if name == QUERY else master_state[name]
                if not torch.equal(value, expected):
                    raise ValueError(f"initial tensor mismatch: t{tokens}/{name}")
        return result


def materialize_initial(output):
    """Create verified zero-update checkpoints only; never invoke any training routine."""
    manifest = verify_plan(output)
    if manifest["modelinit_verified"]:
        raise ValueError("initial checkpoints already materialized")
    reward_pins = require_fresh_reward(output, manifest)
    import gc

    import torch

    from carryrl.training import checkpoint_storage as storage
    from carryrl.training.sequence import (
        load_sequence_checkpoint,
        save_sequence_checkpoint,
        tensor_hash,
    )
    from carryrl.training.sequence_config import SequenceRunConfig

    root, output = Path(manifest["project_root"]), Path(output).resolve()
    configs = {
        arm["spatial_tokens"]: SequenceRunConfig.load(arm["config"]["path"])
        for arm in manifest["arms"]
    }
    for arm in manifest["arms"]:
        checkpoint = Path(arm["initial_checkpoint"])
        storage.checkpoint_path(root, checkpoint)
        if checkpoint.parent.parent.exists():
            raise FileExistsError(f"initial output already exists: {checkpoint.parent.parent}")
    torch.set_num_threads(2)
    from carryrl.rewards.pretrain import load_screen_reward_checkpoint

    reward, embedded_report = load_screen_reward_checkpoint(
        Path(output) / "reward" / "reward_checkpoint.pt", require_pass=True
    )
    if json.loads(json.dumps(embedded_report, allow_nan=False)) != _read(
        Path(output) / "reward" / "reward_report.json"
    ):
        raise ValueError("fresh reward embedded report differs from its pinned report")
    models = build_initial_agents(configs, seed=PRIMARY_SEED)
    frozen = models[64][0]
    reward_sha = tensor_hash(reward)
    shared_hashes = set()
    for arm in manifest["arms"]:
        agent = models[arm["spatial_tokens"]][0]
        shared = hashlib.sha256()
        for name, parameter in agent.named_parameters():
            if parameter.requires_grad and name != QUERY:
                shared.update(name.encode())
                shared.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        arm["shared_trainable_sha256"] = shared.hexdigest()
        arm["query_sha256"] = hashlib.sha256(
            agent.resampler.queries.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        arm["query_prefix_of_64_verified"] = True
        arm["trainable_parameters"] = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        shared_hashes.add(shared.hexdigest())
    if len(shared_hashes) != 1:
        raise ValueError("common trainable hashes differ across initial arms")
    del agent
    manifest["frozen_module_hashes"] = {
        "vision": tensor_hash(frozen.vision),
        "language": tensor_hash(frozen.language),
        "reward": reward_sha,
    }
    del frozen
    for arm in manifest["arms"]:
        tokens = arm["spatial_tokens"]
        agent, optimizer = models.pop(tokens)
        config = configs[tokens]
        initial = Path(arm["initial_checkpoint"]).parent.parent
        initial.mkdir(parents=True)
        _write(initial / "config.json", config.to_dict())
        checkpoint = save_sequence_checkpoint(
            agent, optimizer, reward, config, initial, rollout=0, decisions=0, revision=0
        )
        expected_agent = tensor_hash(agent)
        del agent, optimizer
        gc.collect()
        checked, checked_optimizer, checked_reward, data = load_sequence_checkpoint(
            checkpoint, config, evaluation_only=True
        )
        if (
            checked_optimizer.state
            or any(data[key] != 0 for key in ("rollout", "decisions", "weights_revision"))
            or tensor_hash(checked) != expected_agent
            or tensor_hash(checked_reward) != reward_sha
        ):
            raise ValueError("initial checkpoint roundtrip verification failed")
        del checked, checked_optimizer, checked_reward, data
        gc.collect()
        arm["initial_manifest"] = _read(checkpoint / "manifest.json")
        arm["initialization_verified"] = True
        _write(output / "manifest.json", manifest)
    # Source/config pins and independently validated reward must still match.
    verify_plan(output)
    if require_fresh_reward(output, manifest) != reward_pins:
        raise ValueError("fresh reward files changed during materialization")
    manifest["fresh_reward_pins"] = reward_pins
    manifest["reward_setup"] = "fresh_reward_validated"
    manifest["modelinit_verified"] = True
    manifest["configuration_validation"] = (
        "all arms parsed by SequenceRunConfig and roundtrip loaded"
    )
    _write(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--instructions", type=Path)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="one shared proposed LR; readiness still requires verified evidence",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--materialize-initial",
        action="store_true",
        help="explicitly create initial checkpoints only; zero training updates",
    )
    mode.add_argument("--verify-plan", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_plan:
        result = verify_plan(args.output)
    else:
        if not args.output.exists():
            prepare_comparison(
                args.root,
                args.output,
                base_config=args.base_config,
                instructions=args.instructions,
                learning_rate=args.learning_rate,
            )
        elif not args.materialize_initial:
            raise FileExistsError(f"comparison output already exists: {args.output}")
        result = (
            materialize_initial(args.output)
            if args.materialize_initial
            else verify_plan(args.output)
        )
    print(
        json.dumps(
            {
                "manifest": str(args.output.resolve() / "manifest.json"),
                "prepared": result["prepared"],
                "modelinit_verified": result["modelinit_verified"],
                "training_executed": False,
                "policy_updates": 0,
            }
        )
    )
    return 0
