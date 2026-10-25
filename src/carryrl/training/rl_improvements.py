"""Preparation-only future PPO profiles and controlled ablations; never fit or launch."""

import copy
import hashlib
import json
from pathlib import Path

from carryrl.envs.atomic_suite import TASKS
from carryrl.rewards.composition import OUTCOME_POTENTIAL_VERSION, TEXT_CONTROL_VERSION
from carryrl.training.sequence_config import SequenceRunConfig
from carryrl.training.task_curriculum import validate_task_curriculum_admission

PROFILE_VERSION = "carryrl.rl-improvements-preparation.v1"
INSTRUCTIONS_PATH = Path(__file__).resolve().parents[3] / "configs" / "atomic_instructions.json"
CONTROLS_PATH = INSTRUCTIONS_PATH.with_name("atomic_controls.json")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def inspect_text_reward_artifact(path, expected_sha256, *, instructions_path=INSTRUCTIONS_PATH,
                                 controls_path=CONTROLS_PATH):
    """Static artifact/readiness checks; no model construction, download, or inference."""
    from carryrl.agent.language import TEXT_ID, TEXT_REVISION
    from carryrl.rewards.control_text import (
        CANDIDATES,
        MODEL_VERSION,
        _checked_calibration,
        calibration_gates,
        content_hash,
    )
    from carryrl.rewards.control_text_structured import (
        STRUCTURED_HEAD_ARCHITECTURE,
        STRUCTURED_MODEL_VERSION,
        checked_structured_calibration,
    )

    directory = Path(path).resolve(strict=True)
    if _sha(directory / "manifest.json") != expected_sha256:
        raise ValueError("Text reward manifest SHA256 mismatch")
    manifest = json.loads((directory / "manifest.json").read_text())
    from carryrl.rewards.control_text_factorized import FACTORIZED_MODEL_VERSION

    if manifest.get("version") == FACTORIZED_MODEL_VERSION:
        from carryrl.rewards.control_text_factorized_artifact import (
            inspect_factorized_control_artifact,
        )

        return inspect_factorized_control_artifact(
            directory, expected_manifest_hash=expected_sha256,
            runtime_instructions=instructions_path, runtime_controls=controls_path,
        )
    structured = manifest.get("version") == STRUCTURED_MODEL_VERSION
    if (manifest.get("version") not in {MODEL_VERSION, STRUCTURED_MODEL_VERSION}
            or manifest.get("encoder_id") != TEXT_ID
            or manifest.get("revision") != TEXT_REVISION
            or manifest.get("candidates") != list(CANDIDATES)
            or (structured and manifest.get("head_architecture") != STRUCTURED_HEAD_ARCHITECTURE)):
        raise ValueError("Unsupported frozen text reward identity")
    files = manifest.get("files", {})
    required = {"encoder.pt", "head.pt", "calibration.json", "validation.json"}
    if structured:
        required |= {"runtime_validation.json", "sealed_validation.json"}
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError("Preparation requires a self-contained frozen text reward bundle")
    actual = set()
    for member in directory.rglob("*"):
        if member.is_symlink():
            raise ValueError("Text reward resources must not contain symlinks")
        if member.is_file():
            actual.add(str(member.relative_to(directory)))
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("Text reward bundle has missing or unhashed resources")
    for filename, expected in files.items():
        member = (directory / filename).resolve(strict=True)
        if not member.is_relative_to(directory) or _sha(member) != expected:
            raise ValueError(f"Text reward resource hash mismatch: {filename}")
    report = json.loads((directory / "validation.json").read_text())
    calibration = json.loads((directory / "calibration.json").read_text())
    if structured:
        from carryrl.rewards.control_text_structured_artifact import _admission

        checked_structured_calibration(calibration)
        runtime = json.loads((directory / "runtime_validation.json").read_text())
        sealed = json.loads((directory / "sealed_validation.json").read_text())
        admission = _admission(report, runtime, sealed, manifest.get("training", {}))
        if (manifest.get("ready") is not True or not admission["passed"]
                or manifest.get("admission") != admission):
            raise ValueError("Frozen structured text reward readiness gates failed")
        identity = {"instructions_sha256": _sha(instructions_path),
                    "controls_sha256": _sha(controls_path)}
        if (any(runtime.get(key) != value for key, value in identity.items())
                or runtime.get("specification_hash") != content_hash(identity)):
            raise ValueError("Text reward runtime instruction/control identity mismatch")
        gates = admission["calibration"]
    else:
        gates = calibration_gates(report)
        _checked_calibration(calibration)
        if (manifest.get("ready") is not True
                or manifest.get("training", {}).get("trained") is not True
                or not gates["passed"] or manifest.get("gates") != gates):
            raise ValueError("Frozen text reward readiness gates failed")
    return {"manifest_sha256": expected_sha256, "version": manifest["version"],
            "files": files, "gates": gates,
            **({"admission": admission, "runtime_identity": identity} if structured else {}),
            "verification": "metadata, resource hashes and calibration gates",
            "runtime_weight_restore": "required again by sequence runner before collection"}


def _visual_reward_pin(path):
    """Pin provided shared visual reward bytes without fitting or loading it."""
    location = Path(path).resolve(strict=True)
    if location.is_symlink():
        raise ValueError("Shared visual reward must be a real local artifact")
    if location.is_file():
        return {"path": str(location), "sha256": _sha(location)}
    files = {str(member.relative_to(location)): _sha(member)
             for member in sorted(location.rglob("*")) if member.is_file()}
    if not files:
        raise ValueError("Shared visual reward artifact is empty")
    if any(member.is_symlink() for member in location.rglob("*")):
        raise ValueError("Shared visual reward resources must not contain symlinks")
    return {"path": str(location), "files": files}


def prepare_rl_improvements(output, *, base_config: SequenceRunConfig,
                            admission_path=None, admission_sha256=None,
                            schedule_rollouts: int | None = None, total_rollouts: int | None = None,
                            control_schedule_rollouts: int | None = None,
                            text_reward_checkpoint=None,
                            text_reward_sha256=None, seed=3176, episodes_per_task=8,
                            instructions_path=INSTRUCTIONS_PATH) -> dict:
    """Write future configs only when all shared dependencies pass preparation gates.

    Missing inputs produce explicit non-executable templates. No policy CP0 or
    learning result is created or claimed by this function.
    """
    # The historical argument remains a shorthand for both independently pinned horizons.
    total_rollouts = schedule_rollouts if total_rollouts is None else total_rollouts
    control_schedule_rollouts = (schedule_rollouts if control_schedule_rollouts is None
                                 else control_schedule_rollouts)
    if type(total_rollouts) is not int or total_rollouts < 1:
        raise ValueError("A positive total rollout count is required")
    if type(control_schedule_rollouts) is not int or control_schedule_rollouts < 1:
        raise ValueError("A positive fixed control schedule horizon is required")
    if type(episodes_per_task) is not int or episodes_per_task < 2:
        raise ValueError("Enhanced volume requires at least two episodes per task")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if not isinstance(base_config, SequenceRunConfig):
        raise TypeError("base_config must be SequenceRunConfig")
    if not base_config.reward_composition.finite_horizon:
        raise ValueError("Preparation requires an explicit finite-horizon primary reward baseline")
    primary_parameters = dict(base_config.reward_composition.to_dict()["parameters"])
    primary_parameters.pop("control_schedule", None)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preparation output must not already exist")
    instructions_path = Path(instructions_path).resolve(strict=True)
    instructions = json.loads(instructions_path.read_text())
    if set(instructions) != set(TASKS):
        raise ValueError("Corrected instructions must cover all sixteen tasks")
    required, checks = [], {}
    if admission_path is None or admission_sha256 is None:
        required.append({"input": "curriculum_admission", "reason": "missing pinned admission"})
    else:
        try:
            admitted = validate_task_curriculum_admission(
                admission_path, level_bank_seed=base_config.level_bank_seed,
                image_size=base_config.env_image_size, max_steps=80,
                expected_sha256=admission_sha256)
            checks["admission"] = {"sha256": admission_sha256,
                                    "stage_spec_sha256": admitted["stage_spec_sha256"]}
        except (ValueError, OSError, KeyError, TypeError) as error:
            required.append({"input": "curriculum_admission", "reason": str(error)})
    if text_reward_checkpoint is None or text_reward_sha256 is None:
        required.append({"input": "frozen_text_reward", "reason": "missing trained gated artifact"})
    else:
        try:
            checks["text_reward"] = inspect_text_reward_artifact(
                text_reward_checkpoint, text_reward_sha256, instructions_path=instructions_path)
        except (ValueError, OSError, KeyError, TypeError) as error:
            required.append({"input": "frozen_text_reward", "reason": str(error)})
    if base_config.reward_checkpoint is None:
        required.append({"input": "shared_visual_reward",
                         "reason": "missing common frozen artifact"})
    else:
        try:
            checks["visual_reward"] = _visual_reward_pin(base_config.reward_checkpoint)
        except (ValueError, OSError) as error:
            required.append({"input": "shared_visual_reward", "reason": str(error)})
    common = base_config.to_dict()
    common.update(seed=seed, tasks=list(TASKS), total_rollouts=total_rollouts,
                  control_schedule_rollouts=control_schedule_rollouts, max_episode_steps=80,
                  episodes_per_task=episodes_per_task, task_episode_multipliers={},
                  curriculum_difficulty=None, curriculum_warmup_rollouts=0,
                  evaluation_difficulty=1.0, ppo_normalization="rollout", ppo_task_balance=True,
                  ppo_minibatch_episodes=32, ppo_microbatch_episodes=1,
                  training_backend="mlx", device="cpu", rollout_memory_budget_mib=2048,
                  rollout_feature_storage="mmap", rollout_disk_budget_mib=6144,
                  motor_initialization="balanced",
                  advantage_std_floor=0.1 * primary_parameters["success_reward"],
                  gradient_probe_every=8, reward_model_mode="screen_v2",
                  reset_optimizer_on_warm_start=False,
                  episode_instructions=instructions, screen_text_language=None,
                  task_curriculum=({"admission_path": str(Path(admission_path).resolve()),
                                    "admission_sha256": admission_sha256}
                                   if admission_path is not None and admission_sha256 else None),
                  text_reward_checkpoint=(str(Path(text_reward_checkpoint).resolve())
                                          if text_reward_checkpoint else None),
                  text_reward_sha256=text_reward_sha256)
    common["agent"].update(motor_version="motor_v2", critic_initialization="zero",
                           text_encoder="multilingual_e5_small", instruction_max_tokens=512)
    parameters = {**primary_parameters, "control_schedule": "zero"}
    common["reward_composition"] = {"version": TEXT_CONTROL_VERSION, "parameters": parameters}
    configs = {}
    for schedule in ("zero", "temporary", "permanent"):
        config = copy.deepcopy(common)
        config["reward_composition"]["parameters"]["control_schedule"] = schedule
        configs[schedule] = {"config": config, "changes": {"control_schedule": schedule},
                             "paired_against": "zero", "cp0_group": "enhanced_shared_cp0"}
    variants = {
        "ablate_volume": {"episodes_per_task": base_config.episodes_per_task},
        "ablate_floor": {"advantage_std_floor": base_config.advantage_std_floor},
        "ablate_curriculum": {"task_curriculum": None},
        "ablate_motor": {"motor_initialization": "legacy",
                         "agent.motor_version": "legacy"},
        "ablate_critic": {"agent.critic_initialization": "legacy"},
        "legacy_paired_base": {"episodes_per_task": base_config.episodes_per_task,
                               "advantage_std_floor": base_config.advantage_std_floor,
                               "task_curriculum": None, "motor_initialization": "legacy",
                               "agent.motor_version": "legacy",
                               "agent.critic_initialization": "legacy"},
    }
    for name, changes in variants.items():
        config = copy.deepcopy(common)
        for field, value in changes.items():
            if field.startswith("agent."):
                config["agent"][field.split(".")[1]] = value
            else:
                config[field] = value
        # Every minibatch contains all tasks even at the smaller baseline volume.
        config["ppo_minibatch_episodes"] = min(32, 16 * config["episodes_per_task"])
        configs[name] = {"config": config, "changes": changes, "paired_against": "zero",
                         "cp0_group": ("enhanced_shared_cp0" if name not in
                                       ("ablate_motor", "ablate_critic", "legacy_paired_base")
                                       else "paired_seed_modified_initialization")}
    # Validate all structural settings even while missing artifact pins block
    # materialization; incomplete text pins are deliberately not fabricated.
    validation_errors = []
    for name, profile in configs.items():
        try:
            if not required:
                SequenceRunConfig.from_dict(profile["config"])
            else:
                structural = copy.deepcopy(profile["config"])
                structural["text_reward_checkpoint"] = None
                structural["text_reward_sha256"] = None
                structural["control_schedule_rollouts"] = 0
                structural["task_curriculum"] = None
                structural["reward_composition"]["version"] = OUTCOME_POTENTIAL_VERSION
                structural["reward_composition"]["parameters"].pop("control_schedule")
                SequenceRunConfig.from_dict(structural)
        except ValueError as error:
            validation_errors.append({"input": f"profile:{name}", "reason": str(error)})
    required.extend(validation_errors)
    ready = not required
    output.mkdir(parents=True)
    matrix = {"version": PROFILE_VERSION, "status": "prepared" if ready else "blocked",
              "preparation_ready": ready, "learning_quality": "unmeasured",
              "training_started": False, "required_inputs": required, "checks": checks,
              "primary_reward": {"parameters": primary_parameters,
                                 "rule": "inherited unchanged from explicit baseline"},
              "instructions": {"path": str(instructions_path), "sha256": _sha(instructions_path)},
              "horizon_rollouts": control_schedule_rollouts,
              "total_rollouts": total_rollouts,
              "volume_choice": {"enhanced_episodes_per_task": episodes_per_task,
                                "baseline_episodes_per_task": base_config.episodes_per_task,
                                "status": "engineering proposal, not a user requirement"},
              "pairing": {"seed": seed, "cp0_created": False,
                          "shared_cp0_hash": None, "resume": None, "warm_start": None,
                          "future_gate": "zero/temporary/permanent require identical initial "
                                         "trainable weight hash and shared frozen artifacts",
                          "ablations": "initialization changes share seed; CP0 identity differs"},
              "comparison": {"normalization": "one shared rollout, equal task means",
                             "minibatch": "all sixteen tasks in every update",
                             "volume": "same rollout horizon; more data and optimizer updates "
                                       "are reported explicitly, no quality claim"},
              "profiles": {}}
    for name, profile in configs.items():
        directory = output / name
        directory.mkdir()
        filename = "config.json" if ready else "config.template.json"
        settings = profile.pop("config")
        episodes = 16 * settings["episodes_per_task"]
        minibatch = settings["ppo_minibatch_episodes"]
        profile["planned_budget"] = {
            "episodes_per_rollout": episodes,
            "decisions_upper_bound": total_rollouts * episodes * 80,
            "optimizer_steps_upper_bound": total_rollouts * settings["n_epochs"]
                                           * ((episodes + minibatch - 1) // minibatch),
            "kl_can_reduce_updates": True}
        _write(directory / filename, settings)
        matrix["profiles"][name] = {**profile, "config": str(directory / filename),
                                    "config_sha256": _sha(directory / filename),
                                    "executable_config": ready,
                                    "future_command": (["uv", "run", "--locked", "python", "-m",
                                                        "carryrl", "train-v1", "--config",
                                                        str(directory / filename), "--output",
                                                        str(directory / "run")] if ready else None)}
    _write(output / "manifest.json", matrix)
    return matrix
