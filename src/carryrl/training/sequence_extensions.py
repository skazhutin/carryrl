"""Pinned optional TRAIN components; never infer state from a mutable journal."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from carryrl.rewards.composition import TEXT_CONTROL_V2_VERSION, TEXT_CONTROL_VERSION
from carryrl.training.run import file_sha256


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def resource_hashes(directory):
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("missing or invalid checkpoint text reward resources")
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("text reward resources must not contain symlinks")
        if path.is_file():
            result[str(path.relative_to(directory))] = file_sha256(path)
    if not result:
        raise ValueError("empty checkpoint text reward resources")
    return result


def prepare_extensions(config, agent, *, checkpoint=None, saved=None):
    """Load only gated models and admitted TRAIN levels, before collection."""
    text_reward = scheduler = admission = None
    if config.reward_composition.version in (TEXT_CONTROL_VERSION, TEXT_CONTROL_V2_VERSION):
        from carryrl.rewards.control_text import load_text_control_reward

        directory = (Path(checkpoint) / "text_reward" if checkpoint is not None
                     else Path(config.text_reward_checkpoint))
        if checkpoint is None and file_sha256(directory / "manifest.json") != (
            config.text_reward_sha256
        ):
            raise ValueError("text reward manifest SHA256 differs from configuration")
        text_reward = load_text_control_reward(directory, shared_encoder=agent.language)
        if saved is not None and text_reward.identity != saved.get("text_reward_identity"):
            raise ValueError("checkpoint text reward identity mismatch")
    if config.task_curriculum is not None:
        from carryrl.training.task_curriculum import (
            TaskCurriculumScheduler,
            validate_task_curriculum_admission,
        )

        settings = config.task_curriculum
        admission_path = (Path(checkpoint) / "curriculum_admission.json"
                          if checkpoint is not None else Path(settings.admission_path))
        admission = validate_task_curriculum_admission(
            admission_path, level_bank_seed=config.level_bank_seed,
            image_size=config.env_image_size, max_steps=config.max_episode_steps,
            expected_sha256=settings.admission_sha256,
        )
        if set(admission["tasks"]) != set(config.tasks):
            raise ValueError("admission tasks differ from configured tasks")
        levels = {task: {stage: data["allowed_level_indices"]
                         for stage, data in stages.items()}
                  for task, stages in admission["tasks"].items()}
        state = None
        if saved is not None:
            if saved.get("curriculum_state") is None:
                raise ValueError("checkpoint is missing curriculum state")
            state = saved["curriculum_state"]
        scheduler = TaskCurriculumScheduler(
            config_hash=json_hash(asdict(settings)), admission_hash=settings.admission_sha256,
            allowed_levels=levels, state=state,
        )
    return text_reward, scheduler, admission


def bind_curriculum_ticket(ticket, admission):
    """Bind a scheduler choice to the exact admitted effective mechanics."""
    rows = admission["tasks"][ticket["task"]][ticket["stage"]]["levels"]
    row = next((row for row in rows if row["level_index"] == ticket["level_index"]), None)
    if row is None:
        raise ValueError("scheduler selected a level outside admission")
    return {**ticket, "runtime_id": row["runtime_id"],
            "stage_spec_sha256": admission["stage_spec_sha256"]}


def checkpoint_extension_metadata(config, *, scheduler_state=None, text_reward=None,
                                  destination=None, curriculum_admission_bytes=None):
    payload, manifest = {}, {}
    if config.task_curriculum is not None:
        if scheduler_state is None:
            raise ValueError("curriculum checkpoint requires scheduler state")
        payload["curriculum_state"] = scheduler_state
        manifest["curriculum_state_sha256"] = json_hash(scheduler_state)
        if curriculum_admission_bytes is None:
            curriculum_admission_bytes = Path(config.task_curriculum.admission_path).read_bytes()
        digest = hashlib.sha256(curriculum_admission_bytes).hexdigest()
        if digest != config.task_curriculum.admission_sha256:
            raise ValueError("checkpoint curriculum admission SHA256 mismatch")
        (Path(destination) / "curriculum_admission.json").write_bytes(curriculum_admission_bytes)
        manifest["curriculum_admission_sha256"] = digest
    elif scheduler_state is not None:
        raise ValueError("unexpected curriculum state")
    if config.reward_composition.version in (TEXT_CONTROL_VERSION, TEXT_CONTROL_V2_VERSION):
        if text_reward is None or destination is None:
            raise ValueError("text reward checkpoint requires gated model resources")
        text_reward.save_resources(Path(destination) / "text_reward", shared_encoder=True)
        payload["text_reward_identity"] = text_reward.identity
        manifest["text_reward_identity"] = text_reward.identity
        manifest["text_reward_resources_sha256"] = resource_hashes(
            Path(destination) / "text_reward")
    return payload, manifest


def validate_checkpoint_extensions(checkpoint, stored, data, manifest):
    if stored.task_curriculum is not None:
        state = data.get("curriculum_state")
        if state is None or json_hash(state) != manifest.get("curriculum_state_sha256"):
            raise ValueError("checkpoint curriculum state missing or mismatched")
        digest = file_sha256(Path(checkpoint) / "curriculum_admission.json")
        if (digest != stored.task_curriculum.admission_sha256
                or digest != manifest.get("curriculum_admission_sha256")):
            raise ValueError("checkpoint curriculum admission SHA256 mismatch")
    if stored.reward_composition.version in (TEXT_CONTROL_VERSION, TEXT_CONTROL_V2_VERSION):
        identity = data.get("text_reward_identity")
        if not identity or identity != manifest.get("text_reward_identity"):
            raise ValueError("checkpoint text reward identity missing or mismatched")
        if resource_hashes(Path(checkpoint) / "text_reward") != manifest.get(
            "text_reward_resources_sha256"
        ):
            raise ValueError("checkpoint text reward resources SHA256 mismatch")
