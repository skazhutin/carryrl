"""Integration contracts for new optional PPO components, without production models."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_reward_composition_integration import DeadlineEnv, PredictedReward, outcome
from test_sequence_training import TinyLanguage, TinyVision, config, factory

from carryrl.agent.model import GeneralistAgent
from carryrl.envs.atomic_suite import TASKS
from carryrl.envs.curriculum import STAGES
from carryrl.rewards.composition import OUTCOME_POTENTIAL_VERSION, TEXT_CONTROL_VERSION
from carryrl.rewards.control_text import CANDIDATES, ControlPrediction, ControlPredictionTable
from carryrl.training import sequence
from carryrl.training.run import file_sha256
from carryrl.training.sequence_config import TaskCurriculumConfig
from carryrl.training.sequence_extensions import (
    checkpoint_extension_metadata,
    json_hash,
    prepare_extensions,
    validate_checkpoint_extensions,
)


def language_factory(settings):
    return GeneralistAgent(settings, vision=TinyVision(), language=TinyLanguage())


def text_config(schedule="temporary"):
    base = config()
    return replace(base, agent=replace(base.agent, text_encoder="multilingual_e5_small"),
                   episode_instructions={task: ("Hold RIGHT to move.",) for task in base.tasks},
                   reward_composition={"version": TEXT_CONTROL_VERSION,
                                       "parameters": {"control_schedule": schedule}},
                   text_reward_checkpoint="future-model", text_reward_sha256="0" * 64,
                   control_schedule_rollouts=8)


class PredictedControls:
    identity = {"model_hash": "1" * 64, "calibration_hash": "2" * 64}

    def save_resources(self, destination, **kwargs):
        destination.mkdir()
        (destination / "manifest.json").write_text(json.dumps(self.identity))

    def predict(self, text):
        self.text = text
        return ControlPredictionTable(hashlib.sha256(text.encode()).hexdigest(), "1" * 64,
            "2" * 64, tuple(ControlPrediction(key, "unknown", "unknown", None, 1.0)
                            for key in CANDIDATES))


def test_text_collection_uses_exact_actor_instruction_and_no_hidden_reward():
    torch.set_num_threads(1)
    cfg, controls = text_config(), PredictedControls()
    episode = sequence.collect_episode(language_factory(cfg.agent), PredictedReward(), cfg,
        task="navigation", seed=1, env_factory=DeadlineEnv, outcome_fn=outcome,
        text_reward=controls, control_coefficient=.5)
    assert controls.text == episode.instruction == "Hold RIGHT to move."
    assert episode.reward_diagnostics["control_coefficient"] == .5
    assert len(episode.reward_diagnostics["control_table_sha256"]) == 64
    assert episode.reward_components["control_bonus"] == [0.] * len(episode)
    np.testing.assert_allclose(np.array(list(episode.reward_components.values())).sum(axis=0),
                               episode.rewards.numpy(), atol=1e-7)


def test_missing_or_wrong_text_prediction_fails_before_environment_construction():
    cfg = text_config()
    def forbidden(*args, **kwargs):
        pytest.fail("environment must not start before text reward validation")
    with pytest.raises(ValueError, match="validated frozen model"):
        sequence.collect_episode(language_factory(cfg.agent), PredictedReward(), cfg,
                                 task="navigation", seed=1, env_factory=forbidden)
    controls = PredictedControls()
    table = controls.predict("different text")
    controls.predict = lambda text: table
    with pytest.raises(ValueError, match="different instruction"):
        sequence.collect_episode(language_factory(cfg.agent), PredictedReward(), cfg,
            task="navigation", seed=1, env_factory=forbidden, text_reward=controls)


@pytest.mark.parametrize("schedule,first,continued", [
    ("zero", 0., 0.), ("temporary", 1., 0.), ("permanent", 1., 1.),
])
def test_text_schedule_and_identity_survive_atomic_checkpoint_resume(
    tmp_path, monkeypatch, schedule, first, continued,
):
    torch.set_num_threads(1)
    artifact = tmp_path / "text_model"
    artifact.mkdir()
    (artifact / "manifest.json").write_text("{}")
    cfg = replace(text_config(schedule), text_reward_checkpoint=str(artifact),
                  text_reward_sha256=file_sha256(artifact / "manifest.json"),
                  total_rollouts=2, control_schedule_rollouts=4)
    monkeypatch.setattr("carryrl.rewards.control_text.load_text_control_reward",
                        lambda *args, **kwargs: PredictedControls())
    sequence.train_sequence(cfg, tmp_path / "original", agent_factory=language_factory,
        reward=PredictedReward(), env_factory=DeadlineEnv, outcome_fn=outcome)
    rows = [json.loads(row) for row in
            (tmp_path / "original/training.jsonl").read_text().splitlines()]
    assert [row["control_coefficient"] for row in rows] == [first, first]
    checkpoint = tmp_path / "original/checkpoints/rollout_0002"
    (artifact / "manifest.json").unlink()
    sequence.train_sequence(replace(cfg, total_rollouts=3), tmp_path / "continued",
        resume=checkpoint, agent_factory=language_factory, reward_factory=PredictedReward,
        env_factory=DeadlineEnv, outcome_fn=outcome)
    row = json.loads((tmp_path / "continued/training.jsonl").read_text())
    assert row["control_coefficient"] == continued
    assert row["text_reward_identity"] == PredictedControls.identity
    with pytest.raises(ValueError, match="composition differs"):
        sequence.load_sequence_checkpoint(checkpoint,
            replace(cfg, total_rollouts=3, control_schedule_rollouts=8),
            agent_factory=language_factory, reward_factory=PredictedReward)


def test_rollout_storage_budget_fails_before_any_action():
    cfg = config()
    class NoStep(DeadlineEnv):
        def step(self, action):
            pytest.fail("budget must be checked before action sampling")
    with pytest.raises(MemoryError, match="storage"):
        sequence.collect_episode(factory(cfg.agent), PredictedReward(), cfg,
            task="navigation", seed=1, env_factory=NoStep, storage_budget_bytes=1)


@pytest.mark.parametrize("changes", [
    {"advantage_std_floor": 0}, {"advantage_std_floor": float("nan")},
    {"ppo_normalization": "unknown"}, {"ppo_normalization": "rollout"},
    {"rollout_memory_budget_mib": -1}, {"gradient_probe_every": True},
])
def test_new_config_rejects_unsafe_values(changes):
    with pytest.raises(ValueError):
        replace(config(), **changes)


def test_curriculum_rejects_legacy_reward_without_classified_outcomes():
    with pytest.raises(ValueError, match="classified finite-horizon outcomes"):
        replace(config(), task_curriculum=TaskCurriculumConfig("admission.json", "0" * 64),
                evaluation_difficulty=1.0)


def curriculum_fixture(tmp_path, monkeypatch):
    admission = {"stage_spec_sha256": "3" * 64, "tasks": {
        task: {stage: {"allowed_level_indices": list(range(16)), "levels": [
            {"level_index": index, "runtime_id": f"{task}-{stage}-{index}"}
            for index in range(16)]} for stage in STAGES} for task in TASKS}}
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(admission))
    cfg = replace(config(), tasks=TASKS, max_episode_steps=2, evaluation_difficulty=1.0,
        task_curriculum=TaskCurriculumConfig(str(path), file_sha256(path)),
        reward_composition={"version": OUTCOME_POTENTIAL_VERSION,
                            "parameters": {"goal_choice_utility": "completion_only"}})
    # This test verifies sequencing/atomic state; real environment admission has its own suite.
    def validate(path, **kwargs):
        assert file_sha256(path) == kwargs["expected_sha256"]
        return json.loads(path.read_text())
    monkeypatch.setattr("carryrl.training.task_curriculum.validate_task_curriculum_admission",
                        validate)
    return cfg, admission


class CurriculumEnv(DeadlineEnv):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.stage = kwargs.get("curriculum_stage")

    def reset(self, *, seed, options=None):
        frame, info = super().reset(seed=seed)
        if options:
            info["runtime_fingerprint"] = f"{self.task}-{self.stage}-{options['level_index']}"
            info["effective_parameters"] = {"fixture": True}
        return frame, info


@pytest.mark.parametrize("accepted", [True, False])
@pytest.mark.parametrize("storage", ["memory", "mmap"])
def test_curriculum_commits_only_after_optimizer_and_resumes_from_packaged_admission(
    tmp_path, monkeypatch, accepted, storage,
):
    torch.set_num_threads(1)
    cfg, _ = curriculum_fixture(tmp_path, monkeypatch)
    cfg = replace(cfg, rollout_feature_storage=storage,
                  rollout_disk_budget_mib=32 if storage == "mmap" else 0)
    update = sequence.update_ppo
    if not accepted:
        def reject(agent, optimizer, episodes, cfg, **kwargs):
            return {"optimizer_steps": 0, "valid_transitions": sum(map(len, episodes))}
        monkeypatch.setattr(sequence, "update_ppo", reject)
    output = tmp_path / "first"
    result = sequence.train_sequence(cfg, output, agent_factory=factory,
        reward=PredictedReward(), reward_factory=PredictedReward, env_factory=CurriculumEnv,
        outcome_fn=outcome)
    checkpoint = output / "checkpoints/rollout_0001"
    agent, _, _, saved = sequence.load_sequence_checkpoint(checkpoint,
        replace(cfg, total_rollouts=2),
        agent_factory=factory, reward_factory=PredictedReward)
    state = saved["curriculum_state"]
    assert state["cycle"] == int(accepted)
    assert not state["issued"] and not state["pending"]
    assert result["optimizer_steps_this_process"] == int(accepted)
    assert not list(output.glob(".rollout-*"))
    original = tmp_path / "admission.json"
    original.unlink()
    _, scheduler, _ = prepare_extensions(cfg, agent, checkpoint=checkpoint, saved=saved)
    assert scheduler.state_dict() == state
    next_ticket = scheduler.select_episode("navigation")
    assert next_ticket["level_index"] == int(accepted)
    monkeypatch.setattr(sequence, "update_ppo", update)
    next_cfg = replace(cfg, total_rollouts=2)
    sequence.train_sequence(next_cfg, tmp_path / "continued", resume=checkpoint,
        agent_factory=factory, reward_factory=PredictedReward, env_factory=CurriculumEnv,
        outcome_fn=outcome)
    assert (tmp_path / "continued/checkpoints/rollout_0002/curriculum_admission.json").is_file()


def test_checkpoint_rejects_missing_or_mutated_scheduler_and_admission(tmp_path, monkeypatch):
    cfg, _ = curriculum_fixture(tmp_path, monkeypatch)
    _, scheduler, _ = prepare_extensions(cfg, factory(cfg.agent))
    payload, manifest = checkpoint_extension_metadata(
        cfg, scheduler_state=scheduler.state_dict(), destination=tmp_path)
    validate_checkpoint_extensions(tmp_path, cfg, payload, manifest)
    payload["curriculum_state"]["cycle"] += 1
    with pytest.raises(ValueError, match="state missing or mismatched"):
        validate_checkpoint_extensions(tmp_path, cfg, payload, manifest)
    manifest["curriculum_state_sha256"] = json_hash(payload["curriculum_state"])
    (tmp_path / "curriculum_admission.json").write_text("{}")
    with pytest.raises(ValueError, match="admission SHA256"):
        validate_checkpoint_extensions(tmp_path, cfg, payload, manifest)
