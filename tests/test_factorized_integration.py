"""Version dispatch and physical horizon safeguards at the actual PPO boundary."""

import hashlib
import json
from dataclasses import replace

import pytest
import torch
from test_reward_composition_integration import DeadlineEnv, PredictedReward, outcome
from test_sequence_extensions import language_factory, text_config

from carryrl.rewards.composition import TEXT_CONTROL_V2_VERSION
from carryrl.rewards.control_text import CANDIDATES, load_text_control_reward
from carryrl.rewards.control_text_factorized import (
    FACTORIZED_MODEL_VERSION,
    FactorizedControlPrediction,
    FactorizedControlPredictionTable,
    FieldDecision,
    PairDecision,
)
from carryrl.training import sequence
from carryrl.training.run import file_sha256


def v2_config():
    return replace(text_config(), control_schedule_rollouts=64,
                   reward_composition={"version": TEXT_CONTROL_V2_VERSION,
                                       "parameters": {"control_schedule": "temporary"}})


class UnspecifiedControls:
    identity = {"version": FACTORIZED_MODEL_VERSION,
                "model_hash": "1" * 64, "calibration_hash": "2" * 64}

    def save_resources(self, destination, **kwargs):
        destination.mkdir()
        (destination / "manifest.json").write_text(json.dumps(self.identity))

    def predict(self, text):
        unknown = FieldDecision("unspecified", 1., False, "insufficient_information")
        return FactorizedControlPredictionTable(
            hashlib.sha256(text.encode()).hexdigest(), "1" * 64, "2" * 64,
            tuple(FactorizedControlPrediction(c, unknown, unknown) for c in CANDIDATES),
            tuple(PairDecision(a, b, unknown) for i, a in enumerate(CANDIDATES)
                  for b in CANDIDATES[i + 1:]))


def test_v2_configuration_cannot_declare_stretched_schedule():
    cfg = v2_config()
    with pytest.raises(ValueError, match="64-cycle"):
        replace(cfg, control_schedule_rollouts=192)


def test_actual_environment_horizon_must_match_penalty_denominator():
    torch.set_num_threads(1)
    cfg = v2_config()
    with pytest.raises(ValueError, match="actual environment"):
        sequence.collect_episode(
            language_factory(cfg.agent), PredictedReward(), cfg, task="navigation", seed=1,
            env_factory=DeadlineEnv, outcome_fn=outcome, text_reward=UnspecifiedControls(),
            control_coefficient=1.)


def test_common_loader_dispatches_new_version(tmp_path, monkeypatch):
    from carryrl.rewards import control_text_factorized_artifact as artifact

    (tmp_path / "manifest.json").write_text(json.dumps({"version": FACTORIZED_MODEL_VERSION}))
    sentinel = object()
    seen = []
    monkeypatch.setattr(artifact, "load_factorized_control_reward",
                        lambda path, **kw: (seen.append((path, kw)), sentinel)[1])
    encoder = object()
    assert load_text_control_reward(tmp_path, shared_encoder=encoder) is sentinel
    assert seen == [(tmp_path.resolve(), {"shared_encoder": encoder})]


def test_v2_checkpoint_resume_keeps_absolute_schedule_and_packaged_model(tmp_path, monkeypatch):
    class PhysicalDeadlineEnv(DeadlineEnv):
        def __init__(self, task, **options):
            super().__init__(task, **options)
            self.max_steps = self.limit

    torch.set_num_threads(1)
    artifact = tmp_path / "original_model"
    artifact.mkdir()
    (artifact / "manifest.json").write_text("{}")
    cfg = replace(v2_config(), text_reward_checkpoint=str(artifact),
                  text_reward_sha256=file_sha256(artifact / "manifest.json"),
                  total_rollouts=32, checkpoint_every_rollouts=32, evaluate_initial=False,
                  eval_every_rollouts=1000, n_epochs=1)
    loaded = []

    def load(path, **kwargs):
        loaded.append(path)
        return UnspecifiedControls()

    monkeypatch.setattr("carryrl.rewards.control_text.load_text_control_reward", load)
    sequence.train_sequence(cfg, tmp_path / "first", agent_factory=language_factory,
                            reward=PredictedReward(), env_factory=PhysicalDeadlineEnv,
                            outcome_fn=outcome)
    checkpoint = tmp_path / "first/checkpoints/rollout_0032"
    (artifact / "manifest.json").unlink()
    sequence.train_sequence(replace(cfg, total_rollouts=33), tmp_path / "resumed",
                            resume=checkpoint, agent_factory=language_factory,
                            reward_factory=PredictedReward, env_factory=PhysicalDeadlineEnv,
                            outcome_fn=outcome)
    row = json.loads((tmp_path / "resumed/training.jsonl").read_text())
    assert row["rollout"] == 33
    assert row["control_coefficient"] == 31 / 32
    assert row["text_reward_identity"] == UnspecifiedControls.identity
    assert loaded[0] == artifact
    assert checkpoint / "text_reward" in loaded[1:]
