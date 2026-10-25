"""Offline reward composition, complete-rollout rejection and checkpoint provenance."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from test_sequence_training import TinyReward, config, factory

from carryrl.rewards.composition import (
    LEGACY_REWARD_VERSION,
    OUTCOME_POTENTIAL_VERSION,
    RewardCompositionConfig,
)
from carryrl.training import sequence
from carryrl.training.run import file_sha256
from carryrl.training.sequence_config import SequenceRunConfig


def finite_config(**changes):
    return replace(config(), reward_composition=RewardCompositionConfig(
        version=OUTCOME_POTENTIAL_VERSION), **changes)


class PredictedReward(TinyReward):
    def predict(self, frame):
        return {"score": float(frame[0, 0, 0]), "score_confidence": 1.0,
                "success_probability": 0.01, "failure_probability": 0.01, "known": True}


class DeadlineEnv:
    """Predictions depend only on actual RGB; hidden reward/info are poison."""

    terminal = False

    def __init__(self, task, **options):
        self.limit = options["max_steps"]
        self.task = task
        self.steps = 0

    def reset(self, *, seed):
        return np.zeros((64, 64, 3), dtype=np.uint8), {"level_id": "fixture"}

    def step(self, action):
        self.steps += 1
        frame = np.full((64, 64, 3), self.steps, dtype=np.uint8)
        return (frame, float("nan"), self.terminal and self.steps == self.limit,
                not self.terminal and self.steps == self.limit,
                {"success": True, "quality": 100, "teacher_reward": float("nan")})

    def close(self):
        pass


def outcome(frame):
    return {"success": False}


def make_checkpoint(tmp_path, cfg=None):
    cfg = cfg or config()
    torch.set_num_threads(1)
    agent, reward = factory(cfg.agent), PredictedReward()
    optimizer = torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad])
    path = sequence.save_sequence_checkpoint(agent, optimizer, reward, cfg, tmp_path,
                                              rollout=0, decisions=0, revision=0)
    return path, cfg


def rewrite(path, mutate):
    archive, manifest_path = path / "state.pt", path / "manifest.json"
    data = torch.load(archive, weights_only=True)
    manifest = json.loads(manifest_path.read_text())
    mutate(data, manifest)
    torch.save(data, archive)
    manifest["sha256"] = file_sha256(archive)
    manifest_path.write_text(json.dumps(manifest))


def load(path, cfg, **kwargs):
    return sequence.load_sequence_checkpoint(path, cfg, agent_factory=factory,
                                             reward_factory=PredictedReward, **kwargs)


def test_composition_config_roundtrip_defaults_and_context_rejection():
    cfg = finite_config()
    assert SequenceRunConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    legacy = config().to_dict()
    legacy.pop("reward_composition")
    assert SequenceRunConfig.from_dict(legacy).reward_composition.version == LEGACY_REWARD_VERSION
    with pytest.raises(ValueError, match="unknown"):
        replace(cfg, reward_composition={"version": OUTCOME_POTENTIAL_VERSION, "typo": 1})
    with pytest.raises(ValueError, match="goal_choice"):
        replace(cfg, tasks=("goal_choice",))
    with pytest.raises(ValueError, match="trajectory costs"):
        replace(cfg, reward_composition={"version": OUTCOME_POTENTIAL_VERSION,
                                        "parameters": {"time_cost": 1}})


@pytest.mark.parametrize("finite", [False, True])
def test_collection_components_and_deadline_bootstrap_ignore_hidden_rewards(finite):
    torch.set_num_threads(1)
    cfg = finite_config() if finite else config()
    episode = sequence.collect_episode(factory(cfg.agent), PredictedReward(), cfg,
        task="navigation", seed=1, env_factory=DeadlineEnv, outcome_fn=outcome)
    components = np.array(list(episode.reward_components.values())).sum(axis=0)
    np.testing.assert_allclose(components, episode.rewards.numpy(), atol=1e-7)
    assert episode.truncated[-1] and not episode.terminated[-1]
    assert episode.reward_diagnostics["composition"] == sequence._reward_composition_identity(cfg)
    if finite:
        assert episode.bootstrap == 0
        assert episode.reward_components["time_cost"] == [-0.001] * 4
        assert all(value < 0 for value in episode.reward_components["input_cost"])
    else:
        expected = torch.tensor([sequence._transition(
            {"score": index, "success_probability": .01, "failure_probability": .01},
            {"score": index + 1, "success_probability": .01, "failure_probability": .01})
            for index in range(4)])
        assert torch.equal(expected, episode.rewards)
        assert episode.bootstrap != 0


def test_missing_learned_terminal_rejects_whole_rollout_before_optimizer(tmp_path, monkeypatch):
    class InvalidSecondEpisode(DeadlineEnv):
        def step(self, action):
            self.terminal = self.task == "memory"
            return super().step(action)

    updates = []
    monkeypatch.setattr(sequence, "update_ppo", lambda *a, **k: updates.append(True))
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="entire rollout rejected.*terminated_without"):
        sequence.train_sequence(finite_config(), output, agent_factory=factory,
            reward=PredictedReward(), reward_factory=PredictedReward,
            env_factory=InvalidSecondEpisode, outcome_fn=outcome)
    assert not updates
    assert len(list((output / "trajectories").rglob("*.npz"))) == 1
    assert not (output / "training.jsonl").exists()
    latest = json.loads((output / "latest.json").read_text())
    initial = torch.load(latest["checkpoint"] + "/state.pt", weights_only=True)
    assert initial["weights_revision"] == 0
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("load_optimizer", [False, True])
@pytest.mark.parametrize("evaluation_only", [False, True])
@pytest.mark.parametrize("damage", ["state_missing", "manifest_missing", "both_missing",
                                    "both_wrong", "config_changed"])
def test_checkpoint_identity_is_verified_in_every_load_mode(
        tmp_path, load_optimizer, evaluation_only, damage):
    path, cfg = make_checkpoint(tmp_path)

    def mutate(data, manifest):
        if damage in {"state_missing", "both_missing"}:
            data.pop("reward_composition")
        if damage in {"manifest_missing", "both_missing"}:
            manifest.pop("reward_composition")
        if damage == "both_wrong":
            data["reward_composition"] = manifest["reward_composition"] = {"version": "fake"}
        if damage == "config_changed":
            data["config"]["reward_confidence_threshold"] = .8

    rewrite(path, mutate)
    with pytest.raises(ValueError, match="reward composition"):
        load(path, cfg, load_optimizer=load_optimizer, evaluation_only=evaluation_only)


@pytest.mark.parametrize("evaluation_only", [False, True])
def test_historical_absent_config_and_both_identities_keep_legacy(tmp_path, evaluation_only):
    path, cfg = make_checkpoint(tmp_path)

    def historical(data, manifest):
        data["config"].pop("reward_composition")
        data.pop("reward_composition")
        manifest.pop("reward_composition")
        # Recognizer identity never determines the reward arithmetic.
        data.pop("reward_model_version", None)
        manifest.pop("reward_model_version", None)

    rewrite(path, historical)
    _, _, _, data = load(path, cfg, evaluation_only=evaluation_only, load_optimizer=False)
    assert "reward_composition" not in data["config"]
    with pytest.raises(ValueError, match="explicit transfer"):
        load(path, finite_config(), evaluation_only=True, load_optimizer=False)
    load(path, finite_config(), evaluation_only=True, load_optimizer=False,
         allow_reward_composition_change=True)
    with pytest.raises(ValueError, match="evaluation_only"):
        load(path, finite_config(), allow_reward_composition_change=True)


def test_new_checkpoint_identity_snapshot_and_explicit_warm_start_lineage(tmp_path, monkeypatch):
    source_dir = tmp_path / "parent"
    source_dir.mkdir()
    path, _ = make_checkpoint(source_dir)
    cfg = finite_config()
    monkeypatch.setattr(sequence, "evaluate_sequence", lambda *a, **k: {"summaries": {}})
    output = tmp_path / "child"
    sequence.train_sequence(cfg, output, warm_start=path, agent_factory=factory,
        reward_factory=PredictedReward, env_factory=DeadlineEnv, outcome_fn=outcome)
    metadata = json.loads((output / "metadata.json").read_text())
    lineage = metadata["warm_start_parent"]
    assert lineage["reward_composition_changed"] is True
    assert lineage["source_reward_composition"]["version"] == LEGACY_REWARD_VERSION
    assert lineage["target_reward_composition"]["version"] == OUTCOME_POTENTIAL_VERSION
    journal = json.loads((output / "training.jsonl").read_text())
    identity = sequence._reward_composition_identity(cfg)
    assert journal["reward_composition"] == metadata["reward_composition"] == identity
    for record in (output / "trajectories").rglob("*.npz"):
        sidecar = json.loads(record.with_suffix(".json").read_text())
        assert sidecar["reward_composition"] == identity
        assert sidecar["reward_sha256"] == journal["reward_snapshot_sha256"]
        with np.load(record, allow_pickle=False) as data:
            assert json.loads(str(data["reward_composition"])) == identity
            summed = sum(data["reward_" + name] for name in (
                "outcome", "progress", "time_cost", "input_cost"))
            np.testing.assert_allclose(summed, data["rewards"], atol=1e-7)
            assert data["discounted_return"] == pytest.approx(sum(
                cfg.gamma ** i * float(r) for i, r in enumerate(data["rewards"])))
    latest = json.loads((output / "latest.json").read_text())
    checkpoint = Path(latest["checkpoint"])
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    saved = torch.load(checkpoint / "state.pt", weights_only=True)
    assert saved["reward_composition"] == manifest["reward_composition"] == identity
    assert manifest["reward_sha256"] == journal["reward_snapshot_sha256"]


@pytest.mark.parametrize("damage", ["state_missing", "manifest_missing", "both_wrong"])
def test_direct_frozen_loader_checks_composition_before_model_allocation(tmp_path, monkeypatch,
                                                                         damage):
    from carryrl.dashboard import frozen_playback

    path, _ = make_checkpoint(tmp_path)

    def mutate(data, manifest):
        if damage == "state_missing":
            data.pop("reward_composition")
        elif damage == "manifest_missing":
            manifest.pop("reward_composition")
        else:
            data["reward_composition"] = manifest["reward_composition"] = {"version": "fake"}

    rewrite(path, mutate)
    monkeypatch.setattr(frozen_playback, "GeneralistAgent", lambda *a, **k: pytest.fail(
        "invalid source composition must fail before graph allocation"))
    with pytest.raises(ValueError, match="reward composition"):
        frozen_playback.load_frozen_agent(path)


def test_direct_frozen_loader_accepts_real_historical_legacy_config(tmp_path, monkeypatch):
    from carryrl.dashboard import frozen_playback

    path, cfg = make_checkpoint(tmp_path)

    def historical(data, manifest):
        data["config"].pop("reward_composition")
        data.pop("reward_composition")
        manifest.pop("reward_composition")

    rewrite(path, historical)
    monkeypatch.setattr(frozen_playback, "GeneralistAgent", lambda cfg, **kwargs: factory(cfg))
    _, loaded, _ = frozen_playback.load_frozen_agent(path)
    assert loaded == cfg and loaded.reward_composition.version == LEGACY_REWARD_VERSION


@pytest.mark.parametrize("bad_event", ["unknown", "conflict", "uncertain"])
def test_invalid_learned_terminal_aborts_without_consulting_hidden_outcome(bad_event):
    class TerminalEnv(DeadlineEnv):
        terminal = True

    class TerminalReward(PredictedReward):
        def predict(self, frame):
            prediction = super().predict(frame)
            if frame[0, 0, 0] == 4:
                if bad_event == "unknown":
                    prediction["known"] = False
                else:
                    probability = 1.0 if bad_event == "conflict" else .5
                    prediction.update(success_probability=probability,
                                      failure_probability=probability)
            return prediction

    cfg = finite_config()
    with pytest.raises(ValueError, match="entire rollout rejected.*terminal_event_" + bad_event):
        sequence.collect_episode(factory(cfg.agent), TerminalReward(), cfg, task="navigation",
            seed=1, env_factory=TerminalEnv, outcome_fn=lambda _: pytest.fail(
                "hidden/diagnostic outcome must not resolve learned event uncertainty"))


def test_new_composition_uncertainty_counts_initial_and_actual_step_confidence():
    class UncertainScore(PredictedReward):
        def predict(self, frame):
            result = super().predict(frame)
            if int(frame[0, 0, 0]) in {0, 2}:
                result.pop("score_confidence")
            if int(frame[0, 0, 0]) == 3:
                result["success_probability"] = .2  # Below the configured event confidence.
            return result

    cfg = finite_config()
    episode = sequence.collect_episode(factory(cfg.agent), UncertainScore(), cfg,
        task="navigation", seed=1, env_factory=DeadlineEnv, outcome_fn=outcome)
    assert episode.reward_uncertain_steps == 3
    assert episode.reward_diagnostics["initial_uncertain"]
    assert not episode.reward_diagnostics["steps"][1]["score_trusted"]
    assert episode.reward_diagnostics["steps"][2]["event_status"] == "uncertain"


@pytest.mark.parametrize(("change", "allow"), [(False, False), (True, False),
                                               (True, True), (False, True)])
def test_imitation_cli_requires_explicit_actual_transfer_and_records_lineage(
        tmp_path, monkeypatch, change, allow):
    import sys

    from test_imitation_feature_storage import script

    source = tmp_path / "parent"
    source.mkdir()
    path, cfg = make_checkpoint(source)
    target = finite_config() if change else cfg
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(target.to_dict()))
    output = tmp_path / "imitation"
    monkeypatch.setattr(script, "load_sequence_checkpoint", load)
    monkeypatch.setattr(script, "provenance", lambda: {})

    class CollectionReached(Exception):
        pass

    def stop_collection(*args, **kwargs):
        raise CollectionReached()

    monkeypatch.setattr(script, "collect_scripted_demonstrations", stop_collection)
    arguments = ["train_imitation.py", "--config", str(config_path), "--warm-start", str(path),
                 "--output", str(output), "--training-backend", "torch", "--epochs", "1"]
    if allow:
        arguments.append("--allow-reward-composition-change")
    monkeypatch.setattr(sys, "argv", arguments)
    if change and not allow:
        with pytest.raises(ValueError, match="explicit transfer"):
            script.main()
    elif allow and not change:
        with pytest.raises(ValueError, match="actual objective change"):
            script.main()
    else:
        # Stop at the first corpus collection; validate real checkpoint loading and metadata
        # without collecting teacher data or performing imitation/optimizer updates.
        with pytest.raises(CollectionReached):
            script.main()
        transfer = json.loads((output / "metadata.json").read_text())["reward_composition_transfer"]
        assert transfer == {"source": sequence._reward_composition_identity(cfg),
                            "target": sequence._reward_composition_identity(target),
                            "changed": change, "explicitly_allowed": allow}
    assert not (output / "latest.json").exists()


def test_explicit_transfer_never_bypasses_source_identity_validation(tmp_path):
    path, _ = make_checkpoint(tmp_path)
    target = finite_config()

    def tamper(data, manifest):
        # Even two matching declarations cannot contradict the source config.
        data["reward_composition"] = manifest["reward_composition"] = (
            sequence._reward_composition_identity(target))

    rewrite(path, tamper)
    with pytest.raises(ValueError, match="metadata mismatch"):
        load(path, target, evaluation_only=True, load_optimizer=False,
             allow_reward_composition_change=True)
