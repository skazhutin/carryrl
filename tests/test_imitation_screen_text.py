"""Opt-in OCR observes displayed RGB and replays actual physical command history."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_imitation import CorrectionLearner, TinyLanguage, TinyVision, make_agent
from test_imitation_validation import trace_for
from test_sequence_capture import OutcomeEnv, outcome
from torch import nn

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.model import GeneralistAgent
from carryrl.evaluation.imitation import evaluate_imitation_validation
from carryrl.perception.screen_text import (
    ScreenTextBlock,
    ScreenTextResult,
    ScreenTextSettings,
)
from carryrl.training.imitation import (
    ImitationScreenText,
    collect_scripted_corrections,
    train_imitation,
)
from carryrl.training.sequence import (
    FrozenFrameCache,
    configure_screen_text,
    evaluate_sequence,
    load_sequence_checkpoint,
    save_sequence_checkpoint,
    sequence_evaluation_protocol,
    train_sequence,
)
from carryrl.training.sequence_config import SequenceRunConfig


class TokenLanguage(TinyLanguage):
    def encode_tokens(self, text):
        return self.feature.detach()[None].repeat(2, 1) + len(text) / 1000


class FakeReader:
    def __init__(self, language="eng", **kwargs):
        self.language = language
        self.frames = []

    def read(self, frame, *, step=None):
        self.frames.append((frame.copy(), step))
        height, width = frame.shape[:2]
        blocks = ((ScreenTextBlock(f"pixel {int(frame[0, 0, 0])}", .9,
                                   (0, 0, width, height)),) if frame.any() else ())
        return ScreenTextResult("ok" if blocks else "no_text", blocks,
            hashlib.sha256(frame.tobytes()).hexdigest(), step, width, height,
            "fake", "1", self.language, ScreenTextSettings(1., width * height, width, height))


def text_agent():
    agent = make_agent(instruction=True)
    agent.language = TokenLanguage().requires_grad_(False)
    return agent


def text_config(agent, **kwargs):
    return SequenceRunConfig(agent=agent.config, tasks=("navigation",), max_episode_steps=3,
        eval_episodes_per_task=1, screen_text_language="eng", **kwargs)


def test_config_opt_in_is_strict_and_legacy_protocol_unchanged():
    cfg = SequenceRunConfig()
    assert cfg.screen_text_language is None
    assert "screen_text_language" not in sequence_evaluation_protocol(cfg)
    with pytest.raises(ValueError, match="text_encoder"):
        replace(cfg, screen_text_language="eng")
    enabled = text_config(text_agent())
    assert enabled.episode_instructions == {}
    assert SequenceRunConfig.from_dict(enabled.to_dict()) == enabled
    assert sequence_evaluation_protocol(enabled)["screen_text_language"] == "eng"
    for invalid in ("", " eng", "eng+", True):
        with pytest.raises(ValueError, match="screen_text_language"):
            replace(enabled, screen_text_language=invalid)


def test_disk_observations_preserve_frame_step_and_empty_text(tmp_path):
    frames = [np.zeros((2, 2, 3), np.uint8), np.full((2, 2, 3), 7, np.uint8)]
    dataset = SimpleNamespace(frames=frames, episodes=[SimpleNamespace(frame_indices=(1, 0, 1))])
    reader = FakeReader()
    store = ImitationScreenText(dataset, reader, tmp_path / "observations.jsonl")
    records = [json.loads(line) for line in store.path.read_text().splitlines()]
    assert [row["frame_index"] for row in records] == [1, 0, 1]
    assert [row["step"] for row in records] == [0, 1, 2]
    assert records[1]["status"] == "no_text" and records[1]["blocks"] == []
    for record in records:
        assert record["frame_sha256"] == hashlib.sha256(
            frames[record["frame_index"]].tobytes()).hexdigest()
        assert store(record["frame_index"], record["step"]) == record["model_text"]
        assert json.loads(record["model_text"])["step"] == record["step"]
    assert not any(isinstance(value, torch.Tensor) for value in vars(store).values())


@pytest.mark.parametrize("stale", [{"frame_sha256": "stale"}, {"source_width": 99},
                                    {"source_height": 99}, {"step": 99}])
def test_ocr_replay_rejects_stale_frame_result(tmp_path, monkeypatch, stale):
    class StaleReader(FakeReader):
        def read(self, frame, *, step=None):
            return replace(super().read(frame, step=step), **stale)

    agent = text_agent()
    cfg = text_config(agent)
    trace = trace_for(cfg)
    dataset = SimpleNamespace(frames=list(trace.frames),
        episodes=[SimpleNamespace(frame_indices=(0, 1, 2))])
    with pytest.raises(ValueError, match="replay RGB frame/step"):
        ImitationScreenText(dataset, StaleReader(), tmp_path / "observations.jsonl")
    monkeypatch.setattr("carryrl.perception.screen_text.ScreenTextReader", StaleReader)
    with pytest.raises(ValueError, match="replay RGB frame/step"):
        evaluate_imitation_validation(agent, cfg, traces=[trace])


def test_bc_ocr_and_parent_share_displayed_frame_text_and_actual_history(tmp_path, monkeypatch):
    dataset = collect_scripted_corrections(CorrectionLearner(PhysicalAction()), 1,
        tasks=("navigation",), max_steps=3, expert_probability=0)
    agent = text_agent()
    reader = agent.screen_text_reader = FakeReader()
    store = ImitationScreenText(dataset, reader, tmp_path / "observations.jsonl")
    recorded = []
    original = GeneralistAgent.evaluate_episode

    def evaluate(self, patches, actions, **kwargs):
        recorded.append((self is agent, kwargs["screen_text"], kwargs["executed_actions"]))
        return original(self, patches, actions, **kwargs)

    monkeypatch.setattr(GeneralistAgent, "evaluate_episode", evaluate)
    result = train_imitation(agent, dataset, screen_text_provider=store,
                             canonical_anchor_coefficient=.1)
    assert result["optimizer_steps"] == 1
    assert [row[0] for row in recorded] == [True, False]
    assert recorded[0][1:] == recorded[1][1:]
    assert recorded[0][2] == dataset.episodes[0].executed_actions
    assert recorded[0][2] != dataset.episodes[0].actions
    assert [json.loads(text)["step"] for text in recorded[0][1]] == [0, 1, 2]
    with pytest.raises(ValueError, match="screen_text_provider"):
        train_imitation(agent, dataset)


def test_frozen_evaluation_ocr_uses_live_rgb_even_with_patch_cache(monkeypatch):
    monkeypatch.setattr("carryrl.perception.screen_text.ScreenTextReader", FakeReader)
    agent = text_agent()
    cfg = replace(text_config(agent), max_episode_steps=2)
    reader = configure_screen_text(agent, cfg)
    captured = []
    result = evaluate_sequence(agent, nn.Linear(1, 1), cfg, env_factory=OutcomeEnv,
                               outcome_fn=outcome, feature_cache=FrozenFrameCache(agent, 2),
                               on_playback=lambda record, *args: captured.append(record))
    observed = result["records"][0]["screen_text"]
    assert captured[0]["screen_text_language"] == "eng"
    assert captured[0]["screen_text"] == observed
    assert len(observed) == len(reader.frames) == 2
    assert [step for _, step in reader.frames] == [0, 1]
    for row, (frame, _) in zip(observed, reader.frames, strict=True):
        assert row["frame_sha256"] == hashlib.sha256(frame.tobytes()).hexdigest()
    assert "screen_text" not in result["records"][1]  # Random physical baseline.
    assert "screen_text_language" not in result["records"][1]


def test_live_correction_ocr_uses_displayed_rgb_even_with_patch_cache():
    agent = text_agent()
    reader = agent.screen_text_reader = FakeReader()
    dataset = collect_scripted_corrections(agent, 1, tasks=("navigation",), max_steps=3,
        expert_probability=0, feature_cache=FrozenFrameCache(agent, 3))
    assert len(reader.frames) == len(dataset.episodes[0].frame_indices)
    for (frame, _), index in zip(reader.frames, dataset.episodes[0].frame_indices, strict=True):
        np.testing.assert_array_equal(frame, dataset.frames[index])
    assert [step for _, step in reader.frames] == [0, 1, 2]


def test_validation_nll_ocr_tracks_actual_trace_rgb_and_history(monkeypatch):
    monkeypatch.setattr("carryrl.perception.screen_text.ScreenTextReader", FakeReader)
    agent = text_agent()
    cfg = text_config(agent)
    frames = tuple(np.full((32, 32, 3), value, np.uint8) for value in (7, 0, 3))
    trace = trace_for(cfg, frames=frames)
    recorded = []
    evaluate = agent.evaluate_episode

    def record(patches, actions, **kwargs):
        recorded.append(kwargs)
        return evaluate(patches, actions, **kwargs)

    monkeypatch.setattr(agent, "evaluate_episode", record)
    result = evaluate_imitation_validation(agent, cfg, traces=[trace])
    observations = result["records"][0]["screen_text"]
    assert len(observations) == len(trace.target_actions)
    assert recorded[0]["executed_actions"] == list(trace.executed_actions)
    assert recorded[0]["screen_text"] == [row["model_text"] for row in observations]
    assert [row["frame_sha256"] for row in observations] == [
        hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
    assert observations[1]["status"] == "no_text"
    assert result["protocol"]["screen_text_language"] == "eng"


def test_mlx_bc_replays_same_ocr_features_to_parent(tmp_path, monkeypatch):
    pytest.importorskip("mlx.core")
    from carryrl.agent.mlx_policy import MLXPolicy

    dataset = collect_scripted_corrections(CorrectionLearner(PhysicalAction()), 1,
        tasks=("navigation",), max_steps=3, expert_probability=0)
    agent = text_agent()
    store = ImitationScreenText(dataset, FakeReader(), tmp_path / "observations.jsonl")
    recorded = []
    evaluate = MLXPolicy.evaluate_episode

    def record(self, patches, actions, **kwargs):
        recorded.append((kwargs["screen_text_features"], kwargs["executed_actions"]))
        return evaluate(self, patches, actions, **kwargs)

    monkeypatch.setattr(MLXPolicy, "evaluate_episode", record)
    result = train_imitation(agent, dataset, backend="mlx", screen_text_provider=store,
                             canonical_anchor_coefficient=.1)
    assert result["optimizer_steps"] == 1
    assert len(recorded) == 2
    assert all(feature.shape == (2, 384) for feature in recorded[0][0])
    for left, right in zip(recorded[0][0], recorded[1][0], strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    assert recorded[0][1] == recorded[1][1] == dataset.episodes[0].executed_actions


def test_ppo_ocr_fails_before_creating_run(tmp_path):
    cfg = text_config(text_agent())
    with pytest.raises(ValueError, match="not PPO"):
        train_sequence(cfg, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_disabled_configuration_never_constructs_reader(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("RGB-only policy must not probe the OCR executable")

    monkeypatch.setattr("carryrl.perception.screen_text.ScreenTextReader", forbidden)
    agent = make_agent()
    cfg = SequenceRunConfig(agent=agent.config, max_episode_steps=3)
    assert configure_screen_text(agent, cfg) is None
    assert agent.screen_text_reader is None


def test_checkpoint_restores_ocr_and_rejects_silent_disable(tmp_path, monkeypatch):
    monkeypatch.setattr("carryrl.perception.screen_text.ScreenTextReader", FakeReader)
    agent = text_agent()
    cfg = text_config(agent, episode_instructions={"navigation": ("go",)})
    reward = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW([value for value in agent.parameters() if value.requires_grad])
    checkpoint = save_sequence_checkpoint(agent, optimizer, reward, cfg, tmp_path,
                                          rollout=0, decisions=0, revision=0)

    def factory(config):
        return GeneralistAgent(config, vision=TinyVision(), language=TokenLanguage())

    loaded, _, _, saved = load_sequence_checkpoint(checkpoint, cfg,
        agent_factory=factory, reward_factory=lambda _: nn.Linear(1, 1),
        evaluation_only=True, load_optimizer=False)
    assert loaded.screen_text_reader.language == saved["config"]["screen_text_language"] == "eng"
    with pytest.raises(ValueError, match="OCR checkpoint requires"):
        load_sequence_checkpoint(checkpoint, replace(cfg, screen_text_language=None),
            agent_factory=factory, reward_factory=lambda _: nn.Linear(1, 1),
            evaluation_only=True, load_optimizer=False)
