"""Frozen per-frame OCR tokens, masked attention, and live/replay equivalence."""

import hashlib
import warnings
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_agent_instructions import make_agent, make_language

from carryrl.agent.actions import PhysicalAction
from carryrl.perception.screen_text import (
    ScreenTextBlock,
    ScreenTextResult,
    ScreenTextSettings,
    format_model_text,
)


def test_ocr_tokens_keep_each_attended_feature_and_truncate_without_pooling():
    language = make_language(max_tokens=12)
    features = language.encode_tokens("long screen observation")
    text, kwargs = language.tokenizer.calls[0]
    assert text == "query: long screen observation"
    assert kwargs["truncation"] is True and kwargs["max_length"] == 12
    assert features.shape == (12, 384)
    expected_ids = language.tokenizer(text)["input_ids"][:, :12]
    expected = language.backbone.embedding(expected_ids)[0]
    torch.testing.assert_close(features, expected, rtol=0, atol=0)
    assert not torch.equal(features[0], features[1])
    assert not features.requires_grad and features.device.type == "cpu"
    with pytest.raises(ValueError, match="never truncated"):
        language("long screen observation")
    assert language.backbone.calls == 1
    short = make_language().encode_tokens("x")
    # The fixture tokenizer emits one masked trailing token: never expose it.
    assert len(short) == len("query: x") + 2


def test_ocr_cache_is_separate_bounded_copied_and_invalidated():
    language = make_language(cache_size=2)
    original = language.encode_tokens("a").clone()
    language("a")
    language.encode_tokens("b")
    language.encode_tokens("a").zero_()
    language.encode_tokens("c")
    assert list(language._token_cache) == ["a", "c"]
    assert list(language._cache) == ["a"]
    torch.testing.assert_close(language.encode_tokens("a"), original, rtol=0, atol=0)
    language.load_state_dict(language.state_dict(), strict=True)
    assert not language._cache and not language._token_cache
    language.encode_tokens("a")
    language.to(dtype=torch.float64)
    assert not language._token_cache
    assert language.encode_tokens("a").dtype == torch.float64


@pytest.mark.parametrize("enabled", [True, False])
def test_absent_or_empty_ocr_is_exact_legacy_path_with_unchanged_state_dict(enabled):
    agent = make_agent(enabled=enabled)
    patches, actions = torch.randn(6, 16, 384), [PhysicalAction()] * 6
    inventory = {name: value.clone() for name, value in agent.state_dict().items()}
    original = agent.evaluate_episode(patches, actions)
    for text in (None, [None] * 6, ["", " ", None, "\n", "", None]):
        actual = agent.evaluate_episode(patches, actions, screen_text=text)
        for key in original:
            torch.testing.assert_close(actual[key], original[key], rtol=0, atol=0)
    assert set(agent.state_dict()) == set(inventory)
    restored = make_agent(enabled=enabled)
    restored.load_state_dict(inventory, strict=True)
    actual = restored.evaluate_episode(patches, actions)
    for key in original:
        torch.testing.assert_close(actual[key], original[key], rtol=0, atol=0)
    assert agent.last_screen_text_result is None


def test_ocr_changes_policy_and_trains_existing_projection_only():
    agent = make_agent().train()
    patches, actions = torch.randn(7, 16, 384), [PhysicalAction()] * 7
    frozen = {name: value.clone() for name, value in agent.language.state_dict().items()}
    inventory = set(agent.state_dict())
    baseline = agent.evaluate_episode(patches, actions)
    observed = agent.evaluate_episode(patches, actions, screen_text=["visible score 1"] * 7)
    assert not torch.allclose(baseline["log_prob"], observed["log_prob"])
    (observed["log_prob"].mean() + observed["value"].square().mean()).backward()
    assert agent.instruction_projection.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in agent.language.parameters())
    assert set(agent.state_dict()) == inventory
    for name, value in agent.language.state_dict().items():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    after = make_agent()
    after.load_state_dict(agent.state_dict(), strict=True)
    replay = after.evaluate_episode(patches, actions, screen_text=["visible score 1"] * 7)
    for key in observed:
        torch.testing.assert_close(replay[key], observed[key], rtol=0, atol=0)


def test_variable_ocr_padding_is_ignored_in_summary_and_actor_attention():
    agent = make_agent()
    patches = torch.randn(3, 16, 384)
    tokens, mask = agent._screen_text_batch([None, "a", "much longer text"], 3)
    assert not mask[:, :agent.config.spatial_tokens].any()
    assert mask[0, agent.config.spatial_tokens:].all()
    assert len(agent.encode_screen_text("a")) < len(agent.encode_screen_text("much longer text"))
    changed = tokens.clone()
    changed[mask[:, agent.config.spatial_tokens:]] = 1e4
    spatial, summary = agent._spatial_summary(
        patches, [PhysicalAction()] * 3, screen_text_features=tokens, key_padding_mask=mask,
    )
    other_spatial, other_summary = agent._spatial_summary(
        patches, [PhysicalAction()] * 3, screen_text_features=changed, key_padding_mask=mask,
    )
    assert spatial.shape[1] == agent.config.spatial_tokens + tokens.shape[1]
    torch.testing.assert_close(summary, other_summary, rtol=0, atol=0)
    states = torch.randn(3, agent.config.width)
    torch.testing.assert_close(
        agent._actor_features(states, spatial, key_padding_mask=mask),
        agent._actor_features(states, other_spatial, key_padding_mask=mask), rtol=0, atol=0,
    )
    baseline_spatial, baseline_summary = agent._spatial_summary(patches, [PhysicalAction()] * 3)
    torch.testing.assert_close(summary[0], baseline_summary[0])
    torch.testing.assert_close(spatial[:, :agent.config.spatial_tokens], baseline_spatial)


def test_live_and_replay_match_changing_text_and_actual_commands_across_memory_boundaries():
    agent = make_agent()
    patches = torch.randn(10, 16, 384)
    texts = ["a", None, "longer visible line", "", "b", "c", None, "next", "stage", "end"]
    actual_commands = [PhysicalAction(keys=("w",)) if i % 2 else PhysicalAction()
                       for i in range(10)]
    agent.reset_episode(instruction="Move")
    decisions = []
    for patch, text, command in zip(patches, texts, actual_commands, strict=True):
        peek = agent.peek_value_from_patches(patch, screen_text=text)
        decision = agent.act_from_patches(patch, screen_text=text)
        assert decision.value == peek
        decisions.append(decision)
        agent.observe_action(command)
    state, previous, step = agent._episode_state, agent._previous_action, agent._screen_text_step
    replay = agent.evaluate_episode(
        patches, [decision.action for decision in decisions], instruction="Move",
        executed_actions=actual_commands, screen_text=texts,
    )
    assert agent._episode_state is state and agent._previous_action is previous
    assert agent._screen_text_step == step
    torch.testing.assert_close(replay["log_prob"], torch.tensor([d.log_prob for d in decisions]),
                               rtol=2e-5, atol=3e-5)
    torch.testing.assert_close(replay["value"], torch.tensor([d.value for d in decisions]),
                               rtol=2e-5, atol=3e-5)
    changed = agent.evaluate_episode(
        patches, [decision.action for decision in decisions], instruction="Move",
        executed_actions=actual_commands, screen_text=list(reversed(texts)),
    )
    assert not torch.allclose(replay["log_prob"], changed["log_prob"])


class RecordingReader:
    def __init__(self):
        self.frames = []
        self.results = []

    def read(self, frame, *, step):
        self.frames.append(frame.copy())
        height, width = frame.shape[:2]
        result = ScreenTextResult(
            "ok", (ScreenTextBlock(f"visible {int(frame[0, 0, 0])}", .9, (1, 2, 3, 4)),),
            hashlib.sha256(frame.tobytes()).hexdigest(), step, width, height,
            "fixture", "1", "eng", ScreenTextSettings(1, 10000, width, height),
        )
        self.results.append(result)
        return result


def test_live_reader_uses_actual_rgb_serializes_current_result_and_resets_step():
    agent = make_agent()
    inventory = set(agent.state_dict())
    reader = RecordingReader()
    agent.screen_text_reader = reader
    frames = [np.full((28, 28, 3), index, dtype=np.uint8) for index in (17, 99)]
    frames[0].setflags(write=False)
    decisions = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for frame in frames:
            decisions.append(agent.act(frame, deterministic=True))
    assert [result.step for result in reader.results] == [0, 1]
    assert agent.last_screen_text_result is reader.results[-1]
    assert set(agent.state_dict()) == inventory
    for frame, observed in zip(frames, reader.frames, strict=True):
        np.testing.assert_array_equal(frame, observed)
    patches = agent.encode_frames(torch.from_numpy(np.stack(frames)))
    replay = agent.evaluate_episode(
        patches, [decision.action for decision in decisions],
        screen_text=[format_model_text(result) for result in reader.results],
    )
    torch.testing.assert_close(replay["value"], torch.tensor([d.value for d in decisions]),
                               rtol=2e-5, atol=3e-5)
    torch.testing.assert_close(replay["log_prob"], torch.tensor([d.log_prob for d in decisions]),
                               rtol=2e-5, atol=3e-5)
    with pytest.raises(ValueError, match="cannot be combined"):
        agent.act(frames[0], screen_text="external")
    assert len(reader.frames) == 2
    agent.reset_episode()
    assert agent.last_screen_text_result is None and agent._screen_text_step == 0
    agent.act(frames[0])
    assert reader.results[-1].step == 0
    agent.act_from_patches(patches[0], screen_text="explicit patch text")
    assert len(reader.frames) == 3  # A patch-only call never invokes a live reader.


def test_reader_failure_propagates_without_advancing_episode():
    class BrokenReader:
        def read(self, frame, *, step):
            raise OSError("OCR failed")

    agent = make_agent()
    agent.screen_text_reader = BrokenReader()
    agent.reset_episode()
    state = agent._episode_state
    with pytest.raises(OSError, match="OCR failed"):
        agent.act(np.zeros((28, 28, 3), dtype=np.uint8))
    assert agent._episode_state is state
    assert agent._screen_text_step == 0 and agent.last_screen_text_result is None


@pytest.mark.parametrize("texts", [[None], "text", [None, 42]])
def test_invalid_text_sequence_is_rejected(texts):
    agent = make_agent()
    with pytest.raises(ValueError, match="screen_text"):
        agent.evaluate_episode(torch.randn(2, 16, 384), [PhysicalAction()] * 2,
                               screen_text=texts)


def test_nonempty_ocr_requires_encoder_and_malformed_tokens_fail_closed():
    with pytest.raises(ValueError, match="enabled text_encoder"):
        make_agent(enabled=False).encode_screen_text("visible")
    agent = make_agent()
    for value in (torch.zeros(384), torch.zeros(2, 383),
                  torch.full((2, 384), float("nan")), torch.empty(0, 384),
                  torch.zeros(513, 384)):
        agent.language.encode_tokens = lambda text, value=value: value
        with pytest.raises(RuntimeError, match="finite"):
            agent.encode_screen_text("visible")



def test_ocr_character_limit_rejects_before_tokenization_or_cache():
    language = make_language()
    with pytest.raises(ValueError, match="characters"):
        language.encode_tokens("x" * 8193)
    assert not language.tokenizer.calls and not language._token_cache
    assert language.backbone.calls == 0


@pytest.mark.parametrize("method", ["act_from_patches", "peek_value_from_patches"])
def test_live_reader_cannot_silently_drop_ocr_on_patch_only_path(method):
    agent = make_agent()
    agent.screen_text_reader = RecordingReader()
    with pytest.raises(ValueError, match="require explicit screen_text"):
        getattr(agent, method)(torch.randn(16, 384))
    assert not agent.screen_text_reader.frames
    assert agent._screen_text_step == 0


@pytest.mark.parametrize("changed", [
    {"frame_sha256": "wrong-frame"}, {"source_width": 99}, {"source_height": 99}, {"step": 1},
])
def test_live_reader_rejects_result_from_another_frame_or_step(changed):
    class StaleReader(RecordingReader):
        def read(self, frame, *, step):
            return replace(super().read(frame, step=step), **changed)

    agent = make_agent()
    agent.screen_text_reader = StaleReader()
    agent.reset_episode()
    state = agent._episode_state
    with pytest.raises(ValueError, match="does not match the current RGB"):
        agent.act(np.zeros((28, 28, 3), dtype=np.uint8))
    assert agent._episode_state is state
    assert agent.last_screen_text_result is None and agent._screen_text_step == 0
