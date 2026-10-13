"""Offline instruction conditioning, frozen encoding, and replay contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.language import (
    INSTRUCTION_MAX_CHARS,
    TEXT_ID,
    TEXT_REVISION,
    MultilingualInstructionEncoder,
)
from carryrl.agent.model import AgentConfig, GeneralistAgent


class TinyTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        ids = [1, *[ord(char) % 251 for char in text], 2, 0]
        return {"input_ids": torch.tensor([ids]),
                "attention_mask": torch.tensor([[1] * (len(ids) - 1) + [0]])}


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=384)
        self.embedding = nn.Embedding(251, 384)
        self.dropout = nn.Dropout(0.9)
        self.calls = 0

    def forward(self, input_ids, attention_mask, return_dict):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))


class TinyVision(nn.Module):
    def forward(self, frames):
        return torch.zeros(len(frames), 16, 384, device=frames.device)


def make_language(**kwargs):
    return MultilingualInstructionEncoder(tokenizer=TinyTokenizer(), backbone=TinyBackbone(),
                                          **kwargs)


def make_agent(*, enabled=True):
    torch.manual_seed(71)
    config = AgentConfig(pretrained=False, image_size=28, width=32, heads=4, layers=2,
                         spatial_tokens=4, memory_tokens=2, segment_length=4,
                         text_encoder="multilingual_e5_small" if enabled else "none")
    return GeneralistAgent(config, vision=TinyVision(),
                           language=make_language() if enabled else None)


@pytest.mark.parametrize("instruction", ["Move left", "Двигайся влево", "向左移动"])
def test_masked_pooling_normalization_query_prefix_and_cpu_frozen_cache(instruction):
    language = make_language()
    language.train()
    assert not language.training and not language.backbone.training
    feature = language(instruction)
    text, kwargs = language.tokenizer.calls[0]
    assert text == "query: " + instruction
    assert kwargs["truncation"] is False
    tokens = language.tokenizer(text)
    hidden = language.backbone.embedding(tokens["input_ids"])
    mask = tokens["attention_mask"].bool()
    expected = F.normalize(hidden.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1)[:, None],
                           p=2, dim=1)[0]
    torch.testing.assert_close(feature, expected)
    assert feature.device.type == "cpu" and not feature.requires_grad
    assert all(not p.requires_grad for p in language.parameters())
    feature.zero_()
    torch.testing.assert_close(language(instruction), expected)
    assert language.backbone.calls == 1


def test_language_cache_is_bounded_and_invalidated_on_weights_or_dtype_changes():
    language = make_language(cache_size=2)
    language("first")
    language("second")
    language("first")
    language("third")
    assert list(language._cache) == ["first", "third"]
    language.load_state_dict(language.state_dict())
    assert not language._cache
    language("first")
    language.to(dtype=torch.float64)
    assert not language._cache
    assert language("first").dtype == torch.float64


def test_collection_replay_matches_same_instruction_across_memory_boundaries():
    agent = make_agent()
    patches = torch.randn(10, 16, 384)
    agent.reset_episode(instruction="Нажми W и двигайся вперёд")
    decisions = [agent.act_from_patches(patch) for patch in patches]
    replay = agent.evaluate_episode(patches, [d.action for d in decisions],
                                    instruction="Нажми W и двигайся вперёд")
    torch.testing.assert_close(replay["log_prob"], torch.tensor([d.log_prob for d in decisions]),
                               rtol=2e-5, atol=3e-5)
    torch.testing.assert_close(replay["value"], torch.tensor([d.value for d in decisions]),
                               rtol=2e-5, atol=3e-5)
    assert agent.language.backbone.calls == 1


def test_counterfactual_instruction_changes_likelihoods_past_memory_boundaries():
    agent = make_agent()
    patches = torch.randn(10, 16, 384)
    actions = [PhysicalAction(keys=("w",))] * len(patches)
    left = agent.evaluate_episode(patches, actions, instruction="Move left")
    right = agent.evaluate_episode(patches, actions, instruction="Двигайся вправо")
    assert not torch.allclose(left["log_prob"], right["log_prob"])
    assert not torch.allclose(left["log_prob"][8:], right["log_prob"][8:])


def test_replay_and_peek_do_not_change_active_prompt_or_temporal_state():
    agent = make_agent()
    patches = torch.randn(3, 16, 384)
    agent.reset_episode(instruction="Move left")
    first = agent.act_from_patches(patches[0], deterministic=True)
    state, previous = agent._episode_state, agent._previous_action
    feature = agent._instruction_features
    peek = agent.peek_value_from_patches(patches[1])
    agent.evaluate_episode(patches, [PhysicalAction()] * 3, instruction="Move right")
    assert agent._episode_state is state
    assert agent._previous_action is previous
    assert agent._instruction_features is feature
    assert agent._instruction == "Move left"
    assert agent.peek_value_from_patches(patches[1]) == peek
    assert agent.act_from_patches(patches[1], deterministic=True).value == peek
    agent.reset_episode(instruction="Move left")
    assert agent.act_from_patches(patches[0], deterministic=True) == first
    agent.reset_episode()
    assert agent._instruction is None and agent._instruction_features is None
    clean = agent.act_from_patches(patches[0], deterministic=True)
    replay = agent.evaluate_episode(patches[:1], [clean.action])
    torch.testing.assert_close(replay["log_prob"], torch.tensor([clean.log_prob]))


def test_updates_recompute_projection_and_preserve_frozen_encoder():
    agent = make_agent().train()
    assert not agent.language.training
    frozen = {name: value.clone() for name, value in agent.language.state_dict().items()}
    projection = agent.instruction_projection.weight.detach().clone()
    patches, actions = torch.randn(5, 16, 384), [PhysicalAction()] * 5
    optimizer = torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad), lr=0.01)
    before = agent.evaluate_episode(patches, actions, instruction="Collect coins")
    optimizer.zero_grad()
    (before["value"].square().mean() - before["log_prob"].mean()).backward()
    assert agent.instruction_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in agent.language.parameters())
    optimizer.step()
    after = agent.evaluate_episode(patches, actions, instruction="Collect coins")
    assert agent.language.backbone.calls == 1
    assert not torch.equal(projection, agent.instruction_projection.weight)
    assert not torch.allclose(before["log_prob"], after["log_prob"])
    for name, value in agent.language.state_dict().items():
        torch.testing.assert_close(value, frozen[name], atol=0, rtol=0)
    assert all(not v.requires_grad and v.device.type == "cpu"
               for v in agent.language._cache.values())


def test_parent_state_loading_invalidates_language_cache_and_active_features():
    agent = make_agent()
    agent.reset_episode(instruction="Move left")
    agent.load_state_dict(agent.state_dict())
    assert not agent.language._cache
    assert agent._instruction_features is None and agent._episode_state is None
    agent.act_from_patches(torch.randn(16, 384))
    assert agent.language.backbone.calls == 2


def test_disabled_architecture_has_no_language_parameters_and_strict_old_config_roundtrip():
    original = make_agent(enabled=False)
    old_config = original.config.to_dict()
    del old_config["text_encoder"]
    del old_config["instruction_max_tokens"]
    restored = GeneralistAgent(AgentConfig.from_dict(old_config), vision=TinyVision())
    restored.load_state_dict(original.state_dict(), strict=True)
    assert not any("instruction" in name or "language" in name for name in original.state_dict())
    patches, actions = torch.randn(5, 16, 384), [PhysicalAction()] * 5
    before = original.evaluate_episode(patches, actions)
    after = restored.evaluate_episode(patches, actions)
    for name in before:
        torch.testing.assert_close(before[name], after[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match="enabled text_encoder"):
        original.reset_episode(instruction="Move left")
    with pytest.raises(ValueError, match="enabled text_encoder"):
        original.evaluate_episode(patches, actions, instruction="Move left")
    with pytest.raises(ValueError, match="injected language"):
        GeneralistAgent(original.config, vision=TinyVision(), language=make_language())


@pytest.mark.parametrize("value", [1, 513, True, 3.5, "512"])
def test_invalid_instruction_token_limits_rejected(value):
    with pytest.raises(ValueError, match="between 2 and 512"):
        AgentConfig(instruction_max_tokens=value)
    with pytest.raises(ValueError, match="between 2 and 512"):
        make_language(max_tokens=value)


def test_overlong_instruction_is_rejected_before_encoding_without_truncation():
    language = make_language(max_tokens=20)
    with pytest.raises(ValueError, match="never truncated"):
        language("a" * 30)
    assert language.backbone.calls == 0 and not language._cache
    with pytest.raises(ValueError, match="characters"):
        language("a" * (INSTRUCTION_MAX_CHARS + 1))
    assert len(language.tokenizer.calls) == 1
    with pytest.raises(ValueError, match="string or None"):
        language(42)


def test_pinned_loading_uses_safetensors_without_remote_code(monkeypatch):
    calls = {}

    def tokenizer(identifier, **kwargs):
        calls["tokenizer"] = (identifier, kwargs)
        return TinyTokenizer()

    def model(identifier, **kwargs):
        calls["model"] = (identifier, kwargs)
        return TinyBackbone()

    monkeypatch.setattr("carryrl.agent.language.AutoTokenizer.from_pretrained", tokenizer)
    monkeypatch.setattr("carryrl.agent.language.AutoModel.from_pretrained", model)
    MultilingualInstructionEncoder(local_files_only=True)
    for identifier, kwargs in calls.values():
        assert identifier == TEXT_ID and kwargs["revision"] == TEXT_REVISION
        assert kwargs["trust_remote_code"] is False and kwargs["local_files_only"] is True
    assert calls["model"][1]["use_safetensors"] is True


@pytest.mark.parametrize("component", ["AutoTokenizer", "AutoModel"])
def test_loading_failure_propagates_without_random_fallback(monkeypatch, component):
    monkeypatch.setattr("carryrl.agent.language.AutoTokenizer.from_pretrained",
                        lambda *a, **k: TinyTokenizer())

    def fail(*args, **kwargs):
        raise OSError("offline checkpoint missing")

    monkeypatch.setattr(f"carryrl.agent.language.{component}.from_pretrained", fail)
    with pytest.raises(OSError, match="offline checkpoint missing"):
        MultilingualInstructionEncoder(local_files_only=True)


def test_local_resource_roundtrip_requires_strict_weight_restoration(tmp_path, monkeypatch):
    class LocalTokenizer(TinyTokenizer):
        def save_pretrained(self, directory):
            (directory / "tokenizer.fixture").write_text("offline", encoding="utf-8")

    calls = []

    def local_tokenizer(directory, **kwargs):
        assert (directory / "tokenizer.fixture").read_text(encoding="utf-8") == "offline"
        calls.append(kwargs)
        return LocalTokenizer()

    monkeypatch.setattr("carryrl.agent.language.AutoTokenizer.from_pretrained", local_tokenizer)
    original = MultilingualInstructionEncoder(
        tokenizer=LocalTokenizer(), backbone=BertModel(BertConfig(
            hidden_size=384, num_hidden_layers=1, num_attention_heads=4,
            intermediate_size=48, vocab_size=251, max_position_embeddings=512,
        )),
    )
    original.save_resources(tmp_path)
    restored = MultilingualInstructionEncoder.from_resources(tmp_path)
    assert calls == [{"local_files_only": True, "trust_remote_code": False}]
    with pytest.raises(RuntimeError, match="strict checkpoint"):
        restored("Двигайся влево")
    incomplete = original.state_dict().copy()
    incomplete.pop(next(iter(incomplete)))
    restored.load_state_dict(incomplete, strict=False)
    with pytest.raises(RuntimeError, match="strict checkpoint"):
        restored("Двигайся влево")
    restored.load_state_dict(original.state_dict(), strict=True)
    torch.testing.assert_close(restored("Двигайся влево"), original("Двигайся влево"),
                               atol=0, rtol=0)
    (tmp_path / "carryrl_language.json").write_text('{"revision": "wrong"}', encoding="utf-8")
    with pytest.raises(ValueError, match="identity/revision"):
        MultilingualInstructionEncoder.from_resources(tmp_path)


def test_actor_has_direct_instruction_access_with_identical_temporal_states():
    agent = make_agent()
    states, spatial = torch.randn(2, 32), torch.randn(2, 4, 32)
    left = agent._instruction_context(agent._encode_instruction("Move left"))
    right = agent._instruction_context(agent._encode_instruction("Move right"))
    assert not torch.allclose(agent._actor_features(states, spatial, left),
                              agent._actor_features(states, spatial, right))


@pytest.mark.parametrize("value", ["unknown", None, [], 1])
def test_invalid_text_encoder_config_is_rejected(value):
    with pytest.raises(ValueError, match="text_encoder"):
        AgentConfig(text_encoder=value)
