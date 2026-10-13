"""Offline checks for complete-frame geometry and causal recurrent memory."""

import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from carryrl.agent.memory import TemporalCore
from carryrl.agent.vision import (
    DINO_ID,
    DINO_PARAMETER_COUNT,
    DINO_REVISION,
    DinoVision,
    SpatialResampler,
)


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_vision(*, frozen=True):
    config = Dinov2Config(
        hidden_size=24, num_hidden_layers=2, num_attention_heads=4,
        image_size=28, patch_size=14, mlp_ratio=2,
    )
    return DinoVision(pretrained=False, frozen=frozen, image_size=28, config=config)


def tiny_core():
    torch.manual_seed(13)
    return TemporalCore(width=16, layers=2, heads=4, segment_length=3, memory_tokens=2)


def test_letterboxing_preserves_both_image_edges_and_normalizes_mean_padding():
    vision = tiny_vision()
    pixels = torch.zeros(1, 7, 14, 3, dtype=torch.uint8)
    pixels[:, :, :3, 0] = 255
    pixels[:, :, -3:, 1] = 255
    normalized = vision.preprocess(pixels)
    assert normalized.shape == (1, 3, 28, 28)
    assert torch.count_nonzero(normalized[:, :, :7]) == 0
    assert torch.count_nonzero(normalized[:, :, 21:]) == 0
    restored = normalized * vision.pixel_std + vision.pixel_mean
    assert restored[0, 0, 14, 0] > 0.99
    assert restored[0, 1, 14, -1] > 0.99
    patches = vision(pixels)
    assert patches.shape == (1, 4, 24)
    with torch.no_grad():
        raw = vision.backbone(pixel_values=normalized).last_hidden_state
    torch.testing.assert_close(patches, raw[:, 1:])


def test_frozen_backbone_stays_eval_but_resampler_trains_and_vision_can_unfreeze():
    vision = tiny_vision()
    resampler = SpatialResampler(width=16, tokens=3, input_width=24)
    vision.train()
    assert vision.training and not vision.backbone.training
    assert all(not p.requires_grad for p in vision.backbone.parameters())
    pixels = torch.randint(256, (2, 17, 31, 3), dtype=torch.uint8)
    patches = vision(pixels)
    assert not patches.requires_grad
    output = resampler(patches)
    assert output.shape == (2, 3, 16)
    output[:, :, 0].square().mean().backward()
    assert resampler.queries.grad is not None
    assert resampler.queries.grad.abs().sum() > 0
    assert resampler.position_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in vision.backbone.parameters())
    vision.set_frozen(False)
    assert vision.backbone.training
    patches = vision(pixels)
    patches[:, :, 0].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in vision.backbone.parameters())


def test_scratch_uses_pinned_small_graph_without_a_download(monkeypatch):
    def no_download(*args, **kwargs):
        raise AssertionError("scratch construction must not load a checkpoint")

    monkeypatch.setattr(Dinov2Model, "from_pretrained", no_download)
    vision = DinoVision(pretrained=False)
    assert sum(p.numel() for p in vision.backbone.parameters()) == DINO_PARAMETER_COUNT
    assert vision.hidden_size == 384 and vision.patch_grid == (16, 16)


def test_cached_loading_is_pinned_and_failure_does_not_fall_back(monkeypatch):
    calls = []

    def absent_checkpoint(model_id, **kwargs):
        calls.append((model_id, kwargs))
        raise OSError("weights missing from the local cache")

    monkeypatch.setattr(Dinov2Model, "from_pretrained", absent_checkpoint)
    with pytest.raises(OSError, match="weights missing"):
        DinoVision.from_cached()
    assert calls[0][0] == DINO_ID
    assert calls[0][1]["revision"] == DINO_REVISION
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["use_safetensors"] is True


@pytest.mark.parametrize("pixels, exception", [
    (torch.zeros(1, 2, 2, 3), TypeError),
    (torch.zeros(2, 2, 3, dtype=torch.uint8), ValueError),
    (torch.zeros(1, 2, 2, 4, dtype=torch.uint8), ValueError),
    (torch.zeros(0, 2, 2, 3, dtype=torch.uint8), ValueError),
])
def test_vision_rejects_ambiguous_or_empty_rgb(pixels, exception):
    with pytest.raises(exception):
        tiny_vision()(pixels)


def test_resampler_is_sensitive_to_patch_location():
    torch.manual_seed(2)
    resampler = SpatialResampler(width=16, tokens=3, input_width=24)
    patches = torch.randn(1, 4, 24)
    forward = resampler(patches)
    swapped = resampler(patches.flip(1))
    assert not torch.allclose(forward, swapped, atol=1e-5)
    with pytest.raises(ValueError, match="square"):
        resampler(torch.randn(1, 3, 24))


@pytest.mark.parametrize("changed_at", [1, 3, 6])
def test_future_changes_cannot_affect_previous_decisions_even_across_segments(changed_at):
    core = tiny_core().eval()
    sequence = torch.randn(9, 16)
    changed = sequence.clone()
    changed[changed_at:] = torch.randn_like(changed[changed_at:]) * 17
    with torch.no_grad():
        baseline = core.forward_sequence(sequence)
        altered = core.forward_sequence(changed)
    torch.testing.assert_close(baseline[:changed_at], altered[:changed_at], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(baseline[changed_at:], altered[changed_at:])


def test_streaming_matches_vectorized_sequence_and_bounds_cache_across_segments():
    core = tiny_core().eval()
    sequence = torch.randn(10, 16)
    with torch.no_grad():
        expected = core.forward_sequence(sequence)
        state = core.initial_state(device="cpu", dtype=torch.float32)
        streamed = []
        for index, summary in enumerate(sequence):
            old_state = state
            old_keys = tuple(cache.keys.clone() for cache in state.layer_caches)
            output, state = core.step(summary, state)
            streamed.append(output)
            assert state.position == (index + 1) % core.segment_length
            for cache in state.layer_caches:
                assert cache.keys.shape[1] == core.memory_tokens + state.position
            for cache, before in zip(old_state.layer_caches, old_keys, strict=True):
                torch.testing.assert_close(cache.keys, before, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(streamed), expected, rtol=1e-5, atol=2e-6)


def test_episode_reset_discards_history_and_detach_cuts_rollout_graph():
    core = tiny_core()
    sequence = torch.randn(7, 16)
    state = core.initial_state()
    first, _ = core.step(sequence[0], state)
    for summary in sequence:
        _, state = core.step(summary, state)
    fresh_first, _ = core.step(sequence[0], core.initial_state())
    torch.testing.assert_close(first, fresh_first, rtol=0, atol=0)
    detached = state.detach()
    assert not detached.memory.requires_grad
    assert all(not cache.keys.requires_grad and not cache.values.requires_grad
               for cache in detached.layer_caches)
    assert detached.position == state.position


def test_later_segment_loss_backpropagates_through_memory_write_to_earlier_inputs():
    core = tiny_core()
    sequence = torch.randn(7, 16, requires_grad=True)
    states = core.forward_sequence(sequence)
    states[-1, 0].backward()
    assert torch.isfinite(sequence.grad).all()
    assert sequence.grad[:3].abs().sum() > 1e-8
    assert core.write_tokens.grad is not None and core.write_tokens.grad.abs().sum() > 1e-8
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in core.parameters())


def test_default_temporal_width_backward_is_finite():
    core = TemporalCore(segment_length=2)
    sequence = torch.randn(5, 256, requires_grad=True)
    loss = core.forward_sequence(sequence)[:, 0].square().mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(sequence.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in core.parameters())


def test_core_rejects_incompatible_shapes_and_state_requests():
    core = tiny_core()
    with pytest.raises(ValueError, match="summary"):
        core.forward_sequence(torch.zeros(2, 17))
    with pytest.raises(ValueError, match="floating dtype"):
        core.forward_sequence(torch.zeros(2, 16, dtype=torch.int64))
    with pytest.raises(ValueError, match="exactly one"):
        core.step(torch.zeros(2, 16), core.initial_state())
    with pytest.raises(ValueError, match="dtype"):
        core.initial_state(dtype=torch.float64)
    assert core.forward_sequence(torch.empty(0, 16)).shape == (0, 16)
