"""Fixed local-context architecture checks; no optimization or model fitting."""

import pytest
import torch

from carryrl.rewards.control_text import (
    GROUNDED_HEAD_ARCHITECTURE,
    LOCAL_GROUNDED_HEAD_ARCHITECTURE,
    GroundedTextControlHead,
    LocalGroundedTextControlHead,
    encoder_state_hash,
    text_control_head_for_architecture,
)


def heads_with_identical_state():
    torch.manual_seed(20261021)
    original = GroundedTextControlHead().eval()
    torch.manual_seed(20261021)
    local = LocalGroundedTextControlHead().eval()
    return original, local


def remove_global_attention(head):
    # Isolate local attention: real E5 vectors are already contextual, and the
    # global branch otherwise lets any token change every candidate's output.
    with torch.no_grad():
        head.attention.out_proj.weight.zero_()
        head.attention.out_proj.bias.zero_()


def test_identical_initial_state_keys_shapes_values_and_parameter_count():
    original, local = heads_with_identical_state()
    assert original.state_dict().keys() == local.state_dict().keys()
    assert encoder_state_hash(original.state_dict()) == encoder_state_hash(local.state_dict())
    assert sum(p.numel() for p in original.parameters()) == sum(
        p.numel() for p in local.parameters()
    )
    local.load_state_dict(original.state_dict(), strict=True)
    assert (
        type(text_control_head_for_architecture(GROUNDED_HEAD_ARCHITECTURE))
        is GroundedTextControlHead
    )
    assert (
        type(text_control_head_for_architecture(LOCAL_GROUNDED_HEAD_ARCHITECTURE))
        is LocalGroundedTextControlHead
    )


def test_neighbor_changes_v4_but_strict_v3_does_not_change():
    original, local = heads_with_identical_state()
    for head in (original, local):
        remove_global_attention(head)
    tokens = torch.randn(1, 64, 384)
    grounding = torch.zeros(1, 36, 64, dtype=torch.bool)
    grounding[:, 0, 16] = True
    neighbor = tokens.clone()
    neighbor[:, 20] = torch.randn_like(neighbor[:, 20]) * 5
    original_before = original(tokens, grounding=grounding)
    original_after = original(neighbor, grounding=grounding)
    local_before = local(tokens, grounding=grounding)
    local_after = local(neighbor, grounding=grounding)
    for kind in original_before:
        torch.testing.assert_close(original_before[kind], original_after[kind])
    assert not torch.allclose(local_before["status"][:, 0], local_after["status"][:, 0])
    torch.testing.assert_close(local_before["pair"], local_before["pair"].transpose(1, 2))


def test_far_tokens_and_padding_do_not_change_local_attention():
    _, head = heads_with_identical_state()
    remove_global_attention(head)
    tokens = torch.randn(1, 64, 384)
    grounding = torch.zeros(1, 36, 64, dtype=torch.bool)
    grounding[:, 0, 16] = True
    original = head(tokens, grounding=grounding)
    far = tokens.clone()
    far[:, 41:] = torch.randn_like(far[:, 41:]) * 10
    changed = head(far, grounding=grounding)
    for kind in original:
        torch.testing.assert_close(original[kind], changed[kind])
    padded_tokens = torch.cat((tokens, torch.randn(1, 9, 384) * 10), dim=1)
    padded_grounding = torch.nn.functional.pad(grounding, (0, 9))
    # Even stray supplied anchors in padding cannot affect a candidate.
    padded_grounding[:, :, 64:] = True
    padded = head(
        padded_tokens, torch.tensor([[False] * 64 + [True] * 9]), grounding=padded_grounding
    )
    for kind in original:
        torch.testing.assert_close(original[kind], padded[kind], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_local_bias_shape_nearest_distance_window_slopes_and_missing_fallback(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    head = LocalGroundedTextControlHead().to(device).eval()
    grounding = torch.zeros(2, 36, 64, dtype=torch.bool, device=device)
    grounding[0, 0, 12] = True
    grounding[0, 0, 20] = True
    grounding[1, 1, 7] = True
    padding = torch.zeros(2, 64, dtype=torch.bool, device=device)
    padding[:, 62:] = True
    bias = head._mention_attention_mask(grounding, padding, dtype=torch.float32)
    assert bias.shape == (2 * 4, 36, 64)
    assert bias.dtype == torch.float32 and bias.device.type == device
    shaped = bias.reshape(2, 4, 36, 64).cpu()
    slopes = torch.tensor([1 / 3, 1 / 6, 1 / 12, 1 / 24])
    torch.testing.assert_close(shaped[0, :, 0, 14], -2 * slopes)
    # At token 19 the second mention is nearest (distance 1, not 7).
    torch.testing.assert_close(shaped[0, :, 0, 19], -slopes)
    torch.testing.assert_close(shaped[1, :, 1, 9], -2 * slopes)
    assert (shaped[0, :, 0, 20] == 0).all()
    assert torch.isfinite(shaped[0, :, 0, 44]).all()  # radius 24 included
    assert torch.isneginf(shaped[0, :, 0, 45]).all()  # radius 25 excluded
    assert torch.isneginf(shaped[..., 62:]).all()
    assert (torch.isfinite(shaped[0, :, 2]).sum(-1) == 1).all()
    assert (shaped[0, :, 2, 0] == 0).all()
    captured = []
    handle = head.fusion.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    output = head(torch.randn(2, 64, 384, device=device), padding, grounding=grounding)
    handle.remove()
    assert all(torch.isfinite(values).all() for values in output.values())
    assert not captured[0][0, 2, 128:256].any()  # missing mention output remains zero
