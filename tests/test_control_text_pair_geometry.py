"""Pair geometry is lexical input to a learned classifier, never an alias rule."""

import math

import pytest
import torch
from test_control_text_grounding import ResourceEncoder, calibration, passing_report

from carryrl.rewards.control_text import (
    PAIR_GEOMETRY_HEAD_ARCHITECTURE,
    LocalGroundedTextControlHead,
    PairGeometryTextControlHead,
    lexical_pair_geometry,
    load_text_control_reward,
    save_text_control_artifact,
    text_control_head_for_architecture,
)


def anchor_layouts(device="cpu"):
    grounding = torch.zeros(2, 36, 128, dtype=torch.bool, device=device)
    grounding[:, 0, [5, 60]] = True
    grounding[0, 1, [5, 66]] = True
    grounding[1, 1, [5, 110]] = True
    return grounding


def test_eight_symmetric_features_shared_header_exclusive_distance_and_absence():
    grounding = anchor_layouts()
    geometry = lexical_pair_geometry(grounding)
    assert geometry.shape == (2, 36, 36, 8)
    assert not geometry.requires_grad and torch.isfinite(geometry).all()
    assert torch.equal(geometry, geometry.transpose(1, 2))
    assert torch.equal(geometry[:1], lexical_pair_geometry(grounding[:1]))
    pair = geometry[:, 0, 1]
    assert (pair[:, :2] == 1).all()
    assert (pair[:, 2] == 0).all()  # both mention the same header token
    torch.testing.assert_close(
        pair[:, 3], torch.tensor([math.log1p(6), math.log1p(50)]) / math.log1p(512)
    )
    torch.testing.assert_close(pair[:, 5], torch.tensor([1 / 3, 1 / 3]))
    assert (geometry[:, 2, 3, :2] == 0).all()
    assert (geometry[:, 2, 3, 2:4] == 1).all()  # missing sentinel, not zero distance
    assert (geometry[:, 2, 3, 4:] == 0).all()
    shifted = torch.nn.functional.pad(grounding, (7, 0))
    torch.testing.assert_close(geometry, lexical_pair_geometry(shifted), rtol=0, atol=0)


def test_zero_extension_preserves_local_readout_and_only_adds_512_parameters():
    torch.manual_seed(20261022)
    local = LocalGroundedTextControlHead().eval()
    torch.manual_seed(20261022)
    paired = PairGeometryTextControlHead().eval()
    assert (
        sum(p.numel() for p in paired.parameters()) - sum(p.numel() for p in local.parameters())
        == 512
    )
    for name, value in local.state_dict().items():
        actual = paired.state_dict()[name]
        if name == "pair.0.weight":
            torch.testing.assert_close(actual[:, :256], value, atol=0, rtol=0)
            assert not actual[:, 256:].any()
        else:
            torch.testing.assert_close(actual, value, atol=0, rtol=0)
    features = torch.randn(1, 36, 128).expand(2, -1, -1)
    old = local._readout(features)
    new = paired._grounded_readout(features, anchor_layouts())
    for kind in old:
        torch.testing.assert_close(old[kind], new[kind], atol=0, rtol=0)


def test_same_token_vectors_different_anchor_geometry_can_be_learned_in_either_direction():
    head = PairGeometryTextControlHead().eval()
    features = torch.zeros(2, 36, 128)
    grounding = anchor_layouts()
    old_head = LocalGroundedTextControlHead().eval()
    old = old_head._readout(features)
    torch.testing.assert_close(old["pair"][0], old["pair"][1], rtol=0, atol=0)
    # Demonstrate expressivity with specified classifier weights, without a fit:
    # close can be different, far can be same. Geometry returns no class labels.
    with torch.no_grad():
        for layer in (head.pair[0], head.pair[-1]):
            layer.weight.zero_()
            layer.bias.zero_()
        head.pair[0].weight[0, 256 + 3] = 1
        values = torch.nn.functional.gelu(lexical_pair_geometry(grounding)[:, 0, 1, 3])
        midpoint = values.mean()
        head.pair[-1].weight[0, 0] = 1
        head.pair[-1].weight[1, 0] = -1
        head.pair[-1].bias[0] = -midpoint
        head.pair[-1].bias[1] = midpoint
        head.pair[-1].bias[2] = -10
    output = head._grounded_readout(features, grounding)
    assert output["pair"][:, 0, 1].argmax(-1).tolist() == [1, 0]
    assert torch.equal(output["pair"], output["pair"].transpose(1, 2))
    assert not torch.equal(output["pair"][0, 0, 1], output["pair"][1, 0, 1])
    torch.testing.assert_close(output["status"][0], output["status"][1])
    torch.testing.assert_close(output["mode"][0], output["mode"][1])
    # Full-head check: every contextual token vector is exactly identical, so
    # only lexical positions can distinguish these two layouts.
    tokens = torch.ones(2, 128, 384)
    legacy = old_head(tokens, grounding=grounding)
    torch.testing.assert_close(legacy["pair"][0], legacy["pair"][1], atol=1e-6, rtol=1e-5)
    full = head(tokens, grounding=grounding)
    assert full["pair"][:, 0, 1].argmax(-1).tolist() == [1, 0]


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_full_forward_padding_symmetry_and_backend(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    head = PairGeometryTextControlHead().eval().to(device)
    with torch.no_grad():
        head.pair[0].weight[:, 256:].normal_(std=0.1)
    grounding = anchor_layouts(device)
    geometry = lexical_pair_geometry(grounding)
    assert torch.equal(geometry, geometry.transpose(1, 2))
    assert torch.isfinite(geometry).all() and not geometry.requires_grad
    tokens = torch.randn(2, 128, 384, device=device)
    original = head(tokens, grounding=grounding)
    assert torch.equal(original["pair"], original["pair"].transpose(1, 2))
    padded = torch.nn.functional.pad(grounding, (0, 5), value=True)
    pad_mask = torch.tensor([[False] * 128 + [True] * 5] * 2, device=device)
    changed = head(
        torch.cat((tokens, torch.randn(2, 5, 384, device=device)), dim=1),
        pad_mask,
        grounding=padded,
    )
    for kind in original:
        assert torch.isfinite(changed[kind]).all()
        torch.testing.assert_close(original[kind], changed[kind], atol=1e-5, rtol=1e-4)


def test_geometry_artifact_factory_save_load_export_and_runtime(tmp_path):
    encoder = ResourceEncoder()
    head = text_control_head_for_architecture(PAIR_GEOMETRY_HEAD_ARCHITECTURE)
    assert type(head) is PairGeometryTextControlHead
    directory = tmp_path / "geometry"
    manifest = save_text_control_artifact(
        directory,
        encoder=encoder,
        head=head,
        calibration=calibration(),
        validation=passing_report(),
        training={"trained": True},
    )
    assert manifest["head_architecture"] == PAIR_GEOMETRY_HEAD_ARCHITECTURE
    restored = load_text_control_reward(directory, shared_encoder=encoder)
    assert type(restored.head) is PairGeometryTextControlHead
    expected = restored.predict("Hold A; press the D key")
    exported = restored.save_resources(tmp_path / "exported")
    again = load_text_control_reward(exported, shared_encoder=encoder)
    assert again.model_hash == restored.model_hash
    assert again.predict("Hold A; press the D key") == expected
