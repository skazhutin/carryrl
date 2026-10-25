"""Technical regressions for the single fresh-head repair; no production fits."""

import hashlib

import pytest
import torch
from torch import nn

from carryrl.rewards.control_text import (
    CANDIDATES,
    ResidualTextControlHead,
    TextControlHead,
    encoder_state_hash,
)
from carryrl.rewards.control_text_training import (
    _targets,
    class_balance_from_train,
    fit_text_control_head,
    head_loss,
    load_verified_disk_token_features,
    prepare_disk_token_features,
)


def labeled_row(split="train"):
    labels = [
        {"candidate": candidate, "status": "unsupported", "mode": "unknown", "alias_group": 0}
        for candidate in CANDIDATES
    ]
    for index in (0, 1):
        labels[index].update(status="supported", mode="held", alias_group=1)
    labels[2].update(status="supported", mode="press", alias_group=2)
    labels[-1].update(status="unknown")
    return {"text": split, "split": split, "labels": labels}


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(384))
        self.texts = []

    def encode_instruction_tokens(self, text):
        self.texts.append(text)
        return self.weight.detach()[None].repeat(4, 1)


def test_balanced_rare_class_gradients_and_unique_pair_mask():
    row = labeled_row()
    balance = class_balance_from_train([row])
    assert balance["counts"] == {"status": [3, 32, 1], "mode": [2, 1, 33], "pair": [1, 594, 35]}
    logits = {
        "status": torch.zeros(1, 36, 3, requires_grad=True),
        "mode": torch.zeros(1, 36, 3, requires_grad=True),
        "pair": torch.zeros(1, 36, 36, 3, requires_grad=True),
    }
    head_loss(
        logits, _targets([row], "cpu"), class_weights=balance["weights"], unique_pairs=True
    ).backward()
    assert logits["status"].grad[0, -1].norm() > logits["status"].grad[0, 3].norm()
    assert logits["mode"].grad[0, 2].norm() > logits["mode"].grad[0, 3].norm()
    assert logits["pair"].grad[0, 0, 1].norm() > logits["pair"].grad[0, 0, 2].norm()
    assert not logits["pair"].grad[0].diagonal(dim1=0, dim2=1).any()
    assert not logits["pair"].grad[0, 1, 0].any()
    with pytest.raises(ValueError, match="TRAIN"):
        class_balance_from_train([labeled_row("dev")])


def test_candidate_identity_survives_single_token_and_padding():
    torch.manual_seed(11)
    tokens = torch.randn(1, 1, 384)
    legacy = TextControlHead().eval()(tokens)
    torch.testing.assert_close(legacy["status"][:, :1].expand(-1, 36, -1), legacy["status"])
    repaired = ResidualTextControlHead().eval()
    output = repaired(tokens)
    assert output["status"].std(dim=1).max() > 0.01
    torch.testing.assert_close(output["pair"], output["pair"].transpose(1, 2))
    padded = torch.cat((tokens, torch.randn(1, 3, 384)), dim=1)
    masked = repaired(padded, torch.tensor([[False, True, True, True]]))
    for kind in output:
        torch.testing.assert_close(masked[kind], output[kind], atol=1e-6, rtol=1e-5)


def test_fresh_balanced_fit_records_train_weights_dev_slices_and_empty_optimizer():
    from carryrl.rewards.control_text import RESIDUAL_HEAD_ARCHITECTURE

    torch.manual_seed(20261021)
    initial_hash = encoder_state_hash(ResidualTextControlHead().state_dict())
    encoder = TinyEncoder()
    rows = [labeled_row(split) for split in ("train", "dev", "calibration", "sealed_test")]
    _, training = fit_text_control_head(
        encoder, rows, seed=20261021, epochs=1, head_variant="residual_balanced_v2"
    )
    assert encoder.texts == ["train", "dev"]
    assert training["initial_trainable_sha256"] == initial_hash
    assert training["optimizer_started_empty"] and training["optimizer_initial_state_entries"] == 0
    assert training["imported_head_checkpoints"] == []
    assert training["head_architecture"] == RESIDUAL_HEAD_ARCHITECTURE
    assert training["selection"] == "dev_train_weighted_nll"
    assert training["class_balance"] == class_balance_from_train([rows[0]])
    slices = training["history"][0]["dev_metrics"]["slices"]
    assert slices["mode"]["press"]["target_count"] == 1
    assert slices["pair"]["same"]["target_count"] == 1


def test_verified_feature_reuse_is_read_only_and_rejects_identity_or_mask_drift(tmp_path):
    encoder = TinyEncoder()
    rows = [labeled_row(split) for split in ("train", "dev", "calibration", "sealed_test")]
    directory = tmp_path / "old_features"
    prepare_disk_token_features(encoder, rows, directory, dataset_sha256="d" * 64)
    (directory / "selected_head.pt").write_bytes(b"failed original head evidence")
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()
    }
    manifest = {"records_sha256": "d" * 64, "candidates": list(CANDIDATES)}
    features = load_verified_disk_token_features(
        directory, encoder=encoder, records=rows, dataset_manifest=manifest
    )
    assert not features.data.flags.writeable and features.verification["read_only"]
    assert before == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()
    }
    bad_manifest = {**manifest, "candidates": list(reversed(CANDIDATES))}
    with pytest.raises(ValueError, match="identity"):
        load_verified_disk_token_features(
            directory, encoder=encoder, records=rows, dataset_manifest=bad_manifest
        )
    with pytest.raises(ValueError, match="text set"):
        load_verified_disk_token_features(
            directory, encoder=encoder, records=rows[:-2], dataset_manifest=manifest
        )
    encoder.tokenizer = lambda texts, **kwargs: {"input_ids": [[1, 2] for _ in texts]}
    with pytest.raises(ValueError, match="instruction masks"):
        load_verified_disk_token_features(
            directory, encoder=encoder, records=rows, dataset_manifest=manifest
        )


def test_feature_reuse_rejects_swapped_equal_length_text_spans(tmp_path):
    import json

    encoder = TinyEncoder()
    rows = [labeled_row(split) for split in ("train", "dev", "calibration", "sealed_test")]
    directory = tmp_path / "features"
    prepare_disk_token_features(encoder, rows, directory, dataset_sha256="d" * 64)
    path = directory / "manifest.json"
    metadata = json.loads(path.read_text())
    left, right = list(metadata["rows"])[:2]
    assert metadata["rows"][left]["length"] == metadata["rows"][right]["length"]
    metadata["rows"][left], metadata["rows"][right] = (
        metadata["rows"][right], metadata["rows"][left]
    )
    # Byte hash and contiguous spans still match; text-to-span association does not.
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="text-to-feature offsets"):
        load_verified_disk_token_features(
            directory, encoder=encoder, records=rows,
            dataset_manifest={"records_sha256": "d" * 64, "candidates": list(CANDIDATES)},
        )


def test_reuse_wrapper_puts_new_diagnostics_outside_original_cache(tmp_path, monkeypatch):
    from carryrl.rewards import control_text_training as training

    encoder = TinyEncoder()
    rows = [labeled_row(split) for split in ("train", "dev", "calibration", "sealed_test")]
    cache = tmp_path / "original"
    prepare_disk_token_features(encoder, rows, cache, dataset_sha256="d" * 64)
    (cache / "selected_head.pt").write_bytes(b"original")
    old_files = {p.name: p.read_bytes() for p in cache.iterdir()}
    manifest = {"records_sha256": "d" * 64, "candidates": list(CANDIDATES)}
    monkeypatch.setattr(training, "read_control_dataset", lambda path: (manifest, rows))
    monkeypatch.setattr(
        training,
        "fit_text_control_head",
        lambda *a, **kw: (ResidualTextControlHead(), {"trained": True}),
    )
    calibration = {
        f"{kind}_{field}": value
        for kind in ("status", "mode", "pair")
        for field, value in (("temperature", 1.0), ("threshold", 0.99))
    }
    monkeypatch.setattr(training, "calibrate_text_control_head", lambda *a, **kw: (calibration, {}))
    from carryrl.rewards import control_text_runtime

    monkeypatch.setattr(
        control_text_runtime,
        "validate_runtime_text_controls",
        lambda *a, **kw: {"passed": False, "specification_hash": "c" * 64},
    )

    def save(output, **kwargs):
        output.mkdir()
        return {
            "gates": {"passed": True},
            "files": {"head.pt": "a" * 64, "calibration.json": "b" * 64},
        }

    monkeypatch.setattr(training, "save_text_control_artifact", save)
    workspace = tmp_path / "repair_fit"
    training.train_text_control_artifact(
        "unused",
        tmp_path / "repair_model",
        encoder=encoder,
        reuse_feature_cache=cache,
        fit_workspace=workspace,
        head_variant="residual_balanced_v2",
    )
    assert old_files == {p.name: p.read_bytes() for p in cache.iterdir()}
    assert (workspace / "selected_head.pt").is_file()
    assert (workspace / "selected_calibration.json").is_file()
    assert (workspace / "selected_validation.json").is_file()
