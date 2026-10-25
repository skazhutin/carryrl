import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from carryrl.rewards.control_text import ControlPrediction
from carryrl.rewards.control_text_runtime import (
    runtime_control_rows,
    validate_runtime_text_controls,
)
from carryrl.rewards.control_text_training import DiskTokenFeatures, prepare_disk_token_features

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = ROOT / "configs/atomic_instructions.json"
CONTROLS = ROOT / "configs/atomic_controls.json"


def oracle_predictor(rows):
    tables = {
        row["text"]: SimpleNamespace(
            predictions=tuple(
                ControlPrediction(
                    label["candidate"],
                    label["status"],
                    label["mode"]
                    if label["candidate"].startswith(("key:", "mouse:"))
                    else "unknown",
                    (
                        str(label["alias_group"])
                        if label["alias_group"]
                        else label["candidate"]
                        if label["status"] == "unsupported"
                        else None
                    ),
                    1.0,
                )
                for label in row["labels"]
            )
        )
        for row in rows
    }
    return SimpleNamespace(predict=lambda text: tables[text])


def test_runtime_labels_exact_all32_and_unknown_conditional_layout():
    rows, identity = runtime_control_rows(INSTRUCTIONS, CONTROLS)
    assert len(rows) == 32 and len(identity["specification_hash"]) == 64
    for row in rows:
        labels = {label["candidate"]: label for label in row["labels"]}
        if row["task"] == "ui":
            for key in ("a", "d", "w", "s", "left", "right", "up", "down"):
                assert labels[f"key:{key}"]["status"] == "unknown"
            assert labels["key:tab"]["status"] == "supported"
            assert labels["key:tab"]["mode"] == "press"
            assert labels["mouse:left"]["status"] == "unsupported"
        if row["task"] == "shooting":
            assert sum(label["status"] == "supported" for label in labels.values()) == 13
            assert labels["key:w"]["alias_group"] == labels["key:up"]["alias_group"]
            assert labels["key:w"]["alias_group"] != labels["key:d"]["alias_group"]
            assert labels["mouse:left"]["mode"] == "press"
            assert labels["key:w"]["mode"] == "held"
    report = validate_runtime_text_controls(
        oracle_predictor(rows), instructions=INSTRUCTIONS, controls=CONTROLS
    )
    assert report["passed"] and report["passed_records"] == 32


def test_public_text_states_held_mode_and_mentions_every_supported_physical_input():
    from transformers import AutoTokenizer

    from carryrl.agent.language import TEXT_ID, TEXT_REVISION
    from carryrl.rewards.control_text import CANDIDATES
    from carryrl.rewards.control_text_grounding import build_grounding

    tokenizer = AutoTokenizer.from_pretrained(TEXT_ID, revision=TEXT_REVISION,
                                              local_files_only=True)
    rows, _ = runtime_control_rows(INSTRUCTIONS, CONTROLS)
    for row in rows:
        mentioned = build_grounding(row["text"], tokenizer).any(-1)
        for label in row["labels"]:
            if label["status"] != "supported":
                continue
            assert mentioned[CANDIDATES.index(label["candidate"])], (
                row["task"], row["language"], label["candidate"]
            )
            if label["candidate"].startswith("key:") and label["mode"] == "held":
                assert ("удерж" if row["language"] == "ru" else "hold") in row["text"].lower()


def test_runtime_cannot_pass_by_abstaining_or_guessing_conditional_inputs():
    rows, _ = runtime_control_rows(INSTRUCTIONS, CONTROLS)
    abstain = SimpleNamespace(
        predict=lambda text: SimpleNamespace(
            predictions=tuple(
                ControlPrediction(label["candidate"], "unknown", "unknown", None, 0.0)
                for label in rows[0]["labels"]
            )
        )
    )
    assert not validate_runtime_text_controls(
        abstain, instructions=INSTRUCTIONS, controls=CONTROLS
    )["passed"]
    for row in rows:
        if row["task"] == "ui":
            row["labels"][0].update(status="supported", mode="held", alias_group=99)
    report = validate_runtime_text_controls(
        oracle_predictor(rows), instructions=INSTRUCTIONS, controls=CONTROLS
    )
    assert not report["passed"]
    assert any(item["errors"] for item in report["results"] if item["task"] == "ui")


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(384))
        self.texts = []

    def encode_instruction_tokens(self, text):
        self.texts.append(text)
        return self.weight.detach()[None].repeat(4, 1)


def test_disk_features_bounded_deduplicated_sealed_excluded_and_authenticated(tmp_path):
    language = TinyEncoder()
    rows = [
        {"text": text, "split": split}
        for text, split in (
            ("train", "train"),
            ("train", "train"),
            ("dev", "dev"),
            ("sealed", "sealed_test"),
        )
    ]
    features = prepare_disk_token_features(
        language, rows, tmp_path / "features", dataset_sha256="0" * 64
    )
    assert language.texts == ["train", "dev"]
    assert isinstance(features.data, __import__("numpy").memmap)
    assert features.metadata["actual_bytes"] == 2 * 4 * 384 * 2
    torch.testing.assert_close(features.encode_instruction_tokens("train"), torch.ones(4, 384))
    with (tmp_path / "features/features.f16").open("r+b") as stream:
        stream.write(b"xx")
    with pytest.raises(ValueError, match="hash mismatch"):
        DiskTokenFeatures(tmp_path / "features")


def test_feature_budget_failure_preserves_partial_artifacts(tmp_path):
    with pytest.raises(RuntimeError, match="disk byte budget"):
        prepare_disk_token_features(
            TinyEncoder(),
            [{"text": "train", "split": "train"}],
            tmp_path / "features",
            dataset_sha256="0" * 64,
            max_bytes=1,
        )
    assert json.loads((tmp_path / "features/manifest.json").read_text())["status"] == "failed"
    assert (tmp_path / "features/features.f16").exists()


@pytest.mark.parametrize("runtime_passed", (True, False))
def test_runtime_gate_finalization_is_pending_until_hashed_report(
    monkeypatch, tmp_path, runtime_passed
):
    from carryrl.rewards import control_text_training as training

    feature_path = tmp_path / "features"
    feature_path.mkdir()
    features = SimpleNamespace(metadata={"actual_bytes": 100, "throughput_probe": {"records": 8}})
    monkeypatch.setattr(
        training, "read_control_dataset", lambda path: ({"records_sha256": "d" * 64}, [])
    )
    monkeypatch.setattr(training, "prepare_disk_token_features", lambda *a, **kw: features)
    monkeypatch.setattr(
        training, "fit_text_control_head", lambda *a, **kw: (nn.Linear(384, 3), {"trained": True})
    )
    calibration = {
        f"{kind}_{field}": value
        for kind in ("status", "mode", "pair")
        for field, value in (("temperature", 1.0), ("threshold", 0.99))
    }
    monkeypatch.setattr(training, "calibrate_text_control_head", lambda *a, **kw: (calibration, {}))

    def save(output, **kwargs):
        assert kwargs["training"]["trained"] is False
        assert kwargs["training"]["runtime_gate_pending"] is True
        output.mkdir()
        return {
            "ready": False,
            "gates": {"passed": True},
            "files": {"head.pt": "a" * 64, "calibration.json": "b" * 64},
        }

    monkeypatch.setattr(training, "save_text_control_artifact", save)
    from carryrl.rewards import control_text_runtime

    monkeypatch.setattr(
        control_text_runtime,
        "validate_runtime_text_controls",
        lambda *a, **kw: {"passed": runtime_passed, "specification_hash": "c" * 64},
    )
    result = training.train_text_control_artifact(
        "dataset", tmp_path / "model", encoder=TinyEncoder(), feature_cache=feature_path
    )
    assert result["ready"] is runtime_passed
    assert result["runtime_gate"]["passed"] is runtime_passed
    report = json.loads((tmp_path / "model/runtime_validation.json").read_text())
    assert report["head_sha256"] == "a" * 64
    assert report["dataset_sha256"] == "d" * 64
    assert (tmp_path / "model.fit/selected_head.pt").exists()


def test_batched_feature_extraction_masks_padding_and_rejects_overflow():
    from carryrl.rewards.control_text_training import _encode_feature_batch

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(384))

        def forward(self, input_ids, attention_mask, return_dict):
            return SimpleNamespace(last_hidden_state=input_ids[..., None] * self.weight)

    def tokenize(texts, **kwargs):
        assert kwargs["padding"] and not kwargs["truncation"]
        length = max(map(len, texts))
        ids = torch.zeros(len(texts), length)
        mask = torch.zeros(len(texts), length, dtype=torch.long)
        for index, text in enumerate(texts):
            ids[index, : len(text)] = torch.arange(len(text))
            mask[index, : len(text)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    language = SimpleNamespace(tokenizer=tokenize, backbone=Backbone(), max_tokens=512)
    features = _encode_feature_batch(language, ["short", "longer sentence"])
    assert [len(value) for value in features] == [12, 22]
    assert all(not value.requires_grad for value in features)
    with pytest.raises(ValueError, match="never truncated"):
        _encode_feature_batch(language, ["x" * 506])


@pytest.mark.parametrize("passed", (True, False))
def test_dataset_cli_writes_frozen_sealed_report_and_returns_gate_exit(
    tmp_path, monkeypatch, passed
):
    import importlib.util

    module_spec = importlib.util.spec_from_file_location(
        "validate_text_cli", ROOT / "scripts/validate_text_control_reward.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "load_text_control_reward", lambda path: SimpleNamespace(identity="frozen-model")
    )
    monkeypatch.setattr(
        module,
        "read_control_dataset",
        lambda path: ({"records_sha256": "d" * 64}, [{"split": "sealed_test"}, {"split": "dev"}]),
    )

    def validate(model, rows):
        assert rows == [{"split": "sealed_test"}]
        return {"gates": {"passed": passed}, "records": 1}

    monkeypatch.setattr(module, "validate_text_control_predictions", validate)
    output = tmp_path / "sealed.json"
    result = module.main(
        [
            "--model",
            "frozen",
            "--dataset",
            "unused",
            "--split",
            "sealed_test",
            "--allow-sealed-test-after-freeze",
            "--output",
            str(output),
        ]
    )
    assert result == (0 if passed else 1)
    report = json.loads(output.read_text())
    assert report["sealed_test_after_freeze_attested"] and report["frozen_model_loaded"]
    assert report["dataset_sha256"] == "d" * 64 and report["gates"]["passed"] == passed


def _training_cli():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_text_cli", ROOT / "scripts/train_text_control_reward.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_text_cli_budget_obeys_stage_global_and_cleanup_reserve():
    module = _training_cli()
    assert module.bounded_fit_seconds(10800, now=100, environ={}) == 10800
    assert (
        module.bounded_fit_seconds(10800, now=100, environ={"CARRYRL_STAGE_MAX_SECONDS": "8500"})
        == 8380
    )
    assert (
        module.bounded_fit_seconds(
            10800,
            now=100,
            environ={"CARRYRL_STAGE_MAX_SECONDS": "8500", "CARRYRL_CYCLE_DEADLINE_UNIX": "1000"},
        )
        == 780
    )
    with pytest.raises(TimeoutError, match="no positive"):
        module.bounded_fit_seconds(10800, now=100, environ={"CARRYRL_CYCLE_DEADLINE_UNIX": "220"})
    with pytest.raises(ValueError, match="finite"):
        module.bounded_fit_seconds(10800, now=100, environ={"CARRYRL_STAGE_MAX_SECONDS": "nan"})


def test_text_cli_sigterm_raises_for_wrapper_cleanup_and_restores_handler(monkeypatch):
    module = _training_cli()
    import signal

    previous = signal.getsignal(signal.SIGTERM)
    observed = {}
    monkeypatch.setenv("CARRYRL_STAGE_MAX_SECONDS", "8500")
    monkeypatch.delenv("CARRYRL_CYCLE_DEADLINE_UNIX", raising=False)
    monkeypatch.setattr(
        module,
        "MultilingualInstructionEncoder",
        lambda **kwargs: SimpleNamespace(to=lambda device: None),
    )

    def fit(*args, **kwargs):
        observed["max_seconds"] = kwargs["max_seconds"]
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    monkeypatch.setattr(module, "train_text_control_artifact", fit)
    with pytest.raises(TimeoutError, match="preserve partial artifacts"):
        module.main(["--dataset", "unused", "--output", "unused"])
    assert 8370 < observed["max_seconds"] <= 8380
    assert signal.getsignal(signal.SIGTERM) == previous
