"""Explicit user instructions enable text input without silently changing old checkpoints."""

import json
from pathlib import Path

import pytest

from carryrl.cli import main
from carryrl.training.sequence_config import SequenceRunConfig


def test_cli_instruction_file_enables_multilingual_input(tmp_path, monkeypatch):
    import carryrl.training.sequence as sequence

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(SequenceRunConfig().to_dict()))
    instruction_path = Path(__file__).resolve().parents[1] / "configs/atomic_instructions.json"
    seen = {}

    def train(config, output, resume, **kwargs):
        seen.update(config=config, output=output, resume=resume)
        return {"status": "completed"}

    monkeypatch.setattr(sequence, "train_sequence", train)
    output = tmp_path / "output"
    assert main(["train-v1", "--config", str(config_path), "--instructions",
                 str(instruction_path), "--output", str(output)]) == 0
    config = seen["config"]
    assert config.agent.text_encoder == "multilingual_e5_small"
    assert config.agent.instruction_max_tokens == 512
    assert set(config.episode_instructions) == set(config.tasks)
    assert "Tab" in config.episode_instructions["ui"][0]
    assert "Press Tab" in config.episode_instructions["ui"][1]
    assert seen["output"] == output and seen["resume"] is None


def test_cli_does_not_add_text_input_when_not_requested(tmp_path, monkeypatch):
    import carryrl.training.sequence as sequence

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(SequenceRunConfig().to_dict()))
    seen = []
    monkeypatch.setattr(sequence, "train_sequence",
                        lambda config, *_: seen.append(config) or {"status": "completed"})
    assert main(["train-v1", "--config", str(config_path),
                 "--output", str(tmp_path / "output")]) == 0
    assert seen[0].agent.text_encoder == "none"
    assert seen[0].episode_instructions == {}


def test_cli_selects_mlx_training_without_changing_frozen_encoder_device(tmp_path, monkeypatch):
    import carryrl.training.sequence as sequence

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(SequenceRunConfig().to_dict()))
    seen = []
    monkeypatch.setattr(sequence, "train_sequence",
                        lambda config, *_: seen.append(config) or {"status": "completed"})
    assert main(["train-v1", "--config", str(config_path), "--training-backend", "mlx",
                 "--output", str(tmp_path / "output")]) == 0
    assert seen[0].training_backend == "mlx"
    assert seen[0].device == "cpu"


def test_evaluation_rejects_instructions_for_legacy_config_before_creating_output(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(SequenceRunConfig().to_dict()))
    instruction_path = tmp_path / "instructions.json"
    instruction_path.write_text('{"ui": ["Press Tab"]}')
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="checkpoint trained with a text input"):
        main(["evaluate-v1", "--config", str(config_path), "--checkpoint",
              str(tmp_path / "old-checkpoint"), "--instructions", str(instruction_path),
              "--output", str(output)])
    assert not output.exists()
