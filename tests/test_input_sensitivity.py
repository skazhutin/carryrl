import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic_suite import TASKS
from carryrl.evaluation.input_sensitivity import (
    SensitivityPair,
    evaluate_input_sensitivity,
    load_sensitivity_bundle,
    prepare_mouse_position_bundle,
)


class ReplayAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.language = SimpleNamespace(encode_instruction_tokens=self.validate_text)
        self.calls = []
        self.reset_count = 0

    def validate_text(self, text):
        if len(text) > 512:
            raise ValueError("strict tokens overflow")
        return torch.zeros(1, 384)

    def evaluate_episode(self, patches, actions, *, instruction, executed_actions):
        self.calls.append((list(actions), list(executed_actions), instruction))
        # Stateful toy computes prefix dependence from RGB and recorded inputs.
        image = patches.sum(dim=(1, 2)).cumsum(0)
        history = torch.tensor([0, *[len(a.keys) + a.dx for a in executed_actions[:-1]]]).cumsum(0)
        candidates = torch.tensor([1 + len(a.keys) + a.dx for a in actions])
        text = 0 if instruction is None else len(instruction)
        return {"log_prob": -(image + history + text) * candidates}

    def reset_episode(self):
        self.reset_count += 1


def pair(kind="rgb_position", *, task="navigation"):
    original = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    perturbed = original.copy()
    history = (PhysicalAction(keys=("a",)), PhysicalAction(), PhysicalAction())
    other_history = history
    original_text = perturbed_text = None
    if kind == "rgb_position":
        perturbed[2, 0, 0, 0] = 1
    elif kind == "memory_cue":
        perturbed[0, 0, 0, 0] = 1
    elif kind == "text_control":
        original_text, perturbed_text = "A moves left", "A moves right"
    elif kind == "executed_history":
        other_history = (PhysicalAction(keys=("w", "a")), PhysicalAction(), PhysicalAction())
    return SensitivityPair(
        "case",
        task,
        kind,
        original,
        perturbed,
        original_text,
        perturbed_text,
        history,
        other_history,
        (PhysicalAction(keys=("d",)), PhysicalAction(dx=1)),
        2,
        {
            "split": "validation",
            "original_executed_source": "recorded-A",
            "perturbed_executed_source": "recorded-B",
        },
    )


def encode(frame):
    return torch.tensor([[[float(frame.sum())]]]).reshape(1, 1)


@pytest.mark.parametrize("kind", ("rgb_position", "text_control", "memory_cue", "executed_history"))
def test_shared_candidate_likelihoods_preserve_real_executed_history(kind):
    agent, probe = ReplayAgent(), pair(kind)
    report = evaluate_input_sensitivity(agent, [probe], encode_frame=encode)
    row = report["cases"][0]
    assert row["candidate_results"][0]["action"] == PhysicalAction(keys=("d",)).to_dict()
    assert len(agent.calls) == 4
    for labels, history, instruction in agent.calls[:2]:
        assert history == list(probe.original_executed)
        assert labels[:-1] == list(probe.original_executed[:-1])
        assert labels[-1] in probe.candidates
        assert history[-1] == PhysicalAction()  # Never substituted target labels into history.
    assert all(result["log_prob_delta"] != 0 for result in row["candidate_results"])
    assert report["training_updates"] == 0 and agent.reset_count == 1
    assert not agent.weight.requires_grad and not agent.training
    assert "not" not in report["interpretation"] or "success" in report["interpretation"]


@pytest.mark.parametrize("task", TASKS)
def test_all_sixteen_task_identifiers_supported_without_task_lookup(task):
    assert pair(task=task).validate().task == task


def test_cue_probe_same_current_frame_has_previous_rgb_dependency():
    probe = pair("memory_cue")
    report = evaluate_input_sensitivity(ReplayAgent(), [probe], encode_frame=encode)
    result = report["cases"][0]
    assert result["original_current_frame_sha256"] == result["perturbed_current_frame_sha256"]
    assert result["original_frames_sha256"] != result["perturbed_frames_sha256"]
    assert result["candidate_results"][0]["nll_delta"] == 2


def test_matched_history_and_current_cue_constraints_reject_confounding():
    probe = pair("rgb_position")
    with pytest.raises(ValueError, match="matched text/history"):
        replace(probe, perturbed_executed=pair("executed_history").perturbed_executed).validate()
    cue = pair("memory_cue")
    bad = cue.perturbed_frames.copy()
    bad[-1] = 1
    with pytest.raises(ValueError, match="matched current RGB"):
        replace(cue, perturbed_frames=bad).validate()
    with pytest.raises(ValueError, match="precede"):
        replace(pair("executed_history"), probe_step=0).validate()


def test_validation_budget_and_strict_instruction_tokens_are_enforced():
    probe = pair()
    with pytest.raises(ValueError, match="validation-only"):
        replace(probe, provenance={**probe.provenance, "split": "test"}).validate()
    too_long = np.zeros((81, 4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="1..80"):
        replace(probe, original_frames=too_long, perturbed_frames=too_long).validate()
    text = pair("text_control")
    with pytest.raises(ValueError, match="strict tokens"):
        evaluate_input_sensitivity(
            ReplayAgent(), [replace(text, original_instruction="x" * 513)], encode_frame=encode
        )


def test_preparation_bundle_roundtrip_uses_validation_same_geometry_and_actual_noops(tmp_path):
    path = prepare_mouse_position_bundle(tmp_path / "pairs", image_size=64)
    pairs, identity = load_sensitivity_bundle(path)
    assert len(pairs) == 1 and pairs[0].task == "mouse"
    assert pairs[0].provenance["split"] == "validation"
    assert pairs[0].original_executed == pairs[0].perturbed_executed == (PhysicalAction(),)
    assert not np.array_equal(pairs[0].original_frames, pairs[0].perturbed_frames)
    assert len(identity["frames_sha256"]) == 64
    result = evaluate_input_sensitivity(ReplayAgent(), pairs, encode_frame=encode)
    assert result["cases"][0]["prefix_steps"] == 1


def test_bundle_rejects_tampered_frames_and_test_split(tmp_path):
    path = prepare_mouse_position_bundle(tmp_path / "pairs", image_size=64)
    data = json.loads(path.read_text())
    data["split"] = "test"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="validation sensitivity"):
        load_sensitivity_bundle(path)
    data["split"] = "validation"
    path.write_text(json.dumps(data))
    (path.parent / "frames.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        load_sensitivity_bundle(path)


def test_cli_help_and_validation_only_options(capsys):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "input_sensitivity_cli", root / "scripts/evaluate_input_sensitivity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as result:
        module.main(["--help"])
    assert result.value.code == 0
    assert "--pairs" in capsys.readouterr().out


def test_real_recurrent_policy_replays_complete_cue_prefix_across_memory_windows():
    from carryrl.agent.model import AgentConfig, GeneralistAgent

    class Vision(nn.Module):
        def forward(self, frames):
            return torch.zeros(len(frames), 4, 384)

    torch.manual_seed(17)
    agent = GeneralistAgent(
        AgentConfig(
            pretrained=False,
            image_size=28,
            width=32,
            heads=4,
            layers=2,
            spatial_tokens=4,
            memory_tokens=2,
            segment_length=4,
        ),
        vision=Vision(),
    )
    original = np.zeros((9, 4, 4, 3), dtype=np.uint8)
    perturbed = original.copy()
    perturbed[0, 0, 0, 0] = 1
    history = tuple(
        PhysicalAction(keys=("a",)) if index == 1 else PhysicalAction() for index in range(9)
    )
    probe = replace(
        pair("memory_cue"),
        original_frames=original,
        perturbed_frames=perturbed,
        original_executed=history,
        perturbed_executed=history,
        probe_step=8,
    )

    def features(frame):
        # Deterministic frozen RGB features; the real recurrent policy and
        # autoregressive decoder still process all nine observations/actions.
        return torch.full((4, 384), float(frame.sum()))

    report = evaluate_input_sensitivity(agent, [probe], encode_frame=features)
    result = report["cases"][0]
    assert result["prefix_steps"] == 9
    patches = torch.stack([features(frame) for frame in perturbed])
    for candidate in result["candidate_results"]:
        labels = [*history[:-1], PhysicalAction.from_dict(candidate["action"])]
        expected = agent.evaluate_episode(patches, labels, executed_actions=list(history))
        assert candidate["perturbed_log_prob"] == pytest.approx(float(expected["log_prob"][-1]))
