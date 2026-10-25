"""Authenticated real RGB -> causal scene targets -> unfitted head computations."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from test_imitation_correction_replacement import INSTRUCTIONS, bundle, bundle_sha
from torch import nn

from carryrl.agent.actions import PhysicalAction, encode_actions
from carryrl.agent.model import AgentConfig
from carryrl.rewards.history import SceneHistoryHead, history_reward_loss
from carryrl.rewards.history_data import (
    _data_identity,
    encode_history_episode,
    export_history_reward_corpus,
    load_history_reward_corpus,
)
from carryrl.training.imitation import CORRECTION_SOURCE
from carryrl.training.sequence_config import SequenceRunConfig


@pytest.fixture
def source_bundle(tmp_path):
    return bundle.__wrapped__(tmp_path)


def prepare(source_bundle, tmp_path):
    directory, original = source_bundle
    config = source_config()
    output = tmp_path / "history"
    report = export_history_reward_corpus(directory, expected_sha256=bundle_sha(directory),
                                          source_config=config, output=output)
    corpus = load_history_reward_corpus(output, expected_sha256=report["corpus_sha256"])
    return directory, original, config, output, report, corpus


def source_config():
    return SequenceRunConfig(level_bank_seed=20261008, tasks=tuple(INSTRUCTIONS),
        agent=AgentConfig(text_encoder="multilingual_e5_small"), episode_instructions=INSTRUCTIONS)


def test_real_export_retains_groups_and_only_actual_input_history(source_bundle, tmp_path):
    _, original, _, _, report, corpus = prepare(source_bundle, tmp_path)
    assert report["episodes"] == 4 and report["runtime_groups"] == 2
    assert report["source_replay_validated"] and not report["model_fit"]
    assert not report["ppo_eligible"] and not report["test_used"]
    assert report["prefix_label_conflicts"] == []
    assert report["coverage"]["interaction"]["terminal_success_valid_frames"] == 0
    assert report["coverage"]["camera"]["supported"] is False
    for index, episode in enumerate(corpus.episodes):
        original_episode = original.episodes[index]
        executed = original_episode.executed_actions or original_episode.actions
        for step in range(len(episode["frame_indices"])):
            inputs = corpus.prefix(index, step)
            assert set(inputs) == {"frames", "executed_previous_actions", "goal"}
            assert inputs["frames"].shape == (step + 1, 96, 96, 3)
            assert (inputs["frames"][:, :28] == 0).all()
            assert torch.equal(inputs["executed_previous_actions"],
                               encode_actions([PhysicalAction(), *executed[:step]]))
        if episode["source"] == CORRECTION_SOURCE:
            assert episode["evaluation_only"]["teacher_target_differences"] > 0
        labels = corpus.targets(index)
        assert not labels["terminal_success_valid"].any()
        assert np.isnan(labels["terminal_success"]).all()
        assert not labels["event_valid"][0].any()


def test_no_future_or_diagnostic_metadata_enters_prefix(source_bundle, tmp_path):
    _, _, _, _, _, corpus = prepare(source_bundle, tmp_path)
    before = corpus.prefix(0, 1)
    episode = corpus.episodes[0]
    episode["frame_indices"][2:] = [episode["frame_indices"][-1]] * (
        len(episode["frame_indices"]) - 2)
    episode["executed_actions"][1:] = [PhysicalAction(keys=("space",)).to_dict()] * (
        len(episode["executed_actions"]) - 1)
    episode["evaluation_only"] = {"outcome": "poison", "teacher_labels": "poison"}
    episode["seed"], episode["task"], episode["annotations"] = -999, "poison", ["poison"]
    after = corpus.prefix(0, 1)
    np.testing.assert_array_equal(after["frames"], before["frames"])
    assert torch.equal(after["executed_previous_actions"], before["executed_previous_actions"])
    assert after["goal"] == before["goal"]
    for bad in (-1, True, 10000):
        with pytest.raises(ValueError, match="prefix index"):
            corpus.prefix(0, bad)


def test_source_pin_is_checked_before_public_replay(source_bundle, tmp_path, monkeypatch):
    import carryrl.rewards.history_data as data

    directory, _ = source_bundle
    monkeypatch.setattr(data, "load_correction_replacement_bundle", lambda *a, **kw: pytest.fail(
        "wrong SHA must fail before source replay"))
    with pytest.raises(ValueError, match="source bundle SHA256"):
        export_history_reward_corpus(directory, expected_sha256="0" * 64,
            source_config=SequenceRunConfig(), output=tmp_path / "history")
    assert not (tmp_path / "history").exists()


@pytest.mark.parametrize("filename", ("frames.npy", "episodes.json"))
def test_output_pin_detects_changed_rgb_or_targets(source_bundle, tmp_path, filename):
    _, _, _, output, report, _ = prepare(source_bundle, tmp_path)
    with (output / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="corpus SHA256"):
        load_history_reward_corpus(output, expected_sha256=report["corpus_sha256"])


@pytest.mark.parametrize("mutation", ("target", "group", "action", "split"))
def test_recomputed_pin_does_not_bypass_semantic_replay(source_bundle, tmp_path, mutation):
    _, _, _, output, _, _ = prepare(source_bundle, tmp_path)
    payload = json.loads((output / "episodes.json").read_text())
    row = payload["episodes"][0]
    if mutation == "target":
        row["annotations"][0]["potential"] = .123
    elif mutation == "group":
        row["runtime_group"] = "fake"
    elif mutation == "action":
        row["executed_actions"][0] = PhysicalAction().to_dict()
    else:
        row["split"] = "validation"
    (output / "episodes.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="annotations differ|runtime group|replay|episode"):
        load_history_reward_corpus(output, expected_sha256=_data_identity(output)[0])


def test_existing_and_source_outputs_are_protected(source_bundle, tmp_path):
    directory, _, config, output, report, _ = prepare(source_bundle, tmp_path)
    for destination in (directory, directory / "derived", directory.parent):
        with pytest.raises(ValueError, match="outside"):
            export_history_reward_corpus(directory, expected_sha256=bundle_sha(directory),
                source_config=config, output=destination)
    with pytest.raises(FileExistsError):
        export_history_reward_corpus(directory, expected_sha256=bundle_sha(directory),
            source_config=config, output=output)
    link = tmp_path / "symlink"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="outside"):
        export_history_reward_corpus(directory, expected_sha256=report["corpus_sha256"],
            source_config=config, output=link)


class FrozenVision(nn.Module):
    def forward(self, rgb):
        assert rgb.dtype == torch.uint8 and (rgb[:, :28] == 0).all()
        return rgb.float().mean(dim=(1, 2))[:, None].repeat(1, 2, 1) / 255


class FrozenGoal(nn.Module):
    def forward(self, text):
        assert isinstance(text, str) and "item" in text
        return torch.ones(4)


def test_masked_rgb_and_explicit_goal_reach_head_loss_without_metadata(source_bundle, tmp_path):
    _, _, _, _, _, corpus = prepare(source_bundle, tmp_path)
    vision, goal = FrozenVision().train(), FrozenGoal().train()
    index = next(i for i, row in enumerate(corpus.episodes) if row["source"] == CORRECTION_SOURCE)
    encoded = encode_history_episode(corpus, index, vision_encoder=vision, goal_encoder=goal)
    assert vision.training and goal.training
    assert set(encoded) == {"scene_features", "executed_previous_actions", "explicit_goal_features"}
    model = SceneHistoryHead(scene_feature_size=3, goal_feature_size=4, hidden_size=8)
    before = deepcopy(model.state_dict())
    outputs = model(**encoded)
    targets = {name: torch.from_numpy(value).unsqueeze(0)
               for name, value in corpus.targets(index).items()}
    loss = history_reward_loss(outputs, targets)
    assert torch.isfinite(loss)
    loss.backward()  # Computation/gradient check only: no optimizer or weight update.
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    with pytest.raises(ValueError, match="frozen"):
        encode_history_episode(corpus, index, vision_encoder=nn.Linear(3, 3), goal_encoder=goal)


def test_encoder_adapter_aligns_action_and_goal_dtype(source_bundle, tmp_path):
    _, _, _, _, _, corpus = prepare(source_bundle, tmp_path)

    class DoubleVision(FrozenVision):
        def forward(self, rgb):
            return super().forward(rgb).double()

    encoded = encode_history_episode(corpus, 0, vision_encoder=DoubleVision(),
                                      goal_encoder=FrozenGoal())
    assert all(value.dtype == torch.float64 for value in encoded.values())
    model = SceneHistoryHead(scene_feature_size=3, goal_feature_size=4, hidden_size=8).double()
    assert torch.isfinite(model(**encoded)["potential"]).all()


def test_encoder_adapter_bounds_frame_batches_without_changing_sequence(source_bundle, tmp_path):
    _, _, _, _, _, corpus = prepare(source_bundle, tmp_path)

    class BoundedVision(FrozenVision):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, rgb):
            self.batch_sizes.append(len(rgb))
            return super().forward(rgb)

    vision = BoundedVision()
    actual = encode_history_episode(corpus, 0, vision_encoder=vision,
                                    goal_encoder=FrozenGoal(), frame_batch_size=2)
    expected = encode_history_episode(corpus, 0, vision_encoder=FrozenVision(),
                                      goal_encoder=FrozenGoal(), frame_batch_size=32)
    assert max(vision.batch_sizes) <= 2
    assert sum(vision.batch_sizes) == len(corpus.episodes[0]["frame_indices"])
    assert all(torch.equal(value, expected[name]) for name, value in actual.items())
    with pytest.raises(ValueError, match="batch"):
        encode_history_episode(corpus, 0, vision_encoder=vision, goal_encoder=FrozenGoal(),
                               frame_batch_size=True)


def test_cli_exports_audits_and_refuses_overwrite(source_bundle, tmp_path, monkeypatch):
    module = importlib.util.spec_from_file_location("prepare_history_script",
        Path(__file__).parents[1] / "scripts/prepare_history_reward.py")
    script = importlib.util.module_from_spec(module)
    module.loader.exec_module(script)
    monkeypatch.setattr(script, "provenance", lambda: {})
    directory, _ = source_bundle
    config = source_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()))
    output = tmp_path / "prepared"
    args = ["--bundle", str(directory), "--bundle-sha256", bundle_sha(directory),
            "--source-config", str(config_path), "--output", str(output)]
    assert script.main(args) == 0
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "completed" and status["updates"] == 0
    assert not status["ppo_eligible"]
    assert status["causal_prefixes_checked"] > 4
    with pytest.raises(FileExistsError):
        script.main(args)
