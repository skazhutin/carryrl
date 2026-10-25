"""Frozen masked caches cannot alias policy RGB caches or leak diagnostic inputs."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from carryrl.agent.actions import PhysicalAction
from carryrl.rewards.history_data import HistoryRewardCorpus
from carryrl.rewards.history_features import (
    _pin,
    load_history_features,
    prepare_history_features,
)


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.ones(1))
        self.calls = []

    def forward(self, rgb):
        assert not self.training and (rgb[:, :28] == 0).all()
        self.calls.append(len(rgb))
        return rgb.float().mean((1, 2))[:, None, :].expand(-1, 2, -1) * self.scale


class Goal(nn.Module):
    def forward(self, text):
        assert text in {"reach exit", "collect item"}
        return torch.tensor([len(text), 1.0])


def corpus_fixture():
    frames = np.zeros((7, 96, 96, 3), np.uint8)
    frames[:, 30:] = np.arange(7)[:, None, None, None] * 5
    episodes = [{"goal": goal, "split": split, "frame_indices": indices,
                 "executed_actions": [PhysicalAction(keys=("w",)).to_dict()] * (len(indices) - 1)}
                for goal, split, indices in (("reach exit", "train", [0, 1, 3, 6]),
                    ("collect item", "validation", [0, 2, 4, 5]))]
    corpus = HistoryRewardCorpus(frames, episodes)
    corpus.corpus_sha256 = "c" * 64
    return corpus


def test_deduplicated_bounded_features_keep_only_actual_input_tensors(tmp_path):
    corpus, vision, goal = corpus_fixture(), Vision().train(), Goal().train()
    state = torch.get_rng_state().clone()
    report = prepare_history_features(corpus, tmp_path / "cache", vision_encoder=vision,
                                      goal_encoder=goal, frame_batch_size=2)
    assert vision.calls == [2, 2, 2, 1]
    assert vision.training and goal.training and torch.equal(state, torch.get_rng_state())
    assert report["updates"] == 0 and not report["model_fit"] and not report["test_used"]
    features = load_history_features(tmp_path / "cache", corpus,
                                     expected_sha256=report["feature_sha256"])
    original = features.frames.copy()
    inputs = features.model_inputs(corpus, 0)
    assert set(inputs) == {"scene_features", "executed_previous_actions", "explicit_goal_features"}
    assert inputs["scene_features"].shape == (1, 4, 3)
    assert inputs["executed_previous_actions"].shape == (1, 4, 33)
    assert inputs["executed_previous_actions"][0, 0].count_nonzero() == 0
    inputs["scene_features"].zero_()
    np.testing.assert_array_equal(features.frames, original)
    corpus.episodes[0]["annotations"] = "not an input"
    corpus.episodes[0]["evaluation_only"] = {"reward": 10000}
    assert torch.equal(features.model_inputs(corpus, 0)["scene_features"], torch.tensor(
        original[[0, 1, 3, 6]]).unsqueeze(0))


@pytest.mark.parametrize("bad", [0, True, 33])
def test_invalid_batch_refused_before_encoding(tmp_path, bad):
    vision = Vision()
    with pytest.raises(ValueError, match="batch"):
        prepare_history_features(corpus_fixture(), tmp_path / "cache", vision_encoder=vision,
                                 goal_encoder=Goal(), frame_batch_size=bad)
    assert not vision.calls


@pytest.mark.parametrize("mutation", ["hud", "test", "identity", "trainable"])
def test_untrusted_inputs_fail_before_output(tmp_path, mutation):
    corpus, vision = corpus_fixture(), Vision()
    if mutation == "hud":
        corpus.frames[0, 0] = 10
    elif mutation == "test":
        corpus.episodes[0]["split"] = "test"
    elif mutation == "identity":
        del corpus.corpus_sha256
    else:
        vision.scale = nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError):
        prepare_history_features(corpus, tmp_path / "cache", vision_encoder=vision,
                                 goal_encoder=Goal())
    assert not (tmp_path / "cache").exists()


def test_external_hash_and_corpus_binding_required(tmp_path):
    corpus = corpus_fixture()
    output = tmp_path / "cache"
    report = prepare_history_features(corpus, output, vision_encoder=Vision(), goal_encoder=Goal())
    with pytest.raises(ValueError, match="SHA256"):
        load_history_features(output, corpus, expected_sha256="0" * 64)
    changed = deepcopy(corpus)
    changed.corpus_sha256 = "a" * 64
    with pytest.raises(ValueError, match="corpus_sha256"):
        load_history_features(output, changed, expected_sha256=report["feature_sha256"])
    frame_array = np.load(output / "frames.npy", allow_pickle=False)
    frame_array[0, 0] += 1
    np.save(output / "frames.npy", frame_array, allow_pickle=False)
    with pytest.raises(ValueError, match="frames features"):
        load_history_features(output, corpus, expected_sha256=_pin(output)[0])


def test_failed_callback_restores_modes_and_rng(tmp_path):
    vision, goal = Vision().train(), Goal().eval()
    state = torch.get_rng_state().clone()

    def fail(event):
        torch.rand(1)
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        prepare_history_features(corpus_fixture(), tmp_path / "cache", vision_encoder=vision,
                                 goal_encoder=goal, frame_batch_size=2, on_progress=fail)
    assert vision.training and not goal.training and torch.equal(state, torch.get_rng_state())
