"""Explicit TRAIN rescue authenticates correction RGB, labels and actual commands."""

import json
import random
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv, _identity, atomic_level_banks, verify_visible_outcome
from carryrl.envs.atomic_suite import SCORING_VERSION
from carryrl.evaluation import rescue
from carryrl.training.imitation import (
    CORRECTION_SOURCE,
    QUALITY_SOURCE,
    DemonstrationDataset,
    DemonstrationEpisode,
    demonstration_episode_digest,
    scripted_teacher,
)
from carryrl.training.sequence_config import SequenceRunConfig


def correction(task="navigation", bank=20261009):
    """A real TRAIN fixture with RGB teacher labels and separately executed noops."""
    dataset = DemonstrationDataset()
    env = AtomicEnv(task, split="train", image_size=96, difficulty=1.0, max_steps=80,
                    level_bank_seed=bank)
    teacher = scripted_teacher(task)
    try:
        rgb, initial = env.reset(seed=13288, options={"level_index": 0})
        indices, labels, executed = [], [], []
        for _ in range(80):
            index = dataset.intern_frame(rgb)
            indices.append(index)
            labels.append(teacher.act(dataset.frames[index]))
            actual = PhysicalAction()
            executed.append(actual)
            rgb, _, terminated, truncated, info = env.step(actual)
            teacher.observe_action(actual)
            if terminated or truncated:
                break
        episode = DemonstrationEpisode(
            frame_indices=tuple(indices), actions=tuple(labels),
            final_frame_index=dataset.intern_frame(rgb), instruction=None,
            task=task, seed=13288, level_id=initial["level_id"], level_index=0, difficulty=1.0,
            outcome=verify_visible_outcome(rgb), source=CORRECTION_SOURCE,
            executed_actions=tuple(executed), terminated=terminated, truncated=truncated,
            level_bank_seed=bank, expert_probability=0.0, quality=info["quality"],
            scoring_version=SCORING_VERSION, quality_source=QUALITY_SOURCE)
        dataset.add_episode(episode)
        return dataset, episode
    finally:
        env.close()


def config(task="navigation"):
    return SequenceRunConfig(tasks=(task,), level_bank_seed=20261008, max_episode_steps=80)


def by_anchor(result):
    return {row["anchor_id"]: row for row in result.report["attempts"]}


@pytest.mark.parametrize("task", ["navigation", "interaction"])
def test_true_train_correction_rescue_keeps_source_split_bank_and_actual_history(task):
    dataset, episode = correction(task)
    cfg = config(task)
    original_config = cfg.to_dict()
    original_digest = demonstration_episode_digest(dataset, episode)
    constructed = []

    def env_factory(*args, **kwargs):
        constructed.append(dict(kwargs))
        return AtomicEnv(*args, **kwargs)

    result = rescue.diagnose_correction_rescue(cfg, dataset, episode, anchors=(0, 5, 79),
                                              env_factory=env_factory)
    report = result.report
    assert report["source_kind"] == "authenticated_train_correction"
    assert report["source_episode"]["split"] == "train"
    assert report["source_episode"]["source"] == CORRECTION_SOURCE
    assert report["source_episode"]["level_bank_seed"] == 20261009
    assert report["source_episode"]["level_id"] == _identity(
        atomic_level_banks(task, 20261009)["train"][episode.level_index])
    assert "policy" not in report["source_episode"]
    assert "episode_index" not in report["source_episode"]
    assert report["source_demonstration_sha256"] == original_digest
    assert report["source_actions_sha256"] == rescue._json_hash(
        [action.to_dict() for action in episode.executed_actions])
    assert report["source_teacher_labels_sha256"] == rescue._json_hash(
        [action.to_dict() for action in episode.actions])
    assert report["source_actions_sha256"] != report["source_teacher_labels_sha256"]
    assert report["source_outcome"] == episode.outcome
    assert report["source_quality"] == episode.quality
    assert report["source_trace_schema"] == "carryrl.train_correction_trace.v1"
    assert constructed and all(row["split"] == "train" and row["level_bank_seed"] == 20261009
                               for row in constructed)
    assert by_anchor(result)["step_0"]["status"] == "rescued_success"
    selected = by_anchor(result)["step_5"]
    assert selected["status"] == "rescued_success"
    assert selected["prefix_actual_actions_sha256"] == rescue._json_hash(
        [action.to_dict() for action in episode.executed_actions[:5]])
    assert selected["prefix_teacher_proposals_sha256"] == rescue._json_hash(
        [action.to_dict() for action in episode.actions[:5]])
    late = by_anchor(result)["step_79"]
    assert late["status"] == "teacher_timeout"
    assert late["total_steps"] == 80 and late["suffix_steps"] == 1
    assert late["remaining_budget_steps"] == 1 and late["truncated"] is True
    for row in report["attempts"]:
        assert row["total_steps"] <= 80
    assert report["bank_sha256_before"] == report["bank_sha256_after"]
    assert f"{task}:20261009" in report["bank_sha256_before"]
    assert f"{task}:20261008" in report["bank_sha256_before"]
    assert f"{task}:default" in report["bank_sha256_before"]
    assert cfg.to_dict() == original_config
    assert demonstration_episode_digest(dataset, episode) == original_digest
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("corrupt", ["outcome", "rgb", "label", "origin"])
def test_public_train_entry_reauthenticates_source_not_only_importer(corrupt):
    dataset, episode = correction()
    if corrupt == "outcome":
        episode.outcome["score"] = 4.0  # Mutation after the dataset originally accepted the row.
    elif corrupt == "rgb":
        changed = dataset.frames[episode.frame_indices[3]].copy()
        changed[50, 0] ^= 1
        indices = list(episode.frame_indices)
        indices[3] = dataset.intern_frame(changed)
        episode = replace(episode, frame_indices=tuple(indices))
    elif corrupt == "label":
        labels = list(episode.actions)
        labels[0] = PhysicalAction()
        assert labels[0] != episode.actions[0]
        episode = replace(episode, actions=tuple(labels))
    else:
        episode = replace(episode, level_id="f" * 64)
    invoked = []

    def forbidden_teacher(task):
        invoked.append(task)
        raise AssertionError("experimental teacher reached unauthenticated source")

    with pytest.raises(ValueError):
        rescue.diagnose_correction_rescue(config(), dataset, episode,
                                          teacher_factory=forbidden_teacher)
    assert not invoked


def test_mutated_actual_commands_cannot_be_replaced_by_teacher_labels():
    dataset, episode = correction()
    actual = list(episode.executed_actions)
    actual[0] = episode.actions[0]
    quality = {**episode.quality, "active_commands": 1, "neutral_commands": 79, "input_units": 1}
    forged = replace(episode, executed_actions=tuple(actual), quality=quality)
    with pytest.raises(ValueError, match="executed history"):
        rescue.diagnose_correction_rescue(config(), dataset, forged)


def test_experimental_teacher_error_is_separate_from_standard_source_authentication():
    dataset, episode = correction()

    class BrokenTeacher:
        def __init__(self, task):
            pass

        def act(self, rgb):
            raise RuntimeError("experimental controller failed")

    result = rescue.diagnose_correction_rescue(config(), dataset, episode, anchors=(0,),
                                              teacher_factory=BrokenTeacher)
    selected = by_anchor(result)["step_0"]
    assert selected["status"] == "teacher_error"
    assert selected["error"]["stage"] == "suffix_act"
    assert selected["error"]["type"] == "RuntimeError"
    assert result.report["source_kind"] == "authenticated_train_correction"


def test_train_auth_and_continuations_preserve_rng_banks_without_actor_or_optimizer(monkeypatch):
    dataset, episode = correction()
    banks_before = deepcopy(atomic_level_banks("navigation", episode.level_bank_seed))
    torch_state, python_state, numpy_state = (torch.get_rng_state(), random.getstate(),
                                             np.random.get_state())

    def forbidden(*args, **kwargs):
        raise AssertionError("actor or optimizer allocation")

    monkeypatch.setattr(torch.nn.Linear, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden)

    class RandomTeacher:
        def __init__(self, task):
            self.teacher = scripted_teacher(task)

        def act(self, rgb):
            assert not rgb.flags.writeable and rgb.shape == (96, 96, 3)
            torch.rand(1)
            random.random()
            np.random.random()
            return self.teacher.act(rgb)

        def observe_action(self, action):
            self.teacher.observe_action(action)

    rescue.diagnose_correction_rescue(config(), dataset, episode, anchors=(0, 5),
                                     teacher_factory=RandomTeacher)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]
    assert atomic_level_banks("navigation", episode.level_bank_seed) == banks_before


@pytest.mark.parametrize("invalid", ["source", "executed", "quality", "split", "difficulty",
                                     "native", "varied"])
def test_train_rescue_rejects_unsupported_or_incomplete_sources(invalid):
    dataset, episode = correction()
    cfg = config()
    if invalid == "source":
        object.__setattr__(episode, "source", "scripted_teacher")
    elif invalid == "executed":
        object.__setattr__(episode, "executed_actions", None)
    elif invalid == "quality":
        object.__setattr__(episode, "quality", None)
    elif invalid == "split":
        object.__setattr__(episode, "split", "validation")
    elif invalid == "difficulty":
        object.__setattr__(episode, "difficulty", 0.5)
    elif invalid == "native":
        object.__setattr__(episode, "native_size", 64)
    else:
        from carryrl.envs.presentation import PresentationConfig

        cfg = replace(cfg, visual_config=PresentationConfig(mode="varied"))
    with pytest.raises(ValueError, match="TRAIN corrections"):
        rescue.diagnose_correction_rescue(cfg, dataset, episode)


def test_train_trace_hash_binds_train_identity_labels_and_rgb():
    dataset, episode = correction()
    result = rescue.diagnose_correction_rescue(config(), dataset, episode, anchors=(0,))
    frames = [dataset.frames[index] for index in
              (*episode.frame_indices, episode.final_frame_index)]
    trace = {
        "identity": {"split": "train", "source": CORRECTION_SOURCE,
                     "demonstration_episode_sha256": demonstration_episode_digest(dataset, episode),
                     "level_bank_seed": episode.level_bank_seed, "level_id": episode.level_id},
        "frame_sha256": [rescue._frame_hash(rgb) for rgb in frames],
        "frame_shapes": [list(rgb.shape) for rgb in frames],
        "executed_actions": [action.to_dict() for action in episode.executed_actions],
        "teacher_labels": [action.to_dict() for action in episode.actions],
    }
    assert result.report["source_trace_sha256"] == rescue._json_hash(trace)
    assert len(result.report["source_trace_sha256"]) == 64


def test_train_rescue_obeys_shared_frame_preflight_budget_and_anchor_guards():
    dataset, episode = correction()
    with pytest.raises(MemoryError, match="32 MiB"):
        rescue.diagnose_correction_rescue(config(), dataset, episode, anchors=tuple(range(80)))
    with pytest.raises(ValueError, match="anchors"):
        rescue.diagnose_correction_rescue(config(), dataset, episode, anchors=(20,))
