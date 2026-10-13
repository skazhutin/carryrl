"""True native RGB demonstrations and size-aware posthoc appearance replay."""

from dataclasses import replace

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv, _identity, atomic_level_banks, verify_visible_outcome
from carryrl.envs.atomic_suite import TASKS
from carryrl.envs.presentation import PresentationProfile, present_frame
from carryrl.training.imitation import (
    NATIVE_SOURCE,
    DemonstrationDataset,
    augment_demonstrations,
    collect_memory_cue_counterparts,
    collect_native_scripted_demonstrations,
    collect_scripted_demonstrations,
)


@pytest.mark.parametrize("size", [64, 128])
def test_all16_actual_native_train_episodes_have_visible_success_and_causal_rgb(size):
    dataset = collect_native_scripted_demonstrations(
        native_train_sizes=(size,), native_episodes_per_task=4,
        seed=0, level_bank_seed=20261008, max_steps=80)
    assert len(dataset.episodes) == 4 * len(TASKS)
    assert {frame.shape for frame in dataset.frames} == {(size, size, 3)}
    for episode in dataset.episodes:
        assert episode.source == NATIVE_SOURCE and episode.native_size == size
        assert episode.split == "train" and episode.difficulty == 1.0
        assert episode.level_id == _identity(
            atomic_level_banks(episode.task, 20261008)["train"][episode.level_index])
        assert episode.origin_level_id == episode.runtime_level_id == episode.level_id
        assert episode.outcome["success"] is True
        # Recreate the actual environment at its recorded native size. Actions are
        # in image pixels, and every RGB row must be an environment observation.
        env = AtomicEnv(episode.task, split="train", image_size=size,
                        difficulty=1.0, max_steps=80, level_bank_seed=20261008)
        try:
            frame, _ = env.reset(seed=episode.seed, options={"level_index": episode.level_index})
            for index, action in zip(episode.frame_indices, episode.actions, strict=True):
                np.testing.assert_array_equal(dataset.frames[index], frame)
                frame, _, ended, truncated, _ = env.step(action)
            assert ended and not truncated
            np.testing.assert_array_equal(dataset.frames[episode.final_frame_index], frame)
        finally:
            env.close()
    assert dataset.provenance()["native_demonstrations"]["canonical_rollouts"] == 64


def test_native_disabled_preserves_canonical_serialized_provenance_and_pixels():
    disabled = collect_native_scripted_demonstrations()
    assert disabled.episodes == [] and disabled.frames == []
    baseline = collect_scripted_demonstrations(2, tasks=("memory", "mouse"), seed=0)
    explicit = collect_scripted_demonstrations(2, tasks=("memory", "mouse"), seed=0,
                                               image_size=96, native_size=None)
    explicit.extend(disabled)
    assert baseline.provenance() == explicit.provenance()
    for before, after in zip(baseline.frames, explicit.frames, strict=True):
        np.testing.assert_array_equal(before, after)
    assert "native_demonstrations" not in baseline.provenance()
    assert all("native_size" not in episode.to_dict() for episode in baseline.episodes)


def test_native_appearance_is_size_matched_posthoc_and_preserves_inputs_and_outcomes():
    canonical = collect_native_scripted_demonstrations(
        native_train_sizes=(64, 128), native_episodes_per_task=2,
        tasks=("mouse", "memory"), seed=0, level_bank_seed=20261008)
    copies = augment_demonstrations(canonical, seed=11, copies_per_episode=3)
    for ordinal, copy in enumerate(copies.episodes):
        origin = canonical.episodes[ordinal // 3]
        profile = PresentationProfile(**copy.presentation)
        assert profile.native_size == origin.native_size
        assert profile.theme in ("slate", "sand", "forest")
        assert copy.actions == origin.actions
        assert copy.executed_actions == origin.executed_actions
        assert copy.outcome == origin.outcome and copy.instruction == origin.instruction
        assert copy.to_dict()["appearance_application"] == "posthoc replay augmentation"
        for old, new in zip((*origin.frame_indices, origin.final_frame_index),
                            (*copy.frame_indices, copy.final_frame_index), strict=True):
            np.testing.assert_array_equal(copies.frames[new],
                                          present_frame(canonical.frames[old], profile))
            assert verify_visible_outcome(copies.frames[new]) == verify_visible_outcome(
                canonical.frames[old])
    record = copies.provenance()
    assert record["appearance_augmentation"]["native_sizes"] == [64, 128]
    assert record["native_demonstrations"]["styled_learner_rollouts"] is False
    assert record["native_demonstrations"]["canonical_rollouts"] == 0


@pytest.mark.parametrize("sizes,count", [((64,), 0), ((), 1), ((96,), 1),
                                        ((64, 64), 1), ((True,), 1), ((64,), 65)])
def test_invalid_native_settings_abort_before_environment_creation(sizes, count):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid settings must not open an environment")
    with pytest.raises(ValueError):
        collect_native_scripted_demonstrations(native_train_sizes=sizes,
            native_episodes_per_task=count, env_factory=forbidden)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_native_collection_never_opens_held_out_banks(split):
    def forbidden(*args, **kwargs):
        raise AssertionError("held-out environment must never be opened")
    with pytest.raises(ValueError, match="train-only"):
        collect_native_scripted_demonstrations(native_train_sizes=(64,),
            native_episodes_per_task=1, split=split, env_factory=forbidden)


def test_native_failure_aborts_without_filtering_and_budget_is_hard():
    class IdleTeacher:
        def act(self, frame):
            assert frame.shape == (64, 64, 3) and not frame.flags.writeable
            return PhysicalAction()
    with pytest.raises(RuntimeError, match="quality gate failed"):
        collect_native_scripted_demonstrations(native_train_sizes=(64,),
            native_episodes_per_task=1, tasks=("navigation",), max_steps=2,
            teacher_factory=lambda task: IdleTeacher())
    with pytest.raises(MemoryError, match="byte budget"):
        collect_native_scripted_demonstrations(native_train_sizes=(64,),
            native_episodes_per_task=1, tasks=("navigation",), max_frame_bytes=1)


def test_native_wrong_size_mixed_shapes_provenance_and_duplicate_origin_are_rejected():
    class WrongSizeEnv(AtomicEnv):
        def reset(self, **kwargs):
            frame, info = super().reset(**kwargs)
            return frame[:63], info
    with pytest.raises(ValueError, match="native_size"):
        collect_native_scripted_demonstrations(native_train_sizes=(64,),
            native_episodes_per_task=1, tasks=("navigation",), env_factory=WrongSizeEnv)
    dataset = collect_native_scripted_demonstrations(native_train_sizes=(64,),
        native_episodes_per_task=1, tasks=("navigation",))
    episode = dataset.episodes[0]
    with pytest.raises(ValueError, match="duplicate native"):
        dataset.add_episode(episode)
    with pytest.raises(ValueError, match="native_size"):
        dataset.add_episode(replace(episode, native_size=128))
    with pytest.raises(ValueError, match="training bank"):
        heldout_id = _identity(atomic_level_banks("navigation")["validation"][0])
        dataset.add_episode(replace(episode, level_id=heldout_id,
            origin_level_id=heldout_id, runtime_level_id=heldout_id))
    mixed = dataset.intern_frame(np.zeros((128, 128, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="native_size"):
        dataset.add_episode(replace(episode, frame_indices=(mixed, *episode.frame_indices[1:])))


def test_native_append_leaves_canonical_memory_counterparts_and_order_unchanged():
    canonical = collect_scripted_demonstrations(4, tasks=("memory",),
        seed=0, difficulty=1.0, level_bank_seed=20261008)
    before = collect_memory_cue_counterparts(canonical, max_pairs=4)
    native = collect_native_scripted_demonstrations(native_train_sizes=(64, 128),
        native_episodes_per_task=4, tasks=("memory",), seed=0, level_bank_seed=20261008)
    combined = DemonstrationDataset()
    combined.extend(canonical)
    combined.extend(before)
    combined.extend(native)
    assert combined.episodes[:4] == canonical.episodes
    after = collect_memory_cue_counterparts(combined, max_pairs=4)
    assert before.provenance() == after.provenance()
    assert {frame.shape for frame in after.frames} == {(96, 96, 3)}
    assert all(row.native_size is None for row in after.episodes)
