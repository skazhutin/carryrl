"""Completed rollout raw arrays do not survive into checkpoints or evaluation."""

import weakref

from test_sequence_training import TinyReward, config, factory

import carryrl.training.sequence as sequence


def test_episode_storage_is_released_before_final_save_and_evaluation(tmp_path, monkeypatch):
    references = []
    original_collect = sequence.collect_episode
    original_save = sequence.save_sequence_checkpoint
    original_evaluate = sequence.evaluate_sequence
    observed = []

    def collect(*args, **kwargs):
        episode = original_collect(*args, **kwargs)
        references.extend([weakref.ref(episode), weakref.ref(episode.patches),
                           weakref.ref(episode.frames)])
        return episode

    def save(*args, **kwargs):
        if kwargs["rollout"]:
            assert references and all(reference() is None for reference in references)
            observed.append("saved_without_corpus")
        return original_save(*args, **kwargs)

    def evaluate(*args, **kwargs):
        assert references and all(reference() is None for reference in references)
        observed.append("evaluated_without_corpus")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(sequence, "collect_episode", collect)
    monkeypatch.setattr(sequence, "save_sequence_checkpoint", save)
    monkeypatch.setattr(sequence, "evaluate_sequence", evaluate)
    result = sequence.train_sequence(config(), tmp_path / "release", agent_factory=factory,
                                     reward=TinyReward(), reward_factory=TinyReward)
    assert result["optimizer_steps_this_process"] == 1
    assert observed == ["saved_without_corpus", "evaluated_without_corpus"]
