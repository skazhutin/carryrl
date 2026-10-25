"""Actual rollout accounting and narrow admission gates for new demonstrations."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_imitation_correction_replacement import bundle as replacement_bundle  # noqa: F401
from test_imitation_correction_replacement import load as load_bundle
from test_imitation_demonstration_reuse import corpus as saved_corpus  # noqa: F401
from test_imitation_demonstration_reuse import refresh_demonstration_hash

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv, _identity, atomic_level_banks
from carryrl.training.imitation import (
    MEMORY_COUNTERPART_SOURCE,
    NATIVE_SOURCE,
    QUALITY_SOURCE,
    SOURCE,
    DemonstrationDataset,
    DemonstrationEpisode,
    SavedDemonstrationReuse,
    _flipped_memory_level,
    _MemoryCueRuntimeEnv,
    augment_demonstrations,
    collect_scripted_corrections,
    collect_scripted_demonstrations,
    demonstration_episode_digest,
    scripted_teacher,
)


class IdleLearner:
    training = False

    def __init__(self, action=PhysicalAction()):
        self.action = action

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def reset_episode(self, **kwargs):
        self.history = []

    def act(self, frame, deterministic):
        return SimpleNamespace(action=self.action)

    def observe_action(self, action):
        self.history.append(action)


@pytest.mark.parametrize("extra", [PhysicalAction(), PhysicalAction(wheel=1)])
def test_successful_navigation_with_detour_or_irrelevant_input_is_rejected(extra):
    class ExtraFirstCommand:
        def __init__(self, task):
            self.delegate = scripted_teacher(task)
            self.first = True

        def act(self, frame):
            if self.first:
                self.first = False
                return extra
            return self.delegate.act(frame)

    with pytest.raises(RuntimeError, match="teacher quality gate failed"):
        collect_scripted_demonstrations(1, tasks=("navigation",),
                                       teacher_factory=ExtraFirstCommand)


def test_equivalent_shortest_routes_are_admitted():
    # Train navigation level 2 has two two-command routes around an empty corner.
    class Route:
        def __init__(self, keys):
            self.keys = iter(keys)

        def act(self, frame):
            return PhysicalAction(keys=(next(self.keys),))

    records = []
    for route in (("up", "right"), ("right", "up")):
        dataset = collect_scripted_demonstrations(1, tasks=("navigation",), seed=2,
                                                 teacher_factory=lambda _: Route(route))
        records.append(dataset.episodes[0])
    assert records[0].actions != records[1].actions
    for record in records:
        assert record.quality["excess_command_units"] == 0
        assert record.quality["completion_efficiency"] == 1


def test_non_navigation_success_with_audited_irrelevant_input_is_rejected():
    class UnnecessaryWheel:
        def __init__(self, task):
            self.delegate = scripted_teacher(task)

        def act(self, frame):
            return replace(self.delegate.act(frame), wheel=1)

    with pytest.raises(RuntimeError, match="teacher quality gate failed"):
        collect_scripted_demonstrations(1, tasks=("memory",), teacher_factory=UnnecessaryWheel)


def test_necessary_waits_and_safe_low_payoff_are_admitted():
    dataset = collect_scripted_demonstrations(1, tasks=("memory", "goal_choice"))
    memory, choice = dataset.episodes
    assert memory.quality["neutral_commands"] > 0
    assert memory.quality["unnecessary_input_units"] == 0
    assert choice.outcome["success"] and choice.outcome["score"] in (1, 2)
    assert choice.quality["terminal_payoff"] in (1, 2)
    assert choice.quality["optimal_command_units"] is None


@pytest.mark.parametrize("change", ["missing", "reset", "old_version"])
def test_new_collector_requires_final_current_quality(change):
    class InvalidQuality(AtomicEnv):
        def reset(self, **kwargs):
            frame, info = super().reset(**kwargs)
            self.initial_quality = deepcopy(info["quality"])
            return frame, info

        def step(self, action):
            frame, reward, terminated, truncated, info = super().step(action)
            if change == "missing":
                del info["quality"]
            elif change == "reset":
                info["quality"] = self.initial_quality
            else:
                info["quality"]["quality_version"] = "old"
            return frame, reward, terminated, truncated, info

    with pytest.raises(RuntimeError, match="quality gate failed"):
        collect_scripted_demonstrations(1, tasks=("navigation",), env_factory=InvalidQuality)


def test_truncated_correction_keeps_actual_quality_and_executed_history():
    learner = IdleLearner()
    dataset = collect_scripted_corrections(learner, 1, tasks=("navigation",), max_steps=2,
                                         expert_probability=0)
    episode = dataset.episodes[0]
    assert not episode.outcome["success"] and episode.truncated
    assert episode.executed_actions == tuple(learner.history) == (PhysicalAction(),) * 2
    assert episode.actions != episode.executed_actions
    assert episode.quality_source == QUALITY_SOURCE
    assert episode.quality["commands"] == episode.quality["neutral_commands"] == 2
    assert episode.quality["input_units"] == 0
    assert episode.quality["excess_command_units"] is None
    # Replacing target labels does not alter accounting of the executed mixture.
    relabeled = replace(episode, actions=(PhysicalAction(wheel=1),) * 2)
    assert relabeled.quality == episode.quality


def test_failed_correction_is_retained():
    learner = IdleLearner(PhysicalAction(keys=("left",)))
    dataset = collect_scripted_corrections(learner, 1, tasks=("memory",), expert_probability=0)
    episode = dataset.episodes[0]
    assert episode.outcome["failure"] and episode.terminated and not episode.truncated
    assert episode.executed_actions == tuple(learner.history)
    assert episode.quality["input_units"] == len(episode)
    assert episode.actions != episode.executed_actions


def test_successful_correction_detour_is_retained():
    dataset = collect_scripted_corrections(IdleLearner(), 1, tasks=("navigation",),
                                         expert_probability=1, learner_prefix_steps=1)
    episode = dataset.episodes[0]
    assert episode.outcome["success"] and episode.quality["excess_command_units"] == 1
    assert episode.executed_actions[0] == PhysicalAction()


def test_correction_retains_audited_unnecessary_inputs():
    learner = IdleLearner(PhysicalAction(wheel=1))
    dataset = collect_scripted_corrections(learner, 1, tasks=("navigation",), max_steps=2,
                                         expert_probability=0)
    episode = dataset.episodes[0]
    assert episode.quality["unnecessary_input_units"] == 2
    assert episode.executed_actions == (PhysicalAction(wheel=1),) * 2


@pytest.mark.parametrize("source", [SOURCE, NATIVE_SOURCE, MEMORY_COUNTERPART_SOURCE])
def test_successful_real_extra_input_cannot_be_reclassified_as_pure_teacher(source):
    task = "memory" if source == MEMORY_COUNTERPART_SOURCE else "navigation"
    runtime = None

    def factory(task, **options):
        nonlocal runtime
        if source == MEMORY_COUNTERPART_SOURCE:
            runtime = _flipped_memory_level(atomic_level_banks(task)["train"][0])
            return _MemoryCueRuntimeEnv(runtime, level_bank_seed=0, max_steps=options["max_steps"])
        if source == NATIVE_SOURCE:
            options["image_size"] = 64
        return AtomicEnv(task, **options)

    dataset = collect_scripted_corrections(IdleLearner(PhysicalAction(wheel=1)), 1,
        tasks=(task,), expert_probability=1, learner_prefix_steps=1, env_factory=factory)
    episode = dataset.episodes[0]
    assert episode.outcome["success"] and episode.terminated and not episode.truncated
    assert episode.quality["unnecessary_input_units"] == 1
    options = dict(source=source, actions=episode.executed_actions, executed_actions=None,
                   learner_prefix_steps=0, expert_probability=None)
    if source == NATIVE_SOURCE:
        assert all(frame.shape == (64, 64, 3) for frame in dataset.frames)
        options.update(native_size=64, origin_level_id=episode.level_id,
                       runtime_level_id=episode.level_id)
    elif source == MEMORY_COUNTERPART_SOURCE:
        options.update(origin_level_id=episode.level_id, runtime_level_id=_identity(runtime),
                       runtime_level_override=runtime)
    with pytest.raises(ValueError, match="pure teacher quality"):
        replace(episode, **options)
    # Exactly the same physical record remains reusable as an unmeasured legacy row.
    legacy = replace(episode, **options, quality=None, quality_source=None)
    assert "quality" not in legacy.to_dict()


def test_successful_real_detour_cannot_be_reclassified_as_pure_teacher():
    dataset = collect_scripted_corrections(IdleLearner(), 1, tasks=("navigation",),
                                         expert_probability=1, learner_prefix_steps=1)
    episode = dataset.episodes[0]
    assert episode.outcome["success"] and episode.quality["unnecessary_input_units"] == 0
    assert episode.quality["excess_command_units"] == 1
    with pytest.raises(ValueError, match="pure teacher quality"):
        replace(episode, source=SOURCE, actions=episode.executed_actions, executed_actions=None,
                learner_prefix_steps=0, expert_probability=None)


@pytest.fixture
def measured():
    return collect_scripted_demonstrations(1, tasks=("navigation",))


@pytest.mark.parametrize("changes", [
    {"commands": 0}, {"unnecessary_input_units": False}, {"game_seconds": float("nan")},
    {"input_units": -1}, {"game_seconds": 10**500},
    {"task": "memory"}, {"quality_version": "old"},
    {"optimal_policy_certified": True}, {"completion_efficiency": 2},
])
def test_supplied_quality_is_validated(measured, changes):
    episode = measured.episodes[0]
    with pytest.raises(ValueError, match="quality|navigation"):
        replace(episode, quality={**episode.quality, **changes})


def test_quality_requires_honest_source_and_explicit_compatible_scoring(measured):
    episode = measured.episodes[0]
    for changes in ({"quality_source": None}, {"quality_source": "expert_targets"},
                    {"scoring_version": None}, {"scoring_version": "legacy.v0"}):
        with pytest.raises(ValueError, match="quality"):
            replace(episode, **changes)


def _from_json(record):
    values = json.loads(json.dumps(record))
    values["frame_indices"] = tuple(values["frame_indices"])
    values["actions"] = tuple(PhysicalAction.from_dict(row) for row in values["actions"])
    return DemonstrationEpisode(**values)


def test_legacy_omission_json_roundtrip_and_quality_digest(measured):
    episode = measured.episodes[0]
    record = episode.to_dict()
    restored = _from_json(record)
    assert restored.to_dict() == record
    assert demonstration_episode_digest(measured, restored) == demonstration_episode_digest(
        measured, episode)
    legacy = replace(episode, quality=None, quality_source=None)
    old_record = legacy.to_dict()
    assert "quality" not in old_record and "quality_source" not in old_record
    assert _from_json(old_record).to_dict() == old_record
    assert demonstration_episode_digest(measured, legacy) != demonstration_episode_digest(
        measured, episode)
    record["quality"]["commands"] = 0
    assert episode.quality["commands"] == len(episode)


def test_appearance_copies_and_extend_preserve_physical_rollout_accounting(measured):
    copies = augment_demonstrations(measured, copies_per_episode=1)
    combined = DemonstrationDataset()
    combined.extend(copies)
    copied = combined.episodes[0]
    original = measured.episodes[0]
    assert copied.presentation is not None
    assert copied.actions == original.actions
    assert copied.quality == original.quality and copied.quality_source == QUALITY_SOURCE
    assert copied.quality is not original.quality


def test_saved_reuse_quality_and_legacy_omission(saved_corpus):  # noqa: F811
    run, dataset, config, _ = saved_corpus
    reused = SavedDemonstrationReuse.prepare(run, config=config, max_frame_bytes=256 * 1024**2)
    assert [row.to_dict() for row in reused.dataset.episodes] == [
        row.to_dict() for row in dataset.episodes]
    corpus = json.loads((run / "demonstrations.json").read_text())
    for episode in corpus["episodes"]:
        del episode["quality"], episode["quality_source"]
    (run / "demonstrations.json").write_text(json.dumps(corpus))
    refresh_demonstration_hash(run)
    legacy = SavedDemonstrationReuse.prepare(run, config=config, max_frame_bytes=256 * 1024**2)
    assert all(row.quality is None for row in legacy.dataset.episodes)
    assert legacy.dataset.provenance() == corpus


def test_saved_reuse_rejects_real_waste_reclassified_as_pure(saved_corpus):  # noqa: F811
    run, dataset, config, _ = saved_corpus
    corrections = collect_scripted_corrections(IdleLearner(PhysicalAction(wheel=1)), 1,
        tasks=("memory",), seed=8288, level_bank_seed=config.level_bank_seed,
        episode_instructions=config.episode_instructions,
        expert_probability=1, learner_prefix_steps=1)
    assert corrections.episodes[0].outcome["success"]
    dataset.extend(corrections)
    corpus = dataset.provenance()
    row = corpus["episodes"][-1]
    row["source"] = SOURCE
    row["actions"] = row.pop("executed_actions")
    for field in ("terminated", "truncated", "expert_probability", "learner_selection",
                  "mixing_rng", "learner_prefix_steps", "learner_prefix_steps_executed",
                  "action_selection_schedule"):
        row.pop(field, None)
    np.savez_compressed(run / "demonstration_frames.npz", **{
        f"frame_{index:06}": frame for index, frame in enumerate(dataset.frames)})
    (run / "demonstrations.json").write_text(json.dumps(corpus))
    refresh_demonstration_hash(run)
    with pytest.raises(ValueError, match="pure teacher quality"):
        SavedDemonstrationReuse.prepare(run, config=config, max_frame_bytes=256 * 1024**2)


def test_bundle_quality_replay_tamper_detection(replacement_bundle):  # noqa: F811
    directory, original = replacement_bundle
    grouped, _ = load_bundle(directory)
    assert grouped["interaction"].episodes[0].quality == original.episodes[2].quality
    path = directory / "trajectories.json"
    corpus = json.loads(path.read_text())
    # This remains schema-consistent; replay must reject invented audit counters.
    corpus["episodes"][2]["quality"]["unnecessary_input_units"] = 1
    path.write_text(json.dumps(corpus))
    with pytest.raises(ValueError, match="quality does not match"):
        load_bundle(directory)
