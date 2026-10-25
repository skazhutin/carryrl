"""Teacher rescue authenticates actual prefixes and keeps the original episode clock."""

import hashlib
import json
import random
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv, atomic_level_banks, verify_visible_outcome
from carryrl.envs.presentation import PresentationConfig
from carryrl.evaluation import rescue
from carryrl.training.imitation import scripted_teacher
from carryrl.training.sequence_config import SequenceRunConfig


def config(task="navigation", **kwargs):
    return SequenceRunConfig(tasks=(task,), eval_all_levels=True, max_episode_steps=80, **kwargs)


def capture(cfg, *, teacher=False, prefix_noops=0):
    env = AtomicEnv(cfg.tasks[0], split="validation", max_steps=cfg.max_episode_steps,
                    level_bank_seed=cfg.level_bank_seed)
    try:
        rgb, initial = env.reset(seed=1_000_000, options={"level_index": 0})
        frames, actions = [rgb.copy()], []
        controller = scripted_teacher(cfg.tasks[0])
        for index in range(cfg.max_episode_steps):
            action = (controller.act(rgb) if teacher and index >= prefix_noops else
                      PhysicalAction())
            rgb, _, terminated, truncated, info = env.step(action)
            controller.observe_action(action)
            frames.append(rgb.copy())
            actions.append(action.to_dict())
            if terminated or truncated:
                break
        record = {
            "task": cfg.tasks[0], "seed": 1_000_000, "steps": len(actions),
            "instruction": None, "terminated": terminated, "truncated": truncated,
            "policy": "frozen_deterministic", "episode_index": 0, "action_seed": cfg.eval_seed,
            "difficulty": 1.0, "level_index": 0, "level_id": initial["level_id"],
            "semantic_level_id": initial["semantic_level_id"],
            "runtime_fingerprint": initial["runtime_fingerprint"],
            "level_bank_seed": cfg.level_bank_seed, "presentation": None,
            "initial_frame_sha256": hashlib.sha256(frames[0].tobytes()).hexdigest(),
            "quality": info["quality"], "outcome": verify_visible_outcome(rgb),
        }
        return record, frames, actions
    finally:
        env.close()


@pytest.fixture(scope="module")
def loop_source():
    cfg = config()
    return cfg, *capture(cfg)


def attempts(result):
    return {attempt["anchor_id"]: attempt for attempt in result.report["attempts"]}


def test_real_navigation_stall_rescue_is_an_authenticated_continuation(loop_source):
    cfg, record, frames, actions = loop_source
    source_before = [frame.tobytes() for frame in frames]
    result = rescue.diagnose_teacher_rescue(cfg, record, frames, actions)
    report = result.report
    assert report["version"] == rescue.TEACHER_RESCUE_VERSION
    assert len(report["source_trace_sha256"]) == 64
    assert report["source_outcome"] == record["outcome"]
    assert report["source_quality"] == record["quality"]
    assert report["original_deadline_steps"] == 80
    assert report["unchanged_rgb_anchor"] == 8
    assert report["banks_unchanged"] is True
    assert report["bank_sha256_before"] == report["bank_sha256_after"]
    assert report["training_performed"] is False and report["learned_models_loaded"] is False
    for anchor_id, attempt in attempts(result).items():
        assert attempt["status"] == "rescued_success"
        assert attempt["anchor_reached"] is True
        assert attempt["prefix_steps"] == attempt["requested_anchor_step"]
        assert attempt["total_steps"] == attempt["prefix_steps"] + attempt["suffix_steps"] <= 80
        assert attempt["remaining_budget_steps"] == 80 - attempt["prefix_steps"]
        assert attempt["prefix_actual_actions_sha256"] == rescue._json_hash(
            actions[:attempt["prefix_steps"]])
        suffix_frames = result.frames_by_anchor[anchor_id]
        assert len(suffix_frames) == len(attempt["suffix_actions"]) + 1
        assert attempt["frame_sha256"] == [hashlib.sha256(frame.tobytes()).hexdigest()
                                           for frame in suffix_frames]
        assert np.array_equal(suffix_frames[0], frames[attempt["prefix_steps"]])
        assert verify_visible_outcome(suffix_frames[-1])["success"] is True
        assert attempt["quality"]["commands"] == attempt["total_steps"]
        for frame in suffix_frames:
            assert not frame.flags.writeable
            with pytest.raises(ValueError):
                frame.setflags(write=True)
    assert source_before == [frame.tobytes() for frame in frames]
    assert "does not prove physical impossibility" in report["interpretation"]
    json.dumps(report, allow_nan=False)


def test_saved_prefix_not_teacher_proposals_advances_state_and_history(loop_source):
    cfg, record, frames, actions = loop_source
    teachers = []

    class ObservingTeacher:
        def __init__(self, task):
            self.teacher = scripted_teacher(task)
            self.seen_frames, self.observed, self.proposals = [], [], []
            teachers.append(self)

        def act(self, rgb):
            assert rgb.shape == (96, 96, 3) and not rgb.flags.writeable
            self.seen_frames.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            action = self.teacher.act(rgb)
            self.proposals.append(action.to_dict())
            return action

        def observe_action(self, action):
            self.observed.append(action.to_dict())
            self.teacher.observe_action(action)

    result = rescue.diagnose_teacher_rescue(cfg, record, frames, actions, anchors=(0, 5),
                                           teacher_factory=ObservingTeacher)
    teacher = teachers[1]  # Source authentication does not use the experimental controller.
    attempt = attempts(result)["step_5"]
    assert teacher.observed[:5] == actions[:5]
    assert teacher.proposals[:5] != actions[:5]
    assert attempt["prefix_teacher_proposals_sha256"] == rescue._json_hash(teacher.proposals[:5])
    assert attempt["prefix_actual_actions_sha256"] != attempt["prefix_teacher_proposals_sha256"]
    assert teacher.seen_frames[:6] == [hashlib.sha256(frame.tobytes()).hexdigest()
                                      for frame in frames[:6]]
    assert teacher.observed[5:] == attempt["suffix_actions"]
    assert attempt["status"] == "rescued_success"


def test_late_anchor_does_not_restart_original_clock(loop_source):
    cfg, record, frames, actions = loop_source
    result = rescue.diagnose_teacher_rescue(cfg, record, frames, actions, anchors=(0, 79))
    initial, late = attempts(result)["step_0"], attempts(result)["step_79"]
    assert initial["status"] == "rescued_success"
    assert late["status"] == "teacher_timeout"
    assert late["prefix_steps"] == 79 and late["remaining_budget_steps"] == 1
    assert late["suffix_steps"] == 1 and late["total_steps"] == 80
    assert late["truncated"] is True and late["terminated"] is False


def test_terminal_prefix_is_skipped_instead_of_failed_and_nonstall_anchor_is_explicit():
    cfg = config()
    record, frames, actions = capture(cfg, teacher=True)
    assert record["terminated"] is True and len(actions) < 20
    result = rescue.diagnose_teacher_rescue(cfg, record, frames, actions,
                                          anchors=(0, len(actions), 60))
    for anchor in (len(actions), 60):
        attempt = attempts(result)[f"step_{anchor}"]
        assert attempt["status"] == "prefix_already_terminal"
        assert attempt["prefix_steps"] == len(actions)
        assert attempt["suffix_actions"] == []
        assert f"step_{anchor}" not in result.frames_by_anchor
    assert attempts(result)["unchanged_rgb8"]["status"] == "anchor_not_triggered"
    assert "unchanged_rgb8" not in result.frames_by_anchor


def test_real_failed_reaction_reports_teacher_failure_without_impossibility_claim():
    cfg = config("reaction")
    record, frames, actions = capture(cfg)

    class NoopTeacher:
        def __init__(self, task):
            pass

        def act(self, rgb):
            return PhysicalAction()

        def observe_action(self, action):
            pass

    result = rescue.diagnose_teacher_rescue(cfg, record, frames, actions, anchors=(0,),
                                           teacher_factory=NoopTeacher)
    attempt = attempts(result)["step_0"]
    assert attempt["status"] == "teacher_failure"
    assert attempt["outcome"]["failure"] is True
    assert attempt["error"] is None


@pytest.mark.parametrize("stage", ["prefix_act", "prefix_observe_action", "suffix_act",
                                   "suffix_observe_action"])
def test_teacher_errors_are_explicit_stage_outcomes_not_failed_games(loop_source, stage):
    cfg, record, frames, actions = loop_source

    class BrokenTeacher:
        def __init__(self, task):
            self.teacher = scripted_teacher(task)
            self.count = 0

        def act(self, rgb):
            self.count += 1
            if stage.endswith("_act") and self.count == (2 if stage.startswith("prefix") else 1):
                raise RuntimeError("teacher act failed")
            return self.teacher.act(rgb)

        def observe_action(self, action):
            if stage.endswith("observe_action") and self.count == (
                    2 if stage.startswith("prefix") else 1):
                raise RuntimeError("teacher observe failed")
            self.teacher.observe_action(action)

    result = rescue.diagnose_teacher_rescue(cfg, record, frames, actions, anchors=(0, 5),
                                           teacher_factory=BrokenTeacher)
    selected = attempts(result)["step_5" if stage.startswith("prefix") else "step_0"]
    assert selected["status"] == "teacher_error"
    assert selected["error"]["stage"] == stage
    assert selected["error"]["type"] == "RuntimeError"
    assert selected["outcome"]["failure"] is False
    if stage.startswith("prefix"):
        assert selected["anchor_reached"] is False and "step_5" not in result.frames_by_anchor
    else:
        assert selected["anchor_reached"] is True and "step_0" in result.frames_by_anchor


@pytest.mark.parametrize("corrupt", ["rgb", "quality", "action"])
def test_altered_source_aborts_before_any_counterfactual(loop_source, corrupt):
    cfg, record, frames, actions = loop_source
    record, frames, actions = deepcopy(record), [frame.copy() for frame in frames], list(actions)
    if corrupt == "rgb":
        frames[3][50, 0] ^= 1
    elif corrupt == "quality":
        record["quality"]["hits"] = 1
    else:
        actions[0] = PhysicalAction(keys=("down",)).to_dict()
    created = []

    def factory(*args, **kwargs):
        created.append(True)
        return AtomicEnv(*args, **kwargs)

    with pytest.raises(ValueError):
        rescue.diagnose_teacher_rescue(cfg, record, frames, actions, env_factory=factory)
    assert len(created) == 1


def test_environment_failure_propagates_closes_every_env_and_restores_rng(loop_source):
    cfg, record, frames, actions = loop_source
    created, closed = [], []

    class BrokenEnv(AtomicEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.number = len(created)
            created.append(self)

        def step(self, action):
            if self.number == 1:
                raise ValueError("environment step failed")
            return super().step(action)

        def close(self):
            closed.append(self.number)
            super().close()

    torch_state = torch.get_rng_state()
    with pytest.raises(ValueError, match="environment step failed"):
        rescue.diagnose_teacher_rescue(cfg, record, frames, actions, env_factory=BrokenEnv)
    assert closed == [0, 1]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_all_rng_banks_unchanged_and_no_model_or_optimizer_invoked(loop_source, monkeypatch):
    cfg, record, frames, actions = loop_source
    before = deepcopy(atomic_level_banks("navigation"))
    torch_state, python_state, numpy_state = (torch.get_rng_state(), random.getstate(),
                                             np.random.get_state())

    def forbidden(*args, **kwargs):
        raise AssertionError("model/optimizer construction forbidden")

    monkeypatch.setattr(torch.nn.Linear, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden)

    class RandomObservingTeacher:
        def __init__(self, task):
            self.teacher = scripted_teacher(task)

        def act(self, rgb):
            torch.rand(1)
            np.random.random()
            random.random()
            return self.teacher.act(rgb)

        def observe_action(self, action):
            self.teacher.observe_action(action)

    rescue.diagnose_teacher_rescue(cfg, record, frames, actions, anchors=(0, 20),
                                  teacher_factory=RandomObservingTeacher)
    assert atomic_level_banks("navigation") == before
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]


@pytest.mark.parametrize("anchors", [(), (20,), (0, 0), (0, True), (0, -1), (0, 80), None])
def test_invalid_anchor_policy_rejected_before_replay(loop_source, anchors):
    cfg, record, frames, actions = loop_source
    with pytest.raises(ValueError, match="anchors"):
        rescue.diagnose_teacher_rescue(cfg, record, frames, actions, anchors=anchors)


@pytest.mark.parametrize("window", [0, 81, True, 1.5])
def test_invalid_stall_window_rejected(loop_source, window):
    cfg, record, frames, actions = loop_source
    with pytest.raises(ValueError, match="stall_window"):
        rescue.diagnose_teacher_rescue(cfg, record, frames, actions, stall_window=window)


def test_memory_budget_rejects_excessive_anchors_before_replay(loop_source):
    cfg, record, frames, actions = loop_source
    with pytest.raises(MemoryError, match="32 MiB"):
        rescue.diagnose_teacher_rescue(cfg, record, frames, actions, anchors=tuple(range(80)))


@pytest.mark.parametrize("change", ["train", "stochastic", "difficulty", "native", "varied"])
def test_nonordinary_or_nonvalidation_source_rejected(loop_source, change):
    cfg, record, frames, actions = loop_source
    record = dict(record)
    if change == "train":
        record["split"] = "train"
    elif change == "stochastic":
        record["policy"] = "frozen_stochastic"
    elif change == "difficulty":
        cfg = replace(cfg, evaluation_difficulty=0.5)
    elif change == "native":
        cfg = replace(cfg, env_image_size=64)
    else:
        cfg = replace(cfg, visual_config=PresentationConfig(mode="varied"))
    with pytest.raises(ValueError, match="ordinary canonical96"):
        rescue.diagnose_teacher_rescue(cfg, record, frames, actions)


def test_fixed_anchor_has_identical_rescue_when_authenticated_future_commands_differ(loop_source):
    cfg, stalled_record, stalled_frames, stalled_actions = loop_source
    winning_record, winning_frames, winning_actions = capture(cfg, teacher=True, prefix_noops=5)
    assert stalled_record["outcome"]["success"] is False
    assert winning_record["outcome"]["success"] is True
    assert stalled_actions[:5] == winning_actions[:5]
    assert all(np.array_equal(a, b) for a, b in zip(stalled_frames[:6], winning_frames[:6]))
    stalled = rescue.diagnose_teacher_rescue(cfg, stalled_record, stalled_frames, stalled_actions,
                                            anchors=(0, 5))
    winning = rescue.diagnose_teacher_rescue(cfg, winning_record, winning_frames, winning_actions,
                                            anchors=(0, 5))
    assert attempts(stalled)["step_5"] == attempts(winning)["step_5"]
    assert stalled.report["source_trace_sha256"] != winning.report["source_trace_sha256"]


def test_invalid_teacher_command_is_a_typed_teacher_error(loop_source):
    cfg, record, frames, actions = loop_source

    class InvalidTeacher:
        def __init__(self, task):
            pass

        def act(self, rgb):
            return {"keys": ["left"]}

    result = rescue.diagnose_teacher_rescue(cfg, record, frames, actions, anchors=(0,),
                                           teacher_factory=InvalidTeacher)
    attempt = attempts(result)["step_0"]
    assert attempt["status"] == "teacher_error"
    assert attempt["error"]["stage"] == "suffix_act"
    assert attempt["error"]["type"] == "TypeError"
    assert attempt["suffix_actions"] == []
    assert len(result.frames_by_anchor["step_0"]) == 1


def test_cached_bank_mutation_is_detected_even_after_teacher_exception(loop_source):
    cfg, record, frames, actions = loop_source
    level = atomic_level_banks("navigation")["validation"][0]
    original = deepcopy(level)

    class MutatingTeacher:
        def __init__(self, task):
            pass

        def act(self, rgb):
            level["start"] = [0, 0]
            level["rescue_test_mutation"] = True
            raise RuntimeError("teacher changed global data")

    try:
        with pytest.raises(RuntimeError, match="mutated cached level banks"):
            rescue.diagnose_teacher_rescue(cfg, record, frames, actions, anchors=(0,),
                                          teacher_factory=MutatingTeacher)
    finally:
        level.clear()
        level.update(original)


def test_frozen_api_still_rejects_train_and_preserves_validation_trace_schema(loop_source):
    cfg, record, frames, actions = loop_source
    train_record = {**record, "split": "train"}
    with pytest.raises(ValueError, match="frozen validation"):
        rescue.diagnose_teacher_rescue(cfg, train_record, frames, actions)
    original = rescue.diagnose_frozen_trajectory(cfg, record, frames, actions)
    result = rescue.diagnose_teacher_rescue(cfg, record, frames, actions, anchors=(0,))
    assert result.report["source_trace_sha256"] == original["trace_sha256"]
    assert result.report["source_actions_sha256"] == original["actions_sha256"]
    assert "source_kind" not in result.report
    assert "source_demonstration_sha256" not in result.report
    assert result.report["source_episode"]["policy"] == "frozen_deterministic"
