"""Real AtomicEnv capture/replay checks without constructing a learned model."""

import hashlib
import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.model import AgentConfig
from carryrl.envs.atomic import AtomicEnv, verify_visible_outcome
from carryrl.envs.presentation import PresentationConfig
from carryrl.evaluation.trajectory_diagnostics import diagnose_frozen_trajectory
from carryrl.training.imitation import scripted_teacher
from carryrl.training.sequence_config import SequenceRunConfig


def config(task="navigation", **kwargs):
    return SequenceRunConfig(
        agent=AgentConfig(pretrained=False, image_size=28, width=8, heads=2, layers=1,
                          spatial_tokens=2, memory_tokens=1, segment_length=40),
        tasks=(task,), eval_all_levels=True, eval_episodes_per_task=1,
        max_episode_steps=80, **kwargs)


def capture(cfg, *, input_change=lambda step, action: action):
    """Produce the exact callback schema with a controlled policy fixture."""
    task, seed = cfg.tasks[0], 1_000_000
    options = {"max_steps": cfg.max_episode_steps, "image_size": cfg.env_image_size,
               "difficulty": cfg.evaluation_difficulty or 1.0,
               "level_bank_seed": cfg.level_bank_seed}
    if cfg.visual_config is not None:
        options["visual_config"] = cfg.visual_config
    env = AtomicEnv(task, split="validation", **options)
    try:
        frame, initial = env.reset(seed=seed, options={"level_index": 0})
        frames, actions = [frame.copy()], []
        controller = scripted_teacher(task)
        for step in range(cfg.max_episode_steps):
            action = input_change(step, controller.act(env.canonical_render()))
            frame, _, terminated, truncated, info = env.step(action)
            controller.observe_action(action)
            actions.append(action.to_dict())
            frames.append(frame.copy())
            if terminated or truncated:
                break
        return {
            "task": task, "seed": seed, "steps": len(actions), "instruction": None,
            "terminated": terminated, "truncated": truncated,
            "policy": "frozen_deterministic", "episode_index": 0,
            "difficulty": options["difficulty"], "level_index": 0,
            "level_id": initial["level_id"], "semantic_level_id": initial["semantic_level_id"],
            "runtime_fingerprint": initial["runtime_fingerprint"],
            "level_bank_seed": cfg.level_bank_seed, "presentation": initial["presentation"],
            "initial_frame_sha256": hashlib.sha256(frames[0].tobytes()).hexdigest(),
            "quality": info["quality"], "outcome": verify_visible_outcome(frame),
        }, frames, actions
    finally:
        env.close()


def alias_and_waste(step, action):
    if step == 0:
        aliases = {"up": "w", "left": "a", "down": "s", "right": "d"}
        return replace(action, keys=tuple(aliases[key] for key in action.keys))
    return replace(action, wheel=1) if step == 1 else action


class ObservingTeacher:
    def __init__(self, task):
        self.teacher = scripted_teacher(task)
        self.observed = []
        self.input_hashes = []

    def act(self, frame):
        assert not frame.flags.writeable
        self.input_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
        return self.teacher.act(frame)

    def observe_action(self, action):
        self.observed.append(action.to_dict())
        self.teacher.observe_action(action)


def test_alias_disagreement_is_distinct_from_first_confirmed_waste_and_success():
    cfg = config()
    record, frames, actions = capture(cfg, input_change=alias_and_waste)
    observers = []

    def factory(task):
        teacher = ObservingTeacher(task)
        observers.append(teacher)
        return teacher

    result = diagnose_frozen_trajectory(cfg, record, frames, actions, teacher_factory=factory)
    assert result["source"] == "actual_frozen_learned_policy_trajectory"
    assert result["replay_verified"] is True
    assert result["actual_actions"] == actions == observers[0].observed
    assert result["teacher_actions"][0] != actions[0]
    assert result["first_teacher_disagreement"]["step"] == 1
    assert result["first_confirmed_unnecessary_input"]["step"] == 2
    assert result["first_confirmed_unnecessary_input"]["delta_units"] == 1
    assert result["first_confirmed_unnecessary_input"]["total_before"] == 0
    assert result["outcome"]["success"] is True
    assert result["outcome"]["score"] == 4
    assert result["first_visible_failure"] is None
    assert result["first_terminal_event"]["step"] == len(actions)
    assert result["first_terminal_event"]["terminated"] is True
    assert result["initial_frame_sha256"] == record["initial_frame_sha256"]
    assert result["terminal_frame_sha256"] == hashlib.sha256(frames[-1].tobytes()).hexdigest()
    assert "not a proven gameplay error" in result["interpretation"]
    json.dumps(result, allow_nan=False)


def test_varied_presentation_uses_same_state_canonical_rgb_and_exact_replay():
    cfg = config(visual_config=PresentationConfig(
        mode="varied", native_sizes=(96,), themes=("violet",), hud_layouts=("left",)))
    record, frames, actions = capture(cfg)
    assert frames[0].shape != (96, 96, 3)
    teachers = []

    def factory(task):
        teacher = ObservingTeacher(task)
        teachers.append(teacher)
        return teacher

    result = diagnose_frozen_trajectory(cfg, record, frames, actions, teacher_factory=factory)
    assert result["first_teacher_disagreement"] is None
    assert result["first_confirmed_unnecessary_input"] is None
    assert result["presentation"] == record["presentation"]
    assert teachers[0].input_hashes[0] != record["initial_frame_sha256"]
    assert result["outcome"]["success"] is True


def test_first_visible_failure_and_time_limit_have_distinct_events():
    cfg = config("memory")

    def wrong_side(step, action):
        if not action.keys:
            return action
        return PhysicalAction(keys=("right" if "left" in action.keys else "left",))

    record, frames, actions = capture(cfg, input_change=wrong_side)
    result = diagnose_frozen_trajectory(cfg, record, frames, actions)
    assert result["outcome"]["failure"] is True
    assert result["first_visible_failure"]["step"] == len(actions)
    assert result["first_terminal_event"]["terminated"] is True
    short = replace(config(), max_episode_steps=1)
    record, frames, actions = capture(short)
    result = diagnose_frozen_trajectory(short, record, frames, actions)
    assert result["first_visible_failure"] is None
    assert result["first_terminal_event"]["truncated"] is True
    assert result["first_terminal_event"]["outcome"]["success"] is False


@pytest.mark.parametrize("changed", ["initial", "middle", "action", "outcome", "quality",
                                    "level", "presentation", "flags", "difficulty"])
def test_misaligned_capture_fails_closed_and_always_closes_replay(changed):
    cfg = config()
    record, frames, actions = capture(cfg)
    if changed == "initial":
        frames[0] = frames[0].copy()
        frames[0][-1, 0, 0] ^= 1
    elif changed == "middle":
        frames[1] = frames[1].copy()
        frames[1][-1, 0, 0] ^= 1
    elif changed == "action":
        actions[0] = PhysicalAction().to_dict()
    elif changed == "outcome":
        record["outcome"] = {**record["outcome"], "score": 0}
    elif changed == "quality":
        record["quality"] = {**record["quality"], "unnecessary_input_units": 99}
    elif changed == "level":
        record["level_id"] = "wrong-level"
    elif changed == "presentation":
        record["presentation"] = {"theme": "ice"}
    elif changed == "flags":
        record["terminated"], record["truncated"] = False, True
    elif changed == "difficulty":
        record["difficulty"] = .2
    environments = []

    def env_factory(task, **options):
        env = AtomicEnv(task, **options)
        environments.append(env)
        return env

    with pytest.raises(ValueError):
        diagnose_frozen_trajectory(cfg, record, frames, actions, env_factory=env_factory)
    assert all(env._done for env in environments)


def test_diagnostic_preserves_global_rng_and_does_not_mutate_inputs():
    cfg = config()
    record, frames, actions = capture(cfg)
    original_frames = [frame.copy() for frame in frames]
    original_record, original_actions = json.dumps(record), json.dumps(actions)
    python_rng, numpy_rng, torch_rng = (random.getstate(), np.random.get_state(),
                                        torch.random.get_rng_state())

    def factory(task):
        random.random()
        np.random.random()
        torch.rand(1)
        return scripted_teacher(task)

    diagnose_frozen_trajectory(cfg, record, frames, actions, teacher_factory=factory)
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    assert all(np.array_equal(before, after)
               for before, after in zip(original_frames, frames, strict=True))
    assert json.dumps(record) == original_record and json.dumps(actions) == original_actions


def test_rejects_random_policy_wrong_split_incomplete_and_unbounded_capture():
    cfg = config()
    record, frames, actions = capture(cfg)
    with pytest.raises(ValueError, match="learned-policy"):
        diagnose_frozen_trajectory(
            cfg, {**record, "policy": "generic_physical_random"}, frames, actions)
    with pytest.raises(ValueError, match="validation only"):
        diagnose_frozen_trajectory(cfg, record, frames, actions, split="test")
    with pytest.raises(ValueError, match="initial/post-action RGB"):
        diagnose_frozen_trajectory(cfg, record, frames[:-1], actions)
    with pytest.raises(ValueError, match="1..80 actions"):
        diagnose_frozen_trajectory(cfg, record, frames * 81, actions * 81)
