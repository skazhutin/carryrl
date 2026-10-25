"""Held-out likelihood uses separate labels/history and never changes the learner."""

import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.model import AgentConfig, GeneralistAgent
from carryrl.envs.atomic import _identity, atomic_level_banks, atomic_semantic_level_id
from carryrl.evaluation.imitation import (
    ImitationValidationEpisode,
    collect_imitation_validation,
    evaluate_imitation_validation,
)
from carryrl.rewards.visual import draw_visible_hud
from carryrl.training.sequence import tensor_hash
from carryrl.training.sequence_config import SequenceRunConfig


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(3, 384)

    def forward(self, frames):
        return self.project(frames.float().mean((1, 2))[:, None] / 255)


class RecordingAgent(GeneralistAgent):
    def evaluate_episode(self, patches, actions, **kwargs):
        assert not self.training and not torch.is_grad_enabled()
        self.recorded = kwargs["executed_actions"]
        return super().evaluate_episode(patches, actions, **kwargs)


def config(**kwargs):
    return SequenceRunConfig(
        agent=AgentConfig(pretrained=False, image_size=28, width=8, layers=1, heads=2,
                          spatial_tokens=2, memory_tokens=1, segment_length=2),
        tasks=("navigation",), max_episode_steps=3, eval_episodes_per_task=1,
        torch_threads=1, **kwargs)


def agent_for(cfg):
    torch.set_num_threads(1)
    return RecordingAgent(cfg.agent, vision=TinyVision())


def trace_for(cfg, **kwargs):
    task = cfg.tasks[0]
    level = atomic_level_banks(task, cfg.level_bank_seed)["validation"][0]
    fields = dict(
        frames=(np.zeros((32, 32, 3), np.uint8),) * 3,
        target_actions=(PhysicalAction(), PhysicalAction(keys=("right",)),
                        PhysicalAction(keys=("left", "space"), mouse_buttons=("left",),
                                       dx=2, dy=-1, wheel=1)),
        executed_actions=(PhysicalAction(keys=("up",)), PhysicalAction(), PhysicalAction()),
        task=task, seed=1_000_000, episode_index=0, level_index=0,
        level_id=_identity(level), semantic_level_id=atomic_semantic_level_id(task, level),
        difficulty=cfg.evaluation_difficulty or cfg.curriculum_difficulty or 1.0,
        outcome={"known": True, "success": True, "failure": False, "score": 5},
        terminated=True, truncated=False, level_bank_seed=cfg.level_bank_seed)
    return ImitationValidationEpisode(**{**fields, **kwargs})


def test_nll_components_include_eos_history_and_exact_frozen_state_restoration():
    cfg = config()
    agent, reward = agent_for(cfg), nn.Linear(1, 1)
    trace = trace_for(cfg)
    agent.reset_episode()
    agent.observe_action(PhysicalAction(keys=("down",)))
    agent.decoder.eval()  # Preserve individual submodule modes as well as root mode.
    original_runtime = agent._episode_state
    original_modes = [module.training for module in agent.modules()]
    hashes = tensor_hash(agent), tensor_hash(reward)
    torch_state, python_state, numpy_state = (torch.random.get_rng_state(), random.getstate(),
                                             np.random.get_state())
    evaluated = evaluate_imitation_validation(
        agent, cfg, reward=reward, traces=[trace], checkpoint_weights_revision=7,
        encode_batch_size=1)
    summary = evaluated["summary"]["all"]
    assert summary["action_count"] == 3
    assert evaluated["summary"]["active"]["action_count"] == 2
    assert evaluated["summary"]["idle"]["action_count"] == 1
    assert summary["components"]["keys"]["token_count"] == 6  # Three keys + three EOS.
    assert summary["components"]["mouse_buttons"]["token_count"] == 3
    assert summary["joint_nll_sum"] == pytest.approx(
        sum(row["nll_sum"] for row in summary["components"].values()))
    assert summary["joint_nll"] == pytest.approx(
        sum(row["mean_nll_per_action"] for row in summary["components"].values()))
    assert agent.recorded == list(trace.executed_actions)
    assert agent.recorded != list(trace.target_actions)
    assert hashes == (tensor_hash(agent), tensor_hash(reward))
    assert evaluated["agent_sha256"] == hashes[0]
    assert evaluated["reward_sha256"] == hashes[1]
    assert evaluated["checkpoint_weights_revision"] == 7
    assert agent._episode_state is original_runtime
    assert agent._previous_action == PhysicalAction(keys=("down",))
    assert [module.training for module in agent.modules()] == original_modes
    assert all(parameter.grad is None for parameter in agent.parameters())
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    json.dumps(evaluated, allow_nan=False)
    wrong_history = replace(trace, executed_actions=trace.target_actions)
    other = evaluate_imitation_validation(agent, cfg, traces=[wrong_history])
    assert other["summary"]["all"]["joint_nll"] != summary["joint_nll"]
    assert other["records"][0]["trace_sha256"] != evaluated["records"][0]["trace_sha256"]


def test_validation_schema_rejects_training_test_and_train_bank_provenance():
    cfg = config()
    trace = trace_for(cfg)
    for split in ("train", "test"):
        with pytest.raises(ValueError, match="validation only"):
            replace(trace, split=split)
        with pytest.raises(ValueError, match="validation only"):
            evaluate_imitation_validation(agent_for(cfg), cfg, split=split)
        with pytest.raises(ValueError, match="validation only"):
            next(collect_imitation_validation(cfg, split=split))
    train_level = atomic_level_banks("navigation")["train"][0]
    invalid = replace(trace, level_id=_identity(train_level))
    with pytest.raises(ValueError, match="outside validation bank"):
        evaluate_imitation_validation(agent_for(cfg), cfg, traces=[invalid])
    with pytest.raises(ValueError, match="declared evaluation cohort"):
        evaluate_imitation_validation(agent_for(cfg), cfg, traces=[])
    with pytest.raises(ValueError, match="explicit episode cohort"):
        evaluate_imitation_validation(agent_for(cfg), cfg, traces=[trace, trace])
    with pytest.raises(ValueError):
        trace.frames[0].setflags(write=True)
    with pytest.raises(TypeError):
        trace.outcome["success"] = False


def test_failed_teacher_labels_are_excluded_without_fabricating_zero_nll():
    cfg = config()
    trace = trace_for(cfg, outcome={"known": True, "success": False, "failure": True})
    agent = agent_for(cfg)
    result = evaluate_imitation_validation(agent, cfg, traces=[trace])
    assert result["teacher_quality_gate"] == {
        "episodes_attempted": 1, "episodes_accepted": 0, "episodes_rejected": 1}
    assert result["summary"]["all"]["joint_nll"] is None
    assert result["summary"]["all"]["action_count"] == 0
    assert "likelihood" not in result["records"][0]
    assert not hasattr(agent, "recorded")


class FixtureEnv:
    instances = []

    def __init__(self, task, *, split, **options):
        assert split == "validation"
        self.task, self.options, self.closed = task, options, False
        self.instances.append(self)

    def reset(self, *, seed, options):
        self.seed, self.level_index, self.steps = seed, options["level_index"], 0
        self.frame = np.zeros((96, 96, 3), np.uint8)
        self.frame[-1] = 42
        level = atomic_level_banks(self.task, self.options["level_bank_seed"])["validation"][
            self.level_index]
        return self.frame.copy(), {"level_id": _identity(level),
            "semantic_level_id": atomic_semantic_level_id(self.task, level),
            "runtime_fingerprint": "fixture-runtime", "presentation": None}

    def canonical_render(self):
        canonical = self.frame.copy()
        canonical[-1] = 17
        return canonical

    def step(self, action):
        assert action == PhysicalAction(keys=("right",))
        self.steps += 1
        final = self.frame.copy()
        draw_visible_hud(final, 5, "Success")
        return final, 9999, True, False, {}  # The diagnostic must ignore API reward.

    def close(self):
        self.closed = True


class FixtureTeacher:
    def act(self, frame):
        assert not frame.flags.writeable and np.all(frame[-1] == 17)
        return PhysicalAction(keys=("right",))


def test_collector_explicit_indices_difficulty_canonical_rgb_and_progress():
    cfg = replace(config(curriculum_difficulty=.2, evaluation_difficulty=.7),
                  eval_episodes_per_task=2)
    FixtureEnv.instances = []
    events = []
    traces = list(collect_imitation_validation(
        cfg, env_factory=FixtureEnv, teacher_factory=lambda task: FixtureTeacher(),
        on_progress=events.append))
    assert [trace.level_index for trace in traces] == [0, 1]
    assert all(trace.difficulty == .7 and trace.labels_accepted for trace in traces)
    assert all(np.all(trace.frames[0][-1] == 42) for trace in traces)
    assert all(env.options["difficulty"] == .7 and env.closed for env in FixtureEnv.instances)
    assert events[-1]["episodes_completed"] == 2
    assert events[-1]["labels_accepted"] is True
    result = evaluate_imitation_validation(agent_for(cfg), cfg, traces=traces)
    assert result["summary"]["all"]["action_count"] == 2
    assert result["protocol"]["difficulty"] == .7


def test_collector_byte_budget_and_evaluation_failure_restore_state_and_rng():
    cfg = config()
    FixtureEnv.instances = []
    agent = agent_for(cfg)
    initial_hash = tensor_hash(agent)
    agent.reset_episode()
    initial_runtime = agent._episode_state
    rng = torch.random.get_rng_state()
    with pytest.raises(MemoryError, match="byte budget"):
        evaluate_imitation_validation(
            agent, cfg, env_factory=FixtureEnv, teacher_factory=lambda task: FixtureTeacher(),
            max_episode_frame_bytes=1)
    assert FixtureEnv.instances[-1].closed
    assert agent._episode_state is initial_runtime
    assert agent.training and tensor_hash(agent) == initial_hash
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_real_navigation_validation_collection_uses_successful_heldout_teacher():
    cfg = config()
    cfg = replace(cfg, max_episode_steps=80, agent=replace(cfg.agent, segment_length=40))
    traces = list(collect_imitation_validation(cfg))
    assert len(traces) == 1 and traces[0].labels_accepted
    assert traces[0].target_actions == traces[0].executed_actions
    assert traces[0].outcome["score"] == 5
