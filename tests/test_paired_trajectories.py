"""Agent-like fixtures test paired capture; these fixtures are not learned policies."""

import hashlib
import random
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.model import AgentConfig
from carryrl.envs.atomic import AtomicEnv, atomic_level_banks
from carryrl.evaluation import paired_trajectories as paired
from carryrl.training.sequence import _episode_instruction
from carryrl.training.sequence_config import SequenceRunConfig


class HistorySensitiveAgent(nn.Module):
    """Fixture mimicking act's memory advance and actual-command replacement."""

    def __init__(self, *, constant=None, drift=False):
        super().__init__()
        self.constant, self.drift = constant, drift
        self.reset_episode()

    def reset_episode(self, *, instruction=None):
        self.previous, self.steps = PhysicalAction(), 0

    def act(self, frame, *, deterministic):
        assert deterministic is True and not frame.flags.writeable
        if self.constant is not None:
            action = self.constant
        else:
            action = (PhysicalAction(keys=("right",)) if "e" in self.previous.keys else
                      PhysicalAction())
        if self.drift and self.steps == 1:
            action = PhysicalAction(keys=("down",))
        self.steps += 1
        self.previous = action  # act first advances its ordinary model memory.
        return SimpleNamespace(action=action)

    def observe_action(self, action):
        self.previous = action

    def state_dict(self, *args, **kwargs):
        raise AssertionError("per-case model tensor hashing is forbidden; CLI guards phases")


def config():
    return SequenceRunConfig(
        agent=AgentConfig(pretrained=False, text_encoder="multilingual_e5_small"),
        tasks=("navigation",), eval_all_levels=True,
        max_episode_steps=6, episode_instructions={"navigation": ("Reach the exit.",)},
        vision_cache_frames=0)


def reference(cfg):
    env = AtomicEnv("navigation", split="validation", max_steps=cfg.max_episode_steps,
                    level_bank_seed=cfg.level_bank_seed)
    try:
        frame, info = env.reset(seed=1_000_000, options={"level_index": 0})
        return {"task": "navigation", "seed": 1_000_000, "episode_index": 0,
            "instruction": _episode_instruction(cfg, "navigation", 1_000_000),
            "policy": "frozen_deterministic", "action_seed": cfg.eval_seed, "difficulty": 1.0,
            "level_index": 0, "level_id": info["level_id"],
            "semantic_level_id": info["semantic_level_id"],
            "runtime_fingerprint": info["runtime_fingerprint"], "presentation": None,
            "level_bank_seed": cfg.level_bank_seed,
            "initial_frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "steps": 1, "quality": {"stale": True}, "outcome": {"stale": True},
            "trace_sha256": "old-dynamic-fields-must-not-be-copied"}
    finally:
        env.close()


@pytest.fixture
def captured():
    cfg = config()
    row = reference(cfg)
    agent = HistorySensitiveAgent()
    case = paired.capture_frozen_case(agent, cfg, row)
    return agent, cfg, row, case


def test_override_observes_actual_history_after_nontrivial_identical_prefix(captured):
    agent, cfg, row, candidate = captured
    case = paired.capture_frozen_case(agent, cfg, row, role="intervention", override_step=2,
        override_action=PhysicalAction(keys=("e",)), expected_prefix=candidate)
    assert case.proposed_actions[:3] == candidate.proposed_actions[:3]
    assert case.executed_actions[:2] == candidate.executed_actions[:2]
    assert case.proposed_actions[2] == PhysicalAction()
    assert case.executed_actions[2] == PhysicalAction(keys=("e",))
    assert case.proposed_actions[3] == PhysicalAction(keys=("right",))
    assert candidate.proposed_actions[3] == PhysicalAction()
    assert all(actual == proposed for index, (actual, proposed) in enumerate(zip(
        case.executed_actions, case.proposed_actions, strict=True)) if index != 2)
    assert case.record["policy"] == "diagnostic_single_command_intervention"
    assert case.record["source"] == "model_assisted_single_command_intervention"
    assert case.record["steps"] == cfg.max_episode_steps
    assert case.record["quality"]["commands"] == cfg.max_episode_steps
    assert case.record["truncated"] is True
    comparison = paired.compare_frozen_cases(candidate, case)
    assert comparison["shared_prefix_steps"] == 2
    difference = comparison["first_command_difference"]
    assert difference["index"] == 2 and difference["step"] == 3
    assert difference["changed_components"] == ["keys"]
    assert difference["all_prior_rgb_and_actions_identical"] is True
    assert difference["current_rgb_identical"] is True
    assert len(difference["shared_prefix_sha256"]) == 64


def test_sham_reproduces_all_proposals_actual_actions_and_rgb(captured):
    agent, cfg, row, candidate = captured
    sham = paired.capture_frozen_case(agent, cfg, row, role="sham", override_step=2,
        override_action=candidate.executed_actions[2], expected_prefix=candidate)
    paired.verify_sham(candidate, sham)
    report = paired.compare_frozen_cases(candidate, sham)
    assert report["first_command_difference"] is None
    assert report["first_full_rgb_difference"] is None
    assert report["first_viewport_difference"] is None
    assert report["shared_prefix_steps"] == len(candidate.executed_actions)
    assert candidate.record["trace_sha256"] == sham.record["trace_sha256"]


def test_alias_command_difference_can_have_identical_full_and_viewport_rgb():
    cfg = config()
    row = reference(cfg)
    left_fixture = HistorySensitiveAgent(constant=PhysicalAction(keys=("left",)))
    before = paired.capture_frozen_case(left_fixture, cfg, row, role="before")
    after = paired.capture_frozen_case(HistorySensitiveAgent(constant=PhysicalAction(keys=("a",))),
                                     cfg, row)
    report = paired.compare_frozen_cases(before, after)
    assert report["first_command_difference"]["index"] == 0
    assert report["first_command_difference"]["changed_components"] == ["keys"]
    assert report["first_full_rgb_difference"] is None
    assert report["first_viewport_difference"] is None
    assert "not a proven" in report["interpretation"]


def test_capture_uses_fresh_dynamic_diagnostics_and_immutable_frame_storage(captured):
    agent, cfg, row, case = captured
    assert case.record["steps"] == 6 and case.record["quality"] != row["quality"]
    assert case.record["outcome"] != row["outcome"]
    assert case.record["trace_sha256"] != row["trace_sha256"]
    assert case.record["policy_weights_verification"] == "caller_phase_guard_required"
    assert "agent_sha256" not in case.record
    assert agent.training is True
    assert len(case.frames) == len(case.executed_actions) + 1 == 7
    for frame in case.frames:
        with pytest.raises(ValueError):
            frame.setflags(write=True)


def test_prefix_proposal_drift_fails_and_closes_environment(captured):
    _, cfg, row, candidate = captured
    closed = []

    class TrackedEnv(AtomicEnv):
        def close(self):
            closed.append(True)
            super().close()

    with pytest.raises(ValueError, match="prefix differs"):
        paired.capture_frozen_case(HistorySensitiveAgent(drift=True), cfg, row,
            role="intervention", override_step=2, override_action=PhysicalAction(keys=("e",)),
            expected_prefix=candidate, env_factory=TrackedEnv)
    assert closed == [True]


@pytest.mark.parametrize("field", ["rgb", "proposal", "quality", "outcome", "flags"])
def test_sham_corruption_fails(captured, field):
    agent, cfg, row, candidate = captured
    sham = paired.capture_frozen_case(agent, cfg, row, role="sham", override_step=2,
        override_action=candidate.executed_actions[2], expected_prefix=candidate)
    record = deepcopy(sham.record)
    if field == "rgb":
        frames = list(sham.frames)
        image = frames[3].copy()
        image[50, 0] ^= 1
        frames[3] = paired._immutable_rgb(image)
        sham = replace(sham, record=record, frames=tuple(frames))
        record["frame_sha256"] = [paired._frame_hash(frame) for frame in sham.frames]
        record["trace_sha256"] = paired._trace(
            sham.frames, sham.proposed_actions, sham.executed_actions)
    elif field == "proposal":
        proposed = list(sham.proposed_actions)
        proposed[2] = PhysicalAction(keys=("e",))
        sham = replace(sham, record=record, proposed_actions=tuple(proposed))
    elif field == "quality":
        record["quality"]["commands"] = 2
        sham = replace(sham, record=record)
    elif field == "outcome":
        record["outcome"]["score"] = 3.0
        sham = replace(sham, record=record)
    else:
        record["terminated"] = True
        sham = replace(sham, record=record)
    with pytest.raises(ValueError):
        paired.verify_sham(candidate, sham)


@pytest.mark.parametrize("field,value", [("instruction", "different instruction"),
    ("action_seed", 1), ("seed", 1000001), ("level_index", 1), ("level_bank_seed", 42),
    ("policy", "generic_physical_random"), ("split", "train"), ("presentation", {}),
    ("initial_frame_sha256", "wrong"), ("semantic_level_id", "wrong")])
def test_reference_provenance_differences_are_rejected(field, value):
    cfg = config()
    row = {**reference(cfg), field: value}
    with pytest.raises(ValueError):
        paired.capture_frozen_case(HistorySensitiveAgent(), cfg, row)


@pytest.mark.parametrize("options", [{"role": "unknown"}, {"override_step": 0},
    {"role": "sham", "override_step": -1}, {"role": "intervention", "override_step": 6},
    {"role": "intervention", "override_step": True},
    {"role": "intervention", "override_step": 2, "override_action": "left"}])
def test_override_bounds_and_role_guards(captured, options):
    agent, cfg, row, candidate = captured
    if options.get("role") in ("sham", "intervention"):
        options = {"expected_prefix": candidate, "override_action": PhysicalAction(), **options}
    with pytest.raises(ValueError):
        paired.capture_frozen_case(agent, cfg, row, **options)


def test_case_capture_preserves_rng_banks_mode_and_requires_no_optimizer(captured, monkeypatch):
    _, cfg, row, _ = captured
    bank = deepcopy(atomic_level_banks("navigation"))
    torch_state, python_state, numpy_state = (torch.get_rng_state(), random.getstate(),
                                             np.random.get_state())

    class RandomReadingFixture(HistorySensitiveAgent):
        def act(self, frame, *, deterministic):
            torch.rand(1)
            random.random()
            np.random.random()
            return super().act(frame, deterministic=deterministic)

    monkeypatch.setattr(torch.optim.AdamW, "__init__", lambda *a, **kw: pytest.fail("optimizer"))
    agent = RandomReadingFixture().eval()
    paired.capture_frozen_case(agent, cfg, row)
    assert agent.training is False
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0] and np.array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert atomic_level_banks("navigation") == bank


def test_compare_requires_identical_full_descriptor(captured):
    _, _, _, case = captured
    forged = replace(case, record={**case.record, "level_bank_seed": 42})
    with pytest.raises(ValueError, match="descriptors"):
        paired.compare_frozen_cases(case, forged)


def test_equal_commands_cannot_authenticate_a_forged_shared_rgb_prefix(captured):
    _, _, _, candidate = captured
    frames = list(candidate.frames)
    image = frames[3].copy()
    image[50, 0] ^= 1
    frames[3] = paired._immutable_rgb(image)
    record = deepcopy(candidate.record)
    record["frame_sha256"] = [paired._frame_hash(rgb) for rgb in frames]
    record["trace_sha256"] = paired._trace(frames, candidate.proposed_actions,
                                           candidate.executed_actions)
    altered = replace(candidate, record=record, frames=tuple(frames))
    with pytest.raises(ValueError, match="incompatible RGB"):
        paired.compare_frozen_cases(candidate, altered)


@pytest.mark.parametrize("frame_index", [1, 2])
def test_first_command_difference_requires_identical_prior_and_current_rgb(captured, frame_index):
    agent, cfg, row, candidate = captured
    intervention = paired.capture_frozen_case(agent, cfg, row, role="intervention", override_step=2,
        override_action=PhysicalAction(keys=("e",)), expected_prefix=candidate)
    frames = list(intervention.frames)
    image = frames[frame_index].copy()
    image[50, 0] ^= 1
    frames[frame_index] = paired._immutable_rgb(image)
    record = deepcopy(intervention.record)
    record["frame_sha256"] = [paired._frame_hash(rgb) for rgb in frames]
    record["trace_sha256"] = paired._trace(frames, intervention.proposed_actions,
                                           intervention.executed_actions)
    altered = replace(intervention, record=record, frames=tuple(frames))
    with pytest.raises(ValueError, match="RGB history drifted"):
        paired.compare_frozen_cases(candidate, altered)
