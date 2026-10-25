"""Targeted paired frozen-policy trajectories and explicitly marked command interventions."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv, atomic_level_banks, verify_visible_outcome
from carryrl.envs.atomic_suite import SCORING_VERSION, SUITE_VERSION
from carryrl.evaluation.imitation import _immutable_rgb, _preserve_rng
from carryrl.evaluation.paired import episode_descriptor
from carryrl.evaluation.trajectory_diagnostics import _frame_hash, _json_hash
from carryrl.rewards.visual import HUD_HEIGHT
from carryrl.training.imitation import _validate_trajectory_quality
from carryrl.training.sequence import _episode_instruction, sequence_evaluation_protocol

PAIRED_TRAJECTORIES_VERSION = "carryrl.paired_trajectories.v1"
_ROLES = ("before", "candidate", "sham", "intervention")
_COMPONENTS = ("keys", "mouse_buttons", "dx", "dy", "wheel")
_MAX_FRAME_BYTES = 32 * 1024**2


@dataclass(frozen=True)
class FrozenPolicyCase:
    record: dict
    frames: tuple[np.ndarray, ...]
    proposed_actions: tuple[PhysicalAction, ...]
    executed_actions: tuple[PhysicalAction, ...]


def _banks(config):
    hashes = {f"{task}:{seed}": _json_hash(atomic_level_banks(task, seed))
              for task in config.tasks
              for seed in dict.fromkeys((0, 20261008, 20261015, config.level_bank_seed))}
    hashes.update({f"{task}:default": _json_hash(atomic_level_banks(task))
                   for task in config.tasks})
    return hashes


def _rgb(frame):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (96, 96, 3):
        raise ValueError("paired trajectories require canonical uint8 96 RGB")
    return _immutable_rgb(frame)


def _identity(record):
    # Interventions have a separate policy label but retain exactly the same episode cohort.
    policy = record.get("policy")
    if policy not in ("frozen_deterministic", "diagnostic_single_command_intervention"):
        raise ValueError("case does not describe an allowed frozen/intervention policy")
    return {**episode_descriptor({**record, "policy": "frozen_deterministic"}),
            "split": record.get("split"), "level_bank_seed": record.get("level_bank_seed"),
            "protocol_sha256": record.get("protocol_sha256")}


def _trace(frames, proposed, executed):
    return _json_hash({"frame_sha256": [_frame_hash(frame) for frame in frames],
        "frame_shapes": [list(frame.shape) for frame in frames],
        "proposed_actions": [action.to_dict() for action in proposed],
        "executed_actions": [action.to_dict() for action in executed]})


def _validate_case(case):
    if not isinstance(case, FrozenPolicyCase):
        raise TypeError("expected FrozenPolicyCase")
    row = case.record
    _identity(row)
    if (row.get("role") not in _ROLES or row.get("split") != "validation"
            or row.get("presentation") is not None or row.get("difficulty") != 1.0
            or not 1 <= len(case.executed_actions) <= 80
            or len(case.proposed_actions) != len(case.executed_actions)
            or len(case.frames) != len(case.executed_actions) + 1
            or row.get("steps") != len(case.executed_actions)):
        raise ValueError("case must contain a whole canonical validation trajectory")
    if any(not isinstance(action, PhysicalAction) for action in
           (*case.proposed_actions, *case.executed_actions)):
        raise TypeError("case commands require complete PhysicalAction values")
    for frame in case.frames:
        _rgb(frame)
    if (row.get("frame_sha256") != [_frame_hash(frame) for frame in case.frames]
            or row.get("initial_frame_sha256") != _frame_hash(case.frames[0])
            or row.get("proposed_actions_sha256") != _json_hash(
                [action.to_dict() for action in case.proposed_actions])
            or row.get("executed_actions_sha256") != _json_hash(
                [action.to_dict() for action in case.executed_actions])
            or row.get("trace_sha256") != _trace(case.frames, case.proposed_actions,
                                                 case.executed_actions)):
        raise ValueError("case RGB/command trace hashes differ")
    if (type(row.get("terminated")) is not bool or type(row.get("truncated")) is not bool
            or not (row["terminated"] or row["truncated"])
            or row["terminated"] and row["truncated"]
            or row.get("outcome") != verify_visible_outcome(case.frames[-1])):
        raise ValueError("case outcome or terminal boundary differs from its actual RGB")
    if any(visible.get("success") or visible.get("failure") for visible in
           (verify_visible_outcome(frame) for frame in case.frames[:-1])):
        raise ValueError("case contains observations after a visible terminal boundary")
    if row["truncated"] and row["steps"] != row.get("original_deadline_steps"):
        raise ValueError("case truncation differs from its original clock")
    if row["role"] == "intervention":
        override = row.get("override")
        if (row["policy"] != "diagnostic_single_command_intervention"
                or not isinstance(override, dict) or type(override.get("index")) is not int
                or not 0 <= override["index"] < len(case.executed_actions)):
            raise ValueError("intervention requires its explicit override provenance")
        if any(proposed != actual and index != override["index"]
               for index, (proposed, actual) in enumerate(zip(
                   case.proposed_actions, case.executed_actions, strict=True))):
            raise ValueError("intervention changes more than one actual command")
    elif row["policy"] != "frozen_deterministic" or case.proposed_actions != case.executed_actions:
        raise ValueError("ordinary frozen/sham cases must execute their own proposals")
    _validate_trajectory_quality(row.get("quality"), task=row["task"],
        actions=case.executed_actions, outcome=row["outcome"], scoring_version=SCORING_VERSION)


def _reference(config, record):
    protocol = sequence_evaluation_protocol(config)
    if (not config.eval_all_levels or config.eval_stochastic or protocol["difficulty"] != 1.0
            or type(config.max_episode_steps) is not int or not 1 <= config.max_episode_steps <= 80
            or config.env_image_size != 96
            or (config.visual_config is not None and config.visual_config.mode != "canonical")
            or record.get("split", "validation") != "validation"
            or record.get("policy") != "frozen_deterministic"
            or record.get("task") not in config.tasks or record.get("presentation") is not None
            or type(record.get("difficulty")) not in (int, float) or record["difficulty"] != 1.0
            or record.get("level_bank_seed") != config.level_bank_seed):
        raise ValueError("paired capture requires canonical96 full1 deterministic full validation")
    task, index = record["task"], record.get("episode_index")
    if (type(index) is not int or not 0 <= index < protocol["counts"][task]
            or type(record.get("seed")) is not int or record["seed"] != 1_000_000 + index
            or type(record.get("level_index")) is not int
            or record["level_index"] != index % len(atomic_level_banks(
                task, config.level_bank_seed)["validation"])
            or type(record.get("action_seed")) is not int
            or record["action_seed"] != (config.eval_seed + config.tasks.index(task) * 100_000
                                         + index) % 2**32
            or record.get("instruction") != _episode_instruction(config, task, record["seed"])):
        raise ValueError("reference seed/index/instruction/action_seed differs from the protocol")
    identity = episode_descriptor(record)
    for field in ("level_id", "semantic_level_id", "runtime_fingerprint", "presentation"):
        if field not in record:
            raise ValueError(f"reference lacks initial {field} provenance")
    identity.update(split="validation", level_bank_seed=config.level_bank_seed,
                    protocol_sha256=_json_hash(protocol))
    return identity


def capture_frozen_case(agent, config, reference_record, *, role="candidate", override_step=None,
                        override_action=None, expected_prefix=None, env_factory=AtomicEnv):
    """Capture one fresh closed-loop case; override only an explicitly declared raw command.

    The model always proposes first. Its memory then observes the command actually
    executed, including the override; later proposals use the actual evolving RGB.
    """
    if role not in _ROLES:
        raise ValueError("case role must be before, candidate, sham or intervention")
    identity = _reference(config, reference_record)
    if (config.max_episode_steps + 1) * 96 * 96 * 3 > _MAX_FRAME_BYTES:
        raise MemoryError("paired capture exceeds its 32 MiB RGB frame budget")
    overridden = role in ("sham", "intervention")
    if overridden:
        if (type(override_step) is not int or not 0 <= override_step < config.max_episode_steps
                or not isinstance(override_action, PhysicalAction)):
            raise ValueError("sham/intervention requires a bounded step and PhysicalAction")
        _validate_case(expected_prefix)
        if (expected_prefix.record["role"] != "candidate"
                or _identity(expected_prefix.record) != identity
                or override_step >= len(expected_prefix.executed_actions)):
            raise ValueError("override requires the identical candidate episode prefix")
        if role == "sham" and override_action != expected_prefix.executed_actions[override_step]:
            raise ValueError("sham must execute the original candidate command")
    elif override_step is not None or override_action is not None or expected_prefix is not None:
        raise ValueError("ordinary before/candidate capture cannot override commands")
    prior_mode = agent.training
    frames, proposed, executed = [], [], []
    applied_override = None
    with _preserve_rng():
        before_banks = _banks(config)
        try:
            agent.eval()
            env = env_factory(identity["task"], split="validation", image_size=96,
                difficulty=1.0, max_steps=config.max_episode_steps,
                level_bank_seed=config.level_bank_seed,
                **({"visual_config": config.visual_config}
                   if config.visual_config is not None else {}))
            try:
                agent.reset_episode(instruction=identity["instruction"])
                frame, initial = env.reset(seed=identity["seed"],
                    options={"level_index": identity["level_index"]})
                if _frame_hash(_rgb(frame)) != identity["initial_frame_sha256"]:
                    raise ValueError("initial RGB differs from the reference")
                for field in ("level_id", "semantic_level_id",
                              "runtime_fingerprint", "presentation"):
                    if initial.get(field) != identity[field]:
                        raise ValueError(f"initial {field} differs from the reference")
                if (initial.get("split") != "validation" or initial.get("difficulty") != 1.0
                        or initial.get("level_bank_seed") != config.level_bank_seed
                        or initial.get("level_index") != identity["level_index"]
                        or initial.get("suite_version") != SUITE_VERSION
                        or initial.get("scoring_version") != SCORING_VERSION):
                    raise ValueError("initial runtime provenance differs from the reference")
                frames.append(_rgb(frame))
                for index in range(config.max_episode_steps):
                    proposal = agent.act(frames[-1], deterministic=True).action
                    if not isinstance(proposal, PhysicalAction):
                        raise TypeError("frozen agent must propose a complete PhysicalAction")
                    if overridden and index <= override_step:
                        if (not np.array_equal(frames[-1], expected_prefix.frames[index])
                                or proposal != expected_prefix.proposed_actions[index]
                                or tuple(executed) != expected_prefix.executed_actions[:index]):
                            raise ValueError("candidate RGB/proposal/actual history prefix differs")
                    actual = override_action if overridden and index == override_step else proposal
                    if overridden and index == override_step:
                        applied_override = {"index": index, "step": index + 1,
                            "proposed_action": proposal.to_dict(),
                            "executed_action": actual.to_dict()}
                    frame, _, terminated, truncated, info = env.step(actual)
                    proposed.append(proposal)
                    executed.append(actual)
                    frames.append(_rgb(frame))
                    agent.observe_action(actual)
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError("paired capture exceeded its original episode deadline")
                if overridden and applied_override is None:
                    raise ValueError("candidate ended before its requested override")
                outcome = verify_visible_outcome(frames[-1])
                record = {**identity, "role": role, "steps": len(executed),
                    "policy": ("diagnostic_single_command_intervention" if role == "intervention"
                               else "frozen_deterministic"),
                    "source": ("model_assisted_single_command_intervention"
                               if role == "intervention"
                               else "actual_frozen_learned_policy_closed_loop"),
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "outcome": outcome, "quality": deepcopy(info.get("quality")),
                    "policy_weights_verification": "caller_phase_guard_required",
                    "original_deadline_steps": config.max_episode_steps,
                    "proposed_actions_sha256": _json_hash([a.to_dict() for a in proposed]),
                    "executed_actions_sha256": _json_hash([a.to_dict() for a in executed]),
                    "frame_sha256": [_frame_hash(rgb) for rgb in frames],
                    "trace_sha256": _trace(frames, proposed, executed),
                    "actual_command_history_observed": True, "training_performed": False}
                if overridden:
                    record["override"] = applied_override
                case = FrozenPolicyCase(record, tuple(frames), tuple(proposed), tuple(executed))
                _validate_case(case)
            finally:
                env.close()
        finally:
            try:
                agent.reset_episode()
            finally:
                agent.train(prior_mode)
                if _banks(config) != before_banks:
                    raise RuntimeError("paired capture mutated cached level banks")
    return case


def _prefix_sha(case, index):
    return _json_hash({"frame_sha256": [_frame_hash(frame) for frame in case.frames[:index + 1]],
                       "executed_actions": [a.to_dict() for a in case.executed_actions[:index]]})


def _first_rgb(before, after, *, viewport=False):
    for index, (left, right) in enumerate(zip(before.frames, after.frames)):
        left, right = (left[HUD_HEIGHT:], right[HUD_HEIGHT:]) if viewport else (left, right)
        if not np.array_equal(left, right):
            return {"frame_index": index, "after_step": index,
                    "before_sha256": _frame_hash(left), "after_sha256": _frame_hash(right)}
    return None


def compare_frozen_cases(before, after):
    """Describe the earliest observed differences; aliases/mismatches are not game errors."""
    _validate_case(before)
    _validate_case(after)
    if _identity(before.record) != _identity(after.record):
        raise ValueError("paired cases have different episode descriptors")
    first = None
    for index, (left, right) in enumerate(zip(before.executed_actions, after.executed_actions)):
        if left == right:
            continue
        prior_equal = (before.executed_actions[:index] == after.executed_actions[:index]
            and all(np.array_equal(a, b) for a, b in
                    zip(before.frames[:index], after.frames[:index])))
        current_equal = np.array_equal(before.frames[index], after.frames[index])
        if not prior_equal or not current_equal:
            raise ValueError("RGB history drifted before the first different command")
        shared = _prefix_sha(before, index)
        first = {"index": index, "step": index + 1,
            "before": left.to_dict(), "after": right.to_dict(),
            "changed_components": [name for name in _COMPONENTS
                                   if getattr(left, name) != getattr(right, name)],
            "shared_prefix_sha256": shared, "current_rgb_identical": bool(current_equal),
            "all_prior_rgb_and_actions_identical": prior_equal}
        break
    if first is None and (len(before.executed_actions) != len(after.executed_actions)
            or len(before.frames) != len(after.frames)
            or any(not np.array_equal(a, b) for a, b in zip(before.frames, after.frames))):
        raise ValueError("equal command prefixes have incompatible RGB trajectories or lengths")
    shared_index = 0
    common = min(len(before.executed_actions), len(after.executed_actions))
    while (shared_index < common
            and before.executed_actions[shared_index] == after.executed_actions[shared_index]
            and np.array_equal(before.frames[shared_index + 1], after.frames[shared_index + 1])):
        shared_index += 1
    outcomes = {name: {key: case.record[key] for key in
                      ("outcome", "quality", "steps", "terminated", "truncated", "trace_sha256")}
                for name, case in (("before", before), ("after", after))}
    return {"version": PAIRED_TRAJECTORIES_VERSION, "episode": _identity(before.record), **outcomes,
            "first_command_difference": first,
            "first_full_rgb_difference": _first_rgb(before, after),
            "first_viewport_difference": _first_rgb(before, after, viewport=True),
            "shared_prefix_sha256": _prefix_sha(before, shared_index),
            "shared_prefix_steps": shared_index,
            "command_count_difference": None
                if len(before.executed_actions) == len(after.executed_actions)
                else {"before": len(before.executed_actions), "after": len(after.executed_actions)},
            "interpretation": "Observed raw-command and RGB differences only. Aliases may have "
                "identical visible effects. Divergence or teacher disagreement is not a proven "
                "game error or a causal explanation for an outcome."}


def verify_sham(original, sham):
    """Require complete model proposals, actual commands, RGB and final diagnostics to match."""
    _validate_case(original)
    _validate_case(sham)
    if (original.record["role"] != "candidate" or sham.record["role"] != "sham"
            or _identity(original.record) != _identity(sham.record)
            or original.proposed_actions != sham.proposed_actions
            or original.executed_actions != sham.executed_actions
            or len(original.frames) != len(sham.frames)
            or any(not np.array_equal(a, b) for a, b in zip(original.frames, sham.frames))
            or any(original.record[field] != sham.record[field] for field in
                   ("outcome", "quality", "steps", "terminated", "truncated"))):
        raise ValueError("sham does not exactly reproduce the original candidate trajectory")
