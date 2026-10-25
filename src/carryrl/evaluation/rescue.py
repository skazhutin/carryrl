"""RGB-teacher continuations from separately authenticated validation and TRAIN states.

This diagnostic measures continuations found by one controller before the original
clock expires. It never turns controller failures into impossibility claims.
"""

from dataclasses import dataclass, replace

import numpy as np

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv, atomic_level_banks, verify_visible_outcome
from carryrl.envs.atomic_suite import SCORING_VERSION
from carryrl.evaluation.imitation import _callable_identity, _immutable_rgb, _preserve_rng
from carryrl.evaluation.trajectory_diagnostics import (
    _align_frame,
    _audit_units,
    _frame_hash,
    _json_hash,
    diagnose_frozen_trajectory,
)
from carryrl.training.imitation import (
    CORRECTION_SOURCE,
    QUALITY_SOURCE,
    _validate_replacement_trace,
    demonstration_episode_digest,
    scripted_teacher,
)
from carryrl.training.sequence import sequence_evaluation_protocol

TEACHER_RESCUE_VERSION = "carryrl.teacher_rescue.v1"
_MAX_FRAME_BYTES = 32 * 1024**2


@dataclass(frozen=True)
class TeacherRescueResult:
    report: dict
    frames_by_anchor: dict[str, tuple[np.ndarray, ...]]


def _banks(config):
    hashes = {f"{task}:{seed}": _json_hash(atomic_level_banks(task, seed))
              for task in config.tasks
              for seed in dict.fromkeys((0, 20261008, 20261015, config.level_bank_seed))}
    # lru_cache keeps the legacy omitted-seed call separate from the explicit zero call.
    hashes.update({f"{task}:default": _json_hash(atomic_level_banks(task))
                   for task in config.tasks})
    return hashes


def _rgb(frame):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (96, 96, 3):
        raise ValueError("teacher rescue requires canonical uint8 96 RGB")
    return _immutable_rgb(frame)


def _stall_anchor(frames, window):
    unchanged = 0
    for index in range(1, len(frames)):
        unchanged = unchanged + 1 if np.array_equal(frames[index - 1], frames[index]) else 0
        if unchanged == window:
            return index
    return None


def _attempt(config, record, source_frames, source_actions, anchor_id, anchor,
             *, env_factory, teacher_factory, split="validation"):
    deadline = config.max_episode_steps
    terminal_prefix = min(anchor, len(source_actions)) if anchor is not None else 0
    attempt = {
        "anchor_id": anchor_id, "requested_anchor_step": anchor,
        "prefix_steps": terminal_prefix, "remaining_budget_steps": deadline - terminal_prefix,
        "anchor_reached": False, "status": "anchor_not_triggered",
        "prefix_actual_actions_sha256": _json_hash(source_actions[:terminal_prefix]),
        "prefix_teacher_proposals_sha256": None, "suffix_actions": [], "frame_sha256": [],
        "suffix_steps": 0, "total_steps": terminal_prefix, "terminated": None,
        "truncated": None, "outcome": None, "quality": None, "error": None,
    }
    if anchor is None:
        return attempt, ()
    if anchor >= len(source_actions):
        attempt.update(status="prefix_already_terminal", terminated=record["terminated"],
                       truncated=record["truncated"], outcome=record["outcome"],
                       quality=record["quality"])
        return attempt, ()
    env = env_factory(record["task"], split=split, image_size=96,
                      difficulty=1.0, max_steps=deadline, level_bank_seed=config.level_bank_seed,
                      **({"visual_config": config.visual_config}
                         if config.visual_config is not None else {}))
    proposals, actual_prefix, suffix, observed = [], [], [], []
    total, terminated, truncated = 0, False, False
    failure = None
    try:
        reset_options = {"options": {"level_index": record["level_index"]}} \
            if record["level_index"] is not None else {}
        frame, info = env.reset(seed=record["seed"], **reset_options)
        _align_frame(frame, source_frames[0], frame_index=0)
        for field in ("level_id", "semantic_level_id", "runtime_fingerprint", "presentation"):
            if info.get(field) != record[field]:
                raise ValueError(f"rescue reset {field} differs from the source provenance")
        _audit_units(info, record["task"])
        teacher = teacher_factory(record["task"])
        for total_before in range(deadline):
            prefix = total_before < anchor
            canonical = _rgb(env.canonical_render())
            _align_frame(canonical, frame, frame_index=total_before)
            if not prefix and not observed:
                observed.append(_rgb(frame))
                attempt["anchor_reached"] = True
            stage = "prefix_act" if prefix else "suffix_act"
            try:
                proposed = teacher.act(canonical)
                if not isinstance(proposed, PhysicalAction):
                    raise TypeError("RGB teacher must return PhysicalAction")
            except Exception as error:
                failure = {"stage": stage, "type": type(error).__name__,
                           "message": str(error), "step": total_before}
                break
            if prefix:
                proposals.append(proposed.to_dict())
                action = PhysicalAction.from_dict(source_actions[total_before])
            else:
                action = proposed
            frame, _, terminated, truncated, info = env.step(action)
            total = total_before + 1
            _audit_units(info, record["task"])
            if info["quality"].get("commands") != total:
                raise ValueError("rescue environment clock differs from executed command count")
            if prefix:
                actual_prefix.append(action.to_dict())
                _align_frame(frame, source_frames[total], frame_index=total)
                if terminated or truncated:
                    raise ValueError("rescue prefix ended before the declared nonterminal anchor")
            else:
                suffix.append(action.to_dict())
                observed.append(_rgb(frame))
            stage = "prefix_observe_action" if prefix else "suffix_observe_action"
            try:
                teacher.observe_action(action)
            except Exception as error:
                failure = {"stage": stage, "type": type(error).__name__,
                           "message": str(error), "step": total}
                break
            if terminated or truncated:
                break
        outcome = verify_visible_outcome(frame)
        if failure is not None:
            status = "teacher_error"
        elif terminated and outcome.get("known") and outcome.get("success") is True:
            status = "rescued_success"
        elif terminated and outcome.get("known") and outcome.get("failure") is True:
            status = "teacher_failure"
        elif truncated or total == deadline:
            status = "teacher_timeout"
        else:
            raise ValueError("rescue ended without a known visible terminal outcome")
        attempt.update(status=status, prefix_steps=len(actual_prefix),
            remaining_budget_steps=deadline - len(actual_prefix),
            prefix_actual_actions_sha256=_json_hash(actual_prefix),
            prefix_teacher_proposals_sha256=_json_hash(proposals),
            suffix_actions=suffix, frame_sha256=[_frame_hash(rgb) for rgb in observed],
            suffix_steps=len(suffix), total_steps=total, terminated=bool(terminated),
            truncated=bool(truncated), outcome=outcome, quality=info["quality"], error=failure)
        return attempt, tuple(observed)
    finally:
        env.close()


def _rescue_limits(config, frame_count, anchors, stall_window):
    deadline = config.max_episode_steps
    try:
        anchors = tuple(anchors)
    except TypeError as error:
        raise ValueError("rescue anchors must be unique integer prefix lengths including zero") \
            from error
    if (not anchors or any(type(anchor) is not int or not 0 <= anchor < deadline
                           for anchor in anchors) or len(set(anchors)) != len(anchors)
            or 0 not in anchors):
        raise ValueError("rescue anchors must be unique integers in [0,deadline), including zero")
    if type(stall_window) is not int or not 1 <= stall_window <= 80:
        raise ValueError("stall_window must be an integer in [1,80]")
    estimated_frames = sum(deadline - anchor + 1 for anchor in anchors) + deadline + 1
    if (estimated_frames + frame_count) * 96 * 96 * 3 > _MAX_FRAME_BYTES:
        raise MemoryError("teacher rescue exceeds the explicit 32 MiB RGB frame budget")
    return anchors


def diagnose_teacher_rescue(config, record, frames, actions, *, anchors=(0, 20, 40, 60),
                            stall_window=8, env_factory=AtomicEnv,
                            teacher_factory=scripted_teacher):
    """Authenticate a source trace, then test fresh teacher suffixes on its exact prefixes.

    The source audit uses the standard RGB teacher independently of the experimental
    ``teacher_factory``. Prefix proposals are unused: only saved inputs advance the
    environment and enter the teacher's observed history. No learned model is called.
    """
    deadline = config.max_episode_steps
    protocol = sequence_evaluation_protocol(config)
    if (type(deadline) is not int or not 1 <= deadline <= 80
            or config.env_image_size != 96 or protocol["difficulty"] != 1.0
            or record.get("split", "validation") != "validation"
            or record.get("policy") != "frozen_deterministic"
            or record.get("presentation") is not None
            or (config.visual_config is not None and config.visual_config.mode != "canonical")):
        raise ValueError("teacher rescue requires ordinary canonical96 full1 frozen validation")
    anchors = _rescue_limits(config, len(frames), anchors, stall_window)
    source_frames = tuple(_rgb(frame) for frame in frames)
    source_actions = [action.to_dict() if isinstance(action, PhysicalAction) else
                      PhysicalAction.from_dict(action).to_dict() for action in actions]
    with _preserve_rng():
        before = _banks(config)
        try:
            source = diagnose_frozen_trajectory(config, record, source_frames, source_actions,
                                                env_factory=env_factory)
            stall, attempts, frames_by_anchor = _collect_attempts(
                config, record, source_frames, source_actions, anchors, stall_window,
                env_factory=env_factory, teacher_factory=teacher_factory)
        finally:
            after = _banks(config)
            if before != after:
                raise RuntimeError("teacher rescue mutated cached level banks")
    return _result(config, record, source, anchors, stall_window, stall, attempts,
                   frames_by_anchor, before, after, teacher_factory)


def _collect_attempts(config, record, source_frames, source_actions, anchors, stall_window,
                      *, env_factory, teacher_factory, split="validation"):
    frames_by_anchor, attempts = {}, []
    stall = _stall_anchor(source_frames, stall_window)
    selections = [(f"step_{anchor}", anchor) for anchor in anchors]
    selections.append((f"unchanged_rgb{stall_window}", stall))
    for anchor_id, anchor in selections:
        attempt, suffix_frames = _attempt(config, record, source_frames, source_actions,
            anchor_id, anchor, env_factory=env_factory,
            teacher_factory=teacher_factory, split=split)
        attempts.append(attempt)
        if suffix_frames:
            frames_by_anchor[anchor_id] = suffix_frames
    return stall, attempts, frames_by_anchor


def _result(config, record, source, anchors, stall_window, stall, attempts,
            frames_by_anchor, before, after, teacher_factory):
    return TeacherRescueResult({
        "version": TEACHER_RESCUE_VERSION, "source_trace_sha256": source["trace_sha256"],
        "source_actions_sha256": source["actions_sha256"], "source_outcome": record["outcome"],
        "source_quality": record["quality"],
        "source_episode": {key: record.get(key) for key in (
            "task", "split", "policy", "seed", "episode_index", "level_index", "level_id",
            "semantic_level_id", "runtime_fingerprint", "level_bank_seed", "instruction",
            "initial_frame_sha256", "difficulty")},
        "source_authentication_controller": _callable_identity(scripted_teacher),
        "attempt_controller": _callable_identity(teacher_factory),
        "original_deadline_steps": config.max_episode_steps, "anchors": list(anchors),
        "stall_window": stall_window, "unchanged_rgb_anchor": stall, "attempts": attempts,
        "bank_sha256_before": before, "bank_sha256_after": after, "banks_unchanged": True,
        "teacher_input": "immutable current canonical RGB and actual executed command history",
        "training_performed": False, "learned_models_loaded": False,
        "interpretation": "Success proves only that this controller found a continuation from "
            "this state before the original deadline. Failure, timeout or teacher error does not "
            "prove physical impossibility, a single-action cause, or learned-policy quality. "
            "Unchanged full RGB is an evaluator-selected observation, not a proven gameplay error.",
    }, frames_by_anchor)


def _correction_record(config, episode, first_frame, env_factory):
    """Read origin metadata from the actual TRAIN reset, without a policy/controller."""
    env = env_factory(episode.task, split="train", image_size=96, difficulty=1.0,
                      max_steps=config.max_episode_steps, level_bank_seed=episode.level_bank_seed,
                      **({"visual_config": config.visual_config}
                         if config.visual_config is not None else {}))
    try:
        frame, info = env.reset(seed=episode.seed, options={"level_index": episode.level_index})
        _align_frame(frame, first_frame, frame_index=0)
        if (info.get("level_id") != episode.level_id or info.get("split") != "train"
                or info.get("level_index") != episode.level_index
                or info.get("level_bank_seed") != episode.level_bank_seed
                or info.get("difficulty") != 1.0 or info.get("presentation") is not None):
            raise ValueError("TRAIN correction reset differs from its recorded origin")
        _audit_units(info, episode.task)
        return {
            "task": episode.task, "split": "train", "source": episode.source,
            "seed": episode.seed, "level_index": episode.level_index,
            "level_id": info["level_id"], "semantic_level_id": info["semantic_level_id"],
            "runtime_fingerprint": info["runtime_fingerprint"],
            "level_bank_seed": episode.level_bank_seed, "instruction": episode.instruction,
            "initial_frame_sha256": _frame_hash(first_frame), "difficulty": 1.0,
            "presentation": None, "outcome": episode.outcome, "quality": episode.quality,
            "terminated": episode.terminated, "truncated": episode.truncated,
            "learner_prefix_steps": episode.learner_prefix_steps,
            "expert_probability": episode.expert_probability,
        }
    finally:
        env.close()


def diagnose_correction_rescue(config, dataset, episode, *, anchors=(0, 20, 40, 60),
                               stall_window=8, env_factory=AtomicEnv,
                               teacher_factory=scripted_teacher):
    """Authenticate a TRAIN correction and continue its actual history on that TRAIN bank.

    Labels are verified with the standard RGB teacher independently of the attempted
    controller. Actual recorded inputs, rather than those labels, reconstruct every
    anchor. This entry does not evaluate a frozen policy or relabel TRAIN as validation.
    """
    if (episode.source != CORRECTION_SOURCE or episode.split != "train"
            or episode.task not in config.tasks
            or type(episode.difficulty) not in (int, float) or episode.difficulty != 1.0
            or episode.presentation is not None or episode.native_size is not None
            or episode.executed_actions is None or episode.quality is None
            or episode.scoring_version != SCORING_VERSION
            or episode.quality_source != QUALITY_SOURCE
            or config.env_image_size != 96 or type(config.max_episode_steps) is not int
            or not 1 <= config.max_episode_steps <= 80
            or (config.visual_config is not None and config.visual_config.mode != "canonical")):
        raise ValueError("correction rescue requires canonical96 full1 TRAIN corrections "
                         "with actual executed actions and current trajectory quality")
    anchors = _rescue_limits(config, len(episode) + 1, anchors, stall_window)
    # Only this explicit TRAIN entry uses the episode's authenticated source bank.
    source_config = replace(config, level_bank_seed=episode.level_bank_seed)
    with _preserve_rng():
        before = _banks(source_config)
        try:
            dataset._validate_episode(episode)
            source_frames = tuple(_rgb(dataset.frames[index]) for index in
                                  (*episode.frame_indices, episode.final_frame_index))
            if verify_visible_outcome(source_frames[-1]) != episode.outcome:
                raise ValueError("TRAIN correction outcome differs from final visible RGB")
            _validate_replacement_trace(dataset, episode, config.max_episode_steps)
            record = _correction_record(source_config, episode, source_frames[0], env_factory)
            source_actions = [action.to_dict() for action in episode.executed_actions]
            labels = [action.to_dict() for action in episode.actions]
            demonstration_sha = demonstration_episode_digest(dataset, episode)
            trace = {
                "identity": {"split": "train", "source": CORRECTION_SOURCE,
                             "demonstration_episode_sha256": demonstration_sha,
                             "level_bank_seed": episode.level_bank_seed,
                             "level_id": episode.level_id},
                "frame_sha256": [_frame_hash(frame) for frame in source_frames],
                "frame_shapes": [list(frame.shape) for frame in source_frames],
                "executed_actions": source_actions, "teacher_labels": labels,
            }
            source = {"trace_sha256": _json_hash(trace),
                      "actions_sha256": _json_hash(source_actions)}
            stall, attempts, frames_by_anchor = _collect_attempts(
                source_config, record, source_frames, source_actions, anchors, stall_window,
                env_factory=env_factory, teacher_factory=teacher_factory, split="train")
        finally:
            after = _banks(source_config)
            if before != after:
                raise RuntimeError("teacher rescue mutated cached level banks")
    result = _result(source_config, record, source, anchors, stall_window, stall, attempts,
                     frames_by_anchor, before, after, teacher_factory)
    result.report.update(
        source_kind="authenticated_train_correction",
        source_demonstration_sha256=demonstration_sha,
        source_teacher_labels_sha256=_json_hash(labels),
        source_trace_schema="carryrl.train_correction_trace.v1",
        source_authentication_method="dataset validation and standard RGB replacement replay",
        source_episode={key: value for key, value in record.items()
                        if key not in {"outcome", "quality"}})
    return result
