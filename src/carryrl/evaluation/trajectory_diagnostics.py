"""Authenticate and diagnose an existing frozen-policy trajectory without inference.

Replay executes only the recorded physical inputs. Teacher proposals remain
separate annotations; differences do not establish gameplay mistakes or causes.
"""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv, atomic_level_banks, verify_visible_outcome
from carryrl.envs.atomic_suite import SCORING_VERSION, SUITE_VERSION
from carryrl.evaluation.imitation import _immutable_rgb, _preserve_rng
from carryrl.evaluation.quality import QUALITY_VERSION
from carryrl.training.imitation import scripted_teacher
from carryrl.training.sequence import sequence_evaluation_protocol

TRAJECTORY_DIAGNOSTICS_VERSION = "carryrl.frozen_trajectory_diagnostics.v1"


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _frame_hash(frame):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[-1] != 3):
        raise ValueError("trajectory frames must be uint8 HWC RGB")
    return hashlib.sha256(frame.tobytes()).hexdigest()


def _align_frame(actual, recorded, *, frame_index):
    if actual.shape != recorded.shape or _frame_hash(actual) != _frame_hash(recorded):
        raise ValueError(f"replay RGB alignment failed at frame {frame_index}")


def _audit_units(info, task):
    quality = info.get("quality")
    if (not isinstance(quality, Mapping) or quality.get("quality_version") != QUALITY_VERSION
            or quality.get("task") != task):
        raise ValueError("replay requires the current task quality audit")
    units = quality.get("unnecessary_input_units")
    if type(units) is not int or units < 0:
        raise ValueError("replay quality audit requires cumulative unnecessary_input_units")
    if info.get("unnecessary_input_units") != units:
        raise ValueError("replay top-level and quality unnecessary-input counters disagree")
    return units


def diagnose_frozen_trajectory(
    config, record: Mapping, frames: Sequence[np.ndarray], actions: Sequence, *,
    split: str = "validation", env_factory: Callable = AtomicEnv,
    teacher_factory: Callable = scripted_teacher,
) -> dict:
    """Replay a captured ``evaluate_sequence`` callback, failing closed on mismatch.

    ``frames`` contains the initial RGB followed by one RGB per applied action;
    ``actions`` accepts PhysicalAction or codec dictionaries from that callback.
    At most 80 actions are replayed. No learned policy, reward predictor, optimizer
    or additional evaluation episode is called. Validation is the only split.
    """
    if split != "validation" or record.get("split", split) != split:
        raise ValueError("frozen trajectory diagnostics require validation only")
    if record.get("policy") not in {"frozen_deterministic", "frozen_stochastic"}:
        raise ValueError("diagnostics require a real frozen learned-policy trajectory")
    if not actions or len(actions) > min(80, config.max_episode_steps):
        raise ValueError("diagnostics require 1..80 actions within the configured episode budget")
    if len(frames) != len(actions) + 1 or record.get("steps") != len(actions):
        raise ValueError("captured initial/post-action RGB and record steps must align")
    actual_actions = tuple(action if isinstance(action, PhysicalAction)
                           else PhysicalAction.from_dict(action) for action in actions)
    frame_hashes = [_frame_hash(frame) for frame in frames]
    if record.get("initial_frame_sha256") != frame_hashes[0]:
        raise ValueError("captured initial RGB hash does not match the original record")
    protocol = sequence_evaluation_protocol(config, split=split)
    task, seed = record.get("task"), record.get("seed")
    if (task not in config.tasks or type(seed) is not int or not 0 <= seed < 2**32
            or record.get("difficulty") != protocol["difficulty"]
            or record.get("level_bank_seed") != config.level_bank_seed):
        raise ValueError("record provenance does not match the frozen evaluation config")
    level_index = record.get("level_index")
    if config.eval_all_levels:
        level_count = len(atomic_level_banks(task, config.level_bank_seed)[split])
        episode_index = record.get("episode_index")
        if (type(episode_index) is not int or type(level_index) is not int
                or level_index != episode_index % level_count):
            raise ValueError("record does not match the explicit level selection protocol")
    elif level_index is not None:
        raise ValueError("record level selection differs from the seeded sampling protocol")
    for field in ("terminated", "truncated"):
        if type(record.get(field)) is not bool:
            raise ValueError("record requires actual terminated/truncated flags")
    if not (record["terminated"] or record["truncated"]):
        raise ValueError("diagnostics require a complete ended frozen trajectory")
    labels = []
    first_disagreement = first_unnecessary = first_failure = first_terminal = None
    env_options = {"image_size": config.env_image_size, "max_steps": config.max_episode_steps,
                   "difficulty": protocol["difficulty"],
                   "level_bank_seed": config.level_bank_seed}
    if config.visual_config is not None:
        env_options["visual_config"] = config.visual_config
    with _preserve_rng():
        env = env_factory(task, split=split, **env_options)
        try:
            reset_options = {"options": {"level_index": level_index}} \
                if level_index is not None else {}
            frame, initial_info = env.reset(seed=seed, **reset_options)
            _align_frame(frame, frames[0], frame_index=0)
            for field in ("level_id", "semantic_level_id", "runtime_fingerprint", "presentation"):
                if field not in record or initial_info.get(field) != record[field]:
                    raise ValueError(f"replay {field} provenance differs from the original record")
            if (initial_info.get("suite_version") != SUITE_VERSION
                    or initial_info.get("scoring_version") != SCORING_VERSION
                    or initial_info.get("split") != split
                    or initial_info.get("level_bank_seed") != config.level_bank_seed
                    or initial_info.get("difficulty") != protocol["difficulty"]):
                raise ValueError("replay runtime suite/scoring/split provenance mismatch")
            previous_units = _audit_units(initial_info, task)
            if previous_units:
                raise ValueError("replay initial unnecessary-input counter must start at zero")
            canonical = getattr(env, "canonical_render", None)
            if not callable(canonical):
                raise ValueError("trajectory teacher requires canonical RGB of the same live state")
            teacher = teacher_factory(task)
            if not callable(getattr(teacher, "observe_action", None)):
                raise ValueError("trajectory teacher must observe actual executed actions")
            for index, actual_action in enumerate(actual_actions):
                teacher_rgb = _immutable_rgb(canonical())
                teacher_action = teacher.act(teacher_rgb)
                if not isinstance(teacher_action, PhysicalAction):
                    raise TypeError("trajectory RGB teacher must return PhysicalAction")
                labels.append(teacher_action.to_dict())
                context = {"action_index": index, "step": index + 1,
                           "actual_action": actual_action.to_dict(),
                           "rgb_before_sha256": frame_hashes[index]}
                if first_disagreement is None and teacher_action != actual_action:
                    first_disagreement = {
                        **context, "teacher_action": teacher_action.to_dict(),
                        "teacher_canonical_rgb_sha256": _frame_hash(teacher_rgb),
                        "meaning": "First physical-action disagreement; not a proven game error."}
                frame, _, terminated, truncated, info = env.step(actual_action)
                teacher.observe_action(actual_action)
                _align_frame(frame, frames[index + 1], frame_index=index + 1)
                units = _audit_units(info, task)
                if units < previous_units:
                    raise ValueError("replay cumulative unnecessary-input counter decreased")
                if first_unnecessary is None and units > previous_units:
                    first_unnecessary = {
                        **context, "delta_units": units - previous_units,
                        "total_before": previous_units, "total_after": units,
                        "counter": "info.quality.unnecessary_input_units",
                        "meaning": "First confirmed wasted input under the current mechanic audit."}
                previous_units = units
                outcome = verify_visible_outcome(frame)
                if first_failure is None and outcome.get("failure") is True:
                    first_failure = {**context, "outcome": outcome,
                                     "rgb_after_sha256": frame_hashes[index + 1]}
                if terminated or truncated:
                    first_terminal = {**context, "outcome": outcome,
                                      "terminated": bool(terminated), "truncated": bool(truncated),
                                      "rgb_after_sha256": frame_hashes[index + 1]}
                    if index != len(actual_actions) - 1:
                        raise ValueError("replay terminated before the recorded trajectory ended")
                    break
            if (first_terminal is None or terminated != record["terminated"]
                    or truncated != record["truncated"]):
                raise ValueError("replay terminal flags differ from the original trajectory")
            if outcome != record.get("outcome"):
                raise ValueError("replay final visible outcome differs from the original record")
            if info.get("quality") != record.get("quality"):
                raise ValueError("replay final quality audit differs from the original record")
        finally:
            env.close()
    serialized_actions = [action.to_dict() for action in actual_actions]
    return {
        "version": TRAJECTORY_DIAGNOSTICS_VERSION,
        "source": "actual_frozen_learned_policy_trajectory", "split": split,
        "task": task, "policy": record["policy"], "seed": seed,
        "level_index": initial_info.get("level_index"),
        "level_id": record["level_id"], "semantic_level_id": record["semantic_level_id"],
        "runtime_fingerprint": record["runtime_fingerprint"],
        "difficulty": protocol["difficulty"], "level_bank_seed": config.level_bank_seed,
        "presentation": record["presentation"], "steps": len(actual_actions),
        "suite_version": SUITE_VERSION, "scoring_version": SCORING_VERSION,
        "quality_version": QUALITY_VERSION, "replay_verified": True,
        "protocol_sha256": _json_hash(protocol),
        "initial_frame_sha256": frame_hashes[0], "terminal_frame_sha256": frame_hashes[-1],
        "frame_sha256": frame_hashes, "actions_sha256": _json_hash(serialized_actions),
        "trace_sha256": _json_hash({"frame_sha256": frame_hashes,
                                    "frame_shapes": [list(frame.shape) for frame in frames],
                                    "actions": serialized_actions}),
        "teacher_labels_sha256": _json_hash(labels), "actual_actions": serialized_actions,
        "teacher_actions": labels, "outcome": outcome, "quality": info["quality"],
        "first_teacher_disagreement": first_disagreement,
        "first_confirmed_unnecessary_input": first_unnecessary,
        "first_visible_failure": first_failure, "first_terminal_event": first_terminal,
        "interpretation": "Teacher labels use canonical RGB and actual executed history. "
                          "A label mismatch is not a proven gameplay error: action aliases and "
                          "alternative valid actions may differ. Labels are not independently "
                          "certified on learner states. An audited wasted input does not prove "
                          "a causal explanation for failure.",
    }
