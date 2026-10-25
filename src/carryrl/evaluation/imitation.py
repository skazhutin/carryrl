"""Read-only held-out teacher-conditional likelihood; never a training dataset.

The controller supplies labels on its own successful validation trajectories.
This measures imitation under that trajectory distribution, not policy wins.
Only RGB, executed-input history and optional prose reach the evaluated policy.
"""

import hashlib
import json
import random
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import torch

from carryrl.agent.actions import KEY_VOCAB, MOUSE_BINS, MOUSE_BUTTONS, WHEEL_BINS, PhysicalAction
from carryrl.agent.language import validate_instruction
from carryrl.envs.atomic import (
    AtomicEnv,
    _identity,
    atomic_level_banks,
    atomic_semantic_level_id,
    verify_visible_outcome,
)
from carryrl.training.imitation import read_imitation_screen_text, scripted_teacher
from carryrl.training.sequence import (
    _episode_instruction,
    configure_screen_text,
    sequence_evaluation_protocol,
    tensor_hash,
)

VALIDATION_NLL_VERSION = "carryrl.imitation.validation_nll.v1"
COMPONENTS = ("keys", "mouse_buttons", "dx", "dy", "wheel")


def imitation_validation_protocol(config, *, teacher_factory=scripted_teacher,
                                  provided_traces=False):
    """Describe the actual diagnostic cohort before collection or model loading."""
    return {**sequence_evaluation_protocol(config, split="validation"),
            "policies": ["teacher_conditional_likelihood"],
            "level_selection": "explicit_validation_level_indices",
            "diagnostic_version": VALIDATION_NLL_VERSION,
            "teacher_factory": None if provided_traces else _callable_identity(teacher_factory),
            "label_source": ("provided_validation_traces" if provided_traces else
                             "collected_rgb_teacher")}


def _callable_identity(factory):
    return (f"{getattr(factory, '__module__', type(factory).__module__)}."
            f"{getattr(factory, '__qualname__', type(factory).__qualname__)}")


def _immutable_rgb(frame):
    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("validation traces require uint8 HWC RGB")
    # A bytes backing buffer prevents callers from re-enabling write access.
    return np.frombuffer(frame.tobytes(), dtype=np.uint8).reshape(frame.shape)


@dataclass(frozen=True)
class ImitationValidationEpisode:
    """Immutable diagnostic-only trace with labels separate from applied inputs."""

    frames: tuple[np.ndarray, ...]
    target_actions: tuple[PhysicalAction, ...]
    executed_actions: tuple[PhysicalAction, ...]
    task: str
    seed: int
    episode_index: int
    level_index: int
    level_id: str
    semantic_level_id: str
    difficulty: float
    outcome: Mapping
    terminated: bool
    truncated: bool
    instruction: str | None = None
    level_bank_seed: int = 0
    runtime_fingerprint: str | None = None
    presentation_json: str | None = None
    teacher_source: str = "external_labelled_validation_trace"
    split: str = "validation"

    def __post_init__(self):
        if self.split != "validation":
            raise ValueError("imitation diagnostics accept validation only, never train or test")
        if (not self.target_actions or len(self.frames) != len(self.target_actions)
                or len(self.executed_actions) != len(self.target_actions)):
            raise ValueError("validation RGB, targets and executed inputs must match")
        for actions in (self.target_actions, self.executed_actions):
            if any(not isinstance(action, PhysicalAction) for action in actions):
                raise TypeError("validation traces require physical action labels and history")
        if (type(self.level_index) is not int or self.level_index < 0
                or not self.level_id or not self.semantic_level_id):
            raise ValueError("validation traces require explicit level provenance")
        if (type(self.terminated) is not bool or type(self.truncated) is not bool
                or not (self.terminated or self.truncated)):
            raise ValueError("validation traces must be complete ended episodes")
        validate_instruction(self.instruction)
        object.__setattr__(self, "frames", tuple(_immutable_rgb(frame) for frame in self.frames))
        object.__setattr__(self, "target_actions", tuple(self.target_actions))
        object.__setattr__(self, "executed_actions", tuple(self.executed_actions))
        object.__setattr__(self, "outcome", MappingProxyType(dict(self.outcome)))

    @property
    def labels_accepted(self):
        return (self.terminated and not self.truncated and self.outcome.get("known") is True
                and self.outcome.get("success") is True and self.outcome.get("failure") is False)


@contextmanager
def _preserve_rng():
    torch_state, python_state, numpy_state = (torch.random.get_rng_state(), random.getstate(),
                                             np.random.get_state())
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(torch_state)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def collect_imitation_validation(
    config, *, split="validation", env_factory: Callable = AtomicEnv,
    teacher_factory: Callable = scripted_teacher, on_progress: Callable | None = None,
    max_episode_frame_bytes: int = 64 * 1024**2,
):
    """Yield one explicit validation-level teacher trace at a time, including gate failures.

    No corpus or feature cache is retained. A rejected teacher episode remains in
    coverage/outcome reporting but contributes no likelihood labels. The teacher
    sees canonical RGB from the same live state if presentation transforms exist.
    """
    if split != "validation":
        raise ValueError("imitation diagnostics collect validation only")
    if type(max_episode_frame_bytes) is not int or max_episode_frame_bytes < 1:
        raise ValueError("max_episode_frame_bytes must be positive")
    protocol = sequence_evaluation_protocol(config, split="validation")
    total = sum(protocol["counts"].values())
    completed = 0
    for task_index, task in enumerate(config.tasks):
        level_count = len(atomic_level_banks(task, config.level_bank_seed)["validation"])
        for index in range(protocol["counts"][task]):
            seed, level_index = 1_000_000 + index, index % level_count
            event = {"task": task, "episode_index": index, "episodes_total": total,
                     "episodes_completed": completed, "policy": "validation_rgb_teacher"}
            with _preserve_rng():
                action_seed = (config.eval_seed + task_index * 100_000 + index) % 2**32
                random.seed(action_seed)
                np.random.seed(action_seed)
                torch.manual_seed(action_seed)
                options = {"image_size": config.env_image_size,
                           "max_steps": config.max_episode_steps,
                           "difficulty": protocol["difficulty"],
                           "level_bank_seed": config.level_bank_seed}
                if config.visual_config is not None:
                    options["visual_config"] = config.visual_config
                env = env_factory(task, split="validation", **options)
                try:
                    frame, info = env.reset(seed=seed, options={"level_index": level_index})
                    teacher = teacher_factory(task)
                    frames, targets, executed = [], [], []
                    frame_bytes = 0
                    if on_progress is not None:
                        on_progress({**event, "event": "episode_started", "step": 0})
                    for step in range(config.max_episode_steps):
                        frame_bytes += frame.nbytes
                        if frame_bytes > max_episode_frame_bytes:
                            raise MemoryError("validation episode RGB exceeds its byte budget")
                        frames.append(_immutable_rgb(frame))
                        canonical = getattr(env, "canonical_render", None)
                        if config.visual_config is not None and not callable(canonical):
                            raise ValueError("styled validation teacher requires canonical_render")
                        rgb = canonical() if callable(canonical) else frame
                        target = teacher.act(_immutable_rgb(rgb))
                        if not isinstance(target, PhysicalAction):
                            raise TypeError("validation teacher must return PhysicalAction")
                        targets.append(target)
                        # This collector executes the teacher, but records the applied
                        # history independently so replay never substitutes target labels.
                        executed.append(target)
                        frame, _, terminated, truncated, _ = env.step(executed[-1])
                        if callable(getattr(teacher, "observe_action", None)):
                            teacher.observe_action(executed[-1])
                        if on_progress is not None:
                            on_progress({**event, "event": "step", "step": step + 1})
                        if terminated or truncated:
                            break
                    else:
                        raise RuntimeError("validation teacher exceeded its episode budget")
                    episode = ImitationValidationEpisode(
                        frames=tuple(frames), target_actions=tuple(targets),
                        executed_actions=tuple(executed), task=task, seed=seed,
                        episode_index=index, level_index=level_index,
                        level_id=info.get("level_id"),
                        semantic_level_id=info.get("semantic_level_id"),
                        difficulty=protocol["difficulty"], outcome=verify_visible_outcome(frame),
                        terminated=bool(terminated), truncated=bool(truncated),
                        instruction=_episode_instruction(config, task, seed),
                        level_bank_seed=config.level_bank_seed,
                        runtime_fingerprint=info.get("runtime_fingerprint"),
                        presentation_json=json.dumps(info["presentation"], sort_keys=True)
                        if info.get("presentation") is not None else None,
                        teacher_source=_callable_identity(teacher_factory),
                    )
                finally:
                    env.close()
            completed += 1
            if on_progress is not None:
                on_progress({**event, "event": "episode_completed", "step": len(targets),
                             "episodes_completed": completed,
                             "labels_accepted": episode.labels_accepted})
            yield episode


def _component_nll(output, actions):
    """Decode the documented ordered categorical trace, including key EOS masks."""
    trace = output["decoder_trace"]
    key_slots = max(len(action.keys) for action in actions) + 1
    if len(trace) != key_slots + 4:
        raise ValueError("unexpected action decoder trace layout")
    count = len(actions)
    values = torch.zeros((count, len(COMPONENTS)), dtype=torch.float64)
    tokens = torch.zeros_like(values, dtype=torch.int64)
    for step, (logits, mask) in enumerate(trace):
        if step < key_slots:
            component = 0
            targets = [KEY_VOCAB.index(action.keys[step]) if step < len(action.keys)
                       else len(KEY_VOCAB) for action in actions]
        else:
            component = step - key_slots + 1
            if component == 1:
                targets = [sum(1 << bit for bit, button in enumerate(MOUSE_BUTTONS)
                               if button in action.mouse_buttons) for action in actions]
            else:
                bins = WHEEL_BINS if component == 4 else MOUSE_BINS
                attribute = COMPONENTS[component]
                targets = [bins.index(getattr(action, attribute)) for action in actions]
        picked = torch.tensor(targets, device=logits.device)
        nll = -logits.gather(1, picked[:, None]).squeeze(1)
        values[:, component] += torch.where(mask, nll, 0).detach().double().cpu()
        tokens[:, component] += mask.detach().long().cpu()
    joint = -output["log_prob"].detach().double().cpu()
    if (not torch.isfinite(values).all() or not torch.isfinite(joint).all()
            or not torch.allclose(values.sum(1), joint, atol=2e-5, rtol=2e-5)):
        raise FloatingPointError("decoder component likelihoods do not match finite joint NLL")
    return values, tokens


class _NLLTotals:
    """Constant-sized totals; retain no corpus frames, features or per-action losses."""

    def __init__(self):
        self.count = 0
        self.values = torch.zeros(5, dtype=torch.float64)
        self.tokens = torch.zeros(5, dtype=torch.int64)

    def add(self, values, tokens):
        self.count += len(values)
        self.values += values.sum(0)
        self.tokens += tokens.sum(0)
        return self

    def summary(self):
        count, values, tokens = self.count, self.values, self.tokens
        return {"action_count": count, "joint_nll_sum": float(values.sum()),
                "joint_nll": float(values.sum()) / count if count else None,
                "components": {name: {"nll_sum": float(values[index]),
                                      "token_count": int(tokens[index]),
                                      "mean_nll_per_action": float(values[index]) / count
                                      if count else None,
                                      "mean_nll_per_token": float(values[index])
                                      / int(tokens[index])
                                      if tokens[index] else None}
                               for index, name in enumerate(COMPONENTS)}}


def evaluate_imitation_validation(
    agent, config, *, reward=None, split="validation",
    traces: Iterable[ImitationValidationEpisode] | None = None,
    env_factory: Callable = AtomicEnv, teacher_factory: Callable = scripted_teacher,
    on_progress: Callable | None = None, encode_batch_size: int = 8,
    max_episode_frame_bytes: int = 64 * 1024**2,
    checkpoint_weights_revision: int | str | None = None,
) -> dict:
    """Frozen PyTorch likelihood on validation labels, one complete episode at a time.

    MLX callers must synchronize weights to this PyTorch policy first. Returned
    visible outcomes are the teacher's label-quality gate, never learned-policy
    success. Autonomous policy wins belong to evaluate_sequence separately.
    """
    if split != "validation":
        raise ValueError("imitation diagnostics evaluate validation only")
    if type(encode_batch_size) is not int or encode_batch_size < 1:
        raise ValueError("encode_batch_size must be positive")
    screen_reader = configure_screen_text(agent, config)
    owned = traces is None
    protocol = imitation_validation_protocol(
        config, teacher_factory=teacher_factory, provided_traces=not owned)
    modules = [agent] + ([reward] if reward is not None else [])
    hashes = [tensor_hash(module) for module in modules]
    modes = [(child, child.training) for module in modules for child in module.modules()]
    runtime = {name: getattr(agent, name) for name in (
        "_episode_state", "_previous_action", "_instruction", "_instruction_features",
        "_screen_text_step", "last_screen_text_result")
        if hasattr(agent, name)}
    episodes = iter(collect_imitation_validation(
        config, env_factory=env_factory, teacher_factory=teacher_factory,
        on_progress=on_progress, max_episode_frame_bytes=max_episode_frame_bytes)
        if owned else traces)
    records, buckets = [], {name: _NLLTotals() for name in ("all", "active", "idle")}
    task_rows = {task: {name: _NLLTotals() for name in buckets} for task in config.tasks}
    seen = set()
    try:
        with _preserve_rng(), torch.no_grad():
            agent.eval()
            for episode in episodes:
                if not isinstance(episode, ImitationValidationEpisode):
                    raise TypeError("diagnostics require immutable validation-only traces")
                if (episode.task not in task_rows
                        or episode.difficulty != protocol["difficulty"]
                        or episode.level_bank_seed != config.level_bank_seed):
                    raise ValueError("validation trace does not match evaluation protocol")
                bank = atomic_level_banks(episode.task, episode.level_bank_seed)["validation"]
                cohort_key = episode.task, episode.episode_index
                if (type(episode.episode_index) is not int
                        or not 0 <= episode.episode_index < protocol["counts"][episode.task]
                        or cohort_key in seen
                        or episode.level_index != episode.episode_index % len(bank)
                        or episode.seed != 1_000_000 + episode.episode_index
                        or episode.instruction != _episode_instruction(
                            config, episode.task, episode.seed)):
                    raise ValueError("validation trace does not match the explicit episode cohort")
                seen.add(cohort_key)
                if (episode.level_index >= len(bank)
                        or episode.level_id != _identity(bank[episode.level_index])
                        or episode.semantic_level_id != atomic_semantic_level_id(
                            episode.task, bank[episode.level_index])):
                    raise ValueError("validation trace level provenance is outside validation bank")
                digest = hashlib.sha256()
                for frame, target, executed in zip(episode.frames, episode.target_actions,
                                                    episode.executed_actions, strict=True):
                    digest.update(str(frame.shape).encode())
                    digest.update(frame.tobytes())
                    digest.update(json.dumps([target.to_dict(), executed.to_dict()],
                                             sort_keys=True).encode())
                record = {"task": episode.task, "split": episode.split,
                          "seed": episode.seed, "episode_index": episode.episode_index,
                          "level_index": episode.level_index, "level_id": episode.level_id,
                          "semantic_level_id": episode.semantic_level_id,
                          "runtime_fingerprint": episode.runtime_fingerprint,
                          "difficulty": episode.difficulty,
                          "level_bank_seed": episode.level_bank_seed,
                          "presentation": json.loads(episode.presentation_json)
                          if episode.presentation_json is not None else None,
                          "teacher_outcome": dict(episode.outcome),
                          "teacher_source": episode.teacher_source,
                          "trace_sha256": digest.hexdigest(),
                          "labels_accepted": episode.labels_accepted,
                          "steps": len(episode.target_actions)}
                if episode.labels_accepted:
                    if sum(frame.nbytes for frame in episode.frames) > max_episode_frame_bytes:
                        raise MemoryError("validation episode RGB exceeds its byte budget")
                    agent.reset_episode()
                    patches = torch.cat([agent.encode_frames(torch.from_numpy(np.stack(
                        episode.frames[start:start + encode_batch_size])))
                        for start in range(0, len(episode.frames), encode_batch_size)])
                    screen_text = None
                    if screen_reader is not None:
                        from carryrl.perception.screen_text import format_model_text

                        observations = [read_imitation_screen_text(screen_reader, frame, step=step)
                                        for step, frame in enumerate(episode.frames)]
                        screen_text = [format_model_text(item) for item in observations]
                        record["screen_text"] = [
                            {**item.to_dict(), "model_text": text}
                            for item, text in zip(observations, screen_text, strict=True)]
                    output = agent.evaluate_episode(
                        patches, list(episode.target_actions), instruction=episode.instruction,
                        executed_actions=list(episode.executed_actions), return_decoder_trace=True,
                        **({"screen_text": screen_text} if screen_text is not None else {}))
                    values, tokens = _component_nll(output, episode.target_actions)
                    active = torch.tensor([action != PhysicalAction()
                                           for action in episode.target_actions])
                    record["likelihood"] = _NLLTotals().add(values, tokens).summary()
                    buckets["all"].add(values, tokens)
                    buckets["active"].add(values[active], tokens[active])
                    buckets["idle"].add(values[~active], tokens[~active])
                    task_rows[episode.task]["all"].add(values, tokens)
                    task_rows[episode.task]["active"].add(values[active], tokens[active])
                    task_rows[episode.task]["idle"].add(values[~active], tokens[~active])
                    del patches, output
                records.append(record)
            if len(records) != sum(protocol["counts"].values()):
                raise ValueError("validation traces do not cover the declared evaluation cohort")
    finally:
        if owned:
            episodes.close()
        for name, value in runtime.items():
            setattr(agent, name, value)
        for child, mode in modes:
            child.training = mode
        if [tensor_hash(module) for module in modules] != hashes:
            raise RuntimeError("frozen validation mutated policy or reward tensors")
    return {"version": VALIDATION_NLL_VERSION, "split": "validation",
            "backend": "torch_canonical", "diagnostic_only": True,
            "policy_and_reward_frozen": True,
            "agent_sha256": hashes[0], "reward_sha256": hashes[1] if reward is not None else None,
            "checkpoint_weights_revision": checkpoint_weights_revision,
            "conditioning": "teacher trajectory RGB and actual executed history; target prefixes",
            "interpretation": "Teacher-conditional NLL is not independently measured policy wins; "
                              "label disagreement does not prove a gameplay error.",
            "activity_partition": "target differs from the complete neutral physical action",
            "key_eos_included": True,
            "config": config.to_dict(),
            "protocol": protocol,
            "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode())
            .hexdigest(), "records": records,
            "teacher_quality_gate": {"episodes_attempted": len(records),
                                     "episodes_accepted": sum(row["labels_accepted"]
                                                              for row in records),
                                     "episodes_rejected": sum(not row["labels_accepted"]
                                                              for row in records)},
            "summary": {name: totals.summary() for name, totals in buckets.items()},
            "tasks": {task: {**totals["all"].summary(),
                             "active": totals["active"].summary(),
                             "idle": totals["idle"].summary()}
                      for task, totals in task_rows.items()}}
