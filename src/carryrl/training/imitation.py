"""Explicit train-only scripted RGB-teacher behavioural cloning, before optional RL.

These are synthetic demonstrations, never human data or learned-agent results.
Teachers receive only immutable current RGB and retain their own action/cue history.
Outcome and level metadata are provenance; policy replay receives RGB, prior physical
inputs, and optional user-facing prose only. Frozen features precede all learned layers.
"""

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from carryrl.agent.actions import MOUSE_BINS, PhysicalAction
from carryrl.agent.language import validate_instruction
from carryrl.envs.atomic import (
    BACKGROUND,
    BALL,
    PLAYER,
    TARGET,
    TICK_SECONDS,
    AtomicEnv,
    _centroid,
    _identity,
    _visible_navigation_action,
    atomic_level_banks,
    atomic_semantic_level_id,
    verify_visible_outcome,
)
from carryrl.envs.atomic_suite import (
    EXTRA_TASKS,
    SCORING_VERSION,
    SUITE_VERSION,
    TASKS,
    get_extra_controller,
)
from carryrl.envs.runtime_identity import (
    allowed_runtime_train_indices,
    atomic_runtime_level_id,
    runtime_heldout_collisions,
)
from carryrl.evaluation.quality import QUALITY_VERSION, command_input_units
from carryrl.training.sequence import FrozenFrameCache

SOURCE = "scripted_teacher"
NATIVE_SOURCE = "scripted_native_rgb_teacher"
CORRECTION_SOURCE = "scripted_rgb_correction"
MEMORY_COUNTERPART_SOURCE = "scripted_memory_cue_counterpart"
NAVIGATION_GEOMETRY_SOURCE = "scripted_navigation_geometry"
REACTION_DYNAMICS_SOURCE = "scripted_reaction_dynamics"
METHOD = "behavioural_cloning_joint_action_nll"
QUALITY_SOURCE = "actual_executed_trajectory"


def _validate_trajectory_quality(quality, *, task, actions, outcome, scoring_version):
    """Check accounting of executed inputs, independently of teacher target labels."""
    counters = {"commands", "active_commands", "neutral_commands", "input_units",
                "command_units", "unnecessary_input_units"}
    optional_counters = {"optimal_command_units", "excess_command_units", "terminal_payoff",
                         "hits", "misses"}
    expected = counters | optional_counters | {"quality_version", "task", "source",
                                               "game_seconds", "completion_efficiency"}
    if (not isinstance(quality, dict) or set(quality) != expected
            or quality.get("quality_version") != QUALITY_VERSION
            or quality.get("task") != task
            or quality.get("source") != "diagnostic environment accounting; not policy/reward input"
            or scoring_version != SCORING_VERSION):
        raise ValueError("trajectory quality requires current environment accounting and scoring")
    for name in counters | optional_counters:
        value = quality[name]
        if value is None and name in optional_counters:
            continue
        if type(value) is not int or value < 0:
            raise ValueError(f"trajectory quality {name} must be a nonnegative integer")
    for name in ("game_seconds", "completion_efficiency"):
        value = quality[name]
        if value is None and name == "completion_efficiency":
            continue
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < 0:
            raise ValueError(f"trajectory quality {name} must be finite and nonnegative")
    units = sum(command_input_units(action) for action in actions)
    neutral = sum(command_input_units(action) == 0 for action in actions)
    if (quality["commands"] != len(actions) or quality["neutral_commands"] != neutral
            or quality["active_commands"] != len(actions) - neutral
            or quality["input_units"] != units
            or quality["command_units"] != units + neutral
            or quality["unnecessary_input_units"] > units
            or not math.isclose(quality["game_seconds"], len(actions) * TICK_SECONDS)):
        raise ValueError("trajectory quality counters must match actual executed actions")
    success = outcome.get("success") is True and outcome.get("failure") is False
    navigation = ("optimal_command_units", "excess_command_units", "completion_efficiency")
    if task == "navigation":
        optimum, excess, efficiency = (quality[name] for name in navigation)
        if optimum is None:
            raise ValueError("navigation quality requires its shortest cardinal input count")
        if success:
            commands = quality["command_units"]
            if (excess != commands - optimum or efficiency is None
                    or not 0 <= efficiency <= 1
                    or not math.isclose(efficiency, optimum / commands)):
                raise ValueError("navigation quality completion accounting is inconsistent")
        elif excess is not None or efficiency is not None:
            raise ValueError("unfinished navigation cannot certify completion efficiency")
    elif any(quality[name] is not None for name in navigation):
        raise ValueError("only navigation has certified optimal command accounting")
    payoff = quality["terminal_payoff"]
    if task == "goal_choice":
        if ((success and (payoff is None or not 1 <= payoff <= 5))
                or (outcome.get("failure") is True and payoff != 0)
                or (not success and outcome.get("failure") is not True and payoff is not None)):
            raise ValueError("goal quality payoff must describe its actual terminal outcome")
    elif payoff is not None:
        raise ValueError("terminal payoff belongs only to goal_choice")
    for name, supported in (("hits", {"reaction", "mouse", "shooting"}),
                            ("misses", {"mouse", "shooting"})):
        if (quality[name] is None) != (task not in supported):
            raise ValueError(f"trajectory quality {name} is incompatible with its task")
        if quality[name] is not None and quality[name] > len(actions):
            raise ValueError(f"trajectory quality {name} cannot exceed executed commands")


def _validate_pure_teacher_quality(quality, *, task, outcome):
    if (outcome.get("known") is not True or outcome.get("success") is not True
            or outcome.get("failure") is not False
            or quality["unnecessary_input_units"] != 0
            or (task == "navigation" and quality["excess_command_units"] != 0)):
        raise ValueError("pure teacher quality requires visible success, no audited unnecessary "
                         "inputs, and a shortest navigation trajectory")


def _collected_quality(info, *, task, actions, outcome, pure_teacher):
    """Admit real final diagnostics; corrections need not succeed or be efficient."""
    quality = deepcopy(info.get("quality")) if isinstance(info, Mapping) else None
    try:
        _validate_trajectory_quality(quality, task=task, actions=actions, outcome=outcome,
            scoring_version=info.get("scoring_version") if isinstance(info, Mapping) else None)
        if pure_teacher:
            _validate_pure_teacher_quality(quality, task=task, outcome=outcome)
    except ValueError as exc:
        kind = "teacher" if pure_teacher else "trajectory"
        raise RuntimeError(f"scripted {kind} quality gate failed: {task}: {exc}") from exc
    return quality


@dataclass(frozen=True)
class DemonstrationEpisode:
    frame_indices: tuple[int, ...]
    actions: tuple[PhysicalAction, ...]
    final_frame_index: int
    instruction: str | None
    task: str
    seed: int
    level_id: str
    level_index: int
    difficulty: float
    outcome: dict
    split: str = "train"
    source: str = SOURCE
    executed_actions: tuple[PhysicalAction, ...] | None = None
    terminated: bool | None = None
    truncated: bool | None = None
    level_bank_seed: int = 0
    expert_probability: float | None = None
    presentation: dict | None = None
    learner_prefix_steps: int = 0
    origin_level_id: str | None = None
    runtime_level_id: str | None = None
    runtime_level_override: dict | None = None
    native_size: int | None = None
    collection_type: str | None = None
    scoring_version: str | None = None
    quality: dict | None = None
    # Presentation copies retain accounting of their source physical rollout.
    quality_source: str | None = None

    def __post_init__(self):
        if self.split != "train" or self.source not in {
                SOURCE, NATIVE_SOURCE, CORRECTION_SOURCE, MEMORY_COUNTERPART_SOURCE,
                NAVIGATION_GEOMETRY_SOURCE, REACTION_DYNAMICS_SOURCE}:
            raise ValueError("imitation accepts only train scripted_teacher demonstrations "
                             "or scripted_rgb_correction trajectories")
        if not self.actions or len(self.actions) != len(self.frame_indices):
            raise ValueError("whole demonstration RGB rows must match physical actions")
        if any(not isinstance(action, PhysicalAction) for action in self.actions):
            raise TypeError("demonstrations require complete PhysicalAction labels")
        if self.executed_actions is not None and (len(self.executed_actions) != len(self.actions)
                or any(not isinstance(action, PhysicalAction) for action in self.executed_actions)):
            raise ValueError("executed actions must match the complete correction trajectory")
        if self.source == CORRECTION_SOURCE:
            if (self.executed_actions is None or type(self.terminated) is not bool
                    or type(self.truncated) is not bool or not (self.terminated or self.truncated)):
                raise ValueError("corrections require actual inputs and a whole ended trajectory")
            if self.outcome.get("known") is not True:
                raise ValueError("corrections require a known final visible RGB outcome")
        elif (self.outcome.get("known") is not True or self.outcome.get("success") is not True
                or self.outcome.get("failure") is not False):
            raise ValueError("scripted teacher quality gate requires visible success")
        if self.source in {NAVIGATION_GEOMETRY_SOURCE, REACTION_DYNAMICS_SOURCE}:
            expected_task = ("navigation" if self.source == NAVIGATION_GEOMETRY_SOURCE
                             else "reaction")
            if (self.task != expected_task or type(self.difficulty) not in (int, float)
                    or self.difficulty != 1.0 or self.native_size is not None
                    or self.origin_level_id is None or self.runtime_level_id is None
                    or not isinstance(self.runtime_level_override, dict)
                    or self.terminated is not True or self.truncated is not False
                    or self.executed_actions != self.actions or self.quality is None):
                kind = ("navigation geometry" if self.source == NAVIGATION_GEOMETRY_SOURCE
                        else "reaction dynamics")
                raise ValueError(f"{kind} requires canonical full-difficulty "
                                 "teacher actions and complete runtime provenance")
        elif self.source == MEMORY_COUNTERPART_SOURCE:
            if (self.task != "memory" or isinstance(self.difficulty, bool) or self.difficulty != 1.0
                    or type(self.level_index) is not int
                    or self.origin_level_id is None or self.runtime_level_id is None
                    or not isinstance(self.runtime_level_override, dict)):
                raise ValueError("memory counterparts require full-difficulty runtime provenance")
            if self.terminated is not True or self.truncated is not False:
                raise ValueError("memory counterparts require a complete successful teacher trace")
            if self.executed_actions is not None and self.executed_actions != self.actions:
                raise ValueError("memory counterpart executed inputs must equal RGB teacher labels")
        elif self.source == NATIVE_SOURCE:
            if (type(self.native_size) is not int or self.native_size not in (64, 128)
                    or isinstance(self.difficulty, bool) or self.difficulty != 1.0
                    or self.origin_level_id != self.level_id
                    or self.runtime_level_id != self.level_id
                    or self.runtime_level_override is not None
                    or self.terminated is not True or self.truncated is not False):
                raise ValueError("native demonstrations require full-difficulty "
                                 "native RGB provenance")
        elif any(value is not None for value in (
                self.origin_level_id, self.runtime_level_id, self.runtime_level_override)):
            raise ValueError("counterpart runtime provenance requires its explicit source")
        if self.native_size is not None and self.source != NATIVE_SOURCE:
            raise ValueError("native_size requires the explicit native teacher source")
        if self.task not in TASKS:
            raise ValueError("unsupported demonstration task")
        if type(self.learner_prefix_steps) is not int or self.learner_prefix_steps < 0:
            raise ValueError("learner_prefix_steps must be a nonnegative integer")
        if self.learner_prefix_steps and self.source != CORRECTION_SOURCE:
            raise ValueError("learner prefixes belong only to correction trajectories")
        if type(self.level_bank_seed) is not int or not 0 <= self.level_bank_seed < 2**32:
            raise ValueError("level_bank_seed must be a uint32 integer")
        if self.expert_probability is not None and (
                isinstance(self.expert_probability, bool)
                or not isinstance(self.expert_probability, (int, float))
                or not math.isfinite(self.expert_probability)
                or not 0 <= self.expert_probability <= 1):
            raise ValueError("expert_probability provenance must be finite and in [0,1]")
        if self.presentation is not None:
            from carryrl.envs.presentation import THEME_SPLITS, PresentationProfile

            profile = PresentationProfile(**self.presentation)
            if (profile.native_size != (self.native_size or 96)
                    or profile.theme not in THEME_SPLITS["train"]):
                raise ValueError("imitation presentation must match native size "
                                 "and train appearances")
        if self.collection_type is not None and (
                self.collection_type != "live_styled" or self.source != CORRECTION_SOURCE
                or self.presentation is None or self.scoring_version != SCORING_VERSION):
            raise ValueError("live styled collection requires styled corrections "
                             "and current scoring")
        if self.scoring_version is not None and self.scoring_version not in {
                "legacy.v0", SCORING_VERSION}:
            raise ValueError("unsupported demonstration scoring version")
        if self.quality is None:
            if self.quality_source is not None:
                raise ValueError("quality_source requires actual trajectory diagnostics")
        else:
            if self.quality_source != QUALITY_SOURCE:
                raise ValueError("quality_source must identify the actual executed trajectory")
            _validate_trajectory_quality(self.quality, task=self.task,
                actions=(self.executed_actions if self.executed_actions is not None
                         else self.actions),
                outcome=self.outcome, scoring_version=self.scoring_version)
            if self.source != CORRECTION_SOURCE:
                _validate_pure_teacher_quality(self.quality, task=self.task, outcome=self.outcome)
            object.__setattr__(self, "quality", deepcopy(self.quality))
        validate_instruction(self.instruction)

    @property
    def effective_scoring_version(self) -> str:
        return self.scoring_version or "legacy.v0"

    def __len__(self):
        return len(self.actions)

    def to_dict(self) -> dict:
        if self.source in {NAVIGATION_GEOMETRY_SOURCE, REACTION_DYNAMICS_SOURCE}:
            self.__post_init__()
            _episode_runtime_level(self)
        return {
            "source": self.source, "split": self.split, "task": self.task,
            "seed": self.seed, "level_id": self.level_id, "level_index": self.level_index,
            "difficulty": self.difficulty, "instruction": self.instruction,
            "level_bank_seed": self.level_bank_seed,
            "frame_indices": list(self.frame_indices), "final_frame_index": self.final_frame_index,
            "actions": [action.to_dict() for action in self.actions], "outcome": dict(self.outcome),
            **({"executed_actions": [action.to_dict() for action in self.executed_actions],
                "terminated": self.terminated, "truncated": self.truncated}
               if self.executed_actions is not None else {}),
            **({"expert_probability": self.expert_probability,
                "learner_selection": "deterministic", "mixing_rng": "isolated_numpy_generator"}
               if self.source == CORRECTION_SOURCE else {}),
            **({"presentation": dict(self.presentation)} if self.presentation is not None else {}),
            **({"scoring_version": self.scoring_version} if self.scoring_version else {}),
            **({"quality": deepcopy(self.quality), "quality_source": self.quality_source}
               if self.quality is not None else {}),
            **({"collection_type": self.collection_type,
                "teacher_presentation": "canonical RGB of the same live state",
                "appearance_application": "live environment learner rollout"}
               if self.collection_type else {}),
            **({"learner_prefix_steps": self.learner_prefix_steps,
                "learner_prefix_steps_executed": min(self.learner_prefix_steps, len(self.actions)),
                "action_selection_schedule": "learner-only prefix, then expert probability mixture"}
               if self.learner_prefix_steps else {}),
            **({"origin_level_id": self.origin_level_id,
                "runtime_level_id": self.runtime_level_id,
                "runtime_level_override": deepcopy(self.runtime_level_override),
                "terminated": self.terminated, "truncated": self.truncated}
               if self.source in {MEMORY_COUNTERPART_SOURCE, NAVIGATION_GEOMETRY_SOURCE,
                                  REACTION_DYNAMICS_SOURCE} else {}),
            **({"native_size": self.native_size, "origin_level_id": self.origin_level_id,
                "runtime_level_id": self.runtime_level_id,
                "terminated": self.terminated, "truncated": self.truncated,
                "teacher_presentation": "canonical native RGB",
                "appearance_application": ("posthoc replay augmentation" if self.presentation
                                           else "canonical environment rollout")}
               if self.source == NATIVE_SOURCE else {}),
        }


def _episode_runtime_level(episode):
    banks = (atomic_level_banks(episode.task, episode.level_bank_seed)
             if episode.level_bank_seed else atomic_level_banks(episode.task))
    bank = banks["train"]
    if (type(episode.level_index) is not int or not 0 <= episode.level_index < len(bank)
            or _identity(bank[episode.level_index]) != episode.level_id):
        raise ValueError("demonstration level must belong to the training bank")
    runtime = bank[episode.level_index]
    if episode.source == MEMORY_COUNTERPART_SOURCE:
        if episode.origin_level_id != episode.level_id:
            raise ValueError("memory counterpart origin must match its training-bank identity")
        runtime = _flipped_memory_level(runtime)
        if (episode.runtime_level_override != runtime
                or episode.runtime_level_id != _identity(runtime)):
            raise ValueError("counterpart runtime must equal its origin with only cue flipped")
    if episode.source == NAVIGATION_GEOMETRY_SOURCE:
        from carryrl.training.navigation_geometry import validate_navigation_runtime

        runtime = validate_navigation_runtime(episode, runtime)
    if episode.source == REACTION_DYNAMICS_SOURCE:
        from carryrl.training.reaction_dynamics import validate_reaction_runtime

        runtime = validate_reaction_runtime(episode, runtime)
    return runtime


def validate_dataset_runtime_holdout(dataset, *, level_bank_seed: int,
                                     evaluation_difficulty: float) -> None:
    """Reject source or endpoint held-out matches before corpus publication/encoding.

    Runtime provenance is reconstructed from each source bank; this check reads
    episode records only and never changes the rows or the evaluation protocol.
    """
    if type(level_bank_seed) is not int or not 0 <= level_bank_seed < 2**32:
        raise ValueError("level_bank_seed must be a uint32 integer")
    if (isinstance(evaluation_difficulty, bool)
            or not isinstance(evaluation_difficulty, (int, float))
            or not math.isfinite(evaluation_difficulty) or not 0 <= evaluation_difficulty <= 1):
        raise ValueError("runtime evaluation difficulty must be finite and in [0,1]")
    for episode in dataset.episodes:
        if not isinstance(episode, DemonstrationEpisode):
            raise ValueError("runtime hold-out validation requires demonstration episodes")
        episode.__post_init__()
        runtime = _episode_runtime_level(episode)
        for bank_seed in dict.fromkeys((episode.level_bank_seed, level_bank_seed)):
            matches = runtime_heldout_collisions(episode.task, runtime, episode.difficulty,
                bank_seed, heldout_difficulties=(evaluation_difficulty,))
            if matches:
                raise ValueError("demonstration overlaps a held-out runtime semantic level "
                    f"for evaluation endpoint: {episode.task}, source bank "
                    f"{episode.level_bank_seed}, level {episode.level_index}; match {matches[0]}")


class DemonstrationDataset:
    """Exact unique RGB storage with a hard byte budget and full causal episodes."""

    def __init__(self, *, max_frame_bytes: int = 256 * 1024**2):
        if type(max_frame_bytes) is not int or max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be a positive integer")
        self.max_frame_bytes = max_frame_bytes
        self.frame_bytes = 0
        self.frames: list[np.ndarray] = []
        self.episodes: list[DemonstrationEpisode] = []
        self._index: dict[tuple, int] = {}
        self.memory_cue_counterpart_reports: list[dict] = []

    def intern_frame(self, frame: np.ndarray) -> int:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("demonstration frames must be uint8 HWC RGB")
        key = (frame.shape, hashlib.sha256(frame.tobytes()).digest())
        if key in self._index:
            return self._index[key]
        if self.frame_bytes + frame.nbytes > self.max_frame_bytes:
            raise MemoryError("scripted demonstrations exceed the configured RGB byte budget")
        # Immutable bytes backing prevents teachers from re-enabling ndarray writes.
        stored = np.frombuffer(frame.tobytes(), dtype=np.uint8).reshape(frame.shape)
        index = len(self.frames)
        self._index[key] = index
        self.frames.append(stored)
        self.frame_bytes += stored.nbytes
        return index

    def _validate_episode(self, episode: DemonstrationEpisode) -> None:
        episode.__post_init__()
        indices = (*episode.frame_indices, episode.final_frame_index)
        if any(type(index) is not int or not 0 <= index < len(self.frames) for index in indices):
            raise ValueError("demonstration references a missing RGB frame")
        for index in indices:
            frame = self.frames[index]
            if (frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3
                    or frame.flags.writeable):
                raise ValueError("demonstrations require immutable uint8 HWC RGB")
            try:
                frame.setflags(write=True)
            except ValueError:
                pass
            else:
                frame.setflags(write=False)
                raise ValueError("demonstration RGB must have immutable backing storage")
        if episode.presentation is not None:
            from carryrl.envs.presentation import PresentationProfile

            width, height = PresentationProfile(**episode.presentation).canvas_size
            if any(self.frames[index].shape != (height, width, 3) for index in indices):
                raise ValueError("episode RGB must match its stable presentation canvas")
        elif episode.native_size is not None:
            if any(self.frames[index].shape != (episode.native_size, episode.native_size, 3)
                   for index in indices):
                raise ValueError("native episode RGB must match its declared native_size")
        runtime = _episode_runtime_level(episode)
        if runtime_heldout_collisions(episode.task, runtime, episode.difficulty,
                                      episode.level_bank_seed):
            raise ValueError("demonstration overlaps a held-out runtime semantic level")
        visible = verify_visible_outcome(self.frames[episode.final_frame_index])
        if visible != episode.outcome:
            raise ValueError("demonstration outcome must match its final visible RGB")
        if episode.source == MEMORY_COUNTERPART_SOURCE:
            _validate_memory_teacher_trace(self, episode, runtime)
        if episode.source == NAVIGATION_GEOMETRY_SOURCE:
            from carryrl.training.navigation_geometry import validate_navigation_teacher_trace

            validate_navigation_teacher_trace(self, episode, runtime)
        if episode.source == REACTION_DYNAMICS_SOURCE:
            from carryrl.training.reaction_dynamics import validate_reaction_teacher_trace

            validate_reaction_teacher_trace(self, episode, runtime)
    def add_episode(self, episode: DemonstrationEpisode) -> None:
        self._validate_episode(episode)
        if episode.native_size is not None:
            key = (episode.task, episode.level_bank_seed, episode.level_id, episode.native_size,
                   episode.instruction, episode.presentation)
            if any((row.task, row.level_bank_seed, row.level_id, row.native_size,
                    row.instruction, row.presentation) == key for row in self.episodes):
                raise ValueError("duplicate native demonstration origin")
        self.episodes.append(episode)

    def extend(self, other: "DemonstrationDataset") -> None:
        """Combine separately collected train difficulty stages, deduplicating RGB."""
        mapping = [self.intern_frame(frame) for frame in other.frames]
        for episode in other.episodes:
            self.add_episode(replace(episode,
                frame_indices=tuple(mapping[index] for index in episode.frame_indices),
                final_frame_index=mapping[episode.final_frame_index]))
        if getattr(other, "memory_cue_counterpart_reports", None):
            self.memory_cue_counterpart_reports.extend(deepcopy(other.memory_cue_counterpart_reports))

    def provenance(self) -> dict:
        sources = sorted({episode.source for episode in self.episodes})
        augmented = [episode for episode in self.episodes if episode.presentation is not None]
        native = [episode for episode in self.episodes if episode.native_size is not None]
        appearance_sizes = sorted({episode.native_size or 96 for episode in augmented})
        counterparts = [episode for episode in self.episodes
                        if episode.source == MEMORY_COUNTERPART_SOURCE]
        counterpart_reports = getattr(self, "memory_cue_counterpart_reports", [])
        live_styled = [episode for episode in self.episodes
                       if episode.collection_type == "live_styled"]
        schedules = {}
        if any(episode.learner_prefix_steps for episode in self.episodes):
            for episode in self.episodes:
                if episode.source == CORRECTION_SOURCE:
                    key = episode.learner_prefix_steps, episode.expert_probability
                    schedules[key] = schedules.get(key, 0) + 1
        return {
            "source": sources[0] if len(sources) == 1 else (
                "mixed_scripted_rgb" if sources else SOURCE),
            "sources": sources, "method": METHOD, "split": "train",
            "suite_version": SUITE_VERSION,
            "teacher_input": "immutable canonical RGB and executed action/cue history",
            "actor_input": "RGB, own previous physical actions, optional prose instruction",
            "outcome_source": "independent RGB HUD verifier; demonstrations require success",
            "teacher_presentation": "canonical only; varied-presentation competence unverified",
            "actor_presentation": ("canonical_and_train_live_styled_corrections"
                if live_styled else "canonical_and_train_appearance_augmentation"
                if augmented and len(augmented) != len(self.episodes) else
                "train_appearance_augmentation" if augmented else "canonical"),
            "appearance_augmentation": {
                "episodes": len(augmented), "native_size": (
                    appearance_sizes[0] if len(appearance_sizes) == 1 else None),
                "themes": sorted({episode.presentation["theme"] for episode in augmented}),
                "hud_layouts": sorted({episode.presentation["hud_layout"]
                                       for episode in augmented}),
                "labels": "recorded canonical teacher targets; physical inputs unchanged",
                "profile_input_to_actor": False,
                **({"application": "posthoc copies and live styled learner corrections",
                    "live_learner_episodes": len(live_styled)} if live_styled else {}),
                **({"native_sizes": appearance_sizes,
                    "application": "posthoc canonical RGB replay; no styled learner rollout"}
                   if native and not live_styled else {}),
            },
            "frame_shapes": [list(shape) for shape in sorted({frame.shape
                                                             for frame in self.frames})],
            "episodes": [episode.to_dict() for episode in self.episodes],
            "unique_frames": len(self.frames), "frame_bytes": self.frame_bytes,
            "max_frame_bytes": self.max_frame_bytes,
            **({"scoring_versions": sorted({row.effective_scoring_version
                                           for row in self.episodes})}
               if any(row.scoring_version for row in self.episodes) else {}),
            **({"live_styled_corrections": {
                "episodes": len(live_styled), "split": "train", "native_size": 96,
                "teacher_input": "immutable canonical RGB of the same live state",
                "learner_input": "styled RGB and own executed physical action history",
                "executed_history": "actual mixture inputs observed by learner and teacher",
                "heldout_exclusion": "semantic validation/test matches in banks 0, 20261008 "
                                     "and the collection bank",
            }} if live_styled else {}),
            **({"native_demonstrations": {
                "episodes": len(native),
                "native_sizes": sorted({row.native_size for row in native}),
                "canonical_rollouts": sum(row.presentation is None for row in native),
                "posthoc_appearance_copies": sum(row.presentation is not None for row in native),
                "environment_presentation": "canonical native RGB",
                "styled_learner_rollouts": False, "split": "train",
            }} if native else {}),
            **({"correction_schedules": [
                {"learner_prefix_steps": prefix, "expert_probability_after_prefix": probability,
                 "episodes": count}
                for (prefix, probability), count in schedules.items()]} if schedules else {}),
            **({"memory_cue_counterparts": {
                "episodes": len(counterparts), "origin_split": "train",
                "runtime_change": "cue flip only; same geometry and timing",
                "heldout_exclusion": "semantic validation/test matches in banks 0 and origin seed",
                "runtime_metadata_input_to_actor": False,
                "reports": deepcopy(counterpart_reports),
            }} if counterparts or counterpart_reports else {}),
        }


def augment_demonstrations(
    dataset: DemonstrationDataset, *, seed: int = 0, copies_per_episode: int = 1,
    max_frame_bytes: int = 256 * 1024**2,
    themes: Sequence[str] | None = None, hud_layouts: Sequence[str] | None = None,
    textures: bool = True, canvas_padding: int = 8,
) -> DemonstrationDataset:
    """Return train appearance copies of canonical episodes without modifying originals.

    Each copy uses one stable size-matched profile for every RGB frame, including its
    final verifier frame. Placement is translation only, preserving physical deltas.
    Teachers are never queried on transformed RGB: their recorded labels and the
    actual input history remain unchanged. Extend the original dataset explicitly
    when both canonical and transformed episodes should be used for training.
    Optional appearance controls only select train themes and placement; they do
    not resize the native viewport or change the physical trajectory.
    """
    from carryrl.envs.presentation import (
        HUD_LAYOUTS,
        THEME_SPLITS,
        PresentationProfile,
        present_frame,
    )

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("augmentation seed must be a uint32 integer")
    if type(copies_per_episode) is not int or copies_per_episode < 1:
        raise ValueError("copies_per_episode must be a positive integer")
    selections = []
    for name, values, allowed in (("themes", themes, THEME_SPLITS["train"]),
                                  ("hud_layouts", hud_layouts, HUD_LAYOUTS)):
        try:
            selected = tuple(allowed if values is None else values)
        except TypeError as error:
            raise ValueError(f"augmentation {name} must be a nonempty unique selection") from error
        if (not selected or any(not isinstance(value, str) or value not in allowed
                                for value in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError(f"augmentation {name} must be a nonempty unique selection "
                             f"from {allowed}")
        selections.append(selected)
    themes, hud_layouts = selections
    if type(textures) is not bool:
        raise ValueError("augmentation textures must be a boolean")
    if type(canvas_padding) is not int or not 4 <= canvas_padding <= 64:
        raise ValueError("augmentation canvas_padding must be an integer in 4..64")
    augmented = DemonstrationDataset(max_frame_bytes=max_frame_bytes)
    # Validate all inputs before starting transformations; bank/split checks remain
    # identical to collection and training, including manually assembled datasets.
    for episode in dataset.episodes:
        dataset._validate_episode(episode)
        if episode.presentation is not None or any(
                dataset.frames[index].shape != (episode.native_size or 96,
                                               episode.native_size or 96, 3)
                for index in (*episode.frame_indices, episode.final_frame_index)):
            raise ValueError("augmentation requires canonical native 96x96 RGB episodes "
                             "or explicitly recorded native 64/128 episodes")
    rng = np.random.default_rng(seed)
    for episode in dataset.episodes:
        for _ in range(copies_per_episode):
            profile = PresentationProfile(
                native_size=episode.native_size or 96,
                theme=str(rng.choice(themes)),
                hud_layout=str(rng.choice(hud_layouts)),
                texture_seed=int(rng.integers(2**31)), textures=textures,
                canvas_padding=canvas_padding,
            )
            # Exact repeated RGB within a trajectory receives the exact same style.
            mapping = {}
            for index in (*episode.frame_indices, episode.final_frame_index):
                if index not in mapping:
                    transformed = present_frame(dataset.frames[index], profile)
                    if verify_visible_outcome(transformed) != verify_visible_outcome(
                            dataset.frames[index]):
                        raise ValueError("appearance augmentation changed the visible RGB outcome")
                    mapping[index] = augmented.intern_frame(transformed)
            augmented.add_episode(replace(episode,
                frame_indices=tuple(mapping[index] for index in episode.frame_indices),
                final_frame_index=mapping[episode.final_frame_index], presentation=asdict(profile)))
    return augmented


class _CoreRGBTeacher:
    def __init__(self, task: str):
        self.task = task
        self.previous_buttons = ()
        self.memory_choice = None

    def observe_action(self, action: PhysicalAction) -> None:
        self.previous_buttons = action.mouse_buttons

    def act(self, frame: np.ndarray) -> PhysicalAction:
        if self.task == "navigation":
            action = _visible_navigation_action(frame, False)
        elif self.task == "reaction":
            ball, paddle = _centroid(frame, BALL), _centroid(frame, PLAYER)
            if ball is None or paddle is None:
                raise RuntimeError("reaction teacher cannot recognize visible objects")
            difference = ball[0] - paddle[0]
            action = PhysicalAction(keys=() if abs(difference) < 3 else
                                    ("right" if difference > 0 else "left",))
        elif self.task == "mouse":
            cursor, target = _centroid(frame, PLAYER), _centroid(frame, TARGET)
            if cursor is None or target is None:
                raise RuntimeError("mouse teacher cannot recognize visible objects")
            dx, dy = target[0] - cursor[0], target[1] - cursor[1]
            buttons = (("left",) if abs(dx) <= 3 and abs(dy) <= 3
                       and "left" not in self.previous_buttons else ())
            action = PhysicalAction(mouse_buttons=buttons,
                dx=min(MOUSE_BINS, key=lambda value: abs(value - dx)),
                dy=min(MOUSE_BINS, key=lambda value: abs(value - dy)))
        elif self.task == "memory":
            cue = _centroid(frame, TARGET)
            if cue is not None:
                self.memory_choice = "left" if cue[0] < frame.shape[1] / 2 else "right"
            if self.memory_choice is None:
                raise RuntimeError("memory teacher requires its own initial RGB cue")
            # Read the visible phase label. Holding the remembered key through Wait
            # would leak the vanished cue into the learner's previous-action input.
            from PIL import Image, ImageDraw, ImageFont

            from carryrl.rewards.visual import HUD_HEIGHT

            template = Image.new("RGB", (frame.shape[1], frame.shape[0]), BACKGROUND)
            ImageDraw.Draw(template).text((frame.shape[1] // 2 - 18, HUD_HEIGHT + 7),
                                         "Choose", fill=BALL,
                                         font=ImageFont.load_default(size=10))
            region = np.asarray(template)
            choose_pixels = np.any(region != BACKGROUND, axis=-1)
            # At native 64 the lower glyph edges blend into the doors. Read the
            # visible upper glyphs on the background instead of assuming all text
            # is drawn over empty space; native 96 retains its original mask.
            choose_pixels[frame.shape[0] - 24:] = False
            ready = np.array_equal(frame[choose_pixels], region[choose_pixels])
            action = PhysicalAction(keys=(self.memory_choice,) if ready else ())
        else:
            raise ValueError("unsupported core RGB teacher task")
        self.previous_buttons = action.mouse_buttons
        return action


def scripted_teacher(task: str):
    return get_extra_controller(task, fail=False) if task in EXTRA_TASKS else _CoreRGBTeacher(task)


def _flipped_memory_level(origin: dict) -> dict:
    runtime = deepcopy(origin)
    if type(runtime.get("cue")) is not int or runtime["cue"] not in (0, 1):
        raise ValueError("memory origin must have a binary cue")
    runtime["cue"] = 1 - runtime["cue"]
    return runtime


def _memory_heldout_collisions(runtime: dict, level_bank_seed: int) -> list[dict]:
    return runtime_heldout_collisions("memory", runtime, 1.0, level_bank_seed)


class _MemoryCueRuntimeEnv(AtomicEnv):
    """Local train-only runtime override; bank dictionaries are never modified."""

    def __init__(self, runtime: dict, *, level_bank_seed: int, max_steps: int):
        super().__init__("memory", split="train", difficulty=1.0, image_size=96,
                         max_steps=max_steps, level_bank_seed=level_bank_seed)
        self._counterpart_runtime = deepcopy(runtime)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._level = deepcopy(self._counterpart_runtime)
        self._effective_level_id = _identity(self._level)
        self._semantic_level_id = atomic_semantic_level_id("memory", self._level)
        self._frame = self._draw()
        return self._frame.copy(), self._info(verify_visible_outcome(self._frame))


def _validate_memory_teacher_trace(dataset, episode, runtime):
    """Authenticate complete RGB/action provenance against the recorded runtime."""
    env = _MemoryCueRuntimeEnv(runtime, level_bank_seed=episode.level_bank_seed,
                               max_steps=max(80, len(episode)))
    teacher = scripted_teacher("memory")
    try:
        frame, _ = env.reset(seed=episode.seed, options={"level_index": episode.level_index})
        for step, (index, action) in enumerate(zip(
                episode.frame_indices, episode.actions, strict=True)):
            expected = frame
            if episode.presentation is not None:
                from carryrl.envs.presentation import PresentationProfile, present_frame

                expected = present_frame(frame, PresentationProfile(**episode.presentation))
            if not np.array_equal(dataset.frames[index], expected):
                raise ValueError("memory teacher trace RGB does not match its declared runtime")
            rgb = np.frombuffer(frame.tobytes(), dtype=np.uint8).reshape(frame.shape)
            if action != teacher.act(rgb):
                raise ValueError("memory trace labels must come from the neutral-wait RGB teacher")
            frame, _, terminated, truncated, final_info = env.step(action)
            teacher.observe_action(action)
            if (terminated or truncated) and step != len(episode) - 1:
                raise ValueError("memory teacher trace must be a whole episode")
        if episode.presentation is not None:
            frame = present_frame(frame, PresentationProfile(**episode.presentation))
        if not np.array_equal(dataset.frames[episode.final_frame_index], frame):
            raise ValueError("memory teacher final RGB does not match its declared runtime")
        if not terminated or truncated or not verify_visible_outcome(frame)["success"]:
            raise ValueError("memory teacher trace must end in visible success")
        if episode.quality is not None and episode.quality != final_info.get("quality"):
            raise ValueError("memory quality does not match its actual executed trajectory")
    finally:
        env.close()


def collect_memory_cue_counterparts(
    dataset: DemonstrationDataset, *, max_pairs: int = 64, max_steps: int = 80,
    max_frame_bytes: int = 256 * 1024**2,
    teacher_factory: Callable = scripted_teacher,
    on_episode: Callable[[dict], None] | None = None,
) -> DemonstrationDataset:
    """Add only safe opposite-cue peers of existing canonical train memory demos.

    Full-difficulty teacher origins retain geometry, cue duration, unlock delay and
    prose instruction. Every teacher sees immutable RGB only. Semantic matches to
    any validation/test level in bank zero or the origin's bank are excluded before
    rollout. The returned dataset contains counterpart episodes only; its report
    distinguishes accepted, skipped and unattempted origins.
    """
    if type(max_pairs) is not int or not 1 <= max_pairs <= 64:
        raise ValueError("max_pairs must be an integer in [1,64]")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    result = DemonstrationDataset(max_frame_bytes=max_frame_bytes)
    origins, seen = [], set()
    for episode in dataset.episodes:
        if (episode.task != "memory" or episode.source != SOURCE or episode.difficulty != 1.0
                or episode.presentation is not None):
            continue
        dataset._validate_episode(episode)
        if any(dataset.frames[index].shape != (96, 96, 3)
               for index in (*episode.frame_indices, episode.final_frame_index)):
            raise ValueError("memory cue counterparts require canonical 96x96 RGB origins")
        key = episode.level_bank_seed, episode.level_id, episode.instruction
        if key not in seen:
            seen.add(key)
            origins.append(episode)
    report = {"max_pairs": max_pairs, "eligible_origins": len(origins), "origins_attempted": 0,
              "pairs_collected": 0, "skipped_origins": [], "unattempted_origins": len(origins)}
    result.memory_cue_counterpart_reports.append(report)
    for origin in origins:
        if len(result.episodes) >= max_pairs:
            break
        report["origins_attempted"] += 1
        report["unattempted_origins"] -= 1
        original_level = atomic_level_banks("memory", origin.level_bank_seed)["train"][
            origin.level_index]
        _validate_memory_teacher_trace(dataset, origin, original_level)
        runtime = _flipped_memory_level(original_level)
        collisions = _memory_heldout_collisions(runtime, origin.level_bank_seed)
        if collisions:
            skipped = {"origin_level_id": origin.level_id, "level_index": origin.level_index,
                       "level_bank_seed": origin.level_bank_seed,
                       "reason": "heldout_semantic_collision", "matches": collisions}
            report["skipped_origins"].append(skipped)
            if on_episode is not None:
                on_episode({"task": "memory", "source": MEMORY_COUNTERPART_SOURCE,
                    "status": "skipped", "pairs_collected": len(result.episodes),
                    "eligible_origins": len(origins),
                    "origins_attempted": report["origins_attempted"],
                    **skipped})
            continue
        env = _MemoryCueRuntimeEnv(runtime, level_bank_seed=origin.level_bank_seed,
                                   max_steps=max_steps)
        try:
            frame, _ = env.reset(seed=origin.seed, options={"level_index": origin.level_index})
            teacher = teacher_factory("memory")
            indices, actions = [], []
            for _ in range(max_steps):
                index = result.intern_frame(frame)
                action = teacher.act(result.frames[index])
                if not isinstance(action, PhysicalAction):
                    raise TypeError("memory counterpart teacher must return a PhysicalAction")
                indices.append(index)
                actions.append(action)
                frame, _, terminated, truncated, final_info = env.step(action)
                if callable(getattr(teacher, "observe_action", None)):
                    teacher.observe_action(action)
                if terminated or truncated:
                    break
            outcome = verify_visible_outcome(frame)
            if (not terminated or truncated or outcome.get("success") is not True
                    or outcome.get("known") is not True or outcome.get("failure") is not False):
                raise RuntimeError("memory counterpart teacher quality gate failed")
            quality = _collected_quality(final_info, task="memory", actions=actions,
                                         outcome=outcome, pure_teacher=True)
            result.add_episode(DemonstrationEpisode(tuple(indices), tuple(actions),
                result.intern_frame(frame), origin.instruction, "memory", origin.seed,
                origin.level_id, origin.level_index, 1.0, dict(outcome),
                source=MEMORY_COUNTERPART_SOURCE, terminated=True, truncated=False,
                scoring_version=SCORING_VERSION, quality=quality, quality_source=QUALITY_SOURCE,
                level_bank_seed=origin.level_bank_seed, origin_level_id=origin.level_id,
                runtime_level_id=_identity(runtime), runtime_level_override=deepcopy(runtime)))
            report["pairs_collected"] = len(result.episodes)
            if on_episode is not None:
                on_episode({"task": "memory", "source": MEMORY_COUNTERPART_SOURCE,
                    "status": "collected", "pairs_collected": len(result.episodes),
                    "eligible_origins": len(origins),
                    "origins_attempted": report["origins_attempted"],
                    "origin_level_id": origin.level_id, "level_index": origin.level_index,
                    "level_bank_seed": origin.level_bank_seed, "steps": len(actions)})
        finally:
            env.close()
    return result


def _validate_collection(episodes_per_task, tasks, seed, difficulty, image_size, max_steps,
                         episode_instructions, split, native_size=None):
    tasks = tuple(tasks)
    if split != "train":
        raise ValueError("scripted imitation collection is train-only")
    if type(episodes_per_task) is not int or episodes_per_task < 1:
        raise ValueError("episodes_per_task must be positive")
    if type(seed) is not int or not 0 <= seed < 2**32 - 160000 - episodes_per_task:
        raise ValueError("seed must leave room for unique uint32 episode seeds")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in TASKS for task in tasks):
        raise ValueError("tasks must be a nonempty unique atomic task selection")
    if native_size is not None:
        if (type(native_size) is not int or native_size not in (64, 128)
                or image_size != native_size or difficulty != 1.0
                or episodes_per_task > 64):
            raise ValueError("native collection requires size 64/128, full difficulty, "
                             "and at most 64 unique train origins per task")
    elif image_size != 96:
        raise ValueError("scripted imitation teachers are quality-gated at canonical 96 pixels")
    if (isinstance(difficulty, bool) or not isinstance(difficulty, (int, float))
            or not math.isfinite(difficulty) or not 0 <= difficulty <= 1):
        raise ValueError("difficulty must be finite and in [0,1]")
    instructions = episode_instructions or {}
    for task in tasks:
        if task in instructions and (isinstance(instructions[task], str)
                or not instructions[task]
                or any(validate_instruction(text) is None for text in instructions[task])):
            raise ValueError("instruction variants must be nonempty prose strings")
    return tasks, instructions


def collect_scripted_demonstrations(
    episodes_per_task: int = 8, *, tasks: Sequence[str] = TASKS, seed: int = 0,
    difficulty: float = 1.0, image_size: int = 96, max_steps: int = 80,
    episode_instructions: Mapping[str, Sequence[str]] | None = None,
    max_frame_bytes: int = 256 * 1024**2, split: str = "train",
    env_factory: Callable = AtomicEnv, teacher_factory: Callable = scripted_teacher,
    on_episode: Callable[[dict], None] | None = None,
    level_bank_seed: int = 0,
    native_size: int | None = None,
) -> DemonstrationDataset:
    """Record successful whole episodes; any teacher failure aborts the quality gate.

    Explicit level indices cover the training bank before repeating. No held-out
    demonstration collection is permitted. Diagnostic env info/reward never enters
    the teacher; only level_id is copied to provenance after the RGB-only rollout.
    """
    tasks, instructions = _validate_collection(episodes_per_task, tasks, seed, difficulty,
        image_size, max_steps, episode_instructions, split, native_size)
    if type(level_bank_seed) is not int or not 0 <= level_bank_seed < 2**32:
        raise ValueError("level_bank_seed must be a uint32 integer")
    dataset = DemonstrationDataset(max_frame_bytes=max_frame_bytes)
    allowed_by_task = {task: allowed_runtime_train_indices(task, level_bank_seed, difficulty)
                       for task in tasks}
    for task, allowed_indices in allowed_by_task.items():
        if native_size is not None:
            bank = atomic_level_banks(task, level_bank_seed)["train"]
            unique_indices = {}
            for level_index in allowed_indices:
                unique_indices.setdefault(atomic_runtime_level_id(
                    task, bank[level_index], difficulty), level_index)
            allowed_by_task[task] = list(unique_indices.values())
            if len(unique_indices) < episodes_per_task:
                raise ValueError("native training bank lacks enough unique runtime levels "
                                 "without held-out semantic matches")
    for task in tasks:
        allowed_indices = allowed_by_task[task]
        env = env_factory(task, split="train", image_size=image_size,
                          max_steps=max_steps, difficulty=difficulty,
                          **({"level_bank_seed": level_bank_seed} if level_bank_seed else {}))
        try:
            for index in range(episodes_per_task):
                episode_seed = seed + 10000 * TASKS.index(task) + index
                level_index = allowed_indices[(seed + index) % len(allowed_indices)]
                frame, info = env.reset(seed=episode_seed, options={"level_index": level_index})
                level_id = info["level_id"]
                teacher = teacher_factory(task)
                frame_indices, actions = [], []
                for _ in range(max_steps):
                    if native_size is not None and frame.shape != (native_size, native_size, 3):
                        raise ValueError("native environment RGB does not match "
                                         "requested native_size")
                    frame_index = dataset.intern_frame(frame)
                    action = teacher.act(dataset.frames[frame_index])
                    if not isinstance(action, PhysicalAction):
                        raise TypeError("scripted teacher must return a complete PhysicalAction")
                    frame_indices.append(frame_index)
                    actions.append(action)
                    frame, _, terminated, truncated, final_info = env.step(action)
                    if terminated or truncated:
                        break
                if native_size is not None and frame.shape != (native_size, native_size, 3):
                    raise ValueError("native environment RGB does not match requested native_size")
                outcome = verify_visible_outcome(frame)
                if (outcome.get("known") is not True or outcome.get("success") is not True
                        or outcome.get("failure") is not False):
                    native_context = f", native_size {native_size}" if native_size else ""
                    raise RuntimeError(f"scripted teacher quality gate failed: {task}, "
                                       f"level {level_index}, seed {episode_seed}"
                                       f"{native_context}: {outcome}")
                if native_size is not None and (terminated is not True or truncated is not False):
                    raise RuntimeError("native teacher success must end its environment trajectory")
                quality = _collected_quality(final_info, task=task, actions=actions,
                                             outcome=outcome, pure_teacher=True)
                variants = instructions.get(task)
                instruction = variants[index % len(variants)] if variants else None
                dataset.add_episode(DemonstrationEpisode(
                    tuple(frame_indices), tuple(actions), dataset.intern_frame(frame), instruction,
                    task, episode_seed, level_id, level_index, float(difficulty), dict(outcome),
                    level_bank_seed=level_bank_seed, scoring_version=SCORING_VERSION,
                    quality=quality, quality_source=QUALITY_SOURCE,
                    **({"source": NATIVE_SOURCE, "native_size": native_size,
                        "origin_level_id": level_id, "runtime_level_id": level_id,
                        "terminated": terminated, "truncated": truncated}
                       if native_size is not None else {})))
                if on_episode is not None:
                    on_episode({"task": task, "episodes_completed": len(dataset.episodes),
                                "episodes_total": len(tasks) * episodes_per_task,
                                "steps": len(actions), "unique_frames": len(dataset.frames)})
        finally:
            env.close()
    return dataset


def collect_native_scripted_demonstrations(
    *, native_train_sizes: Sequence[int] = (), native_episodes_per_task: int = 0,
    max_frame_bytes: int = 256 * 1024**2, **collection_options,
) -> DemonstrationDataset:
    """Collect opt-in actual native train trajectories; every RGB failure aborts.

    This adds canonical environment rollouts only. Appearance copies are a separate
    replay transform, and neither native corrections nor held-out banks are used.
    """
    sizes = tuple(native_train_sizes)
    if (len(set(sizes)) != len(sizes)
            or any(type(size) is not int or size not in (64, 128) for size in sizes)):
        raise ValueError("native_train_sizes must be a unique selection of 64 and 128")
    if (type(native_episodes_per_task) is not int
            or not 0 <= native_episodes_per_task <= 64
            or bool(sizes) != bool(native_episodes_per_task)):
        raise ValueError("native sizes and episodes must be enabled together; episodes in [1,64]")
    if any(key in collection_options for key in ("image_size", "native_size", "episodes_per_task")):
        raise ValueError("native collection controls its RGB size and episode count")
    result = DemonstrationDataset(max_frame_bytes=max_frame_bytes)
    for size in sizes:
        collected = collect_scripted_demonstrations(
            native_episodes_per_task, image_size=size, native_size=size,
            max_frame_bytes=max_frame_bytes, **collection_options)
        result.extend(collected)
    return result


def collect_scripted_corrections(
    agent, episodes_per_task: int = 8, *, tasks: Sequence[str] = TASKS, seed: int = 0,
    difficulty: float = 1.0, image_size: int = 96, max_steps: int = 80,
    expert_probability: float = 0.5,
    episode_instructions: Mapping[str, Sequence[str]] | None = None,
    max_frame_bytes: int = 256 * 1024**2, split: str = "train",
    env_factory: Callable = AtomicEnv, teacher_factory: Callable = scripted_teacher,
    feature_cache: FrozenFrameCache | None = None,
    on_episode: Callable[[dict], None] | None = None,
    level_bank_seed: int = 0,
    learner_prefix_steps: int = 0,
    presentation_config=None,
) -> DemonstrationDataset:
    """Label bounded train rollouts of a frozen learner/expert mixture from RGB.

    Every teacher query sees the actual trajectory, including learner mistakes. Targets
    and executed inputs are stored separately; failures and truncations remain valid
    corrections when the final visible HUD verifies their provenance. The independent
    NumPy generator controls mixing; the learner uses deterministic policy decisions.
    An optional learner-only prefix visits learner states before the mixture begins;
    the teacher still labels every frame of that actual trajectory.
    """
    tasks, instructions = _validate_collection(episodes_per_task, tasks, seed, difficulty,
        image_size, max_steps, episode_instructions, split)
    if presentation_config is not None:
        from carryrl.envs.presentation import THEME_SPLITS, PresentationConfig

        if isinstance(presentation_config, dict):
            presentation_config = PresentationConfig(**presentation_config)
        if (not isinstance(presentation_config, PresentationConfig)
                or presentation_config.mode != "varied"
                or presentation_config.native_sizes != (96,)
                or image_size != 96 or presentation_config.style_split != "train"
                or not set(presentation_config.themes).issubset(THEME_SPLITS["train"])
                or episodes_per_task > 8):
            raise ValueError("live styled corrections require train-only varied presentation, "
                             "native 96 RGB and at most 8 episodes per task")
    if type(level_bank_seed) is not int or not 0 <= level_bank_seed < 2**32:
        raise ValueError("level_bank_seed must be a uint32 integer")
    if (isinstance(expert_probability, bool) or not isinstance(expert_probability, (int, float))
            or not math.isfinite(expert_probability) or not 0 <= expert_probability <= 1):
        raise ValueError("expert_probability must be finite and in [0,1]")
    if (type(learner_prefix_steps) is not int
            or not 0 <= learner_prefix_steps <= max_steps):
        raise ValueError("learner_prefix_steps must be an integer in [0,max_steps]")
    if feature_cache is not None:
        if feature_cache.agent is not agent:
            raise ValueError("correction cache belongs to a different agent")
        feature_cache.assert_unchanged()
    allowed_by_task = {task: allowed_runtime_train_indices(task, level_bank_seed, difficulty)
                       for task in tasks}
    if presentation_config is not None and any(
            len(indices) < episodes_per_task for indices in allowed_by_task.values()):
        raise ValueError("live styled training bank lacks enough levels "
                         "without held-out semantic matches")
    dataset = DemonstrationDataset(max_frame_bytes=max_frame_bytes)
    rng = np.random.default_rng(seed)
    training = agent.training
    agent.eval()
    try:
        with torch.no_grad():
            for task in tasks:
                allowed_indices = allowed_by_task[task]
                env = env_factory(task, split="train", image_size=image_size,
                                  max_steps=max_steps, difficulty=difficulty,
                                  **({"visual_config": presentation_config}
                                     if presentation_config is not None else {}),
                                  **({"level_bank_seed": level_bank_seed}
                                     if level_bank_seed else {}))
                try:
                    for index in range(episodes_per_task):
                        episode_seed = seed + 10000 * TASKS.index(task) + index
                        level_index = allowed_indices[(seed + index) % len(allowed_indices)]
                        frame, info = env.reset(seed=episode_seed,
                                                options={"level_index": level_index})
                        level_id = info["level_id"]
                        presentation = None
                        if presentation_config is not None:
                            from carryrl.envs.presentation import PresentationProfile

                            presentation = info.get("presentation")
                            if (not isinstance(presentation, dict)
                                    or PresentationProfile(**presentation).native_size != 96
                                    or presentation["theme"] not in THEME_SPLITS["train"]):
                                raise ValueError("live environment did not apply "
                                                 "train presentation")
                            if not callable(getattr(env, "canonical_render", None)):
                                raise TypeError("live styled teacher requires canonical_render")
                        variants = instructions.get(task)
                        instruction = variants[index % len(variants)] if variants else None
                        agent.reset_episode(instruction=instruction)
                        teacher = teacher_factory(task)
                        if not callable(getattr(teacher, "observe_action", None)):
                            raise TypeError("correction teacher must observe actual inputs")
                        frame_indices, labels, executed = [], [], []
                        terminated = truncated = False
                        expert_steps = 0
                        for step in range(max_steps):
                            frame_index = dataset.intern_frame(frame)
                            rgb = dataset.frames[frame_index]
                            if presentation is None:
                                teacher_rgb = rgb
                            else:
                                canonical = env.canonical_render()
                                if canonical.dtype != np.uint8 or canonical.shape != (96, 96, 3):
                                    raise ValueError("live teacher canonical RGB must be native 96")
                                teacher_rgb = np.frombuffer(canonical.tobytes(), dtype=np.uint8
                                                            ).reshape(canonical.shape)
                            label = teacher.act(teacher_rgb)
                            if not isinstance(label, PhysicalAction):
                                raise TypeError("scripted teacher must return a PhysicalAction")
                            # The learner advances its memory once at every actual frame,
                            # even when mixing chooses the expert's physical input.
                            learner = (agent.act(rgb.copy(), deterministic=True)
                                if feature_cache is None
                                or getattr(agent, "screen_text_reader", None) is not None
                                else agent.act_from_patches(
                                    feature_cache.encode(rgb.copy()), deterministic=True))
                            use_expert = (step >= learner_prefix_steps
                                          and rng.random() < expert_probability)
                            action = label if use_expert else learner.action
                            if not isinstance(action, PhysicalAction):
                                raise TypeError("learner must return a complete PhysicalAction")
                            frame_indices.append(frame_index)
                            labels.append(label)
                            executed.append(action)
                            frame, _, terminated, truncated, final_info = env.step(action)
                            agent.observe_action(action)
                            teacher.observe_action(action)
                            expert_steps += int(use_expert)
                            if terminated or truncated:
                                break
                        outcome = verify_visible_outcome(frame)
                        quality = _collected_quality(final_info, task=task, actions=executed,
                                                     outcome=outcome, pure_teacher=False)
                        dataset.add_episode(DemonstrationEpisode(
                            tuple(frame_indices), tuple(labels), dataset.intern_frame(frame),
                            instruction, task, episode_seed, level_id, level_index,
                            float(difficulty), dict(outcome), source=CORRECTION_SOURCE,
                            executed_actions=tuple(executed), terminated=bool(terminated),
                            truncated=bool(truncated or not terminated),
                            level_bank_seed=level_bank_seed,
                            expert_probability=float(expert_probability),
                            learner_prefix_steps=learner_prefix_steps,
                            presentation=presentation,
                            collection_type="live_styled" if presentation is not None else None,
                            scoring_version=SCORING_VERSION,
                            quality=quality, quality_source=QUALITY_SOURCE))
                        if on_episode is not None:
                            on_episode({"task": task, "source": CORRECTION_SOURCE,
                                "episodes_completed": len(dataset.episodes),
                                "episodes_total": len(tasks) * episodes_per_task,
                                "steps": len(labels), "expert_steps": expert_steps,
                                "expert_probability": expert_probability,
                                **({"learner_prefix_steps": learner_prefix_steps}
                                   if learner_prefix_steps else {}),
                                "unique_frames": len(dataset.frames)})
                finally:
                    env.close()
    finally:
        agent.train(training)
    if feature_cache is not None:
        feature_cache.assert_unchanged()
    return dataset


def demonstration_episode_digest(dataset: DemonstrationDataset,
                                 episode: DemonstrationEpisode) -> str:
    """Content identity independent of RGB intern indices, including own input history."""
    def frame_identity(index):
        frame = dataset.frames[index]
        return {"shape": list(frame.shape), "sha256": hashlib.sha256(frame.tobytes()).hexdigest()}

    return _episode_content_digest(episode.to_dict(), frame_identity)


def _episode_content_digest(record, frame_identity):
    record = dict(record)
    record["frame_indices"] = [frame_identity(index) for index in record["frame_indices"]]
    record["final_frame_index"] = frame_identity(record["final_frame_index"])
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _bundle_file_identity(path, limit):
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise ValueError(f"correction bundle file exceeds its byte budget: {path.name}")
            digest.update(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _bundle_json(path, limit):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate correction bundle JSON field")
            result[key] = value
        return result

    def invalid_constant(_):
        raise ValueError("correction bundle JSON requires finite numbers")

    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("correction bundle JSON exceeds its byte budget")
    value = json.loads(raw, object_pairs_hook=object_pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("correction bundle JSON must contain an object")
    return value


@dataclass
class SavedDemonstrationReuse:
    """Validated complete saved corpus; its producer is independent of optimizer lineage."""

    directory: Path
    dataset: DemonstrationDataset
    files: dict
    summary: dict

    @classmethod
    def prepare(cls, directory, *, config, max_frame_bytes):
        directory = Path(directory).resolve(strict=True)
        limits = {"demonstrations.json": 64 * 1024**2,
                  "demonstration_frames.npz": 4 * 1024**3,
                  "metadata.json": 32 * 1024**2, "config.json": 4 * 1024**2,
                  "feature_manifest.json": 64 * 1024**2}
        if not directory.is_dir():
            raise ValueError("demonstration reuse requires a saved run directory")
        for name in limits:
            path = directory / name
            if path.is_symlink() or not path.is_file() or path.resolve().parent != directory:
                raise ValueError("demonstration reuse files must belong directly to the saved run")
        identities = {name: _bundle_file_identity(directory / name, limit)
                      for name, limit in limits.items()}
        metadata = _bundle_json(directory / "metadata.json", limits["metadata.json"])
        source_config = _bundle_json(directory / "config.json", limits["config.json"])
        corpus = _bundle_json(directory / "demonstrations.json", limits["demonstrations.json"])
        features = _bundle_json(directory / "feature_manifest.json",
                                limits["feature_manifest.json"])
        if (metadata.get("kind") != "imitation_training"
                or metadata.get("suite_version") != SUITE_VERSION
                or metadata.get("config") != source_config
                or corpus.get("suite_version") != SUITE_VERSION
                or corpus.get("split") != "train" or corpus.get("method") != METHOD
                or features.get("demonstrations_sha256") != identities[
                    "demonstrations.json"]["sha256"]):
            raise ValueError("saved demonstration provenance or hash mismatch")
        current = json.loads(json.dumps(config.to_dict()))
        for name in ("agent", "tasks", "episode_instructions", "max_episode_steps",
                     "env_image_size", "level_bank_seed"):
            if source_config.get(name) != current.get(name):
                raise ValueError(f"saved demonstration configuration differs: {name}")
        frame_count = corpus.get("unique_frames")
        records = corpus.get("episodes")
        if (type(frame_count) is not int or not 0 < frame_count <= 100000
                or not isinstance(records, list) or not records
                or type(corpus.get("frame_bytes")) is not int
                or type(corpus.get("max_frame_bytes")) is not int
                or corpus["max_frame_bytes"] < corpus["frame_bytes"]
                or corpus["frame_bytes"] > max_frame_bytes):
            raise ValueError("saved demonstration counts or RGB byte budget are invalid")
        _, logical_bytes = _archive_rgb_identities(directory / "demonstration_frames.npz",
            frame_count, max_logical_bytes=max_frame_bytes)
        if logical_bytes != corpus["frame_bytes"]:
            raise ValueError("saved demonstration RGB byte count mismatch")
        dataset = DemonstrationDataset(max_frame_bytes=max_frame_bytes)
        with np.load(directory / "demonstration_frames.npz", allow_pickle=False) as archive:
            if archive.files == ["frames"]:
                stacked = archive["frames"]
                rows = (stacked[index] for index in range(frame_count))
            else:
                rows = (archive[f"frame_{index:06}"] for index in range(frame_count))
            for index, frame in enumerate(rows):
                if dataset.intern_frame(frame) != index:
                    raise ValueError("saved demonstration frame indices contain duplicate RGB")
        names = {field.name for field in fields(DemonstrationEpisode)}
        # A fresh combined corpus may retain extra-bank rows from an earlier producer.
        # Their original frozen overlap proof remains in the saved reuse lineage.
        producer_metadata = metadata
        while "correction_replacement" not in producer_metadata:
            reuse_lineage = producer_metadata.get("demonstration_reuse", {})
            if not isinstance(reuse_lineage, dict):
                raise ValueError("saved demonstration producer reuse lineage is malformed")
            inherited = reuse_lineage.get("producer_lineage", {})
            if not isinstance(inherited, dict):
                raise ValueError("saved demonstration producer lineage is malformed")
            if not inherited:
                break
            producer_metadata = inherited
        replacement = producer_metadata.get("correction_replacement", {})
        if not isinstance(replacement, dict):
            raise ValueError("saved correction producer lineage is malformed")
        collection = replacement.get("actual_collection", {})
        if not isinstance(collection, dict):
            raise ValueError("saved correction collection lineage is malformed")
        correction_bank = collection.get("level_bank_seed")
        correction_tasks = replacement.get("selected_tasks", [])
        if (correction_bank is not None and (
                type(correction_bank) is not int or not 0 <= correction_bank < 2**32
                or not isinstance(correction_tasks, list)
                or any(task not in config.tasks for task in correction_tasks))):
            raise ValueError("saved correction training bank provenance is malformed")
        heldout_semantics = {}
        try:
            for row in records:
                if (not isinstance(row, dict) or row.get("task") not in config.tasks
                        or type(row.get("seed")) is not int
                        or type(row.get("level_index")) is not int
                        or type(row.get("difficulty")) not in (int, float)
                        or not 0 <= row["difficulty"] <= 1
                        or not isinstance(row.get("outcome"), dict)
                        or not isinstance(row.get("frame_indices"), list)
                        or not 1 <= len(row["frame_indices"]) <= config.max_episode_steps):
                    raise ValueError("saved demonstration episode metadata is malformed")
                values = {key: value for key, value in row.items() if key in names}
                values["frame_indices"] = tuple(row["frame_indices"])
                values["actions"] = tuple(PhysicalAction.from_dict(value)
                                          for value in row["actions"])
                if "executed_actions" in row:
                    values["executed_actions"] = tuple(PhysicalAction.from_dict(value)
                                                       for value in row["executed_actions"])
                episode = DemonstrationEpisode(**values)
                if (episode.level_bank_seed != config.level_bank_seed
                        and not (episode.source == CORRECTION_SOURCE
                                 and episode.level_bank_seed == correction_bank
                                 and episode.task in correction_tasks)):
                    raise ValueError("saved demonstration has an unexplained training bank")
                if episode.level_bank_seed != config.level_bank_seed:
                    if (replacement.get("overlap_guard_passed") is not True
                            or replacement.get("producer_frozen") is not True):
                        raise ValueError("saved correction training bank lacks "
                                         "frozen overlap proof")
                    if episode.task not in heldout_semantics:
                        heldout_semantics[episode.task] = {
                            atomic_semantic_level_id(episode.task, level)
                            for bank_seed in {0, config.level_bank_seed}
                            for split in ("validation", "test")
                            for level in atomic_level_banks(episode.task, bank_seed)[split]}
                    level = atomic_level_banks(episode.task, episode.level_bank_seed)[
                        "train"][episode.level_index]
                    if (atomic_semantic_level_id(episode.task, level)
                            in heldout_semantics[episode.task]):
                        raise ValueError("saved correction training bank overlaps held-out levels")
                if episode.to_dict() != row:
                    raise ValueError("saved demonstration episode would change on import")
                variants = config.episode_instructions.get(episode.task) or ()
                if ((variants and episode.instruction not in variants)
                        or (not variants and episode.instruction is not None)):
                    raise ValueError("saved demonstration instruction differs from configuration")
                dataset.add_episode(episode)
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("saved demonstration episode is malformed") from error
        dataset.memory_cue_counterpart_reports = deepcopy(
            corpus.get("memory_cue_counterparts", {}).get("reports", []))
        rebuilt = dataset.provenance()
        rebuilt["max_frame_bytes"] = corpus.get("max_frame_bytes")
        if rebuilt != corpus:
            raise ValueError("saved demonstration summary differs from its actual complete corpus")
        # Authenticate all action-input RGB against the original frozen encoder identity.
        from carryrl.training.feature_store import feature_cache_identity

        try:
            identity = features.get("feature_reuse", {}).get("identity")
            if identity is None:
                pack = Path(features["feature_cache_dir"])
                if not pack.is_absolute() or pack.is_symlink():
                    raise ValueError("saved demonstration feature pack path is invalid")
                identity = _bundle_json(pack / "manifest.json", 32 * 1024**2)["identity"]
            used = sorted({index for episode in dataset.episodes
                           for index in episode.frame_indices})
            options = {key: identity[key] for key in
                       ("vision_sha256", "preprocessing", "feature_shape", "dtype")}
            if (features.get("vision_sha256") != identity["vision_sha256"]
                    or features.get("dtype") != identity["dtype"]
                    or features.get("frames") != len(used)
                    or feature_cache_identity(dataset.frames, used, **options) != identity):
                raise ValueError("saved demonstration RGB identity differs "
                                 "from frozen feature inputs")
        except (OSError, KeyError, TypeError, AttributeError, IndexError) as error:
            raise ValueError("saved demonstration frozen feature identity is malformed") from error
        producer = {key: deepcopy(metadata[key]) for key in (
            "source_checkpoint", "source_checkpoint_sha256", "initial_agent_sha256",
            "reward_sha256", "correction_replacement", "demonstration_reuse",
            "navigation_geometry", "runtime_diversity") if key in metadata}
        for name in ("source_checkpoint_sha256", "initial_agent_sha256", "reward_sha256"):
            digest = producer.get(name)
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError("saved demonstration producer identity is missing or malformed")
        result = cls(directory, dataset, identities, {
            "source_run": str(directory), "files": identities, "producer_lineage": producer,
            "source_collection_settings": deepcopy(metadata.get("settings", {})),
            "source_difficulty": {
                key: source_config.get(key)
                for key in ("curriculum_difficulty", "evaluation_difficulty")},
            "episodes": len(dataset.episodes), "decisions": sum(map(len, dataset.episodes)),
            "unique_frames": frame_count, "frame_bytes": logical_bytes,
            "exact_complete_corpus": True, "collection_skipped": True,
            "storage": "verified source corpus references; no archive or feature copy",
        })
        result.assert_unchanged()
        return result

    def assert_unchanged(self):
        for name, identity in self.files.items():
            path = self.directory / name
            if (path.is_symlink() or not path.is_file()
                    or _bundle_file_identity(path, identity["bytes"]) != identity):
                raise ValueError("saved demonstration source changed after validation")

    def link_corpus(self, output):
        """Publish read-only replay references; never reopen source files for writing."""
        self.assert_unchanged()
        for name in ("demonstrations.json", "demonstration_frames.npz"):
            destination = Path(output) / name
            destination.symlink_to(self.directory / name)
            if _bundle_file_identity(destination, self.files[name]["bytes"]) != self.files[name]:
                raise ValueError("saved demonstration source changed while linking")
        self.assert_unchanged()


def _validate_replacement_trace(dataset, episode, max_steps):
    """Authenticate imported train RGB, teacher labels, and executed physical history."""
    env = AtomicEnv(episode.task, split="train", image_size=96, difficulty=1.0,
                    level_bank_seed=episode.level_bank_seed, max_steps=max_steps)
    teacher = scripted_teacher(episode.task)
    try:
        frame, _ = env.reset(seed=episode.seed, options={"level_index": episode.level_index})
        terminated = truncated = False
        executed = episode.executed_actions or episode.actions
        for step, (index, label, action) in enumerate(zip(
                episode.frame_indices, episode.actions, executed, strict=True)):
            if not np.array_equal(dataset.frames[index], frame):
                raise ValueError("replacement RGB does not match its recorded executed history")
            rgb = np.frombuffer(frame.tobytes(), dtype=np.uint8).reshape(frame.shape)
            if teacher.act(rgb) != label:
                raise ValueError("replacement expert labels do not match the RGB teacher")
            frame, _, terminated, truncated, final_info = env.step(action)
            if episode.source == CORRECTION_SOURCE:
                teacher.observe_action(action)
            if (terminated or truncated) and step != len(episode) - 1:
                raise ValueError("replacement must retain a whole ended trajectory")
        if not np.array_equal(dataset.frames[episode.final_frame_index], frame):
            raise ValueError("replacement final RGB does not match its executed history")
        if episode.source == CORRECTION_SOURCE:
            if terminated != episode.terminated or truncated != episode.truncated:
                raise ValueError("replacement termination flags do not match its actual trajectory")
        elif not terminated or truncated:
            raise ValueError("replacement supporting teacher must complete successfully")
        if episode.quality is not None and episode.quality != final_info.get("quality"):
            raise ValueError("replacement quality does not match its actual executed trajectory")
    finally:
        env.close()


def load_correction_replacement_bundle(
    directory: Path, *, expected_sha256: str, correction_tasks: Sequence[str],
    episodes_per_task: int, seed: int, difficulty: float, max_steps: int,
    level_bank_seed: int, episode_instructions: Mapping[str, Sequence[str]] | None,
    source_identity: Mapping, initial_agent_sha256: str, reward_sha256: str,
    max_frame_bytes: int = 256 * 1024**2, max_json_bytes: int = 16 * 1024**2,
) -> tuple[dict[str, DemonstrationDataset], dict]:
    """Validate a pinned preflight bundle; return corrections only, grouped by task.

    Supporting teachers authenticate the cohort and are never admitted to training.
    The caller supplies the previously audited combined digest, independently of the
    bundle itself. All validation finishes before any current-run collection/encoding.
    """
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)):
        raise ValueError("correction replacement requires its trusted SHA256")
    if (difficulty != 1.0 or type(episodes_per_task) is not int
            or not 1 <= episodes_per_task <= 64 or type(max_steps) is not int or max_steps < 1
            or type(max_frame_bytes) is not int or max_frame_bytes < 1
            or type(max_json_bytes) is not int or max_json_bytes < 1):
        raise ValueError("replacement requires full difficulty and bounded positive counts/budgets")
    directory = Path(directory).resolve(strict=True)
    limits = {"coverage.json": max_json_bytes, "trajectories.json": max_json_bytes,
              "frames.npy": max_frame_bytes + 65536}
    files = {}
    for name, limit in limits.items():
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("correction replacement requires three regular bundle files")
        files[name] = _bundle_file_identity(path, limit)
    if sum(row["bytes"] for row in files.values()) > 256 * 1024**2:
        raise ValueError("correction replacement bundle exceeds 256 MiB")
    combined = hashlib.sha256(json.dumps(files, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
    if combined != expected_sha256:
        raise ValueError("correction replacement bundle SHA256 mismatch")
    coverage = _bundle_json(directory / "coverage.json", max_json_bytes)
    trajectories = _bundle_json(directory / "trajectories.json", max_json_bytes)
    guard, declared = coverage.get("overlap_guard", {}), coverage.get("preregistration", {})
    if not isinstance(guard, dict) or not isinstance(declared, dict):
        raise ValueError("replacement requires object collection and overlap declarations")
    expected_hashes = {"actor": initial_agent_sha256, "reward": reward_sha256}
    if (coverage.get("status") != "complete" or guard.get("passed") is not True
            or guard.get("collisions") != []
            or coverage.get("loaded_tensors_match_preregistered_identity") is not True
            or coverage.get("policy_and_reward_frozen") is not True
            or coverage.get("frozen_hashes_before") != expected_hashes
            or coverage.get("frozen_hashes_after") != expected_hashes):
        raise ValueError("replacement requires completed overlap and frozen producer hash proofs")
    producer = declared.get("expected_checkpoint_identity", {})
    if (not isinstance(producer, dict)
            or any(producer.get(key) != source_identity.get(key)
            for key in ("sha256", "rollout", "weights_revision"))
            or producer.get("agent_sha256") != initial_agent_sha256
            or producer.get("reward_sha256") != reward_sha256):
        raise ValueError("replacement producer must match the current warm-start checkpoint")
    fresh_seed = declared.get("fresh_train_bank_seed")
    if (type(fresh_seed) is not int or not 0 < fresh_seed < 2**32
            or fresh_seed == level_bank_seed or level_bank_seed not in guard.get(
                "original_bank_seeds", [])
            or declared.get("correction_collection_seed") != seed
            or declared.get("episodes_per_task") != episodes_per_task
            or declared.get("max_steps_per_episode") != max_steps
            or declared.get("difficulty") != 1.0 or declared.get("native_rgb_size") != 96
            or declared.get("expert_probability") != 0.0
            or declared.get("learner_prefix_steps") != 0
            or declared.get("training_updates") != 0 or declared.get("optimizer_steps") != 0
            or declared.get("heldout_policy_rollouts") != 0
            or declared.get("test_outcomes_read") is not False):
        raise ValueError("replacement collection settings do not match the admitted fresh cohort")
    records = trajectories.get("episodes")
    if (not isinstance(records, list) or not records
            or len(records) > 2 * len(TASKS) * episodes_per_task
            or any(not isinstance(row, dict) or row.get("source") not in {SOURCE, CORRECTION_SOURCE}
                   for row in records)):
        raise ValueError("replacement requires bounded correction and supporting teacher rows")
    corrections = [row for row in records if row["source"] == CORRECTION_SOURCE]
    if any(row.get("task") not in TASKS for row in records):
        raise ValueError("replacement requires supported task names")
    tasks = sorted({row.get("task") for row in corrections})
    if (not tasks or not set(tasks).issubset(correction_tasks)
            or sorted(declared.get("tasks", [])) != tasks
            or len(records) != 2 * len(tasks) * episodes_per_task
            or len(coverage.get("episodes", [])) != len(corrections)
            or len(guard.get("selected", [])) != len(corrections)):
        raise ValueError("replacement declarations do not match actual correction tasks/counts")
    if (trajectories.get("frame_array") != "frames.npy"
            or trajectories.get("frame_shape") != [96, 96, 3]):
        raise ValueError("replacement requires canonical physical 96x96 RGB")
    frames = np.load(directory / "frames.npy", allow_pickle=False, mmap_mode="r")
    if (not isinstance(frames, np.ndarray) or frames.dtype != np.uint8 or frames.ndim != 4
            or frames.shape[1:] != (96, 96, 3) or not len(frames)
            or len(frames) > len(records) * (max_steps + 1)
            or frames.nbytes > max_frame_bytes
            or trajectories.get("frame_count") != len(frames)
            or trajectories.get("raw_frame_bytes") != frames.nbytes):
        raise ValueError("replacement RGB dtype, shape, count, or byte budget is invalid")
    whole = DemonstrationDataset(max_frame_bytes=max_frame_bytes)
    mapping, frame_hashes, parsed = {}, {}, []
    fields = {"source", "split", "task", "seed", "level_id", "level_index", "difficulty",
              "instruction", "level_bank_seed", "frame_indices", "final_frame_index", "actions",
              "outcome", "executed_actions", "terminated", "truncated", "expert_probability",
              "learner_selection", "mixing_rng", "learner_prefix_steps", "scoring_version",
              "quality", "quality_source"}
    required = {"source", "split", "task", "seed", "level_id", "level_index", "difficulty",
                "instruction", "level_bank_seed", "frame_indices", "final_frame_index",
                "actions", "outcome"}
    instructions = episode_instructions or {}
    for row in records:
        if (set(row) - fields or not required.issubset(row) or row.get("split") != "train"
                or row.get("task") not in tasks or not isinstance(row.get("outcome"), dict)):
            raise ValueError("replacement requires unstyled train correction/teacher rows")
        task = row["task"]
        if type(row.get("seed")) is not int:
            raise ValueError("replacement episode seed must be an integer")
        index = row.get("seed", -1) - seed - 10000 * TASKS.index(task)
        if (not 0 <= index < episodes_per_task or type(row.get("level_bank_seed")) is not int
                or row.get("level_bank_seed") != fresh_seed
                or type(row.get("difficulty")) not in (int, float) or row.get("difficulty") != 1.0
                or type(row.get("level_index")) is not int
                or row.get("level_index") != (seed + index) % 64):
            raise ValueError("replacement seeds/bank/index/difficulty differ from the cohort")
        variants = instructions.get(task)
        expected_instruction = variants[index % len(variants)] if variants else None
        if row.get("instruction") != expected_instruction:
            raise ValueError("replacement instruction differs from the current configuration")
        if row["source"] == CORRECTION_SOURCE and (
                row.get("expert_probability") != 0.0 or row.get("learner_prefix_steps", 0) != 0
                or row.get("learner_selection") != "deterministic"
                or row.get("mixing_rng") != "isolated_numpy_generator"):
            raise ValueError("replacement requires deterministic learner-only executed histories")
        indices = row.get("frame_indices")
        if (not isinstance(indices, list) or not 1 <= len(indices) <= max_steps
                or any(type(value) is not int or not 0 <= value < len(frames)
                       for value in (*indices, row.get("final_frame_index")))):
            raise ValueError("replacement episode RGB indices/length are invalid")
        for value in (*indices, row["final_frame_index"]):
            if value not in mapping:
                mapping[value] = whole.intern_frame(frames[value])
                frame_hashes[value] = hashlib.sha256(frames[value].tobytes()).hexdigest()
        actions = row.get("actions")
        executed = row.get("executed_actions")
        if row["source"] == SOURCE and executed is not None:
            raise ValueError("supporting teacher rows must preserve their original teacher history")
        if not isinstance(actions, list) or len(actions) != len(indices):
            raise ValueError("replacement actions must align with all actual RGB rows")
        if row["source"] == CORRECTION_SOURCE and (
                not isinstance(executed, list) or len(executed) != len(indices)):
            raise ValueError("replacement corrections require complete executed input histories")
        episode = DemonstrationEpisode(
            tuple(mapping[value] for value in indices),
            tuple(PhysicalAction.from_dict(value) for value in actions),
            mapping[row["final_frame_index"]], row["instruction"], task, row["seed"],
            row["level_id"], row["level_index"], row["difficulty"], row["outcome"],
            source=row["source"], level_bank_seed=fresh_seed,
            executed_actions=(tuple(PhysicalAction.from_dict(value) for value in executed)
                              if executed is not None else None),
            terminated=row.get("terminated"), truncated=row.get("truncated"),
            expert_probability=row.get("expert_probability"),
            scoring_version=row.get("scoring_version"),
            quality=deepcopy(row.get("quality")), quality_source=row.get("quality_source"))
        whole.add_episode(episode)
        parsed.append(episode)
    grouped = {}
    for task in tasks:
        selected = [episode for episode in parsed if episode.task == task]
        for source in (SOURCE, CORRECTION_SOURCE):
            rows = [episode for episode in selected if episode.source == source]
            if (len(rows) != episodes_per_task or len({episode.seed for episode in rows})
                    != episodes_per_task):
                raise ValueError("replacement requires one teacher/correction per cohort seed")
        for episode in selected:
            _validate_replacement_trace(whole, episode, max_steps)
        imported = DemonstrationDataset(max_frame_bytes=max_frame_bytes)
        for episode in sorted((row for row in selected if row.source == CORRECTION_SOURCE),
                              key=lambda row: row.seed):
            remapped = {value: imported.intern_frame(whole.frames[value])
                        for value in (*episode.frame_indices, episode.final_frame_index)}
            imported.add_episode(replace(episode,
                frame_indices=tuple(remapped[value] for value in episode.frame_indices),
                final_frame_index=remapped[episode.final_frame_index]))
        grouped[task] = imported
    if any(_bundle_file_identity(directory / name, limits[name]) != identity
           for name, identity in files.items()):
        raise ValueError("correction replacement files changed during validation")
    summary = {
        "bundle_directory": str(directory), "bundle_sha256": combined, "files": files,
        "selected_tasks": tasks, "episodes_per_task": episodes_per_task,
        "replacement_episodes": len(corrections), "supporting_teacher_episodes_excluded": len(
            records) - len(corrections), "source_checkpoint": declared.get("checkpoint"),
        "source_checkpoint_identity": producer, "producer_hashes": expected_hashes,
        "producer_frozen": True, "overlap_guard_passed": True,
        "actual_collection": {"expert_probability": 0.0, "learner_prefix_steps": 0,
                              "difficulty": 1.0, "level_bank_seed": fresh_seed, "seed": seed},
        "rgb_teacher_and_executed_history_replay_validated": True,
        "referenced_frame_sha256": {
            str(index): value for index, value in sorted(frame_hashes.items())},
        "replacement_episode_content_sha256": {
            task: [demonstration_episode_digest(value, episode) for episode in value.episodes]
            for task, value in grouped.items()},
    }
    return grouped, summary


def _archive_rgb_identities(path, frame_count, *, max_logical_bytes=8 * 1024**3):
    """Hash lossless NPZ RGB rows in bounded chunks, never materializing the archive."""
    identities, logical_bytes = {}, 0

    def header(stream):
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            result = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            result = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("control RGB requires supported nonpickle NPY headers")
        shape, fortran, dtype = result
        if (fortran or dtype != np.uint8
                or any(type(value) is not int or value < 1 for value in shape)):
            raise ValueError("control RGB must be contiguous uint8 arrays")
        return shape

    def consume(stream, shape, index):
        nonlocal logical_bytes
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError("control RGB must have HWC shape")
        count = math.prod(shape)
        logical_bytes += count
        if count > 64 * 1024**2 or logical_bytes > max_logical_bytes:
            raise ValueError("control RGB logical byte budget exceeded")
        digest = hashlib.sha256()
        remaining = count
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("truncated control RGB archive member")
            digest.update(chunk)
            remaining -= len(chunk)
        identities[index] = {"shape": list(shape), "sha256": digest.hexdigest()}

    with ZipFile(path) as archive:
        names = archive.namelist()
        if names == ["frames.npy"]:
            with archive.open("frames.npy") as stream:
                shape = header(stream)
                if len(shape) != 4 or shape[0] != frame_count:
                    raise ValueError("control stacked RGB count mismatch")
                for index in range(frame_count):
                    consume(stream, shape[1:], index)
                if stream.read(1):
                    raise ValueError("control RGB has trailing array data")
        else:
            expected = [f"frame_{index:06}.npy" for index in range(frame_count)]
            if len(names) != frame_count or set(names) != set(expected):
                raise ValueError("control mixed RGB archive has missing/extra/duplicate members")
            for index, name in enumerate(expected):
                with archive.open(name) as stream:
                    consume(stream, header(stream), index)
                    if stream.read(1):
                        raise ValueError("control RGB has trailing array data")
    return identities, logical_bytes


def audit_correction_replacement(
    dataset: DemonstrationDataset, control_run: Path,
    replacements: Mapping[str, DemonstrationDataset], replacement_summary: Mapping,
) -> dict:
    """Fail before encoding if any uncontrolled slot or appearance profile changed."""
    control_run = Path(control_run).resolve(strict=True)
    paths = {"demonstrations.json": 64 * 1024**2,
             "demonstration_frames.npz": 4 * 1024**3, "metadata.json": 16 * 1024**2}
    files = {name: _bundle_file_identity(control_run / name, limit)
             for name, limit in paths.items()}
    control = _bundle_json(control_run / "demonstrations.json", paths["demonstrations.json"])
    metadata = _bundle_json(control_run / "metadata.json", paths["metadata.json"])
    original = control.get("episodes")
    frame_count = control.get("unique_frames")
    if (not isinstance(original, list) or len(original) != len(dataset.episodes)
            or type(frame_count) is not int or not 0 < frame_count <= 100000):
        raise ValueError("replacement control must have exactly the same episode slot count")
    producer = replacement_summary["producer_hashes"]
    if (metadata.get("initial_agent_sha256") != producer["actor"]
            or metadata.get("reward_sha256") != producer["reward"]
            or metadata.get("source_checkpoint_sha256") != replacement_summary[
                "source_checkpoint_identity"]["sha256"]):
        raise ValueError("replacement control must use the same frozen producer checkpoint")
    control_frames, logical_bytes = _archive_rgb_identities(
        control_run / "demonstration_frames.npz", frame_count)
    current_frames, expected_frames = {}, {}

    def current_identity(index):
        if index not in current_frames:
            frame = dataset.frames[index]
            current_frames[index] = {"shape": list(frame.shape),
                                     "sha256": hashlib.sha256(frame.tobytes()).hexdigest()}
        return current_frames[index]

    expected_by_seed = {(task, episode.seed): (value, episode)
                        for task, value in replacements.items() for episode in value.episodes}
    changed, unchanged, unstyled, styled = [], [], 0, 0
    unstyled_seeds = set()
    old_digests, new_digests = [], []
    for index, (old, episode) in enumerate(zip(original, dataset.episodes, strict=True)):
        current = episode.to_dict()
        if not isinstance(old, dict):
            raise ValueError("control episode must contain its full provenance object")
        old_hash = _episode_content_digest(old, control_frames.__getitem__)
        new_hash = _episode_content_digest(current, current_identity)
        old_digests.append(old_hash)
        new_digests.append(new_hash)
        selected = old.get("source") == CORRECTION_SOURCE and old.get("task") in replacements
        if not selected:
            if old_hash != new_hash:
                raise ValueError(f"unexpected mutation of unchanged control episode slot {index}")
            unchanged.append(index)
            continue
        if any(old.get(key) != current.get(key) for key in (
                "source", "split", "task", "seed", "instruction", "difficulty", "presentation")):
            raise ValueError(f"replacement changed cohort/order/appearance at slot {index}")
        pair = expected_by_seed.get((episode.task, episode.seed))
        if pair is None:
            raise ValueError("replacement control slot has no admitted bundle trajectory")
        value, expected = pair
        expected_record = expected.to_dict()
        profile = episode.presentation
        if profile is None:
            cohort_key = episode.task, episode.seed
            if cohort_key in unstyled_seeds:
                raise ValueError("replacement control repeats an unstyled correction seed")
            unstyled_seeds.add(cohort_key)
        if profile is not None:
            expected_record["presentation"] = profile

        def expected_identity(frame_index):
            cache_key = episode.task, frame_index, json.dumps(profile, sort_keys=True)
            if cache_key not in expected_frames:
                frame = value.frames[frame_index]
                if profile is not None:
                    from carryrl.envs.presentation import PresentationProfile, present_frame

                    frame = present_frame(frame, PresentationProfile(**profile))
                expected_frames[cache_key] = {"shape": list(frame.shape),
                                             "sha256": hashlib.sha256(frame.tobytes()).hexdigest()}
            return expected_frames[cache_key]

        if new_hash != _episode_content_digest(expected_record, expected_identity):
            raise ValueError(f"replacement slot {index} differs from its admitted bundle/copy")
        if old_hash == new_hash:
            raise ValueError(f"declared replacement slot {index} did not change")
        changed.append(index)
        unstyled += int(profile is None)
        styled += int(profile is not None)
    expected_count = sum(len(value.episodes) for value in replacements.values())
    if unstyled != expected_count or unstyled_seeds != set(expected_by_seed):
        raise ValueError("replacement must occupy exactly its original unstyled correction slots")
    if any(_bundle_file_identity(control_run / name, paths[name]) != identity
           for name, identity in files.items()):
        raise ValueError("replacement control artifacts changed during the import audit")
    return {
        "passed": True, "control_run": str(control_run), "control_files": files,
        "control_rgb_logical_bytes": logical_bytes,
        "control_frame_hashing": "streamed NPZ/NPY RGB in chunks <=1MiB; no full archive array",
        "bundle_sha256": replacement_summary["bundle_sha256"], "producer_hashes": producer,
        "episodes": len(original), "changed_slots": len(changed), "unchanged_slots": len(unchanged),
        "replaced_unstyled_slots": unstyled, "replaced_appearance_slots": styled,
        "changed_positions": changed,
        "control_ordered_episode_content_sha256": hashlib.sha256(
            json.dumps(old_digests, separators=(",", ":")).encode()).hexdigest(),
        "current_ordered_episode_content_sha256": hashlib.sha256(
            json.dumps(new_digests, separators=(",", ":")).encode()).hexdigest(),
        "all_other_slots_identical": True, "replacement_copy_profiles_unchanged": True,
        "comparison": "full episode provenance, labels, own executed inputs, instruction, "
                      "presentation, ordered RGB content and final RGB; intern indices excluded",
    }


def imitation_action_weights(actions: Sequence[PhysicalAction], *,
                             action_balance: str = "uniform") -> np.ndarray:
    """Per-episode expert-label weights, independent of the executed input history."""
    if action_balance not in {"uniform", "active_idle"}:
        raise ValueError("action_balance must be uniform or active_idle")
    if not actions or any(not isinstance(action, PhysicalAction) for action in actions):
        raise ValueError("action weights require nonempty PhysicalAction expert labels")
    active = _active_action_mask(actions)
    count = int(active.sum())
    weights = np.full(len(actions), 1 / len(actions), dtype=np.float64)
    if action_balance == "active_idle" and 0 < count < len(actions):
        weights[active] = 1 / (2 * count)
        weights[~active] = 1 / (2 * (len(actions) - count))
    return weights


def _active_action_mask(actions):
    return np.asarray([bool(action.keys or action.mouse_buttons or action.dx
                            or action.dy or action.wheel) for action in actions], dtype=bool)


def _episode_nlls(log_prob, actions, action_balance):
    """Objective plus unweighted diagnostics; missing expert groups are None."""
    weights = imitation_action_weights(actions, action_balance=action_balance)
    mask = _active_action_mask(actions)
    if log_prob.shape != (len(actions),):
        raise FloatingPointError("invalid imitation joint action likelihood")
    if isinstance(log_prob, torch.Tensor):
        if not torch.isfinite(log_prob).all():
            raise FloatingPointError("invalid imitation joint action likelihood")
        weights = torch.as_tensor(weights, dtype=log_prob.dtype, device=log_prob.device)

        def group_nll(selected):
            return -log_prob[torch.as_tensor(selected, device=log_prob.device)].mean()
    else:
        import mlx.core as mx

        weights = mx.array(weights, dtype=log_prob.dtype)

        def group_nll(selected):
            return -log_prob[mx.array(np.flatnonzero(selected), dtype=mx.int32)].mean()

    nll = -log_prob.mean()
    # Keep the exact existing mean operation for the default and single-group cases.
    objective = (-(log_prob * weights).sum()
                 if action_balance == "active_idle" and mask.any() and (~mask).any() else nll)
    return (objective, nll, group_nll(mask) if mask.any() else None,
            group_nll(~mask) if (~mask).any() else None)


def imitation_nll(agent, patches: torch.Tensor, actions: Sequence[PhysicalAction],
                  *, instruction: str | None = None,
                  executed_actions: Sequence[PhysicalAction] | None = None,
                  action_balance: str = "uniform",
                  screen_text: Sequence[str | None] | None = None) -> torch.Tensor:
    """Joint action NLL; optionally balance active/idle expert labels within an episode."""
    imitation_action_weights(actions, action_balance=action_balance)
    output = agent.evaluate_episode(patches, list(actions), instruction=instruction,
                                   executed_actions=executed_actions,
                                   **({"screen_text": screen_text}
                                      if screen_text is not None else {}))
    log_prob = output["log_prob"]
    return _episode_nlls(log_prob, actions, action_balance)[0]


def _validate_canonical_replay_weight(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError("canonical replay weight must be positive and finite")


def _replay_domain(dataset, episode):
    if episode.presentation is not None:
        return "appearance_copy"
    if episode.native_size is not None:
        return f"native_{episode.native_size}"
    if (episode.source in {SOURCE, CORRECTION_SOURCE, MEMORY_COUNTERPART_SOURCE,
                           NAVIGATION_GEOMETRY_SOURCE, REACTION_DYNAMICS_SOURCE}
            and all(dataset.frames[index].shape == (96, 96, 3)
                    for index in (*episode.frame_indices, episode.final_frame_index))):
        return "canonical_unstyled_96"
    return "other_unstyled_rgb"


def _replay_summary(domains, canonical_replay_weight, *, normalization=None):
    counts = {domain: domains.count(domain) for domain in sorted(set(domains))}
    weights = {domain: canonical_replay_weight if domain == "canonical_unstyled_96" else 1.0
               for domain in counts}
    scale = max(weights.values(), default=1.0)
    scaled_mass = {domain: count * (weights[domain] / scale)
                   for domain, count in counts.items()}
    total = sum(scaled_mass.values())
    if normalization is None:
        normalization = (_replay_normalization([
            canonical_replay_weight if domain == "canonical_unstyled_96" else 1.0
            for domain in domains]) if domains else (1.0, 1.0))
    return {
        "canonical_replay_weight": canonical_replay_weight,
        "canonical_domain": "unstyled scripted teacher, correction, or memory counterpart; "
                            "no native_size; all replay and final RGB frames physically 96x96",
        "aggregation": "sum(episode_weight * episode_objective) / sum(episode_weight)",
        "objective_version": "global_episode_mass.v2",
        "minibatch_normalization": "global dataset mean episode weight; stable max-weight scaling",
        "dataset_weight_scale": normalization[0],
        "scaled_dataset_mean_weight": normalization[1],
        "sampling": "unchanged uniform episode shuffle",
        "domains": {domain: {
            "episodes": count, "episode_weight": weights[domain],
            # Preserve an exact interpretable mass even beyond float's finite range.
            "effective_episode_mass": (count * weights[domain]
                if math.isfinite(count * weights[domain]) else f"{count} * {weights[domain]}"),
            "effective_mass_fraction": scaled_mass[domain] / total,
        } for domain, count in counts.items()},
    }


def imitation_replay_summary(dataset: DemonstrationDataset, *,
                             canonical_replay_weight: float = 1.0) -> dict:
    """Loss-only episode masses derived from presentation/native provenance and RGB size."""
    _validate_canonical_replay_weight(canonical_replay_weight)
    return _replay_summary([_replay_domain(dataset, episode) for episode in dataset.episodes],
                           canonical_replay_weight)


def _replay_normalization(weights):
    scale = max(weights)
    return scale, math.fsum(weight / scale for weight in weights) / len(weights)


def _replay_objective(objectives, weights, *, normalization=None):
    """Scale minibatch masses by the dataset mean, preserving all-one arithmetic.

    Without an explicit dataset normalization this retains the helper's original
    weighted-mean behaviour. Training computes the normalizer once for all episodes;
    a one-episode minibatch therefore retains its relative replay weight.
    """
    if normalization is None:
        normalization = _replay_normalization(weights)
    scale, mean = normalization
    if scale == 1.0 and mean == 1.0 and all(weight == 1.0 for weight in weights):
        return objectives.mean()
    scaled = np.asarray(weights, dtype=np.float64) / scale
    if isinstance(objectives, torch.Tensor):
        factors = torch.as_tensor(scaled, dtype=objectives.dtype, device=objectives.device)
    else:
        import mlx.core as mx

        factors = mx.array(scaled, dtype=objectives.dtype)
    return (objectives * factors).mean() / mean


def _reported_replay_objective(objectives, weights):
    if not objectives:
        return None
    if all(weight == 1.0 for weight in weights):
        return float(np.mean(objectives))
    return float(np.average(objectives, weights=np.asarray(weights) / max(weights)))


def _validate_canonical_anchor_coefficient(value):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError("canonical anchor coefficient must be nonnegative and finite")


def teacher_prefix_conditional_kl(parent_trace, student_trace):
    """Mean physical-decision sum of KL(parent || student) at expert prefixes.

    This is not full joint-action KL: prefixes follow expert targets, while
    episode replay still conditions on the inputs actually executed previously.
    Both traces use the same sorted-key support and active EOS padding masks.
    """
    if len(parent_trace) != len(student_trace) or not parent_trace:
        raise ValueError("conditional KL requires matching nonempty decoder traces")
    terms = []
    for (parent, active), (student, _) in zip(parent_trace, student_trace, strict=True):
        if parent.shape != student.shape:
            raise ValueError("conditional KL requires matching categorical shapes")
        if isinstance(student, torch.Tensor):
            parent = parent.detach()
            minimum = torch.finfo(student.dtype).min
            delta = parent.clamp_min(minimum) - student.clamp_min(minimum)
            terms.append((parent.exp() * delta).sum(-1) * active)
        else:
            import mlx.core as mx

            parent = mx.stop_gradient(parent)
            minimum = mx.finfo(student.dtype).min
            delta = mx.maximum(parent, minimum) - mx.maximum(student, minimum)
            terms.append(mx.sum(mx.exp(parent) * delta, axis=-1) * active)
    return sum(terms).mean()


def _canonical_anchor_parent(agent, engine):
    """Snapshot the policy only, sharing frozen encoders and excluding optimizer state."""
    if engine is not None:
        import mlx.core as mx

        from carryrl.agent.mlx_policy import MLXPolicy

        weights = {name: mx.stop_gradient(mx.array(value))
                   for name, value in engine.policy.weights.items()}
        mx.eval(weights)
        parent = MLXPolicy(agent.config, weights,
                           motor_initialization=engine.policy.motor_initialization)
        parent.freeze()
        arrays = {name: np.asarray(value) for name, value in weights.items()}
    else:
        # Memo entries prevent deepcopy from allocating a second DINO/language encoder.
        memo = {id(agent.vision): agent.vision}
        if hasattr(agent, "language"):
            memo[id(agent.language)] = agent.language
        parent = deepcopy(agent, memo)
        arrays = {name: parameter.detach().cpu().numpy()
                  for name, parameter in parent.named_parameters() if parameter.requires_grad}
        parent.requires_grad_(False)
        parent.eval()
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(str((value.shape, str(value.dtype))).encode())
        digest.update(value.tobytes())
    return parent, digest.hexdigest()


def read_imitation_screen_text(reader, frame: np.ndarray, *, step: int):
    """Require OCR provenance to match the exact presented policy RGB and decision."""
    observed = reader.read(frame, step=step)
    if (observed.frame_sha256 != hashlib.sha256(frame.tobytes(order="C")).hexdigest()
            or observed.source_width != frame.shape[1]
            or observed.source_height != frame.shape[0] or observed.step != step):
        raise ValueError("screen text reader result does not match the replay RGB frame/step")
    return observed


class ImitationScreenText:
    """Disk-backed, frame-and-step keyed OCR observations; no resident token corpus."""

    def __init__(self, dataset, reader, path: Path):
        from carryrl.perception.screen_text import format_model_text

        self.path = Path(path)
        self.offsets = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("xb") as stream:
            for episode in dataset.episodes:
                for step, index in enumerate(episode.frame_indices):
                    key = index, step
                    if key in self.offsets:
                        continue
                    observed = read_imitation_screen_text(
                        reader, dataset.frames[index].copy(), step=step)
                    record = {"frame_index": index, **observed.to_dict(),
                              "model_text": format_model_text(observed)}
                    self.offsets[key] = stream.tell()
                    stream.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))

    def __call__(self, index: int, step: int) -> str:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index, step])
            return json.loads(stream.readline())["model_text"]


def train_imitation(
    agent, dataset: DemonstrationDataset, *, epochs: int = 1, backend: str = "torch",
    optimizer=None, mlx_backend=None, learning_rate: float = 1e-4, weight_decay: float = 0.01,
    minibatch_episodes: int = 1, max_grad_norm: float = 0.5, seed: int = 0,
    feature_cache: FrozenFrameCache | None = None,
    max_feature_bytes: int = 256 * 1024**2,
    feature_provider: Callable[[int], object] | None = None,
    on_epoch: Callable[[dict], None] | None = None,
    on_update: Callable[[dict], None] | None = None, max_wall_seconds: float | None = None,
    on_epoch_start: Callable[[dict], None] | None = None,
    action_balance: str = "uniform",
    canonical_replay_weight: float = 1.0,
    canonical_anchor_coefficient: float = 0.0,
    canonical_anchor_exclude_corrections: bool = False,
    screen_text_provider: Callable[[int, int], str | None] | None = None,
) -> dict:
    """Train optionally weighted per-episode NLL; keep provenance outside the policy graph.

    ``feature_provider(index)`` may supply externally cached frozen raw DINO patches,
    never learned projections. Otherwise a byte-bounded exact-RGB cache is used.
    External providers own their feature-memory budget. Epoch callbacks see synchronized
    Torch weights and optimizer state on both backends.
    ``screen_text_provider(frame_index, step)`` supplies OCR from that saved RGB at
    the actual episode decision; frozen text tokens exist only within each minibatch.
    Update callbacks record completed optimizer revisions immediately. Metal weights
    synchronize at epoch boundaries and in finally, including callback interruptions.
    The supplied persistent MLX backend may subsequently continue into on-policy PPO.
    Active/idle balancing and canonical replay weighting change the optimizer objective
    only. Reported NLL stays unweighted; active/idle diagnostics average per-episode
    means of present groups. Replay weighting leaves collection and sampling unchanged.
    Excluding corrections zeroes their anchor contribution while retaining the original
    canonical-episode denominator; every episode still contributes its usual BC objective.
    """
    if backend not in {"torch", "mlx"}:
        raise ValueError("imitation backend must be torch or mlx")
    if screen_text_provider is not None and getattr(agent.config, "text_encoder", "none") == "none":
        raise ValueError("screen OCR replay requires an enabled text_encoder")
    if getattr(agent, "screen_text_reader", None) is not None and screen_text_provider is None:
        raise ValueError("screen OCR imitation requires a frame-aligned screen_text_provider")
    if action_balance not in {"uniform", "active_idle"}:
        raise ValueError("action_balance must be uniform or active_idle")
    _validate_canonical_replay_weight(canonical_replay_weight)
    _validate_canonical_anchor_coefficient(canonical_anchor_coefficient)
    if type(canonical_anchor_exclude_corrections) is not bool:
        raise ValueError("canonical anchor exclude corrections must be bool")
    if (type(epochs) is not int or epochs < 1 or type(minibatch_episodes) is not int
            or minibatch_episodes < 1 or not dataset.episodes):
        raise ValueError("imitation requires positive epochs, batch size and nonempty episodes")
    for value in (learning_rate, max_grad_norm):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("learning rate and gradient norm must be positive and finite")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight decay must be nonnegative and finite")
    if max_wall_seconds is not None and (not math.isfinite(max_wall_seconds)
                                       or max_wall_seconds <= 0):
        raise ValueError("wall limit must be positive and finite")
    if any(parameter.requires_grad for parameter in agent.vision.parameters()):
        raise ValueError("imitation features require a frozen vision backbone")
    if feature_cache is not None and feature_cache.agent is not agent:
        raise ValueError("imitation cache belongs to a different agent")
    if feature_cache is None and feature_provider is None:
        feature_cache = FrozenFrameCache(agent, max(1, len(dataset.frames)),
                                         max_bytes=max_feature_bytes)
    if feature_cache is not None:
        feature_cache.assert_unchanged()
    # Revalidate externally assembled datasets before admitting them to training.
    for episode in dataset.episodes:
        dataset._validate_episode(episode)
    replay_domains = [_replay_domain(dataset, episode) for episode in dataset.episodes]
    replay_weights = [canonical_replay_weight if domain == "canonical_unstyled_96" else 1.0
                      for domain in replay_domains]
    replay_normalization = _replay_normalization(replay_weights)
    replay_summary = _replay_summary(replay_domains, canonical_replay_weight,
                                    normalization=replay_normalization)
    source = dataset.provenance()["source"]
    parameters = [parameter for parameter in agent.parameters() if parameter.requires_grad]
    if optimizer is None:
        optimizer = (mlx_backend.optimizer.canonical if mlx_backend is not None
                     else torch.optim.AdamW(parameters, lr=learning_rate,
                                            weight_decay=weight_decay))
    engine = mlx_backend
    if backend == "mlx":
        from carryrl.training.mlx import MLXPPOBackend
        if engine is None:
            engine = MLXPPOBackend(agent, optimizer)
        if engine.agent is not agent or engine.optimizer.canonical is not optimizer:
            raise ValueError("MLX backend belongs to another policy or optimizer")
    elif engine is not None:
        raise ValueError("an MLX engine requires backend mlx")
    parent, parent_hash = (_canonical_anchor_parent(agent, engine)
                           if canonical_anchor_coefficient else (None, None))
    anchor_eligible = [domain == "canonical_unstyled_96" and not (
        canonical_anchor_exclude_corrections and episode.source == CORRECTION_SOURCE)
        for episode, domain in zip(dataset.episodes, replay_domains, strict=True)]
    anchor_summary = {
        "coefficient": canonical_anchor_coefficient,
        "exclude_corrections": canonical_anchor_exclude_corrections,
        "parent_trainable_sha256": parent_hash,
        "parent_identity": "initial student policy; frozen independent trainable snapshot",
        "eligible_domain": "canonical_unstyled_96",
        "eligible_episodes": sum(anchor_eligible),
        "normalization_eligible_episodes": replay_domains.count("canonical_unstyled_96"),
        "excluded_correction_episodes": sum(
            canonical_anchor_exclude_corrections and episode.source == CORRECTION_SOURCE
            and domain == "canonical_unstyled_96"
            for episode, domain in zip(dataset.episodes, replay_domains, strict=True)),
        "aggregation": (
            "sum eligible episode(mean decision(sum conditional KL(parent||student))) / "
            "pre-mask canonical eligible episodes; excluded corrections contribute zero"
            if canonical_anchor_exclude_corrections else
            "mean eligible episode(mean decision(sum conditional KL(parent||student)))"),
        "prefixes": "expert target action components; episode history is actual executed actions",
    }

    def encode(episode):
        patches = []
        for index in episode.frame_indices:
            value = (feature_provider(index) if feature_provider is not None
                     else feature_cache.encode(dataset.frames[index].copy()))
            if isinstance(value, torch.Tensor):
                if value.requires_grad or value.ndim != 2 or value.shape[-1] != 384:
                    raise ValueError("imitation features require detached raw DINO patches")
                patches.append(value.detach().cpu())
            elif backend == "mlx":
                import mlx.core as mx
                if not isinstance(value, mx.array) or value.ndim != 2 or value.shape[-1] != 384:
                    raise ValueError("imitation feature provider requires raw DINO patches")
                patches.append(value)
            else:
                raise ValueError("Torch imitation feature provider requires detached Torch patches")
        if any(not isinstance(value, torch.Tensor) for value in patches):
            import mlx.core as mx

            from carryrl.agent.mlx_policy import to_mlx
            return mx.stack([to_mlx(value) for value in patches])
        return torch.stack(patches)

    rng = random.Random(seed)
    start, history, steps = time.monotonic(), [], 0
    agent.train()
    primary_error = None
    try:
        for epoch in range(epochs):
            epoch_started_at = time.monotonic()
            if on_epoch_start is not None:
                on_epoch_start({"epoch": epoch + 1, "optimizer_steps": steps,
                                "batches_per_epoch": math.ceil(
                                    len(dataset.episodes) / minibatch_episodes),
                                "seconds": time.monotonic() - start})
            order = list(range(len(dataset.episodes)))
            rng.shuffle(order)
            losses, task_losses, norms = [], {}, []
            objectives, task_objectives, active_losses, idle_losses = [], {}, [], []
            observed_weights, observed_domains, task_weights, batch_objectives = [], [], {}, []
            anchor_losses, bc_batches, total_batches = [], [], []
            anchor_eligible_count = 0
            completed = True
            for offset in range(0, len(order), minibatch_episodes):
                if max_wall_seconds is not None and time.monotonic() - start >= max_wall_seconds:
                    completed = False
                    break
                indices = order[offset:offset + minibatch_episodes]
                episodes = [dataset.episodes[index] for index in indices]
                batch_weights = [replay_weights[index] for index in indices]
                batch_domains = [replay_domains[index] for index in indices]
                batch_anchor_eligible = [parent is not None and anchor_eligible[index]
                                         for index in indices]
                batch_anchor_population = sum(parent is not None and
                    domain == "canonical_unstyled_96" for domain in batch_domains)
                batch = [(episode, encode(episode)) for episode in episodes]
                screen_batch = ([[screen_text_provider(index, step)
                                  for step, index in enumerate(episode.frame_indices)]
                                 for episode in episodes]
                                if screen_text_provider is not None else [None] * len(episodes))
                if backend == "torch":
                    optimizer.zero_grad(set_to_none=True)
                    parts, anchors = [], []
                    for (episode, patches), eligible, texts in zip(
                            batch, batch_anchor_eligible, screen_batch, strict=True):
                        output = agent.evaluate_episode(patches, list(episode.actions),
                            instruction=episode.instruction,
                            executed_actions=episode.executed_actions,
                            **({"screen_text": texts} if texts is not None else {}),
                            **({"return_decoder_trace": True} if eligible else {}))
                        parts.append(_episode_nlls(output["log_prob"], episode.actions,
                                                  action_balance))
                        if eligible:
                            with torch.no_grad():
                                teacher = parent.evaluate_episode(patches, list(episode.actions),
                                    instruction=episode.instruction,
                                    executed_actions=episode.executed_actions,
                                    **({"screen_text": texts} if texts is not None else {}),
                                    return_decoder_trace=True)
                            anchors.append(teacher_prefix_conditional_kl(
                                teacher["decoder_trace"], output["decoder_trace"]))
                    bc_objective = _replay_objective(torch.stack([part[0] for part in parts]),
                        batch_weights, normalization=replay_normalization)
                    anchor_kl = ((torch.stack(anchors).sum() / batch_anchor_population
                                  if canonical_anchor_exclude_corrections else
                                  torch.stack(anchors).mean()) if anchors
                                 else bc_objective.new_zeros(()))
                    objective = (bc_objective + canonical_anchor_coefficient * anchor_kl
                                 if canonical_anchor_coefficient else bc_objective)
                    objective.backward()
                    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm,
                                                         error_if_nonfinite=True)
                    optimizer.step()
                    objective_numbers = [float(part[0].detach()) for part in parts]
                    numbers = [float(part[1].detach()) for part in parts]
                    active_numbers = [float(part[2].detach()) for part in parts
                                      if part[2] is not None]
                    idle_numbers = [float(part[3].detach()) for part in parts
                                    if part[3] is not None]
                    norm = float(norm)
                    batch_objective = float(objective.detach())
                    bc_number = float(bc_objective.detach())
                    anchor_number = float(anchor_kl.detach())
                    anchor_numbers = [float(value.detach()) for value in anchors]
                else:
                    import mlx.core as mx
                    import mlx.nn as mnn

                    from carryrl.agent.mlx_policy import to_mlx

                    mlx_batch = [(episode, to_mlx(patches),
                                  to_mlx(agent._encode_instruction(episode.instruction))
                                  if episode.instruction is not None else None,
                                  [to_mlx(value) if value is not None else None
                                   for value in map(agent.encode_screen_text, texts)]
                                  if texts is not None else None)
                                 for (episode, patches), texts in zip(batch, screen_batch,
                                                                    strict=True)]

                    def loss_fn():
                        parts, anchors = [], []
                        for (episode, patches, feature, screen_features), eligible in zip(
                                mlx_batch, batch_anchor_eligible, strict=True):
                            output = engine.policy.evaluate_episode(
                                patches, episode.actions,
                                instruction_features=feature,
                                executed_actions=episode.executed_actions,
                                **({"screen_text_features": screen_features}
                                   if screen_features is not None else {}),
                                **({"return_decoder_trace": True} if eligible else {}))
                            parts.append(_episode_nlls(output["log_prob"], episode.actions,
                                                      action_balance))
                            if eligible:
                                teacher = parent.evaluate_episode(patches, episode.actions,
                                    instruction_features=feature,
                                    executed_actions=episode.executed_actions,
                                    **({"screen_text_features": screen_features}
                                       if screen_features is not None else {}),
                                    return_decoder_trace=True)
                                anchors.append(teacher_prefix_conditional_kl(
                                    teacher["decoder_trace"], output["decoder_trace"]))
                        metrics = {"objective": mx.stack([part[0] for part in parts]),
                                   "nll": mx.stack([part[1] for part in parts]),
                                   "active_nll": [part[2] for part in parts if part[2] is not None],
                                   "idle_nll": [part[3] for part in parts if part[3] is not None]}
                        bc = _replay_objective(metrics["objective"], batch_weights,
                                               normalization=replay_normalization)
                        anchor = ((mx.stack(anchors).sum() / batch_anchor_population
                                   if canonical_anchor_exclude_corrections else
                                   mx.stack(anchors).mean()) if anchors else mx.array(0.0))
                        metrics.update(bc_objective=bc, anchor_kl=anchor, anchors=anchors)
                        return (bc + canonical_anchor_coefficient * anchor
                                if canonical_anchor_coefficient else bc), metrics

                    (objective, metrics), grads = mnn.value_and_grad(engine.policy, loss_fn)()
                    mx.eval(objective, metrics, grads)
                    batch_objective = float(objective)
                    bc_number = float(metrics["bc_objective"])
                    anchor_number = float(metrics["anchor_kl"])
                    anchor_numbers = [float(value) for value in metrics["anchors"]]
                    objective_numbers = [float(value) for value in metrics["objective"]]
                    numbers = [float(value) for value in metrics["nll"]]
                    active_numbers = [float(value) for value in metrics["active_nll"]]
                    idle_numbers = [float(value) for value in metrics["idle_nll"]]
                    if not all(math.isfinite(value) for value in
                               (batch_objective, anchor_number, *numbers, *objective_numbers,
                                *active_numbers, *idle_numbers)):
                        raise FloatingPointError("invalid MLX imitation NLL")
                    active = set(engine.policy.weights) - {"value_head.weight", "value_head.bias"}
                    if all(feature is None and not any(value is not None
                            for value in (screen_features or []))
                           for _, _, feature, screen_features in mlx_batch):
                        active.discard("instruction_projection.weight")
                    if all(len(episode) <= agent.config.segment_length for episode in episodes):
                        active.discard("temporal.write_tokens")
                    norm = engine.apply_gradients(grads, max_grad_norm, active_names=active)
                steps += 1
                if on_update is not None:
                    on_update({
                        "epoch": epoch + 1, "optimizer_steps": steps,
                        "source": source, "method": METHOD, "backend": backend,
                        "nll": float(np.mean(numbers)), "gradient_norm": norm,
                        "objective": batch_objective,
                        "bc_objective": bc_number, "anchor_kl": anchor_number,
                        "total_objective": batch_objective,
                        "canonical_anchor_coefficient": canonical_anchor_coefficient,
                        "canonical_anchor_exclude_corrections": (
                            canonical_anchor_exclude_corrections),
                        "anchor_eligible_episodes": len(anchor_numbers),
                        "unweighted_objective": float(np.mean(objective_numbers)),
                        "replay_weighting": _replay_summary(batch_domains, canonical_replay_weight,
                                                             normalization=replay_normalization),
                        "action_balance": action_balance,
                        "active_nll": float(np.mean(active_numbers)) if active_numbers else None,
                        "idle_nll": float(np.mean(idle_numbers)) if idle_numbers else None,
                        "batch_episodes": len(episodes),
                        "tasks": [episode.task for episode in episodes],
                        "seconds": time.monotonic() - start,
                    })
                norms.append(norm)
                losses.extend(numbers)
                objectives.extend(objective_numbers)
                observed_weights.extend(batch_weights)
                observed_domains.extend(batch_domains)
                batch_objectives.append(batch_objective)
                anchor_losses.extend(anchor_numbers)
                if canonical_anchor_exclude_corrections:
                    anchor_losses.extend([0.0] * (batch_anchor_population - len(anchor_numbers)))
                anchor_eligible_count += len(anchor_numbers)
                bc_batches.append(bc_number)
                total_batches.append(batch_objective)
                active_losses.extend(active_numbers)
                idle_losses.extend(idle_numbers)
                for episode, number, objective, replay_weight in zip(
                        episodes, numbers, objective_numbers, batch_weights, strict=True):
                    task_losses.setdefault(episode.task, []).append(number)
                    task_objectives.setdefault(episode.task, []).append(objective)
                    task_weights.setdefault(episode.task, []).append(replay_weight)
            if engine is not None:
                engine.sync_to_torch()
            row = {
                "epoch": epoch + 1, "complete": completed, "source": source, "method": METHOD,
                "backend": backend, "nll": float(np.mean(losses)) if losses else None,
                "objective": _reported_replay_objective(objectives, observed_weights),
                "unweighted_objective": float(np.mean(objectives)) if objectives else None,
                "bc_objective": _reported_replay_objective(objectives, observed_weights),
                "anchor_kl": float(np.mean(anchor_losses)) if anchor_losses else 0.0,
                "total_objective": (_reported_replay_objective(objectives, observed_weights)
                    + canonical_anchor_coefficient * float(np.mean(anchor_losses))
                    if objectives and anchor_losses else
                    _reported_replay_objective(objectives, observed_weights)),
                "mean_batch_bc_objective": float(np.mean(bc_batches)) if bc_batches else None,
                "mean_batch_total_objective": (float(np.mean(total_batches))
                                               if total_batches else None),
                "anchor_eligible_episodes": anchor_eligible_count,
                "mean_batch_objective": (float(np.mean(batch_objectives))
                                         if batch_objectives else None),
                "replay_weighting": _replay_summary(observed_domains, canonical_replay_weight,
                                                     normalization=replay_normalization),
                "action_balance": action_balance,
                "active_nll": float(np.mean(active_losses)) if active_losses else None,
                "idle_nll": float(np.mean(idle_losses)) if idle_losses else None,
                "per_task_nll": {task: float(np.mean(values))
                                 for task, values in task_losses.items()},
                "per_task_objective": {task: _reported_replay_objective(values, task_weights[task])
                                       for task, values in task_objectives.items()},
                "per_task_unweighted_objective": {task: float(np.mean(values))
                                                  for task, values in task_objectives.items()},
                "episodes": len(losses), "optimizer_steps": steps,
                "max_gradient_norm": max(norms, default=0.0), "seconds": time.monotonic() - start,
                "epoch_training_seconds": time.monotonic() - epoch_started_at,
            }
            history.append(row)
            if on_epoch is not None:
                on_epoch(dict(row))
            if not completed:
                break
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            if engine is not None:
                engine.sync_to_torch()
        except BaseException as sync_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"Final MLX synchronization failed: {type(sync_error).__name__}: {sync_error}")
        finally:
            agent.eval()
    if feature_cache is not None:
        feature_cache.assert_unchanged()
    return {
        "source": source, "method": METHOD, "backend": backend, "history": history,
        "action_balance": action_balance,
        "replay_weighting": replay_summary,
        "canonical_anchor": anchor_summary,
        "epochs_completed": sum(row["complete"] for row in history), "optimizer_steps": steps,
        "stop_reason": "completed" if history[-1]["complete"] else "wall_limit",
        "feature_cache": feature_cache.statistics() if feature_cache is not None else None,
        "dataset": {key: value for key, value in dataset.provenance().items() if key != "episodes"},
    }
