"""Read-only diversity accounting for validated training RGB demonstrations."""

import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict

from carryrl.envs.atomic_suite import SUITE_VERSION
from carryrl.envs.presentation import PresentationProfile
from carryrl.envs.runtime_identity import atomic_runtime_level_id
from carryrl.training.imitation import (
    CORRECTION_SOURCE,
    DemonstrationDataset,
    _episode_runtime_level,
    validate_dataset_runtime_holdout,
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _profile(episode):
    if episode.presentation is None:
        return {"mode": "canonical", "native_size": episode.native_size or 96}
    return {"mode": "varied", **asdict(PresentationProfile(**episode.presentation))}


def _effective_profile(profile):
    result = dict(profile)
    if not result.get("textures", False):
        result.pop("texture_seed", None)
    return result


def _collection_kind(episode):
    if episode.collection_type == "live_styled":
        return "live_styled"
    return "posthoc_appearance_copy" if episode.presentation is not None else "canonical"


def audit_diversity(dataset, *, corpus_identity=None, source_config=None):
    """Count actual records; metadata equivalence never certifies independent experience.

    Every episode passes the same RGB, bank, outcome and complete-trajectory checks
    as collection. An optional source config also checks the evaluation endpoint
    runtime guard. Neither labels, instructions nor appearance identify a physical
    origin: actual executed actions are used for corrections, teacher actions only
    for teacher rollouts. No Cartesian cross-product of marginal values is assumed.
    """
    if not isinstance(dataset, DemonstrationDataset):
        raise TypeError("diversity audit requires a DemonstrationDataset")
    for episode in dataset.episodes:
        dataset._validate_episode(episode)
    if source_config is not None:
        validate_dataset_runtime_holdout(dataset,
            level_bank_seed=source_config.level_bank_seed,
            evaluation_difficulty=(source_config.evaluation_difficulty
                if source_config.evaluation_difficulty is not None else
                source_config.curriculum_difficulty
                if source_config.curriculum_difficulty is not None else 1.0))
    actual_bytes = sum(frame.nbytes for frame in dataset.frames)
    if actual_bytes != dataset.frame_bytes or actual_bytes > dataset.max_frame_bytes:
        raise ValueError("dataset RGB byte accounting differs from actual storage")

    kinds, sources, native, declared, effective = (Counter() for _ in range(5))
    origins, runtime_groups, cells = {}, {}, {}
    instructions = defaultdict(Counter)
    tasks = defaultdict(lambda: {"entries": 0, "decisions": 0,
                                 "physical_origins": set(), "runtime_groups": set()})
    actual_history, teacher_history = 0, 0
    for episode in dataset.episodes:
        size = episode.native_size or 96
        runtime_id = atomic_runtime_level_id(episode.task,
            _episode_runtime_level(episode), episode.difficulty)
        runtime_key = (episode.task, runtime_id, size)
        runtime_groups[runtime_key] = runtime_groups.get(runtime_key, 0) + 1
        actions = episode.executed_actions
        if actions is None:
            if episode.source == CORRECTION_SOURCE:
                raise ValueError("correction physical origins require executed history")
            actions = episode.actions
            teacher_history += 1
        else:
            actual_history += 1
        origin_id = _digest({"task": episode.task, "runtime_parameters_id": runtime_id,
            "native_size": size, "seed": episode.seed,
            "executed_actions": [action.to_dict() for action in actions],
            "terminated": episode.terminated, "truncated": episode.truncated,
            "decisions": len(episode)})
        origins[origin_id] = origins.get(origin_id, 0) + 1
        profile = _profile(episode)
        declared[_json(profile)] += 1
        effective[_json(_effective_profile(profile))] += 1
        kind = _collection_kind(episode)
        kinds[kind] += 1
        sources[episode.source] += 1
        native[(size, kind)] += 1
        instruction_id = (hashlib.sha256(episode.instruction.encode()).hexdigest()
                          if episode.instruction is not None else None)
        instructions[episode.task][instruction_id] += 1
        # Keep observed exact joint cells, rather than implying missing combinations
        # are required by an undeclared factorial collection design.
        cell = {"task": episode.task, "source": episode.source,
                "collection_kind": kind, "native_size": size,
                "theme": profile.get("theme"), "hud_layout": profile.get("hud_layout"),
                "textures": profile.get("textures", False),
                "instruction_sha256": instruction_id}
        key = _json(cell)
        if key not in cells:
            cells[key] = {**cell, "entries": 0, "decisions": 0, "origins": set()}
        cells[key]["entries"] += 1
        cells[key]["decisions"] += len(episode)
        cells[key]["origins"].add(origin_id)
        task = tasks[episode.task]
        task["entries"] += 1
        task["decisions"] += len(episode)
        task["physical_origins"].add(origin_id)
        task["runtime_groups"].add(runtime_key)

    return {
        "kind": "demonstration_diversity_audit", "schema_version": 1,
        "suite_version": SUITE_VERSION, "split": "train", "status": "completed",
        "audit_only": True, "updates": 0, "test_used": False,
        "policy_model_loaded": False, "optimizer_allocated": False,
        "source_corpus_identity": deepcopy(corpus_identity),
        "validation": {"all_episode_rgb_and_runtime_checks": True,
                       "evaluation_endpoint_guard": source_config is not None},
        "counts": {"entries": len(dataset.episodes),
                   "decisions": sum(len(row) for row in dataset.episodes),
                   "canonical_record_entries": kinds["canonical"],
                   "posthoc_appearance_copy_entries": kinds["posthoc_appearance_copy"],
                   "live_styled_rollout_entries": kinds["live_styled"],
                   "metadata_physical_origins": len(origins),
                   "runtime_parameter_groups": len(runtime_groups),
                   "declared_profile_configs": len(declared),
                   "effective_profile_configs": len(effective),
                   "unique_rgb_frames": len(dataset.frames), "rgb_payload_bytes": actual_bytes,
                   "history_from_executed_actions_entries": actual_history,
                   "history_from_teacher_actions_entries": teacher_history},
        "sources": dict(sorted(sources.items())),
        "tasks": {name: {key: len(value) if isinstance(value, set) else value
                         for key, value in report.items()}
                  for name, report in sorted(tasks.items())},
        "native_sizes": [{"native_size": size, "collection_kind": kind, "entries": count}
                         for (size, kind), count in sorted(native.items())],
        "runtime_parameter_groups": [{"task": task, "runtime_parameters_id": identity,
                                      "native_size": size, "entries": count}
            for (task, identity, size), count in sorted(runtime_groups.items())],
        "physical_origin_entry_multiplicities": dict(sorted(Counter(origins.values()).items())),
        "instructions": {task: [{"sha256": identity, "entries": count}
                                 for identity, count in sorted(counts.items(),
                                     key=lambda item: item[0] or "")]
                         for task, counts in sorted(instructions.items())},
        "declared_profiles": [{"profile": json.loads(key), "entries": count}
                              for key, count in sorted(declared.items())],
        "effective_profiles": [{"profile": json.loads(key), "entries": count}
                               for key, count in sorted(effective.items())],
        "cross_cells": [{**{key: value for key, value in cell.items() if key != "origins"},
                         "metadata_physical_origins": len(cell["origins"])}
                        for _, cell in sorted(cells.items())],
        "definitions": {
            "entries": "Corpus episode records, including repeated views; "
                "not independent sessions.",
            "metadata_physical_origins": "Task, runtime parameters, native size, seed, actual "
                "executed inputs and ending boundary; teacher inputs only for teacher sources. "
                "Excludes instruction, presentation, source and teacher target labels. "
                "Metadata equivalence groups, not certified independent experience or sessions.",
            "runtime_parameter_groups": "Task plus atomic_runtime_level_id at episode difficulty "
                "and native size; parameter diversity does not prove distinct geometry.",
            "effective_profile_configs": "Declared stable episode profiles with texture_seed "
                "removed when textures are disabled. Count of potentially pixel-affecting "
                "configurations, not a proof that every profile produces distinct RGB.",
            "unique_rgb_frames": "Interned exact RGB frames, including final and unused stored "
                "frames; byte accounting uses actual ndarray payloads, not compressed file size.",
            "cross_cells": "Observed exact task/source/collection_kind/native_size/theme/"
                "hud_layout/textures/instruction SHA256 combinations. Counts are record counts. "
                "No required grid declared; absent Cartesian combinations are not inferred.",
            "live_styled": "Only explicit collection_type=live_styled records; presentation "
                "metadata alone never proves an actual styled learner rollout.",
            "interpretation": "No learned-policy quality, generalization, efficacy or causal "
                "benefit is measured. Test banks are used only for exclusion checks, "
                "not evaluation.",
        },
    }
