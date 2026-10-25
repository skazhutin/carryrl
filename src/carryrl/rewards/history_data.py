"""Causal masked-RGB supervision corpus, separate from policy imitation data.

Scene annotations are disclosed suite-specific RGB references. They are training
targets and diagnostics, never inputs or a substitute primary reward predictor.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from carryrl.agent.actions import PhysicalAction, encode_actions
from carryrl.envs.atomic import AtomicEnv, atomic_level_banks
from carryrl.envs.atomic_suite import TASKS
from carryrl.envs.runtime_identity import atomic_runtime_level_id
from carryrl.evaluation.scene_progress import (
    EVENT_NAMES,
    GOALS,
    SCENE_PROGRESS_VERSION,
    SUPPORTED_TASKS,
    SceneProgressAnnotator,
)
from carryrl.rewards.history import MASK_VERSION, mask_reward_hud
from carryrl.training.imitation import (
    CORRECTION_SOURCE,
    SOURCE,
    _bundle_file_identity,
    _bundle_json,
    load_correction_replacement_bundle,
)
from carryrl.training.run import atomic_json, file_sha256

HISTORY_CORPUS_VERSION = "carryrl.scene_history_corpus.v1"
MAX_CORPUS_BYTES = 256 * 1024**2
MAX_JSON_BYTES = 16 * 1024**2


def _combined_identity(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _data_identity(directory):
    files = {name: _bundle_file_identity(directory / name, limit)
             for name, limit in (("frames.npy", MAX_CORPUS_BYTES + 65536),
                                 ("episodes.json", MAX_JSON_BYTES))}
    return _combined_identity(files), files


def _validate_source(directory, expected_sha256, config):
    """Authenticate all teacher/correction replay before deriving any new target."""
    files = {name: _bundle_file_identity(directory / name, limit)
             for name, limit in (("frames.npy", MAX_CORPUS_BYTES + 65536),
                 ("trajectories.json", MAX_JSON_BYTES), ("coverage.json", MAX_JSON_BYTES))}
    if _combined_identity(files) != expected_sha256:
        raise ValueError("history source bundle SHA256 mismatch")
    coverage = _bundle_json(directory / "coverage.json", MAX_JSON_BYTES)
    declared = coverage["preregistration"]
    producer = declared["expected_checkpoint_identity"]
    imported, report = load_correction_replacement_bundle(
        directory, expected_sha256=expected_sha256, correction_tasks=declared["tasks"],
        episodes_per_task=declared["episodes_per_task"],
        seed=declared["correction_collection_seed"],
        difficulty=1.0, max_steps=config.max_episode_steps, level_bank_seed=config.level_bank_seed,
        episode_instructions=config.episode_instructions, source_identity=producer,
        initial_agent_sha256=producer["agent_sha256"], reward_sha256=producer["reward_sha256"],
        max_frame_bytes=MAX_CORPUS_BYTES)
    del imported
    records = _bundle_json(directory / "trajectories.json", MAX_JSON_BYTES)["episodes"]
    frames = np.load(directory / "frames.npy", allow_pickle=False, mmap_mode="r")
    return frames, records, report


def _prefix_identity(previous, frame, action):
    digest = hashlib.sha256(previous)
    digest.update(bytes.fromhex(hashlib.sha256(frame.tobytes()).hexdigest()))
    if action is not None:
        digest.update(json.dumps(action, sort_keys=True, separators=(",", ":")).encode())
    return digest.digest()


class HistoryRewardCorpus:
    """Only prefix() defines model inputs; metadata and targets are separate."""

    def __init__(self, frames, episodes, *, event_names=EVENT_NAMES):
        self.frames = frames
        self.episodes = episodes
        self.event_names = tuple(event_names)

    def prefix(self, episode_index, frame_index):
        episode = self.episodes[episode_index]
        indices = episode["frame_indices"]
        if type(frame_index) is not int or not 0 <= frame_index < len(indices):
            raise ValueError("prefix index outside the episode")
        # One initial frame, then the observation following each actual command.
        actual = [PhysicalAction.from_dict(row)
                  for row in episode["executed_actions"][:frame_index]]
        previous = encode_actions([PhysicalAction(), *actual])
        return {"frames": np.stack([self.frames[index] for index in indices[:frame_index + 1]]),
                "executed_previous_actions": previous, "goal": episode["goal"]}

    def targets(self, episode_index):
        rows = self.episodes[episode_index]["annotations"]
        return {"potential": np.asarray([row["potential"]
                    if row["potential_valid"] else float("nan") for row in rows], np.float32),
                "potential_valid": np.asarray([row["potential_valid"] for row in rows], bool),
                "events": np.asarray([[float(row["event_targets"][name])
                                       for name in self.event_names]
                                       for row in rows], np.float32),
                "event_valid": np.asarray([[row["event_valid"][name] for name in self.event_names]
                                            for row in rows], bool),
                "terminal_success": np.asarray([row["terminal_success"]
                    if row["terminal_success_valid"] else float("nan")
                    for row in rows], np.float32),
                "terminal_success_valid": np.asarray([row["terminal_success_valid"]
                                                        for row in rows], bool)}


@torch.no_grad()
def encode_history_episode(corpus, episode_index, *, vision_encoder, goal_encoder,
                           frame_batch_size=4):
    """Encode verified masked RGB afresh; never consult a policy feature store.

    Encoders must be frozen and return per-frame patches or vectors, and an
    explicit-goal vector respectively. Only the separate history head is learned.
    No target, provenance field or teacher recommendation enters either encoder.
    """
    if type(frame_batch_size) is not int or not 1 <= frame_batch_size <= 32:
        raise ValueError("frame_batch_size must be between 1 and 32")
    if any(parameter.requires_grad for module in (vision_encoder, goal_encoder)
           for parameter in module.parameters()):
        raise ValueError("scene/goal encoders must be frozen")
    episode = corpus.episodes[episode_index]
    inputs = corpus.prefix(episode_index, len(episode["frame_indices"]) - 1)
    modes = vision_encoder.training, goal_encoder.training
    vision_encoder.eval()
    goal_encoder.eval()
    try:
        device = next(vision_encoder.parameters(), torch.empty(0)).device
        batches = []
        for start in range(0, len(inputs["frames"]), frame_batch_size):
            rgb = torch.from_numpy(inputs["frames"][start:start + frame_batch_size]).to(device)
            features = vision_encoder(rgb)
            if (not isinstance(features, torch.Tensor) or features.ndim not in (2, 3)
                    or features.shape[0] != len(rgb) or not features.is_floating_point()
                    or not torch.isfinite(features).all()):
                raise ValueError("scene encoder must return finite per-frame vectors or patches")
            batches.append(features.mean(dim=1) if features.ndim == 3 else features)
        features = torch.cat(batches)
        goal = goal_encoder(inputs["goal"])
        if (not isinstance(goal, torch.Tensor) or goal.ndim != 1
                or not goal.is_floating_point() or not torch.isfinite(goal).all()
                or not goal.count_nonzero()):
            raise ValueError("goal encoder must return a finite nonzero explicit-goal vector")
        return {"scene_features": features.detach().unsqueeze(0),
                "executed_previous_actions": inputs["executed_previous_actions"].to(
                    device=features.device, dtype=features.dtype).unsqueeze(0),
                "explicit_goal_features": goal.detach().to(
                    device=features.device, dtype=features.dtype).unsqueeze(0)}
    finally:
        vision_encoder.train(modes[0])
        goal_encoder.train(modes[1])


def export_history_reward_corpus(source, *, expected_sha256, source_config, output,
                                 on_progress=None):
    """Derive TRAIN targets from a pinned replay-validated fresh correction bundle.

    Supporting teachers remain explicitly labelled diagnostic probes here, not
    imitation additions. No random frame/episode split is invented: all source
    trajectories retain TRAIN and paired runtime levels remain one group.
    """
    requested_output = Path(output)
    source, output = Path(source).resolve(strict=True), requested_output.resolve()
    if (output == source or source in output.parents or output in source.parents
            or requested_output.is_symlink()):
        raise ValueError("history output must be outside its source bundle")
    if output.exists():
        raise FileExistsError("history output already exists")
    frames, records, replay = _validate_source(source, expected_sha256, source_config)
    output.mkdir(parents=True, exist_ok=False)
    saved, indices, episodes, unsupported, conflicts = [], {}, [], Counter(), []
    signatures = {}
    coverage = {}
    for number, row in enumerate(records):
        task = row["task"]
        if task not in SUPPORTED_TASKS:
            unsupported[task] += 1
            continue
        raw_actions = (row["executed_actions"] if row["source"] == CORRECTION_SOURCE
                       else row["actions"])
        actions = [PhysicalAction.from_dict(action) for action in raw_actions]
        raw_indices = [*row["frame_indices"], row["final_frame_index"]]
        annotator = SceneProgressAnnotator(task)
        annotations, remapped = [], []
        prefix_hash = hashlib.sha256((MASK_VERSION + "\0" + GOALS[task]).encode()).digest()
        for step, raw_index in enumerate(raw_indices):
            frame = frames[raw_index]
            masked = mask_reward_hud(frame)
            if masked is None:
                raise ValueError("authenticated source has no unambiguous removable HUD")
            key = hashlib.sha256(masked.tobytes()).hexdigest()
            if key not in indices:
                if (len(saved) + 1) * masked.nbytes > MAX_CORPUS_BYTES:
                    raise MemoryError("masked scene corpus exceeds RGB byte budget")
                indices[key] = len(saved)
                saved.append(masked)
            remapped.append(indices[key])
            annotation = annotator.observe(masked)
            annotations.append(annotation)
            prefix_hash = _prefix_identity(prefix_hash, masked,
                                           raw_actions[step - 1] if step else None)
            signature = {"potential": annotation["potential"],
                "potential_valid": annotation["potential_valid"],
                "event_targets": annotation["event_targets"],
                "event_valid": annotation["event_valid"],
                "terminal_success": annotation["terminal_success"],
                "terminal_success_valid": annotation["terminal_success_valid"]}
            previous = signatures.setdefault(prefix_hash.hex(), signature)
            if previous != signature:
                conflicts.append({"task": task, "source_row": number, "step": step,
                                  "prefix_sha256": prefix_hash.hex()})
        bank = atomic_level_banks(task, row["level_bank_seed"])["train"]
        runtime = atomic_runtime_level_id(task, bank[row["level_index"]], row["difficulty"])
        episodes.append({"task": task, "source": row["source"], "split": "train",
            "seed": row["seed"], "level_index": row["level_index"], "runtime_group": runtime,
            "level_bank_seed": row["level_bank_seed"], "difficulty": row["difficulty"],
            "goal": GOALS[task], "frame_indices": remapped, "executed_actions": raw_actions,
            "annotations": annotations,
            "evaluation_only": {"source_outcome": row["outcome"],
                "teacher_target_differences": sum(a.to_dict() != b for a, b in zip(
                    actions, row["actions"], strict=True)) if row["source"] == CORRECTION_SOURCE
                    else 0}})
        if on_progress:
            on_progress({"task": task, "episodes_exported": len(episodes)})
    if not episodes or conflicts:
        raise ValueError("empty or contradictory causal scene corpus")
    for task in TASKS:
        selected = [row for row in episodes if row["task"] == task]
        annotations = [value for row in selected for value in row["annotations"]]
        coverage[task] = {"supported": task in SUPPORTED_TASKS, "episodes": len(selected),
            "runtime_groups": len({row["runtime_group"] for row in selected}),
            "frames": len(annotations),
            "potential_valid_frames": sum(row["potential_valid"] for row in annotations),
            "terminal_success_valid_frames": sum(row["terminal_success_valid"]
                                                  for row in annotations),
            "event_positives": {name: sum(row["event_valid"][name]
                and row["event_targets"][name] for row in annotations) for name in EVENT_NAMES},
            "unsupported_source_episodes": unsupported[task]}
    frame_array = np.lib.format.open_memmap(output / "frames.npy", mode="w+", dtype=np.uint8,
                                           shape=(len(saved), 96, 96, 3))
    for index, frame in enumerate(saved):
        frame_array[index] = frame
    frame_array.flush()
    del frame_array
    payload = {"format": HISTORY_CORPUS_VERSION, "mask_version": MASK_VERSION,
        "annotation_version": SCENE_PROGRESS_VERSION, "event_names": list(EVENT_NAMES),
        "max_episode_steps": source_config.max_episode_steps,
        "frame_count": len(saved), "frame_shape": [96, 96, 3], "episodes": episodes}
    atomic_json(output / "episodes.json", payload)
    digest, files = _data_identity(output)
    # Revalidate raw bytes after derivation too; the public importer authenticated
    # the three pinned source files immediately before this bounded read.
    for name, identity in replay["files"].items():
        if file_sha256(source / name) != identity["sha256"]:
            raise ValueError("source bundle changed during scene target export")
    groups = {}
    for row in episodes:
        groups.setdefault(row["runtime_group"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("scene corpus splits overlap by runtime level")
    report = {"status": "completed", "format": HISTORY_CORPUS_VERSION,
        "source_directory": str(source), "source_bundle_sha256": expected_sha256,
        "source_replay_validated": True, "source_episodes": len(records),
        "episodes": len(episodes), "runtime_groups": len(groups),
        "sources": dict(Counter(row["source"] for row in episodes)),
        "split": "train", "validation_collected": False, "test_used": False,
        "future_frame_inputs": False, "teacher_targets_as_inputs": False,
        "hud_inputs": False, "prefix_label_conflicts": conflicts,
        "corpus_sha256": digest, "files": files,
        "rgb_bytes": len(saved) * 96 * 96 * 3, "unique_masked_frames": len(saved),
        "coverage": coverage, "model_fit": False, "ppo_eligible": False,
        "updates": 0,
        "interpretation": "Suite-specific RGB scene supervision, not learned reward quality. "
                          "Paired teacher/correction episodes share one runtime group; "
                          "unknown targets stay masked. No autonomous goal discovery."}
    atomic_json(output / "audit.json", report)
    return report


def load_history_reward_corpus(directory, *, expected_sha256):
    directory = Path(directory).resolve(strict=True)
    digest, _ = _data_identity(directory)
    if digest != expected_sha256:
        raise ValueError("scene history corpus SHA256 mismatch")
    payload = _bundle_json(directory / "episodes.json", MAX_JSON_BYTES)
    if (payload.get("format") != HISTORY_CORPUS_VERSION
            or payload.get("mask_version") != MASK_VERSION
            or payload.get("annotation_version") != SCENE_PROGRESS_VERSION
            or payload.get("event_names") != list(EVENT_NAMES)
            or payload.get("frame_shape") != [96, 96, 3]
            or type(payload.get("max_episode_steps")) is not int
            or not 1 <= payload["max_episode_steps"] <= 128):
        raise ValueError("unsupported scene history corpus contract")
    frames = np.load(directory / "frames.npy", allow_pickle=False, mmap_mode="r")
    episodes = payload.get("episodes")
    if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (96, 96, 3)
            or frames.shape[0] != payload.get("frame_count") or frames.nbytes > MAX_CORPUS_BYTES
            or not isinstance(episodes, list) or not episodes):
        raise ValueError("invalid bounded scene corpus frames or episodes")
    for episode in episodes:
        task, indices = episode.get("task"), episode.get("frame_indices")
        actions, annotations = episode.get("executed_actions"), episode.get("annotations")
        if (task not in SUPPORTED_TASKS or episode.get("goal") != GOALS[task]
                or episode.get("split") != "train" or episode.get("source") not in {SOURCE,
                    CORRECTION_SOURCE} or not isinstance(indices, list)
                or not 2 <= len(indices) <= 129
                or any(type(index) is not int or not 0 <= index < len(frames) for index in indices)
                or not isinstance(actions, list) or len(actions) != len(indices) - 1
                or not isinstance(annotations, list) or len(annotations) != len(indices)):
            raise ValueError("invalid causal scene corpus episode")
        for action in actions:
            PhysicalAction.from_dict(action)
        if (type(episode.get("level_bank_seed")) is not int
                or not 0 <= episode["level_bank_seed"] < 2**32
                or type(episode.get("level_index")) is not int
                or not 0 <= episode["level_index"] < 64
                or type(episode.get("seed")) is not int
                or type(episode.get("difficulty")) not in (int, float)
                or episode["difficulty"] != 1.0):
            raise ValueError("invalid scene corpus runtime provenance")
        bank = atomic_level_banks(task, episode["level_bank_seed"])["train"]
        group = atomic_runtime_level_id(task, bank[episode["level_index"]], episode["difficulty"])
        if episode.get("runtime_group") != group:
            raise ValueError("scene corpus runtime group differs from actual level")
        annotator = SceneProgressAnnotator(task)
        for index, annotation in zip(indices, annotations, strict=True):
            if annotation != annotator.observe(frames[index]):
                raise ValueError("scene corpus annotations differ from masked RGB replay")
        _validate_masked_replay(frames, episode, payload["max_episode_steps"])
    return HistoryRewardCorpus(frames, episodes)


def _validate_masked_replay(frames, episode, max_steps):
    env = AtomicEnv(episode["task"], split="train", image_size=96, max_steps=max_steps,
                    difficulty=episode["difficulty"], level_bank_seed=episode["level_bank_seed"])
    try:
        frame, _ = env.reset(seed=episode["seed"],
                             options={"level_index": episode["level_index"]})
        actions = episode["executed_actions"]
        for index, frame_index in enumerate(episode["frame_indices"]):
            if not np.array_equal(mask_reward_hud(frame), frames[frame_index]):
                raise ValueError("scene corpus actual masked RGB replay mismatch")
            if index == len(actions):
                break
            frame, _api_reward, terminated, truncated, _info = env.step(
                PhysicalAction.from_dict(actions[index]))
            if (terminated or truncated) != (index == len(actions) - 1):
                raise ValueError("scene corpus actual episode boundary mismatch")
    finally:
        env.close()
