"""Split-before-collection RGB probes for the diagnostic history reward benchmark."""

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv, atomic_level_banks
from carryrl.envs.atomic_suite import SCORING_VERSION, SUITE_VERSION, TASKS
from carryrl.envs.runtime_identity import allowed_runtime_train_indices, atomic_runtime_level_id
from carryrl.evaluation.scene_suite import (
    EVENT_NAMES,
    GOALS,
    SCENE_SUITE_VERSION,
    SuiteSceneAnnotator,
)
from carryrl.rewards.history import MASK_VERSION, mask_reward_hud
from carryrl.rewards.history_data import (
    MAX_CORPUS_BYTES,
    MAX_JSON_BYTES,
    HistoryRewardCorpus,
    _combined_identity,
    _prefix_identity,
)
from carryrl.rewards.visual import verify_visible_outcome
from carryrl.training.imitation import _bundle_file_identity, _bundle_json, scripted_teacher
from carryrl.training.run import atomic_json

HISTORY_SUITE_CORPUS_VERSION = "carryrl.scene_history_suite.v1"
PROBE_STRATEGIES = ("teacher", "neutral", "perturbed_teacher")
_OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
    "w": "s",
    "s": "w",
    "a": "d",
    "d": "a",
}
_KNOWN_TRAIN_BANKS = (0, 20261008, 20261015)


def _validate_config(config):
    if not isinstance(config, dict) or set(config) != {
        "tasks",
        "levels_per_split",
        "bank_seed",
        "seed",
        "max_steps",
        "goal_choice_utility",
    }:
        raise ValueError("invalid scene benchmark configuration fields")
    tasks = config["tasks"]
    if (
        not isinstance(tasks, (list, tuple))
        or not tasks
        or any(not isinstance(task, str) or task not in TASKS for task in tasks)
        or len(set(tasks)) != len(tasks)
        or type(config["levels_per_split"]) is not int
        or not 1 <= config["levels_per_split"] <= 16
        or type(config["bank_seed"]) is not int
        or not 0 <= config["bank_seed"] < 2**32
        or type(config["seed"]) is not int
        or not 0 <= config["seed"] < 2**32 - 160000
        or type(config["max_steps"]) is not int
        or not 1 <= config["max_steps"] <= 128
        or config["goal_choice_utility"] not in (None, "completion_only")
    ):
        raise ValueError("invalid scene benchmark collection settings")
    if "goal_choice" in tasks and config["goal_choice_utility"] != "completion_only":
        raise ValueError("goal_choice requires explicit completion_only scene objective")
    return config


def _identity(directory):
    files = {
        name: _bundle_file_identity(directory / name, limit)
        for name, limit in (
            ("frames.npy", MAX_CORPUS_BYTES + 65536),
            ("episodes.json", MAX_JSON_BYTES),
        )
    }
    return _combined_identity(files), files


def _select_levels(tasks, bank_seed, count, seed):
    """Exclude known train/test runtime aliases before any benchmark rollout."""
    selected = {}
    for task in tasks:
        bank = atomic_level_banks(task, bank_seed)
        indices = allowed_runtime_train_indices(task, bank_seed, 1.0)
        protected_train, protected_test, protected_validation = set(), set(), set()
        for source_seed in dict.fromkeys((*_KNOWN_TRAIN_BANKS, bank_seed)):
            source = atomic_level_banks(task, source_seed)
            protected_train.update(
                atomic_runtime_level_id(task, source["train"][index])
                for index in allowed_runtime_train_indices(task, source_seed, 1.0)
            )
            protected_test.update(atomic_runtime_level_id(task, level) for level in source["test"])
            protected_validation.update(
                atomic_runtime_level_id(task, level) for level in source["validation"]
            )
        train = []
        used = set()
        for offset in range(len(indices)):
            index = indices[(seed + offset) % len(indices)]
            runtime = atomic_runtime_level_id(task, bank["train"][index])
            if runtime not in used | protected_test | protected_validation:
                train.append(index)
                used.add(runtime)
            if len(train) == count:
                break
        validation, seen = [], set()
        for offset in range(len(bank["validation"])):
            index = (seed + offset) % len(bank["validation"])
            runtime = atomic_runtime_level_id(task, bank["validation"][index])
            if runtime not in protected_train | protected_test | seen:
                validation.append(index)
                seen.add(runtime)
            if len(validation) == count:
                break
        if len(train) != count or len(validation) != count:
            raise ValueError(f"{task}: insufficient distinct nonoverlapping scene benchmark levels")
        selected[task] = {"train": train, "validation": validation}
    return selected


def _action(teacher, frame, strategy, step):
    proposed = teacher.act(frame)
    if strategy == "neutral":
        return PhysicalAction(), None
    if strategy == "perturbed_teacher":
        if step < 2:
            return PhysicalAction(), proposed
        if step % 7 == 2:
            return PhysicalAction(
                keys=tuple(_OPPOSITE.get(key, key) for key in proposed.keys),
                mouse_buttons=(),
                dx=-proposed.dx,
                dy=-proposed.dy,
                wheel=-proposed.wheel,
            ), proposed
    return proposed, proposed


class HistorySuiteCorpus(HistoryRewardCorpus):
    def __init__(self, frames, episodes):
        super().__init__(frames, episodes, event_names=EVENT_NAMES)

    def targets(self, episode_index):
        return {**super().targets(episode_index), "event_names": self.event_names}


def collect_history_suite(
    output,
    *,
    tasks=TASKS,
    levels_per_split=2,
    bank_seed=20261008,
    seed=8176,
    max_steps=80,
    goal_choice_utility=None,
    on_progress=None,
):
    """Collect actual TRAIN/validation probes; neither teachers nor rewards are fitted.

    These are diagnostic scripts, not policy evaluations or successful expert
    demonstrations. Failed and unfinished branches stay in their original split.
    Alternate strategies of one runtime level share one group. Test is never played.
    """
    settings = _validate_config(
        {
            "tasks": tasks,
            "levels_per_split": levels_per_split,
            "bank_seed": bank_seed,
            "seed": seed,
            "max_steps": max_steps,
            "goal_choice_utility": goal_choice_utility,
        }
    )
    tasks = tuple(tasks)
    requested = Path(output)
    output = requested.resolve()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError("scene benchmark output already exists")
    selected = _select_levels(tasks, bank_seed, levels_per_split, seed)
    output.mkdir(parents=True, exist_ok=False)
    masked_frames, frame_map, episodes, signatures = [], {}, [], {}
    shared_prefixes = set()
    for task in tasks:
        for split in ("train", "validation"):
            bank = atomic_level_banks(task, bank_seed)[split]
            for offset, level_index in enumerate(selected[task][split]):
                runtime = atomic_runtime_level_id(task, bank[level_index])
                episode_seed = seed + TASKS.index(task) * 10000 + offset
                for strategy in PROBE_STRATEGIES:
                    env = AtomicEnv(
                        task,
                        split=split,
                        image_size=96,
                        max_steps=max_steps,
                        difficulty=1.0,
                        level_bank_seed=bank_seed,
                    )
                    annotator = SuiteSceneAnnotator(task, goal_choice_utility=goal_choice_utility)
                    teacher = scripted_teacher(task)
                    actual, indices, labels, previous_action = [], [], [], None
                    interventions = 0
                    terminated = truncated = False
                    prefix_hash = hashlib.sha256(
                        (MASK_VERSION + "\0" + GOALS[task]).encode()
                    ).digest()
                    try:
                        frame, _ = env.reset(
                            seed=episode_seed, options={"level_index": level_index}
                        )
                        for step in range(max_steps + 1):
                            masked = mask_reward_hud(frame)
                            if masked is None:
                                raise ValueError("ordinary scene benchmark HUD cannot be removed")
                            digest = hashlib.sha256(masked.tobytes()).hexdigest()
                            if digest not in frame_map:
                                if (len(masked_frames) + 1) * masked.nbytes > MAX_CORPUS_BYTES:
                                    raise MemoryError("scene benchmark exceeds masked RGB budget")
                                frame_map[digest] = len(masked_frames)
                                masked_frames.append(masked)
                            indices.append(frame_map[digest])
                            label = annotator.observe(masked, previous_action=previous_action)
                            labels.append(label)
                            prefix_hash = _prefix_identity(
                                prefix_hash,
                                masked,
                                previous_action.to_dict() if previous_action is not None else None,
                            )
                            signature = {
                                name: label[name]
                                for name in (
                                    "potential",
                                    "potential_valid",
                                    "event_targets",
                                    "event_valid",
                                    "terminal_success",
                                    "terminal_success_valid",
                                )
                            }
                            prior = signatures.setdefault(prefix_hash.hex(), (split, signature))
                            if prior[0] != split:
                                shared_prefixes.add(prefix_hash.hex())
                            if prior[1] != signature:
                                raise ValueError("conflicting targets for identical causal inputs")
                            if step and (terminated or truncated):
                                break
                            if step == max_steps:
                                raise ValueError("scene probe did not finish at its deadline")
                            action, proposal = _action(teacher, frame, strategy, step)
                            interventions += int(proposal is not None and proposal != action)
                            if hasattr(teacher, "observe_action"):
                                teacher.observe_action(action)
                            actual.append(action.to_dict())
                            frame, _api_reward, terminated, truncated, _info = env.step(action)
                            previous_action = action
                        outcome = verify_visible_outcome(frame)
                    finally:
                        env.close()
                    episodes.append(
                        {
                            "task": task,
                            "split": split,
                            "strategy": strategy,
                            "source": "scripted_rgb_probe",
                            "seed": episode_seed,
                            "level_bank_seed": bank_seed,
                            "level_index": level_index,
                            "runtime_group": runtime,
                            "difficulty": 1.0,
                            "goal": GOALS[task],
                            "frame_indices": indices,
                            "executed_actions": actual,
                            "annotations": labels,
                            "evaluation_only": {
                                "outcome": outcome,
                                "terminated": bool(terminated),
                                "truncated": bool(truncated),
                                "interventions": interventions,
                            },
                        }
                    )
                    if on_progress:
                        on_progress(
                            {
                                "task": task,
                                "split": split,
                                "strategy": strategy,
                                "episodes": len(episodes),
                            }
                        )
    array = np.lib.format.open_memmap(
        output / "frames.npy", mode="w+", dtype=np.uint8, shape=(len(masked_frames), 96, 96, 3)
    )
    for index, frame in enumerate(masked_frames):
        array[index] = frame
    array.flush()
    del array
    payload = {
        "format": HISTORY_SUITE_CORPUS_VERSION,
        "annotation_version": SCENE_SUITE_VERSION,
        "mask_version": MASK_VERSION,
        "suite_version": SUITE_VERSION,
        "scoring_version": SCORING_VERSION,
        "event_names": list(EVENT_NAMES),
        "frame_count": len(masked_frames),
        "frame_shape": [96, 96, 3],
        "config": {**settings, "tasks": list(tasks)},
        "selected_levels": selected,
        "episodes": episodes,
    }
    atomic_json(output / "episodes.json", payload, indent=None)
    digest, files = _identity(output)
    report = _report(episodes)
    report.update(
        status="completed",
        format=HISTORY_SUITE_CORPUS_VERSION,
        corpus_sha256=digest,
        files=files,
        unique_masked_frames=len(masked_frames),
        rgb_bytes=len(masked_frames) * 96 * 96 * 3,
        model_fit=False,
        ppo_eligible=False,
        updates=0,
        test_levels_played=False,
        test_outcomes_read=False,
        test_runtime_ids_used_for_overlap_guard=True,
        protected_train_bank_seeds=list(dict.fromkeys((*_KNOWN_TRAIN_BANKS, bank_seed))),
        shared_causal_prefixes_between_distinct_split_levels=len(shared_prefixes),
        interpretation="Scripted scene-reference benchmark, not policy or learned reward quality. "
        "Implemented targets have validity masks; some events are unobservable.",
    )
    atomic_json(output / "audit.json", report)
    return report


def _report(episodes):
    terminal_disagreements, premature, final_labels = [], [], 0
    for episode in episodes:
        labels = episode["annotations"]
        for index, label in enumerate(labels):
            if (
                label["terminal_success_valid"]
                and label["terminal_success"]
                and index != len(labels) - 1
            ):
                premature.append(
                    {
                        "task": episode["task"],
                        "split": episode["split"],
                        "strategy": episode["strategy"],
                        "frame": index,
                    }
                )
        final = labels[-1]
        outcome = episode["evaluation_only"]["outcome"]
        if final["terminal_success_valid"]:
            final_labels += 1
            if not outcome["known"] or final["terminal_success"] != outcome["success"]:
                terminal_disagreements.append(
                    {
                        "task": episode["task"],
                        "split": episode["split"],
                        "strategy": episode["strategy"],
                    }
                )
    if terminal_disagreements or premature:
        raise ValueError("scene terminal annotations contradict independent diagnostic outcomes")
    groups = {
        split: {row["runtime_group"] for row in episodes if row["split"] == split}
        for split in ("train", "validation")
    }
    if groups["train"] & groups["validation"]:
        raise ValueError("scene benchmark runtime groups overlap between splits")
    coverage = {}
    for split in ("train", "validation"):
        coverage[split] = {}
        for task in TASKS:
            selected = [row for row in episodes if row["split"] == split and row["task"] == task]
            labels = [label for row in selected for label in row["annotations"]]
            outcomes = Counter(
                "success"
                if row["evaluation_only"]["outcome"]["success"]
                else "failure"
                if row["evaluation_only"]["outcome"]["failure"]
                else "unfinished"
                for row in selected
            )
            coverage[split][task] = {
                "episodes": len(selected),
                "frames": len(labels),
                "runtime_groups": len({row["runtime_group"] for row in selected}),
                "outcomes": dict(outcomes),
                "potential_valid_frames": sum(row["potential_valid"] for row in labels),
                "known_scene_frames": sum(row["known"] for row in labels),
                "terminal_success_valid_frames": sum(
                    row["terminal_success_valid"] for row in labels
                ),
                "event_positives": {
                    name: sum(
                        row["event_valid"][name] and row["event_targets"][name] for row in labels
                    )
                    for name in EVENT_NAMES
                },
                "event_valid_frames": {
                    name: sum(row["event_valid"][name] for row in labels) for name in EVENT_NAMES
                },
            }
    return {
        "episodes": len(episodes),
        "runtime_groups": {split: len(values) for split, values in groups.items()},
        "runtime_group_overlap": [],
        "valid_final_terminal_labels": final_labels,
        "terminal_label_disagreements": terminal_disagreements,
        "premature_terminal_labels": premature,
        "coverage": coverage,
    }


def load_history_suite(directory, *, expected_sha256):
    directory = Path(directory).resolve(strict=True)
    if _identity(directory)[0] != expected_sha256:
        raise ValueError("scene suite corpus SHA256 mismatch")
    payload = _bundle_json(directory / "episodes.json", MAX_JSON_BYTES)
    if any(
        payload.get(name) != expected
        for name, expected in (
            ("format", HISTORY_SUITE_CORPUS_VERSION),
            ("annotation_version", SCENE_SUITE_VERSION),
            ("mask_version", MASK_VERSION),
            ("event_names", list(EVENT_NAMES)),
            ("suite_version", SUITE_VERSION),
            ("scoring_version", SCORING_VERSION),
            ("frame_shape", [96, 96, 3]),
        )
    ):
        raise ValueError("unsupported scene suite corpus contract")
    config, episodes = _validate_config(payload["config"]), payload["episodes"]
    if (
        not isinstance(episodes, list)
        or not episodes
        or len(episodes) > 2 * len(TASKS) * 16 * len(PROBE_STRATEGIES)
    ):
        raise ValueError("invalid bounded scene suite episodes")
    frames = np.load(directory / "frames.npy", allow_pickle=False, mmap_mode="r")
    if (
        frames.dtype != np.uint8
        or frames.shape != (payload["frame_count"], 96, 96, 3)
        or frames.nbytes > MAX_CORPUS_BYTES
    ):
        raise ValueError("invalid bounded scene suite RGB")
    selected = _select_levels(
        tuple(config["tasks"]), config["bank_seed"], config["levels_per_split"], config["seed"]
    )
    if payload["selected_levels"] != selected:
        raise ValueError("scene suite selected levels differ from preregistered splits")
    expected = {
        (task, split, index, strategy)
        for task in selected
        for split in ("train", "validation")
        for index in selected[task][split]
        for strategy in PROBE_STRATEGIES
    }
    observed = set()
    for episode in episodes:
        task, split = episode["task"], episode["split"]
        key = task, split, episode["level_index"], episode["strategy"]
        if (
            key not in expected
            or key in observed
            or episode["goal"] != GOALS[task]
            or episode["source"] != "scripted_rgb_probe"
            or episode["difficulty"] != 1.0
            or episode["level_bank_seed"] != config["bank_seed"]
        ):
            raise ValueError("scene suite episode does not match preregistered cohort")
        offset = selected[task][split].index(episode["level_index"])
        if episode["seed"] != config["seed"] + TASKS.index(task) * 10000 + offset:
            raise ValueError("scene suite seed differs from preregistered episode")
        observed.add(key)
        bank = atomic_level_banks(task, config["bank_seed"])[split]
        if episode["runtime_group"] != atomic_runtime_level_id(task, bank[episode["level_index"]]):
            raise ValueError("scene suite runtime group mismatch")
        indices, actions, labels = (
            episode[name] for name in ("frame_indices", "executed_actions", "annotations")
        )
        if (
            not 1 <= len(actions) <= config["max_steps"]
            or len(indices) != len(actions) + 1
            or len(labels) != len(indices)
            or any(type(index) is not int or not 0 <= index < len(frames) for index in indices)
        ):
            raise ValueError("invalid causal scene suite episode shape")
        annotator = SuiteSceneAnnotator(task, goal_choice_utility=config["goal_choice_utility"])
        env = AtomicEnv(
            task,
            split=split,
            image_size=96,
            max_steps=config["max_steps"],
            level_bank_seed=config["bank_seed"],
            difficulty=1.0,
        )
        teacher = scripted_teacher(task)
        previous_action = None
        interventions = 0
        try:
            frame, _ = env.reset(
                seed=episode["seed"], options={"level_index": episode["level_index"]}
            )
            for step, index in enumerate(indices):
                masked = mask_reward_hud(frame)
                if not np.array_equal(masked, frames[index]):
                    raise ValueError("scene suite actual masked RGB replay mismatch")
                if annotator.observe(masked, previous_action=previous_action) != labels[step]:
                    raise ValueError("scene suite annotations differ from causal replay")
                if step == len(actions):
                    break
                actual = PhysicalAction.from_dict(actions[step])
                planned, proposal = _action(teacher, frame, episode["strategy"], step)
                if planned != actual:
                    raise ValueError("scene suite probe strategy differs from actual commands")
                interventions += int(proposal is not None and proposal != actual)
                if hasattr(teacher, "observe_action"):
                    teacher.observe_action(actual)
                frame, _api_reward, terminated, truncated, _info = env.step(actual)
                if (terminated or truncated) != (step == len(actions) - 1):
                    raise ValueError("scene suite actual episode boundary mismatch")
                previous_action = actual
            if episode["evaluation_only"] != {
                "outcome": verify_visible_outcome(frame),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "interventions": interventions,
            }:
                raise ValueError("scene suite diagnostic outcome mismatch")
        finally:
            env.close()
    if observed != expected:
        raise ValueError("scene suite missing preregistered episodes")
    _report(episodes)
    corpus = HistorySuiteCorpus(frames, episodes)
    corpus.corpus_sha256 = expected_sha256
    corpus.directory = directory
    return corpus
