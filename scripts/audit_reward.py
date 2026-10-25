"""Audit a frozen RGB reward recognizer on TRAIN teachers and public TRAIN HUDs.

No policy decisions, optimizer, reward updates or training corpus are produced.
Pixel-verifier answers are used only for diagnostics, never reward composition.
"""

import argparse
import gc
import math
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from carryrl.envs.atomic import AtomicEnv
from carryrl.envs.atomic_suite import SCORING_VERSION, SUITE_VERSION, TASKS
from carryrl.envs.presentation import HUD_LAYOUTS, THEME_SPLITS, render_hud
from carryrl.envs.runtime_identity import allowed_runtime_train_indices
from carryrl.rewards.composition import (
    OUTCOME_POTENTIAL_VERSION,
    OutcomePotentialConfig,
    RewardComposer,
    RewardCompositionConfig,
    classify_learned_event,
    reward_composition_identity,
)
from carryrl.rewards.visual import (
    STATUSES,
    LearnedVisualReward,
    ScreenVisualReward,
    draw_visible_hud,
    verify_visible_outcome,
)
from carryrl.training.imitation import collect_scripted_demonstrations, demonstration_episode_digest
from carryrl.training.run import atomic_json, file_sha256, provenance
from carryrl.training.sequence import _prediction, load_sequence_checkpoint, tensor_hash
from carryrl.training.sequence_config import SequenceRunConfig


def _label(frame):
    label = verify_visible_outcome(frame)
    status = ("unknown" if not label["known"] else "success" if label["success"]
              else "failure" if label["failure"] else "play")
    return {**label, "status": status}


def _readout(reward, frame, threshold):
    prediction, uncertain = _prediction(reward, frame, threshold)
    # Verifier remains outside the predictor and composition input.
    return prediction, _label(frame), bool(uncertain)


def _classification(prediction, label, *, event_threshold, score_threshold):
    status = classify_learned_event(prediction, event_threshold)
    score_trusted = (bool(prediction.get("known", True))
                     and prediction.get("score_confidence", -1) >= score_threshold)
    return {"truth": label["status"], "predicted": status,
            "event_correct": status == label["status"], "score_trusted": score_trusted,
            "score_correct": (prediction["score"] == label["score"] if label["known"] else None)}


def _summarize_readouts(rows):
    known = [row for row in rows if row["truth"] != "unknown"]
    trusted = [row for row in known if row["score_trusted"]]
    confusion = {}
    for row in rows:
        key = row["truth"] + "->" + row["predicted"]
        confusion[key] = confusion.get(key, 0) + 1
    return {"frames": len(rows), "known_truth_frames": len(known),
            "event_correct": sum(row["event_correct"] for row in rows),
            "event_confusion": confusion,
            "trusted_score_frames": len(trusted),
            "trusted_score_correct": sum(row["score_correct"] for row in trusted),
            "all_known_score_correct": sum(row["score_correct"] for row in known)}


def _hud_probes(fixed_size=None):
    """Public vocabulary probes, not held-out scenes or a generalization benchmark."""
    for size in ((fixed_size,) if fixed_size is not None else (64, 96, 128)):
        for score in range(6):
            for status in STATUSES:
                frame = np.full((size, size, 3), 71, dtype=np.uint8)
                draw_visible_hud(frame, score, status)
                yield f"canonical/{size}/{score}/{status}", frame
    for theme in (() if fixed_size is not None else THEME_SPLITS["train"]):
        for layout in HUD_LAYOUTS:
            for score in range(6):
                for status in STATUSES:
                    frame = np.full((192, 192, 3), 71, dtype=np.uint8)
                    panel = render_hud(theme, layout, score, status)
                    height, width = panel.shape[:2]
                    frame[8:8 + height, 8:8 + width] = panel
                    yield f"train_hud/{theme}/{layout}/{score}/{status}", frame
    size = fixed_size or 96
    yield "unknown/blank", np.zeros((size, size, 3), dtype=np.uint8)
    ambiguous = np.full((size, size, 3), 71, dtype=np.uint8)
    draw_visible_hud(ambiguous, 0, "Play")
    ambiguous[-28:] = ambiguous[:28].copy()
    yield "unknown/two_panels", ambiguous


def _replay(dataset, episode, horizon):
    """Authenticate actual RGB and obtain structural boundaries independently of labels."""
    env = AtomicEnv(episode.task, split="train", image_size=episode.native_size or 96,
                    max_steps=horizon,
                    difficulty=episode.difficulty, level_bank_seed=episode.level_bank_seed)
    try:
        frame, _ = env.reset(seed=episode.seed, options={"level_index": episode.level_index})
        actions = episode.executed_actions or episode.actions
        for index, action in enumerate(actions):
            if not np.array_equal(frame, dataset.frames[episode.frame_indices[index]]):
                raise ValueError("teacher audit replay RGB mismatch")
            frame, _api_reward, terminated, truncated, _info = env.step(action)
            final = index == len(actions) - 1
            if (terminated or truncated) != final:
                raise ValueError("teacher audit replay boundary mismatch")
            expected = (episode.final_frame_index if final else episode.frame_indices[index + 1])
            if not np.array_equal(frame, dataset.frames[expected]):
                raise ValueError("teacher audit replay next RGB mismatch")
            yield frame, action, bool(terminated), bool(truncated)
    finally:
        env.close()


def audit_reward(reward, *, tasks=TASKS, levels_per_task=2, level_bank_seed=20261008,
                 seed=8176, horizon=80, gamma=0.99, score_threshold=0.6,
                 parameters=None, max_frame_bytes=128 * 1024**2, on_progress=None):
    tasks = tuple(tasks)
    if (not tasks or len(set(tasks)) != len(tasks) or any(task not in TASKS for task in tasks)
            or type(levels_per_task) is not int or levels_per_task < 1
            or type(level_bank_seed) is not int or not 0 <= level_bank_seed < 2**32
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(horizon) is not int or not 1 <= horizon <= 128
            or type(max_frame_bytes) is not int or max_frame_bytes < 1):
        raise ValueError("invalid reward audit configuration")
    config = RewardCompositionConfig(OUTCOME_POTENTIAL_VERSION, parameters)
    config.validate_context(gamma=gamma, horizon=horizon, tasks=tasks,
                            score_confidence_threshold=score_threshold)
    fixed_size = (reward.image_size if isinstance(reward, LearnedVisualReward)
                  and not isinstance(reward, ScreenVisualReward) else None)
    if fixed_size is not None and fixed_size not in (64, 96, 128):
        raise ValueError("teacher reward audit supports fixed-size legacy RGB only at 64/96/128")
    native_size = fixed_size if fixed_size in (64, 128) else None
    for task in tasks:
        if levels_per_task > len(allowed_runtime_train_indices(task, level_bank_seed, 1.0)):
            raise ValueError("reward audit requires distinct allowed train levels")
    started = time.monotonic()
    records, task_reports, all_readouts = [], {}, []
    peak_rgb = 0
    mode = reward.training
    reward.eval()
    try:
        with torch.no_grad():
            for task in tasks:
                if on_progress:
                    on_progress({"task": task, "event": "task_started"})
                dataset = collect_scripted_demonstrations(
                    tasks=(task,), episodes_per_task=levels_per_task,
                    level_bank_seed=level_bank_seed,
                    seed=seed, max_steps=horizon, max_frame_bytes=max_frame_bytes,
                    image_size=fixed_size or 96, native_size=native_size)
                peak_rgb = max(peak_rgb, dataset.frame_bytes)
                task_rows = []
                for episode in dataset.episodes:
                    first = dataset.frames[episode.frame_indices[0]]
                    prediction, label, _ = _readout(reward, first, score_threshold)
                    classifications = [_classification(prediction, label,
                        event_threshold=config.parameters.event_confidence_threshold,
                        score_threshold=score_threshold)]
                    composers = {"legacy": RewardComposer(RewardCompositionConfig(), gamma=gamma,
                        horizon=horizon, score_confidence_threshold=score_threshold,
                        initial_prediction=prediction), "candidate": RewardComposer(config,
                        gamma=gamma, horizon=horizon, score_confidence_threshold=score_threshold,
                        initial_prediction=prediction)}
                    steps = {key: [] for key in composers}
                    for frame, action, terminated, truncated in _replay(dataset, episode, horizon):
                        prediction, label, _ = _readout(reward, frame, score_threshold)
                        classifications.append(_classification(prediction, label,
                            event_threshold=config.parameters.event_confidence_threshold,
                            score_threshold=score_threshold))
                        for key, composer in composers.items():
                            steps[key].append(composer.step(prediction, action,
                                terminated=terminated, truncated=truncated).to_dict())
                    all_readouts.extend(classifications)
                    task_rows.extend(classifications)
                    records.append({"task": task, "seed": episode.seed,
                        "level_index": episode.level_index, "level_id": episode.level_id,
                        "trace_sha256": demonstration_episode_digest(dataset, episode),
                        "actions": len(episode), "actual_rgb_replay_verified": True,
                        "boundary_source": "replayed_environment_flags_only",
                        "final_truth": label, "final_prediction": prediction,
                        "valid_for_learning": all(step["valid_for_learning"]
                                                  for step in steps["candidate"]),
                        "rewards": {key: {"sum": sum(row["total"] for row in rows),
                            "discounted_return": math.fsum(gamma**i * row["total"]
                                                           for i, row in enumerate(rows)),
                            "component_sums": {name: math.fsum(row[name] for row in rows)
                                for name in ("outcome", "progress", "time_cost", "input_cost")},
                            "steps": rows} for key, rows in steps.items()}})
                task_reports[task] = _summarize_readouts(task_rows)
                del dataset
                if on_progress:
                    on_progress({"task": task, "event": "task_completed"})
            probes, probe_rows = [], []
            for name, frame in _hud_probes(fixed_size):
                prediction, label, _ = _readout(reward, frame, score_threshold)
                classification = _classification(prediction, label,
                    event_threshold=config.parameters.event_confidence_threshold,
                    score_threshold=score_threshold)
                probe_rows.append(classification)
                probes.append({"name": name, "label": label, "prediction": prediction,
                               **classification})
    finally:
        reward.train(mode)
    return {"kind": "frozen_train_rgb_reward_audit", "status": "completed", "split": "train",
        "created_utc": datetime.now(UTC).isoformat(), "seconds": time.monotonic() - started,
        "suite_version": SUITE_VERSION, "scoring_version": SCORING_VERSION,
        "updates": 0, "policy_decisions": 0, "rgb_saved": False, "corpus_saved": False,
        "config": {"tasks": list(tasks), "levels_per_task": levels_per_task,
            "level_bank_seed": level_bank_seed, "seed": seed, "horizon": horizon, "gamma": gamma,
            "score_confidence_threshold": score_threshold, "difficulty": 1.0,
            "teacher_image_size": fixed_size or 96},
        "candidate_composition": reward_composition_identity(config, gamma=gamma, horizon=horizon,
            score_confidence_threshold=score_threshold),
        "episodes": len(records), "episodes_valid_for_candidate_learning": sum(
            row["valid_for_learning"] for row in records),
        "trajectory_readouts": _summarize_readouts(all_readouts), "tasks": task_reports,
        "hud_probe_readouts": _summarize_readouts(probe_rows), "hud_probes": probes,
        "hud_probe_scope": {"sizes": [fixed_size] if fixed_size else [64, 96, 128],
            "train_styles": "skipped_unsupported_by_fixed_size_legacy_recognizer"
                            if fixed_size else "all_train_themes_and_layouts"},
        "peak_retained_rgb_payload_bytes": peak_rgb,
        "memory_scope": "one game's corpus RGB payload; not process RSS or whole Mac RAM",
        "interpretation": "TRAIN teacher traces and public TRAIN HUD vocabulary only; "
            "not policy improvement, reward calibration or generalization to unknown games. "
            "Completed means audit finished; inspect recognition errors and invalid episodes.",
        "records": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--levels-per-task", type=int, default=2)
    parser.add_argument("--level-bank-seed", type=int, default=20261008)
    parser.add_argument("--seed", type=int, default=8176)
    parser.add_argument("--goal-choice-utility", choices=("completion_only",))
    args = parser.parse_args(argv)
    config = replace(SequenceRunConfig.load(args.config), device="cpu")
    parameters = OutcomePotentialConfig(goal_choice_utility=args.goal_choice_utility)
    RewardCompositionConfig(OUTCOME_POTENTIAL_VERSION, parameters).validate_context(
        gamma=config.gamma, horizon=config.max_episode_steps, tasks=args.tasks,
        score_confidence_threshold=config.reward_confidence_threshold)
    checkpoint, output = args.checkpoint.resolve(strict=True), args.output.resolve()
    source = next((parent.parent for parent in checkpoint.parents
                   if parent.name in {"checkpoints", "checkpoints_by_epoch"}), checkpoint)
    if (output == source or source in output.parents or output in checkpoint.parents
            or args.output.is_symlink()):
        raise ValueError("audit output must be a new directory outside the source run")
    output.mkdir(parents=True, exist_ok=False)
    paths = [path for path in sorted(checkpoint.rglob("*")) if path.is_file()]
    before = {str(path.relative_to(checkpoint)): file_sha256(path) for path in paths}
    atomic_json(output / "metadata.json", {"checkpoint": str(checkpoint), "updates": 0,
                "checkpoint_files_sha256_before": before, **provenance()})
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.torch_threads)
        agent, optimizer, reward, state = load_sequence_checkpoint(
            checkpoint, config, evaluation_only=True, load_optimizer=False)
        if optimizer is not None:
            raise RuntimeError("reward audit must not load an optimizer")
        revision = state["weights_revision"]
        del agent, optimizer, state
        gc.collect()
        reward_hash = tensor_hash(reward)

        def progress(event):
            atomic_json(output / "status.json", {"status": "running", **event})
            print(f"{event['task']}: {event['event']}", flush=True)

        report = audit_reward(reward, tasks=args.tasks, levels_per_task=args.levels_per_task,
            level_bank_seed=args.level_bank_seed, seed=args.seed, horizon=config.max_episode_steps,
            gamma=config.gamma, score_threshold=config.reward_confidence_threshold,
            parameters=parameters, on_progress=progress)
        after = {str(path.relative_to(checkpoint)): file_sha256(path) for path in paths}
        if before != after or tensor_hash(reward) != reward_hash:
            raise RuntimeError("frozen checkpoint or reward tensors changed during audit")
        report.update(reward_sha256=reward_hash, weights_revision=revision,
                      checkpoint_files_unchanged=True, reward_tensors_unchanged=True)
        atomic_json(output / "audit.json", report)
        atomic_json(output / "status.json", {"status": "completed", "updates": 0,
            "episodes": report["episodes"], "episodes_valid_for_candidate_learning":
                report["episodes_valid_for_candidate_learning"]})
        return 0
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "interrupted"
            if isinstance(error, KeyboardInterrupt) else "failed", "updates": 0,
            "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        torch.set_num_threads(threads)


if __name__ == "__main__":
    raise SystemExit(main())
