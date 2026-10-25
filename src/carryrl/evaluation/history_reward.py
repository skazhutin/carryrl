"""Frozen auxiliary history-head diagnostics on an unchanged reference cohort."""

import hashlib
import json
import math
import random
from collections.abc import Mapping

import numpy as np
import torch

from carryrl.agent.actions import ACTION_FEATURE_SIZE

DEFAULT_CONTROLS = ("full", "no_scene", "no_actions", "constant_goal", "single_frame", "time_only")
_INPUT_NAMES = {"scene_features", "executed_previous_actions", "explicit_goal_features"}


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _tensor_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _classification(probability, target, valid, threshold):
    p, y = probability[valid], target[valid]
    predicted, positive = p >= threshold, y >= 0.5
    tp = int((predicted & positive).sum())
    fp = int((predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    tn = int((~predicted & ~positive).sum())
    clipped = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    return {
        "known": int(valid.sum()),
        "unknown": int((~valid).sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": _divide(tp, tp + fp),
        "recall": _divide(tp, tp + fn),
        "accuracy": _divide(tp + tn, len(y)),
        "bce": float((-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))).mean())
        if len(y)
        else None,
        "brier": float(np.square(p - y).mean()) if len(y) else None,
    }


def _validity(probability, valid, threshold):
    proposed = probability >= threshold
    known, unknown = int(valid.sum()), int((~valid).sum())
    return {
        "frames_or_cells": int(valid.size),
        "known": known,
        "unknown": unknown,
        "proposed_valid": int(proposed.sum()),
        "coverage": float(proposed.mean()),
        "accuracy": float((proposed == valid).mean()),
        "known_coverage": _divide(int((proposed & valid).sum()), known),
        "unknown_coverage": _divide(int((proposed & ~valid).sum()), unknown),
        "proposed_abstentions_known": int((~proposed & valid).sum()),
        "proposed_abstentions_unknown": int((~proposed & ~valid).sum()),
    }


def _summarize(rows, names, event_threshold, terminal_threshold, validity_threshold):
    def cat(name):
        return np.concatenate([row[name] for row in rows], axis=0)

    potential, target, valid = cat("potential"), cat("potential_target"), cat("potential_valid")
    errors = potential[valid] - target[valid]
    moving, agree, equal, drift = 0, 0, 0, 0.0
    directions = {name: {"pairs": 0, "correct": 0} for name in ("up", "down", "flat")}
    tolerance = 1e-6
    for row in rows:
        pair = row["potential_valid"][:-1] & row["potential_valid"][1:]
        actual = np.diff(row["potential_target"])[pair]
        predicted = np.diff(row["potential"])[pair]
        changes = np.abs(actual) > tolerance
        actual_direction = np.where(actual > tolerance, 1, np.where(actual < -tolerance, -1, 0))
        predicted_direction = np.where(
            predicted > tolerance, 1, np.where(predicted < -tolerance, -1, 0)
        )
        for name, value in (("up", 1), ("down", -1), ("flat", 0)):
            selected = actual_direction == value
            directions[name]["pairs"] += int(selected.sum())
            directions[name]["correct"] += int((predicted_direction[selected] == value).sum())
        moving += int(changes.sum())
        agree += int((actual_direction[changes] == predicted_direction[changes]).sum())
        equal += int((~changes).sum())
        drift += float(np.abs(predicted[~changes]).sum())
    event_probability, event_target, event_valid = (
        cat("events"),
        cat("event_target"),
        cat("event_valid"),
    )
    terminal_probability, terminal_target = cat("terminal"), cat("terminal_target")
    terminal_valid = cat("terminal_valid")
    final_probability = np.asarray([row["terminal"][-1] for row in rows])
    final_target = np.asarray([row["terminal_target"][-1] for row in rows])
    final_valid = np.asarray([row["terminal_valid"][-1] for row in rows])
    return {
        "episodes": len(rows),
        "frames": int(len(potential)),
        "events": {
            name: _classification(
                event_probability[:, i], event_target[:, i], event_valid[:, i], event_threshold
            )
            for i, name in enumerate(names)
        },
        "potential": {
            "known": int(valid.sum()),
            "unknown": int((~valid).sum()),
            "mae": float(np.abs(errors).mean()) if len(errors) else None,
            "rmse": float(np.sqrt(np.square(errors).mean())) if len(errors) else None,
            "changing_adjacent_pairs": moving,
            "sign_agreement": _divide(agree, moving),
            "direction_tolerance": tolerance,
            "direction": {
                name: {
                    "pairs": values["pairs"],
                    "agreement": _divide(values["correct"], values["pairs"]),
                }
                for name, values in directions.items()
            },
            "equal_reference_adjacent_pairs": equal,
            "equal_reference_absolute_drift": _divide(drift, equal),
        },
        "terminal": {
            "all_frames": _classification(
                terminal_probability, terminal_target, terminal_valid, terminal_threshold
            ),
            "final_frames": _classification(
                final_probability, final_target, final_valid, terminal_threshold
            ),
        },
        "validity": {
            "potential": _validity(cat("potential_valid_probability"), valid, validity_threshold),
            "events": {
                name: _validity(
                    cat("event_valid_probability")[:, i], event_valid[:, i], validity_threshold
                )
                for i, name in enumerate(names)
            },
            "terminal": _validity(
                cat("terminal_valid_probability"), terminal_valid, validity_threshold
            ),
        },
    }


def _macro(summaries):
    """Average defined per-unit rates/errors; never report averaged counts as counts."""
    count_names = {
        "episodes",
        "frames",
        "known",
        "unknown",
        "tp",
        "fp",
        "fn",
        "tn",
        "frames_or_cells",
        "proposed_valid",
        "proposed_abstentions_known",
        "proposed_abstentions_unknown",
        "changing_adjacent_pairs",
        "equal_reference_adjacent_pairs",
        "pairs",
        "direction_tolerance",
    }

    def recurse(values):
        result = {}
        for key in values[0]:
            if key in count_names:
                continue
            items = [value[key] for value in values]
            if isinstance(items[0], dict):
                result[key] = recurse(items)
            else:
                defined = [value for value in items if value is not None]
                result[key] = {
                    "mean": float(np.mean(defined)) if defined else None,
                    "defined_units": len(defined),
                }
        return result

    return {"units": len(summaries), "metrics": recurse(summaries)}


def _control_inputs(original, control):
    inputs = {name: value.detach().clone() for name, value in original.items()}
    scene, actions, goal = (
        inputs[name]
        for name in ("scene_features", "executed_previous_actions", "explicit_goal_features")
    )
    if control in ("no_scene", "time_only"):
        scene.zero_()
    if control in ("no_actions", "time_only"):
        actions.zero_()
    if control in ("constant_goal", "time_only"):
        goal.fill_(1 / math.sqrt(goal.shape[-1]))
    if control == "single_frame":
        steps = scene.shape[1]
        inputs["scene_features"] = scene.reshape(steps, 1, scene.shape[-1])
        inputs["executed_previous_actions"] = torch.zeros_like(actions).reshape(
            steps, 1, actions.shape[-1]
        )
        inputs["explicit_goal_features"] = goal.expand(steps, -1).clone()
    return inputs


def _prediction_row(outputs, targets, steps, names, single_frame):
    result = {}
    mapping = (
        ("potential", "potential", False),
        ("events", "event_logits", True),
        ("terminal", "terminal_success_logits", True),
        ("potential_valid_probability", "potential_valid_logits", True),
        ("event_valid_probability", "event_valid_logits", True),
        ("terminal_valid_probability", "terminal_valid_logits", True),
    )
    for name, source, logits in mapping:
        tensor = outputs[source]
        event = source in ("event_logits", "event_valid_logits")
        expected = ((steps, 1) if single_frame else (1, steps)) + ((len(names),) if event else ())
        if tensor.shape != expected or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid output {source}: expected finite {expected}")
        value = tensor.sigmoid() if logits else tensor
        if ((value < 0) | (value > 1)).any():
            raise ValueError(f"Invalid probability output {source}")
        result[name] = (
            value.detach().cpu().numpy().reshape((steps, len(names)) if event else (steps,))
        )
    for name, target_name, valid_name in (
        ("potential", "potential", "potential_valid"),
        ("event", "events", "event_valid"),
        ("terminal", "terminal_success", "terminal_success_valid"),
    ):
        target = np.asarray(targets[target_name])
        valid = np.asarray(targets[valid_name])
        shape = (steps, len(names)) if name == "event" else (steps,)
        if target.shape != shape or valid.shape != shape or valid.dtype != np.bool_:
            raise ValueError(f"Invalid reference shape or validity mask: {target_name}")
        if (
            not np.isfinite(target[valid]).all()
            or ((target[valid] < 0) | (target[valid] > 1)).any()
        ):
            raise ValueError(f"Invalid known reference values: {target_name}")
        if name != "potential" and not np.isin(target[valid], [0, 1]).all():
            raise ValueError(f"Classification reference must be binary: {target_name}")
        result[f"{name}_target"] = target.copy()
        result[f"{name}_valid"] = valid.copy()
    return result


def evaluate_history_reward(
    head,
    corpus,
    feature_set,
    *,
    split="validation",
    controls=DEFAULT_CONTROLS,
    event_threshold=0.5,
    terminal_threshold=0.5,
    validity_threshold=0.5,
    on_progress=None,
):
    """Measure reference-valid predictions; ablations are OOD sensitivity probes.

    No policy outcomes or calibrated confidence are claimed. The unchanged masks
    determine all main denominators, even when the head proposes abstention.
    """
    if split not in ("train", "validation"):
        raise ValueError("Only train or validation is permitted; test is held out")
    controls = tuple(controls)
    if not controls or len(set(controls)) != len(controls) or set(controls) - set(DEFAULT_CONTROLS):
        raise ValueError("Unknown, duplicate or empty controls")
    for threshold in (event_threshold, terminal_threshold, validity_threshold):
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 < threshold < 1
        ):
            raise ValueError("Thresholds must be finite in (0,1)")
    names = tuple(corpus.event_names)
    if tuple(head.event_names) != names:
        raise ValueError("Head event vocabulary/order differs from corpus")
    identity = json.loads(json.dumps(feature_set.identity, allow_nan=False))
    corpus_hash = getattr(corpus, "corpus_sha256", None)
    source_hash = identity.get("corpus_sha256")
    if identity.get("event_names") != list(names):
        raise ValueError("Feature event vocabulary/order differs from corpus")
    if not corpus_hash or source_hash != corpus_hash:
        raise ValueError("Feature source corpus identity differs from corpus")
    if (
        head.scene_feature_size != feature_set.feature_width
        or head.goal_feature_size != feature_set.goal_width
    ):
        raise ValueError("Head feature dimensions differ from feature set")
    groups = {"train": set(), "validation": set()}
    for episode in corpus.episodes:
        if episode["split"] not in groups:
            raise ValueError("Corpus has unknown/test split")
        group = episode.get("runtime_group")
        if not isinstance(group, str) or not group:
            raise ValueError("Corpus is missing runtime group identity")
        groups[episode["split"]].add(group)
    if groups["train"] & groups["validation"]:
        raise ValueError("Train/validation runtime group overlap")
    indices = [i for i, episode in enumerate(corpus.episodes) if episode["split"] == split]
    if not indices:
        raise ValueError("Selected split has no episodes")
    cohort = [
        {
            "episode_index": i,
            "task": corpus.episodes[i]["task"],
            "strategy": corpus.episodes[i]["strategy"],
            "runtime_group": corpus.episodes[i]["runtime_group"],
        }
        for i in indices
    ]
    source = {
        "corpus_sha256": corpus_hash,
        "feature_identity": identity,
        "event_names": list(names),
    }
    before = _tensor_hash(head)
    modes = [(module, module.training) for module in head.modules()]
    gradients = [(parameter, parameter.requires_grad) for parameter in head.parameters()]
    random_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    device = next(head.parameters(), torch.empty(0)).device
    accelerator_state = None
    if device.type == "mps":
        accelerator_state = torch.mps.get_rng_state()
    elif device.type == "cuda":
        accelerator_state = torch.cuda.get_rng_state(device)
    collected = {control: [] for control in controls}
    try:
        head.eval()
        for parameter, _ in gradients:
            parameter.requires_grad_(False)
        with torch.no_grad():
            for completed, index in enumerate(indices, 1):
                original = feature_set.model_inputs(corpus, index, device=device)
                if not isinstance(original, Mapping) or set(original) != _INPUT_NAMES:
                    raise ValueError(
                        "Feature inputs must contain only scene, own actions and explicit goal"
                    )
                if any(
                    not isinstance(value, torch.Tensor)
                    or not value.is_floating_point()
                    or not torch.isfinite(value).all()
                    for value in original.values()
                ):
                    raise ValueError("Feature inputs must be finite floating tensors")
                snapshots = {name: tensor.detach().clone() for name, tensor in original.items()}
                targets = corpus.targets(index)
                if tuple(targets.get("event_names", ())) != names:
                    raise ValueError("Target event vocabulary/order differs from corpus")
                steps = len(targets["potential"])
                if steps < 1:
                    raise ValueError("Reference episode must contain at least one frame")
                if original["scene_features"].shape != (1, steps, feature_set.feature_width):
                    raise ValueError("Feature sequence differs from reference cohort")
                if original["executed_previous_actions"].shape != (1, steps, ACTION_FEATURE_SIZE):
                    raise ValueError("Own action sequence differs from reference cohort")
                if original["explicit_goal_features"].shape != (1, feature_set.goal_width):
                    raise ValueError("Goal feature shape differs")
                if not original["explicit_goal_features"].count_nonzero():
                    raise ValueError("Explicit goal feature must be nonzero")
                for control in controls:
                    predictions = head(**_control_inputs(original, control))
                    collected[control].append(
                        _prediction_row(
                            predictions, targets, steps, names, control == "single_frame"
                        )
                    )
                if any(
                    not torch.equal(original[name], tensor) for name, tensor in snapshots.items()
                ):
                    raise RuntimeError("Evaluation modified feature inputs")
                if on_progress is not None:
                    on_progress({"completed": completed, "total": len(indices), "split": split})
    finally:
        for parameter, required in gradients:
            parameter.requires_grad_(required)
        for module, mode in modes:
            module.training = mode
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if device.type == "mps":
            torch.mps.set_rng_state(accelerator_state)
        elif device.type == "cuda":
            torch.cuda.set_rng_state(accelerator_state, device)
        if _tensor_hash(head) != before:
            raise RuntimeError("Frozen evaluation modified model tensors")
    reports = {}
    for control, rows in collected.items():

        def summarize(selected):
            return _summarize(
                selected, names, event_threshold, terminal_threshold, validity_threshold
            )

        per_game = {
            game: summarize(
                [row for row, item in zip(rows, cohort, strict=True) if item["task"] == game]
            )
            for game in sorted({c["task"] for c in cohort})
        }
        per_strategy = {
            strategy: summarize(
                [
                    row
                    for row, item in zip(rows, cohort, strict=True)
                    if item["strategy"] == strategy
                ]
            )
            for strategy in sorted({c["strategy"] for c in cohort})
        }
        reports[control] = {
            "cohort_sha256": _json_hash(cohort),
            "global": summarize(rows),
            "per_game": per_game,
            "per_strategy": per_strategy,
            "macro_episodes": _macro([summarize([row]) for row in rows]),
            "macro_games": _macro(list(per_game.values())),
        }
    return {
        "format": "carryrl.history_reward_evaluation.v1",
        "split": split,
        "source_identity": source,
        "source_sha256": _json_hash(source),
        "head_sha256_before": before,
        "head_sha256_after": _tensor_hash(head),
        "cohort": cohort,
        "cohort_sha256": _json_hash(cohort),
        "thresholds": {
            "events": event_threshold,
            "terminal": terminal_threshold,
            "validity": validity_threshold,
        },
        "controls": reports,
        "frozen": True,
        "updates": 0,
        "ppo_eligible": False,
        "test_levels_played": False,
        "test_outcomes_read": False,
        "interpretation": "Auxiliary reference diagnostics. Controls are deliberate OOD "
        "sensitivity probes, not causal proof, modality necessity or alternate-goal quality. "
        "Validity outputs are proposed coverage, not calibrated confidence. Main metrics use "
        "unchanged reference-valid masks; global metrics are frame-weighted.",
    }
