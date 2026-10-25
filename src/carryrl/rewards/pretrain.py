"""Reward-first fitting and an explicit gate for supported screen presentations."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from carryrl.rewards.visual import (
    SCREEN_REWARD_VERSION,
    ScreenVisualReward,
    draw_visible_hud,
    screen_reward_readout,
    train_visual_reward,
    verify_visible_outcome,
)

REQUIRED_COMBINATIONS = tuple(
    f"{score}:{status}" for score in range(6) for status in ("Play", "Success", "Failed")
)
GATED_PARTITIONS = ("familiar_validation", "unseen_validation", "canonical_validation")


def _synthetic_glyph_dataset(config, themes, *, canonical_sizes=(), seed=0):
    """Render the full taught glyph vocabulary; targets are read back from RGB.

    These are explicitly synthetic ordinary UI examples, not played episodes.
    Scenery RNG differs between train/validation while familiar glyph identities
    necessarily recur. Held-out themes remain absent from fitting.
    """
    from carryrl.envs.presentation import PresentationProfile, present_frame

    readouts, groups, tasks = [], [], []
    labels = {name: [] for name in ("score", "success", "failure")}
    rng = np.random.default_rng(seed)
    specifications = [
        (size, theme, layout) for size in config.native_sizes
        for theme in themes for layout in config.hud_layouts
    ] + [(size, "canonical", "top") for size in canonical_sizes]
    for size, theme, layout in specifications:
        for score in range(6):
            for status in ("Play", "Success", "Failed"):
                frame = rng.integers(70, 180, (size, size, 3), dtype=np.uint8)
                draw_visible_hud(frame, score, status)
                if theme != "canonical":
                    frame = present_frame(frame, PresentationProfile(
                        size, theme, layout, texture_seed=seed,
                        textures=config.textures, canvas_padding=config.canvas_padding,
                    ))
                observed, readout = verify_visible_outcome(frame), screen_reward_readout(frame)
                if not observed["known"] or readout is None:
                    raise RuntimeError("Synthetic ordinary HUD has unverifiable RGB glyphs")
                readouts.append(readout)
                for name in labels:
                    labels[name].append(float(observed[name]))
                groups.append(f"{size}:{theme}:{layout}")
                tasks.append("synthetic_hud")
    if not readouts:
        raise ValueError("No synthetic glyph profiles selected")
    return (np.stack(readouts), {name: np.asarray(values, np.float32)
                                for name, values in labels.items()}, groups, tasks, [])


def _merge_datasets(*datasets):
    return (
        np.concatenate([dataset[0] for dataset in datasets]),
        {name: np.concatenate([dataset[1][name] for dataset in datasets])
         for name in ("score", "success", "failure")},
        [group for dataset in datasets for group in dataset[2]],
        [task for dataset in datasets for task in dataset[3]],
        sorted({episode for dataset in datasets for episode in dataset[4]}),
    )


@dataclass(frozen=True)
class RewardGate:
    """Configurable engineering tolerances, not established project requirements."""

    max_score_mae: float = 0.10
    max_score_error: float = 0.35
    min_event_recall: float = 1.0
    max_false_positive_rate: float = 0.0
    max_event_brier: float = 0.02
    max_spurious_reward_change: float = 0.10
    min_positive_frames: int = 1

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "min_positive_frames":
                if type(value) is not int or value < 1:
                    raise ValueError("min_positive_frames must be a positive integer")
            elif type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative tolerance")
        if not 0 <= self.min_event_recall <= 1 or self.max_false_positive_rate > 1:
            raise ValueError("Event rates must lie in [0,1]")


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _representatives(frames, labels, provenance) -> list:
    selected = {}
    for index, episode_id in enumerate(provenance["frame_episode_ids"]):
        values = tuple(float(labels[name][index]) for name in ("score", "success", "failure"))
        selected.setdefault((episode_id, values), index)
    return list(selected.values())


def _augment(episodes_by_size, themes, config):
    from carryrl.envs.presentation import PresentationProfile, present_frame

    readouts, labels, groups, tasks = [], {name: [] for name in (
        "score", "success", "failure"
    )}, [], []
    episode_ids = set()
    for size, (frames, canonical_labels, provenance) in episodes_by_size.items():
        episodes = {episode["episode_id"]: episode for episode in provenance["episodes"]}
        for index in _representatives(frames, canonical_labels, provenance):
            episode_id = provenance["frame_episode_ids"][index]
            episode_ids.add(f"{size}:{episode_id}")
            for theme in themes:
                for layout in config.hud_layouts:
                    profile = PresentationProfile(
                        size, theme, layout, texture_seed=index + config.seed,
                        textures=config.textures, canvas_padding=config.canvas_padding,
                    )
                    frame = present_frame(frames[index], profile)
                    # Labels are reverified after presentation. Canonical labels
                    # are used only to select representative visited states.
                    observed = verify_visible_outcome(frame)
                    readout = screen_reward_readout(frame)
                    if not observed["known"] or readout is None:
                        raise RuntimeError(
                            "Supported reward screen could not be independently read"
                        )
                    readouts.append(readout)
                    for name in labels:
                        labels[name].append(float(observed[name]))
                    groups.append(f"{size}:{theme}:{layout}")
                    tasks.append(episodes[episode_id]["task"])
    if not readouts:
        raise ValueError("No reward examples for the selected presentation styles")
    arrays = {name: np.asarray(value, np.float32) for name, value in labels.items()}
    return np.stack(readouts), arrays, groups, tasks, sorted(episode_ids)


def _deduplicate(readouts, labels):
    unique = {}
    for index, frame in enumerate(readouts):
        key = frame.tobytes()
        if key in unique:
            previous = unique[key]
            if any(labels[name][index] != labels[name][previous] for name in labels):
                raise ValueError("RGB-identical reward examples have conflicting labels")
        else:
            unique[key] = index
    indices = np.asarray(list(unique.values()))
    return readouts[indices], {name: values[indices] for name, values in labels.items()}


@torch.no_grad()
def _predictions(model, readouts):
    pieces = []
    device = next(model.parameters()).device
    for frames in np.array_split(readouts, max(1, int(np.ceil(len(readouts) / 128)))):
        tensors = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(device) / 255
        result = model(tensors)
        pieces.append(torch.stack((result["score"].clamp(0, 5),
                                   result["success_logits"].sigmoid(),
                                   result["failure_logits"].sigmoid()), dim=1).cpu().numpy())
    return np.concatenate(pieces)


def _metrics(predictions, labels, indices):
    indices = np.asarray(indices, dtype=int)
    result = {"frames": len(indices)}
    if not len(indices):
        return result
    expected = np.stack([labels[name][indices] for name in ("score", "success", "failure")], 1)
    selected = predictions[indices]
    errors = np.abs(selected[:, 0] - expected[:, 0])
    result.update(score_mae=float(errors.mean()), max_score_error=float(errors.max()))
    combination_counts = {combination: 0 for combination in REQUIRED_COMBINATIONS}
    for score, success, failure in expected:
        status = "Success" if success else "Failed" if failure else "Play"
        combination_counts[f"{int(score)}:{status}"] += 1
    result["combination_counts"] = combination_counts
    for column, event in enumerate(("success", "failure"), 1):
        truth = expected[:, column].astype(bool)
        decisions = selected[:, column] >= 0.5
        result[event] = {
            "positive_frames": int(truth.sum()), "negative_frames": int((~truth).sum()),
            "recall": float(decisions[truth].mean()) if truth.any() else None,
            "false_positive_rate": float(decisions[~truth].mean()) if (~truth).any() else None,
            "brier_score": float(((selected[:, column] - expected[:, column]) ** 2).mean()),
        }
    # Changes in appearance or scenery must not create progress when the visible
    # score/outcome is identical. Compare each observed semantic state across UIs.
    spurious = 0.0
    predicted_reward_state = selected[:, 0] + selected[:, 1] - selected[:, 2]
    for state in np.unique(expected, axis=0):
        same = (expected == state).all(axis=1)
        spurious = max(spurious, float(np.ptp(predicted_reward_state[same])))
    result["max_spurious_reward_change"] = spurious
    return result


def evaluate_screen_reward(model, dataset) -> dict:
    readouts, labels, groups, tasks, episode_ids = dataset
    predictions = _predictions(model, readouts)
    result = _metrics(predictions, labels, range(len(readouts)))
    result["episodes"] = len(episode_ids)
    result["by_profile"] = {
        group: _metrics(predictions, labels, np.flatnonzero(np.asarray(groups) == group))
        for group in sorted(set(groups))
    }
    result["by_task"] = {
        task: _metrics(predictions, labels, np.flatnonzero(np.asarray(tasks) == task))
        for task in sorted(set(tasks) - {"synthetic_hud"})
    }
    synthetic = np.asarray(tasks) == "synthetic_hud"
    result["synthetic_glyph_frames"] = int(synthetic.sum())
    result["played_episode_frames"] = int((~synthetic).sum())
    return result


def check_reward_gate(partitions: dict, gate: RewardGate | None = None,
                      expected_profiles: dict | None = None) -> dict:
    gate = gate or RewardGate()
    failures = []
    for partition in GATED_PARTITIONS:
        metrics = partitions.get(partition, {})
        if not metrics.get("frames"):
            failures.append(f"{partition}: no held-out frames")
            continue
        profiles = metrics.get("by_profile", {})
        expected = (expected_profiles or {}).get(partition, ())
        for missing in sorted(set(expected) - set(profiles)):
            failures.append(f"{partition}/{missing}: missing profile coverage")
        if not profiles:
            failures.append(f"{partition}: no per-profile coverage")
        for name, item in (("all", metrics), *profiles.items()):
            context = f"{partition}/{name}"
            counts = item.get("combination_counts", {})
            for combination in REQUIRED_COMBINATIONS:
                count = counts.get(combination, 0)
                if type(count) is not int or count < 1:
                    failures.append(f"{context}: missing score/status combination {combination}")
            for key, limit in (("score_mae", gate.max_score_mae),
                               ("max_score_error", gate.max_score_error),
                               ("max_spurious_reward_change", gate.max_spurious_reward_change)):
                value = item.get(key)
                if value is None or not np.isfinite(value) or value > limit:
                    failures.append(f"{context}: {key}={value} exceeds {limit}")
            for event in ("success", "failure"):
                values = item.get(event, {})
                if values.get("positive_frames", 0) < gate.min_positive_frames:
                    failures.append(f"{context}: missing {event} positives")
                if not values.get("negative_frames", 0):
                    failures.append(f"{context}: missing {event} negatives")
                for key, limit, lower in (("recall", gate.min_event_recall, True),
                                           ("false_positive_rate", gate.max_false_positive_rate,
                                            False), ("brier_score", gate.max_event_brier, False)):
                    value = values.get(key)
                    if value is None or not np.isfinite(value) or (
                        value < limit if lower else value > limit
                    ):
                        failures.append(f"{context}: {event} {key}={value} fails {limit}")
    return {"passed": not failures, "failures": failures, "tolerances": asdict(gate),
            "tolerances_source": "configurable experiment engineering choices"}


def pretrain_screen_reward(
    output: Path | str, *, tasks=None, episodes_per_task: int = 4,
    validation_episodes_per_task: int = 2, epochs: int = 40, seed: int = 0,
    device: str = "cpu", image_size: int = 96, max_steps: int = 80,
    presentation_config=None, gate: RewardGate | None = None,
    level_bank_seed: int = 0,
) -> tuple[ScreenVisualReward, dict]:
    """Split complete canonical episodes first, then augment only their RGB.

    Validation uses its separate complete level bank. Themes absent from fitting
    are assessed on validation episodes; no test levels are collected or fitted.
    Every target is independently verified from the presented pixels.
    """
    from carryrl.envs.atomic import collect_bootstrap_dataset
    from carryrl.envs.atomic_suite import TASKS
    from carryrl.envs.presentation import THEME_SPLITS, PresentationConfig

    output = Path(output)
    protected = ("reward_checkpoint.pt", "reward_report.json", "metadata.json", "metrics.jsonl")
    if output.exists() and any((output / name).exists() for name in protected):
        raise FileExistsError(
            "Choose a fresh reward output; existing experiment artifacts preserved"
        )
    if device not in {"cpu", "mps"}:
        raise ValueError("Reward device must be cpu or mps")
    config = (PresentationConfig(mode="varied") if presentation_config is None else
              PresentationConfig(**presentation_config) if isinstance(presentation_config, dict)
              else presentation_config)
    if not isinstance(config, PresentationConfig) or config.mode != "varied":
        raise ValueError("Screen reward pretraining requires varied PresentationConfig")
    selected = TASKS if tasks is None else tuple(tasks)
    train_themes = tuple(theme for theme in config.themes if theme in THEME_SPLITS["train"])
    unseen_themes = tuple(theme for theme in config.themes if theme not in train_themes)
    if not train_themes or not unseen_themes:
        raise ValueError("Reward fitting requires separate familiar and held-out theme banks")
    canonical = {}
    for split, count, offset in (("train", episodes_per_task, 0),
                                 ("validation", validation_episodes_per_task, 50000)):
        canonical[split] = {
            size: collect_bootstrap_dataset(
                episodes_per_task=count, seed=seed + offset, image_size=size, split=split,
                max_steps=max_steps, tasks=selected,
                level_bank_seed=level_bank_seed,
            ) for size in config.native_sizes
        }
    fitted = _augment(canonical["train"], train_themes, config)
    familiar = _augment(canonical["validation"], train_themes, config)
    unseen = _augment(canonical["validation"], unseen_themes, config)
    canonical_sizes = tuple(sorted(set(config.native_sizes) | {image_size}))
    fitted = _merge_datasets(fitted, _synthetic_glyph_dataset(
        config, train_themes, canonical_sizes=canonical_sizes, seed=seed + 200000,
    ))
    familiar = _merge_datasets(familiar, _synthetic_glyph_dataset(
        config, train_themes, seed=seed + 300000,
    ))
    unseen = _merge_datasets(unseen, _synthetic_glyph_dataset(
        config, unseen_themes, seed=seed + 400000,
    ))
    canonical_validation = _synthetic_glyph_dataset(
        config, (), canonical_sizes=canonical_sizes, seed=seed + 500000,
    )
    train_ids = set(fitted[4])
    validation_ids = set(familiar[4]) | set(unseen[4])
    if train_ids & validation_ids:
        raise RuntimeError("Reward train and validation episodes overlap")
    level_ids = {
        split: {str(episode["level_id"]) for _, _, provenance in batches.values()
                for episode in provenance["episodes"] if "level_id" in episode}
        for split, batches in canonical.items()
    }
    if level_ids["train"] & level_ids["validation"]:
        raise RuntimeError("Reward train and validation level identities overlap")
    readouts, labels = _deduplicate(fitted[0], fitted[1])
    torch.manual_seed(seed)
    model = ScreenVisualReward(image_size=image_size).to(device)
    fit = train_visual_reward(model, readouts, labels, epochs=epochs, seed=seed,
                              batch_size=64, learning_rate=0.001, min_samples_per_epoch=1024)
    partitions = {"train": evaluate_screen_reward(model, fitted),
                  "familiar_validation": evaluate_screen_reward(model, familiar),
                  "unseen_validation": evaluate_screen_reward(model, unseen),
                  "canonical_validation": evaluate_screen_reward(model, canonical_validation)}
    expected_profiles = {
        partition: [f"{size}:{theme}:{layout}" for size in config.native_sizes
                    for theme in themes for layout in config.hud_layouts]
        for partition, themes in (("familiar_validation", train_themes),
                                   ("unseen_validation", unseen_themes))
    }
    expected_profiles["canonical_validation"] = [
        f"{size}:canonical:top" for size in canonical_sizes
    ]
    report = {
        "model_version": SCREEN_REWARD_VERSION, "tasks": list(selected), "seed": seed,
        "level_bank_seed": level_bank_seed,
        "initialization": "random", "imported_checkpoints": [], "old_corpora": [],
        "image_size": image_size, "presentation_config": asdict(config),
        "train_themes": list(train_themes), "heldout_themes": list(unseen_themes),
        "supervision": "independent exact RGB verification of ordinary supported HUDs",
        "scope": "supervised localized HUD readout; not autonomous reward discovery",
        "inputs": "RGB only; border localization and text contrast/geometry normalization",
        "split": "complete episodes and disjoint level banks before style augmentation",
        "test_levels_used": False, "policy_demonstrations": False,
        "unique_training_readouts": len(readouts), "fit": fit, "partitions": partitions,
        "score_event_pixel_support": "separate score-line and status-line CNNs",
        "synthetic_glyph_coverage": {
            "source": "ordinary rendered RGB HUD; labels independently verified from RGB",
            "required_combinations": list(REQUIRED_COMBINATIONS),
            "training_frames": sum(task == "synthetic_hud" for task in fitted[3]),
            "validation_frames": sum(
                sum(task == "synthetic_hud" for task in dataset[3])
                for dataset in (familiar, unseen, canonical_validation)
            ),
            "heldout_themes": list(unseen_themes),
            "familiar_glyphs": "same taught glyph identities recur across separate scenes",
            "not_played_episodes": True,
        },
        "train_episode_ids": sorted(train_ids), "validation_episode_ids": sorted(validation_ids),
        "train_level_ids": sorted(level_ids["train"]),
        "validation_level_ids": sorted(level_ids["validation"]),
        "expected_profiles": expected_profiles,
        "gate": check_reward_gate(partitions, gate, expected_profiles),
        "state_sha256": _state_hash(model),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "reward_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary = output / "reward_checkpoint.pt.tmp"
    torch.save({"version": SCREEN_REWARD_VERSION, "image_size": image_size,
                "state": model.state_dict(), "report": report}, temporary)
    temporary.replace(output / "reward_checkpoint.pt")
    return model.eval().requires_grad_(False), report


def load_screen_reward_checkpoint(path: Path | str, *, device: str = "cpu",
                                  require_pass: bool = True) -> tuple[ScreenVisualReward, dict]:
    if device not in {"cpu", "mps"}:
        raise ValueError("Reward device must be cpu or mps")
    data = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(data, dict) or data.get("version") != SCREEN_REWARD_VERSION:
        raise ValueError("Unsupported screen reward checkpoint version")
    report = data.get("report")
    if not isinstance(report, dict) or report.get("model_version") != SCREEN_REWARD_VERSION:
        raise ValueError("Screen reward checkpoint has no versioned validation report")
    expected_profiles = report.get("expected_profiles", {})
    if not all(expected_profiles.get(name) for name in GATED_PARTITIONS):
        raise ValueError("Screen reward checkpoint has no required held-out profile catalogue")
    train_ids, validation_ids = report.get("train_episode_ids", []), report.get(
        "validation_episode_ids", []
    )
    if not train_ids or not validation_ids or set(train_ids) & set(validation_ids):
        raise ValueError("Screen reward checkpoint has missing or overlapping episode splits")
    if set(report.get("train_level_ids", [])) & set(report.get("validation_level_ids", [])):
        raise ValueError("Screen reward checkpoint has overlapping level splits")
    recomputed = check_reward_gate(
        report.get("partitions", {}), RewardGate(**report.get("gate", {}).get("tolerances", {})),
        expected_profiles,
    )
    if require_pass and (not report.get("gate", {}).get("passed") or not recomputed["passed"]):
        raise ValueError("Screen reward checkpoint failed the held-out validation gate")
    if data.get("image_size") != report.get("image_size"):
        raise ValueError("Screen reward checkpoint image-size metadata disagrees")
    model = ScreenVisualReward(image_size=data["image_size"]).to(device)
    model.load_state_dict(data["state"], strict=True)
    if _state_hash(model) != report.get("state_sha256"):
        raise ValueError("Screen reward checkpoint state hash does not match its report")
    return model.eval().requires_grad_(False), report
