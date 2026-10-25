"""Separate frozen presentation/instruction diagnostic; never a training quality curve."""

import argparse
import hashlib
import importlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from carryrl.envs.atomic import AtomicEnv, _identity, atomic_level_banks, atomic_semantic_level_id
from carryrl.envs.presentation import PresentationConfig
from carryrl.evaluation.dynamics import (
    REACTION_SPEED_PROFILES,
    ReactionSpeedEvaluationEnv,
    reaction_speed_condition,
    reaction_speed_level,
    validate_reaction_speed_validation_bank,
)
from carryrl.evaluation.paired import episode_descriptor, paired_outcomes
from carryrl.training.run import atomic_json
from carryrl.training.sequence import (
    evaluate_sequence,
    load_sequence_checkpoint,
    sequence_evaluation_protocol,
    tensor_hash,
)
from carryrl.training.sequence_config import SequenceRunConfig


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _checkpoint_identity(checkpoint):
    hashes = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            hashes[str(path.relative_to(checkpoint))] = digest.hexdigest()
    if not hashes:
        raise ValueError("checkpoint directory has no files")
    return {"path": str(checkpoint), "files_sha256": hashes}


def _environment_identity():
    """Keep native-mechanics fixes distinguishable in saved diagnostic matrices."""
    modules = (
        "carryrl.envs.atomic", "carryrl.envs.atomic_motion",
        "carryrl.envs.atomic_interaction", "carryrl.envs.atomic_combat",
        "carryrl.envs.presentation", "carryrl.rewards.visual",
    )
    return {
        name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
        for name in modules
    }


def presentation_profiles(split="validation"):
    if split not in {"validation", "test"}:
        raise ValueError("matrix requires a held-out validation or test split")
    if split == "test":
        profiles = {"canonical96": PresentationConfig(
            mode="canonical", native_sizes=(96,), themes=("slate",), hud_layouts=("top",))}
        for size, layout in ((64, "bottom"), (96, "left"), (128, "top")):
            for theme in ("ice", "ember"):
                profiles[f"varied{size}_{theme}_{layout}"] = PresentationConfig(
                    mode="varied", native_sizes=(size,), themes=(theme,),
                    hud_layouts=(layout,), style_split="test")
        return profiles
    return {
        "canonical96": PresentationConfig(mode="canonical", native_sizes=(96,),
                                           themes=("slate",), hud_layouts=("top",)),
        "varied96_slate_left": PresentationConfig(mode="varied", native_sizes=(96,),
                                                 themes=("slate",), hud_layouts=("left",),
                                                 style_split="train"),
        "varied64_violet_bottom": PresentationConfig(
            mode="varied", native_sizes=(64,), themes=("violet",), hud_layouts=("bottom",),
            style_split="validation"),
        "varied128_violet_top": PresentationConfig(
            mode="varied", native_sizes=(128,), themes=("violet",), hud_layouts=("top",),
            style_split="validation"),
    }


def appearance_profiles(split="validation"):
    """Hold native size/layout/texture fixed; themes remain declared families."""
    if split != "validation":
        raise ValueError("appearance96 accepts validation only; test remains reserved")
    profiles = {"canonical96": PresentationConfig(mode="canonical", native_sizes=(96,),
        themes=("slate",), hud_layouts=("top",), textures=False)}
    for theme, style_split in (("slate", "train"), ("sand", "train"),
                               ("violet", "validation")):
        profiles[f"theme96_{theme}_left_plain"] = PresentationConfig(
            mode="varied", native_sizes=(96,), themes=(theme,), hud_layouts=("left",),
            textures=False, canvas_padding=8, seed=0, style_split=style_split)
    return profiles


def _profiles(split, matrix_kind):
    if matrix_kind == "composite":
        return presentation_profiles(split)
    if matrix_kind == "appearance96":
        return appearance_profiles(split)
    if matrix_kind == "reaction_dynamics":
        if split != "validation":
            raise ValueError("reaction dynamics accepts validation only; test remains reserved")
        return {name: PresentationConfig(mode="canonical", native_sizes=(96,),
                themes=("slate",), hud_layouts=("top",), textures=False)
                for name in REACTION_SPEED_PROFILES}
    raise ValueError("unknown robustness matrix kind")


def build_matrix(config, *, split="validation", episodes_per_task=2, all_levels=False,
                 matrix_kind="composite"):
    profiles = _profiles(split, matrix_kind)
    if matrix_kind == "reaction_dynamics":
        if "reaction" not in config.tasks:
            raise ValueError("reaction dynamics requires the reaction task")
        validate_reaction_speed_validation_bank(config.level_bank_seed)
        config = replace(config, tasks=("reaction",), curriculum_difficulty=1.0,
                         evaluation_difficulty=1.0,
                         task_episode_multipliers={task: count for task, count in
                             config.task_episode_multipliers.items() if task == "reaction"},
                         episode_instructions={"reaction": config.episode_instructions.get(
                             "reaction", ())})
    if config.agent.text_encoder == "none":
        raise ValueError("instruction matrix requires a checkpoint with an enabled text encoder")
    instructions = {}
    for task in config.tasks:
        variants = config.episode_instructions.get(task, ())
        if len(variants) < 2 or any(not isinstance(value, str) or not value.strip()
                                    for value in variants[:2]):
            raise ValueError(f"{task} requires RU/EN instruction variants at positions 0/1")
        instructions[task] = variants[:2]
    base = replace(config, eval_episodes_per_task=episodes_per_task,
                   eval_all_levels=all_levels, eval_stochastic=False)
    strata = []
    for profile, visual in profiles.items():
        for position, language in enumerate(("ru", "en")):
            selected = replace(base, env_image_size=96, visual_config=visual,
                               episode_instructions={task: (values[position],)
                                                     for task, values in instructions.items()})
            protocol = sequence_evaluation_protocol(selected, split=split)
            spec = {"id": f"{profile}_{language}", "profile": profile,
                            "language": language, "instruction_variant_position": position,
                            "presentation": asdict(visual), "protocol": protocol,
                            "protocol_sha256": _hash(protocol)}
            if matrix_kind == "reaction_dynamics":
                condition = reaction_speed_condition(profile)
                spec.update(base_protocol=protocol, base_protocol_sha256=_hash(protocol),
                            runtime_intervention=condition)
                spec["protocol"] = {**protocol, "runtime_intervention": condition}
                spec["protocol_sha256"] = _hash(spec["protocol"])
            strata.append((spec, selected))
    return strata


def _cohorts(records):
    result = {}
    for row in records:
        key = f'{row["policy"]}/{row["task"]}'
        cohort = result.setdefault(key, {"episodes": [], "distinct_level_ids": [],
                                        "distinct_semantic_level_ids": [],
                                        "distinct_runtime_fingerprints": []})
        cohort["episodes"].append({name: row.get(name) for name in (
            "episode_index", "seed", "level_index", "level_id", "semantic_level_id",
            "runtime_fingerprint", "difficulty", "action_seed", "initial_frame_sha256")})
    for cohort in result.values():
        for name in ("level_id", "semantic_level_id", "runtime_fingerprint"):
            plural = {"level_id": "distinct_level_ids",
                      "semantic_level_id": "distinct_semantic_level_ids",
                      "runtime_fingerprint": "distinct_runtime_fingerprints"}[name]
            cohort[plural] = sorted({row[name] for row in cohort["episodes"]
                                     if row[name] is not None})
            cohort[f"{plural}_count"] = len(cohort[plural])
        cohort["episodes_started"] = len(cohort["episodes"])
    return result


def _comparisons(reports):
    baseline = reports["canonical96_ru"]["mechanical_cohorts"]
    comparisons = {}
    for stratum, report in reports.items():
        comparisons[stratum] = {}
        for key, cohort in report["mechanical_cohorts"].items():
            left, right = baseline[key]["episodes"], cohort["episodes"]
            fields = ("seed", "action_seed", "level_index", "level_id", "semantic_level_id",
                      "runtime_fingerprint", "initial_frame_sha256")
            comparisons[stratum][key] = {
                "baseline": "canonical96_ru", "episodes_compared": min(len(left), len(right)),
                "equal_episode_counts": len(left) == len(right),
                "matches": {field: sum(a[field] == b[field] and a[field] is not None
                                       for a, b in zip(left, right, strict=False))
                            for field in fields},
            }
    return comparisons


def _theme_contrasts(reports):
    """Only appearance/initial pixels may vary; preserve their source identities."""
    result = {}
    for language in ("ru", "en"):
        left_name = f"theme96_slate_left_plain_{language}"
        left_report = reports[left_name]
        left = [row for row in left_report["records"] if row["policy"] == "frozen_deterministic"]
        for theme in ("sand", "violet"):
            right_name = f"theme96_{theme}_left_plain_{language}"
            right_report = reports[right_name]
            right = [row for row in right_report["records"]
                     if row["policy"] == "frozen_deterministic"]
            if len(left) != len(right):
                raise ValueError("appearance contrast episode counts differ")
            projected = []
            for before, after in zip(left, right, strict=True):
                a, b = episode_descriptor(before), episode_descriptor(after)
                for key in ("presentation", "initial_frame_sha256"):
                    a.pop(key)
                    b.pop(key)
                if (a != b or before.get("level_bank_seed") != after.get("level_bank_seed")):
                    raise ValueError("appearance contrast changes mechanics/instruction/cohort")
                # These two differences ARE the declared intervention, so they
                # cannot define equality for this explicitly conditional table.
                projected.append({**after, "presentation": before["presentation"],
                                  "initial_frame_sha256": before["initial_frame_sha256"]})
            counts = paired_outcomes(left, projected)
            result[f"slate_to_{theme}_{language}"] = {
                "axis": "theme_family_colors_and_hud_font", "left_stratum": left_name,
                "right_stratum": right_name,
                "left_protocol_sha256": left_report["protocol_sha256"],
                "right_protocol_sha256": right_report["protocol_sha256"],
                "varied_descriptor_fields": ["presentation", "initial_frame_sha256"],
                "mechanical_matching_verified": True, "outcomes": counts,
                "interpretation": "Frozen sensitivity to theme at matched native96/layout/"
                    "texture/instruction. No training benefit or causal language claim.",
            }
    return result


def _verify_reaction_records(records, config, condition):
    bank = atomic_level_banks("reaction", config.level_bank_seed)["validation"]
    origins = {_identity(level): (index, level) for index, level in enumerate(bank)}
    for row in records:
        if (row["task"] != "reaction" or row["level_id"] not in origins
                or row["difficulty"] != 1.0 or row["level_bank_seed"] != config.level_bank_seed
                or row["presentation"] is not None):
            raise ValueError("reaction matrix changed origin bank, task or presentation")
        index, origin = origins[row["level_id"]]
        level = reaction_speed_level(origin, condition["speed_magnitude"])
        runtime = atomic_semantic_level_id("reaction", level)
        if (row["semantic_level_id"] != runtime or row["runtime_fingerprint"] != runtime
                or row["level_index"] not in (None, index)):
            raise ValueError("reaction matrix did not execute the declared runtime")
        env = AtomicEnv("reaction", split="validation", image_size=96, difficulty=1.0,
                        level_bank_seed=config.level_bank_seed)
        try:
            initial, _ = env.reset(seed=row["seed"], options={"level_index": index})
            if hashlib.sha256(initial.tobytes()).hexdigest() != row["initial_frame_sha256"]:
                raise ValueError("reaction speed intervention changed initial visible positions")
        finally:
            env.close()


def run_matrix(checkpoint, config, output, *, split="validation", episodes_per_task=2,
               all_levels=False, matrix_kind="composite"):
    """Declare the full matrix first, then load one frozen checkpoint and evaluate it."""
    _profiles(split, matrix_kind)  # Reject invalid split before output/model allocation.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reports, current = {}, None
    try:
        strata = build_matrix(config, split=split, episodes_per_task=episodes_per_task,
                              all_levels=all_levels, matrix_kind=matrix_kind)
        identity = _checkpoint_identity(Path(checkpoint).resolve(strict=True))
        declaration = {
            "format": "carryrl.robustness_matrix.v1", "checkpoint": identity,
            "environment_source_sha256": _environment_identity(),
            "strata": [spec for spec, _ in strata], "updates": 0,
            "coverage_mode": "full_bank" if all_levels else "seeded_diagnostic_subset",
            "interpretation": "Seeded sampling can repeat levels; inspect distinct coverage. "
            "Native sizes can change runtime mechanics; matching bank IDs does not imply "
            "identical trajectories. RU/EN variants express the same goal and do not prove "
            "causal language understanding. This is separate from ordinary quality curves.",
        }
        if matrix_kind != "composite":
            declaration["matrix_kind"] = matrix_kind
        if matrix_kind == "reaction_dynamics":
            declaration["environment_source_sha256"].update({
                "carryrl.evaluation.dynamics": hashlib.sha256(Path(importlib.import_module(
                    "carryrl.evaluation.dynamics").__file__).read_bytes()).hexdigest(),
                "matrix_runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            declaration["interpretation"] = (
                "Reaction-only validation speed sensitivity. Positions, sign, native96 and "
                "other physics stay fixed; altered speeds are distinct runtime conditions. "
                "RU/EN express the same goal. No training benefit, causal language, velocity "
                "estimation, memory benefit or new-game generalization is measured.")
        declaration["matrix_sha256"] = _hash(declaration)
        atomic_json(output / "protocol.json", declaration)
        atomic_json(output / "status.json", {"status": "running", "completed_strata": []})
        agent, _, reward, progress = load_sequence_checkpoint(
            Path(identity["path"]), config, evaluation_only=True, load_optimizer=False)
        agent.eval().requires_grad_(False)
        reward.eval().requires_grad_(False)
        frozen = (tensor_hash(agent), tensor_hash(reward))
        for spec, selected in strata:
            current = spec["id"]
            condition = spec.get("runtime_intervention")
            if condition is None:
                result = evaluate_sequence(agent, reward, selected, split=split)
            else:
                def factory(task, **kwargs):
                    return ReactionSpeedEvaluationEnv(task,
                        speed_magnitude=condition["speed_magnitude"], **kwargs)

                result = evaluate_sequence(agent, reward, selected, split=split,
                                           env_factory=factory)
                if (result["protocol_sha256"] != spec["base_protocol_sha256"]
                        or result.get("protocol", spec["base_protocol"]) != spec["base_protocol"]):
                    raise RuntimeError("evaluated base protocol differs from declared stratum")
                _verify_reaction_records(result["records"], selected, condition)
                result.update(base_protocol=spec["base_protocol"],
                              base_protocol_sha256=result["protocol_sha256"],
                              protocol=spec["protocol"], protocol_sha256=spec["protocol_sha256"],
                              runtime_intervention=condition, runtime_intervention_verified=True)
            if (tensor_hash(agent), tensor_hash(reward)) != frozen:
                raise RuntimeError("matrix evaluation mutated frozen checkpoint tensors")
            if result["protocol_sha256"] != spec["protocol_sha256"]:
                raise RuntimeError("evaluated protocol differs from declared stratum")
            if result.get("policy_and_reward_frozen") is not True:
                raise RuntimeError("stratum did not confirm frozen evaluation")
            expected = sum(spec["protocol"]["counts"].values()) * len(
                spec["protocol"]["policies"])
            if len(result["records"]) != expected:
                raise RuntimeError("stratum returned an incomplete episode cohort")
            result.update(stratum=current, matrix_sha256=declaration["matrix_sha256"],
                          mechanical_cohorts=_cohorts(result["records"]))
            atomic_json(output / f"{current}.json", result)
            reports[current] = result
            atomic_json(output / "status.json", {"status": "running",
                                                  "completed_strata": list(reports)})
        aggregate = {"status": "completed", **declaration,
                     "weights_revision": progress["weights_revision"],
                     "metrics": {name: result["summaries"] for name, result in reports.items()},
                     "mechanical_cohorts": {name: result["mechanical_cohorts"]
                                            for name, result in reports.items()},
                     "comparison_diagnostics": _comparisons(reports)}
        if matrix_kind == "appearance96":
            aggregate["theme_contrasts"] = _theme_contrasts(reports)
        atomic_json(output / "aggregate.json", aggregate)
        atomic_json(output / "status.json", {"status": "completed",
                                              "completed_strata": list(reports)})
        return aggregate
    except BaseException as error:
        atomic_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "failed_stratum": current, "completed_strata": list(reports),
            "error_type": type(error).__name__, "error": str(error),
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--all-levels", action="store_true")
    parser.add_argument("--matrix-kind", choices=("composite", "appearance96", "reaction_dynamics"),
                        default="composite")
    args = parser.parse_args()
    config = SequenceRunConfig.load(args.config)
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no implicit device fallback")
    report = run_matrix(args.checkpoint, config, args.output, split=args.split,
                        episodes_per_task=args.episodes_per_task, all_levels=args.all_levels,
                        matrix_kind=args.matrix_kind)
    print(json.dumps({"status": report["status"], "matrix_sha256": report["matrix_sha256"],
                      "strata": len(report["strata"])}))


if __name__ == "__main__":
    main()
