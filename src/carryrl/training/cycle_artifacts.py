"""Prepare and audit concrete artifacts for the registered fresh all16 cycle."""

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

from carryrl.envs.atomic_suite import TASKS
from carryrl.training.run import atomic_json, file_sha256
from carryrl.training.sequence_config import SequenceRunConfig


def read(path):
    return json.loads(Path(path).read_text())


def _require_sealed_text_validation(plan, text, *, require_legacy_split=True):
    from carryrl.rewards.control_text_factorized import FACTORIZED_MODEL_VERSION
    from carryrl.rewards.control_text_structured import STRUCTURED_MODEL_VERSION
    from carryrl.training.rl_improvements import inspect_text_reward_artifact

    manifest = text / "manifest.json"
    if read(manifest).get("version") in (STRUCTURED_MODEL_VERSION, FACTORIZED_MODEL_VERSION):
        # The version's static admission authenticates and recomputes the sealed
        # report inside the bundle; an external legacy report cannot admit it.
        inspect_text_reward_artifact(text, file_sha256(manifest))
    else:
        sealed = read(plan / "text_sealed_validation.json")
        if ((require_legacy_split and sealed.get("split") != "sealed_test")
                or sealed.get("gates", {}).get("passed") is not True):
            raise ValueError("text reward has no frozen sealed validation")


def prepare_profiles(plan, *, visual_validation=None, text_reward=None):
    from carryrl.training.rl_improvements import prepare_rl_improvements

    plan = Path(plan).resolve()
    visual = plan / "reward_visual/reward_checkpoint.pt"
    text = Path(text_reward) if text_reward is not None else plan / "reward_text"
    gate = read(visual_validation or plan / "reward_visual/validation_report.json")
    if (gate.get("passed") is not True or gate.get("coverage_passed") is not True
            or gate.get("checkpoint_sha256") != file_sha256(visual)):
        raise ValueError("new visual reward has no matching successful stage gate")
    runtime = read(text / "runtime_validation.json")
    if runtime.get("passed") is not True or runtime.get("records") != 32:
        raise ValueError("text reward has no complete runtime instruction gate")
    _require_sealed_text_validation(plan, text)
    base = replace(SequenceRunConfig.load(plan / "policy_base.json"),
                   reward_checkpoint=str(visual), max_wall_seconds=14400,
                   minimum_free_disk_mib=8192)
    result = prepare_rl_improvements(
        plan / "profiles", base_config=base,
        admission_path=plan / "admission.json",
        admission_sha256=file_sha256(plan / "admission.json"),
        total_rollouts=32, control_schedule_rollouts=32,
        text_reward_checkpoint=text, text_reward_sha256=file_sha256(text / "manifest.json"),
        seed=3176, episodes_per_task=8)
    if result["preparation_ready"] is not True:
        raise ValueError("executable profiles did not pass dependency checks")
    directory = plan / "configs"
    directory.mkdir(exist_ok=False)
    for schedule in ("zero", "temporary"):
        config = SequenceRunConfig.load(plan / "profiles" / schedule / "config.json")
        config = replace(config, eval_every_rollouts=32, checkpoint_every_rollouts=16,
                         checkpoint_keep_raw=2, max_wall_seconds=14400,
                         rollout_disk_budget_mib=4096)
        atomic_json(directory / f"pilot_{schedule}.json", config.to_dict())
    probe = replace(config, seed=3175, total_rollouts=2, evaluate_initial=False,
                    eval_all_levels=False, eval_episodes_per_task=1,
                    eval_every_rollouts=1000, checkpoint_every_rollouts=2,
                    max_wall_seconds=3000)
    atomic_json(directory / "probe_curriculum.json", probe.to_dict())
    full_probe = replace(probe, seed=3174, task_curriculum=None, curriculum_difficulty=1.0)
    atomic_json(directory / "probe_full.json", full_probe.to_dict())
    return result


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite value in actual PPO diagnostics")
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)


def audit_probe(curriculum_run, full_run, output):
    """Require applied updates, frozen encoders and independent visible outcome parity."""
    import numpy as np

    from carryrl.envs.atomic import verify_visible_outcome

    reports, timings, episodes, matches = [], [], 0, 0
    for run in (Path(curriculum_run), Path(full_run)):
        job = read(run / "cycle_job.json")
        if (job["status"] != "completed" or job["rollouts"] != 2
                or not job["frozen_unchanged"] or not job["trainable_changed"]
                or job["optimizer_steps"] <= 0):
            raise ValueError("technical probe did not apply complete fresh PPO cycles")
        rows = [json.loads(line) for line in (run / "training.jsonl").open()]
        if len(rows) != 2:
            raise ValueError("probe journal does not contain exactly two cycles")
        for row in rows:
            _finite(row)
            ratio = row.get("first_minibatch_ratio")
            if not ratio or ratio["log_ratio_abs_max"] > 0.001:
                raise ValueError("probe collection/replay likelihoods disagree before update")
            if row["rollout"] == 1 and not row.get("gradient_probes"):
                raise ValueError("probe has no first-cycle component gradient diagnostics")
            task_counts = {task: sum(episode["task"] == task
                                    for episode in row["episode_records"]) for task in TASKS}
            if (row["optimizer_steps"] <= 0 or len(row["episode_records"]) != 128
                    or set(task_counts.values()) != {8}):
                raise ValueError("probe cycle omitted a game or PPO update")
            if run == Path(full_run):
                timings.append(row["cycle_wall_seconds"])
        for trace in sorted((run / "trajectories").glob("*/*.npz")):
            metadata = read(trace.with_suffix(".json"))
            if metadata.get("npz_sha256") != file_sha256(trace):
                raise ValueError("probe trajectory hash mismatch")
            with np.load(trace, allow_pickle=False) as data:
                visible = verify_visible_outcome(data["final_frame"])
            tail = metadata["reward_diagnostics"]["steps"][-1]["event_status"]
            expected = ("success" if visible["success"] else
                        "failure" if visible["failure"] else "play")
            if tail != expected or not visible["known"]:
                raise ValueError("probe reward outcome disagrees with independent RGB outcome")
            episodes += 1
            matches += 1
        if list(run.glob(".rollout-features-*")):
            raise ValueError("probe leaked temporary rollout features")
        reports.append({"run": str(run.resolve()), **job})
    if episodes != 512:
        raise ValueError("probe did not preserve all 4x128 episode traces")
    result = {"passed": True, "rollout_seconds": timings,
              "rgb_outcome_matches": matches, "episodes": episodes, "runs": reports,
              "scope": "technical integration and throughput only; no learned-game claim"}
    atomic_json(output, result)
    return result


def prepare_full_config(plan, output):
    plan = Path(plan)
    decisions = read(plan / "cycle_state.json")["decisions"]
    schedule, horizon = decisions["selected_schedule"], decisions["full_rollouts"]
    if schedule not in ("zero", "temporary") or horizon not in (128, 160, 192):
        raise ValueError("full run lacks an admitted pilot and feasible fixed horizon")
    config = SequenceRunConfig.load(plan / "configs" / f"pilot_{schedule}.json")
    config = replace(config, seed=3177, total_rollouts=horizon, control_schedule_rollouts=32,
                     eval_every_rollouts=64, checkpoint_every_rollouts=32,
                     max_wall_seconds=42000)
    atomic_json(output, config.to_dict())
    return {"config": str(Path(output).resolve()), "seed": 3177,
            "total_rollouts": horizon, "control_schedule_rollouts": 32}


def frozen_candidates(run, output):
    """Choose from actual complete validation checkpoints; never inspect test."""
    from carryrl.training.rl_cycle import select_final

    run = Path(run)
    candidates = []
    for path in sorted((run / "evaluations").glob("rollout_*.json")):
        rollout = int(path.stem.split("_")[-1])
        pixel = run / "evaluations_pixel" / path.name
        checkpoint = run / "checkpoints" / path.stem
        if not pixel.exists() or not (checkpoint / "manifest.json").exists():
            continue
        styles = {}
        identity = read(checkpoint / "manifest.json")
        for style, file in (("canonical", path), ("pixel_art", pixel)):
            raw = read(file)
            if (raw.get("rollout") != rollout or raw.get("split") != "validation"
                    or raw.get("agent_sha256") != identity.get("agent_sha256")):
                raise ValueError("validation report does not authenticate its checkpoint")
            games = {}
            for task in TASKS:
                rows = [row for row in raw["records"]
                        if row["task"] == task and row["policy"] == "frozen_deterministic"]
                if len(rows) != 16 or any(not row["outcome"].get("known") for row in rows):
                    raise ValueError("final validation selection has incomplete greedy cohort")
                games[task] = {"episodes": 16, "successes": sum(
                    row["outcome"].get("success") is True for row in rows)}
            styles[style] = {"episodes": 256, "successes": sum(
                row["successes"] for row in games.values()), "games": games}
        report = {"split": "validation", "mode": "greedy", "frozen_validation": True,
                  "styles": styles}
        candidates.append({"checkpoint": str(checkpoint.resolve()),
                           "rollout": rollout, "report": report})
    selected = select_final(candidates)
    selected["checkpoint_manifest_sha256"] = file_sha256(
        Path(selected["checkpoint"]) / "manifest.json")
    selected["candidates"] = candidates
    atomic_json(output, selected)
    return selected


def clean_successful_text_features(plan):
    """Only this cycle's reproducible cache; retain its inventory and fitted artifact."""
    plan = Path(plan).resolve()
    model = read(plan / "reward_text/manifest.json")
    if model.get("ready") is not True:
        raise ValueError("failed or unvalidated text artifacts must be preserved for diagnosis")
    _require_sealed_text_validation(plan, plan / "reward_text", require_legacy_split=False)
    cache = plan / "text_features"
    if cache.is_symlink() or cache.resolve().parent != plan:
        raise ValueError("text cache must belong directly to this cycle")
    files = [path for path in cache.rglob("*") if path.is_file()]
    if any(path.is_symlink() for path in cache.rglob("*")):
        raise ValueError("text cache contains symlinks")
    result = {"cache": str(cache), "files": len(files),
              "bytes": sum(path.stat().st_size for path in files),
              "model_manifest_sha256": file_sha256(plan / "reward_text/manifest.json"),
              "reason": "reproducible frozen-token cache after verified self-contained fit"}
    atomic_json(plan / "text_feature_cleanup.json", result)
    shutil.rmtree(cache)
    return result


def _wilson(successes, total):
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total
                          + z * z / (4 * total * total)) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def _score_table(report, policy):
    result = {}
    for task in TASKS:
        rows = [row for row in report["records"]
                if row["task"] == task and row["policy"] == policy]
        if len(rows) != 16 or any(not row["outcome"].get("known") for row in rows):
            raise ValueError("endpoint comparison needs complete known per-game cohorts")
        wins = sum(row["outcome"].get("success") is True for row in rows)
        result[task] = {
            "wins": wins, "episodes": 16,
            "failures": sum(row["outcome"].get("failure") is True for row in rows),
            "timeouts": sum(row["truncated"] for row in rows),
            "perfect_score_wins": sum(row["outcome"].get("success") is True
                                       and row["outcome"].get("score") == 5 for row in rows),
            "wilson_95_reference_interval": _wilson(wins, 16),
        }
    return result


def _cohort_signature(report):
    rows = [row for row in report["records"] if row["policy"] == "frozen_stochastic"]
    return sorted((row["task"], row["episode_index"], row["runtime_fingerprint"],
                   row["initial_frame_sha256"], row["instruction"], row["action_seed"])
                  for row in rows)


def final_cycle_result(plan, run, output):
    """Retain each action seed/style/mode; a completed process is not a skill claim."""
    plan, run = Path(plan), Path(run)
    decision = read(plan / "cycle_state.json")["decisions"]
    selected = Path(decision["checkpoint"])
    if (decision.get("model_frozen") is not True
            or file_sha256(selected / "manifest.json")
            != decision["checkpoint_manifest_sha256"]):
        raise ValueError("final result requires an unchanged frozen selected checkpoint")
    artifacts, tables = {}, {}
    for file in sorted((plan / "endpoints").glob("*/*.json")):
        if file.name == "status.json":
            continue
        report = read(file)
        if "records" not in report:
            continue
        key = f"{file.parent.name}/{file.stem}"
        tables[key] = {mode: _score_table(report, policy) for mode, policy in (
            ("greedy", "frozen_deterministic"), ("stochastic", "frozen_stochastic"),
            ("random", "generic_physical_random"))}
        artifacts[str(file)] = file_sha256(file)
    for style, folder in (("canonical", "evaluations"), ("pixel_art", "evaluations_pixel")):
        file = run / folder / "rollout_0000.json"
        report = read(file)
        tables[f"own_cp0/{style}"] = {mode: _score_table(report, policy) for mode, policy in (
            ("greedy", "frozen_deterministic"), ("stochastic", "frozen_stochastic"),
            ("random", "generic_physical_random"))}
        artifacts[str(file)] = file_sha256(file)
        selected_report = read(plan / "endpoints/selected_validation"
                               / f"validation_{style}_80001.json")
        previous_report = read(plan / "endpoints/previous_validation"
                               / f"validation_{style}_80001.json")
        if not (_cohort_signature(report) == _cohort_signature(selected_report)
                == _cohort_signature(previous_report)):
            raise ValueError("endpoint controls changed levels, RGB, instructions or action seeds")
    expected = [f"selected_validation/validation_{style}_{seed}"
                for style in ("canonical", "pixel_art") for seed in (80001, 80002, 80003)]
    expected += [f"previous_validation/validation_{style}_80001"
                 for style in ("canonical", "pixel_art")]
    expected += [f"selected_test/test_{style}_80001" for style in ("canonical", "pixel_art")]
    if any(key not in tables for key in expected):
        raise ValueError("final result is missing registered endpoint evaluations")
    comparisons = {}
    for style in ("canonical", "pixel_art"):
        current = tables[f"selected_validation/validation_{style}_80001"]["stochastic"]
        initial = tables[f"own_cp0/{style}"]["stochastic"]
        previous = tables[f"previous_validation/validation_{style}_80001"]["stochastic"]
        comparisons[style] = {
            "episodes": 256,
            "selected_wins": sum(row["wins"] for row in current.values()),
            "own_cp0_wins": sum(row["wins"] for row in initial.values()),
            "previous_cp128_wins": sum(row["wins"] for row in previous.values()),
            "games_improved_vs_cp0": [task for task in TASKS
                                       if current[task]["wins"] > initial[task]["wins"]],
            "games_regressed_vs_cp0": [task for task in TASKS
                                        if current[task]["wins"] < initial[task]["wins"]],
        }
    goal = all(row["wins"] == row["episodes"] for key in expected
               if key.startswith(("selected_validation/", "selected_test/"))
               for mode in ("greedy", "stochastic") for row in tables[key][mode].values())
    result = {
        "status": "completed", "checkpoint": str(selected), "goal_100_fixed_banks": goal,
        "comparisons": comparisons, "tables": tables, "artifacts": artifacts,
        "independent_full_training_seeds": 1, "validation_used_for_selection": True,
        "control_cohorts_match": True,
        "game_test_consumed_after_freeze": True,
        "uncertainty_scope": "Wilson intervals are binomial reference intervals; repeated "
        "action seeds/styles do not add independent levels or independent training runs.",
    }
    atomic_json(output, result)
    return result
