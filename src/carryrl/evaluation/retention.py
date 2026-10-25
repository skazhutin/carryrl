"""Descriptive retention counts and acceptance gates for a predeclared anchor pair.

These functions validate saved JSON evidence. They neither run a policy nor establish
statistical significance, general memory competence, or a winning training method.
"""

import json
from collections.abc import Mapping

from carryrl.evaluation.paired import paired_outcomes

VERSION = "carryrl.retention.v1"
# Quality counters are optional diagnostics; they do not define observed win outcomes.
_VERSIONS = ("suite_version", "scoring_version", "reward_version",
             "reward_model_version", "reward_composition")
_MEMORY_DESCRIPTOR = (
    "mode", "level_index", "seed", "origin_bank_level_id", "origin_semantic_level_id",
    "origin_runtime_fingerprint", "counterfactual_runtime_fingerprint",
    "counterfactual_level_id", "cue", "expected_direction", "unlock_ticks", "cue_ticks",
    "instruction", "presentation",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _same_fields(reports, fields):
    for field in fields:
        _require(all((field in row) == (field in reports[0])
                     and row.get(field) == reports[0].get(field) for row in reports[1:]),
                 f"retention protocol differs: {field}")


def _evaluation(report):
    _require(isinstance(report, Mapping), "evaluation must be a JSON mapping")
    _require(report.get("split") == "validation", "retention requires validation only")
    _require(report.get("policy_and_reward_frozen") is True,
             "retention requires frozen policy and reward")
    for field in ("protocol_sha256", "agent_sha256", "reward_sha256"):
        _require(_sha(report.get(field)), f"missing evaluation identity: {field}")
    for field in ("suite_version", "scoring_version"):
        _require(isinstance(report.get(field), str) and report[field],
                 f"missing evaluation version: {field}")
    records = report.get("records")
    _require(isinstance(records, list) and records
             and all(isinstance(row, Mapping) for row in records), "missing evaluation records")
    deterministic = [row for row in records if row.get("policy") == "frozen_deterministic"]
    _require(bool(deterministic), "missing frozen deterministic episodes")
    for row in deterministic:
        _require(row.get("split", "validation") == "validation", "test episode is forbidden")
        _require(row.get("presentation") is None or isinstance(row["presentation"], Mapping),
                 "presentation must be canonical null or a styled mapping")
    paired_outcomes(deterministic, deterministic)
    return deterministic


def _triple(rows):
    result = dict.fromkeys(("episodes", "known_triples", "unknown_triples", "old_wins",
                           "old_nonwins", "rescued_old_win", "harmed_old_win",
                           "blocked_acquisition", "extra_acquisition"), 0)
    result["known_by_arm"] = {arm: 0 for arm in ("baseline", "control", "anchored")}
    for triple in zip(*rows, strict=True):
        result["episodes"] += 1
        outcomes = [row["outcome"] for row in triple]
        for arm, outcome in zip(result["known_by_arm"], outcomes, strict=True):
            result["known_by_arm"][arm] += int(outcome["known"])
        if not all(outcome["known"] for outcome in outcomes):
            result["unknown_triples"] += 1
            continue
        result["known_triples"] += 1
        baseline, control, anchored = (outcome["success"] for outcome in outcomes)
        result["old_wins" if baseline else "old_nonwins"] += 1
        for category, pattern in (("rescued_old_win", (True, False, True)),
                                  ("harmed_old_win", (True, True, False)),
                                  ("blocked_acquisition", (False, True, False)),
                                  ("extra_acquisition", (False, False, True))):
            result[category] += int((baseline, control, anchored) == pattern)
    result["coverage"] = result["known_triples"] / result["episodes"]
    return result


def _scope(rows):
    if not rows[0]:
        return {"available": False, "reason": "no matching saved episodes"}
    return {"available": True, "triple_outcomes": _triple(rows), "pair_tables": {
        "baseline_to_control": paired_outcomes(rows[0], rows[1]),
        "baseline_to_anchor": paired_outcomes(rows[0], rows[2]),
        "control_to_anchor": paired_outcomes(rows[1], rows[2]),
    }}


def _select(rows, predicate):
    return [[row for row in arm if predicate(row)] for arm in rows]


def retention_outcomes(baseline, control, anchored):
    """Compare identical ordered validation episodes; unknown wins stay unknown.

    Full evaluate_sequence artifacts may include random/stochastic controls. Only
    frozen_deterministic records enter these tables. Missing styled or memory
    controls remain explicitly unmeasured. All denominators use known triples.
    """
    reports = baseline, control, anchored
    rows = [_evaluation(report) for report in reports]
    _same_fields(reports, ("protocol_sha256", "reward_sha256", *_VERSIONS))
    overall = _scope(rows)  # Validate all ordered descriptors before selecting any scope.
    return {"version": VERSION, "descriptive_only": True,
            "interpretation": "one predeclared pair; counts do not select a winner",
            "protocol_sha256": baseline["protocol_sha256"], **overall,
            "games": [{"task": task, **_scope(_select(rows, lambda row: row["task"] == task))}
                      for task in dict.fromkeys(row["task"] for row in rows[0])],
            "canonical": _scope(_select(rows, lambda row: row["presentation"] is None)),
            "styled": _scope(_select(rows, lambda row: row["presentation"] is not None)),
            "memory_game": {**_scope(_select(rows, lambda row: row["task"] == "memory")),
                            "interpretation": "memory-game wins alone do not prove memory"},
            "memory_controls": {"available": False,
                                "reason": "dedicated cue/mode artifacts not provided"}}


def _memory_rows(report):
    _require(isinstance(report, Mapping), "memory artifact must be a mapping")
    _require(report.get("diagnostic") == "paired_memory_cues", "wrong memory diagnostic")
    _require(report.get("split") == "validation", "memory retention requires validation only")
    _require(report.get("policy_and_reward_frozen") is True, "memory evaluation is not frozen")
    for field in ("agent_sha256", "reward_sha256", "bank_sha256"):
        _require(_sha(report.get(field)), f"missing memory identity: {field}")
    for field in ("difficulty", "level_bank_seed", "max_steps", "protocol", "first_choose_nll"):
        _require(field in report and report[field] is not None, f"missing memory protocol: {field}")
    _require("deterministic" in str(report["protocol"]).lower(),
             "memory protocol must declare deterministic policy")
    records = report.get("records")
    _require(isinstance(records, list) and records, "missing dedicated memory episodes")
    rows, identities = [], set()
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping)
                 and all(key in record for key in _MEMORY_DESCRIPTOR),
                 "incomplete memory episode descriptor")
        _require(isinstance(record["mode"], str) and record["mode"] in {"normal", "neutral_wait"}
                 and type(record["cue"]) is int and record["cue"] in {0, 1}
                 and type(record["level_index"]) is int and record["level_index"] >= 0
                 and type(record["seed"]) is int,
                 "invalid memory mode/cue")
        _require(record["expected_direction"] == ("left" if record["cue"] == 0 else "right"),
                 "memory expected direction does not match cue")
        identity = record["mode"], record["level_index"], record["cue"]
        _require(identity not in identities, "duplicate memory mode/level/cue")
        identities.add(identity)
        descriptor = {key: record[key] for key in _MEMORY_DESCRIPTOR}
        rows.append({"task": "memory", "seed": record["seed"],
                     "instruction": record["instruction"], "policy": "frozen_deterministic",
                     "action_seed": None, "episode_index": index,
                     "difficulty": report["difficulty"], "level_index": record["level_index"],
                     "level_id": record["counterfactual_level_id"],
                     "presentation": record["presentation"], "initial_frame_sha256": None,
                     "runtime_fingerprint": descriptor, "outcome": record.get("visible_outcome")})
    levels = {identity[1] for identity in identities}
    _require(identities == {(mode, level, cue) for mode in ("normal", "neutral_wait")
                            for level in levels for cue in (0, 1)},
             "incomplete memory mode/level/cue cohort")
    paired_outcomes(rows, rows)
    return rows


def memory_retention_outcomes(baseline=None, control=None, anchored=None):
    """Compare dedicated memory diagnostics without erasing cue or execution mode."""
    reports = baseline, control, anchored
    if any(report is None for report in reports):
        return {"available": False, "reason": "all three dedicated memory artifacts are required"}
    rows = [_memory_rows(report) for report in reports]
    # The caller authenticates each checkpoint identity; it changes between arms.
    # Everything except measured outputs, checkpoint identity and run timing is protocol.
    ignored = {"records", "summaries", "agent_sha256", "created_utc", "elapsed_seconds",
               "vision_cache", "checkpoint", "checkpoint_sha256", "weights_revision"}
    protocols = [{key: value for key, value in report.items() if key not in ignored}
                 for report in reports]
    _require(protocols[0] == protocols[1] == protocols[2], "memory protocol differs")
    return {"version": VERSION, "descriptive_only": True, **_scope(rows),
            "modes": {mode: _scope(_select(rows, lambda row: row["runtime_fingerprint"]["mode"]
                                          == mode)) for mode in ("normal", "neutral_wait")}}


def _json_equal(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def validate_retention_run(plan, arm_name, run_metadata, run_summary, checkpoint_manifest,
                           final_evaluation):
    """Reject incomplete/unmatched training evidence before controlled comparison.

    The caller must independently authenticate files and checkpoint archive contents.
    This gate checks their supplied JSON contracts, not filesystem or tensor hashes.
    """
    try:
        _require(arm_name in {"anchor_off", "anchor_on"}, "unknown retention arm")
        settings = plan["arms"][arm_name]["settings"]
        config = plan["effective_config"]
        source, corpus = plan["source_checkpoint"], plan["source_corpus"]
        metadata, summary, manifest = run_metadata, run_summary, checkpoint_manifest
        _require(_json_equal(metadata["settings"], settings), "run settings differ from plan")
        _require(_json_equal(metadata["config"], config), "effective config differs from plan")
        planned_provenance, observed_provenance = plan["source_provenance"], metadata["provenance"]
        for provenance in (planned_provenance, observed_provenance):
            _require(isinstance(provenance, Mapping), "source provenance must be a mapping")
            _require(isinstance(provenance["source_identity_version"], str)
                     and provenance["source_identity_version"]
                     and _sha(provenance["source_and_lock_sha256"]),
                     "source provenance identity is missing or malformed")
            dependencies = provenance["dependencies"]
            _require(isinstance(dependencies, Mapping) and bool(dependencies)
                     and all(isinstance(name, str) and name and isinstance(version, str) and version
                             for name, version in dependencies.items()),
                     "source provenance dependencies are missing or malformed")
        for field in ("source_identity_version", "source_and_lock_sha256", "dependencies"):
            _require(_json_equal(observed_provenance[field], planned_provenance[field]),
                     f"source provenance differs: {field}")
        for field, expected in (("source_checkpoint", source["path"]),
                                ("source_checkpoint_sha256", source["sha256"]),
                                ("initial_agent_sha256", source["agent_sha256"]),
                                ("reward_sha256", source["reward_sha256"]),
                                ("optimizer_state_preserved", True)):
            _require(_json_equal(metadata[field], expected), f"source identity differs: {field}")
        reuse = metadata["demonstration_reuse"]
        for field in ("source_run", "files", "producer_lineage", "source_collection_settings",
                      "source_difficulty", "episodes", "decisions", "unique_frames", "frame_bytes",
                      "exact_complete_corpus", "collection_skipped"):
            _require(_json_equal(reuse[field], corpus[field]), f"source corpus differs: {field}")
        _require(reuse["exact_complete_corpus"] is True and reuse["collection_skipped"] is True
                 and not reuse.get("appended_episodes"), "run must reuse the exact complete corpus")
        _require(_json_equal(summary["dataset"], {key: value for key, value in
                                               corpus["provenance"].items() if key != "episodes"}),
                 "training dataset provenance differs")
        _require(summary["demonstration_decisions"] == corpus["decisions"],
                 "training decision count differs")
        _require(summary["initial_agent_sha256"] == source["agent_sha256"]
                 and summary["reward_sha256"] == source["reward_sha256"],
                 "summary source hashes differ")
        epochs, updates = plan["expected_epochs"], plan["expected_optimizer_updates"]
        _require(type(epochs) is int and epochs > 0 and type(updates) is int and updates > 0,
                 "invalid planned training budget")
        batches = plan["batches_per_epoch"]
        _require(batches == -(-corpus["episodes"] // settings["minibatch_episodes"])
                 and updates == epochs * batches and settings["epochs"] == epochs,
                 "planned training budget is inconsistent")
        _require(type(summary["epochs_completed"]) is int and summary["epochs_completed"] == epochs
                 and type(summary["optimizer_steps"]) is int
                 and summary["optimizer_steps"] == updates
                 and summary["stop_reason"] == "completed", "training budget was not completed")
        history = summary["history"]
        _require(isinstance(history, list) and len(history) == epochs,
                 "completed epoch history is missing")
        for epoch, row in enumerate(history, 1):
            _require(row["epoch"] == epoch and row["complete"] is True
                     and row["optimizer_steps"] == epoch * batches
                     and row["episodes"] == corpus["episodes"],
                     "epoch history does not prove complete matched updates")
        coefficient = settings["canonical_anchor_coefficient"]
        _require((arm_name == "anchor_off" and coefficient == 0)
                 or (arm_name == "anchor_on" and coefficient > 0), "wrong anchor coefficient")
        anchor = metadata["canonical_anchor"]
        for field, expected in (("coefficient", coefficient), ("parent_checkpoint", source["path"]),
                                ("parent_checkpoint_sha256", source["sha256"]),
                                ("parent_agent_sha256", source["agent_sha256"])):
            _require(anchor[field] == expected, f"anchor parent differs: {field}")
        expected_objective = ("teacher-prefix conditional KL(parent||student); separately mean "
                              "eligible canonical unstyled 96 episodes and physical decisions")
        _require(anchor["objective"] == expected_objective, "anchor objective differs")
        training_anchor = summary["canonical_anchor"]
        _require(all(anchor.get(key) == value for key, value in training_anchor.items())
                 and training_anchor["coefficient"] == coefficient,
                 "anchor summary differs from metadata")
        _require(training_anchor["eligible_domain"] == "canonical_unstyled_96"
                 and training_anchor["prefixes"] ==
                 "expert target action components; episode history is actual executed actions",
                 "anchor eligibility or executed history differs")
        _require(training_anchor["parent_identity"] ==
                 "initial student policy; frozen independent trainable snapshot"
                 and training_anchor["aggregation"] ==
                 "mean eligible episode(mean decision(sum conditional KL(parent||student)))",
                 "anchor parent or conditional aggregation differs")
        if "replay" in plan:
            _require(training_anchor["eligible_episodes"] ==
                     plan["replay"]["domains"]["canonical_unstyled_96"]["episodes"],
                     "anchor eligibility differs from planned corpus")
        if coefficient:
            _require(type(training_anchor["eligible_episodes"]) is int
                     and training_anchor["eligible_episodes"] > 0
                     and _sha(training_anchor["parent_trainable_sha256"]),
                     "anchor-on lacks eligible canonical episodes or frozen parent hash")
        else:
            _require(training_anchor["parent_trainable_sha256"] is None,
                     "anchor-off unexpectedly declares an active parent")
        for field in ("suite_version", "scoring_version"):
            _require(manifest[field] == final_evaluation[field]
                     == plan["evaluation_protocol"][field], f"checkpoint version differs: {field}")
        _require(manifest["training_backend"] == summary["backend"]
                 == settings["training_backend"], "training backend differs")
        _require(_sha(manifest["sha256"]) and _sha(manifest["agent_sha256"]),
                 "checkpoint hashes missing")
        _require(manifest["agent_sha256"] == summary["final_agent_sha256"]
                 == final_evaluation["agent_sha256"], "final agent identity differs")
        _require(manifest["reward_sha256"] == final_evaluation["reward_sha256"]
                 == source["reward_sha256"], "frozen reward changed")
        revision = source["weights_revision"] + updates
        _require(type(manifest["weights_revision"]) is int
                 and manifest["weights_revision"] == revision
                 and final_evaluation["weights_revision"] == revision,
                 "final revision does not match actual optimizer updates")
        _require(final_evaluation["epoch"] == epochs, "final evaluation is from another epoch")
        _require(final_evaluation["protocol_sha256"] == plan["evaluation_protocol_sha256"],
                 "final evaluation protocol differs from plan")
        deterministic = _evaluation(final_evaluation)
        protocol = plan["evaluation_protocol"]
        _require(protocol["split"] == "validation" and protocol["eval_all_levels"] is True,
                 "controlled comparison requires full validation only")
        expected = [(task, index) for task in protocol["tasks"]
                    for index in range(protocol["counts"][task])]
        _require([(row["task"], row["episode_index"]) for row in deterministic] == expected,
                 "final evaluation is not the complete ordered validation bank")
        _require([(row["policy"], row["task"], row["episode_index"])
                  for row in final_evaluation["records"]]
                 == [(policy, task, index) for policy in protocol["policies"]
                     for task, index in expected],
                 "final evaluation is missing ordered policy controls")
        _require(all(row.get("split", "validation") == "validation"
                     for row in final_evaluation["records"]), "test episode is forbidden")
        for field, reported in (("eval_seed", "evaluation_seed"),
                                ("difficulty", "evaluation_difficulty"),
                                ("level_bank_seed", "level_bank_seed")):
            if field in protocol:
                _require(final_evaluation.get(reported) == protocol[field],
                         f"final evaluation protocol field differs: {field}")
    except (KeyError, TypeError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("incomplete or malformed retention run evidence") from error
    return {"valid": True, "arm": arm_name, "epochs_completed": epochs,
            "optimizer_steps": updates, "weights_revision": revision,
            "agent_sha256": manifest["agent_sha256"]}
