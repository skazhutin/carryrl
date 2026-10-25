"""Retention reports describe matched frozen outcomes and reject uncontrolled runs."""

from copy import deepcopy

import pytest

from carryrl.evaluation.retention import (
    memory_retention_outcomes,
    retention_outcomes,
    validate_retention_run,
)


def record(index, success=False, *, known=True, task="navigation", styled=False):
    return {"task": task, "seed": index + 1000000, "instruction": "Go",
            "policy": "frozen_deterministic", "action_seed": index, "episode_index": index,
            "difficulty": 1.0, "level_index": index, "level_id": f"level-{index}",
            "presentation": {"theme": "night"} if styled else None,
            "initial_frame_sha256": "a" * 64,
            "outcome": {"known": known, "success": success}}


def evaluation(rows, agent="a"):
    return {"split": "validation", "policy_and_reward_frozen": True,
            "protocol_sha256": "d" * 64, "agent_sha256": agent * 64,
            "reward_sha256": "e" * 64, "suite_version": "suite.v1",
            "scoring_version": "score.v1", "records": rows}


def test_all_eight_known_triples_are_descriptive_and_scopes_are_separate():
    states = [(bool(index & 4), bool(index & 2), bool(index & 1)) for index in range(8)]
    reports = [evaluation([record(index, state[arm], styled=index >= 4)
                           for index, state in enumerate(states)], agent=agent)
               for arm, agent in enumerate("abc")]
    saved = deepcopy(reports)
    report = retention_outcomes(*reports)
    counts = report["triple_outcomes"]
    assert counts["episodes"] == counts["known_triples"] == 8
    assert counts["old_wins"] == counts["old_nonwins"] == 4
    for field in ("rescued_old_win", "harmed_old_win", "blocked_acquisition", "extra_acquisition"):
        assert counts[field] == 1
    assert counts["unknown_triples"] == 0 and counts["coverage"] == 1
    for table in report["pair_tables"].values():
        assert table["gained"] == table["lost"] == table["retained_wins"] == 2
    assert report["canonical"]["triple_outcomes"]["old_nonwins"] == 4
    assert report["styled"]["triple_outcomes"]["old_wins"] == 4
    assert report["games"][0]["triple_outcomes"] == counts
    assert report["descriptive_only"] is True and "winner" not in report
    assert report["memory_game"]["available"] is False
    assert report["memory_controls"]["available"] is False
    assert reports == saved


def test_unknown_masks_have_pair_specific_coverage_and_never_imply_zero_success():
    baseline = evaluation([record(0, True), record(1, True), record(2, known=False)])
    control = evaluation([record(0), record(1, known=False), record(2, True)], "b")
    anchor = evaluation([record(0, True), record(1), record(2)], "c")
    report = retention_outcomes(baseline, control, anchor)
    assert report["triple_outcomes"]["known_triples"] == 1
    assert report["triple_outcomes"]["old_wins"] == 1
    assert report["triple_outcomes"]["unknown_triples"] == 2
    assert report["triple_outcomes"]["rescued_old_win"] == 1
    assert report["pair_tables"]["baseline_to_control"]["known_pairs"] == 1
    assert report["pair_tables"]["baseline_to_anchor"]["known_pairs"] == 2
    assert report["pair_tables"]["control_to_anchor"]["known_pairs"] == 2
    assert report["styled"] == {"available": False, "reason": "no matching saved episodes"}


@pytest.mark.parametrize("damage", ["order", "protocol", "split", "frozen", "version",
                                    "reward", "descriptor", "duplicate", "random_only"])
def test_comparison_rejects_incompatible_or_unfrozen_evidence(damage):
    baseline = evaluation([record(0), record(1)])
    changed = deepcopy(baseline)
    if damage == "order":
        changed["records"].reverse()
    elif damage == "protocol":
        changed["protocol_sha256"] = "f" * 64
    elif damage == "split":
        changed["split"] = "test"
    elif damage == "frozen":
        changed["policy_and_reward_frozen"] = False
    elif damage == "version":
        changed["reward_model_version"] = None  # Presence matters, even when null.
    elif damage == "reward":
        changed["reward_sha256"] = "f" * 64
    elif damage == "descriptor":
        changed["records"][0]["initial_frame_sha256"] = "b" * 64
    elif damage == "duplicate":
        changed["records"][1] = changed["records"][0]
    else:
        for row in changed["records"]:
            row["policy"] = "generic_physical_random"
    with pytest.raises(ValueError):
        retention_outcomes(baseline, changed, baseline)


def test_random_controls_are_excluded_and_memory_game_remains_a_separate_scope():
    rows = [record(0, True, task="memory"), record(0, task="mouse")]
    rows.append({**record(0, task="memory"), "policy": "generic_physical_random"})
    result = retention_outcomes(*(evaluation(rows) for _ in range(3)))
    assert result["triple_outcomes"]["episodes"] == 2
    assert result["memory_game"]["triple_outcomes"]["episodes"] == 1
    assert result["memory_controls"]["available"] is False


def test_legacy_and_new_quality_diagnostics_do_not_change_outcome_protocol_or_counts():
    legacy = evaluation([record(0, True), record(1)])
    newer = deepcopy(legacy)
    newer.update(quality_version="carryrl.quality.v1", quality_contracts={"navigation": {
        "diagnostic_counter": "new optional metadata"}})
    newer["records"][1]["outcome"]["success"] = True
    report = retention_outcomes(legacy, newer, newer)
    assert report["pair_tables"]["baseline_to_control"]["gained"] == 1
    assert report["triple_outcomes"]["old_wins"] == 1
    assert report["triple_outcomes"]["old_nonwins"] == 1
    assert "quality_version" not in report and "quality_contracts" not in report
    assert "quality" not in report["pair_tables"]["baseline_to_control"]


@pytest.mark.parametrize("field", ["suite_version", "scoring_version", "reward_version",
                                   "reward_model_version"])
def test_outcome_versions_remain_strict_despite_optional_quality_diagnostics(field):
    baseline = evaluation([record(0)])
    changed = {**baseline, field: "different-version", "quality_version": "optional.v1"}
    with pytest.raises(ValueError, match=field):
        retention_outcomes(baseline, baseline, changed)


def memory_report():
    report = {"diagnostic": "paired_memory_cues", "split": "validation", "difficulty": 1.0,
              "agent_sha256": "a" * 64, "reward_sha256": "e" * 64,
              "bank_sha256": "b" * 64, "level_bank_seed": 5, "max_steps": 80,
              "policy_and_reward_frozen": True, "protocol": "Deterministic RGB actor",
              "first_choose_nll": "disabled", "records": []}
    for mode in ("normal", "neutral_wait"):
        for cue in (0, 1):
            report["records"].append({
                "mode": mode, "level_index": 0, "seed": 1000000,
                "origin_bank_level_id": "origin", "origin_semantic_level_id": "semantic",
                "origin_runtime_fingerprint": "runtime",
                "counterfactual_runtime_fingerprint": f"runtime-{cue}",
                "counterfactual_level_id": f"counterfactual-{cue}", "cue": cue,
                "expected_direction": "left" if cue == 0 else "right", "unlock_ticks": 3,
                "cue_ticks": 2, "instruction": "Remember", "presentation": None,
                "visible_outcome": {"known": True, "success": cue == 1}})
    return report


def test_memory_diagnostic_preserves_modes_cues_and_separate_protocol():
    baseline = memory_report()
    anchored = deepcopy(baseline)
    anchored["records"][0]["visible_outcome"]["success"] = True
    result = memory_retention_outcomes(baseline, baseline, anchored)
    assert result["pair_tables"]["control_to_anchor"]["gained"] == 1
    assert result["modes"]["normal"]["pair_tables"]["control_to_anchor"]["gained"] == 1
    assert result["modes"]["neutral_wait"]["pair_tables"]["control_to_anchor"]["gained"] == 0
    assert memory_retention_outcomes(baseline, None, anchored)["available"] is False


def test_real_cli_memory_checkpoint_identities_change_without_changing_protocol():
    reports = []
    for revision, (arm, identity) in enumerate(zip(("baseline", "control", "anchor"), "abc"), 10):
        report = memory_report()
        report.update(checkpoint=f"/runs/{arm}/checkpoints/rollout_0000",
                      checkpoint_sha256={"state.pt": identity * 64, "manifest.json": "d" * 64},
                      weights_revision=revision,
                      agent_sha256=identity * 64)
        reports.append(report)
    result = memory_retention_outcomes(*reports)
    assert result["available"] is True
    assert result["triple_outcomes"]["known_triples"] == 4
    reports[2]["reward_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="memory protocol differs"):
        memory_retention_outcomes(*reports)


@pytest.mark.parametrize("damage", ["cue", "mode", "seed", "order", "missing", "protocol",
                                    "bank", "test", "frozen", "nll"])
def test_memory_diagnostic_rejects_changed_descriptor_or_protocol(damage):
    baseline = memory_report()
    other = deepcopy(baseline)
    if damage in {"cue", "mode", "seed"}:
        other["records"][0][damage] = {"cue": 1, "mode": "neutral_wait", "seed": 1000001}[damage]
    elif damage == "order":
        other["records"].reverse()
    elif damage == "missing":
        other["records"].pop()
    else:
        field, value = {"protocol": ("max_steps", 40), "bank": ("bank_sha256", "f" * 64),
                        "test": ("split", "test"), "frozen": ("policy_and_reward_frozen", False),
                        "nll": ("first_choose_nll", "enabled")}[damage]
        other[field] = value
    with pytest.raises(ValueError):
        memory_retention_outcomes(baseline, other, baseline)


def run_evidence(arm="anchor_on"):
    coefficient = 1.0 if arm == "anchor_on" else 0.0
    settings = {"epochs": 2, "minibatch_episodes": 2, "canonical_anchor_coefficient": coefficient,
                "training_backend": "mlx", "dataset_seed": 17}
    source = {"path": "/source/checkpoint", "sha256": "1" * 64, "agent_sha256": "a" * 64,
              "reward_sha256": "e" * 64, "weights_revision": 10}
    corpus = {"source_run": "/corpus", "files": {"demonstrations.json": {"sha256": "f" * 64}},
              "producer_lineage": {"initial_agent_sha256": "9" * 64},
              "source_collection_settings": {"seed": 5}, "source_difficulty": {"train": 1.0},
              "episodes": 3, "decisions": 8, "unique_frames": 10, "frame_bytes": 100,
              "exact_complete_corpus": True, "collection_skipped": True}
    anchor = {"coefficient": coefficient,
              "parent_trainable_sha256": "3" * 64 if coefficient else None,
              "eligible_domain": "canonical_unstyled_96", "eligible_episodes": 2,
              "parent_identity": "initial student policy; frozen independent trainable snapshot",
              "aggregation": "mean eligible episode(mean decision("
                             "sum conditional KL(parent||student)))",
              "prefixes": "expert target action components; "
                          "episode history is actual executed actions"}
    dataset = {"split": "train", "unique_frames": 10}
    provenance = {"source_identity_version": "carryrl.source.v2",
                  "source_and_lock_sha256": "4" * 64,
                  "dependencies": {"torch": "2.8.0", "numpy": "2.0.0"},
                  "git_commit": "old-commit", "git_dirty": ["untracked"]}
    plan = {"arms": {arm: {"settings": settings}}, "effective_config": {"max_grad_norm": 0.5},
            "source_provenance": deepcopy(provenance),
            "source_checkpoint": source, "source_corpus": {**corpus,
                "provenance": {**dataset, "episodes": [{"task": "navigation"}]}},
            "expected_epochs": 2, "expected_optimizer_updates": 4, "batches_per_epoch": 2,
            "evaluation_protocol_sha256": "d" * 64,
            "evaluation_protocol": {"split": "validation", "eval_all_levels": True,
                "policies": ["frozen_deterministic"],
                "tasks": ["navigation"], "counts": {"navigation": 2},
                "suite_version": "suite.v1", "scoring_version": "score.v1"}}
    metadata = {"settings": deepcopy(settings), "config": {"max_grad_norm": 0.5},
                "provenance": deepcopy(provenance),
                "source_checkpoint": source["path"], "source_checkpoint_sha256": source["sha256"],
                "initial_agent_sha256": source["agent_sha256"],
                "reward_sha256": source["reward_sha256"],
                "optimizer_state_preserved": True, "demonstration_reuse": deepcopy(corpus),
                "canonical_anchor": {**anchor, "parent_checkpoint": source["path"],
                    "parent_checkpoint_sha256": source["sha256"],
                    "parent_agent_sha256": source["agent_sha256"],
                    "objective": "teacher-prefix conditional KL(parent||student); separately mean "
                                 "eligible canonical unstyled 96 episodes and physical decisions"}}
    summary = {"dataset": dataset, "initial_agent_sha256": "a" * 64, "reward_sha256": "e" * 64,
               "epochs_completed": 2, "optimizer_steps": 4, "stop_reason": "completed",
               "demonstration_decisions": 8, "canonical_anchor": anchor, "backend": "mlx",
               "final_agent_sha256": "b" * 64,
               "history": [{"epoch": index, "complete": True, "optimizer_steps": index * 2,
                            "episodes": 3} for index in (1, 2)]}
    manifest = {"sha256": "c" * 64, "agent_sha256": "b" * 64, "reward_sha256": "e" * 64,
                "weights_revision": 14, "suite_version": "suite.v1", "scoring_version": "score.v1",
                "training_backend": "mlx"}
    final = {**evaluation([record(0), record(1)], "b"), "epoch": 2, "weights_revision": 14}
    return plan, metadata, summary, manifest, final


@pytest.mark.parametrize("arm", ["anchor_on", "anchor_off"])
def test_completed_controlled_run_matches_actual_epochs_updates_parent_and_final_identity(arm):
    plan, *evidence = run_evidence(arm)
    assert validate_retention_run(plan, arm, *evidence) == {
        "valid": True, "arm": arm, "epochs_completed": 2, "optimizer_steps": 4,
        "weights_revision": 14, "agent_sha256": "b" * 64}


def test_run_provenance_ignores_git_bookkeeping_when_sources_and_dependencies_match():
    plan, metadata, *rest = run_evidence()
    metadata["provenance"].update(git_commit="new-commit", git_dirty=[], path="/different/checkout")
    assert validate_retention_run(plan, "anchor_on", metadata, *rest)["valid"] is True


@pytest.mark.parametrize("field", ["source_identity_version", "source_and_lock_sha256",
                                   "dependencies"])
@pytest.mark.parametrize("damage", ["changed", "missing", "malformed"])
@pytest.mark.parametrize("side", ["plan", "run"])
def test_run_gate_requires_matching_source_lock_and_dependency_provenance(field, damage, side):
    plan, metadata, *rest = run_evidence()
    provenance = plan["source_provenance"] if side == "plan" else metadata["provenance"]
    if damage == "missing":
        del provenance[field]
    elif damage == "changed":
        provenance[field] = {"source_identity_version": "carryrl.source.v3",
                             "source_and_lock_sha256": "5" * 64,
                             "dependencies": {"torch": "2.9.0", "numpy": "2.0.0"}}[field]
    else:
        provenance[field] = {"source_identity_version": None,
                             "source_and_lock_sha256": "invalid-sha256",
                             "dependencies": {}}[field]
    with pytest.raises(ValueError):
        validate_retention_run(plan, "anchor_on", metadata, *rest)


@pytest.mark.parametrize("damage", ["settings", "config", "source", "optimizer", "corpus_files",
                                    "corpus_order", "appended", "dataset", "epochs", "steps",
                                    "wall", "history", "parent", "coefficient", "eligible",
                                    "parent_hash", "objective", "final_agent", "revision", "reward",
                                    "protocol", "test", "partial_bank", "missing_controls"])
def test_run_gate_rejects_partial_mutated_or_unmatched_training(damage):
    plan, metadata, summary, manifest, final = run_evidence()
    if damage == "settings":
        metadata["settings"]["dataset_seed"] += 1
    elif damage == "config":
        metadata["config"]["max_grad_norm"] = 1.0
    elif damage == "source":
        metadata["source_checkpoint_sha256"] = "f" * 64
    elif damage == "optimizer":
        metadata["optimizer_state_preserved"] = False
    elif damage == "corpus_files":
        metadata["demonstration_reuse"]["files"]["demonstrations.json"]["sha256"] = "0" * 64
    elif damage == "corpus_order":
        metadata["demonstration_reuse"]["producer_lineage"] = {}
    elif damage == "appended":
        metadata["demonstration_reuse"]["appended_episodes"] = 1
    elif damage == "dataset":
        summary["dataset"]["unique_frames"] = 9
    elif damage == "epochs":
        summary["epochs_completed"] = 1
    elif damage == "steps":
        summary["optimizer_steps"] = 3
    elif damage == "wall":
        summary["stop_reason"] = "wall_limit"
    elif damage == "history":
        summary["history"][-1]["complete"] = False
    elif damage == "parent":
        metadata["canonical_anchor"]["parent_agent_sha256"] = "f" * 64
    elif damage in {"coefficient", "eligible", "parent_hash"}:
        field, value = {"coefficient": ("coefficient", 0), "eligible": ("eligible_episodes", 0),
                        "parent_hash": ("parent_trainable_sha256", None)}[damage]
        summary["canonical_anchor"][field] = value
        metadata["canonical_anchor"][field] = value
    elif damage == "objective":
        metadata["canonical_anchor"]["objective"] = "trajectory KL"
    elif damage == "final_agent":
        final["agent_sha256"] = "f" * 64
    elif damage == "revision":
        manifest["weights_revision"] = final["weights_revision"] = 15
    elif damage == "reward":
        manifest["reward_sha256"] = "f" * 64
    elif damage == "protocol":
        final["protocol_sha256"] = "f" * 64
    elif damage == "test":
        final["split"] = "test"
    elif damage == "missing_controls":
        plan["evaluation_protocol"]["policies"].append("generic_physical_random")
    else:
        final["records"].pop()
    with pytest.raises(ValueError):
        validate_retention_run(plan, "anchor_on", metadata, summary, manifest, final)
