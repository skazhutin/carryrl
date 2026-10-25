import importlib.util
import json
from pathlib import Path

import pytest
from test_rl_improvements import text_bundle

from carryrl.envs.atomic_suite import TASKS
from carryrl.training.cycle_artifacts import (
    _cohort_signature,
    _finite,
    audit_probe,
    clean_successful_text_features,
    frozen_candidates,
    prepare_full_config,
    prepare_profiles,
)
from carryrl.training.sequence_config import SequenceRunConfig


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def greedy_report(successes, rollout=0):
    return {"split": "validation", "rollout": rollout, "agent_sha256": "b" * 64, "records": [
        {"task": task, "policy": "frozen_deterministic",
         "outcome": {"known": True, "success": i < successes[task]}}
        for task in TASKS for i in range(16)]}


def test_final_selection_prefers_worst_game_and_then_earlier_checkpoint(tmp_path):
    balanced = dict.fromkeys(TASKS, 2)
    concentrated = {**dict.fromkeys(TASKS, 16), "mouse": 0}
    for cycle, counts in ((0, balanced), (64, concentrated), (128, balanced)):
        for folder in ("evaluations", "evaluations_pixel"):
            write(tmp_path / folder / f"rollout_{cycle:04d}.json", greedy_report(counts, cycle))
        write(tmp_path / "checkpoints" / f"rollout_{cycle:04d}" / "manifest.json",
              {"sha256": "a", "agent_sha256": "b" * 64})
    result = frozen_candidates(tmp_path, tmp_path / "selection.json")
    assert result["rollout"] == 0
    assert result["minimum_success_rate"] == 2 / 16
    assert result["model_frozen"] and result["test_allowed"]


def test_final_selection_rejects_unknown_game_outcome(tmp_path):
    report = greedy_report(dict.fromkeys(TASKS, 1), 64)
    report["records"][0]["outcome"]["known"] = False
    for folder in ("evaluations", "evaluations_pixel"):
        write(tmp_path / folder / "rollout_0064.json", report)
    write(tmp_path / "checkpoints/rollout_0064/manifest.json", {"agent_sha256": "b" * 64})
    with pytest.raises(ValueError, match="incomplete"):
        frozen_candidates(tmp_path, tmp_path / "bad.json")


def test_full_config_requires_registered_admitted_horizon(tmp_path):
    write(tmp_path / "cycle_state.json", {"decisions": {
        "selected_schedule": "temporary", "full_rollouts": 64}})
    with pytest.raises(ValueError, match="admitted"):
        prepare_full_config(tmp_path, tmp_path / "bad.json")


def test_profiles_require_the_explicit_current_visual_gate(tmp_path):
    from carryrl.training.run import file_sha256

    checkpoint = tmp_path / "reward_visual/reward_checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"same frozen visual weights")
    write(tmp_path / "reward_visual/validation_report.json", {"passed": True,
          "coverage_passed": True, "checkpoint_sha256": file_sha256(checkpoint)})
    revised_gate = tmp_path / "reward_visual/validation_report_v2.json"
    write(revised_gate, {"passed": False, "coverage_passed": False,
                        "checkpoint_sha256": file_sha256(checkpoint)})
    with pytest.raises(ValueError, match="successful stage gate"):
        prepare_profiles(tmp_path, visual_validation=revised_gate)


def test_profiles_read_the_explicit_repaired_text_bundle(tmp_path):
    from carryrl.training.run import file_sha256

    checkpoint = tmp_path / "reward_visual/reward_checkpoint.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"unchanged visual weights")
    write(tmp_path / "reward_visual/validation_report.json", {"passed": True,
          "coverage_passed": True, "checkpoint_sha256": file_sha256(checkpoint)})
    # A failed repair must not inherit the readiness of the previous directory.
    write(tmp_path / "reward_text/runtime_validation.json", {"passed": True, "records": 32})
    repaired = tmp_path / "reward_text_repair1"
    write(repaired / "runtime_validation.json", {"passed": False, "records": 32})
    with pytest.raises(ValueError, match="runtime instruction gate"):
        prepare_profiles(tmp_path, text_reward=repaired)


def test_structured_bundle_materializes_profiles_using_its_authenticated_sealed_report(
    tmp_path, monkeypatch
):
    import carryrl.training.rl_improvements as improvements
    from carryrl.rewards.composition import OUTCOME_POTENTIAL_VERSION
    from carryrl.training.run import file_sha256

    visual = tmp_path / "reward_visual/reward_checkpoint.pt"
    visual.parent.mkdir()
    visual.write_bytes(b"frozen visual weights")
    write(visual.with_name("validation_report.json"), {
        "passed": True, "coverage_passed": True, "checkpoint_sha256": file_sha256(visual)})
    text = tmp_path / "reward_text_structured"
    digest = text_bundle(text)
    write(tmp_path / "admission.json", {})
    base = SequenceRunConfig(level_bank_seed=20261020, reward_composition={
        "version": OUTCOME_POTENTIAL_VERSION,
        "parameters": {"goal_choice_utility": "completion_only"}})
    write(tmp_path / "policy_base.json", base.to_dict())
    monkeypatch.setattr(improvements, "validate_task_curriculum_admission", lambda *a, **kw: {
        "stage_spec_sha256": "a" * 64})
    # A stale failed external legacy report cannot override this bundle's sealed gate.
    write(tmp_path / "text_sealed_validation.json", {"gates": {"passed": False}})
    result = prepare_profiles(tmp_path, text_reward=text)
    assert result["preparation_ready"]
    assert result["checks"]["text_reward"]["admission"]["sealed"]["passed"]
    config = SequenceRunConfig.load(tmp_path / "configs/pilot_temporary.json")
    assert config.text_reward_checkpoint == str(text)
    assert config.text_reward_sha256 == digest
    assert config.total_rollouts == config.control_schedule_rollouts == 32


@pytest.mark.parametrize("structured", [False, True])
def test_feature_cleanup_accepts_version_appropriate_sealed_gate(tmp_path, structured):
    text_bundle(tmp_path / "reward_text", structured=structured)
    if not structured:
        write(tmp_path / "text_sealed_validation.json", {"gates": {"passed": True}})
    cache = tmp_path / "text_features"
    cache.mkdir()
    (cache / "features.bin").write_bytes(b"reproducible")
    result = clean_successful_text_features(tmp_path)
    assert result["files"] == 1 and result["bytes"] == 12
    assert not cache.exists()
    assert (tmp_path / "reward_text/encoder.pt").is_file()


@pytest.mark.parametrize("fault", ["missing_sealed", "tampered"])
def test_feature_cleanup_preserves_cache_when_structured_evidence_is_invalid(tmp_path, fault):
    model = tmp_path / "reward_text"
    text_bundle(model)
    if fault == "missing_sealed":
        (model / "sealed_validation.json").unlink()
    else:
        (model / "sealed_validation.json").write_text("{}")
    write(tmp_path / "text_sealed_validation.json", {"gates": {"passed": True}})
    cache = tmp_path / "text_features"
    cache.mkdir()
    with pytest.raises(ValueError):
        clean_successful_text_features(tmp_path)
    assert cache.is_dir()
    assert not (tmp_path / "text_feature_cleanup.json").exists()


def test_full_config_preserves_fixed_aux_horizon_and_fresh_seed(tmp_path):
    write(tmp_path / "cycle_state.json", {"decisions": {
        "selected_schedule": "zero", "full_rollouts": 160}})
    from carryrl.rewards.composition import TEXT_CONTROL_VERSION

    base = SequenceRunConfig().to_dict()
    base["agent"]["text_encoder"] = "multilingual_e5_small"
    base.update(episode_instructions={task: ["Explicit public controls"] for task in TASKS},
                text_reward_checkpoint="frozen_text", text_reward_sha256="a" * 64,
                control_schedule_rollouts=32,
                reward_composition={"version": TEXT_CONTROL_VERSION,
                                    "parameters": {"control_schedule": "zero",
                                                   "goal_choice_utility": "completion_only"}})
    write(tmp_path / "configs/pilot_zero.json", base)
    result = prepare_full_config(tmp_path, tmp_path / "full.json")
    config = SequenceRunConfig.load(tmp_path / "full.json")
    assert result["total_rollouts"] == 160
    assert config.seed == 3177 and config.control_schedule_rollouts == 32
    assert config.eval_every_rollouts == 64 and config.checkpoint_every_rollouts == 32


@pytest.mark.parametrize("value", [float("nan"), {"nested": [float("inf")]}])
def test_probe_rejects_nonfinite_diagnostics(value):
    with pytest.raises(ValueError, match="non-finite"):
        _finite(value)


def test_endpoint_control_identity_includes_visible_scene_text_and_action_seed():
    record = {"task": "memory", "episode_index": 0, "runtime_fingerprint": "geometry",
              "initial_frame_sha256": "pixels", "instruction": "Remember the door",
              "action_seed": 80001, "policy": "frozen_stochastic"}
    original = _cohort_signature({"records": [record]})
    for field, value in (("initial_frame_sha256", "different pixels"),
                         ("instruction", "different rules"), ("action_seed", 80002)):
        assert _cohort_signature({"records": [{**record, field: value}]}) != original


@pytest.mark.parametrize("name", ["evaluate_cycle_frozen", "prepare_cycle_artifacts",
                                  "run_cycle_policy"])
def test_cycle_script_imports_without_training(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


@pytest.fixture
def probe_runs(tmp_path):
    import numpy as np

    from carryrl.envs.atomic import AtomicEnv
    from carryrl.training.run import file_sha256

    env = AtomicEnv("navigation", split="validation", level_bank_seed=20261020)
    env.reset(seed=0)
    frames = []
    for status in ("Success", "Failed", "Play"):
        env._status, env._score = status, 5 if status == "Success" else 0
        frames.append(env._draw())
    env.close()
    runs = [tmp_path / "curriculum", tmp_path / "full"]
    for run in runs:
        write(run / "cycle_job.json", {"status": "completed", "rollouts": 2,
              "frozen_unchanged": True, "trainable_changed": True, "optimizer_steps": 16})
        rows = [{"rollout": cycle, "optimizer_steps": 8, "cycle_wall_seconds": 20,
                 "first_minibatch_ratio": {"log_ratio_abs_max": 0.00001},
                 "gradient_probes": [{"actor": 0.1}] if cycle == 1 else [],
                 "episode_records": [{"task": task} for task in TASKS for _ in range(8)]}
                for cycle in (1, 2)]
        (run / "training.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        directory = run / "trajectories/rollout_0001"
        directory.mkdir(parents=True)
        for index in range(256):
            file = directory / f"episode_{index:04d}.npz"
            np.savez_compressed(file, final_frame=frames[index % 3])
            write(file.with_suffix(".json"), {
                "npz_sha256": file_sha256(file), "reward_diagnostics": {"steps": [
                    {"event_status": ("success", "failure", "play")[index % 3]}]}})
    return runs


def test_probe_audits_all_512_real_rgb_outcomes(probe_runs, tmp_path):
    result = audit_probe(*probe_runs, tmp_path / "audit.json")
    assert result["passed"] and result["rgb_outcome_matches"] == 512
    assert result["rollout_seconds"] == [20, 20]


@pytest.mark.parametrize("fault", ["tail", "hash", "missing"])
def test_probe_rejects_inconsistent_or_incomplete_primary_artifacts(probe_runs, tmp_path, fault):
    file = probe_runs[0] / "trajectories/rollout_0001/episode_0000.npz"
    metadata = file.with_suffix(".json")
    if fault == "missing":
        file.unlink()
    else:
        report = json.loads(metadata.read_text())
        if fault == "tail":
            report["reward_diagnostics"]["steps"][-1]["event_status"] = "failure"
        else:
            report["npz_sha256"] = "0" * 64
        write(metadata, report)
    with pytest.raises(ValueError):
        audit_probe(*probe_runs, tmp_path / "audit.json")
