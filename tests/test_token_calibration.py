"""Short LR trials share fresh lineage without mutating the primary comparison."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_token_comparison import fake_project as _fake_project

from carryrl.training import token_calibration as calibration
from carryrl.training.sequence_config import SequenceRunConfig
from carryrl.training.token_comparison import prepare_comparison, verify_plan

fake_project = _fake_project


@pytest.fixture
def materialized_plan(fake_project, monkeypatch):
    root, output, _, _ = fake_project
    plan = prepare_comparison(root, output)
    arm = next(arm for arm in plan["arms"] if arm["spatial_tokens"] == 32)
    checkpoint = Path(arm["initial_checkpoint"])
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"fresh verified weights and empty optimizer")
    manifest = {
        "sha256": hashlib.sha256((checkpoint / "state.pt").read_bytes()).hexdigest(),
        "agent_sha256": "same-fresh-agent",
        "reward_sha256": "same-fresh-reward",
        "rollout": 0,
        "decisions": 0,
        "weights_revision": 0,
    }
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    arm.update(initialization_verified=True, initial_manifest=manifest)
    plan.update(modelinit_verified=True, fresh_reward_pins=[{"test_gate": "validated"}])
    # The production gate verifies reward artifact contents; replace only that dependency.
    monkeypatch.setattr(calibration, "require_fresh_reward", lambda *_: plan["fresh_reward_pins"])
    monkeypatch.setattr(calibration, "verify_plan", lambda _: copy.deepcopy(plan))
    return output, plan


def test_both_lr_trials_are_fair_immutable_and_keep_primary_configs(materialized_plan):
    output, plan = materialized_plan
    primary = {arm["config"]["path"]: Path(arm["config"]["path"]).read_bytes()
               for arm in plan["arms"]}
    artifacts = [calibration.prepare_calibration(output, lr) for lr in calibration.LEARNING_RATES]
    configs = []
    for artifact in artifacts:
        config = SequenceRunConfig.load(artifact["config"]["path"])
        assert config.agent.spatial_tokens == 32
        assert config.total_rollouts == 2 and config.n_epochs == 2
        assert config.episodes_per_task == 1 and len(config.tasks) == 16
        assert config.evaluate_initial and config.eval_every_rollouts == 2
        assert config.eval_episodes_per_task == 2 and not config.eval_all_levels
        assert config.evaluation_canonical and config.evaluation_pixel_art
        assert not config.eval_stochastic and not config.reset_optimizer_on_warm_start
        assert config.reward_checkpoint == str(output / "reward/reward_checkpoint.pt")
        assert artifact["initial_optimizer_state_entries"] == 0
        assert artifact["initial_manifest"]["rollout"] == 0
        assert Path(artifact["training_output"]).parent == output.parent
        assert Path(artifact["training_output"]).name.startswith(output.name + "_calibration_lr")
        assert not Path(artifact["training_output"]).exists()
        assert calibration.prepare_calibration(output, config.learning_rate) == artifact
        raw = config.to_dict()
        raw.pop("learning_rate")
        configs.append(raw)
    assert configs[0] == configs[1]
    assert artifacts[0]["initial_checkpoint"] == artifacts[1]["initial_checkpoint"]
    assert artifacts[0]["initial_manifest"] == artifacts[1]["initial_manifest"]
    assert all(Path(path).read_bytes() == before for path, before in primary.items())
    assert not plan["ready_to_train"]


def test_calibration_rejects_missing_model_and_fresh_reward(fake_project):
    root, output, _, _ = fake_project
    prepare_comparison(root, output)
    with pytest.raises(ValueError, match="materialize"):
        calibration.prepare_calibration(output, 0.00001)
    assert verify_plan(output)["modelinit_verified"] is False
    assert not (output / "calibration").exists()


def test_calibration_requires_pinned_materialized_reward(materialized_plan, monkeypatch):
    output, _ = materialized_plan
    monkeypatch.setattr(calibration, "require_fresh_reward", lambda *_: [{"changed": True}])
    with pytest.raises(ValueError, match="pinned fresh reward"):
        calibration.prepare_calibration(output, 0.00001)


def test_calibration_rejects_config_or_lineage_edits(materialized_plan):
    output, _ = materialized_plan
    artifact = calibration.prepare_calibration(output, 0.00001)
    config_path = Path(artifact["config"]["path"])
    original = config_path.read_text()
    config_path.write_text("{}")
    with pytest.raises(ValueError, match="immutable calibration config"):
        calibration.prepare_calibration(output, 0.00001)
    config_path.write_text(original)
    manifest_path = config_path.with_name("manifest.json")
    manifest_path.write_text("{}")
    with pytest.raises(ValueError, match="immutable calibration lineage"):
        calibration.prepare_calibration(output, 0.00001)


def test_calibration_rejects_nonzero_initial_counters(materialized_plan):
    output, plan = materialized_plan
    checkpoint = Path(next(arm for arm in plan["arms"] if arm["spatial_tokens"] == 32)
                      ["initial_checkpoint"])
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    manifest["weights_revision"] = 1
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="zero-update"):
        calibration.prepare_calibration(output, 0.00001)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def completed_calibrations(materialized_plan):
    output, plan = materialized_plan
    for lr in calibration.LEARNING_RATES:
        artifact = calibration.prepare_calibration(output, lr)
        initial = artifact["initial_manifest"]
        run = Path(artifact["training_output"])
        _json(run / "summary.json", {
            "status": "completed", "rollouts": 2, "optimizer_steps_this_process": 4,
            "weights_revision": 4, "initial_weights_revision": 0,
            "reward_sha256": initial["reward_sha256"], "evaluation_rollout": 2,
            "evaluation_pixel_rollout": 2,
        })
        _json(run / "metadata.json", {"warm_start_parent": {
            "checkpoint": artifact["initial_checkpoint"],
            "checkpoint_sha256": initial["sha256"],
            "agent_sha256": initial["agent_sha256"],
            "reward_sha256": initial["reward_sha256"],
            "source_counters": {"rollout": 0, "decisions": 0, "weights_revision": 0},
            "source_optimizer": {"state_entries": 0},
            "new_optimizer_hyperparameters": {"learning_rate": lr},
        }})
        _json(run / "config.json", SequenceRunConfig.load(artifact["config"]["path"]).to_dict())
        _json(run / "evaluation.json", {"rollout": 2})
        _json(run / "evaluation_pixel.json", {"rollout": 2})
        for name in ("evaluations", "evaluations_pixel"):
            _json(run / name / "rollout_0000.json", {
                "rollout": 0, "weights_revision": 0,
                "agent_sha256": initial["agent_sha256"],
                "reward_sha256": initial["reward_sha256"],
            })
        _json(run.with_name(run.name + "_launch") / "process.json", {
            "status": "exited", "actual_exit_code": 0, "stop_reason": None,
        })
        rows = []
        for rollout in (1, 2):
            rows.append({
                "rollout": rollout, "optimizer_steps": 2, "complete": True,
                "reward_snapshot_sha256": initial["reward_sha256"], "kl_early_stop": False,
                "epochs": [{
                    "policy_loss": -0.1, "value_loss": 0.1, "entropy": 1.0,
                    "approx_kl": 0.001, "clip_fraction": 0.0, "gradient_norm": 0.2,
                    "optimizer_steps": 2, "kl_early_stop": False,
                }],
            })
        (run / "training.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return output, plan


def test_finalize_shared_original_lr_pins_actual_evidence(completed_calibrations):
    output, _ = completed_calibrations
    selection = calibration.finalize_common_calibration(output)
    plan = json.loads((output / "manifest.json").read_text())
    assert selection["approved"] and selection["selected_learning_rate"] == 0.00001
    assert [trial["optimizer_steps"] for trial in selection["trials"]] == [4, 4]
    assert plan["ready_to_train"]
    assert plan["calibration_evidence_pins"] == selection["evidence_pins"]
    assert plan["calibration_selection_pin"] == calibration._pin(
        output / "calibration/selection.json"
    )
    assert all(calibration._pin(pin["path"]) == pin for pin in selection["evidence_pins"])
    assert calibration.finalize_common_calibration(output) == selection


@pytest.mark.parametrize("case", ["missing", "failed", "stopped", "lineage", "no_updates",
                                  "nonfinite", "kl", "wrong_lr", "missing_baseline"])
def test_finalize_rejects_incomplete_or_unhealthy_real_evidence(completed_calibrations, case):
    output, _ = completed_calibrations
    artifact = calibration.prepare_calibration(output, 0.00001)
    run = Path(artifact["training_output"])
    process_path = run.with_name(run.name + "_launch") / "process.json"
    if case == "missing":
        (run / "summary.json").unlink()
    elif case in ("failed", "stopped"):
        process = json.loads(process_path.read_text())
        process["actual_exit_code" if case == "failed" else "stop_reason"] = (
            1 if case == "failed" else "system swap growth"
        )
        _json(process_path, process)
    elif case in ("lineage", "wrong_lr"):
        metadata = json.loads((run / "metadata.json").read_text())
        if case == "lineage":
            metadata["warm_start_parent"]["agent_sha256"] = "another-policy"
        else:
            metadata["warm_start_parent"]["new_optimizer_hyperparameters"]["learning_rate"] = 1
        _json(run / "metadata.json", metadata)
    elif case == "no_updates":
        summary = json.loads((run / "summary.json").read_text())
        summary["optimizer_steps_this_process"] = 0
        _json(run / "summary.json", summary)
    elif case == "missing_baseline":
        (run / "evaluations_pixel/rollout_0000.json").unlink()
    else:
        rows = [json.loads(line) for line in (run / "training.jsonl").read_text().splitlines()]
        rows[0]["epochs"][0]["approx_kl"] = float("nan") if case == "nonfinite" else 0.02
        (run / "training.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises((ValueError, FileNotFoundError)):
        calibration.finalize_common_calibration(output)
    assert not (output / "calibration/selection.json").exists()
    assert not json.loads((output / "manifest.json").read_text())["ready_to_train"]
