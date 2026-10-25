"""Future experiment preparation cannot train or claim missing artifacts are ready."""

import hashlib
import json

import pytest

from carryrl.envs.atomic_suite import TASKS
from carryrl.rewards.composition import OUTCOME_POTENTIAL_VERSION
from carryrl.training.rl_improvements import (
    CONTROLS_PATH,
    INSTRUCTIONS_PATH,
    inspect_text_reward_artifact,
    prepare_rl_improvements,
)
from carryrl.training.sequence_config import SequenceRunConfig


def text_bundle(path, *, structured=True):
    """Static fixture deliberately has no loadable weights or downloaded encoder."""
    from test_control_text import calibration, report

    from carryrl.agent.language import TEXT_ID, TEXT_REVISION
    from carryrl.rewards.control_text import (
        CANDIDATES,
        MODEL_VERSION,
        calibration_gates,
        content_hash,
    )
    from carryrl.rewards.control_text_structured import (
        STRUCTURED_DECODER_VERSION,
        STRUCTURED_HEAD_ARCHITECTURE,
        STRUCTURED_MODEL_VERSION,
        STRUCTURED_POPULATION_CONTRACT,
    )
    from carryrl.rewards.control_text_structured_artifact import _admission

    path.mkdir()
    for name in ("encoder.pt", "head.pt"):
        (path / name).write_bytes(b"not loadable: static preparation only")
    validation = report()
    records = {"validation.json": validation, "calibration.json": calibration()}
    manifest = {"version": MODEL_VERSION, "encoder_id": TEXT_ID,
                "revision": TEXT_REVISION, "candidates": list(CANDIDATES),
                "training": {"trained": True}, "ready": True,
                "gates": calibration_gates(validation)}
    if structured:
        identity = {name: hashlib.sha256(source.read_bytes()).hexdigest()
                    for name, source in (("instructions_sha256", INSTRUCTIONS_PATH),
                                         ("controls_sha256", CONTROLS_PATH))}
        runtime = {"passed": True, "records": 32, "passed_records": 32, **identity,
                   "specification_hash": content_hash(identity), "results": [
                       {"task": task, "language": lang, "passed": True, "errors": [],
                        "counts": {name: {"total": 10, "correct": 10}
                                   for name in ("supported", "unsupported", "unknown", "held",
                                                "press", "aliases")}}
                       for task in TASKS for lang in ("ru", "en")]}
        records.update({"runtime_validation.json": runtime, "sealed_validation.json": report(),
                        "calibration.json": {
                            "joint_temperature": 1., "joint_threshold": .99,
                            "architecture": STRUCTURED_HEAD_ARCHITECTURE,
                            "decoder_version": STRUCTURED_DECODER_VERSION,
                            "population_contract": STRUCTURED_POPULATION_CONTRACT}})
        manifest.pop("gates")
        manifest.update(version=STRUCTURED_MODEL_VERSION,
                        head_architecture=STRUCTURED_HEAD_ARCHITECTURE,
                        training={"trained": True, "selected_development_feasible": True,
                                  "sealed_test_used": True})
        manifest["admission"] = _admission(validation, runtime, records["sealed_validation.json"],
                                            manifest["training"])
    for name, value in records.items():
        (path / name).write_text(json.dumps(value))
    return seal_text_bundle(path, manifest)


def seal_text_bundle(path, manifest):
    manifest["files"] = {str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
                         for file in path.rglob("*")
                         if file.is_file() and file.name != "manifest.json"}
    (path / "manifest.json").write_text(json.dumps(manifest))
    return hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()


def _base(**changes):
    composition = changes.pop("reward_composition", {
        "version": OUTCOME_POTENTIAL_VERSION,
        "parameters": {"goal_choice_utility": "completion_only"}})
    return SequenceRunConfig(level_bank_seed=20261020, episodes_per_task=2,
                             reward_composition=composition, **changes)


def test_preserves_primary_reward_recipe_and_scales_floor_with_success_reward(tmp_path):
    baseline = _base(reward_composition={"version": OUTCOME_POTENTIAL_VERSION,
        "parameters": {"success_reward": 2., "failure_penalty": 0.,
                       "progress_weight": .3, "time_cost": 0., "input_cost": 0.,
                       "goal_choice_utility": "completion_only"}})
    manifest = prepare_rl_improvements(tmp_path / "future", base_config=baseline,
                                      schedule_rollouts=8)
    primary = baseline.reward_composition.to_dict()["parameters"]
    assert manifest["primary_reward"]["parameters"] == primary
    for name, profile in manifest["profiles"].items():
        config = json.loads(open(profile["config"]).read())
        params = config["reward_composition"]["parameters"]
        params.pop("control_schedule")
        assert params == primary
        if name not in {"ablate_floor", "legacy_paired_base"}:
            assert config["advantage_std_floor"] == .2


def test_missing_dependencies_write_only_templates_and_no_fake_pins(tmp_path):
    output = tmp_path / "future"
    manifest = prepare_rl_improvements(output, base_config=_base(), schedule_rollouts=128)
    assert manifest["status"] == "blocked"
    assert not manifest["preparation_ready"]
    assert not manifest["training_started"]
    assert manifest["learning_quality"] == "unmeasured"
    assert manifest["pairing"]["shared_cp0_hash"] is None
    assert {item["input"] for item in manifest["required_inputs"]} == {
        "curriculum_admission", "frozen_text_reward", "shared_visual_reward"}
    assert not list(output.rglob("config.json"))
    assert len(list(output.rglob("config.template.json"))) == 9
    for name in ("zero", "temporary", "permanent"):
        profile = manifest["profiles"][name]
        config = json.loads(open(profile["config"]).read())
        assert config["text_reward_checkpoint"] is None
        assert config["text_reward_sha256"] is None
        assert config["task_curriculum"] is None
        assert config["tasks"] == list(TASKS)
        assert config["total_rollouts"] == config["control_schedule_rollouts"] == 128
        assert config["reward_composition"]["parameters"]["control_schedule"] == name
        assert config["ppo_minibatch_episodes"] == 32
        assert config["ppo_microbatch_episodes"] == 1
        assert config["agent"]["motor_version"] == "motor_v2"
        assert config["agent"]["critic_initialization"] == "zero"
        assert config["advantage_std_floor"] == .1
        assert config["rollout_memory_budget_mib"] == 2048
        assert config["rollout_feature_storage"] == "mmap"
        assert config["rollout_disk_budget_mib"] == 6144
        assert config["gradient_probe_every"] == 8
    assert manifest["instructions"]["sha256"] == hashlib.sha256(
        INSTRUCTIONS_PATH.read_bytes()).hexdigest()


def test_passing_preparation_materializes_runner_configs_with_controlled_profiles(
    tmp_path, monkeypatch
):
    import carryrl.training.rl_improvements as module

    visual = tmp_path / "visual.pt"
    visual.write_bytes(b"pinned visual artifact")
    admission = tmp_path / "admission.json"
    admission.write_text("{}")
    text = tmp_path / "text"
    text.mkdir()
    monkeypatch.setattr(module, "validate_task_curriculum_admission", lambda *a, **kw: {
        "stage_spec_sha256": "a" * 64})
    monkeypatch.setattr(module, "inspect_text_reward_artifact", lambda *a, **kw: {
        "gates": {"passed": True}})
    output = tmp_path / "prepared"
    manifest = prepare_rl_improvements(
        output, base_config=_base(reward_checkpoint=str(visual)),
        admission_path=admission, admission_sha256="a" * 64, schedule_rollouts=128,
        text_reward_checkpoint=text, text_reward_sha256="b" * 64)
    assert manifest["preparation_ready"]
    configs = {}
    for name, profile in manifest["profiles"].items():
        assert profile["executable_config"]
        config = SequenceRunConfig.load(profile["config"])
        configs[name] = config.to_dict()
        assert config.total_rollouts == config.control_schedule_rollouts == 128
        assert config.episode_instructions == SequenceRunConfig.from_dict(
            configs["zero"]).episode_instructions
        assert profile["future_command"][:6] == ["uv", "run", "--locked", "python", "-m", "carryrl"]
    for schedule in ("temporary", "permanent"):
        configs[schedule]["reward_composition"]["parameters"]["control_schedule"] = "zero"
        assert configs[schedule] == configs["zero"]
    assert configs["ablate_volume"]["episodes_per_task"] == 2
    assert configs["ablate_curriculum"]["task_curriculum"] is None
    assert configs["ablate_motor"]["agent"]["motor_version"] == "legacy"
    assert configs["ablate_critic"]["agent"]["critic_initialization"] == "legacy"
    assert configs["ablate_floor"]["advantage_std_floor"] == 1e-8
    assert manifest["profiles"]["zero"]["planned_budget"]["episodes_per_rollout"] == 128
    assert manifest["profiles"]["ablate_volume"]["planned_budget"]["episodes_per_rollout"] == 32
    with pytest.raises(FileExistsError):
        prepare_rl_improvements(output, base_config=_base(), schedule_rollouts=128)


def test_failed_admission_blocks_every_executable_profile(tmp_path, monkeypatch):
    import carryrl.training.rl_improvements as module

    def reject(*args, **kwargs):
        raise ValueError("goal_choice/bridge: insufficient admitted levels")

    monkeypatch.setattr(module, "validate_task_curriculum_admission", reject)
    manifest = prepare_rl_improvements(
        tmp_path / "blocked", base_config=_base(), admission_path="missing",
        admission_sha256="a" * 64, schedule_rollouts=8)
    assert not manifest["preparation_ready"]
    assert manifest["required_inputs"][0]["reason"].startswith("goal_choice/bridge")
    assert all(not row["executable_config"] for row in manifest["profiles"].values())


def test_static_text_gate_rejects_forged_or_unready_bundle_without_loading_models(tmp_path):
    from carryrl.agent.language import TEXT_ID, TEXT_REVISION
    from carryrl.rewards.control_text import CANDIDATES, MODEL_VERSION

    manifest = {"version": MODEL_VERSION, "encoder_id": TEXT_ID, "revision": TEXT_REVISION,
                "candidates": list(CANDIDATES), "ready": True, "files": {}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="self-contained"):
        inspect_text_reward_artifact(tmp_path, digest)
    with pytest.raises(ValueError, match="SHA256"):
        inspect_text_reward_artifact(tmp_path, "f" * 64)


@pytest.mark.parametrize("structured", [False, True])
def test_static_text_gate_accepts_both_authenticated_versions_without_loading(tmp_path, structured):
    path = tmp_path / "model"
    digest = text_bundle(path, structured=structured)
    checked = inspect_text_reward_artifact(path, digest)
    assert checked["gates"]["passed"]
    assert ("admission" in checked) is structured
    assert checked["runtime_weight_restore"].startswith("required again")


@pytest.mark.parametrize("fault", ["tampered", "missing_sealed", "runtime_incomplete",
                                  "sealed_failed", "calibration_failed", "admission_forged"])
def test_static_structured_gate_rejects_untrusted_or_incomplete_evidence(tmp_path, fault):
    path = tmp_path / "model"
    digest = text_bundle(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if fault == "tampered":
        (path / "head.pt").write_bytes(b"tampered")
    elif fault == "missing_sealed":
        (path / "sealed_validation.json").unlink()
    elif fault == "admission_forged":
        manifest["admission"]["runtime_passed"] = False
    else:
        name = {"runtime_incomplete": "runtime_validation.json",
                "sealed_failed": "sealed_validation.json",
                "calibration_failed": "calibration.json"}[fault]
        report = json.loads((path / name).read_text())
        if fault == "runtime_incomplete":
            report["results"].pop()
        elif fault == "sealed_failed":
            report["critical_errors"] = 1
        else:
            report["joint_threshold"] = .98
        (path / name).write_text(json.dumps(report))
    if fault != "tampered":
        # Even repinning changed bytes and leaving ready=True cannot skip gates.
        digest = seal_text_bundle(path, manifest)
    with pytest.raises(ValueError):
        inspect_text_reward_artifact(path, digest)


@pytest.mark.parametrize("argument", ["instructions_path", "controls_path"])
def test_static_structured_gate_binds_runtime_to_current_configuration(tmp_path, argument):
    path = tmp_path / "model"
    digest = text_bundle(path)
    changed = tmp_path / "changed.json"
    changed.write_text("changed configuration")
    with pytest.raises(ValueError, match="instruction/control identity"):
        inspect_text_reward_artifact(path, digest, **{argument: changed})


def test_cli_prepares_blocked_templates_without_launching(tmp_path):
    import importlib.util

    location = INSTRUCTIONS_PATH.parents[1] / "scripts" / "prepare_rl_improvements.py"
    spec = importlib.util.spec_from_file_location("prepare_rl_improvements_cli", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "base.json"
    config.write_text(json.dumps(_base().to_dict()))
    output = tmp_path / "future"
    assert module.main(["--config", str(config), "--output", str(output),
                        "--schedule-rollouts", "128"]) == 1
    assert (output / "manifest.json").is_file()
    assert not list(output.rglob("*.pt"))
    assert not list(output.rglob("status.json"))


def test_independent_total_and_control_schedule_horizons(tmp_path):
    manifest = prepare_rl_improvements(tmp_path / "independent", base_config=_base(),
                                      total_rollouts=192, control_schedule_rollouts=32)
    assert manifest["total_rollouts"] == 192
    assert manifest["horizon_rollouts"] == 32
    for profile in manifest["profiles"].values():
        config = json.loads(open(profile["config"]).read())
        assert config["total_rollouts"] == 192
        assert config["control_schedule_rollouts"] == 32
        assert profile["planned_budget"]["decisions_upper_bound"] == (
            192 * profile["planned_budget"]["episodes_per_rollout"] * 80)
