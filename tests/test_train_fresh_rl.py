"""Fresh PPO execution guards without fitting or policy inference."""

import importlib.util
import json
import signal
from dataclasses import asdict
from pathlib import Path

import pytest

from carryrl.envs.atomic_suite import TASKS
from carryrl.envs.presentation import THEME_SPLITS, PresentationConfig
from carryrl.rewards.composition import OUTCOME_POTENTIAL_VERSION
from carryrl.training.sequence_config import SequenceRunConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_fresh_rl.py"
spec = importlib.util.spec_from_file_location("fresh_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def pin_configs(plan):
    manifest = runner.read(plan / "manifest.json")
    manifest["config_pins"] = {
        name: runner.sha(plan / name) for name in ("policy_config.json", "reward_config.json")}
    runner.publish(plan / "manifest.json", manifest)


@pytest.fixture
def bundle(tmp_path):
    root, plan = tmp_path / "project", tmp_path / "bundle"
    (root / "src").mkdir(parents=True)
    (root / "scripts").mkdir()
    for name in ("src/core.py", "scripts/runner.py", "pyproject.toml", "uv.lock"):
        (root / name).write_text("# fixture\n")
    policy = SequenceRunConfig(
        tasks=TASKS, level_bank_seed=20261008, reward_model_mode="screen_v2",
        reward_confidence_threshold=0.9, max_wall_seconds=12600,
        reward_composition={"version": OUTCOME_POTENTIAL_VERSION,
                            "parameters": {"goal_choice_utility": "completion_only"}},
    )
    runner.publish(plan / "policy_config.json", policy.to_dict())
    runner.publish(plan / "reward_config.json", {
        "tasks": list(TASKS), "level_bank_seed": 20261008, "epochs": 1,
        "episodes_per_task": 1, "validation_episodes_per_task": 1, "max_steps": 80,
        "image_size": 96, "seed": 7, "device": "cpu",
        "presentation_config": asdict(PresentationConfig(
            mode="varied", themes=THEME_SPLITS["train"] + THEME_SPLITS["validation"])),
    })
    runner.publish(plan / "manifest.json", {
        "format": runner.FORMAT, "budget_seconds": 14400, "reserve_seconds": 1800,
        "source_pins": {name: runner.sha(root / name) for name in runner.source_files(root)},
        "diagnostics": {"horizon": 80, "levels": 4, "seed": 15},
    })
    pin_configs(plan)
    return root, plan


@pytest.fixture
def operations(monkeypatch):
    calls = []

    def fit(output, config):
        calls.append(("fit", config))
        runner.publish(output / "reward_report.json", {"gate": {"passed": True}})
        (output / "reward_checkpoint.pt").write_bytes(b"fresh reward")
        return object(), {"gate": {"passed": True}, "state_sha256": "fresh"}

    def validate(reward, policy, diagnostics):
        calls.append(("validate", diagnostics))
        return {"hud_passed": True, "progress_passed": True,
                "coverage": {task: {"passed": True} for task in TASKS}}

    def train(policy, output, reward, stop_requested):
        calls.append(("train", policy, stop_requested))
        initial = output / "checkpoints/rollout_0000"
        final = output / "checkpoints/rollout_0001"
        initial.mkdir(parents=True)
        final.mkdir()
        runner.publish(output / "latest.json", {"checkpoint": str(final)})
        return {"status": "completed", "rollouts": 1, "decisions": 32,
                "initial_rollout": 0, "initial_decisions": 0,
                "evaluation_summaries": {"validation": {}}}

    monkeypatch.setattr(runner, "fit_reward", fit)
    monkeypatch.setattr(runner, "validate_reward", validate)
    monkeypatch.setattr(runner, "train_policy", train)
    return calls


def test_preflight_is_inert_and_does_not_register(bundle, operations):
    root, plan = bundle
    manifest, policy, _ = runner.verify_plan(plan, root=root)
    assert manifest["budget_seconds"] == 14400
    assert policy.tasks == TASKS
    assert operations == []
    assert not (plan / "execution.json").exists()
    assert not (root / ".cache").exists()


def test_default_cli_only_preflights(bundle, operations, monkeypatch, capsys):
    root, plan = bundle
    verify = runner.verify_plan
    monkeypatch.setattr(runner, "verify_plan", lambda value: verify(value, root=root))
    assert runner.main(["--plan", str(plan)]) == 0
    assert json.loads(capsys.readouterr().out)["training_started"] is False
    assert operations == []
    assert not (plan / "execution.json").exists()


def test_fresh_execution_gates_and_provenance(bundle, operations):
    root, plan = bundle
    registry = root / ".cache/active_learning_launch.json"
    runner.publish(registry, {"run": "old-run", "launch": "old-launch"})
    result = runner.execute(plan, root=root, stop_requested=lambda: False)
    assert [call[0] for call in operations] == ["fit", "validate", "train"]
    assert result["status"] == "completed"
    assert result["resume"] is None and result["warm_start"] is None
    assert result["imported_checkpoints"] == result["old_corpora"] == []
    assert result["initial_counters"] == {"rollout": 0, "decisions": 0, "weights_revision": 0}
    assert 12500 < operations[-1][1].max_wall_seconds <= 12600
    assert operations[-1][1].reward_checkpoint is None
    assert result["final_checkpoint_exists"]
    assert Path(result["initial_checkpoint"]).exists()
    assert runner.read(registry) == {
        "run": str(runner.policy_output(plan, root)), "launch": str(plan / "stages/main")}
    assert runner.read(plan / "previous_active_learning_launch.json")["run"] == "old-run"
    provenance = runner.read(runner.policy_output(plan, root) / "fresh_reward_provenance.json")
    assert provenance["reward_bootstrap_executed"]
    assert provenance["checkpoint_sha256"] == runner.sha(plan / "reward/reward_checkpoint.pt")
    assert "old-run" not in json.dumps(result)


@pytest.mark.parametrize("stage", ["fit", "validate"])
def test_failed_gate_aborts_before_policy(bundle, operations, monkeypatch, stage):
    root, plan = bundle
    if stage == "fit":
        monkeypatch.setattr(runner, "fit_reward", lambda *args: (object(), {"gate": {
            "passed": False}}))
    else:
        monkeypatch.setattr(runner, "validate_reward", lambda *args: {
            "hud_passed": True, "progress_passed": False, "coverage": {}})
    with pytest.raises(RuntimeError, match="gate failed"):
        runner.execute(plan, root=root, stop_requested=lambda: False)
    assert not runner.policy_output(plan, root).exists()
    assert runner.read(plan / "execution.json")["status"] == "failed"
    assert not any(call[0] == "train" for call in operations)


@pytest.mark.parametrize("stage", ["before_fit", "after_fit", "after_validation"])
def test_stop_during_preparation_never_constructs_policy(bundle, operations, monkeypatch, stage):
    root, plan = bundle
    stopped = stage == "before_fit"
    if stage != "before_fit":
        name = "fit_reward" if stage == "after_fit" else "validate_reward"
        original = getattr(runner, name)

        def stop_after(*args):
            nonlocal stopped
            result = original(*args)
            stopped = True
            return result

        monkeypatch.setattr(runner, name, stop_after)
    result = runner.execute(plan, root=root, stop_requested=lambda: stopped)
    assert result["status"] == "stopped"
    assert not runner.policy_output(plan, root).exists()
    assert not any(call[0] == "train" for call in operations)


def test_budget_counts_reward_preparation_and_callback(bundle, operations, monkeypatch):
    root, plan = bundle
    tick = [100.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: tick[0])
    fit = runner.fit_reward

    def costly_fit(*args):
        tick[0] += 400
        return fit(*args)

    monkeypatch.setattr(runner, "fit_reward", costly_fit)
    runner.execute(plan, root=root, stop_requested=lambda: False)
    _, effective, stop = operations[-1]
    assert effective.max_wall_seconds == 12200
    assert not stop()
    tick[0] = 12700
    assert stop()


def test_expired_preparation_stops_before_policy(bundle, operations, monkeypatch):
    root, plan = bundle
    tick = [0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: tick[0])
    fit = runner.fit_reward

    def costly_fit(*args):
        tick[0] = 12601
        return fit(*args)

    monkeypatch.setattr(runner, "fit_reward", costly_fit)
    assert runner.execute(plan, root=root, stop_requested=lambda: False)["status"] == "stopped"
    assert not any(call[0] == "train" for call in operations)


@pytest.mark.parametrize("name", [
    "execution.json", "reward", "policy", "stages/main",
    "stages/main/launch.json", "stages/main/status.json",
])
def test_existing_outputs_refused(bundle, operations, name):
    root, plan = bundle
    target = runner.policy_output(plan, root) if name == "policy" else plan / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("preserve")
    with pytest.raises(FileExistsError):
        runner.execute(plan, root=root, stop_requested=lambda: False)
    assert target.read_text() == "preserve"
    assert operations == []


def test_guard_created_stage_is_preserved(bundle, operations):
    root, plan = bundle
    guard = plan / "stages/main/process.json"
    runner.publish(guard, {"status": "running", "guard_pid": 12345})
    original = guard.read_bytes()
    result = runner.execute(plan, root=root, stop_requested=lambda: False)
    assert result["status"] == "completed"
    assert guard.read_bytes() == original
    assert runner.read(plan / "stages/main/status.json")["status"] == "completed"


@pytest.mark.parametrize("name", ["stages", "stages/main"])
def test_symlinked_stage_refused(bundle, operations, name, tmp_path):
    root, plan = bundle
    outside = tmp_path / "outside"
    outside.mkdir()
    target = plan / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(FileExistsError, match="real stage"):
        runner.execute(plan, root=root, stop_requested=lambda: False)
    assert operations == []
    assert list(outside.iterdir()) == []
    assert not (plan / "execution.json").exists()


@pytest.mark.parametrize("target", ["policy_config.json", "reward_config.json", "src/core.py"])
def test_hash_mismatch_fails_without_inference(bundle, operations, target):
    root, plan = bundle
    path = (root if target.startswith("src/") else plan) / target
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.execute(plan, root=root, stop_requested=lambda: False)
    assert operations == []
    assert runner.read(plan / "execution.json")["status"] == "failed"
    assert not (root / ".cache").exists()


def test_new_unpinned_source_rejected(bundle):
    root, plan = bundle
    (root / "src/new.py").write_text("unreviewed = True\n")
    with pytest.raises(ValueError, match="inventory"):
        runner.verify_plan(plan, root=root)


def test_policy_exception_retains_checkpoint_receipt(bundle, operations, monkeypatch):
    root, plan = bundle
    original = runner.train_policy

    def fail_after_checkpoint(*args):
        original(*args)
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(runner, "train_policy", fail_after_checkpoint)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        runner.execute(plan, root=root, stop_requested=lambda: False)
    result = runner.read(plan / "execution.json")
    assert result["status"] == "failed" and result["final_checkpoint_exists"]
    assert Path(result["initial_checkpoint"]).exists()


def test_native_adapter_injects_only_fresh_reward(monkeypatch, tmp_path):
    import carryrl.training.sequence as sequence

    reward = object()
    policy = object()

    def callback():
        return False

    def train(config, output, **kwargs):
        assert config is policy and output == tmp_path
        assert kwargs == {"reward": reward, "stop_requested": callback}
        return {"status": "completed"}

    monkeypatch.setattr(sequence, "train_sequence", train)
    assert runner.train_policy(policy, tmp_path, reward, callback)["status"] == "completed"


@pytest.mark.parametrize("changes", [
    {"reward_checkpoint": "/old/policy/state.pt"},
    {"tasks": [TASKS[0]]}, {"level_bank_seed": 1},
])
def test_config_rejects_old_inputs_or_wrong_scope(bundle, changes):
    root, plan = bundle
    runner.publish(plan / "policy_config.json", {
        **runner.read(plan / "policy_config.json"), **changes})
    pin_configs(plan)
    with pytest.raises(ValueError):
        runner.verify_plan(plan, root=root)


@pytest.mark.parametrize("themes", [
    THEME_SPLITS["train"],
    THEME_SPLITS["train"] + THEME_SPLITS["validation"] + THEME_SPLITS["test"],
    THEME_SPLITS["train"] + THEME_SPLITS["validation"] + ("ice",),
    THEME_SPLITS["train"] + THEME_SPLITS["validation"] + ("ember",),
])
def test_reward_rejects_missing_validation_or_test_themes(bundle, themes):
    root, plan = bundle
    reward = runner.read(plan / "reward_config.json")
    reward["presentation_config"]["themes"] = list(themes)
    runner.publish(plan / "reward_config.json", reward)
    pin_configs(plan)
    with pytest.raises(ValueError, match="train/validation HUD themes"):
        runner.verify_plan(plan, root=root)


@pytest.mark.parametrize("budget,reserve", [(14401, 1800), (14400, 100), (1800, 1800)])
def test_invalid_budget_rejected(bundle, budget, reserve):
    root, plan = bundle
    runner.publish(plan / "manifest.json", {
        **runner.read(plan / "manifest.json"), "budget_seconds": budget,
        "reserve_seconds": reserve})
    with pytest.raises(ValueError, match="budget"):
        runner.verify_plan(plan, root=root)


def test_sigusr1_handler_is_active_before_execute(monkeypatch, tmp_path):
    previous = signal.getsignal(signal.SIGUSR1)

    def execute(plan, *, stop_requested):
        signal.raise_signal(signal.SIGUSR1)
        assert stop_requested()
        return {"status": "stopped"}

    monkeypatch.setattr(runner, "execute", execute)
    assert runner.main(["--plan", str(tmp_path), "--execute"]) == 2
    assert signal.getsignal(signal.SIGUSR1) == previous
