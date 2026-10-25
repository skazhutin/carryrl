"""Paired initialization, no-model planning, and immutable provenance checks."""

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_sequence_training import TinyLanguage, TinyReward, TinyVision

from carryrl.agent.model import AgentConfig, GeneralistAgent
from carryrl.training.sequence import (
    load_sequence_checkpoint,
    save_sequence_checkpoint,
    tensor_hash,
)
from carryrl.training.sequence_config import SequenceRunConfig
from carryrl.training.token_comparison import (
    BASE_CONFIG,
    INSTRUCTIONS,
    PRIMARY_SEED,
    QUERY,
    TOKENS,
    build_initial_agents,
    comparison_configs,
    materialize_initial,
    prepare_comparison,
    require_fresh_reward,
    verify_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def fake_project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    base = json.loads((ROOT / BASE_CONFIG).read_text())
    instructions = json.loads((ROOT / INSTRUCTIONS).read_text())
    _json(root / BASE_CONFIG, base)
    _json(root / INSTRUCTIONS, instructions)
    (root / "pyproject.toml").write_text('[project]\nname="carryrl"\n')
    (root / "uv.lock").write_text("pinned lock")
    source = root / "src/carryrl/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("# source snapshot\n")
    return root, root / "runs" / "comparison", base, instructions


def test_configs_are_fair_isolated_and_parse_with_real_schema(fake_project):
    _, _, base, instructions = fake_project
    original = copy.deepcopy(base)
    arms = comparison_configs(base, instructions=instructions)
    assert base == original
    normalized = []
    for tokens, raw in arms.items():
        config = SequenceRunConfig.from_dict(raw)
        assert config.agent.spatial_tokens == tokens
        assert config.agent.pretrained and config.agent.text_encoder == "multilingual_e5_small"
        assert config.training_backend == "mlx" and config.device == "cpu"
        assert config.level_bank_seed == 20261008
        assert len(config.tasks) * config.episodes_per_task == 16
        assert config.task_episode_multipliers == {} and config.reward_checkpoint is None
        assert config.total_rollouts * 16 == 2048
        assert config.minimum_free_disk_mib == 8192
        assert config.max_wall_seconds == 36000
        assert config.eval_all_levels and config.evaluate_initial
        assert config.evaluation_canonical and config.evaluation_pixel_art
        assert config.reward_composition.version == "carryrl.reward.outcome_potential_cost.v1"
        assert config.reward_composition.parameters.goal_choice_utility == "completion_only"
        assert config.visual_config.mode == "canonical" and config.visual_config.pixel_art
        normalized_raw = copy.deepcopy(raw)
        normalized_raw["agent"].pop("spatial_tokens")
        normalized.append(normalized_raw)
    assert normalized[0] == normalized[1] == normalized[2]
    arms[16]["episode_instructions"]["memory"].append("changed")
    assert arms[32]["episode_instructions"] == instructions
    assert comparison_configs(base, instructions=instructions)[16]["seed"] == PRIMARY_SEED


def test_plan_is_independent_of_all_old_runs_and_waits_for_new_reward(fake_project):
    root, output, _, _ = fake_project
    assert not (root / "runs").exists()
    manifest = prepare_comparison(root, output)
    assert manifest["prepared"] and not manifest["modelinit_verified"]
    assert not manifest["training_executed"] and manifest["policy_updates"] == 0
    assert not manifest["ready_to_train"]
    assert manifest["reward_setup"] == "pending_fresh_training_and_validation"
    assert manifest["old_experiment_dependencies"] == []
    assert len(manifest["arms"]) == 3
    assert all(
        not row["materialized"] and not row["executed"] for row in manifest["confirmation_seeds"]
    )
    assert not (output / "initial").exists() and not (output / "reward").exists()
    assert not list((root / "runs").glob("comparison_s*"))
    assert verify_plan(output) == manifest
    for arm in manifest["arms"]:
        assert Path(arm["training_output"]).parent == root / "runs"
        assert "scripts/run_token_comparison.py" in arm["future_argv"]
        assert arm["future_argv"][-1] == "--execute"
        cfg = SequenceRunConfig.load(arm["config"]["path"])
        assert cfg.reward_checkpoint == str(output / "reward/reward_checkpoint.pt")
    recipe = json.loads((output / "reward_recipe.json").read_text())
    assert not recipe["executed"] and recipe["test_opened"] is False
    assert "train-reward" in recipe["training_argv"]
    cfg = SequenceRunConfig.load(recipe["config"]["path"])
    assert cfg.reward_checkpoint is None and cfg.visual_config.mode == "varied"
    assert len(cfg.tasks) == 16
    with pytest.raises(ValueError, match="fresh reward is pending"):
        materialize_initial(output)
    assert not (output / "initial").exists()


def test_shared_calibrated_lr_is_explicit_and_cannot_skip_readiness(fake_project):
    root, output, base, instructions = fake_project
    arms = comparison_configs(base, instructions=instructions, learning_rate=1e-4)
    assert {raw["learning_rate"] for raw in arms.values()} == {1e-4}
    manifest = prepare_comparison(root, output, learning_rate=1e-4)
    assert not manifest["ready_to_train"] and not manifest["modelinit_verified"]
    assert all(
        json.loads(Path(a["config"]["path"]).read_text())["learning_rate"] == 1e-4
        for a in manifest["arms"]
    )
    for invalid in (0, -1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            comparison_configs(base, learning_rate=invalid)


def test_plan_does_not_import_training_runtime_or_torch(fake_project):
    root, output, _, _ = fake_project
    code = """
import importlib.abc, sys
class DenyModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"torch", "carryrl.agent.model", "carryrl.training.sequence"}:
            raise AssertionError("model import during plan: " + fullname)
sys.meta_path.insert(0, DenyModels())
from carryrl.training.token_comparison import prepare_comparison, materialize_initial
prepare_comparison(sys.argv[1], sys.argv[2])
try:
    materialize_initial(sys.argv[2])
except ValueError as error:
    assert "fresh reward is pending" in str(error)
else:
    raise AssertionError("pending reward unexpectedly allowed materialization")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(root), str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_no_external_learned_reward_and_no_incomplete_gate(fake_project):
    root, output, base, instructions = fake_project
    base["reward_checkpoint"] = "runs/old/reward_checkpoint.pt"
    with pytest.raises(ValueError, match="learned reward artifact"):
        comparison_configs(base, instructions=instructions)
    manifest = prepare_comparison(root, output)
    for name in ("reward_checkpoint.pt", "reward_report.json", "validation_report.json"):
        _json(output / "reward" / name, {"passed": True})
    with pytest.raises(ValueError, match="validation/provenance is incomplete"):
        require_fresh_reward(output, manifest)


def test_ready_flag_cannot_bypass_missing_calibration_evidence(fake_project):
    root, output, _, _ = fake_project
    manifest = prepare_comparison(root, output)
    manifest.update(ready_to_train=True, modelinit_verified=True)
    _json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="verified calibration evidence"):
        verify_plan(output)


def test_verify_rejects_changed_live_source_and_config_and_snapshot(fake_project):
    root, output, _, _ = fake_project
    manifest = prepare_comparison(root, output)
    config = Path(manifest["arms"][0]["config"]["path"])
    original = config.read_bytes()
    config.write_text("{}")
    with pytest.raises(ValueError, match="pinned file changed"):
        verify_plan(output)
    config.write_bytes(original)
    copied = output / "source_snapshot/src/carryrl/fixture.py"
    copied.write_text("changed")
    with pytest.raises(ValueError, match="source snapshot changed"):
        verify_plan(output)
    copied.write_bytes((root / "src/carryrl/fixture.py").read_bytes())
    (root / "uv.lock").write_text("changed")
    with pytest.raises(ValueError, match="pinned file changed"):
        verify_plan(output)


def test_verify_rejects_self_consistent_replacement_of_initial_weights(fake_project):
    root, output, _, _ = fake_project
    manifest = prepare_comparison(root, output)
    arm = manifest["arms"][0]
    checkpoint = Path(arm["initial_checkpoint"])
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"prepared fresh initialization")
    expected = {"sha256": hashlib.sha256((checkpoint / "state.pt").read_bytes()).hexdigest()}
    _json(checkpoint / "manifest.json", expected)
    arm.update(initial_manifest=expected, initialization_verified=True)
    _json(output / "manifest.json", manifest)
    verify_plan(output)
    # An internally consistent loader manifest must still match the recorded preparation identity.
    (checkpoint / "state.pt").write_bytes(b"swapped weights")
    replacement = {"sha256": hashlib.sha256((checkpoint / "state.pt").read_bytes()).hexdigest()}
    _json(checkpoint / "manifest.json", replacement)
    with pytest.raises(ValueError, match="initial checkpoint identity changed"):
        verify_plan(output)


def test_reward_gate_checks_actual_training_bank_and_fresh_initialization(fake_project):
    root, output, _, _ = fake_project
    manifest = prepare_comparison(root, output)
    cfg = json.loads(Path(manifest["reward_config_pin"]["path"]).read_text())
    directory = output / "reward"
    directory.mkdir()
    checkpoint = directory / "reward_checkpoint.pt"
    checkpoint.write_bytes(b"new reward fixture; no model deserialization in this gate")
    report = {
        "gate": {"passed": True},
        "policy_demonstrations": False,
        "test_levels_used": False,
        "seed": cfg["seed"],
        "tasks": cfg["tasks"],
        "level_bank_seed": cfg["level_bank_seed"],
        "initialization": "random",
        "imported_checkpoints": [],
        "old_corpora": [],
    }
    validation = {
        "format": "carryrl.fresh_reward_validation.v1",
        "passed": True,
        "tasks": cfg["tasks"],
        "hud_passed": True,
        "progress_passed": True,
        "initialization": "random",
        "imported_checkpoints": [],
        "old_corpora": [],
        "level_bank_seed": cfg["level_bank_seed"],
        "recipe_sha256": manifest["recipe_pin"]["sha256"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }

    def write_report(candidate):
        _json(directory / "reward_report.json", candidate)
        validation["reward_report_sha256"] = hashlib.sha256(
            (directory / "reward_report.json").read_bytes()
        ).hexdigest()
        _json(directory / "validation_report.json", validation)

    write_report(report)
    assert len(require_fresh_reward(output, manifest)) == 3
    for field, value in [
        ("level_bank_seed", 0),
        ("initialization", "warm_start"),
        ("imported_checkpoints", ["old.pt"]),
        ("old_corpora", ["old"]),
    ]:
        write_report(dict(report, **{field: value}))
        with pytest.raises(ValueError, match="validation/provenance"):
            require_fresh_reward(output, manifest)


def _tiny_factory(config, *, vision=None, language=None):
    return GeneralistAgent(
        config, vision=vision or TinyVision(), language=language or TinyLanguage()
    )


def _tiny_configs():
    agent = AgentConfig(
        pretrained=False,
        image_size=28,
        width=16,
        layers=1,
        heads=4,
        spatial_tokens=64,
        memory_tokens=2,
        segment_length=2,
        text_encoder="multilingual_e5_small",
    )
    common = SequenceRunConfig(
        agent=agent,
        tasks=("memory",),
        max_episode_steps=4,
        torch_threads=1,
        episode_instructions={"memory": ("Remember",)},
    )
    return {
        tokens: replace(common, agent=replace(agent, spatial_tokens=tokens)) for tokens in TOKENS
    }


def test_fresh_common_tensors_query_prefixes_shared_encoders_empty_optimizers_and_rng(tmp_path):
    torch.set_num_threads(1)
    configs = _tiny_configs()
    frozen = SimpleNamespace(
        vision=TinyVision().requires_grad_(False), language=TinyLanguage().requires_grad_(False)
    )
    before = (tensor_hash(frozen.vision), tensor_hash(frozen.language))
    state = torch.random.get_rng_state().clone()
    agents = build_initial_agents(configs, frozen, seed=PRIMARY_SEED, agent_factory=_tiny_factory)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert (tensor_hash(frozen.vision), tensor_hash(frozen.language)) == before
    master = agents[64][0]
    for tokens, (agent, optimizer) in agents.items():
        assert agent.vision is frozen.vision and agent.language is frozen.language
        assert not optimizer.state
        assert len(optimizer.state_dict()["state"]) == 0
        assert agent.resampler.queries.shape == (tokens, 16)
        for name, value in agent.state_dict().items():
            expected = (
                master.state_dict()[name][:tokens] if name == QUERY else master.state_dict()[name]
            )
            assert torch.equal(value, expected)
        assert agent.resampler(torch.zeros(2, 9, 384)).shape == (2, tokens, 16)
        initial = tmp_path / f"t{tokens}"
        initial.mkdir()
        checkpoint = save_sequence_checkpoint(
            agent,
            optimizer,
            TinyReward(),
            configs[tokens],
            initial,
            rollout=0,
            decisions=0,
            revision=0,
        )
        loaded, loaded_optimizer, _, progress = load_sequence_checkpoint(
            checkpoint,
            configs[tokens],
            evaluation_only=True,
            agent_factory=_tiny_factory,
            reward_factory=TinyReward,
        )
        assert tensor_hash(loaded) == tensor_hash(agent)
        assert not loaded_optimizer.state
        assert all(progress[key] == 0 for key in ("rollout", "decisions", "weights_revision"))
    assert agents[16][0].value_head.weight.data_ptr() != master.value_head.weight.data_ptr()
    repeated = build_initial_agents(configs, frozen, seed=PRIMARY_SEED, agent_factory=_tiny_factory)
    assert all(tensor_hash(repeated[n][0]) == tensor_hash(agents[n][0]) for n in TOKENS)


def test_builder_rejects_unfrozen_encoders():
    configs = _tiny_configs()
    frozen = SimpleNamespace(vision=TinyVision(), language=TinyLanguage())
    with pytest.raises(ValueError, match="vision must be frozen"):
        build_initial_agents(configs, frozen, seed=PRIMARY_SEED, agent_factory=_tiny_factory)
