"""The robustness runner only orchestrates the existing frozen evaluator."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
from torch import nn

from carryrl.agent.model import AgentConfig
from carryrl.envs.atomic import AtomicEnv, _identity, atomic_level_banks
from carryrl.envs.atomic_suite import TASKS
from carryrl.envs.presentation import PresentationProfile
from carryrl.training.sequence import _environment_options
from carryrl.training.sequence_config import SequenceRunConfig

spec = importlib.util.spec_from_file_location(
    "evaluate_robustness_matrix",
    Path(__file__).parents[1] / "scripts/evaluate_robustness_matrix.py")
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)


def config():
    return SequenceRunConfig(
        tasks=("navigation", "mouse"), device="cpu",
        agent=replace(AgentConfig(), text_encoder="multilingual_e5_small"),
        episode_instructions={"navigation": ("Иди к цели", "Go to the goal"),
                              "mouse": ("Нажми на цель", "Click the target")})


def install_stubs(monkeypatch, output, *, fail_at=None):
    calls, loads = [], []
    agent, reward = nn.Linear(1, 1), nn.Linear(1, 1)

    def load(checkpoint, cfg, *, evaluation_only):
        assert evaluation_only is True
        loads.append(cfg)
        return agent, object(), reward, {"weights_revision": 3}

    def evaluate(actor, evaluator_reward, cfg, *, split):
        assert actor is agent and evaluator_reward is reward
        assert all(not parameter.requires_grad for parameter in actor.parameters())
        assert not actor.training and not evaluator_reward.training
        declared = json.loads((output / "protocol.json").read_text())
        assert len(declared["strata"]) == (8 if split == "validation" else 14)
        assert cfg.agent == loads[0].agent  # no architecture substitution
        assert cfg.eval_seed == loads[0].eval_seed
        assert cfg.level_bank_seed == loads[0].level_bank_seed
        assert cfg.curriculum_difficulty == loads[0].curriculum_difficulty
        assert not cfg.eval_stochastic
        assert cfg.tasks == loads[0].tasks
        assert all(len(variants) == 1 for variants in cfg.episode_instructions.values())
        calls.append(cfg)
        if fail_at == len(calls):
            raise RuntimeError("deliberate evaluator failure")
        protocol = matrix.sequence_evaluation_protocol(cfg, split=split)
        records = []
        for policy in protocol["policies"]:
            for task in cfg.tasks:
                for index in range(protocol["counts"][task]):
                    records.append({"task": task, "policy": policy, "seed": 1_000_000 + index,
                                    "episode_index": index, "level_index": None,
                                    "level_id": f"{task}-repeated", "semantic_level_id": task,
                                    "runtime_fingerprint":
                                    f"{task}-{cfg.visual_config.native_sizes}",
                                    "difficulty": 1, "action_seed": index,
                                    "initial_frame_sha256": str(cfg.visual_config)})
        return {"protocol_sha256": matrix._hash(protocol), "policy_and_reward_frozen": True,
                "records": records, "summaries": {"stub": {"episodes": len(records)}}}

    monkeypatch.setattr(matrix, "load_sequence_checkpoint", load)
    monkeypatch.setattr(matrix, "evaluate_sequence", evaluate)
    return calls, loads


def checkpoint(tmp_path):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    (directory / "manifest.json").write_text('{"test": true}')
    return directory


def test_complete_declared_matrix_frozen_and_separate_reports(tmp_path, monkeypatch):
    output = tmp_path / "matrix"
    calls, loads = install_stubs(monkeypatch, output)
    report = matrix.run_matrix(checkpoint(tmp_path), config(), output)
    assert len(loads) == 1 and len(calls) == 8
    assert report["status"] == "completed"
    assert report["coverage_mode"] == "seeded_diagnostic_subset"
    assert len(list(output.glob("*.json"))) == 11
    assert [cfg.visual_config.native_sizes for cfg in calls] == [
        (96,), (96,), (96,), (96,), (64,), (64,), (128,), (128,)]
    assert [cfg.visual_config.hud_layouts for cfg in calls][::2] == [
        ("top",), ("left",), ("bottom",), ("top",)]
    assert [cfg.visual_config.themes for cfg in calls][::2] == [
        ("slate",), ("slate",), ("violet",), ("violet",)]
    assert all(cfg.visual_config.textures for cfg in calls)
    assert all(cfg.visual_config.style_split == "validation" for cfg in calls[4:])
    assert calls[0].episode_instructions["navigation"] == ("Иди к цели",)
    assert calls[1].episode_instructions["navigation"] == ("Go to the goal",)
    cohort = report["mechanical_cohorts"]["canonical96_ru"][
        "frozen_deterministic/navigation"]
    assert cohort["episodes_started"] == 2
    assert len(cohort["distinct_level_ids"]) == 1  # actual duplicate diagnostic coverage
    compared = report["comparison_diagnostics"]["varied64_violet_bottom_en"][
        "frozen_deterministic/navigation"]
    assert compared["matches"]["level_id"] == 2
    assert compared["matches"]["runtime_fingerprint"] == 0
    assert json.loads((output / "status.json").read_text())["status"] == "completed"


def test_hash_covers_checkpoint_matrix_and_instruction(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    hashes = []
    for index in range(3):
        output = tmp_path / f"matrix{index}"
        install_stubs(monkeypatch, output)
        cfg = config()
        if index == 1:
            cfg = replace(cfg, episode_instructions={**cfg.episode_instructions,
                          "mouse": ("Новая цель", "New goal")})
        if index == 2:
            (path / "manifest.json").write_text('{"test": false}')
        hashes.append(matrix.run_matrix(path, cfg, output)["matrix_sha256"])
    assert len(set(hashes)) == 3
    base = matrix.build_matrix(config())
    changed = matrix.build_matrix(config(), episodes_per_task=3)
    assert base[0][0]["protocol_sha256"] != changed[0][0]["protocol_sha256"]


def test_missing_variant_fails_before_loading(tmp_path, monkeypatch):
    output = tmp_path / "matrix"
    calls, loads = install_stubs(monkeypatch, output)
    cfg = replace(config(), episode_instructions={"navigation": ("Only one",),
                                                 "mouse": ("ru", "en")})
    with pytest.raises(ValueError, match="positions 0/1"):
        matrix.run_matrix(checkpoint(tmp_path), cfg, output)
    assert not calls and not loads
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["failed_stratum"] is None


def test_failed_stratum_records_partial_completion_without_fallback(tmp_path, monkeypatch):
    output = tmp_path / "matrix"
    calls, loads = install_stubs(monkeypatch, output, fail_at=2)
    with pytest.raises(RuntimeError, match="deliberate"):
        matrix.run_matrix(checkpoint(tmp_path), config(), output)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["completed_strata"] == ["canonical96_ru"]
    assert status["failed_stratum"] == "canonical96_en"
    assert len(calls) == 2 and len(loads) == 1
    assert not (output / "aggregate.json").exists()


def test_all_levels_is_explicit_and_protocol_counts_cover_bank():
    strata = matrix.build_matrix(config(), all_levels=True)
    assert all(cfg.eval_all_levels for _, cfg in strata)
    assert all(spec["protocol"]["counts"]["navigation"] == 16 for spec, _ in strata)
    with pytest.raises(ValueError, match="held-out"):
        matrix.build_matrix(config(), split="train")
    with pytest.raises(ValueError, match="enabled text"):
        matrix.build_matrix(SequenceRunConfig())


@pytest.mark.parametrize("split", ("validation", "test"))
@pytest.mark.parametrize("task", TASKS)
def test_all_strata_render_declared_profiles_on_real_heldout_environments(split, task):
    cfg = replace(config(), tasks=(task,), episode_instructions={task: ("Цель", "Goal")})
    for declared, selected in matrix.build_matrix(cfg, split=split):
        visual = selected.visual_config
        env = AtomicEnv(task, split=split, **_environment_options(selected))
        try:
            frame, info = env.reset(seed=1_000_000)
            assert frame.dtype.name == "uint8" and frame.shape[-1] == 3
            assert info["split"] == split
            assert info["level_id"] in {
                _identity(level) for level in atomic_level_banks(task, cfg.level_bank_seed)[split]}
            if visual.mode == "canonical":
                assert info["presentation"] is None
                assert frame.shape == (96, 96, 3)
                assert env.image_size == 96
            else:
                observed = info["presentation"]
                assert observed["native_size"] == visual.native_sizes[0]
                assert observed["theme"] == visual.themes[0]
                assert observed["hud_layout"] == visual.hud_layouts[0]
                assert observed["textures"] is visual.textures
                assert env.image_size == visual.native_sizes[0]
                width, height = PresentationProfile(**observed).canvas_size
                assert frame.shape == (height, width, 3)
            position = declared["instruction_variant_position"]
            assert selected.episode_instructions[task] == (
                cfg.episode_instructions[task][position],)
        finally:
            env.close()


def test_test_matrix_declares_both_reserved_themes_at_each_native_size():
    strata = matrix.build_matrix(config(), split="test", all_levels=True)
    assert len(strata) == 14
    varied = [(spec, cfg) for spec, cfg in strata if cfg.visual_config.mode == "varied"]
    assert {(cfg.visual_config.native_sizes[0], cfg.visual_config.themes[0], spec["language"])
            for spec, cfg in varied} == {
                (size, theme, language) for size in (64, 96, 128)
                for theme in ("ice", "ember") for language in ("ru", "en")}
    assert all(cfg.visual_config.style_split == "test" for _, cfg in varied)
    assert all(spec["protocol"]["split"] == "test" for spec, _ in strata)


def test_invalid_split_fails_before_output_or_model(tmp_path, monkeypatch):
    output = tmp_path / "matrix"
    calls, loads = install_stubs(monkeypatch, output)
    with pytest.raises(ValueError, match="held-out"):
        matrix.run_matrix(tmp_path / "missing", config(), output, split="train")
    assert not output.exists() and not calls and not loads


def test_test_matrix_runner_propagates_split(tmp_path, monkeypatch):
    output = tmp_path / "matrix"
    calls, loads = install_stubs(monkeypatch, output)
    report = matrix.run_matrix(checkpoint(tmp_path), config(), output, split="test")
    assert len(calls) == 14 and len(loads) == 1
    assert report["status"] == "completed"


def test_cli_test_split_selects_reserved_matrix(tmp_path, monkeypatch):
    output = tmp_path / "matrix"
    calls, _ = install_stubs(monkeypatch, output)
    monkeypatch.setattr(matrix.SequenceRunConfig, "load", lambda _: config())
    monkeypatch.setattr("sys.argv", ["matrix", "--checkpoint", str(checkpoint(tmp_path)),
                                    "--config", "unused", "--output", str(output),
                                    "--split", "test"])
    matrix.main()
    assert len(calls) == 14
    assert all(cfg.visual_config.style_split == "test" for cfg in calls[2:])
