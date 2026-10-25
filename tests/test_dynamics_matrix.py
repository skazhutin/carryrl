"""Frozen speed conditions preserve positions but carry distinct runtime protocols."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_robustness_matrix import checkpoint, config, matrix

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv, atomic_level_banks
from carryrl.evaluation.dynamics import ReactionSpeedEvaluationEnv


def reaction_config():
    return replace(config(), tasks=("reaction",),
                   episode_instructions={"reaction": ("Поймай мяч", "Catch the ball")})


def test_matrix_declares_distinct_speed_protocols_and_fixed_validation_conditions():
    original = reaction_config()
    before = deepcopy(original.to_dict())
    strata = matrix.build_matrix(original, matrix_kind="reaction_dynamics", all_levels=True)
    assert original.to_dict() == before and len(strata) == 6
    assert len({spec["protocol_sha256"] for spec, _ in strata}) == 6
    assert len({spec["base_protocol_sha256"] for spec, _ in strata}) == 2
    for spec, cfg in strata:
        assert cfg.tasks == ("reaction",) and cfg.env_image_size == 96
        assert cfg.evaluation_difficulty == 1 and cfg.curriculum_difficulty == 1
        assert cfg.level_bank_seed == original.level_bank_seed
        assert cfg.visual_config.mode == "canonical" and not cfg.eval_stochastic
        assert spec["protocol"]["runtime_intervention"] == spec["runtime_intervention"]
        assert spec["protocol"]["counts"] == {"reaction": 16}
    with pytest.raises(ValueError, match="reserved"):
        matrix.build_matrix(original, split="test", matrix_kind="reaction_dynamics")
    with pytest.raises(ValueError, match="requires the reaction"):
        matrix.build_matrix(config(), matrix_kind="reaction_dynamics")


def test_reaction_subset_preserves_only_its_task_multipliers():
    original = replace(config(), tasks=("navigation", "mouse", "reaction"),
        episode_instructions={**config().episode_instructions,
                              "reaction": ("Поймай мяч", "Catch the ball")},
        task_episode_multipliers={"navigation": 2, "reaction": 3})
    strata = matrix.build_matrix(original, matrix_kind="reaction_dynamics")
    assert original.task_episode_multipliers == {"navigation": 2, "reaction": 3}
    assert all(cfg.task_episode_multipliers == {"reaction": 3} for _, cfg in strata)


@pytest.mark.parametrize("magnitude", [None, 2.0, 2.5])
def test_actual_speed_cache_positions_initial_pixels_and_banks(magnitude):
    banks = deepcopy(atomic_level_banks("reaction"))
    control = AtomicEnv("reaction", split="validation")
    varied = ReactionSpeedEvaluationEnv("reaction", speed_magnitude=magnitude)
    try:
        before, _ = control.reset(seed=42, options={"level_index": 0})
        initial, _ = varied.reset(seed=42, options={"level_index": 0})
        np.testing.assert_array_equal(initial, before)
        assert varied._ball_x == control._ball_x and varied._paddle_x == control._paddle_x
        assert varied._reaction_drop == control._reaction_drop
        assert varied._required_hits == control._required_hits == 3
        assert varied._paddle_radius == control._paddle_radius
        assert abs(varied._ball_dx) == (abs(control._ball_dx) if magnitude is None else magnitude)
        assert np.sign(varied._ball_dx) == np.sign(control._ball_dx)
        expected = varied._ball_x + varied._ball_dx
        varied.step(PhysicalAction())
        assert varied._ball_x == expected
        assert atomic_level_banks("reaction") == banks
    finally:
        control.close()
        varied.close()


@pytest.mark.parametrize("kwargs", [
    {"task": "mouse"}, {"split": "train"}, {"split": "test"},
    {"speed_magnitude": True}, {"speed_magnitude": float("nan")},
    {"speed_magnitude": 1.5}, {"speed_magnitude": -2.0},
    {"image_size": 64}, {"difficulty": 0.5},
])
def test_reaction_evaluation_rejects_other_axes(kwargs):
    with pytest.raises(ValueError):
        ReactionSpeedEvaluationEnv(**{"task": "reaction", **kwargs})


def install_runner_stubs(monkeypatch, *, mutation=None):
    actor, reward = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    calls = []

    def load(*args, **kwargs):
        assert kwargs == {"evaluation_only": True, "load_optimizer": False}
        return actor, None, reward, {"weights_revision": 8}

    def evaluate(_actor, _reward, cfg, *, split, env_factory):
        assert all(not parameter.requires_grad for parameter in actor.parameters())
        protocol = matrix.sequence_evaluation_protocol(cfg, split=split)
        records = []
        for policy in protocol["policies"]:
            env = env_factory("reaction", split=split, image_size=96,
                difficulty=1.0, level_bank_seed=cfg.level_bank_seed,
                visual_config=cfg.visual_config)
            try:
                initial, info = env.reset(seed=1_000_000, options={"level_index": 0})
                calls.append(env.speed_magnitude)
                records.append({"task": "reaction", "seed": 1_000_000,
                    "instruction": cfg.episode_instructions["reaction"][0],
                    "policy": policy, "action_seed": 0, "episode_index": 0,
                    "difficulty": 1.0, "level_index": 0,
                    "level_id": info["level_id"], "semantic_level_id": info["semantic_level_id"],
                    "runtime_fingerprint": info["runtime_fingerprint"],
                    "level_bank_seed": cfg.level_bank_seed, "presentation": info["presentation"],
                    "initial_frame_sha256": matrix.hashlib.sha256(initial.tobytes()).hexdigest(),
                    "outcome": {"known": True, "success": False}})
            finally:
                env.close()
        if mutation is not None:
            records[0][mutation] = "changed"
        # evaluate_sequence returns its protocol SHA, not the full protocol body.
        return {"records": records,
            "protocol_sha256": matrix._hash(protocol), "policy_and_reward_frozen": True,
            "summaries": {}}

    monkeypatch.setattr(matrix, "load_sequence_checkpoint", load)
    monkeypatch.setattr(matrix, "evaluate_sequence", evaluate)
    return calls


def test_runner_authenticates_effective_protocol_and_actual_runtime(tmp_path, monkeypatch):
    calls = install_runner_stubs(monkeypatch)
    output = tmp_path / "result"
    result = matrix.run_matrix(checkpoint(tmp_path), reaction_config(), output,
                               episodes_per_task=1, matrix_kind="reaction_dynamics")
    assert result["status"] == "completed" and result["updates"] == 0
    assert set(calls) == {None, 2.0, 2.5} and len(calls) == 12
    assert "carryrl.evaluation.dynamics" in result["environment_source_sha256"]
    report = json.loads((output / "reaction96_speed2_ru.json").read_text())
    assert report["runtime_intervention_verified"]
    assert report["protocol_sha256"] != report["base_protocol_sha256"]
    assert report["base_protocol"].get("runtime_intervention") is None
    assert report["protocol"]["runtime_intervention"]["speed_magnitude"] == 2.0


def test_transformed_validation_alias_rejected_before_loading_model(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "load_sequence_checkpoint",
                        lambda *_a, **_k: pytest.fail("unsafe cohort loaded a model"))
    with pytest.raises(ValueError, match="transformed validation overlaps protected TRAIN"):
        matrix.run_matrix(tmp_path / "unused_checkpoint",
            replace(reaction_config(), level_bank_seed=20261015), tmp_path / "result",
            episodes_per_task=1, matrix_kind="reaction_dynamics")


@pytest.mark.parametrize("mutation", ["runtime_fingerprint", "initial_frame_sha256", "level_id"])
def test_runner_rejects_wrong_runtime_or_initial_positions(tmp_path, monkeypatch, mutation):
    install_runner_stubs(monkeypatch, mutation=mutation)
    with pytest.raises(ValueError, match="reaction"):
        matrix.run_matrix(checkpoint(tmp_path), reaction_config(), tmp_path / "result",
                          episodes_per_task=1, matrix_kind="reaction_dynamics")
