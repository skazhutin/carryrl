"""Experiment options must be explicit and preserve archived configurations."""

import json

import pytest

from carryrl.agent.model import AgentConfig
from carryrl.training.sequence_config import SequenceRunConfig


def test_archived_config_keeps_legacy_training_and_reward_defaults(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"tasks": ["navigation", "mouse"]}))
    config = SequenceRunConfig.load(path)
    assert config.motor_initialization == "legacy"
    assert config.level_bank_seed == 0
    assert config.ppo_minibatch_episodes == 0
    assert config.ppo_task_balance is False
    assert config.eval_stochastic is False
    assert config.evaluate_initial is False
    assert config.visual_config is None
    assert config.reward_model_mode == "legacy_hud"
    assert config.agent.text_encoder == "none"
    assert config.agent.instruction_max_tokens == 512
    assert config.episode_instructions == {}
    assert SequenceRunConfig.from_dict(config.to_dict()) == config


def test_explicit_learning_options_round_trip_and_copy_sampling_weights():
    weights = {"navigation": 3, "mouse": 2}
    config = SequenceRunConfig(
        motor_initialization="balanced", ppo_minibatch_episodes=4, ppo_task_balance=True,
        target_kl=0.02, eval_stochastic=True, eval_all_levels=True, evaluate_initial=True,
        task_episode_multipliers=weights, curriculum_difficulty=0.2,
        curriculum_warmup_rollouts=8, reward_model_mode="screen_v2",
        reward_checkpoint=".cache/reward/reward_checkpoint.pt",
        level_bank_seed=7193,
    )
    weights["navigation"] = 99
    assert config.task_episode_multipliers["navigation"] == 3
    assert SequenceRunConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("options", [
    {"motor_initialization": "task_mask"},
    {"ppo_minibatch_episodes": -1},
    {"ppo_minibatch_episodes": True},
    {"ppo_task_balance": 1},
    {"target_kl": float("nan")},
    {"target_kl": -0.1},
    {"eval_seed": 2**32},
    {"level_bank_seed": -1},
    {"level_bank_seed": True},
    {"level_bank_seed": 1.0},
    {"level_bank_seed": 2**32},
    {"eval_stochastic": "yes"},
    {"eval_all_levels": None},
    {"task_episode_multipliers": {"navigation": 0}},
    {"task_episode_multipliers": {"navigation": True}},
    {"tasks": ["navigation"], "task_episode_multipliers": {"mouse": 2}},
    {"curriculum_difficulty": -0.1},
    {"curriculum_difficulty": float("nan")},
    {"curriculum_difficulty": True},
    {"curriculum_warmup_rollouts": 2},
    {"reward_model_mode": "unknown"},
    {"reward_checkpoint": ""},
])
def test_invalid_learning_options_fail_before_starting_a_run(options):
    with pytest.raises(ValueError):
        SequenceRunConfig(**options)


def test_explicit_instruction_variants_preserve_prose_copy_and_json_roundtrip():
    variants = ["Move toward the green exit.", "  Дойди до зелёного выхода.  "]
    instructions = {"navigation": variants}
    config = SequenceRunConfig(
        agent=AgentConfig(text_encoder="multilingual_e5_small"), tasks=("navigation",),
        episode_instructions=instructions)
    variants[0] = "changed"
    instructions["navigation"] = ["also changed"]
    assert config.episode_instructions["navigation"] == (
        "Move toward the green exit.", "  Дойди до зелёного выхода.  ")
    assert SequenceRunConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config


@pytest.mark.parametrize("instructions", [
    None, [], {"mouse": ["Click the target."]}, {"navigation": "Move toward the exit."},
    {"navigation": []}, {"navigation": [""]}, {"navigation": [" \n "]},
    {"navigation": [None]}, {"navigation": [1]}, {"navigation": ["x" * 8193]},
    {},
])
def test_text_enabled_config_requires_valid_prose_for_each_configured_task(instructions):
    with pytest.raises(ValueError, match="instruction"):
        SequenceRunConfig(agent=AgentConfig(text_encoder="multilingual_e5_small"),
                          tasks=("navigation",), episode_instructions=instructions)


def test_text_disabled_config_rejects_instruction_mapping_and_accepts_old_agent_defaults():
    with pytest.raises(ValueError, match="enabled text_encoder"):
        SequenceRunConfig(tasks=("navigation",),
                          episode_instructions={"navigation": ["Move toward the exit."]})
    old = SequenceRunConfig.from_dict({"agent": {"pretrained": False},
                                      "tasks": ["navigation"]})
    assert old.agent.text_encoder == "none" and old.episode_instructions == {}


def test_text_enabled_config_requires_instruction_for_all_configured_tasks():
    with pytest.raises(ValueError, match="every task"):
        SequenceRunConfig(agent=AgentConfig(text_encoder="multilingual_e5_small"),
                          tasks=("navigation", "memory"),
                          episode_instructions={"navigation": ["Move toward the exit."]})
