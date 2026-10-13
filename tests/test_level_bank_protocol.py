"""Opt-in full-difficulty semantic banks preserve the archived seed-zero suite."""

import hashlib
import json
import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import (
    TASKS,
    AtomicEnv,
    _easier_level,
    _identity,
    atomic_level_banks,
    atomic_semantic_level_id,
)
from carryrl.training.sequence import _environment_options, evaluate_sequence, tensor_hash
from carryrl.training.sequence_config import SequenceRunConfig

LEGACY_BANK_HASHES = {
    "navigation": "eff17b6ee0b783969e01172b623f68d2ef666aa0bad4059d9fbb8a5187c68aa2",
    "reaction": "a5e68f37deff2bf61ee8e141462bae3dc5294cea4905747881c25405dc6e38b2",
    "mouse": "012497f3e52539f06f8c7fcb5ee1fb29a1d01296d99f91f9678780af077bb45b",
    "memory": "64002ec498a312165544a1cc9a830344795c84a8ec84833cf38f89f15502c326",
    "timing": "fdcb2e0a1c9bd2f1f468402cba0e5b12edf0a735caf7e0452e50c62cb8581310",
    "platforming": "1200ab765e33be7c15aacf3ce4707c159f32973c7129b4fd56a1a9004aafca38",
    "steering": "f0b62500d981a6f3899159b710a682ec6b3b44088f16728f3f33a2de5060c0ed",
    "camera": "dc202bfc4fce720286d4a810917c70b4921196c1794abc51ad364b080b620e7a",
    "interaction": "f799e110505253ef3dd3db422f4572bf47cab0224cf62dd7727b7c1cc8d84576",
    "causality": "67f1a338f363fb2c20fc185b18db8af7b73a92dab218fc25389f25d76675bfe8",
    "key_door": "0efcb0b61bacb36a4624217fd230d2c1e50dfaf4400bdacdac59ad3413e675ee",
    "goal_choice": "8b625e05baebb09e8fcd4dd4fffe1118d1c8e0d7804393d73ac19114c8dd4d4b",
    "hazards": "73aa8319d8dd2b382719b56903c31c60148acb3691c4f2263b180287589cc4a3",
    "shooting": "864761f658ba5dfb70aec31c90a20dcce98c7fb08c58fd775dea32410650b21b",
    "ui": "65121791b2760adec7dcb61418e2ccdee3b1400b81e495563632788c0a187d58",
    "chain": "9e3c54e3dc70c8d3c8e92930446dfd450dd95e86ebfad78332df6c60d1da26a6",
}


@pytest.mark.parametrize("task", TASKS)
def test_seed_zero_keeps_every_legacy_bank_byte(task):
    assert _identity(atomic_level_banks(task)) == LEGACY_BANK_HASHES[task]
    assert atomic_level_banks(task) == atomic_level_banks(task, 0)
    implicit = AtomicEnv(task)
    explicit = AtomicEnv(task, level_bank_seed=0)
    first, first_info = implicit.reset(seed=7)
    second, second_info = explicit.reset(seed=7)
    np.testing.assert_array_equal(first, second)
    assert first_info == second_info


@pytest.mark.parametrize("task", TASKS)
def test_opt_in_banks_repeat_and_separate_full_difficulty_semantics(task):
    banks = atomic_level_banks(task, 7193)
    atomic_level_banks.cache_clear()
    assert banks == atomic_level_banks(task, 7193)
    assert banks != atomic_level_banks(task, 7194)
    assert banks != atomic_level_banks(task, 0)
    signatures = {split: {atomic_semantic_level_id(task, level) for level in bank}
                  for split, bank in banks.items()}
    assert [len(signatures[split]) for split in ("train", "validation", "test")] == [64, 16, 16]
    assert len(set.union(*signatures.values())) == 96
    legacy = {atomic_semantic_level_id(task, level)
              for bank in atomic_level_banks(task).values() for level in bank}
    assert legacy.isdisjoint(set.union(*signatures.values()))
    env = AtomicEnv(task, split="validation", level_bank_seed=7193)
    _, info = env.reset(seed=7, options={"level_index": 0})
    assert info["level_bank_seed"] == 7193
    assert info["level_id"] == _identity(banks["validation"][0])
    assert info["semantic_level_id"] == atomic_semantic_level_id(task, banks["validation"][0])
    assert info["runtime_fingerprint"] == info["semantic_level_id"]
    reduced = AtomicEnv(task, level_bank_seed=7193, difficulty=0)
    _, reduced_info = reduced.reset(seed=7)
    assert reduced_info["runtime_fingerprint"] is None


def test_ui_render_only_changes_do_not_create_new_semantics():
    level = atomic_level_banks("ui", 7193)["train"][0]
    decorated = {**level, "request_x": 96 - level["request_x"],
                 "menu_offset": level["menu_offset"] + 1, "mechanic": "metadata"}
    assert _identity(level) != _identity(decorated)
    assert atomic_semantic_level_id("ui", level) == atomic_semantic_level_id("ui", decorated)
    changed = {**level, "selection": (level["selection"] + 1) % len(level["items"])}
    assert atomic_semantic_level_id("ui", level) != atomic_semantic_level_id("ui", changed)


def test_fresh_ui_has_four_real_choices_and_excludes_known_legacy_mechanics():
    legacy = {atomic_semantic_level_id("ui", _easier_level("ui", level, difficulty))
              for bank in atomic_level_banks("ui").values() for level in bank
              for difficulty in (0, 1)}
    fresh = atomic_level_banks("ui", 7193)
    assert all(sorted(level["items"]) == [0, 1, 2, 3]
               for bank in fresh.values() for level in bank)
    assert legacy.isdisjoint({atomic_semantic_level_id("ui", level)
                             for bank in fresh.values() for level in bank})


@pytest.mark.parametrize("layout, key", [(0, "left"), (1, "up")])
def test_four_item_inventory_wraps_to_fourth_item_and_requires_confirmation(layout, key):
    from carryrl.envs.atomic_combat import UIGame

    game = UIGame({"items": [0, 1, 2, 3], "request": 3, "selection": 0,
                   "layout": layout, "request_x": 48, "menu_offset": 0}, 96)
    game.step(PhysicalAction(keys=("tab",)))
    game.step(PhysicalAction(keys=(key,)))
    assert game.selection == 3 and game.score == 2 and game.status == "Play"
    assert game.render().shape == (96, 96, 3)
    game.step(PhysicalAction(keys=("enter",)))
    assert game.confirmed and game.status == "Play"
    game.step(PhysicalAction(keys=("escape",)))
    assert game.status == "Success"


@pytest.mark.parametrize("seed", [-1, True, 1.0, "1", 2**32])
def test_bank_seed_validation_is_strict(seed):
    with pytest.raises(ValueError, match="level_bank_seed"):
        atomic_level_banks("ui", seed)
    with pytest.raises(ValueError, match="level_bank_seed"):
        AtomicEnv("ui", level_bank_seed=seed)


class _FrozenPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.ones(()))

    def reset_episode(self):
        pass

    def act(self, frame, deterministic=True):
        return SimpleNamespace(action=PhysicalAction())


def test_eval_records_new_bank_and_keeps_frozen_state_and_versioned_protocol_hash():
    config = SequenceRunConfig(tasks=("ui",), max_episode_steps=1,
                               eval_all_levels=True, eval_episodes_per_task=1)
    assert "level_bank_seed" not in _environment_options(config)
    fresh = replace(config, level_bank_seed=7193)
    assert _environment_options(fresh)["level_bank_seed"] == 7193
    agent, reward = _FrozenPolicy(), nn.Linear(1, 1)
    agent_hash, reward_hash = tensor_hash(agent), tensor_hash(reward)
    torch_state, python_state, numpy_state = (torch.random.get_rng_state(), random.getstate(),
                                             np.random.get_state())
    legacy = evaluate_sequence(agent, reward, config)
    current = evaluate_sequence(agent, reward, fresh)
    assert tensor_hash(agent) == agent_hash and tensor_hash(reward) == reward_hash
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    final_numpy_state = np.random.get_state()
    assert final_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(final_numpy_state[1], numpy_state[1])
    assert final_numpy_state[2:] == numpy_state[2:]
    legacy_protocol = {
        "split": "validation", "tasks": ["ui"], "counts": {"ui": 16},
        "policies": ["frozen_deterministic", "generic_physical_random"],
        "eval_seed": config.eval_seed, "eval_all_levels": True,
        "max_episode_steps": 1, "difficulty": 1.0, "env_image_size": 96,
        "visual_config": None, "episode_instructions": {},
    }
    assert legacy["protocol_sha256"] != hashlib.sha256(
        json.dumps(legacy_protocol, sort_keys=True).encode()).hexdigest()
    versioned_protocol = {
        **legacy_protocol, "suite_version": "carryrl.atomic.v3",
        "scoring_version": "carryrl.atomic.efficient.v1",
    }
    assert legacy["protocol_sha256"] == hashlib.sha256(
        json.dumps(versioned_protocol, sort_keys=True).encode()).hexdigest()
    assert legacy["protocol_sha256"] != current["protocol_sha256"]
    assert current["level_bank_seed"] == 7193 and len(current["records"]) == 32
    expected = {atomic_semantic_level_id("ui", level)
                for level in atomic_level_banks("ui", 7193)["validation"]}
    assert {row["semantic_level_id"] for row in current["records"]} == expected
    assert all(row["runtime_fingerprint"] == row["semantic_level_id"]
               and row["level_bank_seed"] == 7193 for row in current["records"])
    assert current["summaries"]["frozen_deterministic"]["distinct_semantic_level_ids"] == 16
