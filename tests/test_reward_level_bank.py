"""Fresh reward collection follows the configured disjoint level bank, without fitting."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carryrl.envs.atomic import TASKS, _identity, atomic_level_banks, collect_bootstrap_dataset
from carryrl.envs.presentation import PresentationConfig
from carryrl.training.sequence_config import SequenceRunConfig


@pytest.mark.parametrize("bank_seed", (0, 20261008))
def test_bootstrap_uses_selected_bank_and_records_disjoint_splits(bank_seed):
    collected = {}
    for split in ("train", "validation"):
        _, _, provenance = collect_bootstrap_dataset(
            episodes_per_task=2, max_steps=1, split=split, level_bank_seed=bank_seed)
        assert provenance["level_bank_seed"] == bank_seed
        collected[split] = {episode["level_id"] for episode in provenance["episodes"]}
        for episode in provenance["episodes"]:
            assert episode["level_bank_seed"] == bank_seed
            levels = atomic_level_banks(episode["task"], bank_seed)[split]
            assert episode["level_id"] in {_identity(level) for level in levels}
    assert collected["train"].isdisjoint(collected["validation"])
    if bank_seed:
        legacy_ids = {_identity(level) for task in TASKS
                      for bank in atomic_level_banks(task, 0).values() for level in bank}
        assert (collected["train"] | collected["validation"]).isdisjoint(legacy_ids)


def test_bootstrap_default_preserves_zero_bank():
    options = dict(episodes_per_task=2, max_steps=1, tasks=("mouse",))
    default = collect_bootstrap_dataset(**options)
    explicit = collect_bootstrap_dataset(**options, level_bank_seed=0)
    np.testing.assert_array_equal(default[0], explicit[0])
    assert default[2] == explicit[2]


@pytest.mark.parametrize("bank_seed", (-1, True, 1.0, 2**32))
def test_bootstrap_rejects_invalid_bank_seed(bank_seed):
    with pytest.raises(ValueError, match="level_bank_seed"):
        collect_bootstrap_dataset(episodes_per_task=2, level_bank_seed=bank_seed)


def test_screen_reward_forwards_bank_and_records_report_without_fitting(tmp_path, monkeypatch):
    import carryrl.envs.atomic as atomic
    import carryrl.rewards.pretrain as pretrain

    original = atomic.collect_bootstrap_dataset
    calls = []

    def collect(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(atomic, "collect_bootstrap_dataset", collect)
    monkeypatch.setattr(pretrain, "train_visual_reward", lambda *a, **kw: {"stubbed": True})
    monkeypatch.setattr(pretrain, "evaluate_screen_reward", lambda *a: {})
    _, report = pretrain.pretrain_screen_reward(
        tmp_path / "reward", tasks=("mouse",), episodes_per_task=2,
        max_steps=1, level_bank_seed=20261008,
        presentation_config=PresentationConfig(
            mode="varied", native_sizes=(96,), themes=("slate", "violet"),
            hud_layouts=("top",)))
    assert [(call["split"], call["level_bank_seed"]) for call in calls] == [
        ("train", 20261008), ("validation", 20261008)]
    assert report["level_bank_seed"] == 20261008
    assert set(report["train_level_ids"]).isdisjoint(report["validation_level_ids"])
    stored = json.loads((tmp_path / "reward/reward_report.json").read_text())
    assert stored["level_bank_seed"] == report["level_bank_seed"]
    assert stored["train_level_ids"] == report["train_level_ids"]
    assert stored["validation_level_ids"] == report["validation_level_ids"]
    checkpoint = torch.load(tmp_path / "reward/reward_checkpoint.pt", weights_only=True)
    assert checkpoint["report"]["level_bank_seed"] == 20261008
    for source in (report, stored, checkpoint["report"]):
        assert source["initialization"] == "random"
        assert source["imported_checkpoints"] == []
        assert source["old_corpora"] == []


def test_cli_and_sequence_fresh_screen_reward_forward_configured_bank(tmp_path, monkeypatch):
    import carryrl.cli as cli
    import carryrl.rewards.pretrain as pretrain
    from carryrl.training.sequence import prepare_reward

    config = SequenceRunConfig(level_bank_seed=20261008, reward_model_mode="screen_v2")
    calls = []

    def prepare(output, **kwargs):
        calls.append(kwargs)
        return torch.nn.Linear(1, 1), {
            "model_version": "stub", "tasks": list(config.tasks),
            "level_bank_seed": kwargs["level_bank_seed"], "gate": {"passed": True}}

    monkeypatch.setattr(pretrain, "pretrain_screen_reward", prepare)
    monkeypatch.setattr(SequenceRunConfig, "load", lambda path: config)
    monkeypatch.setattr(cli.torch, "set_num_threads", lambda _: None)
    args = SimpleNamespace(config=tmp_path / "config.json", seed=None, device=None,
                           output=tmp_path / "cli_reward")
    assert cli._main_reward(args) == 0
    prepare_reward(config, tmp_path)
    assert [call["level_bank_seed"] for call in calls] == [20261008, 20261008]
    report = json.loads((tmp_path / "reward_bootstrap.json").read_text())
    assert report["level_bank_seed"] == 20261008


def test_sequence_legacy_reward_collects_both_configured_banks(tmp_path, monkeypatch):
    import carryrl.envs.atomic as atomic
    import carryrl.rewards.visual as visual
    import carryrl.training.sequence as sequence

    original = atomic.collect_bootstrap_dataset
    calls = []

    def collect(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    config = SequenceRunConfig(level_bank_seed=20261008, tasks=("mouse",),
                               max_episode_steps=1, reward_bootstrap_episodes_per_task=2)
    monkeypatch.setattr(atomic, "collect_bootstrap_dataset", collect)
    monkeypatch.setattr(visual, "train_visual_reward", lambda *a, **kw: {"stubbed": True})
    monkeypatch.setattr(sequence, "_reward_validation", lambda *a: {})
    sequence.prepare_reward(config, tmp_path, lambda size: torch.nn.Linear(1, 1))
    assert [(call["split"], call["level_bank_seed"]) for call in calls] == [
        ("train", 20261008), ("validation", 20261008)]
    report = json.loads((tmp_path / "reward_bootstrap.json").read_text())
    for key in ("fit_labels_provenance", "validation_labels_provenance"):
        assert report[key]["level_bank_seed"] == 20261008
