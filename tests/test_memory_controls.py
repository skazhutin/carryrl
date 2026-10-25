"""Paired controls must distinguish RGB memory from blind and action-relay policies."""

import copy
import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import TARGET, _identity, atomic_level_banks, atomic_semantic_level_id
from carryrl.training.sequence_config import SequenceRunConfig

spec = importlib.util.spec_from_file_location(
    "evaluate_memory_controls", Path(__file__).parents[1] / "scripts/evaluate_memory_controls.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class RGBActor(nn.Module):
    """A fake actor whose only observed cue comes from a frozen RGB encoder."""

    def __init__(self, strategy="cue"):
        super().__init__()
        self.vision = nn.Linear(1, 1).requires_grad_(False)
        self.strategy = strategy
        self.replays = []
        self.reset_episode()

    def reset_episode(self, *, instruction=None):
        self.cue = None
        self.previous = PhysicalAction()

    def encode_frames(self, frames):
        encoded = []
        for frame in frames.numpy():
            target = np.argwhere(np.all(frame == TARGET, axis=-1))
            cue = (-1 if not len(target) else int(target[:, 1].mean() >= frame.shape[1] / 2))
            encoded.append(torch.tensor([[float(cue)]]))
        return torch.stack(encoded)

    def act_from_patches(self, patches, deterministic=False):
        assert deterministic
        cue = int(patches[0, 0])
        if cue >= 0:
            self.cue = cue
        if self.strategy == "blind":
            action = PhysicalAction(keys=("left",))
        elif self.strategy == "relay" and cue < 0:
            action = self.previous
        else:
            action = PhysicalAction(keys=("left" if self.cue == 0 else "right",))
        self.previous = action
        return SimpleNamespace(action=action)

    def observe_action(self, action):
        self.previous = action

    def evaluate_episode(self, patches, actions, *, instruction=None, executed_actions=None):
        self.replays.append((actions, executed_actions))
        return {"log_prob": torch.full((len(actions),), -0.7)}


def config():
    return SequenceRunConfig(level_bank_seed=20261008, tasks=("memory",))


def run(strategy="cue", **kwargs):
    actor = RGBActor(strategy)
    reward = nn.Linear(1, 1)
    return diagnostic.evaluate_memory_controls(actor, reward, config(), levels=2, **kwargs), actor


def test_paired_rgb_actor_both_cues_and_actual_wait_commands_leave_bank_unchanged():
    bank = atomic_level_banks("memory", 20261008)
    before = copy.deepcopy(bank)
    report, actor = run()
    assert bank == before
    assert report["policy_and_reward_frozen"]
    assert report["vision_cache"]["hits"] > report["vision_cache"]["misses"]
    for summary in report["summaries"].values():
        assert summary["episodes"] == 4
        assert summary["visible_wins"] == 4
        assert summary["first_choose_correct"] == 4
        assert summary["paired_both_first_choose_correct"] == 2
        assert summary["cue_flip_changes_chosen_direction"] == 2
        assert summary["timeouts"] == 0
        assert summary["first_choose_correct_action_nll_count"] == 4
    normal = report["summaries"]["normal"]
    neutral = report["summaries"]["neutral_wait"]
    assert normal["nonneutral_wait_commands"] > 0
    assert neutral["nonneutral_wait_outputs"] > 0
    assert neutral["nonneutral_wait_commands"] == 0
    neutral_rows = [row for row in report["records"] if row["mode"] == "neutral_wait"]
    neutral_dict = PhysicalAction().to_dict()
    for row in neutral_rows:
        assert row["origin_bank_level_id"] == _identity(before["validation"][row["level_index"]])
        assert row["first_choose"]["step"] == row["unlock_ticks"] >= 66
        wait = [decision for decision in row["decisions"] if decision["phase"] == "Wait"]
        assert all(decision["executed"] == neutral_dict for decision in wait)
        same_geometry = [other for other in neutral_rows
                         if other["level_index"] == row["level_index"]]
        assert len({other["counterfactual_runtime_fingerprint"] for other in same_geometry}) == 2
        assert len({other["origin_bank_level_id"] for other in same_geometry}) == 1
    # Expert labels do not become previous-action inputs in the replay.
    for labels, actual in actor.replays[4:]:
        assert all(action == PhysicalAction() for action in actual[:-1])
        assert labels[:-1] == actual[:-1]
        assert labels[-1].keys in (("left",), ("right",))


def test_blind_actor_cannot_pass_paired_first_choice():
    report, _ = run("blind")
    for summary in report["summaries"].values():
        assert summary["first_choose_correct"] == 2
        assert summary["visible_wins"] == 2
        assert summary["paired_both_first_choose_correct"] == 0
        assert summary["cue_flip_changes_chosen_direction"] == 0


def test_neutral_wait_breaks_previous_action_relay():
    report, _ = run("relay")
    assert report["summaries"]["normal"]["paired_both_first_choose_correct"] == 2
    neutral = report["summaries"]["neutral_wait"]
    assert neutral["first_choose_correct"] == 0
    assert neutral["paired_both_first_choose_correct"] == 0
    assert neutral["visible_wins"] == 0
    assert neutral["timeouts"] == 4
    assert neutral["nonneutral_wait_commands"] == 0


def test_frozen_modes_rng_restored_and_short_budget_records_no_choose():
    actor, reward = RGBActor(), nn.Linear(1, 1)
    actor.train()
    reward.eval()
    torch_before = torch.random.get_rng_state().clone()
    python_before, numpy_before = random.getstate(), np.random.get_state()
    report = diagnostic.evaluate_memory_controls(actor, reward, config(), levels=1, max_steps=2)
    assert actor.training and not reward.training
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert all(row["first_choose"] is None and row["truncated"] for row in report["records"])


def test_replay_api_limitation_is_explicit():
    actor = RGBActor()
    actor.evaluate_episode = None
    report = diagnostic.evaluate_memory_controls(actor, nn.Linear(1, 1), config(), levels=1)
    assert report["first_choose_nll"].startswith("unsupported:")
    assert report["summaries"]["neutral_wait"]["first_choose_correct_action_nll_count"] == 0


@pytest.mark.parametrize("levels,max_steps", [(0, 80), (17, 80), (1, 81), (True, 80)])
def test_rejects_unbounded_or_invalid_budget(levels, max_steps):
    with pytest.raises(ValueError):
        diagnostic.evaluate_memory_controls(RGBActor(), nn.Linear(1, 1), config(),
                                            levels=levels, max_steps=max_steps)


def test_test_geometry_selection_guards_both_cues_and_preserves_bank_indices():
    bank, indices = diagnostic._selected_indices(config(), "test", None)
    banks = atomic_level_banks("memory", config().level_bank_seed)
    train_ids = {atomic_semantic_level_id("memory", level) for level in banks["train"]}
    assert bank is banks["test"]
    assert indices == [index for index in range(16) if index != 10]
    assert all(atomic_semantic_level_id("memory", {**bank[index], "cue": cue}) not in train_ids
               for index in indices for cue in (0, 1))
    assert diagnostic._selected_indices(config(), "validation", None)[1] == list(range(16))
    with pytest.raises(ValueError, match="1..15"):
        diagnostic._selected_indices(config(), "test", 16)


def test_test_short_fixture_reports_selected_bank_and_actual_cue_identity():
    report, _ = run(split="test", max_steps=2)
    assert report["split"] == "test"
    bank = atomic_level_banks("memory", config().level_bank_seed)["test"]
    assert report["bank_sha256"] == _identity(bank)
    assert report["geometry_selection"]["eligible_count"] == 15
    assert report["geometry_selection"]["excluded_level_indices"] == [10]
    assert report["geometry_selection"]["selected_level_indices"] == [0, 1]
    for row in report["records"]:
        level = bank[row["level_index"]]
        assert row["origin_bank_level_id"] == _identity(level)
        assert row["counterfactual_runtime_fingerprint"] == atomic_semantic_level_id(
            "memory", {**level, "cue": row["cue"]})


@pytest.mark.parametrize("split,levels", [("train", None), ("other", 1), ("test", 16)])
def test_invalid_split_or_coverage_fails_before_cli_model_or_output(
        tmp_path, monkeypatch, split, levels):
    output = tmp_path / "report.json"
    args = ["memory", "--checkpoint", str(tmp_path / "missing"), "--config", "unused",
            "--output", str(output), "--split", split]
    if levels is not None:
        args.extend(["--levels", str(levels)])
    monkeypatch.setattr("sys.argv", args)
    monkeypatch.setattr(diagnostic.SequenceRunConfig, "load", lambda _: config())
    def forbidden(*args, **kwargs):
        pytest.fail("invalid selection must not load weights")
    monkeypatch.setattr(diagnostic, "load_sequence_checkpoint", forbidden)
    with pytest.raises((ValueError, SystemExit)):
        diagnostic.main()
    assert not output.exists()


def test_invalid_api_split_fails_before_actor_access():
    with pytest.raises(ValueError, match="held-out"):
        diagnostic.evaluate_memory_controls(None, None, config(), split="train")


def test_cli_test_split_propagates_without_evaluating_policy(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text("{}")
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["memory", "--checkpoint", str(checkpoint),
                                    "--config", "unused", "--output", str(output),
                                    "--split", "test"])
    monkeypatch.setattr(diagnostic.SequenceRunConfig, "load", lambda _: config())
    def frozen_load(*args, **kwargs):
        assert kwargs["evaluation_only"] is True and kwargs["load_optimizer"] is False
        return None, None, None, {"weights_revision": 3}

    monkeypatch.setattr(diagnostic, "load_sequence_checkpoint", frozen_load)
    captured = []
    def fixture_report(agent, reward, cfg, **kwargs):
        captured.append(kwargs)
        bank, indices = diagnostic._selected_indices(cfg, kwargs["split"], kwargs["levels"])
        return {"split": kwargs["split"], "indices": indices,
                "bank_sha256": _identity(bank), "summaries": {}}
    monkeypatch.setattr(diagnostic, "evaluate_memory_controls", fixture_report)
    diagnostic.main()
    assert captured[0]["split"] == "test" and captured[0]["levels"] is None
    assert output.exists()
