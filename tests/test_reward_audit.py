"""Frozen reward auditing keeps independent answers out of the reward input."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from torch import nn

from carryrl.rewards.composition import OutcomePotentialConfig
from carryrl.rewards.visual import LearnedVisualReward, ScreenVisualReward, verify_visible_outcome

spec = importlib.util.spec_from_file_location(
    "audit_reward_script", Path(__file__).parents[1] / "scripts/audit_reward.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class OracleRecognizer(nn.Module):
    """Diagnostic test double; deliberately not a learned agent or real recognizer."""

    def predict(self, frame):
        label = verify_visible_outcome(frame)
        return {"score": label["score"] if label["known"] else 0.0,
            "success_probability": float(label["success"]) if label["known"] else 0.5,
            "failure_probability": float(label["failure"]) if label["known"] else 0.5,
            "score_confidence": 1.0, "known": label["known"]}


def test_audit_replays_train_traces_and_restores_model_mode():
    reward = OracleRecognizer().train()
    report = script.audit_reward(reward, tasks=("navigation", "memory", "goal_choice"),
        levels_per_task=1, parameters=OutcomePotentialConfig(goal_choice_utility="completion_only"))
    assert reward.training
    assert report["status"] == "completed" and report["updates"] == 0
    assert report["policy_decisions"] == 0 and report["corpus_saved"] is False
    assert report["episodes"] == report["episodes_valid_for_candidate_learning"] == 3
    assert report["trajectory_readouts"]["frames"] == sum(
        row["actions"] + 1 for row in report["records"])
    assert all(row["actual_rgb_replay_verified"] for row in report["records"])
    for row in report["records"]:
        assert row["rewards"]["candidate"]["steps"][-1]["outcome"] == 1
        assert row["rewards"]["candidate"]["discounted_return"] > 0
    probes = report["hud_probe_readouts"]
    assert probes["frames"] == 272
    assert probes["known_truth_frames"] == 270
    assert probes["event_correct"] == 272
    assert probes["trusted_score_correct"] == 270
    assert probes["event_confusion"]["unknown->unknown"] == 2
    assert all("ice" not in row["name"] and "ember" not in row["name"]
               and "violet" not in row["name"] for row in report["hud_probes"])


def test_unknown_recognition_is_reported_without_substituting_verifier_answers():
    class Unknown(nn.Module):
        def predict(self, frame):
            return {"score": 0., "success_probability": .5,
                    "failure_probability": .5, "known": False}

    reward = Unknown().eval()
    report = script.audit_reward(reward, tasks=("navigation",), levels_per_task=1)
    assert not reward.training
    assert report["episodes_valid_for_candidate_learning"] == 0
    row = report["records"][0]
    assert row["final_truth"]["success"] is True
    assert row["final_prediction"]["known"] is False
    candidate = row["rewards"]["candidate"]
    assert candidate["component_sums"]["outcome"] == 0
    assert candidate["component_sums"]["progress"] == 0
    assert candidate["component_sums"]["time_cost"] < 0
    assert candidate["steps"][-1]["invalid_reason"] == "terminal_event_unknown"
    assert row["rewards"]["legacy"]["sum"] == 0


def test_audit_uses_replayed_actual_commands_not_recomputed_labels(monkeypatch):
    collect = script.collect_scripted_demonstrations

    def corrupt(**kwargs):
        dataset = collect(**kwargs)
        index = dataset.episodes[0].final_frame_index
        dataset.frames[index] = dataset.frames[index].copy()
        dataset.frames[index][40, 40] ^= 1
        return dataset

    monkeypatch.setattr(script, "collect_scripted_demonstrations", corrupt)
    reward = OracleRecognizer().train()
    with pytest.raises(ValueError, match="next RGB mismatch"):
        script.audit_reward(reward, tasks=("navigation",), levels_per_task=1)
    assert reward.training


def test_goal_choice_utility_is_explicit_before_collecting(monkeypatch):
    monkeypatch.setattr(script, "collect_scripted_demonstrations", lambda **kw: pytest.fail(
        "must reject undeclared goal utility before collection"))
    with pytest.raises(ValueError, match="goal_choice"):
        script.audit_reward(OracleRecognizer(), tasks=("goal_choice",), levels_per_task=1)


def test_audit_refuses_duplicate_train_level_coverage(monkeypatch):
    monkeypatch.setattr(script, "allowed_runtime_train_indices", lambda *a: [0])
    with pytest.raises(ValueError, match="distinct allowed train"):
        script.audit_reward(OracleRecognizer(), tasks=("navigation",), levels_per_task=2)


def test_absence_of_score_confidence_is_not_a_trusted_score():
    row = script._classification({"known": True, "score": 4.,
        "success_probability": 0., "failure_probability": 0.},
        {"known": True, "status": "play", "score": 4.},
        event_threshold=.9, score_threshold=.6)
    assert row["event_correct"] and row["score_correct"]
    assert not row["score_trusted"]


def test_hud_probes_do_not_alias_mutable_frames():
    rows = list(script._hud_probes())
    before = rows[1][1].copy()
    rows[0][1][:] = 0
    assert np.array_equal(rows[1][1], before)


@pytest.mark.parametrize("size", (64, 96, 128))
def test_legacy_recognizer_only_receives_its_actual_input_contract(size):
    reward = LearnedVisualReward(size).train()
    report = script.audit_reward(reward, tasks=("navigation",), levels_per_task=1)
    assert reward.training
    assert report["status"] == "completed"
    assert report["config"]["teacher_image_size"] == size
    assert report["hud_probe_readouts"]["frames"] == 20
    assert report["hud_probe_scope"]["sizes"] == [size]
    assert report["hud_probe_scope"]["train_styles"].startswith("skipped_unsupported")
    assert report["trajectory_readouts"]["trusted_score_frames"] == 0


def test_unsupported_legacy_teacher_size_fails_before_collection(monkeypatch):
    monkeypatch.setattr(script, "collect_scripted_demonstrations", lambda **kw: pytest.fail(
        "unsupported teacher size must fail before collection"))
    with pytest.raises(ValueError, match="fixed-size legacy"):
        script.audit_reward(LearnedVisualReward(80), tasks=("navigation",), levels_per_task=1)


def test_screen_recognizer_checks_all_supported_shapes_despite_legacy_inheritance():
    reward = ScreenVisualReward(96).eval()
    report = script.audit_reward(reward, tasks=("navigation",), levels_per_task=1)
    assert not reward.training
    assert report["hud_probe_readouts"]["frames"] == 272
    assert report["hud_probe_scope"]["sizes"] == [64, 96, 128]
    assert report["hud_probe_scope"]["train_styles"] == "all_train_themes_and_layouts"
