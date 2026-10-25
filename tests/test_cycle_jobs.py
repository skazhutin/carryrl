import json

import pytest
import torch

from carryrl.envs.atomic_suite import TASKS
from carryrl.training.cycle_jobs import (
    _no_progress_at_two_evaluations,
    inspect_initial_checkpoint,
    normalize_evaluation,
    policy_selection_report,
)
from carryrl.training.run import file_sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def initial(tmp_path, *, counter=0, optimizer=None, critic=0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    state = {"rollout": counter, "decisions": 0, "weights_revision": 0,
             "optimizer_state": {"state": optimizer or {}},
             "agent_state": {"vision.weight": torch.ones(2),
                             "language.weight": torch.ones(2),
                             "decoder.weight": torch.ones(2),
                             "value_head.weight": torch.full((1, 2), float(critic)),
                             "value_head.bias": torch.zeros(1)}}
    torch.save(state, tmp_path / "state.pt")
    write(tmp_path / "manifest.json", {"sha256": file_sha256(tmp_path / "state.pt")})
    return tmp_path


def evaluation(success=0):
    return {"split": "validation", "records": [
        {"task": task, "policy": "frozen_stochastic",
         "outcome": {"known": True, "success": i < success}}
        for task in TASKS for i in range(16)]}


def test_cp0_authentication_and_pairing(tmp_path):
    a = inspect_initial_checkpoint(initial(tmp_path / "a"))
    b = inspect_initial_checkpoint(initial(tmp_path / "b"), expected=a)
    assert a["initial_trainable_sha256"] == b["initial_trainable_sha256"]
    assert a["frozen_sha256"] == b["frozen_sha256"]
    with pytest.raises(ValueError, match="paired CP0"):
        inspect_initial_checkpoint(tmp_path / "b", expected={**a, "frozen_sha256": "0" * 64})


@pytest.mark.parametrize("kwargs", [dict(counter=1), dict(optimizer={0: {"step": 1}}),
                                    dict(critic=1)])
def test_cp0_rejects_used_weights_optimizer_and_random_critic(tmp_path, kwargs):
    with pytest.raises(ValueError):
        inspect_initial_checkpoint(initial(tmp_path, **kwargs))


def test_normalization_requires_full_known_per_game_cohort(tmp_path):
    path = tmp_path / "eval.json"
    write(path, evaluation(1))
    result = normalize_evaluation(path)
    assert result["successes"] == 16 and result["episodes"] == 256
    report = evaluation()
    report["records"][0]["outcome"]["known"] = False
    write(path, report)
    with pytest.raises(ValueError, match="incomplete"):
        normalize_evaluation(path)


def test_successful_runtime_levels_not_episode_repeats(tmp_path):
    for folder in ("evaluations", "evaluations_pixel"):
        write(tmp_path / folder / "rollout_0032.json", evaluation(1))
    episode = {"task": "mouse", "outcome": {"success": True},
               "provenance": {"curriculum": {"runtime_id": "same-runtime"}}}
    write(tmp_path / "training.jsonl", {"rollout": 32, "episode_records": [episode, episode]})
    result = policy_selection_report(tmp_path, rollout=32,
                                     initial={"initial_trainable_sha256": "a" * 64}, seed=3176)
    assert result["rollouts"] == 32
    assert result["train"]["games"]["mouse"]["successful_effective_levels"] == ["same-runtime"]


def test_registered_stall_rule_respects_actual_stage_advancement(tmp_path):
    for folder in ("evaluations", "evaluations_pixel"):
        for step in (0, 64, 128):
            write(tmp_path / folder / f"rollout_{step:04d}.json", evaluation(1))
    path = tmp_path / "training.jsonl"
    write(path, {"task_curriculum": {"tasks": {"mouse": {"stage": 0}}}})
    assert _no_progress_at_two_evaluations(tmp_path)
    write(path, {"task_curriculum": {"tasks": {"mouse": {"stage": 1}}}})
    assert not _no_progress_at_two_evaluations(tmp_path)
