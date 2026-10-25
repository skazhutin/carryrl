"""Actual split-before-collection probes, schema-bound losses and replay integrity."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from carryrl.agent.actions import PhysicalAction, encode_actions
from carryrl.envs.atomic import atomic_level_banks
from carryrl.envs.atomic_suite import TASKS
from carryrl.envs.runtime_identity import atomic_runtime_level_id
from carryrl.evaluation.scene_suite import EVENT_NAMES, GOALS, SuiteSceneAnnotator
from carryrl.rewards.history import SceneHistoryHead, history_reward_loss
from carryrl.rewards.history_suite_data import (
    PROBE_STRATEGIES,
    _identity,
    _report,
    _select_levels,
    collect_history_suite,
    load_history_suite,
)
from carryrl.training.run import atomic_json


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    output = tmp_path_factory.mktemp("scene_suite") / "corpus"
    report = collect_history_suite(
        output, tasks=("navigation", "memory", "chain", "ui"), levels_per_split=1
    )
    return output, report


def test_registry_covers_all_tasks_and_requires_explicit_goal_utility():
    assert len(EVENT_NAMES) == len(set(EVENT_NAMES)) == 21
    assert set(GOALS) == set(TASKS)
    for task in TASKS:
        parser = SuiteSceneAnnotator(task, goal_choice_utility="completion_only")
        row = parser.observe(np.zeros((96, 96, 3), np.uint8))
        assert row["known"] is False
        assert set(row["event_valid"]) == set(EVENT_NAMES)
        assert not any(row["event_valid"].values())
        assert row["potential_valid"] is False
    with pytest.raises(ValueError, match="completion_only"):
        SuiteSceneAnnotator("goal_choice")


def test_real_probes_replay_and_groups_stay_in_their_split(cohort):
    output, report = cohort
    corpus = load_history_suite(output, expected_sha256=report["corpus_sha256"])
    assert report["episodes"] == 24
    assert report["runtime_groups"] == {"train": 4, "validation": 4}
    assert report["runtime_group_overlap"] == []
    assert report["terminal_label_disagreements"] == report["premature_terminal_labels"] == []
    assert report["valid_final_terminal_labels"] > 0
    assert not report["model_fit"] and not report["ppo_eligible"] and report["updates"] == 0
    assert not report["test_levels_played"] and not report["test_outcomes_read"]
    assert corpus.event_names == EVENT_NAMES
    groups = {}
    for index, episode in enumerate(corpus.episodes):
        groups.setdefault(episode["runtime_group"], set()).add(episode["split"])
        for step in (0, len(episode["frame_indices"]) - 1):
            inputs = corpus.prefix(index, step)
            assert set(inputs) == {"frames", "executed_previous_actions", "goal"}
            assert inputs["frames"].shape == (step + 1, 96, 96, 3)
            assert (inputs["frames"][:, :28] == 0).all()
            commands = [PhysicalAction.from_dict(row) for row in episode["executed_actions"][:step]]
            assert torch.equal(
                inputs["executed_previous_actions"], encode_actions([PhysicalAction(), *commands])
            )
        assert corpus.targets(index)["event_names"] == EVENT_NAMES
    assert all(len(splits) == 1 for splits in groups.values())
    assert {row["strategy"] for row in corpus.episodes} == set(PROBE_STRATEGIES)
    assert any(row["evaluation_only"]["outcome"]["success"] is not True for row in corpus.episodes)
    assert any(row["evaluation_only"]["interventions"] > 0 for row in corpus.episodes)


def test_mixed_vocabulary_head_loss_binds_target_names_and_order(cohort):
    output, report = cohort
    corpus = load_history_suite(output, expected_sha256=report["corpus_sha256"])
    labels = corpus.targets(0)
    targets = {
        name: torch.from_numpy(value).unsqueeze(0) if isinstance(value, np.ndarray) else value
        for name, value in labels.items()
    }
    steps = len(targets["potential"][0])
    model = SceneHistoryHead(
        scene_feature_size=3, goal_feature_size=4, hidden_size=8, event_names=EVENT_NAMES
    )
    inputs = corpus.prefix(0, steps - 1)
    # Tiny frozen scene features for a computation check, not fitted recognition.
    scene = torch.from_numpy(inputs["frames"]).float().mean(dim=(1, 2))[None] / 255
    predictions = model(scene, inputs["executed_previous_actions"][None], torch.ones(1, 4))
    assert predictions["event_logits"].shape == (1, steps, 21)
    loss = model.supervised_loss(predictions, targets)
    assert torch.isfinite(loss)
    loss.backward()
    with pytest.raises(ValueError, match="vocabulary/order"):
        history_reward_loss(predictions, targets)
    with pytest.raises(ValueError, match="vocabulary/order"):
        model.supervised_loss(predictions, {**targets, "event_names": tuple(reversed(EVENT_NAMES))})
    assert SceneHistoryHead().event_names != EVENT_NAMES  # Original six-event default is preserved.


@pytest.mark.parametrize(
    "mutation",
    (
        "annotation",
        "split",
        "group",
        "seed",
        "action",
        "strategy",
        "missing",
        "vocabulary",
        "config",
    ),
)
def test_resigned_file_does_not_bypass_replay_or_preregistered_cohort(cohort, tmp_path, mutation):
    source, _ = cohort
    directory = tmp_path / "tampered"
    directory.mkdir()
    shutil.copyfile(source / "frames.npy", directory / "frames.npy")
    payload = json.loads((source / "episodes.json").read_text())
    row = payload["episodes"][0]
    if mutation == "annotation":
        row["annotations"][0]["potential"] = 0.1234
    elif mutation == "split":
        row["split"] = "validation"
    elif mutation == "group":
        row["runtime_group"] = "fake"
    elif mutation == "seed":
        row["seed"] += 1
    elif mutation == "action":
        row["executed_actions"][0] = PhysicalAction().to_dict()
    elif mutation == "strategy":
        row["strategy"] = "neutral"
    elif mutation == "missing":
        payload["episodes"].pop()
    elif mutation == "vocabulary":
        payload["event_names"].reverse()
    else:
        payload["config"]["max_steps"] = True
    atomic_json(directory / "episodes.json", payload, indent=None)
    with pytest.raises(ValueError):
        load_history_suite(directory, expected_sha256=_identity(directory)[0])


def test_wrong_pin_fails_before_episode_replay(cohort, monkeypatch):
    import carryrl.rewards.history_suite_data as data

    output, _ = cohort
    monkeypatch.setattr(data, "AtomicEnv", lambda *a, **kw: pytest.fail("SHA must fail first"))
    with pytest.raises(ValueError, match="SHA256"):
        load_history_suite(output, expected_sha256="0" * 64)


@pytest.mark.parametrize("premature", (False, True))
def test_diagnostic_outcome_gate_detects_false_scene_terminal_labels(cohort, premature):
    output, report = cohort
    corpus = load_history_suite(output, expected_sha256=report["corpus_sha256"])
    episode = corpus.episodes[0]
    if premature:
        episode["annotations"][0]["terminal_success_valid"] = True
        episode["annotations"][0]["terminal_success"] = True
    else:
        final = episode["annotations"][-1]
        final["terminal_success_valid"] = True
        final["terminal_success"] = not episode["evaluation_only"]["outcome"]["success"]
    with pytest.raises(ValueError, match="independent diagnostic outcomes"):
        _report(corpus.episodes)


def test_collection_validates_settings_and_split_preflight_before_output(tmp_path, monkeypatch):
    import carryrl.rewards.history_suite_data as data

    output = tmp_path / "corpus"
    for kwargs in (
        {"tasks": ("goal_choice",)},
        {"levels_per_split": True},
        {"max_steps": 129},
        {"goal_choice_utility": "invented"},
    ):
        with pytest.raises(ValueError):
            collect_history_suite(output, **kwargs)
        assert not output.exists()
    monkeypatch.setattr(
        data,
        "_select_levels",
        lambda *a: (_ for _ in ()).throw(ValueError("no distinct safe levels")),
    )
    with pytest.raises(ValueError, match="safe levels"):
        collect_history_suite(output, tasks=("navigation",))
    assert not output.exists()


@pytest.mark.parametrize(
    "task,seed", (("steering", 26), ("steering", 51), ("ui", 16), ("ui", 19), ("ui", 50))
)
def test_train_and_validation_never_play_other_protected_banks_test_alias(task, seed):
    selected = _select_levels((task,), 20261008, 1, seed)[task]
    protected = {
        atomic_runtime_level_id(task, level)
        for bank_seed in (0, 20261008, 20261015)
        for level in atomic_level_banks(task, bank_seed)["test"]
    }
    bank = atomic_level_banks(task, 20261008)
    for split in ("train", "validation"):
        assert all(
            atomic_runtime_level_id(task, bank[split][index]) not in protected
            for index in selected[split]
        )


def test_cli_runs_actual_probes_without_model_and_protects_existing_output(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "collect_history_suite_script",
        Path(__file__).parents[1] / "scripts/collect_history_suite.py",
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "provenance", lambda: {})
    output = tmp_path / "corpus"
    args = ["--output", str(output), "--tasks", "navigation", "--levels-per-split", "1"]
    assert script.main(args) == 0
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "completed" and status["episodes"] == 6
    assert status["updates"] == 0 and not status["ppo_eligible"]
    assert json.loads((output / "metadata.json").read_text())["model_loaded"] is False
    with pytest.raises(FileExistsError):
        script.main(args)


def test_compact_atomic_json_preserves_default_format_and_nan_rejection(tmp_path):
    default, compact = tmp_path / "default.json", tmp_path / "compact.json"
    record = {"names": list(EVENT_NAMES), "targets": [False] * 21}
    atomic_json(default, record)
    atomic_json(compact, record, indent=None)
    assert default.read_text() == json.dumps(record, indent=2) + "\n"
    assert json.loads(compact.read_text()) == record
    assert compact.stat().st_size < default.stat().st_size
    with pytest.raises(ValueError):
        atomic_json(compact, {"bad": float("nan")}, indent=None)
    assert json.loads(compact.read_text()) == record
