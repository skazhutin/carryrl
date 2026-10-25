"""Reference-denominator metrics and frozen, whitelist-only history diagnostics."""

import copy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from carryrl.evaluation.history_reward import DEFAULT_CONTROLS, evaluate_history_reward


class Corpus:
    event_names = ("pickup", "unobservable")
    corpus_sha256 = "a" * 64

    def __init__(self):
        self.episodes = [
            {
                "split": "train",
                "task": "navigation",
                "strategy": "teacher",
                "runtime_group": "train-level",
            },
            {
                "split": "validation",
                "task": "navigation",
                "strategy": "random",
                "runtime_group": "validation-level",
            },
        ]
        self.labels = {
            "event_names": self.event_names,
            "potential": np.asarray([0, 0.5, 0.5, np.nan], np.float32),
            "potential_valid": np.asarray([True, True, True, False]),
            "events": np.asarray(
                [[0, np.nan], [1, np.nan], [1, np.nan], [np.nan, np.nan]], np.float32
            ),
            "event_valid": np.asarray(
                [[True, False], [True, False], [True, False], [False, False]]
            ),
            "terminal_success": np.asarray([0, 0, 0, 1], np.float32),
            "terminal_success_valid": np.ones(4, bool),
        }

    def targets(self, index):
        return self.labels


class Features:
    feature_width, goal_width = 2, 3

    def __init__(self, corpus):
        self.identity = {
            "corpus_sha256": corpus.corpus_sha256,
            "event_names": list(corpus.event_names),
        }
        actions = torch.zeros(1, 4, 33)
        actions[0, 1:, 0] = 1
        self.cached = {
            "scene_features": torch.tensor([[[0.25, 0.4], [0.75, 0.3], [0.25, 0.2], [0.75, 0.1]]]),
            "executed_previous_actions": actions,
            "explicit_goal_features": torch.tensor([[0.1, 0.2, 0.3]]),
        }

    def model_inputs(self, corpus, index, device="cpu"):
        # Deliberately return shared cache to ensure controls cannot mutate it.
        return self.cached


class SyntheticHead(nn.Module):
    event_names = Corpus.event_names
    scene_feature_size, goal_feature_size = 2, 3

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.child = nn.Dropout()
        self.calls = []

    def forward(self, scene_features, executed_previous_actions, explicit_goal_features):
        assert not self.training and not self.weight.requires_grad and not torch.is_grad_enabled()
        assert not (explicit_goal_features == 0).all(dim=-1).any()
        assert (executed_previous_actions[:, 0] == 0).all()
        self.calls.append(
            {
                "scene_features": scene_features.clone(),
                "executed_previous_actions": executed_previous_actions.clone(),
                "explicit_goal_features": explicit_goal_features.clone(),
            }
        )
        # Exercise state restoration even for unusual stochastic eval modules.
        random.random()
        np.random.random()
        torch.rand(1)
        potential = scene_features[..., 0].clamp(0.01, 0.99)
        event = torch.logit(potential)[..., None].expand(-1, -1, 2)
        return {
            "potential": potential,
            "event_logits": event,
            "terminal_success_logits": torch.full_like(potential, -10),
            "potential_valid_logits": torch.full_like(potential, -10),
            "event_valid_logits": torch.full_like(event, -10),
            "terminal_valid_logits": torch.full_like(potential, -10),
        }


def setup():
    corpus = Corpus()
    return SyntheticHead(), corpus, Features(corpus)


def test_exact_metrics_use_reference_validity_and_ignore_unknown_nan():
    head, corpus, features = setup()
    report = evaluate_history_reward(head, corpus, features, controls=("full",))
    metrics = report["controls"]["full"]["global"]
    event = metrics["events"]["pickup"]
    assert {name: event[name] for name in ("tp", "fp", "fn", "tn")} == {
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "tn": 1,
    }
    assert event["known"] == 3 and event["unknown"] == 1
    assert event["precision"] == 1 and event["recall"] == 0.5
    assert event["brier"] == pytest.approx((0.25**2 * 2 + 0.75**2) / 3)
    assert event["bce"] == pytest.approx((-2 * np.log(0.75) - np.log(0.25)) / 3)
    assert metrics["events"]["unobservable"]["precision"] is None
    assert metrics["events"]["unobservable"]["bce"] is None
    potential = metrics["potential"]
    assert potential["known"] == 3
    assert potential["mae"] == potential["rmse"] == 0.25
    assert potential["changing_adjacent_pairs"] == 1 and potential["sign_agreement"] == 1
    assert potential["equal_reference_adjacent_pairs"] == 1
    assert potential["equal_reference_absolute_drift"] == 0.5
    assert potential["direction"]["up"] == {"pairs": 1, "agreement": 1.0}
    assert potential["direction"]["flat"] == {"pairs": 1, "agreement": 0.0}
    terminal = metrics["terminal"]
    assert terminal["all_frames"]["accuracy"] == 0.75
    assert terminal["final_frames"]["accuracy"] == 0
    assert terminal["final_frames"]["fn"] == 1
    assert metrics["validity"]["potential"]["coverage"] == 0
    assert metrics["validity"]["potential"]["proposed_abstentions_known"] == 3
    assert metrics["validity"]["potential"]["proposed_abstentions_unknown"] == 1
    assert report["head_sha256_before"] == report["head_sha256_after"]
    json.dumps(report, allow_nan=False)


def test_controls_change_only_whitelisted_modalities_and_keep_cohort():
    head, corpus, features = setup()
    original = {name: value.clone() for name, value in features.cached.items()}
    report = evaluate_history_reward(head, corpus, features)
    calls = dict(zip(DEFAULT_CONTROLS, head.calls, strict=True))
    assert all(torch.equal(calls["full"][name], value) for name, value in original.items())
    for name in ("no_scene", "time_only"):
        assert calls[name]["scene_features"].count_nonzero() == 0
    for name in ("no_actions", "time_only"):
        assert calls[name]["executed_previous_actions"].count_nonzero() == 0
    constant = torch.ones(1, 3) / np.sqrt(3)
    for name in ("constant_goal", "time_only"):
        assert torch.allclose(calls[name]["explicit_goal_features"], constant)
    for control, changed in (
        ("no_scene", "scene_features"),
        ("no_actions", "executed_previous_actions"),
        ("constant_goal", "explicit_goal_features"),
    ):
        assert all(
            torch.equal(calls[control][name], value)
            for name, value in original.items()
            if name != changed
        )
    single = calls["single_frame"]
    assert single["scene_features"].shape == (4, 1, 2)
    assert torch.equal(single["scene_features"][:, 0], original["scene_features"][0])
    assert single["executed_previous_actions"].count_nonzero() == 0
    assert torch.equal(
        single["explicit_goal_features"], original["explicit_goal_features"].expand(4, -1)
    )
    assert all(torch.equal(features.cached[name], value) for name, value in original.items())
    assert {row["cohort_sha256"] for row in report["controls"].values()} == {
        report["cohort_sha256"]
    }
    assert all(row["global"]["potential"]["known"] == 3 for row in report["controls"].values())


def test_model_modes_grad_flags_and_rng_restored_even_after_failure():
    head, corpus, features = setup()
    head.train()
    head.child.eval()
    head.weight.requires_grad_(False)
    for fail in (False, True):
        python_state, numpy_state, torch_state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )

        def progress(_):
            if fail:
                raise RuntimeError("callback failure")

        if fail:
            with pytest.raises(RuntimeError, match="callback failure"):
                evaluate_history_reward(head, corpus, features, on_progress=progress)
        else:
            evaluate_history_reward(head, corpus, features, on_progress=progress)
        assert head.training and not head.child.training and not head.weight.requires_grad
        assert random.getstate() == python_state
        assert np.array_equal(np.random.get_state()[1], numpy_state[1])
        assert np.random.get_state()[2:] == numpy_state[2:]
        assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("test_request", "Only train or validation"),
        ("empty", "no episodes"),
        ("split", "unknown/test split"),
        ("overlap", "overlap"),
        ("corpus", "source corpus identity"),
        ("feature_events", "vocabulary/order"),
        ("head_events", "vocabulary/order"),
        ("target_events", "vocabulary/order"),
        ("input_metadata", "only scene"),
        ("known_nan", "known reference"),
    ],
)
def test_identity_and_cohort_fail_closed(mutation, match):
    head, corpus, features = setup()
    kwargs = {"controls": ("full",)}
    if mutation == "test_request":
        kwargs["split"] = "test"
    elif mutation == "empty":
        corpus.episodes.pop()
    elif mutation == "split":
        corpus.episodes[0]["split"] = "test"
    elif mutation == "overlap":
        corpus.episodes[1]["runtime_group"] = corpus.episodes[0]["runtime_group"]
    elif mutation == "corpus":
        features.identity["corpus_sha256"] = "b" * 64
    elif mutation == "feature_events":
        features.identity["event_names"].reverse()
    elif mutation == "head_events":
        head.event_names = tuple(reversed(head.event_names))
    elif mutation == "target_events":
        corpus.labels["event_names"] = tuple(reversed(head.event_names))
    elif mutation == "input_metadata":
        features.cached["task_id"] = torch.tensor(3)
    elif mutation == "known_nan":
        corpus.labels["potential"][0] = np.nan
    with pytest.raises(ValueError, match=match):
        evaluate_history_reward(head, corpus, features, **kwargs)


def test_macro_episodes_and_games_do_not_reduce_to_frame_weighted_global():
    head, corpus, features = setup()
    corpus.episodes.append(
        {
            "split": "validation",
            "task": "memory",
            "strategy": "teacher",
            "runtime_group": "other-validation-level",
        }
    )
    second = copy.deepcopy(corpus.labels)
    for name, value in second.items():
        if isinstance(value, np.ndarray):
            second[name] = value[:1].copy()
    second["potential"][0] = 1
    corpus.targets = lambda index: second if index == 2 else corpus.labels
    original_model_inputs = features.model_inputs

    def inputs(corpus, index, device="cpu"):
        values = original_model_inputs(corpus, index, device)
        if index != 2:
            return values
        return {name: value[:, :1] if value.ndim == 3 else value for name, value in values.items()}

    features.model_inputs = inputs
    report = evaluate_history_reward(head, corpus, features, controls=("full",))["controls"]["full"]
    assert report["global"]["potential"]["mae"] == 0.375
    assert report["macro_episodes"]["metrics"]["potential"]["mae"] == {
        "mean": 0.5,
        "defined_units": 2,
    }
    assert report["macro_games"]["metrics"]["potential"]["mae"]["mean"] == 0.5
    assert set(report["per_game"]) == {"navigation", "memory"}
    assert set(report["per_strategy"]) == {"teacher", "random"}


def test_real_history_head_all_controls_are_frozen():
    from carryrl.rewards.history import SceneHistoryHead

    _, corpus, features = setup()
    head = SceneHistoryHead(
        scene_feature_size=2, goal_feature_size=3, hidden_size=5, event_names=corpus.event_names
    )
    report = evaluate_history_reward(head, corpus, features)
    assert set(report["controls"]) == set(DEFAULT_CONTROLS)
    assert report["head_sha256_before"] == report["head_sha256_after"]
    assert head.training and all(parameter.requires_grad for parameter in head.parameters())


def test_mutating_model_tensors_is_detected_and_modes_restored():
    class MutatingHead(SyntheticHead):
        def forward(self, **kwargs):
            result = super().forward(**kwargs)
            self.weight.add_(1)
            return result

    _, corpus, features = setup()
    head = MutatingHead()
    with pytest.raises(RuntimeError, match="modified model tensors"):
        evaluate_history_reward(head, corpus, features, controls=("full",))
    assert head.training and head.weight.requires_grad
