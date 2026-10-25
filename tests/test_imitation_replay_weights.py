"""Loss-only replay weighting with tiny CPU policies and authenticated RGB provenance."""

import importlib.util
import json
import random
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from carryrl.agent.actions import PhysicalAction
from carryrl.training.imitation import (
    CORRECTION_SOURCE,
    _episode_nlls,
    _replay_domain,
    _replay_normalization,
    _replay_objective,
    augment_demonstrations,
    collect_memory_cue_counterparts,
    collect_native_scripted_demonstrations,
    collect_scripted_demonstrations,
    imitation_replay_summary,
    train_imitation,
)

spec = importlib.util.spec_from_file_location(
    "imitation_replay_script", Path(__file__).resolve().parents[1] / "scripts/train_imitation.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.tensor(0.25))
        self.vision = torch.nn.Identity()
        self.config = SimpleNamespace(segment_length=2)
        self.seen = []

    def evaluate_episode(self, patches, actions, *, instruction=None, executed_actions=None):
        self.seen.append((tuple(actions), instruction, executed_actions))
        return {"log_prob": -(self.scalar - patches[:, 0, 0]).square()}


@pytest.fixture
def replay_dataset():
    dataset = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=1)
    dataset.extend(collect_native_scripted_demonstrations(
        native_train_sizes=(64, 128), native_episodes_per_task=1, tasks=("navigation",),
        difficulty=1))
    dataset.extend(augment_demonstrations(dataset, copies_per_episode=1, seed=23))
    return dataset


def features_for(dataset):
    return {index: torch.full((1, 384), (index + 1) / 20)
            for index in range(len(dataset.frames))}


def parts_for(agent, dataset, features, action_balance="uniform"):
    return [_episode_nlls(agent.evaluate_episode(
        torch.stack([features[index] for index in episode.frame_indices]), episode.actions,
        instruction=episode.instruction, executed_actions=episode.executed_actions)["log_prob"],
        episode.actions, action_balance) for episode in dataset.episodes]


def test_domains_use_actual_rgb_and_explicit_appearance_native_provenance(replay_dataset):
    dataset = replay_dataset
    teacher = dataset.episodes[0]
    dataset.add_episode(replace(teacher, source=CORRECTION_SOURCE,
        executed_actions=teacher.actions, terminated=True, truncated=False,
        expert_probability=1.0))
    memory = collect_scripted_demonstrations(1, tasks=("memory",), difficulty=1,
                                             level_bank_seed=20261008)
    counterparts = collect_memory_cue_counterparts(memory, max_pairs=1)
    assert len(counterparts.episodes) == 1
    dataset.extend(counterparts)
    # Externally assembled teacher metadata alone does not establish physical 96 RGB.
    dataset.add_episode(replace(dataset.episodes[1], source="scripted_teacher", native_size=None,
        origin_level_id=None, runtime_level_id=None))
    metadata_before = deepcopy(dataset.provenance())
    frames_before = [frame.tobytes() for frame in dataset.frames]
    summary = imitation_replay_summary(dataset, canonical_replay_weight=4.0)
    domains = summary["domains"]
    assert domains["canonical_unstyled_96"] == {
        "episodes": 3, "episode_weight": 4.0, "effective_episode_mass": 12.0,
        "effective_mass_fraction": 12 / 18,
    }
    assert domains["appearance_copy"]["episodes"] == 3
    for name in ("appearance_copy", "native_64", "native_128", "other_unstyled_rgb"):
        assert domains[name]["episode_weight"] == 1.0
    assert _replay_domain(dataset, dataset.episodes[3]) == "appearance_copy"
    assert dataset.provenance() == metadata_before
    assert [frame.tobytes() for frame in dataset.frames] == frames_before


@pytest.mark.parametrize("explicit", [False, True])
def test_default_replay_weight_preserves_original_mean_update_exactly(replay_dataset, explicit):
    dataset, agent = replay_dataset, TinyPolicy()
    reference = deepcopy(agent)
    features = features_for(dataset)
    order = list(range(len(dataset.episodes)))
    random.Random(0).shuffle(order)
    shuffled = SimpleNamespace(episodes=[dataset.episodes[index] for index in order])
    expected = torch.stack([part[0] for part in parts_for(reference, shuffled, features)]).mean()
    expected.backward()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    reference_optimizer.step()
    optimizer = torch.optim.SGD(agent.parameters(), lr=0.01)
    result = train_imitation(agent, dataset, optimizer=optimizer,
        minibatch_episodes=len(dataset.episodes), max_grad_norm=1e6,
        feature_provider=features.__getitem__,
        **({"canonical_replay_weight": 1.0} if explicit else {}))
    assert torch.equal(agent.scalar, reference.scalar)
    assert torch.equal(agent.scalar.grad, reference.scalar.grad)
    assert result["history"][0]["objective"] == pytest.approx(float(expected.detach()), abs=1e-7)
    assert result["history"][0]["objective"] == result["history"][0]["nll"]
    assert result["replay_weighting"]["canonical_replay_weight"] == 1.0


@pytest.mark.parametrize("action_balance", ["uniform", "active_idle"])
def test_weighted_gradient_objective_and_raw_diagnostics_are_distinct(
        replay_dataset, action_balance):
    dataset, agent = replay_dataset, TinyPolicy()
    features, reference = features_for(dataset), deepcopy(agent)
    parts = parts_for(reference, dataset, features, action_balance)
    weights = [3.0] + [1.0] * (len(parts) - 1)
    expected = sum(weight * part[0] for weight, part in zip(weights, parts, strict=True))
    expected = expected / sum(weights)
    expected.backward()
    expected_nll = float(torch.stack([part[1] for part in parts]).mean().detach())
    updates = []
    result = train_imitation(agent, dataset, optimizer=torch.optim.SGD(agent.parameters(), lr=.01),
        minibatch_episodes=len(parts), max_grad_norm=1e6, canonical_replay_weight=3.0,
        action_balance=action_balance, feature_provider=features.__getitem__,
        on_update=updates.append)
    torch.testing.assert_close(agent.scalar.grad, reference.scalar.grad, atol=1e-6, rtol=1e-6)
    row = result["history"][0]
    assert row["nll"] == pytest.approx(expected_nll, abs=1e-6)
    assert row["per_task_nll"]["navigation"] == row["nll"]
    assert row["objective"] == pytest.approx(float(expected.detach()), abs=1e-6)
    assert updates[0]["objective"] == pytest.approx(float(expected.detach()), abs=1e-6)
    assert row["objective"] != row["unweighted_objective"]
    assert row["mean_batch_objective"] == updates[0]["objective"]
    assert row["replay_weighting"] == result["replay_weighting"]


def test_weight_does_not_change_shuffle_action_history_or_feature_requests(replay_dataset):
    dataset, features = replay_dataset, features_for(replay_dataset)
    observations = []
    for weight in (1.0, 4.0):
        requests, updates, agent = [], [], TinyPolicy()

        def provide(index):
            requests.append(index)
            return features[index]

        train_imitation(agent, dataset, optimizer=torch.optim.SGD(agent.parameters(), lr=.01),
            epochs=2, minibatch_episodes=2, seed=87, feature_provider=provide,
            canonical_replay_weight=weight, on_update=updates.append)
        observations.append((requests, agent.seen, [row["tasks"] for row in updates]))
    assert observations[0] == observations[1]


@pytest.mark.parametrize("canonical_weight", [1.0, 3.0])
@pytest.mark.parametrize("minibatch", [1, 2, 8])
def test_mlx_cpu_weighted_objective_and_gradient_match_torch(
        replay_dataset, canonical_weight, minibatch):
    mx = pytest.importorskip("mlx.core")
    import mlx.nn as mnn

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        dataset, agent = replay_dataset, TinyPolicy()
        reference, features = deepcopy(agent), features_for(dataset)
        optimizer = torch.optim.SGD(agent.parameters(), lr=.01)

        class Policy(mnn.Module):
            def __init__(self):
                super().__init__()
                self.weights = {"scalar": mx.array(0.25)}

            def evaluate_episode(self, patches, actions, **kwargs):
                return {"log_prob": -(self.weights["scalar"] - patches[:, 0, 0]) ** 2}

        class Engine:
            def __init__(self):
                self.agent = agent
                self.optimizer = SimpleNamespace(canonical=optimizer)
                self.policy = Policy()
                self.gradient = None

            def apply_gradients(self, grads, max_grad_norm, *, active_names):
                assert active_names == {"scalar"}
                self.gradient = float(grads["weights"]["scalar"])
                self.policy.weights["scalar"] -= .01 * grads["weights"]["scalar"]
                return abs(self.gradient)

            def sync_to_torch(self):
                with torch.no_grad():
                    agent.scalar.copy_(torch.tensor(float(self.policy.weights["scalar"])))

        engine = Engine()
        options = dict(minibatch_episodes=minibatch,
                       canonical_replay_weight=canonical_weight,
                       action_balance="active_idle", max_grad_norm=1e6,
                       feature_provider=features.__getitem__)
        torch_result = train_imitation(reference, dataset,
            optimizer=torch.optim.SGD(reference.parameters(), lr=.01), **options)
        mlx_result = train_imitation(agent, dataset, optimizer=optimizer,
                                     backend="mlx", mlx_backend=engine, **options)
        assert engine.gradient == pytest.approx(float(reference.scalar.grad), abs=1e-6)
        torch.testing.assert_close(agent.scalar, reference.scalar, atol=1e-6, rtol=1e-6)
        for name in ("nll", "objective", "unweighted_objective", "mean_batch_objective"):
            assert mlx_result["history"][0][name] == pytest.approx(
                torch_result["history"][0][name], abs=1e-6)
    finally:
        mx.set_default_device(previous)


def test_replay_weighting_preserves_active_idle_weights_inside_each_episode():
    actions = (PhysicalAction(), PhysicalAction(), PhysicalAction(keys=("right",)))
    log_probs = torch.tensor([[-1., -3., -8.], [-2., -4., -5.]], requires_grad=True)
    parts = [_episode_nlls(row, actions, "active_idle") for row in log_probs]
    loss = _replay_objective(torch.stack([part[0] for part in parts]), [4.0, 1.0])
    loss.backward()
    assert float(loss.detach()) == pytest.approx(4.8)
    torch.testing.assert_close(log_probs.grad,
        torch.tensor([[-.2, -.2, -.4], [-.05, -.05, -.1]]))
    assert [float(part[1].detach()) for part in parts] == pytest.approx([4., 11 / 3])


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf"), -float("inf"), True])
def test_invalid_weight_rejected_before_policy_or_features(replay_dataset, weight):
    agent = TinyPolicy()
    with pytest.raises(ValueError, match="canonical replay weight must be positive and finite"):
        train_imitation(agent, replay_dataset, canonical_replay_weight=weight,
                        feature_provider=lambda _: pytest.fail("invalid weight encoded features"))
    assert not agent.seen
    with pytest.raises(ValueError, match="canonical replay weight"):
        imitation_replay_summary(replay_dataset, canonical_replay_weight=weight)


@pytest.mark.parametrize("weight", ["0", "-1", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_weight_before_loading_config(monkeypatch, capsys, weight):
    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "unused.json",
        "--warm-start", "unused", "--output", "unused", f"--canonical-replay-weight={weight}"])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
    assert "canonical replay weight must be positive and finite" in capsys.readouterr().err


def test_extreme_finite_weight_mean_does_not_overflow():
    values = torch.tensor([2.0, 8.0, 6.0], requires_grad=True)
    objective = _replay_objective(values, [1e308, 1.0, 1e308])
    objective.backward()
    assert float(objective.detach()) == 4.0
    torch.testing.assert_close(values.grad, torch.tensor([.5, 0., .5]))


@pytest.mark.parametrize("minibatch", [1, 2, 8])
@pytest.mark.parametrize("weight", [1.0, 3.0, 1e308])
def test_train_minibatch_gradients_average_to_dataset_weighted_objective(
        replay_dataset, minibatch, weight):
    """Frozen parameters separate objective scaling from optimizer trajectory changes."""
    dataset, agent = replay_dataset, TinyPolicy()
    reference, features = deepcopy(agent), features_for(dataset)
    parts = parts_for(reference, dataset, features)
    weights = [weight] + [1.0] * (len(parts) - 1)
    expected = _replay_objective(torch.stack([part[0] for part in parts]), weights)
    expected.backward()
    optimizer = torch.optim.SGD(agent.parameters(), lr=0.0)
    gradients = []
    optimizer.register_step_pre_hook(lambda *_: gradients.append(float(agent.scalar.grad)))
    result = train_imitation(agent, dataset, optimizer=optimizer,
        minibatch_episodes=minibatch, max_grad_norm=1e6, canonical_replay_weight=weight,
        feature_provider=features.__getitem__)
    sizes = [min(minibatch, len(parts) - offset)
             for offset in range(0, len(parts), minibatch)]
    aggregate = sum(size * grad for size, grad in zip(sizes, gradients, strict=True)) / len(parts)
    assert aggregate == pytest.approx(float(reference.scalar.grad), abs=1e-6)
    assert result["history"][0]["objective"] == pytest.approx(float(expected.detach()), abs=1e-6)
    if weight == 3.0:
        unweighted = TinyPolicy()
        unweighted_parts = parts_for(unweighted, dataset, features)
        torch.stack([part[0] for part in unweighted_parts]).mean().backward()
        assert aggregate != pytest.approx(float(unweighted.scalar.grad))


def test_global_normalizer_keeps_relative_mass_for_single_episode_batches():
    weights = [4.0, 1.0]
    values = torch.tensor([2.0, 8.0], requires_grad=True)
    normalization = _replay_normalization(weights)
    losses = [_replay_objective(values[index:index + 1], [weight], normalization=normalization)
              for index, weight in enumerate(weights)]
    assert [float(loss.detach()) for loss in losses] == pytest.approx([3.2, 3.2])
    torch.stack(losses).mean().backward()
    torch.testing.assert_close(values.grad, torch.tensor([.8, .2]))


@pytest.mark.parametrize("weight", [None, 2.5])
@pytest.mark.parametrize("rejected_endpoint_difficulty", [None, .25])
@pytest.mark.parametrize("exclude_corrections", [False, True])
def test_cli_persists_weight_and_effective_domain_masses(
        monkeypatch, tmp_path, replay_dataset, weight, rejected_endpoint_difficulty,
        exclude_corrections):
    from carryrl.training.run import atomic_json
    from carryrl.training.sequence_config import SequenceRunConfig

    source, output = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    atomic_json(source / "manifest.json", {"sha256": "test archive"})
    config = SequenceRunConfig(tasks=("navigation",))
    agent = SimpleNamespace(vision=object(), summary_query=torch.zeros(1),
                            encode_frames=lambda frames: torch.zeros(len(frames), 1, 384))
    optimizer, reward = SimpleNamespace(param_groups=[{}]), object()
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(script, "provenance", lambda: {})
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k:
        (agent, optimizer, reward, {"decisions": 0, "weights_revision": 0,
                                   "config": config.to_dict()}))
    monkeypatch.setattr(script, "tensor_hash", lambda _: "test identity")
    monkeypatch.setattr(script, "collect_scripted_demonstrations", lambda *a, **k: replay_dataset)
    monkeypatch.setattr(script, "evaluate_sequence", lambda *a, **k: {"summaries": {
        "frozen_deterministic": {"outcomes": {"success": 0}, "episodes_started": 1,
                                 "tasks": {"navigation": {"success_rate_lower": 0.0}}}}})

    class Prepared(Exception):
        pass

    def stop_before_training(*args, **kwargs):
        assert kwargs["canonical_replay_weight"] == (1.0 if weight is None else weight)
        assert kwargs["canonical_anchor_exclude_corrections"] is exclude_corrections
        raise Prepared

    monkeypatch.setattr(script, "train_imitation", stop_before_training)
    argv = ["train_imitation.py", "--config", "unused.json", "--warm-start", str(source),
            "--output", str(output), "--training-backend", "torch",
            "--epochs", "2", "--minibatch-episodes", "2"]
    if weight is not None:
        argv += ["--canonical-replay-weight", str(weight)]
    if exclude_corrections:
        argv.append("--canonical-anchor-exclude-corrections")
    if rejected_endpoint_difficulty is not None:
        class EndpointRejected(Exception):
            pass

        def reject_endpoint(dataset, *, level_bank_seed, evaluation_difficulty):
            assert dataset is replay_dataset
            assert level_bank_seed == config.level_bank_seed
            assert evaluation_difficulty == rejected_endpoint_difficulty
            raise EndpointRejected

        monkeypatch.setattr(script, "validate_dataset_runtime_holdout", reject_endpoint)
        agent.encode_frames = lambda *_: pytest.fail("unsafe corpus reached feature encoding")
        argv += ["--evaluation-difficulty", str(rejected_endpoint_difficulty)]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(EndpointRejected):
            script.main()
        assert "replay_weighting" not in json.loads((output / "metadata.json").read_text())
        assert not (output / "demonstrations.json").exists()
        return
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(Prepared):
        script.main()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["settings"]["difficulty"] == [1.0]
    expected = 1.0 if weight is None else weight
    assert metadata["settings"]["canonical_replay_weight"] == expected
    if exclude_corrections:
        assert metadata["settings"]["canonical_anchor_exclude_corrections"] is True
    else:
        assert "canonical_anchor_exclude_corrections" not in metadata["settings"]
    assert metadata["canonical_anchor"]["exclude_corrections"] is exclude_corrections
    assert metadata["replay_weighting"] == imitation_replay_summary(
        replay_dataset, canonical_replay_weight=expected)
    assert json.loads((output / "demonstrations.json").read_text()) == replay_dataset.provenance()

    # Exercise the strict completion gate with the actual CLI-persisted settings,
    # rather than only comparing argparse defaults before metadata is serialized.
    from test_retention_evaluation import run_evidence

    from carryrl.evaluation.retention import validate_retention_run
    from carryrl.training.retention import _settings

    plan, evidence_metadata, summary, manifest, final = run_evidence("anchor_off")
    plan["arms"]["anchor_off"]["settings"] = _settings({
        "config": "unused.json", "warm_start": str(source), "training_backend": "torch",
        "epochs": 2, "minibatch_episodes": 2, "canonical_replay_weight": expected},
        coefficient=0.0, output=output)
    evidence_metadata["settings"] = metadata["settings"]
    summary["backend"] = manifest["training_backend"] = "torch"
    if exclude_corrections:
        with pytest.raises(ValueError, match="run settings differ from plan"):
            validate_retention_run(plan, "anchor_off", evidence_metadata, summary, manifest, final)
    else:
        assert metadata["settings"] == plan["arms"]["anchor_off"]["settings"]
        assert validate_retention_run(
            plan, "anchor_off", evidence_metadata, summary, manifest, final)["valid"] is True
