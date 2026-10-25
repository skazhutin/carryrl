"""Synthetic teacher provenance, RGB/action alignment and genuine causal BC updates."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.model import AgentConfig, GeneralistAgent
from carryrl.envs.atomic import AtomicEnv, _identity, atomic_level_banks
from carryrl.envs.atomic_suite import TASKS
from carryrl.training.imitation import (
    CORRECTION_SOURCE,
    MEMORY_COUNTERPART_SOURCE,
    METHOD,
    SOURCE,
    DemonstrationDataset,
    augment_demonstrations,
    collect_memory_cue_counterparts,
    collect_scripted_corrections,
    collect_scripted_demonstrations,
    imitation_action_weights,
    imitation_nll,
    scripted_teacher,
    train_imitation,
)
from carryrl.training.sequence import FrozenFrameCache


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 384)
        self.calls = 0

    def forward(self, frames):
        self.calls += 1
        rgb = frames.permute(0, 3, 1, 2).float() / 255
        return self.projection(F.adaptive_avg_pool2d(rgb, (2, 2)).flatten(2).transpose(1, 2))


class TinyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Parameter(torch.linspace(-1, 1, 384))

    def forward(self, instruction):
        return self.feature


def make_agent(*, instruction=False, segment_length=2):
    torch.manual_seed(17)
    torch.set_num_threads(2)
    config = AgentConfig(pretrained=False, image_size=28, width=16, heads=2, layers=1,
                         spatial_tokens=2, memory_tokens=2, segment_length=segment_length,
                         text_encoder="multilingual_e5_small" if instruction else "none")
    return GeneralistAgent(config, vision=TinyVision(),
                           language=TinyLanguage() if instruction else None)


def test_all16_teacher_demonstrations_pass_visible_quality_gate_on_train():
    # Simplified platforming collapses train/held-out mechanics and is rejected
    # before collection; all-sixteen quality coverage uses separated full levels.
    dataset = collect_scripted_demonstrations(2, difficulty=1.0)
    assert len(dataset.episodes) == 32
    assert {episode.task for episode in dataset.episodes} == set(TASKS)
    for episode in dataset.episodes:
        assert episode.outcome["known"] and episode.outcome["success"]
        assert not episode.outcome["failure"]
        assert episode.source == SOURCE and episode.split == "train"
        assert len(episode.frame_indices) == len(episode.actions)
        assert all(isinstance(action, PhysicalAction) for action in episode.actions)
        assert episode.level_id == _identity(atomic_level_banks(episode.task)["train"][
            episode.level_index])
    assert any("right" in action.keys and "space" in action.keys
               for episode in dataset.episodes if episode.task == "platforming"
               for action in episode.actions)
    assert dataset.frame_bytes == sum(frame.nbytes for frame in dataset.frames)
    assert dataset.frame_bytes <= dataset.max_frame_bytes
    assert len(dataset.frames) < sum(len(episode) + 1 for episode in dataset.episodes)
    record = dataset.provenance()
    assert record["source"] == SOURCE and record["method"] == METHOD
    assert "canonical" in record["teacher_presentation"]
    assert "codec_version" in record["episodes"][0]["actions"][0]


def test_teacher_receives_only_immutable_rgb_and_labels_align_with_executed_actions():
    environments = []

    class RecordingEnv:
        def __init__(self, *args, **kwargs):
            self.env = AtomicEnv(*args, **kwargs)
            self.frames, self.actions = [], []
            environments.append(self)

        def reset(self, **kwargs):
            frame, info = self.env.reset(**kwargs)
            self.frames.append(frame.copy())
            return frame, info

        def step(self, action):
            self.actions.append(action)
            result = self.env.step(action)
            self.frames.append(result[0].copy())
            return result

        def close(self):
            self.env.close()

    class Teacher:
        def __init__(self, task):
            self.delegate = scripted_teacher(task)

        def act(self, frame):
            assert isinstance(frame, np.ndarray) and frame.dtype == np.uint8
            assert not frame.flags.writeable
            with pytest.raises(ValueError):
                frame.setflags(write=True)
            with pytest.raises(ValueError):
                frame[0, 0, 0] = 0
            return self.delegate.act(frame)

    dataset = collect_scripted_demonstrations(1, tasks=("mouse", "memory", "platforming"),
        env_factory=RecordingEnv, teacher_factory=Teacher,
        episode_instructions={"memory": ("Remember the cue and choose its door",)})
    for episode, environment in zip(dataset.episodes, environments, strict=True):
        assert episode.actions == tuple(environment.actions)
        for index, frame in zip(episode.frame_indices, environment.frames[:-1], strict=True):
            np.testing.assert_array_equal(dataset.frames[index], frame)
        np.testing.assert_array_equal(dataset.frames[episode.final_frame_index],
                                      environment.frames[-1])
    assert dataset.episodes[1].instruction == "Remember the cue and choose its door"


@pytest.mark.parametrize("split", ["validation", "test"])
def test_held_out_collection_is_rejected_before_environment_creation(split):
    def forbidden(*args, **kwargs):
        raise AssertionError("held-out environment must never be opened")
    with pytest.raises(ValueError, match="train-only"):
        collect_scripted_demonstrations(split=split, env_factory=forbidden)


def test_teacher_failure_or_unknown_final_hud_aborts_collection():
    class IdleTeacher:
        def act(self, frame):
            return PhysicalAction()
    with pytest.raises(RuntimeError, match="quality gate failed"):
        collect_scripted_demonstrations(1, tasks=("navigation",), max_steps=2,
                                       teacher_factory=lambda task: IdleTeacher())

    class CorruptHUD(AtomicEnv):
        def step(self, action):
            frame, reward, terminated, truncated, info = super().step(action)
            if terminated or truncated:
                frame = np.zeros_like(frame)
            return frame, reward, terminated, truncated, info
    with pytest.raises(RuntimeError, match="quality gate failed"):
        collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=0,
                                       env_factory=CorruptHUD)


def test_dataset_deduplicates_exact_rgb_and_enforces_byte_budget():
    dataset = DemonstrationDataset(max_frame_bytes=24)
    zero = np.zeros((2, 2, 3), np.uint8)
    assert dataset.intern_frame(zero) == dataset.intern_frame(zero.copy()) == 0
    assert dataset.intern_frame(zero + 1) == 1
    assert dataset.frame_bytes == 24
    with pytest.raises(MemoryError, match="byte budget"):
        dataset.intern_frame(zero + 2)
    assert dataset.frame_bytes == 24 and len(dataset.frames) == 2
    with pytest.raises(ValueError, match="uint8"):
        dataset.intern_frame(zero.astype(np.float32))


def test_training_bank_and_visible_final_outcome_are_revalidated():
    dataset = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=0)
    episode = dataset.episodes[0]
    with pytest.raises(ValueError, match="training bank"):
        dataset.add_episode(replace(episode,
            level_id=_identity(atomic_level_banks("navigation")["validation"][0])))
    with pytest.raises(ValueError, match="final visible RGB"):
        dataset.add_episode(replace(episode, final_frame_index=episode.frame_indices[0]))
    with pytest.raises(ValueError, match="train scripted_teacher"):
        replace(episode, split="validation")
    with pytest.raises(ValueError, match="match physical actions"):
        replace(episode, actions=())
    dataset.episodes[0] = replace(episode, level_id="held-out")
    with pytest.raises(ValueError, match="training bank"):
        train_imitation(make_agent(), dataset)


def test_combine_difficulty_stages_preserves_labels_and_deduplicates():
    left = collect_scripted_demonstrations(1, tasks=("mouse",), difficulty=0)
    right = collect_scripted_demonstrations(1, tasks=("mouse",), difficulty=1)
    expected = [right.frames[index].copy() for index in right.episodes[0].frame_indices]
    left.extend(right)
    assert [episode.difficulty for episode in left.episodes] == [0, 1]
    assert left.episodes[1].actions == right.episodes[0].actions
    for index, frame in zip(left.episodes[1].frame_indices, expected, strict=True):
        np.testing.assert_array_equal(left.frames[index], frame)
    before = left.frame_bytes
    left.extend(right)
    assert left.frame_bytes == before


def test_joint_nll_has_finite_causal_gradients_for_multikey_actions_and_text():
    agent = make_agent(instruction=True)
    patches = torch.randn(7, 4, 384)
    actions = [PhysicalAction(keys=("right", "space"), mouse_buttons=("left", "middle"),
                              dx=8, dy=-4, wheel=1)] * 7
    before = imitation_nll(agent, patches, actions, instruction="Move and jump")
    before.backward()
    assert torch.isfinite(before)
    for name in ("instruction_projection.weight", "decoder.key_head.weight",
                 "temporal.write_tokens", "previous_action_embedding.weight"):
        gradient = dict(agent.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert agent.value_head.weight.grad is None
    assert all(parameter.grad is None for parameter in agent.vision.parameters())
    assert all(parameter.grad is None for parameter in agent.language.parameters())
    altered = patches.clone()
    altered[5:] += 10
    output = agent.evaluate_episode(patches, actions, instruction="Move and jump")
    changed = agent.evaluate_episode(altered, actions[:5] + [PhysicalAction()] * 2,
                                     instruction="Move and jump")
    torch.testing.assert_close(output["log_prob"][:5], changed["log_prob"][:5])


def test_training_reduces_demonstration_nll_reuses_frozen_features_and_preserves_critic():
    agent = make_agent()
    dataset = collect_scripted_demonstrations(2, tasks=("navigation",), difficulty=0)
    cache = FrozenFrameCache(agent, 100, max_bytes=1024**2)
    optimizer = torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad), lr=0.003)
    before_vision = {key: value.clone() for key, value in agent.vision.state_dict().items()}
    before_critic = {key: value.clone() for key, value in agent.value_head.state_dict().items()}
    rows = []
    result = train_imitation(agent, dataset, epochs=10, optimizer=optimizer,
                             feature_cache=cache, on_epoch=rows.append)
    assert len(rows) == result["epochs_completed"] == 10
    assert result["source"] == SOURCE and result["method"] == METHOD
    assert rows[-1]["nll"] < rows[0]["nll"]
    assert result["stop_reason"] == "completed" and result["optimizer_steps"] == 20
    assert agent.vision.calls == len({index for episode in dataset.episodes
                                     for index in episode.frame_indices})
    assert cache.statistics()["hits"] > 0 and cache.statistics()["bytes"] <= 1024**2
    assert all(torch.equal(value, before_vision[key])
               for key, value in agent.vision.state_dict().items())
    assert all(torch.equal(value, before_critic[key])
               for key, value in agent.value_head.state_dict().items())
    assert not optimizer.state.get(agent.value_head.weight)


def test_external_raw_feature_provider_cannot_supply_trainable_projections():
    agent = make_agent()
    dataset = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=0)
    with pytest.raises(ValueError, match="detached raw DINO"):
        train_imitation(agent, dataset,
            feature_provider=lambda index: torch.zeros(4, 384, requires_grad=True))


@pytest.mark.parametrize("task", ["navigation", "memory"])
def test_mlx_and_torch_bc_weights_moments_and_unused_parameters_match(task):
    from copy import deepcopy

    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("MLX imitation requires Metal")
    from carryrl.training.mlx import MLXPPOBackend

    instruction = task == "memory"
    torch_agent = make_agent(instruction=True)
    mlx_agent = deepcopy(torch_agent)
    dataset = collect_scripted_demonstrations(1, tasks=(task,), difficulty=0,
        episode_instructions={task: ("Remember the cue",)} if instruction else None)
    providers = {}
    for index, frame in enumerate(dataset.frames):
        providers[index] = torch_agent.encode_frames(torch.tensor(frame)[None])[0].cpu()
    optimizers = [torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad),
                                   lr=3e-4, weight_decay=.03)
                  for agent in (torch_agent, mlx_agent)]
    engine = MLXPPOBackend(mlx_agent, optimizers[1])
    torch_result = train_imitation(torch_agent, dataset, epochs=2, optimizer=optimizers[0],
                                   feature_provider=providers.__getitem__)
    # Native immutable Metal raw features may stay resident across every epoch.
    gpu_features = {index: mx.array(value.numpy()) for index, value in providers.items()}
    mlx_result = train_imitation(mlx_agent, dataset, epochs=2, optimizer=optimizers[1],
        backend="mlx", mlx_backend=engine, feature_provider=gpu_features.__getitem__)
    assert mlx_result["history"][0]["nll"] == pytest.approx(
        torch_result["history"][0]["nll"], abs=3e-6)
    for name, parameter in torch_agent.named_parameters():
        torch.testing.assert_close(parameter, dict(mlx_agent.named_parameters())[name],
                                   atol=4e-6, rtol=3e-4)
    states = [optimizer.state_dict() for optimizer in optimizers]
    assert states[0]["param_groups"] == states[1]["param_groups"]
    assert set(states[0]["state"]) == set(states[1]["state"])
    for key, values in states[0]["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, states[1]["state"][key][name],
                                       atol=3e-7, rtol=3e-4)
    assert "value_head.weight" not in engine.optimizer.state
    if task == "navigation":
        assert "temporal.write_tokens" not in engine.optimizer.state
        assert "instruction_projection.weight" not in engine.optimizer.state


@pytest.mark.parametrize("backend", ["torch", "mlx"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_update_callback_records_completed_revision_and_interrupt_preserves_exact_weights(
    backend, interrupt,
):
    if backend == "mlx":
        mx = pytest.importorskip("mlx.core")
        if not mx.metal.is_available():
            pytest.skip("MLX imitation requires Metal")
    agent = make_agent()
    dataset = collect_scripted_demonstrations(2, tasks=("navigation",), difficulty=0)
    optimizer = torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad), lr=3e-4)
    before = agent.decoder.key_head.weight.detach().clone()
    updates, epochs = [], []

    def on_update(row):
        updates.append(row)
        if interrupt:
            raise KeyboardInterrupt("stop after a completed optimizer revision")

    options = dict(epochs=2, backend=backend, optimizer=optimizer,
                   on_update=on_update, on_epoch=epochs.append)
    if interrupt:
        with pytest.raises(KeyboardInterrupt, match="completed optimizer revision"):
            train_imitation(agent, dataset, **options)
        assert len(updates) == 1 and not epochs
    else:
        result = train_imitation(agent, dataset, **options)
        assert len(updates) == result["optimizer_steps"] == 4
        assert [row["optimizer_steps"] for row in epochs] == [2, 4]
    assert [row["optimizer_steps"] for row in updates] == list(range(1, len(updates) + 1))
    assert all(row["source"] == SOURCE and row["method"] == METHOD for row in updates)
    assert all(np.isfinite(row["nll"]) and row["batch_episodes"] == 1 for row in updates)
    assert float(optimizer.state[agent.decoder.key_head.weight]["step"]) == len(updates)
    assert not torch.equal(before, agent.decoder.key_head.weight)
    assert not agent.training


def test_epoch_start_callback_reports_visited_epoch_without_faking_completion():
    agent = make_agent()
    dataset = collect_scripted_demonstrations(2, tasks=("navigation",), difficulty=0)
    starts, updates = [], []
    result = train_imitation(agent, dataset, epochs=2, max_wall_seconds=1e-12,
                             on_epoch_start=starts.append, on_update=updates.append)
    assert starts[0]["epoch"] == 1 and starts[0]["batches_per_epoch"] == 2
    assert starts[0]["optimizer_steps"] == 0
    assert not updates and result["optimizer_steps"] == result["epochs_completed"] == 0
    assert result["history"][0]["epoch"] == 1 and not result["history"][0]["complete"]
    assert result["stop_reason"] == "wall_limit"


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_callback_error_survives_final_mlx_sync_failure_without_gpu(error_type):
    agent = make_agent()
    dataset = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=0)
    optimizer = torch.optim.AdamW(parameter for parameter in agent.parameters()
                                  if parameter.requires_grad)
    sync_calls = []
    primary = error_type("snapshot callback failed")

    def sync():
        sync_calls.append("sync")
        if len(sync_calls) == 2:
            raise RuntimeError("cleanup bridge failed")

    def callback(row):
        assert sync_calls == ["sync"]
        raise primary

    engine = SimpleNamespace(agent=agent, optimizer=SimpleNamespace(canonical=optimizer),
                             sync_to_torch=sync)
    with pytest.raises(error_type, match="snapshot callback failed") as raised:
        train_imitation(agent, dataset, epochs=1, backend="mlx", optimizer=optimizer,
                        mlx_backend=engine, max_wall_seconds=1e-12, on_epoch=callback,
                        feature_provider=lambda _: pytest.fail("no GPU work expected"))
    assert raised.value is primary
    assert primary.__notes__ == [
        "Final MLX synchronization failed: RuntimeError: cleanup bridge failed"]
    assert sync_calls == ["sync", "sync"]
    assert not agent.training


def test_clean_body_final_mlx_sync_failure_propagates_and_restores_eval_without_gpu():
    agent = make_agent()
    dataset = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=0)
    optimizer = torch.optim.AdamW(parameter for parameter in agent.parameters()
                                  if parameter.requires_grad)
    sync_calls, rows = [], []
    cleanup_error = RuntimeError("cleanup bridge failed")

    def sync():
        sync_calls.append("sync")
        if len(sync_calls) == 2:
            raise cleanup_error

    engine = SimpleNamespace(agent=agent, optimizer=SimpleNamespace(canonical=optimizer),
                             sync_to_torch=sync)
    with pytest.raises(RuntimeError, match="cleanup bridge failed") as raised:
        train_imitation(agent, dataset, epochs=1, backend="mlx", optimizer=optimizer,
                        mlx_backend=engine, max_wall_seconds=1e-12, on_epoch=rows.append,
                        feature_provider=lambda _: pytest.fail("no GPU work expected"))
    assert raised.value is cleanup_error
    assert len(rows) == 1 and rows[0]["optimizer_steps"] == 0
    assert sync_calls == ["sync", "sync"]
    assert not agent.training


def test_epoch_training_seconds_excludes_scheduled_evaluation_callback(monkeypatch):
    agent = make_agent()
    dataset = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=0)
    now = [10.0]
    monkeypatch.setattr("carryrl.training.imitation.time.monotonic", lambda: now[0])

    def started(row):
        now[0] += 2

    def evaluated(row):
        now[0] += 10

    result = train_imitation(agent, dataset, epochs=2, on_epoch_start=started,
                             on_epoch=evaluated)
    assert [row["epoch_training_seconds"] for row in result["history"]] == [2, 2]
    assert [row["seconds"] for row in result["history"]] == [2, 14]


def test_demonstration_progress_counts_only_accepted_real_teacher_episodes():
    progress = []
    dataset = collect_scripted_demonstrations(
        2, tasks=("navigation", "memory"), difficulty=0, on_episode=progress.append)
    assert [row["episodes_completed"] for row in progress] == [1, 2, 3, 4]
    assert all(row["episodes_total"] == 4 for row in progress)
    assert [row["steps"] for row in progress] == [len(episode) for episode in dataset.episodes]
    assert progress[-1]["unique_frames"] == len(dataset.frames)


def test_executed_history_conditions_policy_without_changing_teacher_targets(monkeypatch):
    agent = make_agent()
    patches = torch.randn(5, 4, 384)
    labels = [PhysicalAction(keys=("right",))] * 5
    executed = [PhysicalAction(keys=("left",))] * 5
    observed = []
    observed_previous = []
    decode = agent.decoder.evaluate

    def record(features, actions, *, previous_actions=None):
        observed.append(list(actions))
        observed_previous.append(previous_actions)
        return decode(features, actions, previous_actions=previous_actions)

    monkeypatch.setattr(agent.decoder, "evaluate", record)
    baseline = agent.evaluate_episode(patches, labels)
    correction = agent.evaluate_episode(patches, labels, executed_actions=executed)
    assert observed == [labels, labels]
    assert observed_previous == [[PhysicalAction(), *labels[:-1]],
                                 [PhysicalAction(), *executed[:-1]]]
    torch.testing.assert_close(baseline["value"][:1], correction["value"][:1])
    assert not torch.allclose(baseline["value"][1:], correction["value"][1:])
    changed_last = agent.evaluate_episode(patches, labels,
        executed_actions=executed[:-1] + [PhysicalAction()])
    torch.testing.assert_close(correction["log_prob"], changed_last["log_prob"])
    torch.testing.assert_close(baseline["log_prob"], agent.evaluate_episode(
        patches, labels, executed_actions=labels)["log_prob"])
    with pytest.raises(ValueError, match="executed actions"):
        agent.evaluate_episode(patches, labels, executed_actions=executed[:-1])


class CorrectionLearner(nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action
        self.inputs, self.feedback = [], []
        self._previous_action = PhysicalAction()

    def reset_episode(self, *, instruction=None):
        self._previous_action = PhysicalAction()
        self.instruction = instruction

    def act(self, frame, deterministic=False):
        from types import SimpleNamespace
        assert deterministic and not self.training and not torch.is_grad_enabled()
        self.inputs.append((frame.copy(), self._previous_action))
        self._previous_action = self.action
        return SimpleNamespace(action=self.action)

    def observe_action(self, action):
        self.feedback.append(action)
        self._previous_action = action


def test_corrections_accept_actual_failure_and_preserve_distinct_provenance():
    learner = CorrectionLearner(PhysicalAction(mouse_buttons=("left",)))
    dataset = collect_scripted_corrections(learner, 1, tasks=("mouse",),
                                         expert_probability=0, difficulty=1)
    episode = dataset.episodes[0]
    assert learner.training  # Prior mode restored after frozen collection.
    assert episode.source == CORRECTION_SOURCE and episode.outcome["failure"]
    assert episode.terminated and not episode.truncated
    assert episode.executed_actions == tuple(learner.feedback)
    assert episode.actions != episode.executed_actions
    assert dataset.provenance()["source"] == CORRECTION_SOURCE
    assert episode.to_dict()["executed_actions"][0]["mouse_buttons"] == ["left"]
    assert all(not frame.flags.writeable for frame in dataset.frames)
    with pytest.raises(ValueError, match="visible success"):
        replace(episode, source=SOURCE)
    with pytest.raises(ValueError, match="train scripted_teacher"):
        replace(episode, split="validation")
    with pytest.raises(ValueError, match="training bank"):
        dataset.add_episode(replace(episode,
            level_id=_identity(atomic_level_banks("mouse")["test"][0])))
    with pytest.raises(ValueError, match="final visible RGB"):
        dataset.add_episode(replace(episode, final_frame_index=episode.frame_indices[0]))


@pytest.mark.parametrize("split", ["validation", "test"])
def test_correction_collection_rejects_held_out_splits_before_creating_env(split):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not open a held-out environment")
    with pytest.raises(ValueError, match="train-only"):
        collect_scripted_corrections(CorrectionLearner(PhysicalAction()), split=split,
                                     env_factory=forbidden)


def test_correction_mixing_uses_actual_prefix_and_isolated_numpy_rng():
    learner = CorrectionLearner(PhysicalAction())
    np.random.seed(7)
    state = np.random.get_state()
    progress = []
    dataset = collect_scripted_corrections(learner, 1, tasks=("navigation",), max_steps=8,
        expert_probability=0.5, difficulty=1, on_episode=progress.append)
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    episode = dataset.episodes[0]
    assert episode.terminated or episode.truncated
    for index, ((frame, previous), frame_index) in enumerate(zip(
            learner.inputs, episode.frame_indices, strict=True)):
        np.testing.assert_array_equal(frame, dataset.frames[frame_index])
        assert previous == (PhysicalAction() if index == 0 else episode.executed_actions[index - 1])
    assert progress[0]["source"] == CORRECTION_SOURCE
    assert 0 < progress[0]["expert_steps"] < len(episode)
    again = collect_scripted_corrections(CorrectionLearner(PhysicalAction()), 1,
        tasks=("navigation",), max_steps=8, expert_probability=0.5)
    assert again.episodes[0].executed_actions == episode.executed_actions
    truncated = collect_scripted_corrections(CorrectionLearner(PhysicalAction()), 1,
        tasks=("navigation",), max_steps=3, expert_probability=0).episodes[0]
    assert truncated.truncated and not truncated.outcome["success"]


def test_correction_training_replays_actual_actions(monkeypatch):
    dataset = collect_scripted_corrections(CorrectionLearner(PhysicalAction()), 1,
        tasks=("navigation",), max_steps=3, expert_probability=0)
    agent = make_agent()
    observed = []
    evaluate = agent.evaluate_episode

    def record(patches, actions, **kwargs):
        observed.append((tuple(actions), kwargs["executed_actions"]))
        return evaluate(patches, actions, **kwargs)

    monkeypatch.setattr(agent, "evaluate_episode", record)
    result = train_imitation(agent, dataset, epochs=1)
    assert observed == [(dataset.episodes[0].actions, dataset.episodes[0].executed_actions)]
    assert result["source"] == CORRECTION_SOURCE


def test_memory_teacher_waits_neutrally_until_visible_choose_and_crosses_64_steps():
    dataset = collect_scripted_demonstrations(1, tasks=("memory",), difficulty=1)
    episode = dataset.episodes[0]
    assert len(episode) > 64
    nonneutral = [index for index, action in enumerate(episode.actions) if action.keys]
    assert nonneutral and min(nonneutral) >= 64
    assert all(action == PhysicalAction() for action in episode.actions[:min(nonneutral)])
    # Opposite disappearing cues have identical neutral prior inputs through the delay.
    assert episode.outcome["success"]


def test_rgb_controllers_use_actual_edge_feedback_instead_of_hypothetical_labels():
    teacher = scripted_teacher("interaction")
    env = AtomicEnv("interaction", difficulty=0)
    try:
        frame, _ = env.reset(seed=0)
        for _ in range(20):
            label = teacher.act(frame)
            if "e" in label.keys:
                # A recommendation was not executed: E remains available immediately.
                teacher.observe_action(PhysicalAction())
                assert "e" in teacher.act(frame).keys
                # An actual E press needs a release before the next rising edge.
                teacher.observe_action(PhysicalAction(keys=("e",)))
                assert teacher.act(frame) == PhysicalAction()
                break
            frame, _, terminated, truncated, _ = env.step(label)
            teacher.observe_action(label)
            assert not terminated and not truncated
        else:
            pytest.fail("interaction teacher never reached a visible E target")
    finally:
        env.close()
    timing = scripted_teacher("timing")
    timing.observe_action(PhysicalAction(keys=("space",)))
    assert timing.previous_flap
    timing.observe_action(PhysicalAction())
    assert not timing.previous_flap
    shooting = scripted_teacher("shooting")
    shooting.observe_action(PhysicalAction(mouse_buttons=("left",)))
    assert shooting.pressed
    shooting.observe_action(PhysicalAction())
    assert not shooting.pressed


def test_observe_actual_action_does_not_advance_policy_memory():
    agent = make_agent()
    agent.reset_episode()
    agent.act_from_patches(torch.randn(4, 384), deterministic=True)
    memory = agent._episode_state
    actual = PhysicalAction(keys=("e",))
    agent.observe_action(actual)
    assert agent._episode_state is memory and agent._previous_action == actual


@pytest.mark.parametrize("layout", [0, 1])
def test_legacy_three_item_ui_layout_and_controller_remain_compatible(layout):
    from carryrl.envs.atomic_combat import UIGame
    level = {"request": 1, "items": [2, 0, 1], "selection": 0,
             "layout": layout, "request_x": 48, "menu_offset": 2}
    game = UIGame(level, 96)
    for index in range(3):
        x, y = ((19 + 29 * index, 37) if layout == 0 else (48, 29 + 14 * index))
        cx, cy = game._point(x, y)
        radius = 6 if layout == 0 else 3
        assert game._item_rect(index) == (cx - radius, cy - radius, cx + radius, cy + radius)
    teacher = scripted_teacher("ui")
    for _ in range(20):
        action = teacher.act(game.render())
        game.step(action)
        teacher.observe_action(action)
        if game.status != "Play":
            break
    assert game.status == "Success"


@pytest.mark.parametrize("layout", [0, 1])
@pytest.mark.parametrize("requested", range(4))
def test_four_item_ui_teacher_reads_visible_request_and_selects_correctly(layout, requested):
    from carryrl.envs.atomic_combat import ITEM_COLORS, UIGame
    level = {"request": requested, "items": [3, 2, 0, 1], "selection": 3,
             "layout": layout, "request_x": 48, "menu_offset": 2}
    game = UIGame(level, 96)
    teacher = scripted_teacher("ui")
    for _ in range(20):
        frame = game.render()
        if game.opened:
            assert all(np.any(np.all(frame == color, axis=-1)) for color in ITEM_COLORS)
        action = teacher.act(frame)
        game.step(action)
        teacher.observe_action(action)
        if game.status != "Play":
            break
    assert game.status == "Success"


def test_correction_dataset_rejects_mutable_rgb_backing():
    dataset = collect_scripted_corrections(CorrectionLearner(PhysicalAction()), 1,
        tasks=("navigation",), max_steps=2, expert_probability=0)
    episode = dataset.episodes[0]
    index = episode.frame_indices[0]
    mutable = dataset.frames[index].copy()
    mutable.setflags(write=False)
    dataset.frames[index] = mutable
    with pytest.raises(ValueError, match="immutable backing"):
        dataset._validate_episode(episode)


def test_appearance_augmentation_preserves_whole_trajectory_actions_and_outcomes():
    from carryrl.envs.presentation import PresentationProfile, present_frame
    from carryrl.rewards.visual import verify_visible_outcome

    canonical = collect_scripted_demonstrations(1, tasks=("mouse", "memory", "ui"),
                                               difficulty=0)
    before = [frame.tobytes() for frame in canonical.frames]
    episodes_before = canonical.episodes.copy()
    augmented = augment_demonstrations(canonical, seed=19, copies_per_episode=2)
    assert len(augmented.episodes) == 2 * len(canonical.episodes)
    for index, transformed in enumerate(augmented.episodes):
        original = canonical.episodes[index // 2]
        assert transformed.actions is original.actions
        assert transformed.executed_actions is original.executed_actions
        assert transformed.level_id == original.level_id
        assert transformed.task == original.task and transformed.source == original.source
        assert transformed.outcome == original.outcome
        assert transformed.instruction == original.instruction
        profile = PresentationProfile(**transformed.presentation)
        assert profile.native_size == 96 and profile.textures
        x0, y0, x1, y1 = profile.viewport_box
        assert (x1 - x0, y1 - y0) == (96, 96)
        # Every frame, including final outcome RGB, uses the exact same profile.
        for old, new in zip((*original.frame_indices, original.final_frame_index),
                            (*transformed.frame_indices, transformed.final_frame_index),
                            strict=True):
            np.testing.assert_array_equal(augmented.frames[new],
                                           present_frame(canonical.frames[old], profile))
            assert verify_visible_outcome(augmented.frames[new]) == verify_visible_outcome(
                canonical.frames[old])
            assert not augmented.frames[new].flags.writeable
    assert before == [frame.tobytes() for frame in canonical.frames]
    assert episodes_before == canonical.episodes
    assert all(episode.presentation is None for episode in canonical.episodes)
    metadata = augmented.provenance()
    assert metadata["actor_presentation"] == "train_appearance_augmentation"
    assert metadata["appearance_augmentation"]["profile_input_to_actor"] is False
    assert metadata["appearance_augmentation"]["native_size"] == 96
    assert augmented.episodes[0].to_dict()["presentation"] == augmented.episodes[0].presentation


def test_appearance_augmentation_uses_train_only_themes_four_layouts_and_isolated_rng():
    from carryrl.envs.presentation import HUD_LAYOUTS, THEME_SPLITS

    canonical = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=0)
    np.random.seed(7)
    state = np.random.get_state()
    augmented = augment_demonstrations(canonical, seed=11, copies_per_episode=24)
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    profiles = [episode.presentation for episode in augmented.episodes]
    assert {profile["theme"] for profile in profiles} == set(THEME_SPLITS["train"])
    assert {profile["hud_layout"] for profile in profiles} == set(HUD_LAYOUTS)
    assert len({profile["texture_seed"] for profile in profiles}) == len(profiles)
    shapes = {frame.shape for frame in augmented.frames}
    assert shapes == {(148, 112, 3), (112, 184, 3)}
    again = augment_demonstrations(canonical, seed=11, copies_per_episode=24)
    assert [episode.presentation for episode in again.episodes] == profiles
    assert [frame.tobytes() for frame in again.frames] == [frame.tobytes()
                                                         for frame in augmented.frames]


def test_appearance_augmentation_preserves_correction_targets_executed_context_and_failure():
    canonical = collect_scripted_corrections(
        CorrectionLearner(PhysicalAction(mouse_buttons=("left",))), 1,
        tasks=("mouse",), expert_probability=0)
    augmented = augment_demonstrations(canonical, seed=3)
    before, after = canonical.episodes[0], augmented.episodes[0]
    assert after.actions != after.executed_actions
    assert after.actions == before.actions and after.executed_actions == before.executed_actions
    assert after.outcome["failure"] and after.terminated
    assert after.source == CORRECTION_SOURCE
    combined = DemonstrationDataset()
    combined.extend(canonical)
    combined.extend(augmented)
    assert combined.provenance()["actor_presentation"] == (
        "canonical_and_train_appearance_augmentation")
    assert combined.provenance()["appearance_augmentation"]["episodes"] == 1


def test_appearance_augmentation_revalidates_train_bank_shape_and_metadata():
    canonical = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=0)
    episode = canonical.episodes[0]
    canonical.episodes[0] = replace(episode,
        level_id=_identity(atomic_level_banks("navigation")["test"][0]))
    with pytest.raises(ValueError, match="training bank"):
        augment_demonstrations(canonical)
    canonical.episodes[0] = episode
    index = episode.frame_indices[0]
    old = canonical.frames[index]
    canonical.frames[index] = np.frombuffer(old[:64, :64].tobytes(),
                                          dtype=np.uint8).reshape(64, 64, 3)
    with pytest.raises(ValueError, match="canonical native 96x96"):
        augment_demonstrations(canonical)
    canonical.frames[index] = old
    transformed = augment_demonstrations(canonical)
    with pytest.raises(ValueError, match="canonical native 96x96"):
        augment_demonstrations(transformed)
    profile = dict(transformed.episodes[0].presentation)
    profile["theme"] = "violet"
    with pytest.raises(ValueError, match="train appearances"):
        replace(transformed.episodes[0], presentation=profile)
    with pytest.raises(ValueError, match="train scripted_teacher"):
        replace(episode, split="test")
    bad_canvas = transformed.frames[0]
    transformed.frames[0] = np.frombuffer(bad_canvas[:-1].tobytes(), dtype=np.uint8).reshape(
        bad_canvas.shape[0] - 1, bad_canvas.shape[1], 3)
    with pytest.raises(ValueError, match="stable presentation canvas"):
        transformed._validate_episode(transformed.episodes[0])


def test_appearance_augmentation_deduplicates_repeated_rgb_and_enforces_budget():
    canonical = collect_scripted_corrections(CorrectionLearner(PhysicalAction()), 1,
        tasks=("navigation",), max_steps=5, expert_probability=0)
    augmented = augment_demonstrations(canonical)
    episode = augmented.episodes[0]
    assert len(set(episode.frame_indices)) == 1
    assert len(augmented.frames) == 1
    assert augmented.frame_bytes == augmented.frames[0].nbytes
    with pytest.raises(MemoryError, match="byte budget"):
        augment_demonstrations(canonical, max_frame_bytes=augmented.frame_bytes - 1)
    for invalid in (0, -1, True):
        with pytest.raises(ValueError, match="copies_per_episode"):
            augment_demonstrations(canonical, copies_per_episode=invalid)


def test_appearance_profile_never_enters_policy_replay(monkeypatch):
    canonical = collect_scripted_corrections(CorrectionLearner(PhysicalAction()), 1,
        tasks=("navigation",), max_steps=3, expert_probability=0)
    augmented = augment_demonstrations(canonical)
    agent = make_agent()
    evaluate = agent.evaluate_episode
    arguments = []

    def record(patches, actions, **kwargs):
        arguments.append(kwargs)
        return evaluate(patches, actions, **kwargs)

    monkeypatch.setattr(agent, "evaluate_episode", record)
    result = train_imitation(agent, augmented, epochs=1)
    assert set(arguments[0]) == {"instruction", "executed_actions"}
    assert arguments[0]["executed_actions"] == augmented.episodes[0].executed_actions
    assert result["dataset"]["actor_presentation"] == "train_appearance_augmentation"


def test_appearance_augmentation_preserves_independent_training_bank_membership():
    canonical = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=0,
                                               level_bank_seed=123)
    augmented = augment_demonstrations(canonical, seed=4)
    episode = augmented.episodes[0]
    assert episode.level_bank_seed == 123
    assert episode.level_id == _identity(atomic_level_banks("navigation", 123)["train"][
        episode.level_index])
    with pytest.raises(ValueError, match="training bank"):
        augmented.add_episode(replace(episode, level_bank_seed=0))


def test_learner_prefix_visits_actual_states_before_expert_takeover():
    learner = CorrectionLearner(PhysicalAction())
    teacher_inputs, teacher_feedback = [], []
    progress = []

    class RecordingTeacher:
        def __init__(self, task):
            self.delegate = scripted_teacher(task)

        def act(self, frame):
            assert not frame.flags.writeable
            teacher_inputs.append(frame.copy())
            return self.delegate.act(frame)

        def observe_action(self, action):
            teacher_feedback.append(action)
            self.delegate.observe_action(action)

    dataset = collect_scripted_corrections(learner, 1, tasks=("navigation",),
        max_steps=12, expert_probability=1, learner_prefix_steps=4,
        teacher_factory=RecordingTeacher, on_episode=progress.append)
    episode = dataset.episodes[0]
    assert len(episode) > 4
    assert episode.executed_actions[:4] == (PhysicalAction(),) * 4
    assert episode.actions[:4] != episode.executed_actions[:4]
    assert episode.executed_actions[4:] == episode.actions[4:]
    assert episode.outcome["success"]
    assert tuple(teacher_feedback) == episode.executed_actions
    for index, ((frame, previous), frame_index) in enumerate(zip(
            learner.inputs, episode.frame_indices, strict=True)):
        np.testing.assert_array_equal(frame, teacher_inputs[index])
        np.testing.assert_array_equal(frame, dataset.frames[frame_index])
        assert previous == (PhysicalAction() if index == 0 else episode.executed_actions[index - 1])
    assert progress[0]["expert_steps"] == len(episode) - 4
    assert progress[0]["learner_prefix_steps"] == 4
    assert episode.to_dict()["learner_prefix_steps"] == 4
    assert dataset.provenance()["correction_schedules"] == [
        {"learner_prefix_steps": 4, "expert_probability_after_prefix": 1.0, "episodes": 1}]
    augmented = augment_demonstrations(dataset)
    assert augmented.episodes[0].learner_prefix_steps == 4
    assert augmented.provenance()["correction_schedules"] == dataset.provenance()[
        "correction_schedules"]


def test_live_styled_corrections_separate_live_rgb_and_preserve_executed_history():
    from carryrl.envs.atomic_suite import SCORING_VERSION
    from carryrl.envs.presentation import PresentationConfig, PresentationProfile, present_frame

    learner = CorrectionLearner(PhysicalAction())
    teacher_inputs, feedback = [], []

    class Teacher:
        def __init__(self, task):
            self.delegate = scripted_teacher(task)

        def act(self, frame):
            assert frame.shape == (96, 96, 3) and not frame.flags.writeable
            with pytest.raises(ValueError):
                frame.setflags(write=True)
            teacher_inputs.append(frame.copy())
            return self.delegate.act(frame)

        def observe_action(self, action):
            feedback.append(action)
            self.delegate.observe_action(action)

    dataset = collect_scripted_corrections(learner, 1, tasks=("navigation",),
        max_steps=20, expert_probability=1, learner_prefix_steps=4,
        level_bank_seed=20261008, teacher_factory=Teacher,
        presentation_config=PresentationConfig(mode="varied", native_sizes=(96,),
            themes=("slate",), hud_layouts=("left",), style_split="train", seed=123))
    episode = dataset.episodes[0]
    assert learner.training and episode.collection_type == "live_styled"
    assert episode.scoring_version == SCORING_VERSION
    assert episode.executed_actions[:4] == (PhysicalAction(),) * 4
    assert episode.executed_actions[4:] == episode.actions[4:]
    assert tuple(feedback) == tuple(learner.feedback) == episode.executed_actions
    profile = PresentationProfile(**episode.presentation)
    for step, ((learner_rgb, previous), frame_index) in enumerate(zip(
            learner.inputs, episode.frame_indices, strict=True)):
        assert learner_rgb.shape != teacher_inputs[step].shape
        np.testing.assert_array_equal(learner_rgb, dataset.frames[frame_index])
        np.testing.assert_array_equal(learner_rgb, present_frame(teacher_inputs[step], profile))
        assert previous == (PhysicalAction() if step == 0 else episode.executed_actions[step - 1])
    assert dataset.provenance()["live_styled_corrections"]["episodes"] == 1
    assert episode.to_dict()["appearance_application"] == "live environment learner rollout"
    legacy = replace(episode, collection_type=None, scoring_version=None,
                     quality=None, quality_source=None)
    assert legacy.effective_scoring_version == "legacy.v0"
    assert "scoring_version" not in legacy.to_dict()


@pytest.mark.parametrize("options", [
    {"themes": ("violet",), "style_split": "validation"},
    {"themes": ("slate", "ice"), "style_split": "train"},
    {"themes": ("slate",), "style_split": None},
    {"native_sizes": (64,), "style_split": "train"},
])
def test_live_styled_rejects_nontrain_profiles_before_environment_creation(options):
    from carryrl.envs.presentation import PresentationConfig

    config = {"mode": "varied", "native_sizes": (96,), "themes": ("slate",), **options}
    with pytest.raises(ValueError, match="train-only"):
        collect_scripted_corrections(CorrectionLearner(PhysicalAction()), 1,
            tasks=("navigation",), presentation_config=PresentationConfig(**config),
            env_factory=lambda *a, **k: pytest.fail("invalid profile reached environment"))


def test_live_styled_failure_remains_valid_with_visible_provenance():
    from carryrl.envs.presentation import PresentationConfig

    dataset = collect_scripted_corrections(
        CorrectionLearner(PhysicalAction(mouse_buttons=("left",))), 1, tasks=("mouse",),
        expert_probability=0, level_bank_seed=20261008,
        presentation_config=PresentationConfig(mode="varied", native_sizes=(96,),
            themes=("forest",), style_split="train"))
    episode = dataset.episodes[0]
    assert episode.outcome["failure"] and episode.terminated
    assert episode.executed_actions != episode.actions
    assert episode.collection_type == "live_styled"


def test_live_styled_excludes_semantic_heldout_matches_from_additional_bank(monkeypatch):
    import carryrl.training.imitation as imitation
    from carryrl.envs.presentation import PresentationConfig

    real_banks = atomic_level_banks

    def overlapping(task, seed=0):
        banks = real_banks(task, seed)
        if seed == 20261009:
            return {**banks, "train": list(real_banks(task, 20261008)["validation"])}
        return banks

    monkeypatch.setattr(imitation, "atomic_level_banks", overlapping)
    monkeypatch.setattr("carryrl.envs.runtime_identity.atomic_level_banks", overlapping)
    with pytest.raises(ValueError, match="without held-out matches"):
        collect_scripted_corrections(CorrectionLearner(PhysicalAction()), 1,
            tasks=("navigation",), level_bank_seed=20261009,
            presentation_config=PresentationConfig(mode="varied", native_sizes=(96,),
                themes=("slate",), style_split="train"),
            env_factory=lambda *a, **k: pytest.fail("heldout collision reached environment"))


def test_live_styled_renderer_failure_restores_learner_mode_and_closes_environment():
    from carryrl.envs.presentation import PresentationConfig

    learner = CorrectionLearner(PhysicalAction())
    closed = []

    class BrokenRenderer(AtomicEnv):
        def canonical_render(self):
            raise RuntimeError("canonical rendering failed")

        def close(self):
            closed.append(True)
            super().close()

    with pytest.raises(RuntimeError, match="canonical rendering failed"):
        collect_scripted_corrections(learner, 1, tasks=("navigation",),
            level_bank_seed=20261008, env_factory=BrokenRenderer,
            presentation_config=PresentationConfig(mode="varied", native_sizes=(96,),
                themes=("slate",), style_split="train"))
    assert learner.training and closed == [True]
    assert not learner.inputs and not learner.feedback


@pytest.mark.parametrize("prefix", [0, 3, 12])
def test_learner_prefix_then_mixture_follows_isolated_rng_without_prefix_draws(prefix):
    dataset = collect_scripted_corrections(CorrectionLearner(PhysicalAction()), 1,
        tasks=("navigation",), seed=9, max_steps=12, expert_probability=0.5,
        learner_prefix_steps=prefix)
    episode = dataset.episodes[0]
    rng = np.random.default_rng(9)
    expected = []
    for step, label in enumerate(episode.actions):
        expected.append(label if step >= prefix and rng.random() < 0.5 else PhysicalAction())
    assert episode.executed_actions == tuple(expected)
    if prefix == 12:
        assert len(episode) == 12 and episode.truncated
        assert episode.executed_actions == (PhysicalAction(),) * 12
    if prefix == 0:
        implicit_default = collect_scripted_corrections(CorrectionLearner(PhysicalAction()), 1,
            tasks=("navigation",), seed=9, max_steps=12, expert_probability=0.5)
        assert implicit_default.episodes[0].to_dict() == episode.to_dict()
        assert "learner_prefix_steps" not in episode.to_dict()
        assert "correction_schedules" not in dataset.provenance()


@pytest.mark.parametrize("prefix", [-1, 9, True, False, 1.0, "2", None])
def test_learner_prefix_rejects_invalid_parameters_before_opening_environment(prefix):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid prefix must not open an environment")
    learner = CorrectionLearner(PhysicalAction())
    with pytest.raises(ValueError, match="learner_prefix_steps"):
        collect_scripted_corrections(learner, 1, tasks=("navigation",), max_steps=8,
            learner_prefix_steps=prefix, env_factory=forbidden)
    assert learner.training and not learner.inputs


def test_learner_prefix_provenance_retains_zero_and_nonzero_schedules_when_combining():
    learner = CorrectionLearner(PhysicalAction())
    plain = collect_scripted_corrections(learner, 1, tasks=("navigation",),
        max_steps=4, expert_probability=0)
    prefixed = collect_scripted_corrections(learner, 1, tasks=("navigation",),
        max_steps=4, expert_probability=1, learner_prefix_steps=4)
    plain.extend(prefixed)
    assert plain.provenance()["correction_schedules"] == [
        {"learner_prefix_steps": 0, "expert_probability_after_prefix": 0.0, "episodes": 1},
        {"learner_prefix_steps": 4, "expert_probability_after_prefix": 1.0, "episodes": 1}]
    with pytest.raises(ValueError, match="learner_prefix_steps"):
        replace(prefixed.episodes[0], learner_prefix_steps=True)


def test_learner_prefix_restarts_for_every_episode_and_records_early_end():
    dataset = collect_scripted_corrections(CorrectionLearner(PhysicalAction()), 3,
        tasks=("navigation",), max_steps=8, expert_probability=1, learner_prefix_steps=3)
    for episode in dataset.episodes:
        assert len(episode) > 3
        assert episode.executed_actions[:3] == (PhysicalAction(),) * 3
        assert episode.executed_actions[3:] == episode.actions[3:]
        assert episode.to_dict()["learner_prefix_steps_executed"] == 3
    failed = collect_scripted_corrections(
        CorrectionLearner(PhysicalAction(mouse_buttons=("left",))), 1,
        tasks=("mouse",), expert_probability=1, learner_prefix_steps=8).episodes[0]
    assert failed.outcome["failure"] and len(failed) == 1
    assert failed.to_dict()["learner_prefix_steps"] == 8
    assert failed.to_dict()["learner_prefix_steps_executed"] == 1


def test_active_idle_weights_cover_every_physical_input_and_sum_to_one():
    active = [PhysicalAction(keys=("e",)), PhysicalAction(mouse_buttons=("left",)),
              PhysicalAction(dx=4), PhysicalAction(dy=-2), PhysicalAction(wheel=1)]
    labels = active + [PhysicalAction()] * 15
    weights = imitation_action_weights(labels, action_balance="active_idle")
    np.testing.assert_allclose(weights[:5], np.full(5, 1 / 10))
    np.testing.assert_allclose(weights[5:], np.full(15, 1 / 30))
    assert weights.sum() == pytest.approx(1)
    assert weights[:5].sum() == pytest.approx(.5)
    assert weights[5:].sum() == pytest.approx(.5)
    for group in (active, [PhysicalAction()] * 7):
        np.testing.assert_array_equal(imitation_action_weights(group, action_balance="active_idle"),
                                      imitation_action_weights(group))


def test_balanced_objective_weights_expert_labels_and_preserves_executed_history():
    labels = [PhysicalAction()] * 3 + [PhysicalAction(keys=("e",))]
    executed = [PhysicalAction(keys=("right",))] * 4

    class RecordedLikelihood(nn.Module):
        def __init__(self):
            super().__init__()
            self.nll = nn.Parameter(torch.tensor([1., 3., 7., 9.]))
            self.history = []

        def evaluate_episode(self, patches, actions, **kwargs):
            assert actions == labels
            self.history.append(kwargs["executed_actions"])
            return {"log_prob": -self.nll}

    policy = RecordedLikelihood()
    objective = imitation_nll(policy, torch.zeros(4, 1, 384), labels,
                             executed_actions=executed, action_balance="active_idle")
    assert float(objective.detach()) == pytest.approx(.5 * (11 / 3 + 9))
    objective.backward()
    torch.testing.assert_close(policy.nll.grad, torch.tensor([1 / 6, 1 / 6, 1 / 6, .5]))
    assert policy.history == [executed]
    uniform = imitation_nll(policy, torch.zeros(4, 1, 384), labels, executed_actions=executed)
    assert torch.equal(uniform, policy.nll.mean())


@pytest.mark.parametrize("backend", ["torch", "mlx"])
@pytest.mark.parametrize("task", ["navigation", "memory"])
def test_balanced_single_group_keeps_uniform_objective_and_missing_metric_none(backend, task):
    if backend == "mlx":
        mx = pytest.importorskip("mlx.core")
        if not mx.metal.is_available():
            pytest.skip("MLX imitation requires Metal")
    dataset = (collect_scripted_demonstrations(1, tasks=(task,), difficulty=0)
        if task == "navigation" else collect_scripted_corrections(
            CorrectionLearner(PhysicalAction()), 1, tasks=(task,), max_steps=3,
            expert_probability=0))
    result = train_imitation(make_agent(), dataset, backend=backend, action_balance="active_idle")
    row = result["history"][0]
    assert row["objective"] == row["nll"]
    assert row["action_balance"] == "active_idle"
    present, missing = (("active_nll", "idle_nll") if task == "navigation"
                        else ("idle_nll", "active_nll"))
    assert row[present] == pytest.approx(row["nll"], abs=1e-6)
    assert row[missing] is None


def test_balanced_reporting_keeps_unweighted_nll_and_per_task_nll():
    dataset = collect_scripted_demonstrations(1, tasks=("memory",), difficulty=0)
    agent = make_agent()
    patches = {index: agent.encode_frames(torch.tensor(frame)[None])[0].cpu()
               for index, frame in enumerate(dataset.frames)}
    episode = dataset.episodes[0]
    with torch.no_grad():
        vector = -agent.evaluate_episode(torch.stack([patches[index]
            for index in episode.frame_indices]), list(episode.actions))["log_prob"]
    active = torch.tensor([bool(action.keys) for action in episode.actions])
    expected_objective = .5 * (vector[active].mean() + vector[~active].mean())
    updates = []
    result = train_imitation(agent, dataset, action_balance="active_idle",
                             feature_provider=patches.__getitem__, on_update=updates.append)
    row = result["history"][0]
    for actual in (row, updates[0]):
        assert actual["nll"] == pytest.approx(float(vector.mean()), abs=1e-6)
        assert actual["objective"] == pytest.approx(float(expected_objective), abs=1e-6)
        assert actual["active_nll"] == pytest.approx(float(vector[active].mean()), abs=1e-6)
        assert actual["idle_nll"] == pytest.approx(float(vector[~active].mean()), abs=1e-6)
    assert abs(row["objective"] - row["nll"]) > .1
    assert row["per_task_nll"]["memory"] == row["nll"]
    assert row["per_task_objective"]["memory"] == row["objective"]


def test_uniform_training_is_numerically_identical_to_original_mean_update():
    from copy import deepcopy

    dataset = collect_scripted_demonstrations(1, tasks=("memory",), difficulty=0)
    reference = make_agent()
    actual = deepcopy(reference)
    providers = {index: reference.encode_frames(torch.tensor(frame)[None])[0].cpu()
                 for index, frame in enumerate(dataset.frames)}
    optimizers = [torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad),
                                   lr=1e-4, weight_decay=.01) for agent in (reference, actual)]
    episode = dataset.episodes[0]
    patches = torch.stack([providers[index] for index in episode.frame_indices])
    old_loss = -reference.evaluate_episode(patches, list(episode.actions))["log_prob"].mean()
    torch.stack([old_loss]).mean().backward()
    torch.nn.utils.clip_grad_norm_([p for p in reference.parameters() if p.requires_grad], .5,
                                  error_if_nonfinite=True)
    optimizers[0].step()
    result = train_imitation(actual, dataset, optimizer=optimizers[1],
                             feature_provider=providers.__getitem__)
    assert result["history"][0]["nll"] == float(old_loss.detach())
    assert result["history"][0]["objective"] == result["history"][0]["nll"]
    for name, value in reference.state_dict().items():
        assert torch.equal(value, actual.state_dict()[name]), name
    left, right = (optimizer.state_dict()["state"] for optimizer in optimizers)
    for key, values in left.items():
        for name, value in values.items():
            assert torch.equal(value, right[key][name]), (key, name)


def test_balanced_torch_metal_gradients_optimizer_and_64_step_memory_write_match():
    from copy import deepcopy

    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("MLX imitation requires Metal")
    import mlx.nn as mnn

    from carryrl.training.imitation import _episode_nlls
    from carryrl.training.mlx import MLXPPOBackend

    dataset = collect_scripted_corrections(CorrectionLearner(PhysicalAction(keys=("right",))), 1,
        tasks=("memory",), difficulty=1, expert_probability=1, learner_prefix_steps=4,
        episode_instructions={"memory": ("Remember the cue",)})
    episode = dataset.episodes[0]
    assert len(episode) > 64 and episode.actions != episode.executed_actions
    torch_agent = make_agent(instruction=True, segment_length=64)
    mlx_agent = deepcopy(torch_agent)
    providers = {index: torch_agent.encode_frames(torch.tensor(frame)[None])[0].cpu()
                 for index, frame in enumerate(dataset.frames)}
    patches = torch.stack([providers[index] for index in episode.frame_indices])
    optimizers = [torch.optim.AdamW((p for p in agent.parameters() if p.requires_grad),
                                   lr=3e-4, weight_decay=.03)
                  for agent in (torch_agent, mlx_agent)]
    engine = MLXPPOBackend(mlx_agent, optimizers[1])
    torch_loss = imitation_nll(torch_agent, patches, episode.actions,
        instruction=episode.instruction, executed_actions=episode.executed_actions,
        action_balance="active_idle")
    torch_loss.backward()
    feature = mx.array(torch_agent._encode_instruction(episode.instruction).numpy())

    def objective():
        output = engine.policy.evaluate_episode(patches, episode.actions,
            instruction_features=feature, executed_actions=episode.executed_actions)
        return _episode_nlls(output["log_prob"], episode.actions, "active_idle")[0]

    loss, gradients = mnn.value_and_grad(engine.policy, objective)()
    mx.eval(loss, gradients)
    assert float(loss) == pytest.approx(float(torch_loss.detach()), abs=3e-6)
    gradient = torch_agent.temporal.write_tokens.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    for name, parameter in torch_agent.named_parameters():
        if parameter.requires_grad:
            actual = torch.from_numpy(np.array(gradients["weights"][name]))
            expected = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            torch.testing.assert_close(actual, expected, atol=4e-6, rtol=5e-4)
    torch_agent.zero_grad(set_to_none=True)
    torch_result = train_imitation(torch_agent, dataset, epochs=2, optimizer=optimizers[0],
        action_balance="active_idle", feature_provider=providers.__getitem__)
    gpu_features = {index: mx.array(value.numpy()) for index, value in providers.items()}
    mlx_result = train_imitation(mlx_agent, dataset, epochs=2, optimizer=optimizers[1],
        backend="mlx", mlx_backend=engine, action_balance="active_idle",
        feature_provider=gpu_features.__getitem__)
    for left, right in zip(torch_result["history"], mlx_result["history"], strict=True):
        for metric in ("nll", "objective", "active_nll", "idle_nll"):
            assert right[metric] == pytest.approx(left[metric], abs=4e-6)
    for name, parameter in torch_agent.named_parameters():
        torch.testing.assert_close(parameter, dict(mlx_agent.named_parameters())[name],
                                   atol=4e-6, rtol=3e-4)
    states = [optimizer.state_dict() for optimizer in optimizers]
    assert set(states[0]["state"]) == set(states[1]["state"])
    for key, values in states[0]["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, states[1]["state"][key][name],
                                       atol=3e-7, rtol=3e-4)
    assert "temporal.write_tokens" in engine.optimizer.state
    assert "value_head.weight" not in engine.optimizer.state


def test_action_balance_rejects_unknown_mode_before_training():
    dataset = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=0)
    with pytest.raises(ValueError, match="action_balance"):
        train_imitation(make_agent(), dataset, action_balance="onsets")
    with pytest.raises(ValueError, match="action_balance"):
        imitation_action_weights([PhysicalAction()], action_balance="onsets")


def test_memory_counterparts_preserve_banks_and_collect_only_safe_full_cue_pairs():
    from copy import deepcopy

    from carryrl.envs.atomic import TARGET, _centroid, atomic_semantic_level_id

    bank_seed = 20261008
    banks = atomic_level_banks("memory", bank_seed)
    before = deepcopy(banks)
    origins = collect_scripted_demonstrations(64, tasks=("memory",), difficulty=1,
        level_bank_seed=bank_seed, episode_instructions={"memory": ("Remember the green door",)})
    origin_metadata = [episode.to_dict() for episode in origins.episodes]
    origin_frames = [frame.tobytes() for frame in origins.frames]
    progress = []
    counterparts = collect_memory_cue_counterparts(origins, max_pairs=64,
                                                    on_episode=progress.append)
    assert banks == before
    assert [episode.to_dict() for episode in origins.episodes] == origin_metadata
    assert [frame.tobytes() for frame in origins.frames] == origin_frames
    assert len(counterparts.episodes) == 63
    report = counterparts.memory_cue_counterpart_reports[0]
    assert report["origins_attempted"] == report["eligible_origins"] == 64
    assert report["pairs_collected"] == 63 and report["unattempted_origins"] == 0
    assert len(report["skipped_origins"]) == 1
    skipped = report["skipped_origins"][0]
    assert skipped["level_index"] == 62 and skipped["reason"] == "heldout_semantic_collision"
    assert any(match["split"] == "test" and match["level_index"] == 10
               and match["level_bank_seed"] == bank_seed for match in skipped["matches"])
    heldout = {atomic_semantic_level_id("memory", level)
        for seed in (0, bank_seed) for split in ("validation", "test")
        for level in atomic_level_banks("memory", seed)[split]}
    for paired in counterparts.episodes:
        origin = origins.episodes[paired.level_index]
        level = banks["train"][origin.level_index]
        runtime = paired.runtime_level_override
        assert paired.source == MEMORY_COUNTERPART_SOURCE
        assert paired.origin_level_id == paired.level_id == origin.level_id
        assert paired.runtime_level_id == _identity(runtime)
        assert runtime == {**level, "cue": 1 - level["cue"]}
        assert atomic_semantic_level_id("memory", runtime) not in heldout
        assert paired.instruction == origin.instruction
        assert paired.outcome["success"] and origin.outcome["success"]
        assert len(paired) > 64 and len(origin) > 64
        active = [action for action in paired.actions if action.keys]
        original_active = [action for action in origin.actions if action.keys]
        assert active and original_active and active[0].keys != original_active[0].keys
        assert all(action == PhysicalAction() for action in paired.actions[:64])
        old_cue = _centroid(origins.frames[origin.frame_indices[0]], TARGET)
        new_cue = _centroid(counterparts.frames[paired.frame_indices[0]], TARGET)
        assert (old_cue[0] < 48) != (new_cue[0] < 48)
        for tick in range(level["cue_ticks"], level["unlock"]):
            np.testing.assert_array_equal(origins.frames[origin.frame_indices[tick]],
                counterparts.frames[paired.frame_indices[tick]])
    assert sum(row["status"] == "collected" for row in progress) == 63
    assert sum(row["status"] == "skipped" for row in progress) == 1
    assert progress[-1]["pairs_collected"] == 63
    assert counterparts.provenance()["memory_cue_counterparts"][
        "runtime_metadata_input_to_actor"] is False


def test_memory_counterpart_rejects_forged_origin_runtime_source_reward_and_rgb():
    origins = collect_scripted_demonstrations(1, tasks=("memory",), difficulty=1,
                                            level_bank_seed=20261008)
    pairs = collect_memory_cue_counterparts(origins, max_pairs=1)
    episode = pairs.episodes[0]
    with pytest.raises(ValueError, match="training bank"):
        pairs.add_episode(replace(episode, level_id="forged"))
    with pytest.raises(ValueError, match="origin must match"):
        pairs.add_episode(replace(episode, origin_level_id="forged"))
    for override in ({**episode.runtime_level_override, "reward": 5},
                     {**episode.runtime_level_override, "unlock": 2},
                     {**episode.runtime_level_override,
                      "cue": 1 - episode.runtime_level_override["cue"]}):
        with pytest.raises(ValueError, match="only cue flipped"):
            pairs.add_episode(replace(episode, runtime_level_override=override,
                                      runtime_level_id=_identity(override)))
    with pytest.raises(ValueError, match="explicit source"):
        replace(episode, source=SOURCE)
    with pytest.raises(ValueError, match="full-difficulty"):
        replace(episode, difficulty=0)
    with pytest.raises(ValueError, match="final visible RGB"):
        pairs.add_episode(replace(episode, outcome={**episode.outcome, "reward": 1}))
    with pytest.raises(ValueError, match="neutral-wait RGB teacher"):
        pairs.add_episode(replace(episode, actions=(PhysicalAction(keys=("left",)),
                                                   *episode.actions[1:]),
                                  quality=None, quality_source=None))
    wrong = pairs.intern_frame(origins.frames[origins.episodes[0].frame_indices[0]])
    with pytest.raises(ValueError, match="trace RGB"):
        pairs.add_episode(replace(episode, frame_indices=(wrong, *episode.frame_indices[1:])))


def test_memory_counterparts_skip_known_heldout_match_without_counterpart_rollout():
    from carryrl.training.imitation import _flipped_memory_level

    seed = 20261008
    origins = collect_scripted_demonstrations(1, tasks=("memory",), seed=62,
                                            difficulty=1, level_bank_seed=seed)
    level = atomic_level_banks("memory", seed)["train"][62]
    assert _flipped_memory_level(level) == atomic_level_banks("memory", seed)["test"][10]
    pairs = collect_memory_cue_counterparts(origins)
    assert not pairs.episodes and not pairs.frames
    report = pairs.provenance()["memory_cue_counterparts"]["reports"][0]
    assert report["pairs_collected"] == 0 and report["origins_attempted"] == 1
    assert len(report["skipped_origins"]) == 1
    assert "memory_cue_counterparts" not in origins.provenance()
    assert all(field not in origins.episodes[0].to_dict() for field in
               ("origin_level_id", "runtime_level_id", "runtime_level_override"))


def test_memory_counterpart_collection_quality_gate_and_origin_trace_are_not_weakened():
    origins = collect_scripted_demonstrations(1, tasks=("memory",), difficulty=1)

    class IdleTeacher:
        def act(self, frame):
            assert not frame.flags.writeable
            return PhysicalAction()

    with pytest.raises(RuntimeError, match="quality gate failed"):
        collect_memory_cue_counterparts(origins, teacher_factory=lambda _: IdleTeacher())
    original = origins.episodes[0]
    origins.episodes[0] = replace(original,
        actions=(PhysicalAction(keys=("left",)), *original.actions[1:]),
        quality=None, quality_source=None)
    with pytest.raises(ValueError, match="neutral-wait RGB teacher"):
        collect_memory_cue_counterparts(origins)
    for invalid in (-1, 0, 65, True):
        with pytest.raises(ValueError, match="max_pairs"):
            collect_memory_cue_counterparts(origins, max_pairs=invalid)


def test_memory_counterpart_metadata_does_not_enter_training_policy_and_cap_is_honest(monkeypatch):
    origins = collect_scripted_demonstrations(2, tasks=("memory",), difficulty=1)
    pairs = collect_memory_cue_counterparts(origins, max_pairs=1)
    report = pairs.memory_cue_counterpart_reports[0]
    assert report["eligible_origins"] == 2 and report["origins_attempted"] == 1
    assert report["unattempted_origins"] == 1 and report["pairs_collected"] == 1
    agent = make_agent(segment_length=64)
    evaluate = agent.evaluate_episode
    kwargs_seen = []

    def recorded(patches, actions, **kwargs):
        kwargs_seen.append(kwargs)
        return evaluate(patches, actions, **kwargs)

    monkeypatch.setattr(agent, "evaluate_episode", recorded)
    result = train_imitation(agent, pairs)
    assert set(kwargs_seen[0]) == {"instruction", "executed_actions"}
    assert result["source"] == MEMORY_COUNTERPART_SOURCE
    combined = DemonstrationDataset()
    combined.extend(origins)
    combined.extend(pairs)
    assert combined.provenance()["memory_cue_counterparts"]["episodes"] == 1
    assert combined.provenance()["memory_cue_counterparts"]["reports"] == [report]


def test_memory_pairs_cli_rejects_invalid_flag_before_loading_configuration(monkeypatch, capsys):
    import sys

    from test_imitation_feature_storage import script

    for count in ("-1", "65"):
        monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "missing.json",
            "--warm-start", "missing", "--output", "unused", "--memory-cue-pairs", count])
        with pytest.raises(SystemExit) as error:
            script.main()
        assert error.value.code == 2
        assert "memory cue pairs must be in [0,64]" in capsys.readouterr().err


def test_memory_pairs_cli_accounts_actual_accepted_skipped_and_combined_budget(tmp_path):
    from test_imitation_feature_storage import script

    from carryrl.training.progress import TrainingProgress

    origins = collect_scripted_demonstrations(2, tasks=("memory",), difficulty=1,
                                            level_bank_seed=20261008)
    progress = TrainingProgress(tmp_path / "accepted.json", epochs_total=1)
    progress.phase("collecting", total=2, unit="episodes")
    metadata = script._add_memory_cue_pairs(origins, max_pairs=4, max_steps=80,
        frame_budget=origins.max_frame_bytes, progress=progress)
    assert metadata["episodes"] == 2
    assert progress.state["memory_pairs_requested"] == 4
    assert progress.state["memory_pairs_collected"] == 2
    assert progress.state["phase_units_done"] == progress.state["phase_units_total"] == 4
    assert progress.state["memory_origins_eligible"] == 2
    collision = collect_scripted_demonstrations(1, tasks=("memory",), seed=62,
                                              difficulty=1, level_bank_seed=20261008)
    rejected = TrainingProgress(tmp_path / "skipped.json", epochs_total=1)
    metadata = script._add_memory_cue_pairs(collision, max_pairs=64, max_steps=80,
        frame_budget=collision.max_frame_bytes, progress=rejected)
    assert metadata["episodes"] == 0
    assert rejected.state["memory_pairs_requested"] == 64
    assert rejected.state["memory_pairs_collected"] == 0
    assert rejected.state["memory_pairs_skipped"] == 1
    assert rejected.state["phase_units_done"] == rejected.state["phase_units_total"] == 1
    limited = collect_scripted_demonstrations(2, tasks=("memory",), difficulty=1,
                                            level_bank_seed=20261008)
    original_count, original_bytes = len(limited.episodes), limited.frame_bytes
    budget_progress = TrainingProgress(tmp_path / "budget.json", epochs_total=1)
    with pytest.raises(MemoryError, match="aggregate RGB frame budget"):
        script._add_memory_cue_pairs(limited, max_pairs=1, max_steps=80,
            frame_budget=limited.frame_bytes, progress=budget_progress)
    assert len(limited.episodes) == original_count and limited.frame_bytes == original_bytes
