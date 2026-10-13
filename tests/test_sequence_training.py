"""Offline integration checks for history replay, PPO weighting, and atomic continuation."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.model import AgentConfig, GeneralistAgent
from carryrl.envs.atomic import AtomicEnv
from carryrl.envs.atomic_suite import TASKS
from carryrl.training.sequence import (
    Episode,
    FrozenFrameCache,
    collect_episode,
    evaluate_sequence,
    gae,
    load_sequence_checkpoint,
    prepare_reward,
    tensor_hash,
    train_sequence,
    update_ppo,
)
from carryrl.training.sequence_config import SequenceRunConfig


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 384)

    def forward(self, frames):
        assert frames.dtype == torch.uint8 and frames.shape[-1] == 3
        small = F.adaptive_avg_pool2d(frames.permute(0, 3, 1, 2).float() / 255, (4, 4))
        return self.projection(small.flatten(2).transpose(1, 2))


class TinyReward(nn.Module):
    def __init__(self, image_size=96):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def predict(self, frame):
        return {"score": float(self.scale.detach()) * float(frame[..., 0].mean()) / 255,
                "success_probability": 0.5, "failure_probability": 0.5}


class TinyLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, instruction):
        self.calls.append(instruction)
        frequency = sum(map(ord, instruction)) % 31 + 1
        return torch.sin(torch.arange(384, device=self.scale.device) * frequency / 384) * self.scale


def factory(config):
    return GeneralistAgent(config, vision=TinyVision())


def config(**changes):
    return SequenceRunConfig(agent=AgentConfig(pretrained=False, image_size=28, width=32,
                             layers=1, heads=4, spatial_tokens=4, memory_tokens=2,
                             segment_length=2), tasks=("navigation", "memory"),
                             total_rollouts=1, max_episode_steps=4, n_epochs=1,
                             eval_episodes_per_task=1, torch_threads=1, **changes)


def instruction_config(**changes):
    base = config()
    return replace(base, agent=replace(base.agent, text_encoder="multilingual_e5_small"),
                   episode_instructions={
                       "navigation": ("Reach the green exit.", "  Дойди до зелёного выхода.  "),
                       "memory": ("Remember and repeat the flashing sequence.",
                                  "Повтори показанную последовательность."),
                   }, **changes)


def instruction_factory(agent_config):
    return GeneralistAgent(agent_config, vision=TinyVision(), language=TinyLanguage())


def test_frozen_evaluation_callbacks_actual_final_rgb_and_progress_preserve_policy():
    from carryrl.envs.atomic import verify_visible_outcome

    torch.set_num_threads(1)
    cfg = config()
    agent, reward = factory(cfg.agent), TinyReward()
    events, captures = [], []
    before = tensor_hash(agent)
    evaluated = evaluate_sequence(
        agent, reward, cfg, on_progress=events.append,
        on_playback=lambda row, frames, actions: captures.append((row, frames, actions)))
    assert len(captures) == len(cfg.tasks)
    completed = [row for row in events if row["event"] == "episode_completed"]
    assert [row["episodes_completed"] for row in completed] == [1, 2, 3, 4]
    assert all(row["episodes_total"] == 4 for row in events)
    for record, frames, actions in captures:
        assert record["policy"] == "frozen_deterministic" and record["episode_index"] == 0
        assert len(frames) == len(actions) + 1 == record["steps"] + 1
        assert verify_visible_outcome(frames[-1]) == record["outcome"]
        assert all({"keys", "mouse_buttons", "dx", "dy", "wheel"} <= set(action)
                   for action in actions)
        # Captured post-action observations are independent copies, including terminal RGB.
        assert all(not np.shares_memory(left, right)
                   for left, right in zip(frames, frames[1:]))
    assert tensor_hash(agent) == before and agent.training
    assert evaluated["elapsed_seconds"] >= 0 and len(evaluated["protocol_sha256"]) == 64


def test_gae_terminal_and_truncation_final_history_bootstrap():
    rewards, values = torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.6])
    terminal, targets = gae(rewards, values, [False, True], [False, False], 100, 0.99, 0.95)
    truncated, _ = gae(rewards, values, [False, False], [False, True], 3, 0.99, 0.95)
    assert terminal[-1].item() == pytest.approx(1.4)
    assert truncated[-1].item() == pytest.approx(4.37)
    assert terminal[0].item() == pytest.approx(1.094 + 0.99 * 0.95 * 1.4)
    assert torch.allclose(targets, terminal + values)
    with pytest.raises(ValueError, match="one complete episode"):
        gae(rewards, values, [False, False], [True, True], 3, 0.99, 0.95)


def test_collection_replay_ratios_identity_and_effective_update_with_frozen_reward():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = config()
    agent, reward = factory(cfg.agent), TinyReward().eval().requires_grad_(False)
    episode = collect_episode(agent, reward, cfg, task="navigation", seed=21)
    assert len(episode) == 4 and episode.truncated[-1]
    assert episode.reward_uncertain_steps > 0  # Uncalibrated uncertainty never becomes a hard gate.
    replay = agent.evaluate_episode(episode.patches, episode.actions)
    assert torch.allclose(replay["log_prob"].detach(), episode.old_log_prob, atol=2e-5)
    assert torch.allclose(replay["value"].detach(), episode.old_values, atol=2e-5)
    assert torch.allclose((replay["log_prob"].detach() - episode.old_log_prob).exp(),
                          torch.ones(4), atol=2e-5)
    with torch.no_grad():
        final_patch = agent.encode_frames(torch.as_tensor(episode.final_frame)[None])[0].cpu()
        extended = agent.evaluate_episode(torch.cat([episode.patches, final_patch[None]]),
                                          [*episode.actions, PhysicalAction()])
        assert episode.bootstrap == pytest.approx(float(extended["value"][-1]), abs=2e-5)
        agent.reset_episode()
        reset_bootstrap = agent.peek_value_from_patches(final_patch)
        assert abs(episode.bootstrap - reset_bootstrap) > 1e-5
    old_actor = {name: tensor.clone() for name, tensor in agent.decoder.state_dict().items()}
    old_value = agent.value_head.weight.detach().clone()
    old_vision, old_reward = tensor_hash(agent.vision), tensor_hash(reward)
    optimizer = torch.optim.Adam([p for p in agent.parameters() if p.requires_grad], lr=0.001)
    metrics = update_ppo(agent, optimizer, [episode], cfg)
    assert metrics["valid_transitions"] == 4
    assert any(not torch.equal(old_actor[name], value)
               for name, value in agent.decoder.state_dict().items())
    assert not torch.equal(old_value, agent.value_head.weight)
    assert tensor_hash(agent.vision) == old_vision
    assert tensor_hash(reward) == old_reward


def test_collection_ignores_hidden_api_reward_including_nan():
    class HiddenRewardEnv(AtomicEnv):
        def step(self, action):
            frame, _, terminated, truncated, info = super().step(action)
            return frame, float("nan"), terminated, truncated, info

    torch.set_num_threads(1)
    cfg = config()
    episode = collect_episode(factory(cfg.agent), TinyReward().eval().requires_grad_(False),
                              cfg, task="navigation", seed=21, env_factory=HiddenRewardEnv)
    assert torch.isfinite(episode.rewards).all()
    assert len(episode) == cfg.max_episode_steps


def test_ppo_weights_gradient_by_valid_transitions_not_episode_mean():
    class LinearAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Parameter(torch.zeros(()))
            self.critic = nn.Parameter(torch.zeros(()))

        def evaluate_episode(self, patches, actions):
            xs = patches[:, 0, 0]
            return {"log_prob": self.actor * xs, "value": self.critic.expand(len(actions)),
                    "entropy": torch.zeros(len(actions))}

    agent = LinearAgent()
    episodes = []
    for xs, rewards in (([1.0, 2.0], [1.0, 0.0]), ([3.0, 4.0, 5.0, 6.0], [0., 0., 0., 2.])):
        patches = torch.zeros(len(xs), 1, 384)
        patches[:, 0, 0] = torch.tensor(xs)
        episodes.append(Episode(patches, np.zeros((len(xs), 96, 96, 3), dtype=np.uint8),
                        [PhysicalAction()] * len(xs), torch.zeros(len(xs)), torch.zeros(len(xs)),
                        torch.tensor(rewards), [False] * (len(xs) - 1) + [True],
                        [False] * len(xs), 0., "navigation", 0, {}))
    cfg = replace(config(), value_coef=0, entropy_coef=0, max_grad_norm=1e6)
    raw = torch.cat([gae(ep.rewards, ep.old_values, ep.terminated, ep.truncated,
                          0., cfg.gamma, cfg.gae_lambda)[0] for ep in episodes])
    advantages = (raw - raw.mean()) / raw.std(unbiased=False)
    expected = 0.1 * (advantages * torch.arange(1., 7.)).sum() / 6
    update_ppo(agent, torch.optim.SGD(agent.parameters(), lr=0.1), episodes, cfg)
    assert agent.actor.item() == pytest.approx(float(expected), abs=1e-6)


def test_atomic_checkpoint_resume_optimizer_frozen_evaluation_and_no_overwrite(tmp_path):
    cfg = config()
    first = train_sequence(cfg, tmp_path / "first", agent_factory=factory, reward=TinyReward(),
                           reward_factory=TinyReward)
    assert first["status"] == "completed" and first["decisions"] == 8
    checkpoint = Path(first["latest_checkpoint"])
    built = []

    def offline_factory(agent_config):
        built.append(agent_config.pretrained)
        return factory(agent_config)

    resumed_config = replace(cfg, total_rollouts=2)
    agent, optimizer, reward, progress = load_sequence_checkpoint(
        checkpoint, resumed_config, agent_factory=offline_factory, reward_factory=TinyReward)
    assert built == [False] and optimizer.state_dict()["state"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["weight_decay"] == cfg.weight_decay
    assert progress["weights_revision"] == 1
    assert progress["vision_weight_id"] == (
        "facebook/dinov2-small" if cfg.agent.pretrained else None
    )
    before_agent, before_reward = tensor_hash(agent), tensor_hash(reward)
    evaluation = evaluate_sequence(agent, reward, cfg)
    assert evaluation["split"] == "validation"
    assert set(evaluation["summaries"]["frozen_deterministic"]["tasks"]) == set(cfg.tasks)
    assert tensor_hash(agent) == before_agent and tensor_hash(reward) == before_reward
    second = train_sequence(resumed_config, tmp_path / "second", checkpoint,
                            agent_factory=offline_factory, reward_factory=TinyReward)
    assert second["rollouts"] == 2 and second["decisions"] == 16
    assert second["initial_rollout"] == 1 and second["initial_decisions"] == 8
    assert second["attempted_train_decisions_this_process"] == 8
    metadata = json.loads((tmp_path / "second/metadata.json").read_text())
    assert metadata["resume_rng_restored"] is False
    records = list((tmp_path / "first/trajectories/rollout_0001").glob("*.npz"))
    with np.load(records[0], allow_pickle=False) as record:
        assert record["frames"].dtype == np.uint8 and len(record["old_log_prob"]) == 4
    with pytest.raises(FileExistsError):
        train_sequence(cfg, tmp_path / "first", agent_factory=factory, reward=TinyReward())
    archive = checkpoint / "state.pt"
    archive.write_bytes(archive.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="SHA256"):
        load_sequence_checkpoint(checkpoint, resumed_config, agent_factory=factory,
                                 reward_factory=TinyReward)


def test_reproducibility_and_failure_status_env_cleanup(tmp_path):
    cfg = config()
    first = train_sequence(cfg, tmp_path / "a", agent_factory=factory, reward=TinyReward(),
                           reward_factory=TinyReward)
    second = train_sequence(cfg, tmp_path / "b", agent_factory=factory, reward=TinyReward(),
                            reward_factory=TinyReward)
    assert first["evaluation_summaries"] == second["evaluation_summaries"]
    a = torch.load(Path(first["latest_checkpoint"]) / "state.pt", weights_only=True)
    b = torch.load(Path(second["latest_checkpoint"]) / "state.pt", weights_only=True)
    assert all(torch.equal(value, b["agent_state"][name])
               for name, value in a["agent_state"].items())
    closed = []

    class BrokenEnv(AtomicEnv):
        def step(self, action):
            raise RuntimeError("synthetic adapter failure")

        def close(self):
            closed.append(True)
            super().close()

    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="adapter failure"):
        train_sequence(cfg, output, agent_factory=factory, reward=TinyReward(),
                       env_factory=BrokenEnv, reward_factory=TinyReward)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed" and status["latest_checkpoint_retained"]
    assert status["attempted_train_decisions_this_process"] == 1 and closed == [True]


@pytest.mark.parametrize("changes", [
    {"unknown": 1}, {"seed": True}, {"device": "auto"}, {"max_episode_steps": 129},
    {"tasks": ["navigation", "navigation"]}, {"learning_rate": float("nan")},
    {"reward_bootstrap_episodes_per_task": 1},
])
def test_config_rejects_invalid_contract(changes):
    with pytest.raises((ValueError, TypeError)):
        SequenceRunConfig.from_dict({**config().to_dict(), **changes})


def test_cli_v1_branch_and_legacy_regression(tmp_path, monkeypatch):
    import carryrl.cli as cli
    import carryrl.training.sequence as sequence

    path = tmp_path / "sequence.json"
    path.write_text(json.dumps(config().to_dict()))
    calls = []
    monkeypatch.setattr(sequence, "train_sequence", lambda *args, **kwargs: (
        calls.append(args[0]) or {"status": "resource_stopped"}))
    assert cli.main(["train-v1", "--config", str(path), "--output", str(tmp_path / "run")]) == 1
    assert isinstance(calls[0], SequenceRunConfig)
    monkeypatch.setattr(cli, "train", lambda *args: {"status": "completed"})
    old_path = tmp_path / "old.json"
    old_path.write_text("{}")
    assert cli.main(["train", "--config", str(old_path), "--output", str(tmp_path / "old")]) == 0


def test_complete_suite_registry_and_legacy_explicit_task_config():
    assert TASKS == ("navigation", "reaction", "mouse", "memory", "timing", "platforming",
                     "steering", "camera", "interaction", "causality", "key_door",
                     "goal_choice", "hazards", "shooting", "ui", "chain")
    assert SequenceRunConfig().tasks == TASKS
    old = replace(config(), tasks=TASKS[:4]).to_dict()
    old.pop("eval_every_rollouts")
    loaded = SequenceRunConfig.from_dict(old)
    assert loaded.tasks == TASKS[:4] and loaded.eval_every_rollouts == 1
    with pytest.raises(ValueError, match="eval_every_rollouts"):
        replace(loaded, eval_every_rollouts=0)


def test_reward_bootstrap_collects_only_selected_tasks(tmp_path, monkeypatch):
    import carryrl.envs.atomic as atomic
    import carryrl.rewards.visual as visual

    calls = []

    def collect(**kwargs):
        calls.append(kwargs)
        return (np.zeros((1, 96, 96, 3), dtype=np.uint8),
                {"score": np.zeros(1), "success": np.zeros(1), "failure": np.zeros(1)},
                {"tasks": list(kwargs["tasks"])})

    def fit(model, *args, **kwargs):
        model.eval().requires_grad_(False)
        return {"fit_is_synthetic": True}

    monkeypatch.setattr(atomic, "collect_bootstrap_dataset", collect)
    monkeypatch.setattr(visual, "train_visual_reward", fit)
    cfg = config()
    reward = prepare_reward(cfg, tmp_path, TinyReward)
    assert [call["split"] for call in calls] == ["train", "validation"]
    assert all(call["tasks"] == cfg.tasks for call in calls)
    assert all(not parameter.requires_grad for parameter in reward.parameters())
    recorded = json.loads((tmp_path / "reward_bootstrap.json").read_text())
    assert recorded["fit_labels_provenance"]["tasks"] == list(cfg.tasks)


def test_periodic_evaluation_is_frozen_and_final_rollout_always_recorded(tmp_path, monkeypatch):
    import carryrl.training.sequence as sequence

    evaluations = []
    original = sequence.evaluate_sequence

    def tracked(agent, reward, *args, **kwargs):
        before = tensor_hash(agent), tensor_hash(reward)
        result = original(agent, reward, *args, **kwargs)
        assert before == (tensor_hash(agent), tensor_hash(reward))
        evaluations.append(before)
        return result

    monkeypatch.setattr(sequence, "evaluate_sequence", tracked)
    cfg = replace(config(), total_rollouts=3, eval_every_rollouts=2)
    summary = train_sequence(cfg, tmp_path / "periodic", agent_factory=factory,
                             reward=TinyReward(), reward_factory=TinyReward)
    records = sorted((tmp_path / "periodic/evaluations").glob("*.json"))
    assert [path.name for path in records] == ["rollout_0002.json", "rollout_0003.json"]
    assert len(evaluations) == 2 and evaluations[0][1] == evaluations[1][1]
    assert evaluations[0][0] != evaluations[1][0]
    assert summary["evaluation_rollout"] == 3
    final = json.loads(records[-1].read_text())
    assert json.loads((tmp_path / "periodic/evaluation.json").read_text()) == final
    assert final["decisions"] == 24 and final["weights_revision"] == 3
    assert final["wall_seconds"] > 0 and final["wall_budget_overshoot_seconds"] == 0
    assert set(final["summaries"]["frozen_deterministic"]["tasks"]) == set(cfg.tasks)


@pytest.mark.parametrize("stop_rollout", [2, 3])
def test_budget_stop_evaluates_latest_saved_policy_once(tmp_path, monkeypatch, stop_rollout):
    import carryrl.training.sequence as sequence

    clock = {"now": 0.0, "rollout": 0}
    evaluations = []
    original_update, original_evaluate = sequence.update_ppo, sequence.evaluate_sequence

    def update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        clock["rollout"] += 1
        clock["now"] = 11.0 if clock["rollout"] == stop_rollout else float(clock["rollout"])
        return result

    def evaluate(agent, reward, *args, **kwargs):
        before = tensor_hash(agent), tensor_hash(reward)
        result = original_evaluate(agent, reward, *args, **kwargs)
        assert before == (tensor_hash(agent), tensor_hash(reward))
        evaluations.append(before)
        clock["now"] += 0.25
        return result

    monkeypatch.setattr(sequence.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(sequence, "update_ppo", update)
    monkeypatch.setattr(sequence, "evaluate_sequence", evaluate)
    cfg = replace(config(), total_rollouts=5, eval_every_rollouts=2, max_wall_seconds=10)
    output = tmp_path / "budget"
    summary = train_sequence(cfg, output, agent_factory=factory, reward=TinyReward(),
                             reward_factory=TinyReward)

    assert summary["status"] == "resource_stopped"
    assert summary["rollouts"] == summary["evaluation_rollout"] == stop_rollout
    records = sorted((output / "evaluations").glob("*.json"))
    expected = ["rollout_0002.json"]
    if stop_rollout == 3:
        expected.append("rollout_0003.json")
    assert [path.name for path in records] == expected
    assert len(evaluations) == len(expected)
    assert len({reward_hash for _, reward_hash in evaluations}) == 1
    if stop_rollout == 3:
        assert evaluations[0][0] != evaluations[1][0]
    final = json.loads((output / "evaluation.json").read_text())
    checkpoint = json.loads((Path(summary["latest_checkpoint"]) / "manifest.json").read_text())
    assert final == json.loads(records[-1].read_text())
    assert final["rollout"] == final["weights_revision"] == stop_rollout
    assert final["agent_sha256"] == checkpoint["agent_sha256"] == evaluations[-1][0]
    assert final["reward_sha256"] == checkpoint["reward_sha256"] == evaluations[-1][1]
    assert summary["evaluation_summaries"] == final["summaries"]
    assert final["wall_seconds"] == 0.25
    assert summary["wall_budget_overshoot_seconds"] == 1.25
    assert final["wall_budget_overshoot_seconds"] == 1.25
    assert json.loads((output / "status.json").read_text())["status"] == "resource_stopped"
    metadata = json.loads((output / "metadata.json").read_text())
    assert "final-evaluation overshoot" in metadata["wall_stop_rule"]


def test_evaluation_reports_every_selected_task_and_only_known_visible_scores():
    torch.set_num_threads(1)
    cfg = replace(config(), tasks=TASKS)

    def protocol_env(task, **kwargs):
        # Exercise evaluator bookkeeping independently of individual game mechanics.
        return AtomicEnv("navigation", **kwargs)

    outcomes = iter([{"known": True, "score": 2.0, "success": True, "failure": False},
                     {"known": False, "score": None, "success": None, "failure": None}]
                    * len(TASKS))
    result = evaluate_sequence(factory(cfg.agent), TinyReward(), cfg,
                               env_factory=protocol_env, outcome_fn=lambda _: next(outcomes))
    for summary in result["summaries"].values():
        assert summary["episodes_started"] == len(TASKS)
        assert set(summary["tasks"]) == set(TASKS)
        assert summary["mean_visible_final_score"] == 2.0
        assert summary["visible_final_score_count"] == len(TASKS) // 2
        assert summary["unknown_outcomes"] == len(TASKS) // 2
        for index, task in enumerate(TASKS):
            task_summary = summary["tasks"][task]
            assert task_summary["episodes_started"] == 1
            assert task_summary["mean_visible_final_score"] == (2.0 if index % 2 == 0 else None)


def test_episode_minibatches_replay_full_histories_and_count_actual_optimizer_steps(monkeypatch):
    torch.set_num_threads(1)
    cfg = replace(config(), n_epochs=3, ppo_minibatch_episodes=1)
    agent, reward = factory(cfg.agent), TinyReward().eval().requires_grad_(False)
    episodes = [collect_episode(agent, reward, cfg, task=task, seed=20 + index)
                for index, task in enumerate(cfg.tasks)]
    replayed = []
    original = agent.evaluate_episode

    def replay(patches, actions):
        replayed.append((patches.data_ptr(), len(actions)))
        return original(patches, actions)

    monkeypatch.setattr(agent, "evaluate_episode", replay)
    optimizer = torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad], lr=1e-4)
    result = update_ppo(agent, optimizer, episodes, cfg)
    assert result["optimizer_steps"] == 6
    assert [row["optimizer_steps"] for row in result["epochs"]] == [2, 2, 2]
    assert all(row["replayed_transitions"] == sum(map(len, episodes))
               for row in result["epochs"])
    for episode in episodes:
        assert replayed.count((episode.patches.data_ptr(), len(episode))) == 3
    assert not agent.training


def test_task_balance_equalizes_short_and_long_task_value_gradients():
    class ConstantAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.zeros(()))

        def evaluate_episode(self, patches, actions):
            value = self.value.expand(len(actions))
            return {"value": value, "log_prob": value * 0, "entropy": value * 0}

    episodes = []
    for task, length, final_reward in (("navigation", 2, 1.), ("memory", 4, 4.)):
        rewards = torch.zeros(length)
        rewards[-1] = final_reward
        episodes.append(Episode(torch.zeros(length, 1, 384),
                                np.zeros((length, 96, 96, 3), dtype=np.uint8),
                                [PhysicalAction()] * length, torch.zeros(length),
                                torch.zeros(length), rewards,
                                [False] * (length - 1) + [True], [False] * length,
                                0., task, 0, {}))
    cfg = replace(config(), ppo_task_balance=True, value_coef=1, entropy_coef=0,
                  gamma=1, gae_lambda=1, max_grad_norm=1e6)
    agent = ConstantAgent()
    result = update_ppo(agent, torch.optim.SGD(agent.parameters(), lr=.1), episodes, cfg)
    assert agent.value.item() == pytest.approx(.25)
    assert result["optimizer_task_loss_weights"] == {"navigation": .5, "memory": .5}
    assert result["task_transitions"] == {"navigation": 2, "memory": 4}


def test_target_kl_rejects_update_and_reports_clipping():
    torch.set_num_threads(1)
    cfg = replace(config(), target_kl=.01)
    agent, reward = factory(cfg.agent), TinyReward().eval().requires_grad_(False)
    episode = collect_episode(agent, reward, cfg, task="navigation", seed=31)
    episode.old_log_prob -= 1
    before = tensor_hash(agent)
    optimizer = torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad], lr=1e-4)
    result = update_ppo(agent, optimizer, [episode], cfg)
    assert result["kl_early_stop"] and result["optimizer_steps"] == 0
    assert result["optimizer_task_loss_weights"] is None
    assert result["optimizer_task_transition_replays"] == {"navigation": 0}
    assert result["epochs"][0]["clip_fraction"] == 1
    assert result["epochs"][0]["approx_kl"] > cfg.target_kl
    assert tensor_hash(agent) == before and not optimizer.state


def test_stochastic_frozen_evaluation_preserves_rng_and_explicitly_covers_level_bank():
    import random

    torch.set_num_threads(1)
    cfg = replace(config(), eval_stochastic=True, eval_all_levels=True)
    agent, reward = factory(cfg.agent), TinyReward().eval().requires_grad_(False)
    agent.decoder.initialize_motor_prior("balanced")
    torch.manual_seed(99)
    random.seed(99)
    np.random.seed(99)
    cpu_state = torch.random.get_rng_state().clone()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    mps_state = torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None
    agent.train()
    before = tensor_hash(agent), tensor_hash(reward)
    first = evaluate_sequence(agent, reward, cfg)
    assert agent.training and before == (tensor_hash(agent), tensor_hash(reward))
    assert torch.equal(cpu_state, torch.random.get_rng_state())
    assert python_state == random.getstate()
    current_numpy = np.random.get_state()
    assert numpy_state[0] == current_numpy[0]
    assert np.array_equal(numpy_state[1], current_numpy[1])
    assert numpy_state[2:] == current_numpy[2:]
    if mps_state is not None:
        assert torch.equal(mps_state, torch.mps.get_rng_state())
    second = evaluate_sequence(agent, reward, cfg)
    assert first == second
    assert set(first["summaries"]) == {
        "frozen_deterministic", "frozen_stochastic", "generic_physical_random"}
    for policy, summary in first["summaries"].items():
        assert summary["episodes_started"] == 32
        for task in cfg.tasks:
            records = [row for row in first["records"]
                       if row["policy"] == policy and row["task"] == task]
            assert {row["level_index"] for row in records} == set(range(16))
            assert summary["tasks"][task]["distinct_level_ids"] == 16
            assert len({row["initial_frame_sha256"] for row in records}) > 1
            references = {row["initial_frame_sha256"] for row in first["records"]
                          if row["policy"] == "frozen_deterministic" and row["task"] == task}
            assert {row["initial_frame_sha256"] for row in records} == references


def test_initial_baseline_same_prior_minimum_all_tasks_and_optimizer_checkpoint_roundtrip(tmp_path):
    cfg = replace(config(), tasks=TASKS, evaluate_initial=True, motor_initialization="balanced",
                  ppo_minibatch_episodes=2, task_episode_multipliers={"navigation": 2},
                  curriculum_difficulty=.25)
    output = tmp_path / "balanced"
    result = train_sequence(cfg, output, agent_factory=factory, reward=TinyReward(),
                            reward_factory=TinyReward)
    baseline = json.loads((output / "baseline.json").read_text())
    assert baseline["rollout"] == baseline["decisions"] == baseline["weights_revision"] == 0
    manifest = json.loads((output / "checkpoints/rollout_0000/manifest.json").read_text())
    assert baseline["agent_sha256"] == manifest["agent_sha256"]
    assert result["optimizer_steps_this_process"] == result["weights_revision"] == 9
    history = json.loads((output / "training.jsonl").read_text())
    assert set(history["task_transitions"]) == set(TASKS)
    assert history["curriculum_difficulty"] == .25
    episodes = history["episode_records"]
    assert sum(row["task"] == "navigation" for row in episodes) == 2
    assert all(sum(row["task"] == task for row in episodes) >= 1 for task in TASKS)
    checkpoint = Path(result["latest_checkpoint"])
    loaded, _, loaded_reward, counters = load_sequence_checkpoint(
        checkpoint, replace(cfg, total_rollouts=2), agent_factory=factory,
        reward_factory=TinyReward)
    assert loaded.decoder.motor_initialization == "balanced"
    assert counters["weights_revision"] == 9
    assert tensor_hash(loaded_reward) == result["reward_sha256"]



def test_exact_frame_cache_is_byte_bounded_and_never_reuses_trainable_history():
    torch.set_num_threads(1)
    cfg = config()
    torch.manual_seed(117)
    agent, reward = factory(cfg.agent), TinyReward().eval().requires_grad_(False)
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    patch = agent.encode_frames(torch.as_tensor(frame)[None])[0]
    byte_limit = patch.numel() * patch.element_size() * 2
    cache = FrozenFrameCache(agent, capacity=100, max_bytes=byte_limit)
    assert torch.equal(cache.encode(frame), patch)
    assert torch.equal(cache.encode(frame), patch)
    assert cache.hits == 1 and cache.misses == 1
    cache.encode(np.full_like(frame, 1))
    cache.encode(np.full_like(frame, 2))
    assert cache.bytes <= byte_limit and len(cache.entries) == 2 and cache.evictions == 1
    rng = torch.random.get_rng_state()
    ordinary = collect_episode(agent, reward, cfg, task="memory", seed=21)
    torch.random.set_rng_state(rng)
    cached = collect_episode(agent, reward, cfg, task="memory", seed=21, feature_cache=cache)
    assert ordinary.actions == cached.actions
    assert torch.equal(ordinary.patches, cached.patches)
    assert torch.equal(ordinary.old_log_prob, cached.old_log_prob)
    assert torch.equal(ordinary.old_values, cached.old_values)
    before = agent.evaluate_episode(cached.patches, cached.actions)
    with torch.no_grad():
        agent.value_head.bias += 1
    after = agent.evaluate_episode(cached.patches, cached.actions)
    assert torch.allclose(after["value"], before["value"] + 1, atol=1e-6)
    cache.assert_unchanged()
    with torch.no_grad():
        agent.vision.projection.bias += 1
    with pytest.raises(RuntimeError, match="backbone changed"):
        cache.assert_unchanged()
    agent.vision.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen vision"):
        FrozenFrameCache(agent, 2)


def test_cached_frozen_evaluation_matches_uncached_policy_and_reports_hits():
    torch.set_num_threads(1)
    cfg = replace(config(), eval_stochastic=True)
    agent, reward = factory(cfg.agent), TinyReward().eval().requires_grad_(False)
    ordinary = evaluate_sequence(agent, reward, cfg)
    cached = evaluate_sequence(agent, reward, replace(cfg, vision_cache_frames=8))
    assert ordinary["records"] == cached["records"]
    assert ordinary["summaries"] == cached["summaries"]
    assert cached["vision_cache"]["hits"] > 0
    assert cached["vision_cache"]["bytes"] <= cached["vision_cache"]["capacity_bytes"]


def test_screen_reward_checkpoint_is_self_contained_after_external_path_disappears(tmp_path):
    from carryrl.rewards.visual import ScreenVisualReward

    cfg = replace(config(), reward_model_mode="screen_v2",
                  reward_checkpoint=str(tmp_path / "deleted_reward_checkpoint.pt"))
    output = tmp_path / "screen_v2"
    summary = train_sequence(cfg, output, agent_factory=factory,
                             reward=ScreenVisualReward(image_size=cfg.env_image_size))
    agent, _, reward, saved = load_sequence_checkpoint(
        Path(summary["latest_checkpoint"]), replace(cfg, total_rollouts=2), agent_factory=factory)
    assert isinstance(reward, ScreenVisualReward)
    assert not Path(cfg.reward_checkpoint).exists()
    assert tensor_hash(reward) == summary["reward_sha256"]
    assert saved["config"]["reward_model_mode"] == "screen_v2"
    assert saved["reward_model_version"] == reward.model_version
    assert agent.decoder.motor_initialization == "legacy"



def test_frozen_evaluation_uses_explicit_fixed_difficulty_and_honors_full_override():
    torch.set_num_threads(1)
    cfg = replace(config(), curriculum_difficulty=.15, curriculum_warmup_rollouts=1)
    calls = []

    def recorded_factory(task, **kwargs):
        calls.append(dict(kwargs))
        return AtomicEnv(task, **kwargs)

    agent, reward = factory(cfg.agent), TinyReward().eval().requires_grad_(False)
    easier = evaluate_sequence(agent, reward, cfg, env_factory=recorded_factory)
    assert calls and all(kwargs["difficulty"] == .15 for kwargs in calls)
    assert easier["evaluation_difficulty"] == .15
    assert all(row["difficulty"] == .15 for row in easier["records"])
    calls.clear()
    full = evaluate_sequence(agent, reward, replace(cfg, curriculum_difficulty=1),
                             env_factory=recorded_factory)
    assert calls and all("difficulty" not in kwargs for kwargs in calls)
    assert full["evaluation_difficulty"] == 1
    assert all(row["difficulty"] == 1 for row in full["records"])


@pytest.mark.parametrize("reject_last", [False, True])
def test_reported_optimizer_task_weights_reflect_oversampling_and_kl_stop(monkeypatch, reject_last):
    import carryrl.training.sequence as sequence

    class ConstantAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.critic = nn.Parameter(torch.zeros(()))

        def evaluate_episode(self, patches, actions):
            value = self.critic.expand(len(actions))
            return {"value": value, "log_prob": value * 0, "entropy": value * 0}

    episodes = []
    for task, length in (("navigation", 2), ("navigation", 3), ("memory", 4)):
        old_logp = torch.zeros(length)
        if reject_last and task == "memory":
            old_logp -= 1
        episodes.append(Episode(torch.zeros(length, 1, 384),
                                np.zeros((length, 96, 96, 3), dtype=np.uint8),
                                [PhysicalAction()] * length, old_logp, torch.zeros(length),
                                torch.zeros(length), [False] * (length - 1) + [True],
                                [False] * length, 0., task, 0, {}))
    monkeypatch.setattr(sequence, "_episode_minibatches", lambda *args: [[0], [1], [2]])
    cfg = replace(config(), ppo_task_balance=True, ppo_minibatch_episodes=1, target_kl=.01)
    agent = ConstantAgent()
    result = update_ppo(agent, torch.optim.SGD(agent.parameters(), lr=.1), episodes, cfg)
    assert result["optimizer_steps"] == (2 if reject_last else 3)
    assert result["optimizer_task_loss_weights"] == {
        "navigation": 1. if reject_last else pytest.approx(2 / 3),
        "memory": 0. if reject_last else pytest.approx(1 / 3)}
    assert result["optimizer_task_transition_replays"] == {
        "navigation": 5, "memory": 0 if reject_last else 4}
    assert result["task_transitions"] == {"navigation": 5, "memory": 4}
    assert "excludes rejected KL minibatches" in result["optimizer_task_loss_weights_scope"]


def test_warm_start_preserves_parent_tensors_moments_and_cumulative_counters(tmp_path):
    from carryrl.envs.presentation import PresentationConfig

    cfg = replace(config(), motor_initialization="balanced")
    parent = train_sequence(cfg, tmp_path / "parent", agent_factory=factory,
                            reward=TinyReward(), reward_factory=TinyReward)
    parent_checkpoint = Path(parent["latest_checkpoint"])
    changed = replace(cfg, seed=37, curriculum_difficulty=.15, evaluate_initial=True,
                      learning_rate=.0003, weight_decay=.03,
                      visual_config=PresentationConfig(mode="varied", native_sizes=(64, 96)))
    with pytest.raises(ValueError, match="resume must preserve"):
        load_sequence_checkpoint(parent_checkpoint, replace(changed, total_rollouts=2),
                                 agent_factory=factory, reward_factory=TinyReward)
    output = tmp_path / "next_stage"
    child = train_sequence(changed, output, warm_start=parent_checkpoint,
                           agent_factory=factory, reward_factory=TinyReward)
    initial_checkpoint = output / "checkpoints/rollout_0000"
    parent_state = torch.load(parent_checkpoint / "state.pt", weights_only=True)
    initial_state = torch.load(initial_checkpoint / "state.pt", weights_only=True)
    for section in ("agent_state", "reward_state"):
        assert all(torch.equal(value, initial_state[section][name])
                   for name, value in parent_state[section].items())
    parent_moments, child_moments = (state["optimizer_state"]["state"]
                                   for state in (parent_state, initial_state))
    assert parent_moments and set(parent_moments) == set(child_moments)
    for parameter, values in parent_moments.items():
        assert all(torch.equal(value, child_moments[parameter][name])
                   for name, value in values.items())
    assert initial_state["rollout"] == 0
    assert initial_state["decisions"] == parent["decisions"]
    assert initial_state["weights_revision"] == parent["weights_revision"]
    for group in initial_state["optimizer_state"]["param_groups"]:
        assert group["lr"] == changed.learning_rate and group["weight_decay"] == .03
    assert child["rollouts"] == 1 and child["initial_rollout"] == 0
    assert child["initial_decisions"] == parent["decisions"]
    assert child["initial_weights_revision"] == parent["weights_revision"]
    assert child["decisions"] == parent["decisions"] + child[
        "attempted_train_decisions_this_process"]
    assert child["weights_revision"] == parent["weights_revision"] + child[
        "optimizer_steps_this_process"]
    assert child["reward_sha256"] == parent["reward_sha256"]
    baseline = json.loads((output / "baseline.json").read_text())
    assert baseline["baseline_kind"] == "stage_initial"
    assert baseline["baseline_policy_previously_trained"] is True
    assert baseline["weights_revision"] == parent["weights_revision"]
    assert baseline["evaluation_difficulty"] == .15
    metadata = json.loads((output / "metadata.json").read_text())
    lineage = metadata["warm_start_parent"]
    assert metadata["resume"] is None and metadata["warm_start_rng_restored"] is False
    assert SequenceRunConfig.from_dict(lineage["source_config"]) == cfg
    assert lineage["source_counters"] == {
        "rollout": 1, "decisions": parent["decisions"],
        "weights_revision": parent["weights_revision"]}
    assert lineage["optimizer_state_preserved"] is True
    assert lineage["previous_optimizer_hyperparameters"] == [
        {"learning_rate": cfg.learning_rate, "weight_decay": cfg.weight_decay}]
    assert lineage["new_optimizer_hyperparameters"] == {
        "learning_rate": changed.learning_rate, "weight_decay": changed.weight_decay}


@pytest.mark.parametrize("change,message", [
    ({"motor_initialization": "balanced"}, "motor prior"),
    ({"reward_model_mode": "screen_v2"}, "reward model mode"),
    ({"reward_checkpoint": "new_external_reward.pt"}, "external reward checkpoint"),
    ({"agent": {"pretrained": False, "image_size": 28, "width": 64, "layers": 1,
                "heads": 4, "spatial_tokens": 4, "memory_tokens": 2,
                "segment_length": 2}}, "architecture"),
])
def test_warm_start_rejects_architecture_motor_and_reward_substitution(tmp_path, change, message):
    cfg = config()
    parent = train_sequence(cfg, tmp_path / "parent", agent_factory=factory,
                            reward=TinyReward(), reward_factory=TinyReward)
    changed = SequenceRunConfig.from_dict({**cfg.to_dict(), **change})
    with pytest.raises(ValueError, match=message):
        train_sequence(changed, tmp_path / "invalid_stage",
                       warm_start=Path(parent["latest_checkpoint"]),
                       agent_factory=factory, reward_factory=TinyReward)
    assert not (tmp_path / "invalid_stage/latest.json").exists()


def test_resume_warm_start_and_injected_reward_are_exclusive_before_output_creation(tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_sequence(config(), output, resume=tmp_path / "parent", warm_start=tmp_path / "parent")
    with pytest.raises(ValueError, match="injected reward"):
        train_sequence(config(), output, warm_start=tmp_path / "parent", reward=TinyReward())
    assert not output.exists()


def test_checkpoint_records_actual_reward_model_version_and_rejects_factory_mismatch(tmp_path):
    class VersionedReward(TinyReward):
        model_version = "test.reward.v3"

    cfg = config()
    parent = train_sequence(cfg, tmp_path / "versioned", agent_factory=factory,
                            reward=VersionedReward(), reward_factory=VersionedReward)
    checkpoint = Path(parent["latest_checkpoint"])
    archive = torch.load(checkpoint / "state.pt", weights_only=True)
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert archive["reward_model_version"] == manifest["reward_model_version"] == "test.reward.v3"
    load_sequence_checkpoint(checkpoint, replace(cfg, total_rollouts=2),
                             agent_factory=factory, reward_factory=VersionedReward)
    with pytest.raises(ValueError, match="reward model version"):
        load_sequence_checkpoint(checkpoint, replace(cfg, total_rollouts=2),
                                 agent_factory=factory, reward_factory=TinyReward)


def test_instruction_collection_replays_exact_prompt_and_bootstraps_same_history(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = instruction_config()
    agent, reward = instruction_factory(cfg.agent), TinyReward().eval().requires_grad_(False)
    episode = collect_episode(agent, reward, cfg, task="navigation", seed=21)
    prompt = cfg.episode_instructions["navigation"][1]
    assert episode.instruction == prompt
    assert prompt == "  Дойди до зелёного выхода.  "
    replay = agent.evaluate_episode(episode.patches, episode.actions, instruction=prompt)
    assert torch.allclose(replay["log_prob"].detach(), episode.old_log_prob, atol=2e-5)
    assert torch.allclose(replay["value"].detach(), episode.old_values, atol=2e-5)
    with torch.no_grad():
        final_patch = agent.encode_frames(torch.as_tensor(episode.final_frame)[None])[0].cpu()
        extended = agent.evaluate_episode(torch.cat([episode.patches, final_patch[None]]),
                                          [*episode.actions, PhysicalAction()], instruction=prompt)
        assert episode.bootstrap == pytest.approx(float(extended["value"][-1]), abs=2e-5)
    replay_prompts = []
    original = agent.evaluate_episode

    def replay_with_instruction(patches, actions, *, instruction=None):
        replay_prompts.append(instruction)
        return original(patches, actions, instruction=instruction)

    monkeypatch.setattr(agent, "evaluate_episode", replay_with_instruction)
    old_language = tensor_hash(agent.language)
    old_projection = agent.instruction_projection.weight.detach().clone()
    # The task label is bookkeeping: PPO sends the recorded prose, with no task argument.
    episode.task = "memory"
    optimizer = torch.optim.Adam([p for p in agent.parameters() if p.requires_grad], lr=.001)
    update_ppo(agent, optimizer, [episode], cfg)
    assert replay_prompts == [prompt]
    assert set(agent.language.calls) == {prompt}
    assert tensor_hash(agent.language) == old_language
    assert all(not parameter.requires_grad for parameter in agent.language.parameters())
    assert not torch.equal(old_projection, agent.instruction_projection.weight)


def test_instruction_evaluation_matches_text_across_frozen_and_random_policies(monkeypatch):
    torch.set_num_threads(1)
    cfg = instruction_config(eval_stochastic=True, eval_episodes_per_task=2)
    agent, reward = instruction_factory(cfg.agent), TinyReward().eval().requires_grad_(False)
    reset_prompts = []
    original = agent.reset_episode

    def reset(*, instruction=None):
        reset_prompts.append(instruction)
        original(instruction=instruction)

    monkeypatch.setattr(agent, "reset_episode", reset)
    before = tensor_hash(agent), tensor_hash(reward)
    result = evaluate_sequence(agent, reward, cfg)
    assert before == (tensor_hash(agent), tensor_hash(reward))
    assert len(result["records"]) == 12
    for task in cfg.tasks:
        for index in range(2):
            records = [row for row in result["records"]
                       if row["task"] == task and row["episode_index"] == index]
            assert len(records) == 3
            expected = cfg.episode_instructions[task][(1, 0)[index]]
            assert {row["instruction"] for row in records} == {expected}
            assert len({row["initial_frame_sha256"] for row in records}) == 1
    assert reset_prompts[:-1] == [row["instruction"] for row in result["records"]]
    assert reset_prompts[-1] is None


@pytest.mark.parametrize("evaluate", [False, True])
def test_instruction_encoding_failure_closes_environment(evaluate, monkeypatch):
    cfg = instruction_config()
    agent, reward = instruction_factory(cfg.agent), TinyReward().eval().requires_grad_(False)
    closed = []

    class TrackedEnv(AtomicEnv):
        def close(self):
            closed.append(True)
            super().close()

    original = agent.reset_episode

    def failed_encoding(*, instruction=None):
        if instruction is not None:
            raise ValueError("instruction exceeds token capacity")
        original()

    monkeypatch.setattr(agent, "reset_episode", failed_encoding)
    with pytest.raises(ValueError, match="token capacity"):
        if evaluate:
            evaluate_sequence(agent, reward, cfg, env_factory=TrackedEnv)
        else:
            collect_episode(agent, reward, cfg, task="navigation", seed=21,
                            env_factory=TrackedEnv)
    assert closed == [True]


def test_instruction_collection_rejects_missing_task_prompt_before_environment_creation():
    cfg = instruction_config()
    agent, reward = instruction_factory(cfg.agent), TinyReward().eval().requires_grad_(False)
    created = []

    def environment(*args, **kwargs):
        created.append(True)
        raise AssertionError("environment should not be constructed")

    with pytest.raises(ValueError, match="configured episode instruction"):
        collect_episode(agent, reward, cfg, task="mouse", seed=21, env_factory=environment)
    assert not created


@pytest.mark.parametrize("variant_count", [2, 5])
def test_instruction_seed_hash_covers_variants_despite_rollout_stride_without_rng_changes(
    variant_count,
):
    import random

    from carryrl.training.sequence import _episode_instruction

    variants = tuple(f"Follow instruction alternative {index}." for index in range(variant_count))
    cfg = replace(instruction_config(), tasks=("navigation",),
                  episode_instructions={"navigation": variants}, episodes_per_task=1)
    # The actual single-episode training schedule advances a task's seed by 10000 per rollout.
    seeds = [cfg.seed + rollout * 10000 for rollout in range(32)]
    torch_state = torch.random.get_rng_state().clone()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    selected = [_episode_instruction(cfg, "navigation", seed) for seed in seeds]
    restored = SequenceRunConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert selected == [_episode_instruction(restored, "navigation", seed) for seed in seeds]
    assert set(selected) == set(variants)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    assert np.array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]


def test_instruction_checkpoint_metadata_roundtrip_and_exact_episode_records(tmp_path):
    from carryrl.agent.language import TEXT_ID, TEXT_REVISION

    cfg = instruction_config()
    output = tmp_path / "text"
    result = train_sequence(cfg, output, agent_factory=instruction_factory, reward=TinyReward(),
                            reward_factory=TinyReward)
    assert SequenceRunConfig.load(output / "config.json") == cfg
    checkpoint = Path(result["latest_checkpoint"])
    archive = torch.load(checkpoint / "state.pt", weights_only=True)
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    for record in (archive, manifest, metadata):
        assert record["language_weight_id"] == TEXT_ID
        assert record["language_weight_revision"] == TEXT_REVISION
    assert SequenceRunConfig.from_dict(archive["config"]) == cfg
    assert metadata["instruction_selection"] == (
        "SHA256 of ASCII decimal episode seed, big-endian integer modulo prose variants")
    history = json.loads((output / "training.jsonl").read_text())
    trajectories = {json.loads(path.read_text())["seed"]: json.loads(path.read_text())
                    for path in (output / "trajectories/rollout_0001").glob("*.json")}
    for row in history["episode_records"]:
        expected = cfg.episode_instructions[row["task"]][{0: 1, 1000: 0}[row["seed"]]]
        assert row["instruction"] == trajectories[row["seed"]]["instruction"] == expected
    loaded, _, _, saved = load_sequence_checkpoint(
        checkpoint, replace(cfg, total_rollouts=2), agent_factory=instruction_factory,
        reward_factory=TinyReward)
    assert tensor_hash(loaded) == manifest["agent_sha256"]
    assert saved["language_weight_revision"] == TEXT_REVISION
    disabled = replace(config(), total_rollouts=2)
    with pytest.raises(ValueError, match="architecture"):
        load_sequence_checkpoint(checkpoint, disabled, agent_factory=factory,
                                 reward_factory=TinyReward, evaluation_only=True)
    manifest["language_weight_revision"] = "incorrect"
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="language model identity"):
        load_sequence_checkpoint(checkpoint, replace(cfg, total_rollouts=2),
                                 agent_factory=instruction_factory, reward_factory=TinyReward)


def test_old_checkpoint_without_instruction_fields_preserves_disabled_topology(tmp_path):
    import hashlib

    cfg = config()
    result = train_sequence(cfg, tmp_path / "old", agent_factory=factory,
                            reward=TinyReward(), reward_factory=TinyReward)
    checkpoint = Path(result["latest_checkpoint"])
    archive_path = checkpoint / "state.pt"
    archive = torch.load(archive_path, weights_only=True)
    archive["config"].pop("episode_instructions")
    archive["config"]["agent"].pop("text_encoder")
    archive["config"]["agent"].pop("instruction_max_tokens")
    torch.save(archive, archive_path)
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    loaded, _, _, _ = load_sequence_checkpoint(
        checkpoint, replace(cfg, total_rollouts=2), agent_factory=factory,
        reward_factory=TinyReward)
    assert loaded.config.text_encoder == "none"
    assert not hasattr(loaded, "instruction_projection")
    assert tensor_hash(loaded) == manifest["agent_sha256"]


def test_default_text_checkpoint_loader_uses_packaged_resources_then_strict_state(
    tmp_path, monkeypatch,
):
    import carryrl.training.sequence as sequence
    from carryrl.agent.language import MultilingualInstructionEncoder

    cfg = instruction_config()
    result = train_sequence(cfg, tmp_path / "source", agent_factory=instruction_factory,
                            reward=TinyReward(), reward_factory=TinyReward)
    checkpoint = Path(result["latest_checkpoint"])
    resources = checkpoint / "language"
    resources.mkdir()
    (resources / "config.json").write_text('{"hidden_size":384}')
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["language_resources_sha256"] = sequence._language_resource_hashes(resources)
    manifest_path.write_text(json.dumps(manifest))
    resource_calls = []

    def local_resources(path, *, max_tokens):
        resource_calls.append((path, max_tokens))
        language = TinyLanguage()
        with torch.no_grad():
            language.scale.fill_(99)
        return language

    def default_agent(agent_config, *, language=None):
        assert not agent_config.pretrained and language is not None
        return GeneralistAgent(agent_config, vision=TinyVision(), language=language)

    monkeypatch.setattr(MultilingualInstructionEncoder, "from_resources", local_resources)
    monkeypatch.setattr(sequence, "GeneralistAgent", default_agent)
    loaded, _, _, _ = load_sequence_checkpoint(
        checkpoint, replace(cfg, total_rollouts=2), agent_factory=default_agent,
        reward_factory=TinyReward)
    assert resource_calls == [(checkpoint / "language", 512)]
    assert loaded.language.scale.item() == 1
    (resources / "config.json").write_text('{"hidden_size":128}')
    with pytest.raises(ValueError, match="language resource SHA256"):
        load_sequence_checkpoint(checkpoint, replace(cfg, total_rollouts=2),
                                 agent_factory=default_agent, reward_factory=TinyReward)


@pytest.mark.parametrize("untrusted", ["unknown", "low_score_confidence"])
def test_untrusted_reward_frame_cannot_invent_progress_on_becoming_recognized(untrusted):
    import carryrl.training.sequence as sequence

    class IntermittentReward(TinyReward):
        def predict(self, frame):
            result = {"score": float(frame[0, 0, 0]), "success_probability": .5,
                      "failure_probability": .5}
            if not frame[0, 0, 0]:
                result.update({"known": False} if untrusted == "unknown"
                              else {"score_confidence": .55})
            else:
                result["score_confidence"] = .9
            return result

    reward = IntermittentReward()
    unknown, uncertain = sequence._prediction(reward, np.zeros((1, 1, 3), dtype=np.uint8), .6)
    known, _ = sequence._prediction(reward, np.ones((1, 1, 3), dtype=np.uint8), .6)
    assert uncertain
    assert sequence._transition(unknown, known) == 0
    assert sequence._transition(known, unknown) == 0
    assert sequence._transition(known, {**known, "score": 2}) == pytest.approx(.99)


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -.1, 1.1, None, "high"])
def test_reward_score_confidence_rejects_invalid_probability(confidence):
    import carryrl.training.sequence as sequence

    class InvalidReward(TinyReward):
        def predict(self, frame):
            return {**super().predict(frame), "score_confidence": confidence}

    with pytest.raises(ValueError, match="score confidence"):
        sequence._prediction(InvalidReward(), np.zeros((1, 1, 3), dtype=np.uint8), .6)


@pytest.mark.parametrize("value", [True, -.1, 1.1, float("nan"), float("inf"), "1"])
def test_evaluation_difficulty_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="evaluation_difficulty"):
        replace(config(), evaluation_difficulty=value)


def test_hard_evaluation_override_preserves_easy_training_collection():
    from carryrl.training.sequence import sequence_evaluation_protocol

    torch.set_num_threads(1)
    cfg = replace(config(), curriculum_difficulty=0, curriculum_warmup_rollouts=4,
                  evaluation_difficulty=1)
    calls = []

    def recorded_factory(task, **kwargs):
        calls.append(kwargs)
        return AtomicEnv(task, **kwargs)

    agent, reward = factory(cfg.agent), TinyReward().eval().requires_grad_(False)
    collect_episode(agent, reward, cfg, task="navigation", seed=cfg.seed,
                    difficulty=cfg.curriculum_difficulty, env_factory=recorded_factory)
    assert calls and all(row["difficulty"] == 0 for row in calls)
    calls.clear()
    result = evaluate_sequence(agent, reward, cfg, env_factory=recorded_factory)
    assert calls and all(row.get("difficulty", 1) == 1 for row in calls)
    assert result["evaluation_difficulty"] == 1
    assert all(row["difficulty"] == 1 for row in result["records"])
    assert sequence_evaluation_protocol(cfg)["difficulty"] == 1
    historical = replace(cfg, evaluation_difficulty=None)
    assert sequence_evaluation_protocol(historical)["difficulty"] == 0


def test_evaluate_cli_difficulty_overrides_configured_evaluation_difficulty(tmp_path, monkeypatch):
    import carryrl.training.sequence as sequence
    from carryrl.cli import main

    cfg = replace(config(), curriculum_difficulty=0, evaluation_difficulty=1)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    agent, reward = factory(cfg.agent), TinyReward()
    observed = []

    def load(checkpoint, run_config, **kwargs):
        observed.append(run_config)
        return agent, None, reward, {"weights_revision": 0}

    monkeypatch.setattr(sequence, "load_sequence_checkpoint", load)
    output = tmp_path / "evaluation"
    assert main(["evaluate-v1", "--config", str(path), "--checkpoint", str(tmp_path / "fake"),
                 "--output", str(output), "--difficulty", "0.25"]) == 0
    assert observed[0].evaluation_difficulty == .25
    assert observed[0].curriculum_difficulty == .25
    assert json.loads((output / "evaluation.json").read_text())["evaluation_difficulty"] == .25
