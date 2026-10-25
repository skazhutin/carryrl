"""Contracts for an unfitted causal reward candidate; no real training runs."""

import numpy as np
import pytest
import torch

from carryrl.agent.actions import ACTION_FEATURE_SIZE
from carryrl.envs.presentation import (
    HUD_LAYOUTS,
    HUD_THEMES,
    PresentationProfile,
    present_frame,
    render_hud,
)
from carryrl.rewards.history import (
    HUD_MASK_COLOR,
    SceneHistoryHead,
    history_reward_loss,
    mask_reward_hud,
)
from carryrl.rewards.visual import LearnedVisualReward, draw_visible_hud, locate_visible_hud


@pytest.mark.parametrize("size", (64, 96, 128))
@pytest.mark.parametrize("layout", HUD_LAYOUTS)
def test_mask_erases_all_scores_statuses_preserves_non_hud(size, layout):
    rng = np.random.default_rng(41)
    original = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    for theme in HUD_THEMES:
        baseline = None
        for score in range(6):
            for status in ("Play", "Success", "Failed"):
                frame = original.copy()
                draw_visible_hud(frame, score, status)
                frame = present_frame(frame, PresentationProfile(size, theme, layout))
                before = frame.copy()
                y, x, height, width, _ = locate_visible_hud(frame)
                masked = mask_reward_hud(frame)
                np.testing.assert_array_equal(frame, before)
                assert not np.shares_memory(frame, masked)
                panel = np.zeros(frame.shape[:2], dtype=bool)
                panel[y:y + height, x:x + width] = True
                np.testing.assert_array_equal(masked[~panel], frame[~panel])
                assert np.all(masked[panel] == HUD_MASK_COLOR)
                if baseline is None:
                    baseline = masked
                else:
                    np.testing.assert_array_equal(masked, baseline)


@pytest.mark.parametrize("size", (64, 96, 128))
def test_canonical_mask_invariant_and_unknown_abstains(size):
    frame = np.full((size, size, 3), 70, np.uint8)
    baseline = None
    for score, status in ((0, "Play"), (5, "Success"), (3, "Failed")):
        draw_visible_hud(frame, score, status)
        masked = mask_reward_hud(frame)
        assert masked is not None
        if baseline is None:
            baseline = masked
        else:
            np.testing.assert_array_equal(masked, baseline)
    assert mask_reward_hud(np.zeros((size, size, 3), np.uint8)) is None
    assert mask_reward_hud(frame.astype(np.float32)) is None
    assert mask_reward_hud(np.zeros((size, size), np.uint8)) is None


def test_two_visible_panels_abstain_and_mask_never_reads_outcomes(monkeypatch):
    import carryrl.rewards.visual as visual

    def forbidden(*args, **kwargs):
        raise AssertionError("Must not invoke outcome recognition")

    monkeypatch.setattr(visual, "verify_visible_outcome", forbidden)
    monkeypatch.setattr(visual, "_verify_presented_huds", forbidden)
    frame = np.full((160, 220, 3), 71, np.uint8)
    frame[4:32, 4:100] = render_hud("sand", "top", 5, "Success")
    assert mask_reward_hud(frame) is not None
    frame[60:88, 4:100] = render_hud("sand", "top", 0, "Play")
    assert mask_reward_hud(frame) is None


def _inputs(batch=2, steps=5):
    torch.manual_seed(10)
    scenes = torch.randn(batch, steps, 7)
    actions = torch.zeros(batch, steps, ACTION_FEATURE_SIZE)
    goals = torch.randn(batch, 4)
    return scenes, actions, goals


def _head():
    torch.manual_seed(17)
    return SceneHistoryHead(scene_feature_size=7, goal_feature_size=4, hidden_size=9)


def test_causal_outputs_and_gradients_do_not_read_future():
    head = _head()
    scenes, actions, goals = _inputs()
    scenes.requires_grad_()
    full = head(scenes, actions, goals)
    changed_scene, changed_actions = scenes.detach().clone(), actions.clone()
    changed_scene[:, 3:] += 100
    changed_actions[:, 3:, 0] = 1
    changed = head(changed_scene, changed_actions, goals)
    prefix = head(scenes[:, :3], actions[:, :3], goals)
    for name, value in full.items():
        torch.testing.assert_close(value[:, :3], changed[name][:, :3])
        torch.testing.assert_close(value[:, :3], prefix[name])
    full["potential"][:, :3].sum().backward()
    assert torch.count_nonzero(scenes.grad[:, :3]) > 0
    assert torch.count_nonzero(scenes.grad[:, 3:]) == 0


def test_own_actions_and_explicit_goals_affect_head_and_padding_does_not():
    head = _head()
    scenes, actions, goals = _inputs()
    original = head(scenes, actions, goals)
    actions[:, 2, 0] = 1
    changed_actions = head(scenes, actions, goals)
    changed_goals = head(scenes, torch.zeros_like(actions), -goals)
    assert not torch.allclose(original["event_logits"][:, 2:],
                              changed_actions["event_logits"][:, 2:])
    assert not torch.allclose(original["potential"], changed_goals["potential"])
    short = head(scenes[:1, :2], actions[:1, :2], goals[:1])
    padded = head(scenes, actions, goals, lengths=torch.tensor([2, 5]))
    for name in short:
        torch.testing.assert_close(short[name][0], padded[name][0, :2])
        assert torch.count_nonzero(padded[name][0, 2:]) == 0
    assert original["event_logits"].shape == (2, 5, 6)
    assert torch.all((original["potential"] > 0) & (original["potential"] < 1))
    assert not hasattr(head, "predict") and not hasattr(head, "mark_fitted")


def test_command_budget_includes_initial_observation():
    outputs = _head()(*_inputs(batch=1, steps=129))
    assert outputs["potential"].shape == (1, 129)
    assert torch.isfinite(history_reward_loss(outputs, _targets(batch=1, steps=129)))


@pytest.mark.parametrize("bad", ("initial", "binary", "range", "nan", "zero_goal",
                                  "bad_lengths", "too_long", "no_steps"))
def test_head_rejects_misalignment_and_invalid_input(bad):
    scenes, actions, goals = _inputs()
    kwargs = {}
    if bad == "initial":
        actions[:, 0, 0] = 1
    elif bad == "binary":
        actions[:, 1, 0] = 0.5
    elif bad == "range":
        actions[:, 1, -1] = 2
    elif bad == "nan":
        scenes[0, 1, 0] = float("nan")
    elif bad == "zero_goal":
        goals[0] = 0
    elif bad == "bad_lengths":
        kwargs["lengths"] = torch.tensor([0, 5])
    else:
        scenes, actions, goals = _inputs(steps=130 if bad == "too_long" else 0)
    with pytest.raises(ValueError):
        _head()(scenes, actions, goals, **kwargs)


def _targets(batch=2, steps=5):
    return {
        "potential": torch.full((batch, steps), 0.7),
        "potential_valid": torch.ones(batch, steps, dtype=torch.bool),
        "events": torch.ones(batch, steps, 6),
        "event_valid": torch.ones(batch, steps, 6, dtype=torch.bool),
        "terminal_success": torch.ones(batch, steps),
        "terminal_success_valid": torch.ones(batch, steps, dtype=torch.bool),
    }


def test_unknown_nan_targets_do_not_supply_main_head_gradients_but_train_validity():
    outputs = {name: value.detach().requires_grad_() for name, value in _head()(*_inputs()).items()}
    targets = _targets()
    for name, valid in (("potential", "potential_valid"), ("events", "event_valid"),
                        ("terminal_success", "terminal_success_valid")):
        targets[name][:, 1] = float("nan")
        targets[valid][:, 1] = False
        targets[name].requires_grad_()
    loss = history_reward_loss(outputs, targets, lengths=torch.tensor([3, 5]))
    assert torch.isfinite(loss)
    loss.backward()
    for name in ("potential", "event_logits", "terminal_success_logits"):
        assert torch.count_nonzero(outputs[name].grad[:, 1]) == 0
        assert torch.count_nonzero(outputs[name].grad[0, 3:]) == 0
        assert torch.count_nonzero(outputs[name].grad[:, 0]) > 0
    for name in ("potential_valid_logits", "event_valid_logits", "terminal_valid_logits"):
        assert (outputs[name].grad[:, 1] > 0).all()  # Unknown -> validity false.
        assert (outputs[name].grad[:, 0] < 0).all()  # Observed -> validity true.
        assert torch.count_nonzero(outputs[name].grad[0, 3:]) == 0
    for name in ("potential", "events", "terminal_success"):
        assert torch.count_nonzero(targets[name].grad[:, 1]) == 0
        assert torch.count_nonzero(targets[name].grad[0, 3:]) == 0


def test_loss_is_mean_of_episode_losses_not_frame_weighted():
    outputs = _head()(*_inputs())
    targets = _targets()
    lengths = torch.tensor([2, 5])
    combined = history_reward_loss(outputs, targets, lengths)
    separate = []
    for episode, length in enumerate(lengths.tolist()):
        episode_outputs = {name: value[episode:episode + 1, :length]
                           for name, value in outputs.items()}
        episode_targets = {name: value[episode:episode + 1, :length]
                           for name, value in targets.items()}
        separate.append(history_reward_loss(episode_outputs, episode_targets))
    torch.testing.assert_close(combined, torch.stack(separate).mean())


def test_loss_rejects_invalid_observed_targets_and_masks():
    outputs = _head()(*_inputs())
    targets = _targets()
    targets["potential"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Observed potential"):
        history_reward_loss(outputs, targets)
    targets = _targets()
    targets["event_valid"] = targets["event_valid"].float()
    with pytest.raises(ValueError, match="boolean"):
        history_reward_loss(outputs, targets)


def test_small_synthetic_optimizer_step_and_legacy_model_stay_independent():
    head = _head()
    legacy = LearnedVisualReward(64)
    legacy_before = {name: value.clone() for name, value in legacy.state_dict().items()}
    before = head.event_head.weight.detach().clone()
    loss = history_reward_loss(head(*_inputs()), _targets())
    optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert not torch.equal(head.event_head.weight, before)
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(value, legacy_before[name])
