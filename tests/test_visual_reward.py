"""Visible supervision and learned, frozen reward bootstrap behavior."""

import numpy as np
import pytest
import torch

from carryrl.envs.atomic import collect_bootstrap_dataset
from carryrl.rewards.visual import (
    HUD_HEIGHT,
    LearnedVisualReward,
    draw_visible_hud,
    evaluate_visual_reward,
    train_visual_reward,
    verify_visible_outcome,
)


def test_pixel_outcomes_are_explicit_and_corrupt_frames_remain_unknown():
    frame = np.zeros((96, 96, 3), np.uint8)
    for status, success, failure in (
        ("Play", False, False),
        ("Success", True, False),
        ("Failed", False, True),
    ):
        draw_visible_hud(frame, 2, status)
        assert verify_visible_outcome(frame) == {
            "score": 2.0,
            "success": success,
            "failure": failure,
            "known": True,
        }
    frame[4, 5] = (1, 2, 3)
    unknown = verify_visible_outcome(frame)
    assert not unknown["known"]
    assert all(unknown[name] is None for name in ("score", "success", "failure"))
    for invalid in (frame.astype(np.float32), frame[:, :, 0], frame[:20]):
        assert not verify_visible_outcome(invalid)["known"]


def test_visual_reward_learns_rgb_labels_freezes_and_validates_heldout_episodes():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        frames, labels, _ = collect_bootstrap_dataset(episodes_per_task=4)
        validation, validation_labels, _ = collect_bootstrap_dataset(
            episodes_per_task=2, split="validation", seed=100
        )
        torch.manual_seed(5)
        model = LearnedVisualReward()
        before = evaluate_visual_reward(model, frames, labels)
        result = train_visual_reward(model, frames, labels, epochs=40, seed=5)
        after = evaluate_visual_reward(model, validation, validation_labels)
        assert result["final_epoch_loss"] < result["first_epoch_loss"]
        assert result["train_metrics"]["score_mse"] < before["score_mse"]
        assert after["score_mae"] < 0.8
        assert after["success_accuracy"] > 0.95 and after["failure_accuracy"] > 0.95
        assert after["success_recall"] > 0.9 and after["failure_recall"] > 0.9
        assert after["success_false_positive_rate"] == 0
        assert after["failure_false_positive_rate"] == 0
        assert after["success_brier_score"] < 0.02 and after["failure_brier_score"] < 0.02
        assert after["success_positive_probability_mean"] > 0.9
        assert after["failure_positive_probability_mean"] > 0.9
        for metrics in (result["train_metrics"], after):
            assert max(item["score_mae"] for item in metrics["score_mae_by_combination"]) < 0.8
        # Successful choices with fewer points must also be read numerically.
        assert any(
            item["success"] and 0 < item["score"] < 5
            for item in result["train_metrics"]["score_mae_by_combination"]
        )
        assert not model.training and all(
            not parameter.requires_grad for parameter in model.parameters()
        )
        frame = validation[0]
        prediction = model.predict(frame)
        altered = frame.copy()
        altered[HUD_HEIGHT:] = 255 - altered[HUD_HEIGHT:]
        assert model.predict(altered) == prediction  # Its disclosed scope is RGB HUD recognition.
        assert set(prediction) == {"score", "success_probability", "failure_probability"}
        assert np.isfinite(model.transition(frame, validation[-1]))
    finally:
        torch.set_num_threads(previous_threads)


def test_visual_reward_rejects_invalid_shapes_and_nonfinite_inputs():
    model = LearnedVisualReward()
    for invalid in (
        torch.zeros(1, 96, 96, 3),
        torch.zeros(1, 3, 64, 64),
        torch.full((1, 3, 96, 96), float("nan")),
        torch.full((1, 3, 96, 96), 255.0),
    ):
        with pytest.raises(ValueError):
            model(invalid)
    with pytest.raises(ValueError):
        model.predict(np.zeros((96, 96, 3), np.float32))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS unavailable")
@pytest.mark.parametrize("image_size", (64, 96))
def test_visual_reward_divisible_pooling_forward_backward_on_mps(image_size):
    torch.manual_seed(4)
    cpu = LearnedVisualReward(image_size=image_size)
    mps = LearnedVisualReward(image_size=image_size).to("mps")
    mps.load_state_dict(cpu.state_dict())
    frames = torch.rand(2, 3, image_size, image_size)
    expected, actual = cpu(frames), mps(frames.to("mps"))
    sum(value.sum() for value in actual.values()).backward()
    torch.mps.synchronize()
    for name in expected:
        torch.testing.assert_close(actual[name].cpu(), expected[name], atol=1e-4, rtol=1e-4)
    assert all(torch.isfinite(parameter.grad).all() for parameter in mps.parameters())
