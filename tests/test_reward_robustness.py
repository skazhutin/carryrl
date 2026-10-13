"""Independent pixel labels, abstention, and reward-first quality gates."""

import copy

import numpy as np
import pytest
import torch

from carryrl.envs.presentation import (
    HUD_THEMES,
    PresentationConfig,
    PresentationProfile,
    present_frame,
    render_hud,
)
from carryrl.rewards.pretrain import (
    GATED_PARTITIONS,
    REQUIRED_COMBINATIONS,
    RewardGate,
    _synthetic_glyph_dataset,
    check_reward_gate,
    load_screen_reward_checkpoint,
    pretrain_screen_reward,
)
from carryrl.rewards.visual import (
    LearnedVisualReward,
    ScreenVisualReward,
    draw_visible_hud,
    locate_visible_hud,
    screen_reward_readout,
    train_visual_reward,
    verify_visible_outcome,
)


@pytest.mark.parametrize("size", (64, 96, 128))
@pytest.mark.parametrize("layout", ("top", "bottom", "left", "right"))
def test_independent_teacher_reads_resolutions_layouts_and_themes(size, layout):
    frame = np.zeros((size, size, 3), np.uint8)
    for theme in HUD_THEMES:
        for score, status in ((0, "Play"), (2, "Success"), (5, "Failed")):
            draw_visible_hud(frame, score, status)
            styled = present_frame(frame, PresentationProfile(size, theme, layout))
            assert verify_visible_outcome(styled) == {
                "score": float(score), "success": status == "Success",
                "failure": status == "Failed", "known": True,
            }
            readout = screen_reward_readout(styled)
            assert readout.shape == (32, 64, 3) and readout.dtype == np.uint8


def test_missing_ambiguous_and_corrupted_screens_are_explicit():
    background = np.full((160, 220, 3), (11, 15, 23), np.uint8)
    model = ScreenVisualReward()
    unknown = model.predict(background)
    assert unknown == {"score": 0.0, "success_probability": 0.5,
                       "failure_probability": 0.5, "known": False}
    assert model.transition(background, background) == 0
    panel = render_hud("sand", "top", 2, "Success")
    background[4:32, 4:100] = panel
    assert verify_visible_outcome(background)["known"]
    background[60:88, 4:100] = panel
    assert locate_visible_hud(background) is None
    assert not verify_visible_outcome(background)["known"]
    assert not model.predict(background)["known"]
    background[60:88, 4:100] = (11, 15, 23)
    background[10, 10] = (1, 2, 3)
    assert not verify_visible_outcome(background)["known"]


def test_screen_preprocessing_does_not_read_scene_or_use_score_templates():
    frame = np.zeros((96, 96, 3), np.uint8)
    draw_visible_hud(frame, 2, "Success")
    profile = PresentationProfile(96, "violet", "right")
    styled = present_frame(frame, profile)
    readout = screen_reward_readout(styled)
    changed = styled.copy()
    x0, y0, x1, y1 = profile.viewport_box
    changed[y0:y1, x0:x1] = 255 - changed[y0:y1, x0:x1]
    np.testing.assert_array_equal(screen_reward_readout(changed), readout)
    # A damaged glyph stays a pixel input to the learned recognizer; unlike the
    # independent exact verifier, preprocessing does not recognize its label.
    hx, hy, _, _ = profile.hud_box
    changed[hy + 6, hx + 6] = (255, 255, 255)
    assert not verify_visible_outcome(changed)["known"]
    assert screen_reward_readout(changed) is not None


def _passing_metrics():
    return {"frames": 18, "score_mae": 0.01, "max_score_error": 0.02,
            "max_spurious_reward_change": 0.01,
            "combination_counts": dict.fromkeys(REQUIRED_COMBINATIONS, 1),
            **{event: {"positive_frames": 6, "negative_frames": 12, "recall": 1.0,
                       "false_positive_rate": 0.0, "brier_score": 0.001}
               for event in ("success", "failure")}}


def test_reward_gate_rejects_wrong_scores_missing_positives_and_missing_profiles():
    partitions = {}
    for name in GATED_PARTITIONS:
        partitions[name] = {**_passing_metrics(), "by_profile": {"profile": _passing_metrics()}}
    expected = {name: ["profile"] for name in partitions}
    assert check_reward_gate(partitions, expected_profiles=expected)["passed"]
    wrong = copy.deepcopy(partitions)
    wrong["unseen_validation"]["by_profile"]["profile"]["score_mae"] = 2.0
    assert not check_reward_gate(wrong)["passed"]
    missing = copy.deepcopy(partitions)
    missing["unseen_validation"]["by_profile"]["profile"]["failure"]["positive_frames"] = 0
    assert not check_reward_gate(missing)["passed"]
    missing_combination = copy.deepcopy(partitions)
    missing_combination["familiar_validation"]["by_profile"]["profile"][
        "combination_counts"
    ]["3:Success"] = 0
    assert not check_reward_gate(missing_combination)["passed"]
    expected["unseen_validation"].append("missing_style")
    assert not check_reward_gate(partitions, expected_profiles=expected)["passed"]
    assert not check_reward_gate({})["passed"]
    with pytest.raises(ValueError):
        RewardGate(min_positive_frames=0)


def test_screen_version_is_explicit_and_legacy_parameters_stay_loadable(tmp_path):
    original = LearnedVisualReward(64)
    loaded = LearnedVisualReward(64)
    loaded.load_state_dict(original.state_dict(), strict=True)
    frame = np.zeros((64, 64, 3), np.uint8)
    draw_visible_hud(frame, 3, "Play")
    assert original.predict(frame) == loaded.predict(frame)
    assert ScreenVisualReward.model_version == "carryrl.screen_reward.v3"
    with pytest.raises(ValueError):
        ScreenVisualReward()(torch.zeros(1, 3, 96, 96))
    torch.save({"version": "legacy"}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="version"):
        load_screen_reward_checkpoint(tmp_path / "bad.pt")


def test_reward_first_splits_episodes_preserves_failed_gate_and_freezes(tmp_path, monkeypatch):
    import carryrl.envs.atomic as atomic

    calls = []

    def collect(*, image_size, split, **kwargs):
        calls.append(split)
        frames, ids, episodes = [], [], []
        labels = {name: [] for name in ("score", "success", "failure")}
        for episode, status in enumerate(("Success", "Failed")):
            episode_id = f"{split}:reaction:{episode}"
            episodes.append({"episode_id": episode_id, "task": "reaction"})
            for score, state in ((0, "Play"), (2, status)):
                frame = np.zeros((image_size, image_size, 3), np.uint8)
                draw_visible_hud(frame, score, state)
                observed = verify_visible_outcome(frame)
                frames.append(frame)
                ids.append(episode_id)
                for name in labels:
                    labels[name].append(float(observed[name]))
        return np.stack(frames), {name: np.array(value, np.float32)
                                for name, value in labels.items()}, {
            "frame_episode_ids": ids, "episodes": episodes,
        }

    monkeypatch.setattr(atomic, "collect_bootstrap_dataset", collect)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model, report = pretrain_screen_reward(
            tmp_path / "reward", tasks=("reaction",), epochs=1,
            presentation_config=PresentationConfig(
                mode="varied", native_sizes=(96,), themes=("slate", "violet"),
                hud_layouts=("top",),
            ),
        )
    finally:
        torch.set_num_threads(threads)
    assert calls == ["train", "validation"]
    assert not set(report["train_episode_ids"]) & set(report["validation_episode_ids"])
    assert report["test_levels_used"] is False
    assert report["gate"]["passed"] is False
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    checkpoint = tmp_path / "reward" / "reward_checkpoint.pt"
    with pytest.raises(ValueError, match="validation gate"):
        load_screen_reward_checkpoint(checkpoint)
    loaded, _ = load_screen_reward_checkpoint(checkpoint, require_pass=False)
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name])
    # A stored success boolean cannot bypass recomputed failing metrics.
    data = torch.load(checkpoint, weights_only=True)
    data["report"]["gate"]["passed"] = True
    torch.save(data, tmp_path / "forged.pt")
    with pytest.raises(ValueError, match="validation gate"):
        load_screen_reward_checkpoint(tmp_path / "forged.pt")
    with pytest.raises(FileExistsError):
        pretrain_screen_reward(tmp_path / "reward")


def test_score_and_status_have_disjoint_pixel_support():
    torch.manual_seed(71)
    model = ScreenVisualReward().eval()
    original = torch.rand(1, 3, 32, 64)
    status_changed = original.clone()
    status_changed[:, :, 16:] = torch.rand(1, 3, 16, 64)
    score_changed = original.clone()
    score_changed[:, :, :16] = torch.rand(1, 3, 16, 64)
    before, status, score = model(original), model(status_changed), model(score_changed)
    torch.testing.assert_close(before["score_logits"], status["score_logits"], rtol=0, atol=0)
    for name in ("success_logits", "failure_logits"):
        torch.testing.assert_close(before[name], score[name], rtol=0, atol=0)


def test_trained_score_does_not_follow_terminal_status_for_three_and_four():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        config = PresentationConfig(mode="varied", native_sizes=(96,))
        frames, labels, _, _, _ = _synthetic_glyph_dataset(
            config, ("slate", "forest"), canonical_sizes=(96,), seed=1,
        )
        torch.manual_seed(5)
        model = ScreenVisualReward()
        train_visual_reward(model, frames, labels, epochs=20, seed=5,
                            min_samples_per_epoch=1024)
        for theme in ("canonical", "violet", "ice"):
            for score in range(6):
                for status in ("Play", "Success", "Failed"):
                    frame = np.zeros((96, 96, 3), np.uint8)
                    draw_visible_hud(frame, score, status)
                    if theme != "canonical":
                        frame = present_frame(frame, PresentationProfile(96, theme, "right"))
                    observed = verify_visible_outcome(frame)
                    prediction = model.predict(frame)
                    assert prediction["score"] == observed["score"]
                    assert (prediction["success_probability"] >= 0.5) == observed["success"]
                    assert (prediction["failure_probability"] >= 0.5) == observed["failure"]
    finally:
        torch.set_num_threads(threads)
