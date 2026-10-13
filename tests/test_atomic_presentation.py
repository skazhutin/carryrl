"""Appearance-independent mechanics and reachable curriculum stages."""

from dataclasses import replace

import numpy as np
import pytest

from carryrl.agent.actions import MOUSE_BINS, PhysicalAction
from carryrl.envs.atomic import PLAYER, TASKS, AtomicEnv, atomic_level_banks
from carryrl.envs.presentation import (
    HUD_LAYOUTS,
    HUD_THEMES,
    THEME_SPLITS,
    PresentationConfig,
    PresentationProfile,
    choose_presentation,
    present_frame,
    render_hud,
)
from carryrl.rewards.visual import verify_visible_outcome


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("native_size", (64, 96, 128))
def test_all_mechanics_present_episode_stable_rectangular_rgb(task, native_size):
    config = PresentationConfig(mode="varied", native_sizes=(native_size,))
    env = AtomicEnv(task, visual_config=config, max_steps=1)
    first, info = env.reset(seed=8)
    profile = env.presentation
    assert first.shape != (native_size, native_size, 3)
    assert env.observation_space.contains(first)
    assert info["known"] and info["score"] == 0
    repeated, repeated_info = env.reset(seed=8)
    np.testing.assert_array_equal(first, repeated)
    assert info == repeated_info
    result, _, terminated, truncated, _ = env.step(PhysicalAction())
    assert env.presentation == profile
    assert result.shape == first.shape and not terminated and truncated
    assert verify_visible_outcome(result)["known"]


@pytest.mark.parametrize("layout", HUD_LAYOUTS)
@pytest.mark.parametrize("theme", tuple(HUD_THEMES))
def test_saved_rgb_transformation_retains_pixel_only_outcome(theme, layout):
    env = AtomicEnv("mouse")
    canonical, info = env.reset(seed=4)
    profile = PresentationProfile(96, theme, layout, texture_seed=97)
    presented = present_frame(canonical, profile)
    assert verify_visible_outcome(presented) == {
        key: info[key] for key in ("score", "success", "failure", "known")
    }
    x0, y0, x1, y1 = profile.hud_box
    np.testing.assert_array_equal(presented[y0:y1, x0:x1], render_hud(theme, layout, 0, "Play"))
    bad = canonical.copy()
    bad[:28] = 0
    with pytest.raises(ValueError, match="unknown"):
        present_frame(bad, profile)


@pytest.mark.parametrize("task", TASKS)
def test_styles_do_not_change_level_selection_or_physical_mechanics(task):
    config = PresentationConfig(mode="varied", native_sizes=(96,), themes=("slate",),
                                hud_layouts=("left",))
    other = replace(config, themes=("sand",), hud_layouts=("bottom",), seed=81)
    first, second = AtomicEnv(task, visual_config=config), AtomicEnv(task, visual_config=other)
    first_frame, first_info = first.reset(seed=19)
    second_frame, second_info = second.reset(seed=19)
    assert first_info["level_id"] == second_info["level_id"]
    assert first_info["effective_level_id"] == second_info["effective_level_id"]
    assert first_frame.shape != second_frame.shape
    actions = (PhysicalAction(keys=("w",)), PhysicalAction(keys=("d",), dx=2, dy=-1),
               PhysicalAction(keys=("e",)), PhysicalAction(mouse_buttons=("left",)))
    for action in actions:
        left = first.step(action)
        right = second.step(action)
        assert left[1:4] == right[1:4]
        assert [left[4][key] for key in ("score", "success", "failure")] == [
            right[4][key] for key in ("score", "success", "failure")
        ]
        if left[2] or left[3]:
            break


@pytest.mark.parametrize("layout", HUD_LAYOUTS)
def test_mouse_deltas_stay_image_pixels_when_viewport_is_translated(layout):
    config = PresentationConfig(mode="varied", native_sizes=(96,), themes=("slate",),
                                hud_layouts=(layout,), textures=False)
    canonical, varied = AtomicEnv("mouse"), AtomicEnv("mouse", visual_config=config)
    original, _ = canonical.reset(seed=4)
    placed, _ = varied.reset(seed=4)
    offset_x, offset_y, *_ = varied.presentation.viewport_box
    original_y, original_x = np.where(np.all(original == PLAYER, axis=-1))
    placed_y, placed_x = np.where(np.all(placed == PLAYER, axis=-1))
    assert placed_x.mean() == pytest.approx(original_x.mean() + offset_x)
    assert placed_y.mean() == pytest.approx(original_y.mean() + offset_y)
    original, *_ = canonical.step(PhysicalAction(dx=2, dy=-1))
    placed, *_ = varied.step(PhysicalAction(dx=2, dy=-1))
    original_y, original_x = np.where(np.all(original == PLAYER, axis=-1))
    next_y, next_x = np.where(np.all(placed == PLAYER, axis=-1))
    assert next_x.mean() == pytest.approx(placed_x.mean() + 2)
    assert next_y.mean() == pytest.approx(placed_y.mean() - 1)
    assert next_x.mean() == pytest.approx(original_x.mean() + offset_x)
    assert next_y.mean() == pytest.approx(original_y.mean() + offset_y)


def test_holdout_appearance_and_native_size_sampling_do_not_use_task_identity():
    config = PresentationConfig(mode="varied")
    train = {choose_presentation(config, "train", seed).theme for seed in range(32)}
    validation = {choose_presentation(config, "validation", seed).theme for seed in range(32)}
    test = {choose_presentation(config, "test", seed).theme for seed in range(32)}
    assert train == set(THEME_SPLITS["train"])
    assert validation == set(THEME_SPLITS["validation"])
    assert test == set(THEME_SPLITS["test"])
    assert train.isdisjoint(validation | test)
    assert {choose_presentation(config, "train", seed).native_size for seed in range(32)} == {
        64, 96, 128
    }
    first, second = AtomicEnv("navigation", visual_config=config), AtomicEnv("shooting",
                                                                            visual_config=config)
    first.reset(seed=21)
    second.reset(seed=21)
    assert first.presentation == second.presentation


@pytest.mark.parametrize("task", TASKS)
def test_enumerated_validation_levels_have_full_stable_provenance(task):
    env = AtomicEnv(task, split="validation", difficulty=0.0)
    level_ids = []
    for index in range(len(atomic_level_banks(task)["validation"])):
        _, info = env.reset(seed=7, options={"level_index": index})
        assert info["level_index"] == index and info["difficulty"] == 0
        level_ids.append(info["level_id"])
    assert len(set(level_ids)) == 16
    for bad in (True, -1, 16, 1.5):
        with pytest.raises(ValueError, match="level_index"):
            env.reset(options={"level_index": bad})
    with pytest.raises(ValueError, match="level_index"):
        env.reset(options={"task_id": 0})


def test_initial_navigation_and_mouse_stages_require_real_actions_but_are_shorter():
    navigation = AtomicEnv("navigation", difficulty=0)
    navigation.reset(seed=9)
    assert navigation._position != navigation._goal
    dr = navigation._goal[0] - navigation._position[0]
    dc = navigation._goal[1] - navigation._position[1]
    key = {(-1, 0): "w", (1, 0): "s", (0, -1): "a", (0, 1): "d"}[(dr, dc)]
    _, _, done, _, info = navigation.step(PhysicalAction(keys=(key,)))
    assert done and info["success"]
    mouse = AtomicEnv("mouse", difficulty=0)
    mouse.reset(seed=4)
    _, _, done, _, info = mouse.step(PhysicalAction(mouse_buttons=("left",)))
    assert not done and not info["failure"]
    assert mouse._misses == 1  # A visible limited budget, not an immediate terminal click.
    mouse.step(PhysicalAction())
    target = mouse._mouse_location(mouse._level["targets"][0])
    dx, dy = target[0] - mouse._cursor[0], target[1] - mouse._cursor[1]
    _, _, done, _, info = mouse.step(PhysicalAction(
        mouse_buttons=("left",),
        dx=min(MOUSE_BINS, key=lambda value: abs(value - dx)),
        dy=min(MOUSE_BINS, key=lambda value: abs(value - dy)),
    ))
    assert np.hypot(dx, dy) <= 14.001 and done and info["success"]


def test_initial_memory_stage_still_requires_remembering_a_disappeared_cue():
    env = AtomicEnv("memory", difficulty=0)
    _, info = env.reset(seed=0)
    assert not info["success"] and env._level["unlock"] < 10
    assert env._level["unlock"] > env._level["cue_ticks"]
    cue = env._level["cue"]
    for _ in range(env._level["unlock"]):
        _, _, done, _, _ = env.step(PhysicalAction())
        assert not done
    while not done:
        _, _, done, _, info = env.step(PhysicalAction(keys=("left" if cue == 0 else "right",)))
    assert info["success"] and info["elapsed_ticks"] < 16


@pytest.mark.parametrize("bad", (True, -0.1, 1.1, float("nan"), float("inf"), "easy"))
def test_difficulty_rejects_invalid_values(bad):
    with pytest.raises(ValueError, match="difficulty"):
        AtomicEnv("navigation", difficulty=bad)
