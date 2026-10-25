"""Opt-in pixel materials vary episodes while preserving HUD and mechanics."""

from dataclasses import asdict

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv
from carryrl.envs.atomic_suite import TASKS
from carryrl.envs.pixel_art import PIXEL_STYLE_SPLITS, style_pixel_art
from carryrl.envs.presentation import (
    PresentationConfig,
    PresentationProfile,
    choose_presentation,
    present_frame,
)
from carryrl.rewards.visual import HUD_HEIGHT, draw_visible_hud, verify_visible_outcome


@pytest.mark.parametrize("task", TASKS)
def test_opt_in_changes_every_game_preserves_hud_and_physics(task):
    raw = AtomicEnv(task, max_steps=20)
    styled = AtomicEnv(task, max_steps=20, visual_config={"pixel_art": True})
    frame, info = raw.reset(seed=83, options={"level_index": 2})
    pixel_frame, pixel_info = styled.reset(seed=83, options={"level_index": 2})
    profile = pixel_info.pop("presentation")
    assert profile["pixel_art"] and profile["canonical_hud"]
    assert info.pop("presentation") is None and info == pixel_info
    assert pixel_frame.shape == frame.shape == (96, 96, 3)
    assert styled.observation_space.shape == raw.observation_space.shape
    assert not np.array_equal(frame[HUD_HEIGHT:], pixel_frame[HUD_HEIGHT:])
    actions = (PhysicalAction(), PhysicalAction(keys=("right",)),
               PhysicalAction(keys=("space",)), PhysicalAction(keys=("down",)),
               PhysicalAction(mouse_buttons=("left",), dx=4, dy=-4))
    for tick in range(20):
        np.testing.assert_array_equal(frame[:HUD_HEIGHT], pixel_frame[:HUD_HEIGHT])
        np.testing.assert_array_equal(raw.canonical_render(), styled.canonical_render())
        np.testing.assert_array_equal(styled.render(), pixel_frame)
        action = actions[tick % len(actions)]
        frame, reward, done, truncated, info = raw.step(action)
        pixel_frame, pixel_reward, pixel_done, pixel_truncated, pixel_info = styled.step(action)
        assert pixel_info.pop("presentation") == profile
        info.pop("presentation")
        assert (reward, done, truncated, info) == (
            pixel_reward, pixel_done, pixel_truncated, pixel_info)
        np.testing.assert_array_equal(frame[:HUD_HEIGHT], pixel_frame[:HUD_HEIGHT])
        if done or truncated:
            break


def test_episode_variation_reproducibility_and_split_families():
    config = PresentationConfig(pixel_art=True, seed=51)
    env = AtomicEnv("navigation", visual_config=config)
    frames, profiles = [], []
    for seed in range(12):
        frame, info = env.reset(seed=seed, options={"level_index": 3})
        frames.append(frame)
        profiles.append(info["presentation"])
        repeated, repeated_info = env.reset(seed=seed, options={"level_index": 3})
        np.testing.assert_array_equal(frame, repeated)
        assert repeated_info["presentation"] == info["presentation"]
    assert len({frame.tobytes() for frame in frames}) == len(frames)
    assert len({profile["pixel_style"] for profile in profiles}) > 1
    families = [set(styles) for styles in PIXEL_STYLE_SPLITS.values()]
    assert all(not left & right for i, left in enumerate(families) for right in families[i + 1:])
    for split, styles in PIXEL_STYLE_SPLITS.items():
        for seed in range(16):
            assert choose_presentation(config, split, seed).pixel_style in styles
            assert choose_presentation(
                PresentationConfig(pixel_art=True, style_split=split), "train", seed
            ).pixel_style in styles


def test_same_episode_style_does_not_use_level_rng_or_level_identity():
    env = AtomicEnv("navigation", visual_config={"pixel_art": True})
    _, first = env.reset(seed=8, options={"level_index": 0})
    _, second = env.reset(seed=8, options={"level_index": 5})
    assert first["level_id"] != second["level_id"]
    assert first["presentation"] == second["presentation"]
    other = AtomicEnv("reaction", visual_config={"pixel_art": True})
    _, third = other.reset(seed=8)
    for name in ("texture_seed", "pixel_style", "native_size"):
        assert first["presentation"][name] == third["presentation"][name]
    assert first["presentation"]["pixel_task"] == "navigation"
    assert third["presentation"]["pixel_task"] == "reaction"


def test_world_materials_stay_fixed_and_sprite_shading_moves_with_sprite():
    frame = np.full((96, 96, 3), (31, 38, 52), dtype=np.uint8)
    frame[40:48, 30:38] = (35, 180, 235)
    moved = np.full_like(frame, (31, 38, 52))
    moved[42:50, 33:41] = (35, 180, 235)
    styled = style_pixel_art(frame, "moss", 13)
    styled_moved = style_pixel_art(moved, "moss", 13)
    common_ground = np.all(frame == (31, 38, 52), axis=-1)
    common_ground &= np.all(moved == (31, 38, 52), axis=-1)
    np.testing.assert_array_equal(styled[common_ground], styled_moved[common_ground])
    np.testing.assert_array_equal(styled[40:48, 30:38], styled_moved[42:50, 33:41])


@pytest.mark.parametrize("status", ("Play", "Success", "Failed"))
@pytest.mark.parametrize("score", range(6))
def test_all_outcomes_preserve_every_canonical_hud_pixel(status, score):
    frame = np.full((96, 96, 3), (31, 38, 52), dtype=np.uint8)
    draw_visible_hud(frame, score, status)
    for styles in PIXEL_STYLE_SPLITS.values():
        for style in styles:
            profile = PresentationProfile(96, "slate", "top", pixel_art=True,
                                          pixel_style=style, canonical_hud=True)
            styled = present_frame(frame, profile)
            np.testing.assert_array_equal(frame[:HUD_HEIGHT], styled[:HUD_HEIGHT])
            assert verify_visible_outcome(styled) == verify_visible_outcome(frame)


def test_old_default_and_serialized_profiles_still_work():
    assert choose_presentation(PresentationConfig(), "train", 0) is None
    assert choose_presentation({}, "train", 0) is None
    profile = PresentationProfile(**{"native_size": 96, "theme": "slate", "hud_layout": "top"})
    assert profile.pixel_art is False and profile.canonical_hud is False
    env = AtomicEnv("navigation")
    default = AtomicEnv("navigation", visual_config={"mode": "canonical"})
    np.testing.assert_array_equal(env.reset(seed=21)[0], default.reset(seed=21)[0])
    styled = choose_presentation(PresentationConfig(pixel_art=True), "train", 9)
    assert PresentationProfile(**asdict(styled)) == styled


@pytest.mark.parametrize("invalid", (1, "yes", None))
def test_pixel_art_requires_boolean(invalid):
    with pytest.raises(ValueError, match="pixel_art must be a boolean"):
        PresentationConfig(pixel_art=invalid)


@pytest.mark.parametrize("task", ("ui", "shooting", "goal_choice", "memory", "mouse"))
def test_selection_health_text_and_crosshairs_keep_their_actual_pixels(task):
    env = AtomicEnv(task, visual_config={"pixel_art": True})
    env.reset(seed=83, options={"level_index": 2})
    for action in (PhysicalAction(), PhysicalAction(keys=("tab",)), PhysicalAction()):
        actual, _, done, truncated, _ = env.step(action)
        canonical = env.canonical_render()
        for color in ((245, 247, 252), (250, 245, 218), (142, 242, 187)):
            mask = np.all(canonical == color, axis=-1)
            np.testing.assert_array_equal(actual[mask], canonical[mask])
        if task == "shooting":
            # Small green health components must never become portal icons.
            green = np.all(canonical == (69, 207, 100), axis=-1)
            for row in range(HUD_HEIGHT, 43):
                if 0 < green[row].sum() <= 9:
                    np.testing.assert_array_equal(actual[row, green[row]],
                                                  canonical[row, green[row]])
        if task == "mouse":
            cursor = np.all(canonical == (35, 180, 235), axis=-1)
            np.testing.assert_array_equal(actual[cursor], canonical[cursor])
        if done or truncated:
            break


def test_ui_items_are_simple_two_color_symbols():
    from carryrl.envs.atomic_combat import ITEM_COLORS

    env = AtomicEnv("ui", visual_config={"pixel_art": True})
    env.reset(seed=83, options={"level_index": 2})
    actual, *_ = env.step(PhysicalAction(keys=("tab",)))
    canonical = env.canonical_render()
    visible_items = 0
    for color in ITEM_COLORS:
        mask = np.all(canonical == color, axis=-1)
        if not mask.any():
            continue
        visible_items += 1
        assert len(np.unique(actual[mask], axis=0)) == 2
    assert visible_items >= 3


def test_road_dashes_follow_visible_curves_inside_existing_road():
    frame = np.full((96, 96, 3), (31, 38, 52), dtype=np.uint8)
    centers = {}
    for y in range(HUD_HEIGHT, 96):
        center = 48 + int(15 * np.sin(y / 10))
        centers[y] = center
        frame[y, center - 12:center + 13] = (187, 198, 212)
    styled = style_pixel_art(frame, "moss", 13, task="steering")
    for y, center in centers.items():
        if y % 11 < 5:
            np.testing.assert_array_equal(styled[y, center], styled[y, center - 12])
            assert not np.array_equal(styled[y, center], styled[y, center - 1])


def test_role_renderer_changes_no_unrecognized_text_or_pips():
    frame = np.full((96, 96, 3), (31, 38, 52), dtype=np.uint8)
    frame[40, 10:13] = (69, 207, 100)
    frame[44:47, 20:23] = (233, 244, 238)
    actual = style_pixel_art(frame, "moss", 13, task="shooting")
    np.testing.assert_array_equal(actual[40, 10:13], frame[40, 10:13])
    np.testing.assert_array_equal(actual[44:47, 20:23], frame[44:47, 20:23])


def test_material_families_differ_in_structure_beyond_palette():
    frame = np.full((96, 96, 3), (187, 198, 212), dtype=np.uint8)
    patterns = []
    for styles in PIXEL_STYLE_SPLITS.values():
        for style in styles:
            actual = style_pixel_art(frame, style, 13, task="navigation")[HUD_HEIGHT:]
            colors, counts = np.unique(actual.reshape(-1, 3), axis=0, return_counts=True)
            dominant = colors[counts.argmax()]
            patterns.append(np.all(actual == dominant, axis=-1).tobytes())
    assert len(set(patterns)) == 7


@pytest.mark.parametrize("task", TASKS)
def test_material_seed_changes_scenery_without_changing_visible_roles(task):
    env = AtomicEnv(task)
    frame, _ = env.reset(seed=83, options={"level_index": 2})
    original = frame.copy()
    scenery = np.zeros(frame.shape[:2], dtype=bool)
    for color in ((31, 38, 52), (187, 198, 212), (58, 67, 80)):
        scenery |= np.all(frame == color, axis=-1)
    scenery[:HUD_HEIGHT] = False
    variants = [style_pixel_art(frame, "moss", seed, task=task) for seed in range(6)]
    for variant in variants:
        np.testing.assert_array_equal(variant[~scenery], variants[0][~scenery])
        np.testing.assert_array_equal(variant[:HUD_HEIGHT], frame[:HUD_HEIGHT])
    assert len({variant[scenery].tobytes() for variant in variants}) == len(variants)
    # Re-render after other seeds and after a game step: light has no clock/state.
    env.step(PhysicalAction())
    np.testing.assert_array_equal(
        variants[0], style_pixel_art(frame, "moss", 0, task=task))
    np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize("style", sum(PIXEL_STYLE_SPLITS.values(), ()))
def test_each_family_varies_material_scale_and_light_with_episode_seed(style):
    frame = np.full((96, 96, 3), (31, 38, 52), dtype=np.uint8)
    frame[32:64] = (187, 198, 212)
    frame[64:] = (58, 67, 80)
    variants = [style_pixel_art(frame, style, seed, task="navigation") for seed in range(16)]
    for start, end in ((HUD_HEIGHT, 32), (32, 64), (64, 96)):
        assert len({variant[start:end].tobytes() for variant in variants}) >= 8
    # Horizontal equal-color runs ignore palette/light changes and expose the
    # material's spatial pattern. Several transition counts require more than
    # merely translating a fixed texture or changing its tint.
    counts = {
        int(np.any(variant[33:63, 1:] != variant[33:63, :-1], axis=-1).sum())
        for variant in variants
    }
    assert len(counts) >= 3
