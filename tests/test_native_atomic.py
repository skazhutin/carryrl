"""Native geometry remains visible and preserves the canonical 96px protocol."""

import hashlib

import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv, verify_visible_outcome
from carryrl.envs.atomic_combat import ITEM_COLORS, UIGame, _ui_panel_top
from carryrl.training.imitation import scripted_teacher

_CANONICAL_DIGESTS = {
    "reaction": "b2bab48a7359c9578c4166707a4c7bb329569aae504cea023aa4de461b73e738",
    "ui": "05fe72706f7d7b738026fc33101f05f3a8ed535c4165b9a0e53ef9d3e5aefedf",
}


def _episode(env, index, digest=None):
    frame, _ = env.reset(seed=8176 + index, options={"level_index": index})
    teacher = scripted_teacher(env.task)
    if digest is not None:
        digest.update(frame.tobytes())
    for step in range(80):
        action = teacher.act(frame)
        if digest is not None:
            digest.update(repr(action).encode())
        frame, reward, terminated, truncated, info = env.step(action)
        if digest is not None:
            # The historical fingerprint still checks every frame, action and physical
            # outcome. Exclude only the additive, versioned efficiency diagnostics.
            historical_info = {key: value for key, value in info.items() if key not in {
                "scoring_version", "unnecessary_input_units", "efficiency_scope",
                "optimal_action_units", "action_units", "completion_efficiency",
            }}
            digest.update(frame.tobytes())
            digest.update(repr((reward, terminated, truncated, historical_info)).encode())
        if terminated or truncated:
            break
    outcome = verify_visible_outcome(frame)
    assert outcome["known"] and outcome["success"] and outcome["score"] == 5, (
        env.image_size, env.task, index, step + 1, outcome,
    )
    return step + 1


@pytest.mark.parametrize("task", ("reaction", "ui"))
@pytest.mark.parametrize("size", (64, 128))
def test_full_native_train_bank_completes_with_rgb_teacher_within_public_budget(task, size):
    env = AtomicEnv(task, image_size=size, difficulty=1, max_steps=80,
                    level_bank_seed=20261008)
    try:
        for index in range(64):
            _episode(env, index)
    finally:
        env.close()


@pytest.mark.parametrize("task", ("reaction", "ui"))
def test_canonical_96_trajectories_match_before_native_geometry_fix_byte_for_byte(task):
    # Captured before the fix: all 64 full-difficulty train levels, including
    # RGB frames, physical labels, rewards, termination flags and public info.
    digest = hashlib.sha256()
    env = AtomicEnv(task, image_size=96, level_bank_seed=20261008)
    try:
        for index in range(64):
            _episode(env, index, digest)
    finally:
        env.close()
    assert digest.hexdigest() == _CANONICAL_DIGESTS[task]


@pytest.mark.parametrize("size", (64, 96, 128))
def test_reaction_presentation_does_not_change_native_physics(size):
    native = AtomicEnv("reaction", image_size=size, level_bank_seed=20261008)
    varied = AtomicEnv("reaction", image_size=size, level_bank_seed=20261008,
                       visual_config={"mode": "varied", "native_sizes": (size,)})
    for env in (native, varied):
        env.reset(seed=7, options={"level_index": 0})
    assert native._reaction_drop == varied._reaction_drop
    assert (size - 12 - native._ball_y) / native._reaction_drop == pytest.approx(51 / 2.5)
    for _ in range(20):
        native.step(PhysicalAction())
        varied.step(PhysicalAction())
        assert (native._ball_x, native._ball_y, native._paddle_x) == (
            varied._ball_x, varied._ball_y, varied._paddle_x,
        )


@pytest.mark.parametrize("task", ("reaction", "ui"))
@pytest.mark.parametrize("size", (80, 112))
def test_geometry_generalizes_to_other_valid_native_sizes(task, size):
    env = AtomicEnv(task, image_size=size, level_bank_seed=20261008)
    try:
        _episode(env, 3)
    finally:
        env.close()


def test_small_inventory_panel_does_not_clip_visible_request_glyph():
    game = UIGame({"request": 1, "items": [0, 1, 2, 3], "selection": 0,
                   "layout": 1, "request_x": 48, "menu_offset": 0}, 64)
    before = game.render()[28:_ui_panel_top(64)]
    game.step(PhysicalAction(keys=("tab",)))
    after = game.render()[28:_ui_panel_top(64)]
    assert (before == after).all()
    assert (after == ITEM_COLORS[1]).all(axis=-1).sum() == 49


@pytest.mark.parametrize("size", (63, 0, -1, True, 64.0, "128", None))
def test_invalid_native_sizes_are_rejected(size):
    with pytest.raises(ValueError, match="image_size must be an integer >= 64"):
        AtomicEnv("reaction", image_size=size)
