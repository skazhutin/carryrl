import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from carryrl.dashboard.frozen_playback import MAX_FRAMES, write_frozen_playback
from carryrl.envs.atomic import AtomicEnv, verify_visible_outcome


def episode():
    from carryrl.agent.actions import PhysicalAction

    env = AtomicEnv("navigation", split="validation", difficulty=0, max_steps=2)
    try:
        initial, info = env.reset(seed=1_000_000, options={"level_index": 0})
        action = PhysicalAction()
        final, _, terminated, truncated, _ = env.step(action)
        record = {"task": "navigation", "policy": "frozen_deterministic", "split": "validation",
                  "seed": 1_000_000, "level_index": 0, "level_id": info["level_id"],
                  "steps": 1, "outcome": verify_visible_outcome(final),
                  "terminated": terminated, "truncated": truncated,
                  "initial_frame_sha256": hashlib.sha256(initial.tobytes()).hexdigest()}
        return record, [initial.copy(), final.copy()], [action.to_dict()]
    finally:
        env.close()


def test_writer_hash_alignment_lossless_sprite_and_preserves_existing_files(tmp_path):
    record, frames, actions = episode()
    path = write_frozen_playback(tmp_path, record, frames, actions, epoch=80,
                                 agent_sha256="a" * 64, weights_revision=6837)
    row = json.loads(path.read_text())
    assert row["source"] == "actual_frozen_learned_policy_closed_loop_RGB_playback"
    assert row["teacher_actions_used"] is False and row["training_performed"] is False
    assert row["action_count"] + 1 == row["observation_frame_count"] == 2
    assert row["checkpoint"] is None and row["checkpoint_linkage"] == "evaluated_policy_hash"
    assert hashlib.sha256(path.with_suffix(".gif").read_bytes()).hexdigest() == row["gif_sha256"]
    sprite = path.parent / row["sprite_filename"]
    assert hashlib.sha256(sprite.read_bytes()).hexdigest() == row["sprite_sha256"]
    with Image.open(sprite) as image:
        rgb = np.asarray(image)
    width, height = row["frame_width"], row["frame_height"]
    assert np.array_equal(rgb[:height, :width], frames[0])
    assert np.array_equal(rgb[:height, width:2 * width], frames[-1])
    assert row["actions"][0]["observation_sha256"] == row["frame_sha256"][0]
    assert row["actions"][0]["next_observation_sha256"] == row["frame_sha256"][-1]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_frozen_playback(tmp_path, record, frames, actions, epoch=80,
                              agent_sha256="a" * 64, weights_revision=6837)
    assert path.read_bytes() == before


def test_writer_rejects_teacher_policy_bad_alignment_and_checkpoint_linkage(tmp_path):
    record, frames, actions = episode()
    options = {"epoch": 80, "agent_sha256": "a" * 64, "weights_revision": 6837}
    with pytest.raises(ValueError, match="deterministic learned policy"):
        write_frozen_playback(tmp_path, {**record, "policy": "scripted_teacher"},
                              frames, actions, **options)
    with pytest.raises(ValueError, match="initial RGB"):
        write_frozen_playback(tmp_path, record, frames[:-1], actions, **options)
    with pytest.raises(ValueError, match="at most 81"):
        write_frozen_playback(tmp_path, record, frames * MAX_FRAMES, actions, **options)
    with pytest.raises(ValueError, match="checkpoint manifest"):
        write_frozen_playback(tmp_path, record, frames, actions, checkpoint=tmp_path,
                              manifest={"agent_sha256": "b" * 64}, **options)
