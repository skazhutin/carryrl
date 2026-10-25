import gzip
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from carryrl.dashboard.frozen_playback import MAX_FRAMES, write_frozen_playback
from carryrl.envs.atomic import AtomicEnv, verify_visible_outcome


@pytest.mark.parametrize("archived", [False, True])
def test_frozen_loader_restores_ocr_reader_without_new_policy_parameters(
        tmp_path, monkeypatch, archived):
    from test_agent_instructions import TinyBackbone, TinyTokenizer
    from test_sequence_training import TinyReward, TinyVision, config

    from carryrl.agent.language import MultilingualInstructionEncoder
    from carryrl.agent.model import GeneralistAgent
    from carryrl.dashboard import frozen_playback
    from carryrl.training.sequence import save_sequence_checkpoint, tensor_hash

    class LocalLanguage(MultilingualInstructionEncoder):
        def __init__(self):
            super().__init__(tokenizer=TinyTokenizer(), backbone=TinyBackbone())

        def save_resources(self, directory):
            directory.mkdir(parents=True)
            for name in ("carryrl_language.json", "config.json", "tokenizer.json",
                         "tokenizer_config.json"):
                (directory / name).write_text("{}")

    class EmptyOptimizer:
        def state_dict(self):
            return {}

    cfg = config()
    cfg = replace(cfg, agent=replace(cfg.agent, text_encoder="multilingual_e5_small"),
                  screen_text_language="eng")
    agent = GeneralistAgent(cfg.agent, vision=TinyVision(), language=LocalLanguage())
    run = tmp_path / "runs" / "frozen"
    run.mkdir(parents=True)
    checkpoint = save_sequence_checkpoint(agent, EmptyOptimizer(), TinyReward(), cfg,
        run, rollout=0, decisions=0, revision=1)
    if archived:
        state = checkpoint / "state.pt"
        state.with_suffix(".pt.gz").write_bytes(gzip.compress(state.read_bytes()))
        state.unlink()
        monkeypatch.setattr("carryrl.training.checkpoint_storage.shutil.disk_usage",
                            lambda path: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(MultilingualInstructionEncoder, "from_resources",
                        lambda *args, **kwargs: LocalLanguage())
    monkeypatch.setattr(frozen_playback, "GeneralistAgent", lambda model_config, language:
        GeneralistAgent(model_config, vision=TinyVision(), language=language))
    readers = []

    def reader(**kwargs):
        readers.append(kwargs)
        return object()

    monkeypatch.setattr("carryrl.perception.screen_text.ScreenTextReader", reader)
    restored, restored_cfg, _ = frozen_playback.load_frozen_agent(checkpoint)
    assert readers == [{"language": "eng", "cache_entries": 128}]
    assert restored.screen_text_reader is not None
    assert restored_cfg.screen_text_language == "eng"
    assert tensor_hash(restored) == tensor_hash(agent)
    assert (checkpoint / "state.pt").is_file()
    assert not (checkpoint / "state.pt.gz").exists()


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
