"""Controlled train appearance copies preserve physical trajectories and legacy RGB."""

import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import atomic_level_banks
from carryrl.envs.presentation import (
    HUD_LAYOUTS,
    THEME_SPLITS,
    PresentationProfile,
    present_frame,
)
from carryrl.training.imitation import (
    DemonstrationDataset,
    augment_demonstrations,
    collect_memory_cue_counterparts,
    collect_native_scripted_demonstrations,
    collect_scripted_corrections,
    collect_scripted_demonstrations,
)
from carryrl.training.sequence_config import SequenceRunConfig

spec = importlib.util.spec_from_file_location(
    "augmentation_training_script", Path(__file__).resolve().parents[1]
    / "scripts/train_imitation.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


@pytest.fixture
def canonical():
    return collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=1)


def test_defaults_match_legacy_profiles_rgb_and_serialized_provenance(canonical):
    expected = DemonstrationDataset()
    rng = np.random.default_rng(31)
    for episode in canonical.episodes:
        for _ in range(5):
            # Historical draws and profile defaults, deliberately independent of
            # the controlled augmentation implementation.
            profile = PresentationProfile(native_size=episode.native_size or 96,
                theme=str(rng.choice(THEME_SPLITS["train"])),
                hud_layout=str(rng.choice(HUD_LAYOUTS)),
                texture_seed=int(rng.integers(2**31)), textures=True)
            mapping = {}
            for index in (*episode.frame_indices, episode.final_frame_index):
                if index not in mapping:
                    mapping[index] = expected.intern_frame(
                        present_frame(canonical.frames[index], profile))
            expected.add_episode(replace(episode,
                frame_indices=tuple(mapping[index] for index in episode.frame_indices),
                final_frame_index=mapping[episode.final_frame_index], presentation=asdict(profile)))
    actual = augment_demonstrations(canonical, seed=31, copies_per_episode=5)
    assert json.dumps(actual.provenance(), sort_keys=True) == json.dumps(
        expected.provenance(), sort_keys=True)
    assert [frame.tobytes() for frame in actual.frames] == [
        frame.tobytes() for frame in expected.frames]


def test_fixed_plain_profile_preserves_memory_pair_equality_and_actual_context():
    canonical = collect_scripted_demonstrations(1, tasks=("memory",), difficulty=1,
        level_bank_seed=20261008, episode_instructions={"memory": ("Remember the door",)})
    canonical.extend(collect_memory_cue_counterparts(canonical, max_pairs=1))
    original_frames = [frame.tobytes() for frame in canonical.frames]
    augmented = augment_demonstrations(canonical, themes=("slate",), hud_layouts=("left",),
        textures=False, canvas_padding=12)
    assert len(augmented.episodes) == 2
    for before, after in zip(canonical.episodes, augmented.episodes, strict=True):
        assert after.actions is before.actions
        assert after.executed_actions is before.executed_actions
        assert after.instruction == before.instruction
        assert replace(after, frame_indices=before.frame_indices,
            final_frame_index=before.final_frame_index, presentation=None) == before
        profile = PresentationProfile(**after.presentation)
        assert (profile.theme, profile.hud_layout, profile.textures, profile.canvas_padding) == (
            "slate", "left", False, 12)
        x0, y0, x1, y1 = profile.viewport_box
        assert (x1 - x0, y1 - y0) == (96, 96)
        for source, target in zip((*before.frame_indices, before.final_frame_index),
                                 (*after.frame_indices, after.final_frame_index), strict=True):
            np.testing.assert_array_equal(augmented.frames[target],
                                          present_frame(canonical.frames[source], profile))
    level = atomic_level_banks("memory", 20261008)["train"][canonical.episodes[0].level_index]
    left, right = augmented.episodes
    assert not np.array_equal(augmented.frames[left.frame_indices[0]],
                              augmented.frames[right.frame_indices[0]])
    for tick in range(level["cue_ticks"], level["unlock"]):
        np.testing.assert_array_equal(augmented.frames[left.frame_indices[tick]],
                                      augmented.frames[right.frame_indices[tick]])
    assert original_frames == [frame.tobytes() for frame in canonical.frames]


@pytest.mark.parametrize("options", [
    {"themes": ()}, {"themes": ("slate", "slate")}, {"themes": ("violet",)},
    {"themes": ("ice",)}, {"themes": ("ember",)}, {"themes": ("slate", "ice")},
    {"themes": 1}, {"themes": (["slate"],)},
    {"hud_layouts": ()}, {"hud_layouts": ("top", "top")},
    {"hud_layouts": ("unknown",)}, {"hud_layouts": 1},
    {"textures": 1}, {"textures": None},
    {"canvas_padding": 3}, {"canvas_padding": 65}, {"canvas_padding": True},
    {"canvas_padding": 8.0},
])
def test_invalid_controls_fail_before_any_transformation(canonical, monkeypatch, options):
    monkeypatch.setattr("carryrl.envs.presentation.present_frame",
                        lambda *a: pytest.fail("invalid controls reached RGB transformation"))
    with pytest.raises(ValueError, match="augmentation"):
        augment_demonstrations(canonical, **options)


def test_all_shapes_are_checked_before_transforming_earlier_episodes(canonical, monkeypatch):
    canonical.extend(collect_native_scripted_demonstrations(native_train_sizes=(64,),
        native_episodes_per_task=1, tasks=("navigation",), difficulty=1))
    index = canonical.episodes[-1].final_frame_index
    canonical.frames[index] = canonical.frames[index][:32, :32]
    monkeypatch.setattr("carryrl.envs.presentation.present_frame",
                        lambda *a: pytest.fail("bad shape reached RGB transformation"))
    with pytest.raises(ValueError):
        augment_demonstrations(canonical, themes=("sand",), textures=False)


def test_controlled_correction_keeps_distinct_labels_and_executed_history():
    class IdleLearner:
        training = False

        def eval(self):
            pass

        def train(self, mode=True):
            pass

        def reset_episode(self, **kwargs):
            pass

        def act(self, frame, deterministic):
            return SimpleNamespace(action=PhysicalAction())

        def observe_action(self, action):
            pass

    canonical = collect_scripted_corrections(IdleLearner(), 1,
        tasks=("navigation",), expert_probability=0, max_steps=3)
    transformed = augment_demonstrations(canonical, themes=("forest",),
        hud_layouts=("right",), textures=False, canvas_padding=4)
    before, after = canonical.episodes[0], transformed.episodes[0]
    assert before.actions != before.executed_actions
    assert after.actions is before.actions
    assert after.executed_actions is before.executed_actions
    assert after.outcome == before.outcome


@pytest.mark.parametrize("flag", [
    ["--augmentation-themes", "slate"], ["--augmentation-hud-layouts", "top"],
    ["--augmentation-textures"], ["--no-augmentation-textures"],
    ["--augmentation-canvas-padding=8"],
])
@pytest.mark.parametrize("reuse", [False, True])
def test_cli_rejects_explicit_controls_without_copies_or_with_reuse(monkeypatch, capsys, flag,
                                                                  reuse):
    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "unused",
        "--warm-start", "unused", "--output", "unused", *flag,
        *(["--demonstration-reuse-run", "unused", "--presentation-copies-per-episode", "1"]
          if reuse else [])])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
    assert ("reuse excludes collection flags" if reuse else "controls require positive") in (
        capsys.readouterr().err)


def test_cli_legacy_control_defaults(monkeypatch):
    parsed = []
    parse = script.argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        pass

    def capture(parser, *args, **kwargs):
        result = parse(parser, *args, **kwargs)
        parsed.append(result)
        raise Parsed

    monkeypatch.setattr(script.argparse.ArgumentParser, "parse_args", capture)
    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "unused",
        "--warm-start", "unused", "--output", "unused"])
    with pytest.raises(Parsed):
        script.main()
    assert (parsed[0].augmentation_themes, parsed[0].augmentation_hud_layouts,
            parsed[0].augmentation_textures, parsed[0].augmentation_canvas_padding) == (
        None, None, True, 8)
    assert (parsed[0].appearance_bundle, parsed[0].appearance_bundle_sha256,
            parsed[0].appearance_arm) == (None, None, None)


@pytest.mark.parametrize("flags", [
    ["--augmentation-themes", "slate", "slate"],
    ["--augmentation-hud-layouts", "top", "top"],
    ["--augmentation-themes", "violet"],
    ["--augmentation-canvas-padding", "3"],
    ["--augmentation-canvas-padding", "65"],
])
def test_cli_invalid_controls_fail_before_config_or_model(monkeypatch, capsys, flags):
    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "unused",
        "--warm-start", "unused", "--output", "unused", "--presentation-copies-per-episode",
        "1", *flags])
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
    assert "augmentation" in capsys.readouterr().err


def test_cli_forwards_controls_to_combined_canonical_and_native_copies(monkeypatch, tmp_path,
                                                                     canonical):
    config = SequenceRunConfig(tasks=("navigation",))
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "manifest.json").write_text(json.dumps({"sha256": "5" * 64}))
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)
    monkeypatch.setattr(script, "provenance", lambda: {})
    monkeypatch.setattr(script, "tensor_hash", lambda _: "6" * 64)
    monkeypatch.setattr(script, "load_sequence_checkpoint", lambda *a, **k: (
        object(), SimpleNamespace(param_groups=[{}]), object(),
        {"decisions": 0, "weights_revision": 0, "config": config.to_dict()}))
    monkeypatch.setattr(script, "collect_scripted_demonstrations", lambda *a, **k: canonical)
    native = collect_native_scripted_demonstrations(native_train_sizes=(64, 128),
        native_episodes_per_task=1, tasks=("navigation",), difficulty=1)
    monkeypatch.setattr(script, "collect_native_scripted_demonstrations", lambda **k: native)
    called = []

    class Prepared(Exception):
        pass

    def stop_at_augmentation(dataset, **options):
        called.append(options)
        assert {frame.shape for frame in dataset.frames} == {
            (64, 64, 3), (96, 96, 3), (128, 128, 3)}
        transformed = augment_demonstrations(dataset, **options)
        assert {episode.native_size for episode in transformed.episodes} == {None, 64, 128}
        for episode in transformed.episodes:
            assert episode.presentation["native_size"] == (episode.native_size or 96)
        raise Prepared

    monkeypatch.setattr(script, "augment_demonstrations", stop_at_augmentation)
    monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "unused",
        "--warm-start", str(parent), "--output", str(tmp_path / "output"),
        "--training-backend", "torch", "--presentation-copies-per-episode", "1",
        "--native-train-sizes", "64", "128", "--native-episodes-per-task", "1",
        "--augmentation-themes", "sand", "--augmentation-hud-layouts", "bottom",
        "--no-augmentation-textures", "--augmentation-canvas-padding", "16"])
    with pytest.raises(Prepared):
        script.main()
    assert called == [{"seed": 8176, "copies_per_episode": 1, "max_frame_bytes": 256 * 1024**2,
                       "themes": ["sand"], "hud_layouts": ["bottom"], "textures": False,
                       "canvas_padding": 16}]
