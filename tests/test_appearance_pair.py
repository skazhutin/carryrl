"""Real canonical TRAIN origins support two exact posthoc RGB arms, without training."""

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
import torch

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.model import AgentConfig
from carryrl.training.appearance import (
    ARMS,
    PreparedAppearanceReuse,
    _combined_identity,
    _identities,
    prepare_appearance_bundle,
)
from carryrl.training.imitation import (
    collect_scripted_corrections,
    collect_scripted_demonstrations,
)
from carryrl.training.sequence_config import SequenceRunConfig


@pytest.fixture
def pair(tmp_path):
    config = SequenceRunConfig(tasks=("navigation", "reaction"),
        agent=AgentConfig(text_encoder="multilingual_e5_small"),
        level_bank_seed=20261008, curriculum_difficulty=1.0, evaluation_difficulty=1.0,
        episode_instructions={"navigation": ("Reach the goal",), "reaction": ("Catch the ball",)})
    dataset = collect_scripted_demonstrations(1, tasks=config.tasks, seed=8288,
        level_bank_seed=config.level_bank_seed, episode_instructions=config.episode_instructions)
    directory = tmp_path / "pair"
    report = prepare_appearance_bundle(dataset, directory, config=config)
    return directory, config, dataset, report


def load(pair, arm="control", **kwargs):
    directory, config, _, report = pair
    return PreparedAppearanceReuse.prepare(directory,
        expected_sha256=kwargs.pop("expected_sha256", report["source_bundle_sha256"]),
        arm=arm, config=kwargs.pop("config", config), **kwargs)


def repin(pair):
    return _combined_identity(_identities(pair[0], 256 * 1024**2))


def test_real_pair_roundtrip_equal_counts_actual_history_and_source_preservation(pair):
    directory, _, original, report = pair
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in directory.iterdir()}
    control, shift = (load(pair, arm=name) for name in ARMS)
    assert report["status"] == "prepared_not_trained"
    assert report["updates"] == 0 and not report["model_loaded"]
    assert not report["checkpoint_created"] and report["loader_roundtrip_verified"]
    for reuse in (control, shift):
        assert len(reuse.dataset.episodes) == 2 * len(original.episodes)
        assert sum(map(len, reuse.dataset.episodes)) == 2 * sum(map(len, original.episodes))
        assert reuse.summary["source_rows"] == len(original.episodes)
        assert reuse.summary["posthoc_copies"] == len(original.episodes)
        assert reuse.dataset.frame_bytes < reuse.summary["shared_frame_bytes"]
        source, views = (reuse.dataset.episodes[:len(original.episodes)],
                         reuse.dataset.episodes[len(original.episodes):])
        for original_row, raw, view in zip(original.episodes, source, views, strict=True):
            assert replace(view, frame_indices=raw.frame_indices,
                final_frame_index=raw.final_frame_index, presentation=None) == raw
            assert raw.to_dict() == original_row.to_dict()
            for previous, current in zip((*original_row.frame_indices,
                                          original_row.final_frame_index),
                                         (*raw.frame_indices, raw.final_frame_index), strict=True):
                np.testing.assert_array_equal(original.frames[previous],
                                              reuse.dataset.frames[current])
        reuse.assert_unchanged()
    for left, right in zip(control.dataset.episodes[:2], shift.dataset.episodes[:2], strict=True):
        assert left.to_dict() == right.to_dict()
        for a, b in zip((*left.frame_indices, left.final_frame_index),
                        (*right.frame_indices, right.final_frame_index), strict=True):
            np.testing.assert_array_equal(control.dataset.frames[a], shift.dataset.frames[b])
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in directory.iterdir()}
    with pytest.raises(FileExistsError):
        prepare_appearance_bundle(original, directory, config=pair[1])


@pytest.mark.parametrize("mutation", ["label", "parent", "profile", "source_order",
                                      "view_missing", "split", "scoring", "seed"])
def test_repinned_metadata_tampering_rejected(pair, mutation):
    path = pair[0] / "bundle.json"
    bundle = json.loads(path.read_text())
    view = bundle["arms"]["control"][0]
    if mutation == "label":
        view["episode"]["actions"][0]["keys"] = ["left"]
    elif mutation == "parent":
        view["parent_slot"] = 1
    elif mutation == "profile":
        view["episode"]["presentation"]["textures"] = True
    elif mutation == "source_order":
        bundle["sources"].reverse()
    elif mutation == "view_missing":
        bundle["arms"]["control"].pop()
    elif mutation == "split":
        bundle["sources"][0]["split"] = "validation"
    elif mutation == "scoring":
        bundle["scoring_version"] = "legacy.v0"
    else:
        bundle["seed"] += 1
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError):
        load(pair, expected_sha256=repin(pair))


def test_pin_config_budget_and_changed_source(pair):
    reuse = load(pair)
    with pytest.raises(ValueError, match="SHA256"):
        load(pair, expected_sha256="a" * 64)
    with pytest.raises(ValueError):
        load(pair, max_frame_bytes=1)
    with pytest.raises(ValueError, match="configuration"):
        load(pair, config=replace(pair[1], episode_instructions={"navigation": ("Another goal",),
                                                        "reaction": ("Catch the ball",)}))
    # A training consumer may inherit the recorded full difficulty after importing.
    assert load(pair, config=replace(pair[1], curriculum_difficulty=0.0)).summary[
        "source_difficulty"]["curriculum_difficulty"] == 1.0
    path = pair[0] / "bundle.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        reuse.assert_unchanged()
    with pytest.raises(ValueError, match="SHA256"):
        load(pair)


def test_repinned_final_view_pixel_tampering_rejected(pair):
    directory = pair[0]
    bundle = json.loads((directory / "bundle.json").read_text())
    index = bundle["arms"]["theme_shift"][0]["episode"]["final_frame_index"]
    with np.load(directory / "frames.npz", allow_pickle=False) as archive:
        frames = [archive[f"frame_{slot:06}"] for slot in range(bundle["unique_frames"])]
    frames[index][0, 0, 0] ^= 1
    with ZipFile(directory / "frames.npz", "w", compression=ZIP_DEFLATED) as archive:
        for slot, frame in enumerate(frames):
            with archive.open(f"frame_{slot:06}.npy", "w") as stream:
                np.lib.format.write_array(stream, frame, allow_pickle=False)
    with pytest.raises(ValueError, match="RGB differs"):
        load(pair, expected_sha256=repin(pair))


def test_symlink_rejected(pair, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(pair[0], target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        PreparedAppearanceReuse.prepare(alias, config=pair[1], arm="control",
            expected_sha256=pair[3]["source_bundle_sha256"])


def test_preparation_never_loads_model_optimizer_or_checkpoint(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("RGB preparation attempted model, weights or optimizer allocation")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    from carryrl.agent import model

    monkeypatch.setattr(model, "GeneralistAgent", forbidden)
    config = SequenceRunConfig(tasks=("navigation", "reaction"),
        curriculum_difficulty=1.0, evaluation_difficulty=1.0)
    dataset = collect_scripted_demonstrations(1, tasks=config.tasks, seed=8288)
    report = prepare_appearance_bundle(dataset, tmp_path / "data", config=config)
    assert report["updates"] == 0
    assert not list((tmp_path / "data").glob("*.pt"))


def test_cli_uses_effective_full_difficulty_and_all_tasks(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("appearance_prepare_cli",
        Path(__file__).resolve().parents[1] / "scripts/prepare_appearance.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    config = SequenceRunConfig(curriculum_difficulty=0.0)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(config.to_dict()))
    observed = {}
    sentinel = object()

    def collect(count, **kwargs):
        observed.update(count=count, **kwargs)
        return sentinel

    def prepare(dataset, output, **kwargs):
        assert dataset is sentinel
        assert kwargs["config"].curriculum_difficulty == 1.0
        assert kwargs["config"].evaluation_difficulty == 1.0
        return {"source_bundle_sha256": "b" * 64}

    monkeypatch.setattr(script, "collect_scripted_demonstrations", collect)
    monkeypatch.setattr(script, "prepare_appearance_bundle", prepare)
    assert script.main(["--config", str(path), "--output", str(tmp_path / "pair")]) == 0
    assert observed["count"] == 1 and observed["difficulty"] == 1.0
    assert len(observed["tasks"]) == 16 and observed["seed"] == 8288
    assert path.read_text() == json.dumps(config.to_dict())


def test_actual_correction_history_remains_separate_from_teacher_labels(tmp_path):
    class Learner:
        training = False

        def eval(self):
            return self

        def train(self, mode=True):
            self.training = mode

        def reset_episode(self, **kwargs):
            pass

        def act(self, frame, deterministic):
            return SimpleNamespace(action=PhysicalAction(keys=("left",)))

        def observe_action(self, action):
            pass

    config = SequenceRunConfig(tasks=("navigation",),
        curriculum_difficulty=1.0, evaluation_difficulty=1.0)
    data = collect_scripted_corrections(Learner(), 1, tasks=config.tasks, seed=8288,
                                        expert_probability=0.0)
    assert data.episodes[0].executed_actions != data.episodes[0].actions
    report = prepare_appearance_bundle(data, tmp_path / "corrections", config=config)
    loaded = PreparedAppearanceReuse.prepare(tmp_path / "corrections",
        expected_sha256=report["source_bundle_sha256"], arm="theme_shift", config=config)
    raw, view = loaded.dataset.episodes
    assert raw.actions == view.actions == data.episodes[0].actions
    assert raw.executed_actions == view.executed_actions == data.episodes[0].executed_actions
    assert raw.terminated == view.terminated
    assert raw.truncated == view.truncated
    assert raw.quality == view.quality


def test_duplicate_source_rows_do_not_claim_distinct_physical_origins(tmp_path):
    from carryrl.training.diversity import audit_diversity

    config = SequenceRunConfig(tasks=("navigation",),
        curriculum_difficulty=1.0, evaluation_difficulty=1.0)
    data = collect_scripted_demonstrations(1, tasks=config.tasks, seed=8288)
    data.add_episode(data.episodes[0])
    report = prepare_appearance_bundle(data, tmp_path / "duplicates", config=config)
    for arm in ARMS:
        reuse = PreparedAppearanceReuse.prepare(tmp_path / "duplicates",
            expected_sha256=report["source_bundle_sha256"], arm=arm, config=config)
        assert reuse.summary["source_rows"] == 2
        assert reuse.summary["posthoc_copies"] == 2
        assert reuse.summary["episodes"] == 4
        assert "physical_origins" not in reuse.summary
        audit = audit_diversity(reuse.dataset)
        assert audit["counts"]["metadata_physical_origins"] == 1
