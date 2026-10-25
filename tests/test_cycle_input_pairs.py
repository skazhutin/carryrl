import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import atomic_level_banks, atomic_semantic_level_id
from carryrl.evaluation.cycle_input_pairs import _memory_index, prepare_cycle_input_pairs
from carryrl.evaluation.input_sensitivity import KINDS, MAX_BUNDLE_BYTES, load_sensitivity_bundle

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = ROOT / "configs/atomic_instructions.json"


@pytest.fixture
def bundle(tmp_path):
    return prepare_cycle_input_pairs(tmp_path / "pairs", instructions=INSTRUCTIONS)


def test_all_four_kinds_roundtrip_bounded_with_source_hashes(bundle):
    pairs, identity = load_sensitivity_bundle(bundle)
    manifest = json.loads(bundle.read_text())
    assert len(pairs) == 4 and {pair.kind for pair in pairs} == set(KINDS)
    assert manifest["retained_rgb_bytes"] < MAX_BUNDLE_BYTES
    assert manifest["image_size"] == 96 and manifest["training_updates"] == 0
    assert manifest["test_used"] is False
    assert manifest["instructions_sha256"] == hashlib.sha256(INSTRUCTIONS.read_bytes()).hexdigest()
    assert len(identity["frames_sha256"]) == 64
    for pair in pairs:
        assert pair.original_frames.shape[1:] == (96, 96, 3)
        assert len(pair.provenance["original_executed_sha256"]) == 64
        assert len(pair.provenance["original_frames_sha256"]) == 64
        assert pair.provenance["candidate_actions_are_likelihood_labels"]
        assert pair.provenance["split"] == "validation"
    assert sorted(path.name for path in bundle.parent.iterdir()) == ["frames.npz", "pairs.json"]


def test_actual_memory_cue_crosses_segment64_without_current_frame_or_history_leak(bundle):
    pairs, _ = load_sensitivity_bundle(bundle)
    cue = next(pair for pair in pairs if pair.kind == "memory_cue")
    assert cue.probe_step >= 66 and len(cue.original_frames) == cue.probe_step + 1
    assert not np.array_equal(cue.original_frames[0], cue.perturbed_frames[0])
    np.testing.assert_array_equal(
        cue.original_frames[cue.probe_step], cue.perturbed_frames[cue.probe_step]
    )
    assert cue.original_executed == cue.perturbed_executed
    assert set(cue.original_executed) == {PhysicalAction()}
    origins = cue.provenance["scene_origins"]
    assert {origin["cue"] for origin in origins} == {0, 1}
    assert {origin["probe_phase"] for origin in origins} == {"Choose"}
    assert origins[0]["bank_level_id"] == origins[1]["bank_level_id"]


def test_executed_history_pair_has_real_opposite_blocked_inputs_and_identical_pixels(bundle):
    pairs, _ = load_sensitivity_bundle(bundle)
    history = next(pair for pair in pairs if pair.kind == "executed_history")
    assert history.probe_step == 64
    np.testing.assert_array_equal(history.original_frames, history.perturbed_frames)
    assert PhysicalAction(keys=("left",)) in history.original_executed[:64]
    assert PhysicalAction(keys=("right",)) in history.perturbed_executed[:64]
    assert history.original_executed[-1] == history.perturbed_executed[-1] == PhysicalAction()
    assert {origin["probe_phase"] for origin in history.provenance["scene_origins"]} == {"Wait"}
    assert all(
        origin["actual_env_step_calls"] == 65 for origin in history.provenance["scene_origins"]
    )


def test_text_pair_marks_counterfactual_as_intervention_and_preserves_executed_inputs(bundle):
    pairs, _ = load_sensitivity_bundle(bundle)
    text = next(pair for pair in pairs if pair.kind == "text_control")
    np.testing.assert_array_equal(text.original_frames, text.perturbed_frames)
    assert text.original_executed == text.perturbed_executed == (PhysicalAction(),)
    assert text.original_instruction != text.perturbed_instruction
    assert text.provenance["diagnostic_intervention"]
    assert text.provenance["mechanics_match_counterfactual_text"] is False


def test_memory_selection_excludes_both_train_cues_and_preparation_preserves_cached_bank(tmp_path):
    bank = atomic_level_banks("memory", 20261020)
    before = copy.deepcopy(bank)
    index = _memory_index(20261020)
    train_ids = {atomic_semantic_level_id("memory", level) for level in bank["train"]}
    assert all(
        atomic_semantic_level_id("memory", {**bank["validation"][index], "cue": cue})
        not in train_ids
        for cue in (0, 1)
    )
    prepare_cycle_input_pairs(tmp_path / "pairs", instructions=INSTRUCTIONS)
    assert bank == before
    with pytest.raises(FileExistsError):
        prepare_cycle_input_pairs(tmp_path / "pairs", instructions=INSTRUCTIONS)


def test_preparation_cli_has_no_test_or_training_options(capsys):
    spec = importlib.util.spec_from_file_location(
        "cycle_pairs_cli", ROOT / "scripts/prepare_cycle_input_pairs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as result:
        module.main(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "--level-bank-seed" in help_text and "--split" not in help_text
