"""Appearance interventions keep mechanics, layout and instruction paired."""

from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from test_robustness_matrix import checkpoint, config, matrix


def _reports():
    reports = {}
    for spec, cfg in matrix.build_matrix(config(), episodes_per_task=1, matrix_kind="appearance96"):
        records = [{"task": task, "seed": 1000000, "instruction": cfg.episode_instructions[task][0],
                    "policy": "frozen_deterministic", "action_seed": None, "episode_index": 0,
                    "difficulty": 1.0, "level_index": 0, "level_id": task + "/level0",
                    "semantic_level_id": task, "runtime_fingerprint": task + "/runtime",
                    "level_bank_seed": cfg.level_bank_seed,
                    "presentation": asdict(cfg.visual_config),
                    "initial_frame_sha256": spec["id"],
                    "outcome": {"known": True, "success": spec["profile"] !=
                                "theme96_sand_left_plain"}}
                   for task in cfg.tasks]
        reports[spec["id"]] = {"records": records, "protocol_sha256": spec["protocol_sha256"]}
    return reports


def test_native96_strata_fix_axes_and_reserve_test():
    original = config()
    before = deepcopy(original.to_dict())
    strata = matrix.build_matrix(original, matrix_kind="appearance96")
    assert len(strata) == 8 and original.to_dict() == before
    profiles = matrix.appearance_profiles()
    seen, shift, heldout = [profiles[f"theme96_{theme}_left_plain"]
                            for theme in ("slate", "sand", "violet")]
    for candidate in (shift, heldout):
        differences = {key for key in asdict(seen) if asdict(seen)[key] != asdict(candidate)[key]}
        assert differences <= {"themes", "style_split"}
        assert candidate.native_sizes == (96,) and not candidate.textures
        assert candidate.hud_layouts == ("left",) and candidate.canvas_padding == 8
    for spec, cfg in strata:
        assert cfg.env_image_size == 96 and cfg.visual_config.native_sizes == (96,)
        assert not cfg.eval_stochastic and len(cfg.episode_instructions["mouse"]) == 1
    with pytest.raises(ValueError, match="reserved"):
        matrix.build_matrix(original, split="test", matrix_kind="appearance96")


def test_theme_tables_condition_on_changed_pixels_but_keep_full_mechanical_identity():
    reports = _reports()
    result = matrix._theme_contrasts(reports)
    pair = result["slate_to_sand_ru"]
    assert pair["outcomes"]["lost"] == 2 and pair["outcomes"]["known_pairs"] == 2
    assert pair["varied_descriptor_fields"] == ["presentation", "initial_frame_sha256"]
    assert pair["left_protocol_sha256"] != pair["right_protocol_sha256"]
    assert result["slate_to_violet_en"]["outcomes"]["retained_wins"] == 2
    assert reports["theme96_sand_left_plain_ru"]["records"][0]["presentation"]["themes"] == (
        "sand",)


@pytest.mark.parametrize("field", ["instruction", "runtime_fingerprint", "seed", "level_id",
                                    "level_bank_seed"])
def test_contrast_rejects_mechanics_instruction_or_cohort_change(field):
    reports = _reports()
    reports["theme96_sand_left_plain_ru"]["records"][0][field] = "changed"
    with pytest.raises(ValueError, match="mechanics/instruction/cohort"):
        matrix._theme_contrasts(reports)


def test_unknown_outcomes_are_not_losses():
    reports = _reports()
    reports["theme96_sand_left_plain_ru"]["records"][0]["outcome"] = {"known": False}
    pair = matrix._theme_contrasts(reports)["slate_to_sand_ru"]["outcomes"]
    assert pair["lost"] == 1 and pair["unknown_pairs"] == 1 and pair["coverage"] == 0.5


def test_new_matrix_mode_loads_no_optimizer_and_records_declared_contrasts(tmp_path, monkeypatch):
    reports = _reports()
    actor, reward = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    loads = []

    def load(*args, **kwargs):
        assert kwargs == {"evaluation_only": True, "load_optimizer": False}
        loads.append(1)
        return actor, None, reward, {"weights_revision": 3}

    def evaluate(*args, **kwargs):
        cfg = args[2]
        protocol = matrix.sequence_evaluation_protocol(cfg, split="validation")
        pin = matrix._hash(protocol)
        result = deepcopy(next(value for value in reports.values()
                               if value["protocol_sha256"] == pin))
        frozen = result["records"]
        result["records"] = [*frozen, *[{**row, "policy": "random_physical"}
                                       for row in frozen]]
        result.update(policy_and_reward_frozen=True, summaries={})
        return result

    monkeypatch.setattr(matrix, "load_sequence_checkpoint", load)
    monkeypatch.setattr(matrix, "evaluate_sequence", evaluate)
    result = matrix.run_matrix(checkpoint(tmp_path), config(), tmp_path / "result",
                               episodes_per_task=1, matrix_kind="appearance96")
    assert loads == [1] and len(result["theme_contrasts"]) == 4
    assert result["matrix_kind"] == "appearance96" and result["status"] == "completed"
