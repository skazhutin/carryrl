"""Stage readiness covers admitted mechanics and presentation without hidden labels."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.curriculum import STAGES
from carryrl.envs.presentation import THEME_SPLITS
from carryrl.evaluation import fresh_reward as fresh
from carryrl.evaluation import stage_reward as stage
from carryrl.rewards.composition import OUTCOME_POTENTIAL_VERSION, RewardCompositionConfig
from carryrl.rewards.pretrain import RewardGate
from carryrl.rewards.visual import draw_visible_hud, verify_visible_outcome


class RGBReward:
    def predict(self, frame):
        observed = verify_visible_outcome(frame)
        if not observed["known"]:
            return {"known": False, "score": 0, "score_confidence": 0,
                    "success_probability": 0.5, "failure_probability": 0.5}
        return {"known": observed["known"], "score": observed["score"],
                "score_confidence": float(observed["known"]),
                "success_probability": float(observed["success"]),
                "failure_probability": float(observed["failure"])}


class Teacher:
    def act(self, frame):
        return PhysicalAction(keys=("f",))


class StageEnv:
    constructions = []
    resets = []

    def __init__(self, task, **options):
        self.options = options
        self.constructions.append((task, options))
        self.tick = self.progress = 0

    def frame(self, success=False):
        frame = np.full((96, 96, 3), 71, dtype=np.uint8)
        draw_visible_hud(frame, self.progress, "Success" if success else "Play")
        return frame

    def reset(self, *, seed, options):
        self.resets.append((self.options["split"], self.options.get("curriculum_stage"),
                            options["level_index"]))
        return self.frame(), {"level_id": str(options["level_index"]),
                              "level_bank_seed": stage.BANK}

    def step(self, action):
        self.tick += 1
        self.progress += int(action.keys == ("f",))
        success = self.progress == 2
        return self.frame(success), float("nan"), success, (
            not success and self.tick == 80), {"success": "poison hidden info"}

    def close(self):
        pass


def composition():
    return RewardCompositionConfig(version=OUTCOME_POTENTIAL_VERSION,
                                   parameters={"goal_choice_utility": "completion_only"})


def admission():
    return {"tasks": {task: {name: {"allowed_level_indices": [9, 7, 4, 11]}
                             for name in STAGES} for task in stage.TASKS}}


def diagnose(reward=None, *, env_factory=StageEnv, tasks=("navigation",), levels=1):
    return stage._validate_cells(
        reward or RGBReward(), admission(), tasks=tasks, config=composition(),
        gamma=0.99, horizon=80, levels=levels, seed=100,
        env_factory=env_factory, teacher_factory=lambda task: Teacher())


def test_stage_core_uses_admitted_train_indices_and_split_styles_only():
    StageEnv.constructions.clear()
    StageEnv.resets.clear()
    result = diagnose()
    assert result["hud_passed"] and result["progress_passed"], result["failure_examples"]
    assert len(result["coverage"]) == 4
    assert {options["level_bank_seed"] for _, options in StageEnv.constructions} == {stage.BANK}
    assert {split for split, _, _ in StageEnv.resets} == {"train", "validation"}
    assert all(index == (9 if split == "train" else 0)
               for split, _, index in StageEnv.resets)
    for cell in result["coverage"].values():
        assert cell["passed"] and cell["successes"] == 1 and cell["nonwins"] == 2
        assert cell["styles"] == ["canonical", *stage.PIXEL_STYLE_SPLITS[cell["split"]]]


def test_missing_stage_success_fails_even_with_perfect_rgb_readout():
    class NoEasySuccess(StageEnv):
        def step(self, action):
            if self.options.get("curriculum_stage") == "easy":
                action = PhysicalAction()
            return super().step(action)

    result = diagnose(env_factory=NoEasySuccess)
    assert result["hud_passed"] and not result["progress_passed"]
    assert not result["coverage"]["navigation/train/easy"]["passed"]
    assert result["coverage"]["navigation/train/full"]["passed"]


def test_random_wins_all_four_levels_with_neutral_nonwin_passes(monkeypatch):
    monkeypatch.setattr(fresh, "_random_action", lambda rng: PhysicalAction(keys=("f",)))
    result = diagnose(levels=4)
    assert result["hud_passed"] and result["progress_passed"], result["failure_examples"]
    assert result["failure_counts"] == {"hud": 0, "progress": 0}
    for cell in result["coverage"].values():
        random = [row for row in cell["episodes"] if row["strategy"] == "random"]
        assert len(random) == 4 and all(row["success"] for row in random)
        assert any(row["strategy"] == "neutral" and row["completed"] and not row["success"]
                   for row in cell["episodes"])
        assert cell["passed"] and all(cell["requirements"].values())
        assert set(cell["requirements"]) == {
            "observed_success", "observed_nonwin", "random_trajectory", "all_episodes_completed"}
    assert stage.FORMAT == "carryrl.fresh_stage_reward_validation.v2"


def test_missing_actual_nonwin_fails_even_with_successful_random_trajectory():
    class EveryStrategyWins(StageEnv):
        def step(self, action):
            return super().step(PhysicalAction(keys=("f",)))

    result = diagnose(env_factory=EveryStrategyWins)
    assert result["hud_passed"] and not result["progress_passed"]
    for cell in result["coverage"].values():
        assert not cell["passed"] and cell["nonwins"] == 0
        assert not cell["requirements"]["observed_nonwin"]
        assert cell["requirements"]["random_trajectory"]
    assert result["failure_counts"] == {"hud": 0, "progress": 4}


def test_single_hit_easy_need_not_fabricate_nonterminal_intermediate_score():
    class SingleHitEasy(StageEnv):
        def step(self, action):
            if self.options.get("curriculum_stage") == "easy" and action.keys == ("f",):
                self.progress = 1
            return super().step(action)

    result = diagnose(env_factory=SingleHitEasy)
    assert result["hud_passed"] and result["progress_passed"], result["failure_examples"]
    assert result["coverage"]["navigation/train/easy"]["intermediate_scores"] == []
    assert result["coverage"]["navigation/train/full"]["intermediate_scores"] == [1.0]


def test_empty_tasks_never_certify_progress():
    result = diagnose(tasks=())
    assert not result["progress_passed"] and result["coverage"] == {}


def test_fresh_fit_recipe_does_not_open_test_themes_or_change_gates():
    config = stage.fresh_stage_reward_config(seed=123)
    assert config["tasks"] == list(stage.TASKS) and config["level_bank_seed"] == 20261020
    assert (config["episodes_per_task"], config["validation_episodes_per_task"],
            config["epochs"]) == (8, 2, 40)
    presentation = config["presentation_config"]
    assert presentation["mode"] == "varied" and presentation["pixel_art"] is False
    assert presentation["themes"] == THEME_SPLITS["train"] + THEME_SPLITS["validation"]
    assert not set(presentation["themes"]) & set(THEME_SPLITS["test"])
    assert config["gate"] == asdict(RewardGate())


def test_public_gate_rejects_empty_coverage_and_pins_admission(tmp_path, monkeypatch):
    path = tmp_path / "admission.json"
    path.write_text("pinned")
    digest = stage._sha(path)
    calls = []

    def validate(path, **kwargs):
        calls.append(kwargs)
        return admission()

    monkeypatch.setattr(stage, "validate_task_curriculum_admission", validate)
    monkeypatch.setattr(stage, "_validate_cells", lambda *args, **kwargs: {
        "hud_passed": True, "progress_passed": True, "coverage": {},
        "checks": {"hud": 1, "progress": 1}, "failure_counts": {"hud": 0, "progress": 0},
        "failure_examples": []})
    result = stage.validate_stage_reward(RGBReward(), admission_path=path,
                                         admission_sha256=digest, config=composition())
    assert not result["passed"] and result["gate"] == {"passed": False}
    assert not result["coverage_passed"] and result["covered_cell_count"] == 0
    assert result["required_cells"] == 240 and result["admission_sha256"] == digest
    assert calls[0]["expected_sha256"] == digest
    assert result["test_levels_used"] is False and result["test_themes_used"] is False


@pytest.mark.parametrize("change", [None, "missing_stage", "missing_style", "empty_episodes"])
def test_public_gate_requires_all_240_cells_and_completed_evidence(tmp_path, monkeypatch, change):
    path = tmp_path / "admission.json"
    path.write_text("pinned")
    coverage = {f"{task}/{split}/{name}": {
        "styles": ["canonical", *stage.PIXEL_STYLE_SPLITS[split]],
        "passed": True, "episodes": [{"completed": True}]}
        for task in stage.TASKS for split, stages in (("train", STAGES),
                                                     ("validation", ("full",)))
        for name in stages}
    key = next(iter(coverage))
    if change == "missing_stage":
        del coverage[key]
    elif change == "missing_style":
        coverage[key]["styles"].pop()
    elif change == "empty_episodes":
        coverage[key]["episodes"] = []
    monkeypatch.setattr(stage, "validate_task_curriculum_admission", lambda *args, **kwargs: {})
    monkeypatch.setattr(stage, "_validate_cells", lambda *args, **kwargs: {
        "hud_passed": True, "progress_passed": True, "coverage": coverage})
    result = stage.validate_stage_reward(RGBReward(), admission_path=path,
                                         admission_sha256=stage._sha(path), config=composition())
    assert result["passed"] is (change is None)
    assert result["coverage_passed"] is (change is None)
    assert result["required_cells"] == 240
    if change is None:
        assert result["covered_cell_count"] == 240


@pytest.mark.parametrize("kwargs", [{"levels": 0}, {"levels": 5}, {"horizon": 79},
                                    {"level_bank_seed": 20261008}])
def test_protocol_cannot_silently_change_bank_or_bounds(kwargs):
    with pytest.raises(ValueError, match="protocol requires"):
        stage.validate_stage_reward(RGBReward(), admission_path="missing",
                                     admission_sha256="a" * 64, config=composition(), **kwargs)


@pytest.fixture
def checkpoint_bundle(tmp_path, monkeypatch):
    recipe = stage.fresh_stage_reward_config(seed=123)
    policy = {"tasks": list(stage.TASKS), "env_image_size": 96, "level_bank_seed": stage.BANK,
              "reward_composition": composition().to_dict(), "gamma": 0.99,
              "max_episode_steps": 80}
    training_report = {"initialization": "random", "imported_checkpoints": [], "old_corpora": [],
                       "test_levels_used": False, "policy_demonstrations": False,
                       "tasks": list(stage.TASKS), "level_bank_seed": stage.BANK, "seed": 123,
                       "presentation_config": recipe["presentation_config"],
                       "gate": {"passed": True, "tolerances": asdict(RewardGate())}}
    for name, content in (("reward_config.json", recipe), ("policy_config.json", policy),
                          ("reward_report.json", training_report)):
        (tmp_path / name).write_text(json.dumps(content))
    (tmp_path / "reward_checkpoint.pt").write_bytes(b"loader injected; no fitting")
    monkeypatch.setattr(stage, "load_screen_reward_checkpoint",
                        lambda *args, **kwargs: (RGBReward(), training_report))
    monkeypatch.setattr(stage, "validate_stage_reward", lambda *args, **kwargs: {
        "passed": True, "gate": {"passed": True}, "covered_cell_count": 240})
    return tmp_path, training_report


def validate_checkpoint(directory):
    return stage.validate_stage_reward_checkpoint(
        directory / "reward_checkpoint.pt", admission_path=directory / "admission.json",
        admission_sha256="a" * 64, policy_config=directory / "policy_config.json",
        reward_config=directory / "reward_config.json", output=directory / "validation.json")


def test_checkpoint_report_binds_all_inputs_and_refuses_overwrite(checkpoint_bundle):
    directory, _ = checkpoint_bundle
    result = validate_checkpoint(directory)
    for name, filename in (("checkpoint", "reward_checkpoint.pt"),
                           ("reward_report", "reward_report.json"),
                           ("policy_config", "policy_config.json"),
                           ("reward_config", "reward_config.json")):
        assert result[f"{name}_sha256"] == stage._sha(directory / filename)
    assert json.loads((directory / "validation.json").read_text()) == result
    with pytest.raises(FileExistsError, match="already exists"):
        validate_checkpoint(directory)


@pytest.mark.parametrize("field,value", [("test_levels_used", True),
                                         ("imported_checkpoints", ["old.pt"]),
                                         ("level_bank_seed", 20261008),
                                         ("initialization", "imported")])
def test_checkpoint_refuses_unfresh_report(checkpoint_bundle, field, value):
    directory, training_report = checkpoint_bundle
    training_report[field] = value
    (directory / "reward_report.json").write_text(json.dumps(training_report))
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_checkpoint(directory)
    assert not (directory / "validation.json").exists()


def test_checkpoint_refuses_relaxed_embedded_numerical_gate(checkpoint_bundle):
    directory, training_report = checkpoint_bundle
    training_report["gate"]["tolerances"]["max_score_mae"] = 1.0
    (directory / "reward_report.json").write_text(json.dumps(training_report))
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_checkpoint(directory)


def test_checkpoint_refuses_inputs_changed_during_diagnostics(checkpoint_bundle, monkeypatch):
    directory, _ = checkpoint_bundle

    def mutate(*args, **kwargs):
        (directory / "policy_config.json").write_text("changed")
        return {"passed": True}

    monkeypatch.setattr(stage, "validate_stage_reward", mutate)
    with pytest.raises(ValueError, match="changed during validation"):
        validate_checkpoint(directory)
    assert not (directory / "validation.json").exists()
