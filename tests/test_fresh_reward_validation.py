"""Fresh reward diagnostics report evidence and reject missing coverage/provenance."""

import json
from pathlib import Path

import numpy as np
import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.evaluation import fresh_reward as fresh
from carryrl.rewards.composition import OUTCOME_POTENTIAL_VERSION, RewardCompositionConfig
from carryrl.rewards.visual import draw_visible_hud, verify_visible_outcome
from carryrl.training.token_comparison import prepare_comparison

ROOT = Path(__file__).resolve().parents[1]


class TinyReward:
    """Exact RGB oracle used solely to test diagnostic accounting without fitting."""

    def predict(self, frame):
        observed = verify_visible_outcome(frame)
        if not observed["known"]:
            return {"known": False, "score": 0, "score_confidence": 0,
                    "success_probability": 0.5, "failure_probability": 0.5}
        return {"known": True, "score": observed["score"], "score_confidence": 1,
                "success_probability": float(observed["success"]),
                "failure_probability": float(observed["failure"])}


class TinyTeacher:
    def act(self, frame):
        return PhysicalAction(keys=("f",))


class TinyEnv:
    """Only the diagnostic teacher's distinctive action advances this fixture."""

    def __init__(self, task, **options):
        assert options["split"] == "validation" and options["level_bank_seed"] == fresh.BANK
        self.limit = options["max_steps"]
        self.tick = self.progress = 0

    def _frame(self, status="Play"):
        frame = np.full((96, 96, 3), 71, dtype=np.uint8)
        draw_visible_hud(frame, self.progress, status)
        return frame

    def reset(self, *, seed, options):
        return self._frame(), {"level_id": str(options["level_index"]),
                               "level_bank_seed": fresh.BANK}

    def step(self, action):
        self.tick += 1
        self.progress += int(action.keys == ("f",))
        success = self.progress == 2
        return self._frame("Success" if success else "Play"), float("nan"), success, (
            not success and self.tick == self.limit), {"success": "poison hidden diagnostics"}

    def close(self):
        pass


def config():
    return RewardCompositionConfig(version=OUTCOME_POTENTIAL_VERSION,
                                   parameters={"goal_choice_utility": "completion_only"})


def diagnose(reward, **kwargs):
    return fresh._validate_reward(reward, tasks=("navigation", "memory"), config=config(),
                                  horizon=3, levels=1, env_factory=TinyEnv,
                                  teacher_factory=lambda task: TinyTeacher(), **kwargs)


def test_fresh_rgb_core_observes_success_nonwins_intermediate_and_composition():
    report = diagnose(TinyReward())
    assert report["hud_passed"] and report["progress_passed"]
    assert report["failure_counts"] == {"hud": 0, "progress": 0}
    for task in ("navigation", "memory"):
        coverage = report["coverage"][task]
        assert coverage["passed"] and coverage["intermediate_scores"] == [1.0]
        assert coverage["successes"] == 1 and coverage["nonwins"] == 2
        assert {episode["strategy"] for episode in coverage["episodes"]} == {
            "diagnostic_teacher", "neutral", "random"}


def test_missing_success_is_coverage_failure_even_when_hud_is_perfect():
    class NoSuccess(TinyEnv):
        def step(self, action):
            return super().step(PhysicalAction())

    report = fresh._validate_reward(TinyReward(), tasks=("memory",), config=config(),
                                   horizon=3, levels=1, env_factory=NoSuccess,
                                   teacher_factory=lambda task: TinyTeacher())
    assert report["hud_passed"] and not report["progress_passed"]
    coverage = report["coverage"]["memory"]
    assert not coverage["requirements"]["observed_success"]
    assert not coverage["requirements"]["observed_intermediate"]


def test_wrong_scores_false_events_and_unknown_terminal_are_rejected():
    class BadReward(TinyReward):
        def predict(self, frame):
            prediction = super().predict(frame)
            prediction.update(known=True, score=5, score_confidence=0.6,
                              success_probability=1.0)
            return prediction

    report = diagnose(BadReward())
    assert not report["hud_passed"] and not report["progress_passed"]
    assert report["failure_counts"]["hud"] > 0
    assert any("score confidence" in row["reason"] for row in report["failure_examples"])


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "configs").mkdir(parents=True)
    for name in ("agent_v1_learning.json", "atomic_instructions.json"):
        (root / "configs" / name).write_bytes((ROOT / "configs" / name).read_bytes())
    (root / "pyproject.toml").write_text('[project]\nname="fixture"\n')
    (root / "uv.lock").write_text("fixture lock")
    (root / "src/carryrl").mkdir(parents=True)
    (root / "src/carryrl/fixture.py").write_text("# pinned source\n")
    output = root / "runs/new-comparison"
    manifest = prepare_comparison(root, output)
    reward_config = json.loads(Path(manifest["reward_config_pin"]["path"]).read_text())
    directory = output / "reward"
    directory.mkdir()
    (directory / "reward_checkpoint.pt").write_bytes(b"tiny reward; loader injected")
    training_report = {"initialization": "random", "imported_checkpoints": [], "old_corpora": [],
                       "level_bank_seed": fresh.BANK, "seed": reward_config["seed"],
                       "tasks": reward_config["tasks"], "test_levels_used": False,
                       "policy_demonstrations": False, "gate": {"passed": True}}
    (directory / "reward_report.json").write_text(json.dumps(training_report))
    calls = []

    def loader(path, *, require_pass):
        calls.append((path, require_pass))
        return TinyReward(), training_report

    monkeypatch.setattr(fresh, "load_screen_reward_checkpoint", loader)
    return root, output, training_report, calls


def test_public_validator_requires_all_tasks_and_writes_pinned_report_once(bundle, monkeypatch):
    _, output, _, calls = bundle
    real = fresh._validate_reward

    def core(reward, **kwargs):
        assert kwargs["tasks"] == fresh.TASKS and kwargs["horizon"] == 80
        return real(reward, tasks=fresh.TASKS, config=kwargs["config"], horizon=3, levels=1,
                    env_factory=TinyEnv, teacher_factory=lambda task: TinyTeacher())

    monkeypatch.setattr(fresh, "_validate_reward", core)
    report = fresh.validate_fresh_reward(output)
    assert report["passed"] and report["tasks"] == list(fresh.TASKS)
    assert report["checkpoint_sha256"] == fresh._sha(output / "reward/reward_checkpoint.pt")
    assert report["recipe_sha256"] == fresh._sha(output / "reward_recipe.json")
    assert calls == [(output / "reward/reward_checkpoint.pt", True)]
    assert not report["policy_evaluated"] and report["policy_updates"] == 0
    assert report["styles"] == ["canonical", "frost", "circuit"]
    assert json.loads((output / "reward/validation_report.json").read_text()) == report
    with pytest.raises(FileExistsError, match="already exists"):
        fresh.validate_fresh_reward(output)


@pytest.mark.parametrize("field,value", [("level_bank_seed", 0), ("test_levels_used", True),
                                         ("initialization", "imported"),
                                         ("imported_checkpoints", ["old.pt"]),
                                         ("old_corpora", ["old-data"])])
def test_public_validator_rejects_unfresh_training_provenance(bundle, field, value):
    _, output, report, _ = bundle
    report[field] = value
    (output / "reward/reward_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="provenance mismatch"):
        fresh.validate_fresh_reward(output)
    assert not (output / "reward/validation_report.json").exists()


def test_public_validator_rejects_source_and_input_hash_changes(bundle):
    root, output, _, calls = bundle
    (root / "configs/atomic_instructions.json").write_text("{}")
    with pytest.raises(ValueError, match="pinned file changed"):
        fresh.validate_fresh_reward(output)
    assert calls == []


def test_cli_plan_is_directory_and_bounds_are_explicit(bundle):
    _, output, _, _ = bundle
    with pytest.raises(ValueError, match="bundle directory"):
        fresh.validate_fresh_reward(output / "manifest.json")
    with pytest.raises(ValueError, match="levels"):
        fresh.validate_fresh_reward(output, levels=5)


def test_real_atomic_and_teacher_apis_on_fresh_navigation_memory_bank():
    report = fresh._validate_reward(TinyReward(), tasks=("navigation", "memory"),
                                    config=config(), horizon=80, levels=1)
    assert report["hud_passed"] and report["progress_passed"], report["failure_examples"]
    assert report["coverage"]["memory"]["intermediate_scores"] == [1.0]


@pytest.mark.parametrize("changed", [False, True])
def test_embedded_report_matches_nested_tuple_json_and_rejects_real_change(
    bundle, monkeypatch, changed,
):
    _, output, report, _ = bundle
    report["presentation_config"] = {"native_sizes": [64, 96, 128],
                                     "nested": {"themes": ["frost", "circuit"]}}
    (output / "reward/reward_report.json").write_text(json.dumps(report))
    embedded = {**report, "presentation_config": {"native_sizes": (64, 96, 128),
                "nested": {"themes": ("frost", "different" if changed else "circuit")}}}
    expected_threads = json.loads((output / "configs/fresh_reward.json").read_text())[
        "torch_threads"]
    thread_calls = []
    monkeypatch.setattr(fresh.torch, "set_num_threads", thread_calls.append)

    def loader(path, *, require_pass):
        assert thread_calls == [expected_threads] and require_pass
        return TinyReward(), embedded

    monkeypatch.setattr(fresh, "load_screen_reward_checkpoint", loader)
    monkeypatch.setattr(fresh, "_validate_reward", lambda *args, **kwargs: {
        "hud_passed": True, "progress_passed": True,
        "coverage": {}, "checks": {"hud": 1, "progress": 1},
        "failure_counts": {"hud": 0, "progress": 0}, "failure_examples": []})
    if changed:
        with pytest.raises(ValueError, match="provenance mismatch"):
            fresh.validate_fresh_reward(output)
        assert not (output / "reward/validation_report.json").exists()
    else:
        result = fresh.validate_fresh_reward(output)
        assert result["passed"]
    assert thread_calls == [expected_threads]
