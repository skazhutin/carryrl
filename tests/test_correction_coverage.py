"""Coverage describes actual inputs, E edges, RGB milestones and explicit unknowns."""

from types import SimpleNamespace

import pytest

from carryrl.agent.actions import PhysicalAction
from carryrl.evaluation.correction_coverage import (
    describe_correction_coverage,
    summarize_correction_coverage,
)
from carryrl.evaluation.scene_suite import EVENT_NAMES
from carryrl.training.imitation import (
    collect_scripted_corrections,
    collect_scripted_demonstrations,
    scripted_teacher,
)


class FixedLearner:
    """Controlled fixture commands, never a trained policy."""

    training = False

    def __init__(self, actions):
        self.actions = actions

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False

    def reset_episode(self, **_):
        self.index = 0

    def act(self, frame, deterministic):
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return SimpleNamespace(action=action)

    def observe_action(self, action):
        pass


class InteractionLearner(FixedLearner):
    def __init__(self, *, skip_use):
        self.skip_use = skip_use

    def reset_episode(self, **_):
        self.teacher = scripted_teacher("interaction")
        self.picked = False

    def act(self, frame, deterministic):
        action = self.teacher.act(frame)
        if "e" in action.keys:
            if self.picked and self.skip_use:
                action = PhysicalAction()
            self.picked = True
        return SimpleNamespace(action=action)

    def observe_action(self, action):
        self.teacher.observe_action(action)


def correction(task="interaction", *, learner=None, max_steps=80):
    dataset = collect_scripted_corrections(learner or FixedLearner([PhysicalAction()]), 1,
        tasks=(task,), seed=8176, difficulty=1.0, max_steps=max_steps,
        level_bank_seed=20261015, expert_probability=0.0)
    return dataset, dataset.episodes[0]


def test_coverage_uses_actual_noops_instead_of_nonempty_teacher_targets():
    dataset, episode = correction()
    assert episode.actions != episode.executed_actions
    row = describe_correction_coverage(dataset, episode)
    assert row["commands"]["neutral"] == row["steps"] == 80
    assert row["commands"]["movement"] == 0
    assert row["viewport_rgb"]["unchanged_transitions"] == 80
    assert row["viewport_rgb"]["longest_run"] == 80
    assert row["viewport_rgb"]["first_window_step"] == 8
    assert row["outcome"]["success"] is False and row["truncated"]
    assert row["first_visible_failure_step"] is None


def test_e_edge_hold_release_counts_follow_executed_history():
    e, idle = PhysicalAction(keys=("e",)), PhysicalAction()
    dataset, episode = correction("camera", learner=FixedLearner([e, e, idle, e, e, idle]),
                                  max_steps=6)
    row = describe_correction_coverage(dataset, episode)
    assert {key: row["commands"][key] for key in ("e_down", "e_rising", "e_held", "e_release")} == {
        "e_down": 4, "e_rising": 2, "e_held": 2, "e_release": 2}


def test_visible_item_then_unused_use_is_candidate_not_an_invented_event():
    dataset, episode = correction(learner=InteractionLearner(skip_use=True))
    row = describe_correction_coverage(dataset, episode)
    assert row["visible_events"]["item_collected"]["count"] == 1
    assert row["visible_events"]["item_collected"]["first_step"] is not None
    opportunities = [value for value in row["visible_interaction_opportunities"]
                     if value["post_item_use"]]
    assert opportunities and all(not value["actual_e_edge"] for value in opportunities)
    assert any(value["kind"] == "teacher_e_actual_absent"
               for value in row["interaction_candidates"])
    assert "item_used" not in row["visible_events"]
    assert not row["outcome"]["success"] and row["truncated"]


def test_final_interaction_success_comes_from_visible_hud_not_a_scene_event():
    dataset, episode = correction(learner=InteractionLearner(skip_use=False))
    row = describe_correction_coverage(dataset, episode)
    assert row["outcome"]["success"] is True and row["terminated"]
    assert row["visible_events"]["item_collected"]["count"] == 1
    assert "item_used" not in row["visible_events"]


def test_unknown_scene_keeps_valid_denominators_and_does_not_invent_absence():
    class Unknown:
        def __init__(self, task):
            pass

        def observe(self, frame, previous_action=None):
            return {"known": False, "visible_state": None,
                    "event_targets": {}, "event_valid": {}}

    dataset, episode = correction(max_steps=4)
    row = describe_correction_coverage(dataset, episode, annotator_factory=Unknown)
    assert row["scene_unknown_frames"] == row["scene_frames"] == 5
    assert row["scene_known_frames"] == 0
    assert row["visible_interaction_opportunities"] == []
    assert set(row["visible_events"]) == set(EVENT_NAMES)
    assert all(value["valid_transitions"] == value["count"] == 0
               for value in row["visible_events"].values())


def test_no_displacement_does_not_claim_a_wall_or_unnecessary_action():
    dataset, episode = correction("navigation", learner=FixedLearner([
        PhysicalAction(keys=("left",))]))
    row = describe_correction_coverage(dataset, episode)
    assert row["movement_with_grid_position_observed"] == 80
    assert len(row["movement_without_visible_displacement_steps"]) >= 75
    assert row["viewport_rgb"]["longest_run"] >= 75
    assert "blocked" not in row and "unnecessary_actions" not in row


def test_summary_separates_episode_counts_from_repeated_steps_and_rejects_duplicates():
    dataset, episode = correction(learner=InteractionLearner(skip_use=True))
    row = describe_correction_coverage(dataset, episode)
    summary = summarize_correction_coverage([row])
    game = summary["by_game"]["interaction"]
    assert summary["episodes"] == game["episodes"] == 1
    assert game["steps"] == 80 and game["outcomes"]["timeout"] == 1
    assert game["episode_coverage"]["unused_post_item_use"] == 1
    assert game["visible_events"]["item_collected"]["episodes"] == 1
    with pytest.raises(ValueError, match="distinct correction episodes"):
        summarize_correction_coverage([row, row])


def test_pure_teacher_source_is_not_reported_as_a_correction():
    dataset = collect_scripted_demonstrations(1, tasks=("interaction",), seed=8176,
                                             level_bank_seed=20261015)
    with pytest.raises(ValueError, match="TRAIN corrections"):
        describe_correction_coverage(dataset, dataset.episodes[0])
