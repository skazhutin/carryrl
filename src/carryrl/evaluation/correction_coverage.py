"""Observed coverage of canonical TRAIN corrections; never an error or reward oracle."""

from collections import Counter

import numpy as np

from carryrl.agent.actions import PhysicalAction
from carryrl.evaluation.scene_suite import EVENT_NAMES, SCENE_SUITE_VERSION, SuiteSceneAnnotator
from carryrl.evaluation.trajectory_diagnostics import _frame_hash
from carryrl.rewards.visual import HUD_HEIGHT, verify_visible_outcome
from carryrl.training.imitation import CORRECTION_SOURCE, demonstration_episode_digest

COVERAGE_VERSION = "carryrl.correction_coverage.v1"
_MOVEMENT = frozenset(("w", "a", "s", "d", "up", "down", "left", "right"))
_INTERACTIONS = frozenset(("item", "use", "button", "key", "switch", "door"))


def _unchanged_summary(flags, window):
    longest = run = total = 0
    first = None
    for step, unchanged in enumerate(flags, 1):
        run = run + 1 if unchanged else 0
        total += int(unchanged)
        longest = max(longest, run)
        if run == window and first is None:
            first = step
    return {"transitions": len(flags), "unchanged_transitions": total,
            "longest_run": longest, "first_window_step": first}


def describe_correction_coverage(dataset, episode, *, stall_window=8, early_steps=20,
                                 annotator_factory=SuiteSceneAnnotator):
    """Describe actual RGB/commands and separately identify interaction candidates.

    Dataset invariants are checked here. Exact environment/teacher replay belongs to
    the bundle importer and correction-rescue authentication. Scene references know
    the canonical atomic presentation; an abstention is not absence of an event.
    """
    if type(stall_window) is not int or not 1 <= stall_window <= 80:
        raise ValueError("coverage stall_window must be an integer in [1,80]")
    if type(early_steps) is not int or not 1 <= early_steps <= 80:
        raise ValueError("coverage early_steps must be an integer in [1,80]")
    if (episode.source != CORRECTION_SOURCE or episode.split != "train"
            or episode.difficulty != 1.0 or episode.presentation is not None
            or episode.native_size is not None or episode.executed_actions is None
            or len(episode) > 80):
        raise ValueError("coverage requires whole canonical96 full1 TRAIN corrections")
    dataset._validate_episode(episode)
    frames = [dataset.frames[index] for index in
              (*episode.frame_indices, episode.final_frame_index)]
    if any(frame.shape != (96, 96, 3) for frame in frames):
        raise ValueError("coverage requires canonical96 RGB")
    actual, labels = episode.executed_actions, episode.actions
    options = {"goal_choice_utility": "completion_only"} if episode.task == "goal_choice" else {}
    annotator = annotator_factory(episode.task, **options)
    scenes = [annotator.observe(frame, previous_action=actual[index - 1] if index else None)
              for index, frame in enumerate(frames)]
    full_flags = [np.array_equal(left, right) for left, right in zip(frames, frames[1:])]
    viewport_flags = [np.array_equal(left[HUD_HEIGHT:], right[HUD_HEIGHT:])
                      for left, right in zip(frames, frames[1:])]
    commands = Counter(total=len(actual))
    interaction_candidates, opportunities, no_displacement, hazard_contacts = [], [], [], []
    movement_observed = 0
    previous_e = False
    for index, (action, label) in enumerate(zip(actual, labels, strict=True)):
        step = index + 1
        current_e = "e" in action.keys
        edge = current_e and not previous_e
        commands["neutral"] += int(action == PhysicalAction())
        commands["movement"] += int(bool(_MOVEMENT.intersection(action.keys)))
        commands["e_down"] += int(current_e)
        commands["e_rising"] += int(edge)
        commands["e_held"] += int(current_e and previous_e)
        commands["e_release"] += int(previous_e and not current_e)
        if "e" in label.keys and not edge:
            interaction_candidates.append({"step": step,
                "kind": "teacher_e_actual_held" if current_e else "teacher_e_actual_absent"})
        before, after = scenes[index], scenes[index + 1]
        state = before.get("visible_state") or {}
        target = state.get("next_target")
        if (before["known"] and target in _INTERACTIONS and state.get("distance") == 0):
            opportunities.append({"step": step, "target": target, "actual_e_edge": edge,
                "actual_e_held": current_e and previous_e,
                "post_item_use": target == "use" and state.get("completed", {}).get(
                    "item") is True})
        if before["known"] and after["known"]:
            next_state = after.get("visible_state") or {}
            left, right = state.get("player"), next_state.get("player")
            if left is not None and right is not None and _MOVEMENT.intersection(action.keys):
                movement_observed += 1
                if left == right:
                    no_displacement.append(step)
            hazard = next_state.get("objects", {}).get("hazard")
            if (hazard is not None and right == hazard
                    and verify_visible_outcome(frames[index + 1]).get("failure") is True):
                hazard_contacts.append(step)
        previous_e = current_e
    event_counts, event_valid, first_events = Counter(), Counter(), {}
    for step, row in enumerate(scenes[1:], 1):
        for name in EVENT_NAMES:
            if row["event_valid"].get(name) is True:
                event_valid[name] += 1
                if row["event_targets"].get(name) is True:
                    event_counts[name] += 1
                    first_events.setdefault(name, step)
    outcome = verify_visible_outcome(frames[-1])
    failures = [step for step, frame in enumerate(frames)
                if verify_visible_outcome(frame).get("failure") is True]
    return {"version": COVERAGE_VERSION, "scene_suite_version": SCENE_SUITE_VERSION,
        "episode_content_sha256": demonstration_episode_digest(dataset, episode),
        "source_episode": {"task": episode.task, "split": episode.split,
            "source": episode.source, "seed": episode.seed, "level_id": episode.level_id,
            "level_index": episode.level_index, "level_bank_seed": episode.level_bank_seed},
        "steps": len(actual), "outcome": outcome, "terminated": episode.terminated,
        "truncated": episode.truncated, "frame_sha256": [_frame_hash(frame) for frame in frames],
        "distinct_full_rgb": len({_frame_hash(frame) for frame in frames}),
        "distinct_viewport_rgb": len({_frame_hash(frame[HUD_HEIGHT:]) for frame in frames}),
        "full_rgb": _unchanged_summary(full_flags, stall_window),
        "viewport_rgb": _unchanged_summary(viewport_flags, stall_window),
        "stall_window": stall_window, "commands": dict(commands),
        "scene_frames": len(scenes), "scene_known_frames": sum(row["known"] for row in scenes),
        "scene_unknown_frames": sum(not row["known"] for row in scenes),
        "visible_events": {name: {"count": event_counts[name],
            "valid_transitions": event_valid[name], "first_step": first_events.get(name)}
            for name in EVENT_NAMES},
        "interaction_candidates": interaction_candidates,
        "visible_interaction_opportunities": opportunities,
        "movement_with_grid_position_observed": movement_observed,
        "movement_without_visible_displacement_steps": no_displacement,
        "visible_hazard_contact_with_failure_steps": hazard_contacts,
        "first_visible_failure_step": failures[0] if failures else None,
        "early_steps": early_steps,
        "early_visible_failure": bool(failures and failures[0] <= early_steps),
        "interpretation": "Observed coverage, not error counts or independent examples. "
            "Teacher E recommendations and unused visible opportunities are candidates; "
            "movement happens before E, and E requires an edge in actual command history. "
            "Unchanged RGB/displacement is not proof of a wall or wasted action. Scene "
            "abstention is unknown; final interaction use may have no scene event. "
            "No hidden game state or reward API is an input to these observations."}


def summarize_correction_coverage(reports):
    """Keep episode denominators separate from counts of repeated transitions."""
    by_game, seen = {}, set()
    for row in reports:
        identity = row["episode_content_sha256"]
        if identity in seen:
            raise ValueError("coverage summary requires distinct correction episodes")
        seen.add(identity)
        task = row["source_episode"]["task"]
        summary = by_game.setdefault(task, {"episodes": 0, "steps": 0,
            "outcomes": Counter(), "commands": Counter(), "episode_coverage": Counter(),
            "visible_events": {name: {"episodes": 0, "transitions": 0,
                                      "valid_transitions": 0} for name in EVENT_NAMES},
            "scene_known_frames": 0, "scene_unknown_frames": 0})
        summary["episodes"] += 1
        summary["steps"] += row["steps"]
        outcome = row["outcome"]
        category = ("success" if outcome.get("success") is True else "failure"
                    if outcome.get("failure") is True else "timeout" if row["truncated"]
                    else "unknown")
        summary["outcomes"][category] += 1
        summary["commands"].update(row["commands"])
        opportunities = row["visible_interaction_opportunities"]
        for name, present in {
            "viewport_stall": row["viewport_rgb"]["first_window_step"] is not None,
            "full_rgb_stall": row["full_rgb"]["first_window_step"] is not None,
            "teacher_e_candidate": bool(row["interaction_candidates"]),
            "unused_visible_interaction": any(not x["actual_e_edge"] for x in opportunities),
            "unused_post_item_use": any(x["post_item_use"] and not x["actual_e_edge"]
                                        for x in opportunities),
            "movement_without_visible_displacement": bool(
                row["movement_without_visible_displacement_steps"]),
            "early_visible_failure": row["early_visible_failure"],
        }.items():
            summary["episode_coverage"][name] += int(present)
        summary["scene_known_frames"] += row["scene_known_frames"]
        summary["scene_unknown_frames"] += row["scene_unknown_frames"]
        for name, event in row["visible_events"].items():
            target = summary["visible_events"][name]
            target["episodes"] += int(event["count"] > 0)
            target["transitions"] += event["count"]
            target["valid_transitions"] += event["valid_transitions"]
    return {"episodes": sum(row["episodes"] for row in by_game.values()), "by_game": by_game,
            "interpretation": "Counts describe this fixed TRAIN cohort only. Zero valid "
                "scene transitions means unavailable coverage, not absence of an event. "
                "Repeated frames/commands are not independent examples or policy error rates."}
