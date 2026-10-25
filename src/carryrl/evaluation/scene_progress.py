"""Bounded, suite-specific RGB supervision for auditing canonical grid scenes.

This public rendering parser is diagnostic supervision, never a policy input or
primary reward. Potential is an engineering target, not win probability, game
score, or an optimal value function. It uses completed prerequisite phases plus
geodesic proximity to the next target. A fixed 24-step simple-path bound avoids
normalizing with future routes; proximity stays below a completed phase.
Only canonical RGB64/96/128 grids are supported, without presentation styling.
"""

from collections import Counter, deque

import numpy as np

from carryrl.envs.atomic_interaction import BACKGROUND, COLORS, GRID_SIZE, PLAYER, _box
from carryrl.rewards.visual import HUD_HEIGHT

SCENE_PROGRESS_VERSION = "carryrl.scene_progress.v1"
SUPPORTED_TASKS = ("interaction", "causality", "key_door", "chain")
EVENT_NAMES = (
    "item_collected", "key_collected", "button_activated", "switch_activated",
    "door_opened", "exit_reached",
)
GOALS = {
    "interaction": "Collect the item, then use it beside the purple pad.",
    "causality": "Activate the remote button to open the door, then reach the green exit.",
    "key_door": "Collect the key, unlock the door, then reach the green exit.",
    "chain": "Collect the key, activate the switch, unlock the door, then reach the green exit.",
}
_TASK_OBJECTS = {
    "interaction": ("item", "use"),
    "causality": ("button", "door", "exit"),
    "key_door": ("key", "door", "exit"),
    "chain": ("key", "switch", "door", "exit"),
}
_TASK_EVENTS = {
    "interaction": {"item_collected"},
    "causality": {"button_activated", "door_opened", "exit_reached"},
    "key_door": {"key_collected", "door_opened", "exit_reached"},
    "chain": {"key_collected", "switch_activated", "door_opened", "exit_reached"},
}
_REPLACEMENTS = {
    "item": "floor", "key": "floor", "button": "spent", "switch": "spent",
    "door": "open_door",
}
_PHASES = {"interaction": 2, "causality": 2, "key_door": 3, "chain": 4}
_CELLS = tuple((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE))


def _near(first, second):
    return abs(first[0] - second[0]) + abs(first[1] - second[1]) <= 1


def _neighbors(cell):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = cell[0] + dr, cell[1] + dc
        if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
            yield neighbor


def _distance(position, target, grid, *, interact):
    blocked = {cell for cell, kind in grid.items() if kind in {"wall", "hazard", "door"}}
    goals = {cell for cell in _CELLS if _near(cell, target)} if interact else {target}
    goals -= blocked
    queue, seen = deque([(position, 0)]), {position}
    while queue:
        cell, distance = queue.popleft()
        if cell in goals:
            return distance
        for neighbor in _neighbors(cell):
            if neighbor not in blocked and neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


def _parse(frame):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("expected uint8 RGB frame")
    if frame.shape not in ((64, 64, 3), (96, 96, 3), (128, 128, 3)):
        raise ValueError("unsupported RGB geometry")
    grid, players = {}, []
    covered = np.zeros(frame.shape[:2], dtype=bool)
    covered[:HUD_HEIGHT] = True  # HUD labels and scores never participate.
    for cell in _CELLS:
        x0, y0, x1, y1 = _box(frame.shape[0], cell)
        patch = frame[y0:y1, x0:x1]
        center = tuple(int(v) for v in patch[(y1 - y0 - 1) // 2, (x1 - x0 - 1) // 2])
        rim = tuple(int(v) for v in patch[0, 0])
        underlying = COLORS.get(rim)
        if underlying is None or underlying == "player":
            raise ValueError("unrecognized cell rim")
        expected = np.empty_like(patch)
        expected[:] = rim
        if center == PLAYER:
            expected[1:-1, 1:-1] = PLAYER
            players.append(cell)
        if not np.array_equal(patch, expected):
            raise ValueError("noncanonical cell pattern")
        grid[cell] = underlying
        covered[y0:y1, x0:x1] = True
    if not np.all(frame[~covered] == BACKGROUND):
        raise ValueError("noncanonical scene background")
    if len(players) != 1 or grid[players[0]] in {"wall", "door"}:
        raise ValueError("expected exactly one traversable player")
    return grid, players[0]


class SceneProgressAnnotator:
    """Track visible one-way milestones from RGB only, one instance per episode.

    A complete initial layout is required so disappeared objects retain known
    coordinates. Unknown frames break transition supervision; resumption can
    recover a consistent visible state but never invent an edge across the gap.
    Interaction's final use changes no scene pixels, so its success is unknown.
    """

    def __init__(self, task):
        self.task = task
        self._origin = None
        self._previous = None
        self._previous_position = None
        self._previous_done = None
        self._positions = {}
        self._gap = True
        self._seen_events = set()

    def _unknown(self, reason):
        self._gap = True
        return {
            "known": False, "potential": None, "potential_valid": False,
            "events": [], "event_valid": dict.fromkeys(EVENT_NAMES, False),
            "event_targets": dict.fromkeys(EVENT_NAMES, False), "visible_state": {},
            "terminal_success": None, "terminal_success_valid": False, "reason": reason,
        }

    def _initialize(self, grid):
        counts = Counter(grid.values())
        required = {*_TASK_OBJECTS[self.task], "hazard"}
        if any(counts[kind] != 1 for kind in required):
            raise ValueError("complete initial object layout required")
        if set(counts) - (required | {"floor", "wall"}):
            raise ValueError("unexpected initial object vocabulary")
        self._positions = {kind: next(c for c, k in grid.items() if k == kind)
                           for kind in required}
        self._origin = grid.copy()

    def _state(self, grid):
        done = {}
        for cell, original in self._origin.items():
            current = grid[cell]
            if current != original and current != _REPLACEMENTS.get(original):
                raise ValueError("changed object coordinates or structural topology")
            if self._previous is not None:
                previous = self._previous[cell]
                if previous != original and current != previous:
                    raise ValueError("reversed irreversible visible milestone")
        for kind in _TASK_OBJECTS[self.task]:
            if kind in _REPLACEMENTS:
                done[kind] = grid[self._positions[kind]] != kind
        if self.task == "causality" and done["button"] != done["door"]:
            raise ValueError("button and door state disagree")
        if self.task in {"key_door", "chain"} and done["door"] and not done["key"]:
            raise ValueError("door opened without key")
        if self.task == "chain" and (
            (done["switch"] and not done["key"]) or (done["door"] and not done["switch"])
        ):
            raise ValueError("invalid visible prerequisite order")
        return done

    def _transition(self, grid, position, done):
        # Ordinary physical commands move vertically, then horizontally, at most
        # one cell per axis. No action label or recommendation is consulted.
        before = self._previous_position
        if abs(position[0] - before[0]) > 1 or abs(position[1] - before[1]) > 1:
            raise ValueError("nonphysical player transition")
        corner = position[0], before[1]
        for destination in (corner, position):
            if self._previous[destination] in {"wall", "door"}:
                raise ValueError("player crossed a blocked cell")
        if self._previous[corner] == "hazard" and position != corner:
            raise ValueError("player moved after touching hazard")
        if self._previous[before] == "hazard" and position != before:
            raise ValueError("player moved after visible failure")
        changed = {kind for kind in done if done[kind] and not self._previous_done[kind]}
        for kind in changed:
            source = "button" if self.task == "causality" and kind == "door" else kind
            if not _near(position, self._positions[source]):
                raise ValueError("milestone changed outside interaction proximity")
        if self.task in {"key_door", "chain"} and "door" in changed:
            if not self._previous_done["key"]:
                raise ValueError("collecting key cannot simultaneously open door")
        if self.task == "chain" and "switch" in changed:
            if not self._previous_done["key"] or "door" in changed:
                raise ValueError("multiple ordered interactions in one transition")
        if changed and grid[position] == "hazard":
            raise ValueError("interaction cannot occur on visible failure")
        return changed

    def observe(self, frame):
        """Return JSON-compatible labels; unsupported or inconsistent RGB abstains."""
        if self.task not in SUPPORTED_TASKS:
            return self._unknown("unsupported task")
        initializing = self._origin is None
        try:
            grid, position = _parse(frame)
            if self._origin is None:
                self._initialize(grid)
            done = self._state(grid)
            transition_valid = self._previous is not None and not self._gap
            changed = self._transition(grid, position, done) if transition_valid else set()
            success = None if self.task == "interaction" else (
                done["door"] and position == self._positions["exit"]
            )
            previous_success = bool(
                self._previous_done is not None and self.task != "interaction"
                and self._previous_done["door"]
                and self._previous_position == self._positions["exit"]
            )
            previous_failure = (
                self._previous is not None
                and self._previous[self._previous_position] == "hazard"
            )
            if (previous_success or previous_failure) and (
                position != self._previous_position or grid != self._previous
            ):
                raise ValueError("scene changed after visible terminal state")
            completed, target, interact = self._next_target(done)
            distance = _distance(position, self._positions[target], grid, interact=interact)
            if distance is None:
                raise ValueError("required target unreachable in visible topology")
            potential = 1.0 if success else (
                completed + (24 - distance) / 25
            ) / _PHASES[self.task]
            if not 0.0 <= potential < 1.0 and not success:
                raise ValueError("invalid bounded potential")
        except (ValueError, TypeError, IndexError) as error:
            if initializing:
                self._origin, self._positions = None, {}
            return self._unknown(str(error))

        event_targets = dict.fromkeys(EVENT_NAMES, False)
        for kind, event in (("item", "item_collected"), ("key", "key_collected"),
                            ("button", "button_activated"), ("switch", "switch_activated"),
                            ("door", "door_opened")):
            event_targets[event] = kind in changed
        event_targets["exit_reached"] = bool(success and transition_valid and not previous_success)
        events = [event for event in EVENT_NAMES if event_targets[event]
                  and event not in self._seen_events]
        # Edges are emitted once; repeated images and interactions are negatives.
        event_targets = {event: event in events for event in EVENT_NAMES}
        self._seen_events.update(events)
        self._previous, self._previous_position = grid.copy(), position
        self._previous_done, self._gap = done.copy(), False
        visible_state = {
            "player": list(position),
            "objects": {kind: list(cell) for kind, cell in self._positions.items()},
            "completed": done,
            "next_target": target, "distance": distance, "completed_phases": completed,
        }
        return {
            "known": True, "potential": float(potential), "potential_valid": True,
            "events": events,
            "event_valid": {event: transition_valid and event in _TASK_EVENTS[self.task]
                            for event in EVENT_NAMES},
            "event_targets": event_targets, "visible_state": visible_state,
            "terminal_success": success, "terminal_success_valid": self.task != "interaction",
            "reason": "visible state; interaction terminal use is unobservable"
            if self.task == "interaction" else "canonical visible state",
        }

    def _next_target(self, done):
        if self.task == "interaction":
            return (1, "use", True) if done["item"] else (0, "item", True)
        if self.task == "causality":
            return (1, "exit", False) if done["button"] else (0, "button", True)
        if not done["key"]:
            return 0, "key", True
        if self.task == "chain" and not done["switch"]:
            return 1, "switch", True
        if not done["door"]:
            return (2 if self.task == "chain" else 1), "door", True
        return (_PHASES[self.task] - 1), "exit", False
