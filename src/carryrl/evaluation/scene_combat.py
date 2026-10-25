"""Disclosed RGB diagnostic reference for the four canonical combat/UI scenes.

This parser is neither a policy input nor a primary reward. Potentials describe
observable task completion and proximity, never win probabilities or hidden
utility. Goal choice explicitly means completion at EITHER visible target; the
white payoff pips and guard do not introduce an invented utility preference.
Unknown pixels/occlusion abstain, and missing observations invalidate edges.
"""

from collections import Counter, deque

import numpy as np
from PIL import Image, ImageDraw

from carryrl.envs.atomic_combat import (
    BACKGROUND,
    CONFIRMED,
    CURSOR,
    DANGER,
    ENEMY,
    HUD_HEIGHT,
    ITEM_COLORS,
    PANEL,
    PATROL,
    PLAYER,
    RISKY,
    SAFE,
    SELECTED,
    _ui_panel_top,
)

SUPPORTED_TASKS = ("goal_choice", "hazards", "shooting", "ui")
GOALS = {
    "goal_choice": "Completion only: reach either visible target; no payoff/risk utility weights.",
    "hazards": "Reach the green goal while avoiding visible traps and the patrol corridor.",
    "shooting": "Remove all visible enemy health pips by aiming and firing from range.",
    "ui": "Open inventory, select the requested item, confirm, then close inventory.",
}
EVENT_NAMES = (
    "exit_reached", "enemy_damaged", "shot_missed", "inventory_opened", "item_selected",
    "item_confirmed", "inventory_closed",
)
_TASK_EVENTS = {
    "goal_choice": {"exit_reached"}, "hazards": {"exit_reached"},
    "shooting": {"enemy_damaged", "shot_missed"},
    "ui": {"inventory_opened", "item_selected", "item_confirmed", "inventory_closed"},
}
_FLOOR = (122, 133, 153)


def _mask(frame, color):
    result = np.all(frame == color, axis=-1)
    result[:HUD_HEIGHT] = False
    return result


def _bounds(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("required visible object missing or occluded")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _center(mask):
    x0, y0, x1, y1 = _bounds(mask)
    return (x0 + x1) / 2, (y0 + y1) / 2


def _components(mask):
    remaining = set(zip(*np.nonzero(mask), strict=True))
    parts = []
    while remaining:
        start = remaining.pop()
        part, queue = {start}, [start]
        while queue:
            y, x = queue.pop()
            for other in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if other in remaining:
                    remaining.remove(other)
                    part.add(other)
                    queue.append(other)
        result = np.zeros_like(mask)
        ys, xs = zip(*part, strict=True)
        result[ys, xs] = True
        parts.append(result)
    return parts


def _blank(size):
    result = np.empty((size, size, 3), dtype=np.uint8)
    result[:] = BACKGROUND
    return result


def _draw_shape(size, box, *, ellipse=False, outline=False):
    image = Image.new("L", (size, size))
    draw = ImageDraw.Draw(image)
    method = draw.ellipse if ellipse else draw.rectangle
    method(box, **({"outline": 1} if outline else {"fill": 1}))
    result = np.asarray(image).astype(bool)
    result[:HUD_HEIGHT] = False
    return result


def _rectangle(mask):
    box = _bounds(mask)
    if not np.array_equal(mask, _draw_shape(mask.shape[0], box)):
        raise ValueError("noncanonical rectangular object")
    return box


def _same(actual, expected):
    if not np.array_equal(actual[HUD_HEIGHT:], expected[HUD_HEIGHT:]):
        raise ValueError("noncanonical or occluded scene geometry")


def _world_distance(first, second, size):
    return float(np.hypot((first[0] - second[0]) * 96 / size,
                         (first[1] - second[1]) * 68 / (size - HUD_HEIGHT)))


class CombatSceneAnnotator:
    """One canonical episode, RGB plus optionally the actually executed action.

    Actions are accepted for a uniform history interface; RGB alone determines
    labels. They are never interpreted as a teacher recommendation. A fresh UI
    closed frame cannot reveal whether an earlier confirmation happened.
    """

    def __init__(self, task):
        self.task = task
        self._origin = None
        self._previous = None
        self._gap = True
        self._size = None

    def _unknown(self, reason):
        self._gap = True
        return {
            "known": False, "potential": None, "potential_valid": False,
            "events": [], "event_valid": dict.fromkeys(EVENT_NAMES, False),
            "event_targets": dict.fromkeys(EVENT_NAMES, False), "visible_state": {},
            "terminal_success": None, "terminal_success_valid": False, "reason": reason,
        }

    def observe(self, frame, previous_action=None):
        """Return JSON labels and masks; a corrupt frame breaks event continuity."""
        if self.task not in SUPPORTED_TASKS:
            return self._unknown("unsupported task")
        initializing = self._origin is None
        try:
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
                raise ValueError("expected uint8 RGB frame")
            if frame.shape not in ((64, 64, 3), (96, 96, 3), (128, 128, 3)):
                raise ValueError("unsupported RGB geometry")
            if self._size is not None and self._size != frame.shape[0]:
                raise ValueError("episode RGB geometry changed")
            state, potential, success, success_valid = getattr(self, "_" + self.task)(frame)
            continuous = self._previous is not None and not self._gap
            if continuous:
                self._validate_transition(state)
            events = self._events(state) if continuous else []
        except (ValueError, TypeError, IndexError) as error:
            if initializing:
                self._origin = None
            return self._unknown(str(error))
        self._size = frame.shape[0]
        self._previous, self._gap = state, False
        return {
            "known": True, "potential": potential, "potential_valid": potential is not None,
            "events": events,
            "event_valid": {name: continuous and name in _TASK_EVENTS[self.task]
                            for name in EVENT_NAMES},
            "event_targets": {name: name in events for name in EVENT_NAMES},
            "visible_state": state, "terminal_success": success,
            "terminal_success_valid": success_valid,
            "reason": "canonical RGB diagnostic; completion-only goal" if self.task == "goal_choice"
            else "canonical RGB diagnostic",
        }

    def _validate_transition(self, state):
        before = self._previous
        if before.get("reached") and state != before:
            raise ValueError("scene changed after visible success")
        if self.task in {"goal_choice", "hazards", "shooting"}:
            first, second = before["player"], state["player"]
            if self.task == "hazards":
                bounds = (1, 1)
            else:
                bounds = (4 * self._size / 96 + 1,
                          4 * (self._size - HUD_HEIGHT) / 68 + 1)
            if any(abs(a - b) > bound for a, b, bound in zip(first, second, bounds, strict=True)):
                raise ValueError("nonphysical player transition")
        if self.task == "shooting":
            damage, misses = before["hp"] - state["hp"], state["misses"] - before["misses"]
            if damage not in (0, 1) or misses not in (0, 1) or damage + misses > 1:
                raise ValueError("nonphysical shot transition")

    def _events(self, state):
        before, events = self._previous, []
        if self.task in {"goal_choice", "hazards"}:
            if state["reached"] and not before["reached"]:
                events.append("exit_reached")
        elif self.task == "shooting":
            if state["hp"] < before["hp"]:
                events.append("enemy_damaged")
            if state["misses"] > before["misses"]:
                events.append("shot_missed")
        else:
            if state["opened"] and not before["opened"]:
                events.append("inventory_opened")
            if (state["opened"] and before["opened"] and state["correct"]
                    and state["selection"] != before["selection"]):
                events.append("item_selected")
            if state["confirmed"] and not before["confirmed"]:
                events.append("item_confirmed")
            if before["opened"] and not state["opened"]:
                events.append("inventory_closed")
        return events

    def _hazards(self, frame):
        size, grid = frame.shape[0], {}
        expected = _blank(size)
        xs = np.linspace(3, size - 3, 6, dtype=int)
        ys = np.linspace(HUD_HEIGHT + 2, size - 3, 6, dtype=int)
        vocabulary = {PLAYER: "player", SAFE: "goal", DANGER: "trap", ENEMY: "enemy",
                      PATROL: "patrol", _FLOOR: "floor"}
        for row in range(5):
            for column in range(5):
                box = xs[column] + 1, ys[row] + 1, xs[column + 1] - 1, ys[row + 1] - 1
                color = tuple(int(v) for v in frame[box[1], box[0]])
                if color not in vocabulary:
                    raise ValueError("unknown hazard tile")
                grid[row, column] = vocabulary[color]
                expected[_draw_shape(size, box)] = color
        _same(frame, expected)
        counts = Counter(grid.values())
        if counts["player"] != 1:
            raise ValueError("expected exactly one player")
        player = next(cell for cell, kind in grid.items() if kind == "player")
        if self._origin is None:
            if counts["goal"] != 1 or counts["trap"] != 4 or counts["enemy"] != 1 \
                    or counts["patrol"] != 1:
                raise ValueError("complete initial goal/trap/patrol layout required")
            self._origin = {"grid": grid.copy(),
                            "goal": next(c for c, k in grid.items() if k == "goal"),
                            "traps": {c for c, k in grid.items() if k == "trap"},
                            "patrol": {c for c, k in grid.items() if k in {"enemy", "patrol"}}}
        origin = self._origin
        for cell, kind in grid.items():
            if cell == player:
                continue
            initial = origin["grid"][cell]
            allowed = {"enemy", "patrol"} if cell in origin["patrol"] else {
                "floor" if initial == "player" else initial}
            if kind not in allowed:
                raise ValueError("hazard topology changed")
        if counts["enemy"] != (0 if player in origin["patrol"] and counts["patrol"] == 1 else 1):
            raise ValueError("inconsistent visible patrol occupancy")
        failed = player in origin["traps"] or (player in origin["patrol"] and not counts["enemy"])
        reached = player == origin["goal"] and not failed
        blocked = origin["traps"] | origin["patrol"]
        queue, visited, distance = deque([(player, 0)]), {player}, None
        while queue:
            cell, steps = queue.popleft()
            if cell == origin["goal"]:
                distance = steps
                break
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                other = cell[0] + dr, cell[1] + dc
                if 0 <= other[0] < 5 and 0 <= other[1] < 5 \
                        and other not in blocked and other not in visited:
                    visited.add(other)
                    queue.append((other, steps + 1))
        potential = 0.0 if failed else (1.0 if reached else (
            None if distance is None else (24 - distance) / 25))
        return {"player": list(player), "goal": list(origin["goal"]),
                "traps": [list(c) for c in sorted(origin["traps"])],
                "patrol": [list(c) for c in sorted(origin["patrol"])],
                "distance": distance, "failed": failed, "reached": reached}, \
            potential, reached, True

    def _goal_choice(self, frame):
        size = frame.shape[0]
        player_mask = _mask(frame, PLAYER)
        player_box = _rectangle(player_mask)
        if player_box[2] - player_box[0] != 4 or player_box[3] - player_box[1] != 4:
            raise ValueError("invalid player geometry")
        player = _center(player_mask)
        if self._origin is None:
            safe_mask = _mask(frame, SAFE)
            if not np.array_equal(safe_mask, _draw_shape(size, _bounds(safe_mask), ellipse=True)):
                raise ValueError("complete initial safe target required")
            templates = []
            for sx, gy in self._integer_world_positions(_center(safe_mask), size):
                rx = 96 - sx
                outline = _draw_shape(size, ((rx - 9) * size / 96,
                    HUD_HEIGHT + 21 * (size - HUD_HEIGHT) / 68, (rx + 9) * size / 96,
                    HUD_HEIGHT + 50 * (size - HUD_HEIGHT) / 68), outline=True)
                for safe_value in range(1, 6):
                    for risky_value in range(1, 6):
                        base = _blank(size)
                        base[outline] = DANGER
                        goals = []
                        for x, value, color in ((sx, safe_value, SAFE), (rx, risky_value, RISKY)):
                            cx, cy = x * size / 96, HUD_HEIGHT + gy * (size - HUD_HEIGHT) / 68
                            radius = max(3, size / 16)
                            circle = _draw_shape(size, (cx - radius, cy - radius,
                                                       cx + radius, cy + radius), ellipse=True)
                            goals.append(_center(circle))
                            base[circle] = color
                            for index in range(value):
                                px = cx - (value - 1) * 1.5 + index * 3
                                pip = _draw_shape(size, (px, cy - radius - 3,
                                                         px + 1, cy - radius - 2))
                                base[pip] = SELECTED
                        # Guard and player are painted after these static shapes.
                        exposed = ~(_mask(frame, DANGER) | player_mask)
                        exposed[:HUD_HEIGHT] = False
                        if np.array_equal(base[exposed], frame[exposed]):
                            templates.append({"base": base, "goals": goals, "risky_x": [rx]})
        else:
            templates = [self._origin]
        matches = []
        radius = max(3, size / 22)
        for template in templates:
            for x in template["risky_x"]:
                px = x * size / 96
                for y in range(22, 49):
                    py = HUD_HEIGHT + y * (size - HUD_HEIGHT) / 68
                    guard = _draw_shape(size, (px - radius, py - radius,
                                              px + radius, py + radius), ellipse=True)
                    expected = template["base"].copy()
                    expected[guard], expected[player_mask] = DANGER, PLAYER
                    if np.array_equal(expected[HUD_HEIGHT:], frame[HUD_HEIGHT:]):
                        matches.append((template, (px, py), (x, y)))
        if not matches:
            raise ValueError("goal/guard/corridor geometry unknown")
        self._origin, guard_center, _ = matches[0]
        goals = self._origin["goals"]
        distance = min(_world_distance(player, target, size) for target in goals)
        # Rendered rectangles/ellipses quantize integer world coordinates. All
        # compatible coordinates must agree before a terminal label is valid.
        possible_player = self._integer_world_positions(player, size)
        possible_goals = [self._integer_world_positions(target, size) for target in goals]
        contact = [any(all(np.hypot(p[0] - g[0], p[1] - g[1]) <= 7 for g in goal)
                       for goal in possible_goals) for p in possible_player]
        guard_distances = [np.hypot(p[0] - match[2][0], p[1] - match[2][1])
                           for p in possible_player for match in matches]
        guard_clear = bool(guard_distances and min(guard_distances) > 8)
        failed = bool(guard_distances and max(guard_distances) <= 8)
        reached = bool(contact and all(contact) and guard_clear)
        certainly_outside = all(all(np.hypot(p[0] - g[0], p[1] - g[1]) > 7
                                   for goal in possible_goals for g in goal)
                                for p in possible_player)
        success_valid = reached or certainly_outside or failed
        potential = 0.0 if failed else (1.0 if reached else min(.99, max(0.0, 1 - distance / 120)))
        state = {"player": list(player), "targets": [list(c) for c in goals],
                 "guard": list(guard_center), "distance": distance, "reached": reached,
                 "utility": "completion_only", "failed": failed}
        return state, potential, reached if success_valid else None, success_valid

    @staticmethod
    def _integer_world_positions(center, size):
        xs = [x for x in range(97) if int(x * size / 96) == center[0]]
        ys = [y for y in range(69)
              if int(HUD_HEIGHT + y * (size - HUD_HEIGHT) / 68) == center[1]]
        return [(x, y) for x in xs for y in ys]

    def _shooting(self, frame):
        size = frame.shape[0]
        player_mask, cursor_mask = _mask(frame, PLAYER), _mask(frame, CURSOR)
        player_box = _bounds(player_mask)
        full_player = _draw_shape(size, player_box)
        if (player_box[2] - player_box[0] != 4 or player_box[3] - player_box[1] != 4
                or not np.array_equal(player_mask, full_player & ~cursor_mask)):
            raise ValueError("player occluded or malformed")
        cy = int(np.argmax(cursor_mask.sum(axis=1)))
        cx = int(np.argmax(cursor_mask.sum(axis=0)))
        cursor = np.zeros_like(cursor_mask)
        cursor[max(HUD_HEIGHT, cy - 4):min(size, cy + 5), cx] = True
        cursor[cy, max(0, cx - 4):min(size, cx + 5)] = True
        if not np.array_equal(cursor, cursor_mask):
            raise ValueError("cursor occluded or malformed")
        hp_parts, miss_parts = _components(_mask(frame, SAFE)), _components(_mask(frame, ENEMY))
        for part in hp_parts:
            box = _rectangle(part)
            if box[2] - box[0] != 2 or box[3] - box[1] not in (0, 1):
                raise ValueError("health pip occluded or malformed")
        for index, part in enumerate(sorted(miss_parts, key=lambda part: _bounds(part)[0])):
            if _rectangle(part) != (3 + index * 4, size - 4, 5 + index * 4, size - 2):
                raise ValueError("miss pip occluded or malformed")
        if self._origin is None:
            enemy_mask = _mask(frame, DANGER)
            if not np.array_equal(enemy_mask, _draw_shape(size, _bounds(enemy_mask), ellipse=True)):
                raise ValueError("complete initial enemy geometry required")
            if not hp_parts or miss_parts:
                raise ValueError("initial visible unspent health/firing budget required")
            self._origin = {"enemy_mask": enemy_mask, "enemy": _center(enemy_mask),
                            "hp_slots": sorted(hp_parts, key=lambda part: _bounds(part)[0])}
        origin, hp, misses = self._origin, len(hp_parts), len(miss_parts)
        if hp > len(origin["hp_slots"]) or misses > 6:
            raise ValueError("invalid visible health/firing budget")
        expected = _blank(size)
        expected[origin["enemy_mask"]] = DANGER
        for slot in origin["hp_slots"][:hp]:
            if np.any(slot & (player_mask | cursor_mask)):
                raise ValueError("health pips occluded")
            expected[slot] = SAFE
        # Missing pips covered by another sprite cannot certify damage.
        if any(np.any(slot & (player_mask | cursor_mask)) for slot in origin["hp_slots"][hp:]):
            raise ValueError("missing health pips occluded")
        expected[player_mask], expected[cursor_mask] = PLAYER, CURSOR
        for part in miss_parts:
            expected[part] = ENEMY
        _same(frame, expected)
        if self._previous is not None and (hp > self._previous["hp"]
                                           or misses < self._previous["misses"]):
            raise ValueError("reversed irreversible damage/firing cue")
        reached = hp == 0
        total_hp = len(origin["hp_slots"])
        potential = 1.0 if reached else .95 * (total_hp - hp) / total_hp
        return {"player": list(_center(player_mask)), "enemy": list(origin["enemy"]),
                "cursor": [cx, cy], "hp": hp, "misses": misses, "reached": reached}, \
            potential, reached, True

    def _ui(self, frame):
        size, panel_top = frame.shape[0], _ui_panel_top(frame.shape[0])
        expected = _blank(size)
        request_region = np.zeros((size, size), dtype=bool)
        request_region[HUD_HEIGHT:panel_top] = True
        requests = [(index, _mask(frame, color) & request_region)
                    for index, color in enumerate(ITEM_COLORS)]
        requests = [(index, mask) for index, mask in requests if np.any(mask)]
        if len(requests) != 1:
            raise ValueError("request glyph missing or ambiguous")
        requested, request = requests[0]
        request_box = _rectangle(request)
        if request_box[2] - request_box[0] != 6 or request_box[3] - request_box[1] != 6:
            raise ValueError("invalid request glyph")
        expected[request] = ITEM_COLORS[requested]
        opened = bool(np.any(_mask(frame, PANEL)))
        confirmed = bool(np.any(_mask(frame, CONFIRMED)))
        selected, correct = None, False
        items = []
        if opened:
            expected[panel_top:size - 3, 3:size - 3] = PANEL
            for index, color in enumerate(ITEM_COLORS):
                mask = _mask(frame, color) & ~request_region
                if np.any(mask):
                    box = _rectangle(mask)
                    items.append((index, box))
                    expected[mask] = color
            if len(items) not in (3, 4) or requested not in {index for index, _ in items}:
                raise ValueError("incomplete item layout")
            white = _mask(frame, SELECTED)
            selected_center = _center(white)
            selected, box = min(items, key=lambda item: np.hypot(
                (item[1][0] + item[1][2]) / 2 - selected_center[0],
                (item[1][1] + item[1][3]) / 2 - selected_center[1]))
            border = 2 if box[2] - box[0] >= 2 * max(3, size / 16) - 1 else 1
            border_box = box[0] - border, box[1] - border, box[2] + border, box[3] + border
            expected[_draw_shape(size, border_box)] = SELECTED
            expected[_draw_shape(size, box)] = ITEM_COLORS[selected]
            correct = selected == requested
            if confirmed:
                if not correct:
                    raise ValueError("confirmation disagrees with visible selected request")
                expected[panel_top:panel_top + 4, 3:7] = CONFIRMED
        else:
            x, y = 48 * size / 96, HUD_HEIGHT + 43 * (size - HUD_HEIGHT) / 68
            expected[_draw_shape(size, (x - 5, y - 4, x + 5, y + 4), outline=True)] = SELECTED
        _same(frame, expected)
        if self._previous is not None and self._previous["confirmed"] and opened and not confirmed:
            raise ValueError("confirmation reversed while inventory is open")
        origin = {"request": requested, "request_mask": request}
        if self._origin is None:
            self._origin = origin
        if opened:
            if "items" in self._origin and items != self._origin["items"]:
                raise ValueError("item layout changed within episode")
            self._origin["items"] = items
        if requested != self._origin["request"] or not np.array_equal(
                request, self._origin["request_mask"]):
            raise ValueError("request changed within episode")
        closed_after_confirmation = bool(
            not opened and self._previous is not None and not self._gap
            and self._previous["confirmed"])
        closed_success_held = bool(
            not opened and self._previous is not None and not self._gap
            and self._previous["reached"])
        reached = closed_after_confirmation or closed_success_held
        if reached:
            confirmed = True
        potential = 1.0 if reached else (
            .75 if confirmed else (.5 if correct else (.25 if opened else 0)))
        # A closed image after a missing observation does not disclose its past.
        success_valid = opened or reached
        return {"opened": opened, "request": requested, "selection": selected,
                "correct": correct, "confirmed": confirmed, "reached": reached}, \
            potential if success_valid else None, reached if success_valid else None, success_valid
