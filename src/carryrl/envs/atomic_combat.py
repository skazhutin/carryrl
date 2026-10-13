"""Four screen-only value, avoidance, combat and interface exercises.

These are controlled exercises, not full commercial games. Reference controllers
read rendered pixels and their own button history; they do not read game objects.
"""

from collections import deque

import numpy as np
from PIL import Image, ImageDraw

from carryrl.agent.actions import MOUSE_BINS, PhysicalAction

HUD_HEIGHT = 28
BACKGROUND = (31, 38, 52)
PLAYER = (35, 180, 235)
SAFE = (69, 207, 100)
RISKY = (245, 194, 63)
DANGER = (230, 62, 72)
ENEMY = (243, 131, 48)
PATROL = (174, 128, 91)
CURSOR = (250, 245, 218)
PANEL = (82, 89, 110)
SELECTED = (245, 247, 252)
ITEM_COLORS = ((220, 88, 105), (99, 208, 131), (166, 112, 229), (235, 187, 69))
CONFIRMED = (142, 242, 187)


def _ui_panel_top(size):
    scale = (size - HUD_HEIGHT) / 68
    # The request glyph retains its readable pixel radius at small sizes.
    # Leave it entirely above the menu instead of painting over its lower rows.
    return max(int(HUD_HEIGHT + 12 * scale), int(HUD_HEIGHT + 8 * scale + 3) + 1)


def _directions(action):
    keys = set(action.keys)
    return (
        int(bool(keys & {"right", "d"})) - int(bool(keys & {"left", "a"})),
        int(bool(keys & {"down", "s"})) - int(bool(keys & {"up", "w"})),
    )


def _center(frame, color, region=None):
    pixels = np.all(frame == color, axis=-1)
    pixels[:HUD_HEIGHT] = False
    if region is not None:
        allowed = np.zeros_like(pixels)
        allowed[region] = True
        pixels &= allowed
    ys, xs = np.nonzero(pixels)
    return None if not len(xs) else (float(xs.mean()), float(ys.mean()))


def _mouse_delta(value):
    return min(MOUSE_BINS, key=lambda delta: abs(delta - value))


def _keys_towards(origin, target, tolerance=2):
    keys = []
    if target[0] - origin[0] > tolerance:
        keys.append("d")
    elif target[0] - origin[0] < -tolerance:
        keys.append("a")
    if target[1] - origin[1] > tolerance:
        keys.append("s")
    elif target[1] - origin[1] < -tolerance:
        keys.append("w")
    return tuple(keys)


class _Game:
    def __init__(self, level, image_size):
        self.level, self.size = level, image_size
        self.score, self.status, self.ticks = 0, "Play", 0
        self._previous_keys, self._previous_buttons = (), ()

    def _begin(self, action):
        if not isinstance(action, PhysicalAction):
            raise TypeError("Atomic games accept PhysicalAction only")
        if self.status != "Play":
            raise RuntimeError("A finished game must be reset")
        self.ticks += 1

    def _remember(self, action):
        self._previous_keys, self._previous_buttons = action.keys, action.mouse_buttons

    def _key_edge(self, action, key):
        return key in action.keys and key not in self._previous_keys

    def _click(self, action):
        return "left" in action.mouse_buttons and "left" not in self._previous_buttons

    def _point(self, x, y):
        return x * self.size / 96, HUD_HEIGHT + y * (self.size - HUD_HEIGHT) / 68

    def _canvas(self):
        image = Image.new("RGB", (self.size, self.size), BACKGROUND)
        return image, ImageDraw.Draw(image)


class GoalChoiceGame(_Game):
    """Choose a small safe payoff or cross a visible guard for a larger payoff."""

    @classmethod
    def make_level(cls, rng):
        return {
            "safe_side": int(rng.integers(2)),
            "safe_value": int(rng.integers(1, 3)),
            "risky_value": int(rng.integers(3, 6)),
            "goal_y": int(rng.integers(8, 17)),
            "guard_y": int(rng.integers(25, 44)),
            "guard_direction": int(rng.choice((-1, 1))),
            "guard_speed": int(rng.integers(1, 4)),
        }

    def __init__(self, level, image_size):
        super().__init__(level, image_size)
        self.position = [48.0, 59.0]
        self.guard_y = float(level["guard_y"])
        self.guard_direction = level["guard_direction"]
        self._initial_goal_distance = min(
            np.hypot(48 - x, 59 - level["goal_y"]) for x in (self._safe_x, self._risky_x)
        )

    @property
    def _safe_x(self):
        spread = self.level.get("goal_spread", 31)
        return 48 - spread if self.level["safe_side"] == 0 else 48 + spread

    @property
    def _risky_x(self):
        return 96 - self._safe_x

    def step(self, action):
        self._begin(action)
        dx, dy = _directions(action)
        self.position[0] = float(np.clip(self.position[0] + 4 * dx, 6, 90))
        self.position[1] = float(np.clip(self.position[1] + 4 * dy, 5, 63))
        self.guard_y += self.guard_direction * self.level["guard_speed"]
        if self.guard_y < 22 or self.guard_y > 48:
            self.guard_y = float(np.clip(self.guard_y, 22, 48))
            self.guard_direction *= -1
        if np.hypot(self.position[0] - self._risky_x, self.position[1] - self.guard_y) <= 8:
            self.status = "Failed"
        else:
            for x, value in (
                (self._safe_x, self.level["safe_value"]),
                (self._risky_x, self.level["risky_value"]),
            ):
                if np.hypot(self.position[0] - x, self.position[1] - self.level["goal_y"]) <= 7:
                    self.status, self.score = "Success", value
                    break
            if self.status == "Play":
                distance = min(np.hypot(self.position[0] - x,
                                       self.position[1] - self.level["goal_y"])
                               for x in (self._safe_x, self._risky_x))
                # Preserve the actual small safe payoff: partial progress never exceeds one.
                if distance <= 0.8 * self._initial_goal_distance:
                    self.score = max(self.score, 1)
        self._remember(action)

    def render(self):
        image, draw = self._canvas()
        x0, y0 = self._point(self._risky_x - 9, 21)
        x1, y1 = self._point(self._risky_x + 9, 50)
        draw.rectangle((x0, y0, x1, y1), outline=DANGER)
        for x, value, color in (
            (self._safe_x, self.level["safe_value"], SAFE),
            (self._risky_x, self.level["risky_value"], RISKY),
        ):
            cx, cy = self._point(x, self.level["goal_y"])
            radius = max(3, self.size / 16)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
            # Each white pip is one actual payoff unit, visible before choosing.
            for index in range(value):
                px = cx - (value - 1) * 1.5 + index * 3
                draw.rectangle((px, cy - radius - 3, px + 1, cy - radius - 2), fill=SELECTED)
        cx, cy = self._point(self._risky_x, self.guard_y)
        radius = max(3, self.size / 22)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=DANGER)
        cx, cy = self._point(*self.position)
        draw.rectangle((cx - 2, cy - 2, cx + 2, cy + 2), fill=PLAYER)
        return np.asarray(image, dtype=np.uint8).copy()


def _grid_path(start, goal, blocked):
    queue, visited = deque([(start, ())]), {start}
    while queue:
        cell, path = queue.popleft()
        if cell == goal:
            return path
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            other = cell[0] + dr, cell[1] + dc
            if (
                0 <= other[0] < 5 and 0 <= other[1] < 5
                and other not in visited and other not in blocked
            ):
                visited.add(other)
                queue.append((other, (*path, other)))
    return None


class HazardsGame(_Game):
    """Variable trap layouts and a visibly marked, moving two-cell patrol."""

    @classmethod
    def make_level(cls, rng):
        cells = [(row, column) for row in range(5) for column in range(5)]
        while True:
            start, goal = (4, int(rng.integers(5))), (0, int(rng.integers(5)))
            first = int(rng.integers(1, 4)), int(rng.integers(5))
            neighbors = [
                (first[0] + dr, first[1] + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if (
                    0 <= first[0] + dr < 5 and 0 <= first[1] + dc < 5
                    and (first[0] + dr, first[1] + dc) not in (start, goal)
                )
            ]
            second = neighbors[int(rng.integers(len(neighbors)))]
            patrol = first, second
            choices = [cell for cell in cells if cell not in (*patrol, start, goal)]
            traps = [choices[int(index)] for index in rng.choice(len(choices), 4, replace=False)]
            if _grid_path(start, goal, set(traps) | set(patrol)) is not None:
                return {
                    "traps": [list(cell) for cell in traps],
                    "start": list(start), "goal": list(goal),
                    "patrol": [list(cell) for cell in patrol],
                    "enemy_index": int(rng.integers(2)),
                    "enemy_period": int(rng.integers(1, 4)),
                }

    def __init__(self, level, image_size):
        super().__init__(level, image_size)
        self.position = tuple(level["start"])
        self.goal = tuple(level["goal"])
        self.traps = {tuple(cell) for cell in level["traps"]}
        self.patrol = tuple(tuple(cell) for cell in level["patrol"])
        self.enemy_index = level["enemy_index"]
        self._progress_blocked = self.traps | set(self.patrol)
        self._initial_distance = len(_grid_path(self.position, self.goal, self._progress_blocked))

    def step(self, action):
        self._begin(action)
        horizontal, vertical = _directions(action)
        for dr, dc in ((vertical, 0), (0, horizontal)):
            destination = self.position[0] + dr, self.position[1] + dc
            if 0 <= destination[0] < 5 and 0 <= destination[1] < 5:
                self.position = destination
                if self.position in self.traps or self.position == self.patrol[self.enemy_index]:
                    self.status = "Failed"
                    break
        if self.status == "Play" and self.ticks % self.level["enemy_period"] == 0:
            self.enemy_index = 1 - self.enemy_index
            if self.position == self.patrol[self.enemy_index]:
                self.status = "Failed"
        if self.status == "Play" and self.position == self.goal:
            self.status, self.score = "Success", 5
        elif self.status == "Play":
            route = _grid_path(self.position, self.goal, self._progress_blocked)
            if route is not None:
                progress = int(5 * (self._initial_distance - len(route)) / self._initial_distance)
                self.score = max(self.score, min(4, progress))
        self._remember(action)

    def _rectangle(self, row, column):
        xs = np.linspace(3, self.size - 3, 6, dtype=int)
        ys = np.linspace(HUD_HEIGHT + 2, self.size - 3, 6, dtype=int)
        return xs[column] + 1, ys[row] + 1, xs[column + 1] - 1, ys[row + 1] - 1

    def render(self):
        image, draw = self._canvas()
        for row in range(5):
            for column in range(5):
                cell = row, column
                color = (122, 133, 153)
                if cell in self.traps:
                    color = DANGER
                elif cell == self.goal:
                    color = SAFE
                elif cell == self.patrol[self.enemy_index]:
                    color = ENEMY
                elif cell in self.patrol:
                    color = PATROL
                if cell == self.position:
                    color = PLAYER
                draw.rectangle(self._rectangle(row, column), fill=color)
        return np.asarray(image, dtype=np.uint8).copy()


class ShootingGame(_Game):
    """Close firing distance with WASD, independently aim and fire three shots."""

    @classmethod
    def make_level(cls, rng):
        return {
            "player_x": int(rng.integers(10, 28)),
            "player_y": int(rng.integers(54, 63)),
            "enemy_x": int(rng.integers(60, 86)),
            "enemy_y": int(rng.integers(9, 23)),
            "cursor_x": int(rng.integers(12, 42)),
            "cursor_y": int(rng.integers(36, 62)),
        }

    def __init__(self, level, image_size):
        super().__init__(level, image_size)
        self.position = [float(level["player_x"]), float(level["player_y"])]
        self.enemy = [float(level["enemy_x"]), float(level["enemy_y"])]
        self.cursor = list(self._point(level["cursor_x"], level["cursor_y"]))
        self.hits, self.misses = 0, 0
        self.required_hits = level.get("required_hits", 3)
        self.aim_radius = level.get("aim_radius", 5)
        self._initial_distance = float(np.linalg.norm(np.array(self.position) - self.enemy))

    def step(self, action):
        self._begin(action)
        dx, dy = _directions(action)
        self.position[0] = float(np.clip(self.position[0] + 4 * dx, 4, 92))
        self.position[1] = float(np.clip(self.position[1] + 4 * dy, 4, 64))
        self.cursor[0] = float(np.clip(self.cursor[0] + action.dx, 2, self.size - 3))
        self.cursor[1] = float(np.clip(self.cursor[1] + action.dy, HUD_HEIGHT + 2, self.size - 3))
        target = self._point(*self.enemy)
        on_target = (
            np.hypot(self.cursor[0] - target[0], self.cursor[1] - target[1]) <= self.aim_radius
        )
        distance = float(np.linalg.norm(np.array(self.position) - self.enemy))
        in_range = distance <= 27
        if self._click(action):
            if on_target and in_range:
                self.hits += 1
                self.score = max(self.score, min(4, self.hits + 2))
                if self.hits == self.required_hits:
                    self.status, self.score = "Success", 5
            else:
                self.misses += 1
                if self.misses == 6:
                    self.status = "Failed"
        if self.status == "Play":
            if in_range or distance <= 0.8 * self._initial_distance:
                self.score = max(self.score, 1)
            if on_target and (action.dx or action.dy):
                self.score = max(self.score, 2)
        self._remember(action)

    def render(self):
        image, draw = self._canvas()
        ex, ey = self._point(*self.enemy)
        radius = max(3, self.size / 18)
        if self.level.get("difficulty", 1.0) < 1:
            radius = max(radius, self.aim_radius)
        draw.ellipse((ex - radius, ey - radius, ex + radius, ey + radius), fill=DANGER)
        for index in range(self.required_hits - self.hits):
            x = ex - 5 + index * 4
            draw.rectangle((x, ey - radius - 3, x + 2, ey - radius - 2), fill=SAFE)
        px, py = self._point(*self.position)
        draw.rectangle((px - 2, py - 2, px + 2, py + 2), fill=PLAYER)
        cx, cy = self.cursor
        draw.line((cx - 4, cy, cx + 4, cy), fill=CURSOR)
        draw.line((cx, cy - 4, cx, cy + 4), fill=CURSOR)
        # Red pips are spent shots; they make the limited firing budget visible.
        for index in range(self.misses):
            draw.rectangle((3 + index * 4, self.size - 4, 5 + index * 4, self.size - 2), fill=ENEMY)
        return np.asarray(image, dtype=np.uint8).copy()


class UIGame(_Game):
    """Open inventory, select a visibly requested item, confirm and close it."""

    @classmethod
    def make_level(cls, rng):
        return {
            "request": int(rng.integers(3)),
            "items": rng.permutation(3).tolist(),
            "selection": int(rng.integers(3)),
            "layout": int(rng.integers(2)),
            "request_x": int(rng.choice((16, 32, 48, 64, 80))),
            "menu_offset": int(rng.integers(4)),
        }

    def __init__(self, level, image_size):
        super().__init__(level, image_size)
        self.opened, self.confirmed = False, False
        self.selection = level["selection"]

    def step(self, action):
        self._begin(action)
        newly_opened = False
        if self._key_edge(action, "tab") and not self.confirmed:
            self.opened = not self.opened
            newly_opened = self.opened
            if newly_opened:
                self.score = max(self.score, 1)
        if self.opened:
            horizontal, vertical = _directions(action)
            movement = horizontal if self.level["layout"] == 0 else vertical
            if not self.confirmed:
                self.selection = (self.selection + movement) % len(self.level["items"])
                if (movement and not newly_opened
                        and self.level["items"][self.selection] == self.level["request"]):
                    self.score = max(self.score, 2)
                if self._key_edge(action, "enter"):
                    if self.level["items"][self.selection] == self.level["request"]:
                        self.confirmed, self.score = True, 3
                    else:
                        self.status = "Failed"
            if self.status == "Play" and self._key_edge(action, "escape"):
                self.opened = False
                if self.confirmed:
                    self.status, self.score = "Success", 5
        self._remember(action)

    def _item_rect(self, index):
        offset = self.level["menu_offset"]
        if self.level["layout"] == 0:
            x = 16 + index * 21 if len(self.level["items"]) == 4 else 19 + index * 29
            cx, cy = self._point(x, 35 + offset)
        else:
            y = 21 + index * 14 if len(self.level["items"]) == 4 else 27 + index * 14
            cx, cy = self._point(48, y + offset)
        radius = max(3, self.size / 16) if self.level["layout"] == 0 else max(2, self.size / 32)
        return cx - radius, cy - radius, cx + radius, cy + radius

    def render(self):
        image, draw = self._canvas()
        rx, ry = self._point(self.level["request_x"], 8)
        draw.rectangle((rx - 3, ry - 3, rx + 3, ry + 3), fill=ITEM_COLORS[self.level["request"]])
        if self.opened:
            panel_top = _ui_panel_top(self.size)
            draw.rectangle((3, panel_top, self.size - 4, self.size - 4), fill=PANEL)
            for index, item in enumerate(self.level["items"]):
                box = self._item_rect(index)
                if index == self.selection:
                    border = 2 if self.level["layout"] == 0 else 1
                    draw.rectangle((
                        box[0] - border, box[1] - border, box[2] + border, box[3] + border,
                    ), fill=SELECTED)
                draw.rectangle(box, fill=ITEM_COLORS[item])
            if self.confirmed:
                draw.rectangle((3, panel_top, 6, panel_top + 3), fill=CONFIRMED)
        else:
            # A neutral inventory glyph gives the closed interface a visible affordance.
            x, y = self._point(48, 43)
            draw.rectangle((x - 5, y - 4, x + 5, y + 4), outline=SELECTED)
        return np.asarray(image, dtype=np.uint8).copy()


GAMES = {
    "goal_choice": GoalChoiceGame,
    "hazards": HazardsGame,
    "shooting": ShootingGame,
    "ui": UIGame,
}


class _GoalController:
    def __init__(self, fail):
        self.fail = fail

    def observe_action(self, action: PhysicalAction) -> None:
        pass  # Navigation is inferred from actual RGB positions.

    def act(self, frame):
        player = _center(frame, PLAYER)
        target = _center(frame, DANGER) if self.fail else _center(frame, SAFE)
        if player is None or target is None:
            return PhysicalAction()
        # Move to the safe column before climbing, avoiding the other column's guard.
        if not self.fail and abs(player[0] - target[0]) > 2:
            return PhysicalAction(keys=("d" if target[0] > player[0] else "a",))
        return PhysicalAction(keys=_keys_towards(player, target, 2))


def _grid_cell(frame, color):
    center = _center(frame, color)
    if center is None:
        return None
    size = frame.shape[0]
    xs = np.linspace(3, size - 3, 6, dtype=int)
    ys = np.linspace(HUD_HEIGHT + 2, size - 3, 6, dtype=int)
    return int(np.searchsorted(ys, center[1]) - 1), int(np.searchsorted(xs, center[0]) - 1)


class _HazardController:
    def __init__(self, fail):
        self.fail = fail

    def observe_action(self, action: PhysicalAction) -> None:
        pass  # Navigation is inferred from actual RGB positions.

    def act(self, frame):
        player, target = _grid_cell(frame, PLAYER), _grid_cell(frame, SAFE)
        if player is None or target is None:
            return PhysicalAction()
        size = frame.shape[0]
        traps, blocked = set(), set()
        for row in range(5):
            for column in range(5):
                x = int(3 + (column + 0.5) * (size - 6) / 5)
                y = int(HUD_HEIGHT + 2 + (row + 0.5) * (size - HUD_HEIGHT - 5) / 5)
                color = tuple(frame[y, x])
                if color == DANGER:
                    traps.add((row, column))
                if color in (DANGER, ENEMY, PATROL):
                    blocked.add((row, column))
        if self.fail:
            paths = [_grid_path(player, trap, {target}) for trap in traps]
            paths = [path for path in paths if path]
            path = min(paths, key=len) if paths else None
        else:
            # The colored patrol corridor is a visible warning. Avoid its entire
            # trajectory rather than relying on an engine-state motion prediction.
            path = _grid_path(player, target, blocked)
        if not path:
            return PhysicalAction()
        dr, dc = path[0][0] - player[0], path[0][1] - player[1]
        key = {(-1, 0): "w", (1, 0): "s", (0, -1): "a", (0, 1): "d"}[dr, dc]
        return PhysicalAction(keys=(key,))


class _ShootingController:
    def __init__(self, fail):
        self.fail, self.pressed = fail, False

    def observe_action(self, action: PhysicalAction) -> None:
        self.pressed = "left" in action.mouse_buttons

    def act(self, frame):
        if self.fail:
            self.pressed = not self.pressed
            return PhysicalAction(mouse_buttons=("left",) if self.pressed else ())
        player, target, cursor = (_center(frame, color) for color in (PLAYER, DANGER, CURSOR))
        if player is None or target is None or cursor is None:
            return PhysicalAction()
        size = frame.shape[0]
        distance = np.hypot(
            (player[0] - target[0]) * 96 / size,
            (player[1] - target[1]) * 68 / (size - HUD_HEIGHT),
        )
        keys = _keys_towards(player, target, 2) if distance > 22 else ()
        dx, dy = _mouse_delta(target[0] - cursor[0]), _mouse_delta(target[1] - cursor[1])
        aligned = np.hypot(cursor[0] + dx - target[0], cursor[1] + dy - target[1]) <= 4
        fire = distance <= 24 and aligned and not self.pressed
        self.pressed = fire
        return PhysicalAction(keys=keys, mouse_buttons=("left",) if fire else (), dx=dx, dy=dy)


class _UIController:
    def __init__(self, fail):
        self.fail = fail
        self._previous_keys = ()

    def observe_action(self, action: PhysicalAction) -> None:
        self._previous_keys = action.keys

    def act(self, frame):
        action = self._recommend(frame)
        if any(key in self._previous_keys for key in action.keys
               if key in {"tab", "enter", "escape"}):
            action = PhysicalAction()
        self._previous_keys = action.keys
        return action

    def _recommend(self, frame):
        if _center(frame, CONFIRMED) is not None:
            return PhysicalAction(keys=("escape",))
        if _center(frame, PANEL) is None:
            return PhysicalAction(keys=("tab",))
        panel_top = _ui_panel_top(frame.shape[0])
        request_end = HUD_HEIGHT + 11 if frame.shape[0] == 96 else panel_top
        request_region = (slice(HUD_HEIGHT, request_end), slice(None))
        requested = next(
            (index for index, color in enumerate(ITEM_COLORS)
             if _center(frame, color, request_region) is not None), None,
        )
        if requested is None:
            return PhysicalAction()
        item_region = (slice(panel_top, None), slice(None))
        item_centers = [(_center(frame, color, item_region), index)
                        for index, color in enumerate(ITEM_COLORS)]
        item_centers = [(center, index) for center, index in item_centers if center is not None]
        if len(item_centers) < 3 or requested not in {index for _, index in item_centers}:
            return PhysicalAction()
        white = _center(frame, SELECTED, item_region)
        if white is None:
            return PhysicalAction()
        selection = min(item_centers, key=lambda pair: np.hypot(
            pair[0][0] - white[0], pair[0][1] - white[1],
        ))
        correct = selection[1] == requested
        if correct != self.fail:
            return PhysicalAction(keys=("enter",))
        target = next(center for center, item in item_centers if (item != requested) == self.fail)
        horizontal = max(c[0] for c, _ in item_centers) - min(c[0] for c, _ in item_centers) > 5
        if horizontal:
            return PhysicalAction(keys=("right" if target[0] > selection[0][0] else "left",))
        return PhysicalAction(keys=("down" if target[1] > selection[0][1] else "up",))


def make_controller(task, fail=False):
    """Return an explicit pixel-based reference policy, never the learned agent."""
    factories = {
        "goal_choice": _GoalController,
        "hazards": _HazardController,
        "shooting": _ShootingController,
        "ui": _UIController,
    }
    if task not in factories:
        raise ValueError(f"Unknown combat/interface task: {task}")
    return factories[task](fail)
