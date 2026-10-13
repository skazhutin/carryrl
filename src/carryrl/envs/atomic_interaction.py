"""RGB grid tasks for proximity interaction, causal switches and ordered goals."""

from collections import deque

import numpy as np
from PIL import Image, ImageDraw

from carryrl.agent.actions import PhysicalAction
from carryrl.rewards.visual import HUD_HEIGHT

GRID_SIZE = 5
BACKGROUND = (31, 38, 52)
FLOOR = (187, 198, 212)
WALL = (58, 67, 80)
PLAYER = (35, 180, 235)
HAZARD = (224, 64, 80)
ITEM = (242, 202, 55)
USE_PAD = (172, 96, 215)
BUTTON = (236, 120, 63)
KEY = (255, 220, 40)
SWITCH = (245, 95, 200)
DOOR = (112, 118, 160)
OPEN_DOOR = (123, 165, 184)
EXIT = (69, 207, 100)
SPENT = (113, 147, 106)
COLORS = {
    FLOOR: "floor", WALL: "wall", PLAYER: "player", HAZARD: "hazard", ITEM: "item",
    USE_PAD: "use", BUTTON: "button", KEY: "key", SWITCH: "switch", DOOR: "door",
    OPEN_DOOR: "open_door", EXIT: "exit", SPENT: "spent",
}


def _cells():
    return [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]


def _neighbors(cell):
    r, c = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbor = r + dr, c + dc
        if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
            yield neighbor


def _path(start, goals, blocked):
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        cell, route = queue.popleft()
        if cell in goals:
            return route
        for neighbor in _neighbors(cell):
            if neighbor not in seen and neighbor not in blocked:
                seen.add(neighbor)
                queue.append((neighbor, [*route, neighbor]))
    return None


def _box(size, cell):
    xs = np.linspace(3, size - 3, GRID_SIZE + 1, dtype=int)
    ys = np.linspace(HUD_HEIGHT + 2, size - 3, GRID_SIZE + 1, dtype=int)
    r, c = cell
    return int(xs[c]), int(ys[r]), int(xs[c + 1]), int(ys[r + 1])


def _near(first, second):
    return abs(first[0] - second[0]) + abs(first[1] - second[1]) <= 1


def _transform(cell, rotation, reflect):
    r, c = cell
    if reflect:
        c = GRID_SIZE - 1 - c
    for _ in range(rotation):
        r, c = c, GRID_SIZE - 1 - r
    return [int(r), int(c)]


def _door_level(rng, *, switch):
    """A locked one-cell passage separates the starting room from the exit."""
    while True:
        row = int(rng.integers(GRID_SIZE))
        door, start = (row, 2), (row, 1)
        left = [(r, c) for r in range(GRID_SIZE) for c in (0, 1)]
        right = [(r, c) for r in range(GRID_SIZE) for c in (3, 4)]
        candidates = [cell for cell in left if not _near(cell, start)]
        key = candidates[int(rng.integers(len(candidates)))]
        available = [cell for cell in left if cell not in (start, key)]
        rng.shuffle(available)
        trigger, hazard = available[:2]
        goal = right[int(rng.integers(len(right)))]
        walls = {(r, 2) for r in range(GRID_SIZE) if r != row}
        left_walls = [cell for cell in available[2:] if cell != trigger]
        right_walls = [cell for cell in right if cell != goal]
        walls.add(left_walls[int(rng.integers(len(left_walls)))])
        walls.add(right_walls[int(rng.integers(len(right_walls)))])
        closed = walls | {door, hazard}
        required = (key, trigger) if switch else (key,)
        if any(_path(start, {cell}, closed) is None for cell in required):
            continue
        if _path(start, {goal}, walls | {hazard}) is None:
            continue
        # Failure probes must also be able to enter the visible hazard from the starting room.
        if _path(start, {hazard}, walls | {door}) is None:
            continue
        rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
        level = {
            "walls": [_transform(cell, rotation, reflection) for cell in sorted(walls)],
            "start": _transform(start, rotation, reflection),
            "door": _transform(door, rotation, reflection),
            "key": _transform(key, rotation, reflection),
            "hazard": _transform(hazard, rotation, reflection),
            "exit": _transform(goal, rotation, reflection),
        }
        if switch:
            level["switch"] = _transform(trigger, rotation, reflection)
        return level


class Game:
    """Local game contract: ordinary held movement and rising-edge E interactions."""

    def __init__(self, level: dict, image_size: int):
        self.level, self.image_size = level, image_size
        self.ticks, self.score, self.status = 0, 0, "Play"
        self.position = tuple(level["start"])
        self.walls = frozenset(tuple(cell) for cell in level["walls"])
        self.hazard = tuple(level["hazard"])
        self._previous_keys = ()
        self._approach_target = tuple(level.get("item", level.get("button", level.get("key"))))
        self._approach_blocked = set(self.walls) | {self.hazard}
        if "door" in level:
            self._approach_blocked.add(tuple(level["door"]))
        route = _path(self.position, {self._approach_target}, self._approach_blocked)
        self._approach_distance = len(route) if route is not None else None

    def _blocked(self, destination):
        return destination in self.walls

    def _interact(self):
        raise NotImplementedError

    def _objects(self):
        raise NotImplementedError

    def step(self, action: PhysicalAction):
        if not isinstance(action, PhysicalAction):
            raise TypeError("Game accepts PhysicalAction only")
        if self.status != "Play":
            raise RuntimeError("Cannot step an ended game")
        self.ticks += 1
        keys = set(action.keys)
        horizontal = int(bool(keys & {"right", "d"})) - int(bool(keys & {"left", "a"}))
        vertical = int(bool(keys & {"down", "s"})) - int(bool(keys & {"up", "w"}))
        for dr, dc in ((vertical, 0), (0, horizontal)):
            destination = self.position[0] + dr, self.position[1] + dc
            if (
                0 <= destination[0] < GRID_SIZE and 0 <= destination[1] < GRID_SIZE
                and not self._blocked(destination)
            ):
                self.position = destination
                if destination == self.hazard:
                    self.status = "Failed"
                    break
        if self.status == "Play":
            route = _path(self.position, {self._approach_target}, self._approach_blocked)
            if (route is not None and self._approach_distance is not None
                    and len(route) < self._approach_distance):
                self.score = max(self.score, 1)
            if "e" in keys and "e" not in self._previous_keys:
                self._interact()
        self._previous_keys = action.keys
        self._finish()

    def _finish(self):
        pass

    def render(self):
        image = Image.new("RGB", (self.image_size, self.image_size), BACKGROUND)
        draw = ImageDraw.Draw(image)
        objects = self._objects()
        objects[self.hazard] = HAZARD
        for cell in _cells():
            color = WALL if cell in self.walls else objects.get(cell, FLOOR)
            x0, y0, x1, y1 = _box(self.image_size, cell)
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=color)
            if cell == self.position:
                # Keep a one-pixel object rim visible when the player occupies its cell.
                draw.rectangle((x0 + 1, y0 + 1, x1 - 2, y1 - 2), fill=PLAYER)
        return np.asarray(image, dtype=np.uint8).copy()


class Interaction(Game):
    """Approach and press E to pick up an item, then use it at another location."""

    @classmethod
    def make_level(cls, rng):
        cells = _cells()
        while True:
            indices = rng.choice(len(cells), size=7, replace=False)
            start, item, use, hazard, *walls = [cells[int(index)] for index in indices]
            if _near(start, item) or abs(item[0] - use[0]) + abs(item[1] - use[1]) < 3:
                continue
            blocked = set(walls) | {hazard}
            if any(_path(start, {cell}, blocked) is None for cell in (item, use)):
                continue
            if _path(start, {hazard}, set(walls)) is None:
                continue
            return {
                "start": list(start), "item": list(item), "use": list(use),
                "hazard": list(hazard), "walls": [list(cell) for cell in walls],
            }

    def __init__(self, level, image_size):
        super().__init__(level, image_size)
        self.item, self.use = tuple(level["item"]), tuple(level["use"])
        self.carrying = False

    def _interact(self):
        if not self.carrying and _near(self.position, self.item):
            self.carrying, self.score = True, 2
        elif self.carrying and _near(self.position, self.use):
            self.status, self.score = "Success", 5

    def _objects(self):
        objects = {self.use: USE_PAD}
        if not self.carrying:
            objects[self.item] = ITEM
        return objects


class KeyDoor(Game):
    """Collect a key away from a locked passage, return, unlock and reach the exit."""

    @classmethod
    def make_level(cls, rng):
        return _door_level(rng, switch=False)

    def __init__(self, level, image_size):
        super().__init__(level, image_size)
        self.key, self.door, self.exit = (
            tuple(level[name]) for name in ("key", "door", "exit")
        )
        self.has_key, self.door_open = False, False

    def _blocked(self, destination):
        return super()._blocked(destination) or (destination == self.door and not self.door_open)

    def _interact(self):
        if not self.has_key and _near(self.position, self.key):
            self.has_key, self.score = True, 2
        elif self.has_key and not self.door_open and _near(self.position, self.door):
            self.door_open, self.score = True, 3

    def _objects(self):
        objects = {self.door: OPEN_DOOR if self.door_open else DOOR, self.exit: EXIT}
        if not self.has_key:
            objects[self.key] = KEY
        return objects

    def _finish(self):
        if self.status == "Play" and self.door_open and self.position == self.exit:
            self.status, self.score = "Success", 5


class Causality(KeyDoor):
    """A remote button opens a physically blocked passage to the exit."""

    @classmethod
    def make_level(cls, rng):
        level = _door_level(rng, switch=False)
        level["button"] = level.pop("key")
        return level

    def __init__(self, level, image_size):
        Game.__init__(self, level, image_size)
        self.button, self.door, self.exit = (
            tuple(level[name]) for name in ("button", "door", "exit")
        )
        self.door_open = False

    def _interact(self):
        if not self.door_open and _near(self.position, self.button):
            self.door_open, self.score = True, 2

    def _objects(self):
        return {
            self.button: SPENT if self.door_open else BUTTON,
            self.door: OPEN_DOOR if self.door_open else DOOR, self.exit: EXIT,
        }


class Chain(KeyDoor):
    """The key enables a remote switch; the switch then enables door unlocking."""

    @classmethod
    def make_level(cls, rng):
        return _door_level(rng, switch=True)

    def __init__(self, level, image_size):
        super().__init__(level, image_size)
        self.switch = tuple(level["switch"])
        self.switch_on = False

    def _interact(self):
        if not self.has_key and _near(self.position, self.key):
            self.has_key, self.score = True, 2
        elif self.has_key and not self.switch_on and _near(self.position, self.switch):
            self.switch_on, self.score = True, 3
        elif self.switch_on and not self.door_open and _near(self.position, self.door):
            self.door_open, self.score = True, 4

    def _objects(self):
        objects = super()._objects()
        objects[self.switch] = SPENT if self.switch_on else SWITCH
        return objects


GAMES = {"interaction": Interaction, "causality": Causality, "key_door": KeyDoor, "chain": Chain}


class _VisualController:
    """Scripted bootstrap probe that reads rendered cells and its own E history."""

    def __init__(self, task, fail):
        self.task, self.fail = task, fail
        self._map, self._previous_e = {}, False

    def observe_action(self, action: PhysicalAction) -> None:
        self._previous_e = "e" in action.keys

    def act(self, frame):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise TypeError("Visual controller requires a uint8 RGB frame")
        if frame.ndim != 3 or frame.shape[0] != frame.shape[1] or frame.shape[2] != 3:
            raise ValueError("Visual controller requires a square RGB frame")
        position = None
        for cell in _cells():
            x0, y0, x1, y1 = _box(frame.shape[0], cell)
            kind = COLORS.get(tuple(frame[(y0 + y1 - 1) // 2, (x0 + x1 - 1) // 2]))
            if kind == "player":
                position = cell
                underlying = COLORS.get(tuple(frame[y0, x0]))
                if underlying is None or underlying == "player":
                    raise ValueError("Unrecognized object pixels beneath the player")
                self._map[cell] = underlying
            elif kind is not None:
                self._map[cell] = kind
            else:
                raise ValueError("Unrecognized interaction-game cell pixels")
        if position is None:
            raise ValueError("Player is not visible")
        if self._previous_e:
            self._previous_e = False
            return PhysicalAction()
        present = set(self._map.values())
        if self.fail:
            objective, interact = "hazard", False
        elif self.task == "interaction":
            objective, interact = ("item" if "item" in present else "use"), True
        elif self.task == "causality":
            objective = "button" if "button" in present else "exit"
            interact = objective == "button"
        else:
            if "key" in present:
                objective = "key"
            elif self.task == "chain" and "switch" in present:
                objective = "switch"
            else:
                objective = "door" if "door" in present else "exit"
            interact = objective != "exit"
        targets = {cell for cell, kind in self._map.items() if kind == objective}
        if not targets:
            raise RuntimeError("The required visual target is absent")
        if interact and any(_near(position, target) for target in targets):
            self._previous_e = True
            return PhysicalAction(keys=("e",))
        blocked = {
            cell for cell, kind in self._map.items() if kind in {"wall", "door", "hazard"}
        }
        if self.fail:
            blocked -= targets
        goals = {neighbor for target in targets for neighbor in _neighbors(target)} \
            if interact else targets
        goals -= blocked
        route = _path(position, goals, blocked)
        if not route:
            raise RuntimeError("The visual target is unreachable")
        dr, dc = route[0][0] - position[0], route[0][1] - position[1]
        key = {(1, 0): "s", (-1, 0): "w", (0, 1): "d", (0, -1): "a"}[dr, dc]
        return PhysicalAction(keys=(key,))


def make_controller(task, fail=False):
    if task not in GAMES:
        raise ValueError(f"Unknown interaction task: {task}")
    return _VisualController(task, fail)
