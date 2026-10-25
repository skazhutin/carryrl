"""Suite-specific RGB diagnostics for core scenes, never policy input or reward.

Potential is bounded geometric goal proximity, not score or probability of a win.
Only canonical 64/96/128 scenes are understood. HUD pixels are ignored. Each
annotator belongs to one episode; action arguments must be commands actually run.
"""

from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from carryrl.envs.atomic import BACKGROUND, BALL, DOOR, FLOOR, PLAYER, TARGET, WALL, _cell
from carryrl.rewards.visual import HUD_HEIGHT

SUPPORTED_TASKS = ("navigation", "reaction", "mouse", "memory")
EVENT_NAMES = ("exit_reached", "ball_caught", "target_clicked", "cue_disappeared",
               "memory_unlocked")
GOALS = {
    "navigation": "Reach the green destination through the visible grid.",
    "reaction": "Align the paddle with the falling ball and catch it.",
    "mouse": "Move the cursor onto each green target and click it.",
    "memory": "Remember the initial green door, wait for Choose, then enter that door.",
}
_TASK_EVENTS = {"navigation": {"exit_reached"}, "reaction": {"ball_caught"},
                "mouse": {"target_clicked"},
                "memory": {"cue_disappeared", "memory_unlocked", "exit_reached"}}


def _canvas(size):
    image = Image.new("RGB", (size, size), BACKGROUND)
    return image, ImageDraw.Draw(image)


def _same(frame, image):
    return np.array_equal(frame[HUD_HEIGHT:], np.asarray(image)[HUD_HEIGHT:])


def _bounds(frame, color):
    ys, xs = np.where(np.all(frame == color, axis=-1))
    if not len(xs):
        raise ValueError("required scene object missing")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _point(frame, color):
    x0, y0, x1, y1 = _bounds(frame, color)
    return (x0 + x1) / 2, (y0 + y1) / 2


def _distance(start, goal, walls):
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        cell, distance = queue.popleft()
        if cell == goal:
            return distance
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            neighbor = cell[0] + dr, cell[1] + dc
            if (0 <= neighbor[0] < 5 and 0 <= neighbor[1] < 5
                    and neighbor not in walls and neighbor not in seen):
                queue.append((neighbor, distance + 1))
                seen.add(neighbor)
    raise ValueError("visible goal unreachable")


class CoreSceneAnnotator:
    """Read canonical scene pixels and retained visual history, one episode each."""

    def __init__(self, task):
        self.task = task
        self._previous = None
        self._gap = True
        self._walls = self._goal = self._cue = None
        self._buttons = None
        self._exited = False

    def _unknown(self, reason):
        self._gap = True
        self._buttons = None
        return {"known": False, "potential": None, "potential_valid": False,
                "events": [], "event_valid": dict.fromkeys(EVENT_NAMES, False),
                "event_targets": dict.fromkeys(EVENT_NAMES, False), "visible_state": {},
                "terminal_success": None, "terminal_success_valid": False, "reason": reason}

    def observe(self, frame, previous_action=None):
        """Annotate a frame; absent executed action disables action-dependent edges."""
        if self.task not in SUPPORTED_TASKS:
            return self._unknown("unsupported task")
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.shape not in ((64, 64, 3), (96, 96, 3), (128, 128, 3))):
            return self._unknown("expected canonical uint8 RGB64/96/128")
        # Copy and replace the entire HUD before any object recognition.
        frame = frame.copy()
        frame[:HUD_HEIGHT] = BACKGROUND
        transition = self._previous is not None and not self._gap
        try:
            state, potential, success, events, valid = getattr(self, "_" + self.task)(
                frame, transition, previous_action)
        except (ValueError, TypeError, IndexError, AttributeError) as error:
            return self._unknown(str(error))
        self._previous, self._gap = state, False
        if previous_action is not None:
            self._buttons = tuple(previous_action.mouse_buttons)
        else:
            self._buttons = None
        return {"known": True, "potential": potential, "potential_valid": potential is not None,
                "events": events,
                "event_valid": {name: name in valid for name in EVENT_NAMES},
                "event_targets": {name: name in events for name in EVENT_NAMES},
                "visible_state": state, "terminal_success": success,
                "terminal_success_valid": success is not None,
                "reason": "canonical scene and retained RGB history"}

    def _navigation(self, frame, transition, action):
        size, walls, players, goals = len(frame), set(), [], []
        image, draw = _canvas(size)
        for row in range(5):
            for column in range(5):
                cell = row, column
                x0, y0, x1, y1 = _cell(size, *cell)
                color = tuple(frame[(y0 + y1) // 2, (x0 + x1) // 2])
                if color not in (WALL, FLOOR, PLAYER, TARGET):
                    raise ValueError("unknown grid cell")
                if color == WALL:
                    walls.add(cell)
                elif color == PLAYER:
                    players.append(cell)
                elif color == TARGET:
                    goals.append(cell)
                draw.rectangle((x0 + 1, y0 + 1, x1 - 2, y1 - 2), fill=color)
        if not _same(frame, image) or len(players) != 1 or len(goals) > 1:
            raise ValueError("noncanonical or ambiguous grid")
        player = players[0]
        goal = goals[0] if goals else self._goal
        if goal is None or (not goals and player != goal):
            raise ValueError("goal hidden without visual history")
        if self._walls is not None and (walls != self._walls or goal != self._goal):
            raise ValueError("grid topology changed")
        if transition:
            before = self._previous["player"]
            corner = player[0], before[1]
            if (abs(player[0] - before[0]) > 1 or abs(player[1] - before[1]) > 1
                    or corner in walls or (self._exited and list(player) != before)):
                raise ValueError("nonphysical grid transition")
        distance = _distance(player, goal, walls)
        success = player == goal
        events = ["exit_reached"] if transition and success and not self._exited else []
        self._walls, self._goal = walls, goal
        self._exited |= success
        return ({"player": list(player), "goal": list(goal), "distance": distance},
                float(1 - distance / 25), success, events,
                {"exit_reached"} if transition else set())

    def _reaction(self, frame, transition, action):
        size = len(frame)
        px0, py0, px1, py1 = _bounds(frame, PLAYER)
        if (py0, py1) != (size - 11, size - 7):
            raise ValueError("noncanonical paddle")
        radius = (px1 - px0) / 2
        if px0 == 0 or px1 == size - 1:
            if not transition:
                raise ValueError("clipped paddle radius requires visual history")
            radius = self._previous["paddle_radius"]
        if not 10 <= radius <= 18:
            raise ValueError("noncanonical paddle radius")
        paddle = (px1-radius if px0 == 0 else px0+radius)
        bx0, by0, bx1, by1 = _bounds(frame, BALL)
        matches = []
        # The paddle can cover the lower half of the ball on the last frame.
        for x in range(max(0, bx0 - 3), min(size, bx1 + 4)):
            y = by0 + 3
            image, draw = _canvas(size)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=BALL)
            draw.rectangle((paddle-radius, py0, paddle+radius, py1), fill=PLAYER)
            if _same(frame, image):
                matches.append((x, y))
        if len(matches) != 1:
            raise ValueError("noncanonical or ambiguous ball")
        x, y = matches[0]
        previous = self._previous if transition else None
        drop = 2.5 * (size - HUD_HEIGHT - 17) / (96 - HUD_HEIGHT - 17)
        reset = previous is not None and y < previous["ball"][1] - 3
        at_bottom = y >= size - 12
        separation = abs(x - paddle)
        # Rendering rounds coordinates down. Near the physical acceptance edge,
        # pixels cannot establish which side the subpixel centers occupied.
        bottom_success = (True if separation <= radius + 1 else
                          False if separation > radius + 3 else None)
        caught = bool(previous is not None and (
            (reset and y == HUD_HEIGHT + 5
             and previous["ball"][1] >= size - 13 - drop
             and abs(previous["ball"][0]-paddle) <= radius+5)
            or (at_bottom and previous["ball"][1] < size - 12
                and bottom_success is True)))
        if reset and not caught:
            raise ValueError("ball reset outside a visible catch")
        if previous and not reset and (abs(y - previous["ball"][1]) > drop + 1.1
                                      or abs(x - previous["ball"][0]) > 3):
            raise ValueError("nonphysical ball transition")
        success = bottom_success if at_bottom else None
        # Proximity to the currently visible interception line; no elapsed-time term.
        potential = float(max(0, 1 - abs(x - paddle) / size) * .95)
        if success:
            potential = 1.0
        return ({"ball": [x, y], "paddle_x": paddle, "paddle_radius": radius},
                potential, success, ["ball_caught"] if caught else [],
                {"ball_caught"} if transition and not (at_bottom and success is None)
                else set())

    def _mouse(self, frame, transition, action):
        size = len(frame)
        cx0, cy0, cx1, cy1 = _bounds(frame, PLAYER)
        cursor = cx0 + 3, cy0 + 3
        matches = []
        # Match complete rendering, including cursor occlusion and visible miss marks.
        miss_color = (234, 82, 84)
        miss_mask = np.all(frame == miss_color, axis=-1)
        for row in range(5):
            for column in range(5):
                x = 10 + column * (size - 20) / 4
                y = HUD_HEIGHT + 10 + row * (size - HUD_HEIGHT - 20) / 4
                for radius in range(7, 12):
                    image, draw = _canvas(size)
                    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=TARGET)
                    cx, cy = cursor
                    draw.line((cx-3, cy, cx+3, cy), fill=PLAYER, width=2)
                    draw.line((cx, cy-3, cx, cy+3), fill=PLAYER, width=2)
                    expected = np.asarray(image).copy()
                    for index in range(5):
                        x0 = 3 + index * 5
                        patch = miss_mask[size-4:size-1, x0:x0+3]
                        if patch.all():
                            expected[size-4:size-1, x0:x0+3] = miss_color
                    if np.array_equal(frame[HUD_HEIGHT:], expected[HUD_HEIGHT:]):
                        matches.append(((x, y), radius + 1))
        if len(matches) != 1:
            raise ValueError("noncanonical or ambiguous cursor/target")
        target, radius = matches[0]
        distance = float(np.hypot(cursor[0]-target[0], cursor[1]-target[1]))
        changed = transition and list(target) != self._previous["target"]
        buttons = None if action is None else tuple(action.mouse_buttons)
        click_known = transition and buttons is not None and self._buttons is not None
        click = click_known and "left" in buttons and "left" not in self._buttons
        # A target change proves a hit. An unchanged final target needs the
        # executed rising edge and an unambiguous geometric acceptance decision.
        # Rasterized cursor coordinates are floors of possibly subpixel centers.
        # Bound distance over that entire pixel square instead of turning the
        # uncertain rim into a supervised negative.
        minimum = np.hypot(*(max(low-high, high-(low+1), 0)
                             for low, high in zip(cursor, target, strict=True)))
        maximum = np.hypot(*(max(abs(low-high), abs(low+1-high))
                             for low, high in zip(cursor, target, strict=True)))
        acceptance_known = maximum <= radius or minimum > radius
        hit = bool(changed or (click and maximum <= radius))
        event_known = bool(changed or (click_known and (not click or acceptance_known)))
        if changed and click_known and not click:
            raise ValueError("target changed without executed click edge")
        potential = float(.95 * max(0, 1 - distance / (size * 2 ** .5)))
        return ({"cursor": list(cursor), "target": list(target), "distance": distance},
                potential, None, ["target_clicked"] if hit else [],
                {"target_clicked"} if transition and event_known else set())

    def _memory(self, frame, transition, action):
        size = len(frame)
        door_mask = np.all(frame == DOOR, axis=-1) | np.all(frame == TARGET, axis=-1)
        ys, xs = np.where(door_mask)
        if not len(xs) or ys.min() != size-24 or ys.max() != size-5:
            raise ValueError("memory doors missing")
        left_xs, right_xs = xs[xs < size/2], xs[xs > size/2]
        if not len(left_xs) or not len(right_xs):
            raise ValueError("memory door ambiguous")
        doors = [int(left_xs.min())+7, int(right_xs.min())+7]
        px0, py0, px1, py1 = _bounds(frame, PLAYER)
        if (py0, py1) != (size-18, size-12) or px1-px0 != 6:
            raise ValueError("noncanonical memory player")
        player = px0+3
        matches = []
        for cue in (None, 0, 1):
            for phase in ("Wait", "Choose"):
                image, draw = _canvas(size)
                for index, x in enumerate(doors):
                    draw.rectangle((x-7, size-24, x+7, size-5),
                                   fill=TARGET if index == cue else DOOR)
                draw.ellipse((player-3, size-18, player+3, size-12), fill=PLAYER)
                draw.text((size//2-18, HUD_HEIGHT+7), phase, fill=BALL,
                          font=ImageFont.load_default(size=10))
                if _same(frame, image):
                    matches.append((cue, phase))
        if len(matches) != 1:
            raise ValueError("noncanonical memory cue or phase")
        cue, phase = matches[0]
        if cue is not None and self._cue is not None and cue != self._cue:
            raise ValueError("memory cue changed")
        if self._cue is None and cue is None:
            raise ValueError("initial visual cue absent from history")
        remembered = self._cue if cue is None else cue
        if transition and ((self._previous["phase"] == "Choose" and phase == "Wait")
                           or (not self._previous["cue_visible"] and cue is not None)
                           or doors != self._previous["doors"]
                           or abs(player-self._previous["player_x"]) > 8
                           or (phase == "Wait" and player != self._previous["player_x"])):
            raise ValueError("nonphysical memory transition")
        events = []
        if transition:
            if self._previous["cue_visible"] and cue is None:
                events.append("cue_disappeared")
            if self._previous["phase"] == "Wait" and phase == "Choose":
                events.append("memory_unlocked")
        reached = phase == "Choose" and (player <= doors[0]+6 or player >= doors[1]-6)
        correct = player <= doors[0]+6 if remembered == 0 else player >= doors[1]-6
        success = bool(correct) if reached else False
        if success and transition and not self._exited:
            events.append("exit_reached")
        self._exited |= success
        self._cue = remembered
        distance = max(0, abs(player-doors[remembered])-6)
        # Necessary waiting holds a constant geometric potential, with no tick term.
        potential = 1.0 if success else float(.95 * max(0, 1-distance/size))
        state = {"doors": doors, "player_x": player, "remembered_door": remembered,
                 "cue_visible": cue is not None, "phase": phase, "distance": distance}
        return state, potential, success, events, _TASK_EVENTS["memory"] if transition else set()
