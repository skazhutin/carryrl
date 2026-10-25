"""RGB-only diagnostic supervision for the four canonical motion fixtures.

These suite-specific targets are observable geometry, not reward, win probability,
controller actions, or policy inputs. Raster quantization and invisible finish
conditions deliberately leave some labels unknown. One annotator owns one episode.
"""

import numpy as np
from PIL import Image, ImageDraw

from carryrl.envs.atomic_motion import BACKGROUND, BALL, FLOOR, HUD_HEIGHT, PLAYER, TARGET, WALL

SUPPORTED_TASKS = ("timing", "platforming", "steering", "camera")
EVENT_NAMES = ("pipe_passed", "jump_started", "landed", "obstacle_passed", "door_aligned")
GOALS = {
    "timing": "Keep the bird inside the visible pipe openings and pass the pipes.",
    "platforming": "Jump over the obstacles and reach the green goal.",
    "steering": "Keep the car aligned with the visible winding road.",
    "camera": "Center the doorway with a level view and approach it.",
}
_TASK_EVENTS = {
    "timing": {"pipe_passed"},
    "platforming": {"jump_started", "landed", "obstacle_passed"},
    "steering": set(),
    "camera": {"door_aligned"},
}
_PALETTE = (BACKGROUND, BALL, FLOOR, PLAYER, TARGET, WALL)


def _mask(scene, color):
    return np.all(scene == color, axis=-1)


def _box(mask):
    yy, xx = np.where(mask)
    if not len(xx):
        raise ValueError("required visible object missing")
    return int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())


def _shape(shape, box, ellipse=False):
    image = Image.new("1", (shape[1], shape[0]))
    draw = ImageDraw.Draw(image)
    (draw.ellipse if ellipse else draw.rectangle)(box, fill=1)
    return np.asarray(image, dtype=bool)


def _groups(columns):
    return [group for group in np.split(columns, np.where(np.diff(columns) > 1)[0] + 1)
            if len(group)]


def _unit(scene, point):
    return [float(point[0]) * 96 / (scene.shape[1] - 1),
            float(point[1]) * 68 / (scene.shape[0] - 1)]


def _point(scene, x, y):
    return round(x * (scene.shape[1] - 1) / 96), round(y * (scene.shape[0] - 1) / 68)


def _player(scene, *, rectangle=False):
    mask = _mask(scene, PLAYER)
    box = _box(mask)
    if not np.array_equal(mask, _shape(mask.shape, box, ellipse=not rectangle)):
        raise ValueError("noncanonical player geometry")
    position = _unit(scene, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    width, height = _unit(scene, (box[2] - box[0], box[3] - box[1]))
    if not 4 <= width <= 8 or not (6 <= height <= 11 if rectangle else 4 <= height <= 8):
        raise ValueError("unrecognized player size")
    return mask, box, position


class MotionSceneAnnotator:
    """Read canonical 64/96/128 RGB frames; HUD and action proposals are ignored.

    ``previous_action`` may be the actual past command; geometry alone supplies
    these labels, so holding or repeating a command never creates an event.
    Unknown observations invalidate the next transition edge.
    """

    def __init__(self, task):
        self.task = task
        self._previous = None
        self._gap = True
        self._static = None
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
        """Return JSON-compatible, partially valid labels from observed pixels only."""
        if self.task not in SUPPORTED_TASKS:
            return self._unknown("unsupported task")
        try:
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
                raise ValueError("expected uint8 RGB frame")
            if frame.shape not in ((64, 64, 3), (96, 96, 3), (128, 128, 3)):
                raise ValueError("unsupported RGB geometry")
            if self._size is not None and self._size != frame.shape:
                raise ValueError("changed image size within episode")
            scene = frame[HUD_HEIGHT:]
            palette = np.stack([_mask(scene, color) for color in _PALETTE])
            if not palette.any(axis=0).all():
                raise ValueError("noncanonical scene palette")
            state, potential, success, success_valid = getattr(self, "_" + self.task)(scene)
            edge = self._previous is not None and not self._gap
            events = self._events(state) if edge else []
        except (ValueError, TypeError, IndexError) as error:
            return self._unknown(str(error))
        self._previous, self._gap, self._size = state, False, frame.shape
        return {
            "known": True, "potential": potential, "potential_valid": potential is not None,
            "events": events,
            "event_valid": {event: edge and event in _TASK_EVENTS[self.task]
                            for event in EVENT_NAMES},
            "event_targets": {event: event in events for event in EVENT_NAMES},
            "visible_state": state, "terminal_success": success,
            "terminal_success_valid": success_valid,
            "reason": "canonical visible geometry; finish may be unobservable",
        }

    def _timing(self, scene):
        player, _, position = _player(scene)
        if abs(position[0] - 22) > 1.7:
            raise ValueError("bird outside canonical horizontal band")
        if _mask(scene, FLOOR).any() or _mask(scene, TARGET).any() or _mask(scene, BALL).any():
            raise ValueError("unexpected timing objects")
        walls = _mask(scene, WALL)
        pipes = []
        for group in _groups(np.flatnonzero(walls.any(axis=0))):
            left, right = int(group[0]), int(group[-1])
            row_pattern = walls[:, left:right + 1].any(axis=1)
            # Every column must show a top and bottom solid pipe, separated by
            # one opening. The bird may occlude a pipe edge during collision.
            top = np.flatnonzero(walls[:, left])[0]
            bottom = np.flatnonzero(walls[:, left])[-1]
            if top != 0 or bottom != len(scene) - 1:
                raise ValueError("pipe does not span playfield")
            gap_rows = np.flatnonzero(~row_pattern)
            if not len(gap_rows) or not np.all(np.diff(gap_rows) == 1):
                raise ValueError("ambiguous pipe opening")
            opening = _unit(scene, (0, gap_rows[-1] - gap_rows[0]))[1]
            if not 24 <= opening <= 41:
                raise ValueError("unrecognized pipe opening")
            expected = row_pattern[:, None]
            if np.any((walls[:, left:right + 1] != expected) & ~player[:, left:right + 1]):
                raise ValueError("nonrectangular pipe")
            width = _unit(scene, (right - left, 0))[0]
            if width > 12 or (left > 0 and right < scene.shape[1] - 1 and width < 8):
                raise ValueError("unrecognized pipe width")
            pipes.append({"left": _unit(scene, (left, 0))[0],
                          "right": _unit(scene, (right, 0))[0],
                          "gap_y": _unit(scene, (0, (gap_rows[0] + gap_rows[-1]) / 2))[1]})
        if not pipes:
            raise ValueError("no visible pipe geometry")
        upcoming = [pipe for pipe in pipes if pipe["right"] >= 19]
        target = upcoming[0] if upcoming else None
        alignment = max(0.0, 1 - abs(position[1] - target["gap_y"]) / 34) if target else None
        return {"player": position, "pipes": pipes, "alignment": alignment}, alignment, None, False

    def _platforming(self, scene):
        player, player_box, position = _player(scene)
        if _mask(scene, BALL).any():
            raise ValueError("unexpected platforming object")
        if self._static is None:
            floor = _mask(scene, FLOOR)
            floor_y = _point(scene, 0, 56)[1]
            expected_floor = np.zeros(floor.shape, dtype=bool)
            expected_floor[floor_y:] = True
            walls = _mask(scene, WALL)
            target = _mask(scene, TARGET)
            goal_box = _box(target)
            if not np.array_equal(target, _shape(target.shape, goal_box)):
                raise ValueError("ambiguous goal rectangle")
            goal_width = _unit(scene, (goal_box[2] - goal_box[0], 0))[0]
            if not 2.5 <= goal_width <= 6:
                raise ValueError("unrecognized goal width")
            goal = _unit(scene, ((goal_box[0] + goal_box[2]) / 2, 0))[0]
            if not 85 <= goal <= 94 or goal_box[1:] != (
                _point(scene, 0, 38)[1], goal_box[2], _point(scene, 0, 55)[1]
            ):
                raise ValueError("unrecognized goal geometry")
            obstacles = []
            for group in _groups(np.flatnonzero(walls.any(axis=0))):
                box = _box(walls[:, group[0]:group[-1] + 1])
                box = (int(group[0]), box[1], int(group[-1]), box[3])
                rectangle = _shape(walls.shape, box)
                if not np.array_equal(walls & rectangle, rectangle) or box[3] != floor_y:
                    raise ValueError("unrecognized obstacle geometry")
                width, height = _unit(scene, (box[2] - box[0], box[3] - box[1]))
                if not 3.5 <= width <= 10 or not 5 <= height <= 12:
                    raise ValueError("unrecognized obstacle size")
                obstacles.append(_unit(scene, (box[2], 0))[0])
                expected_floor[rectangle] = False
            if len(obstacles) != 2 or not np.array_equal(floor & ~player, expected_floor & ~player):
                raise ValueError("complete canonical platform layout required")
            static = scene.copy()
            static[player] = np.asarray(BACKGROUND, dtype=np.uint8)
            static[floor_y:][player[floor_y:]] = FLOOR
            self._static = (static, obstacles, goal)
        static, obstacles, goal = self._static
        if not np.array_equal(scene[~player], static[~player]):
            raise ValueError("changed platform geometry")
        grounded = player_box[3] == _point(scene, 0, 56)[1]
        potential = float(np.clip(1 - abs(goal - 3 - position[0]) / 96, 0, 1))
        # The integer goal is rendered at finite precision. Only a clearly
        # separated player position certifies either side of the finish band.
        tolerance = 96 / (scene.shape[1] - 1)
        distance = position[0] - (goal - 3)
        valid = abs(distance) > tolerance
        success = distance > tolerance if valid else None
        state = {"player": position, "goal_x": goal, "obstacle_rights": obstacles,
                 "grounded": grounded}
        return state, potential, success, valid

    def _steering(self, scene):
        player, _, position = _player(scene, rectangle=True)
        if _mask(scene, WALL).any() or _mask(scene, TARGET).any() or _mask(scene, BALL).any():
            raise ValueError("unexpected road objects")
        if abs(position[1] - 50) > 1.5:
            raise ValueError("car outside canonical vertical band")
        floor = _mask(scene, FLOOR)
        intervals = {}
        rendered_rows = {_point(scene, 0, y)[1] for y in range(69)}
        for row in range(len(scene)):
            if row not in rendered_rows:
                if floor[row].any():
                    raise ValueError("road outside canonical raster rows")
                continue
            columns = np.flatnonzero(floor[row])
            if not len(columns):
                raise ValueError("road row missing")
            left, right = int(columns[0]), int(columns[-1])
            if not (floor[row] | player[row])[left:right + 1].all():
                raise ValueError("fragmented road")
            width = (right - left) * 96 / (scene.shape[1] - 1)
            if not player[row].any() and not 21 <= width <= 30:
                raise ValueError("unrecognized road width")
            intervals[row] = (left, right)
        row = _point(scene, 0, 43)[1]
        left, right = intervals[row]
        center = (left + right) * 48 / (scene.shape[1] - 1)
        half_width = (right - left) * 48 / (scene.shape[1] - 1)
        alignment = float(np.clip(1 - abs(position[0] - center) / half_width, 0, 1))
        return {"player": position, "road_center_x": center, "alignment": alignment}, (
            alignment
        ), None, False

    def _camera(self, scene):
        yy, xx = _point(scene, 48, 32)[1], _point(scene, 48, 32)[0]
        cyan, white = _mask(scene, PLAYER), _mask(scene, BALL)
        reticle = np.zeros(scene.shape[:2], dtype=bool)
        x0, y0 = _point(scene, 46, 30)
        x1, y1 = _point(scene, 50, 34)
        reticle[yy, x0:x1 + 1] = True
        reticle[y0:y1 + 1, xx] = True
        if not (np.array_equal(cyan, reticle) and not white.any()
                or np.array_equal(white, reticle) and not cyan.any()):
            raise ValueError("unrecognized camera reticle")
        walls, floor = _mask(scene, WALL), _mask(scene, FLOOR)
        if not walls.any() or not floor.any():
            raise ValueError("camera perspective missing")
        # Canonical perspective has a solid horizon and at least three long
        # horizontal floor lines; a target/reticle alone is insufficient.
        horizon = np.flatnonzero((walls | floor).any(axis=1))[0]
        if not 8 <= _unit(scene, (0, horizon))[1] <= 30:
            raise ValueError("unrecognized camera horizon")
        if _mask(scene, BACKGROUND)[horizon:].any():
            raise ValueError("noncanonical camera ground plane")
        if np.count_nonzero(floor.sum(axis=1) > scene.shape[1] * 0.8) < 2:
            raise ValueError("camera floor geometry missing")
        target = _mask(scene, TARGET)
        aligned = bool(cyan.any())
        if not target.any():
            if aligned:
                raise ValueError("aligned reticle without doorway")
            return {"door": None, "aligned": False}, None, None, False
        box = _box(target)
        expected = _shape(target.shape, box) & ~reticle
        if not np.array_equal(target, expected):
            raise ValueError("ambiguous doorway geometry")
        center = _unit(scene, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        width, height = _unit(scene, (box[2] - box[0], box[3] - box[1]))
        if not 4 <= width <= 27 or not 4 <= height <= 27 or abs(width - height) > 3.5:
            raise ValueError("unrecognized doorway size")
        if aligned and (abs(center[0] - 48) > 8 or abs(center[1] - 32) > 4):
            raise ValueError("reticle and doorway alignment disagree")
        approach = min(width / 24, 1.0)
        alignment = max(0.0, 1 - abs(center[0] - 48) / 48 - abs(center[1] - 32) / 16)
        potential = float(approach * alignment)
        return {"door": center, "door_width": width, "aligned": aligned,
                "alignment": alignment, "approach": approach}, potential, None, False

    def _events(self, state):
        before = self._previous
        events = []
        if self.task == "timing":
            for pipe in state["pipes"]:
                matches = [old for old in before["pipes"]
                           if 0 <= old["right"] - pipe["right"] <= 5
                           and abs(old["gap_y"] - pipe["gap_y"]) <= 1.1]
                if len(matches) == 1 and matches[0]["right"] >= 19 > pipe["right"]:
                    events.append("pipe_passed")
        elif self.task == "platforming":
            if before["grounded"] and state["player"][1] < before["player"][1] - 1.5:
                events.append("jump_started")
            if not before["grounded"] and state["grounded"]:
                events.append("landed")
            if any(before["player"][0] - 3 <= right < state["player"][0] - 3
                   for right in state["obstacle_rights"]):
                events.append("obstacle_passed")
        elif self.task == "camera" and not before["aligned"] and state["aligned"]:
            events.append("door_aligned")
        return events
