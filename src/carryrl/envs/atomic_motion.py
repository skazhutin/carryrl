"""Small motion games with real temporal dynamics and one physical input codec.

The first-person camera game is a geometric technical stand, not a full 3D
engine. Diagnostic controllers below read rendered pixels and their own history;
they are not policies, demonstrations, or privileged environment controllers.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

from carryrl.agent.actions import MOUSE_BINS, PhysicalAction

HUD_HEIGHT = 28
BACKGROUND = (31, 38, 52)
FLOOR = (187, 198, 212)
WALL = (58, 67, 80)
PLAYER = (35, 180, 235)
TARGET = (69, 207, 100)
BALL = (250, 245, 218)


def _directions(action: PhysicalAction) -> tuple[int, int]:
    keys = set(action.keys)
    return (
        int(bool(keys & {"right", "d"})) - int(bool(keys & {"left", "a"})),
        int(bool(keys & {"up", "w"})) - int(bool(keys & {"down", "s"})),
    )


class _Canvas:
    """Aspect-aware 96 x 68 canonical playfield, below the common HUD."""

    def __init__(self, size: int):
        self.size = size
        self.image = Image.new("RGB", (size, size), BACKGROUND)
        self.draw = ImageDraw.Draw(self.image)

    def point(self, point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        return round(x * (self.size - 1) / 96), HUD_HEIGHT + round(
            y * (self.size - HUD_HEIGHT - 1) / 68
        )

    def rectangle(self, box: tuple[float, float, float, float], color: tuple[int, int, int]):
        self.draw.rectangle((self.point(box[:2]), self.point(box[2:])), fill=color)

    def ellipse(self, box: tuple[float, float, float, float], color: tuple[int, int, int]):
        self.draw.ellipse((self.point(box[:2]), self.point(box[2:])), fill=color)

    def line(self, points: tuple[tuple[float, float], ...], color: tuple[int, int, int]):
        self.draw.line([self.point(point) for point in points], fill=color)

    def frame(self) -> np.ndarray:
        return np.asarray(self.image).copy()


def _centroid(frame: np.ndarray, color: tuple[int, int, int]) -> tuple[float, float] | None:
    rows, columns = np.where(np.all(frame[HUD_HEIGHT:] == color, axis=-1))
    if len(rows) == 0:
        return None
    return (
        float(columns.mean()) * 96 / (frame.shape[1] - 1),
        float(rows.mean()) * 68 / (frame.shape[0] - HUD_HEIGHT - 1),
    )


class _Game:
    def __init__(self, level: dict, image_size: int):
        self.image_size = image_size
        self.score, self.status, self.ticks = 0, "Play", 0

    def _tick(self, action: PhysicalAction) -> bool:
        if not isinstance(action, PhysicalAction):
            raise TypeError("Motion games accept PhysicalAction only")
        if self.status != "Play":
            return False
        self.ticks += 1
        return True


class TimingGame(_Game):
    """Gravity and edge-triggered flaps through three horizontally moving pipes."""

    @classmethod
    def make_level(cls, rng: np.random.Generator) -> dict:
        return {
            "pipe_x": int(rng.integers(92, 107)),
            "spacing": int(rng.integers(43, 50)),
            "speed": float(rng.choice((2.8, 3.0, 3.2))),
            "gaps": rng.integers(28, 41, size=3).tolist(),
            "gap_height": int(rng.integers(33, 39)),
            "start_y": int(rng.integers(25, 40)),
        }

    def __init__(self, level: dict, image_size: int):
        super().__init__(level, image_size)
        self.y, self.velocity = float(level["start_y"]), 0.0
        self.pipes = [float(level["pipe_x"] + i * level["spacing"]) for i in range(3)]
        self.gaps = list(level["gaps"])
        self.required_pipes = level.get("required_pipes", 3)
        self.pipes = self.pipes[:self.required_pipes]
        self.gaps = self.gaps[:self.required_pipes]
        self.gap_height, self.speed = level["gap_height"], level["speed"]
        self._previous_flap = False
        self._passed = set()
        self._prepared = False

    def step(self, action: PhysicalAction):
        if not self._tick(action):
            return
        flap = "space" in action.keys
        flap_edge = flap and not self._previous_flap
        if flap_edge:
            self.velocity = -2.8
        self._previous_flap = flap
        self.velocity = min(self.velocity + 0.45, 3.0)
        self.y += self.velocity
        self.pipes = [x - self.speed for x in self.pipes]
        if not 3 <= self.y <= 65:
            self.status = "Failed"
            return
        for index, (x, gap) in enumerate(zip(self.pipes, self.gaps, strict=True)):
            if x - 5 <= 25 and x + 5 >= 19:
                if abs(self.y - gap) + 3 > self.gap_height / 2:
                    self.status = "Failed"
                    return
            if x + 5 < 19:
                self._passed.add(index)
        if (flap_edge and self.pipes[0] - 5 > 25
                and abs(self.y - self.gaps[0]) + 3 <= self.gap_height / 2):
            self._prepared = True
        self.score = max(self.score, int(self._prepared), min(4, len(self._passed) + 1)
                         if self._passed else 0)
        if len(self._passed) == self.required_pipes:
            self.score, self.status = 5, "Success"

    def render(self) -> np.ndarray:
        canvas = _Canvas(self.image_size)
        for x, gap in zip(self.pipes, self.gaps, strict=True):
            if -5 <= x <= 101:
                canvas.rectangle((x - 5, 0, x + 5, gap - self.gap_height / 2), WALL)
                canvas.rectangle((x - 5, gap + self.gap_height / 2, x + 5, 68), WALL)
        canvas.ellipse((19, self.y - 3, 25, self.y + 3), PLAYER)
        return canvas.frame()


class PlatformingGame(_Game):
    """Horizontal movement, grounded edge-triggered jumps, and two solid hazards."""

    @classmethod
    def make_level(cls, rng: np.random.Generator) -> dict:
        return {
            "start_x": int(rng.integers(6, 13)),
            "obstacles": [
                [int(rng.integers(31, 39)), int(rng.integers(5, 9)), int(rng.integers(7, 11))],
                [int(rng.integers(64, 72)), int(rng.integers(5, 9)), int(rng.integers(7, 11))],
            ],
            "goal_x": int(rng.integers(87, 93)),
        }

    def __init__(self, level: dict, image_size: int):
        super().__init__(level, image_size)
        self.x, self.y, self.velocity = float(level["start_x"]), 53.0, 0.0
        self.start_x = self.x
        self.obstacles = [tuple(obstacle) for obstacle in level["obstacles"]]
        self.goal = level["goal_x"]
        self._previous_jump = False

    def step(self, action: PhysicalAction):
        if not self._tick(action):
            return
        horizontal, _ = _directions(action)
        jump = "space" in action.keys
        if jump and not self._previous_jump and self.y >= 52.99:
            self.velocity = -4.5
        self._previous_jump = jump
        self.x = float(np.clip(self.x + 2 * horizontal, 3, 93))
        self.velocity += 0.55
        self.y += self.velocity
        if self.y >= 53:
            self.y, self.velocity = 53.0, 0.0
        for x, width, height in self.obstacles:
            if x - 3 <= self.x <= x + width + 3 and self.y + 3 >= 56 - height:
                self.status = "Failed"
                return
        progress = int(5 * (self.x - self.start_x) / (self.goal - 3 - self.start_x))
        self.score = max(self.score, min(4, progress))
        if self.x >= self.goal - 3:
            self.score, self.status = 5, "Success"

    def render(self) -> np.ndarray:
        canvas = _Canvas(self.image_size)
        canvas.rectangle((0, 56, 96, 68), FLOOR)
        for x, width, height in self.obstacles:
            canvas.rectangle((x, 56 - height, x + width, 56), WALL)
        canvas.rectangle((self.goal - 2, 38, self.goal + 2, 55), TARGET)
        canvas.ellipse((self.x - 3, self.y - 3, self.x + 3, self.y + 3), PLAYER)
        return canvas.frame()


class SteeringGame(_Game):
    """A winding course with inertial lateral control and visible upcoming road."""

    @classmethod
    def make_level(cls, rng: np.random.Generator) -> dict:
        base = int(rng.integers(39, 58))
        amplitude, sign = int(rng.integers(11, 15)), int(rng.choice((-1, 1)))
        return {
            "centers": [base, base + sign * amplitude, base - sign * amplitude,
                        base + sign * amplitude, base],
            "width": int(rng.integers(24, 28)),
            "speed": float(rng.choice((1.1, 1.2, 1.3))),
        }

    def __init__(self, level: dict, image_size: int):
        super().__init__(level, image_size)
        self.centers = list(level["centers"])
        self.road_width, self.speed = level["width"], level["speed"]
        self.required_progress = level.get("required_progress", 72)
        self.x, self.velocity, self.progress = float(self.centers[0]), 0.0, 0.0

    def center(self, progress: float) -> float:
        return float(np.interp(progress, (0, 18, 36, 54, 72), self.centers))

    def step(self, action: PhysicalAction):
        if not self._tick(action):
            return
        horizontal, _ = _directions(action)
        self.velocity = float(np.clip(0.90 * self.velocity + 0.65 * horizontal, -3.5, 3.5))
        self.x += self.velocity
        self.progress += self.speed
        if abs(self.x - self.center(self.progress)) + 3 > self.road_width / 2:
            self.status = "Failed"
            return
        self.score = max(self.score, min(4, int(5 * self.progress / self.required_progress)))
        if self.progress >= self.required_progress:
            self.score, self.status = 5, "Success"

    def render(self) -> np.ndarray:
        canvas = _Canvas(self.image_size)
        for y in range(69):
            center = self.center(self.progress + 50 - y)
            canvas.rectangle((center - self.road_width / 2, y,
                              center + self.road_width / 2, y), FLOOR)
        canvas.rectangle((self.x - 3, 46, self.x + 3, 54), PLAYER)
        return canvas.frame()


def _angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


class CameraGame(_Game):
    """Approach and face a doorway using mouse yaw/pitch and WASD translation.

    Arrival requires the visible doorway centered on a level camera. A cyan
    crosshair signals alignment; proximity alone does not finish the task.
    """

    @classmethod
    def make_level(cls, rng: np.random.Generator) -> dict:
        return {
            "distance": float(rng.choice((6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0))),
            "bearing": float(rng.integers(-90, 91)) / 100,
            "yaw_error": float(rng.integers(20, 66) * rng.choice((-1, 1))) / 100,
            "pitch": float(rng.integers(14, 31) * rng.choice((-1, 1))) / 100,
        }

    def __init__(self, level: dict, image_size: int):
        super().__init__(level, image_size)
        self.initial_distance = level["distance"]
        self.goal = np.array((math.cos(level["bearing"]), math.sin(level["bearing"])))
        self.goal *= self.initial_distance
        self.position = np.zeros(2)
        self.yaw, self.pitch = level["bearing"] + level["yaw_error"], level["pitch"]
        self.difficulty = level.get("difficulty", 1.0)
        self._aimed, self._approach = False, 0

    def _bearing(self) -> float:
        relative = self.goal - self.position
        return _angle(math.atan2(relative[1], relative[0]) - self.yaw)

    def _aligned(self) -> bool:
        return (abs(self._bearing()) <= 0.12 + 0.04 * (1 - self.difficulty)
                and abs(self.pitch) <= 0.08 + 0.02 * (1 - self.difficulty))

    def step(self, action: PhysicalAction):
        if not self._tick(action):
            return
        self.yaw = _angle(self.yaw + action.dx * 0.018)
        self.pitch = float(np.clip(self.pitch + action.dy * 0.02, -0.7, 0.7))
        horizontal, forward = _directions(action)
        move = np.array((math.cos(self.yaw), math.sin(self.yaw))) * forward
        move += np.array((-math.sin(self.yaw), math.cos(self.yaw))) * horizontal
        if horizontal and forward:
            move /= math.sqrt(2)
        self.position += 0.25 * move
        distance = float(np.linalg.norm(self.goal - self.position))
        if distance <= 0.65 and self._aligned():
            self.score, self.status = 5, "Success"
        elif float(np.linalg.norm(self.position)) > 13:
            self.status = "Failed"
        else:
            self._aimed |= self._aligned()
            progress = int(4 * (self.initial_distance - distance) / (self.initial_distance - 0.65))
            self._approach = max(self._approach, min(3, progress))
            self.score = max(self.score, self._approach + int(self._aimed))

    def render(self) -> np.ndarray:
        canvas = _Canvas(self.image_size)
        horizon = 20 - self.pitch * 12
        canvas.rectangle((0, horizon, 96, 68), WALL)
        for y in (36, 48, 64):
            canvas.line(((0, y - self.pitch * 12), (96, y - self.pitch * 12)), FLOOR)
        for offset in (-60, -30, 0, 30, 60):
            end_x = 48 + offset + 16 * math.sin(self.yaw)
            canvas.line(((48, horizon), (end_x, 68)), FLOOR)
        relative = self.goal - self.position
        bearing = self._bearing()
        if abs(bearing) <= 1.05:
            x = 48 + 45 * bearing
            y = 32 - 20 * self.pitch
            width = min(12, max(3, 22 / max(float(np.linalg.norm(relative)), 0.7)))
            canvas.rectangle((x - width, y - width, x + width, y + width), TARGET)
        reticle = PLAYER if self._aligned() else BALL
        canvas.line(((46, 32), (50, 32)), reticle)
        canvas.line(((48, 30), (48, 34)), reticle)
        return canvas.frame()


GAMES = {
    "timing": TimingGame,
    "platforming": PlatformingGame,
    "steering": SteeringGame,
    "camera": CameraGame,
}


class _TimingController:
    def __init__(self, fail: bool):
        self.fail, self.previous_y, self.previous_flap = fail, None, False

    def observe_action(self, action: PhysicalAction) -> None:
        self.previous_flap = "space" in action.keys

    def act(self, frame: np.ndarray) -> PhysicalAction:
        if self.fail:
            return PhysicalAction()
        bird = _centroid(frame, PLAYER)
        if bird is None:
            raise RuntimeError("Timing probe cannot locate the visible bird")
        target_y = 34.0
        walls = np.all(frame[HUD_HEIGHT:] == WALL, axis=-1)
        columns = np.flatnonzero(walls.any(axis=0))
        bird_column = round(bird[0] * (frame.shape[1] - 1) / 96)
        upcoming = columns[columns >= bird_column - 2]
        if len(upcoming):
            pipe_column = int(upcoming[0])
            empty = np.flatnonzero(~walls[:, pipe_column])
            if len(empty):
                target_y = float(empty.mean()) * 68 / (frame.shape[0] - HUD_HEIGHT - 1)
        falling = self.previous_y is None or bird[1] >= self.previous_y - 0.5
        flap = bird[1] > target_y + 1 and falling and not self.previous_flap
        self.previous_y, self.previous_flap = bird[1], flap
        return PhysicalAction(keys=("space",) if flap else ())


class _PlatformingController:
    def __init__(self, fail: bool):
        self.fail, self.previous_jump = fail, False
        self.ground_y, self.previous_y, self.jumping = None, None, False
        self.clearing_x = 0.0

    def observe_action(self, action: PhysicalAction) -> None:
        if hasattr(self, "_jump_snapshot"):
            self.jumping, self.clearing_x = self._jump_snapshot
            if "space" in action.keys and not self._last_executed_jump and self._grounded:
                self.jumping, self.clearing_x = True, self._next_clearing_x
        self.previous_jump = "space" in action.keys

    def act(self, frame: np.ndarray) -> PhysicalAction:
        player = _centroid(frame, PLAYER)
        if player is None:
            raise RuntimeError("Platforming probe cannot locate the player")
        if self.ground_y is None:
            self.ground_y = player[1]
        grounded = abs(player[1] - self.ground_y) < 0.2 and (
            self.previous_y is None or abs(player[1] - self.previous_y) < 0.2
        )
        if grounded:
            self.jumping = False
        walls = np.all(frame[HUD_HEIGHT:] == WALL, axis=-1)
        columns = np.flatnonzero(walls.any(axis=0))
        groups = np.split(columns, np.where(np.diff(columns) > 1)[0] + 1)
        upcoming = [group * 96 / (frame.shape[1] - 1) for group in groups if len(group)]
        upcoming = [group for group in upcoming if group[-1] > player[0] - 3]
        distance = float(upcoming[0][0] - player[0]) if upcoming else 100
        self._jump_snapshot = self.jumping, self.clearing_x
        self._last_executed_jump, self._grounded = self.previous_jump, grounded
        self._next_clearing_x = float(upcoming[0][-1] + 4) if upcoming else player[0]
        jump = not self.fail and grounded and distance <= 12 and not self.previous_jump
        wait_for_landing = (
            not self.fail and self.jumping and player[0] >= self.clearing_x and distance <= 12
        )
        if jump:
            self.jumping = True
            self.clearing_x = float(upcoming[0][-1] + 4)
        self.previous_y = player[1]
        self.previous_jump = jump
        if wait_for_landing:
            return PhysicalAction()
        return PhysicalAction(keys=("right", "space") if jump else ("right",))


class _SteeringController:
    def __init__(self, fail: bool):
        self.fail, self.previous_x = fail, None

    def observe_action(self, action: PhysicalAction) -> None:
        pass  # Motion history is inferred from consecutive actual RGB frames.

    def act(self, frame: np.ndarray) -> PhysicalAction:
        if self.fail:
            return PhysicalAction(keys=("left",))
        car = _centroid(frame, PLAYER)
        if car is None:
            raise RuntimeError("Steering probe cannot locate the car")
        row = HUD_HEIGHT + round(43 * (frame.shape[0] - HUD_HEIGHT - 1) / 68)
        road = np.flatnonzero(np.all(frame[row] == FLOOR, axis=-1))
        if not len(road):
            raise RuntimeError("Steering probe cannot locate the upcoming road")
        target = float(road.mean()) * 96 / (frame.shape[1] - 1)
        velocity = 0 if self.previous_x is None else car[0] - self.previous_x
        desired_velocity = float(np.clip((target - car[0]) * 0.25, -2.5, 2.5))
        difference = desired_velocity - velocity
        self.previous_x = car[0]
        keys = () if abs(difference) < 0.35 else ("right" if difference > 0 else "left",)
        return PhysicalAction(keys=keys)


class _CameraController:
    def __init__(self, fail: bool):
        self.fail = fail

    def observe_action(self, action: PhysicalAction) -> None:
        pass  # This controller retains no input-edge history.

    def act(self, frame: np.ndarray) -> PhysicalAction:
        if self.fail:
            return PhysicalAction(keys=("s",))
        target = _centroid(frame, TARGET)
        if target is None:
            return PhysicalAction(dx=8)
        dx = min(MOUSE_BINS, key=lambda value: abs(value - (target[0] - 48) / 0.81))
        dy = min(MOUSE_BINS, key=lambda value: abs(value - (target[1] - 32) / 0.4))
        return PhysicalAction(keys=("w",), dx=dx, dy=dy)


def make_controller(task: str, fail: bool = False):
    controllers = {
        "timing": _TimingController,
        "platforming": _PlatformingController,
        "steering": _SteeringController,
        "camera": _CameraController,
    }
    try:
        return controllers[task](fail)
    except KeyError as exc:
        raise ValueError(f"Unknown motion task: {task}") from exc
