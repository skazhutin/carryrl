"""Small navigation game with ordinary arrow-key actions and RGB observations.

Simulation advances only when ``step`` applies one held-key tick (0.1 seconds).
Rendered frames provide all reward and success information. ``human`` rendering
is a view of this same game, not an OS screen-capture or input-injection adapter.
"""

import hashlib
import itertools
from functools import lru_cache

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from carryrl.rewards.navigation import (
    BACKGROUND,
    FLOOR,
    GOAL,
    GRID_SIZE,
    PLAYER,
    WALL,
    WINNER,
    RecognitionError,
    VisibleNavigationState,
    cell_bounds,
    navigation_reward,
    recognize_navigation,
    shortest_distance,
)

ACTION_NAMES = ("noop", "up", "down", "left", "right")
ACTION_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
TICK_SECONDS = 0.1
SPLITS = ("train", "validation", "test")


def _wall_id(walls: frozenset[tuple[int, int]]) -> str:
    return hashlib.sha256(repr(sorted(walls)).encode("ascii")).hexdigest()


@lru_cache(maxsize=1)
def level_banks() -> dict[str, tuple[frozenset[tuple[int, int]], ...]]:
    """Fixed disjoint obstacle banks: 64 training, 16 validation, 16 test maps.

    All remaining cells are connected. Banks are shuffled once with a fixed
    generation seed, independent of episode or learning seeds. Map identifiers
    hash obstacle geometry; level identifiers also include the start and goal.
    """
    cells = tuple(itertools.product(range(GRID_SIZE), repeat=2))
    layouts = []
    for pair in itertools.combinations(cells, 2):
        walls = frozenset(pair)
        floor = [cell for cell in cells if cell not in walls]
        try:
            for cell in floor[1:]:
                shortest_distance(floor[0], cell, walls)
        except RecognitionError:
            continue
        layouts.append(walls)
    rng = np.random.default_rng(20261008)
    order = rng.permutation(len(layouts))
    shuffled = [layouts[int(index)] for index in order]
    return {
        "train": tuple(shuffled[:64]),
        "validation": tuple(shuffled[64:80]),
        "test": tuple(shuffled[80:96]),
    }


# Tiny ordinary status text, drawn identically without a window or system font.
_FONT = {
    "P": ("110", "101", "110", "100", "100"),
    "L": ("100", "100", "100", "100", "111"),
    "A": ("010", "101", "111", "101", "101"),
    "Y": ("101", "101", "010", "010", "010"),
    "W": ("101", "101", "101", "111", "101"),
    "I": ("111", "010", "010", "010", "111"),
    "N": ("101", "111", "111", "111", "101"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("110", "001", "010", "100", "111"),
    "3": ("110", "001", "010", "001", "110"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "110", "001", "110"),
    "6": ("011", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "110"),
    " ": ("000",) * 5,
}


class NavigationEnv(gym.Env):
    """Navigate a visible cyan player to a green goal around dark obstacles.

    Actions: 0=noop, 1=up, 2=down, 3=left, 4=right. A blocked arrow keeps the
    player still; there are no planner or target-relative actions. Training and
    evaluation have disjoint obstacle maps. The policy observation is RGB only.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        split: str = "train",
        size: int = 64,
        max_steps: int = 64,
    ):
        super().__init__()
        if render_mode not in (None, "rgb_array", "human"):
            raise ValueError("render_mode must be None, 'rgb_array', or 'human'")
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 48:
            raise ValueError("size must be an integer >= 48")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self.render_mode, self.split, self.size, self.max_steps = (
            render_mode,
            split,
            size,
            max_steps,
        )
        self.observation_space = spaces.Box(0, 255, shape=(size, size, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self._initialized = False
        self._done = False
        self._window = None
        self._clock = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options:
            raise ValueError("NavigationEnv does not accept reset options")
        bank = level_banks()[self.split]
        self._walls = bank[int(self.np_random.integers(len(bank)))]
        floor = [
            cell
            for cell in itertools.product(range(GRID_SIZE), repeat=2)
            if cell not in self._walls
        ]
        self._player = floor[int(self.np_random.integers(len(floor)))]
        goals = [cell for cell in floor if shortest_distance(self._player, cell, self._walls) >= 2]
        self._goal = goals[int(self.np_random.integers(len(goals)))]
        self._elapsed_ticks, self._done, self._initialized = 0, False, True
        self._map_id = _wall_id(self._walls)
        self._level_id = hashlib.sha256(
            repr((sorted(self._walls), self._player, self._goal)).encode("ascii")
        ).hexdigest()
        frame = self._draw_frame()
        visible = recognize_navigation(frame)
        self._previous_frame = frame.copy()
        if self.render_mode == "human":
            self._show_window(frame)
        return frame, self._info(visible)

    def step(self, action):
        if not self._initialized:
            raise RuntimeError("Call reset before step")
        if self._done:
            raise RuntimeError("Episode ended; call reset before another step")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"Expected an integer action in 0..{len(ACTION_NAMES) - 1}")
        dr, dc = ACTION_DELTAS[int(action)]
        destination = self._player[0] + dr, self._player[1] + dc
        if (
            0 <= destination[0] < GRID_SIZE
            and 0 <= destination[1] < GRID_SIZE
            and destination not in self._walls
        ):
            self._player = destination
        self._elapsed_ticks += 1
        frame = self._draw_frame()
        try:
            reward, visible = navigation_reward(self._previous_frame, frame)
        except RecognitionError:
            self._done = True
            raise
        terminated = visible.success
        truncated = self._elapsed_ticks >= self.max_steps and not terminated
        self._done = terminated or truncated
        self._previous_frame = frame.copy()
        if self.render_mode == "human":
            self._show_window(frame)
        return frame, reward, terminated, truncated, self._info(visible)

    def _info(self, visible: VisibleNavigationState) -> dict:
        return {
            "success": visible.success,
            "distance": visible.distance,
            "map_id": self._map_id,
            "level_id": self._level_id,
            "split": self.split,
            "elapsed_ticks": self._elapsed_ticks,
            "action_ticks": 1,
            "tick_seconds": TICK_SECONDS,
            "recognition": "ok",
        }

    def _draw_frame(self) -> np.ndarray:
        frame = np.full((self.size, self.size, 3), BACKGROUND, dtype=np.uint8)
        status = ("WIN" if self._player == self._goal else "PLAY") + f" {self._elapsed_ticks}"
        for index, letter in enumerate(status):
            x = 3 + 4 * index
            if x + 3 >= self.size - 2:
                break
            for y, line in enumerate(_FONT[letter], start=3):
                for dx, bit in enumerate(line):
                    if bit == "1":
                        frame[y, x + dx] = FLOOR
        for row in range(GRID_SIZE):
            for column in range(GRID_SIZE):
                position = row, column
                color = WALL if position in self._walls else FLOOR
                if position == self._goal:
                    color = GOAL
                if position == self._player:
                    color = WINNER if position == self._goal else PLAYER
                y0, y1, x0, x1 = cell_bounds(self.size, row, column)
                frame[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1] = color
        return frame

    def render(self):
        if not self._initialized:
            raise RuntimeError("Call reset before render")
        frame = self._draw_frame()
        if self.render_mode == "human":
            self._show_window(frame)
            return None
        if self.render_mode == "rgb_array":
            return frame
        return None

    def _show_window(self, frame: np.ndarray):
        import pygame

        if self._window is None:
            pygame.init()
            self._window = pygame.display.set_mode((512, 512))
            pygame.display.set_caption("carryrl navigation — arrows move, R reset, Esc quit")
            self._clock = pygame.time.Clock()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._window.blit(pygame.transform.scale(surface, (512, 512)), (0, 0))
        pygame.display.flip()
        pygame.event.pump()

    def close(self):
        if self._window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self._window, self._clock = None, None
