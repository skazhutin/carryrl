"""Pure pixel recognition and reward rules for the controlled navigation game.

The recognizer has no game object or simulator state. Colors describe ordinary
visible floor, walls, player and goal tiles. The gold player tile appears only
when the player reaches the goal. Unknown or inconsistent frames raise an error.
"""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

GRID_SIZE = 5
BACKGROUND = (18, 24, 36)
FLOOR = (210, 220, 229)
WALL = (57, 68, 85)
PLAYER = (29, 181, 232)
GOAL = (87, 197, 110)
WINNER = (255, 208, 70)
TILE_COLORS = {FLOOR, WALL, PLAYER, GOAL, WINNER}


class RecognitionError(ValueError):
    """A frame does not expose one unambiguous valid navigation state."""


@dataclass(frozen=True)
class VisibleNavigationState:
    player: tuple[int, int]
    goal: tuple[int, int]
    walls: frozenset[tuple[int, int]]
    distance: int
    success: bool


@lru_cache(maxsize=512)
def cell_bounds(size: int, row: int, column: int) -> tuple[int, int, int, int]:
    """Board bounds shared by drawing and recognition, with a normal status HUD."""
    x_edges = np.linspace(2, size - 2, GRID_SIZE + 1, dtype=int)
    y_edges = np.linspace(12, size - 2, GRID_SIZE + 1, dtype=int)
    return (
        int(y_edges[row]),
        int(y_edges[row + 1]),
        int(x_edges[column]),
        int(x_edges[column + 1]),
    )


def shortest_distance(
    player: tuple[int, int], goal: tuple[int, int], walls: frozenset[tuple[int, int]]
) -> int:
    """Shortest traversable distance using only objects recognized in the frame."""
    frontier = deque([(player, 0)])
    seen = {player}
    while frontier:
        (row, column), distance = frontier.popleft()
        if (row, column) == goal:
            return distance
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = row + dr, column + dc
            if (
                0 <= neighbor[0] < GRID_SIZE
                and 0 <= neighbor[1] < GRID_SIZE
                and neighbor not in walls
                and neighbor not in seen
            ):
                seen.add(neighbor)
                frontier.append((neighbor, distance + 1))
    raise RecognitionError("Visible goal is unreachable from the visible player")


def recognize_navigation(frame: np.ndarray) -> VisibleNavigationState:
    """Recognize the complete visible board or fail without inventing an outcome."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] != frame.shape[1]
        or frame.shape[0] < 48
    ):
        raise RecognitionError("Expected a square RGB uint8 navigation frame, size >= 48")
    size = frame.shape[0]
    if not np.all(frame[:, :2] == BACKGROUND) or not np.all(frame[-2:] == BACKGROUND):
        raise RecognitionError("Navigation frame border is missing or corrupted")
    objects: dict[tuple[int, int, int], list[tuple[int, int]]] = {
        color: [] for color in TILE_COLORS
    }
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            y0, y1, x0, x1 = cell_bounds(size, row, column)
            pixels = frame[y0 + 1 : y1 - 1, x0 + 1 : x1 - 1]
            color = tuple(int(value) for value in pixels[0, 0])
            if color not in objects or not np.all(pixels == color):
                raise RecognitionError(f"Unknown or corrupted visible tile at ({row}, {column})")
            objects[color].append((row, column))
    if len(objects[WALL]) != 2:
        raise RecognitionError("Expected two visible obstacle tiles")
    if objects[WINNER]:
        if len(objects[WINNER]) != 1 or objects[PLAYER] or objects[GOAL]:
            raise RecognitionError("Ambiguous visible victory state")
        player = goal = objects[WINNER][0]
        success = True
    else:
        if len(objects[PLAYER]) != 1 or len(objects[GOAL]) != 1:
            raise RecognitionError("Expected one visible player and one visible goal")
        player, goal = objects[PLAYER][0], objects[GOAL][0]
        success = False
    walls = frozenset(objects[WALL])
    return VisibleNavigationState(
        player, goal, walls, shortest_distance(player, goal, walls), success
    )


def navigation_reward(
    previous: np.ndarray, current: np.ndarray
) -> tuple[float, VisibleNavigationState]:
    """Screen-derived shaping, plus an independently reported goal outcome.

    Each fixed tick costs 0.01. A one-cell decrease in shortest visible path
    distance gives 0.04, an increase costs 0.04; reaching the goal gives 1.0.
    This is heuristic dense shaping, not a claim of policy-invariant shaping.
    """
    before, after = recognize_navigation(previous), recognize_navigation(current)
    if before.success:
        raise RecognitionError("Reward cannot continue after a visible victory")
    if before.walls != after.walls or before.goal != after.goal:
        raise RecognitionError("Visible layout or goal changed during a transition")
    movement = abs(before.player[0] - after.player[0]) + abs(before.player[1] - after.player[1])
    if movement > 1:
        raise RecognitionError("Visible player moved farther than one physical action tick")
    reward = -0.01 + 0.04 * (before.distance - after.distance) + float(after.success)
    return float(reward), after
