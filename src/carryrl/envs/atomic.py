"""Sixteen controlled RGB mechanics sharing one physical input interface.

The common, ordinary score/outcome HUD enables an explicitly supervised visual
reward bootstrap. It is not evidence of reward discovery for unknown games.
Only step advances simulation, by one 0.1-second tick; no OS input is injected.
"""

import hashlib
import json
import math
from collections import deque
from dataclasses import asdict
from functools import lru_cache

import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont

from carryrl.agent.actions import MOUSE_BINS, PhysicalAction
from carryrl.envs.atomic_suite import (
    EXTRA_TASKS,
    SCORING_VERSION,
    SUITE_VERSION,
    TASKS,
    get_extra_controller,
    get_extra_game,
)
from carryrl.envs.curriculum import constructor_parameters, resolve_effective_level
from carryrl.envs.navigation import level_banks
from carryrl.envs.presentation import PresentationConfig, choose_presentation, present_frame
from carryrl.evaluation.quality import QUALITY_VERSION, command_input_units
from carryrl.rewards.navigation import shortest_distance
from carryrl.rewards.visual import HUD_HEIGHT, draw_visible_hud, verify_visible_outcome

SPLITS = ("train", "validation", "test")
TICK_SECONDS = 0.1
BACKGROUND = (31, 38, 52)
FLOOR = (187, 198, 212)
WALL = (58, 67, 80)
PLAYER = (35, 180, 235)
TARGET = (69, 207, 100)
BALL = (250, 245, 218)
DOOR = (112, 118, 160)

_HORIZONTAL_KEYS = frozenset(("left", "a", "right", "d"))
_VERTICAL_KEYS = frozenset(("up", "w", "down", "s"))
_MOVEMENT_KEYS = _HORIZONTAL_KEYS | _VERTICAL_KEYS
_TASK_KEYS = {
    "navigation": _MOVEMENT_KEYS,
    "reaction": _HORIZONTAL_KEYS,
    "mouse": frozenset(),
    "memory": _HORIZONTAL_KEYS,
    "timing": frozenset(("space",)),
    "platforming": _HORIZONTAL_KEYS | {"space"},
    "steering": _HORIZONTAL_KEYS,
    "camera": _MOVEMENT_KEYS,
    "interaction": _MOVEMENT_KEYS | {"e"},
    "causality": _MOVEMENT_KEYS | {"e"},
    "key_door": _MOVEMENT_KEYS | {"e"},
    "goal_choice": _MOVEMENT_KEYS,
    "hazards": _MOVEMENT_KEYS,
    "shooting": _MOVEMENT_KEYS,
    "ui": _MOVEMENT_KEYS | {"tab", "enter", "escape"},
    "chain": _MOVEMENT_KEYS | {"e"},
}


def _unnecessary_input_units(task: str, action: PhysicalAction, layout: int | None) -> int:
    """Count mechanically irrelevant inputs, without classifying necessary waits.

    This is a conservative input audit, not a planner for moving hazards,
    jumping, aiming, memory, or interaction order. A release/noop is allowed in
    those games. Duplicate aliases and opposing directions never add control.
    """
    keys = set(action.keys)
    allowed = _TASK_KEYS[task]
    if task == "ui":
        allowed = (allowed - _MOVEMENT_KEYS) | (
            _HORIZONTAL_KEYS if layout == 0 else _VERTICAL_KEYS
        )
    unnecessary = len(keys - allowed)
    for negative, positive in (({"left", "a"}, {"right", "d"}),
                               ({"up", "w"}, {"down", "s"})):
        selected = keys & allowed & (negative | positive)
        if selected:
            effective = int(bool(selected & negative) != bool(selected & positive))
            unnecessary += len(selected) - effective
    mouse_controls = task in {"mouse", "camera", "shooting"}
    buttons = {"left"} if task in {"mouse", "shooting"} else set()
    unnecessary += len(set(action.mouse_buttons) - buttons)
    unnecessary += int(bool(action.wheel))
    if not mouse_controls:
        unnecessary += int(bool(action.dx)) + int(bool(action.dy))
    return unnecessary


def _navigation_action_units(action: PhysicalAction) -> int:
    """A shortest route costs one input unit per cardinal cell displacement.

    Atomic navigation executes two cardinal movements for diagonal input, so
    ticks alone are not a comparable path-length metric. Blocked arrows still
    cost a unit; noops and every unused keyboard/mouse component cost too.
    """
    return max(1, command_input_units(action))


def _identity(level: dict) -> str:
    return hashlib.sha256(json.dumps(level, sort_keys=True).encode()).hexdigest()


def atomic_semantic_level_id(task: str, level: dict) -> str:
    """Identify full-difficulty mechanics, excluding metadata and UI decoration.

    This signature describes the original difficulty-one level. Curriculum
    transformations can collapse different levels and do not inherit its split
    separation guarantee.
    """
    excluded = {"mechanic", "difficulty", "original_obstacle"}
    if task == "ui":
        excluded.update(("request_x", "menu_offset"))
    return _identity({"task": task, "level": {
        key: value for key, value in level.items() if key not in excluded
    }})


@lru_cache(maxsize=128, typed=True)
def atomic_level_banks(task: str, level_bank_seed: int = 0) -> dict[str, tuple[dict, ...]]:
    """Fixed level banks independent of episode/learning seeds.

    Seed zero preserves the legacy generator exactly. Nonzero seeds additionally
    deduplicate full-difficulty semantic IDs across all three splits.
    """
    if task not in TASKS:
        raise ValueError(f"Unknown atomic task: {task}")
    if type(level_bank_seed) is not int or not 0 <= level_bank_seed < 2**32:
        raise ValueError("level_bank_seed must be an integer in [0, 2**32)")
    generator_seed = 8821 + TASKS.index(task)
    rng = np.random.default_rng(generator_seed if level_bank_seed == 0 else
                                np.random.SeedSequence([generator_seed, level_bank_seed]))
    result = {}
    used = set()
    if level_bank_seed:
        # Conservatively avoid old full and simplified dictionaries as well as
        # keeping fresh splits separate. Only full-difficulty separation is
        # certified: simplified mechanics can collapse after transformation.
        for bank in atomic_level_banks(task, 0).values():
            for level in bank:
                used.add(atomic_semantic_level_id(task, level))
                used.add(atomic_semantic_level_id(task, _easier_level(task, level, 0)))
    for split, count in zip(SPLITS, (64, 16, 16), strict=True):
        levels = []
        attempts = 0
        while len(levels) < count:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError(f"Unable to generate distinct {task} {split} levels")
            index = len(levels)
            if task == "navigation":
                walls = level_banks()[split][index]
                cells = [(r, c) for r in range(5) for c in range(5) if (r, c) not in walls]
                start = cells[int(rng.integers(len(cells)))]
                goals = [cell for cell in cells if shortest_distance(start, cell, walls) >= 2]
                level = {
                    "walls": sorted(walls),
                    "start": start,
                    "goal": goals[int(rng.integers(len(goals)))],
                }
            elif task == "reaction":
                level = {
                    "ball_x": int(rng.integers(12, 85)),
                    "ball_dx": float(rng.choice((-1.5, -1.0, 1.0, 1.5))),
                    "paddle_x": int(rng.integers(15, 82)),
                }
            elif task == "mouse":
                locations = rng.choice(25, size=4, replace=False).tolist()
                level = {"cursor": locations[0], "targets": locations[1:]}
            elif task == "memory":
                level = {
                    "cue": int(rng.integers(2)),
                    "cue_ticks": int(rng.integers(2, 5)),
                    "unlock": int(rng.integers(66, 70)),
                    "left_x": int(rng.integers(10, 20)),
                    "right_x": int(rng.integers(76, 87)),
                }
            elif task == "ui" and level_bank_seed:
                # Four real inventory choices create room for new mechanics
                # rather than merely moving decorations on the legacy menu.
                level = {
                    "request": int(rng.integers(4)),
                    "items": rng.permutation(4).tolist(),
                    "selection": int(rng.integers(4)),
                    "layout": int(rng.integers(2)),
                    "request_x": int(rng.choice((16, 32, 48, 64, 80))),
                    "menu_offset": int(rng.integers(4)),
                    "mechanic": task,
                }
            else:
                level = get_extra_game(task).make_level(rng)
                if not isinstance(level, dict):
                    raise TypeError("Atomic game levels must be dictionaries")
                level = {**level, "mechanic": task}
            identity = (atomic_semantic_level_id(task, level) if level_bank_seed else
                        _identity(level))
            if identity not in used:
                used.add(identity)
                levels.append(level)
        result[split] = tuple(levels)
    return result


def _cell(size: int, row: int, column: int) -> tuple[int, int, int, int]:
    xs = np.linspace(3, size - 3, 6, dtype=int)
    ys = np.linspace(HUD_HEIGHT + 2, size - 3, 6, dtype=int)
    return int(xs[column]), int(ys[row]), int(xs[column + 1]), int(ys[row + 1])


def _grid_route(start, goal, blocked):
    start, goal = tuple(start), tuple(goal)
    blocked = {tuple(cell) for cell in blocked}
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        cell, path = queue.popleft()
        if cell == goal:
            return path
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = cell[0] + dr, cell[1] + dc
            if (0 <= neighbor[0] < 5 and 0 <= neighbor[1] < 5
                    and neighbor not in blocked and neighbor not in visited):
                queue.append((neighbor, path + [neighbor]))
                visited.add(neighbor)
    return None


def _easier_level(task: str, original: dict, difficulty: float) -> dict:
    """Shorten real mechanics, preserving visible objectives and physical rules."""
    if difficulty == 1:
        return original
    level = json.loads(json.dumps(original))
    level["difficulty"] = difficulty
    if task == "navigation":
        route = _grid_route(level["start"], level["goal"], level["walls"])
        if route:
            remaining = max(1, math.ceil((len(route) - 1) * difficulty))
            level["start"] = list(route[max(0, len(route) - 1 - remaining)])
    elif task == "memory":
        # Even the first stage has a disappearing cue before choice is enabled.
        level["unlock"] = max(level["cue_ticks"] + 2,
                              round(6 + difficulty * (level["unlock"] - 6)))
    elif task == "timing":
        level["pipe_x"] = round(46 + difficulty * (level["pipe_x"] - 46))
        level["gap_height"] = round(52 + difficulty * (level["gap_height"] - 52))
        level["required_pipes"] = 1 + math.ceil(2 * difficulty)
    elif task == "platforming":
        if difficulty < 0.5:
            first = level["obstacles"][0]
            level["obstacles"] = [[level["start_x"] + 11, 4, 5]]
            level["goal_x"] = level["start_x"] + 29
            level["original_obstacle"] = first
        else:
            level["obstacles"] = [[x, max(4, round(width * difficulty)),
                                    max(5, round(height * difficulty))]
                                   for x, width, height in level["obstacles"]]
    elif task == "steering":
        level["required_progress"] = 36 + 36 * difficulty
        level["width"] += round(4 * (1 - difficulty))
    elif task == "camera":
        level["distance"] = 1.75 + difficulty * (level["distance"] - 1.75)
        level["yaw_error"] = math.copysign(0.2 + difficulty * (
            abs(level["yaw_error"]) - 0.2), level["yaw_error"])
        level["pitch"] = math.copysign(0.14 + difficulty * (
            abs(level["pitch"]) - 0.14), level["pitch"])
    elif task in {"interaction", "causality", "key_door", "chain"}:
        target_name = {"interaction": "item", "causality": "button"}.get(task, "key")
        blocked = [*level["walls"], level["hazard"]]
        if "door" in level:
            blocked.append(level["door"])
        route = _grid_route(level["start"], level[target_name], blocked)
        if route:
            remaining = max(1, math.ceil((len(route) - 1) * difficulty))
            level["start"] = list(route[max(0, len(route) - 1 - remaining)])
        if task == "interaction" and difficulty < 0.5:
            route = _grid_route(level["item"], level["use"], blocked)
            if route:
                level["use"] = list(route[min(3, len(route) - 1)])
    elif task == "goal_choice":
        level["goal_y"] = round(43 + difficulty * (level["goal_y"] - 43))
        level["goal_spread"] = 12 + 19 * difficulty
    elif task == "hazards":
        route = _grid_route(level["start"], level["goal"],
                            [*level["traps"], *level["patrol"]])
        if route:
            remaining = max(1, math.ceil((len(route) - 1) * difficulty))
            level["start"] = list(route[max(0, len(route) - 1 - remaining)])
        level["enemy_period"] += round(4 * (1 - difficulty))
    elif task == "shooting":
        # Shorter approach, but at least one movement and separate aim/fire remain.
        for axis in ("x", "y"):
            distance = level[f"player_{axis}"] - level[f"enemy_{axis}"]
            level[f"player_{axis}"] = level[f"enemy_{axis}"] + distance * (0.55 + 0.45 * difficulty)
        level["required_hits"] = 1 + math.ceil(2 * difficulty)
        level["aim_radius"] = 9 - 4 * difficulty
    elif task == "ui" and difficulty < 0.5:
        # Selection and confirmation still matter; the initial selection is one
        # ordinary arrow press from the requested item, rather than already correct.
        target_index = level["items"].index(level["request"])
        level["selection"] = (target_index + 1) % len(level["items"])
        level["layout"] = 0
    return level


class AtomicEnv:
    """Atomic keyboard/mouse actions, RGB-only observations, screen-derived diagnostics.

    Keys up/down/left/right and their w/s/a/d aliases are held for one tick.
    Opposing directions cancel. Mouse clicks are button-down edges, so another
    click requires release; relative motion is in image pixels, clamped at edges.
    Unused physical inputs remain valid and leave the mechanics unchanged, but
    disqualify a perfect visible score. Navigation additionally requires a
    shortest cardinal route for score 5. Other tasks audit irrelevant inputs;
    their score does not certify an optimal trajectory or waiting schedule.
    """

    def __init__(self, task: str, split: str = "train", image_size: int = 96, max_steps: int = 80,
                 visual_config: PresentationConfig | dict | None = None,
                 difficulty: float = 1.0, level_bank_seed: int = 0,
                 curriculum_stage: str | None = None):
        if task not in TASKS or split not in SPLITS:
            raise ValueError("Unknown task or data split")
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 64:
            raise ValueError("image_size must be an integer >= 64")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if (isinstance(difficulty, bool) or not isinstance(difficulty, (int, float))
                or not math.isfinite(difficulty) or not 0 <= difficulty <= 1):
            raise ValueError("difficulty must be a finite number in [0, 1]")
        if type(level_bank_seed) is not int or not 0 <= level_bank_seed < 2**32:
            raise ValueError("level_bank_seed must be an integer in [0, 2**32)")
        if curriculum_stage is not None:
            from carryrl.envs.curriculum import STAGES
            if curriculum_stage not in STAGES or split != "train" or difficulty != 1:
                raise ValueError("Task curriculum stages require TRAIN and difficulty=1")
        self.curriculum_stage = curriculum_stage
        if isinstance(visual_config, dict):
            visual_config = PresentationConfig(**visual_config)
        if visual_config is not None and not isinstance(visual_config, PresentationConfig):
            raise TypeError("Expected PresentationConfig, configuration dict, or None")
        self.task, self.split, self.image_size, self.max_steps = task, split, image_size, max_steps
        self._canonical_size = image_size
        self.visual_config, self.difficulty = visual_config, float(difficulty)
        self.level_bank_seed = level_bank_seed
        self.presentation = None
        self._visual_episode = 0
        self.observation_space = spaces.Box(0, 255, (image_size, image_size, 3), np.uint8)
        self._rng = np.random.default_rng()
        self._initialized = False
        self._done = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options is not None and (
            not isinstance(options, dict) or set(options) - {"level_index"}
        ):
            raise ValueError("AtomicEnv reset supports level_index only")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        bank = atomic_level_banks(self.task, self.level_bank_seed)[self.split]
        level_index = options.get("level_index") if options else None
        if level_index is None:
            level_index = int(self._rng.integers(len(bank)))
        elif (isinstance(level_index, bool) or not isinstance(level_index, int)
              or not 0 <= level_index < len(bank)):
            raise ValueError("level_index outside the selected level bank")
        self._level_index = level_index
        original_level = bank[level_index]
        self._level_id = _identity(original_level)
        self._semantic_level_id = atomic_semantic_level_id(self.task, original_level)
        self._level = resolve_effective_level(
            self.task, original_level, difficulty=self.difficulty,
            curriculum_stage=self.curriculum_stage)
        self._constructor_parameters = constructor_parameters(
            self.task, self._level, self.difficulty)
        self._effective_level_id = _identity(self._level)
        style_seed = seed if seed is not None else self._visual_episode
        self._visual_episode += 1
        self.presentation = choose_presentation(self.visual_config, self.split,
                                                style_seed, self._canonical_size, task=self.task)
        self.image_size = (
            self.presentation.native_size if self.presentation else self._canonical_size
        )
        if self.presentation:
            width, height = self.presentation.canvas_size
        else:
            width = height = self.image_size
        self.observation_space = spaces.Box(0, 255, (height, width, 3), np.uint8)
        self._ticks, self._score, self._status = 0, 0, "Play"
        self._unnecessary_inputs, self._navigation_units = 0, 0
        self._input_units, self._neutral_commands = 0, 0
        self._previous_buttons = ()
        self._done, self._initialized = False, True
        if self.task == "navigation":
            self._walls = frozenset(tuple(cell) for cell in self._level["walls"])
            self._position, self._goal = tuple(self._level["start"]), tuple(self._level["goal"])
            self._initial_distance = shortest_distance(self._position, self._goal, self._walls)
        elif self.task == "reaction":
            scale = self.image_size / 96
            self._ball_x = self._level["ball_x"] * scale
            self._ball_y = float(HUD_HEIGHT + 5)
            self._ball_dx = self._level["ball_dx"] * scale
            self._paddle_x = self._level["paddle_x"] * scale
            self._reaction_drop = self._constructor_parameters["reaction_drop"]
            if self.image_size != 96:
                # Scale the actual start-to-catch distance so native rendering
                # and presentation profiles share the canonical round duration.
                self._reaction_drop *= (self.image_size - HUD_HEIGHT - 17) / (
                    96 - HUD_HEIGHT - 17
                )
            self._hits = 0
            self._required_hits = self._constructor_parameters["required_hits"]
            self._paddle_radius = self._constructor_parameters["paddle_radius"]
        elif self.task == "mouse":
            self._cursor = list(self._mouse_location(self._level["cursor"]))
            self._target_index = 0
            self._required_targets = self._constructor_parameters["required_targets"]
            self._mouse_radius = self._constructor_parameters["mouse_radius"]
            self._misses = 0
            self._miss_limit = self._constructor_parameters["miss_limit"]
            if self._constructor_parameters["cursor_distance"] is not None:
                target = self._mouse_location(self._level["targets"][0])
                dx, dy = self._cursor[0] - target[0], self._cursor[1] - target[1]
                distance = max(1, np.hypot(dx, dy))
                remaining = self._constructor_parameters["cursor_distance"]
                self._cursor = [target[0] + dx * min(1, remaining / distance),
                                target[1] + dy * min(1, remaining / distance)]
        elif self.task == "memory":
            self._position_x = self.image_size / 2
        else:
            self._game = get_extra_game(self.task)(self._level, self.image_size)
        self._frame = self._draw()
        visible = verify_visible_outcome(self._frame)
        return self._frame.copy(), self._info(visible)

    def _mouse_location(self, index: int) -> tuple[float, float]:
        row, column = divmod(index, 5)
        return (
            10 + column * (self.image_size - 20) / 4,
            HUD_HEIGHT + 10 + row * (self.image_size - HUD_HEIGHT - 20) / 4,
        )

    def step(self, action: PhysicalAction):
        if not self._initialized or self._done:
            raise RuntimeError("Reset before starting or continuing an ended episode")
        if not isinstance(action, PhysicalAction):
            raise TypeError("AtomicEnv accepts PhysicalAction only")
        keys = set(action.keys)
        horizontal = int(bool(keys & {"right", "d"})) - int(bool(keys & {"left", "a"}))
        vertical = int(bool(keys & {"down", "s"})) - int(bool(keys & {"up", "w"}))
        self._ticks += 1
        input_units = command_input_units(action)
        self._input_units += input_units
        self._neutral_commands += int(input_units == 0)
        self._unnecessary_inputs += _unnecessary_input_units(
            self.task, action, self._level.get("layout")
        )
        if self.task == "navigation":
            self._navigation_units += _navigation_action_units(action)
            for dr, dc in ((vertical, 0), (0, horizontal)):
                destination = self._position[0] + dr, self._position[1] + dc
                if (
                    0 <= destination[0] < 5
                    and 0 <= destination[1] < 5
                    and destination not in self._walls
                ):
                    self._position = destination
            if self._position == self._goal:
                perfect = self._navigation_units == self._initial_distance
                self._status, self._score = "Success", 5 if perfect else 4
            else:
                distance = shortest_distance(self._position, self._goal, self._walls)
                progress = int(5 * (self._initial_distance - distance) / self._initial_distance)
                self._score = max(self._score, min(4, progress))
        elif self.task == "reaction":
            self._paddle_x = float(np.clip(self._paddle_x + 5 * horizontal, 9, self.image_size - 9))
            self._ball_x += self._ball_dx
            self._ball_y += self._reaction_drop
            if self._ball_x < 4 or self._ball_x > self.image_size - 4:
                self._ball_dx *= -1
                self._ball_x = float(np.clip(self._ball_x, 4, self.image_size - 4))
            if self._ball_y >= self.image_size - 12:
                if abs(self._ball_x - self._paddle_x) <= self._paddle_radius + 2:
                    self._hits += 1
                    self._score = max(self._score, min(4, self._hits + 1))
                    if self._hits == self._required_hits:
                        self._status, self._score = "Success", 5
                    else:
                        self._ball_x = (
                            (self._level["ball_x"] + 13 * self._hits) % 80 + 8
                        ) * self.image_size / 96
                        self._ball_y = float(HUD_HEIGHT + 5)
                else:
                    self._status = "Failed"
            if (self._status == "Play" and horizontal
                    and abs(self._ball_x - self._paddle_x) <= self._paddle_radius + 2):
                self._score = max(self._score, 1)
        elif self.task == "mouse":
            self._cursor[0] = float(np.clip(self._cursor[0] + action.dx, 3, self.image_size - 4))
            self._cursor[1] = float(
                np.clip(self._cursor[1] + action.dy, HUD_HEIGHT + 3, self.image_size - 4)
            )
            click = "left" in action.mouse_buttons and "left" not in self._previous_buttons
            target = self._mouse_location(self._level["targets"][self._target_index])
            distance = np.hypot(self._cursor[0] - target[0], self._cursor[1] - target[1])
            if click:
                if distance <= self._mouse_radius:
                    self._target_index += 1
                    self._score = max(self._score, min(4, self._target_index + 1))
                    if self._target_index == self._required_targets:
                        self._status, self._score = "Success", 5
                else:
                    self._misses += 1
                    if self._misses >= self._miss_limit:
                        self._status = "Failed"
            if (self._status == "Play" and (action.dx or action.dy)
                    and distance <= self._mouse_radius):
                self._score = max(self._score, 1)
        elif self.task == "memory" and self._ticks >= self._level["unlock"]:
            self._position_x += 8 * horizontal
            # This readiness milestone is identical for either remembered answer.
            # The disappearing cue must never leak through intermediate score.
            if self._position_x != self.image_size / 2:
                self._score = max(self._score, 1)
            left = self._level["left_x"] * self.image_size / 96
            right = self._level["right_x"] * self.image_size / 96
            if self._position_x <= left + 6 or self._position_x >= right - 6:
                choice = 0 if self._position_x <= left + 6 else 1
                self._status = "Success" if choice == self._level["cue"] else "Failed"
                if self._status == "Success":
                    self._score = 5
        elif self.task in EXTRA_TASKS:
            self._game.step(action)
            if (isinstance(self._game.score, (bool, np.bool_))
                    or not isinstance(self._game.score, (int, np.integer))
                    or not 0 <= self._game.score <= 5):
                raise ValueError("Atomic game score must be an integer in 0..5")
            if self._game.status not in {"Play", "Success", "Failed"}:
                raise ValueError("Invalid atomic game outcome status")
            if self._game.ticks != self._ticks:
                raise ValueError("Atomic game must advance exactly one tick per action")
            self._score, self._status = int(self._game.score), self._game.status
        # Outcome and quality are distinct: irrelevant input never prevents a
        # mechanical win, but it disqualifies the perfect visible score.
        if self._score == 5 and self._unnecessary_inputs:
            self._score = 4
        self._previous_buttons = action.mouse_buttons
        previous = verify_visible_outcome(self._frame)
        self._frame = self._draw()
        visible = verify_visible_outcome(self._frame)
        if not previous["known"] or not visible["known"]:
            self._done = True
            raise RuntimeError("Atomic HUD outcome is unknown; no diagnostic reward fabricated")
        terminated = bool(visible["success"] or visible["failure"])
        truncated = self._ticks >= self.max_steps and not terminated
        self._done = terminated or truncated
        # Diagnostic only. Main PPO must use its separately frozen learned visual recognizer.
        reward = (
            visible["score"]
            - previous["score"]
            - 0.01
            + float(visible["success"])
            - float(visible["failure"])
        )
        return self._frame.copy(), float(reward), terminated, truncated, self._info(visible)

    def _curriculum_runtime_id(self):
        from carryrl.envs.runtime_identity import atomic_runtime_level_id
        return atomic_runtime_level_id(
            self.task, self._level, curriculum_stage="full")

    def _info(self, visible: dict) -> dict:
        info = {
            **visible,
            "quality": self._quality(visible),
            "suite_version": SUITE_VERSION,
            "scoring_version": SCORING_VERSION,
            "unnecessary_input_units": self._unnecessary_inputs,
            "efficiency_scope": ("shortest cardinal route and all physical input components"
                                 if self.task == "navigation" else
                                 "task-irrelevant inputs only; optimal trajectory not certified"),
            "optimal_action_units": (self._initial_distance
                                     if self.task == "navigation" else None),
            "action_units": self._navigation_units if self.task == "navigation" else None,
            "completion_efficiency": (
                self._initial_distance / self._navigation_units
                if self.task == "navigation" and visible.get("success") else None
            ),
            "level_id": self._level_id,
            "effective_level_id": self._effective_level_id,
            "semantic_level_id": self._semantic_level_id,
            "runtime_fingerprint": (
                self._curriculum_runtime_id() if self.curriculum_stage is not None
                else self._semantic_level_id if self.difficulty == 1 else None),
            "level_bank_seed": self.level_bank_seed,
            "level_index": self._level_index,
            "difficulty": self.difficulty,
            "presentation": asdict(self.presentation) if self.presentation else None,
            "map_id": self._level_id,
            "split": self.split,
            "elapsed_ticks": self._ticks,
            "action_ticks": 1,
            "tick_seconds": TICK_SECONDS,
            "reward_source": "diagnostic exact visible HUD; ignore in primary learned-reward PPO",
        }

        if self.curriculum_stage is not None:
            info.update(curriculum_stage=self.curriculum_stage,
                        effective_parameters=self._level)
        return info

    def _quality(self, visible: dict) -> dict:
        """Evaluator-only diagnostics; never fed to the RGB policy or reward model."""
        command_units = self._input_units + self._neutral_commands
        success = visible.get("known") and visible.get("success") is True
        completed_navigation = self.task == "navigation" and success
        payoff = None
        if self.task == "goal_choice" and visible.get("known"):
            if success:
                # The HUD can cap 5 to 4 for irrelevant inputs. Preserve the
                # earned goal payoff separately from that visible quality score.
                payoff = int(self._game.score)
            elif visible.get("failure"):
                payoff = 0
        return {
            "quality_version": QUALITY_VERSION,
            "task": self.task,
            "source": "diagnostic environment accounting; not policy/reward input",
            "game_seconds": self._ticks * TICK_SECONDS,
            "commands": self._ticks,
            "active_commands": self._ticks - self._neutral_commands,
            "neutral_commands": self._neutral_commands,
            "input_units": self._input_units,
            "command_units": command_units,
            "unnecessary_input_units": self._unnecessary_inputs,
            "optimal_command_units": (self._initial_distance
                                      if self.task == "navigation" else None),
            "excess_command_units": (command_units - self._initial_distance
                                     if completed_navigation else None),
            "completion_efficiency": (self._initial_distance / command_units
                                      if completed_navigation else None),
            "terminal_payoff": payoff,
            "hits": (self._hits if self.task == "reaction" else
                     self._target_index if self.task == "mouse" else
                     self._game.hits if self.task == "shooting" else None),
            "misses": (self._misses if self.task == "mouse" else
                       self._game.misses if self.task == "shooting" else None),
        }

    def _draw_canonical(self) -> np.ndarray:
        size = self.image_size
        if self.task in EXTRA_TASKS:
            frame = self._game.render()
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                    or frame.shape != (size, size, 3)):
                raise ValueError("Atomic game must render RGB uint8 HWC at its configured size")
            frame = frame.copy()
            draw_visible_hud(frame, self._score, self._status)
            return frame
        image = Image.new("RGB", (size, size), BACKGROUND)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=10)
        if self.task == "navigation":
            for row in range(5):
                for column in range(5):
                    position = row, column
                    color = WALL if position in self._walls else FLOOR
                    if position == self._goal:
                        color = TARGET
                    if position == self._position:
                        color = PLAYER
                    x0, y0, x1, y1 = _cell(size, row, column)
                    draw.rectangle((x0 + 1, y0 + 1, x1 - 2, y1 - 2), fill=color)
        elif self.task == "reaction":
            x, y = self._ball_x, self._ball_y
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=BALL)
            draw.rectangle(
                (self._paddle_x - self._paddle_radius, size - 11,
                 self._paddle_x + self._paddle_radius, size - 7), fill=PLAYER
            )
        elif self.task == "mouse":
            target_index = min(self._target_index, 2)
            x, y = self._mouse_location(self._level["targets"][target_index])
            radius = self._mouse_radius - 1
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=TARGET)
            x, y = self._cursor
            draw.line((x - 3, y, x + 3, y), fill=PLAYER, width=2)
            draw.line((x, y - 3, x, y + 3), fill=PLAYER, width=2)
            if self._miss_limit > 1:
                for index in range(self._misses):
                    draw.rectangle((3 + index * 5, size - 4, 5 + index * 5, size - 2),
                                   fill=(234, 82, 84))
        else:
            left = self._level["left_x"] * size / 96
            right = self._level["right_x"] * size / 96
            for index, x in enumerate((left, right)):
                color = (
                    TARGET
                    if self._ticks < self._level["cue_ticks"] and self._level["cue"] == index
                    else DOOR
                )
                draw.rectangle((x - 7, size - 24, x + 7, size - 5), fill=color)
            draw.ellipse(
                (self._position_x - 3, size - 18, self._position_x + 3, size - 12), fill=PLAYER
            )
            text = "Wait" if self._ticks < self._level["unlock"] else "Choose"
            draw.text((size // 2 - 18, HUD_HEIGHT + 7), text, fill=BALL, font=font)
        frame = np.asarray(image).copy()
        draw_visible_hud(frame, self._score, self._status)
        return frame

    def _draw(self) -> np.ndarray:
        frame = self._draw_canonical()
        return present_frame(frame, self.presentation) if self.presentation else frame

    def render(self) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("Reset before rendering")
        return self._frame.copy()

    def canonical_render(self) -> np.ndarray:
        """Render the current live state for canonical RGB teacher supervision."""
        if not self._initialized:
            raise RuntimeError("Reset before rendering")
        return self._draw_canonical()

    def close(self) -> None:
        self._done = True


def _centroid(frame: np.ndarray, color: tuple[int, int, int]) -> tuple[float, float] | None:
    rows, columns = np.where(np.all(frame[HUD_HEIGHT:] == color, axis=-1))
    return None if len(rows) == 0 else (float(columns.mean()), float(rows.mean() + HUD_HEIGHT))


def _visible_navigation_action(frame: np.ndarray, fail: bool) -> PhysicalAction:
    if fail:
        return PhysicalAction()
    positions = {}
    walls = set()
    for row in range(5):
        for column in range(5):
            x0, y0, x1, y1 = _cell(frame.shape[0], row, column)
            color = tuple(frame[(y0 + y1) // 2, (x0 + x1) // 2])
            if color == WALL:
                walls.add((row, column))
            elif color == PLAYER:
                positions["player"] = row, column
            elif color == TARGET:
                positions["goal"] = row, column
    frontier = deque([(positions["player"], ())])
    visited = {positions["player"]}
    while frontier:
        position, path = frontier.popleft()
        if position == positions["goal"]:
            return PhysicalAction(keys=(path[0],))
        for key, dr, dc in (("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)):
            neighbor = position[0] + dr, position[1] + dc
            if (
                0 <= neighbor[0] < 5
                and 0 <= neighbor[1] < 5
                and neighbor not in walls
                and neighbor not in visited
            ):
                visited.add(neighbor)
                frontier.append((neighbor, path + (key,)))
    raise RuntimeError("Scripted pixel probe could not recognize a reachable navigation goal")


def collect_bootstrap_dataset(
    episodes_per_task: int = 16,
    seed: int = 0,
    image_size: int = 96,
    split: str = "train",
    max_steps: int = 80,
    tasks: tuple[str, ...] | list[str] | None = None,
    difficulty: float = 1.0,
    level_bank_seed: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Collect whole scripted pixel-only probe episodes, including failure/timeout cases.

    Controllers are named diagnostic scripts, not learned policies or human
    demonstrations. Every label comes from the independent visible HUD verifier.
    Validation uses its separate level bank and is never fitted by this helper.
    """
    if episodes_per_task < 2:
        raise ValueError("Use at least two episodes per task to probe both outcomes")
    if type(level_bank_seed) is not int or not 0 <= level_bank_seed < 2**32:
        raise ValueError("level_bank_seed must be an integer in [0, 2**32)")
    selected = TASKS if tasks is None else tuple(tasks)
    if not selected or len(set(selected)) != len(selected) or any(t not in TASKS for t in selected):
        raise ValueError("Bootstrap tasks must be a nonempty unique supported selection")
    frames, labels, frame_episode_ids, episodes = (
        [],
        {name: [] for name in ("score", "success", "failure")},
        [],
        [],
    )
    for task in selected:
        task_index = TASKS.index(task)
        env = AtomicEnv(task, split, image_size, max_steps, difficulty=difficulty,
                        level_bank_seed=level_bank_seed)
        for episode in range(episodes_per_task):
            episode_seed = seed + 10000 * task_index + episode
            frame, info = env.reset(seed=episode_seed)
            episode_id = f"{split}:{task}:{episode_seed}:{info['level_id']}"
            fail = episode % 2 == 1
            previous_buttons = ()
            memory_choice = None
            controller = get_extra_controller(task, fail=fail) if task in EXTRA_TASKS else None
            terminated = truncated = False
            steps = 0
            while True:
                label = verify_visible_outcome(frame)
                if not label["known"]:
                    raise RuntimeError("Unknown bootstrap HUD; refusing to generate labels")
                frames.append(frame.copy())
                frame_episode_ids.append(episode_id)
                for name in labels:
                    labels[name].append(float(label[name]))
                if terminated or truncated:
                    break
                if task == "navigation":
                    action = _visible_navigation_action(frame, fail)
                elif task == "reaction":
                    ball, paddle = _centroid(frame, BALL), _centroid(frame, PLAYER)
                    target_x = (0 if ball[0] > image_size / 2 else image_size) if fail else ball[0]
                    difference = target_x - paddle[0]
                    action = PhysicalAction(
                        keys=(
                            () if abs(difference) < 3 else ("right" if difference > 0 else "left",)
                        )
                    )
                elif task == "mouse":
                    cursor, target = _centroid(frame, PLAYER), _centroid(frame, TARGET)
                    dx, dy = target[0] - cursor[0], target[1] - cursor[1]
                    buttons = (
                        ("left",)
                        if fail
                        or (abs(dx) <= 3 and abs(dy) <= 3 and "left" not in previous_buttons)
                        else ()
                    )
                    action = PhysicalAction(
                        mouse_buttons=buttons,
                        dx=0 if fail else min(MOUSE_BINS, key=lambda value: abs(value - dx)),
                        dy=0 if fail else min(MOUSE_BINS, key=lambda value: abs(value - dy)),
                    )
                elif task == "memory":
                    cue = _centroid(frame, TARGET)
                    if cue is not None:
                        memory_choice = "left" if cue[0] < image_size / 2 else "right"
                    if memory_choice is None:
                        raise RuntimeError("Visible memory cue missing at episode start")
                    choice = (
                        ("right" if memory_choice == "left" else "left") if fail else memory_choice
                    )
                    action = PhysicalAction(keys=(choice,))
                else:
                    action = controller.act(frame)
                previous_buttons = action.mouse_buttons
                frame, _, terminated, truncated, info = env.step(action)
                steps += 1
            episodes.append(
                {
                    "episode_id": episode_id,
                    "task": task,
                    "split": split,
                    "level_id": info["level_id"],
                    "level_bank_seed": level_bank_seed,
                    "seed": episode_seed,
                    "strategy": "scripted pixel-only failure/timeout probe"
                    if fail
                    else "scripted pixel-only success probe",
                    "success": info["success"],
                    "failure": info["failure"],
                    "truncated": truncated,
                    "ticks": steps,
                    "scoring_version": SCORING_VERSION,
                }
            )
        env.close()
    arrays = {name: np.asarray(values, np.float32) for name, values in labels.items()}
    provenance = {
        "split": split,
        "level_bank_seed": level_bank_seed,
        "suite_version": SUITE_VERSION,
        "scoring_version": SCORING_VERSION,
        "tasks": list(selected),
        "label_source": "independent exact RGB ordinary HUD verifier",
        "controller_source": "scripted controllers using current RGB and own cue history",
        "not_autonomous_reward_discovery": True,
        "episodes": episodes,
        "frame_episode_ids": frame_episode_ids,
        "label_counts": {
            "play": int(((arrays["success"] == 0) & (arrays["failure"] == 0)).sum()),
            "success": int(arrays["success"].sum()),
            "failure": int(arrays["failure"].sum()),
        },
        "score_counts": {str(value): int((arrays["score"] == value).sum()) for value in range(6)},
    }
    return np.stack(frames), arrays, provenance
