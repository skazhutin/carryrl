"""Pure, explicit TRAIN task stages and shared constructor defaults.

A stage changes real mechanics. Provenance is deliberately separate from the
runtime parameters used to detect collisions with held-out mechanics.
"""

import copy
import math

from carryrl.rewards.visual import HUD_HEIGHT

CURRICULUM_VERSION = "carryrl.task-curriculum.v1"
STAGES = ("easy", "bridge", "full")
RUNTIME_FIELDS = {
    "navigation": "walls start goal",
    "reaction": "ball_x ball_dx paddle_x",
    "mouse": "cursor targets",
    "memory": "cue cue_ticks unlock left_x right_x",
    "timing": "pipe_x spacing speed gaps gap_height start_y",
    "platforming": "start_x obstacles goal_x",
    "steering": "centers width speed",
    "camera": "distance bearing yaw_error pitch",
    "interaction": "start walls hazard item use",
    "causality": "start walls hazard button door exit",
    "key_door": "start walls hazard key door exit",
    "chain": "start walls hazard key door exit switch",
    "goal_choice": "safe_side safe_value risky_value goal_y guard_y guard_direction guard_speed",
    "hazards": "start goal traps patrol enemy_index enemy_period",
    "shooting": "player_x player_y enemy_x enemy_y cursor_x cursor_y",
    "ui": "request items selection layout",
}


def constructor_parameters(task: str, level: dict, difficulty: float = 1.0) -> dict:
    """Defaults consumed by both constructors and runtime identity."""
    if task == "reaction":
        return {"required_hits": level.get("required_hits", 1 + math.ceil(2 * difficulty)),
                "paddle_radius": level.get("paddle_radius", 10 + round(8 * (1 - difficulty))),
                "reaction_drop": level.get("reaction_drop", 2.5)}
    if task == "mouse":
        return {"required_targets": level.get("required_targets", 1 + math.ceil(2 * difficulty)),
                "mouse_radius": level.get("mouse_radius", 8 + round(4 * (1 - difficulty))),
                "miss_limit": level.get("miss_limit", 1 + round(4 * (1 - difficulty))),
                "cursor_distance": level.get("cursor_distance", 14 + 30 * difficulty
                                             if difficulty < 0.5 else None)}
    if task == "timing":
        return {"required_pipes": level.get("required_pipes", 3)}
    if task == "steering":
        return {"required_progress": level.get("required_progress", 72)}
    if task == "camera":
        d = level.get("difficulty", 1.0)
        return {"yaw_tolerance": level.get("yaw_tolerance", 0.12 + 0.04 * (1 - d)),
                "pitch_tolerance": level.get("pitch_tolerance", 0.08 + 0.02 * (1 - d))}
    if task == "goal_choice":
        return {"goal_spread": level.get("goal_spread", 31)}
    if task == "shooting":
        return {"required_hits": level.get("required_hits", 3),
                "aim_radius": level.get("aim_radius", 5)}
    return {}


def resolve_effective_level(task: str, original: dict, *, difficulty: float = 1.0,
                            curriculum_stage: str | None = None) -> dict:
    """Return an independent effective level; legacy scalar behaviour is preserved."""
    from carryrl.envs.atomic import _easier_level, _grid_route

    if task not in RUNTIME_FIELDS:
        raise ValueError(f"Unknown task: {task}")
    if curriculum_stage is None:
        return copy.deepcopy(_easier_level(task, original, difficulty))
    if curriculum_stage not in STAGES:
        raise ValueError("curriculum_stage must be easy, bridge, or full")
    if difficulty != 1:
        raise ValueError("Explicit task stages require scalar difficulty=1")
    level = copy.deepcopy(original)
    if curriculum_stage == "full":
        return level
    easy = curriculum_stage == "easy"

    def suffix(target, blocked, steps):
        route = _grid_route(level["start"], level[target], blocked)
        if not route:
            raise ValueError(f"No safe path for {task} stage projection")
        level["start"] = list(route[max(0, len(route) - 1 - steps)])

    if task == "navigation":
        suffix("goal", level["walls"], 2 if easy else 4)
    elif task == "reaction":
        level.update(required_hits=1 if easy else 2, paddle_radius=16 if easy else 13,
                     reaction_drop=1.5 if easy else 2.0,
                     ball_dx=level["ball_dx"] * (0.6 if easy else 0.8))
    elif task == "mouse":
        level.update(required_targets=1 if easy else 2, mouse_radius=12 if easy else 10,
                     cursor_distance=16 if easy else 32, miss_limit=5 if easy else 3)
    elif task == "memory":
        level["unlock"] = level["cue_ticks"] + 4 if easy else 32
    elif task == "timing":
        level.update(required_pipes=1 if easy else 2, pipe_x=46 if easy else 72,
                     speed=1.8 if easy else 2.4, gap_height=52 if easy else 42)
    elif task == "platforming":
        if easy:
            first = level["obstacles"][0]
            level["obstacles"] = [first]
            level["goal_x"] = first[0] + first[1] + 10
        else:
            level["obstacles"] = [[x, w, min(h, 8)] for x, w, h in level["obstacles"]]
    elif task == "steering":
        level.update(required_progress=36 if easy else 54,
                     width=level["width"] + (2 if easy else 1), speed=0.8 if easy else 1.0)
    elif task == "camera":
        level.update(distance=2 if easy else 4,
                     yaw_error=math.copysign(0.20 if easy else (0.20 + abs(level["yaw_error"])) / 2,
                                             level["yaw_error"]),
                     pitch=math.copysign(0.14 if easy else (0.14 + abs(level["pitch"])) / 2,
                                         level["pitch"]),
                     yaw_tolerance=0.16 if easy else 0.14,
                     pitch_tolerance=0.10 if easy else 0.09)
    elif task in {"interaction", "causality", "key_door", "chain"}:
        blocked = [*level["walls"], level["hazard"]]
        if "door" in level:
            blocked.append(level["door"])
        target = {"interaction": "item", "causality": "button"}.get(task, "key")
        suffix(target, blocked, 2 if easy else 4)
        if task == "interaction" and easy:
            route = _grid_route(level["item"], level["use"], blocked)
            if not route:
                raise ValueError("No safe item-to-use path for interaction stage")
            level["use"] = list(route[min(3, len(route) - 1)])
    elif task == "goal_choice":
        level.update(goal_y=43 if easy else 28, goal_spread=12 if easy else 22,
                     guard_speed=1 if easy else min(level["guard_speed"], 2))
    elif task == "hazards":
        suffix("goal", [*level["traps"], *level["patrol"]], 2 if easy else 4)
        level["enemy_period"] = 6 if easy else 4
    elif task == "shooting":
        delta = [level[f"player_{a}"] - level[f"enemy_{a}"] for a in ("x", "y")]
        length = math.hypot(*delta)
        distance = 24 if easy else 36
        for axis, value in zip(("x", "y"), delta, strict=True):
            level[f"player_{axis}"] = level[f"enemy_{axis}"] + value * distance / length
        # Cursor must still require independent aim before the first shot.
        delta = [level[f"cursor_{a}"] - level[f"enemy_{a}"] for a in ("x", "y")]
        length = math.hypot(*delta)
        if length < 16:
            level["cursor_x"] = level["enemy_x"] - 16
            level["cursor_y"] = level["enemy_y"]
        level.update(required_hits=1 if easy else 2, aim_radius=9 if easy else 7)
    elif task == "ui":
        if len(level["items"]) != 4:
            raise ValueError("Task curriculum requires four UI items in the source bank")
        target = level["items"].index(level["request"])
        if easy or level["selection"] == target:
            level["selection"] = (target + 1) % len(level["items"])
        if easy:
            level["layout"] = 0
    return level


def runtime_parameters(task: str, level: dict, difficulty: float = 1.0) -> dict:
    """Canonical 96px effective mechanics; no stage, source seed, or artwork."""
    used = {key: copy.deepcopy(level[key]) for key in RUNTIME_FIELDS[task].split()}
    used.update(constructor_parameters(task, level, difficulty))
    for name in ("walls", "traps"):
        if name in used:
            used[name] = sorted({tuple(cell) for cell in used[name]})
    if task == "mouse":
        def location(index):
            row, column = divmod(index, 5)
            return [10 + column * 19, HUD_HEIGHT + 10 + row * (76 - HUD_HEIGHT) / 4]
        cursor = location(level["cursor"])
        distance = used.pop("cursor_distance")
        if distance is not None:
            target = location(level["targets"][0])
            delta = [a - b for a, b in zip(cursor, target, strict=True)]
            ratio = min(1, distance / max(1, math.hypot(*delta)))
            cursor = [b + a * ratio for a, b in zip(delta, target, strict=True)]
        used["cursor"] = cursor
        used["targets"] = used["targets"][:used["required_targets"]]
    elif task == "timing":
        count = used.pop("required_pipes")
        pipe_x = used.pop("pipe_x")
        used["pipes"] = [float(pipe_x + i * used["spacing"]) for i in range(count)]
        used.pop("spacing")
        used["gaps"] = used["gaps"][:count]
    return used
