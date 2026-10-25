"""Episode-stable pixel scenery and visible-role sprites, derived only from RGB.

Task selects an art direction, never positions or state. All marks stay inside
observed color masks; canonical HUD, text, health pips and selection remain exact.
"""

from functools import lru_cache

import numpy as np

PIXEL_STYLE_SPLITS = {
    "train": ("moss", "crypt", "copper"),
    "validation": ("frost", "circuit"),
    "test": ("ember", "marble"),
}
# Background, walkable stone, solid stone, distant scenery.
_PALETTES = {
    "moss": ((21, 36, 32), (154, 177, 144), (67, 86, 68), (32, 49, 40)),
    "crypt": ((24, 27, 43), (169, 173, 191), (68, 69, 88), (36, 39, 56)),
    "copper": ((36, 27, 25), (196, 174, 143), (94, 72, 57), (49, 37, 31)),
    "frost": ((18, 31, 45), (160, 193, 204), (58, 82, 99), (29, 44, 59)),
    "circuit": ((19, 31, 36), (157, 184, 177), (53, 78, 77), (29, 46, 51)),
    "ember": ((39, 23, 32), (196, 163, 159), (92, 57, 69), (53, 33, 44)),
    "marble": ((28, 30, 39), (195, 191, 174), (75, 76, 85), (41, 43, 53)),
}
BACKGROUND = (31, 38, 52)
FLOOR = (187, 198, 212)
WALL = (58, 67, 80)
PLAYER = (35, 180, 235)
TARGET = (69, 207, 100)
_GRID_TASKS = {"navigation", "interaction", "causality", "key_door", "chain", "hazards"}
_GLYPHS = {
    "agent": ("..###..", ".#####.", ".#ww##.", ".#####.", "..###..", ".##.##.", ".#...#."),
    "exit": (".#####.", "##...##", "#..+..#", "#.+w+.#", "#..+..#", "#.....#", "#######"),
    "key": (".###...", "#...#..", "#...#..", ".####..", "....#..", "....##.", "....#.."),
    "door": ("#######", "#s#s#s#", "#s#s#s#", "#s#s#w#", "#s#s#s#", "#s#s#s#", "#######"),
    "hazard": ("...w...", "..###..", "..#i#..", ".##i##.", ".#####.", "###i###", "#######"),
    "enemy": (".#####.", "#######", "#ii#ii#", "#######", ".#i#i#.", "..###..", ".#...#."),
    "item": (".#####.", "##+++##", "##+#s##", "##+#s##", "##+++##", "##sss##", ".#####."),
    "pad": (".......", ".#####.", ".#...#.", ".#.w.#.", ".#...#.", ".#####.", "......."),
    "switch": ("....w..", "...##..", "..##...", ".##....", "#######", "#sssss#", "#######"),
    "check": (".......", ".....w.", "....##.", ".#.##..", ".###...", "..#....", "......."),
    "car": (".sssss.", ".s+++s.", "#siiis#", "#siiis#", ".s###s.", "#s###s#", "#s###s#",
            ".swwws.", ".sssss."),
    "bird": ("..###..", ".#####.", "###w##.", "##ii###", ".###...", "..##...", "......."),
    "gem": ("...w...", "..#+#..", ".#+++#.", "##++ss#", ".#sss#.", "..#s#..", "...#..."),
    "potion": ("..www..", "...#...", "..###..", ".#+++#.", ".#++s#.", ".#sss#.", "..###.."),
}
_ROLES = {
    (242, 202, 55): "item", (172, 96, 215): "pad", (236, 120, 63): "switch",
    (255, 220, 40): "key", (245, 95, 200): "switch", (112, 118, 160): "door",
    (123, 165, 184): "exit", (113, 147, 106): "check", (224, 64, 80): "hazard",
    (230, 62, 72): "hazard", (243, 131, 48): "enemy", (245, 194, 63): "item",
    (220, 88, 105): "gem", (99, 208, 131): "potion", (166, 112, 229): "gem",
    (235, 187, 69): "item",
}


def _shade(color, offset):
    return np.clip(np.asarray(color, dtype=np.int16) + offset, 0, 255).astype(np.uint8)


@lru_cache(maxsize=128)
def _materials(size: int, style: str, seed: int, textures: bool, task: str | None):
    """Quiet scenery, with detail concentrated on actual solid surfaces."""
    rng = np.random.default_rng(seed)
    ground, floor, wall, accent = _PALETTES[style]
    tint = int(rng.integers(-4, 5))
    spacing = int(rng.choice((14, 18, 22)))
    phase_x, phase_y = rng.integers(0, spacing, size=2)
    y, x = np.indices((size, size))
    tx, ty = (x + phase_x) % spacing, (y + phase_y) % spacing
    materials = [np.broadcast_to(_shade(color, tint), (size, size, 3)).copy()
                 for color in (ground, floor, wall)]
    if textures:
        background, paving, stone = materials
        # Restrained distant bands leave text, cursors and moving objects clear.
        background[y > size * .58] = _shade(ground, tint + 3)
        if task in {"platforming", "timing"}:
            ridge = size * .77 + 3 * np.sin(x / 11 + phase_x)
            background[y > ridge] = _shade(accent, tint - 3)
        elif task == "steering":
            grass = ((tx == 3) & (ty == 4)) | ((tx == 4) & (ty == 3))
            background[grass] = _shade(accent, tint)
        elif task is None:
            # Legacy taskless transforms retain subdued, sparse material variation.
            background[(tx == 3) & (ty == 4)] = _shade(accent, tint - 3)
        # Material families differ in structure as well as palette. Validation
        # cracks/circuit traces never occur in the training wood/stone/metal set.
        # Scale material coordinates only; observed geometry and sprite masks
        # always retain their original pixels.
        scale = float(rng.choice((.75, 1., 1.25, 1.5)))
        px = np.floor((x + phase_x) / scale).astype(int)
        py = np.floor((y + phase_y) / scale).astype(int)
        if style == "moss":
            grain = ((py % 7 == 3) & (px % 21 >= 4) & (px % 21 <= 10))
            paving[grain] = _shade(floor, tint - 8)
            joint = (py % 7 == 0) | ((px + (py // 7 % 2) * 7) % 21 == 0)
            stone[joint] = _shade(wall, tint - 19)
            stone[(py % 7 == 1) & ~joint] = _shade(wall, tint + 11)
        elif style == "crypt":
            paving[(px % 17 == 0) | (py % 13 == 0)] = _shade(floor, tint - 10)
            joint = (py % 8 == 0) | ((px + (py // 8 % 2) * 8) % 16 == 0)
            stone[joint] = _shade(wall, tint - 20)
            stone[(py % 8 == 1) & ~joint] = _shade(wall, tint + 13)
        elif style == "copper":
            seam = (px % 18 == 0) | (py % 12 == 0)
            rivet = (px % 18 == 2) & (py % 12 == 2)
            paving[seam] = _shade(floor, tint - 10)
            paving[rivet] = _shade(floor, tint + 15)
            stone[seam] = _shade(wall, tint - 18)
            stone[rivet] = _shade(wall, tint + 29)
        elif style == "frost":
            crack = (px % 29 == (py // 4) % 7 + 7)
            crack &= py % 31 < 23
            branch = ((px + py) % 31 == 6) & (py % 31 < 8)
            paving[crack | branch] = _shade(floor, tint - 13)
            stone[crack | branch] = _shade(wall, tint + 24)
        elif style == "circuit":
            trace = ((px % 23 == 4) & (py % 19 >= 4) & (py % 19 <= 12))
            trace |= (py % 19 == 12) & (px % 23 >= 4) & (px % 23 <= 10)
            paving[trace] = _shade(floor, tint - 13)
            stone[trace] = _shade(wall, tint + 24)
            stone[(px % 23 == 10) & (py % 19 == 12)] = _shade(wall, tint + 37)
        elif style == "ember":
            fault = ((px + py // 3) % 19 == 0) | ((py - px // 5) % 17 == 0)
            paving[fault] = _shade(floor, tint - 9)
            stone[fault] = _shade(wall, tint - 22)
        else:  # Diagonal marble inlay is held out with the test palette.
            inlay = ((px + py) % 23 == 0) | ((px - py) % 23 == 0)
            paving[inlay] = _shade(floor, tint - 11)
            stone[(px + py) % 23 == 0] = _shade(wall, tint + 14)
    for material in materials:
        material.setflags(write=False)
    return tuple(materials)


def _light_surfaces(result, frame, playable, style, seed):
    """Episode-fixed light and material contrast, restricted to scenery RGB."""
    rng = np.random.default_rng(seed)
    cast = rng.integers(-7, 8, size=3)
    contrast = float(rng.choice((.65, .85, 1., 1.2, 1.4)))
    slope = int(rng.choice((-8, -4, 4, 8)))
    height = frame.shape[0]
    light = np.linspace(-slope, slope, height)[:, None, None] + cast
    for color, base in zip((BACKGROUND, FLOOR, WALL), _PALETTES[style][:3], strict=True):
        mask = np.all(frame == color, axis=-1) & playable
        center = np.asarray(base, dtype=np.float32)
        lit = center + (result.astype(np.float32) - center) * contrast + light
        result[mask] = np.clip(np.rint(lit[mask]), 0, 255).astype(np.uint8)


def _neighbors(mask):
    above, below, left, right = (np.zeros_like(mask) for _ in range(4))
    above[1:], below[:-1] = mask[:-1], mask[1:]
    left[:, 1:], right[:, :-1] = mask[:, :-1], mask[:, 1:]
    return above, below, left, right


def _components(mask):
    """Connected run boxes, without a computer-vision/runtime dependency."""
    boxes, parents, previous = [], [], []

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for y in np.flatnonzero(mask.any(axis=1)):
        edges = np.diff(np.r_[False, mask[y], False].astype(np.int8))
        runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))
        current = []
        for x0, x1 in runs:
            touching = [root(index) for px0, px1, py, index in previous
                        if py == y - 1 and x0 < px1 and x1 > px0]
            index = min(touching) if touching else len(boxes)
            if not touching:
                boxes.append([int(x0), int(y), int(x1), int(y + 1)])
                parents.append(index)
            for other in touching:
                if other != index:
                    a, b = boxes[index], boxes[other]
                    boxes[index] = [min(a[0], b[0]), min(a[1], b[1]),
                                    max(a[2], b[2]), max(a[3], b[3])]
                    parents[other] = index
            box = boxes[index]
            boxes[index] = [min(box[0], int(x0)), box[1], max(box[2], int(x1)), int(y + 1)]
            current.append((x0, x1, y, index))
        previous = current
    return [tuple(box) for i, box in enumerate(boxes) if root(i) == i]


def _sprite(result, mask, color, role, *, token=False, simple=False):
    for x0, y0, x1, y1 in _components(mask):
        width, height = x1 - x0, y1 - y0
        if min(width, height) < 5:
            continue  # Health pips, counters, and thin outlines remain exact.
        region = mask[y0:y1, x0:x1]
        patch = result[y0:y1, x0:x1]
        if role == "exit" and width < 7 and height >= 10:
            patch[region] = _shade(color, -37)
            patch[:, width // 2][region[:, width // 2]] = _shade(color, 45)
            patch[:2][region[:2]] = color
            continue
        glyph = _GLYPHS[role]
        gh, gw = len(glyph), len(glyph[0])
        if width < gw or height < gh:
            # Five-pixel agents still get a tiny visor, entirely within their mask.
            if role == "agent":
                strip = np.zeros_like(region)
                strip[height // 3, max(1, width // 3):max(2, width - 1)] = True
                patch[strip & region] = (218, 247, 255)
            continue
        above, below, left, right = _neighbors(region)
        border = region & ~(above & below & left & right)
        patch[region] = _shade(color, -65 if token else -12)
        patch[border] = _shade(color, -65 if simple else -23)
        # Oversized grid cells are tokens; small free-moving sprites fill their
        # original silhouette, keeping the perceived collision footprint intact.
        scale = max(1, min(width // gw, height // gh))
        scale = min(scale, 2)
        gx, gy = (width - gw * scale) // 2, (height - gh * scale) // 2
        palette = {"#": color, "+": _shade(color, 37), "s": _shade(color, -45),
                   "i": (22, 31, 43), "w": (233, 244, 238)}
        if simple:
            palette = dict.fromkeys(palette, color)
        for row, line in enumerate(glyph):
            for column, mark in enumerate(line):
                if mark == ".":
                    continue
                ya, xa = gy + row * scale, gx + column * scale
                target = patch[ya:ya + scale, xa:xa + scale]
                visible = region[ya:ya + scale, xa:xa + scale]
                target[visible] = palette[mark]


def _surfaces(result, frame, playable, style, task):
    floor = np.all(frame == FLOOR, axis=-1) & playable
    wall = np.all(frame == WALL, axis=-1) & playable
    ground, paving, stone, _ = _PALETTES[style]
    above, below, left, right = _neighbors(floor)
    if task == "steering":
        result[floor] = _shade(stone, 13)
        result[floor & (~left | ~right)] = _shade(paving, 15)
        # Row extents come from visible road pixels, including curves.
        for y in np.flatnonzero(floor.any(axis=1)):
            columns = np.flatnonzero(floor[y])
            center = (int(columns[0]) + int(columns[-1])) // 2
            if y % 11 < 5 and floor[y, center]:
                result[y, center] = _shade(paving, 15)
        return
    if task == "camera":
        ys = np.flatnonzero(wall.any(axis=1))
        horizon = int(ys[0]) if len(ys) else frame.shape[0]
        for y in ys:
            result[y, wall[y]] = _shade(stone, min(10, (int(y) - horizon) // 7) - 9)
        result[floor] = _shade(paving, -28)
        return
    if task == "platforming":
        result[floor] = _shade(stone, 3)
        result[floor & ~above] = _shade(paving, -3)
        result[floor & np.roll(~above, 1, axis=0)] = _shade(paving, -20)
    elif task in _GRID_TASKS:
        # Thin seams follow the public five-by-five viewport, never a second
        # competing tile grid. Small chips supply material without visual noise.
        ys = np.linspace(30, frame.shape[0] - 3, 6, dtype=int)
        xs = np.linspace(3, frame.shape[1] - 3, 6, dtype=int)
        for y in ys:
            result[y, floor[y]] = _shade(paving, -16)
        for x in xs:
            result[floor[:, x], x] = _shade(paving, -16)
    if task in {"platforming", "timing"}:
        a, b, wall_left, wall_right = _neighbors(wall)
        result[wall] = _shade(stone, 1)
        result[wall & (~a | ~wall_left)] = _shade(stone, 30)
        result[wall & (~b | ~wall_right)] = _shade(stone, -20)
        if task == "timing":
            # Metal tubes with a restrained highlight, not stone columns.
            for x0, y0, x1, y1 in _components(wall):
                if x1 - x0 >= 4:
                    region = wall[y0:y1, x0:x1]
                    patch = result[y0:y1, x0:x1]
                    patch[:, 1:3][region[:, 1:3]] = _shade(stone, 24)
                    patch[:2][region[:2]] = _shade(stone, 30)
                    patch[-2:][region[-2:]] = _shade(stone, 30)
        else:
            # Stone crates have an inset diagonal brace, within solid RGB only.
            for x0, y0, x1, y1 in _components(wall):
                if min(x1 - x0, y1 - y0) < 5:
                    continue
                yy, xx = np.indices((y1 - y0, x1 - x0))
                brace = abs(xx / (x1 - x0 - 1) - yy / (y1 - y0 - 1)) < .12
                brace |= abs(1 - xx / (x1 - x0 - 1) - yy / (y1 - y0 - 1)) < .12
                patch = result[y0:y1, x0:x1]
                patch[brace & wall[y0:y1, x0:x1]] = _shade(stone, 20)


def style_pixel_art(frame: np.ndarray, style: str, seed: int, *, textures: bool = True,
                    task: str | None = None):
    """Render scenery and sprites using RGB masks; the HUD is never touched."""
    from carryrl.rewards.visual import HUD_HEIGHT

    if style not in _PALETTES:
        raise ValueError("Unknown pixel-art style")
    result = frame.copy()
    materials = _materials(frame.shape[0], style, seed, textures, task)
    playable = np.ones(frame.shape[:2], dtype=bool)
    playable[:HUD_HEIGHT] = False
    for color, material in zip((BACKGROUND, FLOOR, WALL), materials, strict=True):
        mask = np.all(frame == color, axis=-1) & playable
        result[mask] = material[mask]
    if not textures:
        return result
    _surfaces(result, frame, playable, style, task)
    _light_surfaces(result, frame, playable, style, seed)
    if task == "hazards":
        neutral = np.all(frame == (122, 133, 153), axis=-1) & playable
        result[neutral] = _shade(_PALETTES[style][1], -20)
    roles = dict(_ROLES)
    roles[PLAYER] = "car" if task == "steering" else "bird" if task == "timing" else "agent"
    roles[TARGET] = "exit"
    for color, role in roles.items():
        if color == PLAYER and task in {"mouse", "camera"}:
            continue  # Crosshairs remain literal, readable physical mouse controls.
        mask = np.all(frame == color, axis=-1) & playable
        if task == "mouse" and color == TARGET:
            # A visible target gains concentric aiming rings inside its hit area.
            for x0, y0, x1, y1 in _components(mask):
                yy, xx = np.indices((y1 - y0, x1 - x0))
                radius = min(x1 - x0, y1 - y0) / 2
                distance = np.hypot(xx - (x1 - x0 - 1) / 2, yy - (y1 - y0 - 1) / 2)
                patch = result[y0:y1, x0:x1]
                region = mask[y0:y1, x0:x1]
                patch[region & (distance < radius * .65)] = _shade(TARGET, -55)
                patch[region & (distance < radius * .3)] = _shade(TARGET, 30)
            continue
        if task == "reaction" and color == PLAYER:
            a, b, _, _ = _neighbors(mask)
            result[mask & ~a] = _shade(PLAYER, 45)
            result[mask & ~b] = _shade(PLAYER, -45)
            continue
        _sprite(result, mask, color, role, token=task in _GRID_TASKS or task == "ui",
                simple=task == "ui")
    return result
