"""Fresh factorized text supervision; specifications precede split-owned wording.

Offline only. This module never reads runtime instructions, game names, model
predictions, or the lexical grounding matcher. Character witnesses are emitted by
the renderer as it writes control names. They are independent grounding oracles.
"""

import hashlib
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from carryrl.rewards.control_text_data import CONTROL_CANDIDATES

CONTROL_TEXT_DATA_VERSION = "carryrl.control_text_data.v14"
PROTOCOL_VERSION = "carryrl.control_text_factorized.v2"
GENERATION_REVISION = "carryrl.control_text_data.v14.r4"
DEFAULT_SEED = 20261024
SPLIT_GROUPS = {"train": 2544, "dev": 600, "calibration": 400, "sealed_test": 600}
APPLICABILITY = ("supported", "unsupported", "unspecified")
MODES = ("held", "press", "relative", "unspecified")
PAIR_RELATIONS = ("same", "different", "unspecified")
_BASE_SCENARIO_KINDS = (
    "complete", "incomplete", "mixed_known_unknown", "mode_unspecified",
    "conditional", "negation", "citation", "contradiction", "aliases",
    "pair_unspecified", "wasd", "horizontal", "pointer", "morphology",
    "length_order",
)
HETEROGENEOUS_KINDS = (
    "heterogeneous_key_aliases", "cross_modal_aliases", "partial_alias_modes",
)
SCENARIO_KINDS = _BASE_SCENARIO_KINDS
# Number of semantic groups of EACH heterogeneous kind. They replace ordinary
# compositions within the unchanged overall split quotas; six views add no units.
HETEROGENEOUS_GROUP_QUOTAS = {"train": 48, "dev": 8, "calibration": 8, "sealed_test": 16}
_RELATIVE = tuple(c for c in CONTROL_CANDIDATES if c.startswith(("dx:", "dy:", "wheel:")))
_DIGITAL = tuple(c for c in CONTROL_CANDIDATES if c not in _RELATIVE)
_NAVIGATION = (("key:w", "key:up"), ("key:s", "key:down"),
               ("key:a", "key:left"), ("key:d", "key:right"))
# These are binding grammars, not only different wrapper strings. No family is
# reused across a split, including DEV temperature versus selection subsets.
_GRAMMARS = {
    "train": (0, 1, 2), "dev_temperature": (3, 4, 5),
    "dev_selection": (6, 7, 8), "calibration": (9, 10, 11),
    "sealed_test": (12, 13, 14),
}


@dataclass(frozen=True)
class Operation:
    candidates: tuple[str, ...]
    mode: str
    condition: str | None = None
    mode_overrides: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if (not isinstance(self.candidates, tuple) or not self.candidates
                or len(set(self.candidates)) != len(self.candidates)
                or not set(self.candidates) <= set(CONTROL_CANDIDATES)
                or self.mode not in MODES):
            raise ValueError("invalid factorized operation")
        overrides = (tuple(self.mode_overrides.items()) if isinstance(self.mode_overrides, Mapping)
                     else tuple(self.mode_overrides))
        if any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in overrides):
            raise ValueError("mode overrides require candidate/mode pairs")
        overrides = tuple(tuple(item) for item in overrides)
        if (len({candidate for candidate, _ in overrides}) != len(overrides)
                or any(candidate not in self.candidates or mode not in MODES
                       for candidate, mode in overrides)):
            raise ValueError("invalid or duplicated per-candidate mode override")
        ordered = tuple((candidate, dict(overrides)[candidate]) for candidate in self.candidates
                        if candidate in dict(overrides))
        object.__setattr__(self, "mode_overrides", ordered)
        for candidate in self.candidates:
            mode = self.mode_for(candidate)
            if candidate in _RELATIVE and mode not in {"relative", "unspecified"}:
                raise ValueError("relative input requires relative or unspecified timing")
            if candidate in _DIGITAL and mode == "relative":
                raise ValueError("digital input cannot have relative timing")

    def mode_for(self, candidate):
        if candidate not in self.candidates:
            raise ValueError("candidate is not a member of this operation")
        return dict(self.mode_overrides).get(candidate, self.mode)


@dataclass(frozen=True)
class FactorizedSpec:
    operations: tuple[Operation, ...]
    unsupported: tuple[str, ...] = ()
    unspecified: tuple[str, ...] = ()
    complete: bool = False
    pair_knowledge: bool = True
    kind: str = "incomplete"
    contradiction: tuple[str, ...] = ()
    citation: tuple[str, ...] = ()


def annotate_factorized_spec(spec):
    """Derive independent fields without making timing a support prerequisite."""
    labels = {c: {"candidate": c,
                  "applicability": "unsupported" if spec.complete else "unspecified",
                  "mode": "unspecified"} for c in CONTROL_CANDIDATES}
    memberships, assigned = {}, set()
    for group, operation in enumerate(spec.operations, 1):
        if operation.mode not in MODES or not operation.candidates:
            raise ValueError("invalid factorized operation")
        for candidate in operation.candidates:
            if candidate not in labels or candidate in assigned:
                raise ValueError("invalid or duplicated candidate")
            mode = operation.mode_for(candidate)
            if candidate in _RELATIVE and mode not in {"relative", "unspecified"}:
                raise ValueError("relative input requires relative or unspecified timing")
            if candidate in _DIGITAL and mode == "relative":
                raise ValueError("digital input cannot have relative timing")
            assigned.add(candidate)
            memberships[candidate] = group
            labels[candidate].update(applicability="supported", mode=mode)
    for candidates, status in ((spec.unsupported, "unsupported"),
                               (spec.unspecified + spec.contradiction, "unspecified")):
        for candidate in candidates:
            if candidate not in labels or candidate in assigned:
                raise ValueError("overlapping or invalid factorized evidence")
            assigned.add(candidate)
            labels[candidate].update(applicability=status, mode="unspecified")
    pairs = []
    for left, right in itertools.combinations(sorted(memberships), 2):
        relation = ("same" if memberships[left] == memberships[right] else "different")
        if not spec.pair_knowledge and memberships[left] != memberships[right]:
            relation = "unspecified"
        pairs.append({"left": left, "right": right, "relation": relation})
    legacy = []
    for candidate, label in labels.items():
        legacy.append({"candidate": candidate,
                       "status": label["applicability"].replace("unspecified", "unknown"),
                       "mode": label["mode"] if label["mode"] in {"held", "press"} else "unknown",
                       "alias_group": memberships.get(candidate, 0) if spec.pair_knowledge else 0})
    return list(labels.values()), pairs, legacy


def factorized_mapping_identity(labels, pairs):
    semantic = {"fields": sorted((v["candidate"], v["applicability"], v["mode"])
                                  for v in labels),
                "relations": sorted((v["left"], v["right"], v["relation"]) for v in pairs)}
    return hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()


def _name(candidate, language, variant):
    ru = language == "ru"
    category, value = candidate.split(":")
    if category == "key":
        if value == "space":
            return ("пробела" if variant == 1 else "пробел") if ru else "Space"
        if value in {"left", "right", "up", "down"}:
            directions = {"left": "влево", "right": "вправо", "up": "вверх", "down": "вниз"}
            return f"стрелка {directions[value]}" if ru else f"{value.title()} arrow"
        return f"клавиша {value.upper()}" if ru else f"{value.upper()} key"
    if category == "mouse":
        adjective = {"left": "левая", "right": "правая", "middle": "средняя"}[value]
        return f"{adjective} кнопка мыши" if ru else f"{value} mouse button"
    return {
        "dx:-": ("движение мыши влево", "mouse movement left"),
        "dx:+": ("движение мыши вправо", "mouse movement right"),
        "dy:-": ("движение мыши вверх", "mouse movement up"),
        "dy:+": ("движение мыши вниз", "mouse movement down"),
        "wheel:-": ("прокрутка колеса вниз", "wheel scrolling down"),
        "wheel:+": ("прокрутка колеса вверх", "wheel scrolling up"),
    }[candidate][0 if ru else 1]


# Fifteen actual clause compositions. Their grammar IDs are reserved before
# translation. The same meaning is rendered in RU and EN by the paired entry.
_BINDINGS = (
    ("Операция {id}: {names}. Режим: {mode}.", "Operation {id}: {names}. Mode: {mode}."),
    ("Для операции {id} доступны {names}; используйте так: {mode}.",
     "For operation {id}, {names} are enabled; use this timing: {mode}."),
    ("{names} выполняет операцию {id} с режимом «{mode}».",
     "{names} performs operation {id} with timing '{mode}'."),
    ("Связь операции {id} с вводом {names} задана; длительность — {mode}.",
     "Operation {id} is bound to {names}; its duration is {mode}."),
    ("Вызов {id} через {names} разрешён. Для этого ввода указано: {mode}.",
     "Invoking {id} through {names} is allowed. Its input timing is {mode}."),
    ("Поддерживаемый ввод {names}: назначение {id}, исполнение {mode}.",
     "Enabled input {names}: function {id}, execution {mode}."),
    ("Чтобы выполнить {id}, используйте {names}, причём {mode}.",
     "To perform {id}, use {names}, with timing {mode}."),
    ("Правило {id} задаёт ввод {names} и способ исполнения: {mode}.",
     "Rule {id} defines input {names} and its execution timing: {mode}."),
    ("Привязка {id} допускает {names}; временной режим этого ввода: {mode}.",
     "Binding {id} permits {names}; this input has temporal mode {mode}."),
    ("Разрешение на {names} относится к {id}, с исполнением {mode}.",
     "Permission for {names} belongs to {id}, executed with timing {mode}."),
    ("{id} можно вызвать вводом {names}. Временное правило: {mode}.",
     "{id} may be invoked using {names}. Temporal rule: {mode}."),
    ("Запись {id} содержит доступный ввод {names}; способ: {mode}.",
     "Entry {id} lists enabled input {names}; timing: {mode}."),
    ("Исполнение {mode} для {names} вызывает {id}.",
     "Execution with timing {mode} for {names} invokes {id}."),
    ("В таблице привязок {id} включён через {names}; режим исполнения — {mode}.",
     "In the binding table, {id} is enabled via {names}; execution mode is {mode}."),
    ("Доступная команда {id} получает {names}; для длительности записано «{mode}».",
     "Enabled command {id} receives {names}; its duration is recorded as '{mode}'."),
)
# Every partition has a prose family with distinct described functions instead
# of arbitrary action identifiers. The function names are annotation-neutral:
# their assignment is local to this specification and never a task lookup.
_NATURAL_BINDINGS = {
    2: ("{names}: {mode}, чтобы {function}.",
        "{names}: {mode}, to {function}."),
    5: ("Чтобы {function}, используйте {names}; {mode}.",
        "To {function}, use {names}; {mode}."),
    8: ("Через {names} можно {function}; исполнение — {mode}.",
        "Through {names} you can {function}; execution uses {mode}."),
    11: ("Задача — {function}. Для неё доступны {names}, с режимом {mode}.",
         "The task is to {function}. Its enabled inputs are {names}, with timing {mode}."),
    14: ("Для того чтобы {function}, служит {names}; применяйте {mode}.",
         "To accomplish this function, {function}, use {names}; apply {mode}."),
}
_IMPLICIT_MODE_BINDINGS = (
    ("{names} позволяет {function}.", "{names} lets you {function}."),
    ("Для того чтобы {function}, доступен {names}.",
     "To {function}, {names} is enabled."),
    ("С помощью {names} можно {function}.", "Using {names}, you can {function}."),
    ("Чтобы {function}, служит {names}.", "To {function}, use {names}."),
    ("{names} выполняет функцию: {function}.", "{names} performs this function: {function}."),
)
_PAIR_IMPLICIT_MODE_BINDINGS = (
    ("Поддерживается {names}.", "{names} is supported."),
    ("Доступный способ ввода — {names}.", "An enabled input is {names}."),
    ("Ввод {names} разрешён.", "Input {names} is allowed."),
    ("Можно использовать {names}.", "You may use {names}."),
    ("В справке разрешён {names}.", "This reference permits {names}."),
)

_PAIR_UNSPECIFIED_BINDINGS = (
    ("Доступный ввод: {names}; {mode}.", "Enabled input: {names}; {mode}."),
    ("Включён {names}; способ исполнения — {mode}.",
     "{names} is enabled; its execution timing is {mode}."),
    ("{names} допустим; длительность задаётся так: {mode}.",
     "{names} is permitted; its duration is specified as {mode}."),
    ("Разрешённый ввод — {names}, временной режим {mode}.",
     "An allowed input is {names}, with temporal rule {mode}."),
    ("Можно использовать {names}; для времени исполнения указано {mode}.",
     "You may use {names}; execution timing is described as {mode}."),
)
_PAIR_UNSPECIFIED_ENDINGS = (
    ("Отношения между отдельно описанными действиями не указаны. ",
     "Relations between separately described actions are unspecified. "),
    ("Для отдельных описаний не сообщается, совпадают ли их действия. ",
     "Separate descriptions do not state whether their actions coincide. "),
    ("Эквивалентность отдельно перечисленных действий не определена. ",
     "The equivalence of separately listed actions is not defined. "),
    ("Неизвестно, относятся ли разные описания к одному действию. ",
     "Whether separate descriptions refer to one action is unknown. "),
    ("Связь между действиями отдельных описаний оставлена без пояснений. ",
     "The relation between actions in separate descriptions is left unexplained. "),
)
_COMPACT_NAVIGATION = (
    ("Двигайте маркер вверх, влево, вниз и вправо клавишами {keys} соответственно; "
     "{arrows} соответствующих направлений — равнозначные замены. ",
     "Move the marker up, left, down and right with keys {keys} respectively; "
     "{arrows} of the matching directions are equivalent alternatives. "),
    ("Клавиши {keys} задают по порядку движения вверх, влево, вниз, вправо; "
     "каждую из них заменяют {arrows} соответствующего направления. ",
     "Keys {keys} specify movement up, left, down, right in that order; "
     "each is interchangeable with {arrows} of the corresponding direction. "),
    ("Через {keys} доступны направления вверх, влево, вниз, вправо в этом порядке. "
     "Ввод {arrows} с соответствующим направлением выполняет то же движение. ",
     "Through {keys}, movement up, left, down, right is available in that order. "
     "Input from {arrows} in the corresponding direction performs the same movement. "),
    ("Для четырёх направлений — вверх, влево, вниз, вправо — служит {keys} по порядку. "
     "Как эквивалентную замену используйте {arrows} соответствующего направления. ",
     "For the four directions up, left, down, right, use {keys} in order. "
     "As an equivalent alternative, use {arrows} of the matching direction. "),
    ("Движение выполняется через {keys}: вверх, влево, вниз, вправо соответственно. "
     "Те же движения допускают {arrows} соответствующих направлений. ",
     "Movement uses {keys}: up, left, down, right respectively. "
     "The same movements also accept {arrows} of the corresponding directions. "),
)
_COMPACT_HORIZONTAL = (
    ("Двигайте маркер влево и вправо клавишами {keys} соответственно; "
     "{arrows} — равнозначные замены для этих направлений. ",
     "Move the marker left and right using keys {keys} respectively; "
     "{arrows} are equivalent alternatives for those directions. "),
    ("Клавиши {keys} передвигают маркер влево и вправо по порядку. "
     "Те же действия выполняют {arrows} соответственно. ",
     "Keys {keys} move the marker left and right in order. "
     "The same actions are performed by {arrows}, respectively. "),
    ("Доступны два движения: влево и вправо через {keys} соответственно. "
     "Для каждого движения равнозначны {arrows} в соответствующую сторону. ",
     "Two movements are enabled: left and right via {keys}, respectively. "
     "For each movement, {arrows} in the matching direction are equivalent. "),
    ("Пара {keys} предназначена для движения влево и вправо именно в этом порядке; "
     "вместо неё доступны {arrows} с тем же соответствием. ",
     "The pair {keys} is assigned to left and right movement in exactly that order; "
     "{arrows} are available alternatives with the same correspondence. "),
    ("Направления влево и вправо вызываются через {keys} соответственно. "
     "Альтернативное управление теми же направлениями — {arrows} по порядку. ",
     "Directions left and right are invoked through {keys}, respectively. "
     "Alternative controls for the same directions are {arrows} in order. "),
)
_COMPACT_TIMING = (
    ("Для движения {direction}: {mode}. ", "For movement {direction}: {mode}. "),
    ("Движение {direction} выполняется так: {mode}. ",
     "Movement {direction} is executed this way: {mode}. "),
    ("В направлении {direction} применяется {mode}. ",
     "In direction {direction}, use timing {mode}. "),
    ("Режим движения {direction} — {mode}. ",
     "The mode for movement {direction} is {mode}. "),
    ("Чтобы двигаться {direction}, требуется {mode}. ",
     "To move {direction}, the required timing is {mode}. "),
)

_FUNCTIONS = (
    ("передвинуть синий маркер", "move the blue marker"),
    ("открыть медные ворота", "open the copper gate"),
    ("повернуть стеклянную призму", "rotate the glass prism"),
    ("включить маяк", "activate the beacon"),
    ("поднять платформу", "raise the platform"),
    ("опустить мост", "lower the bridge"),
    ("переключить лампу", "switch the lamp"),
    ("запустить дрон", "launch the drone"),
    ("выбрать звезду", "select the star"),
    ("сбросить индикатор", "reset the gauge"),
    ("отметить цель", "mark the target"),
    ("закрыть панель", "close the panel"),
    ("сменить цвет фигуры", "change the shape color"),
    ("сохранить снимок", "save the picture"),
)

_MODE_TEXT = {
    "held": ("удерживать несколько последовательных интервалов до отпускания",
             "hold for several consecutive intervals until release"),
    "press": ("короткое нажатие на один интервал, затем отпускание перед повтором",
              "a brief press for one interval, then release before repeating"),
    "relative": ("относительное смещение за каждый интервал без нажатия и отпускания",
                 "relative displacement per interval without a press or release"),
    "unspecified": ("длительность не указана", "duration is unspecified"),
}


class _Writer:
    def __init__(self):
        self.text = ""
        self.mentions = []

    def add(self, text):
        self.text += text

    def name(self, candidate, language, variant):
        value = _name(candidate, language, variant)
        start = len(self.text)
        self.add(value)
        self.mentions.append({"candidate": candidate, "start": start, "end": len(self.text)})

    def compact(self, spelling, candidates):
        start = len(self.text)
        self.add(spelling)
        for candidate in candidates:
            self.mentions.append({"candidate": candidate, "start": start, "end": len(self.text)})


def _compact_direction_bindings(writer, spec, language, grammar):
    """Describe compact physical groups without spelling their members again.

    Directional ordering explicitly supplies the alias mapping. Timing remains
    per direction, so a mode contrast can change only one direction's timing.
    """
    ru, li = language == "ru", 0 if language == "ru" else 1
    navigation = spec.kind == "wasd"
    groups = _NAVIGATION if navigation else _NAVIGATION[2:]
    selected = {frozenset(pair): next(op for op in spec.operations
                                     if frozenset(op.candidates) == frozenset(pair))
                for pair in groups}
    family = grammar // 3
    template = (_COMPACT_NAVIGATION if navigation else _COMPACT_HORIZONTAL)[family][li]
    before, between, after = template.replace("{keys}", "\0").replace("{arrows}", "\0").split("\0")
    writer.add(before)
    writer.compact("WASD" if navigation else "A/D",
                   ("key:w", "key:a", "key:s", "key:d") if navigation else ("key:a", "key:d"))
    writer.add(between)
    writer.compact(("стрелки" if ru else "the arrow keys") if navigation else
                   ("стрелки влево/вправо" if ru else "left/right arrows"),
                   ("key:up", "key:left", "key:down", "key:right") if navigation else
                   ("key:left", "key:right"))
    writer.add(after)
    directions = {"key:w": ("вверх", "up"), "key:s": ("вниз", "down"),
                  "key:a": ("влево", "left"), "key:d": ("вправо", "right")}
    for pair in groups:
        direction = directions[pair[0]][li]
        operation = selected[frozenset(pair)]
        if len({operation.mode_for(candidate) for candidate in operation.candidates}) != 1:
            raise ValueError("compact direction timing requires homogeneous physical modes")
        timing = _MODE_TEXT[operation.mode_for(pair[0])][li]
        writer.add(_COMPACT_TIMING[family][li].format(direction=direction, mode=timing))
    return {candidate for pair in groups for candidate in pair}


def rendering_metadata(spec, *, grammar, variant, seed=0):
    compact = (spec.kind in {"wasd", "horizontal"} and variant == 1
               and all(len({op.mode_for(c) for c in op.candidates}) == 1
                       for op in spec.operations))
    return {"compact_only": compact,
            "natural_no_action_ids": (grammar in _NATURAL_BINDINGS
                                      or not spec.pair_knowledge or compact),
            "condition_present": any(op.condition for op in spec.operations) and variant != 1,
            "implicit_mode_absence": any(op.mode_for(c) == "unspecified" for op in spec.operations
                                        for c in op.candidates) and variant == 2,
            "heterogeneous_alias_modes": any(
                len({op.mode_for(c) for c in op.candidates}) > 1 for op in spec.operations),
            "train_timing_context": grammar == 2 and variant == 2 and seed % 4 == 0}


def render_factorized_spec(spec, *, language, grammar, variant, seed, include_binding_spans=False):
    """Return text plus exact control witnesses; no matcher or tokenizer is used."""
    if language not in {"ru", "en"} or not 0 <= grammar < len(_BINDINGS):
        raise ValueError("unknown rendering grammar")
    ru, li = language == "ru", 0 if language == "ru" else 1
    writer = _Writer()
    binding_spans = []
    operations = list(enumerate(spec.operations, 1))
    random.Random(seed).shuffle(operations)
    if spec.kind == "length_order":
        writer.add(("Цель — завершить маршрут и сохранить результат. " if ru else
                    "The objective is to complete the route and retain the result. ")
                   * (variant + 2))
    metadata = rendering_metadata(spec, grammar=grammar, variant=variant, seed=seed)
    compact_candidates = (_compact_direction_bindings(writer, spec, language, grammar)
                          if metadata["compact_only"] else set())
    for number, operation in operations:
        if set(operation.candidates) <= compact_candidates:
            continue
        heterogeneous = len({operation.mode_for(c) for c in operation.candidates}) > 1
        physical_groups = ([(candidate,) for candidate in operation.candidates] if heterogeneous
                           else [operation.candidates])
        for physical_index, candidates in enumerate(physical_groups):
            mode = operation.mode_for(candidates[0])
            implicit_mode = (mode == "unspecified" and variant == 2
                             and not metadata["train_timing_context"])
            prose = (_NATURAL_BINDINGS[(grammar // 3) * 3 + 2] if metadata["compact_only"] else
                     _NATURAL_BINDINGS.get(grammar, _BINDINGS[grammar]))
            template = _IMPLICIT_MODE_BINDINGS[grammar // 3][li] if implicit_mode else prose[li]
            mode_payload = _MODE_TEXT[mode][li]
            if metadata["train_timing_context"]:
                if mode == "unspecified":
                    mode_payload = "не указана" if ru else "not specified"
                template = ("{names} помогает {function}; длительность: {mode}." if ru else
                            "{names} is available to {function}; duration: {mode}.")
            # Unknown inter-operation relations do not erase known within-operation aliases.
            if not spec.pair_knowledge:
                template = (_PAIR_IMPLICIT_MODE_BINDINGS[grammar // 3][li] if implicit_mode else
                            _PAIR_UNSPECIFIED_BINDINGS[grammar // 3][li])
                if metadata["train_timing_context"]:
                    # A shared named function would invent a cross-operation relation.
                    template = ("{names} разрешён для отдельно описанного действия; "
                                "длительность: {mode}." if ru else
                                "{names} is enabled for a separately described action; "
                                "duration: {mode}.")
            if physical_index:
                writer.add("Равнозначный способ выполнить то же действие: " if ru else
                           "An equivalent way to perform that same action: ")
            if operation.condition and metadata["condition_present"]:
                writer.add("Когда горит сигнал, " if ru else "When the signal is lit, ")
            binding_start, mention_start = len(writer.text), len(writer.mentions)
            mode_span, function_span = None, None
            tagged = template.format(id=f"{number:02d}", function="\0function\0",
                                     names="\0names\0", mode="\0mode\0")
            for part in re.split(r"(\0(?:function|names|mode)\0)", tagged):
                if part == "\0names\0":
                    for position, candidate in enumerate(candidates):
                        if position:
                            writer.add(" или равнозначная замена " if ru else
                                       " or equivalent alternative ")
                        writer.name(candidate, language, variant)
                elif part == "\0mode\0":
                    start = len(writer.text)
                    writer.add(mode_payload)
                    mode_span = [start, len(writer.text)]
                elif part == "\0function\0":
                    start = len(writer.text)
                    writer.add(_FUNCTIONS[number - 1][li])
                    function_span = [start, len(writer.text)]
                else:
                    writer.add(part)
            binding_end = len(writer.text)
            last_name = max(v["end"] for v in writer.mentions[mention_start:])
            binding_spans.append({
                "candidates": list(candidates), "mode": mode, "operation_index": number - 1,
                "start": binding_start, "end": binding_end, "mode_payload_span": mode_span,
                "function_span": function_span,
                "duration_cue": bool(re.search(r"duration|длительность",
                                                writer.text[binding_start:binding_end], re.I)),
                "function_between_candidate_and_mode": bool(
                    function_span and mode_span and last_name <= function_span[0]
                    and function_span[1] <= mode_span[0]),
            })
            writer.add(" ")
    if spec.pair_knowledge and len(operations) > 1 and not metadata["natural_no_action_ids"]:
        writer.add("Разные номера обозначают разные действия. " if ru else
                   "Different numbers denote different actions. ")
    if not spec.pair_knowledge:
        writer.add(_PAIR_UNSPECIFIED_ENDINGS[grammar // 3][li])
    for candidate in spec.unsupported:
        writer.add("Не используйте " if ru else "Do not use ")
        writer.name(candidate, language, variant)
        writer.add(": этот ввод не поддерживается. " if ru else ": this input is unsupported. ")
    for candidate in spec.unspecified:
        writer.add("Для " if ru else "For ")
        writer.name(candidate, language, variant)
        writer.add(" применимость не описана. " if ru else " applicability is unspecified. ")
    for candidate in spec.contradiction:
        writer.name(candidate, language, variant)
        writer.add(" одновременно разрешён и запрещён; справка противоречива. " if ru else
                   " is both enabled and forbidden; the reference contradicts itself. ")
    for candidate in spec.citation:
        writer.add("Историческая цитата, не правило текущей игры: «" if ru else
                   "Historical quotation, not a rule of the current game: '")
        writer.name(candidate, language, variant)
        writer.add(" поддерживается». " if ru else " is supported'. ")
    writer.add(("Это полный список; все неописанные физические вводы не поддерживаются."
                if ru else "This list is complete; every undescribed physical input "
                "is unsupported.")
               if spec.complete else
               ("Это неполный список; о прочих вводах сведений нет." if ru else
                "This list is incomplete; other inputs are not described."))
    if include_binding_spans:
        return writer.text, writer.mentions, binding_spans
    return writer.text, writer.mentions


def _mandatory_schedule():
    """Each split covers each candidate, known/unspecified timing, completeness."""
    entries = []
    for candidate in CONTROL_CANDIDATES:
        for complete in (False, True):
            modes = ("relative", "unspecified") if candidate in _RELATIVE else (
                "held", "press", "unspecified")
            entries.extend(("mode_unspecified" if mode == "unspecified" else
                            "complete" if complete else "incomplete", candidate, mode, complete)
                           for mode in modes)
    entries.extend((kind, None, None, None) for kind in SCENARIO_KINDS)
    return entries


def _sample_spec(rng, kind, forced=None, forced_mode=None, complete=None):
    candidates = list(CONTROL_CANDIDATES)
    rng.shuffle(candidates)
    complete = bool(rng.randrange(2)) if complete is None else complete
    groups = []
    if kind == "wasd":
        groups = [Operation(pair, "held") for pair in _NAVIGATION]
    elif kind == "horizontal":
        groups = [Operation(pair, "held") for pair in _NAVIGATION[2:]]
    elif kind == "pointer":
        groups = [Operation((c,), "relative") for c in _RELATIVE]
    elif kind == "morphology":
        groups = [Operation(("key:space",), "press")]
    elif forced is not None:
        groups = [Operation((forced,), forced_mode)]
    elif kind in {"aliases", "pair_unspecified"}:
        selected = rng.sample(_DIGITAL, 2)
        groups = [Operation(tuple(selected), rng.choice(("held", "press", "unspecified")))]
    else:
        first = candidates[0]
        groups = [Operation((first,), "relative" if first in _RELATIVE else
                            "unspecified" if kind == "mode_unspecified" else
                            rng.choice(("held", "press")))]
    used = {c for op in groups for c in op.candidates}
    candidates = [c for c in candidates if c not in used]
    # Random companion mappings make all 4,144 semantic targets independently
    # unique; action names, syntax and translation never enter their identity.
    for _ in range(rng.randint(2, 7)):
        candidate = candidates.pop()
        mode = rng.choice(("relative", "unspecified")) if candidate in _RELATIVE else rng.choice(
            ("held", "press", "unspecified"))
        groups.append(Operation((candidate,), mode))
    if kind == "conditional":
        op = groups[0]
        groups[0] = replace(op, condition="indicator_lit")
    unsupported, unspecified, contradiction, citation = (candidates.pop(),), (), (), ()
    if kind == "contradiction":
        contradiction = (candidates.pop(),)
    elif kind == "citation":
        citation = (candidates.pop(),)
    if kind in {"mixed_known_unknown", "incomplete", "pair_unspecified"}:
        unspecified = (candidates.pop(),)
    return FactorizedSpec(tuple(groups), unsupported, unspecified, complete,
                          kind != "pair_unspecified", kind, contradiction, citation)


def _operation_mode_contrast(operation):
    """Change effective physical timing, preserving every unrelated override."""
    candidate = operation.candidates[0]
    current = operation.mode_for(candidate)
    other = {"held": "press", "press": "held", "relative": "unspecified",
             "unspecified": "relative" if candidate in _RELATIVE else "held"}[current]
    if not operation.mode_overrides:
        return replace(operation, mode=other)
    overrides = dict(operation.mode_overrides)
    overrides[candidate] = other
    return replace(operation, mode_overrides=overrides)


def _split_schedule(split, paired):
    schedule = _mandatory_schedule()
    if paired:
        schedule = [entry for entry in schedule if entry[2] != "press"
                    and not (entry[1] in _RELATIVE and entry[2] == "unspecified")]
    return schedule


def _heterogeneous_slots(split, count, schedule, width):
    """Reserve existing alias slots without changing R2 random draws or order."""
    eligible = []
    for index in range(count // width):
        kind = (schedule[index][0] if index < len(schedule) else
                _BASE_SCENARIO_KINDS[index % len(_BASE_SCENARIO_KINDS)])
        if kind in {"aliases", "pair_unspecified"}:
            eligible.append(index)
    per_kind = HETEROGENEOUS_GROUP_QUOTAS[split] // width
    return {index: kind for kind, start in zip(HETEROGENEOUS_KINDS,
                                              range(0, 3 * per_kind, per_kind), strict=True)
            for index in eligible[start:start + per_kind]}


def _with_heterogeneous_alias(spec, kind, seed):
    """Modify one reserved R2 alias operation; keep other draws reproducible."""
    rng = random.Random(seed)
    occupied = {c for op in spec.operations[1:] for c in op.candidates}
    occupied.update(spec.unsupported + spec.unspecified + spec.contradiction + spec.citation)
    keys = [c for c in CONTROL_CANDIDATES if c.startswith("key:") and c not in occupied]
    rng.shuffle(keys)
    if kind == "cross_modal_aliases":
        wheels = [c for c in ("wheel:-", "wheel:+") if c not in occupied]
        # A saturated reference may already mention both wheel directions.
        # Free one locally, without resampling this or later R2 specifications.
        fallback = tuple(c for c in ("wheel:-", "wheel:+") if c not in spec.unsupported)
        wheel = rng.choice(wheels or fallback)
        operations = tuple(replace(op, candidates=tuple(c for c in op.candidates if c != wheel),
                                   mode_overrides=tuple((c, m) for c, m in op.mode_overrides
                                                        if c != wheel))
                           for op in spec.operations[1:] if any(c != wheel for c in op.candidates))
        spec = replace(spec, operations=(spec.operations[0], *operations),
                       unsupported=tuple(c for c in spec.unsupported if c != wheel),
                       unspecified=tuple(c for c in spec.unspecified if c != wheel),
                       contradiction=tuple(c for c in spec.contradiction if c != wheel),
                       citation=tuple(c for c in spec.citation if c != wheel))
        operation = Operation((keys[0], wheel), "press", spec.operations[0].condition,
                              ((wheel, "relative"),))
    else:
        second_mode = "press" if kind == "heterogeneous_key_aliases" else "unspecified"
        operation = Operation(tuple(keys[:2]), "held", spec.operations[0].condition,
                              ((keys[1], second_mode),))
    return replace(spec, operations=(operation, *spec.operations[1:]))


def _generate_r3_rows(*, seed=DEFAULT_SEED, split_groups=None, excluded_mapping_ids=()):
    """Allocate semantic groups/DEV roles before rendering all six RU/EN views.

    TRAIN/DEV contain genuine target-changing minimal pairs in split-local
    contrast clusters. CAL/SEALED groups remain independent statistical units.
    """
    counts = dict(SPLIT_GROUPS if split_groups is None else split_groups)
    if set(counts) != set(SPLIT_GROUPS) or any(v <= 0 for v in counts.values()):
        raise ValueError("all four positive split group quotas are required")
    if any(counts[split] % 2 for split in ("train", "dev")):
        raise ValueError("TRAIN and DEV require even quotas for contrast pairs")
    rng, used, records = random.Random(seed), set(excluded_mapping_ids), []
    for split, count in counts.items():
        paired = split in {"train", "dev"}
        schedule = _split_schedule(split, paired)
        width = 2 if paired else 1
        heterogeneous_slots = _heterogeneous_slots(split, count, schedule, width)
        for cluster_index in range(count // width):
            kind, forced, mode, complete = (schedule[cluster_index]
                if cluster_index < len(schedule) else
                (_BASE_SCENARIO_KINDS[cluster_index % len(_BASE_SCENARIO_KINDS)],
                 None, None, None))
            heterogeneous_kind = heterogeneous_slots.get(cluster_index)
            contrast_type = ("completeness" if paired and (
                heterogeneous_kind or cluster_index >= len(schedule)
                and (cluster_index - len(schedule)) % 4 >= 2) else "mode")
            for _ in range(10000):
                original = _sample_spec(rng, kind, forced, mode, complete)
                if heterogeneous_kind:
                    local_seed = f"{seed}:{split}:{cluster_index}:heterogeneous"
                    original = _with_heterogeneous_alias(original, heterogeneous_kind, local_seed)
                specs = [original]
                if paired and contrast_type == "completeness":
                    specs.append(replace(original, complete=not original.complete))
                elif paired:
                    specs.append(replace(original, operations=(
                        _operation_mode_contrast(original.operations[0]),
                        *original.operations[1:])))
                targets = [annotate_factorized_spec(spec) for spec in specs]
                identities = [factorized_mapping_identity(labels, pairs)
                              for labels, pairs, _ in targets]
                if len(set(identities)) == width and not set(identities) & used:
                    break
            else:
                raise ValueError("unable to allocate a fresh semantic mapping")
            used.update(identities)
            role = (("temperature" if cluster_index % 2 == 0 else "selection")
                    if split == "dev" else None)
            owner = f"dev_{role}" if role else split
            cluster = f"v14-{split}-contrast-{cluster_index:04d}" if paired else None
            for member, (spec, identity, targets_one) in enumerate(zip(specs, identities, targets,
                                                                     strict=True)):
                index = cluster_index * width + member
                labels, pairs, legacy = targets_one
                group = f"v14-{split}-{index:04d}-{identity[:12]}"
                for language in ("ru", "en"):
                    for variant, grammar in enumerate(_GRAMMARS[owner]):
                        render_seed = seed + cluster_index * 19 + variant
                        text, mentions, bindings = render_factorized_spec(
                            spec, language=language, grammar=grammar, variant=variant,
                            seed=render_seed, include_binding_spans=True)
                        records.append({
                            "id": f"{group}-{language}-{variant}", "group_id": group,
                            "mapping_id": identity, "semantic_mapping_id": identity,
                            "split": split, "development_role": role,
                            "scenario_kind": kind, "language": language, "text": text,
                            "heterogeneous_alias_kind": heterogeneous_kind,
                            "template_family": f"v14-binding-{grammar:02d}",
                            "factorized_labels": labels, "pair_labels": pairs, "labels": legacy,
                            "expected_mentions": mentions, "binding_spans": bindings,
                            "specification": asdict(spec),
                            "minimal_pair_group": cluster or group,
                            "contrast_cluster_id": cluster,
                            "independent_group_id": cluster or group,
                            "contrast_member": member if paired else None,
                            "contrast_type": contrast_type if paired else None,
                            "rendering": rendering_metadata(spec, grammar=grammar, variant=variant,
                                                            seed=render_seed),
                            "render_seed": seed + cluster_index * 19 + variant,
                            "grammar": grammar, "variant": variant,
                        })
    return records



RELATIONAL_PANEL_VERSION = "carryrl.relational_panel.v1"
RELATIONAL_FAMILIES = ("simple", "compressed", "natural")
RELATIONAL_FOCAL_PAIRS = 48
RELATIONAL_CONTEXTS = 4
RELATIONAL_GROUPS = 576


def relational_focal_pairs(seed):
    """Prospective physical coverage, independent of games and model outcomes."""
    rng = random.Random(f"{seed}:relational-bank")
    candidates = list(CONTROL_CANDIDATES)
    rng.shuffle(candidates)
    # A cycle gives every physical command two focal neighbours. The remaining
    # edges add variety; there is no unsupported claim of all-630 coverage.
    pairs = {tuple(sorted((candidates[i], candidates[(i + 1) % len(candidates)])))
             for i in range(len(candidates))}
    while len(pairs) < RELATIONAL_FOCAL_PAIRS:
        pairs.add(tuple(sorted(rng.sample(candidates, 2))))
    result = sorted(pairs)
    rng.shuffle(result)
    return result


def _relational_spec(focal, background, modes, state, complete, unsupported):
    partitions = ([tuple(focal)] if state == "same" else [(focal[0],), (focal[1],)])
    partitions += [tuple(pair) for pair in background]
    operations = tuple(Operation(group, modes[group[0]], mode_overrides=tuple(
        (candidate, modes[candidate]) for candidate in group[1:]
        if modes[candidate] != modes[group[0]])) for group in partitions)
    return FactorizedSpec(operations, (unsupported,), complete=complete,
                          pair_knowledge=state != "unspecified", kind="relational")


def _relational_rendering(spec, family):
    return {"compact_only": False, "natural_no_action_ids": True, "condition_present": False,
            "implicit_mode_absence": family == "natural" and any(
                op.mode_for(c) == "unspecified" for op in spec.operations for c in op.candidates),
            "heterogeneous_alias_modes": any(
                len({op.mode_for(c) for c in op.candidates}) > 1 for op in spec.operations),
            "train_timing_context": False}


def render_relational_panel(spec, panel, *, language, seed):
    """TRAIN-owned physical bindings plus scoped, explicit relational evidence.

    No numbered functions are assigned to an unknown relation. The same physical
    body and order occur in all three counterfactual states of a base context.
    """
    ru, li, family = language == "ru", 0 if language == "ru" else 1, panel["family"]
    if family not in RELATIONAL_FAMILIES:
        raise ValueError("unknown relational TRAIN grammar")
    writer, bindings = _Writer(), []
    modes = {c: op.mode_for(c) for op in spec.operations for c in op.candidates}
    memberships = {c: i for i, op in enumerate(spec.operations) for c in op.candidates}
    order = sorted(modes)
    random.Random(seed).shuffle(order)
    starts = {
        "simple": ("Правила физических команд. ", "Physical command rules. "),
        "compressed": ("Краткая памятка ввода: ", "Compact input reference: "),
        "natural": ("Используйте перечисленные способы ввода. ",
                    "Use the input methods described here. "),
    }
    writer.add(starts[family][li])
    short_modes = {
        "held": ("удерживать до отпускания", "hold until release"),
        "press": ("один интервал, затем отпустить", "one interval, then release"),
        "relative": ("относительное смещение за интервал", "relative delta per interval"),
        "unspecified": ("режим не указан", "timing unspecified"),
    }
    for candidate in order:
        mode = modes[candidate]
        start = len(writer.text)
        if family == "simple":
            writer.add("Для ввода " if ru else "For input ")
        elif family == "natural":
            writer.add("Доступен ввод " if ru else "You may use ")
        writer.name(candidate, language, RELATIONAL_FAMILIES.index(family))
        implicit = family == "natural" and mode == "unspecified"
        mode_span = None
        if not implicit:
            writer.add(("; длительность: " if ru else "; duration: ")
                       if family == "simple" else ": ")
            left = len(writer.text)
            payload = short_modes[mode][li] if family == "compressed" else _MODE_TEXT[mode][li]
            writer.add(payload)
            mode_span = [left, len(writer.text)]
        writer.add("; " if family == "compressed" else ". ")
        bindings.append({"candidates": [candidate], "mode": mode,
                         "operation_index": memberships[candidate], "start": start,
                         "end": len(writer.text), "mode_payload_span": mode_span,
                         "function_span": None,
                         "duration_cue": family == "simple" or (not implicit
                             and mode == "unspecified" and family != "compressed"),
                         "function_between_candidate_and_mode": False})

    def pair_names(pair, join):
        writer.name(pair[0], language, RELATIONAL_FAMILIES.index(family))
        writer.add(join)
        writer.name(pair[1], language, RELATIONAL_FAMILIES.index(family))

    entries = [(pair, "same") for pair in panel["background_pairs"]]
    entries.append((panel["focal_pair"], panel["state"]))
    random.Random(f"{seed}:relation-order").shuffle(entries)
    for pair, relation in entries:
        if family == "compressed":
            writer.add("Связь " if ru else "Relation ")
            pair_names(pair, " / ")
            writer.add({"same": (": одно действие. ", ": same action. "),
                        "different": (": разные действия. ", ": different actions. "),
                        "unspecified": (": не указана. ", ": unspecified. ")}[relation][li])
        else:
            pair_names(pair, " и " if ru else " and ")
            if family == "natural":
                endings = {
                    "same": (" взаимозаменяемы для одной и той же функции. ",
                             " are interchangeable alternatives for one and the same function. "),
                    "different": (" служат для разных функций. ",
                                  " serve distinct functions. "),
                    "unspecified": (" доступны, но не описано, совпадают ли их функции. ",
                                    " are enabled, but whether their functions coincide "
                                    "is not stated. "),
                }
            else:
                endings = {
                    "same": (" — равнозначные способы выполнить одно действие. ",
                             " are equivalent ways to perform one action. "),
                    "different": (" выполняют разные действия. ", " perform different actions. "),
                    "unspecified": (" доступны, но связь их действий не уточнена. ",
                                    " are enabled, but their action relation is not stated. "),
                }
            writer.add(endings[relation][li])
    state = panel["state"]
    # Explicit scope prevents the unknown footer from retracting known same.
    if state == "unspecified":
        writer.add("Указанные выше равнозначности сохраняются; связи всех остальных пар "
                   "не определены. " if ru else
                   "The explicit equivalences above hold; relations of all other pairs "
                   "are unspecified. ")
    else:
        writer.add("Только явно названные равнозначные пары выполняют одно действие; "
                   "любая другая пара выполняет разные действия. " if ru else
                   "Only the explicitly equivalent pairs share an action; every other pair "
                   "performs different actions. ")
    writer.add("Не поддерживается " if ru else "Unsupported input: ")
    writer.name(spec.unsupported[0], language, RELATIONAL_FAMILIES.index(family))
    writer.add(". ")
    writer.add(("Список полный; прочие физические вводы не поддерживаются." if ru else
                "The list is complete; other physical inputs are unsupported.")
               if spec.complete else
               ("Список неполный; о прочих физических вводах сведений нет." if ru else
                "The list is incomplete; other physical inputs are not described."))
    return writer.text, writer.mentions, bindings


def _relational_triplets(seed, used, excluded):
    rows = []
    for focal_index, focal in enumerate(relational_focal_pairs(seed)):
        backgrounds_seen = set()
        for context in range(RELATIONAL_CONTEXTS):
            rng = random.Random(f"{seed}:relational-context:{focal_index}:{context}")
            for attempt in range(10000):
                others = [c for c in CONTROL_CANDIDATES if c not in focal]
                rng.shuffle(others)
                background = [tuple(sorted(others[:2])), tuple(sorted(others[2:4]))]
                signature = tuple(sorted(background))
                if signature in backgrounds_seen:
                    continue
                candidates = (*focal, *others[:4])
                modes = {}
                for i, candidate in enumerate(candidates):
                    options = ("relative", "unspecified") if candidate in _RELATIVE else (
                        "held", "press", "unspecified")
                    modes[candidate] = options[(context + focal_index + i) % len(options)]
                specs = [_relational_spec(focal, background, modes, state, context % 2 == 0,
                                          others[4]) for state in PAIR_RELATIONS]
                targets = [annotate_factorized_spec(spec) for spec in specs]
                identities = [factorized_mapping_identity(labels, pairs)
                              for labels, pairs, _ in targets]
                if len(set(identities)) == 3 and not set(identities) & (used | excluded):
                    break
            else:
                raise ValueError("unable to allocate relational semantic context")
            backgrounds_seen.add(signature)
            used.update(identities)
            base = f"v14-r4-rel-{focal_index:02d}-{context}"
            reference = {(p["left"], p["right"]): p["relation"] for p in targets[0][1]}
            for state, spec, target, identity in zip(PAIR_RELATIONS, specs, targets, identities,
                                                     strict=True):
                labels, pairs, legacy = target
                changed = [{"left": p["left"], "right": p["right"],
                            "from": reference[p["left"], p["right"]], "to": p["relation"]}
                           for p in pairs if p["relation"] != reference[p["left"], p["right"]]]
                group = f"{base}-{state}-{identity[:12]}"
                for language in ("ru", "en"):
                    for variant, family in enumerate(RELATIONAL_FAMILIES):
                        panel = {"version": RELATIONAL_PANEL_VERSION, "focal_pair": list(focal),
                                 "context_index": context, "state": state,
                                 "background_pairs": [list(pair) for pair in background],
                                 "family": family, "bank_seed": seed,
                                 "changed_pairs_from_same": changed}
                        render_seed = seed + focal_index * 31 + context * 7 + variant
                        text, mentions, bindings = render_relational_panel(
                            spec, panel, language=language, seed=render_seed)
                        rows.append({
                            "id": f"{group}-{language}-{variant}", "group_id": group,
                            "mapping_id": identity, "semantic_mapping_id": identity,
                            "split": "train", "development_role": None,
                            "scenario_kind": "relational", "language": language, "text": text,
                            "heterogeneous_alias_kind": None,
                            "template_family": f"v14-relational-{family}",
                            "factorized_labels": labels, "pair_labels": pairs, "labels": legacy,
                            "expected_mentions": mentions, "binding_spans": bindings,
                            "specification": asdict(spec), "minimal_pair_group": base,
                            "contrast_cluster_id": base, "independent_group_id": base,
                            "contrast_member": PAIR_RELATIONS.index(state),
                            "contrast_type": "relations",
                            "rendering": _relational_rendering(spec, family),
                            "render_seed": render_seed, "grammar": variant, "variant": variant,
                            "relational_panel": panel,
                        })
    return rows


def generate_factorized_rows(*, seed=DEFAULT_SEED, split_groups=None, excluded_mapping_ids=()):
    """Preserve R3 records outside 288 replaced nonmandatory TRAIN clusters."""
    original = _generate_r3_rows(seed=seed, split_groups=split_groups,
                                 excluded_mapping_ids=excluded_mapping_ids)
    representatives = {r["group_id"]: r for r in original if r["split"] == "train"}
    mandatory = len(_split_schedule("train", True))
    eligible = []
    for row in representatives.values():
        if row["contrast_member"] != 0:
            continue
        index = int(row["contrast_cluster_id"].rsplit("-", 1)[1])
        if (index >= mandatory and not row["heterogeneous_alias_kind"]
                and row["scenario_kind"] not in {
                    "aliases", "pair_unspecified", "wasd", "horizontal"}
                and sum(v["applicability"] == "supported" for v in row["factorized_labels"]) >= 6):
            eligible.append(row["contrast_cluster_id"])
    random.Random(f"{seed}:relational-replacements").shuffle(eligible)
    if len(eligible) < RELATIONAL_GROUPS // 2:
        raise ValueError("TRAIN budget has insufficient nonmandatory six-command clusters")
    replaced = set(eligible[:RELATIONAL_GROUPS // 2])
    retained = [r for r in original if r["contrast_cluster_id"] not in replaced]
    used = {r["semantic_mapping_id"] for r in retained}
    panels = _relational_triplets(seed, used, set(excluded_mapping_ids))
    return ([r for r in retained if r["split"] == "train"] + panels
            + [r for r in retained if r["split"] != "train"])

def _specification_from_row(row):
    source = row["specification"]
    return FactorizedSpec(
        tuple(Operation(tuple(op["candidates"]), op["mode"], op.get("condition"),
                        tuple(tuple(item) for item in op.get("mode_overrides", ())))
              for op in source["operations"]),
        tuple(source["unsupported"]), tuple(source["unspecified"]), source["complete"],
        source["pair_knowledge"], source["kind"], tuple(source["contradiction"]),
        tuple(source["citation"]))


def _validate_relational_cluster(members):
    if len(members) != 3:
        raise ValueError("relational cluster needs all three semantic states")
    by_state = {row["relational_panel"]["state"]: row for row in members.values()}
    if set(by_state) != set(PAIR_RELATIONS):
        raise ValueError("relational states missing or duplicated")
    reference = by_state["same"]
    panel = reference["relational_panel"]
    focal, background = panel["focal_pair"], panel["background_pairs"]
    candidates = focal + [c for pair in background for c in pair]
    if (len(focal) != 2 or len(background) != 2 or any(len(pair) != 2 for pair in background)
            or len(candidates) != 6 or len(set(candidates)) != 6):
        raise ValueError("relational panel needs six distinct physical commands")
    if tuple(focal) not in relational_focal_pairs(panel["bank_seed"]):
        raise ValueError("focal pair is not in the prospective physical bank")
    modes = {v["candidate"]: v["mode"] for v in reference["factorized_labels"]
             if v["applicability"] == "supported"}
    if set(modes) != set(candidates):
        raise ValueError("relational panel changed physical support")
    base_pairs = {(p["left"], p["right"]): p["relation"] for p in reference["pair_labels"]}
    for state, row in by_state.items():
        current = row["relational_panel"]
        if (row["split"] != "train" or row["scenario_kind"] != "relational"
                or any(current[key] != panel[key] for key in (
                    "version", "focal_pair", "background_pairs", "context_index", "bank_seed"))):
            raise ValueError("relational context crosses boundaries")
        expected = _relational_spec(focal, background, modes, state,
                                    reference["specification"]["complete"],
                                    reference["specification"]["unsupported"][0])
        if asdict(expected) != asdict(_specification_from_row(row)):
            raise ValueError("relational contrast changed mode/status or functional partition")
        labels, pairs, legacy = annotate_factorized_spec(expected)
        if (labels != row["factorized_labels"] or pairs != row["pair_labels"]
                or legacy != row["labels"]):
            raise ValueError("relational targets disagree with formal partition")
        changed = [{"left": p["left"], "right": p["right"],
                    "from": base_pairs[p["left"], p["right"]], "to": p["relation"]}
                   for p in pairs if p["relation"] != base_pairs[p["left"], p["right"]]]
        if current["changed_pairs_from_same"] != changed:
            raise ValueError("relational changed edges are not exact")
        focal_gold = next(p["relation"] for p in pairs if {p["left"], p["right"]} == set(focal))
        if focal_gold != state or len(pairs) != 15:
            raise ValueError("focal relation is inconsistent")
    return tuple(focal), panel["context_index"], modes, tuple(tuple(p) for p in background)


def validate_factorized_coverage(rows, *, expected_groups=None, require_full=True):
    """Fail closed on missing combinations, split leakage or broken witnesses."""
    groups, owners, grammar_owners, coverage = {}, {}, {}, defaultdict(Counter)
    clusters, contrast_counts = defaultdict(dict), Counter()
    relational_contexts = defaultdict(dict)
    seen_ids = set()
    for row in rows:
        if row["id"] in seen_ids:
            raise ValueError("duplicate factorized row id")
        seen_ids.add(row["id"])
        split = row["split"]
        if split not in SPLIT_GROUPS:
            raise ValueError("invalid factorized split")
        group = row["group_id"]
        mapping = factorized_mapping_identity(row["factorized_labels"], row["pair_labels"])
        if row["semantic_mapping_id"] != mapping or row["mapping_id"] != mapping:
            raise ValueError("factorized mapping identity mismatch")
        if mapping in owners and owners[mapping] != (split, group):
            raise ValueError("semantic mapping overlap across groups or splits")
        owners[mapping] = (split, group)
        role = row.get("development_role")
        if split == "dev" and role not in {"temperature", "selection"}:
            raise ValueError("DEV role must be allocated before translation")
        owner = f"dev_{role}" if split == "dev" else split
        family = row["template_family"]
        if family in grammar_owners and grammar_owners[family] != owner:
            raise ValueError("template family reused across splits or DEV roles")
        grammar_owners[family] = owner
        prior = groups.setdefault(group, [])
        if prior and (prior[0]["split"] != split or prior[0].get("development_role") != role
                      or prior[0]["semantic_mapping_id"] != mapping):
            raise ValueError("semantic group or minimal pair crosses a partition")
        prior.append(row)
        for mention in row["expected_mentions"]:
            if (mention["candidate"] not in CONTROL_CANDIDATES
                    or not 0 <= mention["start"] < mention["end"] <= len(row["text"])):
                raise ValueError("invalid independent grounding witness")
        spec = _specification_from_row(row)
        panel = row.get("relational_panel")
        if panel:
            if (panel.get("version") != RELATIONAL_PANEL_VERSION or split != "train"
                    or panel["family"] != RELATIONAL_FAMILIES[row["variant"]]):
                raise ValueError("invalid TRAIN relational family ownership")
            if prior[:-1] and any(
                    {key: value for key, value in old["relational_panel"].items()
                     if key != "family"}
                    != {key: value for key, value in panel.items() if key != "family"}
                    for old in prior[:-1]):
                raise ValueError("relational group views disagree on counterfactual context")
            exact_labels, exact_pairs, exact_legacy = annotate_factorized_spec(spec)
            if (exact_labels != row["factorized_labels"] or exact_pairs != row["pair_labels"]
                    or exact_legacy != row["labels"]):
                raise ValueError("relational view disagrees with actual target partition")
            metadata = _relational_rendering(spec, panel["family"])
            expected_text, expected_mentions, expected_bindings = render_relational_panel(
                spec, panel, language=row["language"], seed=row["render_seed"])
            expected_family = f"v14-relational-{panel['family']}"
        else:
            metadata = rendering_metadata(spec, grammar=row["grammar"], variant=row["variant"],
                                          seed=row["render_seed"])
            expected_text, expected_mentions, expected_bindings = render_factorized_spec(
                spec, language=row["language"], grammar=row["grammar"], variant=row["variant"],
                seed=row["render_seed"], include_binding_spans=True)
            expected_family = f"v14-binding-{row['grammar']:02d}"
        if (row["rendering"] != metadata or row["text"] != expected_text
                or row["expected_mentions"] != expected_mentions
                or row.get("binding_spans") != expected_bindings):
            raise ValueError("rendering does not match independent specification witnesses")
        if row["template_family"] != expected_family:
            raise ValueError("template family disagrees with actual binding grammar")
        for name in ("compact_only", "natural_no_action_ids", "implicit_mode_absence",
                     "heterogeneous_alias_modes", "train_timing_context"):
            if metadata[name]:
                coverage[split][("rendering", name, row["scenario_kind"], row["language"])] += 1
        for binding in expected_bindings:
            coverage[split][("binding_duration_cue", binding["mode"],
                             binding["duration_cue"])] += 1
            coverage[split][("binding_function_bridge", binding["mode"],
                             binding["function_between_candidate_and_mode"])] += 1
            if metadata["condition_present"] and metadata["train_timing_context"]:
                coverage[split][("timing_context_condition", True)] += 1
        if any(op.condition for op in spec.operations):
            coverage[split][("condition_view", metadata["condition_present"], row["language"])] += 1
        if len(prior) == 1:
            expected_labels, expected_pairs, expected_legacy = annotate_factorized_spec(spec)
            if (expected_labels != row["factorized_labels"] or expected_pairs != row["pair_labels"]
                    or expected_legacy != row["labels"]):
                raise ValueError("formal specification disagrees with factorized targets")
            cluster = row.get("contrast_cluster_id")
            if cluster:
                if split not in {"train", "dev"}:
                    raise ValueError("CAL/SEALED must retain independent semantic groups")
                clusters[cluster][group] = row
            if row["independent_group_id"] != (cluster or group):
                raise ValueError("invalid independent statistical unit")
            coverage[split][("kind", row["scenario_kind"])] += 1
            if row.get("heterogeneous_alias_kind"):
                coverage[split][("heterogeneous_kind", row["heterogeneous_alias_kind"])] += 1
            for operation in spec.operations:
                modes = {operation.mode_for(c) for c in operation.candidates}
                if len(modes) > 1:
                    coverage[split][("heterogeneous_alias_modes", *sorted(modes))] += 1
            coverage[split][("complete", row["specification"]["complete"])] += 1
            for label in row["factorized_labels"]:
                coverage[split][("status", label["candidate"], label["applicability"])] += 1
                if label["applicability"] == "supported":
                    coverage[split][("mode", label["candidate"], label["mode"])] += 1
            for pair in row["pair_labels"]:
                coverage[split][("pair", pair["relation"])] += 1
    for group_rows in groups.values():
        if Counter(v["language"] for v in group_rows) != {"ru": 3, "en": 3}:
            raise ValueError("each group needs exactly three RU and three EN views")
        if any(op["condition"] for op in group_rows[0]["specification"]["operations"]):
            for language in ("ru", "en"):
                states = {v["rendering"]["condition_present"] for v in group_rows
                          if v["language"] == language}
                if states != {False, True}:
                    raise ValueError(
                        "conditional group needs equivalent views with and without condition")
    for members in clusters.values():
        representative = next(iter(members.values()))
        if representative.get("relational_panel"):
            focal, context, modes, background = _validate_relational_cluster(members)
            if context in relational_contexts[focal]:
                raise ValueError("duplicate relational base context")
            relational_contexts[focal][context] = (modes, background)
            contrast_counts["train", "relations"] += 1
            continue
        if len(members) != 2:
            raise ValueError("a target-changing minimal pair needs two semantic groups")
        left, right = members.values()
        if (left["split"] != right["split"]
                or left.get("development_role") != right.get("development_role")):
            raise ValueError("contrast cluster crosses a partition")
        a, b = left["specification"], right["specification"]
        contrast_type = left.get("contrast_type")
        if right.get("contrast_type") != contrast_type:
            raise ValueError("inconsistent minimal-pair contrast type")
        if contrast_type == "mode":
            if ({key: value for key, value in a.items() if key != "operations"}
                    != {key: value for key, value in b.items() if key != "operations"}
                    or a["operations"][1:] != b["operations"][1:]
                    or a["operations"][0]["candidates"] != b["operations"][0]["candidates"]
                    or {key: value for key, value in a["operations"][0].items()
                        if key not in {"mode", "mode_overrides"}}
                    != {key: value for key, value in b["operations"][0].items()
                        if key not in {"mode", "mode_overrides"}}
                    or left["factorized_labels"] == right["factorized_labels"]
                    or left["pair_labels"] != right["pair_labels"]):
                raise ValueError("contrast cluster is not a genuine timing minimal pair")
        elif contrast_type == "completeness":
            if ({key: value for key, value in a.items() if key != "complete"}
                    != {key: value for key, value in b.items() if key != "complete"}
                    or a["complete"] == b["complete"]
                    or left["factorized_labels"] == right["factorized_labels"]):
                raise ValueError("contrast cluster is not a genuine completeness minimal pair")
        else:
            raise ValueError("unknown minimal-pair contrast type")
        owner = f"dev_{left['development_role']}" if left["split"] == "dev" else left["split"]
        contrast_counts[owner, contrast_type] += 1
    actual = Counter(v[0]["split"] for v in groups.values())
    if expected_groups is not None and actual != expected_groups:
        raise ValueError("factorized split group quota mismatch")
    if require_full:
        for split in SPLIT_GROUPS:
            required = [("kind", kind) for kind in SCENARIO_KINDS]
            required += [("complete", value) for value in (False, True)]
            required += [("pair", relation) for relation in PAIR_RELATIONS]
            required += [("heterogeneous_alias_modes", "held", "press"),
                         ("heterogeneous_alias_modes", "press", "relative")]
            if not any(key[0] == "heterogeneous_alias_modes" and "unspecified" in key
                       for key in coverage[split]):
                required.append(("heterogeneous_alias_modes", "known", "unspecified"))
            for kind in HETEROGENEOUS_KINDS:
                if (coverage[split][("heterogeneous_kind", kind)]
                        != HETEROGENEOUS_GROUP_QUOTAS[split]):
                    raise ValueError(f"heterogeneous alias quota mismatch in {split}: {kind}")
            for language in ("ru", "en"):
                required += [("condition_view", present, language) for present in (False, True)]
                required += [("rendering", "compact_only", kind, language)
                             for kind in ("wasd", "horizontal")]
                for name in ("natural_no_action_ids", "implicit_mode_absence"):
                    if not any(key[0:2] == ("rendering", name) and key[-1] == language
                               for key in coverage[split]):
                        required.append(("rendering", name, "any", language))
            for candidate in CONTROL_CANDIDATES:
                required.extend(("status", candidate, status) for status in APPLICABILITY)
                modes = ("relative", "unspecified") if candidate in _RELATIVE else (
                    "held", "press", "unspecified")
                required.extend(("mode", candidate, mode) for mode in modes)
            missing = [key for key in required if not coverage[split][key]]
            if missing:
                raise ValueError(f"missing required coverage in {split}: {missing[:8]}")
        if len(relational_contexts) != RELATIONAL_FOCAL_PAIRS:
            raise ValueError("missing required relational focal pair bank")
        seeds = {row["relational_panel"]["bank_seed"] for row in rows
                 if row.get("relational_panel")}
        if (len(seeds) != 1
                or set(relational_contexts) != set(relational_focal_pairs(next(iter(seeds))))):
            raise ValueError("relational physical bank differs from prospective definition")
        for focal, contexts in relational_contexts.items():
            if set(contexts) != set(range(RELATIONAL_CONTEXTS)):
                raise ValueError("relational focal pair lacks four independent contexts")
            if len({values[1] for values in contexts.values()}) != RELATIONAL_CONTEXTS:
                raise ValueError("relational contexts differ only cosmetically")
            for candidate in focal:
                if len({values[0][candidate] for values in contexts.values()}) < 2:
                    raise ValueError("physical mode is confounded with relational panel")
        for mode in MODES:
            for present in (False, True):
                for feature in ("binding_duration_cue", "binding_function_bridge"):
                    if not coverage["train"][feature, mode, present]:
                        raise ValueError(
                            f"missing TRAIN timing context coverage: {feature, mode, present}")
        if not coverage["train"]["timing_context_condition", True]:
            raise ValueError("missing TRAIN timing context with conditional wording")
        for owner in ("train", "dev_temperature", "dev_selection"):
            for contrast_type in ("mode", "completeness"):
                if not contrast_counts[owner, contrast_type]:
                    raise ValueError(f"missing required {contrast_type} contrast in {owner}")
    return {"semantic_groups": len(groups), "groups_by_split": dict(actual),
            "rows": len(rows), "template_families": len(grammar_owners),
            "required_combinations_present": require_full,
            "relational_panels": {
                "independent_contexts": sum(map(len, relational_contexts.values())),
                "focal_pairs": len(relational_contexts),
                "semantic_groups": 3 * sum(map(len, relational_contexts.values()))},
            "contrast_clusters_by_owner_type": {
                "|".join(key): value for key, value in contrast_counts.items()},
            "dev_groups_by_role": dict(Counter(v[0]["development_role"] for v in groups.values()
                                               if v[0]["split"] == "dev")),
            "coverage": {split: {"|".join(map(str, key)): value for key, value in values.items()}
                         for split, values in coverage.items()}}


def _source_pins():
    paths = [Path(__file__), Path(__file__).with_name("control_text_grounding.py")]
    return {str(path.name): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def prepare_factorized_dataset(output, *, seed=DEFAULT_SEED, split_groups=None,
                               excluded_mapping_ids=(), exclusion_manifest=None):
    """Write a new immutable corpus; existing preparations are never overwritten."""
    counts = dict(SPLIT_GROUPS if split_groups is None else split_groups)
    records = generate_factorized_rows(seed=seed, split_groups=counts,
                                      excluded_mapping_ids=excluded_mapping_ids)
    coverage = validate_factorized_coverage(records, expected_groups=counts)
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
                       for row in records)
    excluded = sorted(set(excluded_mapping_ids))
    manifest = {
        "version": CONTROL_TEXT_DATA_VERSION, "protocol_version": PROTOCOL_VERSION,
        "records": len(records), "generation_revision": GENERATION_REVISION,
        "seed": seed, "status": "prepared", "model_loaded": False, "updates": 0,
        "readiness": False, "ppo_eligible": False,
        "records_sha256": hashlib.sha256(payload).hexdigest(), "records_bytes": len(payload),
        "source_sha256": _source_pins(), "coverage": coverage,
        "groups_by_split": counts, "views_per_group": 6,
        "views_by_language": {"ru": 3, "en": 3},
        "development_groups": coverage["dev_groups_by_role"],
        "contrast_clusters": "TRAIN R3 pairs plus R4 triplets; DEV pairs; CAL/SEALED independent",
        "relational_training": {"version": RELATIONAL_PANEL_VERSION, "groups": RELATIONAL_GROUPS,
                                "focal_pairs": RELATIONAL_FOCAL_PAIRS,
                                "contexts_per_pair": RELATIONAL_CONTEXTS,
                                "heldout_records_preserved": True,
                                "all_630_physical_pairs_required": False},
        "heterogeneous_alias_groups_per_kind": HETEROGENEOUS_GROUP_QUOTAS,
        "mode_semantics": "per physical candidate; functional aliases may have different timing",
        "timing_context_augmentation": "TRAIN natural02 deterministic quarter; labels unchanged",
        "binding_spans": "renderer emitted; never consumed by inference",
        "template_families_disjoint": True, "internal_grammar_families_disjoint": True,
        "semantic_mappings_disjoint": True, "split_before_translation": True,
        "runtime_text_copied": False, "conditions": "offline metadata; globally supported",
        "mention_expectations": "renderer-emitted exact character spans; no matcher used",
        "excluded_mapping_count": len(excluded),
        "excluded_mapping_ids_sha256": hashlib.sha256(json.dumps(excluded).encode()).hexdigest(),
        "exclusion_manifest": exclusion_manifest,
        "legacy_labels": "compatibility only; factorized_labels and pair_labels are authoritative",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "records.jsonl").write_bytes(payload)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def read_factorized_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("version") != CONTROL_TEXT_DATA_VERSION
            or manifest.get("protocol_version") != PROTOCOL_VERSION
            or manifest.get("generation_revision") != GENERATION_REVISION):
        raise ValueError("unsupported factorized dataset protocol")
    payload = (directory / "records.jsonl").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get("records_sha256"):
        raise ValueError("factorized dataset records hash mismatch")
    rows = [json.loads(line) for line in payload.splitlines()]
    validate_factorized_coverage(rows, expected_groups=manifest["groups_by_split"])
    if manifest.get("source_sha256") != _source_pins():
        raise ValueError("factorized corpus source hash mismatch; prepare a new dataset")
    return manifest, rows
