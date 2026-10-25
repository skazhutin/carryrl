"""V8 structural supervision with explicit historical exposure partitions.

Specifications own labels and semantic identities. Rendering never reads games,
public runtime strings, teacher predictions, or prior example texts. Historical
mapping IDs may train only when the authenticated registry permits development;
old sealed mappings remain blocked everywhere. New evaluation mappings are fresh.
"""

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from carryrl.rewards.control_text_data import (
    CONTROL_CANDIDATES,
    SPLIT_WEIGHTS,
    _semantic_split,
    semantic_mapping_identity,
)
from carryrl.rewards.control_text_data_v4 import ControlSpec, OperationSpec
from carryrl.rewards.control_text_data_v5 import REMAINDER_FORMS
from carryrl.rewards.control_text_data_v6 import (
    _AXES,
    _GOALS,
    _HORIZONTAL,
    _NAVIGATION,
    _validated_exclusions,
    annotate_spec,
)

CONTROL_TEXT_DATA_VERSION = "carryrl.control_text_data.v8"
PARTITION_REGISTRY_VERSION = "carryrl.control_text_partition_registry.v8"
DEFAULT_SEED = 20261024
# These IDs identify actual binding compositions, rather than outer wrappers.
INTERNAL_FAMILIES = {
    "train": tuple(range(12)),
    "dev": tuple(range(12, 16)),
    "calibration": tuple(range(16, 20)),
    "sealed_test": tuple(range(20, 24)),
}
SCENARIO_KINDS = (
    "navigation", "horizontal", "singleton_keys", "normal_aliases", "mouse_axes",
    "click_release", "adjacent_modes", "partial_unknown", "conditional", "mixed", "unknown",
    "cross_modal_alias",
)
# Quotas are applied to accepted groups, after all identity/reservation filters.
_SCHEDULE = (
    "navigation", "horizontal", "singleton_keys", "normal_aliases", "mouse_axes",
    "click_release", "adjacent_modes", "singleton_keys", "partial_unknown", "conditional",
    "navigation", "horizontal", "normal_aliases", "adjacent_modes", "mixed", "mouse_axes",
    "singleton_keys", "click_release", "partial_unknown", "conditional", "mixed", "unknown",
    "normal_aliases", "singleton_keys", "adjacent_modes", "mouse_axes", "mixed",
    "cross_modal_alias", "partial_unknown", "conditional", "singleton_keys", "normal_aliases",
    "adjacent_modes", "mixed", "unknown", "mouse_axes", "click_release", "mixed",
    "partial_unknown", "conditional",
)
_KEYS = tuple(value for value in CONTROL_CANDIDATES if value.startswith("key:"))
_BUTTONS = tuple(value for value in CONTROL_CANDIDATES if value.startswith("mouse:"))
_RELATIVE = tuple(
    value for value in CONTROL_CANDIDATES if value.startswith(("dx:", "dy:", "wheel:"))
)


@dataclass(frozen=True)
class PartitionRegistry:
    hard_old_sealed_ids: frozenset = frozenset()
    developmental_ids: frozenset = frozenset()
    previous_mapping_ids: frozenset = frozenset()

    def allows(self, identity, split):
        if identity in self.hard_old_sealed_ids:
            return False
        if split == "train":
            return identity not in self.previous_mapping_ids or identity in self.developmental_ids
        return identity not in self.previous_mapping_ids


def read_partition_registry(path, excluded_semantic_ids=()):
    """Authenticate labels-only IDs; strict historical seals take precedence."""
    if path is None:
        raise ValueError("v8 preparation requires an authenticated partition registry")
    payload = Path(path).read_bytes()
    document = json.loads(payload)
    if not isinstance(document, dict) or document.get("version") != PARTITION_REGISTRY_VERSION:
        raise ValueError("invalid v8 partition registry version")
    values = {}
    for field in ("hard_old_sealed_ids", "developmental_ids", "previous_mapping_ids"):
        if not isinstance(document.get(field), list):
            raise ValueError(f"partition registry requires a {field} list")
        values[field] = _validated_exclusions(document[field])
        if len(values[field]) != len(document[field]):
            raise ValueError(f"partition registry contains duplicate {field}")
    if not (values["hard_old_sealed_ids"] | values["developmental_ids"]) <= values[
        "previous_mapping_ids"
    ]:
        raise ValueError("previous_mapping_ids must include every historical partition ID")
    explicit = _validated_exclusions(excluded_semantic_ids)
    registry = PartitionRegistry(
        values["hard_old_sealed_ids"] | explicit,
        values["developmental_ids"],
        values["previous_mapping_ids"] | explicit,
    )
    return registry, {
        "partition_registry_version": PARTITION_REGISTRY_VERSION,
        "partition_registry": str(Path(path).resolve()),
        "partition_registry_sha256": hashlib.sha256(payload).hexdigest(),
        "reservation_registry": str(Path(path).resolve()),
        "reservation_registry_sha256": hashlib.sha256(payload).hexdigest(),
        "hard_old_sealed_mapping_count": len(registry.hard_old_sealed_ids),
        "developmental_mapping_count": len(registry.developmental_ids),
        "previous_mapping_count": len(registry.previous_mapping_ids),
        "hard_seal_developmental_overlap_count": len(
            registry.hard_old_sealed_ids & registry.developmental_ids
        ),
        "explicit_excluded_semantic_ids_count": len(explicit),
        "partition_policy": {
            "hard_old_sealed_ids": "excluded from every split; highest priority",
            "developmental_ids": "TRAIN only, unless hard sealed",
            "previous_mapping_ids": "excluded from fresh dev, calibration and sealed_test",
            "public_structural_compatibility": "TRAIN only; no runtime text read by generator",
        },
    }


def _spec(groups, modes, kind, *, remainder="unsupported", uncertain=(), forbidden=()):
    return ControlSpec(
        tuple(OperationSpec(f"{index:03x}", tuple(group), mode)
              for index, (group, mode) in enumerate(zip(groups, modes, strict=True))),
        tuple(uncertain), tuple(forbidden), remainder, kind,
    )


def elementary_specs():
    """Mandatory clean mappings, before hash allocation or random compositions.

    These effective groups are unique, not duplications with decorative labels.
    Each receives four Russian and four English syntactic variants in TRAIN.
    """
    result = []
    for kind, groups in (("navigation", _NAVIGATION), ("horizontal", _HORIZONTAL)):
        for remainder in ("unsupported", "unknown"):
            for mode in ("held", "press"):
                result.append(_spec(groups, [mode] * len(groups), kind, remainder=remainder))
    for candidate in (*_KEYS, *_BUTTONS):
        for mode in ("held", "press"):
            kind = "click_release" if candidate.startswith("mouse:") else "singleton_keys"
            result.append(_spec([(candidate,)], [mode], kind))
    for remainder in ("unsupported", "unknown"):
        result.append(_spec([(value,) for value in _AXES], ["unknown"] * 4,
                            "mouse_axes", remainder=remainder))
    for candidate in _RELATIVE:
        result.append(_spec([(candidate,)], ["unknown"], "mouse_axes"))
    for pair in _NAVIGATION:
        for mode in ("held", "press"):
            result.append(_spec([pair], [mode], "normal_aliases"))
    for remainder in ("unsupported", "unknown"):
        for modes in (("held", "press"), ("press", "held")):
            result.append(_spec([("key:w",), ("key:s",)], modes, "adjacent_modes",
                                remainder=remainder))
    return result



def _resolved_elementary_specs(registry):
    """Keep the four complete canonical cores mandatory; report historical gaps.

    A blocked secondary complete rule may be replaced with an explicitly partial
    reference. Its labels and text both use unknown remainder; the old effective
    mapping is never relabeled, unsealed, or silently reused.
    """
    selected, gaps, seen = [], [], set()
    for spec in elementary_specs():
        identity, _ = _identity(spec)
        if registry.allows(identity, "train"):
            if identity not in seen:
                selected.append(spec)
                seen.add(identity)
            continue
        if spec.kind in {"navigation", "horizontal"} and spec.remainder == "unsupported":
            raise ValueError(f"mandatory clean TRAIN mapping blocked: {identity}")
        alternate = replace(spec, remainder="unknown")
        alternate_id, _ = _identity(alternate)
        usable = (alternate_id != identity and registry.allows(alternate_id, "train"))
        gaps.append({
            "blocked_mapping_id": identity, "scenario_kind": spec.kind,
            "declared_candidates": list(spec.declared_candidates),
            "modes": [operation.mode for operation in spec.operations],
            "blocked_remainder": spec.remainder,
            "alternative_mapping_id": alternate_id if usable else None,
            "alternative_remainder": "unknown" if usable else None,
        })
        if usable and alternate_id not in seen:
            selected.append(alternate)
            seen.add(alternate_id)
    return selected, gaps

def _sample_spec(rng, kind):
    """Mostly ordinary singletons/directional aliases; at most five operations."""
    remainder = "unknown" if rng.random() < 0.05 else "unsupported"
    groups, modes = [], []
    if kind == "unknown":
        return _spec([], [], kind, remainder="unknown",
                     uncertain=rng.sample(CONTROL_CANDIDATES, rng.randint(1, 5)))
    if kind in {"navigation", "horizontal"}:
        groups = list(_NAVIGATION if kind == "navigation" else _HORIZONTAL)
        shared = rng.choice(("held", "press"))
        modes = [shared if rng.random() < 0.6 else rng.choice(("held", "press"))
                 for _ in groups]
    elif kind == "normal_aliases":
        groups = rng.sample(_NAVIGATION, rng.randint(1, 2))
    elif kind == "mouse_axes":
        selected = _AXES if rng.random() < 0.5 else rng.sample(_RELATIVE, rng.randint(1, 4))
        groups = [(value,) for value in selected]
    elif kind == "click_release":
        groups = [(rng.choice(_BUTTONS),)]
        modes = ["press"]
    elif kind == "cross_modal_alias":
        groups = [(rng.choice(_KEYS), rng.choice(_BUTTONS))]
    elif kind == "adjacent_modes":
        groups = [(value,) for value in rng.sample(_KEYS + _BUTTONS, 2)]
        modes = rng.sample(("held", "press"), 2)
    # Every random group has some composition; clean elementary cases above are
    # reserved explicitly and cannot disappear through uniqueness/hash filtering.
    used = {value for group in groups for value in group}
    pool = [value for value in CONTROL_CANDIDATES if value not in used]
    rng.shuffle(pool)
    if kind == "singleton_keys":
        pool = [value for value in pool if value.startswith("key:")]
    maximum = 4 if kind == "conditional" else 5
    target = rng.randint(max(2, len(groups) + 1), maximum) if len(groups) < maximum else maximum
    while len(groups) < target:
        groups.append((pool.pop(),))
    while len(modes) < len(groups):
        modes.append(rng.choice(("held", "press")))
    for index, group in enumerate(groups):
        if all(value in _RELATIVE for value in group):
            modes[index] = "unknown"
    used = {value for group in groups for value in group}
    remaining = [value for value in CONTROL_CANDIDATES if value not in used]
    uncertain, forbidden = (), ()
    if kind == "partial_unknown" and rng.random() < 0.4:
        discrete = [index for index, group in enumerate(groups)
                    if all(value not in _RELATIVE for value in group)]
        if discrete:
            modes[rng.choice(discrete)] = "unknown"
    if kind in {"partial_unknown", "conditional"}:
        uncertain = (rng.choice(remaining),)
        if kind == "conditional":
            forbidden = (rng.choice([value for value in remaining if value not in uncertain]),)
        remainder = "unknown"
    elif kind == "mouse_axes" and set(_AXES) <= used and rng.random() < 0.65:
        # Four axes plus one extra have a small finite complete mapping space.
        # An explicit partial reference and one negative add real independent
        # semantic coverage, keeping five operations plus one status clause.
        remainder = "unknown"
        forbidden = (rng.choice(remaining),)
    elif rng.random() < 0.1:
        forbidden = (rng.choice(remaining),)
    order = rng.sample(range(len(groups)), len(groups))
    return _spec([groups[index] for index in order], [modes[index] for index in order], kind,
                 remainder=remainder, uncertain=uncertain, forbidden=forbidden)


def _name(candidate, language, *, accusative=False):
    """Case-safe names: templates choose nominative labels or direct objects."""
    kind, value = candidate.split(":")
    if language == "en":
        if kind == "key":
            return f"{value.title()} arrow" if value in {"up", "down", "left", "right"} else (
                "Space key" if value == "space" else f"{value.upper()} key"
            )
        if kind == "mouse":
            return f"{value} mouse button"
    elif kind == "key":
        if value in {"up", "down", "left", "right"}:
            direction = {"up": "вверх", "down": "вниз", "left": "влево", "right": "вправо"}[value]
            return f"{'стрелку' if accusative else 'стрелка'} {direction}"
        if value == "space":
            return "Пробел"
        return f"{'клавишу' if accusative else 'клавиша'} {value.upper()}"
    elif kind == "mouse":
        names = {"left": ("левая", "левую"), "right": ("правая", "правую"),
                 "middle": ("средняя", "среднюю")}
        return f"{names[value][int(accusative)]} {'кнопку' if accusative else 'кнопка'} мыши"
    forms = {
        "dx:-": ("движение мыши влево", "mouse movement left"),
        "dx:+": ("движение мыши вправо", "mouse movement right"),
        "dy:-": ("движение мыши вверх", "mouse movement up"),
        "dy:+": ("движение мыши вниз", "mouse movement down"),
        "wheel:-": ("прокрутка колеса вниз", "scrolling the wheel down"),
        "wheel:+": ("прокрутка колеса вверх", "scrolling the wheel up"),
    }
    return forms[candidate][0 if language == "ru" else 1]


def _binding(operation, language, family):
    """Twenty-four actual local grammars, twelve of them available to TRAIN."""
    ru = language == "ru"
    joiner = " или " if ru else " or "
    names = joiner.join(_name(value, language) for value in operation.candidates)
    objects = joiner.join(_name(value, language, accusative=True) for value in operation.candidates)
    relative = all(value in _RELATIVE for value in operation.candidates)
    mixed = any(value in _RELATIVE for value in operation.candidates) and not relative
    if mixed:
        # This stratum is not currently sampled, but rendering remains explicit.
        parts = [_binding(OperationSpec(operation.identifier, (value,), operation.mode),
                          language, family) for value in operation.candidates]
        return ("Равнозначные варианты: " if ru else "Equivalent choices: ") + "; ".join(parts)
    return _binding_clause(names=names, objects=objects, mode=operation.mode,
                           language=language, family=family, relative=relative,
                           equivalent=len(operation.candidates) > 1)


def _binding_clause(*, names, objects, mode, language, family, relative=False, equivalent=False):
    """Render actual names directly, before sentence capitalization.

    Compact canonical sets use the same binding grammar without impersonating
    a physical input and replacing its spelling after capitalization.
    """
    ru = language == "ru"
    if relative:
        action = f"выполни ввод «{names}»" if ru else f"use {names}"
        timing = "движение доступно" if ru else "movement is available"
    elif mode == "held":
        action = f"удерживай {objects}" if ru else f"hold {objects}"
        timing = "удержание повторяет действие" if ru else "holding repeats the action"
    elif mode == "press":
        action = (f"нажми {objects} один раз и отпусти" if ru else
                  f"press {objects} once and release")
        timing = "отпускай между нажатиями" if ru else "release between presses"
    else:
        action = f"ввод «{names}»: режим неизвестен" if ru else f"{names}: timing is unknown"
        timing = "доступность известна, длительность не указана" if ru else (
            "availability is known, duration is unspecified"
        )
    # Name-first and command-first bindings, including explicit anaphoric timing.
    forms = (
        (
            f"{action}; {timing}",
            f"{names}: {action}; {timing}",
            f"Команда — {names}. Для неё: {action}; {timing}",
            f"{timing}; соответствующий ввод: {names}; {action}",
            f"Действие через ввод «{names}»: {action}; {timing}",
            f"Используй ввод «{names}» так: {action}; {timing}",
            f"{names}. Этот ввод работает так: {action}; {timing}",
            f"{action}. Именно этот ввод, {names}, выполняет действие; {timing}",
            f"Способ ввода — {names}; правило: {action}; {timing}",
            f"Чтобы выполнить действие: {action}; {timing}; ввод — {names}",
            f"{names} запускает действие; требуется: {action}; {timing}",
            f"Здесь доступен ввод «{names}»; {timing}; {action}",
            f"Для действия сначала выбери ввод «{names}», затем {action}; {timing}",
            f"{names}: правило действия — {timing}; выполняй его так: {action}",
            f"{timing}. Это правило относится к вводу «{names}»: {action}",
            f"Действие: {action}; его ввод — {names}. Помни: {timing}",
            f"У ввода «{names}» такое правило: {timing}. Выполнение: {action}",
            f"{action}; ввод «{names}» подчиняется правилу: {timing}",
            f"С вводом «{names}» действие задаётся так: {action}. Условие: {timing}",
            f"Правило: {timing}; ввод «{names}» выполняет действие так: {action}",
            f"Ввод «{names}» и его правило: {action}. Для повторения: {timing}",
            f"Действуй так: {action}. Это ввод «{names}»; {timing}",
            f"{names} связан с таким правилом: {timing}; действие выполняй так: {action}",
            f"Сначала правило — {timing}; затем действие: {action}; выбранный ввод — {names}",
        ) if ru else (
            f"{action}; {timing}",
            f"{names}: {action}; {timing}",
            f"The control is {names}. For this control: {action}; {timing}",
            f"{timing}; the corresponding input is {names}; {action}",
            f"The action through {names}: {action}; {timing}",
            f"Use {names} as follows: {action}; {timing}",
            f"{names}. This input works as follows: {action}; {timing}",
            f"{action}. This input, {names}, performs the action; {timing}",
            f"The input is {names}; its rule: {action}; {timing}",
            f"To perform the action: {action}; {timing}; the input is {names}",
            f"{names} performs the action; it requires: {action}; {timing}",
            f"The available input is {names}; {timing}; {action}",
            f"For the action first select {names}, then {action}; {timing}",
            f"{names}: the action rule is {timing}; execute it as follows: {action}",
            f"{timing}. This rule refers to {names}: {action}",
            f"Action: {action}; its input is {names}. Remember: {timing}",
            f"The rule for {names} is {timing}. Execution: {action}",
            f"{action}; the input {names} follows the rule: {timing}",
            f"With {names}, the action is specified as {action}. Condition: {timing}",
            f"Rule: {timing}; the input {names} performs its action as {action}",
            f"The input {names} and its rule: {action}. To repeat: {timing}",
            f"Act as follows: {action}. This is the input {names}; {timing}",
            f"{names} has this rule: {timing}; perform its action as {action}",
            f"First the rule, {timing}; then the action: {action}; selected input: {names}",
        )
    )
    result = forms[family]
    if equivalent:
        result += "; варианты равнозначны" if ru else "; these alternatives are equivalent"
    return result[0].upper() + result[1:]



def _canonical_names(kind, language):
    if kind == "navigation":
        return "WASD или стрелки" if language == "ru" else "WASD or arrow keys"
    return ("A/D или стрелки влево/вправо" if language == "ru"
            else "A/D or left/right arrows")


def _canonical(spec, language, family):
    """Compact ordinary aliases still describe separate directional actions."""
    ru = language == "ru"
    groups = _NAVIGATION if spec.kind == "navigation" else _HORIZONTAL
    core = [operation for operation in spec.operations if operation.candidates in groups]
    if len(core) != len(groups) or len({operation.mode for operation in core}) != 1:
        return [], ""
    names = _canonical_names(spec.kind, language)
    mode = core[0].mode
    description = ("удержание повторяет движение" if ru else "holding repeats movement") if (
        mode == "held"
    ) else ("нажимай один раз и отпускай между шагами" if ru else (
        "press once and release between steps"
    ))
    # Explicit scope with a compact conventional name, shared timing, and an
    # internal composition selected by split. No runtime sentence is consulted.
    text = _binding_clause(names=names, objects=names, mode=mode,
                           language=language, family=family)
    text += ("; направления имеют обычный смысл; " if ru else (
        "; directions have their usual meanings; "
    )) + description
    return core, text



def _aggregate_axes(language, family):
    """Natural whole-device descriptions of four independent signed inputs."""
    forms = (
        (
            "Сдвиг курсора задаётся движением мыши по обеим осям",
            "Прицел следует движению мыши в каждой из двух координат",
            "Двигай мышь, чтобы менять положение указателя",
            "Для поворота камеры используй движение мыши в любую сторону",
            "Положение прицела меняется, когда двигаешь мышь",
            "Курсор перемещается вслед за движением мыши",
            "Мышь меняет направление взгляда по горизонтали и вертикали",
            "Перемещение указателя выполняется движением мыши",
            "Для наведения доступно движение мыши; обе координаты прицела следуют ему",
            "Движение мыши задаёт смещение камеры по двум осям",
            "Двигай мышь для наведения; прицел можно смещать в любом направлении",
            "Курсор реагирует на движение мыши, независимо по каждой координате",
            "Чтобы сместить прицел в любую сторону, сначала выбери движение мыши",
            "Движение мыши: по обеим координатам указатель следует этому вводу",
            "Обе координаты камеры доступны. Их меняет движение мыши",
            "Наведение выполняется так: двигай мышь; прицел следует этому вводу",
            "У движения мыши есть две оси; обе меняют положение курсора",
            "Двигай мышь; этот ввод перемещает указатель по любой из двух осей",
            "С движением мыши прицел смещается по горизонтали и вертикали",
            "Для камеры заданы две координаты; движение мыши меняет каждую из них",
            "Движение мыши и его назначение: смещать прицел в любую сторону",
            "Наводи указатель так: двигай мышь; доступны все направления",
            "Курсор связан с движением мыши; независимо меняются две координаты",
            "Сначала выбери направление прицеливания; затем двигай мышь в нужную сторону",
        ) if language == "ru" else (
            "Cursor displacement comes from mouse motion along both axes",
            "The aiming point follows mouse movement in each of its two coordinates",
            "Move the mouse to change the pointer position",
            "To turn the camera, use mouse movement in any direction",
            "The aim position changes when you move the mouse",
            "The cursor travels in response to mouse motion",
            "Mouse movement changes the view horizontally and vertically",
            "Pointer placement is performed through mouse movement",
            "Mouse motion is available for aiming; both aim coordinates follow it",
            "Mouse movement sets camera displacement along two axes",
            "Move the mouse for aiming; the aiming point can shift in any direction",
            "The cursor responds to mouse movement independently in each coordinate",
            "To shift the aiming point in any direction, first select mouse movement",
            "Mouse movement: the pointer follows this input in both coordinates",
            "Both camera coordinates are available. Mouse movement changes them",
            "Aim as follows: move the mouse; the aiming point follows this input",
            "Mouse movement has two axes; both change the cursor position",
            "Move the mouse; this input places the pointer along either of two axes",
            "With mouse movement the aiming point shifts horizontally and vertically",
            "The camera has two coordinates; mouse movement changes each of them",
            "Mouse movement and its purpose: shift the aiming point in any direction",
            "Position the pointer as follows: move the mouse; every direction is available",
            "The cursor is linked to mouse movement; its two coordinates change independently",
            "First select the aiming direction; then move the mouse in the desired direction",
        )
    )
    return forms[family]

def render_spec(spec, *, language, variant, split, seed, pair_seed):
    ru, rng = language == "ru", random.Random(seed)
    families = INTERNAL_FAMILIES[split]
    offset = int(pair_seed[:8], 16) if len(pair_seed) >= 8 else 0
    family = families[(offset + variant) % len(families)]
    operations = list(spec.operations)
    rng.shuffle(operations)
    canonical, compact = _canonical(spec, language, family) if spec.kind in {
        "navigation", "horizontal"
    } else ([], "")
    chunks = [compact + "."] if compact else []
    axes = [operation for operation in operations
            if len(operation.candidates) == 1 and operation.candidates[0] in _AXES]
    labels = {label["candidate"]: label for label in annotate_spec(spec)}
    aggregate = len(axes) == 4 and all(
        labels[value]["status"] == "supported" and labels[value]["mode"] == "unknown"
        for value in _AXES
    ) and len({labels[value]["alias_group"] for value in _AXES}) == 4
    if aggregate:
        chunks.append(_aggregate_axes(language, family) + ".")
    else:
        axes = []
    clauses = [_binding(operation, language, family) for operation in operations
               if operation not in canonical and operation not in axes]
    if clauses:
        chunks.append(". ".join(clauses) + ".")
    for candidate in spec.uncertain:
        name = _name(candidate, language)
        if spec.kind == "conditional":
            chunks.append(f"Ввод «{name}» разрешён только при неизвестном условии; "
                           "пока условие не установлено, доступность неизвестна." if ru else (
                               f"Input {name} is permitted only under an unknown condition; "
                               "until the condition is known, availability is unknown."
                           ))
        else:
            chunks.append(f"Ввод «{name}»: доступность неизвестна." if ru else (
                f"Input {name}: availability is unknown."
            ))
    for candidate in spec.forbidden:
        name = _name(candidate, language)
        chunks.append(f"Ввод «{name}» запрещён." if ru else f"Input {name} is unsupported.")
    if rng.random() < 0.3:
        chunks.insert(0, rng.choice(_GOALS[language]))
    chunks.append(REMAINDER_FORMS[spec.remainder][language][(offset + variant) % 8])
    grammar_families = [f"v8-binding-composition-{family:02d}"]
    if aggregate:
        grammar_families.append(f"v8-aggregate-axes-{family:02d}")
    return " ".join(chunks), grammar_families


def _identity(spec):
    labels = annotate_spec(spec)
    return semantic_mapping_identity(labels, spec.declared_candidates), labels


def _unique_groups(records):
    return list({row["group_id"]: row for row in records}.values())


def generate_control_text_records(
    *, seed=DEFAULT_SEED, scenario_groups=4000, partition_registry=None, excluded_semantic_ids=()
):
    if type(scenario_groups) is not int or scenario_groups < 200 or scenario_groups % 100:
        raise ValueError("scenario_groups must be a multiple of 100, at least 200")
    if scenario_groups > 40000:
        raise ValueError("too many unique scenario mappings")
    registry = partition_registry or PartitionRegistry()
    explicit = _validated_exclusions(excluded_semantic_ids)
    registry = PartitionRegistry(registry.hard_old_sealed_ids | explicit,
                                 registry.developmental_ids,
                                 registry.previous_mapping_ids | explicit)
    rng, seen, accepted = random.Random(seed), set(), []
    mandatory, _ = _resolved_elementary_specs(registry)
    for spec in mandatory:
        identity, labels = _identity(spec)
        if not registry.allows(identity, "train"):
            raise ValueError(f"mandatory clean TRAIN mapping blocked: {identity}")
        if identity in seen:
            raise ValueError("elementary specification duplicates an effective mapping")
        seen.add(identity)
        accepted.append(("train", spec, identity, labels, True))
    for split, weight in SPLIT_WEIGHTS:
        target = scenario_groups * weight // 100
        existing = len(mandatory) if split == "train" else 0
        if existing > target:
            raise ValueError("scenario_groups cannot accommodate mandatory clean TRAIN mappings")
        for local in range(target - existing):
            kind = _SCHEDULE[local % len(_SCHEDULE)]
            for _ in range(100000):
                spec = _sample_spec(rng, kind)
                identity, labels = _identity(spec)
                if identity in seen or not registry.allows(identity, split):
                    continue
                # Previously exposed developmental IDs belong to TRAIN even if
                # the hash partition differs. Clean mandatory groups do as well.
                assigned = "train" if identity in registry.developmental_ids else _semantic_split(
                    identity, seed
                )
                if assigned != split:
                    continue
                seen.add(identity)
                accepted.append((split, spec, identity, labels, False))
                break
            else:
                raise ValueError(f"post-filter quota exhausted for {split}/{kind}")
    records, text_labels = [], {}
    for index, (split, spec, identity, labels, elementary) in enumerate(accepted):
        variants = 8 if elementary else 6
        for variant in range(variants):
            language = "ru" if variant < variants // 2 else "en"
            text, grammars = render_spec(spec, language=language, variant=variant, split=split,
                                        seed=rng.getrandbits(64), pair_seed=identity)
            if text_labels.setdefault(text, identity) != identity:
                raise ValueError("same text has conflicting semantic labels")
            records.append({
                "group_id": f"v8-scenario-{index:05d}", "split": split,
                "mapping_id": identity, "semantic_mapping_id": identity,
                "declared_candidates": list(spec.declared_candidates),
                "template_family": grammars[0], "grammar_families": grammars,
                "scenario_kind": spec.kind, "language": language, "text": text,
                "labels": labels, "specification": asdict(spec),
                "elementary": elementary, "variant_index": variant,
                "group_variants": variants,
                "developmental_compatibility": identity in registry.developmental_ids,
            })
    return records


def wording_manifest(records):
    groups = _unique_groups(records)
    distributions = {}
    coverage = {}
    for split, _ in SPLIT_WEIGHTS:
        selected = [row for row in groups if row["split"] == split]
        labels = [label for row in selected for label in row["labels"]]
        distributions[split] = {
            "scenario_groups": dict(Counter(row["scenario_kind"] for row in selected)),
            "supported_count_groups": dict(Counter(str(sum(
                label["status"] == "supported" for label in row["labels"]
            )) for row in selected)),
            "status_candidates": dict(Counter(label["status"] for label in labels)),
            "mode_candidates": dict(Counter(label["mode"] for label in labels)),
            "candidate_status_mode": dict(Counter(
                f"{label['candidate']}|{label['status']}|{label['mode']}" for label in labels
            )),
        }
        coverage[split] = {
            "independent_groups": len(selected),
            "elementary_groups": sum(row["elementary"] for row in selected),
            "developmental_groups": sum(row["developmental_compatibility"] for row in selected),
            "syntax_compositions": sorted({row["template_family"] for row in records
                                           if row["split"] == split}),
            "supported_sign_groups": sum(any(label["status"] == "supported"
                                             for label in row["labels"]) for row in selected),
            "unsupported_sign_groups": sum(any(label["status"] == "unsupported"
                                               for label in row["labels"]) for row in selected),
            "held_groups": sum(any(label["mode"] == "held" for label in row["labels"])
                               for row in selected),
            "press_groups": sum(any(label["mode"] == "press" for label in row["labels"])
                                for row in selected),
            "normal_direction_alias_groups": sum(any(
                tuple(operation["candidates"]) in _NAVIGATION
                for operation in row["specification"]["operations"]
            ) for row in selected),
        }
    eligible = {status: coverage["calibration"][f"{status}_sign_groups"]
                for status in ("supported", "unsupported")}
    if len(groups) >= 4000 and min(eligible.values()) < 300:
        raise ValueError("insufficient independent calibration groups for either sign")
    return {
        "version": CONTROL_TEXT_DATA_VERSION, "wording_version": 8,
        "semantic_generator_version": CONTROL_TEXT_DATA_VERSION,
        "scenario_kinds": list(SCENARIO_KINDS),
        "template_families_disjoint": True, "internal_grammar_families_disjoint": True,
        "outer_template_families_disjoint": False,
        "template_family_definition": "Actual local binding compositions; split-owned IDs.",
        "syntax_primitives_policy": (
            "Timing and naming primitives overlap; exact compositions do not."
        ),
        "template_family_split_ownership": {
            f"v8-binding-composition-{family:02d}": split
            for split, families in INTERNAL_FAMILIES.items() for family in families
        },
        "aggregate_axis_grammar_split_ownership": {
            f"v8-aggregate-axes-{family:02d}": split
            for split, families in INTERNAL_FAMILIES.items() for family in families
        },
        "aggregate_axis_group_counts": {
            split: sum(any(family.startswith("v8-aggregate-axes-")
                           for family in row["grammar_families"])
                       for row in groups if row["split"] == split)
            for split, _ in SPLIT_WEIGHTS
        },
        "maximum_independent_operation_clauses": 5,
        "maximum_status_extra_clauses": 2,
        "maximum_operations_plus_status_extra_clauses": 6,
        "variants_per_group": "variable; 8 elementary, 6 compositional",
        "variants_per_group_counts": dict(Counter(str(row["group_variants"]) for row in groups)),
        "post_filter_group_quotas": {
            split: dict(Counter(row["scenario_kind"] for row in groups
                                if row["split"] == split and not row["elementary"]))
            for split, _ in SPLIT_WEIGHTS
        },
        "mandatory_clean_train_groups": 4,
        "admitted_elementary_train_groups": sum(row["elementary"] for row in groups),
        "elementary_count": sum(row["elementary"] for row in groups),
        "elementary_split_groups": dict(Counter(row["split"] for row in groups
                                               if row["elementary"])),
        "developmental_split_groups": dict(Counter(row["split"] for row in groups
                                                  if row["developmental_compatibility"])),
        "basic_coverage_post_filter": coverage,
        "distributions": distributions,
        "calibration_eligible_sign_groups": eligible,
        "mode_sampling": (
            "Independent per operation; explicit shared canonical cases; no convention inference."
        ),
        "relative_mode_identity": "Supported dx/dy/wheel normalize mode to unknown.",
        "runtime_instructions_used": False,
    }


def prepare_control_text_dataset(
    output, *, seed=DEFAULT_SEED, scenario_groups=4000, excluded_semantic_ids=(),
    reservation_registry=None,
):
    """Write authenticated preparation artifacts only; never load or fit a model."""
    registry, metadata = read_partition_registry(reservation_registry, excluded_semantic_ids)
    records = generate_control_text_records(seed=seed, scenario_groups=scenario_groups,
                                            partition_registry=registry)
    summary = wording_manifest(records)
    _, gaps = _resolved_elementary_specs(registry)
    summary["historical_seal_elementary_coverage_gaps"] = gaps
    summary["historical_seal_elementary_gap_count"] = len(gaps)
    groups = _unique_groups(records)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                      for row in records).encode()
    manifest = {
        "seed": seed, "scenario_groups": scenario_groups, "records": len(records),
        "candidates": list(CONTROL_CANDIDATES),
        "split_records": dict(Counter(row["split"] for row in records)),
        "split_groups": dict(Counter(row["split"] for row in groups)),
        "records_sha256": hashlib.sha256(payload).hexdigest(), "records_bytes": len(payload),
        "split_before_paraphrase": True, "mapping_disjoint": True,
        "semantic_mapping_disjoint": True, "semantic_mapping_unique_per_group": True,
        "mapping_identity_definition": "v3 canonical candidate/status/mode/alias assignment",
        "null_mapping_exception": "All-unknown mappings partition the sorted named candidate set.",
        "model_input_fields": ["text"], "same_text_labels_consistent": True,
        "labels_derived_from": "structural specification only, before rendering",
        "maximum_supported_candidates": max(sum(label["status"] == "supported"
                                                  for label in row["labels"]) for row in groups),
        "statuses": ["supported", "unsupported", "unknown"],
        "modes": ["held", "press", "unknown"],
        "alias_group_semantics": "local equivalence classes; 0 means unspecified",
        "status": "prepared", "updates": 0, "model_loaded": False,
        "readiness": False, "ppo_eligible": False,
        "limitations": ["Synthetic specifications only; no learned quality measured.",
                        "Fresh sealed test is reserved until model freeze."],
        **summary, **metadata,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "records.jsonl").write_bytes(payload)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return manifest
