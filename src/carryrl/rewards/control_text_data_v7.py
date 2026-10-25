"""V7: sparse controls and split-held-out internal binding compositions.

Specifications and effective semantic identity precede rendering. Shared words
are lexical primitives; the eight binding/order patterns are assigned to splits,
not sampled across them. This module never reads runtime/game instructions.
"""

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from carryrl.rewards.control_text_data import (
    CONTROL_CANDIDATES,
    SPLIT_WEIGHTS,
    _semantic_split,
    semantic_mapping_identity,
)
from carryrl.rewards.control_text_data_v4 import ControlSpec, OperationSpec, _name
from carryrl.rewards.control_text_data_v5 import REMAINDER_FORMS
from carryrl.rewards.control_text_data_v6 import (
    _AXES,
    _DISTRACTORS,
    _GOALS,
    _HORIZONTAL,
    _NAVIGATION,
    _group_name,
    _validated_exclusions,
    annotate_spec,
    read_reservation_registry,
)
from carryrl.rewards.control_text_data_v6 import (
    _sample_spec as _sample_v6,
)

CONTROL_TEXT_DATA_VERSION = "carryrl.control_text_data.v7"
DEFAULT_SEED = 20261023
_SCHEDULE = (
    "navigation",
    "horizontal",
    "shared_timing",
    "adjacent_modes",
    "mouse_context",
    "sparse_partial",
    "conditional",
    "dense",
    "implicit",
    "compact_alias",
    "navigation",
    "horizontal",
    "shared_timing",
    "adjacent_modes",
    "mouse_context",
    "goal_context",
    "complete",
    "incomplete",
    "mixed",
    "unknown",
)
SCENARIO_KINDS = tuple(dict.fromkeys(_SCHEDULE))
INTERNAL_FAMILIES = {
    split: (2 * index, 2 * index + 1) for index, (split, _) in enumerate(SPLIT_WEIGHTS)
}


def _sample_spec(rng, kind):
    if kind in {"navigation", "horizontal"}:
        pairs = _NAVIGATION if kind == "navigation" else _HORIZONTAL
        core = {candidate for pair in pairs for candidate in pair}
        pool = [value for value in CONTROL_CANDIDATES if value not in core]
        rng.shuffle(pool)
        # Minimal cores are valid; no mandatory positive or negative noise.
        extras = 0 if rng.random() < 0.55 else rng.randint(1, 7 - len(pairs))
        shared = rng.choice(("held", "press")) if rng.random() < 0.5 else None
        groups = list(pairs) + [(value,) for value in pool[:extras]]
        operations = [
            OperationSpec(
                f"{identifier:03x}",
                group,
                shared if shared is not None and group in pairs else rng.choice(("held", "press")),
            )
            for identifier, group in zip(rng.sample(range(4096), len(groups)), groups, strict=True)
        ]
        rng.shuffle(operations)
        remainder = "unknown" if rng.random() < 0.25 else "unsupported"
        forbidden_count = 0 if rng.random() < 0.75 else rng.randint(1, 3)
        return ControlSpec(
            tuple(operations), (), tuple(pool[extras : extras + forbidden_count]), remainder, kind
        )
    if kind in {"shared_timing", "adjacent_modes"}:
        keys = [candidate for candidate in CONTROL_CANDIDATES if candidate.startswith("key:")]
        rng.shuffle(keys)
        count = rng.randint(2, 5)
        modes = [rng.choice(("held", "press")) for _ in range(count)]
        if kind == "shared_timing":
            modes = [rng.choice(("held", "press"))] * count
        else:
            modes[:2] = rng.sample(("held", "press"), 2)
        operations = []
        for identifier, mode in zip(rng.sample(range(4096), count), modes, strict=True):
            size = rng.choice((1, 2))
            operations.append(OperationSpec(f"{identifier:03x}", tuple(keys[:size]), mode))
            del keys[:size]
        rng.shuffle(operations)
        return ControlSpec(
            tuple(operations), (), (), "unknown" if rng.random() < 0.25 else "unsupported", kind
        )
    source = "mouse_axes" if kind == "mouse_context" else kind
    for _ in range(1000):
        spec = _sample_v6(rng, source)
        # Bound independent clauses, not physical control count (up to 13).
        # This is a generation constraint, never truncation of rendered text.
        if len(spec.operations) <= 7:
            return replace(spec, kind=kind)
    raise ValueError("unable to sample a specification within the clause budget")


def _timing(names, mode, language, family, *, relative=False):
    """Eight actual internal syntaxes, two per split; no outer-only holdout."""
    ru = language == "ru"
    if mode == "unknown":
        forms = (
            (
                f"Доступен ввод {names}; режим не указан",
                f"{names} доступны; длительность не задана",
                f"Режим не указан для доступного ввода {names}",
                f"Доступность {names} подтверждена, режим неизвестен",
                f"Для {names} известна доступность, но не режим",
                f"Ввод {names} доступен без сведений о длительности",
                f"Доступный ввод: {names}; о режиме сведений нет",
                f"Можно использовать {names}; как долго — не указано",
            )
            if ru
            else (
                f"Input {names} is available; timing is unspecified",
                f"{names} are available; duration is not stated",
                f"Timing is unspecified for available input {names}",
                f"Availability of {names} is confirmed, timing unknown",
                f"For {names}, availability is known but timing is not",
                f"Input {names} is available without duration information",
                f"Available input: {names}; no timing information",
                f"You can use {names}; how long is unspecified",
            )
        )
        return forms[family]
    if relative:
        held = "непрерывно повторяй движение" if ru else "repeat the movement continuously"
        press = (
            "сделай одно короткое движение и остановись"
            if ru
            else "make one brief movement and stop"
        )
    else:
        held = (
            "удерживай нужный ввод; удержание повторяет действие"
            if ru
            else "hold the desired input; holding repeats its action"
        )
        press = (
            "нажми один раз и отпусти между действиями"
            if ru
            else "press once and release between actions"
        )
    command = held if mode == "held" else press
    forms = (
        (
            f"Для {names}: {command}",
            f"{names}: {command}",
            f"Чтобы выполнить действие, {command}: {names}",
            f"Режим для {names} — {command}",
            f"Используя {names}, {command}",
            f"Действие через {names} требует: {command}",
            f"{command}; ввод: {names}",
            f"Команды {names} выполняются так: {command}",
        )
        if ru
        else (
            f"For {names}: {command}",
            f"{names}: {command}",
            f"To perform the action, {command}: {names}",
            f"The timing for {names} is: {command}",
            f"When using {names}, {command}",
            f"An action through {names} requires: {command}",
            f"{command}; input: {names}",
            f"Controls {names} are executed as follows: {command}",
        )
    )
    return forms[family]


def _binding(operation, language, family):
    values = operation.candidates
    relative = [value.startswith(("dx:", "dy:", "wheel:")) for value in values]
    if any(relative) and not all(relative):
        choices = (" или " if language == "ru" else " or ").join(
            _binding(
                OperationSpec(operation.identifier, (value,), operation.mode), language, family
            )
            for value in values
        )
        return ("Равнозначные варианты: " if language == "ru" else "Equivalent choices: ") + choices
    names = _group_name(operation, language, compact=False)
    clause = _timing(names, operation.mode, language, family, relative=all(relative))
    if len(values) > 1:
        clause += (
            "; варианты равнозначны" if language == "ru" else "; these alternatives are equivalent"
        )
    return clause


def _canonical(operations, kind, language, family):
    ru = language == "ru"
    names = (
        ("WASD или стрелки" if ru else "WASD or arrow keys")
        if kind == "navigation"
        else ("A/D или стрелки влево/вправо" if ru else "A/D or left/right arrows")
    )
    mode = operations[0].mode
    # The compact conventional directional notation has its normal directional
    # aliases. Different directions are separate actions, not one alias group.
    if mode == "held":
        direct = (
            f"Удерживай нужную клавишу {names}; удержание повторяет движение"
            if ru
            else f"Hold the desired {names}; holding repeats movement"
        )
    else:
        direct = (
            f"Нажми нужную клавишу {names} один раз; отпускай между нажатиями"
            if ru
            else f"Press the desired {names} once; release between presses"
        )
    if family == 0:
        return direct
    return _timing(names, mode, language, family)


def _mouse_axes(language, family):
    ru = language == "ru"
    forms = (
        (
            "Движение мыши перемещает указатель влево, вправо, вверх и вниз",
            "Движение мыши управляет прицелом по горизонтали и вертикали в обе стороны",
            "Влево, вправо, вверх и вниз указатель перемещается движением мыши",
            "Для прицеливания доступно движение мыши: оба направления горизонтали и вертикали",
            "Перемещай указатель движением мыши; доступны обе стороны каждой из двух осей",
            "Прицел следует движению мыши: обе стороны горизонтали и вертикали",
            "Указатель следует движениям мыши: влево/вправо и вверх/вниз",
            "Доступное управление прицелом — движение мыши, обе стороны каждой из двух осей",
        )
        if ru
        else (
            "Mouse movement moves the cursor left, right, up and down",
            "Mouse movement controls aim horizontally and vertically in both directions",
            "Left, right, up and down cursor placement is provided by mouse movement",
            "For aiming mouse movement is available: both horizontal and vertical directions",
            "Move the mouse to position the cursor; both signs of its two axes are available",
            "Aim follows mouse motion: both horizontal and vertical directions",
            "The cursor follows mouse movements: left/right and up/down",
            "Aiming uses mouse movement, both signs of horizontal and vertical axes",
        )
    )
    return forms[family]


def render_spec(spec, *, language, variant, split, seed, pair_seed):
    rng, ru = random.Random(seed), language == "ru"
    family = INTERNAL_FAMILIES[split][variant % 2]
    operations = list(spec.operations)
    rng.shuffle(operations)
    chunks, families = [], {f"v7-internal-{family}"}
    canonical = []
    if spec.kind in {"navigation", "horizontal"}:
        pairs = _NAVIGATION if spec.kind == "navigation" else _HORIZONTAL
        core = [operation for operation in operations if operation.candidates in pairs]
        if len(core) == len(pairs) and len({operation.mode for operation in core}) == 1:
            canonical = core
            chunks.append(_canonical(core, spec.kind, language, family) + ".")
            families.add(f"v7-internal-{family}-canonical-shared")
        else:
            families.add(f"v7-internal-{family}-canonical-expanded")
    axes = [
        operation
        for operation in operations
        if spec.kind == "mouse_context" and operation.candidates[0] in _AXES
    ]
    if len(axes) == 4:
        chunks.append(_mouse_axes(language, family) + ".")
        families.add(f"v7-internal-{family}-mouse-context")
    else:
        axes = []
    clauses = [
        _binding(operation, language, family)
        for operation in operations
        if operation not in canonical and operation not in axes
    ]
    if clauses:
        # Conjunction separates operations; 'or' remains local to alternatives.
        joiner = ("; затем " if ru else "; then ") if family % 2 else ("; и " if ru else "; and ")
        chunks.append(
            ("Независимые действия: " if ru else "Independent actions: ")
            + joiner.join(clauses)
            + "."
        )
    for candidate in spec.uncertain:
        name = _name(candidate, language)
        if spec.kind == "conditional":
            clause = (
                f"Ввод {name} доступен только при неизвестном условии"
                if ru
                else f"Input {name} is available only under an unknown condition"
            )
        else:
            clause = f"{name}: доступность неизвестна" if ru else f"{name}: availability is unknown"
        chunks.append(clause + ".")
    for candidate in spec.forbidden:
        name = _name(candidate, language)
        chunks.append((f"Ввод {name} запрещён" if ru else f"Input {name} is unsupported") + ".")
    if spec.kind == "goal_context" or rng.random() < 0.25:
        chunks.insert(0, rng.choice(_GOALS[language]))
        chunks.append(rng.choice(_DISTRACTORS[language]))
    # Shared scope vocabulary is a primitive, not the held-out internal grammar.
    offset = int(hashlib.sha256(pair_seed.encode()).hexdigest(), 16) % 8
    chunks.append(REMAINDER_FORMS[spec.remainder][language][(offset + variant) % 8])
    return " ".join(chunks), sorted(families)


def generate_control_text_records(
    *, seed=DEFAULT_SEED, scenario_groups=4000, excluded_semantic_ids=()
):
    if type(scenario_groups) is not int or scenario_groups < 100 or scenario_groups % 100:
        raise ValueError("scenario_groups must be a positive multiple of 100")
    if scenario_groups > 40000:
        raise ValueError("too many unique scenario mappings")
    excluded = _validated_exclusions(excluded_semantic_ids)
    rng, seen, records = random.Random(seed), set(), []
    for split, weight in SPLIT_WEIGHTS:
        for local in range(scenario_groups * weight // 100):
            kind = _SCHEDULE[local % len(_SCHEDULE)]
            for _ in range(100000):
                spec = _sample_spec(rng, kind)
                labels = annotate_spec(spec)
                identity = semantic_mapping_identity(labels, spec.declared_candidates)
                if (
                    identity not in excluded
                    and identity not in seen
                    and _semantic_split(identity, seed) == split
                ):
                    seen.add(identity)
                    break
            else:
                raise ValueError("requested budget exhausts assigned semantic mappings")
            index = len(seen) - 1
            for variant, language in enumerate(("ru", "ru", "en", "en")):
                text, grammars = render_spec(
                    spec,
                    language=language,
                    variant=variant,
                    split=split,
                    seed=rng.getrandbits(64),
                    pair_seed=identity,
                )
                records.append(
                    {
                        "group_id": f"v7-scenario-{index:05d}",
                        "split": split,
                        "mapping_id": identity,
                        "semantic_mapping_id": identity,
                        "declared_candidates": list(spec.declared_candidates),
                        "template_family": f"v7-internal-{INTERNAL_FAMILIES[split][variant % 2]}",
                        "grammar_families": grammars,
                        "scenario_kind": kind,
                        "language": language,
                        "text": text,
                        "labels": labels,
                        "specification": asdict(spec),
                    }
                )
    text_labels = {}
    for row in records:
        if (
            text_labels.setdefault(row["text"], row["semantic_mapping_id"])
            != row["semantic_mapping_id"]
        ):
            raise ValueError("same text has conflicting labels")
    return records


def wording_manifest(records):
    from carryrl.rewards.control_text_data_v6 import wording_manifest as summarize_v6

    summary = summarize_v6(records)
    groups = records[::4]
    summary.update(
        {
            "version": CONTROL_TEXT_DATA_VERSION,
            "wording_version": 7,
            "semantic_generator_version": CONTROL_TEXT_DATA_VERSION,
            "scenario_kinds": list(SCENARIO_KINDS),
            "template_family_definition": (
                "Internal binding syntax and order; two families per split."
            ),
            "internal_grammar_families_disjoint": True,
            "outer_template_families_disjoint": False,
            "paraphrase_holdout": (
                "Internal binding/order compositions held out by split; lexical primitives shared."
            ),
            "id_presence_semantics": "Anonymous controls; no operation IDs in model text.",
            "redundant_negative_wording": "All explicitly forbidden inputs are rendered.",
            "maximum_independent_operation_clauses": 7,
            "canonical_group_patterns": {
                split: dict(
                    Counter(
                        family
                        for row in groups
                        if row["split"] == split
                        for family in row["grammar_families"]
                        if "canonical-" in family
                    )
                )
                for split, _ in SPLIT_WEIGHTS
            },
            "canonical_supported_count_groups": {
                split: {
                    kind: dict(
                        Counter(
                            str(sum(label["status"] == "supported" for label in row["labels"]))
                            for row in groups
                            if row["split"] == split and row["scenario_kind"] == kind
                        )
                    )
                    for kind in ("navigation", "horizontal")
                }
                for split, _ in SPLIT_WEIGHTS
            },
            "mode_sampling": (
                "Independent random operation timing, plus randomly shared timing for some groups; "
                "never scenario index."
            ),
            "runtime_instructions_used": False,
        }
    )
    return summary


def prepare_control_text_dataset(
    output,
    *,
    seed=DEFAULT_SEED,
    scenario_groups=4000,
    excluded_semantic_ids=(),
    reservation_registry=None,
):
    """Write new authenticated artifacts only; no fitting or evaluation."""
    exclusions, metadata = read_reservation_registry(reservation_registry, excluded_semantic_ids)
    records = generate_control_text_records(
        seed=seed, scenario_groups=scenario_groups, excluded_semantic_ids=exclusions
    )
    summary = wording_manifest(records)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records
    ).encode()
    (output / "records.jsonl").write_bytes(payload)
    groups = records[::4]
    distributions = {}
    for split, _ in SPLIT_WEIGHTS:
        rows = [row for row in groups if row["split"] == split]
        labels = [label for row in rows for label in row["labels"]]
        distributions[split] = {
            "scenario_groups": dict(Counter(row["scenario_kind"] for row in rows)),
            "supported_count_groups": dict(
                Counter(
                    str(sum(label["status"] == "supported" for label in row["labels"]))
                    for row in rows
                )
            ),
            "status_candidates": dict(Counter(label["status"] for label in labels)),
            "mode_candidates": dict(Counter(label["mode"] for label in labels)),
            "candidate_status_mode": dict(
                Counter(
                    f"{label['candidate']}|{label['status']}|{label['mode']}" for label in labels
                )
            ),
            "operation_modes_by_position": {
                str(position): dict(
                    Counter(
                        operation["mode"]
                        for row in rows
                        for offset, operation in enumerate(row["specification"]["operations"])
                        if offset == position
                    )
                )
                for position in range(13)
            },
        }
    manifest = {
        "seed": seed,
        "scenario_groups": scenario_groups,
        "records": len(records),
        "variants_per_group": 4,
        "candidates": list(CONTROL_CANDIDATES),
        "split_records": dict(Counter(row["split"] for row in records)),
        "split_groups": dict(Counter(row["split"] for row in groups)),
        "records_sha256": hashlib.sha256(payload).hexdigest(),
        "records_bytes": len(payload),
        "split_before_paraphrase": True,
        "mapping_disjoint": True,
        "semantic_mapping_disjoint": True,
        "semantic_mapping_unique_per_group": True,
        "mapping_identity_definition": "v3 canonical candidate/status/mode/alias assignment",
        "null_mapping_exception": "All-unknown mappings partition the sorted named candidate set.",
        "model_input_fields": ["text"],
        "same_text_labels_consistent": True,
        "operation_ids_and_render_order_randomized": True,
        "distributions": distributions,
        "maximum_supported_candidates": max(
            sum(label["status"] == "supported" for label in row["labels"]) for row in groups
        ),
        "statuses": ["supported", "unsupported", "unknown"],
        "modes": ["held", "press", "unknown"],
        "alias_group_semantics": "local equivalence classes; 0 means unspecified",
        "status": "prepared",
        "updates": 0,
        "model_loaded": False,
        "readiness": False,
        "ppo_eligible": False,
        "limitations": [
            "Synthetic specifications only; no learned quality measured.",
            "Sealed test reserved until model freeze; runtime instructions not used.",
        ],
        **summary,
        **metadata,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
