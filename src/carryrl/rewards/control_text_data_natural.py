"""Optional natural TRAIN views for a frozen relation-grounded V9 corpus.

Exactly one Russian and one English view per TRAIN group are replaced. Other
JSONL lines, including every evaluation row, retain their original bytes. Natural
wording is assembled from structural specifications and labels, without runtime
instructions, diagnostics, predictions, or manually selected evaluation text.
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from carryrl.rewards.control_text_data_v5 import REMAINDER_FORMS
from carryrl.rewards.control_text_data_v8 import _GOALS, _name, wording_manifest
from carryrl.rewards.control_text_data_v9 import _known_groups

NATURAL_VERSION = "carryrl.control_text_natural.v1"
NATURAL_FAMILIES = tuple(f"v9-natural-composition-{index:02d}" for index in range(12))
_ORDINALS = {
    "ru": ("первая", "вторая", "третья", "четвёртая", "пятая"),
    "en": ("first", "second", "third", "fourth", "fifth"),
}


def _names(candidates, language, *, accusative=False):
    return (" или " if language == "ru" else " or ").join(
        _name(candidate, language, accusative=accusative) for candidate in candidates
    )


def _binding(candidates, mode, language, style, *, defer_release=False):
    """Timing primitives vary independently of control identity and labels."""
    ru = language == "ru"
    names = _names(candidates, language)
    objects = _names(candidates, language, accusative=True)
    relative = all(candidate.startswith(("dx:", "dy:", "wheel:")) for candidate in candidates)
    mouse_only = all(candidate.startswith("mouse:") for candidate in candidates)
    mixed = any(candidate.startswith("mouse:") for candidate in candidates) and not mouse_only
    if relative:
        forms = (
            (f"Используй {names}; это доступное относительное движение, режим не указан.",
             f"Доступен ввод «{names}»; относительное перемещение "
             f"выполняется без заданного режима.",
             f"Эта функция использует {names}; движение доступно, сведения о режиме отсутствуют.")
            if ru else (
                f"Use {names}; this relative movement is available, with timing unspecified.",
                f"The input {names} is available; its relative movement has no specified timing.",
                f"This function uses {names}; movement is available without timing information.",
            )
        )
        return forms[style % len(forms)]
    if mode == "held":
        if defer_release:
            return (f"Удерживай {objects}; функция повторяется, пока ввод зажат." if ru else
                    f"Hold {objects}; this function repeats while the input stays down.")
        forms = (
            (f"Удерживай {objects}; функция повторяется до отпускания.",
             f"Держи ввод «{names}» зажатым; отпускай, когда функция должна прекратиться.",
             f"Функция повторяется, пока нажат ввод «{names}»; отпускание прекращает повторы.",
             f"Зажми {objects} и удерживай до окончания функции; затем отпусти.")
            if ru else (
                f"Hold {objects}; this function repeats until you release the input.",
                f"Keep {objects} down to repeat this function; letting go stops it.",
                f"This function repeats while {names} is pressed; release the input to stop.",
                f"Leave {objects} held down; release the input when this function should stop.",
            )
        )
        return forms[style % len(forms)]
    if mode != "press":
        raise ValueError("natural discrete binding requires executable held or press timing")
    if defer_release:
        if ru:
            return f"Коротко нажми {objects} один раз для этого выполнения."
        return (f"Give {objects} a single brief press for this execution." if mixed else
                f"Tap {objects} once for this execution.")
    if mouse_only:
        forms = (
            (f"Сделай один короткий щелчок: ввод «{names}»; отпускай между щелчками.",
             f"Ввод для щелчка — {names}; кратко нажми его и отпусти перед повторением.",
             f"Для этой функции выбери ввод «{names}», щёлкни и отпусти.")
            if ru else (
                f"Click {objects} once; release between clicks.",
                f"Use one short click on {objects}; let go before the next execution.",
                f"Give {objects} a quick press and let go before repeating this function.",
            )
        )
    elif mixed:
        forms = (
            (f"Кратко нажми выбранный ввод: {names}; затем отпусти до следующего выполнения.",
             f"Нужно короткое нажатие одного из вводов «{names}» с отпусканием между повторами.",
             f"Нажми выбранный ввод «{names}» и сразу отпусти; повтор требует нового нажатия.")
            if ru else (
                f"Give the chosen input, {names}, one brief press; then release before repeating.",
                f"Use a short press of either {names}; release between separate executions.",
                f"Briefly press the chosen input, {names}, and let go; repeat with a new press.",
            )
        )
    else:
        forms = (
            (f"Коротко нажми {objects}; перед повторением отпусти.",
             f"Нужно краткое нажатие ввода «{names}» с отпусканием между повторами.",
             f"Нажми {objects} и сразу отпусти; следующий запуск требует нового нажатия.")
            if ru else (
                f"Tap {objects} once for this function, then let go before repeating it.",
                f"Give {objects} a short press; release between separate executions.",
                f"Press {objects} briefly and release; repeat with a new press.",
            )
        )
    return forms[style % len(forms)]


def _relation_header(language, family):
    forms = (
        (
            "Ниже перечислены разные функции; они выполняют разные действия "
            "и не заменяют друг друга.",
            "Каждая функция имеет отдельный эффект; эффекты разных функций "
            "различны и невзаимозаменяемы.",
            "Функции ниже различаются по эффекту; ввод одной функции не заменяет ввод другой.",
            "Разные перечисленные функции выполняют разные действия без взаимной замены.",
            "У каждой следующей функции свой эффект; разные функции нельзя заменять друг другом.",
            "Следующие функции разделены по эффектам; эффекты различаются и "
            "взаимная замена исключена.",
            "Здесь разные функции имеют разные эффекты и не могут заменять друг друга.",
            "Для функций ниже действия различны; ввод другой функции не служит заменой.",
            "Перечисленные функции имеют разное назначение и разные невзаимозаменяемые эффекты.",
            "Функции далее выполняют разные действия; замена между разными функциями невозможна.",
            "Различай следующие функции: их действия разные и не взаимозаменяемы.",
            "Разные функции этого списка имеют разные эффекты; вводы между "
            "функциями не заменяются.",
        ) if language == "ru" else (
            "The following functions perform different actions and cannot replace one another.",
            "Each function has a separate effect; effects of different "
            "functions differ and are not interchangeable.",
            "The functions below differ in effect; the input for one cannot "
            "replace the input for another.",
            "Different listed functions perform different actions without mutual substitution.",
            "Each following function has its own effect; different "
            "functions cannot replace each other.",
            "The following functions are separated by effect; their effects "
            "differ and substitution is excluded.",
            "Different functions here have different effects and cannot replace one another.",
            "Actions of the functions below differ; an input for another "
            "function is not a substitute.",
            "The listed functions have different purposes and different "
            "noninterchangeable effects.",
            "The functions below perform different actions; substitution "
            "across functions is impossible.",
            "Distinguish the following functions: their actions differ and "
            "are not interchangeable.",
            "Different functions in this list have different effects; their "
            "inputs cannot be substituted across functions.",
        )
    )
    return forms[family]


def _natural_view(row, *, seed):
    language, ru = row["language"], row["language"] == "ru"
    shared = row.get("contrastive_pair_id", row["group_id"])
    render_key = f"{NATURAL_VERSION}:{seed}:{shared}:{row['variant_index']}:{language}"
    digest = hashlib.sha256(render_key.encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    family = int(digest[16:24], 16) % len(NATURAL_FAMILIES)
    groups = _known_groups(row)
    labels = {label["candidate"]: label for label in row["labels"]}
    operations = row["specification"]["operations"]
    known_ops = [operation for operation in operations
                 if any(labels[candidate]["status"] == "supported"
                        for candidate in operation["candidates"])]
    chunks = [_relation_header(language, family)] if groups else []
    deferred = None
    for index, (group, operation) in enumerate(zip(groups, known_ops, strict=True)):
        modes = {labels[candidate]["mode"] for candidate in group}
        relative = all(candidate.startswith(("dx:", "dy:", "wheel:")) for candidate in group)
        if not relative and len(modes) != 1:
            raise ValueError("one natural discrete function has inconsistent supervised modes")
        mode = "unknown" if relative else next(iter(modes))
        defer = family % 3 == 0 and deferred is None and index < len(groups) - 1 and not relative
        ordinal = _ORDINALS[language][index]
        label = f"{ordinal.capitalize()} функция" if ru else f"The {ordinal} function"
        chunks.append(label + ": " + _binding(group, mode, language, family + index,
                                             defer_release=defer))
        if len(group) > 1:
            chunks.append("Указанные варианты равнозначны для этой функции." if ru else
                          "The listed choices are interchangeable for this function.")
        if defer:
            deferred = (ordinal, mode)
    if deferred is not None:
        ordinal, mode = deferred
        if ru:
            if mode == "press":
                genitive = {"первая": "первой", "вторая": "второй", "третья": "третьей",
                            "четвёртая": "четвёртой", "пятая": "пятой"}[ordinal]
                chunks.append(f"После каждого выполнения {genitive} функции отпускай её ввод "
                              "перед следующим выполнением.")
            else:
                chunks.append(f"{ordinal.capitalize()} функция повторяется до отпускания её ввода; "
                              "отпусти его, когда повторы должны прекратиться.")
        elif mode == "press":
            chunks.append(f"After each execution of the {ordinal} function, release its input "
                          "before the next execution.")
        else:
            chunks.append(f"The {ordinal} function repeats until its input is released; "
                          "release it when repetition should stop.")
    for operation in operations:
        for candidate in operation["candidates"]:
            if labels[candidate]["status"] != "unknown":
                continue
            name = _name(candidate, language)
            chunks.append(f"Ввод «{name}» описан как доступный, но режим нажатия неизвестен; "
                          "исполнимость не установлена." if ru else (
                              f"Input {name} is described as available, but its timing is "
                              "unspecified; executable support is unknown."
                          ))
    for candidate in row["specification"]["uncertain"]:
        name = _name(candidate, language)
        conditional = row["specification"]["kind"] == "conditional"
        chunks.append((f"Ввод «{name}» зависит от неизвестного условия; доступность неизвестна."
                       if conditional else f"Ввод «{name}»: доступность неизвестна.") if ru else (
                           f"Input {name} depends on an unknown condition; availability is unknown."
                           if conditional else f"Input {name}: availability is unknown."
                       ))
    for candidate in row["specification"]["forbidden"]:
        name = _name(candidate, language)
        chunks.append(f"Ввод «{name}» запрещён." if ru else f"Input {name} is unsupported.")
    remainder = row["specification"]["remainder"]
    chunks.append(REMAINDER_FORMS[remainder][language][rng.randrange(8)])
    result = dict(row)
    result["text"] = " ".join(chunks)
    result["template_family"] = NATURAL_FAMILIES[family]
    result["grammar_families"] = [NATURAL_FAMILIES[family]]
    narrative = rng.random() < 0.25
    if narrative:
        contexts = (
            ("Полосатые области закрывают часть поля; "
             "неинтерактивные значки подсказок показывают справку.",
             "После короткой заставки начинается активная сцена; функции ниже описывают эту сцену.",
             "Декоративные значки показывают ход задания; их цвета не меняют описанные функции.")
            if ru else (
                "Striped regions cover part of the field; "
                "noninteractive hint icons display help.",
                "An active scene follows a brief title card; the functions "
                "below describe that scene.",
                "Decorative icons show task progress; their colors do not "
                "change the described functions.",
            )
        )
        result["text"] = (
            rng.choice(_GOALS[language]) + " " + rng.choice(contexts) + " " + result["text"]
        )
    return result, deferred is not None, narrative


def _targets(records):
    groups = {}
    for row in records:
        if row["split"] == "train":
            groups.setdefault(row["group_id"], []).append(row)
    selected = set()
    for group_id, rows in groups.items():
        variants = rows[0]["group_variants"]
        if variants not in {6, 8} or len(rows) != variants:
            raise ValueError("natural repair requires complete six- or eight-view TRAIN groups")
        if any(row["group_variants"] != variants for row in rows):
            raise ValueError("TRAIN group has inconsistent variant metadata")
        if {row["variant_index"] for row in rows} != set(range(variants)):
            raise ValueError("TRAIN group has duplicated or missing variant indices")
        targets = (2, 5) if variants == 6 else (3, 7)
        if {row["language"] for row in rows if row["variant_index"] == targets[0]} != {"ru"} or {
            row["language"] for row in rows if row["variant_index"] == targets[1]
        } != {"en"}:
            raise ValueError("natural repair target views must be Russian then English")
        selected.update((group_id, variant) for variant in targets)
    return selected


def _line_digests(records, lines):
    return {split: hashlib.sha256(b"".join(line for row, line in zip(records, lines, strict=True)
                                           if row["split"] == split)).hexdigest()
            for split in ("dev", "calibration", "sealed_test")}


def prepare_natural_control_dataset(base_v9, output, *, seed=20261026):
    """Prepare two natural TRAIN views per group, preserving every other line."""
    from carryrl.rewards.control_text_training import read_control_dataset

    base_v9, output = Path(base_v9), Path(output)
    manifest, records = read_control_dataset(base_v9)
    if manifest.get("version") != "carryrl.control_text_data.v9":
        raise ValueError("natural TRAIN repair requires a V9 base dataset")
    if manifest.get("natural_augmentation") is not None:
        raise ValueError("natural TRAIN repair requires an unaugmented V9 base")
    payload = (base_v9 / "records.jsonl").read_bytes()
    source_manifest = (base_v9 / "manifest.json").read_bytes()
    if (hashlib.sha256(payload).hexdigest() != manifest["records_sha256"]
            or json.loads(source_manifest) != manifest):
        raise ValueError("base V9 changed during natural preparation")
    lines = payload.splitlines(keepends=True)
    if len(lines) != len(records):
        raise ValueError("base JSONL must have one record per line")
    targets = _targets(records)
    repaired, output_lines, deferred_count, narrative_count = [], [], 0, 0
    for row, line in zip(records, lines, strict=True):
        if row["split"] == "train" and (row["group_id"], row["variant_index"]) in targets:
            row, deferred, narrative = _natural_view(row, seed=seed)
            deferred_count += deferred
            narrative_count += narrative
            line = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        repaired.append(row)
        output_lines.append(line)
    output_payload = b"".join(output_lines)
    if _line_digests(records, lines) != _line_digests(repaired, output_lines):
        raise ValueError("natural preparation changed evaluation bytes")
    summary = wording_manifest(repaired)
    witness_coverage, witness_groups = Counter(), {}
    for row in repaired:
        groups = _known_groups(row)
        has_witness = any(family.startswith("v9-relation-witness-")
                          for family in row["grammar_families"])
        split = row["split"]
        if has_witness:
            witness_coverage[f"{split}|witness_records"] += 1
            witness_coverage[f"{split}|multiple_known_group_records"] += len(groups) > 1
            witness_coverage[f"{split}|within_group_alias_records"] += any(
                len(g) > 1 for g in groups
            )
            witness_groups.setdefault(split, set()).add(row["group_id"])
        elif groups:
            witness_coverage[f"{split}|known_records_without_bracket_witness"] += 1
        else:
            witness_coverage[f"{split}|no_known_group_records"] += 1
    metadata = {
        "version": NATURAL_VERSION, "seed": seed,
        "base_dataset": str(base_v9.resolve()),
        "base_records_sha256": hashlib.sha256(payload).hexdigest(),
        "base_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
        "changed_records": len(targets), "changed_records_split": {"train": len(targets)},
        "changed_train_groups": len(targets) // 2,
        "replacement_variant_indices": {"6": [2, 5], "8": [3, 7]},
        "deferred_release_records": deferred_count,
        "narrative_context_records": narrative_count,
        "unselected_record_bytes_preserved": True, "heldout_record_bytes_preserved": True,
        "heldout_records_sha256_by_split": _line_digests(records, lines),
        "structural_labels_ids_splits_pair_metadata_unchanged": True,
        "relation_semantics": (
            "Different functions have different effects; choices within a function are equivalent."
        ),
        "timing_core_families": ["hold", "keep_down", "repeat_while_pressed", "tap",
                                 "short_press", "press_then_release", "mouse_click",
                                 "ordinal_deferred_release", "relative_movement"],
        "style_sampling": (
            "Seeded independently of physical names/modes; matched pair views share the seed."
        ),
        "runtime_diagnostic_or_evaluation_text_used_for_generation": False,
        "effectiveness_measured": False,
    }
    current_relation = dict(manifest.get("relation_grounding_repair", {}))
    current_relation["coverage_records"] = dict(witness_coverage)
    current_relation["coverage_scope"] = (
        "Current rows; natural views have explicit unbracketed relations."
    )
    updated = {
        **manifest, **summary,
        "version": "carryrl.control_text_data.v9", "wording_version": 9,
        "semantic_generator_version": "carryrl.control_text_data.v9",
        "scenario_kinds": manifest["scenario_kinds"],
        "records_sha256": hashlib.sha256(output_payload).hexdigest(),
        "records_bytes": len(output_payload),
        "template_family_split_ownership": {
            **manifest["template_family_split_ownership"],
            **dict.fromkeys(NATURAL_FAMILIES, "train"),
        },
        "natural_family_split_ownership": dict.fromkeys(NATURAL_FAMILIES, "train"),
        "relation_witness_group_counts": {
            split: len(witness_groups.get(split, set()))
            for split in ("train", "dev", "calibration", "sealed_test")
        },
        "relation_grounding_repair": current_relation,
        "natural_augmentation": metadata,
        "status": "prepared", "updates": 0, "model_loaded": False,
        "readiness": False, "ppo_eligible": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "records.jsonl").write_bytes(output_payload)
    (output / "manifest.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return updated
