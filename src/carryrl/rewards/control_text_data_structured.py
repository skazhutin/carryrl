"""V12 TRAIN text with concrete task effects and multiple binding structures.

Only remaining ordinary TRAIN views are replaced. Existing natural/transfer
views and all evaluation bytes remain frozen. Effects are synthetic task-local
meanings, assigned once per semantic group or matched context; they are not game
API labels. The renderer reads no runtime instructions, diagnostics or predictions.
"""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from carryrl.rewards.control_text_data_natural import _binding, _names
from carryrl.rewards.control_text_data_v5 import REMAINDER_FORMS
from carryrl.rewards.control_text_data_v8 import _name, wording_manifest
from carryrl.rewards.control_text_data_v9 import _known_groups

CONTROL_TEXT_DATA_VERSION = "carryrl.control_text_data.v12"
STRUCTURED_VERSION = "carryrl.control_text_structured.v1"
FAMILIES = tuple(f"v12-structured-{index}" for index in range(4))
_EFFECTS = (
    ("rotate_symbol", "повернуть узорный символ", "rotate the patterned symbol"),
    ("increase_counter", "увеличить счётчик", "increase the counter"),
    ("change_highlight", "сменить цвет подсветки", "change the highlight color"),
    ("reset_gauge", "сбросить шкалу", "reset the gauge"),
    ("confirm_arrangement", "подтвердить расстановку", "confirm the arrangement"),
    ("swap_textures", "поменять местами две текстуры", "swap the two textures"),
    ("expand_figure", "увеличить контурную фигуру", "expand the outlined figure"),
    ("reverse_shapes", "развернуть порядок фигур", "reverse the order of the shapes"),
)
_DIRECTIONS = {
    "up": ("вверх", "upward"), "down": ("вниз", "downward"),
    "left": ("влево", "left"), "right": ("вправо", "right"),
}
_NAVIGATION = {frozenset(("key:w", "key:up")): "up",
               frozenset(("key:s", "key:down")): "down",
               frozenset(("key:a", "key:left")): "left",
               frozenset(("key:d", "key:right")): "right"}
_RELATIVE = {"dx:-": "left", "dx:+": "right", "dy:-": "up", "dy:+": "down"}
_BRIDGE = {
    "ru": (
        "У светлой границы расположены узорные фигуры разного размера. "
        "Краткая подпись описывает их поверхность и расположение, а декоративный фон "
        "помогает различать участки поля. Эти детали относятся к виду сцены; они не "
        "меняют назначение указанного действия."
    ),
    "en": (
        "Patterned figures of several sizes sit beside the pale boundary. "
        "A short caption describes their surfaces and placement, while the decorative "
        "background distinguishes regions of the field. Return to the pale region in "
        "the description; these details describe appearance and do not change the named action."
    ),
}


def _seed(row, seed, *, view=False):
    context = row.get("contrastive_pair_id", row["group_id"])
    suffix = f":{row['variant_index']}:{row['language']}" if view else ""
    payload = f"{STRUCTURED_VERSION}:{seed}:{context}{suffix}".encode()
    return int(hashlib.sha256(payload).hexdigest(), 16)


def structured_effects(row, *, seed=20261028):
    """Bind one distinct concrete effect to each supported specification group."""
    rng, available = random.Random(_seed(row, seed)), list(_EFFECTS)
    rng.shuffle(available)
    result = []
    for group in _known_groups(row):
        direction = _NAVIGATION.get(frozenset(group)) if row["specification"]["kind"] in {
            "navigation", "horizontal"
        } else None
        if direction:
            ru, en = _DIRECTIONS[direction]
            effect = (f"token_{direction}", f"сдвинуть жетон {ru}", f"move the token {en}")
        elif len(group) == 1 and group[0] in _RELATIVE:
            direction = _RELATIVE[group[0]]
            ru, en = _DIRECTIONS[direction]
            effect = (f"marker_{direction}", f"сдвинуть метку {ru}", f"move the marker {en}")
        elif len(group) == 1 and group[0].startswith("wheel:"):
            down = group[0] == "wheel:-"
            effect = ("strip_down" if down else "strip_up",
                      "прокрутить ленту вниз" if down else "прокрутить ленту вверх",
                      "scroll the strip down" if down else "scroll the strip up")
        else:
            effect = available.pop()
        result.append((group, effect))
    if len({effect[0] for _, effect in result}) != len(result):
        raise ValueError("different operation groups must have different concrete effects")
    return result


def _label(effect, language):
    return effect[1 if language == "ru" else 2]


def _aliases(group, effect, language):
    if len(group) < 2:
        return ""
    name = _label(effect, language)
    return (f"Указанные варианты равнозначны: каждый выполняет именно действие «{name}»."
            if language == "ru" else
            f"These input choices are interchangeable: each performs exactly the action '{name}'.")


def _distinct(effects, language):
    if len(effects) < 2:
        return ""
    names = "; ".join(f"«{_label(effect, language)}»" if language == "ru" else
                      f"'{_label(effect, language)}'" for _, effect in effects)
    return (f"Действия {names} имеют разные эффекты и не заменяют друг друга."
            if language == "ru" else
            f"The actions {names} have different effects and cannot replace one another.")


def _statuses(row, language):
    ru, chunks = language == "ru", []
    labels = {label["candidate"]: label for label in row["labels"]}
    for operation in row["specification"]["operations"]:
        for candidate in operation["candidates"]:
            if labels[candidate]["status"] != "unknown":
                continue
            name = _name(candidate, language)
            chunks.append(f"Ввод «{name}» указан как доступный, но режим нажатия не описан; "
                          "исполнимость пока неизвестна." if ru else
                          f"Input {name} is listed as available, "
                          "but its press timing is not specified; "
                          "executable support remains unknown.")
    for candidate in row["specification"]["uncertain"]:
        name = _name(candidate, language)
        if row["specification"]["kind"] == "conditional":
            chunks.append(f"Ввод «{name}» доступен только при неизвестном условии; "
                          "выполнено ли условие, неизвестно, поэтому доступность не установлена."
                          if ru else
                          f"Input {name} requires an unknown condition; "
                          "whether it holds is unknown, "
                          "so availability is not established.")
        else:
            chunks.append(f"Доступность ввода «{name}» неизвестна." if ru else
                          f"Availability of input {name} is unknown.")
    for candidate in row["specification"]["forbidden"]:
        name = _name(candidate, language)
        chunks.append(f"Ввод «{name}» запрещён." if ru else f"Input {name} is unsupported.")
    return chunks


def render_structured_view(row, *, seed=20261028):
    """Use one of four actual layouts, preserving the complete supervision."""
    if row["split"] != "train":
        raise ValueError("structured text rendering is TRAIN only")
    language, ru = row["language"], row["language"] == "ru"
    effects = structured_effects(row, seed=seed)
    rng = random.Random(_seed(row, seed, view=True))
    size = row["group_variants"]
    natural_slots = {2, 5} if size == 6 else {3, 7}
    ordinary_slots = [index for index in range(size) if index not in natural_slots]
    if row["variant_index"] not in ordinary_slots:
        raise ValueError("structured renderer accepts only remaining ordinary views")
    frame = (ordinary_slots.index(row["variant_index"]) + _seed(row, seed)) % len(FAMILIES)
    labels = {label["candidate"]: label for label in row["labels"]}
    chunks, info = [], {"effects": [(list(group), effect[0]) for group, effect in effects],
                        "frame": frame}

    def direct(group, effect, position):
        name, mode = _label(effect, language), labels[group[0]]["mode"]
        timing = _binding(group, mode, language, position + row["variant_index"])
        if frame == 0:
            text = f"Здесь нужно {name}. {timing}" if ru else f"Here you need to {name}. {timing}"
        elif frame == 1:
            names = _names(group, language)
            text = (f"Ввод «{names}» выполняет действие «{name}». {timing}" if ru else
                    f"Input {names} performs the action '{name}'. {timing}")
        else:
            text = (f"Способ выполнить действие «{name}»: {timing}" if ru else
                    f"To perform '{name}': {timing}")
        return [text, _aliases(group, effect, language)]

    if frame == 2 and effects:
        names = "; ".join(_label(effect, language) for _, effect in effects)
        chunks.append(f"Задача допускает такие действия: {names}." if ru else
                      f"The task offers these actions: {names}.")
        for position, (group, effect) in enumerate(reversed(effects)):
            chunks.extend(direct(group, effect, position))
    elif frame == 3 and len(effects) > 1:
        discrete = [(group, effect) for group, effect in effects
                    if not any(candidate.startswith(("dx:", "dy:", "wheel:"))
                               for candidate in group)]
        if discrete:
            group, effect = discrete[0]
            name, names = _label(effect, language), _names(group, language)
            chunks.append(f"Действие «{name}» выполняется через ввод «{names}»." if ru else
                          f"The action '{name}' uses input {names}.")
            chunks.append(_aliases(group, effect, language))
            remaining = [(other, other_effect) for other, other_effect in effects if other != group]
            for position, (other, other_effect) in enumerate(remaining):
                chunks.extend(direct(other, other_effect, position))
            chunks.append(_BRIDGE[language])
            mode = labels[group[0]]["mode"]
            if mode == "held":
                rule = (f"Правило действия «{name}»: оно повторяется, пока его ввод зажат; "
                        "отпускание прекращает повторы." if ru else
                        f"Rule for '{name}': it repeats while its input is held down; "
                        "releasing the input stops repetition.")
            elif mode == "press":
                rule = (f"Правило действия «{name}»: требуется одно короткое нажатие; "
                        "после каждого выполнения отпускай его ввод перед следующим выполнением."
                        if ru else f"Rule for '{name}': use one brief press; after each execution "
                        "release its input before the next execution.")
            else:
                raise ValueError("supported discrete action requires held or press timing")
            chunks.append(rule)
            info.update(named_input_text=names, timing_rule_text=rule)
        else:
            for position, (group, effect) in enumerate(effects):
                chunks.extend(direct(group, effect, position))
    else:
        for position, (group, effect) in enumerate(effects):
            chunks.extend(direct(group, effect, position))
    chunks.append(_distinct(effects, language))
    chunks.extend(_statuses(row, language))
    chunks.append(REMAINDER_FORMS[row["specification"]["remainder"]][language][rng.randrange(8)])
    result = dict(row)
    result["text"] = " ".join(chunk for chunk in chunks if chunk)
    result["template_family"] = FAMILIES[frame]
    result["grammar_families"] = [FAMILIES[frame]]
    return result, info


def prepare_structured_control_dataset(source, output, *, seed=20261028):
    """Replace only ordinary TRAIN views in an immutable V11 preparation."""
    from carryrl.rewards.control_text_training import read_control_dataset

    source, output = Path(source), Path(output)
    manifest, records = read_control_dataset(source)
    if manifest.get("version") != "carryrl.control_text_data.v11":
        raise ValueError("structured preparation requires an immutable V11 source")
    payload = (source / "records.jsonl").read_bytes()
    source_manifest = (source / "manifest.json").read_bytes()
    if (hashlib.sha256(payload).hexdigest() != manifest["records_sha256"]
            or json.loads(source_manifest) != manifest):
        raise ValueError("V11 source changed during structured preparation")
    lines = payload.splitlines(keepends=True)
    if len(lines) != len(records):
        raise ValueError("source JSONL must have one record per line")
    output_lines, repaired, counts, changed_groups = [], [], Counter(), set()
    for row, line in zip(records, lines, strict=True):
        current = row
        if row["split"] == "train" and row["template_family"].startswith("v8-binding-composition-"):
            current, _ = render_structured_view(row, seed=seed)
            line = (json.dumps(current, ensure_ascii=False, sort_keys=True) + "\n").encode()
            counts[f"{row['language']}|{current['template_family']}"] += 1
            changed_groups.add(row["group_id"])
        output_lines.append(line)
        repaired.append(current)
    output_payload = b"".join(output_lines)
    summary = wording_manifest(repaired)
    witness_coverage, witness_groups = Counter(), defaultdict(set)
    for row in repaired:
        groups = _known_groups(row)
        if any(family.startswith("v9-relation-witness-") for family in row["grammar_families"]):
            witness_coverage[f"{row['split']}|witness_records"] += 1
            witness_coverage[f"{row['split']}|multiple_known_group_records"] += len(groups) > 1
            witness_coverage[f"{row['split']}|within_group_alias_records"] += any(
                len(g) > 1 for g in groups
            )
            witness_groups[row["split"]].add(row["group_id"])
        elif groups:
            witness_coverage[f"{row['split']}|known_records_without_bracket_witness"] += 1
        else:
            witness_coverage[f"{row['split']}|no_known_group_records"] += 1
    relation_metadata = dict(manifest["relation_grounding_repair"])
    relation_metadata["coverage_records"] = dict(witness_coverage)
    relation_metadata["coverage_scope"] = (
        "Current V12 rows; concrete TRAIN effects have no bracket witness."
    )
    metadata = {
        "version": STRUCTURED_VERSION, "seed": seed,
        "base_dataset": str(source.resolve()),
        "base_records_sha256": hashlib.sha256(payload).hexdigest(),
        "base_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
        "replacement_records": sum(counts.values()),
        "replacement_existing_groups": len(changed_groups),
        "replacement_counts_by_language_frame": dict(counts),
        "new_independent_semantic_groups": 0, "added_records": 0,
        "effect_assignment": (
            "Task-local concrete effects fixed per semantic group or matched context."
        ),
        "unchanged_record_bytes_preserved": True, "heldout_record_bytes_preserved": True,
        "natural_and_transfer_views_preserved": True,
        "labels_ids_splits_pair_metadata_unchanged": True,
        "heldout_records_sha256_by_split": {
            split: hashlib.sha256(b"".join(line for row, line in zip(records, lines, strict=True)
                                           if row["split"] == split)).hexdigest()
            for split in ("dev", "calibration", "sealed_test")
        },
        "renderer_reads_runtime_instructions_or_predictions": False,
        "effectiveness_measured": False,
    }
    updated = {
        **manifest, **summary, "version": CONTROL_TEXT_DATA_VERSION, "wording_version": 12,
        "semantic_generator_version": CONTROL_TEXT_DATA_VERSION,
        "scenario_kinds": manifest["scenario_kinds"],
        "records_sha256": hashlib.sha256(output_payload).hexdigest(),
        "records_bytes": len(output_payload),
        "template_family_split_ownership": {**manifest["template_family_split_ownership"],
                                             **dict.fromkeys(FAMILIES, "train")},
        "relation_grounding_repair": relation_metadata,
        "relation_witness_group_counts": {
            split: len(witness_groups[split])
            for split in ("train", "dev", "calibration", "sealed_test")
        },
        "structured_augmentation": metadata,
        "status": "prepared", "updates": 0, "model_loaded": False,
        "readiness": False, "ppo_eligible": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "records.jsonl").write_bytes(output_payload)
    (output / "manifest.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return updated
