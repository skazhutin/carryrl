"""V11 targeted TRAIN wording replacements, without new semantic groups.

This preparation consumes structural TRAIN specifications only when selecting
and rendering replacements. All other JSONL lines, especially evaluation lines,
retain their exact bytes. The renderer reads no model outputs, diagnostics,
or public runtime instructions.
"""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from carryrl.rewards.control_text_data_natural import _binding, _names, _relation_header
from carryrl.rewards.control_text_data_v5 import REMAINDER_FORMS
from carryrl.rewards.control_text_data_v8 import _name, wording_manifest
from carryrl.rewards.control_text_data_v9 import _known_groups

CONTROL_TEXT_DATA_VERSION = "carryrl.control_text_data.v11"
TRANSFER_VERSION = "carryrl.control_text_transfer.v1"
_AXES = ("dx:-", "dx:+", "dy:-", "dy:+")
_BRIDGE = {
    "en": (
        "The patterned field contains several pale markers and a striped boundary. "
        "A small counter indicates progress through the arrangement, while the decorative "
        "background alternates between two neutral shades. Return to the patterned area in "
        "the description and enter the pale region on the diagram; these are narrative "
        "landmarks, while the surrounding pattern only indicates how the scene is arranged."
    ),
    "ru": (
        "На узорном поле видны несколько светлых меток и полосатая граница. "
        "Небольшой счётчик показывает ход расстановки, а декоративный фон чередует "
        "два нейтральных оттенка. Возле границы расположены разные по размеру фигуры "
        "с ровной или зернистой поверхностью. Краткая подпись описывает только внешний "
        "вид этих ориентиров; рисунок и оттенки помогают различать участки сцены, "
        "но сами по себе не меняют назначение перечисленных функций."
    ),
}


def _labels(row):
    return {label["candidate"]: label for label in row["labels"]}


def plan_transfer_views(records, *, seed=20261027, timing_pairs=24, conditionals=64,
                        mouse_per_mode=32):
    """Choose disjoint existing groups; one natural view per admitted group."""
    for value in (timing_pairs, conditionals, mouse_per_mode):
        if type(value) is not int or value < 0:
            raise ValueError("transfer quotas must be nonnegative integers")
    groups = {}
    for row in records:
        if row["split"] == "train":
            groups.setdefault(row["group_id"], row)
    rng, plan, selected = random.Random(seed), [], set()

    def add(row, kind, language, **details):
        gid = row["group_id"]
        if gid in selected:
            raise ValueError("transfer strata overlap")
        selected.add(gid)
        size = row["group_variants"]
        if size not in {6, 8}:
            raise ValueError("transfer repair requires six- or eight-view groups")
        variant = (2 if language == "ru" else 5) if size == 6 else (
            3 if language == "ru" else 7
        )
        plan.append({"group_id": gid, "variant_index": variant, "language": language,
                     "kind": kind, **details})

    axis_rows = []
    for row in groups.values():
        labels = _labels(row)
        if all(labels[candidate]["status"] == "supported" for candidate in _AXES):
            axis_groups = [group for group in _known_groups(row) if set(group) & set(_AXES)]
            if len(axis_groups) == 4 and all(len(group) == 1 for group in axis_groups):
                axis_rows.append(row)
    rng.shuffle(axis_rows)
    for index, row in enumerate(axis_rows):
        add(row, "aggregate_axes", "ru" if index % 2 == 0 else "en")

    pairs = defaultdict(list)
    for row in groups.values():
        if row.get("contrastive_kind") == "adjacent_timing":
            pairs[row["contrastive_pair_id"]].append(row)
    pair_ids = sorted(pairs)
    rng.shuffle(pair_ids)
    if len(pair_ids) < timing_pairs:
        raise ValueError("insufficient paired timing TRAIN groups")
    for index, pair_id in enumerate(pair_ids[:timing_pairs]):
        members = pairs[pair_id]
        if len(members) != 2:
            raise ValueError("timing contrast must have two effective TRAIN groups")
        for row in members:
            target = _known_groups(row)[0][0]
            add(row, "deferred_timing", "ru" if index % 2 == 0 else "en", target=target)

    for mode, other_mode in (("press", "held"), ("held", "press")):
        eligible = []
        for row in groups.values():
            if row["group_id"] in selected:
                continue
            labels = _labels(row)
            known = _known_groups(row)
            buttons = [group[0] for group in known if len(group) == 1
                       and group[0].startswith("mouse:") and labels[group[0]]["mode"] == mode]
            keys = [group[0] for group in known if len(group) == 1
                    and group[0].startswith("key:") and labels[group[0]]["mode"] == other_mode]
            if buttons and keys:
                eligible.append((row, buttons[0], keys[0]))
        rng.shuffle(eligible)
        if len(eligible) < mouse_per_mode:
            raise ValueError(f"insufficient mouse {mode} / key {other_mode} TRAIN groups")
        for index, (row, button, key) in enumerate(eligible[:mouse_per_mode]):
            add(row, f"deferred_mouse_{mode}", "ru" if index % 2 == 0 else "en",
                target=button, intervening=key)

    available = defaultdict(list)
    for row in groups.values():
        if row["group_id"] in selected or row["specification"]["kind"] != "conditional":
            continue
        labels = _labels(row)
        for candidate in row["specification"]["uncertain"]:
            if (candidate.startswith(("key:", "mouse:"))
                    and labels[candidate]["status"] == "unknown"):
                available[candidate].append(row)
                break
    controls = sorted(available)
    rng.shuffle(controls)
    for rows in available.values():
        rng.shuffle(rows)
    admitted = 0
    while admitted < conditionals:
        progressed = False
        for control in controls:
            if admitted == conditionals:
                break
            if not available[control]:
                continue
            row = available[control].pop()
            add(row, "conditional_mechanic", "ru" if admitted % 2 == 0 else "en",
                target=control, hypothetical_mode="held" if (admitted // 2) % 2 == 0 else "press")
            admitted += 1
            progressed = True
        if not progressed:
            raise ValueError("insufficient conditional unknown discrete TRAIN groups")
    return plan


def _known_clauses(row, groups, language, *, offset=0):
    labels, ru, chunks = _labels(row), language == "ru", []
    for index, group in enumerate(groups, offset):
        mode = labels[group[0]]["mode"]
        label = f"Отдельная функция {index + 1}" if ru else f"Separate function {index + 1}"
        chunks.append(label + ": " + _binding(group, mode, language, index))
        if len(group) > 1:
            chunks.append("Варианты этой функции равнозначны." if ru else
                          "The choices for this function are interchangeable.")
    return chunks


def _status_clauses(row, language, *, exclude=()):
    labels, chunks, ru = _labels(row), [], language == "ru"
    unknown = set(row["specification"]["uncertain"])
    unknown.update(candidate for operation in row["specification"]["operations"]
                   for candidate in operation["candidates"]
                   if labels[candidate]["status"] == "unknown")
    for candidate in sorted(unknown - set(exclude)):
        name = _name(candidate, language)
        chunks.append(f"Ввод «{name}»: доступность или исполнимый режим неизвестны." if ru else
                      f"Input {name}: availability or executable timing is unknown.")
    for candidate in row["specification"]["forbidden"]:
        name = _name(candidate, language)
        chunks.append(f"Ввод «{name}» запрещён." if ru else f"Input {name} is unsupported.")
    return chunks


def render_transfer_view(row, entry):
    """Render one existing semantic view and return inspectable timing spans."""
    if row["split"] != "train" or row["language"] != entry["language"]:
        raise ValueError("transfer rendering requires the selected TRAIN language")
    if not row["template_family"].startswith("v9-natural-"):
        raise ValueError("transfer rendering replaces only an existing natural view")
    groups, labels = _known_groups(row), _labels(row)
    language, ru, kind = row["language"], row["language"] == "ru", entry["kind"]
    chunks, info = [_relation_header(language, 0)] if groups else [], {}
    if kind.startswith("deferred_"):
        target = entry["target"]
        group = next(group for group in groups if target in group)
        remaining = [other for other in groups if other != group]
        if not remaining:
            raise ValueError("deferred timing needs an intervening independent binding")
        if entry.get("intervening"):
            remaining.sort(key=lambda values: entry["intervening"] not in values)
        names = _names(group, language)
        chunks.append(f"Первая функция использует ввод «{names}»." if ru else
                      f"The first function uses {names}.")
        competing_cue = None
        if kind.startswith("deferred_mouse_"):
            competitor = remaining[0]
            if (not target.startswith("mouse:") or len(competitor) != 1
                    or not competitor[0].startswith("key:")
                    or labels[competitor[0]]["mode"] == labels[target]["mode"]):
                raise ValueError("deferred mouse challenge requires an opposite-timing key")
            competing_cue = _binding(competitor, labels[competitor[0]]["mode"], language, 0)
            chunks.append(("Вторая: " if ru else "Second: ") + competing_cue)
            chunks.append(_BRIDGE[language])
            chunks.extend(_known_clauses(row, remaining[1:], language, offset=2))
        else:
            chunks.append(_BRIDGE[language])
            chunks.extend(_known_clauses(row, remaining, language, offset=1))
        mode = labels[target]["mode"]
        if mode == "press":
            rule = ("Правило первой функции: каждое выполнение требует одного краткого нажатия; "
                    "отпускай её ввод перед следующим выполнением." if ru else
                    "Rule for the first function: each execution requires a single brief press; "
                    "release its input before the next execution.")
        elif mode == "held":
            rule = ("Правило первой функции: действие повторяется всё время, пока её ввод зажат; "
                    "отпускание прекращает повторы." if ru else
                    "Rule for the first function: the effect repeats "
                    "while its input stays held down; "
                    "releasing the input stops repetition.")
        else:
            raise ValueError("deferred target requires known discrete timing")
        chunks.append(rule)
        if len(group) > 1:
            chunks.append("Варианты первой функции равнозначны." if ru else
                          "The choices for the first function are interchangeable.")
        info = {"named_input_text": names, "timing_rule_text": rule, "target_mode": mode}
        if competing_cue is not None:
            info["competing_cue_text"] = competing_cue
    elif kind == "conditional_mechanic":
        chunks.extend(_known_clauses(row, groups, language))
        target, mode = entry["target"], entry["hypothetical_mode"]
        if labels[target]["status"] != "unknown" or labels[target]["mode"] != "unknown":
            raise ValueError("conditional target must retain effective unknown status and timing")
        hypothetical = _binding((target,), mode, language, 0)
        if ru:
            chunks.append("Условное правило действует только при выполнении неизвестного условия: "
                          + hypothetical + " Выполнено ли условие, неизвестно; доступность и "
                          "исполнимый режим этого ввода сейчас неизвестны.")
        else:
            chunks.append("The following rule applies only if an unspecified "
                          "necessary condition holds: "
                          + hypothetical + " Whether the condition holds is unknown; availability "
                          "and executable timing of this input are currently unknown.")
    elif kind == "aggregate_axes":
        if not all(labels[candidate]["status"] == "supported" and
                   labels[candidate]["mode"] == "unknown" for candidate in _AXES):
            raise ValueError("aggregate axes require four supported relative inputs")
        if len({labels[candidate]["alias_group"] for candidate in _AXES}) != 4:
            raise ValueError("aggregate signs must have four independent effects")
        chunks.append(
            "Относительное движение доступно: движение мыши меняет горизонтальную и вертикальную "
            "координаты. Два знака каждой координаты дают четыре разных направления действия; "
            "эти эффекты не взаимозаменяемы. Режим этого относительного ввода не указан." if ru else
            "Relative movement is available: mouse movement changes both horizontal and vertical "
            "coordinates. The two signs of each coordinate produce four different "
            "directional effects; "
            "these effects are not interchangeable. Timing for this relative input is unspecified."
        )
        chunks.extend(_known_clauses(row, [group for group in groups
                                         if not set(group) & set(_AXES)], language))
    else:
        raise ValueError("unknown transfer stratum")
    chunks.extend(_status_clauses(row, language, exclude=(entry["target"],) if (
        kind == "conditional_mechanic"
    ) else ()))
    chunks.append(REMAINDER_FORMS[row["specification"]["remainder"]][language][0])
    result = dict(row)
    result["text"] = " ".join(chunks)
    result["template_family"] = f"v11-transfer-{kind}"
    result["grammar_families"] = [f"v11-transfer-{kind}"]
    return result, info


def prepare_transfer_control_dataset(base_v10, output, *, seed=20261027, timing_pairs=24,
                                     conditionals=64, mouse_per_mode=32):
    """Write targeted V11 TRAIN views; counts and all effective groups stay fixed."""
    from carryrl.rewards.control_text_training import read_control_dataset

    base, output = Path(base_v10), Path(output)
    manifest, records = read_control_dataset(base)
    if manifest.get("version") != "carryrl.control_text_data.v10":
        raise ValueError("transfer preparation requires an immutable V10 base")
    payload = (base / "records.jsonl").read_bytes()
    source_manifest = (base / "manifest.json").read_bytes()
    if (hashlib.sha256(payload).hexdigest() != manifest["records_sha256"]
            or json.loads(source_manifest) != manifest):
        raise ValueError("V10 preparation changed during transfer preparation")
    lines = payload.splitlines(keepends=True)
    if len(lines) != len(records):
        raise ValueError("source JSONL must contain one record per line")
    plan = plan_transfer_views(records, seed=seed, timing_pairs=timing_pairs,
                               conditionals=conditionals, mouse_per_mode=mouse_per_mode)
    targets = {(entry["group_id"], entry["variant_index"]): entry for entry in plan}
    output_lines, counts, admitted, families = [], Counter(), set(), set()
    repaired = []
    for row, line in zip(records, lines, strict=True):
        current = row
        key = (row["group_id"], row["variant_index"])
        if row["split"] == "train" and key in targets:
            current, _ = render_transfer_view(row, targets[key])
            line = (json.dumps(current, ensure_ascii=False, sort_keys=True) + "\n").encode()
            counts[f"{targets[key]['kind']}|{row['language']}"] += 1
            admitted.add(key)
            families.add(current["template_family"])
        output_lines.append(line)
        repaired.append(current)
    if admitted != set(targets):
        raise ValueError("selected transfer views are missing from the base corpus")
    output_payload = b"".join(output_lines)
    metadata = {
        "version": TRANSFER_VERSION, "seed": seed,
        "base_dataset": str(base.resolve()),
        "base_records_sha256": hashlib.sha256(payload).hexdigest(),
        "base_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
        "replacement_records": len(plan), "replacement_existing_semantic_groups": len(plan),
        "new_independent_semantic_groups": 0, "added_records": 0,
        "replacement_counts_by_kind_language": dict(counts),
        "selection_plan": plan,
        "unchanged_record_bytes_preserved": True, "heldout_record_bytes_preserved": True,
        "labels_ids_splits_pair_metadata_unchanged": True,
        "heldout_records_sha256_by_split": {
            split: hashlib.sha256(b"".join(line for row, line in zip(records, lines, strict=True)
                                           if row["split"] == split)).hexdigest()
            for split in ("dev", "calibration", "sealed_test")
        },
        "conditional_operations_unchanged": True,
        "runtime_or_model_predictions_used": False, "effectiveness_measured": False,
        "evaluation_wording_protocol_changed": False,
        "inherited_stage_metadata_scope": (
            "Historical preparations; current wording protocol is V11."
        ),
    }
    summary = wording_manifest(repaired)
    updated = {
        **manifest, **summary, "version": CONTROL_TEXT_DATA_VERSION, "wording_version": 11,
        "semantic_generator_version": CONTROL_TEXT_DATA_VERSION,
        "scenario_kinds": manifest["scenario_kinds"],
        "records_sha256": hashlib.sha256(output_payload).hexdigest(),
        "records_bytes": len(output_payload),
        "template_family_split_ownership": {**manifest["template_family_split_ownership"],
                                             **dict.fromkeys(families, "train")},
        "transfer_augmentation": metadata,
        "status": "prepared", "updates": 0, "model_loaded": False,
        "readiness": False, "ppo_eligible": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "records.jsonl").write_bytes(output_payload)
    (output / "manifest.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return updated
