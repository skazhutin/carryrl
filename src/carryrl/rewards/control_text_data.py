"""Deterministic offline text-only control supervision, with grouped sealed splits.

These synthetic public control specifications are annotation sources only. Runtime
inference must consume the text and may never recover labels through this module.
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from carryrl.agent.actions import KEY_VOCAB, MOUSE_BUTTONS

CONTROL_TEXT_DATA_VERSION = "carryrl.control_text_data.v3"
CONTROL_CANDIDATES = (
    *(f"key:{key}" for key in KEY_VOCAB),
    *(f"mouse:{button}" for button in MOUSE_BUTTONS),
    "dx:-",
    "dx:+",
    "dy:-",
    "dy:+",
    "wheel:-",
    "wheel:+",
)
SPLIT_WEIGHTS = (("train", 60), ("dev", 15), ("calibration", 10), ("sealed_test", 15))
SCENARIO_KINDS = (
    "remapping",
    "unknown",
    "incomplete",
    "negation",
    "conflict",
    "aliases",
    "held",
    "press",
    "dense",
    "mixed_mouse",
    "abbreviations",
    "conditional",
    "directional",
)
# Entire syntactic families belong to one split before any wording is rendered.
TEMPLATES = {
    "train": (
        ("ru", "Правила управления. {body}"),
        ("ru", "Описание ввода: {body}"),
        ("en", "Control instructions. {body}"),
        ("en", "Input description: {body}"),
    ),
    "dev": (
        ("ru", "Перед началом прочтите памятку:\n{body}\nКонец памятки."),
        ("ru", "Инструкция оператора\n{body}\nСледуйте этим указаниям."),
        ("en", "Read this reference before starting:\n{body}\nEnd of reference."),
        ("en", "Operator guide\n{body}\nFollow these directions."),
    ),
    "calibration": (
        ("ru", "Вопрос: какие команды доступны? Ответ: {body}"),
        ("ru", "Справка отвечает на вопрос об управлении следующим образом: {body}"),
        ("en", "Question: which commands are available? Answer: {body}"),
        ("en", "The help entry answers the controls question as follows: {body}"),
    ),
    "sealed_test": (
        ("ru", "[Начало объявления]\n{body}\n[Конец объявления]"),
        ("ru", "Сообщение на экране «Управление»: «{body}»"),
        ("en", "[Notice begins]\n{body}\n[Notice ends]"),
        ("en", "On-screen message entitled Controls: ‘{body}’"),
    ),
}


def _name(candidate, language):
    kind, value = candidate.split(":")
    if kind == "key":
        return f"клавиша {value.upper()}" if language == "ru" else f"the {value.upper()} key"
    if kind == "mouse":
        names = {"left": "левая", "right": "правая", "middle": "средняя"}
        return f"{names[value]} кнопка мыши" if language == "ru" else f"the {value} mouse button"
    names = {
        "dx:-": ("движение мыши влево", "moving the mouse left"),
        "dx:+": ("движение мыши вправо", "moving the mouse right"),
        "dy:-": ("движение мыши вверх", "moving the mouse up"),
        "dy:+": ("движение мыши вниз", "moving the mouse down"),
        "wheel:-": ("прокрутка колеса вниз", "scrolling the wheel down"),
        "wheel:+": ("прокрутка колеса вверх", "scrolling the wheel up"),
    }
    return names[candidate][0 if language == "ru" else 1]


def _render_dense(mapping, kind, language, variant, split):
    """Compact multi-operation references; labels are generated before wording."""
    ru = language == "ru"
    labels = {
        candidate: {
            "candidate": candidate,
            "status": "unsupported",
            "mode": "unknown",
            "alias_group": 0,
        }
        for candidate in CONTROL_CANDIDATES
    }
    # Several independent alias pairs plus singletons, up to thirteen inputs.
    supported = mapping[:-2]
    chunks, index, group = [], 0, 1
    while index < len(supported):
        size = 2 if group <= 4 and index + 1 < len(supported) else 1
        candidates = supported[index : index + size]
        mode = "press" if group % 3 == 0 and not (kind == "directional" and group <= 4) else "held"
        conditional = kind == "conditional" and group <= 2
        for candidate in candidates:
            if conditional:
                labels[candidate].update(status="unknown")
            else:
                labels[candidate].update(status="supported", mode=mode, alias_group=group)
        names = "/".join(_name(candidate, language) for candidate in candidates)
        timing = (
            ("удерживать" if mode == "held" else "нажать и отпустить")
            if ru
            else ("hold" if mode == "held" else "tap and release")
        )
        if conditional:
            chunks.append(
                f"Только при активной панели: {names}; без вида панели доступность "
                "и режим неизвестны."
                if ru
                else f"Only with the active panel: {names}; without the panel view, "
                "availability and mode are unknown."
            )
        else:
            if split == "train":
                chunks.append(f"{names}: {timing}, {'действие' if ru else 'operation'} {group}.")
            elif split == "dev":
                chunks.append(
                    f"{'Действие' if ru else 'Operation'} {group}: "
                    f"{'ввод' if ru else 'input'}={names}; "
                    f"{'режим' if ru else 'mode'}={timing}."
                )
            elif split == "calibration":
                chunks.append(
                    f"{'Чем выполнить' if ru else 'Which inputs perform'} {group}? "
                    f"{names}. {'Как' if ru else 'How'}? {timing}."
                )
            else:
                chunks.append(
                    f"[{group} → {names}] ({timing}); {'разрешено' if ru else 'enabled'}."
                )
        index += size
        group += 1
    if kind == "directional":
        chunks = chunks[4:]
        chunks.insert(
            0,
            "Движение — WASD или стрелки: A/Left влево, D/Right вправо, "
            "W/Up вверх, S/Down вниз; удерживайте направление."
            if ru
            else "Move with WASD or arrow keys: A/Left left, D/Right right, "
            "W/Up up, S/Down down; hold a direction.",
        )
    if kind == "abbreviations":
        # The literal compact spelling appears alongside its explicit expansion.
        chunks.insert(
            0,
            "ЛКМ=левая кнопка мыши; ПКМ=правая кнопка мыши; "
            "WASD=W/A/S/D; стрелки=Left/Right/Up/Down."
            if ru
            else "LMB=left mouse button; RMB=right mouse button; "
            "WASD=W/A/S/D; arrows=Left/Right/Up/Down.",
        )
    if variant:
        chunks.reverse()
    chunks.append(
        "В каждой строке варианты одной команды равнозначны. Полный список; "
        "остальной ввод не поддерживается."
        if ru
        else "Alternatives in each line perform the same operation. Complete list; "
        "all remaining input is unsupported."
    )
    return " ".join(chunks), list(labels.values())


def _render(mapping, kind, language, variant, split):
    if kind in {"dense", "mixed_mouse", "abbreviations", "conditional", "directional"}:
        return _render_dense(mapping, kind, language, variant, split)
    first, second, third, fourth = (_name(candidate, language) for candidate in mapping)
    ru = language == "ru"
    held = kind != "press"
    mode = "held" if held else "press"
    labels = {
        candidate: {
            "candidate": candidate,
            "status": "unknown",
            "mode": "unknown",
            "alias_group": 0,
        }
        for candidate in CONTROL_CANDIDATES
    }
    if kind == "unknown":
        body = (
            f"На экране перечислены {first}, {second}, {third}, {fourth}, но назначения, "
            "доступность и режимы всех команд пока неизвестны."
            if ru
            else f"The screen lists {first}, {second}, {third}, {fourth}, but bindings, "
            "availability and modes of all commands remain unknown."
        )
        return body, list(labels.values())
    verb = ("Удерживайте" if variant == 0 else "Держите нажатым") if held else "Нажмите один раз"
    if not ru:
        verb = ("Hold" if variant == 0 else "Keep active") if held else "Press once"
    # The annotation explicitly states duration even for relative input candidates.
    duration = (" в каждом такте" if ru else " on each interval") if held else ""

    def binding(name, action, binding_mode):
        command = verb if binding_mode == mode else ("Нажмите один раз" if ru else "Press once")
        timing = duration if binding_mode == mode else ""
        description = (
            ("удержание в каждом такте" if binding_mode == "held" else "одиночное нажатие")
            if ru
            else ("held on each interval" if binding_mode == "held" else "one press")
        )
        if split == "train":
            return (
                f"{command} {name}{timing} для действия {action}. "
                if ru
                else f"{command} {name}{timing} for operation {action}. "
            )
        if split == "dev":
            return (
                f"Действие {action} выполняется через {name}; режим: {description}. "
                if ru
                else f"Operation {action} uses {name}; mode: {description}. "
            )
        if split == "calibration":
            return (
                f"Чем выполнить действие {action}? {name}; как? {description}. "
                if ru
                else f"How is operation {action} performed? {name}; timing? {description}. "
            )
        return (
            f"Назначение [{action} → {name}], разрешено; {description}. "
            if ru
            else f"Binding [{action} → {name}], enabled; {description}. "
        )

    body = binding(first, "A", mode)
    labels[mapping[0]].update(status="supported", mode=mode, alias_group=1)
    if kind == "aliases":
        body += (
            f"{second} — равнозначная замена с тем же режимом. "
            if ru
            else f"{second} is an equivalent alternative with the same mode. "
        )
        labels[mapping[1]].update(status="supported", mode=mode, alias_group=1)
    elif kind == "conflict":
        body += (
            f"Справка противоречива: {second} разрешена и одновременно запрещена; "
            "её статус и режим неизвестны. "
            if ru
            else f"The reference conflicts: {second} is both enabled and forbidden; "
            "its status and mode are unknown. "
        )
    elif kind == "negation":
        body += (
            f"Не используйте {second}: этот ввод не поддерживается. "
            if ru
            else f"Do not use {second}: this input is unsupported. "
        )
        labels[mapping[1]].update(status="unsupported")
    else:
        body += binding(second, "B", "press")
        labels[mapping[1]].update(status="supported", mode="press", alias_group=2)
    if kind in {"conflict", "negation"}:
        body += binding(third, "C", "press")
        labels[mapping[2]].update(status="supported", mode="press", alias_group=3)
    if kind in {"incomplete", "conflict"}:
        body += (
            f"Ввод {fourth} явно запрещён и не поддерживается. "
            if ru
            else f"Input {fourth} is explicitly forbidden and unsupported. "
        )
        labels[mapping[3]].update(status="unsupported")
    if kind == "remapping":
        body += (
            f"Прежнее назначение {third} отменено. "
            if ru
            else f"The previous binding for {third} has been removed. "
        )
    if kind in {"incomplete", "conflict"}:
        body += (
            "Про остальные команды сведений нет."
            if ru
            else "No information is available about the remaining commands."
        )
    else:
        body += (
            "Все остальные команды не поддерживаются."
            if ru
            else "All other commands are unsupported."
        )
        for label in labels.values():
            if label["status"] == "unknown":
                label["status"] = "unsupported"
    return body, list(labels.values())


def semantic_mapping_identity(labels, declared_candidates=()):
    """Canonical effective supervision, independent of wording and unused bindings.

    Fully unknown examples have no effective binding. Only for that null case,
    partition the named candidate set so unknown examples can occur in each split.
    """
    memberships = {}
    for label in labels:
        if label["alias_group"] > 0:
            memberships.setdefault(label["alias_group"], []).append(label["candidate"])
    canonical = {
        group: index + 1
        for index, (_, group) in enumerate(
            sorted((tuple(sorted(members)), group) for group, members in memberships.items())
        )
    }
    entries = sorted(
        (label["candidate"], label["status"], label["mode"], canonical.get(label["alias_group"], 0))
        for label in labels
    )
    null_mapping = all(label["status"] == "unknown" for label in labels)
    identity = {"labels": entries}
    if null_mapping:
        identity["null_named_candidates"] = sorted(declared_candidates)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _semantic_split(identity, seed):
    bucket = int(hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest(), 16) % 100
    boundary = 0
    for split, weight in SPLIT_WEIGHTS:
        boundary += weight
        if bucket < boundary:
            return split
    raise AssertionError("split weights must sum to 100")


def generate_control_text_records(*, seed=20261020, scenario_groups=4000):
    """Build four variants per mapping after assigning mapping and family splits.

    Every ordered four-candidate mapping is unique, including between splits.
    Labels are supervision; only ``text`` is a model input. Group identifiers,
    mapping identifiers, language and template metadata must not be encoded.
    """
    if type(scenario_groups) is not int or scenario_groups < 100 or scenario_groups % 100:
        raise ValueError("scenario_groups must be a positive multiple of 100")
    if scenario_groups > 40000:
        raise ValueError("too many unique scenario mappings")
    rng = random.Random(seed)
    seen, seen_semantics = set(), set()
    records, offset = [], 0
    for split, weight in SPLIT_WEIGHTS:
        for local in range(scenario_groups * weight // 100):
            index = offset + local
            kind = SCENARIO_KINDS[local % len(SCENARIO_KINDS)]
            # Assign effective mappings by hash before producing any paraphrase.
            # Effective bindings are unique globally, including within each split.
            for _ in range(100000):
                if kind in {"dense", "mixed_mouse", "abbreviations", "conditional", "directional"}:
                    count = rng.randint(8, 13)
                    if kind == "directional":
                        fixed = (
                            "key:a",
                            "key:left",
                            "key:d",
                            "key:right",
                            "key:w",
                            "key:up",
                            "key:s",
                            "key:down",
                        )
                        count = rng.randint(10, 13)
                        extra = rng.sample(
                            [value for value in CONTROL_CANDIDATES if value not in fixed], count - 6
                        )
                        mapping = (*fixed, *extra)
                    else:
                        mapping = tuple(rng.sample(CONTROL_CANDIDATES, count + 2))
                    if kind == "mixed_mouse" and not any(
                        candidate.startswith("mouse:") for candidate in mapping[:-2]
                    ):
                        continue
                else:
                    mapping = tuple(rng.sample(CONTROL_CANDIDATES, 4))
                if mapping in seen:
                    continue
                _, labels = _render(mapping, kind, "en", 0, split)
                semantic_id = semantic_mapping_identity(labels, mapping)
                if (
                    semantic_id not in seen_semantics
                    and _semantic_split(semantic_id, seed) == split
                ):
                    seen.add(mapping)
                    seen_semantics.add(semantic_id)
                    break
            else:
                raise ValueError("requested group budget exhausts assigned semantic mappings")
            mapping_id = hashlib.sha256(json.dumps(mapping).encode()).hexdigest()
            for variant, (language, template) in enumerate(TEMPLATES[split]):
                body, labels = _render(mapping, kind, language, variant % 2, split)
                records.append(
                    {
                        "group_id": f"scenario-{index:05d}",
                        "split": split,
                        "mapping_id": mapping_id,
                        "semantic_mapping_id": semantic_id,
                        "declared_candidates": list(mapping),
                        "template_family": f"{split}-{variant}",
                        "scenario_kind": kind,
                        "language": language,
                        "text": template.format(body=body),
                        "labels": labels,
                    }
                )
        offset += scenario_groups * weight // 100
    return records


def prepare_control_text_dataset(output, *, seed=20261020, scenario_groups=4000):
    """Write preparation artifacts only; never instantiate or fit any model."""
    output = Path(output)
    records = generate_control_text_records(seed=seed, scenario_groups=scenario_groups)
    output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with (output / "records.jsonl").open("wb") as stream:
        for row in records:
            line = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            stream.write(line)
            digest.update(line)
    manifest = {
        "version": CONTROL_TEXT_DATA_VERSION,
        "seed": seed,
        "scenario_groups": scenario_groups,
        "records": len(records),
        "variants_per_group": 4,
        "candidates": list(CONTROL_CANDIDATES),
        "split_records": dict(Counter(row["split"] for row in records)),
        "split_groups": {split: scenario_groups * weight // 100 for split, weight in SPLIT_WEIGHTS},
        "scenario_kinds": list(SCENARIO_KINDS),
        "records_sha256": digest.hexdigest(),
        "split_before_paraphrase": True,
        "mapping_disjoint": True,
        "template_families_disjoint": True,
        "model_input_fields": ["text"],
        "mapping_identity_definition": "canonical candidate/status/mode/alias assignment",
        "semantic_mapping_disjoint": True,
        "semantic_mapping_unique_per_group": True,
        "null_mapping_exception": "All-unknown mappings partition the sorted named candidate set.",
        "annotation_vectors_disjoint": False,
        "statuses": ["supported", "unsupported", "unknown"],
        "modes": ["held", "press", "unknown"],
        "alias_group_semantics": "local equivalence classes; 0 means unspecified",
        "calibration_eligible_sign_groups": {
            status: len(
                {
                    row["group_id"]
                    for row in records
                    if row["split"] == "calibration"
                    and any(label["status"] == status for label in row["labels"])
                }
            )
            for status in ("supported", "unsupported")
        },
        "maximum_supported_candidates": max(
            sum(label["status"] == "supported" for label in row["labels"]) for row in records
        ),
        "records_bytes": (output / "records.jsonl").stat().st_size,
        "status": "prepared",
        "updates": 0,
        "model_loaded": False,
        "readiness": False,
        "ppo_eligible": False,
        "limitations": [
            "Synthetic offline specifications only; no learned quality measured.",
            "Sealed test is reserved for evaluation after model freeze.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
