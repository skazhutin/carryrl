"""V9: explicitly verbalize the effect relation of structural control groups.

Existing specifications, labels, identities and splits are immutable. This repair
adds only text and its grammar provenance, including new evaluation wording under
an explicit protocol change. It does not consult model outputs, runtime strings,
or manually selected evaluation examples.
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from carryrl.rewards.control_text_data_v8 import INTERNAL_FAMILIES, _name, wording_manifest

CONTROL_TEXT_DATA_VERSION = "carryrl.control_text_data.v9"
REPAIR_VERSION = "carryrl.control_text_relation_grounding.v1"


def _known_groups(row):
    """Validate spec-defined executable groups, excluding every unknown input."""
    labels = {label["candidate"]: label for label in row["labels"]}
    groups, aliases, candidates = [], set(), set()
    for operation in row["specification"]["operations"]:
        known = tuple(candidate for candidate in operation["candidates"]
                      if labels[candidate]["status"] == "supported")
        if not known:
            continue
        identifiers = {labels[candidate]["alias_group"] for candidate in known}
        if len(identifiers) != 1 or 0 in identifiers or aliases.intersection(identifiers):
            raise ValueError("specification groups disagree with supervised effect relations")
        if candidates.intersection(known):
            raise ValueError("supported candidate occurs in multiple operation groups")
        groups.append(known)
        aliases.update(identifiers)
        candidates.update(known)
    if candidates != {candidate for candidate, label in labels.items()
                      if label["status"] == "supported"}:
        raise ValueError("supported labels are not covered by structural operation groups")
    return groups


def _witness(groups, language, family):
    """Brackets delimit groups; the same grammar covers equality and difference.

    A single [W / left mouse button] and [W]; [left mouse button] use precisely
    the same relational rule. Only the grouping changes. No missing alias is
    silently interpreted as a different effect.
    """
    names = "; ".join("[" + " / ".join(_name(candidate, language) for candidate in group) + "]"
                      for group in groups)
    forms = (
        (
            f"Группы действий: {names}. Внутри каждой скобки варианты равнозначны; "
            "разные скобки задают разные действия и не заменяют друг друга.",
            f"Разбиение действий: {names}. Варианты одной скобки взаимозаменяемы; "
            "между скобками действия различаются и невзаимозаменяемы.",
            f"Действия заданы группами {names}. В каждой группе эффект один; "
            "у разных групп эффекты разные, замены между группами нет.",
            f"Ввод разделён так: {names}. Одна скобка означает одинаковый эффект её вариантов; "
            "разные скобки означают разные, невзаимозаменяемые эффекты.",
            f"Распределение команд: {names}. В пределах скобки команды равнозначны; "
            "в разных скобках команды выполняют разные действия и не заменяют друг друга.",
            f"Группировка эффектов: {names}. Варианты внутри скобки дают один эффект; "
            "за границами своей скобки они не заменяют варианты других, разных эффектов.",
            f"Отдельные действия перечислены в скобках: {names}. Одна скобка объединяет "
            "равнозначные варианты; разные скобки обозначают разные, невзаимозаменяемые действия.",
            f"Схема действий: {names}. Внутри скобки вводы заменяют друг друга; "
            "разные скобки выполняют разные действия, замена между ними невозможна.",
            f"Для эффектов задано разбиение {names}. Варианты одной скобки равнозначны; "
            "разные скобки имеют разные эффекты и не взаимозаменяемы.",
            f"Список групп: {names}. Вводы одной скобки дают одинаковое действие; "
            "вводы разных скобок дают разные действия и не могут заменять друг друга.",
            f"Здесь действия разделены на группы: {names}. Варианты внутри скобки равнозначны; "
            "разные скобки разделяют разные действия без взаимной замены.",
            f"Соответствие эффектов: {names}. Для одной скобки эффект общий; "
            "для разных скобок эффекты различаются и взаимная замена не разрешена.",
            f"Сначала учти группы {names}: внутри скобки варианты равнозначны; "
            "между скобками действия разные и не заменяют друг друга.",
            f"Разные действия разграничены скобками: {names}. В пределах скобки эффект один; "
            "эффекты разных скобок различны и не взаимозаменяемы.",
            f"Правило групп относится к {names}: одна скобка объединяет равнозначные варианты; "
            "разные скобки разделяют разные действия, не заменяющие друг друга.",
            f"Группы {names} читаются так: внутри скобки варианты взаимозаменяемы; "
            "между скобками эффекты разные, замена исключена.",
            f"У разбиения {names} такое правило: одна скобка — один эффект вариантов; "
            "разные скобки — разные эффекты без взаимной замены.",
            f"Для {names} задано правило: внутри скобки варианты дают одинаковый эффект; "
            "в разных скобках эффекты разные и варианты не заменяют друг друга.",
            f"В схеме {names} скобка объединяет равнозначные вводы; "
            "разные скобки задают разные действия без возможности взаимной замены.",
            f"Правило разбиения: {names}. В одной скобке варианты равнозначны; "
            "для разных скобок действия различаются, взаимной замены нет.",
            f"Группы {names} и их смысл: внутри скобки эффект общий; "
            "между скобками эффекты разные и не заменяют друг друга.",
            f"Различай действия по скобкам {names}: вводы внутри скобки равнозначны; "
            "вводы разных скобок имеют разные эффекты и не взаимозаменяемы.",
            f"Разбиение {names} связывает варианты внутри скобки одним эффектом; "
            "разные скобки имеют разные, невзаимозаменяемые эффекты.",
            f"При чтении групп {names} помни: внутри скобки варианты заменяют друг друга; "
            "между скобками действия различны и замена невозможна.",
        ) if language == "ru" else (
            f"Action groups: {names}. Choices inside each bracket are equivalent; "
            "different brackets perform different actions and cannot replace one another.",
            f"Action partition: {names}. Choices in one bracket are interchangeable; "
            "actions across brackets differ and are not interchangeable.",
            f"Actions are grouped as {names}. Each group has one effect; "
            "different groups have different effects, with no substitution across groups.",
            f"Inputs are divided as {names}. One bracket means its choices have the same effect; "
            "different brackets mean different, noninterchangeable effects.",
            f"Command distribution: {names}. Commands within a bracket are equivalent; "
            "commands in different brackets perform different actions "
            "and cannot replace each other.",
            f"Effect grouping: {names}. Choices within a bracket share one effect; "
            "outside their bracket they cannot replace choices with other, different effects.",
            f"Separate actions are listed in brackets: {names}. One bracket combines equivalent "
            "choices; different brackets denote different, noninterchangeable actions.",
            f"Action scheme: {names}. Inputs inside a bracket can replace each other; "
            "different brackets perform different actions, "
            "so cross-bracket replacement is impossible.",
            f"Effects are partitioned as {names}. Choices in one bracket are equivalent; "
            "different brackets have different effects and are not interchangeable.",
            f"Group list: {names}. Inputs in one bracket produce the same action; "
            "inputs in different brackets produce different actions and cannot replace each other.",
            f"Actions here are divided into groups: {names}. "
            "Choices within a bracket are equivalent; "
            "different brackets separate different actions without mutual replacement.",
            f"Effect correspondence: {names}. One bracket has a shared effect; "
            "different brackets have different effects, and mutual replacement is not allowed.",
            f"First consider groups {names}: choices within a bracket are equivalent; "
            "actions across brackets differ and cannot replace one another.",
            f"Different actions are delimited by brackets: {names}. "
            "Within a bracket the effect is one; "
            "effects of different brackets differ and are not interchangeable.",
            f"The group rule applies to {names}: one bracket combines equivalent choices; "
            "different brackets separate different actions that cannot replace one another.",
            f"Read groups {names} as follows: choices inside a bracket are interchangeable; "
            "effects across brackets differ, so substitution is excluded.",
            f"The rule for partition {names} is: one bracket gives one effect for its choices; "
            "different brackets give different effects without mutual replacement.",
            f"For {names}, choices inside a bracket have the same effect; "
            "effects in different brackets differ, and the choices cannot replace one another.",
            f"In scheme {names}, one bracket combines equivalent inputs; "
            "different brackets specify different actions without mutual substitution.",
            f"Partition rule: {names}. Choices inside one bracket are equivalent; "
            "actions for different brackets differ, with no mutual replacement.",
            f"Groups {names} and their meaning: within a bracket the effect is shared; "
            "across brackets effects differ and cannot replace one another.",
            f"Distinguish actions by brackets {names}: inputs inside a bracket are equivalent; "
            "inputs in different brackets have different effects and are not interchangeable.",
            f"Partition {names} links choices inside a bracket with one effect; "
            "different brackets have different, noninterchangeable effects.",
            f"When reading groups {names}, remember: choices inside a bracket replace one another; "
            "actions across brackets differ, so replacement is impossible.",
        )
    )
    return forms[family]


def relation_grounded_records(records):
    """Return copied rows with explicit relations; never change supervision."""
    repaired, counts = [], Counter()
    for row in records:
        match = re.fullmatch(r"v8-binding-composition-(\d{2})", row["template_family"])
        if match is None or int(match[1]) not in INTERNAL_FAMILIES.get(row["split"], ()):
            raise ValueError("base row lacks a valid split-owned V8 binding composition")
        family = int(match[1])
        groups = _known_groups(row)
        result = dict(row)
        if groups:
            result["text"] = row["text"] + " " + _witness(groups, row["language"], family)
            result["grammar_families"] = list(row["grammar_families"]) + [
                f"v9-relation-witness-{family:02d}"
            ]
            counts[f"{row['split']}|witness_records"] += 1
            counts[f"{row['split']}|multiple_known_group_records"] += len(groups) > 1
            counts[f"{row['split']}|within_group_alias_records"] += any(
                len(group) > 1 for group in groups
            )
        else:
            counts[f"{row['split']}|no_known_group_records"] += 1
        repaired.append(result)
    return repaired, dict(counts)


def prepare_relation_grounded_dataset(base_dataset, output):
    """Write a separate V9 wording protocol; all structural splits stay frozen."""
    from carryrl.rewards.control_text_training import read_control_dataset

    base_dataset, output = Path(base_dataset), Path(output)
    base_manifest, records = read_control_dataset(base_dataset)
    if base_manifest.get("version") != "carryrl.control_text_data.v8":
        raise ValueError("relation grounding repair requires a V8 base preparation")
    source_payload = (base_dataset / "records.jsonl").read_bytes()
    source_manifest = (base_dataset / "manifest.json").read_bytes()
    if (hashlib.sha256(source_payload).hexdigest() != base_manifest["records_sha256"]
            or json.loads(source_manifest) != base_manifest):
        raise ValueError("base preparation changed during relation grounding repair")
    repaired, coverage = relation_grounded_records(records)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                      for row in repaired).encode()
    summary = wording_manifest(repaired)
    metadata = {
        "version": REPAIR_VERSION,
        "base_dataset": str(base_dataset.resolve()),
        "base_records_sha256": hashlib.sha256(source_payload).hexdigest(),
        "base_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
        "base_version": base_manifest["version"],
        "structural_supervision_unchanged": True,
        "semantic_mapping_ids_unchanged": True,
        "group_ids_and_splits_unchanged": True,
        "evaluation_wording_protocol_changed": True,
        "heldout_mappings_moved_to_train": 0,
        "scope": "Programmatic relations for supported specification operation groups only.",
        "unknown_and_unsupported_in_witness": False,
        "contrastive_grammar_policy": (
            "Matched pair views share a witness grammar; grouping alone differs."
        ),
        "coverage_records": coverage,
        "runtime_or_diagnostic_text_used": False,
        "effectiveness_measured": False,
        "inherited_augmentation_metadata_scope": (
            "Historical V8 stage provenance; current evaluation wording is V9."
        ),
    }
    manifest = {
        **base_manifest, **summary,
        "version": CONTROL_TEXT_DATA_VERSION, "wording_version": 9,
        "semantic_generator_version": CONTROL_TEXT_DATA_VERSION,
        "scenario_kinds": base_manifest["scenario_kinds"],
        "records_sha256": hashlib.sha256(payload).hexdigest(), "records_bytes": len(payload),
        "relation_witness_split_ownership": {
            f"v9-relation-witness-{family:02d}": split
            for split, families in INTERNAL_FAMILIES.items() for family in families
        },
        "relation_witness_group_counts": {
            split: len({row["group_id"] for row in repaired if row["split"] == split
                        and any(family.startswith("v9-relation-witness-")
                                for family in row["grammar_families"])})
            for split in INTERNAL_FAMILIES
        },
        "relation_grounding_repair": metadata,
        "status": "prepared", "updates": 0, "model_loaded": False,
        "readiness": False, "ppo_eligible": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "records.jsonl").write_bytes(payload)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return manifest
