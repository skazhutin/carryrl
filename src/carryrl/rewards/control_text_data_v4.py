"""Text-only synthetic controls, sampled as semantic specifications before rendering.

This is an offline annotation source, never a runtime input or control resolver.
Action identifiers, order, modes and wording are sampled independently. Semantic
identity and split assignment reuse v3's canonical effective-label definition.
"""

import hashlib
import json
import random
import string
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from carryrl.rewards.control_text_data import (
    CONTROL_CANDIDATES,
    SPLIT_WEIGHTS,
    _semantic_split,
    semantic_mapping_identity,
)

CONTROL_TEXT_DATA_VERSION = "carryrl.control_text_data.v4"
SCENARIO_KINDS = (
    "complete",
    "incomplete",
    "negation",
    "conflict",
    "conditional",
    "aliases",
    "timing_unknown",
    "navigation",
    "horizontal",
    "unknown",
)
_OPERATION_IDS = tuple(a + b for a in string.ascii_uppercase for b in string.ascii_uppercase)


@dataclass(frozen=True)
class OperationSpec:
    identifier: str
    candidates: tuple[str, ...]
    mode: str


@dataclass(frozen=True)
class ControlSpec:
    operations: tuple[OperationSpec, ...]
    uncertain: tuple[str, ...]
    forbidden: tuple[str, ...]
    remainder: str
    kind: str

    @property
    def declared_candidates(self):
        return (
            tuple(candidate for operation in self.operations for candidate in operation.candidates)
            + self.uncertain
            + self.forbidden
        )


def annotate_spec(spec):
    """Derive labels solely from the specification, before selecting any syntax."""
    labels = {
        candidate: dict(candidate=candidate, status=spec.remainder, mode="unknown", alias_group=0)
        for candidate in CONTROL_CANDIDATES
    }
    for group, operation in enumerate(spec.operations, 1):
        for candidate in operation.candidates:
            if operation.mode == "unknown":
                # The inference contract accepts a control only with an executable
                # timing. Known availability alone is insufficient for acceptance.
                labels[candidate].update(status="unknown")
            else:
                labels[candidate].update(status="supported", mode=operation.mode, alias_group=group)
    for candidate in spec.uncertain:
        labels[candidate].update(status="unknown")
    for candidate in spec.forbidden:
        labels[candidate].update(status="unsupported")
    return list(labels.values())


def _sample_spec(rng, kind):
    candidates = list(CONTROL_CANDIDATES)
    rng.shuffle(candidates)
    if kind == "unknown":
        return ControlSpec((), tuple(candidates[: rng.randint(1, 13)]), (), "unknown", kind)
    count = rng.randint(1, 13)
    groups = []
    if kind in {"navigation", "horizontal"}:
        groups = [("key:a", "key:left"), ("key:d", "key:right")]
        if kind == "navigation":
            groups += [("key:w", "key:up"), ("key:s", "key:down")]
        count = rng.randint(2 * len(groups), 13)
        candidates = [value for value in candidates if value not in sum(groups, ())]
    remaining = count - sum(map(len, groups))
    while remaining:
        size = 2 if remaining > 1 and rng.random() < 0.45 else 1
        groups.append(tuple(candidates[:size]))
        del candidates[:size]
        remaining -= size
    # No operation's timing is a function of its position, name or alias class.
    identifiers = rng.sample(_OPERATION_IDS, len(groups))
    operations = [
        OperationSpec(identifier, group, rng.choice(("held", "press")))
        for identifier, group in zip(identifiers, groups, strict=True)
    ]
    if kind == "timing_unknown":
        selected = rng.randrange(len(operations))
        operation = operations[selected]
        operations[selected] = OperationSpec(operation.identifier, operation.candidates, "unknown")
    rng.shuffle(operations)
    # Explicit negative evidence is available even in partial references. Unknown
    # candidates are separate from positive operations, never silently overwritten.
    uncertain_count = rng.randint(1, 3) if kind in {"incomplete", "conflict", "conditional"} else 0
    uncertain = tuple(candidates[:uncertain_count])
    forbidden = tuple(candidates[uncertain_count : uncertain_count + rng.randint(1, 3)])
    remainder = "unknown" if kind in {"incomplete", "conflict", "conditional"} else "unsupported"
    return ControlSpec(tuple(operations), uncertain, forbidden, remainder, kind)


def _name(candidate, language):
    kind, value = candidate.split(":")
    ru = language == "ru"
    if kind == "key":
        names = {
            "space": ("Пробел", "Space"),
            "up": ("стрелку вверх", "Up arrow"),
            "down": ("стрелку вниз", "Down arrow"),
            "left": ("стрелку влево", "Left arrow"),
            "right": ("стрелку вправо", "Right arrow"),
        }
        if value in names:
            return names[value][0 if ru else 1]
        return f"клавишу {value.upper()}" if ru else f"the {value.upper()} key"
    if kind == "mouse":
        name = {"left": "левую", "right": "правую", "middle": "среднюю"}[value]
        return f"{name} кнопку мыши" if ru else f"the {value} mouse button"
    return {
        "dx:-": ("мышь влево", "the mouse left"),
        "dx:+": ("мышь вправо", "the mouse right"),
        "dy:-": ("мышь вверх", "the mouse up"),
        "dy:+": ("мышь вниз", "the mouse down"),
        "wheel:-": ("колесо вниз", "the wheel down"),
        "wheel:+": ("колесо вверх", "the wheel up"),
    }[candidate][0 if ru else 1]


def _command(candidate, mode, language, grammar):
    ru = language == "ru"
    name = _name(candidate, language)
    relative = candidate.startswith(("dx:", "dy:", "wheel:"))
    movement = (
        ("Крутите" if candidate.startswith("wheel:") else "Двигайте")
        if ru
        else ("Scroll" if candidate.startswith("wheel:") else "Move")
    )
    if mode == "unknown":
        return (
            f"Ввод {name} доступен; длительность не указана"
            if ru
            else (f"Input {name} is enabled; timing is unspecified")
        )
    if relative:
        timing = (
            ("непрерывно" if mode == "held" else "один раз")
            if ru
            else ("continuously" if mode == "held" else "once")
        )
        return f"{movement} {name} {timing}"
    if grammar == "imperative":
        if ru:
            return f"Удерживайте {name}" if mode == "held" else f"Нажмите {name} и отпустите"
        return f"Hold {name}" if mode == "held" else f"Tap {name} and release"
    if grammar == "duration":
        if ru:
            return f"Держите {name} нажатой" if mode == "held" else f"Коротко нажмите {name}"
        return f"Keep {name} held down" if mode == "held" else f"Press {name} once"
    if grammar == "reference":
        timing = (
            ("удержание" if mode == "held" else "одиночное нажатие")
            if ru
            else ("held input" if mode == "held" else "single press")
        )
        return f"{name}: {timing}"
    templates = {
        "train": (
            ("Продолжайте зажимать {}", "Щёлкните {} однократно"),
            ("Continue holding {}", "Give {} a single tap"),
        ),
        "dev": (
            ("Не отпускайте {} во время действия", "Быстро нажмите и отпустите {}"),
            ("Leave {} down throughout the action", "Briefly tap {} then let go"),
        ),
        "calibration": (
            ("Используйте {} с удержанием", "Используйте {} разовым нажатием"),
            ("Use {} with sustained pressure", "Use {} with one brief press"),
        ),
        "sealed_test": (
            ("Для длительного ввода зажмите {}", "Для разового ввода нажмите {}"),
            ("Keep pressure on {} for the duration", "Activate {} with one tap"),
        ),
    }
    return templates[grammar][0 if ru else 1][0 if mode == "held" else 1].format(name)


def render_spec(spec, *, language, variant, split, seed):
    """Render natural commands; syntax selection cannot change the annotations."""
    rng = random.Random(seed)
    ru = language == "ru"
    operations = list(spec.operations)
    rng.shuffle(operations)
    chunks, grammars = [], set()
    if spec.kind == "navigation":
        chunks.append(
            "WASD или стрелки — парные замены."
            if ru
            else "WASD or arrow keys are paired alternatives."
        )
    elif spec.kind == "horizontal":
        chunks.append(
            "A/D или стрелки влево/вправо — парные замены."
            if ru
            else "A/D or left/right arrows are paired alternatives."
        )
    for operation in operations:
        grammar = rng.choice(("imperative", "duration", "reference")) if variant == 0 else split
        grammars.add(grammar)
        commands = [
            _command(value, operation.mode, language, grammar) for value in operation.candidates
        ]
        joiner = " или " if ru else " or "
        chunks.append(joiner.join(commands) + f" ({operation.identifier}).")
    if spec.uncertain:
        names = ", ".join(_name(value, language) for value in spec.uncertain)
        if spec.kind == "conflict":
            chunks.append(
                f"{names}: справка разрешает и запрещает; статус неизвестен."
                if ru
                else f"{names}: the help both enables and forbids these; status is unknown."
            )
        elif spec.kind == "conditional":
            chunks.append(
                f"{names} доступны только с разрешением; оно неизвестно."
                if ru
                else f"{names} work only with permission; permission is unknown."
            )
        else:
            chunks.append(
                f"{names}: доступность и режим неизвестны."
                if ru
                else f"{names}: availability and timing are unknown."
            )
    if spec.forbidden:
        names = ", ".join(_name(value, language) for value in spec.forbidden)
        chunks.append(
            f"Не используйте {names}: запрещено." if ru else f"Do not use {names}: unsupported."
        )
    if spec.remainder == "unknown":
        chunks.append(
            "Об остальных командах сведений нет." if ru else "No information about other controls."
        )
    else:
        chunks.append(
            "Все остальные команды отключены." if ru else "All other controls are disabled."
        )
    return " ".join(chunks), sorted(grammars)


def generate_control_text_records(*, seed=20261020, scenario_groups=4000, wording_version=4):
    if type(wording_version) is not int or wording_version not in (4, 5):
        raise ValueError("wording_version must be 4 or 5")
    if type(scenario_groups) is not int or scenario_groups < 100 or scenario_groups % 100:
        raise ValueError("scenario_groups must be a positive multiple of 100")
    if scenario_groups > 40000:
        raise ValueError("too many unique scenario mappings")
    rng, seen, records = random.Random(seed), set(), []
    index = 0
    for split, weight in SPLIT_WEIGHTS:
        for local in range(scenario_groups * weight // 100):
            # Five percent fully unknown leaves >=350/400 calibration clusters
            # with both positive and negative evidence. All other kinds recur.
            kind = "unknown" if local % 20 == 19 else SCENARIO_KINDS[local % 9]
            for _ in range(100000):
                spec = _sample_spec(rng, kind)
                labels = annotate_spec(spec)
                identity = semantic_mapping_identity(labels, spec.declared_candidates)
                if identity not in seen and _semantic_split(identity, seed) == split:
                    seen.add(identity)
                    break
            else:
                raise ValueError("requested budget exhausts assigned semantic mappings")
            for variant, language in enumerate(("ru", "ru", "en", "en")):
                render_seed = rng.getrandbits(64)
                if wording_version == 4:
                    text, grammars = render_spec(
                        spec,
                        language=language,
                        variant=variant % 2,
                        split=split,
                        seed=render_seed,
                    )
                else:
                    from carryrl.rewards.control_text_data_v5 import render_spec as render_v5

                    text, grammars = render_v5(
                        spec,
                        language=language,
                        variant=variant,
                        split=split,
                        seed=render_seed,
                        pair_seed=identity,
                    )
                records.append(
                    {
                        "group_id": f"v4-scenario-{index:05d}",
                        "split": split,
                        "mapping_id": identity,
                        "semantic_mapping_id": identity,
                        "declared_candidates": list(spec.declared_candidates),
                        "template_family": (
                            "shared-core" if variant % 2 == 0 else f"{split}-paraphrase"
                        )
                        if wording_version == 4
                        else f"v5-{split}-outer-{variant % 2}",
                        "grammar_families": grammars,
                        "scenario_kind": kind,
                        "language": language,
                        "text": text,
                        "labels": labels,
                        "specification": asdict(spec),
                    }
                )
            index += 1
    return records


def prepare_control_text_dataset(
    output,
    *,
    seed=20261020,
    scenario_groups=4000,
    wording_version=4,
    excluded_semantic_ids=(),
    reservation_registry=None,
):
    """Prepare authenticated corpus and audit summaries without loading a model."""
    if type(wording_version) is int and wording_version == 6:
        from carryrl.rewards.control_text_data_v6 import (
            generate_control_text_records as generate_v6,
        )
        from carryrl.rewards.control_text_data_v6 import read_reservation_registry

        exclusions, reservation_metadata = read_reservation_registry(
            reservation_registry, excluded_semantic_ids
        )
        records = generate_v6(
            seed=seed, scenario_groups=scenario_groups, excluded_semantic_ids=exclusions
        )
    else:
        if reservation_registry is not None or excluded_semantic_ids:
            raise ValueError("semantic reservations require v6")
        records = generate_control_text_records(
            seed=seed, scenario_groups=scenario_groups, wording_version=wording_version
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with (output / "records.jsonl").open("wb") as stream:
        for row in records:
            line = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
            stream.write(line)
            digest.update(line)
    groups = records[::4]
    distributions = {}
    for split, _ in SPLIT_WEIGHTS:
        subset = [row for row in groups if row["split"] == split]
        labels = [label for row in subset for label in row["labels"]]
        distributions[split] = {
            "scenario_groups": dict(Counter(row["scenario_kind"] for row in subset)),
            "supported_count_groups": dict(
                Counter(
                    str(sum(label["status"] == "supported" for label in row["labels"]))
                    for row in subset
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
                        for row in subset
                        for offset, operation in enumerate(row["specification"]["operations"])
                        if offset == position
                    )
                )
                for position in range(13)
            },
        }
    text_labels = {}
    for row in records:
        # Identical strings must never carry different effective supervision.
        previous = text_labels.setdefault(row["text"], row["semantic_mapping_id"])
        if previous != row["semantic_mapping_id"]:
            raise ValueError("same text has conflicting labels")
    manifest = {
        "version": CONTROL_TEXT_DATA_VERSION,
        "seed": seed,
        "scenario_groups": scenario_groups,
        "records": len(records),
        "variants_per_group": 4,
        "candidates": list(CONTROL_CANDIDATES),
        "scenario_kinds": list(SCENARIO_KINDS),
        "split_records": dict(Counter(row["split"] for row in records)),
        "split_groups": dict(Counter(row["split"] for row in groups)),
        "records_sha256": digest.hexdigest(),
        "records_bytes": (output / "records.jsonl").stat().st_size,
        "split_before_paraphrase": True,
        "mapping_disjoint": True,
        "semantic_mapping_disjoint": True,
        "semantic_mapping_unique_per_group": True,
        "mapping_identity_definition": "v3 canonical candidate/status/mode/alias assignment",
        "null_mapping_exception": "All-unknown mappings partition the sorted named candidate set.",
        "template_families_disjoint": False,
        "paraphrase_holdout": (
            "One variant per language uses split-specific syntax; core grammar is shared."
        ),
        "model_input_fields": ["text"],
        "same_text_labels_consistent": True,
        "modes_sampled_independently_per_operation": True,
        "operation_ids_and_render_order_randomized": True,
        "distributions": distributions,
        "calibration_eligible_sign_groups": {
            status: sum(
                row["split"] == "calibration"
                and any(label["status"] == status for label in row["labels"])
                for row in groups
            )
            for status in ("supported", "unsupported")
        },
        "maximum_supported_candidates": max(
            sum(label["status"] == "supported" for label in row["labels"]) for row in groups
        ),
        "statuses": ["supported", "unsupported", "unknown"],
        "modes": ["held", "press", "unknown"],
        "alias_group_semantics": "local equivalence classes; 0 means unspecified",
        "label_semantics": (
            "Supported requires explicit availability and known executable timing; "
            "enabled input with unspecified timing abstains as unknown, mode unknown, alias 0."
        ),
        "status": "prepared",
        "updates": 0,
        "model_loaded": False,
        "readiness": False,
        "ppo_eligible": False,
        "limitations": [
            "Synthetic specifications only; no learned quality measured.",
            "Sealed test reserved until model freeze; runtime instructions not used.",
        ],
    }
    if wording_version == 5:
        from carryrl.rewards.control_text_data_v5 import wording_manifest

        manifest.update(wording_manifest(records))
    elif wording_version == 6:
        from carryrl.rewards.control_text_data_v6 import wording_manifest

        manifest.update(wording_manifest(records))
        manifest.update(reservation_metadata)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
