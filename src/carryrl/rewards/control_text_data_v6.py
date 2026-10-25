"""Compositional synthetic control specifications, independent of runtime games.

V6 separates sparse/complete scope from auxiliary evidence, supports compact
alias groups and explicit two-axis mouse descriptions, and mixes binding clauses
with irrelevant context. Missing timing always abstains; it is never inferred
from a movement verb or a hidden control specification.
"""

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from carryrl.rewards.control_text_data import (
    CONTROL_CANDIDATES,
    SPLIT_WEIGHTS,
    _semantic_split,
    semantic_mapping_identity,
)
from carryrl.rewards.control_text_data_v4 import (
    ControlSpec,
    OperationSpec,
    _name,
)
from carryrl.rewards.control_text_data_v4 import (
    annotate_spec as annotate_v4,
)
from carryrl.rewards.control_text_data_v5 import _WRAPPERS, REMAINDER_FORMS

CONTROL_TEXT_DATA_VERSION = "carryrl.control_text_data.v6"
DEFAULT_SEED = 20261022
_AXES = ("dx:-", "dx:+", "dy:-", "dy:+")
_SCHEDULE = (
    "navigation",
    "horizontal",
    "mixed",
    "sparse_partial",
    "conditional",
    "dense",
    "mouse_axes",
    "goal_context",
    "remote_press",
    "scoped_alias",
    "complete",
    "compact_alias",
    "mixed",
    "sparse_partial",
    "incomplete",
    "dense",
    "mouse_axes",
    "remote_press",
    "implicit",
    "unknown",
)
SCENARIO_KINDS = tuple(dict.fromkeys(_SCHEDULE))
_GOALS = {
    "en": (
        "Place the bronze marker inside the dotted frame.",
        "Arrange the patterned tiles beside the violet ring.",
        "Bring the silver token to the striped panel.",
        "Keep the patterned marker inside the pale region.",
    ),
    "ru": (
        "Помести бронзовую метку внутрь пунктирной рамки.",
        "Расположи узорные плитки рядом с фиолетовым кольцом.",
        "Доставь серебряный жетон к полосатой панели.",
        "Сохраняй узорную метку внутри светлой области.",
    ),
}
_DISTRACTORS = {
    "en": (
        "The picture on the right is purple; its color does not change the controls.",
        "A decorative caption says 'hold'; it is not an input instruction.",
        "The background flashes twice; this does not request any input.",
        "A small triangle at the top only shows progress.",
    ),
    "ru": (
        "Картинка справа фиолетовая; её цвет не меняет управление.",
        "Декоративная подпись говорит «нажми»; это не команда ввода.",
        "Фон мигает дважды; это не требует ввода.",
        "Маленький треугольник сверху только показывает прогресс.",
    ),
}
_PHASES = {
    "en": (
        "No input is used during preparation. Preparation finishes before the active phase; "
        "the bindings below then apply.",
        "The introductory interval blocks input temporarily. "
        "Once it ends, use the following controls.",
        "Commands wait during the opening cue. The cue clears before action starts; "
        "these controls become available then.",
    ),
    "ru": (
        "Во время подготовки ввод не используется. Подготовка заканчивается до активной фазы; "
        "затем действуют описанные ниже команды.",
        "Вступительный интервал временно блокирует ввод. "
        "После его окончания используй эти команды.",
        "Команды ждут во время начального сигнала. Сигнал исчезает до начала действия; "
        "после этого управление доступно.",
    ),
}


_SIGNALS = (
    ("amber", "янтарного"),
    ("copper", "медного"),
    ("violet", "фиолетового"),
    ("silver", "серебряного"),
    ("bronze", "бронзового"),
    ("azure", "лазурного"),
    ("coral", "кораллового"),
    ("olive", "оливкового"),
    ("indigo", "синего"),
    ("pearl", "жемчужного"),
    ("ruby", "рубинового"),
    ("teal", "бирюзового"),
    ("ochre", "охристого"),
    ("plum", "сливового"),
    ("lime", "салатового"),
    ("rose", "розового"),
)
_NAVIGATION = (
    ("key:w", "key:up"),
    ("key:s", "key:down"),
    ("key:a", "key:left"),
    ("key:d", "key:right"),
)
_HORIZONTAL = _NAVIGATION[2:]


def annotate_spec(spec):
    """Use the runtime's distinct timing contract for discrete vs relative inputs."""
    labels = annotate_v4(spec)
    relative_groups = {
        candidate: group
        for group, operation in enumerate(spec.operations, 1)
        for candidate in operation.candidates
        if candidate.startswith(("dx:", "dy:", "wheel:"))
    }
    for label in labels:
        if label["candidate"] in relative_groups:
            label.update(
                status="supported", mode="unknown", alias_group=relative_groups[label["candidate"]]
            )
    return labels


def _sample_spec(rng, kind):
    pool = list(CONTROL_CANDIDATES)
    rng.shuffle(pool)
    if kind == "unknown":
        return ControlSpec((), tuple(pool[: rng.randint(1, 13)]), (), "unknown", kind)
    count = rng.randint(1, 13)
    if kind in {"dense", "mixed"}:
        count = rng.randint(8, 13)
    if kind == "sparse_partial":
        count = rng.randint(1, 3)
    if kind == "implicit":
        pool = [candidate for candidate in pool if candidate.startswith("key:")]
        count = rng.randint(1, 6)
    groups, modes = [], []
    if kind in {"navigation", "horizontal"}:
        groups = list(_NAVIGATION if kind == "navigation" else _HORIZONTAL)
        used = {candidate for group in groups for candidate in group}
        pool = [candidate for candidate in pool if candidate not in used]
        count = rng.randint(len(used) + 1, 13)
        shared = rng.choice(("held", "press")) if rng.random() < 0.5 else None
        modes = [shared or rng.choice(("held", "press")) for _ in groups]
    if kind == "mouse_axes":
        # Each sign is a different physical input, never an equivalence class.
        mode = rng.choice(("held", "press", "unknown"))
        groups.extend((candidate,) for candidate in _AXES)
        modes.extend([mode] * 4)
        pool = [candidate for candidate in pool if candidate not in _AXES]
        count = rng.randint(4, 13)
    if kind in {"compact_alias", "scoped_alias"}:
        count = rng.randint(4, 13)
        for _ in range(2):
            groups.append(tuple(pool[:2]))
            del pool[:2]
            modes.append(rng.choice(("held", "press")))
    if kind == "mixed":
        selected = [
            rng.choice([value for value in pool if value.startswith(prefix)])
            for prefix in ("key:", "mouse:", "wheel:")
        ]
        for value in selected:
            pool.remove(value)
            groups.append((value,))
            modes.append(rng.choice(("held", "press")))
    remaining = count - sum(map(len, groups))
    while remaining:
        size = 2 if remaining > 1 and rng.random() < 0.4 else 1
        groups.append(tuple(pool[:size]))
        del pool[:size]
        remaining -= size
        modes.append(rng.choice(("held", "press")))
    if kind == "implicit":
        modes = ["unknown"] * len(groups)
    if kind == "remote_press":
        # At least one discrete input whose timing can be stated remotely.
        discrete = [
            i
            for i, group in enumerate(groups)
            if all(value.startswith(("key:", "mouse:")) for value in group)
        ]
        if not discrete:
            return _sample_spec(rng, kind)
        modes[rng.choice(discrete)] = "press"
    operations = [
        OperationSpec(f"{identifier:03x}", group, mode)
        for identifier, group, mode in zip(
            rng.sample(range(4096), len(groups)), groups, modes, strict=True
        )
    ]
    rng.shuffle(operations)
    partial = kind in {"sparse_partial", "incomplete", "conditional", "implicit"}
    if kind in {"navigation", "horizontal"} and rng.random() < 0.4:
        partial = True
    uncertain_count = rng.randint(1, 3) if kind in {"incomplete", "conditional"} else 0
    # Sparse positives deliberately have zero explicit unknown/negative extras.
    forbidden_count = 0 if kind == "sparse_partial" else rng.randint(1, 3)
    uncertain = tuple(pool[:uncertain_count])
    forbidden = tuple(pool[uncertain_count : uncertain_count + forbidden_count])
    return ControlSpec(
        tuple(operations), uncertain, forbidden, "unknown" if partial else "unsupported", kind
    )


def _group_name(operation, language, compact=False):
    return (" или " if language == "ru" else " or ").join(
        value.split(":")[1].upper().replace("SPACE", "Space" if language == "en" else "Пробел")
        if (
            compact
            and value.startswith("key:")
            and value.split(":")[1] not in {"up", "down", "left", "right"}
        )
        else _name(value, language)
        for value in operation.candidates
    )


def _binding(operation, language, family):
    names, ru = _group_name(operation, language, compact=family % 2 == 0), language == "ru"
    if operation.mode == "unknown":
        return f"Движение выполняется через {names}" if ru else f"Movement uses {names}"
    relative = any(value.startswith(("dx:", "dy:", "wheel:")) for value in operation.candidates)
    if relative:
        if any(value.startswith(("key:", "mouse:")) for value in operation.candidates):
            # A held key must be held, while a relative movement must be repeated.
            alternatives = (" или " if ru else " or ").join(
                _binding(
                    OperationSpec(operation.identifier, (value,), operation.mode), language, family
                )
                for value in operation.candidates
            )
            return (
                "Равнозначные варианты: " if ru else "Equivalent input choices: "
            ) + alternatives
        timing = (
            (
                "непрерывно повторяйте выбранный ввод"
                if operation.mode == "held"
                else "выполните одно короткое движение и остановитесь"
            )
            if ru
            else (
                "repeat the selected input continuously"
                if operation.mode == "held"
                else "make one brief movement and stop"
            )
        )
        return f"Выберите {names}; {timing}" if ru else f"Use {names}; {timing}"
    held = {
        "en": (
            "Hold {names}",
            "Keep {names} down throughout the action",
            "Use {names} with sustained pressure",
            "Leave {names} held down",
        ),
        "ru": (
            "Удерживайте {names}",
            "Держите {names} нажатой всё время действия",
            "Используйте {names} с удержанием",
            "Не отпускайте {names} до конца действия",
        ),
    }
    press = {
        "en": (
            "Press {names} once",
            "Tap {names} and release",
            "Give {names} one brief press",
            "Use {names} for a single press, then let go",
        ),
        "ru": (
            "Нажмите {names} один раз",
            "Коротко нажмите {names} и отпустите",
            "Выполните одиночное нажатие: {names}",
            "Нажмите {names} однократно и отпустите",
        ),
    }
    if (
        len(operation.candidates) == 1
        and operation.candidates[0].startswith("mouse:")
        and operation.mode == "press"
        and family == 2
    ):
        return f"Щёлкните {names} один раз" if ru else f"Click {names} once"
    return (held if operation.mode == "held" else press)[language][family].format(names=names)


def render_spec(spec, *, language, variant, split, seed, pair_seed):
    rng, ru = random.Random(seed), language == "ru"
    operations = list(spec.operations)
    rng.shuffle(operations)
    chunks, grammars, remote = [], set(), []
    used_ids = {operation.identifier for operation in operations}
    signals = dict(
        zip(
            (operation.identifier for operation in operations),
            rng.sample(_SIGNALS, len(operations)),
            strict=True,
        )
    )

    def statement(text, identifier=None):
        if variant % 2 and rng.random() < 0.5:
            if identifier is None:
                while True:
                    identifier = f"{rng.randrange(4096):03x}"
                    if identifier not in used_ids:
                        used_ids.add(identifier)
                        break
            text += f" [{identifier}]"
        return text + "."

    axes = [op for op in operations if spec.kind == "mouse_axes" and op.candidates[0] in _AXES]
    canonical = []
    if spec.kind in {"navigation", "horizontal"}:
        pairs = set(_NAVIGATION if spec.kind == "navigation" else _HORIZONTAL)
        canonical = [operation for operation in operations if operation.candidates in pairs]
        names = (
            ("WASD или стрелки" if ru else "WASD or arrow keys")
            if spec.kind == "navigation"
            else ("A/D или стрелки влево/вправо" if ru else "A/D or left/right arrows")
        )
        count = 4 if spec.kind == "navigation" else 2
        modes = {operation.mode for operation in canonical}
        if len(modes) == 1:
            mode = next(iter(modes))
            timing = (
                (
                    "Удерживай выбранную клавишу всё время действия"
                    if mode == "held"
                    else "Нажми выбранную клавишу один раз, затем отпусти"
                )
                if ru
                else (
                    "Hold the chosen key throughout the action"
                    if mode == "held"
                    else "Press the chosen key once, then release it"
                )
            )
            chunks.append(
                f"{names}: {count} разных направления; "
                f"замены одного направления равнозначны. {timing}."
                if ru
                else f"{names} provide {count} distinct direction commands; "
                f"matching directions are equivalent alternatives. {timing}."
            )
            grammars.add(f"canonical-{spec.kind}-shared-{mode}")
        else:
            canonical = []
            chunks.append(
                f"{names}: режим каждого направления указан отдельно."
                if ru
                else f"{names}: timing is specified separately for each direction."
            )
            grammars.add(f"canonical-{spec.kind}-expanded-mixed")
    if axes:
        names = (
            "Движение мыши перемещает указатель. Обе оси и оба знака каждой оси доступны; "
            "это четыре разных ввода, не взаимозаменяемые команды"
            if ru
            else "Mouse movement positions the pointer. Both axes and both signs of each axis "
            "are enabled; these are four distinct inputs, not equivalent alternatives"
        )
        if axes[0].mode == "held":
            timing = (
                "Непрерывно двигайте мышь в течение действия"
                if ru
                else "Keep moving the mouse continuously throughout the action"
            )
        elif axes[0].mode == "press":
            timing = (
                "Каждая команда — одно короткое движение мыши, затем остановка"
                if ru
                else "Each command is one brief mouse movement, then stop"
            )
        else:
            timing = "Длительность движения не задана" if ru else "Movement timing is unspecified"
        # A second generic lexical family describes physical axes through view angles.
        if rng.random() < 0.5:
            names = (
                "Мышь меняет азимут по горизонтали и угол наклона по вертикали. "
                "Оба направления каждой оси доступны как разные команды"
                if ru
                else "Mouse motion changes yaw on the horizontal axis and pitch on the "
                "vertical axis. Both directions of each axis work as distinct commands"
            )
        chunks.append(names + ". " + timing + ".")
        grammars.add("generic-independent-mouse-axes")
    scoped = []
    for operation in operations:
        if operation in axes or operation in canonical:
            continue
        family = rng.randrange(4)
        grammars.add(f"compact-{operation.mode}-{family}")
        can_remote = operation.mode == "press" and all(
            value.startswith(("key:", "mouse:")) for value in operation.candidates
        )
        if spec.kind == "remote_press" and can_remote:
            names = _group_name(operation, language, compact=family % 2 == 0)
            signal_en, signal_ru = signals[operation.identifier]
            clause = (
                f"Используйте {names} для {signal_ru} сигнала"
                if ru
                else f"Use {names} for the {signal_en} signal"
            )
            remote.append(names)
            grammars.add("remote-release-between")
        else:
            clause = _binding(operation, language, family)
        if ru and rng.random() < 0.5:
            for formal, familiar in (
                ("Удерживайте", "Удерживай"),
                ("Держите", "Держи"),
                ("Нажмите", "Нажимай"),
                ("Используйте", "Используй"),
                ("Выполните", "Выполни"),
                ("Выберите", "Выбери"),
                ("Не отпускайте", "Не отпускай"),
                ("отпустите", "отпусти"),
            ):
                clause = clause.replace(formal, familiar)
            grammars.add("ru-singular-imperative")
        if spec.kind == "scoped_alias":
            # Conjunction joins different operations; disjunction stays within
            # one local equivalence class, whose duration appears only once.
            scoped.append(statement(clause, operation.identifier)[:-1])
        else:
            chunks.append(statement(clause, operation.identifier))
    if scoped:
        chunks.append(("; и " if ru else "; and ").join(scoped) + ".")
        grammars.add("scoped-or-with-and-between-operations")
    for candidate in spec.uncertain:
        name = _name(candidate, language)
        if spec.kind == "conditional":
            text = (
                (
                    f"Если неизвестное условие выполнено, доступен ввод {name}; "
                    "без сведений об условии его доступность неизвестна"
                )
                if ru
                else (
                    f"Input {name} works if an unspecified condition holds; "
                    "without the condition its availability is unknown"
                )
            )
        else:
            text = f"{name}: доступность неизвестна" if ru else f"{name}: availability is unknown"
        chunks.append(statement(text))
    for candidate in spec.forbidden:
        # Keep partial negative evidence; complete references may rely on scope.
        if spec.remainder == "unsupported" and rng.random() < 0.5:
            continue
        name = _name(candidate, language)
        text = f"Не используйте {name}: запрещено" if ru else f"Do not use {name}: unsupported"
        chunks.append(statement(text))
    if remote:
        chunks.insert(
            0,
            "Сигналы с разными названиями — независимые действия."
            if ru
            else "Differently named signals are independent operations.",
        )
        # Every named group still contains or; and separates independent groups.
        names = ("; и " if ru else "; and ").join(remote)
        chunks.append(
            f"Для каждого из этих вводов — {names} — отпускайте его между нажатиями; "
            "один сигнал требует одного короткого нажатия."
            if ru
            else f"For each of these inputs — {names} — release it between activations; "
            "one signal requires one brief press."
        )
    if spec.kind == "goal_context" or rng.random() < 0.3:
        chunks.insert(0, rng.choice(_GOALS[language]))
        chunks.insert(rng.randrange(1, len(chunks) + 1), rng.choice(_DISTRACTORS[language]))
        grammars.add("irrelevant-goal-and-distractor-context")
    if spec.kind not in {"implicit", "unknown"} and rng.random() < 0.15:
        chunks.insert(0, rng.choice(_PHASES[language]))
        grammars.add("guaranteed-phase-transition")
    offset = 2 * (int(hashlib.sha256(pair_seed.encode()).hexdigest(), 16) % 4)
    chunks.append(REMAINDER_FORMS[spec.remainder][language][(offset + variant) % 8])
    body = " ".join(chunks)
    return _WRAPPERS[split][language][variant % 2].format(body=body), sorted(grammars)


def _validated_exclusions(identifiers):
    exclusions = set(identifiers)
    if any(
        not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity)
        for identity in exclusions
    ):
        raise ValueError("excluded semantic IDs must be lowercase SHA256 hashes")
    return frozenset(exclusions)


def read_reservation_registry(path=None, excluded_semantic_ids=()):
    """Read labels-only canonical hashes, never examples or model predictions."""
    explicit = _validated_exclusions(excluded_semantic_ids)
    registered, digest = frozenset(), None
    if path is not None:
        payload = Path(path).read_bytes()
        document = json.loads(payload)
        if not isinstance(document, dict) or not isinstance(document.get("mapping_ids"), list):
            raise ValueError("reservation registry must contain a mapping_ids list")
        registered = _validated_exclusions(document["mapping_ids"])
        digest = hashlib.sha256(payload).hexdigest()
    exclusions = explicit | registered
    return exclusions, {
        "reservation_registry": str(Path(path).resolve()) if path is not None else None,
        "reservation_registry_sha256": digest,
        "reservation_registry_mapping_count": len(registered),
        "explicit_excluded_semantic_ids_count": len(explicit),
        "excluded_semantic_mapping_count": len(exclusions),
        "excluded_semantic_mappings_observed": 0,
        "reservation_exclusion_scope": "all splits, before split assignment and rendering",
    }


def generate_control_text_records(
    *, seed=DEFAULT_SEED, scenario_groups=4000, excluded_semantic_ids=()
):
    if type(scenario_groups) is not int or scenario_groups < 100 or scenario_groups % 100:
        raise ValueError("scenario_groups must be a positive multiple of 100")
    if scenario_groups > 40000:
        raise ValueError("too many unique scenario mappings")
    excluded = _validated_exclusions(excluded_semantic_ids)
    rng, seen, records = random.Random(seed), set(), []
    index = 0
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
                        "group_id": f"v6-scenario-{index:05d}",
                        "split": split,
                        "mapping_id": identity,
                        "semantic_mapping_id": identity,
                        "declared_candidates": list(spec.declared_candidates),
                        "template_family": f"v6-{split}-outer-{variant % 2}",
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


def wording_manifest(records):
    from carryrl.rewards.control_text_data_v5 import wording_manifest as summarize_v5

    summary = summarize_v5(records)
    groups = records[::4]
    summary.update(
        {
            "version": CONTROL_TEXT_DATA_VERSION,
            "wording_version": 6,
            "semantic_generator_version": CONTROL_TEXT_DATA_VERSION,
            "semantic_rng_stream_preserved": False,
            "scenario_kinds": list(SCENARIO_KINDS),
            "id_presence_semantics": (
                "Half variants anonymous; optional status-agnostic reference IDs."
            ),
            "label_semantics": (
                "Key/button acceptance requires explicit timing; "
                "implicit keyboard movement abstains. "
                "Explicitly available relative axes remain supported when timing is unspecified. "
                "Generic two-axis mouse descriptions specify four independent signed inputs."
            ),
            "relative_mode_identity": (
                "Supported dx/dy/wheel normalize mode to unknown, matching runtime output."
            ),
            "modes_sampled_independently_per_operation": False,
            "mode_sampling": (
                "Ordinary groups sample independent modes; canonical direction sets can "
                "sample one shared explicit mode or independent per-direction modes."
            ),
            "canonical_group_patterns": {
                split: dict(
                    Counter(
                        family
                        for row in groups
                        if row["split"] == split
                        for family in row["grammar_families"]
                        if family.startswith("canonical-")
                    )
                )
                for split, _ in SPLIT_WEIGHTS
            },
            "canonical_group_patterns_unit": "independent_semantic_groups",
            "new_composition_groups": {
                split: dict(
                    Counter(row["scenario_kind"] for row in groups if row["split"] == split)
                )
                for split, _ in SPLIT_WEIGHTS
            },
            "sparse_partial_definition": (
                "Known positive bindings, zero uncertain, zero forbidden, unknown remainder."
            ),
            "minimal_calibration_eligible_sign_groups": 300,
            "runtime_instructions_used": False,
        }
    )
    eligible = {
        status: sum(
            row["split"] == "calibration"
            and any(label["status"] == status for label in row["labels"])
            for row in groups
        )
        for status in ("supported", "unsupported")
    }
    # Small unit-test preparations need not meet the production statistical budget.
    if len(groups) >= 4000 and min(eligible.values()) < 300:
        raise ValueError("insufficient independent calibration groups for either sign")
    summary["calibration_eligible_sign_groups"] = eligible
    for split, _ in SPLIT_WEIGHTS:
        subset = [row for row in records if row["split"] == split]
        distribution = summary["wording_distributions"][split]
        tagged = sum(bool(re.search(r"\[[0-9a-f]{3}\]", row["text"])) for row in subset)
        distribution["tagged_records"] = tagged
        distribution["anonymous_records"] = len(subset) - tagged
    return summary


def prepare_control_text_dataset(
    output,
    *,
    seed=DEFAULT_SEED,
    scenario_groups=4000,
    excluded_semantic_ids=(),
    reservation_registry=None,
):
    from carryrl.rewards.control_text_data_v4 import prepare_control_text_dataset as prepare_v4

    return prepare_v4(
        output,
        seed=seed,
        scenario_groups=scenario_groups,
        wording_version=6,
        excluded_semantic_ids=excluded_semantic_ids,
        reservation_registry=reservation_registry,
    )
