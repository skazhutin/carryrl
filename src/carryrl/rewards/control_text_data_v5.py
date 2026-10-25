"""Wording-only v5: shared lexical primitives, optional IDs and explicit scope.

Semantic generation stays in v4, including its RNG stream and split assignment.
Only outer compositions are held out; basic mode and remainder words are shared.
This module is an offline renderer, never an inference-time control resolver.
"""

import hashlib
import random
import re
import string
from collections import Counter

from carryrl.rewards.control_text_data_v4 import _name

CONTROL_TEXT_DATA_VERSION = "carryrl.control_text_data.v5"
REMAINDER_FORMS = {
    "unsupported": {
        "en": (
            "All other controls are unsupported.",
            "All other controls are disabled.",
            "No other controls are used.",
            "This is the full list of available inputs; all remaining inputs are unsupported.",
            "All other keys, buttons, mouse movements and wheel inputs are disabled.",
            "Only the listed available controls may be used; all other inputs are unsupported.",
            "All remaining controls outside this list are disabled.",
            "Any input not described above is unsupported.",
        ),
        "ru": (
            "Все остальные команды не поддерживаются.",
            "Все остальные команды отключены.",
            "Другие команды не используются.",
            "Это полный список доступного ввода; все остальные команды не поддерживаются.",
            "Остальные клавиши, кнопки, движения мыши и колесо отключены.",
            "Разрешены только перечисленные доступные команды; прочий ввод не поддерживается.",
            "Все оставшиеся команды вне списка отключены.",
            "Любой ввод, не описанный выше, не поддерживается.",
        ),
    },
    "unknown": {
        "en": (
            "No information is available about other controls.",
            "Other inputs are unspecified.",
            "The status of all remaining controls is unknown.",
            "Other keys, buttons, mouse movements and wheel inputs have not been specified.",
            "Do not infer availability or timing for any other input.",
            "Other inputs may or may not work; the reference does not say.",
            "Availability and timing of inputs not mentioned above are unknown.",
            "Only part of the controls is described; nothing is known about the remaining inputs.",
        ),
        "ru": (
            "Об остальных командах сведений нет.",
            "Прочий ввод не описан.",
            "Статус всех оставшихся команд неизвестен.",
            "Другие клавиши, кнопки, движения мыши и колесо не описаны.",
            "Не делайте выводов о доступности и режиме прочего ввода.",
            "Остальные команды могут работать или нет; справка не уточняет.",
            "Доступность и режим ввода, не упомянутого выше, неизвестны.",
            "Описана только часть команд; об оставшемся вводе ничего не известно.",
        ),
    },
}
_HELD = {
    "en": (
        "Hold {name}",
        "Keep {name} held down",
        "Leave {name} down throughout the action",
        "Use {name} with sustained pressure",
        "Keep continuous pressure on {name}",
        "Keep {name} down until the action ends",
        "Hold {name} continuously throughout the action",
        "For this input, {name} must be held",
    ),
    "ru": (
        "Удерживайте {name}",
        "Держите {name} нажатой",
        "Не отпускайте {name} во время действия",
        "Используйте {name} с удержанием",
        "Продолжайте зажимать {name}",
        "Держите {name} до конца действия",
        "Непрерывно удерживайте {name} всё время действия",
        "Для этого ввода нужно удерживать {name}",
    ),
}
_PRESS = {
    "en": (
        "Press {name} once",
        "Tap {name} and release",
        "Give {name} a single brief press",
        "Briefly press and release {name}",
        "Use {name} with one brief press",
        "Give {name} one tap, then release",
        "Press {name} for a single activation, then let go",
        "Tap {name} briefly",
    ),
    "ru": (
        "Нажмите {name} один раз",
        "Коротко нажмите {name} и отпустите",
        "Выполните одиночное нажатие: {name}",
        "Быстро нажмите и отпустите {name}",
        "Используйте {name} разовым нажатием",
        "Нажмите {name} однократно, затем отпустите",
        "Нажмите {name} для одного действия и сразу отпустите",
        "Кратко нажмите {name}",
    ),
}
_WRAPPERS = {
    "train": {"en": ("{body}", "Controls: {body}"), "ru": ("{body}", "Управление: {body}")},
    "dev": {
        "en": ("Operator note: {body} End of note.", "Before starting: {body}"),
        "ru": ("Памятка оператора: {body} Конец памятки.", "Перед началом: {body}"),
    },
    "calibration": {
        "en": ("Help entry: {body} End of entry.", "Input reference: {body}"),
        "ru": ("Раздел справки: {body} Конец раздела.", "Справочник ввода: {body}"),
    },
    "sealed_test": {
        "en": ("Notice begins. {body} Notice ends.", "Control reference follows: {body}"),
        "ru": ("Начало объявления. {body} Конец объявления.", "Сведения о вводе: {body}"),
    },
}


def _command(candidate, mode, language, family):
    name, ru = _name(candidate, language), language == "ru"
    if mode == "unknown":
        return (
            f"Разрешено использовать {name}; длительность не указана"
            if ru
            else f"Input {name} is enabled; timing is unspecified"
        )
    if not candidate.startswith(("dx:", "dy:", "wheel:")):
        return (_HELD if mode == "held" else _PRESS)[language][family].format(name=name)
    wheel = candidate.startswith("wheel:")
    verb = ("Крутите" if wheel else "Двигайте") if ru else ("Scroll" if wheel else "Move")
    if mode == "held":
        timings = (
            ("непрерывно", "всё время действия", "повторяя в каждом такте")
            if ru
            else ("continuously", "throughout the action", "repeatedly on every interval")
        )
    else:
        timings = (
            ("один раз", "одним коротким движением", "однократно")
            if ru
            else ("once", "with one brief movement", "a single time")
        )
    return f"{verb} {name} {timings[family % 3]}"


def render_spec(spec, *, language, variant, split, seed, pair_seed=""):
    """Four paired variants change lexical scope and IDs without changing labels.

    The first variant per language is anonymous. The second independently adds
    optional IDs to any statement, irrespective of status or mode. Footer offsets
    use one semantic-group hash, so paired variants cannot pick the same wording.
    """
    rng = random.Random(seed)
    operations = list(spec.operations)
    rng.shuffle(operations)  # Same initial permutation as the corresponding v4 render.
    # A complete footer already forbids every unlisted input. Sometimes omit the
    # redundant explicit negatives, yielding concise single-binding references.
    # Partial references must retain all explicit negative evidence.
    forbidden = spec.forbidden
    if spec.remainder == "unsupported" and rng.random() < 0.5:
        forbidden = ()
    ru = language == "ru"
    chunks, grammars, used_ids = [], set(), {op.identifier for op in operations}
    statement_count = len(operations) + len(spec.uncertain) + len(forbidden)
    required_tag = rng.randrange(statement_count) if statement_count else None
    statement_index = 0

    def statement(text, identifier=None):
        nonlocal statement_index
        # Independent per-statement draws, shared across all supervision classes.
        tagged = (rng.random() < 0.65 or statement_index == required_tag) and variant % 2 == 1
        statement_index += 1
        if tagged:
            if identifier is None:
                while True:
                    identifier = "".join(rng.choices(string.ascii_uppercase, k=2))
                    if identifier not in used_ids:
                        used_ids.add(identifier)
                        break
            text += f" ({identifier})"
        return text + "."

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
        family = rng.randrange(8)
        grammars.add(f"shared-{operation.mode}-{family}")
        command = (" или " if ru else " or ").join(
            _command(value, operation.mode, language, family) for value in operation.candidates
        )
        if len(operation.candidates) > 1 and rng.random() < 0.5:
            command = ("Равнозначные варианты: " if ru else "Equivalent alternatives: ") + command
        chunks.append(statement(command, operation.identifier))
    # Chunking unknown/negative statements individually makes IDs optional for each
    # input class and prevents a positive-only '(ID)' marker from becoming a cue.
    for candidate in spec.uncertain:
        name = _name(candidate, language)
        if spec.kind == "conflict":
            text = (
                f"{name}: справка разрешает и запрещает; статус неизвестен"
                if ru
                else f"{name}: the help both enables and forbids this; status is unknown"
            )
        elif spec.kind == "conditional":
            text = (
                f"{name}: ввод доступен только с разрешением; оно неизвестно"
                if ru
                else f"{name} works only with permission; permission is unknown"
            )
        else:
            text = (
                f"{name}: доступность и режим неизвестны"
                if ru
                else f"{name}: availability and timing are unknown"
            )
        chunks.append(statement(text))
    for candidate in forbidden:
        name = _name(candidate, language)
        forms = (
            (
                f"Не используйте {name}: запрещено",
                f"Ввод {name} не поддерживается",
                f"Не нажимайте и не активируйте {name}: отключено",
            )
            if ru
            else (
                f"Do not use {name}: unsupported",
                f"Input {name} is disabled",
                f"Do not activate {name}: this input is not used",
            )
        )
        chunks.append(statement(rng.choice(forms)))
    offset = 2 * (int(hashlib.sha256(str(pair_seed).encode()).hexdigest(), 16) % 4)
    chunks.append(REMAINDER_FORMS[spec.remainder][language][(offset + variant) % 8])
    body = " ".join(chunks)
    return _WRAPPERS[split][language][variant % 2].format(body=body), sorted(grammars)


def wording_manifest(records):
    distributions = {}
    for split in ("train", "dev", "calibration", "sealed_test"):
        subset = [row for row in records if row["split"] == split]
        distributions[split] = {
            "grammar_records": dict(
                Counter(family for row in subset for family in row["grammar_families"])
            ),
            "anonymous_records": sum(not re.search(r"\([A-Z]{2}\)", row["text"]) for row in subset),
            "tagged_records": sum(bool(re.search(r"\([A-Z]{2}\)", row["text"])) for row in subset),
            "remainder_records": {
                language: dict(
                    Counter(
                        footer
                        for row in subset
                        if row["language"] == language
                        for forms in REMAINDER_FORMS.values()
                        for footer in forms[language]
                        if footer in row["text"]
                    )
                )
                for language in ("ru", "en")
            },
        }
    return {
        "version": CONTROL_TEXT_DATA_VERSION,
        "wording_version": 5,
        "semantic_generator_version": "carryrl.control_text_data.v4",
        "semantic_rng_stream_preserved": True,
        "template_families_disjoint": True,
        "template_family_definition": (
            "Outer wrapper composition only; grammar_families contain shared lexical cores."
        ),
        "outer_template_families_disjoint": True,
        "paraphrase_holdout": "Split-specific outer compositions; basic lexical primitives shared.",
        "lexical_primitives_shared_across_splits": True,
        "semantic_independence_claim": (
            "Splits separate effective mappings, not unseen basic words. "
            "Statistical model generalization requires separate measurement."
        ),
        "id_presence_semantics": (
            "First variant per language anonymous; optional per-statement IDs on second, "
            "using the same draws for positive, negative and uncertain statements; "
            "one uniformly selected statement is tagged to guarantee paired presence variation."
        ),
        "paired_remainder_wording_varies": True,
        "redundant_negative_wording": (
            "Complete references sometimes omit explicit negatives already covered by the footer; "
            "partial references retain every explicit negative."
        ),
        "wording_distributions": distributions,
    }


def generate_control_text_records(*, seed=20261020, scenario_groups=4000):
    from carryrl.rewards.control_text_data_v4 import generate_control_text_records as generate_v4

    return generate_v4(seed=seed, scenario_groups=scenario_groups, wording_version=5)


def prepare_control_text_dataset(output, *, seed=20261020, scenario_groups=4000):
    from carryrl.rewards.control_text_data_v4 import prepare_control_text_dataset as prepare_v4

    return prepare_v4(output, seed=seed, scenario_groups=scenario_groups, wording_version=5)
