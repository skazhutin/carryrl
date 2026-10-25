"""Lexical control mentions aligned to frozen E5 tokens, without semantic labels.

This module identifies spellings only. Negation, uncertainty, action mode, aliases
and whether an unmentioned control is supported remain learned from the text.
"""

import re

import torch

from carryrl.agent.actions import KEY_VOCAB, MOUSE_BUTTONS

# The canonical order is the physical-action vocabulary, never a game control map.
_CANDIDATES = (
    tuple(f"key:{key}" for key in KEY_VOCAB)
    + tuple(f"mouse:{button}" for button in MOUSE_BUTTONS)
    + tuple(f"{axis}:{sign}" for axis in ("dx", "dy", "wheel") for sign in ("-", "+"))
)
_KEY_NAMES = {
    "space": r"space(?:bar)?|пробел(?:а|у|ом|е)?",
    "shift": r"shift|шифт",
    "ctrl": r"ctrl|control|контрол",
    "escape": r"escape|esc|эскейп",
    "tab": r"tab|таб",
    "enter": r"enter|return|энтер",
    "left": r"left|влево",
    "right": r"right|вправо",
    "up": r"up|вверх",
    "down": r"down|вниз",
}
_ARROW = (
    r"(?:arrow\s+keys?|arrows?|стрелк(?:ами|ах|ой|ою|ам|а|и|у|ы|е)|стрелок|"
    r"стрелочн\w*\s+клавиш\w*)(?!\w)"
)
_DIRECTION = r"(?:left|right|up|down|влево|вправо|вверх|вниз)"
_JOIN = r"\s*(?:[/,;−–—-]|and|or|и|или)\s*"


def _ordinary_enter_verb(text, match):
    """Disambiguate a bare spelling only; explicit key contexts remain valid."""
    word = match.group().lower()
    determiner = r"(?:the|an?|this|that|these|those|your|our|their|its)\b"
    tail = text[match.end():]
    if word == "return":
        return bool(re.match(rf"\s+(?:to\s+{determiner}|{determiner}|"
                             r"(?:back|home|here|there)\b)", tail, re.I))
    if word == "enter":
        return bool(re.match(rf"\s+(?:{determiner}|into\b)", tail, re.I))
    return False


def _compact_key_lists(text, spans):
    """Transmit literal key names through an input phrase and its coordination.

    Connectors delimit spelling lists only; they never supply alias relations.
    A lower-case English article is not a compact spelling of the A key.
    """
    spellings = {
        key: (r"(?-i:A)" if key == "a" else _KEY_NAMES.get(key, re.escape(key)))
        for key in KEY_VOCAB
        if key not in ("left", "right", "up", "down")
    }
    names = "|".join(f"(?P<name_{key}>{value})" for key, value in spellings.items())
    item = re.compile(
        rf"\s*(?:the\s+)?(?:(?:hotkeys?|keys?|клавиш\w*)\s+)?"
        rf"(?<!\w)(?:{names})(?!\w)(?:\s+(?:hotkeys?|keys?|клавиш\w*)\b)?",
        re.I,
    )
    context = re.compile(
        r"\b(?:hold|press|tap|release|uses?|keep|leave|give|input|"
        r"наж(?:м|им)\w*|удерж\w*|отпус[тк]\w*|держ(?:и|ите)|использ\w*|"
        r"выб(?:ери|ерите)|ввод\w*|выполняется\s+через|нажати\w*\s*:)\s*",
        re.I,
    )

    def record(match):
        group = match.lastgroup
        key = group.removeprefix("name_")
        spans[f"key:{key}"].append(match.span(group))
        return match.end()

    ends = [
        end
        for candidate, mentions in spans.items()
        if candidate.startswith(("key:", "mouse:"))
        for _, end in mentions
    ]
    for lead in context.finditer(text):
        match = item.match(text, lead.end())
        if match:
            ends.append(record(match))
    # Explicit names may end before their key/arrow noun. Consume only that
    # lexical suffix, then require a connector and another physical key name.
    suffix = re.compile(r"\s+(?:arrow\s+keys?|arrows?|hotkeys?|keys?|клавиш\w*)\b", re.I)
    link = re.compile(r"\s*(?:[/,]|(?:and|or|или|и)\b)\s*", re.I)
    # A bare digit followed by an ordinary noun is a quantity ("1 target"),
    # not another key name. Explicit key nouns and input postmodifiers remain
    # valid list syntax; recognizing this boundary assigns no action mode.
    numeric_tail = re.compile(
        r"\s+(?:and|or|или|и|with|for|on|until|throughout|once|down|up|held|"
        r"one|single|brief|is|are|was|were|с|до|для|между|нажат\w*|однократн\w*|один)\b",
        re.I,
    )

    def coordinated_name(match):
        if not match.group(match.lastgroup).isdigit():
            return True
        group = match.lastgroup
        explicit_noun = re.search(r"\b(?:hotkeys?|keys?|клавиш\w*)\b", match.group(), re.I)
        tail_is_word = re.match(r"\s+[^\W\d_]", text[match.end():])
        return bool(explicit_noun or not tail_is_word or numeric_tail.match(text, match.end(group)))

    visited = set()
    while ends:
        end = ends.pop()
        if end in visited:
            continue
        visited.add(end)
        noun = suffix.match(text, end)
        connector = link.match(text, noun.end() if noun else end)
        match = item.match(text, connector.end()) if connector else None
        if match and coordinated_name(match):
            ends.append(record(match))


def _mention_spans(text):
    spans = {candidate: [] for candidate in _CANDIDATES}

    def add(candidate, match, group=0):
        spans[candidate].append(match.span(group))

    def scan(candidate, pattern, group=0):
        for match in re.finditer(pattern, text, re.I):
            add(candidate, match, group)

    # A key noun already following a physical spelling is a suffix, not a
    # second prefix: "5 key: a brief press" does not name the A key.
    spellings = "|".join(
        r"(?:(?-i:A)|the\s+a|[\"'«“‘]a(?=[\"'»”’]))"
        if key == "a" else _KEY_NAMES.get(key, re.escape(key))
        for key in KEY_VOCAB
    )
    named_key_suffix = re.compile(
        rf"(?<!\w)(?:{spellings})[\"'»”’]?\s*$", re.I
    )
    opening_quote = r"[\"'«“‘]?"
    closing_quote = r"[\"'»”’]?"

    # A single letter needs keyboard context; English articles and operation A
    # are deliberately not bare-key mentions. Every occurrence is retained.
    for key in KEY_VOCAB:
        spelling = _KEY_NAMES.get(key, re.escape(key))
        token = rf"(?<!\w)(?:{spelling})(?!\w)"
        if key in ("left", "right", "up", "down"):
            # "Keep the R key down" describes R's state, not the Down key.
            # Direction names before "key", explicit lists/separators and
            # uppercase prefix spellings still identify physical arrow keys.
            prefix = r"(?:keys?\s*[=:]\s*|keys\s+|клавиш\w*\s*[=:]?\s*)"
            scan(f"key:{key}", rf"{prefix}({token})", 1)
            scan(f"key:{key}", rf"\bkey\s+((?-i:{key.upper()}))(?!\w)", 1)
            scan(
                f"key:{key}",
                rf"\b(?:hold|press|tap|release|use|keep|leave)\s+(?:the\s+)?key\s+({token})",
                1,
            )
        else:
            prefix = rf"\b(?:hotkeys?|keys?|клавиш\w*)\s*[=:–—-]?\s*{opening_quote}({token})"
            for match in re.finditer(prefix, text, re.I):
                # Only English key nouns can precede an article: a Russian
                # "клавишей A" explicitly names A even after a direction word.
                if (
                    key == "a"
                    and match.group().lower().startswith(("key", "hotkey"))
                    and named_key_suffix.search(text[:match.start()])
                ):
                    continue
                add(f"key:{key}", match, 1)
        # Lowercase English "a key" / "press a button" contains an article.
        # A named lowercase a still has explicit context in "key a" or "the a key".
        command_token = (
            r"(?<!\w)(?:(?-i:A)|(?<=[\"'«“‘])a(?=[\"'»”’]))(?!\w)"
            if key == "a" else token
        )
        scan(
            f"key:{key}",
            rf"({command_token}){closing_quote}\s+(?:hotkeys?|keys?|клавиш\w*)\b",
            1,
        )
        if key == "a":
            scan("key:a", r"\bthe\s+(a)\s+key\b", 1)
        if key not in ("left", "right", "up", "down"):
            scan(
                f"key:{key}",
                rf"(?:hold|press|tap|release|наж(?:м|им|ат)\w*|удерж\w*|отпус[тк]\w*)"
                rf"\s+({command_token})",
                1,
            )
        if len(key) > 1 and key not in ("left", "right", "up", "down"):
            for match in re.finditer(token, text, re.I):
                # "Return to the room" and "enter the area" name actions in
                # the scene. Earlier qualified scans still retain a literal
                # key in "press Return to the menu" or "the Enter key".
                if key != "enter" or not _ordinary_enter_verb(text, match):
                    add(f"key:{key}", match)
    for match in re.finditer(r"(?<!\w)WASD(?!\w)", text, re.I):
        for key in "wasd":
            add(f"key:{key}", match)
    # Conventional slash key lists bind each spelling to its own span. A/D is
    # a control spelling in its own right, as is W/S; arbitrary operation names are not.
    listed = (
        r"(?:W\s*/\s*A\s*/\s*S\s*/\s*D|A\s*/\s*D|W\s*/\s*S|"
        r"[WASD]\s*/\s*(?:Left|Right|Up|Down))"
    )
    for match in re.finditer(rf"(?<!\w){listed}(?!\w)", text, re.I):
        for part in re.finditer(r"[A-Za-z]+", match.group()):
            key = part.group().lower()
            spans[f"key:{key}"].append((match.start() + part.start(), match.start() + part.end()))

    motion = (
        r"(?:mov\w*\s+(?:the\s+)?mouse|mouse\s+(?:mov\w*|motion)|"
        r"движ\w*\s+мыш\w*|двиг\w*\s+мыш\w*)"
    )
    axis_name = r"(?:horizontal\w*|vertical\w*|горизонтал\w*|вертикал\w*)"
    motion_direction = (
        rf"{_DIRECTION}"
        rf"(?!\s+(?:{_ARROW}|(?:hotkeys?|keys?|клавиш\w*|(?:mouse\s+)?buttons?)\b))"
    )
    motion_pattern = (
        rf"(?:(?P<axis_before>{axis_name})\s+)?{motion}"
        rf"(?:\s+(?:to\s+the\s+)?(?P<directions>{motion_direction}(?:{_JOIN}{motion_direction})*)"
        rf"|\s+(?:по\s+)?(?P<axis_after>{axis_name}))?"
    )
    bare_motion = (
        rf"\b(?:mouse|мыш(?:ь|и|ью))\s+(?P<direction>{_DIRECTION})"
        rf"(?!\w)(?!\s+(?:buttons?|кнопк\w*)\b)"
    )
    wheel = r"(?:(?:scroll\w*\s+(?:the\s+)?)?(?:mouse\s+)?wheel|(?:прокрутк\w*\s+)?кол[её]с\w*)"
    wheel_direction = re.compile(r"\s+(?:scroll\w*\s+)?(up|down|вверх|вниз)\b", re.I)
    motion_matches = list(re.finditer(motion_pattern, text, re.I))
    bare_motion_matches = list(re.finditer(bare_motion, text, re.I))
    # A shared connector cannot transfer an explicitly typed mouse/wheel
    # direction to an arrow noun in the next control phrase.
    other_direction_starts = set()
    for match in motion_matches:
        if match.group("directions"):
            other_direction_starts.update(
                match.start("directions") + part.start()
                for part in re.finditer(_DIRECTION, match.group("directions"), re.I)
            )
    other_direction_starts.update(match.start("direction") for match in bare_motion_matches)
    for match in re.finditer(wheel, text, re.I):
        direction = wheel_direction.match(text, match.end())
        if direction:
            other_direction_starts.add(direction.start(1))

    # Qualified arrow lists use exactly the named subset. Unqualified arrows
    # spell the conventional four-key group. These are mentions, not aliases.
    arrow_pattern = (
        rf"(?:(?P<before>{_DIRECTION}(?:{_JOIN}{_DIRECTION})*)\s+)?{_ARROW}"
        rf"(?:\s*(?:[=:]\s*)?(?P<after>{_DIRECTION}(?:{_JOIN}{_DIRECTION})*))?"
    )
    direction_keys = {
        "left": "left",
        "right": "right",
        "up": "up",
        "down": "down",
        "влево": "left",
        "вправо": "right",
        "вверх": "up",
        "вниз": "down",
    }
    for match in re.finditer(arrow_pattern, text, re.I):
        group = "before" if match.group("before") else "after" if match.group("after") else None
        if group:
            for part in re.finditer(_DIRECTION, match.group(group), re.I):
                start = match.start(group) + part.start()
                if start in other_direction_starts or re.match(
                    r"\s+(?:mouse\s+)?buttons?\b", text[match.start(group) + part.end():], re.I
                ):
                    continue
                spans[f"key:{direction_keys[part.group().lower()]}"].append(
                    (match.start(group) + part.start(), match.start(group) + part.end())
                )
        else:
            for key in ("left", "right", "up", "down"):
                add(f"key:{key}", match)

    for button, english, russian, abbreviation in (
        ("left", "left", "лев", "LMB|ЛКМ"),
        ("right", "right", "прав", "RMB|ПКМ"),
        ("middle", "middle|center", "средн|центральн", "MMB|СКМ"),
    ):
        scan(f"mouse:{button}", rf"(?<!\w)(?:{abbreviation})(?!\w)")
        scan(f"mouse:{button}", rf"\b(?:{english})\s+(?:mouse\s+)?buttons?\b")
        scan(f"mouse:{button}", rf"\bmouse\s+(?:{english})\s+buttons?\b")
        scan(f"mouse:{button}", rf"\b(?:{russian})\w*\s+кнопк\w*\s+мыш\w*")
        scan(f"mouse:{button}", rf"\bкнопк\w*\s+мыш\w*\s+(?:{russian})\w*")
    motion_candidates = {
        "left": "dx:-",
        "right": "dx:+",
        "up": "dy:-",
        "down": "dy:+",
        "влево": "dx:-",
        "вправо": "dx:+",
        "вверх": "dy:-",
        "вниз": "dy:+",
    }
    # Axis descriptions are physical mouse spellings, independent of whether
    # the surrounding text enables, forbids or leaves the input unspecified.
    axis_descriptions = []
    axis_phrase = (
        r"\b(?:mouse\s+(?:motion|movement)\s+changes?|мыш(?:ь|и|ью)\s+меня\w*)"
        r"\s+[^.!?\n]{0,160}"
    )
    for match in re.finditer(axis_phrase, text, re.I):
        axes = set()
        for word in re.finditer(axis_name, match.group(), re.I):
            axis = "dx" if word.group().lower().startswith(("horiz", "гориз")) else "dy"
            axes.add(axis)
        if axes:
            axis_descriptions.append((match.start(), match.end(), axes))
            for axis in axes:
                for sign in ("-", "+"):
                    add(f"{axis}:{sign}", match)
    # Noun-only spellings occur in support, negation and uncertainty alike.
    # Requiring the direction after the mouse noun keeps button names distinct.
    for match in bare_motion_matches:
        add(motion_candidates[match.group("direction").lower()], match)
    for match in motion_matches:
        if match.group("directions"):
            for part in re.finditer(_DIRECTION, match.group("directions"), re.I):
                add(motion_candidates[part.group().lower()], match)
        else:
            axis_word = (match.group("axis_before") or match.group("axis_after") or "").lower()
            axes = (
                ("dx",)
                if axis_word.startswith(("horiz", "гориз"))
                else ("dy",)
                if axis_word
                else ("dx", "dy")
            )
            if not axis_word:
                for start, end, described_axes in axis_descriptions:
                    if start <= match.start() < end:
                        axes = described_axes
                        break
            for axis in axes:
                for sign in ("-", "+"):
                    add(f"{axis}:{sign}", match)
    for match in re.finditer(wheel, text, re.I):
        direction = wheel_direction.match(text, match.end())
        signs = (
            ("+",)
            if direction and direction[1].lower() in ("up", "вверх")
            else ("-",)
            if direction
            else ("-", "+")
        )
        for sign in signs:
            add(f"wheel:{sign}", match)
    _compact_key_lists(text, spans)
    return spans


def build_grounding(text, tokenizer, *, expected_tokens=None):
    """Return bool [36,L] mention mask in strict ``query: `` E5 token order.

    Fast offset mapping is required. No truncation or special/padding token
    mentions are permitted. ``expected_tokens`` checks cached feature alignment.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("grounding requires nonempty instruction text")
    if getattr(tokenizer, "is_fast", True) is False:
        raise ValueError("grounding requires a fast tokenizer with offset mapping")
    encoded = tokenizer(
        "query: " + text,
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    if not {"input_ids", "offset_mapping", "attention_mask"} <= encoded.keys():
        raise ValueError("grounding requires tokenizer offsets and attention mask")
    offsets = torch.as_tensor(encoded["offset_mapping"])
    attention = torch.as_tensor(encoded["attention_mask"])
    ids = torch.as_tensor(encoded["input_ids"])
    if (
        offsets.shape != (*ids.shape, 2)
        or ids.ndim != 2
        or ids.shape[0] != 1
        or attention.shape != ids.shape
    ):
        raise ValueError("grounding tokenizer offsets are not aligned")
    if (
        offsets.dtype not in (torch.int32, torch.int64)
        or not ((attention == 0) | (attention == 1)).all()
        or (offsets < 0).any()
        or (offsets > len("query: " + text)).any()
        or (offsets[..., 1] < offsets[..., 0]).any()
    ):
        raise ValueError("grounding tokenizer offsets or attention mask are invalid")
    if ids.shape[1] > 512:
        raise ValueError("grounding requires at most 512 tokens; text is never truncated")
    offsets = offsets[0, attention[0].bool()].tolist()
    count = len(offsets)
    if not 0 < count <= 512:
        raise ValueError("grounding requires at most 512 tokens; text is never truncated")
    if expected_tokens is not None and count != expected_tokens:
        raise ValueError("grounding token count differs from contextual features")
    mask = torch.zeros(len(_CANDIDATES), count, dtype=torch.bool)
    prefix = len("query: ")
    spans = _mention_spans(text)
    for index, candidate in enumerate(_CANDIDATES):
        for start, end in spans[candidate]:
            for token, (left, right) in enumerate(offsets):
                if right > left and left < end + prefix and right > start + prefix:
                    mask[index, token] = True
    return mask
