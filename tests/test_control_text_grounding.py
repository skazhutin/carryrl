"""Lexical grounding and architecture regressions; no ML fitting or game metadata."""

import ast
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from carryrl.rewards.control_text import (
    CANDIDATES,
    GROUNDED_HEAD_ARCHITECTURE,
    LEGACY_HEAD_ARCHITECTURE,
    LOCAL_GROUNDED_HEAD_ARCHITECTURE,
    RESIDUAL_HEAD_ARCHITECTURE,
    FrozenTextControlReward,
    GroundedTextControlHead,
    LocalGroundedTextControlHead,
    ResidualTextControlHead,
    TextControlHead,
    _checked_calibration,
    calibrated_head_logits,
    load_text_control_reward,
    save_text_control_artifact,
    text_control_head_for_architecture,
)
from carryrl.rewards.control_text_grounding import build_grounding


class OffsetTokenizer:
    is_fast = True

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        offsets = [(0, 0), *((i, i + 1) for i in range(len(text))), (0, 0)]
        return {
            "input_ids": torch.arange(len(offsets))[None],
            "attention_mask": torch.ones(1, len(offsets), dtype=torch.long),
            "offset_mapping": torch.tensor([offsets]),
        }


def mentions(text):
    mask = build_grounding(text, OffsetTokenizer())
    return {name for name, row in zip(CANDIDATES, mask, strict=True) if row.any()}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A bird performs operation A, then operation D.", set()),
        ("Press a button, hold a handle, find a key.", set()),
        ("press the a key; key a", {"key:a"}),
        (
            "Hold A; do not press the D key. Клавиша F, клавиша Q.",
            {"key:a", "key:d", "key:f", "key:q"},
        ),
        ("WASD or W/A/S/D; A/D", {f"key:{key}" for key in "wasd"}),
        ("W/S — движение вверх и вниз в интерфейсе.", {"key:w", "key:s"}),
        ("Операция W, затем операция S.", set()),
        ("left-right arrows", {"key:left", "key:right"}),
        ("up and down arrow keys", {"key:up", "key:down"}),
        ("стрелка влево", {"key:left"}),
        ("стрелки=Left/Right/Up/Down", {"key:left", "key:right", "key:up", "key:down"}),
        ("arrows; стрелки", {"key:left", "key:right", "key:up", "key:down"}),
        ("A/Left and D/Right", {"key:a", "key:d", "key:left", "key:right"}),
        ("Space, пробел", {"key:space"}),
        ("LMB, ПКМ, middle mouse button", {"mouse:left", "mouse:right", "mouse:middle"}),
        ("левая кнопка мыши, правая кнопка мыши", {"mouse:left", "mouse:right"}),
        ("moving the mouse left", {"dx:-"}),
        ("движение мыши вниз", {"dy:+"}),
        ("mouse movement left/right", {"dx:-", "dx:+"}),
        ("horizontal mouse movement", {"dx:-", "dx:+"}),
        ("движение мыши по вертикали", {"dy:-", "dy:+"}),
        ("move the mouse in both axes", {"dx:-", "dx:+", "dy:-", "dy:+"}),
        ("scrolling the wheel down", {"wheel:-"}),
        ("прокрутка колеса вверх", {"wheel:+"}),
        ("mouse wheel", {"wheel:-", "wheel:+"}),
    ],
)
def test_lexical_spelling_sets(text, expected):
    assert mentions(text) == expected


@pytest.mark.parametrize("name", ["пробел", "пробела", "пробелу", "пробелом", "пробеле"])
def test_russian_space_singular_cases_keep_exact_spelling_offsets(name):
    text = f"({name}),"
    mask = build_grounding(text, OffsetTokenizer())
    expected = torch.zeros_like(mask)
    start = 1 + len("query: ") + 1
    expected[CANDIDATES.index("key:space"), start : start + len(name)] = True
    assert torch.equal(mask, expected)


@pytest.mark.parametrize(
    "text",
    ["пробельный", "пробелами", "пробелы", "xпробела", "пробелуx", "id_пробелом"],
)
def test_russian_space_case_spellings_require_word_boundaries(text):
    assert mentions(text) == set()


def test_timing_russian_press_and_release_offsets_survive_tokenizer_padding():
    text = (
        "Пролетай через просветы движущихся препятствий. Короткое нажатие пробела "
        "подбрасывает персонажа. Отпускай пробел перед следующим взмахом; "
        "не касайся препятствий и границ. Это полный список доступного управления; "
        "остальные клавиши, кнопки и колесо не поддерживаются."
    )
    # Independent string offsets from the actual timing instruction: retain both
    # the primary press mention and the later release mention, without labels.
    assert text[65:72] == "пробела"
    assert text[106:112] == "пробел"
    assert text[240:246] == "колесо"

    def padded_tokenizer(*args, **kwargs):
        encoded = OffsetTokenizer()(*args, **kwargs)
        # Even a nonempty padding offset overlapping a mention must be ignored.
        encoded["input_ids"] = torch.cat(
            (encoded["input_ids"], torch.zeros(1, 2, dtype=torch.long)), dim=1
        )
        encoded["attention_mask"] = torch.cat(
            (encoded["attention_mask"], torch.zeros(1, 2, dtype=torch.long)), dim=1
        )
        encoded["offset_mapping"] = torch.cat(
            (encoded["offset_mapping"], torch.tensor([[[72, 79], [0, 0]]])), dim=1
        )
        return encoded

    mask = build_grounding(text, padded_tokenizer)
    expected = torch.zeros_like(mask)
    # Character tokenizer prepends one special token and the seven-character
    # E5 prefix; trailing special and padding tokens receive no mentions.
    expected[CANDIDATES.index("key:space"), 73:80] = True
    expected[CANDIDATES.index("key:space"), 114:120] = True
    expected[CANDIDATES.index("wheel:-"), 248:254] = True
    expected[CANDIDATES.index("wheel:+"), 248:254] = True
    assert torch.equal(mask, expected)
    assert torch.equal(mask, build_grounding(text, OffsetTokenizer()))


@pytest.mark.parametrize(
    "english,russian,candidate",
    [
        ("left", "влево", "dx:-"),
        ("right", "вправо", "dx:+"),
        ("up", "вверх", "dy:-"),
        ("down", "вниз", "dy:+"),
    ],
)
def test_bare_mouse_direction_has_same_anchor_in_support_negation_and_unknown(
    english, russian, candidate
):
    for name, templates in (
        (
            f"mouse {english}",
            ("Use the {}", "Do not use the {}", "the {}: availability is unknown"),
        ),
        (
            f"мышь {russian}",
            ("Используй {}", "Не используй {}", "{}: доступность неизвестна"),
        ),
    ):
        for template in templates:
            text = template.format(name)
            mask = build_grounding(text, OffsetTokenizer())
            assert mentions(text) == {candidate}
            row = mask[CANDIDATES.index(candidate), 1 + len("query: ") : -1]
            anchored = "".join(char for char, selected in zip(text, row, strict=True) if selected)
            assert anchored == name


@pytest.mark.parametrize(
    "text,expected",
    [
        ("left mouse button", {"mouse:left"}),
        ("mouse left button", {"mouse:left"}),
        ("right mouse button", {"mouse:right"}),
        ("mouse right button", {"mouse:right"}),
    ],
)
def test_mouse_button_names_do_not_anchor_movement(text, expected):
    assert mentions(text) == expected


def test_all_repeated_and_negated_mentions_are_retained_without_labels():
    text = "Hold A; never press the A key; клавиша A."
    tokenizer = OffsetTokenizer()
    mask = build_grounding(text, tokenizer)
    indices = [i + 1 + len("query: ") for i, char in enumerate(text) if char == "A"]
    assert mask[CANDIDATES.index("key:a"), indices].all()
    assert mask[CANDIDATES.index("key:a")].sum() == 3
    assert not mask[:, : len("query: ") + 1].any() and not mask[:, -1].any()
    assert tokenizer.calls[0][0] == "query: " + text
    assert tokenizer.calls[0][1]["truncation"] is False
    assert torch.equal(mask, build_grounding(text.replace("never", "often"), tokenizer))


def test_strict_offsets_and_feature_alignment():
    tokenizer = OffsetTokenizer()
    count = build_grounding("Hold A", tokenizer).shape[1]
    assert build_grounding("Hold A", tokenizer, expected_tokens=count).shape == (36, count)
    with pytest.raises(ValueError, match="token count"):
        build_grounding("Hold A", tokenizer, expected_tokens=count - 1)
    with pytest.raises(ValueError, match="never truncated"):
        build_grounding("A" * 510, tokenizer)
    tokenizer.is_fast = False
    with pytest.raises(ValueError, match="fast tokenizer"):
        build_grounding("Hold A", tokenizer)
    with pytest.raises(ValueError, match="offsets"):
        build_grounding("Hold A", lambda *args, **kwargs: {"input_ids": [[1]]})

    def invalid_offsets(*args, **kwargs):
        encoded = OffsetTokenizer()(*args, **kwargs)
        encoded["offset_mapping"][0, 2] = torch.tensor([-1, 1])
        return encoded

    with pytest.raises(ValueError, match="offsets"):
        build_grounding("Hold A", invalid_offsets)


def test_grounding_changes_candidate_binding_and_preserves_padding_and_pair_symmetry():
    torch.manual_seed(100)
    head = GroundedTextControlHead().eval()
    tokens = torch.randn(2, 5, 384)
    grounding = torch.zeros(2, 36, 5, dtype=torch.bool)
    grounding[:, 0, 1] = True
    grounding[:, 1, 3] = True
    original = head(tokens, grounding=grounding)
    swapped = grounding.clone()
    swapped[:, [0, 1]] = grounding[:, [1, 0]]
    changed = head(tokens, grounding=swapped)
    assert not torch.allclose(original["status"][:, :2], changed["status"][:, :2])
    torch.testing.assert_close(original["status"][:, 2:], changed["status"][:, 2:])
    torch.testing.assert_close(original["pair"], original["pair"].transpose(1, 2))
    padded = torch.cat((tokens, torch.randn(2, 4, 384)), dim=1)
    padding = torch.tensor([[False] * 5 + [True] * 4] * 2)
    padded_grounding = torch.nn.functional.pad(grounding, (0, 4))
    output = head(padded, padding, grounding=padded_grounding.float())
    for kind in original:
        torch.testing.assert_close(original[kind], output[kind], atol=1e-6, rtol=1e-5)
    original["pair"][:, 0, 1].sum().backward()
    assert head.projection.weight.grad.abs().sum() > 0
    assert head.mention_attention.in_proj_weight.grad.abs().sum() > 0


def test_missing_anchors_are_learned_inputs_and_invalid_grounding_fails():
    head = GroundedTextControlHead().eval()
    tokens = torch.randn(1, 4, 384)
    missing = torch.zeros(1, 36, 4, dtype=torch.bool)
    # Deliberately select supported/press/same with learned biases, including
    # every absent mention. No lexical rule overrides any classifier output.
    for name in ("status", "mode", "pair"):
        classifier = getattr(head, name)
        classifier[-1].weight.data.zero_()
        classifier[-1].bias.data.copy_(torch.tensor([10.0, -10.0, -10.0]))
    output = head(tokens, grounding=missing)
    assert all((logits.argmax(-1) == 0).all() for logits in output.values())
    with pytest.raises(ValueError, match="grounding"):
        head(tokens)
    with pytest.raises(ValueError, match="grounding"):
        head(tokens, grounding=missing[:, :, :3])
    with pytest.raises(ValueError, match="binary"):
        head(tokens, grounding=missing.float() + 0.5)
    with pytest.raises(ValueError, match="unpadded"):
        head(tokens, torch.ones(1, 4, dtype=torch.bool), grounding=missing)


def calibration():
    return {
        f"{kind}_{name}": value
        for kind in ("status", "mode", "pair")
        for name, value in (("threshold", 0.99), ("temperature", 1.0))
    }


def test_class_weight_correction_precedes_temperature_and_validates_optional_values():
    values = calibration()
    logits = torch.tensor([[1.0, 3.0, 4.0]])
    torch.testing.assert_close(calibrated_head_logits(logits, values, "status"), logits)
    values.update(status_log_class_weights=[1.0, 2.0, 3.0], status_temperature=2.0)
    assert _checked_calibration(values) == values
    torch.testing.assert_close(
        calibrated_head_logits(logits, values, "status"), torch.tensor([[0.0, 0.5, 0.5]])
    )
    for invalid in ([1.0], [1.0, 2.0, float("nan")], [1.0, True, 3.0]):
        with pytest.raises(ValueError, match="log_class_weights"):
            _checked_calibration({**values, "status_log_class_weights": invalid})


class ResourceEncoder(nn.Module):
    max_tokens = 512

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(384))
        self.tokenizer = OffsetTokenizer()

    def save_resources(self, directory):
        directory.mkdir()
        (directory / "tokenizer.fixture").write_text("offset-fixture")

    def encode_instruction_tokens(self, text):
        length = len(self.tokenizer("query: " + text)["input_ids"][0])
        return self.weight.detach()[None].expand(length, -1)


def passing_report():
    counts = dict(
        clear_count=1000,
        clear_accepted=1000,
        accepted_count=1000,
        accepted_correct=1000,
        accepted_scenarios=400,
        error_scenarios=0,
    )
    return {
        **counts,
        "critical_errors": 0,
        "signs": {kind: counts for kind in ("supported", "unsupported")},
        "modes": {
            kind: dict(clear_count=20, accepted_count=20, correct_count=20)
            for kind in ("held", "press")
        },
        "aliases": dict(clear_count=20, accepted_count=20, correct_count=20),
    }


@pytest.mark.parametrize(
    "architecture,head_type",
    [
        (LEGACY_HEAD_ARCHITECTURE, TextControlHead),
        (RESIDUAL_HEAD_ARCHITECTURE, ResidualTextControlHead),
        (GROUNDED_HEAD_ARCHITECTURE, GroundedTextControlHead),
        (LOCAL_GROUNDED_HEAD_ARCHITECTURE, LocalGroundedTextControlHead),
    ],
)
def test_factory_artifact_roundtrip_and_legacy_default(tmp_path, architecture, head_type):
    head = text_control_head_for_architecture(architecture)
    assert isinstance(head, head_type)
    encoder = ResourceEncoder()
    path = tmp_path / "artifact"
    manifest = save_text_control_artifact(
        path,
        encoder=encoder,
        head=head,
        calibration=calibration(),
        validation=passing_report(),
        training={"trained": True},
    )
    assert manifest["head_architecture"] == architecture
    if architecture == LEGACY_HEAD_ARCHITECTURE:
        manifest.pop("head_architecture")
        (path / "manifest.json").write_text(json.dumps(manifest))
    restored = load_text_control_reward(path, shared_encoder=encoder)
    assert type(restored.head) is head_type
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, restored.head.state_dict()[key])
    restored.predict("Hold A")
    exported = restored.save_resources(tmp_path / "exported")
    again = load_text_control_reward(exported, shared_encoder=encoder)
    assert again.model_hash == restored.model_hash


def test_runtime_passes_text_only_anchors_and_no_metadata_imports():
    class RecordingHead(GroundedTextControlHead):
        def forward(self, tokens, padding_mask=None, *, grounding=None):
            self.received = grounding
            return super().forward(tokens, padding_mask, grounding=grounding)

    head = RecordingHead()
    model = FrozenTextControlReward(
        ResourceEncoder(),
        head,
        model_hash="0" * 64,
        calibration=calibration(),
        calibration_hash="1" * 64,
        _verified=True,
    )
    model.predict("Hold A")
    assert head.received[0, CANDIDATES.index("key:a")].any()
    assert head.received.sum() == 1
    for filename in ("control_text.py", "control_text_grounding.py"):
        source = Path(__file__).parents[1] / "src/carryrl/rewards" / filename
        modules = [
            node.module or ""
            for node in ast.walk(ast.parse(source.read_text()))
            if isinstance(node, ast.ImportFrom)
        ]
        assert not any("control_text_data" in module or "envs" in module for module in modules)


@pytest.mark.parametrize(
    "text",
    [
        "Use arrow keys left/right.",
        "Управляй стрелками влево/вправо.",
        "Не используйте стрелки влево/вправо.",
    ],
)
def test_qualified_arrow_spellings_do_not_invent_vertical_mentions(text):
    from transformers import AutoTokenizer

    from carryrl.agent.language import TEXT_ID, TEXT_REVISION
    from carryrl.rewards.control_text import CANDIDATES
    from carryrl.rewards.control_text_grounding import build_grounding

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_ID, revision=TEXT_REVISION, local_files_only=True
    )
    mask = build_grounding(text, tokenizer).any(-1)
    for name in ("left", "right"):
        assert mask[CANDIDATES.index(f"key:{name}")]
    for name in ("up", "down"):
        assert not mask[CANDIDATES.index(f"key:{name}")]


@pytest.mark.parametrize(
    "text",
    [
        "Нажимай и отпускай E между действиями.",
        "Не нажимайте E; доступность не указана.",
        "Отпускайте E между нажатиями.",
    ],
)
def test_russian_inflected_action_verbs_keep_literal_key_mention(text):
    from transformers import AutoTokenizer

    from carryrl.agent.language import TEXT_ID, TEXT_REVISION
    from carryrl.rewards.control_text import CANDIDATES
    from carryrl.rewards.control_text_grounding import build_grounding

    tokenizer = AutoTokenizer.from_pretrained(
        TEXT_ID, revision=TEXT_REVISION, local_files_only=True
    )
    assert build_grounding(text, tokenizer)[CANDIDATES.index("key:e")].any()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hold the right mouse button or 1.", {"mouse:right", "key:1"}),
        ("Use R with sustained pressure.", {"key:r"}),
        ("Give 6 one brief press.", {"key:6"}),
        ("Выполните одиночное нажатие: E или 6.", {"key:e", "key:6"}),
        ("Выполни одиночное нажатие: 6 или 5.", {"key:6", "key:5"}),
        ("Используй 9 с удержанием.", {"key:9"}),
        ("Держи E или 7 нажатой.", {"key:e", "key:7"}),
        ("Движение выполняется через E или TAB.", {"key:e", "key:tab"}),
        ("Movement uses A or 2.", {"key:a", "key:2"}),
        ("Hold A and D / 5 or W.", {"key:a", "key:d", "key:5", "key:w"}),
        ("Hold the F key, Q and the 7 key or 9.", {"key:f", "key:q", "key:7", "key:9"}),
        ("Отпускай клавишу R и клавишу 2 или 7.", {"key:r", "key:2", "key:7"}),
        ("Use R or the left mouse button or 1.", {"key:r", "mouse:left", "key:1"}),
        ("Hold A or Left arrow and 6.", {"key:a", "key:left", "key:6"}),
        ("Do not use R or 3: unsupported.", {"key:r", "key:3"}),
        ("Input A or E: availability is unknown.", {"key:a", "key:e"}),
    ],
)
def test_compact_physical_names_and_coordinated_lists_keep_every_mention(text, expected):
    assert mentions(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Press a button and an input.", set()),
        ("Keep a marker inside goal 1; operation A or D [abc].", set()),
        ("The picture on the right is purple; a triangle marks progress.", set()),
        ("Press 7 for operation R [a1e]; then operation 9.", {"key:7"}),
        ("Use A; operation D and 6 [f84].", {"key:a"}),
        ("Do not use the 1 key, operation A.", {"key:1"}),
        ("Use a marker or an input for operation A [abc].", set()),
    ],
)
def test_compact_input_context_does_not_turn_articles_goals_or_ids_into_keys(text, expected):
    assert mentions(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Return to the room and press E.", {"key:e"}),
        ("Return your item to its place using the W key.", {"key:w"}),
        ("Enter the marked area with the W key.", {"key:w"}),
        ("Enter an empty room, then return home.", set()),
        ("Move into position and enter into the scene.", set()),
        ("Press Return to the menu.", {"key:enter"}),
        ("Never press Return to confirm.", {"key:enter"}),
        ("The Return key opens the menu.", {"key:enter"}),
        ("Use the Enter key to return to the area.", {"key:enter"}),
        ("Enter performs confirmation; Return to confirm also works.", {"key:enter"}),
        ("Enter to start; Return to confirm.", {"key:enter"}),
        ("Return is unavailable; Enter availability is unknown.", {"key:enter"}),
    ],
)
def test_ordinary_verbs_and_explicit_enter_key_names_remain_distinct(text, expected):
    assert mentions(text) == expected


def test_verb_filter_removes_only_the_ordinary_occurrence_of_the_same_spelling():
    from carryrl.rewards.control_text_grounding import _mention_spans

    text = "Return to the room; press Return to confirm, then enter the area."
    spans = _mention_spans(text)["key:enter"]
    assert {(start, text[start:end]) for start, end in spans} == {
        (text.index("Return", 1), "Return"),
    }


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Mouse motion changes yaw on the horizontal axis and pitch on the vertical axis.",
            {"dx:-", "dx:+", "dy:-", "dy:+"},
        ),
        (
            "Мышь меняет азимут по горизонтали и угол наклона по вертикали.",
            {"dx:-", "dx:+", "dy:-", "dy:+"},
        ),
        ("Mouse motion changes yaw on the horizontal axis.", {"dx:-", "dx:+"}),
        ("Мышь меняет азимут по горизонтали.", {"dx:-", "dx:+"}),
        ("Мышь меняет угол наклона по вертикали.", {"dy:-", "dy:+"}),
        (
            "Mouse motion changes pitch on the vertical axis: availability is unknown.",
            {"dy:-", "dy:+"},
        ),
        ("Не используй мышь: она меняет азимут по горизонтали.", set()),
    ],
)
def test_generic_mouse_axis_spellings_preserve_only_named_axes(text, expected):
    assert mentions(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Keep the R key down throughout the action.", {"key:r"}),
        ("Leave the 9 key held down.", {"key:9"}),
        ("Keep the SHIFT key or the 3 key down.", {"key:shift", "key:3"}),
        ("Leave Up arrow held down.", {"key:up"}),
        ("Use the DOWN key.", {"key:down"}),
        ("key: down; key UP", {"key:down", "key:up"}),
        ("Удерживайте клавишу S нажатой.", {"key:s"}),
        ("All remaining controls are unsupported.", set()),
        ("All other keys and buttons are unknown.", set()),
    ],
)
def test_key_state_words_are_distinct_from_named_physical_arrow_keys(text, expected):
    assert mentions(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hold key left.", {"key:left"}),
        ("Press key down.", {"key:down"}),
        ("Tap the key up.", {"key:up"}),
        ("Use key right.", {"key:right"}),
        ("Hold the W key down.", {"key:w"}),
        ("Keep the R key down.", {"key:r"}),
    ],
)
def test_explicit_key_prefix_after_input_verb_differs_from_postfix_key_state(text, expected):
    assert mentions(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Move the mouse left and 1 target appears.", {"dx:-"}),
        ("Use the mouse left and 1 target appears.", {"dx:-"}),
        ("Scroll the wheel up and 1 point is awarded.", {"wheel:+"}),
        ("Hold the W key and 1 point is awarded.", {"key:w"}),
        ("Hold W and 1 target appears.", {"key:w"}),
        ("Удерживай W и 1 очко начисляется.", {"key:w"}),
        ("Hold the W key and 1.", {"key:w", "key:1"}),
        ("Hold the W key and 1 down throughout the action.", {"key:w", "key:1"}),
        ("Use W or 1 with sustained pressure.", {"key:w", "key:1"}),
        ("Give W or 1 one brief press.", {"key:w", "key:1"}),
        ("Hold W and the 1 key.", {"key:w", "key:1"}),
        ("Удерживай W или 1 до конца действия.", {"key:w", "key:1"}),
        ("Используй D или 2 для жемчужного сигнала.", {"key:d", "key:2"}),
        ("Use W or 1 for the coral signal.", {"key:w", "key:1"}),
    ],
)
def test_coordinated_bare_digits_require_an_input_name_not_a_counted_noun(text, expected):
    assert mentions(text) == expected


@pytest.mark.parametrize("direction,candidate", [("up", "wheel:+"), ("down", "wheel:-")])
@pytest.mark.parametrize(
    "spelling",
    [
        "wheel scrolling {}",
        "mouse wheel scrolling {}",
        "scrolling the wheel {}",
        "scroll the mouse wheel {}",
    ],
)
@pytest.mark.parametrize("context", ["Use {}.", "Do not use {}.", "{}: availability is unknown."])
def test_wheel_direction_survives_both_scroll_and_noun_word_orders(
    direction, candidate, spelling, context
):
    assert mentions(context.format(spelling.format(direction))) == {candidate}


@pytest.mark.parametrize("name,candidate", [("5", "key:5"), ("Q", "key:q"), ('"Q"', "key:q")])
@pytest.mark.parametrize("noun", ["key", "KEY", "Key"])
@pytest.mark.parametrize("separator", [": ", " - ", " — "])
@pytest.mark.parametrize("article", ["a", "A"])
def test_key_noun_after_named_key_does_not_rebind_following_article(
    name, candidate, noun, separator, article
):
    text = f"{name} {noun}{separator}{article} brief press, then release."
    assert mentions(text) == {candidate}


@pytest.mark.parametrize("direction", ["влево", "вправо", "вверх", "вниз"])
@pytest.mark.parametrize(
    "template",
    [
        "Двигайте маркер {direction} клавишей {spelling} или клавишей LEFT.",
        "Для движения {direction} клавишу {spelling} удерживайте.",
        "Для движения {direction} клавиши {spelling} и LEFT равнозначны.",
        "Двигайте маркер {direction} КЛАВИШЕЙ {spelling}.",
    ],
)
@pytest.mark.parametrize("spelling", ["A", "a", '"A"', "«a»"])
def test_russian_explicit_a_after_direction_keeps_only_its_literal_anchor(
    direction, template, spelling
):
    text = template.format(direction=direction, spelling=spelling)
    mask = build_grounding(text, OffsetTokenizer())
    row = mask[CANDIDATES.index("key:a"), 1 + len("query: ") : -1]
    start = text.index(spelling) + (1 if len(spelling) > 1 else 0)
    expected = torch.zeros(len(text), dtype=torch.bool)
    expected[start] = True
    assert torch.equal(row, expected)


@pytest.mark.parametrize("direction", ["LEFT", "RIGHT", "UP", "DOWN"])
@pytest.mark.parametrize("noun", ["key", "KEY", "hotkey", "HOTKEY"])
@pytest.mark.parametrize("article", ["a", "A"])
def test_english_direction_key_suffix_does_not_anchor_a_following_article(
    direction, noun, article
):
    text = f"{direction} {noun}: {article} brief press, then release."
    assert mentions(text) == {"key:" + direction.lower()}


@pytest.mark.parametrize(
    "spelling",
    [
        "key a",
        "key A",
        "key: a",
        "KEY: A",
        "key - a",
        "Key — A",
        "the a key",
        "A key",
        'key "a"',
        "key: 'a'",
        'the "a" key',
        "the 'A' key",
        "the «a» key",
    ],
)
def test_explicit_a_key_grammar_retains_only_the_literal_name_span(spelling):
    mask = build_grounding(spelling, OffsetTokenizer())
    assert mentions(spelling) == {"key:a"}
    row = mask[CANDIDATES.index("key:a"), 1 + len("query: ") : -1]
    anchored = "".join(char for char, selected in zip(spelling, row, strict=True) if selected)
    assert anchored in ("a", "A")


def test_article_filter_does_not_suppress_a_later_explicit_a_key():
    text = "5 key: a brief press. Then press key: a."
    mask = build_grounding(text, OffsetTokenizer())
    row = mask[CANDIDATES.index("key:a"), 1 + len("query: ") : -1]
    assert mentions(text) == {"key:5", "key:a"}
    assert row.sum() == 1 and row[text.rindex("a")]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hotkey: Q.", {"key:q"}),
        ("Do not use hotkeys Q and E.", {"key:q", "key:e"}),
        ("Use hotkey A.", {"key:a"}),
        ("Use hotkey Q or 5 with sustained pressure.", {"key:q", "key:5"}),
        ("Hotkey: 'a'.", {"key:a"}),
        ("The Q hotkey and E are available.", {"key:q", "key:e"}),
        ("Monkey: Q.", set()),
        ("Do not use monkeys Q and E.", set()),
        ("Use monkey A.", set()),
    ],
)
def test_explicit_hotkey_nouns_preserve_key_grammar_without_word_substrings(text, expected):
    assert mentions(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Press a key: A.", {"key:a"}),
        ("Use a key: a.", {"key:a"}),
        ("Press a key:'A'.", {"key:a"}),
        ('Use a key: "a".', {"key:a"}),
        ("Press a key:A and Q.", {"key:a", "key:q"}),
    ],
)
def test_article_before_key_noun_does_not_hide_its_explicit_name(text, expected):
    assert mentions(text) == expected


@pytest.mark.parametrize("name", ["A", "the a", '"a"', "'a'", "«a»"])
def test_a_key_suffix_still_rejects_a_later_article(name):
    text = f"{name} key: a brief press."
    mask = build_grounding(text, OffsetTokenizer())
    row = mask[CANDIDATES.index("key:a"), 1 + len("query: ") : -1]
    assert mentions(text) == {"key:a"}
    assert row.sum() == 1 and not row[text.rindex("a")]


@pytest.mark.parametrize(
    "mouse_direction,mouse_candidate,key_direction,key_candidate",
    [
        ("up", "dy:-", "Right", "key:right"),
        ("down", "dy:+", "Left", "key:left"),
        ("left", "dx:-", "Down", "key:down"),
        ("right", "dx:+", "Up", "key:up"),
    ],
)
@pytest.mark.parametrize("connector", [" and ", " / ", " or ", ", ", "; ", " — "])
@pytest.mark.parametrize("noun", ["arrow", "key"])
@pytest.mark.parametrize("reverse", [False, True])
def test_typed_mouse_and_keyboard_directions_do_not_cross_coordinate_boundary(
    mouse_direction, mouse_candidate, key_direction, key_candidate, connector, noun, reverse
):
    parts = [f"mouse movement {mouse_direction}", f"{key_direction} {noun}"]
    if reverse:
        parts.reverse()
    text = connector.join(parts)
    mask = build_grounding(text, OffsetTokenizer())
    assert mentions(text) == {mouse_candidate, key_candidate}
    mouse_row = mask[CANDIDATES.index(mouse_candidate)]
    key_row = mask[CANDIDATES.index(key_candidate)]
    assert not (mouse_row & key_row).any()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("mouse movement up and right", {"dy:-", "dx:+"}),
        ("moving the mouse left/down", {"dx:-", "dy:+"}),
        ("Up and Right arrows", {"key:up", "key:right"}),
        ("arrow keys: Left / Down", {"key:left", "key:down"}),
        ("mouse up and Right arrow", {"dy:-", "key:right"}),
        ("moving the mouse up / Right arrow", {"dy:-", "key:right"}),
        ("wheel scrolling up and Right arrow", {"wheel:+", "key:right"}),
        ("scrolling the wheel down / Up arrow", {"wheel:-", "key:up"}),
        ("mouse movement up and left mouse button", {"dy:-", "mouse:left"}),
        ("arrows up and right mouse button", {"key:up", "mouse:right"}),
        ("mouse movement up and key: Right", {"dy:-", "key:right"}),
        ("движение мыши вверх и стрелка вправо", {"dy:-", "key:right"}),
        ("стрелка вверх и движение мыши вправо", {"key:up", "dx:+"}),
    ],
)
def test_typed_direction_qualifiers_preserve_homogeneous_and_mixed_lists(text, expected):
    assert mentions(text) == expected


@pytest.mark.parametrize(
    "text,candidate",
    [
        ("mouse movement upwards", "dy:-"),
        ("mouse motion downwards", "dy:+"),
        ("moving the mouse leftwards", "dx:-"),
        ("mouse movement rightwards", "dx:+"),
    ],
)
def test_existing_mouse_direction_adverbs_keep_their_single_axis(text, candidate):
    assert mentions(text) == {candidate}
