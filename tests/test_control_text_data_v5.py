"""V5 varies lexical scope and IDs while retaining the exact v4 semantic stream."""

import hashlib
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from carryrl.rewards.control_text_data_v4 import (
    ControlSpec,
    OperationSpec,
    annotate_spec,
)
from carryrl.rewards.control_text_data_v4 import (
    generate_control_text_records as generate_v4,
)
from carryrl.rewards.control_text_data_v5 import (
    REMAINDER_FORMS,
    generate_control_text_records,
    prepare_control_text_dataset,
    render_spec,
)


@pytest.fixture(scope="module")
def records():
    return generate_control_text_records(seed=20261022)


def test_semantic_stream_and_v4_default_unchanged(records):
    legacy = generate_v4(seed=20261022)
    assert legacy == generate_v4(seed=20261022, wording_version=4)
    assert len(legacy) == len(records) == 16000
    fields = (
        "group_id",
        "split",
        "mapping_id",
        "semantic_mapping_id",
        "declared_candidates",
        "labels",
        "specification",
        "scenario_kind",
        "language",
    )
    for old, new in zip(legacy, records, strict=True):
        assert {key: old[key] for key in fields} == {key: new[key] for key in fields}
    for old in legacy:
        if old["specification"]["remainder"] == "unsupported":
            assert old["text"].endswith(
                ("All other controls are disabled.", "Все остальные команды отключены.")
            )
    assert records == generate_v4(seed=20261022, wording_version=5)


def test_every_group_has_paired_remainder_and_id_variation(records):
    for start in range(0, len(records), 4):
        group = records[start : start + 4]
        for left, right in ((group[0], group[1]), (group[2], group[3])):
            assert not re.search(r"\([A-Z]{2}\)", left["text"])
            assert re.search(r"\([A-Z]{2}\)", right["text"])
            forms = REMAINDER_FORMS[left["specification"]["remainder"]][left["language"]]
            left_footer = [footer for footer in forms if footer in left["text"]]
            right_footer = [footer for footer in forms if footer in right["text"]]
            assert len(left_footer) == len(right_footer) == 1
            assert left_footer != right_footer
            assert left["labels"] == right["labels"]


def test_every_split_shares_mode_and_scope_lexical_primitives(records):
    for split in ("train", "dev", "calibration", "sealed_test"):
        subset = [row for row in records if row["split"] == split]
        for language in ("ru", "en"):
            local = [row for row in subset if row["language"] == language]
            joined = "\n".join(row["text"] for row in local)
            for status in ("unsupported", "unknown"):
                assert all(footer in joined for footer in REMAINDER_FORMS[status][language])
            if language == "en":
                for primitive in (
                    "Hold ",
                    "held",
                    "down",
                    "continuously",
                    "sustained pressure",
                    "throughout",
                    "Press ",
                    "Tap ",
                    "one",
                    "single",
                    "brief",
                    "release",
                ):
                    assert primitive in joined
            else:
                for primitive in (
                    "Удерживайте",
                    "нажатой",
                    "Не отпускайте",
                    "Непрерывно",
                    "один раз",
                    "одиночное нажатие",
                    "Коротко",
                    "отпустите",
                ):
                    assert primitive in joined
            grammars = {family for row in local for family in row["grammar_families"]}
            assert {
                f"shared-{mode}-{index}" for mode in ("held", "press") for index in range(8)
            } <= grammars


def test_anonymous_basic_forms_and_aliases_are_in_actual_corpus(records):
    for split in ("train", "dev", "calibration", "sealed_test"):
        subset = [row for row in records if row["split"] == split and row["language"] == "en"]
        assert any("Hold the W key." in row["text"] for row in subset)
        assert any(
            "Press " in row["text"]
            and " once." in row["text"]
            and not re.search(r"\([A-Z]{2}\)", row["text"])
            for row in subset
        )
        assert any(
            " or " in row["text"] and not re.search(r"\([A-Z]{2}\)", row["text"])
            for row in subset
            if row["scenario_kind"] == "aliases"
        )


def test_ids_do_not_determine_status_or_mode():
    spec = ControlSpec(
        (OperationSpec("AA", ("key:w",), "held"), OperationSpec("BB", ("key:a",), "press")),
        ("key:q",),
        ("key:e",),
        "unknown",
        "incomplete",
    )
    labels = annotate_spec(spec)
    tagged = Counter()
    for seed in range(400):
        text, _ = render_spec(spec, language="en", variant=1, split="train", seed=seed)
        for candidate in ("W", "A", "Q", "E"):
            tagged[candidate] += bool(re.search(rf"the {candidate} key[^.]*\([A-Z]{{2}}\)", text))
        assert annotate_spec(spec) == labels
    assert all(0.63 < count / 400 < 0.85 for count in tagged.values())
    assert max(tagged.values()) - min(tagged.values()) < 50


def test_remainder_words_never_relabel_explicit_positives():
    for mode in ("held", "press"):
        spec = ControlSpec(
            (OperationSpec("AA", ("key:w",), mode),), (), ("key:q",), "unsupported", "complete"
        )
        labels = {label["candidate"]: label for label in annotate_spec(spec)}
        assert labels["key:w"]["status"] == "supported"
        assert labels["key:w"]["mode"] == mode
        assert labels["key:a"]["status"] == "unsupported"
        for pair_seed in range(30):
            text, _ = render_spec(
                spec, language="en", variant=0, split="train", seed=0, pair_seed=str(pair_seed)
            )
            assert "the W key" in text
            assert "(AA)" not in text
            assert annotate_spec(spec) == list(labels.values())


def test_concise_single_binding_references_and_partial_negative_evidence():
    spec = ControlSpec(
        (OperationSpec("AA", ("key:w",), "held"),), (), ("key:q",), "unsupported", "complete"
    )
    texts = [
        render_spec(
            spec, language="en", variant=0, split="train", seed=seed, pair_seed="scope-pair"
        )[0]
        for seed in range(200)
    ]
    assert any(text.startswith("Hold the W key. ") and "the Q key" not in text for text in texts)
    assert any("the Q key" in text for text in texts)
    partial = ControlSpec(spec.operations, (), spec.forbidden, "unknown", "incomplete")
    assert all(
        "the Q key"
        in render_spec(partial, language="en", variant=variant, split="train", seed=seed)[0]
        for seed in range(20)
        for variant in range(4)
    )


def test_texts_and_semantics_do_not_conflict_and_calibration_is_large(records):
    texts, semantics = defaultdict(set), defaultdict(set)
    for row in records:
        texts[row["text"]].add(row["semantic_mapping_id"])
        semantics[row["semantic_mapping_id"]].add(row["split"])
    assert all(len(values) == 1 for table in (texts, semantics) for values in table.values())
    assert len(semantics) == 4000
    calibration = [row for row in records[::4] if row["split"] == "calibration"]
    for status in ("supported", "unsupported"):
        assert (
            sum(any(label["status"] == status for label in row["labels"]) for row in calibration)
            >= 350
        )


def test_manifest_is_honest_authenticated_and_untrained(tmp_path):
    manifest = prepare_control_text_dataset(tmp_path / "v5", seed=20261022, scenario_groups=100)
    assert manifest == json.loads((tmp_path / "v5/manifest.json").read_text())
    assert manifest["version"] == "carryrl.control_text_data.v5"
    assert (
        manifest["records_sha256"]
        == hashlib.sha256((tmp_path / "v5/records.jsonl").read_bytes()).hexdigest()
    )
    assert manifest["semantic_rng_stream_preserved"] is True
    assert manifest["lexical_primitives_shared_across_splits"] is True
    assert manifest["model_input_fields"] == ["text"]
    assert manifest["readiness"] is manifest["ppo_eligible"] is manifest["model_loaded"] is False
    assert manifest["updates"] == 0
    assert manifest["wording_distributions"]["train"]["anonymous_records"] > 0
    assert manifest["wording_distributions"]["train"]["tagged_records"] > 0


def test_cli_supports_v5(tmp_path, capsys):
    script_spec = importlib.util.spec_from_file_location(
        "prepare_control_v5_cli",
        Path(__file__).resolve().parents[1] / "scripts/prepare_text_control_reward.py",
    )
    script = importlib.util.module_from_spec(script_spec)
    script_spec.loader.exec_module(script)
    assert (
        script.main(
            [
                "--output",
                str(tmp_path / "v5"),
                "--scenario-groups",
                "100",
                "--corpus-version",
                "v5",
                "--seed",
                "20261022",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["version"] == "carryrl.control_text_data.v5"


def test_all_references_fit_local_e5_context(records):
    from transformers import AutoTokenizer

    from carryrl.agent.language import TEXT_ID, TEXT_REVISION

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            TEXT_ID, revision=TEXT_REVISION, local_files_only=True
        )
    except OSError:
        pytest.skip("pinned local tokenizer unavailable")
    tokens = tokenizer(
        ["query: " + row["text"] for row in records], padding=False, truncation=False
    )
    assert max(map(len, tokens["input_ids"])) <= 512


@pytest.mark.parametrize("version", (3, 6, "5", True))
def test_invalid_wording_version_rejected(version):
    with pytest.raises(ValueError):
        generate_v4(scenario_groups=100, wording_version=version)
