"""New partial-field protocol, genuine contrast clusters and sealed allocation."""

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from carryrl.rewards.control_text_data_v14 import (
    CONTROL_CANDIDATES,
    HETEROGENEOUS_GROUP_QUOTAS,
    HETEROGENEOUS_KINDS,
    FactorizedSpec,
    Operation,
    annotate_factorized_spec,
    generate_factorized_rows,
    prepare_factorized_dataset,
    read_factorized_dataset,
    render_factorized_spec,
    validate_factorized_coverage,
)


@pytest.fixture(scope="module")
def full_rows():
    return generate_factorized_rows()


def test_partial_fields_are_independent_and_conditions_do_not_erase_support():
    spec = FactorizedSpec((Operation(("key:space",), "unspecified", "grounded"),
                           Operation(("key:e",), "press")), pair_knowledge=False)
    labels, pairs, _ = annotate_factorized_spec(spec)
    labels = {v["candidate"]: v for v in labels}
    assert labels["key:space"] == {"candidate": "key:space", "applicability": "supported",
                                   "mode": "unspecified"}
    assert labels["key:e"]["mode"] == "press"
    assert labels["key:w"]["applicability"] == "unspecified"
    assert pairs == [{"left": "key:e", "right": "key:space", "relation": "unspecified"}]
    complete = FactorizedSpec(spec.operations, complete=True)
    labels, _, _ = annotate_factorized_spec(complete)
    assert next(v for v in labels if v["candidate"] == "key:w")["applicability"] == "unsupported"


def test_alias_same_different_and_unspecified_are_separate():
    spec = FactorizedSpec((Operation(("key:a", "key:left"), "held"),
                           Operation(("key:d",), "unspecified")))
    _, pairs, _ = annotate_factorized_spec(spec)
    by_pair = {(v["left"], v["right"]): v["relation"] for v in pairs}
    assert by_pair["key:a", "key:left"] == "same"
    assert by_pair["key:a", "key:d"] == "different"
    _, uncertain, _ = annotate_factorized_spec(
        FactorizedSpec(spec.operations, pair_knowledge=False))
    assert {v["relation"] for v in uncertain} == {"same", "unspecified"}


def test_exact_budget_splits_views_and_independent_calibration(full_rows):
    coverage = validate_factorized_coverage(full_rows, expected_groups={
        "train": 2544, "dev": 600, "calibration": 400, "sealed_test": 600})
    assert coverage["rows"] == 24864
    assert coverage["semantic_groups"] == 4144
    assert coverage["dev_groups_by_role"] == {"temperature": 300, "selection": 300}
    assert coverage["required_combinations_present"]
    for split, expected in (("calibration", 400), ("sealed_test", 600)):
        groups = {r["group_id"]: r for r in full_rows if r["split"] == split}
        assert len({r["independent_group_id"] for r in groups.values()}) == expected
        assert all(r["contrast_cluster_id"] is None for r in groups.values())
        for sign in ("supported", "unsupported"):
            assert sum(any(v["applicability"] == sign for v in r["factorized_labels"])
                       for r in groups.values()) == expected
    train = [r for r in full_rows if r["split"] == "train"]
    assert len({r["independent_group_id"] for r in train}) == 1176


def test_actual_minimal_pairs_change_mode_or_completeness_in_one_partition(full_rows):
    clusters = defaultdict(dict)
    for row in full_rows:
        if row["contrast_cluster_id"]:
            clusters[row["contrast_cluster_id"]][row["group_id"]] = row
    assert len(clusters) == 1176 + 300
    for members in clusters.values():
        if next(iter(members.values()))["contrast_type"] == "relations":
            assert len(members) == 3
            continue
        a, b = members.values()
        assert a["split"] == b["split"]
        assert a["development_role"] == b["development_role"]
        assert a["factorized_labels"] != b["factorized_labels"]
        assert a["pair_labels"] == b["pair_labels"]
        assert a["specification"]["operations"][1:] == b["specification"]["operations"][1:]
        assert a["contrast_type"] == b["contrast_type"]
        if a["contrast_type"] == "completeness":
            assert a["specification"]["operations"] == b["specification"]["operations"]
            assert a["specification"]["complete"] != b["specification"]["complete"]
        else:
            assert a["specification"]["complete"] == b["specification"]["complete"]
    for split in ("train", "dev"):
        assert {r["contrast_type"] for r in full_rows if r["split"] == split} == {
            "mode", "completeness", *( ("relations",) if split == "train" else () )}


def test_fail_closed_missing_required_combination(full_rows):
    # Keep six-view groups but remove a required non-heterogeneous scenario.
    rows = [r for r in full_rows if not (r["split"] == "calibration"
                                         and r["scenario_kind"] == "morphology")]
    with pytest.raises(ValueError, match="missing required coverage"):
        validate_factorized_coverage(rows)


def test_fail_closed_malformed_witness_and_leaking_grammar(full_rows):
    rows = copy.deepcopy(full_rows[:6])
    rows[0]["expected_mentions"][0]["end"] = len(rows[0]["text"]) + 1
    with pytest.raises(ValueError, match="grounding witness"):
        validate_factorized_coverage(rows, require_full=False)
    train = full_rows[:6]
    dev = copy.deepcopy(next([r for r in full_rows if r["group_id"] == first["group_id"]]
                             for first in full_rows if first["split"] == "dev"))
    dev[0]["template_family"] = train[0]["template_family"]
    with pytest.raises(ValueError, match="template family"):
        validate_factorized_coverage(train + dev, require_full=False)


def test_generator_spans_are_authored_without_using_matcher(monkeypatch):
    import carryrl.rewards.control_text_grounding as grounding

    monkeypatch.setattr(grounding, "_mention_spans", lambda _: pytest.fail("matcher was used"))
    spec = FactorizedSpec((Operation(("key:space",), "press"),), kind="morphology")
    text, mentions = render_factorized_spec(spec, language="ru", grammar=0, variant=1, seed=1)
    assert [(m["candidate"], text[m["start"]:m["end"]]) for m in mentions] == [
        ("key:space", "пробела")]


def test_roundtrip_pins_and_rejects_tampering(tmp_path):
    counts = {"train": 2544, "dev": 600, "calibration": 400, "sealed_test": 600}
    output = tmp_path / "factorized"
    manifest = prepare_factorized_dataset(output, split_groups=counts)
    loaded, rows = read_factorized_dataset(output)
    assert loaded == manifest
    assert manifest["records"] == sum(counts.values()) * 6
    assert manifest["source_sha256"].keys() == {"control_text_data_v14.py",
                                               "control_text_grounding.py"}
    assert manifest["runtime_text_copied"] is False
    assert Counter(r["language"] for r in rows) == {"ru": 12432, "en": 12432}
    with pytest.raises(FileExistsError):
        prepare_factorized_dataset(output, split_groups=counts)
    with (output / "records.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="records hash mismatch"):
        read_factorized_dataset(output)


def test_runtime_strings_are_never_generator_inputs(full_rows):
    root = Path(__file__).resolve().parents[1]
    texts = json.loads((root / "configs/atomic_instructions.json").read_text())
    assert not {text for pair in texts.values() for text in pair} & {r["text"] for r in full_rows}
    assert all(tuple(v["candidate"] for v in r["factorized_labels"]) == CONTROL_CANDIDATES
               for r in full_rows)


def test_condition_and_no_condition_views_keep_identical_global_targets(full_rows):
    groups = defaultdict(list)
    for row in full_rows:
        if row["scenario_kind"] == "conditional":
            groups[row["group_id"]].append(row)
    for rows in groups.values():
        assert {r["rendering"]["condition_present"] for r in rows} == {False, True}
        assert all(r["factorized_labels"] == rows[0]["factorized_labels"] for r in rows)
        for row in rows:
            phrase = "Когда горит сигнал" if row["language"] == "ru" else "When the signal is lit"
            assert (phrase in row["text"]) == row["rendering"]["condition_present"]


def test_compact_only_views_do_not_repeat_individual_primary_bindings(full_rows):
    counts = Counter()
    for row in full_rows:
        if not row["rendering"]["compact_only"]:
            continue
        kind, text = row["scenario_kind"], row["text"]
        primary = {"key:a", "key:d", "key:left", "key:right"}
        allowed = {"A/D", "стрелки влево/вправо", "left/right arrows"}
        if kind == "wasd":
            primary |= {"key:w", "key:s", "key:up", "key:down"}
            allowed = {"WASD", "стрелки", "the arrow keys"}
            assert ("вверх, влево, вниз, вправо" in text
                    or "вверх, влево, вниз и вправо" in text
                    or "вверх, влево, вниз, вправо" in text
                    or "up, left, down, right" in text
                    or "up, left, down and right" in text)
        witnesses = [v for v in row["expected_mentions"] if v["candidate"] in primary]
        assert {v["candidate"] for v in witnesses} == primary
        assert len(witnesses) == len(primary)
        assert {text[v["start"]:v["end"]] for v in witnesses} <= allowed
        counts[row["split"], kind, row["language"]] += 1
    assert len(counts) == 4 * 2 * 2


def test_natural_families_use_functions_without_action_ids(full_rows):
    import re

    counts = Counter()
    for row in full_rows:
        if row["rendering"]["natural_no_action_ids"]:
            assert not re.search(r"(?:операци[яию]|Operation|operation) \d{2}", row["text"])
            assert "Разные номера" not in row["text"] and "Different numbers" not in row["text"]
            counts[row["split"], row["language"]] += 1
    assert len(counts) == 8


def test_partial_pair_relations_preserve_known_aliases(full_rows):
    rows = [r for r in full_rows if r["scenario_kind"] == "pair_unspecified"]
    assert rows
    assert all({p["relation"] for p in row["pair_labels"]} == {"same", "unspecified"}
               for row in rows)


def test_coverage_rejects_removing_completeness_contrasts(full_rows):
    rows = [r for r in full_rows if r["contrast_type"] != "completeness"]
    with pytest.raises(ValueError, match="completeness contrast|heterogeneous alias quota"):
        validate_factorized_coverage(rows)


def test_renderer_coverage_rejects_faked_compact_or_condition_metadata(full_rows):
    row = copy.deepcopy(next(r for r in full_rows if r["rendering"]["compact_only"]))
    row["rendering"]["compact_only"] = False
    with pytest.raises(ValueError, match="independent specification witnesses"):
        validate_factorized_coverage([row], require_full=False)


def test_implicit_missing_mode_preserves_supported_status():
    spec = FactorizedSpec((Operation(("key:space",), "unspecified"),))
    for language in ("ru", "en"):
        explicit, _ = render_factorized_spec(spec, language=language, grammar=1, variant=1, seed=2)
        implicit, _ = render_factorized_spec(spec, language=language, grammar=2, variant=2, seed=2)
        phrase = "длительность не указана" if language == "ru" else "duration is unspecified"
        assert phrase in explicit
        assert ("длительность" if language == "ru" else "duration") not in implicit
        labels, _, _ = annotate_factorized_spec(spec)
        space = next(v for v in labels if v["candidate"] == "key:space")
        assert space["applicability"] == "supported" and space["mode"] == "unspecified"


@pytest.mark.parametrize("candidates,base,overrides,expected", [
    (("key:a", "key:e"), "held", {"key:e": "press"}, {"key:a": "held", "key:e": "press"}),
    (("key:space", "wheel:+"), "press", {"wheel:+": "relative"},
     {"key:space": "press", "wheel:+": "relative"}),
    (("key:a", "key:e"), "held", {"key:e": "unspecified"},
     {"key:a": "held", "key:e": "unspecified"}),
])
def test_functional_aliases_keep_independent_physical_modes(candidates, base, overrides, expected):
    operation = Operation(candidates, base, mode_overrides=overrides)
    labels, pairs, legacy = annotate_factorized_spec(FactorizedSpec((operation,)))
    labels = {v["candidate"]: v for v in labels}
    assert {candidate: labels[candidate]["mode"] for candidate in candidates} == expected
    assert {labels[c]["applicability"] for c in candidates} == {"supported"}
    assert pairs == [{"left": min(candidates), "right": max(candidates), "relation": "same"}]
    assert len({v["alias_group"] for v in legacy if v["candidate"] in candidates}) == 1


@pytest.mark.parametrize("candidates,base,overrides", [
    (("key:a",), "held", (("key:e", "press"),)),
    (("key:a",), "held", (("key:a", "press"), ("key:a", "held"))),
    (("key:a",), "held", (("key:a", "relative"),)),
    (("key:a",), "held", (("key:a", "invalid"),)),
    (("key:a", "wheel:+"), "press", ()),
])
def test_invalid_mode_overrides_fail_closed(candidates, base, overrides):
    with pytest.raises(ValueError):
        Operation(candidates, base, mode_overrides=overrides)


def test_heterogeneous_rendering_names_each_physical_timing_and_shared_function():
    from carryrl.rewards.control_text_data_v14 import _MODE_TEXT

    spec = FactorizedSpec((Operation(("key:space", "wheel:+"), "press",
                                     mode_overrides={"wheel:+": "relative"}),))
    for grammar in range(15):
        for language in ("ru", "en"):
            text, mentions = render_factorized_spec(
                spec, language=language, grammar=grammar, variant=grammar % 3, seed=1)
            li = 0 if language == "ru" else 1
            assert _MODE_TEXT["press"][li] in text
            assert _MODE_TEXT["relative"][li] in text
            assert ("то же действие" if language == "ru" else "that same action") in text
            assert [v["candidate"] for v in mentions] == ["key:space", "wheel:+"]
            assert all(text[v["start"]:v["end"]] for v in mentions)
            if grammar % 3 != 2:
                assert text.count("01") == 2  # one functional ID, two physical timing clauses


def test_unknown_mode_of_one_alias_does_not_erase_other_mode_or_equivalence():
    spec = FactorizedSpec((Operation(("key:a", "key:e"), "held",
                                     mode_overrides={"key:e": "unspecified"}),))
    labels, pairs, _ = annotate_factorized_spec(spec)
    labels = {v["candidate"]: v for v in labels}
    assert labels["key:a"]["mode"] == "held" and labels["key:e"]["mode"] == "unspecified"
    assert pairs[0]["relation"] == "same"
    explicit, _ = render_factorized_spec(spec, language="en", grammar=1, variant=1, seed=1)
    implicit, _ = render_factorized_spec(spec, language="en", grammar=2, variant=2, seed=1)
    assert "duration is unspecified" in explicit and "duration is unspecified" not in implicit
    assert "hold for several" in implicit and "that same action" in implicit


def test_heterogeneous_alias_quotas_are_fixed_inside_existing_split_budgets(full_rows):
    groups = {r["group_id"]: r for r in full_rows}
    for split, quota in HETEROGENEOUS_GROUP_QUOTAS.items():
        for kind in HETEROGENEOUS_KINDS:
            matching = [r for r in groups.values() if r["split"] == split
                        and r["heterogeneous_alias_kind"] == kind]
            assert len(matching) == quota
            assert all(r["rendering"]["heterogeneous_alias_modes"] for r in matching)
            if split == "dev":
                assert Counter(r["development_role"] for r in matching) == {
                    "temperature": quota // 2, "selection": quota // 2}


def test_coverage_rejects_removing_heterogeneous_alias_cases(full_rows):
    rows = [r for r in full_rows if r["heterogeneous_alias_kind"] != "cross_modal_aliases"]
    with pytest.raises(ValueError, match="heterogeneous alias quota"):
        validate_factorized_coverage(rows)


def test_override_aware_mode_contrast_changes_a_real_physical_target():
    from carryrl.rewards.control_text_data_v14 import _operation_mode_contrast

    original = Operation(("key:a", "wheel:+"), "press", "indicator_lit",
                         {"key:a": "held", "wheel:+": "relative"})
    contrast = _operation_mode_contrast(original)
    assert original.mode_for("key:a") != contrast.mode_for("key:a")
    assert contrast.mode_for("wheel:+") == "relative"
    assert original.condition == contrast.condition and original.candidates == contrast.candidates
    a, pairs_a, _ = annotate_factorized_spec(FactorizedSpec((original,)))
    b, pairs_b, _ = annotate_factorized_spec(FactorizedSpec((contrast,)))
    assert a != b and pairs_a == pairs_b


def test_compact_rendering_falls_back_for_heterogeneous_direction_aliases():
    from carryrl.rewards.control_text_data_v14 import _NAVIGATION, rendering_metadata

    operations = [Operation(pair, "held") for pair in _NAVIGATION]
    operations[0] = Operation(operations[0].candidates, "held", mode_overrides={"key:up": "press"})
    spec = FactorizedSpec(tuple(operations), kind="wasd")
    assert rendering_metadata(spec, grammar=1, variant=1)["compact_only"] is False
    text, mentions = render_factorized_spec(spec, language="en", grammar=1, variant=1, seed=1)
    assert "WASD" not in text and "brief press" in text and "hold for several" in text
    assert {v["candidate"] for v in mentions} == {c for op in operations for c in op.candidates}


@pytest.mark.parametrize("candidate,mode", [
    ("key:a", "held"), ("key:e", "press"), ("wheel:+", "relative"),
    ("key:space", "unspecified"),
])
def test_train_duration_cue_and_function_bridge_are_independent_of_mode(candidate, mode):
    from carryrl.rewards.control_text_data_v14 import _FUNCTIONS

    spec = FactorizedSpec((Operation((candidate,), mode, "indicator_lit"),))
    labels_before = annotate_factorized_spec(spec)
    for language in ("ru", "en"):
        augmented, mentions, spans = render_factorized_spec(
            spec, language=language, grammar=2, variant=2, seed=4, include_binding_spans=True)
        simple, _, original = render_factorized_spec(
            spec, language=language, grammar=2, variant=2, seed=1, include_binding_spans=True)
        span = spans[0]
        assert span["mode"] == mode and span["duration_cue"]
        assert span["function_between_candidate_and_mode"]
        mode_start, mode_end = span["mode_payload_span"]
        function_start, function_end = span["function_span"]
        assert mentions[0]["end"] <= function_start < function_end <= mode_start < mode_end
        assert augmented[function_start:function_end] == _FUNCTIONS[0][0 if language == "ru" else 1]
        assert augmented[mode_start:mode_end]
        assert not original[0]["function_between_candidate_and_mode"]
        assert "Когда горит сигнал" in augmented or "When the signal is lit" in augmented
        assert augmented != simple and annotate_factorized_spec(spec) == labels_before


def test_unknown_companion_duration_cue_does_not_fake_known_binding_coverage():
    spec = FactorizedSpec((Operation(("key:a",), "held"),
                           Operation(("key:e",), "unspecified")))
    text, _, bindings = render_factorized_spec(
        spec, language="en", grammar=1, variant=1, seed=1, include_binding_spans=True)
    assert "duration" in text
    by_candidate = {v["candidates"][0]: v for v in bindings}
    assert by_candidate["key:a"]["duration_cue"] is False
    assert by_candidate["key:e"]["duration_cue"] is True


def test_full_train_binding_coverage_crosses_modes_cues_and_function_positions(full_rows):
    seen = Counter()
    for row in full_rows:
        if row["split"] != "train":
            assert row["rendering"]["train_timing_context"] is False
            continue
        for binding in row["binding_spans"]:
            clause = row["text"][binding["start"]:binding["end"]]
            assert binding["duration_cue"] == ("duration" in clause.lower()
                                               or "длительность" in clause.lower())
            seen["cue", binding["mode"], binding["duration_cue"]] += 1
            seen["function", binding["mode"], binding["function_between_candidate_and_mode"]] += 1
    for mode in ("held", "press", "relative", "unspecified"):
        for present in (False, True):
            assert seen["cue", mode, present] > 0
            assert seen["function", mode, present] > 0


def test_r4_preserves_every_unreplaced_r3_record_and_all_heldout(full_rows):
    from carryrl.rewards.control_text_data_v14 import _generate_r3_rows

    r3 = _generate_r3_rows()
    old = {r["id"]: r for r in r3}
    retained = [r for r in full_rows if not r.get("relational_panel")]
    assert len(retained) == len(r3) - 576 * 6
    assert all(old[r["id"]] == r for r in retained)
    for split in ("dev", "calibration", "sealed_test"):
        assert [r for r in r3 if r["split"] == split] == [
            r for r in full_rows if r["split"] == split]


def test_relational_triplets_have_actual_same_different_unknown_targets(full_rows):
    panels = defaultdict(dict)
    for row in full_rows:
        if row.get("relational_panel"):
            panels[row["independent_group_id"]][row["relational_panel"]["state"]] = row
    assert len(panels) == 192
    for states in panels.values():
        assert set(states) == {"same", "different", "unspecified"}
        reference = states["same"]
        properties = [(v["candidate"], v["applicability"], v["mode"])
                      for v in reference["factorized_labels"]]
        for state, row in states.items():
            assert [(v["candidate"], v["applicability"], v["mode"])
                    for v in row["factorized_labels"]] == properties
            focal = set(row["relational_panel"]["focal_pair"])
            pairs = row["pair_labels"]
            assert next(p["relation"] for p in pairs if {p["left"], p["right"]} == focal) == state
            assert len(pairs) == 15
            counts = Counter(p["relation"] for p in pairs)
            assert counts == ({"same": 3, "different": 12} if state == "same" else
                              {"same": 2, "different": 13} if state == "different" else
                              {"same": 2, "unspecified": 13})
            assert len(row["relational_panel"]["changed_pairs_from_same"]) == {
                "same": 0, "different": 1, "unspecified": 13}[state]


def test_relational_physical_coverage_and_all_frames_are_counterbalanced(full_rows):
    from carryrl.rewards.control_text_data_v14 import relational_focal_pairs

    bank = relational_focal_pairs(20261024)
    degree = Counter(c for pair in bank for c in pair)
    assert len(bank) == 48 and set(degree) == set(CONTROL_CANDIDATES)
    assert min(degree.values()) >= 2
    units = defaultdict(set)
    for row in full_rows:
        panel = row.get("relational_panel")
        if panel:
            units[tuple(panel["focal_pair"]), panel["state"], panel["family"], row["language"]].add(
                row["independent_group_id"])
    assert len(units) == 48 * 3 * 3 * 2
    assert {len(v) for v in units.values()} == {4}


def test_relational_binding_bodies_match_across_states_and_scope_preserves_aliases(full_rows):
    by_context = defaultdict(dict)
    for row in full_rows:
        if row.get("relational_panel"):
            key = row["independent_group_id"], row["language"], row["relational_panel"]["family"]
            by_context[key][row["relational_panel"]["state"]] = row
    for states in by_context.values():
        bodies = {row["text"][:max(b["end"] for b in row["binding_spans"])]
                  for row in states.values()}
        assert len(bodies) == 1
        unknown = states["unspecified"]
        assert ("равнозначности сохраняются" in unknown["text"]
                or "explicit equivalences above hold" in unknown["text"])
        assert all(b["function_span"] is None for b in unknown["binding_spans"])
        for row in states.values():
            assert len(row["binding_spans"]) == 6
            assert all(0 <= m["start"] < m["end"] <= len(row["text"])
                       for m in row["expected_mentions"])


def test_relational_validator_rejects_missing_state_changed_edges_and_view_context(full_rows):
    panel_rows = [r for r in full_rows if r.get("relational_panel")]
    first = panel_rows[0]["independent_group_id"]
    cluster = [r for r in panel_rows if r["independent_group_id"] == first]
    missing = [r for r in cluster if r["relational_panel"]["state"] != "unspecified"]
    with pytest.raises(ValueError, match="all three semantic states"):
        validate_factorized_coverage(missing, require_full=False)
    changed = copy.deepcopy(cluster)
    changed[0]["relational_panel"]["changed_pairs_from_same"] = [{"fake": True}]
    with pytest.raises(ValueError, match="context|changed edges"):
        validate_factorized_coverage(changed, require_full=False)
    incomplete_bank = [r for r in full_rows if r["independent_group_id"] != first]
    with pytest.raises(ValueError, match="four independent contexts"):
        validate_factorized_coverage(incomplete_bank)


def test_relational_clause_positions_are_shuffled_without_state_shortcuts(full_rows):
    positions, contexts = set(), defaultdict(dict)
    for row in full_rows:
        panel = row.get("relational_panel")
        if not panel:
            continue
        mentions = row["expected_mentions"]
        pair_order = tuple(frozenset((mentions[i]["candidate"], mentions[i + 1]["candidate"]))
                           for i in (6, 8, 10))
        positions.add(pair_order.index(frozenset(panel["focal_pair"])))
        key = row["independent_group_id"], panel["family"], row["language"]
        contexts[key][panel["state"]] = pair_order
    assert positions == {0, 1, 2}
    assert all(len(set(states.values())) == 1 for states in contexts.values())
