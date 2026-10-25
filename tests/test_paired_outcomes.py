"""Paired wins require the same ordered observations and known outcomes."""

import json
from copy import deepcopy

import pytest

from carryrl.evaluation.paired import (
    DESCRIPTOR_FIELDS,
    OPTIONAL_DESCRIPTOR_FIELDS,
    episode_descriptor,
    paired_outcomes,
)


def record(index, success=False, *, known=True, task='navigation'):
    return {'task': task, 'seed': 1, 'instruction': 'Go',
            'policy': 'frozen_deterministic', 'action_seed': 1, 'episode_index': index,
            'difficulty': 0, 'level_index': 0, 'level_id': 'same-level',
            'presentation': None, 'initial_frame_sha256': 'a' * 64,
            'outcome': {'known': known, 'success': success}}


def test_zero_net_change_exposes_gain_and_loss_separately():
    before = [record(0), record(1, True), record(2, True), record(3)]
    after = [record(0, True), record(1), record(2, True), record(3)]
    result = paired_outcomes(before, after)
    assert result['gained'] == result['lost'] == 1
    assert result['retained_wins'] == result['retained_nonwins'] == 1
    assert result['net_wins'] == 0
    assert result['known_pairs'] == result['episodes'] == 4
    assert result['unknown_pairs'] == 0
    assert result['coverage'] == 1
    assert result['games'][0] == {'task': 'navigation', **{
        key: value for key, value in result.items() if key not in ('version', 'games')}}
    assert before[0]['outcome']['success'] is False


def test_unknown_outcomes_are_not_losses_or_gains_and_scores_do_not_infer_wins():
    before = [record(0, known=False), record(1, True), record(2), record(0, task='mouse')]
    after = [record(0, True), record(1, known=False), record(2), record(0, True, task='mouse')]
    before[0]['outcome']['score'] = 0
    after[1]['outcome']['score'] = 5
    result = paired_outcomes(before, after)
    assert result['gained'] == 1
    assert result['lost'] == result['retained_wins'] == 0
    assert result['retained_nonwins'] == 1
    assert result['known_before'] == result['known_after'] == 3
    assert result['known_pairs'] == result['unknown_pairs'] == 2
    assert result['coverage'] == .5
    assert result['games'][0]['gained'] == 0
    assert result['games'][1]['gained'] == 1


@pytest.mark.parametrize('field', DESCRIPTOR_FIELDS + OPTIONAL_DESCRIPTOR_FIELDS)
def test_every_descriptor_field_must_match(field):
    before = record(0)
    after = deepcopy(before)
    after[field] = 'changed'
    with pytest.raises(ValueError, match='отличается'):
        paired_outcomes([before], [after])


@pytest.mark.parametrize('field', OPTIONAL_DESCRIPTOR_FIELDS)
def test_optional_missing_is_distinct_from_null(field):
    before = record(0)
    after = {**before, field: None}
    with pytest.raises(ValueError, match='отличается'):
        paired_outcomes([before], [after])
    assert paired_outcomes([after], [after])['episodes'] == 1


def test_order_size_and_duplicate_descriptors_are_rejected():
    records = [record(0), record(1)]
    for after in (list(reversed(records)), records[:1]):
        with pytest.raises(ValueError, match='отличается'):
            paired_outcomes(records, after)
    with pytest.raises(ValueError, match='повторённый'):
        paired_outcomes([record(0), record(0)], [record(0), record(0)])


@pytest.mark.parametrize('damage', ['missing', 'record', 'outcome', 'known', 'success', 'nan'])
def test_malformed_records_are_rejected(damage):
    row = record(0)
    if damage == 'missing':
        del row['episode_index']
    elif damage == 'record':
        row = None
    elif damage == 'outcome':
        row['outcome'] = None
    elif damage == 'known':
        row['outcome']['known'] = 1
    elif damage == 'success':
        row['outcome']['success'] = 1
    else:
        row['difficulty'] = float('nan')
    with pytest.raises(ValueError):
        paired_outcomes([row], [row])


def test_descriptor_json_matches_historical_protocol_for_old_and_new_records():
    for row in (record(0), {**record(0), 'runtime_fingerprint': None,
                           'semantic_level_id': 'semantic'}):
        old_descriptor = {**{key: row[key] for key in DESCRIPTOR_FIELDS},
                          **{key: row[key] for key in OPTIONAL_DESCRIPTOR_FIELDS if key in row}}
        assert json.dumps(episode_descriptor(row), sort_keys=True) == json.dumps(
            old_descriptor, sort_keys=True)
