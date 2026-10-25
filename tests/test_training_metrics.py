"""Dashboard retains observed reward semantics without inventing missing budgets."""

import pytest

from carryrl.dashboard.overview_data import _compact
from carryrl.dashboard.training_metrics import compact_episode, cycle_summary


def test_factorized_reward_facts_survive_compaction():
    composition = {'version': 'carryrl.reward.outcome_potential_text.v2'}
    table = {'predictions': [{'candidate': 'key:a', 'status': 'supported',
                             'mode_decision': {'value': 'unspecified',
                                               'reason': 'insufficient_information'}}],
             'pair_decisions': [{'left': 'key:a', 'right': 'key:b',
                                 'decision': {'value': 'same', 'accepted': True}}]}
    episode = {'task': 'a', 'reward_components': {'control_bonus': [0.0005, 0.0015],
                                                  'unsupported_penalty': [0, -0.0001]},
               'reward_diagnostics': {'composition': composition, 'control_coefficient': .5,
                                      'control_table': table, 'control_table_sha256': 'digest',
                                      'steps': [{'control_events': [
                                          {'event': 'supported_award', 'amount': .0005},
                                          {'event': 'unknown', 'reason': 'low_confidence'}]},
                                                {'control_events': [
                                                    {'event': 'unsupported_step_penalty'}]}]}}
    compact = _compact({'complete': True, 'reward_composition': composition,
                        'episode_records': [episode]})
    assert compact['reward_composition'] == composition
    saved = compact['episode_records'][0]
    assert saved['reward_composition'] == composition
    assert saved['control_table'] == table
    assert saved['control_events'] == {'supported_award': 1, 'unknown': 1,
                                       'unsupported_step_penalty': 1}
    assert saved['control_event_reasons'] == {'low_confidence': 1}
    assert saved['control_budget'] == {'positive_paid': .002,
                                       'negative_contribution': -.0001}
    assert saved['control_coefficient'] == .5
    assert 'steps' not in saved


def test_missing_reward_measurements_are_not_zero_budgets():
    saved = compact_episode({'reward_diagnostics': {'steps': [{}]}})
    assert 'control_budget' not in saved
    assert 'control_events' not in saved
    assert 'control_event_reasons' not in saved
    assert 'reward_composition' not in saved


@pytest.mark.parametrize('artifacts,expected', [
    ([{'quality_gate_passed': False}], False),
    ([{'quality_gate_passed': True}], True),
    ([{'quality_gate_passed': False}, {'quality_gate_passed': False}], False),
    ([{'quality_gate_passed': True}, {'quality_gate_passed': True}], True),
    ([{'quality_gate_passed': True}, {'quality_gate_passed': False}], None),
    ([{'quality_gate_passed': False}, {}], None),
    ([{'quality_gate_passed': True}, {'quality_gate_passed': None}], None),
    ([{'quality_gate_passed': 0}], None),
    ([{'quality_gate_passed': 'false'}], None),
    ([{'ready': False}], None),
    ([None], None),
    ([], None),
    ({'quality_gate_passed': False}, None),
    (None, None),
])
def test_cycle_quality_uses_only_unanimous_explicit_artifact_booleans(artifacts, expected):
    state = {'status': 'failed', 'stages': {'text_fit_probe': {
        'phase': 'rewards', 'status': 'failed', 'actual_exit_code': 0,
        'process_complete': True, 'artifacts': artifacts}}}
    reward = cycle_summary(state, now=1000, updated=999)['rewards'][0]
    if expected is None:
        assert 'quality_check_passed' not in reward
    else:
        assert reward['quality_check_passed'] is expected
    assert not reward['confirmed']


def test_cycle_quality_does_not_replace_technical_exit_evidence():
    state = {'stages': {'text_fit_probe': {
        'phase': 'rewards', 'status': 'failed', 'actual_exit_code': 2,
        'artifacts': [{'quality_gate_passed': False}]}}}
    reward = cycle_summary(state, now=1000, updated=999)['rewards'][0]
    assert reward['quality_check_passed'] is False
    assert reward['actual_exit_code'] == 2
    assert not reward['confirmed']
