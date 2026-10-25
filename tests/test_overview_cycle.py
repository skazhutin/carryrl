"""Cycle display keeps preparation, process evidence and policy results distinct."""

import json

import pytest

from carryrl.dashboard.overview_data import OverviewReader
from carryrl.dashboard.training_metrics import cycle_summary


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


IDENTITY = {'pid': 123, 'pgid': 123, 'started': 'Wed Oct 21 12:00:00 2026',
            'command': 'python fresh reward'}


def cycle_state(**updates):
    return {'version': 'carryrl.rl-cycle.v1', 'status': 'prepared',
            'first_compute_unix': 1000, 'deadline_unix': 87400, 'updated_unix': 1590,
            'stages': {}, 'decisions': {}, **updates}


def reader_fixture(tmp_path, state):
    root = tmp_path / 'runs'
    old = root / 'old_model'
    write(old / 'metadata.json', {'kind': 'sequence_training'})
    write(old / 'status.json', {'status': 'completed'})
    plan = root / 'new_cycle'
    write(plan / 'manifest.json', {'status': 'engineering_preparation'})
    write(plan / 'cycle_state.json', state)
    registry = tmp_path / 'registry.json'
    write(registry, {'run': str(old), 'cycle_plan': str(plan)})
    now = [1600]
    identity = [IDENTITY]
    reader = OverviewReader(root, registry, clock=lambda: now[0],
                            cycle_process_identity=lambda pid: identity[0],
                            process_alive=lambda pid: False)
    return reader, plan, now, identity


def test_cycle_preparation_never_relabels_previous_model_or_its_metrics(tmp_path):
    reader, _, now, _ = reader_fixture(tmp_path, cycle_state())
    for tick in (1600, 1601):
        now[0] = tick
        result = reader.snapshot(include_experiments=False)
        assert result['current'] is None and result['training_current'] is None
        assert result['selected_run_id'] is None and result['quality']['value'] is None
        assert not result['cycle']['policy_created']
        assert result['cycle']['budget']['remaining_seconds'] == 87400 - tick
        assert result['warnings'] == []
    archive = reader.snapshot(include_experiments=False, run_id='old_model')
    assert archive['current']['run_id'] == 'old_model'
    assert archive['training_current'] is None


def test_cycle_current_policy_requires_created_metadata_and_owned_output(tmp_path):
    output = tmp_path / 'runs/new_policy'
    state = cycle_state(stages={'probe': {'phase': 'probe', 'policy_output': str(output),
                                         'status': 'running', 'process_identity': IDENTITY}},
                        current_stage='probe', status='running')
    reader, _, _, _ = reader_fixture(tmp_path, state)
    assert reader.snapshot(include_experiments=False)['current'] is None
    write(output / 'metadata.json', {'kind': 'sequence_training', 'config': {'total_rollouts': 8}})
    write(output / 'status.json', {'status': 'running', 'pid': 123, 'phase': 'collecting'})
    snapshot = reader.snapshot(include_experiments=False)
    assert snapshot['cycle']['policy_created']
    assert snapshot['current']['run_id'] == 'new_policy'
    assert snapshot['training_current']['run_id'] == 'new_policy'


def test_live_cycle_policy_ignores_exited_legacy_launch_retained_in_registry(tmp_path):
    output = tmp_path / 'runs/new_policy'
    state = cycle_state(status='running', current_stage='probe', stages={
        'probe': {'phase': 'probe', 'policy_output': str(output), 'status': 'running',
                  'process_identity': IDENTITY}})
    reader, _, now, _ = reader_fixture(tmp_path, state)
    reader.process_alive = lambda pid: pid == IDENTITY['pid']
    write(output / 'metadata.json', {'kind': 'sequence_training', 'config': {'total_rollouts': 2}})
    write(output / 'status.json', {'status': 'running', 'pid': IDENTITY['pid'],
                                  'phase': 'collecting', 'updated_at': 1599})
    old_plan = reader.root / 'old_plan'
    old_launch = old_plan / 'stages/main'
    write(old_plan / 'manifest.json', {'format': 'carryrl.fresh_rl.v1'})
    write(old_launch / 'status.json', {'format': 'carryrl.fresh_rl.v1',
                                      'plan': str(old_plan),
                                      'policy_output': str(reader.root / 'old_model')})
    write(old_launch / 'process.json', {'status': 'exited', 'actual_exit_code': 1,
                                       'command_pid': 999, 'memory_throttled': True,
                                       'resource_sample': {'utc': 1599}})
    registry = json.loads(reader.registry.read_text())
    write(reader.registry, {**registry, 'launch': str(old_launch)})
    for tick in (1600, 1601):
        now[0] = tick
        snapshot = reader.snapshot(include_experiments=False)
        for current in (snapshot['current'], snapshot['training_current']):
            assert current['run_id'] == 'new_policy'
            assert current['running'] is True and current['status'] == 'running'
            assert current['failure_reason'] is None
            assert not current['resource_guard']['memory_throttled']
        assert snapshot['cycle']['running'] is True


def test_cycle_running_requires_matching_process_identity_and_reports_staleness(tmp_path):
    state = cycle_state(status='running', current_stage='visual',
                        stages={'visual': {'phase': 'rewards', 'status': 'running',
                                           'process_identity': IDENTITY, 'memory_paused': True}})
    reader, _, now, identity = reader_fixture(tmp_path, state)
    result = reader.snapshot(include_experiments=False)['cycle']
    assert result['running'] and result['identity_confirmed'] and result['memory_paused']
    assert result['phase_label'] == 'Подготовка и проверка наград'
    assert result['age_seconds'] == 10 and not result['stale']
    identity[0] = {**IDENTITY, 'started': 'reused pid'}
    now[0] = 2000
    result = reader.snapshot(include_experiments=False)['cycle']
    assert not result['running'] and result['status'] == 'stale'
    assert not result['memory_paused'] and result['stale']


@pytest.mark.parametrize('status', ['completed', 'stopped', 'failed'])
def test_terminal_cycle_budget_freezes_at_saved_finish(tmp_path, status):
    reader, _, now, _ = reader_fixture(tmp_path, cycle_state(status=status, finished_unix=1500))
    for tick in (1600, 2000):
        now[0] = tick
        result = reader.snapshot(include_experiments=False)['cycle']
        assert not result['running']
        assert result['budget']['elapsed_seconds'] == 500
        assert result['budget']['remaining_seconds'] == 85900
        assert result['budget']['frozen']
        assert result['budget']['freeze_source'] == 'finished_unix'


@pytest.mark.parametrize('status,code', [('stopped', 0), ('completed', 0), ('failed', 1)])
def test_saved_terminal_receipt_does_not_need_new_heartbeats(status, code):
    receipt = {'phase': 'rewards', 'status': status, 'process_complete': True,
               'actual_exit_code': code, 'finished_unix': 1499, 'process_identity': IDENTITY}
    state = cycle_state(status=status, current_stage='text_fit', finished_unix=1500,
                        stages={'text_fit': receipt})
    result = cycle_summary(state, now=5000, updated=1500, process_identity=IDENTITY)
    assert result['age_seconds'] == 3500 and result['terminal_result_saved']
    assert not result['stale'] and not result['running'] and not result['identity_confirmed']
    assert result['budget']['remaining_seconds'] == 85900


@pytest.mark.parametrize('missing', ['process_complete', 'actual_exit_code',
                                    'stage_finished', 'cycle_finished'])
def test_unconfirmed_terminal_receipt_still_becomes_stale(missing):
    receipt = {'phase': 'rewards', 'status': 'stopped', 'process_complete': True,
               'actual_exit_code': 0, 'finished_unix': 1499}
    state = cycle_state(status='stopped', current_stage='text_fit', finished_unix=1500,
                        stages={'text_fit': receipt})
    if missing == 'cycle_finished':
        del state['finished_unix']
    elif missing == 'stage_finished':
        del receipt['finished_unix']
    else:
        del receipt[missing]
    result = cycle_summary(state, now=5000, updated=1500)
    assert result['stale'] and not result['terminal_result_saved']


def test_running_cycle_with_matching_pid_and_old_heartbeat_stays_stale():
    state = cycle_state(status='running', current_stage='text_fit', stages={
        'text_fit': {'phase': 'rewards', 'status': 'running', 'process_identity': IDENTITY}})
    result = cycle_summary(state, now=5000, updated=1500, process_identity=IDENTITY)
    assert result['running'] and result['identity_confirmed'] and result['stale']
    assert not result['terminal_result_saved']


def test_prepared_unstarted_budget_and_invalid_clock_are_explicit():
    result = cycle_summary(cycle_state(first_compute_unix=None, deadline_unix=None),
                           now=1600, updated=1590)
    assert result['budget']['remaining_seconds'] == 86400
    assert not result['budget']['started']
    result = cycle_summary(cycle_state(deadline_unix=99999), now=1600, updated=1590)
    assert result['budget'] is None and not result['budget_valid']


def test_terminal_legacy_receipt_freezes_budget_with_explicit_fallback_source():
    state = cycle_state(status='stopped', stages={'visual': {'finished_unix': 1500}})
    for now in (2000, 9000):
        result = cycle_summary(state, now=now, updated=1600)
        assert result['budget']['elapsed_seconds'] == 500
        assert result['budget']['freeze_source'] == 'last_stage_or_state_mtime'


@pytest.mark.parametrize('damage', ['exit_code', 'process_incomplete', 'missing_artifacts', 'bool'])
def test_reward_completion_requires_exit_and_artifact_evidence(damage):
    receipt = {'phase': 'rewards', 'status': 'completed', 'actual_exit_code': 0,
               'process_complete': True, 'artifacts': {'report.json': 'a' * 64}}
    if damage == 'exit_code':
        receipt['actual_exit_code'] = 1
    elif damage == 'process_incomplete':
        receipt['process_complete'] = False
    elif damage == 'bool':
        receipt['actual_exit_code'] = False
    else:
        receipt['artifacts'] = {}
    result = cycle_summary(cycle_state(stages={'visual_gate': receipt}), now=1600, updated=1590)
    assert not result['rewards'][0]['confirmed']


def test_reward_completion_requires_all_saved_evidence():
    receipt = {'phase': 'rewards', 'status': 'completed', 'actual_exit_code': 0,
               'process_complete': True, 'artifacts': {'report.json': 'a' * 64}}
    result = cycle_summary(cycle_state(stages={'visual_gate': receipt}), now=1600, updated=1590)
    assert result['rewards'][0]['confirmed']


def test_cycle_plan_cannot_escape_runs_or_follow_symlink(tmp_path):
    reader, plan, _, _ = reader_fixture(tmp_path, cycle_state())
    alias = reader.root / 'alias'
    alias.symlink_to(plan, target_is_directory=True)
    write(reader.registry, {'cycle_plan': str(alias), 'run': str(reader.root / 'old_model')})
    result = reader.snapshot(include_experiments=False)
    assert result['cycle'] is None and result['current'] is None
    assert result['warnings']


@pytest.mark.parametrize('stage_id,artifact,message', [
    ('text_fit', 'reward_text/manifest.json:ready',
     'Текстовый оценщик не прошёл проверку качества.'),
    ('text_fit_repair1', 'reward_text_repair1/manifest.json:ready',
     'Текстовый оценщик не прошёл проверку качества.'),
    ('visual_gate_repair1', 'reward_visual/validation_report.json:passed',
     'Визуальный оценщик не прошёл проверку награды.'),
])
def test_known_reward_artifact_gate_has_readable_reason_and_preserves_details(
    stage_id, artifact, message,
):
    raw = f'CycleStopped: Stage artifact gate failed: /tmp/cycle/{artifact}'
    receipt = {'phase': 'rewards', 'status': 'stopped', 'actual_exit_code': 0,
               'process_complete': True, 'error': raw}
    result = cycle_summary(cycle_state(status='stopped', current_stage=stage_id,
                                       stages={stage_id: receipt}), now=1600, updated=1590)
    assert result['failure_reason'] == message + ' Обучение игрового агента не запускалось.'
    assert result['failure_details'] == raw


@pytest.mark.parametrize('stage_id,reason,code,complete', [
    ('text_fit', 'CycleStopped: Subprocess did not complete successfully', 1, True),
    ('text_fit', 'RuntimeError: model forward crashed', 1, True),
    ('text_fit', 'CycleStopped: Stage artifact gate failed: '
     '/tmp/cycle/reward_text/manifest.json:ready', 1, True),
    ('text_fit', 'CycleStopped: Stage artifact gate failed: '
     '/tmp/cycle/reward_text/manifest.json:ready', 0, False),
    ('text_data', 'CycleStopped: Stage artifact gate failed: '
     '/tmp/cycle/reward_text/manifest.json:ready', 0, True),
    ('text_fit', 'CycleStopped: Stage artifact gate failed: '
     '/tmp/cycle/reward_text/manifest.json:trained', 0, True),
])
def test_process_crashes_and_other_artifact_failures_do_not_become_quality_failures(
    stage_id, reason, code, complete,
):
    receipt = {'phase': 'rewards', 'status': 'stopped', 'actual_exit_code': code,
               'process_complete': complete, 'error': reason}
    result = cycle_summary(cycle_state(status='stopped', current_stage=stage_id,
                                       stages={stage_id: receipt}), now=1600, updated=1590)
    assert result['failure_reason'] == reason
    assert result['failure_details'] is None


def test_reward_progress_keeps_only_supplied_scalar_facts():
    progress = {'stage': 'epoch_complete', 'status': 'running', 'epoch': 3,
                'total_epochs': 8, 'optimizer_updates': 42, 'train_examples': 120,
                'loss': .25, 'percent': 99, 'frames': ['large data']}
    state = cycle_state(stages={
        'text_fit_factorized_v2_corpus_r2_probe': {
            'phase': 'rewards', 'status': 'running', 'progress': progress},
        'text_fit_factorized_v2_corpus_r2_full': {
            'phase': 'rewards', 'status': 'pending'},
        'text_fit_factorized_v2_micro': {
            'phase': 'rewards', 'status': 'completed',
            'progress': {'epoch': True, 'loss': float('nan'), 'train_examples': -1}},
    })
    result = cycle_summary(state, now=1600, updated=1590)
    saved = result['rewards'][0]['progress']
    assert saved == {key: value for key, value in progress.items()
                     if key not in ('percent', 'frames')}
    assert 'progress' not in result['rewards'][1]
    assert 'progress' not in result['rewards'][2]
    assert not result['rewards'][0]['confirmed']
