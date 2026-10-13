"""Regression checks for visible progress and comparable learned-agent results."""

import hashlib
import json
import os

import pytest

from carryrl.dashboard.overview_data import MAX_GIF_BYTES, MAX_JSON_BYTES, TASKS, OverviewReader

HASH = 'a' * 64
REWARD = 'c' * 64


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def evaluation(epoch, revision, agent, *, wins=0, count=1, seed=100):
    records = []
    for index, task in enumerate(TASKS):
        for episode in range(count if task != TASKS[0] else count * 3):
            records.append({'task': task, 'seed': seed + episode, 'instruction': task,
                            'policy': 'frozen_deterministic', 'action_seed': 1,
                            'episode_index': episode, 'difficulty': 0,
                            'level_index': episode, 'level_id': f'{task}-{episode}',
                            'presentation': None, 'initial_frame_sha256': HASH,
                            'steps': 5, 'outcome': {'known': True, 'success': index < wins,
                                                   'failure': index >= wins}})
    return {'epoch': epoch, 'weights_revision': revision, 'agent_sha256': agent,
            'split': 'validation', 'policy_and_reward_frozen': True,
            'level_selection': 'explicit covering level indices', 'records': records}


def make_run(root, name, *, epochs=4, revision=140, agent='b' * 64, source=None):
    run = root / name
    metadata = {'kind': 'imitation_training', 'reward_sha256': REWARD,
                'initial_agent_sha256': HASH,
                'config': {'max_episode_steps': 80, 'env_image_size': 96},
                'settings': {'epochs': epochs, 'minibatch_episodes': 8, 'final_all_levels': True}}
    if source:
        manifest = json.loads((source / 'checkpoints/rollout_0000/manifest.json').read_text())
        metadata.update(source_checkpoint=str(source / 'checkpoints/rollout_0000'),
                        source_checkpoint_sha256=manifest['sha256'],
                        initial_agent_sha256=manifest['agent_sha256'])
    write(run / 'metadata.json', metadata)
    write(run / 'status.json', {'status': 'completed', 'phase': 'finished', 'epochs': epochs,
                              'demonstration_episodes': 32, 'optimizer_steps': epochs * 4,
                              'wall_seconds': 100})
    write(run / 'summary.json', {'epochs_completed': epochs, 'final_agent_sha256': agent,
                               'dataset': {'suite_version': 'test.v1'}})
    checkpoint = run / 'checkpoints/rollout_0000'
    write(checkpoint / 'manifest.json', {'sha256': HASH, 'agent_sha256': agent,
                                      'weights_revision': revision, 'codec_version': 'test.v1',
                                      'reward_model_version': 'test.v1'})
    write(run / 'latest.json', {'checkpoint': str(checkpoint)})
    write(run / 'evaluations/epoch_0000.json', evaluation(
        0, revision - epochs * 4, metadata['initial_agent_sha256']))
    write(run / f'evaluations/epoch_{epochs:04}.json', evaluation(epochs, revision, agent, wins=8))
    return run


def reader_for(tmp_path, active, **kwargs):
    registry = tmp_path / 'registry.json'
    write(registry, {'run': str(active)})
    return OverviewReader(tmp_path / 'runs', registry, **kwargs)


def test_macro_equal_game_weight_not_pooled_episode_rate(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    data = reader_for(tmp_path, run).snapshot()
    assert data['quality']['value'] == 50
    assert data['quality']['successes'] / data['quality']['episodes'] != .5
    assert len(data['quality']['games']) == 16
    assert data['current']['epochs_completed'] == 4
    assert data['current']['optimizer_steps_total'] == 16
    assert data['current']['eta_seconds'] == 0


def test_valid_lineage_deduplicates_same_weights_and_excludes_unrelated(tmp_path):
    root = tmp_path / 'runs'
    first = make_run(root, 'first', revision=116, agent='b' * 64)
    second = make_run(root, 'second', revision=132, agent='d' * 64, source=first)
    initial = evaluation(0, 116, 'b' * 64, wins=8)
    write(second / 'evaluations/epoch_0000.json', initial)
    make_run(root, 'unrelated', revision=1000, agent='e' * 64)
    data = reader_for(tmp_path, second).snapshot()
    history = data['quality']['history']
    assert [point['run_id'] for point in history] == ['first', 'first', 'second']
    assert [point['updates'] for point in history] == [0, 16, 32]
    assert len(data['experiments']) == 3
    assert history[-1]['games'][TASKS[0]]['episodes'] == 3


def test_bad_lineage_manifest_ends_history(tmp_path):
    root = tmp_path / 'runs'
    first = make_run(root, 'first')
    second = make_run(root, 'second', source=first)
    metadata = json.loads((second / 'metadata.json').read_text())
    metadata['source_checkpoint_sha256'] = 'd' * 64
    write(second / 'metadata.json', metadata)
    data = reader_for(tmp_path, second).snapshot()
    assert all(row['run_id'] == 'second' for row in data['quality']['history'])
    assert any('происхождение' in warning for warning in data['warnings'])


def test_protocol_mismatch_and_incomplete_games_omitted_with_warnings(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    write(run / 'evaluations/epoch_0004.json', evaluation(4, 140, 'b' * 64, wins=8, seed=200))
    data = reader_for(tmp_path, run).snapshot()
    assert len(data['quality']['history']) == 1
    assert any('протокол' in warning for warning in data['warnings'])
    incomplete = evaluation(4, 140, 'b' * 64)
    incomplete['records'] = incomplete['records'][:-1]
    write(run / 'evaluations/epoch_0004.json', incomplete)
    data = reader_for(tmp_path, run).snapshot()
    assert len(data['quality']['history']) == 1
    assert any('16 игр' in warning for warning in data['warnings'])


def test_active_cohort_keeps_current_when_parent_config_differs(tmp_path):
    root = tmp_path / 'runs'
    first = make_run(root, 'first', revision=116)
    second = make_run(root, 'second', revision=132, agent='d' * 64, source=first)
    metadata = json.loads((second / 'metadata.json').read_text())
    metadata['config']['max_episode_steps'] = 300
    write(second / 'metadata.json', metadata)
    data = reader_for(tmp_path, second).snapshot()
    assert data['quality']['latest_run_id'] == 'second'
    assert all(row['run_id'] == 'second' for row in data['quality']['history'])
    assert any('протокол' in warning for warning in data['warnings'])


def test_quick_different_bank_never_enters_main_history(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    quick = evaluation(2, 132, 'e' * 64, wins=16, seed=900)
    quick['level_selection'] = 'sampled levels'
    write(run / 'evaluations/epoch_0002.json', quick)
    data = reader_for(tmp_path, run).snapshot()
    assert len(data['quality']['history']) == 2
    assert data['quality']['value'] == 50
    assert data['quality']['quick']['value'] == 100


def test_incomplete_epoch_not_counted_eta_needs_live_and_measurements(tmp_path):
    run = make_run(tmp_path / 'runs', 'first', epochs=10)
    (run / 'summary.json').unlink()
    rows = [{'epoch': epoch, 'complete': True, 'seconds': epoch * 10,
             'optimizer_steps': epoch * 4, 'nll': 2 / epoch} for epoch in range(1, 4)]
    rows.append({'epoch': 4, 'complete': False, 'optimizer_steps': 13, 'nll': .2})
    (run / 'training.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    write(run / 'status.json', {'status': 'running', 'phase': 'imitation', 'pid': 123,
                              'updated_at': 1000, 'epoch': 4, 'epochs_completed': 3,
                              'optimizer_steps': 13, 'demonstration_episodes': 32,
                              'batch': 1})
    reader = reader_for(tmp_path, run, clock=lambda: 1010, process_alive=lambda pid: True)
    current = reader.snapshot()['current']
    assert current['epochs_completed'] == 3
    assert current['epoch'] == 4
    assert current['batch'] == 1
    assert current['eta_seconds'] == pytest.approx(67.5)
    assert current['eta_scope'] == 'training'
    assert len(current['loss_history']) == 3
    reader.clock = lambda: 1200
    assert reader.snapshot()['current']['eta_seconds'] is None
    reader.process_alive = lambda pid: False
    assert reader.snapshot()['current']['status'] == 'interrupted'
    reader.process_alive = lambda pid: None
    reader.clock = lambda: 1010
    assert reader.snapshot()['current']['eta_seconds'] is None


def test_fresh_loading_has_no_fabricated_eta(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    write(run / 'status.json', {'status': 'running', 'phase': 'loading', 'pid': 1,
                              'updated_at': 1000})
    data = reader_for(tmp_path, run, clock=lambda: 1001,
                      process_alive=lambda pid: True).snapshot()
    assert data['current']['eta_seconds'] is None
    assert data['current']['phase_index'] == 1


def add_playback(run):
    gif = b'GIF89a\x01\x00\x01\x00\x80\x00\x00' + b'test-content'
    directory = run / 'playbacks'
    directory.mkdir(exist_ok=True)
    path = directory / 'actual.gif'
    path.write_bytes(gif)
    manifest = json.loads((run / 'checkpoints/rollout_0000/manifest.json').read_text())
    write(directory / 'actual.json', {
        'source': 'actual_frozen_learned_policy_closed_loop_RGB_playback',
        'policy': 'frozen_deterministic', 'teacher_actions_used': False,
        'training_performed': False, 'task': TASKS[0], 'gif': str(path),
        'gif_sha256': hashlib.sha256(gif).hexdigest(),
        'checkpoint': str(run / 'checkpoints/rollout_0000'),
        'checkpoint_sha256': manifest['sha256'], 'agent_sha256': manifest['agent_sha256'],
        'weights_revision': manifest['weights_revision'], 'action_count': 5,
        'outcome': {'known': True, 'success': True}})
    return path, gif


def test_playbacks_newest_first_across_revisions_and_missing_dates(tmp_path):
    older = make_run(tmp_path / 'runs', 'older', revision=100)
    current = make_run(tmp_path / 'runs', 'current', revision=200, source=older)
    for run, stamp in ((older, '2026-10-09T01:00:00Z'),
                       (current, '2026-10-09T03:00:00+03:00')):
        add_playback(run)
        sidecar = run / 'playbacks/actual.json'
        metadata = json.loads(sidecar.read_text())
        metadata['created_utc'] = stamp
        write(sidecar, metadata)
    metadata = json.loads((current / 'playbacks/actual.json').read_text())
    fallback_gif = current / 'playbacks/fallback.gif'
    fallback_gif.write_bytes((current / 'playbacks/actual.gif').read_bytes())
    metadata.update(gif=str(fallback_gif), created_utc='invalid')
    fallback_sidecar = current / 'playbacks/fallback.json'
    write(fallback_sidecar, metadata)
    os.utime(fallback_sidecar, (1791509400, 1791509400))  # 2026-10-09 01:30 UTC

    result = reader_for(tmp_path, current)._playbacks([older, current], {})
    assert [item['id'] for item in result['items']] == [
        'current/fallback.gif', 'older/actual.gif', 'current/actual.gif']
    assert result['selected_id'] == 'current/fallback.gif'


def test_media_digest_changed_symlink_oversize_and_traversal_rejected(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    path, gif = add_playback(run)
    reader = reader_for(tmp_path, run)
    assert reader.media('first', 'actual.gif') == gif
    assert reader.media('first', '../actual.gif') is None
    assert reader.media('../first', 'actual.gif') is None
    path.write_bytes(b'GIF89a' + b'changed')
    assert reader.media('first', 'actual.gif') is None
    path.unlink()
    outside = tmp_path / 'outside.gif'
    outside.write_bytes(gif)
    path.symlink_to(outside)
    assert reader.media('first', 'actual.gif') is None
    path.unlink()
    with path.open('wb') as stream:
        stream.write(b'GIF89a')
        stream.truncate(MAX_GIF_BYTES + 1)
    assert reader.media('first', 'actual.gif') is None


def test_nan_json_omitted_with_warning_and_cache_tracks_changed_fingerprint(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    reader = reader_for(tmp_path, run)
    assert reader.snapshot()['current']['optimizer_steps'] == 16
    status = run / 'status.json'
    status.write_text('{"optimizer_steps":NaN}')
    os.utime(status, None)
    data = reader.snapshot()
    assert data['current']['optimizer_steps'] is None
    assert any('non-finite' in warning for warning in data['warnings'])
    assert reader.run_details('../first') is None
    assert 'checkpoint_manifests' in reader.run_details('first')


def test_unknown_visible_outcome_stays_in_denominator(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    final = evaluation(4, 140, 'b' * 64, wins=16)
    final['records'][0]['outcome'] = {'known': False, 'success': True}
    write(run / 'evaluations/epoch_0004.json', final)
    data = reader_for(tmp_path, run).snapshot()
    game = data['quality']['games'][0]
    assert game['episodes'] == 3
    assert game['successes'] == 2
    assert game['unknown_outcomes'] == 1
    assert data['quality']['episodes'] == 18
    assert data['quality']['successes'] == 17


def test_sprite_verified_dimensions_actions_media_and_gif_fallback(tmp_path):
    from PIL import Image

    run = make_run(tmp_path / 'runs', 'first')
    add_playback(run)
    sidecar = run / 'playbacks/actual.json'
    metadata = json.loads(sidecar.read_text())
    sprite = run / 'playbacks/actual.frames.png'
    Image.new('RGB', (8 * 96, 96)).save(sprite)
    metadata.update(sprite_filename=sprite.name,
                    sprite_sha256=hashlib.sha256(sprite.read_bytes()).hexdigest(),
                    sprite_columns=8, frame_width=96, frame_height=96, frame_count=6,
                    frame_duration_ms=100, terminal_hold_ms=1200,
                    actions=[{'action': {'keys': ['right'], 'mouse_buttons': [],
                                         'dx': 0, 'dy': 0, 'wheel': 0}}] * 5)
    write(sidecar, metadata)
    reader = reader_for(tmp_path, run)
    item = reader.snapshot()['playbacks']['items'][0]
    assert item['sprite_url'] == '/media/first/actual.frames.png'
    assert len(item['actions']) == 5
    assert item['actions'][0]['keys'] == ['right']
    assert reader.media('first', 'actual.frames.png') == sprite.read_bytes()
    metadata['frame_width'] = 64
    write(sidecar, metadata)
    data = reader.snapshot()
    assert len(data['playbacks']['items']) == 1
    assert 'sprite_url' not in data['playbacks']['items'][0]
    assert reader.media('first', 'actual.frames.png') is None
    assert any('GIF retained' in warning for warning in data['warnings'])


def test_interim_playback_requires_exact_frozen_evidence(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    add_playback(run)
    sidecar = run / 'playbacks/actual.json'
    metadata = json.loads(sidecar.read_text())
    evidence = evaluation(2, 132, 'e' * 64, wins=16, seed=900)
    evidence['level_selection'] = 'sampled levels'
    record = evidence['records'][0]
    metadata.update({field: record[field] for field in (
        'task', 'seed', 'level_id', 'instruction', 'presentation', 'difficulty', 'outcome')})
    metadata.update(checkpoint=None, epoch=2, agent_sha256='e' * 64, weights_revision=132)
    write(sidecar, metadata)
    write(run / 'evaluations/epoch_0002.json', evidence)
    reader = reader_for(tmp_path, run)
    assert len(reader.snapshot()['playbacks']['items']) == 1
    metadata['action_count'] = 6
    write(sidecar, metadata)
    assert len(reader.snapshot()['playbacks']['items']) == 0


def test_final_suffix_full_bank_keeps_final_epoch_quick_separate(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    (run / 'evaluations/epoch_0004.json').rename(run / 'evaluations/epoch_0004_final.json')
    quick = evaluation(4, 140, 'b' * 64, wins=16, seed=900)
    quick['level_selection'] = 'sampled levels'
    write(run / 'evaluations/epoch_0004.json', quick)
    data = reader_for(tmp_path, run).snapshot()
    assert data['quality']['value'] == 50
    assert data['quality']['quick']['value'] == 100
    assert len(data['quality']['history']) == 2


def test_training_eta_prefers_measured_training_time_without_validation(tmp_path):
    run = make_run(tmp_path / 'runs', 'first', epochs=10)
    (run / 'summary.json').unlink()
    rows = [{'epoch': epoch, 'complete': True, 'seconds': epoch * 100,
             'epoch_training_seconds': 8, 'optimizer_steps': epoch * 4}
            for epoch in range(1, 4)]
    (run / 'training.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    write(run / 'status.json', {'status': 'running', 'phase': 'imitation', 'pid': 123,
                              'updated_at': 1000, 'epoch': 4, 'epochs_completed': 3,
                              'optimizer_steps': 12, 'demonstration_episodes': 32})
    reader = reader_for(tmp_path, run, clock=lambda: 1010, process_alive=lambda pid: True)
    assert reader.snapshot()['current']['eta_seconds'] == 56


def test_live_warm_start_protocol_uses_verified_source_manifest_before_save(tmp_path):
    root = tmp_path / 'runs'
    first = make_run(root, 'first', revision=116, agent='b' * 64)
    second = make_run(root, 'second', revision=132, agent='d' * 64, source=first)
    write(second / 'evaluations/epoch_0000.json', evaluation(0, 116, 'b' * 64, wins=8))
    (second / 'summary.json').unlink()
    (second / 'checkpoints/rollout_0000/manifest.json').unlink()
    (second / 'evaluations/epoch_0004.json').unlink()
    write(second / 'demonstrations.json', {'suite_version': 'test.v1'})
    data = reader_for(tmp_path, second).snapshot()
    assert [point['run_id'] for point in data['quality']['history']] == ['first', 'first']
    assert data['quality']['value'] == 50
    assert not any('протокол' in warning for warning in data['warnings'])


@pytest.mark.parametrize('compact_source', ['summary', 'metadata'])
@pytest.mark.parametrize('different_bank', [False, True])
def test_large_demonstrations_use_compact_suite_and_preserve_bank_matching(
        tmp_path, monkeypatch, compact_source, different_bank):
    root = tmp_path / 'runs'
    first = new_bank_run(root, 'first', revision=116)
    second = new_bank_run(root, 'second', source=first, revision=132)
    (second / 'evaluations/epoch_0004.json').unlink()
    (second / 'checkpoints/rollout_0000/manifest.json').unlink()
    metadata = json.loads((second / 'metadata.json').read_text())
    if compact_source == 'metadata':
        (second / 'summary.json').unlink()
        metadata['suite_version'] = 'test.v1'
    else:
        # The completed dataset summary takes precedence over metadata.
        metadata['suite_version'] = 'other.v1'
    write(second / 'metadata.json', metadata)
    provenance = second / 'demonstrations.json'
    with provenance.open('wb') as stream:
        stream.write(b'{"suite_version": "test.v1"}')
        stream.write(b' ' * MAX_JSON_BYTES)
    if different_bank:
        initial = json.loads((second / 'evaluations/epoch_0000.json').read_text())
        initial['records'][0]['level_id'] = 'different-level'
        write(second / 'evaluations/epoch_0000.json', initial)
    reader = reader_for(tmp_path, second)
    original_read = reader._read

    def read_without_provenance(path, **kwargs):
        assert path != provenance, 'compact suite must avoid reading large provenance'
        return original_read(path, **kwargs)

    monkeypatch.setattr(reader, '_read', read_without_provenance)
    data = reader.snapshot()
    quality = data['quality']
    assert quality['available'] is True
    assert quality['cohort']['level_bank_seed'] == 20261008
    assert quality['cohort']['difficulty'] == 1
    if different_bank:
        assert [point['run_id'] for point in quality['history']] == ['second']
        assert all(point['run_id'] == 'first' for point in quality['previous_history'])
        assert any('протокол' in warning for warning in data['warnings'])
    else:
        assert [point['run_id'] for point in quality['history']] == ['first', 'first']
        assert [point['updates'] for point in quality['history']] == [0, 16]
        assert quality['previous_history'] == []
        assert not any('протокол' in warning for warning in data['warnings'])
    assert not any('JSON byte limit' in warning for warning in data['warnings'])


@pytest.mark.parametrize('legacy', [None, '{broken', '{"suite_version": null}'])
def test_unknown_legacy_suite_does_not_establish_comparable_quality(tmp_path, legacy):
    run = make_run(tmp_path / 'runs', 'first')
    (run / 'summary.json').unlink()
    if legacy is not None:
        (run / 'demonstrations.json').write_text(legacy)
    data = reader_for(tmp_path, run).snapshot()
    assert data['quality']['available'] is False
    assert data['quality']['history'] == []
    assert data['quality']['cohort'] is None
    assert any('нет подтверждённой версии набора игр' in warning for warning in data['warnings'])


def test_ppo_detail_does_not_invent_imitation_epochs_or_updates(tmp_path):
    run = make_run(tmp_path / 'runs', 'ppo')
    write(run / 'metadata.json', {'kind': 'sequence_training',
                                 'config': {'total_rollouts': 32}})
    write(run / 'summary.json', {})
    write(run / 'status.json', {'status': 'completed', 'rollouts': 20,
                              'decisions': 41015, 'weights_revision': 437})
    current = reader_for(tmp_path, run).snapshot()['current']
    assert current['method'] == 'ppo'
    assert current['rollouts'] == 20
    assert current['decisions'] == 41015
    assert current['training_percent'] == 62.5
    assert current['epoch'] is None
    assert current['epochs_completed'] is None
    assert current['optimizer_steps'] is None
    assert current['optimizer_steps_total'] is None


@pytest.mark.parametrize('phase', ['collecting', 'encoding', 'baseline', 'initial_evaluation',
                                  'evaluation', 'final_evaluation'])
def test_live_phase_eta_observed_throughput_only_and_scope(tmp_path, phase):
    run = make_run(tmp_path / 'runs', 'first')
    status = {'status': 'running', 'phase': phase, 'pid': 123, 'updated_at': 1008,
              'phase_started_at': 1000, 'phase_elapsed_seconds': 8,
              'phase_units_done': 4, 'phase_units_total': 20, 'phase_unit_kind': 'episodes'}
    write(run / 'status.json', status)
    reader = reader_for(tmp_path, run, clock=lambda: 1008, process_alive=lambda pid: True)
    current = reader.snapshot()['current']
    assert current['eta_seconds'] == 32
    assert current['eta_scope'] == 'phase'
    assert 'последующие этапы отдельно' in current['eta_reason']
    assert current['eta_low_seconds'] is None
    assert current['eta_high_seconds'] is None
    assert current['finish_estimated_at'] is None
    assert current['phase_unit'] == 'episodes'
    reader.process_alive = lambda pid: None
    assert reader.snapshot()['current']['eta_seconds'] is None
    reader.process_alive = lambda pid: False
    assert reader.snapshot()['current']['eta_seconds'] is None
    reader.process_alive = lambda pid: True
    reader.clock = lambda: 1209
    assert reader.snapshot()['current']['eta_seconds'] is None
    reader.clock = lambda: 1008
    status['phase_units_done'] = 2
    write(run / 'status.json', status)
    assert reader.snapshot()['current']['eta_seconds'] is None


def test_pending_capture_waits_quietly_for_reference_then_publishes(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    add_playback(run)
    reference = run / 'evaluations/epoch_0002.json'
    sidecar = run / 'playbacks/actual.json'
    metadata = json.loads(sidecar.read_text())
    evidence = evaluation(2, 132, 'e' * 64, wins=16, seed=900)
    evidence['level_selection'] = 'sampled levels'
    record = evidence['records'][0]
    metadata.update({field: record[field] for field in (
        'task', 'seed', 'level_id', 'instruction', 'presentation', 'difficulty', 'outcome')})
    metadata.update(checkpoint=None, epoch=2, agent_sha256='e' * 64, weights_revision=132,
                    evaluation_reference=str(reference))
    write(sidecar, metadata)
    write(run / 'status.json', {'status': 'running', 'phase': 'evaluation', 'pid': 123,
                              'updated_at': 1000})
    reader = reader_for(tmp_path, run, clock=lambda: 1001, process_alive=lambda pid: True)
    data = reader.snapshot()
    assert data['playbacks']['items'] == []
    assert not any('playback lacks' in warning for warning in data['warnings'])
    write(reference, evidence)
    assert len(reader.snapshot()['playbacks']['items']) == 1
    reference.unlink()
    reader.clock = lambda: 1201
    data = reader.snapshot()
    assert data['playbacks']['items'] == []
    assert any('playback lacks' in warning for warning in data['warnings'])


def full_evaluation(epoch, revision, agent, *, difficulty=0, seed=0, wins=8):
    data = evaluation(epoch, revision, agent, count=16, wins=wins)
    data['records'] = [record for record in data['records'] if record['episode_index'] < 16]
    for record in data['records']:
        record['difficulty'] = difficulty
    data.update(evaluation_difficulty=difficulty, level_bank_seed=seed)
    return data


def new_bank_run(root, name, *, source=None, difficulty=1, seed=20261008, revision=132):
    run = make_run(root, name, revision=revision, agent='d' * 64, source=source)
    metadata = json.loads((run / 'metadata.json').read_text())
    metadata['config']['level_bank_seed'] = seed
    metadata['config']['curriculum_difficulty'] = difficulty
    write(run / 'metadata.json', metadata)
    write(run / 'evaluations/epoch_0000.json', full_evaluation(
        0, revision - 16, metadata['initial_agent_sha256'], difficulty=difficulty, seed=seed))
    write(run / 'evaluations/epoch_0004.json', full_evaluation(
        4, revision, 'd' * 64, difficulty=difficulty, seed=seed, wins=12))
    return run


def test_description_fingerprints_derive_accepted_full_difficulty_bank(tmp_path):
    run = new_bank_run(tmp_path / 'runs', 'new-bank')
    data = reader_for(tmp_path, run).snapshot()
    quality = data['quality']
    assert quality['available'] is True
    assert quality['cohort']['difficulty'] == 1
    assert quality['cohort']['level_bank_seed'] == 20261008
    assert len(quality['cohort']['episode_bank_sha256']) == 64
    assert len(quality['cohort']['initial_frame_bank_sha256']) == 64
    assert 'сложность 1' in quality['protocol_description']
    assert 'новый набор уровней, seed 20261008' in quality['protocol_description']
    assert 'оценивается отдельно' in quality['protocol_description']
    assert quality['history'][0]['episodes'] == 256
    assert quality['history'][0]['updates'] == 0
    assert quality['value'] == 75


def test_distinct_seed_isolates_identical_rgb_and_episode_descriptors(tmp_path):
    root = tmp_path / 'runs'
    old = new_bank_run(root, 'old-bank', seed=0, revision=116)
    fresh = new_bank_run(root, 'fresh-bank', source=old, seed=20261008, revision=132)
    # Both files have deliberately identical episode descriptors and initial RGB hashes.
    old_records = json.loads((old / 'evaluations/epoch_0004.json').read_text())['records']
    new_records = json.loads((fresh / 'evaluations/epoch_0000.json').read_text())['records']
    assert all(all(left[field] == right[field] for field in (
        'task', 'seed', 'instruction', 'level_id', 'difficulty', 'initial_frame_sha256'))
        for left, right in zip(old_records, new_records, strict=True))
    data = reader_for(tmp_path, fresh).snapshot()
    assert all(point['run_id'] == 'fresh-bank' for point in data['quality']['history'])
    assert data['quality']['history'][0]['updates'] == 0
    assert all(point['run_id'] == 'old-bank'
               for point in data['quality']['previous_history'])
    assert data['quality']['previous_history'][0]['cohort']['level_bank_seed'] == 0
    assert not any('протокол' in warning for warning in data['warnings'])


def test_collecting_new_bank_without_baseline_never_inherits_old_warmup_wins(tmp_path):
    root = tmp_path / 'runs'
    old = make_run(root, 'motor-warmup', revision=116, agent='b' * 64)
    fresh = new_bank_run(root, 'fresh-bank', source=old)
    (fresh / 'evaluations/epoch_0000.json').unlink()
    (fresh / 'evaluations/epoch_0004.json').unlink()
    write(fresh / 'status.json', {'status': 'running', 'phase': 'collecting', 'pid': 123,
                                'updated_at': 1000})
    data = reader_for(tmp_path, fresh, clock=lambda: 1001,
                      process_alive=lambda pid: True).snapshot()
    quality = data['quality']
    assert quality['available'] is False
    assert quality['value'] is None
    assert quality['baseline'] is None
    assert quality['latest_run_id'] is None
    assert quality['history'] == []
    assert quality['cohort'] is None
    assert 'ещё не опубликована' in quality['protocol_description']
    assert all(point['run_id'] == 'motor-warmup' for point in quality['previous_history'])
    assert len(quality['previous_history']) == 2


def test_missing_and_explicit_legacy_zero_bank_seed_are_comparable(tmp_path):
    root = tmp_path / 'runs'
    old = make_run(root, 'legacy', revision=116, agent='b' * 64)
    fresh = make_run(root, 'explicit-zero', source=old, revision=132, agent='d' * 64)
    metadata = json.loads((fresh / 'metadata.json').read_text())
    metadata['config']['level_bank_seed'] = 0
    write(fresh / 'metadata.json', metadata)
    write(fresh / 'evaluations/epoch_0000.json', evaluation(0, 116, 'b' * 64, wins=8))
    quality = reader_for(tmp_path, fresh).snapshot()['quality']
    assert len(quality['history']) == 3
    assert quality['previous_history'] == []
    assert quality['cohort']['level_bank_seed'] == 0


def test_partial_new_bank_initial_is_not_a_256_episode_baseline(tmp_path):
    run = new_bank_run(tmp_path / 'runs', 'fresh-bank')
    partial = full_evaluation(0, 116, HASH, difficulty=1, seed=20261008)
    partial['records'] = [r for r in partial['records'] if r['episode_index'] < 2]
    write(run / 'evaluations/epoch_0000.json', partial)
    data = reader_for(tmp_path, run).snapshot()
    assert data['quality']['available'] is False
    assert data['quality']['history'] == []
    assert any('256 эпизодов' in warning for warning in data['warnings'])


def test_copied_reused_measurement_keeps_verified_original_time(tmp_path):
    root = tmp_path / 'runs'
    source_run = make_run(root, 'source')
    source = source_run / 'evaluations/epoch_0000.json'
    original = json.loads(source.read_text())
    original['created_utc'] = '2026-10-08T19:12:08.337830+00:00'
    write(source, original)
    target = make_run(root, 'copy')
    reused = json.loads(source.read_text())
    reused['created_utc'] = '2026-10-08T21:00:00+00:00'
    reused['initial_evaluation_reuse'] = {
        'measurement_reused': True, 'source': str(source),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_created_utc': original['created_utc'], 'source_epoch': original['epoch'],
        'source_weights_revision': original['weights_revision']}
    destination = target / 'evaluations/epoch_0000.json'
    write(destination, reused)
    os.utime(destination, (2000000000, 2000000000))
    reader = reader_for(tmp_path, target)
    assert reader.snapshot()['quality']['history'][0]['evaluated_at'] == original['created_utc']
    # A later copy cannot move this measurement along the time axis.
    os.utime(destination, (2100000000, 2100000000))
    assert reader.snapshot()['quality']['history'][0]['evaluated_at'] == original['created_utc']
    # An unverified provenance timestamp is ignored in favor of valid created_utc.
    reused['initial_evaluation_reuse']['source_sha256'] = 'f' * 64
    write(destination, reused)
    assert reader.snapshot()['quality']['history'][0]['evaluated_at'] == reused['created_utc']


@pytest.mark.parametrize('timestamp', ['invalid-date', '2026-10-08T19:12:08', None, 123])
def test_malformed_or_naive_measurement_time_uses_legacy_mtime(tmp_path, timestamp):
    run = make_run(tmp_path / 'runs', 'legacy-time')
    path = run / 'evaluations/epoch_0000.json'
    evaluation_data = json.loads(path.read_text())
    evaluation_data['created_utc'] = timestamp
    evaluation_data['initial_evaluation_reuse'] = {
        'measurement_reused': True, 'source_created_utc': 'not-a-date'}
    write(path, evaluation_data)
    os.utime(path, (1000000000, 1000000000))
    reader = reader_for(tmp_path, run)
    assert reader.snapshot()['quality']['history'][0]['evaluated_at'] == '2001-09-09T01:46:40+00:00'


def test_measurement_timezone_normalizes_to_utc_instead_of_file_time(tmp_path):
    run = make_run(tmp_path / 'runs', 'zoned-time')
    path = run / 'evaluations/epoch_0000.json'
    data = json.loads(path.read_text())
    data['created_utc'] = '2026-10-08T22:12:08.337830+03:00'
    write(path, data)
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert quality['history'][0]['evaluated_at'] == '2026-10-08T19:12:08.337830+00:00'


def compact_transfer(run, *, wins=1, environment=HASH, difficulty=1, cohort=HASH):
    checkpoint = run / 'checkpoints/rollout_0000'
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    strata = [{'id': f'{profile}_{language}', 'profile': profile, 'language': language,
               'protocol': {'difficulty': difficulty, 'level_bank_seed': 20261008,
                            'counts': dict.fromkeys(TASKS, 2),
                            'policies': ['frozen_deterministic']}}
              for profile in ('canonical96', 'varied96_slate_left',
                              'varied64_violet_bottom', 'varied128_violet_top')
              for language in ('ru', 'en')]
    protocol = {'format': 'carryrl.robustness_matrix.v1',
                'checkpoint': {'path': str(checkpoint), 'files_sha256': {
                    'manifest.json': hashlib.sha256(
                        (checkpoint / 'manifest.json').read_bytes()).hexdigest()}},
                'environment_source_sha256': {'atomic': environment},
                'strata': strata, 'coverage_mode': 'seeded_diagnostic_subset'}
    protocol['matrix_sha256'] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    directory = run / 'diagnostics/robustness_matrix'
    write(directory / 'protocol.json', protocol)
    write(directory / 'aggregate.json', {
        **protocol, 'status': 'completed', 'weights_revision': manifest['weights_revision'],
        'metrics': {s['id']: {'frozen_deterministic': {
            'episodes_started': 32, 'unknown_outcomes': 0, 'outcomes': {'success': wins * 16}}}
                    for s in strata},
        'mechanical_cohorts': {s['id']: {f'frozen_deterministic/{task}': {
            'episodes_started': 2, 'episodes': [
                {'episode_index': index, 'seed': index, 'level_index': None,
                 'level_id': cohort, 'semantic_level_id': cohort, 'runtime_fingerprint': cohort,
                 'difficulty': difficulty, 'action_seed': index, 'initial_frame_sha256': HASH}
                for index in range(2)]} for task in TASKS} for s in strata}})
    write(run / 'diagnostics/memory_controls.json', {
        'diagnostic': 'paired_memory_cues', 'policy_and_reward_frozen': True,
        'checkpoint': str(checkpoint), 'weights_revision': manifest['weights_revision'],
        'checkpoint_sha256': protocol['checkpoint']['files_sha256'],
        'created_utc': '2026-10-09T00:00:00+00:00',
        'summaries': {mode: {'episodes': 4, 'first_choose_correct': 3, 'first_choose_observed': 4,
                             'paired_both_first_choose_correct': 1, 'geometries': 2}
                      for mode in ('normal', 'neutral_wait')},
        'records': [{'not_exposed': True}]})


def test_transfer_completed_compares_models_without_changing_canonical_curve(tmp_path):
    first = make_run(tmp_path / 'runs', 'first', revision=116)
    second = make_run(tmp_path / 'runs', 'second', source=first, revision=132)
    reader = reader_for(tmp_path, second)
    quality_before = reader.snapshot()['quality']
    compact_transfer(first)
    compact_transfer(second, wins=2)
    snapshot = reader.snapshot()
    transfer = snapshot['diagnostics']['transfer']
    assert transfer['current_weights'] is True
    assert transfer['run_id'] == 'second'
    assert len(transfer['rows']) == 8
    assert all(row['change_pp'] == 50 for row in transfer['rows'])
    assert transfer['checkpoint'].endswith('second/checkpoints/rollout_0000')
    assert transfer['evaluated_at'] is not None
    assert snapshot['quality'] == quality_before
    assert snapshot['diagnostics']['memory']['modes'][0]['first_choose_correct'] == 3
    assert snapshot['diagnostics']['memory']['current_weights'] is True
    assert 'records' not in snapshot['diagnostics']['memory']
    assert reader.run_details('second')['diagnostics'] == snapshot['diagnostics']


def test_transfer_current_missing_labels_ancestor_model_and_memory(tmp_path, monkeypatch):
    first = make_run(tmp_path / 'runs', 'first', revision=116)
    second = make_run(tmp_path / 'runs', 'second', source=first, revision=132)
    compact_transfer(first)
    # Prefer the corrected environment report over the older matrix.
    (first / 'diagnostics/robustness_matrix').rename(
        first / 'diagnostics/robustness_corrected_environment')
    reader = reader_for(tmp_path, second)
    original_read = reader._read

    def compact_only(path, **kwargs):
        assert not ('diagnostics' in path.parts and path.name not in {
            'aggregate.json', 'protocol.json', 'memory_controls.json'})
        return original_read(path, **kwargs)

    monkeypatch.setattr(reader, '_read', compact_only)
    diagnostics = reader.snapshot()['diagnostics']
    assert diagnostics['current_report_available'] is False
    assert diagnostics['transfer']['run_id'] == 'first'
    assert diagnostics['transfer']['current_weights'] is False
    assert diagnostics['memory']['run_id'] == 'first'
    assert diagnostics['memory']['current_weights'] is False
    assert all(row['change_pp'] is None for row in diagnostics['transfer']['rows'])
    assert 'текущих весов ещё не опубликована' in diagnostics['comparison_reason']


@pytest.mark.parametrize('mismatch', ['environment', 'difficulty', 'cohort'])
def test_transfer_mismatched_protocol_environment_or_cohort_has_no_delta(tmp_path, mismatch):
    first = make_run(tmp_path / 'runs', 'first', revision=116)
    second = make_run(tmp_path / 'runs', 'second', source=first, revision=132)
    compact_transfer(first)
    compact_transfer(second, wins=2, **{mismatch: 0 if mismatch == 'difficulty' else REWARD})
    diagnostics = reader_for(tmp_path, second).snapshot()['diagnostics']
    assert diagnostics['current_report_available'] is True
    assert all(row['change_pp'] is None for row in diagnostics['transfer']['rows'])
    assert 'отличаются' in diagnostics['comparison_reason']


def test_transfer_modified_checkpoint_cannot_claim_current_weights(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    compact_transfer(run)
    checkpoint = run / 'checkpoints/rollout_0000/manifest.json'
    manifest = json.loads(checkpoint.read_text())
    manifest['weights_revision'] += 1
    write(checkpoint, manifest)
    snapshot = reader_for(tmp_path, run).snapshot()
    assert snapshot['diagnostics']['transfer'] is None
    assert snapshot['diagnostics']['memory']['current_weights'] is False
    assert any('отпечаток checkpoint' in warning for warning in snapshot['warnings'])


@pytest.mark.parametrize('revision', [None, -1, True, 1.5])
def test_transfer_invalid_previous_revision_never_allows_delta(tmp_path, revision):
    first = make_run(tmp_path / 'runs', 'first', revision=116)
    second = make_run(tmp_path / 'runs', 'second', source=first, revision=132)
    compact_transfer(first)
    compact_transfer(second, wins=2)
    path = first / 'diagnostics/robustness_matrix/aggregate.json'
    aggregate = json.loads(path.read_text())
    if revision is None:
        aggregate.pop('weights_revision')
    else:
        aggregate['weights_revision'] = revision
    write(path, aggregate)
    snapshot = reader_for(tmp_path, second).snapshot()
    assert snapshot['diagnostics']['current_report_available'] is True
    assert all(row['change_pp'] is None for row in snapshot['diagnostics']['transfer']['rows'])
    assert any('ревизии переноса' in warning for warning in snapshot['warnings'])


@pytest.mark.parametrize('field', ['episodes_started', 'success', 'unknown_outcomes'])
def test_transfer_boolean_counters_are_not_results(tmp_path, field):
    run = make_run(tmp_path / 'runs', 'first')
    compact_transfer(run)
    path = run / 'diagnostics/robustness_matrix/aggregate.json'
    aggregate = json.loads(path.read_text())
    metric = next(iter(aggregate['metrics'].values()))['frozen_deterministic']
    target = metric['outcomes'] if field == 'success' else metric
    target[field] = True
    write(path, aggregate)
    snapshot = reader_for(tmp_path, run).snapshot()
    assert snapshot['diagnostics']['transfer'] is None
    assert snapshot['diagnostics']['current_report_available'] is False
    assert any('повреждённые итоги переноса' in warning for warning in snapshot['warnings'])


@pytest.mark.parametrize('damage', ['empty', 'missing_task', 'empty_entry', 'count', 'descriptor'])
def test_transfer_incomplete_identical_cohorts_are_not_comparable(tmp_path, damage):
    first = make_run(tmp_path / 'runs', 'first', revision=116)
    second = make_run(tmp_path / 'runs', 'second', source=first, revision=132)
    for run in (first, second):
        compact_transfer(run)
        path = run / 'diagnostics/robustness_matrix/aggregate.json'
        aggregate = json.loads(path.read_text())
        for cohort in aggregate['mechanical_cohorts'].values():
            if damage == 'empty':
                cohort.clear()
            elif damage == 'missing_task':
                cohort.pop('frozen_deterministic/navigation')
            elif damage == 'empty_entry':
                cohort['frozen_deterministic/navigation'] = {}
            elif damage == 'count':
                cohort['frozen_deterministic/navigation']['episodes_started'] = 3
            else:
                cohort['frozen_deterministic/navigation']['episodes'][0].pop('runtime_fingerprint')
        write(path, aggregate)
    snapshot = reader_for(tmp_path, second).snapshot()
    assert snapshot['diagnostics']['transfer'] is None
    assert snapshot['diagnostics']['current_report_available'] is False
    assert any('эпизодов переноса' in warning for warning in snapshot['warnings'])


@pytest.mark.parametrize('damage', ['empty', 'missing_mode', 'missing_count', 'negative',
                                  'boolean', 'fractional', 'nonfinite', 'over_denominator',
                                  'zero_denominator'])
def test_memory_incomplete_or_invalid_summaries_never_display_as_current(tmp_path, damage):
    run = make_run(tmp_path / 'runs', 'first')
    compact_transfer(run)
    path = run / 'diagnostics/memory_controls.json'
    memory = json.loads(path.read_text())
    if damage == 'empty':
        memory['summaries'] = {}
    elif damage == 'missing_mode':
        memory['summaries'].pop('neutral_wait')
    elif damage == 'missing_count':
        memory['summaries']['normal'].pop('paired_both_first_choose_correct')
    else:
        values = {'negative': -1, 'boolean': True, 'fractional': 1.5,
                  'nonfinite': float('inf'), 'over_denominator': 5, 'zero_denominator': 0}
        field = 'first_choose_observed' if damage == 'zero_denominator' else 'first_choose_correct'
        memory['summaries']['normal'][field] = values[damage]
    write(path, memory)
    snapshot = reader_for(tmp_path, run).snapshot()
    assert snapshot['diagnostics']['memory'] is None
    assert any('memory_controls.json' in warning for warning in snapshot['warnings'])


@pytest.mark.parametrize('seed', [0, 20261008])
def test_intermediate_covering_bank_enters_history_without_final_filename(tmp_path, seed):
    run = new_bank_run(tmp_path / 'runs', 'first', seed=seed)
    intermediate = full_evaluation(2, 124, 'e' * 64, difficulty=1, seed=seed, wins=10)
    write(run / 'evaluations/epoch_0002_full_retention.json', intermediate)
    data = reader_for(tmp_path, run).snapshot()
    history = data['quality']['history']
    assert [point['epoch'] for point in history] == [0, 2, 4]
    assert [point['successes'] for point in history] == [128, 160, 192]
    assert data['quality']['quick'] is None
    assert data['quality']['latest_revision'] == 132


def test_final_epoch_sampled_32_remains_quick_next_to_intermediate_full(tmp_path):
    run = new_bank_run(tmp_path / 'runs', 'first')
    (run / 'evaluations/epoch_0004.json').rename(run / 'evaluations/epoch_0004_final.json')
    write(run / 'evaluations/epoch_0002_full_retention.json', full_evaluation(
        2, 124, 'e' * 64, difficulty=1, seed=20261008, wins=10))
    quick = full_evaluation(4, 132, 'd' * 64, difficulty=1, seed=20261008, wins=16)
    quick['records'] = [record for record in quick['records'] if record['episode_index'] < 2]
    quick['level_selection'] = 'sampled levels'
    write(run / 'evaluations/epoch_0004.json', quick)
    data = reader_for(tmp_path, run).snapshot()
    assert [point['epoch'] for point in data['quality']['history']] == [0, 2, 4]
    assert data['quality']['latest_revision'] == 132
    assert data['quality']['value'] == 75
    assert data['quality']['quick'] == {
        'value': 100, 'successes': 32, 'episodes': 32, 'epoch': 4}


@pytest.mark.parametrize('damage', ['incomplete', 'mismatch'])
def test_intermediate_covering_bank_requires_full_matching_cohort(tmp_path, damage):
    run = new_bank_run(tmp_path / 'runs', 'first')
    intermediate = full_evaluation(2, 124, 'e' * 64, difficulty=1, seed=20261008, wins=10)
    if damage == 'incomplete':
        intermediate['records'].pop()
    else:
        intermediate['records'][0]['initial_frame_sha256'] = REWARD
    write(run / 'evaluations/epoch_0002_full_retention.json', intermediate)
    data = reader_for(tmp_path, run).snapshot()
    assert [point['epoch'] for point in data['quality']['history']] == [0, 4]
    assert data['quality']['quick'] is None
    assert any(('256 эпизодов' if damage == 'incomplete' else 'протокол') in warning
               for warning in data['warnings'])


def ppo_from_parent(tmp_path):
    parent = new_bank_run(tmp_path / 'runs', 'parent', revision=17301)
    checkpoint = parent / 'checkpoints/rollout_0000'
    manifest = json.loads((checkpoint / 'manifest.json').read_text())
    manifest['reward_sha256'] = REWARD
    write(checkpoint / 'manifest.json', manifest)
    for path in parent.joinpath('evaluations').glob('*.json'):
        data = json.loads(path.read_text())
        data['protocol_sha256'] = HASH
        write(path, data)
    run = tmp_path / 'runs/ppo'
    write(run / 'metadata.json', {
        'kind': 'sequence_training', 'suite_version': 'test.v1', 'codec_version': 'test.v1',
        'reward_model_version': 'test.v1',
        'config': {'total_rollouts': 32, 'level_bank_seed': 20261008,
                   'max_episode_steps': 80, 'env_image_size': 96,
                   'visual_config': {'mode': 'varied'}},
        'warm_start_parent': {'checkpoint': str(checkpoint), 'checkpoint_sha256': HASH,
                              'agent_sha256': manifest['agent_sha256'], 'reward_sha256': REWARD,
                              'source_counters': {'weights_revision': 17301}}})
    write(run / 'status.json', {
        'status': 'running', 'phase': 'training', 'pid': 123, 'updated_at': 1000,
        'rollouts': 0, 'rollouts_total': 32, 'optimizer_steps_this_process': 0,
        'optimizer_steps_upper_bound': 1024, 'weights_revision': 17301})
    return run, parent


def ppo_evaluation(rollout, revision, agent='b' * 64, *, sampled=True, frame=HASH):
    data = full_evaluation(0, revision, agent, difficulty=1, seed=20261008, wins=14)
    data.pop('epoch')
    data.update(rollout=rollout, reward_sha256=REWARD, protocol_sha256=HASH)
    if sampled:
        data['records'] = [r for r in data['records'] if r['episode_index'] < 2]
        data['level_selection'] = 'sampled levels'
        data['protocol_sha256'] = 'e' * 64
        for record in data['records']:
            record['presentation'] = {'mode': 'varied', 'native_sizes': [64]}
    data['records'][0]['initial_frame_sha256'] = frame
    return data


def test_ppo_inherits_verified_parent_measurement_without_claiming_new_weights(tmp_path):
    run, parent = ppo_from_parent(tmp_path)
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert quality['available'] is True
    assert quality['latest_run_id'] == parent.name
    assert quality['latest_revision'] == 17301
    assert quality['value'] == 75
    assert quality['canonical_measurement_current'] is False
    assert quality['inherited_from']['run_id'] == parent.name
    assert quality['diagnostic_series'] == []
    assert quality['quick'] is None


@pytest.mark.parametrize('field', ['checkpoint_sha256', 'agent_sha256', 'reward_sha256'])
def test_ppo_parent_lineage_requires_matching_saved_hashes(tmp_path, field):
    run, _ = ppo_from_parent(tmp_path)
    metadata = json.loads((run / 'metadata.json').read_text())
    metadata['warm_start_parent'][field] = 'f' * 64
    write(run / 'metadata.json', metadata)
    snapshot = reader_for(tmp_path, run).snapshot()
    assert snapshot['quality']['available'] is False
    assert snapshot['quality']['inherited_from'] is None
    assert any('происхождение весов' in warning for warning in snapshot['warnings'])


def test_ppo_sampled_varied_rollouts_form_separate_series_not_canonical_curve(tmp_path):
    run, parent = ppo_from_parent(tmp_path)
    for rollout, revision in ((1, 17304), (2, 17308)):
        write(run / f'evaluations/rollout_{rollout:04}.json', ppo_evaluation(rollout, revision))
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert quality['latest_run_id'] == parent.name
    assert quality['value'] == 75
    assert all(point['run_id'] == parent.name for point in quality['history'])
    assert len(quality['diagnostic_series']) == 1
    series = quality['diagnostic_series'][0]
    assert series['coverage'] == 'sampled'
    assert [point['rollout'] for point in series['history']] == [1, 2]
    assert [point['episodes'] for point in series['history']] == [32, 32]
    assert [point['updates'] for point in series['history']] == [3, 7]
    assert quality['quick'] is None


@pytest.mark.parametrize('damage', ['presentation', 'protocol'])
def test_ppo_distinct_diagnostic_protocols_cannot_share_series(tmp_path, damage):
    run, _ = ppo_from_parent(tmp_path)
    first = ppo_evaluation(1, 17304)
    second = ppo_evaluation(2, 17308)
    if damage == 'presentation':
        second['records'][0]['presentation']['native_sizes'] = [128]
    else:
        second['protocol_sha256'] = REWARD
    write(run / 'evaluations/rollout_0001.json', first)
    write(run / 'evaluations/rollout_0002.json', second)
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert len(quality['diagnostic_series']) == 2
    assert quality['value'] == 75


@pytest.mark.parametrize('frame', [HASH, REWARD])
def test_ppo_full_canonical_joins_curve_only_with_same_protocol_and_episode_bank(tmp_path, frame):
    run, parent = ppo_from_parent(tmp_path)
    write(run / 'evaluations/rollout_0032_canonical_full.json', ppo_evaluation(
        32, 17400, sampled=False, frame=frame))
    quality = reader_for(tmp_path, run).snapshot()['quality']
    if frame == HASH:
        assert quality['latest_run_id'] == run.name
        assert quality['value'] == 87.5
        assert quality['history'][-1]['rollout'] == 32
        assert quality['canonical_measurement_current'] is True
        assert quality['diagnostic_series'] == []
    else:
        assert quality['latest_run_id'] == parent.name
        assert quality['value'] == 75
        assert quality['canonical_measurement_current'] is False
        assert quality['diagnostic_series'][0]['coverage'] == 'full'


def test_ppo_copied_parent_weights_do_not_become_a_new_canonical_point(tmp_path):
    run, parent = ppo_from_parent(tmp_path)
    write(run / 'evaluations/rollout_0000.json', ppo_evaluation(
        0, 17301, 'd' * 64, sampled=False))
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert quality['latest_run_id'] == parent.name
    assert quality['canonical_measurement_current'] is False
    assert all(point['run_id'] == parent.name for point in quality['history'])


def test_ppo_local_updates_and_cycle_eta_use_measured_completed_cycles(tmp_path):
    run, _ = ppo_from_parent(tmp_path)
    reader = reader_for(tmp_path, run, clock=lambda: 1010, process_alive=lambda pid: True)
    current = reader.snapshot()['current']
    assert current['optimizer_steps'] == 0
    assert current['weights_revision'] == 17301
    assert current['optimizer_steps_upper_bound'] == 1024
    assert current['cycle_eta_seconds'] is None
    status = json.loads((run / 'status.json').read_text())
    status.update(rollouts=2, optimizer_steps_this_process=7, weights_revision=17308)
    write(run / 'status.json', status)
    rows = [{'rollout': 1, 'cycle_wall_seconds': 10}, {'rollout': 2, 'cycle_wall_seconds': 20},
            {'rollout': 3, 'complete': False, 'cycle_wall_seconds': 999}]
    (run / 'training.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    current = reader.snapshot()['current']
    assert current['optimizer_steps'] == 7
    assert current['optimizer_steps_total'] is None
    assert current['training_percent'] == 6.25
    assert current['cycle_eta_seconds'] == 450
    assert current['cycle_eta_low_seconds'] == 300
    assert current['cycle_eta_high_seconds'] == 600
    reader.process_alive = lambda pid: False
    assert reader.snapshot()['current']['cycle_eta_seconds'] is None


def test_ppo_protocol_anchor_is_exact_accepted_parent_measurement(tmp_path):
    run, parent = ppo_from_parent(tmp_path)
    unrelated = ppo_evaluation(0, 17301, 'd' * 64, sampled=False)
    unrelated['protocol_sha256'] = REWARD
    unrelated['records'] = []
    write(parent / 'evaluations/aaa_other_full.json', unrelated)
    current = ppo_evaluation(32, 17400, sampled=False)
    current['protocol_sha256'] = REWARD
    write(run / 'evaluations/rollout_0032.json', current)
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert quality['latest_run_id'] == parent.name
    assert quality['canonical_measurement_current'] is False
    assert quality['diagnostic_series'][0]['protocol_sha256'] == REWARD


def test_ppo_changed_agent_requires_revision_increase(tmp_path):
    run, parent = ppo_from_parent(tmp_path)
    write(run / 'evaluations/rollout_0032.json', ppo_evaluation(32, 17301, sampled=False))
    snapshot = reader_for(tmp_path, run).snapshot()
    assert snapshot['quality']['latest_run_id'] == parent.name
    assert snapshot['quality']['canonical_measurement_current'] is False
    assert snapshot['quality']['diagnostic_series'] == []
    assert any('идентичности проверки PPO' in warning for warning in snapshot['warnings'])


@pytest.mark.parametrize('phase', ['collecting', 'optimizing', 'evaluation', 'final_evaluation'])
def test_ppo_runtime_phase_plan_and_actual_stage_units(tmp_path, phase):
    run, _ = ppo_from_parent(tmp_path)
    status = json.loads((run / 'status.json').read_text())
    status.update(phase=phase, current_rollout=1, phase_started_at=1000,
                  phase_units_done=4, phase_units_total=64, phase_unit_kind='episodes',
                  phase_plan=[{'id': 'preparation', 'phases': ['loading']},
                              {'id': 'baseline', 'phases': ['initial_evaluation']},
                              {'id': 'training',
                               'phases': ['collecting', 'optimizing', 'evaluation']},
                              {'id': 'final_evaluation', 'phases': ['final_evaluation']},
                              {'id': 'saving', 'phases': ['saving']},
                              {'id': 'completed', 'phases': ['finished']}])
    if phase == 'optimizing':
        status.update(phase_units_total=None, phase_unit_kind='optimizer_updates',
                      phase_units_done=0)
    write(run / 'status.json', status)
    current = reader_for(tmp_path, run, clock=lambda: 1010,
                         process_alive=lambda pid: True).snapshot()['current']
    assert current['phase_total'] == 6
    assert current['phase_index'] == (4 if phase == 'final_evaluation' else 3)
    assert current['phase_steps'][current['phase_index'] - 1]['state'] == 'active'
    assert current['current_rollout'] == 1
    assert current['rollouts'] == 0
    assert current['optimizer_steps'] == 0
    assert current['cycle_eta_seconds'] is None
    if phase == 'optimizing':
        assert current['phase_label'] == 'Обновление политики PPO'
        assert current['eta_seconds'] is None
        assert current['phase_units_total'] is None
    else:
        assert current['eta_seconds'] == 150
        assert current['eta_scope'] == 'phase'


def test_ppo_resource_stop_preserves_actual_cycles_and_local_updates(tmp_path):
    run, _ = ppo_from_parent(tmp_path)
    status = json.loads((run / 'status.json').read_text())
    status.update(status='resource_stopped', phase='finished', rollouts=2,
                  optimizer_steps_this_process=7, weights_revision=17308,
                  stop_reason='minimum_free_disk_mib')
    write(run / 'status.json', status)
    current = reader_for(tmp_path, run, process_alive=lambda pid: False).snapshot()['current']
    assert current['status'] == 'resource_stopped'
    assert current['rollouts'] == 2
    assert current['rollouts_total'] == 32
    assert current['optimizer_steps'] == 7
    assert current['stop_reason'] == 'minimum_free_disk_mib'
    assert current['cycle_eta_seconds'] is None
    assert current['training_percent'] == 6.25


@pytest.mark.parametrize('terminal', ['completed', 'resource_stopped'])
def test_ppo_six_group_finished_snapshot_preserves_legacy_phase_bounds(tmp_path, terminal):
    run, _ = ppo_from_parent(tmp_path)
    status = json.loads((run / 'status.json').read_text())
    status.update(status=terminal, phase='finished', phase_index=6,
                  phase_plan=[{'id': str(index), 'phases': [
                      'finished' if index == 6 else f'phase_{index}']}
                      for index in range(1, 7)])
    write(run / 'status.json', status)
    current = reader_for(tmp_path, run, process_alive=lambda pid: False).snapshot()['current']
    assert current['status'] == terminal
    assert current['phase_index'] == current['phase_total'] == 6
    assert current['phase_label'] == 'Завершено'
    assert all(step['state'] == 'complete' for step in current['phase_steps'])


@pytest.mark.parametrize('reason', ['minimum_free_disk_mib', 'max_wall_seconds'])
def test_ppo_cycle_eta_is_cancelled_as_soon_as_stop_reason_persists(tmp_path, reason):
    run, _ = ppo_from_parent(tmp_path)
    status = json.loads((run / 'status.json').read_text())
    status.update(rollouts=3, optimizer_steps_this_process=37, weights_revision=17338,
                  phase='final_evaluation', phase_started_at=1000,
                  phase_units_done=4, phase_units_total=64, stop_reason=reason)
    write(run / 'status.json', status)
    (run / 'training.jsonl').write_text('\n'.join(json.dumps({
        'rollout': index, 'complete': True, 'cycle_wall_seconds': 60}) for index in range(1, 4)))
    current = reader_for(tmp_path, run, clock=lambda: 1010,
                         process_alive=lambda pid: True).snapshot()['current']
    assert current['status'] == 'running'
    assert current['optimizer_steps'] == 37
    assert current['rollouts'] == 3
    assert current['cycle_eta_seconds'] is None
    assert current['cycle_eta_low_seconds'] is None
    assert current['cycle_eta_high_seconds'] is None
    assert current['stop_reason'] == reason
    assert current['eta_scope'] == 'phase'
    assert current['eta_seconds'] == 150


def test_ppo_actions_counter_exposes_local_attempts_without_changing_cumulative(tmp_path):
    run = tmp_path / "runs/ppo"
    write(run / "metadata.json", {"kind": "sequence_training",
                                  "config": {"total_rollouts": 32}})
    write(run / "status.json", {"status": "running", "decisions": 41015,
                                "attempted_train_decisions_this_process": 37})
    current = reader_for(tmp_path, run)._current(run, {})
    assert current["attempted_train_decisions_this_process"] == 37
    assert current["decisions"] == 41015


def test_ppo_actions_counter_missing_or_invalid_remains_unknown(tmp_path):
    run = tmp_path / "runs/ppo"
    write(run / "metadata.json", {"kind": "sequence_training",
                                  "config": {"total_rollouts": 32}})
    reader = reader_for(tmp_path, run)
    for value in (None, -1, True, 1.5):
        status = {"status": "running", "decisions": 41015}
        if value is not None:
            status["attempted_train_decisions_this_process"] = value
        write(run / "status.json", status)
        current = reader._current(run, {})
        assert current["attempted_train_decisions_this_process"] is None
        assert current["decisions"] == 41015


def test_snapshot_reuses_evidence_but_refreshes_process_and_source_changes(tmp_path, monkeypatch):
    run = make_run(tmp_path / 'runs', 'first')
    write(run / 'status.json', {'status': 'running', 'pid': 123, 'updated_at': 1000})
    reader = reader_for(tmp_path, run, clock=lambda: 1010, process_alive=lambda pid: True)
    reader.snapshot()
    original = reader._quality_data
    computed = []

    def counted(chain):
        computed.append(tuple(path.name for path in chain))
        return original(chain)

    monkeypatch.setattr(reader, '_quality_data', counted)
    reader.process_alive = lambda pid: False
    cached = reader.snapshot()
    assert not computed
    assert cached['current']['status'] == 'interrupted'
    assert cached['experiments'][0]['status'] == 'interrupted'
    write(run / 'evaluations/epoch_0004.json', evaluation(4, 140, 'b' * 64, wins=16))
    updated = reader.snapshot()
    assert computed
    assert updated['quality']['value'] == 100
    assert updated['experiments'][0]['experiment']['measurement']['value'] == 100
    computed.clear()
    write(run / 'evaluations/epoch_0005.json', evaluation(5, 144, 'd' * 64, wins=4))
    assert reader.snapshot()['quality']['value'] == 25
    assert computed


def test_scope_exposes_unrelated_run_media_and_preserves_registry(tmp_path):
    root = tmp_path / 'runs'
    active = make_run(root, 'active')
    other = make_run(root, 'other', revision=200, agent='e' * 64)
    _, gif = add_playback(other)
    reader = reader_for(tmp_path, active)
    scoped = reader.snapshot(run_id='other')
    assert scoped['current']['run_id'] == scoped['selected_run_id'] == 'other'
    assert {item['run_id'] for item in scoped['experiments']} == {'active', 'other'}
    item = scoped['playbacks']['items'][0]
    assert item['run_id'] == 'other'
    assert item['experiment_title'] == scoped['current']['label']
    assert item['checkpoint'] == str(other / 'checkpoints/rollout_0000')
    assert reader.media('other', 'actual.gif') == gif
    assert reader.run_details('other')['playbacks']['items'][0]['id'] == item['id']
    assert reader.snapshot()['selected_run_id'] == 'active'


def test_experiment_conclusions_use_own_measurement_and_unknown_stays_unknown(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    reader = reader_for(tmp_path, run)
    experiment = reader.snapshot()['current']['experiment']
    assert experiment['measurement']['run_id'] == 'first'
    assert experiment['measurement']['value'] == 50
    assert '50.0%' in experiment['conclusion']
    assert any(source['path'] == 'evaluations/epoch_0004.json'
               for source in experiment['sources'])
    unknown = tmp_path / 'runs/unknown'
    write(unknown / 'metadata.json', {'opaque': True})
    detail = reader.run_details('unknown')['experiment']
    assert detail['title'] == 'Запуск без описания'
    assert 'не записано' in detail['purpose']
    assert detail['measurement'] is None
    assert 'не опубликован' in detail['conclusion']


def test_launch_process_evidence_distinguishes_failure_from_interruption(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    write(run / 'status.json', {'status': 'running', 'updated_at': 1000})
    launch = tmp_path / 'runs/first_launch/process.json'
    write(launch, {'command_pid': 123, 'status': 'running'})
    reader = reader_for(tmp_path, run, process_alive=lambda pid: False)
    assert reader.snapshot()['current']['status'] == 'interrupted'
    write(launch, {'command_pid': 123, 'status': 'exited', 'actual_exit_code': 2})
    assert reader.snapshot()['current']['status'] == 'failed'
    write(run / 'status.json', {'status': 'completed'})
    assert reader.snapshot()['current']['status'] == 'completed'


def test_maximum_score_is_separate_from_wins_and_requires_scoring_version(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    for epoch, revision, agent, wins in ((0, 124, HASH, 0), (4, 140, 'b' * 64, 8)):
        data = evaluation(epoch, revision, agent, wins=wins)
        data['scoring_version'] = 'carryrl.atomic.efficient.v1'
        for row in data['records']:
            row['outcome']['score'] = 5 if row['task'] == TASKS[0] else 3
        write(run / f'evaluations/epoch_{epoch:04}.json', data)
    reader = reader_for(tmp_path, run)
    quality = reader.snapshot()['quality']
    assert quality['value'] == 50
    assert quality['perfect_score_successes'] == 3
    assert quality['perfect_score_percent'] == 6.25
    assert quality['games'][0]['perfect_score_percent'] == 100
    assert quality['scoring_version'] == 'carryrl.atomic.efficient.v1'
    for epoch in (0, 4):
        path = run / f'evaluations/epoch_{epoch:04}.json'
        data = json.loads(path.read_text())
        data.pop('scoring_version')
        write(path, data)
    quality = reader.snapshot()['quality']
    assert quality['value'] == 50
    assert quality['perfect_score_percent'] is None
    assert 'Прежняя версия' in quality['scoring_version_label']


def test_new_scoring_cohort_does_not_merge_with_legacy_scores(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    new = evaluation(4, 140, 'b' * 64, wins=16)
    new['scoring_version'] = 'carryrl.atomic.efficient.v1'
    write(run / 'evaluations/epoch_0004.json', new)
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert quality['value'] == 0
    assert len(quality['history']) == 1


def test_human_experiment_sidecar_and_storage_are_fresh_source_backed(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    reader = reader_for(tmp_path, run)
    reader.snapshot()
    write(run / 'experiment.json', {'title': 'Проверка цветов',
                                   'purpose': 'Проверить новые цвета.',
                                   'conclusion': 'Отчёт ещё не готов.',
                                   'limitations': ['Один запуск.']})
    write(run / 'storage_manifest.json', {'format': 'carryrl-checkpoint-storage-v1',
                                         'checkpoints': {'checkpoints/rollout_0000': {
                                             'state': 'archived', 'requires_restore': True}}})
    snapshot = reader.snapshot()
    assert snapshot['current']['label'] == 'Проверка цветов'
    assert snapshot['experiments'][0]['experiment']['conclusion'] == 'Отчёт ещё не готов.'
    assert snapshot['experiments'][0]['storage']['checkpoints'][
        'checkpoints/rollout_0000']['requires_restore'] is True
    assert reader.run_details('first')['experiment']['limitations'] == ['Один запуск.']


def test_own_scoring_baseline_replaces_legacy_lineage_cohort(tmp_path):
    root = tmp_path / 'runs'
    old = new_bank_run(root, 'legacy', revision=116)
    fresh = new_bank_run(root, 'new-scoring', source=old, revision=132)
    for path in (fresh / 'evaluations').glob('epoch_*.json'):
        data = json.loads(path.read_text())
        data['scoring_version'] = 'carryrl.atomic.efficient.v1'
        if data['epoch'] == 4:
            data['agent_sha256'] = 'e' * 64
        for row in data['records']:
            row['outcome']['score'] = 5 if row['outcome']['success'] else 0
        write(path, data)
    quality = reader_for(tmp_path, fresh).snapshot()['quality']
    assert quality['latest_run_id'] == 'new-scoring'
    assert all(point['run_id'] == 'new-scoring' for point in quality['history'])
    assert quality['value'] == quality['perfect_score_percent'] == 75
    assert quality['perfect_score_run_id'] == 'new-scoring'
    assert quality['perfect_score_episodes'] == quality['episodes'] == 256


def test_only_own_initial_measurement_never_claims_lineage_gain(tmp_path):
    root = tmp_path / 'runs'
    parent = make_run(root, 'parent', revision=116, agent='b' * 64)
    active = make_run(root, 'trial', source=parent, revision=132, agent='d' * 64)
    source = parent / 'evaluations/epoch_0004.json'
    original = json.loads(source.read_text())
    original['created_utc'] = '2026-10-09T01:00:00Z'
    original['protocol_sha256'] = HASH
    write(source, original)
    copied = {**original, 'epoch': 0, 'initial_evaluation_reuse': {
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_created_utc': original['created_utc'], 'source_epoch': 4,
        'source_weights_revision': 116, 'measurement_reused': True}}
    write(active / 'evaluations/epoch_0000.json', copied)
    (active / 'evaluations/epoch_0004.json').unlink()
    write(active / 'status.json', {'status': 'running', 'epochs_completed': 3,
                                 'epoch': 4, 'optimizer_steps': 13, 'pid': 123})
    quality = reader_for(tmp_path, active, process_alive=lambda pid: False).snapshot()['quality']
    assert quality['measurement_stage'] == 'initial'
    assert quality['value'] == quality['baseline'] == 50
    assert quality['change_pp'] is None
    assert quality['historical_comparison']['change_pp'] == 50
    assert quality['measurement_run_id'] == 'parent'
    assert quality['measurement_source'] == str(source)
    assert quality['measurement_container_run_id'] == 'trial'
    assert quality['measurement_reused'] is True
    assert quality['current_weights_measured'] is False
    assert quality['run_history'][0]['epoch'] == 0
    assert all(game['change_pp'] is None for game in quality['games'])


def test_own_post_training_delta_uses_own_initial_not_first_ancestor(tmp_path):
    root = tmp_path / 'runs'
    parent = make_run(root, 'parent', revision=116, agent='b' * 64)
    active = make_run(root, 'trial', source=parent, revision=132, agent='d' * 64)
    write(active / 'evaluations/epoch_0000.json', evaluation(0, 116, 'b' * 64, wins=8))
    write(active / 'evaluations/epoch_0004.json', evaluation(4, 132, 'd' * 64, wins=12))
    quality = reader_for(tmp_path, active).snapshot()['quality']
    assert quality['measurement_stage'] == 'final'
    assert quality['baseline'] == 50
    assert quality['value'] == 75
    assert quality['change_pp'] == 25
    assert quality['historical_comparison']['change_pp'] == 75
    assert quality['current_weights_measured'] is True


def test_reused_standalone_frozen_report_has_original_source_without_epoch(tmp_path):
    run = make_run(tmp_path / 'runs', 'trial')
    source = tmp_path / 'runs/control/evaluations/efficient_baseline.json'
    original = evaluation(0, 124, HASH, wins=8)
    original.pop('epoch')
    original.update(created_utc='2026-10-13T13:33:49Z', protocol_sha256=HASH,
                    reward_sha256=REWARD)
    write(source, original)
    copied = {**original, 'epoch': 0, 'initial_evaluation_reuse': {
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_created_utc': original['created_utc'], 'source_epoch': None,
        'source_weights_revision': 124, 'measurement_reused': True}}
    write(run / 'evaluations/epoch_0000.json', copied)
    (run / 'evaluations/epoch_0004.json').unlink()
    reader = reader_for(tmp_path, run)
    quality = reader.snapshot()['quality']
    assert quality['measurement_stage'] == 'initial'
    assert quality['measurement_run_id'] == 'control'
    assert quality['measurement_source'] == str(source)
    assert quality['measurement_reused'] is True
    copied['initial_evaluation_reuse']['source_sha256'] = 'e' * 64
    write(run / 'evaluations/epoch_0000.json', copied)
    quality = reader.snapshot()['quality']
    assert quality['measurement_run_id'] == 'trial'
    assert quality['measurement_reused'] is False


def test_picker_verdict_compares_own_baseline_and_recommends_verified_source(tmp_path):
    root = tmp_path / 'runs'
    parent = make_run(root, 'control', revision=116, agent='b' * 64)
    run = make_run(root, 'new', source=parent, revision=132, agent='d' * 64)
    write(run / 'evaluations/epoch_0000.json', evaluation(0, 116, 'b' * 64, wins=8))
    write(run / 'evaluations/epoch_0004.json', evaluation(4, 132, 'd' * 64, wins=4))
    reader = reader_for(tmp_path, run)
    experiment = reader.snapshot()['current']['experiment']
    # Epoch-zero provenance owns the copied measurement only when SHA verified.
    assert experiment['verdict']['state'] == 'worse'
    assert experiment['verdict']['change_pp'] == -25
    assert experiment['recommended_run_id'] is None
    assert '6/18 побед' in experiment['picker_label']
    assert all(term not in experiment['picker_label'] for term in ('MLX', 'эпох', 'пример'))
    original = json.loads((parent / 'evaluations/epoch_0004.json').read_text())
    original.update(created_utc='2026-10-13T13:33:49Z', protocol_sha256=HASH)
    source = parent / 'evaluations/epoch_0004.json'
    write(source, original)
    copied = {**original, 'epoch': 0, 'initial_evaluation_reuse': {
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_created_utc': original['created_utc'], 'source_epoch': 4,
        'source_weights_revision': 116, 'measurement_reused': True}}
    write(run / 'evaluations/epoch_0000.json', copied)
    experiment = reader.snapshot()['current']['experiment']
    assert experiment['recommended_run_id'] == 'control'
    assert '16 мини-игр' in experiment['recommendation']
    (run / 'evaluations/epoch_0004.json').unlink()
    write(run / 'status.json', {'status': 'running', 'pid': 123})
    reader.process_alive = lambda pid: False
    experiment = reader.snapshot()['current']['experiment']
    assert experiment['verdict']['state'] == 'pending'
    assert experiment['recommended_run_id'] is None
    assert 'без итоговой проверки' in experiment['picker_label']


def test_scoped_model_keeps_actual_registry_training_visible_and_live(tmp_path):
    root = tmp_path / 'runs'
    live = make_run(root, 'live')
    other = make_run(root, 'selected')
    write(live / 'status.json', {'status': 'running', 'pid': 123, 'optimizer_steps': 2})
    reader = reader_for(tmp_path, live, process_alive=lambda pid: True)
    scoped = reader.snapshot(run_id=other.name)
    assert scoped['selected_run_id'] == 'selected'
    assert scoped['current']['run_id'] == 'selected'
    assert scoped['training_current']['run_id'] == 'live'
    assert scoped['training_current']['running'] is True
    reader.process_alive = lambda pid: False
    cached = reader.snapshot(run_id=other.name)
    assert cached['training_current']['running'] is False
    assert cached['training_current']['status'] == 'interrupted'


def test_picker_date_uses_user_moscow_timezone_without_technical_timezone_text(tmp_path):
    run = make_run(tmp_path / 'runs', 'trial')
    status = json.loads((run / 'status.json').read_text())
    status['started_at'] = '2026-10-13T13:39:05Z'
    write(run / 'status.json', status)
    label = reader_for(tmp_path, run).snapshot()['current']['experiment']['picker_label']
    assert label.startswith('13.10 16:39 · ')
    assert 'UTC' not in label
    status['started_at'] = '2026-10-13T22:39:05Z'
    write(run / 'status.json', status)
    label = reader_for(tmp_path, run).snapshot()['current']['experiment']['picker_label']
    assert label.startswith('14.10 01:39 · ')
