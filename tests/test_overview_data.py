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
    assert data['current']['evaluation_only'] is False


def test_all_game_guides_explain_visible_objects(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    guides = reader_for(tmp_path, run).snapshot()['task_guides']
    assert set(guides) == set(TASKS)
    for guide in guides.values():
        assert isinstance(guide['legend'], list)
        assert 2 <= len(guide['legend']) <= 4
        assert all(isinstance(item, str) and item.strip() for item in guide['legend'])
    interaction = ' '.join(guides['interaction']['legend'])
    assert 'Голубой прямоугольник — ваш персонаж' in interaction
    assert 'Жёлтая клетка — предмет' in interaction
    assert 'после подбора он исчезает' in interaction
    assert 'Фиолетовая площадка — место использования предмета' in interaction
    assert 'Красная клетка — опасность' in interaction


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
    metadata.update(terminated=False, truncated=True, sprite_filename=sprite.name,
                    sprite_sha256=hashlib.sha256(sprite.read_bytes()).hexdigest(),
                    sprite_columns=8, frame_width=96, frame_height=96, frame_count=6,
                    frame_duration_ms=100, terminal_hold_ms=1200,
                    actions=[{'action': {'keys': ['right'], 'mouse_buttons': [],
                                         'dx': 0, 'dy': 0, 'wheel': 0}}] * 5)
    write(sidecar, metadata)
    reader = reader_for(tmp_path, run)
    item = reader.snapshot()['playbacks']['items'][0]
    assert item['sprite_url'] == '/media/first/actual.frames.png'
    assert item['terminated'] is False
    assert item['truncated'] is True
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


@pytest.mark.parametrize('observed', [0, 2])
def test_memory_reports_all_attempts_including_missing_choices(tmp_path, observed):
    run = make_run(tmp_path / 'runs', 'first')
    compact_transfer(run)
    path = run / 'diagnostics/memory_controls.json'
    memory = json.loads(path.read_text())
    memory['summaries']['normal'].update(
        first_choose_observed=observed, first_choose_correct=min(1, observed),
        paired_both_first_choose_correct=0)
    write(path, memory)
    row = reader_for(tmp_path, run).snapshot()['diagnostics']['memory']['modes'][0]
    assert row['episodes'] == 4
    assert row['first_choose_observed'] == observed
    assert row['missing_choices'] == 4 - observed


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
                   'max_episode_steps': 80, 'env_image_size': 96},
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


def test_saved_comment_cannot_replace_completed_experiment_evidence(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    reader = reader_for(tmp_path, run)
    computed = reader.snapshot()['current']['experiment']
    comment = {'conclusion': 'Проверка агента продолжается.',
               'limitations': ['Итоговая проверка ещё не завершена.']}
    write(run / 'experiment.json', comment)
    for experiment in (reader.snapshot()['current']['experiment'],
                       reader.run_details('first')['experiment']):
        assert experiment['conclusion'] == computed['conclusion']
        assert '50.0%' in experiment['conclusion']
        assert experiment['limitations'] == computed['limitations']
        assert experiment['saved_comment'] == {**comment, 'source': 'experiment.json'}


@pytest.mark.parametrize('evaluation_metadata, expected_method, expected_title', [
    ({'updates': 0}, 'evaluate', 'Проверка сохранённого агента'),
    ({'command': 'evaluate'}, 'evaluate', 'Проверка сохранённого агента'),
    ({'command': 'random'}, 'random', 'Случайные действия: контроль'),
])
def test_evaluation_config_does_not_classify_run_as_training(
        tmp_path, evaluation_metadata, expected_method, expected_title):
    run = tmp_path / 'runs/evaluation'
    write(run / 'metadata.json', {**evaluation_metadata, 'config': {'total_rollouts': 32},
                                  'curriculum': {'warmup_rollouts': 4}})
    write(run / 'status.json', {'status': 'completed'})
    current = reader_for(tmp_path, run).snapshot()['current']
    assert current['method'] == expected_method
    assert current['evaluation_only'] is True
    assert 'PPO' not in current['method_description']
    assert current['experiment']['title'] == expected_title
    assert current['experiment']['display_name'] == expected_title
    assert current['experiment']['purpose'] == 'Измерить успешность игры без обновления весов.'


@pytest.mark.parametrize('metadata, title', [
    ({'kind': 'frozen_train_correction_collection',
      'config': {'total_rollouts': 32, 'device': 'cpu'},
      'curriculum': {'warmup_rollouts': 4}}, 'Сбор исправлений ошибок'),
    ({'command': 'prepare_history_features', 'updates': 0},
     'Подготовка признаков истории'),
    ({'command': 'prepare_history_reward', 'updates': 0},
     'Подготовка данных награды по истории'),
    ({'kind': 'train_correction_coverage_and_rescue', 'updates': 0},
     'Проверка покрытия исправлений'),
    ({'kind': 'frozen_imitation_diagnostics', 'config': {'total_rollouts': 32}},
     'Диагностика сохранённого агента'),
    ({'kind': 'frozen_teacher_rescue_audit', 'updates': 0},
     'Проверка помощи учителя'),
])
@pytest.mark.parametrize('status', ['running', 'completed'])
def test_technical_experiment_precedes_training_and_evaluation_heuristics(
        tmp_path, metadata, title, status):
    run = tmp_path / 'runs/technical'
    write(run / 'metadata.json', metadata)
    write(run / 'status.json', {'status': status, 'pid': 123})
    reader = reader_for(tmp_path, run, process_alive=lambda pid: True)
    current = reader.run_details(run.name)['current']
    assert current['technical_only'] is True
    assert current['evaluation_only'] is False
    assert current['method'] == metadata.get('command', metadata.get('kind'))
    assert current['rollouts_total'] is None
    assert current['phase_total'] == 1
    assert current['phase_steps'] == [{'label': title,
                                       'state': 'active' if status == 'running' else 'complete'}]
    experiment = reader.run_details(run.name)['experiment']
    assert experiment['title'] == title
    assert experiment['display_name'] == title
    assert 'PPO' not in experiment['method']
    assert experiment['measurement'] is None
    assert experiment['verdict']['label'] == 'технический запуск'
    assert experiment['recommended_run_id'] is None
    assert 'Дождитесь' not in experiment['recommendation']
    assert 'артефактам' in experiment['recommendation']
    assert json.loads((run / 'metadata.json').read_text()) == metadata
    editorial = {'title': 'Явное название', 'purpose': 'Явное назначение',
                 'method': 'Явный метод'}
    write(run / 'experiment.json', editorial)
    experiment = reader.run_details(run.name)['experiment']
    for key, value in editorial.items():
        assert experiment[key] == value
    assert json.loads((run / 'experiment.json').read_text()) == editorial


def test_ordinary_ppo_experiment_keeps_training_description(tmp_path):
    run = tmp_path / 'runs/ppo'
    write(run / 'metadata.json', {'kind': 'sequence_training',
                                 'config': {'total_rollouts': 32, 'device': 'cpu'}})
    write(run / 'status.json', {'status': 'running', 'pid': 123})
    experiment = reader_for(tmp_path, run, process_alive=lambda pid: True).run_details(
        run.name)['experiment']
    assert experiment['title'] == 'Обучение на собственной игре · CPU, 32 циклов'
    assert experiment['display_name'] == 'Обучение на собственной игре'
    assert 'PPO' in experiment['method']
    assert 'Дождитесь итоговой проверки' in experiment['recommendation']


def test_technical_classification_preserves_recorded_game_measurements(tmp_path):
    run = make_run(tmp_path / 'runs', 'diagnostic')
    metadata = json.loads((run / 'metadata.json').read_text())
    metadata['kind'] = 'frozen_imitation_diagnostics'
    metadata['updates'] = 0
    write(run / 'metadata.json', metadata)
    detail = reader_for(tmp_path, run).run_details(run.name)
    assert detail['current']['technical_only'] is True
    assert detail['quality']['available'] is True
    assert detail['experiment']['measurement']['value'] == 50
    assert '50.0%' in detail['experiment']['conclusion']


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


@pytest.mark.parametrize('state', ['failed', 'error', 'stopped', 'interrupted',
                                   'completed', 'resource_stopped'])
def test_inactive_run_without_final_check_does_not_recommend_waiting(tmp_path, state):
    run = make_run(tmp_path / 'runs', 'first')
    (run / 'evaluations/epoch_0004.json').unlink()
    write(run / 'status.json', {'status': state})
    current = reader_for(tmp_path, run).snapshot()['current']
    assert current['experiment']['verdict']['state'] == 'pending'
    assert current['experiment']['recommendation'] == (
        'Запуск не выполняется. Для оценки результата нужна '
        'отдельная проверка сохранённых весов.')
    assert current['failure_reason'] is None


def test_running_run_without_final_check_can_recommend_waiting(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    (run / 'evaluations/epoch_0004.json').unlink()
    write(run / 'status.json', {'status': 'running', 'pid': 123})
    current = reader_for(tmp_path, run, process_alive=lambda pid: True).snapshot()['current']
    assert 'Дождитесь итоговой проверки' in current['experiment']['recommendation']


@pytest.mark.parametrize('saved_error, expected', [
    ('epoch snapshot does not match its frozen evaluation',
     'RuntimeError: epoch snapshot does not match its frozen evaluation'),
    ('short message\nlarge trace follows', 'RuntimeError: short message'),
    ('x' * 1000, ('RuntimeError: ' + 'x' * 1000)[:500]),
])
def test_failure_reason_uses_bounded_saved_error(tmp_path, saved_error, expected):
    run = make_run(tmp_path / 'runs', 'first')
    write(run / 'status.json', {'status': 'failed', 'error_type': 'RuntimeError',
                               'error': saved_error})
    current = reader_for(tmp_path, run).snapshot()['current']
    assert current['failure_reason'] == expected


def test_failure_reason_uses_recorded_stop_reason_or_exit_code(tmp_path):
    run = make_run(tmp_path / 'runs', 'first')
    write(run / 'status.json', {'status': 'resource_stopped', 'stop_reason': 'memory_limit'})
    reader = reader_for(tmp_path, run)
    assert reader.snapshot()['current']['failure_reason'] == 'memory_limit'
    write(run / 'status.json', {'status': 'running'})
    write(tmp_path / 'runs/first_launch/process.json',
          {'status': 'exited', 'actual_exit_code': 2})
    assert reader.snapshot()['current']['failure_reason'] == 'Процесс завершился с кодом 2.'


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
    experiment = snapshot['experiments'][0]['experiment']
    assert '50.0%' in experiment['conclusion']
    assert experiment['saved_comment']['conclusion'] == 'Отчёт ещё не готов.'
    assert snapshot['experiments'][0]['storage']['checkpoints'][
        'checkpoints/rollout_0000']['requires_restore'] is True
    detail = reader.run_details('first')['experiment']
    assert detail['saved_comment']['limitations'] == ['Один запуск.']
    assert detail['limitations'] != ['Один запуск.']


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
    initial_path = parent / 'evaluations/epoch_0000.json'
    initial = json.loads(initial_path.read_text())
    initial['protocol_sha256'] = HASH
    write(initial_path, initial)
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
    assert experiment['picker_label'].startswith('Обучение по экранным примерам · ')
    assert all(term not in experiment['picker_label'] for term in ('MLX', 'эпох', 'побед', 'хуже'))
    original = json.loads((parent / 'evaluations/epoch_0004.json').read_text())
    original.update(created_utc='2026-10-13T13:33:49Z', protocol_sha256=HASH)
    final_path = run / 'evaluations/epoch_0004.json'
    final = json.loads(final_path.read_text())
    final['protocol_sha256'] = HASH
    write(final_path, final)
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
    assert 'без итоговой проверки' not in experiment['picker_label']


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
    assert scoped['training_current']['experiment']['title'] == scoped['training_current']['label']
    reader.process_alive = lambda pid: False
    cached = reader.snapshot(run_id=other.name)
    assert cached['training_current']['running'] is False
    assert cached['training_current']['status'] == 'interrupted'
    assert cached['training_current']['experiment'] == scoped['training_current']['experiment']


def test_picker_date_uses_user_moscow_timezone_without_technical_timezone_text(tmp_path):
    run = make_run(tmp_path / 'runs', 'trial')
    status = json.loads((run / 'status.json').read_text())
    status['started_at'] = '2026-10-13T13:39:05Z'
    write(run / 'status.json', status)
    label = reader_for(tmp_path, run).snapshot()['current']['experiment']['picker_label']
    assert label.endswith(' · 13.10 16:39')
    assert 'UTC' not in label
    status['started_at'] = '2026-10-13T22:39:05Z'
    write(run / 'status.json', status)
    label = reader_for(tmp_path, run).snapshot()['current']['experiment']['picker_label']
    assert label.endswith(' · 14.10 01:39')


def test_training_telemetry_bounds_completed_bc_measurements_and_saved_mlx_memory(tmp_path):
    run = make_run(tmp_path / 'runs', 'metrics')
    rows = [{'complete': True, 'epoch': n, 'optimizer_steps': n * 4,
             'nll': 1 / n, 'active_nll': 2 / n, 'idle_nll': None,
             'max_gradient_norm': n, 'epoch_training_seconds': 3,
             'mlx_memory_bytes': {'active': n * 1024 ** 2, 'peak': n * 2 * 1024 ** 2}}
            for n in range(1, 302)]
    rows.append({'complete': False, 'epoch': 302, 'optimizer_steps': 1206, 'nll': 99})
    (run / 'training.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    telemetry = reader_for(tmp_path, run).snapshot()['current']['telemetry']
    series = {item['key']: item for item in telemetry['series']}
    assert telemetry['rows_count'] == 301
    assert telemetry['axis_label'] == 'Шаг оптимизатора'
    assert len(series['nll']['points']) == 200
    assert series['nll']['points'][0] == {'step': 4, 'value': 1}
    assert series['nll']['points'][-1] == {'step': 1204, 'value': 1 / 301}
    assert series['nll']['latest'] == 1 / 301
    assert 'idle_nll' not in series
    resources = {item['key']: item for item in telemetry['resources']}
    assert resources['memory_active_mib']['latest'] == 301
    assert resources['memory_peak_mib']['unit'] == 'MiB'
    assert 'не общая RAM' in resources['memory_peak_mib']['description']
    assert 'memory_cache_mib' not in resources


def test_training_telemetry_ppo_uses_last_epoch_and_keeps_signed_losses():
    from carryrl.dashboard.training_metrics import training_telemetry

    rows = [{'complete': True, 'rollout': 1, 'cycle_wall_seconds': 23,
             'epochs': [{'policy_loss': 100}, {'policy_loss': -0.2, 'entropy': 0.7,
                        'approx_kl': 0.03, 'clip_fraction': 0.25, 'gradient_norm': 1.2,
                        'value_loss': float('inf')}]},
            {'complete': False, 'rollout': 2, 'epochs': [{'policy_loss': 999}]},
            {'rollout': 3, 'epochs': [{'policy_loss': 999}]}]
    result = training_telemetry(rows, ppo=True)
    series = {item['key']: item for item in result['series']}
    assert result['axis_label'] == 'Цикл PPO'
    assert result['rows_count'] == 1
    assert result['resources'] == []
    assert series['policy_loss']['points'] == [{'step': 1, 'value': -0.2}]
    assert series['policy_loss']['source'] == 'training.jsonl:epochs[-1].policy_loss'
    assert series['clip_fraction']['latest'] == 0.25
    assert series['clip_fraction']['unit'] == 'fraction'
    assert series['cycle_wall_seconds']['latest'] == 23
    assert 'value_loss' not in series


def test_training_telemetry_legacy_axis_missing_metrics_and_invalid_numbers():
    from carryrl.dashboard.training_metrics import training_telemetry

    rows = [{'complete': True, 'epoch': 1, 'nll': 0.8},
            {'complete': True, 'epoch': 2, 'optimizer_steps': 4, 'nll': 0.5},
            {'complete': True, 'epoch': 3, 'nll': True, 'objective': float('nan')},
            {'complete': True, 'epoch': False, 'nll': 2},
            {'complete': False, 'epoch': 4, 'nll': 0.1}]
    result = training_telemetry(rows, imitation=True)
    assert result['axis_label'] == 'Эпоха'
    assert result['series'][0]['points'] == [{'step': 1, 'value': 0.8}, {'step': 2, 'value': 0.5}]
    assert len(result['series']) == 1
    assert training_telemetry([], imitation=True)['series'] == []


@pytest.mark.parametrize('field', ['suite_version', 'protocol_sha256', 'scoring_version',
                                    'runtime_fingerprint', 'semantic_level_id'])
def test_changed_evaluation_conditions_exclude_bc_comparison(tmp_path, field):
    run = make_run(tmp_path / 'runs', 'conditions')
    path = run / 'evaluations/epoch_0004.json'
    final = json.loads(path.read_text())
    if field in {'runtime_fingerprint', 'semantic_level_id'}:
        final['records'][0][field] = 'changed'
    else:
        final[field] = 'f' * 64 if field == 'protocol_sha256' else 'changed'
    write(path, final)
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert [point['epoch'] for point in quality['run_history']] == [0]
    assert quality['comparison']['comparable'] is False
    assert 'различаются' in quality['comparison']['reason']
    assert quality['change_pp'] is None


@pytest.mark.parametrize('field', ['suite_version', 'scoring_version', 'runtime_fingerprint',
                                    'semantic_level_id'])
def test_ppo_declared_hash_cannot_override_actual_condition_mismatch(tmp_path, field):
    run, parent = ppo_from_parent(tmp_path)
    final = ppo_evaluation(32, 17400, sampled=False)
    if field in {'runtime_fingerprint', 'semantic_level_id'}:
        final['records'][0][field] = 'changed'
    else:
        final[field] = 'changed'
    write(run / 'evaluations/rollout_0032.json', final)
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert quality['latest_run_id'] == parent.name
    assert quality['canonical_measurement_current'] is False
    assert quality['comparison']['comparable'] is False
    assert quality['change_pp'] is None
    assert quality['diagnostic_series'][0]['coverage'] == 'full'


def test_matching_measurements_explain_comparability(tmp_path):
    run = make_run(tmp_path / 'runs', 'comparable')
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert quality['comparison']['comparable'] is True
    assert quality['change_pp'] == 50


@pytest.mark.parametrize('sampled', [True, False])
@pytest.mark.parametrize('diagnostic_revision', [17304, 17400])
def test_ppo_alternative_check_does_not_hide_later_canonical_comparison(
        tmp_path, sampled, diagnostic_revision):
    run, _ = ppo_from_parent(tmp_path)
    alternative = ppo_evaluation(1, diagnostic_revision, sampled=sampled)
    alternative['records'][0]['runtime_fingerprint'] = 'alternative'
    write(run / 'evaluations/rollout_0001.json', alternative)
    write(run / 'evaluations/rollout_0032.json', ppo_evaluation(32, 17400, sampled=False))
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert quality['comparison']['comparable'] is True
    assert quality['comparison']['excluded_measurements_count'] == 1
    assert quality['change_pp'] == 12.5
    assert len(quality['diagnostic_series']) == 1


def test_newer_incompatible_full_check_does_not_claim_current_comparison(tmp_path):
    run, _ = ppo_from_parent(tmp_path)
    write(run / 'evaluations/rollout_0001.json', ppo_evaluation(1, 17304, sampled=False))
    alternative = ppo_evaluation(32, 17400, sampled=False)
    alternative['scoring_version'] = 'different'
    write(run / 'evaluations/rollout_0032.json', alternative)
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert quality['comparison']['comparable'] is False
    assert quality['change_pp'] is None


def test_bc_later_compatible_full_check_restores_comparison(tmp_path):
    run = make_run(tmp_path / 'runs', 'conditions')
    alternative = evaluation(2, 120, 'c' * 64, wins=4)
    alternative['suite_version'] = 'different'
    write(run / 'evaluations/epoch_0002.json', alternative)
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert quality['comparison']['comparable'] is True
    assert quality['comparison']['excluded_measurements_count'] == 1
    assert quality['change_pp'] == 50


@pytest.mark.parametrize('rejected_norm', [0, 1e9])
def test_ppo_early_stop_omits_biased_norm_but_preserves_other_diagnostics(rejected_norm):
    from carryrl.dashboard.training_metrics import training_telemetry

    rows = [{'complete': True, 'rollout': 1, 'epochs': [
                {'gradient_norm': 2, 'policy_loss': -0.1,
                 'optimizer_steps': 3, 'kl_early_stop': False}]},
            {'complete': True, 'rollout': 2, 'epochs': [
                {'gradient_norm': rejected_norm, 'policy_loss': -0.2,
                 'optimizer_steps': 0, 'kl_early_stop': True}]},
            {'complete': True, 'rollout': 3, 'epochs': [{'kl_early_stop': 1}]}]
    series = {item['key']: item for item in training_telemetry(rows, ppo=True)['series']}
    assert series['gradient_norm']['points'] == [{'step': 1, 'value': 2}]
    assert series['gradient_norm']['latest_step'] == 1
    assert series['policy_loss']['latest'] == -0.2
    assert series['policy_loss']['latest_step'] == 2
    assert series['optimizer_steps']['points'] == [{'step': 1, 'value': 3}, {'step': 2, 'value': 0}]
    assert series['kl_early_stop']['points'] == [{'step': 1, 'value': 0}, {'step': 2, 'value': 1}]
    assert series['kl_early_stop']['source'] == 'training.jsonl:epochs[-1].kl_early_stop'
    assert 'исключены' in series['gradient_norm']['description']


@pytest.mark.parametrize('settings,name', [
    ({'demonstration_reuse_run': 'prior', 'canonical_replay_weight': 1},
     'Повторное обучение на прежних примерах'),
    ({'demonstration_reuse_run': 'prior', 'canonical_replay_weight': 0.25},
     'Меньше веса прежним примерам'),
    ({'styled_correction_episodes_per_task': 2}, 'Исправление ошибок на новом оформлении'),
])
def test_picker_names_describe_purpose_without_results_or_training_settings(
        tmp_path, settings, name):
    run = make_run(tmp_path / 'runs', 'human-name')
    metadata = json.loads((run / 'metadata.json').read_text())
    metadata['settings'].update(settings, training_backend='mlx', learning_rate=0.000123)
    metadata['started_at'] = '2026-10-13T13:39:05Z'
    write(run / 'metadata.json', metadata)
    experiment = reader_for(tmp_path, run).snapshot()['current']['experiment']
    assert experiment['display_name'] == name
    assert experiment['display_date'] == '13.10 16:39'
    assert experiment['picker_label'] == name + ' · 13.10 16:39'
    assert experiment['verdict']['state'] == 'improved'


def test_picker_uses_editorial_name_without_technical_suffix(tmp_path):
    run = make_run(tmp_path / 'runs', 'editorial')
    write(run / 'experiment.json', {'title': 'Проверка цветов · MLX, 40 эпох'})
    experiment = reader_for(tmp_path, run).snapshot()['current']['experiment']
    assert experiment['display_name'] == 'Проверка цветов'
    assert experiment['title'] == 'Проверка цветов · MLX, 40 эпох'


def test_newer_incompatible_full_check_blocks_old_experiment_verdict(tmp_path):
    run = make_run(tmp_path / 'runs', 'conditions')
    write(run / 'evaluations/epoch_0002.json', evaluation(2, 120, 'c' * 64, wins=4))
    path = run / 'evaluations/epoch_0004.json'
    final = json.loads(path.read_text())
    final['suite_version'] = 'different'
    write(path, final)
    experiment = reader_for(tmp_path, run).snapshot()['current']['experiment']
    assert experiment['verdict']['state'] == 'incomparable'
    assert experiment['verdict']['change_pp'] is None
    assert experiment['recommended_run_id'] is None
    assert 'п.п.' not in experiment['conclusion']
    assert 'другие условия' in experiment['conclusion']


def test_consistent_diagnostic_pair_keeps_own_experiment_comparison(tmp_path):
    run, _ = ppo_from_parent(tmp_path)
    write(run / 'status.json', {'status': 'completed', 'rollouts': 32})
    write(run / 'evaluations/rollout_0000.json', ppo_evaluation(0, 17301, 'd' * 64))
    write(run / 'evaluations/rollout_0032.json', ppo_evaluation(32, 17400))
    experiment = reader_for(tmp_path, run).snapshot()['current']['experiment']
    assert experiment['verdict']['state'] == 'unchanged'
    assert experiment['verdict']['change_pp'] == 0


def test_paired_bc_counts_swapped_wins_despite_zero_macro_change(tmp_path):
    run = make_run(tmp_path / 'runs', 'paired')
    before = evaluation(0, 124, HASH)
    before['records'][0]['outcome']['success'] = True
    after = evaluation(4, 140, 'b' * 64)
    after['records'][1]['outcome']['success'] = True
    write(run / 'evaluations/epoch_0000.json', before)
    write(run / 'evaluations/epoch_0004.json', after)
    quality = reader_for(tmp_path, run).snapshot()['quality']
    paired = quality['paired_outcomes']
    assert quality['change_pp'] == 0
    assert paired['gained'] == paired['lost'] == 1
    assert paired['net_wins'] == 0
    assert paired['before']['evaluation_file'] == 'epoch_0000.json'
    assert paired['after']['evaluation_file'] == 'epoch_0004.json'
    assert paired['before']['measurement_reused'] is False
    assert all('records' not in point for point in quality['history'])


def test_paired_bc_unknown_to_success_is_not_a_gain(tmp_path):
    run = make_run(tmp_path / 'runs', 'unknown')
    before = evaluation(0, 124, HASH)
    before['records'][0]['outcome']['known'] = False
    after = evaluation(4, 140, 'b' * 64)
    after['records'][0]['outcome']['success'] = True
    write(run / 'evaluations/epoch_0000.json', before)
    write(run / 'evaluations/epoch_0004.json', after)
    paired = reader_for(tmp_path, run).snapshot()['quality']['paired_outcomes']
    assert paired['gained'] == 0
    assert paired['unknown_pairs'] == 1
    assert paired['known_pairs'] == paired['episodes'] - 1


@pytest.mark.parametrize('damage', ['partial', 'incompatible', 'initial_only'])
def test_paired_bc_follows_existing_full_comparison_gate(tmp_path, damage):
    run = make_run(tmp_path / 'runs', 'gated')
    if damage == 'initial_only':
        (run / 'evaluations/epoch_0004.json').unlink()
    else:
        after = evaluation(4, 140, 'b' * 64)
        if damage == 'partial':
            after['level_selection'] = 'sampled levels'
        else:
            after['records'][0]['runtime_fingerprint'] = None
        write(run / 'evaluations/epoch_0004.json', after)
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert quality['paired_outcomes'] is None
    assert quality['comparison']['comparable'] is False


def test_paired_bc_uses_active_initial_and_selected_latest_full_check(tmp_path):
    first = make_run(tmp_path / 'runs', 'ancestor', revision=116)
    run = make_run(tmp_path / 'runs', 'active', revision=132, agent='d' * 64, source=first)
    write(run / 'evaluations/epoch_0000.json', evaluation(0, 116, 'b' * 64, wins=8))
    write(run / 'evaluations/epoch_0002.json', evaluation(2, 124, 'c' * 64, wins=10))
    quick = evaluation(5, 136, 'e' * 64, wins=16)
    quick['level_selection'] = 'sampled levels'
    write(run / 'evaluations/epoch_0005.json', quick)
    paired = reader_for(tmp_path, run).snapshot()['quality']['paired_outcomes']
    assert paired['before']['run_id'] == run.name
    assert paired['before']['evaluation_file'] == 'epoch_0000.json'
    assert paired['after']['evaluation_file'] == 'epoch_0004.json'
    assert paired['gained'] == paired['lost'] == 0


def test_paired_ppo_requires_own_full_initial_and_accepted_latest(tmp_path):
    run, _ = ppo_from_parent(tmp_path)
    write(run / 'evaluations/rollout_0000.json', ppo_evaluation(
        0, 17301, 'd' * 64, sampled=False))
    write(run / 'evaluations/rollout_0032.json', ppo_evaluation(32, 17400, sampled=False))
    quality = reader_for(tmp_path, run).snapshot()['quality']
    paired = quality['paired_outcomes']
    assert paired['episodes'] == paired['known_pairs'] == 256
    assert paired['gained'] == paired['lost'] == 0
    assert paired['before']['run_id'] == paired['after']['run_id'] == run.name
    assert paired['before']['evaluation_file'] == 'rollout_0000.json'
    assert paired['after']['evaluation_file'] == 'rollout_0032.json'
    assert all('records' not in point for point in quality['run_history'])
    (run / 'evaluations/rollout_0000.json').unlink()
    assert reader_for(tmp_path, run).snapshot()['quality']['paired_outcomes'] is None


def test_paired_ppo_rejects_newer_incompatible_full_measurement(tmp_path):
    run, _ = ppo_from_parent(tmp_path)
    write(run / 'evaluations/rollout_0000.json', ppo_evaluation(
        0, 17301, 'd' * 64, sampled=False))
    write(run / 'evaluations/rollout_0001.json', ppo_evaluation(1, 17304, sampled=False))
    after = ppo_evaluation(32, 17400, sampled=False)
    after['scoring_version'] = 'changed'
    write(run / 'evaluations/rollout_0032.json', after)
    quality = reader_for(tmp_path, run).snapshot()['quality']
    assert quality['comparison']['comparable'] is False
    assert quality['paired_outcomes'] is None


@pytest.mark.parametrize('copy_final', [True, False])
def test_paired_copy_provenance_does_not_claim_a_new_measurement(tmp_path, copy_final):
    source_run = make_run(tmp_path / 'runs', 'source')
    source = source_run / 'evaluations/epoch_0000.json'
    original = json.loads(source.read_text())
    original['created_utc'] = '2026-10-08T19:12:08+00:00'
    write(source, original)
    reused = {**original, 'initial_evaluation_reuse': {
        'measurement_reused': True, 'source': str(source),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_created_utc': original['created_utc'], 'source_epoch': 0,
        'source_weights_revision': original['weights_revision']}}
    target = make_run(tmp_path / 'runs', 'target')
    write(target / 'evaluations/epoch_0000.json', reused)
    if copy_final:
        write(target / 'evaluations/epoch_0004.json', {**reused, 'epoch': 4})
    quality = reader_for(tmp_path, target).snapshot()['quality']
    if copy_final:
        assert quality['paired_outcomes'] is None
        assert 'то же исходное измерение' in quality['paired_outcomes_reason']
        assert quality['measurement_reused'] is True
    else:
        assert quality['paired_outcomes']['before']['measurement_reused'] is True
        assert quality['paired_outcomes']['before']['measurement_run_id'] == source_run.name
        assert quality['paired_outcomes']['after']['measurement_reused'] is False


def test_pixel_validation_separate_same_weight_comparison_and_refresh(tmp_path):
    run, _ = ppo_from_parent(tmp_path)
    canonical = ppo_evaluation(1, 17304, sampled=False)
    write(run / 'evaluations/rollout_0001.json', canonical)
    reader = reader_for(tmp_path, run)
    assert reader.snapshot()['quality']['pixel_validation'] is None
    pixel = json.loads(json.dumps(canonical))
    pixel.update(evaluation_pixel_art=True, evaluation_appearance='pixel_art_heldout',
                 evaluation_style_split='validation', protocol_sha256='e' * 64)
    for record in pixel['records']:
        record['presentation'] = {'pixel_art': {'style_split': 'validation'}}
        record['initial_frame_sha256'] = 'f' * 64
        record['outcome']['success'] = False
    write(run / 'evaluations_pixel/rollout_0001.json', pixel)
    quality = reader.snapshot()['quality']
    result = quality['pixel_validation']
    assert result['value'] == 0
    assert result['episodes'] == 256
    assert result['canonical_same_weights'] is True
    assert len(result['games']) == 16
    assert sum(game['canonical']['value'] for game in result['games']) / 16 == 87.5
    assert quality['value'] != result['value']
    pixel['agent_sha256'] = 'c' * 64
    write(run / 'evaluations_pixel/rollout_0001.json', pixel)
    result = reader.snapshot()['quality']['pixel_validation']
    assert result['canonical_same_weights'] is False
    assert all(game['canonical'] is None for game in result['games'])


@pytest.mark.parametrize('field,value', [('protocol_sha256', 'broken'),
                                        ('evaluation_style_split', 'train'),
                                        ('evaluation_pixel_art', False)])
def test_pixel_validation_rejects_unconfirmed_identity(tmp_path, field, value):
    run, _ = ppo_from_parent(tmp_path)
    pixel = ppo_evaluation(1, 17304)
    pixel.update(evaluation_pixel_art=True, evaluation_appearance='pixel_art_heldout',
                 evaluation_style_split='validation')
    pixel[field] = value
    write(run / 'evaluations_pixel/rollout_0001.json', pixel)
    result = reader_for(tmp_path, run).snapshot()
    assert result['quality']['pixel_validation'] is None
    assert result['warnings']


def test_ppo_latest_completed_play_and_external_resources(tmp_path):
    run, _ = ppo_from_parent(tmp_path)
    (run / 'training.jsonl').write_text(json.dumps({
        'complete': True, 'rollout': 1, 'episode_records': [
            {'task': 'navigation', 'outcome': {'known': True, 'success': True},
             'reward_sum': 2.5},
            {'task': 'navigation', 'outcome': {'known': False}, 'reward_sum': -0.5}]
    }) + '\n' + json.dumps({'complete': False, 'rollout': 2}) + '\n')
    launch = run.parent / (run.name + '_launch')
    write(launch / 'process.json', {
        'status': 'running', 'command_pid': 123, 'maximum_rss_bytes': 12 * 1024**3,
        'peak_rss_bytes': 5 * 1024**3,
        'resource_sample': {'process_group_rss_bytes': 4 * 1024**3,
                            'disk_free_bytes': 20 * 1024**3, 'utc': '2026-10-19T10:00:00Z'}})
    status = json.loads((run / 'status.json').read_text())
    status['phase'] = 'pixel_evaluation'
    write(run / 'status.json', status)
    result = reader_for(tmp_path, run).snapshot()['current']
    assert result['method'] == 'ppo'
    assert result['phase_label'] == 'Проверка пиксельного оформления'
    assert result['resource_guard']['peak_rss_bytes'] == 5 * 1024**3
    assert result['training_outcomes'] == {'rollout': 1, 'games': [{
        'task': 'navigation', 'label': result['training_outcomes']['games'][0]['label'],
        'episodes': 2, 'known': 1, 'successes': 1, 'mean_reward': 1.0}]}


def comparison_reader(tmp_path, *, manifest=None):
    root = tmp_path / 'runs'
    make_run(root, 'old_bc')
    plan = root / 'fresh_comparison'
    write(plan / 'manifest.json', manifest or {
        'format': 'carryrl.token_comparison.v1', 'training_executed': True,
        'policy_updates': 999, 'modelinit_verified': True, 'ready_to_train': True,
        'budget': {'rollouts_per_arm': 128, 'environment_episodes_per_arm': 2048},
        'arms': [{'spatial_tokens': tokens, 'seed': 1,
                  'initial_checkpoint': str(plan / f'initial/t{tokens}/checkpoints/rollout_0000'),
                  'training_output': str(root / f'fresh_t{tokens}'),
                  'training_executed': True} for tokens in (16, 32, 64)]})
    registry = tmp_path / 'registry.json'
    write(registry, {'comparison_plan': str(plan)})
    return OverviewReader(root, registry), plan


def test_fresh_comparison_default_is_preparation_without_bc_metrics(tmp_path):
    reader, _ = comparison_reader(tmp_path)
    data = reader.snapshot()
    assert data['current'] is None
    assert data['training_current'] is None
    assert data['selected_run_id'] is None
    assert data['quality']['available'] is False
    assert data['playbacks']['items'] == []
    assert not any('Активный запуск не найден' in warning for warning in data['warnings'])
    plan = data['comparison_plan']
    assert plan['state'] == 'preparation'
    assert plan['policy_updates'] == 0
    assert plan['ready_to_train'] is False
    assert plan['initialization'] == 'pending'
    assert plan['reward'] == 'pending'
    assert [arm['spatial_tokens'] for arm in plan['arms']] == [16, 32, 64]
    archived = reader.snapshot(run_id='old_bc')
    assert archived['current']['run_id'] == 'old_bc'
    assert archived['quality']['available'] is True
    assert archived['training_current'] is None


def test_comparison_snapshot_tracks_actual_reports_and_training_outputs(tmp_path):
    reader, plan = comparison_reader(tmp_path)
    assert reader.snapshot()['comparison_plan']['reward'] == 'pending'
    write(plan / 'reward/reward_report.json', {'gate': {'passed': False}})
    assert reader.snapshot()['comparison_plan']['reward'] == 'failed'
    write(plan / 'reward/validation_report.json', {'passed': False})
    assert reader.snapshot()['comparison_plan']['reward_validation'] == 'failed'
    write(plan / 'reward/reward_report.json', {'gate': {'passed': True}})
    (plan / 'reward/reward_checkpoint.pt').write_bytes(b'checkpoint')
    assert reader.snapshot()['comparison_plan']['reward'] == 'published'
    write(plan / 'initial/t16/checkpoints/rollout_0000/manifest.json', {'rollout': 0})
    assert reader.snapshot()['comparison_plan']['arms'][0]['initialization'] == 'pending'
    (plan / 'initial/t16/checkpoints/rollout_0000/state.pt').write_bytes(b'weights')
    assert reader.snapshot()['comparison_plan']['arms'][0]['initialization'] == 'published'
    run = tmp_path / 'runs/fresh_t16'
    write(run / 'status.json', {'status': 'training', 'optimizer_steps_this_process': 2})
    fresh = reader.snapshot()['comparison_plan']
    assert fresh['policy_updates'] == 2
    assert fresh['arms'][0]['run_id'] == 'fresh_t16'
    write(run / 'status.json', {'status': 'completed', 'optimizer_steps_this_process': 3})
    assert reader.snapshot()['comparison_plan']['policy_updates'] == 3
    write(plan / 'manifest.json', {'format': 'invalid'})
    assert reader.snapshot()['comparison_plan']['state'] == 'unavailable'
    assert reader.snapshot()['current'] is None


def test_missing_comparison_manifest_is_an_error_not_interrupted_training(tmp_path):
    reader, plan = comparison_reader(tmp_path)
    (plan / 'manifest.json').unlink()
    data = reader.snapshot()
    assert data['comparison_plan']['state'] == 'unavailable'
    assert data['current'] is None
    assert data['training_current'] is None
    assert data['warnings']


def test_comparison_registry_can_select_a_real_training_run(tmp_path):
    reader, plan = comparison_reader(tmp_path)
    actual = make_run(tmp_path / 'runs', 'fresh_t16')
    write(reader.registry, {'comparison_plan': str(plan), 'run': str(actual)})
    data = reader.snapshot()
    assert data['current']['run_id'] == 'fresh_t16'
    assert data['training_current']['run_id'] == 'fresh_t16'
    assert data['comparison_plan']['plan_id'] == plan.name


def test_comparison_malformed_optional_artifacts_do_not_invent_readiness(tmp_path):
    reader, plan = comparison_reader(tmp_path)
    write(plan / 'reward/reward_report.json', {'gate': []})
    write(plan / 'reward/validation_report.json', {'passed': 'true'})
    (plan / 'reward/reward_checkpoint.pt').write_bytes(b'checkpoint')
    data = reader.snapshot()['comparison_plan']
    assert data['reward_validation'] == 'pending'
    assert data['ready_to_train'] is False
    manifest = json.loads((plan / 'manifest.json').read_text())
    manifest['arms'] = None
    write(plan / 'manifest.json', manifest)
    assert reader.snapshot()['comparison_plan']['arms'] == []


def test_comparison_preparation_stage_tracks_files_and_checks_actual_pid(tmp_path):
    reader, plan = comparison_reader(tmp_path)
    live = {451: True, 452: True}
    reader.process_alive = live.get
    assert reader.snapshot()['comparison_plan']['preparation_stage'] is None
    fit = plan / 'stages/reward_fit/process.json'
    write(fit, {'status': 'running', 'command_pid': 451,
                'resource_sample': {'utc': '2026-10-19T10:00:00Z',
                                    'process_group_rss_bytes': 1024,
                                    'disk_free_bytes': 4096},
                'peak_rss_bytes': 2048, 'minimum_free_bytes': 4096})
    stage = reader.snapshot()['comparison_plan']['preparation_stage']
    assert stage['kind'] == 'reward_fit'
    assert stage['running'] is True
    assert stage['resource_sample']['process_group_rss_bytes'] == 1024
    assert stage['peak_rss_bytes'] == 2048
    # A cached snapshot must still recheck liveness without a file modification.
    live[451] = False
    stage = reader.snapshot()['comparison_plan']['preparation_stage']
    assert stage['running'] is False
    assert stage['status'] == 'interrupted'
    validation = plan / 'stages/reward_validation/process.json'
    write(validation, {'status': 'running', 'command_pid': 452})
    stage = reader.snapshot()['comparison_plan']['preparation_stage']
    assert stage['kind'] == 'reward_validation'
    assert stage['running'] is True
    write(validation, {'status': 'exited', 'command_pid': 452, 'actual_exit_code': 3})
    stage = reader.snapshot()['comparison_plan']['preparation_stage']
    assert stage['running'] is False
    assert stage['status'] == 'failed'
    assert stage['exitcode'] == 3
    write(plan / 'stages/model_initialization/process.json',
          {'status': 'exited', 'actual_exit_code': 0})
    stage = reader.snapshot()['comparison_plan']['preparation_stage']
    assert stage['kind'] == 'model_initialization'
    assert stage['status'] == 'completed'
    assert reader.snapshot()['training_current'] is None


def test_comparison_preparation_stage_requires_valid_pid_and_prioritizes_running(tmp_path):
    reader, plan = comparison_reader(tmp_path)
    reader.process_alive = lambda pid: pid == 451
    fit = plan / 'stages/reward_fit/process.json'
    write(fit, {'status': 'running', 'command_pid': True})
    stage = reader.snapshot()['comparison_plan']['preparation_stage']
    assert stage['running'] is False
    assert stage['status'] == 'unknown'
    write(fit, {'status': 'running', 'command_pid': 451})
    write(plan / 'stages/reward_validation/process.json',
          {'status': 'exited', 'actual_exit_code': 0})
    assert reader.snapshot()['comparison_plan']['preparation_stage']['kind'] == 'reward_fit'


def fresh_ppo_run(tmp_path):
    run, _ = ppo_from_parent(tmp_path)
    metadata = json.loads((run / 'metadata.json').read_text())
    metadata.pop('warm_start_parent')
    write(run / 'metadata.json', metadata)
    checkpoint = run / 'checkpoints/rollout_0000'
    write(checkpoint / 'manifest.json', {
        'rollout': 0, 'weights_revision': 0, 'sha256': HASH,
        'agent_sha256': 'b' * 64, 'reward_sha256': REWARD})
    write(run / 'latest.json', {'checkpoint': str(checkpoint)})
    write(run / 'evaluations/rollout_0000.json', ppo_evaluation(0, 0, sampled=False))
    return run


def test_fresh_ppo_cp0_establishes_own_verified_deterministic_baseline(tmp_path):
    run = fresh_ppo_run(tmp_path)
    initial = json.loads((run / 'evaluations/rollout_0000.json').read_text())
    initial['records'] += [{**record, 'policy': 'frozen_stochastic',
                            'outcome': {'known': True, 'success': True}}
                           for record in initial['records']]
    write(run / 'evaluations/rollout_0000.json', initial)
    reader = reader_for(tmp_path, run)
    quality = reader.snapshot(include_experiments=False)['quality']
    assert quality['available'] is True
    assert quality['measurement_stage'] == 'initial'
    assert quality['episodes'] == 256
    assert quality['successes'] == 224
    assert quality['value'] == 87.5
    assert quality['diagnostic_series'] == []
    assert quality['history'][0]['updates'] == 0
    write(run / 'evaluations/rollout_0001.json', ppo_evaluation(
        1, 4, agent='f' * 64, sampled=False))
    quality = reader.snapshot(include_experiments=False)['quality']
    assert quality['history'][-1]['updates'] == 4
    assert quality['comparison']['comparable'] is True


@pytest.mark.parametrize('field', ['agent_sha256', 'reward_sha256', 'weights_revision', 'sha256'])
def test_fresh_ppo_cp0_rejects_mismatched_checkpoint_provenance(tmp_path, field):
    run = fresh_ppo_run(tmp_path)
    path = run / 'checkpoints/rollout_0000/manifest.json'
    manifest = json.loads(path.read_text())
    manifest[field] = 9 if field == 'weights_revision' else 'invalid'
    write(path, manifest)
    snapshot = reader_for(tmp_path, run).snapshot(include_experiments=False)
    assert snapshot['quality']['available'] is False
    assert any('checkpoint' in warning for warning in snapshot['warnings'])


def fresh_launch(tmp_path, run):
    plan = tmp_path / 'runs/fresh_plan'
    launch = plan / 'stages/main'
    write(plan / 'manifest.json', {'format': 'carryrl.fresh_rl.v1',
                                  'budget_seconds': 14400, 'reserve_seconds': 1800})
    write(plan / 'supervisor.json', {'status': 'running', 'started_utc': 1000,
                                    'budget_seconds': 14400})
    write(launch / 'status.json', {'format': 'carryrl.fresh_rl.v1', 'status': 'running',
                                   'phase': 'reward_fitting', 'updated_utc': 1050,
                                   'plan': str(plan), 'policy_output': str(run)})
    write(launch / 'process.json', {
        'status': 'running', 'command_pid': 123, 'memory_throttled': True,
        'pause_reason': 'available memory below threshold', 'wait_duration_seconds': 500,
        'active_wall_seconds': 100, 'resource_sample': {'utc': 1590}})
    registry = tmp_path / 'registry.json'
    write(registry, {'run': str(run), 'launch': str(launch)})
    return registry, launch


def test_fresh_launch_preparation_before_policy_directory_with_safe_cached_transition(tmp_path):
    run = tmp_path / 'runs/fresh_main'
    registry, launch = fresh_launch(tmp_path, run)
    reader = OverviewReader(tmp_path / 'runs', registry, clock=lambda: 1600,
                            process_alive=lambda pid: pid == 123)
    for _ in range(2):
        snapshot = reader.snapshot(include_experiments=False)
        current = snapshot['training_current']
        assert current == snapshot['current']
        assert current['preparing_policy'] is True
        assert current['running'] is True
        assert current['status_label'] == 'Пауза по памяти'
        assert current['phase_label'] == 'Подготовка модели награды'
        assert current['stale'] is True  # Independent from guard telemetry freshness.
        assert current['resource_guard']['sample_age_seconds'] == 10
        assert current['resource_guard']['sample_stale'] is False
        assert current['budget']['elapsed_seconds'] == 600  # Includes the 500s pause.
        assert current['budget']['remaining_seconds'] == 13800
        assert current['budget']['reserve_seconds'] == 1800
        assert current['budget']['hard_deadline_at'] == '1970-01-01T04:16:40+00:00'
        assert snapshot['warnings'] == []
    write(run / 'metadata.json', {'kind': 'sequence_training',
                                 'config': {'total_rollouts': 128}})
    write(run / 'status.json', {'status': 'running', 'phase': 'initial_evaluation',
                               'updated_at': 1599, 'evaluation_task': 'memory',
                               'evaluation_policy': 'frozen_stochastic', 'evaluation_step': 9,
                               'total_decisions_upper_bound': 327680,
                               'latest_checkpoint': str(run / 'checkpoints/rollout_0000'),
                               'latest_checkpoint_weights_revision': 0,
                               'current_policy_saved': True})
    current = reader.snapshot(include_experiments=False)['training_current']
    assert current['preparing_policy'] is False
    assert current['evaluation_task'] == 'memory'
    assert current['evaluation_policy'] == 'frozen_stochastic'
    assert current['evaluation_step'] == 9
    assert current['latest_checkpoint_weights_revision'] == 0
    assert current['current_policy_saved'] is True
    assert current['total_decisions_upper_bound'] == 327680
    assert current['eta_seconds'] is None
    process = json.loads((launch / 'process.json').read_text())
    process['status'] = 'exited'
    process['actual_exit_code'] = 2
    write(launch / 'process.json', process)
    current = reader.snapshot(include_experiments=False)['training_current']
    assert current['running'] is False
    assert current['status'] == 'failed'


@pytest.mark.parametrize('damage', ['outside', 'symlink', 'wrong_policy'])
def test_preparation_fallback_rejects_untrusted_launch_paths(tmp_path, damage):
    run = tmp_path / 'runs/fresh_main'
    registry, launch = fresh_launch(tmp_path, run)
    if damage == 'outside':
        write(registry, {'run': str(run), 'launch': str(tmp_path / 'outside')})
    elif damage == 'symlink':
        alias = tmp_path / 'runs/alias'
        alias.symlink_to(launch, target_is_directory=True)
        write(registry, {'run': str(run), 'launch': str(alias)})
    else:
        status = json.loads((launch / 'status.json').read_text())
        status['policy_output'] = str(tmp_path / 'outside')
        write(launch / 'status.json', status)
    snapshot = OverviewReader(tmp_path / 'runs', registry).snapshot(include_experiments=False)
    assert snapshot['training_current'] is None
    assert snapshot['current'] is None


def test_stale_resource_sample_does_not_claim_current_memory_pause(tmp_path):
    run = tmp_path / 'runs/fresh_main'
    registry, _ = fresh_launch(tmp_path, run)
    reader = OverviewReader(tmp_path / 'runs', registry, clock=lambda: 2000,
                            process_alive=lambda pid: True)
    current = reader.snapshot(include_experiments=False)['current']
    assert current['resource_guard']['sample_stale'] is True
    assert current['resource_guard']['sample_age_seconds'] == 410
    assert current['status_label'] == 'Работает'


def test_training_log_larger_than_json_limit_keeps_compact_recent_metrics(tmp_path):
    run = tmp_path / 'runs/large_log'
    run.mkdir(parents=True)
    with (run / 'training.jsonl').open('w') as stream:
        for rollout in range(18):
            stream.write(json.dumps({
                'rollout': rollout, 'complete': True, 'cycle_wall_seconds': 2,
                'epochs': [{'policy_loss': -0.25, 'gradient_norm': 1.0}],
                'episode_records': [{'task': 'memory', 'reward_sum': 1,
                                     'outcome': {'known': True, 'success': True},
                                     'steps': 'x' * 1_000_000}]}) + '\n')
        stream.write('{"rollout":')  # Concurrent writer has not published a whole row.
    reader = reader_for(tmp_path, run)
    rows = reader._training_rows(run)
    assert len(rows) == 18
    assert rows[-1]['rollout'] == 17
    assert rows[-1]['epochs'][0]['policy_loss'] == -0.25
    assert 'steps' not in rows[-1]['episode_records'][0]
    assert any('неполная строка' in warning for warning in reader.warnings)


def test_training_log_tail_skips_partial_and_overlong_lines(tmp_path, monkeypatch):
    from carryrl.dashboard import overview_data
    monkeypatch.setattr(overview_data, 'MAX_TAIL_BYTES', 300)
    monkeypatch.setattr(overview_data, 'MAX_LINE_BYTES', 100)
    run = tmp_path / 'runs/tail'
    run.mkdir(parents=True)
    (run / 'training.jsonl').write_text('x' * 1000 + '\n' +
                                       json.dumps({'rollout': 8, 'complete': True}) + '\n')
    reader = reader_for(tmp_path, run)
    assert reader._training_rows(run) == [{'rollout': 8, 'complete': True}]
    assert any('ограниченный хвост' in warning for warning in reader.warnings)


def test_sequence_playback_requires_matching_published_episode_not_only_checkpoint(tmp_path):
    run = fresh_ppo_run(tmp_path)
    reader = reader_for(tmp_path, run)
    report = json.loads((run / 'evaluations/rollout_0000.json').read_text())
    record = report['records'][0]
    metadata = {**record, 'epoch': 0, 'weights_revision': 0, 'agent_sha256': 'b' * 64,
                'action_count': record['steps'],
                'evaluation_reference': str(run / 'evaluations/rollout_0000.json')}
    assert reader._sequence_playback_evidence(run, metadata) is True
    metadata['level_index'] = 15
    assert reader._sequence_playback_evidence(run, metadata) is False
    metadata['level_index'] = record['level_index']
    metadata['evaluation_pixel_art'] = True
    metadata['evaluation_reference'] = str(run / 'evaluations_pixel/rollout_0000.json')
    assert reader._sequence_playback_evidence(run, metadata) is None
    report['evaluation_pixel_art'] = True
    write(run / 'evaluations_pixel/rollout_0000.json', report)
    assert reader._sequence_playback_evidence(run, metadata) is True
    report['reward_sha256'] = 'f' * 64
    write(run / 'evaluations_pixel/rollout_0000.json', report)
    assert reader._sequence_playback_evidence(run, metadata) is False


@pytest.mark.parametrize('explicit_rollout', [False, True])
def test_ppo_playback_legacy_epoch_is_exposed_only_as_rollout(tmp_path, explicit_rollout):
    run = fresh_ppo_run(tmp_path)
    add_playback(run)
    report = json.loads((run / 'evaluations/rollout_0000.json').read_text())
    report['rollout'] = 64
    write(run / 'evaluations/rollout_0064.json', report)
    checkpoint = run / 'checkpoints/rollout_0064'
    manifest = json.loads((run / 'checkpoints/rollout_0000/manifest.json').read_text())
    manifest['rollout'] = 64
    write(checkpoint / 'manifest.json', manifest)
    path = run / 'playbacks/actual.json'
    metadata = json.loads(path.read_text())
    metadata.update(report['records'][0])
    metadata.update(epoch=64, checkpoint=str(checkpoint),
                    action_count=report['records'][0]['steps'],
                    evaluation_reference=str(run / 'evaluations/rollout_0064.json'))
    if explicit_rollout:
        metadata['rollout'] = 64
    write(path, metadata)
    reader = reader_for(tmp_path, run)
    item = reader._playbacks([run], {})['items'][0]
    assert item['epoch'] is None
    assert item['rollout'] == 64
    assert 'цикл 64' in item['source_label']
    assert 'эпоха' not in item['source_label']


def test_fresh_baseline_rejects_missing_level_even_when_total_count_is_full(tmp_path):
    run = fresh_ppo_run(tmp_path)
    report = json.loads((run / 'evaluations/rollout_0000.json').read_text())
    report['records'][0]['level_index'] = 1
    write(run / 'evaluations/rollout_0000.json', report)
    snapshot = reader_for(tmp_path, run).snapshot(include_experiments=False)
    assert snapshot['quality']['available'] is False
    assert any('16 × 16' in warning for warning in snapshot['warnings'])


def test_automatic_first_level_playback_exposes_validation_selector_contract(tmp_path):
    run = fresh_ppo_run(tmp_path)
    add_playback(run)
    report = json.loads((run / 'evaluations/rollout_0000.json').read_text())
    record = report['records'][0]
    sidecar = run / 'playbacks/actual.json'
    metadata = json.loads(sidecar.read_text())
    metadata.update(record, epoch=0, action_count=record['steps'],
                    evaluation_reference=str(run / 'evaluations/rollout_0000.json'))
    write(sidecar, metadata)
    items = reader_for(tmp_path, run)._playbacks([run], {})['items']
    matches = [item for item in items if item['mode'] == 'greedy'
               and item['checkpoint_id'] == 'rollout_0000'
               and item['appearance'] == 'canonical'
               and item['level_index'] == 0 and item['task'] == TASKS[0]]
    assert len(matches) == 1
    assert matches[0]['policy'] == 'frozen_deterministic'


def test_training_log_append_reads_and_parses_only_new_rows(tmp_path, monkeypatch):
    from carryrl.dashboard import overview_data

    run = tmp_path / 'runs/incremental'
    run.mkdir(parents=True)
    path = run / 'training.jsonl'

    def row(index):
        return json.dumps({'rollout': index, 'complete': True, 'padding': 'x' * 32768}) + '\n'

    path.write_text(''.join(row(index) for index in range(200)))
    reader = reader_for(tmp_path, run)
    assert len(reader._training_rows(run)) == 200
    original_fdopen = overview_data.os.fdopen
    original_loads = overview_data.json.loads
    read_bytes = 0
    parses = 0

    class Tracked:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, *args):
            nonlocal read_bytes
            value = self.stream.read(*args)
            read_bytes += len(value)
            return value

        def readline(self, *args):
            nonlocal read_bytes
            value = self.stream.readline(*args)
            read_bytes += len(value)
            return value

    def counted_loads(*args, **kwargs):
        nonlocal parses
        parses += 1
        return original_loads(*args, **kwargs)

    monkeypatch.setattr(overview_data.os, 'fdopen', lambda *args: Tracked(original_fdopen(*args)))
    monkeypatch.setattr(overview_data.json, 'loads', counted_loads)
    appended_bytes = 0
    for index in range(200, 205):
        content = row(index)
        appended_bytes += len(content.encode())
        with path.open('a') as stream:
            stream.write(content)
        rows = reader._training_rows(run)
        assert len(rows) == index + 1
        assert rows[-1]['rollout'] == index
    assert parses == 5
    assert read_bytes <= appended_bytes + 5 * 4 * overview_data.TRAINING_GUARD_BYTES
    measured = read_bytes, parses
    # Heavy evaluation/media churn must not evict compact historical logs.
    for index in range(600):
        reader._cache(('json', index), {'value': index})
    assert reader._training_rows(run) is rows
    assert (read_bytes, parses) == measured
    assert parses == 5
    assert read_bytes <= appended_bytes + 5 * 4 * overview_data.TRAINING_GUARD_BYTES


def test_training_log_partial_appends_do_not_duplicate_or_lose_rows(tmp_path):
    run = tmp_path / 'runs/partial_append'
    run.mkdir(parents=True)
    path = run / 'training.jsonl'
    reader = reader_for(tmp_path, run)
    path.write_text('{"rollout": 1}\n{"rollout": 2')
    assert [row['rollout'] for row in reader._training_rows(run)] == [1]
    with path.open('a') as stream:
        stream.write('}')
    assert [row['rollout'] for row in reader._training_rows(run)] == [1, 2]
    with path.open('a') as stream:
        stream.write('\n{"rollout": 3}\nmalformed\n{"rollout":')
    assert [row['rollout'] for row in reader._training_rows(run)] == [1, 2, 3]
    with path.open('a') as stream:
        stream.write('4}\n')
    assert [row['rollout'] for row in reader._training_rows(run)] == [1, 2, 3, 4]


@pytest.mark.parametrize('mutation', ['replacement', 'truncate', 'rewrite', 'rewrite_grow'])
def test_training_log_resets_on_replacement_truncate_and_rewrite(tmp_path, mutation):
    run = tmp_path / 'runs/reset'
    run.mkdir(parents=True)
    path = run / 'training.jsonl'
    reader = reader_for(tmp_path, run)
    path.write_text('{"rollout": 1}\n{"rollout": 2}\n')
    assert [row['rollout'] for row in reader._training_rows(run)] == [1, 2]
    before = path.stat()
    if mutation == 'replacement':
        replacement = path.with_suffix('.new')
        replacement.write_text('{"rollout": 3}\n{"rollout": 4}\n')
        os.replace(replacement, path)
    elif mutation == 'truncate':
        path.write_text('{"rollout": 3}\n')
    elif mutation == 'rewrite':
        path.write_text('{"rollout": 3}\n{"rollout": 4}\n')
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    else:
        path.write_text('{"rollout": 3}\n{"rollout": 4}\n{"rollout": 5}\n')
    expected = {'truncate': [3], 'rewrite_grow': [3, 4, 5]}.get(mutation, [3, 4])
    assert [row['rollout'] for row in reader._training_rows(run)] == expected


def test_training_log_overlong_partial_recovers_after_later_newline(tmp_path, monkeypatch):
    from carryrl.dashboard import overview_data

    monkeypatch.setattr(overview_data, 'MAX_LINE_BYTES', 64)
    run = tmp_path / 'runs/overlong_append'
    run.mkdir(parents=True)
    path = run / 'training.jsonl'
    reader = reader_for(tmp_path, run)
    path.write_text('{"rollout": 1}\n' + 'x' * 200)
    assert reader._training_rows(run) == [{'rollout': 1}]
    with path.open('a') as stream:
        stream.write('x' * 200)
    assert reader._training_rows(run) == [{'rollout': 1}]
    with path.open('a') as stream:
        stream.write('\n{"rollout": 2}\n')
    assert reader._training_rows(run) == [{'rollout': 1}, {'rollout': 2}]


def test_training_log_append_keeps_byte_window_and_row_cache_bounded(tmp_path, monkeypatch):
    from carryrl.dashboard import overview_data

    monkeypatch.setattr(overview_data, 'MAX_TAIL_BYTES', 100)
    monkeypatch.setattr(overview_data, 'MAX_TRAINING_CACHE_BYTES', 10000)
    run = tmp_path / 'runs/tail_append'
    run.mkdir(parents=True)
    path = run / 'training.jsonl'
    reader = reader_for(tmp_path, run)
    path.write_text(''.join(json.dumps({'rollout': i}) + '\n' for i in range(6)))
    reader._training_rows(run)
    for i in range(6, 15):
        with path.open('a') as stream:
            stream.write(json.dumps({'rollout': i}) + '\n')
        rows = reader._training_rows(run)
        fresh = reader_for(tmp_path, run)._training_rows(run)
        assert rows == fresh
        assert reader.training_cache_bytes <= overview_data.MAX_TRAINING_CACHE_BYTES
    for i in range(12):
        other = run.with_name(f'tail_other_{i}')
        other.mkdir()
        (other / 'training.jsonl').write_bytes(path.read_bytes())
        reader._training_rows(other)
        assert reader.training_cache_bytes <= overview_data.MAX_TRAINING_CACHE_BYTES
    assert len(reader.training_cache) < 12


def test_ppo_playbacks_validate_shared_report_once_and_keep_episode_checks(tmp_path, monkeypatch):
    run = fresh_ppo_run(tmp_path)
    add_playback(run)
    report_path = run / 'evaluations/rollout_0000.json'
    report = json.loads(report_path.read_text())
    sidecar = run / 'playbacks/actual.json'
    metadata = json.loads(sidecar.read_text())
    metadata.update(report['records'][0], epoch=0, evaluation_reference=str(report_path))
    write(sidecar, metadata)
    for index in range(127):
        write(sidecar.with_name(f'copy_{index:03d}.json'), metadata)
    reader = reader_for(tmp_path, run)
    reader.dependencies = {}
    original = reader._records
    validations = 0

    def counted(evaluation):
        nonlocal validations
        validations += 1
        return original(evaluation)

    monkeypatch.setattr(reader, '_records', counted)
    assert len(reader._playbacks([run], {})['items']) == 128
    assert validations == 1
    reader.dependencies = {}
    assert len(reader._playbacks([run], {})['items']) == 128
    assert validations == 1
    assert report_path in reader.dependencies
    # Cached report validation must not certify a sidecar with another episode.
    metadata['level_index'] = 999
    write(sidecar, metadata)
    assert len(reader._playbacks([run], {})['items']) == 127
    assert validations == 1
    # Invalid reports are also validated only once; each caller still rejects them.
    report['split'] = 'test'
    write(report_path, report)
    assert reader._playbacks([run], {})['items'] == []
    assert validations == 2
    assert reader._playbacks([run], {})['items'] == []
    assert validations == 2


@pytest.mark.parametrize('mutation', ['rewrite', 'replacement'])
def test_ppo_snapshot_rechecks_report_with_same_size_and_restored_mtime(tmp_path, mutation):
    run = fresh_ppo_run(tmp_path)
    add_playback(run)
    report_path = run / 'evaluations/rollout_0000.json'
    report = json.loads(report_path.read_text())
    sidecar = run / 'playbacks/actual.json'
    metadata = json.loads(sidecar.read_text())
    metadata.update(report['records'][0], epoch=0, evaluation_reference=str(report_path))
    write(sidecar, metadata)
    reader = reader_for(tmp_path, run)
    assert len(reader.snapshot(include_experiments=False)['playbacks']['items']) == 1
    before = report_path.stat()
    damaged = report_path.read_text().replace('"split": "validation"', '"split": "testxxxxxx"')
    assert len(damaged.encode()) == before.st_size
    if mutation == 'rewrite':
        report_path.write_text(damaged)
    else:
        replacement = report_path.with_suffix('.new')
        replacement.write_text(damaged)
        os.replace(replacement, report_path)
    os.utime(report_path, ns=(before.st_atime_ns, before.st_mtime_ns))
    result = reader.snapshot(include_experiments=False)
    assert result['playbacks']['items'] == []
    assert any('matching PPO evaluation evidence' in warning for warning in result['warnings'])


def test_validated_record_cache_is_bounded(tmp_path, monkeypatch):
    from carryrl.dashboard import overview_data

    run = fresh_ppo_run(tmp_path)
    report_path = run / 'evaluations/rollout_0000.json'
    report = json.loads(report_path.read_text())
    monkeypatch.setattr(overview_data, 'MAX_RECORD_CACHE_BYTES', report_path.stat().st_size + 4096)
    reader = reader_for(tmp_path, run)
    for index in range(3):
        path = report_path.with_name(f'copy_{index:04d}.json')
        write(path, report)
        assert len(reader._validated_records(path, reader._read(path))[0]) == 256
        assert reader.record_cache_bytes <= overview_data.MAX_RECORD_CACHE_BYTES
    assert len(reader.record_cache) == 1


@pytest.mark.parametrize('damaged_complete_line', [False, True])
def test_training_partial_warning_clears_after_completion(tmp_path, damaged_complete_line):
    run = tmp_path / 'runs/warning_append'
    run.mkdir(parents=True)
    path = run / 'training.jsonl'
    path.write_text(('malformed\n' if damaged_complete_line else '') + '{"rollout": 1')
    reader = reader_for(tmp_path, run)
    assert reader._training_rows(run) == []
    assert any('неполная строка' in warning for warning in reader.warnings)
    reader.warnings = []
    reader._training_rows(run)
    assert any('неполная строка' in warning for warning in reader.warnings)
    with path.open('a') as stream:
        stream.write('}\n')
    reader.warnings = []
    assert reader._training_rows(run) == [{'rollout': 1}]
    assert any('неполная строка' in warning for warning in reader.warnings) is damaged_complete_line


def test_ppo_compact_diagnostics_preserve_measured_actions_and_not_raw_steps(tmp_path):
    from carryrl.dashboard.overview_data import _compact
    from carryrl.dashboard.training_metrics import training_telemetry

    row = _compact({
        'complete': True, 'rollout': 3,
        'epochs': [{'policy_loss': 99, 'gradient_norm': 99, 'kl_early_stop': True}],
        'updates': [{'update': 1, 'policy_loss': -.2, 'entropy': 1., 'approx_kl': .01,
                     'clip_fraction': .1, 'gradient_norm': 2., 'clip_coefficient': .5},
                    {'update': 2, 'policy_loss': -.4, 'entropy': 2., 'approx_kl': .03,
                     'clip_fraction': .3, 'gradient_norm': 4., 'clip_coefficient': .25}],
        'task_curriculum': {'tasks': {'navigation': {'stage': 1}}},
        'advantage_diagnostics': {'tasks': {'navigation': {'raw_advantage': {'mean': 2.}}}},
        'gradient_probes': [{'cycle': 3, 'post_update_kl': .02}],
        'control_coefficient': .8, 'rollout_storage_bytes': 1234,
        'episode_records': [{
            'task': 'navigation', 'outcome': {'known': True, 'success': False},
            'provenance': {'curriculum': {'stage': 'easy', 'level_index': 2}},
            'reward_components': {'control_bonus': [.01, 0, 0],
                                  'unsupported_penalty': [0, 0, -.01]},
            'actions': [{'keys': ['w']}, {'keys': ['w', 'x']}, {'keys': []}],
            'reward_diagnostics': {
                'control_table_sha256': 'a' * 64,
                'control_table': {'predictions': [
                    {'candidate': 'key:w', 'status': 'supported'},
                    {'candidate': 'key:x', 'status': 'unsupported'}]},
                'steps': [{'control_events': [{'event': 'supported_first_use'}]},
                          {'control_events': [{'event': 'unsupported_first_use'}]}]},
        }]})
    episode = row['episode_records'][0]
    assert 'actions' not in episode and 'reward_diagnostics' not in episode
    assert episode['reward_totals'] == {'control_bonus': .01, 'unsupported_penalty': -.01}
    assert episode['action_summary'] == {
        'steps': 3, 'frequencies': {'w': 2, 'x': 1}, 'held': 1, 'released': 2,
        'unsupported_inputs': 1}
    assert episode['control_events']['unsupported_first_use'] == 1
    assert episode['provenance']['curriculum']['stage'] == 'easy'
    series = {item['key']: item for item in training_telemetry([row], ppo=True)['series']}
    assert series['policy_loss']['latest'] == pytest.approx(-.3)
    assert series['policy_loss']['source'] == 'training.jsonl:updates.policy_loss'
    assert series['gradient_norm']['latest'] == 3.
    assert series['optimizer_steps']['latest'] == 2
    assert series['kl_early_stop']['latest'] == 1
    run, _ = ppo_from_parent(tmp_path)
    row['episode_records'][0]['actions'] = [{'keys': ['w']}, {'keys': []}]
    next_row = {**row, 'rollout': 4, 'gradient_probes': []}
    (run / 'training.jsonl').write_text(json.dumps(row) + '\n' + json.dumps(next_row) + '\n')
    current = reader_for(tmp_path, run).snapshot()['current']
    assert current['rl_details']['episodes'][0]['action_summary']['released'] == 1
    assert current['rl_details']['task_curriculum']['tasks']['navigation']['stage'] == 1
    assert current['rl_details']['gradient_probes'][0]['post_update_kl'] == .02
    assert current['rl_details']['control_coefficient'] == .8


def test_ppo_missing_actions_and_control_diagnostics_are_not_measured_zeros():
    from carryrl.dashboard.training_metrics import compact_episode, training_telemetry

    assert compact_episode({'task': 'navigation'}) == {'task': 'navigation'}
    series = {item['key']: item for item in training_telemetry([
        {'complete': True, 'rollout': 1, 'updates': [],
         'epochs': [{'policy_loss': 100, 'gradient_norm': 100}]}], ppo=True)['series']}
    assert 'policy_loss' not in series
    assert 'gradient_norm' not in series
    assert series['optimizer_steps']['latest'] == 0
