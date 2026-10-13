"""Read-only, bounded training overview; never imports trainers or loads weights."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import re
import struct
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TASKS = ('navigation', 'reaction', 'mouse', 'memory', 'timing', 'platforming', 'steering',
         'camera', 'interaction', 'causality', 'key_door', 'goal_choice', 'hazards',
         'shooting', 'ui', 'chain')
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_GIF_BYTES = 8 * 1024 * 1024
MAX_RUNS = 200
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z')
SHA = re.compile(r'[a-f0-9]{64}\Z')
DISPLAY_TIMEZONE = ZoneInfo('Europe/Moscow')
DESCRIPTOR_FIELDS = ('task', 'seed', 'instruction', 'policy', 'action_seed', 'episode_index',
                     'difficulty', 'level_index', 'level_id', 'presentation',
                     'initial_frame_sha256')
PHASES = ('Подготовка', 'Начальная проверка', 'Обучение', 'Итоговая проверка', 'Сохранение')
PHASE_LABELS = {'loading': 'Загрузка моделей', 'collecting': 'Сбор экранных примеров',
                'optimizing': 'Обновление политики PPO',
                'encoding': 'Кодирование кадров', 'baseline': 'Начальная проверка',
                'initial_evaluation': 'Начальная проверка', 'initial_eval': 'Начальная проверка',
                'imitation': 'Обучение на примерах', 'training': 'Обучение',
                'evaluation': 'Промежуточная проверка',
                'interim_evaluation': 'Промежуточная проверка',
                'final_evaluation': 'Итоговая проверка', 'final_eval': 'Итоговая проверка',
                'saving': 'Сохранение контрольной точки', 'finished': 'Завершено'}
MEASURED_PHASES = {'collecting', 'encoding', 'baseline', 'initial_evaluation', 'initial_eval',
                   'evaluation', 'interim_evaluation', 'final_evaluation', 'final_eval'}


def _utc(timestamp):
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def _timestamp(value):
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return value if math.isfinite(value) else None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except (ValueError, AttributeError, TypeError):
        return None


def _measurement_timestamp(value):
    """Require an explicit timezone for saved measurement provenance."""
    if not isinstance(value, str):
        return None
    try:
        measured = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if measured.tzinfo is None or measured.utcoffset() is None:
            return None
        stamp = measured.timestamp()
        return stamp if math.isfinite(stamp) else None
    except (ValueError, OverflowError, TypeError):
        return None


def _number(value):
    return (value if isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0 else None)


def task_guides():
    # Reuse prose without importing play.py, which imports the simulation and agent.
    source = Path(__file__).parents[1] / 'play.py'
    try:
        for node in ast.parse(source.read_text()).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'TASK_GUIDE'
                for target in node.targets
            ):
                guides = ast.literal_eval(node.value)
                return {task: {'label': values[0], 'objective': values[1], 'controls': values[2]}
                        for task, values in guides.items()}
    except (OSError, ValueError, SyntaxError):
        pass
    return {task: {'label': task, 'objective': '', 'controls': ''} for task in TASKS}


class OverviewReader:
    """Fingerprint cached immutable evaluations and media, bounded to 512 entries."""

    def __init__(self, runs: Path | str, registry: Path | str, *, clock=time.time,
                 process_alive=None):
        self.root = Path(runs).resolve()
        self.registry = Path(registry).resolve()
        self.clock = clock
        self.process_alive = process_alive or self._process_alive
        self.cache = OrderedDict()
        self.cache_sizes = {}
        self.cache_bytes = 0
        self.lock = threading.RLock()
        self.guides = task_guides()
        self.warnings = []
        self.snapshots = OrderedDict()
        self.derived = OrderedDict()
        self.dependencies = None

    @staticmethod
    def _fingerprint(path):
        try:
            stat = path.lstat()
            return stat.st_mtime_ns, stat.st_size, stat.st_mode, stat.st_ino
        except OSError:
            return None

    def _track(self, path):
        if self.dependencies is not None:
            self.dependencies[path] = self._fingerprint(path)

    def _track_run(self, run):
        # Directory mtimes detect newly published evaluations/media without reading
        # RGB datasets or model weights. Existing files are tracked by _read.
        for path in (run, run / 'evaluations', run / 'playbacks', run / 'checkpoints',
                     run / 'diagnostics', run / 'diagnostics/robustness_matrix',
                     run / 'diagnostics/robustness_corrected_environment'):
            self._track(path)

    @staticmethod
    def _process_alive(pid):
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return None
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _warn(self, message):
        if message not in self.warnings and len(self.warnings) < 100:
            self.warnings.append(message)

    def _safe(self, path, parent=None):
        parent = parent or self.root
        try:
            return (not path.is_symlink() and path.is_file() and parent in path.resolve().parents
                    and not any(p.is_symlink() for p in path.parents if p != parent
                                and parent in p.parents))
        except OSError:
            return False

    def _read(self, path, *, optional=True, outside=False):
        self._track(path)
        if not path.exists() and optional:
            return {}
        try:
            if not self._safe(path, self.registry.parent if outside else self.root):
                raise ValueError('not a regular file within the allowed directory')
            stat = path.stat()
            if stat.st_size > MAX_JSON_BYTES:
                raise ValueError('JSON byte limit exceeded')
            key = ('json', str(path), stat.st_mtime_ns, stat.st_size)
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            with path.open('rb') as stream:
                content = stream.read(MAX_JSON_BYTES + 1)
            if len(content) > MAX_JSON_BYTES:
                raise ValueError('JSON byte limit exceeded')
            data = json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f'non-finite JSON number: {value}')))
            if not isinstance(data, dict):
                raise ValueError('expected a JSON object')
            for field in ('config', 'settings', 'summaries', 'dataset', 'per_task_nll'):
                if field in data and not isinstance(data[field], dict):
                    self._warn(f'{path.name}: malformed {field} omitted')
                    data[field] = {}
            self._cache(key, data)
            return data
        except (OSError, ValueError, UnicodeError) as error:
            self._warn(f'{path.name}: {error}')
            return {}

    def _cache(self, key, value):
        size = len(value) if isinstance(value, bytes) else len(json.dumps(value))
        self.cache_bytes -= self.cache_sizes.pop(key, 0)
        self.cache[key] = value
        self.cache_sizes[key] = size
        self.cache_bytes += size
        self.cache.move_to_end(key)
        while len(self.cache) > 512 or self.cache_bytes > 64 * 1024 * 1024:
            removed, _ = self.cache.popitem(last=False)
            self.cache_bytes -= self.cache_sizes.pop(removed)

    def _run(self, run_id):
        if not isinstance(run_id, str) or not NAME.fullmatch(run_id) or run_id in {'.', '..'}:
            return None
        path = self.root / run_id
        if path.is_symlink() or not path.is_dir() or path.resolve().parent != self.root:
            return None
        return path

    def _active(self, registry):
        value = registry.get('run', registry.get('run_id'))
        if isinstance(value, str):
            path = Path(value)
            path = path if path.is_absolute() else self.root / path.name
            if path.resolve().parent == self.root:
                run = self._run(path.name)
                if run:
                    return run
        self._warn('Активный запуск не найден в реестре.')
        return None

    def _lineage(self, active):
        chain, seen = [], set()
        while active is not None and len(chain) < 32:
            if active.name in seen:
                self._warn('Цикл в истории контрольных точек; цепочка обрезана.')
                break
            seen.add(active.name)
            metadata = self._read(active / 'metadata.json')
            chain.append(active)
            warm = metadata.get('warm_start_parent')
            warm = warm if isinstance(warm, dict) else {}
            counters = warm.get('source_counters')
            counters = counters if isinstance(counters, dict) else {}
            source = metadata.get('source_checkpoint', warm.get('checkpoint'))
            if not source:
                break
            if not isinstance(source, str):
                self._warn(f'{active.name}: повреждён путь исходной контрольной точки.')
                break
            checkpoint = Path(source)
            if not checkpoint.is_absolute():
                checkpoint = self.root.parent / checkpoint
            if self.root not in checkpoint.resolve().parents or checkpoint.is_symlink():
                self._warn(f'{active.name}: исходная контрольная точка вне каталога запусков.')
                break
            relative = checkpoint.resolve().relative_to(self.root)
            if len(relative.parts) != 3 or relative.parts[1] != 'checkpoints':
                self._warn(f'{active.name}: неподдерживаемый путь исходной контрольной точки.')
                break
            manifest = self._read(checkpoint / 'manifest.json', optional=False)
            expected = metadata.get('source_checkpoint_sha256', warm.get('checkpoint_sha256'))
            if (not isinstance(expected, str) or not SHA.fullmatch(expected)
                    or not isinstance(manifest.get('agent_sha256'), str)
                    or not SHA.fullmatch(manifest['agent_sha256'])
                    or manifest.get('sha256') != expected
                    or manifest.get('agent_sha256') != metadata.get(
                        'initial_agent_sha256', warm.get('agent_sha256'))
                    or warm and (manifest.get('reward_sha256') != warm.get('reward_sha256')
                                 or not isinstance(warm.get('reward_sha256'), str)
                                 or not SHA.fullmatch(warm['reward_sha256'])
                                 or manifest.get('weights_revision') != counters.get(
                                     'weights_revision'))):
                self._warn(f'{active.name}: происхождение весов не подтверждено; история обрезана.')
                break
            active = self._run(relative.parts[0])
        return list(reversed(chain))

    def _records(self, evaluation):
        records = evaluation.get('records')
        if (not isinstance(records, list) or len(records) > 20000
                or evaluation.get('policy_and_reward_frozen') is not True
                or evaluation.get('split') != 'validation'):
            raise ValueError('нет подтверждённой замороженной проверки validation')
        selected = [r for r in records if isinstance(r, dict)
                    and r.get('policy') == 'frozen_deterministic']
        games = {task: {'successes': 0, 'episodes': 0, 'unknown_outcomes': 0}
                 for task in TASKS}
        scoring_version = evaluation.get('scoring_version')
        scoring_version = scoring_version if isinstance(scoring_version, str) else None
        for game in games.values():
            game.update(scoring_version=scoring_version, perfect_score_successes=0
                        if scoring_version else None, perfect_score_percent=None)
        identities = set()
        for record in selected:
            task = record.get('task')
            outcome = record.get('outcome')
            if task not in games or not all(field in record for field in DESCRIPTOR_FIELDS):
                raise ValueError('неполный протокол эпизода')
            descriptor = json.dumps([record[field] for field in DESCRIPTOR_FIELDS], sort_keys=True)
            if descriptor in identities:
                raise ValueError('повторённый эпизод проверки')
            identities.add(descriptor)
            if (not isinstance(outcome, dict) or not isinstance(outcome.get('known'), bool)
                    or outcome['known'] and not isinstance(outcome.get('success'), bool)):
                raise ValueError('исход эпизода повреждён')
            games[task]['episodes'] += 1
            games[task]['unknown_outcomes'] += int(not outcome['known'])
            games[task]['successes'] += int(outcome['known'] and outcome.get('success') is True)
            if scoring_version:
                games[task]['perfect_score_successes'] += int(
                    outcome['known'] and outcome.get('success') is True
                    and outcome.get('score') == 5)
        if any(game['episodes'] == 0 for game in games.values()):
            raise ValueError('проверка не охватывает все 16 игр')
        for game in games.values():
            game['value'] = 100 * game['successes'] / game['episodes']
            if scoring_version:
                game['perfect_score_percent'] = (100 * game['perfect_score_successes']
                                                 / game['episodes'])
        return selected, games

    def _protocol(self, run, metadata, evaluation, records):
        config = metadata.get('config', {})
        warm = metadata.get('warm_start_parent')
        warm = warm if isinstance(warm, dict) else {}
        summary = self._read(run / 'summary.json')
        latest = self._read(run / 'latest.json')
        checkpoint = Path(latest.get('checkpoint', str(run / 'checkpoints/rollout_0000')))
        manifest = (self._read(checkpoint / 'manifest.json')
                    if self.root in checkpoint.resolve().parents else {})
        source_value = metadata.get('source_checkpoint', warm.get('checkpoint'))
        if not manifest and isinstance(source_value, str):
            source = Path(source_value)
            if not source.is_absolute():
                source = self.root.parent / source
            if self.root in source.resolve().parents:
                candidate = self._read(source / 'manifest.json')
                if candidate.get('sha256') == metadata.get(
                        'source_checkpoint_sha256', warm.get('checkpoint_sha256')):
                    manifest = candidate
        suite = summary.get('dataset', {}).get('suite_version')
        if suite is None:
            suite = metadata.get('suite_version')
        if suite is None:
            suite = self._read(run / 'demonstrations.json').get('suite_version')
        if not isinstance(suite, str) or not suite.strip():
            raise ValueError('нет подтверждённой версии набора игр')
        descriptor = {
            'episodes': [{key: r[key] for key in DESCRIPTOR_FIELDS} for r in records],
            'reward_sha256': metadata.get('reward_sha256', warm.get('reward_sha256')),
            'config': {key: config.get(key) for key in (
                'max_episode_steps', 'env_image_size', 'agent', 'visual_config',
                'reward_model_mode', 'episode_instructions')},
            'level_bank_seed': self._level_bank_seed(metadata, evaluation),
            'split': evaluation.get('split'),
            'suite': suite,
            'scoring_version': evaluation.get('scoring_version'),
            'codec': metadata.get('codec_version', manifest.get('codec_version')),
            'verifier': metadata.get('reward_model_version', manifest.get('reward_model_version')),
        }
        return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _level_bank_seed(metadata, evaluation):
        seed = metadata.get('config', {}).get('level_bank_seed', 0)
        if (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32):
            raise ValueError('неверный seed набора уровней')
        if 'level_bank_seed' in evaluation and evaluation['level_bank_seed'] != seed:
            raise ValueError('seed проверки отличается от конфигурации запуска')
        return seed

    def _cohort(self, metadata, evaluation, records):
        difficulties = sorted({_number(record['difficulty']) for record in records},
                              key=lambda value: -1 if value is None else value)
        if not difficulties or any(value is None or value > 1 for value in difficulties):
            raise ValueError('неверная сложность контрольного набора')
        declared = evaluation.get('evaluation_difficulty')
        if declared is not None and (len(difficulties) != 1 or declared != difficulties[0]):
            raise ValueError('сложность проверки не совпадает с эпизодами')
        descriptors = [{field: record[field] for field in DESCRIPTOR_FIELDS} for record in records]
        frames = [{field: record[field] for field in ('task', 'level_id', 'initial_frame_sha256')}
                  for record in records]
        return {'difficulty': difficulties[0] if len(difficulties) == 1 else difficulties,
                'level_bank_seed': self._level_bank_seed(metadata, evaluation),
                'episode_bank_sha256': hashlib.sha256(
                    json.dumps(descriptors, sort_keys=True).encode()).hexdigest(),
                'initial_frame_bank_sha256': hashlib.sha256(
                    json.dumps(frames, sort_keys=True).encode()).hexdigest()}

    @staticmethod
    def _cohort_description(cohort):
        difficulty = cohort['difficulty']
        formatted = (f'{difficulty:g}' if isinstance(difficulty, (int, float)) else
                     ', '.join(f'{value:g}' for value in difficulty))
        seed = cohort['level_bank_seed']
        bank = f'новый набор уровней, seed {seed}' if seed else 'исходный набор уровней, seed 0'
        return (f'Фиксированный контрольный набор 16 мини-игр, сложность {formatted}; {bank}. '
                'Замороженная детерминированная политика; среднее процентов побед по играм. '
                'Перенос на новые игры и оформление оценивается отдельно.')

    def _evaluation_time(self, path, evaluation, *, provenance=False):
        """A copied evaluation retains its verified measurement time, not copy time."""
        measured = _measurement_timestamp(evaluation.get('created_utc'))
        reuse = evaluation.get('initial_evaluation_reuse')
        if isinstance(reuse, dict) and reuse.get('measurement_reused') is True:
            original_time = _measurement_timestamp(reuse.get('source_created_utc'))
            source_value, expected = reuse.get('source'), reuse.get('source_sha256')
            if (original_time is not None and isinstance(source_value, str)
                    and isinstance(expected, str) and SHA.fullmatch(expected)):
                source = Path(source_value)
                if not source.is_absolute():
                    source = self.root.parent / source
                if (self._safe(source) and source.parent.name == 'evaluations'
                        and NAME.fullmatch(source.name) and source.suffix == '.json'):
                    try:
                        stat = source.stat()
                        if stat.st_size <= MAX_JSON_BYTES:
                            key = ('json_sha', str(source), stat.st_mtime_ns, stat.st_size)
                            digest = self.cache.get(key)
                            if digest is None:
                                with source.open('rb') as stream:
                                    content = stream.read(MAX_JSON_BYTES + 1)
                                if len(content) <= MAX_JSON_BYTES:
                                    digest = hashlib.sha256(content).hexdigest()
                                    self._cache(key, digest)
                            if digest == expected:
                                original = self._read(source)
                                if (original.get('policy_and_reward_frozen') is True
                                        and original.get('split') == 'validation'
                                        and _measurement_timestamp(original.get('created_utc')) ==
                                        original_time
                                        and reuse.get('source_epoch') == original.get('epoch')
                                        and reuse.get('source_weights_revision') ==
                                        original.get('weights_revision') ==
                                        evaluation.get('weights_revision')
                                        and original.get('agent_sha256') ==
                                        evaluation.get('agent_sha256')
                                        and original.get('reward_sha256') ==
                                        evaluation.get('reward_sha256')
                                        and original.get('protocol_sha256') ==
                                        evaluation.get('protocol_sha256')
                                        and original.get('records') == evaluation.get('records')):
                                    return ({'measurement_run_id': source.parent.parent.name,
                                             'measurement_source': str(source),
                                             'measurement_reused': True}
                                            if provenance else original_time)
                    except (OSError, ValueError, TypeError):
                        pass
        if provenance:
            return {'measurement_run_id': path.parent.parent.name,
                    'measurement_source': str(path), 'measurement_reused': False}
        return measured if measured is not None else path.stat().st_mtime

    def _scope_quality(self, quality, chain):
        own = quality.get('run_history', [])
        initial = next((point for point in own if point.get('epoch') == 0
                        or point.get('rollout') == 0), None)
        latest = own[-1] if own else None
        history = quality.get('history', [])
        quality['historical_comparison'] = {
            'baseline': history[0]['value'] if history else None,
            'change_pp': history[-1]['value'] - history[0]['value'] if history else None,
            'baseline_run_id': history[0]['run_id'] if history else None}
        quality['measurement_stage'] = ('initial' if latest is initial and initial else
                                         'final' if latest else
                                         'inherited' if quality.get('available') else 'unavailable')
        inherited = quality.get('inherited_from')
        if initial is None and isinstance(inherited, dict):
            initial = next((point for point in quality.get('history', [])
                            if point['run_id'] == inherited['run_id']
                            and point['revision'] == inherited['revision']), None)
        if not latest and quality.get('available'):
            quality['baseline'], quality['change_pp'] = quality['value'], None
            for game in quality['games']:
                game['baseline'], game['change_pp'] = game['value'], None
        if latest and initial:
            quality.update(value=latest['value'], successes=latest['successes'],
                           episodes=latest['episodes'], baseline=initial['value'],
                           change_pp=None if latest is initial else
                           latest['value'] - initial['value'],
                           latest_run_id=(quality.get('latest_run_id') if latest is initial else
                                          latest['run_id']),
                           latest_revision=latest['revision'], evaluated_at=latest['evaluated_at'])
            quality['games'] = [{'task': task, 'label': self.guides[task]['label'], **game,
                                 'baseline': initial['games'][task]['value'],
                                 'change_pp': None if latest is initial else
                                 game['value'] - initial['games'][task]['value']}
                                for task, game in latest['games'].items()]
        point = latest or (quality['history'][-1] if quality.get('history') else None)
        if point:
            quality.update({key: point.get(key) for key in (
                'measurement_run_id', 'measurement_source', 'measurement_reused')})
            quality['measurement_agent_sha256'] = point['agent_sha256']
            quality['measurement_container_run_id'] = point['run_id']
        quality['current_weights_measured'] = False
        if latest and chain:
            current = self._current(chain[-1], {})
            completed = current.get('epochs_completed')
            epoch = latest.get('epoch')
            # A partial epoch changes live weights without producing a saved measurement.
            quality['current_weights_measured'] = (
                epoch is not None and epoch == completed
                and (current.get('status') == 'completed'
                     or not current.get('optimizer_steps') and epoch == 0))
            latest_checkpoint = self._read(chain[-1] / 'latest.json').get('checkpoint')
            if isinstance(latest_checkpoint, str):
                path = Path(latest_checkpoint) / 'manifest.json'
                manifest = self._read(path) if self.root in path.resolve().parents else {}
                quality['current_weights_measured'] &= (
                    manifest.get('agent_sha256') == latest['agent_sha256'])

    def _quality(self, chain):
        key = tuple(run.name for run in chain)
        cached = self.derived.get(key)
        if cached and all(self._fingerprint(path) == fingerprint
                          for path, fingerprint in cached['dependencies'].items()):
            if self.dependencies is not None:
                self.dependencies.update(cached['dependencies'])
            for warning in cached['warnings']:
                self._warn(warning)
            self.derived.move_to_end(key)
            return cached['result']
        outer = self.dependencies
        self.dependencies = {}
        before = set(self.warnings)
        for run in chain:
            self._track(run / 'evaluations')
        try:
            result = self._quality_data(chain)
            self._scope_quality(result, chain)
            games = result.get('games', [])
            own_points = [point for point in result.get('run_history', [])
                          if chain and point['run_id'] == chain[-1].name]
            own_points.extend(point for series in result.get('diagnostic_series', [])
                              for point in series.get('history', []))
            score_point = max(own_points, key=lambda point: (
                point['revision'], _timestamp(point.get('evaluated_at')) or 0)
                ) if own_points else None
            if score_point:
                games = list(score_point['games'].values())
            else:
                games = []
            scoring_version = games[0].get('scoring_version') if games else None
            result.update(scoring_version=scoring_version,
                          scoring_version_label=('Счёт с учётом действий · ' + scoring_version
                                                 if scoring_version else
                                                 'Прежняя версия счёта: сравнение недоступно'),
                          perfect_score_successes=(sum(game['perfect_score_successes']
                                                       for game in games)
                                                   if scoring_version else None),
                          perfect_score_percent=(sum(game['perfect_score_percent']
                                                    for game in games) / len(games)
                                                 if scoring_version else None),
                          perfect_score_run_id=score_point['run_id'] if score_point else None,
                          perfect_score_measurement_run_id=(score_point.get('measurement_run_id')
                                                            if score_point else None),
                          perfect_score_measurement_source=(score_point.get('measurement_source')
                                                            if score_point else None),
                          perfect_score_measurement_stage=(
                              'initial' if score_point and (score_point.get('epoch') == 0
                                                           or score_point.get('rollout') == 0) else
                              'final' if score_point else 'unavailable'),
                          perfect_score_revision=score_point['revision'] if score_point else None,
                          perfect_score_evaluation_file=(score_point['evaluation_file']
                                                         if score_point else None),
                          perfect_score_episodes=score_point['episodes'] if score_point else None,
                          perfect_score_scope='Последняя собственная проверка выбранного запуска',
                          perfect_score_games=([
                              {'task': task, 'label': self.guides[task]['label'], **game}
                              for task, game in score_point['games'].items()]
                                               if score_point else []))
            dependencies = self.dependencies
        finally:
            self.dependencies = outer
        if outer is not None:
            outer.update(dependencies)
        self.derived[key] = {'result': result, 'dependencies': dependencies,
                             'warnings': [w for w in self.warnings if w not in before]}
        while len(self.derived) > MAX_RUNS:
            self.derived.popitem(last=False)
        return result

    def _quality_data(self, chain):
        if chain and self._read(chain[-1] / 'metadata.json').get('kind') == 'sequence_training':
            return self._ppo_quality(chain)
        quality = {'available': False, 'label': 'Победы агента · 16 игр', 'value': None,
                   'baseline': None, 'change_pp': None, 'successes': None, 'episodes': None,
                   'latest_run_id': None, 'latest_revision': None, 'evaluated_at': None,
                   'protocol_description': 'Начальная проверка текущего контрольного набора '
                   'ещё не опубликована. Прежние замеры относятся к предыдущим этапам.',
                   'note': 'Этот замер показывает освоение текущих мини-игр; перенос на новые '
                   'игры и оформление оценивается отдельно.',
                   'cohort': None, 'history': [], 'previous_history': [],
                   'games': [], 'quick': None, 'run_history': []}
        protocol, seen, previous_seen = None, set(), set()
        # Only the active run's full initial bank can establish the current cohort.
        # Preparation must never present inherited motor-warmup wins as a new benchmark.
        if chain:
            active = chain[-1]
            active_metadata = self._read(active / 'metadata.json')
            initial = self._read(active / 'evaluations/epoch_0000.json')
            if initial:
                try:
                    records, _ = self._records(initial)
                    if (initial.get('epoch') != 0
                            or _number(initial.get('weights_revision')) is None
                            or not SHA.fullmatch(initial.get('agent_sha256') or '')
                            or initial.get('level_selection') != 'explicit covering level indices'):
                        raise ValueError('неполная начальная проверка текущего набора')
                    cohort = self._cohort(active_metadata, initial, records)
                    if cohort['level_bank_seed'] and len(records) != 256:
                        raise ValueError('новый полный контрольный набор должен '
                                         'содержать 256 эпизодов')
                    protocol = self._protocol(active, active_metadata, initial, records)
                    quality['cohort'] = cohort
                    quality['protocol_description'] = self._cohort_description(cohort)
                except (ValueError, TypeError, KeyError) as error:
                    self._warn(f'{active.name}/epoch_0000.json: {error}')
        previous_origins = {}
        for run in chain:
            metadata = self._read(run / 'metadata.json')
            directory = run / 'evaluations'
            files = sorted(directory.glob('epoch_*.json'))[:256] if directory.is_dir() else []
            for path in files:
                evaluation = self._read(path, optional=False)
                try:
                    records, games = self._records(evaluation)
                    epoch, revision = evaluation.get('epoch'), evaluation.get('weights_revision')
                    agent = evaluation.get('agent_sha256')
                    if (_number(epoch) is None or _number(revision) is None
                            or not SHA.fullmatch(agent or '')):
                        raise ValueError('нет номера эпохи, ревизии или отпечатка весов')
                    signature = self._protocol(run, metadata, evaluation, records)
                    value = sum(game['value'] for game in games.values()) / len(TASKS)
                    item = {'run_id': run.name, 'epoch': epoch, 'revision': revision,
                            'evaluation_file': path.name,
                            'protocol_sha256': evaluation.get('protocol_sha256'),
                            'value': value,
                            'successes': sum(g['successes'] for g in games.values()),
                            'episodes': len(records), 'agent_sha256': agent,
                            'evaluated_at': _utc(self._evaluation_time(path, evaluation)),
                            'games': games}
                    item.update(self._evaluation_time(path, evaluation, provenance=True))
                    is_full = (evaluation.get('level_selection')
                               == 'explicit covering level indices')
                    if not is_full:
                        if run == chain[-1]:
                            quality['quick'] = {key: item[key] for key in
                                                ('value', 'successes', 'episodes', 'epoch')}
                        continue
                    cohort = self._cohort(metadata, evaluation, records)
                    if cohort['level_bank_seed'] and len(records) != 256:
                        raise ValueError('новый полный контрольный набор должен '
                                         'содержать 256 эпизодов')
                    if protocol is None or signature != protocol:
                        if run != chain[-1] and (agent, signature) not in previous_seen:
                            previous_seen.add((agent, signature))
                            origin = previous_origins.setdefault(signature, revision)
                            quality['previous_history'].append({
                                **item, 'updates': revision - origin, 'cohort': cohort,
                                'protocol_description': self._cohort_description(cohort)})
                        if protocol is not None and (run == chain[-1] or (
                                cohort['level_bank_seed'] == quality['cohort']['level_bank_seed']
                                and cohort['difficulty'] == quality['cohort']['difficulty'])):
                            raise ValueError('протокол/набор эпизодов отличается; точка исключена')
                        continue
                    if run == chain[-1]:
                        quality['run_history'].append(item)
                    if (agent, signature) in seen:
                        continue
                    seen.add((agent, signature))
                    quality['history'].append(item)
                except (ValueError, TypeError, KeyError) as error:
                    self._warn(f'{run.name}/{path.name}: {error}')
        history = quality['history']
        if not history:
            return quality
        origin = history[0]['revision']
        for item in history:
            item['updates'] = item['revision'] - origin
        first, last = history[0], history[-1]
        quality.update({'available': True, 'value': last['value'], 'baseline': first['value'],
                        'change_pp': last['value'] - first['value'],
                        'successes': last['successes'], 'episodes': last['episodes'],
                        'latest_run_id': last['run_id'], 'latest_revision': last['revision'],
                        'evaluated_at': last['evaluated_at']})
        quality['games'] = [{'task': task, 'label': self.guides[task]['label'],
                             **last['games'][task], 'baseline': first['games'][task]['value'],
                             'change_pp': last['games'][task]['value'] -
                             first['games'][task]['value']} for task in TASKS]
        return quality

    def _ppo_quality(self, chain):
        active = chain[-1]
        metadata = self._read(active / 'metadata.json')
        warm = metadata.get('warm_start_parent')
        warm = warm if isinstance(warm, dict) else {}
        counters = warm.get('source_counters')
        counters = counters if isinstance(counters, dict) else {}
        quality = copy.deepcopy(self._quality(chain[:-1]))
        quality['run_history'] = []
        quality['quick'] = None
        quality['diagnostic_series'] = []
        quality['inherited_from'] = None
        quality['canonical_measurement_current'] = False
        anchor = None
        if quality['history'] and len(chain) > 1:
            last = quality['history'][-1]
            if (last['agent_sha256'] == warm.get('agent_sha256')
                    and last['revision'] == counters.get('weights_revision')):
                quality['inherited_from'] = {
                    'run_id': last['run_id'], 'revision': last['revision'],
                    'checkpoint': warm.get('checkpoint'), 'evaluated_at': last['evaluated_at'],
                    'value': last['value'], 'successes': last['successes'],
                    'episodes': last['episodes']}
                anchor_run = self._run(last['run_id'])
                if anchor_run in chain[:-1] and NAME.fullmatch(last.get('evaluation_file', '')):
                    candidate = self._read(anchor_run / 'evaluations' / last['evaluation_file'])
                    if (candidate.get('agent_sha256') == last['agent_sha256']
                            and candidate.get('weights_revision') == last['revision']
                            and candidate.get('level_selection')
                            == 'explicit covering level indices'
                            and isinstance(candidate.get('protocol_sha256'), str)
                            and SHA.fullmatch(candidate['protocol_sha256'])
                            and candidate['protocol_sha256'] == last.get('protocol_sha256')):
                        anchor = candidate
            else:
                self._warn(f'{active.name}: исходная полная проверка '
                           'не подтверждает идентичность исходных весов.')
                quality = self._quality([]) | {
                    'diagnostic_series': [], 'inherited_from': None,
                    'canonical_measurement_current': False}
        series = {}
        directory = active / 'evaluations'
        for path in sorted(directory.glob('rollout_*.json'))[:256]:
            evaluation = self._read(path)
            try:
                records, games = self._records(evaluation)
                rollout, revision = evaluation.get('rollout'), evaluation.get('weights_revision')
                agent, reward = evaluation.get('agent_sha256'), evaluation.get('reward_sha256')
                declared_protocol = evaluation.get('protocol_sha256')
                initial_revision = counters.get('weights_revision', revision)
                if (type(rollout) is not int or rollout < 0
                        or type(revision) is not int or revision < 0
                        or type(initial_revision) is not int or revision < initial_revision
                        or warm and revision == initial_revision and agent != warm.get(
                            'agent_sha256')
                        or not isinstance(agent, str) or not SHA.fullmatch(agent)
                        or not isinstance(reward, str) or not SHA.fullmatch(reward)
                        or not isinstance(declared_protocol, str)
                        or not SHA.fullmatch(declared_protocol)
                        or warm and reward != warm.get('reward_sha256')):
                    raise ValueError('нет подтверждённой идентичности проверки PPO')
                cohort = self._cohort(metadata, evaluation, records)
                signature = self._protocol(active, metadata, evaluation, records)
                signature = hashlib.sha256(
                    f'{signature}:{declared_protocol}'.encode()).hexdigest()
                value = sum(game['value'] for game in games.values()) / len(TASKS)
                item = {'run_id': active.name, 'epoch': None, 'rollout': rollout,
                        'evaluation_file': path.name, 'protocol_sha256': declared_protocol,
                        'revision': revision, 'agent_sha256': agent,
                        'value': value, 'successes': sum(g['successes'] for g in games.values()),
                        'episodes': len(records), 'games': games,
                        'evaluated_at': _utc(self._evaluation_time(path, evaluation)),
                        'updates': revision - initial_revision}
                item.update(self._evaluation_time(path, evaluation, provenance=True))
                full = evaluation.get('level_selection') == 'explicit covering level indices'
                if full and cohort['level_bank_seed'] and len(records) != 256:
                    raise ValueError('полный набор PPO должен содержать 256 эпизодов')
                canonical = (full and anchor is not None and cohort == quality['cohort']
                             and evaluation.get('protocol_sha256') == anchor['protocol_sha256'])
                if canonical:
                    quality['run_history'].append(item)
                    if any(point['agent_sha256'] == agent for point in quality['history']):
                        continue
                    item['updates'] = revision - quality['history'][0]['revision']
                    quality['history'].append(item)
                    quality.update({'value': value, 'successes': item['successes'],
                                    'episodes': len(records), 'latest_run_id': active.name,
                                    'latest_revision': revision,
                                    'evaluated_at': item['evaluated_at'],
                                    'change_pp': value - quality['baseline'],
                                    'canonical_measurement_current': True})
                    for game in quality['games']:
                        game.update(games[game['task']])
                        game['change_pp'] = game['value'] - game['baseline']
                else:
                    group = series.setdefault(signature, {
                        'protocol_sha256': evaluation.get('protocol_sha256'),
                        'cohort': cohort, 'coverage': 'full' if full else 'sampled',
                        'description': 'Отдельная проверка PPO; оформление и набор эпизодов '
                                       'не объединяются с основной проверкой.', 'history': []})
                    group['history'].append(item)
            except (ValueError, TypeError, KeyError) as error:
                self._warn(f'{active.name}/{path.name}: {error}')
        quality['diagnostic_series'] = list(series.values())
        return quality

    def _training_rows(self, run):
        path = run / 'training.jsonl'
        self._track(path)
        if not self._safe(path):
            return []
        try:
            stat = path.stat()
            if stat.st_size > MAX_JSON_BYTES:
                self._warn('training.jsonl: превышен лимит чтения.')
                return []
            key = ('training', str(path), stat.st_mtime_ns, stat.st_size)
            if key in self.cache:
                return self.cache[key]
            rows = []
            with path.open() as stream:
                for line in stream:
                    try:
                        row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError(f'non-finite JSON number: {value}')))
                        if isinstance(row, dict):
                            rows.append(row)
                    except (ValueError, UnicodeError):
                        self._warn('training.jsonl: неполная строка пропущена.')
            rows = rows[-10000:]
            self._cache(key, rows)
            return rows
        except OSError:
            return []

    def _current(self, run, registry):
        metadata = self._read(run / 'metadata.json')
        status = self._read(run / 'status.json')
        summary = self._read(run / 'summary.json')
        settings = metadata.get('settings', {})
        config = metadata.get('config', {})
        imitation = (metadata.get('kind') == 'imitation_training'
                     or summary.get('method') == 'behavioural_cloning_joint_action_nll')
        ppo = not imitation and config.get('total_rollouts') is not None
        rows = self._training_rows(run)
        complete_rows = [row for row in rows if row.get('complete') is True]
        completed = max([_number(row.get('epoch')) or 0 for row in complete_rows] +
                        [_number(status.get('epochs_completed')) or 0,
                         _number(summary.get('epochs_completed')) or 0])
        last = rows[-1] if rows else {}
        epochs = _number(status.get('epochs_total', settings.get('epochs'))) if imitation else None
        demonstrations = _number(status.get('demonstration_episodes'))
        minibatch = _number(settings.get('minibatch_episodes'))
        batches = _number(status.get('batches_per_epoch')) if imitation else None
        if imitation and batches is None and demonstrations and minibatch:
            batches = math.ceil(demonstrations / minibatch)
        total_steps = _number(status.get('optimizer_steps_total')) if imitation else None
        if total_steps is None and epochs is not None and batches is not None:
            total_steps = epochs * batches
        steps = (_number(status.get('optimizer_steps', last.get('optimizer_steps')))
                 if imitation else None)
        if ppo:
            steps = _number(status.get('optimizer_steps_this_process',
                                       summary.get('optimizer_steps_this_process')))
        now = self.clock()
        updated = _timestamp(status.get('updated_at', status.get('updated_utc')))
        if updated is None:
            status_path = run / 'status.json'
            updated = status_path.stat().st_mtime if status_path.exists() else now
        age = max(0, now - updated)
        stale = age > 120
        process = {}
        launch = registry.get('launch', str(self.root / (run.name + '_launch')))
        if isinstance(launch, str):
            launch_path = Path(launch)
            if self.root in launch_path.resolve().parents:
                process = self._read(launch_path / 'process.json')
        running = (False if process.get('status') == 'exited' else self.process_alive(
            status.get('pid', process.get('command_pid'))))
        state = status.get('status', 'unknown')
        if state in {'completed', 'resource_stopped', 'failed', 'error', 'interrupted', 'stopped'}:
            running = False
        elif running is False:
            state = 'failed' if process.get('actual_exit_code') not in (None, 0) else 'interrupted'
        elif running is True:
            state = 'running'
        elif stale:
            state = 'stale'
        else:
            state = 'unknown'
        phase = status.get('phase', 'finished' if state == 'completed' else 'unknown')
        if not isinstance(phase, str):
            phase = 'unknown'
        phase_index = (5 if phase in {'saving', 'finished'} else
                       4 if phase in {'final_evaluation', 'final_eval'} else
                       3 if phase in {'imitation', 'training', 'evaluation',
                                     'interim_evaluation'} else
                       2 if phase in {'initial_evaluation', 'initial_eval', 'baseline'} else 1)
        phase_plan = status.get('phase_plan', metadata.get('phase_plan')) if ppo else None
        if (not isinstance(phase_plan, list) or not phase_plan
                or any(not isinstance(group, dict) or not isinstance(group.get('id'), str)
                       or not isinstance(group.get('phases'), list) for group in phase_plan)):
            phase_plan = None
        if phase_plan:
            matched = next((index for index, group in enumerate(phase_plan, 1)
                            if phase in group['phases']), None)
            if matched is not None:
                phase_index = matched
        elapsed = _number(status.get('elapsed_seconds', status.get('wall_seconds')))
        phase_elapsed = _number(status.get('phase_elapsed_seconds'))
        started = _timestamp(status.get('started_at', process.get('started_utc')))
        phase_started = _timestamp(status.get('phase_started_at'))
        if state == 'running' and started is not None:
            elapsed = max(0, now - started)
        if state == 'running' and phase_started is not None:
            phase_elapsed = max(0, now - phase_started)
        units_done = _number(status.get('phase_units_done'))
        units_total = _number(status.get('phase_units_total'))
        if phase in {'imitation', 'training'} and units_total is None:
            units_done, units_total = steps, total_steps
        eta = low = high = None
        scope, reason = None, 'Недостаточно измерений скорости текущей фазы.'
        if state == 'completed':
            eta = low = high = 0
            scope, reason = 'run', 'Запуск завершён.'
        elif state != 'running' or stale:
            reason = 'Нет подтверждённого свежего активного процесса.'
        elif (phase in MEASURED_PHASES and units_done is not None and units_done >= 3
              and units_total is not None and units_total >= units_done
              and phase_elapsed is not None and phase_elapsed > 0):
            eta = (units_total - units_done) * phase_elapsed / units_done
            scope = 'phase'
            reason = ('До конца текущей фазы; последующие этапы отдельно. '
                      'Приблизительно по среднему темпу с начала фазы.')
        elif phase in {'imitation', 'training'}:
            measured = [r for r in complete_rows if _number(r.get('seconds')) is not None]
            rates = []
            recent = measured[-6:]
            for row in recent:
                duration = _number(row.get('epoch_training_seconds'))
                if duration and batches:
                    rates.append(duration / batches)
            if not rates:
                every = _number(settings.get('eval_every'))
                for previous, current in zip(recent, recent[1:]):
                    # Legacy cumulative seconds include the previous epoch's synchronous
                    # validation callback. Exclude that interval from training speed.
                    if every and previous.get('epoch', 0) % every == 0:
                        continue
                    dt = current['seconds'] - previous['seconds']
                    du = ((_number(current.get('optimizer_steps')) or 0) -
                          (_number(previous.get('optimizer_steps')) or 0))
                    if dt > 0 and du > 0:
                        rates.append(dt / du)
            if len(rates) >= 2 and steps is not None and total_steps is not None:
                remaining = max(0, total_steps - steps)
                eta = remaining * sum(rates) / len(rates)
                low, high = remaining * min(rates), remaining * max(rates)
                scope = 'training'
                reason = 'Только оставшиеся обучающие эпохи; итоговая проверка и сохранение '
                reason += 'после них в прогноз не входят.'
        training_percent = (min(100, 100 * steps / total_steps)
                            if steps is not None and total_steps else None)
        cycle_eta = cycle_low = cycle_high = None
        if not imitation:
            rollouts = _number(status.get('rollouts', status.get('rollout')))
            rollout_total = _number(status.get('rollouts_total', config.get('total_rollouts')))
            if rollouts is not None and rollout_total:
                training_percent = min(100, 100 * rollouts / rollout_total)
            cycle_rows = {row['rollout']: row for row in rows
                          if type(row.get('rollout')) is int and row['rollout'] > 0
                          and row.get('complete') is not False
                          and _number(row.get('cycle_wall_seconds'))
                          and rollouts is not None and row['rollout'] <= rollouts}
            durations = [row['cycle_wall_seconds']
                         for _, row in sorted(cycle_rows.items())[-6:]]
            if (ppo and state == 'running' and not stale and durations and rollout_total
                    and not status.get('stop_reason', summary.get('stop_reason'))):
                remaining = max(0, rollout_total - rollouts)
                cycle_eta = remaining * sum(durations) / len(durations)
                cycle_low, cycle_high = remaining * min(durations), remaining * max(durations)
            if ppo and state == 'completed':
                cycle_eta = cycle_low = cycle_high = 0
        batch = _number(status.get('batch')) if imitation else None
        if (batch is None and state == 'completed' and steps == total_steps
                and completed == epochs and len(complete_rows) >= 2):
            final, previous = complete_rows[-1], complete_rows[-2]
            if (final.get('epoch') == epochs and final.get('optimizer_steps') == steps
                    and final['optimizer_steps'] - previous.get('optimizer_steps', 0) == batches):
                batch = batches
        current = {
            'run_id': run.name, 'label': run.name, 'status': state,
            'status_label': {'completed': 'Завершено', 'running': 'Работает',
                             'resource_stopped': 'Остановлен по лимиту ресурсов',
                             'interrupted': 'Прерван: процесс не работает',
                             'stopped': 'Остановлен', 'error': 'Остановлено с ошибкой',
                             'failed': 'Остановлено с ошибкой', 'stale': 'Данные устарели',
                             'unknown': 'Статус неизвестен'}.get(state, state),
            'running': running, 'phase': phase,
            'stop_reason': status.get('stop_reason', summary.get('stop_reason')),
            'phase_label': PHASE_LABELS.get(phase) or PHASES[min(phase_index, len(PHASES)) - 1],
            'phase_index': phase_index, 'phase_total': 5,
            'epoch': status.get('epoch', status.get('epochs', last.get('epoch')))
            if imitation else None,
            'epochs_completed': completed if imitation else None, 'epochs_total': epochs,
            'optimizer_steps': steps,
            'optimizer_steps_total': total_steps, 'batch': batch,
            'batches_per_epoch': batches, 'training_percent': training_percent,
            'elapsed_seconds': elapsed, 'phase_elapsed_seconds': phase_elapsed,
            'eta_seconds': eta, 'eta_low_seconds': low, 'eta_high_seconds': high,
            'eta_scope': scope, 'eta_reason': reason,
            'finish_estimated_at': (_utc(now + eta) if eta is not None and eta > 0
                                    and scope == 'run' else None),
            'last_update_at': _utc(updated), 'data_age_seconds': age, 'stale': stale,
            'method': ('behavioural_cloning_joint_action_nll' if imitation else
                       'ppo' if config.get('total_rollouts') is not None else
                       metadata.get('command', metadata.get('kind', 'unknown'))),
            'method_description': ('Обучение на экранных примерах программного учителя. '
                                   'Ошибка показывает точность повторения; качество игры '
                                   'проверяется отдельно.' if imitation else
                                   'PPO: обучение по награде на собранных игровых эпизодах.'
                                   if config.get('total_rollouts') is not None else
                                   'Метод и протокол указаны в исходных данных запуска.'),
            'rollouts': status.get('rollouts', status.get('rollout', last.get('rollout'))),
            'rollouts_total': status.get('rollouts_total', config.get('total_rollouts')),
            'optimizer_steps_upper_bound': _number(status.get('optimizer_steps_upper_bound')),
            'cycle_eta_seconds': cycle_eta, 'cycle_eta_low_seconds': cycle_low,
            'cycle_eta_high_seconds': cycle_high,
            'decisions': status.get('decisions'),
            'attempted_train_decisions_this_process': (
                status['attempted_train_decisions_this_process']
                if type(status.get('attempted_train_decisions_this_process')) is int
                and status['attempted_train_decisions_this_process'] >= 0 else None),
            'weights_revision': status.get('weights_revision'),
            'nll': _number(status.get('nll', last.get('nll'))),
            'loss_history': [{'epoch': r['epoch'], 'nll': r['nll']} for r in complete_rows
                             if _number(r.get('nll')) is not None and 'epoch' in r],
            'games_count': len(TASKS),
            'phase_steps': [{'label': label, 'state': 'complete' if state == 'completed' or
                             index < phase_index else 'active' if index == phase_index else
                             'pending'} for index, label in enumerate(PHASES, 1)],
            'phase_units_done': units_done, 'phase_units_total': units_total,
            'phase_history': status.get('phase_history', []),
            'phase_unit': status.get('phase_unit', status.get('phase_unit_kind')),
            'settings': {'config': metadata.get('config', {}), 'training': settings},
            'per_task_nll': last.get('per_task_nll', {}),
        }
        if ppo:
            current['phase_label'] = ('Сбор игровых эпизодов PPO' if phase == 'collecting' else
                                      PHASE_LABELS.get(phase, 'Обучение по награде'))
            current['current_rollout'] = status.get('current_rollout', status.get('current_cycle'))
        if phase_plan:
            labels = {'preparation': 'Подготовка', 'baseline': 'Начальная проверка',
                      'training': 'Обучение', 'final_evaluation': 'Итоговая проверка',
                      'saving': 'Сохранение', 'completed': 'Готово'}
            current['phase_index'], current['phase_total'] = phase_index, len(phase_plan)
            current['phase_steps'] = [
                {'label': labels.get(group['id'], group['id']),
                 'state': 'complete' if state in {'completed', 'resource_stopped'}
                 or index < phase_index else
                 'active' if index == phase_index else 'pending'}
                for index, group in enumerate(phase_plan, 1)]
        return current

    def _gif(self, path, digest):
        self._track(path)
        if not SHA.fullmatch(digest or '') or not self._safe(path) or path.suffix != '.gif':
            raise ValueError('GIF path or SHA256 is invalid')
        stat = path.stat()
        if stat.st_size > MAX_GIF_BYTES:
            raise ValueError('GIF byte limit exceeded')
        key = ('gif', str(path), stat.st_mtime_ns, stat.st_size, digest)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        with path.open('rb') as stream:
            data = stream.read(MAX_GIF_BYTES + 1)
        if (len(data) > MAX_GIF_BYTES or len(data) < 13
                or data[:6] not in {b'GIF87a', b'GIF89a'}
                or not all(1 <= v <= 2048 for v in struct.unpack('<HH', data[6:10]))
                or hashlib.sha256(data).hexdigest() != digest):
            raise ValueError('GIF content/hash verification failed')
        self._cache(key, data)
        return data

    def _sprite(self, path, metadata):
        self._track(path)
        digest = metadata.get('sprite_sha256')
        width, height = metadata.get('frame_width'), metadata.get('frame_height')
        count, columns = metadata.get('frame_count'), metadata.get('sprite_columns')
        if (not all(isinstance(v, int) and not isinstance(v, bool)
                    for v in (width, height, count, columns))
                or not 1 <= width <= 256 or not 1 <= height <= 256
                or not 1 <= count <= 81 or columns != 8
                or metadata.get('frame_duration_ms') != 100
                or not SHA.fullmatch(digest or '') or not self._safe(path)
                or path.suffix != '.png'):
            raise ValueError('invalid sprite dimensions or SHA256')
        stat = path.stat()
        if stat.st_size > MAX_GIF_BYTES:
            raise ValueError('sprite byte limit exceeded')
        key = ('sprite', str(path), stat.st_mtime_ns, stat.st_size, digest,
               width, height, count, columns)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        with path.open('rb') as stream:
            data = stream.read(MAX_GIF_BYTES + 1)
        if (len(data) > MAX_GIF_BYTES or len(data) < 33
                or data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR'
                or struct.unpack('>II', data[16:24]) !=
                (width * columns, height * math.ceil(count / columns))
                or hashlib.sha256(data).hexdigest() != digest):
            raise ValueError('sprite content/dimensions/hash verification failed')
        self._cache(key, data)
        return data

    def _frozen_playback_evidence(self, run, metadata):
        epoch = metadata.get('epoch')
        if not isinstance(epoch, int) or isinstance(epoch, bool) or not 0 <= epoch <= 9999:
            return False
        path = run / 'evaluations' / f'epoch_{epoch:04}.json'
        final = path.with_name(f'epoch_{epoch:04}_final.json')
        reference = metadata.get('evaluation_reference', metadata.get('eval_reference'))
        if reference is not None:
            if not isinstance(reference, str):
                return False
            declared = Path(reference)
            if not declared.is_absolute():
                declared = self.root.parent / declared
            if (declared.resolve().parent != (run / 'evaluations').resolve()
                    or declared.name not in {path.name, final.name}):
                return False
            path = declared
        elif final.exists():
            path = final
        if not path.exists():
            status = self._read(run / 'status.json')
            updated = _timestamp(status.get('updated_at', status.get('updated_utc')))
            if updated is None and (run / 'status.json').exists():
                updated = (run / 'status.json').stat().st_mtime
            if (status.get('status') == 'running'
                    and status.get('phase') in MEASURED_PHASES - {'collecting', 'encoding'}
                    and updated is not None and 0 <= self.clock() - updated <= 120):
                return None  # Captures precede atomic publication of the evaluation JSON.
            return False
        evaluation = self._read(path)
        if (evaluation.get('agent_sha256') != metadata.get('agent_sha256')
                or evaluation.get('weights_revision') != metadata.get('weights_revision')
                or evaluation.get('policy_and_reward_frozen') is not True):
            return False
        return any(isinstance(record, dict) and all(record.get(field) == metadata.get(field)
                   for field in ('task', 'seed', 'level_id', 'instruction', 'presentation',
                                 'difficulty', 'policy'))
                   and record.get('steps') == metadata.get('action_count')
                   and record.get('outcome') == metadata.get('outcome')
                   for record in evaluation.get('records', []))

    def _playbacks(self, chain, quality):
        items = []
        current_agent = current_revision = None
        if chain:
            active = chain[-1]
            latest = self._read(active / 'latest.json')
            checkpoint_value = latest.get('checkpoint')
            if isinstance(checkpoint_value, str):
                checkpoint = Path(checkpoint_value)
                if active in checkpoint.resolve().parents:
                    manifest = self._read(checkpoint / 'manifest.json')
                    current_agent = manifest.get('agent_sha256')
                    current_revision = manifest.get('weights_revision')
            if current_agent is None:
                # Before the first save, the latest frozen evaluation is the only
                # observable weight snapshot. Never inspect the live model.
                for path in sorted((active / 'evaluations').glob('epoch_*.json'))[:256]:
                    evaluation = self._read(path)
                    revision = _number(evaluation.get('weights_revision'))
                    try:
                        self._records(evaluation)
                        if revision is not None and (current_revision is None
                                                     or revision > current_revision):
                            current_revision = revision
                            current_agent = evaluation.get('agent_sha256')
                    except (ValueError, TypeError, KeyError):
                        pass
        for run in reversed(chain):
            directory = run / 'playbacks'
            if not directory.is_dir() or directory.is_symlink():
                continue
            for sidecar in sorted(directory.glob('*.json'))[:128]:
                metadata = self._read(sidecar)
                try:
                    if (metadata.get('policy') != 'frozen_deterministic'
                            or metadata.get('source') !=
                            'actual_frozen_learned_policy_closed_loop_RGB_playback'
                            or metadata.get('teacher_actions_used') is not False
                            or metadata.get('training_performed') is not False
                            or metadata.get('task') not in TASKS):
                        raise ValueError('playback is not a frozen learned policy')
                    filename = Path(metadata.get('gif', '')).name
                    if not NAME.fullmatch(filename) or not filename.endswith('.gif'):
                        raise ValueError('invalid GIF filename')
                    self._gif(directory / filename, metadata.get('gif_sha256'))
                    checkpoint_value = metadata.get('checkpoint')
                    if checkpoint_value:
                        checkpoint = Path(checkpoint_value)
                        if run not in checkpoint.resolve().parents:
                            raise ValueError('checkpoint does not belong to playback run')
                        manifest = self._read(checkpoint / 'manifest.json')
                        if (metadata.get('checkpoint_sha256') != manifest.get('sha256') or
                                metadata.get('agent_sha256') != manifest.get('agent_sha256') or
                                metadata.get('weights_revision') !=
                                manifest.get('weights_revision')):
                            raise ValueError('playback weights do not match checkpoint manifest')
                    else:
                        evidence = self._frozen_playback_evidence(run, metadata)
                        if evidence is None:
                            continue
                        if not evidence:
                            raise ValueError('playback lacks matching frozen evaluation evidence')
                    task = metadata['task']
                    current = (run == chain[-1] and metadata.get('weights_revision') ==
                               current_revision and metadata.get('agent_sha256') == current_agent)
                    epoch = metadata.get('epoch')
                    if epoch is None:
                        epoch = self._read(run / 'summary.json').get('epochs_completed')
                    created_at = metadata.get('created_utc')
                    if _measurement_timestamp(created_at) is None:
                        created_at = _utc(sidecar.stat().st_mtime)
                    item = {'id': f'{run.name}/{filename}', 'run_id': run.name,
                            'task': task, 'label': self.guides[task]['label'],
                            'url': f'/media/{run.name}/{filename}', 'epoch': epoch,
                            'revision': metadata.get('weights_revision'),
                            'policy': metadata['policy'], 'outcome': metadata.get('outcome'),
                            'steps': metadata.get('action_count'),
                            'created_at': created_at,
                            'source_label': ('Последние проверенные веса' if current else
                                             'Предыдущий этап обучения') +
                            f' · эпоха {epoch} · ревизия {metadata.get("weights_revision")}',
                            'current_weights': current}
                    experiment = self._experiment(run, quality=quality)
                    item.update(experiment_title=experiment['title'],
                                group_id=experiment['group_id'],
                                checkpoint=metadata.get('checkpoint'),
                                agent_sha256=metadata.get('agent_sha256'))
                    sprite_name = metadata.get('sprite_filename')
                    if sprite_name is not None:
                        try:
                            if (not NAME.fullmatch(sprite_name) or
                                    sprite_name != Path(filename).stem + '.frames.png'):
                                raise ValueError('invalid sprite filename')
                            self._sprite(directory / sprite_name, metadata)
                            transitions = metadata.get('actions')
                            count = metadata['frame_count']
                            if not isinstance(transitions, list) or len(transitions) != count - 1:
                                raise ValueError('sprite frame/action counts do not align')
                            actions = []
                            for transition in transitions:
                                action = transition.get('action')
                                if (not isinstance(action, dict) or not all(
                                    isinstance(action.get(field), list)
                                    and len(action[field]) <= 32
                                    and all(isinstance(v, str) and len(v) <= 32
                                            for v in action[field])
                                    for field in ('keys', 'mouse_buttons')) or not all(
                                    isinstance(action.get(field), int)
                                    and not isinstance(action[field], bool)
                                    for field in ('dx', 'dy', 'wheel'))):
                                    raise ValueError('invalid playback physical action')
                                actions.append({key: action[key] for key in
                                                ('keys', 'mouse_buttons', 'dx', 'dy', 'wheel')})
                            item.update({key: metadata[key] for key in
                                         ('sprite_columns', 'frame_width', 'frame_height',
                                          'frame_count', 'frame_duration_ms', 'terminal_hold_ms')})
                            item['sprite_url'] = f'/media/{run.name}/{sprite_name}'
                            item['actions'] = actions
                        except (ValueError, TypeError, KeyError, AttributeError) as error:
                            self._warn(f'{run.name}/{sidecar.name}: {error}; GIF retained.')
                    items.append(item)
                except (ValueError, TypeError, OSError) as error:
                    self._warn(f'{run.name}/{sidecar.name}: {error}')
        items.sort(key=lambda item: (-_measurement_timestamp(item['created_at']), item['id']))
        return {'items': items, 'selected_id': items[0]['id'] if items else None}

    def media(self, run_id, filename):
        with self.lock:
            if (not NAME.fullmatch(filename or '')
                    or not filename.endswith(('.gif', '.frames.png'))):
                return None
            overview = self.snapshot(include_experiments=False, run_id=run_id)
            selected = next((item for item in overview['playbacks']['items']
                             if item['run_id'] == run_id and any(
                                 Path(item.get(field, '')).name == filename
                                 for field in ('url', 'sprite_url'))), None)
            if selected is None:
                return None
            run = self._run(run_id)
            for path in (run / 'playbacks').glob('*.json'):
                metadata = self._read(path)
                try:
                    if Path(metadata.get('gif', '')).name == filename:
                        return self._gif(run / 'playbacks' / filename, metadata.get('gif_sha256'))
                    if metadata.get('sprite_filename') == filename:
                        return self._sprite(run / 'playbacks' / filename, metadata)
                except (ValueError, TypeError, OSError):
                    return None
            return None

    def _experiment(self, run, *, quality=None):
        metadata = self._read(run / 'metadata.json')
        summary = self._read(run / 'summary.json')
        settings = metadata.get('settings', {})
        config = metadata.get('config', {})
        source = metadata.get('source_checkpoint', metadata.get('warm_start',
                              metadata.get('resume', metadata.get('resume_from',
                              metadata.get('checkpoint')))))
        baseline_run = None
        if isinstance(source, str):
            parts = Path(source).parts
            if 'checkpoints' in parts:
                index = parts.index('checkpoints')
                baseline_run = parts[index - 1] if index else None
        imitation = metadata.get('kind') == 'imitation_training'
        ppo = (metadata.get('kind') == 'sequence_training'
               or config.get('total_rollouts') is not None)
        method = 'Метод не описан в сохранённых метаданных.'
        purpose = 'Назначение этого запуска не записано; доступные артефакты приведены ниже.'
        title = 'Запуск без описания'
        group_id = run.name
        if imitation:
            title = 'Обучение по экранным примерам'
            purpose = 'Проверить, помогает ли повторение действий программного учителя '
            purpose += 'самостоятельно решать мини-игры на контрольных уровнях.'
            details = []
            for field, label in (('epochs', 'эпох'), ('episodes_per_task', 'примеров на механику'),
                                 ('learning_rate', 'скорость обучения')):
                if field == 'episodes_per_task' and settings.get('demonstration_reuse_run'):
                    continue
                if settings.get(field) is not None:
                    details.append(f'{settings[field]} {label}')
            if settings.get('correction_episodes_per_task'):
                details.append('исправления после действий ученика')
                title = 'Исправления ошибок ученика'
            if settings.get('memory_cue_pairs'):
                details.append('парные примеры для памяти')
                title = 'Парные примеры и исправления'
            if settings.get('presentation_copies_per_episode'):
                details.append('изменённое оформление тех же механик')
                title = 'Обучение с разным оформлением'
            if settings.get('native_train_sizes'):
                details.append('размеры кадров ' + '/'.join(
                    map(str, settings['native_train_sizes'])))
                title = 'Оформление и размеры кадров'
            if settings.get('correction_replacement_bundle'):
                title = 'Обновлённые исправления ошибок'
            if settings.get('demonstration_reuse_run'):
                weight = _number(settings.get('canonical_replay_weight', 1))
                weight_label = f'{weight:g}' if weight is not None else 'не указан'
                title = ('Контроль: обычный вес экранных примеров' if weight == 1 else
                         f'Опыт: вес обычных экранных примеров {weight_label}')
                purpose = ('Проверить, помогает ли уменьшение веса обычного оформления '
                           'играть при изменённых цветах, текстурах и размере кадра.')
                details.append(f'вес обычного оформления {weight_label}')
                demonstrations = _number(self._read(run / 'status.json').get(
                    'demonstration_episodes'))
                if demonstrations is not None:
                    details.append(f'{demonstrations:g} сохранённых эпизодов из прежнего набора')
                group_id = 'visual_replay:' + str(settings['demonstration_reuse_run'])
            method = 'Повторение действий учителя (BC)' + (
                ': ' + '; '.join(details) + '.' if details else '.')
        elif ppo:
            title = 'Обучение на собственной игре'
            purpose = 'Проверить, улучшает ли PPO самостоятельную игру после исходных весов.'
            method = 'PPO: агент собирает игровые эпизоды и обновляет действия по экранной награде.'
            if metadata.get('curriculum', {}).get('warmup_rollouts'):
                title = 'Собственная игра: от лёгких уровней к сложным'
            if config.get('learning_rate') is not None:
                method += f' Скорость обучения {config["learning_rate"]}.'
        elif metadata.get('command') in {'evaluate', 'random'} or metadata.get('updates') == 0:
            title = ('Случайные действия: контроль' if metadata.get('command') == 'random'
                     else 'Проверка сохранённого агента')
            purpose = 'Измерить успешность игры без обновления весов.'
            method = 'Самостоятельные эпизоды на сохранённом контрольном наборе.'
        elif metadata.get('candidate'):
            title = 'Техническая проверка обучения навигации'
            purpose = 'Проверить работу небольшого агента PPO по одному RGB-кадру.'
            method = str(metadata['candidate'])
        backend = settings.get('training_backend', metadata.get('training_backend',
                              config.get('training_backend', config.get('device'))))
        budget = settings.get('epochs', config.get('total_rollouts'))
        if not settings.get('demonstration_reuse_run'):
            suffix = [str(backend).upper()] if backend else []
            if budget is not None:
                suffix.append(f'{budget} ' + ('эпох' if imitation else 'циклов'))
            if suffix:
                title += ' · ' + ', '.join(suffix)
            if imitation and settings.get('episodes_per_task') is not None:
                title += f', {settings["episodes_per_task"]} примеров/механику'
            if settings.get('action_balance') is not None:
                title += (' · баланс действий' if settings['action_balance'] == 'active_idle'
                          else ' · равный вес действий')
        limitations = ['Результаты разных контрольных наборов нельзя напрямую сравнивать.',
                       'Победы в этих мини-играх не доказывают перенос на неизвестные игры.']
        if imitation:
            limitations.append('Успех программного учителя и ошибка повторения не являются '
                               'победами обученного агента.')
        conclusion = 'Итоговый замер самостоятельной игры не опубликован.'
        measurement = None
        sources = [{'run_id': run.name, 'path': 'metadata.json', 'role': 'условия опыта'}]
        quality = quality if quality is not None else self._quality([run])
        history = [point for point in quality.get('run_history', [])
                   if point['run_id'] == run.name]
        if not history:
            series = [series['history'] for series in quality.get('diagnostic_series', [])
                      if series.get('history')]
            if series:
                # Each series has one verified protocol/cohort. Never subtract
                # the initial sampled bank from a later full-bank measurement.
                history = max(series, key=lambda points: (
                    points[-1]['revision'], _timestamp(points[-1].get('evaluated_at')) or 0))
        if history:
            last = history[-1]
            is_update = (_number(last.get('epoch')) or _number(last.get('rollout')) or 0) > 0
            prefix = 'Последний замер' if is_update else 'До обучения'
            conclusion = (f'{prefix}: {last["value"]:.1f}% побед, '
                          f'{last["successes"]} из {last["episodes"]} эпизодов.')
            if is_update and len(history) > 1:
                change = last['value'] - history[0]['value']
                conclusion += f' Изменение от начала: {change:+.1f} п.п.'
            if not is_update:
                conclusion += ' Итог опыта пока не измерен.'
            sources.append({'run_id': run.name,
                            'path': 'evaluations/' + last['evaluation_file'],
                            'role': 'замер самостоятельной игры'})
            measurement = {key: last.get(key) for key in (
                'run_id', 'evaluation_file', 'epoch', 'rollout', 'revision',
                'agent_sha256', 'value', 'successes', 'episodes', 'evaluated_at')}
        elif summary.get('final_success_rate_lower') is not None:
            rate = _number(summary['final_success_rate_lower'])
            if rate is not None:
                conclusion = f'Сохранённый итог: {rate * 100:.1f}% побед в протоколе этого запуска.'
                sources.append({'run_id': run.name, 'path': 'summary.json', 'role': 'итог'})
        if not summary:
            limitations.append('Сводка завершения отсутствует; итоговое сравнение не подтверждено.')
        latest = self._read(run / 'latest.json')
        checkpoint = latest.get('checkpoint', summary.get('checkpoint',
                                summary.get('latest_checkpoint')))
        result = {'title': title, 'purpose': purpose, 'method': method,
                'baseline_run_id': baseline_run,
                'baseline': ('Исходные веса из предыдущего запуска; начальный замер этого '
                             'опыта служит точкой сравнения.' if baseline_run else
                             'Исходные веса и начальный замер, если они сохранены.'),
                'conclusion': conclusion, 'limitations': limitations, 'sources': sources,
                'checkpoint': checkpoint, 'group_id': group_id, 'measurement': measurement}
        editorial = self._read(run / 'experiment.json')
        for field in ('title', 'purpose', 'method', 'baseline_run_id', 'baseline',
                      'conclusion', 'group_id'):
            if isinstance(editorial.get(field), str) and 0 < len(editorial[field]) <= 4000:
                result[field] = editorial[field]
        if isinstance(editorial.get('limitations'), list) and all(
                isinstance(value, str) and len(value) <= 4000
                for value in editorial['limitations'][:32]):
            result['limitations'] = editorial['limitations'][:32]
        if isinstance(editorial.get('sources'), list):
            for source in editorial['sources'][:32]:
                if isinstance(source, dict) and all(isinstance(source.get(field), str)
                                                  for field in ('run_id', 'path', 'role')):
                    result['sources'].append(source)
        if editorial:
            result['sources'].append({'run_id': run.name, 'path': 'experiment.json',
                                      'role': 'описание опыта по сохранённым отчётам'})
        current = self._current(run, {})
        initial = history[0] if history and (history[0].get('epoch') == 0
                                             or history[0].get('rollout') == 0) else None
        final = history[-1] if history else None
        complete = (final is not None and final is not initial
                    and current['status'] in {'completed', 'resource_stopped'})
        state, verdict_label, change = 'pending', 'без итоговой проверки', None
        recommendation = 'Дождитесь итоговой проверки на том же наборе уровней.'
        recommended = None
        if complete and initial:
            change = final['value'] - initial['value']
            state = 'improved' if change > 0 else 'worse' if change < 0 else 'unchanged'
            verdict_label = {'improved': 'улучшилась', 'worse': 'хуже исходной',
                             'unchanged': 'без улучшения'}[state]
            source_id = initial.get('measurement_run_id')
            source_run = self._run(source_id)
            baseline_verified = False
            if source_run and source_id != run.name:
                source_latest = self._read(source_run / 'latest.json').get('checkpoint')
                if isinstance(source_latest, str):
                    manifest_path = Path(source_latest) / 'manifest.json'
                    manifest = (self._read(manifest_path)
                                if source_run in manifest_path.resolve().parents else {})
                    baseline_verified = (manifest.get('agent_sha256') == initial['agent_sha256']
                                         and manifest.get('weights_revision')
                                         == initial['revision'])
            if state == 'improved' and quality.get('current_weights_measured'):
                recommended = run.name
                recommendation = ('Для этих 16 мини-игр выберите проверенную новую модель: '
                                  'на том же наборе она выигрывает чаще исходной.')
            elif baseline_verified:
                recommended = source_id
                reason = ('новая выигрывает реже' if state == 'worse' else
                          'новая не даёт больше побед')
                recommendation = ('Для этих 16 мини-игр сохраните исходную модель: '
                                  f'{reason} на том же наборе.')
            else:
                recommendation = ('Сравнение относится только к проверенным весам и этому '
                                  'набору уровней; выбор сохранённой модели не подтверждён.')
        elif complete:
            state, verdict_label = 'incomparable', 'нет сопоставимого начала'
            recommendation = ('Сопоставимая начальная проверка отсутствует; '
                              'выбрать победителя нельзя.')
        elif current['status'] == 'interrupted':
            recommendation = 'Запуск прерван; итоговый результат и победитель не определены.'
        stamp = _timestamp(metadata.get('started_at', metadata.get('created_utc',
                           metadata.get('created_at', self._read(run / 'status.json').get(
                               'started_at')))))
        date_label = (datetime.fromtimestamp(stamp, DISPLAY_TIMEZONE).strftime('%d.%m %H:%M')
                      if stamp else '')
        if not date_label:
            measured = _timestamp(final.get('evaluated_at')) if final else None
            fingerprint = self._fingerprint(run / 'metadata.json')
            observed = measured or (fingerprint[0] / 1e9 if fingerprint else None)
            if observed:
                date_label = ('замер ' if measured else 'файл ') + datetime.fromtimestamp(
                    observed, DISPLAY_TIMEZONE).strftime('%d.%m %H:%M')
        score_label = (f'{final["successes"]}/{final["episodes"]} побед' if final else
                       'результат не измерен')
        if final is initial and initial:
            score_label += ' до обучения'
        result.update(picker_label=' · '.join(value for value in (
            date_label, score_label, verdict_label) if value),
            verdict={'state': state, 'label': verdict_label, 'change_pp': change,
                     'baseline_successes': initial['successes'] if initial else None,
                     'successes': final['successes'] if final else None,
                     'episodes': final['episodes'] if final else None},
            recommended_run_id=recommended, recommendation=recommendation)
        return result

    def _experiments(self):
        if not self.root.is_dir():
            return []
        runs = [p for p in self.root.iterdir() if not p.is_symlink() and p.is_dir()
                and (p / 'metadata.json').is_file()]
        def started(run):
            metadata = self._read(run / 'metadata.json')
            status = self._read(run / 'status.json')
            return (_timestamp(metadata.get('started_at', metadata.get('created_utc',
                               metadata.get('created_at', status.get('started_at')))))
                    or run.stat().st_mtime)
        runs.sort(key=started, reverse=True)
        items = []
        for run in runs[:MAX_RUNS]:
            self._track_run(run)
            metadata = self._read(run / 'metadata.json')
            status = self._read(run / 'status.json')
            summary = self._read(run / 'summary.json')
            config = metadata.get('config', {})
            kind = metadata.get('kind', metadata.get(
                'command', 'evaluation' if metadata.get('updates') == 0 else 'training'))
            item = {'run_id': run.name, 'label': run.name, 'kind': kind,
                    'backend': metadata.get('settings', {}).get('training_backend',
                               config.get('training_backend', config.get('device'))),
                    'status': status.get('status', 'unknown'),
                    'started_at': metadata.get('started_at', metadata.get('created_utc',
                                  metadata.get('created_at'))),
                    'elapsed_seconds': status.get('elapsed_seconds', status.get('wall_seconds')),
                    'epochs': status.get('epochs', summary.get('epochs_completed')),
                    'rollouts': status.get('rollout', status.get('rollouts')),
                    'quality_percent': None, 'cohort_description': None}
            item['experiment'] = self._experiment(run)
            item['storage'] = self._read(run / 'storage_manifest.json')
            item['label'] = item['experiment']['title']
            current = self._current(run, {})
            item['status'], item['status_label'] = current['status'], current['status_label']
            measurement = item['experiment']['measurement']
            if measurement:
                item['quality_percent'] = measurement['value']
                item['measurement_run_id'] = measurement['run_id']
                item['cohort_description'] = (f'{measurement["successes"]} побед из '
                                              f'{measurement["episodes"]} эпизодов; '
                                              'контрольный набор этого запуска')
                items.append(item)
                continue
            evaluation = self._read(run / 'evaluation.json')
            if not evaluation:
                directory = run / 'evaluations'
                paths = sorted([*directory.glob('epoch_*.json'),
                                *directory.glob('rollout_*.json')])[:256]
                if paths:
                    evaluation = self._read(paths[-1])
            summaries = evaluation.get('summaries', {})
            deterministic = summaries.get('frozen_deterministic', {})
            if not deterministic and 'success_rate_lower' in evaluation:
                deterministic = evaluation
            rate = _number(deterministic.get('success_rate_lower'))
            if rate is not None:
                item['quality_percent'] = rate * 100
                item['cohort_description'] = (f'{evaluation.get("split", "unknown")} · '
                                              f'{deterministic.get("episodes_started", "?")} '
                                              'эпизодов; отдельный протокол этого запуска')
            measurement = item['experiment']['measurement']
            if measurement:
                item['quality_percent'] = measurement['value']
                item['measurement_run_id'] = measurement['run_id']
                item['cohort_description'] = (f'{measurement["successes"]} побед из '
                                              f'{measurement["episodes"]} эпизодов; '
                                              'контрольный набор этого запуска')
            items.append(item)
        return items

    def _checkpoint_manifest_digest(self, path):
        self._track(path)
        try:
            if not self._safe(path) or path.stat().st_size > MAX_JSON_BYTES:
                return None
            with path.open('rb') as stream:
                content = stream.read(MAX_JSON_BYTES + 1)
            return hashlib.sha256(content).hexdigest() if len(content) <= MAX_JSON_BYTES else None
        except OSError:
            return None

    def _transfer_report(self, run):
        for name in ('robustness_corrected_environment', 'robustness_matrix'):
            directory = run / 'diagnostics' / name
            aggregate = self._read(directory / 'aggregate.json')
            if not aggregate or aggregate.get('status') != 'completed':
                continue
            protocol = self._read(directory / 'protocol.json')
            try:
                declaration = {k: v for k, v in protocol.items() if k != 'matrix_sha256'}
                digest = hashlib.sha256(
                    json.dumps(declaration, sort_keys=True).encode()).hexdigest()
                if (protocol.get('format') != 'carryrl.robustness_matrix.v1'
                        or protocol.get('matrix_sha256') != digest
                        or any(aggregate.get(k) != v for k, v in protocol.items())
                        or not protocol.get('environment_source_sha256')):
                    raise ValueError('неподтверждённый протокол переноса')
                revision = aggregate.get('weights_revision')
                if type(revision) is not int or revision < 0:
                    raise ValueError('нет подтверждённой ревизии переноса')
                checkpoint = protocol['checkpoint']
                if (not isinstance(checkpoint, dict)
                        or not isinstance(checkpoint.get('files_sha256'), dict)):
                    raise ValueError('нет отпечатков checkpoint отчёта')
                checkpoint_path = Path(checkpoint['path'])
                manifest_path = checkpoint_path / 'manifest.json'
                manifest_digest = self._checkpoint_manifest_digest(manifest_path)
                if (run.resolve() not in checkpoint_path.resolve().parents
                        or manifest_digest is None
                        or manifest_digest
                        != checkpoint['files_sha256'].get('manifest.json')):
                    raise ValueError('отпечаток checkpoint отчёта не подтверждён')
                checkpoint_revision = self._read(manifest_path).get('weights_revision')
                if type(checkpoint_revision) is not int or checkpoint_revision != revision:
                    raise ValueError('ревизия переноса отличается от checkpoint')
                rows = []
                for spec in protocol['strata']:
                    if (not isinstance(spec['profile'], str)
                            or spec['language'] not in ('ru', 'en')):
                        raise ValueError('неверные условия матрицы')
                    metric = aggregate['metrics'][spec['id']]['frozen_deterministic']
                    episodes = metric['episodes_started']
                    success = metric['outcomes']['success']
                    unknown = metric['unknown_outcomes']
                    if (type(episodes) is not int or episodes <= 0
                            or any(type(v) is not int or not 0 <= v <= episodes
                                   for v in (success, unknown)) or success + unknown > episodes):
                        raise ValueError('повреждённые итоги переноса')
                    rows.append({'id': spec['id'], 'profile': spec['profile'],
                                 'language': spec['language'], 'episodes': episodes,
                                 'successes': success, 'unknown_outcomes': unknown,
                                 'value': 100 * success / episodes,
                                 'upper': 100 * (success + unknown) / episodes, 'change_pp': None})
                if len(rows) != 8 or len({row['id'] for row in rows}) != 8:
                    raise ValueError('неполная матрица переноса')
                signature = {key: protocol.get(key) for key in (
                    'format', 'environment_source_sha256', 'strata', 'coverage_mode')}
                cohorts = aggregate.get('mechanical_cohorts')
                if not cohorts or set(cohorts) != {row['id'] for row in rows}:
                    raise ValueError('нет подтверждённого набора эпизодов переноса')
                for spec in protocol['strata']:
                    self._validate_transfer_cohort(spec, cohorts[spec['id']])
                    metric = aggregate['metrics'][spec['id']]['frozen_deterministic']
                    if metric['episodes_started'] != sum(spec['protocol']['counts'].values()):
                        raise ValueError('число эпизодов переноса отличается от протокола')
                signature['cohorts'] = cohorts
                signature = hashlib.sha256(
                    json.dumps(signature, sort_keys=True).encode()).hexdigest()
                return {'run_id': run.name, 'checkpoint': checkpoint['path'],
                        'checkpoint_sha256': checkpoint['files_sha256'],
                        'revision': aggregate.get('weights_revision'),
                        'evaluated_at': _utc((directory / 'aggregate.json').stat().st_mtime),
                        'coverage_mode': protocol.get('coverage_mode'), 'rows': rows,
                        'signature': signature}
            except (ValueError, TypeError, KeyError, OSError) as error:
                self._warn(f'{run.name}/{name}: {error}')
        return None

    @staticmethod
    def _validate_transfer_cohort(spec, cohort):
        if not isinstance(spec.get('protocol'), dict):
            raise ValueError('нет протокола набора эпизодов переноса')
        counts = spec['protocol'].get('counts')
        policies = spec['protocol'].get('policies')
        if (not isinstance(counts, dict) or set(counts) != set(TASKS)
                or any(type(count) is not int or count <= 0 for count in counts.values())
                or not isinstance(policies, list) or not policies
                or any(not isinstance(policy, str) for policy in policies)
                or len(set(policies)) != len(policies)
                or 'frozen_deterministic' not in policies
                or not isinstance(cohort, dict)
                or set(cohort) != {f'{policy}/{task}' for policy in policies for task in counts}):
            raise ValueError('неполная структура набора эпизодов переноса')
        required = {'episode_index', 'seed', 'level_index', 'level_id', 'semantic_level_id',
                    'runtime_fingerprint', 'difficulty', 'action_seed', 'initial_frame_sha256'}
        for policy in policies:
            for task, count in counts.items():
                entry = cohort[f'{policy}/{task}']
                if not isinstance(entry, dict):
                    raise ValueError('повреждённый набор эпизодов переноса')
                episodes = entry.get('episodes')
                if (not isinstance(episodes, list) or len(episodes) != count
                        or type(entry.get('episodes_started')) is not int
                        or entry['episodes_started'] != count
                        or any(not isinstance(episode, dict) or not required <= episode.keys()
                               for episode in episodes)):
                    raise ValueError('неполный набор эпизодов переноса')
                indices = [episode['episode_index'] for episode in episodes]
                if (any(type(index) is not int or index < 0 for index in indices)
                        or len(set(indices)) != count):
                    raise ValueError('повторённые или повреждённые эпизоды переноса')
                for episode in episodes:
                    if (any(type(episode[key]) is not int or episode[key] < 0
                            for key in ('seed', 'action_seed'))
                            or _number(episode['difficulty']) is None
                            or episode['difficulty'] > 1
                            or any(not isinstance(episode[key], str)
                                   or not SHA.fullmatch(episode[key]) for key in (
                                       'level_id', 'semantic_level_id', 'runtime_fingerprint',
                                       'initial_frame_sha256'))):
                        raise ValueError('повреждённые признаки эпизодов переноса')

    def _diagnostics(self, chain):
        result = {'transfer': None, 'memory': None, 'current_report_available': False,
                  'comparison_reason': 'Сопоставимый предыдущий отчёт пока отсутствует.'}
        if not chain:
            return result
        active = chain[-1]
        latest = self._read(active / 'latest.json')
        latest_path = Path(latest.get('checkpoint', str(active / 'checkpoints/rollout_0000')))
        manifest = self._read(latest_path / 'manifest.json') if (
            self.root in latest_path.resolve().parents) else {}
        reports = [report for run in chain if (report := self._transfer_report(run))]
        if reports:
            current = reports[-1]
            current['current_weights'] = (current['run_id'] == active.name
                                          and current['checkpoint'] == str(latest_path)
                                          and current['revision'] is not None
                                          and current['revision'] == manifest.get(
                                              'weights_revision'))
            result['current_report_available'] = current['current_weights']
            if len(reports) > 1 and current['current_weights']:
                previous = reports[-2]
                if current['signature'] == previous['signature']:
                    before = {row['id']: row for row in previous['rows']}
                    for row in current['rows']:
                        if (not row['unknown_outcomes']
                                and not before[row['id']]['unknown_outcomes']):
                            row['change_pp'] = row['value'] - before[row['id']]['value']
                    result['comparison_reason'] = f'Сравнение с {previous["run_id"]}: '
                    result['comparison_reason'] += 'протокол, среда и набор эпизодов совпадают.'
                else:
                    result['comparison_reason'] = ('Изменение не рассчитано: протокол матрицы, '
                                                   'среда или набор эпизодов отличаются.')
            if not current['current_weights']:
                result['comparison_reason'] = ('Проверка текущих весов ещё не опубликована. '
                                               'Показан отчёт предыдущей модели; дельты нет.')
            result['transfer'] = {k: v for k, v in current.items() if k != 'signature'}
        for run in reversed(chain):
            memory = self._read(run / 'diagnostics/memory_controls.json')
            if (memory.get('diagnostic') == 'paired_memory_cues'
                    and memory.get('policy_and_reward_frozen') is True
                    and isinstance(memory.get('summaries'), dict)):
                modes = []
                valid = type(memory.get('weights_revision')) is int
                valid = valid and memory['weights_revision'] >= 0
                for name in ('normal', 'neutral_wait'):
                    summary = memory['summaries'].get(name, {})
                    if not isinstance(summary, dict):
                        summary = {}
                    fields = {key: summary.get(key) for key in (
                        'first_choose_correct', 'first_choose_observed',
                        'paired_both_first_choose_correct', 'geometries')}
                    if (any(type(value) is not int or value < 0 for value in fields.values())
                            or type(summary.get('episodes')) is not int
                            or summary['episodes'] != 2 * fields['geometries']
                            or not fields['first_choose_observed'] or not fields['geometries']
                            or fields['first_choose_observed'] > summary['episodes']
                            or fields['first_choose_correct'] > fields['first_choose_observed']
                            or fields['paired_both_first_choose_correct'] > fields['geometries']
                            or 2 * fields['paired_both_first_choose_correct']
                            > fields['first_choose_correct']):
                        valid = False
                    modes.append({'mode': name, **fields})
                if not valid:
                    self._warn(f'{run.name}/memory_controls.json: '
                               'неполные итоги диагностики памяти')
                    continue
                result['memory'] = {'run_id': run.name, 'checkpoint': memory.get('checkpoint'),
                                    'revision': memory.get('weights_revision'),
                                    'evaluated_at': memory.get('created_utc'), 'modes': modes,
                                    'current_weights': run == active
                                    and memory.get('checkpoint') == str(latest_path)
                                    and memory.get('weights_revision') is not None
                                    and memory.get('weights_revision') == manifest.get(
                                        'weights_revision')
                                    and isinstance(memory.get('checkpoint_sha256'), dict)
                                    and memory['checkpoint_sha256'].get('manifest.json')
                                    == self._checkpoint_manifest_digest(
                                        latest_path / 'manifest.json')}
                break
        return result

    def run_details(self, run_id):
        with self.lock:
            run = self._run(run_id)
            if not run:
                return None
            result = {'run_id': run_id}
            for name in ('metadata', 'config', 'status', 'summary', 'evaluation', 'latest'):
                data = self._read(run / f'{name}.json')
                # Large record/action arrays stay in source files; endpoint remains bounded.
                result[name] = {k: v for k, v in data.items() if k not in {'records', 'history'}}
            evaluation_dir = run / 'evaluations'
            result['evaluations'] = []
            if evaluation_dir.is_dir() and not evaluation_dir.is_symlink():
                paths = sorted([*evaluation_dir.glob('epoch_*.json'),
                                *evaluation_dir.glob('rollout_*.json')])[:256]
                for path in paths:
                    data = self._read(path)
                    result['evaluations'].append({'filename': path.name, **{
                        k: v for k, v in data.items() if k not in {'records', 'vision_cache'}}})
            result['checkpoint_manifests'] = []
            checkpoints = run / 'checkpoints'
            if checkpoints.is_dir() and not checkpoints.is_symlink():
                for path in sorted(checkpoints.glob('*/manifest.json'))[:256]:
                    result['checkpoint_manifests'].append({'checkpoint': path.parent.name,
                                                          **self._read(path)})
            result['artifacts'] = [p.name for p in sorted(run.iterdir())[:256]
                                   if self._safe(p) and p.suffix in {'.json', '.jsonl', '.gif'}]
            result['current'] = self._current(run, {})
            result['quality'] = self._quality(self._lineage(run))
            result['diagnostics'] = self._diagnostics(self._lineage(run))
            result['experiment'] = self._experiment(run)
            result['storage'] = self._read(run / 'storage_manifest.json')
            result['current']['experiment'] = result['experiment']
            result['current']['label'] = result['experiment']['title']
            result['playbacks'] = self._playbacks([run], result['quality'])
            result['warnings'] = list(self.warnings)
            return result

    def snapshot(self, *, include_experiments=True, run_id=None):
        with self.lock:
            key = run_id, include_experiments
            cached = self.snapshots.get(key)
            if cached and all(self._fingerprint(path) == fingerprint
                              for path, fingerprint in cached['dependencies'].items()):
                self.warnings = list(cached['result']['warnings'])
                result = dict(cached['result'])
                active = self._run(result['selected_run_id']) if result['selected_run_id'] else None
                registry = self._read(self.registry, outside=True)
                if active:
                    current = self._current(active, registry if run_id is None else {})
                    current['experiment'] = result['current']['experiment']
                    current['label'] = current['experiment']['title']
                    result['current'] = current
                actual = self._active(registry)
                if actual:
                    training = self._current(actual, registry)
                    previous = cached['result'].get('training_current') or {}
                    training['label'] = previous.get('label', training['label'])
                    result['training_current'] = training
                else:
                    result['training_current'] = None
                if include_experiments:
                    result['experiments'] = []
                    for item in cached['result']['experiments']:
                        current = self._current(self._run(item['run_id']), {})
                        result['experiments'].append({**item, 'status': current['status'],
                                                      'status_label': current['status_label']})
                result['updated_at'] = _utc(self.clock())
                result['warnings'] = list(self.warnings)
                self.snapshots.move_to_end(key)
                return result
            self.warnings = []
            self.dependencies = {}
            self._track(self.root)
            registry = self._read(self.registry, outside=True)
            active = self._run(run_id) if run_id is not None else self._active(registry)
            chain = self._lineage(active) if active else []
            for run in chain:
                self._track_run(run)
            quality = self._quality(chain)
            result = {'schema': 'carryrl.overview.v1', 'updated_at': _utc(self.clock()),
                      'selected_run_id': active.name if active else None,
                      'current': self._current(active, registry if run_id is None else {})
                      if active else None,
                      'quality': quality, 'playbacks': self._playbacks(chain, quality),
                      'diagnostics': self._diagnostics(chain),
                      'task_guides': self.guides}
            if active:
                result['current']['experiment'] = self._experiment(active)
                result['current']['label'] = result['current']['experiment']['title']
            actual = self._active(registry)
            if actual:
                training = self._current(actual, registry)
                training['label'] = self._experiment(actual)['title']
                result['training_current'] = training
            else:
                result['training_current'] = None
            if include_experiments:
                result['experiments'] = self._experiments()
            result['warnings'] = list(self.warnings)
            self.snapshots[key] = {'result': result, 'dependencies': self.dependencies}
            self.dependencies = None
            while len(self.snapshots) > 8:
                self.snapshots.popitem(last=False)
            return result
