"""Bounded display series from saved, completed training measurements."""

import math
import re
from collections import Counter


def cycle_summary(state, *, now, updated, process_identity=None):
    """Read-only cycle facts; saved completion and live identity are separate."""
    saved_status = state.get('status')
    saved_status = saved_status if isinstance(saved_status, str) else 'unknown'
    terminal = saved_status in {'stopped', 'completed', 'failed', 'error', 'interrupted'}
    stages = state.get('stages')
    stages = stages if isinstance(stages, dict) else {}
    stages = {key: value for key, value in stages.items() if isinstance(value, dict)}
    current_id = state.get('current_stage')
    current_id = current_id if isinstance(current_id, str) else None
    if current_id not in stages:
        current_id = next(reversed(stages), None)
    stage = stages.get(current_id, {})
    saved_identity = stage.get('process_identity')
    identity_valid = (isinstance(saved_identity, dict)
                      and type(saved_identity.get('pid')) is int and saved_identity['pid'] > 0
                      and all(key in saved_identity for key in ('pgid', 'started', 'command')))
    running = (not terminal and stage.get('status') == 'running'
               and identity_valid and process_identity == saved_identity)
    terminal_result_saved = (
        terminal
        and stage.get('status') in ('stopped', 'completed', 'failed', 'error', 'interrupted')
        and stage.get('process_complete') is True
        and type(stage.get('actual_exit_code')) is int
        and _finite(stage.get('finished_unix')) and _finite(state.get('finished_unix'))
        and 0 <= stage['finished_unix'] <= state['finished_unix'])
    age = max(0, now - updated)
    stale = age > 120 and not terminal_result_saved
    status = saved_status
    if not terminal and status == 'running' and not running:
        status = 'stale' if stale else 'unknown'
    first, deadline = state.get('first_compute_unix'), state.get('deadline_unix')
    clock_started = _finite(first) and _finite(deadline)
    budget_valid = ((first is None and deadline is None) or
                    (clock_started and first >= 0 and math.isclose(
                        deadline - first, 86400, rel_tol=0, abs_tol=1e-6)))
    finished = state.get('finished_unix')
    freeze_source = 'finished_unix'
    if terminal and not _finite(finished):
        finished = max([row['finished_unix'] for row in stages.values()
                        if _finite(row.get('finished_unix'))] or [updated])
        freeze_source = 'last_stage_or_state_mtime'
    budget_now = finished if terminal and _finite(finished) else now
    budget = None
    if budget_valid:
        budget = {'total_seconds': 86400, 'started': bool(clock_started),
                  'elapsed_seconds': max(0, budget_now - first) if clock_started else 0,
                  'remaining_seconds': max(0, deadline - budget_now) if clock_started else 86400,
                  'deadline_unix': deadline, 'frozen': terminal,
                  'freeze_source': freeze_source if terminal else None}
    rewards = []
    for name, row in stages.items():
        if row.get('phase') != 'rewards':
            continue
        confirmed = (row.get('status') == 'completed' and type(row.get('actual_exit_code')) is int
                     and row['actual_exit_code'] == 0
                     and row.get('process_complete') is True and bool(row.get('artifacts')))
        artifacts = row.get('artifacts')
        quality = {}
        if (isinstance(artifacts, list) and artifacts
                and all(isinstance(item, dict)
                        and type(item.get('quality_gate_passed')) is bool for item in artifacts)):
            decisions = {item['quality_gate_passed'] for item in artifacts}
            if len(decisions) == 1:
                quality['quality_check_passed'] = decisions.pop()
        progress = _reward_progress(row.get('progress'))
        rewards.append({'id': name, 'status': row.get('status', 'unknown'),
                        'confirmed': confirmed, 'actual_exit_code': row.get('actual_exit_code'),
                        **({'progress': progress} if progress else {}), **quality})
    phase = stage.get('phase', 'preparation')
    phase = phase if isinstance(phase, str) else 'preparation'
    labels = {'rewards': 'Подготовка и проверка наград', 'probe': 'Техническая проба PPO',
              'pilot': 'Сравнение расписаний', 'full': 'Основное обучение',
              'final': 'Итоговая проверка', 'preparation': 'Подготовка цикла'}
    decisions = state.get('decisions')
    decisions = decisions if isinstance(decisions, dict) else {}
    reason = stage.get('error') or state.get('error')
    reason = reason[:500].splitlines()[0] if isinstance(reason, str) and reason else None
    details = None
    artifact_gate = (phase == 'rewards' and stage.get('status') in ('stopped', 'failed', 'error')
                     and type(stage.get('actual_exit_code')) is int
                     and stage['actual_exit_code'] == 0 and stage.get('process_complete') is True)
    if artifact_gate and reason:
        text_ready = (re.fullmatch(r'text_fit(?:_[A-Za-z0-9_-]+)?', current_id or '')
                      and re.fullmatch(
                          r'CycleStopped: Stage artifact gate failed: /[^\r\n]+/'
                          r'reward_text(?:_[A-Za-z0-9_-]+)?/manifest\.json:ready', reason))
        visual_ready = (re.fullmatch(r'visual_gate(?:_[A-Za-z0-9_-]+)?', current_id or '')
                        and re.fullmatch(
                            r'CycleStopped: Stage artifact gate failed: /[^\r\n]+/'
                            r'reward_visual(?:_[A-Za-z0-9_-]+)?/validation_report\.json:'
                            r'(?:passed|gate\.passed|coverage_passed)', reason))
        if text_ready or visual_ready:
            details = reason
            reason = ('Текстовый оценщик не прошёл проверку качества.' if text_ready else
                      'Визуальный оценщик не прошёл проверку награды.')
            if not any(row.get('phase') in ('probe', 'pilot', 'full') for row in stages.values()):
                reason += ' Обучение игрового агента не запускалось.'
    return {'status': status, 'saved_status': saved_status,
            'phase': phase, 'phase_label': labels.get(phase, phase), 'stage_id': current_id,
            'stage_status': stage.get('status'), 'running': bool(running),
            'identity_confirmed': bool(running), 'stale': stale, 'age_seconds': age,
            'terminal_result_saved': bool(terminal_result_saved),
            'budget_valid': budget_valid, 'budget': budget, 'rewards': rewards,
            'actual_exit_code': stage.get('actual_exit_code'),
            'failure_reason': reason, 'failure_details': details,
            'memory_paused': running and not stale and stage.get('memory_paused') is True,
            'resource_sample': stage.get('resource_sample') or {},
            'selected_schedule': decisions.get('selected_schedule'),
            'full_rollouts': decisions.get('full_rollouts')}


def _reward_progress(value):
    """Display only supplied scalar progress; do not infer completion or percentages."""
    if not isinstance(value, dict):
        return {}
    result = {key: value[key][:100] for key in ('stage', 'status')
              if isinstance(value.get(key), str) and value[key]}
    result.update({key: value[key] for key in (
        'epoch', 'total_epochs', 'optimizer_updates', 'train_examples', 'loss')
        if _finite(value.get(key)) and (key == 'loss' or value[key] >= 0)})
    return result


def compact_episode(episode):
    """Keep readable aggregates; never cache frames, action streams or step diagnostics."""
    result = {key: episode[key] for key in ('task', 'outcome', 'reward_sum', 'provenance')
              if key in episode}
    components = episode.get('reward_components')
    if isinstance(components, dict):
        result['reward_totals'] = {
            key: sum(values) for key, values in components.items()
            if isinstance(values, list) and values and all(_finite(v) for v in values)}
    actions = episode.get('actions')
    if isinstance(actions, list) and actions and all(
            isinstance(a, dict)
            and all(isinstance(a.get(key, []), list)
                    and all(isinstance(value, str) for value in a.get(key, []))
                    for key in ('keys', 'mouse_buttons'))
            and all(_finite(a.get(axis, 0)) for axis in ('dx', 'dy', 'wheel'))
            for a in actions):
        frequencies, previous, held, released = Counter(), set(), 0, 0
        for action in actions:
            keys = {str(key) for key in action.get('keys', [])}
            keys.update('mouse:' + str(key) for key in action.get('mouse_buttons', []))
            held += len(keys & previous)
            released += len(previous - keys)
            frequencies.update(keys)
            frequencies.update(name for name in ('dx', 'dy', 'wheel') if action.get(name))
            previous = keys
        result['action_summary'] = {'steps': len(actions), 'frequencies': dict(frequencies),
                                    'held': held, 'released': released}
    diagnostics = episode.get('reward_diagnostics')
    if isinstance(diagnostics, dict):
        steps = diagnostics.get('steps')
        steps = steps if isinstance(steps, list) else []
        events = [event for step in steps if isinstance(step, dict)
                  and isinstance(step.get('control_events'), list)
                  for event in step['control_events'] if isinstance(event, dict)]
        if any(isinstance(step, dict) and 'control_events' in step for step in steps):
            result['control_events'] = dict(Counter(event.get('event') for event in events
                                                    if isinstance(event.get('event'), str)))
        reasons = Counter(event['reason'] for event in events
                          if isinstance(event.get('reason'), str))
        if reasons:
            result['control_event_reasons'] = dict(reasons)
        composition = diagnostics.get('composition')
        if isinstance(composition, dict):
            result['reward_composition'] = composition
        if _finite(diagnostics.get('control_coefficient')):
            result['control_coefficient'] = diagnostics['control_coefficient']
        totals = result.get('reward_totals', {})
        budget = {name: totals[key] for name, key in (
            ('positive_paid', 'control_bonus'), ('negative_contribution', 'unsupported_penalty'))
            if key in totals}
        if budget:
            result['control_budget'] = budget
        table = diagnostics.get('control_table')
        if isinstance(table, dict) and isinstance(table.get('predictions'), list):
            result['control_table'] = table
            result['control_table_sha256'] = diagnostics.get('control_table_sha256')
            if 'action_summary' in result:
                predictions = {row.get('candidate'): row.get('status')
                               for row in table.get('predictions', []) if isinstance(row, dict)}
                unsupported = 0
                for action in actions:
                    candidates = ([f'key:{key}' for key in action.get('keys', [])]
                                  + [f'mouse:{key}' for key in action.get('mouse_buttons', [])]
                                  + [f'{axis}:{"+" if action[axis] > 0 else "-"}'
                                     for axis in ('dx', 'dy', 'wheel') if action.get(axis)])
                    unsupported += sum(predictions.get(key) == 'unsupported' for key in candidates)
                result['action_summary']['unsupported_inputs'] = unsupported
    return result

BC_METRICS = (
    ('nll', 'Ошибка повторения (NLL)', 'nat',
     'Среднее эпизодных средних NLL во время обучения; не замороженная validation-проверка.'),
    ('active_nll', 'NLL активных действий', 'nat',
     'Ошибка активных действий: оценивайте вместе с NLL ожидания.'),
    ('idle_nll', 'NLL ожидания', 'nat',
     'Ошибка ожидания: низкое значение само по себе может скрывать плохие активные действия.'),
    ('objective', 'Целевая функция BC', '', 'Сохранённая взвешенная целевая функция BC.'),
    ('unweighted_objective', 'Цель без весов replay', '', 'Средняя цель без весов replay.'),
    ('bc_objective', 'Компонента BC', '', 'Сохранённая компонента BC целевой функции.'),
    ('anchor_kl', 'KL опорной политики', 'nat', 'Средняя KL по доступным опорным эпизодам.'),
    ('total_objective', 'Полная целевая функция', '', 'BC и взвешенная компонента anchor KL.'),
    ('mean_batch_bc_objective', 'Средняя BC по батчам', '', 'Средняя BC по обучающим батчам.'),
    ('mean_batch_total_objective', 'Средняя полная цель по батчам', '',
     'Средняя полная целевая функция по обучающим батчам.'),
    ('max_gradient_norm', 'Максимальная норма градиента', '',
     'Максимум по батчам эпохи, до clipping; не норма после ограничения.'),
    ('epoch_training_seconds', 'Длительность эпохи', 's', 'Измеренное время обучения эпохи.'),
)
PPO_METRICS = (
    ('policy_loss', 'Ошибка политики PPO', '',
     'Целевая функция политики; может быть отрицательной.'),
    ('value_loss', 'Ошибка оценки ценности', '', 'Ошибка предсказания ценности.'),
    ('entropy', 'Энтропия действий', 'nat', 'Энтропия распределения действий.'),
    ('approx_kl', 'Приближённая KL', 'nat', 'Изменение политики относительно сбора эпизодов.'),
    ('clip_fraction', 'Доля clipping', 'fraction', 'Доля отношений вероятностей вне clip_range.'),
    ('gradient_norm', 'Норма градиента PPO', '',
     'Средняя норма до clipping; эпохи с KL early stop исключены: '
     'их сохранённое среднее включает отклонённый батч только в знаменателе.'),
    ('optimizer_steps', 'Принятые шаги последней эпохи PPO', 'steps',
     'Число реально применённых шагов оптимизатора в последней эпохе.'),
    ('kl_early_stop', 'Остановка эпохи по KL', 'flag',
     '1 — эпоха остановлена по KL, 0 — сохранён флаг отсутствия такой остановки.'),
)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _bounded(points):
    if len(points) <= 200:
        return points
    return [points[index * (len(points) - 1) // 199] for index in range(200)]


def training_telemetry(rows, *, imitation=False, ppo=False):
    """Never infer missing values or mix partial epochs into comparable curves."""
    completed = [row for row in rows if row.get('complete') is True]
    axis = 'rollout' if ppo else 'epoch'
    if imitation and completed and all(_finite(row.get('optimizer_steps')) for row in completed):
        axis = 'optimizer_steps'
    result = {'series': [], 'resources': [], 'rows_count': len(completed),
              'axis_label': {'rollout': 'Цикл PPO', 'epoch': 'Эпоха',
                             'optimizer_steps': 'Шаг оптимизатора'}[axis],
              'source': 'training.jsonl'}
    if not imitation and not ppo:
        return result

    def series(key, label, unit, description, *, nested=False, memory=None):
        points = []
        applied = False
        for row in completed:
            step = row.get(axis)
            values = row
            if nested:
                epochs = row.get('epochs')
                values = epochs[-1] if isinstance(epochs, list) and epochs else {}
                updates = row.get('updates')
                if isinstance(updates, list) and key not in {'kl_early_stop'}:
                    applied = True
                    measured = [update[key] for update in updates
                                if isinstance(update, dict) and _finite(update.get(key))]
                    values = {key: len(updates) if key == 'optimizer_steps' else
                              sum(measured) / len(measured) if measured else None}
            if memory:
                values = row.get('mlx_memory_bytes', {})
            value = values.get(memory or key) if isinstance(values, dict) else None
            if (nested and key == 'gradient_norm' and isinstance(values, dict)
                    and values.get('kl_early_stop') is True):
                continue
            if nested and key == 'kl_early_stop':
                value = int(value) if type(value) is bool else None
            if not _finite(step) or step < 0 or not _finite(value):
                continue
            if memory:
                if value < 0:
                    continue
                value /= 1024 ** 2
            points.append({'step': step, 'value': value})
        if not points:
            return
        source_key = (f'updates.{key}' if applied else f'epochs[-1].{key}') if nested else (
            f'mlx_memory_bytes.{memory}' if memory else key)
        if applied:
            description = ('Число применённых обновлений за цикл.' if key == 'optimizer_steps'
                           else 'Среднее по реально применённым обновлениям цикла. '
                           'Отклонённые по KL пакеты исключены. Старые циклы без updates '
                           'показывают последнюю сохранённую эпоху.')
            if key == 'optimizer_steps':
                label = 'Применённые обновления за цикл'
        item = {'key': key, 'label': label, 'unit': unit, 'description': description,
                'source': f'training.jsonl:{source_key}',
                'points': _bounded(points), 'latest': points[-1]['value'],
                'latest_step': points[-1]['step']}
        result['resources' if memory else 'series'].append(item)

    for key, label, unit, description in BC_METRICS if imitation else PPO_METRICS:
        series(key, label, unit, description + (
            ' Последняя эпоха каждого завершённого цикла PPO; сама эпоха могла '
            'быть остановлена по KL.' if ppo else
            ' Завершённые эпохи BC.'), nested=ppo)
    if ppo:
        series('cycle_wall_seconds', 'Длительность цикла PPO', 's',
               'Измеренное время завершённого цикла сбора и обновления PPO.')
    for memory, label, description in (
        ('active', 'Активная память MLX', 'Активные выделения MLX'),
        ('cache', 'Кеш памяти MLX', 'Кеш аллокатора MLX'),
        ('peak', 'Пиковая память MLX', 'Накопленный пик выделений MLX с момента сброса счётчика'),
    ):
        series(f'memory_{memory}_mib', label, 'MiB',
               f'{description}; сохранённый снимок, не общая RAM Mac и не RSS процесса.',
               memory=memory)
    return result
