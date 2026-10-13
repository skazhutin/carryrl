'use strict';
(() => {
  const main = document.getElementById('content');
  const refreshState = document.getElementById('refresh-state');
  const refreshButton = document.getElementById('refresh');
  let data = null, error = '', busy = false, selectedRecording = '', hiddenRecording = false;
  let detail = null, detailId = '', detailError = '', detailLoading = false;
  let selectedRun = '', recordingRun = '', recordingTask = '', recordingPage = 0, lastRenderKey = '';
  const RECORDINGS_PER_PAGE = 8;
  const players = new Map();
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const finite = value => value !== null && value !== undefined && Number.isFinite(Number(value));
  const num = value => finite(value) ? new Intl.NumberFormat('ru-RU', {maximumFractionDigits: 2}).format(Number(value)) : '—';
  const nll = value => finite(value) ? new Intl.NumberFormat('ru-RU', {maximumSignificantDigits: 3}).format(Number(value)) : '—';
  const pct = value => finite(value) ? `${num(value)}%` : 'Нет проверки';
  const date = value => { if (!value) return '—'; const d = new Date(value); return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString('ru-RU', {day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'}); };
  const duration = value => { if (!finite(value)) return '—'; const s = Math.max(0, Math.round(value)); return s >= 3600 ? `${Math.floor(s/3600)} ч ${Math.floor(s%3600/60)} мин` : `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`; };
  const route = () => { const parts = location.hash.replace(/^#\/?/, '').split('/'); return {page:parts[0] || 'overview', id:decodeURIComponent(parts.slice(1).join('/'))}; };
  const change = value => finite(value) ? `<span class="change ${Number(value)<0?'negative':''}">${Number(value)>0?'+':''}${num(value)} п.п.</span>` : '<span class="muted">Нет исходной проверки</span>';
  const empty = message => `<p class="muted">${esc(message)}</p>`;
  const counts = item => finite(item?.successes) && finite(item?.episodes) ? `${num(item.successes)} побед из ${num(item.episodes)} эпизодов` : 'Число игр пока неизвестно';
  const outcome = value => {
    if(!value)return '';
    if(typeof value==='string')return ({success:'Победа',failure:'Поражение',win:'Победа',loss:'Поражение'})[value] || value;
    if(value.success)return 'Победа';
    if(value.failure)return 'Поражение';
    return value.known===false?'Исход не распознан':'Не завершено';
  };
  const kind = value => ({imitation:'Повторение действий',imitation_bc:'Повторение действий',imitation_training:'Повторение действий',sequence_training:'Обучение последовательностям',synthetic_compute_probe:'Проверка производительности',training:'Обучение',evaluation:'Проверка',rl:'Обучение с наградой',ppo:'Обучение с наградой',playback:'Запись игры'})[value] || 'Эксперимент';
  const runLabel = item => item?.experiment?.title || (item?.label && item.label!==item.run_id ? item.label : `${kind(item?.kind || 'imitation')}${(item?.epochs ?? item?.epochs_total)?` · ${num(item.epochs ?? item.epochs_total)} эпох`:''}${item?.started_at?` · ${date(item.started_at)}`:''}`);
  function pickerLabel(item) {
    const e=item?.experiment || {};
    if(e.picker_label)return e.picker_label;
    const measurement=e.measurement || item?.quality;
    const result=finite(measurement?.successes)&&finite(measurement?.episodes)?`${num(measurement.successes)}/${num(measurement.episodes)} побед`:'без итоговой проверки';
    const verdict=e.verdict?.state;
    const suffix=verdict==='worse'?' · хуже исходной':verdict==='improved'?' · улучшилась':verdict==='unchanged'?' · без изменения':'';
    return `${item?.started_at?date(item.started_at):'Дата не указана'} · ${result}${suffix}`;
  }
  function verdictSummary(e,currentId=data?.current?.run_id) {
    const verdict=e?.verdict;
    if(!verdict)return '';
    const state=typeof verdict==='object'?verdict.state:'';
    const text=typeof verdict==='string'?verdict:verdict.label || '';
    const exact=finite(verdict?.successes)&&finite(verdict?.baseline_successes)&&finite(verdict?.episodes)&&['worse','improved','unchanged'].includes(state);
    const message=exact?`${state==='worse'?'Новая модель хуже':state==='improved'?'Новая модель улучшилась':'Результат не изменился'}: ${num(verdict.successes)}/${num(verdict.episodes)} побед; исходная — ${num(verdict.baseline_successes)}/${num(verdict.episodes)}.`:text;
    if(!message)return '';
    const recommended=e.recommended_run_id!==currentId?e.recommended_run_id:null;
    return `<div class="model-verdict ${esc(state)}"><p>${esc(message)}</p>${e.recommendation?`<p class="verdict-recommendation">${esc(e.recommendation)}</p>`:''}${recommended?`<button type="button" class="model-action" data-select-run="${esc(recommended)}">${state==='worse'?'Посмотреть исходную модель':'Посмотреть рекомендованную модель'}</button>`:''}</div>`;
  }
  function experimentTitle(id,c=data?.current) {
    const run=c?.run_id===id?c:(data?.experiments || []).find(r=>r.run_id===id);
    return run?pickerLabel(run):'Предыдущая модель';
  }
  function technicalDetails(id, entries) {
    return `<details id="${esc(id)}" class="technical-details"><summary>Технические сведения</summary><dl class="settings">${entries.filter(([,value])=>value!==null&&value!==undefined&&value!=='').map(([label,value])=>`<dt>${esc(label)}</dt><dd>${esc(value)}</dd>`).join('')}</dl></details>`;
  }
  function qualityStage(q,c=data?.current) {
    if(q?.measurement_stage)return q.measurement_stage==='final'&&q.current_weights_measured===false?'previous':q.measurement_stage;
    if(q?.inherited_from&&!q.canonical_measurement_current)return 'inherited';
    const own=(q?.run_history || q?.history || []).filter(p=>p.run_id===c?.run_id);
    return own.length&&Number(own.at(-1).epoch)===0?'initial':'final';
  }
  function technicalMessages(warnings) {
    return warnings.length?`<details id="technical-messages" class="technical-messages"><summary>Технические сообщения (${warnings.length})</summary><ul>${warnings.map(w=>`<li>${esc(w)}</li>`).join('')}</ul></details>`:'';
  }
  function experimentSelector() {
    const runs=data.experiments || [];
    if(!runs.length)return '';
    const selected=data.current || runs.find(r=>r.run_id===selectedRun);
    const active=data.training_current || data.current;
    const defaultSelected=!selectedRun || selectedRun===active?.run_id;
    const choices=runs.filter(r=>(r.run_id || r.id)!==active?.run_id);
    return `<div class="experiment-picker"><p class="selected-model">Выбранная модель: <strong>${esc(pickerLabel(selected))}</strong></p><label for="experiment-run">Выбрать модель</label><select id="experiment-run"><option value="" ${defaultSelected?'selected':''}>${esc(pickerLabel(active))}</option>${choices.map(r=>`<option value="${esc(r.run_id || r.id)}" ${(r.run_id || r.id)===selectedRun?'selected':''}>${esc(pickerLabel(r))}</option>`).join('')}</select><a href="#/history">Все эксперименты</a></div>`;
  }
  function experimentSummary(c, experiment=c?.experiment) {
    if (!c) return '';
    const e=experiment || {};
    const purpose=e.purpose || c.method_description || 'Проверяем, как обучение меняет игру агента на контрольных уровнях.';
    const conclusion=e.conclusion || (c.running?'Эксперимент продолжается; окончательный результат ещё не получен.':'Вывод не сохранён. Ниже показаны доступные проверки этой модели.');
    const limitations=Array.isArray(e.limitations)?e.limitations:e.limitations?[e.limitations]:[];
    return `<section class="experiment-summary" aria-label="Выбранный эксперимент"><div class="run-heading"><h1>${esc(runLabel({...c,experiment:e}))}</h1>${status(c)}</div>${verdictSummary(e,c.run_id || c.id)}<p class="experiment-purpose">${esc(purpose)}</p><p class="experiment-conclusion"><strong>Результат.</strong> ${esc(conclusion)}</p>${limitations.length?`<p class="experiment-limits"><strong>Ограничения.</strong> ${limitations.map(esc).join(' ')}</p>`:''}<p class="note">${c.started_at?`Начат ${date(c.started_at)}.`:''}${e.baseline_run_id?` Сравнение с <a href="#/history/${encodeURIComponent(e.baseline_run_id)}">${esc(experimentTitle(e.baseline_run_id))}</a>.`:''}</p>${technicalDetails('experiment-identity',[['Запуск',c.run_id || c.id],['Сохранённая модель',e.checkpoint]])}</section>`;
  }
  function metricIdentity(q,c=data.current) {
    const stage=qualityStage(q,c);
    const inherited=stage==='inherited';
    const runId=q?.measurement_run_id || (inherited?q?.inherited_from?.run_id || q?.latest_run_id:q?.latest_run_id || c?.run_id);
    return `<div class="metric-identity"><strong>${stage==='initial'?'Модель до этого обучения':inherited?'Замер предыдущей модели':stage==='previous'?'Последняя проверка сохранённой модели':'Проверка выбранного эксперимента'}</strong><br>${esc(experimentTitle(runId,c))}<br>Проверено ${date(q?.evaluated_at || q?.inherited_from?.evaluated_at)}. ${esc(q?.cohort_description || 'Контрольные уровни; условия указаны ниже.')}${technicalDetails('metric-identity-details',[['Запуск',runId],['Версия модели',q?.latest_revision ?? q?.inherited_from?.revision],['Файл проверки',q?.measurement_source || q?.latest_evaluation_file]])}</div>`;
  }
  function status(item) {
    const s = item?.status || '';
    const labels={completed:'Обучение завершено',running:'Обучение идёт',failed:'Остановлено с ошибкой',error:'Ошибка',interrupted:'Прервано',stopped:'Остановлено',stale:'Данные устарели'};
    return `<span class="status ${item?.running?'running':/fail|error|interrupt/.test(s)?'failed':''}"><span class="dot" aria-hidden="true"></span>${esc(labels[s] || item?.status_label || 'Состояние неизвестно')}</span>`;
  }
  function eta(c) {
    if (c.status === 'completed') return '0:00';
    if (finite(c.eta_low_seconds) && finite(c.eta_high_seconds)) return `${duration(c.eta_low_seconds)}–${duration(c.eta_high_seconds)}`;
    if (finite(c.eta_seconds)) return duration(c.eta_seconds);
    return c.running ? 'Оцениваем темп' : 'Нет оценки';
  }
  function progress(c) {
    if (!c) return `<section class="panel">${empty('Запуск обучения ещё не найден.')}</section>`;
    if(c.method==='ppo')return ppoProgress(c);
    const phases = c.phase_steps?.length ? c.phase_steps : ['Подготовка','Исходная проверка','Обучение','Итоговая проверка','Сохранение'].map((label,i) => ({label,state:c.status==='completed'||i+1<c.phase_index?'done':i+1===c.phase_index?'active':'pending'}));
    const percent = finite(c.training_percent) ? Math.max(0,Math.min(100,Number(c.training_percent))) : null;
    return `<section class="panel ${c.status==='completed'?'completed-progress':''}" aria-label="Состояние обучения"><div class="run-heading"><div><h2>${esc(c.status==='completed'?'Обучение завершено':c.phase_label || 'Обучение')}</h2><div class="run-label">${esc(runLabel(c))} · Этап ${num(c.phase_index)} из ${num(c.phase_total || 5)}</div></div>${c.status==='completed'?'':status(c)}</div>
      <div class="phase-track">${phases.map(p=>`<div class="phase ${esc(p.state)}">${esc(p.label)}</div>`).join('')}</div>
      <div class="progress-label"><span>Обновления модели</span><span>${num(c.optimizer_steps)} / ${num(c.optimizer_steps_total)}${percent===null?'':` · ${num(percent)}%`}</span></div><progress aria-label="Прогресс обновлений модели" ${percent===null?'':`value="${percent}"`} max="100"></progress>
      <div class="facts"><div class="fact"><span>Эпоха</span><strong>${num(c.epoch)} / ${num(c.epochs_total)}</strong>${c.status==='completed'?'':`<small>${num(c.epochs_completed)} завершено</small>`}</div><div class="fact"><span>Пакет в эпохе</span><strong>${num(c.batch)} / ${num(c.batches_per_epoch)}</strong></div><div class="fact"><span>Время запуска</span><strong>${duration(c.elapsed_seconds)}</strong>${finite(c.phase_elapsed_seconds)&&c.status!=='completed'?`<small>Текущий этап: ${duration(c.phase_elapsed_seconds)}</small>`:''}</div><div class="fact"><span>Осталось</span><strong>${eta(c)}</strong>${c.status==='completed'?'':`<small>${esc(c.eta_scope==='training'?'Только обучение; без итоговой проверки':c.eta_scope==='phase'?'До конца текущей фазы':c.eta_scope==='run'?'Весь запуск':c.eta_reason || '')}</small>`}</div></div>
      ${c.phase_units_total&&c.status!=='completed'?`<p class="note">Текущий этап: ${num(c.phase_units_done)} из ${num(c.phase_units_total)}</p>`:''}</section>`;
  }
  function ppoProgress(c) {
    const percent=finite(c.rollouts)&&finite(c.rollouts_total)&&Number(c.rollouts_total)>0?Math.max(0,Math.min(100,Number(c.rollouts)/Number(c.rollouts_total)*100)):null;
    const cycleEta=c.stop_reason?'Циклы остановлены':finite(c.cycle_eta_seconds)?duration(c.cycle_eta_seconds):'Нет измеренного темпа';
    return `<section class="panel ${c.status==='completed'?'completed-progress':''}" aria-label="Состояние обучения"><div class="run-heading"><div><h2>${esc(c.status==='completed'?'Обучение завершено':c.status==='resource_stopped'?'Запуск остановлен по лимиту ресурсов':c.phase_label || 'Обучение с наградой')}</h2><p class="run-label">${esc(c.method_description || 'Агент учится по наградам за игровые действия.')}</p></div>${c.status==='completed'?'':status(c)}</div>${c.phase_steps?.length?`<div class="phase-track" style="grid-template-columns:repeat(${c.phase_steps.length},minmax(0,1fr))">${c.phase_steps.map(p=>`<div class="phase ${esc(p.state)}">${esc(p.label)}</div>`).join('')}</div>`:''}<div class="progress-label"><span>Завершённые циклы обучения</span><span>${num(c.rollouts)} / ${num(c.rollouts_total)}${percent===null?'':` · ${num(percent)}%`}</span></div><progress aria-label="Прогресс циклов обучения" ${percent===null?'':`value="${percent}"`} max="100"></progress><div class="facts completed-facts"><div class="fact"><span>Принятые обновления в этом запуске</span><strong>${num(c.optimizer_steps)}</strong>${finite(c.optimizer_steps_upper_bound)?`<small>Верхний предел: ${num(c.optimizer_steps_upper_bound)}; KL может уменьшить число</small>`:''}</div><div class="fact"><span>Действий в этом запуске</span><strong>${num(c.attempted_train_decisions_this_process)}</strong><small>Совокупный счётчик: ${num(c.decisions)}; включает прошлые этапы</small></div><div class="fact"><span>Время запуска</span><strong>${duration(c.elapsed_seconds)}</strong></div><div class="fact"><span>Осталось циклов по измеренному темпу</span><strong>${cycleEta}</strong><small>Итоговая проверка отдельно</small></div></div>${c.phase_units_total&&c.status!=='completed'?`<p class="note">Текущий этап: ${num(c.phase_units_done)} из ${num(c.phase_units_total)}. ${finite(c.eta_seconds)?`До конца этапа: ${duration(c.eta_seconds)}.`:''}</p>`:''}${c.stop_reason?`<p class="note">${esc(c.stop_reason==='minimum_free_disk_mib'?'Достигнут заданный минимум свободного места.':c.stop_reason==='max_wall_seconds'?'Достигнут лимит времени запуска.':c.stop_reason)}</p>`:''}${technicalDetails('training-revision',[['Совокупная ревизия модели',c.weights_revision]])}</section>`;
  }
  function chart(history, field, label, fixed = false) {
    const points = (history || []).filter(p => finite(p[field]));
    if (!points.length) return empty('График появится после первой проверки.');
    const xValue = p => finite(p.updates)?Number(p.updates):finite(p.epoch)?Number(p.epoch):finite(p.rollout)?Number(p.rollout):0;
    const maxX = Math.max(1,...points.map(xValue));
    const minY = fixed?0:Math.min(...points.map(p=>Number(p[field])))*.9;
    const maxY = fixed?100:Math.max(minY+.001,...points.map(p=>Number(p[field])))*1.08;
    const x = p => 48 + xValue(p)/maxX*492, y = value => 192-(value-minY)/(maxY-minY)*156;
    const ticks = [minY,(minY+maxY)/2,maxY];
    const metric=field==='nll'?nll:num;
    return `<div class="chart"><svg viewBox="0 0 565 230" role="img" aria-label="${esc(label)}"><title>${esc(label)}. ${points.length} точек. Последнее значение ${metric(points.at(-1)[field])}${fixed?' процентов':''}.</title>${ticks.map(v=>`<line class="axis" x1="48" x2="540" y1="${y(v)}" y2="${y(v)}"/><text x="39" y="${y(v)+4}" text-anchor="end">${metric(v)}${fixed?'%':''}</text>`).join('')}<polyline class="line" points="${points.map(p=>`${x(p)},${y(Number(p[field]))}`).join(' ')}"/>${points.map((p,i)=>`<circle id="point-${field}-${i}" cx="${x(p)}" cy="${y(Number(p[field]))}" r="4" tabindex="0" role="img" aria-label="${esc(label)}: ${metric(p[field])}${fixed?'%':''}, ${num(xValue(p))} ${finite(p.updates)?'обновлений':'эпох'}"><title>${metric(p[field])}${fixed?'%':''} · ${num(xValue(p))} ${finite(p.updates)?'обновлений':'эпох'}</title></circle>`).join('')}<text x="48" y="213">0</text><text x="540" y="213" text-anchor="end">${num(maxX)}</text></svg><div class="chart-title">${points.some(p=>finite(p.updates))?'Обновления модели':'Эпохи'}</div></div><details id="chart-${field}"><summary>Значения графика</summary><div class="table-wrap"><table><thead><tr><th>Эпоха</th><th>Обновления</th><th>${esc(label)}</th></tr></thead><tbody>${points.map(p=>`<tr><td>${num(p.epoch)}</td><td>${num(p.updates)}</td><td>${metric(p[field])}${fixed?'%':''}</td></tr>`).join('')}</tbody></table></div></details>`;
  }
  function diagnostics(d,c=data.current) {
    const t=d?.transfer,m=d?.memory;
    const identity=r=>`<p class="note diagnostic-identity">${esc(r.current_weights?'Проверены сохранённые веса выбранного эксперимента':'Проверена предыдущая модель')}: ${esc(experimentTitle(r.run_id,c))}. Проверено ${date(r.evaluated_at)}.</p>${technicalDetails(`diagnostic-${r===t?'appearance':'memory'}-identity`,[['Запуск',r.run_id],['Версия модели',r.revision],['Сохранённая модель',r.checkpoint]])}`;
    const previous=(r,body,key)=>r&&!r.current_weights?`<p class="note">Текущая модель ещё не проверена.</p><details id="previous-${key}"><summary>Проверка предыдущей модели: ${esc(experimentTitle(r.run_id,c))}</summary>${body}</details>`:body;
    const profile=p=>p==='canonical96'?'Обычное оформление · поле 96×96':p.startsWith('varied96')?'Изменённое оформление · поле 96×96':p.startsWith('varied64')?'Новое оформление · поле 64×64':p.startsWith('varied128')?'Новое оформление · поле 128×128':p;
    const appearance=t?`${identity(t)}<div class="table-wrap"><table><thead><tr><th>Условия</th><th>Язык</th><th>Победы</th><th>Эпизоды</th><th>Изменение</th></tr></thead><tbody>${t.rows.map(r=>`<tr><td>${esc(profile(r.profile))}</td><td>${esc(r.language.toUpperCase())}</td><td>${pct(r.value)}${r.unknown_outcomes?`–${pct(r.upper)}`:''}</td><td>${num(r.successes)} / ${num(r.episodes)}</td><td>${finite(r.change_pp)?change(r.change_pp):'—'}</td></tr>`).join('')}</tbody></table></div><p class="note">${t.coverage_mode==='full_bank'?'Полный банк.':'Диагностическая выборка уровней.'} RU/EN — описание игры на русском/английском. Победы не доказывают понимание инструкции; размеры поля могут менять механику.</p>${d.comparison_reason?technicalDetails('appearance-comparison',[['Условия сравнения',d.comparison_reason]]):''}`:empty('Текущая модель ещё не проверена на другом оформлении.');
    const memory=m?`${identity(m)}<div class="table-wrap"><table><thead><tr><th>Режим</th><th>Первый выбор</th><th>Оба варианта пары</th></tr></thead><tbody>${m.modes.map(r=>`<tr><td>${esc(r.mode==='normal'?'Обычный':'Нейтральный Wait')}</td><td>${num(r.first_choose_correct)} / ${num(r.first_choose_observed)}</td><td>${num(r.paired_both_first_choose_correct)} / ${num(r.geometries)}</td></tr>`).join('')}</tbody></table></div><p class="note">Первое направление и пары уровней, где агент правильно реагирует на обе противоположные подсказки.</p>`:empty('Текущая модель ещё не проверена на память.');
    return `<div class="grid diagnostic-summary"><section class="panel"><h2>Проверка на другом оформлении</h2><p class="diagnostic-explanation">Та же модель играет с другими цветами, расположением элементов, размером поля или языком описания. Проверяем, сохраняется ли результат при изменении условий.</p>${previous(t,appearance,'appearance')}</section><section class="panel"><h2>Память: первый выбор и пары подсказок</h2>${previous(m,memory,'memory')}</section></div>`;
  }
  function ppoDiagnostics(q) {
    return (q?.diagnostic_series || []).map((series,i)=>`<details id="ppo-diagnostic-${i}"><summary>Отдельная проверка PPO · ${series.coverage==='full'?'полный набор':'выборка'}</summary><p class="note diagnostic-identity">${esc(series.description)}</p>${technicalDetails(`ppo-protocol-${i}`,[['Протокол',series.protocol_sha256]])}<div class="table-wrap"><table><thead><tr><th>Цикл</th><th>Ревизия</th><th>Победы</th><th>Эпизоды</th></tr></thead><tbody>${series.history.map(p=>`<tr><td>${num(p.rollout)}</td><td>${num(p.revision)}</td><td>${pct(p.value)}</td><td>${counts(p)}</td></tr>`).join('')}</tbody></table></div></details>`).join('');
  }
  function perfectScore(q) {
    const measured=finite(q?.perfect_score_successes)&&finite(q?.perfect_score_percent);
    return `<div class="perfect-score"><h3>Победы с максимальным счётом${q?.perfect_score_measurement_stage==='initial'?' до этого обучения':''}</h3>${measured?`<p><strong>${pct(q.perfect_score_percent)}</strong> · ${num(q.perfect_score_successes)} из ${num(q.perfect_score_episodes ?? q.episodes)} эпизодов с победой и счётом 5/5.</p>${q.perfect_score_run_id?`<p class="note">${esc(experimentTitle(q.perfect_score_measurement_run_id || q.perfect_score_run_id))}. ${esc(q.perfect_score_scope || 'Собственная проверка выбранного эксперимента')}. Условия этой проверки могут отличаться от основной доли побед.</p>${technicalDetails('perfect-score-identity',[['Запуск',q.perfect_score_measurement_run_id || q.perfect_score_run_id],['Версия модели',q.perfect_score_revision],['Файл проверки',q.perfect_score_measurement_source || q.perfect_score_evaluation_file],['Версия правил счёта',q.scoring_version]])}`:''}`:empty('Отдельное измерение побед со счётом 5/5 пока недоступно.')}<p class="note">${esc(q?.scoring_version?'Счёт учитывает управляющие действия агента.':q?.scoring_version_label || (finite(q?.episodes)?'Прежние правила счёта: результаты 5/5 нельзя сравнивать с текущими.':'Правила счёта и результат появятся после проверки.'))}</p>${q?.scoring_version?'<p class="note">В лабиринте 5/5 означает кратчайший маршрут без лишних нажатий. В остальных 15 играх — победу без лишних управляющих действий; это не доказывает, что весь путь был оптимальным.</p>':''}</div>`;
  }
  function quality(q, full=true, c=data.current) {
    if (!q?.available && !finite(q?.value)) return `<section class="panel"><h2>Результат после обучения</h2>${empty('Проверка качества ещё не завершена. Ошибка повторения действий не доказывает, что агент научился побеждать.')}${perfectScore(q)}${ppoDiagnostics(q)}</section>`;
    const stage=qualityStage(q,c);
    const title=stage==='initial'?'До этого обучения':stage==='inherited'?'Проверка предыдущей модели':stage==='previous'?'Последняя проверка сохранённой модели':'Победы на 16 мини-играх';
    const pending=stage!=='final'?`<p class="note result-pending">${stage==='previous'?'Текущие веса ещё не проверены. Показан результат ранее сохранённой модели.':'Результат после этого обучения ещё не получен. Показанный замер не подтверждает улучшение в выбранном эксперименте.'}</p>`:'';
    return `<section class="panel"><div class="panel-top"><h2>${title}</h2><a href="#/games">Результаты по играм</a></div>${pending}<div class="quality-summary"><strong class="quality-value">${pct(q.value)}</strong>${stage==='final'?change(q.change_pp):''}</div>${metricIdentity(q,c)}<p class="note">${counts(q)}.${stage==='final'?` Исходный результат: ${pct(q.baseline)}.`:''}</p>${perfectScore(q)}${full&&stage==='final'?chart(q.run_history || q.history,'value','Доля побед',true):''}<p class="note">${esc(q.protocol_description || 'Фиксированный контрольный набор: 16 мини-игр по 16 эпизодов, начальная сложность. Результат относится к этим условиям.')}</p>${q.inherited_from?`<p class="note">${esc(q.canonical_measurement_current?'Исходная модель для сравнения':'Показана предыдущая модель')}: ${esc(experimentTitle(q.inherited_from.run_id,c))}. Это прежний замер.${stage!=='final'?' Текущая модель после обучения ещё не проверена.':''}</p>`:''}${ppoDiagnostics(q)}${q.quick?`<p class="note">Промежуточная проверка: ${pct(q.quick.value)}, ${counts(q.quick)}. Она не заменяет полную проверку.</p>`:''}</section>`;
  }
  function recordingControls(source, task, filtered, selected) {
    const all=source.playbacks?.items || [];
    const runIds=[...new Set(all.map(p=>p.run_id).filter(Boolean))];
    const selectedId=recordingRun || source.current?.run_id || source.run_id || runIds[0] || '';
    const runItems=all.filter(p=>p.run_id===selectedId);
    const tasks=[...new Set(runItems.map(p=>p.task))];
    const totalPages=Math.max(1,Math.ceil(filtered.length/RECORDINGS_PER_PAGE));
    recordingPage=Math.min(recordingPage,totalPages-1);
    const items=filtered.slice(recordingPage*RECORDINGS_PER_PAGE,(recordingPage+1)*RECORDINGS_PER_PAGE);
    const runTitle=id=>pickerLabel(source.current?.run_id===id?source.current:(data.experiments || []).find(r=>r.run_id===id));
    return `<p class="selected-model recording-model">Модель в записи: <strong>${esc(runTitle(selectedId))}</strong></p><div class="recording-filters"><div class="filter-field"><label for="recording-run">Модель / эксперимент</label><select id="recording-run">${runIds.map(id=>`<option value="${esc(id)}" ${id===selectedId?'selected':''}>${esc(runTitle(id))}</option>`).join('')}${!runIds.includes(selectedId)?`<option value="${esc(selectedId)}" selected>Выбранный эксперимент — записей пока нет</option>`:''}</select></div><div class="filter-field"><label for="recording-task">Игра</label><select id="recording-task" ${task?'disabled':''}>${tasks.map(id=>`<option value="${esc(id)}" ${id===(task || recordingTask)?'selected':''}>${esc(source.task_guides?.[id]?.label || all.find(p=>p.task===id)?.label || id)}</option>`).join('')}</select></div><div class="filter-field recording-choice"><label for="recording">Запись: сначала новые</label><select id="recording" ${!items.length?'disabled':''}>${items.length?items.map(p=>`<option value="${esc(p.id)}" ${p.id===selected?.id?'selected':''}>${date(p.created_at)} · ${esc(outcome(p.outcome) || 'без результата')}</option>`).join(''):'<option>Записей пока нет</option>'}</select></div></div>${totalPages>1?`<div class="recording-pages"><button id="recording-prev" type="button" ${recordingPage===0?'disabled':''}>Более новые</button><span>Записи ${recordingPage*RECORDINGS_PER_PAGE+1}–${Math.min(filtered.length,(recordingPage+1)*RECORDINGS_PER_PAGE)} из ${num(filtered.length)}</span><button id="recording-next" type="button" ${recordingPage===totalPages-1?'disabled':''}>Более ранние</button></div>`:''}`;
  }
  function playback(task='', source=data) {
    const all=source.playbacks?.items || [];
    const runId=recordingRun || source.current?.run_id || source.run_id || all[0]?.run_id;
    const runItems=all.filter(p=>p.run_id===runId);
    const tasks=[...new Set(runItems.map(p=>p.task))];
    if(!task&&!tasks.includes(recordingTask))recordingTask=tasks[0] || '';
    const items=runItems.filter(p=>p.task===(task || recordingTask)).sort((a,b)=>String(b.created_at || '').localeCompare(String(a.created_at || '')) || Number(b.revision || 0)-Number(a.revision || 0));
    recordingPage=Math.min(recordingPage,Math.max(0,Math.ceil(items.length/RECORDINGS_PER_PAGE)-1));
    const existing=items.findIndex(p=>p.id===selectedRecording);
    if(existing>=0)recordingPage=Math.floor(existing/RECORDINGS_PER_PAGE);
    const page=items.slice(recordingPage*RECORDINGS_PER_PAGE,(recordingPage+1)*RECORDINGS_PER_PAGE);
    const selected=page.find(p=>p.id===selectedRecording) || page[0];
    const controls=recordingControls(source,task,items,selected);
    if(!selected)return `<section class="panel recording-panel"><h2>Запись игры</h2>${controls}<div class="preview empty">Для выбранной модели и игры записей пока нет.</div><p class="note">Выберите другой эксперимент или игру.</p></section>`;
    selectedRecording=selected.id;
    const safeUrl=typeof selected.url==='string'&&/^\/media\//.test(selected.url)?selected.url:'';
    const sprite=typeof selected.sprite_url==='string'&&/^\/media\//.test(selected.sprite_url)&&finite(selected.frame_count)&&selected.frame_count>0&&selected.frame_width>0&&selected.frame_height>0;
    const meta=`<div class="playback-meta"><span>${esc(experimentTitle(selected.run_id,source.current))}</span><span>Записано ${date(selected.created_at)}</span><span>${finite(selected.steps)?`${num(selected.steps)} действий`:''}</span><strong>${esc(outcome(selected.outcome))}</strong></div>${technicalDetails('recording-identity',[['Запуск',selected.run_id],['Версия модели',selected.revision],['Эпоха',selected.epoch],['Сохранённая модель',selected.checkpoint],['Запись',selected.id]])}<p class="note">Одна запись показывает один эпизод. Доля побед выше относится к выбранному эксперименту и всей его проверке.</p>`;
    const heading=`<div class="panel-top"><h2>Запись игры</h2><a href="#/games/${encodeURIComponent(selected.task)}">${esc(selected.label || selected.task)}</a></div>${controls}`;
    if(sprite)return spritePlayback(selected,safeUrl,heading,meta);
    return `<section class="panel recording-panel">${heading}<div class="preview ${!safeUrl||hiddenRecording?'empty':''}">${safeUrl&&!hiddenRecording?`<img data-recording="${esc(selected.id)}" src="${esc(safeUrl)}" alt="Запись игры ${esc(selected.label || selected.task)}" loading="eager">`:esc(hiddenRecording?'Запись скрыта. Нажмите «Показать запись», чтобы продолжить просмотр.':'Файл записи недоступен.')}</div><div class="controls"><button id="toggle-recording" type="button">${hiddenRecording?'Показать запись':'Скрыть запись'}</button>${safeUrl?`<a href="${esc(safeUrl)}" target="_blank" rel="noopener">Открыть GIF ↗</a>`:''}</div><p class="record-label">GIF воспроизводится автоматически.</p>${meta}</section>`;
  }
  function spritePlayback(selected,gifUrl,heading,meta) {
    let player=players.get(selected.id);
    if(!player){player={id:selected.id,item:selected,frame:0,playing:true,last:0,image:null,error:false};players.set(selected.id,player);}
    player.item=selected;
    return `<section class="panel recording-panel">${heading}<div class="preview"><canvas id="game-canvas" data-recording="${esc(selected.id)}" width="${Number(selected.frame_width)}" height="${Number(selected.frame_height)}" aria-label="Запись игры ${esc(selected.label || selected.task)}">Запись игры. ${gifUrl?`<a href="${esc(gifUrl)}">Открыть GIF</a>`:''}</canvas></div><div class="player-controls"><button id="play-pause" type="button">${player.playing?'Пауза':'Продолжить'}</button><button id="restart-playback" type="button">С начала</button><label class="sr-only" for="frame-seek">Кадр записи</label><input id="frame-seek" type="range" min="0" max="${selected.frame_count-1}" value="${player.frame}" step="1"><output id="frame-counter" for="frame-seek">${player.frame+1} / ${num(selected.frame_count)}</output></div><p id="action-caption" class="note"></p>${meta}${gifUrl?`<p class="note"><a href="${esc(gifUrl)}" target="_blank" rel="noopener">Открыть GIF ↗</a></p>`:''}</section>`;
  }
  function drawPlayer(player) {
    const canvas=document.getElementById('game-canvas');
    if(!canvas || canvas.dataset.recording!==player.id)return;
    const item=player.item;
    if(player.image?.complete && player.image.naturalWidth){
      const context=canvas.getContext('2d');context.imageSmoothingEnabled=false;
      const columns=Number(item.sprite_columns)||8;
      context.drawImage(player.image,player.frame%columns*item.frame_width,Math.floor(player.frame/columns)*item.frame_height,item.frame_width,item.frame_height,0,0,item.frame_width,item.frame_height);
    }
    const seek=document.getElementById('frame-seek');if(seek)seek.value=player.frame;
    const counter=document.getElementById('frame-counter');if(counter)counter.textContent=`${player.frame+1} / ${num(item.frame_count)}`;
    const action=document.getElementById('action-caption');
    if(action){
      const a=item.actions?.[player.frame];
      const names={left:'Влево',right:'Вправо',up:'Вверх',down:'Вниз',space:'Прыжок',jump:'Прыжок',fire:'Огонь',primary:'Основное действие',secondary:'Дополнительное действие'};
      let text='';
      if(a&&typeof a==='object'&&Array.isArray(a.keys)){
        const inputs=a.keys.map(key=>names[key.toLowerCase()] || key);
        (a.mouse_buttons || []).forEach(button=>inputs.push(`Кнопка мыши: ${{left:'левая',right:'правая',middle:'средняя'}[button] || button}`));
        if(a.dx || a.dy)inputs.push(`Мышь: ${num(a.dx)}, ${num(a.dy)}`);
        if(a.wheel)inputs.push(`Колесо: ${num(a.wheel)}`);
        text=inputs.join(', ');
      }else text=Array.isArray(a)?a.join(', '):typeof a==='string'?a:'';
      action.textContent=player.frame===item.frame_count-1?'Последний кадр записи':a?`Действие агента: ${text || 'без нажатий'}`:'Действие для этого кадра не записано';
    }
  }
  function initializePlayer() {
    const canvas=document.getElementById('game-canvas');if(!canvas)return;
    const player=players.get(canvas.dataset.recording);if(!player)return;
    if(player.error){const host=canvas.closest('.preview');host.classList.add('empty');host.textContent='Не удалось загрузить кадры записи. Откройте GIF по ссылке ниже.';return;}
    if(!player.image){
      player.image=new Image();player.image.onload=()=>drawPlayer(player);
      player.image.onerror=()=>{player.error=true;const current=document.getElementById('game-canvas');if(current?.dataset.recording===player.id){const host=current.closest('.preview');host.classList.add('empty');host.textContent='Не удалось загрузить кадры записи. Откройте GIF по ссылке ниже.';}};
      player.image.src=player.item.sprite_url;
    }
    drawPlayer(player);
  }
  function animatePlayer(time) {
    const canvas=document.getElementById('game-canvas');const p=canvas&&players.get(canvas.dataset.recording);
    if(p?.playing && p.image?.complete && p.image.naturalWidth){
      const delay=p.frame===p.item.frame_count-1?(p.item.terminal_hold_ms || 1200):(p.item.frame_duration_ms || 100);
      if(!p.last)p.last=time;
      if(time-p.last>=delay){p.frame=(p.frame+1)%p.item.frame_count;p.last=time;drawPlayer(p);}
    }
    requestAnimationFrame(animatePlayer);
  }
  function loss(c) {
    return `<section class="panel"><h2>Ошибка повторения действий</h2><div class="quality-summary"><strong class="quality-value">${nll(c?.nll)}</strong></div><p class="note">Меньше — агент точнее повторяет действия из обучающих примеров. Качество игры оценивается отдельно по победам.</p>${chart(c?.loss_history,'nll','Ошибка повторения действий')}<p class="note">${esc(c?.method_description || c?.method || '')}</p></section>`;
  }
  function settings(c) {
    const entries = Object.entries(c?.settings || {});
    return `<section class="panel"><h2>Параметры обучения</h2><p class="muted">${esc(c?.method_description || c?.method || 'Метод не указан.')}</p><details id="technical-settings"><summary>Технические настройки</summary>${entries.length?`<dl class="settings">${entries.map(([key,value])=>`<dt>${esc(key)}</dt><dd>${esc(typeof value==='object'?JSON.stringify(value):value)}</dd>`).join('')}</dl>`:empty('Настройки запуска не сохранены.')}</details></section>`;
  }
  function trainingStatus() {
    const active=data.training_current;
    if(active?.running&&active.run_id!==data.current?.run_id)return `<aside class="active-training"><h2>Сейчас обучается: ${esc(pickerLabel(active))}</h2><p class="note">Результаты выбранной модели ниже относятся к отдельной проверке.</p>${progress(active)}</aside>`;
    return progress(data.current);
  }
  function overview() {
    return `${experimentSelector()}${experimentSummary(data.current)}${trainingStatus()}<div class="grid">${quality(data.quality)}${playback()}</div>${diagnostics(data.diagnostics)}<div class="grid"><section class="panel"><h2>Что показывают результаты</h2><p>Победы показывают, справляется ли агент с задачами в проверке. Сравнение с исходным агентом помогает увидеть изменение после обучения.</p><p class="note">Даже высокий общий результат не означает, что агент освоил каждую игру. Откройте результаты по играм, чтобы увидеть слабые места.</p><a href="#/games">Все ${num(data.current?.games_count || data.quality?.games?.length)} игр</a></section><section class="panel"><h2>${data.current?.method==='ppo'?'Обучение по награде':'Повторение обучающих действий'}</h2>${data.current?.method==='ppo'?`<p>${esc(data.current.method_description || 'Агент меняет поведение по награде за игровые действия.')}</p>`:`<div class="quality-summary"><strong style="font-size:27px">${nll(data.current?.nll)}</strong></div><p class="note">Ошибка повторения действий. Меньше — точнее повторение; победы проверяются отдельно.</p>`}<p><a href="#/training">Подробнее об обучении</a></p></section></div>`;
  }
  function training() {
    const c=data.training_current?.running?data.training_current:data.current;
    const heading=c?.running?'Сейчас обучается':'Обучение';
    if(c?.method==='ppo')return `${experimentSelector()}<h1>${heading}</h1><p class="subtitle">Циклы обучения по награде и игровые действия агента.</p>${progress(c)}<div style="margin-top:22px">${settings(c)}</div>`;
    const perTask = Array.isArray(c?.per_task_nll)?c.per_task_nll:Object.entries(c?.per_task_nll || {}).map(([task,nll])=>({task,nll}));
    return `${experimentSelector()}<h1>${heading}</h1><p class="subtitle">Этапы запуска, скорость работы и точность повторения обучающих действий.</p>${progress(c)}<div class="grid">${loss(c)}${settings(c)}</div><section class="panel" style="margin-top:22px"><h2>Ошибка по играм</h2>${perTask.length?`<div class="table-wrap"><table><thead><tr><th>Игра</th><th>Ошибка повторения действий</th></tr></thead><tbody>${perTask.map(p=>`<tr><td><a href="#/games/${encodeURIComponent(p.task)}">${esc(data.task_guides?.[p.task]?.label || p.label || p.task)}</a></td><td>${nll(p.nll ?? p.value)}</td></tr>`).join('')}</tbody></table></div>`:empty('Значения по играм ещё не опубликованы.')}</section>`;
  }
  function games(task) {
    const results=data.quality?.games || [];
    const games=results.length?results:Object.entries(data.task_guides || {}).map(([task,guide])=>({task,label:guide.label}));
    if(!task)return `${experimentSelector()}<h1>Игры и записи</h1><p class="subtitle">Выберите игру, чтобы увидеть её результат, цель, управление и запись проверки. Здесь показаны все игры, включая те, где агент пока не побеждает.</p><div class="game-list">${games.map(g=>`<a class="game-link" href="#/games/${encodeURIComponent(g.task)}"><strong>${esc(g.label || data.task_guides?.[g.task]?.label || g.task)}</strong><span class="score">${pct(g.value)}</span>${qualityStage(data.quality)==='final'?change(g.change_pp):''}<br><small>${counts(g)}</small></a>`).join('')}</div>${!games.length?empty('Список игр пока недоступен. Нажмите «Обновить», чтобы повторить запрос.'):''}${diagnostics(data.diagnostics)}`;
    const game=games.find(g=>g.task===task);
    const guide=data.task_guides?.[task];
    const back='<a class="back" href="#/games">Все игры</a>';
    if(!game&&!guide)return `${back}<h1>Игра не найдена</h1><p class="subtitle">Выберите игру из общего списка.</p>`;
    const history=(data.quality?.history || []).filter(p=>finite(p.games?.[task]?.value)).map(p=>({...p,...p.games[task]}));
    return `${experimentSelector()}${back}<h1>${esc(game?.label || guide?.label || task)}</h1><p class="subtitle">Результат агента на фиксированном контрольном наборе и запись его игры.</p><div class="grid"><section class="panel"><h2>${qualityStage(data.quality)==='initial'?'До этого обучения':['inherited','previous'].includes(qualityStage(data.quality))?'Результат ранее сохранённой модели':'Результат проверки'}</h2>${metricIdentity(data.quality)}${finite(game?.value)?`<div class="quality-summary"><strong class="quality-value">${pct(game.value)}</strong>${qualityStage(data.quality)==='final'?change(game.change_pp):''}</div><p class="note">${counts(game)}.${qualityStage(data.quality)==='final'?` До обучения: ${pct(game.baseline)}.`:''}</p>${perfectScore({...game,scoring_version:game.scoring_version,scoring_version_label:game.scoring_version?data.quality?.scoring_version_label:'Прежняя версия счёта: сравнение недоступно'})}`:empty('Результат этой игры ещё не доступен.')}<div class="guide"><h3>Цель игры</h3><p>${esc(guide?.objective || 'Описание цели пока не добавлено.')}</p><h3>Действия агента</h3><p>${esc(Array.isArray(guide?.controls)?guide.controls.join(', '):guide?.controls || 'Описание управления пока не добавлено.')}</p></div><p class="note">${esc(data.quality?.protocol_description || 'Фиксированная проверка на сложности 0: по 16 эпизодов для каждой игры.')}</p></section>${playback(task)}</div><section class="panel" style="margin-top:22px"><h2>Как менялась доля побед</h2>${qualityStage(data.quality)==='final'?chart(history,'value','Доля побед',true):empty('Изменение после этого обучения ещё не измерено.')}</section>`;
  }
  function history() {
    const runs=data.experiments || [];
    return `<h1>История экспериментов</h1><p class="subtitle">Что проверяли, какая модель получилась и какие выводы подтверждены. Откройте эксперимент для его проверок и записей.</p><div class="experiment-history">${runs.length?runs.map(r=>{const e=r.experiment || {};const q=r.quality || {};return `<article class="panel experiment-card"><div class="run-heading"><h2><a class="history-name" href="#/history/${encodeURIComponent(r.run_id || r.id)}">${esc(runLabel(r))}</a></h2>${status(r)}</div><p>${esc(e.purpose || r.kind_label || kind(r.kind))}</p><p class="history-result"><strong>Результат:</strong> ${esc(e.conclusion || 'Вывод не сохранён; откройте доступные измерения.')}</p>${finite(q.value ?? r.value ?? r.quality_value ?? r.quality_percent ?? r.win_rate)?`<p class="note">Победы: <strong>${pct(q.value ?? r.value ?? r.quality_value ?? r.quality_percent ?? r.win_rate)}</strong>. ${esc(r.cohort_description || 'Условия проверки — в деталях эксперимента.')}</p>`:''}${e.limitations?.length?`<p class="note"><strong>Ограничения:</strong> ${(Array.isArray(e.limitations)?e.limitations:[e.limitations]).map(esc).join(' ')}</p>`:''}<div class="history-meta"><span>${date(r.started_at || r.created_at || r.updated_at)}</span><span>${duration(r.elapsed_seconds)}</span><span>${finite(r.epochs)?`${num(r.epochs)} эпох`:esc(r.kind_label || kind(r.kind))}</span></div><details id="run-id-${esc(r.run_id || r.id)}"><summary>Технические сведения</summary><code>${esc(r.run_id || r.id)}</code></details></article>`;}).join(''):empty('Сохранённых экспериментов пока нет.')}</div>`;
  }
  function runPage(id) {
    if (detailId!==id) { detailId=id; detail=null; detailError=''; detailLoading=true; fetchDetail(id); }
    const d = detail;
    return `<a class="back" href="#/history">Вернуться к истории запусков</a>${d?experimentSummary(d.current || d.run || d,d.experiment || d.current?.experiment):'<h1>Эксперимент</h1>'}${detailLoading?'<div class="loading" role="status">Загружаем запуск…</div>':''}${detailError?`<div class="error" role="alert">${esc(detailError)} <button id="retry-detail" type="button">Повторить</button></div>`:''}${d?`${progress(d.current || d.run || d)}<div class="grid">${quality(d.quality,true,d.current || d)}${playback('',d)}${d.current?.method==='ppo'||(!finite(d.current?.nll)&&!d.current?.loss_history?.length)?'':loss(d.current || d.run || d)}</div>${diagnostics(d.diagnostics,d.current)}${settings(d.current || d.run || d)}`:''}`;
  }
  function patchChildren(host, incoming) {
    const wanted=[...incoming.childNodes];
    wanted.forEach((next,index)=>{
      let previous=host.childNodes[index];
      if(previous===next)return;
      const key=node=>node.nodeType===1?(node.id || node.dataset.viewKey || ''):'';
      const nextKey=key(next);
      if(nextKey&&key(previous || {})!==nextKey){
        const match=[...host.childNodes].find(node=>key(node)===nextKey);
        if(match){host.insertBefore(match,previous || null);previous=match;}
      }
      if(!previous){host.append(next);return;}
      if(previous.nodeType!==next.nodeType || (next.nodeType===1&&(previous.tagName!==next.tagName || key(previous)!==nextKey))){host.insertBefore(next,previous);return;}
      if(next.nodeType===3){if(previous.textContent!==next.textContent)previous.textContent=next.textContent;return;}
      if(next.nodeType!==1)return;
      // Canvas pixels and GIF playback stay attached across background refreshes.
      if(next.dataset.recording&&previous.dataset.recording===next.dataset.recording&&previous.getAttribute('src')===next.getAttribute('src'))return;
      const selectValue=next.tagName==='SELECT'?next.value:null;
      const expanded=previous.tagName==='DETAILS'&&previous.open;
      const focused=previous===document.activeElement;
      [...previous.attributes].forEach(attr=>{if(!next.hasAttribute(attr.name)&&attr.name!=='open')previous.removeAttribute(attr.name);});
      [...next.attributes].forEach(attr=>{if(!(focused&&attr.name==='value')&&!(previous.tagName==='DETAILS'&&attr.name==='open')&&previous.getAttribute(attr.name)!==attr.value)previous.setAttribute(attr.name,attr.value);});
      patchChildren(previous,next);
      if(expanded)previous.open=true;
      if(previous.tagName==='SELECT'&&!focused)previous.value=selectValue;
    });
    while(host.childNodes.length>wanted.length)host.lastChild.remove();
  }
  function render() {
    const r=route();
    document.querySelectorAll('nav a').forEach(a=> { if(a.hash===`#/${r.page}`) a.setAttribute('aria-current','page');else a.removeAttribute('aria-current'); });
    refreshState.textContent = `${error?'Связь потеряна · ':''}${busy?'Обновляем…':data?`Обновлено ${date(data.updated_at)} · каждые 5 с`:'Загрузка…'}`;
    if (!data) { main.innerHTML = error?`<div class="error" role="alert">${esc(error)} Нажмите «Обновить», чтобы повторить запрос.</div>`:'<div class="loading" role="status">Загружаем состояние обучения и результаты игр…</div>'; return; }
    const renderKey=JSON.stringify([r,selectedRun,recordingRun,recordingTask,recordingPage,selectedRecording,hiddenRecording,error,detailError,detailLoading,data,detail],(key,value)=>key==='updated_at'?undefined:value);
    if(renderKey===lastRenderKey)return;
    const activeId = main.contains(document.activeElement)?document.activeElement.id:'';

    const template=document.createElement('template');
    let page = r.page==='training'?training():r.page==='games'?games(r.id):r.page==='history'?(r.id?runPage(r.id):history()):overview();
    const warnings=(data.warnings || []).map(w=>typeof w==='string'?w:w.message || w.label || '').filter(Boolean).filter(w=>w!=='Этот замер показывает освоение текущих мини-игр; перенос на новые игры и оформление оценивается отдельно.');
    template.innerHTML=`${error?`<div class="error" role="alert">${esc(error)} Показаны последние полученные данные; они могут быть устаревшими.</div>`:''}${data.current?.stale && ['running','stale','unknown'].includes(data.current?.status)?'<div class="warning">Источник давно не обновлялся. Проверьте состояние процесса обучения.</div>':''}${page}${technicalMessages(warnings)}`;
    const viewKeys=new Map();
    [...template.content.children].forEach(element=>{const name=`${element.tagName}-${element.className}`;const count=viewKeys.get(name) || 0;viewKeys.set(name,count+1);element.dataset.viewKey=`${r.page}-${r.id}-${name}-${count}`;});
    patchChildren(main,template.content);
    lastRenderKey=JSON.stringify([r,selectedRun,recordingRun,recordingTask,recordingPage,selectedRecording,hiddenRecording,error,detailError,detailLoading,data,detail],(key,value)=>key==='updated_at'?undefined:value);
    if(activeId&&document.activeElement.id!==activeId) document.getElementById(activeId)?.focus({preventScroll:true});
    initializePlayer();
  }
  async function fetchDetail(id) {
    detailLoading=true;
    try { const response=await fetch(`/api/run/${encodeURIComponent(id)}`, {cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);const result=await response.json();if(detailId!==id)return;detail=result;detailError=''; }
    catch(e){if(detailId===id)detailError=`Не удалось загрузить запуск (${e.message}).`;}
    finally{if(detailId===id){detailLoading=false;render();}}
  }
  async function refresh() {
    if(busy)return;busy=true;refreshButton.disabled=true;refreshState.textContent='Обновляем…';
    const requestedRun=selectedRun;
    try { const response=await fetch(`/api/overview${requestedRun?`?run=${encodeURIComponent(requestedRun)}`:''}`,{cache:'no-store',signal:AbortSignal.timeout(10000)});if(!response.ok)throw new Error(`HTTP ${response.status}`);const next=await response.json();if(!next || typeof next!=='object' || Array.isArray(next))throw new Error('неверный формат данных');if(requestedRun!==selectedRun)return;data=next;error=''; }
    catch(e){error=`Не удалось обновить данные (${e.name==='TimeoutError'?'сервер не ответил вовремя':e.message}).`;}
    finally{busy=false;refreshButton.disabled=false;render();if(requestedRun!==selectedRun)refresh();}
  }
  refreshButton.addEventListener('click',refresh);
  main.addEventListener('change',event=>{
    if(event.target.id==='experiment-run'){selectedRun=event.target.value;recordingRun='';recordingTask='';recordingPage=0;selectedRecording='';hiddenRecording=false;refresh();return;}
    if(event.target.id==='recording-run'){const id=event.target.value;recordingRun='';recordingTask='';recordingPage=0;selectedRecording='';hiddenRecording=false;if(route().page==='history'&&route().id){location.hash=`#/history/${encodeURIComponent(id)}`;}else{selectedRun=id;refresh();}return;}
    if(event.target.id==='recording-task'){recordingTask=event.target.value;recordingPage=0;selectedRecording='';hiddenRecording=false;render();}
    if(event.target.id==='recording'){selectedRecording=event.target.value;hiddenRecording=false;render();}
  });
  main.addEventListener('input',event=>{if(event.target.id==='frame-seek'){const p=players.get(selectedRecording);if(p){p.frame=Number(event.target.value);p.playing=false;p.last=0;drawPlayer(p);document.getElementById('play-pause').textContent='Продолжить';}}});
  main.addEventListener('click',event=>{
    if(event.target.dataset.selectRun){selectedRun=event.target.dataset.selectRun;recordingRun='';recordingTask='';recordingPage=0;selectedRecording='';hiddenRecording=false;if(route().page==='history'&&route().id){location.hash=`#/history/${encodeURIComponent(selectedRun)}`;}else refresh();return;}
    if(event.target.id==='recording-prev'||event.target.id==='recording-next'){recordingPage+=event.target.id==='recording-next'?1:-1;selectedRecording='';render();}
    if(event.target.id==='toggle-recording'){hiddenRecording=!hiddenRecording;render();}
    if(event.target.id==='retry-detail'){fetchDetail(detailId);}
    const p=players.get(selectedRecording);
    if(event.target.id==='play-pause' && p){p.playing=!p.playing;p.last=0;event.target.textContent=p.playing?'Пауза':'Продолжить';}
    if(event.target.id==='restart-playback' && p){p.frame=0;p.last=0;drawPlayer(p);}
  });
  main.addEventListener('error',event=>{if(event.target.tagName==='IMG'){const host=event.target.closest('.preview');if(host){host.classList.add('empty');host.textContent='Не удалось загрузить запись. Откройте GIF по ссылке ниже или выберите другую запись.';}}},true);
  window.addEventListener('hashchange',()=>{selectedRecording='';recordingRun='';recordingTask='';recordingPage=0;hiddenRecording=false;render();});
  document.getElementById('tensorboard').hostname=location.hostname;
  document.getElementById('archive').hostname=location.hostname;
  refresh();setInterval(refresh,5000);requestAnimationFrame(animatePlayer);
})();
