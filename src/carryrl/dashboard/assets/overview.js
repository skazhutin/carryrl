'use strict';
(() => {
  const main = document.getElementById('content');
  const refreshState = document.getElementById('refresh-state');
  const refreshButton = document.getElementById('refresh');
  let data = null, error = '', busy = false, selectedRecording = '', hiddenRecording = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
  let detail = null, detailId = '', detailError = '', detailLoading = false;
  let selectedRun = new URLSearchParams(location.search || '').get('run') || '', recordingRun = '', recordingTask = '', recordingPage = 0, lastRenderKey = '';
  let selectedMetric = '', unavailableRunNotice = '';

  let validation = null, validationRun = '', validationLoading = false, validationError = '';
  const savedView = new URLSearchParams(location.search || '');
  let validationTask = savedView.get('task') || '', validationLevel = savedView.get('level') || '', validationCheckpoint = savedView.get('checkpoint') || '', validationAppearance = savedView.get('appearance')==='pixel'?'pixel':'canonical', validationMode = ['greedy','stochastic','random'].includes(savedView.get('mode'))?savedView.get('mode'):'greedy';
  let historyFilter = 'training';
  let replayJob = null, replaySubmitting = false, replayError = '', validationView = 'level';

  const RECORDINGS_PER_PAGE = 8;
  const modeLabel = value => ({greedy:'Самое вероятное действие',stochastic:'Выбор по вероятностям',random:'Случайные действия',frozen_deterministic:'Самое вероятное действие',frozen_stochastic:'Выбор по вероятностям',generic_physical_random:'Случайные действия'})[value] || 'Режим не указан';
  const phaseLabel = value => ({'Сбор игровых эпизодов PPO':'Сбор игровых попыток','Обновление политики PPO':'Обновление модели'})[value] || value;
  const appearanceLabel = value => value==='pixel'?'Пиксельное оформление':'Обычное оформление';
  const reducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true;
  function saveValidationView() {
    const url=new URL(location.href);
    for(const [key,value] of Object.entries({task:validationTask,level:validationLevel,checkpoint:validationCheckpoint,appearance:validationAppearance,mode:validationMode})) {if(value!=='')url.searchParams.set(key,value);else url.searchParams.delete(key);}
    window.history.replaceState(null,'',url.href);
  }
  const players = new Map();
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const num = value => finite(value) ? new Intl.NumberFormat('ru-RU', {maximumFractionDigits: 2}).format(Number(value)) : '—';
  const nll = value => finite(value) ? new Intl.NumberFormat('ru-RU', {maximumSignificantDigits: 3}).format(Number(value)) : '—';
  const pct = value => finite(value) ? `${num(value)}%` : 'Нет проверки';
  const date = value => { if (!value) return '—'; const d = new Date(value); return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString('ru-RU', {day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'}); };
  const duration = value => { if (!finite(value)) return '—'; const s = Math.max(0, Math.round(value)); return s >= 3600 ? `${Math.floor(s/3600)} ч ${Math.floor(s%3600/60)} мин` : `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`; };
  const route = () => { const parts = location.hash.replace(/^#\/?/, '').split('/'); return {page:parts[0] || 'overview', id:decodeURIComponent(parts.slice(1).join('/'))}; };
  const change = value => finite(value) ? `<span title="Разность долей побед на одинаковых проверочных уровнях, в процентных пунктах" class="change ${Number(value)<0?'negative':''}">${Number(value)>0?'+':''}${num(value)} п.п.</span>` : '<span class="muted">Нет исходной проверки</span>';
  const empty = message => `<p class="muted">${esc(message)}</p>`;
  const counts = item => finite(item?.successes) && finite(item?.episodes) ? `Победы: ${num(item.successes)} из ${num(item.episodes)} попыток` : 'Число игр пока неизвестно';
  const outcome = value => {
    if(!value)return '';
    if(typeof value==='string')return ({success:'Победа',failure:'Поражение',win:'Победа',loss:'Поражение'})[value] || value;
    if(value.success)return 'Победа';
    if(value.failure)return 'Поражение';
    return value.known===false?'Исход не распознан':'Не завершено';
  };
  const recordingOutcome = item => (item?.truncated || item?.outcome?.timeout) && !item.outcome?.success && !item.outcome?.failure
    ? 'Лимит ходов' : outcome(item?.outcome);
  const kind = value => ({imitation:'Повторение действий',imitation_bc:'Повторение действий',imitation_training:'Повторение действий',sequence_training:'Обучение последовательностям',synthetic_compute_probe:'Проверка производительности',training:'Обучение',evaluation:'Проверка',rl:'Обучение с наградой',ppo:'Обучение с наградой',playback:'Запись игры'})[value] || 'Эксперимент';
  const runLabel = item => item?.experiment?.display_name || item?.experiment?.title || (item?.label && item.label!==item.run_id ? item.label : `${kind(item?.kind || 'imitation')}${(item?.epochs ?? item?.epochs_total)?` · ${num(item.epochs ?? item.epochs_total)} эпох`:''}${item?.started_at?` · ${date(item.started_at)}`:''}`);
  function pickerLabel(item) {
    const e=item?.experiment || {};
    if(e.picker_label)return e.picker_label;
    const name=e.display_name || e.title?.split(' · ')[0] || 'Запуск без описания';
    return `${name}${item?.started_at?` · ${date(item.started_at)}`:''}`;
  }
  function selectRun(id) {
    selectedRun=id || '';
    validation=null;validationRun='';validationError='';validationTask='';validationLevel='';validationCheckpoint='';replayJob=null;replayError='';validationView='level';
    unavailableRunNotice='';
    recordingRun='';recordingTask='';recordingPage=0;selectedRecording='';hiddenRecording=reducedMotion();
    const url=new URL(location.href);
    if(selectedRun)url.searchParams.set('run',selectedRun);else url.searchParams.delete('run');
    ['task','level','checkpoint','appearance','mode'].forEach(key=>url.searchParams.delete(key));
    window.history.replaceState(null,'',url.href);
  }
  function selectRouteRun() {
    const r=route();
    if(r.page!=='history'||!r.id)return false;
    if(selectedRun!==r.id)selectRun(r.id);
    return true;
  }
  function savedComment(e,id='saved-comment') {
    const comment=e?.saved_comment;
    if(!comment)return '';
    return `<details id="${esc(id)}" class="saved-comment"><summary>Сохранённая заметка к запуску</summary><p class="note">Комментарий из experiment.json. Он мог быть написан до последних проверок; актуальные результаты показаны выше.</p>${comment.conclusion?`<p>${esc(comment.conclusion)}</p>`:''}${comment.limitations?.length?`<ul>${comment.limitations.map(value=>`<li>${esc(value)}</li>`).join('')}</ul>`:''}</details>`;
  }
  function verdictSummary(e,currentId=data?.current?.run_id) {
    const verdict=e?.verdict;
    if(!verdict)return '';
    const state=typeof verdict==='object'?verdict.state:'';
    const text=typeof verdict==='string'?verdict:verdict.label || '';
    const measuredCurrent=qualityStage(data?.quality,data?.current)==='final';
    const exact=measuredCurrent&&finite(verdict?.successes)&&finite(verdict?.baseline_successes)&&finite(verdict?.episodes)&&['worse','improved','unchanged'].includes(state);
    const message=exact?`После этого обучения: ${num(verdict.successes)}/${num(verdict.episodes)} побед; до обучения — ${num(verdict.baseline_successes)}/${num(verdict.episodes)}. ${state==='worse'?'Побед стало меньше.':state==='improved'?'Побед стало больше.':'Число побед не изменилось.'}`:!measuredCurrent&&data?.current?.run_id===currentId?'Последняя проверенная версия показана ниже. Качество текущих весов ещё не подтверждено.':text;
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
    return own.length&&(own.at(-1).epoch===0 || own.at(-1).rollout===0)?'initial':'final';
  }
  function technicalMessages(warnings) {
    return warnings.length?`<details id="technical-messages" class="technical-messages"><summary>Технические сообщения (${warnings.length})</summary><ul>${warnings.map(w=>`<li>${esc(w)}</li>`).join('')}</ul></details>`:'';
  }
  function experimentSelector() {
    const runs=data.experiments || [];
    if(!runs.length)return '';
    const selected=data.current || runs.find(r=>r.run_id===selectedRun);
    const registered=data.training_current || data.current;
    const active=registered?.run_id===data.current?.run_id?data.current:(runs.find(r=>r.run_id===registered?.run_id) || registered);
    const defaultSelected=!selectedRun || selectedRun===active?.run_id;
    const ids=new Set([selected?.run_id,active?.experiment?.baseline_run_id,active?.experiment?.recommended_run_id,selected?.experiment?.baseline_run_id,selected?.experiment?.recommended_run_id].filter(Boolean));
    const choices=runs.filter(r=>(r.run_id || r.id)!==active?.run_id&&ids.has(r.run_id || r.id));
    return `<div class="experiment-picker"><label for="experiment-run">Запуск обучения</label><select id="experiment-run"><option value="" ${defaultSelected?'selected':''}>${esc(pickerLabel(active))}</option>${choices.map(r=>`<option value="${esc(r.run_id || r.id)}" ${(r.run_id || r.id)===selectedRun?'selected':''}>${esc(pickerLabel(r))}</option>`).join('')}</select><a href="#/history">Все запуски</a></div>`;
  }
  function experimentSummary(c, experiment=c?.experiment) {
    if (!c) return '';
    const e=experiment || {};
    const purpose=e.purpose || c.method_description || 'Проверяем, как обучение меняет игру агента на контрольных уровнях.';
    const conclusion=e.conclusion || (c.running?'Эксперимент продолжается; окончательный результат ещё не получен.':'Вывод не сохранён. Ниже показаны доступные проверки этой модели.');
    const limitations=Array.isArray(e.limitations)?e.limitations:e.limitations?[e.limitations]:[];
    return `<section class="experiment-summary" aria-label="Выбранный эксперимент"><div class="run-heading"><h1>${esc(runLabel({...c,experiment:e}))}</h1>${status(c)}</div>${c.evaluation_only||c.technical_only?'':verdictSummary(e,c.run_id || c.id)}<p class="experiment-purpose">${esc(purpose)}</p><p class="experiment-conclusion"><strong>Результат.</strong> ${esc(conclusion)}</p>${limitations.length?`<p class="experiment-limits"><strong>Ограничения.</strong> ${limitations.map(esc).join(' ')}</p>`:''}<p class="note">${c.started_at?`Начат ${date(c.started_at)}.`:''}${e.baseline_run_id&&(data?.experiments || []).some(r=>r.run_id===e.baseline_run_id)?` ${c.technical_only?'Исходная модель:':'Сравнение с'} <a href="#/history/${encodeURIComponent(e.baseline_run_id)}">${esc(experimentTitle(e.baseline_run_id))}</a>.`:''}</p>${savedComment(e)}${technicalDetails('experiment-identity',[['Запуск',c.run_id || c.id],['Сохранённая модель',e.checkpoint]])}</section>`;
  }
  function metricIdentity(q,c=data.current) {
    if(!finite(q?.value))return empty('Игровая проверка пока не сохранена.');
    const stage=qualityStage(q,c);
    const inherited=stage==='inherited';
    const runId=q?.measurement_run_id || (inherited?q?.inherited_from?.run_id || q?.latest_run_id:q?.latest_run_id || c?.run_id);
    const measured=(q?.run_history || q?.history || []).findLast(p=>p.run_id===runId&&p.revision===q?.latest_revision);
    const version=stage!=='initial'&&!inherited&&measured?(finite(measured.rollout)?` · после ${num(measured.rollout)} циклов`:finite(measured.epoch)?` · после ${num(measured.epoch)} проходов по данным`:''):'';
    return `<div class="metric-identity"><strong>${stage==='initial'?'Начальная модель':inherited?'Предыдущая модель':stage==='previous'?'Последняя проверенная версия':'Последняя проверенная версия этого запуска'}${version}</strong><br>Проверено ${date(q?.evaluated_at || q?.inherited_from?.evaluated_at)}.${!c?.evaluation_only?' Уровни, на которых модель не обучается.<br>Обычное оформление · Самое вероятное действие.':''}${technicalDetails('metric-identity-details',[['Условия проверки',q?.cohort_description || q?.protocol_description],['Название запуска',experimentTitle(runId,c)],['Запуск',runId],['Версия модели',q?.latest_revision ?? q?.inherited_from?.revision],['Файл проверки',q?.measurement_source || q?.latest_evaluation_file]])}</div>`;
  }
  function status(item) {
    const s = item?.status || '';
    const labels={completed:'Завершено',running:'В работе',failed:'Остановлено с ошибкой',error:'Ошибка',interrupted:'Прервано',stopped:'Остановлено',stale:'Данные устарели'};
    return `<span class="status ${item?.running?'running':/fail|error|interrupt/.test(s)?'failed':''}"><span class="dot" aria-hidden="true"></span>${esc(labels[s] || item?.status_label || 'Состояние неизвестно')}</span>`;
  }
  function eta(c) {
    if (c.status === 'completed') return '0:00';
    if (finite(c.eta_low_seconds) && finite(c.eta_high_seconds)) return `${duration(c.eta_low_seconds)}–${duration(c.eta_high_seconds)}`;
    if (finite(c.eta_seconds)) return duration(c.eta_seconds);
    return c.running ? 'Оцениваем темп' : 'Нет оценки';
  }
  function progress(c) {
    if(c?.evaluation_only)return '';
    if (!c) return `<section class="panel">${empty('Запуск обучения ещё не найден.')}</section>`;
    if(c.method==='ppo')return ppoProgress(c);
    const phases = c.phase_steps?.length ? c.phase_steps : ['Подготовка','Исходная проверка','Обучение','Итоговая проверка','Сохранение'].map((label,i) => ({label,state:c.status==='completed'||i+1<c.phase_index?'done':i+1===c.phase_index?'active':'pending'}));
    const percent = finite(c.training_percent) ? Math.max(0,Math.min(100,Number(c.training_percent))) : null;
    return `<section class="panel ${c.status==='completed'?'completed-progress':''}" aria-label="Состояние обучения"><div class="run-heading"><div><h2>${esc(c.status==='completed'?'Обучение завершено':phaseLabel(c.phase_label) || 'Обучение')}</h2><div class="run-label">${esc(runLabel(c))} · Этап ${num(c.phase_index)} из ${num(c.phase_total || 5)}</div></div>${c.status==='completed'?'':status(c)}</div>
      <div class="phase-track">${phases.map(p=>`<div class="phase ${esc(p.state)}">${esc(p.label)}</div>`).join('')}</div>
      <div class="progress-label"><span>Обновления модели</span><span>${num(c.optimizer_steps)} / ${num(c.optimizer_steps_total)}${percent===null?'':` · ${num(percent)}%`}</span></div><p class="note">Проход данных — один просмотр обучающих примеров. Процент показывает выполненную работу, а не успех в играх.</p><progress aria-label="Прогресс обновлений модели" ${percent===null?'':`value="${percent}"`} max="100"></progress>
      <div class="facts"><div class="fact"><span>Проход данных</span><strong>${num(c.epoch)} / ${num(c.epochs_total)}</strong>${c.status==='completed'?'':`<small>${num(c.epochs_completed)} завершено</small>`}</div><div class="fact"><span>Пакет примеров в проходе</span><strong>${num(c.batch)} / ${num(c.batches_per_epoch)}</strong></div><div class="fact"><span>Прошло времени</span><strong>${duration(c.elapsed_seconds)}</strong>${finite(c.phase_elapsed_seconds)&&c.status!=='completed'?`<small>Текущий этап: ${duration(c.phase_elapsed_seconds)}</small>`:''}</div><div class="fact"><span>Осталось</span><strong>${eta(c)}</strong>${c.status==='completed'?'':`<small>${esc(c.eta_scope==='training'?'Только обучение; без итоговой проверки':c.eta_scope==='phase'?'До конца текущей фазы':c.eta_scope==='run'?'Весь запуск':c.eta_reason || '')}</small>`}</div></div>
      ${liveContext(c)}${c.phase_units_total&&c.status!=='completed'?`<p class="note">Текущий этап: ${num(c.phase_units_done)} из ${num(c.phase_units_total)}</p>`:''}</section>`;
  }
  function liveContext(c) {
    const budget=c?.budget;
    const evaluating=c?.running===true&&['baseline','initial_evaluation','initial_eval','evaluation','interim_evaluation','pixel_evaluation','final_evaluation','final_eval'].includes(c.phase);
    const policyLabel=modeLabel(c?.evaluation_policy);
    const evaluation=evaluating&&c?.evaluation_task?`<p class="live-evaluation" role="status"><strong>Сейчас проверяется:</strong> ${esc(data?.task_guides?.[c.evaluation_task]?.label || c.evaluation_task)} · ${esc(policyLabel)} · шаг ${num(c.evaluation_step)}. Веса во время проверки не меняются.</p>`:'';
    return `${evaluation}${budget?`<div class="budget-line"><strong>До лимита запуска: ${duration(budget.remaining_seconds)}</strong><span>Общий бюджет ${duration(budget.total_seconds)}, включая паузы. Завершение до ${date(budget.hard_deadline_at)}.</span>${budget.training_deadline_at?`<span>Обучение до ${date(budget.training_deadline_at)}; резерв на проверку и сохранение — ${duration(budget.reserve_seconds)}. Это граница бюджета, не прогноз.</span>`:''}</div>`:''}${c?.latest_checkpoint?`${technicalDetails('latest-checkpoint',[['Сохранённая модель',c.latest_checkpoint],['Цикл',c.latest_checkpoint_rollout],['Ревизия',c.latest_checkpoint_weights_revision],['Текущие веса сохранены',c.current_policy_saved===false?'Нет':'Нет подтверждения']])}`:''}`;
  }
  function memoryFreshness(c) {
    const g=c?.resource_guard;
    if(!g || !('sample_stale' in g))return '';
    return `<div class="${g.sample_stale || g.memory_throttled?'warning':'note'}">${g.memory_throttled?`<strong>Пауза по памяти.</strong> ${esc(g.pause_reason || 'Ожидание освобождения RAM.')} `:''}Последний замер ресурсов: ${date(g.sample_at)}; возраст ${duration(g.sample_age_seconds)}.${g.sample_stale?' Замер устарел: текущее потребление и пауза не подтверждены.':''}${finite(g.wait_duration_seconds)?` Ожидание памяти: ${duration(g.wait_duration_seconds)}.`:''}${finite(g.active_wall_seconds)?` Активная работа: ${duration(g.active_wall_seconds)}.`:''}${g.sample_error?` ${esc(g.sample_error)}`:''}</div>`;
  }
  function evaluationProgress(c) {
    const count=v=>Number.isSafeInteger(v)&&v>=0?v:null;
    const total=count(c.phase_units_total), raw=count(c.phase_units_done);
    const done=raw!==null&&(total===null||raw<=total)?raw:null;
    const measured=done!==null&&total!==null&&total>0;
    const reports=count(c.reports_completed), reportsTotal=count(c.reports_total);
    const saved=reports!==null&&reportsTotal!==null&&reportsTotal>0&&reports<=reportsTotal;
    const paired=c.job_kind==='paired_frozen_evaluation';
    return `<section class="panel" aria-label="Состояние проверки"><div class="run-heading"><div><h2>${paired?'Парная проверка сохранённых моделей':'Проверка сохранённой модели'}</h2><p class="run-label">${esc(c.method_description || 'Веса модели не изменяются.')}</p></div>${status(c)}</div><div class="progress-label"><span>Завершённые эпизоды проверки</span><span>${num(done)} / ${num(total)}</span></div><progress aria-label="Прогресс эпизодов проверки" ${measured?`value="${done}" max="${total}"`:'max="100"'}></progress><div class="facts"><div class="fact"><span>Сохранённые отчёты</span><strong>${saved?`${num(reports)} / ${num(reportsTotal)}`:'Нет счётчика'}</strong></div><div class="fact"><span>Обновлений весов в этом запуске</span><strong>${num(count(c.optimizer_steps))}</strong></div><div class="fact"><span>Время проверки</span><strong>${duration(c.elapsed_seconds)}</strong></div><div class="fact"><span>До конца эпизодов</span><strong>${eta(c)}</strong>${paired?'<small>Подготовка и итоговый аудит отдельно</small>':''}</div></div>${paired?'<p class="note">Сравниваются две модели на одинаковых уровнях навигации в пяти повторных запусках с фиксированной случайностью. Результат относится к навигации.</p>':''}${c.evaluation_note?`<p class="note">${esc(c.evaluation_note)}</p>`:''}</section>`;
  }
  function ppoProgress(c) {
    const percent=finite(c.rollouts)&&finite(c.rollouts_total)&&Number(c.rollouts_total)>0?Math.max(0,Math.min(100,Number(c.rollouts)/Number(c.rollouts_total)*100)):null;
    const cycleEta=c.stop_reason?'Циклы остановлены':finite(c.cycle_eta_seconds)?duration(c.cycle_eta_seconds):'Нет измеренного темпа';
    return `<section class="panel ${c.status==='completed'?'completed-progress':''}" aria-label="Состояние обучения"><div class="run-heading"><div><h2>${esc(c.status==='completed'?'Обучение завершено':c.status==='resource_stopped'?'Запуск остановлен по лимиту ресурсов':phaseLabel(c.phase_label) || 'Обучение с наградой')}</h2></div>${c.status==='completed'?'':status(c)}</div>${c.phase_steps?.length?`<div class="phase-track" style="--phase-count:${c.phase_steps.length}">${c.phase_steps.map(p=>`<div class="phase ${esc(p.state)}">${esc(p.label)}</div>`).join('')}</div>`:''}<div class="progress-label"><span>Завершённые циклы обучения</span><span>${num(c.rollouts)} / ${num(c.rollouts_total)}${percent===null?'':` · ${num(percent)}%`}</span></div><p class="note">Цикл — сбор игровых попыток и обновление модели. Процент показывает выполненную работу, а не долю побед.</p><progress aria-label="Прогресс циклов обучения" ${percent===null?'':`value="${percent}"`} max="100"></progress><div class="facts completed-facts"><div class="fact"><span>Обновления модели</span><strong>${num(c.optimizer_steps)}</strong></div><div class="fact"><span>Игровые действия</span><strong>${num(c.attempted_train_decisions_this_process)}</strong></div><div class="fact"><span>Прошло времени</span><strong>${duration(c.elapsed_seconds)}</strong></div><div class="fact"><span>До конца циклов примерно</span><strong>${cycleEta}</strong><small>Итоговая проверка отдельно</small></div></div>${liveContext(c)}${c.phase_units_total&&c.status!=='completed'?`<p class="note">${esc((c.phase_unit || c.phase_unit_kind)==='episodes'?(c.phase==='collecting'?'Собрано':'Проверено'):(c.phase_unit || c.phase_unit_kind)==='decisions'?'Выполнено действий:':'Текущий этап:')} ${num(c.phase_units_done)} из ${num(c.phase_units_total)}${(c.phase_unit || c.phase_unit_kind)==='episodes'?' попыток':''}. ${finite(c.eta_seconds)?`До конца этапа примерно: ${duration(c.eta_seconds)}.`:''}</p>`:''}${c.stop_reason?`<p class="note">${esc(c.stop_reason==='minimum_free_disk_mib'?'Достигнут заданный минимум свободного места.':c.stop_reason==='max_wall_seconds'?'Достигнут лимит времени запуска.':c.stop_reason)}</p>`:''}${technicalDetails('training-revision',[['Метод',c.method_description],['Совокупная ревизия модели',c.weights_revision],['Верхний предел обновлений',c.optimizer_steps_upper_bound],['Сохранённый счётчик действий',c.decisions],['Ограничение шага обновления','Число принятых обновлений может быть меньше плана из-за защитного лимита изменения вероятностей действий (KL).']])}</section>`;
  }
  function chart(history, field, label, fixed = false, axisLabel = '', chartId = field) {
    const points = (history || []).filter(p => finite(p[field]));
    if (!points.length) return empty('Измерений пока нет. График появится после сохранения данных.');
    if(fixed&&points.length===1)return empty('Пока сохранено одно измерение. Изменение можно сравнить, когда есть хотя бы две проверки в одинаковых условиях.');
    const xValue = p => finite(p.step)?p.step:finite(p.updates)?p.updates:finite(p.epoch)?p.epoch:finite(p.rollout)?p.rollout:0;
    const maxX = Math.max(1,...points.map(xValue));
    const values=points.map(p=>p[field]), low=Math.min(...values), high=Math.max(...values);
    const pad=Math.max((high-low)*.12,Math.abs(high)*.05,.001);
    const minY = fixed?0:low-pad, maxY = fixed?100:high+pad;
    const x = p => 60 + xValue(p)/maxX*475, y = value => 190-(value-minY)/(maxY-minY)*152;
    const ticks = [minY,(minY+maxY)/2,maxY];
    const metric=fixed?num:nll;
    const axis=axisLabel || (points.some(p=>finite(p.updates))?'Обновления модели':points.some(p=>finite(p.rollout))?'Циклы PPO':'Эпохи');
    const suffix=fixed?'%':'';
    return `<div class="chart"><svg viewBox="0 0 565 230" role="img" aria-label="${esc(label)}"><title>${esc(label)}. ${points.length} точек. Последнее значение ${metric(points.at(-1)[field])}${suffix}.</title>${ticks.map(v=>`<line class="axis" x1="60" x2="535" y1="${y(v)}" y2="${y(v)}"/><text x="51" y="${y(v)+4}" text-anchor="end">${metric(v)}${suffix}</text>`).join('')}<polyline class="line" points="${points.map(p=>`${x(p)},${y(p[field])}`).join(' ')}"/>${points.map((p,i)=>`<circle id="point-${esc(chartId)}-${i}" cx="${x(p)}" cy="${y(p[field])}" r="4" tabindex="0" role="img" aria-label="${esc(label)}: ${metric(p[field])}${suffix}; ${esc(axis)}: ${num(xValue(p))}"><title>${metric(p[field])}${suffix} · ${esc(axis)}: ${num(xValue(p))}</title></circle>`).join('')}<text x="60" y="213">0</text><text x="535" y="213" text-anchor="end">${num(maxX)}</text></svg><div class="chart-title">${esc(axis)}</div></div><details id="chart-${esc(chartId)}"><summary>Значения графика (${points.length})</summary><div class="table-wrap"><table><thead><tr><th>${esc(axis)}</th><th>${esc(label)}</th></tr></thead><tbody>${points.map(p=>`<tr><td>${num(xValue(p))}</td><td>${metric(p[field])}${suffix}</td></tr>`).join('')}</tbody></table></div></details>`;
  }
  function diagnostics(d,c=data.current) {
    const t=d?.transfer,m=d?.memory;
    const identity=r=>`<p class="note diagnostic-identity">${esc(r.current_weights?'Проверены сохранённые веса выбранного эксперимента':'Проверена предыдущая модель')}: ${esc(experimentTitle(r.run_id,c))}. Проверено ${date(r.evaluated_at)}.</p>${technicalDetails(`diagnostic-${r===t?'appearance':'memory'}-identity`,[['Запуск',r.run_id],['Версия модели',r.revision],['Сохранённая модель',r.checkpoint]])}`;
    const previous=(r,body,key)=>r&&!r.current_weights?`<p class="note">Текущая модель ещё не проверена.</p><details id="previous-${key}"><summary>Проверка предыдущей модели: ${esc(experimentTitle(r.run_id,c))}</summary>${body}</details>`:body;
    const profile=p=>p==='canonical96'?'Обычное оформление · поле 96×96':p.startsWith('varied96')?'Изменённое оформление · поле 96×96':p.startsWith('varied64')?'Новое оформление · поле 64×64':p.startsWith('varied128')?'Новое оформление · поле 128×128':p;
    const appearance=t?`${identity(t)}<div class="table-wrap"><table><thead><tr><th>Условия</th><th>Язык</th><th>Победы</th><th>Эпизоды</th><th>Изменение</th></tr></thead><tbody>${t.rows.map(r=>`<tr><td>${esc(profile(r.profile))}</td><td>${esc(r.language.toUpperCase())}</td><td>${pct(r.value)}${r.unknown_outcomes?`–${pct(r.upper)}`:''}</td><td>${num(r.successes)} / ${num(r.episodes)}</td><td>${finite(r.change_pp)?change(r.change_pp):'—'}</td></tr>`).join('')}</tbody></table></div><p class="note">${t.coverage_mode==='full_bank'?'Полный банк.':'Диагностическая выборка уровней.'} RU/EN — описание игры на русском/английском. Победы не доказывают понимание инструкции; размеры поля могут менять механику.</p>${d.comparison_reason?technicalDetails('appearance-comparison',[['Условия сравнения',d.comparison_reason]]):''}`:empty('Текущая модель ещё не проверена на другом оформлении.');
    const memory=m?`${identity(m)}<p>Сравниваем обычную игру с ожиданием без нажатий: так проверяем, не использует ли агент историю своих нажатий вместо запоминания подсказки.</p><div class="table-wrap"><table><thead><tr><th scope="col">Режим</th><th scope="col">Правильный первый выбор</th><th scope="col">Оба варианта пары</th></tr></thead><tbody>${m.modes.map(r=>`<tr><th scope="row">${esc(r.mode==='normal'?'Обычная игра':'Без нажатий во время ожидания')}</th><td>${finite(r.episodes)?`${num(r.first_choose_correct)} / ${num(r.episodes)} попыток<small>Выбор сделан: ${num(r.first_choose_observed)}; без выбора: ${num(r.missing_choices)}</small>`:`${num(r.first_choose_correct)} из ${num(r.first_choose_observed)} сделанных выборов<small>Общий объём попыток неизвестен</small>`}</td><td>${num(r.paired_both_first_choose_correct)} / ${num(r.geometries)} пар</td></tr>`).join('')}</tbody></table></div><p class="note">Первое направление после подсказки и пары уровней, где агент правильно реагирует на обе противоположные подсказки. Эта диагностика сама по себе не доказывает полноценную память.</p>`:'';
    if(!t&&!m)return empty('Дополнительные проверки памяти и оформления пока не сохранены.');
    return `<div class="grid diagnostic-summary">${t?`<section class="panel"><div class="help-heading"><h2>Проверка на другом оформлении</h2>${metricHelp('appearance')}</div><p class="diagnostic-explanation">Та же модель играет с другими цветами, расположением элементов, размером поля или языком описания.</p>${previous(t,appearance,'appearance')}</section>`:''}${m?`<section class="panel"><div class="help-heading"><h2>Память: выбор после подсказки</h2>${metricHelp('memory_skill')}</div>${previous(m,memory,'memory')}</section>`:''}</div>`;
  }
  function pixelValidation(q=data.quality,c=data.current) {
    if(c?.method!=='ppo')return '';
    const p=q?.pixel_validation;
    if(!p)return `<section class="panel pixel-validation"><h2>Проверка пиксельного оформления</h2>${empty(c.phase==='pixel_evaluation'?'Проверка выполняется. Результаты появятся после сохранения полного отчёта.':'Проверка на отложенных пиксельных стилях ещё не сохранена. Это не означает ноль побед.')}</section>`;
    return `<section class="panel pixel-validation"><h2>Обычное и пиксельное оформление</h2><p>Отложенные пиксельные стили: <strong>${pct(p.value)}</strong> · ${counts(p)}. ${esc(checkpointLabel(p))}.</p>${finite(c.weights_revision)&&c.weights_revision!==p.revision?'<p class="note">Это последняя сохранённая проверка. Текущие веса модели ещё не проверены на пиксельном оформлении.</p>':''}<p class="note">${p.canonical_same_weights?'Оба столбца относятся к одинаковым весам модели. Наборы оформления проверяются отдельно.':'Для этих весов ещё нет подтверждённой обычной проверки. Отсутствующие значения не означают ноль побед.'} Эти результаты относятся к проверенным играм и стилям.</p><div class="table-wrap"><table><thead><tr><th scope="col">Игра</th><th scope="col">Обычное</th><th scope="col">Пиксельное · отложенные стили</th></tr></thead><tbody>${(p.games || []).map(g=>`<tr><th scope="row"><a href="#/games/${encodeURIComponent(g.task)}">${esc(g.label || g.task)}</a></th><td>${g.canonical?`${pct(g.canonical.value)} · ${counts(g.canonical)}`:'Нет проверки'}</td><td>${pct(g.value)} · ${counts(g)}</td></tr>`).join('')}</tbody></table></div><p class="note">Проверено ${date(p.evaluated_at)}.</p>${technicalDetails('pixel-validation-source',[['Файл проверки',p.evaluation_file],['Версия модели',p.revision],['Протокол',p.protocol_sha256]])}</section>`;
  }
  function trainingOutcomes(c) {
    if(c?.method!=='ppo')return '';
    const t=c.training_outcomes || {}, rows=t.games || [];
    const complete=rows.length&&rows.every(g=>finite(g.successes)&&finite(g.known)&&finite(g.episodes));
    const total=field=>rows.reduce((sum,g)=>sum+g[field],0);
    return `<details id="practice-results" class="panel practice-panel"><summary>Практика во время обучения${complete?`<span>Последний цикл ${num(t.rollout)} · победы: ${num(total('successes'))} из ${num(total('episodes'))} попыток · неизвестные исходы: ${num(total('episodes')-total('known'))}</span>`:'<span>Завершённые попытки пока не сохранены</span>'}</summary><p class="note">Эти попытки используются для обучения. Проверочные уровни оцениваются отдельно. Награда — обучающий сигнал за действия, её сумма не равна числу побед. В таблице победы показаны из попыток с известным исходом; общий объём включает и неизвестные исходы.</p>${rows.length?`<div class="table-wrap"><table><thead><tr><th scope="col">Игра</th><th scope="col">Победы / известные исходы</th><th scope="col">Все попытки</th><th scope="col">Средняя награда</th></tr></thead><tbody>${rows.map(g=>`<tr><th scope="row">${esc(g.label || g.task)}</th><td>${num(g.successes)} / ${num(g.known)}</td><td>${num(g.episodes)}</td><td>${nll(g.mean_reward)}</td></tr>`).join('')}</tbody></table></div>`:empty('Завершённые игровые попытки ещё не сохранены.')}${technicalDetails('practice-source',[['Источник','training.jsonl'],['Цикл',t.rollout]])}</details>`;
  }
  const curriculumStage = value => ({easy:'Начальный',bridge:'Переходный',full:'Полный'})[value] || 'Нет данных';
  function taskProgress(c) {
    if(c?.method!=='ppo')return '';
    const curriculum=c.rl_details?.task_curriculum?.tasks || {}, practice=c.training_outcomes?.games || [], checked=data.quality?.games || [];
    const tasks=[...new Set([...Object.keys(data.task_guides || {}),...practice.map(g=>g.task),...checked.map(g=>g.task),...Object.keys(curriculum)])].filter(Boolean);
    const stages=['easy','bridge','full'];
    return `<section class="panel task-progress"><h2>Прогресс по всем играм</h2><p class="note">Этап сложности относится к обучающим уровням каждой игры. Победы на отложенных уровнях показаны отдельно и относятся к последней сохранённой проверке. Переход этапа сам по себе не доказывает улучшения игры.</p><p class="note">Практика последнего цикла: победы / известные исходы. Отложенная проверка: победы / все попытки.</p><div class="table-wrap"><table><thead><tr><th scope="col">Игра</th><th scope="col">Этап обучения</th><th scope="col">Практика</th><th scope="col">Проверка</th></tr></thead><tbody>${tasks.map(task=>{const state=curriculum[task], train=practice.find(g=>g.task===task), valid=checked.find(g=>g.task===task), stage=stages[state?.stage], window=state?.windows?.[stage];return `<tr><th scope="row"><a href="#/games/${encodeURIComponent(task)}">${esc(data.task_guides?.[task]?.label || train?.label || valid?.label || task)}</a></th><td>${curriculumStage(stage)}${Array.isArray(window)&&window.length?`<small>Сигнал модели награды: ${num(window.filter(r=>r.success===true).length)} / ${num(window.length)}<br>в текущем окне этапа</small>`:''}</td><td>${train&&finite(train.successes)&&finite(train.known)?`${num(train.successes)} / ${num(train.known)}${finite(train.episodes)&&train.episodes>train.known?`<small>неизвестно ${num(train.episodes-train.known)}</small>`:''}`:'Нет завершённых попыток'}</td><td>${valid&&finite(valid.successes)&&finite(valid.episodes)?`${num(valid.successes)} / ${num(valid.episodes)}`:'Не измерено'}</td></tr>`;}).join('')}</tbody></table></div>${!Object.keys(curriculum).length?'<p class="note">Этапы сложности не сохранены в этом запуске.</p>':''}${!tasks.length?empty('Список игр пока недоступен.'):''}</section>`;
  }
  function controlDecision(decision, fallback) {
    if(!decision || typeof decision!=='object')return fallback;
    if(decision.reason==='low_confidence')return 'Оценщик не уверен';
    if(decision.reason==='inconsistent')return 'Несогласованные выводы';
    if(decision.value==='unspecified' || decision.reason==='insufficient_information')return 'Не описано в инструкции';
    return ({supported:'Допустимо',unsupported:'Не предусмотрено',held:'Удержание',press:'Нажатие и отпускание',relative:'Относительное движение',same:'Один класс',different:'Разные классы'})[decision.value] || fallback;
  }
  function controlTable(e,i) {
    const status={supported:'Допустимо',unsupported:'Не предусмотрено',unknown:'Не определено (прежний формат)'}, mode={held:'Удержание',press:'Нажатие',relative:'Относительное движение',unknown:'Не определено (прежний формат)'};
    const rows=e.control_table.predictions || [], pairs=e.control_table.pair_decisions || [];
    return `<div class="table-wrap"><table><thead><tr><th>Кнопка / движение</th><th>Допустимость</th><th>Режим</th><th>Класс</th><th>Уверенность</th></tr></thead><tbody>${rows.map(p=>`<tr><td>${esc(p.candidate)}</td><td>${esc(controlDecision(p.applicability_decision,status[p.status] || 'Нет данных'))}${finite(p.applicability_decision?.confidence)?`<small>${nll(p.applicability_decision.confidence)}</small>`:''}</td><td>${esc(controlDecision(p.mode_decision,mode[p.mode] || 'Нет данных'))}${finite(p.mode_decision?.confidence)?`<small>${nll(p.mode_decision.confidence)}</small>`:''}</td><td>${esc(p.group ?? 'Не определён')}</td><td>${nll(p.confidence)}</td></tr>`).join('')}</tbody></table></div>${pairs.length?`<details id="control-pairs-${i}"><summary>Отношения между кнопками · ${num(pairs.length)}</summary><p class="note">Первые ${num(Math.min(100,pairs.length))} сохранённых пар. Неописанная связь и неуверенность оценщика показаны отдельно.</p><ul>${pairs.slice(0,100).map(p=>`<li>${esc(p.left)} / ${esc(p.right)}: ${esc(controlDecision(p.decision,'Нет данных'))} · ${nll(p.decision?.confidence)}</li>`).join('')}</ul></details>`:''}${technicalDetails(`control-source-${i}`,[['SHA256 таблицы',e.control_table_sha256]])}`;
  }
  function controlEvents(e) {
    const labels={unsupported_first_use:'Первое применение недопустимого класса',supported_first_use:'Допустимый класс с бонусом',repeat:'Повторный класс',unsupported_step_penalty:'Штраф за команду с недопустимым вводом',supported_award:'Выплаченный бонус',unknown:'Допустимость не подтверждена',deduplication_unknown:'Группа синонимов не подтверждена',release:'Отпускание',unsupported_active:'Недопустимый активный ввод',terminal_no_bonus:'Завершение без бонуса',unsupported_veto:'Бонус отменён из-за недопустимого ввода',command_positive_limit:'Лимит бонуса на команду',positive_cap_or_zero_schedule:'Предел бонуса или нулевой коэффициент',supported_base:'Базовый бонус',supported_mode_upgrade:'Доплата за режим',low_confidence:'Оценщик не уверен',insufficient_information:'Не описано в инструкции',inconsistent:'Несогласованные выводы'};
    const events=e.control_events, reasons=e.control_event_reasons;
    const display=values=>Object.entries(values || {}).map(([key,value])=>`${esc(labels[key] || key)}: ${num(value)}`).join('<br>');
    const budget=Object.entries(e.control_budget || {}).filter(([,value])=>finite(value)).map(([key,value])=>`${esc(({positive_paid:'Выплачено бонуса',negative_contribution:'Начислено штрафа'})[key] || key)}: ${nll(value)}`).join(' · ');
    return `${events?display(events):'Нет измерений'}${reasons?`<small>Причины: ${display(reasons)}</small>`:''}${budget?`<small>Использованные бюджеты: ${budget}</small>`:''}${finite(e.control_coefficient)?`<small>Множитель эпизода: ${nll(e.control_coefficient)}</small>`:''}`;
  }
  function rlDetails(c) {
    if(c?.method!=='ppo')return '';
    const d=c.rl_details || {}, a=d.advantage_diagnostics || {}, cf=a.auxiliary_counterfactual || {}, updates=d.updates || [], episodes=d.episodes || [];
    const names={outcome:'Итог попытки',progress:'Продвижение',time_cost:'Время',input_cost:'Нажатия',control_bonus:'Бонус за допустимый класс',unsupported_penalty:'Штраф за недопустимый класс'};
    const label=task=>data.task_guides?.[task]?.label || task;
    const actionRows=episodes.filter(e=>e.action_summary || e.control_events || e.reward_totals);
    const versions=[...new Set(episodes.map(e=>e.reward_composition?.version).filter(Boolean))];
    const version=d.reward_composition?.version || (versions.length===1?versions[0]:null);
    const controlCopy=version==='carryrl.reward.outcome_potential_text.v2'?'Штраф за недопустимый ввод начисляется один раз за исполненную команду: 0,01 × R / H, где R — награда за победу, H — предел шагов эпизода. Базовый бонус 0,0005 × R и доплата 0,0015 × R за подтверждённый режим выдаются один раз на группу синонимов; общий предел бонуса — 1% R. Удержание: два последовательных шага; нажатие: один шаг, затем отпускание. Временное расписание: полная сила в циклах 1–32, затем снижение до нуля в цикле 64.':version==='carryrl.reward.outcome_potential_text.v1'?'Бонус и штраф учитываются один раз за класс управления в эпизоде; повторные нажатия и синонимы не дают повторного бонуса.':'Версия композиции награды не сохранена; правила начисления не подтверждены.';
    return `<details id="rl-diagnostics" class="panel"><summary>Как награда влияет на обучение</summary><p>Бонус за допустимую кнопку означает, что её класс предусмотрен текстом управления. Это не оценка хорошего действия в текущей ситуации. ${controlCopy} Удержание и отпускание видны отдельно в частотах.</p><p class="note">Продвижение, итог игры и затраты времени — отдельные части награды. Изменения параметров подтверждают работу оптимизатора; рост побед на тех же отложенных уровнях подтверждает улучшение игры. Остановку процесса и ошибки смотрите в состоянии запуска выше.</p>${finite(d.control_coefficient)?`<p>Множитель бонуса управления: <strong>${nll(d.control_coefficient)}</strong>.</p>`:''}${actionRows.length?`<h3>Действия и части награды · последний завершённый цикл</h3><div class="table-wrap"><table><thead><tr><th scope="col">Игра</th><th scope="col">Части награды · сумма</th><th scope="col">Исполненные действия</th><th scope="col">События управления</th></tr></thead><tbody>${actionRows.map(e=>`<tr><th scope="row">${esc(label(e.task))}</th><td>${Object.entries(e.reward_totals || {}).map(([k,v])=>`${esc(names[k] || k)}: ${nll(v)}`).join('<br>') || 'Нет измерений'}</td><td>${e.action_summary?`${num(e.action_summary.steps)} команд; удержаний: ${num(e.action_summary.held)}; отпусканий: ${num(e.action_summary.released)}${finite(e.action_summary.unsupported_inputs)?`; недопустимых вводов: ${num(e.action_summary.unsupported_inputs)}`:''}<small>${Object.entries(e.action_summary.frequencies || {}).map(([k,v])=>`${esc(k)}: ${num(v)}`).join(' · ') || 'Без активных нажатий'}</small>`:'Нет измерений'}</td><td>${controlEvents(e)}</td></tr>`).join('')}</tbody></table></div><p class="note">Удержания — число кнопок, оставшихся нажатыми между соседними командами; отпускания — число снятых кнопок. Частоты — команды с каждой кнопкой или движением. События управления показывают только сохранённые счётчики и причины.</p>`:empty('Части награды и частоты действий в этом запуске не сохранены.')}${Object.keys(a.tasks || {}).length?`<h3>Сигнал для обновления по играм</h3><p class="note">Advantage показывает, какие действия оказались лучше ожидания модели. Нормализация меняет масштаб этого сигнала. Это не доля побед.</p><div class="table-wrap"><table><thead><tr><th scope="col">Игра</th><th scope="col">Исходный сигнал · среднее / разброс</th><th scope="col">После нормализации · среднее / разброс</th><th scope="col">Изменения знака без бонуса и штрафа</th></tr></thead><tbody>${Object.entries(a.tasks).map(([task,row])=>`<tr><th scope="row">${esc(label(task))}</th><td>${nll(row.raw_advantage?.mean)} / ${nll(row.raw_advantage?.std)}</td><td>${nll(row.normalized_advantage?.mean)} / ${nll(row.normalized_advantage?.std)}</td><td>${cf.status==='measured'?`${num(cf.tasks?.[task]?.sign_changes)}<small>Средний прямой вклад: ${nll(cf.tasks?.[task]?.direct_centered_effect?.mean)}<br>Вклад смены масштаба: ${nll(cf.tasks?.[task]?.denominator_rescaling_effect?.mean)}</small>`:'Не измерено'}</td></tr>`).join('')}</tbody></table></div><p class="note">Проба без бонуса и штрафа пересчитывает сигнал на тех же эпизодах с прежними оценками ценности. Она не измеряет, как играла бы отдельно обученная модель.</p>`:empty('Диагностика сигнала обучения не сохранена.')}${Array.isArray(d.updates)?`<p>Реально применённых обновлений за цикл: <strong>${num(updates.length)}</strong>. Отклонённые по KL пакеты сюда не входят.</p>`:empty('Список применённых обновлений не сохранён.')}${updates.length?`<details id="applied-updates"><summary>Измерения применённых обновлений</summary><p class="note">Последние 50 обновлений последнего завершённого цикла. KL — насколько изменилась политика; энтропия — разнообразие действий; clipping — доля ограниченных отношений вероятностей. Коэффициент градиента меньше 1 означает уменьшение шага из-за большой нормы.</p><div class="table-wrap"><table><thead><tr><th>Обновление</th><th>KL</th><th>Энтропия</th><th>Доля clipping</th><th>Норма градиента</th><th>Коэффициент градиента</th></tr></thead><tbody>${updates.slice(-50).map(u=>`<tr><td>${num(u.update)}</td><td>${nll(u.approx_kl)}</td><td>${nll(u.entropy)}</td><td>${nll(u.clip_fraction)}</td><td>${nll(u.gradient_norm)}</td><td>${nll(u.clip_coefficient)}</td></tr>`).join('')}</tbody></table></div></details>`:''}${d.gradient_probes?.length?`<h3>Периодическая проба градиентов</h3><p class="note">Проба измеряет вклад обучения действий, оценки ценности и разнообразия действий в изменение параметров. Показана последняя сохранённая проба; измерения выполняются с заданным интервалом. Её цикл может предшествовать последнему циклу обучения.</p><div class="table-wrap"><table><thead><tr><th>Цикл</th><th>Политика / ценность / разнообразие</th><th>KL до / после</th><th>Коэффициент ограничения</th></tr></thead><tbody>${d.gradient_probes.map(p=>`<tr><td>${num(p.cycle)}</td><td>${nll(p.component_gradient_norms?.actor)} / ${nll(p.component_gradient_norms?.value)} / ${nll(p.component_gradient_norms?.entropy)}</td><td>${nll(p.pre_update_kl)} / ${nll(p.post_update_kl)}</td><td>${nll(p.clip_coefficient)}</td></tr>`).join('')}</tbody></table></div>`:''}${episodes.some(e=>e.control_table)?`<details id="control-predictions"><summary>Как распознан текст управления</summary>${episodes.filter(e=>e.control_table).map((e,i)=>`<details id="control-table-${i}"><summary>${esc(label(e.task))}</summary>${controlTable(e,i)}</details>`).join('')}</details>`:''}${technicalDetails('rl-detail-source',[['Источник','training.jsonl'],['Цикл',c.training_outcomes?.rollout],['Байты сохранённых данных цикла',d.rollout_storage_bytes]])}</details>`;
  }
  function guardResources(c) {
    const g=c?.resource_guard || {}, m=g.sample || {};
    const gib=v=>finite(v)?`${nll(v/1073741824)} GiB`:'Нет замера';
    const fields=[['RSS группы процессов',m.process_group_rss_bytes],['Пик RSS',g.peak_rss_bytes],['Лимит RSS',g.maximum_rss_bytes],['Свободно на диске',m.disk_free_bytes],['Минимум свободного места',g.minimum_free_bytes],['Порог свободного места',g.minimum_free_bytes_limit],['Доступно RAM системы',m.system_available_bytes],['Использовано swap системы',m.system_swap_used_bytes ?? m.swap_used_bytes],['Размер запуска на диске',m.run_bytes]];
    if(!fields.some(([,v])=>finite(v)))return '';
    return `<div class="guard-resources"><h3>Ресурсы процесса и компьютера</h3><p class="note">Снимок внешнего наблюдателя: ${date(m.utc)}. RSS — оперативная память, занятая группой процессов. Измерения включают группу процессов ${c?.evaluation_only?'проверки':'обучения'}.</p><dl class="resource-values">${fields.filter(([,v])=>finite(v)).map(([label,v])=>`<div><dt>${esc(label)}</dt><dd>${gib(v)}</dd></div>`).join('')}</dl><p class="note">GiB — единица объёма памяти, примерно миллиард байт. Swap — данные из оперативной памяти, временно перенесённые на диск.</p>${finite(m.system_memory_pressure_level)?`<p class="note">Системный уровень давления памяти: ${num(m.system_memory_pressure_level)}.</p>`:''}</div>`;
  }
  function ppoDiagnostics(q) {
    return (q?.diagnostic_series || []).map((series,i)=>`<details id="ppo-diagnostic-${i}"><summary>Отдельная проверка · ${series.coverage==='full'?'полный набор':'выборка'}</summary><p class="note diagnostic-identity">${esc(series.description)}</p>${technicalDetails(`ppo-protocol-${i}`,[['Протокол',series.protocol_sha256]])}<div class="table-wrap"><table><thead><tr><th>Цикл</th><th>Ревизия</th><th>Победы</th><th>Эпизоды</th></tr></thead><tbody>${series.history.map(p=>`<tr><td>${num(p.rollout)}</td><td>${num(p.revision)}</td><td>${pct(p.value)}</td><td>${counts(p)}</td></tr>`).join('')}</tbody></table></div></details>`).join('');
  }
  function perfectScore(q) {
    const measured=finite(q?.perfect_score_successes)&&finite(q?.perfect_score_percent);
    return `<div class="perfect-score"><div class="help-heading"><h3>Победы с максимальным счётом${q?.perfect_score_measurement_stage==='initial'?' до этого обучения':''}</h3>${metricHelp('perfect','full-perfect')}</div>${measured?`<p><strong>${pct(q.perfect_score_percent)}</strong> · ${num(q.perfect_score_successes)} из ${num(q.perfect_score_episodes ?? q.episodes)} эпизодов с победой и счётом 5/5.</p>${q.perfect_score_run_id?`<p class="note">${esc(experimentTitle(q.perfect_score_measurement_run_id || q.perfect_score_run_id))}. ${esc(q.perfect_score_scope || 'Собственная проверка выбранного эксперимента')}. Условия этой проверки могут отличаться от основной доли побед.</p>${technicalDetails('perfect-score-identity',[['Запуск',q.perfect_score_measurement_run_id || q.perfect_score_run_id],['Версия модели',q.perfect_score_revision],['Файл проверки',q.perfect_score_measurement_source || q.perfect_score_evaluation_file],['Версия правил счёта',q.scoring_version]])}`:''}`:empty('Отдельное измерение побед со счётом 5/5 пока недоступно.')}<details class="scoring-rules"><summary>Правила счёта и ограничения</summary><p class="note">${esc(q?.scoring_version?'Счёт учитывает управляющие действия агента.':q?.scoring_version_label || (finite(q?.episodes)?'Прежние правила счёта: результаты 5/5 нельзя сравнивать с текущими.':'Правила счёта и результат появятся после проверки.'))}</p>${q?.scoring_version?'<p class="note">В лабиринте 5/5 означает кратчайший маршрут без лишних нажатий. В остальных 15 играх — победу без лишних управляющих действий; это не доказывает, что весь путь был оптимальным.</p>':''}</details></div>`;
  }
  function quality(q, full=true, c=data.current) {
    if(c?.evaluation_only)return evaluationResult(q,c);
    if (!q?.available && !finite(q?.value)) return `<section class="panel"><h2>Результат после обучения</h2>${empty('Проверка качества ещё не завершена. Ошибка повторения действий не доказывает, что агент научился побеждать.')}${perfectScore(q)}${ppoDiagnostics(q)}</section>`;
    const stage=qualityStage(q,c);
    const title=stage==='initial'?'До этого обучения':stage==='inherited'?'Проверка предыдущей модели':stage==='previous'?'Последняя проверка сохранённой модели':'Победы на 16 мини-играх';
    const pending=stage!=='final'?`<p class="note result-pending">${stage==='previous'?'Текущие веса ещё не проверены. Показан результат ранее сохранённой модели.':'Результат после этого обучения ещё не получен. Показанный замер не подтверждает улучшение в выбранном эксперименте.'}</p>`:'';
    return `<section class="panel"><div class="panel-top"><h2>${title}</h2><a href="#/games">Результаты по играм</a></div>${pending}<div class="quality-summary"><strong class="quality-value">${pct(q.value)}</strong>${metricHelp('wins','quality-wins')}${stage==='final'&&q.comparison?.comparable!==false?change(q.change_pp):''}</div>${metricIdentity(q,c)}<p class="note">${counts(q)}.${stage==='final'&&q.comparison?.comparable!==false&&finite(q.baseline)?` Начальная модель: ${pct(q.baseline)}.`:''}</p><details id="quality-score"><summary>Счёт и правила оценки</summary>${perfectScore(q)}</details>${full&&stage==='final'?chart(q.run_history || q.history,'value','Доля побед',true):''}<p class="note">${esc(q.protocol_description || 'Фиксированный контрольный набор: 16 мини-игр по 16 эпизодов, начальная сложность. Результат относится к этим условиям.')}</p>${q.inherited_from?`<p class="note">${esc(q.canonical_measurement_current?'Исходная модель для сравнения':'Показана предыдущая модель')}: ${esc(experimentTitle(q.inherited_from.run_id,c))}. Это прежний замер.${stage!=='final'?' Текущая модель после обучения ещё не проверена.':''}</p>`:''}${ppoDiagnostics(q)}${q.quick?`<p class="note">Промежуточная проверка: ${pct(q.quick.value)}, ${counts(q.quick)}. Она не заменяет полную проверку.</p>`:''}</section>`;
  }
  function recordingControls(source, task, filtered, selected) {
    const runId=source.current?.run_id || source.run_id || filtered[0]?.run_id;
    const runItems=(source.playbacks?.items || []).filter(p=>p.run_id===runId);
    const tasks=[...new Set(runItems.map(p=>p.task))];
    const totalPages=Math.max(1,Math.ceil(filtered.length/RECORDINGS_PER_PAGE));
    recordingPage=Math.min(recordingPage,totalPages-1);
    const items=filtered.slice(recordingPage*RECORDINGS_PER_PAGE,(recordingPage+1)*RECORDINGS_PER_PAGE);
    return `<div class="recording-filters">${!task&&tasks.length>1?`<div class="filter-field"><label for="recording-task">Игра в записи</label><select id="recording-task">${tasks.map(id=>`<option value="${esc(id)}" ${id===recordingTask?'selected':''}>${esc(source.task_guides?.[id]?.label || data.task_guides?.[id]?.label || id)}</option>`).join('')}</select></div>`:''}${filtered.length>1?`<div class="filter-field recording-choice"><label for="recording">Запись</label><select id="recording">${items.map(p=>`<option value="${esc(p.id)}" ${p.id===selected?.id?'selected':''}>${esc(checkpointLabel(p))} · ${date(p.created_at)} · ${esc(recordingOutcome(p) || 'Исход неизвестен')}</option>`).join('')}</select></div>`:''}</div>${totalPages>1?`<div class="recording-pages"><button id="recording-prev" type="button" ${recordingPage===0?'disabled':''}>Более новые</button><span>Записи ${recordingPage*RECORDINGS_PER_PAGE+1}–${Math.min(filtered.length,(recordingPage+1)*RECORDINGS_PER_PAGE)} из ${num(filtered.length)}</span><button id="recording-next" type="button" ${recordingPage===totalPages-1?'disabled':''}>Более ранние</button></div>`:''}`;
  }
  function recordingGuide(task,source=data) {
    const guide=source.task_guides?.[task] || data.task_guides?.[task];
    if(!guide)return '';
    return `<div class="recording-guide">${route().page==='games'&&route().id===task?'':`<p><strong>Цель:</strong> ${esc(guide.objective)}</p>`}${guide.legend?.length?`<details id="legend-${esc(task)}"><summary>Что на экране</summary><ul>${guide.legend.map(item=>`<li>${esc(item)}</li>`).join('')}</ul><p class="note">В другом оформлении оттенки и текстуры могут отличаться.</p></details>`:''}<details id="controls-${esc(task)}"><summary>Управление в записи</summary><p>${esc(Array.isArray(guide.controls)?guide.controls.join(', '):guide.controls)}</p></details></div>`;
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
    if(!selected)return `<section class="panel recording-panel"><h2>Запись игры</h2>${runItems.length?controls:`<p class="note">${esc(pickerLabel(source.current))}</p>`}<div class="preview empty">${runItems.length?'Для этой игры записей в выбранном запуске нет.':'В этом запуске не сохранено записей игр.'}</div><p><a href="#/history">Выбрать другой запуск в архиве</a></p>${runItems.length?'<p><a href="#/games">Посмотреть другие игры</a></p>':''}</section>`;
    selectedRecording=selected.id;
    const safeUrl=typeof selected.url==='string'&&/^\/(?:media|replay-media)\//.test(selected.url)?selected.url:'';
    const sprite=typeof selected.sprite_url==='string'&&/^\/(?:media|replay-media)\//.test(selected.sprite_url)&&finite(selected.frame_count)&&selected.frame_count>0&&selected.frame_width>0&&selected.frame_height>0;
    const version=validation?.checkpoints?.find(c=>c.id===selected.checkpoint_id);
    const level=finite(selected.level_index)?`Уровень ${num(selected.level_index+1)}`:'Номер уровня не сохранён';
    const meta=`${selected.reference_match===false?'<p class="warning">Ручная проба отличается от опубликованной проверки. Она не заменяет её результат.</p>':''}${selected.notes?.length?`<p class="note">${selected.notes.map(esc).join(' ')}</p>`:''}${recordingGuide(selected.task,source)}<div class="playback-meta"><span>Записано ${date(selected.created_at)}</span><span>${finite(selected.steps)?`${num(selected.steps)} действий`:''}</span><strong>${esc(recordingOutcome(selected))}</strong></div><p class="note">${selected.manual?'Ручная проба. ':''}Одна запись показывает одну попытку и не добавляется в общую долю побед.${recordingOutcome(selected)==='Лимит ходов'?' Лимит ходов: игра остановлена после допустимого числа действий без победы или поражения.':''}</p>${technicalDetails('recording-identity',[['Запуск',selected.run_id],['Версия модели',selected.revision],['Проход данных',selected.epoch],['Сохранённая модель',selected.checkpoint],['Запись',selected.id]])}`;
    const heading=`<h2>Запись игры</h2><p class="recording-context">${esc(level)} · ${esc(checkpointLabel(version || selected))}<br>${esc(modeLabel(selected.mode))} · ${esc(appearanceLabel(selected.appearance))}</p>${controls}`;
    if(sprite)return spritePlayback(selected,safeUrl,heading,meta);
    const hideMedia=hiddenRecording;
    return `<section class="panel recording-panel">${heading}<div class="preview ${!safeUrl||hideMedia?'empty':''}">${safeUrl&&!hideMedia?`<img data-recording="${esc(selected.id)}" src="${esc(safeUrl)}" alt="Запись игры ${esc(selected.label || data?.task_guides?.[selected.task]?.label || selected.task)}" loading="eager">`:esc(hideMedia?'Запись скрыта. Нажмите «Показать запись», чтобы продолжить просмотр.':'Файл записи недоступен.')}</div><div class="controls"><button id="toggle-recording" type="button">${hideMedia?'Показать запись':'Скрыть запись'}</button>${safeUrl?`<a href="${esc(safeUrl)}" target="_blank" rel="noopener">Открыть запись</a>`:''}</div><p class="record-label">Анимированная запись одной попытки.</p>${meta}</section>`;
  }
  function spritePlayback(selected,gifUrl,heading,meta) {
    let player=players.get(selected.id);
    if(!player){player={id:selected.id,item:selected,frame:0,playing:!reducedMotion(),last:0,image:null,error:false};players.set(selected.id,player);}
    player.item=selected;
    return `<section class="panel recording-panel">${heading}<div class="preview"><canvas id="game-canvas" data-recording="${esc(selected.id)}" width="${Number(selected.frame_width)}" height="${Number(selected.frame_height)}" aria-label="Запись игры ${esc(selected.label || data?.task_guides?.[selected.task]?.label || selected.task)}">Запись игры. ${gifUrl?`<a href="${esc(gifUrl)}">Открыть запись</a>`:''}</canvas></div><div class="player-controls"><button id="play-pause" type="button">${player.playing?'Пауза':'Продолжить'}</button><button id="restart-playback" type="button">С начала</button><label class="sr-only" for="frame-seek">Кадр записи</label><input id="frame-seek" type="range" min="0" max="${selected.frame_count-1}" value="${player.frame}" step="1"><output id="frame-counter" for="frame-seek">${player.frame+1} / ${num(selected.frame_count)}</output></div><p id="action-caption" class="note"></p>${meta}${gifUrl?`<p class="note"><a href="${esc(gifUrl)}" target="_blank" rel="noopener">Открыть запись</a></p>`:''}</section>`;
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
    if(player.error){const host=canvas.closest('.preview');host.classList.add('empty');host.textContent='Не удалось загрузить кадры записи. Откройте запись по ссылке ниже.';return;}
    if(!player.image){
      player.image=new Image();player.image.onload=()=>drawPlayer(player);
      player.image.onerror=()=>{player.error=true;const current=document.getElementById('game-canvas');if(current?.dataset.recording===player.id){const host=current.closest('.preview');host.classList.add('empty');host.textContent='Не удалось загрузить кадры записи. Откройте запись по ссылке ниже.';}};
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
    const groups=c?.settings || {}, cfg=groups.training || {}, model=groups.config || {};
    const fields=[['Метод',c?.technical_only?'Техническая работа':c?.evaluation_only?'Проверка без обучения':c?.method==='ppo'?'PPO · обучение по награде':c?.method==='behavioural_cloning_joint_action_nll'?'BC · повторение действий':'Не указан'],['Вычисления',cfg.training_backend || model.training_backend],['Скорость обучения',cfg.learning_rate ?? model.learning_rate],['Эпизодов в пакете',cfg.minibatch_episodes],['Запланировано эпох',cfg.epochs],['Запланировано циклов',c?.method==='ppo'?model.total_rollouts:null],['Проверка каждые N эпох',cfg.eval_every],['Сложность валидации',cfg.evaluation_difficulty],['Лимит времени',finite(cfg.max_wall_seconds)?duration(cfg.max_wall_seconds):null]];
    const items=fields.filter(([key,v])=>v!==undefined&&v!==null&&(!(c?.evaluation_only||c?.technical_only)||['Метод','Вычисления'].includes(key)));
    const raw=Object.entries(groups).flatMap(([group,values])=>values&&typeof values==='object'?Object.entries(values).map(([k,v])=>[`${group}.${k}`,v]):[[group,values]]);
    return `<section class="panel"><h2>${c?.technical_only?'Параметры технического запуска':c?.evaluation_only?'Параметры проверки':'Параметры обучения'}</h2><dl class="settings readable-settings">${items.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl><details id="technical-settings"><summary>Все сохранённые параметры</summary>${raw.length?`<dl class="settings">${raw.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(typeof v==='object'?JSON.stringify(v):v)}</dd>`).join('')}</dl>`:empty('Настройки запуска не сохранены.')}</details></section>`;
  }
  function trainingStatus() {
    const active=data.training_current;
    if(active?.running&&active.run_id!==data.current?.run_id)return `<aside class="active-training"><h2>${active.evaluation_only?'Сейчас проверяется':'Сейчас обучается'}: ${esc(pickerLabel(active))}</h2><p class="note">Результаты выбранной модели ниже относятся к отдельной проверке.</p>${active.evaluation_only?evaluationProgress(active):progress(active)}</aside>`;
    return progress(data.current);
  }
  function preparationProcess(p) {
    const stage=p.preparation_stage;
    if(!stage)return '';
    const name=({reward_fit:'обучение распознавателя награды',reward_validation:'проверка награды',model_initialization:'создание начальных весов'})[stage.kind] || 'подготовка';
    const state=stage.running?'Выполняется':({completed:'Завершено',failed:'Ошибка',error:'Ошибка',interrupted:'Процесс завершился без подтверждённого результата',exited:'Процесс завершился; результат не подтверждён',unknown:'Активность процесса не подтверждена'})[stage.status] || stage.status;
    const sample=stage.resource_sample || {};
    const fields=[['RSS группы процессов',sample.process_group_rss_bytes],['Свободно на диске',sample.disk_free_bytes],['Пик RSS',stage.peak_rss_bytes],['Минимум свободного места',stage.minimum_free_bytes]].filter(([,value])=>finite(value));
    return `<section class="panel preparation-process"><h2>${stage.running?'Сейчас':'Последний этап'}: ${esc(name)}</h2><p>${esc(state)}.${finite(stage.exitcode)?` Код завершения: ${num(stage.exitcode)}.`:''}</p>${fields.length?`<p class="note">Последние сохранённые измерения: ${date(sample.utc)}.</p><dl class="resource-values">${fields.map(([label,value])=>`<div><dt>${esc(label)}</dt><dd>${nll(value/1073741824)} GiB</dd></div>`).join('')}</dl>`:empty('Измерения ресурсов этого этапа пока не сохранены.')}</section>`;
  }
  function rewardProgress(p,id) {
    if(!p || typeof p!=='object')return '';
    const stage={input_preflight:'Проверка входных данных',features:'Подготовка признаков',fit:'Обучение оценщика',calibration_gate:'Проверка на калибровочном наборе',sealed_test:'Проверка на закрытом тестовом наборе',epoch_complete:'Эпоха завершена',public_runtime:'Проверка 32 игровых инструкций',complete:'Вычисления завершены',unknown:'Этап не подтверждён',pytest:'Проверка тестов'};
    const status={running:'В работе',completed:'Завершено',failed:'Ошибка',starting:'Запускается',unknown:'Состояние не подтверждено'};
    const fields=[p.stage?`Этап: ${stage[p.stage] || p.stage}`:null,p.status?`Состояние: ${status[p.status] || p.status}`:null,
      finite(p.epoch)?`Эпоха: ${num(p.epoch)}${finite(p.total_epochs)?` / ${num(p.total_epochs)}`:''}`:null,
      finite(p.optimizer_updates)?`Обновлений: ${num(p.optimizer_updates)}`:null,
      finite(p.train_examples)?`TRAIN-примеров: ${num(p.train_examples)}`:null].filter(Boolean);
    return `${fields.length?`<small class="reward-progress">${fields.map(esc).join(' · ')}</small>`:''}${finite(p.loss)?technicalDetails(`reward-metrics-${id}`,[['Ошибка обучения (loss)',nll(p.loss)],['Интерпретация','Ошибка обучения сама по себе не подтверждает качество оценщика или игры.']]):''}`;
  }
  function cycleCard() {
    const c=data.cycle;
    if(!c)return '';
    const b=c.budget;
    const state=c.memory_paused?'Пауза по памяти':c.running?'Процесс подтверждён':({prepared:'Подготовлен',completed:'Завершён',stopped:'Остановлен',failed:'Ошибка',error:'Ошибка',interrupted:'Прерван',stale:'Данные устарели',unknown:'Активность не подтверждена'})[c.status] || 'Активность не подтверждена';
    const rewardName=id=>{
      const labels={text_data:'Подготовка текстовых примеров',visual_fit:'Обучение визуального оценщика',visual_gate:'Проверка визуального оценщика',text_fit:'Обучение текстового оценщика',text_sealed:'Независимая проверка текста',text_gate:'Проверка текстового оценщика',text_runtime:'Проверка текста управления',text_cache_cleanup:'Очистка временных текстовых признаков',history_fit:'Обучение оценщика истории',history_gate:'Проверка оценщика истории'};
      const factorized=id.match(/^text_fit_factorized_v2(?:_corpus_r([1-9][0-9]*))?_(micro|probe|full|checks)$/);
      if(factorized){
        const name={micro:'Короткая TRAIN-проверка',probe:'Проба на полном корпусе · 8 эпох',full:'Полное обучение текстового оценщика',checks:'Проверка исходников перед обучением'}[factorized[2]];
        return `${name}${factorized[1]?` · R${factorized[1]}`:''}`;
      }
      if(id==='visual_fit_entrypoint_fix')return `${labels.visual_fit} · повтор после исправления запуска`;
      const base=Object.keys(labels).find(key=>id===key || id.startsWith(`${key}_`));
      if(!base)return 'Этап подготовки награды';
      if(id!==base)return `${labels[base]} · повтор`;
      const retried=c.rewards.some(row=>row.id.startsWith(`${base}_`));
      return `${labels[base]}${retried?' · первая попытка':''}`;
    };
    const rewardState=r=>Number.isSafeInteger(r.actual_exit_code)&&r.actual_exit_code>0?`Ошибка · код ${num(r.actual_exit_code)}`:r.actual_exit_code===0&&r.quality_check_passed===false?'Проверка качества не пройдена':r.confirmed?'Завершение подтверждено':r.actual_exit_code===0&&['failed','stopped','error'].includes(r.status)?'Процесс завершён; результат не принят':({starting:'Запускается',running:'В работе',stopped:'Остановлено',failed:'Ошибка',completed:'Результат не подтверждён'})[r.status] || 'Ожидается';
    const rewards=c.rewards || [], current=rewards.find(r=>r.id===c.stage_id);
    const historical=c.stage_id?rewards.filter(r=>r!==current&&!['pending','starting','running'].includes(r.status)):[];
    const remaining=c.stage_id?rewards.filter(r=>r!==current&&!historical.includes(r)):rewards;
    const rewardRow=r=>`<li><span>${esc(rewardName(r.id))}</span><strong>${esc(rewardState(r))}</strong>${rewardProgress(r.progress,r.id)}</li>`;
    const currentLabel=c.running&&['starting','running'].includes(current?.status)?'Текущий этап':'Последний этап';
    const rewardList=`${current?`<div id="cycle-current-stage"><h3>${currentLabel}</h3><ul class="cycle-rewards">${rewardRow(current)}</ul></div>`:''}${remaining.length?`<ul class="cycle-rewards">${remaining.map(rewardRow).join('')}</ul>`:''}${historical.length?`<details id="cycle-reward-history"><summary>Предыдущие этапы (${num(historical.length)})</summary><ul class="cycle-rewards">${historical.map(rewardRow).join('')}</ul></details>`:''}`;
    return `<section id="rl-cycle" class="panel cycle-card" aria-label="Новый цикл обучения"><div class="run-heading"><h2>Новый цикл · 24 часа</h2><span>${esc(state)}</span></div><p><strong>${esc(c.phase_label)}</strong>${c.policy_created?'':' · Модель политики ещё не создана.'}</p><div class="facts"><div class="fact"><span>Прошло с начала вычислений</span><strong>${b?duration(b.elapsed_seconds):'Не подтверждено'}</strong></div><div class="fact"><span>${b?.frozen?'Оставалось при остановке':'Осталось в общем бюджете'}</span><strong>${b?duration(b.remaining_seconds):'Не подтверждено'}</strong><small>${b?.frozen?'Отсчёт зафиксирован.':b?.started?'Включая подготовку, проверки и паузы.':'Вычисления ещё не начались.'}</small></div><div class="fact"><span>${c.terminal_result_saved?'Состояние результата':'Свежесть состояния'}</span><strong>${c.terminal_result_saved?'Итог сохранён':duration(c.age_seconds)}</strong><small>${c.terminal_result_saved?(c.status==='completed'?'Цикл завершён; обновления не ожидаются.':'Сохранено при остановке; обновления не ожидаются.'):c.stale?'Данные устарели.':'Сохранённые данные актуальны.'}</small></div></div>${rewardList}${c.failure_reason?`<p class="warning">${esc(c.failure_reason)}</p>`:''}${c.budget_valid===false?'<p class="warning">Сохранённые границы общего бюджета не согласованы.</p>':''}${technicalDetails('cycle-source',[['План',c.plan_id],['Текущий этап',c.stage_id],['Техническая причина остановки',c.failure_details],['Код завершения этапа',c.actual_exit_code],['Выбранное расписание',c.selected_schedule],['Основных циклов PPO',c.full_rollouts]])}</section>`;
  }
  function cyclePreparation() {
    return `<h1>Подготовка нового агента</h1><p class="subtitle">Сначала готовятся и проверяются модели награды. Результаты политики появятся после её создания и игровой проверки.</p>${cycleCard()}<p><a href="#/history">Предыдущие модели и записи в архиве</a></p>`;
  }
  function preparation() {
    const p=data.comparison_plan,b=p.budget || {},observed=p.training_observed===true;
    const stage=value=>({published:'Файл отчёта опубликован; проверка готовности перед запуском',failed:'Проверка не пройдена',pending:'Ожидается'})[value] || 'Ожидается';
    return `<div class="overview-heading"><h1>Подготовка сравнения 16 / 32 / 64 токенов</h1><p>Новый эксперимент с обучением по награде. ${observed?'Найдены файлы запусков вариантов; их состояние доступно по ссылкам ниже.':'Обучение политики ещё не запущено.'}</p></div>${preparationProcess(p)}${p.state==='unavailable'?'<aside class="warning" role="alert">Не удалось прочитать план сравнения. Проверьте технические сообщения ниже.</aside>':''}<section class="panel"><h2>План эксперимента</h2><p>Три варианта получают одинаковый бюджет взаимодействия со средой. Готовность к запуску ещё не подтверждена.</p><p>Запланировано на вариант: ${num(b.rollouts_per_arm)} циклов, ${num(b.environment_episodes_per_arm)} эпизодов. Обновлений политики по сохранённым счётчикам: ${num(p.policy_updates)}.</p><div class="table-wrap"><table><thead><tr><th scope="col">Пространственных токенов</th><th scope="col">Начальные веса</th><th scope="col">Обучение</th></tr></thead><tbody>${(p.arms || []).map(a=>`<tr><th scope="row">${num(a.spatial_tokens)}</th><td>${esc(stage(a.initialization))}</td><td>${a.run_id?`<a href="#/history/${encodeURIComponent(a.run_id)}">Открыть сохранённый запуск</a>`:'Не запущено'}</td></tr>`).join('')}</tbody></table></div></section><section class="panel"><h2>Что осталось подготовить</h2><dl class="settings"><dt>Новая модель награды</dt><dd>${esc(stage(p.reward))}</dd><dt>Независимая проверка награды</dt><dd>${esc(stage(p.reward_validation))}</dd><dt>Общие начальные веса</dt><dd>${esc(stage(p.initialization))}</dd><dt>Калибровка общего расписания</dt><dd>${esc(stage(p.calibration))}</dd></dl><p class="note">Файлы отчётов и начальных весов сами по себе не подтверждают готовность: перед запуском нужны проверки происхождения, целостности и одинаковых условий.</p></section>${observed?'<section class="panel"><h2>Нагрузка на компьютер</h2><p>Откройте конкретный запуск для просмотра сохранённых измерений.</p></section>':resources(null)}<p><a href="#/history">Архив предыдущих запусков и записей</a></p>`;
  }
  function activity() {
    const c=data.training_current;
    if(data.cycle){const cycle=data.cycle;return `<aside id="activity" class="activity ${cycle.running?'live':cycle.stale?'unknown':'idle'}" aria-label="Текущий процесс" aria-live="polite"><div><strong>${esc(cycle.phase_label)}${cycle.running?' · процесс подтверждён':''}</strong><span>${cycle.policy_created?'Новый цикл обучения.':'Модель политики ещё не создана.'}${cycle.stale?' Сохранённые данные устарели.':''}</span></div><a href="#/training">Состояние нового цикла</a></aside>`;}
    if(data.comparison_plan&&!c)return `<aside id="activity" class="activity idle" aria-label="Текущий процесс" aria-live="polite"><div><strong>${data.comparison_plan.training_observed?'Сравнение · найдены сохранённые запуски':'Подготовка сравнения · обучение не запущено'}</strong><span>План 16 / 32 / 64 пространственных токенов. ${data.comparison_plan.training_observed?'Состояние процесса смотрите в конкретном запуске.':'Результаты появятся после реальных запусков.'}</span></div><a href="#/training">Этапы подготовки</a></aside>`;
    const live=c?.running===true;
    const unknown=!c || ['unknown','stale'].includes(c.status);
    const checking=c?.evaluation_only===true;
    const validEpisodes=Number.isSafeInteger(c?.phase_units_done)&&Number.isSafeInteger(c?.phase_units_total)&&c.phase_units_done>=0&&c.phase_units_total>0&&c.phase_units_done<=c.phase_units_total;
    const amount=checking?(validEpisodes?`${num(c.phase_units_done)} / ${num(c.phase_units_total)} эпизодов · `:''):(finite(c?.training_percent)?`${num(c.training_percent)}% ${c.method==='ppo'?'циклов':'обновлений'} · `:'');
    return `<aside id="activity" class="activity ${live?'live':unknown?'unknown':'idle'}" aria-label="Текущий процесс" aria-live="polite"><div><strong>${live?(c.stale?'Процесс работает, данные задерживаются':checking?'Проверка выполняется':'Обучение идёт'):unknown?'Активность не подтверждена':checking?'Сейчас проверка не запущена':'Сейчас обучение не запущено'}</strong><span>${live?`${esc(checking&&c.job_kind==='paired_frozen_evaluation'?'Парная проверка навигации':phaseLabel(c.phase_label))} · ${amount}последние данные ${date(c.last_update_at)}`:unknown?'Нет свежего подтверждения работающего процесса.':'Можно изучать сохранённые результаты и записи.'}</span></div>${c?`<button type="button" data-open-training="${esc(c.run_id)}">${live?(checking?'Следить за проверкой':'Следить за обучением'):(checking?'Последняя проверка':'Последнее обучение')}</button>`:''}</aside>`;
  }
  function comparison(q=data.quality) {
    const comparable=q?.comparison?.comparable ?? (qualityStage(q)==='final'&&finite(q?.change_pp));
    return `<div class="comparison ${comparable?'comparable':''}"><strong>${comparable?'Сравнение на одинаковых проверочных уровнях':'Сравнение недоступно'}</strong> ${metricHelp('change','comparison-change')}<p>${esc(q?.comparison?.reason || (comparable?'Сопоставлены проверки до и после обучения: те же уровни, сложность, правила и режим игры.':'Нужны две проверки на одинаковых уровнях и по одним правилам. Результаты разных наборов уровней не показывают улучшение модели.'))}${q?.comparison?.excluded_measurements_count?` Отдельно исключено проверок с другими условиями: ${num(q.comparison.excluded_measurements_count)}.`:''}</p>${comparable?pairedOutcomes(q):''}</div>`;
  }
  function pairedOutcomes(q) {
    const p=q?.paired_outcomes;
    if(!p)return `<p class="note">Парные исходы не сохранены${q?.paired_outcomes_reason?`: ${esc(q.paired_outcomes_reason)}`:'.'} Это не означает отсутствие потерь навыков.</p>`;
    const rows=p.games || [];
    return `<div class="paired-outcomes"><p><strong>Новых побед: ${num(p.gained)} · Потерянных: ${num(p.lost)}</strong></p><p>Сохранённых побед: ${num(p.retained_wins)}. Без победы в обеих проверках: ${num(p.retained_nonwins)}.</p><p class="note">Сопоставлены ${num(p.known_pairs)} из ${num(p.episodes)} эпизодов. Неизвестен хотя бы один исход: ${num(p.unknown_pairs)}; такие пары не считаются выигрышем или потерей.</p>${rows.length?`<details id="paired-games"><summary>Приобретённые и потерянные победы по играм</summary><div class="table-wrap"><table><thead><tr><th scope="col">Игра</th><th scope="col">Новые</th><th scope="col">Потеряны</th><th scope="col">Сохранены</th><th scope="col">Известные пары</th></tr></thead><tbody>${rows.map(g=>`<tr><th scope="row">${esc(data?.task_guides?.[g.task]?.label || g.task)}</th><td>${num(g.gained)}</td><td>${num(g.lost)}</td><td>${num(g.retained_wins)}</td><td>${num(g.known_pairs)} / ${num(g.episodes)}</td></tr>`).join('')}</tbody></table></div></details>`:''}</div>`;
  }
  function evaluationResult(q,c) {
    if(c?.job_kind==='paired_frozen_evaluation') {
      const rows=c.paired_evaluation_result;
      const view=c.paired_comparison || {};
      return `<section class="panel"><h2>${esc(view.title || 'Сравнение моделей в навигации')}</h2>${Array.isArray(rows)&&rows.length===6?`<p>Аудит 20 отчётов завершён.</p><div class="table-wrap"><table><thead><tr><th scope="col">Оформление</th><th scope="col">Выбор действий</th><th scope="col">${esc(view.control_label || 'Контроль')}</th><th scope="col">${esc(view.candidate_label || 'Кандидат')}</th><th scope="col">Разница побед</th><th scope="col">Повторов с приростом</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.appearance==='canonical'?'Обычное':'Пиксельное')}</td><td>${esc(({frozen_stochastic:'По вероятностям',frozen_deterministic:'Наиболее вероятное',generic_physical_random:'Случайный контроль'})[r.policy] || r.policy)}</td><td>${num(r.control_wins)} / ${num(r.episodes)}</td><td>${num(r.candidate_wins)} / ${num(r.episodes)}</td><td>${num(r.delta_wins)}</td><td>${num(r.positive_paired_seeds)} / 5</td></tr>`).join('')}</tbody></table></div>`:empty('Итог сравнения появится после завершения и аудита всех отчётов.')}<p class="note">16 уровней проверяются по 5 раз. Это не 80 независимых уровней и не проверка всех игр. Условие продолжения гипотезы: не менее 8 дополнительных побед и прирост в 3 из 5 повторов при выборе по вероятностям в каждом оформлении. Выбор наиболее вероятного действия показан отдельно.</p></section>`;
    }
    return `<section class="panel"><h2>Игровая проверка</h2>${finite(q?.value)?`<div class="quality-summary"><strong class="quality-value">${pct(q.value)}</strong>${metricHelp('wins','evaluation-wins')}</div><p>${counts(q)}</p>${metricIdentity(q,c)}`:empty('Для текущего контрольного набора проверка не найдена. Это не означает нулевую долю побед.')}<p>${esc(c?.experiment?.conclusion || '')}</p>${perfectScore(q)}<details id="evaluation-source"><summary>Источники результата</summary>${technicalDetails('evaluation-source-details',[['Запуск',c?.run_id],['Файл проверки',q?.measurement_source || q?.latest_evaluation_file]])}</details></section>`;
  }
  function resultBrief(full=true) {
    const q=data.quality || {}, c=data.current;
    if(c?.evaluation_only)return evaluationResult(q,c);
    const stage=qualityStage(q), measured=finite(q.value);
    const comparable=stage==='final'&&q.comparison?.comparable!==false;
    const label=stage==='initial'?'Начальная проверка':stage==='inherited'?'Проверка предыдущей модели':stage==='previous'?'Последняя проверенная версия':'Результат игровой проверки';
    return `<section class="panel result-brief"><div class="panel-top"><h2>${label}</h2><a href="#/games">Результаты по играм</a></div>${measured?`<div class="quality-summary"><strong class="quality-value">${pct(q.value)}</strong>${metricHelp('wins','overview-wins')}${comparable&&finite(q.change_pp)?change(q.change_pp):''}</div><p>${counts(q)}</p>${metricIdentity(q,c)}`:empty('Игровое качество ещё не измерено. Результат появится после сохранения проверки.')}${stage!=='final'&&measured?`<p class="result-pending">${stage==='initial'?'Это точка отсчёта до обновления модели.':'Показана сохранённая проверка; текущие веса ещё не проверены.'}</p>`:''}${comparable&&finite(q.baseline)?`<p>Начальная модель: <strong>${pct(q.baseline)}</strong>. ${finite(q.change_pp)?`Изменение: ${change(q.change_pp)}`:''}</p>`:''}${full&&measured?`<details id="result-details"><summary>Сравнение, дополнительные измерения и источники</summary>${comparison(q)}${perfectScore(q)}${stage==='final'?chart(q.run_history || q.history,'value','Доля побед',true):''}${ppoDiagnostics(q)}</details>`:''}</section>`;
  }
  function skillMap() {
    const games=[...(data.quality?.games || [])].filter(g=>finite(g.value)).sort((a,b)=>a.value-b.value);
    const comparable=qualityStage(data.quality)==='final'&&(data.quality?.comparison?.comparable ?? finite(data.quality?.change_pp));
    const worse=comparable?games.filter(g=>finite(g.change_pp)&&g.change_pp<0):[];
    return `<section class="panel skill-map"><div class="panel-top"><h2>Где агенту трудно</h2><a href="#/games">Все игры и записи</a></div><p class="note">Сначала игры с меньшей долей побед.${comparable?` Ухудшились ${worse.length} из ${games.length} игр на тех же проверочных уровнях.`:' Показаны результаты последней сохранённой проверки.'}</p><div class="skill-bars">${games.length?games.slice(0,4).map(g=>`<a href="#/games/${encodeURIComponent(g.task)}" class="skill-row"><span>${esc(g.label || data.task_guides?.[g.task]?.label || g.task)}</span><span class="skill-meter" aria-hidden="true"><i style="width:${Math.max(0,Math.min(100,g.value))}%"></i></span><strong>${pct(g.value)}</strong><small>${comparable&&finite(g.change_pp)?`${g.change_pp>0?'+':''}${num(g.change_pp)} п.п.`:`${num(g.successes)}/${num(g.episodes)}`}</small></a>`).join(''):empty('Результаты по играм появятся после проверки.')}</div></section>`;
  }
  function technicalRun(c,q,experiment=c?.experiment) {
    return `${experimentSummary(c,experiment)}${failureNotice(c)}${finite(q?.value)?evaluationResult(q,c):''}<details id="technical-run-settings" class="panel"><summary>Параметры технического запуска</summary>${settings(c)}</details>`;
  }
  function overview() {
    const c=data.current,e=c?.experiment || {};
    if(data.cycle&&!c)return cyclePreparation();
    if(data.comparison_plan&&!c)return preparation();
    if(c?.technical_only)return `${experimentSelector()}${technicalRun(c,data.quality)}`;
    if(c?.job_kind==='paired_frozen_evaluation')return `${experimentSelector()}<div class="overview-heading"><h1>Сравнение сохранённых моделей</h1><p>Навигация · обычное и пиксельное оформление</p></div>${failureNotice(c)}${evaluationProgress(c)}${resultBrief()}${resources(c)}<p><a href="#/history">Полные проверки 16 игр — в запусках обучения в архиве</a></p>`;
    return `${experimentSelector()}<div class="overview-heading"><h1>Состояние агента</h1><p>${c?.running?'Обучение продолжается. Здесь — последние сохранённые результаты.':'Здесь — результаты сохранённых проверок выбранного запуска.'}</p></div>${c?.evaluation_only || e.verdict?.state==='pending'?'':verdictSummary(e)}${failureNotice(c)}${c?.evaluation_only?evaluationProgress(c):''}<div class="overview-grid">${resultBrief()}${skillMap()}</div><p class="next-step"><a href="#/games">Выбрать игру и посмотреть попытку агента</a>${c?.running?' · <a href="#/training">Посмотреть ход обучения</a>':''}</p><details id="experiment-context" class="panel context-panel"><summary>Цель запуска, выводы и ограничения</summary>${experimentSummary(c)}</details>${data.diagnostics?.transfer || data.diagnostics?.memory || data.quality?.pixel_validation?`<details id="additional-checks" class="panel"><summary>Дополнительные проверки: память и оформление</summary>${pixelValidation()}${diagnostics(data.diagnostics)}</details>`:''}`;
  }
  const METRIC_HELP = {
    memory_skill: ['Проверка памяти', 'Первый выбор — сколько раз агент выбрал правильное направление после подсказки. Обе версии пары — сколько пар пройдено правильно при двух противоположных подсказках. Пары помогают отличить запоминание подсказки от привычки идти в одну сторону. Показаны количества удачных случаев и всего проверенных случаев.'],
    appearance: ['Проверка другого оформления', 'Доля побед той же модели при изменении внешнего вида, языка или размеров поля. Каждая строка — отдельные условия. Изменение в п.п. считается только против исходной модели в совпадающих условиях этой строки. Сравнение строк разных размеров не изолирует эффект внешнего вида: размер может менять механику.'],
    wins: ['Доля побед', 'Процент эпизодов, в которых агент выполнил цель игры. В общем результате каждая игра имеет одинаковый вес: сначала считаем процент побед в каждой, затем усредняем. Выше — лучше на этом проверочном наборе. Высокий средний результат может скрывать слабые игры и не доказывает успех на новых играх.'],
    change: ['Изменение в процентных пунктах', 'Разность долей побед после и до обучения на одинаковой валидации. Например, рост с 50% до 60% — это +10 процентных пунктов (п.п.). Минус означает регрессию. Разницу на разных уровнях или по другим правилам так сравнивать нельзя. Одна проверка не доказывает, что улучшение устойчиво.'],
    perfect: ['Победы со счётом 5/5', 'Доля эпизодов, где агент победил и получил максимальный счёт по правилам игры. Это строже обычной победы. В лабиринте 5/5 требует кратчайшего маршрута без лишних нажатий; в остальных играх — победы без лишних управляющих действий. Это не доказывает оптимальность всего пути. Число эпизодов и версия правил указаны в подробностях.'],
    baseline: ['Исходная модель', 'Модель до обновления весов в выбранном запуске. Её проверяют на том же наборе, что и обученную модель. Это точка отсчёта для оценки изменений, а не лучший результат среди всех запусков.'],
    nll: ['Ошибка повторения (NLL)', 'Показывает, насколько маловероятным модель считает правильное действие из обучающего примера. Меньше — точнее повторение. Почти ноль означает высокую уверенность в правильных действиях этих примеров. Это не процент ошибок: используется логарифмическая единица «нат». Измеряется во время обучения, а не отдельной проверкой замороженной модели. Снижение NLL само по себе не означает больше побед.'],
    active_nll: ['Ошибка активных действий', 'Та же ошибка повторения, но только на примерах, где нужно действовать: двигаться, нажимать, выбирать. Меньше — лучше повторение этих действий. Полезна отдельно от ожидания: агент может хорошо ждать, но плохо действовать. Победы проверяются отдельно.'],
    idle_nll: ['Ошибка ожидания', 'Ошибка на примерах, где правильное действие — ждать. Меньше — точнее ожидание. Малое значение не доказывает навык игры: сравните его с ошибкой активных действий и результатами валидации.'],
    objective: ['Целевая функция BC', 'Число, которое оптимизатор старается уменьшить при обучении повторению действий. Учитывает заданные веса примеров и типов действий, поэтому может отличаться от обычной NLL. Меньше — лучше по этой цели. Сравнивать значения полезно при одинаковых настройках весов; это не доля побед.'],
    unweighted_objective: ['Цель без весов повторного обучения', 'Та же обучающая цель, но без дополнительных весов источников примеров. Помогает увидеть влияние усиления или ослабления старых данных. Это ещё не проверка качества на новых уровнях.'],
    bc_objective: ['Компонента повторения действий', 'Часть общей цели, отвечающая за повторение действий учителя. Меньше — точнее выполнение этой обучающей цели. Отдельно смотрите отклонение от исходной модели и победы.'],
    anchor_kl: ['Отклонение от опорной политики (KL)', 'Измеряет, насколько распределение действий отличается от сохранённой опорной модели на выбранных примерах. Ноль — совпадение; больше — сильнее изменение. Меньше не всегда лучше: слишком сильное сохранение старого поведения может мешать новым навыкам. Победы и сохранение навыков проверяются отдельно.'],
    total_objective: ['Полная целевая функция', 'Объединяет обучение по примерам и взвешенный штраф за отклонение от опорной модели. Именно их сочетание определяет цель обновления. Значение зависит от коэффициентов; оно не является игровым счётом.'],
    mean_batch_bc_objective: ['Средняя цель повторения по пакетам', 'Среднее значение компоненты повторения действий по обработанным обучающим пакетам. Отличается способом усреднения от метрики по эпизодам. Полезно для наблюдения оптимизации, но не для вывода о победах.'],
    mean_batch_total_objective: ['Средняя полная цель по пакетам', 'Среднее по обучающим пакетам для общей цели: повторение действий и штраф отклонения от опорной модели. При смене состава или весов данных значения могут измениться без улучшения игры.'],
    max_gradient_norm: ['Максимальная норма градиента', 'Величина сигнала, который направляет изменение весов. Показан максимум по пакетам эпохи до ограничения (clipping). Всплески помогают заметить резкое изменение обучения, но большого числа недостаточно для вывода об ошибке. Смотрите его вместе с целью обучения и проверками игры.'],
    gradient_norm: ['Норма градиента PPO', 'Размер сигнала для обновления весов до ограничения (clipping). Резкие изменения могут потребовать внимания, но универсального «хорошего» значения нет. Для эпох, остановленных по лимиту KL, эта метрика не показывается: сохранённое усреднение там может быть неполным.'],
    epoch_training_seconds: ['Время обучения эпохи', 'Сколько секунд занял один проход обучения по примерам. Меньше — быстрее при одинаковых данных и настройках. Это длительность этапа, а не оставшееся время и не показатель качества модели.'],
    cycle_wall_seconds: ['Время цикла PPO', 'Время завершённого цикла сбора игровых попыток и обновления политики. Помогает оценить скорость работы; разные циклы могут содержать разный объём работы. Не измеряет качество игры.'],
    policy_loss: ['Цель политики PPO', 'Целевая функция, по которой PPO меняет вероятности действий с учётом полученного опыта. Может быть отрицательной — это нормально. Знак и снижение значения не доказывают больше побед; нужны отдельные игровые проверки.'],
    value_loss: ['Ошибка оценки будущей награды', 'Насколько оценка будущей суммарной награды отличается от обучающей цели. Меньше означает точнее предсказание этой цели на собранном опыте. Масштаб зависит от наград; это не ошибка выбора действия и не доля поражений.'],
    entropy: ['Разнообразие возможных действий', 'Энтропия показывает, насколько распределение действий рассеяно. Выше — модель допускает больше вариантов; ниже — сильнее предпочитает некоторые действия. Ни максимум, ни минимум не являются целью сами по себе: низкое разнообразие может означать как навык, так и застревание.'],
    approx_kl: ['Насколько изменилась политика PPO', 'Приближённая мера изменения вероятностей действий относительно модели, собиравшей опыт. Слишком большой шаг может вызвать остановку обновлений по заданному лимиту KL. Читайте вместе с этой остановкой и числом принятых обновлений; это не показатель побед.'],
    clip_fraction: ['Доля ограничиваемых изменений', 'Доля примеров, где отношение новой и прежней вероятности действия вышло за установленный диапазон PPO. Шкала от 0 до 1: 0,2 означает 20%. Большая доля показывает активное ограничение шага, но не равна доле ошибочных действий.'],
    optimizer_steps: ['Принятые обновления PPO', 'Сколько обновлений весов было принято в последней эпохе цикла PPO. Это не суммарный счётчик всего запуска. Ноль возможен при ранней остановке по KL; проверьте соответствующую метрику.'],
    kl_early_stop: ['Остановка по лимиту KL', '1 — последняя эпоха PPO остановлена из-за слишком большого изменения политики; 0 — такой остановки не было. Это защитное ограничение размера шага, а не обязательно сбой запуска. Показатель появляется только если флаг сохранён.'],
    memory: ['Память MLX', 'Активная — выделения аллокатора MLX; кеш — память, удержанная для повторного использования; пик — максимальные выделения с момента сброса счётчика. MiB — мебибайты. Это сохранённый снимок, а не текущая память всего процесса или всей системы.'],
  };
  function metricHelp(key, id=key) {
    const hint=METRIC_HELP[key];
    return hint?`<details id="help-${esc(id)}" class="metric-help"><summary aria-label="Подсказка: ${esc(hint[0])}" title="Пояснение метрики">?</summary><div><button type="button" class="hint-close" data-close-help="${esc(id)}" aria-label="Закрыть подсказку">×</button><strong>${esc(hint[0])}</strong><p>${esc(hint[1])}</p></div></details>`:'';
  }
  function telemetry(c) {
    const t=c?.telemetry || {}, series=(t.series || []).map(m=>({...m,unit:({s:'с',nat:'нат',fraction:'доля 0–1',steps:'обновлений',flag:'0/1'})[m.unit] || m.unit}));
    const metric=series.find(m=>m.key===selectedMetric) || series[0];
    const latestStep=Math.max(0,...series.map(m=>m.latest_step || 0));
    return `<section class="panel metrics-panel"><div class="panel-top"><h2>Измерения обновлений модели</h2><span class="note">Завершено ${c?.method==='ppo'?'циклов':'эпох'}: ${num(t.rows_count || 0)}</span></div><p class="note">Метрики оптимизации объясняют ход обучения. Победы на фиксированной валидации показывают качество игры.</p>${series.length?`<div class="metric-choices" role="group" aria-label="Выбор метрики">${series.map(m=>`<div class="metric-choice"><button type="button" data-metric="${esc(m.key)}" aria-pressed="${m.key===metric.key}"><span>${esc(m.label)}</span><strong>${nll(m.latest)} <small>${esc(m.unit || '')}</small></strong>${finite(m.latest_step)&&m.latest_step<latestStep?`<small>Ранее: ${esc(t.axis_label)} ${num(m.latest_step)}</small>`:''}</button>${metricHelp(m.key,`tile-${m.key}`)}</div>`).join('')}</div><div class="metric-description"><h3>${esc(metric.label)}</h3><p>${esc(METRIC_HELP[metric.key]?.[1] || metric.description)}</p><p class="note">${esc(metric.description)}</p>${finite(metric.latest_step)?`<p class="note">Последний замер: ${esc(t.axis_label)} ${num(metric.latest_step)}.</p>`:''}</div>${chart(metric.points,'value',`${metric.label}${metric.unit?` (${metric.unit})`:''}`,false,t.axis_label,`training-${metric.key}`)}<p class="note">Источник: training.jsonl. Точки — завершённые этапы; это не текущий мини-пакет. Для длинных историй показывается до 200 точек.</p>`:empty('Завершённые этапы с метриками ещё не сохранены. Текущая фаза и прогресс показаны выше.')}</section>`;
  }
  function resources(c) {
    if(!c)return `<section class="panel"><h2>Нагрузка на компьютер</h2>${empty('Обучение не запущено. Измерений памяти процесса обучения пока нет.')}</section>`;
    const items=c?.telemetry?.resources || [];
    return `<section class="panel"><div class="help-heading"><h2>Нагрузка на компьютер</h2>${metricHelp('memory')}</div>${guardResources(c)}${memoryFreshness(c)}<p class="note">Последнее состояние: ${date(c?.last_update_at)}. ${c?.running?`Возраст данных: ${duration(c.data_age_seconds)}.`:'Сохранённые измерения завершённого или остановленного запуска.'}</p>${items.length?`<dl class="resource-values">${items.map(m=>`<div><dt>${esc(m.label)}</dt><dd>${num(m.latest)} ${esc(m.unit)}</dd></div>`).join('')}</dl><p class="note">Память аллокатора MLX в момент сохранения. Это не вся RAM компьютера; пик накоплен с начала измерения. Значения не являются текущим потреблением.</p>`:empty(c?.evaluation_only?'Память проверки измеряет внешний наблюдатель. Отсутствие замера не означает нулевое потребление.':'Измерения памяти аллокатора MLX не сохранены. Отсутствие данных не означает нулевое потребление.')}<details id="phase-events"><summary>История этапов</summary>${Array.isArray(c?.phase_history)&&c.phase_history.length?`<ol class="phase-events">${c.phase_history.slice(-30).map(p=>`<li>${esc(({loading:'Загрузка моделей',collecting:'Сбор примеров',encoding:'Кодирование кадров',baseline:'Начальная проверка',imitation:'Обучение на примерах',training:'Обучение',evaluation:'Проверка',pixel_evaluation:'Проверка пиксельного оформления',final_evaluation:'Итоговая проверка',saving:'Сохранение'})[p.phase] || p.label || p.phase || 'Этап')}<span>${date(p.started_at)}${finite(p.elapsed_seconds)?` · ${duration(p.elapsed_seconds)}`:''}</span></li>`).join('')}</ol>`:empty('История переходов между этапами не сохранена.')}</details></section>`;
  }
  function failureNotice(c) {
    if(!['failed','error','interrupted','resource_stopped','stopped'].includes(c?.status))return '';
    const reason=c.failure_reason;
    return `<aside class="warning failure-notice"><strong>Почему запуск остановлен</strong><p>${esc(reason || 'Причина остановки не сохранена.')}</p>${reason?.includes('epoch snapshot does not match its frozen evaluation')?'<p>Сохранённый снимок модели не совпал с весами, для которых выполнялась проверка. Итог этого запуска нельзя считать подтверждённым; перед новой проверкой нужно устранить несовпадение.</p>':''}</aside>`;
  }
  function trainingProgress(c) {
    if(c?.evaluation_only)return '';
    if(!c || c.running || !['completed','stopped','interrupted','resource_stopped','failed','error'].includes(c.status))return progress(c);
    return `<details id="completed-progress" class="panel progress-details"><summary>${esc(c.status_label || 'Запуск завершён')} · ${duration(c.elapsed_seconds)} · ${c.method==='ppo'?`Циклов: ${num(c.rollouts)}`:`Эпох: ${num(c.epochs_completed)}`}<span>Этапы и счётчики</span></summary>${progress(c)}</details>`;
  }
  function training() {
    const c=data.current;
    if(data.cycle&&!c)return cyclePreparation();
    if(data.comparison_plan&&!c)return preparation();
    if(c?.technical_only)return `${experimentSelector()}${technicalRun(c,data.quality)}`;
    if(c?.evaluation_only)return `${experimentSelector()}<h1>Проверка модели</h1><p class="subtitle">В этом запуске веса не обучались. Здесь показаны доступные результаты проверки.</p>${failureNotice(c)}${evaluationProgress(c)}${resultBrief()}<details id="computer-load" class="panel"><summary>Нагрузка на компьютер</summary>${resources(c)}</details><details id="evaluation-settings" class="panel"><summary>Как настроена проверка</summary>${settings(c)}</details>`;
    const perTask = Array.isArray(c?.per_task_nll)?c.per_task_nll:Object.entries(c?.per_task_nll || {}).map(([task,nll])=>({task,nll}));
    const hasSeparateChecks = data.quality?.diagnostic_series?.some(series=>series.history?.length);
    return `${experimentSelector()}<h1>${c?.running?'Обучение в работе':'Разбор обучения'}</h1><p class="subtitle">${c?.method==='ppo'?'Агент играет, получает награду и обновляет модель.':'Агент учится повторять действия из примеров.'} Проверки ниже показывают, меняется ли качество игры.</p>${cycleCard()}${failureNotice(c)}${trainingProgress(c)}<section class="panel evaluation-panel"><div class="help-heading"><h2>Меняется ли качество игры</h2>${metricHelp('wins','training-wins')}</div>${resultBrief(false)}${comparison()}${finite(data.quality?.value)?chart(data.quality.run_history || [],'value','Доля побед на проверочных уровнях',true,'','training-wins'):empty(hasSeparateChecks?'Сохранённые проверки показаны отдельно ниже. Общая сопоставимая проверка недоступна.':'Нет игровой проверки этого запуска.')}${ppoDiagnostics(data.quality)}<a href="#/games">Проверки отдельных игр и записи</a></section>${taskProgress(c)}${trainingOutcomes(c)}${rlDetails(c)}<div class="training-layout"><details id="optimization-details" class="training-details"><summary>Как меняются параметры модели</summary><p class="note">Эти числа помогают понять обновления модели. Уменьшение ошибки (loss) не означает больше побед.</p>${telemetry(c)}</details><div class="training-sidebar"><details id="computer-load" class="panel training-details"><summary>Нагрузка на компьютер</summary>${resources(c)}</details><details id="training-settings" class="panel training-details"><summary>Как настроено обучение</summary>${settings(c)}</details></div></div>${c?.method==='ppo'?'':`<section class="panel per-task-panel"><div class="help-heading"><h2>Ошибка повторения по играм</h2>${metricHelp('nll','per-game-nll')}</div><p class="note">Обучающие примеры: меньшая ошибка не означает больше побед. Показаны последние сохранённые значения.</p>${perTask.length?`<div class="table-wrap"><table><thead><tr><th>Игра</th><th>NLL</th></tr></thead><tbody>${perTask.sort((a,b)=>(b.nll ?? b.value)-(a.nll ?? a.value)).map(p=>`<tr><td><a href="#/games/${encodeURIComponent(p.task)}">${esc(data.task_guides?.[p.task]?.label || p.label || p.task)}</a></td><td>${nll(p.nll ?? p.value)}</td></tr>`).join('')}</tbody></table></div>`:empty('Значения по играм ещё не опубликованы.')}</section>`}<details id="training-appearance" class="panel"><summary>Проверка другого оформления</summary>${pixelValidation()}</details>`;
  }
  const checkpointLabel = c => {if(!c)return 'Версия недоступна';if(c.epoch===0 || c.rollout===0)return 'Начальная модель';const plural=(n,one,few,many)=>n%10===1&&n%100!==11?one:n%10>=2&&n%10<=4&&(n%100<12||n%100>14)?few:many;if(finite(c.epoch))return `После ${num(c.epoch)} ${plural(c.epoch,'прохода','проходов','проходов')} данных`;if(finite(c.rollout))return `После ${num(c.rollout)} ${plural(c.rollout,'цикла','циклов','циклов')}`;return finite(c.revision)?`Сохранённая версия ${num(c.revision)}`:'Сохранённая версия';};
  function validationChoices(task) {
    if(validationTask!==task){validationTask=task;validationLevel='';selectedRecording='';replayError='';}
    const levels=validation?.tasks?.find(t=>t.task===task)?.levels || [];
    if(!levels.some(l=>String(l.level_index)===validationLevel))validationLevel=levels.length?String(levels[0].level_index):'';
    const checkpoints=validation?.checkpoints || [];
    if(!checkpoints.some(c=>c.id===validationCheckpoint))validationCheckpoint=checkpoints.filter(c=>c.available).at(-1)?.id || checkpoints.at(-1)?.id || '';
    saveValidationView();
    return {levels,checkpoints};
  }
  function validationRecords(task) {
    const records=[...(validation?.records || [])];
    if(replayJob?.status==='completed'&&replayJob.request?.run===validationRun&&replayJob.replay&&!records.some(r=>r.replay_id===replayJob.id))records.push({...replayJob.replay,...replayJob.request,checkpoint_id:replayJob.request.checkpoint,manual:true,replay_id:replayJob.id});
    return records.filter(r=>r.task===task&&String(r.level_index)===validationLevel&&r.appearance===validationAppearance&&r.mode===validationMode);
  }
  function manualRecording(p,id=p.id,request=p.request) {
    return {...request,...p,id,manual:true,run_id:request?.run || validationRun,task:request?.task || p.task,
      checkpoint_id:request?.checkpoint || p.checkpoint_id,url:p.url || p.gif_url,
      created_at:p.created_at || p.created_utc,revision:p.revision ?? p.weights_revision,
      actions:p.actions?.map(a=>a.action || a)};
  }
  function validationPlayback(task) {
    const saved=(validation?.replays || []).map(p=>manualRecording(p));
    if(replayJob?.status==='completed'&&replayJob.request?.run===validationRun&&replayJob.replay&&!saved.some(p=>p.id===replayJob.id))saved.push(manualRecording(replayJob.replay,replayJob.id,replayJob.request));
    const all=[...(data.playbacks?.items || []),...saved];
    if(validationView==='archive')return playback(task,{...data,playbacks:{items:all}});
    const level=validation?.tasks?.find(t=>t.task===task)?.levels?.find(l=>String(l.level_index)===validationLevel);
    const checkpoint=validation?.checkpoints?.find(c=>c.id===validationCheckpoint);
    const items=all.filter(p=>p.task===task&&p.run_id===validationRun&&p.appearance===validationAppearance&&p.mode===validationMode&&(p.level_index!==undefined?String(p.level_index)===validationLevel:level?.level_id&&p.level_id===level.level_id)&&(p.checkpoint_id===validationCheckpoint || checkpoint?.weights_revision!==undefined&&p.revision===checkpoint.weights_revision));
    if(!items.length){const available=checkpoint?.available&&validationLevel!==''&&validationMode==='greedy'&&!replaySubmitting&&!['running','queued'].includes(replayJob?.status);return `<section class="panel recording-panel"><h2>Запись выбранного уровня</h2><div class="preview empty">Запись этой версии в выбранных условиях пока не сохранена.</div><p class="note">${available?'Нажмите «Проиграть уровень», чтобы сохранить одну попытку. Обучение не запускается.':validationMode!=='greedy'?'Для новой записи выберите «Самое вероятное действие».':replaySubmitting || ['running','queued'].includes(replayJob?.status)?'Попытка выполняется. Дождитесь завершения.':'Для новой записи выберите доступную версию модели в таблице.'}</p></section>`;}
    return playback(task,{...data,playbacks:{items}});
  }
  function validationWorkbench(task) {
    if(!validation && validationError)return `<section class="panel validation-state" role="alert"><h2>Проверочные уровни не загрузились</h2><p>${esc(validationError)}</p><button type="button" id="retry-validation">Повторить загрузку</button></section>`;
    if(!validation)return `<section class="panel validation-state" role="status" aria-live="polite"><h2>Загружаем проверочные уровни</h2><p>Здесь появятся сохранённые версии модели и результаты их игры.</p></section>`;
    const {levels,checkpoints}=validationChoices(task);
    if(!levels.length || !checkpoints.length)return `<section class="panel validation-state"><h2>${!levels.length?'Проверочных уровней пока нет':'Версии модели пока не сохранены'}</h2><p>Попробуйте обновить данные позже или выберите другую игру.</p><button type="button" id="retry-validation">Обновить данные</button></section>`;
    const records=validationRecords(task), official=records.filter(r=>!r.manual), manual=records.filter(r=>r.manual);
    const pending=replaySubmitting || ['running','queued'].includes(replayJob?.status);
    const selected=checkpoints.find(c=>c.id===validationCheckpoint);
    const disabled=pending || !selected?.available || validationMode!=='greedy';
    const modes=[...new Set(['greedy','stochastic','random',...(validation?.records || []).filter(r=>r.task===task).map(r=>r.mode)])];
    const conditions=r=>r?`Уровень: ${r.level_id || 'нет ID'}; semantic ID: ${r.semantic_level_id || 'нет ID'}; seed: ${r.seed ?? 'нет данных'}; сложность: ${r.difficulty ?? 'нет данных'}; инструкция: ${r.instruction || 'нет данных'}; протокол: ${r.protocol_sha256 || 'нет данных'}`:'';
    const outcomeLabel=r=>r.timeout || r.outcome?.timeout || r.truncated&&!r.outcome?.success&&!r.outcome?.failure?'Лимит ходов':outcome(r.outcome) || 'Исход неизвестен';
    const table=(rows,isManual=false)=>`<div class="table-wrap level-history"><table><caption>${isManual?'Ручные пробы — отдельно от общей доли побед':'Плановые проверки выбранного уровня'}</caption><thead><tr><th scope="col">Версия модели${isManual?'':' · выберите версию'}</th><th scope="col">Исход</th><th scope="col">Ходы</th><th scope="col">Счёт из 5</th></tr></thead><tbody>${checkpoints.map(c=>{const found=rows.filter(r=>r.checkpoint_id===c.id);return (found.length?found:isManual?[]:[null]).map(r=>`<tr ${c.id===validationCheckpoint?'class="selected-checkpoint"':''}><th scope="row"><button type="button" data-validation-checkpoint="${esc(c.id)}" ${r?.replay_id?`data-validation-replay="${esc(r.replay_id)}"`:''} aria-pressed="${c.id===validationCheckpoint}">${esc(checkpointLabel(c))}</button>${!c.available?'<small>Файл модели недоступен для новой попытки</small>':''}</th><td>${r?esc(outcomeLabel(r)):'Нет плановой проверки'}</td><td>${num(r?.steps)}</td><td>${num(r?.outcome?.score ?? r?.quality?.visible_score ?? r?.quality?.score)}</td></tr>`).join('');}).join('')}</tbody></table></div>`;
    return `<div class="validation-workbench"><section class="panel level-panel"><h2>Один уровень через всё обучение</h2><p class="note">Проверочные уровни не используются для обновления модели. Сравните её версии на одном уровне. Общая доля побед считается по всему набору, а не по этой попытке.</p>${validationError?`<p class="warning" role="status">Не удалось обновить уровни. Показаны последние полученные данные. <button type="button" id="retry-validation">Повторить</button></p>`:''}<div class="level-filters"><div class="filter-field"><label for="validation-level">Проверочный уровень</label><select id="validation-level">${levels.map(l=>`<option value="${l.level_index}" ${String(l.level_index)===validationLevel?'selected':''}>Уровень ${num(l.level_index+1)}</option>`).join('')}</select></div><div class="filter-field"><label for="validation-appearance">Оформление</label><select id="validation-appearance"><option value="canonical" ${validationAppearance==='canonical'?'selected':''}>Обычное</option><option value="pixel" ${validationAppearance==='pixel'?'selected':''}>Пиксельное</option></select></div><div class="filter-field"><label for="validation-mode">Способ выбора действий</label><select id="validation-mode">${modes.map(m=>`<option value="${esc(m)}" ${m===validationMode?'selected':''}>${esc(modeLabel(m))}</option>`).join('')}</select></div></div><details id="mode-help" class="mode-help"><summary>Чем отличаются способы выбора действий</summary><p>Самое вероятное действие — модель всегда выбирает свой первый вариант. Выбор по вероятностям — она пробует варианты с учётом своей уверенности, поэтому попытки могут различаться. Случайные действия — контроль без решений модели.</p><p>Новая ручная попытка доступна только для самого вероятного действия. Сохранённые проверки остальных способов можно сравнить в таблице.</p></details>${table(official)}${manual.length?`<details id="manual-results"><summary>Ручные пробы (${manual.length})</summary><p class="note">Эти попытки создаются кнопкой ниже и не входят в общую долю побед.</p>${table(manual,true)}</details>`:''}<p class="note">Лимит ходов означает, что игра остановлена после допустимого числа действий без победы или поражения.</p><div class="replay-controls"><p>Выбрана: <strong>${esc(checkpointLabel(selected))}</strong></p><button type="button" id="start-replay" ${disabled?'disabled':''}>${pending?'Попытка выполняется…':'Проиграть уровень'}</button></div>${!selected?.available?'<p class="note">Результат сохранён, но файл этой версии недоступен для новой попытки. Выберите доступную версию в таблице.</p>':validationMode!=='greedy'?'<p class="note">Для новой попытки выберите «Самое вероятное действие». Здесь показаны сохранённые проверки выбранного способа.</p>':''}<div id="replay-status" role="status" aria-live="polite">${pending?'<p>Сохранённая модель играет одну попытку. Запись появится после завершения; обучение не запускается.</p>':replayJob?.status==='completed'?'<p>Попытка завершена. Запись доступна при выборе её уровня и версии модели.</p>':''}</div>${replayError || replayJob?.error?`<p class="error" role="alert">${esc(replayError || replayJob.error)}</p>`:''}${records.length?`<details id="level-conditions"><summary>Условия и источники проверок</summary><p class="note">Для сравнения должны совпадать содержимое уровня, инструкция, сложность и протокол.</p>${records.map(r=>`<p class="note"><strong>${esc(checkpointLabel(checkpoints.find(c=>c.id===r.checkpoint_id)))}${r.manual?' · ручная проба':''}</strong><br>${esc(conditions(r))}${r.evaluation_file?`<br>Источник: ${esc(r.evaluation_file)}`:''}</p>`).join('')}${technicalDetails('checkpoint-identities',checkpoints.map(c=>[checkpointLabel(c),`${c.id} · ревизия ${c.weights_revision ?? 'не указана'}`]))}</details>`:''}${validation.warnings?.length?`<details id="validation-warnings"><summary>Ограничения данных</summary>${validation.warnings.map(w=>`<p class="note">${esc(w)}</p>`).join('')}</details>`:''}</section><div class="level-player"><div class="filter-field recording-scope"><label for="validation-view">Записи игры</label><select id="validation-view"><option value="level" ${validationView==='level'?'selected':''}>Выбранный уровень и версия</option><option value="archive" ${validationView==='archive'?'selected':''}>Все записи этой игры</option></select></div>${validationPlayback(task)}</div></div>`;
  }
  async function fetchValidation() {
    const id=data?.current?.run_id;
    if(!id || validationLoading)return;
    validationLoading=true;validationRun=id;validationError='';render();
    try {const response=await fetch(`/api/validation?run=${encodeURIComponent(id)}`,{cache:'no-store',signal:AbortSignal.timeout(10000)});const result=await response.json();if(!response.ok)throw new Error(result.error || `HTTP ${response.status}`);if(validationRun!==id)return;validation=result;validationError='';if(['running','queued'].includes(result.active_job?.status)&&!['running','queued'].includes(replayJob?.status))replayJob=result.active_job;}
    catch(e){if(validationRun===id)validationError=`Не удалось загрузить уровни (${e.message}).`;}
    finally{validationLoading=false;render();}
  }
  async function startReplay() {
    const c=validation?.checkpoints?.find(c=>c.id===validationCheckpoint);
    if(replaySubmitting || ['running','queued'].includes(replayJob?.status) || !c?.available || validationLevel==='' || validationMode!=='greedy')return;
    const request={run:validationRun,checkpoint:validationCheckpoint,task:validationTask,level_index:Number(validationLevel),appearance:validationAppearance,mode:validationMode};
    replaySubmitting=true;replayError='';replayJob=null;render();
    try {const response=await fetch('/api/replay',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});const result=await response.json();if(!response.ok)throw new Error(result.error || `HTTP ${response.status}`);if(validationRun===request.run){replayJob={...result,request:result.request || request};validationView='level';selectedRecording='';}}
    catch(e){if(validationRun===request.run)replayError=`Не удалось проиграть уровень (${e.message}).`;}
    finally{replaySubmitting=false;render();}
  }
  async function pollReplay() {
    if(!replayJob?.id || !['running','queued'].includes(replayJob.status))return;
    const id=replayJob.id;
    try {const response=await fetch(`/api/replay?id=${encodeURIComponent(id)}`,{cache:'no-store',signal:AbortSignal.timeout(10000)});const result=await response.json();if(!response.ok)throw new Error(result.error || `HTTP ${response.status}`);if(replayJob?.id===id){replayJob={...result,request:result.request || replayJob.request};replayError='';if(result.status==='completed')await fetchValidation();}}
    catch(e){if(replayJob?.id===id)replayError=`Не удалось получить состояние попытки (${e.message}). Следующее обновление повторит запрос.`;}
    render();
  }
  function games(task) {
    const results=data.quality?.games || [];
    const games=Object.entries(data.task_guides || {}).map(([task,guide])=>({task,label:guide.label,...results.find(g=>g.task===task)}));
    for(const result of results)if(!games.some(g=>g.task===result.task))games.push(result);
    const stage=qualityStage(data.quality);
    if(!task)return `${experimentSelector()}<h1>Игры и записи</h1><p class="subtitle">Выберите игру, чтобы сравнить версии модели и посмотреть одну попытку. Проценты ниже относятся к сохранённой проверке всего набора.</p>${metricIdentity(data.quality)}<div class="table-wrap"><table class="games-table"><thead><tr><th scope="col">Игра</th><th scope="col">Доля побед</th><th scope="col">Победы / попытки</th><th scope="col">Проверенная версия</th></tr></thead><tbody>${games.map(g=>`<tr><th scope="row"><a href="#/games/${encodeURIComponent(g.task)}">${esc(g.label || g.task)}</a></th><td>${pct(g.value)}</td><td>${finite(g.successes)&&finite(g.episodes)?`${num(g.successes)} / ${num(g.episodes)}`:'Не измерено'}</td><td>${finite(g.value)?esc(stage==='initial'?'Начальная модель':stage==='inherited'?'Предыдущая модель':stage==='previous'?'Ранее сохранённая версия':'Последняя проверенная версия'):'Нет проверки'}</td></tr>`).join('')}</tbody></table></div>${!games.length?empty('Список игр пока недоступен. Нажмите «Обновить», чтобы повторить запрос.'):''}<details id="games-extra-checks" class="panel"><summary>Дополнительные проверки оформления и памяти</summary>${pixelValidation()}${diagnostics(data.diagnostics)}</details>`;
    const game=games.find(g=>g.task===task),guide=data.task_guides?.[task];
    if(!game&&!guide)return `${experimentSelector()}<a class="back" href="#/games">Все игры</a><h1>Игра не найдена</h1><p>Выберите игру из общего списка.</p>`;
    const history=(data.quality?.history || []).filter(p=>finite(p.games?.[task]?.value)).map(p=>({...p,...p.games[task]}));
    return `${experimentSelector()}<a class="back" href="#/games">Все игры</a><h1>${esc(game?.label || guide?.label || task)}</h1><p class="subtitle"><strong>Цель:</strong> ${esc(guide?.objective || 'Описание цели пока не добавлено.')}</p>${validationWorkbench(task)}<section class="panel game-summary"><h2>Проверка по всем уровням этой игры</h2>${finite(game?.value)?`<div class="quality-summary"><strong class="quality-value">${pct(game.value)}</strong>${stage==='final'&&data.quality?.comparison?.comparable!==false?change(game.change_pp):''}</div><p>${counts(game)}</p>${metricIdentity(data.quality)}${stage==='final'&&data.quality?.comparison?.comparable!==false&&finite(game.baseline)?`<p>Начальная модель: ${pct(game.baseline)}.</p>`:''}`:empty('Результат этой игры ещё не измерен. Это не означает ноль побед.')}<details id="game-measurement-source"><summary>История, счёт и условия проверки</summary>${perfectScore({...game,scoring_version:game?.scoring_version,scoring_version_label:game?.scoring_version?data.quality?.scoring_version_label:'Прежняя версия счёта: сравнение недоступно'})}${stage==='final'?chart(history,'value','Доля побед',true):empty('Для этого обучения пока нет двух сопоставимых проверок.')}<p class="note">${esc(data.quality?.protocol_description || 'Точные условия появятся после сохранения отчёта.')}</p></details><details id="game-controls"><summary>Действия агента в этой игре</summary><p>${esc(Array.isArray(guide?.controls)?guide.controls.join(', '):guide?.controls || 'Описание управления пока не добавлено.')}</p></details></section>`;
  }
  function history() {
    const category=r=>{
      const type=r.kind || r.method || '';
      if(r.evaluation_only || /evaluation|evaluate|diagnostic|audit|compare_retention/.test(type) || type==='random')return 'evaluation';
      if(/probe|technical|synthetic|prepar|collection|coverage/.test(type))return 'technical';
      return ['sequence_training','imitation_training','imitation','imitation_bc','training','ppo','rl'].includes(type)?'training':'unknown';
    };
    const all=data.experiments || [], runs=historyFilter==='all'?all:all.filter(r=>category(r)===historyFilter);
    return `<h1>Запуски</h1><p class="subtitle">Откройте запуск, чтобы посмотреть результаты, условия и записи. Состояние процесса показывает, завершилась ли работа; качество игры измеряется отдельно по победам.</p><div class="history-filters" role="group" aria-label="Тип запуска">${[['training','Обучение'],['evaluation','Проверки'],['technical','Технические'],['all','Все']].map(([id,label])=>`<button type="button" data-history-filter="${id}" aria-pressed="${historyFilter===id}">${label}</button>`).join('')}</div><div class="experiment-history">${runs.length?runs.map(r=>{
      const e=r.experiment || {},q=r.quality || {},id=r.run_id || r.id;
      const value=q.value ?? r.value ?? r.quality_value ?? r.quality_percent ?? r.win_rate;
      const measured=e.measurement;
      const result=finite(value)?`<p>Доля побед: <strong>${pct(value)}</strong>${measured?`. ${counts(measured)}. ${measured.rollout===0||measured.epoch===0?'Начальная модель.':'Сохранённая проверка.'}`:'. Условия и проверенная версия — внутри запуска.'}</p>`:e.conclusion?`<p class="history-result">${esc(e.conclusion)}</p>`:empty('Игровая проверка не сохранена.');
      return `<article class="panel experiment-card"><div class="run-heading"><h2><a class="history-name" href="#/history/${encodeURIComponent(id)}">${esc(pickerLabel(r))}</a></h2>${status(r)}</div>${result}${finite(r.elapsed_seconds)?`<p class="history-meta">Продолжительность: ${duration(r.elapsed_seconds)}</p>`:''}<details id="context-${esc(id)}"><summary>Подробности запуска</summary><p>${esc(e.purpose || r.kind_label || kind(r.kind))}</p>${finite(value)&&e.conclusion?`<p>${esc(e.conclusion)}</p>`:''}${r.cohort_description?`<p>${esc(r.cohort_description)}</p>`:''}<p>${esc(e.method || '')}</p>${savedComment(e,`comment-${id}`)}${technicalDetails(`run-id-${id}`,[['Запуск',id],['Название',runLabel(r)]])}</details></article>`;
    }).join(''):empty('В этой категории запусков пока нет. Выберите «Все», чтобы открыть остальные.')}</div>`;
  }
  function runPage(id) {
    if (detailId!==id) { detailId=id; detail=null; detailError=''; detailLoading=true; fetchDetail(id); }
    const d = detail;
    if(d?.current?.technical_only)return `<a class="back" href="#/history">Вернуться к истории запусков</a>${technicalRun(d.current,d.quality,d.experiment || d.current.experiment)}`;
    return `<a class="back" href="#/history">Вернуться к истории запусков</a>${d?experimentSummary(d.current || d.run || d,d.experiment || d.current?.experiment):'<h1>Эксперимент</h1>'}${detailLoading?'<div class="loading" role="status">Загружаем запуск…</div>':''}${detailError?`<div class="error" role="alert">${esc(detailError)} <button id="retry-detail" type="button">Повторить</button></div>`:''}${d?`${failureNotice(d.current || d.run || d)}${trainingProgress(d.current || d.run || d)}<div class="grid">${quality(d.quality,true,d.current || d)}${playback('',d)}${d.current?.method==='ppo'||(!finite(d.current?.nll)&&!d.current?.loss_history?.length)?'':loss(d.current || d.run || d)}</div>${d.diagnostics&&Object.values(d.diagnostics).some(Boolean)?`<details id="archive-diagnostics" class="panel"><summary>Дополнительные проверки</summary>${diagnostics(d.diagnostics,d.current)}</details>`:''}<details id="archive-settings" class="panel"><summary>Параметры запуска</summary>${settings(d.current || d.run || d)}</details>`:''}`;
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
    const renderKey=JSON.stringify([r,selectedRun,recordingRun,recordingTask,recordingPage,selectedRecording,selectedMetric,hiddenRecording,error,unavailableRunNotice,detailError,detailLoading,data,detail,validation,validationError,validationLoading,validationLevel,validationCheckpoint,validationAppearance,validationMode,validationView,replayJob,replaySubmitting,replayError],(key,value)=>key==='updated_at'?undefined:value);
    if(renderKey===lastRenderKey)return;
    const activeId = main.contains(document.activeElement)?document.activeElement.id:'';

    const template=document.createElement('template');
    let page = r.page==='training'?training():r.page==='games'?games(r.id):r.page==='history'?(r.id?runPage(r.id):history()):overview();
    const warnings=(data.warnings || []).map(w=>typeof w==='string'?w:w.message || w.label || '').filter(Boolean).filter(w=>w!=='Этот замер показывает освоение текущих мини-игр; перенос на новые игры и оформление оценивается отдельно.');
    const sameTraining=r.page==='training'&&data.training_current?.run_id===data.current?.run_id;
    template.innerHTML=`${sameTraining?'':activity()}${unavailableRunNotice?`<div class="warning" role="status">${esc(unavailableRunNotice)}</div>`:''}${error?`<div class="error" role="alert">${esc(error)} Показаны последние полученные данные; они могут быть устаревшими.</div>`:''}${data.current?.stale && ['running','stale','unknown'].includes(data.current?.status)?'<div class="warning">Источник давно не обновлялся. Проверьте состояние процесса обучения.</div>':''}${page}${technicalMessages(warnings)}`;
    const viewKeys=new Map();
    [...template.content.children].forEach(element=>{const name=`${element.tagName}-${element.className}`;const count=viewKeys.get(name) || 0;viewKeys.set(name,count+1);element.dataset.viewKey=`${r.page}-${r.id}-${name}-${count}`;});
    patchChildren(main,template.content);
    lastRenderKey=JSON.stringify([r,selectedRun,recordingRun,recordingTask,recordingPage,selectedRecording,selectedMetric,hiddenRecording,error,unavailableRunNotice,detailError,detailLoading,data,detail,validation,validationError,validationLoading,validationLevel,validationCheckpoint,validationAppearance,validationMode,validationView,replayJob,replaySubmitting,replayError],(key,value)=>key==='updated_at'?undefined:value);
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
    let requestedRun=selectedRun;
    try {
      const load=id=>fetch(`/api/overview${id?`?run=${encodeURIComponent(id)}`:''}`,{cache:'no-store',signal:AbortSignal.timeout(10000)});
      let response=await load(requestedRun), unavailable=false;
      if(response.status===404 && requestedRun && requestedRun===selectedRun) {
        selectRun('');requestedRun='';unavailable=true;
        response=await load('');
      }
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      const next=await response.json();
      if(!next || typeof next!=='object' || Array.isArray(next))throw new Error('неверный формат данных');
      if(requestedRun!==selectedRun)return;
      data=next;error='';
      if(route().page==='games')await fetchValidation();
      await pollReplay();
      if(unavailable)unavailableRunNotice='Ранее выбранный запуск недоступен. Показано текущее обучение.';
    }
    catch(e){if(requestedRun===selectedRun)error=`Не удалось обновить данные (${e.name==='TimeoutError'?'сервер не ответил вовремя':e.message}).`;}
    finally{busy=false;refreshButton.disabled=false;render();if(requestedRun!==selectedRun)refresh();}
  }
  refreshButton.addEventListener('click',refresh);
  main.addEventListener('change',event=>{
    const validationFields={'validation-level':'level','validation-checkpoint':'checkpoint','validation-appearance':'appearance','validation-mode':'mode','validation-view':'view'};
    if(validationFields[event.target.id]){const value=event.target.value;if(event.target.id==='validation-level')validationLevel=value;if(event.target.id==='validation-checkpoint')validationCheckpoint=value;if(event.target.id==='validation-appearance')validationAppearance=value;if(event.target.id==='validation-mode')validationMode=value;if(event.target.id==='validation-view')validationView=value;selectedRecording='';replayError='';saveValidationView();render();return;}
    if(event.target.id==='experiment-run'){selectRun(event.target.value);refresh();return;}
    if(event.target.id==='recording-task'){recordingTask=event.target.value;recordingPage=0;selectedRecording='';hiddenRecording=reducedMotion();render();}
    if(event.target.id==='recording'){selectedRecording=event.target.value;hiddenRecording=reducedMotion();render();}
  });
  main.addEventListener('input',event=>{if(event.target.id==='frame-seek'){const p=players.get(selectedRecording);if(p){p.frame=Number(event.target.value);p.playing=false;p.last=0;drawPlayer(p);document.getElementById('play-pause').textContent='Продолжить';}}});
  main.addEventListener('click',event=>{
    const historyCategory=event.target.closest?.('[data-history-filter]');
    if(historyCategory){historyFilter=historyCategory.dataset.historyFilter;lastRenderKey='';render();return;}
    if(event.target.id==='start-replay'){startReplay();return;}
    if(event.target.id==='retry-validation'){fetchValidation();return;}
    const checkpoint=event.target.closest?.('[data-validation-checkpoint]');
    if(checkpoint){validationCheckpoint=checkpoint.dataset.validationCheckpoint;selectedRecording=checkpoint.dataset.validationReplay || '';validationView='level';saveValidationView();render();return;}
    const closeHelp=event.target.closest?.('[data-close-help]');
    if(closeHelp){const hint=closeHelp.closest('details');hint.open=false;hint.querySelector('summary').focus();return;}
    const help=event.target.closest?.('.metric-help>summary');
    if(help)main.querySelectorAll('.metric-help[open]').forEach(d=>{if(d!==help.parentElement)d.open=false;});
    const openRun=event.target.closest?.('[data-open-run]');
    if(openRun){selectRun(openRun.dataset.openRun);location.hash='#/overview';refresh();return;}
    const metric=event.target.closest?.('[data-metric]');
    if(metric){selectedMetric=metric.dataset.metric;render();return;}
    const trainingLink=event.target.closest?.('[data-open-training]');
    if(trainingLink){selectRun(trainingLink.dataset.openTraining);location.hash='#/training';refresh();return;}
    if(event.target.dataset.selectRun){selectRun(event.target.dataset.selectRun);if(route().page==='history'&&route().id){location.hash=`#/history/${encodeURIComponent(selectedRun)}`;}else refresh();return;}
    if(event.target.id==='recording-prev'||event.target.id==='recording-next'){recordingPage+=event.target.id==='recording-next'?1:-1;selectedRecording='';render();}
    if(event.target.id==='toggle-recording'){hiddenRecording=!hiddenRecording;render();}
    if(event.target.id==='retry-detail'){fetchDetail(detailId);}
    const p=players.get(selectedRecording);
    if(event.target.id==='play-pause' && p){p.playing=!p.playing;p.last=0;event.target.textContent=p.playing?'Пауза':'Продолжить';}
    if(event.target.id==='restart-playback' && p){p.frame=0;p.last=0;drawPlayer(p);}
  });
  main.addEventListener('error',event=>{if(event.target.tagName==='IMG'){const host=event.target.closest('.preview');if(host){host.classList.add('empty');host.textContent='Не удалось загрузить запись. Откройте запись по ссылке ниже или выберите другую запись.';}}},true);
  main.addEventListener('keydown',event=>{if(event.key==='Escape'){const hint=main.querySelector('.metric-help[open]');if(hint){hint.open=false;hint.querySelector('summary').focus();}}});
  window.addEventListener('hashchange',()=>{if(selectRouteRun())refresh();if(route().page==='games')fetchValidation();selectedRecording='';recordingRun='';recordingTask='';recordingPage=0;hiddenRecording=reducedMotion();render();window.scrollTo({top:0,behavior:'instant'});});
  document.getElementById('tensorboard').hostname=location.hostname;
  document.getElementById('manual-play').hostname=location.hostname;
  selectRouteRun();
  refresh();setInterval(refresh,5000);requestAnimationFrame(animatePlayer);
})();
