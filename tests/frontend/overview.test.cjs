const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function harness(search = '', hash = '#/overview', options = {}) {
  const listeners = {};
  const elements = new Map();
  const get = id => elements.get(id) || (elements.set(id, {id, hostname: '', href: '/archive', addEventListener(type, listener) { listeners[`${id}:${type}`] = listener; } }), elements.get(id));
  const document = {getElementById: get, activeElement: null};
  const context = {...options.globals, document, URL, URLSearchParams, Intl, Date, Map, Set, Number, String, Math, Array, JSON, encodeURIComponent, decodeURIComponent, location: {hash, hostname: 'localhost', search, href: `http://localhost/${search}${hash}`}, window: {matchMedia: () => ({matches: options.reducedMotion === true}), history: {replaceState(_state,_title,url) { context.location.href=url; context.location.search=new URL(url).search; }}, addEventListener() {}}, Image: class {}};
  let source = fs.readFileSync(path.join(__dirname, '../../src/carryrl/dashboard/assets/overview.js'), 'utf8');
  if(options.skipRender)source=source.replace('function render() {', 'function render() {return;');
  const exposed = `globalThis.ui = {games, resultBrief, checkpointLabel, validationWorkbench, setValidationState({loading=false,error=''}) {validationLoading=loading;validationError=error;}, playerState(id) {return players.get(id);}, validationChoices, validationRecords, validationPlayback, fetchValidation, startReplay, pollReplay, liveContext, memoryFreshness, setValidation(value) {validation=value;validationRun=value.run_id;}, setReplay(value) {replayJob=value;}, refresh, pixelValidation, trainingOutcomes, taskProgress, rlDetails, resources, selectRouteRun, selectRun, savedComment, failureNotice, recordingGuide, metricHelp, activity, comparison, telemetry, training, chart, skillMap, overview, playback, quality, perfectScore, experimentSummary, experimentSelector, pickerLabel, verdictSummary, trainingStatus, history, diagnostics, technicalMessages, patchChildren, setData(value) {data=value;}, setPage(value) {recordingPage=value; selectedRecording='';}, setRecording(value) {selectedRecording=value;}, state() {return {selectedRun,selectedRecording,recordingPage,recordingTask,data,error,unavailableRunNotice,validationLevel,validationCheckpoint,validationAppearance,validationMode,validationView,replayJob,replayError,replaySubmitting};}};`;
  vm.runInNewContext(source.replace('refresh();setInterval(refresh,5000);requestAnimationFrame(animatePlayer);', exposed), context);
  return {ui: context.ui, document, listeners, location: context.location};
}
function snapshot() {
  return {current: {run_id: 'selected', experiment: {title: 'Проверка памяти'}}, experiments: [], quality: {}, task_guides: {a: {label: 'Игра А'}, b: {label: 'Игра Б'}}, playbacks: {items: Array.from({length: 25}, (_,index)=>({id: `record-${index}`, run_id: index===24?'previous':'selected', task: index===23?'b':'a', url: `/media/run/${index}.gif`, epoch: index, revision: index, created_at: new Date(Date.UTC(2026,0,1,0,index)).toISOString(), outcome: {success: true}}))}};
}
test('recordings default to the selected model and one game; choices are bounded and paginated',()=>{
  const {ui}=harness(); ui.setData(snapshot());
  let html=ui.playback();
  const options=html.match(/<select id="recording"[\s\S]*?<\/select>/)[0];
  assert.equal((options.match(/<option/g) || []).length,8);
  assert.equal(ui.state().selectedRecording,'record-22');
  assert(!options.includes('record-24'));
  assert(!options.includes('record-23'));
  assert(html.includes('из 23'));
  ui.setPage(1); html=ui.playback();
  assert.equal(ui.state().selectedRecording,'record-14');
  assert(html.includes('Записи 9–16'));
});
test('a retained recording stays selected when a newer recording arrives',()=>{
  const {ui}=harness(); const data=snapshot(); ui.setData(data); ui.playback(); ui.setRecording('record-8');
  data.playbacks.items.push({id: 'new', run_id:'selected', task:'a', created_at:'2026-10-01', url:'/media/run/new.gif'});
  ui.playback();
  assert.equal(ui.state().selectedRecording,'record-8');
  assert.equal(ui.state().recordingPage,1);
});
test('detail playback uses its own data, and metric identity names that run',()=>{
  const {ui}=harness(); ui.setData(snapshot());
  const detail={current:{run_id:'historical'},playbacks:{items:[{id:'historic-video',run_id:'historical',task:'a',url:'/media/historical/a.gif',revision:4}]}};
  assert(ui.playback('',detail).includes('historic-video'));
  assert(!ui.playback('',detail).includes('record-22'));
  const html=ui.quality({available:true,value:50,episodes:2,successes:1,evaluated_at:'2026-01-01'},false,detail.current);
  assert(html.includes('<dd>historical</dd>'));
  assert(html.includes('<summary>Технические сведения</summary>'));
});
test('checked version shows collection cycles rather than optimizer update revision',()=>{
  const {ui}=harness();ui.setData(snapshot());
  const html=ui.quality({value:25,measurement_stage:'final',current_weights_measured:false,latest_run_id:'selected',latest_revision:128,run_history:[{run_id:'selected',revision:128,rollout:32,value:25}]});
  assert(html.includes('Последняя проверенная версия · после 32 циклов'));
  assert(!html.includes('после 128 циклов'));
});
test('experiment descriptions are escaped and keep limitations visible',()=>{
  const {ui}=harness();ui.setData(snapshot());
  const html=ui.experimentSummary({run_id:'a',status:'completed',experiment:{title:'<script>x</script>',purpose:'Цель',conclusion:'Результат',limitations:['Мало уровней']}});
  assert(html.includes('&lt;script&gt;'));
  assert(!html.includes('<script>'));
  assert(html.includes('Мало уровней'));
});

class Element {
  constructor(tagName, attrs={}, children=[]) { this.nodeType=1;this.tagName=tagName;this.childNodes=children;this.attrs={...attrs};this.dataset={};this.id=attrs.id || '';this.value=attrs.value || '';this.open=false;children.forEach(child=>child.parent=this); }
  get attributes(){return Object.entries(this.attrs).map(([name,value])=>({name,value}));}
  getAttribute(name){return this.attrs[name] ?? null;}
  hasAttribute(name){return name in this.attrs;}
  setAttribute(name,value){this.attrs[name]=value;}
  removeAttribute(name){delete this.attrs[name];}
  insertBefore(node,before){if(node.parent)node.remove();const index=before?this.childNodes.indexOf(before):this.childNodes.length;this.childNodes.splice(index,0,node);node.parent=this;}
  append(node){this.insertBefore(node,null);}
  get lastChild(){return this.childNodes.at(-1);}
  remove(){const index=this.parent.childNodes.indexOf(this);this.parent.childNodes.splice(index,1);this.parent=null;}
}
test('incremental updates retain media and focused inputs while updating other text',()=>{
  const {ui,document}=harness();
  const canvas=new Element('CANVAS',{id:'game-canvas',width:'96'});canvas.dataset.recording='same';
  const input=new Element('INPUT',{id:'frame-seek',value:'13'});input.value='13';
  const host=new Element('DIV',{},[canvas,input]);document.activeElement=input;
  const nextCanvas=new Element('CANVAS',{id:'game-canvas',width:'96'});nextCanvas.dataset.recording='same';
  const incoming=new Element('DIV',{},[nextCanvas,new Element('INPUT',{id:'frame-seek',value:'0'})]);
  ui.patchChildren(host,incoming);
  assert.equal(host.childNodes[0],canvas);
  assert.equal(host.childNodes[1],input);
  assert.equal(input.getAttribute('value'),'13');
});
test('select value survives moving new options out of its template',()=>{
  const {ui}=harness();
  const existing=new Element('SELECT',{id:'recording',value:'old'});
  const next=new Element('SELECT',{id:'recording',value:'new'},[new Element('OPTION',{value:'new'})]);
  ui.patchChildren(new Element('DIV',{},[existing]),new Element('DIV',{},[next]));
  assert.equal(existing.value,'new');
});

test('paired outcomes show gains and losses separately and keep unknown pairs visible',()=>{
  const {ui}=harness();ui.setData(snapshot());
  const q={comparison:{comparable:true},paired_outcomes:{gained:15,lost:27,retained_wins:128,retained_nonwins:85,known_pairs:255,episodes:256,unknown_pairs:1,games:[{task:'a',gained:2,lost:3,retained_wins:4,known_pairs:15,episodes:16}]}};
  const html=ui.comparison(q);
  assert(html.includes('Новых побед: 15 · Потерянных: 27'));
  assert(html.includes('Сопоставлены 255 из 256'));
  assert(html.includes('Неизвестен хотя бы один исход: 1'));
  assert(html.includes('Игра А'));
  assert(html.includes('<td>15 / 16</td>'));
  assert(!ui.comparison({...q,comparison:{comparable:false}}).includes('Новых побед:'));
  assert(ui.comparison({comparison:{comparable:true}}).includes('не означает отсутствие потерь'));
});

test('perfect-score wins are distinct and historical missing scoring remains non-comparable',()=>{
  const {ui}=harness();
  const measured=ui.perfectScore({episodes:100,perfect_score_successes:40,perfect_score_percent:40,scoring_version:'v2'});
  assert(measured.includes('40%'));
  assert(measured.includes('40 из 100'));
  assert(measured.includes('это не доказывает'));
  const legacy=ui.perfectScore({episodes:100,successes:80});
  assert(legacy.includes('нельзя сравнивать'));
  assert(!legacy.includes('80%'));
});

test('initial-only and unmeasured current models cannot show an improvement delta',()=>{
  const {ui}=harness();ui.setData(snapshot());
  const q={available:true,value:55.86,change_pp:40.23,baseline:15.63,successes:143,episodes:256,measurement_stage:'initial'};
  let html=ui.quality(q,false);
  assert(html.includes('<h2>До этого обучения</h2>'));
  assert(!html.includes('40,23 п.п.'));
  assert(html.includes('Результат после этого обучения ещё не получен'));
  html=ui.quality({...q,measurement_stage:'final',current_weights_measured:false},false);
  assert(!html.includes('40,23 п.п.'));
  assert(html.includes('Текущие веса ещё не проверены'));
});
test('technical warnings and diagnostics from previous models stay collapsed',()=>{
  const {ui}=harness();ui.setData(snapshot());
  const warnings=ui.technicalMessages(['raw/path/checkpoint.json failed','old-run mismatch']);
  assert(warnings.startsWith('<details '));
  assert(!warnings.includes(' open'));
  assert(warnings.includes('Технические сообщения (2)'));
  const html=ui.diagnostics({transfer:{run_id:'old',current_weights:false,rows:[],checkpoint:'/long/raw/path'}});
  assert(html.includes('Текущая модель ещё не проверена'));
  assert(html.includes('<details id="previous-appearance">'));
  assert(!html.includes('<details id="previous-appearance" open'));
  assert(html.indexOf('/long/raw/path')>html.indexOf('Технические сведения'));
});
test('perfect-score provenance uses its own episode count when the main win cohort differs',()=>{
  const {ui}=harness();ui.setData(snapshot());
  const html=ui.perfectScore({episodes:256,perfect_score_episodes:8,perfect_score_successes:4,perfect_score_percent:50,perfect_score_run_id:'selected',perfect_score_scope:'Диагностическая выборка',scoring_version:'v2'});
  assert(html.includes('4 из 8'));
  assert(!html.includes('4 из 256'));
  assert(html.includes('Диагностическая выборка'));
});

test('model selectors use meaningful names and keep unrelated runs in the archive',()=>{
  const {ui}=harness();const data=snapshot();
  data.current.experiment={title:'MLX 100 epochs 400000 examples',picker_label:'Исправление ошибок · 13.10 13:39'};
  data.experiments=[data.current,{run_id:'unrelated',experiment:{picker_label:'Техническая проба'}}];ui.setData(data);
  const html=ui.experimentSelector();
  assert(!html.includes('Техническая проба'));assert(html.includes('Все запуски'));
  assert(!html.includes('MLX'));assert(!html.includes('400000'));
  assert(!html.includes('Выбранная модель:'));
  assert(html.includes('Запуск обучения'));
  assert.equal((html.match(/Исправление ошибок/g) || []).length,1);
  const video=ui.playback();
  assert(!video.includes('MLX'));
});
test('a verified worse verdict gives the exact scoped comparison and baseline selection action',()=>{
  const {ui}=harness();
  const html=ui.verdictSummary({verdict:{state:'worse',successes:143,baseline_successes:155,episodes:256},recommended_run_id:'baseline',recommendation:'Для этого набора оставьте исходную модель.'});
  assert(html.includes('После этого обучения: 143/256'));
  assert(html.includes('до обучения — 155/256'));
  assert(html.includes('Побед стало меньше.'));
  assert(html.includes('155/256'));
  assert(html.includes('Посмотреть исходную модель'));
  assert(html.includes('data-select-run="baseline"'));
});
test('viewing a saved model retains the separate active training state',()=>{
  const {ui}=harness();const data=snapshot();
  data.training_current={run_id:'trainee',running:true,status:'running',experiment:{picker_label:'13.10 15:00 · без итоговой проверки'}};
  ui.setData(data);
  const html=ui.trainingStatus();
  assert(html.includes('Сейчас обучается'));
  assert(html.includes('13.10 15:00'));
  assert(html.includes('Результаты выбранной модели ниже относятся к отдельной проверке'));
});


test('active process remains visible while inspecting a different saved model',()=>{
  const {ui}=harness();const data=snapshot();
  data.training_current={run_id:'live',running:true,phase_label:'Сбор примеров',last_update_at:'2026-10-15T10:00:00Z'};
  ui.setData(data);
  assert(ui.activity().includes('Обучение идёт'));
  assert(ui.activity().includes('data-open-training="live"'));
  assert(ui.training().includes('Разбор обучения'));
});
test('selected metrics expose real saved points and never manufacture missing diagnostics',()=>{
  const {ui}=harness();ui.setData(snapshot());
  const c={telemetry:{rows_count:2,axis_label:'Цикл PPO',series:[{key:'policy_loss',label:'Ошибка политики',unit:'',description:'Последняя эпоха PPO',latest:-.2,points:[{step:1,value:-.4},{step:2,value:-.2}]}]}};
  const html=ui.telemetry(c);
  assert(html.includes('data-metric="policy_loss"'));
  assert(html.includes('Последняя эпоха PPO'));
  assert(html.includes('Цикл PPO'));
  assert(!html.includes('NaN'));
  assert(!html.includes('Infinity'));
  assert(ui.telemetry({}).includes('ещё не сохранены'));
});
test('different validation conditions suppress improvement summaries',()=>{
  const {ui}=harness();const data=snapshot();
  data.quality={value:60,change_pp:null,comparison:{comparable:false,reason:'Условия проверки различаются'},games:[{task:'a',value:60,change_pp:null}]};
  ui.setData(data);
  assert(ui.comparison().includes('Сравнение недоступно'));
  assert(ui.comparison().includes('Условия проверки различаются'));
  assert(!ui.skillMap().includes('Ухудшились'));
});
test('default picker uses selected model metadata when registry has no description',()=>{
  const {ui}=harness();const data=snapshot();
  data.current.experiment={picker_label:'Исправление ошибок · 13.10 13:39'};
  data.training_current={run_id:'selected'};
  data.experiments=[data.current];ui.setData(data);
  const html=ui.experimentSelector();
  assert(html.includes('<option value="" selected>Исправление ошибок · 13.10 13:39</option>'));
});


test('metric explanations disclose meaning, interpretation and limits without nested buttons',()=>{
  const {ui}=harness();ui.setData(snapshot());
  const hint=ui.metricHelp('nll');
  assert(hint.includes('aria-label="Подсказка: Ошибка повторения (NLL)"'));
  assert(hint.includes('не означает больше побед'));
  assert(hint.includes('aria-label="Закрыть подсказку"'));
  assert(hint.startsWith('<details'));
  assert(ui.metricHelp('change').includes('+10 процентных пунктов'));
  assert.equal(ui.metricHelp('missing'), '');
  const html=ui.telemetry({telemetry:{series:[{key:'nll',label:'NLL',latest:.4,points:[]}]}});
  assert(html.includes('</button><details id="help-tile-nll"'));
});


test('model selection is restored from the URL and updated for reloads and bookmarks',()=>{
  const {ui,location}=harness('?run=baseline');
  assert.equal(ui.state().selectedRun,'baseline');
  ui.selectRun('model with spaces');
  assert.equal(ui.state().selectedRun,'model with spaces');
  assert.equal(new URL(location.href).searchParams.get('run'),'model with spaces');
  assert.equal(harness(location.search).ui.state().selectedRun,'model with spaces');
  ui.selectRun('');
  assert.equal(ui.state().selectedRun,'');
  assert.equal(new URL(location.href).searchParams.has('run'),false);
});
test('saved editorial comments are collapsed and identified as potentially outdated',()=>{
  const {ui}=harness();
  const html=ui.savedComment({saved_comment:{conclusion:'Проверка продолжается <b>',limitations:['Старый вывод']}});
  assert(html.startsWith('<details '));
  assert(!html.includes(' open'));
  assert(html.includes('до последних проверок'));
  assert(html.includes('&lt;b&gt;'));
});

test('failed launches display recorded causes with an explicit missing-data fallback',()=>{
  const {ui}=harness();
  assert(ui.failureNotice({status:'error',failure_reason:'epoch snapshot does not match its frozen evaluation'}).includes('не совпал с весами'));
  assert(ui.failureNotice({status:'failed'}).includes('Причина остановки не сохранена'));
  assert.equal(ui.failureNotice({status:'completed'}),'');
});


test('recording legends belong to the displayed game and escape source text',()=>{
  const {ui}=harness();ui.setData(snapshot());
  const html=ui.recordingGuide('interaction',{task_guides:{interaction:{objective:'Подберите предмет',legend:['Жёлтая клетка — предмет <x>'],controls:'E рядом с предметом'}}});
  assert(html.includes('Что на экране'));
  assert(html.includes('Жёлтая клетка — предмет &lt;x&gt;'));
  assert(!ui.recordingGuide('unknown').includes('Что на экране'));
});
test('evaluation-only runs do not display made-up training stages or metrics',()=>{
  const {ui}=harness();const data=snapshot();data.current.evaluation_only=true;ui.setData(data);
  const html=ui.training();
  assert(html.includes('В этом запуске веса не обучались'));
  assert(!html.includes('Динамика обучения'));
  assert(!html.includes('Эпоха'));
  assert(!html.includes('Ошибка повторения по играм'));
});


test('archive detail owns the selected run across navigation and a direct permalink',()=>{
  const {ui,location}=harness('?run=control');
  location.hash='#/history/treatment';
  location.href='http://localhost/?run=control#/history/treatment';
  assert.equal(ui.selectRouteRun(),true);
  assert.equal(ui.state().selectedRun,'treatment');
  assert.equal(new URL(location.href).searchParams.get('run'),'treatment');
  location.hash='#/games';
  assert.equal(ui.selectRouteRun(),false);
  assert.equal(ui.state().selectedRun,'treatment');
  assert.equal(harness('?run=control','#/history/treatment').ui.state().selectedRun,'treatment');
});
test('a run with no recordings offers a real archive route instead of empty filters',()=>{
  const {ui}=harness();const data=snapshot();data.playbacks.items=[];ui.setData(data);
  const html=ui.playback();
  assert(html.includes('В этом запуске не сохранено записей игр'));
  assert(html.includes('<a href="#/history">Выбрать другой запуск в архиве</a>'));
  assert(!html.includes('id="recording-task"'));
  assert(!html.includes('Выберите другой эксперимент или игру'));
});


test('heldout pixel results show separate same-weight game counts without missing zeros',()=>{
  const {ui}=harness(); const d=snapshot(); d.current.method='ppo'; d.current.phase='pixel_evaluation';ui.setData(d);
  assert(ui.pixelValidation().includes('Проверка выполняется'));
  d.quality.pixel_validation={value:25,successes:1,episodes:4,rollout:3,revision:100,canonical_same_weights:false,games:[{task:'a',label:'Игра А',value:25,successes:1,episodes:4,canonical:null}]};
  let html=ui.pixelValidation();
  assert(html.includes('Нет проверки'));
  assert(html.includes('25%'));
  assert(!html.includes('0%'));
  d.quality.pixel_validation.canonical_same_weights=true;
  d.quality.pixel_validation.games[0].canonical={value:75,successes:3,episodes:4};
  html=ui.pixelValidation(); assert(html.includes('75%'));assert(html.includes('одинаковым весам'));
  assert(ui.training().includes('Обычное и пиксельное оформление'));
});

test('PPO training outcomes and guard RSS retain their source and unknown outcomes',()=>{
  const {ui}=harness(); const d=snapshot(); d.current.method='ppo';
  d.current.training_outcomes={rollout:2,games:[{label:'Игра А',successes:1,known:2,episodes:3,mean_reward:-0.5}]};
  d.current.resource_guard={sample:{process_group_rss_bytes:4294967296,disk_free_bytes:21474836480,utc:'2026-10-19T10:00:00Z'},maximum_rss_bytes:12884901888};ui.setData(d);
  const outcomes=ui.trainingOutcomes(d.current);
  assert(outcomes.includes('training.jsonl'));assert(outcomes.includes('1 / 2'));assert(outcomes.includes('Проверочные уровни оцениваются отдельно'));
  const html=ui.resources(d.current);
  assert(html.includes('RSS группы процессов'));assert(html.includes('Лимит RSS'));assert(html.includes('GiB'));assert(html.includes('внешнего наблюдателя'));
});


test('obsolete selected run 404 restores active training and removes only its query selection',async()=>{
  const requests=[];const active=snapshot();active.current.run_id='active-ppo';
  const {ui,location}=harness('?run=obsolete&view=full','#/games/a',{skipRender:true,globals:{AbortSignal,fetch:async url=>{
    requests.push(url);
    return requests.length===1?{ok:false,status:404}:{ok:true,status:200,json:async()=>active};
  }}});
  await ui.refresh();
  assert.deepEqual(requests,['/api/overview?run=obsolete','/api/overview','/api/validation?run=active-ppo']);
  assert.equal(ui.state().selectedRun,'');
  assert.equal(new URL(location.href).searchParams.get('run'),null);
  assert.equal(new URL(location.href).searchParams.get('view'),'full');
  assert.equal(new URL(location.href).hash,'#/games/a');
  assert.equal(ui.state().data.current.run_id,'active-ppo');
  assert.equal(ui.state().error,'');
  assert.equal(ui.state().unavailableRunNotice,'Ранее выбранный запуск недоступен. Показано текущее обучение.');
  await ui.refresh();
  assert.equal(requests.at(-2),'/api/overview');
  assert.equal(requests.at(-1),'/api/validation?run=active-ppo');
});

for(const status of [401,500])test(`selected run HTTP ${status} keeps selection and saved metrics without fallback`,async()=>{
  const requests=[];
  const {ui,location}=harness('?run=valid','#/overview',{skipRender:true,globals:{AbortSignal,fetch:async url=>{
    requests.push(url);return {ok:false,status};
  }}});
  const saved=snapshot();ui.setData(saved);ui.setRecording('record-8');
  await ui.refresh();
  assert.deepEqual(requests,['/api/overview?run=valid']);
  assert.equal(ui.state().selectedRun,'valid');
  assert.equal(new URL(location.href).searchParams.get('run'),'valid');
  assert.equal(ui.state().selectedRecording,'record-8');
  assert.equal(ui.state().data,saved);
  assert(ui.state().error.includes(`HTTP ${status}`));
  assert.equal(ui.state().unavailableRunNotice,'');
});

test('a valid selected run refresh preserves its route and selected recording',async()=>{
  const active=snapshot();const {ui,location}=harness('?run=valid','#/games/a',{skipRender:true,globals:{AbortSignal,fetch:async()=>({ok:true,status:200,json:async()=>active})}});
  ui.setRecording('record-8');await ui.refresh();
  assert.equal(ui.state().selectedRecording,'record-8');
  assert.equal(new URL(location.href).hash,'#/games/a');
  assert.equal(ui.state().selectedRun,'valid');
  assert.equal(ui.state().unavailableRunNotice,'');
});

test('fresh comparison renders preparation without previous BC metrics or fake resources',()=>{
  const {ui}=harness();
  const data=snapshot();
  data.current=null;data.training_current=null;
  data.comparison_plan={state:'preparation',policy_updates:0,budget:{rollouts_per_arm:128,environment_episodes_per_arm:2048},reward:'pending',reward_validation:'pending',initialization:'pending',calibration:'pending',arms:[16,32,64].map(spatial_tokens=>({spatial_tokens,initialization:'pending'}))};
  ui.setData(data);
  for(const html of [ui.overview(),ui.training()]){
    assert(html.includes('Подготовка сравнения 16 / 32 / 64'));
    assert(html.includes('Калибровка общего расписания'));
    assert(html.includes('Независимая проверка награды'));
    assert(html.includes('128'));assert(html.includes('2 048') || html.includes('2 048'));
    assert(html.includes('Архив предыдущих запусков'));
    assert(!html.includes('Ошибка повторения'));assert(!html.includes('NLL'));
    assert(!html.includes('100%'));assert(!html.includes('record-'));
    assert(!html.includes('остановленного запуска'));
    assert(!html.includes('RSS'));
  }
  assert(ui.activity().includes('обучение не запущено'));
  assert(!ui.activity().includes('Активность не подтверждена'));
  assert(ui.resources(null).includes('Обучение не запущено'));
  data.current={run_id:'old_bc',experiment:{title:'Архив BC'},method:'ppo'};
  assert(ui.overview().includes('Архив BC'));
  assert(!ui.overview().includes('<h1>Подготовка сравнения'));
});

test('preparation stages show actual saved resources without implying policy training',()=>{
  const {ui}=harness();
  const data={current:null,training_current:null,comparison_plan:{state:'preparation',arms:[],budget:{},policy_updates:0,preparation_stage:{kind:'reward_fit',status:'running',running:true,resource_sample:{utc:'2026-10-19T10:00:00Z',process_group_rss_bytes:1073741824,disk_free_bytes:2147483648}}}};
  ui.setData(data);
  for(const html of [ui.overview(),ui.training()]){
    assert(html.includes('Сейчас: обучение распознавателя награды'));
    assert(html.includes('Обучение политики ещё не запущено'));
    assert(html.includes('RSS группы процессов'));
    assert(html.includes('Свободно на диске'));
    assert(html.includes('1 GiB'));assert(html.includes('2 GiB'));
    assert(!html.includes('NLL'));assert(!html.includes('ETA'));
  }
  data.comparison_plan.preparation_stage={kind:'reward_validation',status:'failed',running:false,exitcode:3,resource_sample:{}};
  let html=ui.overview();
  assert(html.includes('Последний этап: проверка награды'));
  assert(html.includes('Ошибка. Код завершения: 3'));
  assert(!html.includes('RSS группы процессов'));
  assert(!html.includes('GiB'));
  data.comparison_plan.preparation_stage={kind:'model_initialization',status:'interrupted',running:false};
  html=ui.training();
  assert(html.includes('создание начальных весов'));
  assert(html.includes('Процесс завершился без подтверждённого результата'));
  assert(!html.includes('Сейчас:'));
});


test('training exposes separate PPO checks without claiming no evaluation or merging cohorts',()=>{
  const {ui}=harness(); const d=snapshot();d.current.method='ppo';
  d.quality={available:false,value:null,comparison:{comparable:false,reason:'Разные условия'},diagnostic_series:[{coverage:'full',description:'Собственные условия PPO',protocol_sha256:'separate',history:[{rollout:128,revision:1019,value:3.125,successes:8,episodes:256}]}]};
  ui.setData(d);const html=ui.training();
  assert(html.includes('Отдельная проверка · полный набор'));
  assert(html.includes('Победы: 8 из 256 попыток'));
  assert(html.includes('Сохранённые проверки показаны отдельно ниже'));
  assert(html.includes('Разные условия'));
  assert(!html.includes('Нет игровой проверки этого запуска.'));
});
test('training keeps missing PPO checks distinct from a measured zero',()=>{
  const {ui}=harness();const d=snapshot();d.current.method='ppo';ui.setData(d);
  assert(ui.training().includes('Нет игровой проверки этого запуска.'));
  d.quality={diagnostic_series:[{coverage:'full',description:'Проверено',history:[{rollout:64,revision:512,value:0,successes:0,episodes:256}]}]};
  const html=ui.training();assert(html.includes('Победы: 0 из 256 попыток'));
  assert(!html.includes('Нет игровой проверки этого запуска.'));
});

test('frozen paired evaluation exposes real progress, resources and zero updates on both pages',()=>{
  const {ui}=harness();const data=snapshot();
  const current={run_id:'pair',evaluation_only:true,job_kind:'paired_frozen_evaluation',method:'evaluate',running:true,status:'running',phase_label:'Проверка',phase_units_done:48,phase_units_total:960,reports_completed:1,reports_total:20,optimizer_steps:0,elapsed_seconds:90,eta_seconds:1800,resource_guard:{sample:{process_group_rss_bytes:1073741824,system_available_bytes:2147483648,system_swap_used_bytes:6442450944,disk_free_bytes:21474836480}}};
  current.evaluation_note='Новые: 480; сохранённые: 480 <script>';
  data.current=current;data.training_current=current;ui.setData(data);
  for(const html of [ui.training(),ui.overview()]) {
    assert(html.includes('Парная проверка сохранённых моделей'));
    assert(html.includes('value="48" max="960"'));
    assert(html.includes('<strong>1 / 20</strong>'));
    assert(html.includes('Обновлений весов в этом запуске</span><strong>0</strong>'));
    assert(html.includes('Время проверки'));
    assert(html.includes('До конца эпизодов'));
    assert(html.includes('Подготовка и итоговый аудит отдельно'));
    assert(html.includes('Новые: 480; сохранённые: 480 &lt;script&gt;'));
    assert(!html.includes('480 <script>'));
    assert(html.includes('RSS группы процессов'));
    assert(html.includes('Использовано swap системы'));
    assert(html.includes('6 GiB'));
    assert(html.includes('группу процессов проверки'));
    assert(!html.includes('Прогресс циклов обучения'));
    assert(!html.includes('Ошибка повторения по играм'));
  }
  const activity=ui.activity();assert(activity.includes('Проверка выполняется'));
  assert(activity.includes('48 / 960 эпизодов'));
  assert(activity.includes('Следить за проверкой'));
  assert(activity.includes('Парная проверка навигации'));
  assert(!ui.overview().includes('Где агенту трудно'));
  assert(!ui.overview().includes('Текущая модель ещё не проверена на память'));
  assert(!activity.includes('Обучение идёт'));
  current.status='completed';current.running=false;
  assert(ui.activity().includes('Последняя проверка'));
});

test('evaluation missing or invalid counters stay unknown and labels are escaped',()=>{
  const {ui}=harness();const data=snapshot();
  data.current={evaluation_only:true,status:'running',running:true,method_description:'<script>x</script>',phase_units_done:99,phase_units_total:10,reports_completed:21,reports_total:20};ui.setData(data);
  const html=ui.training();
  assert(html.includes('&lt;script&gt;'));
  assert(!html.includes('<script>'));
  assert(html.includes('Обновлений весов в этом запуске</span><strong>—</strong>'));
  assert(html.includes('Нет счётчика'));
  assert(html.includes('aria-label="Прогресс эпизодов проверки" max="100"'));
  assert(!html.includes('value="99"'));
  data.training_current=data.current;assert(!ui.activity().includes('99 / 10'));
});

test('audited navigation pair is shown separately from all-game quality',()=>{
  const {ui}=harness();const data=snapshot();
  data.current={job_kind:'paired_frozen_evaluation',evaluation_only:true,status:'completed',optimizer_steps:0,paired_evaluation_result:['canonical','pixel'].flatMap(appearance=>['frozen_stochastic','frozen_deterministic','generic_physical_random'].map(policy=>({appearance,policy,control_wins:40,candidate_wins:50,delta_wins:10,episodes:80,positive_paired_seeds:4})))};
  ui.setData(data);const html=ui.training();
  assert(html.includes('Аудит 20 отчётов завершён'));
  assert(html.includes('40 / 80'));
  assert(html.includes('50 / 80'));
  assert(html.includes('не проверка всех игр'));
  assert(html.includes('не 80 независимых уровней'));
  assert(!html.includes('Игровая проверка</h2>'));
  assert(!html.includes('Победы с максимальным счётом'));
});

test('paired comparison uses its own mode labels and does not invent learning rates',()=>{
  const {ui}=harness();const data=snapshot();
  data.current={job_kind:'paired_frozen_evaluation',evaluation_only:true,status:'completed',optimizer_steps:0,paired_comparison:{title:'Сравнение объёма сбора перед обновлениями',control_label:'Сбор 1 эпизод',candidate_label:'Сбор 8 эпизодов'},paired_evaluation_result:['canonical','pixel'].flatMap(appearance=>['frozen_stochastic','frozen_deterministic','generic_physical_random'].map(policy=>({appearance,policy,control_wins:40,candidate_wins:50,delta_wins:10,episodes:80,positive_paired_seeds:4})))};
  ui.setData(data);let html=ui.training();
  assert(html.includes('Сравнение объёма сбора перед обновлениями'));
  assert(html.includes('Сбор 1 эпизод'));
  assert(html.includes('Сбор 8 эпизодов'));
  assert(!html.includes('LR 0,0003'));
  data.current.paired_comparison={title:'Сравнение расчёта целей PPO',control_label:'λ = 0,95',candidate_label:'λ = 1'};
  html=ui.training();
  assert(html.includes('Сравнение расчёта целей PPO'));
  assert(html.includes('λ = 0,95'));
  assert(html.includes('λ = 1'));
  assert(!html.includes('Сбор 8 эпизодов'));
  delete data.current.paired_comparison;html=ui.training();
  assert(html.includes('Сравнение моделей в навигации'));
  assert(html.includes('scope="col">Контроль'));
  assert(!html.includes('learning rate'));
  data.current.paired_comparison={title:'<script>x</script>',control_label:'<b>left</b>',candidate_label:'<b>right</b>'};html=ui.training();
  assert(html.includes('&lt;script&gt;'));
  assert(html.includes('&lt;b&gt;left&lt;/b&gt;'));
  assert(!html.includes('<script>'));
});

test('PPO activity percent denotes collection cycles rather than accepted updates',()=>{
  const {ui}=harness();const data=snapshot();
  data.training_current={running:true,status:'running',method:'ppo',training_percent:62.5};
  ui.setData(data);assert(ui.activity().includes('% циклов'));
  assert(!ui.activity().includes('% обновлений'));
  data.training_current.method='imitation';assert(ui.activity().includes('% обновлений'));
});

function validationFixture() {
  return {run_id:'selected',checkpoints:[{id:'cp0',rollout:0,weights_revision:0,available:true},{id:'cp1',rollout:1,weights_revision:8,available:true}],tasks:[{task:'a',levels:[{level_index:0,level_id:'a-0'},{level_index:1,level_id:'a-1'}]}],records:[{checkpoint_id:'cp0',task:'a',level_index:0,appearance:'canonical',mode:'greedy',steps:32,outcome:{success:true,score:5}},{checkpoint_id:'cp1',task:'a',level_index:0,appearance:'canonical',mode:'stochastic',steps:99,outcome:{success:false}}]};
}
test('level history filters action modes and preserves chosen level and checkpoint on new snapshots',()=>{
  let requests=0;const {ui,listeners}=harness('','#/games/a',{skipRender:true,globals:{fetch:()=>{requests++;}}});
  ui.setData(snapshot());const value=validationFixture();ui.setValidation(value);
  const html=ui.validationWorkbench('a');assert(html.includes('Один уровень через всё обучение'));assert(html.includes('Победа'));assert(!html.includes('<td>99</td>'));assert.equal(requests,0);
  listeners['content:change']({target:{id:'validation-level',value:'1'}});
  listeners['content:change']({target:{id:'validation-checkpoint',value:'cp0'}});
  value.checkpoints.push({id:'cp2',rollout:2,available:true});ui.setValidation(value);ui.validationWorkbench('a');
  assert.equal(ui.state().validationLevel,'1');assert.equal(ui.state().validationCheckpoint,'cp0');assert.equal(requests,0);
});
test('manual replay sends exactly selected frozen validation request once and disables while pending',async()=>{
  const requests=[];let resolve;
  const {ui,listeners}=harness('','#/games/a',{skipRender:true,globals:{fetch:(url,options)=>{requests.push({url,options});return new Promise(r=>{resolve=r;});}}});
  ui.setData(snapshot());ui.setValidation(validationFixture());ui.validationChoices('a');
  listeners['content:change']({target:{id:'validation-level',value:'1'}});
  listeners['content:change']({target:{id:'validation-appearance',value:'pixel'}});
  const pending=ui.startReplay();await ui.startReplay();
  assert.equal(requests.length,1);assert.equal(requests[0].url,'/api/replay');assert.deepEqual(JSON.parse(requests[0].options.body),{run:'selected',checkpoint:'cp1',task:'a',level_index:1,appearance:'pixel',mode:'greedy'});
  assert(ui.validationWorkbench('a').includes('id="start-replay" disabled'));assert(ui.state().replaySubmitting);
  resolve({ok:true,json:async()=>({id:'job',status:'running'})});await pending;
  assert(ui.validationWorkbench('a').includes('Попытка выполняется'));await ui.startReplay();assert.equal(requests.length,1);
});
test('replay service rejection shows useful error and permits an explicit retry',async()=>{
  const {ui}=harness('','#/games/a',{skipRender:true,globals:{fetch:async()=>({ok:false,status:409,json:async()=>({error:'Другой эпизод уже выполняется'})})}});
  ui.setData(snapshot());ui.setValidation(validationFixture());ui.validationChoices('a');await ui.startReplay();
  const html=ui.validationWorkbench('a');assert(html.includes('Другой эпизод уже выполняется'));assert(html.includes('role="alert"'));assert(!html.includes('id="start-replay" disabled'));
});
test('completed replay uses local replay media and survives refresh without selecting a new checkpoint',async()=>{
  const {ui}=harness('','#/games/a',{skipRender:true,globals:{AbortSignal,fetch:async url=>({ok:true,json:async()=>url.startsWith('/api/validation')?validationFixture():({id:'job',status:'completed',replay:{gif_url:'/replay-media/selected/job/a.gif',sprite_url:'/replay-media/selected/job/a.png',frame_count:5,frame_width:96,frame_height:96}})})}});
  ui.setData(snapshot());ui.setValidation(validationFixture());ui.validationChoices('a');
  ui.setReplay({id:'job',status:'running',request:{run:'selected',checkpoint:'cp1',task:'a',level_index:0,appearance:'canonical',mode:'greedy'}});await ui.pollReplay();
  const html=ui.validationPlayback('a');assert(html.includes('/replay-media/selected/job/a.gif'));assert(html.includes('data-recording="job"'));
  ui.setData(snapshot());assert(ui.validationPlayback('a').includes('data-recording="job"'));assert.equal(ui.state().validationCheckpoint,'cp1');
});
test('sampled history and unavailable checkpoint cannot start a replay',async()=>{
  let requests=0;const {ui,listeners}=harness('','#/games/a',{skipRender:true,globals:{fetch:()=>{requests++;}}});ui.setData(snapshot());const value=validationFixture();ui.setValidation(value);ui.validationChoices('a');
  listeners['content:change']({target:{id:'validation-mode',value:'stochastic'}});await ui.startReplay();assert.equal(requests,0);assert(ui.validationWorkbench('a').includes('<td>99</td>'));
  listeners['content:change']({target:{id:'validation-mode',value:'greedy'}});value.checkpoints[1].available=false;await ui.startReplay();assert.equal(requests,0);
});
test('live context distinguishes budget deadline, evaluation progress, memory pause and stale samples',()=>{
  const {ui}=harness();ui.setData(snapshot());
  const html=ui.liveContext({running:true,phase:'evaluation',evaluation_task:'a',evaluation_policy:'stochastic',evaluation_step:17,budget:{remaining_seconds:7200,total_seconds:14400,reserve_seconds:1800,hard_deadline_at:'2026-10-20T12:00:00Z',training_deadline_at:'2026-10-20T11:30:00Z'}});
  assert(html.includes('Игра А'));assert(html.includes('шаг 17'));assert(html.includes('Общий бюджет 4 ч 0 мин, включая паузы'));assert(html.includes('не прогноз'));
  const memory=ui.memoryFreshness({resource_guard:{memory_throttled:true,sample_stale:true,sample_age_seconds:150,wait_duration_seconds:60}});assert(memory.includes('Пауза по памяти'));assert(memory.includes('текущее потребление и пауза не подтверждены'));assert(memory.includes('Ожидание памяти: 1:00'));
});

test('stale evaluation fields cannot describe collection or stopped runs as an active evaluation',()=>{
  const {ui}=harness();ui.setData(snapshot());
  const c={running:true,phase:'collecting',evaluation_task:'a',evaluation_policy:'generic_physical_random',evaluation_step:14};
  assert(!ui.liveContext(c).includes('Сейчас проверяется'));assert(!ui.liveContext(c).includes('Веса во время проверки'));
  assert(!ui.liveContext({...c,running:false,phase:'evaluation'}).includes('Сейчас проверяется'));
  for(const phase of ['initial_evaluation','evaluation','interim_evaluation','pixel_evaluation','final_evaluation']){
    const html=ui.liveContext({...c,phase});assert(html.includes('Сейчас проверяется'));assert(html.includes('Случайные действия'));assert(!html.includes('generic_physical_random'));
  }
  assert(ui.liveContext({...c,phase:'evaluation',evaluation_policy:'frozen_deterministic'}).includes('Самое вероятное действие'));
  assert(ui.liveContext({...c,phase:'evaluation',evaluation_policy:'frozen_stochastic'}).includes('Выбор по вероятностям'));
});

test('level labels remain concise and exact condition identity stays in technical details',()=>{
  const {ui}=harness();ui.setData(snapshot());const value=validationFixture();value.tasks[0].levels[0].level_id='a'.repeat(64);Object.assign(value.records[0],{level_id:'a'.repeat(64),semantic_level_id:'semantic',seed:1000000,difficulty:0,instruction:'<move>',protocol_sha256:'protocol'});ui.setValidation(value);
  const html=ui.validationWorkbench('a');assert(html.includes('>Уровень 1</option>'));assert(!html.includes('Уровень 1 · '));assert(html.includes('<details id="level-conditions">'));assert(html.includes('seed: 1000000'));assert(html.includes('&lt;move&gt;'));assert(!html.includes('<move>'));
});

function persistedReplay(id,checkpoint) {
  return {id,checkpoint_id:checkpoint,manual:true,task:'a',level_index:0,appearance:'canonical',mode:'greedy',weights_revision:checkpoint==='cp0'?0:8,frame_count:5,frame_width:96,frame_height:96,created_utc:'2026-10-20T12:00:00Z',url:`/replay-media/selected/${id}/a.gif`,sprite_url:`/replay-media/selected/${id}/a.png`,actions:[{action:{keys:['left']}}],request:{run:'selected',checkpoint,task:'a',level_index:0,appearance:'canonical',mode:'greedy'}};
}
test('persisted manual recordings survive reload and switching checkpoints after a second replay',()=>{
  const {ui,listeners}=harness('','#/games/a',{skipRender:true});ui.setData(snapshot());const value=validationFixture();value.replays=[persistedReplay('first','cp0'),persistedReplay('second','cp1')];value.records.push(...value.replays.map(p=>({...p,replay_id:p.id,steps:5,outcome:{success:true,score:5}})));ui.setValidation(value);ui.validationChoices('a');
  assert(ui.validationPlayback('a').includes('data-recording="second"'));
  listeners['content:change']({target:{id:'validation-checkpoint',value:'cp0'}});assert(ui.validationPlayback('a').includes('data-recording="first"'));
  ui.setReplay({id:'second',status:'completed',request:value.replays[1].request,replay:value.replays[1]});assert.equal(ui.validationRecords('a').filter(r=>r.replay_id==='second').length,1);
  listeners['content:change']({target:{id:'validation-checkpoint',value:'cp1'}});const html=ui.validationPlayback('a');assert.equal((html.match(/data-recording="second"/g)||[]).length,1);assert(!html.includes('id="recording"'));
});
test('catalog recovers an active replay after reload without submitting a new job',async()=>{
  const requests=[];const value=validationFixture();value.active_job={id:'recover',status:'running',request:persistedReplay('recover','cp0').request};const {ui}=harness('','#/games/a',{skipRender:true,globals:{AbortSignal,fetch:async url=>{requests.push(url);return {ok:true,json:async()=>value};}}});ui.setData(snapshot());await ui.fetchValidation();assert.equal(ui.state().replayJob.id,'recover');assert(ui.validationWorkbench('a').includes('id="start-replay" disabled'));assert.deepEqual(requests,['/api/validation?run=selected']);
});

test('finished timeout replay is not presented as an unfinished recording',()=>{
  const {ui}=harness();const value=snapshot();
  value.playbacks.items=[{id:'timeout',run_id:'selected',task:'a',truncated:true,
    steps:80,outcome:{known:true,success:false,failure:false,score:2},url:'/media/selected/timeout.gif'}];
  ui.setData(value);const html=ui.playback('a');
  assert(html.includes('Лимит ходов'));assert(!html.includes('Не завершено'));
  assert(html.includes('Игра А'));
});


test('overview has one run choice, a next action, four weak games and no media player',()=>{
  const {ui}=harness();const d=snapshot();d.experiments=[d.current];
  d.quality={value:25,successes:1,episodes:4,measurement_stage:'initial',games:Array.from({length:6},(_,i)=>({task:`game${i}`,value:i,successes:i,episodes:16}))};ui.setData(d);
  const html=ui.overview();
  assert.equal((html.match(/id="experiment-run"/g)||[]).length,1);
  assert(!html.includes('id="recording-run"'));assert(!html.includes('selected-model'));
  assert(!html.includes('recording-panel'));assert(!html.includes('game-canvas'));
  assert.equal((html.match(/class="skill-row"/g)||[]).length,4);
  assert(html.includes('Выбрать игру и посмотреть попытку агента'));
  assert(!html.includes('Память: выбор'));assert(!html.includes('Проверка пиксельного оформления'));
});
test('initial-only brief has no duplicate baseline and keeps unrelated scoring inside details',()=>{
  const {ui}=harness();const d=snapshot();
  d.quality={value:25,successes:1,episodes:4,baseline:25,change_pp:0,measurement_stage:'initial',perfect_score_percent:80,perfect_score_successes:8,perfect_score_episodes:10,perfect_score_revision:9};ui.setData(d);
  const html=ui.resultBrief();
  assert(html.includes('Начальная проверка'));assert(html.includes('точка отсчёта'));
  assert(!html.includes('Исходная доля побед'));assert(!html.includes('Начальная модель: <strong>25%'));
  assert(!html.includes('0 п.п.'));assert(html.indexOf('80%')>html.indexOf('id="result-details"'));
  d.quality={...d.quality,measurement_stage:'final',current_weights_measured:false};
  assert(ui.resultBrief().includes('текущие веса ещё не проверены'));
  assert(!ui.resultBrief().includes('После этого обучения'));
});
test('incomparable baseline cannot be presented as a valid training comparison',()=>{
  const {ui}=harness();const d=snapshot();d.quality={value:25,successes:1,episodes:4,baseline:80,change_pp:-55,measurement_stage:'final',comparison:{comparable:false,reason:'Разные условия'}};ui.setData(d);
  assert(!ui.resultBrief().includes('Начальная модель: <strong>80%'));
  assert(!ui.resultBrief().includes('-55 п.п.'));assert(!ui.quality(d.quality).includes('Начальная модель: 80%'));
});
test('a previous measurement verdict never claims the current weights improved',()=>{
  const {ui}=harness();const d=snapshot();d.quality={value:50,measurement_stage:'final',current_weights_measured:false};ui.setData(d);
  const html=ui.verdictSummary({verdict:{state:'improved',successes:2,baseline_successes:1,episodes:4,label:'После этого обучения лучше'}});
  assert(!html.includes('После этого обучения'));assert(html.includes('Качество текущих весов ещё не подтверждено'));
});
test('validation loading, failure and empty states exclude impossible player actions',()=>{
  const {ui}=harness();ui.setData(snapshot());ui.setValidationState({loading:true});
  let html=ui.validationWorkbench('a');assert(html.includes('Загружаем проверочные уровни'));
  assert(!html.includes('Версии модели пока не сохранены'));assert(!html.includes('Проиграть уровень'));
  ui.setValidationState({error:'Сервер не ответил'});html=ui.validationWorkbench('a');
  assert(html.includes('role="alert"'));assert(html.includes('id="retry-validation"'));
  assert(!html.includes('Загружаем'));assert(!html.includes('Проиграть уровень'));
  ui.setValidation({run_id:'selected',tasks:[],checkpoints:[]});ui.setValidationState({});html=ui.validationWorkbench('a');
  assert(html.includes('Проверочных уровней пока нет'));assert(!html.includes('Нажмите «Проиграть уровень»'));
});
test('game workbench exposes one version table and plain action modes without repeated game selectors',()=>{
  const {ui}=harness();const d=snapshot();ui.setData(d);ui.setValidation(validationFixture());
  const html=ui.games('a');assert(!html.includes('id="validation-game"'));assert(!html.includes('id="validation-checkpoint"'));
  assert(!html.includes('id="recording-run"'));assert(!html.includes('id="recording-task"'));
  assert(html.includes('Проверочный уровень'));assert(html.includes('Самое вероятное действие'));
  assert(html.includes('Выбор по вероятностям'));assert(html.includes('Случайные действия'));
  assert(html.includes('Проверочные уровни не используются для обновления модели'));
  assert(html.includes('Начальная модель'));assert(html.includes('После 1 цикла'));
  assert(html.indexOf('<strong>Цель:</strong>')<html.indexOf('validation-workbench'));
});
test('unavailable and probability-mode empty recordings explain the actionable blocker',()=>{
  const {ui,listeners}=harness('','#/games/a',{skipRender:true});ui.setData(snapshot());const v=validationFixture();ui.setValidation(v);ui.validationChoices('a');
  listeners['content:change']({target:{id:'validation-mode',value:'stochastic'}});
  let html=ui.validationPlayback('a');assert(!html.includes('Нажмите «Проиграть уровень»'));assert(html.includes('Для новой записи выберите «Самое вероятное действие»'));
  listeners['content:change']({target:{id:'validation-mode',value:'greedy'}});v.checkpoints[1].available=false;
  html=ui.validationPlayback('a');assert(!html.includes('Нажмите «Проиграть уровень»'));assert(html.includes('доступную версию модели'));
});
test('validation selections persist across reload and preserve selected run query',()=>{
  const {ui,listeners,location}=harness('?run=selected&other=keep','#/games/a',{skipRender:true});ui.setData(snapshot());ui.setValidation(validationFixture());ui.validationChoices('a');
  for(const [id,value] of [['validation-level','1'],['validation-checkpoint','cp0'],['validation-appearance','pixel'],['validation-mode','stochastic']])listeners['content:change']({target:{id,value}});
  const query=new URL(location.href).searchParams;
  assert.equal(query.get('run'),'selected');assert.equal(query.get('other'),'keep');assert.equal(query.get('task'),'a');
  const restored=harness(location.search,'#/games/a').ui;restored.setData(snapshot());restored.setValidation(validationFixture());restored.validationChoices('a');
  assert.equal(restored.state().validationLevel,'1');assert.equal(restored.state().validationCheckpoint,'cp0');assert.equal(restored.state().validationAppearance,'pixel');assert.equal(restored.state().validationMode,'stochastic');
});
test('single recording hides its selector and reduced motion pauses the sprite player',()=>{
  const {ui}=harness('','#/games/a',{reducedMotion:true});const d=snapshot();d.playbacks.items=[{id:'one',run_id:'selected',task:'a',mode:'greedy',appearance:'canonical',level_index:0,rollout:0,url:'/media/one.gif',sprite_url:'/media/one.png',frame_width:96,frame_height:96,frame_count:4}];ui.setData(d);
  const html=ui.playback('a');assert(!html.includes('id="recording"'));assert(!html.includes('id="recording-run"'));assert(!html.includes('id="recording-task"'));
  assert.equal(ui.playerState('one').playing,false);assert(html.includes('Самое вероятное действие'));assert(html.includes('Обычное оформление'));assert(html.includes('Начальная модель'));assert(html.includes('Уровень 1'));
});
test('memory diagnostics retain missing choices in the complete denominator',()=>{
  const {ui}=harness();ui.setData(snapshot());
  let html=ui.diagnostics({memory:{run_id:'selected',current_weights:true,modes:[{mode:'neutral_wait',first_choose_correct:0,first_choose_observed:0,missing_choices:32,episodes:32,paired_both_first_choose_correct:0,geometries:16}]}});
  assert(html.includes('0 / 32 попыток'));assert(html.includes('Выбор сделан: 0; без выбора: 32'));assert(html.includes('Без нажатий во время ожидания'));assert(html.includes('историю своих нажатий'));
  html=ui.diagnostics({memory:{run_id:'old',modes:[{mode:'normal',first_choose_correct:2,first_choose_observed:3,paired_both_first_choose_correct:1,geometries:2}]}});
  assert(html.includes('2 из 3 сделанных выборов'));assert(html.includes('Общий объём попыток неизвестен'));
});


test('training prioritizes quality and keeps practice, resources and optimization collapsed',()=>{
  const {ui}=harness();const d=snapshot();d.current={...d.current,method:'ppo',running:true,status:'running',phase:'evaluation',phase_label:'Проверка',phase_unit:'episodes',phase_units_done:378,phase_units_total:768,rollouts:16,rollouts_total:128,cycle_eta_seconds:3600,method_description:'PPO technical',training_outcomes:{rollout:16,games:[{task:'a',successes:1,known:2,episodes:3}]}};
  d.quality={value:25,successes:1,episodes:4,measurement_stage:'initial',run_history:[{rollout:0,value:25}]};ui.setData(d);
  const html=ui.training();assert(html.indexOf('Меняется ли качество игры')<html.indexOf('practice-results'));
  for(const id of ['optimization-details','computer-load','training-settings','practice-results']){assert(html.includes(`<details id="${id}"`));assert(!html.includes(`<details id="${id}" open`));}
  assert(html.includes('Проверено 378 из 768 попыток'));assert(html.includes('До конца циклов примерно'));
  assert(html.includes('победы: 1 из 3 попыток · неизвестные исходы: 1'));
  assert(html.includes('Пока сохранено одно измерение'));assert(!html.includes('aria-label="Доля побед на проверочных уровнях"'));
  d.current.phase='collecting';assert(ui.training().includes('Собрано 378 из 768 попыток'));
});

test('run categories separate training, checks and preparation without inventing unknown types',()=>{
  const {ui,listeners}=harness('','#/history',{skipRender:true});
  const d=snapshot();
  d.experiments=[
    {run_id:'trained',kind:'sequence_training',experiment:{display_name:'Trained'}},
    {run_id:'random',kind:'random',experiment:{display_name:'Random check'}},
    {run_id:'audit',kind:'frozen_teacher_rescue_audit',experiment:{display_name:'Audit check'}},
    {run_id:'prepared',kind:'prepare_history_reward',experiment:{display_name:'Prepared only'}},
    {run_id:'unknown',kind:'unrecognized',experiment:{display_name:'Unknown kind'}},
  ];
  ui.setData(d);
  let html=ui.history();
  assert(html.includes('Trained'));
  assert(!html.includes('Random check'));assert(!html.includes('Unknown kind'));
  const select=value=>listeners['content:click']({target:{closest:()=>({dataset:{historyFilter:value}})}});
  select('evaluation');html=ui.history();
  assert(html.includes('Random check'));assert(html.includes('Audit check'));assert(!html.includes('Prepared only'));
  select('technical');html=ui.history();
  assert(html.includes('Prepared only'));assert(!html.includes('Trained'));
  select('all');assert(ui.history().includes('Unknown kind'));
});

test('technical runs never show policy training stages or promise a future game evaluation',()=>{
  const {ui}=harness();const d=snapshot();
  d.current={run_id:'prepared',technical_only:true,status:'completed',method:'prepare_history_features',experiment:{title:'Подготовка признаков истории',purpose:'Обработка игровых историй',conclusion:'Смотреть артефакты',verdict:{state:'pending'},recommendation:'Дождитесь итоговой проверки'}};
  d.quality={};ui.setData(d);
  for(const html of [ui.overview(),ui.training()]){
    assert(html.includes('Подготовка признаков истории'));
    assert(html.includes('Параметры технического запуска'));
    assert(!html.includes('Дождитесь итоговой проверки'));
    assert(!html.includes('Завершённые циклы обучения'));
    assert(!html.includes('Результат после обучения'));
  }
});


test('all-game progress separates curriculum reward windows, practice and heldout wins',()=>{
  const {ui}=harness();const d=snapshot();d.current.method='ppo';
  d.current.rl_details={task_curriculum:{tasks:{a:{stage:1,windows:{bridge:[{success:true},{success:false}]}}}}};
  d.current.training_outcomes={games:[{task:'a',successes:0,known:1,episodes:2}]};
  d.quality.games=[{task:'a',successes:3,episodes:8}];ui.setData(d);
  const html=ui.taskProgress(d.current);
  assert(html.includes('Переходный'));
  assert(html.includes('Сигнал модели награды: 1 / 2'));
  assert(html.includes('3 / 8'));
  assert(html.includes('неизвестно 1'));
  assert(html.includes('Не измерено'));
  assert(html.includes('Игра Б'));
  assert(!html.includes('NaN'));
});
test('new RL diagnostics stay collapsed and disclose auxiliary reward limits and missing data',()=>{
  const {ui}=harness();const d=snapshot();d.current.method='ppo';ui.setData(d);
  d.current.rl_details={reward_composition:{version:'carryrl.reward.outcome_potential_text.v1'}};
  const missing=ui.rlDetails(d.current);
  assert(missing.includes('<details id="rl-diagnostics" class="panel">'));
  assert(!missing.includes(' open'));
  assert(missing.includes('Это не оценка хорошего действия'));
  assert(missing.includes('один раз за класс'));
  assert(missing.includes('не сохранены'));
  assert(!missing.includes('удержаний: 0'));
  d.current.rl_details={control_coefficient:.8,updates:[{update:1,approx_kl:.01,entropy:2,clip_fraction:.2,gradient_norm:3,clip_coefficient:.4}],episodes:[{task:'a',action_summary:{steps:3,frequencies:{'<script>':2},held:1,released:2,unsupported_inputs:1},reward_totals:{progress:.5,control_bonus:.01},control_events:{unsupported_first_use:1}}]};
  const html=ui.rlDetails(d.current);
  assert(html.includes('недопустимых вводов: 1'));
  assert(html.includes('Первое применение недопустимого класса: 1'));
  assert(html.includes('Продвижение:'));
  assert(html.includes('Измерения применённых обновлений'));
  assert(html.includes('&lt;script&gt;'));
  assert(!html.includes('<script>'));
});

test('cycle preparation shows actual reward phase without previous policy results',()=>{
  const {ui}=harness();const d=snapshot();d.current=null;d.training_current=null;
  d.cycle={status:'prepared',phase_label:'Подготовка и проверка наград',policy_created:false,
    age_seconds:10,budget_valid:true,budget:{elapsed_seconds:600,remaining_seconds:85800,started:true},
    rewards:[{id:'visual_gate',status:'completed',confirmed:false}]};ui.setData(d);
  for(const html of [ui.training(),ui.overview()]){
    assert(html.includes('Модель политики ещё не создана'));
    assert(html.includes('Подготовка и проверка наград'));
    assert(html.includes('Новый цикл · 24 часа'));
    assert(html.includes('Результат не подтверждён'));
    assert(!html.includes('Ошибка повторения'));
    assert(!html.includes('Начальная модель'));
    assert(!html.includes('Доля побед'));
  }
  assert(ui.activity().includes('Модель политики ещё не создана'));
});

test('cycle card preserves game metrics and marks stale stopped budgets and failed reward stages',()=>{
  const {ui}=harness();const d=snapshot();d.current.method='ppo';
  d.cycle={status:'stopped',phase_label:'Подготовка и проверка наград',policy_created:true,
    age_seconds:500,stale:true,budget_valid:true,
    budget:{elapsed_seconds:600,remaining_seconds:85800,started:true,frozen:true},
    failure_reason:'Gate rejected <script>',rewards:[{id:'text_gate',status:'failed',confirmed:false}]};
  ui.setData(d);const html=ui.training();
  assert(html.includes('Оставалось при остановке'));
  assert(html.includes('Отсчёт зафиксирован'));
  assert(html.includes('Данные устарели'));
  assert(html.includes('Gate rejected &lt;script&gt;'));
  assert(html.includes('Проверка текстового оценщика'));
  assert(html.includes('Меняется ли качество игры'));
  assert(html.includes('Проверки отдельных игр и записи'));
  assert(!html.includes('Модель политики ещё не создана'));
});

test('cycle reward stages distinguish corpus preparation, fitting, gates and retries',()=>{
  const {ui}=harness();const d=snapshot();d.current=null;
  d.cycle={status:'running',phase_label:'Подготовка и проверка наград',policy_created:false,
    age_seconds:1,budget_valid:true,budget:{elapsed_seconds:600,remaining_seconds:85800,started:true},
    rewards:[
      {id:'text_data',status:'completed',confirmed:true},
      {id:'visual_fit',status:'stopped',actual_exit_code:1,confirmed:false},
      {id:'visual_fit_entrypoint_fix',status:'completed',confirmed:true},
      {id:'visual_gate',status:'stopped',actual_exit_code:1,confirmed:false},
      {id:'visual_gate_control_coverage_fix',status:'running',confirmed:false},
      {id:'text_fit',status:'starting',confirmed:false},
      {id:'text_sealed',status:'pending',confirmed:false},
    ]};ui.setData(d);const html=ui.training();
  assert(html.includes('<span>Подготовка текстовых примеров</span><strong>Завершение подтверждено'));
  assert(html.includes('<span>Обучение текстового оценщика</span><strong>Запускается'));
  assert(html.includes('Обучение визуального оценщика · первая попытка'));
  assert(html.includes('Обучение визуального оценщика · повтор после исправления запуска'));
  assert(html.includes('Проверка визуального оценщика · первая попытка'));
  assert(html.includes('Проверка визуального оценщика · повтор'));
  assert(html.includes('Независимая проверка текста'));
  assert(html.includes('Ошибка · код 1'));
  assert(!html.includes('Текстовая награда'));
});

test('known reward quality failure keeps raw artifact details inside technical information',()=>{
  const {ui}=harness();const d=snapshot();d.current=null;
  const message='Текстовый оценщик не прошёл проверку качества. Обучение игрового агента не запускалось.';
  const raw='CycleStopped: Stage artifact gate failed: /tmp/<run>/reward_text/manifest.json:ready';
  d.cycle={status:'stopped',phase_label:'Подготовка и проверка наград',policy_created:false,
    failure_reason:message,failure_details:raw,age_seconds:10,rewards:[]};ui.setData(d);
  let html=ui.training();const technical=html.indexOf('<details id="cycle-source"');
  assert(html.includes(`<p class="warning">${message}</p>`));
  assert(technical>0);
  assert(!html.slice(0,technical).includes('CycleStopped'));
  assert(html.slice(technical).includes('Техническая причина остановки'));
  assert(html.slice(technical).includes('/tmp/&lt;run&gt;/reward_text/manifest.json:ready'));
  d.cycle.failure_reason='CycleStopped: Subprocess did not complete successfully';
  d.cycle.failure_details=null;html=ui.training();
  assert(html.includes('Subprocess did not complete successfully'));
  assert(!html.includes('не прошёл проверку качества'));
});

test('saved terminal cycle shows a stored result while live old heartbeats remain stale',()=>{
  const {ui}=harness();const d=snapshot();d.current=null;
  d.cycle={status:'stopped',phase_label:'Подготовка и проверка наград',policy_created:false,
    terminal_result_saved:true,stale:false,age_seconds:3500,rewards:[]};ui.setData(d);
  let html=ui.training();
  assert(html.includes('Итог сохранён'));
  assert(html.includes('Сохранено при остановке; обновления не ожидаются'));
  assert(!html.includes('Данные устарели'));
  assert(!ui.activity().includes('данные устарели'));
  d.cycle.status='running';d.cycle.running=true;d.cycle.terminal_result_saved=false;
  d.cycle.stale=true;html=ui.training();
  assert(html.includes('Данные устарели'));
  assert(!html.includes('Итог сохранён'));
  assert(ui.activity().includes('данные устарели'));
});


test('factorized controls distinguish missing instruction semantics from evaluator abstention',()=>{
  const {ui}=harness(); const d=snapshot(); d.current.method='ppo';
  d.current.rl_details={reward_composition:{version:'carryrl.reward.outcome_potential_text.v2'},episodes:[{task:'a',control_table:{predictions:[
    {candidate:'key:a',status:'supported',applicability_decision:{value:'supported',accepted:true,confidence:.99},mode_decision:{value:'unspecified',accepted:false,reason:'insufficient_information',confidence:.95}},
    {candidate:'key:b',status:'unknown',applicability_decision:{value:'unsupported',accepted:false,reason:'low_confidence',confidence:.4},mode_decision:{value:'held',accepted:false,reason:'inconsistent',confidence:.9}},
    {candidate:'<script>',status:'unknown',mode:'unknown'}],pair_decisions:[{left:'key:a',right:'key:b',decision:{value:'unspecified',reason:'insufficient_information',confidence:.9}}]},control_events:{unsupported_step_penalty:2,supported_award:1},control_event_reasons:{low_confidence:1},control_budget:{positive_paid:.0005,negative_contribution:-.0002},control_coefficient:.5}]};
  ui.setData(d); const html=ui.rlDetails(d.current);
  for(const text of ['Не описано в инструкции','Оценщик не уверен','Несогласованные выводы','Не определено (прежний формат)','циклах 1–32','цикле 64','0,01 × R / H','0,0005 × R','0,0015 × R','Штраф за команду с недопустимым вводом: 2','Выплаченный бонус: 1','Использованные бюджеты','Множитель эпизода','Отношения между кнопками'])assert(html.includes(text),text);
  assert(!html.includes('Бонус и штраф учитываются один раз за класс'));
  assert(!html.includes('Первое применение недопустимого класса: 0'));
  assert(html.includes('&lt;script&gt;')); assert(!html.includes('<script>'));
});

test('reward explanations require saved composition identity and preserve historical rules',()=>{
  const {ui}=harness();const d=snapshot();d.current.method='ppo';ui.setData(d);
  assert(ui.rlDetails(d.current).includes('правила начисления не подтверждены'));
  d.current.rl_details={reward_composition:{version:'carryrl.reward.outcome_potential_text.v1'}};
  const historical=ui.rlDetails(d.current);
  assert(historical.includes('Бонус и штраф учитываются один раз за класс'));
  assert(!historical.includes('0,01 × R / H'));
  d.current.rl_details={episodes:[{reward_composition:{version:'carryrl.reward.outcome_potential_text.v2'}}]};
  assert(ui.rlDetails(d.current).includes('0,01 × R / H'));
});

test('factorized reward phases show distinct labels and only supplied progress',()=>{
  const {ui}=harness();const d=snapshot();d.current=null;
  d.cycle={status:'running',phase_label:'Подготовка и проверка наград',policy_created:false,rewards:[
    {id:'text_fit_factorized_v2_micro',status:'completed',confirmed:true},
    {id:'text_fit_factorized_v2_corpus_r2_micro',status:'completed',confirmed:true},
    {id:'text_fit_factorized_v2_corpus_r2_probe',status:'running',progress:{stage:'epoch_complete',status:'running',epoch:3,total_epochs:8,optimizer_updates:42,train_examples:120,loss:.25}},
    {id:'text_fit_factorized_v2_corpus_r2_full',status:'pending'},
    {id:'text_fit_factorized_v2_full',status:'pending'},
    {id:'text_fit_legacy_fix',status:'failed',progress:{stage:'<script>'}},
  ]};ui.setData(d);const html=ui.training();
  for(const label of ['Короткая TRAIN-проверка','Короткая TRAIN-проверка · R2','Проба на полном корпусе · 8 эпох · R2','Полное обучение текстового оценщика · R2','Эпоха завершена','Эпоха: 3 / 8','Обновлений: 42','TRAIN-примеров: 120','Ошибка обучения (loss)','Обучение текстового оценщика · повтор'])assert(html.includes(label),label);
  assert(html.includes('&lt;script&gt;'));assert(!html.includes('<script>'));
  assert(!html.includes('Эпоха: 0'));assert(!html.includes('NaN'));
  const micro=html.slice(html.indexOf('Короткая TRAIN-проверка'),html.indexOf('Короткая TRAIN-проверка · R2'));
  assert(!micro.includes('reward-progress'));
});


test('R3 reward phases keep corpus identity and show supplied pytest stage',()=>{
  const {ui}=harness();const d=snapshot();d.current=null;
  d.cycle={status:'prepared',phase_label:'Подготовка и проверка наград',policy_created:false,rewards:[
    {id:'text_fit_factorized_v2_corpus_r3_checks',status:'running',progress:{stage:'pytest',status:'running'}},
    ...['micro','probe','full'].map(kind=>({id:`text_fit_factorized_v2_corpus_r3_${kind}`,status:'pending'})),
  ]};ui.setData(d);const html=ui.training();
  for(const label of ['Проверка исходников перед обучением · R3','Короткая TRAIN-проверка · R3','Проба на полном корпусе · 8 эпох · R3','Полное обучение текстового оценщика · R3','Этап: Проверка тестов'])assert(html.includes(label),label);
  assert(!html.includes('Эпоха:'));assert(!html.includes('Обучение текстового оценщика · повтор'));
  assert(!html.includes('Этап: pytest · 100%'));
});


test('positive numeric corpus revisions preserve phase labels and historical names',()=>{
  const {ui}=harness();const d=snapshot();d.current=null;
  const phases={micro:'Короткая TRAIN-проверка',probe:'Проба на полном корпусе · 8 эпох',full:'Полное обучение текстового оценщика',checks:'Проверка исходников перед обучением'};
  for(const revision of [1,2,3,4,12]){
    d.cycle={status:'prepared',rewards:Object.keys(phases).map(kind=>({id:`text_fit_factorized_v2_corpus_r${revision}_${kind}`,status:'pending'}))};
    ui.setData(d);const html=ui.training();
    for(const name of Object.values(phases))assert(html.includes(`${name} · R${revision}`));
    assert(!html.includes('Обучение текстового оценщика · повтор'));
  }
  d.cycle={status:'prepared',rewards:[{id:'text_fit_factorized_v2_micro',status:'pending'},{id:'text_fit_legacy_fix',status:'pending'}]};
  ui.setData(d);const historical=ui.training();
  assert(historical.includes('<span>Короткая TRAIN-проверка</span>'));
  assert(historical.includes('<span>Обучение текстового оценщика · повтор</span>'));
});

test('cycle current stage leads collapsed history and distinguishes stored failure evidence',()=>{
  const {ui}=harness();const d=snapshot();d.current=null;
  d.cycle={status:'running',running:true,stage_id:'text_fit_factorized_v2_corpus_r6_probe',rewards:[
    {id:'text_fit_factorized_v2_corpus_r2_probe',status:'failed',actual_exit_code:0,progress:{stage:'complete',loss:.001}},
    {id:'text_fit_factorized_v2_corpus_r3_probe',status:'failed',actual_exit_code:0,quality_check_passed:false},
    {id:'text_fit_factorized_v2_corpus_r4_checks',status:'failed',actual_exit_code:2,quality_check_passed:false},
    {id:'text_fit_factorized_v2_corpus_r6_probe',status:'running',progress:{stage:'features',epoch:2,total_epochs:8,loss:.01}},
  ]};ui.setData(d);let html=ui.training();
  const history=html.indexOf('<details id="cycle-reward-history">');
  assert(history>0);assert(html.includes('<summary>Предыдущие этапы (3)</summary>'));
  assert(html.indexOf('Проба на полном корпусе · 8 эпох · R6')<history);
  assert(html.includes('<h3>Текущий этап</h3>'));
  assert(html.includes('Процесс завершён; результат не принят'));
  assert(html.includes('Проверка качества не пройдена'));
  assert(html.includes('Ошибка · код 2'));
  assert(html.includes('Этап: Вычисления завершены'));assert(!html.includes('Этап: complete'));
  assert(html.includes('<details id="reward-metrics-text_fit_factorized_v2_corpus_r6_probe"'));
  assert(!html.includes('Loss:'));assert(html.includes('Ошибка обучения сама по себе не подтверждает качество'));
  d.cycle.running=false;d.cycle.status='stopped';d.cycle.rewards[3].status='failed';
  d.cycle.rewards[3].progress={stage:'unknown',status:'unknown'};
  html=ui.training();assert(html.includes('<h3>Последний этап</h3>'));
  assert(!html.includes('<h3>Текущий этап</h3>'));
  assert(html.includes('Этап не подтверждён'));assert(html.includes('Состояние не подтверждено'));
});

test('cycle history disclosure survives incremental refresh and remains closed when collapsed',()=>{
  const {ui}=harness();
  const history=new Element('DETAILS',{id:'cycle-reward-history'},[new Element('SUMMARY')]);history.open=true;
  const host=new Element('DIV',{},[history]);
  ui.patchChildren(host,new Element('DIV',{},[new Element('DETAILS',{id:'cycle-reward-history'},[new Element('SUMMARY'),new Element('UL')])]));
  assert.equal(host.childNodes[0],history);assert.equal(history.open,true);
  history.open=false;
  ui.patchChildren(host,new Element('DIV',{},[new Element('DETAILS',{id:'cycle-reward-history'},[new Element('SUMMARY')])]));
  assert.equal(host.childNodes[0],history);assert.equal(history.open,false);
});


test('text evaluator progress translates observed fitting and evaluation phases',()=>{
  const {ui}=harness();const d=snapshot();d.current=null;
  const phases={fit:'Обучение оценщика',public_runtime:'Проверка 32 игровых инструкций',calibration_gate:'Проверка на калибровочном наборе',sealed_test:'Проверка на закрытом тестовом наборе'};
  for(const [stage,label] of Object.entries(phases)){
    d.cycle={status:'running',running:true,stage_id:'text_fit_factorized_v2_corpus_r6_probe',rewards:[{id:'text_fit_factorized_v2_corpus_r6_probe',status:'running',progress:{stage}}]};
    ui.setData(d);const html=ui.training();
    assert(html.includes(`Этап: ${label}`));assert(!html.includes(`Этап: ${stage}`));
  }
});
