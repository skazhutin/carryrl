const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function harness() {
  const listeners = {};
  const elements = new Map();
  const get = id => elements.get(id) || (elements.set(id, {id, hostname: '', href: '/archive', addEventListener(type, listener) { listeners[`${id}:${type}`] = listener; } }), elements.get(id));
  const document = {getElementById: get, activeElement: null};
  const context = {document, Intl, Date, Map, Set, Number, String, Math, Array, JSON, encodeURIComponent, decodeURIComponent, location: {hash: '#/overview', hostname: 'localhost'}, window: {addEventListener() {}}, Image: class {}};
  const source = fs.readFileSync(path.join(__dirname, '../../src/carryrl/dashboard/assets/overview.js'), 'utf8');
  const exposed = `globalThis.ui = {playback, quality, perfectScore, experimentSummary, experimentSelector, pickerLabel, verdictSummary, trainingStatus, history, diagnostics, technicalMessages, patchChildren, setData(value) {data=value;}, setPage(value) {recordingPage=value; selectedRecording='';}, setRecording(value) {selectedRecording=value;}, state() {return {selectedRecording,recordingPage,recordingTask};}};`;
  vm.runInNewContext(source.replace('refresh();setInterval(refresh,5000);requestAnimationFrame(animatePlayer);', exposed), context);
  return {ui: context.ui, document, listeners};
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

test('model selectors use short result labels and preserve the full chosen label outside options',()=>{
  const {ui}=harness();const data=snapshot();
  data.current.experiment={title:'MLX 100 epochs 400000 examples',picker_label:'13.10 13:39 · 143/256 побед · хуже исходной'};
  data.experiments=[data.current];ui.setData(data);
  const html=ui.experimentSelector();
  assert(!html.includes('MLX'));assert(!html.includes('400000'));
  assert(html.includes('Выбранная модель:'));
  assert((html.match(/143\/256/g) || []).length>=2);
  const video=ui.playback();
  assert(!video.includes('MLX'));
});
test('a verified worse verdict gives the exact scoped comparison and baseline selection action',()=>{
  const {ui}=harness();
  const html=ui.verdictSummary({verdict:{state:'worse',successes:143,baseline_successes:155,episodes:256},recommended_run_id:'baseline',recommendation:'Для этого набора оставьте исходную модель.'});
  assert(html.includes('Новая модель хуже: 143/256'));
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
