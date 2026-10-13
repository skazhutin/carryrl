"""Local human player for the same atomic environments used by the learner.

No trajectories are saved or supplied to training. Simulation advances only on
an explicit, active browser request; there is no background game loop.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from PIL import Image

from carryrl.agent.actions import CODEC_VERSION, MOUSE_BINS, PhysicalAction
from carryrl.envs.atomic import TICK_SECONDS, AtomicEnv, atomic_level_banks
from carryrl.envs.atomic_suite import TASKS
from carryrl.envs.presentation import PresentationConfig

MAX_BODY_BYTES = 8192
SIZES = (64, 96, 128)
LIMITS = (80, 300, 600)

TASK_GUIDE = {
    "navigation": ("Лабиринт", "Доберитесь своим маркером до зелёной клетки, обходя стены.",
                   "WASD или стрелки. Каждые 0,1 с удержания — один шаг по клеткам."),
    "reaction": ("Реакция", "Подставляйте платформу под падающий мяч. Промах завершает игру.",
                 "A / D или ← / →. Нужно несколько раз поймать мяч."),
    "mouse": ("Наведение и клик", "Совместите игровой крестик с круглой целью и нажмите на неё.",
              "Двигайте мышь внутри экрана; левая кнопка — клик. Между кликами отпускайте её."),
    "memory": ("Память", "Запомните, какая дверь подсвечена в начале. После Wait выберите её.",
               "A / D или ← / →, когда появится Choose. До этого движение заблокировано."),
    "timing": ("Тайминг", "Пролетите в просветах движущихся препятствий, не касаясь границ.",
               "Короткие нажатия пробела подбрасывают персонажа. Удержание не повторяет взмах."),
    "platforming": ("Платформер", "Перепрыгните препятствия и доберитесь до флажка справа.",
                    "A / D или ← / → — движение. Пробел — прыжок с земли; отпускайте его."),
    "steering": ("Руль и инерция", "Удерживайте машинку на извилистой дороге до конца участка.",
                 "A / D или ← / →. Движение вперёд автоматическое, руль имеет инерцию."),
    "camera": ("Движение и камера", "Подойдите к двери и выровняйте камеру: дверь по центру.",
               "WASD — движение. Мышь — поворот и наклон камеры. Крестик показывает выравнивание."),
    "interaction": ("Подобрать и использовать", "Подберите предмет и используйте его у площадки.",
                    "WASD / стрелки — движение. E рядом с предметом, затем E рядом с площадкой."),
    "causality": ("Кнопка и дверь", "Нажмите удалённую кнопку, затем пройдите через дверь к выходу.",
                  "WASD / стрелки. E рядом с кнопкой открывает проход. Красная клетка опасна."),
    "key_door": ("Ключ и дверь", "Найдите ключ, вернитесь к двери, откройте её и дойдите до выхода.",
                 "WASD / стрелки. E рядом с ключом и отдельное нажатие E рядом с дверью."),
    "goal_choice": ("Награда и риск", "Выберите цель: безопасную или более ценную за охранником.",
                    "WASD / стрелки. Белые точки у целей показывают награду; избегайте охранника."),
    "hazards": ("Ловушки и враг", "Доберитесь до выхода, избегая ловушек и патрулирующего врага.",
                "WASD / стрелки. Красные клетки и движущийся враг завершают игру при контакте."),
    "shooting": ("Движение и стрельба", "Подойдите к врагу на расстояние выстрела и попадите в него.",
                 "WASD — движение. Мышь — прицел, левая кнопка — выстрел. Промахи ограничены."),
    "ui": ("Инвентарь", "Выберите предмет того же цвета, что образец над меню, и закройте меню.",
           "Tab — открыть. Стрелки — выбор. Enter — подтвердить. Escape — закрыть после выбора."),
    "chain": ("Короткая цепочка", "Ключ → переключатель → дверь → выход. Не попадите в опасность.",
              "WASD / стрелки. Отдельное нажатие E у ключа, переключателя и двери."),
}


class PlayerConflict(ValueError):
    """A stale or inactive browser must not advance simulation."""


class ManualSession:
    """One in-memory human session, serialized across HTTP requests."""

    def __init__(self):
        self.lock = threading.RLock()
        self.env: AtomicEnv | None = None
        self.active = False
        self.done = False
        self.generation = 0
        self.seed = secrets.randbelow(2**31)
        self.settings = {
            "task": TASKS[0], "difficulty": 0.15, "size": 96,
            "presentation": "canonical", "max_steps": 300,
        }
        self.reset(self.settings)

    @staticmethod
    def validate_settings(record: dict) -> dict:
        expected = {"task", "difficulty", "size", "presentation", "max_steps"}
        if not isinstance(record, dict) or set(record) != expected:
            raise ValueError("Ожидаются настройки игры, сложности, размера, оформления и времени")
        if record["task"] not in TASKS:
            raise ValueError("Неизвестная игра")
        difficulty = record["difficulty"]
        if (isinstance(difficulty, bool) or not isinstance(difficulty, (int, float))
                or not math.isfinite(difficulty) or not 0 <= difficulty <= 1):
            raise ValueError("Сложность должна быть числом от 0 до 1")
        for name, choices in (("size", SIZES), ("max_steps", LIMITS)):
            if isinstance(record[name], bool) or not isinstance(record[name], int):
                raise ValueError(f"Некорректное поле {name}")
            if record[name] not in choices:
                raise ValueError(f"Неподдерживаемое поле {name}")
        if record["presentation"] not in {"canonical", "varied"}:
            raise ValueError("Неизвестное оформление")
        return dict(record)

    def reset(self, settings: dict, *, new_level: bool = False) -> dict:
        settings = self.validate_settings(settings)
        with self.lock:
            if new_level:
                self.seed = secrets.randbelow(2**31)
            visual = None
            if settings["presentation"] == "varied":
                visual = PresentationConfig(mode="varied", native_sizes=(settings["size"],))
            env = AtomicEnv(
                settings["task"], image_size=settings["size"], max_steps=settings["max_steps"],
                difficulty=settings["difficulty"], visual_config=visual,
            )
            options = None
            if self.env is not None and self.env.task == settings["task"]:
                level_index = self.info["level_index"]
                if new_level:
                    count = len(atomic_level_banks(settings["task"])["train"])
                    level_index = (level_index + 1 + secrets.randbelow(count - 1)) % count
                options = {"level_index": level_index}
            _, info = env.reset(seed=self.seed, options=options)
            if self.env is not None:
                self.env.close()
            self.env, self.info, self.settings = env, info, settings
            self.active, self.done = False, False
            self.generation += 1
            return self.state()

    def _check_generation(self, generation: int):
        if (isinstance(generation, bool) or not isinstance(generation, int)
                or generation != self.generation):
            raise PlayerConflict("Игра изменилась. Обновите её состояние")

    def control(self, generation: int, active: bool) -> dict:
        if not isinstance(active, bool):
            raise ValueError("active должно быть логическим значением")
        with self.lock:
            self._check_generation(generation)
            if active and self.done:
                raise PlayerConflict("Эпизод завершён. Нажмите «Заново»")
            self.active = active
            self.generation += 1
            return self.state()

    def step(self, generation: int, record: dict) -> dict:
        action = PhysicalAction.from_dict(record)
        with self.lock:
            self._check_generation(generation)
            if not self.active or self.done:
                raise PlayerConflict("Игра на паузе или завершена")
            _, _, terminated, truncated, self.info = self.env.step(action)
            self.done = bool(terminated or truncated)
            if self.done:
                self.active = False
            return self.state()

    def state(self) -> dict:
        with self.lock:
            buffer = io.BytesIO()
            frame = self.env.render()
            Image.fromarray(frame).save(buffer, format="PNG")
            outcome = ("success" if self.info["success"] else
                       "failure" if self.info["failure"] else
                       "timeout" if self.done else "play")
            return {
                "generation": self.generation, "active": self.active, "done": self.done,
                "settings": self.settings.copy(), "outcome": outcome,
                "score": int(self.info["score"]), "ticks": self.info["elapsed_ticks"],
                "seconds_left": round((self.env.max_steps - self.info["elapsed_ticks"])
                                      * TICK_SECONDS, 1),
                "width": frame.shape[1], "height": frame.shape[0],
                "frame": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
            }


def _guide_records():
    return [{"id": task, "title": TASK_GUIDE[task][0], "goal": TASK_GUIDE[task][1],
             "controls": TASK_GUIDE[task][2]} for task in TASKS]


# Inline assets keep the local player usable offline, without a frontend toolchain.
# ruff: noqa: E501
HTML = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>carryrl · Играть самому</title>
<style>
:root{color-scheme:dark;--bg:#192230;--panel:#222e40;--ink:#eef3fa;--muted:#b5c3d4;--line:#42526a;--blue:#67b9f1;--amber:#efbf6a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 Avenir,"Helvetica Neue",Arial,sans-serif}
main{max-width:1200px;margin:auto;padding:28px 24px 36px}header{display:flex;align-items:baseline;gap:18px;flex-wrap:wrap;margin-bottom:22px}
h1{font-size:28px;line-height:1.2;font-weight:650;letter-spacing:-.5px;margin:0}header p{margin:0;color:var(--muted);font-size:14px}
.workbench{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:28px;align-items:start}.stage{min-width:0}
.screen{background:#101722;min-height:360px;display:flex;align-items:center;justify-content:center;padding:24px;border:1px solid var(--line);position:relative}
canvas{display:block;image-rendering:pixelated;max-width:100%;outline:0;cursor:crosshair;background:#101722;touch-action:none}
canvas:focus-visible{outline:3px solid var(--blue);outline-offset:6px}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:16px 0}
button,select,input{font:inherit}button,select{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:5px;min-height:42px}
button{padding:8px 15px;cursor:pointer}button:hover{border-color:var(--blue)}button:disabled{opacity:.5;cursor:default}
button.primary{background:var(--blue);border-color:var(--blue);color:#13243a;font-weight:650}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid var(--amber);outline-offset:3px}
.status{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;font-size:14px;color:var(--muted);margin-bottom:12px}.status strong{color:var(--ink);font-weight:600}
.result{color:var(--amber);min-height:25px}.legend{font-size:14px;color:var(--muted);margin:10px 0}.legend strong{color:var(--ink)}
aside{border-left:1px solid var(--line);padding-left:24px}label{display:block;font-size:14px;color:var(--muted);margin:0 0 6px}select{width:100%;padding:7px 10px;margin-bottom:18px}
input[type=range]{width:100%;accent-color:var(--blue);margin:0 0 3px}.range-value{font-size:13px;color:var(--muted);display:flex;justify-content:space-between;margin-bottom:18px}
.settings-pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}h2{font-size:18px;font-weight:650;margin:8px 0}p{margin:0 0 14px}
.instructions{border-top:1px solid var(--line);padding-top:16px;margin-top:2px;font-size:15px}.controls{color:var(--blue)}.note{font-size:13px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px;margin-top:18px}
.error{padding:12px;border:1px solid #dc7c76;color:#ffbab4;margin-bottom:14px}.error[hidden]{display:none}
@media(max-width:760px){main{padding:20px 16px}.workbench{grid-template-columns:1fr;gap:20px}.screen{min-height:280px;padding:16px}aside{border-left:0;border-top:1px solid var(--line);padding:20px 0 0}h1{font-size:25px}.toolbar button{flex:1}header{gap:8px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style></head><body><main>
<header><h1>carryrl / Играть самому</h1><p>16 мини-игр, в которых учится агент</p></header>
<div id="error" class="error" role="alert" hidden></div>
<div class="workbench"><section class="stage" aria-label="Игровой экран">
<div class="status"><strong id="game-title">Лабиринт</strong><span id="readout">Загрузка игры…</span></div>
<div class="screen"><canvas id="game" tabindex="0" aria-label="Игра. Нажмите Начать, затем используйте клавиатуру и мышь."></canvas></div>
<div class="toolbar"><button id="toggle" class="primary" disabled>Начать</button><button id="restart" disabled>Заново</button><button id="new-level" disabled>Новый уровень</button></div>
<div id="result" class="result" role="status" aria-live="polite">Нажмите «Начать», чтобы управлять игрой.</div>
<p class="legend">Управление работает, пока <strong>игровой экран в фокусе</strong>. Кликните по нему после возврата. Переход к настройкам ставит игру на паузу.</p>
<p class="legend">Игровой прицел двигается <strong>относительно движений мыши</strong> и может не совпадать с системным курсором. Для нового движения верните мышь внутрь экрана.</p>
</section><aside aria-label="Выбор игры и настройки">
<label for="task">Игра</label><select id="task"></select>
<label for="difficulty">Сложность</label><input id="difficulty" type="range" min="0" max="1" step="0.05" value="0.15">
<div class="range-value"><span id="difficulty-value">0,15</span><span>1,00 — полная сложность</span></div>
<div class="settings-pair"><div><label for="size">Размер игры</label><select id="size"><option value="64">64 px</option><option value="96" selected>96 px</option><option value="128">128 px</option></select></div>
<div><label for="duration">Лимит времени</label><select id="duration"><option value="80">8 секунд</option><option value="300" selected>30 секунд</option><option value="600">60 секунд</option></select></div></div>
<label for="presentation">Оформление</label><select id="presentation"><option value="canonical">Исходное</option><option value="varied">Темы, текстуры и разный UI</option></select>
<div class="instructions"><h2>Цель</h2><p id="goal"></p><h2>Управление</h2><p id="controls" class="controls"></p></div>
<p class="note">Экран увеличен для человека без сглаживания. Симуляция делает 10 шагов в секунду. Ваши действия не записываются и не используются для обучения агента.</p>
</aside></div></main>
<script>
'use strict';
const GUIDES=__GUIDES__, MOUSE_BINS=__MOUSE_BINS__, CODEC=__CODEC__, TOKEN=__TOKEN__;
const $=id=>document.getElementById(id), canvas=$('game'), ctx=canvas.getContext('2d');
const heldCodes=new Set(), keyTaps=new Set(), heldButtons=new Set(), buttonTaps=new Set();
let state=null, active=false, epoch=0, timer=null, dx=0, dy=0, wheel=0, lastPointer=null;
let releaseFirst=false, pending=false;
const codeMap={ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right',
Space:'space',Tab:'tab',Enter:'enter',Escape:'escape',ShiftLeft:'shift',ShiftRight:'shift',
ControlLeft:'ctrl',ControlRight:'ctrl'};
for(const letter of 'wasdefqr')codeMap['Key'+letter.toUpperCase()]=letter;
for(let i=1;i<=9;i++)codeMap['Digit'+i]=String(i);
const buttonMap={0:'left',1:'middle',2:'right'};
for(const guide of GUIDES){const option=document.createElement('option');option.value=guide.id;option.textContent=guide.title;$('task').append(option)}
function settings(){return {task:$('task').value,difficulty:Number($('difficulty').value),
size:Number($('size').value),presentation:$('presentation').value,max_steps:Number($('duration').value)}}
function instructions(){const guide=GUIDES.find(g=>g.id===$('task').value);$('game-title').textContent=guide.title;$('goal').textContent=guide.goal;$('controls').textContent=guide.controls;
$('difficulty-value').textContent=Number($('difficulty').value).toFixed(2).replace('.',',')}
function clearInput(){heldCodes.clear();keyTaps.clear();heldButtons.clear();buttonTaps.clear();dx=dy=wheel=0;lastPointer=null}
function stop(){active=false;epoch++;clearTimeout(timer);timer=null;clearInput()}
function fail(error){stop();$('error').hidden=false;$('error').textContent='Игра остановлена: '+error.message+'. Нажмите «Заново», чтобы повторить.';$('toggle').disabled=true}
async function request(path,body){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Carryrl-Token':TOKEN},body:JSON.stringify(body)});
const result=await response.json();if(!response.ok)throw new Error(result.error||'Ошибка сервера');return result}
async function draw(next,expectedEpoch){if(expectedEpoch!==epoch)return;state=next;const image=new Image();await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=reject;image.src=next.frame});
if(expectedEpoch!==epoch)return;canvas.width=next.width;canvas.height=next.height;ctx.imageSmoothingEnabled=false;ctx.drawImage(image,0,0);
const available=canvas.parentElement.clientWidth-(innerWidth<=760?32:48);const scale=Math.min(5,available/next.width,540/next.height);
canvas.style.width=Math.max(1,Math.floor(next.width*scale))+'px';canvas.style.height=Math.max(1,Math.floor(next.height*scale))+'px';
$('readout').textContent='Счёт '+next.score+' · осталось '+next.seconds_left.toFixed(1)+' с · экран '+next.width+'×'+next.height;
$('toggle').disabled=next.done;$('restart').disabled=$('new-level').disabled=false;
$('toggle').textContent=next.active?'Пауза':'Начать';
const results={success:'Победа! Можно повторить или выбрать новый уровень.',failure:'Поражение. Попробуйте ещё раз.',timeout:'Время закончилось. Попробуйте ещё раз.'};
$('result').textContent=results[next.outcome]||(next.active?'Игра идёт.':'Пауза. Нажмите «Начать».');
if(next.done){stop();$('toggle').disabled=true}}
function quantize(value){return MOUSE_BINS.reduce((best,bin)=>Math.abs(bin-value)<Math.abs(best-value)?bin:best,0)}
function action(){if(releaseFirst){releaseFirst=false;return {codec_version:CODEC,keys:[],mouse_buttons:[],dx:0,dy:0,wheel:0}}
const keys=[...new Set([...heldCodes].map(code=>codeMap[code]).concat([...keyTaps]))];keyTaps.clear();
const buttons=[...new Set([...heldButtons,...buttonTaps])];buttonTaps.clear();const mx=quantize(dx),my=quantize(dy);dx-=mx;dy-=my;
const scroll=Math.sign(wheel);wheel=0;return {codec_version:CODEC,keys,mouse_buttons:buttons,dx:mx,dy:my,wheel:scroll}}
async function tick(expectedEpoch){if(!active||expectedEpoch!==epoch)return;const started=performance.now();
try{const next=await request('/api/step',{generation:state.generation,action:action()});await draw(next,expectedEpoch);
if(active&&expectedEpoch===epoch)timer=setTimeout(()=>tick(expectedEpoch),Math.max(0,100-(performance.now()-started)))}catch(error){if(expectedEpoch===epoch)fail(error)}}
async function pause(){if(!active||pending)return;pending=true;stop();const current=epoch;try{await draw(await request('/api/control',{generation:state.generation,active:false}),current)}catch(error){if(current===epoch)fail(error)}finally{if(current===epoch)pending=false}}
async function reset(newLevel=false){stop();pending=true;const current=epoch;$('error').hidden=true;instructions();$('toggle').disabled=true;
try{await draw(await request('/api/reset',{settings:settings(),new_level:newLevel}),current)}catch(error){if(current===epoch)fail(error)}finally{if(current===epoch)pending=false}}
$('toggle').addEventListener('click',async()=>{if(pending||!state)return;if(active){await pause();return}pending=true;stop();const current=epoch;
try{await draw(await request('/api/control',{generation:state.generation,active:true}),current);if(current!==epoch||state.done)return;
active=true;releaseFirst=state.ticks>0;canvas.focus();timer=setTimeout(()=>tick(current),100)}catch(error){if(current===epoch)fail(error)}finally{if(current===epoch)pending=false}});
$('restart').addEventListener('click',()=>reset(false));$('new-level').addEventListener('click',()=>reset(true));
for(const id of ['task','difficulty','size','duration','presentation'])$(id).addEventListener('change',()=>reset(false));
$('difficulty').addEventListener('input',instructions);
function focused(){return active&&document.activeElement===canvas}
window.addEventListener('keydown',event=>{const key=codeMap[event.code];if(!key||!focused())return;event.preventDefault();if(!event.repeat)keyTaps.add(key);heldCodes.add(event.code)});
window.addEventListener('keyup',event=>{if(heldCodes.has(event.code)){event.preventDefault();heldCodes.delete(event.code)}});
canvas.addEventListener('blur',()=>pause());window.addEventListener('blur',()=>pause());
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause()});
canvas.addEventListener('pointerenter',event=>{lastPointer={x:event.clientX,y:event.clientY}});
canvas.addEventListener('pointerleave',()=>{lastPointer=null});
canvas.addEventListener('pointermove',event=>{if(!focused()){lastPointer={x:event.clientX,y:event.clientY};return}
if(lastPointer){const box=canvas.getBoundingClientRect();dx+=(event.clientX-lastPointer.x)*canvas.width/box.width;dy+=(event.clientY-lastPointer.y)*canvas.height/box.height}
lastPointer={x:event.clientX,y:event.clientY}});
canvas.addEventListener('pointerdown',event=>{canvas.focus();if(!active)return;event.preventDefault();const button=buttonMap[event.button];if(button){heldButtons.add(button);buttonTaps.add(button)}});
window.addEventListener('pointerup',event=>{const button=buttonMap[event.button];if(button)heldButtons.delete(button)});
canvas.addEventListener('contextmenu',event=>{if(active)event.preventDefault()});
canvas.addEventListener('wheel',event=>{if(!focused())return;event.preventDefault();wheel+=Math.sign(event.deltaY)},{passive:false});
window.addEventListener('resize',()=>{if(state)draw(state,epoch)});
instructions();reset(false);
</script></body></html>"""


def make_handler(session: ManualSession, token: str):
    html = (HTML.replace("__GUIDES__", json.dumps(_guide_records(), ensure_ascii=False))
            .replace("__MOUSE_BINS__", json.dumps(MOUSE_BINS))
            .replace("__CODEC__", json.dumps(CODEC_VERSION))
            .replace("__TOKEN__", json.dumps(token))).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Human actions and trajectories are deliberately not logged.
            pass

        def _reply(self, status: HTTPStatus, payload: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; "
                             "script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                             "img-src data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: HTTPStatus, record: dict):
            self._reply(status, json.dumps(record, ensure_ascii=False).encode(),
                        "application/json; charset=utf-8")

        def _local_host(self):
            host = self.headers.get("Host", "")
            port = self.server.server_port
            return host in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def do_GET(self):
            if not self._local_host():
                self._json(HTTPStatus.FORBIDDEN, {"error": "Только localhost"})
            elif self.path == "/":
                self._reply(HTTPStatus.OK, html, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(HTTPStatus.OK, session.state())
            elif self.path == "/favicon.ico":
                self._reply(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Страница не найдена"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            if (not self._local_host() or self.headers.get("X-Carryrl-Token") != token
                    or (origin is not None and origin != f"http://{host}")):
                self._json(HTTPStatus.FORBIDDEN, {"error": "Запрос разрешён только из игрока"})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Нужен JSON"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Размер запроса"})
                    return
                record = json.loads(self.rfile.read(length))
                if not isinstance(record, dict):
                    raise ValueError("Нужен объект JSON")
                path = urlsplit(self.path).path
                if path == "/api/reset" and set(record) == {"settings", "new_level"}:
                    if not isinstance(record["new_level"], bool):
                        raise ValueError("new_level должно быть логическим значением")
                    result = session.reset(record["settings"], new_level=record["new_level"])
                elif path == "/api/control" and set(record) == {"generation", "active"}:
                    result = session.control(record["generation"], record["active"])
                elif path == "/api/step" and set(record) == {"generation", "action"}:
                    result = session.step(record["generation"], record["action"])
                else:
                    raise ValueError("Неизвестный запрос или неверные поля")
            except PlayerConflict as error:
                self._json(HTTPStatus.CONFLICT, {"error": str(error)})
            except (ValueError, TypeError, UnicodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            else:
                self._json(HTTPStatus.OK, result)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Локальный ручной игрок carryrl")
    parser.add_argument("--port", type=int, default=6007)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port должен быть в 1..65535")
    session = ManualSession()
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                make_handler(session, secrets.token_urlsafe(32)))
    print(f"Играть: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
