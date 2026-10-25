# Эксперименты carryrl

Реестр составлен 20 октября 2026 года по сохранённым артефактам `runs/`,
локальным аудитам `.cache/`, конфигурациям и истории в [SPEC.md](SPEC.md) и
[DECISIONS.md](DECISIONS.md). Здесь собраны как завершённые опыты, так и
подготовленные, прерванные и заменённые попытки, включая прежние модели.

Это история фактических проверок. Цели и дальнейшая программа остаются в
[SPEC.md](SPEC.md), основания выбора — в [DECISIONS.md](DECISIONS.md).
Наличие опыта в этом реестре не означает, что его результат принят или что
его разрешено автоматически продолжать.

В каждом содержательном опыте нужно различать **смысл проверки**, **условия**,
**измеренные метрики**, **фактический итог** и **вывод с ограничениями**.
Для ранних запусков эти пояснения стоят рядом с именем run; для пар и серий —
перед общей таблицей и после неё. Повторные технические подготовки сгруппированы.
Если заранее записанная гипотеза не сохранилась, описание смысла объясняет
фактически выполненную проверку и не выдаётся за исходный план.

## Навигация

- [Краткий итог](#summary)
- [Модели и правила чтения результатов](#models)
- [Ранние пробы и прежняя модель: 7–15 октября](#historical)
- [Диагностика, награда, корпуса и подготовка: 15 октября](#diagnostics)
- [Продолжение прежней модели: ordinary-серии 15–18 октября](#ordinary)
- [Новый старт RL: 18–20 октября](#fresh)
- [Новая общая RL-модель на 16 играх: 20 октября](#fresh-all16)
- [Подготовка улучшений после fresh-all16: 20 октября](#rl-improvements)
- [Новый ограниченный цикл всех16: 21 октября](#fresh-all16-v2)
- [Новый цикл всех16 и проверка V13: 23 октября](#fresh-all16-v3)
- [Раздельный текстовый оценщик и награда v2: 23 октября](#factorized-text-v2)
- [Исправление привязки текстового оценщика](#text-reward-binding-fix)
- [Диагностика отказа допуска и условные задачи](#text-reward-admission-fix)
- [Свежий fit условного текстового оценщика](#text-reward-conditional-fit)
- [Исправление корпуса и короткие пробы переноса](#text-reward-transfer-fix)
- [Совместное распознавание управления и групп](#text-reward-structured)
- [Технические замеры хранения](#storage)
- [Полнота, сохранность и добавление следующих опытов](#maintenance)

<a id="summary"></a>
## Краткий итог

| Этап | Что проверяли | Главный результат | Как читать |
| --- | --- | --- | --- |
| Раздельный текстовый оценщик, 23 октября | Новый согласованный смысл применимости, частичные знания и награда v2 | Реализация и технические проверки; новых обученных весов пока нет | Новый протокол разрешён пользователем; общий срок прежний, PPO заблокирован до допуска награды |
| Новый цикл всех16, 23 октября | V13 и затем локальный контекст v8 | V8: runtime9/32,held136/176,press17/22,aliases56/88; DEVcoverage прошёл,5criticalerrors | Исторический допуск не пройден; PPO не запускался; следующий протокол разметки изменён отдельно |
| Ранние прототипы и запуск agent v1 | Исполнимость обучения, CPU/MPS/MLX, награда по экрану | Обучение, сохранение и оценка заработали | Технический успех не равен освоению игр |
| Прежняя модель, сопоставимая оценка 13 октября | Исходный агент против дообучения на оформленных ошибках | 155/256 → 143/256 побед; 132/256 → 123/256 со счётом 5/5 | Дообученный кандидат не принят; память проверялась отдельно |
| Ordinary-серии прежней модели | Корпус, удержание прежнего поведения, число эпох, optimizer, коррекции | Проверенные кандидаты не прошли условия замены baseline | Падение loss и успех учителя не доказывают улучшения самостоятельной игры |
| Fresh RL, 19–20 октября | Новый старт, число визуальных токенов, штрафы, затраты, PPO на navigation | Доказанного улучшения относительно свежей политики не получено | Прежние собственные веса не являются стартом этой серии |
| Контроль navigation с пятью seed действий | Свежая политика против последних обученных вариантов | Fresh 58/80; LR1e-4 56/80; group8 52/80; λ1 52/80 в каждом оформлении | Это 16 уровней × 5 повторов, а не 80 независимых уровней |
| Новая общая RL-модель, 20 октября | Независимый старт на всех 16 играх, общий minibatch16 и microbatch4 | Завершены128 циклов; stochastic canonical16/256 → 15/256, pixel17/256 → 14/256; greedy2/256 → 2/256 в обоих | Пилотный порог не пройден; общего роста относительно собственного CP0 нет |
| Новый цикл всех16, 21 октября | Новые модели награды и обязательные проверки перед PPO | Остановлен: визуальная награда допущена, обе текстовые модели не прошли gates; PPO не запускался | Новая игровая модель не создана; улучшение в играх не измерено |
| Исправление текстового оценщика, 21 октября | Контекст названных команд, корпус, режимы и алиасы | 40эпох завершены; положительный охват5129/11540, aliases577/3060; runtime3/32, допуск не пройден | Отрицательный знак прошёл; улучшение игр не измерялось, PPO не запускался |
| Исправление отказа допуска, 21 октября | Потери status→mode→alias, релевантные цели и грамматика | Причины каскада измерены; добавлены conditional recipe и v7, новое обучение не запускалось | Проверка кода и данных не подтверждает качество будущей модели |

| Исправление корпуса и короткие пробы, 21 октября | Отношения команд, формулировки, частота TRAIN и диагностика readout | I: DEV aliases330/882, 15ошибок, public1/32 (F3/32); G не перенеслась в H | Допуск F не пройден; G — диагностика, полного fit/PPO нет |
| Совместное распознавание управления, 21 октября | Новая совместная цель status/mode/group и естественные TRAIN-инструкции | Подготовка новой схемы; измеренного качества пока нет | Старые пороги и исходные 32 инструкции сохранены; full требует успешной короткой проверки |

Свежий conditional fit завершён: 40 эпох, выбранная эпоха1; calibration не пройдена, runtime0/32, ready=false. Причина расхождения TRAIN/dev требует дальнейшей диагностики.

Первоисточники и оговорки к числам приведены в карточках соответствующих серий.
Цель 100% успешных прохождений **каждой** из 16 игр не достигнута. Общий средний
результат, накопленная награда и обучающие победы этот критерий не заменяют.

<a id="models"></a>
## Модели и правила чтения результатов

### Какие поколения здесь различаются

| Обозначение | Что это | Связь с последующими опытами |
| --- | --- | --- |
| Ранний прототип | Небольшой агент и технические navigation/CPU/MPS-пробы | Отдельный этап; не смешивать с полным agent v1 |
| Прежний agent v1 | DINOv2-small и E5 как замороженные внешние кодировщики, обучаемая пространственная обработка, Transformer-память и физические команды | Серии 8–18 октября, включая BC, PPO и ordinary |
| Старый baseline | `agent_v1_mlx_visual_bc_control_20261009T035450Z/checkpoints/rollout_0000` | Контроль многих последующих BC-опытов; 155/256 относится к переоценке по `efficient.v1` |
| Новый старт, fresh CP0 | Снова используются внешние DINOv2/E5, но собственные обучаемые части политики, critic, память и optimizer начинают с нуля; награда готовится заново | Серии `token_comparison_fresh_*`, затем опыты со штрафами и navigation; это не продолжение старого baseline |

Архитектурная отправная точка прежнего агента описана в
[конфигурации](configs/agent_v1_generalization.json): width 256, 4 слоя,
4 головы, 16 пространственных токенов, 8 токенов памяти, сегмент 64 решения.
Конкретный запуск мог менять параметры; приоритет имеет его сохранённый config
или manifest. В новой серии сравнивались 16/32/64 пространственных токена.

### Термины и единицы

- **BC / imitation** — обучение повторению размеченных действий; **PPO / RL** —
  обновление по собственному опыту и награде. BC-опыты не называются RL.
- **Frozen-оценка** — веса не обновляются; рабочая память внутри эпизода работает.
  **Checkpoint / CP** — сохранённое состояние модели; CP0 — начальная точка серии.
- **TRAIN** — данные обучения; **validation** — проверка, используемая при выборе
  модели. Результаты validation не выдаются за независимый финальный test.
- **Stochastic** — выбор действий с семплированием; **greedy / deterministic** —
  детерминированный выбор. Это разные режимы, их результаты записываются отдельно.
- **Canonical / pixel** — варианты оформления. Они не образуют новые независимые
  уровни. Старые матрицы, одновременно меняющие размер среды, язык и оформление,
  не являются чистой проверкой только оформления.
- **Победы / 5 из 5** — разные метрики. На полном основном наборе обычно
  16 игр × 16 уровней = 256 эпизодов политики. Эпизоды random учитываются отдельно;
  768 строк отчёта fresh-серии могут означать 256 stochastic + 256 deterministic
  + 256 random, а не 768 проверок обученной политики.
- **Seed** — начальное значение генератора случайности. Повторы действий одного
  checkpoint с разными seed не заменяют независимых обучений.
- **NLL / loss / objective** — ошибка обучения или воспроизведения размеченной
  команды. Меньшее значение само по себе не означает больше побед.
- **KL-anchor** — штраф за отклонение от поведения исходной модели; проверялся как
  способ уменьшить забывание. **AdamW reset** — сброс накопленного состояния
  оптимизатора при сохранении весов политики.
- **Эпоха** — проход по обучающему корпусу; **цикл / коллекция** в PPO — сбор
  собственных эпизодов с последующими обновлениями. Эти единицы не взаимозаменяемы.
- **Train-fit / forced history** — проверка на записанной истории действий;
  **closed loop** — самостоятельное прохождение с последствиями собственных команд.
- **Corrections** — эпизоды с реально исполненными действиями ученика и отдельно
  записанными целевыми командами учителя для исправления поведения.
- **Retention** — сохранение ранее освоенных навыков после нового обучения;
  **регрессия** — ухудшение, например потеря ранее выигранных уровней.
- **Smoke / preflight** — короткая техническая проверка / проверка готовности к
  запуску. **Gate** — условие допуска или принятия результата.
- **HUD / reward** — видимые счёт и статус на экране / обучающая награда.
  Правильное чтение HUD проверяется отдельно от способности агента выигрывать.

### Статусы и доказательства

**Завершён** означает, что сохранён итоговый результат соответствующей проверки;
для обучения дополнительно нужны сохранённые веса/архив и сведения о завершении.
**Подготовлен** означает наличие конфигурации или данных без результата обучения.
**Прерван / частичный** сохраняет достигнутый этап, но не обещанный итог.
**Заменён** обозначает попытку, после которой исправили условия или протокол.

Старый `status.json` иногда остался `running` или `prepared_not_executed` после
завершения следующего этапа. В таких случаях ниже указано, какой итоговый отчёт
подтверждает состояние. Само поле status, имя каталога или наличие launch-файла
не считается доказательством обученного checkpoint. Идентификаторы с `Z` сохранены
в исходном виде (UTC); дата в имени — идентификатор запуска, не обязательно дата
его завершения.

Артефакты `runs/` и `.cache/` локальные и исключены из Git. Ссылки работают в этой
рабочей копии; копирование одного Markdown не переносит веса, данные и отчёты.


<a id="historical"></a>
## Ранние технические проверки и старая модель (7–15 октября)

Здесь разделены техническая работоспособность, обучение политики и frozen-диагностика. Небольшая ранняя CNN (179 590 параметров), последовательная `agent_v1` и последующие варианты текста/MLX — разные конфигурации. Метрики старого seeded sampling, полного банка и `efficient.v1` нельзя объединять в один временной ряд. Во всех дробях ниже числитель — победы, знаменатель — эпизоды данной политики; random оценивается отдельно. Учитель и его успешные демонстрации не являются победами обученной политики.

### Инфраструктура, старый CLI и навигационная диагностика

- [compute-probe-initial.json](runs/compute-probe-initial.json): синтетическая сеть 6 021 параметр, 30 измеренных обновлений после 5 warmup, batch 32×64×64. CPU 91,00 updates/s, MPS 293,23; конечные loss/градиенты/параметры finite, веса изменились. Это производительность синтетического графа, без игры и передачи изображений.
- [smoke_cpu_2048](runs/smoke_cpu_2048/summary.json) и [smoke_mps_2048](runs/smoke_mps_2048/summary.json): старая CNN, по 2 048 шагов и 32 optimizer epochs, завершены. CPU 12,91 s, MPS 41,05 s; итоговые победы CPU 1/20, MPS 3/20, random 11/20 (проверены по `evaluation.jsonl`, запись `validation` при 2 048 шагах). Это smoke, не доказательство превосходства MPS или освоения навигации.
- [cli_resume_smoke](runs/cli_resume_smoke/summary.json): возобновление с 1 024 до 2 048 шагов, 1 024 новых решений, 16 optimizer epochs, завершено за 5,90 s; 3/20 до и после. Проверка resume, не отдельный результат полноценного обучения.
- [cli_random_baseline](runs/cli_random_baseline/evaluation.json), [cli_eval_seed0](runs/cli_eval_seed0/evaluation.json), корневые [random JSON](runs/cli_random_baseline.json) и [seed0 JSON](runs/cli_eval_seed0.json): frozen/CLI-проверки старой навигации; в корневых JSON random 14/30, детерминированная seed0-модель 1/30. Дублирующие экспортные файлы не считаются независимыми экспериментами.
- `build_verification_learning`, `tensorboard_dashboard_build`, `agent_v1_build`, `agent_v1_suite_build`: сохранены сборочные артефакты; это проверки упаковки, не обучающие запуски и не результаты качества.
- [pilot_navigation_cpu](runs/pilot_navigation_cpu/series.json): первый содержательный pilot старой CNN/SB3 PPO на навигации. [Протокол](runs/pilot_navigation_cpu/protocol.json) зарегистрирован до запусков 7 октября: seed 0/1/2, CPU, по 49 152 шагов, learning_rate 3e-4, 30 validation seeds 1000–1029, test не используется; все три запуска завершились без retry. По seed 0/1/2 initial deterministic 0/30, 4/30, 1/30 → final 1/30, 4/30, 2/30; random 14/30. По 768 optimizer epochs, длительность 170,70/167,92/186,68 s; checkpoint `step_00049152` сохранён для каждой модели. Итог 7/90 — описательная сумма повторных оценок трёх моделей, не 90 независимых инициализаций. Убедительного улучшения над random нет; именно эти веса затем использованы в stochastic diagnostic.
- Службы [manual_player_20261008T132223Z_launch](runs/manual_player_20261008T132223Z_launch), [tensorboard_learning_20261008T122806Z_launch](runs/tensorboard_learning_20261008T122806Z_launch), [tensorboard_monitor_20261008T120008Z_launch](runs/tensorboard_monitor_20261008T120008Z_launch): папки запуска ручного player и мониторинга; самостоятельных результатов обучения им не приписываем.
- [stochastic_diagnostic](runs/stochastic_diagnostic/summary.json): post-hoc проверка трёх frozen навигационных моделей после 49 152 training steps, по 30 уровней × 3 action seed. Обученные: 137/270, по model seed 41/90, 52/90, 44/90; random 42/90. [Начальный контроль](runs/stochastic_diagnostic/initial_control_summary.json): 126/270, по seed 42/90, 41/90, 43/90. Средняя парная разница +4,07 п.п.; три инициализации, повторно используемый random и post-hoc постановка не доказывают статистическое превосходство. Хеши checkpoint до/после сохранены в summary.
- [agent_v1_full_graph_probe](runs/agent_v1_full_graph_probe/results.json): DINOv2-small + последовательная политика (27 955 142 всего, 5 898 566 trainable), 65 решений истории; CPU/MPS прошли forward/replay/backward, ненулевой gradient memory write. Полный синтетический probe 1,06/10,54 s; max logprob difference 0/1,91e-6. Это проверка графа, не обучение.
- [.cache/mlx_policy_benchmark.json](.cache/mlx_policy_benchmark.json): неизменная обучаемая политика 5 996 870 параметров, detached frozen features, 32 решения × 256 patches; медиана CPU 0,07490 s, MLX GPU 0,04665 s, ускорение 1,606×. Энкодеры и gameplay исключены.

### Визуальная reward-модель: первая удачная проверка оказалась недостаточной

[reward_screen16_20261008_r1](runs/reward_screen16_20261008_r1/reward_report.json), `screen_reward.v2`: supervised RGB HUD readout, 40 эпох, 30 уникальных training readouts. Score MAE=0 на train 13 020 кадрах/384 эпизодах, familiar validation 3 264/96 и unseen violet 1 088/96; дополнительный canonical probe 744 кадров также без ошибок. Однако независимый vocabulary audit обнаружил **21/54 неверных предсказаний score**: прежний gate не покрывал все сочетания score/status.

[reward_screen16_20261008_r2](runs/reward_screen16_20261008_r2/reward_report.json), `screen_reward.v3`, заменил r1: score MAE=0 на train 13 722 кадров/384 эпизодах, familiar 3 912/96, unseen 1 304/96 и полном canonical vocabulary 54/54. [replacement_audit.json](runs/reward_screen16_20261008_r2/replacement_audit.json) фиксирует причину замены и passed gate. Это обучение чтению известного HUD, не автономное открытие награды и не доказательство игры; test не использовался.

### Agent v1: технические smoke, PPO и переход к тексту

Далее для каждого запуска приведены фактические status/summary и сохранённые оценки. `D` — frozen deterministic, `S` — frozen stochastic, `R` — generic physical random. Наличие checkpoint обозначает найденные файлы/каталог, а не повторную загрузку весов в этой инвентаризации.


#### Технические прогоны


**[agent_v1_cpu_smoke](runs/agent_v1_cpu_smoke)**. Статус артефакта: `completed`.

**Смысл:** короткая техническая проверка сбора опыта, PPO и сохранения модели на CPU.

**Итог и вывод:** цикл завершился с checkpoint, но deterministic-политика не выиграла ни одного оценочного эпизода; работоспособность обучения не означает освоение игры.

Циклов сбора опыта: 2; Новых решений PPO: 389.

- [evaluation.json](runs/agent_v1_cpu_smoke/evaluation.json): D 0/8; R 1/8; difficulty=не записана, старый протокол.

Сохранённых manifest-файлов checkpoint: 3.
Источники: [summary.json](runs/agent_v1_cpu_smoke/summary.json), [status.json](runs/agent_v1_cpu_smoke/status.json), [metadata.json](runs/agent_v1_cpu_smoke/metadata.json).

**[agent_v1_cpu_resume](runs/agent_v1_cpu_resume)**. Статус артефакта: `completed`.

**Смысл:** техническая проверка продолжения CPU-обучения из сохранения.

**Итог и вывод:** продолжение завершилось и оставило checkpoint; нулевое число deterministic-побед не подтверждает улучшения навыка.

Циклов сбора опыта: 2; Новых решений PPO: 138.

- [evaluation.json](runs/agent_v1_cpu_resume/evaluation.json): D 0/8; R 1/8; difficulty=не записана, старый протокол.

Сохранённых manifest-файлов checkpoint: 2.
Источники: [summary.json](runs/agent_v1_cpu_resume/summary.json), [status.json](runs/agent_v1_cpu_resume/status.json), [metadata.json](runs/agent_v1_cpu_resume/metadata.json).

**[agent_v1_cpu_evaluation](runs/agent_v1_cpu_evaluation)**. Статус артефакта: `completed`.

**Смысл:** отдельная frozen-проверка сохранённой CPU-политики без новых обновлений.

**Итог и вывод:** оценка выполнена, побед нет; это проверка процедуры оценки, а не положительный результат обучения.


- [evaluation.json](runs/agent_v1_cpu_evaluation/evaluation.json): D 0/8; R 1/8; difficulty=не записана, старый протокол.

Это отдельная frozen-оценка checkpoint соответствующего CPU smoke; собственного checkpoint в evaluation-папке не требуется.
Источники: [status.json](runs/agent_v1_cpu_evaluation/status.json), [metadata.json](runs/agent_v1_cpu_evaluation/metadata.json).

**[agent_v1_mps_smoke](runs/agent_v1_mps_smoke)**. Статус артефакта: `completed`.

**Смысл:** короткая техническая проверка PPO на MPS с сохранением результата.

**Итог и вывод:** прогон завершился, но deterministic-побед нет; технический запуск на GPU подтверждён, игровой навык — нет.

Циклов сбора опыта: 2; Новых решений PPO: 400.

- [evaluation.json](runs/agent_v1_mps_smoke/evaluation.json): D 0/8; R 1/8; difficulty=не записана, старый протокол.

Сохранённых manifest-файлов checkpoint: 3.
Источники: [summary.json](runs/agent_v1_mps_smoke/summary.json), [status.json](runs/agent_v1_mps_smoke/status.json), [metadata.json](runs/agent_v1_mps_smoke/metadata.json).

**[agent_v1_suite_cpu_smoke](runs/agent_v1_suite_cpu_smoke)**. Статус артефакта: `completed`.

**Смысл:** проверка общего цикла обучения и оценки набора игр на CPU.

**Итог и вывод:** опыт собран и checkpoint сохранён, но на оценочном наборе побед нет; suite проходит технический цикл без доказанного освоения задач.

Циклов сбора опыта: 1; Новых решений PPO: 847.

- [rollout_0001.json](runs/agent_v1_suite_cpu_smoke/evaluations/rollout_0001.json): D 0/16; R 2/16; difficulty=не записана, старый протокол.

Сохранённых manifest-файлов checkpoint: 2.
Источники: [summary.json](runs/agent_v1_suite_cpu_smoke/summary.json), [status.json](runs/agent_v1_suite_cpu_smoke/status.json), [metadata.json](runs/agent_v1_suite_cpu_smoke/metadata.json).

**[agent_v1_suite_mps_smoke](runs/agent_v1_suite_mps_smoke)**. Статус артефакта: `completed`.

**Смысл:** проверка того же общего цикла набора игр на MPS.

**Итог и вывод:** прогон завершён с сохранениями; отсутствие побед ограничивает вывод технической работоспособностью.

Циклов сбора опыта: 1; Новых решений PPO: 691.

- [rollout_0001.json](runs/agent_v1_suite_mps_smoke/evaluations/rollout_0001.json): D 0/16; R 2/16; difficulty=не записана, старый протокол.

Сохранённых manifest-файлов checkpoint: 2.
Источники: [summary.json](runs/agent_v1_suite_mps_smoke/summary.json), [status.json](runs/agent_v1_suite_mps_smoke/status.json), [metadata.json](runs/agent_v1_suite_mps_smoke/metadata.json).

**[agent_v1_suite_cpu_evaluation](runs/agent_v1_suite_cpu_evaluation)**. Статус артефакта: `completed`.

**Смысл:** отдельная frozen-оценка CPU-checkpoint на наборе игр.

**Итог и вывод:** оценка завершена с нулём побед; сохранённая политика ещё не показывает освоения набора.


- [evaluation.json](runs/agent_v1_suite_cpu_evaluation/evaluation.json): D 0/16; R 2/16; difficulty=не записана, старый протокол.

Это отдельная frozen-оценка checkpoint соответствующего CPU smoke; собственного checkpoint в evaluation-папке не требуется.
Источники: [status.json](runs/agent_v1_suite_cpu_evaluation/status.json), [metadata.json](runs/agent_v1_suite_cpu_evaluation/metadata.json).

#### Начальное PPO и текстовые инструкции


**[agent_v1_pilot_20261008T112902Z](runs/agent_v1_pilot_20261008T112902Z)**. Статус артефакта: `resource_stopped`.

**Смысл:** начальный более продолжительный PPO-прогон с периодическими frozen-оценками.

**Итог и вывод:** до ресурсной остановки ни одна оценка не показала deterministic-побед; увеличение объёма опыта в этом запуске не дало наблюдаемого успеха.

Циклов сбора опыта: 11; Новых решений PPO: 36006.

- [rollout_0004.json](runs/agent_v1_pilot_20261008T112902Z/evaluations/rollout_0004.json): D 0/64; R 2/64; difficulty=не записана, старый протокол.
- [rollout_0008.json](runs/agent_v1_pilot_20261008T112902Z/evaluations/rollout_0008.json): D 0/64; R 2/64; difficulty=не записана, старый протокол.
- [rollout_0011.json](runs/agent_v1_pilot_20261008T112902Z/evaluations/rollout_0011.json): D 0/64; R 2/64; difficulty=не записана, старый протокол.

Сохранённых manifest-файлов checkpoint: 12.
Служебная папка [agent_v1_pilot_20261008T112902Z_launch](runs/agent_v1_pilot_20261008T112902Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_pilot_20261008T112902Z/summary.json), [status.json](runs/agent_v1_pilot_20261008T112902Z/status.json), [metadata.json](runs/agent_v1_pilot_20261008T112902Z/metadata.json).

**[agent_v1_learning_20261008T130507Z](runs/agent_v1_learning_20261008T130507Z)**. Статус артефакта: `completed`.

**Смысл:** проверка динамики PPO на облегчённой сложности с deterministic- и stochastic-оценкой.

**Итог и вывод:** deterministic-результат не изменился; stochastic-победы колебались, поэтому устойчивое улучшение по этим оценкам не установлено.

Циклов сбора опыта: 32; Новых решений PPO: 76530.

- [rollout_0000.json](runs/agent_v1_learning_20261008T130507Z/evaluations/rollout_0000.json): D 3/32; S 5/32; R 4/32; difficulty=0.15, старое семплирование по seed.
- [rollout_0004.json](runs/agent_v1_learning_20261008T130507Z/evaluations/rollout_0004.json): D 3/32; S 4/32; R 4/32; difficulty=0.15, старое семплирование по seed.
- [rollout_0008.json](runs/agent_v1_learning_20261008T130507Z/evaluations/rollout_0008.json): D 3/32; S 2/32; R 4/32; difficulty=0.15, старое семплирование по seed.
- [rollout_0012.json](runs/agent_v1_learning_20261008T130507Z/evaluations/rollout_0012.json): D 3/32; S 6/32; R 4/32; difficulty=0.15, старое семплирование по seed.
- [rollout_0016.json](runs/agent_v1_learning_20261008T130507Z/evaluations/rollout_0016.json): D 3/32; S 6/32; R 4/32; difficulty=0.15, старое семплирование по seed.
- [rollout_0020.json](runs/agent_v1_learning_20261008T130507Z/evaluations/rollout_0020.json): D 3/32; S 6/32; R 4/32; difficulty=0.15, старое семплирование по seed.
- [rollout_0024.json](runs/agent_v1_learning_20261008T130507Z/evaluations/rollout_0024.json): D 3/32; S 7/32; R 4/32; difficulty=0.15, старое семплирование по seed.
- [rollout_0028.json](runs/agent_v1_learning_20261008T130507Z/evaluations/rollout_0028.json): D 3/32; S 8/32; R 4/32; difficulty=0.15, старое семплирование по seed.
- [rollout_0032.json](runs/agent_v1_learning_20261008T130507Z/evaluations/rollout_0032.json): D 3/32; S 6/32; R 4/32; difficulty=0.15, старое семплирование по seed.

Сохранённых manifest-файлов checkpoint: 9.
Служебная папка [agent_v1_learning_20261008T130507Z_launch](runs/agent_v1_learning_20261008T130507Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_learning_20261008T130507Z/summary.json), [status.json](runs/agent_v1_learning_20261008T130507Z/status.json), [metadata.json](runs/agent_v1_learning_20261008T130507Z/metadata.json).

**[instruction_v3_integration_20261008T135724Z](runs/instruction_v3_integration_20261008T135724Z)**. Статус артефакта: `completed`.

**Смысл:** короткий интеграционный прогон версии инструкций v3 в цикле обучения и оценки.

**Итог и вывод:** интеграция прошла полный цикл с сохранениями; deterministic-результат снизился, stochastic не изменился, оснований заявлять рост качества нет.

Циклов сбора опыта: 1; Новых решений PPO: 166.

- [rollout_0000.json](runs/instruction_v3_integration_20261008T135724Z/evaluations/rollout_0000.json): D 3/16; S 3/16; R 2/16; difficulty=0, старое семплирование по seed.
- [rollout_0001.json](runs/instruction_v3_integration_20261008T135724Z/evaluations/rollout_0001.json): D 2/16; S 3/16; R 2/16; difficulty=0, старое семплирование по seed.

Сохранённых manifest-файлов checkpoint: 2.
Служебная папка [instruction_v3_integration_20261008T135724Z_launch](runs/instruction_v3_integration_20261008T135724Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/instruction_v3_integration_20261008T135724Z/summary.json), [status.json](runs/instruction_v3_integration_20261008T135724Z/status.json), [metadata.json](runs/instruction_v3_integration_20261008T135724Z/metadata.json).

**[agent_v1_progress_text_20261008T140552Z](runs/agent_v1_progress_text_20261008T140552Z)**. Статус артефакта: `interrupted`.

**Смысл:** PPO с текстовыми инструкциями на начальной сложности и наблюдением за динамикой побед.

**Итог и вывод:** stochastic-результат вырос, deterministic остался ниже начального; запуск остановлен после plateau ради imitation. Последняя приведённая оценка не оценивает последнее сохранение rollout_0020.

Конфигурация: learning_rate=0.0002, curriculum_difficulty=0, evaluation_difficulty=0.

- [rollout_0000.json](runs/agent_v1_progress_text_20261008T140552Z/evaluations/rollout_0000.json): D 7/32; S 8/32; R 7/32; difficulty=0, старое семплирование по seed.
- [rollout_0004.json](runs/agent_v1_progress_text_20261008T140552Z/evaluations/rollout_0004.json): D 4/32; S 10/32; R 7/32; difficulty=0, старое семплирование по seed.
- [rollout_0008.json](runs/agent_v1_progress_text_20261008T140552Z/evaluations/rollout_0008.json): D 4/32; S 9/32; R 7/32; difficulty=0, старое семплирование по seed.
- [rollout_0012.json](runs/agent_v1_progress_text_20261008T140552Z/evaluations/rollout_0012.json): D 5/32; S 9/32; R 7/32; difficulty=0, старое семплирование по seed.
- [rollout_0016.json](runs/agent_v1_progress_text_20261008T140552Z/evaluations/rollout_0016.json): D 5/32; S 16/32; R 7/32; difficulty=0, старое семплирование по seed.

Сохранённых manifest-файлов checkpoint: 3.
Служебная папка [agent_v1_progress_text_20261008T140552Z_launch](runs/agent_v1_progress_text_20261008T140552Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [status.json](runs/agent_v1_progress_text_20261008T140552Z/status.json), [metadata.json](runs/agent_v1_progress_text_20261008T140552Z/metadata.json), [config.json](runs/agent_v1_progress_text_20261008T140552Z/config.json).

#### MLX imitation: curriculum, расширение банка, native/style и corrections


**[agent_v1_mlx_warmup_20261008T144610Z](runs/agent_v1_mlx_warmup_20261008T144610Z)**. Статус артефакта: `completed`.

**Смысл:** переход от PPO-checkpoint к обучению по демонстрациям в MLX на начальной сложности.

**Итог и вывод:** на полном банке выросли и deterministic-, и stochastic-победы, одновременно снизился NLL; imitation улучшило этот режим, но перенос на полную сложность здесь не проверен.

Эпох: 40; обновлений optimizer: 1280; решений демонстраций: 2284; эпизодов корпуса: 256; NLL первой→последней эпохи: 5.118358→0.884864.
Источник: `agent_v1_progress_text_20261008T140552Z/checkpoints/rollout_0020`.
Конфигурация: learning_rate=0.0002, curriculum_difficulty=0.0, evaluation_difficulty=0.0.

- [epoch_0000.json](runs/agent_v1_mlx_warmup_20261008T144610Z/evaluations/epoch_0000.json): D 48/256; S 79/256; R 63/256; difficulty=0.0, полный набор явных индексов уровней.
- [epoch_0010.json](runs/agent_v1_mlx_warmup_20261008T144610Z/evaluations/epoch_0010.json): D 3/32; S 8/32; R 7/32; difficulty=0.0, старое семплирование по seed.
- [epoch_0020.json](runs/agent_v1_mlx_warmup_20261008T144610Z/evaluations/epoch_0020.json): D 5/32; S 10/32; R 7/32; difficulty=0.0, старое семплирование по seed.
- [epoch_0030.json](runs/agent_v1_mlx_warmup_20261008T144610Z/evaluations/epoch_0030.json): D 12/32; S 14/32; R 7/32; difficulty=0.0, старое семплирование по seed.
- [epoch_0040.json](runs/agent_v1_mlx_warmup_20261008T144610Z/evaluations/epoch_0040.json): D 116/256; S 133/256; R 63/256; difficulty=0.0, полный набор явных индексов уровней.

Сохранённых manifest-файлов checkpoint: 1.
Служебная папка [agent_v1_mlx_warmup_20261008T144610Z_launch](runs/agent_v1_mlx_warmup_20261008T144610Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_mlx_warmup_20261008T144610Z/summary.json), [status.json](runs/agent_v1_mlx_warmup_20261008T144610Z/status.json), [metadata.json](runs/agent_v1_mlx_warmup_20261008T144610Z/metadata.json), [config.json](runs/agent_v1_mlx_warmup_20261008T144610Z/config.json), [demonstrations.json](runs/agent_v1_mlx_warmup_20261008T144610Z/demonstrations.json).

**[agent_v1_mlx_coverage32_20261008T151637Z](runs/agent_v1_mlx_coverage32_20261008T151637Z)**. Статус артефакта: `completed`.

**Смысл:** продолжение imitation на увеличенном корпусе демонстраций при той же начальной сложности.

**Итог и вывод:** обе оценки полного банка улучшились относительно входной модели; результат относится к облегчённым условиям и не доказывает освоение полной сложности.

Эпох: 80; обновлений optimizer: 5120; решений демонстраций: 4562; эпизодов корпуса: 512; NLL первой→последней эпохи: 1.031567→0.179186.
Источник: `agent_v1_mlx_warmup_20261008T144610Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=0.0002, curriculum_difficulty=0.0, evaluation_difficulty=0.0.

- [epoch_0000.json](runs/agent_v1_mlx_coverage32_20261008T151637Z/evaluations/epoch_0000.json): D 116/256; S 133/256; R 63/256; difficulty=0.0, полный набор явных индексов уровней.
- [epoch_0020.json](runs/agent_v1_mlx_coverage32_20261008T151637Z/evaluations/epoch_0020.json): D 16/32; S 15/32; R 7/32; difficulty=0.0, старое семплирование по seed.
- [epoch_0040.json](runs/agent_v1_mlx_coverage32_20261008T151637Z/evaluations/epoch_0040.json): D 16/32; S 16/32; R 7/32; difficulty=0.0, старое семплирование по seed.
- [epoch_0060.json](runs/agent_v1_mlx_coverage32_20261008T151637Z/evaluations/epoch_0060.json): D 18/32; S 17/32; R 7/32; difficulty=0.0, старое семплирование по seed.
- [epoch_0080.json](runs/agent_v1_mlx_coverage32_20261008T151637Z/evaluations/epoch_0080.json): D 175/256; S 176/256; R 63/256; difficulty=0.0, полный набор явных индексов уровней.

Сохранённых manifest-файлов checkpoint: 1.
Служебная папка [agent_v1_mlx_coverage32_20261008T151637Z_launch](runs/agent_v1_mlx_coverage32_20261008T151637Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_mlx_coverage32_20261008T151637Z/summary.json), [status.json](runs/agent_v1_mlx_coverage32_20261008T151637Z/status.json), [metadata.json](runs/agent_v1_mlx_coverage32_20261008T151637Z/metadata.json), [config.json](runs/agent_v1_mlx_coverage32_20261008T151637Z/config.json), [demonstrations.json](runs/agent_v1_mlx_coverage32_20261008T151637Z/demonstrations.json).

**[agent_v1_mlx_generalization_20261008T190225Z](runs/agent_v1_mlx_generalization_20261008T190225Z)**. Статус артефакта: `completed`.

**Смысл:** дообучение прежней модели на демонстрациях полной сложности с оценкой на том же уровне сложности.

**Итог и вывод:** победы полного банка выросли, но большинство эпизодов осталось невыигранным; успехи прежнего облегчённого режима нельзя напрямую переносить сюда.

Эпох: 24; обновлений optimizer: 576; решений демонстраций: 3481; эпизодов корпуса: 192; NLL первой→последней эпохи: 2.869826→0.343539.
Источник: `agent_v1_mlx_coverage32_20261008T151637Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=0.0001, curriculum_difficulty=1.0, evaluation_difficulty=1.0.

- [epoch_0000.json](runs/agent_v1_mlx_generalization_20261008T190225Z/evaluations/epoch_0000.json): D 40/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.
- [epoch_0012.json](runs/agent_v1_mlx_generalization_20261008T190225Z/evaluations/epoch_0012.json): D 8/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0024.json](runs/agent_v1_mlx_generalization_20261008T190225Z/evaluations/epoch_0024.json): D 7/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0024_final.json](runs/agent_v1_mlx_generalization_20261008T190225Z/evaluations/epoch_0024_final.json): D 64/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.

Сохранённых manifest-файлов checkpoint: 1.
Служебная папка [agent_v1_mlx_generalization_20261008T190225Z_launch](runs/agent_v1_mlx_generalization_20261008T190225Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_mlx_generalization_20261008T190225Z/summary.json), [status.json](runs/agent_v1_mlx_generalization_20261008T190225Z/status.json), [metadata.json](runs/agent_v1_mlx_generalization_20261008T190225Z/metadata.json), [config.json](runs/agent_v1_mlx_generalization_20261008T190225Z/config.json), [demonstrations.json](runs/agent_v1_mlx_generalization_20261008T190225Z/demonstrations.json).

**[agent_v1_mlx_varied_generalization_20261008T192110Z](runs/agent_v1_mlx_varied_generalization_20261008T192110Z)**. Статус артефакта: `completed`.

**Смысл:** следующая ветвь imitation на другом корпусе демонстраций полной сложности.

**Итог и вывод:** при заметном снижении NLL итог полного банка улучшился лишь на одну победу; убедительного общего прироста игрового качества этот запуск не показывает.

Эпох: 32; обновлений optimizer: 512; решений демонстраций: 3580; эпизодов корпуса: 128; NLL первой→последней эпохи: 4.202556→0.357532.
Источник: `agent_v1_mlx_generalization_20261008T190225Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=8e-05, curriculum_difficulty=1.0, evaluation_difficulty=1.0.

- [epoch_0000.json](runs/agent_v1_mlx_varied_generalization_20261008T192110Z/evaluations/epoch_0000.json): D 64/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.
- [epoch_0016.json](runs/agent_v1_mlx_varied_generalization_20261008T192110Z/evaluations/epoch_0016.json): D 9/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0032.json](runs/agent_v1_mlx_varied_generalization_20261008T192110Z/evaluations/epoch_0032.json): D 8/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0032_final.json](runs/agent_v1_mlx_varied_generalization_20261008T192110Z/evaluations/epoch_0032_final.json): D 65/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.

Сохранённых manifest-файлов checkpoint: 1.
Служебная папка [agent_v1_mlx_varied_generalization_20261008T192110Z_launch](runs/agent_v1_mlx_varied_generalization_20261008T192110Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_mlx_varied_generalization_20261008T192110Z/summary.json), [status.json](runs/agent_v1_mlx_varied_generalization_20261008T192110Z/status.json), [metadata.json](runs/agent_v1_mlx_varied_generalization_20261008T192110Z/metadata.json), [config.json](runs/agent_v1_mlx_varied_generalization_20261008T192110Z/config.json), [demonstrations.json](runs/agent_v1_mlx_varied_generalization_20261008T192110Z/demonstrations.json).

**[agent_v1_mlx_fullbank_uniform_20261008T194142Z](runs/agent_v1_mlx_fullbank_uniform_20261008T194142Z)**. Статус артефакта: `failed`.

**Смысл:** обучение на расширенном корпусе с оценкой полного банка.

**Итог и вывод:** оценка показала небольшой прирост побед, но сохранение оценённых весов сорвалось из-за нехватки диска; результат нельзя предъявлять как доступную frozen-модель.

эпизодов корпуса: 1060.
Источник: `agent_v1_mlx_generalization_20261008T190225Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=6e-05, curriculum_difficulty=1.0, evaluation_difficulty=1.0.
Причина незавершения: Checkpoint save failed after disk space exhaustion; evaluated policy weights were not retained..

- [epoch_0000.json](runs/agent_v1_mlx_fullbank_uniform_20261008T194142Z/evaluations/epoch_0000.json): D 64/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.
- [epoch_0002_final.json](runs/agent_v1_mlx_fullbank_uniform_20261008T194142Z/evaluations/epoch_0002_final.json): D 68/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.

В `checkpoints/` manifest не найден; завершённую сохраняемую модель по этой папке не утверждаем.
Служебная папка [agent_v1_mlx_fullbank_uniform_20261008T194142Z_launch](runs/agent_v1_mlx_fullbank_uniform_20261008T194142Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [status.json](runs/agent_v1_mlx_fullbank_uniform_20261008T194142Z/status.json), [metadata.json](runs/agent_v1_mlx_fullbank_uniform_20261008T194142Z/metadata.json), [config.json](runs/agent_v1_mlx_fullbank_uniform_20261008T194142Z/config.json), [demonstrations.json](runs/agent_v1_mlx_fullbank_uniform_20261008T194142Z/demonstrations.json).

**[agent_v1_mlx_fullbank_balanced_20261008T204156Z](runs/agent_v1_mlx_fullbank_balanced_20261008T204156Z)**. Статус артефакта: `completed`.

**Смысл:** короткая завершённая ветвь imitation на расширенном корпусе от исходного generalization-checkpoint.

**Итог и вывод:** NLL снизился, полный банк дал небольшой прирост побед и checkpoint сохранён; это локальное улучшение, не освоение большинства уровней.

Эпох: 2; обновлений optimizer: 266; решений демонстраций: 28362; эпизодов корпуса: 1060; NLL первой→последней эпохи: 1.472611→1.113369.
Источник: `agent_v1_mlx_generalization_20261008T190225Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=6e-05, curriculum_difficulty=1.0, evaluation_difficulty=1.0.

- [epoch_0000.json](runs/agent_v1_mlx_fullbank_balanced_20261008T204156Z/evaluations/epoch_0000.json): D 64/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.
- [epoch_0002_final.json](runs/agent_v1_mlx_fullbank_balanced_20261008T204156Z/evaluations/epoch_0002_final.json): D 69/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.

Сохранённых manifest-файлов checkpoint: 1.
Служебная папка [agent_v1_mlx_fullbank_balanced_20261008T204156Z_launch](runs/agent_v1_mlx_fullbank_balanced_20261008T204156Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_mlx_fullbank_balanced_20261008T204156Z/summary.json), [status.json](runs/agent_v1_mlx_fullbank_balanced_20261008T204156Z/status.json), [metadata.json](runs/agent_v1_mlx_fullbank_balanced_20261008T204156Z/metadata.json), [config.json](runs/agent_v1_mlx_fullbank_balanced_20261008T204156Z/config.json), [demonstrations.json](runs/agent_v1_mlx_fullbank_balanced_20261008T204156Z/demonstrations.json).

**[agent_v1_mlx_fullbank_paired40_20261008T210207Z](runs/agent_v1_mlx_fullbank_paired40_20261008T210207Z)**. Статус артефакта: `completed`.

**Смысл:** длительное imitation-дообучение на расширенном корпусе с периодическими оценками и итоговой проверкой полного банка.

**Итог и вывод:** полный банк показал существенный прирост побед относительно входной модели; промежуточные короткие оценки колебались и не заменяют итоговую проверку.

Эпох: 40; обновлений optimizer: 5640; решений демонстраций: 32862; эпизодов корпуса: 1123; NLL первой→последней эпохи: 1.398753→0.220867.
Источник: `agent_v1_mlx_generalization_20261008T190225Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=6e-05, curriculum_difficulty=1.0, evaluation_difficulty=1.0.

- [epoch_0000.json](runs/agent_v1_mlx_fullbank_paired40_20261008T210207Z/evaluations/epoch_0000.json): D 64/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.
- [epoch_0010.json](runs/agent_v1_mlx_fullbank_paired40_20261008T210207Z/evaluations/epoch_0010.json): D 13/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0020.json](runs/agent_v1_mlx_fullbank_paired40_20261008T210207Z/evaluations/epoch_0020.json): D 17/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0030.json](runs/agent_v1_mlx_fullbank_paired40_20261008T210207Z/evaluations/epoch_0030.json): D 12/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0040.json](runs/agent_v1_mlx_fullbank_paired40_20261008T210207Z/evaluations/epoch_0040.json): D 19/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0040_final.json](runs/agent_v1_mlx_fullbank_paired40_20261008T210207Z/evaluations/epoch_0040_final.json): D 133/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.

Сохранённых manifest-файлов checkpoint: 1.
Служебная папка [agent_v1_mlx_fullbank_paired40_20261008T210207Z_launch](runs/agent_v1_mlx_fullbank_paired40_20261008T210207Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_mlx_fullbank_paired40_20261008T210207Z/summary.json), [status.json](runs/agent_v1_mlx_fullbank_paired40_20261008T210207Z/status.json), [metadata.json](runs/agent_v1_mlx_fullbank_paired40_20261008T210207Z/metadata.json), [config.json](runs/agent_v1_mlx_fullbank_paired40_20261008T210207Z/config.json), [demonstrations.json](runs/agent_v1_mlx_fullbank_paired40_20261008T210207Z/demonstrations.json).

**[agent_v1_mlx_recovery_refresh_20261008T215614Z](runs/agent_v1_mlx_recovery_refresh_20261008T215614Z)**. Статус артефакта: `failed`.

**Смысл:** попытка продолжить imitation от paired40 на обновлённом корпусе с проверкой эпохального сохранения.

**Итог и вывод:** проверка соответствия snapshot его frozen-оценке не прошла; промежуточный результат нельзя приписывать проверенному итоговому checkpoint.

эпизодов корпуса: 1159.
Источник: `agent_v1_mlx_fullbank_paired40_20261008T210207Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=6e-05, curriculum_difficulty=1.0, evaluation_difficulty=1.0.
Причина незавершения: epoch snapshot does not match its frozen evaluation.

- [epoch_0000.json](runs/agent_v1_mlx_recovery_refresh_20261008T215614Z/evaluations/epoch_0000.json): D 133/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.
- [epoch_0001.json](runs/agent_v1_mlx_recovery_refresh_20261008T215614Z/evaluations/epoch_0001.json): D 15/32; R 2/32; difficulty=1.0, старое семплирование по seed.

Финального checkpoint в `checkpoints/` нет, но сохранён архив эпохи 1: [state.pt.chunks.json](runs/agent_v1_mlx_recovery_refresh_20261008T215614Z/checkpoints_by_epoch/epoch_0001/checkpoints/rollout_0000/state.pt.chunks.json). Его соответствие frozen-оценке не подтверждено; это не утрата всех весов.
Служебная папка [agent_v1_mlx_recovery_refresh_20261008T215614Z_launch](runs/agent_v1_mlx_recovery_refresh_20261008T215614Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [status.json](runs/agent_v1_mlx_recovery_refresh_20261008T215614Z/status.json), [metadata.json](runs/agent_v1_mlx_recovery_refresh_20261008T215614Z/metadata.json), [config.json](runs/agent_v1_mlx_recovery_refresh_20261008T215614Z/config.json), [demonstrations.json](runs/agent_v1_mlx_recovery_refresh_20261008T215614Z/demonstrations.json).

**[agent_v1_mlx_native_styles12_20261008T221545Z](runs/agent_v1_mlx_native_styles12_20261008T221545Z)**. Статус артефакта: `completed`.

**Смысл:** imitation-ветвь native/style от paired40 с расширенным корпусом и проверкой сохранения canonical-навыков.

**Итог и вывод:** финальный полный банк немного лучше исходного, но хуже промежуточной эпохи 8; снижение NLL не обеспечило монотонного роста побед.

Эпох: 12; обновлений optimizer: 4248; решений демонстраций: 78904; эпизодов корпуса: 2830; NLL первой→последней эпохи: 1.534484→0.458657.
Источник: `agent_v1_mlx_fullbank_paired40_20261008T210207Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=3e-05, curriculum_difficulty=1.0, evaluation_difficulty=1.0.

- [epoch_0000.json](runs/agent_v1_mlx_native_styles12_20261008T221545Z/evaluations/epoch_0000.json): D 133/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.
- [epoch_0004.json](runs/agent_v1_mlx_native_styles12_20261008T221545Z/evaluations/epoch_0004.json): D 15/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0004_full_retention.json](runs/agent_v1_mlx_native_styles12_20261008T221545Z/evaluations/epoch_0004_full_retention.json): D 125/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.
- [epoch_0008.json](runs/agent_v1_mlx_native_styles12_20261008T221545Z/evaluations/epoch_0008.json): D 17/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0008_full_retention.json](runs/agent_v1_mlx_native_styles12_20261008T221545Z/evaluations/epoch_0008_full_retention.json): D 144/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.
- [epoch_0012.json](runs/agent_v1_mlx_native_styles12_20261008T221545Z/evaluations/epoch_0012.json): D 18/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0012_final.json](runs/agent_v1_mlx_native_styles12_20261008T221545Z/evaluations/epoch_0012_final.json): D 137/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.

Сохранённых manifest-файлов checkpoint: 1.
Служебная папка [agent_v1_mlx_native_styles12_20261008T221545Z_launch](runs/agent_v1_mlx_native_styles12_20261008T221545Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_mlx_native_styles12_20261008T221545Z/summary.json), [status.json](runs/agent_v1_mlx_native_styles12_20261008T221545Z/status.json), [metadata.json](runs/agent_v1_mlx_native_styles12_20261008T221545Z/metadata.json), [config.json](runs/agent_v1_mlx_native_styles12_20261008T221545Z/config.json), [demonstrations.json](runs/agent_v1_mlx_native_styles12_20261008T221545Z/demonstrations.json).

**[agent_v1_mlx_native_replay2_20261008T233006Z](runs/agent_v1_mlx_native_replay2_20261008T233006Z)**. Статус артефакта: `completed`.

**Смысл:** отдельная ветвь imitation от того же paired40 на корпусе native/style с проверками retention.

**Итог и вывод:** финальный результат выше исходного, но ниже эпохи 8; дополнительные эпохи ухудшили уже достигнутый результат полного банка.

Эпох: 12; обновлений optimizer: 4248; решений демонстраций: 78904; эпизодов корпуса: 2830; NLL первой→последней эпохи: 1.542559→0.467846.
Источник: `agent_v1_mlx_fullbank_paired40_20261008T210207Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=3e-05, curriculum_difficulty=1.0, evaluation_difficulty=1.0.

- [epoch_0000.json](runs/agent_v1_mlx_native_replay2_20261008T233006Z/evaluations/epoch_0000.json): D 133/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.
- [epoch_0004.json](runs/agent_v1_mlx_native_replay2_20261008T233006Z/evaluations/epoch_0004.json): D 15/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0008.json](runs/agent_v1_mlx_native_replay2_20261008T233006Z/evaluations/epoch_0008.json): D 19/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0008_full_retention.json](runs/agent_v1_mlx_native_replay2_20261008T233006Z/evaluations/epoch_0008_full_retention.json): D 146/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.
- [epoch_0012.json](runs/agent_v1_mlx_native_replay2_20261008T233006Z/evaluations/epoch_0012.json): D 15/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0012_final.json](runs/agent_v1_mlx_native_replay2_20261008T233006Z/evaluations/epoch_0012_final.json): D 139/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.

Сохранённых manifest-файлов checkpoint: 1.
Служебная папка [agent_v1_mlx_native_replay2_20261008T233006Z_launch](runs/agent_v1_mlx_native_replay2_20261008T233006Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_mlx_native_replay2_20261008T233006Z/summary.json), [status.json](runs/agent_v1_mlx_native_replay2_20261008T233006Z/status.json), [metadata.json](runs/agent_v1_mlx_native_replay2_20261008T233006Z/metadata.json), [config.json](runs/agent_v1_mlx_native_replay2_20261008T233006Z/config.json), [demonstrations.json](runs/agent_v1_mlx_native_replay2_20261008T233006Z/demonstrations.json).

**[agent_v1_mlx_fresh_corrections12_20261009T003818Z](runs/agent_v1_mlx_fresh_corrections12_20261009T003818Z)**. Статус артефакта: `completed`.

**Смысл:** imitation с обновлёнными corrections от paired40 и оценкой сохранения навыков полного банка.

**Итог и вывод:** итог выше входной модели и немного ниже эпохи 8; улучшение есть в этом запуске, но сравнение с соседними ветвями само по себе не выделяет причинный эффект corrections.

Эпох: 12; обновлений optimizer: 4248; решений демонстраций: 81586; эпизодов корпуса: 2830; NLL первой→последней эпохи: 1.592853→0.475428.
Источник: `agent_v1_mlx_fullbank_paired40_20261008T210207Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=3e-05, curriculum_difficulty=1.0, evaluation_difficulty=1.0.

- [epoch_0000.json](runs/agent_v1_mlx_fresh_corrections12_20261009T003818Z/evaluations/epoch_0000.json): D 133/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.
- [epoch_0004.json](runs/agent_v1_mlx_fresh_corrections12_20261009T003818Z/evaluations/epoch_0004.json): D 16/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0008.json](runs/agent_v1_mlx_fresh_corrections12_20261009T003818Z/evaluations/epoch_0008.json): D 19/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0008_full_retention.json](runs/agent_v1_mlx_fresh_corrections12_20261009T003818Z/evaluations/epoch_0008_full_retention.json): D 144/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.
- [epoch_0012.json](runs/agent_v1_mlx_fresh_corrections12_20261009T003818Z/evaluations/epoch_0012.json): D 21/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0012_final.json](runs/agent_v1_mlx_fresh_corrections12_20261009T003818Z/evaluations/epoch_0012_final.json): D 143/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.

Сохранённых manifest-файлов checkpoint: 1.
Служебная папка [agent_v1_mlx_fresh_corrections12_20261009T003818Z_launch](runs/agent_v1_mlx_fresh_corrections12_20261009T003818Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_mlx_fresh_corrections12_20261009T003818Z/summary.json), [status.json](runs/agent_v1_mlx_fresh_corrections12_20261009T003818Z/status.json), [metadata.json](runs/agent_v1_mlx_fresh_corrections12_20261009T003818Z/metadata.json), [config.json](runs/agent_v1_mlx_fresh_corrections12_20261009T003818Z/config.json), [demonstrations.json](runs/agent_v1_mlx_fresh_corrections12_20261009T003818Z/demonstrations.json).

#### PPO после imitation и парная visual BC серия


**[agent_v1_mlx_online32_20261009T014848Z](runs/agent_v1_mlx_online32_20261009T014848Z)**. Статус артефакта: `resource_stopped`.

**Смысл:** переход к PPO после imitation для проверки дальнейшего улучшения и сохранения приобретённых навыков.

**Итог и вывод:** запуск остановлен по ресурсам; полный банк хуже родительской модели, короткая оценка тоже снизилась. Полезность этих PPO-обновлений не подтверждена.

Циклов сбора опыта: 3; Новых решений PPO: 7214.
Источник: `agent_v1_mlx_fresh_corrections12_20261009T003818Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=1e-05, curriculum_difficulty=1.0, evaluation_difficulty=1.0.

- [rollout_0000.json](runs/agent_v1_mlx_online32_20261009T014848Z/evaluations/rollout_0000.json): D 9/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [rollout_0003.json](runs/agent_v1_mlx_online32_20261009T014848Z/evaluations/rollout_0003.json): D 6/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [rollout_0003_full_retention.json](runs/agent_v1_mlx_online32_20261009T014848Z/evaluations/rollout_0003_full_retention.json): D 141/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.

Сохранённых manifest-файлов checkpoint: 2.
Служебная папка [agent_v1_mlx_online32_20261009T014848Z_launch](runs/agent_v1_mlx_online32_20261009T014848Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_mlx_online32_20261009T014848Z/summary.json), [status.json](runs/agent_v1_mlx_online32_20261009T014848Z/status.json), [metadata.json](runs/agent_v1_mlx_online32_20261009T014848Z/metadata.json), [config.json](runs/agent_v1_mlx_online32_20261009T014848Z/config.json).

**[agent_v1_mlx_lowerlr32_20261009T020602Z](runs/agent_v1_mlx_lowerlr32_20261009T020602Z)**. Статус артефакта: `completed`.

**Смысл:** PPO от того же imitation-checkpoint с меньшим learning rate.

**Итог и вывод:** завершённое обучение ухудшило полный банк относительно родителя; уменьшение шага в этой конфигурации не предотвратило потери навыков.

Циклов сбора опыта: 32; Новых решений PPO: 77160.
Источник: `agent_v1_mlx_fresh_corrections12_20261009T003818Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=3e-06, curriculum_difficulty=1.0, evaluation_difficulty=1.0.

- [rollout_0000.json](runs/agent_v1_mlx_lowerlr32_20261009T020602Z/evaluations/rollout_0000.json): D 9/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [rollout_0016.json](runs/agent_v1_mlx_lowerlr32_20261009T020602Z/evaluations/rollout_0016.json): D 7/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [rollout_0032.json](runs/agent_v1_mlx_lowerlr32_20261009T020602Z/evaluations/rollout_0032.json): D 6/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [rollout_0032_full_retention.json](runs/agent_v1_mlx_lowerlr32_20261009T020602Z/evaluations/rollout_0032_full_retention.json): D 137/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.

Сохранённых manifest-файлов checkpoint: 2.
Служебная папка [agent_v1_mlx_lowerlr32_20261009T020602Z_launch](runs/agent_v1_mlx_lowerlr32_20261009T020602Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_mlx_lowerlr32_20261009T020602Z/summary.json), [status.json](runs/agent_v1_mlx_lowerlr32_20261009T020602Z/status.json), [metadata.json](runs/agent_v1_mlx_lowerlr32_20261009T020602Z/metadata.json), [config.json](runs/agent_v1_mlx_lowerlr32_20261009T020602Z/config.json).

**[agent_v1_mlx_curriculum8_20261009T030722Z](runs/agent_v1_mlx_curriculum8_20261009T030722Z)**. Статус артефакта: `completed`.

**Смысл:** PPO на облегчённых тренировочных задачах с проверкой результата на полной сложности.

**Итог и вывод:** итог полного банка ниже родительской модели; облегчённый curriculum в этом запуске не дал прироста целевого качества.

Циклов сбора опыта: 8; Новых решений PPO: 8156.
Источник: `agent_v1_mlx_fresh_corrections12_20261009T003818Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=3e-06, curriculum_difficulty=0.0, evaluation_difficulty=1.0.

- [rollout_0000.json](runs/agent_v1_mlx_curriculum8_20261009T030722Z/evaluations/rollout_0000.json): D 7/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [rollout_0004.json](runs/agent_v1_mlx_curriculum8_20261009T030722Z/evaluations/rollout_0004.json): D 9/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [rollout_0008.json](runs/agent_v1_mlx_curriculum8_20261009T030722Z/evaluations/rollout_0008.json): D 6/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [rollout_0008_full_retention.json](runs/agent_v1_mlx_curriculum8_20261009T030722Z/evaluations/rollout_0008_full_retention.json): D 141/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.

Сохранённых manifest-файлов checkpoint: 2.
Служебная папка [agent_v1_mlx_curriculum8_20261009T030722Z_launch](runs/agent_v1_mlx_curriculum8_20261009T030722Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_mlx_curriculum8_20261009T030722Z/summary.json), [status.json](runs/agent_v1_mlx_curriculum8_20261009T030722Z/status.json), [metadata.json](runs/agent_v1_mlx_curriculum8_20261009T030722Z/metadata.json), [config.json](runs/agent_v1_mlx_curriculum8_20261009T030722Z/config.json).

**[agent_v1_mlx_visual_bc_control_20261009T035450Z](runs/agent_v1_mlx_visual_bc_control_20261009T035450Z)**. Статус артефакта: `completed`.

**Смысл:** контрольное продолжение imitation от fresh_corrections для парной visual BC серии: все эпизоды имеют вес 1. Сравнение должно проверить изменение относительного вклада обычного и изменённого оформления при неизменном корпусе.

**Итог и вывод:** полный банк улучшился и результат подтверждён отдельной оценкой efficient.v1; perfect-побед меньше обычных, поэтому выигрыш ещё не означает эффективное прохождение.

Эпох: 4; обновлений optimizer: 1416; решений демонстраций: 81586; эпизодов корпуса: 2830; NLL первой→последней эпохи: 0.453154→0.399788.
Источник: `agent_v1_mlx_fresh_corrections12_20261009T003818Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=3e-05, curriculum_difficulty=1.0, evaluation_difficulty=1.0.

- [efficient_v1_baseline_20261013.json](runs/agent_v1_mlx_visual_bc_control_20261009T035450Z/evaluations/efficient_v1_baseline_20261013.json): D 155/256, perfect 132/256; R 13/256, perfect 0/256; difficulty=1.0, полный набор явных индексов уровней. scoring=carryrl.atomic.efficient.v1.
- [epoch_0000.json](runs/agent_v1_mlx_visual_bc_control_20261009T035450Z/evaluations/epoch_0000.json): D 143/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.
- [epoch_0004.json](runs/agent_v1_mlx_visual_bc_control_20261009T035450Z/evaluations/epoch_0004.json): D 19/32; R 2/32; difficulty=1.0, старое семплирование по seed.
- [epoch_0004_final.json](runs/agent_v1_mlx_visual_bc_control_20261009T035450Z/evaluations/epoch_0004_final.json): D 155/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.

Сохранённых manifest-файлов checkpoint: 1.
Служебная папка [agent_v1_mlx_visual_bc_control_20261009T035450Z_launch](runs/agent_v1_mlx_visual_bc_control_20261009T035450Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_mlx_visual_bc_control_20261009T035450Z/summary.json), [status.json](runs/agent_v1_mlx_visual_bc_control_20261009T035450Z/status.json), [metadata.json](runs/agent_v1_mlx_visual_bc_control_20261009T035450Z/metadata.json), [config.json](runs/agent_v1_mlx_visual_bc_control_20261009T035450Z/config.json), [demonstrations.json](runs/agent_v1_mlx_visual_bc_control_20261009T035450Z/demonstrations.json).

**[agent_v1_mlx_visual_bc_treatment_20261009T035450Z](runs/agent_v1_mlx_visual_bc_treatment_20261009T035450Z)**. Статус: **прерван**; поле `running` в старом артефакте устарело (подробности ниже).

**Смысл:** проверить уменьшение веса обычных canonical96-эпизодов в обучающей ошибке с 1 до 0.5; вес эпизодов с изменённым оформлением и другим размером остаётся 1. Исходные checkpoint, корпус и порядок выборки те же, что у control. Изменение подтверждено полем `replay_weighting` в metadata обеих ветвей.

**Итог и вывод:** доступна только начальная frozen-оценка; без завершённого checkpoint и итоговой оценки нельзя заключить, помогло ли вмешательство относительно control.

эпизодов корпуса: 2830.
Источник: `agent_v1_mlx_fresh_corrections12_20261009T003818Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=3e-05, curriculum_difficulty=1.0, evaluation_difficulty=1.0.

- [epoch_0000.json](runs/agent_v1_mlx_visual_bc_treatment_20261009T035450Z/evaluations/epoch_0000.json): D 143/256; R 13/256; difficulty=1.0, полный набор явных индексов уровней.

В `checkpoints/` manifest не найден; завершённую сохраняемую модель по этой папке не утверждаем.
Служебная папка [agent_v1_mlx_visual_bc_treatment_20261009T035450Z_launch](runs/agent_v1_mlx_visual_bc_treatment_20261009T035450Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [status.json](runs/agent_v1_mlx_visual_bc_treatment_20261009T035450Z/status.json), [metadata.json](runs/agent_v1_mlx_visual_bc_treatment_20261009T035450Z/metadata.json), [config.json](runs/agent_v1_mlx_visual_bc_treatment_20261009T035450Z/config.json), [demonstrations.json](runs/agent_v1_mlx_visual_bc_treatment_20261009T035450Z/demonstrations.json).

#### 13–15 октября: live styled corrections и frozen диагностика


**[agent_v1_mlx_live_styled_corrections4_20261013T133905Z](runs/agent_v1_mlx_live_styled_corrections4_20261013T133905Z)**. Статус артефакта: `completed`.

**Смысл:** дообучение control-модели на корпусе с дополнительными live styled corrections и проверка сохранения canonical-навыков.

**Итог и вывод:** NLL снизился, но обычных и perfect-побед стало меньше; добавление этих corrections в данном режиме ухудшило canonical-качество.

Эпох: 4; обновлений optimizer: 1480; решений демонстраций: 84727; эпизодов корпуса: 2958; NLL первой→последней эпохи: 0.467067→0.400549.
Источник: `agent_v1_mlx_visual_bc_control_20261009T035450Z/checkpoints/rollout_0000`.
Конфигурация: learning_rate=3e-05, curriculum_difficulty=1.0, evaluation_difficulty=1.0.

- [epoch_0000.json](runs/agent_v1_mlx_live_styled_corrections4_20261013T133905Z/evaluations/epoch_0000.json): D 155/256, perfect 132/256; R 13/256, perfect 0/256; difficulty=1.0, полный набор явных индексов уровней. scoring=carryrl.atomic.efficient.v1.
- [epoch_0004_final.json](runs/agent_v1_mlx_live_styled_corrections4_20261013T133905Z/evaluations/epoch_0004_final.json): D 143/256, perfect 123/256; R 13/256, perfect 0/256; difficulty=1.0, полный набор явных индексов уровней. scoring=carryrl.atomic.efficient.v1.

Сохранённых manifest-файлов checkpoint: 1.
Служебная папка [agent_v1_mlx_live_styled_corrections4_20261013T133905Z_launch](runs/agent_v1_mlx_live_styled_corrections4_20261013T133905Z_launch) содержит сведения о происхождении и запуске этого же эксперимента, не отдельный опыт.
Источники: [summary.json](runs/agent_v1_mlx_live_styled_corrections4_20261013T133905Z/summary.json), [status.json](runs/agent_v1_mlx_live_styled_corrections4_20261013T133905Z/status.json), [metadata.json](runs/agent_v1_mlx_live_styled_corrections4_20261013T133905Z/metadata.json), [config.json](runs/agent_v1_mlx_live_styled_corrections4_20261013T133905Z/config.json), [demonstrations.json](runs/agent_v1_mlx_live_styled_corrections4_20261013T133905Z/demonstrations.json).

**[agent_v1_baseline_validation_nll_20261015T162155Z](runs/agent_v1_baseline_validation_nll_20261015T162155Z)**. Статус артефакта: `completed`.

**Смысл:** измерение ошибки предсказания действий validation-учителя исходной frozen-моделью перед сравнением с дообученными эпохами.

**Итог и вывод:** joint NLL 0.576394 задаёт точку сравнения соответствия учителю; active/idle и источник измерения приведены ниже. Это не отдельная оценка побед.

Источники: [status.json](runs/agent_v1_baseline_validation_nll_20261015T162155Z/status.json), [metadata.json](runs/agent_v1_baseline_validation_nll_20261015T162155Z/metadata.json).

**[agent_v1_epoch2_frozen_diagnostics_20261015T161723Z](runs/agent_v1_epoch2_frozen_diagnostics_20261015T161723Z)**. Статус артефакта: `completed`.

**Смысл:** frozen-проверка промежуточной эпохи live styled corrections, чтобы сопоставить обучение по демонстрациям с фактическими победами.

**Итог и вывод:** joint NLL улучшился до 0.557051, но уже промежуточная модель уступает исходному control по обычным и perfect-победам; лучшее предсказание учителя здесь не означает лучшую игру.


- [frozen_full.json](runs/agent_v1_epoch2_frozen_diagnostics_20261015T161723Z/evaluations/frozen_full.json): D 143/256, perfect 120/256; R 13/256, perfect 0/256; difficulty=1.0, полный набор явных индексов уровней. scoring=carryrl.atomic.efficient.v1.

Источники: [status.json](runs/agent_v1_epoch2_frozen_diagnostics_20261015T161723Z/status.json), [metadata.json](runs/agent_v1_epoch2_frozen_diagnostics_20261015T161723Z/metadata.json).

**[agent_v1_epoch4_validation_nll_20261015T162155Z](runs/agent_v1_epoch4_validation_nll_20261015T162155Z)**. Статус артефакта: `completed`.

**Смысл:** измерение validation teacher NLL финальной эпохи live styled corrections для сравнения с baseline и эпохой 2.

**Итог и вывод:** joint NLL ухудшился до 0.607581 относительно baseline и эпохи 2; падение training NLL не сопровождалось улучшением ни validation-предсказаний, ни побед.

Источники: [status.json](runs/agent_v1_epoch4_validation_nll_20261015T162155Z/status.json), [metadata.json](runs/agent_v1_epoch4_validation_nll_20261015T162155Z/metadata.json).

### Существенные выводы и ограничения старой серии

- `agent_v1_suite_check` содержит [training_verification.json](runs/agent_v1_suite_check/training_verification.json): CPU/MPS обучались по всем 16 задачам (847/691 решений), policy weights изменились, reward weights остались прежними, offline reload воспроизвёл 32 записи. [report.json](runs/agent_v1_suite_check/report.json) и [reward_40ep_probe.json](runs/agent_v1_suite_check/reward_40ep_probe.json) — технические проверки suite/reward, не освоение игр.
- `progress_text` остановлен после RL plateau ради MLX imitation, последнее сохранение rollout_0020; [stop_reason.json](runs/agent_v1_progress_text_20261008T140552Z_launch/stop_reason.json). Оценка rollout_0016 не является оценкой checkpoint_0020.
- `fullbank_uniform` прошёл две эпохи и оценку 68/256, но финальные веса потеряны при исчерпании диска. Нельзя выдавать эту оценку за доступную frozen модель. `recovery_refresh` остановился из-за несовпадения epoch snapshot и frozen evaluation; промежуточные 15/32 не подтверждают корректный итоговый checkpoint.
- `visual_bc_treatment` оставил устаревший `running` в status: 3 завершённые эпохи, начало четвёртой, 1 071 optimizer steps; финального summary/checkpoint нет. Это незавершённая ветвь, а не продолжающееся обучение и не парная победа над control. [.cache/visual_bc_pair_20261009](.cache/visual_bc_pair_20261009/protocol.json) — протокол пары; pause/resume служебные события не являются отдельными обучающими запусками.
- Начальный рост imitation на difficulty=0 (48→116→175/256) не переносится прямо на difficulty=1: начало generalization дало 40/256. Расширение корпуса и 40 эпох paired training подняли canonical validation до 133/256; native/style/correction ветви меняли разные компоненты корпуса и не являются одним контролируемым сравнением.
- Последующее PPO не улучшило родительские 143/256: online3 и curriculum8 — 141/256, lowerlr32 — 137/256. [.cache/online_ppo_audit_20261009/canonical_comparison.json](.cache/online_ppo_audit_20261009/canonical_comparison.json) фиксирует относительно родителя 16 новых/18 потерянных побед, [.cache/curriculum8_audit_20261009/canonical_comparison.json](.cache/curriculum8_audit_20261009/canonical_comparison.json) — 8/10. Оба результата скрывают потери отдельных игр.
- [online memory controls](.cache/online_ppo_audit_20261009/memory_controls.json) и [curriculum memory controls](.cache/curriculum8_audit_20261009/memory_controls.json): каждый даёт normal 32/32 и neutral_wait 32/32, обе подсказки верны на 16/16 геометрий; cue flip меняет направление на 16/16. Это узкая контролируемая задача памяти.
- [ppo_gradient_audit](.cache/ppo_gradient_audit_20261009/result.json): passed технический аудит, 64 исходных эпизода/2 331 переход, фиксированная выборка 16 эпизодов/598 переходов (первый эпизод каждой игры), selection не зависел от исходов. Это проверка PPO-вычислений на реальных траекториях, не доказательство полезности обновлений.
- Control visual BC довёл старый canonical банк до 155/256. Отдельная повторная оценка 13 октября по `efficient.v1` также дала 155/256, но perfect score только 132/256. После 128 дополнительных live styled correction эпизодов (корпус 2 830→2 958) четыре эпохи снизили NLL, однако canonical победы ухудшились до 143/256, perfect до 123/256. Epoch2 тоже 143/256, perfect 120/256. Следовательно, снижение imitation loss не гарантирует сохранения навыков.

#### По играм: одинаковый canonical `efficient.v1` validation

| Игра | Control победы | Styled epoch4 победы | Control perfect | Styled perfect |
|---|---:|---:|---:|---:|
| navigation | 10/16 | 10/16 | 6/16 | 7/16 |
| reaction | 16/16 | 16/16 | 16/16 | 16/16 |
| mouse | 15/16 | 10/16 | 15/16 | 10/16 |
| memory | 16/16 | 16/16 | 16/16 | 16/16 |
| timing | 16/16 | 16/16 | 16/16 | 16/16 |
| platforming | 8/16 | 8/16 | 8/16 | 8/16 |
| steering | 11/16 | 13/16 | 11/16 | 13/16 |
| camera | 5/16 | 2/16 | 5/16 | 2/16 |
| interaction | 1/16 | 1/16 | 1/16 | 1/16 |
| causality | 2/16 | 3/16 | 2/16 | 3/16 |
| key_door | 7/16 | 7/16 | 7/16 | 7/16 |
| goal_choice | 16/16 | 16/16 | 0/16 | 0/16 |
| hazards | 5/16 | 3/16 | 5/16 | 3/16 |
| shooting | 11/16 | 9/16 | 10/16 | 9/16 |
| ui | 12/16 | 9/16 | 10/16 | 8/16 |
| chain | 4/16 | 4/16 | 4/16 | 4/16 |

Frozen validation likelihood на одинаковых 32 teacher traces (836 действий) 15 октября, без обновлений и с неизменными checkpoint:

- [agent_v1_baseline_validation_nll_20261015T162155Z](runs/agent_v1_baseline_validation_nll_20261015T162155Z/validation_nll.json): joint NLL 0.576394, active 0.822926, idle 0.242361.
- [agent_v1_epoch2_frozen_diagnostics_20261015T161723Z](runs/agent_v1_epoch2_frozen_diagnostics_20261015T161723Z/validation_nll.json): joint NLL 0.557051, active 0.803505, idle 0.223123.
- [agent_v1_epoch4_validation_nll_20261015T162155Z](runs/agent_v1_epoch4_validation_nll_20261015T162155Z/validation_nll.json): joint NLL 0.607581, active 0.882590, idle 0.234965.

Это teacher-conditional likelihood; не самостоятельные победы агента. Frozen evaluator epoch2 сохранил также paired/trajectory диагностику в своей папке; `evaluation.json` и `evaluations/frozen_full.json` дублируют одну оценку.


<a id="diagnostics"></a>
## Диагностические аудиты и подготовки 15 октября

Эти опыты отделены от обучения политики: `completed` у аудита означает завершённую проверку, а `prepared_not_trained`/`prepared_not_executed` — только подготовку. Ни одна перечисленная ниже подготовленная пара не доказывает пользу метода. Числа проверены по сохранённым JSON; повторный запуск экспериментов при составлении реестра не выполнялся.

### Проверки среды и ручного окна

- [environment_presentation_probe](runs/environment_presentation_probe/reachability.json): проверка достижимости во всех 16 играх, native RGB 64/96/128, сложности 0/1, seeds 0–2: **288/288** записей имеют `success=true`. Файл содержит исходы и ticks, но не удостоверяет обучение или идентичность контроллера; это проверка набора условий среды, не 288 побед обученного агента.
- [navigation_check](runs/navigation_check/report.json): случайные физические действия, seed 0, **64/100** успехов за 3782 ticks; статус `passed`. Отдельная иллюстрация кратчайшего пути явно скриптовая. [window_smoke.json](runs/navigation_check/window_smoke.json): нативный Python-процесс завершился штатно, exit code 0; стрелки/reset/Escape, захват экрана и ввод ОС независимо не проверены, причина закрытия окна не установлена.

### Качество учителя и свежие исправления

[teacher_quality_audit_20261015T163401Z](runs/teacher_quality_audit_20261015T163401Z/audit.json): проверка допуска чистых RGB-демонстраций на всех 64 TRAIN-уровнях каждой из 16 игр, банк 20261008, difficulty 1, максимум 80 шагов. Приняты **1024/1024** эпизода; failed_tasks пуст. Подтверждённого лишнего ввода нет, 64 navigation-маршрута кратчайшие. Модель не загружалась, updates=0, обучающий корпус не сохранялся; peak RGB payload одной игры 109 320 192 байта не является RSS. Проверено качество скриптового учителя, оптимальность остальных игр не установлена.

[fresh_corrections_20261015T164201Z](runs/fresh_corrections_20261015T164201Z/coverage.json): frozen baseline 155/256, банк TRAIN 20261015, по четыре уровня camera/interaction/causality/key_door/hazards/chain, canonical96, difficulty1, лимит 80, полностью самостоятельные действия ученика. Собраны **24 corrections + 24 supporting teachers**; исходы corrections: **5/24 успехов, 13/24 поражений, 6/24 timeout**. Для выбранных 24 runtime нет совпадений с объявленными TRAIN-банками 0/20261008. [Статус](runs/fresh_corrections_20261015T164201Z/status.json): completed, публичный импорт проверен, checkpoint и policy/reward неизменны, optimizer/training updates=0. Это TRAIN-сбор с реальной историей действий, не validation и не результат обучения на этих исправлениях.

### Frozen-аудиты визуальной награды

Цель — проверить чтение RGB и пригодность формулы reward на учительских TRAIN-траекториях baseline, без выбора действий политикой и без обновлений. Во всех трёх аудитах 32 эпизода (два на игру, банк 20261008), **896/896** кадров с правильными событием и score, **32/32** эпизода пригодны для candidate composition; checkpoint/reward неизменны, updates=0.

| Артефакт | HUD-пробы | Статус и границы |
| --- | --- | --- |
| [frozen_reward_audit_20261015T170648Z](runs/frozen_reward_audit_20261015T170648Z/audit.json) | 270 известных + 2 unknown; 272/272 событий верны | completed; ранняя версия аудита |
| [frozen_reward_audit_20261015T171000Z](runs/frozen_reward_audit_20261015T171000Z/audit.json) | 18 известных + 2 unknown; 20/20 событий верны | completed; styled-пробы явно пропущены как неподдерживаемые fixed-size legacy recognizer |
| [frozen_reward_audit_20261015T171100Z](runs/frozen_reward_audit_20261015T171100Z/audit.json) | 270/270 известных score и 272/272 событий | completed; canonical 64/96/128 и все TRAIN themes/layouts |

Последний аудит уточняет полный поддерживаемый словарь. Ни один не измеряет улучшение игры после PPO, калибровку на неизвестных играх или все состояния ученика.

### История сцены: корпуса и frozen-признаки

Цель — подготовить причинные RGB-префиксы с удалённым HUD, реально исполненными прошлыми командами и явной целью для будущего readout. Учительские метки, исход и provenance не являются входом. Во всех завершённых артефактах `model_fit=false`, `ppo_eligible=false`, updates=0.

- [scene_history_corpus_20261015T173038Z](runs/scene_history_corpus_20261015T173038Z/audit.json) и повторный [scene_history_corpus_20261015T173500Z](runs/scene_history_corpus_20261015T173500Z/audit.json): из 48 исходных эпизодов fresh corrections экспортированы **32** (16 teacher + 16 corrections) для interaction/causality/key_door/chain, **16 runtime-групп**, только TRAIN. В каждой версии проверены **873 причинных префикса**, конфликтов меток нет; **247** уникальных masked кадров, 6 829 056 байт RGB. Хеши корпусов различны, обе проверки completed; без отдельного diff причина различия не приписывается обучению.
- [scene_history_suite_20261015T174414Z](runs/scene_history_suite_20261015T174414Z/status.json): **failed** — `episodes.json` превысил byte budget (сохранилось 20 587 532 байта); это незавершённая подготовка.
- [scene_history_suite_20261015T174848Z](runs/scene_history_suite_20261015T174848Z/audit.json): исправленная компактная подготовка завершена, **192 эпизода**, по **32 runtime-группы** TRAIN/validation, overlap=0; **2316** masked кадров, 64 032 768 байт RGB; replay проверен.
- [scene_history_suite_20261015T175308Z](runs/scene_history_suite_20261015T175308Z/audit.json): последующая завершённая версия с явным контролем финальных меток. Все 16 игр, по два уровня каждой игры в TRAIN и validation и три скриптовые ветки (teacher/neutral/perturbed): **96+96 эпизодов**. **123/192** финальные метки достоверны, расхождений и преждевременных положительных меток — 0; остальные 69 неизвестны. Test не играли, его runtime-ID использовали только для защиты от пересечений.
- [scene_history_features_20261015T180950Z](runs/scene_history_features_20261015T180950Z/audit.json): frozen DINO/E5 для предыдущей suite, **2316 кадров + 16 целей**, ширины 384/384, batch4, payload 3 581 952 байта. Загрузчик проверен на **192/192** эпизодах, энкодеры/checkpoint неизменны.
- [scene_history_suite_20261015T190905Z](runs/scene_history_suite_20261015T190905Z/audit.json) и [scene_history_features_20261015T190905Z](runs/scene_history_features_20261015T190905Z/audit.json): пересоздание после нормализации множеств стен/ловушек в runtime-ID. Изменены ID **48/192** записей; все остальные поля и masked RGB идентичны. Корпус снова replay-проверен, прежние frozen-признаки перепривязаны без модели и повторного кодирования. Числа эпизодов и валидных меток сохранены. Старые версии следует считать историческими; это исправление идентичности, не новый результат readout.

### Retention: ретроспектива и неисполненные пары

[retention_historical_audit_20261015T183114Z](runs/retention_historical_audit_20261015T183114Z/comparison.json) сопоставляет baseline **155/256** с историческими epoch2 (740 updates) и epoch4 (1480 updates), обе **143/256**. Каждая относительно baseline сохранила 128/155 прежних побед, потеряла **27/155**, приобрела **15/101** прежних непобед. Между epoch2 и epoch4 — 19 приобретений и 19 потерь на 256 пар. Имена generic API `control`/`anchored` здесь обозначают исторические эпохи: `controlled_design_verified=false`, `anchor_effect_measured=false`, никакого вмешательства anchor не было. Это описательный аудит регрессий при разных бюджетах, не причинный эксперимент retention.

Следующие четыре папки содержат только планы одной mixed-corpus пары:

- [retention_pair_20261015T183114Z](runs/retention_pair_20261015T183114Z/manifest.json);
- [retention_pair_20261015T183700Z](runs/retention_pair_20261015T183700Z/manifest.json);
- [retention_pair_20261015T220644Z](runs/retention_pair_20261015T220644Z/manifest.json);
- [retention_pair_20261015T220920Z](runs/retention_pair_20261015T220920Z/manifest.json), с [проверкой пересоздания](runs/retention_pair_20261015T220920Z/recreation_audit.json).

Во всех manifest: **prepared_not_executed**, updates=0, модель/optimizer не создавались, training_executed=false, checkpoint_created=false. План: baseline 155/256, **2958 эпизодов / 84 727 решений**, anchor 0 против 1.0, одинаковый LR 3e-5, dataset seed8288, batch8, **запланированы**, но не выполнены четыре эпохи / 1480 updates на ветвь. В последнем снимке после подключения OCR сохранены прежние данные и порядок; OCR выключен в обеих ветвях. По SPEC source pin устарел после дальнейших изменений 16 октября: это исторические планы, не актуальные команды запуска и не свидетельство пользы anchor.

### Разнообразие данных и оформление

[diversity_audit_20261015T185026Z](runs/diversity_audit_20261015T185026Z/audit.json): полный сохранённый mixed-корпус, **2958 записей / 84 727 решений**, в том числе 1415 canonical-записей, 1415 posthoc appearance-копий и 128 live-styled rollout-записей. **1543** metadata physical origins, **1379** runtime-групп, **61 360** уникальных RGB; payload 2 666 905 536 байт. Actual-action history — 272 записи, teacher history — 2686. Completed, model/optimizer не загружены, updates=0. Число записей/конфигураций не равно числу независимых опытов или геометрий; качество агента здесь не измерялось.

[appearance_pair_20261015T185026Z](runs/appearance_pair_20261015T185026Z/audit.json): подготовка сравнения `control`/`theme_shift`, **16 источников всех игр + 16 posthoc-копий = 32 записи и 834 решения в каждой ветви**. Loader roundtrip и identity источников проверены. **prepared_not_trained**, updates=0, модель/checkpoint не создавались. Изменение картинки не является новым исполнением; польза аугментации не измерена.

[appearance96_frozen_20261015T185223Z](runs/appearance96_frozen_20261015T185223Z/aggregate.json): completed, baseline frozen, 8 условий RU/EN × canonical96/три theme96, по одному validation-уровню каждой игры. Всего **128 эпизодов политики + 128 random**, updates=0.

| Оформление | RU | EN |
| --- | ---: | ---: |
| Canonical96 | 10/16 | 9/16 |
| Slate, HUD слева, без текстур | 3/16 | 3/16 |
| Sand, HUD слева, без текстур | 4/16 | 7/16 |
| Violet, HUD слева, без текстур | 4/16 | 6/16 |

Random — **2/16** в каждом условии. Наблюдается чувствительность к оформлению прежних весов. Это малая диагностика, не сравнение обучения с/без аугментации; canonical→styled меняет несколько элементов, Violet включает другой шрифт HUD.

### Новая геометрия navigation и динамика reaction

[navigation_geometry_pair_20261015T191500Z](runs/navigation_geometry_pair_20261015T191500Z/manifest.json) и уточнённый [navigation_geometry_pair_20261015T191628Z](runs/navigation_geometry_pair_20261015T191628Z/manifest.json): **prepared_not_trained**, updates=0. К базе 2958 эпизодов подготовлены `repeat` и `new_walls`: **8/8 пар**, по **8 эпизодов / 37 решений** в каждой дельте. Последний manifest подтверждает 8 новых карт, одинаковые упорядоченные длины; [verification](runs/navigation_geometry_pair_20261015T191628Z/verification.json) проверяет replay/родителей. Новые карты исключают фиксированные карты train/validation/test и исходного корпуса, сохраняют старт/цель/кратчайшую дистанцию, меняют действия учителя. Это ограниченное семейство 5×5 с двумя стенами; политики не обучались.

[reaction_dynamics_pair_20261015T193403Z](runs/reaction_dynamics_pair_20261015T193403Z/manifest.json) и уточнённый [reaction_dynamics_pair_20261015T193944Z](runs/reaction_dynamics_pair_20261015T193944Z/manifest.json): **prepared_not_trained**, updates=0. Восемь повторов против восьми новых RGB-исполнений с модулем горизонтальной скорости мяча 2.0 при прежнем знаке и остальных условиях; по **8 эпизодов / 504 решения** на ветвь, длина каждого 63. Активный input различается: 221→269 единиц, нейтральные команды 283→235. Равенство бюджета относится к решениям. [Проверка](runs/reaction_dynamics_pair_20261015T193944Z/verification.json) подтверждает replay и сохранность источников; польза данных для обучения не измерена.

[reaction_dynamics_frozen_20261015T193944Z](runs/reaction_dynamics_frozen_20261015T193944Z/status.json): **failed до первого условия**, `ValueError: task_episode_multipliers must map configured tasks to positive integers`.

[reaction_dynamics_frozen_20261015T194224Z](runs/reaction_dynamics_frozen_20261015T194224Z/aggregate.json): исправленный запуск completed; frozen baseline, только reaction, одинаковые 16 validation-уровней банка 20261008, canonical96, difficulty1, RU/EN. **96 policy + 96 random эпизодов**, updates=0.

| Скорость | RU | EN |
| --- | ---: | ---: |
| Исходная | 16/16 | 16/16 |
| Модуль 2.0 | 14/16 | 15/16 |
| Модуль 2.5 | 12/16 | 10/16 |

Random: 1/16 для 2.0 в каждом языке, 0/16 в остальных условиях. Изменённая динамика снижает качество baseline. Это чувствительность frozen-политики, не доказательство обучения скорости/памяти; RGB-учитель действует реактивно.

### OCR: чтение и фактический вход политики

[screen_text_actual_20261015T202427Z](runs/screen_text_actual_20261015T202427Z/report.json): OCR начального и последнего кадров 101 canonical-непобеды — **202/202** кадров обработаны, источники неизменны; входы actor/reward не менялись, training_updates=0. [HUD-аудит](runs/screen_text_actual_20261015T202427Z/known_hud_readout_audit.json): score и статус точны в **202/202**, покрытие Play135/Failed67. Однако **12/12** navigation-кадров содержат ложные слова в сцене: 58 блоков, max confidence 0.574. Success, RU, новые шрифты и текстуры не проверены; confidence OCR не калиброванная вероятность.

[ocr_policy_input_smoke_20261015T220543Z](runs/ocr_policy_input_smoke_20261015T220543Z/audit.json) и повторная [ocr_policy_input_smoke_20261015T221024Z](runs/ocr_policy_input_smoke_20261015T221024Z/audit.json): завершённые проверки wiring на трёх реальных командах TRAIN navigation, банк 20261016/level48. Frozen checkpoint с Tesseract/E5, **371/383/370** OCR-токенов; live/replay и Torch/MLX совпали. Удаление OCR при одинаковых RGB/истории меняет выходы (максимальная разница 1.0455), checkpoint/actor/reward неизменны, optimizer не создавался, updates=0. Версии имеют разные source pins; вторая дополнена [verification](runs/ocr_policy_input_smoke_20261015T221024Z/verification.json). Это доказательство подключения OCR, не игрового качества, понимания текста или полноценного обучения.


<a id="ordinary"></a>
## Обычные игры: диагностика baseline и BC-серии 15–18 октября 2026

Эта серия не заменила принятую модель: O1, anchor0/1, O2, увеличение длительности, изменение области KL-anchor и сброс AdamW завершились без прохождения критериев принятия. Общий исходный checkpoint — `runs/agent_v1_mlx_visual_bc_control_20261009T035450Z/checkpoints/rollout_0000`, actor SHA `ee379609…`, revision18717, далее **baseline155**. Его frozen deterministic validation: **155/256** побед, по 16 уровней каждой из 16 игр. Это ветвь старой политики Agent v1: RGB96, визуальный encoder с входом 224, ширина 256,4 слоя/4 головы,16 spatial/8 memory tokens, segment64, multilingual-e5-small. Обучающие запуски используют MLX, behavioural cloning совместной физической команды, а не RL; reward заморожен. Здесь нет результатов новой модели и нет test-оценки.

Общий протокол ordinary-пилотов: canonical96, difficulty1, банк 20261008, максимум 80 команд, seed обучения 3176, LR3e-5, batch8 эпизодов, active_idle, canonical replay1; обычные команды ученика и teacher targets хранятся раздельно. Каждый новый опыт снова стартует из baseline155, кроме явно оговорённого сравнения собственных e2/e4 и повторного использования контроля. Повторные seed для отвергнутых пилотов не запускались. Итоговые evaluation/paired-comparison проверены по JSON, а не только по документации. Служебные preparatory status-файлы могут оставаться устаревшими: для факта исполнения ниже приоритет имеют `execution/report.json`, training summary и оценки.

### 15 октября: ошибки baseline, спасение учителем и покрытие corrections

- **`ordinary_baseline_failures_20261015T200714Z` — завершённая frozen-диагностика.** Повторены все 256 validation-прохождений; сохранены все 101 непобеды baseline:67 поражений и 34 эпизода, исчерпавших горизонт. Непобеды по играм: navigation6, mouse1, platforming8, steering5, camera11, interaction15, causality14, key_door9, hazards11, shooting5, ui4, chain12. У всех 101 зафиксировано хотя бы одно расхождение с учителем; это наблюдение, а не доказанная причина поражения. RGB/реальные действия/quality воспроизведены. Updates0; исходные checkpoint/reward неизменны. Источники: [evaluation](runs/ordinary_baseline_failures_20261015T200714Z/evaluation.json), [наблюдения](runs/ordinary_baseline_failures_20261015T200714Z/ordinary_error_observations.json), [status](runs/ordinary_baseline_failures_20261015T200714Z/status.json).
- **`ordinary_teacher_rescue_20261015T204900Z` — завершённая проверка контроллера на всех 101 validation-непобедах.** Учитель с начала спас 101/101. После 20 команд ученика —61/61 ещё незавершённых, остальные 40 уже терминальны; после 40 —51/52, один timeout,49 терминальных; после 60 —36/36,65 терминальных. При trigger «8 неизменных RGB» —40/43 спасений,3 timeout; ещё 57 trigger не достигли и 1 префикс уже терминален. Ошибок исполнения учителя 0. Это разные anchors одних эпизодов, не независимые выборки; teacher recovery не является победой обученной политики, а неудача одного контроллера не доказывает физическую невозможность. Validation-продолжения не импортированы в train. [aggregate](runs/ordinary_teacher_rescue_20261015T204900Z/aggregate.json).
- **`ordinary_train_correction_audit_20261015T210301Z` — завершённый аудит 24 свежих TRAIN-corrections шести игр, по 4.** Исходные learner-трассы имеют 5/24 побед: camera1/4, causality1/4, chain0/4, hazards1/4, interaction1/4, key_door1/4. Учитель с начала 24/24; после 20/40/60 команд —13/13,10/10,7/7 среди ещё незавершённых (11/14/17 уже терминальны). Trigger неизменных RGB:9/10, один timeout,14 не достигли trigger. Это аудит фиксированного корпуса и фаз, без обучения и импорта продолжений; не репрезентативная оценка всей TRAIN-ошибки. [aggregate](runs/ordinary_train_correction_audit_20261015T210301Z/aggregate.json).
- **`ordinary_navigation_corrections_20261015T210636Z` — завершённый сбор данных без обучения.** Созданы 8 learner navigation-corrections и 8 поддерживающих teacher-эпизодов,16 всего; банк 20261016. Bundle SHA `8598d4c5…`, revision18717, replacement import проверен, optimizer0. [status](runs/ordinary_navigation_corrections_20261015T210636Z/status.json), [coverage](runs/ordinary_navigation_corrections_20261015T210636Z/coverage.json), [split guard](runs/ordinary_navigation_corrections_20261015T210636Z/additional_split_guard.json).
- **`ordinary_navigation_correction_audit_20261015T210714Z` — завершённый аудит этих 8 corrections.** Learner4/8 побед и 4/8 timeout;340 команд. Учитель с начала 8/8; после 20/40/60 —по 4/4 среди незавершённых ипо 4 уже терминальных. Все 4 сработавших RGB-stall anchors спасены; остальные 4 не сработали. Labels не заменялись, обучение 0. Это подтверждает возможность recovery на выбранных застреваниях, но не гарантирует, что BC научится их исправлять. [aggregate](runs/ordinary_navigation_correction_audit_20261015T210714Z/aggregate.json).
- **`ordinary_regression_single_command_20261015T212802Z` — завершённая интервенционная диагностика прежней styled-corrections4 модели.** Сравниваются baseline155 и `agent_v1_mlx_live_styled_corrections4_20261013T133905Z` (143/256); это не O1. Из полного 256-cohort выбраны все 27 потерянных baseline-побед (на полном cohort также 15 приобретений). На выбранной подвыборке baseline27/27, candidate0/27, sham0/27; замена одной команды кандидата на baseline-команду с последующим candidate closed loop спасла 24/27. Это результат внешнего вмешательства, не качество самостоятельной политики и не причинное объяснение обучения. Updates0, frozen pins неизменны. [metadata](runs/ordinary_regression_single_command_20261015T212802Z/metadata.json), [aggregate](runs/ordinary_regression_single_command_20261015T212802Z/aggregate.json).
- **`ordinary_remaining_regression_20261015T213741Z` — завершённый разбор трёх оставшихся случаев, без inference или обучения.** Camera level4, UI level15 и chain level11 после первого исправления снова расходятся с baseline на шагах 5,3,31 соответственно, хотя текущий RGB в точке нового расхождения одинаков. В сохранённом TRAIN teacher-корпусе соответствующие фазы присутствуют: camera64/64 имеют движение/перецентровку, UI64/64 правильное подтверждение, chain64/64 post-door движение и достижение выхода. Наличие фаз не доказывает достаточное покрытие состояний после ошибок. [observations](runs/ordinary_remaining_regression_20261015T213741Z/observations.json), [train coverage](runs/ordinary_remaining_regression_20261015T213741Z/train_coverage.json), [final checks](runs/ordinary_remaining_regression_20261015T213741Z/final_checks.json).

### 16 октября: O1 — свежие navigation-corrections

**`ordinary_o1_20261016T072658Z`**: обе ветви завершены, exit0, по 2 эпохи/258 optimizer updates. Цель — проверить замену 8 исходных correction-слотов свежими navigation-трассами при общей практике всех 16 игр. Каждая ветвь:1024 общих teacher-эпизода (64/игру)+8 corrections; anchor0. Контроль 27306 действий, treatment27621. Исходный AdamW сохранён. Manifest SHA `915e913c…`.

Фиксированный e2: **control143/256, treatment143/256**, baseline155/256. Navigation10/16→10/16→11/16. От baseline контроль приобрёл 15 и потерял 27 побед, treatment18/30; между control и treatment11 приобретений и 11 потерь на 256 парных уровнях. Повторная frozen full-оценка e1 дала 151/256 и 144/256; это отдельная последующая диагностика, не замена endpoint. Свежие corrections улучшили целевую navigation, но общая регрессия и потери других игр не позволили принять модель.

TRAIN navigation-fit с принудительной записанной историей проверен на 64 teacher и двух наборах по 8 corrections. Active/idle objective на свежих 8: baseline3.598163, control3.173727, treatment0.757350; усвоение labels произошло, автономная польза ограничена. На общих teacher64 objective0.019055/0.006465/0.015013. Memory-контроли и повторная проверка всех 16 игр завершены; не следует превращать teacher-forced fit в утверждение о полном автономном TRAIN.

Неполные попытки сохранены в том же каталоге: `execution_attempt_00` прервана KeyboardInterrupt в initial_evaluation,0 updates; `execution_attempt_01` завершилась exit1 при loading,0 updates, потому что повторно используемая evaluation должна принадлежать каталогу evaluations исходного warm-start. Они не являются самостоятельными завершёнными опытами. Финальное `execution/report.json` подтверждает завершение последующего запуска обеих ветвей.

Источники: [manifest](runs/ordinary_o1_20261016T072658Z/manifest.json), [execution](runs/ordinary_o1_20261016T072658Z/execution/report.json), [paired comparison](runs/ordinary_o1_20261016T072658Z/analysis/comparison.json), [e1 full comparison](runs/ordinary_o1_20261016T072658Z/analysis/epoch1_full_comparison.json), [TRAIN-fit](runs/ordinary_o1_20261016T072658Z/analysis/train_fit_comparison.json), [completion verification](runs/ordinary_o1_20261016T072658Z/analysis/completion_verification.json).

### 17 октября: уменьшение забывания — anchor0 против anchor1

**`ordinary_o1_retention_20261016T105024Z`** подготовлен 16-го, реально обучен 17-го. Это новая пара из baseline155 и исходного AdamW на точном O1 treatment-корпусе 1032 эпизода, единственное основное различие — KL-anchor0/1. По 2 полные эпохи/258 updates, оба exit0. E1:144/256 без anchor,156/256 с anchor; заранее выбранный e2: **142/256 и 152/256**. Приобретения/потери относительно baseline:17/30 и 16/19. Anchor уменьшил забывание в этом одном seed, но обе финальные ветви ниже 155 и теряют отдельные навыки; модель не принята. E1=156 не выбирается задним числом.

Memory normal и neutral_wait в обеих ветвях:32/32 правильных первых выборов и побед,16/16 противоположных cue-пар на режим. Frozen TRAIN-fit baseline/anchor1-e2:1032 эпизода,64 teacher на каждую игру плюс 8 corrections; objective0.210336→0.172630. Camera teacher objective0.501246→0.461056, shooting0.886237→0.7736, но interaction0.035548→0.040669 и causality0.016361→0.0277. CPU/MLX parity проверена на первом полном teacher-эпизоде каждой игры; рост fit не доказал улучшение автономной игры.

Результаты и отдельные служебные запуски:

- [execution/report](runs/ordinary_o1_retention_20261016T105024Z/execution/report.json) — оба обучения завершены; внутри также частичный `train_fit16_spotcheck_20261017T094856Z` и последующий полный [TRAIN-fit comparison](runs/ordinary_o1_retention_20261016T105024Z/train_fit16_full_20261017T095151Z/comparison.json).
- **`ordinary_retention_comparison_20261017T093729Z`** — сравнение прервано из-за самоблокировки вложенных readers одного checkpoint-run, updates0; результата сравнения не даёт. [status](runs/ordinary_retention_comparison_20261017T093729Z/status.json).
- **`ordinary_retention_comparison_20261017T093729Z_lock_fixed`** — сравнение после исправления повторного захвата lock в том же потоке завершено; `controlled_design_verified=true`, updates0; веса не переобучались. [comparison](runs/ordinary_retention_comparison_20261017T093729Z_lock_fixed/comparison.json).
- **`ordinary_retention_readout_20261017T093729Z`** — завершённый memory/readout, итог `accepted_model=false`, baseline retained. [decision](runs/ordinary_retention_readout_20261017T093729Z/decision.json), [status](runs/ordinary_retention_readout_20261017T093729Z/status.json).

### 17 октября: O2 — свежие corrections шести обычных игр

**`ordinary_o2_20261017T122207Z`**:1024 прежних teacher-эпизода плюс 24 corrections (camera, interaction, causality, key_door, hazards, chain; по 4). Контроль использует исходные correction-слоты, treatment — fresh bundle от 15 октября. Anchor1 в обеих ветвях выбран после предыдущего пилота. По 2 эпохи/262 updates, exit0, e0 точно воспроизводит baseline155. E1:147/256 и 151/256; fixed e2: **146/256 и 146/256**. Целевые 6 игр:baseline24/96, control28/96, treatment23/96. От baseline приобретения/потери 17/26 и 19/28; control→treatment21/21. Равная сумма не означает одинаковые политики. Обе отклонены.

Полный frozen TRAIN-fit на одном treatment-корпусе 1048 эпизодов/28120 действий: active/idle objective0.291945→0.241241; pooled NLL0.429296→0.346834. Это диагностическое усвоение, не победы. Отдельно на 4 camera TRAIN-slots учитель спас 4/4 с начала и 4/4 после 20 команд; после 40 —3/3 (один уже терминален);2/2 сработавших RGB-stall anchors. Это не проверка всех labels или любых validation-состояний. Полные failure-traces обеих ветвей сохранены. Memory normal/neutral_wait:по 32/32 выборов и побед,16/16 пар в каждой ветви и режиме. Test не использован.

До успешного запуска `attempt_without_frame_cache` прерван в initial_evaluation после 121/512 evaluation-эпизодов; epochs0, optimizer0. Это неполная техническая попытка, не обученная модель. **`ordinary_o2_launch_20261017T122207Z`** — отдельная завершённая data-only подготовка (exit0,206.2 с), source/config/manifest/criteria и копия source; `training_executed=false` относится к подготовке, не отменяет последующее обучение в основном каталоге.

Источники: [execution](runs/ordinary_o2_20261017T122207Z/execution/report.json), [comparison](runs/ordinary_o2_20261017T122207Z/readout/comparison.json), [decision](runs/ordinary_o2_20261017T122207Z/readout/decision.json), [TRAIN-fit](runs/ordinary_o2_20261017T122207Z/train_fit_full_20261017T131435Z/comparison.json), [camera rescue](runs/ordinary_o2_20261017T122207Z/readout/camera_train_rescue/summary.json), [launch preparation](runs/ordinary_o2_launch_20261017T122207Z/preparation_inputs.json).

### 17 октября: длительность — две против четырёх эпох

**`ordinary_duration_20261017T132858Z` — неудачная первая попытка.** Тот же O2 treatment-корпус, baseline+AdamW, anchor1, план 4 эпохи. После e1 guard обнаружил несовпадение actor hash с историческим O2/e1, хотя все 512 policy/random исходов, steps и quality совпали; BC-objective отличается на 1.96e-8, train NLL на 2.73e-8. Причина численного расхождения не установлена. Выполнено 131 update, training exit130, e1 checkpoint не сохранён, `contrast_valid=false`; результата четырёх эпох нет. [interruption analysis](runs/ordinary_duration_20261017T132858Z/interruption_analysis.json), [execution status](runs/ordinary_duration_20261017T132858Z/execution/status.json).

**`ordinary_duration_trajectory_20261017T135245Z` — завершённый повтор с корректным сравнением внутри одной траектории.** Собственный e2 сравнивается с собственным e4; исторический O2/e2 только ориентир.4 эпохи/524 updates, exit0; e1/e2/e3/e4: **151/146/150/147 из 256**. Между e2/e4 приобретено 26 и потеряно 25 побед; от baseline к e4 —16/24. Целевые 6 игр:24/96→23/96→24/96 для baseline/e2/e4. Fixed e4 отклонён. Исторический и собственный e2 имеют разные hashes, но одинаковые 256 policy outcomes; это не битовая воспроизводимость.

Validation teacher NLL0.576394(e0)→0.565486(e2)→0.596213(e4). Online BC objective по эпохам 0.290355→0.260095→0.247830→0.223182, но полный frozen TRAIN-fit всех 1048 на e4 не измерялся. На 9 camera TRAIN-anchors при принудительно записанной истории ни baseline, ни e2, ни e4 не повторили точную greedy-команду учителя:0/9 у всех; снижение teacher NLL не означает эквивалентность действий или recovery. Memory normal/neutral_wait сохранена:32/32 и 16/16 пар. Это один пилот, не доказательство бесполезности любых дополнительных эпох.

Источники: [execution status](runs/ordinary_duration_trajectory_20261017T135245Z/execution/status.json), [comparison](runs/ordinary_duration_trajectory_20261017T135245Z/readout/comparison.json), [decision](runs/ordinary_duration_trajectory_20261017T135245Z/readout/decision.json), [camera anchors](runs/ordinary_duration_trajectory_20261017T135245Z/readout/camera_policy_anchors.json). **`ordinary_duration_launch_20261017T132858Z`** и **`ordinary_duration_trajectory_launch_20261017T135245Z`** — launch/hot-input/storage-policy metadata соответствующих попыток; сами не отдельные обучения. Их `not_started`/`training_launched=false` — снимки подготовки, реальные исполнения описаны выше.

### 17 октября: область KL-anchor и направление AdamW

**`ordinary_anchor_scope_20261017T143503Z` — только подготовка без исполнения.** Manifest, preflight и verification готовы, `training_executed=false`, optimizer0. **`ordinary_anchor_scope_launch_20261017T143503Z`** содержит hot inputs и прежнюю storage policy этой подготовки. [status](runs/ordinary_anchor_scope_20261017T143503Z/status.json), [verification](runs/ordinary_anchor_scope_20261017T143503Z/verification.json).

**`ordinary_anchor_scope_20261017T144837Z` — реально завершённая пара.** Обе ветви стартуют из baseline+AdamW, точный O2 treatment-корпус 1048, anchor1,4 эпохи/524 updates каждая. Контроль якорит все 1048 эпизодов; treatment исключает 24 corrections из KL, сохраняя их BC. KL teacher нормируется по прежней популяции 1048; runtime coverage1024/1048 против 1048/1048 проверена. Manifest SHA `b75bf156…`. Первый запуск supervisor (`execution_prelaunch_failure_20261017T145306Z`) завершился до обучения с `AttributeError: 'str' object has no attribute 'open'`; последующий report подтверждает exit0 обеих ветвей и `contrast_valid=true`.

Control по эпохам 151/146/150/147, treatment151/150/148/146 из 256. Fixed e4: **147/256 против 146/256**, baseline155. Приобретения/потери baseline→control16/24, baseline→treatment16/25; control→treatment21/22. Camera улучшилась до 9/16 против baseline5/16 и control6/16; platforming10/16 против 8/16 и 5/16. Однако steering8/16, hazards2/16, chain0/16 у treatment; шесть correction-игр суммарно 21/96 против baseline24/96. Новая модель не принята. Memory normal/neutral_wait:по 32/32 выборов и побед,16/16 пар в каждой ветви. Снижение BC и отдельные улучшения не компенсируют утраченные игры.

Завершённые последующие TRAIN-диагностики, updates0:

- 9 camera anchors: точная greedy-команда teacher0/9 у baseline/control/treatment; средний teacher joint NLL21.722964/14.790536/13.447659. Только forced recorded history, не автономное прохождение.
- 22 уникальных minibatch первого shuffle, содержащих все 24 corrections, исследованы при baseline и treatment e4. Raw descent снижает aggregate teacher BC22/22 у обоих, correction BC22/22 и 21/22. Массовый конфликт суммарных raw gradients на этой подвыборке не подтверждён; это не исторические градиенты всех тренировочных шагов.
- Арифметика AdamW без применения update на тех же 22 minibatch: saved moments baseline дают направление увеличения teacher BC5/22, zero moments0/22; correction BC снижается 22/22 в обоих случаях. У treatment-e4 teacher BC снижается 22/22, correction BC21/22 saved и 22/22 zero. Zero moments меняют и масштаб: медианная норма шага около 0.073 вместо 0.016, приблизительно 4.5×. Локальные линейные проекции не доказывают причину игровых регрессий или пользу reset.

Источники: [execution](runs/ordinary_anchor_scope_20261017T144837Z/execution/report.json), [comparison](runs/ordinary_anchor_scope_20261017T144837Z/readout/comparison.json), [decision](runs/ordinary_anchor_scope_20261017T144837Z/readout/decision.json), [camera anchors](runs/ordinary_anchor_scope_20261017T144837Z/readout/camera_policy_anchors.json), [BC gradients](runs/ordinary_anchor_scope_20261017T144837Z/readout/bc_gradients/summary.json), [AdamW direction](runs/ordinary_anchor_scope_20261017T144837Z/readout/adamw_direction/summary.json). **`ordinary_anchor_scope_launch_20261017T144837Z`** — служебные hot inputs завершённой пары, не ещё один эксперимент. Корневой `status.json` основной пары остался `prepared_not_executed`; итог определяется завершённым execution report и обученными checkpoint/evaluation, а не этим старым снимком.

### 17 октября: initial AdamW reset

**`ordinary_optimizer_reset_20261017T160848Z`** — завершённая treatment-ветвь. В производном warm-start сохранены actor/reward baseline и остальные поля,128 записей optimizer state очищены до пустого словаря. Те же 1048 эпизодов,4 эпохи/524 updates, source/settings; original baseline не перезаписан. Контроль 147/256 из anchor-scope повторно использован после identity-проверки, новый control не запускался. Manifest SHA `caca72cc…`.

E1/e2/e3/e4: **151/141/144/147 из 256**; fixed e4 равен reused-control147 и ниже baseline155. От baseline12 приобретений и 20 потерь; control→reset18/18. Reset изменил траекторию и набор выигранных уровней, но улучшение качества не подтверждено. Memory normal/neutral_wait:32/32 выборов и побед,16/16 пар. Final validation joint NLL0.583222 против baseline0.576394. Exit0, contrast_valid, pending artifacts пуст; кандидат отклонён, test закрыт. Корневой prepared status устарел; `execution/report.json` подтверждает завершение. **`ordinary_optimizer_reset_launch_20261017T160848Z`** — metadata hot-inputs для original baseline, derived reset input, reused-control и корпуса.

Источники: [execution](runs/ordinary_optimizer_reset_20261017T160848Z/execution/report.json), [comparison](runs/ordinary_optimizer_reset_20261017T160848Z/readout/comparison.json), [decision](runs/ordinary_optimizer_reset_20261017T160848Z/readout/decision.json), [training summary](runs/ordinary_optimizer_reset_20261017T160848Z/treatment_training/summary.json).

### Сводка по играм: фиксированные endpoints ordinary-серии

Все ячейки — победы **из 16** validation-уровней; сумма — **из 256**. O1/O2/retention — e2; duration/scope/reset — e4. Таблица не ранжирует модели по лучшей промежуточной эпохе. Scope-control —147/256; именно этот контроль переиспользован в reset.

| Игра | Baseline | O1 C | O1 T | Anchor 0 | Anchor 1 | O2 C | O2 T | Duration | Scope C | Scope T | Reset |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| navigation | 10 | 10 | 11 | 11 | 11 | 10 | 11 | 10 | 10 | 11 | 11 |
| reaction | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| mouse | 15 | 13 | 12 | 12 | 14 | 11 | 11 | 14 | 14 | 13 | 14 |
| memory | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| timing | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| platforming | 8 | 7 | 7 | 6 | 9 | 6 | 5 | 5 | 5 | 10 | 6 |
| steering | 11 | 12 | 12 | 12 | 13 | 11 | 14 | 11 | 11 | 8 | 12 |
| camera | 5 | 2 | 2 | 2 | 5 | 4 | 5 | 6 | 6 | 9 | 5 |
| interaction | 1 | 4 | 3 | 3 | 2 | 6 | 2 | 3 | 3 | 2 | 3 |
| causality | 2 | 3 | 5 | 5 | 2 | 3 | 3 | 2 | 2 | 1 | 4 |
| key_door | 7 | 6 | 8 | 8 | 7 | 7 | 7 | 6 | 6 | 7 | 6 |
| goal_choice | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| hazards | 5 | 3 | 4 | 4 | 6 | 5 | 4 | 4 | 4 | 2 | 1 |
| shooting | 11 | 8 | 7 | 7 | 8 | 6 | 9 | 8 | 8 | 8 | 10 |
| ui | 12 | 9 | 8 | 8 | 8 | 10 | 9 | 11 | 11 | 11 | 10 |
| chain | 4 | 2 | 0 | 0 | 3 | 3 | 2 | 3 | 3 | 0 | 1 |
| Всего/256 | 155 | 143 | 143 | 142 | 152 | 146 | 146 | 147 | 147 | 146 | 147 |

### 18 октября: TRAIN recovery-корпус и полный автономный аудит 12 игр

**`ordinary_recovery_20261018T203258Z`** — завершённый сбор, без обучения. Для 12 несовершенных обычных игр по 8 TRAIN-уровней:96 pure learner-трасс и 96 парных teacher-продолжений,192 эпизода всего. Anchors фиксировались после pure-трасс, до recovery-исходов. Baseline/reward/source неизменны, updates0; сбор exit0,211.0 с.3880 команд,2579 уникальных RGB;2456 входных кадров с float16 признаками,482869248 байт. Полный round-trip и replay проверены.

Чистый ученик **85/96**:steering4/8, camera4/8, key_door7/8, chain6/8; остальные 8 игр 8/8. Отдельно teacher-from-start18/18, ненулевой learner-prefix→teacher recovery78/78. Эти 96 teacher-эпизодов не являются независимыми 96 примерами или победами новой политики. Фазовый аудит: нужные action-input фазы interaction/causality/UI16/16 парных эпизодов каждой игры, chain post-door14/16; финальные кадры без следующего действия не засчитываются. `fit_ready=false`, `training_contrast_prepared=false`; это сохранённые данные и диагностика, не обученная recovery-модель.

Источники: [plan](runs/ordinary_recovery_20261018T203258Z/plan.json), [collection execution](runs/ordinary_recovery_20261018T203258Z/collection_execution/status.json), [coverage/readout](runs/ordinary_recovery_20261018T203258Z/readout/compactsummary.json).

**`ordinary_train_autonomous_20261018T211543Z`** — завершённый frozen TRAIN-аудит baseline, начат 18 октября UTC (19-го 00:16 МСК), без обучения. Все 64 допустимых уникальных runtime-уровня каждой из 12 игр,768 всего;96 точных прежних seed/level/instruction трасс аутентифицированы и переиспользованы,672 новые. На уровне одна заранее выбранная инструкция:RU/EN32/32 на игру, не оба языка на каждом уровне. Результат **645/768** побед,94 поражения,29 timeout,0 unknown/unfinished.123 непобеды сохранены:11 прежних и 112 новых. Полное model-free воспроизведение 672+96 подтвердило RGB, команды и teacher labels; exit0, updates0, веса/source неизменны. Пик RSS2.052GiB, минимум свободного диска 11.710GiB,566.4 с. Это TRAIN-результат 12 игр, не validation/test и не доказательство 100% всех 16.

| Игра | Победы/64 | Поражения | Timeout |
| --- | ---: | ---: | ---: |
| navigation |64|0|0|
| mouse |62|0|2|
| platforming |44|20|0|
| steering |41|23|0|
| camera |28|36|0|
| interaction |59|2|3|
| causality |64|0|0|
| key_door |60|2|2|
| hazards |64|0|0|
| shooting |41|1|22|
| ui |64|0|0|
| chain |54|10|0|
| Всего/768 |645|94|29|

Наличие 123 ошибок на знакомом TRAIN показывает, что проблема не сводится к переносу на validation. Одновременно navigation/causality/hazards/UI имеют 64/64 TRAIN при неполном validation; разные игры требуют отдельных объяснений. Аудит не устанавливает единственную причину ошибок и не доказывает полезность следующего BC-сбора. Источники: [summary](runs/ordinary_train_autonomous_20261018T211543Z/readout/summary.json), [independent verification](runs/ordinary_train_autonomous_20261018T211543Z/readout/autonomous_verification.json), [plan](runs/ordinary_train_autonomous_20261018T211543Z/plan.json).

### Полнота ordinary-инвентаря

В разделе учтены все 26 top-level каталогов `runs/ordinary_*` на момент проверки:20 основных/диагностических/подготовительных и 6 launch-каталогов. Launch-каталоги не считаются дополнительными обучениями. Неполные попытки O1, O2, duration и prelaunch scope не скрыты и не включены в успешные endpoints. Финальные модели могут храниться через обратимое холодное хранилище; отсутствие горячего `state.pt` само по себе не отменяет существующие execution/checkpoint identity отчёты.


<a id="fresh"></a>
## Pixel RL и новое обучение с нуля, 18–20 октября 2026

В этом разделе **fresh** означает новую обучаемую часть политики и пустой optimizer;
внешние предобученные DINOv2 и multilingual-E5 остаются замороженными. Это не обучение
всего стека с нуля. Reward для fresh-серии обучен заново и затем заморожен; старый
BC-agent155 не является её инициализацией. Политика обучается PPO на собственном опыте,
без teacher-ответов в качестве policy targets. Числа ниже — победы, если не указано иное.
Canonical и pixel считаются отдельно; одинаковый результат не означает независимые уровни.

### Подготовка pixel reward и две удалённые попытки PPO

Цель preflight — проверить чтение HUD и композицию награды при смене оформления,
до затрат на обучение политики. Использован существующий ScreenVisualReward принятой
BC-модели; это диагностические проверки, а не новый обученный агент.

| Каталог | Что реально выполнено | Итог и ограничение |
|---|---|---|
| [pixel_rl_reward_preflight_20261018T214953Z](runs/pixel_rl_reward_preflight_20261018T214953Z/report.json) | 112 начальных кадров, 2016 сочетаний HUD, 861 терминальная граница из 123 исходных TRAIN-эпизодов в 7 стилях, 2 unknown-case | `passed=true`, блокеров нет; 0 policy calls / 0 updates. Исходные 123 эпизода: 94 поражения и 29 таймаутов; 861 — повторное оформление, не новые игры |
| [pixel_rl_reward_preflight_20261018T215038Z](runs/pixel_rl_reward_preflight_20261018T215038Z/report.json) | Повтор проверки с тем же объёмом и сохранённым helper | Также пройдено; минимальная уверенность score 0.999299, event 0.999968; не оценка качества политики |
| [pixel_rl_reward_preflight_progress_20261018T215712Z](runs/pixel_rl_reward_preflight_progress_20261018T215712Z/) | Сохранился пустой каталог | Нет отчёта или результата, завершение не подтверждено |
| [pixel_rl_reward_preflight_progress_20261018T215738Z](runs/pixel_rl_reward_preflight_progress_20261018T215738Z/verification_summary.json) | 53 диагностические траектории, 371 оформленная временная последовательность, 16240 чтений кадров; промежуточный progress всех 16 игр | Пройдено; 0 policy calls / 0 updates; максимальная ошибка телескопирования potential 1.52e−17 |

В последней проверке повтор уже заработанного события в interaction, causality,
key_door, ui и chain не дал положительного итогового return: примерно −0.0561.
Суммарный дисконтированный potential обнуляется на границе эпизода при начальном score0.
Он перераспределяет обратную связь, а не добавляет отдельную цель «набрать score».
Нейтральное действие может давать progress через пассивное выживание; это не навык.

Две последующие попытки были **дообучением BC155**, а не fresh-серией:

| Удалённый каталог и его `_launch` | Реальная остановка | Сохранность результата |
|---|---|---|
| `pixel_rl_ppo_20261018T215616Z`, `pixel_rl_ppo_20261018T215616Z_launch` | Начальная оценка; 0 циклов, 0 PPO-updates; exit −2, ограничение роста системного swap | Обученных весов не было; впоследствии оба каталога удалены |
| `pixel_rl_ppo_20261018T220831Z`, `pixel_rl_ppo_20261018T220831Z_launch` | 1 цикл, 610 решений, 4 PPO-updates; exit −2 из-за резерва диска | Сохранение новых весов не завершилось: частичный ZIP без центрального каталога; восстановимого обученного checkpoint нет |

Вторая попытка до обновлений дала canonical155/256 и pixel51/256 при одинаковом
agent SHA. Это визуальный сдвиг исходной модели, а не ухудшение от RL. Старый
`status=running, phase=saving` не означает живой процесс или сохранённые веса.
Обе попытки удалены по поручению пользователя. Перепроверить исходные оценки уже нельзя:
доступны только [метаданные удаления](.cache/pixel_rl_removed_attempts_20261019.json),
[удаление неполного checkpoint](.cache/pixel_rl_partial_checkpoint_cleanup.json),
[сведения о хранении остановленной попытки](.cache/pixel_rl_stopped_storage.json)
и [маркер остановки](.cache/pixel_rl_current.json). `.cache/pixel_rl_live_overview.json`
и `.cache/pixel_rl_dashboard.json` — исторические снимки интерфейса/сервера, не
доказательство действующего обучения. **Эффект дообучения не подтверждён.**

### Fresh16/32/64: подготовка и калибровка общего LR

Цель — сравнить число пространственных токенов при одинаковых исходных общих тензорах
и префиксах общего набора 64 queries. Один seed20261019, внешние замороженные
DINOv2/E5, width256, 4 слоя/4 головы, 8 memory-токенов, сегмент 64; TRAIN — pixel-art
при исходной механике. AdamW начинается без состояния; счётчики политики нулевые.

| Bundle | Подготовка и реальное обучение |
|---|---|
| [token_comparison_fresh_20261019](runs/token_comparison_fresh_20261019/manifest.json) | Подготовлены конфиги/рецепт; `modelinit_verified=false`, reward ожидается, политика не обучалась |
| [token_comparison_fresh_20261019_v2](runs/token_comparison_fresh_20261019_v2/manifest.json) | Ещё одна подготовка; те же незавершённые gates, 0 policy updates |
| [token_comparison_fresh_20261019_v3](runs/token_comparison_fresh_20261019_v3/manifest.json) | Reward действительно обучен и сохранён; его validation-процесс завершился exit1. Политика ещё не материализована/не обучалась |
| [token_comparison_fresh_20261019_v4](runs/token_comparison_fresh_20261019_v4/manifest.json) | Исправлена проверка tuple/list отчёта; свежий reward V3 перенесён byte-for-byte, подтверждён [lineage](runs/token_comparison_fresh_20261019_v4/reward/fresh_fit_lineage.json). Инициализация проверена; общий LR ещё не выбран |
| [token_comparison_fresh_20261019_v5](runs/token_comparison_fresh_20261019_v5/manifest.json) | Новые нулевые политики для 16/32/64, общий LR1e−4 и свежий reward; gates пройдены. `training_executed=false` в manifest описывает подготовку bundle, а не последующее обучение дочерних run |

[Проверка reward V4](runs/token_comparison_fresh_20261019_v4/reward/validation_report.json):
46766 HUD-проверок и 39164 проверок progress, ошибок 0, покрытие 16/16 игр.
Это диагностические trajectories scripted teacher/neutral/random; политика не
оценивалась и не обновлялась, test не использован. V5 повторно проверил этот reward.

Все четыре fresh32-пробы ниже стартовали независимо от нулевой политики, без передачи
обученных весов между опытами. Их цель — численная/ресурсная пригодность общего LR,
а не выбор по победам. Итоговые checkpoint существуют; реальные процессы завершились exit0.

| Run | Циклы / TRAIN-эпизоды / updates / решения | Greedy canonical / pixel после | Fresh baseline |
|---|---|---|---|
| [token_comparison_fresh_20261019_v4_calibration_lr0.00001](runs/token_comparison_fresh_20261019_v4_calibration_lr0.00001/evaluation.json) | 2 / 32 / 16 / 1237 | 0/32 / 1/32 | 1/32 / 1/32 |
| [token_comparison_fresh_20261019_v4_calibration_lr0.00003](runs/token_comparison_fresh_20261019_v4_calibration_lr0.00003/evaluation.json) | 2 / 32 / 16 / 1237 | 0/32 / 0/32 | 1/32 / 1/32 |
| [token_comparison_fresh_20261019_v4_probe_lr0.0001_cycles2](runs/token_comparison_fresh_20261019_v4_probe_lr0.0001_cycles2/evaluation.json) | 2 / 32 / 16 / 1451 | 0/32 / 0/32 | 1/32 / 1/32 |
| [token_comparison_fresh_20261019_v4_probe_lr0.0001_cycles8](runs/token_comparison_fresh_20261019_v4_probe_lr0.0001_cycles8/evaluation.json) | 8 / 128 / 64 / 5913 | 0/32 / 0/32 | 1/32 / 1/32 |

Random дал 2/32 в каждом оформлении; stochastic-policy в этих отчётах отсутствует.
Максимальный epoch approximate-KL: 9.29e−6, 7.99e−5, 8.16e−4 и 8.16e−4
соответственно. Независимый fixed-TRAIN probe измерял изменение политики на 16 коротких
префиксах, не качество игры. Выбран LR1e−4 как инженерный кандидат с ограниченным KL;
**улучшение игры и оптимальность LR не подтверждены**.

Источники: [selection](runs/token_comparison_fresh_20261019_v5/calibration/selection.json),
[fixed policy shift](runs/token_comparison_fresh_20261019_v4/additional_calibration/fixed_policy_shift.json).
Каталоги [token_comparison_fresh_20261019_v4_calibration_lr0.00001_launch](runs/token_comparison_fresh_20261019_v4_calibration_lr0.00001_launch/process.json)
и [token_comparison_fresh_20261019_v4_calibration_lr0.00003_launch](runs/token_comparison_fresh_20261019_v4_calibration_lr0.00003_launch/process.json)
содержат запуск/ресурсы этих же проб, не дополнительные модели.

### Fresh16/32/64: завершённое основное сравнение V5

Каждая ветвь прошла 128 циклов ×16 игр =2048 TRAIN-эпизодов; LR1e−4,
reward `success=+1, failure=−1`, time/input costs0.001, gamma0.99.
Итоговые `state.pt`/manifest CP128 существуют, процессы exit0 и отсутствие их групп
зафиксированы в results. Это один seed обучения, не многосидовое подтверждение.

| Run / результат | Updates | Решения | TRAIN-победы | Greedy canonical | Greedy pixel |
|---|---:|---:|---:|---|---|
| [token_comparison_fresh_20261019_v5_s20261019_t16](runs/token_comparison_fresh_20261019_v5/results_t16.json) | 1021 | 110522 | 78/2048 | 7/256→1/256 | 4/256→1/256 |
| [token_comparison_fresh_20261019_v5_s20261019_t32](runs/token_comparison_fresh_20261019_v5/results_t32.json) | 973 | 112102 | 77/2048 | 7/256→1/256 | 4/256→1/256 |
| [token_comparison_fresh_20261019_v5_s20261019_t64](runs/token_comparison_fresh_20261019_v5/results_t64.json) | 1022 | 113795 | 61/2048 | 7/256→1/256 | 4/256→1/256 |

Финальная единственная победа каждой ветви — reaction1/16; остальные 15 игр 0/16.
Random13/256. **Эффект fresh-PPO не подтверждён: greedy хуже собственной свежей
инициализации; победителя 16/32/64 нет.** В V5 измерялся greedy, не stochastic;
позднейшие stochastic-оценки других run нельзя приписывать этому сравнению.
Промежуточные подъёмы, например t64 CP64=10/256, не заменяют финальный CP128.

Сопутствующие каталоги запуска, все exit0:
[token_comparison_fresh_20261019_v5_s20261019_t16_launch](runs/token_comparison_fresh_20261019_v5_s20261019_t16_launch/process.json),
[token_comparison_fresh_20261019_v5_s20261019_t32_launch](runs/token_comparison_fresh_20261019_v5_s20261019_t32_launch/process.json),
[token_comparison_fresh_20261019_v5_s20261019_t64_launch](runs/token_comparison_fresh_20261019_v5_s20261019_t64_launch/process.json).

### Failure penalty: подготовительные пробы и прерванные ветви

Цель — проверить единственную запланированную дельту `failure_penalty:1→0`.
Обе ветви заново используют fresh64 CP0 V5, пустой optimizer, LR1e−4,
gamma0.99, GAE0.95; не дообучают результаты token-series. Победы оцениваются
stochastic, greedy и random, отдельно canonical/pixel.

| Каталог | Фактическое исполнение |
|---|---|
| [failure_penalty64_20261019](runs/failure_penalty64_20261019/experiment.json) | Вложенный `runs/probe_control` действительно завершил 2 цикла,16 updates,1451 решение, exit0; main ещё не выполнен |
| [failure_penalty64_20261019_v2](runs/failure_penalty64_20261019_v2/experiment.json) | Перенос layout и переиспользование готовой control-пробы; создана no_failure-проба и начат main-control |
| [failure_penalty64_20261019_v2_probe_control](runs/failure_penalty64_20261019_v2_probe_control/experiment_receipt.json) | Копия численных доказательств исходной control-пробы, не новое обучение: [receipt](runs/failure_penalty64_20261019_v2/reused_probe_control.json) |
| [failure_penalty64_20261019_v2_probe_no_failure](runs/failure_penalty64_20261019_v2_probe_no_failure/experiment_receipt.json) | Новые 2 цикла,16 updates,1270 решений; завершено, checkpoint2 существует |
| [failure_penalty64_20261019_v2_main_control](runs/failure_penalty64_20261019_v2_main_control/status.json) | Прерван после 2 циклов/16 updates/1451 решения; checkpoint2 сохранён; итоговая 128-цикловая оценка не выполнена |
| [failure_penalty64_20261019_v3](runs/failure_penalty64_20261019_v3/experiment.json) | Bundle повторного запуска после изменения ресурсного supervisor |
| [failure_penalty64_20261019_v3_probe_control](runs/failure_penalty64_20261019_v3_probe_control/experiment_receipt.json), [failure_penalty64_20261019_v3_probe_no_failure](runs/failure_penalty64_20261019_v3_probe_no_failure/experiment_receipt.json) | Переиспользованные V2-пробы; [reused_probes](runs/failure_penalty64_20261019_v3/reused_probes.json) явно фиксирует `new_training_executed=false` |

Обе реальные короткие пробы: stochastic3/32→2/32, greedy1/32→0/32 в каждом
оформлении; random1/32. Они проверили исполнимость, а не пользу снятия штрафа.
Main V2 остановлен из-за global-swap-only guard при приемлемой RAM/диске;
[partial-stop receipt](runs/failure_penalty64_20261019_v2/main_control_partial_stop.json)
фиксирует отмену оставшейся оценки и запрет переноса этих обученных весов в main V3.

### Failure penalty V3: итог A128 и B128 после resume

| Run | Реальное исполнение | Stochastic canonical / pixel | Greedy canonical / pixel |
|---|---|---|---|
| Fresh CP0 | 0 updates | 15/256 / 15/256 | 7/256 / 4/256 |
| [failure_penalty64_20261019_v3_main_control](runs/failure_penalty64_20261019_v3/results_control.json), A: штраф 1 | 128 циклов,1022 updates,113795 решений; checkpoint128, итоговые оценки, exit0 | 5/256 / 5/256 | 1/256 / 1/256 |
| [failure_penalty64_20261019_v3_main_no_failure](runs/failure_penalty64_20261019_v3/no_failure_0064_audit.json), B: штраф 0 | Прерван на 64 циклах,510 updates,48521 решении; checkpoint64 сохранён | Последующая frozen CP64:9/256 / 9/256 | 1/256 / 1/256 |
| [failure_penalty64_20261019_v3_main_no_failure_resume64](runs/failure_penalty64_20261019_v3/results_no_failure_resumed.json), B продолжение | До 128 циклов суммарно,1019 updates (509 новых),96560 решений;122/2048 TRAIN-побед; checkpoint128, оценки, exit0 | 14/256 / 14/256 | 8/256 / 8/256 |

В финале A stochastic: navigation4/16, timing1/16; остальные 14 игр 0/16.
B stochastic: navigation4/16, memory7/16, key_door1/16, ui2/16; остальные 12 игр 0/16.
B greedy: navigation1/16 и memory7/16; остальные 14 игр 0/16. Random13/256 везде.
[Проверка memory](runs/failure_penalty64_20261019_v3/memory_choice_final128_audit.json)
восстановила правый конечный выбор во всех 16 эпизодах каждой политики/оформления:
7 правых cue выиграны,9 левых проиграны. Это не подтверждение памяти.

**Эффект снятия штрафа не подтверждён относительно fresh:** B14/256 ниже 15/256.
B лучше A по конечным весам, но resume восстановил параметры/optimizer, а не RNG;
начал последовательность случайности и episode seed заново и добавил CP64-оценку.
Поэтому [парное сравнение](runs/failure_penalty64_20261019_v3/pair_results_resumed.json)
описательное, не чистый причинный эффект штрафа. Один seed, test закрыт.

На первых 1024 траекториях B discounted return поражения−0.02046 при 22.65 шага,
таймаута−0.05930 при 80 шагах. [Арифметическое удаление costs](runs/failure_penalty64_20261019_v3/cost_arithmetic_first64_audit.json)
даёт непобедам 0, победам gamma^(T−1), ошибка≤1.12e−16. Это пересчёт тех же
траекторий, не обучение. Сохранение gamma0.99 оставляет предпочтение ранних побед.

### Отмена time/input costs, 64 цикла

[cost_ablation64_20261020](runs/cost_ablation64_20261020/results.json) — bundle;
[cost_ablation64_20261020_main](runs/cost_ablation64_20261020_main/evaluation.json) —
само обучение. С того же fresh CP0, без переноса B64, обнулены обе стоимости
при уже нулевом failure penalty; прочие reward/PPO/архитектурные условия сохранены.
Контроль — непрерывные первые 64 цикла B, а не B128 с resume.

Завершены 64 цикла,512 updates,48441 решение;50/1024 TRAIN-побед. Checkpoint64
существует, итоговые frozen-оценки выполнены, exit0. Во всех 1024 конечных событиях
RGB соответствует reward:50 Success,564 Failed,410 Play; расхождений 0.
Это [проверка терминальных событий](runs/cost_ablation64_20261020/terminal_reward_audit.json),
не всех промежуточных кадров.

В каждом оформлении stochastic10/256 против B64 9/256 и fresh15/256;
greedy1/256 против B64 1/256 и fresh7/256 canonical,4/256 pixel; random13/256.
Stochastic: navigation8/16, causality1/16, key_door1/16, остальные 13 игр 0/16;
greedy только reaction1/16. **Эффект отмены costs не подтверждён**: инженерный
критерий не пройден, роста относительно свежей модели нет.

### Навигация: четыре независимые fresh-пробы по 512 TRAIN-эпизодов

Цель — проверить обучение одному навыку, затем LR, сбор нескольких эпизодов и GAE.
Каждая проба начинается с того же fresh64 CP0 и пустого optimizer, difficulty1,
reward с failure/time/input=0, gamma0.99; обучается только navigation.
Полная frozen validation после обучения содержит 16 игр ×16 уровней ×3 политики
=768 записей в каждом оформлении. Последние checkpoint существуют; результаты
подтверждают завершение итоговых оценок и exit0. TRAIN-победы не являются validation.

| Bundle и run | Изменение | Updates / решения / TRAIN-победы | Stochastic: все игры; navigation | Greedy: все игры; navigation |
|---|---|---|---|---|
| [navigation_rl_probe512_20261020](runs/navigation_rl_probe512_20261020/results.json); [navigation_rl_probe512_20261020_main](runs/navigation_rl_probe512_20261020_main/evaluation.json) | LR1e−4, сбор 1, λ.95,512 циклов | 1024 /26657 /291/512 | 27/256;13/16 | 2/256;1/16 |
| [navigation_lr3e4_probe512_20261020](runs/navigation_lr3e4_probe512_20261020/results.json); [navigation_lr3e4_probe512_20261020_main](runs/navigation_lr3e4_probe512_20261020_main/evaluation.json) | Только LR1e−4→3e−4 к сбор 1 | 1023 /29390 /214/512 | 9/256;4/16 | 10/256;1/16 |
| [navigation_group8_probe512_20261020](runs/navigation_group8_probe512_20261020/results.json); [navigation_group8_probe512_20261020_main](runs/navigation_group8_probe512_20261020_main/evaluation.json) | LR1e−4,64 коллекции×8 эпизодов,λ.95 | 1017 /25216 /327/512 | 24/256;11/16 | 2/256;1/16 |
| [navigation_lambda1_probe512_20261020](runs/navigation_lambda1_probe512_20261020/results.json); [navigation_lambda1_probe512_20261020_main](runs/navigation_lambda1_probe512_20261020_main/evaluation.json) | Единственная дельта к group8: λ.95→1 | 1022 /26916 /299/512 | 20/256;11/16 | 1/256;0/16 |

В таблице показатели одинаковы в canonical/pixel. Fresh CP0 на исходной полной
оценке: stochastic15/256 (navigation12/16), greedy7/256 canonical и 4/256 pixel
(navigation2/16 и 3/16); random13/256 (navigation7/16).
Первый заранее заданный порог ≥12/16 navigation для обеих политик/оформлений не пройден.

Ненулевые stochastic-игры, кроме navigation (каждый знаменатель 16): сбор 1 — memory7,
ui4, reaction1, interaction1, chain1; LR3e−4 — memory3, ui2; group8 — memory4,
interaction2, causality1, goal_choice2, hazards1, ui3; λ1 — reaction1, memory3,
interaction1, causality1, ui3. Во всех остальных играх 0/16.
Greedy9/16 memory у LR3e−4 объясняется левым конечным выбором 16/16 в canonical,
а не зависимостью от cue; [аудит](runs/navigation_lr3e4_probe512_20261020/memory_choice_canonical_interim_audit.json).
Для прироста memory/UI сбор 1 механизм не установлен; навык памяти не заявляется.

В сбор 1 подтверждены изменения actor26/26, temporal54/54, text projection1/1;
замороженные vision/language424/424 тензора не изменились
([parameter_deltas](runs/navigation_rl_probe512_20261020/parameter_deltas.json)).
Это подтверждает обновление весов, не успешное обучение. У group8 меняются также
native seeds и задержка обновлений: это не чистая абляция нормализации advantages.
У λ1 одновременно меняются advantages и цели critic; его предполагаемое смещение
не измерено. Разный бюджет navigation относительно многозадачного опыта не позволяет
выделить причинный эффект смешения игр.

### Навигация: парные frozen-проверки по пяти action seeds и fresh baseline

Все проверки ниже — **оценка без обучения**, те же 16 уровней повторены 5 раз,
а не 80 независимых уровней. По три политики и два оформления. LR/group8/λ1 —
по 20 новых отчётов/960 эпизодов; свежий контроль —10 новых отчётов/480 эпизодов
и 10 ранее измеренных λ1-отчётов/480 эпизодов по проверенным ссылкам.
Все завершены с итоговым аудитом/exit0, обновлений 0; trained checkpoints не менялись.

| Bundle | Stochastic control→candidate, в каждом оформлении | Дельты пяти повторов | Greedy control→candidate |
|---|---|---|---|
| [navigation_lr_pair5_20261020](runs/navigation_lr_pair5_20261020/results.json) | LR1e−4:56/80→LR3e−4:20/80 | −9,−10,−6,−6,−5 | 5/80→5/80 |
| [navigation_group8_pair5_20261020](runs/navigation_group8_pair5_20261020/results.json) | Сбор 1:56/80→сбор 8:52/80 | −2,−2,0,0,0 | 5/80→5/80 |
| [navigation_lambda1_pair5_20261020](runs/navigation_lambda1_pair5_20261020/results.json) | λ.95:52/80→λ1:52/80 | 0,−2,+3,−4,+3 | 5/80→0/80 |
| [navigation_fresh_pair5_20261020](runs/navigation_fresh_pair5_20261020/results.json) | Fresh58/80→λ1:52/80 | −1,−5,+1,−3,+2 | Canonical10/80→0/80; pixel15/80→0/80 |

Random35/80 у всех, совпадает по эпизодам. Порог продолжения кандидата: +8/80
stochastic-побед и положительная дельта минимум 3/5 повторов в каждом оформлении;
ни один кандидат его не прошёл. Greedy публикуется отдельно, в paired-screen gate не входит.

**Эффект RL относительно fresh не подтверждён:** fresh58/80, сбор 1 56/80,
сбор 8 52/80, λ1 52/80; исходные 12/16 нельзя механически умножать на 5 вместо
повторного измерения. Разница 56/80 против 58/80 не объявляется статистически значимой.
Все выводы ограничены одним seed обучения и повторениями 16 validation-уровней;
test не открывался. Высокий TRAIN-success и локальный single-seed13/16 не доказывают
обучение навыку сверх инициализации.

### Дополнительные диагностики: replay, нейтральный контроль и допуск curriculum

[Replay alignment V2](runs/failure_penalty64_20261019_v3/replay_alignment_v2_result.json)
проверил сохранённый 80-шаговый memory-эпизод, в том числе границу сегмента 63–65:
collection, Torch streaming/replay и MLX replay согласованы; max log-prob error1.91e−6,
value6.56e−7. Веса не менялись; это численное согласование, не качество игры.
V1 остановился до построения графа из-за tuple/list; V2 нормализовал JSON-представление.

[Easy-neutral audit](runs/failure_penalty64_20261019_v3/easy_neutral_train_audit.json)
прогнал без модели и updates по 64 TRAIN-уровня ×16 игр ×2 сложности =2048 эпизодов.
На difficulty0 нейтральные действия выигрывают reaction24/64, timing43/64,
steering24/64; остальные 13 игр 0/64. На difficulty1 только reaction1/64, остальные 15 игр 0/64.
Это подтверждает пассивные победы в облегчённых вариантах, а не пользу curriculum.

[Curriculum admission probe](.cache/curriculum_admission_probe.json) — только
метаданные допустимости TRAIN-уровней в пяти предложенных точках сложности;
обучение не запускалось, curriculum не выбран. Например, при difficulty0
platforming имеет 0/64 допустимых уровней без совпадений с held-out, mouse45/64,
memory61/64. Проверка пяти точек не проверяет непрерывное расписание, а PPO collector
на момент этого аудита admission-filter не использовал. **Эффект curriculum не проверен.**


<a id="rl-improvements"></a>
## Подготовка улучшений после fresh-all16: 20 октября

### `rl_improvements_20261020` · реализация и подготовка, без обучения

- Вопрос: дают ли более связные физические действия, нулевой начальный critic,
  независимые стадии всех 16 игр и ограниченная текстовая награда улучшение побед.
- Статус: код реализован, технические проверки пройдены; подготовка обучения заблокирована
  отсутствующими новыми frozen text/visual reward-артефактами.
  Production PPO и обучение текстового оценщика не запущены;
  новый checkpoint не создан. Unit-тесты с маленькими моделями проверяют код.
- Модель: будущий независимый старт; исходные DINOv2/E5 разрешены, веса собственных
  прежних политик не являются входом. Сравнение zero/temporary/permanent должно
  использовать одинаковую инициализацию и замороженные модели награды.
- Данные: TRAIN easy/bridge/full прошёл проверку эффективной семантики,
  отсутствия пассивных побед и технической достижимости за 80 действий.
  Текстовый корпус: 4 000 групп × 4 RU/EN-варианта, разбиение 60/15/10/15
  до перефразирования, с изоляцией фактических правил управления.
- Контроль: отдельные однофакторные конфигурации управления, critic, curriculum,
  объёма опыта и floor advantages. Бонусы/штрафы управления ограничены по знакам
  0.01 награды победы за эпизод; временное расписание симметрично для обоих знаков.
- Протокол будущей оценки: собственный CP0 и сохранённые циклы, TRAIN отдельно,
  полная validation-сложность, каждая игра n/N, stochastic/greedy и canonical/pixel
  отдельно. Test политики не используется. Test текстового оценщика открывается
  только после выбора head/calibration; это отдельный корпус, не игровые уровни.
- Бюджет этой подготовки: ограниченные технические проверки и генерация данных;
  бюджет production-обучения не выбран. Seed корпуса 20261020, первый предлагаемый
  seed сравнения 3176. Расписание будущих ветвей фиксируется до запуска.
- Критерий готовности: корректные likelihood/GAE/checkpoint, admission всех 48
  пар игра×стадия и проверенные frozen reward-артефакты. Критерий игрового успеха
  не заменяется этими проверками или снижением loss.
- Фактическая подготовка: все48 пар игра×стадия имеют≥16 различных допущенных
  уровней; минимум26 у UI/easy. Admission seed20261020, image96, horizon80,
  SHA256 `ffee347f8f6d377adbc72640e84e7ad229c0ec8163904f6133244824f896aa0c`.
  Первый admission был неуспешным: RGB-probe goal_choice/bridge колебался между
  двумя колонками из-за смещения видимого центра закрытой игроком цели. Исправлен
  только probe; исходная механика и порог допуска сохранены, старый отчёт оставлен.
- Текстовая подготовка: актуален `text_control_data_v2`,16000 записей,
  2400/600/400/600 групп train/dev/calibration/sealed_test. В calibration350
  независимых пригодных групп каждого знака; температуры выбираются на dev.
  Регрессии подтверждают достижимость gate для идеального oracle и отказ для
  модели, игнорирующей supported-команды либо разделяющей настоящие aliases.
  Это проверки методики, не измеренная точность обученного оценщика.
- Будущие сравнения: актуален `comparisons_v3/manifest.json`,9 профилей.
  Базовая первичная награда наследуется без изменения: failure_penalty0,
  time_cost0, input_cost0, success_reward1, progress_weight0.2. Новые штрафы
  не включаются незаметно. Ветки zero/temporary/permanent отличаются расписанием;
  дополнительные профили проверяют объём, управление, critic, curriculum и floor.
  При отсутствующих reward-моделях созданы только `config.template.json`,
  `preparation_ready=false`, команд запуска нет. Предыдущие `comparisons*`
  сохранены как заменённая подготовка; они не исполнялись.
- Ресурсы и наблюдение: mmap frozen features с пределами2048MiB прямых RAM-массивов
  и6144MiB диска; память ОС/моделей/optimizer/временных вычислений отдельно.
  Новая таблица дэшборда проверена в1440/390px: без переполнения и console errors;
  refresh сохраняет canvas/запись, reload восстанавливает уровень и checkpoint.
  Новые поля на историческом запуске отсутствуют и проверены fixtures.
- Результат: игровые победы новой схемы ещё не измерены; полезность не установлена.
  Парные RGB-сцены подготовлены, frozen-оценка production checkpoint не запускалась.
- Проверка реализации: полный Python-прогон3250 passed, frontend68/68,
  Ruff и `git diff --check` прошли. Поздние уточнения конфигураций отдельно
  проверены6 тестами подготовки и23 тестами интеграции/подготовки.
  Независимые ревизии ML и mmap завершены; три найденные ошибки gates исправлены.
- Артефакты: [admission](runs/rl_improvements_20261020/admission.json),
  [первый неуспешный probe](runs/rl_improvements_20261020/admission_failed_probe.json),
  [текстовый корпус](runs/rl_improvements_20261020/text_control_data_v2/manifest.json),
  [сравнения](runs/rl_improvements_20261020/comparisons_v3/manifest.json),
  [парные RGB-сцены](runs/rl_improvements_20261020/input_pairs/pairs.json),
  [проверки и hashes исходников](runs/rl_improvements_20261020/verification.json).

<a id="fresh-all16-v2"></a>
## Новый ограниченный цикл всех16: 21 октября

### `fresh_all16_v2_20261021` · остановлен на проверке текстовой награды

- Вопрос: даёт ли реализованный общий путь с motor_v2, нулевым начальным critic,
  нормировкой PPO и отдельными стадиями игр улучшение относительно свежего CP0
  и предыдущего CP128; помогает ли ограниченная временная текстовая добавка.
- Статус: **остановлен до PPO** после проверки обеих моделей награды и
  единственной направленной ML-доработки. Игровая политика не создавалась.
  Последний parent launcher завершился с exit2; оба text-fit — с exit0,
  но `ready=false`. Отсутствие оставшихся owned/paused процессов проверено.
- Визуальная награда: fresh fit завершён за92.48с, fit-gate пройден.
  Первая проверка стадий остановилась на ошибке покрытия (exit1,252.00с);
  исправленная проверка на **тех же весах** прошла240/240 сочетаний:
  HUD203204 и композиция170128 проверок без ошибок. Это проверка награды,
  не измерение навыка игровой политики.
- Первая текстовая модель:10 эпох,3000 updates,276.42с. Calibration не приняла
  ни одной команды обоих знаков, runtime0/32. Диагностика raw predictions:
  supported719/7692, press0/2000, aliases0/2316. TRAIN содержит86.38% unknown
  для mode и0.60% same для pair; unweighted loss/dev-selection допускает
  игнорирование обязательных редких классов. Ошибок detach или соответствия
  кеша и labels не найдено. Неудачные веса и отчёты сохранены.
- `text_repair1`: зарегистрированный до fit совместный рецепт — fresh
  candidate-conditioned head с query residual/LayerNorm, TRAIN-balanced
  status/mode loss и balanced unique off-diagonal pair loss; выбор по balanced
  development objective. E5/data/split/seed20261021 и gates сохранены.
  Запущен21.10 в01:32:40Москва из `source_text_repair1`; завершён за145.66с,
  10 эпох/3000 updates, выбран epoch10. Initial hash и пустой optimizer
  подтверждены, импортированных head checkpoints нет. Learning repair1/1
  использован; два прежних исправления относятся к инфраструктуре.
- Допуск после repair **не пройден**. На1600 calibration-текстах/400 смысловых
  группах уверенно и правильно распознаны применимые команды16/7692,
  неприменимые2223/36780; всего приняты2240 предсказаний, из них2239 верны,
  один ошибочный штраф. Held16/4572, press0/2000, aliases0/2316; runtime0/32.
  Для применимых команд принято12 независимых групп, nominal lower95=0.7791;
  для неприменимых240 групп, lower95=0.9168 — оба ниже0.99.
  Эти показатели относятся к текстовому оценщику, не к победам в играх.
  Calibration после адаптации считается development, не независимой
  сертификацией; sealed text test остался закрыт.
- Проверки кода: до ML-доработки полный Python3377 passed за525.96с;
  после неё целевые text51/runner56/dashboard251 passed,
  независимый ML-review без блокирующих замечаний. Последний полный прогон
  после ML-доработки:3399 passed за460.34с. Последнее исправление состояния
  сохранённого итога:32 целевых Python и73 frontend passed. Ruff и diff-check
  чистые. Дэшборд проверен на390/1440px
  без overflow и ошибок консоли; старый launch-процесс не подменяет новый,
  известный отказ допуска объясняется по-русски с деталями отдельно;
  сохранённый итог не считается устаревшим только из-за возраста.
- Выполнение core остановлено через4556.50с от первого вычисления;
  фиксация результата и проверки также входят в общий24ч. Всего до конечной
  проверки использовано5022.62с (см. `verification_finished_unix` в итоговом артефакте).
  Резерв доработки освобождён в расчёте full horizon; пределы общего времени,
  конечной оценки2ч и full12ч не увеличивались.
- План политики (не выполнялся): новые обучаемые веса, пустой optimizer;
  внешние DINOv2/E5 заморожены. Старые policy-веса не импортируются;
  проба, пилот и main не передают друг другу обученные policy-веса.
  Фактически обучены новый visual reward и две свежие text-heads.
- Данные: bank20261020, TRAIN easy/bridge/full из pinned admission; полная
  validation-сложность, canonical/pixel отдельно. Корпус текстовой награды
  расширен до16000 примеров/4000 смысловых групп с разделением до
  перефразирования; calibration содержит369 пригодных групп каждого знака.
  Игровой test закрыт до выбора и заморозки конечной модели.
- Контроль (запланирован, не выполнялся): zero/temporary с одинаковой свежей инициализацией seed3176,
  по32 цикла ×8 эпизодов каждой игры. Общие frozen rewards, основной reward:
  success1, progress0.2, failure/time/input0. H_aux32: полный коэффициент
  первые8 циклов, спад до0 к16, затем0 в пилоте и основном запуске.
- Бюджет: общий предел86400с с первого нового вычислительного этапа, без
  обнуления при повторе; rewards3ч, probe1ч, pilot4ч, full12ч, final2ч,
  резерв2ч. Неиспользованное время переносится с сохранением конечного резерва.
- Допуск пилота: в каждом оформлении stochastic прирост≥16/256 против CP0
  и улучшены≥4 игры; TRAIN-успехи на≥2 разных runtime-уровнях минимум8 игр,
  включая≥2 из mouse/platforming/steering/camera/shooting. Temporary выбирается
  при допуске и преимуществе≥8/256 в обоих оформлениях без снижения breadth;
  при близких результатах zero. Это инженерный фильтр, не освоение всех16.
- Основной запуск условный: новый seed3177, по8 эпизодов/игру, минимум128 и
  потолок192 цикла кратно32 по фактической скорости и оставшемуся бюджету.
  Оценки CP0/64/128/final; выбор validation по худшей greedy-игре, затем среднему,
  затем более раннему checkpoint. Test только после заморозки выбора.
- Результат: отрицательный допуск награды, дальнейшие этапы не выполнялись.
  Не проводились PPO-проба, парный пилот, main, CP0/random/старыйCP128,
  проверки памяти и игровых входов, game validation/test. Игровых n/N нет;
  отсутствие оценки не заменяется нулём побед. Новая основная policy и
  подтверждённое улучшение по16 играм не получены.
- Технические отказы: первый visual-fit завершился exit1 до создания весов:
  вызов `python -m carryrl` не поддерживается. Entry point исправлен на
  `carryrl.cli` без повторного обучения уже созданных весов;
  все10 различных будущих CLI прошли `--help`. Исправление имеет отдельный ID
  `visual_fit_entrypoint_fix`; исходный отказ сохранён. Затем проверка покрытия
  ошибочно потребовала random-nonwin в navigation/easy: случайные действия
  выиграли4/4, а neutral дал1/1 nonwin. HUD203204 проверок без ошибок,
  фактических ошибок композиции награды нет; единственная ошибка отчёта —
  отсутствие именно random-nonwin. Исправлен критерий покрытия:
  требуется успех, nonwin любой стратегии и законченная random-траектория.
  Пороги RGB/композиции и240 сочетаний не меняются, исходный отчёт сохранён.
  Инфраструктурные исправления отдельно учитываются от единственной
  направленной ML-доработки; они не разрешают повтор fit или PPO и не обнуляют
  общий срок. Это исправление реализации, а не подбор модели по результату.
  Аудит до ML-доработки:2 инфраструктурных исправления, learning repair0/1; первый
  legacy CLI-repair переклассифицирован с проверкой архивных state/log/plan.
  Подготовленный план со случайно перенесёнными путями frozen-инструкций
  отклонён preflight до запуска; действующий план сохраняет исходные пути.
- Артефакты: [итог остановки](runs/fresh_all16_v2_20261021/result.json),
  [протокол](runs/fresh_all16_v2_20261021/protocol.json),
  [manifest](runs/fresh_all16_v2_20261021/manifest.json),
  [исходный исполняемый протокол](runs/fresh_all16_v2_20261021/execution_plan.json),
  [исправление entrypoint](runs/fresh_all16_v2_20261021/execution_plan_repair1.json),
  [исправление проверяющего кода](runs/fresh_all16_v2_20261021/execution_plan_infrastructure_fix_inputs.json),
  [действующий протокол](runs/fresh_all16_v2_20261021/execution_plan_text_repair1.json),
  [состояние этапов](runs/fresh_all16_v2_20261021/cycle_state.json),
  [родительский supervisor](runs/fresh_all16_v2_20261021/supervisor.json),
  [fit визуальной награды](runs/fresh_all16_v2_20261021/reward_visual/reward_report.json),
  [исходная проверка стадий](runs/fresh_all16_v2_20261021/reward_visual/validation_report.json),
  [успешная проверка стадий v2](runs/fresh_all16_v2_20261021/reward_visual/validation_report_v2.json),
  [недопущенная текстовая модель](runs/fresh_all16_v2_20261021/reward_text/manifest.json),
  [проверка текстовой модели](runs/fresh_all16_v2_20261021/reward_text/validation.json),
  [runtime-инструкции](runs/fresh_all16_v2_20261021/reward_text/runtime_validation.json),
  [проверенный отказ допуска](runs/fresh_all16_v2_20261021/text_fit_failed_gate_receipt.json),
  [диагностика текста](runs/fresh_all16_v2_20261021/text_reward_diagnosis.json),
  [протокол единственной ML-доработки](runs/fresh_all16_v2_20261021/text_repair_protocol.json),
  [повторная текстовая модель](runs/fresh_all16_v2_20261021/reward_text_repair1/manifest.json),
  [её проверка допуска](runs/fresh_all16_v2_20261021/reward_text_repair1/validation.json),
  [её runtime-проверка](runs/fresh_all16_v2_20261021/reward_text_repair1/runtime_validation.json),
  [preflight](runs/fresh_all16_v2_20261021/preflight.json). Код выполняется из
  зафиксированной копии `source_text_repair1`; прежние `source_final` и
  `source_infrastructure_fix` сохранены. Промежуточная `source_validation_fix`
  не запускалась.
  Срок86400с начался20.10 в21:19:09UTC (21.10 в00:19:09Москва), жёсткий конец
  21.10 в21:19:09UTC (22.10 в00:19:09Москва). Цикл остановлен раньше
  из-за качества награды. Heartbeat `carryrl-24` отключён после фиксации итога.
  Receipt и реальные exit codes сохранены для каждого выполненного этапа.

<a id="fresh-all16"></a>
## Новая общая RL-модель на 16 играх: 20 октября

### `fresh_rl_all16_20261020` — независимый старт и общие обновления всех игр

- **Статус:** завершён штатно 20.10.2026. Начало11:05:57 UTC, окончание supervisor
  13:56:34 UTC (16:56:34 Москва): 2ч50мин37с из разрешённых четырёх часов.
  Жёсткий срок15:05:57 UTC не достигнут; hard timeout и soft stop не срабатывали.
  Проверка20.10 в14:28 UTC: supervisor75741, guard75742 и runner75743 отсутствуют,
  supervisor и guard сохранили реальные exit code0. Выполнены128/128 циклов,
  4096 TRAIN-эпизодов, 185111 команд и512 обновлений optimizer; KL не прервал
  ни один цикл. CP0 и конечный CP128 существуют, SHA-256 обоих `state.pt`
  совпадают с manifest. Последние полные canonical/pixel оценки относятся к CP128.
- **Вопрос:** появится ли прогресс собственной свежей политики при равной практике
  всех 16 игр и включении каждой игры в каждый PPO-minibatch. Это инженерная
  гипотеза, а не доказанный рецепт достижения 100%.
- **Исходная модель:** собственных исходных весов нет. Политика, critic, память,
  AdamW и распознаватель награды создаются заново; старые checkpoint, reward,
  корпуса и оценки не импортируются. Используются только внешние замороженные
  DINOv2/E5. Архитектура: width256, 4 слоя/4 головы, 64 spatial/8 memory tokens,
  segment64. Новый начальный CP0 должен быть сохранён и оценён до PPO.
- **Контроль:** собственный CP0 на тех же validation-уровнях, инструкциях,
  оформлениях и action seed. Исторические результаты служат контекстом;
  причинный эффект одного параметра этим новым запуском не выделяется.
- **Обучение:** только PPO для политики, по 2 свежих TRAIN-эпизода каждой игры
  за цикл; полный minibatch16 содержит все игры, microbatch4 ограничивает
  память без промежуточного шага optimizer. Градиенты нормируются по полному
  minibatch; KL, clipping и AdamW применяются один раз. LR1e-4, 2 эпохи,
  gamma0.99, GAE0.95, clip0.1, targetKL0.01, entropy0.001.
  До запуска объём одного сбора ограничен 32 эпизодами для сокращения RAM;
  оценка размера patch tensors — до 960 MiB без модели и графа градиентов.
- **Данные:** банк20261008, difficulty1, горизонт80, RGB96, текущие версии
  AtomicEnv/efficient.v1; pixel-оформление меняется между эпизодами, инструкции
  RU/EN из `configs/atomic_instructions.json`. TRAIN/validation разделены;
  test закрыт. Данные собираются заново; использование прежнего определения
  банка не означает импорт прежних корпусов.
- **Награда:** новый `screen_v2` обучается на новых RGB с отдельными
  TRAIN/validation-уровнями и темами. Это supervised подготовка награды,
  не BC политики. До PPO обязательны fit gate и проверка HUD, исходов и
  промежуточного счёта всех 16 игр. Success1, failure/time/input0,
  progress0.2; дисконтирование сохраняет предпочтение ранней победы.
- **Seed:** политика20261020, reward20261023, evaluation80001.
- **Бюджет:** максимум128 циклов/4096 TRAIN-эпизодов/512 optimizer steps;
  фактический объём может быть меньше из-за KL, времени или ресурсов.
  До четырёх часов от начала исполнения, включая подготовку награды и оценки;
  последние30 минут резервируются под завершение цикла, сохранение и оценку.
  RAM: пауза при доступной памяти менее1GiB или RSS12GiB, продолжение при1.5GiB;
  диск: резерв8GiB, мягкая остановка заранее при10GiB.
- **Протокол:** полная начальная, периодическая и конечная validation-оценка
  16×16 уровней, stochastic и greedy отдельно, random отдельно, canonical
  и pixel отдельно. Показывать каждую игру, TRAIN-покрытие успешными
  траекториями, победы/поражения/таймауты, 5/5 и checkpoint фактической оценки.
- **Критерий пилотного сигнала:** конечный stochastic результат выше собственного
  CP0 минимум на16/256, улучшены минимум4 игры в каждом оформлении. Это
  заранее выбранный инженерный порог для дальнейшей проверки, не статистическое
  доказательство и не освоение всех игр. Итоговая цель остаётся16/16 каждой игры;
  память требует отдельных противоположных cue-проверок.
- **Артефакты подготовки:** [policy config](runs/fresh_rl_all16_20261020/policy_config.json),
  [reward config](runs/fresh_rl_all16_20261020/reward_config.json),
  [manifest](runs/fresh_rl_all16_20261020/manifest.json) с хешами кода и конфигураций,
  [копия исходников](runs/fresh_rl_all16_20261020/source_snapshot/).
  Целевые проверки: 220 тестов прошли; после последних исправлений runner,
  supervisor и microbatch — 60 тестов, Ruff без ошибок. Проверен реальный
  запуск preflight под resource guard без обучения
  ([process receipt](.cache/fresh_rl_guard_preflight_20261020/process.json)).
  Policy run: `runs/fresh_rl_all16_20261020_main` (создан после проверки награды).
  Ход исполнения: [execution](runs/fresh_rl_all16_20261020/execution.json),
  [supervisor](runs/fresh_rl_all16_20261020/supervisor.json),
  [resource guard](runs/fresh_rl_all16_20261020/stages/main/process.json).
- **Подготовка завершена:** новый reward прошёл fit gate, HUD/progress gate
  и coverage всех16 игр
  ([fit](runs/fresh_rl_all16_20261020/reward/reward_report.json),
  [validation](runs/fresh_rl_all16_20261020/reward/validation_report.json)).
  Собственный [CP0](runs/fresh_rl_all16_20261020_main/checkpoints/rollout_0000/)
  существует; полные начальные оценки опубликованы:
  [canonical](runs/fresh_rl_all16_20261020_main/evaluations/rollout_0000.json),
  [pixel](runs/fresh_rl_all16_20261020_main/evaluations_pixel/rollout_0000.json).
  В каждом режиме каждого оформления256 эпизодов, неизвестных исходов0.
  Canonical: greedy2/256 побед (1/256 с5/5), stochastic16/256, random13/256.
  Pixel: greedy2/256 (0/256 с5/5), stochastic17/256, random13/256.
  Greedy в обоих оформлениях: navigation1/16, reaction1/16,
  остальные14 игр0/16. Stochastic: canonical navigation11/16,
  memory1/16, interaction1/16, ui3/16; pixel navigation12/16,
  memory1/16, interaction1/16, ui3/16; остальные12 игр0/16.
  Это исходная свежая политика, а не результат PPO; отдельная проверка памяти
  противоположными cue и перенос на неизвестное оформление не выполнены.
- **Техническая проверка дэшборда во время PPO:** navigation, validation-уровень
  index7 (в интерфейсе8), canonical, greedy. [Повтор CP0](runs/fresh_rl_all16_20261020_main/dashboard_replays/cc036601096246f09655a853f73017ec/status.json)
  и [повтор CP16](runs/fresh_rl_all16_20261020_main/dashboard_replays/1f96ddd961d14f9ca14ee47f27979cda/status.json)
  завершились таймаутом после80 команд; счёт2 и3 соответственно, побед0/1 в каждом.
  Совпали level/semantic ID, начальный RGB, инструкция, seed и протокол.
  В обоих случаях веса неизменны, optimizer не создавался; результаты видны
  после перезагрузки страницы. CP0 совпал с опубликованными терминальными
  метриками; полной оценки CP16 ещё нет. Это проверка работы интерфейса,
  а не свидетельство общего улучшения. [Первая неудачная попытка](runs/fresh_rl_all16_20261020_main/dashboard_replays/8387af7cffb04274b5be7514bf7a7232/status.json)
  сохранена: несовместимые флаги оформления остановили повтор до игровых действий;
  исправлена конфигурация только ручного воспроизведения.
- **Конечные артефакты:** [summary](runs/fresh_rl_all16_20261020_main/summary.json),
  [CP128](runs/fresh_rl_all16_20261020_main/checkpoints/rollout_0128/manifest.json),
  [canonical CP128](runs/fresh_rl_all16_20261020_main/evaluations/rollout_0128.json),
  [pixel CP128](runs/fresh_rl_all16_20261020_main/evaluations_pixel/rollout_0128.json).
  CP128: ревизия512, 185111 команд; `state.pt` — 631506387 байт,
  SHA-256 `da76c688ed052de41baafe0f2a08f327b390ab830a5b9e2b09c928e0aae831c7`.
  Agent SHA финальных отчётов совпадает с manifest. Максимальный измеренный RSS
  группы процесса — 2185265152 байт (2,04 GiB); минимальное свободное место —
  20278145024 байт. Пауз resource guard не было. Эти показатели описывают ресурсы,
  а не качество игры.
- **Сопоставимость:** CP0 и CP128 проверены на одинаковых уровнях, semantic/runtime
  ID, инструкциях, action seed, оформлении и начальных RGB. Protocol SHA совпадают
  внутри каждого оформления; все четыре отчёта — frozen validation. По256 эпизодов
  на режим и оформление, неизвестных исходов0. Это одни и те же256 уровней
  в двух оформлениях, а не512 независимых уровней.

**Validation: победы собственного CP0 → конечного CP128.**
Greedy — самое вероятное действие; stochastic — выборка из распределения политики.

| Игра | Greedy canonical | Greedy pixel | Stochastic canonical | Stochastic pixel |
| --- | --- | --- | --- | --- |
| navigation | 1/16 → 1/16 | 1/16 → 1/16 | 11/16 → 7/16 | 12/16 → 6/16 |
| reaction | 1/16 → 1/16 | 1/16 → 1/16 | 0/16 → 1/16 | 0/16 → 1/16 |
| mouse | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 |
| memory | 0/16 → 0/16 | 0/16 → 0/16 | 1/16 → 0/16 | 1/16 → 0/16 |
| timing | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 |
| platforming | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 |
| steering | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 |
| camera | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 |
| interaction | 0/16 → 0/16 | 0/16 → 0/16 | 1/16 → 2/16 | 1/16 → 2/16 |
| causality | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 2/16 | 0/16 → 2/16 |
| key_door | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 |
| goal_choice | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 3/16 | 0/16 → 3/16 |
| hazards | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 |
| shooting | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 |
| ui | 0/16 → 0/16 | 0/16 → 0/16 | 3/16 → 0/16 | 3/16 → 0/16 |
| chain | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 | 0/16 → 0/16 |
| **Всего** | **2/256 → 2/256** | **2/256 → 2/256** | **16/256 → 15/256** | **17/256 → 14/256** |

Финальные поражения/таймауты: greedy canonical82/256 и172/256,
greedy pixel82/256 и172/256; stochastic canonical145/256 и96/256,
stochastic pixel145/256 и97/256. Победы со счётом5/5: greedy canonical1/256 → 1/256,
pixel0/256 → 1/256; stochastic0/256 → 0/256 в обоих оформлениях.

**Random-контроль validation отдельно.** Числа одинаковы при CP0 и CP128;
политика `generic_physical_random` не обучалась.

| Игра | Canonical | Pixel |
| --- | --- | --- |
| navigation | 7/16 | 7/16 |
| reaction | 0/16 | 0/16 |
| mouse | 0/16 | 0/16 |
| memory | 1/16 | 1/16 |
| timing | 1/16 | 1/16 |
| platforming | 0/16 | 0/16 |
| steering | 0/16 | 0/16 |
| camera | 0/16 | 0/16 |
| interaction | 1/16 | 1/16 |
| causality | 2/16 | 2/16 |
| key_door | 0/16 | 0/16 |
| goal_choice | 0/16 | 0/16 |
| hazards | 0/16 | 0/16 |
| shooting | 0/16 | 0/16 |
| ui | 1/16 | 1/16 |
| chain | 0/16 | 0/16 |
| **Всего** | **13/256** | **13/256** |

В каждом оформлении random: поражения150/256, таймауты93/256,
неизвестные исходы0/256, победы5/5 — 0/256.

**TRAIN отдельно, за весь запуск.** Источник —
[training.jsonl](runs/fresh_rl_all16_20261020_main/training.jsonl), `episode_records`.
Это практика меняющейся sampled-политики, а не frozen-оценка финальной модели.
Все4096 эпизодов имеют pixel-оформление с варьируемыми темами; canonical TRAIN
в этом запуске не измерялся.

| Игра | Победы TRAIN |
| --- | --- |
| navigation | 149/256 |
| reaction | 2/256 |
| mouse | 0/256 |
| memory | 20/256 |
| timing | 1/256 |
| platforming | 0/256 |
| steering | 0/256 |
| camera | 0/256 |
| interaction | 12/256 |
| causality | 9/256 |
| key_door | 2/256 |
| goal_choice | 13/256 |
| hazards | 9/256 |
| shooting | 0/256 |
| ui | 27/256 |
| chain | 1/256 |
| **Всего** | **245/4096** |

TRAIN: поражения2404/4096, без победы и поражения1447/4096,
неизвестные исходы0/4096, победы5/5 — 0/4096.
В пяти играх (mouse, platforming, steering, camera, shooting) за всю практику
нет ни одной успешной траектории. Равное число эпизодов и общий minibatch сами
по себе не обеспечили успешное исследование этих игр.

**Итог и ограничения:** пилотный порог не выполнен. Stochastic прирост —
−1/256 canonical и−3/256 pixel вместо требуемых+16/256. В каждом оформлении
улучшились4 игры (reaction, interaction, causality, goal_choice), но потеряны
победы navigation, memory и ui. Greedy остался2/256. Освоение всех16 игр и общий
рост относительно собственного CP0 не показаны. Это один seed обучения и одна
stochastic попытка на уровень; небольшие локальные изменения не доказывают
устойчивый эффект. Порог нельзя задним числом менять под результат.

Test, независимые проверки памяти с противоположными cue и перенос на неизвестное
оформление не проводились. Совпадений TRAIN level ID с validation не найдено;
TRAIN-журнал не сохраняет semantic/runtime ID, поэтому независимая проверка
фактического семантического разделения по одному журналу ограничена. Генератор
зафиксированного банка разделяет семантические уровни до сбора данных.
Запуск одновременно меняет несколько условий относительно прошлых опытов;
причинный эффект общего minibatch отдельно не установлен. Новые обучения,
повторы и продолжения после этого результата не запускались.

#### Диагностика отсутствия улучшения, 20.10 после завершения

Это анализ того же запуска, без новых игровых эпизодов и optimizer updates.
[Машинный аудит](runs/fresh_rl_all16_20261020_main/diagnostics/learning_signal_audit.json)
содержит хеши входных файлов, счётчики по играм и пересчёт GAE;
[служебный скрипт](.cache/audit_fresh_learning_signal.py) читает существующие
`training.jsonl` и `trajectories/rollout_0001/*.npz`. Проверены закреплённые исходники;
семь относящихся к PPO/политике/композиции файлов текущего дерева совпадают
с `manifest.source_pins`. 210 профильных тестов microbatch, Torch/MLX,
sequence training, физических действий и reward composition прошли.

**Подтверждено по данным и коду:**

1. **Обновления не отключены.** 512 шагов optimizer выполнены, trainable actor,
   temporal, resampler и проекция инструкции изменились. Средняя записанная
   preclip gradient norm в первых/последних 16 циклах — 1,064/1,073.
   KL-stop не было; mean epoch-KL 0,000085 → 0,000213, entropy 8,216 → 8,585.
   Ошибки с дополнительным делением градиента microbatch 4 или сбросом памяти
   на границе сегмента в проверенном пути не обнаружены. Эти проверки не доказывают
   полезность обучившегося поведения.
2. **Terminal reward распознаётся согласованно.** Последний learned event совпал
   с отдельным RGB outcome во всех 4096 эпизодах: 245 success, 2404 failure,
   1447 play/timeout. Ложных success 0/3851, пропущенных 0/245; uncertain и invalid
   transitions отсутствуют. Это не независимая проверка каждого промежуточного кадра.
3. **Итоговая цель почти всегда не даёт различающего сигнала.** При failure/time/input=0,
   `F=.2*(.99*Phi_next-Phi_current)` и нулевом terminal potential все 3851 nonwin
   имеют discounted return в пределах −2,50e−9…6,67e−9 (погрешность float32).
   Победы дают 0,452…0,990. Potential shaping может помогать локальным GAE/critic,
   но не ранжирует незавершённые попытки по полному return. Это свойство выбранной
   цели, а не обнаруженная ошибка арифметики
   ([закреплённая композиция](runs/fresh_rl_all16_20261020/source_snapshot/src/carryrl/rewards/composition.py)).
4. **Исследование действий не даёт нужных сочетаний и удержаний.** В platforming
   лишь 39/20354 команд содержат фронт Space с движением вправо, максимальное
   непрерывное движение вправо — 4 команды; максимальный перевес вправо за эпизод — 12,
   при расстоянии до флага примерно 36–42 перемещения. В
   [decoder](runs/fresh_rl_all16_20261020/source_snapshot/src/carryrl/agent/decoder.py)
   постоянный EOS bonus +3,5 после первой клавиши делает добавление Space маловероятным.
   Начальный mouse prior даёт ненулевое движение по хотя бы одной оси примерно 82,5%;
   фактически mouse 85,8%, camera 86,5%. При полной сложности первый промах завершает
   mouse; final stochastic mouse проиграл 16/16 с 0 попаданий и средней длиной 4,625.
   Жёсткие конфликты направлений редки (примерно 1,3–1,7% команд этих пяти игр).
5. **Случайный critic загрязняет первый обучающий сигнал.** `value_head` создаётся
   обычной случайной Linear. В первом цикле на 389 шагах эпизодов с полностью нулевой
   наградой raw advantage лежит в диапазоне −0,111…0,873. Пересчёт тех же траекторий
   с V=0 даёт ровно 0. Средний normalized advantage mouse меняется +1,564 → −0,105,
   navigation +0,017 → +1,278. Это проверка начального baseline при неизменных
   данных, без обновления политики; роста побед она не измеряет.

| Игра без TRAIN-побед | Победы | Есть положительный progress | Вся награда равна 0 |
| --- | --- | --- | --- |
| mouse | 0/256 | 9/256 | 247/256 |
| platforming | 0/256 | 17/256 | 239/256 |
| steering | 0/256 | 117/256 | 139/256 |
| camera | 0/256 | 84/256 | 172/256 |
| shooting | 0/256 | 97/256 | 159/256 |

В этих пяти играх 1280 эпизодов и 62680 команд не дали ни одной победы.
Счёт 4 в camera не означает завершённую задачу: сохранённый максимум приближения
и выравнивания не гарантирует их одновременного выполнения. В shooting score ≤2
соответствует приближению/наведению; ни одного попадания в этих TRAIN-данных нет.

**Что пока остаётся гипотезой.** Minibatch 16 при 32 эпизодах даёт один эпизод
каждой игры на update, и шаг однокадрового проигрыша получает до 80 раз больший
коэффициент, чем шаг 80-кадровой попытки. Это выбор нормировки и возможный источник
дисперсии, не доказанная причина итогового результата. Доминирование critic/entropy
в суммарном градиенте не измерено. Epoch-метрики повторно взвешиваются числом
transitions, поэтому не являются точным средним по updates.

Отсутствие побед не объясняет весь результат: navigation имел 149/256 TRAIN-побед,
но validation stochastic снизился 11/16 → 7/16 и 12/16 → 6/16. Один seed действий
не отделяет изменение навыка от вариативности траекторий. Frozen DINO/64 токена
и RMT-память требуют отдельных проверок чувствительности к позиции/cue;
их поломка текущим аудитом не установлена. Cue memory заканчивается до
границы 64, а выбор начинается на 66–69-м шаге: задача действительно требует
переноса информации через сжатую память.

Ранее простые изменения LR, λ и стоимости действий уже не дали подтверждённого
роста (карточки выше). Нельзя считать их готовым решением. Также нельзя просто
поставить difficulty=0: прежние metadata/neutral-аудиты обнаружили совпадения
облегчённых уровней между splits и пассивные победы. Предложенные изменения и
условия следующей проверки находятся в [SPEC.md](SPEC.md#диагностика-и-предложение-после-fresh-all16).

<a id="storage"></a>
## Технические замеры хранения

Эти проверки помогают сохранить и воспроизвести опыты. Они не измеряют качество игры.

| Проверка | Условия и результат | Артефакт |
| --- | --- | --- |
| Повторы тензоров в пяти checkpoint, 16 октября | 3 156 152 640 байт payload, 919 346 538 уникальных байт; 2 236 806 102 байт точных повторов. Модель не загружалась. Это обоснование хранения общих блоков, а не измерение размера всех запусков | [Измерение](.cache/checkpoint_tensor_duplicates.json) |
| Реальное восстановление архивного checkpoint, 16 октября | `instruction_v3_integration` rollout 1 восстановлен; проверен SHA-256, 631 341 535 байт. Изолированный Python без site packages, без загрузки модели/optimizer и без обучения | [Receipt восстановления](.cache/real_checkpoint_recovery_20261016.json) |
| Стоимость поиска защищённых путей | Один замер до/после: 1.779 → 1.360 с; совпали все 16 защищённых путей. Не статистический benchmark и не ускорение обучения | [Замер](.cache/storage_pin_benchmark.json) |

Хранилища [`.checkpoint_store`](runs/.checkpoint_store/) и
[`.storage_recovery`](runs/.storage_recovery/) — служебные данные, не дополнительные
обучающие серии. Упакованное [CarryRLNavigation.app](runs/CarryRLNavigation.app/)
также не является отдельным экспериментом с моделью.

<a id="maintenance"></a>
## Полнота, сохранность и добавление следующих опытов

### Когда обновлять реестр

Новые опыты добавляются сюда по мере работы, включая подготовленные, неудачные
и прерванные. Обязательный порядок ведения закреплён в
[AGENTS.md](AGENTS.md#ведение-экспериментов).

До запуска заполнить замысел, условия, бюджет и критерий по шаблону ниже;
результат обозначить «ещё не измерен». После запуска обновить статус, после
завершения или остановки — фактические метрики, итог, вывод, ограничения и ссылки
на подтверждающие артефакты. Не оставлять завершённую попытку со статусом «идёт».
Существенные новые выводы отражать также в краткой сводке этого файла.
Запись о подготовке или возможном следующем опыте сама по себе не разрешает запуск.

### Граница полноты

Реестр охватывает найденные локальные серии и связанные диагностики на момент
составления. Повторные подготовки и launch-каталоги объединены с соответствующими
опытами; они не считаются независимыми обучениями. Вложенные ветви, эпохи и
проверки входят в карточку родительской серии.

Проверка охвата при составлении 20 октября 2026 года: в тексте представлены все **175 видимых каталогов верхнего уровня
`runs/`**, а также корневые JSON-отчёты, значимые локальные аудиты и известные удалённые
попытки. Это число каталогов, а не число независимых экспериментов.

Не включены как отдельные ML-эксперименты: обычные unit-тесты, скриншоты интерфейса,
копии TensorBoard, кеши замороженных признаков и служебные архивы. Значимые
проверки скорости, исполнимости и хранения перечислены отдельно от качества модели.
Если первичные файлы утрачены или сохранился только журнал удаления, это указано
в карточке: такие результаты нельзя независимо пересчитать по отсутствующим данным.
Неизвестные или несохранившиеся опыты без следов в доступных источниках восстановить
по этому реестру невозможно.

Для архивных весов проверять `storage_manifest.json` и ссылку на общее хранилище:
отсутствие raw `state.pt` само по себе не означает потерю checkpoint. Не удалять
каталоги исходя из статуса «заменён» или «не принят»: они могут быть нужны для
проверки происхождения и восстановления результатов.

### Шаблон следующей записи

```markdown
### ID · дата · короткое название

- Вопрос: какую гипотезу проверяем.
- Статус: подготовлен / идёт / завершён / прерван / заменён.
- Модель: исходный run и checkpoint, поколение, что заморожено.
- Условия: данные и split, версия игр/счёта, оформление, инструкции,
  seed, метод и бюджет; единственное изменение относительно контроля.
- Критерий: заранее заданное условие успеха.
- Факт выполнения: достигнутые эпохи/циклы/updates, итоговый checkpoint,
  завершение процесса или причина остановки.
- Результат: TRAIN отдельно; validation/test отдельно; победы n/N,
  5/5, каждая игра, stochastic/greedy, память и оформление отдельно.
- Вывод: принят ли вариант и что именно подтверждено.
- Ограничения: несопоставимость, повторные уровни, один seed, недостающие данные.
- Артефакты: manifest/config, итоговый отчёт, checkpoint/storage manifest,
  диагностики и подтверждение завершения.
```

При продолжении прерванного опыта добавлять запись о resume, исходном checkpoint
и восстановлении RNG/optimizer; не стирать первоначальную попытку. При исправлении
ошибочного протокола сохранять старый результат с пометкой и ссылкой на замену.


<a id="text-reward-binding-fix"></a>
## Исправление привязки текстового оценщика21.10

### `text_reward_binding_fix_20261021` · проверка локального контекста

**Итог текущей ветви:** `v6_grounded_budget` завершил40эпох, exit0, но не
допущен: положительный знак, held/press и aliases не прошли, runtime3/32.
Модельные файлы и завершение процессов проверены; sealed и PPO не запускались.
Источники и неудачные попытки сохранены. Ниже — исходная регистрация и последовательные
изменения этого опыта; текущие условия —
[в исполнении](runs/text_reward_binding_fix_20261021/execution_v6_grounded_budget.json),
фактический процесс —
[в состоянии](runs/text_reward_binding_fix_20261021/training_v6_grounded_budget_state.json).

- Авторизация: пользователь запросил выяснить причину провала и исправить её
  после остановленного [fresh_all16_v2](#fresh-all16-v2). Старый результат
  не переписывается; это новый связанный опыт, без запуска игровой политики.
- Проверенная проблема: head ошибается уже на TRAIN sparse-инструкциях;
  высокий raw recall aliases скрывает множество ложных объединений. В v3
  синтетическом корпусе режим коррелирует с номером operation, а грамматика
  меняется вместе со split. Ошибок градиента, порядка candidates и truncation
  не обнаружено. Исходно причинный эффект не был измерен; matched-контроль ниже проверяет локальную attention.
- Изменение: лексическая привязка candidate к токенам текста при обучаемых
  status/mode/pair; новый v4-корпус со случайными режимами отдельных операций
  и естественными формулировками; поправка TRAIN class weights перед
  температурой. Замороженный E5 сохраняется, head/optimizer создаются заново.
- Данные: v4 seed20261022,4000 смысловых групп, split до парафразирования.
  Старые TRAIN-тексты — только ограниченная техническая проверка способности
  head выучить привязку; её веса не переходят в конечную модель.
- Контроль: фактические предсказания старого v2 на фиксированных nonsealed
  TRAIN/dev; capacity probe, затем новые development/calibration и все32
  фактических RU/EN инструкции. Runtime — проверка совместимости, не
  независимая оценка. Порог0.99 и существующие precision/coverage gates
  сохраняются. Sealed text test — только один раз после заморозки и
  успешных предварительных проверок; игровой test закрыт.
- Бюджет: до3ч на исправление и проверки в остатке ранее разрешённых24ч;
  исходный first_compute/deadline не обнуляется. Capacity≤2000updates/600с,
  основной head≤40эпох; резерв ресурсов и запрет PPO при отказе сохраняются.
- Критерий: существующий допуск по обоим знакам, held/press/aliases, ноль
  критических ошибок и runtime32/32; качество игры этим не измеряется.
- Артефакты: [протокол](runs/text_reward_binding_fix_20261021/protocol.json).
- Capacity-контроль завершён с exit0 за52.37с,1000updates:26 старых TRAIN-групп,
  104 текста. Status3744/3744 и mode3744/3744, raw same150/152,
  precision150/150 после поправки TRAIN-prior. Это способность выучить малую
  TRAIN-выборку, не качество на новых инструкциях; веса не переходят в fit.
- Новый fit: grounded_balanced_v3, seed20261022, новый head/optimizer,
  до40эпох на новом v4-корпусе; E5 заморожен. Source snapshot117файлов,
  максимум7200с внутри общего срока, лимит процесса6GiB RSS, прогноз диска
  4.39GiB и свободный остаток минимум8GiB. Calibration и runtime пока не оценены.
- Проверки:90 целевых Python passed; независимый ML-review нашёл и помог
  закрыть shared-core loader guard, единый lexical anchor для positive/negative
  axis-упоминаний и одинаковую unique-offdiagonal популяцию pair calibration.
- Артефакты: [диагноз исходной головы](runs/text_reward_binding_fix_20261021/diagnosis_old_head.json),
  [capacity](runs/text_reward_binding_fix_20261021/capacity_result.json),
  [версии источников](runs/text_reward_binding_fix_20261021/source_hashes.json),
  [процесс обучения](runs/text_reward_binding_fix_20261021/training_state.json),
  [диагностические пары](runs/text_reward_binding_fix_20261021/counterfactual_development_cases.json).
- V3 full-data попытка контролируемо остановлена для диагностики:13 полных
  эпох +122updates, всего4022updates за635.93с, SIGTERM и фактический exit1.
  Partial checkpoint сохранён, живых owned-процессов после остановки нет.
  Максимум40 эпох не был выполнен; calibration/runtime этой головы не проводились.
  На фиксированной dev-выборке64группы до коррекции mode: held227/720,
  press710/768; held→press472, held→unknown21. После коррекции held218/720,
  press694/768. Ошибка held↔press доминирует до prior correction.
- Изоляция: strict mention-attention читает лишь название клавиши; локальное
  слово режима туда не входит, global branch знает положение лишь неявно
  через frozen E5. TRAIN-only измерение1900bindings: timing-token≤5,
  operation-ID≤21 токена от имени. Это основание проверить локальный контекст,
  а не доказательство его качества.
- Local-v4: единственное изменение относительно V3 — доступ той же локальной
  attention к±24 токенам и фиксированное расстояние для4голов. Число и начальные
  значения обучаемых параметров совпадают, initial SHA для seed20261022
  `5f5a68109fc5bbf9e5b3f0e4ba664d64e69ad467ae9e548f21eb37b32ddc1f25`.
  E5/data/seed/loss/priors/пороги сохраняются; feature-cache повторно читается
  с проверкой SHA, без перезаписи. Отдельный диагностический снимок на4022updates
  позволяет сравнить одинаковый объём обучения. Старые веса не импортируются.
- Полный набор до local-v4:3463 passed за475.38с, один прежний Gym warning.
  Целевые проверки local-v4:84 passed; полный повтор выполняется.
- Дополнительные артефакты: [остановкаV3](runs/text_reward_binding_fix_20261021/training_state.json),
  [сохранённые partial-веса](runs/text_reward_binding_fix_20261021/fit_v3/partial_head.pt),
  [raw/corrected диагноз](runs/text_reward_binding_fix_20261021/diagnosis_partial_v3.json),
  [локальные расстояния](runs/text_reward_binding_fix_20261021/local_binding_distance.json),
  [условия local-v4](runs/text_reward_binding_fix_20261021/execution_local_v4.json),
  [его процесс](runs/text_reward_binding_fix_20261021/training_local_state.json).
- Статус: выполняется local-v4; окончательный допуск и перенос ещё не измерены.
  Sealed text test и игровой test закрыты.

- Matched-контроль4022updates: тот же seed, начальные параметры, v4-данные и
  loss. Raw mode на фиксированном dev64группы:937/1488 →1363/1488;
  TRAIN832/832 held и784/784 press у local-v4. Это результат проверки
  режимов, а не полный допуск. Corrected alias same:221/564, precision221/380;
  перенос на произвольные инструкции не подтверждён.
- Local-v4 остановлен после29 полных эпох для исправления корпуса:1156.71с,
  SIGTERM, фактический exit1 и сохранённый partial. Условие «Hold W» давало
  supported0.9992 с footer «disabled», но0.0016 при равнозначном
  «unsupported»; ID операции также необоснованно повышал уверенность.
  Обнаружены опора на формулировку остаточных команд и служебный ID вместо
  устойчивой привязки к названной клавише. Calibration/runtime этой попытки
  не открывались. Артефакты:
  [matched-метрики](runs/text_reward_binding_fix_20261021/diagnosis_matched_local_v4.json),
  [footer-пары](runs/text_reward_binding_fix_20261021/footer_ablation.json),
  [ID-пары](runs/text_reward_binding_fix_20261021/operation_marker_ablation.json).
- Вариант v5 меняет только формулировки при тех же semantic specs, labels и
  splits: режимы и отрицания представлены разными общими словами во всех
  split, половина16000текстов без ID. E5-контекст максимум327/512.
  Пороги и gates сохранены. Веса создаются заново; лимит80эпох выбран до fit
  в прежнем общем бюджете. Ни веса capacity, ни partial не переносятся.
- Первая v5-попытка завершилась с exit1 до первой эпохи за158.47с:
  холодный первый шаг ошибочно экстраполировался на весь fit. Полный кеш E5
  сохранён1.480GB и проверяется перед read-only reuse. Новый probe использует
  медиану8 прогретых шагов, учитывает холодный шаг и сохраняет hard deadline.
- Следующая техническая попытка контролируемо остановлена за74.93с, exit1:
  независимый review обнаружил, что dev selection повторно применял class
  weights после log-prior correction. Исправлено: выбор по raw weighted devCE,
  corrected logits — для runtime и temperature. Legacy v1/v2 не меняется.
  Regression-тест отдельно отличает корректный criterion от double-weighted.
- Текущий fit: frozen snapshot118файлов, fresh local-v4 head/optimizer,
  v5 wording, seed20261022, до80эпох, read-only reuse E5-признаков,
  raw weighted dev selection. Артефакты:
  [условия](runs/text_reward_binding_fix_20261021/execution_wording_v5_objective.json),
  [исходники](runs/text_reward_binding_fix_20261021/source_v5_objective_hashes.json),
  [процесс](runs/text_reward_binding_fix_20261021/training_wording_objective_state.json),
  [эпохи](runs/text_reward_binding_fix_20261021/fit_wording_v5_objective/fit_progress.json).
  Полного calibration/runtime результата ещё нет. Sealed text и игровой test
  остаются закрытыми. Полный Python до последних двух правок selection/probe
  выполняется; отдельные проверки этих правок12passed.

- Подготовлен matched-контроль pair geometry (до запуска): тот же v5-корпус,
  E5-cache, seed и objective; новый pair MLP получает8 симметричных
  лексических позиционных признаков (+512 весов, нулевая инициализация).
  Старые параметры и initial predictions совпадают; координаты не содержат
  статусов/режимов/алиасов. План40эпох, отдельный снимок4022updates,
  сравнение с current local-v4 при том же объёме. Пороги и финальный gate
  не меняются. Перед запуском текущий owned-процесс останавливается,
  checkpoint4022 и exit проверяются. Это development-проверка уже найденной
  слабости pair readout, не новый независимый statistical test.
  [Условия](runs/text_reward_binding_fix_20261021/execution_pair_geometry_v5.json).
  94 целевых теста прошли, включая CPU/MPS, симметрию, неизменность прежних
  вариантов и отсутствие готовых ответов в geometry. CPU B32/L327:160мс,
  RSS-прирост77MiB в отдельном ограниченном benchmark.

- V5-wording matched4022 завершён как диагностическая остановка:13 полных
  эпох и последующие updates,658.81с, SIGTERM/exit1, сохранены4022 иpartial,
  процесс проверен отдельно. На тех же dev64группы raw held718/720,
  press756/768, итого1474/1488 против1363/1488 у v4-wording. Это причинный
  контроль изменения формулировок при тех же labels/seed/общих весах/объёме;
  dev-selection на обновления4022 не влияет. Alias corrected recall141/564,
  precision141/205; TRAIN198/580, precision198/243. Полный gate не проводился.
- Footer-проверка нового корпуса:3 равнозначных complete формулировки теперь
  сохраняют supported/held для «Hold W»; partial «про остальные неизвестно»
  всё ещё вызывает ошибочную неопределённость. В partial-сценариях генератора
  всегда присутствуют1..3 uncertain и1..3 forbidden команды; минимальной
  частичной инструкции без этих добавлений нет. Это сохраняющийся пробел
  покрытия, а не доказательство прохождения корпуса в целом.
  [Matched-метрики](runs/text_reward_binding_fix_20261021/diagnosis_matched_wording_v5.json),
  [footer](runs/text_reward_binding_fix_20261021/footer_ablation_wording_v5.json),
  [завершение](runs/text_reward_binding_fix_20261021/training_wording_objective_state.json).
- Pair-geometry fit запущен после подтверждённой остановки контроля. Probe9
  шагов: median steady0.2424с, прогноз3912.5с; максимум7118с с резервом до
  прежнего repair deadline. Runtime/public32 исходной версии сохранены.
  [Процесс](runs/text_reward_binding_fix_20261021/training_pair_geometry_state.json),
  [скорость](runs/text_reward_binding_fix_20261021/fit_pair_geometry_v5/fit_throughput_probe.json).
  Независимый ML-review перебором10368 lexical пар подтвердил формулы,
  симметрию и отсутствие labels в geometry; новый fit ещё не оценён.
- Полный Python после probe/selection:3488passed за481.83с, один прежний
  Gym warning. После geometry выполняется отдельный полный повтор.

- Аудит TRAIN v5:9600текстов/2400групп;3040positivepartial всегда имеют
  дополнительные uncertain+forbidden, минимальных таких инструкций0.
  Generic cursor/yaw/pitch, игровые цели/distractors и release-between/click
  также отсутствуют. Это пробел композиционного покрытия; наличие1967TRAIN
  групп сalias-pairs при низкойTRAINrecall остаётся отдельной проблемой readout.
- Runtime-контракт уточнён ДО нового v6-fit:26/32старых текстов требовали
  keyboardheld в разметке, не сообщая его явно. Обновлены27текстов: явное
  удержание нужного направления, release уcausality, физическое движение
  мыши. Набор механически доступных команд и разметка не менялись; UI
  conditional остаётсяunknown. Прежние32 и их оценки сохраняются как старый
  протокол, качество на новых32 не будет выдано за качество на старых.
  [Тексты до/после](runs/text_reward_binding_fix_20261021/public_instruction_clarification.json).
- Новый regression по всем32публичным текстам выявил пропускE в «нажимай и
  отпускайE»; исправлены русские глагольные формы. Qualified «arrow keys
  left/right» и «стрелками влево/вправо» больше не создают ложныеup/down
  lexicalmentions. Это распознавание написаний, не правилstatus/mode/aliases.
  84целевых теста прошли. Frozen geometry-v5 source не меняется.
- Подготавливается отдельный v6-корпус: независимые sparsepartial и natural
  bindings, genericmouse, goal/distractor, удалённый явныйmode; фактические
  32инструкции не копируются вTRAIN. Для axes доступность не требуетheld/press,
  как и в существующемruntime; KEY/button неизвестныйmode остаётсяunknown.
  Fit этого корпуса ещё не начат, sealed закрыт.
- Полный Python до изменений публичного текста/lexical inflections/v6:
  3495passed за484.58с, один прежнийGymwarning; geometry94целевых прошли.

- Geometry-v5 matched4022 сохранён и fit остановлен:1259.81с,13 полных эпох
  плюсupdates доSIGTERM, exit1, partial и matched checkpoint сохранены.
  На том же v5 dev64группы corrected alias recall141/564 →247/564,
  precision141/205 →247/351; TRAIN198/580 →293/580. Raw mode1474/1488 →1473/1488.
  Дополнительные512позиционных весов помогают pair-обучению при одинаковом
  числе updates, но полного допуска не дают. Анализ выполнен по frozen
  source каждого опыта, включая старый grounding.
  [Matched-метрики](runs/text_reward_binding_fix_20261021/diagnosis_matched_geometry_v5.json),
  [остановка](runs/text_reward_binding_fix_20261021/training_pair_geometry_state.json).
- Независимый correctness-review не нашёл нового дефекта head/load/export,
  но выявил прежнюю слабость reuse: перестановка двух span одинаковой
  токенной длины могла пройти проверку cache. Исправлена привязка offsets
  к стабильному порядку точных текстов. Регрессия14passed; read-only audit
  подтвердил корректность реальных v4/v5-кешей — это не установленная причина
  прежнего провала.
  [Аудит](runs/text_reward_binding_fix_20261021/cache_layout_audit.json).
- Финальный новый fit зарегистрирован ДО запуска: corpus v6,
  recordsSHA `a19f2660d17c525d848d037265aaf217c848efa5ae622f089a93d152d3acfd2f`,
  4000групп/16000текстов, maxE5=351/512; calibration eligible360supported,
  340unsupported. Исключены6726 уже использованных effective mappings из
  ВСЕХsplits; null-case сохраняет именованный набор, axes mode нормализован
  к runtimeunknown, одинаковые effective axes не расходятся междуsplits.
  Дефектный предварительный корпус сохранён отдельно и не использовался вfit.
- V6 добавляет sparsepartial без дополнительных блоков, независимые цели,
  genericmouse4раздельныхsigns, scopedaliases, remote явный timing, гарантированные
  фазы и неизвестные условия. CanonicalWASD/arrow иA-D пары сохранены с
  дополнительными случайными controls; фактические32текста вTRAIN не копируются.
  Две найденные review неоднозначности исправлены;62tests и2проверки счётчиков
  прошли, прежние v4/v5 outputs сохранены.
- Новый head/optimizer сseed20261022, `pair_geometry_balanced_v5`, максимум40эпох,
  32batch,4threads, исходныйE5frozen, новый дисковый featurecache≤4GiB.
  Пороги0.99 и все gates неизменны; старыеpartial не импортируются.
  Источники119файлов заморожены, свободно43.70GiB при требуемом остатке8GiB;
  stage≤7200с иRSS≤6GiB. Один час исходного резерва перераспределён в
  allocation исправления (до4ч); original first_compute и24ч deadline
  не меняются. Sealedtext открывается один раз только после полного допуска
  и заморозки, игровойtest закрыт. PPO не запускается в этой задаче.
- Артефакты перед fit:
  [dataset](runs/text_reward_binding_fix_20261021/text_data_v6/manifest.json),
  [reserveddevelopment](runs/text_reward_binding_fix_20261021/development_mapping_reservations.json),
  [новый протокол исполнения](runs/text_reward_binding_fix_20261021/execution_v6.json),
  [исходники](runs/text_reward_binding_fix_20261021/source_v6_hashes.json).

- Финальный v6-fit остановлен для исправления технического пути текста:
 349.56с,exit1 послеSIGTERM, partial сохранён. До этого E5featurescomplete
  за140.47с,1,396,521,984байта, исходный encoderSHA совпал, sealed не кодировался.
  TRAIN-only audit нового компактного синтаксиса обнаружил9525/69368
  supportedbindings без lexicalanchor: вторыеbareбуквы/цифры в списке,
  «UseR», русские инфлекции/двоеточия и часть genericmouse форм.
  Это пропуск в дополнительно введённой привязке; globalcontext ещё доступен,
  но заявленный локальный путь для этихкоманд отсутствует. Длительныйfit
  с известным дефектом не продолжается. Корпус и labels не меняются,
  исправляется только lexical locator; следующаяголова начнёт с нуля,
  полный проверенный E5-cache будет использован read-only.
  [Аудит](runs/text_reward_binding_fix_20261021/v6_lexical_transmission_audit.json),
  [фактическая остановка](runs/text_reward_binding_fix_20261021/training_v6_state.json).

- До запуска зарегистрирована ветвь `v6_grounded` после разрешённой пользователем
  паузы на отсутствие связи. Исправлены lexical coordination/arrow/state границы;
  автоматическая проверка всех13600 незакрытых текстов:108168/108168
  именованных inputs переданы, sealed не используется. TRAIN-проверка отдельно
  включает явные unknown/negative названия; полный кеш E5 аутентифицирован.
  Новые head/optimizer, seed20261022, максимум40эпох; partial веса не импортируются.
  Корпус v6 неизменен, E5-cache только read-only; порог0.99 и gates прежние.
  Исходный24ч deadline и4ч allocation не обнуляются. Запуск ограничен7200с,
  оставшимся общим временем, RSS6GiB и свободным остатком8GiB.
  Оценка calibration/runtime32 входит в запуск; sealed и PPO автоматически не запускаются.
  [Исполнение](runs/text_reward_binding_fix_20261021/execution_v6_grounded.json),
  [предпроверка](runs/text_reward_binding_fix_20261021/v6_grounded_launch_preflight_final.json),
  [проверки кода](runs/text_reward_binding_fix_20261021/v6_grounded_checks.json),
  [source](runs/text_reward_binding_fix_20261021/source_v6_grounded_hashes.json).

- `v6_grounded` фактически запущен 2026-10-21T10:42:33.025278+03:00; supervisorPID61473 и actorPID61514 сверены с полной идентичностью процессов.
  Источники131файл заморожены, разрешённый общий бюджет не обнулён.
  Полный Python3567passed (один прежнийGymwarning); после последних lexical
  edge-изменений отдельно224passed, Ruff иdiff-check проходят.
  Предварительная проверка остановила пропуски числовых клавиш перед «для»;
  исправленный полный preflight прошёл. Этот отрицательный служебный лог сохранён.
  Действующий процесс работает offline; результат обучения/допуск пока не получены.
  [Живое состояние](runs/text_reward_binding_fix_20261021/training_v6_grounded_state.json),
  [лог](runs/text_reward_binding_fix_20261021/fit_v6_grounded_stdout.log).

- `v6_grounded` технически остановился после9updates до полной эпохи:41.78с,
  actualexit1, actor и supervisor завершены, partial/failure сохранены.
  Входной preflight прошёл; медиана тёплого шага0.2331с, оценкаfit3762.59с.
  Проба требует ещё1.5× запаса: этот прогноз немного превышал остаток
  внутреннего4ч allocation после сетевой паузы. Качество модели не оценено.
- ДО нового запуска зарегистрирован `v6_grounded_budget`: те же131файл
  обучающего кода/данных и40эпох, новая голова/optimizer с прежнимseed.
  Меняется только supervisor/выделение времени: ещё1800с из разрешённых24ч
  (allocation4.5ч), first_compute и общий deadline прежние. Запас1.5,
  порог0.99, все admission gates и ресурсные ограничения сохранены.
  Никакие9-update веса не импортируются.
  [Исполнение](runs/text_reward_binding_fix_20261021/execution_v6_grounded_budget.json),
  [остановленная попытка](runs/text_reward_binding_fix_20261021/training_v6_grounded_state.json).

- `v6_grounded_budget` фактически запущен 2026-10-21T10:44:55.876159+03:00; actorPID62365 и supervisorPID62347 сверены живыми.
  Повторный preflight13600текстов/108168mentions прошёл, проба9шагов
  принята:median0.2330с, прогноз3761.23с до дополнительного1.5× запаса.
  После пробы процесс продолжает вычисления; пиковыйRSS при проверке
  около1.8GiB. Веса/optimizer свежие, E5-cache толькоread-only.
  Результата качества пока нет; cal/runtime32 следуют послеfit.
  [Подтверждение запуска](runs/text_reward_binding_fix_20261021/v6_grounded_budget_launch_verification.json),
  [живое состояние](runs/text_reward_binding_fix_20261021/training_v6_grounded_budget_state.json),
  [лог](runs/text_reward_binding_fix_20261021/fit_v6_grounded_budget_stdout.log).

- Итог `v6_grounded_budget`: завершён21.10 в11:47:51МСК, elapsed3776.05с,
  40эпох/12000updates, actualexit0. Actor/supervisor завершены, processgroup
  пуст;131sourcehash и9файлов модели проверены. Freshinit/emptyoptimizer
  подтвержденыmanifest; исходный E5SHA не изменился. Выбранаdev-эпоха33.
- Calibration:1600текстов/400независимых semanticgroups. Применимые команды:
  уверенно5129/11540(44.45%), правильно5126/5129; запрещённые:
  уверенно29178/29352(99.41%), правильно29178/29178. Отрицательный знак
  прошёл gate; положительный не прошёл, scenario lower95=0.9095.
  Held1896/4540уверенных(41.76%), правильно1895/1896; press1823/4688
  (38.89%), правильно1821/1823; aliases577/3060(18.86%), правильно575/577,
  отдельные falsemerge4/falsesplit2. Всего9critical errors, admissionfalse.
- Runtime:полностью3/32 инструкции (mouseEN,timingRU,timingEN). Покомпонентно
  supported77/222, unsupported835/914, held51/176, press7/22, aliases16/88.
  Зафиксированы8ошибочных запретов движения мыши (mouseRU/shootingRU)
  и7ошибокheld/press; прочие отказы включают недостаток уверенных ответов.
  Это публичная совместимость, не независимая сертификация.
- Dev-динамика обучаема: на выбраннойэпохе33 argmax supported17514/17516,
  held6716/6776, press7213/7256, same4101/4540. При этом строгие уверенные
  решения и перенос на игровые инструкции недостаточны; низкийloss/argmax
  не заменяют готовность награды. Последняя эпоха40 devselectionNLL0.01349
  против0.00784 у33; веса40 отдельно не сохранены, её метрики сохранены.
  Старые и новые qualityчисла не являются matched-сравнением из-за нового
  корпуса/публичных инструкций. Суммарные99.99% точности принятых ответов
  скрывают низкий положительный охват и не являются допуском.
- Sealed text test и игровойtest закрыты, policy/PPO не запускались,
  новых повторов после этого завершения нет. Следующий предмет диагностики:
  confidence/alias-rejection и перенос между игровыми формулировками,
  особенно RUмышь и область действия held/press; это ещё не исправленная причина.
  [Проверенный итог](runs/text_reward_binding_fix_20261021/result.json),
  [калибровка](runs/text_reward_binding_fix_20261021/reward_text_v6_grounded_budget/validation.json),
  [runtime32](runs/text_reward_binding_fix_20261021/reward_text_v6_grounded_budget/runtime_validation.json).

<a id="text-reward-admission-fix"></a>
## Диагностика отказа допуска текстовой награды 21.10

### `text_reward_admission_fix_20261021`

- **Запрос и контроль.** Пользователь попросил найти и исправить причину отказа
  перед следующим обучением. Контроль — неизменная модель `v6_grounded_budget`,
  выбранная эпоха 33; SHA головы:
  `652486708e1641de34055881f1ffb35a5debbca3de24b580e816e44f33655f18`.
- **Условия, зарегистрированные до проб.** Проследить status → mode → alias
  на фиксированной dev-подвыборке и 32 публичных инструкциях, проверить
  минимальные изменения формулировок, цели обучения и представление данных.
  Пороги 0.99 сохраняются. Диагностическое отключение проверки не считается
  допуском. Бюджет диагностики — до часа внутри исходных 24 часов;
  first_compute/deadline не обнуляются. Новый fit, PPO и оценка закрытого test
  в этой задаче не запускаются.
  [Протокол](runs/text_reward_admission_fix_20261021/protocol.json).
- **Критерий завершения этой задачи.** Подтвердить механизм отказа первичными
  измерениями, согласованно исправить обучение, inference и оценку, проверить
  дедупликацию/границы награды, старые варианты и подготовленный корпус.
  Качество новой обученной модели проверяется отдельным следующим запуском.

На 64 dev-группах (seed 17, 256 текстов) уверенный статус получили 1447/1448
применимых команд. После проверки режима осталось 1395, после проверки
эквивалентности — 673. Правильные held 554/580 и press 561/588 перед последней
стадией превратились в 237/580 и 263/588. Неуверенная связь с другой операцией
могла отменить уверенно распознанную команду и всю её alias-компоненту.
Raw alias argmax — 328/328; после TRAIN-prior — 287/328, при уверенности 0.99 —
97/328. Формула prior математически корректна и не отменяется.
[Поэтапная диагностика](runs/text_reward_admission_fix_20261021/dev_cascade_diagnostic.json).

На тех же 9600 TRAIN-текстах только 297752/6048000 пар (4.92%) относились
к двум применимым командам; для режима релевантны 55476/345600 целей (16.05%).
Новое обучение разделяет status (3 класса), условный mode (held/press) и
условный pair (same/different). Loss, TRAIN-веса, prior, dev-selection и
температуры используют согласованные популяции. На прежнем корпусе вес same
при такой популяции — 8.121 вместо 109.972; это другая задача обучения,
а не изменение уверенности сохранённой модели задним числом.
[Сравнение популяций](runs/text_reward_admission_fix_20261021/conditional_population_audit.json).

На 32 публичных текстах статус «применимо» получили 208/222 истинно применимых
команд, после mode gate — 159/222, после alias gate — 77/222. Эти числа
характеризуют наличие статуса, а не точность режима. На замороженной модели
явное перечисление направлений мыши в русском тексте повысило status 0/5 → 5/5;
краткая английская фраза удержания повысила mode acceptance 4/8 → 8/8, но
проверку aliases не исправила. Одна замена фразы весь отказ не объясняет.
В старом корпусе внутренняя грамматика TRAIN/dev совпадала; compact-наборы
содержали обязательные дополнительные команды и явное пояснение эквивалентности.
[Зарегистрированные варианты](runs/text_reward_admission_fix_20261021/runtime_counterfactuals_registered.json),
[trace](runs/text_reward_admission_fix_20261021/runtime_gate_trace.json).

Реализован отдельный `supported_conditional_v6`, с новой архитектурой и
явными версиями decoder/calibration. Неизвестная дедупликация теперь сохраняет
semantic status, но даёт `group=None`, `deduplication_known=False` и нулевую
награду до обновления seen/caps. Такие ответы не повышают alias coverage:
знаменатель остаётся всеми истинными alias-парами. Неприменимые команды
дедуплицируются по физическому вводу. Исправлена диагностическая NLL, которая
прежде включала необученные диагонали; прежние модели сохраняют старое поведение.
Отдельно исправлен пропуск буквального W/S в русской UI-инструкции.
Функции admission gates AST-идентичны прежним: 0.99, охват 80%, отсутствие ошибок.

V7 подготовлен с seed 20261023: 4000 смысловых групп, 16000 текстов;
SHA `ebd453165ff40fd086281dc314d15e03aaa1039255310113d0642a90574e700a`.
Из всех splits до рендеринга исключены 10726 прежних mapping ID.
Внутренние grammar families разделены: 0/1, 2/3, 4/5, 6/7.
Минимальные navigation8/horizontal4: TRAIN 28/21, dev 10/6, calibration 3/3.
Для calibration доступны 360 независимых групп положительного знака и 331 —
отрицательного. На 13600 незакрытых текстах переданы 90592/90592 обязательных
упоминаний, максимум 325/512 токенов E5. Обученная модель на sealed не оценивалась;
структурные ID и корректность генерации splits проверялись отдельно.
[Корпус](runs/text_reward_admission_fix_20261021/text_data_v7/manifest.json),
[предпроверка](runs/text_reward_admission_fix_20261021/v7_preflight.json).

**Проверки и итог.** Полный Python: 3630 passed за 488.85 с, один прежний
Gym warning; Ruff и diff-check проходят. Conditional/reward/runtime integration:
46 passed. Независимые correctness и ML reviews не нашли блокирующих ошибок;
проверены 3744 train/dev binding-случая без пропущенных/лишних inputs.
Голова, исходный E5 и manifest прежней модели не менялись. Replay старых logits
повторил 32/32 identity и 1152 решения; полный forward с новым locator сохранил
все status/mode/group, у одной из 32 инструкций изменились confidence values.
[Replay](runs/text_reward_admission_fix_20261021/legacy_decoder_replay.json),
[полный forward](runs/text_reward_admission_fix_20261021/legacy_full_forward.json).

Зафиксированы 137 файлов исходников и конфигурация следующего fit.
CLI проверен на проектном Python 3.12 venv; первоначальная ошибочная ссылка
на системный Python 3.14 была поймана help-проверкой и исправлена до запуска.
Обученные веса не импортируются, optimizer будет пустым. Новый fit, его
качество/admission и игровое улучшение **не измерялись**; старый оценщик остаётся
недопущенным. Изученная calibration прежнего корпуса не является новой
независимой сертификацией.
[Проверенный итог](runs/text_reward_admission_fix_20261021/verification.json),
[следующий запуск, не начат](runs/text_reward_admission_fix_20261021/next_fit.json).

<a id="text-reward-conditional-fit"></a>
## Свежий fit условного текстового оценщика 21.10

### `text_reward_conditional_v6_20261021`

- **Регистрация до запуска.** Пользователь отдельно разрешил новый fit после
  завершённой диагностики. Предыдущая запись «обучение не запускалось» относится
  к диагностической задаче, а не к этой попытке.
- **Гипотеза.** Условные mode/pair цели и корпус с разделённой внутренней
  грамматикой устранят каскад отказов при сохранении строгой проверки награды.
  Контроль — сохранённый `v6_grounded_budget` (runtime 3/32, ready=false);
  корпуса отличаются, поэтому это не matched-сравнение точности.
- **Исходная модель.** Новый `supported_conditional_v6`, свежие head и optimizer,
  seed 20261022; собственные обученные веса не импортируются. Исходный E5-small
  заморожен, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- **Данные.** V7, seed 20261023: TRAIN 9600, dev 2400, calibration 1600;
  sealed 2400 закрыты. SHA records:
  `ebd453165ff40fd086281dc314d15e03aaa1039255310113d0642a90574e700a`.
  Dev выбирает эпоху и температуры; calibration проверяет замороженный вариант.
  Отдельная runtime-проверка всех 32 фактических RU/EN инструкций получает текст.
- **Протокол и критерий.** До 40 эпох, batch32, CPU threads4; новая feature cache.
  Precision 0.99, coverage 80%, доверительная граница и ноль критических ошибок
  остаются обязательными отдельно для знаков, режима и aliases; runtime 32/32.
  Один exit0 или снижение loss не означают готовность. Sealed и PPO автоматически
  не запускаются. Неуспех сохраняется с фактическими метриками.
- **Бюджет.** До 7200 секунд вместе с кодированием, fit и проверками; сохранён
  исходный общий deadline 21.10 21:19:09 UTC, счётчик не обнулён. RSS группы ≤6 GiB,
  RAM reserve ≥1 GiB, диск ≥8 GiB; перед стартом прогноз новых данных 5 GiB.
  Supervisor проверяет собственные PID/идентичности, реальные exit codes и файлы.
- **Исходники.** Неизменный snapshot 137 файлов,
  SHA списка `f3b3260a978a886ee2fbe0eb82a2710165eac7075662818576df2dbf2395e5ad`.
  До fit прошли 3630 Python tests, Ruff и независимые проверки кода/методологии.
- **Фактический запуск.** 21.10 в 10:10:10 UTC (13:10 Москва); supervisor PID75176,
  actor PID75182, идентичности и PGID проверены, исходники137/137 и input pins совпали.
  E5 загружен локально; работа идёт под отдельным supervisor, независимо от чата
  и интернета. Stage cutoff 12:10:10 UTC (15:10 Москва). Качество новой
  обученной головы пока не измерено. Передача90592/90592 упоминаний прошла;
  первая E5-проба8текстов/0.222с оценивает кодирование в376.65с (это прогноз,
  а не фактическое время полного корпуса).
  [Проверка запуска](runs/text_reward_admission_fix_20261021/launch_verification.json),
  [Execution](runs/text_reward_admission_fix_20261021/fit_execution.json),
  [process receipt](runs/text_reward_admission_fix_20261021/training_conditional_v6_state.json),
  [log](runs/text_reward_admission_fix_20261021/fit_conditional_v6_stdout.log).


**Проверенный итог этого fit.** Завершён 21.10 в 11:10:05 UTC (14:10 Москва)
за3594.49с (59мин54с), actual exit0; все40эпох/12000updates выполнены.
Actor и supervisor остановлены, PGID пуст. Head, E5, calibration и обе проверки
сохранены; все manifest SHA и137 исходников сверены. Peak RSS2.17GiB,
новый дисковый кеш1.27GiB. Свежая инициализация и пустой optimizer подтверждены.

Допуск **не пройден**, ready=false. Calibration1600текстов/400смысловых групп:
правильно приняты635/9784 применимых команд (охват6.49%); среди651 положительного
ответа635 верны (97.54%). Для неприменимых:21890/30160 правильно приняты
(72.58%);21890/21904 уверенных отрицательных ответов верны (99.94%).
Ошибочных уверенных ответов30 в6смысловых группах. Held0/3940,
press1/4356, aliases0/2924; gates обоих знаков, режимов и aliases не пройдены.

Все32 фактические RU/EN инструкции проверены после выбора головы:
полностью пройдено0/32. Применимые команды24/222, неприменимые427/914,
held0/176, press0/22, aliases0/88; условно неизвестные16/16 распознаны как
неизвестные. Ошибочных уверенных утверждений на этих32текстах нет: отказ
совместимости связан с недостатком полезных уверенных ответов. Это отдельная
проверка; её ноль ошибок не отменяет30ошибок calibration.

По заранее заданной dev-NLL выбрана эпоха1 (0.228325); к40 dev-NLL ухудшилась
до1.079447, тогда как TRAIN-fit loss упал0.435169 → 0.000365. Это наблюдаемое
переобучение/неперенос на отложенные формулировки; конкретная причина ещё
не установлена. Само продление обучения не подтверждено как исправление.
Веса последней эпохи отдельно не сохранены; её метрики сохранены. Старые и новые
цифры не являются matched-сравнением из-за разных корпусов и grammar holdouts.
Sealed text test и игровойtest закрыты, PPO/новые повторы не запускались.
[Проверенный результат](runs/text_reward_admission_fix_20261021/conditional_v6_result.json),
[calibration](runs/text_reward_admission_fix_20261021/reward_text_conditional_v6/validation.json),
[32 инструкции](runs/text_reward_admission_fix_20261021/reward_text_conditional_v6/runtime_validation.json).

**Диагностика после завершения, без нового fit.** На той же выбранной голове
проведены112 frozen-forward случаев: первые4 TRAIN/dev смысловые группы,
обе RU/EN версии, одинаковые mapping/order seeds при замене только семейства
формулировки, плюс32 публичных текста. На фиксированных TRAIN mappings mode
argmax52/52 в family0 →46/52 вfamily2; на dev mappings56/56 →48/56.
Это небольшая направленная проверка механизма переноса, не оценка всей популяции.
На исходных runtime32 status argmax для применимых222/222, mode183/198,
same-alias88/88; эти диагностические argmax не подтверждают уверенность или допуск.
До mode-temperature уверенных64/198 решений, из них6ошибочных; послеT1.584893
уверенных0/198. Поэтому снять калибровку и принять эти ответы нельзя.
Первый forward завершился, но запись JSON упала на tensor-count; повторV2
с Python-int counts завершил112случаев за3.023с. Веса/пороги не менялись,
первая registration сохранена; ни один из этих запусков не является обучением.

Проверен фактический корпус: TRAIN9600текстов содержит только семьи0/1,
dev2400 —2/3. Явный список «Независимые действия» есть9008/9600 TRAIN-текстов;
это отличается от процедурных игровых инструкций. Среди480 TRAIN nav/horizontal
спецификаций чистых shared-mode complete случаев0; обе чистые shared-mode
спецификации имеют remainder=unknown. All-old/public reservation явно исключает
стандартные полные navigation-held и horizontal-held mappings из всех splits,
хотя по v7seed они попадали бы в TRAIN. Их происхождение включает старый TRAIN
и публичную compatibility-проверку, а не только независимый test.
Это доказанный сдвиг обучающей поддержки. Его доля в общем отказе без matched-fit
не измерена. Данные, выбор эпохи и калибровку необходимо согласовать до следующего
дорогого запуска; новый fit/PPO и sealed test в этой диагностике не выполнялись.
[Измеренные механизмы и ограничения](runs/text_reward_admission_fix_20261021/conditional_cause_evidence.json),
[frozen wording probe](runs/text_reward_admission_fix_20261021/conditional_transfer_diagnostic_v2.json).


<a id="text-reward-transfer-fix"></a>
## Исправление корпуса и обучения с короткими пробами 21.10

### `text_reward_transfer_fix_20261021`

- **Запрос и регистрация до обучения.** Пользователь разрешил исправить корпус
  и обучение, сначала проверить клавиши/режимы/эквивалентность короткими прогонами,
  затем запускать полный fit только после подтверждённого сигнала. Пороги не снижать.
- **Контроль.** Сохранённый conditional v6, выбранная эпоха1: ready=false,
  calibration30ошибок и runtime0/32. Его данные, веса и результаты неизменны.
- **Гипотеза.** Восстановленные elementary TRAIN-случаи и разнообразные конструкции
  привязки команд дадут перенос; полный decoder на новых development-текстах
  должен подтвердить это до дорогого fit.
- **Данные и границы.** Новый v8 формируется после labels-only проверки истории.
  Исторические sealed/test mappings исключаются всегда; прежние изученные
  TRAIN/dev/cal/public mappings допускаются только в TRAIN. Новые dev/cal/sealed
  исключают весь исторический union, смысловые группы назначаются до парафраз.
  Исходные32 публичные инструкции не копируются в генератор и остаются
  development compatibility-проверкой, не независимой сертификацией.
- **Короткие ветви.** A: новая голова/optimizer, seed20261024, новый корпус,
  прежний LR3e-4/dropout0. Только еслиA не проходит —B с теми же инициализацией
  и данными, baseLR1e-4/pairLR3e-4/dropout0.1. До8эпох/900с на ветвь.
  E5 исходный замороженный; собственные обученные веса не импортируются.
- **Протокол выбора.** Dev делится по semantic groups на temperature-fit и
  checkpoint-score. Температуры не используют score; probe не предсказывает
  calibration/sealed. Порог0.99, coverage≥0.8 отдельно для обоих знаков,
  held/press/aliases, critical errors0; score≥100 независимых групп, для каждого
  mode/aliases≥30групп и≥100 golddecisions. Это feasibility, не ready=true.
- **Полный fit.** Только после успешной пробы, заново с seed20261025,
  до40эпох/7200с. Исходные calibration gates/доверительные границы и runtime32/32
  обязательны без послаблений. Игровое PPO в эту задачу не входит.
- **Бюджет и состояние.** Подготовка кода/корпуса, новые fit ещё не начаты.
  Исходный 24-часовой deadline21.10 21:19:09UTC сохраняется, счётчик не обнулён.
  [Протокол](runs/text_reward_transfer_fix_20261021/protocol.json),
  [структурная история splits](runs/text_reward_transfer_fix_20261021/partition_registry_v8.json).


**Подготовка v8 перед пробами.** Сохранены обе подготовки: первая `text_data_v8`
без aggregate-axis фраз не обучалась; независимый review выявил этот пробел.
Активная `text_data_v8b`: seed20261020,4000смысловых групп/24168текстов,
TRAIN14568/dev3600/calibration2400/sealed3600;84 elementary группы,
4 обязательных чистых полных directional mapping восстановлены в TRAIN.
Aggregate mouse-axis independentgroups: TRAIN115/dev21/cal12/sealed29.
Hard-block исторического sealed4864 ID, developmental-known24819 IDs толькоTRAIN.
На20568 незакрытых текстах переданы89840/89840 упоминаний; max278/512E5токенов.
Calibration имеет380/322 независимых групп соответствующих знаков до предсказаний.
Пересечений новых heldout с историей/междуsplit нет, public32 строки не копировались.
Предпроверка не является измерением качества модели.
[Корпус](runs/text_reward_transfer_fix_20261021/text_data_v8b/manifest.json),
[предпроверка](runs/text_reward_transfer_fix_20261021/text_data_v8b_preflight.json).

Код probe/full дополнен grouped dev temperature/score выбором по полному decoder,
раздельными LR и dropout только во время обучения. Пороги `_count_gate` и
`calibration_gates` AST-идентичны прежним. Worker checks80training+15corpus tests
прошли; независимый ML-review не нашёл блокеров. Общий Ruff проходит;
полный Python suite выполняется до старта пробы. Новые модели пока не обучались.

**Регистрация стадии `probe_a` до запуска:** seed20261024, dataset`text_data_v8b`, source`source_text_data_v8b` (137файлов); recipeA:LR3e-4/drop0, лимит900с, [execution](runs/text_reward_transfer_fix_20261021/probe_a/execution.json). Пока подготовка; фактическое состояние проверяется по process.json/PID.


**Проверки и запуск probeA.** Полный Python3656passed/1прежнийGym warning
за523.66с; Ruff и diff-check проходят.137файлов frozen source совпадают с
проверенным рабочим кодом. ProbeA запущен21.10 в12:15:56UTC (15:15Москва),
supervisor87945/actor87952; PID/идентичности и PGID проверены.388TRAIN-групп/
2496текстов, все84 elementary сохранены; dev temperature1800текстов/300групп,
checkpoint-score1800текстов/300другихгрупп, RU/EN. Calibration/sealed не кодируются
и не оцениваются. Пока идёт подготовка признаков; качество ещё не измерено.
[Проверки](runs/text_reward_transfer_fix_20261021/checks_before_probe.json),
[process](runs/text_reward_transfer_fix_20261021/probe_a/process.json).


**Уточнение бюджета до будущей probeB.** Фактический выбор по полному decoder
занимает33–51с на эпоху сверх начального throughput-прогноза. Для возможнойB
заранее выделен предел1200с вместо900с, чтобы завершить те же8эпох и итоговую
проверку. Лимит уже работающейA900с не меняется. Данные, seed, критерии качества,
пороги и исходный общий deadline остаются прежними; увеличение времени проверки
не является улучшением модели. B всё ещё запускается только после результатаA.

**Регистрация стадии `probe_b` до запуска:** seed20261024, dataset`text_data_v8b`, source`source_text_data_v8b` (137файлов); recipeB:base1e-4/pair3e-4/drop.1, лимит1200с, [execution](runs/text_reward_transfer_fix_20261021/probe_b/execution.json). Пока подготовка; фактическое состояние проверяется по process.json/PID.


**Итог probeA.** Фактически завершены8эпох/1248updates за775.63с, exit0;
actor/supervisor остановлены, PGID пуст. Выбрана эпоха3, ни одна эпоха не прошла
feasibility. На checkpoint-score300смысловых групп/1800текстов правильно приняты
1832/7656 применимых и36534/37680 неприменимых команд; held1041/3402,
press647/3156, aliases4/882;7критических ошибок. Public compatibility0/32.
До температур у поздней8эпохи same-alias argmax1649/1716, но после полного decoder
этого недостаточно для допуска; более низкий TRAIN-loss не является успехом.
Последняя8 отдельно: coverage held68.72%,press80.39%,aliases0.68%,8ошибок.
Feasibility остаётся отрицательной; calibration/sealed не использованы,
основной text-fit и PPO не запускались. Исходники и результаты сохранены.
[Полный probeA](runs/text_reward_transfer_fix_20261021/probe_a/output/probe.json).

**Запуск probeB.**21.10 в12:29:59UTC (15:29Москва), supervisor90310/actor90319;
PID/идентичности проверены. Те же source/data/seed2496TRAIN, пустой optimizer,
baseLR1e-4,pairLR3e-4,dropout0.1; read-only cacheA проверяется по хешу и
точному набору текстов. Предел1200с зарегистрирован до запуска. КачествоB
пока не измерено, full по-прежнему заблокирован до положительной feasibility.
[processB](runs/text_reward_transfer_fix_20261021/probe_b/process.json).

**Разбор каскада probeA (замороженная эпоха3, DEVscore).** Собственный mode
верен6531/6558 по argmax,6342/6558 при собственной уверенности0.99, но
после status-gate остаётся1688/6558. Для882 настоящих alias-пар: raw argmax864,
raw correct confidence0.99 —813; после корректной TRAIN-prior поправки226,
после собственной температуры12, оба endpoint supported80, оба eligible5,
полностью верны4. У raw pair0.99 есть26 ложных same-решений, поэтому удаление
поправки/температуры не является исправлением. Среди80 пар с поддержанными
endpoint75 заблокированы неопределённостью собственного ребра,0 — только
другими неопределёнными рёбрами: оснований менять complete-graph decoder нет.
Cross-modal alias argmax6/24 против normal270/270;22/27 mode-ошибок связаны
с binding-family13. Это выделяет недостаток переноса конкретных отношений;
вклад ещё проверяется наB, адресная TRAIN-доработка пока лишь готовится.
Два read-only cached-feature forward по DEV,16+14с/2CPUthreads;
calibration/sealed не использованы.
[Диагностика](runs/text_reward_transfer_fix_20261021/probe_a/diagnostic_selected_dev_v2.json).

**Направленная подготовка probeC после8 отрицательных эпохB.** Итоговый receiptB
ещё ожидается; до fitC он обязателен. Два малых probe по388TRAIN-групп
не подтвердили перенос. C заранее планируется до8эпох/2400с, batch32,
baseLR1e-4/pairLR3e-4/dropout0.1, тот же freshseed20261024 и пустой optimizer.
Все существующие TRAIN плюс72matchedpairs (144новыеTRAIN-группы/864текста):
клавиша↔мышь equivalent/independent; та же явно разрешённая/запрещённая клавиша;
соседние команды с противоположными held/press. Старые dev/cal/sealed сохраняются
побайтно, новые mapping не пересекаются с ними или hardsealed. Вся TRAIN-часть
используется вместо388-группового subset (предел3000групп/16000текстов).
Это совместная проверка разнообразия и контрастов, не доказательство вклада
каждой части. Новые исходники проходят проверки до fit; критерии/пороги и
общий deadline не меняются. Полный fit/PPO не разрешаются отрицательной пробой.

**Итог probeB.**8эпох/1248updates,663.14с, actualexit0; выбраннаяэпоха8,
ни одна не feasible. На техже1800score-текстах/300групп: supported2669/7656,
unsupported36409/37680; held1368/3402,press1246/3156,aliases205/882;
15критических ошибок, public0/32. Инициализация головыSHA37827b7c…14bfa8d
совпадает сA, optimizerentries0; source137файлов неизменны, cachemanifestSHA
совпадает с pin, actor/supervisor отсутствуют и PGID пуст. Рост aliases противA
не достиг установленного допуска и не является готовой наградой.
[РезультатB](runs/text_reward_transfer_fix_20261021/probe_b/output/probe.json),
[receipt](runs/text_reward_transfer_fix_20261021/probe_b/process.json).

**Подготовлен v8c без обучения.**4144группы/25032текста,
TRAIN2544группы/15432текста,dev600/3600,cal400/2400,sealed600/3600.
Все существующие записи — точный byteprefix v8b; каждый heldout split имеет
тот же rawlineSHA. Добавлены только144TRAIN-группы/864текста по72парам.
6 новых регрессионных тестов и Ruff прошли, полный suite повторяется;
структурная/tokenizer предпроверка и независимый review ещё выполняются.
Эффект augmentation не измерен; никакой fitC до проверок не запущен.
[Manifest](runs/text_reward_transfer_fix_20261021/text_data_v8c/manifest.json).

**Регистрация стадии `probe_c` до запуска:** seed20261024, dataset`text_data_v8c`, source`source_text_data_v8c` (139файлов); recipeB:base1e-4/pair3e-4/drop.1, лимит2400с, [execution](runs/text_reward_transfer_fix_20261021/probe_c/execution.json). Пока подготовка; фактическое состояние проверяется по process.json/PID.

**ЗапускC после обязательных проверок.**3662tests прошли за537.79с,actualexit0;
Ruff чистый,137общихsourceфайлов не изменены,2новых проверены. V8c preflight
на21432незакрытых текстах:91712/91712упоминаний, максимум278/512токенов;
все heldout rawlineSHA и864новые labels/IDs независимо сверены, MLreviewGO.
C запущен21.10 в12:49:37UTC (15:49Москва), supervisor93224/actor93238,
PID/идентичности проверены. Новый immutable source139файлов, свежий head/optimizer;
run_config подтверждает весьTRAIN2544группы/15432текста,7716RU+7716EN,
толькоTRAIN/dev features,probe-only,все72пары включены. Результата качества ещё нет.
[Checks](runs/text_reward_transfer_fix_20261021/checks_after_contrastive.json),
[preflight](runs/text_reward_transfer_fix_20261021/text_data_v8c_preflight.json),
[processC](runs/text_reward_transfer_fix_20261021/probe_c/process.json).

**ОтказC до первой эпохи и новый дефект supervision.**C самостоятельно завершился
exit1 за241.94с: guard по9updates оценил fit1498.93с×1.5, что превышало
остаток2400с после подготовки. Это технический отказ, не отрицательная оценка
качества; quality/DEV/runtime не измерены, их нельзя считать нулями.
Featurecache1647118080байт готов, encoding185.06с; partial_head сохранён.
Actor/supervisor отсутствуют, PGIDпуст,139sourcehash неизменны. Сигнал остановки
не отправлялся: проверка identity установила, что процесс уже завершён.
[Failure](runs/text_reward_transfer_fix_20261021/probe_c/output.failure.json),
[throughput](runs/text_reward_transfer_fix_20261021/probe_c/workspace/fit_throughput_probe.json).

Независимый semanticreview12TRAIN-текстов подтвердил отдельный корпусныйдефект:
positivealias явно говорит «варианты равнозначны», а negativealias присваивает
разные группы двум genericописаниям «the action», не сообщая разных эффектов.
Разные предложения и footer о неприменимыхкомандах этого не доказывают;
outputcontract допускает неизвестную связь. Вклад дефекта в числаA/B не измерен.
Поэтому прежний v8нельзя принимать за исправленный истинностныйпротокол.

**Регистрация directedD до подготовки/fit.** Новый v9 переписывает ТЕКСТЫ всех
splits с явными границами групп/разнымиэффектами и равнозначностью внутри,
сохраняя structuralIDs/labels/splits. Старые v8b/v8c inputs/results остаются
нетронутыми; это смена входного протокола, результаты нельзя сравнивать как
одинаковыйDEV. Sealed не оценивается и не токенизируется; corpusbuilder
использует структурнуюspec, не проверочныеответы/диагностики/runtimeстроки.
Новая проба планируется8эпох/2400с,batch32,1000TRAIN-групп/до6400текстов,
включая все84elementary и144contrastiveгруппы атомарнымипарами; seed20261024,
baseLR1e-4,pairLR3e-4,drop0.1. Все численныеqualitygates и исходный24hdeadline
сохраняются. Код/корпус сначала проходят проверки; полноценныйfit не запущен.

**Подготовлен v9 до fitD.**4144группы/25032текста, SHAea598b50…af8e80a7.
Все25032записи после исключения толькоtext/grammar_families побайтно как
структурные объекты совпадают сv8c: IDs/labels/splits не менялись. Отношения
внутри/между скобками теперьявны,24split-ownedформы; неизвестные/запрещённые
в relationwitness не включаются. Preflight21432незакрытыхтекста:91712/91712
упоминаний, max364/512токенов, пересечений сhardsealed/историейheldout нет;
sealed не токенизировался/не оценивался. Readerv9/атомарныйsampler проверены:
D1000групп/6168текстовRU3084+EN3084, все84elementary и144contrastive/72пары
обязательны сразнымиreason; elementarymetadata не подменяется. Недостаточныйcap
явноотказывается, неполная/повреждённая пара отвергается. Throughputguard/gates
не менялись.115worker и29независимых focusedtests прошли; Ruff чистый,
общий suite повторяется до fit. IndependentMLreviewGO. Runtime32 исходных
естественных игровыхинструкций не менялся и остаётся обязательным полнымgate:
syntheticwitnesscoverage не доказывает этот перенос.
[Manifestv9](runs/text_reward_transfer_fix_20261021/text_data_v9/manifest.json),
[preflight](runs/text_reward_transfer_fix_20261021/text_data_v9_preflight.json),
[semantictruth audit](runs/text_reward_transfer_fix_20261021/alias_semantic_truth_audit.json).

**Регистрация стадии `probe_d` до запуска:** seed20261024, dataset`text_data_v9`, source`source_text_data_v9` (141файлов); recipeB:base1e-4/pair3e-4/drop.1, лимит2400с, [execution](runs/text_reward_transfer_fix_20261021/probe_d/execution.json). Пока подготовка; фактическое состояние проверяется по process.json/PID.

**Запуск D после проверок.** Полный Python: 3678 passed, один прежний Gym warning,
529.12 с, actual exit 0; Ruff и diff-check прошли. Гейты AST-идентичны прежним.
D запущен 21.10 в 13:12:50 UTC (16:12 Москва), supervisor 98653 / actor 98676;
идентичность процессов проверена. Заморожен 141 исходный файл. Фактический
run_config: 1000 TRAIN-групп / 6168 текстов (RU/EN по 3084), все 84 elementary
и 144 contrastive-группы / 72 пары; признаки только TRAIN/dev. Head/optimizer
свежие, лимит 2400 с, критерии прежние. Качество D пока не измерено.
[Проверки](runs/text_reward_transfer_fix_20261021/checks_v9.json),
[процесс D](runs/text_reward_transfer_fix_20261021/probe_d/process.json).

**Итог D: допуск не пройден.** 8 эпох / 1544 updates, 1233.18 с, actual exit 0;
выбрана эпоха 8. Source 141 файлов неизменен, PID отсутствуют, PGID пуст,
head-init SHA совпадает с A/B, optimizer entries 0. Score (300 групп / 1800
текстов): supported 2811/7656, unsupported 36616/37680, held 1163/3402,
press 1040/3156, aliases 220/882, critical errors 11. Runtime 0/32 целиком;
отдельно supported 16/222, held 16/176, press 0/22, aliases 4/88,
unsupported 566/914, unknown 16/16. Явных ошибочных runtime решений нет,
но преобладает абстенция. Эти числа не доказывают готовности награды.
Cal/sealed не предсказывались, полноценный fit/PPO не запускались.
[Результат D](runs/text_reward_transfer_fix_20261021/probe_d/output/probe.json),
[фактическое завершение](runs/text_reward_transfer_fix_20261021/probe_d/process.json).

Дополнительный source-аудит обнаружил ограничение core-лексики: 24 внешние
формы сохраняют один hold и один press-once-release. В 7/32 неизменённых
публичных инструкций основная формулировка другая (tap, click, короткое
нажатие, fire, отложенное анафорное release). Подсказка release присутствует,
поэтому это частичный пробел; влияние на числа ещё не измерено.

**Регистрация E до нового fit.** Направленная TRAIN-only доработка заменит
по одному RU/EN виду каждой группы естественными, однозначными описаниями
режима и разных функций, включая отложенную timing-фразу; скобочный witness
не обязателен в новых видах. Остальные TRAIN и все heldout строки сохраняются
побайтно; IDs, labels, splits, пары и режим проверки остаются прежними.
Дополнительно проверяется больше уникальных TRAIN-групп: вся часть
2544/15432 вместо D1000/6168. Это совместная проверка разнообразия форм
и объёма, не отдельная оценка их причинного вклада. Пилот E: 8 эпох, batch32,
seed20261024, fresh head/optimizer, base1e-4/pair3e-4/drop0.1; до3600с
вместо не поместившихся в C2400с, исходный общий deadline сохраняется.
Пороги и actual32 неизменны; calibration/sealed не используются в probe.
Подготовка ещё идёт, fit E не начат; полный fit по-прежнему заблокирован.

**Регистрация стадии `probe_e` до запуска:** seed20261024, dataset`text_data_v9_natural`, source`source_text_data_v9_natural` (143файлов); recipeB:base1e-4/pair3e-4/drop.1, лимит3600с, [execution](runs/text_reward_transfer_fix_20261021/probe_e/execution.json). Пока подготовка; фактическое состояние проверяется по process.json/PID.

**Frozen-диагностика D уточнила причину.** На public32 mode held верен
176/176 и при argmax, и при собственной confidence≥0.99; в decoder остаётся
16/176 из-за status. Supported: raw argmax218/222; raw99correct174, после
корректировки prior28, после температуры19, decoder16. Press: argmax17/22,
после температуры11/22, decoder0. Same-alias: argmax88/88, confidence99
raw88→prior66→T30; оба endpoint приняты только у4пар, все4 сгруппированы верно.
TRAIN status99 до температуры покрывает лишь886/1870 — это также недостаток
обученных margins, не только сдвиг public wording. Отмена prior/T не обоснована.
Контрфактическая подсказка ответов в тексте дала65 вместо16 supported, но
10 явных ошибок и ухудшила held; это исключительно teacher-assisted diagnostic,
не улучшение production или новый benchmark. Ни одного optimizer update.
[Диагностика D](runs/text_reward_transfer_fix_20261021/probe_d/diagnostic_public_transfer.json).

**Natural подготовка и старт E.** ПодготовкаA с двусмысленной фоновой фразой
о menu buttons не обучалась и сохранена отдельно; literal исправлен на явно
неинтерактивные значки. Текущий SHA941df9c0…cfb8664: заменены5088TRAIN views,
1439 с отложенным release,1201 с фоновым контекстом. Все heldout строки
побайтно прежние. Максимум364/512токенов,91712/91712 упоминаний; strictsealed
не токенизировался. Независимый ML-review GO. Полный suite3684passed/1прежний
warning за526.70с, actualexit0; после двух literal-правок повторены6 тестов
этого модуля (4.05с), Ruff всего проекта чистый.
E запущен21.10 в13:52:29UTC (16:52Москва), supervisor3325/actor3340;
реальная identity проверена,143sourcefiles сверены с checks. Run_config
подтверждает весьTRAIN2544группы/15432текстаRU7716+EN7716,84elementary+72пары,
новый head/пустой optimizer,8эпох,до3600с. Неизменныеcalibration/sealed
в probe не предсказываются. Качество E пока не измерено, полный fit запрещён
до положительной feasibility.
[Manifest](runs/text_reward_transfer_fix_20261021/text_data_v9_natural/manifest.json),
[Checks](runs/text_reward_transfer_fix_20261021/checks_natural.json),
[processE](runs/text_reward_transfer_fix_20261021/probe_e/process.json).

**Итог E (фактическое завершение 21.10, 14:26 UTC).** Exit 0, 2027.86 с,
8 эпох / 3864 обновления; выбран CP8, ни одна эпоха не прошла feasibility.
Начальный head SHA37827b7c…14bfa8d совпал с A/B/D, optimizer пустой;
прошлые обученные веса не импортированы. Процессы 3325/3340 завершены,
группа пуста, все 143 замороженных исходника неизменны. Пик RSS группы
3 512 926 208 байт; признаки 2 357 501 184 байта, кодирование 238.28 с.

| Проверка CP8, score DEV | Верно принятые / явно заданные |
| --- | ---: |
| Разрешённые команды | 6917/7656 |
| Неприменимые команды | 36405/37680 |
| Удержание | 3064/3402 |
| Отдельное нажатие | 2801/3156 |
| Эквивалентные команды | 295/882 |

Критических ошибок 16: status 10, mode 2, false split 4. На исходных public
инструкциях целиком прошли **2/32** (mouse RU и timing RU): supported 101/222,
held 84/176, press 5/22, aliases 13/88, unsupported 861/914, unknown 15/16.
Есть пять явных ошибочных ответов: interaction RU (E held вместо press),
key_door EN (неприменимый Enter), shooting RU/EN (mouse:left held вместо press),
ui RU (условная Down принята без наблюдения раскладки). Остальное включает
воздержания; их нельзя считать правильным распознаванием управления.

Вывод: обучение стало осваивать status и mode, но готовность не подтверждена.
Общий допуск блокируют ошибки и низкий охват эквивалентности. У E одновременно
изменены natural TRAIN views и объём обучения; их отдельный причинный вклад
не измерен. Full fit и PPO не запускались; calibration/sealed не предсказывались.
[Результат E](runs/text_reward_transfer_fix_20261021/probe_e/output/probe.json),
[Фактический процесс](runs/text_reward_transfer_fix_20261021/probe_e/process.json).

**Диагностика E: ошибка генератора и отдельный провал cross-modal aliases.**
В v8 `_canonical` использовал dummy `key:w`, а затем заменял имена в строке.
`_binding` капитализировал первую букву: начальное «Клавиша W» не совпадало
с заменяемым «клавиша W». Поэтому некоторые RU horizontal тексты ошибочно
называли W, отсутствующую в фактическом управлении и размеченную unsupported.
В score-half найдено 3/1800 таких строк: они объясняют 3/10 status errors,
но не mode/alias errors. После их диагностического исключения остаются
13 критических ошибок; это не основание принимать модель или исключать эти
страты из допуска. Исправленные тексты должны стать явно новым протоколом.

Отдельно raw pair argmax: cross-modal same 10/24, остальные same 858/858.
Все четыре уверенных false split относятся к клавише и кнопке мыши.
До собственного порога эквивалентности не доходят 372/688 пар с принятыми
endpoint; status отклоняет endpoints 179/882 пар. Поэтому один дефект W
не объясняет основной отказ. Положительная температура не меняет pair argmax;
диагностические границы для отдельной pair-head не являются доказательством
невозможности любой совместной настройки status/mode/pair decoder.
Ни одного optimizer update и никакого calibration/sealed inference.
[Frozen E диагностика](runs/text_reward_transfer_fix_20261021/probe_e/diagnostic_alias_failure.json).

**Регистрация F до обучения: новая формулировка v10 и частота TRAIN.**
В underlying renderer реальные canonical names передаются до капитализации,
без dummy-key и подстановки строки. Новый v10 отдельно исправил115фраз:
TRAIN72, DEV19, sealed24, calibration0. Изменено только поле text точного
дефектного canonical clause; structural IDs/labels/splits, natural views и
остальные строки прежние. Старые входы сохранены. Новый SHA
`d89533c169bcfd25721cd37ee860ab4d138dfb7be9e8ea2d9e49e9f7cf2547bc`.
Это явная смена wording-протокола; не парное сравнение на одинаковых входах.

F проверит направленную гипотезу: больше TRAIN-экспозиций сложного stratum
помогут выучить семантическое различие key↔mouse. Повторяются все106его групп
(636уникальных текстов) ×4, включая обе стороны контрастов; остальные ×1.
Всего17340предъявлений заэпоху, но по-прежнему15432уникальных TRAIN-текста /
2544группы. Classbalance/prior должны считаться по фактическому multiset,
а независимые метрики — по уникальным примерам. DEV/cal/sealed не повторяются.

Остальное как E: fresh head/пустой optimizer, seed20261024,8эпох,batch32,
baseLR1e-4/pairLR3e-4/drop.1,до3600с, всеTRAINgroups и72полные пары.
Пороги, confidence bounds, критических ошибок0 и public32 не меняются.
Calibration/sealed в probe не предсказываются. Пока подготовка и проверки,
не запуск. Успех этой комбинации не установит отдельный вклад каждого изменения.

Для возможного последующего fresh full seed20261025 зарегистрирован предел
10эпох вместо40, максимум7200с прежний. По измеренной E скорости с12.4% дополнительных
TRAIN batches оценка с резервом менее4150с. Это предел вычислений, без изменения
качества допуска; full по-прежнему запрещён без успешной пробы. Исходный общий
deadline21.10 21:19:09UTC не переносится.
[Протокол](runs/text_reward_transfer_fix_20261021/protocol.json),
[Корпус v10](runs/text_reward_transfer_fix_20261021/text_data_v10/manifest.json).

**Регистрация стадии `probe_f` до запуска:** seed20261024, dataset`text_data_v10`, source`source_text_data_v10` (145файлов); recipeB:base1e-4/pair3e-4/drop.1, лимит3600с, [execution](runs/text_reward_transfer_fix_20261021/probe_f/execution.json). Пока подготовка; фактическое состояние проверяется по process.json/PID.

**Фактический запуск F 21.10, 14:53:44 UTC (17:53 Москва).**
Перед fit весь suite3895passed/1прежний warning за532.47с, actualexit0;
Ruff src/scripts/tests чистый,326профильных проверок и независимый ML-review GO.
Gate AST совпадает с E для calibration и feasibility; все145sourcefiles
совпадают с проверенными хешами и заморожены в source_text_data_v10.
Supervisor7407/actor7424 проверены по реальной identity, pins прошли;
процесс работает offline. Новый head/optimizer, старых весов нет.
Лимит3600с и прежний globaldeadline сохранены; диск32.44GB, forecast5GiB
с дополнительным остатком8GiB. Качество F пока не измерено.

В sourceviews15432 записей, но15401различная строка:31повтор той же семантики
не является новым независимым случаем. При×4 планируется17340предъявлений
заэпоху, независимых групп2544. Effective TRAIN class counts проверены:
status[65216,406648,152376],mode[26754,28278],pair[8454,113450].
Full fit и PPO по-прежнему не запущены.
[Checks](runs/text_reward_transfer_fix_20261021/checks_v10.json),
[Exposure audit](runs/text_reward_transfer_fix_20261021/train_exposure_v10_audit.json),
[Execution F](runs/text_reward_transfer_fix_20261021/probe_f/execution.json),
[Process F](runs/text_reward_transfer_fix_20261021/probe_f/process.json).

**Отдельный проверенный пробел lexical input (во время F, без его изменения).**
В key_door EN обычный глагол `Return` в `Return to the door` получает
Enter-anchor span48:54. В TRAINv10 нет ни одного whole-word return;
все3513 enter-упоминаний квалифицированы как названия клавиши. Обычных
verb hard negatives return/enter нет. Это подтверждённый пробел покрытия
и лишняя lexical-привязка; причинный вклад в model prediction пока не измерен.
Никакого нового forward/optimizer или изменения активной F не выполнялось.
[Read-only diagnostic](runs/text_reward_transfer_fix_20261021/probe_e/diagnostic_return_anchor.json).

**Итог F 21.10, 15:31 UTC.** Actualexit0,2251.78с,8эпох/4336обновлений.
Начальная SHA головы37827b7c…14bfa8d совпала с E, optimizerstate0;
фактически использован×4multiset. Процессы7407/7424 завершены, группа пуста,
145исходников/входные pins неизменны. Пик RSS2 840 297 472байта.
Ни одна эпоха не прошла feasibility. По зарегистрированному NLL fallback
выбран CP7: supported6213/7656,unsupported36383/37680,held2877/3402,
press2326/3156,aliases302/882;27criticalerrors(status26,mode1,alias0).

Последняя эпоха8 дала supported81.02%,held89.18%,press69.49%,aliases54.20%
и9errors. Это только сохранённые метрики: отдельные веса CP8 не сохранены,
доступны выбранный CP7 и история. Поэтому CP8 не выдаётся за проверенную
готовую модель и не выбирается задним числом. Для будущих диагностических
прогонов уже есть CLI `--diagnostic-update`, позволяющий заранее сохранить
заданные точки, включая последний update.

На неизменённых public32 CP7 прошёл3/32 целиком: supported48/222,
held32/176,press6/22,aliases3/88,unsupported863/914,unknown15/16.
Четыре явные ошибки: camera EN(dx:- ошибочно unsupported),shootingRU/EN
(mouse:left held вместоpress),uiRU(Down ошибочно известна без layout).
Return/Enter больше не является явной ошибкой этой F, но его ложный lexical
anchor сохраняется; правильное распознавание unsupported не следует из
отсутствия explicit error, возможен abstention.

Вывод: ×4 не обеспечил допуск и не дал устойчивого улучшения public-команд.
Изменены одновременно несколько входных фраз и частота TRAIN, отдельные
эффекты не изолированы. Готовой награды, полного fit и PPO нет.
[Результат F](runs/text_reward_transfer_fix_20261021/probe_f/output/probe.json),
[Процесс F](runs/text_reward_transfer_fix_20261021/probe_f/process.json).

**Capacity G зарегистрирована до оптимизации.** После F выполняется техническая
TRAIN-only диагностика: фиксированные F CP7 features (eval/dropoutoff),636views/
106cross-modal групп, точный264DвходpairMLP; только gold-supported верхние пары.
Проверяются конечность features и одинаковыеX с противоположнымиlabels.
Ветви: копия F pairMLP264→64→2 (до2000updates) и свежийlinear264→2 (до1000).
AdamWlr1e-3/wd0/batch256,seed20261024,2CPUthreads; Fweights/prior неизменны,
T=1,threshold.99, никакогоTfit. Цель TRAINdiagnostic:0argmaxошибок,≥.8correct99
coverage отдельноsame/different и0wrongaccepted на двух последовательных checks.

Скрипт до240с, внешний supervisor до360с (его preflight требует≥300срезерва),
4GiB RSS,256MiB forecast,8GiB diskheadroom; общийdeadlineнепереносится.
E5не загружается, Fcache/readouts исходной модели не меняются. DEV/cal/sealed
не предсказываются. Копия обученного F используется **только в диагностике**;
это не freshfull и не перенос в новую основную модель. Выход всегдаready=false.
Успех покажет возможность подогнать конечную TRAIN-подвыборку, не перенос,
сохранение остальных страт или единственную причину Ffailure. Неуспех в лимите
не докажет неразделимость без exactfeaturecollision.
[Регистрация](runs/text_reward_transfer_fix_20261021/capacity_g/manifest.json).

**Причинная проверка и исправление Return/Enter.** На замороженных E/F головах
изменена только одна ошибочная lexical-привязка в одной публичной инструкции:
у E ложный Enter supported/press .996312 сменился на unsupported/unknown .999934;
у F abstention .979520 — на unsupported/unknown .999933. Остальные 35 команд
не изменились. Это устанавливает причину конкретного ответа, а не остальных
ошибок. Текущий grounder отличает обычные `Return to the room` / `enter the area`
от явных `press Return` / `Enter key`; он по-прежнему не назначает status/mode/alias.
13 регрессионных случаев добавлены, 176 сфокусированных тестов и Ruff прошли.
На всех21432 незакрытых строках корпуса lexical-маски прежние; маска публичного
key_door EN точно совпадает с измеренным удалением одного ложного token-anchor.
Старые snapshot/results не изменены; новое общее качество32 ещё не измерено.
[Counterfactual](runs/text_reward_transfer_fix_20261021/probe_f/diagnostic_return_counterfactual.json),
[Проверка исправления](runs/text_reward_transfer_fix_20261021/verb_grounding_fix_verification.json).

**Итог G.** Actualexit0,12.81с вместе с supervisor (скрипт9.12с), процессы11172/11182
завершены, группа пуста; source/input pins и исходная F голова неизменны.
Из636TRAINviews извлечены3618пар:492same/3126different; exactX/opposite-label
конфликтов0. Копия F MLP прошла TRAIN-диагностику на checks50/100: после100updates
при исходном prior,T=1 и .99 полученыsame454/492,different3117/3126; argmax
ошибок0 и ошибочных принятых ответов0. В key↔mouse срезеdifferent1078/1086.
Свежий linear после1000updates не прошёл: same186/492,13argmaxошибок.
Это доказывает возможность подогнать конечную TRAIN-подвыборку существующим MLP
при фиксированном trunk. Перенос на DEV, сохранение прочих страт, причину всех
отказов или готовность награды G не доказывает. Пик RSS1 081 147 392байта.
Full fit/PPO остаются заблокированы.
[Диагностика](runs/text_reward_transfer_fix_20261021/capacity_g/output/report.json),
[Проверка процесса](runs/text_reward_transfer_fix_20261021/capacity_g/verification.json).

**H зарегистрирована: проверка переноса без optimizer.** F CP7 сравнивается с той же
головой, где заменён только pairMLP из G после100updates. Проверяются весьTRAIN
(ordinary/crossmodal отдельно), DEV и неизменённый decoder. Сначала исходные
Fprior/температуры; затем отдельный вариант с переоценкой толькоpairT на прежней
DEVtemperature-half, оценкой на другой половине. Prior×4,status/modeT и их
readouts неизменны. Калибровочный и закрытый splits не предсказываются. Это не
допуск и не новое обучение основной модели; лимит600с,4GiB RSS.
[Регистрация H](runs/text_reward_transfer_fix_20261021/transfer_h/manifest.json).

**Подготовка V11, ещё без fit.** Read-only аудит подтвердил отсутствие трёх
классов TRAIN-формулировок: общее упоминание двух осей без отдельных witnesses,
далёкое правило кнопки рядом с противоположным правилом клавиши, известная
гипотетическая механика при неизвестной доступности. Это пробелы покрытия,
а не установленная единственная причина численных ошибок F.

Готовится замена291 natural TRAIN-view в существующих непересекающихся группах:
48 views из24timingcontrastpairs,32buttonpress/heldkey,32buttonheld/presskey,
64conditionalavailability,115aggregateaxes;146RU/145EN. Нейтральные bridge-фразы
также содержат обычные Enter/Return-глаголы. Это новые формулировки прежних
семантических групп, не новые независимые случаи. Labels/IDs/splits/spec и
все прочие JSONL строки, включая DEV/cal/sealed, должны остаться прежними.
Runtime32 не копируется в TRAIN и не меняется. Пороговые функции
_count_gate/calibration_gates/developmental_feasibility_gate AST-идентичны F.
Новый корпус материализуется только после структурных и семантических проверок.
[Аудит пробелов](runs/text_reward_transfer_fix_20261021/diagnostic_public_train_support_gaps.json),
[Неизменность gates](runs/text_reward_transfer_fix_20261021/gate_invariants_v11.json).

**Итог H: узкая подгонка G не перенеслась.** Actualexit0,185.43с вместе с
supervisor (скрипт182.48с), процессы12035/12047 завершены, группа пуста. ПикRSS
2 503 344 128байт;0optimizerupdates, исходныеF/G/source/данные неизменны.
Исходный F воспроизвёл score aliases302/882 и27criticalerrors. После замены
толькоpairMLP: aliases272/882; после отдельногоpairTrefit —226/882. Прежние
status/mode и27errors неизменны. PairT вырос1.49624→1.67880.

На всём уникальном TRAIN исходныйsame5363/6978 confidentcorrect и2argmaxerrors;
послеGsame4776/6978 и75errors (послеrefit4251/6978). На DEVscore по отдельным calibrated pair-
ответамsame653/882→518/882→455/882; это отдельно от полногоaliasdecoder.
PairNLL TRAIN .001275→.002678, DEVscore .002592→.005346; послеTrefit .005639.
Следовательно, успех G на выбранной подвыборке сопровождался ухудшением остального
TRAIN и DEV. G-веса не переносятся в новую модель; subset-onlypolish не выбирается.
Планируемая следующая проба использует свежую голову и направленные TRAIN-формулировки
V11 при прежнем F recipe, без экспериментальных импортов весов.
[Отчёт H](runs/text_reward_transfer_fix_20261021/transfer_h/output/report.json),
[Проверка процесса](runs/text_reward_transfer_fix_20261021/transfer_h/verification.json).

**Подготовка свежей пробы I (до запуска).** TRAIN-only V11 SHA9e4f8250…f8cd264
заменяет291view; исходный F — контроль. Сохраняются F recipe: seed20261024,
новая голова/пустойoptimizer,8эпох,allTRAIN15432views/2544groups,batch32,
baseLR1e-4,pairLR3e-4,dropout.1,selective×4 (17340presentations/epoch).
Новых самостоятельных семантических случаев нет. Ни F, ни G-веса не импортируются.

Меняются только проверенные классы TRAIN-языка и genericReturn-groundingfix.
DEV/cal/sealed строки прежние; public32 исходные. ProbeпредсказываетTRAIN/DEV
иpublic32, неcal/sealed; последняяпара остаётся закрыта дляprobe. Feasibility
прежняя: .99threshold,≥.8 по каждому срезу,0criticalerrors, достаточные группы.
Выбор по прежнему протоколу disjointDEVtemperature-half/score-half; проба
всегдаready=false. Заранее сохраняются все8epoch-enddiagnosticupdates
542…4336, без смены selection. Лимит3600с входит в прежний общийdeadline
21.10 21:19:09UTC. До старта требуются codechecks, corpuspreflight иsourcefreeze.
Подготовка не означает запуска, успешного допуска или улучшения игр.

**Регистрация стадии `probe_i` до запуска:** seed20261024, dataset`text_data_v11`, source`source_text_data_v11` (147файлов); recipeB:base1e-4/pair3e-4/drop.1, лимит3600с, [execution](runs/text_reward_transfer_fix_20261021/probe_i/execution.json). Пока подготовка; фактическое состояние проверяется по process.json/PID.

**I запущена21.10 в16:20:38UTC (19:20Москва).** Supervisor13123/actor13136
проверены по фактической идентичности, stage running; inputpins и147frozenfiles
подтверждены. До старта3912тестов прошли за501.45сpytest (502.57сwrapper),
Ruff clean,1прежнееGymwarning. В корпусе21432незакрытыхтекста,91712/91712
expectedmentions,max364/512токена; source/outputV10 не изменены.

На момент записи полных эпох/метрик качества I ещё нет. Runtime32, calibration,
sealed и пригодность новой головы ещё не оценены. Full fit/PPO не запущены;
переход к ним по-прежнему заблокирован до успешных проверок. Общий24часовой
deadline21.10 21:19:09UTC сохраняется.
[Проверки V11](runs/text_reward_transfer_fix_20261021/checks_v11.json),
[Identity audit](runs/text_reward_transfer_fix_20261021/text_data_v11_identity_audit.json),
[Запуск I](runs/text_reward_transfer_fix_20261021/probe_i/process.json).

**Итог I,21.10 16:56UTC.** Actualexit0,2165.31с(36мин05с),8эпох/4336updates.
Фактические процессы13123/13136 завершены, группа пуста,147исходников неизменны,
пикRSS2 997 583 872байта. Свежая initialSHA37827b7c…14bfa8d совпала с F,
optimizerstate0. ВыбранныйCP6 сохранён; дополнительно сохранены все8заранее
назначенныхepoch-enddiagnosticcheckpoints, включаяCP8.

Ни одна эпоха не прошла feasibility. CP6 выбран прежним rawDEVNLLfallback,
не по public32. Его score: supported5916/7656,unsupported36362/37680,
held2899/3402,press2065/3156,aliases330/882;15criticalerrors(status14,mode1).
Последняя эпоха8 также не прошла: supported5899/7656,held2960/3402,
press1978/3156,aliases407/882,10errors. Её сохранение не меняет критерий выбора.

На прежних32RU/EN игровых инструкциях CP6 прошёлтолькоtimingRU,1/32 целиком
(уF3/32). Покомпонентно: supported15/222,held6/176,press3/22,aliases0/88,
unsupported864/914,unknown15/16. Шесть явных ошибок: cameraEN(dx:+unsupported),
interactionRU/EN(Eheldвместоpress),shootingRU/EN(mouse:leftheldвместоpress),
uiRU(Downизвестнавместоunknown). Остальные неполные ответы включают отказ
от уверенного вывода; они не считаются правильными. Эти32 — compatibility
проверка, не независимая сертификация и не оценка побед в играх.

Вывод: V11 исправила конкретные пробелы входного TRAIN-языка, но качество
готового pipeline не улучшила. Воспроизводимость, прохождение unit-тестов
и снижениеloss не обеспечили перенос. Полное обучение награды иPPO не запущены,
численные пороги прежние. Calibration/sealed не предсказывались. Следующего
обучения не зарегистрировано; нужна дальнейшая диагностика.
[Результат I](runs/text_reward_transfer_fix_20261021/probe_i/output/probe.json),
[Проверка завершения и checkpoint](runs/text_reward_transfer_fix_20261021/probe_i/verification.json).

<a id="text-reward-structured"></a>
## 21.10.2026: совместное распознавание управления и групп

**ID:** `text_reward_structured_20261021`. Текущий статус: первая короткая проба
завершена, допуск не пройден; полный fit заблокирован. Ниже сохранены условия до старта.
Пользователь разрешил переделать схему, сначала подтвердить распознавание клавиш,
режимов и эквивалентности короткими проверками реальных инструкций, затем запускать
полный fit. Пороги не снижать. [Протокол](runs/text_reward_structured_20261021/protocol.json).

Контроль — завершённая I: на синтетическом DEV сырые правильные mode-ответы
held6549/6552, press6381/6414, однако конечная совместимость public1/32.
Из31 провала шесть содержат уверенную ошибку;25 имеют недостаточное покрытие
без записанной уверенной ошибки. Публичные сырые вероятности I не сохранены,
поэтому последние25 не доказывают правильный raw argmax.

**Гипотеза:** прямое совместное предсказание статуса, режима и представителя группы
уменьшит каскад отказов при сохранении прежних проверок ошибок. Прежний decoder
требует уверенности во всех отношениях между принятыми командами; joint head
учится36 назначениям вместо отдельных обязательных отношений каждой пары.
Представитель — минимальный индекс команды в TRAIN alias-группе; inference
предсказывает его только по тексту. Неуверенный представитель, конфликт режима
или некорректная ссылка не превращаются в оплачиваемую singleton-группу.
Порог.99 относится к совместному состоянию одной команды, не ко всей инструкции.

Исходные веса: оригинальный замороженный E5, новая собственная голова и пустой
optimizer. Веса F/G/H/I не импортируются. Подготавливается V12: естественные
TRAIN-формулировки вместо явной записи групп в скобках, прежние labels/IDs/splits
и побайтно прежние DEV/calibration/sealed. Исходные public32 не меняются и не
используются как обучающие записи. Меняются и архитектура, и TRAIN-язык;
опыт проверяет пригодность рецепта, а не выделяет эффект одного изменения.

Короткая проба: seed20261029, до8эпох, batch32, LR3e-4, максимум4200с.
Одна температура выбирается на отдельной половине DEV; checkpoint — на другой
половине по feasibility, затем joint NLL. Runtime32 не выбирает checkpoint
или температуру. Calibration/sealed в пробе не предсказываются; sealed не кодируется.
Допуск к full: прежние.99/≥.8 по каждому срезу/0criticalerrors и достаточно
независимых групп, плюс32/32 исходных игровых инструкций.

Full только после допуска: свежий seed20261030, до10эпох/7200с, тот же корпус
и рецепт, отдельные calibration и однократный sealed test после заморозки.
Старые численные пороги и доверительные границы сохраняются. Общий24часовой
счётчик не обнуляется: deadline21.10 21:19:09UTC. Процесс ограничивается по
реальным PID, RSS≤6GiB и свободному диску≥8GiB. Качество и фактические затраты
будут записаны после проверок; новая игровая политика/PPO здесь не запускаются.

**Запуск21.10 в17:30:21UTC (20:30Москва):** `probe_joint`, supervisor21706,
actor21716; фактические идентичности процессов совпали, inputpins и156файлов
замороженного source подтверждены. DatasetSHA fe1b9f9c…d1aa8ffe. До старта
3974теста прошли за501.32с, Ruff иfrontendchecks прошли; исходники не менялись
во время полного набора проверок. Пять функций оценки и допуска AST-идентичны
предыдущей версии. Новых метрик качества на момент запуска ещё нет.
[Процесс](runs/text_reward_structured_20261021/probe_joint/process.json),
[Проверки](runs/text_reward_structured_20261021/checks.json),
[Корпус](runs/text_reward_structured_20261021/dataset_preflight.json),
[Неизменность порогов](runs/text_reward_structured_20261021/gate_invariants.json).


**Результат первой пробы (21.10 17:46UTC):** реальные exit0,977.69с,
8эпох/3864обновления/123456показов TRAIN; checkpoint и156замороженных исходников
проверены, процессов и оставшихся участников process group нет. PeakRSS3.57GiB.
ВыбранCP6 по заранее зафиксированному DEV-правилу, ни одна эпоха не прошла допуск.
DEV score-half: supported481/7572, held0/3120, press0/3312, aliases0/906,
unsupported37221правильных из37229принятых при38538явных отрицательных командах;
8критических ошибок. Public **0/32**, уверенных верных supported0/222,
held0/176,press0/22,aliases0/88. Сырые public status928/1152; совместно верный
режим среди поддерживаемых48/222 (включает relative и ошибки статуса — это не
чистая held/press accuracy). Calibration и sealed не открывались. Полный fit
не запущен. Падение TRAIN loss не подтвердило распознавание реальных инструкций.
[Итог пробы](runs/text_reward_structured_20261021/probe_joint/output/probe_report.json),
[Реальные инструкции](runs/text_reward_structured_20261021/probe_joint/output/runtime.json),
[Проверка завершения](runs/text_reward_structured_20261021/probe_joint/verification.json).

Установлен разрыв формулировок:10344обычных TRAIN views заменены целиком,
DEV3600/3600 остался прежним. Простые mode-фразы `holding repeats the action`
и `once and release` имеют TRAIN0, DEV1464/1455 вхождений; аналогично RU.
Это наблюдаемый дефект покрытия корпуса, но его причинный вклад в провал пока
не измерен. Следующая разрешённая короткая проверка должна сначала отделить
ошибки усвоения TRAIN от переноса на DEV/реальные инструкции; пороги неизменны.

**Дополнительная диагностика перед повтором:** frozenCP6 без обновлений проверен
на32TRAIN и32DEV семантических группах. TRAIN отдельный выбор held/press491/594,
представитель695/696; при выбранной температуре уверенных правильных режимов0/594.
Следовательно, проблема не ограничивается новым языком DEV: режимы недостаточно
усвоены даже на обучающих данных. [Разложение ошибок](runs/text_reward_structured_20261021/probe_joint/postmortem/selected_train_dev_decomposition.json).

**Зарегистрированная малая проверка способности к обучению:** freshhead,64TRAINviews
из8контрастных пар (4timing+4alias), по4RU/EN views каждого участника;
тот же jointCE/LR3e-4/wd.01,batch32, seed20261029, максимум200updates/180с.
Оценка каждые20updates на этих же64записях, безDEV/public/calibration/sealed.
Критерий: отдельныеmode/representative raw100%, уверенное покрытие≥.8 безошибок,
productiondecoder held/press/aliases≥.8 и0criticalerrors. Это проверка усвоения
малой обучающей выборки; она не допускает full и не доказывает перенос.

**Направленная правка корпуса, до следующего запуска:** `mixed_repair/probe_joint`.
V13 смешивает5088исходных простых TRAINviews,5256concreteV12,4797natural и291
сохранённых directedviews. Для каждой TRAIN(group,language) восстанавливается
минимальныйvariant_index из исходных v11ordinaryviews; exactJSONL, без повторного
рендеринга. Все остальные строки,labels/IDs/splits/public32 неизменны. Контроль —
завершённая перваяjointпроба: тот же fresh seed20261029 иархитектура,8эпох,
batch32,LR3e-4,wd.01. Веса/optimizer начинаются заново. Цель — измерить эффект
устранения наблюдаемого разрыва простыхmode-конструкций; улучшение заранее не
утверждается. Допуск full прежний:DEVgate и32/32public, без использования
calibration/sealed. Предел4200с и общийdeadline21:19:09UTC не обнуляются.
[Протокол повторной пробы](runs/text_reward_structured_20261021/mixed_repair/protocol.json).
Подготовка данных и проверки кода ещё не являются состоявшимся запуском.

Малая проверка завершиласьexit0 за24.05с:200updates,64TRAINviews. Режим109/128,
уверенно верный52/128, представитель128/128. Допуск способности к усвоению не
пройден; веса диагностические. JointNLL0.01586 скрывает conditionalmodeNLL0.27707:
режим размечен лишь в128из2304слотов. Это показывает недостаточное усвоение
режима, но200steps не доказывают невозможность архитектуры.
[Малая проверка](runs/text_reward_structured_20261021/probe_joint/postmortem/capacity_micro/result.json).

Перед большим повтором зарегистрирован парный micro-check на тех же64TRAINviews,
той же начальной голове, seed, batch и200updates/180с. Единственное изменение —
loss как сумма средних conditionalNLL: status по всем командам, mode по применимым
дискретным, representative по применимым командам. Это не изменение порога,
разметки или prior-correction; проверяется распределение обучения между задачами.
Production decoder и.99 неизменны. Успех/провал оценивается по тем жеTRAINкритериям,
не даёт допуска полного обучения; DEV/public/calibration/sealed не используются.

Парная200-stepпроверка также не прошла, но режим улучшился109→120/128,
уверенное покрытие52→67/128; groups128/128 в обеих, acceptederrors0.
Результат не подтверждает готовность. Чтобы отличить недостаток числа обновлений
от неспособности усвоить64примера, до запуска зарегистрировано **600updates**
для каждой из тех же двух freshветвей, ≤180с наветвь. Данные,инициализация,
порядок иoptimizer прежние; это сравнение двух кривых при большем одинаковом
числе шагов, не продолжение основной модели и не допуск full. Критерии прежние.

Обе600-stepпроверки прошли: исходная впервые на400updates, conditional-balanced
на360. На600обе имеют128/128уверенных режимов,128/128представителей, held64/64,
press64/64,aliases16/16,0errors. Первые200точно воспроизвели ранее сохранённые
кривые иSHA. Следовательно, предел200был недостаточен; невозможность модели не
подтверждается. [Исходная](runs/text_reward_structured_20261021/probe_joint/postmortem/capacity_micro600/result.json),
[Раздельные средние](runs/text_reward_structured_20261021/probe_joint/postmortem/capacity_balanced_micro600/result.json).

**Уточнение рецепта mixed_repair до запуска:** кроме смесиV13 используем
`conditional_population_mean_v1`: среднееstatusNLL по всем36командам + среднее
modeNLL по применимым дискретным + среднееrepresentativeNLL по применимым.
Изолированная micro-пара показала более быстрое усвоениеmode на одномTRAINнаборе;
это инженерный выбор, не доказательство переноса. Следующая большая короткая
проба меняет и корпус, и objective, поэтому общий эффект нельзя приписать одной
правке. Настоящий jointNLL остаётся отдельной диагностикой и прежним критерием
calibration/selection. Пороги и модель inference не меняются. Полный fit всё ещё
требует успешной проверкиDEV и32/32реальных инструкций.

<a id="fresh-all16-v3"></a>
## fresh-all16-v3 — 23 октября 2026: новый цикл, подготовка

Пользователь утвердил выполнение полного плана и дополнительные 24 часа на этом
Mac. Новый цикл `runs/fresh_all16_v3_20261023` связан с прежним v2 и серией
`text_reward_structured_20261021`; прежние результаты и сроки сохранены.
Общий новый срок записан один раз в `cycle_state.json`; включает подготовку,
проверки, награду, пробы, пилоты, основное обучение и оценки, резерв финала 2 часа.

Первая гипотеза: смесь V13 и `conditional_population_mean_v1` улучшат распознавание
режима управления без снижения порогов. Исходная модель — новый head и optimizer,
исходный замороженный E5; прошлые веса не используются. Контроль — прежняя
неуспешная V12-проба; одновременно меняются смесь TRAIN и objective, причинный
эффект отдельно каждой правки этим опытом не оценивается. TRAIN/DEV/calibration/
sealed и 32 фактические инструкции сохраняют прежние цели и разбиение. Corpus SHA
`26eeb9fce25c020688437ece4588820b516b1f14c66b1eedd3018aaf723c707e`.

Короткая проба: seed20261029, до8эпох, batch32, LR0.0003, до4200с. Выбор по DEV,
затем runtime32; calibration/sealed закрыты. Допуск: прежние DEV gates,
confidence .99, coverage≥.8 по обоим знакам/held/press/aliases, ноль критических
ошибок, достаточные независимые группы; runtime32/32 без ошибочных решений.
Full только после успешной пробы: новый head/optimizer seed20261030, до10эпох/
7200с, тот же рецепт, calibration/runtime и однократный sealed после заморозки.

Снимок158файлов совпадает с исходниками ранее успешных проверок3989pytest/Ruff/
frontend. Интеграция structured-комплекта с подготовкой RL дорабатывается отдельно;
перед PPO будут выполнены новые общие проверки. Визуальный оценщик v2 допускается
после проверки идентичности модели/протокола. Политика ещё не создана. Пилотные
seeds3176, main3177; объёмы, gates выбора и финальная проверка100% каждой из16игр
зафиксированы в [протоколе](runs/fresh_all16_v3_20261023/protocol.json).

Статус: подготовка; [manifest](runs/fresh_all16_v3_20261023/manifest.json),
[проверки текстового снимка](runs/fresh_all16_v3_20261023/checks_text.json),
[проверка корпуса](runs/fresh_all16_v3_20261023/dataset_preflight.json).

Короткая проба запущена 23.10 в 01:27:51 МСК: подтверждены supervisor PID30772
и отдельная группа actor PID30779. Входы и158файлов снимка проверены; загрузка
исходного E5 локальная. Контроль ресурсов: прогноз6ГиБ сверх8ГиБ запаса диска,
лимит RSS группы6ГиБ, резерв памяти системы и предел4200с. Фактические ресурсы,
exit code и результат сохраняются в [process.json](runs/fresh_all16_v3_20261023/probe_joint/process.json).
Сейчас готовятся признаки; чисел качества ещё нет. Замороженные оценки не открыты.

Повторное использование визуальной награды подтверждено отдельной
[проверкой](runs/fresh_all16_v3_20261023/visual_reuse_verification.json):
240/240 ячеек, HUD203204 и progress170128 без ошибок; это sampled-проверки
награды, не качество политики и не исчерпывающая проверка всех уровней.
Пиксельные кадры преобразуются той же `style_pixel_art`, что в политике.

**Результат короткой пробы v3:**8эпох,3864updates,123456TRAIN-предъявлений,
1000.53с, peak RSS группы3214000128байт. Процесс завершёнexit0, actor иsupervisor
отсутствуют, неизменность158файлов иSHA выбранного checkpoint проверены отдельно.
DEV выбрал эпоху6, но допуск не пройден: уверенный охват supported1013/7572,
held33/3120,press48/3312,aliases0/906. Большинство уверенных supported здесь
составляют relative axes, поэтому общий охват не доказывает усвоение held/press.

Фактические инструкции:0/32 прошедших; supported8/222,unsupported838/914,
held3/176,press0/22,aliases0/88,unknown16/16. В goal_choice/RU применимая `key:a`
получила ошибочное unsupported. Calibration/sealed не использовались;
полный fit и PPO не начаты. Все неудачные снимки и признаки сохранены.
[Отчёт пробы](runs/fresh_all16_v3_20261023/probe_joint/output/probe_report.json),
[проверка завершения и результатов](runs/fresh_all16_v3_20261023/probe_joint/verification.json).

Следующий зарегистрированный диагностический шаг не обучает веса: выбранный
замороженный checkpoint на ограниченной воспроизводимой выборке cachedTRAIN и
DEV-score, отдельно pure/mixed held/press и формулировки. Он должен отделить
недостаточное усвоение TRAIN от переноса. Настройка по calibration/sealed запрещена;
направленная обучающая правка ещё не выполнена.

Frozen-разбор завершён за9.15с без обучения иE5forward:62TRAIN-группы/380views
и32DEV-score-группы/192views. В TRAIN pureheld255/256,purepress227/228,
но mixed458/588;DEV pureheld158/162,purepress146/150,mixed208/306.
Вmixed находятся130/132 ошибокрежимаTRAIN и98/106DEV. При .99 jointconfidence
TRAINmixedheld0/294,press0/294;DEVmixedheld1/162,press1/144. Ошибокпринятых
решений в этих срезах0. RepresentativeTRAIN1363/1364,DEV754/768.
[Замороженная диагностика](runs/fresh_all16_v3_20261023/probe_joint/postmortem/selected_mode_generalization.json).
Выборка ограничена; вывод — проблема mixedконтекста уже наTRAIN, не доказанная
невозможность модели или исключительно новый wording.

**Зарегистрирована единственная направленная ML-доработка нового цикла:**
[протокол](runs/fresh_all16_v3_20261023/repair_local_context/protocol.json).
Меняется только агрегация локального контекста команды в текстовом head;
физические команды, корпусV13, frozenE5, objective, decoder и пороги прежние.
Гипотеза: локальное внимание по расстоянию до упоминания команды поможет связать
held/press с нужной командой в смешанной инструкции. Это проверяемая гипотеза,
не установленное улучшение. Резерв этой доработки2часа включён в общий срок;
дальнейшие ML-повторы нового цикла не разрешены протоколом.

До новой8epochпробы — matchedfreshmicro v7/localcontext:32mixedTRAIN-группы,
четыреview/группу, seed20261029,batch32,LR.0003,wd.01,600updates/ветвь,
≤180с/ветвь. Общие параметры и порядок данных совпадают, optimizerпустой.
Micro-допуск: TRAINcoverage≥.8 при.99 отдельноheld/press/aliases,0acceptederrors;
это не DEV/runtimeдопуск. DEV/calibration/sealed иpublicне используются.
Полная8epochпроба возможна только после micro и проверок кода;fullfit иPPO
по-прежнему требуют первоначальных qualitygates. На момент регистрации
изменение head и pairedmicro ещё не завершены.

**Парная micro-проверка завершена:** обе ветви600updates, первая наблюдённая
точка прохождения200steps у обеих. При .99 held244/244,press196/196,aliases32/32,
0acceptederrors; rawconditionalmode440/440 в обеих. Общие начальные параметры
побайтно равны, RNG и порядок600minibatches совпадают, optimizerпустой.
Оба процессаexit0 и отсутствуют, SHAcheckpoint/sourceпроверены.
v7 fit23.72с/peakRSS555696128B, v8 fit26.90с/603455488B; пара61.70с.
[Итог](runs/fresh_all16_v3_20261023/repair_local_context/micro/result.json),
[независимая проверка](runs/fresh_all16_v3_20261023/repair_local_context/micro/verification.json).

Обе модели запомнили128TRAINviews; преимущество v8 и причинность meanpooling
не установлены. Capabilitygate разрешает только следующую8epochпробу с
исходными DEV/runtimegates. Модель v8 имеет534919параметров (+16514);
новый forward использует только E5tokens, mentionmask и расстояние доупоминаний.
Метки и игровое состояние не поступают вforward.42head/artifactтеста иRuff
прошли; сейчас выполняется полный source-boundpytest/Ruff/frontend передпробой.

Полный набор после v8:4008pytest passed (один прежний gymwarning),Ruffexit0,
frontend73/73; source350файлов неизменен. [Общая проверка](runs/fresh_all16_v3_20261023/checks_full.json).
Новая8epochпроба v8 запущена23.10 в02:02:04МСК,seed20261029,batch32,LR.0003,
4200с; источник350файлов заморожен. Head новый, optimizerпустой; переиспользуется
только аутентифицированный TRAIN/DEV-кеш того жеE5 иV13, не веса старогоhead.
Пороги и calibration/sealed split закрыты по исходному протоколу.
[Запуск](runs/fresh_all16_v3_20261023/repair_local_context/probe_joint/execution.json),
[проверка живого процесса](runs/fresh_all16_v3_20261023/repair_local_context/probe_joint/launch_verification.json).
Результатов качества v8 пока нет; полное обучение оценщика и PPO ещё не начаты.

Дополнительная TRAIN-only проверка покрытия обнаружила пробел V13: комбинация
явных unknown-команд с supported-командами и запретом всех неназванных команд
отсутствует (0групп/0views/0семейств). Вconditional иpartial_unknown по231группе/
1386views; все86778 неназванных решений вместе размеченыunknown, ниодногоunsupported.
Всего462группы/2772views/31семейство с такой явной неопределённостью; во всех
ровно одна явная unknown-команда. РеальнаяUIинструкция содержит другую композицию:
неизвестны8направлений, известныTab/Enter/Escape, остальные командызапрещены.
[Подсчёт](runs/fresh_all16_v3_20261023/repair_local_context/conditional_corpus_coverage.json).

Это искусственная корреляция и ограничение покрытия, возможное объяснение прежнего
UIunsupported0/50. Причинность конкретного предсказания не установлена. Корпус,
исходники и текущая v8проба этим анализом не изменены; дополнительных fits он
не разрешает. Calibration/sealed и игровой test не использовались.

**Результат v8-пробы:**8эпох,3864updates,123456TRAIN-предъявлений,704.98с,
exit0; actor иsupervisor завершены,350файлов снимка ивходы неизменны.
Выбранаэпоха8. DEV-score: supported7080/7572,unsupported38372/38538,
held2937/3120,press3019/3312,aliases791/906. Coverageвсехсрезов≥.8,
но5criticalerrors (3ошибочныхsupportedрешения и2режима) запрещаютдопуск.
Результат runtime: **9/32инструкций**;supported179/222,unsupported838/914,
held136/176,press17/22,aliases56/88,unknown7/16. Это существенныйпрогресс
относительноV7, но полногоfit/PPO-допуска нет. Calibration/sealed не открыты.
[Проверка завершения и качества](runs/fresh_all16_v3_20261023/repair_local_context/probe_joint/verification.json).

Frozendecomposition сохранённых32runtime-logits не обновляетвеса: rawjointcorrect
supported213/222,unsupported866/914,unknown1/16; при.99 rawwrong3/46/9соответственно.
Unknownвdecodedответе можетбыть abstention,а не вернымrawвыводом; этичисла
нельзя смешивать сruntimecoverage. [Разбор](runs/fresh_all16_v3_20261023/repair_local_context/probe_joint/postmortem/runtime_failure_decomposition.json).

Найдена отдельная детерминированнаяошибка входа: lexicalgrounding не распознавал
«пробела» вactualtimingRU,такчто видел только позднейшее«Отпускайпробел».
Исправлены пять падежныхформ единственногочисла; тесты независимо проверяют
обаспанa[65:72]и[106:112], границыслов, offsets,paddingи неизменныеwheelanchors.
133целевыхтеста иRuffпрошли. Обучение послеисправления незапускалось;
новый preprocessingтребует новойsourceidentity. Снимокv8сохранённеизменным.
[Проверки исправления](runs/fresh_all16_v3_20261023/repair_local_context/grounding_fix_checks.json).

ДополнительныйTRAIN-аудит:268compactWASD/A-Dviews в134семантическихгруппах
всегда содержат явное bracket-раскрытиепар; naturalcompact безраскрытияпарнет.
Это ограничение переноса наестественныеинструкции,не доказанная причина всех
aliasошибок. RUinteractionunsupported2/27 нельзяобъяснятьconditionalслучаем:
вэтойинструкцииусловности нет, аTRAINсодержит5407RUclosed-worldviews,втомчисле
1369с«остальные…неподдерживаются». Причинаэтогопереноса отдельнопоканеустановлена.

ОднаразрешённаяML-доработкановогоцикла использована. Пользователю направлен
вопросодополнительнойTRAIN-коррекции икороткойпробе втомже24часовомбюджете.
До ответа новыеfits незапускаются; игровойtestзакрыт,основнаяполитиканесоздана.

Проверка исправления Space на замороженном выбранном head: старая обработка
в точности воспроизвела сохранённые logits двух timing-инструкций. При одинаковых
E5-признаках новая маска добавила пропущенное упоминание «пробела»: уверенность
правильного press-ответа RU выросла с 0,938607 до 0,962173. Это ниже порога 0,99,
поэтому итог по-прежнему unknown. Английский контроль не изменился
(press, 0,999482). Обновлений весов — 0, checkpoint до/после совпадает по SHA256;
calibration/sealed не оценивались. Это подтверждает влияние входной ошибки,
но не прохождение допуска и не достаточность одного исправления.
[Замороженное сравнение](runs/fresh_all16_v3_20261023/repair_local_context/probe_joint/postmortem/space_grounding_counterfactual.json).


<a id="factorized-text-v2"></a>
## 23.10.2026: раздельный текстовый оценщик и награда v2

**ID:** fresh-all16-v3/factorized-v2. **Статус:** реализация и подготовка;
обучение ещё не начато. Пользователь утвердил новый полный протокол, заменяющий
предыдущее ограничение одной доработки для этой переработки. Общий срок цикла
сохранён: 23 октября 22:25 UTC / 24 октября 01:25 Москва; прошлые затраты не обнуляются.
[Регистрация](runs/fresh_all16_v3_20261023/factorized_v2/protocol.json).

**Смысл:** устранить смешение применимости команды в игре с доступностью в
текущем состоянии, сохранить частично известные свойства и различать нехватку
сведений в тексте и отказ оценщика. Условия остаются в тексте; отдельного
извлечения или проверки условий нет. Замороженный E5 и локальный контекст
сохраняются, головы применимости/режима/связей становятся раздельными.

**Исходные веса:** новый head, пустой optimizer; веса прошлых head и политик
не переносятся. **Данные:** новый V14, 4144 смысловые группы × 6 вариантов;
TRAIN2544 / development600 / calibration400 / sealed600. Development заранее
разделён 300/300 на выбор checkpoint и температур. Близкие варианты и переводы
не пересекают split; реальные32 инструкции служат совместимостью, не независимым test.

**Протокол:** короткая TRAIN-проверка способности учиться, затем probe seed20261029,
8эпох, batch32, LR0.0003, максимум4200с. Только при успешных development/runtime
проверках — fresh full seed20261030, до10эпох/7200с, calibration и однократный
sealed после выбора. Пороги99%, охват80%, ноль критических ошибок и требования
precision/нижней95%границы99% сохранены, включая реально оплачиваемые решения.

**Награда v2:** база0.0005R и доплата0.0015R на группу, общий положительный
предел0.01R. Отдельное нажатие — один интервал с наблюдённым отпусканием;
удержание — минимум два интервала одной физической команды. Повторный штраф
заменяет разовый:0.01R/H за шаг с неприменимым управлением. Коэффициент полный
на циклах1–32, убывает до нуля к64. Историческая награда остаётся отдельной версией.

**Игровой контроль после допуска:** три80-цикловые ветки seed3176 с одинаковыми
свежими CP0 — zero, text, text_penalty; по128эпизодов/цикл. ОценкиCP0/32/64/80,
оба оформления и режима действий; после64 текстовая добавка выключена.
Переход к основной политике допускается только по игровым победам и широте
успехов, не по loss или сокращению неприменимых действий. Время всех запусков
входит в оставшийся общий бюджет; возможность полного плана ещё не измерена.

**Ограничение сравнения:** изменение разметки и инструкций является новой версией
протокола. Его результаты нельзя выдавать за улучшение прежних9/32 на старом.
Копии исходных инструкций и спецификации сохранены в каталоге регистрации;
старые веса, отчёты и условия не изменяются. Игровой test не открыт.


**Техническая готовность новой версии:** общий pytest4100passed, один прежний
warningGymnasium. После исправления только порядка двух импортов повторены
загрузчики/артефакты/restore, Ruff и75frontend-тестов; первоначальный отказRuff
и оба набора хешей сохранены. [Проверки](runs/fresh_all16_v3_20261023/factorized_v2/checks_full.json).
V14 seed20261029:24864текста,4144группы;9уникальных полных публичных mappings
исключены. Независимая TRAIN/DEV-проверка152004упоминаний в18864текстах прошла
без пропусков; max477/512токенов, прогноз признаков2855871744байта.
[Предварительная проверка](runs/fresh_all16_v3_20261023/factorized_v2/preflight.json).

Следующая короткая TRAIN-only проверка зарегистрирована:32группы/64RUENviews,
seed20261028, до1200updates, batch16, LR0.0003,540с внутри600с supervisor.
Первые100updates — простые примеры, далее смесь условий, эквивалентности и мыши.
Проверяются отдельно held/press/aliases при confidence.99, coverage.8 и нуле
ошибочных уверенных решений. Это проверка способности запоминать TRAIN;
её веса не переносятся, calibration/sealed и runtime32 не используются.
Источники замораживаются после проверок, фактический старт/exit и результат
фиксируются отдельным процессным receipt в каталоге micro.


**Первая TRAIN-only способность:** micro завершён,300updates/36.69с,actualexit0,
оба процесса отсутствуют, checkpoint и замороженные источники проверены.
Raw held124/124,press108/108,aliases16/16; при confidence.99 уверенно правильных
held124/124,press102/108,aliases15/16. Ни одного ошибочного уверенного ответа.
[Проверка](runs/fresh_all16_v3_20261023/factorized_v2/micro/verification.json).
Это64TRAINviews, не перенос или допуск32инструкций; веса дальше не используются.

**До полной пробы обнаружен недоделанный участок корпуса:** контрасты меняли
только режим; отсутствовали пары полноты списка, условия повторялись во всех
вариантах, компактныеWASD/A-D всегда сопровождались индивидуальными bindings.
Начата ограниченная коррекция этих пунктов согласованного плана с независимым
аудитом естественных формулировок. Подготовленный V14 и результат micro
сохранены неизменными; новая версия данных и проверок будет в отдельном
каталоге corpus_r2. Длинная проба на неполном корпусе не запускалась.


**Корпус R2 подготовлен без fit:**4144группы×6, те же квоты splits; SHA256
records772e7258671b775dea669c9d98fde1ce25d5307dcd3fce34b838a021a227f85c.
В TRAIN562completeness-контраста и710mode-контрастов; в каждой ролиDEV38/112
соответственно. Добавлены compact-only направления с явным соответствием,
естественные функции без номеров, отсутствие timing-clause в части представлений,
варианты с/без условия и известные aliases рядом с неизвестными межгрупповыми
связями. Формулировки компактных и pair-unknown привязок действительно разделены
по splits, а не только названы разнымиfamilyIDs. Независимый ML-аудит проверил
эти инварианты на15грамматикахRU/EN; незакрытых существенных замечаний не осталось.
Авторские31целевой тест иRuff прошли; полный набор на новых источниках выполняется.
[Корпус](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r2/data_v14/manifest.json),
[предварительная проверка](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r2/preflight.json).

Реальный TRAIN/DEV preflight:18864текста,147108независимо заданных mentions,
0потерь,max450/512токенов; прогноз признаков2699011584байта. Calibration/sealed
в этой проверке не использованы. Предыдущие данные, source и успешный micro
остаются в parent-каталоге; R2 потребует нового свежего micro перед probe.


R2 общий набор завершён:4110Python-тестов, Ruff и75frontend прошли на неизменных
источниках. [Проверки](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r2/checks_full.json).
Свежий microR2 запущен23.10 в10:09:57Москва по тому же зарегистрированному
TRAIN-only рецепту; прежние веса не загружаются. Все калибровочные/закрытые
проверки и игровые данные исключены из этого fit.
[Процесс](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r2/micro/process.json).


**MicroR2 прошёл:**500updates/56.78с,actualexit0,оба процесса отсутствуют,
checkpoint и368замороженных файлов проверены. При confidence.99 held116/116,
press108/108,aliases16/16; применимость уверенно правильна2289/2304.
Ошибочных уверенных ответов0; это64TRAINviews, не допуск награды.
[Проверка](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r2/micro/verification.json).

Следующая probeR2 зарегистрирована по исходным условиям: новый head/optimizer,
seed20261029,8эпох,batch32,LR0.0003,полныйV14R2TRAIN/DEV,до4200с всего.
Выбор checkpoint итемператур раздельно поDEV, затем32реальныеинструкции;
calibration/sealed не используются. Веса micro не переносятся. Все входы и
прошедшие проверки фиксируются вexecution.json; источник frozen sourceR2.

**Проба R2 запущена23.10 в10:12:24Москва.** Фактический actorPID64415,
владелецPID64404; command/source/data pins сохранены в
[process.json](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r2/probe/process.json).
Это запуск нового head с пустым optimizer, без переноса micro. Качество пока
не установлено; calibration/sealed и игровой test не использовались.

Дэшборд проверен в браузере на1440×1000 и390×844: горизонтального переполнения
и ошибок консоли нет; счётчик updates менялся823→847→872 без перезагрузки.
Показаны реальный этап, эпоха и общий остаток бюджета. Проверка относится
к подготовке; сравнение checkpoint игровой политики ещё ожидает её появления.
[Браузерная проверка](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r2/browser_preparation_check.json).

**Проба R2 завершена с отрицательным допуском:**8эпох,3816updates,122112
показов TRAIN-примеров,1339.87с,actualexit0. Actor и supervisor отсутствуют,
368файлов frozen source, входы, checkpoint и отчёт проверены. ВыбранаDEV-эпоха8:
применимость raw64683/64800, mode10325/11592, pair36572/36720. Эти показатели
не заменяют допуск:20критических ошибок, охват применимых7027/11592,
неприменимых27424/27624, оплачиваемого удержания0/3720 и нажатия0/2934;
эквивалентность169/516. Публичная совместимость0/32 по новому протоколу,
несопоставимому напрямую с историческими9/32. Calibration/sealed не открывались,
fullfit/PPO запрещены по этому результату.
[Проверка завершения](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r2/probe/verification.json).

Диагностический проход288TRAIN/DEV текстов без загрузки E5, обновлений и
изменения checkpoint подтвердил: raw mode наTRAIN576/576, наDEV-temperature
628/684, DEV-selection536/588. Температура mode2.6607 устраняет уверенные
решения при.99 даже наTRAIN; её нельзя просто уменьшить ради охвата. На выборке
DEV-selection166предсказанных применимых команд блокируются неуверенными
внешними pair-связями. Причина переноса ещё исследуется.
[Диагностика](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r2/probe/postmortem/train_dev_diagnostic.json).

Отдельная независимая сверка выявила дефект контракта decoder: функциональные
эквиваленты ошибочно требовали одинакового физического режима. Ограничение убрано,
decoder версионирован; clique/separation и проверка физического режима сохранены.
51целевой тест иRuff прошли. Корпус с индивидуальными режимами участников
группы готовится как новая ревизия без измененияR2. Это исправление не выдаётся
за объяснение0/32: фактические runtime-группы были режимно-однородны.

Подготовка нового игрового launch adapter выполнена без создания executableplan:
три80cycleветки, замер полной оценки, расчёт полного128/160/192horizon и
неизменный общий deadline.63целевых теста бюджета/restore иRuff прошли;
новые policy-исходники потребуют общего набора перед игровым запуском.

**R3 зарегистрирована до fit** с исходным deadline1792794300. Диагностика
TRAIN/DEV обнаружила lexicalshortcut:9668употребленийduration/длительность
вTRAIN связаны только с unspecified,0—с известным режимом. Описание функции
между командой и режимом отсутствовало в66936TRAINbindings, но присутствовало
в1243/1305bindings DEVgrammar08 каждого языка. Вмешательство только в local
features сломало1/6TRAINконтролей при переносе собственных function tokens
перед mode; это ограниченное причинное свидетельство, без переencodingE5.
[Диагноз и границы вывода](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r2/probe/postmortem/local_binding_diagnosis.json).

R3 сохраняет22220текстов и целейR2 точно;1440views относятся к240heteromode
groups, ещё1204изменённыхTRAINviews покрывают cue/function-position при
неизменных целях.46тестов корпуса,Ruff и77frontendпрошли; полный набор
источников пока выполняется.4144×6и splits сохранены;полные mappings32публичных
инструкций исключены9уникальными mappingIDs. SHA256R3records:
`611916b90ae7f5e794946f8fa93c22027c596c6d51382c187ac81f3e4c08c9b0`.
[Протокол](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r3/protocol.json).

Статическая проверка R3:147108/147108упоминаний TRAIN/DEV переданы, максимум
450токенов, прогноз features2744742912байт, свободно26019946496байт при
расчётном пике14079420416+резерв8589934592байт. Новых весов и optimizer нет.
Свежий micro будет использовать32TRAINgroups×все6views=192 и отдельно проверять
каждый режим с/безdurationcue при прежних.99/.8/0ошибок; его веса не перейдут
вprobe. Проба и full сохраняют зарегистрированные seeds, эпохи и ограничения.
[Preflight](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r3/preflight.json).

**R3: общий набор прошёл**4140Python-тестов,Ruff,77frontend,actualexit0;
367кодовых/конфигурационныхфайлов не изменились, процесс и группа завершены.
Независимая сверка147798bindings и9432TRAIN/DEV contrast-views не выявила
новых противоречий (без обучения и model inference наCAL/SEALED).
[Проверка исходников](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r3/checks_verification.json).

Следующий microR3 зарегистрирован: seed20261028,новый head/optimizer,
32TRAINgroups×все6views=192; до1200updates/600с всего, batch16,LR0.0003.
Отдельный80%охват при.99 проверяется по каждому timingmode с/безdurationcue,
при0критическихошибок. Простые формы, условия, pointer иheteromodealiases
включены; веса ни вprobe,ни вfull не переносятся.

**MicroR3 прошёл:**192TRAINviews,600updates,76.46с,actualexit0; actor и
supervisor отсутствуют,368frozenfiles/входы/head проверены. При.99 все
режимы1392/1392 и каждый cue×mode-срез прошли, критических ошибок0.
Это проверка способности учиться наTRAIN, не допуск награды. Веса не переносятся.
[Проверка](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r3/micro/verification.json).

Следующая пробаR3: свежийhead/optimizer,seed20261029,8эпох,batch32,LR0.0003,
до4200с включая features и32runtimeинструкции; толькоTRAIN/раздельныеDEV
используются для fit/выбора. Calibration/sealed не открываются.

**ПробаR3 запущена23.10 в11:09:33Москва:** свежие веса/optimizer,
8эпох по указанному рецепту.
[Процесс и входныеpins](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r3/probe/process.json).
**ПробаR3 завершена:**1352.91с,8эпох/3816updates/122112TRAIN-предъявлений;
выбранаDEV-эпоха8. Фактическийexit0,actor/supervisor отсутствуют,368исходных
файлов и входные хеши совпали. Runtime1/32,допуск не пройден.
Сырые ответы: применимость64707/64800,режим11099/11592,пары36477/36720.
Составные решения при.99: удержание878/3720,нажатие1003/2898,
эквивалентность27/516;51критическая ошибка,0ложных объединений/2ложных
разделения среди29принятых групповых решений. Высокая общая точность пар
не доказывает освоение редкого класса эквивалентности. Полныйfit/PPO
не запускались; calibration/sealed моделью не оценивались.
[Первичная проверка](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r3/probe/verification.json).
Продолжается разборTRAIN/DEV без обновления весов и использования runtime
для настройки.

**Дополнительный структурный QC перед full:** tokenizer/matcher проверили
входы calibration иsealed без E5-forward/head/предсказаний/изменениятекстов:
45012/45012упоминаний переданы, максимум464/463токена соответственно при
лимите512. Sealed-входы были токенизированы только для этой проверки;
закрытая модельная оценка не выполнялась и не использовалась для выбора.
[Структурный QC](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r3/holdout_structure_preflight.json).

**R4 зарегистрирована до fit:** исходный общийdeadline1792794300 сохраняется.
Гипотеза после frozenTRAIN/DEV-диагностики R3 — недостаточное разнообразие
положительных связей и перенос сжатых форм. Предусмотрена замена только
576TRAIN semanticgroups на192 relationalclusters:48 заранее выбранных
физических пар×4контекста×same/different/unspecified. Все три состояния
одной ситуации учитываются одной независимой единицей. Шесть RU/EN views
каждой группы сохраняются; весь R3 DEV/CAL/SEALED и остальные TRAIN остаются
неизменными. Уточнённая область неизвестных связей задаётся только в новых
TRAIN-формулировках. Архитектура,loss,.99/.8/0ошибок и допускающие проверки
не меняются. Это подготовка; E5/head-fit и PPO не запускались. После
структурных проверок и общего набора — свежий micro, затем probe20261029;
full20261030 только после успешного соответствующего probe.
[ПротоколR4](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r4/protocol.json).

**R4 подготовлена без fit:** SHA256records
`73d16fac96e04473e09e75963e26a5abd26e22a27e165677108f92919623da99`.
Первичная независимая сверка подтвердила21408неизменныхviews,3456новыхTRAIN
и все9600heldoutviews побайтно по parsedrecorddicts без изменения.
TRAIN содержит1176независимых единиц:984сохранённых парныхкластера и192тройки.
Доля same15360/287184=5,35%; фиксированные navigation-пары дают456/1184=38,51%
положительных independent(group,pair)единиц против75% уR3.
Все3456новых текстов проверены отдельно:18срезов family×language×relation
по192views;1152сравнения физическихописаний S/D/U совпали. Применимость
и режим не подменяют relation, неизвестные связи не получают functionIDs.
[Формальная сверка](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r4/corpus_integrity_verification.json),
[семантический аудит](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r4/semantic_review.json).

Tokenizer/matcherQC TRAIN/DEV прошёл161160/161160упоминаний, максимум450токенов;
прогноз featurecache2667642624байта, дополнительныйpeak13925219840байт
плюс8589934592байта свободногоостатка помещается в измеренные22620831744байта.
44TRAINgroups/264views выбраны заранее дляmicro:32сохранённыегруппы
и4новыхtripletclusters целиком. TRAIN-capability пороги.99/.8/0ошибок,
максимум1200updates/600с сохраняются. E5/headweights не загружались.
[Preflight](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r4/preflight.json).
42целевыхтеста,Ruff и78frontend уже прошли; общий набор запущен23.10
в12:03:48Москва, завершение пока не подтверждено.
Визуальныйоценщик240/240 переиспользуется после повторной проверки исходников,
артефактов и неизменного primary-composer; нового визуальногоfit нет.

**R4: общий набор прошёл:**4146Python-тестов (575,35с),Ruff и78frontend,
фактическийexit0;367файлов исходников/конфигураций/тестов не изменились.
Helper74495/егогруппа/observer74501 завершены,execsession14978 закрыта.
[Проверка](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r4/checks_verification.json).
Свежий micro зарегистрирован: seed20261028,пустойoptimizer,44группы/264views,
4relationaltriplets целиком; каждая изsame/different/unspecified проверяется
отдельно при.99 иохвате.8 с0критическихошибок. До1200updates/600с,
первые100updates на простых сохранённыхTRAINпримерах; веса не переносятся.

**MicroR4 запущен23.10 в12:15:03Москва:** supervisor75554,actor75567
подтверждены фактическими PID/командами. Новыйhead/optimizer,seed20261028;
[входы и процесс](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r4/micro/process.json).
До запуска выполнен структурный QC тех же6000CAL/SEALEDтекстов без E5/head:
45012/45012упоминаний,максимум464/463токена. Это повторная проверка входов,
закрытая модельная оценка по-прежнему не выполнялась.

**MicroR4 завершён с отрицательным допуском:**1200updates/146,24с,actualexit0;
actor75567/supervisor75554 отсутствуют,368замороженныхфайлов и входы проверены,
E5 не изменился. НовыефокальныеS/D/Uответы поargmax20/24,17/24,24/24;
при.99 приняты0/24,0/24,24/24. Критическихошибок0.
Общиепары5629/5640верны,aliases229/264принятыверно(.8674),режимы1770/1770
argmaxверны. Все11rawpairошибок — новыефокальныеS/D. Поэтомупрежний
aggregateсрезскрылбыдефект; добавленный prospectiveсреззакрылпереходкprobe.
Полныйprobe/full/PPO не запускались,calibration/sealed/runtimeмоделью
не оценивались. Веса micro не переносятся.
[Проверка](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r4/micro/verification.json).
Продолжается толькоTRAIN-диагностика замороженных features/head: различимость
S/Dконтекстов и вклад фокальных решений в loss/градиенты.

**Зарегистрирован узкий TRAIN-capacityдиагноз послеMicroR4:** заморозить
все его признаки/ствол, обучить только копию существующего pair-readout
на72фокальныхрешениях с пустымoptimizer,seed20261031,AdamW/LR.0003/wd.01,
до1200updates/120с. Проверить .99/.8/0ошибок по каждойrelation и регрессии
остальныхпар. Цель — проверить доступность информации текущему readout,
не допуститьмодель: веса не переносятся,MicroR4 остаётсяfailed,
probe/full/PPO запрещены доотдельныхдопусков.
[Протокол](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r4/micro/postmortem/readout_capacity_protocol.json).

**Readout-capacityдиагноз завершён:**1200updates/7,77с,actualexit0,PID76077
отсутствует;72/72фокальныхрешения достигли.99,но возникли35уверенныхошибок
внефокальнойпопуляции (105новыхrawошибок). Ёмкость текущеговыхода показана
только наTRAIN; ничего не допущено, веса не переносятся. Исходныйmicro
остаётсяfailed. [Результат](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r4/micro/postmortem/readout_capacity/result.json).

**Зарегистрирован контролируемый rehearsal-диагноз:** одинаковые копии
исходногоmicro pairmodule,замороженныепризнаки; контроль uniformmean
всех5640пар и50/50среднихфокальной/остальнойпопуляций. По1200updates,
общийпредел180с,seed20261032,LR.0003/wd.01/clip1. Оцениватьвсепары
и ошибки каждого среза; не выбиратьпоэтомуопыту модель дляPPO.
[Протокол](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r4/micro/postmortem/readout_rehearsal_protocol.json).

**Controlled rehearsal завершён:** по1200updates/ветвь,5,14ссуммарно,
actualexit0/PID76192отсутствует; одинаковые начальныеpairweights. Контроль
uniformpopulation: уверенноS11/24,D10/24,U24/24,nonfocal5567/5568,
0уверенныхошибок. Отдельные средниеfocal/other50:50: уверенно
S/D/U24/24каждая,nonfocal5568/5568,всего5640/5640при.99и0ошибок.
Подтверждена только достаточность существующего представления/выхода и
полезность распределения обучения на этихTRAIN-контрастах. Перенос не
проверен; microR4 остаётсяfailed,веса не используются дальше.
[Результат](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r4/micro/postmortem/readout_rehearsal/result.json).

**R5 зарегистрирована до fit:** меняется только обучение pair-loss:
при наличии размеченныхTRAINfocalрешений —.5mean(focal)+.5mean(other),
иначе прежняя средняя. Применимость/режим,архитектура,декодер,пороги
и unweightedDEV-NLL/temperature остаются прежними. Используется в точности
корпусR4 по ссылке; новых текстов/отложенных выборок нет. Свежие head/optimizer
вmicro→probe→full,веса диагностик не переносятся; исходный общийdeadline
1792794300 не меняется. До fits нужны sourcechecks и успешный freshmicro.
[ПротоколR5](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r5/protocol.json).

Ограничение rehearsal-диагноза: ветви различались также samplingfocal
(full72каждыйшаг противuniform328),поэтому отдельно причинный эффект
коэффициента.5 не установлен. R5 проверяет связанную гипотезу распределения
обучения,не объявляет её доказанным общим исправлением. Новаяреализация
прошла41целевойтест/Ruff и независимыйMLreview; malformedfocalmetadata
отклоняется,DEVtemperature/selection остаютсяnatural. Общиепроверки R5
запущены23.10 в12:44:40Москва,actualhelper76515,execsession98740;
завершение пока не подтверждено.

**R5: общий набор прошёл:**4174Python-теста (576,24с),Ruff и78frontend,
фактическийexit0;367проверенныхфайлов неизменны. Helper76515/егогруппа
иobserver76535 завершены,execsession98740 закрыта.
[Проверка](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r5/checks_verification.json).
Tokenizer/heldoutQC переиспользован только после сверки точныхR4datahashes
и неизменных preprocessingисточников; новых CAL/SEALEDпредсказаний нет.
СледующийmicroR5: новыйhead/optimizer,seed20261028,теже44группы/264views,
до1200updates/600с,.99/.8/0критическихошибок по прежним и новым срезам.

**FreshmicroR5 запущен23.10 в12:56:23Москва и завершён:** supervisor77540,
actor77550;1200updates/144,42с,фактическийexit0,оба процессаотсутствуют.
368frozenfiles/входы/checkpoint проверены. Начальныевеса,E5state ивсе264IDs
совпали сR4,батчи задаются прежним детерминированнымsampler.
Допуск не пройден: focalS24/24,D4/24,U24/24приняты при.99;
argmax24/24,16/24,24/24. Критическиеошибки6 (app3,pair3).
Режимы1770/1770rawверны;pair5629/5640какR4,но aliases133/264приняты
вместо229/264уR4;app9472/9504raw вместо9496/9504. Изолированный
frozen-readoutуспехнеперенёсся наjointfreshfit.
Probe/full/PPO не запускались;runtime/calibration/sealed не оценивались.
[Первичная проверка](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r5/micro/verification.json).
Следующий шаг — только read-onlyизмерениеградиентов общего projection/local/
contextствола по app/mode/focalpair/otherpair; новое обучение и изменение
архитектуры до анализа не назначены.

**R5: диагностика общего представления завершена без fit:** информацияS/D
усилилась (среднийконтрастлогитов1,93→9,89), градиентpairв27,05раза
сильнееapp+mode; общийcosine+.037 не доказывает глобальный конфликт задач.
Основныеошибки — compressedD6/8. Зарегистрирован TRAIN-onlyконтроль:
одинаковые копииR4final,seed20261033,одинаковые16-rowбатчи/R5loss;
совместное обновление против толькоpairпри замороженном общемстволе.
До1200updates/300с наветвь,600с всего; .99/.8/0 прежние.
Нетпереносавесов,calibration/sealed/runtime илидопуска.
[Протокол](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r5/micro/postmortem/joint_vs_frozen_protocol.json).

**R6 зарегистрирована до fit:** исходники,корпусR4,потериR5,
архитектура,пороги и рецепт8/10эпох не меняются. КонтрольjointR4final
после1200дополнительныхобновлений прошёл все264TRAINviews; это
не freshfit/необобщение. Проверяем тужеёмкость со свежихвесов: seed20261028,
теже44группы/264views/батчи16,предел2400updates вместо1200 при прежних600с.
Переносавесов нет; отдельные sourcechecks4174/Ruff/78 переиспользованы
после сверки всех367источников, без повторного запуска тестов. Старый
failedmicroR5 остаётсянеизменным; probeR6 разрешён только после успешного
freshmicroR6. Всеостальныепроверки/дедлайн1792794300 прежние.
[ПротоколR6](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r6/protocol.json).

**КонтрольR5 joint/frozen завершён:**241,89с,actualexit0;
actor78270/wrapper78268/launcher78267 отсутствуют. Одинаковыеinitial/
batchtrace подтверждены. Joint1200updates/128,72с: S/D/U24/24каждый
принятпри.99,app9504/9504,mode1770/1770,nonfocalpair5568/5568,
0критическихошибок/регрессий,всеmicroсрезыпройдены. Frozen1200/106,14с:
S13/24,D16/24,U24/24приняты,0критическихошибок,допускнепройден.
Обновленияобщегостволапомогли вданномlatephaseсравнении; необходимость
freeze илиневозможностьjointобученияопровергнуты. Необобщение/неfreshfit.
Диагностическиевеса неиспользуютсявR6.
[Первичнаяпроверка](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r5/micro/postmortem/joint_vs_frozen/verification.json).

**Fresh micro R6 запущен 23.10 в 13:16:26 Москва:** supervisor 78550;
новые веса и пустой optimizer, до 2400 обновлений / 600 с. Предстоит
проверить фактический выход, checkpoint и допуск; сам запуск не означает успеха.
[Процесс](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r6/micro/process.json).

**Fresh micro R6 прошёл:** 1400 обновлений / 166,64 с, фактический exit 0;
actor 78563 и supervisor 78550 отсутствуют. Первые 1200 оценок и loss
точно совпали с R5; дополнительная длительность позволила пройти те же
критерии без изменения весов потерь или данных. При .99: S 24/24, D 23/24,
U 24/24; сырые ответы 24/24 во всех трёх срезах. Критических ошибок 0,
все прежние обязательные срезы также прошли. Это проверка способности
обучаться на 264 TRAIN-текстах, не результат 32 инструкций или независимого
обобщения. 368 frozen-файлов и входные хеши проверены; диагностические
веса не переносились. Следующий этап — восьмиэпоховая проба на полном
TRAIN/DEV со свежим head/optimizer, seed 20261029, batch 32, LR .0003,
лимит 4200 с; calibration/sealed не используются.
[Проверка micro](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r6/micro/verification.json).

**Проба R6 запущена 23.10 в 13:20:02 Москва:** supervisor 78796,
actor 78807 подтверждён по фактической команде процесса.
Восемь эпох TRAIN/DEV со свежими весами; admission/full/PPO не запущены.
[Процесс](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r6/probe/process.json).

**Наблюдение R6:** дэшборд показывает текущий этап перед свёрнутой историей,
сохраняет раскрытие при обновлении и различает технический выход процесса
и непринятый результат. Фазы обучения/проверок подписаны по-русски; loss
оставлен в технических сведениях. 81 frontend-тест прошёл; в браузере
проверены 390×844 и 1440×1000 без переполнения, автообновление счётчиков
и сохранение раскрытия, ошибок консоли нет. 368 frozen-файлов обучения
не изменились; общий набор текущего worktree требуется перед PPO.
Воспроизведение новой политики ещё не проверено: её пока нет.
[Браузерная проверка](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r6/browser_preparation_check.json).

**Проба R6 завершена с отрицательным допуском:** 8 эпох, 3816 обновлений,
122112 TRAIN-exposures, 1340,04 с, фактический exit 0; actor 78807 и
supervisor 78796 отсутствуют. Выбрана эпоха 8 по DEV, runtime **5/32**.
DEV: удержание 990/3720, нажатие 642/2898, эквивалентность 124/516
корректных оплачиваемых решений; 34 критические ошибки. Сырые ответы:
app 64650/64800, mode 11275/11592, pair 36093/36720.
Свежая инициализация, фактическое состояние selected head, runtime-logits,
368 frozen-файлов и все входные хеши проверены. Полное обучение награды,
calibration/sealed-предсказания и PPO не выполнялись. Успех micro остаётся
проверкой TRAIN-capacity; он не доказал перенос.
[Первичная проверка](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r6/probe/verification.json).
Следующая диагностика только read-only: 1800 стратифицированных TRAIN и
1800 DEV selection на этих выбранных весах, без E5-весов/optimizer/fit,
лимит 180 с; сравнить сырые ответы, уверенность и составные блокировки.

**R6, завершённая диагностика TRAIN/DEV и возобновление 23.10:** фактический
exit 0, 59,13 с, sampled RSS 1518174208 байт. На 1800 TRAIN views:
held argmax 3744/3744, press 3039/3042, но оплачиваемые решения
1659/3744 и 1073/3042. Same-pair argmax 1448/1566, raw confidence .99
762/1566, calibrated confidence .99 536/1566, оплачиваемые группы 220/1566.
Низкая уверенность внешних связей — 4663 TRAIN и 6277 DEV составных блокировок
(причины могут пересекаться). Это стратифицированная TRAIN-выборка, а не
популяционная оценка; сравнивать её общую долю с DEV напрямую нельзя.
Намеренно неизвестные связи ограничивают доступность бонуса даже при идеальных
целях: DEV held 3516/3720, press 2766/2898, aliases 444/516. Это не дефект
декодирования. Недостаток уверенности присутствует до temperature scaling;
снижение порога не обосновано. Причина ещё требует контролируемого опыта.
[Сводка](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r6/probe/postmortem/train_dev_diagnosis_summary.json),
[первичная диагностика](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r6/probe/postmortem/train_dev_diagnostic.json).
После паузы пользователь разрешил продолжение; 23.10 около 16:49 Москва
обучающих процессов нет, дэшборд PID 82319 работает. Осталось около 8 ч 36 мин
до исходного дедлайна, включая резерв финала 2 ч. Полный fit, PPO, пилоты,
основная политика и sealed/test по-прежнему не выполнялись.

### R6-pair-population-control-20261023 · распределение обычных связей

- Статус до запуска: зарегистрирован короткий диагностический TRAIN-контроль.
- Гипотеза: балансировка классов только обычных (nonfocal) связей исправит
  недостаток уверенности эквивалентов при неизменных признаках. Пересчёт TRAIN
  подтвердил 14208 SAME из 283728 nonfocal пар; focal уже получает отдельную
  половину потери. Это коэффициенты loss, а не доказанная величина градиента.
- Модель: копии selected R6 pair readout, одинаковые исходные веса и пустые
  AdamW; общий ствол, app/mode и E5-признаки заморожены. E5-веса не загружаются.
- Данные: прежняя фиксированная диагностическая TRAIN-панель 300 групп /
  1800 views, включая 44 micro-группы. Выбор не зависит от ошибок модели.
- Контроль A: прежние .5 mean(focal) + .5 mean(other). B меняет только other
  на среднее CE по присутствующим SAME/DIFFERENT/UNSPECIFIED. Архитектура,
  корпус, декодер, порог .99 и температуры R6 не меняются.
- Seed 20261031, LR .0003, weight decay .01, clip 1.0. Один общий trace
  перемешанных эпох строк, batches до 32 со всеми их размеченными парами;
  1200 updates каждой ветви, суммарно до 600 с, RSS до 6 GiB и резерв диска
  8 GiB. Короткий опыт входит в исходный дедлайн и сохраняет финальный резерв.
- Критерий: B улучшает correct99/total nonfocal SAME при фиксированной
  температуре минимум на 10 п.п. относительно A без увеличения уверенных
  ошибок по raw/fixed-temperature срезам и без потери индивидуально правильных
  оплачиваемых решений или роста ошибочных объединений. Pair-оценка каждые
  200 updates; настоящий whole-row decoder только до и после опыта.
- Ограничения: диагностика на TRAIN существующего представления, не проверка
  переноса или допуска. Веса не переносятся. DEV/calibration/sealed/runtime
  не оцениваются. Даже успех требует свежих micro и probe по прежним gates.
- Артефакты: [протокол](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r6/probe/postmortem/pair_population_protocol.json),
  [аудит коэффициентов](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r6/probe/postmortem/pair_loss_allocation_audit.json),
  результаты будут в `corpus_r6/probe/postmortem/pair_population_control/`.

Запущен 23.10 в 16:59:52 Москва, actor PID 90872. Это только TRAIN-readout
контроль; дальнейшее обучение и продвижение автоматически не запускаются.

**Результат: критерий не пройден, балансировка не принята.** Обе ветви
завершили 1200 updates за 161,47 с суммарно; фактический exit 0, actor
отсутствует, исходные веса/источники/cache не изменены, оба диагностических
checkpoint сохранены. Nonfocal SAME при фиксированной температуре:
A 748/1482, B 930/1482 (+12,28 п.п.); уверенных pair-ошибок raw/fixed в обеих
ветвях 0. Но оплачиваемые эквивалентные группы A 425/1566, B 385/1566;
правильные оплачиваемые режимы A 4922/8190, B 4652/8190. B теряет 397
индивидуальных правильных составных решений при фиксированной температуре
и 380 при raw. Ошибочные положительные объединения/применимость не выросли.
Следовательно увеличение уверенности SAME не решило всю композицию и не
обосновывает production-правку. Пробы/full/PPO после этого опыта не запускались.
[Результат](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r6/probe/postmortem/pair_population_control/result.json),
[проверка завершения](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r6/probe/postmortem/pair_population_control/verification.json).

**Следующий read-only аудит входов:** 30 ошибочных TRAIN-текстов из
heterogeneous-key/partial-alias семейств воспроизведены из замороженного
генератора. Labels, cached coordinates, длины токенов и отсутствие truncation
согласуются; exact-text conflicting targets — 0/15264 TRAIN строк,
exact-feature conflicting pair targets — 0/35208 пар панели.
Найдены два конкретных дефекта `_mention_spans`: `wheel scrolling up/down`
порождает ложное противоположное направление (5/30 примеров),
`5 key: a brief press...` — ложную клавишу A (2/30). Частоты относятся к
отобранным ошибочным примерам, не ко всему TRAIN. Причинный вклад в общий
провал не измерен; это основание исправить входную привязку с регрессионными
тестами и проверить её коротким опытом перед дальнейшим probe.
[Первичный аудит](runs/fresh_all16_v3_20261023/factorized_v2/corpus_r6/probe/postmortem/train_text_structure_audit.json).

### R7-grounding-paired-micro-20261023 · исправленная привязка команд

- Статус: зарегистрирован до обучения; preflight выполнен, fit ещё не запущен.
- Основание: полный TRAIN-аудит 15264 строк / 133572 упоминаний генератора
  подтвердил четыре дефекта привязки: направление `wheel scrolling`, артикль
  после `key:`, ложное движение мыши из соседней клавиши и ложная стрелка из
  направления движения мыши. Исправление устраняет 6499 ложных упоминаний;
  пропусков исходных упоминаний и новых неразобранных лишних — 0. Остальные
  304 дополнительные ссылки на направления признаны семантически верными.
  Это доказательство дефекта входов, ещё не доказательство причины всего R6.
- Регрессионные проверки: 357 тестов и целевой Ruff прошли; независимая
  проверка уточнённой грамматики не выявила оставшихся материальных ошибок.
- Контроль: две свежие полные головы 551820 параметров с идентичной
  инициализацией seed 20261028, пустыми AdamW и одним trace TRAIN-строк.
  Единственное различие — замороженная R6-привязка против исправленной.
  Архитектура, loss, данные, E5-признаки и порог 0,99 неизменны.
- Данные: 64 TRAIN-группы / 384 views, включая прежние 44 / 264 micro и
  все 60 / 360 исходной расширенной панели. Добавлены четыре группы для
  двух дополнительных типов дефектов; выбор не зависит от предсказаний.
  DEV, calibration, runtime и sealed не используются; E5-веса не загружаются.
- Бюджет: до 2400 updates каждой ветви, batch 16, LR .0003, WD .01,
  clip 1; суммарно до 600 с, RSS 6 GiB, свободный диск не менее 8 GiB.
  Первые 100 updates — полные инструкции, затем вся панель; trace общий.
- Оценка каждые 200 updates при равном объёме обеих ветвей. Критерий:
  исправленная ветвь проходит прежние all-head/held/press/alias/timing/focal
  срезы и дополнительные срезы затронутых кандидатов: correct99 coverage
  не менее 0,8 и ноль уверенных ошибок в каждом применимом срезе.
  Веса диагностические, в probe не переносятся. Успех не является допуском
  оценщика: далее обязательны общий набор проверок и свежий probe.
- Preflight: exit 0, 384 строки, длины токенов/cache согласованы, начальная
  голова SHA `b33dc468a48420c262febdba72e42fff85fe5ae9a9a59561a57278a2fb854b49`;
  fit 0 updates. Подготовка протокола сначала остановилась до fit из-за
  различия схем отчётов; чтение исходных mandatory groups исправлено.
- Артефакты: [аудит](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/grounding_prevalence.json),
  [протокол](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/micro_control_protocol.json),
  [preflight](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/micro_control_preflight.json).

**Результат R7 paired micro: критерий пройден.** Обе ветви выполнили ровно
1800 updates за 391,31 с суммарно. Исправленная ветвь прошла каждый срез при
.99, coverage ≥.8 и нуле критических ошибок; исходные aliases 256/264,
фокусные SAME 23/24, DIFFERENT 24/24, UNSPECIFIED 24/24. Контроль со старой
привязкой не прошёл affected applicability после `key:`: 24/41. Таким образом
короткий парный опыт подтверждает пользу исправления входов на этой TRAIN-
панели; он не доказывает устранение всех причин провала полного R6.
Фактический exit 0; actor PID 94098 отсутствует, обе головы/optimizer
сохранены и проверены по хешам, исходники и E5-cache неизменны. Calibration,
DEV, sealed и runtime не использованы. Веса не переносятся. Следующий этап —
общие проверки текущего источника, затем отдельно зарегистрированный свежий
восьмиэпоховый probe с прежними gates, без автоматического full-fit/PPO.
[Результат](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/micro_control/result.json),
[проверка завершения](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/micro_control/verification.json).

### R7-fresh-probe-20261023 · свежая проба после исправления привязки

- Статус: подготовлен и зарегистрирован; запуск только после общего pytest,
  Ruff и frontend на неизменных исходниках. Короткий парный micro пройден.
- Гипотеза: устранение четырёх доказанных дефектов входной привязки позволит
  свежей голове пройти прежний допуск; результат не гарантирован.
- Исходная модель: новая случайная FactorizedControlHead, новый AdamW;
  замороженный E5. Веса micro/R6 не переносятся. Архитектура, loss, разметка,
  декодирование и пороги R6 не меняются; единственное изменение — grounding.
- Данные: прежний `corpus_r4/data_v14`, TRAIN для градиента, DEV для выбора
  эпохи. Неизменный замороженный E5-cache TRAIN/DEV R6 используется read-only;
  полная проверка байтов до и после. Calibration и sealed не оцениваются.
- Протокол: 8 эпох, seed 20261029, batch 32, LR .0003, 4 CPU threads;
  actor до 4170 с, supervisor до 4200 с, RSS группы до 6 GiB, резерв диска
  8 GiB. Исходный общий дедлайн и финальный резерв 7200 с сохраняются.
- Критерии: прежние feasibility-срезы при уверенности .99, coverage ≥.8,
  ноль критических ошибок и runtime 32/32 после выбора головы. Runtime не
  используется для выбора. Полный свежий fit, calibration и однократный sealed
  разрешены только после прохождения probe, автоматического продвижения нет.
- Артефакты: [протокол](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe_protocol.json),
  результаты ожидаются в `grounding_r7/probe/`, источники — `grounding_r7/source/`.

R7 probe запущен 23.10 в 17:51 Москва: actor 94951, supervisor 94936,
368 файлов снимка и входные pins проверены до старта. Общие проверки:
4413 pytest passed (572,29 с), Ruff exit 0, frontend exit 0, хеши исходников
до/после совпадают. Единственный pytest warning — стандартный gymnasium
checker не проверяет альтернативные render modes без environment spec.
Снимок дополнен обязательным `SPEC.md` до запуска, поскольку trainer читает
его provenance-хеш. Runtime/calibration/sealed ещё не оценены.

**Первая техническая попытка R7 probe остановлена до fit:** actor exit 1 через
12,94 с, optimizer updates 0. `read_factorized_dataset` отверг manifest R4:
он включает старый хеш grounding. Модель не обучалась; head/checkpoint,
runtime/calibration/sealed результаты не созданы. Actor и supervisor
отсутствуют, группа процессов пуста. Артефакты сохранены целиком в
`grounding_r7/probe_attempt1_preflight_failure/`, включая исходный launcher
и проверку завершения. Это ошибка подготовки provenance, не провал качества.
Готовится отдельная техническая попытка с новым immutable manifest и
проверенной побайтовой идентичностью всех текстов/меток R4; условия обучения
и критерии не меняются. Исходный `probe_protocol.json` сохраняется.

**R7-fresh-probe-retry1-20261023 зарегистрирован до повторного запуска.**
Новый `grounding_r7/data_v14` штатно сгенерирован за 14,19 с прежним seed
20261029, split 2544/600/400/600 групп и теми же 9 исключениями. Все 24864
строки побайтово идентичны R4, SHA `73d16fac…23da99`; manifest отличается
только `source_sha256`. Штатный reader нового корпуса прошёл. Старый корпус,
cache и первая попытка сохранены. Новая попытка использует тот же проверенный
snapshot и recipe; обновляется только путь к provenance-совместимому корпусу.
[Проверка идентичности](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/corpus_integrity_verification.json),
[протокол технического повтора](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe_retry1_protocol.json).

**Технический retry1 остановлен на features до fit:** общий reuse-loader
требует `dataset_manifest.candidates`, отсутствующий в factorized manifest.
Проверка входной привязки прошла, но optimizer updates 0; checkpoint не создан.
Фактический exit 1, actor/supervisor отсутствуют. Попытка сохранена в
`grounding_r7/probe_attempt2_cache_preflight_failure/` с исходным launcher.
Производственный код, снимок, данные, пороги и ранее пройденные проверки
не меняются; повторное использование cache отменено как неподдерживаемая
оптимизация, проверки идентичности не обходятся.

**R7-fresh-probe-retry2-20261023 зарегистрирован:** штатное новое кодирование
TRAIN/DEV в `grounding_r7/probe/features`, как в R6, вместо reuse. Recipe,
seed 20261029, 8 эпох и общий предел 4200 с прежние, включают кодирование.
Прогноз диска 4 GiB при обязательном свободном резерве 8 GiB; старый cache
сохранён. [Протокол](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe_retry2_protocol.json).

**Итог R7 probe retry2: допуск не пройден.** Свежая голова завершила 8 эпох /
3816 updates / 122112 TRAIN exposures за 1353,82 с полного процесса. Selected
checkpoint — epoch8 по прежнему DEV-правилу. DEV 1800 views: 29 критических
ошибок; правильные оплачиваемые aliases 23/516, held 840/3720, press 506/2898.
Runtime 1/32, суммарно 7 записей ошибок в результатах инструкций. Все исходные
пороги сохранены. Новый cache 2665175040 bytes; peak RSS группы около 4,66 GiB.
Фактический exit 0, actor95270/supervisor95262 отсутствуют (точные identities
в process receipt), выбранный checkpoint/cache/source pins проверены.
`ready_for_fresh_full=false`; calibration/sealed не использовались, полный fit
и PPO не запускаются. Короткий успех grounding не переносится на общий допуск.
[Результат](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/output/probe_report.json),
[проверка завершения](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/verification.json).

Read-only аудит оптимизации: R6/R7 имеют одинаковые 477 updates/epoch,
TRAIN-loss действительно снижается, явного no-op градиентов не обнаружено.
Идентичные TRAIN-тексты не имеют противоречащих меток. Micro выполнил около
75 проходов по 384 views, полный probe — 8 по TRAIN; это гипотеза недостатка
оптимизации, не доказанная причина. Возможное продолжение только как отдельный
короткий TRAIN-диагностический опыт с прежними AdamW/RNG, фиксированными
температурами и без переноса весов; длинный fit пока не разрешён допуском.

Дополнительный runtime input audit R7: 32/32 текста, 238/238 supported
candidate occurrences с непустыми masks, длины 65–127 токенов, omissions 0.
Новых доказанных ложных привязок нет; 72 упоминания unsupported команд
соответствуют явным отрицательным фразам. Это проверка входов, не семантическое
качество модели. Exit 0, 3,27 с, без E5/head forward, fit, CAL/SEALED.
[Первичный audit](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/runtime_grounding_audit.json).

### R7-optimization-control-20261023 · короткая проверка продолжения оптимизации

- Подготовка до fit: измеряется исходное состояние на прежней фиксированной
  TRAIN-панели 300 групп / 1800 views, независимо от ошибок; 0 новых updates.
- Гипотеза: восьми проходов недостаточно для уверенных составных решений.
  Контроль — исходный terminal epoch8; изменение — только дополнительные
  обновления по полному прежнему TRAIN. Это диагностика, не новая архитектура.
- Состояние: selected=terminal epoch8 seed 20261029, восстанавливаются head,
  AdamW, CPU Torch RNG и локальный Python sampler RNG. Последний отдельно
  сверяется воспроизведением восьми исходных shuffles без обучения.
- Данные: полный TRAIN 15264 строки для градиента; только фиксированные
  1800 TRAIN views для оценки до/после. Новых DEV, runtime, CAL/sealed
  предсказаний нет. E5-веса не загружаются; признаки заморожены. Температуры
  выбранной R7 головы фиксируются; raw результаты показываются отдельно.
- Бюджет: максимум 1200 дополнительных updates, batch32, LR .0003, WD .01,
  clip1. Cumulative active preparation+fit+final/save не более600с;
  регистрационная пауза также расходует исходный общий срок. RAM6GiB,
  свободный диск8GiB; финальной оценке резервируется измеренное baseline
  время×1.25+20с. Веса не переносятся, readiness/admission запрещены.
- Критерий короткой проверки: при .99 и фиксированных температурах coverage
  app/mode/pair и оплачиваемых held/press/aliases ≥.8, ноль критических
  ошибок. Изменения NLL, correct99 и индивидуальные gains/losses сообщаются
  отдельно; просто улучшение не считается прохождением этого критерия.
- Draft и baseline ожидаются в `grounding_r7/probe/postmortem/`; fit ещё
  не зарегистрирован и не запущен. Подготовка runner включала проверку
  сохранения состояния при лимите времени; общий срок не обнуляется.

До первого fit R7 optimization control обнаружена ошибка диагностического
критерия на стратифицированной TRAIN-панели. Strict gold oracle через реальный
decoder разрешает held3180/3744, press2376/3042, aliases1128/1566 из-за явно
неизвестных внешних отношений. Поэтому прежний draft требовал от идеальных
меток недостижимого для них полного coverage .8. Это не абсолютный потолок
произвольных предсказаний: отказ по другому supported endpoint может убрать
блокирующее отношение. Исходный baseline (54,62с, подготовка67,08с,0updates)
сохраняется; до fit готовится пересмотр диагностического знаменателя.
Новый диагностический критерий .99/.8/0errors применяется к заранее
фиксированным strict-gold-oracle IDs, числитель ограничивается теми же IDs;
полные n/N, oracle-доли и ошибки по всем исходам сохраняются. Общие
probe/CAL/sealed gates не меняются. Проверка графа совпала с actual decoder
на всех1800TRAIN с синтетическими gold logits; E5/head/fit не использовались.
[Gold-популяция](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/train_panel_gold_payability.json).

R7 optimization control r2 зарегистрирован до fit. Исходный baseline и его
producer сохранены; новый baseline дополнен gold-eligible подсчётами из уже
сохранённых individual outcomes без повторного forward. Подготовка с резервами
заняла69,18с из600с. Fixed-temperature baseline: held1101/3180,
press848/2376, aliases53/1128 среди strict-gold-oracle IDs; полные знаменатели
3744/3042/1566 сохранены. Natural field coverage app64218/64800,
mode10673/11334, pair26163/35208; errors99=0. Short gate не пройден до fit.
[Зарегистрированный протокол](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/optimization_control_protocol.json),
[baseline r2](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/optimization_control_preparation_r2/baseline.json).

R7 optimization control r2 завершён: 1200 дополнительных updates, всего5016,
359,98с с подготовкой и оценкой, actual exit0; процесс завершён, checkpoint
с optimizer/RNG сохранён и проверен отдельно. Короткий критерий не пройден.
При прежней температуре и .99: held1942/3180, press1530/2376, aliases175/1128
на заранее заданных gold-eligible IDs; полные результаты1942/3744,1530/3042,
175/1566. Критических ошибок0. Поля app64574/64800, mode11038/11334,
pair30947/35208. Raw eligible held2898/3180, press2102/2376, aliases753/1128,
но2 критические ошибки. Pair NLL снизился .015472→.004087; fixed индивидуальные
gains/losses: aliases126/4, held868/27, press709/27. Это свидетельство
продолжающейся оптимизации, не доказательство достаточности её длительности.
DEV/runtime/CAL/sealed не оценивались, веса не переносятся.
[Результат](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/optimization_control/result.json),
[Проверка завершения](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/optimization_control/verification.json).

### R7-LR-pair-20261023 · подготовка одного контроля скорости оптимизации

Гипотеза, ещё не доказанная: больший LR ускоряет достижение уверенности SAME.
Обе ветви начинаются с одного диагностического checkpoint5016, включая AdamW,
Torch/Python RNG и cursor середины эпохи; одинаковые следующие1000TRAINupdates
на ветвь. A LR.0003, B LR.001; остальные настройки и фиксированные температуры
прежние. TRAIN15264 для градиентов, прежние1800TRAINviews для оценки;
baseline переиспользуется с проверкой hash, финальная оценка каждой ветви.
Максимум600с всей активной работы, RAM6GiB и диск8GiB; неполная пара не проходит.
B должен выполнить прежний диагностический .99/.8/0errors gate и улучшить
составные исходы относительно A. Raw и полные знаменатели сохраняются.
DEV/runtime/CAL/sealed не открываются, допуска и переноса весов нет.
При неудаче этот опыт не обосновывает дальнейший перебор LR. Runner и
точный протокол готовятся в .cache и grounding_r7/probe/postmortem; fit не начат.

Подготовка LR-pair: первый preflight завершился exit1 до fit/forward,
поскольку receipt закреплял исходные байты протокола, а проверялся hash
нормализованной JSON-копии. Исправляется проверка исходного hash плюс
равенства содержимого копии; 20с неудачной подготовки включаются в600с.
Артефакт: `grounding_r7/probe/postmortem/lr_pair_prepare_failure.json`.

LR-pair зарегистрирован до fit: `grounding_r7/probe/postmortem/lr_pair_protocol.json`.
Подготовка32,23с включая прежнюю неудачу; baseline reuse без forward;
exact5016cursor/AdamW/RNG и общий trace проверены. Статический review пройден.
Критерий улучшения B относительно A: сумма net-gains held/press/aliases>0
без отрицательного net-gain в отдельном срезе, вместе со всеми short gates.

LR-pair завершён: по1000updates в A/B, 518,13с включая подготовку,
actualexit0, процесс отсутствует, обаcheckpoint проверены поhash.
Критерий не пройден; увеличенный LR отвергнут. Fixed gold-eligible A/B:
held2023/3180 против1165/3180; press1576/2376 против1133/2376;
aliases302/1128 против121/1128. Полные знаменатели held3744, press3042,
aliases1566 сохранены; outside-gold-correct0, fixedcritical0 в обеихветвях.
Pair NLL A.004044 против B.029204; rawcritical A0/B21.
B относительно A: aliases28gains/209losses, held154/1012, press90/533.
Ни дальнейший перебор LR, ни длинный fit этими данными не обоснованы.
DEV/runtime/CAL/sealed не оценивались, веса не переносятся.
[Результат](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/lr_pair_control/result.json),
[Завершение](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/lr_pair_control/verification.json).

### R7-gradient-interference-20261023 · подготовка локального аудита без обучения

Гипотеза: градиенты частей objective мешают уверенности nonfocal SAME.
Checkpoint — проверенная ветвь A6016; данные — заранее фиксированные16следующих
TRAINbatch по32 из её сохранённого sampler cursor. Маска низкой уверенности
SAME определяется внутри этих batches при прежней температуре и пороге.99.
Измеряются скалярные произведения/нормы градиента этой цели с app, mode,
focal и nonfocal pair, отдельно общий trunk и pair readout, а также
направленная производная по математически восстановленному шагу AdamW
с прежними moments, clipping и weight decay. Optimizer.step запрещён,
веса/RNG/optimizer должны остаться прежними; DEV/runtime/CAL/sealed не используются.
Бюджет120с, RAM6GiB; исходный общий дедлайн и резерв сохраняются.
Положительный локальный конфликт лишь локализует возможный механизм,
не доказывает исправление; отрицательный результат отвергает эту локальную
гипотезу. Это отличается от прежнего R5micro global-gradient аудита:
другой checkpoint, реальные следующие batches и учёт AdamW вместо Euclidean
endpoint-gradient alone. Runner/точный протокол пока готовятся, запуск не начат.

Gradient audit зарегистрирован до запуска: `grounding_r7/probe/postmortem/gradient_interference_protocol.json`; подготовка12,91с, exit0, 0forward/0updates.
Маска цели: gold nonfocal SAME с P(SAME)<.99; 16независимых локальных
AdamW proposals из одного frozen A6016, не16-step траектория. Review пройден.

Gradient audit завершён:16/16batches,512TRAINviews,217целевых низкоуверенных
nonfocal SAME;35,05с с подготовкой, actualexit0, процесс завершён,
head/optimizer/RNG неизменны,0updates. Математический AdamW delta имеет
отрицательную производную целевого loss в15/16batches по всем параметрам
и общему trunk,13/16 по pair readout; средние соответственно−.03664,
−.03198,−.00466. Доминирующее противодействие обновлений не подтверждено.
Отрицательный aggregate Euclidean dot pair readout (−.00810) сам по себе
не описывает фактическое направление AdamW; менять архитектуру/loss это
не обосновывает. Это локальные первые производные, не finite-step оценка
и не доказательство будущего переноса.
[Аудит](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/gradient_interference.json),
[Проверка](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/gradient_interference.verification.json).

После gradient audit выполнена read-only проверка12первых по порядку корпуса
TRAIN-групп с fixed-false alias outcomes A6016:0281,0282,0284,0316,0346,
0347,0376,0377,0378,0379,0382,0408. Семантических противоречий текста,
labels или binding spans не обнаружено. Из13исходов4gold-ineligible,
среди9eligible rawpass3/rawfail6; все6rawfail — явные cross-mode held/press
aliases. Это выборочное наблюдение, не частота по всей популяции и не причина.
Следующий read-only аудит использует только уже сохранённые A6016 outcomes
на всех1800TRAINviews и groundtruth полного TRAIN15264: fixed/raw n/N
по same-mode/cross-mode плюс представленность отношений. Без forward/fit,
порогов, изменений данных или модели. Артефакт ожидается
`grounding_r7/probe/postmortem/cross_mode_alias_audit.json`.

Cross-mode audit по всем сохранённым A6016 TRAIN outcomes подтвердил
разрыв: same-mode fixed276/486, raw458/486,36независимыхгрупп;
cross-mode fixed26/642, raw345/642,51группа. Held↔press: fixed4/174,
raw75/174,22группы. Во всём TRAIN cross-mode SAME8610пар/264группы,
same-mode SAME6750пар/255групп. SAME среди S/D:5,06% против7,84%;
held↔press4,61%, held↔held10,61%. Это decoder-результат, не изолированный
pair-head; семьи текстов и состав пар различаются, причинность не установлена.
[Первичные подсчёты и pins](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/cross_mode_alias_audit.json).

### R7-pair-blocker-audit-20261023 · подготовка oracle-диагностики decoder

Один forward проверенного A6016 по прежним1800TRAINviews; индивидуальные
logits сохраняются. При неизменных app/mode, температуре и пороге.99
сравниваются исходный decoder и диагностические подстановки уверенно
правильных только gold-SAME, только gold-DIFFERENT, обеих групп рёбер.
UNSPECIFIED и прочие поля не меняются. Исходные alias-outcomes должны
точно совпасть с ранее сохранёнными; восстановленные/потерянные исходы
считаются на прежних gold-eligible IDs по режимам и языку.
Бюджет300с включая подготовку/оценку/сохранение; без fit/E5weights/holdouts.
Oracle-подстановка не является production fix, допуском или измеренным
качеством исходной модели; она только разделяет возможные блокирующие
причины. Runner и протокол готовятся, запуск не начат.

Pair-blocker audit зарегистрирован до запуска: `grounding_r7/probe/postmortem/pair_blocker_protocol.json`; подготовка12,79с, exit0,0forward; статическийreview пройден.

Pair-blocker audit завершён за152,15с с подготовкой, exit0, процесс завершён,
1800TRAINviews forwarded один раз,0updates. Сохранённые raw/fixed baseline
outcomes совпали точно, logits/targets/IDs сохранены и hashes проверены.
Gold-eligible aliases: baseline302/1128; oracle SAME401/1128; oracle
DIFFERENT532/1128; oracle обеих частей1108/1128. Cross-mode26/642→84/642
(SAME),146/642(DIFFERENT),635/642(обе); same-mode276/486→317/486,386/486,
473/486. Потерь исходно правильных outcomes нет. Следовательно, в этой
фиксированной модели обе части pair-решений совместно блокируют bonus;
изолированное улучшение SAME не устраняет основную часть отказов.
Oracle-результат не приписывается модели и не допускает обучение/PPO.
[Результат](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/pair_blocker_audit/result.json),
[Проверка](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/pair_blocker_audit/verification.json).

### R7-required-margin-control-20261023 · подготовка одного парного контроля

Гипотеза после oracle-аудита: средний pair CE оставляет отдельные обязательные
S/D-рёбра ниже.99, хотя decoder требует их совместного принятия. Это ещё
не установленная причина и не принятое изменение production loss.
Обе ветви стартуют из одного проверенного A6016 с AdamW/RNG/cursor;
общий trunk и unary readouts заморожены в обеих, обучается лишь pair readout.
A — прежний pair objective. B — тот же objective плюс коэффициент1
для mean-row worst-required-edge hinge:
`relu(log(99) + logsumexp(other_logits/Tfixed) - gold_logit/Tfixed)`.
Обязательные рёбра берутся из TRAIN gold: компоненты SAME с полным
внутренним SAME и всеми внешними DIFFERENT до других supported-команд;
union внутренних и внешних рёбер сертифицируемых компонентов, включая
одиночные. U не заменяется, остаётся в базовом CE и блокирует компонент.
Порог/температуры/decode неизменны; это TRAIN supervision, не runtime oracle.
По1000следующих одинаковых batch32 наветвь, LR.0003, общий предел600с
с подготовкой и финальными оценками, RAM6GiB/диск8GiB. Только TRAIN;
DEV/runtime/CAL/sealed и E5 weights не используются. Baseline A6016
переиспользуется поhash, финальные оценки — прежние1800TRAINviews.
B обязан пройти прежний short .99/.8/0errors и улучшить сумму whole-outcomes
против A без отрицательного net-gain в отдельном held/press/aliases.
Один коэффициент без перебора. Диагностические веса не переносятся;
успех не является допуском и потребует свежего проверенного протокола.
Runner/точный protocol готовятся, fit не запущен.

Required-margin зарегистрирован до fit: `grounding_r7/probe/postmortem/required_margin_protocol.json`; preparation13,89с, exit0, model0forward/0updates.
Граф13200/15264TRAINстрок имеет обязательные рёбра:12144S+240900D.
Panel goldIDs совпадают точно. Проверки U-блокировки/nonclique/singleton,
p99 boundary/sign, emptyrow0 и gradNone/AdamW freeze пройдены; review пройден.

Required-margin контроль завершён:1000updates/ветвь,437,70с,actualexit0,
обаcheckpointhash и отсутствие процесса проверены. Все frozen weights/buffers
и AdamW состояния побитово неизменны. B прошёл заранее заданный short gate.
Fixed gold-eligible A/B: aliases359/1128 против986/1128; held2300/3180
против2986/3180; press1711/2376 против2199/2376. Полные B знаменатели:
aliases986/1566,held2986/3744,press2199/3042; outside-gold-correct0.
Критические ошибки fixed/raw0 в обеих ветвях; B field coverage
app64556/64800,mode10992/11334,pair34650/35208.
Это контролируемое свидетельство пользы objective для существующего
фиксированного представления на TRAIN. Архитектурный предел не обнаружен;
перенос на DEV/runtime не проверен, допуск модели ещё отсутствует.
Диагностические веса не переносятся. Следующий этап — зарегистрировать
свежий probe с проверенным refinement; calibration/sealed остаются закрыты.
[Результат](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/required_margin_control/result.json),
[Завершение](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/probe/postmortem/required_margin_control/verification.json).

### R8-fresh-staged-probe-20261023 · подготовка свежей пробы

Основание — успешный парный required-margin TRAIN control R7; перенос
на DEV/runtime ещё не доказан. Архитектура, корпус R7, splits, пороги и
calibration/sealed требования неизменны. Новый явно версионированный
`staged_required_pair_margin.v1` заменяет число этапов старого8/10epoch
протокола:8базовых эпох/3816updates,2200продолжения до6016,1000pair-only
refinement до7016. Freshhead/AdamW seed20261029; диагностические веса и
moments не переносятся. Температура лучшего базового DEV-кандидата фиксируется
как training hyperparameter refinement; DEV явно остаётся development,
не независимой оценкой. Терминальная траектория обучения не подменяется
выбранными базовыми весами. Финальные температуры снова naturalDEV-NLL,
выбор между базовыми и refinedкандидатом прежнимDEVключом; runtime послевыбора.
CAL/sealed не используются. Лимитprobe4200с/full7200с и исходный deadline
с резервом7200с сохраняются; partialstagedrun не проходит. Full только
после probepass, отдельно freshseed20261030 с тем же stagedrecipe.
Точная регистрация ожидает реализации, conformance и полного набора тестов:
`required_margin_r8/protocol_draft.json`. Пока это подготовка, не запуск.

R8 production conformance прошёл за10,83с: все253044requirededges
на15264TRAINстроках совпали с успешным diagnostic, значение и gradients
margin совпали точно на64строках с фиксированными синтетическими logits.
Modelforward/updates0; артефакт `required_margin_r8/margin_conformance.json`.

R8 зарегистрирован перед запуском: [точный протокол](runs/fresh_all16_v3_20261023/factorized_v2/required_margin_r8/protocol.json), [неизменный manifest R7](runs/fresh_all16_v3_20261023/factorized_v2/grounding_r7/data_v14/manifest.json).
4455 Python-тестов, Ruff и81frontend прошли за584,33с;368исходников
и контрольные доказательства не изменились. Production/diagnostic conformance
пройден; freshprobe допускается к запуску. Успех требует confidence.99,
coverage≥.8, critical0 и runtime32/32; CAL/sealed не используются.
Новый протокол не является допуском полной награды или PPO.

R8 freshprobe запущен23.10.2026 в20:25:22Москва:actor876,supervisor861;
входные pins и369файлов snapshot проверены до старта.
[Execution](runs/fresh_all16_v3_20261023/factorized_v2/required_margin_r8/probe/execution.json),
[процесс](runs/fresh_all16_v3_20261023/factorized_v2/required_margin_r8/probe/process.json).
Лимит4200с; результат пока отсутствует, full/PPO не запущены.

Для R8 повторно проверена идентичность visual-only ресурсов:25pins,
включая16исходников, checkpoint, полный composition и validation report.
Сохранён допуск240/240прежних ячеек без нового inference;
`required_margin_r8/visual_reuse_verification.json`. Это не проверка
совместной награды с будущим admitted text bundle.

R8 служебный policy adapter теперь проверяет точную цепочку R7 shortcontrol →
R8 probe/full, full protocol/QC, source и visual pins; фиктивный R8 micro
не требуется.21mock regression, Ruff иreview прошли. Реальные prepare,
activate и интеграция не запускались; это совместимость сценария, не допуск.

R8 probe завершён:7016updates,3496,81с поtrainer/3502,74с поsupervisor,
actualexit0; actor/supervisor отсутствуют, checkpoint, freshinit, E5cache
и источники проверены.3816base+2200continuation+1000margin выполнены;
замороженные weights/buffers/AdamW неизменны. Continuation6016 побитово
воспроизводит head диагностического A. ПоDEV выбран refined7016.
**Допуск не пройден:** runtime7/32; DEVcritical76 (22error scenarios),
aliases195/516,held2339/3720,press2028/2898. Rawargmax:app64759/64800,
mode11424/11592,pair36342/36720. Minfeasibilitycoverage.3779069767<.8.
Все пороги прежние. CAL/sealed не открыты; full/PPO не запускались.
[Отчёт](runs/fresh_all16_v3_20261023/factorized_v2/required_margin_r8/probe/output/probe_report.json),
[Проверка завершения](runs/fresh_all16_v3_20261023/factorized_v2/required_margin_r8/probe/verification.json).
Успех короткого TRAIN-контроля не перенёсся в достаточный DEV/runtime
результат; конкретная причина ещё проверяется по сохранённым артефактам.
К21:24Москва полный7200сfit плюс сохранённый7200срезерв уже не помещаются
с обязательным запасом запуска. Дедлайн и обязательные этапы не сокращены.

Read-only сверка финальных checkpoints установила полное равенство R8
и diagnosticB:head tensors,AdamW state/paramgroups,torch/Python RNG,
sampler cursor,7016updates/224512examples. Расхождение реализации не
обнаружено. DiagnosticTRAIN использовалpairT2,113489; finalDEV T3,758374
(×1,778). Вклад сменыT вcoverage отдельно не измерен; rawDEVошибки
41/64800app,168/11592mode,378/36720pair отTне зависят.
Сравнение base3816→refined7016 включает2200ordinary+1000margin и не
является причинным сравнением одного margin. СтаруюTнельзя объявить
исправлением или основанием допуска. Forward/updates в аудите0.
[Аудит](runs/fresh_all16_v3_20261023/factorized_v2/required_margin_r8/probe/postmortem/trajectory_temperature_audit.json).

После завершения исправлена runtime-запись дэшборда: цикл остановлен,
последний этап — R8 qualitygatefail с7016updates,7/32runtime и76DEVerrors.
Reload браузера подтвердил эти сведения и отсутствие новой политики;
консольerrors/warnings0. Код интерфейса не менялся.

### Продолжение25октября2026 · без общего дедлайна

Пользователь разрешил продолжать без дедлайна; результат R8 и прежние
причины остановки сохранены. Проверено: обучающие процессы отсутствуют,
R8verification подтверждает неудачный допуск; веса и отчёты сохранены.
Следующий шаг — локализация ошибок переноса на существующей модели,
без открытияCAL/sealed и без очередного длинного fit до короткого контроля.
Запрошен полный отчёт всей работы после завершения исходной цели.

### R8-transfer-diagnostic-20261025 · подготовка

Гипотеза: смена температуры объясняет часть разницы покрытия между
коротким TRAIN-контролем и R8; остаток может быть связан с переносом.
Исходная модель — неизменный selected7016 R8, побитово равный diagnostic B.
Контроль: те же logits каждой панели декодируются при исходных R7 и
финальных R8 температурах. Межпанельные различия описательны, не доказывают
причину распределительного сдвига. Обучение и подбор температур отсутствуют.
Данные: исходные фиксированные1800TRAIN и1800DEV-selection, seed не
используется; CAL/sealed и runtime не оцениваются. Архитектура и веса прежние.
До интерпретации обязательны точное воспроизведение сохранённых TRAIN B
и DEV R8 результатов. Порог.99; диагностике запрещено давать допуск.
Бюджет:900с активного времени включая подготовку, RSS6GiB, свободный диск
не менее8GiB; общего дедлайна нет по указанию пользователя.
Протокол подготовки: `required_margin_r8/probe/postmortem/transfer_diagnostic_20261025/protocol_draft.json`
внутри `runs/fresh_all16_v3_20261023/factorized_v2/`.
Скрипт `.cache/r8_transfer_diagnostic.py`; Ruff, py_compile и synthetic
selfcheck прошли. Подготовка1,622с,393pins; inference и fit ещё не запускались.
Перед запуском ожидается независимая проверка и регистрация protocol.json.

Протокол зарегистрирован в `transfer_diagnostic_20261025/protocol.json`;
исходный draft сохранён. Запуск после завершения независимой проверки.

Независимая статическая проверка завершена без блокирующих замечаний.
Диагностика запущена25октября, exec session98019; результат пока отсутствует.

Диагностика завершена:110,01с включая подготовку, actualexit0, процесс5674
отсутствует; обе проверки точного воспроизведения прошли, хеши артефактов
проверены (`transfer_diagnostic_20261025/verification.json`). ПикRSS
2485813248байт;3600строк,0optimizerupdates и0E5forward, CAL/sealed закрыты.
При неизменных весах/строках исходнаяT→финальнаяT:
TRAIN aliases986→576/1566, held2986→2532/3744, press2199→1885/3042,
critical0→0. DEV aliases288→195/516, held2875→2339/3720,
press2387→2028/2898, critical144→76. Все знаменатели здесь полные,
не прежние micro-eligible подмножества. Температура причинно влияет на
coverage, но возврат исходнойT повышает DEV ошибки и не даёт допуска.
TRAIN/DEV различия остаются описательными; следующая работа — разбор
сохранённых logits/примеров по источникам ошибок. Веса и gates не менялись.

Разбор сохранённых выходов без новых forward локализовал все27уверенных
SAME→DIFFERENT и15false_split в compact family07, пяти независимых
контрастных ситуациях с≥8командами. Для SAME family06/07/08:
argmax157/172,104/172,157/172; правильно при.99:137/172,60/172,147/172.
У одной DEVгруппы0410 EN up/w explicit P(same)=.999837,
compact P(different)=.999547, naturalexplicit P(same)=.999999.
Рендеринги меняют несколько свойств; причинность пока не изолирована.
Из11592supported целей10776правильно уверенно приняты applicability,
но payable8393:2383/3199потерь возникают после правильного applicability.
76critical:52pair+15split+2merge+4app+3mode, не независимые события.
Следующий подготовительный шаг — TRAIN-only matched compact-block контроль
на замороженной модели и проверка пути grounding/features, без нового fit.

### R8-compact-control-20261025 · подготовка

Вопрос: меняет ли компактное представление WASD/стрелок уверенность
связей на фиксированной модели при сохранении смысла? Это чувствительность
на TRAIN, а не доказательство устранения DEV-ошибок или допуск.
Модель: неизменный R8selected7016; fit0, E5заморожен. Выбраны до inference
8независимых TRAINгрупп по детерминированному правилу, каждая RU/EN.
Для каждой исходный compact-блок сравнивается с явным перечислением
тех же4пар; весь suffix, фон, цели и режимы неизменны. Всего32текста.
Фокальная W/up сбалансирована4held+4press; остальные направленияheld.
Это не баланс всех направлений. Выбор не использует ошибки модели.
Протокол: `required_margin_r8/probe/postmortem/compact_control_20261025/protocol_draft.json`;
точные замены/ID — `selected_rows.json` в той же папке серии.
Оценка: focalSAME margin/уверенные ошибки, режимы и payable отдельно,
при двух заранее сохранённых температурах и пороге.99. DEV/CAL/sealed
не оцениваются; архитектура, корпус и веса не меняются. Seedнеприменим,
случайной выборки/fit нет. Локальный лимит900с/RSS6GiB; новые E5features
только для16expanded-текстов с прежней точностьюfloat16. Успех технический:
валидная семантика пары, неизменность модели, все32измерения в бюджете;
никакое улучшение TRAIN не считается допуском. Inference ещё не запущен.

## Сводный отчёт пользователю: весь fresh-all16-v3, состояние на 25 октября 2026

**Последующее обновление:** по запросу пользователя нативная цель успешно
создана заново и проверена: active, без дедлайна. Ниже сохранено описание
прежней ошибки со старой blocked-карточкой; её текущее состояние уже исправлено.

Это отчёт о выполненной работе на текущий момент, а не отчёт о достижении цели.
Область: весь цикл `runs/fresh_all16_v3_20261023`, включая работу до последней
передачи чата. Предшествующие серии v2 и октября сохранены выше в реестре.
Ниже перечислены существенные изменения, опыты, технические отказы и выводы;
точные команды, конфигурации, снимки исходников и результаты находятся по
ссылкам в соответствующих хронологических карточках выше. Это не дословный
журнал каждого вызова инструмента. Полную поминутную стоимость всей истории
по имеющимся артефактам достоверно не восстановили; отсутствующие данные
не заменяются нулями. Свежую основную политику этого цикла ещё не обучали.

### Цель, разрешения и фактическая остановка продвижения

Цель пользователя: свежая политика всех 16 игр; 100% побед отдельно в каждой
игре на фиксированных проверочных наборах, отдельно оформления и режимы.
Нельзя заменять это TRAIN или средним процентом. Сначала обязателен допуск
текстового оценщика, затем fresh full/calibration/runtime/sealed, интеграция,
технические 2+2 цикла, три 80-цикловых пилота seed3176 и main seed3177 без переноса
пилотных весов, validation-выбор, сравнения и однократный игровой test.

25 октября пользователь отменил общий дедлайн: «без дедлайна просто работай».
Это записано в `.cache/carryrl_goal_state.json`, SPEC и DECISIONS. Исторические
протоколы со старым сроком не переписывались. Порог.99, coverage≥.8,
0critical, runtime32/32 и calibration/sealed требования не ослаблялись.

ВАЖНО: нативная карточка цели НЕ обновлена. `get_goal`25 октября всё ещё
возвращает старую формулировку с дедлайном 24 октября 01:25 и `blocked`.
Доступный `update_goal` меняет только status complete/blocked/paused;
редактирования objective и resume он не предоставляет. `create_goal` не
заменяет незавершённую цель. Ассистент не должен был называть обновление
проектного JSON обновлением самой карточки. Ложное завершение цели ради
замены не выполнялось. Эта техническая проблема не отменяет разрешения
работать, но автоматическое продолжение нативной цели не подтверждено.

### Хронология выполненной работы

Все пути в этой таблице относятся к `runs/fresh_all16_v3_20261023/`.
TRAIN-короткие проверки нигде не являются допуском или игровыми победами.

| Этап | Действие и результат | Артефакты / ограничение |
|---|---|---|
| Старт v3 | Зафиксированы план, seeds, ресурсные пределы, новый head/optimizer и замороженный E5; сохранены прежние результаты | `protocol.json`, `manifest.json`, `cycle_state.json` |
| Первая проба | 8 эпох,3864updates,1000,53 с; runtime0/32; допуска нет | `probe_joint/verification.json` |
| Разбор контекста | 380TRAIN/192DEV; mixed-контекст содержит 130/132 и 98/106 mode-ошибок; fit0 | `probe_joint/postmortem/selected_mode_generalization.json` |
| Local-context v8 | Реализован локальный контекст; парный micro по 600updates: обе ветви held244/244,press196/196,aliases32/32,errors0 | `repair_local_context/micro/`; преимущество новой версии этим micro не установлено |
| Полная проба v8 | 4008pytest/Ruff/73frontend; проба 704,98 с; DEV5critical, runtime9/32 | `repair_local_context/probe_joint/verification.json`; full/PPO не запускались |
| Корпус и Space | Выявлены пробелы unknown+closed-world и compact; исправлены падежи «пробела»; frozen RUpress.938607→.962173, всё ещё ниже.99 | `repair_local_context/conditional_corpus_coverage.json`, `…/postmortem/space_grounding_counterfactual.json` |
| Согласованная factorized-v2 | Разделены applicability/mode/pair; новый V14 и reward composer; пользователь утвердил новый протокол | `factorized_v2/protocol.json`; сравнение со старым 9/32 не является чистым A/B |
| Первый factorized micro | 64TRAINviews,300updates,36,69 с; held124/124,press102/108,aliases15/16,errors0 | `factorized_v2/micro/verification.json`; длинная проба отменена из-за неполного корпуса |
| R2 корпус и micro | Добавлены полнота списка, compact-only, условия, режимы; 4110pytest/Ruff/75frontend; micro500updates56,78 с:116/116held,108/108press,16/16aliases | `factorized_v2/corpus_r2/` |
| R2 probe | 3816updates,1339,87 с; runtime0/32,DEV20critical; held0/3720,press0/2934 | `corpus_r2/probe/verification.json`; затем frozen-диагностика |
| Decoder heteromode | Убрано неверное требование одинаковых физических режимов у эквивалентов; подготовлен launch adapter | Разные held/press могут делить бонус; реальная политика не запускалась |
| R3 корпус и micro | Исправлено покрытие duration и положения функции; 4140pytest/Ruff/77frontend; micro600updates76,46 с,режимы 1392/1392,errors0 | `corpus_r3/micro/verification.json` |
| R3 probe | 1352,91 с; runtime1/32,DEV51critical; aliases27/516 | `corpus_r3/probe/verification.json`; допуска нет |
| R4 relational corpus | 576TRAINгрупп заменены 192S/D/U-контрастами; heldout записи сохранены; 4146pytest/Ruff/78frontend | `corpus_r4/corpus_integrity_verification.json` |
| R4 micro | 1200updates146,24 с; при.99 SAME0/24,DIFFERENT0/24,U24/24 | `corpus_r4/micro/verification.json`; probe не запускалась |
| Readout capacity | Диагностический pair fit1200updates7,77 с; focal72/72, но 35nonfocal ошибок | `corpus_r4/micro/postmortem/readout_capacity/`; не допуск, веса не перенесены |
| Rehearsal control | По 1200updates; uniform не прошёл, focal/other50:50 дал 5640/5640 при.99 | `…/readout_rehearsal/`; sampling и вес менялись вместе, чистый эффект коэффициента не выделен |
| R5 objective/micro | Введён.5focal+.5other; 4174pytest/Ruff/78frontend; micro1200updates144,42 с,6critical,S24/24,D4/24,U24/24 | `corpus_r5/micro/verification.json`; probe не запускалась |
| R5 gradients/joint control | Большая норма pair не доказала конфликт; joint/frozen контроль 241,89 с: joint прошёл 264TRAINviews,frozen нет | `corpus_r5/micro/postmortem/joint_vs_frozen/`; это продолжение, не fresh/generalization |
| R6 micro | Увеличен только предел короткой пробы до 2400updates/600 с; прошла на 1400updates166,64 с,D23/24,errors0 | `corpus_r6/micro/verification.json` |
| R6 probe | 1340,04 с; runtime5/32,DEV34critical; aliases124/516 | `corpus_r6/probe/verification.json`; исходное состояние последней передачи |
| R6 frozen-разбор | 1800TRAIN+1800DEV59,13 с; высокий raw accuracy не даёт составного payablecoverage | `corpus_r6/probe/postmortem/train_dev_diagnostic.json` |
| Pair population A/B | По 1200updates161,47 с; SAME уверенность улучшилась, aliases425→385/1566,режимы 4922→4652/8190 ухудшились | `…/pair_population_control/`; балансировка отвергнута |
| R7 grounding fix | Исправлены wheel-direction, articleA, смешение мыши/клавиш и стрелок;6499 ложных упоминаний устранены на 133572TRAINwitnesses,пропусков 0 | `grounding_r7/grounding_prevalence.json`;357 целевых тестов |
| R7 парный micro | По 1800updates391,31 с; исправленная ветвь прошла, старая не прошла affected applicability24/41 | `grounding_r7/micro_control/verification.json`; эффект доказан на этойTRAINпанели |
| R7 технический отказ 1 | Старый provenance hash корпуса не принят reader; exit1,12,94 с,updates0 | `grounding_r7/probe_attempt1_preflight_failure/`; ошибка подготовки |
| R7 технический отказ 2 | Reuse-loader требует candidates, которого нет в factorized manifest; updates0 | `grounding_r7/probe_attempt2_cache_preflight_failure/`; reuse отменён, данные сохранены |
| R7 fresh retry2 | Новые признаки и прежние тексты/метки;4413pytest/Ruff/frontend;1353,82 с,3816updates;runtime1/32,DEV29critical | `grounding_r7/probe/verification.json`; допуска нет |
| Продолжение оптимизации | +1200updates до 5016,359,98 с; aliases175/1128 eligible,held1942/3180,press1530/2376; shortgate не пройден | `grounding_r7/probe/postmortem/optimization_control/` |
| LR A/B | Ещё 1000updates: LR.0003 лучше.001; aliases302/1128 против 121/1128; у повышенногоLR21rawcritical | `…/lr_pair_control/`; увеличениеLR отвергнуто |
| Gradient interference | 16batches512TRAINviews,0updates,35,05 с; фактическое направлениеAdamW обычно улучшает целевойloss | `…/gradient_interference.json`; доминирующий конфликт не подтверждён |
| Cross-mode audit | Same-mode276/486 противcross-mode26/642; в 12 ручных группах противоречий не найдено | `…/cross_mode_alias_audit.json`; корреляция, не причина |
| Pair blocker oracle | 1800TRAINviews,152,15 с; baseline302/1128; oracleS401,D532,both1108 | `…/pair_blocker_audit/`; oracle не является качеством модели |
| Required margin A/B | По 1000updates437,70 с; B eligiblealiases986/1128,held2986/3180,press2199/2376,critical0 | `…/required_margin_control/`; полезныйTRAINэффект, не перенос |
| R8 implementation | Новый staged recipe:3816base+2200continuation+1000margin; архитектура не изменена;253044requiredTRAINedges проверены | `required_margin_r8/protocol.json`, `margin_conformance.json` |
| R8 проверки | 4455pytest,Ruff,81frontend;584,33 с,источники неизменны; отдельно 78 целевых и 21adapter-regression | `required_margin_r8/checks_full.json`, `.cache/r8_policy_adapter_checks.json` |
| R8 probe | 7016updates224512 предъявлений,3496,81 сtrainer;exit0;runtime7/32,DEV76critical;aliases195/516,held2339/3720,press2028/2898 | `required_margin_r8/probe/verification.json`; full/PPO запрещены gate |
| R8 trajectory audit | R8 иdiagnosticB совпали поhead/AdamW/RNG/cursor; TRAINpairT2.113489,DEV3.758374 | `…/postmortem/trajectory_temperature_audit.json`; расхождения траектории не найдено |
| Продолжение 25 октября | Пользователь снял общий дедлайн; проверены отсутствие обучения и сохранность результатов; рабочее состояние обновлено | Нативная карточка осталась старой; это явно отмечено выше |
| Temperature control | Точное воспроизведение 1800TRAIN+1800DEV;110,01 с,exit0,RSS2485813248 байт;0updates | `…/transfer_diagnostic_20261025/verification.json` |
| Его результат | TRAIN aliases986→576/1566;DEV288→195/516 приold→finalT; DEVcritical144→76 | Температура влияет, но oldT ухудшает ошибки и не является исправлением |
| Локализация compact | Все 27 уверенныхSAME→DIFFERENT и 15false_split вfamily07,5 независимых ситуаций;послеapp теряются 2383/3199 правильных платежей | Готовыеlogits; новыхforward/fit нет |
| Трассировка входа | SharedWASDanchors подтверждены;truncation/cachemisalignment не обнаружены;весьтекст доступенcross-attention | Это гипотеза трудности обучения, не доказанная потеря информации |
| Compact контроль 32 текстов | Подготовлены 8TRAINгруппRU/EN original/expanded; семантика проверена;8anchors,206–308tokens;Ruff/compile прошли | `…/compact_control_20261025/`; inference ещё НЕ запускался на момент отчёта |
| Новый lexical дефект | Первая RUформулировка явно называлаA, но Englisharticleguard подавил её;неудачныйpreflight сохранён | `…/compact_control_20261025/failed_wording_preflight.json`; исправление в работе, влияние настарыйprobe не доказано |

### Реализованные механизмы и интеграция

- Раздельный текстовый оценщик получает инструкцию, признаки замороженного
  E5 и собственные обучаемые выходные слои. Различаются применимость в игре,
  режим, эквивалентность, отсутствие сведений и отказ по низкой уверенности.
- Реализована утверждённая награда: база 0.0005R, доплата 0.0015R,
  положительный предел 0.01R; штраф −0.01R/H один раз за шаг и отрицательный
  предел 0.01R. Коэффициент полный в циклах 1–32, снижается до нуля к 64.
  Учитываются фактически выполненные действия: удержание одной физической
  клавиши, реальное отпускание после нажатия, единая группа эквивалентов,
  блокировка положительного бонуса при неподдерживаемом действии. Resume
  использует абсолютный номер цикла.
- Загрузчики, артефакты, восстановление и адаптер запуска политики подготовлены
  и проверены тестами. Реальная интеграция с допущенным новым текстовым
  комплектом ещё не выполнена: такого комплекта пока нет.
- Визуальная награда не обучалась заново в этой ветке. Подтверждена идентичность
  сохранённой модели и её validation: 240/240 ячеек; HUD 203204 и progress
  170128 без ошибок. В R8 перепроверены 25 pins, включая 16 исходников.
  Это проверка награды, а не побед политики.
- Дэшборд проверялся в широком и узком окне, с раскрытием и обновлением;
  ошибок консоли не было. После R8 он показывает завершённый неуспешный
  probe и отсутствие новой политики. Исторический остановленный цикл
  не выдавался за новый обучающий запуск.
- Веса, кеши, неуспешные попытки и снимки исходников сохранены. Все изменения
  грязной рабочей копии нельзя приписывать одному этому этапу. Пользовательские
  и предыдущие изменения не откатывались.

### Что ещё не выполнено

Нет допущенного текстового head. Не выполнены его fresh full, calibration,
модельная оценка sealed; реальная совместная интеграция нового допущенного
комплекта; технические два curriculum и два full цикла; три пилота по 80
циклов; основное обучение 128–192 цикла; итоговый validation-выбор и сравнения
с CP0, случайной и прежней моделью; проверки памяти и использования входов;
однократный игровой test.

Победы новой политики по 16 играм НЕ ИЗМЕРЕНЫ. Это не 0%, а отсутствие
такого обучения и оценки. CAL/sealed не использовались для выбора модели;
исторический структурный QC не равен модельному test.

### Ошибки и слабые места в моей работе

1. Я назвал обновление проектного JSON обновлением цели, не обозначив ясно,
   что нативная карточка осталась старой. Запрос на изменение карточки не выполнен.
2. После просьбы об итоговом отчёте я завершил ход, хотя работа была разрешена
   и не окончена. Пользователю пришлось спросить «что стоим?».
3. Отчёт был отложен до завершения цели и заменён обещанием. После повторного
   требования я предоставляю этот обзор выполненной работы сейчас.
4. Две технические попытки R7 потеряны на несовместимости provenance и схемы
   кеша. Это следовало выявить до запуска actor. Нулевое число обновлений
   не делает ошибку подготовки несущественной.
5. Успех маленького TRAIN неоднократно не переносился на DEV/runtime. Пороги
   не позволили объявить ложный успех, но сами micro были слабым прогнозом
   переноса. Знаменатели gold-eligible нужно показывать рядом с полными.
6. Некоторые сравнения меняли несколько факторов: V13/objective,
   rehearsal weight/sampling, base→refined R8. Они не доказывают чистый
   причинный эффект одного изменения.
7. Подготовительные скрипты, проверки и сообщения заняли существенную часть
   работы. Точный суммарный overhead не подсчитан. Повторное использование
   проверенной инфраструктуры и более ранняя проверка конкретных ошибок —
   направления улучшения, а не уже доказанная оптимизация.
8. Последний дефект русской A не объясняет старый R8: в существующих 152
   затронутых TRAIN-фразах A сохранялась через A/D fallback; в DEV и runtime
   таких подавлений не найдено. Нельзя обещать допуск на основании этой правки.

### Последнее исправление во время подготовки отчёта

Защита от английского артикля теперь применяется только к английским
`key/hotkey`, а не к явному русскому «клавишей A». Изменены production
`control_text_grounding.py` и его тесты; добавлены 96 регрессионных случаев.
Все 453 теста grounding и целевой Ruff прошли. Старый снимок R8 не изменён.
Независимая проверка правки ещё ожидается; inference и обучение после неё
не запускались. Весь прежний общий набор из 4455 тестов не следует выдавать
за проверку этой более поздней правки.

### Бюджет и границы отчёта

Длительности опытов приведены выше. Их нельзя складывать в полное время:
часть измеряет trainer, часть supervisor, часть подготовку; интервалы могут
пересекаться. Нативная карточка 25 октября показывала 2903218 tokens и
16756 tracked seconds. Эти счётчики остались у blocked-карточки и не
подтверждают стоимость всей последующей работы. Денежная стоимость не измерена.
Пик RSS последней диагностики — 2485813248 байт. Общий объём всех серий
на диске в момент составления этого отчёта не пересчитывался.

Первичные свидетельства: карточки этого реестра начиная со строки 3077,
`.cache/carryrl_goal_state.json`, указанные verification/result/protocol
и замороженные снимки в runs. Будущий итоговый отчёт должен дополнить
этот обзор, сохраняя неудачи и ограничения. Конечная цель не достигнута.

25 октября, после выдачи сводного отчёта: compact-control зарегистрирован
в `compact_control_20261025/protocol.json`. Семантическая проверка и
grounding preflight пройдены; используются неизменные исходники и веса R8,
более поздняя правка RU-A в этот опыт не входит. Запускается32текстовый
контроль без fit; допуск и устранение DEV-ошибок этим опытом не проверяются.

Первый compact-control остановлен ДО inference: actualexit1,
`RuntimeError: prepared masks changed`,16.74с с подготовкой;
headforward0/E5forward0/updates0, процесс7364 отсутствует, артефакты
проверены. Причина проверки исследуется; исходные protocol/script/output
сохраняются. Это техническая ошибка подготовки, не результат качества.

Независимая проверка точечной RU-A правки завершена без блокеров:
изменён только English-article guard, candidate capture/offset сохранены.
96 новых регрессий покрывают русские явные имена и английские артикли;
453 теста и Ruff пройдены. Сверка с immutable R8 подтвердила узкий diff.
Проводится отдельная проверка неизменности существующих TRAIN/DEV/runtime
anchors без модельного inference; общий прежний pytest не повторяется
без причины и не объявляется проверкой новых строк кода.

### R8-compact-sampling-control-20261025 · подготовка, обучение не запущено

Гипотеза: недостаточная доля существующих compact-примеров в обучении
pair-readout ограничивает перенос на DEV family07. Это проверяемая гипотеза,
не доказанная причина. Архитектура, тексты, разметка, loss и температуры
не меняются. Обе ветви стартуют с одинакового R8 selected7016 и полного
состояния AdamW/RNG/sampler, pair-only, по128updates, batch32, LR.0003.
A продолжает прежний sampler; B в каждом том же batch заменяет последние
4позиции детерминированными TRAIN compact WASD RU/EN и horizontal RU/EN.
Оба исходных cursor продвигаются одинаково; выбор не использует ошибки.
Контроль использует старый immutable R8, отдельная RU-A правка не входит.

Оценка только в конце: прежние фиксированные1800TRAIN и1800DEV-selection,
обе сохранённые температуры; CAL/sealed/runtime не открываются, подбор
checkpoint отсутствует. Локальный лимит1200с, RSS6GiB, свободный диск8GiB.
Критерий полезного сигнала B против A: family07 raw SAME ошибок минимум
на10меньше в≥2независимых контрастах, уверенных SAME-ошибок минимум на9меньше,
правильных DEV aliases минимум на26/516больше; вне compact critical не растут,
TRAIN critical0; при старойT направление эффекта не ухудшается. Frozen weights,
buffers и AdamW moments должны сохраниться. Невыполнение любого пункта
отвергает длинную пробу этого рецепта; прохождение НЕ означает допуск модели.
Это короткий pair-only опыт, поэтому отрицательный результат не доказывает
невозможность иных способов обучения; unary/mode таким fit не исправляются.
Подготовка runner/protocol идёт в `compact_sampling_control_20261025/`
под `required_margin_r8/probe/postmortem/`; никаких обновлений пока нет.

Compact retry1 зарегистрирован: причина первого отказа —296различий
Python tuple против JSON list при равных значениях. Producer теперь
нормализует JSON-представление; строгое сравнение сохранено. Regression
подтверждает roundtrip и отказ при реальном изменении span. Исходная
попытка неизменна; новая — `compact_control_20261025/retry1/`.

Compact retry1 завершён:26,68с с подготовкой, actualexit0, процесс7761
отсутствует, все хеши результатов проверены.16original+16expanded head
forward,16expanded E5,0updates; пикRSS1671774208байт. ПриfinalT фокальная
W/up: оплачивается16/16 original против4/16 expanded; raw pair ошибок0→3;
средний required99 margin7.6016→−1.8203; оба режима верны7/16→0/16.
Все aliases42/64→6/64,critical0→1. ПриoldT фокальная оплата16/16→7/16,
rawerrors0→3;aliases64/64→14/64,critical0→2. Новая формулировка ухудшает
результат. Это причинная чувствительность к замене блока целиком, включающей
длину, позиции и явные имена; не доказательство отдельного механизма masks
и не устранение DEV-ошибок. Простое раскрытие WASD отвергнуто как исправление.
Артефакты: `compact_control_20261025/retry1/output/comparison.json`
и `retry1/verification.json`. Пороги/веса/архитектура не менялись.

RU-A regression audit завершён без inference:15264TRAIN,3600DEV,32runtime;
различий уникальных spans/объединённых символьных масок0.152TRAINстроки
добавляют только дубликат уже существующегоA-span;boolean masks прежние.
6000CAL/sealedстрок пропущены доJSON parsing. Это подтверждает отсутствие
влияния данного fix на старые входы R8, а не исправление качества R8.
`compact_control_20261025/ru_a_regression_audit.json`, actualexit0,90,98с.

Уточнение критериев sampling-контроля ДО fit: TRAIN critical должен
оставаться0 в ОБЕИХ ветвях при ОБЕИХ фиксированных температурах.
Неоднозначность обнаружена независимым review и устранена до регистрации
финального protocol; прежние admission gates не меняются.

Sampling A/B зарегистрирован: `compact_sampling_control_20261025/protocol.json`;
исходный draft сохранён отдельно. Независимый review пройден, обе ветви
проверяют TRAIN critical0 при обеихT. Реальный trace:compact156/4096 уA
против652/4096 уB,512различных inject-строк. Подготовка9.85с,
0forward/0updates; начинается короткий128update/ветвь контроль.

Sampling A/B завершён: по128updates,250.06с с подготовкой, actualexit0;
процесс8016отсутствует, хеши checkpoints проверены. Frozen weights/buffers/
AdamW moments по проверкам runner неизменны, исходные условия воспроизведены.
ПикRSS2626863104байт. Рецепт ОТКЛОНЁН.
DEV finalT: raw family07 SAME ошибок A73/B70 (нужно−10,получено−3),
уверенных SAME ошибок30/27 (нужно−9,получено−3),aliases208/516 уобеих
ветвей (нужно+26,получено0); critical внеcompact28→29 ухудшились.
ОбщийDEVcritical75→73 не заменяет остальные критерии. OldT aliases312→307/516,
critical145→157,внеcompact67→79. TRAINcritical0 обеветви/обеT;это недопуск.
A/B TRAINaliases finalT672/1566 и665/1566; oldT1029/1566 и1031/1566.
Увеличение доли существующих compact-примеров при pair-only128updates
не дало достаточного переноса. Длинныйprobeэтого samplingrecipe запрещён
по зарегистрированным критериям. Никакие веса не продвигаются.
Артефакты: `compact_sampling_control_20261025/output/result.json`,
`A_final_assessment.json`, `B_final_assessment.json`, `verification.json`.
Первая служебная проверка hashes ошибочно ожидала плоскую схему artifacts;
остановилась до записи verification. Повтор использовал фактические nested
path/sha256 записи; файлы опыта не менялись, повторногоfitнебыло.

### Исправление состояния нативной цели, 25 октября

По прямому запросу «ну поставь цель заново» вызван create_goal. Вызов
УСПЕШЕН: нативная карточка теперь active, без общего дедлайна, со всеми
исходными этапами, seeds, gates и требованием полного отчёта. Прежнее
утверждение ассистента о невозможности пересоздания было ошибочным:
описание инструмента оказалось недостаточным основанием не пробовать
прямой разрешённый вызов. Это исправляет текущее состояние, а не означает
достижение научной/игровой цели. Старый goal не объявлялся завершённым.
Новые нативные счётчики начались с0; прежние2903218tokens/16756trackedseconds
сохранены в состоянии и отчёте, история опытов не обнулена. Раздел отчёта
выше о blocked-карточке описывает состояние ДО этого успешного пересоздания.

### R8-joint-refinement-control-20261025 · подготовка

После отказа sampling-рецепта проверяется другая ограниченная гипотеза:
замороженные общие/unary слои ограничивают полезное уточнение представления
при required-margin. Одна новая ветвь B размораживает существующий head
целиком; контроль A128 pair-only переиспользуется из завершённого sampling-
опыта по точным хешам исходногоcheckpoint,AdamW,RNG,loss и batch trace.
Это не повтор sampling и не изменение архитектуры/E5/данных/порогов.
Обе ветви начинаются с original R8selected7016; одинаковые128basebatches,
LR.0003,WD.01,clip1,старыйCE+marginT2.113489151,без нового resampling.
У B сохраняются AdamW moments общих слоёв с6016шага: они не сбрасываются.
TRAIN/DEV панели и обе температуры прежние; final-only, CAL/sealed закрыты.
Локальный предел1200с/RSS6GiB/свободныйдиск8GiB. Рецепт/runner готовятся;
критерии эффекта фиксируются до регистрации/fit. Modelforward иfit пока0.

Joint-refinement критерии сигнала зафиксированы до fit: B против сохранённойA
должен уменьшить raw SAME ошибки family07 минимум на10 в≥2независимых
контрастах, уверенные SAME-ошибки приfinalT минимум на9, увеличить верные
DEV aliases минимум на26/516. Critical внеcompact и общие DEVcritical не
могут вырасти ни при одной из двухT; TRAINcritical0 уобеихветвей/обеT.
ПриoldT направления изменений compactwrong/aliases не хужеA. Это строгий
критерий полезности короткой правки, не admission. При провале длиннаяпроба
этого рецепта не запускается. Все54параметра существующегоhead доступны
обучению вB; E5неизменен. Контроль A повторно не обучается и не оценивается.
