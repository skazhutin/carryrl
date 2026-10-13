"""Publish only the registered MLX imitation run to a readable TensorBoard view."""

from __future__ import annotations

import argparse
import fcntl
import io
import json
import logging
import math
import signal
import time
from pathlib import Path

from carryrl.dashboard.data import TASK_LABELS
from carryrl.dashboard.focus_server import FocusServer, FocusServerError
from carryrl.dashboard.tensorboard import TensorBoardExporter, _atomic_json, _fingerprint, _finite

LOGGER = logging.getLogger(__name__)
RUN_LABEL = "СЕЙЧАС — MLX — 16 игр"
GUIDE_TAG = "00 Как смотреть/Начните здесь"
WIN_TAG = "01 Качество игры/01 Победы — все игры (%)"
QUICK_WIN_TAG = "05 Быстрая проверка/01 Основная модель/01 Победы — все игры (%)"
NLL_TAG = "02 Обучение/Ошибка повторения действий (NLL)"
_PHASES = {"loading": "Загрузка модели", "collecting": "Сбор экранных примеров",
           "optimizing": "Обновление весов PPO",
           "encoding": "Подготовка кадров учителя",
           "imitation": "Повторение действий учителя",
           "initial_evaluation": "Первая проверка модели до обучения",
           "evaluation": "Проверка замороженной модели",
           "final_evaluation": "Итоговая полная контрольная проверка",
           "saving": "Сохранение модели",
           "finished": "Обучение завершено"}
_STATES = {"running": "Выполняется", "completed": "Завершено", "failed": "Ошибка",
           "interrupted": "Прервано"}
_MAX_JSON_BYTES = 16 * 1024 * 1024


def _json(path: Path) -> dict:
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"JSON artifact too large: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _epoch(value):
    if _finite(value) and value >= 0 and int(value) == value:
        return int(value)
    raise ValueError("Missing nonnegative integer epoch")


class FocusExporter(TensorBoardExporter):
    """A single visible run, with event-backed idempotence inherited from the exporter.

    A source switch archives the entire previous event directory before publishing
    the new source. Equal epoch numbers from different runs therefore never merge.
    """

    def __init__(self, runs: Path | str, logdir: Path | str, registry: Path | str,
                 *, serve_port: int | None = None):
        super().__init__(runs, logdir)
        self.registry = Path(registry).resolve()
        lock_path = self.logdir / ".focus.lock"
        if lock_path.is_symlink():
            raise ValueError("Focus lock must not be a symlink")
        self._focus_lock = lock_path.open("a")
        try:
            fcntl.flock(self._focus_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._focus_lock.close()
            raise
        self._active = None
        self._cohorts = {}
        self._overview = None
        self.server = FocusServer(self.logdir, serve_port) if serve_port is not None else None

    def _run_directory(self, run_id):
        path = self.logdir / RUN_LABEL
        if path.resolve().parent != self.logdir or path.is_symlink():
            raise ValueError("Focused event directory escapes logdir")
        path.mkdir(exist_ok=True)
        return path

    def _artifact(self, run, relative):
        path = run / relative
        if run not in path.resolve().parents:
            raise ValueError("Run artifact escapes its source directory")
        return path

    def _source(self):
        registry = _json(self.registry)
        value = registry.get("run")
        if not isinstance(value, str):
            raise ValueError("Registry must identify an actual local run")
        run = Path(value).resolve()
        if run.parent != self.reader.root:
            raise ValueError("Registered run must be directly within --runs")
        self.reader.run_path(run.name)
        metadata = _json(self._artifact(run, "metadata.json"))
        if metadata.get("kind") not in {"imitation_training", "sequence_training"}:
            raise ValueError("Focused view requires imitation_training or sequence_training")
        config = _json(self._artifact(run, "config.json"))
        if config.get("training_backend") != "mlx":
            raise ValueError("Focused MLX view requires an actual MLX source")
        tasks = config.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 16 or set(tasks) != set(TASK_LABELS):
            raise ValueError("Focused 16-game view requires the declared sixteen games")
        identity = _fingerprint({"path": str(run), "metadata": metadata, "config": config})
        return run, metadata, config, identity

    def _select(self, run, identity):
        state_path = self.logdir / "focus-state.json"
        if state_path.is_symlink():
            raise ValueError("Focus state must not be a symlink")
        state = _json(state_path) if state_path.exists() else {}
        self._cohorts = (state.get("cohorts", {})
                         if state.get("identity") == identity else {})
        if state.get("identity") not in {None, identity}:
            super().close()
            self._sources.clear()
            current = self.logdir / RUN_LABEL
            if current.exists():
                if current.is_symlink() or current.resolve().parent != self.logdir:
                    raise ValueError("Unsafe focus directory")
                archive = self.logdir.with_name(self.logdir.name + "_archive")
                if archive.is_symlink():
                    raise ValueError("Focus archive must not be a symlink")
                archive.mkdir(exist_ok=True)
                destination = archive / f"{time.time_ns()}-{state['identity'][:12]}"
                current.rename(destination)
            self._active = None
        _atomic_json(state_path, {"identity": identity, "run": str(run),
                                  "cohorts": self._cohorts})
        self._active = run.name

    def _training_rows(self, run, counter="epoch"):
        path = self._artifact(run, "training.jsonl")
        if not path.exists():
            return []
        if counter == "rollout":
            # PPO journals include physical-action trajectories; use the existing
            # incremental bounded reader rather than rereading the whole journal.
            rows = []
            for entry in self.reader._training(path):
                metric = entry["metric"]
                _epoch(metric.get("rollout"))
                rows.append({"rollout": metric["rollout"], "epochs": [metric],
                             "optimizer_steps": metric.get("optimization", {}).get(
                                 "optimizer_steps")})
            return rows
        if path.stat().st_size > _MAX_JSON_BYTES:
            raise ValueError("Training journal exceeds bounded focused reader")
        rows = []
        for line in path.read_text().splitlines(keepends=True):
            if not line.endswith("\n"):
                break  # Concurrent append: retry the incomplete last record next poll.
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Training record must be an object")
            _epoch(row.get(counter))
            rows.append(row)
        return rows

    def _quality_values(self, summary, prefix):
        episodes = summary.get("episodes_started")
        successes = summary.get("outcomes", {}).get("success")
        values = []
        if (_finite(episodes) and episodes > 0 and int(episodes) == episodes
                and _finite(successes) and 0 <= successes <= episodes):
            values.append(self._scalar(f"{prefix}/01 Победы — все игры (%)",
                                       100 * successes / episodes))
        values.extend(self._scalar(f"{prefix}/{label}", summary.get(field)) for label, field in (
            ("05 Эпизодов проверки", "episodes_started"),
            ("03 Истёк лимит времени", "time_limit_episodes"),
            ("04 Неизвестный исход", "unknown_outcomes")))
        for task, stats in summary.get("tasks", {}).items():
            if task not in TASK_LABELS:
                continue
            count = stats.get("episodes_started")
            won = stats.get("outcomes", {}).get("success")
            if _finite(count) and count > 0 and _finite(won) and 0 <= won <= count:
                values.append(self._scalar(f"{prefix}/02 Победы — {TASK_LABELS[task]} (%)",
                                           100 * won / count))
        return values

    def _evaluation_cohort(self, evaluation, config, full):
        """Keep frozen metrics comparable only within the same declared episode bank."""
        records = evaluation.get("records") or []
        fields = ("policy", "task", "seed", "level_index", "level_id", "instruction",
                  "difficulty", "level_bank_seed", "presentation", "initial_frame_sha256",
                  "semantic_level_id", "runtime_fingerprint")
        descriptor = {
            "evaluation": {key: evaluation.get(key) for key in (
                "split", "evaluation_seed", "evaluation_difficulty", "level_selection",
                "level_bank_seed", "protocol_sha256", "random_distribution",
                "visual_config", "presentation", "environment_source_sha256", "suite_version")},
            "config": {key: config.get(key) for key in (
                "visual_config", "max_episode_steps", "env_image_size", "episode_instructions",
                "level_bank_seed")},
            "records": [{key: record.get(key) for key in fields}
                        for record in records if isinstance(record, dict)],
            "counts": {policy: {
                "episodes_started": summary.get("episodes_started"),
                "tasks": {task: stats.get("episodes_started")
                          for task, stats in summary.get("tasks", {}).items()}}
                       for policy, summary in evaluation.get("summaries", {}).items()
                       if isinstance(summary, dict)},
        }
        signature = _fingerprint(descriptor)
        varied = ((evaluation.get("visual_config") or config.get("visual_config") or {}).get(
                      "mode") == "varied"
                  or evaluation.get("presentation") is not None
                  or any(record.get("presentation") is not None
                         for record in records if isinstance(record, dict)))
        kind = "full" if full else "quick"
        # Preserve the existing tags only for a single canonical cohort. A declared
        # varied-presentation check always carries its own scope in the chart name.
        if not varied:
            self._cohorts.setdefault(kind, signature)
        suffix = ""
        if varied or self._cohorts.get(kind) != signature:
            label = "Вариативные условия" if varied else "Другие условия"
            suffix = f"/{label} [{signature[:12]}]"
        return signature, suffix

    def _publish_playbacks(self, run_id):
        from PIL import Image

        from carryrl.dashboard.overview_data import OverviewReader

        if self._overview is None:
            self._overview = OverviewReader(self.reader.root, self.registry)
        items = self._overview.snapshot(include_experiments=False)["playbacks"]["items"]
        latest = {}
        for item in items:
            if item["run_id"] != run_id:
                continue
            task = item["task"]
            if task not in latest or item["revision"] > latest[task]["revision"]:
                latest[task] = item
        exported = 0
        for task, item in latest.items():
            filename = item["url"].rsplit("/", 1)[-1]
            encoded = self._overview.media(run_id, filename)
            if encoded is None:
                continue
            with Image.open(io.BytesIO(encoded)) as frame:
                width, height = frame.size
            caption = (
                f"**{item['label']} — запись игры замороженной модели.** "
                f"Эпоха {item['epoch']}, версия весов {item['revision']}; "
                f"{item['steps']} игровых действий. Это отдельная контрольная игра агента, "
                "а не действия программного учителя и не прямой эфир обучения.\n\n"
                "[Пауза, перемотка и результаты по играм]"
                f"(http://127.0.0.1:6009/#/games/{task})."
            )
            summary = self.sdk[3]
            values = [summary.Summary.Value(
                tag=f"playback/{task}", image=summary.Summary.Image(
                    height=height, width=width, colorspace=3, encoded_image_string=encoded)),
                self._text(f"06 Записи игры/{item['label']}", caption)]
            exported += self._emit(run_id, f"playback/{task}", _fingerprint(item),
                                   _epoch(item["epoch"]), values)
        return exported

    def poll(self):
        run, metadata, config, identity = self._source()
        self._select(run, identity)
        run_id = run.name
        self._writer(run_id)
        imitation = metadata["kind"] == "imitation_training"
        counter = "epoch" if imitation else "rollout"
        unit = "эпоха" if imitation else "цикл PPO"
        rows = self._training_rows(run, counter)
        status_path = self._artifact(run, "status.json")
        status = _json(status_path) if status_path.exists() else {}
        step = _epoch(status.get("epochs" if imitation else "rollouts", 0))
        alive, _ = self.reader._process(run)
        exported = 0
        local_updates = 0 if rows or step == 0 else None
        for row in rows:
            point = _epoch(row[counter])
            if imitation:
                complete = row.get("complete")
                values = [self._scalar(NLL_TAG, row.get("nll")),
                          self._scalar("02 Обучение/Обновлений весов", row.get("optimizer_steps")),
                          self._scalar("02 Обучение/Эпоха завершена", int(complete))
                          if isinstance(complete, bool) else None]
                if complete is False:
                    values.append(self._text("00 Как смотреть/Незавершённая эпоха",
                                             f"Эпоха {point} остановлена до конца. NLL относится "
                                             "только к обработанным примерам этой эпохи."))
            else:
                updates = row.get("optimizer_steps")
                if (_finite(updates) and updates >= 0 and int(updates) == updates
                        and local_updates is not None):
                    local_updates += int(updates)
                else:
                    local_updates = None
                epochs = row.get("epochs") or []
                final = epochs[-1] if epochs and isinstance(epochs[-1], dict) else {}
                values = [self._scalar(f"02 Обучение PPO/{label}", final.get(field))
                          for label, field in (
                              ("Энтропия действий", "entropy"),
                              ("Изменение политики (KL)", "approx_kl"),
                              ("Ошибка политики", "policy_loss"),
                              ("Ошибка оценки ценности", "value_loss"))]
                values.append(self._scalar("02 Обучение PPO/Обновлений весов за цикл",
                                           row.get("optimizer_steps")))
                values.append(self._scalar("02 Обучение PPO/Обновлений весов за этот этап",
                                           local_updates))
            key = f"bc/{point}" if imitation else f"ppo/{point}"
            exported += self._emit(run_id, key, _fingerprint(row), point, values)
        last_full = last_quick = None
        evaluation_dir = self._artifact(run, "evaluations")
        if evaluation_dir.exists():
            for path in sorted(evaluation_dir.glob(f"{counter}_*.json")):
                evaluation = _json(self._artifact(run, str(path.relative_to(run))))
                if (evaluation.get("policy_and_reward_frozen") is not True
                        or evaluation.get("split") not in {"validation", "test"}):
                    LOGGER.warning("Skipping non-frozen or undeclared evaluation: %s", path)
                    continue
                epoch = _epoch(evaluation.get(counter))
                summaries = evaluation.get("summaries", {})
                greedy = summaries.get("frozen_deterministic")
                if not isinstance(greedy, dict):
                    continue
                full = evaluation.get("level_selection") == "explicit covering level indices"
                signature, suffix = self._evaluation_cohort(evaluation, config, full)
                if full:
                    last_full = epoch
                    prefixes = ("01 Качество игры",
                                "03 Проверка со случайным выбором действий модели",
                                "04 Случайный контроль")
                else:
                    last_quick = epoch
                    prefixes = ("05 Быстрая проверка/01 Основная модель",
                                "05 Быстрая проверка/02 Стохастическая модель",
                                "05 Быстрая проверка/03 Случайный контроль")
                prefixes = tuple(prefix + suffix for prefix in prefixes)
                values = self._quality_values(greedy, prefixes[0])
                for policy, prefix in zip(("frozen_stochastic", "generic_physical_random"),
                                          prefixes[1:], strict=True):
                    summary = summaries.get(policy)
                    if isinstance(summary, dict):
                        values.extend(self._quality_values(summary, prefix))
                bank_seed = evaluation.get('level_bank_seed', config.get('level_bank_seed', 0))
                protocol = (f"{unit.capitalize()} {epoch}; замороженная обученная модель; "
                            f"split: {evaluation['split']}; "
                            f"seed: {evaluation.get('evaluation_seed', 'не указан')}; "
                            f"сложность: {evaluation.get('evaluation_difficulty', 'не указана')}; "
                            f"seed банка: {bank_seed}; "
                            f"условия: {suffix.strip(' /') or 'исходный контрольный набор'}; "
                            f"идентификатор набора: {signature}; "
                            f"выбор уровней: {evaluation.get('level_selection', 'не указан')}. "
                            f"{unit.capitalize()} 0 — модель до этого этапа обучения.")
                values.append(self._text("00 Как смотреть/Условия проверки", protocol))
                cohort = "full" if full else "quick"
                key = f"frozen_eval/{cohort}-{signature}/{epoch}/{path.name}"
                exported += self._emit(run_id, key,
                                       _fingerprint(evaluation), epoch, values)
        # A quick check can be replaced by a full check at the same final epoch.
        # Retain the last genuinely published quick/full checks from event markers.
        for key in self._sources[run_id]:
            parts = key.split("/")
            if len(parts) == 4 and parts[0] == "frozen_eval" and parts[2].isdigit():
                point = int(parts[2])
                if parts[1] == "full" or parts[1].startswith("full-"):
                    last_full = max(last_full if last_full is not None else point, point)
                elif parts[1] == "quick" or parts[1].startswith("quick-"):
                    last_quick = max(last_quick if last_quick is not None else point, point)
        phase = _PHASES.get(status.get("phase"),
                            "Обучение завершено" if status.get("status") == "completed"
                            else "Обучение PPO по игровым эпизодам" if not imitation
                            else "Фаза ещё не записана")
        if not imitation and status.get("phase") == "collecting":
            phase = "Сбор собственных игровых эпизодов PPO"
        state = _STATES.get(status.get("status"), "Статус ещё не записан")
        process = "работает" if alive is True else "остановлен" if alive is False else "неизвестен"
        if last_full is None and last_quick is None:
            check = "Первая проверка ещё не завершена; точки побед появятся после неё."
        else:
            full_check = (f"Последняя полная проверка: {unit} {last_full}."
                          if last_full is not None else "Полная проверка ещё не завершена.")
            quick_check = (f"Последняя быстрая проверка: {unit} {last_quick}."
                           if last_quick is not None else "Быстрых проверок пока нет.")
            check = f"{full_check} {quick_check}"
        incomplete = " Последняя записанная эпоха неполная." if imitation and rows and \
            rows[-1].get("complete") is False else ""
        if imitation:
            method = (
                "**Это обучение повторению действий программного учителя (BC), не RL.** "
                "Учитель создаёт RGB-примеры; графики побед показывают отдельные игры "
                "обученной модели с замороженными весами. Действия учителя не входят в победы.")
            optimization = (
                "Уменьшение **02 Обучение / Ошибка повторения действий (NLL)** означает "
                "более точное повторение учителя; само по себе оно не доказывает улучшение игры.")
            axis = "**Горизонтальная ось Step — номер эпохи BC.** Эпоха 0 — исходная модель."
            progress = (f"Записано эпох: {step}; обновлений весов: "
                        f"{status.get('optimizer_steps', 0)}.")
        else:
            method = (
                "**Это обучение с подкреплением PPO на собственных игровых эпизодах.** "
                "Награду предсказывает замороженный визуальный оценщик. Графики побед "
                "показывают отдельную проверку обученной модели с замороженными весами; "
                "начальные веса могут быть уже обучены на предыдущем этапе.")
            optimization = (
                "**02 Обучение PPO** показывает энтропию действий и изменение политики (KL) "
                "на последней оптимизационной эпохе каждого цикла, ошибки политики и оценки "
                "ценности, а также число обновлений весов за цикл. Эти показатели не являются "
                "процентом побед. Обновления за этот этап суммируются из записей обучения; "
                "номер версии весов может включать обновления предыдущего этапа.")
            axis = "**Горизонтальная ось Step — номер цикла PPO (rollout).** Цикл 0 — начало этапа."
            total = config.get("total_rollouts", status.get("rollouts_total"))
            target = f" из {total}" if _finite(total) else ""
            updates = str(local_updates) if local_updates is not None else "неизвестно"
            progress = (f"Записано циклов PPO: {step}{target}; "
                        f"игровых действий: {status.get('decisions', 0)}; "
                        f"обновлений весов за этот этап: {updates}.")
        guide = (
            "# Текущий этап: MLX, 16 игр\n\n"
            f"**{state}. {phase}.** Процесс обучения: {process}. {progress} "
            f"{check}{incomplete}\n\n{method}\n\n"
            "В Scalars откройте **01 Качество игры** и кривую **01 Победы — все игры** "
            "с указанными условиями: "
            "победы от 0 до 100% в полной проверке с покрытием уровней "
            f"при выборе самого вероятного действия. {optimization} "
            "**03** — стохастическая проверка модели; **04** — независимые случайные "
            "действия для сравнения. Эти три группы относятся к полным проверкам. "
            "**05 Быстрая проверка** содержит отдельные кривые для сокращённой выборки "
            "уровней. Полная и быстрая проверки используют разные наборы эпизодов; "
            "их проценты не соединяются одной кривой.\n\n"
            "Наборы с разными seed, сложностью, оформлением или протоколом имеют "
            "отдельные кривые с идентификатором условий. Покрытие уровней относится "
            "только к указанному набору и не доказывает универсальное качество.\n\n"
            f"{axis} Неизвестные исходы остаются в знаменателе; тайм-ауты показаны "
            "отдельно. Между проверками новые победы не вычисляются.\n\n"
            "[Общая страница: прогресс, история качества и записи игр]"
            "(http://127.0.0.1:6009/). В Images → playback находятся записи последней "
            "сохранённой проверки модели.\n\n"
            "[Архив предыдущих запусков](http://127.0.0.1:6008/) открывается отдельно.\n\n"
            f"Источник: `{run}`. Идентификатор источника: `{identity}`. "
            f"Backend: {config['training_backend']}."
        )
        exported += self._emit(run_id, "progress", _fingerprint(guide), step,
                               [self._text(GUIDE_TAG, guide)])
        try:
            exported += self._publish_playbacks(run_id)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            LOGGER.warning("Cannot publish frozen playbacks; retrying next poll: %s", error)
        self._writers[run_id].flush()
        self._save_cursor(run_id)
        _atomic_json(self.logdir / "focus-state.json", {
            "identity": identity, "run": str(run), "cohorts": self._cohorts})
        if self.server is not None:
            self.server.ensure_generation(identity)
        return exported

    def close(self):
        try:
            super().close()
        finally:
            try:
                if self.server is not None:
                    self.server.close()
            finally:
                if not self._focus_lock.closed:
                    fcntl.flock(self._focus_lock, fcntl.LOCK_UN)
                    self._focus_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--registry", type=Path, default=Path(".cache/active_mlx_launch.json"))
    parser.add_argument("--logdir", type=Path, default=Path(".cache/tensorboard_current"))
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--serve-port", type=int,
                        help="Own a TensorBoard server and reset its cache on source switches")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error("--interval must be a finite number of at least one second")
    if args.serve_port is not None and not 1 <= args.serve_port <= 65535:
        parser.error("--serve-port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    exporter = FocusExporter(args.runs, args.logdir, args.registry, serve_port=args.serve_port)

    def stop(_signal, _frame):
        raise KeyboardInterrupt

    previous_term_handler = signal.signal(signal.SIGTERM, stop)
    try:
        while True:
            try:
                count = exporter.poll()
                if count:
                    LOGGER.info("Published %s source updates", count)
            except FocusServerError:
                raise
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                if args.once:
                    raise
                LOGGER.error("Cannot publish active source; retrying next poll: %s", error)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        LOGGER.info("Focused exporter stopped")
    except FocusServerError as error:
        parser.exit(2, f"{error}\n")
    finally:
        try:
            exporter.close()
        finally:
            signal.signal(signal.SIGTERM, previous_term_handler)


if __name__ == "__main__":
    main()
