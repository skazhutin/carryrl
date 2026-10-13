"""Export existing training artifacts to local TensorBoard without touching the trainer."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path

from PIL import Image

from carryrl.dashboard.data import TASK_LABELS, RunReader
from carryrl.dashboard.playback import PlaybackCache

LOGGER = logging.getLogger(__name__)
_SOURCE_TAG = "_carryrl/source"
_MAX_EVENT_FILES = 512
_MAX_EVENT_BYTES = 64 * 1024 * 1024


def _sdk():
    try:
        from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
        from tensorboard.compat.proto import (
            event_pb2,
            summary_pb2,
            tensor_pb2,
            tensor_shape_pb2,
            types_pb2,
        )
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
    except ImportError as error:
        raise RuntimeError("TensorBoard is not installed. Run with uv run --locked "
                           "--extra monitor carryrl-monitor.") from error
    return (EventFileWriter, LegacyEventFileLoader, event_pb2, summary_pb2,
            tensor_pb2, tensor_shape_pb2, types_pb2)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _ratio(numerator, denominator):
    if _finite(numerator) and _finite(denominator) and denominator > 0:
        return numerator / denominator
    return None


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def _atomic_json(path: Path, value: dict):
    descriptor, temporary = tempfile.mkstemp(prefix=".export-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class TensorBoardExporter:
    """Idempotent artifact sidecar with event-backed recovery and advisory cursors.

    Each source update is one event record containing its summaries and source
    marker. The event is flushed before the atomic cursor write. On restart,
    markers in actual event records are authoritative, so a crash between those
    operations neither loses an update nor duplicates its points.
    """

    def __init__(self, runs: Path | str, logdir: Path | str, *, reader: RunReader | None = None):
        self.reader = reader or RunReader(runs)
        self.logdir = Path(logdir).resolve()
        if self.logdir == self.reader.root or self.reader.root in self.logdir.parents:
            raise ValueError("TensorBoard logdir must be outside the training runs directory")
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.sdk = _sdk()
        self._writers = {}
        self._locks = {}
        self._sources = {}
        self._dirty = set()
        self._playbacks = PlaybackCache()

    def _run_directory(self, run_id):
        self.reader.run_path(run_id)  # Validates identifier and source symlinks.
        path = (self.logdir / run_id).resolve()
        if self.logdir not in path.parents:
            raise ValueError("TensorBoard run directory escapes its logdir")
        if path == self.reader.root or self.reader.root in path.parents:
            raise ValueError("TensorBoard run directory must be outside "
                             "the training runs directory")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _recover(self, run_id, path):
        _, loader, *_ = self.sdk
        sources = {}
        files = sorted(path.glob("events.out.tfevents.*"))
        if len(files) > _MAX_EVENT_FILES:
            raise RuntimeError(f"Too many event files to recover safely in {path}")
        for file in files:
            if file.resolve().parent != path or file.stat().st_size > _MAX_EVENT_BYTES:
                raise RuntimeError(f"Unsafe or oversized event file: {file}")
            for event in loader(str(file)).Load():
                for value in event.summary.value:
                    if value.tag == _SOURCE_TAG and value.tensor.string_val:
                        try:
                            marker = json.loads(value.tensor.string_val[0])
                            if isinstance(marker.get("key"), str) \
                                    and isinstance(marker.get("fingerprint"), str):
                                sources[marker["key"]] = marker["fingerprint"]
                        except (ValueError, TypeError, AttributeError) as error:
                            LOGGER.warning("Invalid exporter marker in %s: %s", file, error)
        state = path / "state.json"
        if state.exists():
            try:
                cursor = json.loads(state.read_text())
                prior = cursor.get("sources", {})
                if not isinstance(prior, dict):
                    raise ValueError("sources must be an object")
                if any(sources.get(key) != value for key, value in prior.items()):
                    LOGGER.warning("Cursor/event mismatch for %s; recovering from actual events",
                                   run_id)
            except (OSError, ValueError, AttributeError) as error:
                LOGGER.warning("Invalid export cursor for %s; recovering events: %s", run_id, error)
        return sources

    def _writer(self, run_id):
        if run_id not in self._writers:
            if len(self._writers) >= 64:
                raise RuntimeError("At most 64 concurrent runs can be exported; select --run")
            path = self._run_directory(run_id)
            lock_path = path / ".export.lock"
            if lock_path.is_symlink():
                raise ValueError("Exporter lock must not be a symlink")
            lock = lock_path.open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._sources[run_id] = self._recover(run_id, path)
                writer_type = self.sdk[0]
                self._writers[run_id] = writer_type(str(path), max_queue_size=2, flush_secs=5)
                self._locks[run_id] = lock
                self._dirty.add(run_id)  # Rebuild a missing/stale advisory cursor as well.
            except BaseException:
                lock.close()
                raise
        return self._writers[run_id]

    def _text(self, tag, text, *, source=False):
        _, _, _, summary, tensor, shape, types = self.sdk
        metadata = summary.SummaryMetadata(plugin_data=summary.SummaryMetadata.PluginData(
            plugin_name="carryrl_export" if source else "text"))
        return summary.Summary.Value(
            tag=tag, metadata=metadata,
            tensor=tensor.TensorProto(dtype=types.DT_STRING,
                                      tensor_shape=shape.TensorShapeProto(
                                          dim=[shape.TensorShapeProto.Dim(size=1)]),
                                      string_val=[text.encode()]))

    def _scalar(self, tag, value):
        if not _finite(value):
            return None
        summary = self.sdk[3]
        return summary.Summary.Value(tag=tag, simple_value=float(value))

    def _emit(self, run_id, key, fingerprint, step, values):
        if not _finite(step) or step < 0 or int(step) != step:
            LOGGER.warning("Missing valid decision step for %s/%s; will retry", run_id, key)
            return False
        writer = self._writer(run_id)
        if self._sources[run_id].get(key) == fingerprint:
            return False
        if key in self._sources[run_id] and not key.startswith(("progress", "frame/")):
            LOGGER.warning("Source %s/%s changed after export; recording corrected values",
                           run_id, key)
        _, _, events, summaries, *_ = self.sdk
        values = [value for value in values if value is not None]
        if not values:
            return False
        marker = json.dumps({"key": key, "fingerprint": fingerprint}, sort_keys=True)
        values.append(self._text(_SOURCE_TAG, marker, source=True))
        writer.add_event(events.Event(wall_time=time.time(), step=int(step),
                                      summary=summaries.Summary(value=values)))
        # Do not advance the durable cursor until the containing event is flushed.
        writer.flush()
        self._sources[run_id][key] = fingerprint
        self._dirty.add(run_id)
        return True

    def _save_cursor(self, run_id):
        if run_id in self._dirty:
            _atomic_json(self._run_directory(run_id) / "state.json", {
                "format": "carryrl.tensorboard-export.v1", "run_id": run_id,
                "sources": self._sources[run_id],
            })
            self._dirty.remove(run_id)

    def _rate_values(self, prefix, successes, episodes, unknown=None):
        return [self._scalar(f"{prefix}/success_rate", _ratio(successes, episodes)),
                self._scalar(f"{prefix}/successes", successes),
                self._scalar(f"{prefix}/episodes", episodes),
                self._scalar(f"{prefix}/unknown_episodes", unknown)]

    def _training_values(self, metric):
        values = [self._scalar(f"PPO/{key}", metric.get(key)) for key in
                  ("policy_loss", "value_loss", "entropy", "approx_kl", "gradient_norm")]
        values.extend([self._scalar("progress/rollout", metric.get("rollout")),
                       self._scalar("timing/cycle_seconds", metric.get("cycle_wall_seconds")),
                       self._scalar("train/overall/mean_episode_reward",
                                    metric.get("reward_mean"))])
        values.extend(self._rate_values("train/overall", metric.get("train_successes"),
                                        metric.get("train_episodes"), metric.get("train_unknown")))
        for task, stats in metric.get("tasks", {}).items():
            prefix = f"train/{task}"
            values.extend(self._rate_values(prefix, stats.get("successes"), stats.get("episodes"),
                                            stats.get("unknown")))
            values.extend(self._scalar(f"{prefix}/{tag}", stats.get(field)) for tag, field in
                          (("failures", "failures"), ("unfinished", "unfinished"),
                           ("mean_episode_reward", "mean_reward"), ("mean_steps", "mean_steps"),
                           ("mean_visible_score", "mean_visible_score")))
        return values

    def _optimization_values(self, optimization):
        values = [self._scalar(f"PPO/{key}", optimization.get(key))
                  for key in ("optimizer_steps", "clip_fraction")]
        early_stop = optimization.get("kl_early_stop")
        if isinstance(early_stop, bool):
            values.append(self._scalar("PPO/kl_early_stop", int(early_stop)))
        values.append(self._scalar("curriculum/difficulty",
                                   optimization.get("curriculum_difficulty")))
        values.extend(self._scalar(f"train/{task}/valid_transitions", count) for task, count in
                      optimization.get("task_valid_transitions", {}).items())
        return values

    def _reward_report_values(self, reward):
        values = []
        rows = ["| Раздел | Кадры | Из игровых эпизодов | Синтетические glyphs |",
                "|---|---:|---:|---:|"]
        for partition, metrics in reward["partitions"].items():
            prefix = f"reward_screen/{partition}"
            for field, value in metrics.items():
                if field in {"success", "failure"}:
                    values.extend(self._scalar(f"{prefix}/{field}_{key}", item)
                                  for key, item in value.items())
                else:
                    values.append(self._scalar(f"{prefix}/{field}", value))
            rows.append(f"| {partition} | {metrics.get('frames', '?')} | "
                        f"{metrics.get('played_episode_frames', '?')} | "
                        f"{metrics.get('synthetic_glyph_frames', '?')} |")
        gate = reward.get("gate", {})
        passed = gate.get("passed")
        if isinstance(passed, bool):
            values.append(self._scalar("reward_screen/gate_passed", int(passed)))
        glyphs = reward.get("synthetic_glyph_coverage", {})
        if isinstance(glyphs, dict):
            values.extend(self._scalar(f"reward_screen/synthetic_glyph_coverage/{field}",
                                       glyphs.get(field))
                          for field in ("training_frames", "validation_frames"))
        scope = (f"**Модель награды:** `{reward.get('model_version', 'неизвестно')}`. "
                 f"**Selector конфигурации:** `{reward.get('selector')}`. "
                 f"**Gate passed:** {passed}.\n\n"
                 "Распознавание поддерживаемого обычного HUD по RGB с supervised метками. "
                 "Это проверка чтения score/событий, не автономное открытие награды и "
                 "не доказательство, что игровой агент освоил задачи.\n\n"
                 f"Опубликованный scope: {reward.get('scope', 'не указан')}. "
                 f"Inputs: {reward.get('inputs', 'не указаны')}.\n\n"
                 "`played_episode_frames` — кадры игровых эпизодов; "
                 "`synthetic_glyph_frames` — отдельно отрисованные примеры HUD. "
                 "Синтетические glyphs не считаются сыгранными эпизодами. "
                 "Familiar validation повторяет обученные glyph identities в других сценах; "
                 "unseen validation проверяет отдельные темы, canonical validation — "
                 "каноническое представление в рамках поддерживаемого словаря.\n\n"
                 + "\n".join(rows))
        if gate.get("failures"):
            scope += "\n\nGate failures: " + json.dumps(gate["failures"], ensure_ascii=False)
        if glyphs:
            scope += "\n\nSynthetic coverage:\n```json\n" + json.dumps(
                glyphs, ensure_ascii=False, indent=2) + "\n```"
        values.append(self._text("monitor/reward_scope", scope))
        return values

    def _evaluation_values(self, evaluation, namespace="eval"):
        values = []
        for task, stats in [("overall", evaluation), *evaluation.get("tasks", {}).items()]:
            prefix = f"{namespace}/{task}"
            for policy in ("agent", "random"):
                values.extend([
                    self._scalar(f"{prefix}/{policy}_success_rate", _ratio(
                        stats.get(f"{policy}_successes"), stats.get(f"{policy}_episodes"))),
                    self._scalar(f"{prefix}/{policy}_successes", stats.get(f"{policy}_successes")),
                    self._scalar(f"{prefix}/{policy}_episodes", stats.get(f"{policy}_episodes")),
                    self._scalar(f"{prefix}/{policy}_unknown_episodes",
                                 stats.get(f"{policy}_unknown")),
                ])
            values.append(self._scalar(f"{prefix}/mean_visible_score",
                                       stats.get("mean_visible_score")))
        return values

    @staticmethod
    def _legacy_evaluation(evaluation):
        # Keep fingerprints of already-exported tags stable when adding details.
        value = {key: item for key, item in evaluation.items()
                 if key not in {"details", "stochastic", "context"}}
        value["tasks"] = {task: {key: item for key, item in stats.items() if key != "details"}
                          for task, stats in evaluation.get("tasks", {}).items()}
        return value

    def _evaluation_detail_values(self, evaluation, namespace="eval"):
        values = []
        for task, stats in [("overall", evaluation), *evaluation.get("tasks", {}).items()]:
            prefix = f"{namespace}/{task}"
            details = stats.get("details", {})
            agent_rate = _ratio(stats.get("agent_successes"), stats.get("agent_episodes"))
            random_rate = _ratio(stats.get("random_successes"), stats.get("random_episodes"))
            gap = agent_rate - random_rate if agent_rate is not None and random_rate is not None \
                else None
            values.append(self._scalar(f"{prefix}/success_gap", gap))
            for policy in ("agent", "random"):
                for field in ("mean_visible_score", "failures", "timeouts", "mean_success_steps"):
                    values.append(self._scalar(f"{prefix}/{policy}_{field}",
                                               details.get(f"{policy}_{field}")))
                for field in ("failure", "timeout"):
                    count = details.get(f"{policy}_{field}s")
                    values.append(self._scalar(f"{prefix}/{policy}_{field}_rate",
                                               _ratio(count, stats.get(f"{policy}_episodes"))))
        return values

    def _playback_values(self, playback, final_revision):
        metadata, summary = playback.metadata, self.sdk[3]
        task = metadata["task"]
        revision = metadata["collection_weights_revision"]
        caption = (f"**{TASK_LABELS.get(task, task)} — сохранённая тренировочная игра.** "
                   f"Цикл {metadata['rollout']}, эпизод {metadata['episode']}, "
                   f"seed {metadata['seed']}. Стохастическая политика ревизии **{revision}**, "
                   f"последний checkpoint имеет ревизию **{final_revision}**. "
                   "Эпизод собран до обновления PPO этого цикла.\n\n"
                   f"{metadata['steps']} физических действий, {metadata['frame_count']} "
                   "кадров с конечным состоянием. Кадр t показывается **до действия t**; "
                   "команда для перехода указана под кадром. Последний кадр показывает "
                   "результат последнего действия. **10 FPS** соответствует фиксированному "
                   "интервалу контроллера 0.1 с; это скорость воспроизведения симуляции, "
                   "а не скорость реального сбора данных.\n\n"
                   f"Источник проверен по SHA256 `{metadata['npz_sha256']}`. "
                   "Это тренировочный архив, не видео итоговой замороженной проверки.")
        return [summary.Summary.Value(tag=f"playback/{task}", image=summary.Summary.Image(
            height=playback.height, width=playback.width, colorspace=3,
            encoded_image_string=playback.encoded_gif)),
            self._text(f"monitor/playback/{task}", caption)]

    @staticmethod
    def _metric_guide(tasks, config=None):
        rows = ["| ID | Механика |", "|---|---|"]
        rows.extend(f"| `{task}` | {TASK_LABELS.get(task, task)} |" for task in tasks)
        guide = [
            "**Как понять, что агент играет лучше**",
            "Смотрите `eval/<игра>/agent_success_rate` вместе с `random_success_rate`, "
            "`success_gap` (агент минус random) и `agent_mean_visible_score`. Это отдельные "
            "validation уровни, замороженная детерминированная политика и независимый "
            "визуальный проверщик. Сравнивайте одну игру на одинаковой проверке между циклами.",
            "`agent_failure_rate` и `agent_timeout_rate` показывают поражения и исчерпание "
            "лимита шагов. `agent_mean_success_steps` измеряет длину только успешных эпизодов; "
            "при отсутствии побед точки нет. Знаменатель success_rate включает все "
            "проверенные эпизоды, неизвестные исходы показаны отдельно.",
            "В первом пилоте было лишь **4 проверочных эпизода на игру**: "
            "один успех меняет долю на 25 процентных пунктов. "
            "Это грубый сигнал на маленькой выборке; overall объединяет разные механики, "
            "а сырые score разных игр имеют разные смыслы. В других запусках проверяйте "
            "фактические `agent_episodes` и протокол отбора уровней.",
            "`train/<игра>/...` показывает стохастические тренировочные эпизоды. "
            "`mean_episode_reward` — обученный оценщик награды. Более высокая награда "
            "или меньшее число шагов могут означать более раннюю смерть; сами по себе "
            "они не доказывают улучшение игры.",
            "`PPO/...` — диагностика обновления оптимизатора (последняя эпоха цикла): "
            "loss, entropy, KL и norm градиента. Их изменение подтверждает изменение "
            "процесса обучения, но не освоение игровых задач. `optimizer_steps` считает "
            "применённые шаги за полный цикл; `clip_fraction` относится к последней эпохе, "
            "`kl_early_stop` — фактическая остановка по KL. `curriculum/difficulty` описывает "
            "учебную сложность, а сложность проверки опубликована отдельно в Text.",
            "**Исходная оценка:** `baseline_kind=untrained_initial` — начало обучения; "
            "`stage_initial` — начало нового этапа, `resume_initial` — продолжение. "
            "Локальный цикл 0 сам по себе не означает новые случайные веса. Проверяйте "
            "`baseline_policy_previously_trained` и `evaluation_difficulty` в Text; "
            "сравнивайте результаты при одинаковой сложности проверки.",
            "**Просмотр игры:** Images → `playback/<игра>` — GIF последнего сохранённого "
            "стохастического тренировочного эпизода. Кадр t показан до действия t, затем "
            "идёт конечный кадр; подпись содержит использованную ревизию политики. "
            "Это архив тренировочной игры, а не видео последнего замороженного checkpoint.",
            "\n".join(rows),
        ]
        if config and config.get("eval_stochastic"):
            guide.insert(2, "**Два режима проверки:** `eval/...` сохраняет прежнюю "
                         "детерминированную политику (greedy). `eval_stochastic/...` — "
                         "отдельная замороженная политика с выборкой действий, "
                         f"фиксированным seed {config.get('eval_seed')} и без обновления весов. "
                         "Сравнивайте её с random в том же namespace, отдельно от greedy. "
                         "Seed и способ выбора уровней записаны в Text → eval_stochastic.")
        return "\n\n".join(guide)

    @staticmethod
    def _progress_text(snapshot):
        run, progress = snapshot["run"], snapshot["progress"]
        source = "фаза выведена из сохранённых файлов" if progress["source"] == "inferred" \
            else "телеметрия процесса"
        cycle_text = f"**Цикл:** {progress['rollout']} из {run['total_rollouts']}. " \
            f"**Завершённых эпизодов цикла:** {progress['episodes_completed']} " \
            f"из {progress['episodes_per_rollout']}."
        if progress["stage"] == "finished":
            cycle_text = f"**Завершено циклов:** {run['rollouts']} из {run['total_rollouts']}. " \
                "Текущего собираемого эпизода нет."
        lines = [f"**Статус:** {run['status']}. **Фаза:** {progress['stage']} ({source}).",
                 cycle_text,
                 f"**Действия:** {run['decisions']}. **Устройство:** {run['device']}. "
                 f"**Процесс существует:** {run['process_alive']}.",
                 f"**Артефакты обновлены:** {run['data_updated_utc']}."]
        if progress.get("task"):
            lines.append(f"**Задача:** {progress['task']}; шаг {progress.get('episode_step')}.")
        evaluations = snapshot.get("evaluations", [])
        if evaluations:
            context = evaluations[-1].get("context", {})
            if context.get("evaluation_difficulty") is not None:
                lines.append(f"**Сложность проверки:** {context['evaluation_difficulty']}.")
            baseline = next((item["context"] for item in evaluations
                             if item.get("context", {}).get("baseline_kind") is not None), None)
            if baseline:
                trained = baseline.get("baseline_policy_previously_trained")
                prior = "уже обучалась" if trained is True else "раньше не обучалась" \
                    if trained is False else "история обучения не указана"
                lines.append(f"**Исходная оценка:** `{baseline['baseline_kind']}`; "
                             f"политика {prior}. Начало этапа не означает случайные новые веса.")
        lines.extend(f"\n{warning}" for warning in snapshot.get("warnings", []))
        lines.append("\nPPO: показана последняя эпоха каждого цикла. Успех измеряет независимый "
                     "визуальный проверщик; mean_episode_reward — средняя сумма предсказанных "
                     "наград за эпизод. Оценка использует замороженную политику.")
        return "\n\n".join(lines)

    def export_run(self, run_id):
        """Export new immutable source points and changed status/image summaries."""
        snapshot = self.reader.snapshot(run_id)
        if not isinstance(snapshot.get("config", {}).get("tasks"), list):
            raise ValueError(f"{run_id} is not a generalist sequence training run")
        self._writer(run_id)
        exported = 0
        config_text = json.dumps(snapshot["config"], ensure_ascii=False, indent=2)
        exported += self._emit(run_id, "config", _fingerprint(snapshot["config"]), 0,
                               [self._text("monitor/config", f"```json\n{config_text}\n```")])
        for metric in snapshot["metrics"]:
            legacy_metric = {key: value for key, value in metric.items() if key != "optimization"}
            exported += self._emit(run_id, f"train/{metric['rollout']}",
                                   _fingerprint(legacy_metric),
                                   metric.get("decisions"), self._training_values(metric))
            optimization = metric.get("optimization")
            if optimization:
                exported += self._emit(run_id, f"optimization/{metric['rollout']}",
                                       _fingerprint(optimization), metric.get("decisions"),
                                       self._optimization_values(optimization))
        for evaluation in snapshot["evaluations"]:
            greedy = {key: value for key, value in evaluation.items()
                      if key not in {"stochastic", "context"}}
            legacy = self._legacy_evaluation(evaluation)
            exported += self._emit(run_id, f"eval/{evaluation['rollout']}",
                                   _fingerprint(legacy),
                                   evaluation.get("decisions"), self._evaluation_values(evaluation))
            exported += self._emit(run_id, f"eval_details/{evaluation['rollout']}",
                                   _fingerprint(greedy), evaluation.get("decisions"),
                                   self._evaluation_detail_values(evaluation))
            context = evaluation.get("context")
            if context:
                protocol = "```json\n" + json.dumps(context, ensure_ascii=False, indent=2) + "\n```"
                exported += self._emit(run_id, f"eval_protocol/{evaluation['rollout']}",
                                       _fingerprint(context), evaluation.get("decisions"),
                                       [self._text("monitor/evaluation_protocol", protocol)])
            stochastic = evaluation.get("stochastic")
            if stochastic:
                stochastic_text = ("Замороженная стохастическая проверка, веса не обучаются. "
                                   f"Seed: {stochastic.get('evaluation_seed')}. "
                                   f"Уровни: {stochastic.get('level_selection')}. "
                                   f"Цикл {evaluation['rollout']}, "
                                   f"{evaluation['decisions']} тренировочных действий. "
                                   "Greedy остаётся в eval/...; эта выборка — eval_stochastic/...")
                exported += self._emit(run_id, f"eval_stochastic/{evaluation['rollout']}",
                                       _fingerprint(stochastic), evaluation.get("decisions"),
                                       self._evaluation_values(stochastic, "eval_stochastic")
                                       + self._evaluation_detail_values(
                                           stochastic, "eval_stochastic")
                                       + [self._text("monitor/eval_stochastic", stochastic_text)])
        guide = self._metric_guide(snapshot["config"]["tasks"], snapshot["config"])
        exported += self._emit(run_id, "metric_guide", _fingerprint(guide), 0,
                               [self._text("monitor/how_to_read_metrics", guide)])
        for index, checkpoint in enumerate(snapshot["checkpoints"]):
            exported += self._emit(run_id, f"checkpoint/{checkpoint['rollout']}",
                                   _fingerprint(checkpoint), checkpoint.get("decisions"), [
                                       self._scalar("progress/checkpoints_saved", index + 1),
                                       self._scalar("progress/weights_revision",
                                                    checkpoint.get("weights_revision")),
                                   ])
        reward = snapshot.get("reward")
        if reward and "partitions" in reward:
            exported += self._emit(run_id, "reward_screen_report", _fingerprint(reward), 0,
                                   self._reward_report_values(reward))
        elif reward:
            exported += self._emit(run_id, "reward_bootstrap", _fingerprint(reward), 0,
                                   [self._scalar(f"reward_bootstrap/{key}", value)
                                    for key, value in reward.items()] + [
                                       self._text("monitor/reward_scope", "Оценщик награды: "
                                                  "обучение распознаванию общего HUD на "
                                                  "скриптовых траекториях. Метрики относятся "
                                                  "к отдельным validation уровням; это "
                                                  "не автономное открытие награды.")])
        # Exclude wall-clock age/elapsed from the fingerprint: unchanged polling
        # must not add duplicate points, while process/phase/file changes refresh text.
        progress_data = {"progress": snapshot["progress"], "status": snapshot["run"]["status"],
                         "decisions": snapshot["run"]["decisions"],
                         "alive": snapshot["run"]["process_alive"],
                         "warnings": snapshot["warnings"]}
        contexts = [evaluation["context"] for evaluation in snapshot["evaluations"]
                    if evaluation.get("context")]
        if contexts:
            progress_data["evaluation_context"] = contexts
        exported += self._emit(run_id, "progress", _fingerprint(progress_data),
                               snapshot["run"].get("decisions"), [
                                   self._text("monitor/status", self._progress_text(snapshot))])
        for task in snapshot["config"]["tasks"]:
            frame = self.reader.frame_metadata(run_id, task=task)
            if not frame:
                continue
            key, fingerprint = f"frame/{task}", _fingerprint(frame)
            if self._sources[run_id].get(key) != fingerprint:
                png = self.reader.frame_png(run_id, task, expected=frame)
                if png is None:
                    LOGGER.warning("No valid saved image yet for %s/%s; retrying", run_id, task)
                    continue
                with Image.open(io.BytesIO(png)) as image:
                    height, width = image.height, image.width
                summary = self.sdk[3]
                action_text = json.dumps(frame["action"], ensure_ascii=False)
                caption = (f"**Последний сохранённый эпизод: {task}.** "
                           f"Цикл {frame['rollout']}, эпизод {frame['episode']}, "
                           f"шаг {frame['step']}. Время: {frame['captured_utc']}. "
                           f"Источник: {frame['source']}.\n\n"
                           f"Последнее физическое действие: `{action_text}`. "
                           "Это сохранённый кадр, не прямая видеотрансляция.")
                exported += self._emit(run_id, key, fingerprint,
                                       snapshot["run"].get("decisions"), [
                    summary.Summary.Value(tag=f"last_saved/{task}", image=summary.Summary.Image(
                        height=height, width=width, colorspace=3, encoded_image_string=png)),
                    self._text(f"monitor/last_saved/{task}", caption),
                ])
            try:
                playback = self._playbacks.get(self.reader, run_id, task, expected=frame)
            except (OSError, ValueError, KeyError) as error:
                LOGGER.warning("Cannot render playback for %s/%s; retrying: %s",
                               run_id, task, error)
                continue
            if playback:
                final_revision = snapshot["checkpoints"][-1]["weights_revision"] \
                    if snapshot["checkpoints"] else None
                playback_fingerprint = _fingerprint(playback.metadata | {
                    "latest_checkpoint_revision": final_revision})
                exported += self._emit(run_id, f"playback/{task}", playback_fingerprint,
                                       snapshot["run"].get("decisions"),
                                       self._playback_values(playback, final_revision))
        self._writers[run_id].flush()
        self._save_cursor(run_id)
        return exported

    def poll(self, run_ids=None):
        run_ids = run_ids or [row["id"] for row in self.reader.list_runs()["runs"]]
        exported = 0
        for run_id in run_ids[:64]:
            try:
                exported += self.export_run(run_id)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                LOGGER.error("Cannot export %s; retrying next poll: %s", run_id, error)
        if len(run_ids) > 64:
            LOGGER.warning("Run selection exceeds 64; select specific runs with --run")
        return exported

    def close(self):
        for run_id, writer in self._writers.items():
            try:
                writer.flush()
                self._save_cursor(run_id)
            finally:
                writer.close()
                lock = self._locks.pop(run_id)
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()
        self._writers.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--logdir", type=Path, default=Path(".cache/tensorboard"))
    parser.add_argument("--run", action="append", dest="run_ids")
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error("--interval must be a finite number of at least one second")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        exporter = TensorBoardExporter(args.runs, args.logdir)
    except (RuntimeError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    LOGGER.info("Exporting local training artifacts to %s", args.logdir.resolve())
    try:
        while True:
            exported = exporter.poll(args.run_ids)
            if exported:
                LOGGER.info("Flushed %s new source updates", exported)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        LOGGER.info("Exporter stopped; training was not modified")
    finally:
        exporter.close()


if __name__ == "__main__":
    main()
