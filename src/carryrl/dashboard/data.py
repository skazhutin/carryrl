"""Bounded readers for the sequence trainer's concurrently written artifacts.

Only JSON summaries and the final image member of an episode archive are read.
Training checkpoints and full RGB trajectories are never deserialized here.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import threading
import zipfile
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

TASK_LABELS = {
    "navigation": "Навигация", "reaction": "Реакция", "mouse": "Наведение мышью",
    "memory": "Память", "timing": "Тайминг", "platforming": "Платформинг",
    "steering": "Рулевое управление", "camera": "Движение и камера",
    "interaction": "Взаимодействие", "causality": "Кнопка и дверь",
    "key_door": "Ключ и дверь", "goal_choice": "Выбор цели", "hazards": "Опасности",
    "shooting": "Стрельба", "ui": "Меню и инвентарь", "chain": "Цепочка действий",
}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_ROLLOUT = re.compile(r"rollout_(\d+)\Z")
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_LINE_BYTES = 16 * 1024 * 1024
_MAX_IMAGE_BYTES = 4 * 1024 * 1024
_MAX_RECORDS_PER_REFRESH = 1024
_MAX_BYTES_PER_REFRESH = 64 * 1024 * 1024
_MAX_CACHED_TRAINING_ROWS = 1024
LOGGER = logging.getLogger(__name__)


def utc_stamp(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat() if timestamp is not None \
        else datetime.now(UTC).isoformat()


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _number(value: Any) -> int | float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def _mean(values: list[Any]) -> float | None:
    values = [value for item in values if (value := _number(item)) is not None]
    return sum(values) / len(values) if values else None


def _valid_identifier(value: str) -> bool:
    return bool(_IDENTIFIER.fullmatch(value)) and value not in {".", ".."}


class RunReader:
    """Thread-safe read adapter with bounded caches and incremental JSONL parsing."""

    def __init__(self, runs: Path | str):
        self.root = Path(runs).resolve()
        self._lock = threading.RLock()
        self._json_cache: OrderedDict[Path, tuple[tuple[int, int], dict]] = OrderedDict()
        self._logs: OrderedDict[Path, dict] = OrderedDict()
        self._frames: OrderedDict[Path, tuple[tuple[int, int], bytes]] = OrderedDict()
        self._warned: OrderedDict[tuple, None] = OrderedDict()

    def _safe(self, path: Path) -> Path:
        resolved = path.resolve()
        if resolved == self.root or self.root not in resolved.parents:
            raise ValueError("Path is outside the runs directory")
        return resolved

    def run_path(self, run_id: str) -> Path:
        if not _valid_identifier(run_id):
            raise ValueError("Invalid run identifier")
        path = self._safe(self.root / run_id)
        if not path.is_dir():
            raise FileNotFoundError(run_id)
        return path

    def _signature(self, path: Path) -> tuple[int, int] | None:
        try:
            stat = self._safe(path).stat()
            return stat.st_mtime_ns, stat.st_size
        except (OSError, ValueError):
            return None

    def _warn_once(self, key: tuple, message: str):
        if key not in self._warned:
            LOGGER.warning(message)
            self._warned[key] = None
            if len(self._warned) > 128:
                self._warned.popitem(last=False)

    def _is_training(self, path: Path, metadata: dict, status: dict) -> bool:
        config = metadata.get("config", {})
        if not isinstance(config, dict) or not isinstance(config.get("tasks"), list):
            return False
        kind = metadata.get("kind")
        if kind is not None:
            return kind == "sequence_training"
        if metadata.get("checkpoint") is not None or metadata.get("updates") == 0:
            return False  # Standalone evaluate-v1 output, even with a training config.
        rollouts = status.get("rollouts")
        return (isinstance(rollouts, int) and not isinstance(rollouts, bool) and rollouts >= 0
                or self._signature(path / "training.jsonl") is not None
                or bool(self._json(path / "latest.json"))
                or isinstance(metadata.get("provenance"), dict)
                and isinstance(metadata.get("created_utc"), str)
                and metadata.get("optimizer") is not None)

    def _json(self, path: Path) -> dict:
        signature = self._signature(path)
        if signature is None:
            return {}
        if signature[1] > _MAX_JSON_BYTES:
            LOGGER.warning("Skipping oversized JSON artifact: %s", path)
            return {}
        cached = self._json_cache.get(path)
        if cached and cached[0] == signature:
            self._json_cache.move_to_end(path)
            return cached[1]
        try:
            with self._safe(path).open("rb") as stream:
                value = json.loads(stream.read(_MAX_JSON_BYTES + 1))
            if not isinstance(value, dict):
                return {}
        except (OSError, ValueError, UnicodeError) as error:
            LOGGER.warning("Cannot read JSON artifact %s; retrying next refresh: %s", path, error)
            return {}
        self._json_cache[path] = signature, value
        if len(self._json_cache) > 256:
            self._json_cache.popitem(last=False)
        return value

    @staticmethod
    def _training_summary(row: dict) -> dict:
        epochs = row.get("epochs") or []
        last = epochs[-1] if epochs and isinstance(epochs[-1], dict) else {}
        episodes = row.get("episode_records") or []
        episodes = [{key: ep.get(key) for key in ("task", "steps", "outcome", "reward_sum")}
                    for ep in episodes if isinstance(ep, dict)]
        metric = {key: _number(row.get(key)) for key in
                  ("rollout", "decisions", "cycle_wall_seconds", "weights_revision")}
        metric.update({key: _number(last.get(key)) for key in
                       ("policy_loss", "value_loss", "entropy", "approx_kl", "gradient_norm")})
        metric["reward_mean"] = _mean([ep["reward_sum"] for ep in episodes])
        metric["train_successes"] = sum(
            isinstance(ep["outcome"], dict) and ep["outcome"].get("known") is True
            and ep["outcome"].get("success") is True for ep in episodes)
        metric["train_episodes"] = len(episodes)
        optimization = {key: value for key in
                        ("optimizer_steps", "curriculum_difficulty")
                        if (value := _number(row.get(key))) is not None}
        clip_fraction = _number(last.get("clip_fraction"))
        if clip_fraction is not None:
            optimization["clip_fraction"] = clip_fraction
        early_stop = row.get("kl_early_stop", last.get("kl_early_stop"))
        if isinstance(early_stop, bool):
            optimization["kl_early_stop"] = early_stop
        task_transitions = row.get("task_transitions")
        if isinstance(task_transitions, dict):
            optimization["task_valid_transitions"] = {
                task: count for task, value in task_transitions.items()
                if isinstance(task, str) and (count := _number(value)) is not None}
        if optimization:
            metric["optimization"] = optimization
        outcomes = [ep["outcome"] for ep in episodes if isinstance(ep["outcome"], dict)]
        metric["train_unknown"] = len(episodes) - sum(
            outcome.get("known") is True for outcome in outcomes)
        metric["tasks"] = {}
        for task in dict.fromkeys(ep["task"] for ep in episodes if isinstance(ep["task"], str)):
            selected = [ep for ep in episodes if ep["task"] == task]
            known = [ep["outcome"] for ep in selected if isinstance(ep["outcome"], dict)
                     and ep["outcome"].get("known") is True]
            metric["tasks"][task] = {
                "episodes": len(selected),
                "successes": sum(outcome.get("success") is True for outcome in known),
                "failures": sum(outcome.get("failure") is True for outcome in known),
                "unfinished": sum(not outcome.get("success") and not outcome.get("failure")
                                  for outcome in known),
                "unknown": len(selected) - len(known),
                "mean_reward": _mean([ep["reward_sum"] for ep in selected]),
                "mean_steps": _mean([ep["steps"] for ep in selected]),
                "mean_visible_score": _mean([outcome.get("score") for outcome in known]),
            }
        return {"metric": metric, "episodes": episodes}

    def _training(self, path: Path) -> list[dict]:
        signature = self._signature(path)
        if signature is None:
            return []
        cache = self._logs.get(path)
        if cache is None or signature[1] < cache["offset"]:
            cache = {"offset": 0, "rows": [], "signature": None}
            self._logs[path] = cache
        if cache["signature"] == signature:
            return cache["rows"]
        caught_up = False
        try:
            with self._safe(path).open("rb") as stream:
                stream.seek(cache["offset"])
                initial_offset = cache["offset"]
                # Limits a single refresh even when an old run has a very large log.
                for _ in range(_MAX_RECORDS_PER_REFRESH):
                    start = stream.tell()
                    line = stream.readline(_MAX_LINE_BYTES + 1)
                    if not line or not line.endswith(b"\n"):
                        caught_up = True
                        break  # Keep the offset before an in-flight trailing record.
                    cache["offset"] = stream.tell()
                    if len(line) > _MAX_LINE_BYTES:
                        break
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict):
                            cache["rows"].append(self._training_summary(row))
                    except (ValueError, UnicodeError, TypeError) as error:
                        LOGGER.warning("Malformed training record at byte %s in %s: %s",
                                       start, path, error)
                        cache["offset"] = max(cache["offset"], start + len(line))
                    if stream.tell() - initial_offset >= _MAX_BYTES_PER_REFRESH:
                        break
                cache["rows"] = cache["rows"][-_MAX_CACHED_TRAINING_ROWS:]
        except (OSError, ValueError) as error:
            LOGGER.warning("Cannot read training history %s; retrying: %s", path, error)
            return cache["rows"]
        cache["signature"] = signature if caught_up else None
        self._logs.move_to_end(path)
        if len(self._logs) > 8:
            self._logs.popitem(last=False)
        return cache["rows"]

    def list_runs(self) -> dict:
        with self._lock:
            runs = []
            if self.root.is_dir():
                for path in self.root.iterdir():
                    if not _valid_identifier(path.name) or not path.is_dir():
                        continue
                    metadata = self._json(path / "metadata.json")
                    config = metadata.get("config", {})
                    status = self._json(path / "status.json")
                    if not self._is_training(path, metadata, status):
                        continue
                    runs.append({"id": path.name, "label": path.name,
                                 "status": status.get("status", "unknown"),
                                 "started_utc": metadata.get("created_utc"),
                                 "tasks_count": len(config["tasks"]),
                                 "device": config.get("device")})
            runs.sort(key=lambda row: row["started_utc"] or "", reverse=True)
            return {"runs": runs, "generated_utc": utc_stamp()}

    def _legacy_final_progress(self, path: Path, metadata: dict, status: dict) -> dict | None:
        """Attach the final evaluation only when its own completed run agrees."""
        if not self._is_training(path, metadata, status) or status.get("status") not in {
            "completed", "resource_stopped",
        }:
            return None
        status_checkpoint = status.get("latest_checkpoint")
        latest_checkpoint = self._json(path / "latest.json").get("checkpoint")
        if not isinstance(status_checkpoint, str) or status_checkpoint != latest_checkpoint:
            return None
        try:
            checkpoint = self._safe(Path(status_checkpoint))
        except ValueError:
            return None
        if checkpoint.parent != (path / "checkpoints").resolve():
            return None
        manifest = self._json(checkpoint / "manifest.json")
        evaluation = self._json(path / "evaluation.json")
        if evaluation.get("policy_and_reward_frozen") is not True:
            return None
        if status.get("weights_revision") is not None \
                and status["weights_revision"] != manifest.get("weights_revision"):
            return None
        for key in ("rollout", "decisions"):
            status_key = "rollouts" if key == "rollout" else key
            value = manifest.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0 \
                    or status.get(status_key) != value:
                return None
        for eval_key, manifest_key in (("agent_sha256", "agent_sha256"),
                                      ("reward_sha256", "reward_sha256"),
                                      ("checkpoint_weights_revision", "weights_revision"),
                                      ("weights_revision", "weights_revision")):
            if evaluation.get(eval_key) is not None \
                    and evaluation[eval_key] != manifest.get(manifest_key):
                return None
        times = [self._signature(file) for file in
                 (checkpoint / "manifest.json", path / "evaluation.json", path / "status.json")]
        if any(value is None for value in times) or not times[0][0] <= times[1][0] <= times[2][0]:
            return None
        return {"rollout": manifest["rollout"], "decisions": manifest["decisions"]}

    def _evaluation(self, path: Path, *, fallback_progress: dict | None = None) -> dict | None:
        data = self._json(path)
        summaries = data.get("summaries", {})
        if not isinstance(summaries, dict):
            return None
        agent = summaries.get("frozen_deterministic", {})
        random = summaries.get("generic_physical_random", {})
        if not agent:
            return None
        records = data.get("records")
        records = records if isinstance(records, list) else []

        def details(agent_summary, random_summary, task=None, agent_policy="frozen_deterministic"):
            result = {}
            for policy, summary, name in ((agent_policy, agent_summary, "agent"),
                                          ("generic_physical_random", random_summary, "random")):
                successes = [row.get("steps") for row in records if isinstance(row, dict)
                             and row.get("policy") == policy
                             and (task is None or row.get("task") == task)
                             and isinstance(row.get("outcome"), dict)
                             and row["outcome"].get("known") is True
                             and row["outcome"].get("success") is True]
                result[f"{name}_mean_visible_score"] = _number(
                    summary.get("mean_visible_final_score"))
                result[f"{name}_failures"] = _number(summary.get("outcomes", {}).get("failure"))
                result[f"{name}_timeouts"] = _number(summary.get("time_limit_episodes"))
                result[f"{name}_mean_success_steps"] = _mean(successes)
            return result

        task_summaries = {}
        for task, value in agent.get("tasks", {}).items():
            baseline = random.get("tasks", {}).get(task, {})
            task_summaries[task] = {
                "agent_successes": value.get("outcomes", {}).get("success"),
                "agent_episodes": value.get("episodes_started"),
                "agent_unknown": value.get("unknown_outcomes",
                                           value.get("outcomes", {}).get("unknown")),
                "random_successes": baseline.get("outcomes", {}).get("success"),
                "random_episodes": baseline.get("episodes_started"),
                "random_unknown": baseline.get("unknown_outcomes",
                                             baseline.get("outcomes", {}).get("unknown")),
                "mean_visible_score": value.get("mean_visible_final_score"),
                "details": details(value, baseline, task),
            }
        match = _ROLLOUT.fullmatch(path.stem)
        rollout = data.get("rollout", int(match[1]) if match else None)
        decisions = data.get("decisions")
        if fallback_progress and rollout in (None, fallback_progress["rollout"]) \
                and decisions in (None, fallback_progress["decisions"]):
            rollout, decisions = fallback_progress["rollout"], fallback_progress["decisions"]
        if not isinstance(rollout, int) or isinstance(rollout, bool) or rollout < 0 \
                or not isinstance(decisions, int) or isinstance(decisions, bool) or decisions < 0:
            self._warn_once((path, self._signature(path)),
                            "Omitting evaluation without unambiguous rollout/decision step: "
                            f"{path}")
            return None
        result = {"rollout": rollout, "decisions": decisions,
                "agent_successes": agent.get("outcomes", {}).get("success"),
                "agent_episodes": agent.get("episodes_started"),
                "agent_unknown": agent.get("unknown_outcomes",
                                           agent.get("outcomes", {}).get("unknown")),
                "random_successes": random.get("outcomes", {}).get("success"),
                "random_episodes": random.get("episodes_started"),
                "random_unknown": random.get("unknown_outcomes",
                                             random.get("outcomes", {}).get("unknown")),
                "mean_visible_score": agent.get("mean_visible_final_score"),
                "details": details(agent, random),
                "tasks": task_summaries}
        context = {key: data[key] for key in
                   ("evaluation_difficulty", "baseline_kind", "baseline_policy_previously_trained",
                    "evaluation_seed", "level_selection") if data.get(key) is not None}
        if context:
            result["context"] = context
        stochastic = summaries.get("frozen_stochastic")
        if isinstance(stochastic, dict):
            stochastic_tasks = {}
            for task, value in stochastic.get("tasks", {}).items():
                baseline = random.get("tasks", {}).get(task, {})
                stochastic_tasks[task] = {
                    "agent_successes": value.get("outcomes", {}).get("success"),
                    "agent_episodes": value.get("episodes_started"),
                    "agent_unknown": value.get("unknown_outcomes",
                                               value.get("outcomes", {}).get("unknown")),
                    "random_successes": baseline.get("outcomes", {}).get("success"),
                    "random_episodes": baseline.get("episodes_started"),
                    "random_unknown": baseline.get("unknown_outcomes",
                                                 baseline.get("outcomes", {}).get("unknown")),
                    "mean_visible_score": value.get("mean_visible_final_score"),
                    "details": details(value, baseline, task, "frozen_stochastic"),
                }
            result["stochastic"] = {
                "policy": "frozen_stochastic", "evaluation_seed": data.get("evaluation_seed"),
                "level_selection": data.get("level_selection"),
                "agent_successes": stochastic.get("outcomes", {}).get("success"),
                "agent_episodes": stochastic.get("episodes_started"),
                "agent_unknown": stochastic.get("unknown_outcomes",
                                                stochastic.get("outcomes", {}).get("unknown")),
                "random_successes": result["random_successes"],
                "random_episodes": result["random_episodes"],
                "random_unknown": result["random_unknown"],
                "mean_visible_score": stochastic.get("mean_visible_final_score"),
                "details": details(stochastic, random, agent_policy="frozen_stochastic"),
                "tasks": stochastic_tasks,
            }
        return result

    def _archives(self, path: Path, task: str | None = None) -> list[Path]:
        parent = path / "trajectories"
        if self._signature(parent) is None:
            return []
        archives = []
        # Only the current/latest rollout is relevant to the preview and inferred stage.
        rollouts = sorted(child for child in parent.iterdir()
                          if _ROLLOUT.fullmatch(child.name) and child.is_dir())[-2:]
        for rollout in rollouts:
            if self._signature(rollout) is None:
                continue
            for sidecar in rollout.glob(f"{task}_*.json" if task else "*.json"):
                if self._signature(sidecar) is not None:
                    archives.append(sidecar)
        return sorted(archives, key=lambda file: self._signature(file) or (0, 0))

    def _selected_frame(self, path: Path, task: str | None) -> tuple[dict | None, Path | None]:
        live = self._json(path / "live.json")
        if live and (not task or task == live.get("task")) \
                and self._signature(path / "live_frame.png"):
            return {"task": live.get("task"), "rollout": live.get("rollout"),
                    "episode": live.get("episode"), "step": live.get("episode_step"),
                    "source": "live", "captured_utc": live.get("updated_utc"),
                    "action": live.get("action")}, path / "live_frame.png"
        for sidecar in reversed(self._archives(path, task)):
            npz = sidecar.with_suffix(".npz")
            if not self._signature(npz):
                continue
            data = self._json(sidecar)
            actions = data.get("actions") or []
            if not data or (task and data.get("task") != task):
                continue
            match = _ROLLOUT.fullmatch(sidecar.parent.name)
            try:
                episode = int(sidecar.stem.rsplit("_", 1)[1]) + 1
            except (ValueError, IndexError):
                episode = None
            return {"task": data.get("task"), "rollout": int(match[1]) if match else None,
                    "episode": episode, "step": len(actions) if actions else None,
                    "source": "last_episode",
                    "captured_utc": utc_stamp(self._signature(sidecar)[0] / 1e9),
                    "action": actions[-1] if actions else None}, npz
        return None, None

    def frame_metadata(self, run_id: str, task: str | None = None) -> dict | None:
        """Read preview metadata without advancing the training history cursor."""
        with self._lock:
            path = self.run_path(run_id)
            if task is not None and not _valid_identifier(task):
                raise ValueError("Invalid task identifier")
            metadata, _ = self._selected_frame(path, task)
            return metadata

    def episode_source(self, run_id: str, task: str, *,
                       expected: dict | None = None) -> tuple[dict, Path] | None:
        """Select a complete saved training episode for bounded playback readers."""
        with self._lock:
            path = self.run_path(run_id)
            if not _valid_identifier(task):
                raise ValueError("Invalid task identifier")
            metadata, archive = self._selected_frame(path, task)
            if not metadata or metadata.get("source") != "last_episode" or archive is None:
                return None
            if expected is not None and metadata != expected:
                return None
            sidecar = self._json(archive.with_suffix(".json"))
            if not sidecar or not isinstance(sidecar.get("npz_sha256"), str):
                return None
            return metadata | {"collection_weights_revision":
                               sidecar.get("collection_weights_revision"),
                               "actions": sidecar.get("actions"),
                               "npz_sha256": sidecar["npz_sha256"],
                               "seed": sidecar.get("seed")}, self._safe(archive)

    def frame_png(self, run_id: str, task: str | None = None, *,
                  expected: dict | None = None) -> bytes | None:
        with self._lock:
            path = self.run_path(run_id)
            if task is not None and not _valid_identifier(task):
                raise ValueError("Invalid task identifier")
            selected, file = self._selected_frame(path, task)
            if expected is not None and selected != expected:
                return None  # Newer episode arrived between metadata and image requests.
            if file is None:
                return None
            signature = self._signature(file)
            cached = self._frames.get(file)
            if cached and cached[0] == signature:
                return cached[1]
            try:
                if file.suffix == ".png":
                    if signature[1] > _MAX_IMAGE_BYTES:
                        return None
                    data = self._safe(file).read_bytes()
                    with Image.open(io.BytesIO(data)) as image:
                        if image.width > 1024 or image.height > 1024:
                            return None
                        image.verify()
                else:
                    with zipfile.ZipFile(self._safe(file)) as archive:
                        member = archive.getinfo("final_frame.npy")
                        if member.file_size > _MAX_IMAGE_BYTES:
                            return None
                        with archive.open(member) as stream:
                            content = stream.read(_MAX_IMAGE_BYTES + 1)
                        header = io.BytesIO(content)
                        version = np.lib.format.read_magic(header)
                        if version == (1, 0):
                            shape, _, dtype = np.lib.format.read_array_header_1_0(header)
                        elif version == (2, 0):
                            shape, _, dtype = np.lib.format.read_array_header_2_0(header)
                        else:
                            return None
                        if dtype != np.uint8 or len(shape) != 3 or shape[-1] != 3 \
                                or min(shape[:2]) < 1 or max(shape[:2]) > 1024:
                            return None
                        frame = np.load(io.BytesIO(content), allow_pickle=False)
                    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 \
                            or max(frame.shape[:2]) > 1024:
                        return None
                    target = io.BytesIO()
                    Image.fromarray(frame).save(target, format="PNG")
                    data = target.getvalue()
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
                LOGGER.warning("Cannot read episode frame %s; retrying: %s", file, error)
                return None  # A concurrently written archive is retried next refresh.
            self._frames[file] = signature, data
            if len(self._frames) > 32:
                self._frames.popitem(last=False)
            return data

    def _process(self, path: Path) -> tuple[bool | None, Path]:
        launch = self.root / f"{path.name}_launch"
        process = self._json(launch / "process.json")
        pid = process.get("command_pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return None, launch
        try:
            os.kill(pid, 0)
            return True, launch
        except ProcessLookupError:
            return False, launch
        except PermissionError:
            return None, launch

    def _tail(self, path: Path) -> str:
        try:
            with self._safe(path).open("rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(0, size - 8192))
                return stream.read(8192).decode("utf-8", errors="replace")
        except (OSError, ValueError):
            return ""

    def snapshot(self, run_id: str, task: str | None = None) -> dict:
        with self._lock:
            path = self.run_path(run_id)
            if task is not None and not _valid_identifier(task):
                raise ValueError("Invalid task identifier")
            metadata = self._json(path / "metadata.json")
            config = metadata.get("config") or {}
            status = self._json(path / "status.json")
            live = self._json(path / "live.json")
            rows = self._training(path / "training.jsonl")
            metrics = [row["metric"] for row in rows]
            checkpoints = []
            for manifest in sorted((path / "checkpoints").glob("rollout_*/manifest.json")):
                value = self._json(manifest)
                signature = self._signature(manifest)
                if value and signature:
                    checkpoints.append({key: value.get(key) for key in
                                        ("rollout", "decisions", "weights_revision")}
                                       | {"name": manifest.parent.name,
                                          "created_utc": utc_stamp(signature[0] / 1e9)})
            evaluations = [value for file in sorted((path / "evaluations").glob("rollout_*.json"))
                           if (value := self._evaluation(file)) is not None]
            if not evaluations:
                value = self._evaluation(path / "evaluation.json", fallback_progress=
                                         self._legacy_final_progress(path, metadata, status))
                if value:
                    evaluations.append(value)
            task_rows = []
            for name in config.get("tasks", []):
                episodes = [ep for row in rows for ep in row["episodes"] if ep["task"] == name]
                outcomes = [ep["outcome"] or {} for ep in episodes]
                known = [outcome for outcome in outcomes if outcome.get("known") is True]
                latest_eval = next((evaluation["tasks"][name] for evaluation in
                                    reversed(evaluations) if name in evaluation["tasks"]), None)
                task_rows.append({
                    "id": name, "label": TASK_LABELS.get(name, name),
                    "train_episodes": len(episodes),
                    "train_successes": sum(outcome.get("success") is True for outcome in known),
                    "train_failures": sum(outcome.get("failure") is True for outcome in known),
                    "train_unfinished": sum(not outcome.get("success")
                                            and not outcome.get("failure") for outcome in known),
                    "train_unknown": len(outcomes) - len(known),
                    "train_mean_reward": _mean([ep["reward_sum"] for ep in episodes]),
                    "last_eval": {"successes": latest_eval["agent_successes"],
                                  "episodes": latest_eval["agent_episodes"],
                                  "random_successes": latest_eval["random_successes"],
                                  "random_episodes": latest_eval["random_episodes"],
                                  "mean_visible_score": latest_eval["mean_visible_score"]}
                    if latest_eval else None,
                })
            alive, launch = self._process(path)
            archive_rows = self._archives(path)
            completed = status.get("rollouts", metrics[-1]["rollout"] if metrics else 0)
            decisions = live.get("decisions", status.get("decisions"))
            multipliers = config.get("task_episode_multipliers") or {}
            episode_count = sum(config.get("episodes_per_task", 0) * multipliers.get(task, 1)
                                for task in config.get("tasks", []))
            latest_archive = archive_rows[-1] if archive_rows else None
            current_rollout = int(_ROLLOUT.fullmatch(latest_archive.parent.name)[1]) \
                if latest_archive else completed
            current_archives = [file for file in archive_rows
                                if file.parent == latest_archive.parent] if latest_archive else []
            run_status = status.get("status", "unknown")
            finished = run_status in {"completed", "resource_stopped", "failed", "interrupted"}
            stage = "finished" if finished else "unknown"
            if not finished:
                if current_rollout > completed:
                    stage = "update" if episode_count and len(current_archives) == episode_count \
                        else "collect"
                elif checkpoints:
                    interval = config.get("eval_every_rollouts")
                    evaluated = evaluations[-1]["rollout"] if evaluations else None
                    stage = "evaluate" if completed and interval and completed % interval == 0 \
                        and evaluated != completed else "collect"
                elif not self._json(path / "reward_bootstrap.json"):
                    stage = "bootstrap"
            relevant_files = [path / name for name in
                              ("metadata.json", "status.json", "live.json", "training.jsonl")]
            if latest_archive:
                relevant_files.append(latest_archive)
            timestamps = [sig[0] / 1e9 for file in relevant_files
                          if (sig := self._signature(file)) is not None]
            last_updated = max(timestamps) if timestamps else None
            now = datetime.now(UTC).timestamp()
            created = _timestamp(metadata.get("created_utc"))
            elapsed = status.get("wall_seconds")
            if elapsed is None and created is not None:
                elapsed = max(0, (now if alive is True else last_updated or now) - created)
            frame, _ = self._selected_frame(path, task)
            progress = {"stage": stage, "source": "inferred", "task": None,
                        "episode": None, "episode_step": None,
                        "rollout": completed if stage == "evaluate" else current_rollout
                        if current_rollout > completed else completed + 1,
                        "episodes_completed": len(current_archives)
                        if current_rollout > completed or stage == "evaluate" else 0,
                        "episodes_per_rollout": episode_count, "updated_utc":
                        utc_stamp(last_updated) if last_updated is not None else None}
            if live:
                progress.update({key: live.get(key, progress.get(key)) for key in
                                 ("stage", "task", "episode", "episode_step", "rollout",
                                  "episodes_completed", "updated_utc")})
                progress["stage"] = live.get("phase", live.get("stage", stage))
                progress["source"] = "telemetry"
            if finished:
                progress["stage"] = "finished"
                progress["rollout"] = completed
                progress["episodes_completed"] = len(current_archives) \
                    if current_rollout == completed else None
            reward_data = self._json(path / "reward_bootstrap.json")
            heldout = reward_data.get("heldout") or {}
            reward = {"train_frames": reward_data.get("fit", {}).get("train_metrics", {})
                      .get("frames"), "validation_frames": heldout.get("frames"),
                      "score_mae": heldout.get("score_mae"),
                      "success_recall": heldout.get("success", {}).get("recall"),
                      "failure_recall": heldout.get("failure", {}).get("recall"),
                      "uncertain_fraction": heldout.get("uncertain_fraction")}
            partitions = reward_data.get("partitions")
            if isinstance(partitions, dict):
                reward = {key: reward_data.get(key) for key in
                          ("model_version", "scope", "supervision", "inputs",
                           "synthetic_glyph_coverage") if reward_data.get(key) is not None}
                reward["selector"] = config.get("reward_model_mode")
                reward["partitions"] = {}
                for name, values in partitions.items():
                    if not isinstance(values, dict):
                        continue
                    partition_metrics = {key: value for key in
                               ("frames", "episodes", "score_mae", "max_score_error",
                                "max_spurious_reward_change", "synthetic_glyph_frames",
                                "played_episode_frames")
                               if (value := _number(values.get(key))) is not None}
                    for event in ("success", "failure"):
                        event_values = values.get(event)
                        if isinstance(event_values, dict):
                            partition_metrics[event] = {key: value for key in
                                              ("positive_frames", "negative_frames", "recall",
                                               "false_positive_rate", "brier_score")
                                              if (value := _number(event_values.get(key)))
                                              is not None}
                    reward["partitions"][name] = partition_metrics
                gate = reward_data.get("gate")
                if isinstance(gate, dict):
                    reward["gate"] = {key: gate[key] for key in
                                      ("passed", "failures", "tolerances", "tolerances_source")
                                      if key in gate}
            warnings = []
            if not live and not finished:
                warnings.append("Текущая фаза выведена из файлов. Точный шаг недоступен; "
                                "изображение показывает последний завершённый эпизод.")
            if alive is False and run_status == "running":
                warnings.append("Процесс запуска завершился, но статус ещё отмечен как running.")
            if any(row["train_unknown"] for row in task_rows):
                warnings.append("Есть эпизоды с нераспознанным визуальным исходом.")
            activity = [{"timestamp": row["created_utc"], "type": "checkpoint",
                         "label": f"Снимок после цикла {row['rollout']}",
                         "detail": f"{row['decisions']} действий"} for row in checkpoints[-12:]]
            return {"run": {"id": run_id, "status": run_status,
                            "created_utc": metadata.get("created_utc"),
                            "device": config.get("device"), "seed": config.get("seed"),
                            "task_count": len(config.get("tasks", [])), "rollouts": completed,
                            "total_rollouts": config.get("total_rollouts"),
                            "decisions": decisions,
                            "budget_seconds": config.get("max_wall_seconds"),
                            "elapsed_seconds": elapsed,
                            "data_updated_utc": utc_stamp(last_updated) if last_updated else None,
                            "data_age_seconds": max(0, now - last_updated)
                            if last_updated else None,
                            "process_alive": alive}, "progress": progress, "metrics": metrics,
                    "evaluations": evaluations, "tasks": task_rows, "frame": frame,
                    "activity": activity, "checkpoints": checkpoints,
                    "reward": reward if reward_data else None, "config": config,
                    "warnings": warnings, "log_tail": self._tail(launch / "train.log")}
