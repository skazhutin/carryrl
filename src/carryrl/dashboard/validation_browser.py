"""Bounded validation inventory and explicitly requested, isolated frozen replays."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
MAX_JSON = 16 * 1024**2
MAX_FILES = 128
MAX_REPLAYS = 64
MAX_TOTAL_JSON = 64 * 1024**2
MAX_SECONDS = 180
MIN_AVAILABLE = 1536 * 1024**2
MIN_DISK = 10 * 1024**3
MAX_PROCESS_RSS = 3 * 1024**3
MODES = {"frozen_deterministic": "greedy", "frozen_stochastic": "stochastic",
         "generic_physical_random": "random"}


def safe_child(parent, name):
    if not isinstance(name, str) or not NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError("Недопустимый идентификатор")
    target = parent / name
    if target.is_symlink() or target.resolve().parent != parent.resolve():
        raise ValueError("Ссылки за пределы каталога запрещены")
    return target


def read_json(path, limit=MAX_JSON):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"Недоступный или слишком большой JSON: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("JSON превышает лимит")
    return json.loads(raw)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def resource_snapshot(root, pid=None):
    # Use the same macOS free+inactive+speculative accounting as the training
    # guard; the dashboard's base environment has no optional psutil dependency.
    if sys.platform == "darwin":
        vm = subprocess.run(["vm_stat"], check=True, capture_output=True, text=True,
                            timeout=2).stdout
        page = re.search(r"page size of (\d+) bytes", vm)
        counts = dict(re.findall(r"^([^:\n]+):\s+(\d+)\.", vm, re.MULTILINE))
        fields = ("Pages free", "Pages inactive", "Pages speculative")
        if page is None or any(key not in counts for key in fields):
            raise ValueError("Не удалось прочитать доступную память vm_stat")
        available = int(page.group(1)) * sum(int(counts[key]) for key in fields)
    elif sys.platform.startswith("linux"):
        memory = Path("/proc/meminfo").read_text()
        value = re.search(r"^MemAvailable:\s+(\d+)\s+kB", memory, re.MULTILINE)
        if value is None:
            raise ValueError("Не удалось прочитать MemAvailable")
        available = int(value.group(1)) * 1024
    else:
        raise ValueError("Монитор ресурсов поддерживает macOS и Linux")
    rss = 0
    if pid is not None:
        output = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], check=True,
                                capture_output=True, text=True, timeout=2).stdout
        rows = [tuple(map(int, fields)) for line in output.splitlines()
                if len(fields := line.split()) == 3]
        owned = {pid}
        while True:
            children = {child for child, parent, _ in rows if parent in owned}
            if children <= owned:
                break
            owned.update(children)
        rss = sum(size * 1024 for child, _, size in rows if child in owned)
    return {"available_bytes": available,
            "free_disk_bytes": shutil.disk_usage(root).free, "process_tree_rss_bytes": rss}


def resource_error(snapshot):
    if snapshot["available_bytes"] < MIN_AVAILABLE:
        return "Недостаточно свободной RAM для повтора (нужно 1.5 GiB)"
    if snapshot["free_disk_bytes"] < MIN_DISK:
        return "Недостаточно места для повтора (нужно 10 GiB)"
    if snapshot["process_tree_rss_bytes"] > MAX_PROCESS_RSS:
        return "Повтор превысил лимит RAM 3 GiB"
    return None


class ReplayBusy(RuntimeError):
    pass


class ReplayResources(RuntimeError):
    pass


class ValidationBrowser:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = None
        self.cache = {}

    def run_path(self, run_id):
        run = safe_child(self.root, run_id)
        if not run.is_dir():
            raise ValueError("Неизвестный запуск")
        return run

    def _with_active(self, catalog):
        with self.lock:
            active = self.jobs.get(self.active)
            active = dict(active) if active and active["request"]["run"] == catalog[
                "run_id"] else None
        return {**catalog, "active_job": active}

    def catalog(self, run_id):
        from carryrl.envs.atomic import _identity, atomic_level_banks, atomic_semantic_level_id
        from carryrl.envs.atomic_suite import TASKS

        run = self.run_path(run_id)
        config_path = run / "config.json"
        config = read_json(config_path, 1024**2)
        checkpoint_root = safe_child(run, "checkpoints")
        paths = []
        for folder in ("evaluations", "evaluations_pixel"):
            directory = safe_child(run, folder)
            paths.extend(sorted(directory.glob("*.json"))[-MAX_FILES:])
        manifests = sorted(checkpoint_root.glob("*/manifest.json"))[-MAX_FILES:]
        replay_root = safe_child(run, "dashboard_replays")
        replay_directories = [path for path in replay_root.glob("*")
                              if JOB_ID.fullmatch(path.name) and not path.is_symlink()
                              and path.is_dir()]
        replay_directories.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
        replay_statuses = [path / "status.json"
                           for path in replay_directories[:MAX_REPLAYS]
                           if (path / "status.json").is_file()]
        evaluation_paths = paths
        paths = [config_path, *manifests, *evaluation_paths, *replay_statuses]
        fingerprint = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns)
                            for path in paths)
        with self.lock:
            cached = self.cache.get(run_id)
            if cached and cached[0] == fingerprint:
                return self._with_active(cached[1])
        if sum(row[1] for row in fingerprint) > MAX_TOTAL_JSON:
            raise ValueError("Каталог validation превышает лимит 64 MiB")
        checkpoints, warnings = [], []
        for path in manifests:
            try:
                checkpoint = safe_child(checkpoint_root, path.parent.name)
                manifest = read_json(path, 1024**2)
                available = any((checkpoint / name).is_file()
                                and not (checkpoint / name).is_symlink() for name in
                                ("state.pt", "state.pt.gz", "state.pt.chunks.json"))
                checkpoints.append({"id": checkpoint.name, "rollout": manifest.get("rollout"),
                    "epoch": manifest.get("epoch"), "weights_revision": manifest.get(
                        "weights_revision"), "agent_sha256": manifest.get("agent_sha256"),
                    "checkpoint_sha256": manifest.get("sha256"),
                    "training_backend": manifest.get("training_backend"),
                    "available": available})
            except (OSError, ValueError) as error:
                warnings.append(str(error))
        records = []
        fields = ("task", "level_index", "seed", "steps", "outcome", "quality", "level_id",
                  "semantic_level_id", "instruction", "difficulty", "presentation",
                  "initial_frame_sha256", "terminated", "truncated", "action_seed",
                  "level_bank_seed", "runtime_fingerprint")
        for path in evaluation_paths:
            try:
                evaluation = read_json(path)
                if evaluation.get("split") != "validation":
                    continue
                matches = [row["id"] for row in checkpoints
                           if row["agent_sha256"] and row["agent_sha256"] == evaluation.get(
                               "agent_sha256")]
                for row in evaluation.get("records", []):
                    if row.get("policy") not in MODES or row.get("level_index") is None:
                        continue
                    presentation = row.get("presentation")
                    appearance = ("pixel" if presentation and presentation.get("pixel_art")
                                  else "varied" if presentation else "canonical")
                    if appearance == "varied":
                        warning = f"{path.name}: varied-оформление не поддерживает ручной повтор"
                        if warning not in warnings:
                            warnings.append(warning)
                    for checkpoint_id in matches:
                        if len(records) >= 50000:
                            raise ValueError("Слишком много validation-записей")
                        records.append({**{key: row.get(key) for key in fields},
                            "checkpoint_id": checkpoint_id,
                            "appearance": appearance, "mode": MODES[row["policy"]],
                            "protocol_sha256": evaluation.get("protocol_sha256"),
                            "evaluation_file": str(path.relative_to(run))})
            except (OSError, ValueError) as error:
                warnings.append(f"{path.name}: {error}")
        replays = []
        checkpoint_by_id = {row["id"]: row for row in checkpoints}
        for path in replay_statuses:
            try:
                saved = read_json(path, 1024**2)
                request, replay = saved.get("request", {}), saved.get("replay", {})
                checkpoint = checkpoint_by_id.get(request.get("checkpoint"))
                if (saved.get("status") != "completed" or saved.get("id") != path.parent.name
                        or request.get("run") != run_id or checkpoint is None
                        or replay.get("split") != "validation"
                        or replay.get("policy") != "frozen_deterministic"
                        or replay.get("agent_sha256") != checkpoint["agent_sha256"]
                        or replay.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]
                        or replay.get("source") !=
                        "actual_frozen_learned_policy_closed_loop_RGB_playback"
                        or replay.get("training_performed") is not False
                        or replay.get("mode") != "greedy"
                        or replay.get("appearance") not in {"canonical", "pixel"}
                        or any(replay.get(key) != request.get(key)
                               for key in ("task", "level_index", "appearance", "mode"))):
                    continue
                media_root = safe_child(path.parent, "playbacks")
                sprite = safe_child(media_root, replay["sprite_filename"])
                gif = safe_child(media_root, Path(replay["gif"]).name)
                if (not sprite.is_file() or not gif.is_file()
                        or sprite.stat().st_size > 8 * 1024**2
                        or gif.stat().st_size > 8 * 1024**2):
                    continue
                base = f"/replay-media/{run_id}/{saved['id']}/"
                replays.append({**replay, "id": saved["id"], "manual": True,
                    "checkpoint_id": checkpoint["id"], "request": request,
                    "sprite_url": base + sprite.name, "gif_url": base + gif.name,
                    "url": base + gif.name})
                records.append({**{key: replay.get(key) for key in fields},
                    "checkpoint_id": checkpoint["id"], "appearance": replay["appearance"],
                    "mode": "greedy", "manual": True, "replay_id": saved["id"],
                    "protocol_sha256": replay.get("protocol_sha256"), "evaluation_file": None})
            except (KeyError, TypeError, OSError, ValueError) as error:
                warnings.append(f"Повтор {path.parent.name}: {error}")
        if len(replay_directories) > MAX_REPLAYS:
            warnings.append("Показаны только последние 64 ручных повтора")
        bank_seed = config.get("level_bank_seed", 0)
        tasks = []
        for task in config.get("tasks", []):
            if task not in TASKS:
                continue
            levels = atomic_level_banks(task, bank_seed)["validation"]
            tasks.append({"task": task, "levels": [
                {"level_index": index, "level_id": _identity(level),
                 "semantic_level_id": atomic_semantic_level_id(task, level),
                 "descriptor": level} for index, level in enumerate(levels)]})
        result = {"run_id": run_id, "split": "validation", "checkpoints": checkpoints,
                  "tasks": tasks, "records": records, "supported_modes": ["greedy"],
                  "level_bank_seed": bank_seed, "warnings": warnings, "replays": replays,
                  "replay_history_limit": MAX_REPLAYS}
        with self.lock:
            self.cache = {run_id: (fingerprint, result)}
        return self._with_active(result)

    def start(self, request):
        if not isinstance(request, dict) or set(request) != {
                "run", "checkpoint", "task", "level_index", "appearance", "mode"}:
            raise ValueError("Нужны run, checkpoint, task, level_index, appearance, mode")
        if request["mode"] != "greedy" or request["appearance"] not in {"canonical", "pixel"}:
            raise ValueError("Поддерживается greedy, оформление canonical или pixel")
        if type(request["level_index"]) is not int:
            raise ValueError("level_index должен быть целым числом")
        catalog = self.catalog(request["run"])
        checkpoint = next((row for row in catalog["checkpoints"]
                           if row["id"] == request["checkpoint"] and row["available"]), None)
        task = next((row for row in catalog["tasks"] if row["task"] == request["task"]), None)
        if checkpoint is None or task is None or not 0 <= request["level_index"] < len(
                task["levels"]):
            raise ValueError("Неизвестный checkpoint, задача или validation-уровень")
        run = self.run_path(request["run"])
        with self.lock:
            if self.active is not None:
                raise ReplayBusy("Другой повтор уже выполняется")
            error = resource_error(resource_snapshot(run))
            if error:
                raise ReplayResources(error)
            output = safe_child(run, "dashboard_replays")
            output.mkdir(exist_ok=True)
            job_id = uuid.uuid4().hex
            directory = output / job_id
            directory.mkdir()
            reference = next((row for row in catalog["records"]
                if all(row[key] == request[key] for key in
                       ("task", "level_index", "appearance", "mode"))
                and row["checkpoint_id"] == request["checkpoint"] and not row.get("manual")), None)
            payload = {**request, "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                       "agent_sha256": checkpoint["agent_sha256"], "reference": reference,
                       "parent_pid": os.getpid()}
            write_json(directory / "request.json", payload)
            job = {"id": job_id, "status": "running", "request": dict(request)}
            self.jobs[job_id] = job
            self.active = job_id
            try:
                log = (directory / "worker.log").open("xb")
                with log:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "carryrl.dashboard.validation_replay_job",
                         "--directory", str(directory)], stdin=subprocess.DEVNULL,
                        stdout=log, stderr=log, start_new_session=True,
                        env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                             "TOKENIZERS_PARALLELISM": "false"})
            except Exception:
                self.active = None
                job.update(status="failed", error="Не удалось запустить процесс повтора")
                raise
            threading.Thread(target=self._watch, args=(process, directory, job_id),
                             daemon=True).start()
            return dict(job)

    def _watch(self, process, directory, job_id):
        started = time.monotonic()
        failure = None
        metrics = {}
        while process.poll() is None:
            try:
                metrics = resource_snapshot(directory, process.pid)
                failure = resource_error(metrics)
                if time.monotonic() - started > MAX_SECONDS:
                    failure = "Повтор превысил лимит 180 секунд"
            except Exception as error:
                failure = f"Не удалось проверить ресурсы: {error}"
            if failure:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=3)
                break
            time.sleep(.5)
        try:
            result = read_json(directory / "result.json") if not failure else {}
            if process.returncode != 0 or result.get("status") != "completed":
                failure = failure or result.get("error") or "Процесс повтора завершился с ошибкой"
            replay = result.get("replay")
            if not failure and isinstance(replay, dict):
                base = f"/replay-media/{self.jobs[job_id]['request']['run']}/{job_id}/"
                replay["sprite_url"] = base + replay["sprite_filename"]
                replay["gif_url"] = base + Path(replay["gif"]).name
                replay["url"] = replay["gif_url"]
        except Exception as error:
            failure = failure or f"Не удалось прочитать результат: {error}"
            replay = None
        with self.lock:
            self.jobs[job_id].update(status="failed" if failure else "completed",
                                    resources=metrics)
            if failure:
                self.jobs[job_id]["error"] = failure
            else:
                self.jobs[job_id]["replay"] = replay
            self.active = None
            try:
                write_json(directory / "status.json", self.jobs[job_id])
            except OSError as error:
                self.jobs[job_id].update(status="failed", error=f"Ошибка записи статуса: {error}")
            while len(self.jobs) > 32:
                del self.jobs[next(iter(self.jobs))]

    def status(self, job_id):
        with self.lock:
            return dict(self.jobs[job_id]) if job_id in self.jobs else None

    def media(self, run_id, job_id, filename):
        if not JOB_ID.fullmatch(job_id):
            return None
        run = self.run_path(run_id)
        parent = safe_child(safe_child(run, "dashboard_replays"), job_id)
        path = safe_child(safe_child(parent, "playbacks"), filename)
        if path.suffix not in {".png", ".gif"} or not path.is_file():
            return None
        if path.stat().st_size > 8 * 1024**2:
            return None
        with path.open("rb") as stream:
            content = stream.read(8 * 1024**2 + 1)
        return content if len(content) <= 8 * 1024**2 else None
