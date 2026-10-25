"""Subprocess entrypoint for one frozen CPU replay; never builds an optimizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import threading
import time
from dataclasses import replace
from pathlib import Path

from carryrl.dashboard.validation_browser import (
    MAX_SECONDS,
    read_json,
    resource_error,
    resource_snapshot,
    safe_child,
    write_json,
)


def replay(directory):
    import torch

    from carryrl.dashboard.frozen_playback import load_frozen_agent, write_frozen_playback
    from carryrl.envs.atomic import AtomicEnv, atomic_level_banks, verify_visible_outcome
    from carryrl.envs.presentation import PresentationConfig
    from carryrl.training.sequence import (
        _environment_options,
        _episode_instruction,
        sequence_evaluation_protocol,
        tensor_hash,
    )

    request = read_json(directory / "request.json")
    run = directory.parent.parent
    checkpoint = safe_child(safe_child(run, "checkpoints"), request["checkpoint"])
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    agent, config, manifest = load_frozen_agent(checkpoint)
    if (manifest["sha256"] != request["checkpoint_sha256"]
            or manifest["agent_sha256"] != request["agent_sha256"]):
        raise ValueError("Checkpoint изменился после выбора")
    task, level = request["task"], request["level_index"]
    if task not in config.tasks or not 0 <= level < len(
            atomic_level_banks(task, config.level_bank_seed)["validation"]):
        raise ValueError("Уровень не принадлежит validation-банку checkpoint")
    if config.max_episode_steps > 80 or config.env_image_size > 256:
        raise ValueError("Повтор поддерживает максимум 80 команд и RGB до 256 px")
    reference = request["reference"]
    seed = reference["seed"] if reference else 1_000_000 + level
    difficulty = (reference["difficulty"] if reference else
                  config.evaluation_difficulty if config.evaluation_difficulty is not None else
                  config.curriculum_difficulty if config.curriculum_difficulty is not None else 1.0)
    visual = None
    notes = []
    if request["appearance"] == "pixel":
        visual = config.visual_config
        if visual is None or not visual.pixel_art:
            visual = PresentationConfig(pixel_art=True, style_split="validation")
            notes.append("Pixel: стандартный validation-профиль, отсутствует в конфигурации")
    cfg = replace(config, visual_config=visual, evaluation_canonical=False,
                  evaluation_pixel_art=False)
    protocol = sequence_evaluation_protocol(cfg)
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if reference and reference.get("protocol_sha256") not in (None, protocol_hash):
        raise ValueError("Протокол checkpoint отличается от опубликованной оценки")
    instruction = _episode_instruction(cfg, task, seed)
    if reference and instruction != reference["instruction"]:
        raise ValueError("Инструкция checkpoint отличается от опубликованной оценки")
    env = AtomicEnv(task, split="validation",
                    **_environment_options(cfg, difficulty=difficulty))
    try:
        agent.reset_episode(instruction=instruction)
        frame, info = env.reset(seed=seed, options={"level_index": level})
        frames, actions = [frame.copy()], []
        initial_hash = hashlib.sha256(frame.tobytes()).hexdigest()
        if reference and any(reference.get(key) != actual for key, actual in
                (("level_id", info["level_id"]), ("initial_frame_sha256", initial_hash))):
            raise ValueError("RGB/уровень не совпадает с опубликованными условиями оценки")
        if reference and reference.get("level_bank_seed") not in (None, config.level_bank_seed):
            raise ValueError("Банк уровней отличается от опубликованной оценки")
        if reference:
            for key in ("semantic_level_id", "runtime_fingerprint", "presentation"):
                if (reference.get(key) is not None or key == "presentation") and (
                        reference.get(key) != info.get(key)):
                    raise ValueError(f"Условия {key} отличаются от опубликованной оценки")
        with torch.inference_mode():
            for _ in range(cfg.max_episode_steps):
                action = agent.act(frame, deterministic=True).action
                frame, _, terminated, truncated, final_info = env.step(action)
                frames.append(frame.copy())
                actions.append(action.to_dict())
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("Игра не завершилась в пределах бюджета")
        final_hash = tensor_hash(agent)
        if final_hash != manifest["agent_sha256"]:
            raise RuntimeError("Повтор изменил веса модели")
        record = {"task": task, "policy": "frozen_deterministic", "split": "validation",
                  "seed": seed, "level_index": level, "level_id": info["level_id"],
                  "semantic_level_id": info.get("semantic_level_id"),
                  "runtime_fingerprint": info.get("runtime_fingerprint"),
                  "level_bank_seed": cfg.level_bank_seed, "instruction": instruction,
                  "difficulty": difficulty, "presentation": info.get("presentation"),
                  "steps": len(actions), "terminated": bool(terminated),
                  "truncated": bool(truncated), "initial_frame_sha256": initial_hash,
                  "quality": final_info.get("quality"), "outcome": verify_visible_outcome(frame)}
        comparison_fields = ("steps", "outcome", "terminated", "truncated")
        reference_match = (all(record[key] == reference[key] for key in comparison_fields)
                           if reference else None)
        epoch = manifest.get("epoch", manifest.get("rollout", 0))
        sidecar = write_frozen_playback(directory, record, frames, actions, epoch=epoch,
            agent_sha256=manifest["agent_sha256"], weights_revision=manifest["weights_revision"],
            checkpoint=checkpoint, manifest=manifest)
        result = read_json(sidecar)
        result.update(rollout=manifest.get("rollout"), inference_backend="torch",
            device="cpu", torch_threads=2, optimizer_constructed=False,
            appearance=request["appearance"], mode="greedy", reference_match=reference_match,
            conditions_source="published_evaluation" if reference else "checkpoint_config",
            reference_evaluation=reference.get("evaluation_file") if reference else None,
            protocol_sha256=protocol_hash, protocol=protocol,
            final_agent_sha256=final_hash, notes=notes,
            source_label=f"Frozen · {checkpoint.name} · revision {manifest['weights_revision']}",
            comparison_scope=("initial RGB/instruction and terminal metrics; "
                              "trajectory not certified"))
        write_json(sidecar, result)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    request = read_json(directory / "request.json")
    started = time.monotonic()

    def terminate(signum, _frame):
        # Archive restoration has finally blocks that remove its staging files.
        raise SystemExit(f"Повтор остановлен сигналом {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGALRM, terminate)

    def watchdog():
        while True:
            stop = False
            try:
                if os.getppid() != request["parent_pid"]:
                    stop = True
                if time.monotonic() - started > MAX_SECONDS:
                    stop = True
                if resource_error(resource_snapshot(directory, os.getpid())):
                    stop = True
            except Exception:
                stop = True
            if stop:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
                os._exit(75)
            time.sleep(.5)

    signal.alarm(MAX_SECONDS + 2)
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        result = {"status": "completed", "replay": replay(directory)}
    except (Exception, SystemExit) as error:
        result = {"status": "failed", "error": f"{type(error).__name__}: {str(error)[:500]}"}
    write_json(directory / "result.json", result)


if __name__ == "__main__":
    main()
