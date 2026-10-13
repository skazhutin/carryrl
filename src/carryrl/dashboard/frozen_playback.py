"""Bounded, hash-linked closed-loop RGB playback of actual frozen learned policies."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from carryrl.agent.actions import CODEC_VERSION
from carryrl.agent.model import GeneralistAgent
from carryrl.envs.atomic import AtomicEnv, verify_visible_outcome
from carryrl.training.run import atomic_json
from carryrl.training.sequence import (
    _environment_options,
    _episode_instruction,
    _language_metadata,
    _language_resource_hashes,
    tensor_hash,
)
from carryrl.training.sequence_config import SequenceRunConfig

SOURCE = "actual_frozen_learned_policy_closed_loop_RGB_playback"
MAX_FRAMES = 81
MAX_GIF_BYTES = 8 * 1024**2


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_frozen_playback(run: Path, record: dict, frames: list, actions: list, *,
                          epoch: int, agent_sha256: str, weights_revision: int,
                          checkpoint: Path | None = None, manifest: dict | None = None,
                          evaluation_reference: Path | None = None) -> Path:
    """Persist aligned actual RGB/actions, never replacing a previous capture.

    During training the policy hash/revision identifies the evaluated weights;
    only a saved matching checkpoint supplies checkpoint linkage.
    """
    if record.get("policy") != "frozen_deterministic":
        raise ValueError("playback requires the actual deterministic learned policy")
    if not 2 <= len(frames) <= MAX_FRAMES or len(actions) + 1 != len(frames):
        raise ValueError("playback requires initial RGB plus one RGB per action, at most 81")
    shape = frames[0].shape
    if (len(shape) != 3 or shape[-1] != 3 or max(shape[:2]) > 256
            or any(frame.dtype != np.uint8 or frame.shape != shape for frame in frames)):
        raise ValueError("playback requires consistent bounded uint8 RGB observations")
    frame_hashes = [hashlib.sha256(rgb.tobytes()).hexdigest() for rgb in frames]
    if record["steps"] != len(actions) or record["initial_frame_sha256"] != frame_hashes[0]:
        raise ValueError("evaluation and captured observations/actions do not align")
    if verify_visible_outcome(frames[-1]) != record["outcome"]:
        raise ValueError("playback outcome does not match the final actual RGB")
    if checkpoint is not None:
        if manifest is None or manifest.get("agent_sha256") != agent_sha256 \
                or manifest.get("weights_revision") != weights_revision:
            raise ValueError("playback policy does not match its checkpoint manifest")
    level = record.get("level_index")
    level_label = f"level{level}" if level is not None else "seeded"
    output = run / "playbacks"
    output.mkdir(parents=True, exist_ok=True)
    gif_path = output / (
        f"epoch_{epoch:04d}_{record['task']}_{record['split']}_{level_label}_greedy.gif")
    json_path = gif_path.with_suffix(".json")
    sprite_path = gif_path.with_suffix(".frames.png")
    if gif_path.exists() or json_path.exists() or sprite_path.exists():
        raise FileExistsError(f"playback already exists: {gif_path}")
    images = [Image.fromarray(rgb).resize((384, 384), Image.Resampling.NEAREST)
              for rgb in frames]
    encoded = io.BytesIO()
    images[0].save(encoded, format="GIF", save_all=True, append_images=images[1:],
                   duration=[100] * (len(images) - 1) + [1200], loop=0,
                   optimize=False, disposal=2)
    gif = encoded.getvalue()
    if len(gif) > MAX_GIF_BYTES:
        raise ValueError("playback GIF exceeds 8 MiB")
    with Image.open(io.BytesIO(gif)) as image:
        encoded_frame_count = image.n_frames
    transitions = [{"step": index, "observation_sha256": frame_hashes[index],
                    "action": action, "next_observation_sha256": frame_hashes[index + 1]}
                   for index, action in enumerate(actions)]
    sprite = Image.new("RGB", (shape[1] * 8, shape[0] * ((len(frames) + 7) // 8)))
    for index, frame in enumerate(frames):
        sprite.paste(Image.fromarray(frame), ((index % 8) * shape[1], (index // 8) * shape[0]))
    sprite_encoded = io.BytesIO()
    sprite.save(sprite_encoded, format="PNG")
    sprite_bytes = sprite_encoded.getvalue()
    if len(sprite_bytes) > MAX_GIF_BYTES:
        raise ValueError("playback sprite exceeds 8 MiB")
    sidecar = {
        **record, "created_utc": datetime.now(UTC).isoformat(), "source": SOURCE,
        "epoch": epoch, "agent_sha256": agent_sha256, "weights_revision": weights_revision,
        "source_label": f"Frozen learned policy · epoch {epoch} · revision {weights_revision}",
        "playback_action_source": "GeneralistAgent.act(current_RGB, deterministic=True)",
        "teacher_actions_used": False, "training_performed": False,
        "actor_input": "current_RGB, own_previous_physical_actions, episode_prose_instruction",
        "action_count": len(actions), "observation_frame_count": len(frames),
        "frame_alignment": "frame t precedes action t; last RGB follows the last action",
        "frame_sha256": frame_hashes, "actions": transitions,
        "gif": str(gif_path.resolve()), "gif_sha256": hashlib.sha256(gif).hexdigest(),
        "gif_encoded_frame_count": encoded_frame_count, "gif_display_size": [384, 384],
        "gif_tick_duration_ms": 100, "gif_terminal_hold_ms": 1200,
        "sprite_filename": sprite_path.name,
        "sprite_sha256": hashlib.sha256(sprite_bytes).hexdigest(), "sprite_columns": 8,
        "frame_width": shape[1], "frame_height": shape[0], "frame_count": len(frames),
        "frame_duration_ms": 100, "terminal_hold_ms": 1200,
        "observation_size": list(shape),
        "outcome_source": "independent_visible_RGB_HUD_verifier_on_final_actual_observation",
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "checkpoint_sha256": manifest["sha256"] if manifest else None,
        "current_checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "checkpoint_linkage": "matching_manifest" if checkpoint else "evaluated_policy_hash",
        "checkpoint_training_backend": manifest.get("training_backend") if manifest else None,
        "evaluation_reference": str(evaluation_reference.resolve())
        if evaluation_reference else None,
        "matches_frozen_evaluation_record": evaluation_reference is not None,
        "policy_weights_unchanged": True,
    }
    with gif_path.open("xb") as stream:
        stream.write(gif)
    with sprite_path.open("xb") as stream:
        stream.write(sprite_bytes)
    atomic_json(json_path, sidecar)
    return json_path


def link_matching_playbacks(run: Path, checkpoint: Path, manifest: dict):
    """Link only captures whose actual evaluated tensors equal this saved checkpoint."""
    for path in sorted((run / "playbacks").glob("*.json")):
        row = json.loads(path.read_text())
        if (row.get("source") == SOURCE and row.get("agent_sha256") == manifest["agent_sha256"]
                and row.get("weights_revision") == manifest["weights_revision"]
                and row.get("checkpoint") is None):
            row.update(checkpoint=str(checkpoint.resolve()),
                       current_checkpoint=str(checkpoint.resolve()),
                       checkpoint_sha256=manifest["sha256"], checkpoint_linkage="matching_manifest",
                       checkpoint_training_backend=manifest.get("training_backend"))
            atomic_json(path, row)


def load_frozen_agent(checkpoint: Path):
    """Load a strict offline graph and resources without constructing an optimizer."""
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if sha256_file(checkpoint / "state.pt") != manifest["sha256"]:
        raise ValueError("checkpoint archive SHA256 mismatch")
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True, mmap=True)
    if state.get("format") != "carryrl.sequence.v1" or state.get("codec_version") != CODEC_VERSION:
        raise ValueError("unsupported sequence checkpoint format or codec")
    config = SequenceRunConfig.from_dict(state["config"])
    if (state.get("weights_revision") != manifest.get("weights_revision")
            or manifest.get("codec_version") != CODEC_VERSION
            or state.get("training_backend", "torch") != config.training_backend
            or manifest.get("training_backend", "torch") != config.training_backend):
        raise ValueError("checkpoint revision, codec, or training backend metadata mismatch")
    for name, expected in _language_metadata(config).items():
        if state.get(name) != expected or manifest.get(name) != expected:
            raise ValueError("checkpoint language identity mismatch")
    language = None
    if _language_metadata(config):
        from carryrl.agent.language import MultilingualInstructionEncoder

        if _language_resource_hashes(checkpoint / "language") != manifest.get(
                "language_resources_sha256"):
            raise ValueError("checkpoint language resource SHA256 mismatch")
        language = MultilingualInstructionEncoder.from_resources(
            checkpoint / "language", max_tokens=config.agent.instruction_max_tokens)
    agent = GeneralistAgent(replace(config.agent, pretrained=False), language=language)
    agent.decoder.initialize_motor_prior(config.motor_initialization)
    agent.load_state_dict(state["agent_state"], strict=True)
    agent.config = config.agent
    agent.eval().requires_grad_(False)
    if tensor_hash(agent) != manifest["agent_sha256"]:
        raise ValueError("checkpoint agent tensor SHA256 mismatch")
    return agent, config, manifest


def export_run(run: Path, tasks: tuple[str, ...] | None = None) -> list[Path]:
    run = run.resolve()
    checkpoint = Path(json.loads((run / "latest.json").read_text())["checkpoint"])
    torch.set_num_threads(2)
    agent, config, manifest = load_frozen_agent(checkpoint)
    evaluations = []
    for path in sorted((run / "evaluations").glob("epoch_*.json")):
        evaluation = json.loads(path.read_text())
        if evaluation.get("agent_sha256") == manifest["agent_sha256"]:
            evaluations.append((int(evaluation["epoch"]), path, evaluation))
    if not evaluations:
        raise ValueError("no frozen evaluation matches the saved agent")
    epoch, reference_path, reference = max(evaluations, key=lambda row: row[0])
    tasks = tasks or config.tasks
    if any(task not in config.tasks for task in tasks):
        raise ValueError("requested task is outside the checkpoint task suite")
    paths = []
    for task in tasks:
        expected = next((row for row in reference["records"]
                         if row["task"] == task and row["policy"] == "frozen_deterministic"
                         and row.get("level_index") == 0 and row["seed"] == 1_000_000), None)
        if expected is None:
            raise ValueError(f"no level-zero frozen reference for {task}")
        destination = run / "playbacks" / f"epoch_{epoch:04d}_{task}_validation_level0_greedy.json"
        if destination.exists():
            existing = json.loads(destination.read_text())
            gif_path = destination.with_suffix(".gif")
            if (existing.get("agent_sha256") != manifest["agent_sha256"]
                    or existing.get("checkpoint_sha256") != manifest["sha256"]
                    or sha256_file(gif_path) != existing.get("gif_sha256")):
                raise ValueError(f"existing playback verification failed: {destination}")
            paths.append(destination)
            continue
        instruction = _episode_instruction(config, task, expected["seed"])
        env = AtomicEnv(task, split="validation", **_environment_options(
            config, difficulty=expected["difficulty"]))
        try:
            agent.reset_episode(instruction=instruction)
            frame, info = env.reset(seed=expected["seed"], options={"level_index": 0})
            frames, actions = [frame.copy()], []
            for _ in range(config.max_episode_steps):
                action = agent.act(frame, deterministic=True).action
                frame, _, terminated, truncated, _ = env.step(action)
                frames.append(frame.copy())
                actions.append(action.to_dict())
                if terminated or truncated:
                    break
            actual = {"task": task, "policy": "frozen_deterministic", "split": "validation",
                      "seed": expected["seed"], "level_index": 0, "level_id": info["level_id"],
                      "instruction": instruction, "difficulty": expected["difficulty"],
                      "presentation": info.get("presentation"), "steps": len(actions),
                      "terminated": bool(terminated), "truncated": bool(truncated),
                      "initial_frame_sha256": hashlib.sha256(frames[0].tobytes()).hexdigest(),
                      "outcome": verify_visible_outcome(frame)}
            for field in ("seed", "instruction", "level_id", "initial_frame_sha256",
                          "steps", "outcome", "terminated", "truncated"):
                if actual[field] != expected[field]:
                    raise ValueError(
                        f"actual frozen playback differs from reference: {task}/{field}")
            if tensor_hash(agent) != manifest["agent_sha256"]:
                raise RuntimeError("frozen replay mutated policy tensors")
            path = write_frozen_playback(
                run, actual, frames, actions, epoch=epoch, agent_sha256=manifest["agent_sha256"],
                weights_revision=manifest["weights_revision"], checkpoint=checkpoint,
                manifest=manifest, evaluation_reference=reference_path)
            row = json.loads(path.read_text())
            row.update(optimizer_constructed=False, device="cpu", torch_threads=2,
                       final_agent_sha256=manifest["agent_sha256"])
            atomic_json(path, row)
            paths.append(path)
            print(json.dumps({"task": task, "epoch": epoch, "steps": len(actions),
                              "outcome": actual["outcome"], "sidecar": str(path)}), flush=True)
        finally:
            env.close()
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    args = parser.parse_args()
    paths = export_run(args.run, None if args.tasks == ["all"] else tuple(args.tasks))
    print(json.dumps({"playbacks": len(paths), "training_performed": False}))


if __name__ == "__main__":
    main()
