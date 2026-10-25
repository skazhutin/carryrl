"""Validation-only input interventions with actual environment command histories."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv, atomic_level_banks, atomic_semantic_level_id
from carryrl.evaluation.input_sensitivity import (
    KINDS,
    MAX_BUNDLE_BYTES,
    SENSITIVITY_VERSION,
    SensitivityPair,
    _frames_hash,
    _json_hash,
    load_sensitivity_bundle,
    prepare_mouse_position_bundle,
)

PREPARATION_VERSION = "carryrl.cycle_input_pairs.v1"


def _memory_index(level_bank_seed):
    """Apply the paired-memory diagnostic's both-cue TRAIN exclusion rule."""
    banks = atomic_level_banks("memory", level_bank_seed)
    train = {atomic_semantic_level_id("memory", level) for level in banks["train"]}
    eligible = [
        index
        for index, level in enumerate(banks["validation"])
        if all(
            atomic_semantic_level_id("memory", {**level, "cue": cue}) not in train for cue in (0, 1)
        )
    ]
    if not eligible:
        raise ValueError("no validation memory geometry has two train-disjoint cue alternatives")
    return eligible[0]


def _record_memory(*, index, cue, history_direction, probe_step, image_size, seed, level_bank_seed):
    env = AtomicEnv(
        "memory",
        split="validation",
        image_size=image_size,
        max_steps=80,
        level_bank_seed=level_bank_seed,
        difficulty=1.0,
    )
    try:
        _, origin = env.reset(seed=seed, options={"level_index": index})
        env._level = copy.deepcopy(env._level)
        env._level["cue"] = cue
        env._frame = env._draw()
        unlock = env._level["unlock"]
        if probe_step is None:
            probe_step = unlock
        if not 64 <= probe_step < 79 or (history_direction and probe_step >= unlock):
            raise ValueError("memory probe must cross segment64; history commands remain in Wait")
        frames, executed = [], []
        position = env._position_x
        for step in range(probe_step + 1):
            frames.append(env.render())
            # These are real env.step inputs, after the corresponding RGB. The
            # recurrent evaluator uses only commands preceding its probe action.
            action = (
                PhysicalAction(keys=(history_direction,))
                if (history_direction and env._level["cue_ticks"] <= step < probe_step)
                else PhysicalAction()
            )
            _, _, terminated, truncated, _ = env.step(action)
            executed.append(action)
            if terminated or truncated or env._position_x != position:
                raise RuntimeError("memory diagnostic prefix unexpectedly moved or terminated")
        return (
            np.stack(frames),
            tuple(executed),
            {
                "bank_level_id": origin["level_id"],
                "level_index": index,
                "runtime_semantic_level_id": atomic_semantic_level_id("memory", env._level),
                "cue": cue,
                "unlock_ticks": unlock,
                "cue_ticks": env._level["cue_ticks"],
                "probe_phase": "Wait" if probe_step < unlock else "Choose",
                "actual_env_step_calls": len(executed),
                "copied_level_intervention": {"cue": cue},
            },
        )
    finally:
        env.close()


def _source_metadata(pair, *, seed, level_bank_seed, origins):
    """Hashes cover exactly the RGB, explicit text and actually executed commands."""
    return {
        **pair.provenance,
        "seed": seed,
        "level_bank_seed": level_bank_seed,
        "split": "validation",
        "scene_origins": origins,
        "original_frames_sha256": _frames_hash(pair.original_frames),
        "perturbed_frames_sha256": _frames_hash(pair.perturbed_frames),
        "original_executed_sha256": _json_hash([a.to_dict() for a in pair.original_executed]),
        "perturbed_executed_sha256": _json_hash([a.to_dict() for a in pair.perturbed_executed]),
        "original_instruction_sha256": (
            hashlib.sha256(pair.original_instruction.encode()).hexdigest()
            if pair.original_instruction is not None
            else None
        ),
        "perturbed_instruction_sha256": (
            hashlib.sha256(pair.perturbed_instruction.encode()).hexdigest()
            if pair.perturbed_instruction is not None
            else None
        ),
        "policy_inputs": ["RGB", "explicit_instruction", "actually_executed_commands"],
        "candidate_actions_are_likelihood_labels": True,
        "interpretation": "Diagnostic input intervention; no win-quality claim.",
    }


def _memory_pair(kind, *, index, image_size, seed, level_bank_seed, instruction):
    cue_change = kind == "memory_cue"
    sides = [
        _record_memory(
            index=index,
            cue=side if cue_change else 0,
            history_direction=None if cue_change else ("left" if side == 0 else "right"),
            probe_step=None if cue_change else 64,
            image_size=image_size,
            seed=seed,
            level_bank_seed=level_bank_seed,
        )
        for side in (0, 1)
    ]
    a, b = sides
    pair = SensitivityPair(
        f"memory-{kind}-level{index}",
        "memory",
        kind,
        a[0],
        b[0],
        instruction,
        instruction,
        a[1],
        b[1],
        (PhysicalAction(keys=("left",)), PhysicalAction(), PhysicalAction(keys=("right",))),
        len(a[0]) - 1,
        {
            "split": "validation",
            "original_executed_source": "AtomicEnv.step original memory prefix",
            "perturbed_executed_source": "AtomicEnv.step perturbed memory prefix",
            "controlled_change": "earlier visible cue/target"
            if cue_change
            else "actual blocked Wait left/right commands",
            "segment_boundary": 64,
            "selection": "first validation index with both cues train-disjoint",
        },
    )
    return replace(
        pair,
        provenance=_source_metadata(
            pair, seed=seed, level_bank_seed=level_bank_seed, origins=[a[2], b[2]]
        ),
    ).validate()


def _text_pair(*, image_size, seed, level_bank_seed, instruction):
    env = AtomicEnv(
        "navigation",
        split="validation",
        image_size=image_size,
        max_steps=80,
        level_bank_seed=level_bank_seed,
        difficulty=1.0,
    )
    try:
        frame, origin = env.reset(seed=seed, options={"level_index": 0})
        command = PhysicalAction()
        env.step(command)
        frames = frame[None]
        pair = SensitivityPair(
            "navigation-counterfactual-text",
            "navigation",
            "text_control",
            frames,
            frames.copy(),
            instruction,
            "Diagnostic counterfactual controls: D moves left and A moves right. "
            "All remaining input is unsupported. This changes the stated controls only.",
            (command,),
            (command,),
            (PhysicalAction(keys=("a",)), PhysicalAction(keys=("d",))),
            0,
            {
                "split": "validation",
                "original_executed_source": "AtomicEnv.step navigation noop",
                "perturbed_executed_source": "same actual navigation noop; text-only intervention",
                "controlled_change": "counterfactual textual controls; environment unchanged",
                "diagnostic_intervention": True,
                "mechanics_match_counterfactual_text": False,
            },
        )
        return replace(
            pair,
            provenance=_source_metadata(
                pair,
                seed=seed,
                level_bank_seed=level_bank_seed,
                origins=[
                    {
                        "bank_level_id": origin["level_id"],
                        "level_index": 0,
                        "actual_env_step_calls": 1,
                    }
                ],
            ),
        ).validate()
    finally:
        env.close()


def prepare_cycle_input_pairs(
    directory,
    *,
    instructions="configs/atomic_instructions.json",
    image_size=96,
    seed=20261020,
    level_bank_seed=20261020,
):
    """Write four bounded validation probes; no policy load, fitting or test access."""
    if image_size != 96:
        raise ValueError("the cycle input diagnostic uses image_size=96")
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"input bundle already exists: {directory}")
    instructions = Path(instructions)
    texts = json.loads(instructions.read_text())
    index = _memory_index(level_bank_seed)
    # Reuse the existing position intervention rather than creating a second
    # rendering protocol. Its temporary preparation artifacts are disposable.
    with TemporaryDirectory(prefix="carryrl-input-position-") as scratch:
        path = prepare_mouse_position_bundle(
            Path(scratch) / "position",
            image_size=image_size,
            seed=seed,
            level_bank_seed=level_bank_seed,
        )
        mouse = load_sensitivity_bundle(path)[0][0]
    mouse = replace(
        mouse,
        provenance=_source_metadata(
            mouse,
            seed=seed,
            level_bank_seed=level_bank_seed,
            origins=mouse.provenance["scene_origins"],
        ),
    )
    pairs = [
        mouse,
        _memory_pair(
            "memory_cue",
            index=index,
            image_size=image_size,
            seed=seed,
            level_bank_seed=level_bank_seed,
            instruction=texts["memory"][1],
        ),
        _memory_pair(
            "executed_history",
            index=index,
            image_size=image_size,
            seed=seed,
            level_bank_seed=level_bank_seed,
            instruction=texts["memory"][1],
        ),
        _text_pair(
            image_size=image_size,
            seed=seed,
            level_bank_seed=level_bank_seed,
            instruction=texts["navigation"][1],
        ),
    ]
    if {pair.kind for pair in pairs} != set(KINDS):
        raise RuntimeError("cycle input diagnostics must cover all four supported kinds")
    retained = sum(pair.original_frames.nbytes + pair.perturbed_frames.nbytes for pair in pairs)
    if retained > MAX_BUNDLE_BYTES:
        raise ValueError("cycle input diagnostic exceeds RGB memory budget")
    directory.mkdir(parents=True, exist_ok=False)
    arrays, cases = {}, []
    for number, pair in enumerate(pairs):
        pair.validate()
        row = {
            "case_id": pair.case_id,
            "task": pair.task,
            "kind": pair.kind,
            "probe_step": pair.probe_step,
            "provenance": pair.provenance,
            "candidates": [action.to_dict() for action in pair.candidates],
        }
        for side, frames, text, history in (
            ("original", pair.original_frames, pair.original_instruction, pair.original_executed),
            (
                "perturbed",
                pair.perturbed_frames,
                pair.perturbed_instruction,
                pair.perturbed_executed,
            ),
        ):
            key = f"case{number}_{side}"
            arrays[key] = frames
            row[side] = {
                "frames_key": key,
                "instruction": text,
                "executed_actions": [action.to_dict() for action in history],
            }
        cases.append(row)
    archive = directory / "frames.npz"
    np.savez_compressed(archive, **arrays)
    manifest = {
        "version": SENSITIVITY_VERSION,
        "preparation_version": PREPARATION_VERSION,
        "split": "validation",
        "seed": seed,
        "level_bank_seed": level_bank_seed,
        "image_size": image_size,
        "cases": cases,
        "frames_file": archive.name,
        "frames_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "retained_rgb_bytes": retained,
        "archive_bytes": archive.stat().st_size,
        "instructions_sha256": hashlib.sha256(instructions.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment_source_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "envs/atomic.py").read_bytes()
        ).hexdigest(),
        "training_updates": 0,
        "test_used": False,
        "interpretation": "Paired input dependence only; not game success or win quality.",
    }
    path = directory / "pairs.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    load_sensitivity_bundle(path)  # Verify byte bounds and all pairing invariants before returning.
    return path
