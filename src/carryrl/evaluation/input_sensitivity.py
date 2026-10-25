"""Bounded frozen, paired likelihood probes; sensitivity is not game success.

Candidate actions are labels only. Explicit recorded executed actions supply the
recurrent and decoder histories on both sides of every comparison.
"""

import copy
import hashlib
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.language import validate_instruction
from carryrl.envs.atomic_suite import TASKS

SENSITIVITY_VERSION = "carryrl.input_sensitivity.v1"
KINDS = ("rgb_position", "text_control", "memory_cue", "executed_history")
MAX_STEPS = 80
MAX_CASES = 64
MAX_BUNDLE_BYTES = 256 * 1024**2


def _json_hash(record):
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _frames_hash(frames):
    return _json_hash(
        {
            "shape": frames.shape,
            "dtype": str(frames.dtype),
            "pixels": hashlib.sha256(memoryview(np.ascontiguousarray(frames))).hexdigest(),
        }
    )


@dataclass(frozen=True)
class SensitivityPair:
    case_id: str
    task: str
    kind: str
    original_frames: np.ndarray
    perturbed_frames: np.ndarray
    original_instruction: str | None
    perturbed_instruction: str | None
    original_executed: tuple[PhysicalAction, ...]
    perturbed_executed: tuple[PhysicalAction, ...]
    candidates: tuple[PhysicalAction, ...]
    probe_step: int
    provenance: dict

    def validate(self):
        if self.task not in TASKS or self.kind not in KINDS:
            raise ValueError("unsupported sensitivity task or perturbation")
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("sensitivity case needs an identifier")
        original, perturbed = self.original_frames, self.perturbed_frames
        if (
            not isinstance(original, np.ndarray)
            or not isinstance(perturbed, np.ndarray)
            or original.dtype != np.uint8
            or perturbed.dtype != np.uint8
            or original.ndim != 4
            or original.shape[-1] != 3
            or original.shape != perturbed.shape
            or not 1 <= len(original) <= MAX_STEPS
            or not 1 <= original.shape[1] <= 1024
            or not 1 <= original.shape[2] <= 1024
            or original.nbytes + perturbed.nbytes > MAX_BUNDLE_BYTES
        ):
            raise ValueError("paired RGB must be uint8 [1..80,H,W,3] with identical bounded shapes")
        for actions in (self.original_executed, self.perturbed_executed):
            if len(actions) != len(original) or any(
                not isinstance(a, PhysicalAction) for a in actions
            ):
                raise ValueError("recorded executed history must match every observed frame")
        if not 1 <= len(self.candidates) <= 8 or any(
            not isinstance(action, PhysicalAction) for action in self.candidates
        ):
            raise ValueError("each pair requires 1..8 shared physical action candidates")
        if type(self.probe_step) is not int or not 0 <= self.probe_step < len(original):
            raise ValueError("probe_step must index the bounded recorded trajectory")
        for instruction in (self.original_instruction, self.perturbed_instruction):
            validate_instruction(instruction)
        if (
            not isinstance(self.provenance, dict)
            or self.provenance.get("split") != "validation"
            or not self.provenance.get("original_executed_source")
            or not self.provenance.get("perturbed_executed_source")
        ):
            raise ValueError(
                "validation-only provenance must identify both executed-history sources"
            )
        same_frames = np.array_equal(original, perturbed)
        same_text = self.original_instruction == self.perturbed_instruction
        same_history = self.original_executed == self.perturbed_executed
        if self.kind == "text_control":
            if not same_frames or not same_history or same_text:
                raise ValueError("text probe changes only instruction under identical RGB/history")
        elif self.kind == "executed_history":
            if not same_frames or not same_text or same_history:
                raise ValueError(
                    "history probe needs matched RGB/text and distinct recorded histories"
                )
            if (
                self.original_executed[: self.probe_step]
                == self.perturbed_executed[: self.probe_step]
            ):
                raise ValueError("history perturbation must precede the probe action")
        elif self.kind == "memory_cue":
            if (
                not same_history
                or not same_text
                or self.probe_step == 0
                or not np.array_equal(original[self.probe_step], perturbed[self.probe_step])
                or np.array_equal(original[: self.probe_step], perturbed[: self.probe_step])
            ):
                raise ValueError(
                    "cue probe changes earlier RGB with matched current RGB/text/history"
                )
        elif (
            same_frames
            or not same_text
            or not same_history
            or np.array_equal(original[self.probe_step], perturbed[self.probe_step])
        ):
            raise ValueError("position probe changes current RGB with matched text/history")
        return self


def load_sensitivity_bundle(manifest_path) -> tuple[list[SensitivityPair], dict]:
    """Read JSON metadata + NPZ RGB, without pickle or implicit policy histories."""
    manifest_path = Path(manifest_path)
    if manifest_path.stat().st_size > 4 * 1024**2:
        raise ValueError("sensitivity JSON metadata exceeds byte budget")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("version") != SENSITIVITY_VERSION
        or manifest.get("split") != "validation"
        or not isinstance(manifest.get("cases"), list)
        or not 1 <= len(manifest["cases"]) <= MAX_CASES
    ):
        raise ValueError("unsupported or unbounded validation sensitivity bundle")
    directory = manifest_path.parent.resolve()
    archive = (directory / manifest["frames_file"]).resolve()
    if not archive.is_relative_to(directory):
        raise ValueError("sensitivity RGB archive must be inside the bundle directory")
    if archive.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("sensitivity archive exceeds byte budget")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != manifest.get("frames_sha256"):
        raise ValueError("sensitivity frame archive SHA256 mismatch")
    with zipfile.ZipFile(archive) as container:
        if sum(item.file_size for item in container.infolist()) > MAX_BUNDLE_BYTES:
            raise ValueError("uncompressed sensitivity archive exceeds byte budget")
    pairs, identifiers = [], set()
    retained_bytes = 0
    with np.load(archive, allow_pickle=False) as arrays:
        for row in manifest["cases"]:
            if row["case_id"] in identifiers:
                raise ValueError("sensitivity case identifiers must be unique")
            identifiers.add(row["case_id"])
            original, perturbed = row["original"], row["perturbed"]
            pair = SensitivityPair(
                case_id=row["case_id"],
                task=row["task"],
                kind=row["kind"],
                original_frames=arrays[original["frames_key"]],
                perturbed_frames=arrays[perturbed["frames_key"]],
                original_instruction=original.get("instruction"),
                perturbed_instruction=perturbed.get("instruction"),
                original_executed=tuple(
                    PhysicalAction.from_dict(action) for action in original["executed_actions"]
                ),
                perturbed_executed=tuple(
                    PhysicalAction.from_dict(action) for action in perturbed["executed_actions"]
                ),
                candidates=tuple(PhysicalAction.from_dict(action) for action in row["candidates"]),
                probe_step=row["probe_step"],
                provenance=row["provenance"],
            ).validate()
            retained_bytes += pair.original_frames.nbytes + pair.perturbed_frames.nbytes
            if retained_bytes > MAX_BUNDLE_BYTES:
                raise ValueError("loaded sensitivity pairs exceed retained RGB byte budget")
            pairs.append(pair)
    return pairs, {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "frames_sha256": digest,
        "manifest": str(manifest_path.resolve()),
    }


@torch.no_grad()
def evaluate_input_sensitivity(agent, pairs, *, encode_frame=None):
    """Replay complete prefixes for the same candidates, with explicit executed inputs."""
    if not 1 <= len(pairs) <= MAX_CASES:
        raise ValueError("sensitivity evaluation requires 1..64 paired cases")
    for pair in pairs:
        pair.validate()
    agent.eval().requires_grad_(False)
    if encode_frame is None:
        from carryrl.training.sequence import FrozenFrameCache

        cache = FrozenFrameCache(agent, capacity=160, max_bytes=64 * 1024**2)
        encode_frame = cache.encode
    records = []
    try:
        for pair in pairs:
            # Strict encoder validation also prevents the policy's OCR truncation
            # path from silently accepting an overflowing instruction probe.
            for text in (pair.original_instruction, pair.perturbed_instruction):
                if text is not None:
                    if getattr(agent, "language", None) is None:
                        raise ValueError("text sensitivity requires the checkpoint's text encoder")
                    agent.language.encode_instruction_tokens(text)
            prefix_length = pair.probe_step + 1
            scores = []
            for frames, instruction, executed in (
                (pair.original_frames, pair.original_instruction, pair.original_executed),
                (pair.perturbed_frames, pair.perturbed_instruction, pair.perturbed_executed),
            ):
                patches = torch.stack([encode_frame(frame) for frame in frames[:prefix_length]])
                side = []
                for candidate in pair.candidates:
                    # Labels before probe are immaterial to recurrent input: the
                    # separately supplied recorded executed history is authoritative.
                    labels = [*executed[: pair.probe_step], candidate]
                    replay = agent.evaluate_episode(
                        patches,
                        labels,
                        instruction=instruction,
                        executed_actions=list(executed[:prefix_length]),
                    )
                    value = float(replay["log_prob"][-1])
                    if not math.isfinite(value):
                        raise FloatingPointError("non-finite paired candidate joint log likelihood")
                    side.append(value)
                scores.append(side)
            candidates = [
                {
                    "action": action.to_dict(),
                    "original_log_prob": original,
                    "perturbed_log_prob": perturbed,
                    "log_prob_delta": perturbed - original,
                    "original_nll": -original,
                    "perturbed_nll": -perturbed,
                    "nll_delta": original - perturbed,
                }
                for action, original, perturbed in zip(pair.candidates, *scores, strict=True)
            ]
            records.append(
                {
                    "case_id": pair.case_id,
                    "task": pair.task,
                    "kind": pair.kind,
                    "probe_step": pair.probe_step,
                    "prefix_steps": prefix_length,
                    "original_frames_sha256": _frames_hash(pair.original_frames[:prefix_length]),
                    "perturbed_frames_sha256": _frames_hash(pair.perturbed_frames[:prefix_length]),
                    "original_current_frame_sha256": _frames_hash(
                        pair.original_frames[pair.probe_step]
                    ),
                    "perturbed_current_frame_sha256": _frames_hash(
                        pair.perturbed_frames[pair.probe_step]
                    ),
                    "original_instruction": pair.original_instruction,
                    "perturbed_instruction": pair.perturbed_instruction,
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
                    "original_executed_actions": [
                        a.to_dict() for a in pair.original_executed[:prefix_length]
                    ],
                    "perturbed_executed_actions": [
                        a.to_dict() for a in pair.perturbed_executed[:prefix_length]
                    ],
                    "candidate_results": candidates,
                    "provenance": pair.provenance,
                }
            )
    finally:
        agent.reset_episode()
    summaries = {}
    for kind in KINDS:
        decisions = [
            candidate
            for row in records
            if row["kind"] == kind
            for candidate in row["candidate_results"]
        ]
        if decisions:
            summaries[kind] = {
                "cases": sum(row["kind"] == kind for row in records),
                "candidate_comparisons": len(decisions),
                "mean_abs_log_prob_delta": sum(abs(row["log_prob_delta"]) for row in decisions)
                / len(decisions),
                "mean_nll_delta": sum(row["nll_delta"] for row in decisions) / len(decisions),
            }
    return {
        "version": SENSITIVITY_VERSION,
        "split": "validation",
        "status": "complete",
        "cases": records,
        "summaries": summaries,
        "max_steps": MAX_STEPS,
        "training_updates": 0,
        "interpretation": "Input dependence only; neither sensitivity nor NLL proves game success.",
        "memory_control_reference": "scripts/evaluate_memory_controls.py",
    }


def prepare_mouse_position_bundle(directory, *, image_size=96, seed=20261020, level_bank_seed=1729):
    """One deterministic validation scene pair, same geometry/cursor, target left/right.

    Only the evaluator edits the copied visible target location. The policy sees
    RGB. Use the existing memory-controls evaluator for interactive cue/noop runs.
    """
    from carryrl.envs.atomic import AtomicEnv

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    frames, origins = [], []
    for target in (11, 13):
        env = AtomicEnv(
            "mouse",
            split="validation",
            image_size=image_size,
            max_steps=80,
            level_bank_seed=level_bank_seed,
            difficulty=1.0,
        )
        try:
            _, info = env.reset(seed=seed, options={"level_index": 0})
            env._level = copy.deepcopy(env._level)
            env._level["targets"][0] = target
            env._cursor = list(env._mouse_location(12))
            frame = env._draw()
            env._frame = frame.copy()
            frames.append(frame[None])
            env.step(PhysicalAction())  # Actual recorded command after this observation.
            origins.append({"bank_level_id": info["level_id"], "target_index": target})
        finally:
            env.close()
    archive = directory / "frames.npz"
    np.savez_compressed(archive, original=frames[0], perturbed=frames[1])
    command = PhysicalAction().to_dict()
    manifest = {
        "version": SENSITIVITY_VERSION,
        "split": "validation",
        "frames_file": "frames.npz",
        "frames_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "cases": [
            {
                "case_id": "mouse-target-left-right",
                "task": "mouse",
                "kind": "rgb_position",
                "original": {
                    "frames_key": "original",
                    "instruction": None,
                    "executed_actions": [command],
                },
                "perturbed": {
                    "frames_key": "perturbed",
                    "instruction": None,
                    "executed_actions": [command],
                },
                "probe_step": 0,
                "candidates": [PhysicalAction(dx=amount).to_dict() for amount in (-16, 0, 16)],
                "provenance": {
                    "split": "validation",
                    "seed": seed,
                    "level_bank_seed": level_bank_seed,
                    "original_executed_source": "validation noop executed after original RGB",
                    "perturbed_executed_source": "validation noop executed after perturbed RGB",
                    "controlled_change": "first target column; matched row, geometry, cursor, seed",
                    "scene_origins": origins,
                },
            }
        ],
    }
    path = directory / "pairs.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
