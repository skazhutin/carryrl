"""Replay frozen validation failures and measure bounded RGB-teacher continuations."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from carryrl.agent.actions import CODEC_VERSION, PhysicalAction
from carryrl.dashboard.frozen_playback import SOURCE as PLAYBACK_SOURCE
from carryrl.envs.atomic_suite import SCORING_VERSION, SUITE_VERSION
from carryrl.evaluation.paired import episode_descriptor
from carryrl.evaluation.rescue import diagnose_teacher_rescue
from carryrl.training.checkpoint_storage import checkpoint_read_lock
from carryrl.training.run import atomic_json, file_sha256, provenance
from carryrl.training.sequence import _episode_instruction, sequence_evaluation_protocol
from carryrl.training.sequence_config import SequenceRunConfig


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _inside(directory, path):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(directory):
        raise ValueError("capture artifact must belong to its source run")
    return path


def _key(record):
    return _json_hash(episode_descriptor(record))


@dataclass(frozen=True)
class CaptureSource:
    directory: Path
    config: SequenceRunConfig
    metadata: dict
    evaluation: dict
    entries: tuple[tuple[dict, dict], ...]
    file_hashes: dict[str, str]

    def assert_unchanged(self):
        with checkpoint_read_lock(Path(self.metadata["checkpoint"])):
            if any(file_sha256(Path(path)) != expected
                   for path, expected in self.file_hashes.items()):
                raise RuntimeError("teacher rescue source or checkpoint changed")


def prepare_source(directory):
    """Require one complete, frozen, fully captured validation failure cohort."""
    directory = Path(directory).resolve(strict=True)
    files = {}

    def read(path):
        path = _inside(directory, path)
        files[str(path)] = file_sha256(path)
        return json.loads(path.read_text())

    metadata = read(directory / "metadata.json")
    status = read(directory / "status.json")
    evaluation = read(directory / "evaluation.json")
    full = read(directory / "evaluations/frozen_full.json")
    if full != evaluation:
        raise ValueError("source full evaluation and published evaluation differ")
    if (status.get("status") != "completed" or status.get("updates") != 0
            or status.get("checkpoint_files_unchanged") is not True
            or status.get("policy_and_reward_frozen") is not True
            or metadata.get("kind") != "frozen_imitation_diagnostics"
            or metadata.get("updates") != 0 or metadata.get("split") != "validation"
            or evaluation.get("policy_and_reward_frozen") is not True
            or evaluation.get("split") != "validation"
            or evaluation.get("suite_version") != SUITE_VERSION
            or evaluation.get("scoring_version") != SCORING_VERSION):
        raise ValueError("teacher rescue requires a completed, current frozen validation source")
    config = SequenceRunConfig.from_dict(metadata["config"])
    if (not config.eval_all_levels or config.eval_stochastic or config.env_image_size != 96
            or config.max_episode_steps > 80
            or config.visual_config is not None and config.visual_config.mode != "canonical"):
        raise ValueError(
            "teacher rescue requires full canonical96 deterministic ordinary evaluation")
    protocol = sequence_evaluation_protocol(config, split="validation")
    protocol_sha = _json_hash(protocol)
    if (protocol["difficulty"] != 1.0 or _json_hash(metadata.get("protocol")) != protocol_sha
            or metadata.get("protocol_sha256") != protocol_sha
            or evaluation.get("protocol_sha256") != protocol_sha):
        raise ValueError("teacher rescue source protocol/config differs")
    checkpoint = Path(metadata["checkpoint"]).resolve(strict=True)
    checkpoint_hashes = metadata["checkpoint_files_sha256"]
    if not {"manifest.json", "state.pt"}.issubset(checkpoint_hashes):
        raise ValueError("teacher rescue requires pinned checkpoint state and manifest")
    with checkpoint_read_lock(checkpoint):
        for name, expected in checkpoint_hashes.items():
            path = (checkpoint / name).resolve(strict=True)
            if not path.is_relative_to(checkpoint) or file_sha256(path) != expected:
                raise ValueError("teacher rescue source checkpoint SHA256 differs")
            files[str(path)] = expected
        manifest_path = checkpoint / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("sha256") != checkpoint_hashes["state.pt"]
                or manifest.get("codec_version") != CODEC_VERSION
                or manifest.get("agent_sha256") != evaluation.get("agent_sha256")
                or manifest.get("reward_sha256") != evaluation.get("reward_sha256")
                or manifest.get("weights_revision") != evaluation.get("weights_revision")):
            raise ValueError("teacher rescue source weights/revision/codec differs from checkpoint")
    expected_order = [(policy, task, index) for policy in protocol["policies"]
                      for task in config.tasks for index in range(protocol["counts"][task])]
    actual_order = [(row["policy"], row["task"], row["episode_index"])
                    for row in evaluation["records"]]
    if actual_order != expected_order or any(
        row["level_bank_seed"] != config.level_bank_seed or row["difficulty"] != 1.0
        or row["instruction"] != _episode_instruction(config, row["task"], row["seed"])
        for row in evaluation["records"]
    ):
        raise ValueError("teacher rescue source is not the declared complete ordered cohort")
    failures = [row for row in evaluation["records"]
                if row["policy"] == "frozen_deterministic"
                and row["outcome"].get("success") is not True]
    expected = {_key(row): row for row in failures}
    captures = metadata.get("playback_capture", {})
    if (captures != evaluation.get("playback_capture") or captures != status.get("playback_capture")
            or captures.get("requested_scope") not in ("failed", "all")
            or captures.get("enabled") is not True
            or len(expected) != len(failures)):
        raise ValueError("teacher rescue source capture cohort is incomplete or inconsistent")
    selected = captures["selected_episodes"]
    selected_keys = [_key(row["episode"]) for row in selected]
    if len(set(selected_keys)) != len(selected_keys) or captures["captured_count"] != len(selected):
        raise ValueError("teacher rescue capture identities/count differ")
    if not set(expected).issubset(selected_keys):
        raise ValueError("teacher rescue is missing frozen failure captures")
    desired = (set(expected) if captures["requested_scope"] == "failed" else
               {_key(row) for row in evaluation["records"]
                if row["policy"] == "frozen_deterministic"})
    if set(selected_keys) != desired:
        raise ValueError("teacher rescue capture scope contains a different episode cohort")
    entries = []
    for entry in selected:
        record = expected.get(_key(entry["episode"]))
        if record is None:
            continue
        if (entry["episode"].get("split") != "validation"
                or entry["episode"].get("level_bank_seed") != config.level_bank_seed
                or entry.get("outcome") != record["outcome"]):
            raise ValueError("teacher rescue capture differs from its evaluation record")
        sidecar = read(entry["playback"])
        diagnostic = read(entry["diagnostic"])
        for field in record:
            if sidecar.get(field) != record[field]:
                raise ValueError(f"teacher rescue playback differs from evaluation field {field}")
        if (sidecar.get("source") != PLAYBACK_SOURCE
                or sidecar.get("training_performed") is not False
                or sidecar.get("teacher_actions_used") is not False
                or sidecar.get("agent_sha256") != evaluation["agent_sha256"]
                or sidecar.get("weights_revision") != evaluation["weights_revision"]
                or sidecar.get("checkpoint_sha256") != manifest["sha256"]
                or Path(sidecar["checkpoint"]).resolve() != checkpoint
                or Path(sidecar["evaluation_reference"]).resolve()
                != directory / "evaluations/frozen_full.json"
                or diagnostic.get("replay_verified") is not True
                or diagnostic.get("agent_sha256") != evaluation["agent_sha256"]
                or diagnostic.get("weights_revision") != evaluation["weights_revision"]
                or diagnostic.get("protocol_sha256") != protocol_sha
                or diagnostic.get("outcome") != record["outcome"]
                or diagnostic.get("quality") != record.get("quality")):
            raise ValueError("teacher rescue source provenance or diagnostic differs")
        sprite_name = sidecar.get("sprite_filename")
        if not isinstance(sprite_name, str) or Path(sprite_name).name != sprite_name:
            raise ValueError("teacher rescue sprite must be a sibling filename")
        sprite = _inside(directory, Path(entry["playback"]).parent / sprite_name)
        actual_hash = file_sha256(sprite)
        if actual_hash != sidecar.get("sprite_sha256"):
            raise ValueError("teacher rescue sprite SHA256 differs")
        files[str(sprite)] = actual_hash
        entries.append((record, {**entry, "sidecar": sidecar, "diagnostic_body": diagnostic,
                                 "sprite": str(sprite)}))
    source = CaptureSource(directory, config, metadata, evaluation, tuple(entries), files)
    source.assert_unchanged()
    return source


def load_frames(entry):
    sidecar = entry["sidecar"]
    width, height, count, columns = (sidecar.get(key) for key in
        ("frame_width", "frame_height", "frame_count", "sprite_columns"))
    if (width != 96 or height != 96 or type(count) is not int or not 2 <= count <= 81
            or type(columns) is not int or not 1 <= columns <= 81
            or len(sidecar["actions"]) != count - 1
            or len(sidecar["frame_sha256"]) != count):
        raise ValueError("teacher rescue capture dimensions/count differ")
    frames = []
    with Image.open(entry["sprite"]) as image:
        expected_size = (columns * width, math.ceil(count / columns) * height)
        if image.mode != "RGB" or image.size != expected_size:
            raise ValueError("teacher rescue lossless sprite geometry differs")
        for index in range(count):
            x, y = index % columns * width, index // columns * height
            frame = np.frombuffer(image.crop((x, y, x + width, y + height)).tobytes(),
                                  dtype=np.uint8).reshape(height, width, 3)
            if hashlib.sha256(frame.tobytes()).hexdigest() != sidecar["frame_sha256"][index]:
                raise ValueError("teacher rescue frame SHA256 differs")
            frames.append(frame)
    actions = []
    for index, transition in enumerate(sidecar["actions"]):
        if (transition["step"] != index
                or transition["observation_sha256"] != sidecar["frame_sha256"][index]
                or transition["next_observation_sha256"] != sidecar["frame_sha256"][index + 1]):
            raise ValueError("teacher rescue transition alignment differs")
        actions.append(PhysicalAction.from_dict(transition["action"]))
    diagnostic = entry["diagnostic_body"]
    if ([action.to_dict() for action in actions] != diagnostic["actual_actions"]
            or sidecar["frame_sha256"] != diagnostic["frame_sha256"]):
        raise ValueError("teacher rescue diagnostic RGB/actions differ from capture")
    return tuple(frames), tuple(actions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchors", type=int, nargs="+", default=[0, 20, 40, 60])
    parser.add_argument("--stall-window", type=int, default=8)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args(argv)
    if (0 not in args.anchors or len(set(args.anchors)) != len(args.anchors)
            or any(not 0 <= anchor < 80 for anchor in args.anchors)
            or not 1 <= args.stall_window <= 80):
        parser.error("anchors require unique steps0..79 including0; stall-window requires1..80")
    output, source_dir = args.output.resolve(), args.source_run.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError("teacher rescue output already exists")
    if output.is_relative_to(source_dir):
        raise ValueError("teacher rescue output must be separate from its source")
    source = prepare_source(source_dir)
    checkpoint = Path(source.metadata["checkpoint"]).resolve()
    if output.is_relative_to(checkpoint):
        raise ValueError("teacher rescue output cannot be inside the checkpoint")
    if any(anchor >= source.config.max_episode_steps for anchor in args.anchors):
        parser.error("anchor must precede the source episode deadline")
    if args.tasks and (len(set(args.tasks)) != len(args.tasks)
                       or set(args.tasks) - set(source.config.tasks)):
        parser.error("tasks must select unique source games")
    entries = [(record, entry) for record, entry in source.entries
               if not args.tasks or record["task"] in args.tasks]
    if not entries:
        raise ValueError("teacher rescue selected no source failures")
    output.mkdir(parents=True, exist_ok=False)
    producer = provenance()
    metadata = {"kind": "frozen_teacher_rescue_audit", "training_updates": 0,
        "source_run": str(source.directory), "source_files_sha256": source.file_hashes,
        "source_protocol_sha256": source.evaluation["protocol_sha256"],
        "source_agent_sha256": source.evaluation["agent_sha256"],
        "source_weights_revision": source.evaluation["weights_revision"],
        "source_config": source.config.to_dict(), "anchors": args.anchors,
        "stall_window": args.stall_window, "all_source_nonwins": len(source.entries),
        "selected_nonwins": len(entries), "tasks": args.tasks,
        "coverage": "all_source_nonwins" if not args.tasks else "declared_game_subset",
        "policy_loaded": False, "reward_model_loaded": False, "optimizer_constructed": False,
        "interpretation": "Validation-only controller recovery, not trained policy quality. "
                          "Failure of one teacher is not physical impossibility. "
                          "These source/continuation traces are not training examples.", **producer}
    atomic_json(output / "metadata.json", metadata)
    totals, games, published, teacher_errors = defaultdict(Counter), {}, [], 0
    try:
        for position, (record, entry) in enumerate(entries):
            atomic_json(output / "status.json", {"status": "running", "training_updates": 0,
                "completed": position, "total": len(entries), "task": record["task"]})
            frames, actions = load_frames(entry)
            result = diagnose_teacher_rescue(source.config, record, frames, actions,
                anchors=tuple(args.anchors), stall_window=args.stall_window)
            if result.report["source_trace_sha256"] != entry["diagnostic_body"]["trace_sha256"]:
                raise ValueError(
                    "teacher rescue independently authenticated a different source trace")
            label = f"{record['task']}_episode_{record['episode_index']:04d}"
            destination = output / "episodes" / label
            destination.mkdir(parents=True)
            payload = {f"{anchor}_frame_{index:03d}": frame
                       for anchor, group in result.frames_by_anchor.items()
                       for index, frame in enumerate(group)}
            np.savez_compressed(destination / "suffix_frames.npz", **payload)
            saved = {**result.report, "source_playback": entry["playback"],
                "source_diagnostic": entry["diagnostic"],
                "suffix_frames_sha256": file_sha256(destination / "suffix_frames.npz")}
            atomic_json(destination / "report.json", saved)
            published.append({"episode": episode_descriptor(record),
                "report": str((destination / "report.json").resolve()),
                "report_sha256": file_sha256(destination / "report.json")})
            for attempt in result.report["attempts"]:
                anchor, state = attempt["anchor_id"], attempt["status"]
                totals[anchor][state] += 1
                games.setdefault(record["task"], defaultdict(Counter))[anchor][state] += 1
                teacher_errors += int(state == "teacher_error")
            del result, frames, actions, payload
        source.assert_unchanged()
        if provenance()["source_and_lock_sha256"] != producer["source_and_lock_sha256"]:
            raise RuntimeError("teacher rescue producer source changed during the audit")
        report = {"status": "completed_with_teacher_errors" if teacher_errors else "completed",
            "training_updates": 0, "source_files_unchanged": True,
            "source_episodes": len(source.entries), "selected_episodes": len(entries),
            "teacher_errors": teacher_errors, "by_anchor": dict(totals), "by_game": games,
            "published": published, "interpretation": metadata["interpretation"]}
        atomic_json(output / "aggregate.json", report)
        atomic_json(output / "status.json", {key: value for key, value in report.items()
                                             if key not in ("published", "by_game")})
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "interrupted" if isinstance(
            error, KeyboardInterrupt) else "failed", "training_updates": 0,
            "error_type": type(error).__name__, "error": str(error),
            "completed": len(published), "total": len(entries)})
        raise
    summary_keys = (
        "status", "training_updates", "selected_episodes", "by_anchor", "teacher_errors")
    print(json.dumps({key: report[key] for key in summary_keys}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
