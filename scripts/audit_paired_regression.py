"""Audit every saved VALIDATION lost win using frozen paired closed-loop trajectories."""

import argparse
import gc
import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from carryrl.agent.actions import CODEC_VERSION, PhysicalAction
from carryrl.envs.atomic import atomic_level_banks
from carryrl.envs.atomic_suite import SCORING_VERSION, SUITE_VERSION
from carryrl.evaluation.paired import episode_descriptor, paired_outcomes
from carryrl.evaluation.paired_trajectories import (
    FrozenPolicyCase,
    capture_frozen_case,
    compare_frozen_cases,
    verify_sham,
)
from carryrl.training.checkpoint_storage import checkpoint_read_lock
from carryrl.training.run import atomic_json, file_sha256, provenance
from carryrl.training.sequence import (
    _episode_instruction,
    load_sequence_checkpoint,
    sequence_evaluation_protocol,
    tensor_hash,
)
from carryrl.training.sequence_config import SequenceRunConfig

VERSION = "carryrl.paired_regression_audit.v1"
MAX_CASE_RGB_BYTES = 32 * 1024**2
ROLES = ("before", "candidate", "sham", "intervention")


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _checkpoint_hashes(checkpoint):
    with checkpoint_read_lock(checkpoint):
        return _checkpoint_hashes_locked(checkpoint)


def _checkpoint_hashes_locked(checkpoint):
    hashes = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint pins require regular files without symlinks")
        if path.is_file():
            hashes[path.relative_to(checkpoint).as_posix()] = file_sha256(path)
    return hashes


@dataclass(frozen=True)
class Source:
    checkpoint: Path
    config: SequenceRunConfig
    manifest: dict
    evaluation: dict
    protocol: dict
    file_pins: dict
    checkpoint_pins: dict

    def assert_unchanged(self):
        if (_checkpoint_hashes(self.checkpoint) != self.checkpoint_pins
                or any(file_sha256(Path(path)) != digest
                       for path, digest in self.file_pins.items())):
            raise RuntimeError("paired audit source files changed")

    def describe(self):
        return {"checkpoint": str(self.checkpoint), "manifest": self.manifest,
                "config": self.config.to_dict(), "file_sha256": self.file_pins,
                "checkpoint_files_sha256": self.checkpoint_pins,
                "legacy_weights_manifest": any(key not in self.manifest
                    for key in ("suite_version", "scoring_version"))}


def _source(checkpoint, config_path, evaluation_path, device):
    checkpoint = checkpoint.resolve(strict=True)
    pins = {}

    def read(path):
        path = path.resolve(strict=True)
        pins[str(path)] = file_sha256(path)
        return json.loads(path.read_text())

    config = replace(SequenceRunConfig.from_dict(read(config_path)), device=device,
                     eval_all_levels=True)
    evaluation = read(evaluation_path)
    manifest = read(checkpoint / "manifest.json")
    checkpoint_pins = _checkpoint_hashes(checkpoint)
    if (checkpoint_pins.get("state.pt") != manifest.get("sha256")
            or not {"state.pt", "manifest.json"}.issubset(checkpoint_pins)):
        raise ValueError("checkpoint archive SHA256 mismatch before policy loading")
    language = manifest.get("language_resources_sha256")
    if language is not None:
        actual = {key.removeprefix("language/"): value for key, value in checkpoint_pins.items()
                  if key.startswith("language/")}
        if not language or actual != language:
            raise ValueError("checkpoint language SHA256 mismatch")
    if (manifest.get("codec_version") != CODEC_VERSION
            or "suite_version" in manifest and manifest["suite_version"] != SUITE_VERSION
            or "scoring_version" in manifest and manifest["scoring_version"] != SCORING_VERSION):
        raise ValueError("paired audit requires current checkpoint codec/suite/scoring")
    protocol = sequence_evaluation_protocol(config)
    if (config.eval_stochastic or config.env_image_size != 96
            or config.max_episode_steps > 80 or protocol["difficulty"] != 1.0
            or config.visual_config is not None and config.visual_config.mode != "canonical"):
        raise ValueError("audit requires full canonical96 deterministic validation")
    if (evaluation.get("protocol_sha256") != _json_hash(protocol)
            or evaluation.get("split") != "validation"
            or evaluation.get("suite_version") != SUITE_VERSION
            or evaluation.get("scoring_version") != SCORING_VERSION
            or evaluation.get("policy_and_reward_frozen") is not True
            or evaluation.get("level_bank_seed") != config.level_bank_seed):
        raise ValueError("historical reference protocol/config/current versions differ")
    for field in ("agent_sha256", "reward_sha256", "weights_revision"):
        if field not in manifest or evaluation.get(field) != manifest[field]:
            raise ValueError(f"historical reference checkpoint {field} differs")
    if ("checkpoint_weights_revision" in evaluation
            and evaluation["checkpoint_weights_revision"] != manifest["weights_revision"]):
        raise ValueError("historical reference checkpoint revision differs")
    records = evaluation.get("records", [])
    expected = [(policy, task, index) for policy in protocol["policies"]
                for task in config.tasks for index in range(protocol["counts"][task])]
    if [(row.get("policy"), row.get("task"), row.get("episode_index"))
            for row in records] != expected:
        raise ValueError("historical reference is not the complete ordered full cohort")
    for row in records:
        descriptor = episode_descriptor(row)
        task, index = descriptor["task"], descriptor["episode_index"]
        task_index = config.tasks.index(task)
        level_count = len(atomic_level_banks(task, config.level_bank_seed)["validation"])
        if (row.get("seed") != 1_000_000 + index
                or row.get("action_seed") != (config.eval_seed + task_index * 100_000 + index)
                % 2**32 or row.get("level_index") != index % level_count
                or row.get("difficulty") != 1.0
                or row.get("level_bank_seed") != config.level_bank_seed
                or row.get("instruction") != _episode_instruction(config, task, row["seed"])
                or type(row.get("steps")) is not int
                or not 1 <= row["steps"] <= config.max_episode_steps
                or type(row.get("terminated")) is not bool
                or type(row.get("truncated")) is not bool
                or not (row["terminated"] or row["truncated"])):
            raise ValueError("historical episode differs from its declared cohort")
    # Also validates duplicate identities and malformed outcomes in the random control.
    paired_outcomes(records, records)
    source = Source(checkpoint, config, manifest, evaluation, protocol, pins, checkpoint_pins)
    source.assert_unchanged()
    return source


def _selection(before, after):
    if (_json_hash(before.protocol) != _json_hash(after.protocol)
            or before.evaluation["reward_sha256"] != after.evaluation["reward_sha256"]):
        raise ValueError("before/after evaluation protocol or frozen reward differs")
    records = [[row for row in source.evaluation["records"]
                if row["policy"] == "frozen_deterministic"] for source in (before, after)]
    paired = paired_outcomes(*records)
    selected = [(left, right) for left, right in zip(*records, strict=True)
                if left["outcome"].get("known") is True
                and right["outcome"].get("known") is True
                and left["outcome"].get("success") is True
                and right["outcome"].get("success") is False]
    if not selected:
        raise ValueError("historical references contain no known lost wins")
    return selected, paired


def _prepare_output(output, sources):
    requested = output
    output = output.resolve()
    for source in sources:
        checkpoint = source.checkpoint
        roots = [parent.parent for parent in checkpoint.parents
                 if parent.name in {"checkpoints", "checkpoints_by_epoch"}]
        root = roots[-1] if roots else checkpoint
        if output == root or root in output.parents or output in checkpoint.parents:
            raise ValueError("audit output must be outside both source runs/checkpoints")
        for path in source.file_pins:
            if Path(path).is_relative_to(output):
                raise ValueError("audit output must not contain pinned source files")
    if requested.is_symlink():
        raise FileExistsError("audit output is an existing symlink")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _assert_reference(case, reference):
    for field in ("outcome", "steps", "terminated", "truncated"):
        if case.record.get(field) != reference[field]:
            raise ValueError(f"current frozen replay differs from historical {field}")
    if episode_descriptor(case.record) != episode_descriptor(reference):
        raise ValueError("current frozen replay descriptor differs from historical identity")
    if "quality" in reference and case.record.get("quality") != reference["quality"]:
        raise ValueError("current frozen replay quality differs from historical reference")


def _save_case(directory, case, source, role):
    directory.mkdir(parents=True, exist_ok=False)
    raw_bytes = sum(frame.nbytes for frame in case.frames)
    if raw_bytes > MAX_CASE_RGB_BYTES:
        raise MemoryError("paired case RGB exceeds 32 MiB")
    archive = directory / "frames.npz"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, allowZip64=True) as stream:
        for index, frame in enumerate(case.frames):
            if frame.dtype != np.uint8 or frame.shape != (96, 96, 3):
                raise ValueError("case storage requires canonical uint8 RGB")
            with stream.open(f"frame_{index:04d}.npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, frame, allow_pickle=False)
    frame_hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in case.frames]
    with np.load(archive, allow_pickle=False) as saved:
        expected = [f"frame_{index:04d}" for index in range(len(case.frames))]
        if saved.files != expected or any(
                hashlib.sha256(saved[key].tobytes()).hexdigest() != digest
                for key, digest in zip(expected, frame_hashes, strict=True)):
            raise ValueError("paired case RGB serialization differs")
    record = {**case.record, "audit_role": role, "agent_sha256": source.manifest["agent_sha256"],
              "reward_sha256": source.manifest["reward_sha256"],
              "weights_revision": source.manifest["weights_revision"],
              "checkpoint": str(source.checkpoint), "checkpoint_sha256": source.manifest["sha256"],
              "training_updates": 0, "optimizer_steps": 0,
              "learned_policy_alone": role != "intervention",
              "frames_file": "frames.npz", "frames_sha256": file_sha256(archive),
              "raw_rgb_bytes": raw_bytes, "frame_sha256": frame_hashes,
              "proposed_actions": [action.to_dict() for action in case.proposed_actions],
              "executed_actions": [action.to_dict() for action in case.executed_actions]}
    atomic_json(directory / "trajectory.json", record)
    return {"trajectory": str(directory / "trajectory.json"),
            "trajectory_sha256": file_sha256(directory / "trajectory.json"),
            "frames": str(archive), "frames_sha256": record["frames_sha256"],
            "outcome": record["outcome"], "raw_rgb_bytes": raw_bytes}


def _load_saved_case(entry):
    path = Path(entry["trajectory"])
    archive = Path(entry["frames"])
    if (file_sha256(path) != entry["trajectory_sha256"]
            or file_sha256(archive) != entry["frames_sha256"]):
        raise ValueError("saved baseline case changed")
    record = json.loads(path.read_text())
    with np.load(archive, allow_pickle=False) as saved:
        if sum(saved[key].nbytes for key in saved.files) > MAX_CASE_RGB_BYTES:
            raise MemoryError("saved baseline case exceeds RGB budget")
        frames = tuple(np.frombuffer(saved[key].tobytes(), dtype=np.uint8).reshape(96, 96, 3)
                       for key in saved.files)
    return FrozenPolicyCase(record, frames,
        tuple(PhysicalAction.from_dict(action) for action in record["proposed_actions"]),
        tuple(PhysicalAction.from_dict(action) for action in record["executed_actions"]))


def _load_policy(source):
    source.assert_unchanged()
    agent, optimizer, reward, state = load_sequence_checkpoint(
        source.checkpoint, source.config, evaluation_only=True, load_optimizer=False)
    if optimizer is not None:
        raise RuntimeError("paired diagnostic unexpectedly loaded optimizer")
    if state["weights_revision"] != source.manifest["weights_revision"]:
        raise ValueError("loaded checkpoint revision differs from historical reference")
    _assert_frozen(agent, reward, source)
    del state
    return agent, reward


def _assert_frozen(agent, reward, source):
    hashes = {"actor": tensor_hash(agent), "reward": tensor_hash(reward)}
    if (hashes["actor"] != source.manifest["agent_sha256"]
            or hashes["reward"] != source.manifest["reward_sha256"]):
        raise RuntimeError("paired diagnostic policy or reward tensors changed")
    return hashes


def _aggregate(cases):
    games = defaultdict(lambda: {"cases": 0, **{f"{role}_successes": 0 for role in ROLES}})
    for case in cases:
        counts = games[case["episode"]["task"]]
        counts["cases"] += 1
        for role in ROLES:
            outcome = case["roles"][role]["outcome"]
            counts[f"{role}_successes"] += int(outcome.get("known") is True
                                               and outcome.get("success") is True)
    totals = {key: sum(row[key] for row in games.values())
              for key in ("cases", *(f"{role}_successes" for role in ROLES))}
    return {"overall": totals, "games": dict(games)}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("before", "after"):
        for field in ("checkpoint", "config", "evaluation"):
            parser.add_argument(f"--{prefix}-{field}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--expected-protocol-sha256")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.device == "mps" and (not torch.backends.mps.is_available()
            or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"):
        raise RuntimeError("MPS unavailable or implicit CPU fallback enabled")
    before = _source(args.before_checkpoint, args.before_config,
                     args.before_evaluation, args.device)
    after = _source(args.after_checkpoint, args.after_config, args.after_evaluation, args.device)
    selected, paired = _selection(before, after)
    protocol_sha = _json_hash(before.protocol)
    if args.expected_protocol_sha256 and args.expected_protocol_sha256 != protocol_sha:
        raise ValueError("requested protocol SHA256 differs")
    output = _prepare_output(args.output, (before, after))
    initial_provenance = provenance()
    metadata = {"version": VERSION, "kind": "outcome_selected_validation_regression_diagnostic",
        "training_updates": 0, "optimizer_steps": 0, "split": "validation",
        "selection": "all known before-success/after-non-success pairs from complete full cohort",
        "interpretation": "outcome-selected VALIDATION subset; not a new full policy score",
        "intervention_interpretation": "one baseline command then candidate closed loop; "
                                       "intervention is not learned-policy-alone",
        "protocol": before.protocol, "protocol_sha256": protocol_sha,
        "before": before.describe(), "after": after.describe(), "paired_full_cohort": paired,
        "selected_count": len(selected),
        "full_cohort_model_episodes": sum(before.protocol["counts"].values()),
        "selected_episodes": [episode_descriptor(left)
            for left, _ in selected], "provenance": initial_provenance}
    atomic_json(output / "metadata.json", metadata)
    threads, started = torch.get_num_threads(), time.monotonic()
    entries, completed, phase_hashes = {}, [], []
    agent = reward = None
    phase, case_index = "loading_before", None

    def progress(role, index):
        atomic_json(output / "status.json", {"status": "running", "phase": role,
                                             "case_index": index, "cases": len(selected)})

    try:
        torch.set_num_threads(before.config.torch_threads)
        agent, reward = _load_policy(before)
        for index, (left, _) in enumerate(selected):
            phase, case_index = "before", index
            progress(phase, case_index)
            case = capture_frozen_case(agent, before.config, left, role="before")
            _assert_reference(case, left)
            entries[index] = _save_case(output / "cases" / f"{index:04d}" / "before",
                                        case, before, "before")
            del case
        before_hashes = _assert_frozen(agent, reward, before)
        before.assert_unchanged()
        phase_hashes.append({"phase": "before", "hashes_before": {
            "actor": before.manifest["agent_sha256"], "reward": before.manifest["reward_sha256"]},
            "hashes_after": before_hashes, "unchanged": True})
        agent = reward = None
        gc.collect()
        torch.set_num_threads(after.config.torch_threads)
        phase, case_index = "loading_after", None
        agent, reward = _load_policy(after)
        for index, (left, right) in enumerate(selected):
            phase, case_index = "candidate", index
            progress(phase, case_index)
            baseline = _load_saved_case(entries[index])
            candidate = capture_frozen_case(agent, after.config, right, role="candidate")
            _assert_reference(candidate, right)
            candidate_entry = _save_case(output / "cases" / f"{index:04d}" / "candidate",
                                          candidate, after, "candidate")
            comparison = compare_frozen_cases(baseline, candidate)
            difference = comparison.get("first_command_difference")
            if not isinstance(difference, dict):
                raise ValueError("lost win has no command difference; intervention undefined")
            step = difference["index"]
            phase = "sham"
            progress(phase, case_index)
            sham = capture_frozen_case(agent, after.config, right, role="sham",
                override_step=step, override_action=candidate.executed_actions[step],
                expected_prefix=candidate)
            verify_sham(candidate, sham)
            sham_entry = _save_case(output / "cases" / f"{index:04d}" / "sham", sham, after, "sham")
            del sham
            phase = "intervention"
            progress(phase, case_index)
            intervention = capture_frozen_case(agent, after.config, right, role="intervention",
                override_step=step, override_action=baseline.executed_actions[step],
                expected_prefix=candidate)
            intervention_entry = _save_case(output / "cases" / f"{index:04d}" / "intervention",
                                             intervention, after, "intervention")
            report = {"episode": episode_descriptor(left), "comparison": comparison,
                "sham_verified": True, "single_override_index": step,
                "roles": {"before": entries[index], "candidate": candidate_entry,
                          "sham": sham_entry, "intervention": intervention_entry}}
            atomic_json(output / "cases" / f"{index:04d}" / "audit.json", report)
            completed.append(report)
            del baseline, candidate, intervention
        after_hashes = _assert_frozen(agent, reward, after)
        phase_hashes.append({"phase": "after", "hashes_before": {
            "actor": after.manifest["agent_sha256"], "reward": after.manifest["reward_sha256"]},
            "hashes_after": after_hashes, "unchanged": True})
        before.assert_unchanged()
        after.assert_unchanged()
        if provenance()["source_and_lock_sha256"] != initial_provenance["source_and_lock_sha256"]:
            raise RuntimeError("paired diagnostic source code changed during audit")
        aggregate = {"version": VERSION, "protocol_sha256": protocol_sha,
            "interpretation": metadata["interpretation"], "phase_hashes": phase_hashes,
            "cases": completed, **_aggregate(completed)}
        atomic_json(output / "aggregate.json", aggregate)
        atomic_json(output / "status.json", {"status": "completed", "training_updates": 0,
            "optimizer_steps": 0, "case_count": len(completed), "phase_hashes": phase_hashes,
            "source_pins_unchanged": True, "seconds": time.monotonic() - started})
    except BaseException as error:
        (output / "aggregate.json").unlink(missing_ok=True)
        atomic_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error_type": type(error).__name__, "error": str(error),
            "completed_cases": len(completed), "phase_hashes": phase_hashes,
            "phase": phase, "case_index": case_index})
        raise
    finally:
        agent = reward = None
        gc.collect()
        torch.set_num_threads(threads)
    print(json.dumps({"output": str(output), "status": "completed", "cases": len(completed)}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
