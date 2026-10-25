"""Frozen gameplay and held-out teacher likelihood; never updates model weights."""

import argparse
import gc
import hashlib
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from carryrl.dashboard.frozen_playback import write_frozen_playback
from carryrl.evaluation.imitation import (
    evaluate_imitation_validation,
    imitation_validation_protocol,
)
from carryrl.evaluation.paired import episode_descriptor, paired_outcomes
from carryrl.evaluation.trajectory_diagnostics import diagnose_frozen_trajectory
from carryrl.training.run import atomic_json, file_sha256, provenance
from carryrl.training.sequence import (
    evaluate_sequence,
    load_sequence_checkpoint,
    sequence_evaluation_protocol,
    tensor_hash,
)
from carryrl.training.sequence_config import SequenceRunConfig


def _checkpoint_hashes(checkpoint):
    return {str(path.relative_to(checkpoint)): file_sha256(path)
            for path in sorted(checkpoint.rglob("*")) if path.is_file()}


def _prepare_output(output, checkpoint):
    """Never overwrite measurements or put diagnostic output inside source artifacts."""
    output, checkpoint = output.resolve(), checkpoint.resolve(strict=True)
    source_runs = [parent.parent for parent in checkpoint.parents
                   if parent.name in {"checkpoints", "checkpoints_by_epoch"}]
    source_run = source_runs[-1] if source_runs else checkpoint
    if output == source_run or source_run in output.parents or output in checkpoint.parents:
        raise ValueError("diagnostic output must be separate from the source run/checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _paired_report(before, after):
    """Require the declared cohort and actual ordered episode identities to agree."""
    for field in ("split", "protocol_sha256", "suite_version", "scoring_version",
                  "reward_sha256"):
        if before.get(field) is None or before.get(field) != after.get(field):
            raise ValueError(f"paired evaluation differs or lacks {field}")
    if (before.get("policy_and_reward_frozen") is not True
            or after.get("policy_and_reward_frozen") is not True):
        raise ValueError("paired evaluation requires frozen measurements")
    selected = [[row for row in result["records"]
                 if row.get("policy") == "frozen_deterministic"] for result in (before, after)]
    return paired_outcomes(*selected)


def _capture_report(scope, selected, *, enabled):
    return {"requested_scope": scope, "enabled": enabled,
            "policy": "frozen_deterministic", "captured_count": len(selected),
            "outcomes": {category: sum(row["outcome_category"] == category for row in selected)
                         for category in ("success", "failure", "unfinished", "unknown")},
            "selection": "failed includes every non-success, including unknown outcomes",
            "selected_episodes": selected}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--all-levels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-gameplay", action="store_true")
    parser.add_argument("--capture-scope", choices=("first_per_task", "failed", "all"),
                        default="first_per_task",
                        help="deterministic policy replay: first per task, every non-success, "
                             "or every episode; failed/all require eval_stochastic=false")
    parser.add_argument("--validation-nll", action="store_true")
    parser.add_argument("--nll-episodes-per-task", type=int, default=2)
    parser.add_argument("--compare", type=Path, nargs="*", default=[])
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--expected-validation-protocol-sha256")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.nll_episodes_per_task < 1:
        raise ValueError("NLL episodes per task must be positive")
    if args.skip_gameplay and (not args.validation_nll or args.compare):
        raise ValueError("skip-gameplay requires validation NLL and cannot compare game wins")
    if args.skip_gameplay and args.expected_protocol_sha256:
        raise ValueError("gameplay protocol guard requires gameplay; use validation protocol guard")
    if args.expected_validation_protocol_sha256 and not args.validation_nll:
        raise ValueError("validation protocol guard requires --validation-nll")
    config = replace(SequenceRunConfig.load(args.config), device=args.device,
                     eval_all_levels=args.all_levels)
    if args.capture_scope != "first_per_task" and config.eval_stochastic:
        raise ValueError("failed/all playback capture requires eval_stochastic=False")
    protocol = sequence_evaluation_protocol(config)
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if args.expected_protocol_sha256 and args.expected_protocol_sha256 != protocol_hash:
        raise ValueError("requested evaluation protocol does not match the expected SHA256")
    nll_config = replace(config, eval_all_levels=False,
                         eval_episodes_per_task=args.nll_episodes_per_task)
    nll_protocol = imitation_validation_protocol(nll_config) if args.validation_nll else None
    nll_protocol_hash = (hashlib.sha256(json.dumps(nll_protocol, sort_keys=True).encode())
                         .hexdigest() if nll_protocol is not None else None)
    if (args.expected_validation_protocol_sha256
            and args.expected_validation_protocol_sha256 != nll_protocol_hash):
        raise ValueError("requested validation protocol does not match the expected SHA256")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no implicit device fallback")
    checkpoint = args.checkpoint.resolve(strict=True)
    before = _checkpoint_hashes(checkpoint)
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    references = [(path.resolve(), json.loads(path.read_text())) for path in args.compare]
    output = _prepare_output(args.output, checkpoint)
    selected_captures = []
    metadata = {"kind": "frozen_imitation_diagnostics", "command": "evaluate", "updates": 0,
                "checkpoint": str(checkpoint), "checkpoint_files_sha256": before,
                "config": (nll_config if args.skip_gameplay else config).to_dict(),
                "protocol": None if args.skip_gameplay else protocol,
                "protocol_sha256": None if args.skip_gameplay else protocol_hash,
                "validation_likelihood_protocol": nll_protocol,
                "validation_likelihood_protocol_sha256": nll_protocol_hash,
                "split": "validation",
                "playback_capture": _capture_report(args.capture_scope, selected_captures,
                                                    enabled=not args.skip_gameplay),
                "created_utc": datetime.now(UTC).isoformat(), **provenance()}
    atomic_json(output / "metadata.json", metadata)
    threads, started = torch.get_num_threads(), time.monotonic()

    def progress(event):
        if event["event"] != "step":
            print(json.dumps(event), flush=True)
        atomic_json(output / "status.json", {"status": "running", **event})

    try:
        torch.set_num_threads(config.torch_threads)
        agent, optimizer, reward, state = load_sequence_checkpoint(
            checkpoint, config, evaluation_only=True, load_optimizer=False)
        revision = state["weights_revision"]
        del optimizer, state
        gc.collect()
        agent_hash, reward_hash = tensor_hash(agent), tensor_hash(reward)
        evaluation_path = output / "evaluations" / (
            "frozen_full.json" if config.eval_all_levels else "frozen_quick.json")

        def playback(record, frames, actions):
            # Authenticate every RGB/action transition before publishing its playback.
            diagnostic = diagnose_frozen_trajectory(config, record, frames, actions)
            episode_label = f"{record['task']}_episode_{record['episode_index']:04d}"
            capture_root = (output if args.capture_scope == "first_per_task" else
                            output / "captured_episodes" / episode_label)
            capture_path = write_frozen_playback(
                capture_root, {**record, "split": "validation"}, frames, actions,
                epoch=manifest.get("epoch", 0), agent_sha256=agent_hash,
                weights_revision=revision, checkpoint=checkpoint, manifest=manifest,
                evaluation_reference=evaluation_path)
            diagnostic.update(agent_sha256=agent_hash, weights_revision=revision,
                              playback=str(capture_path.resolve()))
            diagnostic_name = (capture_path.name if args.capture_scope == "first_per_task"
                               else f"{episode_label}.json")
            diagnostic_path = output / "trajectory_diagnostics" / diagnostic_name
            atomic_json(diagnostic_path, diagnostic)
            outcome = record["outcome"]
            category = ("unknown" if not outcome.get("known", False) else
                        "success" if outcome.get("success") is True else
                        "failure" if outcome.get("failure") is True else "unfinished")
            selected_captures.append({
                "episode": {**episode_descriptor(record), "split": "validation",
                            "level_bank_seed": record.get(
                                "level_bank_seed", config.level_bank_seed)},
                "outcome_category": category, "outcome": outcome,
                "playback": str(capture_path.resolve()),
                "diagnostic": str(diagnostic_path.resolve())})
            metadata["playback_capture"] = _capture_report(
                args.capture_scope, selected_captures, enabled=True)
            atomic_json(output / "metadata.json", metadata)

        if not args.skip_gameplay:
            evaluated = evaluate_sequence(agent, reward, config,
                                          on_progress=progress, on_playback=playback,
                                          capture_scope=args.capture_scope)
            evaluated.update(playback_capture=metadata["playback_capture"],
                             checkpoint_weights_revision=revision, weights_revision=revision,
                             agent_sha256=agent_hash, reward_sha256=reward_hash)
            atomic_json(evaluation_path, evaluated)
            # The generic CLI location also makes this a normal evaluation-only run.
            atomic_json(output / "evaluation.json", evaluated)
            for index, (path, reference) in enumerate(references):
                paired = _paired_report(reference, evaluated)
                paired.update(before={"evaluation": str(path), "sha256": file_sha256(path),
                                      "agent_sha256": reference.get("agent_sha256")},
                              after={"evaluation": str(evaluation_path.resolve()),
                                     "agent_sha256": agent_hash})
                atomic_json(output / "comparisons" / f"reference_{index:02d}.json", paired)
        if args.validation_nll:
            report = evaluate_imitation_validation(
                agent, nll_config, reward=reward, on_progress=progress,
                checkpoint_weights_revision=revision)
            report["checkpoint"] = str(checkpoint)
            atomic_json(output / "validation_nll.json", report)
        if tensor_hash(agent) != agent_hash or tensor_hash(reward) != reward_hash:
            raise RuntimeError("diagnostics modified policy or reward tensors")
        if _checkpoint_hashes(checkpoint) != before:
            raise RuntimeError("diagnostics modified checkpoint files")
        atomic_json(output / "status.json", {"status": "completed", "updates": 0,
                    "weights_revision": revision, "seconds": time.monotonic() - started,
                    "checkpoint_files_unchanged": True, "policy_and_reward_frozen": True,
                    "playback_capture": metadata["playback_capture"]})
    except BaseException as error:
        atomic_json(output / "status.json", {
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error_type": type(error).__name__, "error": str(error),
            "playback_capture": metadata["playback_capture"]})
        raise
    finally:
        torch.set_num_threads(threads)
    print(json.dumps({"output": str(output), "status": "completed", "updates": 0}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
