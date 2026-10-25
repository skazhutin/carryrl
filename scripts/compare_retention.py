"""Compare completed, preregistered anchor variants; never launch training."""

import argparse
import hashlib
import json
import time
from contextlib import ExitStack
from pathlib import Path

from carryrl.evaluation.retention import (
    memory_retention_outcomes,
    retention_outcomes,
    validate_retention_run,
)
from carryrl.training.checkpoint_storage import checkpoint_read_lock
from carryrl.training.retention import load_retention_plan
from carryrl.training.run import atomic_json, file_sha256, provenance

MAX_JSON_BYTES = 64 * 1024**2


def _read(path, pins):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("invalid bounded retention JSON source")
    data = path.read_bytes()
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("retention JSON byte budget exceeded")
    pins[str(path.resolve())] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def _load_run(directory, plan, pins):
    with checkpoint_read_lock(directory / "checkpoints" / "rollout_0000"):
        return _load_run_locked(directory, plan, pins)


def _load_run_locked(directory, plan, pins):
    metadata, summary, config, status = [_read(directory / name, pins) for name in (
        "metadata.json", "summary.json", "config.json", "status.json")]
    checkpoint = directory / "checkpoints" / "rollout_0000"
    manifest = _read(checkpoint / "manifest.json", pins)
    archive_sha = file_sha256(checkpoint / "state.pt")
    if archive_sha != manifest["sha256"]:
        raise ValueError("retention checkpoint archive SHA256 mismatch")
    pins[str((checkpoint / "state.pt").resolve())] = archive_sha
    if (status.get("status") != "completed" or status.get("phase") != "finished"
            or status.get("optimizer_steps") != plan["expected_optimizer_updates"]
            or Path(summary["checkpoint"]).resolve() != checkpoint.resolve()
            or Path(status["checkpoint"]).resolve() != checkpoint.resolve()
            or config != metadata.get("config")):
        raise ValueError("retention run is incomplete or its artifact identities differ")
    evaluation_path = directory / "evaluations" / f"epoch_{plan['expected_epochs']:04d}_final.json"
    evaluation = _read(evaluation_path, pins)
    files = [directory / name for name in (
        "metadata.json", "summary.json", "config.json", "status.json")]
    files.extend([evaluation_path, *[path for path in checkpoint.rglob("*") if path.is_file()]])
    return metadata, summary, manifest, evaluation, files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--baseline-evaluation-sha256", required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--anchor-run", type=Path, required=True)
    parser.add_argument("--baseline-memory", type=Path)
    parser.add_argument("--control-memory", type=Path)
    parser.add_argument("--anchor-memory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    optional = [args.baseline_memory, args.control_memory, args.anchor_memory]
    if any(optional) and not all(optional):
        parser.error("memory comparison requires all three diagnostic artifacts")
    output = args.output.resolve()
    paths = [args.plan, args.baseline_evaluation, args.control_run, args.anchor_run,
             *[path for path in optional if path is not None], args.output]
    if any(part.is_symlink() for path in paths for part in (path.absolute(),
                                                          *path.absolute().parents)):
        raise ValueError("retention comparison paths must not contain symlinks")
    sources = [args.plan.resolve(strict=True).parent,
               args.baseline_evaluation.resolve(strict=True).parent,
               args.control_run.resolve(strict=True), args.anchor_run.resolve(strict=True),
               *[path.resolve(strict=True).parent for path in optional if path is not None]]
    if any(output == path or path in output.parents or output in path.parents for path in sources):
        raise ValueError("retention comparison output must be outside source directories")
    if output.exists() or output.is_symlink():
        raise FileExistsError("retention comparison output already exists")
    output.mkdir(parents=True)
    started = time.monotonic()
    try:
        with ExitStack() as readers:
            return _compare(args, output, started, optional, readers)
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "interrupted"
            if isinstance(error, KeyboardInterrupt) else "failed", "updates": 0,
            "error_type": type(error).__name__, "error": str(error)})
        raise


def _compare(args, output, started, optional, readers):
    atomic_json(output / "status.json", {"status": "validating", "updates": 0})
    plan = load_retention_plan(args.plan, expected_sha256=args.plan_sha256)
    before = {str(args.plan.resolve()): args.plan_sha256}
    baseline = _read(args.baseline_evaluation, before)
    if before[str(args.baseline_evaluation.resolve())] != args.baseline_evaluation_sha256:
        raise ValueError("baseline evaluation SHA256 mismatch")
    if (baseline.get("agent_sha256") != plan["source_checkpoint"]["agent_sha256"]
            or baseline.get("reward_sha256") != plan["source_checkpoint"]["reward_sha256"]
            or baseline.get("protocol_sha256") != plan["evaluation_protocol_sha256"]
            or baseline.get("weights_revision") != plan["source_checkpoint"][
                "weights_revision"]):
        raise ValueError("baseline evaluation does not identify the frozen initial checkpoint")
    directories = [directory.resolve() for directory in (args.control_run, args.anchor_run)]
    for directory in sorted(set(directories)):
        readers.enter_context(checkpoint_read_lock(
            directory / "checkpoints" / "rollout_0000"))
    runs = [_load_run_locked(directory, plan, before) for directory in directories]
    files = [args.plan.resolve(), args.baseline_evaluation.resolve(),
             *[path for run in runs for path in run[-1]],
             *[path.resolve() for path in optional if path is not None]]
    for path in files:
        if path.is_symlink():
            raise ValueError("retention checkpoint source must not contain symlinks")
        before.setdefault(str(path.resolve()), file_sha256(path))
    for arm, run in zip(("anchor_off", "anchor_on"), runs, strict=True):
        validate_retention_run(plan, arm, *run[:4])
    report = retention_outcomes(baseline, runs[0][3], runs[1][3])
    memory = [_read(path, before) for path in optional] if all(optional) else [None] * 3
    for artifact, evaluation in zip(memory, (baseline, runs[0][3], runs[1][3]), strict=True):
        if artifact is not None and any(artifact.get(key) != evaluation.get(key)
                for key in ("agent_sha256", "reward_sha256", "weights_revision")):
            raise ValueError("memory diagnostic differs from the selected policy checkpoint")
    report["memory_controls"] = memory_retention_outcomes(*memory)
    after = {path: file_sha256(Path(path)) for path in before}
    if before != after:
        raise RuntimeError("retention comparison inputs changed while reading")
    report.update(controlled_design_verified=True, plan_sha256=args.plan_sha256,
        measured_optimizer_updates=plan["expected_optimizer_updates"],
        measured_epochs=plan["expected_epochs"], updates=0,
        selection="fixed_final_epoch", interpretation=(
            "Descriptive single training-seed pair; no automatic winner, "
            "calibration, statistical robustness or PPO utility is established."))
    atomic_json(output / "comparison.json", report)
    atomic_json(output / "metadata.json", {**provenance(), "command": "compare_retention",
        "input_files_sha256": before, "updates": 0, "source_files_unchanged": True})
    atomic_json(output / "status.json", {"status": "completed", "updates": 0,
        "controlled_design_verified": True, "seconds": time.monotonic() - started})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
