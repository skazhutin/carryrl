"""Compare immutable run artifacts; no models, optimizers or training are loaded."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("retention_comparison_cli",
        Path(__file__).parents[1] / "scripts/compare_retention.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "runs"
    plan_path = source / "plan" / "manifest.json"
    _write(plan_path, {"fixture": True})
    baseline = source / "baseline.json"
    base = {"agent_sha256": "a" * 64, "reward_sha256": "b" * 64,
            "protocol_sha256": "c" * 64, "weights_revision": 4}
    _write(baseline, base)
    plan = {"expected_optimizer_updates": 2, "expected_epochs": 1,
            "source_checkpoint": base, "evaluation_protocol_sha256": "c" * 64}
    runs = [source / name for name in ("control", "anchor")]
    for index, directory in enumerate(runs):
        checkpoint = directory / "checkpoints" / "rollout_0000"
        checkpoint.mkdir(parents=True)
        archive = f"fixture weights {index}".encode()
        (checkpoint / "state.pt").write_bytes(archive)
        _write(checkpoint / "manifest.json", {"sha256": hashlib.sha256(archive).hexdigest()})
        _write(directory / "metadata.json", {"config": {"seed": 7}})
        _write(directory / "config.json", {"seed": 7})
        _write(directory / "summary.json", {"checkpoint": str(checkpoint)})
        _write(directory / "status.json", {"status": "completed", "phase": "finished",
            "optimizer_steps": 2, "checkpoint": str(checkpoint)})
        _write(directory / "evaluations" / "epoch_0001_final.json", {
            **base, "agent_sha256": str(index) * 64, "weights_revision": 6})
    calls = []
    monkeypatch.setattr(module, "provenance", lambda: {"fixture": True})
    monkeypatch.setattr(module, "load_retention_plan", lambda path, **kwargs: plan)
    monkeypatch.setattr(module, "validate_retention_run", lambda *args: calls.append(args[1]))
    monkeypatch.setattr(module, "retention_outcomes", lambda *args: {"available": True})
    monkeypatch.setattr(module, "memory_retention_outcomes", lambda *args: {"available": False})
    output = tmp_path / "report"
    args = ["--plan", str(plan_path), "--plan-sha256", hashlib.sha256(
                plan_path.read_bytes()).hexdigest(),
            "--baseline-evaluation", str(baseline), "--baseline-evaluation-sha256",
            hashlib.sha256(baseline.read_bytes()).hexdigest(),
            "--control-run", str(runs[0]), "--anchor-run", str(runs[1]), "--output", str(output)]
    return module, args, output, runs, baseline, calls


def test_completed_comparison_validates_both_arms_and_preserves_sources(fixture):
    module, args, output, runs, baseline, calls = fixture
    before = baseline.read_bytes()
    assert module.main(args) == 0
    report = json.loads((output / "comparison.json").read_text())
    assert calls == ["anchor_off", "anchor_on"]
    assert report["controlled_design_verified"] and report["updates"] == 0
    assert report["measured_optimizer_updates"] == 2 and report["selection"] == "fixed_final_epoch"
    assert report["memory_controls"]["available"] is False
    assert baseline.read_bytes() == before
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        module.main(args)


@pytest.mark.parametrize("problem", ["status", "archive", "config", "baseline_pin"])
def test_incomplete_or_changed_inputs_cannot_be_reported_as_comparable(fixture, problem):
    module, args, output, runs, baseline, calls = fixture
    if problem == "status":
        path = runs[0] / "status.json"
        data = json.loads(path.read_text())
        data["optimizer_steps"] = 1
        _write(path, data)
    elif problem == "archive":
        (runs[0] / "checkpoints" / "rollout_0000" / "state.pt").write_bytes(b"changed")
    elif problem == "config":
        _write(runs[0] / "config.json", {"seed": 8})
    else:
        baseline.write_bytes(b"{}")
    with pytest.raises(ValueError):
        module.main(args)
    assert not (output / "comparison.json").exists()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_mid_comparison_mutation_is_detected(fixture, monkeypatch):
    module, args, output, runs, baseline, calls = fixture

    def mutate(*args):
        baseline.write_text("{}")
        return {"available": True}

    monkeypatch.setattr(module, "retention_outcomes", mutate)
    with pytest.raises(RuntimeError, match="inputs changed"):
        module.main(args)
    assert not (output / "comparison.json").exists()


def test_partial_memory_artifacts_are_not_silently_dropped(fixture):
    module, args, output, runs, baseline, calls = fixture
    with pytest.raises(SystemExit):
        module.main([*args, "--baseline-memory", str(baseline)])
    assert not output.exists()


def test_output_cannot_be_inside_source_tree(fixture):
    module, args, output, runs, baseline, calls = fixture
    args[args.index("--output") + 1] = str(runs[0] / "comparison")
    with pytest.raises(ValueError, match="outside source"):
        module.main(args)
    assert calls == [] and not output.exists()


def test_comparison_restores_cold_checkpoints_before_pinning_sources(fixture, monkeypatch):
    module, args, output, runs, _, _ = fixture
    monkeypatch.setattr("carryrl.training.checkpoint_storage.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=100 * 1024**3))
    originals = {}
    for run in runs:
        state = run / "checkpoints/rollout_0000/state.pt"
        originals[str(state.resolve())] = hashlib.sha256(state.read_bytes()).hexdigest()
        state.with_suffix(".pt.gz").write_bytes(gzip.compress(state.read_bytes()))
        state.unlink()
    assert module.main(args) == 0
    pins = json.loads((output / "metadata.json").read_text())["input_files_sha256"]
    assert all(pins[path] == digest for path, digest in originals.items())
    assert not any(path.endswith(".gz") for path in pins)
    assert all(Path(path).is_file() for path in originals)
