import json

import pytest
import torch

from carryrl.benchmark import ProbeConfig, run_probe, select_device, write_report


def test_real_cpu_update_is_finite_and_changes_parameters():
    result = run_probe(
        "cpu", ProbeConfig(batch_size=2, image_size=16, steps=2, warmup_steps=1, cpu_threads=1)
    )
    assert result["losses_finite"]
    assert result["final_gradients_finite"]
    assert result["parameters_finite"]
    assert result["parameters_changed"]
    assert result["updates_per_second"] > 0
    assert result["mps_memory_after_update"] is None


def test_explicit_unavailable_mps_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS was requested"):
        select_device("mps")


@pytest.mark.parametrize(
    "configuration",
    [{"steps": 0}, {"batch_size": 257}, {"image_size": 8}, {"warmup_steps": 0}, {"seed": -1}],
)
def test_probe_configuration_rejects_unbounded_or_invalid_work(configuration):
    with pytest.raises(ValueError):
        ProbeConfig(**configuration)


def test_report_is_valid_json_and_preserves_existing_output(tmp_path):
    path = tmp_path / "nested" / "probe.json"
    write_report(path, {"status": "passed"})
    with pytest.raises(FileExistsError):
        write_report(path, {"status": "replacement"})
    assert json.loads(path.read_text()) == {"status": "passed"}
