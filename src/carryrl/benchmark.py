"""Bounded synthetic update probe. Its throughput is not game-learning throughput."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class ProbeConfig:
    batch_size: int = 32
    image_size: int = 64
    steps: int = 30
    warmup_steps: int = 5
    cpu_threads: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        limits = {
            "batch_size": (1, 256),
            "image_size": (16, 128),
            "steps": (1, 200),
            "warmup_steps": (1, 20),
            "cpu_threads": (1, 64),
        }
        for name, (lower, upper) in limits.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an integer between 0 and 2**32 - 1")


def select_device(name: str) -> torch.device:
    """Validate an explicit device without switching to another backend."""
    if name not in {"cpu", "mps"}:
        raise ValueError("Probe supports explicit 'cpu' or 'mps' devices")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable in this Python/PyTorch environment"
        )
    return torch.device(name)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _system_command(arguments: list[str]) -> dict[str, str | None]:
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=3, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        return {"value": None, "unavailable_reason": str(error)}
    return {"value": result.stdout.strip(), "unavailable_reason": None}


def _process_peak_rss_bytes() -> int:
    # macOS reports bytes; Linux reports KiB. This is the process lifetime peak,
    # not the isolated memory allocation of a probe or the whole system.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _mps_memory(device: torch.device) -> dict[str, int | str] | None:
    if device.type != "mps":
        return None
    return {
        "current_tensor_bytes": torch.mps.current_allocated_memory(),
        "current_driver_bytes": torch.mps.driver_allocated_memory(),
        "interpretation": "driver includes tensor bytes; do not add these; snapshot, not peak",
    }


def environment_report() -> dict[str, object]:
    packages = {}
    for name in ("torch", "numpy", "gymnasium", "stable-baselines3", "pillow", "pygame"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    fallback_setting = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    result: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback_env": fallback_setting,
        "mps_fallback_enabled": fallback_setting == "1",
        "mps_fallback_observed": "not instrumented; backend fallback setting is recorded",
        "packages": packages,
    }
    if sys.platform == "darwin":
        result["hardware_model"] = _system_command(["sysctl", "-n", "hw.model"])
        result["physical_memory_bytes"] = _system_command(["sysctl", "-n", "hw.memsize"])
        result["swap_before"] = _system_command(["sysctl", "-n", "vm.swapusage"])
    return result


def run_probe(device_name: str, config: ProbeConfig) -> dict[str, object]:
    """Time fixed, in-memory CNN updates after warmup, then validate the numerics."""
    device = select_device(device_name)
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 5),
    ).to(device)
    # Construct identical batches on CPU for every backend. Data transfer is
    # outside this update-only timer and is disclosed in the report.
    images = torch.rand(config.batch_size, 3, config.image_size, config.image_size).to(device)
    targets = torch.randint(5, (config.batch_size,)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def update() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), targets)
        loss.backward()
        optimizer.step()
        return loss.detach()

    for _ in range(config.warmup_steps):
        update()
    _synchronize(device)
    initial_parameters = [parameter.detach().cpu().clone() for parameter in model.parameters()]
    _synchronize(device)
    started = time.perf_counter()
    losses = [update() for _ in range(config.steps)]
    _synchronize(device)
    elapsed = time.perf_counter() - started
    memory = _mps_memory(device)
    losses_finite = bool(torch.isfinite(torch.stack(losses)).all().item())
    gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    parameters_finite = all(
        bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()
    )
    parameters_changed = any(
        not torch.equal(initial, parameter.detach().cpu())
        for initial, parameter in zip(initial_parameters, model.parameters(), strict=True)
    )
    if not (losses_finite and gradients_finite and parameters_finite and parameters_changed):
        raise RuntimeError(
            f"Synthetic update failed on {device_name}: finite loss={losses_finite}, "
            f"gradient={gradients_finite}, parameters={parameters_finite}, "
            f"parameters_changed={parameters_changed}"
        )
    return {
        "device": device.type,
        "config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed,
        "updates_per_second": config.steps / elapsed,
        "synthetic_images_per_second": config.steps * config.batch_size / elapsed,
        "first_measured_loss": float(losses[0].cpu()),
        "last_measured_loss": float(losses[-1].cpu()),
        "losses_finite": losses_finite,
        "final_gradients_finite": gradients_finite,
        "parameters_finite": parameters_finite,
        "parameters_changed": parameters_changed,
        "mps_memory_after_update": memory,
        "process_lifetime_peak_rss_bytes": _process_peak_rss_bytes(),
        "timing_scope": "fixed in-memory batches; forward/backward/optimizer; no game or transfer",
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    """Use exclusive creation so a probe never silently replaces an earlier result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "mps"), default=["cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        config = ProbeConfig(
            batch_size=arguments.batch_size,
            image_size=arguments.image_size,
            steps=arguments.steps,
            warmup_steps=arguments.warmup_steps,
            cpu_threads=arguments.cpu_threads,
            seed=arguments.seed,
        )
        devices = list(dict.fromkeys(arguments.devices))
        for device in devices:
            select_device(device)
        torch.set_num_threads(config.cpu_threads)
        started_at = datetime.now(UTC)
        report: dict[str, object] = {
            "schema_version": 1,
            "kind": "synthetic_compute_probe",
            "started_at": started_at.isoformat(),
            "environment": environment_report(),
            "results": [run_probe(device, config) for device in devices],
            "limitation": "synthetic throughput does not measure learning quality or full RL loop",
        }
        if sys.platform == "darwin":
            report["swap_after"] = _system_command(["sysctl", "-n", "vm.swapusage"])
        output = arguments.output or Path("runs") / (
            f"compute-probe-{started_at.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        )
        write_report(output, report)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"compute probe failed: {error}\n")
    print(json.dumps({"report": str(output), "results": report["results"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
