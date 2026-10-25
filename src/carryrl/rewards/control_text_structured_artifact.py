"""Authenticated resources for the separately versioned structured text reward."""

import json
from pathlib import Path

import torch

from carryrl.agent.language import TEXT_ID, TEXT_REVISION, MultilingualInstructionEncoder
from carryrl.rewards.control_text import (
    CANDIDATES,
    _sha_file,
    calibration_gates,
    content_hash,
    encoder_state_hash,
)


def _runtime_gate(runtime):
    rows = runtime.get("results", [])
    expected_slices = {"supported", "unsupported", "unknown", "held", "press", "aliases"}
    valid = True
    for row in rows:
        counts = row.get("counts", {})
        valid &= set(counts) == expected_slices
        for name, count in counts.items():
            total, correct = count.get("total"), count.get("correct")
            if (type(total) is not int or type(correct) is not int
                    or not 0 <= correct <= total):
                return False
            valid &= not total or correct / total >= (1 if name == "unknown" else 0.8)
    pairs = {(row.get("task"), row.get("language")) for row in rows}
    tasks = {task for task, _ in pairs}
    return bool(
        valid and len(tasks) == 16 and pairs == {(task, lang) for task in tasks
                                                 for lang in ("ru", "en")}
        and runtime.get("passed") is True
        and runtime.get("records") == runtime.get("passed_records") == 32
        and len(rows) == 32
        and all(row.get("passed") is True and not row.get("errors")
                for row in rows)
    )


def _admission(validation, runtime, sealed, training):
    gates = calibration_gates(validation)
    sealed_gates = calibration_gates(sealed) if sealed is not None else None
    runtime_ok = _runtime_gate(runtime)
    ready = bool(
        training.get("trained") is True
        and training.get("selected_development_feasible") is True
        and gates["passed"] and runtime_ok
        and sealed_gates and sealed_gates["passed"]
        and training.get("sealed_test_used") is True
    )
    return {"calibration": gates, "runtime_passed": runtime_ok,
            "sealed": sealed_gates, "passed": ready}


def save_structured_control_artifact(
    directory, *, encoder, head, calibration, validation, runtime_validation,
    sealed_validation, training,
):
    """Preserve successful and rejected fits; failed fits remain unloadable for RL."""
    from carryrl.rewards.control_text_structured import (
        STRUCTURED_HEAD_ARCHITECTURE,
        STRUCTURED_MODEL_VERSION,
        checked_structured_calibration,
    )

    calibration = checked_structured_calibration(calibration)
    if head.architecture != STRUCTURED_HEAD_ARCHITECTURE:
        raise ValueError("structured artifact requires the registered architecture")
    if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
        raise ValueError("non-finite structured text reward weights")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoder.save_resources(directory / "language")
    torch.save(encoder.state_dict(), directory / "encoder.pt")
    torch.save(head.state_dict(), directory / "head.pt")
    reports = {"calibration.json": calibration, "validation.json": validation,
               "runtime_validation.json": runtime_validation,
               "sealed_validation.json": sealed_validation}
    for name, value in reports.items():
        (directory / name).write_text(json.dumps(value, sort_keys=True) + "\n")
    admission = _admission(validation, runtime_validation, sealed_validation, training)
    manifest = {
        "version": STRUCTURED_MODEL_VERSION,
        "head_architecture": STRUCTURED_HEAD_ARCHITECTURE,
        "encoder_id": TEXT_ID, "revision": TEXT_REVISION,
        "encoder_state_hash": encoder_state_hash(encoder.state_dict()),
        "candidates": list(CANDIDATES), "training": training,
        "admission": admission, "ready": admission["passed"],
        "shared_encoder": False,
        "files": {str(path.relative_to(directory)): _sha_file(path)
                  for path in sorted(directory.rglob("*")) if path.is_file()},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest


def load_structured_control_reward(directory, *, shared_encoder=None):
    """Load only a hashed artifact that still satisfies all recorded admission gates."""
    from carryrl.rewards.control_text_structured import (
        STRUCTURED_HEAD_ARCHITECTURE,
        STRUCTURED_MODEL_VERSION,
        JointStructuredControlHead,
        StructuredTextControlReward,
    )

    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("version") != STRUCTURED_MODEL_VERSION
        or manifest.get("head_architecture") != STRUCTURED_HEAD_ARCHITECTURE
        or manifest.get("encoder_id") != TEXT_ID
        or manifest.get("revision") != TEXT_REVISION
        or manifest.get("candidates") != list(CANDIDATES)
    ):
        raise ValueError("unsupported structured reward identity")
    files = manifest.get("files", {})
    required = {"head.pt", "calibration.json", "validation.json",
                "runtime_validation.json", "sealed_validation.json"}
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file()}
    if not required <= files.keys() or actual != set(files) | {"manifest.json"}:
        raise ValueError("structured reward contains missing or unhashed resources")
    for name, expected in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or _sha_file(path) != expected:
            raise ValueError(f"structured reward resource hash mismatch: {name}")
    reports = {name: json.loads((directory / name).read_text()) for name in required
               if name.endswith(".json")}
    admission = _admission(
        reports["validation.json"], reports["runtime_validation.json"],
        reports["sealed_validation.json"], manifest.get("training", {}),
    )
    if (manifest.get("ready") is not True or not admission["passed"]
            or manifest.get("admission") != admission):
        raise ValueError("structured reward admission failed")
    if shared_encoder is None:
        if "encoder.pt" not in files:
            raise ValueError("structured reward requires its shared E5 encoder")
        state = torch.load(directory / "encoder.pt", map_location="cpu", weights_only=True)
        if encoder_state_hash(state) != manifest.get("encoder_state_hash"):
            raise ValueError("structured reward E5 hash mismatch")
        encoder = MultilingualInstructionEncoder.from_resources(directory / "language")
        encoder.load_state_dict(state, strict=True)
    else:
        encoder = shared_encoder
        if (getattr(encoder, "_awaiting_checkpoint_state", False)
                or encoder.max_tokens != 512
                or encoder_state_hash(encoder.state_dict()) != manifest.get("encoder_state_hash")):
            raise ValueError("structured reward shared E5 identity mismatch")
    encoder.eval().requires_grad_(False)
    head = JointStructuredControlHead()
    head.load_state_dict(torch.load(directory / "head.pt", map_location="cpu", weights_only=True))
    if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
        raise ValueError("non-finite structured text reward weights")
    model = StructuredTextControlReward(
        encoder, head,
        model_hash=content_hash({"version": STRUCTURED_MODEL_VERSION,
                                 "head": files["head.pt"],
                                 "encoder": manifest["encoder_state_hash"]}),
        calibration=reports["calibration.json"],
        calibration_hash=files["calibration.json"], _verified=True,
    )
    model._manifest = manifest
    model._resources = {name: (directory / name).read_bytes() for name in files
                        if name != "encoder.pt"}
    return model


def load_compatible_text_control_reward(directory, *, shared_encoder=None):
    """Preserve historical loaders and select the new one by explicit bundle version."""
    from carryrl.rewards.control_text import MODEL_VERSION, load_text_control_reward
    from carryrl.rewards.control_text_structured import STRUCTURED_MODEL_VERSION

    version = json.loads((Path(directory) / "manifest.json").read_text()).get("version")
    if version == STRUCTURED_MODEL_VERSION:
        return load_structured_control_reward(directory, shared_encoder=shared_encoder)
    if version == MODEL_VERSION:
        return load_text_control_reward(directory, shared_encoder=shared_encoder)
    raise ValueError("unsupported text reward artifact version")
