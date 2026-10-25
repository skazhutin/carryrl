"""Versioned, authenticated factorized reward bundles and strict admission."""

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
from carryrl.rewards.control_text_factorized import (
    FACTORIZED_HEAD_ARCHITECTURE,
    FACTORIZED_MODEL_VERSION,
    FACTORIZED_POPULATION_CONTRACT,
    FactorizedControlHead,
    FactorizedTextControlReward,
    checked_factorized_calibration,
)


def _runtime_gate(runtime):
    expected = {
        "supported",
        "unsupported",
        "unknown",
        "held",
        "press",
        "relative",
        "aliases",
        "different_pairs",
        "supported_payable",
    }
    rows = runtime.get("results", [])
    pairs = {(row.get("task"), row.get("language")) for row in rows}
    tasks = {task for task, _ in pairs}
    if (
        runtime.get("version") != "carryrl.runtime_text_controls.factorized.v2"
        or runtime.get("passed") is not True
        or runtime.get("used_for_selection") is not False
        or runtime.get("records") != 32
        or runtime.get("passed_records") != 32
        or len(rows) != 32
        or len(tasks) != 16
        or pairs != {(task, language) for task in tasks for language in ("ru", "en")}
    ):
        return False
    for row in rows:
        if (
            row.get("passed") is not True
            or row.get("errors")
            or set(row.get("counts", {})) != expected
        ):
            return False
        for name, count in row["counts"].items():
            total, correct = count.get("total"), count.get("correct")
            if (
                type(total) is not int
                or type(correct) is not int
                or not 0 <= correct <= total
                or (total and correct / total < (1 if name == "unknown" else 0.8))
            ):
                return False
    return True


def _checked_admission(validation, runtime, sealed, training):
    from carryrl.rewards.control_text_factorized_training import (
        STAGED_TRAINING_PROTOCOL,
        _staged_training_complete,
    )

    for report in (validation, sealed):
        if report is not None and (
            report.get("population_contract") != FACTORIZED_POPULATION_CONTRACT
            or report.get("payable_compositions") is not True
        ):
            raise ValueError("factorized admission requires actual payable composition evidence")
    calibration = calibration_gates(validation)
    sealed_gates = calibration_gates(sealed) if sealed is not None else None
    runtime_ok = _runtime_gate(runtime)
    passed = bool(
        training.get("trained") is True
        and training.get("selected_development_feasible") is True
        and training.get("encoder_frozen_verified") is True
        and training.get("sealed_test_used") is True
        and calibration["passed"]
        and runtime_ok
        and sealed_gates
        and sealed_gates["passed"]
        and (
            training.get("training_protocol") != STAGED_TRAINING_PROTOCOL
            or _staged_training_complete(training)
        )
    )
    return {
        "calibration": calibration,
        "runtime_passed": runtime_ok,
        "sealed": sealed_gates,
        "passed": passed,
    }


def save_factorized_control_artifact(
    directory,
    *,
    encoder,
    head,
    calibration,
    validation,
    runtime_validation,
    sealed_validation,
    training,
):
    """Keep rejected fits inspectable; only fully admitted bundles may load for RL."""
    checked_factorized_calibration(calibration)
    if head.architecture != FACTORIZED_HEAD_ARCHITECTURE:
        raise ValueError("factorized artifact architecture mismatch")
    if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
        raise ValueError("nonfinite factorized reward weights")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoder.save_resources(directory / "language")
    torch.save(encoder.state_dict(), directory / "encoder.pt")
    torch.save(head.state_dict(), directory / "head.pt")
    specification = Path(__file__).parents[3] / "SPEC.md"
    (directory / "project_specification.md").write_bytes(specification.read_bytes())
    reports = {
        "calibration.json": calibration,
        "validation.json": validation,
        "runtime_validation.json": runtime_validation,
        "sealed_validation.json": sealed_validation,
        "training.json": training,
    }
    for name, report in reports.items():
        (directory / name).write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
    admission = _checked_admission(validation, runtime_validation, sealed_validation, training)
    manifest = {
        "version": FACTORIZED_MODEL_VERSION,
        "head_architecture": FACTORIZED_HEAD_ARCHITECTURE,
        "encoder_id": TEXT_ID,
        "revision": TEXT_REVISION,
        "encoder_state_hash": encoder_state_hash(encoder.state_dict()),
        "candidates": list(CANDIDATES),
        "training": training,
        "pins": training.get("artifact_pins", {}),
        "admission": admission,
        "ready": admission["passed"],
        "shared_encoder": False,
        "files": {
            str(path.relative_to(directory)): _sha_file(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n"
    )
    return manifest


def inspect_factorized_control_artifact(
    directory, *, expected_manifest_hash=None, runtime_instructions=None, runtime_controls=None
):
    """Verify complete inventory, source/protocol/input pins and admission without E5."""
    from carryrl.rewards.control_text_factorized_training import (
        LOSS_PROTOCOL,
        STAGED_TRAINING_PROTOCOL,
        _staged_training_complete,
        protocol_identity,
        source_identity,
    )

    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    if expected_manifest_hash is not None and _sha_file(manifest_path) != expected_manifest_hash:
        raise ValueError("factorized manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("version") != FACTORIZED_MODEL_VERSION
        or manifest.get("head_architecture") != FACTORIZED_HEAD_ARCHITECTURE
        or manifest.get("encoder_id") != TEXT_ID
        or manifest.get("revision") != TEXT_REVISION
        or manifest.get("candidates") != list(CANDIDATES)
    ):
        raise ValueError("unsupported factorized reward identity")
    files = manifest.get("files", {})
    required = {
        "head.pt",
        "calibration.json",
        "validation.json",
        "runtime_validation.json",
        "sealed_validation.json",
        "training.json",
        "project_specification.md",
    }
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    if (
        not isinstance(files, dict)
        or not required <= files.keys()
        or actual != set(files) | {"manifest.json"}
    ):
        raise ValueError("factorized reward has missing or unhashed resources")
    for name, expected in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or _sha_file(path) != expected:
            raise ValueError(f"factorized reward resource hash mismatch: {name}")
    reports = {
        name: json.loads((directory / name).read_text())
        for name in required
        if name.endswith(".json")
    }
    checked_factorized_calibration(reports["calibration.json"])
    training, pins = reports["training.json"], manifest.get("pins", {})
    if training != manifest.get("training") or pins != training.get("artifact_pins"):
        raise ValueError("factorized training/provenance manifest mismatch")
    required_pins = {
        "source_identity",
        "dataset_sha256",
        "instructions_sha256",
        "controls_sha256",
        "specification_hash",
        "protocol_sha256",
        "runtime_report_sha256",
        "sealed_report_sha256",
        "project_specification_sha256",
    }
    if not required_pins <= pins.keys() or pins["source_identity"] != source_identity():
        raise ValueError("factorized source/protocol pins missing or changed")
    for name in required_pins - {"source_identity"}:
        digest = pins[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid factorized provenance digest")
    training_protocol = training.get("training_protocol", LOSS_PROTOCOL)
    if training_protocol == STAGED_TRAINING_PROTOCOL and not _staged_training_complete(training):
        raise ValueError("incomplete staged factorized training recipe")
    if pins["protocol_sha256"] != content_hash(protocol_identity(training_protocol)) or pins[
        "project_specification_sha256"
    ] != _sha_file(directory / "project_specification.md"):
        raise ValueError("factorized training protocol/project specification changed")
    runtime = reports["runtime_validation.json"]
    for key in ("instructions_sha256", "controls_sha256", "specification_hash"):
        if runtime.get(key) != pins[key]:
            raise ValueError("factorized runtime identity mismatch")
    if (
        content_hash(runtime) != pins["runtime_report_sha256"]
        or content_hash(reports["sealed_validation.json"]) != pins["sealed_report_sha256"]
    ):
        raise ValueError("factorized evaluation report identity mismatch")
    for path, key in (
        (runtime_instructions, "instructions_sha256"),
        (runtime_controls, "controls_sha256"),
    ):
        if path is not None and _sha_file(Path(path)) != pins[key]:
            raise ValueError("factorized current instructions/specification mismatch")
    admission = _checked_admission(
        reports["validation.json"], runtime, reports["sealed_validation.json"], training
    )
    if (
        manifest.get("ready") is not True
        or not admission["passed"]
        or manifest.get("admission") != admission
    ):
        raise ValueError("factorized reward admission failed")
    return {
        "manifest": manifest,
        "reports": reports,
        "manifest_sha256": _sha_file(manifest_path),
        "ready": True,
        "version": manifest["version"],
        "files": files,
        "head_architecture": manifest["head_architecture"],
        "pins": pins,
        "gates": admission["calibration"],
        "admission": admission,
    }


def load_factorized_control_reward(directory, *, shared_encoder=None):
    directory = Path(directory).resolve()
    inspected = inspect_factorized_control_artifact(directory)
    manifest, reports = inspected["manifest"], inspected["reports"]
    files = manifest["files"]
    if shared_encoder is None:
        if "encoder.pt" not in files:
            raise ValueError("factorized reward requires its shared E5 encoder")
        state = torch.load(directory / "encoder.pt", map_location="cpu", weights_only=True)
        if encoder_state_hash(state) != manifest["encoder_state_hash"]:
            raise ValueError("factorized E5 identity mismatch")
        encoder = MultilingualInstructionEncoder.from_resources(directory / "language")
        encoder.load_state_dict(state, strict=True)
    else:
        encoder = shared_encoder
        if (
            getattr(encoder, "_awaiting_checkpoint_state", False)
            or encoder.max_tokens != 512
            or encoder_state_hash(encoder.state_dict()) != manifest["encoder_state_hash"]
        ):
            raise ValueError("factorized shared E5 identity mismatch")
    encoder.eval().requires_grad_(False)
    head = FactorizedControlHead()
    head.load_state_dict(
        torch.load(directory / "head.pt", map_location="cpu", weights_only=True), strict=True
    )
    if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
        raise ValueError("nonfinite factorized reward weights")
    model = FactorizedTextControlReward(
        encoder,
        head,
        model_hash=content_hash(
            {
                "version": FACTORIZED_MODEL_VERSION,
                "head": files["head.pt"],
                "encoder": manifest["encoder_state_hash"],
            }
        ),
        calibration=reports["calibration.json"],
        calibration_hash=files["calibration.json"],
        _verified=True,
    )
    model._manifest = manifest
    model._resources = {
        name: (directory / name).read_bytes() for name in files if name != "encoder.pt"
    }
    return model
