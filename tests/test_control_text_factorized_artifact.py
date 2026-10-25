"""Complete authenticated bundle inventory and fail-closed composition admission."""

import json
from pathlib import Path

import pytest
from test_control_text import encoder, report
from test_control_text_factorized import calibration
from test_control_text_factorized_staged import staged_training_report
from test_control_text_structured_artifact import runtime_report

from carryrl.rewards.control_text import _sha_file, content_hash
from carryrl.rewards.control_text_factorized import (
    FACTORIZED_POPULATION_CONTRACT,
    FactorizedControlHead,
)
from carryrl.rewards.control_text_factorized_artifact import (
    inspect_factorized_control_artifact,
    load_factorized_control_reward,
    save_factorized_control_artifact,
)
from carryrl.rewards.control_text_factorized_training import (
    LOSS_PROTOCOL,
    STAGED_TRAINING_PROTOCOL,
    protocol_identity,
    source_identity,
)


def save(path, language, *, sealed=True, training_protocol=LOSS_PROTOCOL, training_updates=None):
    runtime = runtime_report()
    runtime.update(version="carryrl.runtime_text_controls.factorized.v2", used_for_selection=False)
    for row in runtime["results"]:
        row["counts"].update(
            {
                name: {"total": 10, "correct": 10}
                for name in ("relative", "different_pairs", "supported_payable")
            }
        )
    identity = dict(
        instructions_sha256="a" * 64, controls_sha256="b" * 64, specification_hash="c" * 64
    )
    runtime.update(identity)
    validation = {
        **report(),
        "population_contract": FACTORIZED_POPULATION_CONTRACT,
        "payable_compositions": True,
    }
    test = validation if sealed else None
    return save_factorized_control_artifact(
        path,
        encoder=language,
        head=FactorizedControlHead(),
        calibration=calibration(),
        validation=validation,
        runtime_validation=runtime,
        sealed_validation=test,
        training={
            "trained": True,
            "selected_development_feasible": True,
            "sealed_test_used": sealed,
            "encoder_frozen_verified": True,
            **(training_updates or {}),
            "artifact_pins": {
                **identity,
                "source_identity": source_identity(),
                "dataset_sha256": "d" * 64,
                "protocol_sha256": content_hash(protocol_identity(training_protocol)),
                "project_specification_sha256": _sha_file(Path(__file__).parents[1] / "SPEC.md"),
                "runtime_report_sha256": content_hash(runtime),
                "sealed_report_sha256": content_hash(test),
            },
        },
    )


def test_factorized_bundle_verified_load_and_shared_encoder_export(tmp_path):
    language = encoder()
    path = tmp_path / "model"
    assert save(path, language)["ready"]
    assert inspect_factorized_control_artifact(path)["ready"]
    model = load_factorized_control_reward(path, shared_encoder=language)
    exported = model.save_resources(tmp_path / "export")
    restored = load_factorized_control_reward(exported, shared_encoder=language)
    assert restored.identity == model.identity
    with pytest.raises(ValueError, match="shared E5"):
        load_factorized_control_reward(exported, shared_encoder=encoder())
    (exported / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="unhashed"):
        inspect_factorized_control_artifact(exported)


def test_rejected_bundle_stays_rejected_and_tampering_is_detected(tmp_path):
    language = encoder()
    path = tmp_path / "rejected"
    manifest = save(path, language, sealed=False)
    assert not manifest["ready"]
    manifest["ready"] = True
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="admission failed"):
        load_factorized_control_reward(path, shared_encoder=language)
    path = tmp_path / "accepted"
    save(path, language)
    (path / "calibration.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        inspect_factorized_control_artifact(path)


def test_manifest_and_current_instruction_pins_are_checked(tmp_path):
    path = tmp_path / "model"
    save(path, encoder())
    with pytest.raises(ValueError, match="manifest hash"):
        inspect_factorized_control_artifact(path, expected_manifest_hash="f" * 64)
    instructions = tmp_path / "instructions.json"
    instructions.write_text("{}")
    with pytest.raises(ValueError, match="current instructions"):
        inspect_factorized_control_artifact(path, runtime_instructions=instructions)


def test_staged_bundle_authenticates_its_own_protocol_and_complete_stage_metadata(tmp_path):
    path = tmp_path / "staged"
    assert save(
        path,
        encoder(),
        training_protocol=STAGED_TRAINING_PROTOCOL,
        training_updates=staged_training_report(),
    )["ready"]
    assert inspect_factorized_control_artifact(path)["ready"]
    wrong_protocol = tmp_path / "wrong_protocol"
    save(wrong_protocol, encoder(), training_updates=staged_training_report())
    with pytest.raises(ValueError, match="protocol/project"):
        inspect_factorized_control_artifact(wrong_protocol)


def test_partial_staged_bundle_cannot_admit_even_with_all_quality_gates(tmp_path):
    value = staged_training_report()
    value["staged_completion"]["complete"] = False
    path = tmp_path / "partial"
    assert not save(
        path,
        encoder(),
        training_protocol=STAGED_TRAINING_PROTOCOL,
        training_updates=value,
    )["ready"]
    with pytest.raises(ValueError, match="incomplete staged"):
        inspect_factorized_control_artifact(path)
