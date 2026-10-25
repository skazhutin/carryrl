import json

import pytest
import torch
from test_control_text import encoder, report

from carryrl.rewards.control_text import load_text_control_reward
from carryrl.rewards.control_text_structured import (
    STRUCTURED_DECODER_VERSION,
    STRUCTURED_HEAD_ARCHITECTURE,
    STRUCTURED_POPULATION_CONTRACT,
    JointStructuredControlHead,
)
from carryrl.rewards.control_text_structured_artifact import (
    load_compatible_text_control_reward,
    load_structured_control_reward,
    save_structured_control_artifact,
)


def runtime_report():
    return {
        "passed": True, "records": 32, "passed_records": 32,
        "results": [{"task": f"fixture{i}", "language": lang, "passed": True,
                     "errors": [], "counts": {
                         name: {"total": 10, "correct": 10}
                         for name in ("supported", "unsupported", "unknown", "held", "press",
                                      "aliases")}}
                    for i in range(16) for lang in ("ru", "en")],
    }


def save(path, language, *, runtime=None, sealed=True, trained=True):
    return save_structured_control_artifact(
        path, encoder=language, head=JointStructuredControlHead(),
        calibration={"joint_temperature": 1.0, "joint_threshold": .99,
                     "architecture": STRUCTURED_HEAD_ARCHITECTURE,
                     "decoder_version": STRUCTURED_DECODER_VERSION,
                     "population_contract": STRUCTURED_POPULATION_CONTRACT},
        validation=report(), runtime_validation=runtime or runtime_report(),
        sealed_validation=report() if sealed else None,
        training={"trained": trained, "selected_development_feasible": True,
                  "sealed_test_used": sealed},
    )


def test_structured_bundle_roundtrip_and_export_survive_source_archive(tmp_path):
    language = encoder()
    path = tmp_path / "model"
    manifest = save(path, language)
    assert manifest["ready"]
    model = load_compatible_text_control_reward(path, shared_encoder=language)
    # The existing RL checkpoint entry point must dispatch the new bundle too.
    assert load_text_control_reward(path, shared_encoder=language).identity == model.identity
    state = torch.load(path / "head.pt", map_location="cpu", weights_only=True)
    for key, value in model.head.state_dict().items():
        torch.testing.assert_close(value, state[key])
    path.rename(tmp_path / "archived_model")
    exported = model.save_resources(tmp_path / "export", shared_encoder=True)
    assert not (exported / "encoder.pt").exists()
    restored = load_structured_control_reward(exported, shared_encoder=language)
    assert restored.identity == model.identity
    with pytest.raises(ValueError, match="shared E5"):
        load_structured_control_reward(exported, shared_encoder=encoder())
    (exported / "head.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_structured_control_reward(exported, shared_encoder=language)


@pytest.mark.parametrize("failure", ["missing_sealed", "untrained", "runtime_coverage",
                                     "duplicate_runtime"])
def test_structured_loader_recomputes_gate_despite_ready_flag(tmp_path, failure):
    language = encoder()
    runtime = runtime_report()
    if failure == "runtime_coverage":
        runtime["results"][0]["counts"]["aliases"]["correct"] = 7
    if failure == "duplicate_runtime":
        runtime["results"][1] = runtime["results"][0]
    path = tmp_path / "model"
    manifest = save(path, language, runtime=runtime, sealed=failure != "missing_sealed",
                    trained=failure != "untrained")
    assert not manifest["ready"]
    manifest["ready"] = True
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="admission failed"):
        load_structured_control_reward(path, shared_encoder=language)
