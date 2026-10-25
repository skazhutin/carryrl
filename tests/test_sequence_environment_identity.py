"""Resume validates environment identity; explicit weight reuse keeps legacy access."""

import json
from dataclasses import replace

import pytest
import torch
from test_sequence_training import TinyReward, config, factory

from carryrl.envs.atomic_suite import SCORING_VERSION, SUITE_VERSION
from carryrl.training.run import file_sha256
from carryrl.training.sequence import (
    load_sequence_checkpoint,
    save_sequence_checkpoint,
    tensor_hash,
)


@pytest.fixture
def checkpoint(tmp_path):
    cfg = config()
    agent, reward = factory(cfg.agent), TinyReward()
    optimizer = torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad])
    destination = save_sequence_checkpoint(agent, optimizer, reward, cfg, tmp_path,
                                           rollout=1, decisions=2, revision=3)
    return destination, replace(cfg, total_rollouts=2), tensor_hash(agent)


def _rewrite(checkpoint, field, *, value=None, remove=False, manifest=True):
    archive = checkpoint / "state.pt"
    data = torch.load(archive, weights_only=True)
    if remove:
        data.pop(field, None)
    else:
        data[field] = value
    torch.save(data, archive)
    path = checkpoint / "manifest.json"
    declared = json.loads(path.read_text())
    declared["sha256"] = file_sha256(archive)
    if manifest:
        if remove:
            declared.pop(field, None)
        else:
            declared[field] = value
    path.write_text(json.dumps(declared))


def _load(checkpoint, cfg, *, evaluation_only=False):
    return load_sequence_checkpoint(checkpoint, cfg, agent_factory=factory,
                                    reward_factory=TinyReward, evaluation_only=evaluation_only)


def test_new_checkpoints_record_both_environment_versions_and_resume(checkpoint):
    path, cfg, identity = checkpoint
    manifest = json.loads((path / "manifest.json").read_text())
    loaded, _, _, data = _load(path, cfg)
    for field, expected in (("suite_version", SUITE_VERSION), ("scoring_version", SCORING_VERSION)):
        assert data[field] == manifest[field] == expected
    assert tensor_hash(loaded) == identity


@pytest.mark.parametrize("field", ["suite_version", "scoring_version"])
@pytest.mark.parametrize("remove", [False, True])
def test_other_or_unknown_environment_requires_explicit_reuse(checkpoint, field, remove):
    path, cfg, identity = checkpoint
    _rewrite(path, field, value="other.environment.v1", remove=remove)
    with pytest.raises(ValueError, match=f"current {field}"):
        _load(path, cfg)
    loaded, _, _, _ = _load(path, cfg, evaluation_only=True)
    assert tensor_hash(loaded) == identity


@pytest.mark.parametrize("field", ["suite_version", "scoring_version"])
def test_inconsistent_versions_are_rejected_even_for_explicit_reuse(checkpoint, field):
    path, cfg, _ = checkpoint
    _rewrite(path, field, value="other.environment.v1", manifest=False)
    with pytest.raises(ValueError, match=f"{field} metadata mismatch"):
        _load(path, cfg, evaluation_only=True)


def test_legacy_manifest_does_not_claim_resume_compatibility(checkpoint):
    path, cfg, identity = checkpoint
    manifest = path / "manifest.json"
    declared = json.loads(manifest.read_text())
    declared.pop("suite_version")
    declared.pop("scoring_version")
    manifest.write_text(json.dumps(declared))
    with pytest.raises(ValueError, match="current suite_version"):
        _load(path, cfg)
    assert tensor_hash(_load(path, cfg, evaluation_only=True)[0]) == identity
