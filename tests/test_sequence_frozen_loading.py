"""Frozen checkpoint loading skips optimizer allocation without relaxing identity checks."""

import gzip
import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_sequence_training import TinyReward, config, factory

from carryrl.training.run import file_sha256
from carryrl.training.sequence import (
    load_sequence_checkpoint,
    save_sequence_checkpoint,
    tensor_hash,
)


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    cfg = config()
    agent, reward = factory(cfg.agent), TinyReward()
    optimizer = torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad],
                                  lr=.003, weight_decay=.07)
    parameter = optimizer.param_groups[0]['params'][0]
    optimizer.state[parameter] = {'step': torch.tensor(7.),
                                  'exp_avg': torch.full_like(parameter, .25),
                                  'exp_avg_sq': torch.full_like(parameter, .5)}
    path = save_sequence_checkpoint(agent, optimizer, reward, cfg, tmp_path,
                                    rollout=1, decisions=2, revision=3)
    return path, replace(cfg, total_rollouts=2), tensor_hash(agent), tensor_hash(reward)


def _load(path, cfg, **kwargs):
    return load_sequence_checkpoint(path, cfg, agent_factory=factory,
                                    reward_factory=TinyReward, **kwargs)


def _rewrite(path, mutate, *, zip_serialization=True):
    archive = path / 'state.pt'
    data = torch.load(archive, weights_only=True)
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    mutate(data, manifest)
    torch.save(data, archive, _use_new_zipfile_serialization=zip_serialization)
    manifest['sha256'] = file_sha256(archive)
    manifest_path.write_text(json.dumps(manifest))


@pytest.fixture
def storage_space(monkeypatch):
    from carryrl.training import checkpoint_storage

    monkeypatch.setattr(checkpoint_storage.shutil, 'disk_usage',
                        lambda _: SimpleNamespace(free=64 * 1024**3))


def test_archived_checkpoint_restores_before_loading(checkpoint, tmp_path, storage_space):
    source, cfg, agent_hash, reward_hash = checkpoint
    path = tmp_path / 'runs' / 'historical' / 'checkpoints' / source.name
    shutil.copytree(source, path)
    original = (path / 'state.pt').read_bytes()
    (path / 'state.pt.gz').write_bytes(gzip.compress(original))
    (path / 'state.pt').unlink()
    agent, optimizer, reward, _ = _load(path, cfg, evaluation_only=True, load_optimizer=False)
    assert optimizer is None
    assert tensor_hash(agent) == agent_hash and tensor_hash(reward) == reward_hash
    assert (path / 'state.pt').read_bytes() == original


def test_corrupt_archived_checkpoint_is_preserved(checkpoint, tmp_path, storage_space):
    source, cfg, _, _ = checkpoint
    path = tmp_path / 'runs' / 'historical' / 'checkpoints' / source.name
    shutil.copytree(source, path)
    packed = gzip.compress(b'corrupt checkpoint')
    (path / 'state.pt.gz').write_bytes(packed)
    (path / 'state.pt').unlink()
    with pytest.raises(ValueError, match='hash mismatch'):
        _load(path, cfg, evaluation_only=True, load_optimizer=False)
    assert not (path / 'state.pt').exists()
    assert (path / 'state.pt.gz').read_bytes() == packed


@pytest.mark.parametrize('evaluation_only', [True, False])
@pytest.mark.parametrize('zip_serialization', [True, False])
def test_frozen_load_skips_optimizer_and_drops_its_storages(
        checkpoint, monkeypatch, evaluation_only, zip_serialization):
    path, cfg, agent_hash, reward_hash = checkpoint
    if not zip_serialization:
        _rewrite(path, lambda data, manifest: None, zip_serialization=False)
    calls = []
    original_load = torch.load

    def tracked_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    def forbidden_optimizer(*args, **kwargs):
        pytest.fail('frozen loading must not construct AdamW')

    monkeypatch.setattr(torch, 'load', tracked_load)
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden_optimizer)
    agent, optimizer, reward, data = _load(
        path, cfg, evaluation_only=evaluation_only, load_optimizer=False)
    assert optimizer is None and 'optimizer_state' not in data
    assert tensor_hash(agent) == agent_hash and tensor_hash(reward) == reward_hash
    assert not reward.training and not any(p.requires_grad for p in reward.parameters())
    assert data['weights_revision'] == 3 and data['rollout'] == 1
    assert calls == [{'map_location': 'cpu', 'weights_only': True, 'mmap': zip_serialization}]


@pytest.mark.parametrize('evaluation_only', [True, False])
def test_default_loading_preserves_warm_start_and_resume_optimizer(checkpoint, evaluation_only):
    path, cfg, agent_hash, reward_hash = checkpoint
    agent, optimizer, reward, data = _load(path, cfg, evaluation_only=evaluation_only)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]['lr'] == .003
    assert optimizer.param_groups[0]['weight_decay'] == .07
    assert len(optimizer.state) == 1
    state = next(iter(optimizer.state.values()))
    assert state['step'].item() == 7
    assert torch.all(state['exp_avg'] == .25) and torch.all(state['exp_avg_sq'] == .5)
    assert 'optimizer_state' in data
    assert tensor_hash(agent) == agent_hash and tensor_hash(reward) == reward_hash


def test_frozen_loading_still_checks_archive_sha(checkpoint):
    path, cfg, *_ = checkpoint
    archive = path / 'state.pt'
    archive.write_bytes(archive.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='archive SHA256 mismatch'):
        _load(path, cfg, evaluation_only=True, load_optimizer=False)


@pytest.mark.parametrize(('damage', 'message'), [
    ('format', 'unsupported sequence checkpoint'),
    ('suite_version', 'suite_version metadata mismatch'),
    ('scoring_version', 'scoring_version metadata mismatch'),
    ('agent_sha256', 'tensor hash mismatch'),
    ('reward_sha256', 'tensor hash mismatch'),
    ('weights_revision', 'progress counters disagree'),
    ('training_backend', 'training backend metadata mismatch'),
    ('reward_model_version', 'reward model versions disagree'),
])
def test_frozen_loading_preserves_checkpoint_identity_validation(checkpoint, damage, message):
    path, cfg, *_ = checkpoint

    def damage_identity(data, manifest):
        if damage in {'agent_sha256', 'reward_sha256', 'training_backend'}:
            manifest[damage] = 'changed'
        elif damage == 'weights_revision':
            data[damage] += 1
        else:
            data[damage] = 'changed'

    _rewrite(path, damage_identity)
    with pytest.raises(ValueError, match=message):
        _load(path, cfg, evaluation_only=True, load_optimizer=False)


def test_frozen_flag_does_not_bypass_resume_config_and_version_rules(checkpoint):
    path, cfg, *_ = checkpoint
    with pytest.raises(ValueError, match='resume must preserve settings'):
        _load(path, replace(cfg, learning_rate=.009), load_optimizer=False)
    with pytest.raises(ValueError, match='architecture must match'):
        _load(path, replace(cfg, agent=replace(cfg.agent, width=64)),
              evaluation_only=True, load_optimizer=False)

    def change_versions(data, manifest):
        data['suite_version'] = manifest['suite_version'] = 'historical.suite'

    _rewrite(path, change_versions)
    with pytest.raises(ValueError, match='current suite_version'):
        _load(path, cfg, load_optimizer=False)
    assert _load(path, cfg, evaluation_only=True, load_optimizer=False)[1] is None
