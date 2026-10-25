"""Exact cache identity, atomic publication, bounded storage and full policy replay."""

import json
import sys
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_imitation import make_agent
from test_imitation_feature_storage import script

from carryrl.agent.model import GeneralistAgent
from carryrl.training.feature_store import (
    FrozenFeatureStore,
    FrozenFeatureWriter,
    feature_cache_identity,
)
from carryrl.training.imitation import collect_scripted_demonstrations, train_imitation
from carryrl.training.sequence import tensor_hash
from carryrl.training.sequence_config import SequenceRunConfig


@pytest.fixture
def cache_input():
    frames = [np.full((12, 10, 3), index, dtype=np.uint8) for index in range(4)]
    identity = feature_cache_identity(frames, [0, 2, 3], vision_sha256="a" * 64,
        preprocessing={"image_size": 28, "patch_size": 14}, feature_shape=(4, 384), dtype="float16")
    values = np.random.default_rng(13).normal(size=(3, 4, 384)).astype(np.float16)
    return frames, identity, values


def _write(cache_dir, identity, values):
    with FrozenFeatureWriter(cache_dir, identity, max_bytes=values.nbytes) as writer:
        writer.write(identity["indices"], values)
        return writer.publish()


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_lossless_readonly_memmap_values_dtype_and_detached_copies(tmp_path, cache_input, dtype):
    _, identity, values = cache_input
    identity["dtype"] = dtype
    if dtype == "float32":
        values = np.random.default_rng(27).normal(size=values.shape).astype(np.float32)
    store = _write(tmp_path, identity, values)
    loaded = FrozenFeatureStore.load(tmp_path, identity, max_bytes=values.nbytes)
    assert isinstance(loaded.array, np.memmap) and not loaded.array.flags.writeable
    assert len(store) == len(loaded) == 3
    assert loaded.manifest["bytes"] == values.nbytes
    assert loaded.manifest["identity"]["indices"] == [0, 2, 3]
    assert loaded.manifest["created_utc"] == store.manifest["created_utc"]
    for row, index in enumerate(identity["indices"]):
        actual = loaded[index]
        assert actual.dtype == getattr(torch, dtype) and not actual.requires_grad
        np.testing.assert_array_equal(actual.numpy(), values[row])
        actual.fill_(0)
        np.testing.assert_array_equal(loaded[index].numpy(), values[row])
    with pytest.raises(KeyError):
        loaded[1]


def test_differently_shaped_rgb_inputs_and_out_of_order_batches_preserve_index_mapping(
        tmp_path, cache_input):
    frames, _, values = cache_input
    frames[2] = frames[2].reshape(10, 12, 3)
    identity = feature_cache_identity(frames, [0, 2, 3], vision_sha256="a" * 64,
        preprocessing={"image_size": 28, "patch_size": 14}, feature_shape=(4, 384), dtype="float16")
    with FrozenFeatureWriter(tmp_path, identity, max_bytes=values.nbytes) as writer:
        writer.write([3], values[2:])
        writer.write([2, 0], values[[1, 0]])
        store = writer.publish()
    for row, index in enumerate(identity["indices"]):
        np.testing.assert_array_equal(store[index].numpy(), values[row])


def test_budget_checked_before_allocation_and_on_reuse(tmp_path, cache_input):
    _, identity, values = cache_input
    with pytest.raises(MemoryError, match="feature-budget-mib"):
        FrozenFeatureWriter(tmp_path, identity, max_bytes=values.nbytes - 1)
    assert not list(tmp_path.iterdir())
    _write(tmp_path, identity, values)
    with pytest.raises(MemoryError, match="feature-budget-mib"):
        FrozenFeatureStore.load(tmp_path, identity, max_bytes=values.nbytes - 1)


@pytest.mark.parametrize("changed", ["vision", "rgb", "shape", "dtype", "preprocess", "indices"])
def test_changed_identity_never_reuses_entry(tmp_path, cache_input, changed):
    frames, identity, values = cache_input
    _write(tmp_path, identity, values)
    options = {"vision_sha256": identity["vision_sha256"],
               "preprocessing": identity["preprocessing"], "feature_shape": (4, 384),
               "dtype": identity["dtype"]}
    indices = identity["indices"]
    if changed == "vision":
        options["vision_sha256"] = "b" * 64
    elif changed == "rgb":
        frames[2] = frames[2].copy()
        frames[2][0, 0, 0] += 1
    elif changed == "shape":
        frames[2] = frames[2].reshape(10, 12, 3)
    elif changed == "dtype":
        options["dtype"] = "float32"
    elif changed == "preprocess":
        options["preprocessing"] = {"image_size": 56, "patch_size": 14}
    else:
        indices = [0, 1, 3]
    changed_identity = feature_cache_identity(frames, indices, **options)
    assert changed_identity != identity
    assert FrozenFeatureStore.load(tmp_path, changed_identity, max_bytes=values.nbytes * 2) is None


@pytest.mark.parametrize("corruption", ["bytes", "manifest", "missing", "indices"])
def test_corrupt_or_partial_published_cache_is_rejected(tmp_path, cache_input, corruption):
    _, identity, values = cache_input
    store = _write(tmp_path, identity, values)
    data, manifest_path = store.directory / "features.npy", store.directory / "manifest.json"
    if corruption == "bytes":
        with data.open("r+b") as file:
            file.seek(-1, 2)
            file.write(b"\x7f")
    elif corruption == "missing":
        manifest_path.unlink()
    else:
        manifest = json.loads(manifest_path.read_text())
        if corruption == "manifest":
            manifest["identity"]["vision_sha256"] = "wrong"
        else:
            manifest["identity"]["indices"] = [0, 2]
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="feature cache"):
        FrozenFeatureStore.load(tmp_path, identity, max_bytes=values.nbytes)


def test_interrupted_or_incomplete_write_never_publishes_reusable_manifest(tmp_path, cache_input):
    _, identity, values = cache_input
    with pytest.raises(ValueError, match="incomplete"):
        with FrozenFeatureWriter(tmp_path, identity, max_bytes=values.nbytes) as writer:
            writer.write([0], values[:1])
            writer.publish()
    assert not list(tmp_path.iterdir())
    assert FrozenFeatureStore.load(tmp_path, identity, max_bytes=values.nbytes) is None


def test_atomic_publish_failure_cleans_only_own_partial(tmp_path, cache_input, monkeypatch):
    import carryrl.training.feature_store as module

    _, identity, values = cache_input
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()

    def fail_rename(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(module.os, "rename", fail_rename)
    with pytest.raises(OSError, match="disk failure"):
        _write(tmp_path, identity, values)
    assert list(tmp_path.iterdir()) == [unrelated]
    assert FrozenFeatureStore.load(tmp_path, identity, max_bytes=values.nbytes) is None


def test_concurrent_publish_uses_only_verified_winner_and_cleans_losing_partial(
        tmp_path, cache_input):
    _, identity, values = cache_input
    with FrozenFeatureWriter(tmp_path, identity, max_bytes=values.nbytes) as first:
        with FrozenFeatureWriter(tmp_path, identity, max_bytes=values.nbytes) as second:
            first.write(identity["indices"], values)
            second.write(identity["indices"], values)
            winner = first.publish()
            reused = second.publish()
            assert reused.directory == winner.directory
            assert reused.manifest == winner.manifest
    assert list(tmp_path.iterdir()) == [winner.directory]
    np.testing.assert_array_equal(reused.array, values)


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_full_policy_training_from_memmap_matches_exact_rounded_features(tmp_path, backend):
    if backend == "mlx":
        mx = pytest.importorskip("mlx.core")
        if not mx.metal.is_available():
            pytest.skip("MLX requires Metal")
    agent = make_agent()
    reference = deepcopy(agent)
    dataset = collect_scripted_demonstrations(1, tasks=("navigation",), difficulty=0)
    indices = list(range(len(dataset.frames)))
    identity = feature_cache_identity(
        dataset.frames, indices, vision_sha256=tensor_hash(agent.vision),
        preprocessing={"image_size": 28, "patch_size": 14}, feature_shape=(4, 384), dtype="float16")
    values = np.stack([agent.encode_frames(torch.tensor(frame)[None]).detach().half().numpy()[0]
                       for frame in dataset.frames])
    _write(tmp_path, identity, values)
    store = FrozenFeatureStore.load(tmp_path, identity, max_bytes=values.nbytes)
    rounded = {index: torch.tensor(values[index]).float() for index in indices}
    optimizers = [torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
                  for model in (agent, reference)]
    engine = None
    if backend == "mlx":
        from carryrl.training.mlx import MLXPPOBackend

        engine = MLXPPOBackend(agent, optimizers[0])
    result = train_imitation(agent, dataset, optimizer=optimizers[0], backend=backend,
        mlx_backend=engine, feature_provider=lambda index: store[index].float()
        if backend == "torch" else store[index])
    expected = train_imitation(reference, dataset, optimizer=optimizers[1],
                               feature_provider=rounded.__getitem__)
    assert result["history"][0]["nll"] == pytest.approx(expected["history"][0]["nll"], abs=3e-6)
    for name, parameter in agent.named_parameters():
        assert parameter.dtype == torch.float32
        torch.testing.assert_close(parameter, dict(reference.named_parameters())[name],
                                   atol=4e-6, rtol=3e-4)
    assert isinstance(store.array, np.memmap) and store.array.dtype == np.float16


def test_script_disk_cache_hit_skips_encoding_and_reports_original_cache_timestamp(
        tmp_path, monkeypatch):
    prototype = make_agent()
    prototype = GeneralistAgent(replace(prototype.config, segment_length=40),
                                vision=prototype.vision)
    prototype.vision.image_size, prototype.vision.patch_size = 28, 14
    prototype.vision.patch_grid, prototype.vision.hidden_size = (2, 2), 384
    config = SequenceRunConfig(agent=prototype.config, tasks=("navigation",), max_episode_steps=80,
                               eval_episodes_per_task=1)
    warm_start = tmp_path / "warm_start"
    warm_start.mkdir()
    (warm_start / "manifest.json").write_text(json.dumps({"sha256": "source"}))
    monkeypatch.setattr(script.SequenceRunConfig, "load", lambda _: config)

    def load_checkpoint(*args, **kwargs):
        agent = deepcopy(prototype)
        if len(outputs) == 2:
            def forbid_encoding(*args):
                pytest.fail("cache-hit run must not invoke frozen vision")
            agent.vision.forward = forbid_encoding
        optimizer = torch.optim.AdamW(p for p in agent.parameters() if p.requires_grad)
        return agent, optimizer, torch.nn.Linear(1, 1), {
            "weights_revision": 0, "decisions": 0, "config": config.to_dict()}

    def save_checkpoint(agent, optimizer, reward, config, output, **kwargs):
        checkpoint = output / "checkpoints" / "rollout_0000"
        checkpoint.mkdir(parents=True)
        (checkpoint / "manifest.json").write_text(json.dumps({
            "agent_sha256": tensor_hash(agent), "reward_sha256": tensor_hash(reward)}))
        return checkpoint

    monkeypatch.setattr(script, "load_sequence_checkpoint", load_checkpoint)
    monkeypatch.setattr(script, "save_sequence_checkpoint", save_checkpoint)
    monkeypatch.setattr(script, "link_matching_playbacks", lambda *args: None)
    monkeypatch.setattr(script, "evaluate_sequence", lambda *args, **kwargs: {"summaries": {
        "frozen_deterministic": {"outcomes": {"success": 0}, "episodes_started": 1}}})
    outputs = []
    for run in range(2):
        output = tmp_path / f"run{run}"
        outputs.append(output)
        monkeypatch.setattr(sys, "argv", ["train_imitation.py", "--config", "dummy.json",
            "--warm-start", str(warm_start), "--output", str(output), "--training-backend", "torch",
            "--episodes-per-task", "1", "--epochs", "1", "--feature-dtype", "float16",
            "--feature-cache-dir", str(tmp_path / "cache")])
        script.main()
    first, second = [json.loads((output / "feature_manifest.json").read_text())
                     for output in outputs]
    assert first["feature_cache_hit"] is False and second["feature_cache_hit"] is True
    assert first["feature_cache_sha256"] == second["feature_cache_sha256"]
    assert first["feature_cache_created_utc"] == second["feature_cache_created_utc"]
    assert first["demonstrations_sha256"] == second["demonstrations_sha256"]
    status = json.loads((outputs[1] / "status.json").read_text())
    assert status["status"] == "completed" and status["feature_cache_hit"] is True
    assert status["encoded_frames"] == status["encoded_frames_total"] == 0
    assert status["encoding_skipped"] is True and status["cached_frames"] == second["frames"]
