"""Authenticated frozen features of HUD-masked RGB, separate from policy caches."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from carryrl.agent.actions import PhysicalAction, encode_actions
from carryrl.evaluation.scene_suite import SCENE_SUITE_VERSION
from carryrl.rewards.history import MASK_VERSION
from carryrl.rewards.history_data import _combined_identity
from carryrl.training.imitation import _bundle_file_identity, _bundle_json
from carryrl.training.run import atomic_json
from carryrl.training.sequence import tensor_hash

FEATURE_VERSION = "carryrl.scene_history_features.v1"
MAX_FEATURE_BYTES = 128 * 1024**2
MAX_METADATA_BYTES = 1024**2


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _array_sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _pin(directory):
    files = {name: _bundle_file_identity(directory / name, limit)
             for name, limit in (("frames.npy", MAX_FEATURE_BYTES + 65536),
                                 ("goals.npy", MAX_FEATURE_BYTES + 65536),
                                 ("features.json", MAX_METADATA_BYTES))}
    return _combined_identity(files), files


def _corpus_identity(corpus):
    digest = getattr(corpus, "corpus_sha256", None)
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("features require an authenticated corpus_sha256")
    return digest


def _goal_texts(corpus):
    return sorted({row["goal"] for row in corpus.episodes})


class HistoryFeatureSet:
    """Return only fresh tensors of masked scenes, own actions and explicit goals."""

    def __init__(self, frames, goals, goal_texts, identity, *, directory=None):
        self.frames, self.goals = frames, goals
        self.goal_texts = tuple(goal_texts)
        self.identity = identity
        self.feature_width, self.goal_width = frames.shape[1], goals.shape[1]
        self.directory = directory

    def model_inputs(self, corpus, episode_index, *, device="cpu"):
        if (self.identity["corpus_sha256"] != _corpus_identity(corpus)
                or self.identity["event_names"] != list(corpus.event_names)):
            raise ValueError("feature corpus/event identity mismatch")
        episode = corpus.episodes[episode_index]
        actions = [PhysicalAction.from_dict(row) for row in episode["executed_actions"]]
        return {
            "scene_features": torch.tensor(self.frames[episode["frame_indices"]],
                                             device=device).unsqueeze(0),
            "executed_previous_actions": encode_actions([PhysicalAction(), *actions]).to(
                device=device, dtype=torch.float32).unsqueeze(0),
            "explicit_goal_features": torch.tensor(
                self.goals[self.goal_texts.index(episode["goal"])], device=device).unsqueeze(0),
        }


def _pooled(features, count):
    if (not isinstance(features, torch.Tensor) or features.ndim not in (2, 3)
            or features.shape[0] != count or not features.is_floating_point()
            or not torch.isfinite(features).all() or features.shape[-1] == 0):
        raise ValueError("scene encoder must return finite per-frame vectors or patches")
    result = features.mean(dim=1) if features.ndim == 3 else features
    if result.shape[1] > 4096 or not torch.isfinite(result).all():
        raise ValueError("invalid bounded scene feature width")
    return result.detach().cpu().float().numpy()


@torch.no_grad()
def prepare_history_features(corpus, output, *, vision_encoder, goal_encoder,
                             frame_batch_size=4, on_progress=None):
    """No optimizer, statistics fitting, unmasked frames or trainable projections."""
    digest = _corpus_identity(corpus)
    if type(frame_batch_size) is not int or not 1 <= frame_batch_size <= 32:
        raise ValueError("frame_batch_size must be between 1 and 32")
    if any(p.requires_grad for module in (vision_encoder, goal_encoder)
           for p in module.parameters()):
        raise ValueError("scene/goal encoders must be frozen")
    if any(value.device.type != "cpu" for module in (vision_encoder, goal_encoder)
           for value in (*module.parameters(), *module.buffers())):
        raise ValueError("feature preparation currently requires CPU encoders")
    if (corpus.frames.dtype != np.uint8 or corpus.frames.shape[1:] != (96, 96, 3)
            or not len(corpus.frames) or np.any(corpus.frames[:, :28])):
        raise ValueError("features require verified HUD-masked 96px RGB")
    if any(row["split"] not in {"train", "validation"} for row in corpus.episodes):
        raise ValueError("feature preparation excludes test episodes")
    output = Path(output)
    source = getattr(corpus, "directory", None)
    if source is not None and (output.resolve() == source or source in output.resolve().parents
                               or output.resolve() in source.parents):
        raise ValueError("feature output must be outside the source corpus")
    if output.exists() or output.is_symlink():
        raise FileExistsError("feature output already exists")
    before = [tensor_hash(module) for module in (vision_encoder, goal_encoder)]
    modes = [(module, module.training) for parent in (vision_encoder, goal_encoder)
             for module in parent.modules()]
    language_cache = getattr(goal_encoder, "_cache", None)
    saved_cache = language_cache.copy() if language_cache is not None else None
    output.mkdir(parents=True, exist_ok=False)
    vision_encoder.eval()
    goal_encoder.eval()
    try:
        with torch.random.fork_rng(devices=[]):
            device = next(vision_encoder.parameters(), torch.empty(0)).device
            batches, encoded_bytes = [], 0
            for start in range(0, len(corpus.frames), frame_batch_size):
                rgb = torch.tensor(np.asarray(corpus.frames[start:start + frame_batch_size]),
                                   device=device)
                batches.append(_pooled(vision_encoder(rgb), len(rgb)))
                encoded_bytes += batches[-1].nbytes
                if encoded_bytes > MAX_FEATURE_BYTES:
                    raise ValueError("scene feature byte budget exceeded")
                if on_progress is not None:
                    on_progress({"frames_encoded": start + len(rgb), "frames": len(corpus.frames),
                                 "updates": 0})
            frames = np.concatenate(batches)
            texts = _goal_texts(corpus)
            vectors = []
            for text in texts:
                value = goal_encoder(text)
                if (not isinstance(value, torch.Tensor) or value.ndim != 1
                        or not value.is_floating_point() or not torch.isfinite(value).all()
                        or not value.count_nonzero() or not 1 <= len(value) <= 4096):
                    raise ValueError("goal encoder must return a finite nonzero bounded vector")
                vectors.append(value.detach().cpu().float().numpy())
            goals = np.stack(vectors)
            if not np.isfinite(frames).all() or not np.isfinite(goals).all():
                raise ValueError("nonfinite float32 history features")
        after = [tensor_hash(module) for module in (vision_encoder, goal_encoder)]
        if after != before:
            raise RuntimeError("frozen encoder tensors changed during feature extraction")
        identity = {
            "version": FEATURE_VERSION, "corpus_sha256": digest, "mask_version": MASK_VERSION,
            "annotation_version": SCENE_SUITE_VERSION, "pooling": "mean_patch.v1",
            "vision_sha256": before[0], "goal_encoder_sha256": before[1],
            "feature_width": frames.shape[1], "goal_width": goals.shape[1],
            "event_names": list(corpus.event_names), "frames_sha256": _array_sha(frames),
            "goals_sha256": _array_sha(goals), "goal_texts_sha256": _sha(texts),
            "encoder_settings": {
                "vision_class": type(vision_encoder).__qualname__,
                "image_size": getattr(vision_encoder, "image_size", None),
                "goal_class": type(goal_encoder).__qualname__,
                "max_tokens": getattr(goal_encoder, "max_tokens", None),
            },
        }
        np.save(output / "frames.npy", frames, allow_pickle=False)
        np.save(output / "goals.npy", goals, allow_pickle=False)
        atomic_json(output / "features.json", {"identity": identity, "goal_texts": texts,
                                              "frame_count": len(frames)})
        pinned, files = _pin(output)
        return {"feature_sha256": pinned, "identity": identity, "files": files,
                "frames": len(frames), "goals": len(goals), "frame_batch_size": frame_batch_size,
                "feature_payload_bytes": frames.nbytes + goals.nbytes,
                "encoders_unchanged": True, "updates": 0, "model_fit": False,
                "ppo_eligible": False, "test_used": False}
    finally:
        for module, mode in modes:
            module.training = mode
        if saved_cache is not None:
            language_cache.clear()
            language_cache.update(saved_cache)


def load_history_features(directory, corpus, *, expected_sha256):
    """Require an external pin and the independently replay-validated corpus."""
    directory = Path(directory).resolve(strict=True)
    if _pin(directory)[0] != expected_sha256:
        raise ValueError("history feature SHA256 mismatch")
    metadata = _bundle_json(directory / "features.json", MAX_METADATA_BYTES)
    identity, texts = metadata["identity"], metadata["goal_texts"]
    for key, value in (("version", FEATURE_VERSION), ("corpus_sha256", _corpus_identity(corpus)),
                       ("mask_version", MASK_VERSION), ("annotation_version", SCENE_SUITE_VERSION),
                       ("pooling", "mean_patch.v1"), ("event_names", list(corpus.event_names))):
        if identity.get(key) != value:
            raise ValueError(f"history feature {key} identity mismatch")
    if texts != _goal_texts(corpus) or identity.get("goal_texts_sha256") != _sha(texts):
        raise ValueError("history feature explicit goals mismatch")
    frames = np.load(directory / "frames.npy", allow_pickle=False, mmap_mode="r")
    goals = np.load(directory / "goals.npy", allow_pickle=False, mmap_mode="r")
    for name, array, count, width in (("frames", frames, len(corpus.frames), "feature_width"),
                                      ("goals", goals, len(texts), "goal_width")):
        if (array.dtype != np.float32 or array.ndim != 2
                or array.shape != (count, identity.get(width)) or not 1 <= array.shape[1] <= 4096
                or array.nbytes > MAX_FEATURE_BYTES or not np.isfinite(array).all()
                or identity.get(f"{name}_sha256") != _array_sha(array)):
            raise ValueError(f"invalid bounded history {name} features")
    if np.any(~np.any(goals != 0, axis=1)) or metadata["frame_count"] != len(frames):
        raise ValueError("invalid history explicit goals or frame order")
    for name in ("vision_sha256", "goal_encoder_sha256"):
        value = identity.get(name)
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError("invalid history encoder identity")
    return HistoryFeatureSet(frames, goals, texts, identity, directory=directory)
