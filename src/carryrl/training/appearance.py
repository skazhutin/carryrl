"""Pinned, paired TRAIN RGB views, with no policy or optimizer preparation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic_suite import SCORING_VERSION, SUITE_VERSION
from carryrl.envs.presentation import PresentationProfile, present_frame
from carryrl.training.imitation import (
    MEMORY_COUNTERPART_SOURCE,
    DemonstrationDataset,
    DemonstrationEpisode,
    _archive_rgb_identities,
    _bundle_file_identity,
    _bundle_json,
    _validate_replacement_trace,
    augment_demonstrations,
    demonstration_episode_digest,
    validate_dataset_runtime_holdout,
)
from carryrl.training.run import atomic_json
from carryrl.training.sequence_config import SequenceRunConfig

FORMAT = "carryrl.appearance_pair.v1"
ARMS = {"control": "slate", "theme_shift": "sand"}
DEFAULT_FRAME_BYTES = 256 * 1024**2
JSON_LIMIT = 64 * 1024**2
CONFIG_FIELDS = ("agent", "tasks", "episode_instructions", "max_episode_steps",
                 "env_image_size", "level_bank_seed")


def _combined_identity(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _regular_directory(directory):
    path = Path(directory).absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("appearance directory must not traverse symlinks")
    return path.resolve(strict=True)


def _identities(directory, max_frame_bytes):
    limits = {"bundle.json": JSON_LIMIT, "frames.npz": max_frame_bytes * 2 + 65536}
    result = {}
    for name, limit in limits.items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("appearance bundle requires regular owned files")
        result[name] = _bundle_file_identity(path, limit)
    return result


def _episode(row):
    if not isinstance(row, dict):
        raise ValueError("appearance episode must be a JSON object")
    names = {field.name for field in fields(DemonstrationEpisode)}
    try:
        values = {key: value for key, value in row.items() if key in names}
        values["frame_indices"] = tuple(row["frame_indices"])
        values["actions"] = tuple(PhysicalAction.from_dict(value) for value in row["actions"])
        if "executed_actions" in row:
            values["executed_actions"] = tuple(PhysicalAction.from_dict(value)
                                                for value in row["executed_actions"])
        result = DemonstrationEpisode(**values)
        if (type(result.seed) is not int or not 0 <= result.seed < 2**32
                or type(result.difficulty) not in (float, int) or result.difficulty != 1.0
                or result.to_dict() != row):
            raise ValueError("appearance episode metadata would change on import")
        return result
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("malformed appearance episode") from error


def _validate_sources(dataset, config):
    if (not dataset.episodes or config.env_image_size != 96
            or config.curriculum_difficulty != 1.0 or config.evaluation_difficulty != 1.0
            or set(row.task for row in dataset.episodes) != set(config.tasks)):
        raise ValueError("appearance requires all configured tasks at full TRAIN difficulty")
    for row in dataset.episodes:
        dataset._validate_episode(row)
        variants = config.episode_instructions.get(row.task) or ()
        if (row.presentation is not None or row.native_size is not None
                or row.collection_type is not None or row.difficulty != 1.0
                or row.scoring_version != SCORING_VERSION
                or row.level_bank_seed != config.level_bank_seed
                or len(row) > config.max_episode_steps
                or (variants and row.instruction not in variants)
                or (not variants and row.instruction is not None)
                or any(dataset.frames[index].shape != (96, 96, 3)
                       for index in (*row.frame_indices, row.final_frame_index))):
            raise ValueError("appearance source must retain canonical 96 TRAIN provenance")
        if row.source != MEMORY_COUNTERPART_SOURCE:
            _validate_replacement_trace(dataset, row, config.max_episode_steps)
    validate_dataset_runtime_holdout(dataset, level_bank_seed=config.level_bank_seed,
                                     evaluation_difficulty=1.0)


@dataclass
class PreparedAppearanceReuse:
    directory: Path
    dataset: DemonstrationDataset
    files: dict
    summary: dict

    @classmethod
    def prepare(cls, directory, *, expected_sha256, arm, config,
                max_frame_bytes=DEFAULT_FRAME_BYTES):
        if (type(max_frame_bytes) is not int or max_frame_bytes <= 0
                or arm not in ARMS or not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or any(char not in "0123456789abcdef" for char in expected_sha256)):
            raise ValueError("appearance reuse requires a trusted SHA256, arm and byte budget")
        directory = _regular_directory(directory)
        identities = _identities(directory, max_frame_bytes)
        if _combined_identity(identities) != expected_sha256:
            raise ValueError("appearance bundle SHA256 mismatch")
        bundle = _bundle_json(directory / "bundle.json", JSON_LIMIT)
        if (set(bundle) != {"format", "suite_version", "scoring_version", "split",
                           "source_config", "seed", "unique_frames", "frame_bytes",
                           "sources", "arms"}
                or bundle["format"] != FORMAT or bundle["suite_version"] != SUITE_VERSION
                or bundle["scoring_version"] != SCORING_VERSION or bundle["split"] != "train"
                or type(bundle["seed"]) is not int or not 0 <= bundle["seed"] < 2**32):
            raise ValueError("appearance format/suite/scoring/split mismatch")
        source_config = SequenceRunConfig.from_dict(bundle["source_config"])
        normalized = json.loads(json.dumps(source_config.to_dict(), allow_nan=False))
        if normalized != bundle["source_config"]:
            raise ValueError("appearance source configuration must be complete and finite")
        current = json.loads(json.dumps(config.to_dict(), allow_nan=False))
        if any(bundle["source_config"].get(key) != current.get(key) for key in CONFIG_FIELDS):
            raise ValueError("appearance source configuration differs")
        count = bundle["unique_frames"]
        if (type(count) is not int or not 0 < count <= 100000
                or type(bundle["frame_bytes"]) is not int
                or not 0 < bundle["frame_bytes"] <= max_frame_bytes):
            raise ValueError("appearance frame counts or byte budget invalid")
        _, logical_bytes = _archive_rgb_identities(directory / "frames.npz", count,
                                                   max_logical_bytes=max_frame_bytes)
        if logical_bytes != bundle["frame_bytes"]:
            raise ValueError("appearance RGB byte count mismatch")
        shared = DemonstrationDataset(max_frame_bytes=max_frame_bytes)
        with np.load(directory / "frames.npz", allow_pickle=False) as archive:
            for index in range(count):
                frame = archive[f"frame_{index:06}"]
                if shared.intern_frame(frame) != index:
                    raise ValueError("appearance RGB indices must be uniquely interned")
        if not isinstance(bundle["sources"], list) or not bundle["sources"]:
            raise ValueError("appearance requires complete ordered source episodes")
        for row in bundle["sources"]:
            shared.add_episode(_episode(row))
        _validate_sources(shared, source_config)
        sources = tuple(shared.episodes)
        if not isinstance(bundle["arms"], dict) or set(bundle["arms"]) != set(ARMS):
            raise ValueError("appearance requires both paired arms")
        used = {index for row in sources for index in (*row.frame_indices, row.final_frame_index)}
        views = {}
        for name, theme in ARMS.items():
            records = bundle["arms"][name]
            if not isinstance(records, list) or len(records) != len(sources):
                raise ValueError("appearance arm must include every ordered source exactly once")
            views[name] = []
            rng = np.random.default_rng(bundle["seed"])
            for slot, (source, record) in enumerate(zip(sources, records, strict=True)):
                if (not isinstance(record, dict) or set(record) != {"parent_slot", "episode"}
                        or type(record["parent_slot"]) is not int or record["parent_slot"] != slot):
                    raise ValueError("appearance parent ordering mismatch")
                view = _episode(record["episode"])
                try:
                    profile = PresentationProfile(**(view.presentation or {}))
                except TypeError as error:
                    raise ValueError("appearance view requires a complete profile") from error
                expected_profile = PresentationProfile(native_size=96,
                    theme=str(rng.choice((theme,))), hud_layout=str(rng.choice(("left",))),
                    texture_seed=int(rng.integers(2**31)), textures=False, canvas_padding=8)
                if (profile != expected_profile or profile.native_size != 96
                        or profile.theme != theme
                        or profile.hud_layout != "left" or profile.textures is not False
                        or profile.canvas_padding != 8
                        or replace(view, frame_indices=source.frame_indices,
                                   final_frame_index=source.final_frame_index,
                                   presentation=None) != source):
                    raise ValueError("appearance view metadata or fixed profile mismatch")
                shared._validate_episode(view)
                for raw, styled in zip((*source.frame_indices, source.final_frame_index),
                                      (*view.frame_indices, view.final_frame_index), strict=True):
                    if not np.array_equal(shared.frames[styled],
                                          present_frame(shared.frames[raw], profile)):
                        raise ValueError("appearance RGB differs from parent transformation")
                    used.add(styled)
                if name == "theme_shift" and replace(profile, theme="slate") != (
                        PresentationProfile(**views["control"][slot].presentation)):
                    raise ValueError("appearance paired geometry differs")
                views[name].append(view)
        if used != set(range(count)):
            raise ValueError("appearance RGB archive contains unreferenced frames")
        dataset = DemonstrationDataset(max_frame_bytes=max_frame_bytes)
        for row in (*sources, *views[arm]):
            mapping = {index: dataset.intern_frame(shared.frames[index])
                       for index in (*row.frame_indices, row.final_frame_index)}
            dataset.add_episode(replace(row,
                frame_indices=tuple(mapping[index] for index in row.frame_indices),
                final_frame_index=mapping[row.final_frame_index]))
        result = cls(directory, dataset, identities, {
            "format": FORMAT, "source_bundle_sha256": expected_sha256, "arm": arm,
            "source_difficulty": {"curriculum_difficulty": 1.0, "evaluation_difficulty": 1.0},
            "episodes": len(dataset.episodes), "decisions": sum(map(len, dataset.episodes)),
            "source_rows": len(sources), "posthoc_copies": len(sources),
            "ordered_origin_sha256": [demonstration_episode_digest(shared, row)
                                      for row in sources],
            "frame_bytes": dataset.frame_bytes, "shared_frame_bytes": logical_bytes,
            "unique_frames": len(dataset.frames), "count_guard_passed": True,
            "runtime_holdout_guard_passed": True, "exact_complete_corpus": True,
            "collection_skipped": True, "copy_is_new_rollout": False,
        })
        result.assert_unchanged()
        return result

    def assert_unchanged(self):
        if (_regular_directory(self.directory) != self.directory
                or _identities(self.directory, self.summary["shared_frame_bytes"]) != self.files):
            raise ValueError("appearance source changed after validation")


def prepare_appearance_bundle(dataset, output, *, config, seed=8288,
                              max_frame_bytes=DEFAULT_FRAME_BYTES):
    """Publish two equal-count canonical-plus-view arms after authenticating real TRAIN RGB."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("appearance seed must be a uint32 integer")
    _validate_sources(dataset, config)
    shared = DemonstrationDataset(max_frame_bytes=max_frame_bytes)
    shared.extend(dataset)
    sources = [row.to_dict() for row in shared.episodes]
    arms = {}
    for name, theme in ARMS.items():
        copies = augment_demonstrations(dataset, seed=seed, themes=(theme,),
            hud_layouts=("left",), textures=False, canvas_padding=8,
            max_frame_bytes=max_frame_bytes)
        start = len(shared.episodes)
        shared.extend(copies)
        arms[name] = [{"parent_slot": slot, "episode": row.to_dict()}
                      for slot, row in enumerate(shared.episodes[start:])]
    output = Path(output).absolute()
    if any(parent.is_symlink() for parent in (output, *output.parents)):
        raise ValueError("appearance output must not traverse symlinks")
    output.mkdir(parents=True, exist_ok=False)
    with ZipFile(output / "frames.npz", "x", compression=ZIP_DEFLATED) as archive:
        for index, frame in enumerate(shared.frames):
            with archive.open(f"frame_{index:06}.npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, frame, allow_pickle=False)
    atomic_json(output / "bundle.json", {
        "format": FORMAT, "suite_version": SUITE_VERSION, "scoring_version": SCORING_VERSION,
        "split": "train", "source_config": config.to_dict(), "seed": seed,
        "unique_frames": len(shared.frames), "frame_bytes": shared.frame_bytes,
        "sources": sources, "arms": arms,
    })
    digest = _combined_identity(_identities(output, max_frame_bytes))
    loaded = {name: PreparedAppearanceReuse.prepare(output, expected_sha256=digest,
              arm=name, config=config, max_frame_bytes=max_frame_bytes) for name in ARMS}
    origin_digests = [demonstration_episode_digest(dataset, row) for row in dataset.episodes]
    if any(reuse.summary["ordered_origin_sha256"] != origin_digests for reuse in loaded.values()):
        raise ValueError("appearance loader changed the actual ordered source RGB/actions")
    report = {"status": "prepared_not_trained", "format": FORMAT,
              "source_bundle_sha256": digest, "updates": 0, "model_loaded": False,
              "checkpoint_created": False, "loader_roundtrip_verified": True,
              "actual_source_rows_identity_verified": True,
              "arms": {name: reuse.summary for name, reuse in loaded.items()}}
    atomic_json(output / "audit.json", report)
    atomic_json(output / "status.json", report)
    return report
