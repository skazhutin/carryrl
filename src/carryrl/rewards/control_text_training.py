"""Explicit future head fitting and scenario-cluster calibration; never auto-runs."""

import copy
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from carryrl.rewards.control_text import (
    CANDIDATES,
    CONDITIONAL_HEAD_ARCHITECTURE,
    GROUNDED_HEAD_ARCHITECTURE,
    LOCAL_GROUNDED_HEAD_ARCHITECTURE,
    MODES,
    PAIR_GEOMETRY_HEAD_ARCHITECTURE,
    PAIRS,
    STATUSES,
    ConditionalTextControlHead,
    FrozenTextControlReward,
    GroundedTextControlHead,
    LocalGroundedTextControlHead,
    PairGeometryTextControlHead,
    ResidualTextControlHead,
    TextControlHead,
    calibration_gates,
    content_hash,
    save_text_control_artifact,
)

_GROUNDED_ARCHITECTURES = (
    GROUNDED_HEAD_ARCHITECTURE, LOCAL_GROUNDED_HEAD_ARCHITECTURE, PAIR_GEOMETRY_HEAD_ARCHITECTURE,
    CONDITIONAL_HEAD_ARCHITECTURE,
)

CONDITIONAL_POPULATION_CONTRACT = "supported_decisions.v1"
CONDITIONAL_DECODER_VERSION = "independent_semantics_dedup.v2"


def read_control_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    payload = (directory / "records.jsonl").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest.get("records_sha256"):
        raise ValueError("text dataset records hash mismatch")
    records = [json.loads(line) for line in payload.splitlines()]
    groups, mappings, semantic_mappings, templates = {}, {}, {}, {}
    semantic_groups = {}
    shared_grammar = (
        manifest.get("version") in {"carryrl.control_text_data.v4", "carryrl.control_text_data.v5",
                                    "carryrl.control_text_data.v6", "carryrl.control_text_data.v7"}
        and manifest.get("template_families_disjoint") is False
    )
    for row in records:
        if row["split"] not in ("train", "dev", "calibration", "sealed_test"):
            raise ValueError("unknown text data split")
        identities = [
            (groups, row["group_id"]),
            (mappings, row["mapping_id"]),
            (semantic_mappings, row.get("semantic_mapping_id", row["mapping_id"])),
        ]
        if not shared_grammar or row["template_family"] != "shared-core":
            identities.append((templates, row["template_family"]))
        if manifest.get("version") in {"carryrl.control_text_data.v8",
                                       "carryrl.control_text_data.v9",
                                       "carryrl.control_text_data.v10",
                                       "carryrl.control_text_data.v11",
                                       "carryrl.control_text_data.v12",
                                       "carryrl.control_text_data.v13"}:
            if (manifest.get("template_families_disjoint") is not True
                    or manifest.get("internal_grammar_families_disjoint") is not True):
                raise ValueError(
                    "v8/v9/v10/v11/v12 require split-owned templates and grammar families"
                )
            families = row.get("grammar_families")
            if (not isinstance(families, list) or not families
                    or any(not isinstance(family, str) or not family for family in families)):
                raise ValueError("v8/v9/v10/v11/v12 require actual grammar composition identities")
            identities.extend((templates, family) for family in families)
        for seen, key in identities:
            if key in seen and seen[key] != row["split"]:
                raise ValueError("text dataset split leakage")
            seen[key] = row["split"]
        if manifest.get("version") in {
            "carryrl.control_text_data.v3", "carryrl.control_text_data.v4",
            "carryrl.control_text_data.v5", "carryrl.control_text_data.v6",
            "carryrl.control_text_data.v7", "carryrl.control_text_data.v8",
            "carryrl.control_text_data.v9", "carryrl.control_text_data.v10",
            "carryrl.control_text_data.v11", "carryrl.control_text_data.v12",
            "carryrl.control_text_data.v13",
        }:
            from carryrl.rewards.control_text_data import semantic_mapping_identity

            identity = semantic_mapping_identity(row["labels"], row["declared_candidates"])
            if identity != row["semantic_mapping_id"]:
                raise ValueError("text dataset semantic identity mismatch")
            if identity in semantic_groups and semantic_groups[identity] != row["group_id"]:
                raise ValueError("text dataset duplicates effective semantic groups")
            semantic_groups[identity] = row["group_id"]
        if tuple(label["candidate"] for label in row["labels"]) != CANDIDATES:
            raise ValueError("text dataset candidates differ from model")
    if len(records) != manifest.get("records"):
        raise ValueError("text dataset record count mismatch")
    return manifest, records


def verify_grounded_text_inputs(tokenizer, records, *, deadline=None):
    """Offline annotation/input consistency, before encoding or fitting any head.

    Labels are used only to check the supervised corpus's lexical transmission;
    they never supply inference masks or predictions. Sealed inputs are skipped.
    A complete footer may license unmentioned negatives, so they are not required.
    """
    from carryrl.rewards.control_text_grounding import build_grounding

    count = mentions = 0
    failures = []
    candidate_index = {candidate: index for index, candidate in enumerate(CANDIDATES)}
    for row in records:
        if row["split"] == "sealed_test":
            continue
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("text input transmission preflight budget exhausted")
        expected = {label["candidate"] for label in row["labels"]
                    if label["status"] == "supported"}
        spec = row.get("specification", {})
        expected.update(candidate for op in spec.get("operations", [])
                        for candidate in op["candidates"])
        expected.update(spec.get("uncertain", []))
        present = build_grounding(row["text"], tokenizer).any(-1)
        missing = sorted(candidate for candidate in expected
                         if not present[candidate_index[candidate]])
        if missing and len(failures) < 8:
            failures.append({"group_id": row.get("group_id"), "split": row["split"],
                             "candidates": missing})
        count += 1
        mentions += len(expected)
    if failures:
        raise ValueError(f"grounded input transmission lost named control mentions: {failures}")
    return {"passed": True, "records": count, "expected_control_mentions": mentions,
            "missing_mentions": 0, "sealed_inputs_used": False,
            "scope": "offline corpus/input consistency only; not model accuracy"}


class DiskTokenFeatures:
    """Bounded, authenticated float16 E5 features; RAM contains one input only.

    Cache identity is tied to the source dataset and encoder state. The original
    encoder remains the artifact source; this adapter is only used while fitting.
    """

    def __init__(self, directory, *, deadline=None):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "manifest.json").read_text())
        if self.metadata.get("status") != "complete":
            raise ValueError("incomplete text token cache")
        self.rows = self.metadata["rows"]
        self.deadline = deadline
        with (self.directory / "features.f16").open("rb") as stream:
            if (
                hashlib.file_digest(stream, "sha256").hexdigest()
                != self.metadata["features_sha256"]
            ):
                raise ValueError("text token cache hash mismatch")
        self.data = np.memmap(self.directory / "features.f16", dtype=np.float16, mode="r")
        if self.data.nbytes != self.metadata["actual_bytes"]:
            raise ValueError("text token cache size mismatch")

    def eval(self):
        return self

    def requires_grad_(self, value):
        if value:
            raise ValueError("text features must remain frozen")
        return self

    def encode_instruction_tokens(self, text):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("text head wall-clock budget exhausted")
        row = self.rows[hashlib.sha256(text.encode()).hexdigest()]
        values = self.data[row["offset"] : row["offset"] + row["length"] * 384]
        return torch.from_numpy(values.reshape(row["length"], 384).astype(np.float32))


def load_verified_disk_token_features(
    directory, *, encoder, records, dataset_manifest, deadline=None, max_bytes=8 * 1024**3
):
    """Read-only reuse: exact E5/text identities, candidate order, masks and byte hash."""
    from carryrl.rewards.control_text import encoder_state_hash

    features = DiskTokenFeatures(directory, deadline=deadline)
    metadata = features.metadata
    if (
        metadata.get("dataset_sha256") != dataset_manifest["records_sha256"]
        or metadata.get("encoder_state_hash") != encoder_state_hash(encoder.state_dict())
        or dataset_manifest.get("candidates") != list(CANDIDATES)
        or metadata["actual_bytes"] > max_bytes
        or metadata.get("dtype") != "float16"
        or metadata.get("sealed_test_encoded") is not False
    ):
        raise ValueError("reused text feature cache identity/budget mismatch")
    required = {
        hashlib.sha256(row["text"].encode()).hexdigest(): row["text"]
        for row in records
        if row["split"] != "sealed_test"
    }
    if set(required) != set(features.rows):
        raise ValueError("reused text feature cache input text set mismatch")
    end = 0
    # Encoding writes unique instructions in stable descending character length.
    # Tie every span to that exact text order, not merely to a valid set of spans:
    # exchanging equal-token-length rows must not silently exchange embeddings.
    for key in sorted(required, key=lambda key: len(required[key]), reverse=True):
        row = features.rows[key]
        if (
            type(row["offset"]) is not int
            or type(row["length"]) is not int
            or row["offset"] != end
            or not 1 <= row["length"] <= 512
        ):
            raise ValueError("invalid reused text-to-feature offsets or unpadded token lengths")
        end += row["length"] * 384
    if end * 2 != metadata["actual_bytes"]:
        raise ValueError("reused feature token spans do not cover the authenticated bytes")
    # Verify every padding-mask length with the pinned tokenizer, without E5 forwards.
    if hasattr(encoder, "tokenizer"):
        items = list(required.items())
        for offset in range(0, len(items), 64):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("text cache verification budget exhausted")
            batch = items[offset : offset + 64]
            tokenized = encoder.tokenizer(
                ["query: " + text for _, text in batch], padding=False, truncation=False
            )
            if any(
                len(ids) != features.rows[key]["length"]
                for (key, _), ids in zip(batch, tokenized["input_ids"], strict=True)
            ):
                raise ValueError("reused cache token lengths differ from exact instruction masks")
    features.tokenizer = getattr(encoder, "tokenizer", None)
    features.verification = {
        "read_only": True,
        "source_manifest_sha256": hashlib.sha256(
            (Path(directory) / "manifest.json").read_bytes()
        ).hexdigest(),
        "features_sha256": metadata["features_sha256"],
        "input_texts": len(required),
        "dataset_encoder_candidate_order_verified": True,
        "unpadded_token_spans_verified": True,
        "text_to_feature_offsets_verified": True,
        "tokenizer_mask_lengths_verified": hasattr(encoder, "tokenizer"),
    }
    return features


@torch.no_grad()
def _encode_feature_batch(encoder, texts):
    """Batched frozen E5 without truncation or a corpus-sized Python cache."""
    if getattr(encoder, "_awaiting_checkpoint_state", False):
        raise RuntimeError("local E5 graph requires strict checkpoint state restoration")
    if not hasattr(encoder, "tokenizer") or not hasattr(encoder, "backbone"):
        return [encoder.encode_instruction_tokens(text) for text in texts]
    tokens = encoder.tokenizer(
        ["query: " + text for text in texts], return_tensors="pt", padding=True, truncation=False
    )
    if tokens["input_ids"].shape[-1] > encoder.max_tokens:
        raise ValueError("instruction exceeds frozen E5 context (text is never truncated)")
    device = next(encoder.backbone.parameters()).device
    tokens = {name: value.to(device) for name, value in tokens.items()}
    encoder.backbone.eval()
    hidden = encoder.backbone(**tokens, return_dict=True).last_hidden_state
    mask = tokens["attention_mask"].bool()
    if hidden.shape != (*mask.shape, 384) or not mask.any(dim=1).all():
        raise ValueError("invalid frozen E5 batch output")
    return [hidden[index, mask[index]].detach().cpu() for index in range(len(texts))]


def prepare_disk_token_features(
    encoder,
    records,
    directory,
    *,
    dataset_sha256,
    max_bytes=8 * 1024**3,
    deadline=None,
    probe_records=8,
    encoder_batch_size=8,
):
    """Encode once with a small timed probe; reject unfit budgets before full work.

    Failed and partial caches are preserved. Sealed examples are never encoded.
    No E5 feature corpus is retained in RAM, including during repeated epochs.
    """
    from carryrl.rewards.control_text import encoder_state_hash

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rows = [row for row in records if row["split"] != "sealed_test"]
    # Long examples first make the bounded probe conservative for dense references.
    rows.sort(key=lambda row: len(row["text"]), reverse=True)
    rows = list({row["text"]: row for row in rows}.values())
    if not rows or not 1 <= encoder_batch_size <= 16:
        raise ValueError("nonempty fitting data and encoder_batch_size in 1..16 required")
    metadata = {
        "status": "preparing",
        "dataset_sha256": dataset_sha256,
        "encoder_state_hash": encoder_state_hash(encoder.state_dict()),
        "dtype": "float16",
        "rows": {},
        "actual_bytes": 0,
        "max_bytes": max_bytes,
        "sealed_test_encoded": False,
    }
    start = time.monotonic()
    encoder.eval().requires_grad_(False)
    try:
        with (directory / "features.f16").open("wb") as stream:
            digest = hashlib.sha256()
            processed = 0
            for offset in range(0, len(rows), encoder_batch_size):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("text feature extraction budget exhausted")
                batch = rows[offset : offset + encoder_batch_size]
                encoded = _encode_feature_batch(encoder, [row["text"] for row in batch])
                for row, features in zip(batch, encoded, strict=True):
                    features = features.detach().cpu()
                    if (
                        features.ndim != 2
                        or features.shape[-1] != 384
                        or not 0 < len(features) <= 512
                    ):
                        raise ValueError("invalid frozen text features")
                    if not torch.isfinite(features).all():
                        raise ValueError("nonfinite frozen text features")
                    payload = features.to(torch.float16).numpy().tobytes()
                    if metadata["actual_bytes"] + len(payload) > max_bytes:
                        raise RuntimeError("text feature disk byte budget exceeded")
                    key = hashlib.sha256(row["text"].encode()).hexdigest()
                    metadata["rows"][key] = {
                        "offset": metadata["actual_bytes"] // 2,
                        "length": len(features),
                    }
                    stream.write(payload)
                    digest.update(payload)
                    metadata["actual_bytes"] += len(payload)
                    processed += 1
                if "throughput_probe" not in metadata and processed >= min(
                    probe_records, len(rows)
                ):
                    elapsed = time.monotonic() - start
                    estimate = elapsed * len(rows) / processed
                    metadata["throughput_probe"] = {
                        "records": processed,
                        "seconds": elapsed,
                        "encoder_batch_size": encoder_batch_size,
                        "estimated_encoding_seconds": estimate,
                        "estimated_max_bytes": len(rows) * 512 * 384 * 2,
                    }
                    (directory / "probe.json").write_text(json.dumps(metadata["throughput_probe"]))
                    if deadline is not None and estimate * 1.5 > deadline - time.monotonic():
                        raise TimeoutError("throughput probe exceeds remaining text fit budget")
            metadata["features_sha256"] = digest.hexdigest()
        metadata["status"] = "complete"
        metadata["encoding_seconds"] = time.monotonic() - start
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (directory / "manifest.json").write_text(json.dumps(metadata, sort_keys=True) + "\n")
        if hasattr(encoder, "clear_cache"):
            encoder.clear_cache()
    features = DiskTokenFeatures(directory, deadline=deadline)
    features.tokenizer = getattr(encoder, "tokenizer", None)
    return features


def _targets(rows, device, *, conditional=False):
    status = torch.tensor(
        [[STATUSES.index(label["status"]) for label in row["labels"]] for row in rows],
        device=device,
    )
    mode = torch.tensor(
        [[MODES.index(label["mode"]) for label in row["labels"]] for row in rows], device=device
    )
    pair = []
    for row in rows:
        labels = row["labels"]
        pair.append(
            [
                [
                    2
                    if left["status"] == "unknown" or right["status"] == "unknown"
                    else (
                        0
                        if left["alias_group"] > 0 and left["alias_group"] == right["alias_group"]
                        else 1
                    )
                    for right in labels
                ]
                for left in labels
            ]
        )
    targets = {"status": status, "mode": mode, "pair": torch.tensor(pair, device=device)}
    if conditional:
        supported = status == STATUSES.index("supported")
        discrete = torch.tensor(
            [candidate.startswith(("key:", "mouse:")) for candidate in CANDIDATES],
            device=device,
        )
        targets["mode_mask"] = supported & discrete & (mode < 2)
        targets["pair_mask"] = supported[:, :, None] & supported[:, None, :]
    return targets


def _batch(encoder, rows, device):
    encoded = [encoder.encode_instruction_tokens(row["text"]) for row in rows]
    length = max(len(tokens) for tokens in encoded)
    tokens = torch.zeros(len(rows), length, encoded[0].shape[-1], device=device)
    mask = torch.ones(len(rows), length, dtype=torch.bool, device=device)
    for index, features in enumerate(encoded):
        tokens[index, : len(features)] = features.to(device)
        mask[index, : len(features)] = False
    return tokens, mask



def _head_forward(encoder, head, rows, tokens, mask):
    if head.architecture not in _GROUNDED_ARCHITECTURES:
        return head(tokens, mask)
    from carryrl.rewards.control_text_grounding import build_grounding

    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("grounded text head requires its pinned offset tokenizer")
    grounding = torch.zeros(len(rows), len(CANDIDATES), tokens.shape[1],
                            dtype=torch.bool, device=tokens.device)
    for index, row in enumerate(rows):
        length = int((~mask[index]).sum())
        grounding[index, :, :length] = build_grounding(
            row["text"], tokenizer, expected_tokens=length
        ).to(tokens.device)
    return head(tokens, mask, grounding=grounding)


def _prior_correction(head, train):
    if head.architecture not in _GROUNDED_ARCHITECTURES:
        return {}
    weights = class_balance_from_train(
        train, conditional=head.architecture == CONDITIONAL_HEAD_ARCHITECTURE,
    )["weights"]
    return {f"{kind}_log_class_weights": [math.log(value) for value in values]
            for kind, values in weights.items()}


def _loss_arrays(logits, targets, kind, unique_pairs):
    values, labels = logits[kind], targets[kind]
    conditional = logits["mode"].shape[-1] == 2
    decision_mask = targets.get(f"{kind}_mask") if conditional else None
    if conditional and kind in ("mode", "pair") and decision_mask is None:
        raise ValueError("conditional logits require explicit supported-decision target masks")
    if kind == "pair" and (unique_pairs or conditional):
        left, right = torch.triu_indices(len(CANDIDATES), len(CANDIDATES), 1, device=values.device)
        values, labels = values[:, left, right], labels[:, left, right]
        if decision_mask is not None:
            decision_mask = decision_mask[:, left, right]
    values, labels = values.reshape(-1, values.shape[-1]), labels.reshape(-1)
    if decision_mask is not None:
        selected = decision_mask.reshape(-1)
        values, labels = values[selected], labels[selected]
    return values, labels


def class_balance_from_train(rows, *, conditional=False):
    """Fixed inverse-frequency weights; only TRAIN and unique off-diagonal pairs."""
    if conditional:
        return _conditional_class_balance(rows)
    counts = {kind: [0, 0, 0] for kind in ("status", "mode", "pair")}
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("class balancing requires nonempty TRAIN rows only")
    for row in rows:
        groups, known = {}, 0
        for label in row["labels"]:
            counts["status"][STATUSES.index(label["status"])] += 1
            counts["mode"][MODES.index(label["mode"])] += 1
            known += label["status"] != "unknown"
            if label["alias_group"] > 0:
                group = label["alias_group"]
                groups[group] = groups.get(group, 0) + 1
        same = sum(size * (size - 1) // 2 for size in groups.values())
        different = known * (known - 1) // 2 - same
        unknown = len(CANDIDATES) * (len(CANDIDATES) - 1) // 2 - same - different
        for index, value in enumerate((same, different, unknown)):
            counts["pair"][index] += value
    if any(value <= 0 for values in counts.values() for value in values):
        raise ValueError("balanced text fit requires all three classes in every TRAIN head")
    weights = {
        kind: [sum(values) / (3 * count) for count in values] for kind, values in counts.items()
    }
    return {
        "counts": counts,
        "weights": weights,
        "source_split": "train",
        "pair_scope": "unique_off_diagonal",
        "reduction": "weighted_sum_divided_by_decision_count",
    }


def _conditional_class_balance(rows):
    """Count precisely the same meaningful decisions used by the v6 objective."""
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("class balancing requires nonempty TRAIN rows only")
    counts = {"status": [0, 0, 0], "mode": [0, 0], "pair": [0, 0]}
    for offset in range(0, len(rows), 32):
        batch = rows[offset:offset + 32]
        targets = _targets(batch, "cpu", conditional=True)
        logits = {
            "status": torch.zeros(len(batch), len(CANDIDATES), 3),
            "mode": torch.zeros(len(batch), len(CANDIDATES), 2),
            "pair": torch.zeros(len(batch), len(CANDIDATES), len(CANDIDATES), 2),
        }
        for kind in counts:
            _, labels = _loss_arrays(logits, targets, kind, True)
            for index in range(len(counts[kind])):
                counts[kind][index] += int((labels == index).sum())
    if any(count <= 0 for values in counts.values() for count in values):
        raise ValueError("conditional fit requires every status class and both conditional classes")
    return {
        "counts": counts,
        "weights": {kind: [sum(values) / (len(values) * count) for count in values]
                    for kind, values in counts.items()},
        "source_split": "train",
        "population_contract": CONDITIONAL_POPULATION_CONTRACT,
        "mode_scope": "gold_supported_discrete_known_mode",
        "pair_scope": "gold_supported_unique_off_diagonal",
        "reduction": "weighted_sum_divided_by_decision_count",
    }


def _conditional_loss_components(logits, targets, class_weights=None):
    components = {}
    for kind in ("status", "mode", "pair"):
        values, labels = _loss_arrays(logits, targets, kind, True)
        if not labels.numel():
            components[kind] = (values.sum() * 0.0, 0)
            continue
        losses = F.cross_entropy(values, labels, reduction="none")
        if class_weights is not None:
            weights = values.new_tensor(class_weights[kind])
            losses = losses * weights[labels]
        components[kind] = (losses.sum(), labels.numel())
    return components


def _mean_decision_components(components):
    active = [total / count for total, count in components.values() if count]
    if active:
        return sum(active) / len(active)
    return sum(total for total, _ in components.values())


def head_loss(logits, targets, *, class_weights=None, unique_pairs=False):
    if logits["mode"].shape[-1] == 2:
        return _mean_decision_components(
            _conditional_loss_components(logits, targets, class_weights),
        )
    losses = []
    for kind in ("status", "mode", "pair"):
        values, labels = _loss_arrays(logits, targets, kind, unique_pairs)
        losses_by_decision = F.cross_entropy(values, labels, reduction="none")
        if class_weights is not None:
            weight = torch.as_tensor(class_weights[kind], device=values.device, dtype=values.dtype)
            losses_by_decision = losses_by_decision * weight[labels]
        losses.append(losses_by_decision.mean())
    return sum(losses) / 3


@torch.no_grad()
def evaluate_head_metrics(
    encoder, head, rows, *, batch_size=32, class_weights=None, unique_pairs=False,
    log_class_weights=None,
):
    if not rows:
        raise ValueError("NLL evaluation requires a nonempty development split")
    if getattr(head, "architecture", None) == CONDITIONAL_HEAD_ARCHITECTURE:
        return _evaluate_conditional_metrics(
            encoder, head, rows, batch_size=batch_size, class_weights=class_weights,
            log_class_weights=log_class_weights,
        )
    device = next(head.parameters()).device
    head.eval()
    selected_total = original_total = 0.0
    slices = {
        kind: [
            {
                "target_count": 0,
                "argmax_correct": 0,
                "confidence99_accepted": 0,
                "confidence99_correct": 0,
            }
            for _ in range(3)
        ]
        for kind in ("status", "mode", "pair")
    }
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        tokens, mask = _batch(encoder, batch, device)
        logits, targets = _head_forward(encoder, head, batch, tokens, mask), _targets(batch, device)
        selection_loss = head_loss(
            logits, targets, class_weights=class_weights, unique_pairs=unique_pairs,
        )
        if log_class_weights:
            logits = {
                kind: values - values.new_tensor(log_class_weights[f"{kind}_log_class_weights"])
                for kind, values in logits.items()
            }
        # Select on the fitted objective before correcting class-weight priors.
        # Applying weights again to corrected posteriors would count them twice.
        selected_total += float(selection_loss) * len(batch)
        original_total += float(head_loss(
            logits, targets, unique_pairs=(getattr(head, "architecture", None)
                                         in _GROUNDED_ARCHITECTURES),
        )) * len(batch)
        for kind in slices:
            values, labels = _loss_arrays(logits, targets, kind, unique_pairs)
            confidence, predicted = values.softmax(-1).max(-1)
            for index, count in enumerate(slices[kind]):
                expected, accepted = labels == index, (predicted == index) & (confidence >= 0.99)
                count["target_count"] += int(expected.sum())
                count["argmax_correct"] += int((expected & (predicted == index)).sum())
                count["confidence99_accepted"] += int(accepted.sum())
                count["confidence99_correct"] += int((expected & accepted).sum())
    names = {"status": STATUSES, "mode": MODES, "pair": PAIRS}
    return {
        "selection_nll": selected_total / len(rows),
        ("unweighted_unique_off_diagonal_nll"
         if getattr(head, "architecture", None) in _GROUNDED_ARCHITECTURES
         else "unweighted_all_pairs_nll"): original_total / len(rows),
        "slices": {
            kind: dict(zip(names[kind], values, strict=True)) for kind, values in slices.items()
        },
        "records": len(rows),
    }


def _evaluate_conditional_metrics(
    encoder, head, rows, *, batch_size, class_weights, log_class_weights,
):
    device = next(head.parameters()).device
    head.eval()
    totals = {kind: [0.0, 0.0, 0] for kind in ("status", "mode", "pair")}
    names = {"status": STATUSES, "mode": MODES[:2], "pair": PAIRS[:2]}
    slices = {kind: {name: {"target_count": 0, "argmax_correct": 0,
                           "confidence99_accepted": 0, "confidence99_correct": 0}
                     for name in classes} for kind, classes in names.items()}
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        tokens, mask = _batch(encoder, batch, device)
        logits = _head_forward(encoder, head, batch, tokens, mask)
        targets = _targets(batch, device, conditional=True)
        for kind in totals:
            values, labels = _loss_arrays(logits, targets, kind, True)
            if not labels.numel():
                continue
            raw_losses = F.cross_entropy(values, labels, reduction="none")
            if class_weights is not None:
                raw_losses = raw_losses * values.new_tensor(class_weights[kind])[labels]
            totals[kind][0] += float(raw_losses.sum())
            if log_class_weights:
                values = values - values.new_tensor(log_class_weights[f"{kind}_log_class_weights"])
            totals[kind][1] += float(F.cross_entropy(values, labels, reduction="sum"))
            totals[kind][2] += labels.numel()
            confidence, predicted = values.softmax(-1).max(-1)
            for index, counts in enumerate(slices[kind].values()):
                expected = labels == index
                accepted = (predicted == index) & (confidence >= 0.99)
                counts["target_count"] += int(expected.sum())
                counts["argmax_correct"] += int((expected & (predicted == index)).sum())
                counts["confidence99_accepted"] += int(accepted.sum())
                counts["confidence99_correct"] += int((expected & accepted).sum())
    def means(column):
        active = [v[column] / v[2] for v in totals.values() if v[2]]
        return sum(active) / len(active) if active else 0.0
    return {
        "selection_nll": means(0),
        "unweighted_supported_decisions_nll": means(1),
        "population_contract": CONDITIONAL_POPULATION_CONTRACT,
        "decision_counts": {kind: value[2] for kind, value in totals.items()},
        "head_nll": {kind: {"selection_nll": value[0] / value[2] if value[2] else 0.0,
                             "unweighted_nll": value[1] / value[2] if value[2] else 0.0}
                     for kind, value in totals.items()},
        "slices": slices,
        "records": len(rows),
    }


@torch.no_grad()
def evaluate_head_nll(encoder, head, rows, *, batch_size=32):
    return evaluate_head_metrics(encoder, head, rows, batch_size=batch_size)["selection_nll"]


def _head_throughput_probe(step_seconds, *, planned_batches, completed_updates):
    """Use warmed batches, retaining the measured cold cost in the total estimate."""
    steady = step_seconds[1:] or step_seconds
    seconds = statistics.median(steady)
    elapsed = sum(step_seconds)
    remaining = seconds * max(0, planned_batches - completed_updates)
    return {
        "sample_batches": len(step_seconds),
        "cold_step_seconds": step_seconds[0],
        "steady_step_seconds": seconds,
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": remaining,
        "estimated_fit_seconds": elapsed + remaining,
    }


def split_development_groups(records, *, seed):
    """Keep all paraphrases of one semantic mapping on one side of selection."""
    dev = [row for row in records if row["split"] == "dev"]
    groups = sorted({row.get("semantic_mapping_id", row["group_id"]) for row in dev})
    random.Random(seed).shuffle(groups)
    temperature_groups = set(groups[:max(1, len(groups) // 2)])
    temperature = [row for row in dev
                   if row.get("semantic_mapping_id", row["group_id"]) in temperature_groups]
    score = [row for row in dev
             if row.get("semantic_mapping_id", row["group_id"]) not in temperature_groups]
    return temperature, score


def development_split_identity(temperature, score):
    return {name: sorted({row.get("semantic_mapping_id", row["group_id"]) for row in rows})
            for name, rows in (("temperature_groups", temperature), ("score_groups", score))}


def developmental_feasibility_gate(report, rows):
    """Small development feasibility screen; never substitutes for admission gates."""
    mappings = {name: set() for name in ("held", "press", "aliases")}
    for row in rows:
        cluster = row.get("semantic_mapping_id", row["group_id"])
        labels = row["labels"]
        for label in labels:
            if (label["status"] == "supported" and label["mode"] in ("held", "press")
                    and label["candidate"].startswith(("key:", "mouse:"))):
                mappings[label["mode"]].add(cluster)
        groups = [label["alias_group"] for label in labels if label["alias_group"] > 0]
        if len(groups) > len(set(groups)):
            mappings["aliases"].add(cluster)
    slices = {name: report["signs"][name] for name in ("supported", "unsupported")}
    slices.update(report["modes"])
    slices["aliases"] = report["aliases"]
    coverage = {
        name: count.get("clear_accepted", count.get("correct_count", 0)) / count["clear_count"]
        if count["clear_count"] else 0.0 for name, count in slices.items()
    }
    distinct = {name: len(value) for name, value in mappings.items()}
    enough = (report["independent_semantic_clusters"] >= 100 and all(
        distinct[name] >= 30 and slices[name]["clear_count"] >= 100 for name in mappings
    ))
    return {
        "protocol": "developmental_feasibility_v1", "ready": False,
        "threshold": 0.99, "minimum_coverage": min(coverage.values()),
        "coverage": coverage, "distinct_mappings": distinct,
        "minimum_overall_mappings": 100, "minimum_slice_mappings": 30,
        "minimum_slice_gold_decisions": 100, "minimum_slice_coverage": 0.8,
        "enough_evidence": enough,
        "passed": bool(enough and report["critical_errors"] == 0
                       and all(value >= 0.8 for value in coverage.values())),
    }


@torch.no_grad()
def calibrated_development_nll(encoder, head, rows, calibration):
    device = next(head.parameters()).device
    components = {kind: [0.0, 0] for kind in ("status", "mode", "pair")}
    head.eval()
    for offset in range(0, len(rows), 32):
        batch = rows[offset:offset + 32]
        tokens, mask = _batch(encoder, batch, device)
        logits = _head_forward(encoder, head, batch, tokens, mask)
        targets = _targets(batch, device, conditional=True)
        for kind in components:
            values, expected = _loss_arrays(logits, targets, kind, True)
            if expected.numel():
                values = values - values.new_tensor(calibration[f"{kind}_log_class_weights"])
                total = F.cross_entropy(values / calibration[f"{kind}_temperature"], expected,
                                        reduction="sum")
                components[kind][0] += float(total)
                components[kind][1] += expected.numel()
    if not rows or any(not count for _, count in components.values()):
        raise ValueError("checkpoint scoring requires all conditional development populations")
    return _mean_decision_components(components)


def effective_train_presentations(records, *, cross_modal_train_repeats=1):
    """Repeat existing TRAIN views only; keep independent-case identities unchanged."""
    if type(cross_modal_train_repeats) is not int or not 1 <= cross_modal_train_repeats <= 8:
        raise ValueError("cross_modal_train_repeats must be an integer in 1..8")
    train = [row for row in records if row["split"] == "train"]
    groups = {}
    for row in train:
        groups.setdefault(row.get("group_id", row["text"]), []).append(row)
    repeated_kinds = {"cross_modal_alias", "contrastive_cross_modal_alias"}
    factors, kinds, pairs = {}, {}, {}
    for key, rows in groups.items():
        group_kinds = {row.get("scenario_kind", "unspecified") for row in rows}
        if cross_modal_train_repeats != 1 and len(group_kinds) != 1:
            raise ValueError("TRAIN group must have one scenario kind for exposure sampling")
        kind = rows[0].get("scenario_kind", "unspecified")
        kinds[key] = kind
        factors[key] = cross_modal_train_repeats if kind in repeated_kinds else 1
        if cross_modal_train_repeats != 1:
            pair_ids = {row.get("contrastive_pair_id") for row in rows}
            if len(pair_ids) != 1:
                raise ValueError("TRAIN group has inconsistent contrastive pair identity")
            pair_id = pair_ids.pop()
            if pair_id is not None:
                pairs.setdefault(pair_id, []).append(key)
    for members in pairs.values():
        if any(kinds[key] in repeated_kinds for key in members) and (
            len(members) != 2 or len({factors[key] for key in members}) != 1
        ):
            raise ValueError(
                "cross-modal contrastive pair must repeat both complete members equally"
            )
    effective = [row for row in train
                 for _ in range(factors[row.get("group_id", row["text"])])]
    metadata = {
        "cross_modal_train_repeats": cross_modal_train_repeats,
        "population": "TRAIN presentations of existing independent groups",
        "unique_train_groups": len(groups),
        "unique_train_records": len(train),
        "unique_train_texts": len({row["text"] for row in train}),
        "effective_train_presentations_per_epoch": len(effective),
        "repeated_train_group_ids": sorted(key for key, factor in factors.items() if factor > 1),
        "train_group_repeat_factors": dict(sorted(factors.items())),
        "train_group_unique_view_counts": {key: len(groups[key]) for key in sorted(groups)},
        "train_group_presentations_per_epoch": {
            key: len(groups[key]) * factors[key] for key in sorted(groups)},
        "unique_train_groups_by_kind": dict(Counter(kinds.values())),
        "unique_train_records_by_kind": dict(Counter(
            row.get("scenario_kind", "unspecified") for row in train)),
        "effective_train_presentations_by_kind": dict(Counter(
            row.get("scenario_kind", "unspecified") for row in effective)),
    }
    return effective, metadata


def _validate_fit_recipe(*, learning_rate, pair_learning_rate, readout_dropout,
                         early_stopping_patience, selection_protocol, head_variant,
                         cross_modal_train_repeats=1):
    for name, value in (("learning_rate", learning_rate),
                        ("pair_learning_rate", pair_learning_rate)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(readout_dropout) or not 0 <= readout_dropout < 1:
        raise ValueError("readout_dropout must be finite and in [0, 1)")
    if readout_dropout and head_variant != "supported_conditional_v6":
        raise ValueError("readout dropout requires the conditional head")
    if early_stopping_patience is not None and (
        type(early_stopping_patience) is not int or early_stopping_patience < 1
    ):
        raise ValueError("early_stopping_patience must be a positive integer")
    if selection_protocol not in {"dev_nll", "development_feasibility_v1"}:
        raise ValueError("unknown development selection protocol")
    if selection_protocol != "dev_nll" and head_variant != "supported_conditional_v6":
        raise ValueError("feasibility selection requires the conditional head")
    if type(cross_modal_train_repeats) is not int or not 1 <= cross_modal_train_repeats <= 8:
        raise ValueError("cross_modal_train_repeats must be an integer in 1..8")
    if cross_modal_train_repeats != 1 and (
        head_variant != "supported_conditional_v6"
        or selection_protocol != "development_feasibility_v1"
    ):
        raise ValueError(
            "cross-modal repeats require the conditional head and feasibility selection"
        )


def fit_text_control_head(
    encoder,
    records,
    *,
    seed=20261020,
    epochs=10,
    device="cpu",
    batch_size=32,
    deadline=None,
    failure_checkpoint=None,
    head_variant="legacy_v1",
    diagnostic_updates=(),
    learning_rate=3e-4,
    pair_learning_rate=None,
    readout_dropout=0.0,
    early_stopping_patience=None,
    selection_protocol="dev_nll",
    cross_modal_train_repeats=1,
):
    """AdamW only the new head; best epoch selected strictly by development NLL."""
    grounded = head_variant in {"grounded_balanced_v3", "local_grounded_balanced_v4",
                                "pair_geometry_balanced_v5", "supported_conditional_v6"}
    max_epochs = (80 if head_variant == "local_grounded_balanced_v4"
                  else 40 if grounded else 10)
    if type(epochs) is not int or not 1 <= epochs <= max_epochs:
        raise ValueError(f"head fit requires 1..{max_epochs} epochs")
    if (len(diagnostic_updates) > 8
            or any(type(step) is not int or step <= 0 for step in diagnostic_updates)):
        raise ValueError("at most eight positive diagnostic checkpoint updates are allowed")
    if diagnostic_updates and failure_checkpoint is None:
        raise ValueError("diagnostic updates require a checkpoint workspace")
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be in 1..64")
    _validate_fit_recipe(
        learning_rate=learning_rate, pair_learning_rate=pair_learning_rate,
        readout_dropout=readout_dropout, early_stopping_patience=early_stopping_patience,
        selection_protocol=selection_protocol, head_variant=head_variant,
        cross_modal_train_repeats=cross_modal_train_repeats,
    )
    torch.manual_seed(seed)
    rng = random.Random(seed)
    unique_train = [row for row in records if row["split"] == "train"]
    train, train_exposure = effective_train_presentations(
        records, cross_modal_train_repeats=cross_modal_train_repeats,
    )
    dev = [row for row in records if row["split"] == "dev"]
    if not train or not dev:
        raise ValueError("head fit requires nonempty train and dev splits")
    temperature_rows, score_rows = [], []
    if selection_protocol != "dev_nll":
        temperature_rows, score_rows = split_development_groups(records, seed=seed)
    selection_records = train + temperature_rows
    if selection_protocol != "dev_nll" and not score_rows:
        raise ValueError("feasibility selection requires disjoint development group halves")
    encoder.eval().requires_grad_(False)
    if head_variant not in {"legacy_v1", "residual_balanced_v2", "grounded_balanced_v3",
                            "local_grounded_balanced_v4", "pair_geometry_balanced_v5",
                            "supported_conditional_v6"}:
        raise ValueError("unsupported text head training variant")
    balanced = head_variant != "legacy_v1"
    conditional = head_variant == "supported_conditional_v6"
    balance = class_balance_from_train(train, conditional=conditional) if balanced else None
    class_weights = balance["weights"] if balanced else None
    head_class = {"legacy_v1": TextControlHead, "residual_balanced_v2": ResidualTextControlHead,
                  "grounded_balanced_v3": GroundedTextControlHead,
                  "local_grounded_balanced_v4": LocalGroundedTextControlHead,
                  "pair_geometry_balanced_v5": PairGeometryTextControlHead,
                  "supported_conditional_v6": ConditionalTextControlHead}[head_variant]
    head = (head_class(readout_dropout=readout_dropout) if conditional else head_class()).to(device)
    correction = _prior_correction(head, train)
    probe_rows = random.Random(seed).sample(unique_train, min(512, len(unique_train)))
    from carryrl.rewards.control_text import encoder_state_hash

    initial_hash = encoder_state_hash(head.state_dict())
    common_initial_hash = initial_hash
    if head_variant == "pair_geometry_balanced_v5":
        common = dict(head.state_dict())
        common["pair.0.weight"] = common["pair.0.weight"][:, :256].clone()
        common_initial_hash = encoder_state_hash(common)
    parameters = head.parameters()
    if pair_learning_rate is not None:
        pair_ids = {id(parameter) for parameter in head.pair.parameters()}
        parameters = [
            {"params": [p for p in head.parameters() if id(p) not in pair_ids],
             "lr": learning_rate},
            {"params": list(head.pair.parameters()), "lr": pair_learning_rate},
        ]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    fit_config = {
        "planned_epochs": epochs, "learning_rate": learning_rate,
        "pair_learning_rate": pair_learning_rate, "readout_dropout": readout_dropout,
        "early_stopping_patience": early_stopping_patience, "head_variant": head_variant,
        "seed": seed, "batch_size": batch_size, "selection_protocol": selection_protocol,
        "train_exposure": train_exposure,
    }
    if failure_checkpoint is not None:
        Path(failure_checkpoint).with_name("fit_config.json").write_text(
            json.dumps(fit_config, sort_keys=True) + "\n"
        )
    optimizer_initial_state_entries = len(optimizer.state)
    best, best_nll, history, selected_epoch = None, math.inf, [], None
    feasible_best, feasible_rank, selected_calibration = False, None, None
    updates = 0
    fit_probe = {}
    probe_steps = []
    probe_after = min(9, epochs * math.ceil(len(train) / batch_size))
    progress_path = (
        Path(failure_checkpoint).with_name("fit_progress.json")
        if correction and failure_checkpoint is not None else None
    )
    try:
        for epoch in range(epochs):
            rng.shuffle(train)
            head.train()
            train_loss_sum = 0.0
            train_components = {kind: [0.0, 0] for kind in ("status", "mode", "pair")}
            for offset in range(0, len(train), batch_size):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("text head fit budget exhausted; partial head preserved")
                step_start = time.monotonic()
                batch = train[offset : offset + batch_size]
                tokens, mask = _batch(encoder, batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = _head_forward(encoder, head, batch, tokens, mask)
                targets = _targets(batch, device, conditional=conditional)
                loss = head_loss(
                    logits, targets,
                    class_weights=class_weights,
                    unique_pairs=balanced,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite text head training loss")
                loss.backward()
                if correction:
                    gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                    if not torch.isfinite(gradient_norm):
                        raise RuntimeError("non-finite text head gradients")
                optimizer.step()
                if conditional:
                    for kind, (total, count) in _conditional_loss_components(
                        logits, targets, class_weights,
                    ).items():
                        train_components[kind][0] += float(total.detach())
                        train_components[kind][1] += count
                else:
                    train_loss_sum += float(loss.detach()) * len(batch)
                updates += 1
                if updates in diagnostic_updates:
                    torch.save({"head": head.state_dict(), "updates": updates,
                                "diagnostic_only": True, "trained": False},
                               Path(failure_checkpoint).with_name(f"diagnostic_update_{updates}.pt"))
                if updates <= probe_after:
                    probe_steps.append(time.monotonic() - step_start)
                if updates == probe_after:
                    planned_batches = epochs * (
                        math.ceil(len(train) / batch_size) + math.ceil(len(dev) / batch_size)
                        + (math.ceil(len(probe_rows) / batch_size) if correction else 0)
                    ) + math.ceil(len(records) / batch_size)
                    fit_probe = _head_throughput_probe(
                        probe_steps, planned_batches=planned_batches, completed_updates=updates,
                    )
                    fit_probe["excludes_development_decoder_selection"] = (
                        selection_protocol != "dev_nll"
                    )
                    if failure_checkpoint is not None:
                        Path(failure_checkpoint).with_name("fit_throughput_probe.json").write_text(
                            json.dumps(fit_probe, sort_keys=True) + "\n"
                        )
                    if (deadline is not None and fit_probe["estimated_remaining_seconds"] * 1.5
                            > deadline - time.monotonic()):
                        raise TimeoutError("head throughput probe exceeds remaining fit budget")
            metrics = evaluate_head_metrics(
                encoder,
                head,
                dev,
                batch_size=batch_size,
                class_weights=class_weights,
                unique_pairs=balanced,
                log_class_weights=correction,
            )
            nll = metrics["selection_nll"]
            train_fit_loss = (
                _mean_decision_components(train_components) if conditional
                else train_loss_sum / len(train)
            )
            entry = {"epoch": epoch + 1, "dev_nll": nll, "dev_metrics": metrics,
                     "train_fit_loss": train_fit_loss,
                     "cross_modal_train_repeats": cross_modal_train_repeats,
                     "unique_train_groups": train_exposure["unique_train_groups"],
                     "unique_train_texts": train_exposure["unique_train_texts"],
                     "effective_train_presentations": len(train)}
            if correction:
                entry["train_probe_metrics"] = evaluate_head_metrics(
                    encoder, head, probe_rows, batch_size=batch_size,
                    class_weights=class_weights, unique_pairs=balanced,
                    log_class_weights=correction,
                )
            candidate_calibration = None
            feasible, rank = False, None
            if selection_protocol == "development_feasibility_v1":
                selection_start = time.monotonic()
                candidate_calibration = fit_dev_temperatures(
                    encoder, head, selection_records, deadline=deadline,
                )
                candidate_calibration.update(
                    protocol="development_temperature_half_checkpoint_score_v1",
                    temperature_fit_scope="development_temperature_half",
                )
                predictor = FrozenTextControlReward(
                    encoder, copy.deepcopy(head), model_hash="0" * 64,
                    calibration=candidate_calibration,
                    calibration_hash=content_hash(candidate_calibration), _verified=True,
                )
                score_report = validate_text_control_predictions(predictor, score_rows)
                feasibility = developmental_feasibility_gate(score_report, score_rows)
                score_nll = calibrated_development_nll(
                    encoder, head, score_rows, candidate_calibration,
                )
                rank = (score_nll, -feasibility["minimum_coverage"])
                feasible = feasibility["passed"]
                entry.update(checkpoint_score=score_report, feasibility=feasibility,
                             calibrated_score_nll=score_nll,
                             development_selection_seconds=time.monotonic() - selection_start)
            history.append(entry)
            improved = (
                (feasible and (not feasible_best or rank < feasible_rank))
                or (not feasible_best and not feasible and nll < best_nll)
            )
            if improved:
                feasible_best = feasible
                feasible_rank = rank if feasible else None
                selected_calibration = candidate_calibration
                best_nll, best = nll, copy.deepcopy(head.state_dict())
                selected_epoch = epoch + 1
                if failure_checkpoint is not None:
                    torch.save({"head": best, "selected_epoch": selected_epoch,
                                "best_dev_nll": best_nll, "trained": False,
                                "development_calibration": selected_calibration,
                                "development_feasible": feasible_best},
                               Path(failure_checkpoint).with_name("best_head.pt"))
            if failure_checkpoint is not None:
                Path(failure_checkpoint).with_name("fit_history.json").write_text(
                    json.dumps(history, sort_keys=True) + "\n"
                )
            if progress_path is not None:
                progress_path.write_text(json.dumps({
                    "status": "fitting", "epoch": epoch + 1, "planned_epochs": epochs,
                    "updates": updates, "selected_epoch": selected_epoch,
                    "best_dev_nll": best_nll, "latest": entry,
                }, sort_keys=True) + "\n")
            if (early_stopping_patience is not None
                    and (selection_protocol == "dev_nll" or feasible_best)
                    and epoch + 1 - selected_epoch >= early_stopping_patience):
                break
    except BaseException:
        if failure_checkpoint is not None:
            torch.save(
                {
                    "head": head.state_dict(),
                    "updates": updates,
                    "history": history,
                    "trained": False,
                    "throughput_probe": fit_probe,
                    "config": fit_config,
                    "completed_epochs": len(history),
                    "selected_epoch": selected_epoch,
                    "selected_development_calibration": selected_calibration,
                    "selected_development_feasible": feasible_best,
                },
                failure_checkpoint,
            )
        raise
    head.load_state_dict(best, strict=True)
    return head.cpu().eval(), {
        "trained": True,
        "seed": seed,
        "epochs": epochs,
        "planned_epochs": epochs,
        "completed_epochs": len(history),
        "early_stopped": len(history) < epochs,
        "early_stopping_patience": early_stopping_patience,
        "updates": updates,
        "diagnostic_updates": list(diagnostic_updates),
        "batch_size": batch_size,
        "lr": learning_rate,
        "pair_learning_rate": pair_learning_rate,
        "readout_dropout": readout_dropout,
        "cross_modal_train_repeats": cross_modal_train_repeats,
        "train_exposure": train_exposure,
        "train_probe_population": "unique_train_records",
        "weight_decay": 0.01,
        "selection_protocol": selection_protocol,
        "selected_development_feasible": feasible_best,
        "selected_development_calibration": selected_calibration,
        "development_group_split": development_split_identity(temperature_rows, score_rows),
        "selection": ("feasible_then_calibrated_score_nll_then_worst_coverage"
                      if selection_protocol != "dev_nll" else
                      "dev_raw_train_weighted_nll" if correction
                      else "dev_train_weighted_nll" if balanced else "dev_nll"),
        "head_variant": head_variant,
        "head_architecture": head.architecture,
        "objective": ("balanced_supported_decisions_v6" if conditional
                      else "balanced_three_class_unique_pairs_v2" if balanced else "unweighted_v1"),
        **({"population_contract": CONDITIONAL_POPULATION_CONTRACT,
            "decoder_version": CONDITIONAL_DECODER_VERSION} if conditional else {}),
        "class_balance": balance,
        "deployment_logit_prior_correction": correction,
        "gradient_clip_norm": 5.0 if correction else None,
        "selected_epoch": selected_epoch,
        "initial_trainable_sha256": initial_hash,
        "initial_common_local_parameters_sha256": common_initial_hash,
        "imported_head_checkpoints": [],
        "optimizer_initial_state_entries": optimizer_initial_state_entries,
        "optimizer_started_empty": optimizer_initial_state_entries == 0,
        "history": history,
        "throughput_probe": fit_probe,
    }


@torch.no_grad()
def fit_dev_temperatures(encoder, head, records, *, deadline=None):
    """Fit temperatures using development labels only; thresholds remain fixed."""
    temperature_rows = [row for row in records if row["split"] == "dev"]
    temperature_groups = {
        row.get("semantic_mapping_id", row["group_id"]) for row in temperature_rows
    }
    if not temperature_rows:
        raise ValueError("development temperature rows required")
    device = next(head.parameters()).device
    collected = {kind: [] for kind in ("status", "mode", "pair")}
    targets = {kind: [] for kind in collected}
    head.eval()
    conditional = head.architecture == CONDITIONAL_HEAD_ARCHITECTURE
    for offset in range(0, len(temperature_rows), 32):
        batch = temperature_rows[offset : offset + 32]
        tokens, mask = _batch(encoder, batch, device)
        logits = _head_forward(encoder, head, batch, tokens, mask)
        labels = _targets(batch, device, conditional=conditional)
        for kind in collected:
            values, expected = _loss_arrays(
                logits, labels, kind, head.architecture in _GROUNDED_ARCHITECTURES,
            )
            collected[kind].append(values.cpu())
            targets[kind].append(expected.cpu())
    calibration = {f"{kind}_threshold": 0.99 for kind in collected}
    if conditional:
        calibration.update(population_contract=CONDITIONAL_POPULATION_CONTRACT,
                           decoder_version=CONDITIONAL_DECODER_VERSION)
    calibration.update(_prior_correction(head, [row for row in records if row["split"] == "train"]))
    for kind in collected:
        logits, labels = torch.cat(collected[kind]), torch.cat(targets[kind])
        if f"{kind}_log_class_weights" in calibration:
            logits = logits - logits.new_tensor(calibration[f"{kind}_log_class_weights"])
        if conditional and not labels.numel():
            raise ValueError(f"development split has no eligible {kind} temperature decisions")
        choices = torch.logspace(-1, 1, 81)
        losses = []
        for temperature in choices:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("text temperature calibration budget exhausted")
            losses.append(float(F.cross_entropy(logits / temperature, labels)))
        best = min(range(len(losses)), key=losses.__getitem__)
        calibration[f"{kind}_temperature"] = float(choices[best])
    calibration.update(
        temperature_group_count=len(temperature_groups),
        thresholds_selected="fixed_before_data",
        protocol="dev_temperature_calibration_gate_v2",
        pair_population=("gold_supported_unique_off_diagonal" if conditional
                         else "unique_off_diagonal"
                         if head.architecture in _GROUNDED_ARCHITECTURES
                         else "all_ordered_including_diagonal"),
    )
    if conditional:
        calibration.update(
            mode_population="gold_supported_discrete_known_mode",
            status_population="all_candidates",
            temperature_decision_counts={kind: sum(len(value) for value in targets[kind])
                                         for kind in targets},
        )
    return calibration


@torch.no_grad()
def calibrate_text_control_head(encoder, head, records, *, deadline=None):
    """After dev-NLL head selection, fit temperatures on dev; gate on full calibration.

    Thresholds are fixed before observing data. Calibration groups do not choose
    the head, temperatures or thresholds. Insufficient evidence fails readiness.
    """
    temperature_rows = [row for row in records if row["split"] == "dev"]
    gate_rows = [row for row in records if row["split"] == "calibration"]
    temperature_groups = {
        row.get("semantic_mapping_id", row["group_id"]) for row in temperature_rows
    }
    gate_groups = {row.get("semantic_mapping_id", row["group_id"]) for row in gate_rows}
    if not temperature_rows or not gate_rows or temperature_groups & gate_groups:
        raise ValueError("disjoint development-temperature and calibration-gate groups required")
    calibration = fit_dev_temperatures(encoder, head, records, deadline=deadline)
    calibration["gate_group_count"] = len(gate_groups)
    predictor = FrozenTextControlReward(
        encoder,
        head,
        model_hash="0" * 64,
        calibration=calibration,
        calibration_hash=content_hash(calibration),
        _verified=True,
    )
    report = validate_text_control_predictions(predictor, gate_rows)
    return calibration, report


def validate_text_control_predictions(predictor, rows):
    """Sign-specific decisions, input modes and alias false splits/merges."""

    def counts():
        return {
            "clear_count": 0,
            "clear_accepted": 0,
            "accepted_count": 0,
            "accepted_correct": 0,
            "accepted_scenarios": set(),
            "error_scenarios": set(),
        }

    aggregate = counts()
    signs = {name: counts() for name in ("supported", "unsupported")}
    modes = {
        name: {"clear_count": 0, "accepted_count": 0, "correct_count": 0}
        for name in ("held", "press")
    }
    aliases = {
        "clear_count": 0,
        "accepted_count": 0,
        "correct_count": 0,
        "false_split_errors": 0,
        "false_merge_errors": 0,
    }
    conditional = getattr(predictor, "calibration", {}).get(
        "population_contract"
    ) == CONDITIONAL_POPULATION_CONTRACT
    critical_errors = 0
    for row in rows:
        cluster = row.get("semantic_mapping_id", row["group_id"])
        table, labels = predictor.predict(row["text"]), row["labels"]
        for prediction, label in zip(table.predictions, labels, strict=True):
            clear, accepted = label["status"] != "unknown", prediction.status != "unknown"
            correct = prediction.status == label["status"]
            input_mode = label["status"] == "supported" and prediction.candidate.startswith(
                ("key:", "mouse:")
            )
            if conditional:
                input_mode &= label["mode"] in modes
            if input_mode:
                mode = modes[label["mode"]]
                mode["clear_count"] += 1
                mode_accepted = prediction.status == "supported" and prediction.mode != "unknown"
                mode["accepted_count"] += int(mode_accepted)
                mode["correct_count"] += int(mode_accepted and prediction.mode == label["mode"])
                if conditional:
                    mode_error = mode_accepted and prediction.mode != label["mode"]
                    if mode_error:
                        critical_errors += 1
                        aggregate["error_scenarios"].add(cluster)
                        signs["supported"]["error_scenarios"].add(cluster)
                else:
                    correct &= prediction.mode == label["mode"]
            aggregate["clear_count"] += int(clear)
            aggregate["clear_accepted"] += int(clear and accepted)
            if clear:
                sign = signs[label["status"]]
                sign["clear_count"] += 1
                sign["clear_accepted"] += int(prediction.status == label["status"])
            if accepted:
                for count in (aggregate, signs[prediction.status]):
                    count["accepted_count"] += 1
                    count["accepted_correct"] += int(correct)
                    count["accepted_scenarios"].add(cluster)
                    if not correct:
                        count["error_scenarios"].add(cluster)
                critical_errors += int(not correct)
        for left in range(len(CANDIDATES)):
            for right in range(left + 1, len(CANDIDATES)):
                a, b = table.predictions[left], table.predictions[right]
                expected_same = (
                    labels[left]["alias_group"] > 0
                    and labels[left]["alias_group"] == labels[right]["alias_group"]
                )
                accepted = a.status != "unknown" and b.status != "unknown"
                if conditional:
                    accepted = (a.reward_eligible and b.reward_eligible
                                and a.group is not None and b.group is not None)
                actual_same = a.group is not None and a.group == b.group
                if expected_same:
                    aliases["clear_count"] += 1
                    aliases["accepted_count"] += int(accepted)
                    aliases["correct_count"] += int(accepted and actual_same)
                false_split = expected_same and accepted and not actual_same
                false_merge = actual_same and not expected_same
                if conditional:
                    false_merge &= accepted
                if false_split or false_merge:
                    critical_errors += 1
                    aliases["false_split_errors"] += int(false_split)
                    aliases["false_merge_errors"] += int(false_merge)
                    aggregate["error_scenarios"].add(cluster)
                    for prediction in (a, b):
                        if prediction.status != "unknown":
                            signs[prediction.status]["error_scenarios"].add(cluster)
    for count in (aggregate, *signs.values()):
        for name in ("accepted_scenarios", "error_scenarios"):
            count[name] = len(count[name])
    report = {
        **aggregate,
        "critical_errors": critical_errors,
        "signs": signs,
        "modes": modes,
        "aliases": aliases,
        "records": len(rows),
        "scenario_count": len({row["group_id"] for row in rows}),
        "independent_semantic_clusters": len(
            {row.get("semantic_mapping_id", row["group_id"]) for row in rows}
        ),
    }
    report["gates"] = calibration_gates(report)
    return report


def select_probe_records(records, *, seed, max_train_groups=None, max_train_texts=None):
    """Keep complete TRAIN groups, all elementary groups and both sides of every contrast."""
    for name, limit in (("max_train_groups", max_train_groups),
                        ("max_train_texts", max_train_texts)):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError(f"{name} must be a positive integer")
    groups = {}
    for row in records:
        if row["split"] == "train":
            groups.setdefault(row["group_id"], []).append(row)
    elementary = sorted(key for key, rows in groups.items()
                        if any(row.get("elementary", False) for row in rows))
    pairs = {}
    for key, rows in groups.items():
        pair_ids = {row.get("contrastive_pair_id") for row in rows}
        if len(pair_ids) != 1:
            raise ValueError("TRAIN group has inconsistent contrastive pair identity")
        pair_id = pair_ids.pop()
        if pair_id is None:
            if any(row.get("contrastive", False) for row in rows):
                raise ValueError("contrastive TRAIN group is missing its pair identity")
            continue
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("contrastive pair identity must be a nonempty string")
        if any("group_variants" in row for row in rows):
            variants = {row.get("group_variants") for row in rows}
            if (len(variants) != 1 or type(next(iter(variants))) is not int
                    or next(iter(variants)) != len(rows)
                    or {row.get("variant_index") for row in rows} != set(range(len(rows)))):
                raise ValueError("contrastive pair contains an incomplete TRAIN group")
        pairs.setdefault(pair_id, []).append(key)
    if any(len(members) != 2 for members in pairs.values()):
        raise ValueError("contrastive pair must contain exactly two complete TRAIN groups")
    contrastive = sorted(key for members in pairs.values() for key in members)
    mandatory = sorted(set(elementary) | set(contrastive))
    selected = list(mandatory)
    texts = sum(len(groups[key]) for key in selected)
    if ((max_train_groups is not None and len(selected) > max_train_groups)
            or (max_train_texts is not None and texts > max_train_texts)):
        raise ValueError(
            "probe limits cannot retain all mandatory TRAIN groups: "
            f"{len(elementary)} elementary, {len(contrastive)} contrastive, "
            f"{len(mandatory)} total groups / {texts} texts"
        )
    strata = {}
    for key in sorted(groups.keys() - set(mandatory)):
        strata.setdefault(groups[key][0].get("scenario_kind", "general"), []).append(key)
    rng = random.Random(seed)
    for keys in strata.values():
        rng.shuffle(keys)
    while any(strata.values()):
        for kind in sorted(strata):
            if not strata[kind]:
                continue
            key = strata[kind].pop()
            if max_train_groups is not None and len(selected) >= max_train_groups:
                continue
            if max_train_texts is not None and texts + len(groups[key]) > max_train_texts:
                continue
            selected.append(key)
            texts += len(groups[key])
    selected = set(selected)
    result = [row for row in records if row["split"] == "dev"
              or (row["split"] == "train" and row["group_id"] in selected)]
    return result, {"selected_train_group_ids": sorted(selected),
                    "mandatory_elementary_group_ids": elementary,
                    "mandatory_contrastive_group_ids": contrastive,
                    "mandatory_contrastive_pair_ids": sorted(pairs),
                    "mandatory_train_group_ids": mandatory,
                    "mandatory_group_reasons": {
                        key: (["elementary"] if key in elementary else [])
                        + (["contrastive_pair"] if key in contrastive else [])
                        for key in mandatory},
                    "train_texts": texts, "train_groups": len(selected),
                    "max_train_groups": max_train_groups, "max_train_texts": max_train_texts,
                    "train_group_text_counts": {key: len(groups[key]) for key in sorted(selected)},
                    "train_unique_texts": len({row["text"] for row in result
                                               if row["split"] == "train"}),
                    "train_texts_by_language": dict(Counter(
                        row.get("language", "unspecified") for row in result
                        if row["split"] == "train")),
                    "train_groups_by_kind": dict(Counter(
                        groups[key][0].get("scenario_kind", "general") for key in selected))}


def train_text_control_artifact(
    dataset,
    output,
    *,
    encoder,
    seed=20261020,
    epochs=10,
    feature_cache=None,
    reuse_feature_cache=None,
    fit_workspace=None,
    head_variant="legacy_v1",
    calibration_previously_inspected=False,
    diagnostic_updates=(),
    learning_rate=3e-4,
    pair_learning_rate=None,
    readout_dropout=0.0,
    early_stopping_patience=None,
    selection_protocol="dev_nll",
    cross_modal_train_repeats=1,
    probe_only=False,
    max_train_groups=None,
    max_train_texts=None,
    max_feature_bytes=8 * 1024**3,
    max_seconds=10800,
    batch_size=32,
    device="cpu",
    runtime_instructions="configs/atomic_instructions.json",
    runtime_controls="configs/atomic_controls.json",
):
    """Bounded E5 extraction, frozen-head selection/calibration, runtime compatibility.

    Runtime examples never choose epochs or temperatures. Final readiness requires
    calibration and runtime compatibility; failures retain their primary artifacts.
    """
    from carryrl.rewards.control_text import encoder_state_hash
    from carryrl.rewards.control_text_runtime import validate_runtime_text_controls

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"text reward output already exists: {output}")
    if not 0 < max_seconds <= 10800:
        raise ValueError("text fit budget must be positive and at most three hours")
    if runtime_instructions is None or runtime_controls is None:
        raise ValueError("both runtime specification paths are required")
    _validate_fit_recipe(
        learning_rate=learning_rate, pair_learning_rate=pair_learning_rate,
        readout_dropout=readout_dropout, early_stopping_patience=early_stopping_patience,
        selection_protocol=selection_protocol, head_variant=head_variant,
        cross_modal_train_repeats=cross_modal_train_repeats,
    )
    manifest, records = read_control_dataset(dataset)
    start = time.monotonic()
    deadline = start + max_seconds
    if not probe_only and (max_train_groups is not None or max_train_texts is not None):
        raise ValueError("TRAIN subset limits are only allowed in probe mode")
    subset = None
    if probe_only:
        if epochs != 8 or early_stopping_patience is not None:
            raise ValueError(
                "development probe requires exactly eight epochs without early stopping"
            )
        if selection_protocol != "development_feasibility_v1":
            raise ValueError("development probe requires feasibility checkpoint selection")
        records, subset = select_probe_records(
            records, seed=seed, max_train_groups=max_train_groups, max_train_texts=max_train_texts,
        )
    effective_train, train_exposure = effective_train_presentations(
        records, cross_modal_train_repeats=cross_modal_train_repeats,
    )
    if feature_cache is not None and reuse_feature_cache is not None:
        raise ValueError("choose a new feature cache or read-only reuse, not both")
    feature_cache = Path(
        reuse_feature_cache or feature_cache or output.with_name(output.name + ".features")
    )
    workspace = Path(fit_workspace or output.with_name(output.name + ".fit"))
    cache_root, workspace_root = feature_cache.resolve(), workspace.resolve()
    if reuse_feature_cache is not None and (
        output.resolve().is_relative_to(cache_root) or cache_root.is_relative_to(output.resolve())
    ):
        raise ValueError("new model output must be separate from the reused feature cache")
    if workspace_root.is_relative_to(cache_root) or cache_root.is_relative_to(workspace_root):
        raise ValueError("fit workspace must be separate from the feature cache")
    workspace.mkdir(parents=True, exist_ok=False)
    failure_path = output.with_name(output.name + ".failure.json")
    timings = {}
    stage, stage_start = "input_preflight", time.monotonic()
    config = {
        "probe_only": probe_only, "ready": False, "seed": seed, "planned_epochs": epochs,
        "learning_rate": learning_rate, "pair_learning_rate": pair_learning_rate,
        "readout_dropout": readout_dropout, "early_stopping_patience": early_stopping_patience,
        "selection_protocol": selection_protocol, "head_variant": head_variant,
        "cross_modal_train_repeats": cross_modal_train_repeats, "train_exposure": train_exposure,
        "dataset_sha256": manifest["records_sha256"], "subset": subset,
        "encoded_splits": ["train", "dev"] if probe_only else ["train", "dev", "calibration"],
        "sealed_test_used": False,
    }
    (workspace / "run_config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
    try:
        transmission = None
        if head_variant in {"grounded_balanced_v3", "local_grounded_balanced_v4",
                            "pair_geometry_balanced_v5", "supported_conditional_v6"}:
            transmission = verify_grounded_text_inputs(
                encoder.tokenizer, records, deadline=deadline
            )
            (workspace / "input_transmission.json").write_text(
                json.dumps(transmission, sort_keys=True) + "\n"
            )
        timings[stage] = time.monotonic() - stage_start
        stage, stage_start = "features", time.monotonic()
        if reuse_feature_cache is not None:
            features = load_verified_disk_token_features(
                feature_cache,
                encoder=encoder,
                records=records,
                dataset_manifest=manifest,
                deadline=deadline,
                max_bytes=max_feature_bytes,
            )
        else:
            features = prepare_disk_token_features(
                encoder,
                records,
                feature_cache,
                dataset_sha256=manifest["records_sha256"],
                max_bytes=max_feature_bytes,
                deadline=deadline,
            )
        timings[stage] = time.monotonic() - stage_start
        stage, stage_start = "fit", time.monotonic()
        head, training = fit_text_control_head(
            features,
            records,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            deadline=deadline,
            failure_checkpoint=workspace / "partial_head.pt",
            head_variant=head_variant,
            diagnostic_updates=diagnostic_updates,
            learning_rate=learning_rate, pair_learning_rate=pair_learning_rate,
            readout_dropout=readout_dropout, early_stopping_patience=early_stopping_patience,
            selection_protocol=selection_protocol,
            cross_modal_train_repeats=cross_modal_train_repeats,
        )
        timings[stage] = time.monotonic() - stage_start
        stage, stage_start = "validation", time.monotonic()
        torch.save(
            {"head": head.state_dict(), "training": training}, workspace / "selected_head.pt"
        )
        (workspace / "selected_training.json").write_text(
            json.dumps(training, sort_keys=True) + "\n"
        )
        calibration = training.get("selected_development_calibration")
        if calibration is None:
            if probe_only:
                raise RuntimeError("probe selected head is missing its development temperatures")
            calibration_records = records
            if cross_modal_train_repeats != 1:
                calibration_records = effective_train + [
                    row for row in records if row["split"] != "train"
                ]
            calibration, validation = calibrate_text_control_head(
                features, head, calibration_records, deadline=deadline
            )
        else:
            (workspace / "selected_calibration.json").write_text(
                json.dumps(calibration, sort_keys=True) + "\n"
            )
            predictor = FrozenTextControlReward(
                features, head, model_hash=encoder_state_hash(head.state_dict()),
                calibration=calibration, calibration_hash=content_hash(calibration), _verified=True,
            )
            if probe_only:
                temperature_rows, score_rows = split_development_groups(records, seed=seed)
                validation = validate_text_control_predictions(predictor, score_rows)
                dev_report = validate_text_control_predictions(
                    predictor, temperature_rows + score_rows,
                )
                for name, value in (("selected_validation.json", validation),
                                    ("development_validation.json", dev_report)):
                    (workspace / name).write_text(json.dumps(value, sort_keys=True) + "\n")
                timings[stage] = time.monotonic() - stage_start
                stage, stage_start = "public_runtime", time.monotonic()
                runtime_predictor = FrozenTextControlReward(
                    encoder, head, model_hash=encoder_state_hash(head.state_dict()),
                    calibration=calibration, calibration_hash=content_hash(calibration),
                    _verified=True,
                )
                runtime = validate_runtime_text_controls(
                    runtime_predictor, instructions=runtime_instructions,
                    controls=runtime_controls, deadline=deadline,
                )
                runtime["evidence_scope"] = "DEVELOPMENTAL public runtime compatibility"
                result = {
                    "version": "carryrl.control_text_development_probe.v1", "ready": False,
                    "status": "complete", "evidence_scope": "DEVELOPMENTAL feasibility only",
                    "training": training, "config": config, "development": dev_report,
                    "checkpoint_score": validation,
                    "feasibility": developmental_feasibility_gate(validation, score_rows),
                    "runtime": runtime, "calibration_used": False, "sealed_test_used": False,
                    "fit_workspace": str(workspace.resolve()),
                }
                output.mkdir(parents=True, exist_ok=False)
                artifacts = (("probe.json", result), ("selected_calibration.json", calibration),
                             ("selected_validation.json", validation))
                for name, value in artifacts:
                    (output / name).write_text(json.dumps(value, sort_keys=True) + "\n")
                return result
            gate_rows = [row for row in records if row["split"] == "calibration"]
            if not gate_rows:
                raise ValueError("full admission requires calibration rows")
            temperature_groups = set(training["development_group_split"]["temperature_groups"])
            score_groups = set(training["development_group_split"]["score_groups"])
            gate_groups = {row.get("semantic_mapping_id", row["group_id"]) for row in gate_rows}
            if gate_groups & (temperature_groups | score_groups):
                raise ValueError("development/calibration semantic split leakage")
            calibration["gate_group_count"] = len(gate_groups)
            validation = validate_text_control_predictions(predictor, gate_rows)
        (workspace / "selected_calibration.json").write_text(
            json.dumps(calibration, sort_keys=True) + "\n"
        )
        if calibration_previously_inspected:
            validation["evidence_scope"] = (
                "Previously inspected calibration reused as a developmental repair gate; "
                "nominal bounds are not fresh independent certification."
            )
        (workspace / "selected_validation.json").write_text(
            json.dumps(validation, sort_keys=True) + "\n"
        )
        training.update(
            dataset_sha256=manifest["records_sha256"],
            input_transmission_preflight=transmission,
            sealed_test_used=False,
            calibration_previously_inspected=calibration_previously_inspected,
            independent_certification_pending_sealed_test=calibration_previously_inspected,
            feature_cache_bytes=features.metadata["actual_bytes"],
            feature_cache=str(feature_cache.resolve()),
            fit_workspace=str(workspace.resolve()),
            reused_feature_cache_read_only=reuse_feature_cache is not None,
            reuse_verification=getattr(features, "verification", None),
            max_seconds=max_seconds,
            encoding_throughput_probe=features.metadata["throughput_probe"],
        )
        runtime = None
        if runtime_instructions is not None:
            predictor = FrozenTextControlReward(
                encoder,
                head,
                model_hash=encoder_state_hash(head.state_dict()),
                calibration=calibration,
                calibration_hash=content_hash(calibration),
                _verified=True,
            )
            runtime = validate_runtime_text_controls(
                predictor,
                instructions=runtime_instructions,
                controls=runtime_controls,
                deadline=deadline,
            )
            training["runtime_gate_passed"] = runtime["passed"]
        if time.monotonic() >= deadline:
            raise TimeoutError("text artifact budget exhausted before final save")
        result = save_text_control_artifact(
            output,
            encoder=encoder,
            head=head,
            calibration=calibration,
            validation=validation,
            training={
                **training,
                "trained": False,
                "fitted": training["trained"],
                "runtime_gate_pending": True,
            },
        )
        if runtime is not None:
            runtime.update(
                head_sha256=result["files"]["head.pt"],
                dataset_sha256=manifest["records_sha256"],
                calibration_sha256=result["files"]["calibration.json"],
            )
            path = output / "runtime_validation.json"
            path.write_text(json.dumps(runtime, sort_keys=True) + "\n")
            result["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            result["runtime_gate"] = {
                "passed": runtime["passed"],
                "specification_hash": runtime["specification_hash"],
            }
            result["training"] = training
            result["ready"] = bool(
                training["trained"] and result["gates"]["passed"] and runtime["passed"]
                and (selection_protocol == "dev_nll"
                     or training["selected_development_feasible"])
            )
            (output / "manifest.json").write_text(json.dumps(result, sort_keys=True) + "\n")
        return result
    except BaseException as error:
        failure_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "ready": False,
                    "error": f"{type(error).__name__}: {error}",
                    "feature_cache": str(feature_cache),
                    "fit_workspace": str(workspace),
                    "head_variant": head_variant,
                    "dataset_sha256": manifest["records_sha256"],
                    "sealed_test_used": False,
                }
            )
            + "\n"
        )
        raise
    finally:
        timings[stage] = time.monotonic() - stage_start
        timings["total_seconds"] = time.monotonic() - start
        (workspace / "stage_timings.json").write_text(json.dumps(timings, sort_keys=True) + "\n")
