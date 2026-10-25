"""Learned instruction-only control interpretation, with fail-closed frozen loading.

Public engine metadata is deliberately absent from this module. Unknown predictions
abstain. The second model shares only frozen contextual E5 weights with the policy.
"""

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from carryrl.agent.actions import KEY_VOCAB, MOUSE_BUTTONS, PhysicalAction
from carryrl.agent.language import (
    TEXT_ID,
    TEXT_REVISION,
    TEXT_WIDTH,
    MultilingualInstructionEncoder,
)

CANDIDATES = (
    tuple(f"key:{key}" for key in KEY_VOCAB)
    + tuple(f"mouse:{button}" for button in MOUSE_BUTTONS)
    + tuple(f"{axis}:{sign}" for axis in ("dx", "dy", "wheel") for sign in ("-", "+"))
)
STATUSES = ("supported", "unsupported", "unknown")
MODES = ("held", "press", "unknown")
PAIRS = ("same", "different", "unknown")
MODEL_VERSION = "carryrl.control_text.cross_attention.v1"
LEGACY_HEAD_ARCHITECTURE = "cross_attention.v1"
RESIDUAL_HEAD_ARCHITECTURE = "candidate_residual_layernorm.v2"
GROUNDED_HEAD_ARCHITECTURE = "grounded_mentions.v3"
LOCAL_GROUNDED_HEAD_ARCHITECTURE = "grounded_local_context.v4"
PAIR_GEOMETRY_HEAD_ARCHITECTURE = "grounded_local_pair_geometry.v5"
CONDITIONAL_HEAD_ARCHITECTURE = "grounded_conditional_controls.v6"
CONDITIONAL_POPULATION_CONTRACT = "supported_decisions.v1"
INDEPENDENT_DECODER_VERSION = "independent_semantics_dedup.v2"


def content_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encoder_state_hash(state) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ControlPrediction:
    candidate: str
    status: str = "unknown"
    mode: str = "unknown"
    group: str | None = None
    confidence: float = 0.0
    deduplication_known: bool = True

    def __post_init__(self):
        if (
            self.candidate not in CANDIDATES
            or self.status not in STATUSES
            or self.mode not in MODES
        ):
            raise ValueError("invalid control prediction")
        if (
            type(self.confidence) not in (int, float)
            or not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("invalid control confidence")
        if self.group is not None and (not isinstance(self.group, str) or not self.group):
            raise ValueError("invalid alias group")
        if type(self.deduplication_known) is not bool:
            raise ValueError("deduplication_known must be a boolean")
        if not self.deduplication_known and self.group is not None:
            raise ValueError("unresolved deduplication cannot claim an alias group")
        if self.status == "unknown" and self.group is not None:
            raise ValueError("unknown controls cannot claim an alias group")
        if self.status != "unknown" and self.group is None and self.deduplication_known:
            raise ValueError("accepted control requires a deduplication group")

    @property
    def reward_eligible(self) -> bool:
        return self.status != "unknown" and self.deduplication_known and self.group is not None


@dataclass(frozen=True)
class ControlPredictionTable:
    text_hash: str
    model_hash: str
    calibration_hash: str
    predictions: tuple[ControlPrediction, ...]

    def __post_init__(self):
        for value in (self.text_hash, self.model_hash, self.calibration_hash):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError("control table identities must be SHA256 hashes")
        if (
            not isinstance(self.predictions, tuple)
            or any(not isinstance(row, ControlPrediction) for row in self.predictions)
            or tuple(row.candidate for row in self.predictions) != CANDIDATES
        ):
            raise ValueError("control table requires all candidates in canonical order")
        groups = {}
        for row in self.predictions:
            if row.group is not None:
                status, modes = groups.setdefault(row.group, (row.status, set()))
                if status != row.status:
                    raise ValueError("alias group contains conflicting status")
                if row.candidate.startswith(("key:", "mouse:")) and row.mode != "unknown":
                    modes.add(row.mode)
                if len(modes) > 1:
                    raise ValueError("alias group contains conflicting key/button modes")

    @property
    def identity(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict:
        record = asdict(self)
        # Keep historical table hashes unchanged for the legacy decoder.
        for row in record["predictions"]:
            if row["deduplication_known"]:
                row.pop("deduplication_known")
        return record

    @classmethod
    def from_dict(cls, record: dict):
        return cls(
            **{
                **record,
                "predictions": tuple(ControlPrediction(**row) for row in record["predictions"]),
            }
        )


def executed_candidates(action: PhysicalAction) -> tuple[str, ...]:
    if not isinstance(action, PhysicalAction):
        raise TypeError("requires an executed PhysicalAction")
    return (
        tuple(f"key:{key}" for key in action.keys)
        + tuple(f"mouse:{button}" for button in action.mouse_buttons)
        + tuple(
            f"{axis}:{'+' if getattr(action, axis) > 0 else '-'}"
            for axis in ("dx", "dy", "wheel")
            if getattr(action, axis)
        )
    )


class TextControlHead(nn.Module):
    """36 learned candidate queries over frozen E5 contextual token features."""

    architecture = LEGACY_HEAD_ARCHITECTURE

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(TEXT_WIDTH, 128)
        self.queries = nn.Parameter(torch.randn(len(CANDIDATES), 128) * 0.02)
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.status = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3))
        self.mode = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3))
        self.pair = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None):
        if tokens.ndim != 3 or tokens.shape[-1] != TEXT_WIDTH or tokens.shape[1] > 512:
            raise ValueError("expected batch of at most 512 contextual E5 tokens")
        projected = self.projection(tokens)
        queries = self.queries[None].expand(tokens.shape[0], -1, -1)
        features, _ = self.attention(
            queries, projected, projected, key_padding_mask=padding_mask, need_weights=False
        )
        return self._readout(features)

    def _readout(self, features):
        left = features[:, :, None].expand(-1, -1, len(CANDIDATES), -1)
        right = features[:, None, :].expand(-1, len(CANDIDATES), -1, -1)
        # Symmetric representation prevents direction-dependent alias decisions.
        pair = self.pair(torch.cat((left + right, (left - right).abs()), dim=-1))
        return {"status": self.status(features), "mode": self.mode(features), "pair": pair}


class ResidualTextControlHead(TextControlHead):
    """Fresh candidate identities survive attention into every shared classifier."""

    architecture = RESIDUAL_HEAD_ARCHITECTURE

    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(128)
        self.attention_norm = nn.LayerNorm(128)
        self.feed_forward = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
        self.output_norm = nn.LayerNorm(128)

    def forward(self, tokens, padding_mask=None):
        if tokens.ndim != 3 or tokens.shape[-1] != TEXT_WIDTH or tokens.shape[1] > 512:
            raise ValueError("expected batch of at most 512 contextual E5 tokens")
        projected = self.projection(tokens)
        queries = self.query_norm(self.queries)[None].expand(tokens.shape[0], -1, -1)
        attended, _ = self.attention(
            queries, projected, projected, key_padding_mask=padding_mask, need_weights=False
        )
        features = self.attention_norm(queries + attended)
        features = self.output_norm(features + self.feed_forward(features))
        return self._readout(features)


class GroundedTextControlHead(TextControlHead):
    """Learn control meaning from anchored mentions and the full instruction.

    Grounding supplies lexical positions only. The presence or absence of a
    mention never supplies a status, mode or pair target.
    """

    architecture = GROUNDED_HEAD_ARCHITECTURE

    def __init__(self):
        super().__init__()
        self.query_norm = nn.LayerNorm(128)
        self.mention_attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.fusion = nn.Linear(128 * 3 + 1, 128)
        self.mention_norm = nn.LayerNorm(128)
        self.context_norm = nn.LayerNorm(128)
        self.feed_forward = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))
        self.output_norm = nn.LayerNorm(128)

    def forward(self, tokens, padding_mask=None, *, grounding=None):
        if tokens.ndim != 3 or tokens.shape[-1] != TEXT_WIDTH or not 0 < tokens.shape[1] <= 512:
            raise ValueError("expected batch of at most 512 contextual E5 tokens")
        batch, length = tokens.shape[:2]
        if grounding is None or grounding.shape != (batch, len(CANDIDATES), length):
            raise ValueError("grounding must have shape [batch, 36, tokens]")
        grounding = grounding.to(device=tokens.device)
        if grounding.dtype != torch.bool:
            if (
                not torch.is_floating_point(grounding)
                or not ((grounding == 0) | (grounding == 1)).all()
            ):
                raise ValueError("grounding must be a boolean or binary float mention mask")
            grounding = grounding.bool()
        if padding_mask is None:
            padding_mask = torch.zeros(batch, length, dtype=torch.bool, device=tokens.device)
        elif padding_mask.shape != (batch, length) or padding_mask.dtype != torch.bool:
            raise ValueError("padding mask must be boolean [batch, tokens]")
        padding_mask = padding_mask.to(tokens.device)
        if padding_mask.all(dim=1).any():
            raise ValueError("each instruction requires at least one unpadded token")
        grounding = grounding & ~padding_mask[:, None]
        projected = self.projection(tokens)
        identities = self.query_norm(self.queries)[None].expand(batch, -1, -1)
        count = grounding.sum(-1, keepdim=True)
        present = count > 0
        pooled = torch.bmm(grounding.to(projected.dtype), projected) / count.clamp_min(1)
        mention, _ = self.mention_attention(
            identities + pooled,
            projected,
            projected,
            attn_mask=self._mention_attention_mask(grounding, padding_mask, dtype=projected.dtype),
            need_weights=False,
        )
        mention = mention * present.to(mention.dtype)
        anchored = self.mention_norm(
            identities
            + self.fusion(
                torch.cat((pooled, mention, identities, present.to(projected.dtype)), dim=-1)
            )
        )
        context, _ = self.attention(
            anchored, projected, projected, key_padding_mask=padding_mask, need_weights=False
        )
        features = self.context_norm(anchored + context)
        features = self.output_norm(features + self.feed_forward(features))
        return self._grounded_readout(features, grounding)

    def _grounded_readout(self, features, grounding):
        return self._readout(features)

    def _mention_attention_mask(self, grounding, padding_mask, *, dtype):
        # A temporary valid attention position avoids all-masked softmax rows.
        # Its output is erased for absent mentions and is never a class label.
        present = grounding.any(-1, keepdim=True)
        allowed = grounding.clone()
        first_valid = (~padding_mask).to(torch.int64).argmax(-1)
        allowed.scatter_(2, first_valid[:, None, None].expand(-1, len(CANDIDATES), 1), True)
        allowed = torch.where(present, grounding, allowed)
        return (~allowed).repeat_interleave(self.mention_attention.num_heads, dim=0)


class LocalGroundedTextControlHead(GroundedTextControlHead):
    """The same learned head with fixed local context around lexical mentions."""

    architecture = LOCAL_GROUNDED_HEAD_ARCHITECTURE

    def _mention_attention_mask(self, grounding, padding_mask, *, dtype):
        batch, candidates, length = grounding.shape
        # Descending shifts retain the nearest anchor within 24 tokens. This
        # needs O(B*C*L) memory and uses only ordinary CPU/MPS tensor operations.
        distance = torch.full_like(grounding, 25, dtype=dtype)
        for offset in range(min(24, length - 1), 0, -1):
            distance[:, :, offset:].masked_fill_(grounding[:, :, :-offset], offset)
            distance[:, :, :-offset].masked_fill_(grounding[:, :, offset:], offset)
        distance.masked_fill_(grounding, 0)
        # Absent candidates retain the inherited zero mention output, while one
        # finite valid position prevents an all-negative-infinity softmax row.
        absent = ~grounding.any(-1, keepdim=True)
        first_valid = (~padding_mask).to(torch.int64).argmax(-1)
        fallback = torch.zeros_like(grounding)
        fallback.scatter_(2, first_valid[:, None, None].expand(-1, candidates, 1), True)
        distance.masked_fill_(absent & fallback, 0)
        slopes = distance.new_tensor([1 / 3, 1 / 6, 1 / 12, 1 / 24])
        bias = -distance[:, None] * slopes[None, :, None, None]
        bias.masked_fill_(distance[:, None] > 24, -torch.inf)
        bias.masked_fill_(padding_mask[:, None, None], -torch.inf)
        return bias.reshape(batch * self.mention_attention.num_heads, candidates, length)


def lexical_pair_geometry(grounding):
    """Eight symmetric numeric features from unpadded lexical token positions.

    Ordered features: both-present, both-exclusive-present, minimum distance,
    minimum exclusive distance, centroid gap, overlap Jaccard, total token
    cardinality, log-cardinality gap. Distances use log1p / log(513), with 512
    as the missing-distance sentinel. These values never supply a class label.
    Input is bool [B,36,L], with padding removed by the calling head.
    """
    if (
        grounding.dtype != torch.bool
        or grounding.ndim != 3
        or grounding.shape[1] != len(CANDIDATES)
        or not 0 < grounding.shape[2] <= 512
    ):
        raise ValueError("pair geometry requires boolean [batch,36,tokens] grounding")
    values = grounding.float()
    count = values.sum(-1)
    positions = torch.arange(grounding.shape[-1], device=grounding.device, dtype=torch.float32)
    centroid = (values * positions).sum(-1) / count.clamp_min(1)
    overlap = torch.bmm(values, values.transpose(1, 2))
    left_count, right_count = count[:, :, None], count[:, None, :]
    both = (left_count > 0) & (right_count > 0)
    exclusive = (left_count > overlap) & (right_count > overlap)

    def minimum_distance(left, right):
        before = torch.cummax(torch.where(left, positions, -512.0), dim=-1).values
        after = torch.flip(
            torch.cummin(
                torch.flip(torch.where(left, positions, 1024.0), dims=(-1,)), dim=-1
            ).values,
            dims=(-1,),
        )
        nearest = torch.minimum(positions - before, after - positions).clamp_max(512)
        return nearest.masked_fill(~right, 512).amin(-1)

    # Chunk only one candidate dimension, bounding intermediate memory to
    # B*6*36*L, never B*36*36*L*L. Grounding has no trainable gradients.
    minima, exclusive_minima = [], []
    right = grounding[:, None]
    for start in range(0, len(CANDIDATES), 6):
        left = grounding[:, start : start + 6, None]
        minima.append(minimum_distance(left, right))
        exclusive_minima.append(minimum_distance(left & ~right, right & ~left))
    minimum = torch.cat(minima, dim=1)
    minimum_exclusive = torch.cat(exclusive_minima, dim=1)
    gap = (centroid[:, :, None] - centroid[:, None, :]).abs().masked_fill(~both, 0)
    scale = math.log1p(512)
    return torch.stack(
        (
            both.float(),
            exclusive.float(),
            minimum.log1p() / scale,
            minimum_exclusive.log1p() / scale,
            gap.log1p() / scale,
            overlap / (left_count + right_count - overlap).clamp_min(1),
            (left_count + right_count).log1p() / math.log1p(1024),
            (left_count.log1p() - right_count.log1p()).abs() / scale,
        ),
        dim=-1,
    )


class PairGeometryTextControlHead(LocalGroundedTextControlHead):
    """Local head with a learned pair classifier that can inspect token geometry."""

    architecture = PAIR_GEOMETRY_HEAD_ARCHITECTURE

    def __init__(self):
        super().__init__()
        original = self.pair[0]
        extended = nn.Linear(original.in_features + 8, original.out_features)
        with torch.no_grad():
            extended.weight[:, : original.in_features].copy_(original.weight)
            extended.weight[:, original.in_features :].zero_()
            extended.bias.copy_(original.bias)
        self.pair[0] = extended

    def _grounded_readout(self, features, grounding):
        left = features[:, :, None].expand(-1, -1, len(CANDIDATES), -1)
        right = features[:, None, :].expand(-1, len(CANDIDATES), -1, -1)
        geometry = lexical_pair_geometry(grounding).to(features.dtype)
        pair = self.pair(torch.cat((left + right, (left - right).abs(), geometry), dim=-1))
        return {"status": self.status(features), "mode": self.mode(features), "pair": pair}


class ConditionalTextControlHead(PairGeometryTextControlHead):
    """Mode and alias classifiers are conditional on supported controls."""

    architecture = CONDITIONAL_HEAD_ARCHITECTURE

    def __init__(self, *, readout_dropout=0.0):
        super().__init__()
        if not math.isfinite(readout_dropout) or not 0 <= readout_dropout < 1:
            raise ValueError("readout_dropout must be finite and in [0, 1)")
        self.readout_dropout = nn.Dropout(readout_dropout)
        self.mode[-1] = nn.Linear(64, 2)
        self.pair[-1] = nn.Linear(64, 2)

    def _grounded_readout(self, features, grounding):
        return super()._grounded_readout(self.readout_dropout(features), grounding)


def text_control_head_for_architecture(architecture):
    if architecture == LEGACY_HEAD_ARCHITECTURE:
        return TextControlHead()
    if architecture == RESIDUAL_HEAD_ARCHITECTURE:
        return ResidualTextControlHead()
    if architecture == GROUNDED_HEAD_ARCHITECTURE:
        return GroundedTextControlHead()
    if architecture == LOCAL_GROUNDED_HEAD_ARCHITECTURE:
        return LocalGroundedTextControlHead()
    if architecture == PAIR_GEOMETRY_HEAD_ARCHITECTURE:
        return PairGeometryTextControlHead()
    if architecture == CONDITIONAL_HEAD_ARCHITECTURE:
        return ConditionalTextControlHead()
    raise ValueError("unsupported text control head architecture")


def _count_gate(report: dict) -> dict:
    required = (
        "clear_count",
        "clear_accepted",
        "accepted_count",
        "accepted_correct",
        "accepted_scenarios",
        "error_scenarios",
    )
    if any(type(report.get(key)) is not int or report[key] < 0 for key in required):
        raise ValueError("calibration requires nonnegative integer decision/cluster counts")
    if (
        report["clear_accepted"] > report["clear_count"]
        or report["accepted_correct"] > report["accepted_count"]
        or report["error_scenarios"] > report["accepted_scenarios"]
        or report["accepted_scenarios"] > report["accepted_count"]
    ):
        raise ValueError("inconsistent calibration counts")
    n, errors = report["accepted_scenarios"], report["error_scenarios"]
    if n and not errors:
        lower = 0.05 ** (1 / n)
    elif n:
        lower = max(0.0, 1 - errors / n - math.sqrt(math.log(20) / (2 * n)))
    else:
        lower = 0.0
    coverage = report["clear_accepted"] / report["clear_count"] if report["clear_count"] else 0
    precision = (
        report["accepted_correct"] / report["accepted_count"] if report["accepted_count"] else 0
    )
    return {
        "passed": coverage >= 0.8 and precision >= 0.99 and lower >= 0.99,
        "clear_coverage": coverage,
        "accepted_precision": precision,
        "scenario_precision_lower_95": lower,
        "independent_scenarios": n,
    }


def calibration_gates(report: dict) -> dict:
    """Each sign needs useful coverage and its own scenario precision evidence.

    These are two separately stated one-sided 95% bounds, not a simultaneous
    family-wise 95% claim. Mode and alias gates require coverage and zero errors;
    their smaller slices do not independently establish 99% precision.
    """
    if type(report.get("critical_errors")) is not int or report["critical_errors"] < 0:
        raise ValueError("calibration requires critical error counts")
    aggregate = _count_gate(report)
    signs = report.get("signs", {})
    if set(signs) != {"supported", "unsupported"}:
        raise ValueError("calibration requires separate supported/unsupported evidence")
    sign_gates = {name: _count_gate(counts) for name, counts in signs.items()}
    coverage_gates = {}
    for name, counts in {
        **{f"mode_{name}": value for name, value in report.get("modes", {}).items()},
        "aliases": report.get("aliases", {}),
    }.items():
        for key in ("clear_count", "accepted_count", "correct_count"):
            if type(counts.get(key)) is not int or counts[key] < 0:
                raise ValueError("mode/alias calibration requires integer evidence counts")
        if not counts["correct_count"] <= counts["accepted_count"] <= counts["clear_count"]:
            raise ValueError("inconsistent mode/alias calibration counts")
        coverage = counts["accepted_count"] / counts["clear_count"] if counts["clear_count"] else 0
        coverage_gates[name] = {
            "coverage": coverage,
            "passed": coverage >= 0.8 and counts["correct_count"] == counts["accepted_count"],
        }
    if not {"mode_held", "mode_press", "aliases"} <= coverage_gates.keys():
        raise ValueError("calibration requires both input modes and alias evidence")
    passed = (
        report["critical_errors"] == 0
        and aggregate["passed"]
        and all(gate["passed"] for gate in sign_gates.values())
        and all(gate["passed"] for gate in coverage_gates.values())
    )
    return {
        **aggregate,
        "passed": passed,
        "signs": sign_gates,
        "coverage_checks": coverage_gates,
        "confidence_scope": "separate_one_sided_95_per_sign_not_simultaneous",
    }


def _conditional_calibration(record: dict) -> bool:
    population = record.get("population_contract")
    decoder = record.get("decoder_version")
    if population is None and decoder is None:
        return False
    if population != CONDITIONAL_POPULATION_CONTRACT or decoder != INDEPENDENT_DECODER_VERSION:
        raise ValueError("unsupported text control population/decoder contract")
    return True


def _check_head_calibration(head, calibration):
    conditional = _conditional_calibration(calibration)
    if conditional != (getattr(head, "architecture", None) == CONDITIONAL_HEAD_ARCHITECTURE):
        raise ValueError("text control head architecture and calibration contract mismatch")


def _checked_calibration(record: dict) -> dict:
    conditional = _conditional_calibration(record)
    for name in ("status_threshold", "mode_threshold", "pair_threshold"):
        value = record.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0.5 < value <= 1:
            raise ValueError(f"invalid calibrated {name}")
    for name in ("status_temperature", "mode_temperature", "pair_temperature"):
        value = record.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid calibrated {name}")
    for kind in ("status", "mode", "pair"):
        name = f"{kind}_log_class_weights"
        if name in record:
            values = record[name]
            if (
                not isinstance(values, list)
                or len(values) != (2 if conditional and kind != "status" else 3)
                or any(
                    type(value) not in (int, float) or not math.isfinite(value) for value in values
                )
            ):
                raise ValueError(f"invalid calibrated {name}")
    return record


def calibrated_head_logits(logits, calibration, kind):
    """Undo optional TRAIN class weighting before temperature calibration."""
    conditional = _conditional_calibration(calibration)
    width = 2 if conditional and kind != "status" else 3
    if kind not in ("status", "mode", "pair") or logits.shape[-1] != width:
        raise ValueError("calibrated logits require the contract's head class width")
    weights = calibration.get(f"{kind}_log_class_weights")
    if weights is not None:
        if len(weights) != width:
            raise ValueError("calibrated class weights do not match head class width")
        logits = logits - logits.new_tensor(weights)
    return logits / calibration[f"{kind}_temperature"]


class FrozenTextControlReward:
    """Only a verified trained artifact can construct the runtime predictor."""

    def __init__(
        self, encoder, head, *, model_hash, calibration, calibration_hash, _verified=False
    ):
        if not _verified:
            raise ValueError("use load_text_control_reward; unvalidated random heads are forbidden")
        self.encoder, self.head = encoder, head.eval().requires_grad_(False)
        self.model_hash, self.calibration_hash = model_hash, calibration_hash
        self.calibration = _checked_calibration(calibration)
        _check_head_calibration(head, self.calibration)
        self._cache = OrderedDict()

    @property
    def identity(self):
        return {
            "version": MODEL_VERSION,
            "model_hash": self.model_hash,
            "calibration_hash": self.calibration_hash,
        }

    def save_resources(self, destination, *, shared_encoder=True):
        """Export verified local resources; shared E5 weights remain in the agent."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        manifest = json.loads(json.dumps(self._manifest))
        if encoder_state_hash(self.encoder.state_dict()) != manifest["encoder_state_hash"]:
            raise ValueError("shared E5 weights changed after frozen text reward loading")
        for name, payload in self._resources.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        if shared_encoder:
            manifest["files"].pop("encoder.pt", None)
            manifest["shared_encoder"] = True
        else:
            torch.save(self.encoder.state_dict(), destination / "encoder.pt")
            manifest["files"]["encoder.pt"] = _sha_file(destination / "encoder.pt")
            manifest["shared_encoder"] = False
        (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
        return destination

    @torch.no_grad()
    def predict(self, text: str) -> ControlPredictionTable:
        if not isinstance(text, str):
            raise ValueError("text control reward requires a string instruction")
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        key = (text_hash, self.model_hash, self.calibration_hash)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        tokens = self.encoder.encode_instruction_tokens(text)
        device = next(self.head.parameters()).device
        architecture = getattr(self.head, "architecture", LEGACY_HEAD_ARCHITECTURE)
        if architecture in (
            GROUNDED_HEAD_ARCHITECTURE,
            LOCAL_GROUNDED_HEAD_ARCHITECTURE,
            PAIR_GEOMETRY_HEAD_ARCHITECTURE,
            CONDITIONAL_HEAD_ARCHITECTURE,
        ):
            from carryrl.rewards.control_text_grounding import build_grounding

            grounding = build_grounding(text, self.encoder.tokenizer, expected_tokens=len(tokens))
            logits = self.head(tokens[None].to(device), grounding=grounding[None].to(device))
        else:
            logits = self.head(tokens[None].to(device))
        probabilities = {}
        for kind in ("status", "mode", "pair"):
            probabilities[kind] = (
                calibrated_head_logits(logits[kind][0], self.calibration, kind).softmax(-1).cpu()
            )
        if _conditional_calibration(self.calibration):
            table = self._conditional_predictions(probabilities, text_hash)
            self._cache[key] = table
            if len(self._cache) > 128:
                self._cache.popitem(last=False)
            return table
        statuses, modes, confidences = [], [], []
        for i, candidate in enumerate(CANDIDATES):
            confidence, index = probabilities["status"][i].max(0)
            status = STATUSES[int(index)]
            if float(confidence) < self.calibration["status_threshold"]:
                status = "unknown"
            mode_confidence, mode_index = probabilities["mode"][i].max(0)
            mode = (
                MODES[int(mode_index)]
                if (float(mode_confidence) >= self.calibration["mode_threshold"])
                else "unknown"
            )
            if (
                status == "supported"
                and candidate.startswith(("key:", "mouse:"))
                and mode == "unknown"
            ):
                status = "unknown"
            statuses.append(status)
            modes.append(mode if candidate.startswith(("key:", "mouse:")) else "unknown")
            confidences.append(float(confidence))
        adjacency = {i: {i} for i, status in enumerate(statuses) if status != "unknown"}
        uncertain = set()
        for i in adjacency:
            for j in adjacency:
                if j <= i or statuses[i] != statuses[j]:
                    continue
                confidence, index = probabilities["pair"][i, j].max(0)
                if float(confidence) < self.calibration["pair_threshold"] or int(index) == 2:
                    uncertain.update((i, j))
                elif int(index) == 0:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        groups, visited = [], set()
        for origin in adjacency:
            if origin in visited:
                continue
            pending, component = [origin], set()
            while pending:
                index = pending.pop()
                if index not in component:
                    component.add(index)
                    pending.extend(adjacency[index] - component)
            visited.update(component)
            key_modes = {
                modes[k] for k in component if CANDIDATES[k].startswith(("key:", "mouse:"))
            }
            clique = all(component <= adjacency[k] for k in component)
            if component & uncertain or not clique or len(key_modes) > 1:
                # An uncertain or inconsistent alias component cannot safely
                # become distinct bonus-paying singleton controls.
                for index in component:
                    statuses[index] = "unknown"
            else:
                groups.append(sorted(component))
        names = {i: CANDIDATES[min(group)] for group in groups for i in group}
        table = ControlPredictionTable(
            text_hash,
            self.model_hash,
            self.calibration_hash,
            tuple(
                ControlPrediction(
                    candidate,
                    statuses[i],
                    modes[i],
                    names[i] if statuses[i] != "unknown" else None,
                    confidences[i],
                )
                for i, candidate in enumerate(CANDIDATES)
            ),
        )
        self._cache[key] = table
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        return table

    def _conditional_predictions(self, probabilities, text_hash):
        statuses, modes, confidences = [], [], []
        for i, candidate in enumerate(CANDIDATES):
            confidence, index = probabilities["status"][i].max(0)
            status = STATUSES[int(index)]
            if float(confidence) < max(0.99, self.calibration["status_threshold"]):
                status = "unknown"
            mode_confidence, mode_index = probabilities["mode"][i].max(0)
            mode = (
                MODES[int(mode_index)]
                if float(mode_confidence) >= max(0.99, self.calibration["mode_threshold"])
                else "unknown"
            )
            key_or_button = candidate.startswith(("key:", "mouse:"))
            if status == "supported" and key_or_button and mode == "unknown":
                status = "unknown"
            statuses.append(status)
            modes.append(mode if key_or_button else "unknown")
            confidences.append(float(confidence))

        # Every relation between accepted supported candidates must be known.
        # Unknown cross-component relations cannot create paying singletons.
        adjacency = {i: {i} for i, status in enumerate(statuses) if status == "supported"}
        uncertain = set()
        for i in adjacency:
            for j in adjacency:
                if j <= i:
                    continue
                confidence, index = probabilities["pair"][i, j].max(0)
                if float(confidence) < max(0.99, self.calibration["pair_threshold"]):
                    uncertain.update((i, j))
                elif int(index) == 0:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        names, visited = {}, set()
        for origin in adjacency:
            if origin in visited:
                continue
            pending, component = [origin], set()
            while pending:
                index = pending.pop()
                if index not in component:
                    component.add(index)
                    pending.extend(adjacency[index] - component)
            visited.update(component)
            key_modes = {
                modes[k] for k in component if CANDIDATES[k].startswith(("key:", "mouse:"))
            }
            clique = all(component <= adjacency[k] for k in component)
            if component & uncertain or not clique or len(key_modes) > 1:
                continue
            names.update({i: CANDIDATES[min(component)] for i in component})
        # Unsupported labels have no semantic alias target. Penalize each
        # executed physical candidate once; never consult its pair logits.
        names.update({i: CANDIDATES[i] for i, status in enumerate(statuses)
                      if status == "unsupported"})
        return ControlPredictionTable(
            text_hash, self.model_hash, self.calibration_hash,
            tuple(ControlPrediction(
                candidate, statuses[i], modes[i], names.get(i), confidences[i],
                deduplication_known=statuses[i] == "unknown" or i in names,
            ) for i, candidate in enumerate(CANDIDATES)),
        )


def save_text_control_artifact(
    directory, *, encoder, head, calibration, validation, training: dict
) -> dict:
    """Save a reviewable future fit; failed gates remain explicitly unready."""
    _checked_calibration(calibration)
    _check_head_calibration(head, calibration)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoder.save_resources(directory / "language")
    torch.save(encoder.state_dict(), directory / "encoder.pt")
    torch.save(head.state_dict(), directory / "head.pt")
    (directory / "calibration.json").write_text(json.dumps(calibration, sort_keys=True) + "\n")
    (directory / "validation.json").write_text(json.dumps(validation, sort_keys=True) + "\n")
    files = {
        str(path.relative_to(directory)): _sha_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    gates = calibration_gates(validation)
    manifest = {
        "version": MODEL_VERSION,
        "head_architecture": head.architecture,
        "encoder_id": TEXT_ID,
        "revision": TEXT_REVISION,
        "candidates": list(CANDIDATES),
        "files": files,
        "training": training,
        "encoder_state_hash": encoder_state_hash(encoder.state_dict()),
        "gates": gates,
        "ready": bool(training.get("trained") and gates["passed"]),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest


def load_text_control_reward(directory, *, shared_encoder=None):
    """Local weights-only restore, verified resources, calibration and readiness."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    from carryrl.rewards.control_text_factorized import FACTORIZED_MODEL_VERSION
    from carryrl.rewards.control_text_structured import STRUCTURED_MODEL_VERSION

    if manifest.get("version") == FACTORIZED_MODEL_VERSION:
        from carryrl.rewards.control_text_factorized_artifact import load_factorized_control_reward

        return load_factorized_control_reward(directory, shared_encoder=shared_encoder)

    if manifest.get("version") == STRUCTURED_MODEL_VERSION:
        from carryrl.rewards.control_text_structured_artifact import load_structured_control_reward

        return load_structured_control_reward(directory, shared_encoder=shared_encoder)
    if (
        manifest.get("version") != MODEL_VERSION
        or manifest.get("encoder_id") != TEXT_ID
        or manifest.get("revision") != TEXT_REVISION
        or manifest.get("candidates") != list(CANDIDATES)
    ):
        raise ValueError("unsupported text control model identity")
    files = manifest.get("files", {})
    if not {"head.pt", "calibration.json", "validation.json"} <= files.keys():
        raise ValueError("text reward artifact is incomplete")
    actual_files = {
        str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()
    }
    if actual_files != set(files) | {"manifest.json"}:
        raise ValueError("text reward bundle contains unhashed or missing resources")
    for name, expected in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or _sha_file(path) != expected:
            raise ValueError(f"text reward artifact hash mismatch: {name}")
    validation = json.loads((directory / "validation.json").read_text())
    gates = calibration_gates(validation)
    if (
        not manifest.get("ready")
        or not manifest.get("training", {}).get("trained")
        or not gates["passed"]
        or manifest.get("gates") != gates
    ):
        raise ValueError("text reward is not trained and validated; readiness gates failed")
    if shared_encoder is None:
        if "encoder.pt" not in files:
            raise ValueError("text reward artifact requires its checkpoint's shared E5 encoder")
        encoder_state = torch.load(directory / "encoder.pt", map_location="cpu", weights_only=True)
        if encoder_state_hash(encoder_state) != manifest.get("encoder_state_hash"):
            raise ValueError("frozen E5 state hash mismatch")
        encoder = MultilingualInstructionEncoder.from_resources(directory / "language")
        encoder.load_state_dict(encoder_state, strict=True)
    else:
        encoder = shared_encoder
        if getattr(encoder, "_awaiting_checkpoint_state", False) or encoder_state_hash(
            encoder.state_dict()
        ) != manifest.get("encoder_state_hash"):
            raise ValueError("shared E5 weights differ from the frozen reward artifact")
        if encoder.max_tokens != 512:
            raise ValueError("text control reward requires the full strict 512 token budget")
    encoder.eval().requires_grad_(False)
    architecture = manifest.get("head_architecture", LEGACY_HEAD_ARCHITECTURE)
    head = text_control_head_for_architecture(architecture)
    head.load_state_dict(
        torch.load(directory / "head.pt", map_location="cpu", weights_only=True), strict=True
    )
    calibration = json.loads((directory / "calibration.json").read_text())
    if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
        raise ValueError("non-finite text control head weights")
    identity = {
        "version": MODEL_VERSION,
        "head": files["head.pt"],
        "encoder": manifest["encoder_state_hash"],
    }
    if architecture != LEGACY_HEAD_ARCHITECTURE:
        identity["head_architecture"] = architecture
    model_hash = content_hash(identity)
    model = FrozenTextControlReward(
        encoder,
        head,
        model_hash=model_hash,
        calibration=calibration,
        calibration_hash=files["calibration.json"],
        _verified=True,
    )
    model._manifest = manifest
    # Keep small resources immutable in memory: checkpoint retention may archive
    # the source directory before the next complete checkpoint is exported.
    model._resources = {
        name: (directory / name).read_bytes() for name in files if name != "encoder.pt"
    }
    return model
