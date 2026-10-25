"""Joint instruction-only control states with learned canonical alias pointers.

The head is independent of the historical heads. Only frozen contextual E5
features and lexical mention locations enter it; no game metadata is consulted.
"""

import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from carryrl.agent.language import TEXT_WIDTH
from carryrl.rewards.control_text import (
    CANDIDATES,
    ControlPrediction,
    ControlPredictionTable,
    _sha_file,
    encoder_state_hash,
)
from carryrl.rewards.control_text_grounding import build_grounding

STRUCTURED_HEAD_ARCHITECTURE = "grounded_joint_canonical_controls.v8"
STRUCTURED_MODEL_VERSION = "carryrl.control_text.joint_canonical.v2"
STRUCTURED_DECODER_VERSION = "joint_canonical_groups.v1"
STRUCTURED_POPULATION_CONTRACT = "supported_decisions.v1"
STRUCTURED_MODES = ("held", "press", "relative")
UNKNOWN_STATE = 0
UNSUPPORTED_STATE = 1
JOINT_STATE_COUNT = 2 + len(STRUCTURED_MODES) * len(CANDIDATES)
NUM_JOINT_STATES = JOINT_STATE_COUNT
_DISCRETE = tuple(candidate.startswith(("key:", "mouse:")) for candidate in CANDIDATES)


def joint_state(mode: str, representative: int) -> int:
    """Stable class encoding; representatives use canonical candidate indices."""
    if mode not in STRUCTURED_MODES or type(representative) is not int:
        raise ValueError("invalid joint mode or representative")
    if not 0 <= representative < len(CANDIDATES):
        raise ValueError("invalid canonical representative")
    return 2 + STRUCTURED_MODES.index(mode) * len(CANDIDATES) + representative


def decode_joint_state(index: int) -> tuple[str, int | None]:
    if type(index) is not int or not 0 <= index < JOINT_STATE_COUNT:
        raise ValueError("invalid joint state")
    if index < 2:
        return ("unknown" if index == UNKNOWN_STATE else "unsupported"), None
    mode, representative = divmod(index - 2, len(CANDIDATES))
    return STRUCTURED_MODES[mode], representative


def structured_targets(rows, device=None) -> torch.Tensor:
    """Build canonical joint labels, independent of label order or group names.

    Legacy relative-axis mode annotations are ignored: axes are physical
    relative commands. Modes constrain only keys and buttons within an alias
    group. Unsupported/unknown labels cannot carry a positive alias group.
    """
    targets = []
    for row in rows:
        labels = row.get("labels") if isinstance(row, Mapping) else None
        if not isinstance(labels, (list, tuple)) or len(labels) != len(CANDIDATES):
            raise ValueError("structured labels require exactly 36 candidates")
        by_candidate, groups = {}, {}
        for label in labels:
            if not isinstance(label, Mapping) or not {
                "candidate", "status", "mode", "alias_group"
            } <= set(label):
                raise ValueError("invalid structured control label")
            candidate = label.get("candidate")
            if candidate not in CANDIDATES or candidate in by_candidate:
                raise ValueError("structured label candidates must be unique and complete")
            status, mode, group = label.get("status"), label.get("mode"), label.get("alias_group")
            if status not in ("supported", "unsupported", "unknown"):
                raise ValueError("invalid structured label status")
            if mode not in (*STRUCTURED_MODES, "unknown"):
                raise ValueError("invalid structured label mode")
            if type(group) is not int or group < 0:
                raise ValueError("alias groups must be nonnegative integers")
            index = CANDIDATES.index(candidate)
            if status == "supported":
                if group == 0 or (_DISCRETE[index] and mode not in ("held", "press")):
                    raise ValueError("supported controls require a group and a physical input mode")
                groups.setdefault(group, []).append(index)
            elif group != 0:
                raise ValueError("unsupported/unknown controls cannot carry a positive alias group")
            by_candidate[candidate] = label
        for members in groups.values():
            modes = {by_candidate[CANDIDATES[i]]["mode"] for i in members if _DISCRETE[i]}
            if len(modes) > 1:
                raise ValueError("alias group contains conflicting discrete modes")
        encoded = []
        for index, candidate in enumerate(CANDIDATES):
            label = by_candidate[candidate]
            if label["status"] != "supported":
                encoded.append(UNKNOWN_STATE if label["status"] == "unknown" else UNSUPPORTED_STATE)
            else:
                mode = label["mode"] if _DISCRETE[index] else "relative"
                encoded.append(joint_state(mode, min(groups[label["alias_group"]])))
        targets.append(encoded)
    return torch.tensor(targets, dtype=torch.long, device=device).reshape(-1, len(CANDIDATES))


def _valid_states() -> torch.Tensor:
    valid = torch.ones(len(CANDIDATES), JOINT_STATE_COUNT, dtype=torch.bool)
    for index in range(len(CANDIDATES)):
        for mode in STRUCTURED_MODES:
            for representative in range(len(CANDIDATES)):
                valid[index, joint_state(mode, representative)] = (
                    representative <= index and ((mode != "relative") == _DISCRETE[index])
                )
    return valid


class _CandidateContextLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.cross = nn.MultiheadAttention(width, 4, batch_first=True)
        self.self_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.self_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.output_norm = nn.LayerNorm(width)

    def forward(self, candidates, tokens, padding_mask):
        context, _ = self.cross(
            candidates, tokens, tokens, key_padding_mask=padding_mask, need_weights=False
        )
        candidates = self.cross_norm(candidates + context)
        context, _ = self.self_attention(candidates, candidates, candidates, need_weights=False)
        candidates = self.self_norm(candidates + context)
        return self.output_norm(candidates + self.ff(candidates))


def _nearest_mention_offsets(grounding, padding_mask):
    """Signed token distance from the nearest mention; ties use the earlier one.

    Prefix/suffix scans keep storage linear in batch × candidates × tokens.
    Positions count real tokens so inserting padding cannot alter distances.
    Candidates without a mention get zero offsets and are gated separately.
    """
    length = grounding.shape[-1]
    positions = (~padding_mask).cumsum(-1)[:, None]
    previous = torch.where(grounding, positions, -length).cummax(-1).values
    following = torch.where(grounding, positions, 2 * length).flip(-1).cummin(-1).values.flip(-1)
    after = positions - previous
    before = following - positions
    offsets = torch.where(after <= before, after, -before)
    return offsets.masked_fill(~grounding.any(-1, keepdim=True), 0)


class JointStructuredControlHead(nn.Module):
    """Mention-local input attention, two global layers and a joint readout."""

    architecture = STRUCTURED_HEAD_ARCHITECTURE

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(TEXT_WIDTH, 128)
        self.queries = nn.Parameter(torch.randn(len(CANDIDATES), 128) * 0.02)
        self.local_query = nn.Linear(128, 128)
        # Independently learn how quickly context fades before/after a mention.
        self.local_distance_decay = nn.Parameter(torch.full((2,), math.log(math.expm1(.25))))
        self.mention_fusion = nn.Linear(128 * 2 + 1, 128)
        self.input_norm = nn.LayerNorm(128)
        self.layers = nn.ModuleList([_CandidateContextLayer(128) for _ in range(2)])
        self.state_logits = nn.Linear(128, 2 + len(STRUCTURED_MODES))
        self.pointer_query = nn.Linear(128, 128)
        self.pointer_key = nn.Linear(128, 128)
        self.register_buffer("valid_states", _valid_states(), persistent=False)

    def _local_context(self, projected, identities, grounding, padding_mask):
        offsets = _nearest_mention_offsets(grounding, padding_mask).to(projected.dtype)
        decay = nn.functional.softplus(self.local_distance_decay)
        distance_bias = offsets.clamp_max(0) * decay[0] - offsets.clamp_min(0) * decay[1]
        scores = torch.bmm(self.local_query(identities), projected.transpose(1, 2)) / math.sqrt(128)
        scores = (scores + distance_bias).masked_fill(padding_mask[:, None], -torch.inf)
        context = torch.bmm(scores.softmax(-1), projected)
        return context.masked_fill(~grounding.any(-1, keepdim=True), 0)

    def forward(self, tokens, padding_mask=None, *, grounding=None):
        if (tokens.ndim != 3 or tokens.shape[-1] != TEXT_WIDTH
                or not 0 < tokens.shape[0] or not 0 < tokens.shape[1] <= 512):
            raise ValueError("expected batch of at most 512 contextual E5 tokens")
        batch, length = tokens.shape[:2]
        if padding_mask is None:
            padding_mask = torch.zeros(batch, length, dtype=torch.bool, device=tokens.device)
        elif padding_mask.shape != (batch, length) or padding_mask.dtype != torch.bool:
            raise ValueError("padding mask must be boolean [batch, tokens]")
        padding_mask = padding_mask.to(tokens.device)
        if padding_mask.all(-1).any():
            raise ValueError("each instruction requires at least one unpadded token")
        if grounding is None or grounding.shape != (batch, len(CANDIDATES), length):
            raise ValueError("grounding must have shape [batch, 36, tokens]")
        grounding = grounding.to(tokens.device)
        if grounding.dtype != torch.bool:
            if (not torch.is_floating_point(grounding)
                    or not ((grounding == 0) | (grounding == 1)).all()):
                raise ValueError("grounding must be boolean or binary float")
            grounding = grounding.bool()
        grounding = grounding & ~padding_mask[:, None]
        projected = self.projection(tokens.masked_fill(padding_mask[:, :, None], 0))
        identities = self.queries[None].expand(batch, -1, -1)
        local = self._local_context(projected, identities, grounding, padding_mask)
        candidates = self.input_norm(identities + self.mention_fusion(torch.cat(
            (identities, local, grounding.any(-1, keepdim=True).to(projected.dtype)), dim=-1
        )))
        for layer in self.layers:
            candidates = layer(candidates, projected, padding_mask)
        state = self.state_logits(candidates)
        pointers = torch.bmm(
            self.pointer_query(candidates), self.pointer_key(candidates).transpose(1, 2)
        ) / math.sqrt(128)
        supported = state[:, :, 2:, None] + pointers[:, :, None]
        logits = torch.cat((state[:, :, :2], supported.flatten(-2)), dim=-1)
        return logits.masked_fill(~self.valid_states, -torch.inf)


def checked_structured_calibration(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError("structured calibration must be a record")
    if (record.get("architecture") != STRUCTURED_HEAD_ARCHITECTURE
            or record.get("decoder_version") != STRUCTURED_DECODER_VERSION
            or record.get("population_contract") != STRUCTURED_POPULATION_CONTRACT):
        raise ValueError("structured calibration architecture/decoder/population contract mismatch")
    for key, lower, inclusive in (("joint_temperature", 0, False), ("joint_threshold", .99, True)):
        value = record.get(key)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or (value < lower if inclusive else value <= lower)
                or (key == "joint_threshold" and value > 1)):
            raise ValueError(f"invalid structured calibration {key}")
    if any("class_weight" in key or "prior" in key for key in record):
        raise ValueError("structured calibration cannot apply class-prior corrections")
    return record


def calibrated_structured_logits(logits, calibration):
    checked_structured_calibration(calibration)
    if logits.shape[-1] != JOINT_STATE_COUNT:
        raise ValueError("structured logits require 110 joint classes")
    return logits / calibration["joint_temperature"]


def decode_structured_predictions(logits, calibration, text_hash, model_hash, calibration_hash):
    """Accept only confident joint decisions with resolvable canonical pointers.

    Pointer consistency is a structural safeguard, not evidence of semantic
    correctness. Unresolved groups never become bonus-paying singleton groups.
    """
    if logits.shape != (len(CANDIDATES), JOINT_STATE_COUNT):
        raise ValueError("structured decoder requires [36, 110] logits")
    if (torch.isnan(logits).any() or torch.isposinf(logits).any()
            or not torch.isfinite(logits).any(-1).all()):
        raise ValueError("structured decoder requires finite distributions")
    probabilities = calibrated_structured_logits(logits, calibration).softmax(-1).detach().cpu()
    confidences, states = probabilities.max(-1)
    threshold = calibration["joint_threshold"]
    statuses, modes, pointers = [], [], []
    for i, state in enumerate(states.tolist()):
        kind, representative = decode_joint_state(state)
        if float(confidences[i]) < threshold or kind == "unknown":
            statuses.append("unknown")
            modes.append("unknown")
            pointers.append(None)
        elif kind == "unsupported":
            statuses.append("unsupported")
            modes.append("unknown")
            pointers.append(None)
        elif (kind != "relative") != _DISCRETE[i]:
            statuses.append("unknown")
            modes.append("unknown")
            pointers.append(None)
        else:
            statuses.append("supported")
            modes.append(kind if _DISCRETE[i] else "unknown")
            pointers.append(representative)
    groups = {}
    for i, representative in enumerate(pointers):
        if representative is not None:
            groups.setdefault(representative, []).append(i)
    names = {i: CANDIDATES[i] for i, status in enumerate(statuses) if status == "unsupported"}
    for representative, members in groups.items():
        discrete_modes = {modes[i] for i in members if _DISCRETE[i]}
        if (pointers[representative] != representative or any(i < representative for i in members)
                or len(discrete_modes) > 1):
            continue
        names.update({i: CANDIDATES[representative] for i in members})
    return ControlPredictionTable(
        text_hash, model_hash, calibration_hash,
        tuple(ControlPrediction(
            candidate, statuses[i], modes[i], names.get(i), float(confidences[i]),
            deduplication_known=statuses[i] == "unknown" or i in names,
        ) for i, candidate in enumerate(CANDIDATES)),
    )


class StructuredTextControlReward:
    """Frozen instruction-only predictor; artifact verification belongs to loading."""

    def __init__(
        self, encoder, head, *, model_hash, calibration, calibration_hash, _verified=False
    ):
        if not _verified:
            raise ValueError("use a verified structured artifact; random heads are forbidden")
        if getattr(head, "architecture", None) != STRUCTURED_HEAD_ARCHITECTURE:
            raise ValueError("structured predictor requires the structured head")
        for value in (model_hash, calibration_hash):
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError("structured predictor identities must be SHA256 hashes")
        self.calibration = dict(checked_structured_calibration(calibration))
        self.encoder = encoder
        if isinstance(encoder, nn.Module):
            encoder.eval().requires_grad_(False)
        self.head = head.eval().requires_grad_(False)
        self.model_hash, self.calibration_hash = model_hash, calibration_hash
        self._cache = OrderedDict()

    @property
    def identity(self):
        return {"version": STRUCTURED_MODEL_VERSION, "model_hash": self.model_hash,
                "calibration_hash": self.calibration_hash}

    def save_resources(self, destination, *, shared_encoder=True):
        """Re-export a loaded verified bundle for a policy checkpoint."""
        if not hasattr(self, "_manifest") or not hasattr(self, "_resources"):
            raise ValueError("only a loaded verified structured artifact can export resources")
        manifest = json.loads(json.dumps(self._manifest))
        if encoder_state_hash(self.encoder.state_dict()) != manifest["encoder_state_hash"]:
            raise ValueError("shared E5 weights changed after structured reward loading")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
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
        if not isinstance(text, str) or not text.strip():
            raise ValueError("structured control reward requires nonempty instruction text")
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        key = (text_hash, self.model_hash, self.calibration_hash)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        tokens = self.encoder.encode_instruction_tokens(text)
        grounding = build_grounding(text, self.encoder.tokenizer, expected_tokens=len(tokens))
        device = next(self.head.parameters()).device
        logits = self.head(tokens[None].to(device), grounding=grounding[None].to(device))
        table = decode_structured_predictions(
            logits[0], self.calibration, text_hash, self.model_hash, self.calibration_hash
        )
        self._cache[key] = table
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        return table


StructuredTextControlHead = JointStructuredControlHead
FrozenStructuredTextControlReward = StructuredTextControlReward
