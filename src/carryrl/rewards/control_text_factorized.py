"""Frozen instruction-only factorized applicability, timing and symmetric relations."""

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import nn

from carryrl.agent.language import TEXT_WIDTH
from carryrl.rewards.control_text import CANDIDATES, _sha_file, content_hash, encoder_state_hash
from carryrl.rewards.control_text_grounding import build_grounding
from carryrl.rewards.control_text_structured import JointStructuredControlHead

FACTORIZED_HEAD_ARCHITECTURE = "grounded_local_factorized_controls.v1"
FACTORIZED_MODEL_VERSION = "carryrl.control_text.factorized.v1"
FACTORIZED_DECODER_VERSION = "physical_modes_symmetric_cliques.v2"
FACTORIZED_POPULATION_CONTRACT = "factorized_applicability_mode_pairs.v1"
APPLICABILITY = ("supported", "unsupported", "unspecified")
MODES = ("held", "press", "relative", "unspecified")
PAIRS = ("same", "different", "unspecified")


@dataclass(frozen=True)
class FieldDecision:
    value: str
    confidence: float
    accepted: bool
    reason: str | None = None

    def __post_init__(self):
        if (
            self.value not in set(APPLICABILITY + MODES + PAIRS)
            or type(self.confidence) not in (int, float)
            or not 0 <= self.confidence <= 1
            or type(self.accepted) is not bool
            or self.reason
            not in (None, "low_confidence", "insufficient_information", "inconsistent")
        ):
            raise ValueError("invalid factorized field decision")
        if self.accepted != (self.reason is None) or (
            self.accepted and self.value == "unspecified"
        ):
            raise ValueError("inconsistent factorized field acceptance")
        if self.accepted and self.confidence < 0.99:
            raise ValueError("accepted factorized decisions require confidence >= .99")


@dataclass(frozen=True)
class PairDecision:
    left: str
    right: str
    decision: FieldDecision


@dataclass(frozen=True)
class FactorizedControlPrediction:
    candidate: str
    applicability_decision: FieldDecision
    mode_decision: FieldDecision
    group: str | None = None
    deduplication_known: bool = False

    def __post_init__(self):
        if (
            self.candidate not in CANDIDATES
            or not isinstance(self.applicability_decision, FieldDecision)
            or self.applicability_decision.value not in APPLICABILITY
            or not isinstance(self.mode_decision, FieldDecision)
            or self.mode_decision.value not in MODES
            or type(self.deduplication_known) is not bool
        ):
            raise ValueError("invalid factorized prediction fields")
        if self.group is not None and (not isinstance(self.group, str) or not self.group):
            raise ValueError("invalid factorized alias group")
        if self.group is not None and (not self.deduplication_known or self.status == "unknown"):
            raise ValueError("unresolved factorized prediction cannot claim a group")
        if self.status == "supported" and self.deduplication_known and self.group is None:
            raise ValueError("resolved supported prediction requires a group")

    @property
    def status(self):
        d = self.applicability_decision
        return d.value if d.accepted else "unknown"

    @property
    def mode(self):
        return self.mode_decision.value if self.mode_decision.accepted else "unknown"

    @property
    def mode_reliable(self):
        return self.mode_decision.accepted

    @property
    def confidence(self):
        return self.applicability_decision.confidence

    @property
    def reward_eligible(self):
        return self.status == "unsupported" or (
            self.status == "supported" and self.deduplication_known and self.group is not None
        )


@dataclass(frozen=True)
class FactorizedControlPredictionTable:
    text_hash: str
    model_hash: str
    calibration_hash: str
    predictions: tuple[FactorizedControlPrediction, ...]
    pair_decisions: tuple[PairDecision, ...]

    def __post_init__(self):
        for value in (self.text_hash, self.model_hash, self.calibration_hash):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("factorized table identities must be SHA256 hashes")
        if tuple(p.candidate for p in self.predictions) != CANDIDATES:
            raise ValueError("factorized table requires all candidates in canonical order")
        expected = {(a, b) for i, a in enumerate(CANDIDATES) for b in CANDIDATES[i + 1 :]}
        if {(p.left, p.right) for p in self.pair_decisions} != expected:
            raise ValueError("factorized table requires each unordered pair exactly once")
        if len(self.pair_decisions) != len(expected):
            raise ValueError("duplicate factorized pair")

    def __iter__(self):
        return iter(self.predictions)

    @property
    def identity(self):
        return content_hash(self.to_dict())

    def to_dict(self):
        result = asdict(self)
        for row, prediction in zip(result["predictions"], self.predictions, strict=True):
            row.update(
                status=prediction.status,
                mode=prediction.mode,
                confidence=prediction.confidence,
                reward_eligible=prediction.reward_eligible,
                mode_reliable=prediction.mode_reliable,
            )
        return result


class FactorizedControlHead(JointStructuredControlHead):
    """The unchanged V8 mention-local trunk followed by independent readouts."""

    architecture = FACTORIZED_HEAD_ARCHITECTURE

    def __init__(self):
        super().__init__()
        del self.state_logits, self.pointer_query, self.pointer_key
        self.applicability = nn.Linear(128, 3)
        self.mode = nn.Linear(128, 4)
        self.pair = nn.Sequential(nn.Linear(128 * 3, 128), nn.GELU(), nn.Linear(128, 3))

    def forward(self, tokens, padding_mask=None, *, grounding=None):
        # Preserve the V8 validation and mention-local computation before the
        # independently supervised readouts.
        if (
            tokens.ndim != 3
            or tokens.shape[-1] != TEXT_WIDTH
            or not 0 < tokens.shape[0]
            or not 0 < tokens.shape[1] <= 512
        ):
            raise ValueError("expected batch of at most 512 contextual E5 tokens")
        batch, length = tokens.shape[:2]
        if padding_mask is None:
            padding_mask = torch.zeros(batch, length, dtype=torch.bool, device=tokens.device)
        elif padding_mask.shape != (batch, length) or padding_mask.dtype != torch.bool:
            raise ValueError("padding mask must be boolean [batch,tokens]")
        padding_mask = padding_mask.to(tokens.device)
        if padding_mask.all(-1).any():
            raise ValueError("each instruction requires an unpadded token")
        if grounding is None or grounding.shape != (batch, len(CANDIDATES), length):
            raise ValueError("grounding must have shape [batch,36,tokens]")
        grounding = grounding.to(tokens.device)
        if grounding.dtype != torch.bool:
            if (
                not torch.is_floating_point(grounding)
                or not ((grounding == 0) | (grounding == 1)).all()
            ):
                raise ValueError("grounding must be boolean or binary float")
            grounding = grounding.bool()
        grounding = grounding & ~padding_mask[:, None]
        projected = self.projection(tokens.masked_fill(padding_mask[:, :, None], 0))
        identities = self.queries[None].expand(batch, -1, -1)
        local = self._local_context(projected, identities, grounding, padding_mask)
        candidates = self.input_norm(
            identities
            + self.mention_fusion(
                torch.cat(
                    (identities, local, grounding.any(-1, keepdim=True).to(projected.dtype)), dim=-1
                )
            )
        )
        for layer in self.layers:
            candidates = layer(candidates, projected, padding_mask)
        a, b = candidates[:, :, None], candidates[:, None, :]
        # Sum, absolute difference and product are invariant under swapping inputs.
        pair_features = torch.cat((a + b, (a - b).abs(), a * b), dim=-1)
        return {
            "applicability": self.applicability(candidates),
            "mode": self.mode(candidates),
            "pair": self.pair(pair_features),
        }


def checked_factorized_calibration(record):
    if not isinstance(record, dict) or any(
        record.get(k) != v
        for k, v in (
            ("architecture", FACTORIZED_HEAD_ARCHITECTURE),
            ("decoder_version", FACTORIZED_DECODER_VERSION),
            ("population_contract", FACTORIZED_POPULATION_CONTRACT),
        )
    ):
        raise ValueError("factorized calibration identity mismatch")
    for name in ("applicability", "mode", "pair"):
        t, threshold = record.get(f"{name}_temperature"), record.get(f"{name}_threshold")
        if (
            type(t) not in (int, float)
            or not 0 < t < float("inf")
            or type(threshold) not in (int, float)
            or not 0.99 <= threshold <= 1
        ):
            raise ValueError("invalid factorized temperature/threshold")
    if any("prior" in key or "class_weight" in key for key in record):
        raise ValueError("factorized calibration cannot correct class priors")
    return record


def _decisions(logits, names, calibration, kind):
    if logits.shape[-1] != len(names) or not torch.isfinite(logits).all():
        raise ValueError("factorized decoder requires finite complete logits")
    confidence, indices = (
        (logits.detach().cpu() / calibration[f"{kind}_temperature"]).softmax(-1).max(-1)
    )
    return [
        FieldDecision(
            names[i],
            float(c),
            bool(c >= calibration[f"{kind}_threshold"] and names[i] != "unspecified"),
            "low_confidence"
            if c < calibration[f"{kind}_threshold"]
            else "insufficient_information"
            if names[i] == "unspecified"
            else None,
        )
        for c, i in zip(confidence.flatten(), indices.flatten(), strict=True)
    ]


def decode_factorized_predictions(logits, calibration, text_hash, model_hash, calibration_hash):
    """Preserve partial semantics; only complete consistent cliques can pay."""
    checked_factorized_calibration(calibration)
    n = len(CANDIDATES)
    if (
        not isinstance(logits, dict)
        or set(logits) != {"applicability", "mode", "pair"}
        or logits["applicability"].shape != (n, 3)
        or logits["mode"].shape != (n, 4)
        or logits["pair"].shape != (n, n, 3)
    ):
        raise ValueError("invalid factorized logit shapes")
    if not torch.equal(logits["pair"], logits["pair"].transpose(0, 1)):
        raise ValueError("pair logits must be symmetric")
    applicability = _decisions(logits["applicability"], APPLICABILITY, calibration, "applicability")
    modes = _decisions(logits["mode"], MODES, calibration, "mode")
    pairs = _decisions(logits["pair"], PAIRS, calibration, "pair")
    supported = [i for i, d in enumerate(applicability) if d.accepted and d.value == "supported"]
    for i, mode in enumerate(modes):
        if mode.accepted and (
            (mode.value == "relative") == CANDIDATES[i].startswith(("key:", "mouse:"))
        ):
            modes[i] = replace(mode, accepted=False, reason="inconsistent")
    components, unseen = [], set(supported)
    while unseen:
        members, pending = set(), [min(unseen)]
        while pending:
            i = pending.pop()
            if i in members:
                continue
            members.add(i)
            pending.extend(
                j
                for j in supported
                if j not in members
                and pairs[i * n + j].accepted
                and pairs[i * n + j].value == "same"
            )
        unseen -= members
        components.append(members)
    names = {}
    inconsistent_pairs = set()
    for members in components:
        internal = [(i, j) for i in members for j in members if i < j]
        clique = all(
            pairs[i * n + j].accepted and pairs[i * n + j].value == "same" for i, j in internal
        )
        if not clique:
            inconsistent_pairs.update(internal)
        separate = all(
            pairs[i * n + j].accepted and pairs[i * n + j].value == "different"
            for i in members
            for j in supported
            if j not in members
        )
        if clique and separate:
            names.update({i: CANDIDATES[min(members)] for i in members})
    pair_records = tuple(
        PairDecision(
            CANDIDATES[i],
            CANDIDATES[j],
            replace(pairs[i * n + j], accepted=False, reason="inconsistent")
            if (i, j) in inconsistent_pairs
            else pairs[i * n + j],
        )
        for i in range(n)
        for j in range(i + 1, n)
    )
    predictions = tuple(
        FactorizedControlPrediction(
            candidate,
            applicability[i],
            modes[i],
            candidate
            if applicability[i].accepted and applicability[i].value == "unsupported"
            else names.get(i),
            bool(
                i in names
                or (applicability[i].accepted and applicability[i].value == "unsupported")
            ),
        )
        for i, candidate in enumerate(CANDIDATES)
    )
    return FactorizedControlPredictionTable(
        text_hash, model_hash, calibration_hash, predictions, pair_records
    )


class FactorizedTextControlReward:
    def __init__(
        self, encoder, head, *, model_hash, calibration, calibration_hash, _verified=False
    ):
        if not _verified or getattr(head, "architecture", None) != FACTORIZED_HEAD_ARCHITECTURE:
            raise ValueError("use a verified factorized artifact")
        self.calibration = dict(checked_factorized_calibration(calibration))
        self.encoder, self.head = encoder, head.eval().requires_grad_(False)
        self.encoder.eval().requires_grad_(False)
        self.model_hash, self.calibration_hash = model_hash, calibration_hash
        self._cache = OrderedDict()

    @property
    def identity(self):
        return {
            "version": FACTORIZED_MODEL_VERSION,
            "model_hash": self.model_hash,
            "calibration_hash": self.calibration_hash,
        }

    def save_resources(self, destination, *, shared_encoder=True):
        if not hasattr(self, "_manifest") or not hasattr(self, "_resources"):
            raise ValueError("only loaded verified artifacts may export resources")
        manifest = json.loads(json.dumps(self._manifest))
        if encoder_state_hash(self.encoder.state_dict()) != manifest["encoder_state_hash"]:
            raise ValueError("frozen E5 changed after factorized reward loading")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        for name, payload in self._resources.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if shared_encoder:
            manifest["files"].pop("encoder.pt", None)
        else:
            torch.save(self.encoder.state_dict(), destination / "encoder.pt")
            manifest["files"]["encoder.pt"] = _sha_file(destination / "encoder.pt")
        manifest["shared_encoder"] = shared_encoder
        (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
        return destination

    @torch.no_grad()
    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("factorized reward requires nonempty instruction text")
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in self._cache:
            self._cache.move_to_end(digest)
            return self._cache[digest]
        tokens = self.encoder.encode_instruction_tokens(text)
        grounding = build_grounding(text, self.encoder.tokenizer, expected_tokens=len(tokens))
        device = next(self.head.parameters()).device
        logits = self.head(tokens[None].to(device), grounding=grounding[None].to(device))
        table = decode_factorized_predictions(
            {k: v[0] for k, v in logits.items()},
            self.calibration,
            digest,
            self.model_hash,
            self.calibration_hash,
        )
        self._cache[digest] = table
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        return table
