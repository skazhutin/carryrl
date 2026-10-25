"""Fresh factorized fitting with TRAIN pair rehearsal and development-only selection."""

import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from carryrl.rewards.control_text import (
    CANDIDATES,
    calibration_gates,
    content_hash,
    encoder_state_hash,
)
from carryrl.rewards.control_text_factorized import (
    FACTORIZED_DECODER_VERSION,
    FACTORIZED_HEAD_ARCHITECTURE,
    FACTORIZED_MODEL_VERSION,
    FACTORIZED_POPULATION_CONTRACT,
    FactorizedControlHead,
    decode_factorized_predictions,
)
from carryrl.rewards.control_text_training import (
    _batch,
    development_split_identity,
    developmental_feasibility_gate,
    load_verified_disk_token_features,
    prepare_disk_token_features,
)

FACTORIZED_ADAMW_WEIGHT_DECAY = 0.01
LOSS_PROTOCOL = "independent_factorized_train_pair_rehearsal.v2"
STAGED_TRAINING_PROTOCOL = "staged_required_pair_margin.v1"
STAGED_TRAIN_RECORDS = 15264
STAGED_BASE_EPOCHS = 8
STAGED_CONTINUATION_UPDATES = 2200
STAGED_MARGIN_UPDATES = 1000


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("factorized text head wall-clock budget exhausted")


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protocol_identity(training_protocol=LOSS_PROTOCOL):
    if training_protocol not in (LOSS_PROTOCOL, STAGED_TRAINING_PROTOCOL):
        raise ValueError("unknown factorized training protocol")
    protocol = {
        "version": "carryrl.factorized_fit_protocol.v2",
        "probe_seed": 20261029,
        "full_seed": 20261030,
        "probe_max_epochs": 8,
        "full_max_epochs": 10,
        "probe_max_seconds": 4200,
        "full_max_seconds": 7200,
        "batch_size": 32,
        "learning_rate": 0.0003,
        "dev_temperature_groups": 300,
        "dev_selection_groups": 300,
        "threshold": 0.99,
        "minimum_coverage": 0.8,
        "zero_critical_errors": True,
        "loss_protocol": LOSS_PROTOCOL,
        "pair_training_objective": "half_focal_mean_plus_half_other_mean_when_both_present",
        "focal_metadata": "TRAIN_relational_panel.v1_only",
        "development_objective": "natural_population_means",
        "optimizer_weight_decay": FACTORIZED_ADAMW_WEIGHT_DECAY,
    }
    if training_protocol == STAGED_TRAINING_PROTOCOL:
        protocol.update(
            version="carryrl.factorized_staged_fit_protocol.v1",
            training_protocol=training_protocol,
            loss_protocol=training_protocol,
            full_max_epochs=STAGED_BASE_EPOCHS,
            train_records=STAGED_TRAIN_RECORDS,
            base_epochs=STAGED_BASE_EPOCHS,
            base_loss_protocol=LOSS_PROTOCOL,
            continuation_updates=STAGED_CONTINUATION_UPDATES,
            required_margin_updates=STAGED_MARGIN_UPDATES,
            required_margin_lambda=1.0,
            required_margin_temperature="selected_base_DEV_pair_temperature_frozen",
            required_margin_population="all_rows_mean_of_worst_gold_certifiable_required_edge",
            refinement_trainable="pair_readout_only",
            development_training_hyperparameters=["temperature", "selection"],
            candidate_selection="best_base_or_terminal_refined_same_DEV_key",
            device="cpu",
        )
    return protocol


def source_identity():
    """Pin the complete text learner/rendering/decoder and reward source closure."""
    root = Path(__file__).parent
    paths = list(root.glob("control_text*.py")) + [root / "composition.py"]
    paths += [root.parent / "agent" / name for name in ("language.py", "actions.py")]
    paths += [root.parents[2] / "scripts" / "train_text_control_factorized_reward.py"]
    return {
        str(path.relative_to(root.parents[1]))
        if path.is_relative_to(root.parents[1])
        else str(path.relative_to(root.parents[2])): _file_hash(path)
        for path in sorted(paths)
    }


def factorized_targets(rows, device=None):
    """Three independent label populations; pairs are supervised only if applicable."""
    from carryrl.rewards.control_text_factorized import APPLICABILITY, MODES, PAIRS

    targets = {"applicability": [], "mode": [], "pair": [], "pair_focal_mask": []}
    for row in rows:
        labels = row.get("factorized_labels")
        if not isinstance(labels, list) or len(labels) != len(CANDIDATES):
            raise ValueError("factorized labels require all candidates")
        by_candidate = {label["candidate"]: label for label in labels}
        if set(by_candidate) != set(CANDIDATES) or len(by_candidate) != len(labels):
            raise ValueError("factorized labels require unique canonical candidates")
        app, mode = [], []
        for candidate in CANDIDATES:
            label = by_candidate[candidate]
            if label.get("applicability") not in APPLICABILITY or label.get("mode") not in MODES:
                raise ValueError("invalid factorized applicability/mode")
            a, m = label["applicability"], label["mode"]
            if a != "supported" and m != "unspecified":
                raise ValueError("inapplicable controls cannot specify timing")
            if m != "unspecified" and (
                (m == "relative") == candidate.startswith(("key:", "mouse:"))
            ):
                raise ValueError("factorized timing conflicts with physical control")
            app.append(APPLICABILITY.index(a))
            mode.append(MODES.index(m) if a == "supported" else -100)
        pair = torch.full((len(CANDIDATES), len(CANDIDATES)), -100, dtype=torch.long)
        seen = set()
        for label in row.get("pair_labels", []):
            left, right = label.get("left"), label.get("right")
            if left not in CANDIDATES or right not in CANDIDATES or left == right:
                raise ValueError("invalid factorized pair candidates")
            i, j = sorted((CANDIDATES.index(left), CANDIDATES.index(right)))
            if (i, j) in seen or label.get("relation") not in PAIRS:
                raise ValueError("duplicate/invalid factorized pair")
            seen.add((i, j))
            relation = label["relation"]
            supported = app[i] == app[j] == APPLICABILITY.index("supported")
            if not supported and relation != "unspecified":
                raise ValueError("pair semantics require two applicable controls")
            if supported:
                pair[i, j] = pair[j, i] = PAIRS.index(relation)
        required = {
            (i, j)
            for i in range(len(CANDIDATES))
            for j in range(i + 1, len(CANDIDATES))
            if app[i] == app[j] == APPLICABILITY.index("supported")
        }
        if not required <= seen:
            raise ValueError("factorized labels require every applicable unordered pair")
        focal = torch.zeros_like(pair, dtype=torch.bool)
        if "relational_panel" in row:
            panel = row["relational_panel"]
            if row.get("split") != "train":
                raise ValueError("relational focal annotations are TRAIN only")
            if (
                not isinstance(panel, dict)
                or panel.get("version") != "carryrl.relational_panel.v1"
                or panel.get("state") not in PAIRS
            ):
                raise ValueError("invalid relational focal panel version/state")
            candidates = panel.get("focal_pair")
            if (
                not isinstance(candidates, (list, tuple))
                or len(candidates) != 2
                or any(candidate not in CANDIDATES for candidate in candidates)
                or candidates[0] == candidates[1]
            ):
                raise ValueError("relational focal pair requires two distinct physical commands")
            left, right = (CANDIDATES.index(candidate) for candidate in candidates)
            if app[left] != 0 or app[right] != 0 or pair[left, right] == -100:
                raise ValueError("relational focal pair requires two supported commands")
            if pair[left, right] != PAIRS.index(panel["state"]):
                raise ValueError("relational focal panel state differs from pair target")
            focal[left, right] = focal[right, left] = True
        targets["pair_focal_mask"].append(focal)
        targets["applicability"].append(app)
        targets["mode"].append(mode)
        targets["pair"].append(pair)
    return {
        key: torch.stack(values).to(device)
        if key in ("pair", "pair_focal_mask") and values
        else torch.tensor(
            values, dtype=torch.bool if key == "pair_focal_mask" else torch.long, device=device
        ).reshape(
            (-1, len(CANDIDATES), len(CANDIDATES))
            if key in ("pair", "pair_focal_mask")
            else (-1, len(CANDIDATES))
        )
        for key, values in targets.items()
    }


def factorized_required_pair_mask(targets):
    """Select unique gold edges required by certifiable supported components.

    A SAME-connected component is certifiable only when it is a complete SAME
    clique and every edge to another supported control is DIFFERENT. Include
    its internal SAME and external DIFFERENT edges; unspecified edges can block
    certification but never enter the returned upper-triangle mask. Singleton
    components follow the same external-edge rule.
    """
    app, pair = targets["applicability"], targets["pair"]
    n = len(CANDIDATES)
    if (
        not isinstance(app, torch.Tensor)
        or not isinstance(pair, torch.Tensor)
        or app.ndim != 2
        or app.shape[1] != n
        or pair.shape != (app.shape[0], n, n)
        or app.dtype != torch.long
        or pair.dtype != torch.long
        or app.device != pair.device
    ):
        raise ValueError("required pair mask needs factorized applicability/pair targets")
    if (
        ((app < 0) | (app > 2)).any()
        or ~((pair == -100) | ((pair >= 0) & (pair <= 2))).all()
        or not torch.equal(pair, pair.transpose(-1, -2))
        or (pair.diagonal(dim1=-2, dim2=-1) != -100).any()
    ):
        raise ValueError("invalid required pair target values/symmetry")
    supported = app == 0
    applicable_pairs = supported[:, :, None] & supported[:, None, :]
    applicable_pairs &= ~torch.eye(n, dtype=torch.bool, device=pair.device)[None]
    if not torch.equal(pair != -100, applicable_pairs):
        raise ValueError("required pair targets must cover exactly supported unordered pairs")

    mask = torch.zeros_like(pair, dtype=torch.bool)
    for ri, (row_app, row_pair) in enumerate(zip(app.tolist(), pair.tolist(), strict=True)):
        supported_indices = {i for i, label in enumerate(row_app) if label == 0}
        unseen, required = set(supported_indices), set()
        while unseen:
            component, pending = set(), [min(unseen)]
            while pending:
                i = pending.pop()
                if i in component:
                    continue
                component.add(i)
                pending.extend(j for j in supported_indices - component if row_pair[i][j] == 0)
            unseen -= component
            internal = {(i, j) for i in component for j in component if i < j}
            external = {
                tuple(sorted((i, j))) for i in component for j in supported_indices - component
            }
            if all(row_pair[i][j] == 0 for i, j in internal) and all(
                row_pair[i][j] == 1 for i, j in external
            ):
                required |= internal | external
        if required:
            left, right = zip(*sorted(required), strict=True)
            mask[ri, list(left), list(right)] = True
    return mask


def factorized_required_margin_loss(pair_logits, pair_targets, required_mask, *, temperature=1.0):
    """Mean row maximum of required-edge log-odds deficits at confidence .99.

    The edge hinge is relu(log(99) + logsumexp(other/T) - true/T).
    Rows without required edges contribute zero to the mean over all rows.
    This optional term does not alter population CE or its natural metrics.
    """
    n = len(CANDIDATES)
    if (
        not isinstance(pair_logits, torch.Tensor)
        or not isinstance(pair_targets, torch.Tensor)
        or not isinstance(required_mask, torch.Tensor)
        or pair_logits.ndim != 4
        or pair_logits.shape[1:] != (n, n, 3)
        or not pair_logits.is_floating_point()
        or pair_targets.shape != pair_logits.shape[:-1]
        or required_mask.shape != pair_targets.shape
        or pair_targets.dtype != torch.long
        or required_mask.dtype != torch.bool
        or pair_targets.device != pair_logits.device
        or required_mask.device != pair_logits.device
    ):
        raise ValueError("required margin needs matching pair logits, targets and boolean mask")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("required margin temperature must be a finite positive scalar")
    if (
        ~((pair_targets == -100) | ((pair_targets >= 0) & (pair_targets <= 2))).all()
        or required_mask.tril().any()
        or (required_mask & ((pair_targets < 0) | (pair_targets > 1))).any()
    ):
        raise ValueError("required margin mask must contain only upper-triangle SAME/DIFFERENT")
    if not torch.isfinite(pair_logits).all():
        raise ValueError("required margin logits must be finite")
    if not pair_logits.shape[0] or not required_mask.any():
        return pair_logits.sum() * 0
    scaled = pair_logits / temperature
    labels = pair_targets.clamp_min(0)
    true = scaled.gather(-1, labels[..., None]).squeeze(-1)
    others = scaled.masked_fill(F.one_hot(labels, 3).bool(), -torch.inf).logsumexp(-1)
    hinge = F.relu(math.log(99.0) + others - true)
    return hinge.masked_fill(~required_mask, 0).flatten(1).amax(1).mean()


def factorized_population_loss(logits, targets):
    """Optimize TRAIN focal rehearsal, retaining natural NLL sums/counts.

    Applicability and mode retain their original population means. Pair loss
    averages focal and other means equally when both are present. Without a
    focal annotation the objective and its gradients use the exact legacy path.
    Returned components always describe the unweighted natural populations;
    development temperature and checkpoint selection never use rehearsal loss.
    """
    sizes = {
        "applicability": (len(CANDIDATES), 3),
        "mode": (len(CANDIDATES), 4),
        "pair": (len(CANDIDATES), len(CANDIDATES), 3),
    }
    components, means = {}, []
    for kind, shape in sizes.items():
        values, labels = logits[kind], targets[kind]
        if (
            values.shape[1:] != shape
            or labels.shape != values.shape[:-1]
            or labels.dtype != torch.long
        ):
            raise ValueError("invalid factorized loss shape/target dtype")
        mask = labels != -100
        if kind == "pair":
            mask = mask & torch.ones_like(labels, dtype=torch.bool).triu(1)
        count = int(mask.sum())
        total = (
            F.cross_entropy(values[mask], labels[mask], reduction="sum")
            if count
            else values.sum() * 0
        )
        components[kind] = (total, count)
        if kind == "pair" and "pair_focal_mask" in targets:
            focal = targets["pair_focal_mask"]
            if (
                not isinstance(focal, torch.Tensor)
                or focal.dtype != torch.bool
                or focal.shape != labels.shape
                or focal.device != labels.device
            ):
                raise ValueError(
                    "pair focal mask must be boolean with the pair target shape/device"
                )
            if (
                not torch.equal(focal, focal.transpose(-1, -2))
                or focal.diagonal(dim1=-2, dim2=-1).any()
                or (focal.triu(1).sum((-1, -2)) > 1).any()
            ):
                raise ValueError("pair focal mask requires at most one unordered pair per row")
            applicable = targets["applicability"] == 0
            if (
                focal & ((labels == -100) | ~applicable[:, :, None] | ~applicable[:, None, :])
            ).any():
                raise ValueError("pair focal mask cannot include inapplicable targets")
            focal = focal & mask
            other = mask & ~focal
            if focal.any() and other.any():
                focal_mean = F.cross_entropy(values[focal], labels[focal])
                other_mean = F.cross_entropy(values[other], labels[other])
                means.append(0.5 * focal_mean + 0.5 * other_mean)
                continue
        if count:
            means.append(total / count)
    if not means or not components["applicability"][1]:
        raise ValueError("factorized training requires nonempty applicability labels")
    loss = sum(means)
    # Joint here means the sum of population NLLs, never a fabricated joint class.
    components["joint"] = (
        sum(value for value, _ in components.values()),
        sum(count for _, count in components.values()),
    )
    return loss, components


def split_factorized_development(records):
    dev = [row for row in records if row["split"] == "dev"]
    partitions = [
        [row for row in dev if row.get("development_role") == role]
        for role in ("temperature", "selection")
    ]
    groups = [
        {row.get("semantic_mapping_id", row["group_id"]) for row in rows} for rows in partitions
    ]
    if (
        any(len(group) != 300 for group in groups)
        or groups[0] & groups[1]
        or sum(map(len, partitions)) != len(dev)
    ):
        raise ValueError("factorized DEV requires explicit independent 300/300 semantic groups")
    return tuple(partitions)


def _forward(encoder, head, rows, *, deadline=None):
    from carryrl.rewards.control_text_grounding import build_grounding

    _check_deadline(deadline)
    device = next(head.parameters()).device
    tokens, mask = _batch(encoder, rows, device)
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("factorized head requires pinned offset tokenizer")
    grounding = torch.zeros(
        len(rows), len(CANDIDATES), tokens.shape[1], dtype=torch.bool, device=device
    )
    for index, row in enumerate(rows):
        length = int((~mask[index]).sum())
        grounding[index, :, :length] = build_grounding(
            row["text"], tokenizer, expected_tokens=length
        ).to(device)
    logits = head(tokens, mask, grounding=grounding)
    if any(not torch.isfinite(value).all() for value in logits.values()):
        raise FloatingPointError("nonfinite factorized logits")
    return logits


@torch.no_grad()
def collect_factorized_logits(encoder, head, rows, *, batch_size=32, deadline=None):
    if not rows:
        raise ValueError("nonempty factorized evaluation rows required")
    head.eval()
    result = {kind: [] for kind in ("applicability", "mode", "pair")}
    for offset in range(0, len(rows), batch_size):
        values = _forward(encoder, head, rows[offset : offset + batch_size], deadline=deadline)
        for kind, value in values.items():
            result[kind].append(value.cpu())
    return {kind: torch.cat(values) for kind, values in result.items()}


def fit_factorized_temperatures(logits, rows, *, deadline=None):
    if not rows or any(
        row.get("split") != "dev" or row.get("development_role") != "temperature" for row in rows
    ):
        raise ValueError("temperature fitting requires temperature DEV only")
    targets = factorized_targets(rows)
    calibration = {
        "architecture": FACTORIZED_HEAD_ARCHITECTURE,
        "decoder_version": FACTORIZED_DECODER_VERSION,
        "population_contract": FACTORIZED_POPULATION_CONTRACT,
        "thresholds_selected": "fixed_before_data",
        "temperature_groups": sorted(
            {row.get("semantic_mapping_id", row["group_id"]) for row in rows}
        ),
        "temperature_fit": "dev_temperature_population_nll_grid_0.1_to_10_81",
    }
    for kind, values in logits.items():
        labels = targets[kind]
        mask = labels != -100
        if kind == "pair":
            mask &= torch.ones_like(labels, dtype=torch.bool).triu(1)
        choices, losses = torch.logspace(-1, 1, 81), []
        for temperature in choices:
            _check_deadline(deadline)
            losses.append(
                float(F.cross_entropy(values[mask] / temperature, labels[mask]))
                if mask.any()
                else 0.0
            )
        best = min(range(len(losses)), key=losses.__getitem__)
        calibration.update(
            {
                f"{kind}_temperature": float(choices[best]),
                f"{kind}_threshold": 0.99,
                f"{kind}_temperature_nll": losses[best],
                f"{kind}_temperature_decisions": int(mask.sum()),
            }
        )
    return calibration


class FactorizedLogitPredictor:
    def __init__(self, rows, logits, calibration, *, model_hash="0" * 64):
        self.calibration, self.model_hash = calibration, model_hash
        self.calibration_hash = content_hash(calibration)
        self.values = {
            row["text"]: {kind: value[i] for kind, value in logits.items()}
            for i, row in enumerate(rows)
        }

    def predict(self, text):
        return decode_factorized_predictions(
            self.values[text],
            self.calibration,
            hashlib.sha256(text.encode()).hexdigest(),
            self.model_hash,
            self.calibration_hash,
        )


def validate_factorized_predictions(predictor, rows):
    """Gate actual payable compositions and track independent semantic-group errors."""

    def counts():
        return dict(
            clear_count=0,
            clear_accepted=0,
            accepted_count=0,
            accepted_correct=0,
            accepted_scenarios=set(),
            error_scenarios=set(),
        )

    aggregate, signs = counts(), {name: counts() for name in ("supported", "unsupported")}
    modes = {
        name: dict(clear_count=0, accepted_count=0, correct_count=0) for name in ("held", "press")
    }
    aliases = dict(
        clear_count=0, accepted_count=0, correct_count=0, false_split_errors=0, false_merge_errors=0
    )
    critical, assertions, abstentions = 0, 0, {}
    for row in rows:
        cluster = row.get("independent_group_id", row.get("semantic_mapping_id", row["group_id"]))
        table = predictor.predict(row["text"])
        by_candidate = {p.candidate: p for p in table.predictions}
        labels = {label["candidate"]: label for label in row["factorized_labels"]}
        pair_labels = {(p["left"], p["right"]): p["relation"] for p in row["pair_labels"]}
        errors = False
        for prediction in table.predictions:
            label = labels[prediction.candidate]
            expected = label["applicability"]
            clear, accepted = expected != "unspecified", prediction.reward_eligible
            correct = prediction.status == expected
            d = prediction.applicability_decision
            wrong_assertion = d.accepted and d.value != expected
            m = prediction.mode_decision
            wrong_mode = m.accepted and (expected != "supported" or m.value != label["mode"])
            # Timing is conditionally meaningful only for supported applicability.
            wrong_mode &= prediction.status == "supported"
            critical += int(wrong_assertion) + int(wrong_mode)
            errors |= wrong_assertion or wrong_mode
            assertions += int(d.accepted) + int(m.accepted and prediction.status == "supported")
            for decision in (d, m):
                if decision.reason:
                    abstentions[decision.reason] = abstentions.get(decision.reason, 0) + 1
            aggregate["clear_count"] += int(clear)
            aggregate["clear_accepted"] += int(clear and accepted)
            if clear:
                signs[expected]["clear_count"] += 1
                signs[expected]["clear_accepted"] += int(accepted and correct)
            if accepted:
                for count in (aggregate, signs[prediction.status]):
                    count["accepted_count"] += 1
                    count["accepted_correct"] += int(correct)
                    count["accepted_scenarios"].add(cluster)
                    if not correct:
                        count["error_scenarios"].add(cluster)
            if expected == "supported" and label["mode"] in modes:
                count = modes[label["mode"]]
                count["clear_count"] += 1
                mode_accepted = accepted and prediction.status == "supported" and m.accepted
                count["accepted_count"] += int(mode_accepted)
                count["correct_count"] += int(
                    mode_accepted and correct and m.value == label["mode"]
                )
        for pair in table.pair_decisions:
            expected = pair_labels.get(
                (pair.left, pair.right), pair_labels.get((pair.right, pair.left), "unspecified")
            )
            left, right = (by_candidate[c] for c in (pair.left, pair.right))
            relevant = (
                labels[pair.left]["applicability"]
                == labels[pair.right]["applicability"]
                == "supported"
            )
            if relevant and pair.decision.accepted and pair.decision.value != expected:
                critical += 1
                errors = True
            accepted = (
                left.reward_eligible
                and right.reward_eligible
                and left.status == right.status == "supported"
            )
            same = left.group is not None and left.group == right.group
            if expected == "same":
                aliases["clear_count"] += 1
                aliases["accepted_count"] += int(accepted)
                aliases["correct_count"] += int(accepted and same)
            split, merge = expected == "same" and accepted and not same, same and expected != "same"
            if split or merge:
                critical += 1
                errors = True
                aliases["false_split_errors"] += int(split)
                aliases["false_merge_errors"] += int(merge)
        if errors:
            aggregate["error_scenarios"].add(cluster)
            for count in signs.values():
                if cluster in count["accepted_scenarios"]:
                    count["error_scenarios"].add(cluster)
    for count in (aggregate, *signs.values()):
        # Precision evidence concerns actual payments. Wrong nonpaying assertions
        # remain critical errors without inventing a precision denominator.
        count["error_scenarios"] &= count["accepted_scenarios"]
        for name in ("accepted_scenarios", "error_scenarios"):
            count[name] = len(count[name])
    report = {
        **aggregate,
        "critical_errors": critical,
        "signs": signs,
        "modes": modes,
        "aliases": aliases,
        "records": len(rows),
        "scenario_count": len({r["group_id"] for r in rows}),
        "independent_semantic_clusters": len(
            {
                r.get("independent_group_id", r.get("semantic_mapping_id", r["group_id"]))
                for r in rows
            }
        ),
        "population_contract": FACTORIZED_POPULATION_CONTRACT,
        "payable_compositions": True,
        "field_assertions": assertions,
        "abstention_reasons": abstentions,
    }
    report["gates"] = calibration_gates(report)
    return report


def factorized_diagnostics(logits, rows, calibration):
    targets, raw = factorized_targets(rows), {}
    for kind, values in logits.items():
        mask = targets[kind] != -100
        if kind == "pair":
            mask &= torch.ones_like(mask).triu(1)
        actual = values.argmax(-1)
        raw[kind] = {
            "correct": int(((actual == targets[kind]) & mask).sum()),
            "total": int(mask.sum()),
        }
    return {
        "raw_argmax": raw,
        "decoder": validate_factorized_predictions(
            FactorizedLogitPredictor(rows, logits, calibration), rows
        ),
    }


def _selection_key(feasibility, score_nll):
    # No public runtime, calibration or sealed-test information enters this key.
    return (bool(feasibility["passed"]), -score_nll)


def fit_factorized_head(
    features,
    records,
    *,
    seed=20261029,
    epochs=8,
    batch_size=32,
    learning_rate=3e-4,
    device="cpu",
    deadline=None,
    checkpoint_directory=None,
    progress=None,
    training_protocol=LOSS_PROTOCOL,
):
    """Fit a fresh head with TRAIN rehearsal and natural development populations."""
    protocol_identity(training_protocol)
    if training_protocol == STAGED_TRAINING_PROTOCOL:
        return _fit_staged_factorized_head(
            features,
            records,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            deadline=deadline,
            checkpoint_directory=checkpoint_directory,
            progress=progress,
        )
    if type(epochs) is not int or not 1 <= epochs <= 10:
        raise ValueError("factorized fitting requires 1..10 epochs")
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("batch_size must be in 1..128")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("positive finite learning rate required")
    train = [row for row in records if row["split"] == "train"]
    temperature_rows, score_rows = split_factorized_development(records)
    if not train or not temperature_rows or not score_rows:
        raise ValueError("TRAIN and two independent DEV group halves are required")
    _check_deadline(deadline)
    features.eval().requires_grad_(False)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    head = FactorizedControlHead().to(device)
    initial_hash = encoder_state_hash(head.state_dict())
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=FACTORIZED_ADAMW_WEIGHT_DECAY,
    )
    if optimizer.state:
        raise RuntimeError("fresh factorized fit must start with empty optimizer state")
    directory = Path(checkpoint_directory) if checkpoint_directory else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
    history, updates, examples = [], 0, 0
    selected, best_key, selected_epoch, selected_calibration = None, None, None, None
    generator = random.Random(seed)

    def checkpoint(path, epoch):
        torch.save(
            {
                "architecture": FACTORIZED_HEAD_ARCHITECTURE,
                "model_version": FACTORIZED_MODEL_VERSION,
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "optimizer_updates": updates,
                "train_examples": examples,
                "initial_head_sha256": initial_hash,
                "seed": seed,
                "loss_protocol": LOSS_PROTOCOL,
                "torch_rng_state": torch.get_rng_state(),
                "python_rng_state": generator.getstate(),
                "history": history,
            },
            path,
        )

    try:
        for epoch in range(1, epochs + 1):
            _check_deadline(deadline)
            started = time.monotonic()
            order = list(train)
            generator.shuffle(order)
            head.train()
            objective_sum, objective_batches = 0.0, 0
            epoch_components = {
                name: [0.0, 0] for name in ("applicability", "mode", "pair", "joint")
            }
            for offset in range(0, len(order), batch_size):
                _check_deadline(deadline)
                rows = order[offset : offset + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = _forward(features, head, rows, deadline=deadline)
                targets = factorized_targets(rows, device=next(head.parameters()).device)
                loss, components = factorized_population_loss(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite joint training loss")
                loss.backward()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in head.parameters()
                ):
                    raise FloatingPointError("nonfinite joint training gradient")
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                examples += len(rows)
                objective_sum += float(loss.detach())
                objective_batches += 1
                for name, (total, count) in components.items():
                    epoch_components[name][0] += float(total.detach())
                    epoch_components[name][1] += count
                if progress:
                    progress(
                        {
                            "stage": "fit",
                            "epoch": epoch,
                            "optimizer_updates": updates,
                            "train_examples": examples,
                            "loss": float(loss.detach()),
                            "loss_protocol": LOSS_PROTOCOL,
                            "component_metric_kind": "natural_population_nll_unweighted",
                            "objective_components": {
                                name: {"numerator": float(total.detach()), "denominator": count}
                                for name, (total, count) in components.items()
                            },
                        }
                    )
            temperature_logits = collect_factorized_logits(
                features, head, temperature_rows, batch_size=batch_size, deadline=deadline
            )
            calibration = fit_factorized_temperatures(
                temperature_logits, temperature_rows, deadline=deadline
            )
            score_logits = collect_factorized_logits(
                features, head, score_rows, batch_size=batch_size, deadline=deadline
            )
            _, score_components = factorized_population_loss(
                {k: v / calibration[f"{k}_temperature"] for k, v in score_logits.items()},
                factorized_targets(score_rows),
            )
            score_nll = float(
                sum(
                    total / count
                    for name, (total, count) in score_components.items()
                    if name != "joint" and count
                )
            )
            diagnostics = factorized_diagnostics(score_logits, score_rows, calibration)
            feasibility = developmental_feasibility_gate(diagnostics["decoder"], score_rows)
            key = _selection_key(feasibility, score_nll)
            if best_key is None or key > best_key:
                best_key, selected_epoch = key, epoch
                selected = copy.deepcopy(
                    {name: value.detach().cpu() for name, value in head.state_dict().items()}
                )
                selected_calibration = copy.deepcopy(calibration)
            entry = {
                "epoch": epoch,
                "loss_protocol": LOSS_PROTOCOL,
                "train_objective": objective_sum / objective_batches,
                "train_objective_batches": objective_batches,
                "train_objective_sum": objective_sum,
                "component_metric_kind": "natural_population_nll_unweighted",
                "train_objective_components": {
                    name: {
                        "numerator": total,
                        "denominator": count,
                        "mean": total / count if count else None,
                    }
                    for name, (total, count) in epoch_components.items()
                    if name != "joint"
                },
                "train_joint_nll": epoch_components["joint"][0] / epoch_components["joint"][1],
                "train_joint_nll_sum": epoch_components["joint"][0],
                "train_joint_decisions": epoch_components["joint"][1],
                "score_joint_nll": score_nll,
                "calibration": calibration,
                "diagnostics": diagnostics,
                "feasibility": feasibility,
                "optimizer_updates": updates,
                "train_examples": examples,
                "seconds": time.monotonic() - started,
                "selected_epoch": selected_epoch,
            }
            history.append(entry)
            if directory:
                checkpoint(directory / f"epoch_{epoch:02d}.pt", epoch)
                _write_json(directory / "history.json", history)
            if progress:
                progress({"stage": "epoch_complete", **entry})
        head.load_state_dict(selected, strict=True)
        head.eval()
        training = {
            "architecture": FACTORIZED_HEAD_ARCHITECTURE,
            "seed": seed,
            "trained": True,
            "selected_development_feasible": bool(best_key[0]),
            "epochs": epochs,
            "completed_epochs": len(history),
            "history": history,
            "selected_epoch": selected_epoch,
            "selected_development_calibration": selected_calibration,
            "selection_protocol": "feasible_first_then_dev_score_joint_nll.v1",
            "selected_feasible": bool(best_key[0]),
            "diagnostic_only": not bool(best_key[0]),
            "initial_head_sha256": initial_hash,
            "optimizer_initial_state_entries": 0,
            "optimizer": "fresh_AdamW",
            "loss_protocol": LOSS_PROTOCOL,
            "optimizer_weight_decay": FACTORIZED_ADAMW_WEIGHT_DECAY,
            "learning_rate": learning_rate,
            "class_weights": None,
            "prior_correction": None,
            "optimizer_updates": updates,
            "train_examples": examples,
            "train_records": len(train),
            "batch_size": batch_size,
            "development_group_split": development_split_identity(temperature_rows, score_rows),
            "selection_splits": ["dev"],
            "gradient_splits": ["train"],
            "public_used_for_selection": False,
            "sealed_test_used": False,
        }
        if directory:
            torch.save(
                {
                    "head": selected,
                    "training": training,
                    "architecture": FACTORIZED_HEAD_ARCHITECTURE,
                },
                directory / "selected_head.pt",
            )
        return head, training
    except BaseException:
        if directory:
            checkpoint(directory / "partial_head.pt", len(history))
        raise


def _frozen_pair_refinement_state(head, optimizer):
    frozen = {
        name: parameter
        for name, parameter in head.named_parameters()
        if not name.startswith("pair.")
    }
    moments = {
        f"{name}.{key}": value
        for name, parameter in frozen.items()
        for key, value in optimizer.state[parameter].items()
    }
    if any(not isinstance(value, torch.Tensor) for value in moments.values()):
        raise RuntimeError("unsupported non-tensor AdamW state in frozen refinement")
    return {
        "weights_and_buffers_sha256": encoder_state_hash(
            {
                name: value
                for name, value in head.state_dict().items()
                if not name.startswith("pair.")
            }
        ),
        "optimizer_state_sha256": encoder_state_hash(
            {f"{name}:{tuple(value.shape)}": value.reshape(-1) for name, value in moments.items()}
        ),
    }


def _fit_staged_factorized_head(
    features,
    records,
    *,
    seed,
    epochs,
    batch_size,
    learning_rate,
    device,
    deadline,
    checkpoint_directory,
    progress,
):
    if epochs != STAGED_BASE_EPOCHS or batch_size != 32 or learning_rate != 0.0003:
        raise ValueError("staged protocol requires exactly eight base epochs, batch32 and LR.0003")
    if str(torch.device(device)) != "cpu" or seed not in (20261029, 20261030):
        raise ValueError("staged protocol requires CPU and the precommitted probe/full seed")
    if checkpoint_directory is None:
        raise ValueError("staged protocol requires checkpoints for exact terminal-base restoration")
    train = [row for row in records if row["split"] == "train"]
    if len(train) != STAGED_TRAIN_RECORDS:
        raise ValueError("staged protocol requires the fixed 15264-record TRAIN population")
    directory = Path(checkpoint_directory)
    head, training = fit_factorized_head(
        features,
        records,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        deadline=deadline,
        checkpoint_directory=directory,
        progress=progress,
    )
    base_epoch = training["selected_epoch"]
    base_calibration = copy.deepcopy(training["selected_development_calibration"])
    best_key = _selection_key(
        training["history"][base_epoch - 1]["feasibility"],
        training["history"][base_epoch - 1]["score_joint_nll"],
    )
    selected = copy.deepcopy(head.state_dict())
    base_updates = STAGED_BASE_EPOCHS * math.ceil(len(train) / batch_size)
    candidate = {
        "stage": "base",
        "base_epoch": base_epoch,
        "optimizer_updates": training["history"][base_epoch - 1]["optimizer_updates"],
        "checkpoint_file": f"epoch_{base_epoch:02d}.pt",
        "history_index": base_epoch - 1,
    }
    terminal = torch.load(directory / f"epoch_{epochs:02d}.pt", weights_only=True)
    if terminal["epoch"] != epochs or terminal["optimizer_updates"] != base_updates:
        raise RuntimeError("staged base did not reach the exact terminal epoch boundary")
    head.load_state_dict(terminal["head"], strict=True)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=FACTORIZED_ADAMW_WEIGHT_DECAY
    )
    optimizer.load_state_dict(terminal["optimizer"])
    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(terminal["torch_rng_state"])
    generator = random.Random()
    generator.setstate(terminal["python_rng_state"])
    if not torch.equal(torch.get_rng_state(), terminal["torch_rng_state"]):
        raise RuntimeError("terminal base Torch RNG restoration failed")
    temperature = float(base_calibration["pair_temperature"])
    temperature_source = {
        "split": "dev",
        "roles": ["temperature", "selection"],
        "used_for_training": True,
        "base_selected_epoch": base_epoch,
        "base_selected_updates": candidate["optimizer_updates"],
        "base_calibration_sha256": content_hash(base_calibration),
        "pair_temperature": temperature,
    }
    updates, examples = terminal["optimizer_updates"], terminal["train_examples"]
    sampler_epoch, order, offset = epochs + 1, None, 0
    stage, stage_updates, frozen_before = "continuation", 0, None

    def checkpoint(name, complete=False):
        cursor = {"epoch": sampler_epoch, "epoch_order": order, "next_row_offset": offset}
        if order is not None and offset == len(order):
            cursor = {"epoch": sampler_epoch + 1, "epoch_order": None, "next_row_offset": 0}
        frozen_after = _frozen_pair_refinement_state(head, optimizer) if frozen_before else None
        if complete and frozen_before and frozen_before != frozen_after:
            raise RuntimeError("staged refinement changed frozen trunk/unary/AdamW state")
        payload = {
            "architecture": FACTORIZED_HEAD_ARCHITECTURE,
            "model_version": FACTORIZED_MODEL_VERSION,
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stage": stage,
            "stage_updates": stage_updates,
            "stage_complete": complete,
            "base_epochs": epochs,
            "optimizer_updates": updates,
            "train_examples": examples,
            "initial_head_sha256": training["initial_head_sha256"],
            "seed": seed,
            "loss_protocol": STAGED_TRAINING_PROTOCOL,
            "training_protocol": STAGED_TRAINING_PROTOCOL,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": generator.getstate(),
            "resume_cursor": cursor,
            "training_temperature_source": temperature_source,
            "frozen_state_before": frozen_before,
            "frozen_state_after": frozen_after,
            "frozen_verified": bool(frozen_before and frozen_before == frozen_after),
            "history": training["history"],
            "ready": False,
        }
        path = directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    try:
        for stage, limit in (
            ("continuation", STAGED_CONTINUATION_UPDATES),
            ("required_margin", STAGED_MARGIN_UPDATES),
        ):
            stage_updates = 0
            if stage == "required_margin":
                optimizer.zero_grad(set_to_none=True)
                for name, parameter in head.named_parameters():
                    parameter.requires_grad_(name.startswith("pair."))
                frozen_before = _frozen_pair_refinement_state(head, optimizer)
            head.train()
            for _ in range(limit):
                _check_deadline(deadline)
                if order is None or offset == len(order):
                    if order is not None:
                        sampler_epoch += 1
                    order = list(range(len(train)))
                    generator.shuffle(order)
                    offset = 0
                rows = [train[index] for index in order[offset : offset + batch_size]]
                optimizer.zero_grad(set_to_none=True)
                values = _forward(features, head, rows, deadline=deadline)
                targets = factorized_targets(rows, device=device)
                loss, components = factorized_population_loss(values, targets)
                margin = values["pair"].sum() * 0
                if stage == "required_margin":
                    margin = factorized_required_margin_loss(
                        values["pair"],
                        targets["pair"],
                        factorized_required_pair_mask(targets),
                        temperature=temperature,
                    )
                    loss = loss + margin
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite staged training objective")
                loss.backward()
                if stage == "required_margin" and any(
                    parameter.grad is not None
                    for name, parameter in head.named_parameters()
                    if not name.startswith("pair.")
                ):
                    raise RuntimeError("frozen staged parameter received a gradient")
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                stage_updates += 1
                examples += len(rows)
                offset += len(rows)
                if progress:
                    progress(
                        {
                            "stage": stage,
                            "stage_updates": stage_updates,
                            "optimizer_updates": updates,
                            "train_examples": examples,
                            "loss": float(loss.detach()),
                            "required_margin": float(margin.detach()),
                            "loss_protocol": STAGED_TRAINING_PROTOCOL,
                            "component_metric_kind": "natural_population_nll_unweighted",
                            "objective_components": {
                                kind: {"numerator": float(total.detach()), "denominator": count}
                                for kind, (total, count) in components.items()
                            },
                        }
                    )
            checkpoint(f"{stage}_{updates:06d}.pt", complete=True)
        frozen_after = _frozen_pair_refinement_state(head, optimizer)
        if frozen_before != frozen_after:
            raise RuntimeError("staged refinement changed frozen trunk/unary/AdamW state")
        temperature_rows, score_rows = split_factorized_development(records)
        temperature_logits = collect_factorized_logits(
            features, head, temperature_rows, batch_size=batch_size, deadline=deadline
        )
        calibration = fit_factorized_temperatures(
            temperature_logits, temperature_rows, deadline=deadline
        )
        score_logits = collect_factorized_logits(
            features, head, score_rows, batch_size=batch_size, deadline=deadline
        )
        _, components = factorized_population_loss(
            {
                kind: values / calibration[f"{kind}_temperature"]
                for kind, values in score_logits.items()
            },
            factorized_targets(score_rows),
        )
        score_nll = float(
            sum(
                total / count
                for kind, (total, count) in components.items()
                if kind != "joint" and count
            )
        )
        diagnostics = factorized_diagnostics(score_logits, score_rows, calibration)
        feasibility = developmental_feasibility_gate(diagnostics["decoder"], score_rows)
        key = _selection_key(feasibility, score_nll)
        for entry in training["history"]:
            entry["stage"] = "base"
        refined = {
            "stage": "required_margin",
            "optimizer_updates": updates,
            "train_examples": examples,
            "score_joint_nll": score_nll,
            "calibration": calibration,
            "diagnostics": diagnostics,
            "feasibility": feasibility,
        }
        training["history"].append(refined)
        if key > best_key:
            best_key = key
            selected = copy.deepcopy(head.state_dict())
            base_calibration = copy.deepcopy(calibration)
            candidate = {
                "stage": "required_margin",
                "base_epoch": None,
                "optimizer_updates": updates,
                "checkpoint_file": f"required_margin_{updates:06d}.pt",
                "history_index": len(training["history"]) - 1,
            }
        training.update(
            training_protocol=STAGED_TRAINING_PROTOCOL,
            loss_protocol=STAGED_TRAINING_PROTOCOL,
            selected_candidate=candidate,
            selected_epoch=candidate["base_epoch"],
            selected_development_calibration=base_calibration,
            selected_development_feasible=bool(best_key[0]),
            selected_feasible=bool(best_key[0]),
            diagnostic_only=not bool(best_key[0]),
            optimizer_updates=updates,
            train_examples=examples,
            development_training_hyperparameter_provenance=temperature_source,
            staged_completion={
                "complete": True,
                "base_updates": base_updates,
                "continuation_updates": STAGED_CONTINUATION_UPDATES,
                "required_margin_updates": STAGED_MARGIN_UPDATES,
                "terminal_updates": updates,
                "frozen_before": frozen_before,
                "frozen_after": frozen_after,
                "frozen_verified": True,
                "pair_temperature": temperature,
                "temperature_source": temperature_source,
            },
        )
        head.load_state_dict(selected, strict=True)
        head.eval()
        _write_json(directory / "history.json", training["history"])
        torch.save(
            {"head": selected, "training": training, "architecture": FACTORIZED_HEAD_ARCHITECTURE},
            directory / "selected_head.pt",
        )
        return head, training
    except BaseException:
        checkpoint("partial_staged_head.pt")
        raise


def _staged_training_complete(training):
    """Bind successful staged reports to the complete precommitted recipe."""
    completion = training.get("staged_completion", {})
    base_updates = STAGED_BASE_EPOCHS * math.ceil(STAGED_TRAIN_RECORDS / 32)
    final_updates = base_updates + STAGED_CONTINUATION_UPDATES + STAGED_MARGIN_UPDATES
    candidate = training.get("selected_candidate", {})
    history = training.get("history", [])
    index = candidate.get("history_index")
    if not isinstance(history, list) or type(index) is not int or not 0 <= index < len(history):
        return False
    selected = history[index]
    source = completion.get("temperature_source", {})
    source_epoch = source.get("base_selected_epoch")
    if type(source_epoch) is not int or not 1 <= source_epoch <= STAGED_BASE_EPOCHS:
        return False
    if len(history) != STAGED_BASE_EPOCHS + 1:
        return False
    source_entry = history[source_epoch - 1]
    source_calibration = source_entry.get("calibration", {})
    if (
        source_entry.get("stage") != "base"
        or source.get("split") != "dev"
        or source.get("roles") != ["temperature", "selection"]
        or source.get("base_calibration_sha256") != content_hash(source_calibration)
        or source.get("base_selected_updates") != source_entry.get("optimizer_updates")
        or source.get("pair_temperature") != source_calibration.get("pair_temperature")
        or completion.get("pair_temperature") != source.get("pair_temperature")
        or training.get("selected_development_calibration") != selected.get("calibration")
    ):
        return False
    if candidate.get("stage") == "base":
        base_epoch = candidate.get("base_epoch")
        if type(base_epoch) is not int or not 1 <= base_epoch <= STAGED_BASE_EPOCHS:
            return False
        expected_file = f"epoch_{base_epoch:02d}.pt"
        expected_updates = base_epoch * math.ceil(STAGED_TRAIN_RECORDS / 32)
    else:
        base_epoch = None
        expected_file = f"required_margin_{final_updates:06d}.pt"
        expected_updates = final_updates
    return bool(
        training.get("training_protocol") == STAGED_TRAINING_PROTOCOL
        and training.get("train_records") == STAGED_TRAIN_RECORDS
        and training.get("completed_epochs") == STAGED_BASE_EPOCHS
        and training.get("optimizer_updates") == final_updates
        and completion.get("complete") is True
        and completion.get("base_updates") == base_updates
        and completion.get("continuation_updates") == STAGED_CONTINUATION_UPDATES
        and completion.get("required_margin_updates") == STAGED_MARGIN_UPDATES
        and completion.get("terminal_updates") == final_updates
        and completion.get("frozen_verified") is True
        and completion.get("frozen_before") == completion.get("frozen_after")
        and completion.get("frozen_before")
        and completion.get("temperature_source", {}).get("used_for_training") is True
        and completion.get("temperature_source")
        == training.get("development_training_hyperparameter_provenance")
        and candidate.get("stage") in ("base", "required_margin")
        and candidate.get("checkpoint_file") == expected_file
        and candidate.get("optimizer_updates") == expected_updates
        and candidate.get("base_epoch") == base_epoch == training.get("selected_epoch")
        and selected.get("stage") == candidate.get("stage")
        and selected.get("optimizer_updates") == candidate.get("optimizer_updates")
    )


def verify_passed_probe(
    path,
    *,
    dataset_sha256,
    seed,
    encoder_state_sha256,
    learning_rate,
    batch_size,
    runtime_instructions,
    runtime_controls,
    training_protocol=LOSS_PROTOCOL,
):
    """Full training requires a complete, authenticated successful fresh-head probe."""
    from carryrl.rewards.control_text_factorized_artifact import _runtime_gate
    from carryrl.rewards.control_text_factorized_evaluation import (
        factorized_runtime_control_rows as runtime_control_rows,
    )

    path = Path(path)
    digest = _file_hash(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists() or sidecar.read_text().strip() != digest:
        raise ValueError("passed probe report hash is missing or mismatched")
    report = json.loads(path.read_text())
    config = report.get("config", {})
    protocol = protocol_identity(training_protocol)
    _, runtime_identity = runtime_control_rows(runtime_instructions, runtime_controls)
    checkpoint = path.parent / "selected_head.pt"
    if (
        report.get("status") != "complete"
        or config.get("probe_only") is not True
        or config.get("architecture") != FACTORIZED_HEAD_ARCHITECTURE
        or config.get("dataset_sha256") != dataset_sha256
        or config.get("source_identity") != source_identity()
        or config.get("encoder_state_sha256") != encoder_state_sha256
        or config.get("seed") != 20261029
        or seed != 20261030
        or config.get("protocol") != protocol
        or config.get("learning_rate") != learning_rate
        or config.get("batch_size") != batch_size
        or config.get("optimizer_weight_decay") != FACTORIZED_ADAMW_WEIGHT_DECAY
        or config.get("loss_protocol") != training_protocol
        or (
            training_protocol == STAGED_TRAINING_PROTOCOL
            and (
                config.get("training_protocol") != training_protocol
                or not _staged_training_complete(report.get("training", {}))
            )
        )
        or report.get("feasibility", {}).get("passed") is not True
        or report.get("runtime", {}).get("passed") is not True
        or report.get("runtime", {}).get("records") != 32
        or report.get("runtime", {}).get("passed_records") != 32
        or len(report.get("runtime", {}).get("results", [])) != 32
        or any(
            value.get("passed") is not True or value.get("errors")
            for value in report.get("runtime", {}).get("results", [])
        )
        or not _runtime_gate(report.get("runtime", {}))
        or report.get("calibration_used") is not False
        or report.get("sealed_test_used") is not False
        or report.get("encoder_frozen_verified") is not True
        or config.get("encoded_splits") != ["train", "dev"]
        or report.get("training", {}).get("gradient_splits") != ["train"]
        or report.get("training", {}).get("selection_splits") != ["dev"]
        or report.get("training", {}).get("public_used_for_selection") is not False
        or not checkpoint.is_file()
        or _file_hash(checkpoint) != report.get("selected_checkpoint_sha256")
        or any(report["runtime"].get(key) != value for key, value in runtime_identity.items())
    ):
        raise ValueError(
            "full training requires a passed factorized probe, unchanged runtime32 and fresh seed"
        )
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "seed": config["seed"],
    }


def train_factorized_text_control_artifact(
    dataset,
    output,
    *,
    encoder,
    probe_only=True,
    passed_probe=None,
    epochs=8,
    seed=None,
    feature_cache=None,
    reuse_feature_cache=None,
    max_feature_bytes=8 * 1024**3,
    max_seconds=4200,
    batch_size=32,
    learning_rate=3e-4,
    device="cpu",
    progress=None,
    runtime_instructions="configs/atomic_instructions.json",
    runtime_controls="configs/atomic_controls.json",
    evaluate_sealed=True,
    training_protocol=LOSS_PROTOCOL,
):
    """Run bounded training, preserve every checkpoint, keep export readiness false.

    A full run may inspect sealed data only once, after selected head/temperature
    freeze and successful unchanged calibration and exact public runtime gates.
    """
    from carryrl.rewards.control_text_factorized_evaluation import (
        factorized_runtime_control_rows as runtime_control_rows,
    )
    from carryrl.rewards.control_text_factorized_evaluation import (
        validate_factorized_runtime_text_controls as validate_runtime_text_controls,
    )

    protocol = protocol_identity(training_protocol)
    seed = (20261029 if probe_only else 20261030) if seed is None else seed
    if seed != (20261029 if probe_only else 20261030):
        raise ValueError("factorized fit requires the precommitted probe/full seed")
    if batch_size != 32 or learning_rate != 0.0003:
        raise ValueError("factorized protocol fixes batch=32 and learning_rate=.0003")
    if training_protocol == STAGED_TRAINING_PROTOCOL and (
        epochs != STAGED_BASE_EPOCHS or str(torch.device(device)) != "cpu"
    ):
        raise ValueError("staged protocol requires exactly eight base epochs and CPU")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= (4200 if probe_only else 7200):
        raise ValueError("fit budget exceeds the precommitted probe/full limit")
    if probe_only and not 1 <= epochs <= 8:
        raise ValueError("development probes require 1..8 epochs")
    if not probe_only and not 1 <= epochs <= 10:
        raise ValueError("full training is limited to ten epochs")
    if feature_cache and reuse_feature_cache:
        raise ValueError("choose new features or authenticated reuse")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"factorized output already exists: {output}")
    start, deadline = time.monotonic(), time.monotonic() + max_seconds
    from carryrl.rewards.control_text_data_v14 import read_factorized_dataset

    manifest, records = read_factorized_dataset(dataset)
    authorization = None
    if not probe_only and not passed_probe:
        raise ValueError("full training requires --passed-probe")
    encoder.eval().requires_grad_(False)
    original_encoder_hash = encoder_state_hash(encoder.state_dict())
    if not probe_only:
        authorization = verify_passed_probe(
            passed_probe,
            dataset_sha256=manifest["records_sha256"],
            seed=seed,
            encoder_state_sha256=original_encoder_hash,
            learning_rate=learning_rate,
            batch_size=batch_size,
            runtime_instructions=runtime_instructions,
            runtime_controls=runtime_controls,
            training_protocol=training_protocol,
        )
    # Probe extraction and authenticated reuse are restricted to TRAIN/DEV.
    # A full fit requires its own cache including calibration, never sealed inputs.
    cache = Path(
        reuse_feature_cache or feature_cache or output.with_name(output.name + ".features")
    )
    if output.resolve().is_relative_to(cache.resolve()) or cache.resolve().is_relative_to(
        output.resolve()
    ):
        raise ValueError("factorized output and E5 feature cache must be separate")
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "architecture": FACTORIZED_HEAD_ARCHITECTURE,
        "model_version": FACTORIZED_MODEL_VERSION,
        "probe_only": probe_only,
        "epochs": epochs,
        "seed": seed,
        "ready": False,
        "dataset_sha256": manifest["records_sha256"],
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer_weight_decay": FACTORIZED_ADAMW_WEIGHT_DECAY,
        "loss_protocol": training_protocol,
        "max_seconds": max_seconds,
        "feature_cache": str(cache.resolve()),
        "encoded_splits": ["train", "dev"] if probe_only else ["train", "dev", "calibration"],
        "max_feature_bytes": max_feature_bytes,
        "encoder_state_sha256": original_encoder_hash,
        "passed_probe": authorization,
        "source_identity": source_identity(),
        "protocol": protocol,
        "checkpoint_selection": "dev_feasible_first_joint_nll_only",
        "sealed_test_used": False,
    }
    if training_protocol == STAGED_TRAINING_PROTOCOL:
        config["training_protocol"] = training_protocol
    _, initial_runtime_identity = runtime_control_rows(runtime_instructions, runtime_controls)
    config["runtime_identity"] = initial_runtime_identity
    _write_json(output / "run_config.json", config)
    status = {"status": "running", "stage": "input_preflight", "ready": False}
    _write_json(output / "status.json", status)

    def notify(value):
        status.update(
            {
                key: value[key]
                for key in ("stage", "epoch", "optimizer_updates", "train_examples", "loss")
                if key in value
            }
        )
        _write_json(output / "status.json", status)
        if progress:
            progress(value)

    try:
        fit_rows = [row for row in records if row["split"] in ("train", "dev")]
        from carryrl.rewards.control_text_factorized_evaluation import verify_factorized_grounding

        preflight = verify_factorized_grounding(encoder.tokenizer, fit_rows, deadline=deadline)
        _write_json(output / "input_transmission.json", preflight)
        notify({"stage": "features"})
        if reuse_feature_cache:
            features = load_verified_disk_token_features(
                cache,
                encoder=encoder,
                records=fit_rows if probe_only else records,
                dataset_manifest=manifest,
                deadline=deadline,
                max_bytes=max_feature_bytes,
            )
        else:
            features = prepare_disk_token_features(
                encoder,
                fit_rows if probe_only else records,
                cache,
                dataset_sha256=manifest["records_sha256"],
                deadline=deadline,
                max_bytes=max_feature_bytes,
            )
        features.tokenizer = encoder.tokenizer
        head, training = fit_factorized_head(
            features,
            fit_rows,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            deadline=deadline,
            checkpoint_directory=output / "checkpoints",
            progress=notify,
            training_protocol=training_protocol,
        )
        calibration = copy.deepcopy(training["selected_development_calibration"])
        if training_protocol == STAGED_TRAINING_PROTOCOL:
            if not _staged_training_complete(training):
                raise RuntimeError("incomplete staged fit cannot produce a selected artifact")
            candidate = training["selected_candidate"]
            selected_report = training["history"][candidate["history_index"]]
            checkpoint_file = candidate["checkpoint_file"]
        else:
            selected_epoch = training["selected_epoch"]
            selected_report = training["history"][selected_epoch - 1]
            checkpoint_file = f"epoch_{selected_epoch:02d}.pt"
        feasibility = selected_report["feasibility"]
        model_hash = encoder_state_hash(head.state_dict())
        selected_checkpoint = torch.load(
            output / "checkpoints" / checkpoint_file,
            map_location="cpu",
            weights_only=True,
        )
        if training_protocol == STAGED_TRAINING_PROTOCOL:
            if encoder_state_hash(selected_checkpoint["head"]) != model_hash:
                raise RuntimeError("selected staged checkpoint does not match returned candidate")
            selected_checkpoint.update(
                stage=candidate["stage"],
                selected_candidate=candidate,
                training_protocol=training_protocol,
                loss_protocol=training_protocol,
            )
        selected_checkpoint.update(training=training, ready=False)
        torch.save(selected_checkpoint, output / "selected_head.pt")
        _write_json(output / "selected_calibration.json", calibration)
        _write_json(output / "training.json", training)
        notify({"stage": "public_runtime"})
        runtime_rows, _ = runtime_control_rows(runtime_instructions, runtime_controls)
        runtime_logits = collect_factorized_logits(
            encoder, head, runtime_rows, batch_size=batch_size, deadline=deadline
        )
        torch.save(
            {
                "logits": runtime_logits,
                "texts": [row["text"] for row in runtime_rows],
                "text_sha256": [
                    hashlib.sha256(row["text"].encode()).hexdigest() for row in runtime_rows
                ],
                "rows": [{key: row[key] for key in ("task", "language")} for row in runtime_rows],
                "head_state_sha256": model_hash,
                "calibration_hash": content_hash(calibration),
                "used_for_selection": False,
            },
            output / "runtime_logits.pt",
        )
        runtime_predictor = FactorizedLogitPredictor(
            runtime_rows, runtime_logits, calibration, model_hash=model_hash
        )
        runtime = validate_runtime_text_controls(
            runtime_predictor,
            instructions=runtime_instructions,
            controls=runtime_controls,
            deadline=deadline,
        )
        _write_json(output / "runtime.json", runtime)
        _write_json(
            output / "runtime_diagnostics.json",
            factorized_diagnostics(runtime_logits, runtime_rows, calibration),
        )
        validation, sealed = None, None
        if not probe_only:
            notify({"stage": "calibration_gate"})
            gate_rows = [row for row in records if row["split"] == "calibration"]
            logits = collect_factorized_logits(
                features, head, gate_rows, batch_size=batch_size, deadline=deadline
            )
            validation = validate_factorized_predictions(
                FactorizedLogitPredictor(gate_rows, logits, calibration, model_hash=model_hash),
                gate_rows,
            )
            _write_json(output / "calibration_validation.json", validation)
            _write_json(
                output / "calibration_diagnostics.json",
                factorized_diagnostics(logits, gate_rows, calibration),
            )
            if (
                evaluate_sealed
                and feasibility["passed"]
                and runtime["passed"]
                and validation["gates"]["passed"]
            ):
                notify({"stage": "sealed_test"})
                sentinel = output / "sealed_evaluation_started.json"
                # Exclusive create makes a second evaluation of this frozen run an error.
                with sentinel.open("x") as stream:
                    json.dump(
                        {
                            "head_sha256": model_hash,
                            "calibration_hash": content_hash(calibration),
                            "selection_frozen": True,
                        },
                        stream,
                    )
                sealed_rows = [row for row in records if row["split"] == "sealed_test"]
                logits = collect_factorized_logits(
                    encoder, head, sealed_rows, batch_size=batch_size, deadline=deadline
                )
                sealed = validate_factorized_predictions(
                    FactorizedLogitPredictor(
                        sealed_rows, logits, calibration, model_hash=model_hash
                    ),
                    sealed_rows,
                )
                _write_json(output / "sealed_validation.json", sealed)
        final_encoder_hash = encoder_state_hash(encoder.state_dict())
        if final_encoder_hash != original_encoder_hash:
            raise RuntimeError("frozen original E5 changed during joint-head fitting")
        _, runtime_identity = runtime_control_rows(runtime_instructions, runtime_controls)
        if source_identity() != config["source_identity"]:
            raise RuntimeError("factorized learner sources changed during fitting")
        if runtime_identity != initial_runtime_identity:
            raise RuntimeError(
                "factorized runtime instructions/specification changed during fitting"
            )
        artifact_pins = {
            "source_identity": source_identity(),
            "dataset_sha256": manifest["records_sha256"],
            **runtime_identity,
            "protocol_sha256": content_hash(protocol),
            "project_specification_sha256": _file_hash(Path(__file__).parents[3] / "SPEC.md"),
            "runtime_report_sha256": content_hash(runtime),
            "sealed_report_sha256": content_hash(sealed),
        }
        training.update(
            artifact_pins=artifact_pins,
            sealed_test_used=sealed is not None,
            encoder_frozen_verified=True,
            encoder_state_sha256=final_encoder_hash,
            dataset_sha256=manifest["records_sha256"],
        )
        artifact_manifest = None
        if not probe_only:
            from carryrl.rewards.control_text_factorized_artifact import (
                save_factorized_control_artifact,
            )

            artifact_manifest = save_factorized_control_artifact(
                output / "artifact",
                encoder=encoder,
                head=head,
                calibration=calibration,
                validation=validation,
                runtime_validation=runtime,
                sealed_validation=sealed,
                training=training,
            )
        _write_json(output / "training.json", training)
        result = {
            "version": "carryrl.control_text_factorized_training.v1",
            "status": "complete",
            "ready": bool(artifact_manifest and artifact_manifest["ready"]),
            "production_export_pending": artifact_manifest is None,
            "config": config,
            "artifact": str((output / "artifact").resolve()) if artifact_manifest else None,
            "training": training,
            "feasibility": feasibility,
            "runtime": runtime,
            "calibration_validation": validation,
            "sealed_validation": sealed,
            "calibration_used": not probe_only,
            "sealed_test_used": sealed is not None,
            "encoder_frozen_verified": True,
            "encoder_state_sha256": final_encoder_hash,
            "selected_head_state_sha256": model_hash,
            "selected_checkpoint_sha256": _file_hash(output / "selected_head.pt"),
            "feature_cache_bytes": features.metadata["actual_bytes"],
            "elapsed_seconds": time.monotonic() - start,
            "calibration_hash": content_hash(calibration),
            "admission_gates_passed": bool(
                not probe_only
                and feasibility["passed"]
                and runtime["passed"]
                and validation["gates"]["passed"]
                and sealed is not None
                and sealed["gates"]["passed"]
            ),
        }
        report_path = output / ("probe_report.json" if probe_only else "full_report.json")
        _write_json(report_path, result)
        report_path.with_suffix(report_path.suffix + ".sha256").write_text(
            _file_hash(report_path) + "\n"
        )
        status.update(
            status="complete",
            stage="complete",
            ready=result["ready"],
            report=str(report_path.resolve()),
        )
        _write_json(output / "status.json", status)
        return result
    except BaseException as error:
        status.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            ready=False,
            encoder_frozen_verified=encoder_state_hash(encoder.state_dict())
            == original_encoder_hash,
        )
        _write_json(output / "status.json", status)
        raise
