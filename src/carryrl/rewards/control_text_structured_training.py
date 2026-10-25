"""Fresh joint-state fitting with development-only selection and unchanged gates."""

import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from carryrl.rewards.control_text import CANDIDATES, content_hash, encoder_state_hash
from carryrl.rewards.control_text_structured import (
    JOINT_STATE_COUNT,
    STRUCTURED_DECODER_VERSION,
    STRUCTURED_HEAD_ARCHITECTURE,
    STRUCTURED_MODEL_VERSION,
    STRUCTURED_POPULATION_CONTRACT,
    JointStructuredControlHead,
    decode_structured_predictions,
    structured_targets,
)
from carryrl.rewards.control_text_training import (
    _batch,
    development_split_identity,
    developmental_feasibility_gate,
    load_verified_disk_token_features,
    prepare_disk_token_features,
    read_control_dataset,
    split_development_groups,
    validate_text_control_predictions,
    verify_grounded_text_inputs,
)

STRUCTURED_ADAMW_WEIGHT_DECAY = 0.01
LOSS_PROTOCOL = "conditional_population_mean_v1"


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("structured text head wall-clock budget exhausted")


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity():
    """Bind admission to the exact new learner and unchanged evaluation sources."""
    root = Path(__file__).parent
    names = (
        "control_text_structured.py",
        "control_text_structured_training.py",
        "control_text_structured_artifact.py",
        "control_text_grounding.py",
        "control_text_training.py",
        "control_text_runtime.py",
        "control_text.py",
    )
    identity = {name: _file_hash(root / name) for name in names}
    identity["agent/language.py"] = _file_hash(root.parent / "agent/language.py")
    for name in ("train_text_control_structured_reward.py", "train_text_control_reward.py"):
        identity[f"scripts/{name}"] = _file_hash(root.parents[2] / "scripts" / name)
    return identity


def _forward(encoder, head, rows, *, deadline=None):
    from carryrl.rewards.control_text_grounding import build_grounding

    _check_deadline(deadline)
    device = next(head.parameters()).device
    tokens, mask = _batch(encoder, rows, device)
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("joint head requires the original pinned offset tokenizer")
    grounding = torch.zeros(
        len(rows), len(CANDIDATES), tokens.shape[1], dtype=torch.bool, device=device
    )
    for index, row in enumerate(rows):
        length = int((~mask[index]).sum())
        grounding[index, :, :length] = build_grounding(
            row["text"],
            tokenizer,
            expected_tokens=length,
        ).to(device)
    logits = head(tokens, mask, grounding=grounding)
    if logits.shape != (len(rows), len(CANDIDATES), JOINT_STATE_COUNT):
        raise ValueError("joint head returned an invalid state tensor")
    if (
        torch.isnan(logits).any()
        or torch.isposinf(logits).any()
        or not torch.isfinite(logits).any(-1).all()
    ):
        raise FloatingPointError("invalid joint-state logits")
    return logits


@torch.no_grad()
def collect_joint_logits(encoder, head, rows, *, batch_size=32, deadline=None):
    """Bounded E5 feature reads; labels never enter the forward computation."""
    if not rows:
        raise ValueError("nonempty evaluation rows required")
    head.eval()
    return torch.cat(
        [
            _forward(encoder, head, rows[offset : offset + batch_size], deadline=deadline).cpu()
            for offset in range(0, len(rows), batch_size)
        ]
    )


def fit_joint_temperature(logits, rows, *, deadline=None):
    """One scalar temperature from the temperature half of DEV, joint NLL only."""
    if not rows or any(row.get("split") != "dev" for row in rows):
        raise ValueError("temperature selection accepts development rows only")
    targets = structured_targets(rows).reshape(-1)
    values = logits.detach().cpu().reshape(-1, JOINT_STATE_COUNT)
    if len(targets) != len(values):
        raise ValueError("temperature logits and labels differ")
    losses = []
    choices = torch.logspace(-1, 1, 81)
    for temperature in choices:
        _check_deadline(deadline)
        losses.append(float(F.cross_entropy(values / temperature, targets)))
    best = min(range(len(losses)), key=losses.__getitem__)
    return {
        "architecture": STRUCTURED_HEAD_ARCHITECTURE,
        "decoder_version": STRUCTURED_DECODER_VERSION,
        "population_contract": STRUCTURED_POPULATION_CONTRACT,
        "joint_temperature": float(choices[best]),
        "joint_threshold": 0.99,
        "temperature_nll": losses[best],
        "temperature_decisions": targets.numel(),
        "temperature_groups": sorted(
            {row.get("semantic_mapping_id", row["group_id"]) for row in rows}
        ),
        "temperature_fit": "dev_temperature_half_joint_nll_grid_0.1_to_10_81",
        "thresholds_selected": "fixed_before_data",
    }


class JointLogitPredictor:
    """Evaluation adapter: pure learned logits plus the production joint decoder."""

    def __init__(self, rows, logits, calibration, *, model_hash="0" * 64):
        self.calibration = calibration
        self.model_hash = model_hash
        self.calibration_hash = content_hash(calibration)
        self.values = {row["text"]: values for row, values in zip(rows, logits, strict=True)}

    def predict(self, text):
        return decode_structured_predictions(
            self.values[text],
            self.calibration,
            text_hash=hashlib.sha256(text.encode()).hexdigest(),
            model_hash=self.model_hash,
            calibration_hash=self.calibration_hash,
        )


def joint_diagnostics(logits, rows, calibration):
    """Separate representation errors, confidence rejection and decoder decisions."""
    expected = structured_targets(rows)
    probabilities = (logits / calibration["joint_temperature"]).softmax(-1)
    confidence, actual = probabilities.max(-1)
    accepted = confidence >= calibration["joint_threshold"]
    known = expected >= 2
    actual_supported = actual >= 2
    mode_ids = torch.tensor(
        [
            0 if index < 2 else 1 + (index - 2) // len(CANDIDATES)
            for index in range(JOINT_STATE_COUNT)
        ]
    )
    expected_modes, actual_modes = mode_ids[expected], mode_ids[actual]
    expected_status = expected.clamp(max=2)
    actual_status = actual.clamp(max=2)
    return {
        "raw_argmax": {
            "joint_correct": int((actual == expected).sum()),
            "joint_total": expected.numel(),
            "status_correct": int((actual_status == expected_status).sum()),
            "status_total": expected.numel(),
            "supported_mode_correct": int(((expected_modes == actual_modes) & known).sum()),
            "supported_mode_total": int(known.sum()),
            "supported_representative_correct": int(
                (
                    (((actual - 2) % len(CANDIDATES)) == ((expected - 2) % len(CANDIDATES)))
                    & known
                    & actual_supported
                ).sum()
            ),
            "supported_representative_total": int(known.sum()),
        },
        "confidence": {
            "threshold": calibration["joint_threshold"],
            "accepted": int(accepted.sum()),
            "total": expected.numel(),
            "accepted_correct": int(((actual == expected) & accepted).sum()),
            "supported_accepted_correct": int(((actual == expected) & accepted & known).sum()),
            "supported_total": int(known.sum()),
        },
        "decoder": validate_text_control_predictions(
            JointLogitPredictor(rows, logits, calibration),
            rows,
        ),
    }


def _selection_key(feasibility, score_nll):
    # No public runtime, calibration or sealed-test information enters this key.
    return (bool(feasibility["passed"]), -score_nll)


def _safe_logsumexp(values, dim):
    """Exclude wholly masked slices before reducing, including in backward."""
    valid = torch.isfinite(values).any(dim=dim)
    safe = torch.where(valid.unsqueeze(dim), values, 0)
    return safe.logsumexp(dim=dim).masked_fill(~valid, -torch.inf)


def conditional_population_loss(logits, targets):
    """Equal means for status/all, mode/supported-discrete and pointer/supported.

    The current additive mode/pointer head factorizes within its physical mask.
    Conditional mode plus pointer NLL therefore equals conditional joint NLL;
    separate population means intentionally increase sparse positive gradients.
    Joint NLL is returned separately for honest diagnostics and remains the
    unchanged objective for temperature fitting and checkpoint score selection.
    """
    if (
        logits.ndim != 3
        or logits.shape[1:] != (len(CANDIDATES), JOINT_STATE_COUNT)
        or targets.shape != logits.shape[:2]
        or targets.dtype != torch.long
    ):
        raise ValueError(
            "population loss requires [batch,36,110] logits and long [batch,36] targets"
        )
    if not targets.numel():
        raise ValueError("population loss requires a nonempty batch")
    if (targets < 0).any() or (targets >= JOINT_STATE_COUNT).any():
        raise ValueError("joint target outside the structured state vocabulary")
    supported_values = logits[..., 2:].reshape(*targets.shape, 3, len(CANDIDATES))
    status_logits = torch.cat(
        (logits[..., :2], _safe_logsumexp(logits[..., 2:], -1)[..., None]), -1
    )
    status_total = F.cross_entropy(
        status_logits.flatten(0, 1), targets.clamp(max=2).flatten(), reduction="sum"
    )
    # Unlike logits.sum()*0, this zero does not multiply masked -inf by zero.
    zero = logits[..., :2].sum() * 0
    supported = targets >= 2
    discrete = torch.tensor(
        [candidate.startswith(("key:", "mouse:")) for candidate in CANDIDATES],
        device=targets.device,
    )
    mode_eligible = supported & discrete[None]
    mode_count, representative_count = int(mode_eligible.sum()), int(supported.sum())
    mode_total = zero
    if mode_count:
        # Select allowed physical modes before reducing the representative axis.
        mode_logits = _safe_logsumexp(supported_values[mode_eligible][:, :2], -1)
        mode_targets = (targets[mode_eligible] - 2) // len(CANDIDATES)
        mode_total = F.cross_entropy(mode_logits, mode_targets, reduction="sum")
    representative_total = zero
    if representative_count:
        # Representative columns above self are all -inf for every mode. The
        # safe reduction avoids NaN derivatives in those structural columns.
        representative_logits = _safe_logsumexp(supported_values[supported], 1)
        representative_targets = (targets[supported] - 2) % len(CANDIDATES)
        representative_total = F.cross_entropy(
            representative_logits, representative_targets, reduction="sum"
        )
    joint_total = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum")
    components = {
        "status": (status_total, targets.numel()),
        "mode": (mode_total, mode_count),
        "representative": (representative_total, representative_count),
        "joint": (joint_total, targets.numel()),
    }
    objective = sum(
        total / count if count else total
        for name, (total, count) in components.items()
        if name != "joint"
    )
    return objective, components


def fit_structured_head(
    features,
    records,
    *,
    seed=20261021,
    epochs=4,
    batch_size=32,
    learning_rate=3e-4,
    device="cpu",
    deadline=None,
    checkpoint_directory=None,
    progress=None,
):
    """Train the three conditional populations with fresh AdamW, without class priors."""
    if type(epochs) is not int or not 1 <= epochs <= 10:
        raise ValueError("structured fitting requires 1..10 epochs")
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("batch_size must be in 1..128")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("positive finite learning rate required")
    train = [row for row in records if row["split"] == "train"]
    temperature_rows, score_rows = split_development_groups(records, seed=seed)
    if not train or not temperature_rows or not score_rows:
        raise ValueError("TRAIN and two independent DEV group halves are required")
    _check_deadline(deadline)
    features.eval().requires_grad_(False)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    head = JointStructuredControlHead().to(device)
    initial_hash = encoder_state_hash(head.state_dict())
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=STRUCTURED_ADAMW_WEIGHT_DECAY,
    )
    if optimizer.state:
        raise RuntimeError("fresh structured fit must start with empty optimizer state")
    directory = Path(checkpoint_directory) if checkpoint_directory else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
    history, updates, examples = [], 0, 0
    selected, best_key, selected_epoch, selected_calibration = None, None, None, None
    generator = random.Random(seed)

    def checkpoint(path, epoch):
        torch.save(
            {
                "architecture": STRUCTURED_HEAD_ARCHITECTURE,
                "model_version": STRUCTURED_MODEL_VERSION,
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
                name: [0.0, 0] for name in ("status", "mode", "representative", "joint")
            }
            for offset in range(0, len(order), batch_size):
                _check_deadline(deadline)
                rows = order[offset : offset + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = _forward(features, head, rows, deadline=deadline)
                targets = structured_targets(rows, device=logits.device)
                loss, components = conditional_population_loss(logits, targets)
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
                            "objective_components": {
                                name: {"numerator": float(total.detach()), "denominator": count}
                                for name, (total, count) in components.items()
                            },
                        }
                    )
            temperature_logits = collect_joint_logits(
                features, head, temperature_rows, batch_size=batch_size, deadline=deadline
            )
            calibration = fit_joint_temperature(
                temperature_logits, temperature_rows, deadline=deadline
            )
            score_logits = collect_joint_logits(
                features, head, score_rows, batch_size=batch_size, deadline=deadline
            )
            score_nll = float(
                F.cross_entropy(
                    (score_logits / calibration["joint_temperature"]).reshape(
                        -1, JOINT_STATE_COUNT
                    ),
                    structured_targets(score_rows).reshape(-1),
                )
            )
            diagnostics = joint_diagnostics(score_logits, score_rows, calibration)
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
            "architecture": STRUCTURED_HEAD_ARCHITECTURE,
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
            "optimizer_weight_decay": STRUCTURED_ADAMW_WEIGHT_DECAY,
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
                    "architecture": STRUCTURED_HEAD_ARCHITECTURE,
                },
                directory / "selected_head.pt",
            )
        return head, training
    except BaseException:
        if directory:
            checkpoint(directory / "partial_head.pt", len(history))
        raise


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
):
    """Full training requires a complete, authenticated successful fresh-head probe."""
    from carryrl.rewards.control_text_runtime import runtime_control_rows

    path = Path(path)
    digest = _file_hash(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists() or sidecar.read_text().strip() != digest:
        raise ValueError("passed probe report hash is missing or mismatched")
    report = json.loads(path.read_text())
    config = report.get("config", {})
    _, runtime_identity = runtime_control_rows(runtime_instructions, runtime_controls)
    checkpoint = path.parent / "selected_head.pt"
    if (
        report.get("status") != "complete"
        or config.get("probe_only") is not True
        or config.get("architecture") != STRUCTURED_HEAD_ARCHITECTURE
        or config.get("dataset_sha256") != dataset_sha256
        or config.get("source_identity") != source_identity()
        or config.get("encoder_state_sha256") != encoder_state_sha256
        or config.get("seed") == seed
        or config.get("learning_rate") != learning_rate
        or config.get("batch_size") != batch_size
        or config.get("optimizer_weight_decay") != STRUCTURED_ADAMW_WEIGHT_DECAY
        or config.get("loss_protocol") != LOSS_PROTOCOL
        or report.get("feasibility", {}).get("passed") is not True
        or report.get("runtime", {}).get("passed") is not True
        or report.get("runtime", {}).get("records") != 32
        or report.get("runtime", {}).get("passed_records") != 32
        or len(report.get("runtime", {}).get("results", [])) != 32
        or any(
            value.get("passed") is not True or value.get("errors")
            for value in report.get("runtime", {}).get("results", [])
        )
        or report.get("sealed_test_used") is not False
        or not checkpoint.is_file()
        or _file_hash(checkpoint) != report.get("selected_checkpoint_sha256")
        or any(report["runtime"].get(key) != value for key, value in runtime_identity.items())
    ):
        raise ValueError(
            "full training requires a passed structured probe, unchanged runtime32 and fresh seed"
        )
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "seed": config["seed"],
    }


def train_structured_text_control_artifact(
    dataset,
    output,
    *,
    encoder,
    probe_only=True,
    passed_probe=None,
    epochs=4,
    seed=20261021,
    feature_cache=None,
    reuse_feature_cache=None,
    max_feature_bytes=8 * 1024**3,
    max_seconds=10800,
    batch_size=32,
    learning_rate=3e-4,
    device="cpu",
    progress=None,
    runtime_instructions="configs/atomic_instructions.json",
    runtime_controls="configs/atomic_controls.json",
    evaluate_sealed=True,
):
    """Run bounded training, preserve every checkpoint, keep export readiness false.

    A full run may inspect sealed data only once, after selected head/temperature
    freeze and successful unchanged calibration and exact public runtime gates.
    """
    from carryrl.rewards.control_text_runtime import (
        runtime_control_rows,
        validate_runtime_text_controls,
    )

    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 10800:
        raise ValueError("fit budget must be positive and at most three hours")
    if probe_only and epochs not in (4, 8):
        raise ValueError("development probes require four or eight fixed epochs")
    if not probe_only and not 1 <= epochs <= 10:
        raise ValueError("full training is limited to ten epochs")
    if feature_cache and reuse_feature_cache:
        raise ValueError("choose new features or authenticated reuse")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"structured output already exists: {output}")
    start, deadline = time.monotonic(), time.monotonic() + max_seconds
    manifest, records = read_control_dataset(dataset)
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
        )
    # Probe extraction and authenticated reuse are restricted to TRAIN/DEV.
    # A full fit requires its own cache including calibration, never sealed inputs.
    cache = Path(
        reuse_feature_cache or feature_cache or output.with_name(output.name + ".features")
    )
    if output.resolve().is_relative_to(cache.resolve()) or cache.resolve().is_relative_to(
        output.resolve()
    ):
        raise ValueError("structured output and E5 feature cache must be separate")
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "architecture": STRUCTURED_HEAD_ARCHITECTURE,
        "model_version": STRUCTURED_MODEL_VERSION,
        "probe_only": probe_only,
        "epochs": epochs,
        "seed": seed,
        "ready": False,
        "dataset_sha256": manifest["records_sha256"],
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer_weight_decay": STRUCTURED_ADAMW_WEIGHT_DECAY,
        "loss_protocol": LOSS_PROTOCOL,
        "max_seconds": max_seconds,
        "feature_cache": str(cache.resolve()),
        "encoded_splits": ["train", "dev"] if probe_only else ["train", "dev", "calibration"],
        "max_feature_bytes": max_feature_bytes,
        "encoder_state_sha256": original_encoder_hash,
        "passed_probe": authorization,
        "source_identity": source_identity(),
        "checkpoint_selection": "dev_feasible_first_joint_nll_only",
        "sealed_test_used": False,
    }
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
        preflight = verify_grounded_text_inputs(encoder.tokenizer, fit_rows, deadline=deadline)
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
        head, training = fit_structured_head(
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
        )
        calibration = copy.deepcopy(training["selected_development_calibration"])
        selected_epoch = training["selected_epoch"]
        selected_report = training["history"][selected_epoch - 1]
        feasibility = selected_report["feasibility"]
        model_hash = encoder_state_hash(head.state_dict())
        selected_checkpoint = torch.load(
            output / "checkpoints" / f"epoch_{selected_epoch:02d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        selected_checkpoint.update(training=training, ready=False)
        torch.save(selected_checkpoint, output / "selected_head.pt")
        _write_json(output / "selected_calibration.json", calibration)
        _write_json(output / "training.json", training)
        notify({"stage": "public_runtime"})
        runtime_rows, _ = runtime_control_rows(runtime_instructions, runtime_controls)
        runtime_logits = collect_joint_logits(
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
        runtime_predictor = JointLogitPredictor(
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
            joint_diagnostics(runtime_logits, runtime_rows, calibration),
        )
        validation, sealed = None, None
        if not probe_only:
            notify({"stage": "calibration_gate"})
            gate_rows = [row for row in records if row["split"] == "calibration"]
            logits = collect_joint_logits(
                features, head, gate_rows, batch_size=batch_size, deadline=deadline
            )
            validation = validate_text_control_predictions(
                JointLogitPredictor(gate_rows, logits, calibration, model_hash=model_hash),
                gate_rows,
            )
            _write_json(output / "calibration_validation.json", validation)
            _write_json(
                output / "calibration_diagnostics.json",
                joint_diagnostics(logits, gate_rows, calibration),
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
                logits = collect_joint_logits(
                    encoder, head, sealed_rows, batch_size=batch_size, deadline=deadline
                )
                sealed = validate_text_control_predictions(
                    JointLogitPredictor(sealed_rows, logits, calibration, model_hash=model_hash),
                    sealed_rows,
                )
                _write_json(output / "sealed_validation.json", sealed)
        final_encoder_hash = encoder_state_hash(encoder.state_dict())
        if final_encoder_hash != original_encoder_hash:
            raise RuntimeError("frozen original E5 changed during joint-head fitting")
        training.update(
            sealed_test_used=sealed is not None,
            encoder_frozen_verified=True,
            encoder_state_sha256=final_encoder_hash,
            dataset_sha256=manifest["records_sha256"],
        )
        artifact_manifest = None
        if not probe_only:
            from carryrl.rewards.control_text_structured_artifact import (
                save_structured_control_artifact,
            )

            artifact_manifest = save_structured_control_artifact(
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
            "version": "carryrl.control_text_structured_training.v1",
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
