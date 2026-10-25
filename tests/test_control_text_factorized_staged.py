"""Fresh staged continuation keeps terminal state, sampler ownership and DEV selection explicit."""

import copy
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import pytest
import torch
from test_control_text_factorized import calibration
from test_control_text_factorized_training import relational_row
from torch import nn

from carryrl.rewards import control_text_factorized_training as training


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Parameter(torch.randn(1))
        self.applicability = nn.Linear(1, 3)
        self.mode = nn.Linear(1, 4)
        self.pair = nn.Linear(1, 3)


def staged_harness(monkeypatch, *, refined_wins=True, fail_margin=False):
    monkeypatch.setattr(training, "STAGED_BASE_EPOCHS", 2)
    monkeypatch.setattr(training, "STAGED_TRAIN_RECORDS", 64)
    monkeypatch.setattr(training, "STAGED_CONTINUATION_UPDATES", 2)
    monkeypatch.setattr(training, "STAGED_MARGIN_UPDATES", 2)
    records = [{**relational_row(), "id": f"train-{i}"} for i in range(64)]
    dev = [{**relational_row(), "split": "dev", "id": f"dev-{i}"} for i in range(2)]
    for value in dev:
        value.pop("relational_panel")
    calls, source = [], {}

    def forward(features, head, rows, **kwargs):
        if fail_margin and not head.trunk.requires_grad and rows[0]["split"] == "train":
            raise TimeoutError("synthetic margin interruption")
        calls.append(
            {
                "ids": [row["id"] for row in rows],
                "rng": torch.get_rng_state().clone(),
                "trunk": head.trunk.detach().clone(),
                "grad": torch.is_grad_enabled(),
            }
        )
        feature = (head.trunk + torch.rand(1))[:, None]
        return {
            "applicability": head.applicability(feature)[None].expand(len(rows), 36, 3),
            "mode": head.mode(feature)[None].expand(len(rows), 36, 4),
            "pair": head.pair(feature)[None, None].expand(len(rows), 36, 36, 3),
        }

    def base_fit(features, records, *, checkpoint_directory, seed, **kwargs):
        torch.manual_seed(seed)
        head, generator = TinyHead(), random.Random(seed)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.0003, weight_decay=0.01)
        directory = Path(checkpoint_directory)
        directory.mkdir(parents=True, exist_ok=True)
        history, examples = [], 0
        base_calibration = calibration() | {"pair_temperature": 2.5}
        for epoch in (1, 2):
            order = list(range(64))
            generator.shuffle(order)
            for offset in (0, 32):
                batch = [records[index] for index in order[offset : offset + 32]]
                optimizer.zero_grad(set_to_none=True)
                values = forward(features, head, batch)
                loss, _ = training.factorized_population_loss(
                    values, training.factorized_targets(batch)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                examples += len(batch)
            history.append(
                {
                    "epoch": epoch,
                    "optimizer_updates": epoch * 2,
                    "score_joint_nll": (1000.0 if refined_wins else 0.0) + epoch - 1,
                    "feasibility": {"passed": True},
                    "calibration": base_calibration,
                }
            )
            payload = {
                "head": copy.deepcopy(head.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "epoch": epoch,
                "optimizer_updates": epoch * 2,
                "train_examples": examples,
                "torch_rng_state": torch.get_rng_state().clone(),
                "python_rng_state": generator.getstate(),
            }
            torch.save(payload, directory / f"epoch_{epoch:02d}.pt")
        source.update(terminal=payload, first_calls=len(calls))
        selected = torch.load(directory / "epoch_01.pt", weights_only=True)
        head.load_state_dict(selected["head"])
        torch.rand(3)  # Staging must restore terminal RNG, not inherit this process state.
        return head, {
            "trained": True,
            "initial_head_sha256": "initial",
            "selected_epoch": 1,
            "history": history,
            "selected_development_calibration": base_calibration,
            "completed_epochs": 2,
            "epochs": 2,
            "train_records": 64,
        }

    monkeypatch.setattr(training, "fit_factorized_head", base_fit)
    monkeypatch.setattr(training, "_forward", forward)
    monkeypatch.setattr(
        training, "split_factorized_development", lambda records: ([dev[0]], [dev[1]])
    )
    monkeypatch.setattr(
        training,
        "fit_factorized_temperatures",
        lambda *args, **kwargs: calibration() | {"pair_temperature": 7.0},
    )
    monkeypatch.setattr(training, "factorized_diagnostics", lambda *args: {"decoder": {}})
    monkeypatch.setattr(training, "developmental_feasibility_gate", lambda *args: {"passed": True})
    return records, calls, source


@pytest.mark.parametrize("refined_wins", [False, True])
def test_staged_restores_terminal_adamw_rng_cursor_and_freezes_unary_state(
    monkeypatch,
    tmp_path,
    refined_wins,
):
    records, calls, source = staged_harness(monkeypatch, refined_wins=refined_wins)
    head, report = training._fit_staged_factorized_head(
        nn.Identity(),
        records,
        seed=20261029,
        epochs=2,
        batch_size=32,
        learning_rate=0.0003,
        device="cpu",
        deadline=None,
        checkpoint_directory=tmp_path,
        progress=None,
    )
    terminal = source["terminal"]
    subsequent = [
        call for call in calls[source["first_calls"] :] if call["ids"][0].startswith("train")
    ]
    assert torch.equal(subsequent[0]["rng"], terminal["torch_rng_state"])
    assert torch.equal(subsequent[0]["trunk"], terminal["head"]["trunk"])
    generator = random.Random()
    generator.setstate(terminal["python_rng_state"])
    expected = []
    for _ in range(2):
        order = list(range(64))
        generator.shuffle(order)
        expected += [[f"train-{i}" for i in order[offset : offset + 32]] for offset in (0, 32)]
    assert [call["ids"] for call in subsequent] == expected
    continued = torch.load(tmp_path / "continuation_000006.pt", weights_only=True)
    refined = torch.load(tmp_path / "required_margin_000008.pt", weights_only=True)
    assert refined["frozen_verified"]
    assert refined["frozen_state_before"] == refined["frozen_state_after"]
    assert refined["resume_cursor"]["epoch"] == 5
    assert refined["resume_cursor"]["epoch_order"] is None
    assert refined["resume_cursor"]["next_row_offset"] == 0
    for name in refined["head"]:
        if not name.startswith("pair."):
            assert torch.equal(refined["head"][name], continued["head"][name])
    names = list(head.named_parameters())
    for index, (name, _) in enumerate(names):
        state = refined["optimizer"]["state"][index]
        assert int(state["step"]) == (8 if name.startswith("pair.") else 6)
        if not name.startswith("pair."):
            reference = continued["optimizer"]["state"][index]
            assert all(torch.equal(value, reference[key]) for key, value in state.items())
    candidate = report["selected_candidate"]
    assert candidate["stage"] == ("required_margin" if refined_wins else "base")
    assert report["selected_epoch"] == (None if refined_wins else 1)
    assert report["selected_development_calibration"]["pair_temperature"] == (
        7.0 if refined_wins else 2.5
    )
    assert report["staged_completion"]["pair_temperature"] == 2.5
    assert report["staged_completion"]["terminal_updates"] == 8
    assert report["development_training_hyperparameter_provenance"]["used_for_training"]
    assert training._staged_training_complete(report)
    selected_cp = torch.load(tmp_path / candidate["checkpoint_file"], weights_only=True)
    assert all(
        torch.equal(value, selected_cp["head"][name]) for name, value in head.state_dict().items()
    )


def test_interrupted_staged_margin_preserves_actual_partial_cursor_without_completion(
    monkeypatch,
    tmp_path,
):
    records, _, _ = staged_harness(monkeypatch, fail_margin=True)
    with pytest.raises(TimeoutError, match="synthetic margin"):
        training._fit_staged_factorized_head(
            nn.Identity(),
            records,
            seed=20261029,
            epochs=2,
            batch_size=32,
            learning_rate=0.0003,
            device="cpu",
            deadline=None,
            checkpoint_directory=tmp_path,
            progress=None,
        )
    partial = torch.load(tmp_path / "partial_staged_head.pt", weights_only=True)
    assert partial["stage"] == "required_margin" and partial["stage_updates"] == 0
    assert partial["optimizer_updates"] == 6 and partial["ready"] is False
    assert partial["stage_complete"] is False
    assert partial["resume_cursor"]["epoch"] == 4
    assert partial["resume_cursor"]["next_row_offset"] == 0
    assert len(partial["resume_cursor"]["epoch_order"]) == 64
    assert not training._staged_training_complete(partial)


def test_staged_protocol_is_explicit_and_legacy_identity_remains_default():
    legacy = training.protocol_identity()
    staged = training.protocol_identity(training.STAGED_TRAINING_PROTOCOL)
    assert legacy == training.protocol_identity(training.LOSS_PROTOCOL)
    assert "training_protocol" not in legacy
    assert staged["base_epochs"] == 8
    assert staged["continuation_updates"] == 2200 and staged["required_margin_updates"] == 1000
    assert staged["train_records"] == 15264
    assert staged["development_training_hyperparameters"] == ["temperature", "selection"]
    assert staged["full_max_seconds"] == 7200 and staged["probe_max_seconds"] == 4200


def staged_training_report():
    base_calibration = calibration() | {"pair_temperature": 2.5}
    refined_calibration = calibration() | {"pair_temperature": 3.5}
    source = dict(
        split="dev",
        roles=["temperature", "selection"],
        used_for_training=True,
        base_selected_epoch=1,
        base_selected_updates=477,
        base_calibration_sha256=training.content_hash(base_calibration),
        pair_temperature=2.5,
    )
    proof = dict(weights_and_buffers_sha256="a" * 64, optimizer_state_sha256="b" * 64)
    return dict(
        training_protocol=training.STAGED_TRAINING_PROTOCOL,
        train_records=15264,
        completed_epochs=8,
        optimizer_updates=7016,
        selected_epoch=None,
        selected_development_calibration=refined_calibration,
        selected_candidate=dict(
            stage="required_margin",
            base_epoch=None,
            optimizer_updates=7016,
            checkpoint_file="required_margin_007016.pt",
            history_index=8,
        ),
        history=[
            dict(stage="base", optimizer_updates=477 * epoch, calibration=base_calibration)
            for epoch in range(1, 9)
        ]
        + [dict(stage="required_margin", optimizer_updates=7016, calibration=refined_calibration)],
        staged_completion=dict(
            complete=True,
            base_updates=3816,
            continuation_updates=2200,
            required_margin_updates=1000,
            terminal_updates=7016,
            frozen_before=proof,
            frozen_after=proof,
            frozen_verified=True,
            pair_temperature=2.5,
            temperature_source=source,
        ),
        development_training_hyperparameter_provenance=source,
    )


def probe_fixture(tmp_path, monkeypatch, training_protocol):
    from carryrl.rewards import control_text_factorized_artifact as artifact
    from carryrl.rewards import control_text_factorized_evaluation as evaluation

    identity = dict(
        instructions_sha256="instructions",
        controls_sha256="controls",
        specification_hash="specification",
    )
    monkeypatch.setattr(evaluation, "factorized_runtime_control_rows", lambda *args: ([], identity))
    monkeypatch.setattr(artifact, "_runtime_gate", lambda report: True)
    checkpoint = tmp_path / "selected_head.pt"
    checkpoint.write_bytes(b"authenticated fixture checkpoint")
    report = dict(
        status="complete",
        config=dict(
            probe_only=True,
            architecture=training.FACTORIZED_HEAD_ARCHITECTURE,
            dataset_sha256="dataset",
            encoder_state_sha256="encoder",
            seed=20261029,
            source_identity=training.source_identity(),
            protocol=training.protocol_identity(training_protocol),
            learning_rate=0.0003,
            batch_size=32,
            optimizer_weight_decay=0.01,
            loss_protocol=training_protocol,
            encoded_splits=["train", "dev"],
        ),
        training=dict(
            gradient_splits=["train"], selection_splits=["dev"], public_used_for_selection=False
        ),
        feasibility=dict(passed=True),
        runtime=dict(
            passed=True,
            records=32,
            passed_records=32,
            results=[dict(passed=True, errors=[]) for _ in range(32)],
            **identity,
        ),
        calibration_used=False,
        sealed_test_used=False,
        encoder_frozen_verified=True,
        selected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    if training_protocol == training.STAGED_TRAINING_PROTOCOL:
        report["config"]["training_protocol"] = training_protocol
        report["training"].update(staged_training_report())
    return tmp_path / "probe_report.json", report


def verify_probe(path, report, training_protocol):
    path.write_text(json.dumps(report))
    path.with_suffix(".json.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    return training.verify_passed_probe(
        path,
        dataset_sha256="dataset",
        seed=20261030,
        encoder_state_sha256="encoder",
        learning_rate=0.0003,
        batch_size=32,
        runtime_instructions="instructions",
        runtime_controls="controls",
        training_protocol=training_protocol,
    )


def test_full_probe_authorization_binds_legacy_and_staged_recipes(tmp_path, monkeypatch):
    legacy, staged = training.LOSS_PROTOCOL, training.STAGED_TRAINING_PROTOCOL
    path, report = probe_fixture(tmp_path, monkeypatch, legacy)
    assert verify_probe(path, report, legacy)["seed"] == 20261029
    with pytest.raises(ValueError, match="passed factorized probe"):
        verify_probe(path, report, staged)
    path, report = probe_fixture(tmp_path, monkeypatch, staged)
    assert verify_probe(path, report, staged)["seed"] == 20261029
    with pytest.raises(ValueError, match="passed factorized probe"):
        verify_probe(path, report, legacy)


@pytest.mark.parametrize("change", ["partial", "frozen", "temperature", "selected_file"])
def test_staged_probe_rejects_incomplete_or_inconsistent_stage_provenance(
    tmp_path,
    monkeypatch,
    change,
):
    path, report = probe_fixture(tmp_path, monkeypatch, training.STAGED_TRAINING_PROTOCOL)
    value = report["training"]
    if change == "partial":
        value["staged_completion"]["required_margin_updates"] = 999
    elif change == "frozen":
        value["staged_completion"]["frozen_after"] = {"changed": "c" * 64}
    elif change == "temperature":
        value["staged_completion"]["pair_temperature"] = 7.0
    else:
        value["selected_candidate"]["checkpoint_file"] = "epoch_08.pt"
    with pytest.raises(ValueError, match="passed factorized probe"):
        verify_probe(path, report, training.STAGED_TRAINING_PROTOCOL)


@pytest.mark.parametrize("option", [dict(epochs=10), dict(batch_size=16), dict(device="mps")])
def test_staged_fit_rejects_incompatible_settings_before_loading_features(option):
    kwargs = dict(epochs=8, batch_size=32, device="cpu", checkpoint_directory="unused") | option
    with pytest.raises(ValueError, match="staged protocol"):
        training.fit_factorized_head(
            None,
            [],
            training_protocol=training.STAGED_TRAINING_PROTOCOL,
            **kwargs,
        )


def cli_module(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "factorized_staged_cli", scripts / "train_text_control_factorized_reward.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("extra", [["--epochs", "10"], ["--batch-size", "16"], ["--device", "mps"]])
def test_staged_cli_rejects_incompatible_settings_before_encoder(monkeypatch, extra):
    cli = cli_module(monkeypatch)
    monkeypatch.setattr(
        cli, "MultilingualInstructionEncoder", lambda **kwargs: pytest.fail("encoder")
    )
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--dataset",
                "data",
                "--output",
                "out",
                "--probe-only",
                "--training-protocol",
                training.STAGED_TRAINING_PROTOCOL,
                *extra,
            ]
        )
    assert error.value.code == 2


def test_staged_cli_full_defaults_to_eight_base_epochs_and_explicit_recipe(monkeypatch):
    cli, received = cli_module(monkeypatch), {}
    monkeypatch.setattr(cli, "MultilingualInstructionEncoder", lambda **kwargs: nn.Identity())

    def fit(*args, **kwargs):
        received.update(kwargs)
        return dict(
            status="complete", ready=False, feasibility={}, runtime={}, admission_gates_passed=False
        )

    monkeypatch.setattr(cli, "train_factorized_text_control_artifact", fit)
    assert (
        cli.main(
            [
                "--dataset",
                "data",
                "--output",
                "out",
                "--passed-probe",
                "probe",
                "--training-protocol",
                training.STAGED_TRAINING_PROTOCOL,
            ]
        )
        == 0
    )
    assert received["epochs"] == 8 and received["seed"] == 20261030
    assert received["training_protocol"] == training.STAGED_TRAINING_PROTOCOL
