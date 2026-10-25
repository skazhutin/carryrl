import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import BertConfig, BertModel

from carryrl.agent.actions import PhysicalAction
from carryrl.agent.language import MultilingualInstructionEncoder
from carryrl.rewards.composition import (
    OUTCOME_POTENTIAL_VERSION,
    TEXT_CONTROL_VERSION,
    RewardComposer,
    RewardCompositionConfig,
    control_schedule_coefficient,
)
from carryrl.rewards.control_text import (
    CANDIDATES,
    ControlPrediction,
    ControlPredictionTable,
    FrozenTextControlReward,
    ResidualTextControlHead,
    TextControlHead,
    calibration_gates,
    content_hash,
    load_text_control_reward,
    save_text_control_artifact,
)
from carryrl.rewards.control_text_training import (
    fit_text_control_head,
    read_control_dataset,
    validate_text_control_predictions,
)


class Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(kwargs)
        ids = [1, *[ord(char) % 251 for char in text], 2]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids))}

    def save_pretrained(self, directory):
        (directory / "tokenizer.fixture").write_text("local")


def encoder():
    return MultilingualInstructionEncoder(
        tokenizer=Tokenizer(),
        backbone=BertModel(
            BertConfig(
                hidden_size=384,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=32,
                vocab_size=251,
                max_position_embeddings=512,
            )
        ),
    )


def table(**predictions):
    rows = []
    for candidate in CANDIDATES:
        status, group = predictions.get(candidate, ("unknown", None))
        rows.append(
            ControlPrediction(
                candidate, status, "held" if status == "supported" else "unknown", group, 1.0
            )
        )
    return ControlPredictionTable("0" * 64, "1" * 64, "2" * 64, tuple(rows))


def visual(status="play"):
    return {
        "known": True,
        "score": 0.0,
        "score_confidence": 1.0,
        "success_probability": 0.99 if status == "success" else 0.01,
        "failure_probability": 0.99 if status == "failure" else 0.01,
    }


def composer(predictions, **kwargs):
    return RewardComposer(
        RewardCompositionConfig(TEXT_CONTROL_VERSION, {"control_schedule": "permanent", **kwargs}),
        gamma=0.99,
        horizon=20,
        score_confidence_threshold=0.9,
        initial_prediction=visual(),
        control_table=predictions,
        control_coefficient=1.0,
    )


def step(reward, action=PhysicalAction(), *, terminal=False, prediction=None):
    return reward.step(
        visual("success") if terminal else (prediction or visual()),
        action,
        terminated=terminal,
        truncated=False,
    )


def calibration():
    return {
        f"{kind}_{name}": value
        for kind in ("status", "mode", "pair")
        for name, value in (("threshold", 0.99), ("temperature", 1.0))
    }


def report(n=400):
    return {
        "critical_errors": 0,
        "clear_count": 1000,
        "clear_accepted": 1000,
        "accepted_count": 1000,
        "accepted_correct": 1000,
        "accepted_scenarios": n,
        "error_scenarios": 0,
        "signs": {
            name: {
                "clear_count": 500,
                "clear_accepted": 500,
                "accepted_count": 500,
                "accepted_correct": 500,
                "accepted_scenarios": n,
                "error_scenarios": 0,
            }
            for name in ("supported", "unsupported")
        },
        "modes": {
            name: {"clear_count": 20, "accepted_count": 20, "correct_count": 20}
            for name in ("held", "press")
        },
        "aliases": {"clear_count": 20, "accepted_count": 20, "correct_count": 20},
    }


def test_strict_contextual_tokens_reject_overflow_without_truncation_and_copy_cache():
    language = encoder()
    tokens = language.encode_instruction_tokens("hello")
    assert tokens.shape == (14, 384) and not tokens.requires_grad
    assert language.tokenizer.calls[-1]["truncation"] is False
    original = tokens.clone()
    tokens.zero_()
    torch.testing.assert_close(language.encode_instruction_tokens("hello"), original)
    with pytest.raises(ValueError, match="never truncated"):
        language.encode_instruction_tokens("a" * 510)
    language.load_state_dict(language.state_dict())
    assert not language._strict_token_cache


def test_head_shapes_pair_symmetry_and_frozen_constructor_gate():
    head = TextControlHead()
    output = head(torch.randn(2, 8, 384))
    assert output["status"].shape == (2, 36, 3)
    assert output["mode"].shape == (2, 36, 3)
    assert output["pair"].shape == (2, 36, 36, 3)
    torch.testing.assert_close(output["pair"], output["pair"].transpose(1, 2))
    with pytest.raises(ValueError, match="unvalidated"):
        FrozenTextControlReward(
            None, head, model_hash="0" * 64, calibration=calibration(), calibration_hash="0" * 64
        )


def test_unknown_release_and_noop_zero_alias_hold_repress_and_motion_deduplicated():
    reward = composer(
        table(
            **{
                "key:w": ("supported", "forward"),
                "key:up": ("supported", "forward"),
                "dx:+": ("supported", "aim_right"),
            }
        )
    )
    assert step(reward).control_bonus == 0
    assert step(reward, PhysicalAction(keys=("f",))).control_bonus == 0
    assert step(reward, PhysicalAction(keys=("w", "up"))).control_bonus == 0.001
    assert step(reward, PhysicalAction(keys=("w",))).control_bonus == 0
    assert step(reward).control_bonus == 0
    assert step(reward, PhysicalAction(keys=("up",))).control_bonus == 0
    assert step(reward, PhysicalAction(dx=1)).control_bonus == 0.001
    assert step(reward, PhysicalAction(dx=64)).control_bonus == 0


def test_unsupported_veto_even_on_repeat_and_mark_all_used_classes_seen():
    reward = composer(
        table(
            **{
                "key:w": ("supported", "forward"),
                "key:e": ("supported", "interact"),
                "key:f": ("unsupported", "f"),
            }
        )
    )
    first = step(reward, PhysicalAction(keys=("w", "f")))
    assert first.control_bonus == 0 and first.unsupported_penalty == -0.001
    assert step(reward, PhysicalAction(keys=("w",))).control_bonus == 0
    repeated = step(reward, PhysicalAction(keys=("e", "f")))
    assert repeated.control_bonus == 0 and repeated.unsupported_penalty == 0
    assert step(reward, PhysicalAction(keys=("e",))).control_bonus == 0


def test_only_one_positive_class_per_command_and_terminal_penalties():
    reward = composer(
        table(
            **{
                "key:w": ("supported", "forward"),
                "key:e": ("supported", "interact"),
                "key:f": ("unsupported", "f"),
            }
        )
    )
    assert step(reward, PhysicalAction(keys=("w", "e"))).control_bonus == 0.001
    assert step(reward, PhysicalAction(keys=("e",))).control_bonus == 0
    terminal = step(reward, PhysicalAction(keys=("f",)), terminal=True)
    assert terminal.unsupported_penalty == -0.001
    assert terminal.control_bonus == 0
    assert terminal.total == pytest.approx(
        sum(
            (
                terminal.outcome,
                terminal.progress,
                terminal.time_cost,
                terminal.input_cost,
                terminal.control_bonus,
                terminal.unsupported_penalty,
            )
        )
    )
    fresh = composer(table(**{"key:w": ("supported", "forward")}))
    assert step(fresh, PhysicalAction(keys=("w",)), terminal=True).control_bonus == 0


def test_per_sign_episode_cap_and_invalid_visual_terminal_is_preserved():
    all_supported = table(**{candidate: ("supported", candidate) for candidate in CANDIDATES})
    reward = composer(all_supported)
    positives = [
        step(reward, PhysicalAction(keys=(key,))).control_bonus
        for key in ("w", "a", "s", "d", "space", "e", "f", "q", "r", "shift", "ctrl")
    ]
    assert sum(positives) <= 0.01 and positives[0] <= 0.001
    unknown_visual = visual()
    unknown_visual["known"] = False
    result = reward.step(
        unknown_visual, PhysicalAction(keys=("tab",)), terminated=True, truncated=False
    )
    assert not result.valid_for_learning and result.control_bonus == 0


def test_schedule_and_dominance_are_versioned_without_changing_visual_version():
    assert [control_schedule_coefficient("temporary", n, 100) for n in (0, 25, 37, 50, 99)] == [
        1,
        1,
        0.52,
        0,
        0,
    ]
    assert control_schedule_coefficient("zero", 1, 100) == 0
    assert control_schedule_coefficient("permanent", 99, 100) == 1
    params = {"time_cost": 0.98, "input_cost": 0.0}
    RewardCompositionConfig(OUTCOME_POTENTIAL_VERSION, params).validate_context(
        gamma=1, horizon=1, tasks=(), score_confidence_threshold=0.9
    )
    with pytest.raises(ValueError, match="trajectory costs"):
        RewardCompositionConfig(
            TEXT_CONTROL_VERSION, {**params, "control_schedule": "permanent"}
        ).validate_context(gamma=1, horizon=1, tasks=(), score_confidence_threshold=0.9)


def test_gates_require_scenario_sample_size_coverage_precision_and_no_critical_errors():
    assert calibration_gates(report())["passed"]
    assert not calibration_gates(report(100))["passed"]
    assert not calibration_gates({**report(), "clear_accepted": 799})["passed"]
    assert not calibration_gates({**report(), "critical_errors": 1})["passed"]


def test_artifact_gated_roundtrip_hash_tamper_and_shared_encoder_export(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "carryrl.agent.language.AutoTokenizer.from_pretrained", lambda *args, **kwargs: Tokenizer()
    )
    language, head = encoder(), TextControlHead()
    path = tmp_path / "model"
    save_text_control_artifact(
        path,
        encoder=language,
        head=head,
        calibration=calibration(),
        validation=report(),
        training={"trained": True},
    )
    model = load_text_control_reward(path)
    first = model.predict("Hold W")
    assert model.predict("Hold W") is first
    exported = model.save_resources(tmp_path / "export")
    assert not (exported / "encoder.pt").exists()
    archived = tmp_path / "archived_source"
    path.rename(archived)
    model.save_resources(tmp_path / "export_after_archive")
    archived.rename(path)
    shared = load_text_control_reward(exported, shared_encoder=language)
    assert shared.identity == model.identity
    assert shared.predict("Hold W").to_dict() == first.to_dict()
    with pytest.raises(ValueError, match="shared E5"):
        load_text_control_reward(exported)
    with pytest.raises(ValueError, match="differ"):
        load_text_control_reward(exported, shared_encoder=encoder())
    (path / "head.pt").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_text_control_reward(path)


def test_unready_artifact_never_loads_as_runtime_model(tmp_path):
    save_text_control_artifact(
        tmp_path / "model",
        encoder=encoder(),
        head=TextControlHead(),
        calibration=calibration(),
        validation=report(2),
        training={"trained": True},
    )
    with pytest.raises(ValueError, match="readiness gates failed"):
        load_text_control_reward(tmp_path / "model")


def test_prediction_evaluator_counts_grouped_errors_not_paraphrases():
    predicted = table(**{"key:w": ("supported", "forward")})
    labels = [
        {"candidate": name, "status": "unsupported", "mode": "unknown", "alias_group": 0}
        for name in CANDIDATES
    ]
    rows = [{"text": "Press W", "labels": labels, "group_id": "one"}] * 4
    predictor = SimpleNamespace(predict=lambda text: predicted)
    result = validate_text_control_predictions(predictor, rows)
    assert result["accepted_scenarios"] == 1 and result["error_scenarios"] == 1
    assert result["accepted_count"] == 4 and result["critical_errors"] == 4


def test_toy_fit_updates_only_head_and_excludes_sealed_rows():
    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(384))
            self.texts = []

        def encode_instruction_tokens(self, text):
            self.texts.append(text)
            return self.weight.detach()[None].repeat(2, 1)

    labels = [
        {"candidate": name, "status": "unsupported", "mode": "unknown", "alias_group": 0}
        for name in CANDIDATES
    ]
    rows = [
        {"text": split, "labels": labels, "split": split}
        for split in ("train", "dev", "calibration", "sealed_test")
    ]
    language = TinyEncoder()
    original = language.weight.detach().clone()
    _, training = fit_text_control_head(language, rows, epochs=1)
    assert training["updates"] == 1 and training["selection"] == "dev_nll"
    assert language.texts == ["train", "dev"]
    torch.testing.assert_close(language.weight, original)
    assert not language.weight.requires_grad


def test_dataset_reader_rejects_hash_corruption(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"records_sha256": "0" * 64}))
    (tmp_path / "records.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_control_dataset(tmp_path)


def test_table_roundtrip_and_hash_identity_include_prediction_semantics():
    source = table(**{"key:w": ("supported", "forward")})
    restored = ControlPredictionTable.from_dict(json.loads(json.dumps(source.to_dict())))
    assert restored == source and restored.identity == source.identity
    assert source.text_hash != content_hash("instruction")


def test_confident_alias_cliques_do_not_take_transitive_closure():
    class FixedHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = nn.Parameter(torch.zeros(1))

        def forward(self, tokens):
            status = torch.full((1, 36, 3), -20.0)
            status[:, :, 2] = 20
            mode = torch.full_like(status, -20)
            mode[:, :, 0] = 20
            indices = [CANDIDATES.index(name) for name in ("key:w", "key:a", "key:s")]
            for index in indices:
                status[0, index] = torch.tensor([20.0, -20, -20])
            pair = torch.full((1, 36, 36, 3), -20.0)
            pair[:, :, :, 2] = 20
            for left, right in zip(indices, indices[1:], strict=False):
                pair[0, left, right] = pair[0, right, left] = torch.tensor([20.0, -20, -20])
            return {"status": status, "mode": mode, "pair": pair}

    language = SimpleNamespace(encode_instruction_tokens=lambda text: torch.zeros(2, 384))
    model = FrozenTextControlReward(
        language,
        FixedHead(),
        model_hash="0" * 64,
        calibration=calibration(),
        calibration_hash="1" * 64,
        _verified=True,
    )
    predicted = {row.candidate: row for row in model.predict("Remapped controls").predictions}
    # A-B and B-C confident same cannot split into a bonus-paying singleton C
    # when A-C remains unknown. The entire ambiguous component abstains.
    assert all(predicted[name].status == "unknown" for name in ("key:w", "key:a", "key:s"))
    assert predicted["key:e"].status == "unknown"


def test_default_calibration_ideal_oracle_passes_both_signs_aliases_and_modes():
    from carryrl.rewards.control_text_data import generate_control_text_records

    rows = [row for row in generate_control_text_records() if row["split"] == "calibration"]

    class Oracle:
        def __init__(self, *, only_unsupported=False, split_aliases=False):
            self.lookup = {row["text"]: row for row in rows}
            self.only_unsupported, self.split_aliases = only_unsupported, split_aliases

        def predict(self, text):
            predictions = []
            for label in self.lookup[text]["labels"]:
                status = label["status"]
                if self.only_unsupported and status == "supported":
                    status = "unknown"
                group = None
                if status != "unknown":
                    group = (
                        f"alias-{label['alias_group']}"
                        if label["alias_group"] > 0 and not self.split_aliases
                        else label["candidate"]
                    )
                predictions.append(
                    ControlPrediction(label["candidate"], status, label["mode"], group, 1.0)
                )
            return ControlPredictionTable("0" * 64, "1" * 64, "2" * 64, tuple(predictions))

    perfect = validate_text_control_predictions(Oracle(), rows)
    assert perfect["gates"]["passed"]
    assert all(
        perfect["signs"][name]["accepted_scenarios"] >= 299 for name in ("supported", "unsupported")
    )
    unsupported_only = validate_text_control_predictions(Oracle(only_unsupported=True), rows)
    assert not unsupported_only["gates"]["passed"]
    assert unsupported_only["gates"]["signs"]["supported"]["clear_coverage"] == 0
    singleton = validate_text_control_predictions(Oracle(split_aliases=True), rows)
    assert not singleton["gates"]["passed"]
    assert singleton["aliases"]["false_split_errors"] > 0
    assert singleton["critical_errors"] > 0


@pytest.mark.parametrize("head_type", (TextControlHead, ResidualTextControlHead))
def test_head_architecture_roundtrip_and_legacy_manifest_compatibility(tmp_path, head_type):
    language, head = encoder(), head_type().eval()
    directory = tmp_path / "model"
    manifest = save_text_control_artifact(
        directory,
        encoder=language,
        head=head,
        calibration=calibration(),
        validation=report(),
        training={"trained": True},
    )
    assert manifest["head_architecture"] == head.architecture
    if head_type is TextControlHead:
        # Historical v1 bundles did not serialize the optional head discriminator.
        manifest.pop("head_architecture")
        (directory / "manifest.json").write_text(json.dumps(manifest))
    model = load_text_control_reward(directory, shared_encoder=language)
    assert type(model.head) is head_type
    values = torch.randn(2, 8, 384)
    for kind, expected in head(values).items():
        torch.testing.assert_close(model.head(values)[kind], expected)
