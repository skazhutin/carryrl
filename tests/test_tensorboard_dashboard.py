"""Read actual TensorBoard events to verify the sidecar's values and recovery."""

import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorboard", reason="optional monitor extra is not installed")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from carryrl.dashboard import data as dashboard_data
from carryrl.dashboard.tensorboard import TensorBoardExporter
from carryrl.envs.atomic_suite import TASKS


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def training_row(rollout=1, decisions=10):
    return {"rollout": rollout, "decisions": decisions, "cycle_wall_seconds": 3,
            "epochs": [{"policy_loss": 9}, {"policy_loss": -.25, "entropy": 12.5}],
            "episode_records": [
                {"task": "navigation", "steps": 4, "reward_sum": 3,
                 "outcome": {"known": True, "success": True, "failure": False, "score": 5}},
                {"task": "mouse", "steps": 6, "reward_sum": -1,
                 "outcome": {"known": False, "success": True}},
            ]}


@pytest.fixture
def experiment(tmp_path):
    path = tmp_path / "runs" / "pilot"
    write_json(path / "metadata.json", {"created_utc": "2026-10-08T11:29:07+00:00",
                                       "config": {"tasks": ["navigation", "mouse"],
                                                  "episodes_per_task": 1, "total_rollouts": 32,
                                                  "device": "cpu", "seed": 0}})
    write_json(path / "status.json", {"status": "running", "rollouts": 1, "decisions": 10})
    (path / "training.jsonl").write_text(json.dumps(training_row()) + "\n")
    return path, tmp_path / "events"


def accumulator(logdir):
    return EventAccumulator(str(logdir), size_guidance={"scalars": 0, "tensors": 0,
                                                       "images": 0}).Reload()


def scalar_values(events, tag):
    return [(point.step, point.value) for point in events.Scalars(tag)]


def test_real_events_decision_steps_last_epoch_unknowns_and_idempotent_restart(experiment):
    run, logdir = experiment
    writer = TensorBoardExporter(run.parent, logdir)
    assert writer.export_run("pilot") > 0
    assert writer.export_run("pilot") == 0
    writer.close()
    restarted = TensorBoardExporter(run.parent, logdir)
    assert restarted.export_run("pilot") == 0
    restarted.close()
    events = accumulator(logdir / "pilot")
    assert scalar_values(events, "PPO/policy_loss") == [(10, -.25)]
    assert scalar_values(events, "train/overall/mean_episode_reward") == [(10, 1)]
    assert scalar_values(events, "train/overall/success_rate") == [(10, .5)]
    assert scalar_values(events, "train/mouse/success_rate") == [(10, 0)]
    assert scalar_values(events, "train/mouse/unknown_episodes") == [(10, 1)]
    assert "PPO/value_loss" not in events.Tags()["scalars"]
    text = events.Tensors("monitor/status")[0].tensor_proto.string_val[0].decode()
    assert "фаза выведена из сохранённых файлов" in text
    assert "последняя эпоха" in text
    config = events.Tensors("monitor/config")[0].tensor_proto.string_val[0].decode()
    assert '"tasks"' in config
    assert (logdir / "pilot" / "state.json").is_file()
    assert not (run / "state.json").exists()


def test_preserves_per_task_history_and_partial_trailing_record(experiment):
    run, logdir = experiment
    writer = TensorBoardExporter(run.parent, logdir)
    writer.export_run("pilot")
    new_row = json.dumps(training_row(rollout=2, decisions=20))
    with (run / "training.jsonl").open("a") as stream:
        stream.write(new_row[:30])
    writer.export_run("pilot")  # File mtime can refresh status, never an unfinished PPO record.
    assert len(accumulator(logdir / "pilot").Scalars("PPO/policy_loss")) == 1
    with (run / "training.jsonl").open("a") as stream:
        stream.write(new_row[30:] + "\n")
    writer.export_run("pilot")
    writer.close()
    events = accumulator(logdir / "pilot")
    assert scalar_values(events, "train/navigation/success_rate") == [(10, 1), (20, 1)]
    assert scalar_values(events, "train/mouse/unknown_episodes") == [(10, 1), (20, 1)]


def test_evaluation_agent_and_random_use_all_episodes_and_raw_visible_score(experiment):
    run, logdir = experiment
    def stats(success, unknown, score):
        return {"episodes_started": 4, "unknown_outcomes": unknown,
                "outcomes": {"success": success}, "mean_visible_final_score": score}
    agent, random = stats(1, 2, 1.5), stats(2, 1, 3)
    agent["tasks"] = {"navigation": stats(1, 2, 1.5)}
    random["tasks"] = {"navigation": stats(2, 1, 3)}
    write_json(run / "evaluations" / "rollout_0001.json", {
        "rollout": 1, "decisions": 10, "summaries": {
            "frozen_deterministic": agent, "generic_physical_random": random}})
    writer = TensorBoardExporter(run.parent, logdir)
    writer.export_run("pilot")
    writer.close()
    events = accumulator(logdir / "pilot")
    assert scalar_values(events, "eval/overall/agent_success_rate") == [(10, .25)]
    assert scalar_values(events, "eval/overall/random_success_rate") == [(10, .5)]
    assert scalar_values(events, "eval/navigation/agent_unknown_episodes") == [(10, 2)]
    assert scalar_values(events, "eval/navigation/mean_visible_score") == [(10, 1.5)]


def test_real_image_summary_has_saved_frame_and_matching_action_caption(experiment):
    run, logdir = experiment
    folder = run / "trajectories" / "rollout_0002"
    folder.mkdir(parents=True)
    frame = np.full((8, 12, 3), (33, 66, 99), dtype=np.uint8)
    np.savez_compressed(folder / "navigation_0.npz", final_frame=frame)
    write_json(folder / "navigation_0.json", {"task": "navigation", "actions": [
        {"keys": ["w"], "mouse_buttons": [], "dx": 2, "dy": 0, "wheel": 0}]})
    writer = TensorBoardExporter(run.parent, logdir)
    writer.export_run("pilot")
    assert writer.export_run("pilot") == 0
    writer.close()
    events = accumulator(logdir / "pilot")
    image = events.Images("last_saved/navigation")[0]
    assert (image.step, image.height, image.width) == (10, 8, 12)
    with Image.open(io.BytesIO(image.encoded_image_string)) as decoded:
        assert decoded.getpixel((0, 0)) == (33, 66, 99)
    caption = events.Tensors("monitor/last_saved/navigation")[0].tensor_proto.string_val[0].decode()
    assert "Цикл 2, эпизод 1, шаг 1" in caption
    assert "last_episode" in caption
    assert '"keys": ["w"]' in caption
    assert "не прямая видеотрансляция" in caption
    assert "/api/" not in caption


def test_event_markers_recover_if_cursor_lost_without_duplicate_points(experiment):
    run, logdir = experiment
    writer = TensorBoardExporter(run.parent, logdir)
    writer.export_run("pilot")
    writer.close()
    (logdir / "pilot" / "state.json").unlink()
    restarted = TensorBoardExporter(run.parent, logdir)
    assert restarted.export_run("pilot") == 0
    restarted.close()
    assert len(accumulator(logdir / "pilot").Scalars("PPO/policy_loss")) == 1


def test_unbacked_cursor_cannot_silently_drop_source_update(experiment, caplog):
    run, logdir = experiment
    write_json(logdir / "pilot" / "state.json", {
        "sources": {"train/1": "phantom fingerprint without durable event"}})
    writer = TensorBoardExporter(run.parent, logdir)
    writer.export_run("pilot")
    writer.close()
    assert "recovering from actual events" in caplog.text
    assert scalar_values(accumulator(logdir / "pilot"), "PPO/policy_loss") == [(10, -.25)]


def test_live_phase_text_changes_without_new_training_point(experiment):
    run, logdir = experiment
    writer = TensorBoardExporter(run.parent, logdir)
    writer.export_run("pilot")
    write_json(run / "live.json", {"phase": "update", "rollout": 2,
                                  "updated_utc": "2026-10-08T12:00:00+00:00"})
    writer.export_run("pilot")
    writer.close()
    events = accumulator(logdir / "pilot")
    assert len(events.Scalars("PPO/policy_loss")) == 1
    assert len(events.Tensors("monitor/status")) == 2
    text = events.Tensors("monitor/status")[-1].tensor_proto.string_val[0].decode()
    assert "update" in text and "телеметрия процесса" in text


def test_logdir_cannot_write_to_training_artifacts(experiment):
    run, _ = experiment
    with pytest.raises(ValueError, match="outside"):
        TensorBoardExporter(run.parent, run / "events")


def test_malformed_source_is_logged_then_retried(experiment, caplog):
    run, logdir = experiment
    metadata = (run / "metadata.json").read_text()
    (run / "metadata.json").write_text("{ in-flight")
    writer = TensorBoardExporter(run.parent, logdir)
    assert writer.poll(["pilot"]) == 0
    assert "retrying" in caplog.text
    (run / "metadata.json").write_text(metadata)
    assert writer.poll(["pilot"]) > 0
    writer.close()


def test_nonfinite_values_are_not_written_as_scalar_points(experiment):
    run, logdir = experiment
    row = training_row()
    row["epochs"][-1]["value_loss"] = float("nan")
    (run / "training.jsonl").write_text(json.dumps(row) + "\n")
    writer = TensorBoardExporter(run.parent, logdir)
    writer.export_run("pilot")
    writer.close()
    assert "PPO/value_loss" not in accumulator(logdir / "pilot").Tags()["scalars"]


def test_second_exporter_cannot_duplicate_same_running_export(experiment):
    run, logdir = experiment
    first = TensorBoardExporter(run.parent, logdir)
    second = TensorBoardExporter(run.parent, logdir)
    first.export_run("pilot")
    with pytest.raises(BlockingIOError):
        second.export_run("pilot")
    first.close()
    assert second.export_run("pilot") == 0
    second.close()


def test_many_task_previews_do_not_skip_bounded_training_history(experiment, monkeypatch):
    run, logdir = experiment
    metadata = json.loads((run / "metadata.json").read_text())
    metadata["config"]["tasks"] = list(TASKS)
    write_json(run / "metadata.json", metadata)
    monkeypatch.setattr(dashboard_data, "_MAX_RECORDS_PER_REFRESH", 2)
    monkeypatch.setattr(dashboard_data, "_MAX_CACHED_TRAINING_ROWS", 4)
    rows = [training_row(rollout=index, decisions=index * 10) for index in range(1, 21)]
    (run / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    writer = TensorBoardExporter(run.parent, logdir)
    for _ in range(11):
        writer.export_run("pilot")
    writer.close()
    events = accumulator(logdir / "pilot")
    assert [point.step for point in events.Scalars("PPO/policy_loss")] == list(range(10, 201, 10))


def test_per_run_symlink_cannot_redirect_events_into_training_directory(experiment):
    run, _ = experiment
    logdir = run.parent.parent
    (logdir / "pilot").symlink_to(run, target_is_directory=True)
    before = {path.name for path in run.iterdir()}
    writer = TensorBoardExporter(run.parent, logdir)
    with pytest.raises(ValueError, match="outside the training runs"):
        writer.export_run("pilot")
    writer.close()
    assert {path.name for path in run.iterdir()} == before
    assert not list(run.glob("events.out.tfevents.*"))
    assert not (run / "state.json").exists()
    assert not (run / ".export.lock").exists()


def test_extended_evaluation_metrics_and_success_only_steps(experiment):
    run, logdir = experiment
    agent = {"episodes_started": 4, "unknown_outcomes": 1, "outcomes": {
        "success": 1, "failure": 1}, "time_limit_episodes": 2, "mean_visible_final_score": 1.5}
    random = {"episodes_started": 4, "unknown_outcomes": 0, "outcomes": {
        "success": 2, "failure": 0}, "time_limit_episodes": 2, "mean_visible_final_score": 3}
    agent["tasks"], random["tasks"] = {"navigation": dict(agent)}, {"navigation": dict(random)}
    records = [{"policy": "frozen_deterministic", "task": "navigation", "steps": 6,
                "outcome": {"known": True, "success": True}},
               {"policy": "frozen_deterministic", "task": "navigation", "steps": 1,
                "outcome": {"known": True, "failure": True}},
               {"policy": "generic_physical_random", "task": "navigation", "steps": 3,
                "outcome": {"known": True, "success": True}}]
    write_json(run / "evaluations" / "rollout_0001.json", {
        "rollout": 1, "decisions": 10, "records": records, "summaries": {
            "frozen_deterministic": agent, "generic_physical_random": random}})
    writer = TensorBoardExporter(run.parent, logdir)
    writer.export_run("pilot")
    assert writer.export_run("pilot") == 0
    writer.close()
    events = accumulator(logdir / "pilot")
    for prefix in ("eval/overall", "eval/navigation"):
        assert scalar_values(events, prefix + "/success_gap") == [(10, -.25)]
        assert scalar_values(events, prefix + "/agent_mean_visible_score") == [(10, 1.5)]
        assert scalar_values(events, prefix + "/random_mean_visible_score") == [(10, 3)]
        assert scalar_values(events, prefix + "/agent_failure_rate") == [(10, .25)]
        assert scalar_values(events, prefix + "/agent_timeout_rate") == [(10, .5)]
        assert scalar_values(events, prefix + "/agent_mean_success_steps") == [(10, 6)]
    assert len(events.Scalars("eval/navigation/agent_success_rate")) == 1


def test_finished_status_and_missing_success_length_are_truthful(experiment):
    run, logdir = experiment
    write_json(run / "status.json", {"status": "resource_stopped", "rollouts": 1, "decisions": 10})
    agent = {"episodes_started": 4, "outcomes": {"success": 0},
             "tasks": {"navigation": {"episodes_started": 4, "outcomes": {"success": 0}}}}
    write_json(run / "evaluations" / "rollout_0001.json", {
        "rollout": 1, "decisions": 10, "summaries": {"frozen_deterministic": agent}})
    writer = TensorBoardExporter(run.parent, logdir)
    writer.export_run("pilot")
    writer.close()
    events = accumulator(logdir / "pilot")
    assert "eval/overall/agent_mean_success_steps" not in events.Tags()["scalars"]
    assert "eval/overall/random_mean_visible_score" not in events.Tags()["scalars"]
    text = events.Tensors("monitor/status")[-1].tensor_proto.string_val[0].decode()
    assert "Завершено циклов" in text and "Текущего собираемого эпизода нет" in text
    assert "Завершённых эпизодов цикла" not in text
    guide = events.Tensors("monitor/how_to_read_metrics")[-1].tensor_proto.string_val[0].decode()
    assert "более раннюю смерть" in guide
    assert "25 процентных пунктов" in guide
    assert "PPO" in guide and "random_success_rate" in guide


def test_gif_image_summary_is_training_revision_and_idempotent(experiment):
    run, logdir = experiment
    folder = run / "trajectories" / "rollout_0002"
    folder.mkdir(parents=True)
    path = folder / "navigation_0.npz"
    np.savez_compressed(path, frames=np.zeros((2, 8, 12, 3), dtype=np.uint8),
                        final_frame=np.ones((8, 12, 3), dtype=np.uint8))
    write_json(path.with_suffix(".json"), {"task": "navigation", "seed": 8,
        "collection_weights_revision": 20,
        "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "actions": [{"keys": [key], "mouse_buttons": [], "dx": 0, "dy": 0, "wheel": 0}
                    for key in ("w", "d")]})
    write_json(run / "checkpoints" / "rollout_0001" / "manifest.json", {
        "rollout": 1, "decisions": 10, "weights_revision": 22})
    writer = TensorBoardExporter(run.parent, logdir)
    writer.export_run("pilot")
    assert writer.export_run("pilot") == 0
    writer.close()
    events = accumulator(logdir / "pilot")
    image = events.Images("playback/navigation")[0]
    assert image.step == 10
    with Image.open(io.BytesIO(image.encoded_image_string)) as gif:
        assert gif.n_frames == 3
        assert gif.info["duration"] == 100
    caption = events.Tensors("monitor/playback/navigation")[0].tensor_proto.string_val[0].decode()
    assert "ревизии **20**" in caption and "ревизию **22**" in caption
    assert "до действия t" in caption and "не видео итоговой" in caption


def test_seeded_stochastic_evaluation_uses_separate_tags_and_preserves_greedy(experiment):
    run, logdir = experiment
    metadata = json.loads((run / "metadata.json").read_text())
    metadata["config"].update({"eval_stochastic": True, "eval_seed": 80001})
    write_json(run / "metadata.json", metadata)
    def stats(successes, score):
        return {"episodes_started": 4, "unknown_outcomes": 0,
                "outcomes": {"success": successes, "failure": 0},
                "mean_visible_final_score": score, "time_limit_episodes": 4 - successes}
    greedy, stochastic, random = stats(0, 0), stats(2, 3), stats(1, 1)
    for summary in (greedy, stochastic, random):
        summary["tasks"] = {"navigation": dict(summary)}
    path = run / "evaluations" / "rollout_0001.json"
    data = {"rollout": 1, "decisions": 10, "evaluation_seed": 80001,
            "level_selection": "explicit covering level indices", "summaries": {
                "frozen_deterministic": greedy, "generic_physical_random": random}}
    write_json(path, data)
    first = TensorBoardExporter(run.parent, logdir)
    first.export_run("pilot")
    first.close()
    data["summaries"]["frozen_stochastic"] = stochastic
    data["records"] = [{"task": "navigation", "policy": "frozen_stochastic", "steps": 6,
                        "outcome": {"known": True, "success": True}}]
    write_json(path, data)
    second = TensorBoardExporter(run.parent, logdir)
    second.export_run("pilot")
    assert second.export_run("pilot") == 0
    second.close()
    events = accumulator(logdir / "pilot")
    assert scalar_values(events, "eval/navigation/agent_success_rate") == [(10, 0)]
    assert scalar_values(events, "eval_stochastic/navigation/agent_success_rate") == [(10, .5)]
    assert scalar_values(events, "eval_stochastic/navigation/random_success_rate") == [(10, .25)]
    assert scalar_values(events, "eval_stochastic/navigation/success_gap") == [(10, .25)]
    assert scalar_values(events, "eval_stochastic/navigation/agent_mean_success_steps") == [(10, 6)]
    guide = events.Tensors("monitor/how_to_read_metrics")[-1].tensor_proto.string_val[0].decode()
    assert "eval_stochastic/..." in guide and "80001" in guide
    protocol = events.Tensors("monitor/eval_stochastic")[-1].tensor_proto.string_val[0].decode()
    assert "80001" in protocol and "explicit covering level indices" in protocol


def test_optimizer_diagnostics_upgrade_preserves_old_points(experiment):
    run, logdir = experiment
    first = TensorBoardExporter(run.parent, logdir)
    first.export_run("pilot")
    first.close()
    row = training_row()
    row.update(optimizer_steps=6, kl_early_stop=True, curriculum_difficulty=.15,
               task_transitions={"navigation": 4, "mouse": 6})
    row["epochs"][-1]["clip_fraction"] = .25
    (run / "training.jsonl").write_text(json.dumps(row) + "\n")
    second = TensorBoardExporter(run.parent, logdir)
    second.export_run("pilot")
    assert second.export_run("pilot") == 0
    second.close()
    events = accumulator(logdir / "pilot")
    assert scalar_values(events, "PPO/policy_loss") == [(10, -.25)]
    assert scalar_values(events, "PPO/optimizer_steps") == [(10, 6)]
    assert scalar_values(events, "PPO/clip_fraction") == [(10, .25)]
    assert scalar_values(events, "PPO/kl_early_stop") == [(10, 1)]
    assert scalar_values(events, "curriculum/difficulty")[0][1] == pytest.approx(.15)
    assert scalar_values(events, "train/navigation/valid_transitions") == [(10, 4)]


def test_versioned_reward_partitions_distinguish_synthetic_glyphs_and_played_frames(experiment):
    run, logdir = experiment
    metadata = json.loads((run / "metadata.json").read_text())
    metadata["config"]["reward_model_mode"] = "screen_v2"
    write_json(run / "metadata.json", metadata)
    def partition(index):
        return {"frames": 100 + index, "episodes": 2, "score_mae": .01 * index,
                "max_score_error": .03 * index, "synthetic_glyph_frames": 18,
                "played_episode_frames": 82 + index, "success": {
                    "positive_frames": 6, "negative_frames": 94, "recall": 1,
                    "false_positive_rate": 0}, "failure": {"recall": None,
                                                             "false_positive_rate": 0}}
    report = {"model_version": "carryrl.screen_reward.v3", "scope": "supported RGB HUD",
              "supervision": "independent exact RGB verification", "inputs": "RGB only",
              "partitions": {name: partition(index) for index, name in enumerate(
                  ("train", "familiar_validation", "unseen_validation", "canonical_validation"))},
              "gate": {"passed": True, "failures": []}, "synthetic_glyph_coverage": {
                  "training_frames": 18, "validation_frames": 54,
                  "required_combinations": ["0:Play", "1:Success"], "not_played_episodes": True}}
    write_json(run / "reward_bootstrap.json", report)
    writer = TensorBoardExporter(run.parent, logdir)
    writer.export_run("pilot")
    assert writer.export_run("pilot") == 0
    writer.close()
    events = accumulator(logdir / "pilot")
    assert scalar_values(events, "reward_screen/gate_passed") == [(0, 1)]
    assert scalar_values(events, "reward_screen/unseen_validation/frames") == [(0, 102)]
    assert scalar_values(events, "reward_screen/train/synthetic_glyph_frames") == [(0, 18)]
    assert scalar_values(events, "reward_screen/train/played_episode_frames") == [(0, 82)]
    assert "reward_screen/train/failure_recall" not in events.Tags()["scalars"]
    assert scalar_values(events, "reward_screen/train/failure_false_positive_rate") == [(0, 0)]
    scope = events.Tensors("monitor/reward_scope")[-1].tensor_proto.string_val[0].decode()
    assert "carryrl.screen_reward.v3" in scope and "screen_v2" in scope
    assert "не считаются сыгранными эпизодами" in scope
    assert "не автономное открытие награды" in scope


def test_stage_initial_and_evaluation_difficulty_do_not_relabel_warm_start(experiment):
    run, logdir = experiment
    agent = {"episodes_started": 4, "outcomes": {"success": 0}, "tasks": {}}
    write_json(run / "evaluations" / "rollout_0000.json", {
        "rollout": 0, "decisions": 100, "baseline_kind": "stage_initial",
        "baseline_policy_previously_trained": True, "evaluation_difficulty": .15,
        "summaries": {"frozen_deterministic": agent}})
    writer = TensorBoardExporter(run.parent, logdir)
    writer.export_run("pilot")
    writer.close()
    events = accumulator(logdir / "pilot")
    protocol = events.Tensors("monitor/evaluation_protocol")[-1].tensor_proto.string_val[0].decode()
    assert "stage_initial" in protocol and "0.15" in protocol
    status = events.Tensors("monitor/status")[-1].tensor_proto.string_val[0].decode()
    assert "stage_initial" in status and "уже обучалась" in status
    assert "Сложность проверки:** 0.15" in status
    assert scalar_values(events, "eval/overall/agent_success_rate") == [(100, 0)]
