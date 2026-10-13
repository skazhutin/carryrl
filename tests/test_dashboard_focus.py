"""Read TensorBoard events to verify current-run provenance and epoch metrics."""

import json

import pytest

pytest.importorskip("tensorboard")
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from carryrl.dashboard.data import TASK_LABELS
from carryrl.dashboard.focus import (
    GUIDE_TAG,
    NLL_TAG,
    QUICK_WIN_TAG,
    RUN_LABEL,
    WIN_TAG,
    FocusExporter,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def source(tmp_path):
    run = tmp_path / "runs" / "warmup"
    write(run / "metadata.json", {"kind": "imitation_training", "teacher": "scripted RGB"})
    write(run / "config.json", {"tasks": list(TASK_LABELS), "training_backend": "mlx"})
    write(run / "status.json", {"status": "running", "phase": "encoding", "epochs": 0})
    registry = tmp_path / "active.json"
    write(registry, {"run": str(run)})
    return run, registry, tmp_path / "events"


def read(logdir):
    return EventAccumulator(str(logdir / RUN_LABEL),
                            size_guidance={"scalars": 0, "tensors": 0}).Reload()


def evaluation(epoch=1):
    summary = {"episodes_started": 4, "outcomes": {"success": 1},
               "time_limit_episodes": 2, "unknown_outcomes": 1,
               "tasks": {"navigation": {"episodes_started": 2,
                                          "outcomes": {"success": 1}}}}
    return {"epoch": epoch, "split": "validation", "policy_and_reward_frozen": True,
            "evaluation_seed": 17, "level_selection": "explicit covering level indices",
            "summaries": {"frozen_deterministic": summary,
            "generic_physical_random": {**summary, "outcomes": {"success": 0}}}}


def test_preparation_has_no_fabricated_wins_and_russian_provenance(source):
    run, registry, logdir = source
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        assert exporter.poll() == 1
        assert exporter.poll() == 0
    finally:
        exporter.close()
    events = read(logdir)
    assert events.Tags()["scalars"] == []
    guide = events.Tensors(GUIDE_TAG)[-1].tensor_proto.string_val[0].decode()
    assert "Подготовка кадров учителя" in guide
    assert "не RL" in guide and "Первая проверка ещё не завершена" in guide
    assert str(run) in guide


def test_actual_percentages_epoch_axis_and_restart_idempotence(source):
    run, registry, logdir = source
    row = {"epoch": 1, "complete": True, "nll": 2.75, "optimizer_steps": 32}
    (run / "training.jsonl").write_text(json.dumps(row) + "\n")
    write(run / "evaluations" / "epoch_0001.json", evaluation())
    write(run / "status.json", {"status": "running", "phase": "imitation",
                               "epochs": 1, "optimizer_steps": 32})
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        assert exporter.poll() == 3
        assert exporter.poll() == 0
    finally:
        exporter.close()
    # Actual event markers recover idempotence even if the advisory cursor vanished.
    (logdir / RUN_LABEL / "state.json").unlink()
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        assert exporter.poll() == 0
    finally:
        exporter.close()
    events = read(logdir)
    assert [(point.step, point.value) for point in events.Scalars(WIN_TAG)] == [(1, 25)]
    assert [(point.step, point.value) for point in events.Scalars(NLL_TAG)] == [(1, 2.75)]
    assert events.Scalars("01 Качество игры/02 Победы — Навигация (%)")[0].value == 50
    assert events.Scalars("01 Качество игры/03 Истёк лимит времени")[0].value == 2
    assert events.Scalars("04 Случайный контроль/01 Победы — все игры (%)")[0].value == 0


def test_none_nll_incomplete_epoch_and_partial_append(source):
    run, registry, logdir = source
    row = {"epoch": 1, "complete": False, "nll": None, "optimizer_steps": 0}
    path = run / "training.jsonl"
    path.write_text(json.dumps(row) + '\n{"epoch": 2')
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        exporter.poll()
        assert exporter.poll() == 0
    finally:
        exporter.close()
    events = read(logdir)
    assert NLL_TAG not in events.Tags()["scalars"]
    assert events.Scalars("02 Обучение/Эпоха завершена")[0].value == 0
    assert "неполная" in events.Tensors(GUIDE_TAG)[-1].tensor_proto.string_val[0].decode()


def test_unfrozen_evaluation_is_not_exported(source):
    run, registry, logdir = source
    value = evaluation()
    value["policy_and_reward_frozen"] = False
    write(run / "evaluations" / "epoch_0001.json", value)
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        exporter.poll()
    finally:
        exporter.close()
    assert WIN_TAG not in read(logdir).Tags()["scalars"]


def test_images_show_latest_frozen_policy_from_current_run_only(source):
    import io
    from types import SimpleNamespace

    from PIL import Image

    run, registry, logdir = source
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(output, format="GIF")
    item = {"run_id": run.name, "task": "ui", "label": "Инвентарь", "epoch": 2,
            "revision": 7, "steps": 4, "url": f"/media/{run.name}/current.gif"}
    items = [{**item, "run_id": "previous_run", "revision": 99},
             {**item, "revision": 3, "epoch": 1}, item]
    exporter = FocusExporter(run.parent, logdir, registry)
    exporter._overview = SimpleNamespace(
        snapshot=lambda **_: {"playbacks": {"items": items}},
        media=lambda *_: output.getvalue(),
    )
    try:
        exporter.poll()
        assert exporter.poll() == 0
    finally:
        exporter.close()
    events = read(logdir)
    images = events.Images("playback/ui")
    assert len(images) == 1 and images[0].step == 2
    caption = events.Tensors("06 Записи игры/Инвентарь")[-1].tensor_proto.string_val[0].decode()
    assert "версия весов 7" in caption and "не прямой эфир" in caption


def test_registry_switch_archives_instead_of_merging_epoch_one(source):
    run, registry, logdir = source
    write(run / "evaluations" / "epoch_0001.json", evaluation())
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        exporter.poll()
        second = run.parent / "new_warmup"
        write(second / "metadata.json", {"kind": "imitation_training", "teacher": "different"})
        write(second / "config.json", {"tasks": list(TASK_LABELS), "training_backend": "mlx"})
        write(second / "status.json", {"epochs": 1})
        updated = evaluation()
        updated["summaries"]["frozen_deterministic"]["outcomes"]["success"] = 3
        write(second / "evaluations" / "epoch_0001.json", updated)
        write(registry, {"run": str(second)})
        exporter.poll()
    finally:
        exporter.close()
    assert [point.value for point in read(logdir).Scalars(WIN_TAG)] == [75]
    archives = list(logdir.with_name(logdir.name + "_archive").iterdir())
    assert len(archives) == 1
    archived = EventAccumulator(str(archives[0])).Reload()
    assert archived.Scalars(WIN_TAG)[0].value == 25


@pytest.mark.parametrize("bad_kind", ["ppo_training", None])
def test_registry_rejects_unrelated_run(source, bad_kind):
    run, registry, logdir = source
    write(run / "metadata.json", {"kind": bad_kind})
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        with pytest.raises(ValueError, match="imitation_training"):
            exporter.poll()
    finally:
        exporter.close()
    assert not (logdir / RUN_LABEL).exists()


def test_registry_rejects_external_and_symlink_artifacts(source, tmp_path):
    run, registry, logdir = source
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        write(registry, {"run": str(tmp_path)})
        with pytest.raises(ValueError, match="within --runs"):
            exporter.poll()
        write(registry, {"run": str(run)})
        outside = tmp_path / "metadata.json"
        write(outside, {"kind": "imitation_training"})
        (run / "metadata.json").unlink()
        (run / "metadata.json").symlink_to(outside)
        with pytest.raises(ValueError, match="escapes"):
            exporter.poll()
    finally:
        exporter.close()


def test_initial_evaluation_phase_and_archive_link(source):
    run, registry, logdir = source
    write(run / "status.json", {"status": "running", "phase": "initial_evaluation", "epochs": 0})
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        exporter.poll()
    finally:
        exporter.close()
    guide = read(logdir).Tensors(GUIDE_TAG)[-1].tensor_proto.string_val[0].decode()
    assert "Первая проверка модели до обучения" in guide
    assert "http://127.0.0.1:6008/" in guide


def test_ppo_stage_rollout_axis_last_epoch_metrics_and_frozen_wins(source):
    run, registry, logdir = source
    write(run / "metadata.json", {"kind": "sequence_training", "created_utc": "fixed"})
    write(run / "status.json", {"status": "running", "rollouts": 2, "decisions": 80})
    row = {"rollout": 2, "decisions": 80, "optimizer_steps": 6,
           "epochs": [{"entropy": 9, "approx_kl": .8},
                      {"entropy": 3, "approx_kl": .02, "policy_loss": -.4, "value_loss": .7}]}
    (run / "training.jsonl").write_text(json.dumps(row) + "\n")
    value = evaluation()
    value.pop("epoch")
    value["rollout"] = 2
    write(run / "evaluations" / "rollout_0002.json", value)
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        assert exporter.poll() == 3
        assert exporter.poll() == 0
        write(run / "status.json", {"status": "completed", "rollouts": 2, "decisions": 80})
        assert exporter.poll() == 1
    finally:
        exporter.close()
    events = read(logdir)
    assert [(p.step, p.value) for p in events.Scalars(WIN_TAG)] == [(2, 25)]
    assert events.Scalars("02 Обучение PPO/Энтропия действий")[0].value == 3
    assert events.Scalars("02 Обучение PPO/Изменение политики (KL)")[0].value == pytest.approx(.02)
    assert events.Scalars("02 Обучение PPO/Обновлений весов за цикл")[0].value == 6
    assert NLL_TAG not in events.Tags()["scalars"]
    assert not logdir.with_name(logdir.name + "_archive").exists()
    guide = events.Tensors(GUIDE_TAG)[-1].tensor_proto.string_val[0].decode()
    assert "номер цикла PPO (rollout)" in guide
    assert "обучение с подкреплением PPO" in guide
    assert "повторению действий" not in guide


def test_bc_to_ppo_switch_clears_prior_nll_and_initial_wins(source):
    run, registry, logdir = source
    write(run / "evaluations" / "epoch_0001.json", evaluation())
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        exporter.poll()
        second = run.parent / "ppo"
        write(second / "metadata.json", {"kind": "sequence_training"})
        write(second / "config.json", {"tasks": list(TASK_LABELS), "training_backend": "mlx"})
        write(second / "status.json", {"status": "running", "rollouts": 0})
        write(registry, {"run": str(second)})
        exporter.poll()
    finally:
        exporter.close()
    events = read(logdir)
    assert events.Tags()["scalars"] == []
    guide = events.Tensors(GUIDE_TAG)[-1].tensor_proto.string_val[0].decode()
    assert "Первая проверка ещё не завершена" in guide
    assert "PPO" in guide


def test_full_and_quick_cohorts_have_separate_win_curves(source):
    run, registry, logdir = source
    full = evaluation(epoch=0)
    quick = evaluation(epoch=10)
    quick["level_selection"] = "legacy seeded sampling"
    quick["summaries"]["frozen_deterministic"]["episodes_started"] = 32
    quick["summaries"]["frozen_deterministic"]["outcomes"]["success"] = 3
    write(run / "evaluations" / "epoch_0000.json", full)
    write(run / "evaluations" / "epoch_0010.json", quick)
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        exporter.poll()
        assert exporter.poll() == 0
        events = read(logdir)
        assert [(p.step, p.value) for p in events.Scalars(WIN_TAG)] == [(0, 25)]
        assert [(p.step, p.value) for p in events.Scalars(QUICK_WIN_TAG)] == [(10, 9.375)]
        guide = events.Tensors(GUIDE_TAG)[-1].tensor_proto.string_val[0].decode()
        assert "Последняя полная проверка: эпоха 0" in guide
        assert "Последняя быстрая проверка: эпоха 10" in guide
        # The trainer may replace a quick last-epoch file with a full final check.
        # Existing quick points remain a truthful history, while full points use their own curve.
        write(run / "evaluations" / "epoch_0010.json", evaluation(epoch=10))
        exporter.poll()
    finally:
        exporter.close()
    events = read(logdir)
    assert [p.step for p in events.Scalars(WIN_TAG)] == [0, 10]
    assert [p.step for p in events.Scalars(QUICK_WIN_TAG)] == [10]
    guide = events.Tensors(GUIDE_TAG)[-1].tensor_proto.string_val[0].decode()
    assert "Последняя полная проверка: эпоха 10" in guide
    assert "Последняя быстрая проверка: эпоха 10" in guide
    restarted = FocusExporter(run.parent, logdir, registry)
    try:
        assert restarted.poll() == 0
    finally:
        restarted.close()
    quality_tags = sorted(tag for tag in events.Tags()["scalars"]
                          if tag.startswith("01 Качество игры/"))
    assert quality_tags[0] == WIN_TAG


@pytest.mark.parametrize("changed", [
    {"evaluation_seed": 18},
    {"evaluation_difficulty": .5},
    {"level_bank_seed": 42},
    {"protocol_sha256": "different-protocol"},
    {"records": [{"task": "navigation", "presentation": {"theme": "dark"}}]},
])
def test_full_checks_with_changed_conditions_do_not_join_canonical_curve(source, changed):
    run, registry, logdir = source
    write(run / "evaluations" / "epoch_0000.json", evaluation(epoch=0))
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        exporter.poll()
        write(run / "evaluations" / "epoch_0002.json", {**evaluation(epoch=2), **changed})
        exporter.poll()
    finally:
        exporter.close()
    restarted = FocusExporter(run.parent, logdir, registry)
    try:
        assert restarted.poll() == 0
    finally:
        restarted.close()
    events = read(logdir)
    assert [point.step for point in events.Scalars(WIN_TAG)] == [0]
    scoped = [tag for tag in events.Tags()["scalars"]
              if tag.startswith("01 Качество игры/") and tag != WIN_TAG
              and tag.endswith("/01 Победы — все игры (%)")]
    assert len(scoped) == 1
    assert [point.step for point in events.Scalars(scoped[0])] == [2]


def test_varied_full_checks_are_labeled_and_share_only_their_own_cohort(source):
    run, registry, logdir = source
    write(run / "config.json", {"tasks": list(TASK_LABELS), "training_backend": "mlx",
                                "visual_config": {"mode": "varied"}})
    for epoch in (0, 2):
        write(run / "evaluations" / f"epoch_{epoch:04}.json", evaluation(epoch))
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        exporter.poll()
        assert exporter.poll() == 0
    finally:
        exporter.close()
    events = read(logdir)
    assert WIN_TAG not in events.Tags()["scalars"]
    scoped = [tag for tag in events.Tags()["scalars"]
              if tag.startswith("01 Качество игры/Вариативные условия")
              and tag.endswith("/01 Победы — все игры (%)")]
    assert len(scoped) == 1
    assert [point.step for point in events.Scalars(scoped[0])] == [0, 2]


def test_ppo_cumulative_updates_use_journal_counts_and_local_cycle_target(source):
    run, registry, logdir = source
    write(run / "metadata.json", {"kind": "sequence_training"})
    write(run / "config.json", {"tasks": list(TASK_LABELS), "training_backend": "mlx",
                                "total_rollouts": 32})
    write(run / "status.json", {"status": "running", "phase": "optimizing", "rollouts": 2,
                                "weights_revision": 900, "optimizer_steps": 900})
    rows = [{"rollout": 1, "optimizer_steps": 4, "epochs": [{"approx_kl": .02}]},
            {"rollout": 2, "optimizer_steps": 1, "epochs": [{"approx_kl": .04}],
             "kl_early_stop": True}]
    (run / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        exporter.poll()
        assert exporter.poll() == 0
    finally:
        exporter.close()
    events = read(logdir)
    assert [(p.step, p.value) for p in events.Scalars(
        "02 Обучение PPO/Обновлений весов за этот этап")] == [(1, 4), (2, 5)]
    guide = events.Tensors(GUIDE_TAG)[-1].tensor_proto.string_val[0].decode()
    assert "Обновление весов PPO" in guide
    assert "Записано циклов PPO: 2 из 32" in guide
    assert "обновлений весов за этот этап: 5" in guide
    assert "900" not in guide


def test_missing_ppo_update_count_stays_unknown(source):
    run, registry, logdir = source
    write(run / "metadata.json", {"kind": "sequence_training"})
    write(run / "status.json", {"rollouts": 2, "weights_revision": 900,
                                "phase": "collecting"})
    rows = [{"rollout": 1}, {"rollout": 2, "optimizer_steps": 4}]
    (run / "training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        exporter.poll()
    finally:
        exporter.close()
    events = read(logdir)
    assert "02 Обучение PPO/Обновлений весов за этот этап" not in events.Tags()["scalars"]
    guide = events.Tensors(GUIDE_TAG)[-1].tensor_proto.string_val[0].decode()
    assert "обновлений весов за этот этап: неизвестно" in guide
    assert "Сбор собственных игровых эпизодов PPO" in guide


def test_cached_tensorboard_needs_a_fresh_accumulator_on_source_switch(source):
    from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer

    run, registry, logdir = source
    write(run / "evaluations" / "epoch_0020.json", evaluation(epoch=20))
    exporter = FocusExporter(run.parent, logdir, registry)
    try:
        exporter.poll()
        cached = EventMultiplexer().AddRunsFromDirectory(str(logdir)).Reload()
        assert cached.Scalars(RUN_LABEL, WIN_TAG)[-1].step == 20
        second = run.parent / "second"
        write(second / "metadata.json", {"kind": "imitation_training"})
        write(second / "config.json", {"tasks": list(TASK_LABELS), "training_backend": "mlx"})
        write(second / "status.json", {"epochs": 0, "phase": "initial_evaluation"})
        write(registry, {"run": str(second)})
        exporter.poll()
        cached.AddRunsFromDirectory(str(logdir)).Reload()
        # Renaming logs does not discard previously loaded scalar reservoirs.
        assert cached.Scalars(RUN_LABEL, WIN_TAG)[-1].step == 20
        fresh = EventMultiplexer().AddRunsFromDirectory(str(logdir)).Reload()
        assert WIN_TAG not in fresh.Runs()[RUN_LABEL]["scalars"]
    finally:
        exporter.close()


def test_supervised_generation_updates_only_after_successful_export(source, monkeypatch):
    import carryrl.dashboard.focus as focus_module

    run, registry, logdir = source
    calls = []

    class Server:
        def __init__(self, logdir, port):
            assert port == 6006

        def ensure_generation(self, identity):
            cursor = json.loads((logdir / RUN_LABEL / "state.json").read_text())
            assert cursor["run_id"] == json.loads(registry.read_text())["run"].split("/")[-1]
            assert cursor["sources"]
            calls.append(identity)

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(focus_module, "FocusServer", Server)
    exporter = FocusExporter(run.parent, logdir, registry, serve_port=6006)
    try:
        exporter.poll()
        exporter.poll()
        assert len(calls) == 2 and calls[0] == calls[1]
        write(run / "status.json", {"epochs": 0, "phase": "imitation"})
        exporter.poll()
        assert calls[-1] == calls[0]
        second = run.parent / "second"
        write(second / "metadata.json", {"kind": "imitation_training"})
        write(second / "config.json", {"tasks": list(TASK_LABELS), "training_backend": "mlx"})
        write(second / "status.json", {"epochs": 0})
        write(registry, {"run": str(second)})
        exporter.poll()
        assert len(calls) == 4 and calls[-1] != calls[0]
        write(registry, {"run": str(run.parent / "missing")})
        with pytest.raises(FileNotFoundError):
            exporter.poll()
        assert len(calls) == 4
    finally:
        exporter.close()
    assert calls[-1] == "closed"


def test_server_stops_owned_child_before_restarting_and_bounds_stop(tmp_path, monkeypatch):
    import subprocess

    from carryrl.dashboard.focus_server import FocusServer

    actions = []

    class Child:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            actions.append("terminate")

        def wait(self, timeout):
            actions.append(("wait", timeout))
            if actions.count(("wait", timeout)) == 1:
                raise subprocess.TimeoutExpired("tensorboard", timeout)
            self.returncode = 0

        def kill(self):
            actions.append("kill")

    server = FocusServer(tmp_path, 6006)

    def start():
        actions.append("start")
        server.process = Child()

    monkeypatch.setattr(server, "_start", start)
    server.ensure_generation("source1")
    server.ensure_generation("source1")
    assert actions == ["start"]
    server.ensure_generation("source2")
    assert actions == ["start", "terminate", ("wait", 5), "kill", ("wait", 5), "start"]
    server.close()
    assert server.process is None


def test_server_occupied_port_does_not_spawn_or_stop_another_process(tmp_path, monkeypatch):
    import socket

    from carryrl.dashboard import focus_server

    spawned = []
    monkeypatch.setattr(focus_server.subprocess, "Popen", lambda *a, **k: spawned.append(True))
    with socket.socket() as owner:
        owner.bind(("127.0.0.1", 0))
        owner.listen()
        port = owner.getsockname()[1]
        server = focus_server.FocusServer(tmp_path, port)
        with pytest.raises(focus_server.FocusServerError, match="No existing process was stopped"):
            server.ensure_generation("source1")
        assert spawned == []
        assert owner.getsockname()[1] == port
        server.close()


def test_server_command_keeps_text_sampling_and_writes_separate_log(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from carryrl.dashboard import focus_server

    child = MagicMock()
    child.poll.return_value = None
    spawn = MagicMock(return_value=child)
    monkeypatch.setattr(focus_server.subprocess, "Popen", spawn)
    monkeypatch.setattr(focus_server.socket, "socket", MagicMock())
    monkeypatch.setattr(focus_server.socket, "create_connection", MagicMock())
    server = focus_server.FocusServer(tmp_path, 6006)
    try:
        server.ensure_generation("identity")
        command = spawn.call_args.args[0]
        assert command[:3] == [focus_server.sys.executable, "-m", "tensorboard.main"]
        assert command[command.index("--host") + 1] == "127.0.0.1"
        assert command[command.index("--samples_per_plugin") + 1] == "text=1,images=4"
        assert command[command.index("--reload_interval") + 1] == "5"
        assert server.log_path.parent == tmp_path.parent
    finally:
        server.close()
