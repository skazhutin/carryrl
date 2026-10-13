"""The manual player uses actual env ticks and rejects inactive/stale input."""

import base64
import io
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from carryrl.agent.actions import PhysicalAction
from carryrl.envs.atomic import AtomicEnv
from carryrl.envs.atomic_suite import TASKS
from carryrl.play import TASK_GUIDE, ManualSession, PlayerConflict, make_handler


def _settings(task="navigation", size=96, presentation="canonical"):
    return {"task": task, "difficulty": 0.15, "size": size,
            "presentation": presentation, "max_steps": 300}


def _frame(state):
    raw = base64.b64decode(state["frame"].split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)))


@pytest.mark.parametrize("task", TASKS)
def test_manual_game_is_same_atomic_env_and_requires_explicit_start(task):
    session = ManualSession()
    settings = _settings(task)
    state = session.reset(settings)
    assert isinstance(session.env, AtomicEnv)
    assert state["ticks"] == 0 and not state["active"]
    with pytest.raises(PlayerConflict, match="паузе"):
        session.step(state["generation"], PhysicalAction().to_dict())
    active = session.control(state["generation"], True)
    stepped = session.step(active["generation"], PhysicalAction().to_dict())
    assert stepped["ticks"] == 1
    assert stepped["seconds_left"] == 29.9
    np.testing.assert_array_equal(_frame(stepped), session.env.render())


def test_pause_and_reset_invalidate_inflight_actions_without_advancing_new_game():
    session = ManualSession()
    started = session.control(session.generation, True)
    paused = session.control(started["generation"], False)
    with pytest.raises(PlayerConflict, match="изменилась"):
        session.step(started["generation"], PhysicalAction(keys=("right",)).to_dict())
    assert session.state()["ticks"] == 0
    reset = session.reset(_settings("reaction"))
    with pytest.raises(PlayerConflict):
        session.step(paused["generation"], PhysicalAction(keys=("right",)).to_dict())
    assert reset["ticks"] == 0 and reset["settings"]["task"] == "reaction"


def test_terminal_stops_simulation_and_restart_repeats_identical_level():
    session = ManualSession()
    settings = _settings("navigation")
    reset = session.reset(settings)
    initial = _frame(reset)
    started = session.control(reset["generation"], True)
    # Human-facing transport accepts the same physical actions as the pixel probe.
    from carryrl.envs.atomic import _visible_navigation_action

    state = started
    for _ in range(20):
        action = _visible_navigation_action(_frame(state), False)
        state = session.step(state["generation"], action.to_dict())
        if state["done"]:
            break
    assert state["done"] and state["outcome"] == "success" and not state["active"]
    with pytest.raises(PlayerConflict):
        session.step(state["generation"], PhysicalAction().to_dict())
    with pytest.raises(PlayerConflict, match="завершён"):
        session.control(state["generation"], True)
    repeated = session.reset(settings)
    assert repeated["ticks"] == 0
    np.testing.assert_array_equal(_frame(repeated), initial)


def test_new_level_changes_bank_entry_and_restart_preserves_selected_level():
    session = ManualSession()
    settings = _settings()
    initial_index = session.info["level_index"]
    session.reset(settings, new_level=True)
    new_index = session.info["level_index"]
    assert new_index != initial_index
    session.reset(settings)
    assert session.info["level_index"] == new_index


@pytest.mark.parametrize("size", (64, 96, 128))
@pytest.mark.parametrize("presentation", ("canonical", "varied"))
def test_native_frame_geometry_and_relative_mouse_delta_preserved(size, presentation):
    session = ManualSession()
    reset = session.reset(_settings("mouse", size, presentation))
    assert session.env.image_size == size
    if presentation == "canonical":
        assert (reset["width"], reset["height"]) == (size, size)
    else:
        assert session.env.presentation.native_size == size
        assert (reset["width"], reset["height"]) == session.env.presentation.canvas_size
    previous = tuple(session.env._cursor)
    active = session.control(reset["generation"], True)
    stepped = session.step(active["generation"], PhysicalAction(dx=2, dy=1).to_dict())
    assert session.env._cursor[0] == pytest.approx(min(size - 4, previous[0] + 2))
    assert session.env._cursor[1] == pytest.approx(min(size - 4, previous[1] + 1))
    assert _frame(stepped).shape == (reset["height"], reset["width"], 3)


@pytest.mark.parametrize("field,value", [
    ("task", "unknown"), ("difficulty", True), ("difficulty", float("nan")),
    ("difficulty", -0.1), ("size", True), ("size", 999),
    ("presentation", "unknown"), ("max_steps", 100000),
])
def test_invalid_settings_do_not_replace_environment(field, value):
    session = ManualSession()
    original = session.env
    bad = _settings()
    bad[field] = value
    with pytest.raises(ValueError):
        session.reset(bad)
    assert session.env is original


@pytest.fixture
def player_server():
    session = ManualSession()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(session, "test-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, session
    server.shutdown()
    thread.join(timeout=3)
    server.server_close()


def _http(server, method, path, record=None, *, extra_headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    headers = {"Content-Type": "application/json", "X-Carryrl-Token": "test-token"}
    headers.update(extra_headers or {})
    body = None if record is None else json.dumps(record).encode()
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    result = (json.loads(raw) if response.getheader("Content-Type", "").startswith(
        "application/json") else raw)
    status = response.status
    connection.close()
    return status, result


def test_browser_transport_reset_start_step_pause_and_all_guide_labels(player_server):
    server, _ = player_server
    status, html = _http(server, "GET", "/")
    assert status == 200
    assert set(TASK_GUIDE) == set(TASKS)
    for title, _, _ in TASK_GUIDE.values():
        assert title.encode() in html
    status, reset = _http(server, "POST", "/api/reset",
                          {"settings": _settings(), "new_level": False})
    assert status == 200 and reset["ticks"] == 0
    _, started = _http(server, "POST", "/api/control",
                       {"generation": reset["generation"], "active": True})
    status, stepped = _http(server, "POST", "/api/step", {
        "generation": started["generation"], "action": PhysicalAction().to_dict(),
    })
    assert status == 200 and stepped["ticks"] == 1
    status, paused = _http(server, "POST", "/api/control",
                           {"generation": stepped["generation"], "active": False})
    assert status == 200 and not paused["active"]
    status, _ = _http(server, "POST", "/api/step", {
        "generation": stepped["generation"], "action": PhysicalAction().to_dict(),
    })
    assert status == 409


@pytest.mark.parametrize("headers", [
    {"Origin": "https://example.com"}, {"Host": "evil.example"},
    {"X-Carryrl-Token": "wrong"},
])
def test_external_or_unauthenticated_requests_rejected(player_server, headers):
    server, session = player_server
    status, _ = _http(server, "POST", "/api/control",
                      {"generation": session.generation, "active": True}, extra_headers=headers)
    assert status == 403 and not session.active


def test_invalid_action_extra_fields_and_body_limit_rejected(player_server):
    server, session = player_server
    started = session.control(session.generation, True)
    action = PhysicalAction().to_dict()
    action["dx"] = 3
    status, _ = _http(server, "POST", "/api/step",
                      {"generation": started["generation"], "action": action})
    assert status == 400 and session.state()["ticks"] == 0
    status, _ = _http(server, "POST", "/api/control",
                      {"generation": started["generation"], "active": True, "extra": 1})
    assert status == 400
    status, _ = _http(server, "POST", "/api/reset", {"padding": "x" * 9000})
    assert status == 413
