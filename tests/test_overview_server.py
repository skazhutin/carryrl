"""The overview listener exposes only bounded read-only local routes."""

import http.client
import threading

import pytest

from carryrl.dashboard.overview import make_server


class Reader:
    def snapshot(self, *, run_id=None):
        return {'schema': 'test', 'current': {'status': 'completed'},
                'selected_run_id': 'valid' if run_id == 'valid' else None}

    def run_details(self, run_id):
        return {'run_id': run_id} if run_id == 'valid' else None

    def media(self, run_id, filename):
        return b'GIF89a' if (run_id, filename) == ('valid', 'actual.gif') else None


@pytest.fixture
def server(tmp_path):
    (tmp_path / 'index.html').write_text('<html>overview</html>')
    (tmp_path / 'overview.js').write_text('poll()')
    (tmp_path / 'overview.css').write_text('body{}')
    server = make_server(Reader(), port=0, assets=tmp_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield server
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)


def request(server, path, *, method='GET', host=None):
    client = http.client.HTTPConnection('127.0.0.1', server.server_port)
    client.request(method, path, headers={'Host': host} if host else {})
    response = client.getresponse()
    result = (response.status, dict(response.getheaders()), response.read())
    client.close()
    return result


def test_loopback_only():
    with pytest.raises(ValueError, match='only'):
        make_server(Reader(), host='0.0.0.0', port=0)


def test_routes_and_no_cors(server):
    assert request(server, '/')[2] == b'<html>overview</html>'
    assert request(server, '/assets/overview.js')[0] == 200
    status, headers, body = request(server, '/api/overview')
    assert status == 200
    assert b'completed' in body
    assert 'Access-Control-Allow-Origin' not in headers
    assert headers['Cache-Control'] == 'no-store'
    assert request(server, '/api/run/valid')[0] == 200
    assert request(server, '/api/run/missing')[0] == 404
    assert request(server, '/media/valid/actual.gif')[1]['Content-Type'] == 'image/gif'


def test_paths_host_and_writes_rejected(server):
    for path in ['/assets/', '/status.json', '/media/valid/../actual.gif',
                 '/media/valid/%2e%2e%2factual.gif', '/api/run/../valid']:
        assert request(server, path)[0] == 404
    assert request(server, '/api/overview', host='external.example')[0] == 403
    assert request(server, '/api/overview', method='POST')[0] == 501


def test_overview_run_scope_and_unknown_run(server):
    assert request(server, '/api/overview?run=valid')[0] == 200
    assert b'valid' in request(server, '/api/overview?run=valid')[2]
    assert request(server, '/api/overview?run=missing')[0] == 404
    assert request(server, '/api/overview?run=../valid')[0] == 404


def test_api_read_exception_returns_visible_json_failure(tmp_path):
    class FailedReader(Reader):
        def snapshot(self):
            raise RuntimeError('source unavailable')

    server = make_server(FailedReader(), port=0, assets=tmp_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        status, headers, body = request(server, '/api/overview')
        assert status == 500
        assert headers['Content-Type'] == 'application/json; charset=utf-8'
        assert b'RuntimeError' in body
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
