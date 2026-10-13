"""Serve the local, read-only training overview on loopback."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from carryrl.dashboard.overview_data import NAME, OverviewReader

ASSETS = Path(__file__).with_name('assets')


def make_server(reader, host='127.0.0.1', port=6009, *, assets=ASSETS):
    if host not in {'127.0.0.1', 'localhost'}:
        raise ValueError('The overview server binds only to 127.0.0.1 or localhost')
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError('Invalid overview port')
    assets = Path(assets)

    class Handler(BaseHTTPRequestHandler):
        def _respond(self, content, content_type, status=200):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            try:
                self._get()
            except Exception as error:
                message = {'error': 'Не удалось прочитать данные обзора.',
                           'warnings': [f'{type(error).__name__}: {str(error)[:500]}']}
                self._respond(json.dumps(message, ensure_ascii=False).encode(),
                              'application/json; charset=utf-8', 500)

        def _get(self):
            host_header = self.headers.get('Host', '').split(':', 1)[0].lower()
            if host_header not in {'127.0.0.1', 'localhost'}:
                self._respond(b'Forbidden host', 'text/plain', 403)
                return
            route = urlsplit(self.path)
            path = unquote(route.path)
            if path == '/api/overview':
                query = parse_qs(route.query)
                if 'run' in query:
                    run_id = query['run'][0]
                    if not NAME.fullmatch(run_id):
                        self._respond(b'Unknown run', 'text/plain; charset=utf-8', 404)
                        return
                    data = reader.snapshot(run_id=run_id)
                    if data.get('selected_run_id') is None:
                        self._respond(b'Unknown run', 'text/plain; charset=utf-8', 404)
                        return
                else:
                    data = reader.snapshot()
                self._respond(json.dumps(data, ensure_ascii=False, allow_nan=False).encode(),
                              'application/json; charset=utf-8')
                return
            if path.startswith('/api/run/'):
                parts = path.split('/')
                data = reader.run_details(parts[3]) if len(parts) == 4 else None
                if data is not None:
                    self._respond(json.dumps(data, ensure_ascii=False, allow_nan=False).encode(),
                                  'application/json; charset=utf-8')
                    return
            if path.startswith('/media/'):
                parts = path.split('/')
                if len(parts) == 4 and all(NAME.fullmatch(part) for part in parts[2:]):
                    content = reader.media(parts[2], parts[3])
                    if content is not None:
                        self._respond(content, 'image/png' if parts[3].endswith('.png')
                                      else 'image/gif')
                        return
            static = {'/': ('index.html', 'text/html; charset=utf-8'),
                      '/assets/overview.js': ('overview.js', 'text/javascript; charset=utf-8'),
                      '/assets/overview.css': ('overview.css', 'text/css; charset=utf-8')}
            if path in static:
                filename, content_type = static[path]
                target = assets / filename
                try:
                    if target.is_symlink() or target.resolve().parent != assets.resolve():
                        raise OSError('Unsafe asset path')
                    content = target.read_bytes()
                    self._respond(content, content_type)
                    return
                except OSError:
                    pass
            self._respond(b'Not found', 'text/plain; charset=utf-8', 404)

        def log_message(self, format, *args):
            pass

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--registry', type=Path, default=Path('.cache/active_mlx_launch.json'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    args = parser.parse_args()
    try:
        server = make_server(OverviewReader(args.runs, args.registry), args.host, args.port)
    except (OSError, ValueError) as error:
        parser.exit(2, f'Cannot start overview: {error}\n')
    print(f'Training overview: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever(poll_interval=.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
