"""Serve the local neuro-symbolic dashboard.

Reads saved models and results. Does not retrain.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nsf.dashboard import DashboardError, DashboardStore

WEB = ROOT / "web"
STORE = DashboardStore(ROOT)


class ReuseServer(ThreadingHTTPServer):
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._guard(self._get)

    def do_POST(self) -> None:
        self._guard(self._post)

    def _get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            self._file(WEB / "index.html", "text/html; charset=utf-8")
        elif path == "/styles.css":
            self._file(WEB / "styles.css", "text/css; charset=utf-8")
        elif path == "/app.js":
            self._file(WEB / "app.js", "text/javascript; charset=utf-8")
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif path == "/api/overview":
            self._json(200, STORE.overview())
        elif path == "/api/cases":
            domain = query.get("domain", [""])[0]
            strategy = query.get("strategy", [""])[0]
            self._json(200, {"domain": domain, "strategy": strategy, "cases": STORE.cases(domain, strategy)})
        elif path == "/api/case":
            domain = query.get("domain", [""])[0]
            strategy = query.get("strategy", [""])[0]
            try:
                index = int(query.get("index", [""])[0])
            except ValueError:
                raise DashboardError("index must be an integer") from None
            self._json(200, STORE.case(domain, strategy, index))
        elif path == "/api/client_overview":
            domain = query.get("domain", [""])[0]
            token = query.get("token", [""])[0]
            self._json(200, STORE.client_overview(domain, token))
        elif path == "/api/client_cases":
            domain = query.get("domain", [""])[0]
            strategy = query.get("strategy", [""])[0]
            token = query.get("token", [""])[0]
            result = STORE.client_cases(domain, strategy, token)
            self._json(
                200,
                {"domain": domain, "strategy": strategy, "cases": result},
            )
        elif path.startswith("/plots/"):
            self._file(STORE.plot_path(path[len("/plots/") :]), "image/png")
        else:
            self._json(404, {"error": "Not found"})

    def _post(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in ("/api/predict", "/api/login", "/api/logout"):
            self._json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise DashboardError("Content-Length must be an integer") from None
        if length < 0 or length > 100_000:
            raise DashboardError("Request is too large", 413)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            raise DashboardError("Request body must be JSON") from None
        if not isinstance(payload, dict):
            raise DashboardError("Request body must be a JSON object")
        if parsed.path == "/api/login":
            self._json(200, STORE.login(payload.get("client_id"), payload.get("password")))
            return
        if parsed.path == "/api/logout":
            self._json(200, STORE.logout(payload.get("token")))
            return
        result = STORE.predict(
            payload.get("domain"),
            payload.get("strategy"),
            payload.get("inputs"),
            payload.get("entity_id"),
        )
        self._json(200, result)

    def _guard(self, fn) -> None:
        try:
            fn()
        except DashboardError as exc:
            self._json(exc.status, {"error": exc.message})
        except (BrokenPipeError, ConnectionAbortedError):
            return
        except Exception:
            traceback.print_exc()
            self._json(500, {"error": "Internal error"})

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._bytes(status, body, "application/json; charset=utf-8")

    def _file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._json(404, {"error": "Not found"})
            return
        self._bytes(200, path.read_bytes(), content_type)

    def _bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the NSF dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print("Loading saved models and datasets...")
    STORE.load()
    logins = STORE.login_hint()
    print("Client logins (local demo credentials, from client_logins.json):")
    for cid in sorted(logins, key=lambda k: int(k)):
        print(f"  Client {cid}  ->  password: {logins[cid]}")
    server = ReuseServer((args.host, args.port), Handler)
    print(f"Dashboard ready at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
