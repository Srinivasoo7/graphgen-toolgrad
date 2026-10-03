"""Loopback HTTP surface for agents. Ontology UI stays on its own port."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, Tuple
from urllib.parse import urlparse

from crux.access import AccessDenied
from crux.app import CruxApp
from crux.console import console_html, docs_page, pitch_html
from crux.openapi import openapi_document
from crux.registry import UnknownIdentity


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _admin_token(handler: BaseHTTPRequestHandler) -> str | None:
    header = handler.headers.get("Authorization") or ""
    if header.lower().startswith("bearer "):
        return header.split(" ", 1)[1].strip()
    return handler.headers.get("X-Crux-Admin-Token")


def make_handler(app: CruxApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send(self, code: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_html(self, code: int, payload: str, content_type: str = "text/html; charset=utf-8") -> None:
            raw = payload.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_html(200, console_html())
                return
            if path == "/pitch":
                deck = pitch_html()
                if deck is None:
                    self._send(404, {"error": "pitch not found"})
                    return
                self._send_html(200, deck)
                return
            if path.startswith("/docs/"):
                page = docs_page(path[len("/docs/") :])
                if page is None:
                    self._send(404, {"error": "not found"})
                    return
                self._send_html(200, page)
                return
            if path == "/health":
                body = app.health()
                self._send(200 if body.get("status") == "ok" else 503, body)
                return
            if path == "/v1/audit":
                try:
                    app._require_admin(_admin_token(self))
                except AccessDenied as exc:
                    self._send(403, {"error": str(exc)})
                    return
                self._send(200, {"events": json.loads(app.audit.dump())})
                return
            if path == "/v1/tools":
                self._send(200, app.tools())
                return
            if path == "/v1/readiness":
                self._send(200, app.readiness())
                return
            if path == "/v1/openapi.json":
                self._send(200, openapi_document())
                return
            if path == "/v1/enrollments":
                try:
                    self._send(200, {"enrollments": app.list_enrollments(_admin_token(self))})
                except AccessDenied as exc:
                    self._send(403, {"error": str(exc)})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = _read_json(self)
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid json"})
                return
            try:
                if path == "/v1/enroll":
                    self._send(200, app.enroll(body, _admin_token(self)))
                    return
                if path == "/v1/revoke":
                    self._send(200, app.revoke(body, _admin_token(self)))
                    return
                if path == "/v1/ask":
                    self._send(200, app.ask(body))
                    return
                if path == "/v1/catalog":
                    self._send(200, app.catalog(body))
                    return
                if path == "/v1/session":
                    self._send(200, app.start_session(body))
                    return
            except AccessDenied as exc:
                self._send(403, {"error": str(exc)})
                return
            except UnknownIdentity as exc:
                self._send(403, {"error": str(exc)})
                return
            except (KeyError, ValueError) as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(404, {"error": "not found"})

    return Handler


def serve_in_thread(
    app: CruxApp, host: str = "127.0.0.1", port: int = 0
) -> Tuple[ThreadingHTTPServer, str, int]:
    handler = make_handler(app)
    server = ThreadingHTTPServer((host, port), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    bound_host, bound_port = server.server_address[:2]
    return server, bound_host, bound_port


def serve_forever(app: CruxApp, host: str, port: int) -> None:
    handler = make_handler(app)
    server = ThreadingHTTPServer((host, port), handler)
    server.serve_forever()
