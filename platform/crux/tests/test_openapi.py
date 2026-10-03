import json
from http.client import HTTPConnection

from crux.openapi import openapi_document
from crux.server import serve_in_thread
from tests.test_app import _app


def test_openapi_describes_agent_port():
    spec = openapi_document()
    assert spec["info"]["title"] == "Crux"
    paths = spec["paths"]
    assert "/v1/ask" in paths
    assert "/v1/session" in paths
    assert "/v1/catalog" in paths
    assert "/health" in paths
    assert "remember" in str(spec).lower()


def test_openapi_http_route():
    server, host, port = serve_in_thread(_app())
    try:
        conn = HTTPConnection(host, port)
        conn.request("GET", "/v1/openapi.json")
        resp = conn.getresponse()
        assert resp.status == 200
        body = json.loads(resp.read())
        assert "/v1/ask" in body["paths"]
    finally:
        server.shutdown()
