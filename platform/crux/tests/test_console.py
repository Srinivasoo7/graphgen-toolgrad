from http.client import HTTPConnection

import pytest

from crux.branding import ONTOLOGY_MODULE, TOKEN_MODULE
from crux.console import markdown_to_html
from crux.server import serve_in_thread
from tests.test_app import _app


def test_console_names_both_modules_and_forbids_headroom_as_knowledge():
    server, host, port = serve_in_thread(_app())
    try:
        conn = HTTPConnection(host, port)
        conn.request("GET", "/")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "text/html" in resp.getheader("Content-Type", "")
        assert ONTOLOGY_MODULE in body
        assert TOKEN_MODULE in body
        assert "127.0.0.1:1516" in body
        assert "chat_base_url" in body
        assert "knowledge store" not in body.lower()
    finally:
        server.shutdown()


def test_docs_serve_first_boot_qrg():
    server, host, port = serve_in_thread(_app())
    try:
        conn = HTTPConnection(host, port)
        conn.request("GET", "/docs/qrg-setup")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "docker compose" in body.lower()
        assert "Enterprise Ontology" in body
    finally:
        server.shutdown()


def test_pitch_answers_what_why_how():
    server, host, port = serve_in_thread(_app())
    try:
        conn = HTTPConnection(host, port)
        conn.request("GET", "/pitch")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        assert resp.status == 200
        assert "What?" in body
        assert "Why?" in body
        assert "How?" in body
        assert "wrong prediction" in body.lower()
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    "bad_slug",
    ["../etc/passwd", "a/../../secret", "UPPER_SLUG", ""],
)
def test_docs_rejects_traversal_and_bad_slugs(bad_slug):
    server, host, port = serve_in_thread(_app())
    try:
        conn = HTTPConnection(host, port)
        conn.request("GET", "/docs/" + bad_slug)
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 404
    finally:
        server.shutdown()


def test_markdown_blocks_javascript_hrefs():
    html = markdown_to_html("[x](javascript:alert(1))")
    assert "javascript:" not in html
    assert 'href="#"' in html


def test_unknown_doc_is_404():
    server, host, port = serve_in_thread(_app())
    try:
        conn = HTTPConnection(host, port)
        conn.request("GET", "/docs/not-a-real-guide")
        resp = conn.getresponse()
        assert resp.status == 404
    finally:
        server.shutdown()
