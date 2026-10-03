"""Crux product surface: enroll, ask, revoke. Utopia authorizes; Headroom compresses."""

import json
from http.client import HTTPConnection

import pytest

from crux.access import AccessDenied, EnrollmentStatus, Identity, Operation
from crux.app import CruxApp
from crux.audit import MemoryAuditLog
from crux.compress import CallKind, CompressResult, HeadroomWorker
from crux.config import Settings
from crux.dispatch import UtopiaClient
from crux.inflight import InFlightTracker
from crux.registry import EnrollmentRegistry
from crux.server import serve_in_thread


class FakeUtopia(UtopiaClient):
    def __init__(self):
        self.calls = []
        self.token = None

    def mcp_call(self, kb_id, name, arguments):
        self.calls.append((kb_id, name, arguments, self.token))
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "authorized-fact"}]},
        }


def _settings(**overrides):
    values = dict(
        utopia_url="http://utopia.invalid",
        headroom_url="http://headroom.invalid",
        headroom_proxy_token="hr",
        admin_token="admin-secret",
        data_dir="__no_bootstrap__",
        bind="127.0.0.1:0",
        model="gpt-4o",
    )
    values.update(overrides)
    return Settings(**values)


def _app(registry=None, utopia=None, worker=None, settings=None):
    utopia = utopia or FakeUtopia()

    def factory(token):
        utopia.token = token
        return utopia

    def fake_compress(self, messages, *, model, kind, frozen_message_count=0):
        assert kind is CallKind.AGENT_CHAT
        return CompressResult(
            messages=[{"role": "tool", "content": "compressed"}],
            tokens_before=10,
            tokens_after=3,
            tokens_saved=7,
            skipped=False,
        )

    worker = worker or HeadroomWorker("http://headroom.invalid")
    worker.compress = fake_compress.__get__(worker, HeadroomWorker)
    return CruxApp(
        settings=settings or _settings(),
        registry=registry or EnrollmentRegistry(),
        audit=MemoryAuditLog(),
        utopia_factory=factory,
        worker=worker,
        inflight=InFlightTracker(),
        probe=lambda: {"utopia": True, "headroom": True},
    )


def test_ask_pending_is_denied_and_does_not_call_utopia():
    utopia = FakeUtopia()
    app = _app(utopia=utopia)
    with pytest.raises(AccessDenied):
        app.ask(
            {
                "owner_id": "owner-1",
                "agent_id": "agent-a",
                "credential_id": "cred-a",
                "session_id": "s1",
                "kb_id": "kb-production",
                "tool": "search_chunks",
                "arguments": {"query": "secret"},
            }
        )
    assert utopia.calls == []


def test_enroll_then_ask_uses_stored_pat_and_compresses():
    utopia = FakeUtopia()
    app = _app(utopia=utopia)
    identity = Identity("owner-1", "agent-a", "cred-a")
    app.enroll(
        {
            "owner_id": identity.owner_id,
            "agent_id": identity.agent_id,
            "credential_id": identity.credential_id,
            "utopia_user_id": "utp-user-1",
            "utopia_token_id": "utp-tok-1",
            "utopia_pat": "pat-for-agent",
            "granted_kb_ids": ["kb-production"],
            "operations": ["read"],
        },
        admin_token="admin-secret",
    )
    result = app.ask(
        {
            "owner_id": identity.owner_id,
            "agent_id": identity.agent_id,
            "credential_id": identity.credential_id,
            "session_id": "s1",
            "kb_id": "kb-production",
            "tool": "search_chunks",
            "arguments": {"query": "x"},
        }
    )
    assert utopia.calls[0][3] == "pat-for-agent"
    assert result["utopia"]["result"]["content"][0]["text"] == "authorized-fact"
    assert result["compressed"]["tokens_saved"] == 7
    assert result["compressed"]["messages"][0]["content"] == "compressed"
    assert result["text"] == "compressed"


def test_ask_rejects_unpinned_tool_and_does_not_call_utopia():
    utopia = FakeUtopia()
    app = _app(utopia=utopia)
    app.enroll(
        {
            "owner_id": "owner-1",
            "agent_id": "agent-a",
            "credential_id": "cred-a",
            "utopia_user_id": "u",
            "utopia_token_id": "t",
            "utopia_pat": "p",
            "granted_kb_ids": ["kb-production"],
            "operations": ["read"],
        },
        admin_token="admin-secret",
    )
    with pytest.raises(AccessDenied, match="not an instrumented MCP tool"):
        app.ask(
            {
                "owner_id": "owner-1",
                "agent_id": "agent-a",
                "credential_id": "cred-a",
                "session_id": "s1",
                "kb_id": "kb-production",
                "tool": "query_data",
                "arguments": {"sql": "select 1"},
            }
        )
    assert utopia.calls == []


def test_ask_remember_uses_propose_even_if_caller_sends_read():
    utopia = FakeUtopia()
    app = _app(utopia=utopia)
    app.enroll(
        {
            "owner_id": "owner-1",
            "agent_id": "agent-a",
            "credential_id": "cred-a",
            "utopia_user_id": "u",
            "utopia_token_id": "t",
            "utopia_pat": "p",
            "granted_kb_ids": ["kb-production"],
            "operations": ["read"],
        },
        admin_token="admin-secret",
    )
    with pytest.raises(AccessDenied, match="propose"):
        app.ask(
            {
                "owner_id": "owner-1",
                "agent_id": "agent-a",
                "credential_id": "cred-a",
                "session_id": "s1",
                "kb_id": "kb-production",
                "tool": "remember",
                "operation": "read",
                "arguments": {"text": "secret fact"},
            }
        )
    assert utopia.calls == []


def test_catalog_lists_grants_and_tools_without_pats():
    app = _app()
    app.enroll(
        {
            "owner_id": "owner-1",
            "agent_id": "agent-a",
            "credential_id": "cred-a",
            "utopia_user_id": "u",
            "utopia_token_id": "t",
            "utopia_pat": "secret-pat",
            "granted_kb_ids": ["kb-production"],
            "operations": ["read"],
        },
        admin_token="admin-secret",
    )
    catalog = app.catalog(
        {
            "owner_id": "owner-1",
            "agent_id": "agent-a",
            "credential_id": "cred-a",
        }
    )
    blob = json.dumps(catalog)
    assert "secret-pat" not in blob
    assert catalog["status"] == "active"
    assert catalog["granted_kb_ids"] == ["kb-production"]
    assert "search_chunks" in catalog["tools"]
    assert "remember" not in catalog["tools"]
    assert "query_data" not in catalog["tools"]


def test_tools_manifest_is_the_pinned_mcp_set():
    app = _app()
    manifest = app.tools()
    assert "search_chunks" in manifest["read"]
    assert manifest["write"] == ["remember"]
    assert "query_data" not in manifest["read"]
    assert manifest["remember_is_propose"] is True


def test_enroll_rejects_wrong_admin_token():
    app = _app()
    with pytest.raises(AccessDenied):
        app.enroll(
            {
                "owner_id": "o",
                "agent_id": "a",
                "credential_id": "c",
                "utopia_user_id": "u",
                "utopia_token_id": "t",
                "utopia_pat": "p",
                "granted_kb_ids": ["kb-production"],
                "operations": ["read"],
            },
            admin_token="wrong",
        )


def test_revoke_blocks_next_ask_and_cancels_in_flight():
    utopia = FakeUtopia()
    app = _app(utopia=utopia)
    body = {
        "owner_id": "owner-1",
        "agent_id": "agent-a",
        "credential_id": "cred-a",
        "utopia_user_id": "u",
        "utopia_token_id": "t",
        "utopia_pat": "p",
        "granted_kb_ids": ["kb-production"],
        "operations": ["read"],
    }
    app.enroll(body, admin_token="admin-secret")
    identity = Identity("owner-1", "agent-a", "cred-a")
    app.inflight.start("open-1", identity)
    app.revoke(
        {"owner_id": "owner-1", "agent_id": "agent-a", "credential_id": "cred-a"},
        admin_token="admin-secret",
    )
    assert app.inflight.is_cancelled("open-1") is True
    assert app.registry.lookup(identity).status is EnrollmentStatus.REVOKED
    with pytest.raises(AccessDenied):
        app.ask(
            {
                "owner_id": "owner-1",
                "agent_id": "agent-a",
                "credential_id": "cred-a",
                "session_id": "s1",
                "kb_id": "kb-production",
                "tool": "search_chunks",
                "arguments": {},
            }
        )
    assert utopia.calls == []


def test_http_health_and_ask_roundtrip():
    utopia = FakeUtopia()
    app = _app(utopia=utopia)
    app.enroll(
        {
            "owner_id": "owner-1",
            "agent_id": "agent-a",
            "credential_id": "cred-a",
            "utopia_user_id": "u",
            "utopia_token_id": "t",
            "utopia_pat": "p",
            "granted_kb_ids": ["kb-production"],
            "operations": ["read"],
        },
        admin_token="admin-secret",
    )
    server, host, port = serve_in_thread(app)
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/health")
        health = json.loads(conn.getresponse().read().decode("utf-8"))
        assert health["status"] == "ok"
        assert health["product"] == "crux"
        conn.request("GET", "/v1/tools")
        tools = json.loads(conn.getresponse().read().decode("utf-8"))
        assert "search_chunks" in tools["read"]
        conn.request(
            "POST",
            "/v1/catalog",
            body=json.dumps(
                {
                    "owner_id": "owner-1",
                    "agent_id": "agent-a",
                    "credential_id": "cred-a",
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        catalog = json.loads(conn.getresponse().read().decode("utf-8"))
        assert catalog["granted_kb_ids"] == ["kb-production"]
        conn.request(
            "POST",
            "/v1/ask",
            body=json.dumps(
                {
                    "owner_id": "owner-1",
                    "agent_id": "agent-a",
                    "credential_id": "cred-a",
                    "session_id": "s1",
                    "kb_id": "kb-production",
                    "tool": "search_chunks",
                    "arguments": {"query": "x"},
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        asked = json.loads(conn.getresponse().read().decode("utf-8"))
        assert asked["compressed"]["tokens_saved"] == 7
        assert asked["text"] == "compressed"
        conn.request("GET", "/nope")
        assert conn.getresponse().status == 404
        conn.request("POST", "/v1/ask", body="not-json", headers={"Content-Type": "application/json"})
        assert conn.getresponse().status == 400
        conn.request(
            "POST",
            "/v1/enroll",
            body=json.dumps(
                {
                    "owner_id": "owner-2",
                    "agent_id": "agent-b",
                    "credential_id": "cred-b",
                    "utopia_user_id": "u2",
                    "utopia_token_id": "t2",
                    "utopia_pat": "p2",
                    "granted_kb_ids": ["kb-staging"],
                    "operations": ["read"],
                }
            ),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer admin-secret",
            },
        )
        enrolled = json.loads(conn.getresponse().read().decode("utf-8"))
        assert enrolled["status"] == "active"
        conn.request(
            "GET",
            "/v1/audit",
            headers={"Authorization": "Bearer admin-secret"},
        )
        audit_body = json.loads(conn.getresponse().read().decode("utf-8"))
        assert audit_body["events"]
        conn.close()
    finally:
        server.shutdown()


def test_cancelled_request_id_is_denied_before_utopia():
    utopia = FakeUtopia()
    app = _app(utopia=utopia)
    identity = Identity("owner-1", "agent-a", "cred-a")
    app.enroll(
        {
            "owner_id": "owner-1",
            "agent_id": "agent-a",
            "credential_id": "cred-a",
            "utopia_user_id": "u",
            "utopia_token_id": "t",
            "utopia_pat": "p",
            "granted_kb_ids": ["kb-production"],
            "operations": ["read"],
        },
        admin_token="admin-secret",
    )
    app.inflight.start("req-cancel", identity)
    app.inflight.cancel_identity(identity)
    with pytest.raises(AccessDenied):
        app.ask(
            {
                "owner_id": "owner-1",
                "agent_id": "agent-a",
                "credential_id": "cred-a",
                "session_id": "s1",
                "kb_id": "kb-production",
                "tool": "search_chunks",
                "arguments": {},
                "request_id": "req-cancel",
            }
        )
    assert utopia.calls == []

