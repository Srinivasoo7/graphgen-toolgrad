"""Instrumented MCP path (ECC python-testing, mocked Utopia)."""

import httpx
import pytest

from crux.access import AccessDenied, EnrollmentStatus, Operation, Session
from crux.audit import MemoryAuditLog
from crux.compress import CallKind, HeadroomWorker
from crux.dispatch import UtopiaClient, authorized_then_compress, instrumented_mcp


class FakeUtopia(UtopiaClient):
    def __init__(self):
        self.calls = []

    def mcp_call(self, kb_id, name, arguments):
        self.calls.append((kb_id, name, arguments))
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "ok-from-utopia"}]},
        }


def test_denied_is_audited_and_does_not_call_utopia(active_prod_reader):
    utopia = FakeUtopia()
    audit = MemoryAuditLog()
    session = Session("s1", active_prod_reader.identity)
    with pytest.raises(AccessDenied):
        instrumented_mcp(
            enrollment=active_prod_reader,
            session=session,
            kb_id="kb-staging",
            tool="search_chunks",
            arguments={"query": "secret"},
            utopia=utopia,
            audit=audit,
        )
    assert utopia.calls == []
    assert audit.events[-1].status == "denied"
    assert "secret" not in str(audit.dump())


def test_authorized_mcp_then_compress_uses_tool_text(active_prod_reader, monkeypatch):
    utopia = FakeUtopia()
    audit = MemoryAuditLog()
    session = Session("s1", active_prod_reader.identity)
    worker = HeadroomWorker("http://headroom.invalid")

    def fake_compress(self, messages, *, model, kind, frozen_message_count=0):
        from crux.compress import CompressResult

        assert messages[0]["content"] == "ok-from-utopia"
        assert kind is CallKind.AGENT_CHAT
        return CompressResult(messages=messages, tokens_before=10, tokens_after=4, tokens_saved=6, skipped=False)

    monkeypatch.setattr(HeadroomWorker, "compress", fake_compress)
    raw, compressed = authorized_then_compress(
        enrollment=active_prod_reader,
        session=session,
        kb_id="kb-production",
        tool="search_chunks",
        arguments={"query": "x"},
        utopia=utopia,
        audit=audit,
        worker=worker,
        model="gpt-4o",
        kind=CallKind.AGENT_CHAT,
        operation=Operation.READ,
    )
    assert raw["result"]["content"][0]["text"] == "ok-from-utopia"
    assert compressed.tokens_saved == 6
    assert audit.events[-1].status == "ok"


def test_utopia_client_posts_mcp(monkeypatch):
    def handler(request):
        assert request.url.path == "/api/v1/kbs/kb-production/mcp"
        assert request.headers["Authorization"] == "Bearer pat"
        return httpx.Response(200, json={"jsonrpc": "2.0", "result": {"ok": True}})

    real_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    client = UtopiaClient("http://utopia.example/", "pat")
    assert client.mcp_call("kb-production", "search_chunks", {"q": "x"})["result"]["ok"] is True


def test_authorized_then_compress_stringifies_unexpected_shape(active_prod_reader, monkeypatch):
    class OddUtopia(UtopiaClient):
        def mcp_call(self, kb_id, name, arguments):
            return {"odd": True}

    def fake_compress(self, messages, *, model, kind, frozen_message_count=0):
        from crux.compress import CompressResult

        assert "{'odd': True}" in messages[0]["content"] or "odd" in messages[0]["content"]
        return CompressResult(messages=messages, tokens_before=1, tokens_after=1, tokens_saved=0, skipped=False)

    monkeypatch.setattr(HeadroomWorker, "compress", fake_compress)
    raw, compressed = authorized_then_compress(
        enrollment=active_prod_reader,
        session=Session("s1", active_prod_reader.identity),
        kb_id="kb-production",
        tool="search_chunks",
        arguments={},
            utopia=OddUtopia("http://utopia.example", "pat"),
        audit=MemoryAuditLog(),
        worker=HeadroomWorker("http://headroom.invalid"),
        model="gpt-4o",
        kind=CallKind.AGENT_CHAT,
    )
    assert raw == {"odd": True}
    assert compressed.skipped is False


def test_pending_never_reaches_utopia(active_prod_reader):
    active_prod_reader.status = EnrollmentStatus.PENDING
    utopia = FakeUtopia()
    audit = MemoryAuditLog()
    session = Session("s1", active_prod_reader.identity)
    with pytest.raises(AccessDenied):
        instrumented_mcp(
            enrollment=active_prod_reader,
            session=session,
            kb_id="kb-production",
            tool="entity_facts",
            arguments={},
            utopia=utopia,
            audit=audit,
        )
    assert utopia.calls == []
