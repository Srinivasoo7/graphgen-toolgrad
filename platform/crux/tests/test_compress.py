from crux.compress import COMPRESSION_BYPASS_KINDS, CallKind, HeadroomWorker
import httpx


def test_bypass_kinds_are_trusted_config_not_headers():
    assert CallKind.EXTRACTION in COMPRESSION_BYPASS_KINDS
    assert CallKind.EMBEDDING in COMPRESSION_BYPASS_KINDS
    assert CallKind.ONTOLOGY_ADMIN in COMPRESSION_BYPASS_KINDS
    assert CallKind.AGENT_CHAT not in COMPRESSION_BYPASS_KINDS


def test_bypass_does_not_call_headroom():
    worker = HeadroomWorker("http://headroom.invalid")
    messages = [{"role": "user", "content": "extract this chunk"}]
    result = worker.compress(messages, model="gpt-4o", kind=CallKind.EXTRACTION)
    assert result.skipped is True
    assert result.skip_reason == "trusted_config:extraction"
    assert result.messages == messages


def test_fail_open_returns_original_messages():
    worker = HeadroomWorker("http://127.0.0.1:1", timeout=0.1)
    messages = [{"role": "tool", "content": "rows"}]
    result = worker.compress(messages, model="gpt-4o", kind=CallKind.AGENT_CHAT)
    assert result.skipped is True
    assert result.skip_reason.startswith("fail_open:")
    assert result.messages == messages


def test_ccr_markers_fail_open_to_original(monkeypatch):
    original = [{"role": "tool", "content": "authorized rows"}]

    def handler(request):
        return httpx.Response(
            200,
            json={
                "messages": [{"role": "tool", "content": "see <<ccr:abc>>"}],
                "tokens_before": 10,
                "tokens_after": 2,
                "tokens_saved": 8,
            },
        )

    real = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    worker = HeadroomWorker("http://headroom.example")
    result = worker.compress(original, model="gpt-4o", kind=CallKind.AGENT_CHAT)
    assert result.skipped is True
    assert "ccr" in (result.skip_reason or "")
    assert result.messages == original


def test_null_messages_from_headroom_fail_open(monkeypatch):
    original = [{"role": "tool", "content": "authorized rows"}]

    def handler(request):
        return httpx.Response(
            200,
            json={"messages": None, "tokens_before": 10, "tokens_after": 2, "tokens_saved": 8},
        )

    real = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    worker = HeadroomWorker("http://headroom.example")
    result = worker.compress(original, model="gpt-4o", kind=CallKind.AGENT_CHAT)
    assert result.skipped is True
    assert result.messages == original


def test_non_numeric_token_fields_do_not_crash(monkeypatch):
    original = [{"role": "tool", "content": "authorized rows"}]

    def handler(request):
        return httpx.Response(
            200,
            json={
                "messages": original,
                "tokens_before": "n/a",
                "tokens_after": None,
                "tokens_saved": [],
            },
        )

    real = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    worker = HeadroomWorker("http://headroom.example")
    result = worker.compress(original, model="gpt-4o", kind=CallKind.AGENT_CHAT)
    assert result.messages == original
    assert result.tokens_before == 0
    assert result.tokens_after == 0
    assert result.tokens_saved == 0
