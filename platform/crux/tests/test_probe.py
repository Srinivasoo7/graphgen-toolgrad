import httpx

from crux.config import Settings
from crux.probe import live_probe


def _settings():
    return Settings(
        utopia_url="http://utopia.example",
        headroom_url="http://headroom.example",
        headroom_proxy_token="hr",
        admin_token="admin",
        data_dir=".",
        bind="0.0.0.0:8788",
        model="gpt-4o",
    )


def test_live_probe_reports_reachability(monkeypatch):
    def fake_request(request):
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path.endswith("/v1/compress"):
            assert request.headers.get("X-Headroom-Proxy-Token") == "hr"
            return httpx.Response(400, json={"error": "empty"})
        return httpx.Response(404)

    real_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(fake_request)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    assert live_probe(_settings()) == {"utopia": True, "headroom": True}


def test_live_probe_false_when_down(monkeypatch):
    def boom(request):
        raise httpx.ConnectError("down", request=request)

    real_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(boom)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    assert live_probe(_settings()) == {"utopia": False, "headroom": False}
