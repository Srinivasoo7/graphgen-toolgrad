from pathlib import Path

import httpx

from crux.app import CruxApp
from crux.audit import MemoryAuditLog
from crux.compress import HeadroomWorker
from crux.config import Settings
from crux.dispatch import UtopiaClient
from crux.inflight import InFlightTracker
from crux.readiness import chat_url_is_headroom, live_inspector, normalize_kb_readiness
from crux.registry import EnrollmentRegistry
from tests.test_app import FakeUtopia


def test_chat_url_is_headroom_detects_side_door():
    assert chat_url_is_headroom("http://headroom:8787/v1") is True
    assert chat_url_is_headroom("http://127.0.0.1:8787") is True
    assert chat_url_is_headroom("https://api.openai.com/v1") is False
    assert chat_url_is_headroom(None) is False


def test_normalize_kb_readiness():
    body = normalize_kb_readiness(
        {
            "has_chat_model": False,
            "documents": 1,
            "processing": 0,
            "failed": 0,
            "entities": 0,
        }
    )
    assert body["has_chat_model"] is False
    assert body["documents"] == 1
    assert body["entities"] == 0


def test_app_readiness_lists_restricted_bases_and_flags_headroom(tmp_path):
    state = tmp_path / "bootstrap.json"
    state.write_text(
        '{"production_kb_id":"kb-p","staging_kb_id":"kb-s","workspace_id":"ws"}',
        encoding="utf-8",
    )
    settings = Settings(
        utopia_url="http://utopia.invalid",
        headroom_url="http://headroom.invalid",
        headroom_proxy_token="hr",
        admin_token="admin-secret",
        data_dir=str(tmp_path),
        bind="127.0.0.1:0",
        model="gpt-4o",
    )
    utopia = FakeUtopia()

    def factory(token):
        utopia.token = token
        return utopia

    app = CruxApp(
        settings=settings,
        registry=EnrollmentRegistry(),
        audit=MemoryAuditLog(),
        utopia_factory=factory,
        worker=HeadroomWorker("http://headroom.invalid"),
        inflight=InFlightTracker(),
        probe=lambda: {"utopia": True, "headroom": True},
        inspector=lambda: {
            "settings": {"chat_base_url": "http://headroom:8787/v1"},
            "kbs": {
                "kb-p": {
                    "has_chat_model": False,
                    "documents": 1,
                    "processing": 0,
                    "failed": 0,
                    "entities": 0,
                },
                "kb-s": {
                    "has_chat_model": False,
                    "documents": 1,
                    "processing": 0,
                    "failed": 0,
                    "entities": 0,
                },
            },
        },
    )
    body = app.readiness()
    assert body["product"] == "crux"
    assert body["chat_base_url_is_headroom"] is True
    names = {row["name"]: row for row in body["bases"]}
    assert names["production"]["kb_id"] == "kb-p"
    assert names["production"]["documents"] == 1
    assert names["staging"]["has_chat_model"] is False
    assert "chat_base_url_points_at_headroom" in body["blockers"]
    assert "production_missing_chat_model" in body["blockers"]
    assert "staging_missing_chat_model" in body["blockers"]
    assert Path(tmp_path / "bootstrap.json").exists()


def test_live_inspector_fetches_settings_and_kb_readiness(monkeypatch, tmp_path):
    (tmp_path / "bootstrap.json").write_text(
        '{"workspace_id":"ws-1","production_kb_id":"kb-p","staging_kb_id":"kb-s"}',
        encoding="utf-8",
    )
    settings = Settings(
        utopia_url="http://utopia.example",
        headroom_url="http://headroom.example",
        headroom_proxy_token="hr",
        admin_token="admin",
        data_dir=str(tmp_path),
        bind="0.0.0.0:8788",
        model="gpt-4o",
    )

    def handler(request):
        if request.url.path.endswith("/auth/login"):
            return httpx.Response(200, json={"token": "jwt"})
        if request.url.path.endswith("/settings"):
            return httpx.Response(200, json={"chat_base_url": "https://api.openai.com/v1"})
        if request.url.path.endswith("/readiness"):
            return httpx.Response(
                200,
                json={
                    "has_chat_model": True,
                    "documents": 1,
                    "processing": 0,
                    "failed": 0,
                    "entities": 0,
                },
            )
        return httpx.Response(404)

    real_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    inspected = live_inspector(
        settings,
        {"CRUX_ADMIN_EMAIL": "admin@crux.local", "CRUX_ADMIN_PASSWORD": "pw"},
    )
    assert inspected["settings"]["chat_base_url"] == "https://api.openai.com/v1"
    assert inspected["kbs"]["kb-p"]["documents"] == 1


def test_live_inspector_empty_without_admin_password(tmp_path):
    settings = Settings(
        utopia_url="http://utopia.example",
        headroom_url="http://headroom.example",
        headroom_proxy_token="hr",
        admin_token="admin",
        data_dir=str(tmp_path),
        bind="0.0.0.0:8788",
        model="gpt-4o",
    )
    assert live_inspector(settings, {}) == {"settings": {}, "kbs": {}}
