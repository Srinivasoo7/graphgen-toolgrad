import httpx

from crux.bootstrap import UtopiaAdmin


def test_register_or_login_falls_back_and_lists_workspace(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/auth/register"):
            return httpx.Response(409, json={"error": "exists"})
        if request.url.path.endswith("/auth/login"):
            return httpx.Response(
                200, json={"user": {"id": "u1"}, "token": "jwt"}
            )
        if request.url.path.endswith("/workspaces"):
            assert request.headers["Authorization"] == "Bearer jwt"
            return httpx.Response(200, json=[{"id": "ws-1"}])
        return httpx.Response(404)

    admin = UtopiaAdmin("http://utopia.example")
    real_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    data = admin.register_or_login("a@b.c", "password1", "Admin", "Crux")
    assert data["workspace"]["id"] == "ws-1"
    assert admin.token == "jwt"


def test_create_kb_reuses_existing(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/kbs") and request.method == "GET":
            return httpx.Response(
                200, json=[{"id": "kb-1", "name": "Production"}]
            )
        return httpx.Response(500)

    admin = UtopiaAdmin("http://utopia.example")
    admin.token = "jwt"
    real_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    kb = admin.create_restricted_kb("ws-1", "Production", ["schema-org"])
    assert kb["id"] == "kb-1"


def test_create_kb_and_issue_token(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/kbs") and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path.endswith("/kbs") and request.method == "POST":
            body = json_body(request)
            assert body["visibility"] == "restricted"
            return httpx.Response(
                200, json={"id": "kb-new", "name": body["name"]}
            )
        if request.url.path.endswith("/me/tokens"):
            return httpx.Response(
                200, json={"token": "pat", "info": {"id": "tok-1"}}
            )
        return httpx.Response(404)

    admin = UtopiaAdmin("http://utopia.example")
    admin.token = "jwt"
    real_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    kb = admin.create_restricted_kb("ws-1", "Staging", ["schema-org"])
    token = admin.issue_token("crux-staging", "write", [kb["id"]])
    assert kb["id"] == "kb-new"
    assert token["token"] == "pat"


def json_body(request: httpx.Request):
    import json

    return json.loads(request.content)
