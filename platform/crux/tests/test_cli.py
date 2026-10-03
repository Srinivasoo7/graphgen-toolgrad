import json

from crux.__main__ import main


def test_usage_is_nonzero():
    assert main(["nope"]) == 2


def test_health_command(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "hr")
    monkeypatch.setenv("CRUX_ADMIN_TOKEN", "admin")
    monkeypatch.setenv("CRUX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("UTOPIA_URL", "http://utopia.invalid")
    monkeypatch.setenv("HEADROOM_URL", "http://headroom.invalid")

    def fake_probe(settings):
        return {"utopia": True, "headroom": True}

    monkeypatch.setattr("crux.__main__.live_probe", fake_probe)
    assert main(["health"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["status"] == "ok"
    assert body["product"] == "crux"
