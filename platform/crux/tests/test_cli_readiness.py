import json

from crux.__main__ import main


def test_readiness_command_reports_missing_chat_model(monkeypatch, tmp_path, capsys):
    (tmp_path / "bootstrap.json").write_text(
        '{"production_kb_id":"kb-p","staging_kb_id":"kb-s"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("HEADROOM_PROXY_TOKEN", "hr")
    monkeypatch.setenv("CRUX_ADMIN_TOKEN", "admin")
    monkeypatch.setenv("CRUX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("UTOPIA_URL", "http://utopia.invalid")
    monkeypatch.setenv("HEADROOM_URL", "http://headroom.invalid")
    monkeypatch.setattr(
        "crux.__main__.live_probe",
        lambda settings: {"utopia": True, "headroom": True},
    )
    monkeypatch.setattr(
        "crux.__main__.live_inspector",
        lambda settings, environ: {
            "settings": {},
            "kbs": {
                "kb-p": {
                    "has_chat_model": False,
                    "documents": 1,
                    "processing": 0,
                    "failed": 1,
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
    assert main(["readiness"]) == 1
    body = json.loads(capsys.readouterr().out)
    assert "production_missing_chat_model" in body["blockers"]
    assert body["chat_base_url_is_headroom"] is False
