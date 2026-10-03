from crux.config import load_settings


def test_load_settings_from_environ():
    settings = load_settings(
        {
            "UTOPIA_URL": "http://utopia:1516",
            "HEADROOM_URL": "http://headroom:8787",
            "HEADROOM_PROXY_TOKEN": "hr-token",
            "CRUX_ADMIN_TOKEN": "admin-token",
            "CRUX_DATA_DIR": "/data/crux",
            "CRUX_BIND": "0.0.0.0:8788",
            "CRUX_MODEL": "gpt-4o",
        }
    )
    assert settings.utopia_url == "http://utopia:1516"
    assert settings.headroom_url == "http://headroom:8787"
    assert settings.headroom_proxy_token == "hr-token"
    assert settings.admin_token == "admin-token"
    assert settings.data_dir == "/data/crux"
    assert settings.bind == "0.0.0.0:8788"
    assert settings.model == "gpt-4o"
