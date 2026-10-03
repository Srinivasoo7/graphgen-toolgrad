"""Process settings. Agents do not receive these values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Settings:
    utopia_url: str
    headroom_url: str
    headroom_proxy_token: str
    admin_token: str
    data_dir: str
    bind: str
    model: str


def load_settings(environ: Mapping[str, str]) -> Settings:
    return Settings(
        utopia_url=environ.get("UTOPIA_URL", "http://utopia:1516"),
        headroom_url=environ.get("HEADROOM_URL", "http://headroom:8787"),
        headroom_proxy_token=environ["HEADROOM_PROXY_TOKEN"],
        admin_token=environ["CRUX_ADMIN_TOKEN"],
        data_dir=environ.get("CRUX_DATA_DIR", "/data/crux"),
        bind=environ.get("CRUX_BIND", "0.0.0.0:8788"),
        model=environ.get("CRUX_MODEL", "gpt-4o"),
    )
