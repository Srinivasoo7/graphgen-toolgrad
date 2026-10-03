"""Reachability only. Not an access decision."""

from __future__ import annotations

import httpx

from crux.config import Settings


def live_probe(settings: Settings) -> dict[str, bool]:
    return {
        "utopia": _reachable(f"{settings.utopia_url.rstrip('/')}/api/v1/health"),
        "headroom": _headroom(settings),
    }


def _reachable(url: str) -> bool:
    try:
        with httpx.Client(timeout=3.0) as client:
            client.get(url)
            return True
    except httpx.HTTPError:
        return False


def _headroom(settings: Settings) -> bool:
    url = f"{settings.headroom_url.rstrip('/')}/v1/compress"
    headers = {"content-type": "application/json"}
    if settings.headroom_proxy_token:
        headers["X-Headroom-Proxy-Token"] = settings.headroom_proxy_token
    try:
        with httpx.Client(timeout=3.0) as client:
            client.post(
                url,
                json={"messages": [], "model": settings.model, "config": {}},
                headers=headers,
            )
            return True
    except httpx.HTTPError:
        return False
