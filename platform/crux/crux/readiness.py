"""Operator view of restricted bases. Not an access decision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
import httpx

from crux.config import Settings

_HEADROOM_HINTS = ("headroom", ":8787", "/v1/compress")


def chat_url_is_headroom(url: str | None) -> bool:
    if not url:
        return False
    lowered = url.lower()
    return any(hint in lowered for hint in _HEADROOM_HINTS)


def normalize_kb_readiness(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "has_chat_model": bool(raw.get("has_chat_model")),
        "documents": int(raw.get("documents") or 0),
        "processing": int(raw.get("processing") or 0),
        "failed": int(raw.get("failed") or 0),
        "entities": int(raw.get("entities") or 0),
    }


def load_bootstrap_state(data_dir: str) -> dict[str, Any]:
    path = Path(data_dir) / "bootstrap.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def restricted_base_ids(data_dir: str) -> frozenset[str]:
    state = load_bootstrap_state(data_dir)
    return frozenset(
        kb_id
        for kb_id in (state.get("production_kb_id"), state.get("staging_kb_id"))
        if kb_id
    )


def extraction_blockers(
    *,
    chat_base_url_is_headroom: bool,
    bases: list[dict[str, Any]],
) -> list[str]:
    """Operator flags. Not an access decision. Graph extract stays in Utopia UI."""
    blockers: list[str] = []
    if chat_base_url_is_headroom:
        blockers.append("chat_base_url_points_at_headroom")
    for row in bases:
        if not row.get("has_chat_model"):
            blockers.append(f"{row.get('name', 'unknown')}_missing_chat_model")
    return blockers


def live_inspector(settings: Settings, environ: Mapping[str, str]) -> dict[str, Any]:
    """Read Utopia readiness with the admin session. Never uses Headroom."""
    state = load_bootstrap_state(settings.data_dir)
    email = environ.get("CRUX_ADMIN_EMAIL")
    password = environ.get("CRUX_ADMIN_PASSWORD")
    empty: dict[str, Any] = {"settings": {}, "kbs": {}}
    if not email or not password:
        return empty
    base = settings.utopia_url.rstrip("/")
    try:
        with httpx.Client(timeout=10.0) as client:
            login = client.post(
                f"{base}/api/v1/auth/login",
                json={"email": email, "password": password},
            )
            login.raise_for_status()
            token = login.json()["token"]
            headers = {"Authorization": f"Bearer {token}"}
            workspace_id = state.get("workspace_id")
            settings_body: dict[str, Any] = {}
            if workspace_id:
                resp = client.get(
                    f"{base}/api/v1/workspaces/{workspace_id}/settings",
                    headers=headers,
                )
                if resp.status_code < 400:
                    settings_body = resp.json()
            kbs: dict[str, Any] = {}
            for key in ("production_kb_id", "staging_kb_id"):
                kb_id = state.get(key)
                if not kb_id:
                    continue
                ready = client.get(f"{base}/api/v1/kbs/{kb_id}/readiness", headers=headers)
                if ready.status_code < 400:
                    kbs[kb_id] = ready.json()
            return {"settings": settings_body, "kbs": kbs}
    except httpx.HTTPError:
        return empty
