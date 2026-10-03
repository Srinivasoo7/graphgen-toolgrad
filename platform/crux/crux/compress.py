"""Headroom as a compress-only worker after Utopia has authorized the payload.

Bypass is product configuration (call kind), never a caller header or prompt.
Release A does not send config.mode=ccr.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx


class CallKind(str, Enum):
    AGENT_CHAT = "agent_chat"
    EXTRACTION = "extraction"
    EMBEDDING = "embedding"
    ONTOLOGY_ADMIN = "ontology_admin"


# Trusted product config — not request headers.
COMPRESSION_BYPASS_KINDS = frozenset(
    {
        CallKind.EXTRACTION,
        CallKind.EMBEDDING,
        CallKind.ONTOLOGY_ADMIN,
    }
)


@dataclass
class CompressResult:
    messages: list[dict[str, Any]]
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    skipped: bool
    skip_reason: str | None = None


CCR_MARKER = "<<ccr:"


def messages_contain_ccr(messages: Any) -> bool:
    if not isinstance(messages, list):
        return True
    try:
        return CCR_MARKER in json.dumps(messages)
    except (TypeError, ValueError):
        return True


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _fail_open(messages: list[dict[str, Any]], reason: str) -> CompressResult:
    return CompressResult(
        messages=messages,
        tokens_before=0,
        tokens_after=0,
        tokens_saved=0,
        skipped=True,
        skip_reason=reason,
    )


class HeadroomWorker:
    def __init__(
        self,
        base_url: str,
        *,
        proxy_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.proxy_token = proxy_token
        self.timeout = timeout

    def compress(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        kind: CallKind,
        frozen_message_count: int = 0,
    ) -> CompressResult:
        if kind in COMPRESSION_BYPASS_KINDS:
            return CompressResult(
                messages=messages,
                tokens_before=0,
                tokens_after=0,
                tokens_saved=0,
                skipped=True,
                skip_reason=f"trusted_config:{kind.value}",
            )
        headers = {"content-type": "application/json"}
        if self.proxy_token:
            headers["X-Headroom-Proxy-Token"] = self.proxy_token
        # Never send x-headroom-bypass — that would let a caller skip security
        # and/or compression. Bypass is decided above from CallKind.
        body = {
            "messages": messages,
            "model": model,
            "config": {"frozen_message_count": frozen_message_count},
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/v1/compress",
                    json=body,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            return _fail_open(messages, f"fail_open:{exc.__class__.__name__}")
        proposed = data.get("messages")
        if not isinstance(proposed, list):
            return _fail_open(messages, "fail_open:invalid_compress_result")
        if messages_contain_ccr(proposed):
            return _fail_open(messages, "fail_open:ccr_marker")
        return CompressResult(
            messages=proposed,
            tokens_before=_safe_int(data.get("tokens_before")),
            tokens_after=_safe_int(data.get("tokens_after")),
            tokens_saved=_safe_int(data.get("tokens_saved")),
            skipped=bool(data.get("compression_skipped", False)),
            skip_reason=data.get("skip_reason"),
        )
