"""Instrumented-path audit. Redacted metadata only. No plain argument hashes."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


SENSITIVE_ARG_KEYS = frozenset(
    {"text", "query", "sql", "content", "password", "token", "authorization"}
)


@dataclass(frozen=True)
class AuditEvent:
    at: str
    owner_id: str
    agent_id: str
    credential_id: str
    session_id: str
    tool: str
    kb_id: str | None
    status: str
    arg_keys: tuple[str, ...]
    correlation: str | None


def redact_args(args: dict[str, Any] | None) -> tuple[str, ...]:
    if not args:
        return ()
    return tuple(sorted(k for k in args if k not in SENSITIVE_ARG_KEYS))


def keyed_correlation(secret: bytes, parts: list[str]) -> str:
    mac = hmac.new(secret, "|".join(parts).encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()


def build_event(
    *,
    owner_id: str,
    agent_id: str,
    credential_id: str,
    session_id: str,
    tool: str,
    kb_id: str | None,
    status: str,
    args: dict[str, Any] | None,
    correlation_secret: bytes | None = None,
) -> AuditEvent:
    correlation = None
    if correlation_secret:
        correlation = keyed_correlation(
            correlation_secret,
            [owner_id, agent_id, credential_id, session_id, tool, kb_id or ""],
        )
    return AuditEvent(
        at=datetime.now(timezone.utc).isoformat(),
        owner_id=owner_id,
        agent_id=agent_id,
        credential_id=credential_id,
        session_id=session_id,
        tool=tool,
        kb_id=kb_id,
        status=status,
        arg_keys=redact_args(args),
        correlation=correlation,
    )


def event_to_public_dict(event: AuditEvent) -> dict[str, Any]:
    """Safe to persist. Contains no argument values and no plain hashes of them."""
    return asdict(event)


class MemoryAuditLog:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def append(self, event: AuditEvent) -> None:
        self.events.append(event)

    def dump(self) -> str:
        return json.dumps([event_to_public_dict(e) for e in self.events], indent=2)


class FileAuditLog:
    """JSONL on disk. Same public shape as MemoryAuditLog. No argument values."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.events: list[AuditEvent] = []
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as handle:
                lines = handle.readlines()
        except FileNotFoundError:
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["arg_keys"] = tuple(data.get("arg_keys") or ())
            self.events.append(AuditEvent(**data))

    def append(self, event: AuditEvent) -> None:
        self.events.append(event)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_to_public_dict(event)) + "\n")

    def dump(self) -> str:
        return json.dumps([event_to_public_dict(e) for e in self.events], indent=2)
