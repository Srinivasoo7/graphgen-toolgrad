"""Instrumented Utopia MCP calls only. Native shell/file/HTTP are not claimed."""

from __future__ import annotations

from typing import Any

import httpx

from crux.access import Enrollment, Operation, Session, authorize
from crux.audit import MemoryAuditLog, build_event
from crux.compress import CallKind, CompressResult, HeadroomWorker


class UtopiaClient:
    def __init__(self, base_url: str, token: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def mcp_call(self, kb_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/api/v1/kbs/{kb_id}/mcp",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()


def instrumented_mcp(
    *,
    enrollment: Enrollment,
    session: Session,
    kb_id: str,
    tool: str,
    arguments: dict[str, Any],
    utopia: UtopiaClient,
    audit: MemoryAuditLog,
    operation: Operation = Operation.READ,
) -> dict[str, Any]:
    try:
        authorize(enrollment, session, kb_id=kb_id, operation=operation)
    except Exception:
        audit.append(
            build_event(
                owner_id=enrollment.identity.owner_id,
                agent_id=enrollment.identity.agent_id,
                credential_id=enrollment.identity.credential_id,
                session_id=session.session_id,
                tool=tool,
                kb_id=kb_id,
                status="denied",
                args=arguments,
            )
        )
        raise
    result = utopia.mcp_call(kb_id, tool, arguments)
    audit.append(
        build_event(
            owner_id=enrollment.identity.owner_id,
            agent_id=enrollment.identity.agent_id,
            credential_id=enrollment.identity.credential_id,
            session_id=session.session_id,
            tool=tool,
            kb_id=kb_id,
            status="ok",
            args=arguments,
        )
    )
    return result


def authorized_then_compress(
    *,
    enrollment: Enrollment,
    session: Session,
    kb_id: str,
    tool: str,
    arguments: dict[str, Any],
    utopia: UtopiaClient,
    audit: MemoryAuditLog,
    worker: HeadroomWorker,
    model: str,
    kind: CallKind,
    operation: Operation = Operation.READ,
) -> tuple[dict[str, Any], CompressResult]:
    raw = instrumented_mcp(
        enrollment=enrollment,
        session=session,
        kb_id=kb_id,
        tool=tool,
        arguments=arguments,
        utopia=utopia,
        audit=audit,
        operation=operation,
    )
    text = ""
    try:
        text = raw["result"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        text = str(raw)
    messages = [{"role": "tool", "content": text}]
    compressed = worker.compress(messages, model=model, kind=kind)
    return raw, compressed
