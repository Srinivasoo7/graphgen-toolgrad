"""Workflow-agent client. Talks only to Crux, never Utopia or Headroom."""

from __future__ import annotations

from typing import Any, Iterable

import httpx


class CruxClient:
    def __init__(
        self,
        base_url: str,
        *,
        owner_id: str,
        agent_id: str,
        credential_id: str,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.owner_id = owner_id
        self.agent_id = agent_id
        self.credential_id = credential_id
        self.timeout = timeout

    def _identity(self) -> dict[str, str]:
        return {
            "owner_id": self.owner_id,
            "agent_id": self.agent_id,
            "credential_id": self.credential_id,
        }

    def _get(self, path: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}{path}")
            resp.raise_for_status()
            return resp.json()

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}{path}", json=body)
            resp.raise_for_status()
            return resp.json()

    def tools(self) -> dict[str, Any]:
        return self._get("/v1/tools")

    def readiness(self) -> dict[str, Any]:
        return self._get("/v1/readiness")

    def catalog(self) -> dict[str, Any]:
        return self._post("/v1/catalog", self._identity())

    def start_session(
        self,
        kb_ids: Iterable[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        body = dict(self._identity())
        if kb_ids is not None:
            body["kb_ids"] = list(kb_ids)
        if session_id:
            body["session_id"] = session_id
        return self._post("/v1/session", body)

    def ask(
        self,
        *,
        kb_id: str,
        tool: str,
        arguments: dict[str, Any] | None,
        session_id: str,
        session_kb_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        body = dict(self._identity())
        body.update(
            {
                "kb_id": kb_id,
                "tool": tool,
                "arguments": arguments or {},
                "session_id": session_id,
            }
        )
        if session_kb_ids is not None:
            body["session_kb_ids"] = list(session_kb_ids)
        return self._post("/v1/ask", body)

    def remember(
        self,
        *,
        kb_id: str,
        text: str,
        session_id: str,
        session_kb_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        return self.ask(
            kb_id=kb_id,
            tool="remember",
            arguments={"text": text},
            session_id=session_id,
            session_kb_ids=session_kb_ids,
        )

    def find_entities(
        self,
        *,
        kb_id: str,
        query: str,
        session_id: str,
        session_kb_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        return self.ask(
            kb_id=kb_id,
            tool="find_entities",
            arguments={"query": query},
            session_id=session_id,
            session_kb_ids=session_kb_ids,
        )

    def entity_facts(
        self,
        *,
        kb_id: str,
        entity_id: str,
        session_id: str,
        session_kb_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        return self.ask(
            kb_id=kb_id,
            tool="entity_facts",
            arguments={"entity_id": entity_id},
            session_id=session_id,
            session_kb_ids=session_kb_ids,
        )
