"""Product operations. Utopia authorizes knowledge; Headroom compresses."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any, Callable

from crux.access import (
    AccessDenied,
    EnrollmentStatus,
    Identity,
    Operation,
    Session,
    authorize,
    effective_kb_ids,
)
from crux.audit import MemoryAuditLog, build_event
from crux.branding import ONTOLOGY_MODULE, PRODUCT, TOKEN_MODULE
from crux.compress import CallKind, CompressResult, HeadroomWorker
from crux.config import Settings
from crux.dispatch import UtopiaClient, authorized_then_compress
from crux.inflight import InFlightTracker
from crux.readiness import (
    chat_url_is_headroom,
    extraction_blockers,
    load_bootstrap_state,
    normalize_kb_readiness,
    restricted_base_ids,
)
from crux.registry import EnrollmentRegistry, UnknownIdentity
from crux.tools import operation_for, tools_for_operations, tools_manifest


def reply_text(compressed: CompressResult) -> str:
    for message in compressed.messages:
        content = message.get("content")
        if isinstance(content, str) and content:
            return content
    return ""


class CruxApp:
    def __init__(
        self,
        settings: Settings,
        registry: EnrollmentRegistry,
        audit: MemoryAuditLog,
        utopia_factory: Callable[[str], UtopiaClient],
        worker: HeadroomWorker,
        inflight: InFlightTracker,
        probe: Callable[[], dict[str, bool]],
        inspector: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.audit = audit
        self.utopia_factory = utopia_factory
        self.worker = worker
        self.inflight = inflight
        self.probe = probe
        self.inspector = inspector

    def _require_admin(self, admin_token: str | None) -> None:
        if not admin_token or admin_token != self.settings.admin_token:
            raise AccessDenied("admin token required")

    def health(self) -> dict[str, Any]:
        probes = self.probe()
        ok = all(probes.values())
        return {
            "status": "ok" if ok else "degraded",
            "product": PRODUCT.lower(),
            "modules": {
                "ontology": ONTOLOGY_MODULE,
                "token_optimization": TOKEN_MODULE,
            },
            **probes,
        }

    def enroll(self, body: dict[str, Any], admin_token: str | None) -> dict[str, Any]:
        self._require_admin(admin_token)
        granted = frozenset(body["granted_kb_ids"])
        allowed = restricted_base_ids(self.settings.data_dir)
        # Empty allowed means bootstrap.json is absent: unit tests / pre-bootstrap.
        if allowed and not granted <= allowed:
            raise AccessDenied(
                "Release A enrollments may only grant Production or Staging"
            )
        identity = Identity(body["owner_id"], body["agent_id"], body["credential_id"])
        operations = frozenset(Operation(op) for op in body["operations"])
        record = self.registry.activate(
            identity,
            utopia_user_id=body.get("utopia_user_id"),
            utopia_token_id=body.get("utopia_token_id"),
            granted_kb_ids=granted,
            operations=operations,
            utopia_pat=body.get("utopia_pat"),
        )
        return {
            "status": record.status.value,
            "granted_kb_ids": sorted(record.granted_kb_ids),
            "utopia_user_id": record.utopia_user_id,
            "utopia_token_id": record.utopia_token_id,
        }

    def revoke(self, body: dict[str, Any], admin_token: str | None) -> dict[str, Any]:
        self._require_admin(admin_token)
        identity = Identity(body["owner_id"], body["agent_id"], body["credential_id"])
        cancelled = self.inflight.cancel_identity(identity)
        record = self.registry.revoke(identity)
        return {
            "status": record.status.value,
            "cancelled_in_flight": cancelled,
        }

    def list_enrollments(self, admin_token: str | None) -> list[dict[str, Any]]:
        self._require_admin(admin_token)
        return self.registry.public_records()

    def ask(self, body: dict[str, Any]) -> dict[str, Any]:
        identity = Identity(body["owner_id"], body["agent_id"], body["credential_id"])
        enrollment = self.registry.lookup(identity)
        session_kbs = frozenset(body.get("session_kb_ids") or ())
        session = Session(body["session_id"], identity, session_kbs)
        kb_id = body["kb_id"]
        tool = body["tool"]
        arguments = body.get("arguments") or {}
        operation = operation_for(tool)
        request_id = body.get("request_id") or str(uuid.uuid4())
        if self.inflight.is_cancelled(request_id):
            raise AccessDenied("request cancelled")
        authorize(enrollment, session, kb_id=kb_id, operation=operation)
        try:
            token = self.registry.token_for(identity)
        except UnknownIdentity as exc:
            raise AccessDenied(str(exc)) from exc
        self.inflight.start(request_id, identity)
        try:
            if self.inflight.is_cancelled(request_id):
                self.audit.append(
                    build_event(
                        owner_id=identity.owner_id,
                        agent_id=identity.agent_id,
                        credential_id=identity.credential_id,
                        session_id=session.session_id,
                        tool=tool,
                        kb_id=kb_id,
                        status="cancelled",
                        args=arguments,
                    )
                )
                raise AccessDenied("request cancelled")
            utopia = self.utopia_factory(token)
            raw, compressed = authorized_then_compress(
                enrollment=enrollment,
                session=session,
                kb_id=kb_id,
                tool=tool,
                arguments=arguments,
                utopia=utopia,
                audit=self.audit,
                worker=self.worker,
                model=self.settings.model,
                kind=CallKind.AGENT_CHAT,
                operation=operation,
            )
            return {
                "request_id": request_id,
                "utopia": raw,
                "compressed": asdict(compressed),
                "text": reply_text(compressed),
            }
        finally:
            self.inflight.finish(request_id)

    def tools(self) -> dict[str, Any]:
        return tools_manifest()

    def catalog(self, body: dict[str, Any]) -> dict[str, Any]:
        identity = Identity(body["owner_id"], body["agent_id"], body["credential_id"])
        enrollment = self.registry.lookup(identity)
        active = enrollment.status is EnrollmentStatus.ACTIVE
        return {
            "status": enrollment.status.value,
            "granted_kb_ids": sorted(enrollment.granted_kb_ids) if active else [],
            "operations": sorted(op.value for op in enrollment.operations),
            "tools": tools_for_operations(enrollment.operations) if active else [],
        }

    def start_session(self, body: dict[str, Any]) -> dict[str, Any]:
        identity = Identity(body["owner_id"], body["agent_id"], body["credential_id"])
        enrollment = self.registry.lookup(identity)
        if enrollment.status is EnrollmentStatus.PENDING:
            raise AccessDenied("pending enrollment: no private knowledge")
        if enrollment.status is EnrollmentStatus.REVOKED:
            raise AccessDenied("credential revoked")
        session = Session(
            body.get("session_id") or str(uuid.uuid4()),
            identity,
            frozenset(body.get("kb_ids") or ()),
        )
        allowed = effective_kb_ids(enrollment, session)
        return {"session_id": session.session_id, "kb_ids": sorted(allowed)}

    def readiness(self) -> dict[str, Any]:
        body = self.health()
        inspected = self.inspector() if self.inspector else {"settings": {}, "kbs": {}}
        chat_url = (inspected.get("settings") or {}).get("chat_base_url")
        body["chat_base_url_is_headroom"] = chat_url_is_headroom(chat_url)
        state = load_bootstrap_state(self.settings.data_dir)
        bases = []
        for name, key in (
            ("production", "production_kb_id"),
            ("staging", "staging_kb_id"),
        ):
            kb_id = state.get(key)
            if not kb_id:
                continue
            raw = (inspected.get("kbs") or {}).get(kb_id, {})
            row = {"name": name, "kb_id": kb_id}
            row.update(normalize_kb_readiness(raw))
            bases.append(row)
        body["bases"] = bases
        body["blockers"] = extraction_blockers(
            chat_base_url_is_headroom=body["chat_base_url_is_headroom"],
            bases=bases,
        )
        return body
