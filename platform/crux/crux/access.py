"""Utopia-shaped access. Session may only restrict already-granted bases."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EnrollmentStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    REVOKED = "revoked"


class Operation(str, Enum):
    READ = "read"
    PROPOSE = "propose"


@dataclass(frozen=True)
class Identity:
    owner_id: str
    agent_id: str
    credential_id: str


@dataclass
class Enrollment:
    identity: Identity
    status: EnrollmentStatus
    granted_kb_ids: frozenset[str]
    operations: frozenset[Operation]
    utopia_user_id: str | None = None
    utopia_token_id: str | None = None


@dataclass(frozen=True)
class Session:
    session_id: str
    identity: Identity
    kb_ids: frozenset[str] = field(default_factory=frozenset)


class AccessDenied(Exception):
    pass


def effective_kb_ids(enrollment: Enrollment, session: Session) -> frozenset[str]:
    if enrollment.status != EnrollmentStatus.ACTIVE:
        return frozenset()
    if session.identity != enrollment.identity:
        return frozenset()
    extra = session.kb_ids - enrollment.granted_kb_ids
    if extra:
        raise AccessDenied(
            "session cannot grant bases outside Utopia credential scope"
        )
    if not session.kb_ids:
        return enrollment.granted_kb_ids
    return session.kb_ids & enrollment.granted_kb_ids


def authorize(
    enrollment: Enrollment,
    session: Session,
    *,
    kb_id: str,
    operation: Operation,
) -> None:
    if enrollment.status == EnrollmentStatus.PENDING:
        raise AccessDenied("pending enrollment: no private knowledge")
    if enrollment.status == EnrollmentStatus.REVOKED:
        raise AccessDenied("credential revoked")
    if claimed_id_untrusted(enrollment):
        raise AccessDenied("claimed identity is not bound to a verified credential")
    allowed = effective_kb_ids(enrollment, session)
    if kb_id not in allowed:
        raise AccessDenied("knowledge base not in effective grants")
    if operation not in enrollment.operations:
        raise AccessDenied(f"operation {operation.value} not delegated")


def claimed_id_untrusted(enrollment: Enrollment) -> bool:
    return enrollment.utopia_user_id is None or enrollment.utopia_token_id is None
