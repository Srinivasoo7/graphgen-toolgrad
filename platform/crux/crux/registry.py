"""Enrollment registry. Unknown IDs stay pending until Utopia ids bind.

Pass path= to persist in SQLite so enrollments survive process restart.
"""

from __future__ import annotations

import sqlite3
from typing import Dict

from crux.access import Enrollment, EnrollmentStatus, Identity, Operation


class UnknownIdentity(Exception):
    pass


class EnrollmentRegistry:
    def __init__(self, path: str | None = None) -> None:
        self._path = path
        self._records: Dict[Identity, Enrollment] = {}
        self._pats: Dict[Identity, str] = {}
        if path:
            self._init_db()
            self._load()

    def _connect(self) -> sqlite3.Connection:
        assert self._path is not None
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS enrollments (
                    owner_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    credential_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    granted_kb_ids TEXT NOT NULL,
                    operations TEXT NOT NULL,
                    utopia_user_id TEXT,
                    utopia_token_id TEXT,
                    utopia_pat TEXT,
                    PRIMARY KEY (owner_id, agent_id, credential_id)
                )
                """
            )

    def _load(self) -> None:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM enrollments").fetchall()
        for row in rows:
            identity = Identity(row["owner_id"], row["agent_id"], row["credential_id"])
            kbs = frozenset(x for x in (row["granted_kb_ids"] or "").split(",") if x)
            ops = frozenset(
                Operation(x) for x in (row["operations"] or "").split(",") if x
            )
            self._records[identity] = Enrollment(
                identity=identity,
                status=EnrollmentStatus(row["status"]),
                granted_kb_ids=kbs,
                operations=ops,
                utopia_user_id=row["utopia_user_id"],
                utopia_token_id=row["utopia_token_id"],
            )
            if row["utopia_pat"]:
                self._pats[identity] = row["utopia_pat"]

    def _persist(self, record: Enrollment, utopia_pat: str | None) -> None:
        if not self._path:
            return
        pat = utopia_pat if utopia_pat is not None else self._pats.get(record.identity)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO enrollments (
                    owner_id, agent_id, credential_id, status, granted_kb_ids,
                    operations, utopia_user_id, utopia_token_id, utopia_pat
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.identity.owner_id,
                    record.identity.agent_id,
                    record.identity.credential_id,
                    record.status.value,
                    ",".join(sorted(record.granted_kb_ids)),
                    ",".join(sorted(op.value for op in record.operations)),
                    record.utopia_user_id,
                    record.utopia_token_id,
                    pat,
                ),
            )

    def lookup(self, identity: Identity) -> Enrollment:
        existing = self._records.get(identity)
        if existing is not None:
            return existing
        return Enrollment(
            identity=identity,
            status=EnrollmentStatus.PENDING,
            granted_kb_ids=frozenset(),
            operations=frozenset(),
        )

    def activate(
        self,
        identity: Identity,
        *,
        utopia_user_id: str | None,
        utopia_token_id: str | None,
        granted_kb_ids: frozenset[str],
        operations: frozenset[Operation],
        utopia_pat: str | None = None,
    ) -> Enrollment:
        if not utopia_user_id or not utopia_token_id:
            raise UnknownIdentity("activate requires verified Utopia user and token ids")
        record = Enrollment(
            identity=identity,
            status=EnrollmentStatus.ACTIVE,
            granted_kb_ids=granted_kb_ids,
            operations=operations,
            utopia_user_id=utopia_user_id,
            utopia_token_id=utopia_token_id,
        )
        self._records[identity] = record
        if utopia_pat:
            self._pats[identity] = utopia_pat
        self._persist(record, utopia_pat)
        return record

    def revoke(self, identity: Identity) -> Enrollment:
        current = self.lookup(identity)
        revoked = Enrollment(
            identity=identity,
            status=EnrollmentStatus.REVOKED,
            granted_kb_ids=current.granted_kb_ids,
            operations=current.operations,
            utopia_user_id=current.utopia_user_id,
            utopia_token_id=current.utopia_token_id,
        )
        self._records[identity] = revoked
        self._pats.pop(identity, None)
        self._persist(revoked, None)
        return revoked

    def token_for(self, identity: Identity) -> str:
        record = self.lookup(identity)
        if record.status != EnrollmentStatus.ACTIVE:
            raise UnknownIdentity("no active Utopia credential for this identity")
        pat = self._pats.get(identity)
        if not pat:
            raise UnknownIdentity("no Utopia PAT bound for this identity")
        return pat

    def public_records(self) -> list[dict]:
        rows = []
        for identity, record in self._records.items():
            rows.append(
                {
                    "owner_id": identity.owner_id,
                    "agent_id": identity.agent_id,
                    "credential_id": identity.credential_id,
                    "status": record.status.value,
                    "granted_kb_ids": sorted(record.granted_kb_ids),
                    "operations": sorted(op.value for op in record.operations),
                    "utopia_user_id": record.utopia_user_id,
                    "utopia_token_id": record.utopia_token_id,
                }
            )
        return rows
