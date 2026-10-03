"""Create two restricted Utopia bases and bind Crux enrollments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import httpx

from crux.access import EnrollmentStatus, Identity, Operation
from crux.registry import EnrollmentRegistry, UnknownIdentity


class UtopiaAdmin:
    """Talks to Utopia's admin HTTP API. Not a second access engine."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token: str | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def register_or_login(
        self, email: str, password: str, display_name: str, org_name: str
    ) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            registered = client.post(
                f"{self.base_url}/api/v1/auth/register",
                json={
                    "email": email,
                    "password": password,
                    "display_name": display_name,
                    "org_name": org_name,
                },
            )
            if registered.status_code >= 400:
                registered = client.post(
                    f"{self.base_url}/api/v1/auth/login",
                    json={"email": email, "password": password},
                )
            registered.raise_for_status()
            data = registered.json()
            self.token = data["token"]
            if "workspace" not in data:
                workspaces = client.get(
                    f"{self.base_url}/api/v1/workspaces",
                    headers=self._headers(),
                )
                workspaces.raise_for_status()
                listed = workspaces.json()
                data["workspace"] = listed[0]
            return data

    def create_restricted_kb(
        self, workspace_id: str, name: str, ontology_packs: List[str]
    ) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            listed = client.get(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/kbs",
                headers=self._headers(),
            )
            listed.raise_for_status()
            for kb in listed.json():
                if kb.get("name") == name:
                    return kb
            created = client.post(
                f"{self.base_url}/api/v1/workspaces/{workspace_id}/kbs",
                headers=self._headers(),
                json={
                    "name": name,
                    "visibility": "restricted",
                    "ontology_packs": ontology_packs,
                },
            )
            created.raise_for_status()
            return created.json()

    def issue_token(self, name: str, scope: str, kb_ids: List[str]) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            issued = client.post(
                f"{self.base_url}/api/v1/me/tokens",
                headers=self._headers(),
                json={
                    "name": name,
                    "scope": scope,
                    "kb_ids": kb_ids,
                    "expires_in_days": 0,
                },
            )
            issued.raise_for_status()
            return issued.json()


@dataclass
class BootstrapResult:
    production_kb_id: str
    staging_kb_id: str
    workspace_id: str
    user_id: str


def bootstrap_release_a(
    admin: UtopiaAdmin,
    registry: EnrollmentRegistry,
    *,
    email: str,
    password: str,
    owner_id: str,
) -> BootstrapResult:
    auth = admin.register_or_login(email, password, "Crux admin", "Crux")
    workspace_id = auth["workspace"]["id"]
    user_id = auth["user"]["id"]
    packs = ["schema-org"]
    production = admin.create_restricted_kb(workspace_id, "Production", packs)
    staging = admin.create_restricted_kb(workspace_id, "Staging", packs)
    prod_identity = Identity(owner_id, "agent-prod", "cred-prod")
    staging_identity = Identity(owner_id, "agent-staging", "cred-staging")
    if _already_bound(
        registry, prod_identity, staging_identity, production["id"], staging["id"]
    ):
        return BootstrapResult(
            production_kb_id=production["id"],
            staging_kb_id=staging["id"],
            workspace_id=workspace_id,
            user_id=user_id,
        )
    prod_token = admin.issue_token("crux-prod", "read", [production["id"]])
    staging_token = admin.issue_token("crux-staging", "write", [staging["id"]])
    registry.activate(
        prod_identity,
        utopia_user_id=user_id,
        utopia_token_id=prod_token["info"]["id"],
        granted_kb_ids=frozenset({production["id"]}),
        operations=frozenset({Operation.READ}),
        utopia_pat=prod_token["token"],
    )
    registry.activate(
        staging_identity,
        utopia_user_id=user_id,
        utopia_token_id=staging_token["info"]["id"],
        granted_kb_ids=frozenset({staging["id"]}),
        operations=frozenset({Operation.READ, Operation.PROPOSE}),
        utopia_pat=staging_token["token"],
    )
    return BootstrapResult(
        production_kb_id=production["id"],
        staging_kb_id=staging["id"],
        workspace_id=workspace_id,
        user_id=user_id,
    )


def _already_bound(
    registry: EnrollmentRegistry,
    prod_identity: Identity,
    staging_identity: Identity,
    production_kb_id: str,
    staging_kb_id: str,
) -> bool:
    prod = registry.lookup(prod_identity)
    staging = registry.lookup(staging_identity)
    if prod.status is not EnrollmentStatus.ACTIVE:
        return False
    if staging.status is not EnrollmentStatus.ACTIVE:
        return False
    if prod.granted_kb_ids != frozenset({production_kb_id}):
        return False
    if staging.granted_kb_ids != frozenset({staging_kb_id}):
        return False
    try:
        registry.token_for(prod_identity)
        registry.token_for(staging_identity)
    except UnknownIdentity:
        return False
    return True
