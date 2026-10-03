"""Bootstrap two restricted Utopia bases and bind Crux enrollments."""

from crux.access import EnrollmentStatus, Identity, Operation
from crux.bootstrap import UtopiaAdmin, bootstrap_release_a
from crux.registry import EnrollmentRegistry


class FakeAdmin(UtopiaAdmin):
    def __init__(self):
        self.calls = []
        self._kbs = {}

    def register_or_login(self, email, password, display_name, org_name):
        self.calls.append(("auth", email, org_name))
        return {
            "user": {"id": "user-1", "email": email, "is_admin": True},
            "workspace": {"id": "ws-1"},
            "token": "session-jwt",
        }

    def create_restricted_kb(self, workspace_id, name, ontology_packs):
        self.calls.append(("kb", workspace_id, name, tuple(ontology_packs)))
        kb_id = "kb-" + name.lower()
        self._kbs[name] = kb_id
        return {"id": kb_id, "name": name, "visibility": "restricted"}

    def issue_token(self, name, scope, kb_ids):
        self.calls.append(("token", name, scope, tuple(kb_ids)))
        return {"token": "pat-" + name, "info": {"id": "tok-" + name, "scope": scope}}


def test_bootstrap_creates_two_restricted_bases_and_enrollments():
    registry = EnrollmentRegistry()
    admin = FakeAdmin()
    result = bootstrap_release_a(
        admin,
        registry,
        email="admin@crux.local",
        password="change-me-now",
        owner_id="owner-1",
    )
    assert result.production_kb_id == "kb-production"
    assert result.staging_kb_id == "kb-staging"
    kb_names = [c[2] for c in admin.calls if c[0] == "kb"]
    assert kb_names == ["Production", "Staging"]
    assert all(c[3] == ("schema-org",) for c in admin.calls if c[0] == "kb")

    prod = registry.lookup(Identity("owner-1", "agent-prod", "cred-prod"))
    staging = registry.lookup(Identity("owner-1", "agent-staging", "cred-staging"))
    assert prod.status is EnrollmentStatus.ACTIVE
    assert prod.granted_kb_ids == frozenset({"kb-production"})
    assert prod.operations == frozenset({Operation.READ})
    assert staging.granted_kb_ids == frozenset({"kb-staging"})
    assert Operation.PROPOSE in staging.operations
    assert registry.token_for(Identity("owner-1", "agent-prod", "cred-prod")) == "pat-crux-prod"


def test_bootstrap_second_run_reuses_enrollments_and_does_not_issue_tokens():
    registry = EnrollmentRegistry()
    admin = FakeAdmin()
    kwargs = dict(
        email="admin@crux.local",
        password="change-me-now",
        owner_id="owner-1",
    )
    first = bootstrap_release_a(admin, registry, **kwargs)
    token_calls = [c for c in admin.calls if c[0] == "token"]
    assert len(token_calls) == 2
    second = bootstrap_release_a(admin, registry, **kwargs)
    assert second.production_kb_id == first.production_kb_id
    assert [c for c in admin.calls if c[0] == "token"] == token_calls
    assert registry.token_for(Identity("owner-1", "agent-prod", "cred-prod")) == "pat-crux-prod"
