from crux.access import (
    AccessDenied,
    Enrollment,
    EnrollmentStatus,
    Identity,
    Operation,
    Session,
    authorize,
)
from crux.leak import (
    LEAK_SURFACES,
    InMemoryStore,
    RetrievalBundle,
    assert_no_leak,
    bundle_from_mcp_text,
)
import pytest


PROD = "kb-production"
STAGING = "kb-staging"


def test_unauthorized_fact_cannot_leak_across_bases():
    store = InMemoryStore()
    store.put(
        STAGING,
        RetrievalBundle(
            graph_relationships=[{"s": "svc", "p": "ownedBy", "o": "team"}],
            search_hit_counts=4,
            summaries=["staging outage"],
            citations=["doc-9"],
            source_documents=["runbook.md"],
            derived_facts=[{"kind": "inferred"}],
        ),
    )
    identity = Identity("owner", "agent-a", "cred-a")
    enrollment = Enrollment(
        identity=identity,
        status=EnrollmentStatus.ACTIVE,
        granted_kb_ids=frozenset({PROD}),
        operations=frozenset({Operation.READ}),
        utopia_user_id="u1",
        utopia_token_id="t1",
    )
    session = Session("s1", identity)
    with pytest.raises(AccessDenied):
        authorize(enrollment, session, kb_id=STAGING, operation=Operation.READ)
    allowed = frozenset()
    bundle = store.retrieve(allowed, STAGING)
    assert_no_leak(bundle)


def test_authorized_base_still_returns_its_own_facts():
    store = InMemoryStore()
    store.put(PROD, RetrievalBundle(search_hit_counts=2, summaries=["prod ok"]))
    bundle = store.retrieve(frozenset({PROD}), PROD)
    assert bundle.search_hit_counts == 2
    assert bundle.summaries == ["prod ok"]


def test_bundle_treats_empty_entity_list_as_no_leak():
    assert_no_leak(bundle_from_mcp_text("No matching entities."))
    assert_no_leak(
        bundle_from_mcp_text(
            "Invalid entity_id (expected the uuid returned by find_entities)."
        )
    )
    assert_no_leak(bundle_from_mcp_text("Entity not found."))


def test_bundle_records_graph_relationships_and_derived_facts():
    text = (
        "publishes → data classification [90%]\n"
        "publishes → lineage [90%]"
    )
    bundle = bundle_from_mcp_text(text)
    assert bundle.graph_relationships
    assert bundle.derived_facts
    assert any("lineage" in str(item) for item in bundle.graph_relationships)
    assert bundle.search_hit_counts == 0


def test_bundle_records_find_entities_rows_as_derived_facts():
    text = (
        "01a07cbf-668c-73b1-a559-1e5002f5710f | Azure Purview | Service | 4 facts"
    )
    bundle = bundle_from_mcp_text(text)
    assert bundle.derived_facts
    assert any("Azure Purview" in str(item) for item in bundle.derived_facts)


def test_leak_surfaces_are_the_acceptance_set():
    assert LEAK_SURFACES == (
        "graph_relationships",
        "search_hit_counts",
        "summaries",
        "citations",
        "source_documents",
        "derived_facts",
    )
