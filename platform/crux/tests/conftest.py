"""Shared fixtures (ECC python-testing)."""

import pytest

from crux.access import Enrollment, EnrollmentStatus, Identity, Operation


PROD = "kb-production"
STAGING = "kb-staging"


@pytest.fixture
def owner_agent_a():
    return Identity(owner_id="owner-1", agent_id="agent-a", credential_id="cred-a")


@pytest.fixture
def active_prod_reader(owner_agent_a):
    return Enrollment(
        identity=owner_agent_a,
        status=EnrollmentStatus.ACTIVE,
        granted_kb_ids=frozenset({PROD}),
        operations=frozenset({Operation.READ}),
        utopia_user_id="utp-user-1",
        utopia_token_id="utp-tok-1",
    )
