"""In-flight cancel: stop further instrumented calls. Already-sent context is not recalled."""

from crux.access import Identity
from crux.inflight import ALREADY_SENT_CANNOT_RECALL, InFlightTracker


def test_revoke_cancels_open_requests_for_that_identity():
    tracker = InFlightTracker()
    identity = Identity("owner-1", "agent-a", "cred-a")
    other = Identity("owner-1", "agent-b", "cred-b")
    tracker.start("req-1", identity)
    tracker.start("req-2", other)

    cancelled = tracker.cancel_identity(identity)

    assert cancelled == 1
    assert tracker.is_cancelled("req-1") is True
    assert tracker.is_cancelled("req-2") is False
    assert ALREADY_SENT_CANNOT_RECALL is True


def test_finish_clears_an_open_request():
    tracker = InFlightTracker()
    identity = Identity("owner-1", "agent-a", "cred-a")
    tracker.start("req-1", identity)
    tracker.finish("req-1")
    assert tracker.is_cancelled("req-1") is False
    assert tracker.is_open("req-1") is False
