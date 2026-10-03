"""In-flight request cancel. Already-sent model context cannot be recalled."""

from __future__ import annotations

from typing import Dict

from crux.access import Identity

# Revoke/cancel stops the next instrumented call. Tokens already given to a
# model provider stay there; Crux does not claim it can pull them back.
ALREADY_SENT_CANNOT_RECALL = True


class InFlightTracker:
    def __init__(self) -> None:
        self._open: Dict[str, Identity] = {}
        self._cancelled: set[str] = set()

    def start(self, request_id: str, identity: Identity) -> None:
        self._open[request_id] = identity

    def cancel_identity(self, identity: Identity) -> int:
        cancelled = 0
        for request_id, owner in list(self._open.items()):
            if owner == identity:
                self._cancelled.add(request_id)
                del self._open[request_id]
                cancelled += 1
        return cancelled

    def is_cancelled(self, request_id: str) -> bool:
        return request_id in self._cancelled

    def is_open(self, request_id: str) -> bool:
        return request_id in self._open

    def finish(self, request_id: str) -> None:
        self._open.pop(request_id, None)
