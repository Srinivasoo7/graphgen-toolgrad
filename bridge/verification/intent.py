"""Deterministic check that the question asks for the action the chain performs.

A lookup question on a chain that only creates a record is rejected. The
2026-09-20 ITSM row ("What is the status of INC-1042?" via create_ticket)
is the canonical failure.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

_LOOKUP = re.compile(
    r"\b(status of|what is the status|what is status|look ?up|lookup|"
    r"show me|get the|retrieve)\b",
    re.IGNORECASE,
)
_CREATE = re.compile(r"\b(create|file|open|raise|submit|add)\b", re.IGNORECASE)
_CLOSE = re.compile(r"\b(close|resolve|cancel|delete|remove)\b", re.IGNORECASE)

_READ = {"get", "fetch", "search", "find", "list", "lookup", "read", "retrieve", "show"}
_CREATE_VERBS = {"create", "open", "add", "file", "post", "insert", "submit", "raise"}
_UPDATE = {"update", "modify", "set", "append", "change", "edit"}
_DELETE = {"delete", "remove", "close", "cancel", "resolve"}


def tool_effect(tool_name: str) -> str:
    """Classify a tool by the first token of its name. Unknown stays UNKNOWN."""
    verb = str(tool_name or "").split("_")[0].lower()
    if verb in _READ:
        return "READ"
    if verb in _CREATE_VERBS:
        return "CREATE"
    if verb in _UPDATE:
        return "UPDATE"
    if verb in _DELETE:
        return "DELETE"
    return "UNKNOWN"


def question_intent(question: str) -> Optional[str]:
    """Return READ, CREATE, or DELETE when the question states one. Else None."""
    text = question or ""
    if _LOOKUP.search(text):
        return "READ"
    if _CLOSE.search(text):
        return "DELETE"
    if _CREATE.search(text):
        return "CREATE"
    return None


def intent_aligned(question: str, tool_names: Sequence[str]) -> bool:
    """True when the question does not ask for a different action than the tools.

    An unrecognized question is not failed here. A recognized mismatch is.
    """
    intent = question_intent(question)
    if intent is None:
        return True
    effects = [tool_effect(name) for name in tool_names if name]
    if not effects:
        return False
    if intent == "READ":
        return "CREATE" not in effects or "READ" in effects
    if intent == "CREATE":
        return "CREATE" in effects
    if intent == "DELETE":
        return "DELETE" in effects
    return True


def tool_names_of(chain: Iterable[dict]) -> List[str]:
    names = []
    for step in chain:
        if isinstance(step, dict) and step.get("tool"):
            names.append(step["tool"])
        elif isinstance(step, str):
            names.append(step)
    return names
