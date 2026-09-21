"""PII redaction for generated QA pairs.

Enterprise tool outputs regularly contain emails, phone numbers, API keys,
and other sensitive strings; generated training data must not launder them
into a dataset. :func:`redact_qa` scrubs the user-visible text fields of a
QA record (question, answer draft, and tool result previews) and reports
what it replaced. Patterns are conservative by default and configurable —
an enterprise can extend :data:`DEFAULT_PATTERNS` with its own shapes.

This is a safety net, not a guarantee: it runs before packaging and every
redaction is counted in the run report.
"""

from __future__ import annotations

import copy
import re
from typing import Dict, List, Tuple

# Each entry: (label, compiled pattern, replacement template).
# Order matters: specific credential shapes run before the greedy numeric
# rules (phone/ssn/card), so a phone rule can't mangle the digit run
# inside an API key.
DEFAULT_PATTERNS: List[Tuple[str, "re.Pattern", str]] = [
    (
        "email",
        re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "[REDACTED_EMAIL]",
    ),
    (
        "api_key",
        re.compile(r"\bsk-[A-Za-z0-9\-_]{20,}\b"),
        "[REDACTED_KEY]",
    ),
    (
        "openrouter_key",
        re.compile(r"sk-or-[A-Za-z0-9\-_]{16,}"),
        "[REDACTED_KEY]",
    ),
    (
        "aws_access_key",
        re.compile(r"AKIA[0-9A-Z]{16}"),
        "[REDACTED_AWS_KEY]",
    ),
    (
        "private_key_block",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        "api_key_assignment",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|client[_-]?secret|access[_-]?token)\b"
            r"\s*[:=]\s*['\"]?[\w\-.]{12,}['\"]?"
        ),
        "[REDACTED_CREDENTIAL]",
    ),
    (
        "bearer_token",
        re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]{16,}={0,2}"),
        "[REDACTED_TOKEN]",
    ),
    (
        "phone",
        re.compile(
            r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}(?!\d)"
        ),
        "[REDACTED_PHONE]",
    ),
    (
        "ssn",
        re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
        "[REDACTED_SSN]",
    ),
    (
        "credit_card",
        re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"),
        "[REDACTED_CARD]",
    ),
]

# QA text fields that get scrubbed. ``chain`` steps keep their tool names
# and inputs (needed for verification) but result previews are scrubbed.
_QA_TEXT_FIELDS = ("question", "answer_draft")


def redact_text(
    text: str,
    patterns: List[Tuple[str, "re.Pattern", str]] = DEFAULT_PATTERNS,
) -> Tuple[str, Dict[str, int]]:
    """Scrub ``text``; returns (scrubbed_text, {label: replacements})."""
    hits: Dict[str, int] = {}
    if not text:
        return text, hits
    for label, pattern, replacement in patterns:
        text, n = pattern.subn(replacement, text)
        if n:
            hits[label] = hits.get(label, 0) + n
    return text, hits


def _merge_hits(into: Dict[str, int], extra: Dict[str, int]) -> None:
    for label, n in extra.items():
        into[label] = into.get(label, 0) + n


def redact_qa(
    qa: dict,
    patterns: List[Tuple[str, "re.Pattern", str]] = DEFAULT_PATTERNS,
) -> Tuple[dict, Dict[str, int]]:
    """Return a redacted copy of ``qa`` plus aggregate hit counts."""
    scrubbed = copy.deepcopy(qa)
    hits: Dict[str, int] = {}
    for field_name in _QA_TEXT_FIELDS:
        value = scrubbed.get(field_name)
        if isinstance(value, str):
            scrubbed[field_name], field_hits = redact_text(value, patterns)
            _merge_hits(hits, field_hits)
    chain = scrubbed.get("chain")
    if isinstance(chain, list):
        for step in chain:
            preview = step.get("result_preview") if isinstance(step, dict) else None
            if isinstance(preview, str):
                step["result_preview"], step_hits = redact_text(preview, patterns)
                _merge_hits(hits, step_hits)
    if hits:
        provenance = scrubbed.setdefault("provenance", {})
        provenance["pii_redactions"] = hits
    return scrubbed, hits
