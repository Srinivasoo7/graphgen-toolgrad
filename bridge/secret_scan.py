"""Secret scan: the output gate that keeps credentials out of datasets.

Before a dataset is packaged, every generated file is scanned for
high-confidence secret shapes (private keys, cloud access keys, bearer
tokens, ``api_key=...`` assignments, ``sk-or-`` keys). Any finding fails
the run — a dataset that leaks a credential is worse than no dataset.

This is intentionally stricter than :mod:`bridge.redaction` (which scrubs
PII-shaped strings from QA text): the scan looks at the final bytes on
disk, including ledgers and reports, and treats a hit as a defect.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List


@dataclass
class SecretFinding:
    path: str
    line_no: int
    kind: str
    excerpt: str


_PATTERNS: List[tuple] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("aws_secret", re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{30,}")),
    ("openrouter_key", re.compile(r"sk-or-[A-Za-z0-9\-_]{16,}")),
    ("generic_api_key", re.compile(r"(?i)\b(api[_-]?key)\b\s*[:=]\s*['\"]?[\w\-.]{20,}['\"]?")),
    ("bearer_token", re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]{20,}={0,2}")),
    ("password_assignment", re.compile(r"(?i)\bpassword\b\s*[:=]\s*['\"]?\S{8,}['\"]?")),
]

_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules"}
_SKIP_EXTS = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip"}
_MAX_FILE_BYTES = 5 * 1024 * 1024


def _excerpt(line: str, match: "re.Match") -> str:
    # Show context without echoing the secret itself.
    start = max(0, match.start() - 20)
    return line[start:match.start()] + "<SECRET>" + line[match.end():match.end() + 10]


def scan_text(text: str, path: str = "<text>") -> List[SecretFinding]:
    findings: List[SecretFinding] = []
    for i, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in _PATTERNS:
            match = pattern.search(line)
            if match:
                findings.append(SecretFinding(path, i, kind, _excerpt(line, match)))
    return findings


def scan_file(path: str) -> List[SecretFinding]:
    try:
        if os.path.getsize(path) > _MAX_FILE_BYTES:
            return []
    except OSError:
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError):
        return []
    return scan_text(text, path)


def scan_directory(root: str) -> List[SecretFinding]:
    """Recursively scan ``root``; returns all findings (empty = clean)."""
    findings: List[SecretFinding] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in _SKIP_EXTS:
                continue
            findings.extend(scan_file(os.path.join(dirpath, fname)))
    return findings
