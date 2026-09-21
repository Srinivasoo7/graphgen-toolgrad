#!/usr/bin/env python3
"""Demo enterprise ITSM MCP server for the graphgen-toolgrad factory run.

Models a small IT service-desk tool surface: a knowledge base of support
articles, a ticket system, and an asset inventory. State is in-memory for
the life of the server process (one factory run).

Tools:
  search_kb(query, top_k)      - token-overlap search over KB articles
  get_kb_article(article_id)   - full text of one KB article
  create_ticket(...)           - open a ticket, returns ticket_id
  get_ticket(ticket_id)        - fetch a ticket
  update_ticket(ticket_id, status, note) - change status / add a note
  list_open_tickets()          - all tickets not closed/resolved
  lookup_asset(asset_tag)      - hardware inventory lookup
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("itsm")

ARTICLES: Dict[str, Dict[str, str]] = {
    "KB-001": {
        "title": "VPN error 809 on Windows: enable IKEv2 passthrough",
        "category": "Network / VPN",
        "tags": "vpn error 809 windows ikev2 firewall udp",
        "body": (
            "Symptom: The corporate VPN client fails on Windows with error 809 "
            "('The network connection between your computer and the VPN server "
            "could not be established').\n"
            "Cause: IKEv2 UDP ports 500 and 4500 are blocked by the local firewall "
            "or the home router.\n"
            "Resolution steps:\n"
            "1. Open Windows Defender Firewall > Advanced settings > Inbound Rules.\n"
            "2. Allow UDP ports 500 and 4500 for the VPN client executable.\n"
            "3. On the home router, enable VPN/IPSec passthrough.\n"
            "4. Retry the connection. If it still fails, switch the client protocol "
            "from Automatic to IKEv2 explicitly and retry.\n"
            "Escalate to NetOps if the failure persists off the corporate network."
        ),
    },
    "KB-002": {
        "title": "Reset your password with Okta self-service",
        "category": "Identity / Access",
        "tags": "password reset okta self-service locked account",
        "body": (
            "Use Okta self-service to reset a forgotten password without calling "
            "the desk:\n"
            "1. Go to the Okta sign-in page and click 'Need help signing in' > "
            "'Forgot password'.\n"
            "2. Enter your corporate email and complete MFA verification.\n"
            "3. Choose a new password meeting the policy: 14+ characters, upper, "
            "lower, digit, symbol.\n"
            "Note: after a reset, re-authenticate on mobile mail and VPN. "
            "Accounts locked by too many attempts auto-unlock after 30 minutes; "
            "the desk can unlock immediately on verified identity."
        ),
    },
    "KB-003": {
        "title": "Printer shows offline: rejoin the follow-me print queue",
        "category": "Facilities / Printing",
        "tags": "printer offline follow-me print queue",
        "body": (
            "Symptom: the printer shows offline or jobs sit in the queue.\n"
            "Resolution steps:\n"
            "1. Confirm the printer has power and a network link light.\n"
            "2. On your laptop, remove the old direct printer and add "
            "'FollowMe-Print' from the software center.\n"
            "3. Badge in at any FollowMe printer to release your jobs.\n"
            "If jobs still stall, note the printer hostname (sticker on the "
            "front) and open a ticket; do not reinstall drivers manually."
        ),
    },
    "KB-004": {
        "title": "Requesting a new laptop for a new hire",
        "category": "Hardware / Provisioning",
        "tags": "laptop new hire provisioning onboarding hardware request",
        "body": (
            "Laptop requests for new hires go through the hardware catalog:\n"
            "1. The hiring manager submits a request at least 10 business days "
            "before the start date.\n"
            "2. Standard issue: 14-inch business laptop, 16 GB RAM, 512 GB SSD.\n"
            "3. Engineering roles may request the 32 GB RAM configuration with "
            "director approval.\n"
            "4. IT images the machine and ships it to the employee's home or "
            "holds it at the front desk.\n"
            "Track the request with the ticket number issued at submission."
        ),
    },
    "KB-005": {
        "title": "Outlook not syncing: rebuild the OST",
        "category": "Email / Collaboration",
        "tags": "outlook sync ost email stuck",
        "body": (
            "Symptom: Outlook shows 'Disconnected' or new mail never arrives.\n"
            "Resolution steps:\n"
            "1. Check the status bar: if it says 'Working Offline', toggle it off "
            "under Send/Receive.\n"
            "2. If still stuck, close Outlook, delete the .ost file under "
            "%LOCALAPPDATA%\\Microsoft\\Outlook, and reopen Outlook to rebuild.\n"
            "3. For shared mailboxes that stay stale, remove and re-add the "
            "mailbox under Account Settings.\n"
            "Do not delete the mailbox profile itself; the OST rebuild preserves "
            "all server-side mail."
        ),
    },
    "KB-006": {
        "title": "Corporate Wi-Fi certificate renewal (CORP-WIFI)",
        "category": "Network / Wi-Fi",
        "tags": "wifi certificate corp-wifi eap-tls renewal",
        "body": (
            "CORP-WIFI uses EAP-TLS certificates that expire yearly.\n"
            "Renewal steps:\n"
            "1. While on any network, open the Company Portal and install the "
            "'WiFi Certificate 2026' profile.\n"
            "2. Forget CORP-WIFI, then rejoin; authentication is automatic.\n"
            "3. On macOS, approve the new profile in System Settings > Profiles.\n"
            "Expiry warnings start 14 days out. If the portal shows no profile, "
            "open a ticket so Identity can re-push enrollment."
        ),
    },
    "KB-007": {
        "title": "Requesting software installation (admin approval)",
        "category": "Software / Licensing",
        "tags": "software install admin approval catalog license",
        "body": (
            "Approved software is installed from the Company Portal (self-service).\n"
            "For software not in the portal:\n"
            "1. Open a ticket with the software name, version, vendor, and "
            "business justification.\n"
            "2. Your manager approves in the ticket; Security reviews the vendor.\n"
            "3. Typical turnaround is 3 business days after approvals.\n"
            "Never install with local admin rights granted for another purpose; "
            "misuse is revoked and logged."
        ),
    },
    "KB-008": {
        "title": "Laptop running slow: first checks",
        "category": "Hardware / Performance",
        "tags": "slow laptop performance disk startup",
        "body": (
            "Before opening a ticket for a slow laptop, run these checks:\n"
            "1. Restart the machine (not sleep, a full restart).\n"
            "2. Open Task Manager / Activity Monitor: if disk is pinned at 100%, "
            "free at least 20% disk space.\n"
            "3. Disable unneeded login items and browser extensions.\n"
            "4. Confirm the EDR agent shows 'Healthy' in the tray; a failing "
            "agent can throttle the disk.\n"
            "If slowness persists with 20%+ free disk, open a ticket and include "
            "the asset tag."
        ),
    },
    "KB-009": {
        "title": "You received a phishing email: report it",
        "category": "Security / Phishing",
        "tags": "phishing suspicious email report security",
        "body": (
            "If an email looks suspicious:\n"
            "1. Do NOT click links or open attachments.\n"
            "2. Click the 'Report Phish' button in Outlook; this quarantines the "
            "message and alerts Security.\n"
            "3. If you already clicked, disconnect from VPN/Wi-Fi and call the "
            "security hotline immediately.\n"
            "Telltale signs: urgent tone, unexpected attachment, sender domain "
            "that is close-but-wrong, or a request for credentials or gift cards."
        ),
    },
    "KB-010": {
        "title": "VPN on macOS: import the Tunnelblick profile",
        "category": "Network / VPN",
        "tags": "vpn macos tunnelblick profile",
        "body": (
            "macOS users connect with Tunnelblick:\n"
            "1. Install Tunnelblick from the Company Portal.\n"
            "2. Download your personal .tblk profile from the VPN self-service "
            "page (requires MFA).\n"
            "3. Double-click the profile to import, then Connect from the menu "
            "bar icon.\n"
            "Profiles expire after 365 days; re-download from the same page. "
            "Do not share profiles between users."
        ),
    },
    "KB-011": {
        "title": "Requesting access to a shared drive",
        "category": "Identity / Access",
        "tags": "shared drive access permissions file share",
        "body": (
            "Shared-drive access needs the data owner's approval:\n"
            "1. Open a ticket naming the share (\\\\files\\dept-share) and your "
            "business need.\n"
            "2. The desk routes it to the share owner for approval.\n"
            "3. Access is provisioned within 1 business day of approval and "
            "reviewed quarterly.\n"
            "Contractors receive read-only access by default."
        ),
    },
    "KB-012": {
        "title": "BitLocker recovery key retrieval",
        "category": "Security / Encryption",
        "tags": "bitlocker recovery key encryption locked drive",
        "body": (
            "If Windows prompts for a BitLocker recovery key at boot:\n"
            "1. On another device, sign in to the device management portal and "
            "open your device to view the 48-digit key.\n"
            "2. Enter it carefully; 5 wrong attempts lock the volume further.\n"
            "3. After booting, the desk can suspend and re-enable BitLocker to "
            "clear the prompt loop.\n"
            "The desk can read keys only after verifying your identity; keys "
            "are never sent over chat."
        ),
    },
}

ASSETS: Dict[str, Dict[str, str]] = {
    "LT-1042": {"tag": "LT-1042", "type": "laptop", "model": "Business 14 G9",
                "os": "Windows 11 23H2", "assigned_to": "A. Rivera",
                "status": "in-use", "location": "Austin HQ, floor 3"},
    "LT-2210": {"tag": "LT-2210", "type": "laptop", "model": "Business 14 G9",
                "os": "Windows 11 23H2", "assigned_to": "J. Chen",
                "status": "in-use", "location": "Remote"},
    "LT-3305": {"tag": "LT-3305", "type": "laptop", "model": "MacBook Pro 14",
                "os": "macOS 15", "assigned_to": "S. Patel",
                "status": "in-use", "location": "Austin HQ, floor 2"},
    "LT-1188": {"tag": "LT-1188", "type": "laptop", "model": "Business 14 G9",
                "os": "Windows 11 23H2", "assigned_to": "Unassigned",
                "status": "in-stock", "location": "IT closet, Austin HQ"},
    "PRN-042": {"tag": "PRN-042", "type": "printer", "model": "FollowMe MFP",
                "os": "n/a", "assigned_to": "Floor 3 east",
                "status": "in-use", "location": "Austin HQ, floor 3"},
}

TICKETS: Dict[str, Dict[str, Any]] = {}
_NEXT_TICKET = [1042]


def _ok(payload: Any) -> str:
    return json.dumps(payload, indent=2)


def _score(query: str, article: Dict[str, str]) -> float:
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    hay = " ".join([article["title"], article["body"], article["tags"]]).lower()
    hay_tokens = set(re.findall(r"[a-z0-9]+", hay))
    if not tokens:
        return 0.0
    overlap = tokens & hay_tokens
    title_tokens = set(re.findall(r"[a-z0-9]+", article["title"].lower()))
    # Title matches weigh double.
    return (len(overlap) + len(tokens & title_tokens)) / (len(tokens) + 1)


@mcp.tool()
def search_kb(query: str, top_k: int = 5) -> str:
    """Search the IT knowledge base. Returns ranked article summaries
    (id, title, category, snippet). Use get_kb_article for full text."""
    ranked = sorted(ARTICLES.items(), key=lambda kv: _score(query, kv[1]),
                    reverse=True)
    results: List[Dict[str, str]] = []
    for aid, art in ranked[: max(1, top_k)]:
        if _score(query, art) <= 0:
            continue
        results.append({
            "article_id": aid,
            "title": art["title"],
            "category": art["category"],
            "snippet": art["body"][:220] + ("..." if len(art["body"]) > 220 else ""),
        })
    return _ok({"query": query, "results": results})


@mcp.tool()
def get_kb_article(article_id: str) -> str:
    """Fetch the full text of a knowledge-base article by id (e.g. 'KB-001')."""
    art = ARTICLES.get(article_id.strip().upper())
    if art is None:
        return _ok({"error": f"unknown article_id: {article_id!r}",
                    "known_ids": sorted(ARTICLES)})
    return _ok({"article_id": article_id.strip().upper(), **art})


@mcp.tool()
def create_ticket(title: str, description: str, category: str,
                  priority: str = "P3") -> str:
    """Open a service-desk ticket. Category examples: Network, Hardware,
    Software, Identity, Security, Facilities. Priority: P1 (critical) to
    P4 (low). Returns the new ticket record."""
    tid = f"INC-{_NEXT_TICKET[0]}"
    _NEXT_TICKET[0] += 1
    ticket = {"ticket_id": tid, "title": title, "description": description,
              "category": category, "priority": priority.upper(),
              "status": "Open", "notes": []}
    TICKETS[tid] = ticket
    return _ok(ticket)


@mcp.tool()
def get_ticket(ticket_id: str) -> str:
    """Fetch a ticket by id (e.g. 'INC-1042')."""
    t = TICKETS.get(ticket_id.strip().upper())
    if t is None:
        return _ok({"error": f"unknown ticket_id: {ticket_id!r}"})
    return _ok(t)


@mcp.tool()
def update_ticket(ticket_id: str, status: str, note: str = "") -> str:
    """Update a ticket's status (Open, In Progress, Resolved, Closed) and
    optionally append a work note."""
    t = TICKETS.get(ticket_id.strip().upper())
    if t is None:
        return _ok({"error": f"unknown ticket_id: {ticket_id!r}"})
    t["status"] = status
    if note:
        t["notes"].append(note)
    return _ok(t)


@mcp.tool()
def list_open_tickets() -> str:
    """List all tickets that are not Resolved or Closed."""
    open_t = [t for t in TICKETS.values()
              if t["status"] not in ("Resolved", "Closed")]
    return _ok({"count": len(open_t), "tickets": open_t})


@mcp.tool()
def lookup_asset(asset_tag: str) -> str:
    """Look up a hardware asset by tag (e.g. 'LT-1042'). Returns model, OS,
    assignee, status, and location."""
    a = ASSETS.get(asset_tag.strip().upper())
    if a is None:
        return _ok({"error": f"unknown asset_tag: {asset_tag!r}",
                    "known_tags": sorted(ASSETS)})
    return _ok(a)


if __name__ == "__main__":
    mcp.run()
