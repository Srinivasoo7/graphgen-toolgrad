"""python -m crux serve|bootstrap|health|readiness|entry|ask|entities|facts|propose|audit"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from crux.app import CruxApp
from crux.audit import FileAuditLog
from crux.bootstrap import UtopiaAdmin, bootstrap_release_a
from crux.client import CruxClient
from crux.compress import HeadroomWorker
from crux.config import load_settings
from crux.dispatch import UtopiaClient
from crux.inflight import InFlightTracker
from crux.measure import measure_cells
from crux.probe import live_probe
from crux.readiness import live_inspector
from crux.registry import EnrollmentRegistry
from crux.server import serve_forever


def build_app() -> CruxApp:
    settings = load_settings(os.environ)
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    registry = EnrollmentRegistry(path=str(data_dir / "enroll.db"))
    audit = FileAuditLog(str(data_dir / "audit.jsonl"))
    worker = HeadroomWorker(
        settings.headroom_url, proxy_token=settings.headroom_proxy_token
    )
    return CruxApp(
        settings=settings,
        registry=registry,
        audit=audit,
        utopia_factory=lambda token: UtopiaClient(settings.utopia_url, token),
        worker=worker,
        inflight=InFlightTracker(),
        probe=lambda: live_probe(settings),
        inspector=lambda: live_inspector(settings, os.environ),
    )


def cmd_serve() -> int:
    app = build_app()
    host, port_s = app.settings.bind.rsplit(":", 1)
    print(f"crux listening on {host}:{port_s}", flush=True)
    serve_forever(app, host, int(port_s))
    return 0


def cmd_bootstrap() -> int:
    settings = load_settings(os.environ)
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    state_path = data_dir / "bootstrap.json"
    registry = EnrollmentRegistry(path=str(data_dir / "enroll.db"))
    admin = UtopiaAdmin(settings.utopia_url)
    result = bootstrap_release_a(
        admin,
        registry,
        email=os.environ.get("CRUX_ADMIN_EMAIL", "admin@crux.local"),
        password=os.environ["CRUX_ADMIN_PASSWORD"],
        owner_id=os.environ.get("CRUX_OWNER_ID", "owner-1"),
    )
    state = {
        "production_kb_id": result.production_kb_id,
        "staging_kb_id": result.staging_kb_id,
        "workspace_id": result.workspace_id,
        "user_id": result.user_id,
        "owner_id": os.environ.get("CRUX_OWNER_ID", "owner-1"),
        "agents": {
            "production": {"agent_id": "agent-prod", "credential_id": "cred-prod"},
            "staging": {"agent_id": "agent-staging", "credential_id": "cred-staging"},
        },
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2))
    return 0


def cmd_enroll(payload_path: str) -> int:
    if not payload_path:
        print("usage: python -m crux enroll <enroll.json>", file=sys.stderr)
        return 2
    app = build_app()
    body = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    result = app.enroll(body, os.environ["CRUX_ADMIN_TOKEN"])
    print(json.dumps(result, indent=2))
    return 0


def _bootstrap_state() -> dict:
    path = Path(os.environ.get("CRUX_DATA_DIR", "data/crux"), "bootstrap.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(
            f"bootstrap.json not found at {path}. Run: python -m crux bootstrap",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    except json.JSONDecodeError as exc:
        print(f"bootstrap.json is malformed: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


def _agent_client(*, prefer: str = "production") -> tuple[CruxClient, str]:
    kb_id = os.environ.get("CRUX_KB_ID")
    if not kb_id:
        state = _bootstrap_state()
        key = "staging_kb_id" if prefer == "staging" else "production_kb_id"
        kb_id = state[key]
    client = CruxClient(
        os.environ.get("CRUX_URL", "http://127.0.0.1:8788"),
        owner_id=os.environ.get("CRUX_OWNER_ID", "owner-1"),
        agent_id=os.environ["CRUX_AGENT_ID"],
        credential_id=os.environ["CRUX_CREDENTIAL_ID"],
    )
    return client, kb_id


def _print_ask_text(result: dict[str, Any]) -> None:
    text = result.get("text")
    if not text:
        try:
            text = result["utopia"]["result"]["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = json.dumps(result)
    print(text)


def cmd_ask(words: list[str]) -> int:
    if not words:
        print("usage: python -m crux ask <query>", file=sys.stderr)
        return 2
    client, kb_id = _agent_client()
    session = client.start_session(kb_ids=[kb_id])
    result = client.ask(
        kb_id=kb_id,
        tool="search_chunks",
        arguments={"query": " ".join(words)},
        session_id=session["session_id"],
        session_kb_ids=session["kb_ids"],
    )
    _print_ask_text(result)
    return 0


def cmd_entities(words: list[str]) -> int:
    if not words:
        print("usage: python -m crux entities <query>", file=sys.stderr)
        return 2
    client, kb_id = _agent_client()
    session = client.start_session(kb_ids=[kb_id])
    result = client.find_entities(
        kb_id=kb_id,
        query=" ".join(words),
        session_id=session["session_id"],
        session_kb_ids=session["kb_ids"],
    )
    _print_ask_text(result)
    return 0


def cmd_facts(words: list[str]) -> int:
    if len(words) != 1:
        print("usage: python -m crux facts <entity_id>", file=sys.stderr)
        return 2
    client, kb_id = _agent_client()
    session = client.start_session(kb_ids=[kb_id])
    result = client.entity_facts(
        kb_id=kb_id,
        entity_id=words[0],
        session_id=session["session_id"],
        session_kb_ids=session["kb_ids"],
    )
    _print_ask_text(result)
    return 0


def cmd_propose(words: list[str]) -> int:
    if not words:
        print("usage: python -m crux propose <text>", file=sys.stderr)
        return 2
    client, kb_id = _agent_client(prefer="staging")
    session = client.start_session(kb_ids=[kb_id])
    result = client.remember(
        kb_id=kb_id,
        text=" ".join(words),
        session_id=session["session_id"],
        session_kb_ids=session["kb_ids"],
    )
    text = result.get("text")
    if not text:
        messages = (result.get("compressed") or {}).get("messages") or []
        if messages:
            text = messages[0].get("content")
    print(text or json.dumps(result))
    print(
        "proposed - pending Review in Enterprise Ontology; not a live graph edge",
        file=sys.stderr,
    )
    return 0


def cmd_audit() -> int:
    import httpx

    url = os.environ.get("CRUX_URL", "http://127.0.0.1:8788").rstrip("/")
    token = os.environ["CRUX_ADMIN_TOKEN"]
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{url}/v1/audit", headers={"Authorization": f"Bearer {token}"})
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2))
    return 0


def cmd_measure(words: list[str]) -> int:
    query = " ".join(words) or "write path"
    client, kb_id = _agent_client()
    print(json.dumps(measure_cells(client, kb_id=kb_id, query=query), indent=2))
    return 0


def cmd_health() -> int:
    app = build_app()
    body = app.health()
    print(json.dumps(body, indent=2))
    return 0 if body.get("status") == "ok" else 1


def cmd_readiness() -> int:
    app = build_app()
    body = app.readiness()
    print(json.dumps(body, indent=2))
    if body.get("blockers") or body.get("status") != "ok":
        return 1
    return 0


def cmd_entry() -> int:
    settings = load_settings(os.environ)
    deadline = time.time() + 180
    while time.time() < deadline:
        probes = live_probe(settings)
        print("waiting for dependencies", probes, flush=True)
        if probes.get("utopia") and probes.get("headroom"):
            break
        time.sleep(3)
    else:
        print("utopia or headroom did not become reachable", file=sys.stderr)
        return 1
    if os.environ.get("CRUX_AUTO_BOOTSTRAP", "1") == "1":
        last_error: Exception | None = None
        for _ in range(20):
            try:
                cmd_bootstrap()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print("bootstrap retry:", exc, flush=True)
                time.sleep(3)
        if last_error is not None:
            print("bootstrap failed:", last_error, file=sys.stderr)
            return 1
    return cmd_serve()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else "serve"
    if command == "serve":
        return cmd_serve()
    if command == "bootstrap":
        return cmd_bootstrap()
    if command == "health":
        return cmd_health()
    if command == "readiness":
        return cmd_readiness()
    if command == "entry":
        return cmd_entry()
    if command == "enroll":
        return cmd_enroll(args[1] if len(args) > 1 else "")
    if command == "ask":
        return cmd_ask(args[1:])
    if command == "entities":
        return cmd_entities(args[1:])
    if command == "facts":
        return cmd_facts(args[1:])
    if command == "propose":
        return cmd_propose(args[1:])
    if command == "audit":
        return cmd_audit()
    if command == "measure":
        return cmd_measure(args[1:])
    print(
        "usage: python -m crux [serve|bootstrap|health|readiness|entry|enroll|ask|entities|facts|propose|audit|measure]",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
