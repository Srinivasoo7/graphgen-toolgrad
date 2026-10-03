# Crux

Crux is the product: **Enterprise Ontology (Utopia)** plus **Token Optimization (Headroom)**. It is not a third access-control engine.

| Piece | Role |
|---|---|
| Crux — Enterprise Ontology (Utopia) | Who may read or propose; two restricted bases in Release A |
| Crux — Token Optimization (Headroom) | `POST /v1/compress` after Ontology authorized the payload. CCR off. Unpublished. |
| Crux HTTP | Agent port on `127.0.0.1:8788`. Enrollment, instrumented MCP, audit. Operator console at `/`. |

ECC (`affaan-m/ECC`) is installed for Cursor under `.cursor/` (ecc-universal 2.2.0).

## Pins

See [pins.json](pins.json). Verified on Utopia commit `33e12d97`:

- MCP at `POST /api/v1/kbs/{kb_id}/mcp`
- Read tools plus `remember` (propose → Review, not a live edge)
- Write needs token `write` scope **and** editor+
- Each call records `mcp.tool_called`
- `query_data` is not on MCP

## Run

```bash
copy .env.example .env
docker compose up -d --build
```

| Surface | Bind | Who |
|---|---|---|
| Crux console + `/v1/ask` | http://127.0.0.1:8788 | Operators and agents |
| Enterprise Ontology | http://127.0.0.1:1516 | Humans (approve, ontology, permissions) |
| Postgres | `127.0.0.1:1517` | Operators only |
| Token Optimization | unpublished | Crux container only |

Operator QRGs: [docs/user/README.md](docs/user/README.md). Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Pitch: [docs/pitch/crux-pitch.html](docs/pitch/crux-pitch.html) or `http://127.0.0.1:8788/pitch`.

First boot registers the admin, creates Production + Staging restricted bases, and binds `agent-prod` / `agent-staging`. See [scripts/bootstrap_bases.md](scripts/bootstrap_bases.md).

Do **not** point Utopia `chat_base_url` at Headroom. Extraction, embeddings, and ontology-admin stay on the provider (trusted compression bypass). Agent chat: Utopia MCP → Headroom `/v1/compress` → provider.

## Agent ask

```http
POST http://127.0.0.1:8788/v1/ask
Content-Type: application/json

{
  "owner_id": "owner-1",
  "agent_id": "agent-prod",
  "credential_id": "cred-prod",
  "session_id": "s1",
  "kb_id": "<production kb id from data/crux/bootstrap.json>",
  "tool": "search_chunks",
  "arguments": {"query": "what is on fire"}
}
```

A pending or revoked identity is denied. A Production agent asking Staging is denied. Revoke cancels in-flight instrumented calls; already-sent model context cannot be recalled.

Python workflows can use `CruxClient` (no Utopia URL, no Headroom URL):

```python
from crux import CruxClient

client = CruxClient(
    "http://127.0.0.1:8788",
    owner_id="owner-1",
    agent_id="incident-bot",
    credential_id="cred-inc",
)
session = client.start_session(kb_ids=["<production kb id>"])
result = client.ask(
    kb_id="<production kb id>",
    tool="search_chunks",
    arguments={"query": "what is on fire"},
    session_id=session["session_id"],
    session_kb_ids=session["kb_ids"],
)
```

Discover tools and grants (no PATs):

```http
GET  http://127.0.0.1:8788/v1/tools
GET  http://127.0.0.1:8788/v1/readiness
GET  http://127.0.0.1:8788/v1/openapi.json
POST http://127.0.0.1:8788/v1/catalog
POST http://127.0.0.1:8788/v1/session
```

`remember` is always propose (`client.remember(...)`). Extracted facts go to Review; they are not a live graph edge. `query_data` and ontology-admin tools are rejected. Enroll another workflow agent with `python -m crux enroll enroll.json` (or `POST /v1/enroll` with `CRUX_ADMIN_TOKEN`). Release A enrollments may only grant Production or Staging. List bindings (no PATs): `GET /v1/enrollments`.

Operator demo against the running stack:

```bash
set CRUX_AGENT_ID=agent-prod
set CRUX_CREDENTIAL_ID=cred-prod
python -m crux ask write path
python -m crux entities ontology
python -m crux facts 01a07cbf-668c-73b1-a559-1e5002f5710f
python -m crux measure write path
set CRUX_AGENT_ID=agent-staging
set CRUX_CREDENTIAL_ID=cred-staging
python -m crux propose Alice ran a staging check
python -m crux audit
python -m crux readiness
```

Graph extraction is configured in Enterprise Ontology at http://127.0.0.1:1516 (Administration → Models). Do not point `chat_base_url` at Headroom. `python -m crux readiness` lists that as a blocker when it is missing or mis-pointed.

Live two-base isolation: `set CRUX_LIVE=1` then `python -m pytest tests/test_leak_live.py`.

## Develop

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Release A rules

- Scope is access, not a new ontology per agent.
- Session may only restrict already-granted bases.
- Agents use this package. No side-door PAT against a public Utopia URL plus DB creds.
- Audit logs redacted keys only. No plain hashes of arguments.
- Native shell/file/HTTP are not in the audit claim.
