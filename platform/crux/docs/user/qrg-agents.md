# QRG — Agents

Workflows talk only to Crux at `http://127.0.0.1:8788`. They do not get a Utopia URL, a Headroom URL, or database credentials.

## Enroll

```bat
python -m crux enroll scripts/enroll.example.json
```

Or `POST /v1/enroll` with `CRUX_ADMIN_TOKEN`. Release A grants may only include Production or Staging.

## Ask

```http
POST http://127.0.0.1:8788/v1/ask
```

Required identity: `owner_id`, `agent_id`, `credential_id`, `session_id`. Pick a `kb_id` the session already granted.

Useful read tools: `search_chunks`, `search_docs`, `find_entities`, `entity_facts`.

```python
from crux import CruxClient

client = CruxClient(
    "http://127.0.0.1:8788",
    owner_id="owner-1",
    agent_id="agent-prod",
    credential_id="cred-prod",
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

## Propose

`remember` is propose. Facts go to Review in Enterprise Ontology. They are not a live graph edge.

A Production agent asking Staging is **403**. Session may only restrict already-granted bases.

## Audit

`GET /v1/audit` (admin token) stores redacted argument **keys** only. Native shell, file, and independent HTTP are not in the audit claim.

## Discover

```http
GET  /v1/tools
GET  /v1/readiness
GET  /v1/openapi.json
POST /v1/catalog
POST /v1/session
```
