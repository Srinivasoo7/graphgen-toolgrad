# Map

| Path | Purpose |
|---|---|
| `crux/access.py` | Enrollment, session subset, Utopia-shaped authorize |
| `crux/registry.py` | Enrollment registry; SQLite when `path=` is set |
| `crux/audit.py` | Instrumented audit; `FileAuditLog` JSONL |
| `crux/inflight.py` | Cancel open requests on revoke; no recall of sent context |
| `crux/config.py` | Process settings from the environment |
| `crux/compress.py` | Headroom `/v1/compress` worker; trusted bypass kinds |
| `crux/dispatch.py` | Instrumented Utopia MCP only |
| `crux/bootstrap.py` | Two restricted bases + bound enrollments |
| `crux/app.py` | enroll / revoke / ask |
| `crux/client.py` | Workflow-agent SDK (`CruxClient`) |
| `crux/readiness.py` | Restricted-base readiness; Headroom chat_url guard |
| `crux/server.py` | Loopback HTTP for agents |
| `crux/probe.py` | Utopia + Headroom reachability |
| `crux/measure.py` | Token-savings cells A–D; A/C derived from `tokens_before` of the same ask |
| `crux/__main__.py` | `serve`, `bootstrap`, `health`, `readiness`, `entry`, `enroll`, `ask`, `entities`, `facts`, `propose`, `audit`, `measure` |
| `crux/branding.py` | User-facing module names |
| `crux/console.py` | Operator console and QRG HTML |
| `crux/openapi.py` | Static OpenAPI for the agent port |
| `crux/tools.py` | Pinned MCP tools; derive read vs propose |
| `crux/leak.py` | Leak surfaces; `bundle_from_mcp_text` for live isolation |
| `tests/` | pytest; TDD first |
| `docker-compose.yml` | Utopia + Postgres + unpublished Headroom + Crux |
| `docker/headroom/` | Headroom worker image (`--no-ccr`) |
| `docker/crux/` | Crux product image |
| `pins.json` | Pinned Utopia/Headroom/ECC versions and verified MCP facts |
| `scripts/bootstrap_bases.md` | What first boot creates |
| `scripts/enroll.example.json` | Template for `python -m crux enroll` |
| `scripts/measure.md` | Token-savings protocol (hypothesis) |
| `.cursor/skills/crux/` | Product skill — follow on every Crux change |
| `docs/ARCHITECTURE.md` | Deployment, ask path, and module diagrams |
| `docs/adr/` | History / locked decisions |
| `docs/user/` | Operator QRGs |
| `docs/pitch/crux-pitch.html` | What / Why / How pitch |
| `docker/ontology-ui/` | Ontology UI title and banner overlay |
| `docs/STATUS.md` | Current health and blockers |

**Request path (agent chat):** credential → `authorize` → Utopia MCP → Headroom compress → provider. Diagrams: [ARCHITECTURE.md](ARCHITECTURE.md).
