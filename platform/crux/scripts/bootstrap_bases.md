# Release A: two restricted Utopia bases

`docker compose up -d --build` runs `python -m crux entry`, which waits for Utopia and Headroom, then bootstraps:

1. Register (or login) the first admin (`CRUX_ADMIN_EMAIL` / `CRUX_ADMIN_PASSWORD`).
2. Create restricted knowledge bases **Production** and **Staging** with the `schema-org` pack.
3. Issue a Production `read` PAT and a Staging `write` PAT.
4. Bind enrollments:
   - `owner-1` / `agent-prod` / `cred-prod` → Production, `read`
   - `owner-1` / `agent-staging` / `cred-staging` → Staging, `read` + `propose`

State (ids only, no PATs) is written to `data/crux/bootstrap.json`. PATs stay in `data/crux/enroll.db` inside the Crux volume.

To re-run by hand:

```bash
docker compose exec crux python -m crux bootstrap
```

Approve facts and edit ontology in Crux — Enterprise Ontology (Utopia) at http://127.0.0.1:1516. Do not give agents schema or permission operations. Graph extraction needs a chat/embed model there — never Headroom. `python -m crux readiness` lists `*_missing_chat_model` until that is set.

Unknown identities stay `pending`. They get no private knowledge until an admin binds a verified Utopia user and token (`POST /v1/enroll` with `CRUX_ADMIN_TOKEN`). When `bootstrap.json` exists, enrollments may only grant Production or Staging — not the open General base.
