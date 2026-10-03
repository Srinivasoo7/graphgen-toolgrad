# QRG — First boot

You get two restricted bases and a working agent port. You do not get a third access-control product.

## Do this

```bat
copy .env.example .env
docker compose up -d --build
```

Open the operator console: `http://127.0.0.1:8788`

Open **Crux — Enterprise Ontology (Utopia)**: `http://127.0.0.1:1516`

Sign in with `CRUX_ADMIN_EMAIL` / `CRUX_ADMIN_PASSWORD` from `.env`. Do not paste those values into chat logs.

## What first boot already created

- Restricted bases **Production** and **Staging**
- `owner-1` / `agent-prod` / `cred-prod` → Production, read
- `owner-1` / `agent-staging` / `cred-staging` → Staging, read + propose

Ids (not PATs) land in `data/crux/bootstrap.json`.

## Surfaces

- Agents: `http://127.0.0.1:8788` (`/v1/ask`)
- Humans: `http://127.0.0.1:1516`
- Postgres: `127.0.0.1:1517` — operators only
- Token Optimization — unpublished. Only the Crux container may call it.

## Check

```bat
python -m crux readiness
python -m crux health
```

If readiness lists a missing chat model, set it in the Ontology UI (next guide). Do not point `chat_base_url` at Headroom.
