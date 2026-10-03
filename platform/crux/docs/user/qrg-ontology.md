# QRG — Enterprise Ontology

This is the human console. Approve facts, edit ontology, and change permissions here. Agents do not get those tools.

## Models (required for graph extract)

1. Open `http://127.0.0.1:1516`
2. Administration → Models
3. Set a **chat** model and an **embed** model on the provider
4. Leave `chat_base_url` on the provider — **never Headroom**

`python -m crux readiness` lists `*_missing_chat_model` or a Headroom chat_url blocker until this is right.

## Upload and extract

Upload documents into **Production** or **Staging**, not a shared “everything” base. Release A isolation is two restricted bases. Same-base object filters are later.

Wait for extract. Then:

```bat
set CRUX_AGENT_ID=agent-prod
set CRUX_CREDENTIAL_ID=cred-prod
python -m crux entities ontology
```

## Review is not a live edge

`remember` and extracted candidates go to **Review**. They are not a live graph edge until a human approves them in this UI.

Propose ≠ approve ≠ ontology edit ≠ permission change.

## Permissions

Change membership and tokens in this UI. Crux enrollments may only grant Production or Staging once `bootstrap.json` exists. Unknown identities stay pending.

## What stays Utopia-shaped

The in-app nav may still say Utopia in places. The product name is **Crux — Enterprise Ontology (Utopia)**. The banner and browser title use that name. Do not string-replace “Utopia” inside graph text.
