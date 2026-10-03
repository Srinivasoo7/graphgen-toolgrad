# Architecture

Release A packaging. Crux is the product. Enterprise Ontology (Utopia) authorizes knowledge. Token Optimization (Headroom) compresses only after that. There is no third ACL.

Locked decisions: [docs/adr/](adr/). File map: [CODEMAP.md](CODEMAP.md). Health: [STATUS.md](STATUS.md).

Effective access:

```
Utopia grants ∩ credential ∩ session subset of those grants
```

Session can only restrict. Production asking Staging is 403. `remember` is propose → Review, not a live graph edge.

## Deployment

Who can reach what. Headroom has no host port. Postgres is operator-only. Agents never receive Headroom or database credentials.

```mermaid
flowchart TB
  subgraph host [Host loopback]
    human[Human admin]
    agent[Workflow agent or CruxClient]
    operator[Operator CLI]
  end

  subgraph compose [Docker Compose network]
    crux["Crux 8788 console and ask"]
    utopia["Enterprise Ontology Utopia 1516"]
    db["Postgres pgvector 5432"]
    headroom["Token Optimization Headroom 8787 unpublished"]
  end

  provider[Model provider chat and embed]

  human -->|"127.0.0.1:1516 models upload Review ontology"| utopia
  human -->|"127.0.0.1:8788 QRGs pitch"| crux
  agent -->|"127.0.0.1:8788 no Utopia URL no Headroom URL"| crux
  operator -->|"python -m crux"| crux

  crux -->|"MCP after authorize"| utopia
  crux -->|"POST /v1/compress CallKind only"| headroom
  utopia --> db
  utopia -->|"extraction embed ontology-admin not via Headroom"| provider

  opDb["Operator only 127.0.0.1:1517"]
  opDb -.-> db
```

Do not point Utopia `chat_base_url` at Headroom. Extraction, embeddings, and ontology-admin stay on the provider (trusted `CallKind` bypass, not a request header).

## Ask path

The combined product: credential → authorize → Utopia MCP → Headroom compress.

```mermaid
sequenceDiagram
  participant A as Agent
  participant C as Crux
  participant R as Enrollment registry
  participant U as Utopia MCP
  participant H as Headroom compress

  A->>C: POST /v1/ask identity session kb_id tool
  C->>R: load enrollment
  alt pending revoked or kb not in session subset
    C-->>A: 403 AccessDenied
  else authorized
    C->>U: MCP tool on that kb_id only
    U-->>C: authorized payload
    C->>H: POST /v1/compress
    H-->>C: tokens_before after saved
    C-->>A: utopia result plus compressed text
  end
```

## Two modules

```mermaid
flowchart LR
  subgraph ontology [Crux Enterprise Ontology]
    bases[Production and Staging restricted bases]
    review[Review queue]
    graph[Graph extract]
    policy[Roles tokens membership]
  end

  subgraph tokens [Crux Token Optimization]
    compress["POST /v1/compress"]
    bypass[CallKind bypass extract embed admin]
  end

  subgraph glue [Crux packaging]
    enroll[Enroll revoke session]
    audit[Audit arg keys only]
    console[Operator console]
  end

  glue --> ontology
  ontology -->|"authorized payload"| tokens
```

## Surfaces

| Bind | What |
|---|---|
| `127.0.0.1:8788` | Agent port, operator console, QRGs, pitch |
| `127.0.0.1:1516` | Human Ontology UI |
| `127.0.0.1:1517` | Postgres, operators only |
| Headroom `:8787` | Compose-internal only |

Audit covers instrumented MCP only. Native shell, file, and independent HTTP are not in the audit claim.
