"""Product configuration for the graphgen-toolgrad synthetic data factory.

One YAML file describes an entire run: which MCP servers to generate
against, what domain knowledge grounds the generation, which LLM backend
to use, the quality gates, and where the dataset lands. No code changes
are needed to point the factory at a new enterprise tool surface.

Example::

    name: acme-crm
    mcp_servers:
      - name: crm
        transport: stdio
        command: npx
        args: ["-y", "@acme/mcp-crm-server"]
        timeout_s: 30.0
    domain:
      name: crm
      kg_source: corpus            # tools | corpus | spec
      corpus_dir: ./docs/crm
    generation:
      num_chains: 25
    llm:
      backend: openrouter
      model: google/gemini-2.5-flash-lite
      api_key_env: OPENROUTER_API_KEY
    quality:
      pii_redact: true
    output:
      dir: ./runs/acme-crm

Use ``ggt init`` (``bridge.cli``) to scaffold this file.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class ConfigError(ValueError):
    """Raised when a config file is invalid; carries every problem found."""


@dataclass
class MCPServerConfig:
    name: str
    transport: str = "stdio"  # stdio | sse
    command: str = ""  # stdio: executable, e.g. "npx"
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: str = ""  # sse: server URL
    timeout_s: float = 120.0  # stdio cold starts (npx, etc.) often take 60s+
    retries: int = 2


@dataclass
class DomainConfig:
    name: str = "default"
    kg_source: str = "tools"  # tools | corpus | spec
    corpus_dir: str = ""  # kg_source == "corpus"
    spec_file: str = ""  # kg_source == "spec"
    max_entities: int = 200


@dataclass
class GenerationConfig:
    num_chains: int = 10
    apis_per_workflow: int = 5
    seed: int = 42
    min_samples: int = 1  # fail the run below this many generated samples
    max_llm_calls: int = 0  # 0 = no cap; abort remaining chains past this many LLM calls
    # Optional per-tool output hints for ToolKG construction:
    # {tool_name: [[output_prop, output_type], ...]}
    output_hints: Dict[str, List[List[str]]] = field(default_factory=dict)


@dataclass
class LLMConfig:
    backend: str = "openrouter"  # openrouter only for now; else ImportError at build
    model: str = "google/gemini-2.5-flash-lite"
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    max_retries: int = 4
    backoff_s: float = 1.0
    timeout_s: float = 120.0
    max_rpm: float = 0.0  # 0 = no client-side rate limit
    max_tokens: int = 0  # 0 = provider default; set to cap per-call cost
    budget_usd: float = 0.0  # 0 = no cap; aborts the run when exceeded
    price_per_1m_input: float = 0.0  # 0 = unknown; cost then reported as call counts
    price_per_1m_output: float = 0.0


@dataclass
class QualityConfig:
    min_coverage: float = 0.5
    require_chain_valid: bool = True
    require_entities: bool = True
    max_iterations: int = 3
    semantic_filter: str = ""  # "" (off) or "jev"
    jev_threshold: float = 0.5
    jev_cli: str = ""  # required when semantic_filter == "jev": path to the judge CLI
    pii_redact: bool = True


@dataclass
class OutputConfig:
    dir: str = "outputs"
    dataset_name: str = "synthetic-tool-use"
    version: str = "0.1.0"
    train_split: float = 0.9


@dataclass
class RunConfig:
    name: str
    mcp_servers: List[MCPServerConfig] = field(default_factory=list)
    domain: DomainConfig = field(default_factory=DomainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _section(raw: dict, key: str) -> dict:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"config section {key!r} must be a mapping")
    return value


def _mcp_server_from_dict(d: dict, i: int) -> MCPServerConfig:
    if not isinstance(d, dict):
        raise ConfigError(f"mcp_servers[{i}] must be a mapping")
    try:
        return MCPServerConfig(
            name=str(d.get("name", f"server-{i}")),
            transport=str(d.get("transport", "stdio")),
            command=str(d.get("command", "")),
            args=[str(a) for a in d.get("args", []) or []],
            env={str(k): str(v) for k, v in (d.get("env", {}) or {}).items()},
            url=str(d.get("url", "")),
            timeout_s=float(d.get("timeout_s", 120.0)),
            retries=int(d.get("retries", 2)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"mcp_servers[{i}]: bad value: {exc}")


def _float(d: dict, key: str, default: float) -> float:
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        raise ConfigError(f"{key!r} must be a number")


def _int(d: dict, key: str, default: int) -> int:
    try:
        return int(d.get(key, default))
    except (TypeError, ValueError):
        raise ConfigError(f"{key!r} must be an integer")


def _bool(d: dict, key: str, default: bool) -> bool:
    value = d.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key!r} must be true/false")
    return value


def from_dict(raw: dict) -> RunConfig:
    """Build a RunConfig from a parsed mapping (validation happens after)."""
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    domain = _section(raw, "domain")
    generation = _section(raw, "generation")
    llm = _section(raw, "llm")
    quality = _section(raw, "quality")
    output = _section(raw, "output")
    servers_raw = raw.get("mcp_servers", [])
    if not isinstance(servers_raw, list):
        raise ConfigError("mcp_servers must be a list")
    cfg = RunConfig(
        name=str(raw.get("name", "run")),
        mcp_servers=[_mcp_server_from_dict(d, i) for i, d in enumerate(servers_raw)],
        domain=DomainConfig(
            name=str(domain.get("name", "default")),
            kg_source=str(domain.get("kg_source", "tools")),
            corpus_dir=str(domain.get("corpus_dir", "")),
            spec_file=str(domain.get("spec_file", "")),
            max_entities=_int(domain, "max_entities", 200),
        ),
        generation=GenerationConfig(
            num_chains=_int(generation, "num_chains", 10),
            apis_per_workflow=_int(generation, "apis_per_workflow", 5),
            seed=_int(generation, "seed", 42),
            min_samples=_int(generation, "min_samples", 1),
            max_llm_calls=_int(generation, "max_llm_calls", 0),
            output_hints={
                str(k): [[str(p), str(t)] for p, t in v]
                for k, v in (generation.get("output_hints", {}) or {}).items()
            },
        ),
        llm=LLMConfig(
            backend=str(llm.get("backend", "openrouter")),
            model=str(llm.get("model", "google/gemini-2.5-flash-lite")),
            api_key_env=str(llm.get("api_key_env", "OPENROUTER_API_KEY")),
            base_url=str(llm.get("base_url", "https://openrouter.ai/api/v1")),
            max_retries=_int(llm, "max_retries", 4),
            backoff_s=_float(llm, "backoff_s", 1.0),
            timeout_s=_float(llm, "timeout_s", 120.0),
            max_rpm=_float(llm, "max_rpm", 0.0),
            max_tokens=_int(llm, "max_tokens", 0),
            budget_usd=_float(llm, "budget_usd", 0.0),
            price_per_1m_input=_float(llm, "price_per_1m_input", 0.0),
            price_per_1m_output=_float(llm, "price_per_1m_output", 0.0),
        ),
        quality=QualityConfig(
            min_coverage=_float(quality, "min_coverage", 0.5),
            require_chain_valid=_bool(quality, "require_chain_valid", True),
            require_entities=_bool(quality, "require_entities", True),
            max_iterations=_int(quality, "max_iterations", 3),
            semantic_filter=str(quality.get("semantic_filter", "")),
            jev_threshold=_float(quality, "jev_threshold", 0.5),
            jev_cli=str(quality.get("jev_cli", "")),
            pii_redact=_bool(quality, "pii_redact", True),
        ),
        output=OutputConfig(
            dir=str(output.get("dir", "outputs")),
            dataset_name=str(output.get("dataset_name", "synthetic-tool-use")),
            version=str(output.get("version", "0.1.0")),
            train_split=_float(output, "train_split", 0.9),
        ),
    )
    problems = validate_config(cfg)
    if problems:
        raise ConfigError("invalid config:\n- " + "\n- ".join(problems))
    return cfg


def validate_config(cfg: RunConfig) -> List[str]:
    """Return a list of human-readable problems; empty means valid."""
    problems: List[str] = []
    if not cfg.mcp_servers:
        problems.append("mcp_servers: at least one server is required")
    seen = set()
    for i, s in enumerate(cfg.mcp_servers):
        tag = f"mcp_servers[{i}] ({s.name})"
        if s.name in seen:
            problems.append(f"{tag}: duplicate server name")
        seen.add(s.name)
        if s.transport not in ("stdio", "sse"):
            problems.append(f"{tag}: transport must be 'stdio' or 'sse'")
        if s.transport == "stdio" and not s.command:
            problems.append(f"{tag}: stdio transport needs 'command'")
        if s.transport == "sse" and not s.url:
            problems.append(f"{tag}: sse transport needs 'url'")
        if s.timeout_s <= 0:
            problems.append(f"{tag}: timeout_s must be > 0")
        if s.retries < 0:
            problems.append(f"{tag}: retries must be >= 0")
    d = cfg.domain
    if d.kg_source not in ("tools", "corpus", "spec"):
        problems.append("domain.kg_source must be 'tools', 'corpus', or 'spec'")
    if d.kg_source == "corpus" and not d.corpus_dir:
        problems.append("domain.corpus_dir is required when kg_source='corpus'")
    if d.kg_source == "spec" and not d.spec_file:
        problems.append("domain.spec_file is required when kg_source='spec'")
    if d.max_entities < 1:
        problems.append("domain.max_entities must be >= 1")
    g = cfg.generation
    if g.num_chains < 1:
        problems.append("generation.num_chains must be >= 1")
    if g.apis_per_workflow < 1:
        problems.append("generation.apis_per_workflow must be >= 1")
    if g.min_samples < 0:
        problems.append("generation.min_samples must be >= 0")
    if g.min_samples > g.num_chains:
        problems.append("generation.min_samples must be <= generation.num_chains")
    if g.max_llm_calls < 0:
        problems.append("generation.max_llm_calls must be >= 0")
    llm = cfg.llm
    if llm.backend not in ("openrouter",):
        problems.append(f"llm.backend {llm.backend!r} is not supported (try 'openrouter')")
    if llm.max_retries < 0:
        problems.append("llm.max_retries must be >= 0")
    if llm.timeout_s <= 0:
        problems.append("llm.timeout_s must be > 0")
    if llm.budget_usd < 0:
        problems.append("llm.budget_usd must be >= 0")
    q = cfg.quality
    if not 0.0 <= q.min_coverage <= 1.0:
        problems.append("quality.min_coverage must be in [0, 1]")
    if q.max_iterations < 1:
        problems.append("quality.max_iterations must be >= 1")
    if q.semantic_filter not in ("", "jev"):
        problems.append("quality.semantic_filter must be '' or 'jev'")
    if q.semantic_filter == "jev" and not q.jev_cli:
        problems.append("quality.jev_cli is required when semantic_filter='jev'")
    if not 0.0 <= q.jev_threshold <= 1.0:
        problems.append("quality.jev_threshold must be in [0, 1]")
    o = cfg.output
    if not 0.0 <= o.train_split < 1.0:
        problems.append("output.train_split must be in [0, 1)")
    return problems


def load_config(path: str) -> RunConfig:
    """Load a YAML (or JSON) config file."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith(".json"):
        raw = json.loads(text)
    else:
        try:
            import yaml
        except ImportError:
            raise ConfigError("pyyaml is required to load YAML configs")
        raw = yaml.safe_load(text)
    if raw is None:
        raise ConfigError(f"{path}: empty config file")
    return from_dict(raw)


def config_fingerprint(cfg: RunConfig) -> str:
    """Stable sha256 of the canonical config JSON — recorded in run reports."""
    canonical = json.dumps(cfg.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


EXAMPLE_CONFIG = """\
# graphgen-toolgrad factory config — scaffolded by `ggt init`.
# Point mcp_servers at YOUR tool surface, pick how the domain KG is built,
# and run:  python -m bridge.cli run --config {name}.yaml
name: {name}

mcp_servers:
  # stdio example: any MCP server you can launch locally
  - name: tools
    transport: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
    timeout_s: 120.0   # stdio cold starts (npx, etc.) often take 60s+
    retries: 2
  # sse example (commented):
  # - name: remote-tools
  #   transport: sse
  #   url: http://localhost:8000/sse
  #   timeout_s: 30.0

domain:
  name: {name}
  # How the grounding knowledge graph is built:
  #   tools  — from MCP tool schemas alone (no LLM, no docs needed)
  #   corpus — LLM-extracted from markdown/text docs in corpus_dir
  #   spec   — explicit entities/relations YAML (see docs/product.md)
  kg_source: tools
  corpus_dir: ./docs/{name}
  max_entities: 200

generation:
  num_chains: 25
  apis_per_workflow: 5
  seed: 42
  min_samples: 1       # fail the run below this many generated samples
  max_llm_calls: 0     # 0 = no cap; abort remaining chains past this many LLM calls

llm:
  backend: openrouter
  model: google/gemini-2.5-flash-lite
  api_key_env: OPENROUTER_API_KEY
  max_retries: 4
  backoff_s: 1.0
  timeout_s: 120.0
  max_rpm: 0.0        # 0 = no client-side rate limit
  budget_usd: 0.0     # 0 = no cap; aborts the LLMClient path (KG extraction, refinement)
                    # past a spend limit. Generation calls are capped separately
                    # via generation.max_llm_calls (see above).
  price_per_1m_input: 0.0
  price_per_1m_output: 0.0

quality:
  min_coverage: 0.5
  require_chain_valid: true
  require_entities: true
  max_iterations: 3
  semantic_filter: ""   # "" or "jev" (second-stage answer-quality filter)
  jev_threshold: 0.5
  jev_cli: ""            # required when semantic_filter="jev": path to the judge CLI
  pii_redact: true

output:
  dir: ./runs/{name}
  dataset_name: {name}-tool-use
  version: 0.1.0
  train_split: 0.9
"""


def write_example_config(path: str, name: str = "my-domain") -> str:
    """Write a scaffolded config file; returns the path."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(EXAMPLE_CONFIG.format(name=name))
    return path


def mcp_dict_for_server(server: MCPServerConfig) -> dict:
    """Translate one server config into a langchain-mcp-adapters connection dict."""
    if server.transport == "sse":
        return {
            server.name: {
                "transport": "sse",
                "url": server.url,
            }
        }
    entry: Dict[str, Any] = {
        "transport": "stdio",
        "command": server.command,
        "args": list(server.args),
    }
    if server.env:
        entry["env"] = dict(server.env)
    return {server.name: entry}


def mcp_dict_for_config(cfg: RunConfig) -> dict:
    """Merge every server's connection entry into one mcp_dict."""
    merged: Dict[str, Any] = {}
    for server in cfg.mcp_servers:
        merged.update(mcp_dict_for_server(server))
    return merged
