"""ToolGrad-backed sample generation, productized.

This ports the live-run driver's generation stage into the repo as a
reusable provider. Given an ``mcp_dict`` (any MCP servers, from
:func:`bridge.config.mcp_dict_for_config`), rendered KG context text, an
LLM configuration, and generation knobs, it runs ToolGrad's graph on the
fork's native seams and returns executed workflow samples + traces.

Requires the ``Srinivasoo7/toolgrad`` fork importable (native seams:
``kg_context`` in ``PREDICT_WORKFLOW`` / ``ToolGradState``,
``get_mcp_apis`` sampler hook, public ``discover_mcp_tools``). Missing
deps raise :class:`GenerationError` with an actionable message instead of
an ImportError traceback.

Robustness notes (lessons from the live run, now structural):
- one ``MultiServerMCPClient`` is kept alive for the whole generation
  call — the fork's ``discover_mcp_tools`` builds its client as a local,
  so re-discovering per call GCs the client and kills the stdio server;
- the LLM factory swap is runtime composition on the fork's public
  ``create_llm`` seam; the OpenRouter backend is ``langchain_openai``'s
  ``ChatOpenAI`` (native async support, no custom httpx wiring);
- chains are generated independently per seed; a failed chain logs a
  warning and is skipped rather than aborting the run — failures are
  returned per chain, and the run fails when ``min_samples`` is not met;
- ``max_llm_calls`` soft-caps generation LLM calls across chains: the cap is
  checked after every chain outcome (success or failure) and remaining
  chains are skipped once exceeded, but the in-flight chain always runs to
  completion, so the count can overshoot by up to one chain's calls. It is
  not a dollar budget and not a hard pre-call limit (the USD budget in
  ``llm.budget_usd`` covers only the LLMClient path: KG extraction and
  refinement critiques — see docs/product.md).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)


class GenerationError(Exception):
    """Generation could not run; the message says what to fix.

    ``details`` carries structured context (per-chain failures, sample and
    call counts) so failed-stage reports stay debuggable, not just a string.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = dict(details) if details else {}


def _require_fork():
    try:
        import toolgrad as tog  # noqa: F401
        from toolgrad.utils import langchain as langchain_utils
        from toolgrad.utils import mcp as mcp_module
        from toolgrad import prebuilt, modules
    except ImportError as exc:
        raise GenerationError(
            "the ToolGrad fork is required for generation "
            "(pip install git+https://github.com/Srinivasoo7/toolgrad.git"
            "@integrate/graphgen-toolgrad)"
        ) from exc
    return tog, langchain_utils, mcp_module, prebuilt, modules


def _sanitize_proxy_env() -> None:
    """Strip bracketed IPv6 literals from ``no_proxy`` variables.

    Some httpx builds (pulled in via ``langchain_openai``) parse every
    ``no_proxy`` entry as a URL pattern and choke on ``[::1]``-style
    literals (``InvalidURL: Invalid port: ':1]'``). The unbracketed forms
    are also present, so bypass behavior is unchanged. Process-local only.
    """
    for var in ("no_proxy", "NO_PROXY"):
        val = os.environ.get(var)
        if not val:
            continue
        entries = val.split(",")
        kept = [e for e in entries if not e.strip().startswith("[")]
        if len(kept) != len(entries):
            os.environ[var] = ",".join(kept)
            log.info("sanitized %s for httpx proxy parsing", var)


def _llm_factory(llm_cfg: dict, counter: Dict[str, int]) -> Callable:
    """Build the fork's ``create_llm`` replacement: OpenRouter via ChatOpenAI.

    Credential resolution mirrors :func:`bridge.llm.build_llm_client`:
    the ``api_key_env`` variable wins when set; otherwise the connected
    ``custom.openrouter`` credential is used through the authd surrogate
    (only ``hsurr:*`` values are handled — never a raw key).
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise GenerationError(
            "langchain-openai is required for the OpenRouter generation backend"
        ) from exc
    _sanitize_proxy_env()
    api_key_env = llm_cfg.get("api_key_env", "OPENROUTER_API_KEY")
    static_key = os.environ.get(api_key_env, "")

    def _resolve_key() -> str:
        if static_key:
            return static_key
        from bridge.llm import AuthdOpenRouterBackend, _load_authd_helpers

        dc = _load_authd_helpers()
        try:
            entry = dc.dynamic_credential_entry(
                AuthdOpenRouterBackend.CREDENTIAL_NAME, "access_token"
            )
        except Exception as exc:
            raise GenerationError(
                f"no OpenRouter credential: {api_key_env} is not set and the "
                f"{AuthdOpenRouterBackend.CREDENTIAL_NAME} credential is "
                f"unavailable ({exc})"
            ) from exc
        return str(entry["surrogate"]).strip()

    def factory(*args, **kwargs):
        key = _resolve_key()
        counter["n"] += 1
        temperature = kwargs.get("temperature", kwargs.get("temp", 1.0))
        return ChatOpenAI(
            model=llm_cfg.get("model", "google/gemini-2.5-flash-lite"),
            api_key=key,
            base_url=llm_cfg.get("base_url", "https://openrouter.ai/api/v1"),
            temperature=temperature,
        )

    return factory


def generate_samples(
    *,
    mcp_dict: Dict[str, Any],
    kg_text: str,
    llm_cfg: Dict[str, Any],
    num_chains: int = 10,
    apis_per_workflow: int = 5,
    num_iterations: int = 3,
    seed: int = 42,
    output_hints: Optional[Dict[str, Any]] = None,
    workdir: str = "outputs",
    trace_subdir: str = "trace",
    min_samples: int = 1,
    max_llm_calls: int = 0,  # 0 = no cap; soft cap checked after each chain
) -> Dict[str, Any]:
    """Run ToolGrad generation; returns samples, traces, tools, and counts.

    Returns ``{"samples", "sample_paths", "trace_paths", "tools_by_name",
    "llm_calls", "per_chain_failures"}``. Raises :class:`GenerationError`
    on configuration problems and when fewer than ``min_samples`` samples
    were produced; per-chain failures are logged, skipped, and returned —
    they never abort the run early.
    """
    tog, langchain_utils, mcp_module, prebuilt, modules = _require_fork()
    from bridge import toolkg_builder, toolkg_sampler

    llm_calls = {"n": 0}
    langchain_utils.create_llm = _llm_factory(llm_cfg, llm_calls)

    # MCP keepalive: cache one client for the whole run.
    state: Dict[str, Any] = {}

    def _cached_discover(mcp_dict_arg: dict):
        if "tools" not in state:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            state["client"] = MultiServerMCPClient(mcp_dict_arg)  # kept alive
            tools = asyncio.run(state["client"].get_tools())
            state["tools"] = list(tools)
            log.info("discovered %d tools from MCP", len(tools))
        return list(state["tools"])

    mcp_module.discover_mcp_tools = _cached_discover

    tools = _cached_discover(mcp_dict)
    if not tools:
        raise GenerationError("MCP discovery returned no tools; nothing to generate from")
    tools_by_name = {t.name: t for t in tools}

    toolkg = toolkg_builder.build_toolkg(tools, output_hints=output_hints)
    stats = toolkg_builder.toolkg_stats(toolkg)
    log.info("ToolKG: %s", stats)

    trace_dir = os.path.join(workdir, trace_subdir)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(trace_dir, exist_ok=True)

    samples: List[dict] = []
    sample_paths: List[str] = []
    trace_paths: List[str] = []
    per_chain_failures: List[dict] = []
    for chain_i in range(num_chains):
        chain_seed = seed + chain_i
        sampler = toolkg_sampler.make_toolkg_sampler(toolkg, seed=chain_seed)
        app = prebuilt.create_graph_on_mcp(
            sample_seed=chain_seed,
            num_apis=apis_per_workflow,
            num_iterations=num_iterations,
            mcp_dict=mcp_dict,
            api_sampler=sampler,
        )
        tracer = tog.utils.trace_utils.ExecutionTracer(
            output_dir=workdir, seed=chain_seed
        )
        initial = modules.ToolGradState(
            workflow_cur=None,
            api_proposals=None,
            api_reports=None,
            api_selection=None,
            step=0,
            sampled_apis=[],
            tracer=tracer,
            kg_context=kg_text,
        )
        chain_ok = True
        try:
            final_state = app.invoke(
                initial,
                config={
                    "configurable": {"thread_id": chain_seed},
                    "recursion_limit": 1000,
                },
            )
        except Exception as exc:  # noqa: BLE001 — one bad chain must not kill the run
            log.warning("chain %d failed: %s: %s", chain_i, type(exc).__name__, exc)
            per_chain_failures.append(
                {
                    "chain": chain_i,
                    "seed": chain_seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            chain_ok = False
        if chain_ok:
            tracer.save()
            trace_paths.append(os.path.join(trace_dir, f"{chain_seed:05d}.json"))
            data_sample = final_state["workflow_cur"]
            if data_sample is None:
                log.warning("chain %d produced no workflow (all iterations failed)", chain_i)
                per_chain_failures.append(
                    {
                        "chain": chain_i,
                        "seed": chain_seed,
                        "error_type": "NoWorkflow",
                        "error": "all iterations failed; no workflow produced",
                    }
                )
                chain_ok = False
        if chain_ok:
            sample = data_sample.model_dump()
            sample_path = os.path.join(
                workdir, f"seed={chain_seed}__iter={num_iterations}__num_apis={apis_per_workflow}.json"
            )
            with open(sample_path, "w", encoding="utf-8") as fh:
                json.dump(sample, fh, indent=2)
            samples.append(sample)
            sample_paths.append(sample_path)
            log.info("chain %d: workflow saved (%d samples so far)", chain_i, len(samples))
        # Soft cap, enforced after every chain outcome (success or failure):
        # the in-flight chain always runs to completion, so llm_calls can
        # overshoot max_llm_calls by up to one chain's worth of calls.
        if max_llm_calls > 0 and llm_calls["n"] > max_llm_calls:
            log.warning(
                "generation LLM call budget exceeded (%d > %d); stopping after chain %d",
                llm_calls["n"], max_llm_calls, chain_i,
            )
            break

    if len(samples) < min_samples:
        raise GenerationError(
            f"generation produced {len(samples)} sample(s), below min_samples="
            f"{min_samples} ({len(per_chain_failures)} chain failure(s) recorded; "
            "see per_chain_failures)",
            details={
                "num_samples": len(samples),
                "min_samples": min_samples,
                "llm_calls": llm_calls["n"],
                "num_chain_failures": len(per_chain_failures),
                "per_chain_failures": per_chain_failures,
                "sample_paths": sample_paths,
            },
        )

    return {
        "samples": samples,
        "sample_paths": sample_paths,
        "trace_paths": trace_paths,
        "tools_by_name": tools_by_name,
        "llm_calls": llm_calls["n"],
        "per_chain_failures": per_chain_failures,
    }
