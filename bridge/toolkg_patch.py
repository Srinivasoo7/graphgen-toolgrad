"""Wire KG-neighborhood sampling into ToolGrad's MCP sampling path.

Upstream call chain (``toolgrad/modules/graph_lib.py::sample_apis_or_end``)::

    toolgrad_state.sampled_apis = mcp.get_mcp_apis(
        mcp_dict=mcp_dict, num_apis=num_apis, seed=api_sample_seed)

``get_mcp_apis`` discovers the MCP tools and then does
``random.seed(seed); random.sample(tools, num_apis)`` — uniform random.
This patch replaces that *sampling step only*, reusing upstream's own
discovery (client, ``ALLOWED_APIS`` filter, path-prefix wrapper), by
swapping the ``get_mcp_apis`` attribute on the ``toolgrad.utils.mcp``
module — which ``graph_lib`` resolves at call time, so no upstream edit
is needed (same monkeypatch style as ``bridge/toolgrad_patch.py``).

Usage::

    from bridge import toolkg_patch

    # Build (or reuse a cached) ToolKG from the live MCP tools, then patch.
    toolkg_patch.apply_toolkg_patch()
    # ... build and invoke the ToolGrad MCP graph as usual; each iteration
    # ... now samples a composable neighborhood instead of a random bundle ...
    toolkg_patch.remove_toolkg_patch()

``output_hints`` (tool name -> [(output prop, type)]) can be passed to
:func:`apply_toolkg_patch` / :func:`build_toolkg_for_mcp`; without them,
tool outputs are inferred from descriptions (see ``toolkg_builder``).

Note on determinism: upstream reseeds the *global* RNG with ``seed`` on
every call, so every iteration samples the *same* bundle (likely an
upstream quirk). The patched sampler uses a local ``random.Random`` seeded
with a deterministic fold of ``(seed, call_count)`` into one int —
deterministic per call sequence, varying per iteration.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from . import kg_sampler
from . import toolkg_builder

_TOOLKG = None
_TOOLS_BY_NAME: Dict[str, Any] = {}
_ORIGINAL_GET_MCP_APIS = None
_PATCHED = False
_CALL_COUNT = 0


def get_toolkg():
    """The currently cached ToolKG (``None`` until built)."""
    return _TOOLKG


def is_patched() -> bool:
    """Whether ToolKG sampling is currently installed."""
    return _PATCHED


def _discover_all_tools(mcp_dict: dict) -> list:
    """Upstream discovery from ``get_mcp_apis`` *minus* the random sample.

    Reuses upstream's own ``MultiServerMCPClient``, ``ALLOWED_APIS``
    filter, and ``_wrap_with_path_prefix`` so only the sampling step
    changes. (``_wrap_with_path_prefix`` is private upstream API — the
    pinned upstream SHA is recorded in ``docs/phase0-recon.md``.)
    """
    from toolgrad.utils import mcp as mcp_module
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(mcp_dict)
    tools = asyncio.run(client.get_tools())
    tools = [t for t in tools if t.name in mcp_module.ALLOWED_APIS]
    filesystem_path = mcp_module.get_example_filesystem_dir()
    tools = [mcp_module._wrap_with_path_prefix(t, filesystem_path) for t in tools]
    print(f"Found {len(tools)} APIs in MCP.")
    return tools


def build_toolkg_for_mcp(
    mcp_dict: Optional[dict] = None,
    output_hints: Optional[Mapping[str, Sequence[Tuple[str, str]]]] = None,
    threshold: float = toolkg_builder.DEFAULT_THRESHOLD,
    force_rebuild: bool = False,
):
    """Discover MCP tools (npx, keyless) and build/cache the ToolKG."""
    global _TOOLKG, _TOOLS_BY_NAME
    if _TOOLKG is not None and not force_rebuild:
        return _TOOLKG
    from toolgrad.utils import mcp as mcp_module

    if mcp_dict is None:
        mcp_dict = mcp_module.get_default_mcp_dict()
    tools = _discover_all_tools(mcp_dict)
    _TOOLKG = toolkg_builder.build_toolkg(
        tools, output_hints=output_hints, threshold=threshold
    )
    _TOOLS_BY_NAME = {t.name: t for t in tools}
    return _TOOLKG


def _ensure_toolkg(mcp_dict: Optional[dict]):
    if _TOOLKG is None or not _TOOLS_BY_NAME:
        build_toolkg_for_mcp(mcp_dict)
    return _TOOLKG, _TOOLS_BY_NAME


def _effective_seed(seed: Any, call_count: int) -> int:
    """Deterministic int seed from ToolGrad's ``seed`` and the call counter.

    ``random.Random`` only accepts int/float/str/bytes seeds (not tuples),
    so the pair is folded into one int arithmetically — stable across
    processes, unlike ``hash()`` of a tuple.
    """
    base = seed if isinstance(seed, int) else abs(hash(str(seed))) % (2**31)
    return base * 1_000_003 + call_count


def _toolkg_get_mcp_apis(mcp_dict: dict, num_apis: int = 5, seed: int = 42) -> list:
    """Drop-in replacement for ``toolgrad.utils.mcp.get_mcp_apis``."""
    global _CALL_COUNT
    toolkg, tools_by_name = _ensure_toolkg(mcp_dict)
    names = sorted(tools_by_name)
    if num_apis > len(names) or num_apis <= 0:
        raise ValueError(
            f"num_apis should be between 1 and {len(names)}, got {num_apis}."
        )
    _CALL_COUNT += 1
    rng = random.Random(_effective_seed(seed, _CALL_COUNT))
    sampled_names = kg_sampler.sample_api_neighborhood(toolkg, num_apis, rng=rng)
    print(f"ToolKG sampled {len(sampled_names)} APIs: {sampled_names}")
    return [tools_by_name[n] for n in sampled_names]


def apply_toolkg_patch(
    mcp_dict: Optional[dict] = None,
    output_hints: Optional[Mapping[str, Sequence[Tuple[str, str]]]] = None,
    threshold: float = toolkg_builder.DEFAULT_THRESHOLD,
    toolkg=None,
    tools_by_name: Optional[Dict[str, Any]] = None,
) -> None:
    """Install ToolKG-neighborhood sampling on the MCP path.

    Pass a prebuilt ``toolkg`` + ``tools_by_name`` to skip live MCP
    discovery (used by the keyless tests, and by callers who build the
    ToolKG once up front). Otherwise the ToolKG is built lazily from
    ``mcp_dict`` on the first patched call. Idempotent.
    """
    global _PATCHED, _ORIGINAL_GET_MCP_APIS
    global _TOOLKG, _TOOLS_BY_NAME, _CALL_COUNT
    from toolgrad.utils import mcp as mcp_module

    if _PATCHED:
        remove_toolkg_patch()
    _ORIGINAL_GET_MCP_APIS = mcp_module.get_mcp_apis
    _CALL_COUNT = 0
    if toolkg is not None:
        _TOOLKG = toolkg
        _TOOLS_BY_NAME = dict(tools_by_name or {})
    elif mcp_dict is not None or output_hints is not None:
        # Eager build so misconfiguration fails here, not mid-loop.
        build_toolkg_for_mcp(mcp_dict, output_hints, threshold, force_rebuild=True)
    mcp_module.get_mcp_apis = _toolkg_get_mcp_apis
    _PATCHED = True


def remove_toolkg_patch() -> None:
    """Restore upstream's ``get_mcp_apis``. Safe to call when unpatched."""
    global _PATCHED
    if not _PATCHED:
        return
    from toolgrad.utils import mcp as mcp_module

    mcp_module.get_mcp_apis = _ORIGINAL_GET_MCP_APIS
    _PATCHED = False
