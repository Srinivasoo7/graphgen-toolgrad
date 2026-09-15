"""ToolKG-guided API sampling for ToolGrad's native ``sampler`` seam.

The fork (``Srinivasoo7/toolgrad``, ``integrate/graphgen-toolgrad``) added
two public seams this module targets:

* ``toolgrad.utils.mcp.discover_mcp_tools(mcp_dict)`` — the full discovered
  tool catalog (no sampling), used here to build the ToolKG eagerly;
* ``toolgrad.utils.mcp.get_mcp_apis(..., sampler=...)`` — the sampling step
  accepts ``sampler(tools, num_apis) -> tools``.

Usage::

    from toolgrad.prebuilt import create_graph_on_mcp
    from bridge import toolkg_sampler

    toolkg = toolkg_sampler.build_toolkg_for_mcp()   # keyless (npx)
    app = create_graph_on_mcp(
        sample_seed=123, num_apis=5, num_iterations=3,
        mcp_dict=toolgrad.utils.mcp.get_default_mcp_dict(),
        api_sampler=toolkg_sampler.make_toolkg_sampler(toolkg),
    )

No monkeypatching: the private ``_wrap_with_path_prefix`` stays inside the
fork, and discovery is never duplicated here.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import kg_sampler
from . import toolkg_builder

_DEFAULT_TOOLKG = None


def _effective_seed(seed: Any, call_count: int) -> int:
    """Deterministic int seed from ``seed`` and the call counter.

    ``random.Random`` only accepts int/float/str/bytes seeds (not tuples),
    so the pair is folded into one int arithmetically — stable across
    processes, unlike ``hash()`` of a tuple.
    """
    base = seed if isinstance(seed, int) else abs(hash(str(seed))) % (2**31)
    return base * 1_000_003 + call_count


def build_toolkg_for_mcp(
    mcp_dict: Optional[dict] = None,
    output_hints: Optional[Mapping[str, Sequence[Tuple[str, str]]]] = None,
    threshold: float = toolkg_builder.DEFAULT_THRESHOLD,
) -> Any:
    """Discover MCP tools via the fork's public discovery and build the ToolKG.

    ``output_hints`` (tool name -> [(output prop, type)]) can refine tool
    output inference; without them, outputs are inferred from descriptions
    (see ``toolkg_builder``).
    """
    from toolgrad.utils import mcp as mcp_module

    if mcp_dict is None:
        mcp_dict = mcp_module.get_default_mcp_dict()
    tools = mcp_module.discover_mcp_tools(mcp_dict)
    return toolkg_builder.build_toolkg(
        tools, output_hints=output_hints, threshold=threshold
    )


def make_toolkg_sampler(
    toolkg=None,
    seed: int = 42,
) -> Callable[[List[Any], int], List[Any]]:
    """Build a ``sampler(tools, num_apis) -> tools`` for the fork's seam.

    Samples a composable KG neighborhood per call. Uses a local RNG seeded
    with a deterministic fold of ``(seed, call_count)`` — deterministic per
    sampler instance, varying per iteration (upstream's default path reseeds
    the global RNG identically every call, which would repeat one bundle).
    """
    if toolkg is None:
        toolkg = build_toolkg_for_mcp()
    call_count = 0

    def sampler(tools: List[Any], num_apis: int) -> List[Any]:
        nonlocal call_count
        by_name: Dict[str, Any] = {t.name: t for t in tools}
        names = sorted(by_name)
        if num_apis > len(names) or num_apis <= 0:
            raise ValueError(
                f"num_apis should be between 1 and {len(names)}, got {num_apis}."
            )
        call_count += 1
        rng = random.Random(_effective_seed(seed, call_count))
        sampled_names = kg_sampler.sample_api_neighborhood(
            toolkg, num_apis, rng=rng
        )
        print(f"ToolKG sampled {len(sampled_names)} APIs: {sampled_names}")
        return [by_name[n] for n in sampled_names]

    return sampler
