"""Phase 2 — KG-neighborhood API sampling.

Replaces ToolGrad's uniform random ``sample_apis`` with sampling that walks
the ToolKG: start from a seed tool, then follow composability edges
(score-weighted), with occasional restarts. The sampled API sets therefore
reflect *realistic composable chains* instead of arbitrary tool bundles.

Also ships :func:`uniform_sample` (the baseline this replaces) and
:func:`neighborhood_density` (mean composability score over ordered pairs
in a sampled set), used by the tests to quantify the difference.
"""

from __future__ import annotations

import random
from typing import Hashable, List, Optional, Sequence


def sample_api_neighborhood(
    toolkg,
    num_apis: int,
    seed_tool: Optional[Hashable] = None,
    rng: Optional[random.Random] = None,
    restart_prob: float = 0.15,
) -> List[Hashable]:
    """Sample ``num_apis`` tool names via a score-weighted random walk.

    Starts at ``seed_tool`` (or a uniform random node), then repeatedly
    moves to an unvisited successor weighted by edge ``score``; with
    probability ``restart_prob`` (or when the walk dead-ends) it jumps to
    a random unvisited node so disconnected components don't starve the
    sample. Deterministic for a given ``rng``.
    """
    if num_apis <= 0:
        raise ValueError(f"num_apis must be positive, got {num_apis}")
    rng = rng if rng is not None else random.Random()
    nodes = list(toolkg.nodes)
    if not nodes:
        raise ValueError("cannot sample from an empty ToolKG")
    if num_apis > len(nodes):
        raise ValueError(
            f"num_apis={num_apis} exceeds ToolKG size {len(nodes)}"
        )
    start = seed_tool if seed_tool is not None else rng.choice(nodes)
    if start not in toolkg:
        raise KeyError(f"seed_tool {start!r} not in ToolKG")

    sampled: List[Hashable] = [start]
    seen = {start}
    current = start
    while len(sampled) < num_apis:
        nbrs = sorted(n for n in toolkg.successors(current) if n not in seen)
        if nbrs and rng.random() >= restart_prob:
            weights = [toolkg[current][n].get("score", 0.5) or 0.5 for n in nbrs]
            current = rng.choices(nbrs, weights=weights, k=1)[0]
        else:
            # Restart (or dead-end): jump to a random unvisited node so
            # disconnected components don't starve the sample. Always
            # appends, so the loop provably terminates with num_apis items.
            current = rng.choice([n for n in nodes if n not in seen])
        seen.add(current)
        sampled.append(current)
    return sampled


def uniform_sample(
    names: Sequence[Hashable],
    num_apis: int,
    rng: Optional[random.Random] = None,
) -> List[Hashable]:
    """Uniform random sample — the baseline ToolGrad uses (``random.sample``)."""
    rng = rng if rng is not None else random.Random()
    names = list(names)
    if num_apis <= 0 or num_apis > len(names):
        raise ValueError(
            f"num_apis should be between 1 and {len(names)}, got {num_apis}"
        )
    return rng.sample(names, num_apis)


def neighborhood_density(toolkg, names: Sequence[Hashable]) -> float:
    """Mean composability score over all ordered pairs in ``names``.

    A set of tools that chain together scores high; an arbitrary bundle
    scores near zero. Used to compare neighborhood vs uniform sampling.
    """
    names = list(names)
    n = len(names)
    if n < 2:
        return 0.0
    total = 0.0
    for a in names:
        for b in names:
            if a != b and toolkg.has_edge(a, b):
                total += toolkg[a][b].get("score", 0.0) or 0.0
    return total / (n * (n - 1))
