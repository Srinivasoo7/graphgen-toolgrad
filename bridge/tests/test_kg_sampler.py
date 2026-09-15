"""Tests for bridge/kg_sampler.py (keyless)."""

import random

import networkx as nx

from bridge import kg_sampler


def _two_chain_graph():
    """Two 3-tool chains with one weak cross edge, plus an isolate."""
    g = nx.DiGraph()
    for n in ["a1", "a2", "a3", "b1", "b2", "b3", "iso"]:
        g.add_node(n)
    for u, v in [("a1", "a2"), ("a2", "a3"), ("b1", "b2"), ("b2", "b3")]:
        g.add_edge(u, v, score=1.0, relation="composable")
    g.add_edge("a1", "b1", score=0.35, relation="composable")
    return g


def test_determinism_with_seeded_rng():
    g = _two_chain_graph()
    r1 = kg_sampler.sample_api_neighborhood(g, 3, seed_tool="a1",
                                            rng=random.Random(7))
    r2 = kg_sampler.sample_api_neighborhood(g, 3, seed_tool="a1",
                                            rng=random.Random(7))
    assert r1 == r2


def test_sample_shape_and_seed_first():
    g = _two_chain_graph()
    out = kg_sampler.sample_api_neighborhood(g, 4, seed_tool="b2",
                                             rng=random.Random(3))
    assert len(out) == 4
    assert out[0] == "b2"
    assert len(set(out)) == 4
    assert set(out) <= set(g.nodes())


def test_walk_prefers_high_score_edges():
    # From a1 the walk chooses between a2 (score 1.0) and b1 (score 0.35).
    # Score-weighting must strongly favor a2; restarts disabled so the
    # choice is purely the weighted draw.
    g = _two_chain_graph()
    first_picks = [
        kg_sampler.sample_api_neighborhood(
            g, 2, seed_tool="a1", rng=random.Random(s), restart_prob=0.0)[1]
        for s in range(200)
    ]
    a2 = first_picks.count("a2")
    b1 = first_picks.count("b1")
    assert a2 + b1 == 200
    assert a2 > 2 * b1  # edge weights are 1.0 vs 0.35


def test_disconnected_fallback_fills():
    g = _two_chain_graph()
    out = kg_sampler.sample_api_neighborhood(g, 3, seed_tool="iso",
                                             rng=random.Random(11))
    assert len(out) == 3 and len(set(out)) == 3 and out[0] == "iso"


def test_neighborhood_beats_uniform_on_density():
    g = _two_chain_graph()
    names = list(g.nodes())
    trials = 40
    nb = sum(
        kg_sampler.neighborhood_density(
            g, kg_sampler.sample_api_neighborhood(
                g, 3, seed_tool="a1", rng=random.Random(s)))
        for s in range(trials)
    ) / trials
    un = sum(
        kg_sampler.neighborhood_density(
            g, kg_sampler.uniform_sample(names, 3, random.Random(1000 + s)))
        for s in range(trials)
    ) / trials
    print(f"neighborhood mean density={nb:.3f} vs uniform={un:.3f}")
    assert nb > un


def test_errors():
    g = _two_chain_graph()
    for bad in (0, -1):
        try:
            kg_sampler.sample_api_neighborhood(g, bad, rng=random.Random(1))
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    try:
        kg_sampler.sample_api_neighborhood(g, 99, rng=random.Random(1))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        kg_sampler.sample_api_neighborhood(nx.DiGraph(), 1,
                                           rng=random.Random(1))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        kg_sampler.sample_api_neighborhood(g, 2, seed_tool="nope",
                                           rng=random.Random(1))
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
