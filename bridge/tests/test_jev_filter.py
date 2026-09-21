"""Tests for bridge/jev_filter.py (keyless: stub decider, no network)."""

from bridge import jev_filter
from bridge.jev_filter import JevDecider, build_request, jev_filter_pairs
from bridge.refinement_loop import RefineConfig


def _qa(**over):
    qa = {
        "question": "What does Gigafactory Texas produce?",
        "answer_draft": "Model Y and Cybertruck, per the notes file.",
        "entity_refs": ["Gigafactory Texas"],
        "required_tools": ["list_directory", "read_text_file"],
        "chain": [
            {"tool": "list_directory", "tool_input": {"path": "/x"},
             "result_preview": "{'files': [...]}"},
            {"tool": "read_text_file", "tool_input": {"path": "/x/notes.txt"},
             "result_preview": "Gigafactory Texas produces..."},
        ],
    }
    qa.update(over)
    return qa


def _score(**over):
    score = {
        "chain_valid": True,
        "verification": {"chain_valid": True},
        "coverage": 1.0,
        "entity_grounded": True,
        "entity_refs": ["Gigafactory Texas"],
        "required_tools": ["list_directory", "read_text_file"],
        "question_tokens": 10,
        "answer_tokens": 12,
    }
    score.update(over)
    return score


class _StubDecider:
    """Deterministic stand-in for JevDecider (no subprocess, no network)."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.calls = 0

    def decide(self, qa, score, config=None):
        verdict = self.verdicts[self.calls % len(self.verdicts)]
        self.calls += 1
        return {
            "verdict": verdict,
            "confidence": 0.9 if verdict == "accept" else 0.85,
            "grounded": 1.0,
            "usage": {"input_tokens": 300, "output_tokens": 0},
            "latency_ms": 210.0,
            "model": "jev-stub",
            "source": "jev",
            "error": None,
        }


def test_request_shape():
    req = build_request(_qa(), _score())
    assert req["model"] == "jev-latest"
    assert set(req["questions"]) == {"verdict", "grounded"}
    assert req["questions"]["verdict"]["type"] == "choice"
    assert set(req["questions"]["verdict"]["criteria"]) == {"accept", "reject"}
    assert req["questions"]["grounded"]["type"] == "noul"
    assert req["state"]["keyless_scores"]["coverage"] == 1.0
    assert len(req["state"]["chain"]) == 2


def test_filter_splits_on_verdict():
    config = RefineConfig()
    scored = [(_qa(), _score()), (_qa(), _score(chain_valid=False)),
              (_qa(), _score())]
    kept, rejected = jev_filter_pairs(scored, config, _StubDecider(["accept", "reject", "accept"]))
    assert len(kept) == 2 and len(rejected) == 1
    # kept retains (qa, score) tuples; rejected is plain qa dicts
    assert all(isinstance(item, tuple) for item in kept)
    assert all(isinstance(item, dict) for item in rejected)
    # every kept pair carries the jev annotation
    for qa, _ in kept:
        assert qa["jev"]["verdict"] == "accept"
        assert qa["jev"]["confidence"] == 0.9
        assert qa["jev"]["model"] == "jev-stub"
    assert rejected[0]["jev"]["verdict"] == "reject"


def test_fallback_uses_heuristic_gates_on_broken_cli():
    # Real JevDecider with an unusable CLI: no network, subprocess fails
    # immediately, and decide() must fall back to the heuristic gates.
    decider = JevDecider(cli="/nonexistent/jev_decide.py", timeout=5.0)
    config = RefineConfig()
    good = (_qa(), _score())
    bad = (_qa(), _score(chain_valid=False, entity_grounded=False, coverage=0.0))
    d_good = decider.decide(good[0], good[1], config)
    d_bad = decider.decide(bad[0], bad[1], config)
    assert d_good["source"] == "heuristic_fallback" and d_good["verdict"] == "accept"
    assert d_bad["source"] == "heuristic_fallback" and d_bad["verdict"] == "reject"
    assert d_good["error"]  # the failure is recorded, not swallowed
    # and the filter contract still holds end to end
    kept, rejected = jev_filter_pairs([good, bad], config, decider)
    assert len(kept) == 1 and len(rejected) == 1
    assert kept[0][1] is good[1]


def test_parse_decision_uses_p_accept():
    # P(accept) is the decision signal, not the opaque top-level confidence
    verdict, p, grounded = jev_filter._parse_decision(
        {"answers": {"verdict": {"choice": "accept", "confidence": 0.5,
                                 "probabilities": {"accept": 0.75, "reject": 0.25}},
                     "grounded": {"noul": 0.9}}},
        min_confidence=0.5,
    )
    assert verdict == "accept" and p == 0.75 and grounded == 0.9
    verdict, p, _ = jev_filter._parse_decision(
        {"answers": {"verdict": {"choice": "reject", "confidence": 0.44,
                                 "probabilities": {"accept": 0.28, "reject": 0.72}}}},
        min_confidence=0.5,
    )
    assert verdict == "reject" and p == 0.28


def test_parse_decision_rejects_garbage():
    for bad in ({}, {"answers": {}},
                {"answers": {"verdict": {"choice": "accept"}}},  # no probabilities
                {"answers": {"verdict": {"probabilities": {"accept": 1.5}}}}):
        try:
            jev_filter._parse_decision(bad, min_confidence=0.5)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")
