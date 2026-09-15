"""Tests for bridge/eval_harness.py (keyless)."""

from bridge import eval_harness
from bridge.tests import _fixtures


def _qa(question, answer, tools, refs):
    steps = [{"tool": t, "tool_input": {"path": "/x"}, "result_preview": "..."}
             for t in tools]
    return {
        "question": question,
        "answer_draft": answer,
        "required_tools": tools,
        "chain": steps,
        "entity_refs": list(refs),
        "provenance": {},
    }


def _pairs():
    return [
        _qa("What does Gigafactory Texas produce?",
            "Model Y and Cybertruck, per the notes.",
            ["list_directory", "read_text_file"],
            ["Tesla", "Gigafactory Texas"]),
        _qa("What is missing?",
            "Unknown.",
            ["list_directory", "nope_tool"],
            []),
        _qa("Tell me about Tesla.",
            "An electric vehicle manufacturer.",
            ["list_directory"],
            ["Tesla"]),
    ]


def test_evaluate_known_dataset():
    m = eval_harness.evaluate(
        _pairs(),
        kg_context=_fixtures.kg_context(),
        toolkg=_fixtures.toolkg(),
        executor=_fixtures.executor(),
    )
    assert m["n"] == 3
    assert m["fraction_chain_verified"] == 2 / 3
    assert m["entity_grounding_rate"] == 2 / 3
    # list_directory -> read_text_file is a ToolKG edge; single-step chain -> 0.0
    assert m["mean_toolkg_coverage"] == (1.0 + 0.0 + 0.0) / 3
    div = m["tool_diversity"]
    assert div["distinct_tools"] == 3  # list_directory, read_text_file, nope_tool
    assert div["total_steps"] == 5
    assert div["most_common"][0][0] == "list_directory"
    assert m["length_stats"]["question_tokens"]["n"] == 3
    assert m["limits"]  # honest limits are always reported


def test_parametric_answerability_flag():
    kg = _fixtures.kg_context()
    verbatim = "Tesla factory in Austin producing Model Y and Cybertruck"
    pairs = [
        _qa("Where is the Tesla factory?", verbatim, ["list_directory"], ["Tesla"]),
        _qa("What else?", "Something entirely unrelated to any factory records here.",
            ["list_directory"], ["Tesla"]),
    ]
    flags = eval_harness.parametric_answerability_flags(pairs, kg)
    assert flags["flagged"] == 1
    assert flags["flagged_indices"] == [0]
    assert flags["fraction"] == 0.5


def test_no_executor_reports_none_not_zero():
    m = eval_harness.evaluate(_pairs(), kg_context=_fixtures.kg_context())
    assert m["fraction_chain_verified"] is None
    assert "verification_note" in m
    assert m["mean_toolkg_coverage"] is None  # no ToolKG either
    assert "coverage_note" in m


def test_precomputed_verifications_used():
    pairs = _pairs()
    verifs = [{"chain_valid": True}, {"chain_valid": True}, {"chain_valid": False}]
    m = eval_harness.evaluate(pairs, verifications=verifs)
    assert m["fraction_chain_verified"] == 2 / 3
    assert "verification_note" not in m


def test_verifications_length_mismatch_raises():
    try:
        eval_harness.evaluate(_pairs(), verifications=[{"chain_valid": True}])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_empty_dataset():
    m = eval_harness.evaluate([], kg_context=_fixtures.kg_context(),
                              executor=_fixtures.executor())
    assert m["n"] == 0
    assert m["fraction_chain_verified"] == 0.0
    assert m["entity_grounding_rate"] == 0.0
    assert m["parametric_answerability"]["flagged"] == 0
