"""Tests for bridge/sft_mix.py (keyless)."""

import json
import os
import tempfile

from bridge import sft_mix
from bridge.sft_mix import MixConfig


def _qa(question, valid=True, coverage=1.0, refs=("Tesla",)):
    return {
        "question": question,
        "answer_draft": "An answer grounded in the chain.",
        "required_tools": ["list_directory"],
        "chain": [{"tool": "list_directory", "tool_input": {"path": "/x"},
                   "result_preview": "..."}],
        "entity_refs": list(refs),
        "provenance": {"source": "toolgrad_trace"},
        "verification": {"chain_valid": valid},
        "coverage": coverage,
    }


def test_assemble_filters_dedupe_splits():
    pairs = [
        _qa("What does Tesla produce?"),
        _qa("Where is Gigafactory Texas?"),
        _qa("Broken pair", valid=False),
        _qa("what does tesla produce?"),  # duplicate after normalization
    ]
    train, valid, report = sft_mix.assemble(
        pairs, config=MixConfig(valid_frac=0.25, seed=1))
    assert report["n_bridge_in"] == 4
    assert report["n_bridge_after_cutoffs"] == 3
    assert report["n_bridge_dupes_removed"] == 1
    assert report["n_records_total"] == 2
    assert report["n_train"] == 1 and report["n_valid"] == 1
    assert len(train) + len(valid) == 2
    for rec in train + valid:
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert rec["provenance"]["source"] == "graphgen_toolgrad_bridge"
        assert rec["provenance"]["verification"]["chain_valid"] is True


def test_graphgen_rows_normalized_and_tagged():
    chatml = {"messages": [{"role": "user", "content": "Q"},
                           {"role": "assistant", "content": "A"}],
              "topic": "ev"}
    sharegpt = {"conversations": [{"from": "human", "value": "Q2"},
                                 {"from": "gpt", "value": "A2"}]}
    train, valid, report = sft_mix.assemble(
        [], graphgen_pairs=[chatml, sharegpt], config=MixConfig(seed=3))
    assert report["n_graphgen_in"] == 2
    assert report["sources"] == {"graphgen": 2}
    recs = {r["messages"][0]["content"]: r for r in train + valid}
    assert recs["Q"]["provenance"]["topic"] == "ev"  # extra keys preserved
    sg = recs["Q2"]
    assert [m["role"] for m in sg["messages"]] == ["user", "assistant"]


def test_normalize_graphgen_record_rejects_unknown_shape():
    try:
        sft_mix.normalize_graphgen_record({"weird": 1})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_to_sft_record_carries_provenance():
    qa = _qa("What does Tesla produce?")
    rec = sft_mix.to_sft_record(qa)
    assert qa["question"] in rec["messages"][1]["content"]
    prov = rec["provenance"]
    assert prov["entity_refs"] == ["Tesla"]
    assert prov["required_tools"] == ["list_directory"]
    assert prov["chain"] == qa["chain"]
    assert prov["coverage"] == 1.0


def test_write_jsonl_and_dataset_card():
    pairs = [_qa("What does Tesla produce?"), _qa("Where is Gigafactory Texas?")]
    train, valid, report = sft_mix.assemble(pairs, config=MixConfig(seed=5))
    with tempfile.TemporaryDirectory() as d:
        jp = sft_mix.write_jsonl(train + valid, os.path.join(d, "sft_mix.jsonl"))
        lines = open(jp).read().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["messages"]
        cp = sft_mix.write_dataset_card(report, None, os.path.join(d, "card.md"))
        card = open(cp).read()
    assert "train 1 / valid 1" in card
    assert "require_chain_valid: True" in card
