"""Tests for bridge/tabpfn_probe.py + the kg_source='tables' path (keyless).

A stub probe factory stands in for TabPFN: a memorizing estimator keyed on
the first feature. No tabpfn package, no model download, no network.
"""

import json
import os
import tempfile
from collections import Counter
from contextlib import contextmanager

from bridge import config as config_mod
from bridge import kg_builder, tabpfn_probe
from bridge.tabpfn_probe import ProbeError


@contextmanager
def _raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


class _MemorizingEstimator:
    """Predicts by looking up feature 0; falls back to the majority label."""

    def fit(self, X, y):
        self.majority = Counter(y).most_common(1)[0][0]
        by_key = {}
        for row, target in zip(X, y):
            by_key.setdefault(row[0], Counter())[target] += 1
        self.by_key = {k: c.most_common(1)[0][0] for k, c in by_key.items()}
        return self

    def predict(self, X):
        return [self.by_key.get(row[0], self.majority) for row in X]


def _stub_factory(task):
    return _MemorizingEstimator()


def _ticket_rows(n=40):
    """status is fully determined by priority; channel is noise."""
    rows = []
    for i in range(n):
        high = i % 2 == 0
        rows.append({
            "priority": "High" if high else "Low",
            "channel": "email" if i % 3 else "phone",
            "status": "escalated" if high else "closed",
        })
    return rows


def _write_csv(dirpath, name, rows):
    import csv

    with open(os.path.join(dirpath, name), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# loading + stats
# ---------------------------------------------------------------------------


def test_load_tables_dir_csv_and_json():
    with tempfile.TemporaryDirectory() as d:
        _write_csv(d, "tickets.csv", _ticket_rows(5))
        with open(os.path.join(d, "counts.json"), "w") as fh:
            json.dump([{"n": "3"}, {"n": "4"}], fh)
        tables = tabpfn_probe.load_tables_dir(d)
    assert set(tables) == {"tickets", "counts"}
    assert tables["counts"][0]["n"] == 3  # coerced to int
    assert tables["tickets"][0]["priority"] == "High"


def test_load_tables_dir_missing_raises():
    with _raises(ProbeError):
        tabpfn_probe.load_tables_dir("/nonexistent-tables-dir")


def test_column_stats_numeric_and_categorical():
    rows = [{"a": 1, "b": "x"}, {"a": 3, "b": "x"}, {"a": None, "b": "y"}]
    a = tabpfn_probe.column_stats(rows, "a")
    assert a["type"] == "numeric" and a["min"] == 1 and a["max"] == 3
    assert a["missing_frac"] > 0
    b = tabpfn_probe.column_stats(rows, "b")
    assert b["type"] == "categorical" and b["cardinality"] == 2
    assert b["top_values"][0] == {"value": "x", "count": 2}


# ---------------------------------------------------------------------------
# probing with the stub factory
# ---------------------------------------------------------------------------


def test_probe_table_finds_determining_feature():
    findings = tabpfn_probe.probe_table(
        "tickets", _ticket_rows(), probe_factory=_stub_factory
    )
    assert findings["num_rows"] == 40
    tgt = [t for t in findings["targets"] if t["column"] == "status"][0]
    assert tgt["column"] == "status" and tgt["task"] == "classification"
    assert tgt["holdout_score"] == 1.0  # priority determines status exactly
    assert tgt["importances"][0]["feature"] == "priority"
    assert tgt["importances"][0]["importance"] > 0
    noise = [i for i in tgt["importances"] if i["feature"] == "channel"][0]
    assert noise["importance"] == 0


def test_probe_table_explicit_numeric_target_is_regression():
    rows = [{"size": float(i % 5), "price": float(10 * (i % 5))} for i in range(30)]
    findings = tabpfn_probe.probe_table(
        "sales", rows, targets=["price"], probe_factory=_stub_factory
    )
    assert findings["targets"][0]["task"] == "regression"
    assert findings["targets"][0]["holdout_score"] == 1.0


def test_probe_table_unknown_target_raises():
    with _raises(ProbeError):
        tabpfn_probe.probe_table(
            "t", _ticket_rows(5), targets=["nope"], probe_factory=_stub_factory
        )


def test_probe_table_empty_raises():
    with _raises(ProbeError):
        tabpfn_probe.probe_table("t", [], probe_factory=_stub_factory)


def test_tstr_parity_identical_distribution_is_one():
    real = _ticket_rows()
    synth = _ticket_rows()
    result = tabpfn_probe.tstr_parity(real, synth, "status", probe_factory=_stub_factory)
    assert result["real_score"] == 1.0
    assert result["parity"] == 1.0


def test_tstr_parity_broken_synth_scores_lower():
    real = _ticket_rows()
    broken = [dict(r, status="closed") for r in _ticket_rows()]
    result = tabpfn_probe.tstr_parity(real, broken, "status", probe_factory=_stub_factory)
    assert result["parity"] is not None and result["parity"] < 1.0


# ---------------------------------------------------------------------------
# narration docs
# ---------------------------------------------------------------------------


def test_findings_docs_written_without_raw_rows():
    findings = tabpfn_probe.probe_tables(
        {"tickets": _ticket_rows()}, probe_factory=_stub_factory
    )
    with tempfile.TemporaryDirectory() as d:
        paths = tabpfn_probe.write_tables_corpus(findings, d)
        doc = open(os.path.join(d, "tickets.md")).read()
        assert os.path.join(d, "findings.json") not in paths  # audit file aside
    assert "## Measured relationships" in doc
    assert "priority" in doc and "status" in doc


def test_findings_docs_llm_narrative_added_when_llm_given():
    findings = tabpfn_probe.probe_tables(
        {"tickets": _ticket_rows()}, probe_factory=_stub_factory
    )
    with tempfile.TemporaryDirectory() as d:
        tabpfn_probe.write_tables_corpus(
            findings, d, llm_fn=lambda prompt: "Tickets escalate when priority is High."
        )
        narrative = open(os.path.join(d, "tickets_narrative.md")).read()
    assert "escalate" in narrative


# ---------------------------------------------------------------------------
# kg_source='tables' end to end (KG builder + config)
# ---------------------------------------------------------------------------


def test_build_kg_from_tables_structural():
    with tempfile.TemporaryDirectory() as d:
        _write_csv(d, "tickets.csv", _ticket_rows())
        g = kg_builder.build_kg(
            "tables", tables_dir=d, probe_factory=_stub_factory, llm_fn=None
        )
    assert "tickets" in g.nodes
    assert "tickets.priority" in g.nodes
    edge = g.get_edge_data("tickets.priority", "tickets.status")
    assert edge is not None and edge["relation"] == "predicts"


def test_build_kg_from_tables_merges_llm_extraction():
    def fake_llm(prompt):
        if prompt.startswith("Rewrite"):
            return "High-priority tickets get escalated."
        return json.dumps({
            "entities": [{"name": "Escalation", "type": "concept",
                          "description": "priority-driven routing"}],
            "relations": [],
        })

    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as work:
        _write_csv(d, "tickets.csv", _ticket_rows())
        g = kg_builder.build_kg(
            "tables", tables_dir=d, probe_factory=_stub_factory,
            llm_fn=fake_llm, tables_work_dir=work,
        )
        assert os.path.exists(os.path.join(work, "findings.json"))
    assert "Escalation" in g.nodes
    assert "tickets" in g.nodes


def test_build_kg_tables_requires_dir():
    with _raises(kg_builder.KGBuildError):
        kg_builder.build_kg("tables", tables_dir="", probe_factory=_stub_factory)


def _base_config(domain):
    return {
        "name": "t",
        "mcp_servers": [{"name": "s", "transport": "stdio", "command": "true"}],
        "domain": domain,
    }


def test_config_tables_source_requires_tables_dir():
    with _raises(config_mod.ConfigError):
        config_mod.from_dict(_base_config({"kg_source": "tables"}))


def test_config_tables_source_parses_targets():
    cfg = config_mod.from_dict(_base_config({
        "kg_source": "tables",
        "tables_dir": "./data/x",
        "table_targets": {"tickets": "status"},
    }))
    assert cfg.domain.kg_source == "tables"
    assert cfg.domain.table_targets == {"tickets": "status"}
