"""Tests for bridge.probe_common (backend-agnostic probe plumbing).

Covers table loading, column stats, and the findings -> corpus narration
with a Kumo stub factory so no model weights are needed.
"""

import json
import os
import sys
import tempfile

import pytest

from bridge.kumo_probe import probe_tables
from bridge.probe_common import (
    ProbeError,
    column_stats,
    findings_to_markdown,
    load_tables_dir,
    write_tables_corpus,
)


class _KumoStubFactory:
    """Deterministic stub estimator: majority-class classifier."""

    def __call__(self, task: str):
        return _KumoStubEstimator()


class _KumoStubEstimator:
    def fit(self, rows, target, feature_cols, feature_types):
        self._majority = max(
            {r.get(target) for r in rows if r.get(target) is not None},
            key=lambda v: sum(1 for r in rows if r.get(target) == v),
            default=None,
        )

    def predict(self, rows):
        return [self._majority for _ in rows]


def _write_csv(tmpdir, name, rows):
    path = os.path.join(tmpdir, f"{name}.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("ticket_id,priority,channel,hours_open,status\n")
        for i, r in enumerate(rows):
            fh.write(
                f"{i},{r[0]},{r[1]},{r[2]},{r[3]}\n"
            )
    return path


def _tickets(n):
    rows = []
    for i in range(n):
        pri = ["P1", "P2", "P3"][i % 3]
        status = (
            "escalated"
            if pri == "P1" and i % 10 != 0
            else ("open" if i % 2 == 0 else "closed")
        )
        rows.append((pri, "email" if i % 2 == 0 else "chat", i % 48, status))
    return rows


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def test_load_tables_dir_csv_and_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_csv(tmpdir, "tickets", _tickets(5))
        with open(os.path.join(tmpdir, "users.json"), "w") as fh:
            json.dump([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}], fh)
        tables = load_tables_dir(tmpdir)
        assert set(tables) == {"tickets", "users"}
        assert len(tables["tickets"]) == 5
        assert tables["tickets"][0]["hours_open"] == 0  # coerced to int
        assert tables["users"][1]["name"] == "b"


def test_load_tables_dir_missing_raises():
    with pytest.raises(ProbeError):
        load_tables_dir("/no/such/dir")


# ---------------------------------------------------------------------------
# column stats
# ---------------------------------------------------------------------------


def test_column_stats_numeric_and_categorical():
    rows = [
        {"n": 1, "c": "a"},
        {"n": 2, "c": "b"},
        {"n": None, "c": "a"},
    ]
    n = column_stats(rows, "n")
    assert n["type"] == "numeric"
    assert n["cardinality"] == 2
    assert n["missing_frac"] == pytest.approx(1 / 3, abs=1e-4)
    assert n["mean"] == 1.5
    c = column_stats(rows, "c")
    assert c["type"] == "categorical"
    assert c["top_values"][0] == {"value": "a", "count": 2}


# ---------------------------------------------------------------------------
# narration: findings documents contain aggregates, never raw rows
# ---------------------------------------------------------------------------


def test_findings_docs_written_without_raw_rows():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_csv(tmpdir, "tickets", _tickets(60))
        tables = load_tables_dir(tmpdir)
        findings = probe_tables(
            tables,
            targets_by_table={"tickets": "status"},
            probe_factory=_KumoStubFactory(),
        )
        with tempfile.TemporaryDirectory() as outdir:
            paths = write_tables_corpus(findings, outdir, llm_fn=None)
            assert any(p.endswith("tickets.md") for p in paths)
            text = findings_to_markdown(findings["tables"][0])
            assert "# Table: tickets" in text
            assert "60 rows" in text
            # deterministic doc has exactly the audited content
            with open(os.path.join(outdir, "tickets.md")) as fh:
                assert fh.read() == text
            # no raw rows / ticket ids leak into the findings doc
            assert "P1" in text  # aggregated top-values are fine
            assert "\n0," not in text
            # findings.json is the audit artifact
            assert os.path.exists(os.path.join(outdir, "findings.json"))


def test_findings_docs_llm_narrative_added_when_llm_given():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_csv(tmpdir, "tickets", _tickets(12))
        tables = load_tables_dir(tmpdir)
        findings = probe_tables(
            tables,
            targets_by_table={"tickets": "status"},
            probe_factory=_KumoStubFactory(),
        )
        with tempfile.TemporaryDirectory() as outdir:
            paths = write_tables_corpus(
                findings, outdir, llm_fn=lambda prompt: "brief"
            )
            assert any(p.endswith("tickets_narrative.md") for p in paths)
            with open(os.path.join(outdir, "tickets_narrative.md")) as fh:
                assert fh.read().strip() == "brief"
