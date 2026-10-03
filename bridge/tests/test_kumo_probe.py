"""Tests for bridge/kumo_probe.py (keyless).

A stub factory implements the raw-rows estimator protocol, so no sdm
package, no weights, no GPU/CPU inference is needed.
"""

from collections import Counter
from contextlib import contextmanager

from bridge import kumo_probe
from bridge.kumo_probe import KumoProbeFactory, ProbeError


@contextmanager
def _raises(exc_type):
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


class _MemorizingEstimator:
    """Raw-rows protocol: fit(rows, target, feature_cols, feature_types)."""

    def fit(self, rows, target, feature_cols, feature_types):
        self.target = target
        self.key_col = feature_cols[0]
        self.majority = Counter(r[target] for r in rows).most_common(1)[0][0]
        by_key = {}
        for r in rows:
            by_key.setdefault(r[self.key_col], Counter())[r[target]] += 1
        self.by_key = {k: c.most_common(1)[0][0] for k, c in by_key.items()}
        return self

    def predict(self, rows):
        return [self.by_key.get(r[self.key_col], self.majority) for r in rows]


def _stub_factory(task):
    return _MemorizingEstimator()


def _ticket_rows(n=40):
    rows = []
    for i in range(n):
        high = i % 2 == 0
        rows.append({
            "priority": "High" if high else "Low",
            "channel": "email" if i % 3 else "phone",
            "status": "escalated" if high else "closed",
        })
    return rows


def test_probe_table_finds_determining_feature():
    findings = kumo_probe.probe_table(
        "tickets", _ticket_rows(), targets=["status"], probe_factory=_stub_factory
    )
    assert findings["backend"] == "kumo"
    tgt = findings["targets"][0]
    assert tgt["column"] == "status" and tgt["task"] == "classification"
    assert tgt["holdout_score"] == 1.0
    assert tgt["importances"][0]["feature"] == "priority"
    assert tgt["importances"][0]["importance"] > 0


def test_probe_table_empty_raises():
    with _raises(ProbeError):
        kumo_probe.probe_table("t", [], probe_factory=_stub_factory)


def test_probe_table_unknown_target_raises():
    with _raises(ProbeError):
        kumo_probe.probe_table(
            "t", _ticket_rows(5), targets=["nope"], probe_factory=_stub_factory
        )


def test_factory_rejects_bad_size():
    with _raises(ProbeError):
        KumoProbeFactory(size="huge")


def test_factory_requires_sdm_package():
    factory = KumoProbeFactory()
    try:
        import sdm  # noqa: F401
        have_sdm = True
    except ImportError:
        have_sdm = False
    if have_sdm:
        return  # live env: nothing to assert keylessly
    with _raises(ProbeError):
        factory("classification")


def test_tstr_parity_identical_distribution_is_one():
    real = _ticket_rows()
    result = kumo_probe.tstr_parity(
        real, _ticket_rows(), "status", probe_factory=_stub_factory
    )
    assert result["backend"] == "kumo"
    assert result["parity"] == 1.0


def test_probe_tables_marks_generated_by():
    out = kumo_probe.probe_tables(
        {"tickets": _ticket_rows()}, {"tickets": "status"},
        probe_factory=_stub_factory,
    )
    assert out["generated_by"] == "kumo_probe"
    assert out["tables"][0]["table"] == "tickets"
