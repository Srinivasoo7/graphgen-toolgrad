"""Kumo Tabular probe backend (NVIDIA).

Measures what enterprise tables say, before the KG: Kumo is an
*analyzer*, not a reader — it consumes a table sample and returns
measured findings (which columns predict which outcomes, permutation
feature importance). It writes no prose and builds no graph. The
findings are then narrated (deterministically here; an LLM may add a
prose brief) into documents the existing corpus KG extraction consumes
unchanged, and a structural graph is derived from the findings
directly.

Pipeline position: **probe -> KG -> ToolGrad** (``domain.kg_source:
tables``). Privacy note: narration input is findings and aggregates
only — raw rows never leave this module.

The weights are OpenMDW-1.1 (commercial use permitted per NVIDIA),
downloaded from Hugging Face (``nvidia/Kumo-Tabular``) on first use —
no license click-through, no token. ``size="small"`` (28M params) is
the default: it loads in ~30s on CPU and infers a few rows per second,
which is plenty for a probe stage over sampled tables.

Kumo itself is an optional dependency, imported lazily inside
:class:`KumoProbeFactory`. Tests inject a stub factory, so the whole
stage is exercisable keylessly and model-lessly. A missing package is
an explicit :class:`ProbeError`, never a silent pass.

Environment note: on hosts where the ``no_proxy``/``NO_PROXY`` variables
contain bracketed IPv6 literals (``[::1]``), the vendored ``httpx2``
parser fails URL parsing during the weight download. Export cleaned
values (e.g. ``no_proxy=localhost,127.0.0.1,::1``) before first use.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from bridge.probe_common import ProbeError, _score, _split, column_stats

# A factory maps a task ("classification" | "regression") to an estimator
# with fit(rows, target, feature_cols, feature_types) / predict(rows)
# operating on RAW row dicts (Kumo handles its own encoding via sdm).
ProbeFactory = Callable[[str], Any]

_MISSING_CAT = "__missing__"


class KumoProbeFactory:
    """Real probe: KumoTabular estimators, imported lazily on first use."""

    def __init__(
        self,
        size: str = "small",
        device: str = "cpu",
        num_estimators: int = 8,
    ) -> None:
        if size not in ("small", "medium", "large"):
            raise ProbeError(f"unknown Kumo size {size!r}")
        self.size = size
        self.device = device
        self.num_estimators = num_estimators

    def __call__(self, task: str) -> "_KumoEstimator":
        try:
            import sdm  # noqa: F401
        except ImportError as exc:
            raise ProbeError(
                "the Kumo probe needs the 'structured-data-models' package "
                "(pip install git+https://github.com/NVIDIA/structured-data-models.git)"
            ) from exc
        return _KumoEstimator(task, self.size, self.device, self.num_estimators)


class _KumoEstimator:
    """In-context estimator: fit() stores the labeled context, predict()
    runs KumoTabular against it. Stateless across instances."""

    def __init__(self, task: str, size: str, device: str, num_estimators: int) -> None:
        self.task = task
        self.size = size
        self.device = device
        self.num_estimators = num_estimators
        self._model: Any = None
        self._context: Any = None
        self._y_context: Any = None
        self._feature_cols: List[str] = []
        self._cat_maps: Dict[str, Dict[Any, int]] = {}
        self._num_medians: Dict[str, float] = {}

    # -- encoding ------------------------------------------------------
    def _encode_frame(self, rows: Sequence[Dict[str, Any]], fit: bool):
        import pandas as pd

        data: Dict[str, List[Any]] = {c: [] for c in self._feature_cols}
        for r in rows:
            for c in self._feature_cols:
                v = r.get(c)
                if self._feature_types[c] == "categorical":
                    if v is None:
                        v = _MISSING_CAT
                    if fit:
                        self._cat_maps.setdefault(c, {})
                        if v not in self._cat_maps[c]:
                            self._cat_maps[c][v] = len(self._cat_maps[c])
                    data[c].append(self._cat_maps[c].get(v, -1))
                else:
                    data[c].append(float(v) if v is not None else float("nan"))
        df = pd.DataFrame(data)
        if fit:
            for c in self._feature_cols:
                if self._feature_types[c] != "categorical":
                    self._num_medians[c] = float(df[c].median())
        for c in self._feature_cols:
            if self._feature_types[c] != "categorical":
                df[c] = df[c].fillna(self._num_medians[c])
        return df

    # -- sklearn-shaped API over raw rows --------------------------------
    def fit(
        self,
        rows: Sequence[Dict[str, Any]],
        target: str,
        feature_cols: Sequence[str],
        feature_types: Dict[str, str],
    ) -> "_KumoEstimator":
        import sdm

        self._feature_cols = list(feature_cols)
        self._feature_types = dict(feature_types)
        df = self._encode_frame(rows, fit=True)
        stypes = {c: "numerical" for c in self._feature_cols}
        if self.task == "classification":
            # raw labels: sdm encodes the categorical target itself and
            # names the output probability columns by class value, which
            # is the only reliable index -> label mapping.
            df["_target"] = [r.get(target) for r in rows]
            stypes["_target"] = "categorical"
        else:
            df["_target"] = [float(r.get(target)) for r in rows]
            stypes["_target"] = "numerical"
        self._context = sdm.TableTensor.from_pandas(
            df=df, stypes=stypes, device=self.device
        )
        self._y_context = self._context[:, "_target"]
        self._x_context = self._context.drop_columns("_target")
        self._model = sdm.models.KumoTabular(
            task=self.task, size=self.size, device=self.device
        )
        return self

    def predict(self, rows: Sequence[Dict[str, Any]]) -> List[Any]:
        import sdm  # noqa: F401  (imported for side-effect consistency)

        df = self._encode_frame(rows, fit=False)
        stypes = {c: "numerical" for c in self._feature_cols}
        query = sdm.TableTensor.from_pandas(
            df=df, stypes=stypes, device=self.device
        )
        out = self._model(
            x_context=self._x_context,
            y_context=self._y_context,
            x_query=query,
            num_estimators=self.num_estimators,
        )
        # NOTE: sdm returns a TableTensor whose __torch_dispatch__ blocks
        # raw aten ops (argmax, tolist, ...). Its _numerical block is the
        # plain probability/quantile tensor. For classification the output
        # columns are named by class value, in sdm's encoding order — the
        # only reliable index -> label mapping.
        from sdm.tensor.table import Stype

        plain = out._numerical
        if self.task == "classification":
            col_names = list(out._columns[Stype.numerical])
            idx = plain.argmax(dim=-1).tolist()
            return [col_names[i] for i in idx]
        return plain[..., 499].tolist()  # median of the 999 quantiles


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------


def _probe_target(
    rows: Sequence[Dict[str, Any]],
    target: str,
    feature_cols: Sequence[str],
    stats_by_col: Dict[str, Dict[str, Any]],
    probe_factory: ProbeFactory,
    seed: int,
) -> Dict[str, Any]:
    task = (
        "regression"
        if stats_by_col[target]["type"] == "numeric"
        else "classification"
    )
    usable = [r for r in rows if r.get(target) is not None]
    feature_types = {
        c: ("categorical" if stats_by_col[c]["type"] == "categorical" else "numeric")
        for c in feature_cols
    }
    train_rows, test_rows = _split_rows(usable, seed)
    est = probe_factory(task)
    est.fit(train_rows, target, feature_cols, feature_types)
    y_test = [r.get(target) for r in test_rows]
    base_score = _score(task, y_test, list(est.predict(test_rows)))
    if task == "classification":
        counts: Dict[Any, int] = {}
        for v in y_test:
            counts[v] = counts.get(v, 0) + 1
        baseline = round(max(counts.values()) / len(y_test), 4) if y_test else 0.0
    else:
        baseline = 0.0  # R2 of the mean predictor
    importances = []
    rng = random.Random(seed + 1)
    for col in feature_cols:
        shuffled = [dict(r) for r in test_rows]
        column = [r.get(col) for r in shuffled]
        rng.shuffle(column)
        for r, v in zip(shuffled, column):
            r[col] = v
        dropped = _score(task, y_test, list(est.predict(shuffled)))
        importances.append(
            {"feature": col, "importance": round(base_score - dropped, 4)}
        )
    importances.sort(key=lambda d: (-d["importance"], d["feature"]))
    return {
        "column": target,
        "task": task,
        "holdout_score": base_score,
        "baseline_score": baseline,
        "importances": importances,
    }


def _split_rows(
    rows: Sequence[Dict[str, Any]], seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_idx, test_idx = _split(len(rows), seed)
    return [rows[i] for i in train_idx], [rows[i] for i in test_idx]


def probe_table(
    name: str,
    rows: Sequence[Dict[str, Any]],
    targets: Optional[Sequence[str]] = None,
    probe_factory: Optional[ProbeFactory] = None,
    max_rows: int = 1000,
    max_targets: int = 3,
    seed: int = 42,
    size: str = "small",
    device: str = "cpu",
    num_estimators: int = 8,
) -> Dict[str, Any]:
    """Probe one table with Kumo Tabular. Same findings shape as
    :func:`bridge.kumo_probe.probe_tables`."""
    if not rows:
        raise ProbeError(f"table {name!r} has no rows")
    factory = probe_factory or KumoProbeFactory(
        size=size, device=device, num_estimators=num_estimators
    )
    sampled = list(rows)
    if len(sampled) > max_rows:
        sampled = random.Random(seed).sample(sampled, max_rows)
    columns: List[str] = []
    for r in sampled:
        for key in r:
            if key not in columns:
                columns.append(key)
    stats = [column_stats(sampled, c) for c in columns]
    stats_by_col = {s["name"]: s for s in stats}
    if targets:
        chosen = list(targets)
        for t in chosen:
            if t not in stats_by_col:
                raise ProbeError(f"table {name!r}: unknown target column {t!r}")
    else:
        # Kumo's classifier head supports at most 10 classes (ECOC)
        chosen = [
            s["name"]
            for s in stats
            if s["type"] == "categorical" and 2 <= s["cardinality"] <= 10
        ][:max_targets]
    target_findings = []
    for target in chosen:
        features = [c for c in columns if c != target]
        if not features:
            continue
        target_findings.append(
            _probe_target(sampled, target, features, stats_by_col, factory, seed)
        )
    return {
        "table": name,
        "num_rows": len(rows),
        "sampled_rows": len(sampled),
        "columns": stats,
        "targets": target_findings,
        "backend": "kumo",
    }


def probe_tables(
    tables: Dict[str, List[Dict[str, Any]]],
    targets_by_table: Optional[Dict[str, str]] = None,
    probe_factory: Optional[ProbeFactory] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Probe every table; ``targets_by_table`` maps table -> target column."""
    targets_by_table = targets_by_table or {}
    return {
        "generated_by": "kumo_probe",
        "tables": [
            probe_table(
                name,
                rows,
                targets=[targets_by_table[name]] if name in targets_by_table else None,
                probe_factory=probe_factory,
                **kwargs,
            )
            for name, rows in sorted(tables.items())
        ],
    }


def tstr_parity(
    real_rows: Sequence[Dict[str, Any]],
    synth_rows: Sequence[Dict[str, Any]],
    target: str,
    probe_factory: Optional[ProbeFactory] = None,
    seed: int = 42,
    size: str = "small",
    device: str = "cpu",
    num_estimators: int = 8,
) -> Dict[str, Any]:
    """Train-on-synthetic-test-on-real parity with the Kumo backend."""
    factory = probe_factory or KumoProbeFactory(
        size=size, device=device, num_estimators=num_estimators
    )
    all_rows = list(real_rows) + list(synth_rows)
    stats = column_stats(all_rows, target)
    task = "regression" if stats["type"] == "numeric" else "classification"
    columns: List[str] = []
    for r in all_rows:
        for key in r:
            if key not in columns:
                columns.append(key)
    feature_cols = [c for c in columns if c != target]
    stats_by_col = {c: column_stats(all_rows, c) for c in columns}
    feature_types = {
        c: ("categorical" if stats_by_col[c]["type"] == "categorical" else "numeric")
        for c in feature_cols
    }
    real = [r for r in real_rows if r.get(target) is not None]
    synth = [r for r in synth_rows if r.get(target) is not None]
    if not real or not synth:
        raise ProbeError("tstr_parity needs real and synthetic rows with the target set")
    train_rows, test_rows = _split_rows(real, seed)
    y_test = [r.get(target) for r in test_rows]
    est_real = factory(task)
    est_real.fit(train_rows, target, feature_cols, feature_types)
    real_score = _score(task, y_test, list(est_real.predict(test_rows)))
    est_synth = factory(task)
    est_synth.fit(synth, target, feature_cols, feature_types)
    synth_score = _score(task, y_test, list(est_synth.predict(test_rows)))
    return {
        "target": target,
        "task": task,
        "real_score": real_score,
        "synth_trained_score": synth_score,
        "parity": round(synth_score / real_score, 4) if real_score > 0 else None,
        "backend": "kumo",
    }
