"""TabPFN probe stage: measure what enterprise tables say, before the KG.

Implements the ADR in ``docs/adr-tabpfn-tabular-input.md``. TabPFN is an
*analyzer*, not a reader: it consumes a table sample and returns measured
findings — which columns predict which outcomes, conditional structure,
class balance. It writes no prose and builds no graph. The findings are
then narrated (deterministically here; an LLM may add a prose brief) into
documents the existing corpus KG extraction consumes unchanged, and a
structural graph is derived from the findings directly.

Pipeline position: **TabPFN → KG → ToolGrad** (``domain.kg_source:
tables``). Privacy note (from the ADR): narration input is findings and
aggregates only — raw rows never leave this module.

TabPFN itself is an optional dependency, imported lazily inside
:class:`TabPFNProbeFactory`. Tests inject a stub factory, so the whole
stage is exercisable keylessly and model-lessly. A missing package is an
explicit :class:`ProbeError`, never a silent pass.

Licensing: the ``tabpfn`` package is under the Prior Labs License
(Apache 2.0 + attribution). Current package defaults may select
checkpoints whose weights carry non-commercial terms; pin a checkpoint
via ``model_path`` or hold a Prior Labs enterprise license before
commercial use.
"""

from __future__ import annotations

import csv
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class ProbeError(RuntimeError):
    """The probe stage cannot run (bad tables, missing package, bad target)."""


# ---------------------------------------------------------------------------
# loading: CSV / JSON tables -> rows of coerced scalars
# ---------------------------------------------------------------------------


def _coerce(value: Any) -> Any:
    """Coerce a raw cell to int / float / str / None (CSV arrives as str)."""
    if value is None or isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if not text:
        return None
    for conv in (int, float):
        try:
            return conv(text)
        except ValueError:
            continue
    return text


def load_tables_dir(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load every ``.csv`` / ``.json`` table in a directory.

    JSON tables are a list of objects. Returns ``{table_name: rows}`` with
    cell values coerced to numbers where they parse as numbers.
    """
    tables: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.isdir(path):
        raise ProbeError(f"tables_dir not found: {path}")
    for fname in sorted(os.listdir(path)):
        fpath = os.path.join(path, fname)
        name, ext = os.path.splitext(fname)
        if ext.lower() == ".csv":
            with open(fpath, newline="", encoding="utf-8") as fh:
                rows = [
                    {k: _coerce(v) for k, v in row.items() if k}
                    for row in csv.DictReader(fh)
                ]
        elif ext.lower() == ".json":
            with open(fpath, encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list) or not all(
                isinstance(r, dict) for r in data
            ):
                raise ProbeError(f"{fname}: JSON table must be a list of objects")
            rows = [{k: _coerce(v) for k, v in r.items()} for r in data]
        else:
            continue
        tables[name] = rows
    return tables


# ---------------------------------------------------------------------------
# column statistics (pure python — no model involved)
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def column_stats(rows: Sequence[Dict[str, Any]], column: str) -> Dict[str, Any]:
    """Type, cardinality, missingness, and headline values for one column."""
    present = [r.get(column) for r in rows if r.get(column) is not None]
    counts: Dict[Any, int] = {}
    for v in present:
        counts[v] = counts.get(v, 0) + 1
    numeric = bool(present) and all(_is_number(v) for v in present)
    stats: Dict[str, Any] = {
        "name": column,
        "type": "numeric" if numeric else "categorical",
        "cardinality": len(counts),
        "missing_frac": round(1.0 - len(present) / len(rows), 4) if rows else 0.0,
    }
    if numeric and present:
        stats["min"] = min(present)
        stats["max"] = max(present)
        stats["mean"] = round(sum(present) / len(present), 4)
    elif present:
        top = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:5]
        stats["top_values"] = [{"value": str(v), "count": c} for v, c in top]
    return stats


# ---------------------------------------------------------------------------
# probe factories: TabPFN for real, stubs for tests
# ---------------------------------------------------------------------------

# A factory maps a task ("classification" | "regression") to an estimator
# with sklearn-style fit(X, y) / predict(X). X / y are plain lists.
ProbeFactory = Callable[[str], Any]


class TabPFNProbeFactory:
    """Real probe: TabPFN estimators, imported lazily on first use."""

    def __init__(self, model_path: str = "", device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device

    def __call__(self, task: str) -> Any:
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
        except ImportError as exc:
            raise ProbeError(
                "kg_source='tables' needs the 'tabpfn' package "
                "(pip install tabpfn; Prior Labs License — Apache 2.0 + "
                "attribution; check checkpoint terms before commercial use)"
            ) from exc
        cls = TabPFNClassifier if task == "classification" else TabPFNRegressor
        kwargs: Dict[str, Any] = {"device": self.device}
        if self.model_path:
            kwargs["model_path"] = self.model_path
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# encoding + scoring helpers (estimator-agnostic, plain lists)
# ---------------------------------------------------------------------------


def _encode_features(
    rows: Sequence[Dict[str, Any]],
    feature_cols: Sequence[str],
    stats_by_col: Dict[str, Dict[str, Any]],
) -> List[List[float]]:
    """Numeric passthrough (median-imputed); categoricals -> integer codes."""
    encoders: Dict[str, Tuple[str, Any]] = {}
    for col in feature_cols:
        if stats_by_col[col]["type"] == "numeric":
            encoders[col] = ("numeric", float(stats_by_col[col].get("mean", 0.0)))
        else:
            cats = sorted({str(r.get(col)) for r in rows if r.get(col) is not None})
            encoders[col] = ("categorical", {c: float(i) for i, c in enumerate(cats)})
    matrix: List[List[float]] = []
    for r in rows:
        encoded = []
        for col in feature_cols:
            kind, data = encoders[col]
            value = r.get(col)
            if kind == "numeric":
                encoded.append(float(value) if _is_number(value) else data)
            else:
                encoded.append(data.get(str(value), -1.0))
        matrix.append(encoded)
    return matrix


def _encode_target(
    rows: Sequence[Dict[str, Any]], target: str, task: str
) -> Tuple[List[Any], Dict[Any, int]]:
    """Classification targets -> class indices; regression -> floats."""
    if task == "regression":
        return [float(r.get(target)) for r in rows], {}
    classes = sorted({r.get(target) for r in rows}, key=str)
    mapping = {c: i for i, c in enumerate(classes)}
    return [mapping[r.get(target)] for r in rows], mapping


def _score(task: str, y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if task == "classification":
        hits = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        return round(hits / len(y_true), 4) if y_true else 0.0
    mean = sum(y_true) / len(y_true) if y_true else 0.0
    ss_tot = sum((v - mean) ** 2 for v in y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    return round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0


def _split(n: int, seed: int) -> Tuple[List[int], List[int]]:
    """Deterministic 80/20 split; tiny tables score on their own train set."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    if n < 10:
        return idx, idx
    cut = max(1, int(0.8 * n))
    return idx[:cut], idx[cut:]


def _pick(rows: Sequence[Dict[str, Any]], idx: Sequence[int]) -> List[Dict[str, Any]]:
    return [rows[i] for i in idx]


# ---------------------------------------------------------------------------
# the probe itself
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
    X = _encode_features(usable, feature_cols, stats_by_col)
    y, _ = _encode_target(usable, target, task)
    train_idx, test_idx = _split(len(usable), seed)
    est = probe_factory(task)
    est.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    base_score = _score(task, y_test, list(est.predict(X_test)))
    if task == "classification":
        counts: Dict[Any, int] = {}
        for v in y:
            counts[v] = counts.get(v, 0) + 1
        baseline = round(max(counts.values()) / len(y), 4) if y else 0.0
    else:
        baseline = 0.0  # R2 of the mean predictor
    importances = []
    rng = random.Random(seed + 1)
    for j, col in enumerate(feature_cols):
        shuffled = [row[:] for row in X_test]
        column = [row[j] for row in shuffled]
        rng.shuffle(column)
        for i, row in enumerate(shuffled):
            row[j] = column[i]
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
        "holdout": test_idx != train_idx,
        "importances": importances,
    }


def probe_table(
    name: str,
    rows: Sequence[Dict[str, Any]],
    targets: Optional[Sequence[str]] = None,
    probe_factory: Optional[ProbeFactory] = None,
    max_rows: int = 1000,
    max_targets: int = 3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Probe one table: column stats + measured predictive relationships.

    ``targets`` names the outcome columns to model; when omitted, suitable
    categorical columns (2–20 distinct values, not near-unique) are chosen
    automatically, capped at ``max_targets``. Numeric targets are modeled
    only when named explicitly — auto-regressing every numeric column is
    noise, not signal.
    """
    if not rows:
        raise ProbeError(f"table {name!r} has no rows")
    factory = probe_factory or TabPFNProbeFactory()
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
        chosen = [
            s["name"]
            for s in stats
            if s["type"] == "categorical" and 2 <= s["cardinality"] <= 20
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
        "generated_by": "tabpfn_probe",
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
) -> Dict[str, Any]:
    """Train-on-synthetic-test-on-real parity for one target column.

    The ADR's tabular quality gate: synthetic rows are only as good as the
    predictive relationships they preserve. Parity near 1.0 means a model
    trained on synthetic data scores on real data like one trained on real
    data. Returns both scores and their ratio (None when the real-trained
    baseline is not positive, e.g. an R2 at or below zero).
    """
    factory = probe_factory or TabPFNProbeFactory()
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
    real = [r for r in real_rows if r.get(target) is not None]
    synth = [r for r in synth_rows if r.get(target) is not None]
    if not real or not synth:
        raise ProbeError("tstr_parity needs real and synthetic rows with the target set")
    train_idx, test_idx = _split(len(real), seed)
    X_real = _encode_features(real, feature_cols, stats_by_col)
    y_real, _ = _encode_target(real, target, task)
    X_synth = _encode_features(synth, feature_cols, stats_by_col)
    y_synth, _ = _encode_target(synth, target, task)
    X_test = [X_real[i] for i in test_idx]
    y_test = [y_real[i] for i in test_idx]
    est_real = factory(task)
    est_real.fit([X_real[i] for i in train_idx], [y_real[i] for i in train_idx])
    real_score = _score(task, y_test, list(est_real.predict(X_test)))
    est_synth = factory(task)
    est_synth.fit(X_synth, y_synth)
    synth_score = _score(task, y_test, list(est_synth.predict(X_test)))
    return {
        "target": target,
        "task": task,
        "real_score": real_score,
        "synth_trained_score": synth_score,
        "parity": round(synth_score / real_score, 4) if real_score > 0 else None,
    }


# ---------------------------------------------------------------------------
# narration: findings -> corpus documents (findings only, never raw rows)
# ---------------------------------------------------------------------------


def findings_to_markdown(table_findings: Dict[str, Any]) -> str:
    """Render one table's findings as a deterministic markdown document."""
    lines = [
        f"# Table: {table_findings['table']}",
        "",
        f"{table_findings['num_rows']} rows"
        + (
            f" (probed on a sample of {table_findings['sampled_rows']})"
            if table_findings["sampled_rows"] != table_findings["num_rows"]
            else ""
        )
        + ".",
        "",
        "## Columns",
        "",
    ]
    for col in table_findings["columns"]:
        detail = f"{col['type']}, {col['cardinality']} distinct"
        if col["missing_frac"]:
            detail += f", {col['missing_frac']:.1%} missing"
        if col["type"] == "numeric" and "min" in col:
            detail += f", range {col['min']} to {col['max']}, mean {col['mean']}"
        elif col.get("top_values"):
            tops = ", ".join(
                f"{t['value']} ({t['count']})" for t in col["top_values"]
            )
            detail += f"; most common: {tops}"
        lines.append(f"- {col['name']}: {detail}")
    if table_findings["targets"]:
        lines += ["", "## Measured relationships", ""]
        for tgt in table_findings["targets"]:
            metric = "accuracy" if tgt["task"] == "classification" else "R2"
            lines.append(
                f"- {tgt['column']} ({tgt['task']}): holdout {metric} "
                f"{tgt['holdout_score']} vs baseline {tgt['baseline_score']}."
            )
            drivers = [i for i in tgt["importances"] if i["importance"] > 0]
            if drivers:
                named = ", ".join(
                    f"{d['feature']} ({d['importance']})" for d in drivers[:5]
                )
                lines.append(f"  Strongest drivers: {named}.")
    return "\n".join(lines) + "\n"


_NARRATE_PROMPT = """\
Rewrite the measured findings below about an enterprise data table as a
short domain brief for a knowledge graph. Use ONLY facts present in the
findings; do not invent values, entities, or relationships. Name the
business entities and the rules the findings support.

--- findings ---
{doc}
"""


def write_tables_corpus(
    findings: Dict[str, Any],
    out_dir: str,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """Write findings documents (+ optional LLM narrative briefs) to a dir.

    The deterministic findings doc is always written — it is the auditable
    source. When ``llm_fn`` is given, a prose brief narrated from the
    findings (never from raw rows) is written alongside it. Returns the
    list of written paths. Also writes ``findings.json`` for audit.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for table_findings in findings["tables"]:
        doc = findings_to_markdown(table_findings)
        path = os.path.join(out_dir, f"{table_findings['table']}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(doc)
        paths.append(path)
        if llm_fn is not None:
            brief = llm_fn(_NARRATE_PROMPT.format(doc=doc))
            npath = os.path.join(out_dir, f"{table_findings['table']}_narrative.md")
            with open(npath, "w", encoding="utf-8") as fh:
                fh.write(brief if brief.endswith("\n") else brief + "\n")
            paths.append(npath)
    with open(os.path.join(out_dir, "findings.json"), "w", encoding="utf-8") as fh:
        json.dump(findings, fh, indent=2)
    return paths
