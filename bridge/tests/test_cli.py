"""Tests for bridge/cli.py (keyless; run/report commands need live services)."""

import contextlib
import io
import json
import os
import tempfile
from types import SimpleNamespace

from bridge import cli


def _args(**kw):
    return SimpleNamespace(**kw)


def _capture(fn, *a, **kw):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = fn(*a, **kw)
    return rc, out.getvalue(), err.getvalue()


def test_init_writes_scaffold():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "ggt.yaml")
        rc = cli.cmd_init(_args(output=out, name="demo"))
        assert rc == 0
        raw = open(out).read()
    assert "name: demo" in raw
    assert "timeout_s: 120.0" in raw
    assert "min_samples" in raw


def test_validate_config_ok():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "ggt.yaml")
        cli.cmd_init(_args(output=out, name="demo"))
        rc, stdout, _ = _capture(cli.cmd_validate_config, _args(config=out))
    assert rc == 0
    assert "config OK" in stdout


def test_validate_config_rejects_bad():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "bad.yaml")
        with open(out, "w") as fh:
            fh.write("name: bad\nmcp_servers: []\n")
        rc, _, stderr = _capture(cli.cmd_validate_config, _args(config=out))
    assert rc == 1
    assert "invalid config" in stderr


def test_report_missing_dir():
    with tempfile.TemporaryDirectory() as tmp:
        rc, _, stderr = _capture(
            cli.cmd_report, _args(run_dir=os.path.join(tmp, "nope")))
    assert rc == 1
    assert "no run report" in stderr


def test_report_prints_summary():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "run")
        os.makedirs(run_dir)
        report = {
            "run_name": "demo", "status": "failed", "failed_stage": "generate",
            "duration_s": 3.2,
            "stages": {
                "generate": {"status": "failed", "duration_s": 1.0,
                             "error": "below min_samples=1"},
            },
            "llm": {"calls": 0}, "eval": {}, "mix": {},
        }
        with open(os.path.join(run_dir, "run_report.json"), "w") as fh:
            json.dump(report, fh)
        rc, stdout, _ = _capture(cli.cmd_report, _args(run_dir=run_dir))
    assert rc == 0
    assert "failed" in stdout and "generate" in stdout


def test_main_parses_init():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "x.yaml")
        rc, _, _ = _capture(cli.main, ["init", "--output", out, "--name", "n"])
        assert rc == 0
        assert os.path.exists(out)
