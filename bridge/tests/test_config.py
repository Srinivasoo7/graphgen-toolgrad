"""Tests for bridge/config.py (keyless)."""

import json
import os
import tempfile

from bridge import config as config_module
from bridge.config import (
    ConfigError,
    RunConfig,
    from_dict,
    load_config,
    mcp_dict_for_config,
    validate_config,
    write_example_config,
)


def _minimal_dict(**overrides):
    d = {
        "name": "test",
        "mcp_servers": [
            {"name": "fs", "transport": "stdio", "command": "npx",
             "args": ["-y", "server"], "timeout_s": 5.0}
        ],
        "domain": {"name": "test", "kg_source": "tools"},
        "generation": {"num_chains": 2, "apis_per_workflow": 3, "seed": 7},
        "llm": {"backend": "openrouter", "model": "m", "api_key_env": "X"},
        "quality": {"min_coverage": 0.5},
        "output": {"dir": "/tmp/x", "train_split": 0.8},
    }
    d.update(overrides)
    return d


def test_minimal_config_loads():
    cfg = from_dict(_minimal_dict())
    assert cfg.name == "test"
    assert cfg.mcp_servers[0].transport == "stdio"
    assert cfg.generation.num_chains == 2
    assert cfg.output.train_split == 0.8


def test_missing_servers_rejected():
    try:
        from_dict(_minimal_dict(mcp_servers=[]))
    except ConfigError as exc:
        assert "at least one server" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_sse_needs_url():
    d = _minimal_dict(mcp_servers=[{"name": "r", "transport": "sse"}])
    try:
        from_dict(d)
    except ConfigError as exc:
        assert "needs 'url'" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_corpus_needs_dir():
    d = _minimal_dict(domain={"name": "t", "kg_source": "corpus"})
    try:
        from_dict(d)
    except ConfigError as exc:
        assert "corpus_dir" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_jev_needs_cli():
    d = _minimal_dict(quality={"semantic_filter": "jev"})
    try:
        from_dict(d)
    except ConfigError as exc:
        assert "jev_cli" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_bad_backend_rejected():
    d = _minimal_dict(llm={"backend": "telepathy"})
    try:
        from_dict(d)
    except ConfigError as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_train_split_bounds():
    d = _minimal_dict(output={"train_split": 1.0})
    try:
        from_dict(d)
    except ConfigError as exc:
        assert "train_split" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_non_mapping_root_rejected():
    try:
        from_dict([1, 2])
    except ConfigError:
        pass
    else:
        raise AssertionError("expected ConfigError")


def test_mcp_dict_translation():
    cfg = from_dict(_minimal_dict(mcp_servers=[
        {"name": "a", "transport": "stdio", "command": "npx", "args": ["x"],
         "env": {"K": "V"}},
        {"name": "b", "transport": "sse", "url": "http://h/sse"},
    ]))
    md = mcp_dict_for_config(cfg)
    assert md["a"]["transport"] == "stdio"
    assert md["a"]["env"] == {"K": "V"}
    assert md["b"] == {"transport": "sse", "url": "http://h/sse"}


def test_fingerprint_stable():
    a = config_module.config_fingerprint(from_dict(_minimal_dict()))
    b = config_module.config_fingerprint(from_dict(_minimal_dict()))
    assert a == b and len(a) == 16


def test_example_config_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ggt.yaml")
        write_example_config(path, name="acme")
        cfg = load_config(path)
        assert cfg.name == "acme"
        assert cfg.mcp_servers[0].command == "npx"


def test_load_json_config():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ggt.json")
        with open(path, "w") as fh:
            json.dump(_minimal_dict(), fh)
        cfg = load_config(path)
        assert cfg.name == "test"


def test_empty_file_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "empty.yaml")
        open(path, "w").close()
        try:
            load_config(path)
        except ConfigError as exc:
            assert "empty" in str(exc)
        else:
            raise AssertionError("expected ConfigError")


def test_validate_returns_list_directly():
    cfg = RunConfig(name="x")  # no servers
    problems = validate_config(cfg)
    assert any("at least one server" in p for p in problems)


def test_mcp_timeout_default_is_120():
    cfg = from_dict(_minimal_dict(mcp_servers=[
        {"name": "fs", "transport": "stdio", "command": "npx", "args": ["-y", "s"]}
    ]))
    assert cfg.mcp_servers[0].timeout_s == 120.0


def test_generation_yield_and_call_budget_parsed():
    cfg = from_dict(_minimal_dict(generation={
        "num_chains": 4, "apis_per_workflow": 2, "seed": 1,
        "min_samples": 3, "max_llm_calls": 100,
    }))
    assert cfg.generation.min_samples == 3
    assert cfg.generation.max_llm_calls == 100


def test_generation_yield_defaults():
    cfg = from_dict(_minimal_dict())
    assert cfg.generation.min_samples == 1
    assert cfg.generation.max_llm_calls == 0


def test_min_samples_cannot_exceed_num_chains():
    d = _minimal_dict(generation={"num_chains": 2, "min_samples": 5})
    try:
        from_dict(d)
    except ConfigError as exc:
        assert "min_samples" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_negative_max_llm_calls_rejected():
    d = _minimal_dict(generation={"num_chains": 2, "max_llm_calls": -1})
    try:
        from_dict(d)
    except ConfigError as exc:
        assert "max_llm_calls" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def test_scaffold_carries_new_knobs():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ggt.yaml")
        write_example_config(path, name="demo")
        raw = open(path).read()
    assert "timeout_s: 120.0" in raw
    assert "min_samples" in raw and "max_llm_calls" in raw
