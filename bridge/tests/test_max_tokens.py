"""Regression tests for the ``llm.max_tokens`` cap (OpenRouter 402 fix).

OpenRouter pre-authorizes the full max output tokens against the account
balance, so an uncapped generation request 402s low-balance keys. These
tests pin the cap at every handoff it must survive:

config parsing -> the ChatOpenAI factory -> both urllib backends ->
the runner's generation ``llm_cfg``.

Keyless throughout: no network, no credentials, no fork import.
"""

import json
import os
import sys
import tempfile
import types
import urllib.request

from bridge.config import from_dict
from bridge.llm import (
    AuthdOpenRouterBackend,
    OpenRouterBackend,
    build_llm_client,
)
from bridge.runner import FactoryRun
from bridge.toolgrad_gen import _llm_factory


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


# -- config ---------------------------------------------------------------

def test_config_parses_max_tokens():
    cfg = from_dict(_minimal_dict(
        llm={"backend": "openrouter", "model": "m", "api_key_env": "X",
             "max_tokens": 4096}))
    assert cfg.llm.max_tokens == 4096


def test_config_max_tokens_defaults_to_zero():
    cfg = from_dict(_minimal_dict())
    assert cfg.llm.max_tokens == 0  # 0 = provider default (uncapped)


def test_config_as_dict_round_trips_max_tokens():
    cfg = from_dict(_minimal_dict(
        llm={"backend": "openrouter", "model": "m", "api_key_env": "X",
             "max_tokens": 4096}))
    assert cfg.as_dict()["llm"]["max_tokens"] == 4096
    assert from_dict(cfg.as_dict()).llm.max_tokens == 4096


# -- ChatOpenAI factory ----------------------------------------------------

def _stub_chat_openai(monkeypatch):
    """Stub langchain_openai (lazily imported by the factory); return kwargs."""
    captured = {}
    mod = types.ModuleType("langchain_openai")

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    mod.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", mod)
    return captured


def test_llm_factory_passes_max_tokens_to_chat_openai(monkeypatch):
    captured = _stub_chat_openai(monkeypatch)
    monkeypatch.setenv("MT_FACTORY_KEY", "k")
    factory = _llm_factory(
        {"model": "m", "api_key_env": "MT_FACTORY_KEY",
         "base_url": "https://openrouter.ai/api/v1", "max_tokens": 4096},
        {"n": 0})
    factory()
    assert captured["max_tokens"] == 4096


def test_llm_factory_omits_max_tokens_when_unset(monkeypatch):
    captured = _stub_chat_openai(monkeypatch)
    monkeypatch.setenv("MT_FACTORY_KEY", "k")
    factory = _llm_factory(
        {"model": "m", "api_key_env": "MT_FACTORY_KEY"}, {"n": 0})
    factory()
    assert "max_tokens" not in captured


# -- urllib backends --------------------------------------------------------

class _FakeURLopen:
    """Capture the request body; reply with a minimal chat-completion."""

    def __init__(self):
        self.bodies = []

    def __call__(self, request, timeout=None):
        self.bodies.append(json.loads(request.data.decode("utf-8")))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")


def _run_complete(backend):
    fake_open = _FakeURLopen()
    real = urllib.request.urlopen
    urllib.request.urlopen = fake_open
    try:
        assert backend.complete("hello") == "ok"
    finally:
        urllib.request.urlopen = real
    assert len(fake_open.bodies) == 1
    return fake_open.bodies[0]


def test_openrouter_env_backend_sends_max_tokens():
    backend = OpenRouterBackend("k", "m", max_tokens=4096)
    body = _run_complete(backend)
    assert body["max_tokens"] == 4096


def test_openrouter_env_backend_omits_max_tokens_when_zero():
    backend = OpenRouterBackend("k", "m")
    body = _run_complete(backend)
    assert "max_tokens" not in body


class _FakeAuthdHelpers:
    def add_surrogate_to_request(self, request, credential_name, *,
                                 allowed_hosts):
        request.add_header("Authorization", "Bearer <redacted>:fake")

    def read_json_response(self, resp):
        return {"choices": [{"message": {"content": "ok"}}]}


def test_openrouter_authd_backend_sends_max_tokens():
    backend = AuthdOpenRouterBackend("m", max_tokens=4096,
                                    helpers=_FakeAuthdHelpers())
    body = _run_complete(backend)
    assert body["max_tokens"] == 4096


def test_openrouter_authd_backend_omits_max_tokens_when_zero():
    backend = AuthdOpenRouterBackend("m", helpers=_FakeAuthdHelpers())
    body = _run_complete(backend)
    assert "max_tokens" not in body


# -- build_llm_client wiring -------------------------------------------------

def test_build_llm_client_propagates_max_tokens_env_backend(monkeypatch):
    from bridge.config import LLMConfig
    monkeypatch.setenv("MT_BUILD_KEY", "k")
    client = build_llm_client(
        LLMConfig(api_key_env="MT_BUILD_KEY", max_tokens=4096))
    assert client._fn.__self__.max_tokens == 4096


def test_build_llm_client_propagates_max_tokens_authd_backend():
    from bridge.config import LLMConfig
    monkeypatch_env = "MT_BUILD_KEY_MISSING"
    os.environ.pop(monkeypatch_env, None)
    client = build_llm_client(
        LLMConfig(api_key_env=monkeypatch_env, max_tokens=4096))
    backend = client._fn.__self__
    assert type(backend).__name__ == "AuthdOpenRouterBackend"
    assert backend.max_tokens == 4096


# -- runner: generation llm_cfg ----------------------------------------------

def test_runner_generate_llm_cfg_keeps_max_tokens():
    seen = {}

    def fake_generate(workdir, **kwargs):
        seen.update(kwargs)
        return {"samples": [], "sample_paths": [], "trace_paths": [],
                "tools_by_name": {}, "llm_calls": 0}

    with tempfile.TemporaryDirectory() as tmp:
        cfg = from_dict(_minimal_dict(
            llm={"backend": "openrouter", "model": "m", "api_key_env": "X",
                 "max_tokens": 4096},
            output={"dir": os.path.join(tmp, "run"), "train_split": 0.8}))
        runner = FactoryRun(cfg, workdir=os.path.join(tmp, "run"),
                            generate_fn=fake_generate)
        runner._ctx["kg_text"] = "kg"
        runner._stage_generate()
    assert seen["llm_cfg"]["max_tokens"] == 4096
