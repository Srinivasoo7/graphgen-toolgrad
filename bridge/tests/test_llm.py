"""Tests for bridge/llm.py (keyless — no network, fake clock/sleep)."""

import os

from bridge.llm import (
    AuthdOpenRouterBackend,
    BudgetExceeded,
    LLMClient,
    LLMError,
    OpenRouterBackend,
    RateLimitError,
    TransientLLMError,
    build_llm_client,
    credential_source_hint,
)
from bridge.config import LLMConfig


class _Harness:
    def __init__(self):
        self.sleeps = []
        self.now = 1000.0

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s

    def clock(self):
        return self.now


def _client(fn, harness, **kw):
    return LLMClient(
        fn,
        sleep_fn=harness.sleep,
        clock=harness.clock,
        backoff_s=1.0,
        **kw,
    )


def test_success_first_try():
    h = _Harness()
    c = _client(lambda p: "hello " + p, h)
    assert c.complete("x") == "hello x"
    assert c.stats()["calls"] == 1
    assert c.stats()["retries"] == 0
    assert h.sleeps == []


def test_retry_then_success():
    import random

    random.seed(20260920)  # deterministic jitter; suite order must not matter
    h = _Harness()
    attempts = {"n": 0}

    def flaky(prompt):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransientLLMError("boom")
        return "ok"

    c = _client(flaky, h, max_retries=4)
    assert c.complete("p") == "ok"
    st = c.stats()
    assert st["calls"] == 1 and st["retries"] == 2 and st["transient_failures"] == 2
    assert len(h.sleeps) == 2 and h.sleeps[1] > h.sleeps[0]  # exponential


def test_transient_exhaustion_raises():
    h = _Harness()
    c = _client(lambda p: (_ for _ in ()).throw(TimeoutError("t/o")), h, max_retries=2)
    try:
        c.complete("p")
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected TimeoutError")
    assert c.stats()["fatal_failures"] == 1


def test_rate_limit_is_transient():
    h = _Harness()
    calls = {"n": 0}

    def rl(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("429")
        return "ok"

    c = _client(rl, h, max_retries=2)
    assert c.complete("p") == "ok"


def test_non_transient_wrapped_as_llm_error():
    h = _Harness()
    c = _client(lambda p: 1 / 0, h)
    try:
        c.complete("p")
    except LLMError as exc:
        assert "backend failed" in str(exc)
    else:
        raise AssertionError("expected LLMError")
    assert c.stats()["retries"] == 0


def test_rate_limit_pacing():
    h = _Harness()
    c = _client(lambda p: "ok", h, min_interval_s=60.0)
    c.complete("a")
    c.complete("b")
    assert h.sleeps == [60.0]  # second call waited for the interval


def test_budget_blocks_before_call():
    h = _Harness()
    made = {"n": 0}

    def fn(p):
        made["n"] += 1
        return "x" * 100000  # ~25k tokens

    c = _client(fn, h, budget_usd=0.01, price_per_1m_input=1.0,
                price_per_1m_output=1.0)
    try:
        c.complete("prompt")
    except BudgetExceeded as exc:
        assert "spend cap" in str(exc)
    else:
        raise AssertionError("expected BudgetExceeded")
    assert made["n"] == 1  # first call ran, second would be blocked


def test_cost_accounting_estimates():
    h = _Harness()
    c = _client(lambda p: "y" * 400, h, price_per_1m_input=2.0,
                price_per_1m_output=4.0)
    c.complete("x" * 400)
    st = c.stats()
    assert st["est_input_tokens"] == 100
    assert st["est_output_tokens"] == 100
    assert st["cost_is_estimate"] is True
    assert abs(st["est_cost_usd"] - (100 / 1e6 * 2.0 + 100 / 1e6 * 4.0)) < 1e-9


def test_openrouter_from_env_missing_key():
    os.environ.pop("GGLLM_TEST_KEY", None)
    try:
        OpenRouterBackend.from_env("m", api_key_env="GGLLM_TEST_KEY")
    except LLMError as exc:
        assert "GGLLM_TEST_KEY" in str(exc)
    else:
        raise AssertionError("expected LLMError")


def test_build_llm_client_bad_backend():
    cfg = LLMConfig(backend="telepathy")
    try:
        build_llm_client(cfg)
    except LLMError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("expected LLMError")


def test_build_llm_client_wires_knobs():
    os.environ["GGLLM_TEST_KEY2"] = "k"
    try:
        cfg = LLMConfig(api_key_env="GGLLM_TEST_KEY2", max_retries=7,
                        max_rpm=30.0, budget_usd=1.5)
        c = build_llm_client(cfg)
        assert c.max_retries == 7
        assert c.min_interval_s == 2.0
        assert c.budget_usd == 1.5
    finally:
        del os.environ["GGLLM_TEST_KEY2"]


class _FakeAuthdHelpers:
    def __init__(self):
        self.attached = []

    def add_surrogate_to_request(self, request, credential_name, *, allowed_hosts):
        assert credential_name == "custom.openrouter"
        assert "openrouter.ai" in allowed_hosts
        request.add_header("Authorization", "Bearer hsurr:fake")
        self.attached.append(request.full_url)

    def read_json_response(self, resp):
        return {"choices": [{"message": {"content": "authd says hi"}}]}


class _FakeURLopen:
    def __init__(self):
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_build_llm_client_env_wins_over_authd():
    import os
    os.environ["GGLLM_TEST_KEY3"] = "k"
    try:
        cfg = LLMConfig(api_key_env="GGLLM_TEST_KEY3")
        c = build_llm_client(cfg)
        backend = c._fn.__self__
        assert type(backend).__name__ == "OpenRouterBackend"
    finally:
        del os.environ["GGLLM_TEST_KEY3"]


def test_build_llm_client_falls_back_to_authd():
    import os
    os.environ.pop("GGLLM_TEST_KEY_MISSING", None)
    cfg = LLMConfig(api_key_env="GGLLM_TEST_KEY_MISSING")
    c = build_llm_client(cfg)
    backend = c._fn.__self__
    assert type(backend).__name__ == "AuthdOpenRouterBackend"


def test_authd_backend_complete_with_fake_helpers():
    import urllib.request
    helpers = _FakeAuthdHelpers()
    backend = AuthdOpenRouterBackend("some/model", helpers=helpers)
    fake_open = _FakeURLopen()
    real = urllib.request.urlopen
    urllib.request.urlopen = fake_open
    try:
        assert backend.complete("hello") == "authd says hi"
    finally:
        urllib.request.urlopen = real
    assert helpers.attached == ["https://openrouter.ai/api/v1/chat/completions"]
    sent = fake_open.requests[0]
    assert sent.get_header("Authorization") == "Bearer hsurr:fake"
    assert sent.full_url.startswith("https://openrouter.ai/")


def test_authd_backend_credential_failure_is_llm_error():
    class Boom:
        def add_surrogate_to_request(self, *a, **k):
            raise RuntimeError("no authd here")

    backend = AuthdOpenRouterBackend("some/model", helpers=Boom())
    try:
        backend.complete("hi")
    except LLMError as exc:
        assert "custom.openrouter" in str(exc)
    else:
        raise AssertionError("expected LLMError")


def test_credential_source_hint():
    import os
    cfg = LLMConfig(api_key_env="GGLLM_TEST_KEY_MISSING2")
    os.environ.pop("GGLLM_TEST_KEY_MISSING2", None)
    assert "authd surrogate" in credential_source_hint(cfg)
    os.environ["GGLLM_TEST_KEY_MISSING2"] = "k"
    try:
        assert "GGLLM_TEST_KEY_MISSING2" in credential_source_hint(cfg)
    finally:
        del os.environ["GGLLM_TEST_KEY_MISSING2"]
