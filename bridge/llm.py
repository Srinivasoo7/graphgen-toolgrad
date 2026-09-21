"""Resilient LLM client for the factory.

All factory LLM traffic goes through :class:`LLMClient`, which adds the
things raw provider calls lack:

- retries with exponential backoff + jitter on transient failures
  (timeouts, connection errors, HTTP 429/5xx),
- client-side rate limiting (``max_rpm``),
- per-call timeouts,
- call/ retry accounting and cost estimation,
- an optional spend cap (``budget_usd``) that aborts the run before the
  next call instead of after the invoice.

The client is backend-agnostic: it wraps any ``complete_fn(prompt) -> str``.
:func:`build_llm_client` wires the configured backend — currently
OpenRouter (:class:`OpenRouterBackend`, stdlib ``urllib`` only, no extra
deps). API keys come from the environment (never from the config file,
never logged).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional


class LLMError(Exception):
    """Base class for LLM failures."""


class TransientLLMError(LLMError):
    """Retryable: timeouts, connection resets, HTTP 429/5xx."""


class RateLimitError(TransientLLMError):
    """HTTP 429 specifically."""


class BudgetExceeded(LLMError):
    """The run's spend cap was hit; not retryable."""


# Conservative token estimate (~4 chars/token for English). Estimates are
# labeled as such everywhere they surface; pass real per-1M prices to get
# usable cost numbers.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // _CHARS_PER_TOKEN)


class LLMClient:
    """Wraps ``complete_fn(prompt) -> str`` with resilience + accounting."""

    def __init__(
        self,
        complete_fn: Callable[[str], str],
        *,
        max_retries: int = 4,
        backoff_s: float = 1.0,
        backoff_cap_s: float = 60.0,
        timeout_s: float = 120.0,
        min_interval_s: float = 0.0,
        budget_usd: float = 0.0,
        price_per_1m_input: float = 0.0,
        price_per_1m_output: float = 0.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._fn = complete_fn
        self.max_retries = max(0, max_retries)
        self.backoff_s = backoff_s
        self.backoff_cap_s = backoff_cap_s
        self.timeout_s = timeout_s
        self.min_interval_s = min_interval_s
        self.budget_usd = budget_usd
        self.price_per_1m_input = price_per_1m_input
        self.price_per_1m_output = price_per_1m_output
        self._sleep = sleep_fn
        self._clock = clock
        self._rng = rng or random.Random()
        self._last_call_at: Optional[float] = None
        self.calls = 0
        self.retries = 0
        self.transient_failures = 0
        self.fatal_failures = 0
        self.est_input_tokens = 0
        self.est_output_tokens = 0

    # -- accounting ----------------------------------------------------
    @property
    def est_cost_usd(self) -> float:
        return (
            self.est_input_tokens / 1e6 * self.price_per_1m_input
            + self.est_output_tokens / 1e6 * self.price_per_1m_output
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "retries": self.retries,
            "transient_failures": self.transient_failures,
            "fatal_failures": self.fatal_failures,
            "est_input_tokens": self.est_input_tokens,
            "est_output_tokens": self.est_output_tokens,
            "est_cost_usd": round(self.est_cost_usd, 6),
            "cost_is_estimate": True,
        }

    def _check_budget(self) -> None:
        if self.budget_usd > 0 and self.est_cost_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"spend cap reached: est. ${self.est_cost_usd:.4f} "
                f">= budget ${self.budget_usd:.4f}"
            )

    def _pace(self) -> None:
        if self.min_interval_s <= 0 or self._last_call_at is None:
            return
        wait = self.min_interval_s - (self._clock() - self._last_call_at)
        if wait > 0:
            self._sleep(wait)

    def _backoff(self, attempt: int) -> None:
        delay = min(self.backoff_s * (2**attempt), self.backoff_cap_s)
        self._sleep(delay * (0.5 + self._rng.random()))

    # -- main entry ----------------------------------------------------
    def complete(self, prompt: str) -> str:
        """Complete ``prompt``, retrying transient failures."""
        self._check_budget()
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            self._pace()
            try:
                self._last_call_at = self._clock()
                text = self._fn(prompt)
            except (TimeoutError, ConnectionError, TransientLLMError) as exc:
                last_exc = exc
                self.transient_failures += 1
                if attempt < self.max_retries:
                    self.retries += 1
                    self._backoff(attempt)
                    continue
                self.fatal_failures += 1
                raise
            except BudgetExceeded:
                self.fatal_failures += 1
                raise
            except Exception as exc:  # noqa: BLE001 — non-retryable backend error
                last_exc = exc
                self.fatal_failures += 1
                raise LLMError(f"LLM backend failed: {exc}") from exc
            self.calls += 1
            self.est_input_tokens += estimate_tokens(prompt)
            self.est_output_tokens += estimate_tokens(text)
            self._check_budget()
            return text
        # Unreachable: the loop either returns or raises.
        raise TransientLLMError(f"exhausted retries: {last_exc}")


class OpenRouterBackend:
    """OpenRouter chat-completions backend over stdlib ``urllib``.

    ``api_key`` should come from the environment (see :meth:`from_env`);
    it is never logged or written anywhere by this class.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: float = 120.0,
    ) -> None:
        if not api_key:
            raise LLMError("OpenRouter API key is empty")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    @classmethod
    def from_env(
        cls,
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: float = 120.0,
    ) -> "OpenRouterBackend":
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise LLMError(
                f"environment variable {api_key_env} is not set; "
                "export it with your OpenRouter API key"
            )
        return cls(api_key, model, base_url=base_url, timeout_s=timeout_s)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Srinivasoo7/graphgen-toolgrad",
            "X-Title": "graphgen-toolgrad factory",
        }

    def complete(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise RateLimitError(f"OpenRouter 429: {exc}") from exc
            if 500 <= exc.code < 600:
                raise TransientLLMError(f"OpenRouter {exc.code}: {exc}") from exc
            raise LLMError(f"OpenRouter HTTP {exc.code}: {exc}") from exc
        except urllib.error.URLError as exc:
            raise TransientLLMError(f"OpenRouter connection failed: {exc}") from exc
        except TimeoutError as exc:
            raise TransientLLMError(f"OpenRouter timed out: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected OpenRouter response shape: {exc}") from exc


def _load_authd_helpers():
    """Import the authd surrogate helpers (lazy: only on hosts that have them)."""
    sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
    try:
        import dynamic_credentials as dc
    except ImportError as exc:
        raise LLMError(
            "authd credential helpers are not available on this host; "
            "set the OpenRouter API key env var instead"
        ) from exc
    return dc


class AuthdOpenRouterBackend:
    """OpenRouter via the platform's authd credential surrogate.

    Uses the connected ``custom.openrouter`` credential: requests carry an
    ``hsurr:*`` surrogate that authd exchanges for the real key at egress
    time. The raw key never appears in this process — not in memory, not
    in logs, not on disk. Only ever talks to ``openrouter.ai``.
    """

    CREDENTIAL_NAME = "custom.openrouter"
    ALLOWED_HOSTS = ("openrouter.ai",)

    def __init__(
        self,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: float = 120.0,
        helpers=None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._helpers = helpers  # injectable for keyless tests

    def complete(self, prompt: str) -> str:
        dc = self._helpers or _load_authd_helpers()
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Srinivasoo7/graphgen-toolgrad",
                "X-Title": "graphgen-toolgrad factory",
            },
            method="POST",
        )
        try:
            dc.add_surrogate_to_request(
                request,
                self.CREDENTIAL_NAME,
                allowed_hosts=self.ALLOWED_HOSTS,
            )
        except Exception as exc:
            raise LLMError(
                f"could not attach the {self.CREDENTIAL_NAME} credential: {exc}"
            ) from exc
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                body = dc.read_json_response(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise RateLimitError(f"OpenRouter 429: {exc}") from exc
            if 500 <= exc.code < 600:
                raise TransientLLMError(f"OpenRouter {exc.code}: {exc}") from exc
            raise LLMError(f"OpenRouter HTTP {exc.code}: {exc}") from exc
        except urllib.error.URLError as exc:
            raise TransientLLMError(f"OpenRouter connection failed: {exc}") from exc
        except TimeoutError as exc:
            raise TransientLLMError(f"OpenRouter timed out: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected OpenRouter response shape: {exc}") from exc


def credential_source_hint(llm_cfg) -> str:
    """Human-readable description of where the OpenRouter credential comes from."""
    if os.environ.get(llm_cfg.api_key_env):
        return f"environment variable {llm_cfg.api_key_env}"
    return "connected custom.openrouter credential (authd surrogate, no plaintext key)"


def build_llm_client(llm_cfg, *, sleep_fn=time.sleep) -> LLMClient:
    """Build an :class:`LLMClient` from an :class:`LLMConfig`.

    OpenRouter credential resolution: the ``api_key_env`` variable wins
    when set (ordinary enterprise deployment); otherwise the platform's
    connected ``custom.openrouter`` credential is used via the authd
    surrogate (no plaintext key anywhere).
    """
    if llm_cfg.backend != "openrouter":
        raise LLMError(f"unsupported LLM backend: {llm_cfg.backend!r}")
    if os.environ.get(llm_cfg.api_key_env):
        backend = OpenRouterBackend.from_env(
            model=llm_cfg.model,
            api_key_env=llm_cfg.api_key_env,
            base_url=llm_cfg.base_url,
            timeout_s=llm_cfg.timeout_s,
        )
    else:
        backend = AuthdOpenRouterBackend(
            model=llm_cfg.model,
            base_url=llm_cfg.base_url,
            timeout_s=llm_cfg.timeout_s,
        )
    min_interval = 60.0 / llm_cfg.max_rpm if llm_cfg.max_rpm > 0 else 0.0
    return LLMClient(
        backend.complete,
        max_retries=llm_cfg.max_retries,
        backoff_s=llm_cfg.backoff_s,
        timeout_s=llm_cfg.timeout_s,
        min_interval_s=min_interval,
        budget_usd=llm_cfg.budget_usd,
        price_per_1m_input=llm_cfg.price_per_1m_input,
        price_per_1m_output=llm_cfg.price_per_1m_output,
        sleep_fn=sleep_fn,
    )
