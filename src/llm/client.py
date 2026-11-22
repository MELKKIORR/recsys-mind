"""One LLM client for the whole project, with a disk cache (MLE-Ops owns it, MLE-Core uses it).

Providers:
  mock       no network, deterministic answers; for tests and for running the pipeline without a key
  anthropic  Claude API, key in the ANTHROPIC_API_KEY environment variable
  openai     any OpenAI-compatible API (OpenAI, Mistral, Groq, Ollama...), key in LLM_API_KEY,
             base_url in the config
Answers are cached on disk with the hash of (provider, model, settings, prompts) as key, so a rerun costs
nothing and a change of settings (max_tokens, extra_body) does not reuse old answers.
With cache_only=True (used by the website, SE) the client never calls the LLM: a missing answer raises
CacheMiss, the caller serves the graph ranking and fills the cache in the background.

Cost and time (MLE-Ops): every real call records its latency and its token usage (from the API answer,
or estimated at 4 characters per token for the mock). The cost uses the prices of the config
(llm.price_input_per_mtok, llm.price_output_per_mtok, in dollars per million tokens).
`stats` sums everything for a run, `last_call` describes the last call (cache hit or not).

Free tiers (MLE-Core): a "429 Too Many Requests" or a 5xx answer is retried up to max_retries times,
waiting what the API asks (Retry-After) or 2, 4, 8... seconds. An answer cut by max_tokens or empty is
an error: it is never cached, and the caller falls back to the graph ranking. extra_body adds provider
options to the request, for example {"reasoning_effort": "low"} for reasoning models on Groq.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

RETRY_STATUS = {429, 500, 502, 503, 504}


class CacheMiss(RuntimeError):
    """The answer is not in the disk cache and the client may not call the LLM (cache_only)."""


class LLMClient:
    def __init__(self, provider="mock", model="", base_url="", temperature=0, max_tokens=800,
                 cache_dir="results/llm_cache", price_input_per_mtok=0.0, price_output_per_mtok=0.0,
                 timeout=60, cache_only=False, extra_body=None, max_retries=6, **_):
        self.provider, self.model, self.base_url = provider, model, (base_url or "").rstrip("/")
        self.temperature, self.max_tokens = temperature, max_tokens
        self.extra_body = dict(extra_body or {})
        self.max_retries = int(max_retries)
        self.cache_only = cache_only
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.price_in, self.price_out = float(price_input_per_mtok or 0), float(price_output_per_mtok or 0)
        self.timeout = timeout
        self.stats = {"calls": 0, "cache_hits": 0, "cache_misses": 0, "errors": 0, "retries": 0,
                      "seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        self.last_call = {}
        self._usage = (0, 0)

    @classmethod
    def from_config(cls, cfg: dict, cache_only=False):
        return cls(**cfg["llm"], cache_only=cache_only)

    def _key(self, system: str, user: str) -> str:
        settings = [self.provider, self.model, self.temperature]
        if self.provider != "mock":                   # mock answers do not depend on these settings
            settings += [self.max_tokens, sorted(self.extra_body.items())]
        raw = json.dumps(settings + [system, user], default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def complete(self, system: str, user: str) -> str:
        path = self.cache_dir / f"{self._key(system, user)}.json"
        t0 = time.perf_counter()
        if path.exists():
            self.stats["cache_hits"] += 1
            text = json.loads(path.read_text(encoding="utf-8"))["text"]
            self.last_call = {"cached": True, "seconds": time.perf_counter() - t0, "input_tokens": 0,
                              "output_tokens": 0, "cost_usd": 0.0}
            return text
        if self.cache_only:
            self.stats["cache_misses"] += 1
            raise CacheMiss("answer not in the LLM cache")
        try:
            text = getattr(self, f"_call_{self.provider}")(system, user)
        except Exception as e:  # noqa: BLE001  network, quota, bad key, cut answer: the caller falls back
            self.stats["errors"] += 1
            self.last_call = {"cached": False, "seconds": time.perf_counter() - t0, "input_tokens": 0,
                              "output_tokens": 0, "cost_usd": 0.0, "error": str(e)[:200]}
            raise RuntimeError(f"LLM call failed: {e}") from e
        seconds = time.perf_counter() - t0
        tin, tout = self._usage
        if not tin:                                   # mock, or an API that gives no usage
            tin, tout = (len(system) + len(user)) // 4, len(text) // 4
        cost = self.cost(tin, tout)
        self.last_call = {"cached": False, "seconds": seconds, "input_tokens": tin,
                          "output_tokens": tout, "cost_usd": cost}
        self.stats["calls"] += 1
        self.stats["seconds"] += seconds
        self.stats["input_tokens"] += tin
        self.stats["output_tokens"] += tout
        self.stats["cost_usd"] += cost
        # write then rename, so the API never reads a half-written answer from another thread
        tmp = path.with_suffix(f".{os.getpid()}.{time.time_ns()}.tmp")
        tmp.write_text(json.dumps({"text": text, "model": self.model, **self.last_call}),
                       encoding="utf-8")
        os.replace(tmp, path)
        return text

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.price_in + output_tokens * self.price_out) / 1e6

    def reset_stats(self):
        self.stats = {k: 0 if isinstance(v, int) else 0.0 for k, v in self.stats.items()}

    def _post(self, url, headers, payload):
        """POST with retries on 429 and 5xx (free tiers limit the calls per minute)."""
        for attempt in range(self.max_retries + 1):
            r = requests.post(url, timeout=self.timeout, headers=headers, json=payload)
            if r.status_code not in RETRY_STATUS or attempt == self.max_retries:
                r.raise_for_status()
                return r.json()
            self.stats["retries"] += 1
            try:
                wait = float(r.headers.get("retry-after", ""))
            except ValueError:
                wait = 2.0 ** (attempt + 1)
            time.sleep(min(max(wait, 1.0), 60.0))
        raise RuntimeError("unreachable")

    def _call_anthropic(self, system, user):
        self._usage = (0, 0)
        data = self._post("https://api.anthropic.com/v1/messages", {
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
            "content-type": "application/json"}, {
            "model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
            "system": system, "messages": [{"role": "user", "content": user}], **self.extra_body})
        u = data.get("usage") or {}
        self._usage = (u.get("input_tokens", 0), u.get("output_tokens", 0))
        if data.get("stop_reason") == "max_tokens":
            raise RuntimeError("answer cut by max_tokens")
        return "".join(b.get("text", "") for b in data["content"])

    def _call_openai(self, system, user):
        self._usage = (0, 0)
        url = (self.base_url or "https://api.openai.com/v1") + "/chat/completions"
        data = self._post(url, {"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"}, {
            "model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **self.extra_body})
        u = data.get("usage") or {}
        self._usage = (u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        choice = data["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        if choice.get("finish_reason") == "length":
            raise RuntimeError(f"answer cut by max_tokens ({self.max_tokens})")
        if not text.strip():
            raise RuntimeError("empty answer")
        return text

    def _call_mock(self, system, user):
        """Keeps the given order and gives a short reason: lets the whole pipeline run offline."""
        self._usage = (0, 0)
        if '"task": "profile_summary"' in user:
            return json.dumps({"summary": "Reader of the topics listed in the profile."})
        ids = re.findall(r'"id": "([^"]+)"', user)
        return json.dumps({"ranking": [{"id": i, "reason": ""} for i in ids]})
