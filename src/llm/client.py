"""One LLM client for the whole project, with a disk cache (MLE-Ops owns it, MLE-Core uses it).

Providers:
  mock       no network, deterministic answers; for tests and for running the pipeline without a key
  anthropic  Claude API, key in the ANTHROPIC_API_KEY environment variable
  openai     any OpenAI-compatible API (OpenAI, Mistral, Groq, Ollama...), key in LLM_API_KEY,
             base_url in the config
Answers are cached on disk with the hash of (provider, model, prompts) as key, so a rerun costs nothing.
With cache_only=True (used by the website, SE) the client never calls the LLM: a missing answer raises
CacheMiss, the caller serves the graph ranking and fills the cache in the background.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests


class CacheMiss(RuntimeError):
    """The answer is not in the disk cache and the client may not call the LLM (cache_only)."""


class LLMClient:
    def __init__(self, provider="mock", model="", base_url="", temperature=0, max_tokens=800,
                 cache_dir="results/llm_cache", cache_only=False, **_):
        self.provider, self.model, self.base_url = provider, model, base_url.rstrip("/")
        self.temperature, self.max_tokens = temperature, max_tokens
        self.cache_only = cache_only
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"calls": 0, "cache_hits": 0, "cache_misses": 0, "errors": 0, "seconds": 0.0}

    @classmethod
    def from_config(cls, cfg: dict, cache_only=False):
        return cls(**cfg["llm"], cache_only=cache_only)

    def _key(self, system: str, user: str) -> str:
        raw = json.dumps([self.provider, self.model, self.temperature, system, user])
        return hashlib.sha256(raw.encode()).hexdigest()

    def complete(self, system: str, user: str) -> str:
        path = self.cache_dir / f"{self._key(system, user)}.json"
        if path.exists():
            self.stats["cache_hits"] += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        if self.cache_only:
            self.stats["cache_misses"] += 1
            raise CacheMiss("answer not in the LLM cache")
        t0 = time.time()
        try:
            text = getattr(self, f"_call_{self.provider}")(system, user)
        except Exception as e:  # noqa: BLE001  network, quota, bad key: the caller falls back
            self.stats["errors"] += 1
            raise RuntimeError(f"LLM call failed: {e}") from e
        self.stats["calls"] += 1
        self.stats["seconds"] += time.time() - t0
        # write then rename, so the API never reads a half-written answer from another thread
        tmp = path.with_suffix(f".{os.getpid()}.{time.time_ns()}.tmp")
        tmp.write_text(json.dumps({"text": text, "model": self.model}), encoding="utf-8")
        os.replace(tmp, path)
        return text

    def _call_anthropic(self, system, user):
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
            "content-type": "application/json"}, json={
            "model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
            "system": system, "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])

    def _call_openai(self, system, user):
        url = (self.base_url or "https://api.openai.com/v1") + "/chat/completions"
        r = requests.post(url, timeout=60, headers={
            "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"}, json={
            "model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _call_mock(self, system, user):
        """Keeps the given order and gives a short reason: lets the whole pipeline run offline."""
        ids = re.findall(r'"id": "([^"]+)"', user)
        return json.dumps({"ranking": [{"id": i, "reason": ""} for i in ids]})
