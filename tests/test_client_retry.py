"""LLM client: retries on 429, never caches a cut or empty answer (no network needed)."""
import pytest

import src.llm.client as client_mod
from src.llm.client import LLMClient


class FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


def answer(text, finish="stop"):
    return FakeResponse(200, {"choices": [{"message": {"content": text}, "finish_reason": finish}],
                              "usage": {"prompt_tokens": 10, "completion_tokens": 5}})


def make_client(tmp_path):
    return LLMClient(provider="openai", model="m", base_url="http://x", cache_dir=tmp_path,
                     extra_body={"reasoning_effort": "low"}, max_retries=3)


def test_retry_after_429(tmp_path, monkeypatch):
    replies = [FakeResponse(429, headers={"retry-after": "0"}), answer('{"ok": true}')]
    sent = []
    monkeypatch.setattr(client_mod.requests, "post", lambda *a, **k: sent.append(k["json"]) or replies.pop(0))
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    c = make_client(tmp_path)
    assert c.complete("s", "u") == '{"ok": true}'
    assert c.stats["retries"] == 1 and sent[0]["reasoning_effort"] == "low"


@pytest.mark.parametrize("reply", [answer('{"ranking": [', "length"), answer("   ")])
def test_cut_or_empty_answer_is_not_cached(tmp_path, monkeypatch, reply):
    monkeypatch.setattr(client_mod.requests, "post", lambda *a, **k: reply)
    c = make_client(tmp_path)
    with pytest.raises(RuntimeError):
        c.complete("s", "u")
    assert not list(tmp_path.glob("*.json"))


def test_settings_change_the_cache_key(tmp_path):
    a = make_client(tmp_path)
    b = make_client(tmp_path)
    b.max_tokens = 2000
    assert a._key("s", "u") != b._key("s", "u")
