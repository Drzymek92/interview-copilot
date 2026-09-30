"""Tests for `scripts/llm_client.py`'s local-backend `max_tokens` handling (#991).

NO GPU, NO network: `urllib.request.urlopen` and `ChatOpenAI.invoke` are monkeypatched. See
`scripts/reasoning.md` "Ollama silently truncates" section and the 2026-09-23 changelog entry for
the measured root cause this pins: `langchain_openai.ChatOpenAI` renames `max_tokens` to
`max_completion_tokens` on the wire, a name Ollama's OpenAI-compatible endpoint silently ignores.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import llm_client  # noqa: E402


class _FakeSseResponse:
    """Mimics the `with urllib.request.urlopen(...) as response:` SSE iteration `_stream_local`
    does: iterating the object yields raw `b"data: {...}\\n"` lines terminated by `[DONE]`."""

    def __init__(self, chunks: list[dict], usage: dict) -> None:
        lines = [f"data: {json.dumps(c)}".encode() for c in chunks]
        final = {"choices": [{"delta": {}}], "usage": usage, "model": "interview-copilot:14b"}
        lines.append(f"data: {json.dumps(final)}".encode())
        lines.append(b"data: [DONE]")
        self._lines = lines

    def __enter__(self) -> "_FakeSseResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self):
        return iter(self._lines)


def _install_fake_urlopen(monkeypatch, captured: dict, *, out_tokens: int, text: str = "x") -> None:
    """Replace `urllib.request.urlopen` with a fake that records the outgoing request body in
    `captured["body"]` and returns a canned single-token-ish SSE stream capped at `out_tokens`."""
    chunks = [{"choices": [{"delta": {"content": text}}]}] if out_tokens else []

    def fake_urlopen(request, timeout=None):  # noqa: ARG001 - signature must match the real call
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["url"] = request.full_url
        return _FakeSseResponse(chunks, {"prompt_tokens": 12, "completion_tokens": out_tokens})

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", fake_urlopen)


def test_local_capped_call_sends_literal_max_tokens_not_max_completion_tokens(monkeypatch):
    """The regression this subtask fixes: `llm_call(..., max_tokens=1)` on the local backend
    must reach Ollama as a literal `max_tokens` key. Sending `max_completion_tokens` instead
    (what a bare `ChatOpenAI.invoke()` does) was measured to make Ollama ignore the cap entirely
    (2026-09-23: 2486 completion tokens back against a cap of 3)."""
    captured: dict = {}
    _install_fake_urlopen(monkeypatch, captured, out_tokens=1)

    # A poisoned ChatOpenAI.invoke proves the non-streaming langchain path is not used for a
    # capped local call — if it were, this would raise instead of the fake HTTP path answering.
    def _boom(*_a: object, **_kw: object) -> None:
        raise AssertionError("llm_call must not go through ChatOpenAI.invoke() for a capped local call")

    monkeypatch.setattr(llm_client.ChatOpenAI, "invoke", _boom)

    reply = llm_client.llm_call("hello", backend="local", max_tokens=1)

    assert "body" in captured, "the raw HTTP local path was never hit"
    assert captured["body"]["max_tokens"] == 1
    assert "max_completion_tokens" not in captured["body"]
    assert reply.completion_tokens == 1


def test_local_uncapped_call_is_unaffected(monkeypatch):
    """No `max_tokens` passed -> the existing `ChatOpenAI.invoke()` path is untouched; this test
    would fail loudly (via the fake urlopen never being installed/consumed) if the fix's new
    branch over-matched and started intercepting uncapped calls too."""
    from langchain_core.messages import AIMessage

    def fake_invoke(self, messages, *a: object, **kw: object):  # noqa: ARG001
        return AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        )

    monkeypatch.setattr(llm_client.ChatOpenAI, "invoke", fake_invoke)

    def _boom_urlopen(*_a: object, **_kw: object) -> None:
        raise AssertionError("an uncapped local call must not use the raw HTTP path")

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _boom_urlopen)

    reply = llm_client.llm_call("hello", backend="local")
    assert reply.text == "ok"


def test_cloud_capped_call_is_unaffected(monkeypatch):
    """The fix is local-only (#991 scope): a capped cloud call must still go through
    `_call_cloud`, never the Ollama raw-HTTP path."""
    monkeypatch.setenv("CLOUD_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("CLOUD_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("CLOUD_API_KEY", "test-key")

    calls: dict = {}

    def fake_call_cloud(system, prompt, model, max_tokens, on_token):  # noqa: ARG001
        calls["max_tokens"] = max_tokens
        return "ok", {"input_tokens": 5, "output_tokens": 1}, None, model

    monkeypatch.setattr(llm_client, "_call_cloud", fake_call_cloud)

    def _boom_urlopen(*_a: object, **_kw: object) -> None:
        raise AssertionError("a cloud call must not use the local raw-HTTP path")

    monkeypatch.setattr(llm_client.urllib.request, "urlopen", _boom_urlopen)

    reply = llm_client.llm_call("hello", backend="cloud", max_tokens=1)
    assert calls["max_tokens"] == 1
    assert reply.text == "ok"


def test_prefill_bundle_caps_output_at_the_raw_request_level(monkeypatch, tmp_path):
    """`reasoning.prefill_bundle` calls `llm_call(..., max_tokens=1)` on the local backend
    (session 18 measured 335 output tokens escaping the cap). Assert end-to-end, through
    `reasoning.prefill_bundle`, that the request Ollama receives caps at 1."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts import reasoning

    captured: dict = {}
    _install_fake_urlopen(monkeypatch, captured, out_tokens=1)
    monkeypatch.setattr(llm_client.ChatOpenAI, "invoke", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("prefill must use the raw HTTP path, not ChatOpenAI.invoke()")
    ))

    bundle = reasoning.ContextBundle(
        session_id="t", role="r", company="c",
        spoken_language="en", suggestion_language="match",
        job_description="jd", company_brief="cb", resume="", answer_bank=[],
        plan=[], honesty_boundary=[], placeholders=[], source_dir=tmp_path,
        question_bank=[],
    )
    took = reasoning.prefill_bundle(bundle, backend="local", model="interview-copilot:14b")

    assert captured["body"]["max_tokens"] == 1
    assert "max_completion_tokens" not in captured["body"]
    assert took >= 0.0
