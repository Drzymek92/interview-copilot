"""LLM connection for interview_copilot (D14 — BYOK cloud default, local fallback).

Two backends behind one call, selected by `REASONING_BACKEND` (CFG, `config/settings.py`):

| backend  | how it is reached                          | key                    | egress |
|----------|--------------------------------------------|------------------------|--------|
| `local`  | Ollama `localhost:11434/v1`, no vendor SDK | placeholder (`ollama`) | none   |
| `cloud`  | Claude via the official `anthropic` SDK    | `CLOUD_API_KEY`, BYOK  | YES    |

**SI1 (non-waivable).** Interview audio, transcript and the context bundle are local by default.
The cloud backend is an *explicit* transcript egress, so it is:
  * never the default — `REASONING_BACKEND` defaults to `local`;
  * never silently reachable — it refuses to run without `CLOUD_MODEL` + `CLOUD_API_KEY`
    in the environment (no baked-in model, no baked-in key);
  * never quiet — `announce_backend()` prints a one-line egress banner naming the host every
    time a cloud client is built, and `reasoning.py` prints it before the first call.
**Claude has no OpenAI-compatible endpoint** — its API is `/v1/messages` with `x-api-key` +
`anthropic-version`, so `langchain_openai` cannot reach it. The cloud path therefore imports the
official `anthropic` SDK, which CLAUDE.md forbids; the user waived that rule for this path on
2026-09-02 and it is recorded as a Policy Override (D6) in `agent/project.md`. The local backend
still imports no vendor SDK. See `_call_cloud` for the two Claude-specific 400 traps.

Every call is metered — `LlmReply` carries prompt/completion tokens and wall-clock — because the
ambient loop (D12) has to know what a suggestion costs before it can decide how often to fire.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from scripts.logger import get_logger  # noqa: E402

logger = get_logger("llm_client")

# Load env with fallback: project-local, then the central Claude_Projects/config/.env.
# dotenv does NOT override vars already set in the environment, so explicit env still wins.
try:
    from dotenv import load_dotenv

    _here = Path(__file__).resolve()
    for _p in (
        Path("config") / ".env",
        _here.parents[1] / "config" / ".env",
        _here.parents[3] / "config" / ".env",
    ):
        if _p.is_file():
            load_dotenv(_p)
except Exception:  # noqa: BLE001 — a missing dotenv must never block a local run
    pass


# Above this chars-per-token ratio the backend cannot have seen everything that was sent.
TRUNCATION_CHARS_PER_TOKEN = 7.0


def counted_prompt_tokens(usage: dict) -> int:
    """Total prompt tokens the backend actually saw.

    `input_tokens` alone is the UNCACHED REMAINDER once prompt caching is on — the cached
    prefix is billed and reported separately. Summing all three is the only correct measure
    of "did the whole prompt arrive".
    """
    return (
        int(usage.get("input_tokens", 0))
        + int(usage.get("cache_read_tokens", 0))
        + int(usage.get("cache_write_tokens", 0))
    )


def prompt_was_truncated(sent_chars: int, usage: dict) -> bool:
    """Natural language runs ~3-4.5 chars/token; past 7 means text went in that never
    reached the model. Returns False when the backend reported no usage at all."""
    counted = counted_prompt_tokens(usage)
    return bool(counted) and sent_chars / counted > TRUNCATION_CHARS_PER_TOKEN


class BackendUnavailable(RuntimeError):
    """The selected backend cannot be built (missing config, or the server is down)."""


class StreamCancelled(Exception):
    """Raised by a caller's `on_token` to abandon a stream in progress.

    It must travel through `llm_call` untouched: a cancelled suggestion is a normal event
    (the interviewer asked the next question), NOT a backend failure. Wrapping it in
    BackendUnavailable printed a traceback and the word "failed" on every interruption.
    """


@dataclass
class LlmReply:
    """One completion plus what it cost. `tokens_known` is False when the backend
    returned no usage block — better an honest gap than a fabricated number."""

    text: str
    backend: str
    model: str
    latency_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_known: bool = True
    # Prompt-cache accounting (cloud only; Ollama caches its own KV and reports nothing).
    # cache_write is billed at ~1.25x base input, cache_read at ~0.1x — so a non-zero
    # cache_read is the ONLY proof the cache is actually working. If it stays 0 across
    # repeated calls, something in the prefix is changing per request.
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    # Time to the FIRST token. Measured 2026-09-02: 0.15 s against 8.9k prompt tokens, i.e.
    # prefill is free and the whole wait is generation. That is why streaming — not a smaller
    # prompt and not a different model — is the latency lever for a mid-answer reader.
    first_token_seconds: float | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost_line(self) -> str:
        """The one line #322 needs to decide whether it can afford to fire per turn."""
        tokens = (
            f"{self.prompt_tokens} in + {self.completion_tokens} out = {self.total_tokens} tok"
            if self.tokens_known
            else "tokens unreported by backend"
        )
        timing = (
            f"{self.first_token_seconds:.2f}s to first token, {self.latency_seconds:.2f}s total"
            if self.first_token_seconds is not None
            else f"{self.latency_seconds:.2f}s"
        )
        cache = ""
        if self.cache_read_tokens or self.cache_write_tokens:
            cache = f" | cache {self.cache_read_tokens} read / {self.cache_write_tokens} written"
        return f"{timing} | {tokens}{cache} | {self.backend}:{self.model}"


def _cfg(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise BackendUnavailable(
            f"{name} is not set. The cloud backend is BYOK: set CLOUD_API_KEY and CLOUD_MODEL "
            f"in config/.env (see config/.env.example), or run with REASONING_BACKEND=local to "
            f"keep everything on this machine."
        )
    return val


def cloud_ready() -> bool:
    """True iff the BYOK cloud egress is actually configured — both CLOUD_MODEL and
    CLOUD_API_KEY present. The #607 app uses it to enable (or grey out) the "local + api"
    switch: offering an egress the box cannot make would be a lie, and flipping it would fail
    on the first question mid-interview. This only reads config; it never opens a socket."""
    return bool(model_for("cloud")) and bool(os.environ.get("CLOUD_API_KEY"))


def resolve_backend(backend: str | None = None) -> str:
    chosen = (backend or settings.REASONING_BACKEND).strip().lower()
    if chosen not in ("local", "cloud"):
        raise BackendUnavailable(
            f"unknown backend {chosen!r} — REASONING_BACKEND must be 'local' or 'cloud'"
        )
    return chosen


def model_for(backend: str) -> str:
    """Resolve the model for a backend.

    Falls back to the live environment because `config.settings` is imported BEFORE this
    module loads `config/.env`, so a value that lives only in the dotenv file is empty on
    the settings object. That bug made the SI1 egress banner announce "(model )" — naming
    no model on the one line whose whole job is to say where data is going.
    """
    if backend == "cloud":
        return settings.CLOUD_MODEL or os.environ.get("CLOUD_MODEL", "")
    return settings.LOCAL_MODEL or os.environ.get("OLLAMA_MODEL", "")


def announce_backend(backend: str, model: str) -> str:
    """SI1: cloud egress is opt-in *and visible*. Returns the banner so callers can
    print it, log it, or push it to the D18 dashboard — one wording, one place."""
    if backend == "cloud":
        host = urlparse(_cfg("CLOUD_BASE_URL", "")).netloc or "api.anthropic.com"
        return (
            f"!! EGRESS: transcript + context bundle are being sent to {host} "
            f"(model {model}). Set REASONING_BACKEND=local to keep everything on this machine."
        )
    return f"LOCAL ONLY: reasoning runs on this machine via Ollama (model {model}). No egress."


def get_llm(
    backend: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    **kwargs: object,
) -> ChatOpenAI:
    chosen = resolve_backend(backend)
    name = model or model_for(chosen)
    if chosen == "cloud":
        # The cloud path does not go through ChatOpenAI at all — Claude is not
        # OpenAI-compatible. `llm_call` routes straight to `_call_cloud`.
        raise BackendUnavailable(
            "get_llm() is the local (OpenAI-compatible) client only; the cloud backend uses "
            "the anthropic SDK via _call_cloud(). Call llm_call(backend='cloud') instead."
        )
    if True:
        # localhost needs no real key; Ollama ignores the value but the client
        # requires a non-empty string.
        base_url = _cfg("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        api_key = _cfg("OLLAMA_API_KEY", "ollama")
        # Qwen3 is a THINKING model: it burns reasoning tokens Ollama then strips out of
        # `content`, so you pay seconds of latency for text you never see (measured
        # 2026-09-02: 82-135 completion tokens to answer "reply with exactly: ok").
        # `reasoning_effort="none"` cuts that to 2. Verified harmless on non-thinking models
        # (llama3.1:8b answers normally with it set), so it is safe to apply to any local
        # model. NOT applied to cloud — there the same parameter name means something else.
        # `/no_think` in the prompt and `think:false` in the body BOTH fail silently through
        # the OpenAI-compatible endpoint; only this one works.
        if settings.LOCAL_DISABLE_THINKING:
            kwargs.setdefault("reasoning_effort", "none")
    return ChatOpenAI(
        model=name,
        base_url=base_url,
        api_key=api_key,
        temperature=settings.REASONING_TEMPERATURE if temperature is None else temperature,
        timeout=settings.REASONING_TIMEOUT_SECONDS,
        max_retries=0,  # a stale suggestion is worse than none — fail fast, next segment retries
        **kwargs,
    )


def _call_cloud(
    system: str | None, prompt: str, model: str, max_tokens: int | None,
    on_token: Callable[[str], None] | None,
) -> tuple[str, dict, float | None, str]:
    """Call Claude through the official `anthropic` SDK (D14 cloud backend, SI1 egress).

    POLICY OVERRIDE (D6, recorded in agent/project.md): CLAUDE.md says never import a vendor
    SDK. It cannot be honoured here — **Claude has no OpenAI-compatible endpoint**, so
    `langchain_openai` cannot reach it, and Anthropic's own guidance is "official SDK, or raw
    HTTP only where no SDK exists" (Python has one). User waived the rule for this path on
    2026-09-02. The local backend still uses no vendor SDK.

    Two Claude-specific constraints, both of which would 400 the naive port:
      * **No `temperature`.** Sampling parameters were removed on Opus 4.8 / Sonnet 5 — sending
        `temperature=0.0` (which `REASONING_TEMPERATURE` would have supplied) returns a 400.
      * **`max_tokens` is required**, not optional as on the OpenAI-compatible shape.
    Thinking is deliberately left OFF (omitted): on Opus 4.8 an omitted `thinking` field runs
    without thinking, and for a copilot judged on time-to-first-token, reasoning tokens are
    latency the user pays for and never reads.
    """
    import anthropic  # imported lazily so the local path never pays for it

    # An IDENTITY-LINKED key (measured 2026-09-02) is rejected with a 400 unless the request
    # names the workspace it acts in: "anthropic-workspace-id is required when authenticating
    # with an identity-linked API key". Plain org keys don't need it, so it stays optional.
    headers = {}
    workspace = _cfg("CLOUD_WORKSPACE_ID", "")
    if workspace:
        headers["anthropic-workspace-id"] = workspace
    client = anthropic.Anthropic(
        api_key=_require("CLOUD_API_KEY"),
        default_headers=headers or None,
    )
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens or settings.SUGGESTION_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        # PROMPT CACHING. The system block is the whole context bundle and is byte-identical
        # on every call of a session; the volatile part (the transcript segment and history)
        # lives in the user turn, which renders AFTER system — so the cached prefix survives
        # every question. Caching is a PREFIX match, so that ordering is what makes it work.
        # ~14k tokens, far above Opus 4.8's 4096-token minimum cacheable prefix.
        # Default 5-minute TTL is right here: questions arrive every 10-30 s, and the 1h TTL
        # costs 2x to write instead of 1.25x.
        block: dict = {"type": "text", "text": system}
        if settings.CLOUD_PROMPT_CACHE:
            block["cache_control"] = {"type": "ephemeral"}
        kwargs["system"] = [block]

    started = time.perf_counter()
    first_token: float | None = None
    pieces: list[str] = []

    if on_token is None:
        message = client.messages.create(**kwargs)
        text = "".join(b.text for b in message.content if b.type == "text")
    else:
        with client.messages.stream(**kwargs) as stream:
            for delta in stream.text_stream:
                if delta:
                    if first_token is None:
                        first_token = time.perf_counter() - started
                    pieces.append(delta)
                    on_token(delta)
            message = stream.get_final_message()
        text = "".join(pieces)

    usage = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        # Absent on providers/paths that don't cache — never fabricate a zero as a hit.
        "cache_write_tokens": getattr(message.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_tokens": getattr(message.usage, "cache_read_input_tokens", 0) or 0,
    }
    return text, usage, first_token, message.model


def _stream_local(
    system: str | None, prompt: str, model: str, temperature: float, max_tokens: int | None,
    on_token: Callable[[str], None],
) -> tuple[str, dict, float | None, str]:
    """Stream from Ollama over plain HTTP, bypassing langchain.

    MEASURED 2026-09-02, identical prompt and model, 3 reps: **66.2 tok/s here vs 21.3 tok/s
    through `langchain_openai.stream()`** — a 3.1x penalty from per-chunk client overhead, not
    from the model. On a live interview that is the difference between a suggestion completing
    in 4 s and in 10 s, so it is worth the deviation from the house "always langchain" rule
    (recorded as a Policy Override in agent/project.md per D6).

    The deviation is deliberately narrow: **local only**. This talks to `localhost` over the
    same OpenAI-compatible schema, imports no vendor SDK, and the cloud path is untouched —
    that is where a real gateway, real auth and real retry semantics live, and where langchain
    earns its overhead.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    body: dict[str, object] = {
        "model": model, "messages": messages, "stream": True, "temperature": temperature,
        "stream_options": {"include_usage": True},
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if settings.LOCAL_DISABLE_THINKING:
        body["reasoning_effort"] = "none"

    url = _cfg("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    pieces: list[str] = []
    usage: dict = {}
    first_token: float | None = None
    model_name = model
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=settings.REASONING_TIMEOUT_SECONDS) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                # OpenAI names them prompt/completion; LlmReply speaks langchain's input/output.
                usage = {
                    "input_tokens": chunk["usage"].get("prompt_tokens", 0),
                    "output_tokens": chunk["usage"].get("completion_tokens", 0),
                }
            model_name = chunk.get("model") or model_name
            choices = chunk.get("choices") or [{}]
            delta = (choices[0].get("delta") or {}).get("content") or ""
            if delta:
                if first_token is None:
                    first_token = time.perf_counter() - started
                pieces.append(delta)
                on_token(delta)  # may raise to cancel — the `with` closes the connection
    return "".join(pieces), usage, first_token, model_name


def llm_call(
    prompt: str,
    system: str | None = None,
    backend: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    on_token: Callable[[str], None] | None = None,
    **kwargs: object,
) -> LlmReply:
    """One metered completion. Raises BackendUnavailable when the server is unreachable.

    Pass `on_token` to stream: it is called with each text fragment as it arrives, and the
    reply's `first_token_seconds` is then populated. A mid-answer reader cares far more about
    when the first line appears than when the last one does.
    """
    chosen = resolve_backend(backend)
    if max_tokens is not None:
        # `max_tokens`, NOT `max_completion_tokens` — Ollama's OpenAI-compatible endpoint does
        # not honour the newer name and will happily overrun the cap.
        kwargs["max_tokens"] = max_tokens
    if on_token is not None:
        kwargs["stream_usage"] = True  # otherwise the streamed chunks carry no usage block
    llm = None if chosen == "cloud" else get_llm(
        backend=chosen, model=model, temperature=temperature, **kwargs
    )
    messages: list[SystemMessage | HumanMessage] = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    started = time.perf_counter()
    first_token: float | None = None
    try:
        if chosen == "cloud":
            banner = announce_backend(chosen, model or model_for("cloud"))
            logger.warning(banner)
            print(banner, file=sys.stderr, flush=True)
            text, usage, first_token, model_name = _call_cloud(
                system, prompt, model or _require("CLOUD_MODEL"), max_tokens, on_token,
            )
        elif on_token is not None and chosen == "local" and settings.LOCAL_FAST_STREAM:
            text, usage, first_token, model_name = _stream_local(
                system, prompt, model or model_for("local"),
                settings.REASONING_TEMPERATURE if temperature is None else temperature,
                max_tokens, on_token,
            )
        elif on_token is None:
            resp = llm.invoke(messages)
            text = str(resp.content)
            usage = getattr(resp, "usage_metadata", None) or {}
            model_name = getattr(resp, "response_metadata", {}).get("model_name")
        else:
            pieces: list[str] = []
            usage, model_name = {}, None
            for chunk in llm.stream(messages):
                piece = str(chunk.content or "")
                if piece:
                    if first_token is None:
                        first_token = time.perf_counter() - started
                    pieces.append(piece)
                    on_token(piece)
                if getattr(chunk, "usage_metadata", None):
                    usage = chunk.usage_metadata
                model_name = getattr(chunk, "response_metadata", {}).get("model_name") or model_name
            text = "".join(pieces)
    except StreamCancelled:
        raise  # a superseded suggestion is a normal event, not a backend failure
    except BackendUnavailable:
        # A missing CLOUD_MODEL/CLOUD_API_KEY is a config error, not a call failure —
        # re-wrapping it buried the real cause under "Check ... network reachability"
        # and let a test pass by matching the stale wrapper text instead.
        raise
    except Exception as exc:  # noqa: BLE001 — translate to one actionable error
        raise BackendUnavailable(
            f"{chosen} backend call failed ({type(exc).__name__}: {exc}). "
            + (
                "Is the Ollama server up (`ollama list`) and the model pulled?"
                if chosen == "local"
                else "Check CLOUD_API_KEY / CLOUD_MODEL and network reachability."
            )
        ) from exc
    latency = time.perf_counter() - started

    # SILENT-TRUNCATION GUARD. Measured 2026-09-02: Ollama drops whatever does not fit the
    # model's num_ctx and reports the TRUNCATED count as prompt_tokens — no error, no warning.
    # A 12k-char system prompt came back as 2050 tokens with 80% of the context bundle gone,
    # and the only visible symptom was a model that answered oddly. Its OpenAI-compatible
    # endpoint also IGNORES `options.num_ctx`, so the window can only be raised by baking it
    # into the model (`ollama create ... PARAMETER num_ctx N`). Natural language runs ~3-4.5
    # chars/token, so anything past 7 means text went in that never reached the model.
    sent_chars = sum(len(str(m.content)) for m in messages)
    prompt_tokens = int(usage.get("input_tokens", 0))
    # With prompt caching, `input_tokens` is only the UNCACHED REMAINDER — the cached prefix is
    # reported separately. Dividing by it alone made the guard fire on every cached call
    # (39k chars / 77 tokens = 510 chars/token) and cry truncation at a prompt that arrived
    # whole. The real prompt size is the sum of all three.
    counted_tokens = counted_prompt_tokens(usage)
    if prompt_was_truncated(sent_chars, usage):
        logger.error(
            "PROMPT TRUNCATED: sent %d chars but the backend counted only %d prompt tokens "
            "(%.1f chars/token, cache read %d / written %d). The model did NOT see the whole "
            "prompt — raise the model's num_ctx (ollama create with PARAMETER num_ctx) or "
            "shorten the bundle.",
            sent_chars, counted_tokens, sent_chars / counted_tokens,
            usage.get("cache_read_tokens", 0), usage.get("cache_write_tokens", 0),
        )

    reply = LlmReply(
        text=text.strip(),
        backend=chosen,
        model=str(model_name or (llm.model_name if llm is not None else model_for(chosen))),
        latency_seconds=latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=int(usage.get("output_tokens", 0)),
        tokens_known=bool(usage),
        cache_write_tokens=int(usage.get("cache_write_tokens", 0)),
        cache_read_tokens=int(usage.get("cache_read_tokens", 0)),
        first_token_seconds=first_token,
    )
    logger.info("llm_call | %s", reply.cost_line())
    return reply


def smoke_test(backend: str | None = None, model: str | None = None) -> LlmReply:
    """One tiny live call proving the backend answers. Fails loudly with the reason."""
    return llm_call("Reply with exactly: ok", backend=backend, model=model, max_tokens=16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--smoke", action="store_true", help="one tiny live call; prints the metered result")
    parser.add_argument("--backend", choices=("local", "cloud"), help="override REASONING_BACKEND")
    parser.add_argument("--model", help="override the backend's model")
    parser.add_argument("prompt", nargs="?", help="send this prompt instead of the smoke test")
    args = parser.parse_args()

    try:
        if args.prompt:
            reply = llm_call(args.prompt, backend=args.backend, model=args.model)
        elif args.smoke:
            reply = smoke_test(backend=args.backend, model=args.model)
        else:
            parser.error("pass a prompt or --smoke")
    except BackendUnavailable:
        logger.exception("backend unavailable")
        sys.exit(1)
    print(reply.text)
    print(f"[{reply.cost_line()}]", file=sys.stderr)


if __name__ == "__main__":
    main()
