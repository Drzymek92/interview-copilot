"""Tests for the salience gate (D23 / #386).

Same split as `test_reasoning.py`: everything above `test_live_*` is deterministic and runs
with no GPU, no server and no tokens — the gate's own logic (sentence extraction, fail-open,
config precedence) is tested against a stub judge. The live tests at the bottom make a REAL
model call, because this project's record is that its defects are found by running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from scripts import reasoning, salience  # noqa: E402
from scripts.llm_client import BackendUnavailable, smoke_test  # noqa: E402

EXAMPLE_SESSION = "example_ai_engineer"
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "sessions"


@pytest.fixture(scope="module")
def bundle():
    return reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)


def _gate(bundle, **kwargs):
    kwargs.setdefault("enabled", True)
    kwargs.setdefault("is_question", reasoning.looks_like_question)
    return salience.SalienceGate(bundle, **kwargs)


# ── 1. Question-sentence extraction (the biggest measured accuracy lever) ─
def test_extraction_keeps_only_the_interrogative_part():
    """0.57 -> 0.88 F1 on the 3B judge came from this alone: a 30 s merged segment wraps the
    real question in greetings and backchannel, and the judge loses it in the noise."""
    merged = (
        "Dzień dobry, witam serdecznie. Bardzo się cieszę, że udało nam się spotkać. "
        "Jakie są Pana oczekiwania względem nowego miejsca pracy?"
    )
    kept = salience.question_sentences(merged, reasoning.looks_like_question)
    assert "oczekiwania" in kept
    assert "Dzień dobry" not in kept


def test_extraction_falls_back_to_the_whole_text_when_nothing_looks_interrogative():
    text = "Tak jak powiedziałam, u nas zatrudnienie jest w ramach umowy o pracę."
    assert salience.question_sentences(text, reasoning.looks_like_question) == text


def test_extraction_survives_text_with_no_sentence_punctuation():
    assert salience.question_sentences("", reasoning.looks_like_question) == ""


# ── 2. The gate contract ─────────────────────────────────────────────────
def test_gate_off_fires_everything(bundle):
    """`--no-salience` must restore exactly the pre-#386 behaviour."""
    verdict = _gate(bundle, enabled=False).evaluate("cokolwiek")
    assert verdict.fire and verdict.reason == "gate-off"


def test_backend_off_is_equivalent_to_disabled(bundle):
    assert not _gate(bundle, backend="off").active


def test_gate_fails_open_when_the_backend_is_unreachable(bundle):
    """The interview-safe default: a broken gate degrades to today's behaviour rather than
    going silent. A wasted call costs tokens; a missed suggestion costs the interview."""
    gate = _gate(bundle, backend="llm")
    gate._ask_llm = lambda turn: (_ for _ in ()).throw(OSError("connection refused"))
    verdict = gate.evaluate("Jakie są Pana oczekiwania?")
    assert verdict.fire and verdict.reason == "fail-open"


def test_gate_can_be_made_to_fail_closed(bundle):
    gate = _gate(bundle, backend="llm", fail_open=False)
    gate._ask_llm = lambda turn: (_ for _ in ()).throw(OSError("connection refused"))
    verdict = gate.evaluate("Jakie są Pana oczekiwania?")
    assert not verdict.fire and verdict.reason == "fail-closed"


def test_warm_up_failure_is_not_fatal(bundle):
    """A gate that cannot warm must still let the interview start."""
    gate = _gate(bundle, backend="llm")
    gate._ask_llm = lambda turn: (_ for _ in ()).throw(OSError("nope"))
    gate.warm()  # must not raise


def test_verdict_carries_why_for_the_dashboard(bundle):
    """#322 renders only the filtered suggestions, so it needs the reason a turn was dropped."""
    gate = _gate(bundle, backend="llm")
    gate._ask_llm = lambda turn: False
    verdict = gate.evaluate("Czy dobrze mnie słychać?")
    assert not verdict.fire and verdict.reason == "not-salient"
    assert "llm" in verdict.log_line()


# ── 3. CFG (CLAUDE.md: CLI > env > config > default) ─────────────────────
def test_explicit_argument_beats_settings(bundle):
    gate = _gate(bundle, backend="embed", threshold=0.99, model="some-other-model")
    assert gate.backend == "embed" and gate.threshold == 0.99


def test_default_model_is_the_resident_suggestion_model(bundle):
    """The whole point of the default: no second model, no second VRAM claim."""
    assert _gate(bundle).model == settings.LOCAL_MODEL


def test_settings_expose_every_knob():
    for name in ("SALIENCE_GATE_ENABLED", "SALIENCE_BACKEND", "SALIENCE_MODEL",
                 "SALIENCE_THRESHOLD", "SALIENCE_EMBED_MODEL",
                 "SALIENCE_TIMEOUT_SECONDS", "SALIENCE_FAIL_OPEN"):
        assert hasattr(settings, name), f"{name} must be configurable (CFG)"


# ── 4. Topic corpus (used by the rejected embed backend, kept reproducible) ──
def test_bundle_topics_cover_plan_and_star(bundle):
    topics = salience.bundle_topics(bundle)
    ids = {t for t, _ in topics}
    assert len(topics) == len(bundle.plan) + len(bundle.answer_bank)
    assert any(t.startswith("plan:") for t in ids)
    assert any(t.startswith("star:") for t in ids)
    assert all(text.strip() for _, text in topics)


def test_cosine_is_a_cosine():
    assert salience.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert salience.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert salience.cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


# NOTE: the in-sample salience fixture (a real interview transcript) is private and not shipped in
# this public repo, so the regression test that pinned its shape lives only in the internal project.


# ── 6. The ambient loop actually consults the gate (D12 + D23) ───────────
def test_run_ambient_drops_a_non_salient_question(bundle, tmp_path, monkeypatch):
    """The gate sits between the D20 trigger and the suggestion call — proven end to end
    through `run_ambient`, not by inspecting the branch."""
    transcript = tmp_path / "live_transcript_20260101_000000.txt"
    transcript.write_text(
        "# header\n"
        "[00:01-00:05] them (pl): Czy dobrze mnie słychać na tym połączeniu?\n"
        "[00:06-00:20] them (pl): Jakie są Pana oczekiwania względem nowego miejsca pracy?\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "SUGGESTION_COOLDOWN_SECONDS", 0.0)
    calls: list[str] = []
    gate = _gate(bundle, backend="llm")
    gate._ask_llm = lambda turn: "oczekiwania" in turn

    class _Runner:
        busy = False
        def start(self, **kwargs): calls.append(str(kwargs["segment"]))
        def drain(self, timeout): pass

    original = reasoning.SuggestionRunner
    reasoning.SuggestionRunner = _Runner
    try:
        fired = reasoning.run_ambient(
            bundle, transcript, backend="local", model="stub",
            stop_after_idle=0.0, gate=gate,
        )
    finally:
        reasoning.SuggestionRunner = original
    assert fired == 1, "the audio check must be dropped and the real question kept"
    assert len(calls) == 1 and "oczekiwania" in calls[0]


def test_run_ambient_without_the_gate_fires_both(bundle, tmp_path, monkeypatch):
    """The before number in the same shape as the after number."""
    transcript = tmp_path / "live_transcript_20260101_000001.txt"
    transcript.write_text(
        "[00:01-00:05] them (pl): Czy dobrze mnie słychać na tym połączeniu?\n"
        "[00:06-00:20] them (pl): Jakie są Pana oczekiwania względem nowego miejsca pracy?\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "SUGGESTION_COOLDOWN_SECONDS", 0.0)

    class _Runner:
        busy = False
        def start(self, **kwargs): pass
        def drain(self, timeout): pass

    original = reasoning.SuggestionRunner
    reasoning.SuggestionRunner = _Runner
    try:
        fired = reasoning.run_ambient(
            bundle, transcript, backend="local", model="stub", stop_after_idle=0.0,
            gate=_gate(bundle, enabled=False),
        )
    finally:
        reasoning.SuggestionRunner = original
    assert fired == 2


# ── 7. LIVE — a real judgement from the resident model ───────────────────
def _ollama_up() -> bool:
    try:
        smoke_test(backend="local")
        return True
    except BackendUnavailable:
        return False


live = pytest.mark.skipif(not _ollama_up(), reason="Ollama not reachable — start it and re-run")


@live
def test_live_gate_keeps_a_substantive_question(bundle):
    gate = _gate(bundle, backend="llm")
    verdict = gate.evaluate(
        "Dobra, a jeśli chodzi o to, co mógłby Pan wnieść ze swojej strony do zespołu?"
    )
    assert verdict.fire, "recall is the clause that must not rot — this one is in the plan"


@live
def test_live_gate_drops_an_audio_check(bundle):
    gate = _gate(bundle, backend="llm")
    verdict = gate.evaluate("Dzień dobry. Czy dobrze mnie słychać? Czy dobrze mnie widać?")
    assert not verdict.fire


@live
def test_live_gate_is_cheap_relative_to_the_suggestion_it_protects(bundle):
    """Measured 157 ms median against a 5.86 s suggestion. A gate that costs a meaningful
    fraction of the call it saves is not a saving."""
    gate = _gate(bundle, backend="llm")
    gate.warm()
    verdict = gate.evaluate("Jakie są Pana oczekiwania względem nowego miejsca pracy?")
    assert verdict.latency_ms < 1500, f"gate took {verdict.latency_ms:.0f} ms"
