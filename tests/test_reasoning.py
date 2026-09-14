"""Tests for the reasoning layer (#321).

Split deliberately: everything above `test_live_*` is deterministic and runs with no GPU,
no server and no tokens. The live tests at the bottom make a REAL model call, because a
reasoning layer that has only ever been mocked is exactly the class of defect this project
has already shipped three times. They skip (loudly) when Ollama is not reachable.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from scripts import reasoning, salience  # noqa: E402
from scripts.llm_client import (  # noqa: E402
    BackendUnavailable, announce_backend, prompt_was_truncated, smoke_test,
)

FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "question_fixtures.json"
EXAMPLE_SESSION = "example_ai_engineer"
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "sessions"


# ── 1. Context bundle (D13) ──────────────────────────────────────────────
def test_real_bundle_on_disk_loads():
    """Done-when clause 1: at least one real bundle exists and loads without error."""
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    assert bundle.session_id == EXAMPLE_SESSION
    assert bundle.resume, "the resume section must carry real content"
    assert len(bundle.resume) > 50
    assert len(bundle.answer_bank) >= 5
    assert len(bundle.plan) >= 3
    assert bundle.spoken_language == "pl"
    assert bundle.suggestion_language == "match"  # answer each question in its own language


def test_placeholder_sections_are_reported_not_swallowed(tmp_path):
    """A fixture must never pass as real content — that is how this project shipped
    an English-only model 'proven' for Polish. Uses a synthetic bundle: the real one has
    no placeholders since the example bundle ships complete."""
    session = tmp_path / "s1"
    session.mkdir()
    (session / "bundle.json").write_text(
        json.dumps({
            "schema_version": 2,
            "job_description": {"text": "tbd", "status": "placeholder"},
            "resume": {"text": "real content here", "status": "real"},
        }),
        encoding="utf-8",
    )
    bundle = reasoning.load_bundle("s1", sessions_dir=tmp_path)
    assert "job_description" in bundle.placeholders
    assert "resume" not in bundle.placeholders
    assert any("PLACEHOLDER" in line for line in bundle.warn_lines())


def test_the_real_bundle_has_no_placeholders_left():
    """The example bundle: every section is real content, no placeholders."""
    assert reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR).placeholders == []


def test_plan_steps_carry_key_points_and_done_signals():
    plan = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR).plan
    assert all(step.id and step.title for step in plan)
    assert any(step.key_points for step in plan)
    assert any(step.done_signals for step in plan), "the #322 tracker needs these"


def test_star_entries_render_as_prompt_blocks():
    entry = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR).answer_bank[0]
    block = entry.as_prompt_block()
    assert entry.id in block and "S:" in block and "R:" in block


def test_missing_bundle_names_the_available_sessions(tmp_path):
    with pytest.raises(FileNotFoundError, match="Available sessions"):
        reasoning.load_bundle("does_not_exist", sessions_dir=tmp_path)


def test_wrong_schema_version_is_a_hard_stop(tmp_path):
    session = tmp_path / "s1"
    session.mkdir()
    (session / "bundle.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        reasoning.load_bundle("s1", sessions_dir=tmp_path)


def test_missing_referenced_file_is_a_hard_stop(tmp_path):
    session = tmp_path / "s1"
    session.mkdir()
    (session / "bundle.json").write_text(
        json.dumps({"schema_version": 1, "resume": {"file": "nope.md"}}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="nope.md"):
        reasoning.load_bundle("s1", sessions_dir=tmp_path)


# ── 2. Trigger policy (G6) ───────────────────────────────────────────────
@pytest.mark.parametrize(
    "text",
    [
        "Jakie ma Pan doświadczenie z LangChain?",
        "Opowiedz o projekcie, z którego jesteś najbardziej dumny",
        "Proszę wyjaśnić, czym różni się RAG od fine-tuningu",
        "Tell me about a challenging machine learning project you have worked on",
        "Do you have any experience with vector databases?",
    ],
)
def test_questions_fire(text):
    assert reasoning.looks_like_question(text)


@pytest.mark.parametrize(
    "text",
    [
        "Jak Pan widzi, to działa dość szybko nawet na dużych zbiorach",  # "as you can see"
        "Co ciekawe, ten problem pojawił się dopiero na produkcji",       # "interestingly"
        "Powiedzmy, że to był dla nas dość trudny moment",                # "let's say"
        "Now I would like to move on to the technical part",              # aux, not inverted
        "Mhm, jasne",                                                     # below the word floor
    ],
)
def test_statements_do_not_fire(text):
    assert not reasoning.looks_like_question(text)


def test_heuristic_scores_on_the_labelled_fixture_set():
    """Guards the measured number in config/settings.py. If a rule change drops this,
    the comment there is now a lie — fix one or the other."""
    cases = json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]
    tp = sum(1 for c in cases if c["is_question"] and reasoning.looks_like_question(c["text"]))
    fp = sum(1 for c in cases if not c["is_question"] and reasoning.looks_like_question(c["text"]))
    fn = sum(1 for c in cases if c["is_question"] and not reasoning.looks_like_question(c["text"]))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    assert precision >= 0.95, f"precision regressed to {precision:.2f}"
    assert recall >= 0.95, f"recall regressed to {recall:.2f}"


# ── 3. The prompt asymmetry (G8) ─────────────────────────────────────────
def test_prompt_states_the_asymmetry_explicitly():
    """The asymmetry has to be IN the prompt: without it the model answers in the input
    language. This asserts the instruction exists; the live test asserts it works. Forced en
    (the bundle now defaults to 'match', which would answer a Polish question in Polish)."""
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    system, user = reasoning.build_messages(bundle, "Jakie ma Pan doświadczenie?", suggestion_language="en")
    assert "You will therefore receive Polish text" in system
    assert "You MUST write your entire answer in English, never in Polish" in system
    assert "Polish transcript" in user


def test_prompt_flips_with_the_suggestion_language():
    """pl->pl must NOT reuse the cross-lingual wording: substituting the same language into
    it produced 'write your entire answer in Polish, never in Polish'. Regression guard."""
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    system, _ = reasoning.build_messages(bundle, "Jakie ma Pan doświadczenie?", suggestion_language="pl")
    assert "you write in Polish too" in system
    assert "never in Polish" not in system
    assert "even though the input is" not in system


def test_match_mode_answers_a_polish_question_in_polish(monkeypatch):
    """'match' (the new default) answers in the question's own language. A Polish question,
    with the per-segment detection saying pl, must produce the same-language (pl->pl) prompt."""
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    assert bundle.suggestion_language == "match"
    system, _ = reasoning.build_messages(
        bundle, "Jakie ma Pan doświadczenie z LangChain?", spoken_language="pl")
    assert "you write in Polish too" in system      # SAME_LANGUAGE_RULE, Polish
    assert "never in Polish" not in system           # NOT the cross-lingual rule


def test_match_mode_answers_an_english_question_in_english():
    """The same bundle, an English question (detection = en): same-language en->en."""
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    system, _ = reasoning.build_messages(
        bundle, "Can you walk me through your RAG pipeline?", spoken_language="en")
    assert "you write in English too" in system
    assert "never in English" not in system


def test_match_mode_falls_back_to_text_detection_when_no_tag():
    """--text has no detection tag, so match detects from the (clean) text itself."""
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    pl_system, _ = reasoning.build_messages(bundle, "Czy używał Pan wektorowych baz danych?")
    en_system, _ = reasoning.build_messages(bundle, "How did you evaluate answer quality?")
    assert "you write in Polish too" in pl_system
    assert "you write in English too" in en_system


def test_detect_text_language_biases_to_polish():
    assert reasoning.detect_text_language("Czy używał Pan LoRA?") == "pl"      # diacritic
    assert reasoning.detect_text_language("Jak wygladal wybor bazy?") == "pl"  # markers, no diacritics
    assert reasoning.detect_text_language("Tell me about your pipeline") == "en"


def test_honesty_boundary_reaches_the_prompt_as_a_hard_rule():
    """The highest-consequence block: a copilot that suggests claiming Qdrant is worse than
    no copilot. Asserts the rows are present AND framed as a prohibition."""
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    assert len(bundle.honesty_boundary) >= 10
    system, _ = reasoning.build_messages(bundle, "test")
    assert "HONESTY BOUNDARY" in system
    assert "NEVER claim" in system
    assert "Qdrant" in system and "MLflow" in system
    assert "degree NOT completed" in system


def test_prompt_carries_the_bundle():
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    system, _ = reasoning.build_messages(bundle, "test")
    assert "Meridian Analytics" in system      # resume
    assert "star_report_automation" in system   # answer bank
    assert "Your questions" in system           # plan


# ── 4. The consumption seam (D19) ────────────────────────────────────────
def test_parse_transcript_line():
    # Untagged line (pre-#326 transcript / monitor-only run): speaker and language are None.
    assert reasoning.parse_transcript_line("[00:07-00:36] Dzień dobry") == ("00:07-00:36", None, None, "Dzień dobry")
    assert reasoning.parse_transcript_line("# run_id : abc") is None
    assert reasoning.parse_transcript_line("\n") is None
    assert reasoning.parse_transcript_line("no timestamp here") is None


def test_parse_transcript_line_extracts_the_speaker_tag():
    """G9/#326: live_transcribe tags each line with the dominant channel (no language tag)."""
    assert reasoning.parse_transcript_line("[02:11-02:19] them: Proszę opowiedzieć o TEL") == (
        "02:11-02:19", "them", None, "Proszę opowiedzieć o TEL")
    assert reasoning.parse_transcript_line("[02:38-02:43] you: Tak, dobrze") == (
        "02:38-02:43", "you", None, "Tak, dobrze")
    # A line whose text itself starts with a colon-word must NOT be mistaken for a tag.
    assert reasoning.parse_transcript_line("[00:00-00:04] jasne: rozumiem") == (
        "00:00-00:04", None, None, "jasne: rozumiem")


def test_parse_transcript_line_extracts_the_language_tag():
    """Per-question language: `them (pl):` / `you (en):`."""
    assert reasoning.parse_transcript_line("[02:11-02:19] them (pl): Jak podszedł Pan do RAG?") == (
        "02:11-02:19", "them", "pl", "Jak podszedł Pan do RAG?")
    assert reasoning.parse_transcript_line("[02:20-02:25] them (en): Walk me through it") == (
        "02:20-02:25", "them", "en", "Walk me through it")


def test_follow_transcript_picks_up_lines_appended_after_it_attached(tmp_path):
    """The seam's whole point: attach to a file that is still being written."""
    path = tmp_path / "live_transcript_20260101_000000.txt"
    path.write_text("# header\n[00:00-00:05] pierwsza linia\n", encoding="utf-8")

    def append_later():
        time.sleep(0.3)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("[00:05-00:11] druga linia\n")
            fh.flush()

    threading.Thread(target=append_later, daemon=True).start()
    lines = list(reasoning.follow_transcript(path, stop_after_idle=1.5, poll=0.05))
    assert [text for *_, text in lines] == ["pierwsza linia", "druga linia"]


class _FakeRunner:
    """Captures the segments run_ambient would have sent to the model — no GPU, no tokens."""

    def __init__(self) -> None:
        self.segments: list[str] = []
        self.histories: list[list[str]] = []
        self.spoken_languages: list[object] = []
        self.backends: list[object] = []

    busy = False

    def start(self, **kwargs: object) -> None:
        self.segments.append(str(kwargs["segment"]))
        self.histories.append(list(kwargs.get("history") or []))
        self.spoken_languages.append(kwargs.get("spoken_language"))
        self.backends.append(kwargs.get("backend"))

    def drain(self, timeout: float) -> None:
        pass


def _run_ambient_capture(tmp_path, lines: list[str], monkeypatch, **kwargs) -> _FakeRunner:
    path = tmp_path / "live_transcript_20260101_000000.txt"
    path.write_text("# header\n" + "".join(f"{ln}\n" for ln in lines), encoding="utf-8")
    fake = _FakeRunner()
    # `lambda **_kw: fake` (not `lambda: fake`) so a test that passes a `sink` still works —
    # run_ambient then constructs the runner as SuggestionRunner(on_event=sink).
    monkeypatch.setattr(reasoning, "SuggestionRunner", lambda **_kw: fake)
    # Replay compresses time, so consecutive lines fall inside the wall-clock cooldown
    # (documented gotcha). Zero it so this test isolates the speaker filter.
    monkeypatch.setattr(settings, "SUGGESTION_COOLDOWN_SECONDS", 0.0)
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    # Same reason, for the D23 salience gate (#386): it is ON by default and would otherwise
    # make these speaker-filter assertions depend on a live model judgement. The gate has its
    # own tests in test_salience.py.
    kwargs.setdefault("gate", salience.SalienceGate(bundle, enabled=False))
    reasoning.run_ambient(
        bundle, path, backend="local", model="x", stop_after_idle=0.2,
        questions_only=True, **kwargs,
    )
    return fake


def test_run_ambient_answers_only_the_interviewer_by_default(tmp_path, monkeypatch):
    """G9/#326: a candidate's own question (mic channel) must NOT trigger a suggestion."""
    fake = _run_ambient_capture(
        tmp_path,
        [
            "[00:00-00:06] them: Jakie ma Pan doświadczenie z LangChain?",  # interviewer
            "[00:07-00:12] you: Czy dobrze rozumiem to pytanie?",           # candidate's own
        ],
        monkeypatch,
    )
    assert fake.segments == ["Jakie ma Pan doświadczenie z LangChain?"]


def test_run_ambient_any_speaker_restores_channel_blind_firing(tmp_path, monkeypatch):
    fake = _run_ambient_capture(
        tmp_path,
        [
            "[00:00-00:06] them: Jakie ma Pan doświadczenie z LangChain?",
            "[00:07-00:13] you: Czy dobrze rozumiem to pytanie?",
        ],
        monkeypatch,
        answer_speaker="any",
    )
    assert len(fake.segments) == 2


def test_run_ambient_untagged_line_is_treated_as_interviewer(tmp_path, monkeypatch):
    """A pre-#326 / monitor-only transcript has no tag; it must still fire (never silently drop)."""
    fake = _run_ambient_capture(
        tmp_path,
        ["[00:00-00:06] Jakie ma Pan doświadczenie z LangChain?"],
        monkeypatch,
    )
    assert fake.segments == ["Jakie ma Pan doświadczenie z LangChain?"]


def test_history_is_threaded_with_speaker_labels(tmp_path, monkeypatch):
    """G9/#326: prior turns reach the model labelled Interviewer:/You:, and a skipped
    candidate turn is still present as context (not answered, but not lost either)."""
    fake = _run_ambient_capture(
        tmp_path,
        [
            "[00:00-00:06] them: Jakie ma Pan doświadczenie z LangChain?",
            "[00:07-00:11] you: Głównie w projektach RAG.",
            "[00:12-00:18] them: A jak mierzył Pan jakość odpowiedzi?",
        ],
        monkeypatch,
    )
    # Two interviewer questions fired; the candidate turn was context-only.
    assert fake.segments == [
        "Jakie ma Pan doświadczenie z LangChain?",
        "A jak mierzył Pan jakość odpowiedzi?",
    ]
    # The second suggestion's history carries both prior turns, each labelled by speaker.
    history_for_second = fake.histories[1]
    assert "Interviewer: Jakie ma Pan doświadczenie z LangChain?" in history_for_second
    assert "You: Głównie w projektach RAG." in history_for_second


def test_run_ambient_threads_per_question_language(tmp_path, monkeypatch):
    """The detected language on each transcript line reaches suggest() as spoken_language,
    so match mode can answer each question in the language it was asked in."""
    fake = _run_ambient_capture(
        tmp_path,
        [
            "[00:00-00:06] them (pl): Jak podszedł Pan do RAG?",
            "[00:07-00:13] them (en): Walk me through your evaluation setup?",
        ],
        monkeypatch,
    )
    assert fake.segments == ["Jak podszedł Pan do RAG?", "Walk me through your evaluation setup?"]
    assert fake.spoken_languages == ["pl", "en"]


# ── run_ambient live controls (#607 single-app switches) ─────────────────────
def test_run_ambient_suggestions_off_renders_lines_but_fires_nothing(tmp_path, monkeypatch):
    """The transcription/meter half must survive the suggestions switch: every line still
    reaches the sink, but with suggestions OFF nothing is sent to a model (no GPU, no egress)."""
    lines_out: list[str] = []
    controls = reasoning.Controls(suggestions_on=False)
    fake = _run_ambient_capture(
        tmp_path,
        [
            "[00:00-00:06] them (pl): Jak podszedł Pan do RAG?",
            "[00:07-00:13] them (pl): Jakie ma Pan doświadczenie z LangChain?",
        ],
        monkeypatch,
        controls=controls,
        sink=lambda ev: lines_out.append(ev["text"]) if ev.get("kind") == "line" else None,
    )
    assert fake.segments == []                                   # no question fired
    assert lines_out == ["Jak podszedł Pan do RAG?",
                         "Jakie ma Pan doświadczenie z LangChain?"]  # both still rendered


def test_run_ambient_reads_suggestions_flag_per_line_not_once(tmp_path, monkeypatch):
    """A toggle takes effect on the NEXT question — the loop reads Controls live, not at entry.
    Start ON, flip OFF the instant the first question fires, and the second must be skipped."""
    controls = reasoning.Controls(suggestions_on=True)

    class _Runner(_FakeRunner):
        def start(self, **kwargs):
            super().start(**kwargs)
            controls.set_suggestions(False)         # flip OFF right after line 1 fires

    fake = _Runner()
    monkeypatch.setattr(reasoning, "SuggestionRunner", lambda: fake)
    monkeypatch.setattr(settings, "SUGGESTION_COOLDOWN_SECONDS", 0.0)
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    path = tmp_path / "live_transcript_20260101_000000.txt"
    path.write_text("# header\n"
                    "[00:00-00:06] them (pl): Jak podszedł Pan do RAG?\n"
                    "[00:07-00:13] them (pl): Jakie ma Pan doświadczenie z LangChain?\n",
                    encoding="utf-8")
    reasoning.run_ambient(bundle, path, backend="local", model="x", stop_after_idle=0.2,
                          questions_only=True, gate=salience.SalienceGate(bundle, enabled=False),
                          controls=controls)
    assert fake.segments == ["Jak podszedł Pan do RAG?"]         # line 1 fired, line 2 skipped (flipped off)


def test_run_ambient_reads_the_backend_from_controls_at_fire_time(tmp_path, monkeypatch):
    """The 'local + api' switch: run_ambient uses controls.backend, not its fixed arg."""
    controls = reasoning.Controls(suggestions_on=True, backend="cloud", model="claude-x")
    fake = _run_ambient_capture(
        tmp_path,
        ["[00:00-00:06] them (pl): Jak podszedł Pan do RAG?"],
        monkeypatch,
        controls=controls,
    )
    assert fake.segments == ["Jak podszedł Pan do RAG?"]
    assert fake.backends == ["cloud"]                            # the live switch, not backend="local"


def test_follow_transcript_stop_event_ends_the_tail(tmp_path):
    """#607: a set stop_event ends the tail cleanly so the recorder can rotate to a new file."""
    path = tmp_path / "live_transcript_20260101_000000.txt"
    path.write_text("# header\n[00:00-00:05] jedna\n", encoding="utf-8")
    stop = threading.Event()
    seen: list[str] = []

    def consume():
        for *_, text in reasoning.follow_transcript(path, poll=0.05, stop_event=stop):
            seen.append(text)

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    time.sleep(0.2)
    stop.set()
    t.join(timeout=2.0)
    assert not t.is_alive()                                      # the generator returned
    assert seen == ["jedna"]                                     # the already-written line still came through


def test_newest_transcript_prefers_the_latest_stamp(tmp_path):
    for stamp in ("20260101_000000", "20260901_120000", "20260501_000000"):
        (tmp_path / f"live_transcript_{stamp}.txt").write_text("#\n", encoding="utf-8")
    assert reasoning.newest_transcript(tmp_path).name == "live_transcript_20260901_120000.txt"


# ── 5. SI1 — egress is visible ───────────────────────────────────────────
def test_local_backend_is_the_default():
    assert settings.REASONING_BACKEND == "local"


def test_cloud_banner_names_the_destination(monkeypatch):
    monkeypatch.setenv("CLOUD_BASE_URL", "https://api.example.com/v1")
    banner = announce_backend("cloud", "some-model")
    assert "EGRESS" in banner and "api.example.com" in banner
    assert "No egress" in announce_backend("local", "llama3.1:8b")


def test_cloud_backend_refuses_without_byok_config(monkeypatch):
    """Must name the ACTUAL missing variable. This previously asserted on CLOUD_BASE_URL and
    passed by matching a stale wrapper message ("Check CLOUD_BASE_URL...") that was appended
    to every cloud failure — the real cause (CLOUD_MODEL unset) was never checked."""
    for var in ("CLOUD_BASE_URL", "CLOUD_API_KEY", "CLOUD_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(BackendUnavailable, match="CLOUD_MODEL is not set"):
        reasoning.suggest(reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR), "test", backend="cloud")


def test_cloud_config_error_is_not_reported_as_a_call_failure(monkeypatch):
    """A missing key must not be dressed up as a network problem."""
    monkeypatch.setenv("CLOUD_MODEL", "claude-opus-4-8")
    monkeypatch.delenv("CLOUD_API_KEY", raising=False)
    with pytest.raises(BackendUnavailable) as excinfo:
        reasoning.suggest(reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR), "test", backend="cloud")
    assert "CLOUD_API_KEY is not set" in str(excinfo.value)
    assert "network reachability" not in str(excinfo.value)


# ── 5b. Truncation guard vs prompt caching ───────────────────────────────
def test_truncation_guard_counts_cached_tokens_too():
    """With caching on, `input_tokens` is only the uncached remainder. Dividing by it alone
    made the guard fire on every cached cloud call (39k chars / 77 tokens) and report a
    complete prompt as truncated. The cached prefix must count toward the total."""
    cached = {"input_tokens": 77, "cache_read_tokens": 14019, "cache_write_tokens": 0}
    assert not prompt_was_truncated(39_000, cached)
    fresh = {"input_tokens": 14096, "cache_read_tokens": 0, "cache_write_tokens": 0}
    assert not prompt_was_truncated(39_000, fresh)


def test_truncation_guard_still_catches_a_real_truncation():
    """The Ollama failure it exists for: 37.8k chars reported as 2050 tokens."""
    assert prompt_was_truncated(37_800, {"input_tokens": 2050})


def test_truncation_guard_is_silent_without_usage():
    """No usage block means no evidence — never claim truncation on a guess."""
    assert not prompt_was_truncated(39_000, {})


# ── 6. LIVE — a real model call (Done-when clause 2 + 3) ─────────────────
def _ollama_up() -> bool:
    try:
        smoke_test(backend="local")
        return True
    except BackendUnavailable:
        return False


live = pytest.mark.skipif(not _ollama_up(), reason="Ollama not reachable — start it and re-run")


@live
def test_live_llm_seam_returns_text():
    """Clause 2: the seam actually returns text, called from a test."""
    reply = smoke_test(backend="local")
    assert reply.text, "the backend returned an empty completion"
    assert reply.latency_seconds > 0
    assert reply.total_tokens > 0, "no usage reported — the cost line would be a fiction"


@live
def test_live_polish_in_english_out():
    """Clause 3, direction 1: a Polish segment must come back in English."""
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    result = reasoning.suggest(
        bundle, "Jakie ma Pan doświadczenie z integracją modeli językowych w produkcji?",
        suggestion_language="en", backend="local",
    )
    assert result.text
    assert not _looks_polish(result.text), f"expected English, got:\n{result.text}"


@live
def test_live_polish_in_polish_out():
    """Clause 3, direction 2: the same input with the switch flipped must come back in
    Polish. This is the direction that broke: before the language_rule fix, formatting the
    cross-lingual wording with spoken == target told the model to write Polish "never in
    Polish", and llama3.2:3b responded by abandoning the question and translating the plan
    block (3/3 reps). See config/settings.py for the corrected model comparison."""
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    result = reasoning.suggest(
        bundle, "Jakie ma Pan doświadczenie z integracją modeli językowych w produkcji?",
        suggestion_language="pl", backend="local",
    )
    assert result.text
    assert _looks_polish(result.text), f"expected Polish, got:\n{result.text}"


@live
def test_live_honesty_boundary_holds_on_a_trap_question():
    """Asked in Polish whether he has used Qdrant, the suggestion must honour the honesty
    boundary — the example bundle's truth is 'Qdrant in a personal project only; production was
    pgvector', so the answer must draw that limit, not claim production Qdrant experience. This is
    the one regression that would actually cost the user a job."""
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    result = reasoning.suggest(
        bundle, "Czy pracował Pan z bazami wektorowymi, na przykład Qdrant?",
        suggestion_language="en", backend="local",
    )
    lowered = result.text.lower()
    assert "qdrant" in lowered
    # The honest boundary is drawn either by denying production use or by naming the real limit
    # (a personal project / pgvector in production). Any of these means the claim was not inflated.
    honest_markers = (
        "haven't run", "have not run", "haven't used", "have not used", "never used",
        "personal project", "not in production", "pgvector",
    )
    assert any(p in lowered for p in honest_markers), (
        f"the suggestion did not honour the Qdrant honesty boundary:\n{result.text}"
    )


@live
def test_live_prompt_is_not_silently_truncated():
    """The real bundle is ~8.7k prompt tokens against a stock 4096 window. Ollama truncates
    silently rather than erroring, so this asserts the whole prompt actually arrived."""
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    system, user = reasoning.build_messages(bundle, "test segment")
    result = reasoning.suggest(bundle, "test segment", backend="local")
    ratio = (len(system) + len(user)) / result.reply.prompt_tokens
    assert ratio < 7.0, (
        f"{ratio:.1f} chars/token means the prompt was truncated — recreate the "
        f"interview-copilot:8b model with PARAMETER num_ctx 16384"
    )


@live
def test_live_streaming_delivers_first_token_fast():
    """The whole latency design: the POINT line must land in well under a second even though
    the full answer takes seconds. Guards both streaming and the fast local path."""
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
    seen: list[str] = []
    result = reasoning.suggest(
        bundle, "Opowiedz mi o najtrudniejszym projekcie.", backend="local",
        on_token=seen.append,
    )
    assert seen, "nothing was streamed — on_token never fired"
    assert result.text == "".join(seen).strip()
    assert result.reply.first_token_seconds is not None
    assert result.reply.first_token_seconds < 2.0, (
        f"first token took {result.reply.first_token_seconds:.2f}s"
    )


@live
def test_live_a_superseded_suggestion_is_not_a_failure():
    """An interrupted suggestion is a normal event. It must raise Superseded out of
    llm_call untouched — wrapping it in BackendUnavailable printed a traceback and the
    word 'failed' on every interruption during a real call."""
    bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)

    def cancel_after_first(_piece: str) -> None:
        raise reasoning.Superseded

    with pytest.raises(reasoning.Superseded):
        reasoning.suggest(bundle, "Opowiedz mi o projekcie.", backend="local",
                          on_token=cancel_after_first)


def _looks_polish(text: str) -> bool:
    """Cheap language check: Polish-only diacritics, or common Polish function words.
    Not a language identifier — enough to tell two languages this different apart."""
    lowered = text.lower()
    return any(ch in lowered for ch in "łęążćńśź") or any(
        f" {w} " in f" {lowered} " for w in ("jest", "oraz", "które", "nie", "przez")
    )


# ==========================================================================
# The D25 boundary (#399): the provisional never reaches the suggestion path
# ==========================================================================
class TestTheProvisionalNeverReachesReasoning:
    """A DECIDED boundary, not an oversight — so it is asserted, not left to inspection.

    D25 puts provisional text in `live_transcript_*.partial`; the suggestion path reads only
    the final `live_transcript_*.txt`. P5 option D (suggest from a prefix) was measured and
    rejected — at a 5 s prefix the D20 trigger fires on only 9/24 — and D22's reasoning about
    unauditable suggestions applies: the user can check a provisional TRANSCRIPT line against
    what they just heard in the room, and cannot check a suggestion built on one.
    """

    def _both_files(self, tmp_path):
        txt = tmp_path / "live_transcript_20260905_120000.txt"
        txt.write_text("# interview_copilot live transcript\n"
                       "[02:26-02:34] them (pl): Czy może pan opisać jakąś sytuację?\n",
                       encoding="utf-8")
        partial = tmp_path / "live_transcript_20260905_120000.partial"
        partial.write_text("# PROVISIONAL lines (D25) — this is NOT the transcript.\n"
                           "[02:26-02:31] them (pl): PROVISIONAL_MARKER czy może pan\n",
                           encoding="utf-8")
        return txt, partial

    def test_transcript_discovery_cannot_pick_up_a_partial(self, tmp_path):
        txt, partial = self._both_files(tmp_path)
        assert reasoning.newest_transcript(tmp_path) == txt

    def test_a_partial_alone_looks_like_no_transcript_at_all(self, tmp_path):
        """The failure this rules out is the worst one available: a run where the recorder's
        `.txt` is missing and the suggestion loop silently answers provisional text instead."""
        _txt, partial = self._both_files(tmp_path)
        _txt.unlink()
        assert partial.exists()
        assert reasoning.newest_transcript(tmp_path) is None

    def test_the_ambient_loop_reads_the_final_line_and_only_the_final_line(self, tmp_path):
        """Drive `run_ambient` over the `.txt` with the `.partial` sitting right beside it and
        assert on what the loop actually saw — the sink is the loop's own account of it."""
        txt, _partial = self._both_files(tmp_path)
        bundle = reasoning.load_bundle(EXAMPLE_SESSION, sessions_dir=EXAMPLES_DIR)
        seen: list[dict] = []

        class NeverFires:
            active = False

            def describe(self) -> str:
                return "off"

            def warm(self) -> None:
                return None

            def evaluate(self, _text: str) -> salience.SalienceVerdict:
                return salience.SalienceVerdict(False, "test-double", "none")

        reasoning.run_ambient(
            bundle=bundle, path=txt, backend="local", model="",
            stop_after_idle=0.4, sink=seen.append, gate=NeverFires(),
        )
        texts = [e.get("text", "") for e in seen if e.get("kind") == "line"]
        assert texts == ["Czy może pan opisać jakąś sytuację?"]
        assert not any("PROVISIONAL_MARKER" in json.dumps(e) for e in seen)

    def test_reasoning_names_no_partial_file_anywhere(self):
        """Structural backstop: the suggestion module has no business knowing the D25 file
        exists. If a future edit teaches it, this fails before a live call finds out."""
        source = (PROJECT_ROOT / "scripts" / "reasoning.py").read_text(encoding="utf-8")
        assert ".partial" not in source
