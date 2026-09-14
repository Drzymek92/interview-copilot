"""The reasoning layer (D13 + D14): transcript segment in, answer scaffold out.

Subtask #321. Three pieces, each independently testable:

1. **Context bundle (D13)** — `load_bundle()` reads
   `scripts/inputs/sessions/<session_id>/bundle.json` plus the markdown files it names
   (JD, company brief, resume, STAR/answer bank, step-wise interview plan) into a
   `ContextBundle`. "Never start cold": this is what primes the system prompt.

2. **Trigger policy (G6)** — `looks_like_question()` is a *deterministic* pl/en detector
   (CLAUDE.md Determinism First). It costs no tokens and cannot hallucinate; an LLM
   classifier per segment is the expensive option and is not the default. Fixtures +
   measured precision/recall live in `tests/fixtures/question_fixtures.json`.

3. **The suggestion (D14)** — `suggest()` builds a cross-lingual prompt and calls
   `llm_client.llm_call`. **The language asymmetry is stated in the prompt, not assumed:**
   the transcript arrives in Polish (`STT_LANGUAGE=pl`) while the scaffold is written in
   `SUGGESTION_LANGUAGE` (default `en`, switchable at session start). Without saying so
   explicitly the model answers in the input language — measured, not feared.

4. **The salience gate (D23)** — `scripts/salience.py` sits between the trigger and the
   suggestion: D20 answers "is this a question", the gate answers "is it worth a call".
   On the real HR screen that is 24 fired questions down to 10. It is deliberately a
   separate module, and deliberately *after* D20, so the free rule still does the first cut.

**The consumption seam (D19)** — this module *tails the transcript file* that
`live_transcribe.py` already fsyncs line-by-line; it does not import that script's loop and
requires no edit to it. `follow_transcript()` is the one seam #322's dashboard consumes too.

Run it:
    python scripts/reasoning.py --session example_ai_engineer --text "<a Polish question>"
    python scripts/reasoning.py --session example_ai_engineer --watch        # follow the newest run
    python scripts/reasoning.py --session example_ai_engineer --replay FILE  # a finished transcript
    python scripts/reasoning.py --selftest-heuristic                 # no model, no GPU
    python scripts/reasoning.py --selftest-salience                  # score the D23 gate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from scripts.llm_client import (  # noqa: E402
    BackendUnavailable, LlmReply, StreamCancelled, announce_backend, llm_call, model_for, resolve_backend,
)
from scripts.logger import get_logger  # noqa: E402
from scripts.salience import SalienceGate  # noqa: E402

logger = get_logger("reasoning")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "outputs"
SESSIONS_DIR = PROJECT_ROOT / "scripts" / "inputs" / settings.SESSIONS_DIRNAME
# v2 adds the optional `honesty_boundary` block (see HonestyClaim). v1 bundles still load —
# they simply carry no boundary, and the prompt then says so rather than staying silent.
BUNDLE_SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)

LANGUAGE_NAMES = {"en": "English", "pl": "Polish"}


# --------------------------------------------------------------------------
# 1. Context bundle (D13)
# --------------------------------------------------------------------------
@dataclass
class PlanStep:
    """One step of the tracked interview plan (D13). P3 will extend this — #321 only
    consumes `title` + `key_points` as orientation for the model; `done_signals` is
    carried through so the #322 tracker does not need a schema change to start using it."""

    id: str
    title: str
    key_points: list[str] = field(default_factory=list)
    done_signals: list[str] = field(default_factory=list)


@dataclass
class StarEntry:
    """One prepared answer. Situation/Task/Action/Result kept as separate fields so a
    future retriever can match on `tags` without re-parsing prose."""

    id: str
    title: str
    tags: list[str] = field(default_factory=list)
    situation: str = ""
    task: str = ""
    action: str = ""
    result: str = ""

    def as_prompt_block(self) -> str:
        parts = [f"- [{self.id}] {self.title}"]
        for label, value in (("S", self.situation), ("T", self.task), ("A", self.action), ("R", self.result)):
            if value:
                parts.append(f"    {label}: {value}")
        return "\n".join(parts)


@dataclass
class HonestyClaim:
    """One claim that must not round itself up under pressure, and what is actually true.

    This is the highest-consequence block in the bundle. A live copilot that suggests
    "tell them you've used Qdrant" when the candidate has not is worse than no copilot:
    the gap is recoverable, a claim that collapses under a follow-up question is not.
    It goes into the system prompt as a hard constraint, not as background."""

    claim: str
    truth: str

    def as_prompt_block(self) -> str:
        return f"- NEVER claim: {self.claim}\n    TRUE: {self.truth}"


@dataclass
class ContextBundle:
    """Everything the reasoning layer knows before a word is spoken."""

    session_id: str
    role: str
    company: str
    spoken_language: str
    suggestion_language: str
    job_description: str
    company_brief: str
    resume: str
    answer_bank: list[StarEntry]
    plan: list[PlanStep]
    honesty_boundary: list[HonestyClaim]
    placeholders: list[str]
    source_dir: Path

    def warn_lines(self) -> list[str]:
        """Named-as-placeholder sections. Printed at load so a fixture is never mistaken
        for the real thing — the failure mode this project has already shipped twice."""
        return [
            f"PLACEHOLDER: `{name}` is not real content yet — replace it before the interview."
            for name in self.placeholders
        ]


def _read_part(session_dir: Path, spec: object, label: str, placeholders: list[str]) -> str:
    """A bundle section is either inline text or a `{"file": ..., "status": ...}` pointer.
    `status: placeholder` is recorded rather than silently accepted."""
    if spec is None:
        placeholders.append(label)
        return ""
    if isinstance(spec, str):
        return spec.strip()
    if not isinstance(spec, dict):
        raise ValueError(f"bundle section {label!r} must be a string or an object, got {type(spec).__name__}")
    if spec.get("status") == "placeholder":
        placeholders.append(label)
    text = str(spec.get("text", "")).strip()
    filename = spec.get("file")
    if filename:
        path = session_dir / str(filename)
        if not path.is_file():
            raise FileNotFoundError(f"bundle section {label!r} points at a missing file: {path}")
        text = path.read_text(encoding="utf-8").strip()
    return text


def load_bundle(session: str, sessions_dir: Path | None = None) -> ContextBundle:
    """Load `<sessions_dir>/<session>/bundle.json`. Raises with a pointed message rather
    than half-loading: a bundle that silently lost its resume is worse than a hard stop."""
    root = sessions_dir or SESSIONS_DIR
    session_dir = Path(session) if Path(session).is_dir() else root / session
    manifest = session_dir / "bundle.json"
    # Fall back to the shipped examples/ so a bare id (e.g. `example_ai_engineer`) resolves out of
    # the box. Real bundles live under scripts/inputs/sessions/ (gitignored); the example ships
    # under examples/sessions/ so it can be committed. Only when no explicit sessions_dir was given.
    if not manifest.is_file() and sessions_dir is None:
        example_dir = PROJECT_ROOT / "examples" / "sessions" / session
        if (example_dir / "bundle.json").is_file():
            session_dir, manifest = example_dir, example_dir / "bundle.json"
    if not manifest.is_file():
        example_root = PROJECT_ROOT / "examples" / "sessions"
        available = sorted({
            p.parent.name
            for r in (root, example_root) if r.is_dir()
            for p in r.glob("*/bundle.json")
        })
        raise FileNotFoundError(
            f"no bundle at {manifest}. Available sessions: {', '.join(available) or '(none)'}"
        )
    raw = json.loads(manifest.read_text(encoding="utf-8"))

    version = int(raw.get("schema_version", 0))
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"bundle schema_version {version} not in {SUPPORTED_SCHEMA_VERSIONS} supported by this build"
        )

    placeholders: list[str] = []
    languages = raw.get("language", {})
    bundle = ContextBundle(
        session_id=str(raw.get("session_id") or session_dir.name),
        role=str(raw.get("role", "")),
        company=str(raw.get("company", "")),
        # The bundle may pin the languages for this session; env/CFG is the fallback,
        # and --suggestion-language overrides both (CLI > env > config, guides/coding_pipeline).
        spoken_language=str(languages.get("spoken") or settings.STT_LANGUAGE),
        suggestion_language=str(languages.get("suggestions") or settings.SUGGESTION_LANGUAGE),
        job_description=_read_part(session_dir, raw.get("job_description"), "job_description", placeholders),
        company_brief=_read_part(session_dir, raw.get("company_brief"), "company_brief", placeholders),
        resume=_read_part(session_dir, raw.get("resume"), "resume", placeholders),
        answer_bank=[
            StarEntry(
                id=str(e.get("id", f"star{i}")),
                title=str(e.get("title", "")),
                tags=[str(t) for t in e.get("tags", [])],
                situation=str(e.get("situation", "")),
                task=str(e.get("task", "")),
                action=str(e.get("action", "")),
                result=str(e.get("result", "")),
            )
            for i, e in enumerate(raw.get("answer_bank", []))
        ],
        plan=[
            PlanStep(
                id=str(s.get("id", f"p{i}")),
                title=str(s.get("title", "")),
                key_points=[str(k) for k in s.get("key_points", [])],
                done_signals=[str(k) for k in s.get("done_signals", [])],
            )
            for i, s in enumerate(raw.get("plan", []))
        ],
        honesty_boundary=[
            HonestyClaim(claim=str(h.get("claim", "")), truth=str(h.get("truth", "")))
            for h in raw.get("honesty_boundary", [])
        ],
        placeholders=placeholders,
        source_dir=session_dir,
    )
    logger.info(
        "bundle %s loaded | role=%r company=%r | %d STAR, %d plan steps, %d honesty rows | placeholders: %s",
        bundle.session_id, bundle.role, bundle.company, len(bundle.answer_bank), len(bundle.plan),
        len(bundle.honesty_boundary), ", ".join(placeholders) or "none",
    )
    return bundle


# --------------------------------------------------------------------------
# 2. Trigger policy (G6) — deterministic, no model call
# --------------------------------------------------------------------------
# Polish interrogatives. Kept as a prefix set rather than exact forms so the many
# inflections (jaki/jaka/jakie/jakim/jakich...) are covered without listing all of them.
PL_INTERROGATIVE_PREFIXES = (
    "czy", "jak", "jaki", "jaka", "jakie", "jakim", "jakich", "jakiej", "jakas",
    "co", "czego", "czym", "czemu", "dlaczego", "kiedy", "gdzie", "ile", "ilu",
    "kto", "kogo", "komu", "kim", "ktory", "ktora", "ktore", "ktorego", "ktorych",
    "skad", "dokad", "po",
)
# Imperatives that open a request for an answer ("tell me about...", "opowiedz o...").
# These carry no question mark and no interrogative — the commonest interview opener.
# Stems, not exact forms: the same request arrives as an imperative ("opowiedz") or an
# infinitive after `prosze` ("prosze wyjasnic"), and Polish inflects both.
PL_PROMPT_VERBS = (
    "opowied", "powiedz", "wyjasn", "wytlumacz", "opis", "podaj", "przedstaw",
    "omow", "porown", "zaproponuj", "przybliz", "wymien", "pokaz",
)
EN_WH_WORDS = ("what", "how", "why", "when", "where", "which", "who", "whom", "whose")
# Auxiliaries only ask a question by SUBJECT INVERSION, which puts them first ("Do you...",
# "Would you..."). Accepted anywhere in the first three words they also match ordinary
# statements — "Now I *would* like to move on" was exactly that false positive (measured).
EN_AUXILIARIES = ("can", "could", "would", "will", "do", "does", "did", "is", "are", "have", "has")
EN_PROMPT_VERBS = ("tell", "describe", "explain", "walk", "give", "share", "compare", "show", "talk")

# Polish idioms that OPEN with an interrogative word and ask nothing. These are discourse
# markers, not a tuned threshold: "jak Pan widzi" = "as you can see", "co ciekawe" =
# "interestingly", "powiedzmy" = "let's say". Each one was a measured false positive.
PL_NOT_QUESTION_OPENERS = (
    ("jak", "pan", "widzi"), ("jak", "pani", "widzi"), ("jak", "widac"),
    ("jak", "mowilem"), ("jak", "mowilam"), ("jak", "wspomnialem"), ("jak", "juz"),
    ("co", "ciekawe"), ("co", "wiecej"), ("co", "prawda"), ("co", "wazne"),
    ("powiedzmy",),
)
# First person announcing their own intent ("I'd like to tell you about...") is a statement,
# not a request for an answer — and on a mixed-mono transcript it is usually the CANDIDATE
# talking, where a suggestion is worse than useless.
PL_FIRST_PERSON_INTENT = ("chcialbym", "chcialabym", "chce", "moge", "musze", "probuje", "sprobuje")

_DIACRITICS = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def _tokens(text: str) -> list[str]:
    """Lowercase, de-diacritic, punctuation-free words. De-diacritic because Whisper
    occasionally drops a Polish diacritic and 'ktory' must still match 'który'."""
    return re.findall(r"[a-z0-9']+", text.translate(_DIACRITICS).lower())


# Fallback language detector for CLEAN text (the --text one-shot, or an untagged transcript
# line). The live path uses Whisper's per-segment detection instead — this only decides the
# answer language when no detection tag is available. Both languages use the Latin alphabet,
# so a positive Polish signal (a diacritic, or a common Polish function word) means Polish;
# otherwise the text is treated as English.
_PL_DIACRITICS = set("ąćęłńóśźż")
_PL_MARKERS = frozenset({
    "czy", "jak", "jakie", "jaki", "jaka", "co", "gdzie", "kiedy", "dlaczego", "ktory",
    "ktora", "ktore", "pan", "pani", "prosze", "opowiedziec", "jest", "sie", "oraz",
    "moze", "pana", "swoim", "doswiadczenie", "powiedziec",
})


def detect_text_language(text: str) -> str:
    """Return "pl" or "en" for clean text, biased to Polish on any Polish signal."""
    lowered = text.lower()
    if any(ch in _PL_DIACRITICS for ch in lowered):
        return "pl"
    if set(_tokens(text)) & _PL_MARKERS:
        return "pl"
    return "en"


def looks_like_question(text: str, min_words: int | None = None) -> bool:
    """Should the ambient loop spend a model call on this segment? (G6)

    Fires on: an explicit `?`, an interrogative in the first three words, or an
    answer-requesting imperative in the first two. Interrogatives are position-limited
    on purpose — bare `jak` mid-sentence is 'as/like' ('tak jak w produkcji'), and
    accepting it anywhere is what turns a cheap rule into a spam generator.
    """
    floor = settings.SUGGESTION_MIN_WORDS if min_words is None else min_words
    words = _tokens(text)
    if len(words) < floor:
        return False
    if "?" in text:
        return True
    if any(tuple(words[: len(opener)]) == opener for opener in PL_NOT_QUESTION_OPENERS):
        return False
    head = words[:3]
    if any(w.startswith(PL_INTERROGATIVE_PREFIXES) or w in EN_WH_WORDS for w in head):
        # "po" only counts in "po co" — as a bare preposition it is everywhere.
        if head[0] != "po" or (len(words) > 1 and words[1] == "co"):
            return True
    if words[0] in EN_AUXILIARIES:
        return True
    if words[0] in PL_FIRST_PERSON_INTENT:
        return False
    return any(w.startswith(PL_PROMPT_VERBS) or w in EN_PROMPT_VERBS for w in words[:2])


# --------------------------------------------------------------------------
# 3. The suggestion (D14)
# --------------------------------------------------------------------------
# The language rule sits at the END of the system block (session 18, 2026-09-14): Ollama caches the
# prompt PREFIX, and the rule changes whenever the transcript language flips (pl<->en). At the top it
# invalidated the whole ~12k-token bundle on every switch (measured 8.4 s TTFT vs 0.16 s cached);
# at the tail only the last few hundred tokens re-process. The rule is still inside the system turn.
SYSTEM_TEMPLATE = """You are a live interview copilot for a candidate in a job interview. \
The candidate is speaking RIGHT NOW and will read your output mid-answer, so it must be \
skimmable in about three seconds.

Output format — no preamble, no closing remark, nothing else. Emit the sections IN THIS ORDER,
because the candidate starts speaking from the first line while the rest is still arriving:
POINT: one or two sentences giving the single strongest thing to say first — name the concrete \
claim AND the reason, number or example behind it, not just a headline, so the line can stand on \
its own if the candidate says nothing else. Keep it under about 35 words and speakable verbatim: \
this is the line they read while drawing breath, so it must be sayable as written, just fuller \
than a slogan.
- three bullets, max twelve words each, that they can speak from next.
EVIDENCE: one concrete fact from THEIR resume or answer bank that backs it up, or "none".
DETAIL: two to four sentences of the fuller explanation, for them to draw on if the answer \
runs long or the interviewer digs in. This part arrives while they are already talking, so it \
can afford to be denser than the bullets — but it must still be speakable, not written prose.

Ground every claim in the candidate context below. If the context does not support an \
answer, say so in one bullet rather than inventing an achievement.

=== HONESTY BOUNDARY — HARDEST RULE IN THIS PROMPT ===
Each line below is a claim the candidate MUST NOT make, with what is actually true.
Never suggest wording that makes one of these claims, and never soften "has not done X"
into "has experience with X". If a question touches one of them, your POINT line must be
the TRUE version. A gap stated plainly costs the candidate nothing; a claim that collapses
under one follow-up question ends the process.
{honesty_boundary}

=== ROLE ===
{role} at {company}

=== JOB DESCRIPTION ===
{job_description}

=== COMPANY BRIEF ===
{company_brief}

=== CANDIDATE RESUME ===
{resume}

=== PREPARED ANSWERS (STAR) ===
{answer_bank}

=== INTERVIEW PLAN ===
{plan}

{language_rule}
"""

# Human-readable speaker labels for the conversation history (G9/#326). The transcript tags
# are terse channel names (them/you); the model reads these instead.
SPEAKER_LABELS = {"them": "Interviewer", "you": "You"}

USER_TEMPLATE = """Recent conversation ({spoken_name} transcript, oldest first; \
"Interviewer:" is the other party, "You:" is the candidate you are helping):
{history}

The interviewer just said ({spoken_name}):
"{segment}"

Write the {suggestion_name} scaffold now."""


@dataclass
class Suggestion:
    segment: str
    text: str
    language: str
    reply: LlmReply
    fired_because: str

    def render(self) -> str:
        return (
            f"\n--- SUGGESTION [{self.language}] ({self.fired_because}) "
            f"| {self.reply.cost_line()} ---\n{self.text}\n"
        )


# Two rules, not one template with the languages substituted in. Formatting the
# cross-lingual wording with spoken == target produced the self-contradiction
# "write your entire answer in Polish, never in Polish" — caught by a test, and a plausible
# contributor to llama3.2:3b abandoning the question entirely on the pl->pl path.
CROSS_LINGUAL_RULE = """LANGUAGE RULE — THIS IS NOT OPTIONAL:
The interview is spoken in {spoken_name}. You will therefore receive {spoken_name} text.
You MUST write your entire answer in {suggestion_name}, never in {spoken_name}.
Do not translate the question back. Do not explain the language choice.
Write only {suggestion_name}, even though the input is {spoken_name}."""

SAME_LANGUAGE_RULE = """LANGUAGE RULE — THIS IS NOT OPTIONAL:
The interview is spoken in {spoken_name} and you write in {spoken_name} too.
Do not repeat or translate the question. Answer it directly."""


def language_rule(spoken_name: str, suggestion_name: str) -> str:
    template = SAME_LANGUAGE_RULE if spoken_name == suggestion_name else CROSS_LINGUAL_RULE
    return template.format(spoken_name=spoken_name, suggestion_name=suggestion_name)


def _plan_block(bundle: ContextBundle) -> str:
    if not bundle.plan:
        return "(no plan loaded)"
    lines = []
    for step in bundle.plan:
        lines.append(f"- [{step.id}] {step.title}")
        for point in step.key_points:
            lines.append(f"    * {point}")
    return "\n".join(lines)


def resolve_languages(
    bundle: ContextBundle, segment: str,
    suggestion_language: str | None = None, spoken_language: str | None = None,
) -> tuple[str, str]:
    """Resolve (spoken, target) concrete language codes for one turn.

    `suggestion_language="match"` (the default) answers in the question's own language
    (user decision 2026-09-02). The question's language is `spoken_language` when the caller
    knows it (the live path reads it from the transcript's detection tag); otherwise it is
    detected from the segment text. In match mode spoken == target, so a Polish question
    yields a Polish scaffold and an English question an English one. A forced "en"/"pl" still
    answers in that language while `spoken` tracks the real question language, so the prompt's
    cross-lingual vs same-language rule is chosen correctly either way."""
    configured = (suggestion_language or bundle.suggestion_language).lower()
    spoken = (spoken_language or "").lower()
    if configured == "match":
        lang = spoken or detect_text_language(segment)
        return lang, lang
    spoken = spoken or bundle.spoken_language.lower()
    return spoken, configured


def build_messages(
    bundle: ContextBundle, segment: str, history: list[str] | None = None,
    suggestion_language: str | None = None, spoken_language: str | None = None,
) -> tuple[str, str]:
    """Return (system, user). Separated from `suggest()` so a test can assert the
    asymmetry is *in the prompt* without spending a model call."""
    spoken, target = resolve_languages(bundle, segment, suggestion_language, spoken_language)
    spoken_name = LANGUAGE_NAMES.get(spoken, spoken)
    target_name = LANGUAGE_NAMES.get(target, target)
    system = SYSTEM_TEMPLATE.format(
        language_rule=language_rule(spoken_name, target_name),
        honesty_boundary="\n".join(h.as_prompt_block() for h in bundle.honesty_boundary)
        or "(none recorded — do not infer one; ground answers in the resume and answer bank)",
        role=bundle.role or "(role not specified)",
        company=bundle.company or "(company not specified)",
        job_description=bundle.job_description or "(not loaded)",
        company_brief=bundle.company_brief or "(not loaded)",
        resume=bundle.resume or "(not loaded)",
        answer_bank="\n".join(e.as_prompt_block() for e in bundle.answer_bank) or "(none)",
        plan=_plan_block(bundle),
    )
    user = USER_TEMPLATE.format(
        spoken_name=spoken_name,
        suggestion_name=target_name,
        history="\n".join(history or []) or "(nothing yet)",
        segment=segment.strip(),
    )
    return system, user


class Superseded(StreamCancelled):
    """Raised inside the token stream when a newer question has arrived (see SuggestionRunner).

    Subclasses `llm_client.StreamCancelled` so `llm_call` lets it through untouched instead of
    reporting a superseded suggestion as a backend failure."""


def suggest(
    bundle: ContextBundle, segment: str, history: list[str] | None = None,
    suggestion_language: str | None = None, backend: str | None = None,
    model: str | None = None, fired_because: str = "manual",
    on_token: Callable[[str], None] | None = None,
    spoken_language: str | None = None,
) -> Suggestion:
    """One metered suggestion. Propagates BackendUnavailable — the caller decides whether
    a dead backend ends the session or just skips this segment.

    With `on_token` the text is streamed. That is the whole latency story: time-to-first-token
    is ~0.15 s regardless of prompt size, while waiting for the completed answer costs seconds.
    Streaming also makes a bigger, slower model affordable — what matters is when the POINT
    line lands, not when the last token does.
    """
    _spoken, target = resolve_languages(bundle, segment, suggestion_language, spoken_language)
    system, user = build_messages(bundle, segment, history, suggestion_language, spoken_language)
    reply = llm_call(
        user, system=system, backend=backend, model=model,
        max_tokens=settings.SUGGESTION_MAX_TOKENS, on_token=on_token,
    )
    return Suggestion(segment=segment.strip(), text=reply.text, language=target,
                      reply=reply, fired_because=fired_because)


def prefill_bundle(bundle: ContextBundle, backend: str | None = None, model: str | None = None) -> float:
    """Push the full system block through the LOCAL model once so Ollama caches the prompt prefix
    (session 18, 2026-09-14). The salience warm-up only loads the model; the first real question
    still paid the ~12k-token prefill (measured 6.9-7.4 s TTFT vs 0.16-0.24 s once cached). One
    1-token call here moves that cost to start-up. Local only: on a cloud backend a throwaway
    call would bill a cache write. Best-effort: a failure is logged, never raised. Returns seconds."""
    if (backend or "local") != "local":
        return 0.0
    started = time.perf_counter()
    try:
        system, user = build_messages(bundle, "Proszę opowiedzieć o sobie.", [])
        llm_call(user, system=system, backend=backend, model=model, max_tokens=1)
    except Exception as error:  # noqa: BLE001 - warming is best-effort by design
        logger.warning("bundle prefill failed (first question will pay the prefill): %s", error)
        return 0.0
    took = time.perf_counter() - started
    logger.info("bundle prefill | %.2fs | prompt prefix now cached on the local model", took)
    return took


class SuggestionRunner:
    """Runs one suggestion at a time in a worker thread, cancellable by the next question.

    The interviewer does not wait for the copilot. If a new question arrives while a
    suggestion is still streaming, that suggestion is now answering the wrong question — so it
    is abandoned mid-stream rather than left to finish and scroll the useful one off the screen.
    Cancellation is cooperative: `on_token` raises `Superseded`, which unwinds the HTTP stream.
    """

    def __init__(self, on_event: Callable[[dict], None] | None = None) -> None:
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        # #322/D18: an optional structured mirror of what is printed, so the dashboard can
        # render the same stream without this class knowing a websocket exists. None (the
        # CLI path) keeps the stdout behaviour byte-identical.
        self._on_event = on_event
        self._seq = 0

    def _emit(self, **payload: object) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event({"kind": "suggestion", **payload})
        except Exception:  # a broken consumer must never take down the interview
            logger.exception("suggestion event sink raised — continuing")

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self, wait: float = 0.5) -> None:
        if self.busy:
            self._cancel.set()
            self._thread.join(timeout=wait)  # type: ignore[union-attr]

    def drain(self, timeout: float) -> None:
        """Let the last suggestion finish (used at shutdown) rather than cutting it off."""
        if self.busy:
            self._thread.join(timeout=timeout)  # type: ignore[union-attr]

    def start(self, stamp: str | None = None, **kwargs: object) -> None:
        """Supersede whatever is in flight, then stream a new suggestion to stdout.

        `stamp` is carried for the event sink only (it labels which transcript line the
        suggestion is answering) and is never passed to `suggest()`.
        """
        self.cancel()
        cancel = threading.Event()
        self._cancel = cancel
        self._seq += 1
        sid = self._seq

        def emit(piece: str) -> None:
            if cancel.is_set():
                raise Superseded
            print(piece, end="", flush=True)
            self._emit(phase="token", id=sid, text=piece)

        def work() -> None:
            header = (
                f"\n--- SUGGESTION [{kwargs.get('suggestion_language')}] "
                f"({kwargs.get('fired_because')}) ---"
            )
            print(header, flush=True)
            self._emit(phase="start", id=sid, stamp=stamp,
                       segment=str(kwargs.get("segment", "")),
                       language=str(kwargs.get("suggestion_language") or ""),
                       fired_because=str(kwargs.get("fired_because") or ""))
            try:
                result = suggest(on_token=emit, **kwargs)  # type: ignore[arg-type]
            except Superseded:
                print("\n[superseded — a newer question arrived]", flush=True)
                self._emit(phase="end", id=sid, status="superseded",
                           note="superseded — a newer question arrived")
                return
            except BackendUnavailable:
                # One dead call must not end the interview — the transcript keeps flowing.
                logger.exception("suggestion failed — continuing")
                print("\n[suggestion failed — see the log; the transcript is unaffected]", flush=True)
                self._emit(phase="end", id=sid, status="failed",
                           note="suggestion failed — see the log; the transcript is unaffected")
                return
            print(f"\n[{result.reply.cost_line()}]", flush=True)
            self._emit(phase="end", id=sid, status="complete", note=result.reply.cost_line())

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()


# --------------------------------------------------------------------------
# The consumption seam (D19) — tail the transcript file
# --------------------------------------------------------------------------
LINE_RE = re.compile(r"^\[(\d\d:\d\d)-(\d\d:\d\d)\]\s*(?:(them|you)(?:\s*\(([a-z]{2})\))?:\s*)?(.+)$")


def parse_transcript_line(line: str) -> tuple[str, str | None, str | None, str] | None:
    """`[mm:ss-mm:ss] them (pl): text` -> (stamp, speaker, language, text). The speaker
    prefix is optional (G9/#326) and the `(language)` inside it is optional too (per-question
    answer language): an untagged line — a pre-#326 transcript, or a monitor-only run —
    yields speaker/language None, which downstream treats as the interviewer in the spoken
    language. Header (`#`) and blank lines return None."""
    line = line.rstrip("\n")
    if not line or line.startswith("#"):
        return None
    match = LINE_RE.match(line)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}", match.group(3), match.group(4), match.group(5).strip()


def newest_transcript(output_dir: Path | None = None) -> Path | None:
    candidates = sorted((output_dir or OUTPUT_DIR).glob("live_transcript_2*.txt"))
    return candidates[-1] if candidates else None


def follow_transcript(
    path: Path, stop_after_idle: float | None = None, from_start: bool = True,
    poll: float | None = None, stop_event: "threading.Event | None" = None,
):
    """Yield `(stamp, speaker, language, text)` from a transcript file as `live_transcribe.py` writes it.

    Tailing, not an in-process callback, is the D19 seam: the recorder is the
    irreplaceable artifact and must not share a process with a network call, the reasoning
    layer can attach or die mid-interview without touching the call, and a finished
    transcript replays through the identical code path.

    `stop_event` (#607 app mode) lets a manager end the tail cleanly when the recorder is
    stopped and a *new* transcript file will follow — it is checked only between lines, so a
    line already read is always yielded. `None` keeps the CLI/replay path byte-identical.
    """
    interval = settings.TRANSCRIPT_POLL_SECONDS if poll is None else poll
    with open(path, "r", encoding="utf-8") as handle:
        if not from_start:
            handle.seek(0, 2)
        idle = 0.0
        while True:
            if stop_event is not None and stop_event.is_set():
                return
            line = handle.readline()
            if not line:
                if stop_after_idle is not None and idle >= stop_after_idle:
                    return
                time.sleep(interval)
                idle += interval
                continue
            idle = 0.0
            parsed = parse_transcript_line(line)
            if parsed is not None:
                yield parsed


@dataclass
class Controls:
    """Live, thread-safe switches the dashboard app (#607) flips while the ambient loop runs.

    The loop reads these at each fire decision, so a toggle takes effect on the next question
    without restarting anything. With `controls=None`, run_ambient uses its fixed `backend`/
    `model` args and always fires — the CLI and replay paths are byte-identical.

    `backend` maps the UI's suggestion-source switch: "local" = local-only Ollama, "cloud" =
    "local + api" (the BYOK Claude egress, opt-in and announced by llm_client — SI1/D14).
    """

    suggestions_on: bool = True
    backend: str = "local"
    model: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> tuple[bool, str, str]:
        with self._lock:
            return self.suggestions_on, self.backend, self.model

    def set_suggestions(self, on: bool) -> None:
        with self._lock:
            self.suggestions_on = bool(on)

    def set_backend(self, backend: str, model: str = "") -> None:
        with self._lock:
            self.backend, self.model = backend, model


def run_ambient(
    bundle: ContextBundle, path: Path, backend: str, model: str,
    suggestion_language: str | None = None, stop_after_idle: float | None = None,
    from_start: bool = True, questions_only: bool | None = None, limit: int = 0,
    answer_speaker: str | None = None, gate: SalienceGate | None = None,
    sink: Callable[[dict], None] | None = None,
    controls: "Controls | None" = None, stop_event: "threading.Event | None" = None,
) -> int:
    """The ambient loop (D12): every transcript line in, a suggestion out when the
    trigger policy fires AND the salience gate agrees (D23). Returns the number of
    suggestions produced; the trigger-vs-gate split is logged and printed at the end.

    `sink` (#322/D18) receives the same decisions as a structured dict — one `line` event per
    transcript line and one `gate` event per D20-triggered question — so the dashboard renders
    exactly what this loop decided rather than re-deriving it. It is the loop's only concession
    to having a UI: with `sink=None` the CLI path is unchanged.
    """
    only_questions = settings.FIRE_ON_QUESTIONS_ONLY if questions_only is None else questions_only
    answer_speaker = (answer_speaker or settings.ANSWER_SPEAKER).lower()

    def emit(**payload: object) -> None:
        if sink is None:
            return
        try:
            sink(payload)
        except Exception:  # a broken consumer must never take down the interview
            logger.exception("ambient event sink raised — continuing")

    if gate is None:
        gate = SalienceGate(bundle, is_question=looks_like_question)
    if gate.active:
        print(f"  {gate.describe()}", flush=True)
        gate.warm()  # pay the client/model cost now, not on the first question of the call
    prefill_bundle(bundle, backend, model)  # cache the bundle prefix so question 1 streams at once
    history: list[str] = []
    fired = 0
    triggered = 0   # survived D20 — what the loop WOULD have fired before the gate
    gated = 0       # dropped by D23
    gate_ms: list[float] = []
    last_fire = 0.0
    # Constructed without the seam when there is no consumer, so a test double or any other
    # substitute for the runner needs no signature change to keep working (#322).
    runner = SuggestionRunner(on_event=sink) if sink is not None else SuggestionRunner()
    for stamp, speaker, language, text in follow_transcript(
        path, stop_after_idle=stop_after_idle, from_start=from_start, stop_event=stop_event
    ):
        who = speaker or "them"  # an untagged line is the interviewer (see parse_transcript_line)
        lang_note = f" ({language})" if language else ""
        print(f"\n[{stamp}] {who}{lang_note}: {text}", flush=True)
        # Emitted BEFORE any filtering: the dashboard's transcript is what was said, not what
        # the copilot chose to answer. A `you:` line and a gated-out question both still render.
        emit(kind="line", stamp=stamp, speaker=who, language=language, text=text,
             tagged=speaker is not None)
        # History carries the speaker label so the model knows which prior turns were the
        # interviewer's and which were the candidate's own (G9/#326). A skipped `you:` turn
        # still enters history — it is context the copilot should not repeat, just not answer.
        history.append(f"{SPEAKER_LABELS[who]}: {text}")
        history[:] = history[-settings.SUGGESTION_HISTORY_LINES:]

        # #607 app mode: the live switches. Read once per line. When suggestions are OFF the
        # line + meter still flow (emitted above) — only the trigger/gate/fire work is skipped,
        # so nothing touches a model or the GPU. `backend`/`model` are read live so the
        # "local + api" switch takes effect on the very next question.
        fire_backend, fire_model = backend, model
        if controls is not None:
            suggestions_on, fire_backend, fire_model = controls.snapshot()
            if not suggestions_on:
                continue

        # G9/#326: never answer the candidate's own turns. "any" restores channel-blind firing.
        if answer_speaker != "any" and who != answer_speaker:
            logger.info("segment skipped by speaker filter (speaker=%s, answering %s): %r", who, answer_speaker, text[:60])
            continue

        if only_questions and not looks_like_question(text):
            logger.info("segment skipped by trigger policy (not a question): %r", text[:60])
            continue
        triggered += 1

        # D23: the trigger says "a question", the gate says "worth a suggestion". Ordered
        # this way round on purpose — the free rule culls first, so the gate only ever runs
        # on what survived it. A gate failure fires (SALIENCE_FAIL_OPEN); see salience.py.
        verdict = gate.evaluate(text)
        if verdict.latency_ms:
            gate_ms.append(verdict.latency_ms)
        if not verdict.fire:
            gated += 1
            print(f"  [gate] dropped — {verdict.log_line()}", flush=True)
            logger.info("segment dropped by salience gate (%s): %r", verdict.log_line(), text[:60])
            # D23: a dropped question produces NO suggestion on screen. It is reported only as
            # a counter, so the user can see the gate is working without reading a non-answer.
            emit(kind="gate", stamp=stamp, fire=False, detail=verdict.log_line())
            continue
        if gate.active:
            logger.info("segment passed salience gate (%s)", verdict.log_line())
        emit(kind="gate", stamp=stamp, fire=True, detail=verdict.log_line())

        since = time.monotonic() - last_fire
        # The cooldown does NOT apply while a suggestion is still streaming: a genuinely new
        # question must be able to supersede the one being written, or the copilot spends the
        # interview one question behind.
        if last_fire and since < settings.SUGGESTION_COOLDOWN_SECONDS and not runner.busy:
            logger.info("segment skipped: %.1fs into the %.1fs cooldown", since, settings.SUGGESTION_COOLDOWN_SECONDS)
            # NOT a gate event. The cooldown is a different reason for a blank panel and
            # counting it as a D23 drop puts a false number on screen — measured: the
            # `--no-salience` replay reported "2 dropped" with the gate switched OFF.
            emit(kind="skip", stamp=stamp, reason="cooldown",
                 detail=f"{since:.1f}s into the {settings.SUGGESTION_COOLDOWN_SECONDS:.0f}s cooldown")
            continue
        runner.start(
            stamp=stamp,
            bundle=bundle, segment=text, history=history[:-1],
            suggestion_language=suggestion_language or bundle.suggestion_language,
            spoken_language=language,  # the question's own language (match mode answers in it)
            backend=fire_backend, model=fire_model,
            fired_because="question" if only_questions else "every segment",
        )
        last_fire = time.monotonic()
        fired += 1
        if limit and fired >= limit:
            break
    runner.drain(timeout=settings.REASONING_TIMEOUT_SECONDS)
    if gate.active:
        median = sorted(gate_ms)[len(gate_ms) // 2] if gate_ms else 0.0
        summary = (f"salience gate: {triggered} question(s) passed the D20 trigger -> "
                   f"{triggered - gated} fired, {gated} dropped (median {median:.0f} ms/decision)")
        print(f"\n  {summary}", flush=True)
        logger.info(summary)
    return fired


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _selftest_heuristic(fixtures: Path) -> int:
    """Measure the trigger rule against labelled fixtures. No model, no GPU, no tokens."""
    cases = json.loads(fixtures.read_text(encoding="utf-8"))["cases"]
    tp = fp = fn = tn = 0
    misses: list[str] = []
    for case in cases:
        predicted = looks_like_question(case["text"])
        expected = bool(case["is_question"])
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
            misses.append(f"  FALSE POSITIVE [{case['lang']}] {case['text']}")
        elif not predicted and expected:
            fn += 1
            misses.append(f"  FALSE NEGATIVE [{case['lang']}] {case['text']}")
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"trigger heuristic on {len(cases)} labelled lines ({fixtures.name})")
    print(f"  TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    print(f"  precision {precision:.2f}  recall {recall:.2f}  F1 {f1:.2f}")
    for line in misses:
        print(line)
    return 0


def _selftest_salience(fixtures: Path, session: str, backend: str | None,
                       model: str | None, threshold: float | None) -> int:
    """Score the D23 gate against labelled turns, the way `--selftest-heuristic` scores D20.

    Exits non-zero below the recorded baseline so a prompt or model change that quietly
    costs recall is caught. **The labels are IN-SAMPLE** — these 24 turns are the same ones
    the judge prompt was chosen against, exactly as D20's fixtures were. Treat the numbers
    as a regression guard, not an out-of-sample estimate.
    """
    doc = json.loads(fixtures.read_text(encoding="utf-8"))
    cases = doc["cases"]
    bundle = load_bundle(session)
    gate = SalienceGate(bundle, backend=backend, model=model, threshold=threshold,
                        enabled=True, is_question=looks_like_question)
    print(f"salience gate on {len(cases)} labelled turns ({fixtures.name})")
    print(f"  {gate.describe()}")
    print(f"  source: {doc['source']}")
    gate.warm()
    tp = fp = fn = tn = 0
    latencies: list[float] = []
    misses: list[str] = []
    for case in cases:
        verdict = gate.evaluate(case["text"])
        latencies.append(verdict.latency_ms)
        expected = bool(case["salient"])
        if verdict.fire and expected:
            tp += 1
        elif verdict.fire and not expected:
            fp += 1
            misses.append(f"  FALSE POSITIVE {case['stamp']} ({verdict.log_line()}) — {case['why']}")
        elif expected:
            fn += 1
            misses.append(f"  FALSE NEGATIVE {case['stamp']} ({verdict.log_line()}) — {case['why']}")
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    latencies.sort()
    median = latencies[len(latencies) // 2] if latencies else 0.0
    p90 = latencies[int(len(latencies) * 0.9)] if latencies else 0.0
    print(f"  TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    print(f"  precision {precision:.2f}  recall {recall:.2f}  F1 {f1:.2f}")
    print(f"  fires {tp + fp}/{len(cases)} (the ungated trigger fires all {len(cases)})")
    print(f"  gate latency median {median:.0f} ms  p90 {p90:.0f} ms")
    for line in misses:
        print(line)
    print(f"  CAVEAT: {doc['caveat']}")

    floors = doc.get("baseline", {}).get(gate.backend)
    if not floors:
        print(f"  no recorded baseline for backend {gate.backend!r} — not gating on it")
        return 0
    failures = [
        f"{name} {value:.2f} < baseline {floor:.2f}"
        for name, value, floor in (
            ("precision", precision, floors["min_precision"]),
            ("recall", recall, floors["min_recall"]),
            ("F1", f1, floors["min_f1"]),
        )
        if value < floor - 1e-9
    ]
    if failures:
        print("  REGRESSION: " + "; ".join(failures))
        return 1
    print("  OK — at or above the recorded baseline")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--session", help="session id under scripts/inputs/sessions/ (or a path)")
    parser.add_argument("--text", help="one-shot: suggest for this transcript segment and exit")
    parser.add_argument("--watch", action="store_true", help="follow the newest live transcript in scripts/outputs/")
    parser.add_argument("--follow", help="follow this transcript file as it grows")
    parser.add_argument("--replay", help="read a finished transcript through the same seam")
    parser.add_argument("--suggestion-language", choices=("match", "en", "pl"), help="override SUGGESTION_LANGUAGE ('match' = answer in the question's language; default from settings)")
    parser.add_argument("--backend", choices=("local", "cloud"), help="override REASONING_BACKEND")
    parser.add_argument("--model", help="override the backend's model")
    parser.add_argument("--all-segments", action="store_true", help="fire on every segment, not only questions (G6 comparison)")
    parser.add_argument("--answer-speaker", choices=("them", "you", "any"), help="whose turns to answer (default: settings.ANSWER_SPEAKER = 'them', the interviewer only)")
    parser.add_argument("--limit", type=int, default=0, help="stop after N suggestions (0 = unlimited)")
    parser.add_argument("--wait-seconds", type=float, default=0.0, help="with --watch: wait this long for a transcript to appear")
    parser.add_argument("--idle-timeout", type=float, help="stop after this many seconds with no new line")
    parser.add_argument("--selftest-heuristic", action="store_true", help="measure the trigger rule against fixtures; no model call")
    parser.add_argument("--fixtures", default=str(PROJECT_ROOT / "tests" / "fixtures" / "question_fixtures.json"))
    parser.add_argument("--selftest-salience", action="store_true", help="measure the D23 salience gate against labelled turns (needs the local backend); non-zero on regression")
    parser.add_argument("--salience-fixtures", default=str(PROJECT_ROOT / "tests" / "fixtures" / "salience_fixtures.json"))
    parser.add_argument("--salience-backend", choices=("llm", "embed", "off"), help="override SALIENCE_BACKEND ('llm' = one-word YES/NO on the resident model; 'embed' = the measured-and-rejected cosine route)")
    parser.add_argument("--salience-model", help="override SALIENCE_MODEL (default: the local suggestion model, so the gate costs no extra VRAM)")
    parser.add_argument("--salience-threshold", type=float, help="override SALIENCE_THRESHOLD (embed backend only)")
    parser.add_argument("--no-salience", action="store_true", help="disable the D23 gate; every detected question fires (pre-#386 behaviour)")
    args = parser.parse_args()

    if args.selftest_heuristic:
        sys.exit(_selftest_heuristic(Path(args.fixtures)))
    if args.selftest_salience:
        salience_fixtures = Path(args.salience_fixtures)
        if not salience_fixtures.is_file():
            print(
                f"salience self-test skipped: {salience_fixtures.name} is a private, in-sample "
                "fixture (a real interview transcript) and is not shipped with this repo. "
                "Point --salience-fixtures at your own labelled set to run it."
            )
            sys.exit(0)
        sys.exit(_selftest_salience(
            salience_fixtures, args.session or "example_ai_engineer",
            args.salience_backend, args.salience_model, args.salience_threshold,
        ))
    if not args.session:
        parser.error("--session is required (or use --selftest-heuristic / --selftest-salience)")

    run_id = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"  # TRK
    bundle = load_bundle(args.session)
    for warning in bundle.warn_lines():
        print(f"  {warning}", flush=True)

    backend = resolve_backend(args.backend)
    model = args.model or model_for(backend)
    target = args.suggestion_language or bundle.suggestion_language
    print(announce_backend(backend, model), flush=True)
    print(
        f"  session {bundle.session_id} | transcript {bundle.spoken_language} -> suggestions {target} "
        f"| trigger: {'questions only' if not args.all_segments and settings.FIRE_ON_QUESTIONS_ONLY else 'every segment'}",
        flush=True,
    )
    logger.info("run %s START | backend=%s model=%s | session=%s", run_id, backend, model, bundle.session_id)

    try:
        if args.text:
            _, resolved = resolve_languages(bundle, args.text, target)  # "match" -> the text's language
            print(f"\n--- SUGGESTION [{resolved}] (--text) ---", flush=True)
            result = suggest(
                bundle, args.text, suggestion_language=target, backend=backend, model=model,
                fired_because="--text", on_token=lambda p: print(p, end="", flush=True),
            )
            print(f"\n[{result.reply.cost_line()}]")
            fired = 1
        else:
            if args.follow or args.replay:
                path = Path(args.follow or args.replay)
            else:
                deadline = time.monotonic() + args.wait_seconds
                path = newest_transcript()
                while path is None and time.monotonic() < deadline:
                    time.sleep(0.25)
                    path = newest_transcript()
                if path is None:
                    logger.error("no live_transcript_*.txt in %s — start live_transcribe.py first", OUTPUT_DIR)
                    sys.exit(2)
            if not path.is_file():
                logger.error("transcript not found: %s", path)
                sys.exit(2)
            print(f"  following {path}", flush=True)
            gate = SalienceGate(
                bundle, backend=args.salience_backend, model=args.salience_model,
                threshold=args.salience_threshold,
                enabled=False if args.no_salience else None,
                is_question=looks_like_question,
            )
            fired = run_ambient(
                bundle, path, backend=backend, model=model, suggestion_language=target, gate=gate,
                stop_after_idle=args.idle_timeout if args.replay is None else (args.idle_timeout or 1.0),
                questions_only=not args.all_segments,
                limit=args.limit,
                answer_speaker=args.answer_speaker,
            )
    except BackendUnavailable:
        logger.exception("reasoning run failed")
        sys.exit(1)

    logger.info("run %s END | %d suggestion(s)", run_id, fired)
    print(f"\n=== {fired} suggestion(s) ===")


if __name__ == "__main__":
    main()
