"""The mock-interview ENGINE (D30, offline practice track): questions in, a graded report out.

UI-independent and unit-testable — the dashboard/websocket controller (a LATER subtask) drives
this from another thread; nothing here knows a screen exists. Four pieces:

1. **`generate_questions(bundle, n)`** — the interview-side content. Returns `bundle.question_bank`
   when `generate_context.py` (D30) already authored one; otherwise authors N questions + rubrics
   from the bundle's plan/competencies in ONE metered call, reusing generate_context's JSON
   helpers (fenced/malformed tolerance + one repair) so a half-parsed bank never reaches scoring.

2. **`TrainingSession`** — ordered questions, an active-question pointer, and a per-question
   answer buffer. `add_answer_segment()` appends to the active question, `advance()` closes the
   current buffer and moves on, `finish()` ends it. Lock-guarded so a controller thread can feed
   segments while the main thread reads — kept deliberately small.

3. **`score_answer(question, answer_text, bundle)`** — exactly ONE `llm_call` per answer, grading
   against the rubric criteria. **The arithmetic is deterministic Python** (CLAUDE.md Determinism
   First): the model returns only a LEVEL and a NOTE per criterion (plus the concepts/strengths
   prose); `_aggregate` maps each level through `RUBRIC_LEVEL_SCORE * criterion.weight`, sums for
   the total, and compares against `rubric.max_score()`. The model never does the maths.

4. **`score_session(session, bundle)`** — batch-scores every answered question into a report,
   renders `to_markdown()`, and writes `scripts/outputs/training_report_<stamp>.md`
   (write-temp-then-rename; local-only per SI1 — no egress).

Run it:
    python scripts/training.py --selftest                        # no model, no GPU — aggregation math
    python scripts/training.py --session acme --answers ans.json # grade a fixture answers file
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from scripts.generate_context import (  # noqa: E402
    REPAIR_PROMPT, _normalize_question_bank, parse_json_object,
)
from scripts.llm_client import (  # noqa: E402
    BackendUnavailable, LlmReply, llm_call, model_for, resolve_backend,
)
from scripts.logger import get_logger  # noqa: E402
from scripts.reasoning import (  # noqa: E402
    RUBRIC_LEVEL_SCORE, ContextBundle, Question, Rubric, RubricCriterion,
    _parse_question, load_bundle,
)

logger = get_logger("training")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "outputs"


# --------------------------------------------------------------------------
# 1. Question generation — prefer the pre-built bank, else author one
# --------------------------------------------------------------------------
GEN_SYSTEM = """You are an interview-prep assistant. Given a role and the competencies a panel \
would probe, you author graded practice questions for a candidate. You NEVER invent anything \
about the candidate — only the QUESTIONS an interviewer would ask about the role.

Return a SINGLE JSON object and nothing else — no prose, no markdown fences. Its shape:
{
  "question_bank": [
    {
      "competency": "<must match one of the competencies below>",
      "question": "<one interview question>",
      "language": "en",
      "rubric": {
        "criteria": [
          {"id": "depth", "label": "Technical depth", "weight": 1.0,
           "levels": {"excellent": "...", "adequate": "...", "weak": "..."}}
        ]
      }
    }
  ]
}
Author about {n} question(s) in total, spread across the competencies. Each rubric needs 2-3 \
weighted criteria with concrete 'excellent' / 'adequate' / 'weak' descriptors."""

GEN_USER = """ROLE: {role}
COMPANY: {company}
TARGET QUESTION COUNT: {n}

=== COMPETENCIES ===
{competencies}

=== ROLE CONTEXT (job description) ===
{jd}

Author the practice question bank now, as one JSON object."""


def _competency_block(bundle: ContextBundle) -> str:
    """Render the bundle's plan steps as the competency arc the questions must cover."""
    if not bundle.plan:
        return "(no plan loaded — infer the competencies from the job description below)"
    lines: list[str] = []
    for step in bundle.plan:
        lines.append(f"- {step.title}")
        for point in step.key_points:
            lines.append(f"    * {point}")
    return "\n".join(lines)


def _grade_json(prompt: str, system: str | None, backend: str, model: str | None,
                max_tokens: int) -> dict:
    """One metered call returning a JSON object, with ONE repair retry on malformed output.

    Reuses generate_context's robust parser (fenced / trailing-prose tolerant) and repair prompt.
    Raises ValueError with a pointed message if even the repaired reply is not valid JSON — a
    half-parsed structure is worse than a clear stop.
    """
    reply = llm_call(prompt, system=system, backend=backend, model=model, max_tokens=max_tokens,
                     timeout=settings.GENERATE_TIMEOUT_SECONDS)
    logger.info("model call | %s", reply.cost_line())
    try:
        return parse_json_object(reply.text)
    except ValueError as first_error:
        logger.warning("reply was not valid JSON (%s) — attempting one repair call", first_error)
        repaired = llm_call(REPAIR_PROMPT.format(bad=reply.text), system=None,
                            backend=backend, model=model, max_tokens=max_tokens,
                            timeout=settings.GENERATE_TIMEOUT_SECONDS)
        try:
            return parse_json_object(repaired.text)
        except ValueError as second_error:
            raise ValueError(
                f"could not obtain valid JSON from the model after one repair "
                f"(first: {first_error}; after repair: {second_error})."
            ) from second_error


def generate_questions(
    bundle: ContextBundle, n: int | None = None,
    backend: str | None = None, model: str | None = None,
) -> list[Question]:
    """Return the questions to practise against.

    Prefers a pre-built `bundle.question_bank` (authored by generate_context.py, D30) and makes NO
    model call in that case. Only when the bank is empty does it author `n` questions from the
    bundle's plan/competencies in one call (default `n` = settings.TRAINING_QUESTION_COUNT).
    """
    if bundle.question_bank:
        logger.info("using the bundle's pre-built question_bank (%d question(s)) — no model call",
                    len(bundle.question_bank))
        return list(bundle.question_bank)

    count = settings.TRAINING_QUESTION_COUNT if n is None else int(n)
    chosen = resolve_backend(backend or settings.SCORING_BACKEND)
    use_model = model or settings.SCORING_MODEL or None
    logger.info("no question_bank in bundle — authoring %d question(s) | backend=%s model=%s",
                count, chosen, use_model or model_for(chosen))
    system = GEN_SYSTEM.replace("{n}", str(count))
    user = GEN_USER.format(
        role=bundle.role or "(unspecified)", company=bundle.company or "(unspecified)",
        n=count, competencies=_competency_block(bundle),
        jd=bundle.job_description or "(not loaded)",
    )
    parsed = _grade_json(user, system, chosen, use_model, settings.GENERATE_MAX_TOKENS)
    default_lang = bundle.spoken_language or "en"
    normalized = _normalize_question_bank(parsed.get("question_bank"), default_lang)
    questions = [_parse_question(i, q) for i, q in enumerate(normalized)]
    logger.info("authored %d practice question(s)", len(questions))
    return questions


# --------------------------------------------------------------------------
# 2. TrainingSession — ordered questions, active pointer, per-question buffer
# --------------------------------------------------------------------------
class TrainingSession:
    """A running mock interview: an ordered question list, an active-question pointer, and one
    answer buffer per question. Deliberately small and lock-guarded so a dashboard controller can
    feed answer segments from another thread while the main thread reads state.
    """

    def __init__(self, questions: list[Question]) -> None:
        self.questions: list[Question] = list(questions)
        self._buffers: list[list[str]] = [[] for _ in self.questions]
        self._index: int = 0
        self.finished: bool = False
        self._lock = threading.Lock()

    @property
    def active_index(self) -> int:
        return self._index

    @property
    def active_question(self) -> Question | None:
        with self._lock:
            if self.finished or not (0 <= self._index < len(self.questions)):
                return None
            return self.questions[self._index]

    def add_answer_segment(self, text: str) -> bool:
        """Append `text` to the ACTIVE question's answer buffer. Returns False (a no-op) when the
        session is finished or the pointer has run past the last question."""
        with self._lock:
            if self.finished or not (0 <= self._index < len(self.questions)):
                return False
            piece = text.strip()
            if piece:
                self._buffers[self._index].append(piece)
            return True

    def advance(self) -> bool:
        """Close the current answer buffer and move to the next question. Returns True when a new
        active question is now in focus, False when that was the last one (pointer parks past the
        end and `active_question` becomes None)."""
        with self._lock:
            if self._index < len(self.questions):
                self._index += 1
            return 0 <= self._index < len(self.questions)

    def seek(self, index: int) -> bool:
        """Position the active-question pointer at `index` (#648 follow-up): the public way to
        move the pointer, so a caller never has to reach into the private `_index`. `index` is
        clamped into `[0, len(questions)]` — the top bound is the legitimate park-past-the-end
        position `advance()` also reaches. Returns True when a real question is now active."""
        with self._lock:
            if not self.questions:
                self._index = 0
                return False
            self._index = max(0, min(int(index), len(self.questions)))
            return 0 <= self._index < len(self.questions)

    def finish(self) -> None:
        """End the session. Buffers are kept for scoring; only new segments are refused."""
        with self._lock:
            self.finished = True

    def answer_text(self, index: int) -> str:
        """The buffered answer for question `index`, segments joined in arrival order."""
        with self._lock:
            if not (0 <= index < len(self._buffers)):
                return ""
            return " ".join(self._buffers[index]).strip()

    def answered(self) -> list[tuple[Question, str]]:
        """(question, answer_text) for every question with a non-empty buffer, in order."""
        pairs: list[tuple[Question, str]] = []
        for i, question in enumerate(self.questions):
            text = self.answer_text(i)
            if text:
                pairs.append((question, text))
        return pairs


# --------------------------------------------------------------------------
# 3. Scoring — ONE model call per answer; the arithmetic is deterministic
# --------------------------------------------------------------------------
SCORE_SYSTEM = """You are a strict but fair interview-answer grader. You are given ONE interview \
question, its scoring rubric, and the candidate's spoken answer. Grade the answer against EACH \
rubric criterion.

Return a SINGLE JSON object and nothing else — no prose, no markdown fences. Its shape:
{
  "per_criterion": [
    {"id": "<the criterion id, exactly as given>",
     "level": "excellent | adequate | weak | missing",
     "note": "<one short sentence justifying the level, grounded in the answer>"}
  ],
  "concepts_to_refresh": ["<concept the answer was weak or silent on>"],
  "strengths": ["<what the answer did well>"]
}
Use "missing" when the answer does not address a criterion at all. The answer is a spoken, \
speech-to-text transcript: ignore filler, repetition and transcription errors and judge the content. \
Judge ONLY the level and write the notes; do NOT compute any numeric score — the scoring is done outside you. Include exactly one \
entry per criterion, keyed by its id."""

SCORE_USER = """{grounding}QUESTION ({language}):
{question}

=== RUBRIC (grade against every criterion) ===
{rubric}

=== CANDIDATE ANSWER ===
{answer}

Grade the answer now, as one JSON object."""

# The holistic path (a live, ad-hoc question that carries no rubric — D33). Same JSON discipline
# and the same deterministic-aggregation contract: the model returns one overall LEVEL + prose, and
# the numeric score is computed here through RUBRIC_LEVEL_SCORE against a single 'overall'
# criterion. Grading is grounded in the ROLE + CV so it judges the answer for THIS job, not a
# generic notion of a good answer.
SCORE_HOLISTIC_SYSTEM = """You are a strict but fair interview coach. You are given the ROLE and \
company context, the candidate's CV, ONE interview question the interviewer actually asked, and \
the candidate's spoken answer. There is no rubric — judge the answer HOLISTICALLY for this role.

Return a SINGLE JSON object and nothing else — no prose, no markdown fences. Its shape:
{
  "answer_gist": "<FIRST: one or two plain English sentences on what the candidate actually said in \
answer to THIS question — their claim, method or example — ignoring filler and unrelated talk>",
  "overall_level": "excellent | adequate | weak | missing",
  "note": "<one or two sentences justifying the level, grounded in the answer and the role>",
  "concepts_to_refresh": ["<concept the answer was weak or silent on>"],
  "strengths": ["<what the answer did well>"]
}
Grade how well the answer addresses THE QUESTION THAT WAS ASKED. The role, job description and CV \
are context — for relevance and honesty — never extra criteria: do not mark an answer down for not \
mentioning job-description topics the question did not ask about.
Levels (use the full range):
- excellent: answers the question directly and correctly, with something specific (a concrete example, \
number, method or trade-off) — a strong signal for this role.
- adequate: answers correctly but generically, partially, or without structure.
- weak: mostly off the question, incorrect, or too vague to give the interviewer anything.
- missing: no real attempt to answer.
The answer is a LIVE SPEECH-TO-TEXT transcript: ignore filler, repetition, false starts and \
transcription errors, and judge the content, not the prose — never call it incoherent because of \
them. The transcript may run on past the answer into unrelated talk — grade only the part that answers the
question. Write answer_gist BEFORE choosing the level, and base the level on it. An honest "not in
production, but…" is a good answer, not a gap to punish.
Judge ONLY the level and write the prose; do NOT output any numeric score — the scoring is done \
outside you."""

SCORE_HOLISTIC_USER = """{grounding}QUESTION ({language}):
{question}

=== CANDIDATE ANSWER ===
{answer}

Grade the answer now, as one JSON object."""

# The single synthetic criterion a holistic answer is scored against, so it lands on the same
# 0..2 scale as a rubric criterion and the SessionReport totals stay comparable.
HOLISTIC_CRITERION = RubricCriterion(id="overall", label="Overall", weight=1.0)


def _clip(text: str, limit: int) -> str:
    """Trim a bundle section for the grading prompt without paying for the whole thing."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _grounding_block(bundle: ContextBundle | None) -> str:
    """A compact ROLE + JD + CV + honesty-boundary block so grading is anchored to THIS role and
    the candidate's real background (D33), not a generic answer. Returns "" when no bundle is
    given — the block is then omitted from the prompt entirely (backward-compatible)."""
    if bundle is None:
        return ""
    parts: list[str] = []
    header = bundle.role + (f" at {bundle.company}" if bundle.company else "")
    if header.strip():
        parts.append(f"ROLE: {header}")
    if bundle.job_description.strip():
        parts.append("=== JOB DESCRIPTION ===\n" + _clip(bundle.job_description, 1500))
    if bundle.resume.strip():
        parts.append("=== CANDIDATE CV / RESUME ===\n" + _clip(bundle.resume, 1500))
    if bundle.honesty_boundary:
        rows = "\n".join(c.as_prompt_block() for c in bundle.honesty_boundary)
        parts.append("=== HONESTY BOUNDARY (never reward a claim that contradicts these) ===\n" + rows)
    return ("\n\n".join(parts) + "\n\n") if parts else ""


@dataclass
class CriterionScore:
    """One graded criterion. `level` and `note` come from the model; `score` is deterministic —
    RUBRIC_LEVEL_SCORE[level] x weight, computed here, never by the model."""

    id: str
    label: str
    level: str
    weight: float
    score: float
    note: str = ""


@dataclass
class AnswerScore:
    """The grade for one answer. `per_criterion`, `total` and `normalized` are arithmetic over the
    model's per-criterion LEVELS; `concepts_to_refresh` and `strengths` are the model's prose."""

    question: Question
    answer_text: str
    per_criterion: list[CriterionScore]
    total: float
    max_score: float
    concepts_to_refresh: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    reply: LlmReply | None = None

    @property
    def normalized(self) -> float:
        """total / max_score in [0, 1]; 0.0 when the rubric has no scorable weight."""
        return self.total / self.max_score if self.max_score else 0.0


def _rubric_block(rubric: Rubric) -> str:
    """Render the rubric so the model grades against the same yardstick a human would read."""
    if not rubric.criteria:
        return "(no criteria — grade holistically as 'adequate' if the answer is on-topic)"
    lines: list[str] = []
    for c in rubric.criteria:
        lines.append(f"- [{c.id}] {c.label} (weight {c.weight:g})")
        for level, descriptor in c.levels.items():
            if descriptor:
                lines.append(f"    {level}: {descriptor}")
    return "\n".join(lines)


def _aggregate(rubric: Rubric, graded_by_id: dict[str, dict]) -> tuple[list[CriterionScore], float]:
    """The deterministic core: map each criterion's model-returned LEVEL through
    RUBRIC_LEVEL_SCORE x weight and sum. A criterion the model did not grade (or graded with an
    unknown level) scores as 'missing' (0.0) — the arithmetic never guesses."""
    per: list[CriterionScore] = []
    total = 0.0
    for c in rubric.criteria:
        graded = graded_by_id.get(c.id, {})
        level = str(graded.get("level", "missing")).strip().lower()
        if level not in RUBRIC_LEVEL_SCORE:
            level = "missing"
        score = RUBRIC_LEVEL_SCORE[level] * c.weight
        total += score
        per.append(CriterionScore(
            id=c.id, label=c.label, level=level, weight=c.weight, score=score,
            note=str(graded.get("note", "")).strip(),
        ))
    return per, total


def _as_str_list(value: object) -> list[str]:
    """Coerce a model field to a clean list of non-empty strings (it may be a str or absent)."""
    if isinstance(value, str):
        value = [value]
    return [str(v).strip() for v in (value or []) if str(v).strip()]


def score_answer(
    question: Question, answer_text: str, bundle: ContextBundle | None = None,
    backend: str | None = None, model: str | None = None,
) -> AnswerScore:
    """Grade one answer with EXACTLY ONE model call (a malformed reply costs one repair retry).
    Two paths, both deterministic in their arithmetic (the model only returns LEVELS + prose):

    - **Rubric** (the question carries criteria — the D30 practice track): grade against every
      criterion; `_aggregate` maps each level through RUBRIC_LEVEL_SCORE * weight.
    - **Holistic** (no rubric — a live, ad-hoc interviewer question, D33): grade against one
      synthetic 'overall' criterion so a captured answer still lands on the same 0..2 scale.

    When `bundle` is given, the grade is grounded in the ROLE + JD + CV + honesty boundary."""
    chosen = resolve_backend(backend or settings.SCORING_BACKEND)
    use_model = model or settings.SCORING_MODEL or None
    grounding = _grounding_block(bundle)
    holistic = not question.rubric.criteria
    if holistic:
        system = SCORE_HOLISTIC_SYSTEM
        user = SCORE_HOLISTIC_USER.format(
            grounding=grounding, language=question.language,
            question=question.question or "(question missing)",
            answer=answer_text.strip() or "(no answer given)",
        )
    else:
        system = SCORE_SYSTEM
        user = SCORE_USER.format(
            grounding=grounding, language=question.language,
            question=question.question or "(question missing)",
            rubric=_rubric_block(question.rubric),
            answer=answer_text.strip() or "(no answer given)",
        )
    reply = llm_call(user, system=system, backend=chosen, model=use_model,
                     max_tokens=settings.SCORING_MAX_TOKENS, timeout=settings.SCORING_TIMEOUT_SECONDS)
    logger.info("score call | %s%s", "holistic | " if holistic else "", reply.cost_line())
    try:
        parsed = parse_json_object(reply.text)
    except ValueError as first_error:
        logger.warning("grade reply was not valid JSON (%s) — one repair call", first_error)
        repaired = llm_call(REPAIR_PROMPT.format(bad=reply.text), system=None,
                            backend=chosen, model=use_model, max_tokens=settings.SCORING_MAX_TOKENS,
                            timeout=settings.SCORING_TIMEOUT_SECONDS)
        parsed = parse_json_object(repaired.text)  # ValueError here propagates — a hard stop

    if holistic:
        rubric = Rubric(criteria=[HOLISTIC_CRITERION])
        level = str(parsed.get("overall_level", "missing")).strip().lower()
        graded_by_id = {"overall": {"level": level, "note": str(parsed.get("note", "")).strip()}}
    else:
        rubric = question.rubric
        graded_by_id = {
            str(item.get("id")): item
            for item in (parsed.get("per_criterion") or [])
            if isinstance(item, dict) and item.get("id") is not None
        }
    per, total = _aggregate(rubric, graded_by_id)
    return AnswerScore(
        question=question, answer_text=answer_text.strip(), per_criterion=per,
        total=total, max_score=rubric.max_score(),
        concepts_to_refresh=_as_str_list(parsed.get("concepts_to_refresh")),
        strengths=_as_str_list(parsed.get("strengths")),
        reply=reply,
    )


# --------------------------------------------------------------------------
# 4. Session scoring + report
# --------------------------------------------------------------------------
@dataclass
class SessionReport:
    """The batch-scored session: per-answer grades plus deterministic session totals and a
    markdown render. `path` is set once the report is written to disk."""

    session_id: str
    role: str
    company: str
    answers: list[AnswerScore]
    generated_at: str
    run_id: str
    path: Path | None = None
    report_kind: str = "Mock interview"  # "Interview review" for the live post-interview path (D32)

    @property
    def total(self) -> float:
        return sum(a.total for a in self.answers)

    @property
    def max_score(self) -> float:
        return sum(a.max_score for a in self.answers)

    @property
    def normalized(self) -> float:
        return self.total / self.max_score if self.max_score else 0.0

    def to_markdown(self) -> str:
        pct = f"{self.normalized * 100:.0f}%" if self.answers else "n/a"
        header = self.role + (f" at {self.company}" if self.company else "")
        lines: list[str] = [
            f"# {self.report_kind} report — {self.session_id}",
            "",
            f"Generated {self.generated_at} UTC · run `{self.run_id}`",
        ]
        if header.strip():
            lines.append(f"Role: {header}")
        lines += [
            "",
            f"**Overall: {self.total:g} / {self.max_score:g} ({pct})** across "
            f"{len(self.answers)} answered question(s).",
            "",
        ]
        for i, ans in enumerate(self.answers, start=1):
            comp = f" — {ans.question.competency}" if ans.question.competency else ""
            a_pct = f"{ans.normalized * 100:.0f}%" if ans.max_score else "n/a"
            lines += [
                f"## Q{i}{comp}",
                f"> {ans.question.question}",
                "",
                f"**Answer:** {ans.answer_text or '(no answer given)'}",
                "",
                f"**Score: {ans.total:g} / {ans.max_score:g} ({a_pct})**",
            ]
            for c in ans.per_criterion:
                note = f" — {c.note}" if c.note else ""
                lines.append(f"- `{c.level}` {c.label} (weight {c.weight:g}, {c.score:g}){note}")
            if ans.concepts_to_refresh:
                lines.append(f"- Concepts to refresh: {', '.join(ans.concepts_to_refresh)}")
            if ans.strengths:
                lines.append(f"- Strengths: {', '.join(ans.strengths)}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def write_report(report: SessionReport, output_dir: Path | None = None,
                 name_prefix: str = "training_report") -> Path:
    """Write the report markdown to `<output_dir>/<name_prefix>_<stamp>.md`
    (write-temp-then-rename). Local-only per SI1 — nothing leaves the machine."""
    root = output_dir or OUTPUT_DIR
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{name_prefix}_{report.run_id}.md"
    tmp = target.with_suffix(".md.tmp")
    tmp.write_text(report.to_markdown(), encoding="utf-8")
    tmp.replace(target)
    report.path = target
    logger.info("wrote %s", target)
    return target


def score_session(
    session: TrainingSession, bundle: ContextBundle | None = None,
    backend: str | None = None, model: str | None = None,
    output_dir: Path | None = None, write: bool = True,
    report_kind: str = "Mock interview", name_prefix: str = "training_report",
) -> SessionReport:
    """Score every answered question in `session` (one model call each) into a SessionReport, and
    (by default) write it to scripts/outputs/<name_prefix>_<stamp>.md. Unanswered questions are
    skipped — a blank answer is not a graded one. `report_kind` names the report ("Mock interview"
    for the D30 trainer, "Interview review" for the live post-interview path, D32)."""
    stamp = datetime.now(timezone.utc)
    run_id = f"{stamp:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"  # TRK
    role = bundle.role if bundle else ""
    company = bundle.company if bundle else ""
    session_id = bundle.session_id if bundle else "training"
    logger.info("run %s START | scoring %d answered question(s) | session=%s",
                run_id, len(session.answered()), session_id)

    answers = [
        score_answer(question, text, bundle, backend=backend, model=model)
        for question, text in session.answered()
    ]
    report = SessionReport(
        session_id=session_id, role=role, company=company, answers=answers,
        generated_at=f"{stamp:%Y-%m-%d %H:%M:%S}", run_id=run_id, report_kind=report_kind,
    )
    if write:
        write_report(report, output_dir, name_prefix=name_prefix)
    logger.info("run %s END | %.3g / %.3g total", run_id, report.total, report.max_score)
    return report


# --------------------------------------------------------------------------
# --selftest — NO MODEL: the deterministic aggregation math + rubric parsing
# --------------------------------------------------------------------------
def _selftest() -> int:
    """Prove the deterministic core with fixtures and no model call: level -> score x weight,
    the session total, normalization, `max_score()`, and rubric parsing. Mirrors the exit-code
    contract of `reasoning.py --selftest-*`: 0 = pass, non-zero on any regression."""
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    # A rubric with unequal weights, so a wrong weighting cannot hide behind equal ones.
    rubric = Rubric(criteria=[
        RubricCriterion(id="depth", label="Technical depth", weight=2.0),
        RubricCriterion(id="tradeoff", label="Trade-off awareness", weight=1.0),
        RubricCriterion(id="clarity", label="Clarity", weight=0.5),
    ])
    # max: (2.0 + 1.0 + 0.5) * 2.0 (excellent) = 7.0
    check(rubric.max_score() == 7.0, f"max_score expected 7.0, got {rubric.max_score()}")

    # (a) mixed levels through _aggregate.
    graded = {
        "depth": {"level": "excellent", "note": "named recall@k"},   # 2.0 * 2.0 = 4.0
        "tradeoff": {"level": "adequate"},                            # 1.0 * 1.0 = 1.0
        "clarity": {"level": "weak"},                                 # 0.0 * 0.5 = 0.0
    }
    per, total = _aggregate(rubric, graded)
    check(abs(total - 5.0) < 1e-9, f"mixed total expected 5.0, got {total}")
    check(per[0].score == 4.0, f"depth score expected 4.0, got {per[0].score}")
    check(per[1].score == 1.0, f"tradeoff score expected 1.0, got {per[1].score}")
    check(per[2].score == 0.0, f"clarity score expected 0.0, got {per[2].score}")

    # (b) all-excellent tops out at max_score exactly; normalization is 1.0.
    _, top = _aggregate(rubric, {c.id: {"level": "excellent"} for c in rubric.criteria})
    check(top == rubric.max_score(), f"all-excellent expected {rubric.max_score()}, got {top}")

    # (c) an ungraded / unknown-level criterion scores as 'missing' (0.0), never guessed.
    per_missing, miss_total = _aggregate(rubric, {"depth": {"level": "banana"}})
    check(miss_total == 0.0, f"unknown+ungraded total expected 0.0, got {miss_total}")
    check(per_missing[0].level == "missing", f"unknown level expected 'missing', got {per_missing[0].level}")

    # (d) normalization over a whole answer.
    check(abs((5.0 / 7.0) - (total / rubric.max_score())) < 1e-9, "normalization mismatch")

    # (e) rubric PARSING: a raw question dict parses into the typed shape with correct max_score.
    raw = {
        "id": "q1", "competency": "RAG", "question": "How do you evaluate retrieval?",
        "rubric": {"criteria": [
            {"id": "depth", "label": "Depth", "weight": 3.0,
             "levels": {"excellent": "x", "adequate": "y", "weak": "z"}},
            {"id": "clarity", "label": "Clarity", "weight": 1.0, "levels": {}},
        ]},
    }
    q = _parse_question(0, raw)
    check(isinstance(q, Question), "parsed object is not a Question")
    check(len(q.rubric.criteria) == 2, f"expected 2 criteria, got {len(q.rubric.criteria)}")
    check(q.rubric.max_score() == 8.0, f"parsed max_score expected 8.0, got {q.rubric.max_score()}")
    check(q.rubric.criteria[0].weight == 3.0, "parsed weight lost")

    # (f) HOLISTIC aggregation (D33): a rubric-less answer scores on one synthetic 'overall'
    # criterion, on the same 0..2 scale — the same deterministic core, no model.
    holistic_rubric = Rubric(criteria=[HOLISTIC_CRITERION])
    check(holistic_rubric.max_score() == 2.0, f"holistic max expected 2.0, got {holistic_rubric.max_score()}")
    per_h, tot_h = _aggregate(holistic_rubric, {"overall": {"level": "adequate", "note": "ok"}})
    check(tot_h == 1.0, f"holistic adequate expected 1.0, got {tot_h}")
    check(per_h[0].id == "overall" and per_h[0].note == "ok", "holistic criterion mis-shaped")
    _, tot_missing = _aggregate(holistic_rubric, {"overall": {"level": "nonsense"}})
    check(tot_missing == 0.0, f"holistic unknown level expected 0.0, got {tot_missing}")

    # (g) the grounding block is present iff a bundle is given, and omitted otherwise.
    check(_grounding_block(None) == "", "grounding block should be empty without a bundle")

    print("training.py --selftest — deterministic aggregation + rubric parsing + holistic")
    if failures:
        print(f"  FAIL ({len(failures)}):")
        for line in failures:
            print(f"    - {line}")
        return 1
    print("  OK — all aggregation, normalization and parsing checks pass")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _load_answers(path: Path) -> list[dict]:
    """Read a fixture answers file: either a JSON list of {id?, question?, answer} entries, or an
    object with an `answers` list of the same."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc.get("answers") if isinstance(doc, dict) else doc
    return [r for r in (rows or []) if isinstance(r, dict)]


def _session_from_answers(bundle: ContextBundle, rows: list[dict],
                          backend: str | None, model: str | None) -> TrainingSession:
    """Build a TrainingSession over the bundle's questions and fill each answer buffer from `rows`,
    matching an answer to a question by `id` when present, else by position."""
    questions = generate_questions(bundle, backend=backend, model=model)
    session = TrainingSession(questions)
    by_id = {q.id: i for i, q in enumerate(questions)}
    for pos, row in enumerate(rows):
        answer = str(row.get("answer", "")).strip()
        if not answer:
            continue
        index = by_id.get(str(row.get("id"))) if row.get("id") is not None else pos
        if index is None or not (0 <= index < len(questions)):
            logger.warning("answer #%d matches no question (id=%r) — skipped", pos, row.get("id"))
            continue
        session.seek(index)  # position the pointer, then append the whole answer as one segment
        session.add_answer_segment(answer)
    session.finish()
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--selftest", action="store_true",
                        help="no-model self-test of the deterministic aggregation math; 0 = pass")
    parser.add_argument("--session", help="session id under scripts/inputs/sessions/ (or a path)")
    parser.add_argument("--answers", help="JSON file of {id?, answer} rows to grade against the bundle")
    parser.add_argument("--backend", choices=("local", "cloud"), help="override SCORING_BACKEND")
    parser.add_argument("--model", help="override the scoring model")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if not args.session or not args.answers:
        parser.error("--session and --answers are required (or use --selftest)")

    try:
        bundle = load_bundle(args.session)
        rows = _load_answers(Path(args.answers))
        session = _session_from_answers(bundle, rows, args.backend, args.model)
        report = score_session(session, bundle, backend=args.backend, model=args.model)
    except BackendUnavailable:
        logger.exception("scoring backend unavailable")
        sys.exit(1)
    except (ValueError, FileNotFoundError) as error:
        logger.error("scoring failed: %s", error)
        sys.exit(1)

    print(f"\nwrote {report.path}")
    print(f"  {report.session_id} | {len(report.answers)} answer(s) | "
          f"{report.total:g} / {report.max_score:g} ({report.normalized * 100:.0f}%)")


if __name__ == "__main__":
    main()
