"""Turn a raw job description into a schema-v2 context bundle (D30, offline practice track).

`generate_context.py` authors ONLY the interview-side sections of a `bundle.json`:
`job_description` (responsibilities / must-haves / nice-to-haves), a short `company_brief`,
the competency arc as a `plan[]` of PlanStep-shaped steps, and a `question_bank[]` of N
questions per competency, each with a weighted `rubric` (the D30 `Question`/`Rubric`/
`RubricCriterion` shape that `reasoning.load_bundle` parses and `training.py` grades against).
The same generation call also returns a `role` guess derived from the JD (#659) — used only when
`--role` is omitted; passing `--role` always wins, and a missing/blank guess leaves `role` empty
exactly as before (no crash, no second LLM call).

It NEVER writes the candidate's own content. `resume`, `answer_bank` and `honesty_boundary`
are emitted as `{"status": "placeholder"}` so `load_bundle` flags them in `.placeholders` and
`warn_lines()` prints them — the tool must not synthesize the candidate's experience (D22). The
user pastes those in by hand (or reuses a real session's files) before an interview.

Only the *judgement* is an LLM call (via `scripts.llm_client.llm_call`, local backend by default
per SI1); the rubric SCORING SHAPE is fixed and applied deterministically here (CLAUDE.md
Determinism First), so a malformed or fenced model reply is stripped, parsed, and repaired with
ONE retry before failing with a pointed error rather than writing a half-built bundle.

Run it:
    python scripts/generate_context.py --jd path/to/jd.md --session acme_20260917 --role "ML Engineer"
    python scripts/generate_context.py --jd "<pasted JD text>" --session acme --company "Acme" \
        --company-notes notes.md --questions-per-competency 4 --backend local
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from scripts.llm_client import BackendUnavailable, llm_call, model_for, resolve_backend  # noqa: E402
from scripts.doc_text import SUPPORTED_SUFFIXES, read_path  # noqa: E402
from scripts.logger import get_logger  # noqa: E402
from scripts.reasoning import RUBRIC_LEVELS, SESSIONS_DIR, load_bundle  # noqa: E402

logger = get_logger("generate_context")

# The candidate-side sections this tool must NEVER author (D22). Written as placeholder markers
# so load_bundle flags them; the user fills them in before the interview.
CANDIDATE_SECTIONS = ("resume", "answer_bank", "honesty_boundary")
PLACEHOLDER = {"status": "placeholder"}
LANGUAGE_LABELS = {"pl": "Polish", "en": "English"}

SYSTEM_PROMPT = """You are an interview-prep assistant. Given a job description, you author the \
INTERVIEWER'S side of a mock-interview kit for a candidate: what the role demands, a brief on the \
company, the arc of competencies a panel would probe, and graded practice questions.

You NEVER invent anything about the candidate — no resume, no past projects, no achievements. You \
only describe the ROLE and the QUESTIONS an interviewer would ask about it.

Return a SINGLE JSON object and nothing else — no prose, no markdown fences. Its shape:
{
  "role": "<the job title this JD implies, e.g. 'Senior Backend Engineer'; empty string if the \
JD genuinely does not name or imply one>",
  "job_description": {
    "responsibilities": ["..."],
    "must_haves": ["..."],
    "nice_to_haves": ["..."]
  },
  "company_brief": "2-4 sentences about the company and the team, grounded in the JD (and any \
company notes provided). Say 'unknown' rather than inventing facts.",
  "plan": [
    {"id": "p1", "title": "<interview stage or competency>", "probe": false,
     "key_points": ["what a strong candidate shows here"],
     "done_signals": ["short phrases that show THIS step is being discussed"]}
  ],
  "question_bank": [
    {
      "competency": "<must match the title of a plan step with probe=true>",
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
The plan is the ARC OF THE INTERVIEW, EXACTLY 5 steps in the order a panel covers them: an opening \
step (introduction / background, probe=false), then the 3 competency steps the JD demands most \
(probe=true), then a closing step (the candidate's questions / next steps, probe=false). Keep the \
opening and closing steps short (one key point each). Every step has 4-7 done_signals: short words or phrases \
an interviewer or candidate would literally SAY when that step is under way. AT LEAST 3 of them must \
be in the interview's SPOKEN LANGUAGE (given below) — everyday words people use in that step, e.g. \
Polish "opowie", "doświadcz", "projekt", "pytania" — and the rest may be English technical terms \
people say as-is (e.g. "RAG", "agent"). Do NOT just copy phrases from the job description. They are \
matched as case-insensitive substrings of the live transcript, so keep them short and concrete (no \
full sentences) and prefer word STEMS for inflected languages (e.g. "projekt", "agent", "doświadcz").
Author EXACTLY {n} question(s) for EACH probe=true step, and none for the other steps. Each rubric \
needs 2-3 weighted criteria with concrete 'excellent' / 'adequate' / 'weak' descriptors of AT MOST \
15 words each (the whole kit must stay compact)."""

USER_PROMPT = """ROLE: {role}
COMPANY: {company}
INTERVIEW SPOKEN LANGUAGE (for done_signals): {spoken}
QUESTIONS PER COMPETENCY: {n}
{company_notes}
=== JOB DESCRIPTION ===
{jd}

Author the interviewer-side kit now, as one JSON object."""

REPAIR_PROMPT = """The following was supposed to be a single JSON object but did not parse:

{bad}

Return the corrected content as ONE valid JSON object only — no markdown fences, no commentary, \
no trailing text. Preserve all of the content; only fix the JSON syntax and structure."""


# --------------------------------------------------------------------------
# Robust JSON handling (strip fences, parse, repair once)
# --------------------------------------------------------------------------
def strip_code_fences(text: str) -> str:
    """Drop a leading ```json / ``` fence and its closing ``` if the model wrapped its reply."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # Drop the opening fence line (```), which may carry a language tag (```json).
    lines = lines[1:]
    # Drop the closing fence if present.
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_object(text: str) -> dict:
    """Parse `text` into a dict, tolerating a fenced reply and trailing prose.

    Raises ValueError (never returns a non-dict) so the caller can decide to repair-and-retry.
    """
    candidate = strip_code_fences(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Last resort before giving up on this reply: carve out the outermost {...} span and
        # try that. Handles a model that appended a sentence after the object.
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("model reply contains no JSON object")
        parsed = json.loads(candidate[start : end + 1])  # may raise JSONDecodeError -> ValueError below
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _call_model(prompt: str, system: str | None, backend: str, model: str | None) -> str:
    reply = llm_call(
        prompt, system=system, backend=backend, model=model,
        max_tokens=settings.GENERATE_MAX_TOKENS, timeout=settings.GENERATE_TIMEOUT_SECONDS,
    )
    logger.info("generation call | %s", reply.cost_line())
    return reply.text


def generate_sections(
    jd: str, role: str, company: str, company_notes: str,
    n_per_competency: int, backend: str, model: str | None, spoken_language: str = "pl",
) -> dict:
    """One metered generation call, with ONE JSON-repair retry on malformed output.

    Returns the parsed model object (interview-side sections only). Raises BackendUnavailable if
    the backend is down and ValueError with a pointed message if even the repaired reply is not
    valid JSON — a half-parsed bundle is worse than a clear stop.
    """
    notes_block = f"COMPANY NOTES (user-provided, enrich the brief with these):\n{company_notes}\n" if company_notes else ""
    system = SYSTEM_PROMPT.replace("{n}", str(n_per_competency))
    user = USER_PROMPT.format(
        role=role or "(unspecified)", company=company or "(unspecified)",
        n=n_per_competency, company_notes=notes_block, jd=jd,
        spoken=LANGUAGE_LABELS.get(spoken_language, spoken_language or "Polish"),
    )
    raw = _call_model(user, system, backend, model)
    try:
        return parse_json_object(raw)
    except ValueError as first_error:
        logger.warning("model reply was not valid JSON (%s) — attempting one repair call", first_error)
        repaired = _call_model(REPAIR_PROMPT.format(bad=raw), None, backend, model)
        try:
            return parse_json_object(repaired)
        except ValueError as second_error:
            raise ValueError(
                f"could not obtain valid JSON from the model after one repair "
                f"(first: {first_error}; after repair: {second_error}). Re-run, or try "
                f"--backend cloud for a stronger model."
            ) from second_error


# --------------------------------------------------------------------------
# Deterministic shaping — the rubric SCORING shape is fixed; only text is the LLM's
# --------------------------------------------------------------------------
def _render_job_description(jd_section: object) -> str:
    """Render the structured JD block into the markdown text `load_bundle` reads. A model that
    returned a plain string is used verbatim."""
    if isinstance(jd_section, str):
        return jd_section.strip()
    if not isinstance(jd_section, dict):
        return ""
    headings = [
        ("responsibilities", "Responsibilities"),
        ("must_haves", "Must-haves"),
        ("nice_to_haves", "Nice-to-haves"),
    ]
    parts: list[str] = []
    for key, title in headings:
        items = jd_section.get(key) or []
        if not items:
            continue
        parts.append(f"## {title}")
        parts.extend(f"- {str(item).strip()}" for item in items)
    return "\n".join(parts).strip()


def _normalize_rubric(rubric_raw: object) -> dict:
    """Enforce the fixed rubric shape: a list of criteria, each with id/label/weight and a
    `levels` dict carrying exactly the RUBRIC_LEVELS keys. Missing descriptors become "" so the
    grader always sees the same yardstick; the descriptive TEXT is whatever the model wrote."""
    criteria_raw = (rubric_raw or {}).get("criteria", []) if isinstance(rubric_raw, dict) else []
    criteria: list[dict] = []
    for j, crit in enumerate(criteria_raw):
        if not isinstance(crit, dict):
            continue
        levels_raw = crit.get("levels") or {}
        levels = {level: str(levels_raw.get(level, "")).strip() for level in RUBRIC_LEVELS}
        try:
            weight = float(crit.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        criteria.append({
            "id": str(crit.get("id") or f"c{j + 1}"),
            "label": str(crit.get("label", "")).strip(),
            "weight": weight,
            "levels": levels,
        })
    return {"criteria": criteria}


def _normalize_question_bank(questions_raw: object, default_language: str) -> list[dict]:
    """Assign stable question ids and enforce the fixed rubric shape on every question. Text
    content (the question, the competency label, the level descriptors) stays the model's."""
    questions: list[dict] = []
    for i, q in enumerate(questions_raw or []):
        if not isinstance(q, dict):
            continue
        questions.append({
            "id": str(q.get("id") or f"q{i + 1}"),
            "competency": str(q.get("competency", "")).strip(),
            "question": str(q.get("question", "")).strip(),
            "language": str(q.get("language") or default_language),
            "rubric": _normalize_rubric(q.get("rubric")),
        })
    return questions


def _normalize_plan(plan_raw: object) -> list[dict]:
    plan: list[dict] = []
    for i, step in enumerate(plan_raw or []):
        if not isinstance(step, dict):
            continue
        plan.append({
            "id": str(step.get("id") or f"c{i + 1}"),
            "title": str(step.get("title", "")).strip(),
            "key_points": [str(p).strip() for p in (step.get("key_points") or [])],
            # F2 (#1272): without done_signals the live plan rail can never mark a step "mentioned".
            # Short, de-duplicated, lower-cased phrases; a missing list degrades to [] (old behaviour).
            "done_signals": _clean_signals(step.get("done_signals")),
        })
    # A signal listed under two or more steps says nothing about WHICH step is under way (the model
    # likes generic words like "rozwiązanie" / "system") — drop it everywhere. Deterministic.
    counts: dict[str, int] = {}
    for step in plan:
        for s in step["done_signals"]:
            counts[s] = counts.get(s, 0) + 1
    for step in plan:
        step["done_signals"] = [s for s in step["done_signals"] if counts[s] == 1]
    return plan


def _clean_signals(raw: object) -> list[str]:
    out: list[str] = []
    for s in raw if isinstance(raw, list) else []:
        s = str(s).strip().strip(".").lower()
        if s and len(s) <= 40 and s not in out:
            out.append(s)
    return out[:8]


def build_bundle(
    sections: dict, session_id: str, role: str, company: str,
    spoken_language: str, suggestion_language: str, jd_source: str,
    resume_text: str = "", resume_source: str = "",
) -> dict:
    """Assemble the schema-v2 bundle dict: generated interview-side sections + candidate-side
    placeholders. The candidate sections are markers, not empty content, so load_bundle flags
    them (D22)."""
    default_lang = spoken_language or "en"
    # `role` wins whenever the caller passed one explicitly (`--role`); otherwise fall back to
    # whatever the model derived from the JD (D30 extension, #659) — never a second LLM call,
    # just another key on the same generation contract. A missing/empty derived role degrades to
    # "" (the pre-#659 behaviour), never a crash.
    derived_role = sections.get("role")
    resolved_role = role or (str(derived_role).strip() if isinstance(derived_role, str) else "")
    bundle: dict = {
        "schema_version": 2,
        "session_id": session_id,
        "role": resolved_role,
        "company": company,
        "generated_by": "generate_context.py (D30)",
        "jd_source": jd_source,
        "language": {"spoken": spoken_language, "suggestions": suggestion_language},
        # Interview-side, authored by the model (deterministically shaped).
        "job_description": _render_job_description(sections.get("job_description")),
        "company_brief": str(sections.get("company_brief", "")).strip(),
        "plan": _normalize_plan(sections.get("plan")),
        "question_bank": _normalize_question_bank(sections.get("question_bank"), default_lang),
        # Candidate-side — NEVER generated (D22). Placeholders so warn_lines() flags them.
        # The user's OWN CV, copied verbatim when supplied (D30 amendment, D37) — real content, never
        # generated; without one it stays a flagged placeholder.
        "resume": ({"text": resume_text.strip(), "source": resume_source or "provided"}
                   if resume_text.strip() else dict(PLACEHOLDER)),
        "answer_bank": dict(PLACEHOLDER),
        "honesty_boundary": dict(PLACEHOLDER),
    }
    return bundle


def write_bundle(bundle: dict, session_id: str, sessions_dir: Path | None = None) -> Path:
    """Write `<sessions_dir>/<session_id>/bundle.json` (write-temp-then-rename)."""
    root = sessions_dir or SESSIONS_DIR
    session_dir = root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    target = session_dir / "bundle.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _resolve_jd(value: str) -> tuple[str, str]:
    """`--jd` / `--cv` is a path when it points at a file (PDF, .txt, .md — read by doc_text), otherwise
    the text itself. Returns (text, source-label)."""
    path = Path(value)
    if len(value) < 4096 and path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
        return read_path(path).strip(), str(path)          # PDF / .txt / .md (D37)
    if len(value) < 4096 and path.is_file():
        return path.read_text(encoding="utf-8").strip(), str(path)
    return value.strip(), "inline text"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--jd", required=True, help="job description: a path to a file, or the JD text itself")
    parser.add_argument("--session", required=True, help="session id; the bundle is written under scripts/inputs/sessions/<id>/")
    parser.add_argument("--company", default="", help="company name (also used to steer the company_brief)")
    parser.add_argument("--role", default="", help="role title (defaults to what the JD implies if omitted)")
    parser.add_argument("--cv", default="", help="your CV: a path (PDF/.txt/.md) or the text itself — copied verbatim as the bundle's resume, never generated")
    parser.add_argument("--company-notes", default="", help="optional user-pasted company notes (path or text) to enrich the brief")
    parser.add_argument(
        "--questions-per-competency", type=int, default=settings.GENERATE_QUESTIONS_PER_COMPETENCY,
        help="how many practice questions to author per competency (default from settings)",
    )
    parser.add_argument("--backend", choices=("local", "cloud"), help="override GENERATE_BACKEND (default: local, SI1)")
    parser.add_argument("--model", help="override the backend's model")
    parser.add_argument("--spoken-language", default="pl", help="the interview's spoken language for the bundle's language block")
    parser.add_argument("--suggestion-language", default="match", help="the bundle's suggestion language (match|en|pl)")
    return parser


def run(args: argparse.Namespace, sessions_dir: Path | None = None) -> Path:
    """Generate + write one bundle. Split from main() so a test can drive it with a mocked
    llm_call and a tmp_path sessions_dir, no argparse and no network."""
    jd_text, jd_source = _resolve_jd(args.jd)
    if not jd_text:
        raise ValueError("--jd resolved to empty text (missing file or empty string)")
    company_notes, _ = _resolve_jd(args.company_notes) if args.company_notes else ("", "")

    backend = resolve_backend(args.backend or settings.GENERATE_BACKEND)
    model = args.model or settings.GENERATE_MODEL or None
    n = max(1, int(args.questions_per_competency))

    logger.info(
        "generating bundle for session %s | backend=%s model=%s | %d question(s)/competency",
        args.session, backend, model or model_for(backend), n,
    )
    sections = generate_sections(jd_text, args.role, args.company, company_notes, n, backend, model,
                                 spoken_language=args.spoken_language)
    cv_value = getattr(args, "cv", "") or ""
    resume_text, resume_source = _resolve_jd(cv_value) if cv_value else ("", "")
    bundle = build_bundle(
        sections, args.session, args.role, args.company,
        args.spoken_language, args.suggestion_language, jd_source,
        resume_text=resume_text, resume_source=resume_source,
    )
    target = write_bundle(bundle, args.session, sessions_dir)
    logger.info("wrote %s", target)
    return target


def main() -> None:
    args = build_parser().parse_args()
    try:
        target = run(args)
    except BackendUnavailable:
        logger.exception("generation backend unavailable")
        sys.exit(1)
    except ValueError as error:
        logger.error("generation failed: %s", error)
        sys.exit(1)

    # Re-load through the real reader so the summary reflects exactly what a live session will see,
    # and so the candidate-side placeholder warnings are printed the same way (D22).
    bundle = load_bundle(args.session)
    print(f"\nwrote {target}")
    print(
        f"  role={bundle.role!r} company={bundle.company!r} | "
        f"{len(bundle.plan)} competency step(s), {len(bundle.question_bank)} practice question(s)"
    )
    crit_total = sum(len(q.rubric.criteria) for q in bundle.question_bank)
    print(f"  rubric criteria across the bank: {crit_total}")
    warnings = bundle.warn_lines()
    if warnings:
        print("  candidate-side sections left as placeholders (fill these before the interview):")
        for line in warnings:
            print(f"    {line}")


if __name__ == "__main__":
    main()
