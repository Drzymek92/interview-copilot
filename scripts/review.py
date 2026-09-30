"""Post-interview REVIEW (D32): grade the captured question↔answer pairs from a REAL interview
against the candidate's CV + the job info, and write a feedback report.

The reviewer is the SAME machinery the offline practice track (D30) grades mock answers with —
`training.score_session` / `score_answer` / `SessionReport` — reused, not reimplemented (D33 extends
it to ground on the CV + JD and to grade rubric-less answers holistically). The only difference is
the input: a live interview's questions are ad-hoc and carry no rubric, so each pair is graded
HOLISTICALLY (grounded in the ROLE + JD + CV + honesty boundary). `--synthesize-rubrics` opts into
authoring a rubric per captured question first, for structured per-criterion scores. This CLI is one
of the two review entry points (D34); the other is the dashboard's 'Review answers' button.

Capture side: `dashboard.py`'s `QaPairer` appends one JSON line per salient interviewer question
(a D23 gate fire) + the candidate's following answer to `interview_qa_<stamp>.jsonl` during the
call. This script reviews that file afterwards. Everything is local-only (SI1) — the pairs, the
scoring (local Ollama by default), and the report never leave the machine.

Run it:
    python scripts/review.py --selftest                        # no model — the adapter + shaping
    python scripts/review.py --session example_ai_engineer             # newest interview_qa_*.jsonl
    python scripts/review.py --session example_ai_engineer --qa scripts/outputs/interview_qa_<stamp>.jsonl
    python scripts/review.py --session example_ai_engineer --synthesize-rubrics
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from scripts.generate_context import _normalize_rubric, parse_json_object  # noqa: E402
from scripts.llm_client import BackendUnavailable, llm_call, resolve_backend  # noqa: E402
from scripts.logger import get_logger  # noqa: E402
from scripts.reasoning import (  # noqa: E402
    ContextBundle, Question, Rubric, _parse_question, load_bundle,
)
from scripts.training import TrainingSession, score_session  # noqa: E402

logger = get_logger("review")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "outputs"

# The opt-in rubric author. Deterministic SHAPE (fixed by _normalize_rubric); only the text is the
# model's. One call per captured question; a failure falls back to holistic for that question.
RUBRIC_SYSTEM = """You author a compact scoring rubric for ONE interview question. Return a SINGLE \
JSON object and nothing else — no prose, no fences:
{"criteria": [{"id": "depth", "label": "Technical depth", "weight": 1.0,
  "levels": {"excellent": "...", "adequate": "...", "weak": "..."}}]}
Author 2-3 weighted criteria with concrete 'excellent' / 'adequate' / 'weak' descriptors, tuned to \
THIS role and question."""

RUBRIC_USER = """ROLE: {role}
QUESTION: {question}

Author the rubric now, as one JSON object."""


# --------------------------------------------------------------------------
# Q&A pair loading
# --------------------------------------------------------------------------
def load_qa_pairs(path: Path) -> list[dict]:
    """Read an `interview_qa_<stamp>.jsonl` file: one JSON object per line, tolerant of blank or
    malformed lines (a capture log is best-effort — a single bad line must not sink the review)."""
    pairs: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("skipping malformed Q&A line %d in %s", lineno, path.name)
            continue
        if isinstance(obj, dict) and str(obj.get("question", "")).strip():
            pairs.append(obj)
    return pairs


def newest_qa_log(output_dir: Path | None = None) -> Path | None:
    """The most recent `interview_qa_*.jsonl` — the default target when `--qa` is omitted."""
    root = output_dir or OUTPUT_DIR
    logs = sorted(root.glob("interview_qa_*.jsonl"))
    return logs[-1] if logs else None


# --------------------------------------------------------------------------
# pairs -> a scored session (reusing training.score_session)
# --------------------------------------------------------------------------
def _synthesize_rubric(question_text: str, role: str, backend: str, model: str | None) -> Rubric:
    """Author a rubric for one captured question (opt-in). Falls back to an empty rubric (→ the
    holistic path) if the model reply cannot be parsed — never a hard stop."""
    user = RUBRIC_USER.format(role=role or "the role", question=question_text)
    try:
        reply = llm_call(user, system=RUBRIC_SYSTEM, backend=backend, model=model,
                         max_tokens=settings.SCORING_MAX_TOKENS, timeout=settings.SCORING_TIMEOUT_SECONDS)
        parsed = parse_json_object(reply.text)
    except (ValueError, BackendUnavailable):
        logger.warning("rubric synthesis failed for a question — grading it holistically instead")
        return Rubric()
    shaped = _normalize_rubric({"criteria": parsed.get("criteria")})
    # Reuse the bundle parser to get the typed Rubric (via a throwaway Question dict).
    return _parse_question(0, {"rubric": shaped}).rubric


def pairs_to_session(
    pairs: list[dict], default_language: str = "en", *,
    synthesize_rubrics: bool = False, bundle: ContextBundle | None = None,
    backend: str | None = None, model: str | None = None,
) -> TrainingSession:
    """Turn captured pairs into a TrainingSession: one Question per pair (empty rubric → holistic,
    or a synthesized rubric when asked) with the candidate's answer filled in. Pairs with an empty
    answer are dropped — a question the candidate never answered is not a graded one."""
    chosen = resolve_backend(backend or settings.SCORING_BACKEND) if synthesize_rubrics else ""
    role = bundle.role if bundle else ""
    questions: list[Question] = []
    answers: list[str] = []
    for i, pair in enumerate(pairs):
        answer = str(pair.get("answer", "")).strip()
        if not answer:
            continue
        q_text = str(pair.get("question", "")).strip()
        rubric = (_synthesize_rubric(q_text, role, chosen, model)
                  if synthesize_rubrics else Rubric())
        questions.append(Question(
            id=f"q{i}", competency="", question=q_text,
            language=str(pair.get("question_language") or default_language) or default_language,
            rubric=rubric,
        ))
        answers.append(answer)

    session = TrainingSession(questions)
    for index, answer in enumerate(answers):
        session.seek(index)
        session.add_answer_segment(answer)
    session.finish()
    return session


def review(
    session_id: str, qa_path: Path, *, backend: str | None = None, model: str | None = None,
    synthesize_rubrics: bool = False, output_dir: Path | None = None, write: bool = True,
    sessions_dir: Path | None = None,
):
    """Load the bundle + captured pairs and grade them into an 'Interview review' report
    (scripts/outputs/interview_review_<stamp>.md). Returns the SessionReport."""
    bundle = load_bundle(session_id, sessions_dir=sessions_dir)
    pairs = load_qa_pairs(qa_path)
    if not any(str(p.get("answer", "")).strip() for p in pairs):
        raise ValueError(f"no answered question↔answer pairs in {qa_path.name} — nothing to review")
    session = pairs_to_session(pairs, bundle.spoken_language or "en",
                               synthesize_rubrics=synthesize_rubrics, bundle=bundle,
                               backend=backend, model=model)
    return score_session(
        session, bundle, backend=backend, model=model, output_dir=output_dir, write=write,
        report_kind="Interview review", name_prefix="interview_review",
    )


# --------------------------------------------------------------------------
# --selftest — NO MODEL: the pair-loading + session-shaping adapter
# --------------------------------------------------------------------------
def _selftest() -> int:
    import tempfile

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "interview_qa_20260920_101010.jsonl"
        p.write_text(
            '{"question": "What is RAG?", "answer": "Retrieval augmented generation.", "question_language": "en"}\n'
            "\n"                                              # blank line tolerated
            '{"question": "bad json"'                        # malformed line tolerated
            "\n"
            '{"question": "Unanswered?", "answer": "   "}\n'  # empty answer → dropped
            '{"question": "Tell me about vector DBs", "answer": "Qdrant, HNSW."}\n',
            encoding="utf-8",
        )
        pairs = load_qa_pairs(p)
        # 3 well-formed question rows survive parsing (the malformed one is skipped).
        check(len(pairs) == 3, f"expected 3 parsed pairs, got {len(pairs)}")
        session = pairs_to_session(pairs, "en")
        answered = session.answered()
        # Only the 2 with a real answer become graded questions; the empty-answer one is dropped.
        check(len(answered) == 2, f"expected 2 answered questions, got {len(answered)}")
        check(answered[0][0].question == "What is RAG?", "first question mis-shaped")
        check(answered[0][1] == "Retrieval augmented generation.", "first answer lost")
        check(not answered[0][0].rubric.criteria, "live question should default to an empty rubric")
        check(newest_qa_log(Path(tmp)) == p, "newest_qa_log did not find the log")

    print("review.py --selftest — pair loading + session shaping")
    if failures:
        print("  FAIL:")
        for f in failures:
            print(f"   - {f}")
        return 1
    print(f"  OK — {'all checks pass' if not failures else ''}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--selftest", action="store_true",
                        help="no-model self-test of the pair-loading + session adapter; 0 = pass")
    parser.add_argument("--session", help="session id under scripts/inputs/sessions/ (or a path)")
    parser.add_argument("--qa", help="interview_qa_<stamp>.jsonl (default: the newest one)")
    parser.add_argument("--synthesize-rubrics", action="store_true",
                        help="author a rubric per captured question (opt-in; else holistic)")
    parser.add_argument("--backend", choices=("local", "cloud"), help="override SCORING_BACKEND")
    parser.add_argument("--model", help="override the scoring model")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if not args.session:
        parser.error("--session is required (or use --selftest)")

    qa_path = Path(args.qa) if args.qa else newest_qa_log()
    if qa_path is None:
        logger.error("no interview_qa_*.jsonl found in %s — run a live interview first", OUTPUT_DIR)
        sys.exit(1)
    if not qa_path.exists():
        logger.error("no such Q&A log: %s", qa_path)
        sys.exit(1)

    try:
        report = review(args.session, qa_path, backend=args.backend, model=args.model,
                        synthesize_rubrics=args.synthesize_rubrics)
    except BackendUnavailable:
        logger.exception("scoring backend unavailable")
        sys.exit(1)
    except (ValueError, FileNotFoundError) as error:
        logger.error("review failed: %s", error)
        sys.exit(1)

    print(f"\nwrote {report.path}")
    print(f"  {report.session_id} | {len(report.answers)} answer(s) reviewed | "
          f"{report.total:g} / {report.max_score:g} ({report.normalized * 100:.0f}%)")


if __name__ == "__main__":
    main()
