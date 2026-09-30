"""Tests for the post-interview reviewer (D32): loading captured Q&A pairs, shaping them into a
scored session, and the end-to-end review. All deterministic — NO GPU, NO network. `llm_call` is
monkeypatched (in both `training` and `review`) to return fixture JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import review, training  # noqa: E402
from scripts.llm_client import LlmReply  # noqa: E402
from scripts.reasoning import ContextBundle  # noqa: E402


def _reply(text: str) -> LlmReply:
    return LlmReply(text=text, backend="local", model="fake", latency_seconds=0.01,
                    prompt_tokens=10, completion_tokens=20)


def _fixed_reply(payload) -> callable:
    body = payload if isinstance(payload, str) else json.dumps(payload)

    def fake(prompt, system=None, backend=None, model=None, max_tokens=None, **kwargs):
        fake.calls += 1
        return _reply(body)

    fake.calls = 0
    return fake


def _bundle() -> ContextBundle:
    return ContextBundle(
        session_id="hr_test", role="AI Engineer", company="ExampleCo",
        spoken_language="en", suggestion_language="match",
        job_description="Build LLM agents.", company_brief="", resume="Worked on RAG.",
        answer_bank=[], plan=[], honesty_boundary=[], placeholders=[], source_dir=Path("."),
    )


HOLISTIC = {"overall_level": "adequate", "note": "ok", "concepts_to_refresh": ["evals"],
            "strengths": ["concrete"]}

QA_LINES = (
    '{"question": "What is RAG?", "answer": "Retrieval augmented generation.", "question_language": "en"}\n'
    "\n"
    '{"question": "malformed"\n'
    '{"question": "Unanswered?", "answer": "  "}\n'
    '{"question": "Vector DBs?", "answer": "Qdrant, HNSW."}\n'
)


def _write_qa(tmp_path: Path) -> Path:
    p = tmp_path / "interview_qa_20260920_101010.jsonl"
    p.write_text(QA_LINES, encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# pair loading
# --------------------------------------------------------------------------
def test_load_qa_pairs_tolerates_blank_and_malformed_lines(tmp_path):
    pairs = review.load_qa_pairs(_write_qa(tmp_path))
    # 3 well-formed question rows survive; the malformed line is skipped.
    assert [p["question"] for p in pairs] == ["What is RAG?", "Unanswered?", "Vector DBs?"]


def test_newest_qa_log_picks_the_latest(tmp_path):
    (tmp_path / "interview_qa_20260101_000000.jsonl").write_text("{}", encoding="utf-8")
    newest = tmp_path / "interview_qa_20260920_101010.jsonl"
    newest.write_text("{}", encoding="utf-8")
    assert review.newest_qa_log(tmp_path) == newest


# --------------------------------------------------------------------------
# pairs -> session
# --------------------------------------------------------------------------
def test_pairs_to_session_drops_empty_answers_and_defaults_to_holistic(tmp_path):
    pairs = review.load_qa_pairs(_write_qa(tmp_path))
    session = review.pairs_to_session(pairs, "en")
    answered = session.answered()
    assert [q.question for q, _ in answered] == ["What is RAG?", "Vector DBs?"]
    assert answered[0][1] == "Retrieval augmented generation."
    assert all(not q.rubric.criteria for q, _ in answered)   # holistic by default


def test_synthesize_rubrics_attaches_a_rubric(tmp_path, monkeypatch):
    rubric_json = {"criteria": [
        {"id": "depth", "label": "Depth", "weight": 1.0,
         "levels": {"excellent": "x", "adequate": "y", "weak": "z"}}]}
    monkeypatch.setattr(review, "llm_call", _fixed_reply(rubric_json))
    pairs = review.load_qa_pairs(_write_qa(tmp_path))
    session = review.pairs_to_session(pairs, "en", synthesize_rubrics=True, bundle=_bundle())
    assert all(q.rubric.criteria for q, _ in session.answered())
    assert session.answered()[0][0].rubric.criteria[0].id == "depth"


# --------------------------------------------------------------------------
# end-to-end review
# --------------------------------------------------------------------------
def test_review_writes_an_interview_review_report(tmp_path, monkeypatch):
    grade = _fixed_reply(HOLISTIC)
    monkeypatch.setattr(training, "llm_call", grade)
    monkeypatch.setattr(review, "load_bundle", lambda sid, sessions_dir=None: _bundle())
    qa = _write_qa(tmp_path)
    report = review.review("hr_test", qa, output_dir=tmp_path)

    assert grade.calls == 2                                  # one call per answered pair
    assert len(report.answers) == 2
    assert report.report_kind == "Interview review"
    assert report.path.name.startswith("interview_review_")
    assert report.total == 2.0 and report.max_score == 4.0   # two 'adequate' holistic answers


def test_review_raises_when_nothing_was_answered(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "load_bundle", lambda sid, sessions_dir=None: _bundle())
    qa = tmp_path / "interview_qa_20260920_000000.jsonl"
    qa.write_text('{"question": "Q?", "answer": ""}\n', encoding="utf-8")
    import pytest
    with pytest.raises(ValueError):
        review.review("hr_test", qa, output_dir=tmp_path)


def test_selftest_passes():
    assert review._selftest() == 0
