"""Tests for the mock-interview engine (scripts/training.py, D30 offline practice track).

All deterministic — NO GPU, NO network. `llm_call` is monkeypatched to return fixture JSON, so
every test asserts the ENGINE's own logic (never a model's judgement): answer-segment attribution
through the active-question pointer, that `score_answer` parses a mocked reply and aggregates it
DETERMINISTICALLY (level -> score x weight, total, comparison to `max_score()`), that concepts and
strengths are surfaced, that `to_markdown()` renders a report, and that `generate_questions`
prefers a populated `question_bank` over calling the model. Follows the tmp_path + monkeypatch
style of test_dashboard.py / test_generate_context.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import training  # noqa: E402
from scripts.llm_client import LlmReply  # noqa: E402
from scripts.reasoning import (  # noqa: E402
    ContextBundle, Question, Rubric, RubricCriterion,
)


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------
def _reply(text: str) -> LlmReply:
    return LlmReply(text=text, backend="local", model="fake-model", latency_seconds=0.01,
                    prompt_tokens=10, completion_tokens=20)


def _fixed_reply(payload) -> callable:
    """A fake llm_call returning `payload` (a str, or an object json-dumped to str) every call."""
    body = payload if isinstance(payload, str) else json.dumps(payload)

    def fake(prompt, system=None, backend=None, model=None, max_tokens=None, **kwargs):
        fake.calls += 1
        return _reply(body)

    fake.calls = 0
    return fake


def _rubric() -> Rubric:
    """Unequal weights so a wrong weighting cannot pass by luck. max = (2.0 + 1.0) * 2.0 = 6.0."""
    return Rubric(criteria=[
        RubricCriterion(id="depth", label="Technical depth", weight=2.0,
                        levels={"excellent": "names recall@k", "adequate": "some", "weak": "none"}),
        RubricCriterion(id="tradeoff", label="Trade-off awareness", weight=1.0,
                        levels={"excellent": "ANN vs exact", "adequate": "some", "weak": "none"}),
    ])


def _question(qid: str = "q1", competency: str = "RAG design") -> Question:
    return Question(id=qid, competency=competency,
                    question="How would you evaluate retrieval quality?",
                    language="en", rubric=_rubric())


def _bundle(question_bank=None, plan=None) -> ContextBundle:
    return ContextBundle(
        session_id="acme_test", role="ML Engineer", company="Acme",
        spoken_language="en", suggestion_language="match",
        job_description="Build RAG pipelines.", company_brief="Acme builds tooling.",
        resume="", answer_bank=[], plan=plan or [], honesty_boundary=[], placeholders=[],
        source_dir=Path("."), question_bank=question_bank or [],
    )


# A well-formed grade reply for the two-criterion rubric above.
GOOD_GRADE = {
    "per_criterion": [
        {"id": "depth", "level": "excellent", "note": "named recall@k and a baseline"},
        {"id": "tradeoff", "level": "adequate", "note": "mentioned ANN loosely"},
    ],
    "concepts_to_refresh": ["exact vs approximate nearest neighbour"],
    "strengths": ["clear structure", "concrete metric"],
}


# --------------------------------------------------------------------------
# 1. Answer-segment attribution (the active-question pointer + buffers)
# --------------------------------------------------------------------------
def test_segments_append_to_the_active_question_and_advance_switches():
    session = training.TrainingSession([_question("q1"), _question("q2")])
    assert session.active_question.id == "q1"

    session.add_answer_segment("first part")
    session.add_answer_segment("second part")
    assert session.advance() is True                 # a next question exists
    assert session.active_question.id == "q2"

    session.add_answer_segment("answer to two")
    assert session.answer_text(0) == "first part second part"
    assert session.answer_text(1) == "answer to two"


def test_advance_past_the_last_question_parks_the_pointer():
    session = training.TrainingSession([_question("q1")])
    assert session.advance() is False                # no next question
    assert session.active_question is None
    # A segment after the end is a no-op, not an error, and does not leak into the last buffer.
    assert session.add_answer_segment("stray") is False
    assert session.answer_text(0) == ""


def test_finish_refuses_new_segments_but_keeps_the_buffers():
    session = training.TrainingSession([_question("q1")])
    session.add_answer_segment("kept")
    session.finish()
    assert session.add_answer_segment("after finish") is False
    assert session.active_question is None
    assert session.answer_text(0) == "kept"          # buffered content survives for scoring


def test_answered_returns_only_questions_with_content():
    session = training.TrainingSession([_question("q1"), _question("q2"), _question("q3")])
    session.add_answer_segment("q1 answer")
    session.advance()
    # q2 left blank
    session.advance()
    session.add_answer_segment("q3 answer")
    answered = session.answered()
    assert [q.id for q, _ in answered] == ["q1", "q3"]
    assert [text for _, text in answered] == ["q1 answer", "q3 answer"]


# --------------------------------------------------------------------------
# 2. score_answer — ONE call, deterministic aggregation
# --------------------------------------------------------------------------
def test_score_answer_aggregates_deterministically(monkeypatch):
    fake = _fixed_reply(GOOD_GRADE)
    monkeypatch.setattr(training, "llm_call", fake)
    q = _question()

    result = training.score_answer(q, "We use recall@k against a baseline...", _bundle())

    assert fake.calls == 1                            # EXACTLY one model call per answer
    # excellent(2.0) * weight 2.0 = 4.0 ; adequate(1.0) * weight 1.0 = 1.0
    assert result.per_criterion[0].id == "depth" and result.per_criterion[0].score == 4.0
    assert result.per_criterion[1].id == "tradeoff" and result.per_criterion[1].score == 1.0
    assert result.total == 5.0
    assert result.max_score == 6.0 == q.rubric.max_score()
    assert result.normalized == pytest.approx(5.0 / 6.0)


def test_score_answer_surfaces_concepts_and_strengths(monkeypatch):
    monkeypatch.setattr(training, "llm_call", _fixed_reply(GOOD_GRADE))
    result = training.score_answer(_question(), "an answer", _bundle())
    assert result.concepts_to_refresh == ["exact vs approximate nearest neighbour"]
    assert result.strengths == ["clear structure", "concrete metric"]


def test_a_criterion_the_model_did_not_grade_scores_as_missing(monkeypatch):
    """Only 'depth' is graded; 'tradeoff' must default to missing (0.0), never be guessed."""
    monkeypatch.setattr(training, "llm_call", _fixed_reply(
        {"per_criterion": [{"id": "depth", "level": "excellent"}]}
    ))
    result = training.score_answer(_question(), "partial answer", _bundle())
    assert result.per_criterion[0].score == 4.0
    assert result.per_criterion[1].level == "missing" and result.per_criterion[1].score == 0.0
    assert result.total == 4.0


def test_an_unknown_level_is_treated_as_missing(monkeypatch):
    monkeypatch.setattr(training, "llm_call", _fixed_reply(
        {"per_criterion": [{"id": "depth", "level": "amazing"},
                           {"id": "tradeoff", "level": "weak"}]}
    ))
    result = training.score_answer(_question(), "answer", _bundle())
    assert result.per_criterion[0].level == "missing" and result.per_criterion[0].score == 0.0
    assert result.total == 0.0


def test_score_answer_tolerates_a_fenced_reply_without_a_repair(monkeypatch):
    fake = _fixed_reply("```json\n" + json.dumps(GOOD_GRADE) + "\n```")
    monkeypatch.setattr(training, "llm_call", fake)
    result = training.score_answer(_question(), "answer", _bundle())
    assert fake.calls == 1                            # fence stripping, not a repair round-trip
    assert result.total == 5.0


# --------------------------------------------------------------------------
# 3. score_session + to_markdown
# --------------------------------------------------------------------------
def _answered_session():
    session = training.TrainingSession([_question("q1"), _question("q2")])
    session.add_answer_segment("answer to one")
    session.advance()
    session.add_answer_segment("answer to two")
    session.finish()
    return session


def test_score_session_scores_every_answered_question(monkeypatch, tmp_path):
    fake = _fixed_reply(GOOD_GRADE)
    monkeypatch.setattr(training, "llm_call", fake)
    report = training.score_session(_answered_session(), _bundle(), output_dir=tmp_path)

    assert fake.calls == 2                            # one call per answered question
    assert len(report.answers) == 2
    assert report.total == 10.0 and report.max_score == 12.0
    assert report.normalized == pytest.approx(10.0 / 12.0)


def test_to_markdown_renders_a_report(monkeypatch):
    monkeypatch.setattr(training, "llm_call", _fixed_reply(GOOD_GRADE))
    report = training.score_session(_answered_session(), _bundle(), write=False)
    md = report.to_markdown()

    assert "# Mock interview report — acme_test" in md
    assert "How would you evaluate retrieval quality?" in md
    assert "Overall:" in md and "10 / 12" in md
    assert "answer to one" in md
    assert "`excellent`" in md and "`adequate`" in md
    assert "Concepts to refresh:" in md and "Strengths:" in md


def test_score_session_writes_a_timestamped_report(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "llm_call", _fixed_reply(GOOD_GRADE))
    report = training.score_session(_answered_session(), _bundle(), output_dir=tmp_path)

    assert report.path is not None and report.path.exists()
    assert report.path.name.startswith("training_report_") and report.path.suffix == ".md"
    assert report.path.parent == tmp_path
    # No leftover temp file, and the written content is the same render.
    assert list(tmp_path.glob("*.tmp")) == []
    assert report.path.read_text(encoding="utf-8") == report.to_markdown()


def test_score_session_skips_unanswered_questions(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "llm_call", _fixed_reply(GOOD_GRADE))
    session = training.TrainingSession([_question("q1"), _question("q2")])
    session.add_answer_segment("only q1 answered")
    session.finish()
    report = training.score_session(session, _bundle(), output_dir=tmp_path)
    assert len(report.answers) == 1
    assert report.answers[0].question.id == "q1"


# --------------------------------------------------------------------------
# 4. generate_questions — prefer the pre-built bank
# --------------------------------------------------------------------------
def test_generate_questions_prefers_a_populated_bank(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("llm_call must not run when a question_bank is present")

    monkeypatch.setattr(training, "llm_call", explode)
    bank = [_question("q1"), _question("q2")]
    got = training.generate_questions(_bundle(question_bank=bank), n=5)
    assert [q.id for q in got] == ["q1", "q2"]        # the bank, unmodified, no model call


def test_generate_questions_authors_when_the_bank_is_empty(monkeypatch):
    from scripts.reasoning import PlanStep

    fake = _fixed_reply({"question_bank": [
        {"competency": "RAG design", "question": "How do you chunk documents?",
         "language": "en",
         "rubric": {"criteria": [
             {"id": "depth", "label": "Depth", "weight": 1.0,
              "levels": {"excellent": "x", "adequate": "y", "weak": "z"}}]}},
    ]})
    monkeypatch.setattr(training, "llm_call", fake)
    bundle = _bundle(plan=[PlanStep(id="c1", title="RAG design", key_points=["chunking"])])
    got = training.generate_questions(bundle, n=1)

    assert fake.calls == 1
    assert len(got) == 1
    assert got[0].question == "How do you chunk documents?"
    assert got[0].rubric.criteria[0].weight == 1.0


# --------------------------------------------------------------------------
# 4b. Holistic grading + JD/CV grounding (D32/D33)
# --------------------------------------------------------------------------
def _capturing_reply(payload) -> callable:
    """Like `_fixed_reply` but records every (prompt, system) it was called with."""
    body = payload if isinstance(payload, str) else json.dumps(payload)

    def fake(prompt, system=None, backend=None, model=None, max_tokens=None, **kwargs):
        fake.calls += 1
        fake.prompts.append(prompt)
        fake.systems.append(system)
        return _reply(body)

    fake.calls = 0
    fake.prompts = []
    fake.systems = []
    return fake


HOLISTIC_GRADE = {
    "overall_level": "adequate",
    "note": "on-topic but shallow on evaluation",
    "concepts_to_refresh": ["retrieval metrics"],
    "strengths": ["named a concrete tool"],
}


def test_score_answer_holistic_when_the_question_has_no_rubric(monkeypatch):
    fake = _capturing_reply(HOLISTIC_GRADE)
    monkeypatch.setattr(training, "llm_call", fake)
    q = Question(id="q1", competency="", question="Tell me about vector DBs.", language="en")  # empty rubric
    score = training.score_answer(q, "I have used Qdrant with HNSW.")

    assert fake.calls == 1                                   # still exactly one model call
    assert score.max_score == 2.0                            # one synthetic 'overall' criterion
    assert score.total == 1.0                                # 'adequate' -> 1.0 * weight 1.0
    assert score.per_criterion[0].id == "overall"
    assert score.per_criterion[0].note == "on-topic but shallow on evaluation"
    assert score.concepts_to_refresh == ["retrieval metrics"]
    # the holistic system prompt was used, not the rubric one
    assert "no rubric" in fake.systems[0].lower()


def test_grounding_block_is_present_iff_a_bundle_is_given(monkeypatch):
    fake = _capturing_reply(HOLISTIC_GRADE)
    monkeypatch.setattr(training, "llm_call", fake)
    q = Question(id="q1", competency="", question="What is RAG?", language="en")
    bundle = _bundle()   # role=ML Engineer, company=Acme, a JD, empty resume

    training.score_answer(q, "answer", bundle=bundle)
    with_bundle = fake.prompts[-1]
    assert "JOB DESCRIPTION" in with_bundle
    assert "ML Engineer at Acme" in with_bundle

    training.score_answer(q, "answer", bundle=None)
    without_bundle = fake.prompts[-1]
    assert "JOB DESCRIPTION" not in without_bundle


def test_holistic_unknown_level_scores_as_missing(monkeypatch):
    monkeypatch.setattr(training, "llm_call",
                        _fixed_reply({"overall_level": "banana", "note": "n"}))
    q = Question(id="q1", competency="", question="Q?", language="en")
    score = training.score_answer(q, "a")
    assert score.total == 0.0
    assert score.per_criterion[0].level == "missing"


def test_interview_review_report_kind_and_filename(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "llm_call", _fixed_reply(HOLISTIC_GRADE))
    q = Question(id="q1", competency="", question="Q?", language="en")
    session = training.TrainingSession([q])
    session.add_answer_segment("an answer")
    session.finish()
    report = training.score_session(session, _bundle(), output_dir=tmp_path,
                                    report_kind="Interview review", name_prefix="interview_review")
    assert report.path.name.startswith("interview_review_")
    assert report.to_markdown().startswith("# Interview review report —")


# --------------------------------------------------------------------------
# 5. The deterministic selftest exits 0
# --------------------------------------------------------------------------
def test_selftest_passes():
    assert training._selftest() == 0


def test_holistic_prompt_grades_the_question_with_defined_bands():
    """F3 (#1273): the holistic grader must judge the question asked (JD/CV are context, not criteria),
    define all bands, and tolerate speech-to-text noise."""
    from scripts.training import SCORE_HOLISTIC_SYSTEM, SCORE_SYSTEM
    for phrase in ("THE QUESTION THAT WAS ASKED", "never extra criteria", "- excellent:", "- adequate:",
                   "- weak:", "- missing:", "SPEECH-TO-TEXT"):
        assert phrase in SCORE_HOLISTIC_SYSTEM
    assert "speech-to-text transcript" in SCORE_SYSTEM
