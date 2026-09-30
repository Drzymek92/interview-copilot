"""Tests for the D30 bundle generator (scripts/generate_context.py).

All deterministic — NO GPU, NO network. `llm_call` is monkeypatched to return fixture JSON, so
every test asserts the tool's own logic: that a generated bundle loads through the REAL
`reasoning.load_bundle`, that the candidate-side sections are left as flagged placeholders (D22),
that `question_bank[]` parses into `Question`/`Rubric` objects with the fixed rubric shape, that a
fenced or malformed model reply is stripped / repaired, and that CLI parsing works. Follows the
tmp_path + monkeypatch style of test_dashboard.py / test_reasoning.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import generate_context as gc  # noqa: E402
from scripts import reasoning  # noqa: E402
from scripts.llm_client import LlmReply  # noqa: E402
from scripts.reasoning import RUBRIC_LEVELS, Question, RubricCriterion  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures: a well-formed model reply and helpers to mock llm_call
# --------------------------------------------------------------------------
GOOD_SECTIONS = {
    "job_description": {
        "responsibilities": ["Build RAG pipelines", "Own evaluation harnesses"],
        "must_haves": ["Python", "LLM app experience"],
        "nice_to_haves": ["Vector DB experience"],
    },
    "company_brief": "Acme builds developer tooling for LLM apps.",
    "plan": [
        {"id": "c1", "title": "RAG design", "key_points": ["chunking", "retrieval eval"]},
        {"id": "c2", "title": "Production", "key_points": ["latency", "cost"]},
    ],
    "question_bank": [
        {
            "competency": "RAG design",
            "question": "How would you evaluate retrieval quality?",
            "language": "en",
            "rubric": {
                "criteria": [
                    {"id": "depth", "label": "Technical depth", "weight": 2.0,
                     "levels": {"excellent": "names recall@k + a baseline",
                                "adequate": "mentions relevance", "weak": "vague"}},
                    {"id": "tradeoff", "label": "Trade-off awareness", "weight": 1.0,
                     "levels": {"excellent": "ANN vs exact", "adequate": "some", "weak": "none"}},
                ]
            },
        },
        {
            "competency": "Production",
            "question": "How do you keep p95 latency down?",
            "language": "en",
            "rubric": {
                "criteria": [
                    {"id": "concrete", "label": "Concreteness", "weight": 1.0,
                     "levels": {"excellent": "caching + streaming", "adequate": "one lever", "weak": "none"}},
                ]
            },
        },
    ],
}


def _reply(text: str) -> LlmReply:
    return LlmReply(text=text, backend="local", model="fake-model", latency_seconds=0.01,
                    prompt_tokens=10, completion_tokens=20)


def _fixed_reply(payload) -> callable:
    """A fake llm_call that always returns `payload` (a str, or an object json-dumped to str)."""
    body = payload if isinstance(payload, str) else json.dumps(payload)

    def fake(prompt, system=None, backend=None, model=None, max_tokens=None, **kwargs):
        return _reply(body)

    return fake


def _scripted_replies(*texts: str) -> callable:
    """A fake llm_call that returns `texts` in order across successive calls; the call count is
    exposed on the returned function as `.calls`."""
    seq = list(texts)

    def fake(prompt, system=None, backend=None, model=None, max_tokens=None, **kwargs):
        fake.calls += 1
        return _reply(seq[min(fake.calls - 1, len(seq) - 1)])

    fake.calls = 0
    return fake


def _args(tmp_path: Path, session: str = "acme_test", **overrides) -> argparse.Namespace:
    base = dict(
        jd="Build LLM apps. Must know Python.", session=session, company="Acme", role="ML Engineer",
        company_notes="", questions_per_competency=2, backend="local", model=None,
        spoken_language="pl", suggestion_language="match",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------
# 1. End-to-end: a generated bundle loads and flags the candidate side
# --------------------------------------------------------------------------
def test_generated_bundle_loads_and_flags_candidate_sections(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "llm_call", _fixed_reply(GOOD_SECTIONS))
    target = gc.run(_args(tmp_path), sessions_dir=tmp_path)

    assert target.is_file()
    assert target == tmp_path / "acme_test" / "bundle.json"

    bundle = reasoning.load_bundle("acme_test", sessions_dir=tmp_path)
    # (a) candidate-side sections are flagged in .placeholders — never synthesized (D22).
    assert set(gc.CANDIDATE_SECTIONS) <= set(bundle.placeholders)
    assert bundle.resume == "" and bundle.answer_bank == [] and bundle.honesty_boundary == []
    assert any("resume" in w for w in bundle.warn_lines())
    # Interview-side sections ARE populated and NOT flagged.
    assert "Responsibilities" in bundle.job_description and "Python" in bundle.job_description
    assert bundle.company_brief == "Acme builds developer tooling for LLM apps."
    assert "job_description" not in bundle.placeholders
    assert [s.title for s in bundle.plan] == ["RAG design", "Production"]


def test_candidate_sections_are_placeholder_markers_on_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "llm_call", _fixed_reply(GOOD_SECTIONS))
    target = gc.run(_args(tmp_path), sessions_dir=tmp_path)
    raw = json.loads(target.read_text(encoding="utf-8"))
    for name in gc.CANDIDATE_SECTIONS:
        assert raw[name] == {"status": "placeholder"}, name
    assert raw["schema_version"] == 2


# --------------------------------------------------------------------------
# 2. question_bank parses into Question / Rubric with the fixed shape
# --------------------------------------------------------------------------
def test_question_bank_parses_into_question_and_rubric_objects(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "llm_call", _fixed_reply(GOOD_SECTIONS))
    gc.run(_args(tmp_path), sessions_dir=tmp_path)
    bundle = reasoning.load_bundle("acme_test", sessions_dir=tmp_path)

    assert len(bundle.question_bank) == 2
    q0 = bundle.question_bank[0]
    assert isinstance(q0, Question)
    assert q0.competency == "RAG design"
    assert q0.question == "How would you evaluate retrieval quality?"
    assert len(q0.rubric.criteria) == 2
    crit = q0.rubric.criteria[0]
    assert isinstance(crit, RubricCriterion)
    assert crit.label == "Technical depth" and crit.weight == 2.0
    # Fixed scoring shape: every criterion carries exactly the RUBRIC_LEVELS keys.
    assert set(crit.levels) == set(RUBRIC_LEVELS)
    # Deterministic ceiling: (2.0 + 1.0 weights) * 2.0 per 'excellent'.
    assert q0.rubric.max_score() == 6.0


def test_rubric_shape_is_repaired_when_the_model_omits_a_level(tmp_path, monkeypatch):
    """The scoring SHAPE is deterministic here, not the model's: a missing 'weak' descriptor is
    filled with "" so the grader always sees all three bands."""
    payload = json.loads(json.dumps(GOOD_SECTIONS))
    del payload["question_bank"][0]["rubric"]["criteria"][0]["levels"]["weak"]
    payload["question_bank"][0]["rubric"]["criteria"][0].pop("weight")  # also default the weight
    monkeypatch.setattr(gc, "llm_call", _fixed_reply(payload))
    gc.run(_args(tmp_path), sessions_dir=tmp_path)
    bundle = reasoning.load_bundle("acme_test", sessions_dir=tmp_path)
    crit = bundle.question_bank[0].rubric.criteria[0]
    assert set(crit.levels) == set(RUBRIC_LEVELS)
    assert crit.levels["weak"] == ""
    assert crit.weight == 1.0  # default applied deterministically


# --------------------------------------------------------------------------
# 3. Robust JSON handling — fences, repair-once, hard stop
# --------------------------------------------------------------------------
def test_strip_code_fences():
    assert gc.strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert gc.strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'
    assert gc.strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_parse_json_object_tolerates_trailing_prose():
    assert gc.parse_json_object('{"a": 1}\n\nHope that helps!') == {"a": 1}


def test_parse_json_object_raises_on_no_object():
    with pytest.raises(ValueError):
        gc.parse_json_object("sorry, I cannot help with that")


def test_fenced_reply_parses_without_a_repair_call(tmp_path, monkeypatch):
    fake = _scripted_replies("```json\n" + json.dumps(GOOD_SECTIONS) + "\n```")
    monkeypatch.setattr(gc, "llm_call", fake)
    gc.run(_args(tmp_path), sessions_dir=tmp_path)
    assert fake.calls == 1  # fence stripping, not a repair round-trip
    bundle = reasoning.load_bundle("acme_test", sessions_dir=tmp_path)
    assert len(bundle.question_bank) == 2


def test_malformed_reply_triggers_one_repair_then_succeeds(tmp_path, monkeypatch):
    fake = _scripted_replies("this is not json at all", json.dumps(GOOD_SECTIONS))
    monkeypatch.setattr(gc, "llm_call", fake)
    gc.run(_args(tmp_path), sessions_dir=tmp_path)
    assert fake.calls == 2  # one generation + one repair
    bundle = reasoning.load_bundle("acme_test", sessions_dir=tmp_path)
    assert bundle.company_brief.startswith("Acme")


def test_two_malformed_replies_fail_with_a_clear_error(tmp_path, monkeypatch):
    fake = _scripted_replies("nope", "still nope")
    monkeypatch.setattr(gc, "llm_call", fake)
    with pytest.raises(ValueError, match="after one repair"):
        gc.run(_args(tmp_path), sessions_dir=tmp_path)
    assert fake.calls == 2  # generation + exactly one repair, then stop


# --------------------------------------------------------------------------
# 4. JD resolution + CLI arg parsing
# --------------------------------------------------------------------------
def test_jd_is_read_from_a_file_when_the_path_exists(tmp_path, monkeypatch):
    jd_file = tmp_path / "jd.md"
    jd_file.write_text("Senior role. Must know Rust.", encoding="utf-8")
    monkeypatch.setattr(gc, "llm_call", _fixed_reply(GOOD_SECTIONS))
    target = gc.run(_args(tmp_path, jd=str(jd_file)), sessions_dir=tmp_path)
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["jd_source"] == str(jd_file)


def test_cli_arg_parsing():
    parser = gc.build_parser()
    args = parser.parse_args([
        "--jd", "some jd text", "--session", "acme_20260917",
        "--company", "Acme", "--role", "ML Engineer",
        "--questions-per-competency", "4", "--backend", "local",
    ])
    assert args.jd == "some jd text"
    assert args.session == "acme_20260917"
    assert args.company == "Acme"
    assert args.role == "ML Engineer"
    assert args.questions_per_competency == 4
    assert args.backend == "local"


def test_cli_requires_jd_and_session():
    parser = gc.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--company", "Acme"])


# --------------------------------------------------------------------------
# 5. Role derivation from the JD (#659) — --role always wins when passed
# --------------------------------------------------------------------------
def test_derived_role_used_when_role_flag_is_omitted(tmp_path, monkeypatch):
    payload = json.loads(json.dumps(GOOD_SECTIONS))
    payload["role"] = "Senior ML Engineer"
    monkeypatch.setattr(gc, "llm_call", _fixed_reply(payload))
    gc.run(_args(tmp_path, role=""), sessions_dir=tmp_path)
    bundle = reasoning.load_bundle("acme_test", sessions_dir=tmp_path)
    assert bundle.role == "Senior ML Engineer"


def test_explicit_role_flag_overrides_the_derived_role(tmp_path, monkeypatch):
    payload = json.loads(json.dumps(GOOD_SECTIONS))
    payload["role"] = "Senior ML Engineer"
    monkeypatch.setattr(gc, "llm_call", _fixed_reply(payload))
    gc.run(_args(tmp_path, role="Staff Backend Engineer"), sessions_dir=tmp_path)
    bundle = reasoning.load_bundle("acme_test", sessions_dir=tmp_path)
    assert bundle.role == "Staff Backend Engineer"


@pytest.mark.parametrize("missing_role", [None, "", "   ", 42, ["not", "a", "string"]])
def test_missing_or_malformed_derived_role_falls_back_to_empty_without_crashing(
    tmp_path, monkeypatch, missing_role,
):
    payload = json.loads(json.dumps(GOOD_SECTIONS))
    if missing_role is not None:
        payload["role"] = missing_role
    # else: leave "role" absent entirely — the model reply predates #659's schema extension.
    monkeypatch.setattr(gc, "llm_call", _fixed_reply(payload))
    gc.run(_args(tmp_path, role=""), sessions_dir=tmp_path)
    bundle = reasoning.load_bundle("acme_test", sessions_dir=tmp_path)
    assert bundle.role == ""


def test_cv_is_copied_verbatim_as_the_resume(tmp_path, monkeypatch):
    """D30 amendment / D37: a supplied CV becomes the bundle's real resume (never generated);
    answer_bank + honesty_boundary stay placeholders."""
    import argparse
    from scripts import generate_context as gen
    from scripts.reasoning import load_bundle
    monkeypatch.setattr(gen, "generate_sections", lambda *a, **k: {"role": "LLM Engineer", "plan": [], "question_bank": []})
    cv = tmp_path / "cv.md"
    cv.write_text("# Jan Kowalski\n\n- 5 years Python\n- built a RAG assistant\n", encoding="utf-8")
    args = argparse.Namespace(jd="x" * 200, session="withcv", company="", role="", company_notes="", cv=str(cv),
                              questions_per_competency=2, backend="local", model=None,
                              spoken_language="pl", suggestion_language="match")
    gen.run(args, sessions_dir=tmp_path)
    bundle = load_bundle("withcv", sessions_dir=tmp_path)
    assert bundle.resume.startswith("# Jan Kowalski") and "built a RAG assistant" in bundle.resume
    assert set(bundle.placeholders) == {"answer_bank", "honesty_boundary"}


def test_no_cv_keeps_the_resume_placeholder(tmp_path, monkeypatch):
    import argparse
    from scripts import generate_context as gen
    from scripts.reasoning import load_bundle
    monkeypatch.setattr(gen, "generate_sections", lambda *a, **k: {"plan": [], "question_bank": []})
    args = argparse.Namespace(jd="x" * 200, session="nocv", company="", role="", company_notes="",
                              questions_per_competency=2, backend="local", model=None,
                              spoken_language="pl", suggestion_language="match")     # no `cv` attr at all
    gen.run(args, sessions_dir=tmp_path)
    assert "resume" in load_bundle("nocv", sessions_dir=tmp_path).placeholders


def test_plan_keeps_cleaned_done_signals():
    """F2 (#1272): generated plan steps carry done_signals so the live plan rail can track them."""
    from scripts.generate_context import _normalize_plan
    plan = _normalize_plan([
        {"title": "Opening", "done_signals": ["Opowiedz o sobie", "tell me about yourself.", "opowiedz o sobie", "", "x" * 60]},
        {"title": "RAG", "key_points": ["hybrid retrieval"]},               # no signals → [] (old shape)
        "junk",
    ])
    assert [s["title"] for s in plan] == ["Opening", "RAG"]
    assert plan[0]["done_signals"] == ["opowiedz o sobie", "tell me about yourself"]
    assert plan[1]["done_signals"] == []


def test_prompt_asks_for_an_interview_arc_with_signals():
    from scripts.generate_context import SYSTEM_PROMPT, USER_PROMPT
    assert "done_signals" in SYSTEM_PROMPT and "probe=true" in SYSTEM_PROMPT and "EXACTLY 5 steps" in SYSTEM_PROMPT
    assert "{spoken}" in USER_PROMPT


def test_signals_shared_by_several_steps_are_dropped():
    from scripts.generate_context import _normalize_plan
    plan = _normalize_plan([{"title": "A", "done_signals": ["projekt", "rozwiązanie"]},
                            {"title": "B", "done_signals": ["rag", "rozwiązanie"]}])
    assert [s["done_signals"] for s in plan] == [["projekt"], ["rag"]]
