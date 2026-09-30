# `scripts/training.py` — mock-interview engine (D30/D31)

The offline practice track's **engine**: it sources likely questions (with rubrics), holds a mock
interview's per-question spoken answers, then grades each answer against its rubric and emits a
constructive report naming **concepts to refresh**. UI-independent — the dashboard Training mode
(`dashboard.py --training`, D31) drives this same engine; you can also run it from the CLI.

## Quickstart
```bash
python scripts/training.py --selftest                                  # no model — aggregation math
python scripts/training.py --session my_role --answers answers.json \  # grade a fixture answer set
    --backend local --model interview-copilot:8b
```
`--answers` is a JSON list of `{id?, answer}` rows graded against the session bundle's
`question_bank`. Writes `scripts/outputs/training_report_<stamp>.md`. Run scoring under a GPU lease.

## The pieces
- **`generate_questions(bundle, n)`** — returns `bundle.question_bank` when populated (no model call);
  otherwise authors N questions + rubrics per competency from the plan via one `llm_call` (reuses
  `generate_context`'s JSON parser + repair). Count default: `TRAINING_QUESTION_COUNT`.
- **`TrainingSession`** — ordered questions + an active-question pointer + per-question answer
  buffers, all lock-guarded so a controller thread can drive it: `add_answer_segment(text)` appends
  to the active question, `advance()` closes the buffer and moves on, `seek(index)` positions the
  pointer, `finish()` ends it. No UI.
- **`score_answer(question, answer_text, bundle, backend=None)`** — **one** `llm_call` grading the
  answer against the rubric criteria (one repair retry). **Aggregation is deterministic Python**
  (CLAUDE.md Determinism First): each model-returned level maps through
  `RUBRIC_LEVEL_SCORE × criterion.weight` (excellent 2 / adequate 1 / weak 0; unknown/ungraded →
  `missing` = 0), summed for `total` against `Rubric.max_score()`. Only the per-criterion **level +
  note** (and the concepts/strengths prose) come from the model. Returns an `AnswerScore`
  (`per_criterion`, `total`, `max_score`, `normalized`, `concepts_to_refresh`, `strengths`).
- **`score_session(session, bundle)`** → a `SessionReport` with `to_markdown()`, written to
  `scripts/outputs/training_report_<stamp>.md` (write-temp-then-rename, local-only — SI1).
- **`--selftest`** — no-model check of the aggregation/normalization/`max_score`/rubric-parse math;
  exit 0 = pass (mirrors `reasoning.py --selftest-*`).

## Settings (CFG — `config/settings.py`)
`SCORING_BACKEND` (local, SI1) · `SCORING_MODEL` ("" = backend default) · `SCORING_MAX_TOKENS` (700) ·
`SCORING_TIMEOUT_SECONDS` (180) · `TRAINING_QUESTION_COUNT` (6). Scoring defaults to the **local**
backend — practice answers stay on the machine unless `--backend cloud` is chosen (opt-in + banner).

## Verified (2026-09-17, #651)
On a generated bundle, local `interview-copilot:8b`: a strong answer scored 2.6/3.6 (72% — adequate
depth, excellent clarity) and a weak answer 0/3.6, each with per-criterion notes, concepts to refresh,
and strengths; the report rendered + wrote. Tests: `tests/test_training.py` (mocked `llm_call`) +
`--selftest`. The live-mic walkthrough (answer by voice via `dashboard.py --training`) is the one
remaining E2E step.
