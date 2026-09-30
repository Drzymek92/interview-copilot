# `review.py` — live Q&A capture + post-interview review (D32/D33/D34)

Saves what the interviewer asked and how you answered during a **real** interview, then grades those
answers against your CV + the job info and writes a feedback report. The reviewer is the **same**
one the offline mock-trainer uses (`training.py`) — extended, not duplicated.

## The two halves

### 1. Capture (live, in `dashboard.py`)
A `QaPairer` (a `make_sink` consumer — D34, so no new seam) watches the ambient event stream the
live copilot already produces:
- a **salient interviewer question** (a D23 gate *fire* — the same YES/NO that decides a suggestion
  is worth showing) **opens** a pair, flushing the previous one;
- each following **`you:`** transcript line accumulates into that pair's answer;
- the pair is flushed to disk on the next question or when capture stops.

Only salient questions are paired — logistics/rapport turns that the gate drops are skipped. Capture
runs automatically whenever the live copilot runs with a bundle (`dashboard.py --app`, or
`--watch`/`--follow` with suggestions on).

**The pair log (D33 contract):** `scripts/outputs/interview_qa_<stamp>.jsonl`, one JSON object per
line, sharing the run stamp with the transcript/audio/suggestions (TRK):
```json
{"logged_at": "…Z", "question": "…", "question_stamp": "mm:ss-mm:ss", "question_language": "en", "answer": "…"}
```
Local-only (SI1); a write failure is logged and swallowed — losing a record never takes the copilot
down.

### 2. Review (post-interview — CLI or dashboard button)
Each captured pair is graded with **exactly one** model call. A live interviewer's questions are
ad-hoc and carry no rubric, so by default each answer is graded **holistically**, grounded in the
ROLE + JD + CV + honesty boundary from the bundle. `--synthesize-rubrics` authors a rubric per
question first, for structured per-criterion scores ("rubric if available, else holistic"). The
numeric aggregation stays **deterministic** (D30): the model returns only a level + prose, and the
score is `RUBRIC_LEVEL_SCORE × weight` computed in Python. Output:
`scripts/outputs/interview_review_<stamp>.md`.

Grounding + holistic grading are added to the **common reviewer** (`training.score_answer` /
`score_session`), so the mock trainer gains JD/CV grounding too (D33).

## Run it
```bash
python scripts/review.py --selftest                          # no model — pair loading + adapter
python scripts/review.py --session example_ai_engineer               # newest interview_qa_*.jsonl
python scripts/review.py --session example_ai_engineer --qa scripts/outputs/interview_qa_<stamp>.jsonl
python scripts/review.py --session example_ai_engineer --synthesize-rubrics
```
On lab, GPU-lease the scoring run (local model):
```bash
python -m commons.coordination.gpu run --vram 6000 --label "interview review" -- \
  python scripts/review.py --session example_ai_engineer --qa scripts/outputs/interview_qa_<stamp>.jsonl
```
In the dashboard `--app` view, the **Review answers** button runs the same review over the current
run's pairs and renders the result inline (path + per-question feedback). Run it **after** the call
(or lease VRAM) — the 14B + Whisper already sit near the 16 GB ceiling, so don't score mid-call.

## Notes / limits
- An unanswered question is written to the log (faithful record) but **dropped** at review time — a
  blank answer is not a graded one.
- Grounding is *instructed*, not enforced: the honesty boundary only guards **listed** claims (D22).
- No new egress: pairs, scoring (local Ollama default) and the report are local (SI1). Cloud review
  stays opt-in via the announced BYOK path.
- Reuses: `training.score_session` / `score_answer` / `SessionReport`; `reasoning.load_bundle` +
  `Question`/`Rubric`; `generate_context._normalize_rubric` (only for `--synthesize-rubrics`).
