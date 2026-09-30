# `scripts/generate_context.py` — JD → interview-side context bundle (D29/D30)

Turns a raw **job description** into a schema-**v2** `bundle.json` that `reasoning.load_bundle`
consumes unchanged and the mock-interview trainer (`training.py`) reads. Part of the offline
**practice track** (D29). It generates only the **interview-side** sections; it **never synthesizes
the candidate's own experience** (D22) — `resume` / `answer_bank` / `honesty_boundary` are written as
`{"status":"placeholder"}` and surfaced by `ContextBundle.warn_lines()`.

## Quickstart
```bash
python scripts/generate_context.py --jd path/to/jd.md --session my_role \
    --company "ACME" --questions-per-competency 3 --backend local
```
Reads the JD (a file path or the text itself), writes
`scripts/inputs/sessions/<session>/bundle.json`, and prints a summary + the placeholder warnings.
`--jd` and `--cv` accept **PDF, .txt or .md** paths (`scripts/doc_text.py`, PyMuPDF, local — D37).
`--cv cv.pdf` copies **your own CV verbatim** into the bundle's `resume` (never rewritten — D22); without it
`resume` stays a placeholder. The hub's **Generate context** view drives the same `run()` (bundle name → JD → CV
→ company/role).
Run under a GPU lease when the local backend must load a model
(`python -m commons.coordination.gpu run --vram 12000 --label gen -- python scripts/generate_context.py ...`).

## What it generates (interview-side only)
| Section | Content |
|---|---|
| `job_description` | the JD cleaned/structured (responsibilities · must-haves · nice-to-haves), as the markdown string `load_bundle` reads |
| `company_brief` | short role/company framing; enrich with `--company-notes <path-or-text>` (never web-fetched — SI1) |
| `plan[]` | the competency arc as `PlanStep`s (`id` / `title` / `key_points`) |
| `question_bank[]` | N questions per competency, each with a weighted `rubric` (D30 — see `reasoning.py`) |
| `role` | a job-title guess derived from the JD by the same generation call (#659) — see below |

Left as flagged placeholders (never generated — D22): `resume`, `answer_bank`, `honesty_boundary`.

## Role: derived from the JD, `--role` always wins (#659)
Before #659 an omitted `--role` left the bundle's `role` blank, even though the flag's own help
text claimed it "defaults to what the JD implies" — the scorecard header just came up empty. The
generation prompt now also returns a `role` guess (extending the existing JSON contract, no second
LLM call, per CLAUDE.md Determinism First for everything *around* the one judgement call). `build_bundle`
resolves it: `--role` wins whenever passed; otherwise the derived guess is used; a missing, blank, or
non-string guess (a model that ignored the new field, or predates it) falls back to `""` exactly as
before generation — never a crash.

## How it calls the model
One `llm_call` per generation step via the shared seam; the local Ollama backend is the default
(SI1 — building a bundle from a JD is offline prep and stays on the machine unless `--backend cloud`).
Output is parsed by `parse_json_object` (tolerates code fences / trailing prose, carves the outermost
`{...}`) with **exactly one repair retry** before a pointed `ValueError`. The rubric **shape** is
enforced deterministically (`_normalize_rubric` fills all three `RUBRIC_LEVELS`, defaults weights) —
only the textual content is the model's (CLAUDE.md Determinism First).

## Settings (CFG — `config/settings.py`)
`GENERATE_QUESTIONS_PER_COMPETENCY` (3) · `GENERATE_BACKEND` (local) · `GENERATE_MODEL` ("" = backend
default) · `GENERATE_MAX_TOKENS` (4096) · **`GENERATE_TIMEOUT_SECONDS` (300)**.

⚠️ **Timeout:** a whole-bank JSON generation on the local 14b measured **71 s** (~2.9k output tokens),
so generation passes `GENERATE_TIMEOUT_SECONDS` to `llm_call` rather than the 25 s
`REASONING_TIMEOUT_SECONDS` that sizes a live suggestion (measured 2026-09-17, #651 E2E — the 25 s
default would time out).

## Verified (2026-09-17, #651)
On the `example_ai_engineer` JD, local `interview-copilot:14b`, defaults: 4 plan steps, 8 questions, 16 rubric
criteria, candidate-side sections flagged as placeholders; the written bundle round-trips through
`load_bundle`. Tests: `tests/test_generate_context.py` (mocked `llm_call`, no GPU).
