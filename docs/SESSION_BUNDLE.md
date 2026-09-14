# Session context bundle (schema v2)

The copilot is primed per interview with a **context bundle** so it never starts cold. A bundle is
a directory:

```
scripts/inputs/sessions/<session_id>/
  bundle.json          # the manifest (this schema)
  job_description.md   # optional referenced files
  company_brief.md
  resume.md
```

A complete, runnable, **fully synthetic** example ships at
[`examples/sessions/example_ai_engineer/`](../examples/sessions/example_ai_engineer/) — read it
alongside this document. To build your own, copy that directory to
`scripts/inputs/sessions/<your_id>/` and replace every field. `scripts/inputs/` is **gitignored**,
so your real interview data never leaves your machine (privacy is a design invariant — see
[`design/SECURITY_INVARIANTS.md`](../design/SECURITY_INVARIANTS.md)).

Load and inspect a bundle without a full run:

```bash
python scripts/reasoning.py --session example_ai_engineer --text "Tell me about yourself"
```

`load_bundle()` **raises** on a missing referenced file or an unsupported `schema_version` rather
than half-loading — a bundle that silently lost its résumé is worse than a hard stop. Any section
marked as a placeholder is **reported on every load** so a stub is never mistaken for real content.

## `bundle.json`

| key | type | required | notes |
|---|---|---|---|
| `schema_version` | int | yes | `2` (v1 still loads, without the honesty boundary). A mismatch is a hard stop. |
| `session_id` | string | no | Defaults to the directory name. |
| `role` | string | no | Shown to the model in the header. |
| `company` | string | no | Shown to the model in the header. |
| `language.spoken` | `"pl"`/`"en"` | no | Default spoken language for the session (falls back to `STT_LANGUAGE`). |
| `language.suggestions` | `"match"`/`"en"`/`"pl"` | no | `"match"` (default) answers each question in the language it was asked in. |
| `job_description` | section | no | See **Sections** below. |
| `company_brief` | section | no | See **Sections**. |
| `resume` | section | no | See **Sections**. Your own background. |
| `answer_bank` | array of STAR entries | no | Prepared answers (see below). |
| `plan` | array of plan steps | no | The interview plan (see below). |
| `honesty_boundary` | array of claim/truth rows | no | **v2. The highest-consequence block** (see below). |

### Sections (`job_description`, `company_brief`, `resume`)
Each is one of:
- an **inline string** — the text itself, or
- an **object** `{"file": "resume.md", "status": "real"}` — load the text from a file in the bundle
  directory. `"status": "placeholder"` marks a stub, which is recorded and printed on every load.

```json
"resume": {"file": "resume.md", "status": "real"},
"company_brief": "Inline text works too."
```

### `answer_bank[]` — STAR entries
Prepared answers, kept as separate fields so a future retriever can match on `tags`.

```json
{
  "id": "star_rag_eval",
  "title": "Built an evaluation harness for a RAG assistant",
  "tags": ["rag", "evaluation"],
  "situation": "...",
  "task": "...",
  "action": "...",
  "result": "..."
}
```
Only `id` and `title` are required; empty S/T/A/R fields are simply omitted from the prompt.

### `plan[]` — the interview plan
```json
{
  "id": "p2",
  "title": "Technical depth: RAG and evaluation",
  "key_points": ["Retrieval quality is measured, not asserted", "..."],
  "done_signals": ["how do you evaluate", "retrieval", "faithfulness"]
}
```
`key_points` orient the model on what to surface for that section. `done_signals` are literal
strings the dashboard's read-only plan panel badges as *mentioned* when they appear in the
transcript — it is a mention, not an inferred "covered" state.

### `honesty_boundary[]` — the hardest rule (v2)
Each row is a `{claim, truth}` pair. The suggestion prompt enforces these **above everything else**:
never suggest wording that makes a listed claim, never soften "has not done X" into "has experience
with X", and answer a question touching one with the true version.

```json
{
  "claim": "that I have run Qdrant in production",
  "truth": "I used Qdrant in a personal project only; the production vector store I worked with was pgvector."
}
```

> **A copilot that helps you overclaim is worse than no copilot** — a gap stated plainly is
> recoverable, a claim that collapses under one follow-up is not. **The boundary only protects
> claims that are *listed*** (D22): for any topic where you must not round up, add a row, or the
> model may helpfully overstate it.

## Authoring tips
- Keep the whole bundle under the model's context window. The custom Ollama models bake `num_ctx`
  to 16384 (see [`ollama/`](../ollama/)); a bundle of roughly 9–12k tokens leaves room for the
  transcript window and the answer. Ollama **silently truncates** an over-long prompt.
- Do not invent facts. If something is not on record, leave a plan/honesty note telling the copilot
  to defer to you rather than filling the gap.
