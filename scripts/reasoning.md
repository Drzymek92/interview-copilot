# `scripts/reasoning.py` — context bundle + suggestions

Subtask #321. Read this instead of scanning the script (~620 lines).
Companion: `scripts/llm_client.py` (the LLM seam), `scripts/live_transcribe.md` (the producer).

## QUICKSTART

```
cd interview-copilot
ollama list | grep interview-copilot         # MUST exist — see the truncation section below
ollama ps                                    # the GPU is shared — check before a call
python scripts/reasoning.py --session example_ai_engineer --watch --wait-seconds 120
```
…then start the transcript loop in a second terminal (`live_transcribe.md` quickstart).
`--watch` attaches to the newest `scripts/outputs/live_transcript_*.txt` and prints a
suggestion under each question it detects. Ctrl-C to stop; it never touches the recording.

One-shot, no audio at all:
```
python scripts/reasoning.py --session example_ai_engineer --text "Jakie ma Pan doświadczenie z LangChain?"
python scripts/reasoning.py --session example_ai_engineer --suggestion-language pl --text "..."
python scripts/reasoning.py --session example_ai_engineer --replay scripts/outputs/live_transcript_<stamp>.txt
python scripts/reasoning.py --selftest-heuristic          # no model, no GPU, no tokens
python scripts/reasoning.py --selftest-salience            # score the D23 salience gate
python scripts/llm_client.py --smoke                      # is the backend alive?
```

## The three pieces

### 1. Context bundle (D13) — `load_bundle()`
`scripts/inputs/sessions/<session_id>/bundle.json` plus the markdown files it names.

| key | shape | notes |
|---|---|---|
| `schema_version` | `2` (v1 still loads) | a mismatch is a hard stop, not a best-effort load |
| `session_id`, `role`, `company` | strings | shown to the model as the header |
| `language.spoken` / `language.suggestions` | `"pl"` / `"match"` | per-session override of `STT_LANGUAGE` / `SUGGESTION_LANGUAGE`; `"match"` answers in each question's own language (default); `--suggestion-language match\|en\|pl` beats both |
| `job_description`, `company_brief`, `resume` | `{"file": ..., "status": "real"\|"placeholder"}` or inline text | `placeholder` is **recorded and printed on every load** |
| `answer_bank[]` | `id`, `title`, `tags[]`, `situation`, `task`, `action`, `result` | STAR kept as fields so a future retriever can match on `tags` |
| `plan[]` | `id`, `title`, `key_points[]`, `done_signals[]` | P3 sketch — #321 uses title+key_points; `done_signals` is carried for the #322 tracker |
| `honesty_boundary[]` | `claim`, `truth` | **v2, and the highest-consequence block.** Claims that must never be made, with what is true. Goes into the prompt as its hardest rule |

A missing referenced file or a bad schema version **raises**. A bundle that silently lost its
resume is worse than a hard stop.

**`example_ai_engineer` is the bundle shipped with this repo** (under `examples/sessions/`) — a
complete, fully synthetic AI-Engineer bundle with no placeholders that the tests and the quickstart
run against. To use the copilot for a real interview, build your own bundle the same way (schema and
authoring notes in [`../docs/SESSION_BUNDLE.md`](../docs/SESSION_BUNDLE.md)) from your own JD,
résumé, STAR bank and honesty boundary, and drop it under `scripts/inputs/sessions/<your_id>/`
(that path is gitignored — your real interview data never leaves your machine).

### The honesty boundary — why it exists
13+ `{claim, truth}` rows that the system prompt enforces above everything else: never suggest
wording that makes one of these claims, never soften "has not done X" into "has experience with X".
A gap stated plainly costs nothing; a claim that collapses under one follow-up ends the process.

**It only protects claims that are LISTED.** Proven on 2026-09-02: with the full bundle loaded the
loop still invented a LoRA-vs-full-training trade-off, because fine-tuning had no row. Adding one
fixed it. **Any new topic area needs its own row** — the model will otherwise round up helpfully.

### 2. Trigger policy (D20, closes G6) — `looks_like_question()`
Deterministic pl/en rule, no model call. Fires on an explicit `?`, an interrogative in the first
three words, or an answer-requesting imperative in the first two — minus a list of Polish discourse
idioms that open with an interrogative and ask nothing.

Position limits are the whole trick: bare `jak` mid-sentence is *as/like*, and English auxiliaries
(`do`, `would`, `can`) only ask by subject inversion, so they must lead. Tokens are de-diacriticked
because Whisper occasionally drops a Polish diacritic.

**Measured** (`--selftest-heuristic`, 60 hand-labelled lines in `tests/fixtures/question_fixtures.json`):
P 0.86 / R 1.00 before the idiom guards → **P 1.00 / R 1.00** after. **Read the second number with
suspicion** — the false positives that motivated the guards came from this same set, so it is
"no known false-positive class remains", not an out-of-sample rate.

Knobs (all CFG): `FIRE_ON_QUESTIONS_ONLY`, `SUGGESTION_MIN_WORDS`, `SUGGESTION_COOLDOWN_SECONDS`,
`SUGGESTION_HISTORY_LINES`. `--all-segments` bypasses the rule for comparison.

### 2b. Salience gate (D23, #386) — `scripts/salience.py`

D20 answers *"is this a question"*; the gate answers *"is it worth a call"*. It runs **after**
D20, on what survived it, and asks the **already-resident suggestion model** for one word:
did the interviewer just ask the candidate to say something? Only the segment's interrogative
sentences are shown to it — `SEGMENT_MAX_SECONDS=30` merges a whole exchange into one line,
and that extraction is the biggest measured accuracy lever (3B: F1 0.57 → 0.88).

On the real 42-min HR screen: **24 suggestion calls → 13, with all 8 substantive turns still
firing.** Bench numbers on those 24 labelled turns: P 0.80 / R 1.00 / F1 0.89, 157 ms median,
**0 MiB extra VRAM** (it is the model the suggestion already loaded).

```
python scripts/reasoning.py --selftest-salience                     # score it; non-zero on regression
python scripts/reasoning.py --selftest-salience --salience-backend embed   # the rejected route
python scripts/reasoning.py --session example_ai_engineer --replay FILE --no-salience   # pre-#386 behaviour
```

**It fails open** (`SALIENCE_FAIL_OPEN=1`): a timeout or an unreachable backend fires the
segment and degrades to pre-#386 behaviour. A wasted call costs tokens; a suggestion the
candidate needed and did not get costs the interview.

**Why not embed-cosine** (which #386 was scoped around): measured and rejected — across the
24 turns the salient ones average cosine **0.510** and the non-salient **0.518**, so there is
no threshold to pick. Two causes: `nomic-embed-text` cannot read Polish (1/7 top-1 vs 4/4 on
the English translations of the same probes), and, more fundamentally, an HR screen is
topically saturated so topic similarity cannot express a **speech act**. Full table and the
four variants: the `scripts/salience.py` docstring. Reproduce with `--salience-backend embed`.

### 3. The suggestion (D14) — `build_messages()` / `suggest()`
`build_messages` is split out so a test can assert the language asymmetry is *in the prompt*
without spending a call. `resolve_languages()` decides the (spoken, target) pair for the turn,
then two rules are picked by whether they differ:

- **`CROSS_LINGUAL_RULE`** (e.g. pl → en): "you will receive Polish text… you MUST write your
  entire answer in English, never in Polish."
- **`SAME_LANGUAGE_RULE`** (pl → pl, or en → en): plain.

**Answer language = the question's language by default (`SUGGESTION_LANGUAGE="match"`, user
decision 2026-09-02).** A Polish question gets a Polish scaffold, an English question an English
one. The question's language comes from the transcript's `(lang)` detection tag on the live path,
or from `detect_text_language()` for `--text`. `match` therefore always resolves to a same-language
prompt. Forcing `--suggestion-language en`/`pl` overrides it (then a Polish question answered in
English uses the cross-lingual rule). The spoken side tracks the real question language even when
forced, so the correct rule is chosen either way.

⚠️ **The two rules are separate on purpose.** One template with the languages substituted in emitted
*"write your entire answer in Polish, never in Polish"* — caught by a test, and it made
`llama3.2:3b` abandon the question and translate the interview plan instead (3/3 reps).

Output contract the model is asked for, and which #322 will parse:
```
POINT: <one sentence — the strongest thing to say first>
- <3–5 bullets, ≤12 words each>
EVIDENCE: <a fact from the resume/answer bank, or "none">
```

## The consumption seam (D19)
`follow_transcript(path)` tails `live_transcript_*.txt` and yields `(stamp, speaker, language, text)`, including lines
written before it attached. `live_transcribe.py` fsyncs every line, so nothing is lost and **no edit
to that script was needed**.

Why tailing and not an in-process callback: the recorder holds the one irreplaceable artifact of a
live interview and must not share a process with a network call that can hang or raise; a tailing
consumer attaches, dies and re-attaches without touching the recording; and a finished transcript
replays through the identical code path. Cost is `TRANSCRIPT_POLL_SECONDS = 0.25` against a 5–30 s
segment cadence — 1–5% of a budget already dominated by segment length. **#322's dashboard inherits
this seam.**

**`stop_event` (#607).** `follow_transcript` takes an optional `threading.Event`, checked between
lines, so the dashboard app can end a tail cleanly when its managed recorder is stopped and a new
transcript file will follow. `None` (the CLI/replay default) leaves the tail byte-identical.

## Live switches — `Controls` (#607 single-app mode)
`Controls` is a thread-safe object the dashboard's `--app` mode flips while the loop runs; `run_ambient`
reads it **per line** at the fire decision (with `controls=None` the loop uses its fixed args and always
fires — CLI/replay unchanged). Two fields matter mid-call: `suggestions_on` (off skips the whole
trigger→gate→fire block, so the transcript + meter keep flowing with no model or GPU touched) and
`backend` (`local` ↔ `cloud`, the "answers: local vs local + api" switch). The switch UI and the
recorder lifecycle live in `dashboard.py` (`AppController`, `POST /control`); this module only supplies
the flag object and honours it. SI1 is unchanged — the cloud path still announces its egress.

## Backends and egress (SI1)
`llm_client` selects on `REASONING_BACKEND`:

| | endpoint | key | egress |
|---|---|---|---|
| `local` (**default**) | Ollama `localhost:11434/v1` | placeholder | none |
| `cloud` | any OpenAI-compatible base | `CLOUD_API_KEY`, BYOK | **yes — announced** |

Cloud refuses to run unless `CLOUD_BASE_URL`, `CLOUD_MODEL` and `CLOUD_API_KEY` are all set; there
is deliberately no default provider or model, and `announce_backend()` prints a banner naming the
destination host on every client build. **The cloud path has never been executed** — no key was
available when it was written. Treat it as unproven code.

## Cost & latency per suggestion
Measured 2026-09-01, local, model resident, 3 reps/cell:

_Superseded for the real bundle — see the note below the table._

| model | out-lang | median | out tok | usable? |
|---|---|---|---|---|
| llama3.2:3b | en | 0.82 s | 94 | **no** — meta-instructions, not an answer |
| llama3.2:3b | pl | 1.09 s | 87 | weak — grounded, drops the format |
| **llama3.1:8b** | **en** | **1.43 s** | **100** | **yes** |
| **llama3.1:8b** | **pl** | **2.49 s** | **175** | **yes** |

**With a representative ~9k-token bundle: ~8.7–8.9k prompt tokens in, 80–260 out,
1.3–4.2 s per suggestion** on `interview-copilot:8b`. Prompt tokens are near-constant (the whole
bundle is resent every time) — that is the lever if #322 needs to cut cost. Zero currency cost on
the local path. **The ambient loop can afford to fire on every turn:** a few seconds on top of a
line that already arrives 5–30 s late. VRAM 7.0 GB of 15.9 GB, leaving room for `large-v3-turbo`.

## ⚠️ Ollama silently truncates an over-long prompt
Measured 2026-09-02, and it had already broken the copilot before anyone noticed.

Stock `llama3.1:8b` ships `num_ctx=4096`. The real bundle is ~8.7k prompt tokens. Ollama does
**not** error — it discards the overflow and reports the *truncated* count as `prompt_tokens`. A
37.8k-char prompt came back as **2050 tokens**, with the output format and the entire honesty
boundary among the ~80% thrown away. The only symptom was a model that translated the question
instead of answering it.

- Its **OpenAI-compatible endpoint ignores `options.num_ctx`** (probed: identical 2050 with and
  without it). The window can only be raised by baking it into a model.
- **The fix, and it must be repeated on any fresh machine:**
  ```
  printf 'FROM llama3.1:8b\nPARAMETER num_ctx 16384\n' > ~/Modelfile.interview_copilot
  ollama create interview-copilot:8b -f ~/Modelfile.interview_copilot
  ```
  **The Modelfile must live under `$HOME`** — Ollama is a snap and its confinement cannot read `/tmp`.
- `llm_client` now logs a **`PROMPT TRUNCATED`** error above 7.0 chars/token, and
  `test_live_prompt_is_not_silently_truncated` asserts the real bundle arrives whole.

## Gotchas
- **Speaker attribution (G9/P4, #326 — DONE 2026-09-02).** Each transcript line is tagged
  `them:` (monitor/interviewer) or `you:` (mic/candidate) by its dominant channel, and the loop
  answers **only the interviewer** by default (`settings.ANSWER_SPEAKER="them"`). `--answer-speaker
  any` restores the old channel-blind firing; `you` answers your own turns instead. An **untagged**
  line — a pre-#326 transcript, or a `--no-mic` monitor-only run — is treated as `them`, so nothing
  is silently dropped. Accuracy relies on headphones (no interviewer bleed into the mic); on
  speakers the mic hears both sides and the tag degrades. The conversation history fed to the model
  is **labelled by speaker** (`Interviewer:` / `You:`), and a skipped `you:` turn still enters
  history — context the copilot should not repeat, just not answer.
- **Grounding is instructed, not enforced.** Asked about RAG/vector DBs — absent from the bundle —
  the model substituted the nearest STAR entry rather than saying "not supported" (#324).
- **The cooldown is wall-clock.** `--from-wav` and `--replay` compress time, so consecutive segments
  land inside the 8 s window and get skipped. Set `SUGGESTION_COOLDOWN_SECONDS=0` for replays.
- **`--watch` picks the newest transcript by filename stamp.** Start `live_transcribe.py` first, or
  pass `--wait-seconds` and start it within that window.
- Placeholder sections print a `PLACEHOLDER:` line at every load. That is not noise — it is the
  guard against a fixture being mistaken for real content.

## 2026-09-14 latency notes (session 18)
- `SYSTEM_TEMPLATE` ends with `{language_rule}` (was at the top): Ollama caches the prompt PREFIX, so a language flip must not change the first token of the system block. Measured 8.4 s -> 0.2 s TTFT on a switch.
- `prefill_bundle(bundle, backend, model)` — one 1-token local call right after `gate.warm()` in `follow_transcript`, so the ~11k-token bundle prefix is cached before the first question (0.22 s TTFT instead of ~7 s). Local only; best-effort; returns seconds taken.
- The model must stay RESIDENT: the `/v1` client cannot send `keep_alive`, so `launch_copilot.sh` pings the native API every 120 s with `keep_alive: 3h`.
