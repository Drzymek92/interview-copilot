# interview_copilot

[![CI](https://github.com/Drzymek92/interview-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/Drzymek92/interview-copilot/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Use](https://img.shields.io/badge/use-disclosed%20only-orange)

A **disclosed, allowed** live AI copilot for AI-engineer interviews on Microsoft Teams. It is primed
with a per-session context bundle (job description, résumé, STAR/answer bank, interview plan) so it
never starts cold, and runs an **ambient** mode: it transcribes the conversation, tracks the interview
plan, and proactively surfaces talking points, definitions, and code. The candidate only steps in by
typing or picking a suggestion.

It is explicitly **not** a covert or stealth tool (D11): no anti-detection, privacy-preserving by
default, and only for interviews that permit AI assistance.

## Goals

- **O1 — Stay coherent live:** keep the user oriented in the plan (which section, what's covered,
  what's still owed) without them having to think about it.
- **O2 — Answer quality on tap:** for a detected question, surface concise, correct,
  AI-engineer-grade talking points / code, grounded in the loaded session context.
- **O3 — Ambient, low-friction:** useful with zero input; typing or picking a suggestion only deepens.
- **O4 — Disclosed & safe:** no stealth; honours the specific interview's AI policy.

## How it works

```
Teams audio (monitor + mic)
   │  parec (PipeWire monitor + mic), two channels
   ▼
live_transcribe.py  ── VAD segmentation (webrtcvad + adaptive ChannelGate, per channel)
   │                    → faster-whisper (local GPU), per-segment language biased to Polish
   │                    → per-language span decode on a detected code-switch (D26)
   │                    → live_transcript_<stamp>.txt   (+ .partial provisional lines, D25)
   ▼
reasoning.py  ── tails the transcript (D19) → question trigger (D20) → salience gate (D23)
   │              → suggestion from the resident LLM (local Ollama default; BYOK cloud opt-in)
   ▼
dashboard.py  ── local websocket UI: two-sided transcript + bounded-wait meter (P5)
                 + salience-gated suggestions + read-only plan.  Loopback-only, enforced (SI1).
```

Everything decodes **locally** — no interview audio or transcript leaves the machine unless a cloud
LLM/STT is explicitly selected. Privacy is a design invariant (`design/SECURITY_INVARIANTS.md`).

## Stack

- Python 3.11+ · `faster-whisper` (CUDA) · `webrtcvad` · `parec` (pulseaudio-utils) on PipeWire
- LLM via `langchain_openai` over an OpenAI-compatible endpoint — local **Ollama** default
  (`interview-copilot:8b/14b`), BYOK cloud (Claude) opt-in and announced
- FastAPI + websocket dashboard, hand-written UI (no CDN, no webfont — SI1)

## Setup

This has real external dependencies — an NVIDIA GPU, Ollama with a custom-context model, and a
Linux audio stack — so the full walkthrough is in **[SETUP.md](SETUP.md)**. The short version:
install `pulseaudio-utils` + `libportaudio2`, install [Ollama](https://ollama.com) and run
`./ollama/build_models.sh`, then `pip install -r requirements.txt` and `cp config/.env.example
config/.env`. To just run the code/tests (no GPU, no audio), see
[Code-only setup](SETUP.md#code-only-setup-no-gpu-no-audio).

## Quickstart

A complete, **fully synthetic** example session bundle ships at
[`examples/sessions/example_ai_engineer/`](examples/sessions/example_ai_engineer/), so the commands
below run out of the box. Build your own bundle for a real interview per
[`docs/SESSION_BUNDLE.md`](docs/SESSION_BUNDLE.md); real bundles live under
`scripts/inputs/sessions/<your_id>/`, which is **gitignored** (your interview data never leaves your
machine).

```bash
# a one-shot suggestion against the example bundle — no audio, no call
python scripts/reasoning.py --session example_ai_engineer --text "Tell me about a hard ML project"

# is the LLM backend alive?
python scripts/llm_client.py --smoke

# list audio sources, then run the dashboard you actually interview on (single-app mode)
python scripts/live_transcribe.py --list
python scripts/dashboard.py --app --session example_ai_engineer   # → http://127.0.0.1:8765

# tests / lint  (model/live tests self-skip when Ollama is unreachable)
pytest tests/ ; ruff check .
```

## Governance & design docs

Built with a lightweight in-house **design + decision OS**: every locked decision is stated once in
[`design/DECISIONS.md`](design/DECISIONS.md) (`D1`–`D28`) and other docs cite it rather than
restating it. Open questions and design tensions are in
[`design/OPEN_DESIGN.md`](design/OPEN_DESIGN.md), the security invariants in
[`design/SECURITY_INVARIANTS.md`](design/SECURITY_INVARIANTS.md), and a worked measurement study in
[`design/MEASUREMENT_segment_cap_400.md`](design/MEASUREMENT_segment_cap_400.md). A capability-level
history is in [CHANGELOG.md](CHANGELOG.md).

_(The decision register was propagated and audited by a small internal governance CLI; that tooling
is not part of this release, so these docs are included as design documentation — references in them
to internal tooling or absent files (`decision_tools.py`, `ROUTINE_*`, `INDEX.md`, `METHODOLOGY.md`,
`agent/…`, `salience_fixtures.json`) point at that internal material, not files in this repo.)_

## Status

Capture + STT are proven end-to-end on real Teams audio; the live transcript loop, reasoning layer,
salience gate, code-switch repair (D26), provisional lines (D25), and the dashboard with its
bounded-wait meter (D27) are all built. The current open items are the mock/technical-round usability
test (does any of this help a human mid-answer?) and a human-referenced accuracy pass. See
[CHANGELOG.md](CHANGELOG.md) for the capability history and
[`design/OPEN_DESIGN.md`](design/OPEN_DESIGN.md) for the open questions.

## License

[MIT](LICENSE) © 2026 Drzymek92
