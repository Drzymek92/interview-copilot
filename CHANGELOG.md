# Changelog

A capability-level history of the project. This replaces the internal per-session development log
for public readers; the *why* behind each design choice lives in
[`design/DECISIONS.md`](design/DECISIONS.md) (`D#`).

The project is a personal / portfolio build, not a versioned release product, so this is grouped by
capability milestone rather than by semantic version.

## Milestones

### Capture & transcription
- **Live transcript loop** (`live_transcribe.py`): PipeWire/Pulse monitor + mic captured as two
  channels via `parec`, segmented with `webrtcvad` + an adaptive per-channel noise floor
  (`ChannelGate`), decoded on local GPU `faster-whisper`. Proven end-to-end on real call audio.
- **VAD-pause segmentation, not a fixed clock** (D21): segments end on a natural pause, which
  measured markedly better than fixed windows (fixed cuts bisect phrases and starve short windows).
- **Per-segment language detection biased to Polish** (D16): English questions stay English and get
  English answers; Polish stays Polish.
- **Per-language span decode on a detected code-switch** (D26): a segment that straddles a
  Polish↔English switch is cut at the boundary and each side decoded in its own language, instead of
  forcing one language across the whole segment.
- **Speaker attribution** (D-series / #326): each line is tagged `them:` / `you:` by dominant
  channel, so the copilot answers only the interviewer by default.
- **Provisional lines** (D25): the still-open segment is re-decoded periodically to a separate
  `.partial` file so the screen is not blank mid-answer; the final transcript line always supersedes
  it, and the primary transcript contract is untouched.

### Reasoning & suggestions
- **Session context bundle** (D13): JD, company brief, résumé, STAR/answer bank and a structured
  interview plan are loaded at session start so the copilot never starts cold.
- **Honesty boundary** (D22): a `{claim, truth}` list the suggestion prompt enforces as its hardest
  rule — it must never help the user overclaim. A gap stated plainly is recoverable; a claim that
  collapses under a follow-up is not.
- **Deterministic question trigger** (D20): a rule-based pl/en detector fires the model, costing no
  tokens and unable to hallucinate.
- **Salience gate** (D23): a question that clears the trigger fires a suggestion only if the
  already-resident model returns a one-word YES to "did the interviewer just ask the candidate to
  say something?" — cutting wasted calls with zero extra VRAM. An embedding-cosine alternative was
  built, measured, and rejected (kept behind a flag so the rejection stays reproducible).
- **Local-first LLM** (D14 / SI1): local Ollama is the shipped default; a BYOK cloud backend is
  opt-in, announced on every call, and refuses to run without explicit configuration.

### UI
- **Local web dashboard** (D18): a loopback-only FastAPI + websocket UI showing a two-sided
  transcript, salience-gated suggestions, a read-only plan panel, and a bounded-wait meter (P5/D27)
  that tells the user how long since the last line and when the next is due.
- **Suggestion history**: return to a previous suggestion mid-call; completed suggestions are
  persisted locally as JSON lines.
- **Single-app mode** (`--app`): one process manages the recorder as a child and exposes three
  on-screen switches — transcription (a true capture stop that releases the mic and its VRAM),
  suggestions on/off, and answers local vs. local+cloud (with an egress banner).

### Method & measurement
- Built with an in-house lightweight **design + decision OS** (decision register + propagation/audit
  tooling; the CLI itself is internal and not shipped here).
- Accuracy claims are held to a **human-referenced standard** (D24): where no human reference exists
  for spontaneous speech, the project reports *divergence*, never a word-error rate it cannot back.
- A dedicated study of the segmentation cap (`design/MEASUREMENT_segment_cap_400.md`) is an example
  of the measure-before-changing discipline.

## Known limitations
- Linux-only capture path (PipeWire/PulseAudio); a single shared GPU.
- Accuracy on spontaneous speech is characterised by divergence, not a validated WER.
- The plan panel tracks literal `done_signals` mentions only — it does not infer "covered".
- Usability mid-answer (does a live suggestion actually help a human under pressure?) is the open
  question the design docs flag as unresolved.
