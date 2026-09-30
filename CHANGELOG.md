# Changelog

A capability-level history of the project. This replaces the internal per-session development log
for public readers; the *why* behind each design choice lives in
[`design/DECISIONS.md`](design/DECISIONS.md) (`D#`).

The project is a personal / portfolio build, not a versioned release product, so this is grouped by
capability milestone rather than by semantic version.

## 2026-09-30 — hub, practice track, review

- **One hub process (D35)** — `scripts/app_hub.py`: an opening menu (Start a call · Train · Review ·
  Generate context · Settings) and a view per mode; `launch_copilot.sh` now starts it.
- **UI-saved settings (D36)** — `scripts/user_config.py` whitelist registry; precedence
  CLI > env > saved > default; saved to the gitignored `config/user_settings.json`.
- **Offline practice track (D29–D31)** — `generate_context.py` turns a JD (+ CV, now PDF too via
  PyMuPDF, D37) into an interview-side bundle with a question + rubric bank; `training.py` runs a
  mock interview with graded answers.
- **Live Q&A capture + post-interview review (D32–D34)** — `review.py` grades captured
  question/answer pairs against the JD/CV (rubric or holistic).
- **Generic default bundle (D37)** — every mode runs with no bundle selected.
- **Strategic tactics** in the bundle, empty-honesty-boundary fallback rule (D22 amended), mic-only
  practice capture, `STT_HOTWORDS` knob (default off).
- **Install fix** — `webrtcvad` → the maintained `webrtcvad-wheels` fork (fresh venvs lack
  `pkg_resources`; prebuilt Windows wheels).

## Corrections

- **2026-09-23 — local-model VRAM figure corrected (`config/settings.py`).** The comment block
  above `LOCAL_MODEL` claimed a live end-to-end peak of *11.7 GB of 15.9* with Whisper resident.
  That contradicted its own table two lines above (`interview-copilot:14b … 11.5 GB → 13.8 / 15.9`)
  and the `~2.1 GB headroom` note below it. Re-measured independently on the reference machine
  (RTX 5060 Ti, 15.9 GB usable) with a VRAM probe — cold, card cleared first — running the real
  live shape with Whisper `large-v3-turbo` held resident while the 14B generates: **peak 13.3 GB
  of 15.9**. Two further probes decompose it as **11.2 GB** for the 14B alone (44.7 tok/s, 3.5 s
  cold start) and **2.2 GB** for Whisper alone, which sum to 13.4 and corroborate the end-to-end
  figure rather than merely asserting it.
  **Practical effect:** running the 14B leaves about **2.6 GB** spare on a 16 GB card, not ~4.2 GB.
  The advice is unchanged — if something else claims VRAM mid-call, fall back with
  `--model interview-copilot:8b`. Comment only: no code, configuration value or behaviour changed.

## Milestones

### Capture & transcription
- **Live transcript loop** (`live_transcribe.py`): monitor + mic captured as two channels,
  segmented with `webrtcvad` + an adaptive per-channel noise floor (`ChannelGate`), decoded on local
  `faster-whisper`. Proven end-to-end on real call audio (Linux).
- **Cross-platform capture** (`audio_backend.py`): a pluggable backend auto-selected per OS — `parec`
  on Linux, `sounddevice`/PortAudio **WASAPI loopback** on Windows, `sounddevice` + a virtual
  loopback device (e.g. BlackHole) on macOS. Everything above the backend consumes the same fixed
  int16 mono 16 kHz frames, so segmentation/STT are unchanged. The Windows/macOS paths are
  unit-tested (mixdown/resample/framing/selection) but not yet hardware-verified.
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
- Capture runs on Linux/Windows/macOS, but only Linux is hardware-verified; macOS needs a virtual
  loopback device for the interviewer's side and runs STT on CPU. A single shared GPU on Linux/Windows.
- Accuracy on spontaneous speech is characterised by divergence, not a validated WER.
- The plan panel tracks literal `done_signals` mentions only — it does not infer "covered".
- Usability mid-answer (does a live suggestion actually help a human under pressure?) is the open
  question the design docs flag as unresolved.
