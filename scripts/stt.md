# stt.py

## Purpose
The single local speech-to-text seam (D16 — faster-whisper on the lab GPU): every consumer
(`live_transcribe.py`, `spike_capture.py`, `codeswitch_probe.py`, `segment_cap_probe.py`,
`stt_hotwords_probe.py`) calls `Transcriber.transcribe_array`/`transcribe_wav` and nothing else
touches faster-whisper directly. Audio never leaves the machine (SI1).

## Inputs
- A mono float32 numpy buffer in `[-1, 1]` at `settings.SAMPLE_RATE` (`transcribe_array`), or a
  WAV/audio file at any rate/channel count (`transcribe_wav`, resampled via faster-whisper's
  `decode_audio`).

## Outputs
- `TranscriptionResult` (text, per-sub-segment `TranscriptSegment`s, language(s), decode latency,
  `avg_logprob`, whether a code-switch was detected, and `decode_passes` — how many forward passes
  the buffer actually cost, for the P5 meter).

## Key Functions / Classes
| Name | What it does |
|---|---|
| `Transcriber.__init__` | Loads the faster-whisper model once (`STT_MODEL`/`STT_DEVICE`/`STT_COMPUTE_TYPE`). Resolves the `hotwords` knob (#324): `None` -> `settings.STT_HOTWORDS`, `""` -> explicit off, any other string -> that override wins. |
| `Transcriber.transcribe_array` | Plans the decode (`_plan_decode`), runs it, assembles the `TranscriptionResult`. |
| `Transcriber._decide_language` | ONE whole-buffer language vote (D16/#397's proven forced-`pl` fallback when detection is off or unsure). |
| `Transcriber._plan_decode` | Decides `off` / `rescore` / `split` (D26 code-switch handling) based on `STT_CODESWITCH_MODE` and the two-stage switch detector (`_ends_disagree` then `_detect_windows`). |
| `Transcriber._decode_once` | The ONE decode primitive every path (`off`, `_decode_rescored`, `_decode_split`) routes through. Adds `hotwords=` to `model.transcribe()` only when the knob is non-empty (#324) — this is where the jargon-bias knob is threaded in, and the only place it is. |
| `Transcriber._decode_rescored` / `_decode_split` | D26 candidates (a) rescore-and-keep-best / (b) cut-audio-and-decode-each-side. Both call `_decode_once` per language/span. |
| `Transcriber._window_vote` / `_detect_windows` / `_ends_disagree` | Sub-window `detect_language` probes for the switch detector. **These call `Model.detect_language()`, which takes no `hotwords`/`initial_prompt` parameter at all** — language planning structurally cannot see the #324 knob. |
| `weighted_avg_logprob` | Duration-weighted mean `avg_logprob` — the arbiter D26's `rescore` mode uses. |
| `language_runs` / `spans_from_runs` | Pure helpers: collapse per-window language votes into runs, then into contiguous decode spans covering the whole buffer. |

## Dependencies
- Internal: `config.settings` (all STT_* knobs), `scripts.logger`.
- External: `faster_whisper` (deferred import — importing this module for the dataclasses does not
  require it installed), `numpy`.

## Known Gotchas
- **Do not switch to `distil-large-v3`** — it is English-only; forced onto Polish speech it emits
  plausible-looking English nonsense (see `config/settings.py`'s `STT_MODEL` comment).
- **Whisper is not bit-reproducible on hard segments** — temperature-fallback sampling means two
  decodes of the same audio can differ (measured 5/6 identical replays, D25). Any before/after
  measurement needs a noise-floor arm (a baseline decoded twice), not a single pair.
- **`hotwords` (#324) conditions every decode window it is passed to**, not just the terms it
  names — `faster-whisper`'s `get_prompt()` prepends it to every window's prior-context prompt.
  Measured on a real interview call: a 30-term list changed 77.5% of segments' text (vs 2.7% noise
  floor) for one confirmed jargon fix. See `design/MEASUREMENT_stt_hotwords_324.md`. Default is
  empty (off); do not flip the default without re-reading that document.
- `detect_language()` and `.transcribe()` are separate model methods with different parameter
  surfaces — anything added to bias a *decode* (hotwords, initial_prompt, temperature, etc.) must
  be deliberately threaded through `_decode_once`; it will NOT reach language planning by accident,
  and conversely will not automatically apply to a new decode call added elsewhere without going
  through `_decode_once`.

## Open Work
- G8 (`design/OPEN_DESIGN.md`): spontaneous-speech code-switch WER is still open (no human
  reference exists, D24) — `#324`'s hotwords measurement used divergence/term-recall, not WER, for
  the same reason.
- The #324 measurement doc's follow-ups: a shorter per-session-scoped hotwords list, and a human
  WER pass on the segments hotwords changed, before any default flip is reconsidered.
