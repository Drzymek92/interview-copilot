# `scripts/live_transcribe.py` — continuous live transcript loop

Subtask #320. Read this instead of scanning the script.

## QUICKSTART (the 09:55 card — 10 lines)

```
cd ~/Desktop/Claude_Projects/projects/interview_copilot
python scripts/live_transcribe.py --mic <your-mic-source>   # get the exact name from --list
```
1. It prints `them (monitor)` / `you (mic)` — check both lines, then wait for `READY`.
2. Tell HR you are recording (D11): *"Nagrywam rozmowę, żeby zrobić sobie notatki — czy to w porządku?"*
3. Working = a line like `[00:07-00:36] Dzień dobry, ...` appears. **Expect one line every
   ~10-30 s, not instantly** — long segments are what buys accuracy; silence prints nothing.
4. Nothing after a minute of talking? `python scripts/live_transcribe.py --list`, then pass the
   monitor of the sink Teams plays to: `--source <name>.monitor`.
5. **Ctrl-C once** to stop; it drains pending decodes and prints the three output paths.
6. Score it, then delete the WAV (D11).

## What it does

`parec` streams two sources at 16 kHz mono — the sink **monitor** (them; the monitor does *not*
contain your own voice) and your **mic** (you). They are written to one **stereo WAV**
(L = them, R = you) so speakers stay separable, and **mixed to mono** for Whisper.
`scripts.stt.Transcriber` is loaded **once, before capture starts** (~1.2 s warm), and runs in a
worker thread so the GPU never stalls the capture loop.

## Segmentation — VAD, never a fixed clock

Each 20 ms frame is judged per channel by `ChannelGate` = `webrtcvad` **AND** an adaptive noise
floor (frame RMS must exceed `max(VAD_SPEECH_MIN_RMS, 3 x 10th-percentile RMS of the trailing
20 s)`). A segment opens on speech onset (with `SEGMENT_PREROLL_SECONDS` of audio from *before*
the onset, so it never starts mid-word) and closes on a **pause**. All knobs are in
`config/settings.py` — none are hardcoded here.

The only fixed cut is the `SEGMENT_MAX_SECONDS` (30 s) safety valve for an uninterrupted
monologue; it carries `SEGMENT_CARRYOVER_SECONDS` of audio into the next segment and
`drop_overlap()` removes the duplicated words from the text.

**Why the noise floor exists (measured 2026-08-31):** `webrtcvad` alone calls steady mic hiss
speech. On the first live two-stream run the webcam mic's noise never let the VAD see silence, so
every segment ran to the 30 s cap and decoded to garbage (`"...to have worked on..."`, `"KONIEC!"`).
With the floor, the same setup produced clean pause-bounded segments of 4.7-8.4 s.

`--segmentation fixed` exists **only** to reproduce fixed-window baselines for A/B measurement
(numbers in `config/settings.py`). It is not a live mode.

## Outputs (one timestamp per run, TRK)

| File | Purpose |
|---|---|
| `scripts/outputs/live_audio_<stamp>.wav` | stereo recording, L = them / R = you. **Delete after scoring (D11).** |
| `scripts/outputs/live_transcript_<stamp>.txt` | human transcript: `#` header (run_id, sources, policy, model) + `[mm:ss-mm:ss] them\|you (lang): text` (speaker = dominant channel, G9/#326; `lang` = detected question language) |
| `scripts/outputs/live_transcript_plain_<stamp>.txt` | **scorer input** — text only, hand straight to `score_transcript.py` |
| `scripts/outputs/live_transcript_<stamp>.partial` | **provisional lines (D25)** — decodes of segments that were still OPEN. Same format, same parser; the `.txt` line for that segment always supersedes it. Absent when `--no-partials` / `PARTIAL_DECODE_ENABLED=0`. Delete with the WAV (D11). |

Every line is `flush()`ed and `fsync()`ed as it is produced, so a hard kill cannot truncate the
transcript. The WAV is written frame-by-frame (not temp-then-rename): a recorder must survive a
crash with a partial file rather than lose the call.
One row per run is appended to `logs/runs.csv`.

## Provisional lines (D25 / #399)

The screen is blank for a **measured median 28 s** mid-answer, because a line cannot appear until
its segment closes *and* decodes. Every `PARTIAL_DECODE_SECONDS` (default 5 s) the recorder takes a
**growing prefix** of the segment that is still open, decodes it on the model it already has loaded,
and appends the result to the `.partial` file.

- **The recorder writes it, never a consumer.** The cost is VRAM, not compute: a second Whisper in
  the dashboard is ~2.2 GB against the ~2.1 GB the 14B leaves, while the recorder's model is idle
  97.4 % of a call and a 5 s cadence adds only ~10 % GPU duty.
- **Consumers tail it with no new code.** `reasoning.follow_transcript` + `parse_transcript_line`
  read it unchanged. Join a provisional to its final line on the **START stamp**, which is fixed at
  speech onset; the end stamp is "as of this decode" and grows.
- **Nothing provisional can reach the `.txt` or the plain scorer file.** `write_partial` is the only
  writer of the `.partial`, and `write_line` the only writer of the other two — asserted by tests,
  and demonstrated: the same WAV run with partials off and on produced **byte-identical** transcript
  and scorer files.
- **A provisional may have no final line at all** (its segment can still be dropped as silence), and
  an unchanged re-decode is not written twice — an update appears only when there is one.
- **A provisional whose final has already landed is dropped before it costs a decode.** Measured on
  the 2026-09-05 real-time self-test: because finals have GPU priority, a snapshot taken at 02:18
  can be decoded *after* the segment closes at 02:19 and its final is written — it landed 0.39 s
  late carrying a wrong last word. The ones that arrive on time arrived **4.0 s** ahead of their
  finals.
- **Finals always win the GPU.** Snapshots go into a one-slot newest-wins mailbox that the decode
  worker reads only when no final segment is waiting, so a provisional cannot queue ahead of a
  transcript line. The one cost a final can still pay is sitting behind **one** provisional decode
  already in flight (0.248 + 0.0144·audio_s, max 0.96 s measured).
- **`--no-partials`** (or `PARTIAL_DECODE_ENABLED=0`) turns it off; `--partial-seconds N` retunes the
  cadence. Never active under `--segmentation fixed` — that is the A/B harness.
- `--pace N` — with `--from-wav`, feed at N x real time (1.0 = a real call's clock; 0 = as fast as
  possible, the default). Required to exercise the provisional cadence offline.
- **Survival, not accuracy (D24).** A 5 s prefix keeps 0.86 of its tokens into the final line
  (0.94 @10 s, 0.96 @15–20 s) over 86 prefixes of the real HR call. No human reference exists for
  that audio, so this measures **churn**. Whether a provisional line *helps* is still unmeasured and
  needs the #323 dry-run.

## Proving it without a human

```
python scripts/live_transcribe.py --selftest-sink alsa_output.pci-...analog-stereo --no-mic
python scripts/live_transcribe.py --from-wav scripts/outputs/ab_source_en.wav --no-record
```
`--selftest-sink` plays a WAV into a sink, captures that sink's `.monitor`, and auto-stops —
the exact code path a real call uses. `--from-wav` pushes a file through the whole loop
(segmentation + STT + files) as fast as it decodes; that is how the A/B above was measured.

**`--from-wav` needs `--pace 1.0` to demonstrate provisional lines.** Unpaced, audio time runs
~45x faster than the GPU, so every snapshot is displaced by a fresher one before the decoder is free
— measured on the 42-min call: `414 displaced, 0 written, provisional GPU duty 0.0%`. `--pace N`
feeds at N x real time, which is the only way the offline harness exercises the D25 cadence at all;
`--pace 1.0 --seconds 600` on that call gives `90 written from 91 decodes, 0 displaced`.
`--selftest-sink` also plays in real time, through PipeWire, and is the closer analogue of a call.

**The transcript is NOT byte-reproducible run to run, and that is the model, not D25.** Six full
replays of the same WAV (3 with partials on, 3 off) gave five byte-identical transcripts and one
that differed by 2 lines in a single 30 s segment — a partials-ON run, but the other two
partials-ON runs matched the OFF group exactly. On that segment the decode took 1.85 s against
1.21 s: Whisper's temperature fallback retried, and the retry samples. So `--from-wav` is the right
harness for the partials-on/off *invariant* only in the aggregate — **compare more than one pair**,
or a single unlucky pair will tell you the feature broke D19 when it did not.

## Gotchas

- **`--mic auto` picks the DEFAULT input, which is often the webcam mic**, not your external USB mic.
  Always read the `you (mic)` line before the call, or pass `--mic` explicitly.
- **Per-segment language detection (2026-09-02).** Whisper detects each segment's language
  (biased to Polish — see `config/settings.py::STT_DETECT_LANGUAGE`) instead of forcing `pl`, so
  an English question stays English (`"Tell me"`, not `"Powiedz mi"`) and is tagged `(en)`; a Polish
  question with English tech terms detects `pl`. The detection is only trusted when it is a
  candidate language above `STT_LANGUAGE_MIN_PROB`, else it falls back to `pl`. Set
  `STT_DETECT_LANGUAGE=0` to restore the old forced-pl path. Proven on real audio: an English read
  → `(en)`, a real spontaneous Polish read → `(pl)` on every segment.
- A content-poor window makes Whisper hallucinate fluent text in the decoded language. The peak gate
  (`SEGMENT_MIN_PEAK`) and the noise floor are what keep those windows from ever being decoded.
- Ctrl-C once = graceful (drains the queue); twice = immediate. Verified: no orphaned `parec`
  after either.
