# Open Design — Priorities & Architectural Gaps  (LIVING)

Tracks **what still needs deciding/designing** — distinct from `DECISIONS.md` (settled), the SPEC
milestones (build sequencing), and the rationale notes (settled reasoning).

**How to use:** when a gap is resolved, record the decision in `DECISIONS.md` (get a `D#`), then
**move the gap to "Recently closed" with that `D#`** and propagate per the routine. Add new gaps as
they surface. Keep IDs stable (`G#`, `P#`). This is the standing answer to "what's unresolved and
what's next."

_Last reviewed: 2026-08-28 (18 decisions locked; D11–D18 this project's). Deadline: **interview
Tue 2026-09-01 10:00** — dry-run Monday._

---

## Priority design points (ordered — each unblocks a stage/milestone)
| P# | Design point | Resolves gaps | Unblocks |
|----|--------------|---------------|----------|
| ~~P1~~ | ~~**Audio-capture spike**~~ — **DONE 2026-08-31, G2 PASS** (monitor→STT measured; see Recently closed). | G2 | — |
| ~~P2~~ | ~~**Question/turn detection tuning**~~ — **CLOSED 2026-09-01 → D20** (deterministic pl/en rule, measured; LLM classify rejected). The *consumption seam* half of P2 closed as **D19** (consumers tail the transcript file). | G6 | — |
| P3 | **Interview-plan schema** — the structured format of the plan the tracker consumes (steps, done-signals). **Sketched 2026-09-01** in `bundle.json` (`plan[]` = `id`/`title`/`key_points`/`done_signals`, loaded as `reasoning.PlanStep`) — enough for #321 to prime the prompt. Still open: how a step is *marked covered* live. | G7 | plan tracker (#322) |
| P4 | **Speaker attribution in the transcript (NEW 2026-09-01)** — the live transcript is mixed mono, so D20's rule cannot tell the interviewer's question from the candidate's, and the loop will fire on the user's own speech. The stereo WAV already separates L=them/R=you; the transcript throws it away. *Path:* per-channel gate state is already computed in `live_transcribe.ChannelGate` — carry a `them`/`you` tag onto the segment and into the transcript line. | G9 | #322 · #324 |
| P5 | **What the dashboard shows while the interviewer is still talking (NEW 2026-09-04) — A SHIPPED 2026-09-04 (#322); C SHIPPED WHOLE 2026-09-05 (D25/#399) — recorder AND consumer.** A segment-only dashboard is blank for a median **28 s** mid-answer (measured; see Recently closed). Four options were costed; the user picked **A now** and approved **C** as the seam amendment. **A — bounded-wait meter, SHIPPED** in `scripts/dashboard.py` (`meter_state()`, a pure tested function, computed server-side so the browser derives nothing): it renders time since the last line plus the ceiling the line is due within, needs **no new data, no new seam and no GPU**, and renders **no speech**, so it cannot be wrong about the interview. **The ceiling is NOT `SEGMENT_MAX_SECONDS` — corrected on re-measure at build time.** A line cannot reach the screen until its segment closes *and* is decoded; scoring the same 109 arrival gaps as `end + decode` (D25's own model) a bare 30 s ceiling holds only **84/109**, because 54% of segments end exactly at the cap and land at 30.1-30.4 s. The shipped ceiling is `SEGMENT_MAX_SECONDS + METER_DECODE_ALLOWANCE_SECONDS` = **31.0 s**, which holds **108/109**; the one true overrun is **43.2 s**, a silence *between* segments that no segment cap bounds. (This row previously read 107/109 / 2 overruns / max 44 s, measured on end stamps alone — same call, same conclusion, ceiling corrected.) **The overrun state is built and was demonstrated**, not assumed: past the ceiling the meter withdraws the promise ("overdue by N s") instead of showing a bound it has already missed, caught live in a real-time replay of the call's own 43 s gap. Proven end-to-end by a replay of the real 42-min call: 110 lines two-sided, D20 24 triggered, D23 12 fired / 12 dropped, 69 overdue ticks. **C — provisional line → D25, recorder half BUILT 2026-09-05 (#399):** `live_transcribe.py` re-decodes the still-open segment every `PARTIAL_DECODE_SECONDS` (5 s) and appends it to a separate `live_transcript_<stamp>.partial` that `follow_transcript`/`parse_transcript_line` read unchanged; the join key to the final line is the **start stamp**. **D19's contract is demonstrated intact, not asserted:** the same WAV with partials off and on gave **byte-identical** `.txt` and plain scorer files. Snapshots go to a one-slot newest-wins mailbox the decoder reads only when no final is waiting, so a provisional cannot queue ahead of a transcript line and break the 31.0 s ceiling — **the residual cost is stated: a final can still sit behind ONE in-flight provisional decode (max 0.96 s measured), one decode more than the 1.0 s allowance covers.** A real-time self-test (`--selftest-sink`, 229 s) caught a failure mode the design had not anticipated: **a provisional landed 0.39 s AFTER the final for its own segment, carrying a wrong last word** (`terenie` where the final said `TELU`) — precisely because finals have GPU priority. A provisional whose final has already landed is now dropped before it costs a decode; the fixed build re-run on the same harness gives `4 decoded, 2 written, **2 superseded before decode**, 0 displaced`, with the wrong-word line gone. The two that are written arrive **+4.02 s** and **+4.04 s** before their finals, each a truthful prefix of the question being asked. **The consumer is BUILT 2026-09-05 and demonstrated on a paced replay of the real call** (`--from-wav --pace 1.0 --seconds 600`, a flag added because an unpaced replay outruns the GPU ~45x and writes **zero** provisional lines): the dashboard tails the `.partial` through the same `follow_transcript` seam into a **single slot** that is never a transcript line, renders it dashed / dimmed / labelled **provisional** with no animation at all, and clears it when a final at or past its start lands — the clear published AFTER the line, so no frame shows neither. Numbers from that run: **26 finals, 90 provisional lines, all 90 superseded, 0 orphaned, 0 left standing**, GPU duty **2.3% finals + 6.2% provisional**, and the screen showed something a **median 16.0 s earlier** than the final line (max 25 s) against the 28 s blank window. **The residual cost is now bounded, not just stated:** 0 of 28 finals waited behind an in-flight provisional (a small-sample zero — ~1.7 were expected at that duty), and the structural bound is ONE provisional decode (one-slot mailbox, queue re-checked after each), measured max **0.690 s**, which can put a worst-case max-length segment **0.54 s past the 31.0 s ceiling**. **RESOLVED 2026-09-05 → D27 (#428): the allowance stays 1.0 s, and the premise this row stated it under was wrong.** The meter compares an arrival GAP, so what the allowance covers is the latency **STEP** between consecutive lines, not a decode — a uniform slowdown cancels (verified on the real call's own arrival wall clocks: residual mean +0.010 s, median −0.001 s over 109 gaps). Under forced collisions (partials at 1 s, **28/57** finals behind an in-flight provisional, worst **0.767 s**) the measured false-overdue count is **0/51** — frequent collisions hit consecutive lines alike and cancel, so a **rare** collision is the one that shows, which is the opposite of what this row assumed. Measured band: nothing changes between **0.75 s and 1.5 s**; below that the knob bites (0.50 s → 4/112), above ~2.0 s the ceiling passes the one genuinely anomalous event on the real call — a **2.0 s capture stall** (line 29: gap 32.31 s, spacing 30 s, decode 0.70 s vs 0.38 s), the only residual above rounding noise in 105 gaps and the one thing that actually breached 31.0 s. Residual at 1.0 s, named: **1 false overdue in 112** back-to-back gaps on a full-call paced replay, by **0.193 s** ≈ 0.8 of one 0.25 s tick, and that one is a Whisper temperature-fallback decode outlier, not D25 and not D26. **The meter deliberately does NOT count a provisional as an arrival.** **Still unmeasured: whether a provisional line HELPS a human mid-answer** — every number here is churn and latency, not help, and only #323 can answer it. Rejected options were B (a speaking indicator: measurement kills it, someone is speaking **99.3%** of the call) and D (firing the suggestion off provisional text: at a 5 s prefix the D20 trigger fires on only 9/24, and D22's unauditable-suggestion reasoning does apply to a *suggestion*). | — | **#322 built A** · **#399 built C whole (recorder + consumer)** · help unmeasured → #323 |

> Rule of thumb: design just-in-time, one priority ahead of what you're building.

## Architectural gaps (open unknowns)
- **G8 — Polish/English code-switching (NEW 2026-08-31; DOWNGRADED same day, still open).**
  The interview is **mostly Polish with English fragments** (user). STT solved
  (`large-v3-turbo`, forced `pl`).
  *First measurement (spontaneous speech):* both models garbled the Polish→English seam.
  *Second measurement (scripted read, `scripts/inputs/test_script_pl.txt`, 105 words):* **WER 3.8%,
  code-switch terms 10/10 survived** — the seam failure did NOT reproduce.
  **Do not read the second result as "solved".** The two tests differ in a way that explains the
  gap: test 2 was **read speech** (steady pace, no disfluency, clean boundaries); test 1 was
  **spontaneous**, which is what a real interview is. The working hypothesis is therefore *"the seam
  degrades with speech clarity, not with code-switching per se"* — and the interview will be
  spontaneous. **Expect real WER materially above 3.8%.**
  *(a) MEASURED 2026-09-04 on the real 42-min HR call — the seam failure REPRODUCES on spontaneous
  speech, and it lands on exactly the wrong words. **(a) stays open**: what follows is
  divergence + a qualitative failure, **not a WER**, because no human reference exists yet (D24).*
  **The finding.** The live loop's two 30 s segments that STRADDLE a language switch were decoded as
  Polish and came out as translated gibberish, while an offline `large-v3` re-decode of the same
  audio got the English right. At `[14:39-15:09] them (pl)` the interviewer's *"so could we switch
  into English maybe for a while?"* became *"Ok, więc może byśmy przesłuchać do angielskiego?"* and
  **her actual question was destroyed** — *"So maybe could you describe me one of the biggest…"* →
  *"Może byś mi przesłuchać jedną z największych profesjonalnych wydarzeń, coś, co jesteś bardzo
  organy…"*. The switch back at `[17:53-18:23]` broke the same way. For a copilot whose whole job is
  answering `them`'s questions (D20/D23), garbling *the question* at a switch is a functional
  failure, not a WER blip. Once a segment is wholly English the loop recovers on its own — the
  damage is bounded to the straddling segment (~1 per switch, 2 in this call).
  **Mechanism (hypothesis, not yet tested):** `stt._decide_language` picks ONE language per segment;
  a segment that straddles a switch is majority-Polish, so detection returns pl (or falls below
  `STT_LANGUAGE_MIN_PROB=0.7` and falls back to pl), and Whisper *translates* the English inside it.
  `SEGMENT_MAX_SECONDS=30` makes the straddling segment long, so a whole exchange rides on that one
  language call.
  **Numbers, named for what they are** (`divergence_report_20260904_213442.json`): live vs offline
  agreement **83.8%** of offline tokens (87.9% of live tokens) — i.e. ~16% of the offline decode's
  content is absent from the live transcript; **22.5%** divergence over 20 s time-tiled windows;
  live seam **duplication 0.9%** (49/5621 tokens repeat the previous segment — `SEGMENT_CARRYOVER`).
  All three are **machine-vs-machine**; neither side is ground truth, and the offline decode is
  demonstrably capable of being the worse one (it hallucinated a 6× repetition loop at 11:00–11:20).
  **Code-switch term recall on this call's OWN vocabulary** (19 terms extracted from the call and
  recorded in `scripts/inputs/codeswitch_terms_<session>.txt` — `DEFAULT_TERMS` describes the
  scripted fixture and was never about this call): **live 18/19, offline 19/19**. The single loss is
  *"Project Coordinator"* → *"projekt koordynator"*. So **embedded English nouns inside Polish
  survive**; it is the **switch of the utterance's language** that breaks. That is a sharper claim
  than "the seam degrades with clarity" and it narrows (b).
  *Residual, and it is the whole point of (a):* **no true WER exists for spontaneous speech.** The
  sample pack that would produce one is built and waiting — 10 × 20 s windows (4 random-control,
  4 worst-divergence, 2 english-stretch) at `scripts/inputs/wer_reference_20260904_213442.txt` with
  clips at `scripts/outputs/wer_sample_20260904_213442/` — and needs the user's ear (offered
  2026-09-04, deferred by the user). Until those `TRUE:` lines are filled, this project still has
  **no measured accuracy number on spontaneous speech**, and 3.8% remains a read-speech figure.
  *(d) ANSWERED + CLOSED 2026-09-05 → **D26** (#397). Measured on the real call's own audio, all 110
  segments, boundaries reproduced exactly from the WAV through the live gates and segmenter.*
  **The hypothesis in (a) was half right, and the half that was wrong is the important half.** The
  mechanism is not "detection falls below `STT_LANGUAGE_MIN_PROB` and falls back to pl": at
  `[14:39-15:09]` the whole-segment vote is `pl` **p=0.58** (it does fall back) but the segment is
  **70% English**, and at `[17:53-18:23]` the vote is `pl` **p=0.99** — confident, and wrong about
  the 9 s of English the segment opens with. So `STT_LANGUAGE_MIN_PROB` catches one failure and not
  the other, and **the whole-segment probability is not the uncertainty signal this needs.** Nor is
  detection "only looking at the opening window": faster-whisper encodes the whole ≤30 s segment as
  one window and emits **one** language token for it. Sub-window detection over the *same* audio
  recovers the truth in both (votes `pl,pl,en,en,…` and `en,en,pl,pl,…`), which is what D26 triggers on.
  **Both candidates were built and measured; `split` won.** (a) *rescore* — decode twice, keep the
  higher duration-weighted `avg_logprob` — repairs both segments, but it still picks ONE language for
  a straddling segment and can only choose which half to sacrifice: on `[14:39-15:09]` it correctly
  chose `en` for 70% of the audio and **translated the Polish opening into English**. It is also the
  more expensive of the two on the segments it fires on (+1.87 / +2.35 s vs +1.31 / +1.34 s).
  (b) *split* — cut the **audio** at the detected boundary, decode each side in its own language —
  gets **both** halves right, and is admissible where D21's fixed-clock chunking is not because it
  changes **no segmentation at all**: the line keeps its VAD start/end, its single `(lang)` tag, and
  `SEGMENT_MAX_SECONDS` stays #400's lever (now settled → **D28**: it stays 30 s). **Also measured and NOT used:** `argmax(avg_logprob)`
  without the switch gate flips four ordinary Polish segments on margins of 0.013–0.072 nats — the
  gate is load-bearing, not decoration.
  **The control set was the whole call, chosen before any result was seen** (the two failures are 2
  of 110; every other segment is the control). Across 2+2+2 runs, `split` changes **exactly the two
  straddling segments and none of the other 108**, in all four cross-pairs; run-to-run noise floor
  **0/110**. Whole-call token divergence attributable to the fix: **1.19%** (`rescore` 1.40%) — a
  *divergence*, not an accuracy claim (D24), and `diff_transcripts.py` localises it to exactly the
  `[14:40-15:20]` and `[17:40-18:00]` windows.
  **The cost is the detector, not the fix.** A full sliding scan of every segment costs **+1.363 s**
  on a 30 s segment — 54% of this call's segments are 30 s ones, so it would push nearly every
  closing line past the P5 meter's 31.0 s ceiling. The shipped two-stage scan
  (`STT_CODESWITCH_SCAN=ends`) probes only the first and last window unless they disagree: **+0.321 s**
  mean on a 30 s segment (0.698 → 1.019 s), whole-call rtf 0.0247 → 0.0353. **CORRECTED 2026-09-05 by D27/#428:** this row first
  reported "43 of 118 max-length lines past the 31.0 s ceiling", computed as
  `duration + decode > ceiling`. **That is not a meter breach** — the meter compares an arrival
  GAP, and a uniform slowdown cancels between consecutive lines. Measured properly on a
  full-call paced replay, D26 raises the worst back-to-back gap by **+0.27 s** and costs **zero
  additional false overdues** (0/51 vs 0/51 on a matched 1200 s window; 1/112 over the full
  call, and that one is a temperature-fallback outlier that would have breached without D26).
  `METER_DECODE_ALLOWANCE_SECONDS` stays **1.0 s** (D27). *Known limits, both pinned by tests.* (i) A
  switch that **returns** before the segment ends (pl→en→pl) leaves both ends agreeing and is not
  detected. (ii) More importantly, **the minority language must hold at least
  `STT_CODESWITCH_MIN_WINDOWS` consecutive confident windows (~9 s)** — and that bound belongs to
  the run-length rule, not to the cheap first stage. Measured: a mono `--from-wav` replay of this
  same call re-segments it so the switch falls ~2 s before a boundary, and **neither `ends` nor
  `full` fires there**. So D26 repairs the two failures **as the live loop actually segmented them**,
  which is what happened on the call; it does not claim to catch every possible straddle. In that
  other placement the damage is also milder — one garbled clause rather than the interviewer's whole
  question, which is G8's "once a segment is wholly English the loop recovers" showing up again. A
  segment the detector misses is served exactly as it is today, no worse. Two failures is a small sample, and the
  window geometry (6 s / 3 s) and run-length rule were fixed **before** results were seen, not tuned
  after. **The #400 interaction is now measured (D28): at a 20 s cap D26 still fires and repairs both
  straddles — each switch keeps ≥10 s of both languages in one segment — so a shorter cap does not
  silently re-open G8.** *Rel:* D26 · D21 · D24 · D28 · #428.
  *Open:* (a) a **human-referenced** WER on spontaneous Polish+English (pack built, awaiting the
  listening pass); (b) can `initial_prompt` biased with expected jargon harden the seam? — now
  better aimed at the straddling-segment mechanism above. *Rel:* D24 (how any of this may be reported).
  *(c) ANSWERED 2026-08-31, **CLOSED 2026-09-01 → D21** (the measurement had been sitting here
  unclosed; the register row now owns it):* shrinking the window degrades accuracy
  monotonically (60 s → 3.8% WER, 10/10 terms · 15 s → 5.7%, 9/10 · 8 s → 7.6%, 8/10), with losses
  concentrated on the **technical terms** because fixed-time cuts bisect multi-word English phrases.
  **Naive fixed-clock chunking is disqualified for the ambient loop.** Segment on VAD-detected
  silence and/or overlap windows with de-duplication; latency permits it (max per-chunk decode
  0.36 s). *Rel:* D12 · G6.
  *Resolved 2026-08-31 (user):* suggestion language = **English by default**, because the context
  bundle (JD/resume/STAR) is English and the user is comfortable in both — English avoids a
  cross-lingual tax on every call. **Must be selectable at session start** (`SUGGESTION_LANGUAGE`,
  Day-3 dashboard control), not a rebuild. Note the asymmetry this locks in: transcript in **pl**,
  suggestions in **en** — the reasoning layer is cross-lingual by design, so its prompt must state
  that explicitly or it will drift into answering in the input language.
  *Rel:* D13 · D14 · D16. **← next.**
- **G7 — Interview-plan schema (D13).** The exact structured shape the plan tracker reads (ordered
  steps, per-step key points, "covered" signals). *Path:* define a small YAML/JSON schema Day 3.
  **Partly answered 2026-09-01:** the shape now exists and loads (`bundle.json` → `plan[]` →
  `reasoning.PlanStep`, with `done_signals` carried through unused). What is still open is the
  *tracking* half — what marks a step covered, and who owns that state. *Rel:* D13 · P3.
- **G9 — Speaker attribution (NEW 2026-09-01).** The transcript line carries no `them`/`you` tag, so
  every consumer treats the candidate's own speech as interviewer input. Concretely: D20's rule fires
  on the user's own questions, and #322 cannot render two-sided turns. The information is not lost —
  it is in the stereo WAV and in the per-channel `ChannelGate` state — it is just dropped at the
  transcript boundary. *Rel:* D19 · D20 · P4.

## Recently closed
- **`SEGMENT_MAX_SECONDS` 30→20 — the biggest lever on the 28 s wait (P5)** → **measured and NOT
  adopted, D28, 2026-09-06 (#400).** Isolated cleanly by reproducing the **stereo** segmentation with
  only the cap changed (`scripts/segment_cap_probe.py`) — the mono `--from-wav` recipe the row named
  would have confounded it (mono sits ~7.6 %/window from live, ≈ the effect; the stereo substrate sits
  ~2.3 %). 30→20 changes **5.7 % of tokens globally / 10.0 %/20 s-window** (noise floor 0/0 %), **sign
  unmeasured** (D24, #387 unfilled); moves force-cuts **52 %→65 %** (toward D21's disqualified regime);
  raises seam-duplication **2.7 %→4.5 %**. Its only gain — **−10.1 s** median onset→screen — is largely
  **already bought back by D25** (a provisional shows a median 16 s earlier), so the marginal value is
  small. Both clean costs survive 20 s: **D26** repairs both known straddles (2 passes, both languages
  kept), and the **D27** meter holds (ceiling 31→21 s, measured min allowance 0.385 s, 1.0 s → 0/31).
  Verdict: **keep 30 s**; revisit only after technical-round audio (#323) and a human WER pass (#387).
  Detail: `design/MEASUREMENT_segment_cap_400.md`. *Rel:* D28 · D21 · D24 · D25 · D26 · D27.
- **The interim-affordance question (P5)** → **A chosen + D25 minted, 2026-09-04 (measured, analysis-only probe).**
  What the D18 dashboard shows during the 5–30 s a segment is still open. Measured on the real
  42-min call from artefacts already on disk: median segment **30.0 s**, **54%** at the force-cut,
  median screen-still window **28 s**, question-spoken → line-on-screen median **28.4 s**
  (18/24 over 20 s), decode only **2.6%** GPU duty. Then 86 growing-prefix decodes of the call's own
  audio (35 GPU-s under a lease): provisional-token survival into the final line **0.86 @5 s /
  0.94 @10 s / 0.96 @15–20 s**, 1/86 in the forced-`pl` hallucination regime — so **D21's
  fixed-clock catastrophe does not transfer** to prefixes anchored at a true segment start, and at
  the 17:53 language switch the provisional was **right** where the final was **wrong**. A naive
  speaking indicator is dead on measurement (speech occupies **99.3%** of the call). *Rel:* D18 ·
  D19 · D21 · D22 · D24 · G8 · **D25**.
- **G6b — Salience, not just syntax** → **D23, closed 2026-09-04 (measured).** D20 answers "is this
  a question"; on the real 42-min HR screen that fired **24** times and most were audio checks and
  interviewer monologue. D23 adds a gate *after* D20 — a one-word YES/NO from the already-resident
  suggestion model, on the segment's interrogative sentences only — taking the call **24 → 13 with
  all 8 substantive turns still firing, at 0 MiB extra VRAM**. The embed-cosine route this was
  scoped around is measured and rejected (salient turns average cosine 0.510, non-salient 0.518);
  it survives behind `--salience-backend embed` so the rejection stays reproducible. **Unblocks
  #322** — the dashboard renders the filtered set.
- **G6 — Ambient trigger policy** → **D20, closed 2026-09-01 (measured).** A deterministic pl/en
  question rule fires the LLM; an LLM classifier per segment was rejected as the expensive option
  (Determinism First). Scored against 60 hand-labelled lines — precision 0.86 → **1.00** after
  guarding four Polish discourse idioms ("jak Pan widzi", "co ciekawe", "powiedzmy") and one real
  grammar fact (English auxiliaries only ask by inversion, so they must lead). **The 1.00 is
  in-sample** — the false positives that motivated the guards came from this same fixture set, so it
  is "no known false-positive class remains", not a live rate. Re-measure on a real call (#324).
  The re-scoping worry from 08-31 (Polish questions carried by intonation) did not bite: Whisper
  punctuates, and `czy` / the wh-words cover the rest. *Rel:* D12 · D14 · D20.
- **G8 (c) — fixed-clock vs VAD segmentation** → **D21, closed 2026-09-01.** The 08-31 A/B
  (1.8% vs 5.4% vs 24.1% WER) had settled it in practice but never earned a register row; it has one
  now. G8's other halves (spontaneous-speech code-switching, `initial_prompt` jargon bias) stay open.
- **G2 — Teams audio capture on Linux** → **PASS, closed 2026-08-31 (measured).** Monitor-source
  capture → faster-whisper works end-to-end on lab. Verified by loopback
  (`spike_capture.py --selftest-sink auto_null --seconds 8`): speech WAV → sink → `sink.monitor` →
  Whisper returned the **exact** transcript ("Tell me about a challenging machine learning project
  you have worked on."), **decode 0.37 s for an 8 s window (rtf 0.05)** with the model resident;
  ~2.5 s on a cold process incl. the 1.5 s model load. Mic channel verified separately as a distinct
  source (an external USB mic, peak 0.35 live) — two-channel attribution holds.
  **One defect found and fixed:** the D17 approach as originally coded (PortAudio `device="pulse"` +
  `PULSE_SOURCE`) **cannot work on lab** — the conda-forge PortAudio build exposes no `pulse` device
  (`ValueError: No input device matching 'pulse'`). Capture now shells out to **`parec`**, which
  speaks the native protocol and needs no extra install. D17's *approach* (monitor-source capture)
  is unchanged and vindicated; only its *transport* changed.
  **Residual (machine-state, not code):** lab currently has **no real output sink** — only PipeWire's
  `auto_null` dummy. A real `alsa_output.*` sink (and hence the monitor to point at) appears only
  once headphones/speakers are connected. Confirm the real sink name at dry-run. *Rel:* D16 · D17 · SI1.
- **G1 — Deadline** → resolved 2026-08-28: interview **Tue 2026-09-01 10:00**; dry-run Monday. (drives the 4-day plan)
- **G3 — Build vs adopt** → **D15** (focused Python build, not fork AnswerCue).
- **G4 — STT engine** → **D16** (local GPU faster-whisper).
- **G5 — UI form** → **D18** (local web dashboard over websocket, 127.0.0.1).
- **(runtime machine)** → **D17** (single-box on lab, Ubuntu, PipeWire capture).

## Candidate ideas — research-sourced, awaiting confirmation (2026-06-23)
Mined from the `governance-design-decision-management` research set (resources KB). The high-value,
low-noise, deterministic ones were implemented under **D7**; these need a scoping decision before
building. Confirm/approve one to promote it to a `G#` + `D#`.

- **C1 — Reverse-traceability / untracked trace-link recovery** (from *SoK: Software Artifacts
  Traceability*). Surface docs that **cite a `D#` but aren't in that decision's declared targets**
  (the inverse of `coverage`). High value, but **noisy as-is**: the generated GLANCE blocks in
  INDEX/project.md list every `D#`, so a naive scan false-positives. *Path:* exclude GLANCE-marker
  regions + non-target docs (session/working_set/ROUTINE_add/OPEN_DESIGN), then flag only substantive
  design docs. *Decision needed:* hard `check` gate vs advisory. *Rel:* extends D1/D7.
- **C2 — Architecture-erosion composite index** (from *Understanding Software Architecture Erosion*).
  One rolled-up "erosion" signal in `stats`/`tune` combining superseded-rate + edited-after-create
  rate + baseline growth, with a threshold flag. *Path:* add a derived column to `governance_metrics.csv`
  + a `tune` finding. *Decision needed:* the weighting/threshold. *Rel:* extends ROADMAP autotracking.
- **C3 — Dated, reasoned technical-debt register** (from *Technical Debt: A Systematic Mapping Study*).
  Promote `.coverage_baseline` from bare `D#:path` lines to TD items carrying **date + reason**, and
  add a **debt-age** signal to `tune` (old accepted gaps = accruing interest). *Path:* extend the
  baseline format + `cmd_tune`; keep back-compat parsing. *Rel:* extends D7 debt-visibility.
- **C4 — LLM-assisted design-rationale drafting** (from *Using LLMs in Generating Design Rationale*).
  When `check` flags an empty rationale (D7), optionally **draft** a rationale via FuelIX for the human
  to edit. *Tension:* `decision_tools.py` is deliberately **stdlib-only / no harness coupling** — this
  would break that. *Decision needed:* keep tooling pure and do this in the agent workflow instead, or
  add an opt-in seam. *Rel:* extends D7.

## Recently closed
_(move resolved gaps here with the closing `D#` and date)_
- _none yet._
