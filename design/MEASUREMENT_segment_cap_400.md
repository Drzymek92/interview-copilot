# Is `SEGMENT_MAX_SECONDS` 30 → 20 safe? — measured verdict (#400)

_Measured 2026-09-06 on lab, all local (SI1). Instrument: `scripts/segment_cap_probe.py`.
Source: `scripts/outputs/live_audio_20260902_100033.wav` (the real 42-min HR call; not deleted, D11)._

## Verdict — **keep 30 s. 20 s is not demonstrably safe; do not flip the default now.**
The cap change is real and large (≈1 token in 10 per 20 s window changes), its **sign is unmeasured**
(no human reference exists — D24/#387), it pushes 65 % of segments to a mid-speech force-cut, and its
one benefit — latency — has **already been largely bought back by D25 provisionals**. The decision that
matters is the *technical* rounds (O2), which we have not recorded yet. The default flip is yours;
the knob (`SEGMENT_MAX_SECONDS`, CFG) is there to lower per session if you judge latency dominates.

## The trade, in both currencies

| | **30 s (today)** | **20 s** | change |
|---|---|---|---|
| median segment length | 30.0 s | 20.0 s | — |
| segments at the force-cut | **52 %** | **65 %** | +13 pt more mid-speech cuts |
| median onset → line on screen | 30.7 s | 20.5 s | **−10.1 s** (latency win) |
| median screen-still window | 28.5 s | 18.5 s | −10.0 s |
| seam-duplication (carry-over repeats) | 2.7 % | **4.5 %** | +1.8 pt churn |

**Latency won:** ~10 s off the *final* line. But **marginal now** — D25 provisionals already show a
truthful prefix a **median 16 s earlier**, so the cap only trims the tail of an already-shortened
blank window.

**Accuracy paid:** on the true stereo path the cap changes **5.7 % of tokens globally / 10.0 % per
20 s window** (vs a **0.0 %** run-to-run noise floor). Part of that is measurable churn — seam
duplication rises 2.7 → 4.5 % because 65 % of segments now force-cut and each cut re-feeds
`SEGMENT_CARRYOVER_SECONDS` of audio — and the 20 s transcript runs +4.2 % longer, mostly repeats.
The rest is genuine content change whose **direction (better or worse) is unmeasured**: both sides are
Whisper, neither is ground truth (D24). 65 % force-cuts is a step toward the mid-phrase-cut regime D21
disqualified, and it bites hardest on long, jargon-dense *technical* answers — the un-recorded target.

## The two costs that are clean (measured, not assumed)

- **D26 code-switch repair survives.** Both known straddles are still repaired at 20 s: the pl→en
  switch (`Okay, so could we switch into the English…`) and the en→pl switch (`…enough for me / Super,
  a jeśli chodzi…`) each still land ≥10 s of both languages in one segment, so D26 fires (2 passes,
  both languages kept) — same as at 30 s. A shorter cap does **not** silently re-open G8. _(Minor
  side effect: at 20 s the first straddle re-segments with a `you` speaker tag instead of `them`.)_
- **The P5 meter (D27) survives.** The ceiling moves 31.0 → 21.0 s. On a real paced 20 s replay the
  smallest allowance for zero false overdues is **0.385 s** (set by a real 0.730 s decode); at the
  shipped **1.0 s** allowance it is **0/31** false overdues. The latency STEP the meter feels is tiny
  (max +0.386 s) and the cap makes segment spacing tighter and more uniform, so 1.0 s holds with room
  and is not pushed toward either of D27's [0.75 s, 2.0 s] bounds. **D27's verdict stands at 20 s.**

## Why the ledger row's own recipe could not have answered this (the honest crux)
The row said: replay `--from-wav` at 20 s vs the 30 s live transcript and diff. That is **confounded** —
`--from-wav` downmixes to mono and re-segments, so a mono replay departs from the real stereo live
transcript by **7.6 % per window**, *as much as the cap effect itself*. Measured that way the answer is
un-attributable. The fix used here: reproduce the **stereo** segmentation with only the cap changed
(`segment_cap_probe transcribe`, same per-channel gate the live loop ran). That substrate sits **2.3 %**
from the live transcript — far below the 10.0 % cap effect — so the cap is cleanly isolated. The offline
path *can* answer this; the originally-filed mono recipe could not.

## What would let 20 s be reconsidered
1. **Technical-round audio** (#323 / the real technical call) — the cap's cost is concentrated exactly
   where we have no recording: long, jargon-dense answers.
2. **A human WER pass** (#387, deferred) — the only thing that can say whether the 10 %/window of
   changed tokens are worse, or merely different. Until then the sign is unknown (D24).

_Numbers: `scripts/outputs/cap_dist_*.json`, `cap_codeswitch_{20,30}.json`, `meter_cap20.jsonl`._
