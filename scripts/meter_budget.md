# `scripts/meter_budget.py` — sizing the P5 meter's decode allowance (D27 / #428)

Deterministic arithmetic over `live_transcribe.py --latency-trace` rows. No model call.

## The one idea the whole instrument exists to enforce
The meter promises "next line due within ≤ `SEGMENT_MAX_SECONDS + METER_DECODE_ALLOWANCE_SECONDS`",
and it measures a **gap between arrivals**. Line N−1 lands at `end(N−1) + latency(N−1)` and line N at
`end(N) + latency(N)`, so

```
gap = (end(N) − end(N−1))  +  (latency(N) − latency(N−1))
                spacing                latency STEP
```

A **uniform** slowdown cancels in the step. This is not a modelling convenience — it was checked
against the real call's own line-arrival wall clocks (recovered from `logs/live_transcribe_20260902.log`,
109 gaps): residual mean **+0.010 s**, median **−0.001 s**, 87/105 inside the transcript's ±1 s stamp
resolution.

**The trap this replaces.** `duration + decode > ceiling` looks like a breach test and is not one: it
assumes the previous line arrived instantly. Using it over-stated D26's meter cost by an order of
magnitude (43/118 "breaches" that were really 1/112). Don't reintroduce it.

## What counts as a false overdue
Only a **back-to-back** gap (`spacing ≤ SEGMENT_MAX_SECONDS`) can be one. A wider spacing means
silence between segments, or a segment that decoded to nothing — a wait the cap never promised to
bound (the real call's 43.2 s silence), which the meter is right to report at any allowance. The
allowance is not the lever for those; `SEGMENT_MAX_SECONDS` (#400) is.

## Reading the output
- `close->write` by segment-length band — the raw cost, useful for attribution, **not** a breach test.
- `latency STEP` — the part that actually reaches the meter.
- `smallest allowance with ZERO false overdues` plus the gap that sets it.
- the budget table: false overdues vs how much later a genuine stall is reported.

## Runbook
```bash
# 1. capture a trace — MUST be paced at 1.0 (an unpaced replay outruns the GPU and the gaps are fiction)
python -m commons.coordination.gpu run --vram 3000 --label "meter trace" -- \
  python scripts/live_transcribe.py --from-wav scripts/outputs/live_audio_<stamp>.wav \
  --no-record --pace 1.0 --seconds 2530 --latency-trace scripts/outputs/meter_A_ship.jsonl

# force D25 collisions for the stress arm
PARTIAL_DECODE_SECONDS=1.0 ... --latency-trace scripts/outputs/meter_B_forced.jsonl
# and an arm without D26 for attribution
STT_CODESWITCH_MODE=off ... --latency-trace scripts/outputs/meter_C_nod26.jsonl

# 2. read it
python scripts/meter_budget.py --trace scripts/outputs/meter_*.jsonl \
  --labels ship forced no-d26
```

## Gotchas
- **Pace 1.0 or the numbers are meaningless.** Wall-clock gaps are the measurement.
- **Nothing else may touch the GPU during a trace.** The project's own test suite makes live model
  calls; running it against a trace run inflates every latency (measured: a resident Ollama model
  inflated one pass by 28%).
- A `--from-wav` replay is **mono**, so it re-segments the call and may contain no D26-flagged
  segment at all — run A did not. Take the flagged-segment cost from `codeswitch_probe.py --replay`
  instead of assuming a trace contains one.
- Traces are wall-clock; two runs are never byte-identical. Compare distributions, not rows.
