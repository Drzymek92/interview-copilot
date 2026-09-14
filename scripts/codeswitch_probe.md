# `scripts/codeswitch_probe.py` + `scripts/codeswitch_compare.py`

The measurement instruments behind **D26** and the closure of **OPEN_DESIGN G8 (d)** (#397). They
answer "what did the live loop actually decode, and what would a change do to it" without a second
recording and without a live call. Both are deterministic; only `--probe` and `--replay` touch the GPU.

## Why they exist
The two failures G8 (d) describes are *specific segments of one recording*. Anything that reads
those segments off the transcript's `mm:ss` stamps is guessing at their boundaries — a force-cut
segment carries pre-roll and carry-over audio, so its true start is earlier than the stamp. The
probe instead **replays the recorded stereo WAV through the live loop's own gates and segmenter**,
which reproduces the boundaries exactly, and `--verify` proves that before any GPU time is spent.

## `codeswitch_probe.py`

| Mode | GPU | What it does |
|---|---|---|
| `--verify` | no | Replays segmentation from the WAV and matches it to a transcript. Prints how many transcript lines are backed by an exactly-reproduced segment. **Run this first.** |
| `--probe --segments N,N\|all` | yes | Per segment: the whole-buffer `detect_language` vote, a sliding sub-window language track, and a forced decode in each candidate language with its duration-weighted `avg_logprob`. Writes JSON. |
| `--replay MODE` | yes | Re-decodes every reproduced segment through the real `Transcriber` seam under `STT_CODESWITCH_MODE=MODE`, at the **live boundaries**. Writes a transcript in the `.txt` line format plus a per-segment JSON. |

Boundaries are held fixed in `--replay`, which is the point: a difference between two runs is
attributable to the mode and not to segmentation. That is the comparison a `live_transcribe.py
--from-wav` replay cannot make, because there the segmenter moves too.

### Key detail: matching is by stamp, not by position
The live worker writes **no line** for a segment that decodes to empty text, so the replay
legitimately yields more segments than the transcript has lines (113 vs 110 on the real HR call).
`align()` matches on the printed `mm:ss` stamps, which `mmss` truncates.

## `codeswitch_compare.py`
Arithmetic over the `--replay` summaries — no model call. Three blocks:

1. **Noise floor** — the same mode run twice. A single A/B pair cannot separate "the feature changed
   nothing" from "the model samples", so ≥2 runs per mode is the minimum and the within-mode
   difference is printed beside every claim.
2. **Effect** — every run of every other mode against every baseline run, splitting the changed
   segments into those the detector *flagged* and those it did not. A change outside the flagged set
   is a regression, and this is where it would show.
3. **Cost** — decode seconds, rtf, and the per-flagged-segment latency against that run's own
   unflagged rate.

Everything it reports is **divergence, never WER** (D24): both sides are Whisper output.

## Runbook (the D26 measurement, reproducible)
```bash
# 1. prove the slices  (CPU, ~10 s)
python scripts/codeswitch_probe.py --verify \
  --wav scripts/outputs/live_audio_20260902_100033.wav \
  --transcript scripts/outputs/live_transcript_20260902_100033.txt

# 2. the mechanism, all 110 segments  (GPU ~4.5 min — go through the lease board)
python -m commons.coordination.gpu run --vram 3000 --label "codeswitch sweep" -- \
  python scripts/codeswitch_probe.py --probe --segments all --window 6 --hop 3 \
  --wav ... --transcript ... --out scripts/outputs/codeswitch_sweep_A.json

# 3. the A/B, 2 runs per mode  (GPU ~10 min)
for r in 1 2; do for m in off rescore split; do
  python scripts/codeswitch_probe.py --replay $m --wav ... --transcript ... \
    --out scripts/outputs/ce_${m}_r${r}; done; done

# 4. read it  (CPU, instant)
python scripts/codeswitch_compare.py --runs scripts/outputs/ce_*.json --baseline off --targets 37,46
```

## Gotchas
- **Ordinals are transcript-line ordinals, not file line numbers** — the `.txt` header offsets them
  (by 9 on the HR-call transcript). `--verify` numbers them the same way `--probe` does.
- **A busy GPU contaminates every latency number here.** A concurrent Ollama model inflated one
  measured run by 28% (227.7 s vs 178.6 s for the same work). Check `ollama ps` / `gpu status`
  before believing a cost figure, and re-measure on a clean card.
- The WAV must be the recorder's own 16-bit 16 kHz output; `--verify` refuses anything else rather
  than silently reproducing the wrong boundaries.
