# `segment_cap_probe.py` — the #400 `SEGMENT_MAX_SECONDS` instrument (D28)

Measures what cutting the force-cut cap (30 → 20 s) buys and costs, each quantity in the currency
that decides it, and as deterministically as the quantity allows (Determinism First). It is the
instrument behind **D28** and the verdict in `design/MEASUREMENT_segment_cap_400.md`.

## The one trap it exists to avoid
The ledger row's original recipe — replay `--from-wav` at 20 s and diff against the 30 s live
transcript — is **confounded**: `--from-wav` downmixes to mono and re-segments, so a mono replay
sits **~7.6 %/window** from the stereo live transcript, *as much as the cap effect itself*. Measured
that way you cannot tell the cap's effect from the harness's. So the divergence arm here reproduces
the **stereo** segmentation (the same per-channel `ChannelGate` the live loop ran) with **only the
cap changed**, which sits ~2.3 % from live — far below the effect — and isolates the cap cleanly.

## Three modes

| mode | GPU? | answers |
|---|---|---|
| `dist` | no | segment-length + on-screen-wait distributions and the P5 meter's false-overdue budget at each cap. Segmentation is VAD/`ChannelGate` — no sampling — so these are exact and reproducible. |
| `codeswitch` | yes (small) | whether D26 still fires on the two known straddles at a cap (reproduces STEREO segments, decodes only the region-overlapping ones). The firing decision is a forward pass, so it is deterministic. |
| `transcribe` | yes | decode EVERY stereo segment at a cap into a live-format `.txt` — the substrate for a clean cap-only divergence. |
| `divergence` | no | token divergence between two live transcripts (`--b` is the baseline denominator), reusing `diff_transcripts`' single-counted tiling. Divergence, never WER (D24). |

## Reproducible #400 runbook
All GPU work goes through the lease board; evict a resident Ollama model first (`gpu evict-ollama`)
and lease ~3000 MiB for Whisper.

```bash
WAV=scripts/outputs/live_audio_20260902_100033.wav
# 1. distributions + meter budget, both caps (CPU, exact):
python scripts/segment_cap_probe.py dist --wav $WAV --caps 30,20
# 2. does D26 still repair the two straddles at 20 s? (and the 30 s control)
python scripts/segment_cap_probe.py codeswitch --wav $WAV --cap 30
python scripts/segment_cap_probe.py codeswitch --wav $WAV --cap 20
# 3. clean cap-only divergence: decode all stereo segments at each cap, >=2 runs (sampling), then diff
for c in 30 20; do for r in a b; do
  python scripts/segment_cap_probe.py transcribe --wav $WAV --cap $c --out scripts/outputs/cap_stereo_${c}_${r}.txt; done; done
python scripts/segment_cap_probe.py divergence --a scripts/outputs/cap_stereo_20_a.txt --b scripts/outputs/cap_stereo_30_a.txt  # the cap effect
python scripts/segment_cap_probe.py divergence --a scripts/outputs/cap_stereo_30_b.txt --b scripts/outputs/cap_stereo_30_a.txt  # noise floor (expect ~0)
python scripts/segment_cap_probe.py divergence --a scripts/outputs/cap_stereo_30_a.txt --b scripts/outputs/live_transcript_20260902_100033.txt  # harness vs live (must be << the effect)
```

The meter arm's honest limit: `dist` holds decode latency to D25's mean cost model, so it shows the
*shape* the cap gives the arrival gap, not the run-to-run latency VARIATION. That variation is
content- and machine-driven, **not** cap-driven, so validate it with one real paced trace on a
bounded window and read it with `meter_budget.py`:

```bash
SEGMENT_MAX_SECONDS=20 python scripts/live_transcribe.py --from-wav $WAV --no-record \
  --pace 1.0 --seconds 600 --latency-trace scripts/outputs/meter_cap20.jsonl
python scripts/meter_budget.py --trace scripts/outputs/meter_cap20.jsonl --labels cap20 --cap 20
```

## What it found (2026-09-06, D28)
Cap effect on the true stereo path **5.7 % global / 10.0 %/window** (noise floor 0/0 %), sign
unmeasured (D24); force-cuts 52 %→65 %; seam-duplication 2.7 %→4.5 %; median onset→screen −10.1 s.
**D26 fires on both straddles at 20 s; the D27 meter holds (1.0 s → 0/31 at a 21.0 s ceiling).**
Verdict: keep 30 s — see `design/MEASUREMENT_segment_cap_400.md`.
