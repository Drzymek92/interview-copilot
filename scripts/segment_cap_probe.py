"""Measure what SEGMENT_MAX_SECONDS 30->20 buys and costs (#400).

A measurement instrument, not a default change. It answers the three questions the
cap change turns on, each in the currency that question is actually decided in, and
each as deterministically as the quantity allows (CLAUDE.md, Determinism First):

1. **`dist`** (CPU, exact) — reproduce the recorded call's segmentation at each cap and
   report the segment-length and on-screen-wait distributions, plus the P5 meter's
   false-overdue budget at that cap's ceiling. Segmentation (VAD + `ChannelGate`) has no
   sampling, so these numbers are reproducible run to run; only the DECODE text is not.
   The wait a viewer feels is `duration + decode(duration)` (D25's cost model), and the
   meter's arrival gap is `end(N)+decode - end(N-1)-decode` — both arithmetic over the
   reproduced boundaries, so no GPU is spent to learn the distribution.

2. **`codeswitch`** (GPU, small) — reproduce the STEREO segmentation at a cap and decode
   only the segments overlapping the two known code-switch straddles ([14:39-15:09],
   [17:53-18:23]) through the real `Transcriber`, reporting whether D26 still fires. A
   shorter cap cuts each straddling side shorter, and D26 needs the minority language to
   hold >= STT_CODESWITCH_MIN_WINDOWS windows over >= STT_CODESWITCH_MIN_SECONDS, so a cap
   change can silently stop the repair — an accuracy cost no token-divergence number sees.
   The firing decision itself is a forward pass (no sampling), so it is deterministic.

3. **`divergence`** (CPU, exact) — the token divergence between two live-format transcripts
   (the DENOMINATOR is `--b`, the baseline), reusing `diff_transcripts`' tiling. Feed it a
   20 s `--from-wav` replay against a 30 s `--from-wav` replay (both mono, everything else
   identical) to isolate the cap; feed it two 30 s replays for the run-to-run noise floor
   Whisper's temperature fallback imposes. Divergence, never WER (D24): both sides are
   Whisper.

Usage (GPU work goes through the lease board — CLAUDE.md):
    python scripts/segment_cap_probe.py dist --wav W --caps 30,20 --out scripts/outputs/cap_dist.json
    python scripts/segment_cap_probe.py codeswitch --wav W --cap 20 --regions 879-909,1073-1103
    python scripts/segment_cap_probe.py divergence --a cap20.txt --b cap30.txt
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings  # noqa: E402
from scripts.codeswitch_probe import collect_segments, segment_audio  # noqa: E402
from scripts.diff_transcripts import (  # noqa: E402
    agreement,
    bucket_tokens,
    global_agreement,
    parse_live,
)
from scripts.logger import get_logger  # noqa: E402
from scripts.meter_budget import budget_table, classify_gaps, minimum_allowance, pct  # noqa: E402
from scripts.replay_transcript import decode_seconds  # noqa: E402
from scripts.score_transcript import normalise  # noqa: E402

logger = get_logger("segment_cap_probe")

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "outputs"
RUN_STAMP = f"{datetime.now():%Y%m%d_%H%M%S}"

# The two segments that straddle a pl/en switch on the real HR call (#397/D26), in seconds.
STRADDLES = [(879.0, 909.0), (1073.0, 1103.0)]


# --------------------------------------------------------------------------- dist ---


def synth_trace(segments: list) -> list[dict]:
    """A meter latency-trace built from reproduced boundaries + D25's decode cost model.

    Deterministic stand-in for `live_transcribe --latency-trace`: every kept segment
    produces a line, and its close->write is the *modelled* decode `decode_seconds(dur)`.
    This deliberately holds decode latency to its mean model — the run-to-run VARIATION a
    real trace carries (Whisper temperature fallback, capture stalls) is content- and
    machine-driven, NOT cap-driven, so it is reasoned about separately (see the report),
    not manufactured here. `meter_budget`'s own functions score these rows unchanged.
    """
    rows: list[dict] = []
    for seg in segments:
        dur = seg.end - seg.start
        c2w = decode_seconds(dur)
        rows.append(
            {
                "index": seg.index,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "duration": round(dur, 3),
                "dropped": False,
                "written_at": round(seg.end + c2w, 4),
                "close_to_write": round(c2w, 4),
                "queue_wait": 0.0,
                "behind_partial": 0.0,
            }
        )
    return rows


def dist_at_cap(wav_path: Path, cap: float, allowances: list[float]) -> dict:
    """Segment-length, wait and meter-budget distributions for one cap (CPU only)."""
    settings.SEGMENT_MAX_SECONDS = cap  # VadSegmenter.push reads this live (line 311)
    segs = collect_segments(wav_path)
    durations = [s.end - s.start for s in segs]
    at_cap = [d for d in durations if d >= cap - 1e-3]
    waits = [d + decode_seconds(d) for d in durations]  # onset -> on screen (D25 cost model)

    rows = synth_trace(segs)
    gaps = classify_gaps(rows, cap)
    healthy = [g for g in gaps if g["healthy"]]
    hg = [g["gap"] for g in healthy]  # back-to-back arrival gaps = "screen-still window"
    need, worst = minimum_allowance(rows, cap)
    budget = budget_table(rows, cap, allowances)

    return {
        "cap": cap,
        "n_segments": len(segs),
        "seg_len": {
            "median": round(st.median(durations), 2) if durations else 0.0,
            "mean": round(st.mean(durations), 2) if durations else 0.0,
            "p95": round(pct(durations, 95), 2) if durations else 0.0,
            "max": round(max(durations), 2) if durations else 0.0,
            "at_cap_n": len(at_cap),
            "at_cap_pct": round(100 * len(at_cap) / len(durations), 1) if durations else 0.0,
        },
        "wait": {  # question onset -> line on screen
            "median": round(st.median(waits), 2) if waits else 0.0,
            "mean": round(st.mean(waits), 2) if waits else 0.0,
            "p95": round(pct(waits, 95), 2) if waits else 0.0,
            "max": round(max(waits), 2) if waits else 0.0,
        },
        "screen_still": {  # back-to-back arrival gap the meter feels
            "median": round(st.median(hg), 2) if hg else 0.0,
            "p95": round(pct(hg, 95), 2) if hg else 0.0,
            "max": round(max(hg), 2) if hg else 0.0,
            "n_back_to_back": len(healthy),
            "n_across_silence": len(gaps) - len(healthy),
        },
        "meter": {
            "ceiling_at_1s": round(cap + 1.0, 2),
            "minimum_allowance_zero_false_overdue": round(need, 3),
            "worst_gap_index": worst["index"] if worst else None,
            "budget": [
                {"allowance": a, "ceiling": round(c, 2), "false_overdue": f, "back_to_back": n}
                for a, c, f, n in budget
            ],
        },
    }


def run_dist(args: argparse.Namespace) -> int:
    caps = [float(x) for x in args.caps.split(",") if x.strip()]
    allowances = [float(x) for x in args.allowances.split(",") if x.strip()]
    wav_path = Path(args.wav)
    reports = [dist_at_cap(wav_path, cap, allowances) for cap in caps]

    for r in reports:
        sl, w, ss, m = r["seg_len"], r["wait"], r["screen_still"], r["meter"]
        print(f"\n{'=' * 74}\ncap {r['cap']:g}s  ({r['n_segments']} segments)\n{'=' * 74}")
        print(f"  segment length : median {sl['median']:.1f}s  mean {sl['mean']:.1f}s  "
              f"p95 {sl['p95']:.1f}s  max {sl['max']:.1f}s   at cap: {sl['at_cap_n']}/"
              f"{r['n_segments']} ({sl['at_cap_pct']:.0f}%)")
        print(f"  onset->screen  : median {w['median']:.1f}s  mean {w['mean']:.1f}s  "
              f"p95 {w['p95']:.1f}s  max {w['max']:.1f}s")
        print(f"  screen-still   : median {ss['median']:.1f}s  p95 {ss['p95']:.1f}s  "
              f"max {ss['max']:.1f}s   ({ss['n_back_to_back']} back-to-back, "
              f"{ss['n_across_silence']} across a silence)")
        print(f"  meter ceiling @1.0s allowance: {m['ceiling_at_1s']:.1f}s   "
              f"smallest allowance with 0 false overdues: "
              f"{m['minimum_allowance_zero_false_overdue']:.3f}s")
        print(f"    {'allowance':>9} {'ceiling':>8} {'false overdue':>15}")
        for b in m["budget"]:
            mark = "  <- shipped" if abs(b["allowance"] - 1.0) < 1e-9 else ""
            print(f"    {b['allowance']:8.2f}s {b['ceiling']:7.2f}s "
                  f"{b['false_overdue']:9d}/{b['back_to_back']:<3d}{mark}")

    if len(reports) == 2:
        a, b = reports
        print(f"\n{'-' * 74}\ncap {a['cap']:g} -> {b['cap']:g}:  "
              f"onset->screen median {a['wait']['median']:.1f}s -> {b['wait']['median']:.1f}s "
              f"(saves {a['wait']['median'] - b['wait']['median']:+.1f}s median)  |  "
              f"at-cap {a['seg_len']['at_cap_pct']:.0f}% -> {b['seg_len']['at_cap_pct']:.0f}%")

    out = Path(args.out) if args.out else OUTPUT_DIR / f"cap_dist_{RUN_STAMP}.json"
    out.write_text(json.dumps({"wav": wav_path.name, "generated": RUN_STAMP, "caps": reports},
                              indent=2), encoding="utf-8")
    print(f"\nreport: {out}")
    return 0


# --------------------------------------------------------------------- codeswitch ---


def run_codeswitch(args: argparse.Namespace) -> int:
    """Reproduce STEREO segmentation at --cap and test D26 firing on the straddle regions."""
    from scripts.stt import Transcriber

    regions = [tuple(float(x) for x in r.split("-")) for r in args.regions.split(",") if r.strip()]
    settings.SEGMENT_MAX_SECONDS = args.cap
    wav_path = Path(args.wav)
    segs = collect_segments(wav_path)

    # Every reproduced segment that overlaps any target region.
    hits = [s for s in segs if any(s.end > lo and s.start < hi for lo, hi in regions)]
    logger.info("cap=%.0fs: %d segments, %d overlap the target regions", args.cap, len(segs), len(hits))

    transcriber = Transcriber()  # STT_CODESWITCH_MODE defaults to "split" (D26)
    out_segments: list[dict] = []
    for seg in hits:
        audio = segment_audio(seg)
        result = transcriber.transcribe_array(audio, sample_rate=settings.SAMPLE_RATE)
        dur = round(len(audio) / settings.SAMPLE_RATE, 2)
        entry = {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "duration": dur,
            "speaker": seg.speaker,
            "code_switch": bool(result.code_switch),
            "languages": list(result.languages),
            "language": result.language,
            "decode_passes": int(result.decode_passes),
            "text": result.text.strip(),
        }
        out_segments.append(entry)
        print(f"\n[{seg.start:.0f}-{seg.end:.0f}] {seg.speaker} dur={dur:.0f}s  "
              f"code_switch={entry['code_switch']}  langs={entry['languages']}  "
              f"passes={entry['decode_passes']}")
        print(f"  {entry['text'][:400]}")

    fired = [e for e in out_segments if e["code_switch"]]
    print(f"\ncap {args.cap:g}s: {len(hits)} straddle-region segments, "
          f"D26 fired on {len(fired)} "
          f"(min windows {settings.STT_CODESWITCH_MIN_WINDOWS}, "
          f"min seconds {settings.STT_CODESWITCH_MIN_SECONDS:g})")

    out = Path(args.out) if args.out else OUTPUT_DIR / f"cap_codeswitch_{int(args.cap)}_{RUN_STAMP}.json"
    out.write_text(json.dumps({
        "wav": wav_path.name, "cap": args.cap, "generated": RUN_STAMP,
        "regions": [list(r) for r in regions],
        "min_windows": settings.STT_CODESWITCH_MIN_WINDOWS,
        "min_seconds": settings.STT_CODESWITCH_MIN_SECONDS,
        "n_fired": len(fired), "segments": out_segments,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {out}")
    return 0


# --------------------------------------------------------------------- divergence ---


def divergence_between(a: list, b: list, window: float) -> dict:
    """Token divergence of new arm `a` against baseline `b` (b is the denominator).

    Pure — arithmetic over `diff_transcripts`' single-counted tiling, no I/O. `a`/`b` are lists of
    anything with `.start`, `.end`, `.text` (a `diff_transcripts.LiveSegment`).
    """
    duration = max(max((s.end for s in a), default=0.0), max((s.end for s in b), default=0.0))
    n_windows = int(duration // window) + 1
    a_buckets = bucket_tokens([(s.start, s.end, s.text) for s in a], window, n_windows)
    b_buckets = bucket_tokens([(s.start, s.end, s.text) for s in b], window, n_windows)

    tot_base = tot_new = tot_matched = 0
    for k in range(n_windows):
        nt, bt = a_buckets[k], b_buckets[k]
        tot_matched += agreement(bt, nt)  # baseline tokens matched in the new arm
        tot_base += len(bt)
        tot_new += len(nt)
    windowed_div = 1 - tot_matched / tot_base if tot_base else 0.0

    g = global_agreement(normalise(" ".join(s.text for s in a)),
                         normalise(" ".join(s.text for s in b)))  # a=live-slot, b=offline-slot
    return {
        "a_segments": len(a), "b_segments": len(b),
        "a_tokens": g["live_tokens"], "b_tokens": g["offline_tokens"],
        "global_divergence": 1 - g["agreement_vs_offline"],
        "windowed_divergence": windowed_div,
        "windowed_matched": tot_matched, "windowed_baseline_words": tot_base,
    }


def run_divergence(args: argparse.Namespace) -> int:
    """Token divergence between two live transcripts (--b is the baseline denominator)."""
    a, b = parse_live(Path(args.a)), parse_live(Path(args.b))
    r = divergence_between(a, b, args.window)
    print(f"a (new)      : {Path(args.a).name}   {r['a_segments']} segments, {r['a_tokens']} tokens")
    print(f"b (baseline) : {Path(args.b).name}   {r['b_segments']} segments, {r['b_tokens']} tokens")
    print(f"global agreement (baseline tokens matched): {1 - r['global_divergence']:.1%}  "
          f"-> global divergence {r['global_divergence']:.1%}")
    print(f"windowed divergence over {args.window:.0f}s windows: {r['windowed_divergence']:.1%}  "
          f"({r['windowed_matched']} matched / {r['windowed_baseline_words']} baseline words)")
    return 0


def run_transcribe(args: argparse.Namespace) -> int:
    """Decode EVERY reproduced STEREO segment at --cap into a live-format transcript.

    The mono `--from-wav` replay confounds the cap with a downmix and a re-segmentation
    (D26 note in OPEN_DESIGN). Reproducing the segmentation from the stereo WAV with only
    SEGMENT_MAX_SECONDS changed isolates the cap on the SAME per-channel gate the live loop
    ran, so a diff of two of these outputs is the cap and nothing else. Boundaries are CPU
    and deterministic; only the decode text carries Whisper's sampling, so run >=2 per cap.
    """
    from scripts.stt import Transcriber

    settings.SEGMENT_MAX_SECONDS = args.cap
    wav_path = Path(args.wav)
    segs = collect_segments(wav_path)
    transcriber = Transcriber()  # STT_CODESWITCH_MODE defaults to "split" (D26)
    out_lines = [f"# stereo cap replay | cap={args.cap:g}s | model={settings.STT_MODEL}\n",
                 f"# source: {wav_path.name} (stereo segmentation reproduced; only the cap changed)\n"]
    for seg in segs:
        audio = segment_audio(seg)
        result = transcriber.transcribe_array(audio, sample_rate=settings.SAMPLE_RATE)
        text = result.text.strip()
        if text:
            out_lines.append(
                f"[{int(seg.start) // 60:02d}:{int(seg.start) % 60:02d}-"
                f"{int(seg.end) // 60:02d}:{int(seg.end) % 60:02d}] "
                f"{seg.speaker} ({result.language}): {text}\n"
            )
    out = Path(args.out) if args.out else OUTPUT_DIR / f"cap_stereo_{int(args.cap)}_{RUN_STAMP}.txt"
    out.write_text("".join(out_lines), encoding="utf-8")
    print(f"cap {args.cap:g}s: {len(segs)} stereo segments -> {len(out_lines) - 2} lines  {out}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", help="decode all stereo segments at a cap -> .txt (GPU)")
    t.add_argument("--wav", required=True)
    t.add_argument("--cap", type=float, required=True)
    t.add_argument("--out", default="")
    t.set_defaults(func=run_transcribe)

    d = sub.add_parser("dist", help="segment/wait/meter distributions per cap (CPU)")
    d.add_argument("--wav", required=True)
    d.add_argument("--caps", default="30,20", help="comma-separated caps in seconds")
    d.add_argument("--allowances", default="0.25,0.5,0.75,1.0,1.5,2.0")
    d.add_argument("--out", default="")
    d.set_defaults(func=run_dist)

    c = sub.add_parser("codeswitch", help="D26 firing on the straddle regions at a cap (GPU)")
    c.add_argument("--wav", required=True)
    c.add_argument("--cap", type=float, required=True)
    c.add_argument("--regions", default="879-909,1073-1103", help="seconds lo-hi, comma-separated")
    c.add_argument("--out", default="")
    c.set_defaults(func=run_codeswitch)

    v = sub.add_parser("divergence", help="token divergence between two live transcripts (CPU)")
    v.add_argument("--a", required=True, help="the new arm")
    v.add_argument("--b", required=True, help="the baseline (denominator)")
    v.add_argument("--window", type=float, default=20.0)
    v.set_defaults(func=run_divergence)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
