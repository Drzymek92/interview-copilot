"""Localise where the LIVE transcript and an OFFLINE re-decode disagree (G8, #324).

Deterministic — arithmetic only, no model call (CLAUDE.md, Determinism First).

**This does not measure WER (D24).** Both inputs are Whisper output, so every number
here is a *divergence* between two machine decodes: it says WHERE the live loop
and an unconstrained offline decode part company, not which one is right. It
exists to (a) size the cost of the live latency budget and (b) pick the segments
worth spending a human ear on, which is the only thing that yields a true WER.

Three things it reports:
  1. per-live-segment divergence (edit distance vs the offline text covering the
     same wall-clock window), ranked — the localisation;
  2. global token agreement over the whole call (difflib matching blocks);
  3. the live transcript's own **duplication rate** — how many tokens are a repeat
     of the previous segment's tail, an artefact of SEGMENT_CARRYOVER_SECONDS that
     inflates any score computed over the plain file.

`--sample N` builds a human-reference pack: N segments (worst divergences +
random controls + the English stretch), one WAV clip each under
`scripts/outputs/`, and a fill-in reference template under `scripts/inputs/`
(the template is the durable artefact; the clips die with the recording, D11).

Usage:
    python scripts/diff_transcripts.py \
        --live scripts/outputs/live_transcript_20260902_100033.txt \
        --offline scripts/outputs/offline_decode_20260904_212429.json \
        --wav scripts/outputs/live_audio_20260902_100033.wav --sample 12
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.logger import get_logger  # noqa: E402
from scripts.score_transcript import normalise  # noqa: E402

logger = get_logger("diff_transcripts")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "outputs"
INPUT_DIR = PROJECT_ROOT / "scripts" / "inputs"

LIVE_LINE = re.compile(r"^\[(\d+):(\d+)-(\d+):(\d+)\]\s+(\w+)\s*\(([a-z]{2})\):\s*(.*)$")


@dataclass
class LiveSegment:
    index: int
    start: float
    end: float
    speaker: str
    language: str
    text: str


def parse_live(path: Path) -> list[LiveSegment]:
    """Read the tagged live transcript ([mm:ss-mm:ss] speaker (lang): text)."""
    out: list[LiveSegment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LIVE_LINE.match(line.strip())
        if not m:
            continue
        m1, s1, m2, s2, speaker, lang, text = m.groups()
        out.append(
            LiveSegment(
                index=len(out),
                start=int(m1) * 60 + int(s1),
                end=int(m2) * 60 + int(s2),
                speaker=speaker,
                language=lang,
                text=text.strip(),
            )
        )
    return out


def bucket_tokens(
    spans: list[tuple[float, float, str]], window: float, n_windows: int
) -> list[list[str]]:
    """Assign every token to exactly ONE time window.

    A segment's tokens are spread uniformly across its own [start, end] and bucketed
    by that interpolated time. Single-counting is the point: an earlier version
    gathered, per live segment, every offline segment *overlapping* it, which counted
    long offline segments once per live neighbour and inflated the denominator by ~50%.
    Uniform spreading is approximate at the token level but unbiased in aggregate, and
    it lets a 30 s segment contribute to the several windows it actually spans.
    """
    buckets: list[list[str]] = [[] for _ in range(n_windows)]
    for start, end, text in spans:
        toks = normalise(text)
        if not toks:
            continue
        span = max(end - start, 1e-3)
        for i, tok in enumerate(toks):
            t = start + (i + 0.5) * span / len(toks)
            k = min(int(t // window), n_windows - 1)
            buckets[max(0, k)].append(tok)
    return buckets


def agreement(a: list[str], b: list[str]) -> int:
    """Tokens of `a` matched in `b` by difflib's alignment (order-preserving)."""
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return sum(block.size for block in sm.get_matching_blocks())


def global_agreement(live_tokens: list[str], offline_tokens: list[str]) -> dict[str, float]:
    """Token agreement over the whole call, ignoring timing entirely."""
    matched = agreement(offline_tokens, live_tokens)
    return {
        "live_tokens": len(live_tokens),
        "offline_tokens": len(offline_tokens),
        "matched_tokens": matched,
        "agreement_vs_offline": matched / len(offline_tokens) if offline_tokens else 0.0,
        "agreement_vs_live": matched / len(live_tokens) if live_tokens else 0.0,
    }


def offline_window_text(segments: list[dict], start: float, end: float, pad: float = 0.0) -> str:
    """Offline text whose segments overlap [start, end] — for DISPLAY in the sample pack."""
    hits = [s for s in segments if s["end"] > start - pad and s["start"] < end + pad]
    hits.sort(key=lambda s: s["start"])
    return " ".join(s["text"] for s in hits)


def duplication_rate(live: list[LiveSegment]) -> tuple[int, int, list[tuple[int, int]]]:
    """Tokens in each segment that repeat text already present in the previous one.

    `SEGMENT_CARRYOVER_SECONDS` re-feeds 1.5 s of audio into the next segment after a
    max-length force-cut, and the two decodes of that audio rarely agree token-for-token,
    so an exact prefix/suffix test misses it. This counts the longest matching BLOCK
    (difflib) between consecutive segments instead, which catches the reworded repeat.
    """
    per: list[tuple[int, int]] = []
    dup = total = 0
    prev: list[str] = []
    for seg in live:
        toks = normalise(seg.text)
        total += len(toks)
        best = 0
        if prev and toks:
            sm = difflib.SequenceMatcher(a=prev, b=toks, autojunk=False)
            block = sm.find_longest_match(0, len(prev), 0, len(toks))
            # Only a repeat at the SEAM counts: the block must sit near prev's tail and
            # toks' head, or it is just a common phrase recurring mid-conversation.
            if block.size >= 3 and block.a + block.size >= len(prev) - 3 and block.b <= 3:
                best = block.size
        dup += best
        per.append((seg.index, best))
        prev = toks
    return dup, total, per


def write_clip(wav_path: Path, out_path: Path, start: float, end: float, pad: float = 0.5) -> None:
    """Cut [start-pad, end+pad] out of the recording into its own WAV (channels kept)."""
    with wave.open(str(wav_path), "rb") as wf:
        rate, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        first = max(0, int((start - pad) * rate))
        n = max(0, int((end + pad) * rate) - first)
        wf.setpos(min(first, wf.getnframes()))
        frames = wf.readframes(min(n, wf.getnframes() - first))
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(ch)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(frames)


def mmss(t: float) -> str:
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--live", required=True, help="tagged live transcript .txt")
    parser.add_argument("--offline", required=True, help="offline_decode_*.json")
    parser.add_argument("--wav", help="source recording, for --sample clip extraction")
    parser.add_argument("--window", type=float, default=20.0, help="tiling window in seconds (default 20)")
    parser.add_argument("--sample", type=int, default=0, help="build a human-reference pack of N windows")
    parser.add_argument("--worst", type=int, default=4, help="how many of the sample are worst-divergence (rest random)")
    parser.add_argument("--seed", type=int, default=324, help="random-control seed (reproducible sample)")
    parser.add_argument(
        "--terms-file",
        help="call-specific code-switch term list (one per line, # comments) to embed in the sample pack",
    )
    parser.add_argument("--min-words", type=int, default=20, help="ignore windows with fewer offline words when ranking/sampling")
    args = parser.parse_args()

    live = parse_live(Path(args.live))
    offline_doc = json.loads(Path(args.offline).read_text(encoding="utf-8"))
    offline = offline_doc["segments"]
    logger.info("live segments=%d  offline segments=%d", len(live), len(offline))

    duration = max(max((s.end for s in live), default=0.0),
                   max((s["end"] for s in offline), default=0.0))
    n_windows = int(duration // args.window) + 1
    live_buckets = bucket_tokens([(s.start, s.end, s.text) for s in live], args.window, n_windows)
    off_buckets = bucket_tokens([(s["start"], s["end"], s["text"]) for s in offline], args.window, n_windows)

    # --- 1. per-window divergence (single-counted tiling) --------------------------
    rows: list[dict] = []
    for k in range(n_windows):
        lt, ot = live_buckets[k], off_buckets[k]
        if not lt and not ot:
            continue
        matched = agreement(ot, lt)
        w_start, w_end = k * args.window, (k + 1) * args.window
        langs = {s.language for s in live if s.end > w_start and s.start < w_end}
        rows.append(
            {
                "window": k,
                "start": w_start,
                "end": w_end,
                "languages": sorted(langs),
                "offline_words": len(ot),
                "live_words": len(lt),
                "matched": matched,
                "divergence": round(1 - matched / len(ot), 4) if ot else 1.0,
                "live_text": " ".join(s.text for s in live if s.end > w_start and s.start < w_end),
                "offline_text": offline_window_text(offline, w_start, w_end),
            }
        )

    tot_off = sum(r["offline_words"] for r in rows)
    tot_live = sum(r["live_words"] for r in rows)
    tot_matched = sum(r["matched"] for r in rows)
    windowed_div = 1 - tot_matched / tot_off if tot_off else 0.0

    # --- 2. global agreement (whole call, no windowing) ----------------------------
    live_tokens = normalise(" ".join(s.text for s in live))
    offline_tokens = normalise(" ".join(s["text"] for s in offline))
    agree = global_agreement(live_tokens, offline_tokens)

    # --- 3. duplication in the live output ----------------------------------------
    dup, dup_total, per_dup = duplication_rate(live)

    print(f"live segments        : {len(live)}   offline segments: {len(offline)}")
    print(f"live tokens          : {agree['live_tokens']}")
    print(f"offline tokens       : {agree['offline_tokens']}")
    print(f"global agreement     : {agree['agreement_vs_offline']:.1%} of offline tokens matched "
          f"({agree['agreement_vs_live']:.1%} of live tokens)")
    print(f"  -> content the live loop does NOT have: {1 - agree['agreement_vs_offline']:.1%} of the offline decode")
    print(f"windowed divergence  : {windowed_div:.1%} over {args.window:.0f}s windows "
          f"({tot_matched} matched / {tot_off} offline words; live {tot_live})")
    print(f"live duplication     : {dup}/{dup_total} tokens ({dup / dup_total:.1%}) repeat the previous segment at the seam")

    ranked = sorted([r for r in rows if r["offline_words"] >= args.min_words],
                    key=lambda r: r["divergence"], reverse=True)
    print(f"\nworst-diverging {args.window:.0f}s windows (offline_words >= {args.min_words}):")
    for r in ranked[:10]:
        print(f"  [{mmss(r['start'])}-{mmss(r['end'])}] {'/'.join(r['languages']) or '-':5s} "
              f"div={r['divergence']:.0%}  off={r['offline_words']}w live={r['live_words']}w")

    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "live_transcript": Path(args.live).name,
        "offline_decode": Path(args.offline).name,
        "what_this_is": "DIVERGENCE between two machine decodes - NOT word error rate.",
        "window_seconds": args.window,
        "global_agreement": agree,
        "windowed_divergence": round(windowed_div, 4),
        "windowed_offline_words": tot_off,
        "windowed_live_words": tot_live,
        "windowed_matched": tot_matched,
        "live_duplication_tokens": dup,
        "live_total_tokens": dup_total,
        "live_duplication_rate": round(dup / dup_total, 4) if dup_total else 0.0,
        "windows": rows,
    }
    report_path = OUTPUT_DIR / f"divergence_report_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("divergence report -> %s", report_path.name)
    print(f"\nreport: {report_path}")

    # --- 4. human-reference sample pack -------------------------------------------
    if args.sample:
        rng = random.Random(args.seed)
        eligible = [r for r in rows if r["offline_words"] >= args.min_words]
        chosen: dict[int, tuple[str, dict]] = {}
        # Stratum A - adversarial: the worst windows. Scored SEPARATELY; including
        # these in one pooled WER would bias it upward by construction.
        for r in ranked[: args.worst]:
            chosen[r["window"]] = ("worst-divergence", r)
        # Stratum B - the English stretch, so term recall has evidence.
        english = [r for r in eligible if "en" in r["languages"] and r["window"] not in chosen]
        for r in rng.sample(english, min(2, len(english))):
            chosen[r["window"]] = ("english-stretch", r)
        # Stratum C - uniform random over the rest: this is the UNBIASED estimate.
        pool = [r for r in eligible if r["window"] not in chosen]
        for r in rng.sample(pool, min(max(0, args.sample - len(chosen)), len(pool))):
            chosen[r["window"]] = ("random-control", r)

        picks = sorted(chosen.items(), key=lambda kv: kv[1][1]["start"])
        clip_dir = OUTPUT_DIR / f"wer_sample_{stamp}"
        if args.wav:
            clip_dir.mkdir(parents=True, exist_ok=True)
        n_rand = sum(1 for _, (why, _) in picks if why == "random-control")
        lines = [
            f"# Human reference sample - {Path(args.live).name}",
            f"# built {datetime.now():%Y-%m-%d %H:%M} - seed {args.seed} - n={len(picks)} windows of {args.window:.0f}s",
            f"# strata: {n_rand} random-control (the unbiased estimate) + "
            f"{sum(1 for _, (w, _) in picks if w == 'worst-divergence')} worst-divergence (adversarial) + "
            f"{sum(1 for _, (w, _) in picks if w == 'english-stretch')} english-stretch",
            "#",
            "# HOW TO FILL THIS IN: play each clip and, under TRUE:, write what was ACTUALLY",
            "# said - verbatim, in the language spoken, on ONE line. Do not copy either machine",
            "# transcript; they are shown only so you can see what is at stake. Case, punctuation",
            "# and diacritics are ignored by the scorer. Leave TRUE: empty to drop that window.",
            "# If both speakers talk in a window, transcribe both, in the order spoken.",
            "# The LIVE:/OFFLINE: lines show every segment OVERLAPPING the window, so they can run",
            "# a little past the clip's edges. Transcribe ONLY what you hear in the clip.",
            "#",
            f"# Clips: scripts/outputs/wer_sample_{stamp}/  (audio - dies with the recording, D11)",
            f"# Score with: python scripts/score_transcript.py --sample-pack scripts/inputs/wer_reference_{stamp}.txt",
            "",
        ]
        for i, (k, (why, r)) in enumerate(picks, start=1):
            clip_name = f"clip_{i:02d}_{mmss(r['start']).replace(':', 'm')}.wav"
            if args.wav:
                write_clip(Path(args.wav), clip_dir / clip_name, r["start"], r["end"])
            lines += [
                f"=== {i:02d} [{mmss(r['start'])}-{mmss(r['end'])}] {'/'.join(r['languages']) or 'pl'} "
                f"| {why} | divergence {r['divergence']:.0%} | clip {clip_name}",
                f"LIVE:    {r['live_text']}",
                f"OFFLINE: {r['offline_text']}",
                "TRUE:",
                "",
            ]
        if args.terms_file:
            raw = Path(args.terms_file).read_text(encoding="utf-8").splitlines()
            terms = [t.strip() for t in raw if t.strip() and not t.startswith("#")]
            lines += [
                f"=== TERMS ({len(terms)}) - from {Path(args.terms_file).name}",
                "# Strike out (delete) any line that was NOT actually spoken in this call.",
                *terms,
                "",
            ]
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        tmpl = INPUT_DIR / f"wer_reference_{stamp}.txt"
        tmpl.write_text("\n".join(lines), encoding="utf-8")
        logger.info("sample pack: %d windows -> %s", len(picks), tmpl.name)
        print(f"sample template: {tmpl}")
        if args.wav:
            print(f"clips         : {clip_dir}  ({len(picks)} x {args.window:.0f}s)")


if __name__ == "__main__":
    main()
