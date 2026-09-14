"""Compare `codeswitch_probe.py --replay` runs segment by segment (#397).

Deterministic — arithmetic over transcripts, no model call (CLAUDE.md, Determinism First).

Every run compared here decoded the SAME segments at the SAME boundaries, so this can do
what `diff_transcripts.py` cannot: attribute a difference to `STT_CODESWITCH_MODE` alone
rather than to segmentation, and separate a real change from the model's own run-to-run
sampling. That separation is the whole point — a single A/B pair cannot tell "the feature
changed nothing" from "the model is nondeterministic" (session 11's correction), so this
takes >= 2 runs per mode and reports the within-mode noise floor beside every claim.

**Divergence, never WER (D24):** both sides are Whisper output. Nothing here says which
decode is right; a human ear is the only thing that can.

Usage:
    python scripts/codeswitch_compare.py --baseline off --runs scripts/outputs/cs_*.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.logger import get_logger  # noqa: E402

logger = get_logger("codeswitch_compare")

WORD_RE = re.compile(r"[\w'À-ſ]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    """Lowercased word tokens — the unit every agreement number below is counted in."""
    return WORD_RE.findall(text.lower())


def token_agreement(a: str, b: str) -> tuple[int, int]:
    """(matched tokens, tokens in `a`) between two decodes of the same audio."""
    ta, tb = tokens(a), tokens(b)
    matched = sum(bl.size for bl in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_matching_blocks())
    return matched, len(ta)


def load(paths: list[Path]) -> dict[str, list[dict]]:
    """Group run summaries by mode, keeping the per-segment rows."""
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(paths):
        doc = json.loads(p.read_text(encoding="utf-8"))
        doc["_path"] = p.name
        by_mode[doc["mode"]].append(doc)
    return dict(by_mode)


def changed_segments(run_a: dict, run_b: dict) -> list[int]:
    """Ordinals whose decoded text is not byte-identical between two runs."""
    a = {r["ordinal"]: r["text"] for r in run_a["segments"]}
    b = {r["ordinal"]: r["text"] for r in run_b["segments"]}
    return sorted(o for o in a if a[o] != b.get(o))


def divergence(run_a: dict, run_b: dict) -> float:
    """1 - token agreement over the whole call, run_a's tokens as the denominator."""
    a = {r["ordinal"]: r["text"] for r in run_a["segments"]}
    b = {r["ordinal"]: r["text"] for r in run_b["segments"]}
    matched = total = 0
    for o, text in a.items():
        m, t = token_agreement(text, b.get(o, ""))
        matched += m
        total += t
    return 1.0 - (matched / total if total else 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="codeswitch replay summary JSONs")
    ap.add_argument("--baseline", default="off", help="the mode every other mode is compared against")
    ap.add_argument("--targets", default="", help="comma-separated ordinals to print in full (the known failures)")
    ap.add_argument("--out", default="", help="write the comparison as JSON here")
    args = ap.parse_args()

    by_mode = load([Path(p) for p in args.runs])
    if args.baseline not in by_mode:
        ap.error(f"no runs for baseline mode {args.baseline!r} (have {sorted(by_mode)})")
    targets = [int(x) for x in args.targets.split(",") if x.strip()]
    report: dict = {"baseline": args.baseline, "modes": {}, "targets": {}}

    print("=" * 78)
    print("NOISE FLOOR — the same mode run twice on the same audio")
    print("=" * 78)
    for mode, runs in sorted(by_mode.items()):
        if len(runs) < 2:
            print(f"  {mode:8s}: only {len(runs)} run — no noise floor, claims about it are unsupported")
            continue
        pairs = [(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
        rows = []
        for a, b in pairs:
            ch = changed_segments(a, b)
            rows.append((a["_path"], b["_path"], ch, divergence(a, b)))
            print(f"  {mode:8s}: {a['_path']} vs {b['_path']} — {len(ch)}/{len(a['segments'])} "
                  f"segments differ, divergence {rows[-1][3]:.3%}  {ch}")
        report["modes"].setdefault(mode, {})["noise_floor"] = [
            {"a": a, "b": b, "changed": ch, "divergence": d} for a, b, ch, d in rows
        ]

    base_runs = by_mode[args.baseline]
    print()
    print("=" * 78)
    print(f"EFFECT — each mode against every {args.baseline!r} run (the control is all 110 segments)")
    print("=" * 78)
    for mode, runs in sorted(by_mode.items()):
        if mode == args.baseline:
            continue
        entries = []
        for r in runs:
            for b in base_runs:
                ch = changed_segments(b, r)
                flagged = set(r.get("code_switch_ordinals", []))
                entries.append(
                    {
                        "run": r["_path"], "baseline": b["_path"],
                        "changed": ch,
                        "changed_on_flagged": sorted(set(ch) & flagged),
                        "changed_off_flagged": sorted(set(ch) - flagged),
                        "divergence": divergence(b, r),
                    }
                )
                print(f"  {mode:8s}: {r['_path']} vs {b['_path']} — {len(ch)} segments differ "
                      f"(flagged {sorted(set(ch) & flagged)}, NOT flagged {sorted(set(ch) - flagged)}), "
                      f"divergence {entries[-1]['divergence']:.3%}")
        report["modes"].setdefault(mode, {})["vs_baseline"] = entries

    print()
    print("=" * 78)
    print("COST — decode seconds, and what the trigger costs the segments it fires on")
    print("=" * 78)
    for mode, runs in sorted(by_mode.items()):
        for r in runs:
            segs = r["segments"]
            flagged = [s for s in segs if s.get("code_switch")]
            unflagged = [s for s in segs if not s.get("code_switch")]
            # Per-second decode cost of an ordinary segment, used to price the flagged ones.
            base_rate = (sum(s["latency_seconds"] for s in unflagged)
                         / max(1e-9, sum(s["duration"] for s in unflagged)))
            extra = [s["latency_seconds"] - base_rate * s["duration"] for s in flagged]
            print(f"  {mode:8s} {r['_path']}: decode {r['decode_seconds']:.1f}s over "
                  f"{r['audio_seconds']:.0f}s audio (rtf {r['decode_seconds']/r['audio_seconds']:.4f}); "
                  f"fired on {len(flagged)}/{len(segs)}")
            for s, ex in zip(flagged, extra):
                print(f"      #{s['ordinal']:3d} {s['duration']:5.1f}s  latency {s['latency_seconds']:.3f}s "
                      f"(+{ex:+.3f}s over the unflagged rate)  passes={s['decode_passes']} "
                      f"langs={s['languages']}")
            report["modes"].setdefault(mode, {}).setdefault("cost", []).append(
                {
                    "run": r["_path"],
                    "decode_seconds": r["decode_seconds"],
                    "audio_seconds": r["audio_seconds"],
                    "unflagged_rtf": base_rate,
                    "flagged": [
                        {"ordinal": s["ordinal"], "duration": s["duration"],
                         "latency_seconds": s["latency_seconds"], "extra_seconds": ex,
                         "decode_passes": s["decode_passes"]}
                        for s, ex in zip(flagged, extra)
                    ],
                }
            )

    if targets:
        print()
        print("=" * 78)
        print("THE KNOWN FAILURES — text under each mode")
        print("=" * 78)
        for o in targets:
            print(f"\n--- segment #{o} ---")
            for mode, runs in sorted(by_mode.items()):
                for r in runs:
                    row = next((s for s in r["segments"] if s["ordinal"] == o), None)
                    if row is None:
                        continue
                    report["targets"].setdefault(str(o), []).append(
                        {"mode": mode, "run": r["_path"], "languages": row["languages"],
                         "text": row["text"]}
                    )
                    print(f"  [{mode}/{r['_path']}] langs={row['languages']} passes={row['decode_passes']}")
                    print(f"    {row['text']}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
