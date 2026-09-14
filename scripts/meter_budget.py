"""Size the P5 meter's decode allowance from measured line arrivals (#428).

Deterministic — arithmetic over `live_transcribe.py --latency-trace` rows, no model call
(CLAUDE.md, Determinism First).

**What the meter actually promises.** `dashboard.meter_state` measures the wall-clock gap
between LINE ARRIVALS and promises the next one within
`SEGMENT_MAX_SECONDS + METER_DECODE_ALLOWANCE_SECONDS`. So the allowance is not "how long a
decode takes": line N-1 lands at `end(N-1) + latency(N-1)` and line N at `end(N) + latency(N)`,
so what the meter sees is

    gap = (end(N) - end(N-1))  +  (latency(N) - latency(N-1))

A **uniform** slowdown largely cancels in that difference — only the *variation* in latency
reaches the meter. A `duration + decode` proxy therefore over-states the cost badly, and this
script does not use one.

`spacing = end(N) - end(N-1)` is bounded by `SEGMENT_MAX_SECONDS` only when the two segments are
back to back. A wider spacing means silence between them, or a segment that decoded to nothing —
a real wait the cap never promised to bound (the real call's 43.2 s silence is this case), which
the meter is right to report at any allowance. Only back-to-back gaps can be *false* overdues.

**Why a bigger allowance is not free.** The ceiling is also how long the meter waits before it
tells the user something is wrong. Every second added to cover a rare healthy line is a second of
extra blindness on *every* line. Both sides are tabled so the number is chosen, not argued.

Usage:
    python scripts/meter_budget.py --trace scripts/outputs/meter_A_ship.jsonl --labels ship
    python scripts/meter_budget.py --trace A.jsonl B.jsonl C.jsonl --labels ship forced no-d26
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings  # noqa: E402
from scripts.logger import get_logger  # noqa: E402

logger = get_logger("meter_budget")


def load(path: Path) -> list[dict]:
    """Read a latency trace, oldest arrival first."""
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return sorted(rows, key=lambda r: r["written_at"])


def arrivals(rows: list[dict]) -> list[dict]:
    """Only the segments that actually put a line on screen — the meter sees nothing else."""
    return [r for r in rows if not r["dropped"]]


def pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile — no interpolation, so every number quoted is one that happened."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(q / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[k]


def close_to_write_table(rows: list[dict]) -> list[tuple[str, int, float, float, float]]:
    """close->write by segment-duration band: (label, n, mean, p95, max)."""
    bands = [(0.0, 10.0, "<10s"), (10.0, 20.0, "10-20s"), (20.0, 29.9, "20-30s"),
             (29.9, 1e9, "30s (cap)")]
    out = []
    for lo, hi, label in bands:
        vals = [r["close_to_write"] for r in rows
                if r["close_to_write"] is not None and lo <= r["duration"] < hi]
        if vals:
            out.append((label, len(vals), st.mean(vals), pct(vals, 95), max(vals)))
    return out


def classify_gaps(rows: list[dict], cap: float) -> list[dict]:
    """Every arrival gap, with the one fact that decides whether the meter may flag it."""
    seen = arrivals(rows)
    out = []
    for i in range(1, len(seen)):
        prev, cur = seen[i - 1], seen[i]
        out.append(
            {
                "index": cur["index"],
                "gap": cur["written_at"] - prev["written_at"],
                "spacing": cur["end"] - prev["end"],
                "latency_step": (cur["close_to_write"] or 0.0) - (prev["close_to_write"] or 0.0),
                "duration": cur["duration"],
                "close_to_write": cur["close_to_write"] or 0.0,
                "healthy": (cur["end"] - prev["end"]) <= cap + 1e-6,
            }
        )
    return out


def budget_table(
    rows: list[dict], cap: float, allowances: list[float]
) -> list[tuple[float, float, int, int]]:
    """(allowance, ceiling, false overdues, total back-to-back gaps)."""
    healthy = [x for x in classify_gaps(rows, cap) if x["healthy"]]
    return [
        (a, cap + a, sum(1 for x in healthy if x["gap"] > cap + a), len(healthy))
        for a in allowances
    ]


def minimum_allowance(rows: list[dict], cap: float) -> tuple[float, dict | None]:
    """The smallest allowance with ZERO false overdues, and the gap that sets it."""
    healthy = [x for x in classify_gaps(rows, cap) if x["healthy"]]
    if not healthy:
        return 0.0, None
    worst = max(healthy, key=lambda x: x["gap"])
    return max(0.0, worst["gap"] - cap), worst


def report(path: Path, label: str, cap: float, allowances: list[float]) -> dict:
    rows = load(path)
    seen = arrivals(rows)
    dropped = [r for r in rows if r["dropped"]]
    c2w = [r["close_to_write"] for r in rows if r["close_to_write"] is not None]
    behind = [r["behind_partial"] for r in rows if r["behind_partial"] >= 0.001]
    waits = [r["queue_wait"] for r in rows if r["queue_wait"] >= 0.001]

    print(f"\n{'=' * 78}\n{label}  ({path.name})\n{'=' * 78}")
    print(f"segments {len(rows)}  ->  {len(seen)} lines, {len(dropped)} produced no line")
    print(f"close->write, all segments: mean {st.mean(c2w):.3f}s  p95 {pct(c2w, 95):.3f}s  "
          f"max {max(c2w):.3f}s")
    for name, n, mean, p95, mx in close_to_write_table(rows):
        print(f"    {name:>11}  n={n:3d}  mean {mean:.3f}s  p95 {p95:.3f}s  max {mx:.3f}s")
    print(f"queue waits >= 1 ms: {len(waits)}/{len(rows)}"
          + (f"  worst {max(waits):.3f}s" if waits else ""))
    print(f"waited behind an in-flight provisional (D25): {len(behind)}/{len(rows)}"
          + (f"  worst {max(behind):.3f}s" if behind else "  worst 0.000s"))
    over = [r for r in rows
            if (r["close_to_write"] or 0) > settings.METER_DECODE_ALLOWANCE_SECONDS]
    print(f"close->write over the shipped {settings.METER_DECODE_ALLOWANCE_SECONDS:g}s: "
          f"{len(over)}/{len(rows)} segments  <- NOT a meter breach; see the gaps below")

    g = classify_gaps(rows, cap)
    healthy = [x for x in g if x["healthy"]]
    real = [x for x in g if not x["healthy"]]
    steps = [x["latency_step"] for x in healthy]
    print(f"\narrival gaps: {len(g)}  ({len(healthy)} back-to-back, {len(real)} across a silence "
          f"or a segment that produced no line)")
    if healthy:
        hg = [x["gap"] for x in healthy]
        print(f"  back-to-back gap : mean {st.mean(hg):.2f}s  p95 {pct(hg, 95):.2f}s  "
              f"max {max(hg):.2f}s")
        print(f"  latency STEP (the part that actually reaches the meter): "
              f"mean {st.mean(steps):+.3f}s  p95 {pct(steps, 95):+.3f}s  max {max(steps):+.3f}s")
    if real:
        print(f"  gaps the cap never bounded: max {max(x['gap'] for x in real):.2f}s")

    need, worst = minimum_allowance(rows, cap)
    print(f"\nsmallest allowance with ZERO false overdues: {need:.3f}s"
          + (f"   (set by segment #{worst['index']}: gap {worst['gap']:.2f}s, spacing "
             f"{worst['spacing']:.2f}s, close->write {worst['close_to_write']:.3f}s)"
             if worst else ""))
    print(f"\n  {'allowance':>9} {'ceiling':>8} {'false overdue':>16} {'stall reported later by':>24}")
    for a, ceiling, false_overdue, n_healthy in budget_table(rows, cap, allowances):
        mark = "  <- shipped" if abs(a - settings.METER_DECODE_ALLOWANCE_SECONDS) < 1e-9 else ""
        print(f"  {a:8.2f}s {ceiling:7.2f}s {false_overdue:12d}/{n_healthy:<3d} "
              f"{a:22.2f}s{mark}")

    return {
        "label": label, "trace": path.name, "segments": len(rows), "lines": len(seen),
        "close_to_write_mean": st.mean(c2w), "close_to_write_p95": pct(c2w, 95),
        "close_to_write_max": max(c2w),
        "behind_partial_n": len(behind), "behind_partial_max": max(behind) if behind else 0.0,
        "healthy_gaps": len(healthy), "unbounded_gaps": len(real),
        "max_healthy_gap": max((x["gap"] for x in healthy), default=0.0),
        "max_latency_step": max(steps, default=0.0),
        "minimum_allowance": need,
        "by_band": [{"band": n, "n": k, "mean": m, "p95": p, "max": x}
                    for n, k, m, p, x in close_to_write_table(rows)],
        "budget": [{"allowance": a, "ceiling": c, "false_overdue": f, "healthy_gaps": n}
                   for a, c, f, n in budget_table(rows, cap, allowances)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", nargs="+", required=True, help="latency trace JSONL file(s)")
    ap.add_argument("--labels", nargs="*", default=[], help="a label per trace")
    ap.add_argument("--cap", type=float, default=settings.SEGMENT_MAX_SECONDS,
                    help="SEGMENT_MAX_SECONDS the ceiling is built on")
    ap.add_argument("--allowances", default="0.25,0.5,0.75,1.0,1.5,2.0,3.0",
                    help="candidate METER_DECODE_ALLOWANCE_SECONDS values to table")
    ap.add_argument("--out", default="", help="write the report as JSON here")
    args = ap.parse_args()

    allowances = [float(x) for x in args.allowances.split(",") if x.strip()]
    reports = [
        report(Path(raw), args.labels[i] if i < len(args.labels) else Path(raw).stem,
               args.cap, allowances)
        for i, raw in enumerate(args.trace)
    ]
    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
