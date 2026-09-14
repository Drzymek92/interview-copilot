"""Replay a finished transcript into a NEW growing file, at the cadence it really arrived.

Subtask #322. The D19 seam says a finished transcript replays through the identical code path
as a live one — but only if it *arrives* the way a live one does. `reasoning.py --replay` reads
a complete file in one pass, which proves the reasoning path and tells you nothing about a
dashboard whose whole subject is **waiting**. This writes the lines out one at a time, spaced by
the gaps the real call actually had, so the P5 meter is driven by real timing rather than a
demo loop.

**The arrival model, and why it is not just the end stamp.** A line reaches the screen when its
segment closes *and* is decoded. Segment close is the `end` stamp; decode is D25's measured
model on this exact hardware, `0.248 + 0.0144 * audio_seconds` (median 0.59 s, max 0.96 s).
Replaying on end stamps alone would make every one of the 59 at-cap segments arrive a few
hundred ms early and hide the fact that a 30 s ceiling is already breached by decode — the
finding that set `METER_DECODE_ALLOWANCE_SECONDS`.

`--speed N` divides every gap by N. It is honest only if the meter's ceiling is divided too, so
the dashboard is told: pass `--ceiling-seconds` = ceiling/N. `--speed` prints that number.

Run it:
    python scripts/replay_transcript.py scripts/outputs/live_transcript_20260902_100033.txt
    python scripts/replay_transcript.py FILE --limit 9 --speed 1     # the first 9 lines, real time
    python scripts/replay_transcript.py FILE --speed 8               # the whole call in ~5 min
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from scripts.logger import get_logger  # noqa: E402

logger = get_logger("replay_transcript")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "outputs"
STAMP_RE = re.compile(r"^\[(\d\d):(\d\d)-(\d\d):(\d\d)\]")

# D25's measured decode model for `large-v3-turbo` on this box. Kept here, named, rather than
# folded into a magic constant — if the model or the card changes, this is the line to re-measure.
DECODE_BASE_SECONDS = 0.248
DECODE_PER_AUDIO_SECOND = 0.0144


@dataclass(frozen=True)
class ReplayLine:
    text: str
    start: float
    end: float
    arrival: float      # seconds from run start, when this line would hit the screen


def decode_seconds(audio_seconds: float) -> float:
    """D25's measured decode cost for a segment of this length."""
    return DECODE_BASE_SECONDS + DECODE_PER_AUDIO_SECOND * max(0.0, audio_seconds)


def parse_replay_lines(text: str) -> list[ReplayLine]:
    """Every `[mm:ss-mm:ss] ...` line, with the wall-clock moment it would have appeared.

    Header (`#`) and unparseable lines are dropped — the same contract `parse_transcript_line`
    applies (D19).
    """
    lines: list[ReplayLine] = []
    for raw in text.splitlines():
        match = STAMP_RE.match(raw)
        if not match:
            continue
        start = int(match.group(1)) * 60 + int(match.group(2))
        end = int(match.group(3)) * 60 + int(match.group(4))
        lines.append(ReplayLine(text=raw.rstrip("\n"), start=float(start), end=float(end),
                                arrival=end + decode_seconds(end - start)))
    return lines


def schedule(lines: list[ReplayLine], speed: float = 1.0) -> list[float]:
    """Offsets from t0, so the first line lands immediately and the rest keep their gaps."""
    if not lines:
        return []
    base = lines[0].arrival
    factor = max(1e-6, speed)
    return [(line.arrival - base) / factor for line in lines]


def replay(lines: list[ReplayLine], out_path: Path, speed: float = 1.0,
           header: list[str] | None = None, sleep=time.sleep,
           clock=time.monotonic) -> int:
    """Write `lines` into `out_path`, fsync'd per line exactly as `live_transcribe.py` does.

    The fsync is not ceremony: the D19 tail is a `readline()` on a separate file handle, and an
    unflushed line is a line the dashboard has not received. Replaying without it would measure
    the OS buffer, not the seam.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    offsets = schedule(lines, speed)
    written = 0
    with open(out_path, "w", encoding="utf-8") as handle:
        for line in header or []:
            handle.write(line if line.endswith("\n") else line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        t0 = clock()
        for line, offset in zip(lines, offsets):
            wait = offset - (clock() - t0)
            if wait > 0:
                sleep(wait)
            handle.write(line.text + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("source", help="a finished live_transcript_*.txt to replay")
    parser.add_argument("--out", help="destination (default: a fresh scripts/outputs/replay_transcript_<ts>.txt)")
    parser.add_argument("--speed", type=float, default=1.0, help="divide every gap by this (1.0 = real time)")
    parser.add_argument("--from-line", type=int, default=0, help="skip this many transcript lines")
    parser.add_argument("--limit", type=int, default=0, help="replay at most this many lines (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="print the schedule and exit; writes nothing")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    lines = parse_replay_lines(source.read_text(encoding="utf-8"))
    if args.from_line:
        lines = lines[args.from_line:]
    if args.limit:
        lines = lines[: args.limit]
    if not lines:
        raise SystemExit(f"no `[mm:ss-mm:ss]` lines to replay in {source}")

    offsets = schedule(lines, args.speed)
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    ceiling = float(settings.SEGMENT_MAX_SECONDS) + float(settings.METER_DECODE_ALLOWANCE_SECONDS)
    scaled = ceiling / max(1e-6, args.speed)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # NOT `live_transcript_*`: `reasoning.newest_transcript()` globs that name, so a replay left
    # in scripts/outputs/ would be what `dashboard.py --watch` picks up as the newest run — a
    # rehearsal presented as a live interview. The prefix is the guard; `--follow` consumes it.
    out_path = (Path(args.out).expanduser().resolve() if args.out
                else OUTPUT_DIR / f"replay_transcript_{run_stamp}.txt")

    print(f"replaying {len(lines)} line(s) from {source.name} at x{args.speed:g}", flush=True)
    print(f"  wall clock : {offsets[-1]:.1f}s", flush=True)
    if gaps:
        print(f"  gaps       : max {max(gaps):.1f}s  over the scaled ceiling "
              f"({scaled:.2f}s): {sum(1 for g in gaps if g > scaled)}/{len(gaps)}", flush=True)
    print(f"  meter      : pass --ceiling-seconds {scaled:.3f} to the dashboard "
          f"(ceiling {ceiling:g}s / speed {args.speed:g})", flush=True)
    print(f"  writing    : {out_path}", flush=True)
    if args.dry_run:
        return

    header = [
        "# interview_copilot live transcript (REPLAY — not a live call)",
        f"# replay of  : {source.name}",
        f"# replayed   : {datetime.now().isoformat(timespec='seconds')} at x{args.speed:g}",
        f"# lines      : {len(lines)} (from-line {args.from_line}, limit {args.limit or 'all'})",
        "# arrival     : end stamp + D25 decode model (0.248 + 0.0144*audio_s)",
    ]
    logger.info("replay start: source=%s lines=%d speed=%.2f out=%s",
                source.name, len(lines), args.speed, out_path.name)
    written = replay(lines, out_path, speed=args.speed, header=header)
    logger.info("replay done: %d line(s) -> %s", written, out_path)
    print(f"replayed {written} line(s) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
