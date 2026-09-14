"""The local web dashboard (D18) — the surface a person reads while being interviewed.

Subtask #322. It is the D19 seam plus a clock and nothing else: it tails
`scripts/outputs/live_transcript_*.txt` through `reasoning.follow_transcript()`, runs the same
`reasoning.run_ambient()` loop the CLI runs, and streams what that loop decided to a browser
over a websocket. It loads **no model, allocates no GPU and opens no socket off this machine**.

Four panels, and one rule that shapes all of them:

**Nothing on screen may be false.** This is read under pressure, mid-answer, by someone who
cannot check it. A meter promising a line that has already not arrived, a stale suggestion left
standing after the question moved on, or a `them:` tag the transcript does not support are all
worse than a blank panel. Every panel below therefore has a state for "I do not know".

1. **Transcript** (two-sided, G9/#326) — `them:` left, `you:` right, from the tags
   `live_transcribe.py` writes. An **untagged** line is rendered as untagged and labelled, not
   silently attributed to the interviewer. Known defect this will show rather than hide:
   `SEGMENT_MAX_SECONDS=30` merges whole Q&A exchanges into one line and can mis-tag them
   (#326; the session-7 dry-run saw a candidate question land under `them`). Render what the
   transcript says.

2. **The P5 bounded-wait meter (option A)** — "last line N s ago · next due within <=M s".
   Derived from the tail plus a clock: no second Whisper, no second model, no new dependency,
   and **it renders no speech of its own**, so it cannot be wrong about the interview. The math
   is `meter_state()` below — a pure function, tested — and it is computed **here, in Python**,
   never in the browser, so the number the user reads has exactly one implementation.

   **The ceiling is not `SEGMENT_MAX_SECONDS`.** A line cannot reach the screen until its
   segment closes *and* is decoded. Measured on the real 42-min call (109 arrival gaps): a bare
   30 s ceiling holds only **84/109**, because 54% of segments end exactly at the cap and then
   take D25's measured decode on top, landing at 30.1-30.4 s. With
   `SEGMENT_MAX_SECONDS + METER_DECODE_ALLOWANCE_SECONDS` (31.0 s) it holds **108/109**. The one
   true overrun is **43.2 s** — a silence *between* segments, which the cap does not bound. So
   the overrun state is not a defensive nicety: it is a thing that happens, and when the clock
   passes the ceiling the meter stops promising and says **overdue**.

3. **Suggestions** — D23-gated only. A question the gate dropped produces no suggestion; it is
   reported as a counter, never as a non-answer. A finished suggestion is badged **stale** once
   `SUGGESTION_STALE_LINES` new lines have landed under it.

4. **Plan** (D13) — **read-only**. P3/G7 does not settle how a step is marked *covered*, so no
   auto-covered state machine is invented here. A step is badged `mentioned` on a deterministic
   literal match of one of its `done_signals`; the legend says exactly that, and the word
   "covered" does not appear.

**SI1** — Reconciled: the server binds a loopback address only, and `_assert_loopback()` refuses
to start otherwise; it reads local files and talks to the already-local backend the reasoning
layer uses. It adds no egress. The page loads no font, script or stylesheet from a network.
**SI2** — Reconciled: the panel is visibly, permanently disclosed. There is deliberately no
hide/opacity/click-through/always-on-top affordance, and there must never be one (D11).

Run it:
    python scripts/dashboard.py --session example_ai_engineer --watch      # follow the newest run
    python scripts/dashboard.py --session example_ai_engineer --follow FILE
    python scripts/dashboard.py --no-suggestions --follow FILE     # transcript + meter, no model
    python scripts/dashboard.py --session example_ai_engineer --follow FILE --no-salience  # ungated
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import math
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Imported at MODULE level, deliberately, and it must stay that way. This file uses
# `from __future__ import annotations`, so the websocket handler's `socket: WebSocket`
# annotation is a STRING that FastAPI resolves against these module globals. With the import
# inside `create_app()` the name is a local, FastAPI cannot resolve it, and it treats `socket`
# as an unknown query parameter — the route then closes every connection during the handshake
# with a bare HTTP 403. Every unit test still passes; the dashboard is simply dead.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402

from config import settings  # noqa: E402
from scripts.logger import get_logger  # noqa: E402
from scripts.llm_client import announce_backend, cloud_ready, model_for  # noqa: E402
from scripts.reasoning import (  # noqa: E402
    ContextBundle, Controls, PlanStep, follow_transcript, load_bundle, looks_like_question,
    newest_transcript, run_ambient,
)
from scripts.salience import SalienceGate  # noqa: E402

logger = get_logger("dashboard")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "outputs"
UI_FILE = Path(__file__).resolve().parent / "dashboard_ui.html"


# --------------------------------------------------------------------------
# 1. The P5 meter — pure time math, no I/O, no clock of its own
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MeterState:
    """What the meter says, and nothing it cannot support.

    `state` is one of:
      * `cold`    — no line has arrived yet, so there is no interval to measure. The meter
                    makes **no** promise; it says it is waiting for the first line.
      * `waiting` — inside the ceiling (the only state that promises a due-by bound), OR past
                    it while a D25 provisional line fresher than the last final is on screen
                    (#457): positive evidence the pipeline is alive, so the meter withholds
                    `overdue` rather than restate a promise it can no longer support —
                    `due_within_seconds` is `None` in that case, not a bound.
      * `overdue` — the clock has passed the ceiling and no fresher provisional covers the
                    wait. The promise has already been missed, so the meter withdraws it and
                    says so. Measured 1/109 on the real call.
    """

    state: str
    since_seconds: int | None
    due_within_seconds: int | None
    over_by_seconds: int | None
    ceiling_seconds: float
    label: str


def meter_ceiling(
    segment_max_seconds: float | None = None,
    decode_allowance_seconds: float | None = None,
    override_seconds: float | None = None,
) -> float:
    """The bound the meter promises against (see the module docstring for the measurement).

    `override_seconds > 0` wins outright — its one legitimate use is keeping a TIME-COMPRESSED
    replay truthful (`replay_transcript.py --speed N` wants `ceiling / N`).
    """
    override = settings.METER_CEILING_SECONDS if override_seconds is None else override_seconds
    if override and override > 0:
        return float(override)
    cap = settings.SEGMENT_MAX_SECONDS if segment_max_seconds is None else segment_max_seconds
    allow = (settings.METER_DECODE_ALLOWANCE_SECONDS if decode_allowance_seconds is None
             else decode_allowance_seconds)
    return float(cap) + float(allow)


def meter_state(
    last_line_at: float | None,
    now: float,
    ceiling_seconds: float,
    provisional_fresh: bool = False,
) -> MeterState:
    """The whole of the P5 option-A meter: an elapsed time, a ceiling, and a sentence.

    Pure — it takes the clock as an argument and reads nothing. `last_line_at` and `now` are on
    the same monotonic scale; `None` means no line has arrived yet. `provisional_fresh` is the
    one piece of D25 state the meter needs (#457): the caller has already established that a
    provisional line newer than the last final is currently on screen. `meter_state` does not
    look at the provisional itself — it stays pure and just decides what to do with the fact.

    Rounding is chosen so the meter never flatters itself. The age is **floored** and the
    remaining time is **ceiled**, so "next due within <=16 s" is a bound the arrival can only
    beat, never a deadline it can quietly miss by a rounding artefact.
    """
    ceiling = max(0.0, float(ceiling_seconds))
    if last_line_at is None:
        return MeterState(
            state="cold", since_seconds=None, due_within_seconds=None, over_by_seconds=None,
            ceiling_seconds=ceiling, label="waiting for the first transcript line",
        )
    elapsed = max(0.0, now - last_line_at)
    since = int(elapsed)
    if elapsed > ceiling:
        if provisional_fresh:
            # A fresher provisional is positive evidence the pipeline is alive: the segment is
            # visibly still open, not stalled. Suppress `overdue` (#457) — but do not restate
            # the "next due within <=Ns" promise either, since that promise has already been
            # missed and a rounding artefact must not resurrect it as a new one.
            return MeterState(
                state="waiting", since_seconds=since, due_within_seconds=None,
                over_by_seconds=None, ceiling_seconds=ceiling,
                label=(f"last line {since} s ago · past the {ceiling:g} s ceiling, but a "
                       "newer provisional is on screen"),
            )
        over = int(math.ceil(elapsed - ceiling))
        return MeterState(
            state="overdue", since_seconds=since, due_within_seconds=None, over_by_seconds=over,
            ceiling_seconds=ceiling,
            # No promise is restated here on purpose: the previous one has already been missed.
            label=f"last line {since} s ago · overdue by {over} s (ceiling {ceiling:g} s)",
        )
    remaining = int(math.ceil(ceiling - elapsed))
    return MeterState(
        state="waiting", since_seconds=since, due_within_seconds=remaining, over_by_seconds=None,
        ceiling_seconds=ceiling,
        label=f"last line {since} s ago · next due within ≤{remaining} s",
    )


# --------------------------------------------------------------------------
# 2. Server-side view state — what a browser that connects late must be told
# --------------------------------------------------------------------------
@dataclass
class LineView:
    seq: int
    stamp: str
    speaker: str
    language: str | None
    text: str
    tagged: bool


@dataclass
class ProvisionalView:
    """The D25 provisional line for the segment that is STILL OPEN — at most one at a time.

    Deliberately NOT a `LineView` and not in `lines`: it is not a transcript line, it must not
    move the counters, the plan-mention scan or the suggestion-staleness sequence, and it has to
    be removable. It is a separate slot precisely so "replace it on close" is a slot assignment
    rather than a search-and-edit through the transcript.
    """

    stamp: str
    start_seconds: float           # join key: the START of the segment, which cannot drift
    speaker: str
    language: str | None
    text: str
    tagged: bool


def stamp_start_seconds(stamp: str) -> float:
    """Seconds of the START half of a `mm:ss-mm:ss` stamp; -1.0 if it cannot be read.

    Parsed to a number rather than compared as a string: past 99 minutes `"100:00"` sorts
    BEFORE `"99:00"` lexicographically, which would silently stop superseding on a long call.
    """
    head = stamp.split("-", 1)[0].strip().lstrip("[")
    try:
        minutes, seconds = head.split(":")
        return int(minutes) * 60 + float(seconds)
    except ValueError:
        return -1.0


@dataclass
class SuggestionView:
    id: int
    stamp: str | None
    segment: str
    language: str
    fired_because: str
    text: str = ""
    status: str = "streaming"      # streaming | complete | superseded | failed
    note: str = ""
    line_seq: int = 0              # transcript length when it fired -> staleness


@dataclass
class Counters:
    lines: int = 0
    them: int = 0
    you: int = 0
    untagged: int = 0
    triggered: int = 0             # questions that cleared the D20 rule
    fired: int = 0                 # ... and cleared the D23 gate
    dropped: int = 0               # ... and did not. ONLY the gate; see `cooldown`.
    cooldown: int = 0              # held back by SUGGESTION_COOLDOWN_SECONDS, not judged at all
    # D25 provisional accounting, kept apart from `lines` because a provisional is not a line.
    provisional: int = 0           # provisional lines put on screen
    prov_superseded: int = 0       # ... replaced by the final line for the SAME segment
    prov_orphaned: int = 0         # ... cleared by a LATER segment's final; no final ever came
    prov_expired: int = 0          # ... cleared by the clock (the recorder went away)


@dataclass
class PlanView:
    id: str
    title: str
    key_points: list[str] = field(default_factory=list)
    done_signals: list[str] = field(default_factory=list)
    mentioned: list[str] = field(default_factory=list)   # the signals literally seen, verbatim


class DashboardState:
    """Everything the UI shows, held once, guarded by a lock.

    Both the tail thread and the suggestion's own streaming thread write here, so every
    mutation returns the events to broadcast rather than broadcasting inline — the lock is
    never held across an await or a socket write.
    """

    def __init__(self, plan: Iterable[PlanStep] = (), max_lines: int | None = None,
                 track_mentions: bool | None = None) -> None:
        self._lock = threading.Lock()
        cap = settings.DASHBOARD_MAX_LINES if max_lines is None else max_lines
        self.lines: deque[LineView] = deque(maxlen=cap)
        self.dropped_lines = 0
        self.counters = Counters()
        self.suggestion: SuggestionView | None = None
        # Past suggestions, oldest first, so the user can return to an earlier answer mid-call
        # (the live one stays in `self.suggestion`; it moves here when the next one supersedes it).
        self.history: list[SuggestionView] = []
        self.plan: list[PlanView] = [
            PlanView(id=s.id, title=s.title, key_points=list(s.key_points),
                     done_signals=list(s.done_signals)) for s in plan
        ]
        self.track_mentions = (settings.PLAN_MENTION_TRACKING if track_mentions is None
                               else track_mentions)
        self.last_line_at: float | None = None
        self._seq = 0
        # D25. One slot: only one segment is ever open, so only one provisional can be true.
        self.provisional: ProvisionalView | None = None
        self._provisional_at: float | None = None
        self._last_final_start = -1.0

    # -- reads ------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "lines": [asdict(line) for line in self.lines],
                "dropped_lines": self.dropped_lines,
                "counters": asdict(self.counters),
                "suggestion": asdict(self.suggestion) if self.suggestion else None,
                "history": [asdict(v) for v in self.history],
                "plan": [asdict(step) for step in self.plan],
                "track_mentions": self.track_mentions,
                "provisional": asdict(self.provisional) if self.provisional else None,
            }

    def history_event(self) -> dict:
        """The full scrollback, small and capped, re-sent whenever it changes. Cheaper to
        broadcast whole than to diff, and it makes a reconnect and a live update identical."""
        with self._lock:
            return {"kind": "history", "history": [asdict(v) for v in self.history]}

    def tick(self, now: float, ceiling: float) -> dict:
        """The clock-driven half of the UI: the meter, and how stale the suggestion is.

        `provisional_fresh` (#457) is computed here, under the lock, from state `tick` never
        mutates: a provisional currently held in the slot whose segment start is newer than the
        last final's. This does NOT touch `last_line_at` — a provisional is not an arrival
        (D25/P5), and resetting it here would tell the meter the wait ended when no transcript
        line has actually landed.
        """
        with self._lock:
            provisional_fresh = (
                self.provisional is not None
                and self.provisional.start_seconds > self._last_final_start
            )
            meter = meter_state(self.last_line_at, now, ceiling, provisional_fresh)
            stale_after = settings.SUGGESTION_STALE_LINES
            age = None
            if self.suggestion is not None:
                lines_since = self.counters.lines - self.suggestion.line_seq
                age = {
                    "lines_since": lines_since,
                    # A suggestion still streaming is answering the current question by
                    # definition; only a finished one can be left standing past its moment.
                    "stale": self.suggestion.status == "complete" and lines_since >= stale_after,
                    "stale_after": stale_after,
                }
            return {"kind": "tick", "meter": asdict(meter), "suggestion_age": age}

    # -- writes (called from worker threads) ------------------------------
    def add_line(self, stamp: str, speaker: str, language: str | None, text: str,
                 tagged: bool, at: float) -> list[dict]:
        events: list[dict] = []
        with self._lock:
            self._seq += 1
            if self.lines.maxlen is not None and len(self.lines) == self.lines.maxlen:
                self.dropped_lines += 1
            line = LineView(seq=self._seq, stamp=stamp, speaker=speaker, language=language,
                            text=text, tagged=tagged)
            self.lines.append(line)
            self.counters.lines += 1
            if not tagged:
                self.counters.untagged += 1
            elif speaker == "you":
                self.counters.you += 1
            else:
                self.counters.them += 1
            self.last_line_at = at
            self._last_final_start = max(self._last_final_start, stamp_start_seconds(stamp))
            cleared = self._clear_provisional_locked(stamp_start_seconds(stamp))
            events.append({"kind": "line", "line": asdict(line),
                           "counters": asdict(self.counters),
                           "dropped_lines": self.dropped_lines})
            if cleared:
                # AFTER the line event, so the browser never blanks the provisional before the
                # text that replaces it is on screen. This is the #322 bar: at no point does the
                # screen show neither, and at no point does it keep showing something false.
                events.append({"kind": "provisional", "provisional": None,
                               "counters": asdict(self.counters)})
            if self.track_mentions:
                hits = self._note_mentions(text)
                if hits:
                    events.append({"kind": "plan", "plan": [asdict(s) for s in self.plan]})
        return events

    def _clear_provisional_locked(self, final_start: float) -> bool:
        """Drop the provisional once a final line at or past its segment has landed.

        `==` is supersession — the final for that very segment. `>` means a LATER segment
        already closed while this provisional's own final never came: its segment was dropped
        as silence or decoded to nothing (`decode_segment` returns None below the peak floor).
        Both leave text on screen that no longer describes anything being said, so both clear
        it; they are counted apart because an orphan is the case worth noticing.
        """
        current = self.provisional
        if current is None or final_start < current.start_seconds:
            return False
        if final_start == current.start_seconds:
            self.counters.prov_superseded += 1
        else:
            self.counters.prov_orphaned += 1
        self.provisional = None
        self._provisional_at = None
        return True

    def expire_provisional(self, now: float, ceiling: float) -> list[dict]:
        """Clear a provisional the recorder has stopped updating. Called by ONE owner thread.

        Deliberately not folded into `tick()`: `tick` runs once per connected browser, so a
        mutation there would clear the slot for whichever socket ticked first and leave every
        other browser showing the line forever.

        A segment cannot stay open past the ceiling (`SEGMENT_MAX_SECONDS` + the decode
        allowance), so a provisional older than that is not "still being said" — the recorder
        stopped, crashed, or its final was lost. Leaving it up would be the screen asserting
        something it can no longer support.
        """
        with self._lock:
            if (self.provisional is None or self._provisional_at is None
                    or now - self._provisional_at <= ceiling):
                return []
            self.provisional = None
            self._provisional_at = None
            self.counters.prov_expired += 1
            return [{"kind": "provisional", "provisional": None,
                     "counters": asdict(self.counters)}]

    def add_provisional(self, stamp: str, speaker: str, language: str | None, text: str,
                        tagged: bool, at: float) -> list[dict]:
        """Show the D25 provisional for the open segment (or refuse it if it is already late).

        `last_line_at` is deliberately NOT touched: the meter measures the wait for a
        TRANSCRIPT line, and a provisional is not an arrival — it is a look at a segment that
        has not arrived yet. Letting it reset the meter would make the meter say the wait ended
        when the thing being waited for had not been written.
        """
        start = stamp_start_seconds(stamp)
        events: list[dict] = []
        with self._lock:
            if not text or start < 0 or start <= self._last_final_start:
                # Its final beat it here. The recorder drops this case before it costs a decode
                # (RunOutputs.superseded), but the consumer must hold the same line on its own:
                # the two tails are separate files read by separate threads, so arrival order
                # across them is not guaranteed by anything the recorder does.
                return events
            self.provisional = ProvisionalView(
                stamp=stamp, start_seconds=start, speaker=speaker, language=language,
                text=text, tagged=tagged,
            )
            self._provisional_at = at
            self.counters.provisional += 1
            events.append({"kind": "provisional",
                           "provisional": asdict(self.provisional),
                           "counters": asdict(self.counters)})
        return events

    def _note_mentions(self, text: str) -> bool:
        """Deterministic, literal, case-insensitive `done_signals` match. Called under the lock.

        This is the ONLY plan tracking in v1 and it claims exactly what it does: a phrase the
        plan listed was said. It is not evidence the step was covered, and the UI does not say
        it is (P3/G7 is open; CLAUDE.md Determinism First rules out an LLM judge per segment,
        and D20/D23 are the precedent for a rule the user can audit).
        """
        lowered = text.lower()
        changed = False
        for step in self.plan:
            for phrase in step.done_signals:
                token = phrase.strip().lower()
                if token and token in lowered and phrase not in step.mentioned:
                    step.mentioned.append(phrase)
                    changed = True
        return changed

    def note_gate(self, fire: bool) -> dict:
        with self._lock:
            self.counters.triggered += 1
            if fire:
                self.counters.fired += 1
            else:
                self.counters.dropped += 1
            return {"kind": "counters", "counters": asdict(self.counters)}

    def note_skip(self, reason: str) -> dict:
        """A question the gate never judged. Kept apart from `dropped` on purpose: with the
        gate switched off, a cooldown counted as a gate drop reports a gate that is not
        running — which is exactly the kind of false line this panel must not show."""
        with self._lock:
            if reason == "cooldown":
                self.counters.cooldown += 1
            return {"kind": "counters", "counters": asdict(self.counters)}

    def _archive_current(self) -> None:
        """Move the outgoing suggestion into history before the new one overwrites the slot.
        Caller holds the lock. A suggestion still streaming when it is displaced never finished,
        so record it as `superseded`; a `complete`/`failed` one keeps its terminal status."""
        prev = self.suggestion
        if prev is None:
            return
        if prev.status == "streaming":
            prev.status = "superseded"
        self.history.append(prev)
        cap = settings.SUGGESTION_HISTORY_MAX
        if cap and len(self.history) > cap:
            del self.history[: len(self.history) - cap]

    def suggestion_start(self, sid: int, stamp: str | None, segment: str, language: str,
                         fired_because: str) -> dict:
        with self._lock:
            self._archive_current()
            self.suggestion = SuggestionView(
                id=sid, stamp=stamp, segment=segment, language=language,
                fired_because=fired_because, line_seq=self.counters.lines,
            )
            return {"kind": "suggestion", "suggestion": asdict(self.suggestion)}

    def suggestion_token(self, sid: int, text: str) -> dict | None:
        with self._lock:
            if self.suggestion is None or self.suggestion.id != sid:
                return None      # a superseded stream's tail must not append to its successor
            self.suggestion.text += text
            return {"kind": "suggestion_token", "id": sid, "text": text}

    def suggestion_end(self, sid: int, status: str, note: str) -> dict | None:
        with self._lock:
            if self.suggestion is None or self.suggestion.id != sid:
                return None
            self.suggestion.status = status
            self.suggestion.note = note
            return {"kind": "suggestion", "suggestion": asdict(self.suggestion)}


# --------------------------------------------------------------------------
# 3. Fan-out from worker threads to websocket clients
# --------------------------------------------------------------------------
class EventBus:
    """Thread -> asyncio fan-out. Publishers are plain threads; subscribers are websockets.

    A slow or wedged browser must not stall the tail thread, so each subscriber has a bounded
    queue and the OLDEST message is dropped when it overflows — with a counter, so the UI can
    say it fell behind instead of quietly showing an incomplete call.
    """

    def __init__(self, maxsize: int = 2000) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: set[asyncio.Queue] = set()
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self.overflow = 0

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        with self._lock:
            self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._queues.discard(queue)

    def publish(self, event: dict) -> None:
        """Safe to call from any thread, including before a loop exists."""
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._deliver, event)
        except RuntimeError:      # loop already closed during shutdown
            pass

    def _deliver(self, event: dict) -> None:
        with self._lock:
            queues = list(self._queues)
        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self.overflow += 1
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass


# --------------------------------------------------------------------------
# 4. The tail -> state -> bus path (this is what the tests drive)
# --------------------------------------------------------------------------
def suggestions_log_path(transcript_path: Path) -> Path:
    """Where completed suggestions are appended so the call can be reviewed afterwards.
    `live_transcript_<stamp>.txt` -> `live_suggestions_<stamp>.jsonl`, next to the transcript,
    so a session's transcript, audio and suggestions share one stamp."""
    stem = transcript_path.stem
    stamp = stem[len("live_transcript_"):] if stem.startswith("live_transcript_") else stem
    return transcript_path.with_name(f"live_suggestions_{stamp}.jsonl")


class SuggestionLog:
    """Appends one JSON line per completed suggestion. A live-interview log, so a write failure
    is logged and swallowed — losing the on-disk record must never take the copilot down."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, suggestion: dict) -> None:
        try:
            record = {"logged_at": datetime.now(timezone.utc).isoformat(), **suggestion}
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("could not append a suggestion to %s", self.path)


def make_sink(state: DashboardState, bus: EventBus,
              clock: Callable[[], float] = time.monotonic,
              suggestion_log: "SuggestionLog | None" = None) -> Callable[[dict], None]:
    """Translate one `run_ambient` event into state mutations and broadcasts.

    Deliberately the only place that knows both halves. `run_ambient` knows nothing about a
    websocket; the UI knows nothing about the ambient loop.
    """

    def sink(event: dict) -> None:
        kind = event.get("kind")
        if kind == "line":
            for out in state.add_line(
                stamp=str(event.get("stamp", "")), speaker=str(event.get("speaker", "them")),
                language=event.get("language"), text=str(event.get("text", "")),
                tagged=bool(event.get("tagged")), at=clock(),
            ):
                bus.publish(out)
        elif kind == "provisional":
            for out in state.add_provisional(
                stamp=str(event.get("stamp", "")), speaker=str(event.get("speaker", "them")),
                language=event.get("language"), text=str(event.get("text", "")),
                tagged=bool(event.get("tagged")), at=clock(),
            ):
                bus.publish(out)
        elif kind == "gate":
            bus.publish(state.note_gate(bool(event.get("fire"))))
        elif kind == "skip":
            bus.publish(state.note_skip(str(event.get("reason", ""))))
        elif kind == "suggestion":
            phase = event.get("phase")
            sid = int(event.get("id", 0))
            if phase == "start":
                bus.publish(state.suggestion_start(
                    sid, event.get("stamp"), str(event.get("segment", "")),
                    str(event.get("language", "")), str(event.get("fired_because", "")),
                ))
                # The start just archived the previous suggestion; push the new scrollback.
                bus.publish(state.history_event())
            elif phase == "token":
                out = state.suggestion_token(sid, str(event.get("text", "")))
                if out:
                    bus.publish(out)
            elif phase == "end":
                out = state.suggestion_end(sid, str(event.get("status", "complete")),
                                           str(event.get("note", "")))
                if out:
                    bus.publish(out)
                    if suggestion_log is not None and out["suggestion"].get("status") == "complete":
                        suggestion_log.write(out["suggestion"])

    return sink


def tail_only(path: Path, state: DashboardState, bus: EventBus,
              stop_after_idle: float | None = None, from_start: bool = True,
              clock: Callable[[], float] = time.monotonic) -> int:
    """The transcript+meter half with **no model, no GPU and no suggestion** (`--no-suggestions`).

    The same `follow_transcript` seam the ambient loop uses, so the two paths cannot drift.
    """
    sink = make_sink(state, bus, clock=clock)
    count = 0
    for stamp, speaker, language, text in follow_transcript(
        path, stop_after_idle=stop_after_idle, from_start=from_start
    ):
        sink({"kind": "line", "stamp": stamp, "speaker": speaker or "them",
              "language": language, "text": text, "tagged": speaker is not None})
        count += 1
    return count


def partial_path_for(transcript: Path) -> Path:
    """`live_transcript_<stamp>.txt` -> `live_transcript_<stamp>.partial` (D25).

    Derived, not configured: the two files share the run stamp by construction (TRK), so a
    separate `--follow-partial` would only create a way to point them at different runs.
    """
    return transcript.with_suffix(".partial")


def tail_partial(path: Path, state: DashboardState, bus: EventBus,
                 stop_after_idle: float | None = None, from_start: bool = False,
                 clock: Callable[[], float] = time.monotonic,
                 wait_seconds: float = 0.0,
                 stop_event: "threading.Event | None" = None) -> int:
    """Tail the `.partial` (D25) into the provisional slot. No model, no GPU — the same seam.

    The `.partial` has the same line format as the transcript by design, so this is
    `follow_transcript` again: one parser, and a provisional line cannot drift from a final
    one. The file is optional — with `PARTIAL_DECODE_ENABLED=0` it never exists, and that is
    a supported run, not an error.

    `from_start` defaults to **False** where the transcript tail defaults to True, and the
    asymmetry is the point: the transcript is the record and a late browser is owed all of it,
    while a provisional is only ever a claim about *now*. Replaying the file from the top would
    race the transcript tail and briefly show a provisional for a segment that closed minutes
    ago.
    """
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while not path.exists():
        if stop_event is not None and stop_event.is_set():
            return 0
        if time.monotonic() >= deadline:
            logger.info("no %s — running without provisional lines (D25)", path.name)
            return 0
        time.sleep(0.25)
    sink = make_sink(state, bus, clock=clock)
    count = 0
    for stamp, speaker, language, text in follow_transcript(
        path, stop_after_idle=stop_after_idle, from_start=from_start, stop_event=stop_event
    ):
        sink({"kind": "provisional", "stamp": stamp, "speaker": speaker or "them",
              "language": language, "text": text, "tagged": speaker is not None})
        count += 1
    return count


# --------------------------------------------------------------------------
# 5. The server (D18)
# --------------------------------------------------------------------------
def _assert_loopback(host: str) -> None:
    """SI1/D18 enforcement, not documentation: this server does not bind a routable address.

    A misread env var is the realistic way an interview transcript ends up on a LAN, so the
    check is a hard refusal with a named cause rather than a default that can be overridden.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host.lower() in ("localhost",):
            return
        raise SystemExit(
            f"refusing to start: DASHBOARD_HOST={host!r} is not a loopback address. "
            "D18/SI1 binds 127.0.0.1 only — the transcript never leaves this machine."
        ) from None
    if not address.is_loopback:
        raise SystemExit(
            f"refusing to start: DASHBOARD_HOST={host!r} is not a loopback address. "
            "D18/SI1 binds 127.0.0.1 only — the transcript never leaves this machine."
        )


@dataclass
class RunInfo:
    """The provenance line the UI shows, so nobody mistakes a replay for a live call."""

    source: str
    mode: str                  # live | replay | tail-only
    session: str
    salience: str
    ceiling_seconds: float
    suggestions: bool
    provisional: str = "off"   # whether the D25 .partial is being rendered


def create_app(state: DashboardState, bus: EventBus, info: RunInfo,
               ceiling: float, clock: Callable[[], float] = time.monotonic,
               controller: "AppController | None" = None):
    """Build the FastAPI app around an already-running `DashboardState`.

    `controller` (#607 app mode) enables the inbound `POST /control` switch endpoint. It stays
    loopback-only: the app binds 127.0.0.1 (asserted in main before anything opens), so this
    adds no new network surface — the same origin that reads the socket flips the switches."""
    app = FastAPI(title="interview copilot dashboard", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(UI_FILE.read_text(encoding="utf-8"))

    @app.post("/control")
    async def control(payload: dict) -> JSONResponse:
        # The only inbound path. Runs the (blocking) recorder spawn/stop in a worker thread so
        # the event loop that serves the websocket is never held during a Whisper reload.
        if controller is None:
            return JSONResponse({"error": "control endpoint is off (not running in --app mode)"},
                                status_code=404)
        switch = str(payload.get("switch", ""))
        value = payload.get("value")
        try:
            if switch == "transcription":
                state_dict = await asyncio.to_thread(
                    controller.start_transcription if value else controller.stop_transcription)
                return JSONResponse({"ok": True, **state_dict})
            if switch == "suggestions":
                return JSONResponse({"ok": True, **controller.set_suggestions(bool(value))})
            if switch == "backend":
                state_dict, banner = await asyncio.to_thread(controller.set_backend, str(value))
                return JSONResponse({"ok": True, "banner": banner, **state_dict})
            return JSONResponse({"error": f"unknown switch {switch!r}"}, status_code=400)
        except ControlError as exc:
            return JSONResponse({"error": str(exc), **controller.ui_state()}, status_code=400)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        # The D25 tally is here as well as on screen so a headless run can be audited without
        # a browser — which is how the #399 demonstration was graded.
        return JSONResponse({"ok": True, "lines": state.counters.lines,
                             "mode": info.mode, "ceiling_seconds": ceiling,
                             "provisional": {
                                 "showing": state.provisional.stamp if state.provisional else None,
                                 "shown": state.counters.provisional,
                                 "superseded": state.counters.prov_superseded,
                                 "orphaned": state.counters.prov_orphaned,
                                 "expired": state.counters.prov_expired,
                             }})

    @app.websocket("/ws")
    async def stream(socket: WebSocket) -> None:
        await socket.accept()
        bus.bind(asyncio.get_running_loop())
        queue = bus.subscribe()
        try:
            # `state.tick()` carries its own "kind", so it is spread FIRST and "hello" is
            # written last — otherwise the snapshot arrives labelled "tick" and the client
            # renders none of it.
            await socket.send_text(json.dumps({
                **state.tick(clock(), ceiling), **state.snapshot(),
                "info": asdict(info), "kind": "hello",
                "controls": controller.ui_state() if controller is not None else None,
            }))
            ticker = asyncio.create_task(_tick_loop(socket, state, ceiling, clock))
            try:
                while True:
                    event = await queue.get()
                    await socket.send_text(json.dumps(event))
            finally:
                ticker.cancel()
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("websocket closed on an error")
        finally:
            bus.unsubscribe(queue)

    return app


async def _tick_loop(socket, state: DashboardState, ceiling: float,
                     clock: Callable[[], float]) -> None:
    """Push the meter on a cadence. The browser never computes it (see the module docstring)."""
    interval = max(0.05, settings.METER_TICK_SECONDS)
    try:
        while True:
            await socket.send_text(json.dumps(state.tick(clock(), ceiling)))
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        raise
    except Exception:
        return


# --------------------------------------------------------------------------
# 6. CLI
# --------------------------------------------------------------------------
REPLAY_MARKER = "(REPLAY"


def detect_mode(path: Path, declared: str) -> str:
    """A transcript that says it is a replay is labelled a replay, whatever `--mode` claims.

    `--mode` is a human-supplied label and humans forget; the file's own header is evidence. The
    one thing this panel must never say is that a rehearsal is a live interview.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for _ in range(12):
                line = handle.readline()
                if not line or not line.startswith("#"):
                    break
                if REPLAY_MARKER in line:
                    return f"REPLAY — not a live call ({declared})" if declared else "REPLAY — not a live call"
    except OSError:
        pass
    return declared


def _resolve_transcript(args: argparse.Namespace) -> Path:
    if args.follow:
        return Path(args.follow).expanduser().resolve()
    deadline = time.monotonic() + max(0.0, args.wait_seconds)
    while True:
        found = newest_transcript(OUTPUT_DIR)
        if found is not None:
            return found
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"no live_transcript_*.txt in {OUTPUT_DIR} — start live_transcribe.py first, "
                "pass --follow FILE, or raise --wait-seconds."
            )
        time.sleep(0.5)


def _start_provisional(path: Path, state: DashboardState, bus: EventBus,
                       ceiling: float, stop_event: "threading.Event | None" = None) -> None:
    """Two daemon threads for the D25 half: one tails the `.partial`, one ages it out.

    `stop_event` (#607 app mode) ends both when the recorder is stopped and a new transcript
    file will follow, so a toggle does not leak a tail thread per capture cycle."""

    def tail() -> None:
        try:
            tail_partial(partial_path_for(path), state, bus, wait_seconds=2.0,
                         stop_event=stop_event)
        except Exception:
            # A provisional line is a convenience. The transcript half must survive its death.
            logger.exception("the .partial tail died — provisional lines stop, the transcript does not")

    def expire() -> None:
        interval = max(0.25, settings.METER_TICK_SECONDS)
        while stop_event is None or not stop_event.is_set():
            time.sleep(interval)
            try:
                for event in state.expire_provisional(time.monotonic(), ceiling):
                    bus.publish(event)
            except Exception:
                logger.exception("provisional expiry tick failed")

    threading.Thread(target=tail, name="partial-tail", daemon=True).start()
    threading.Thread(target=expire, name="partial-expiry", daemon=True).start()


def _start_worker(args: argparse.Namespace, path: Path, state: DashboardState,
                  bus: EventBus, bundle: ContextBundle | None) -> threading.Thread:
    sink = make_sink(state, bus, suggestion_log=SuggestionLog(suggestions_log_path(path)))

    def work() -> None:
        try:
            if args.no_suggestions or bundle is None:
                tail_only(path, state, bus, from_start=not args.from_end)
                return
            gate = SalienceGate(bundle, is_question=looks_like_question,
                                enabled=not args.no_salience)
            run_ambient(
                bundle=bundle, path=path, backend=args.backend or "local",
                model=args.model or "", suggestion_language=args.suggestion_language,
                from_start=not args.from_end, answer_speaker=args.answer_speaker,
                gate=gate, sink=sink,
            )
        except Exception:
            logger.exception("the transcript worker died — the dashboard will go stale")
            bus.publish({"kind": "worker_dead"})

    thread = threading.Thread(target=work, name="transcript-worker", daemon=True)
    thread.start()
    return thread


SCRIPTS_DIR = Path(__file__).resolve().parent


class ControlError(Exception):
    """A rejected switch (e.g. cloud asked for but not configured) — surfaced to the UI as 400."""


class AppController:
    """#607 single-app mode: owns the recorder subprocess AND the transcript worker threads so
    the browser's three switches drive real state.

      transcription on/off  → spawn / SIGINT the live_transcribe.py child (a TRUE capture stop:
                              mic released, ~2.2 GB Whisper VRAM freed — not a paused display)
      suggestions  on/off   → Controls.suggestions_on; the ambient loop reads it live, so the
                              transcript + meter keep flowing while no model/GPU is touched
      backend local / +api  → Controls.backend: local-only Ollama, or the BYOK cloud egress
                              (opt-in, announced by llm_client — SI1/D14)

    Not a governance seat (fw:D2): it starts/stops processes and flips flags; it decides nothing
    about the interview. D19 holds — the worker still only TAILS the transcript file; the
    controller manages the recorder lifecycle out of band and never writes the transcript.
    """

    def __init__(self, args: argparse.Namespace, state: DashboardState, bus: EventBus,
                 bundle: ContextBundle, ceiling: float, partials_on: bool) -> None:
        self.args = args
        self.state = state
        self.bus = bus
        self.bundle = bundle
        self.ceiling = ceiling
        self.partials_on = partials_on
        self.cloud_available = cloud_ready()
        self.controls = Controls(suggestions_on=True, backend=args.backend or "local",
                                  model=args.model or "")
        self._lock = threading.RLock()
        self._recorder: subprocess.Popen | None = None
        self._worker_stop: threading.Event | None = None
        self._current_path: Path | None = None
        self.transcription_on = False

    # -- state the UI renders --------------------------------------------------
    def ui_state(self) -> dict:
        suggestions_on, backend, _ = self.controls.snapshot()
        return {
            "transcription": self.transcription_on,
            "suggestions": suggestions_on,
            "backend": backend,                     # "local" | "cloud"
            "cloud_available": self.cloud_available,
            "source": self._current_path.name if self._current_path else None,
        }

    def _broadcast(self) -> dict:
        """Push the new switch state to every open tab and return it for the POST response."""
        st = self.ui_state()
        self.bus.publish({"kind": "controls", **st})
        return st

    # -- transcription: spawn / stop the recorder child ------------------------
    def start_transcription(self) -> dict:
        with self._lock:
            if self.transcription_on:
                return self.ui_state()
            path = self._spawn_recorder()          # raises ControlError on failure
            stop = threading.Event()
            self._worker_stop = stop
            self._current_path = path
            self.transcription_on = True
            self._start_workers(path, stop)
            logger.info("transcription ON — recorder pid=%s, following %s",
                        self._recorder.pid if self._recorder else "?", path.name)
            return self._broadcast()

    def stop_transcription(self) -> dict:
        with self._lock:
            if not self.transcription_on:
                return self.ui_state()
            if self._worker_stop is not None:
                self._worker_stop.set()            # ends the follow loops within a poll interval
            self._stop_recorder()                  # SIGINT → wait → kill (we hold the handle)
            self.transcription_on = False
            self._current_path = None
            logger.info("transcription OFF — recorder stopped, capture released")
            return self._broadcast()

    def _spawn_recorder(self) -> Path:
        before = {p.name for p in OUTPUT_DIR.glob("live_transcript_2*.txt")}
        cmd = [sys.executable, str(SCRIPTS_DIR / "live_transcribe.py"),
               "--source", settings.COPILOT_SOURCE, "--mic", settings.COPILOT_MIC]
        if not self.partials_on:
            cmd.append("--no-partials")
        try:
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            raise ControlError(f"could not launch the recorder: {exc}") from exc
        self._recorder = proc
        deadline = time.monotonic() + max(1.0, settings.COPILOT_RECORDER_WAIT_SECONDS)
        while time.monotonic() < deadline:
            for p in sorted(OUTPUT_DIR.glob("live_transcript_2*.txt")):
                if p.name not in before:
                    return p
            if proc.poll() is not None:            # it died before writing a transcript
                self._recorder = None
                raise ControlError(
                    "the recorder exited before capturing audio — check the mic/monitor source "
                    "(COPILOT_MIC / COPILOT_SOURCE) and see logs/live_transcribe_*.log"
                )
            time.sleep(0.25)
        self._stop_recorder()
        raise ControlError(
            f"the recorder wrote no transcript within {settings.COPILOT_RECORDER_WAIT_SECONDS:g}s "
            "— is a capture source available? (COPILOT_MIC / COPILOT_SOURCE)"
        )

    def _stop_recorder(self) -> None:
        proc, self._recorder = self._recorder, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.send_signal(signal.SIGINT)        # clean shutdown: flushes RUN COMPLETE + drift
            proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                logger.exception("recorder did not die on kill()")

    def _start_workers(self, path: Path, stop: threading.Event) -> None:
        sink = make_sink(self.state, self.bus,
                         suggestion_log=SuggestionLog(suggestions_log_path(path)))

        def work() -> None:
            try:
                gate = SalienceGate(self.bundle, is_question=looks_like_question,
                                    enabled=not self.args.no_salience)
                backend, model = self.controls.backend, self.controls.model
                run_ambient(
                    bundle=self.bundle, path=path, backend=backend, model=model,
                    suggestion_language=self.args.suggestion_language,
                    from_start=not self.args.from_end, answer_speaker=self.args.answer_speaker,
                    gate=gate, sink=sink, controls=self.controls, stop_event=stop,
                )
            except Exception:
                logger.exception("the transcript worker died — the dashboard will go stale")
                self.bus.publish({"kind": "worker_dead"})

        threading.Thread(target=work, name="transcript-worker", daemon=True).start()
        if self.partials_on:
            _start_provisional(path, self.state, self.bus, self.ceiling, stop_event=stop)

    # -- suggestions / backend -------------------------------------------------
    def set_suggestions(self, on: bool) -> dict:
        with self._lock:
            self.controls.set_suggestions(on)
            logger.info("suggestions %s", "ON" if on else "OFF")
            return self._broadcast()

    def set_backend(self, value: str) -> tuple[dict, str]:
        value = (value or "").strip().lower()
        if value not in ("local", "cloud"):
            raise ControlError(f"backend must be 'local' or 'cloud', got {value!r}")
        if value == "cloud" and not self.cloud_available:
            raise ControlError(
                "'local + api' is not configured: set CLOUD_MODEL and CLOUD_API_KEY in config/.env "
                "(BYOK). Staying local-only keeps the transcript on this machine (SI1)."
            )
        with self._lock:
            self.controls.set_backend(value, model_for(value))
            banner = announce_backend(value, model_for(value))
            logger.info("suggestion backend → %s: %s", value, banner)
            return self._broadcast(), banner

    def shutdown(self) -> None:
        try:
            self.stop_transcription()
        except Exception:
            logger.exception("shutdown: stop_transcription raised")


def _run_app(args: argparse.Namespace) -> None:
    """Single-app mode (#607): the one process the desktop shortcut launches. It owns the
    recorder and serves the switches. Transcription starts OFF — the user flips it on when the
    call begins, which is also the privacy default (no capture until asked)."""
    if not args.session:
        raise SystemExit("--app requires --session (the interview context bundle)")
    bundle = load_bundle(args.session)
    for line in bundle.warn_lines():
        print(f"  {line}", flush=True)

    ceiling = meter_ceiling(override_seconds=args.ceiling_seconds or None)
    partials_on = settings.DASHBOARD_SHOW_PARTIALS if args.partials is None else args.partials
    state = DashboardState(plan=bundle.plan)
    bus = EventBus()
    controller = AppController(args, state, bus, bundle, ceiling, partials_on)
    info = RunInfo(
        source="— (transcription off)",
        mode=args.mode or "live (app)",
        session=args.session,
        salience="off (ungated)" if args.no_salience else "on (D23)",
        ceiling_seconds=ceiling,
        suggestions=True,
        provisional=("on (D25)" if partials_on else "off"),
    )

    url = f"http://{args.host}:{args.port}"
    print(f"interview copilot: {url}   (D18/SI1 — loopback only)", flush=True)
    print("  switches: transcription (OFF now) · suggestions · backend "
          f"({'local + api available' if controller.cloud_available else 'local only — cloud not configured'})",
          flush=True)
    print(f"  devices : source={settings.COPILOT_SOURCE}  mic={settings.COPILOT_MIC}"
          + ("   (set COPILOT_MIC to your external mic for a real call)" if settings.COPILOT_MIC == "auto" else ""),
          flush=True)
    logger.info("app mode starting on %s:%s session=%s cloud=%s",
                args.host, args.port, args.session, controller.cloud_available)

    if settings.COPILOT_OPEN_BROWSER:
        threading.Thread(target=lambda: (time.sleep(1.0), _open_browser(url)),
                         name="open-browser", daemon=True).start()

    import uvicorn
    app = create_app(state, bus, info, ceiling, controller=controller)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        controller.shutdown()            # SIGINT the recorder on exit — never orphan capture


def _open_browser(url: str) -> None:
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        logger.info("could not open a browser automatically — open %s yourself", url)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--session", help="session id under scripts/inputs/sessions/ (or a path)")
    parser.add_argument("--watch", action="store_true", help="follow the newest transcript in scripts/outputs/")
    parser.add_argument("--follow", help="follow this transcript file as it grows")
    parser.add_argument("--wait-seconds", type=float, default=0.0, help="with --watch: wait this long for a transcript to appear")
    parser.add_argument("--from-end", action="store_true", help="start at the end of the file rather than replaying it")
    parser.add_argument("--no-suggestions", action="store_true", help="transcript + meter only: no model, no GPU, no suggestion call")
    parser.add_argument("--no-partials", dest="partials", action="store_false", default=None, help="do not render D25 provisional lines (default: DASHBOARD_SHOW_PARTIALS)")
    parser.add_argument("--no-salience", action="store_true", help="disable the D23 gate; every detected question fires (the ungated comparison)")
    parser.add_argument("--backend", choices=("local", "cloud"), help="override REASONING_BACKEND")
    parser.add_argument("--model", help="override the backend's model")
    parser.add_argument("--suggestion-language", choices=("match", "en", "pl"), help="override SUGGESTION_LANGUAGE")
    parser.add_argument("--answer-speaker", choices=("them", "you", "any"), help="whose turns to answer (default: settings.ANSWER_SPEAKER)")
    parser.add_argument("--host", default=settings.DASHBOARD_HOST, help="loopback only (D18/SI1)")
    parser.add_argument("--port", type=int, default=settings.DASHBOARD_PORT)
    parser.add_argument("--ceiling-seconds", type=float, default=0.0,
                        help="override the meter ceiling; use ceiling/N for a --speed N replay")
    parser.add_argument("--mode", default="", help="provenance label shown in the UI (live | replay)")
    parser.add_argument("--app", action="store_true",
                        help="single-app mode (#607): the dashboard spawns/kills the recorder itself "
                             "and exposes the three on-screen switches. This is what the desktop "
                             "shortcut launches. Implies a session; no --watch/--follow needed.")
    args = parser.parse_args()

    _assert_loopback(args.host)          # before anything else opens or reads

    if args.app:
        _run_app(args)
        return

    if not (args.watch or args.follow):
        parser.error("choose a transcript: --watch (newest) or --follow FILE")
    path = _resolve_transcript(args)

    bundle: ContextBundle | None = None
    if not args.no_suggestions:
        if not args.session:
            parser.error("--session is required unless --no-suggestions is given")
        bundle = load_bundle(args.session)
        for line in bundle.warn_lines():
            print(f"  {line}", flush=True)

    ceiling = meter_ceiling(override_seconds=args.ceiling_seconds or None)
    partials_on = settings.DASHBOARD_SHOW_PARTIALS if args.partials is None else args.partials
    state = DashboardState(plan=bundle.plan if bundle else ())
    bus = EventBus()
    info = RunInfo(
        source=path.name,
        mode=detect_mode(path, args.mode or ("tail-only" if args.no_suggestions else "live")),
        session=args.session or "-",
        salience="off (ungated)" if args.no_salience else "on (D23)",
        ceiling_seconds=ceiling,
        suggestions=not args.no_suggestions,
        provisional=("on (D25)" if partials_on else "off"),
    )

    print(f"dashboard: http://{args.host}:{args.port}   (D18/SI1 — loopback only)", flush=True)
    print(f"  transcript : {path}", flush=True)
    print(f"  meter      : ceiling {ceiling:g}s "
          f"(SEGMENT_MAX_SECONDS {settings.SEGMENT_MAX_SECONDS:g}s + decode allowance "
          f"{settings.METER_DECODE_ALLOWANCE_SECONDS:g}s)", flush=True)
    print(f"  salience   : {info.salience}", flush=True)
    print(f"  provisional: {info.provisional}"
          + (f" — {partial_path_for(path).name}" if partials_on else ""), flush=True)
    logger.info("dashboard starting on %s:%s source=%s mode=%s ceiling=%.2fs salience=%s",
                args.host, args.port, path.name, info.mode, ceiling, info.salience)

    _start_worker(args, path, state, bus, bundle)
    if partials_on:
        _start_provisional(path, state, bus, ceiling)

    import uvicorn
    app = create_app(state, bus, info, ceiling)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
