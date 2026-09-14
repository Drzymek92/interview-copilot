"""Tests for the D18 dashboard: the P5 meter's time math, and the tail -> render path.

Two halves, both deterministic — no model, no GPU, no browser, no network:

1. **`meter_state()` as a pure function.** It takes the clock as an argument, so every state
   and every boundary is a table row. The point of the tests is not that it formats a string;
   it is that the meter cannot promise something it has already missed — the rounding is
   asserted in the honest direction, and the overrun state is asserted against the real call's
   own numbers.

2. **The tail -> render path.** A transcript file is written line by line the way
   `live_transcribe.py` writes it, and the assertions are on what the browser would be sent:
   the D19 tail through `follow_transcript`, into `DashboardState`, out as websocket events.
   `run_ambient`'s own sink shape is exercised too, so the two cannot drift apart silently.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from scripts import dashboard as dash  # noqa: E402
from scripts import replay_transcript as rp  # noqa: E402
from scripts.reasoning import PlanStep  # noqa: E402

REAL_TRANSCRIPT = PROJECT_ROOT / "scripts" / "outputs" / "live_transcript_20260902_100033.txt"


# ==========================================================================
# 1. The meter — pure time math
# ==========================================================================
class TestMeterState:
    def test_cold_promises_nothing_before_the_first_line(self):
        """No line has arrived, so there is no interval. It must not invent one."""
        state = dash.meter_state(last_line_at=None, now=1000.0, ceiling_seconds=31.0)
        assert state.state == "cold"
        assert state.since_seconds is None
        assert state.due_within_seconds is None
        assert state.over_by_seconds is None
        assert "≤" not in state.label  # no promise of any kind

    @pytest.mark.parametrize(
        "elapsed, since, due",
        [
            (0.0, 0, 31),
            (14.0, 14, 17),
            (14.6, 14, 17),    # age floors, remaining ceils - see below
            (30.4, 30, 1),
            (31.0, 31, 0),     # exactly at the ceiling is still a promise, of zero
        ],
    )
    def test_waiting_states(self, elapsed, since, due):
        state = dash.meter_state(1000.0, 1000.0 + elapsed, 31.0)
        assert state.state == "waiting"
        assert state.since_seconds == since
        assert state.due_within_seconds == due
        assert state.label == f"last line {since} s ago · next due within ≤{due} s"

    def test_rounding_never_flatters_the_promise(self):
        """The bound must be one the arrival can only beat, never miss by a rounding artefact.

        At 14.6 s elapsed the true remaining time is 16.4 s. Rounding that to 16 would promise
        a line 0.4 s sooner than the ceiling actually allows; ceiling it to 17 cannot.
        """
        state = dash.meter_state(0.0, 14.6, 31.0)
        assert state.due_within_seconds == 17
        assert state.due_within_seconds >= 31.0 - 14.6

    def test_the_promise_plus_the_age_always_covers_the_ceiling(self):
        """The invariant behind the whole panel, over the entire waiting range."""
        for tick in range(0, 3100):
            elapsed = tick / 100.0
            state = dash.meter_state(0.0, elapsed, 31.0)
            if state.state != "waiting":
                continue
            assert state.since_seconds + state.due_within_seconds >= 31.0 - 1e-9
            assert state.since_seconds <= elapsed

    def test_overdue_withdraws_the_promise(self):
        state = dash.meter_state(1000.0, 1000.0 + 35.0, 31.0)
        assert state.state == "overdue"
        assert state.due_within_seconds is None      # nothing is promised any more
        assert state.over_by_seconds == 4
        assert "overdue" in state.label
        assert "≤" not in state.label

    def test_the_boundary_is_strict(self):
        """At the ceiling the promise still stands; one instant past it, it does not."""
        assert dash.meter_state(0.0, 31.0, 31.0).state == "waiting"
        assert dash.meter_state(0.0, 31.001, 31.0).state == "overdue"

    def test_overdue_over_by_is_never_understated(self):
        state = dash.meter_state(0.0, 31.2, 31.0)
        assert state.over_by_seconds == 1            # ceil(0.2) - never 0
        assert state.over_by_seconds >= 31.2 - 31.0

    def test_the_real_calls_one_true_overrun_renders_as_overdue(self):
        """The 43 s gap measured on the 42-min HR call (a silence between segments)."""
        state = dash.meter_state(0.0, 43.2, 31.0)
        assert state.state == "overdue"
        assert state.since_seconds == 43
        assert state.over_by_seconds == 13

    def test_a_clock_that_goes_backwards_does_not_produce_a_negative_age(self):
        state = dash.meter_state(1000.0, 999.0, 31.0)
        assert state.since_seconds == 0
        assert state.state == "waiting"

    def test_a_zero_ceiling_is_immediately_overdue_not_a_divide_by_zero(self):
        assert dash.meter_state(0.0, 0.0, 0.0).state == "waiting"
        assert dash.meter_state(0.0, 0.5, 0.0).state == "overdue"

    def test_state_is_immutable(self):
        """The UI is pushed this object; nothing downstream may quietly edit the number."""
        state = dash.meter_state(0.0, 5.0, 31.0)
        with pytest.raises(Exception):
            state.since_seconds = 99  # type: ignore[misc]


class TestProvisionalAwareMeter:
    """#457: suppress `overdue` while a D25 provisional fresher than the last final is on
    screen, instead of widening the 31.0 s ceiling (D27). `meter_state` stays pure — the
    freshness fact is passed in as `provisional_fresh` — and `tick()` is the only place that
    reads `DashboardState` to compute it."""

    def test_meter_state_a_fresh_provisional_suppresses_overdue(self):
        """(a) Past the ceiling, with a fresh provisional: no `overdue`, and no promise is
        resurrected either — `due_within_seconds` stays None, this is not a new bound."""
        state = dash.meter_state(0.0, 35.0, 31.0, provisional_fresh=True)
        assert state.state == "waiting"
        assert state.since_seconds == 35
        assert state.due_within_seconds is None
        assert state.over_by_seconds is None
        assert "overdue" not in state.label
        assert "≤" not in state.label      # not a restated promise

    def test_meter_state_without_a_fresh_provisional_stays_overdue(self):
        """Baseline: `provisional_fresh` defaults to False, so nothing changes for a caller
        that doesn't know about D25 at all."""
        assert dash.meter_state(0.0, 35.0, 31.0).state == "overdue"
        assert dash.meter_state(0.0, 35.0, 31.0, provisional_fresh=False).state == "overdue"

    def test_meter_state_provisional_fresh_does_not_affect_waiting_or_cold(self):
        """The flag only ever matters once the ceiling has already been passed."""
        waiting = dash.meter_state(0.0, 14.0, 31.0, provisional_fresh=True)
        assert waiting.state == "waiting"
        assert waiting.due_within_seconds == 17          # unchanged from the non-suppressed path
        assert dash.meter_state(None, 1000.0, 31.0, provisional_fresh=True).state == "cold"

    def test_tick_suppresses_overdue_when_a_fresh_provisional_is_on_screen(self):
        """(a) end-to-end through `DashboardState.tick()`: a final lands, then a provisional
        for a LATER (still-open) segment, then the clock passes the ceiling."""
        state = dash.DashboardState()
        state.add_line("00:00-00:05", "them", None, "hello", True, at=0.0)
        state.add_provisional("00:10-00:20", "them", None, "still talking", True, at=5.0)

        result = state.tick(now=35.0, ceiling=31.0)
        assert result["meter"]["state"] == "waiting"
        assert result["meter"]["due_within_seconds"] is None

    def test_tick_does_not_suppress_a_provisional_older_than_the_last_final(self):
        """(b) A provisional for a segment that already has a final does NOT suppress overdue.
        `add_provisional` already refuses such a candidate outright (#399: its final beat it
        here), so the slot is empty and `tick` has nothing fresh to point to."""
        state = dash.DashboardState()
        state.add_line("00:10-00:20", "them", None, "already finalized", True, at=0.0)
        events = state.add_provisional("00:00-00:05", "them", None, "stale", True, at=5.0)

        assert events == []
        assert state.provisional is None
        result = state.tick(now=35.0, ceiling=31.0)
        assert result["meter"]["state"] == "overdue"

    def test_last_line_at_is_unchanged_by_suppression(self):
        """(c) A provisional is not an arrival (D25/P5): it must not reset `last_line_at`,
        whether or not it goes on to suppress `overdue` in a later tick."""
        state = dash.DashboardState()
        state.add_line("00:00-00:05", "them", None, "hello", True, at=0.0)
        before = state.last_line_at
        state.add_provisional("00:10-00:20", "them", None, "still talking", True, at=5.0)
        assert state.last_line_at == before == 0.0

        result = state.tick(now=35.0, ceiling=31.0)
        assert result["meter"]["state"] == "waiting"      # suppression did fire
        assert state.last_line_at == before == 0.0         # and still did not move the clock

    def test_genuine_silence_is_still_overdue_with_no_provisional_on_screen(self):
        """(d) The 43.2 s genuine inter-segment silence (the real call's one true overrun) must
        still be reported `overdue` — a real stall is not something suppression may hide."""
        state = dash.DashboardState()
        state.add_line("00:00-00:05", "them", None, "hello", True, at=0.0)
        result = state.tick(now=43.2, ceiling=31.0)
        assert result["meter"]["state"] == "overdue"
        assert result["meter"]["over_by_seconds"] == 13

    def test_an_expired_provisional_no_longer_suppresses(self):
        """A provisional that `expire_provisional` has already cleared (the recorder went away)
        is gone from the slot, so it cannot go on suppressing `overdue` forever."""
        state = dash.DashboardState()
        state.add_line("00:00-00:05", "them", None, "hello", True, at=0.0)
        state.add_provisional("00:10-00:20", "them", None, "still talking", True, at=0.0)
        state.expire_provisional(now=32.0, ceiling=31.0)     # older than the ceiling -> cleared
        assert state.provisional is None

        result = state.tick(now=35.0, ceiling=31.0)
        assert result["meter"]["state"] == "overdue"


class TestMeterCeiling:
    def test_default_is_the_segment_cap_plus_the_decode_allowance(self):
        assert dash.meter_ceiling(override_seconds=None) == pytest.approx(
            settings.SEGMENT_MAX_SECONDS + settings.METER_DECODE_ALLOWANCE_SECONDS
        )

    def test_override_wins(self):
        """The one legitimate use: keeping a time-compressed replay truthful."""
        assert dash.meter_ceiling(override_seconds=3.875) == pytest.approx(3.875)

    def test_explicit_components(self):
        assert dash.meter_ceiling(segment_max_seconds=20.0, decode_allowance_seconds=1.0,
                                  override_seconds=None) == pytest.approx(21.0)

    @pytest.mark.skipif(not REAL_TRANSCRIPT.exists(), reason="the real HR transcript is not present")
    def test_the_ceiling_is_the_one_the_real_call_supports(self):
        """The measurement that made METER_DECODE_ALLOWANCE_SECONDS exist.

        A line reaches the screen when its segment closes AND is decoded. Scored against the
        109 arrival gaps of the real 42-min call, a bare SEGMENT_MAX_SECONDS=30 ceiling is
        breached 25 times — because 54% of segments end exactly at the cap and then take
        decode on top. The shipped ceiling must hold for all but the one true overrun (43.2 s,
        a silence between segments, which no segment cap bounds).
        """
        lines = rp.parse_replay_lines(REAL_TRANSCRIPT.read_text(encoding="utf-8"))
        gaps = [b.arrival - a.arrival for a, b in zip(lines, lines[1:])]
        assert len(gaps) == 109

        bare = sum(1 for gap in gaps if gap > settings.SEGMENT_MAX_SECONDS)
        assert bare == 25, "a bare 30 s ceiling is not the honest one - see the module docstring"

        shipped = dash.meter_ceiling(override_seconds=None)
        overruns = [gap for gap in gaps if gap > shipped]
        assert len(overruns) == 1
        assert overruns[0] == pytest.approx(43.2, abs=0.1)

        # ... and the meter must call that one what it is.
        assert dash.meter_state(0.0, overruns[0], shipped).state == "overdue"


# ==========================================================================
# 2. The tail -> render path
# ==========================================================================
class Recorder:
    """Stands in for the websocket fan-out: keeps every event the UI would receive."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish(self, event: dict) -> None:
        json.dumps(event)      # every event must survive the wire, not just exist
        self.events.append(event)

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e.get("kind") == kind]


@pytest.fixture
def transcript(tmp_path: Path) -> Path:
    path = tmp_path / "live_transcript_20260904_120000.txt"
    path.write_text("# interview_copilot live transcript\n# model : large-v3-turbo\n",
                    encoding="utf-8")
    return path


def append(path: Path, line: str) -> None:
    """Write one line the way live_transcribe.py does — fsync'd, so the tail can see it."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class TestTailToRender:
    def test_a_line_written_to_the_file_reaches_the_ui_two_sided(self, transcript):
        append(transcript, "[00:53-01:06] them (pl): Dzień dobry, czy dobrze mnie słychać?")
        append(transcript, "[01:06-01:18] you (pl): Tak, słyszę Pana bardzo dobrze.")

        state = dash.DashboardState()
        bus = Recorder()
        assert dash.tail_only(transcript, state, bus, stop_after_idle=0.4) == 2

        lines = bus.of("line")
        assert [e["line"]["speaker"] for e in lines] == ["them", "you"]
        assert all(e["line"]["tagged"] for e in lines)
        assert lines[0]["line"]["text"].startswith("Dzień dobry")
        assert lines[0]["line"]["language"] == "pl"
        assert state.counters.them == 1 and state.counters.you == 1

    def test_header_and_blank_lines_never_reach_the_ui(self, transcript):
        append(transcript, "")
        append(transcript, "# a comment the recorder wrote mid-run")
        append(transcript, "[00:01-00:05] them: real")
        state, bus = dash.DashboardState(), Recorder()
        dash.tail_only(transcript, state, bus, stop_after_idle=0.4)
        assert len(bus.of("line")) == 1

    def test_an_untagged_line_is_rendered_as_untagged(self, transcript):
        """A pre-#326 transcript is treated as the interviewer downstream, but the screen must
        not present that inference as an observation. `tagged` is what the UI branches on."""
        append(transcript, "[00:10-00:20] a line with no speaker prefix")
        state, bus = dash.DashboardState(), Recorder()
        dash.tail_only(transcript, state, bus, stop_after_idle=0.4)
        event = bus.of("line")[0]["line"]
        assert event["speaker"] == "them"
        assert event["tagged"] is False
        assert state.counters.untagged == 1
        assert state.counters.them == 0

    def test_the_meter_moves_when_a_line_lands(self, transcript):
        """The tail is what feeds the clock: no line, no interval to measure."""
        state, bus = dash.DashboardState(), Recorder()
        assert state.tick(now=100.0, ceiling=31.0)["meter"]["state"] == "cold"

        append(transcript, "[00:01-00:05] them: hello")
        clock = iter([500.0])
        dash.tail_only(transcript, state, bus, stop_after_idle=0.4, clock=lambda: next(clock))

        assert state.tick(now=510.0, ceiling=31.0)["meter"]["since_seconds"] == 10
        assert state.tick(now=510.0, ceiling=31.0)["meter"]["state"] == "waiting"
        assert state.tick(now=545.0, ceiling=31.0)["meter"]["state"] == "overdue"

    def test_late_joining_browser_is_sent_the_whole_call(self, transcript):
        for i in range(5):
            append(transcript, f"[00:{i:02d}-00:{i + 1:02d}] them: line {i}")
        state, bus = dash.DashboardState(), Recorder()
        dash.tail_only(transcript, state, bus, stop_after_idle=0.4)
        snap = state.snapshot()
        assert len(snap["lines"]) == 5
        assert snap["counters"]["lines"] == 5
        assert snap["dropped_lines"] == 0

    def test_the_ring_buffer_reports_what_it_dropped(self, transcript):
        """A truncated call must be visibly truncated, not silently short."""
        state = dash.DashboardState(max_lines=3)
        bus = Recorder()
        for i in range(6):
            append(transcript, f"[00:{i:02d}-00:{i + 1:02d}] them: line {i}")
        dash.tail_only(transcript, state, bus, stop_after_idle=0.4)
        snap = state.snapshot()
        assert len(snap["lines"]) == 3
        assert snap["dropped_lines"] == 3
        assert snap["counters"]["lines"] == 6

    def test_a_line_arriving_while_the_tail_is_attached_is_pushed(self, transcript):
        """The live case, not the replay one: the file grows under an attached reader."""
        state, bus = dash.DashboardState(), Recorder()
        thread = threading.Thread(
            target=dash.tail_only, args=(transcript, state, bus),
            kwargs={"stop_after_idle": 1.5}, daemon=True,
        )
        thread.start()
        time.sleep(0.4)
        append(transcript, "[00:01-00:05] them: arrived after the reader attached")
        thread.join(timeout=6)
        assert not thread.is_alive()
        assert len(bus.of("line")) == 1


class TestSinkContract:
    """`make_sink` is the only place that knows both the ambient loop and the websocket."""

    def test_gated_questions_produce_a_counter_and_no_suggestion(self):
        """D23: a dropped question shows as a number, never as a non-answer on screen."""
        state, bus = dash.DashboardState(), Recorder()
        sink = dash.make_sink(state, bus)
        sink({"kind": "line", "stamp": "00:01-00:05", "speaker": "them", "language": "pl",
              "text": "Czy mnie słychać?", "tagged": True})
        sink({"kind": "gate", "stamp": "00:01-00:05", "fire": False, "detail": "salience NO"})

        assert state.counters.triggered == 1
        assert state.counters.dropped == 1
        assert state.suggestion is None
        assert bus.of("suggestion") == []

    def test_a_cooldown_skip_is_not_reported_as_a_gate_drop(self):
        """Found by running the real call with `--no-salience`: the panel said "2 dropped"
        while the D23 gate was switched OFF. The cooldown is a different reason for a blank
        panel and gets its own counter, or the screen describes a gate that is not running."""
        state, bus = dash.DashboardState(), Recorder()
        sink = dash.make_sink(state, bus)
        sink({"kind": "skip", "stamp": "01:00-01:30", "reason": "cooldown",
              "detail": "2.0s into the 8s cooldown"})

        assert state.counters.cooldown == 1
        assert state.counters.dropped == 0
        assert state.counters.triggered == 0     # the gate never judged it
        assert state.suggestion is None

    def test_a_fired_suggestion_streams_and_completes(self):
        state, bus = dash.DashboardState(), Recorder()
        sink = dash.make_sink(state, bus)
        sink({"kind": "gate", "stamp": "01:00-01:30", "fire": True, "detail": "salience YES"})
        sink({"kind": "suggestion", "phase": "start", "id": 1, "stamp": "01:00-01:30",
              "segment": "Dlaczego chce Pan zmienić pracę?", "language": "en",
              "fired_because": "question"})
        sink({"kind": "suggestion", "phase": "token", "id": 1, "text": "Because "})
        sink({"kind": "suggestion", "phase": "token", "id": 1, "text": "the role fits."})
        sink({"kind": "suggestion", "phase": "end", "id": 1, "status": "complete",
              "note": "312 tok"})

        assert state.counters.fired == 1
        assert state.suggestion.text == "Because the role fits."
        assert state.suggestion.status == "complete"
        assert state.suggestion.stamp == "01:00-01:30"

    def test_a_superseded_streams_tail_cannot_leak_into_its_successor(self):
        """The runner cancels cooperatively, so a stray token can arrive after the swap.
        Appending it to the new answer would put words in the copilot's mouth."""
        state, bus = dash.DashboardState(), Recorder()
        sink = dash.make_sink(state, bus)
        sink({"kind": "suggestion", "phase": "start", "id": 1, "segment": "old", "language": "en",
              "fired_because": "question"})
        sink({"kind": "suggestion", "phase": "token", "id": 1, "text": "answering the old "})
        sink({"kind": "suggestion", "phase": "start", "id": 2, "segment": "new", "language": "en",
              "fired_because": "question"})
        sink({"kind": "suggestion", "phase": "token", "id": 1, "text": "STRAY"})
        sink({"kind": "suggestion", "phase": "token", "id": 2, "text": "answering the new"})

        assert state.suggestion.id == 2
        assert state.suggestion.text == "answering the new"
        assert "STRAY" not in state.suggestion.text

    def test_a_finished_suggestion_goes_stale_once_the_call_moves_on(self):
        """A confident answer to a question three turns ago is worse than a blank panel."""
        state, bus = dash.DashboardState(), Recorder()
        sink = dash.make_sink(state, bus)
        sink({"kind": "suggestion", "phase": "start", "id": 1, "segment": "q", "language": "en",
              "fired_because": "question"})
        sink({"kind": "suggestion", "phase": "end", "id": 1, "status": "complete", "note": ""})
        assert state.tick(now=0.0, ceiling=31.0)["suggestion_age"]["stale"] is False

        for i in range(settings.SUGGESTION_STALE_LINES):
            sink({"kind": "line", "stamp": f"0{i}:00-0{i}:30", "speaker": "them",
                  "language": "pl", "text": "a later turn", "tagged": True})
        age = state.tick(now=0.0, ceiling=31.0)["suggestion_age"]
        assert age["stale"] is True
        assert age["lines_since"] == settings.SUGGESTION_STALE_LINES

    def test_a_streaming_suggestion_is_never_marked_stale(self):
        """It is answering the question that is being asked right now, by definition."""
        state, bus = dash.DashboardState(), Recorder()
        sink = dash.make_sink(state, bus)
        sink({"kind": "suggestion", "phase": "start", "id": 1, "segment": "q", "language": "en",
              "fired_because": "question"})
        for i in range(5):
            sink({"kind": "line", "stamp": f"0{i}:00-0{i}:30", "speaker": "them",
                  "language": "pl", "text": "later", "tagged": True})
        assert state.tick(now=0.0, ceiling=31.0)["suggestion_age"]["stale"] is False

    def test_a_failed_suggestion_says_so_rather_than_showing_nothing(self):
        state, bus = dash.DashboardState(), Recorder()
        sink = dash.make_sink(state, bus)
        sink({"kind": "suggestion", "phase": "start", "id": 1, "segment": "q", "language": "en",
              "fired_because": "question"})
        sink({"kind": "suggestion", "phase": "end", "id": 1, "status": "failed",
              "note": "suggestion failed — see the log"})
        assert state.suggestion.status == "failed"
        assert "failed" in state.suggestion.note


class TestSuggestionHistory:
    """Return to a previous suggestion mid-call: the live slot holds the current answer, and a
    displaced one moves into a capped scrollback that is also persisted for post-call review."""

    def _fire(self, sink, sid, segment, text, status="complete"):
        sink({"kind": "suggestion", "phase": "start", "id": sid,
              "stamp": f"0{sid}:00-0{sid}:30", "segment": segment, "language": "pl",
              "fired_because": "question"})
        sink({"kind": "suggestion", "phase": "token", "id": sid, "text": text})
        sink({"kind": "suggestion", "phase": "end", "id": sid, "status": status, "note": ""})

    def test_a_completed_suggestion_moves_to_history_when_the_next_one_starts(self):
        state, bus = dash.DashboardState(), Recorder()
        sink = dash.make_sink(state, bus)
        self._fire(sink, 1, "pierwsze pytanie", "POINT: pierwsza odpowiedź")
        assert state.history == []                      # still live, not yet displaced
        assert state.suggestion.id == 1

        self._fire(sink, 2, "drugie pytanie", "POINT: druga odpowiedź")
        assert [h.id for h in state.history] == [1]
        assert state.history[0].status == "complete"    # it had finished before being displaced
        assert state.history[0].text == "POINT: pierwsza odpowiedź"
        assert state.suggestion.id == 2                 # the current one stays in the live slot

    def test_a_suggestion_still_streaming_when_displaced_is_archived_as_superseded(self):
        state, bus = dash.DashboardState(), Recorder()
        sink = dash.make_sink(state, bus)
        sink({"kind": "suggestion", "phase": "start", "id": 1, "stamp": "01:00-01:30",
              "segment": "q1", "language": "pl", "fired_because": "question"})
        sink({"kind": "suggestion", "phase": "token", "id": 1, "text": "half an ans"})
        # a new question arrives before the first answer finished
        sink({"kind": "suggestion", "phase": "start", "id": 2, "stamp": "02:00-02:30",
              "segment": "q2", "language": "pl", "fired_because": "question"})
        assert [h.id for h in state.history] == [1]
        assert state.history[0].status == "superseded"
        assert state.history[0].text == "half an ans"

    def test_history_is_broadcast_and_in_the_snapshot(self):
        state, bus = dash.DashboardState(), Recorder()
        sink = dash.make_sink(state, bus)
        self._fire(sink, 1, "q1", "POINT: a")
        self._fire(sink, 2, "q2", "POINT: b")
        # the UI is told the new scrollback live...
        assert bus.of("history")[-1]["history"][0]["segment"] == "q1"
        # ...and a fresh page load gets the same thing in the snapshot
        assert [h["id"] for h in state.snapshot()["history"]] == [1]

    def test_history_respects_the_cap(self, monkeypatch):
        monkeypatch.setattr(settings, "SUGGESTION_HISTORY_MAX", 3)
        state, bus = dash.DashboardState(), Recorder()
        sink = dash.make_sink(state, bus)
        for sid in range(1, 7):
            self._fire(sink, sid, f"q{sid}", f"POINT: {sid}")
        # 6 fired, the newest is live, the previous 5 want to be history but the cap holds 3
        assert len(state.history) == 3
        assert [h.id for h in state.history] == [3, 4, 5]   # oldest dropped, newest still live (6)

    def test_completed_suggestions_are_appended_to_the_jsonl_log(self, tmp_path):
        log_path = tmp_path / "live_suggestions_test.jsonl"
        state, bus = dash.DashboardState(), Recorder()
        sink = dash.make_sink(state, bus, suggestion_log=dash.SuggestionLog(log_path))
        self._fire(sink, 1, "q1", "POINT: pierwsza")
        self._fire(sink, 2, "q2", "POINT: druga", status="failed")   # not persisted
        self._fire(sink, 3, "q3", "POINT: trzecia")

        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        assert [r["segment"] for r in rows] == ["q1", "q3"]          # the failed one is skipped
        assert rows[0]["text"] == "POINT: pierwsza"
        assert "logged_at" in rows[0]

    def test_suggestions_log_path_shares_the_transcript_stamp(self):
        p = Path("/x/scripts/outputs/live_transcript_20260902_100033.txt")
        assert dash.suggestions_log_path(p).name == "live_suggestions_20260902_100033.jsonl"


class TestPlanPanel:
    PLAN = [
        PlanStep(id="motivation", title="Motivation", done_signals=["dlaczego", "motywacja"]),
        PlanStep(id="experience", title="Experience", done_signals=["doświadczenie"]),
    ]

    def test_a_literal_done_signal_is_badged_mentioned(self):
        state, bus = dash.DashboardState(plan=self.PLAN), Recorder()
        sink = dash.make_sink(state, bus)
        sink({"kind": "line", "stamp": "03:00-03:30", "speaker": "them", "language": "pl",
              "text": "Dlaczego zdecydował się Pan aplikować?", "tagged": True})

        step = state.snapshot()["plan"][0]
        assert step["mentioned"] == ["dlaczego"]           # case-insensitive, literal
        assert state.snapshot()["plan"][1]["mentioned"] == []
        assert bus.of("plan")

    def test_nothing_is_ever_marked_covered(self):
        """P3/G7 is open. The panel claims a phrase was said, not that a step was done."""
        state = dash.DashboardState(plan=self.PLAN)
        keys = set(state.snapshot()["plan"][0])
        assert "covered" not in keys and "done" not in keys and "status" not in keys

    def test_mentions_do_not_repeat(self):
        state, bus = dash.DashboardState(plan=self.PLAN), Recorder()
        sink = dash.make_sink(state, bus)
        for _ in range(3):
            sink({"kind": "line", "stamp": "03:00-03:30", "speaker": "them", "language": "pl",
                  "text": "dlaczego dlaczego", "tagged": True})
        assert state.snapshot()["plan"][0]["mentioned"] == ["dlaczego"]

    def test_tracking_can_be_turned_off(self):
        state, bus = dash.DashboardState(plan=self.PLAN, track_mentions=False), Recorder()
        sink = dash.make_sink(state, bus)
        sink({"kind": "line", "stamp": "03:00-03:30", "speaker": "them", "language": "pl",
              "text": "dlaczego", "tagged": True})
        assert state.snapshot()["plan"][0]["mentioned"] == []
        assert bus.of("plan") == []


class TestSecurityInvariants:
    """SI1/D18 is enforced, not documented: a routable bind is a refusal with a named cause."""

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.1.1"])
    def test_loopback_hosts_are_accepted(self, host):
        dash._assert_loopback(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::", "example.com", ""])
    def test_everything_else_is_refused(self, host):
        with pytest.raises(SystemExit) as excinfo:
            dash._assert_loopback(host)
        assert "loopback" in str(excinfo.value)

    def test_the_ui_loads_nothing_from_a_network(self):
        """A webfont or CDN script on this page would be an egress of an interview surface."""
        html = dash.UI_FILE.read_text(encoding="utf-8")
        for marker in ("http://", "https://", "//cdn", "fonts.googleapis", "integrity="):
            assert marker not in html, f"the dashboard UI must not reference {marker!r}"

    def test_the_ui_has_no_concealment_affordance(self):
        """SI2/D11 — disclosure is a safety floor, so there is no way to hide this panel.

        HTML comments are stripped first: the file's own comment *forbidding* these features
        names them, and a check that fires on the prohibition rather than the affordance would
        be pressure to delete the warning.
        """
        raw = dash.UI_FILE.read_text(encoding="utf-8")
        html = re.sub(r"<!--.*?-->", "", raw, flags=re.S).lower()
        for marker in ("opacity-slider", "click-through", "always-on-top", "stealth",
                       "id=\"hide", "hide-panel", "undetectable"):
            assert marker not in html, f"the dashboard UI must not offer {marker!r}"
        assert "disclosed ai assistance" in html

    def test_call_content_is_never_written_as_html(self):
        """Transcript text is untrusted input to this page; it goes in via textContent."""
        html = dash.UI_FILE.read_text(encoding="utf-8")
        assert ".innerHTML" not in html
        assert "insertAdjacentHTML" not in html


class TestServedEndToEnd:
    """The websocket actually delivering, not just the state that feeds it.

    This class exists because of a real defect. `dashboard.py` uses
    `from __future__ import annotations`, so the handler's `socket: WebSocket` annotation is a
    string FastAPI resolves against MODULE globals; with `fastapi` imported inside
    `create_app()` the name was a local, FastAPI took `socket` for an unknown query parameter,
    and every connection was closed during the handshake with a bare HTTP 403. Every unit test
    above still passed. Nothing short of connecting catches that, so this connects.
    """

    def _client(self, state: dash.DashboardState):
        from fastapi.testclient import TestClient
        info = dash.RunInfo(source="live_transcript_test.txt", mode="replay", session="hr",
                            salience="on (D23)", ceiling_seconds=31.0, suggestions=True)
        return TestClient(dash.create_app(state, dash.EventBus(), info, 31.0))

    def test_a_browser_that_connects_is_sent_the_state(self):
        state = dash.DashboardState()
        state.add_line("00:53-01:06", "them", "pl", "Dzień dobry", True, at=time.monotonic())
        with self._client(state).websocket_connect("/ws") as socket:
            hello = json.loads(socket.receive_text())
        assert hello["kind"] == "hello"          # not "tick" - the spread order matters
        assert hello["info"]["mode"] == "replay"
        assert len(hello["lines"]) == 1
        assert hello["lines"][0]["speaker"] == "them"
        assert hello["meter"]["state"] in ("waiting", "cold")

    def test_the_meter_keeps_ticking_over_the_socket(self):
        """A silent socket is indistinguishable from a calm interview, so it must not be
        silent: the browser marks itself stale when the ticks stop."""
        state = dash.DashboardState()
        with self._client(state).websocket_connect("/ws") as socket:
            kinds = [json.loads(socket.receive_text())["kind"] for _ in range(3)]
        assert kinds[0] == "hello"
        assert "tick" in kinds[1:]

    def test_the_page_is_served_and_names_itself_honestly(self):
        state = dash.DashboardState()
        client = self._client(state)
        page = client.get("/")
        assert page.status_code == 200
        assert "Interview copilot" in page.text
        assert client.get("/healthz").json()["ok"] is True


# ==========================================================================
# 3. The replay driver (the arrival model the meter is demonstrated against)
# ==========================================================================
class TestReplay:
    def test_arrival_is_the_end_stamp_plus_decode_not_the_end_stamp(self):
        """Replaying on end stamps alone would hide the very effect that set the ceiling."""
        lines = rp.parse_replay_lines("[00:00-00:30] them: a 30 s segment\n")
        assert lines[0].end == 30.0
        assert lines[0].arrival == pytest.approx(30.0 + 0.248 + 0.0144 * 30.0)
        assert lines[0].arrival > 30.0

    def test_decode_model_matches_the_measured_envelope(self):
        """D25 measured median 0.59 s and max 0.96 s over the real call's segments."""
        assert rp.decode_seconds(0.0) == pytest.approx(0.248)
        assert rp.decode_seconds(30.0) == pytest.approx(0.68, abs=0.01)
        assert rp.decode_seconds(50.0) < 0.97

    def test_headers_and_junk_are_skipped(self):
        text = "# header\n\nnot a line\n[00:01-00:04] them: real\n"
        assert [line.text for line in rp.parse_replay_lines(text)] == ["[00:01-00:04] them: real"]

    def test_the_first_line_lands_immediately_and_gaps_are_preserved(self):
        lines = rp.parse_replay_lines(
            "[00:00-00:10] them: one\n[00:10-00:40] them: two\n"
        )
        offsets = rp.schedule(lines, speed=1.0)
        assert offsets[0] == 0.0
        assert offsets[1] == pytest.approx(lines[1].arrival - lines[0].arrival)

    def test_speed_divides_every_gap(self):
        lines = rp.parse_replay_lines("[00:00-00:10] them: one\n[00:10-00:40] them: two\n")
        fast = rp.schedule(lines, speed=8.0)
        slow = rp.schedule(lines, speed=1.0)
        assert fast[1] == pytest.approx(slow[1] / 8.0)

    def test_replay_writes_a_tailable_file(self, tmp_path):
        """The whole point: what it writes must come back out of the D19 seam."""
        lines = rp.parse_replay_lines(
            "[00:00-00:05] them (pl): pierwsza\n[00:05-00:10] you (pl): druga\n"
        )
        out = tmp_path / "live_transcript_20260904_130000.txt"
        written = rp.replay(lines, out, speed=1000.0, header=["# replay"])
        assert written == 2

        state, bus = dash.DashboardState(), Recorder()
        assert dash.tail_only(out, state, bus, stop_after_idle=0.4) == 2
        assert [e["line"]["speaker"] for e in bus.of("line")] == ["them", "you"]

    def test_a_replay_of_the_real_call_is_marked_as_a_replay(self, tmp_path):
        """Nothing on screen may be false — including "this is a live interview"."""
        out = tmp_path / "live_transcript_20260904_140000.txt"
        rp.replay(rp.parse_replay_lines("[00:00-00:02] them: x\n"), out, speed=1000.0,
                  header=["# interview_copilot live transcript (REPLAY — not a live call)"])
        assert "REPLAY" in out.read_text(encoding="utf-8").splitlines()[0]

    def test_a_replay_is_never_shown_as_a_live_call(self):
        """`--mode` is a human label and humans forget; the header is evidence. A rehearsal
        presented as a live interview is the one thing this panel must never say."""
        out = Path(rp.OUTPUT_DIR) / "_unit_test_only.txt"
        try:
            rp.replay(rp.parse_replay_lines("[00:00-00:02] them: x\n"), out, speed=1000.0,
                      header=["# interview_copilot live transcript (REPLAY — not a live call)"])
            assert "REPLAY" in dash.detect_mode(out, "live")
            assert dash.detect_mode(out, "") == "REPLAY — not a live call"
        finally:
            out.unlink(missing_ok=True)

    def test_a_real_transcript_keeps_the_declared_mode(self, tmp_path):
        path = tmp_path / "live_transcript_20260904_150000.txt"
        path.write_text("# interview_copilot live transcript\n[00:00-00:02] them: x\n",
                        encoding="utf-8")
        assert dash.detect_mode(path, "live") == "live"

    def test_the_default_replay_name_cannot_be_picked_up_by_watch(self):
        """`newest_transcript()` globs `live_transcript_2*.txt`. A replay left in outputs/ must
        not be what `dashboard.py --watch` calls the newest run."""
        import inspect
        source = inspect.getsource(rp.main)
        assert 'f"replay_transcript_{run_stamp}.txt"' in source
        assert 'f"live_transcript_{run_stamp}.txt"' not in source

    @pytest.mark.skipif(not REAL_TRANSCRIPT.exists(), reason="the real HR transcript is not present")
    def test_the_real_call_parses_two_sided(self):
        lines = rp.parse_replay_lines(REAL_TRANSCRIPT.read_text(encoding="utf-8"))
        assert len(lines) == 110
        assert math.isclose(rp.schedule(lines, 1.0)[-1], lines[-1].arrival - lines[0].arrival)


# ==========================================================================
# 5. The D25 provisional line — the CONSUMER half (#399)
# ==========================================================================
@pytest.fixture
def partial(tmp_path: Path) -> Path:
    """The `.partial` beside the `transcript` fixture, headers and all."""
    path = tmp_path / "live_transcript_20260904_120000.partial"
    path.write_text("# interview_copilot PROVISIONAL lines (D25) — this is NOT the transcript.\n"
                    "# Join key: the START stamp.\n", encoding="utf-8")
    return path


class TestStampStartSeconds:
    def test_the_start_half_is_what_is_parsed(self):
        assert dash.stamp_start_seconds("02:26-02:34") == 146.0

    def test_a_bracketed_stamp_is_accepted(self):
        assert dash.stamp_start_seconds("[00:05-00:09]") == 5.0

    def test_past_an_hour_it_is_a_number_not_a_string(self):
        """The reason this is parsed at all: `"100:00" < "99:00"` lexicographically, so a
        string compare would silently stop superseding 100 minutes into a call."""
        assert dash.stamp_start_seconds("100:00-100:10") > dash.stamp_start_seconds("99:00-99:10")

    def test_junk_is_refused_rather_than_guessed(self):
        assert dash.stamp_start_seconds("not a stamp") == -1.0


class TestPartialPath:
    def test_it_is_derived_from_the_transcript_so_they_cannot_name_different_runs(self):
        got = dash.partial_path_for(Path("/o/live_transcript_20260905_064921.txt"))
        assert got == Path("/o/live_transcript_20260905_064921.partial")


class TestProvisionalOnScreen:
    def test_a_provisional_is_shown_but_is_not_a_transcript_line(self):
        state, bus = dash.DashboardState(), Recorder()
        for event in state.add_provisional("02:26-02:31", "them", "pl",
                                           "Czy może pan stawić", True, at=10.0):
            bus.publish(event)

        shown = bus.of("provisional")[0]["provisional"]
        assert shown["text"] == "Czy może pan stawić"
        assert shown["stamp"] == "02:26-02:31"
        # It must not have become a line: the counters, the ring buffer and the sequence the
        # suggestion staleness rides on are all the transcript's, and a provisional is not one.
        assert state.counters.lines == 0 and state.counters.them == 0
        assert list(state.lines) == []
        assert state.counters.provisional == 1

    def test_the_meter_does_not_treat_a_provisional_as_an_arrival(self):
        """P5's meter measures the wait for a TRANSCRIPT line. A provisional is a look at the
        segment that has not arrived yet; letting it reset the meter would make the meter say
        the wait ended when the thing waited for had not been written."""
        state = dash.DashboardState()
        state.add_provisional("02:26-02:31", "them", "pl", "prefix", True, at=10.0)
        assert state.last_line_at is None
        assert state.tick(now=10.0, ceiling=31.0)["meter"]["state"] == "cold"

    def test_the_final_line_for_the_same_segment_supersedes_it(self):
        state, bus = dash.DashboardState(), Recorder()
        state.add_provisional("02:26-02:31", "them", "pl", "Czy może pan stawić", True, at=10.0)
        for event in state.add_line("02:26-02:34", "them", "pl",
                                    "Czy może pan stawić jakąś sytuację?", True, at=20.0):
            bus.publish(event)

        assert state.provisional is None
        assert state.counters.prov_superseded == 1 and state.counters.prov_orphaned == 0
        # Order matters on screen: the replacement text is sent BEFORE the provisional is taken
        # down, so there is no frame in which the transcript shows neither (#322's bar).
        kinds = [e["kind"] for e in bus.events]
        assert kinds == ["line", "provisional"]
        assert bus.of("provisional")[0]["provisional"] is None

    def test_a_later_segments_final_orphans_a_provisional_whose_own_final_never_came(self):
        """Its segment decoded to nothing or fell below the peak floor. Either way the words
        no longer describe anything being said, so they come off the screen — counted apart,
        because an orphan is the case worth noticing."""
        state, bus = dash.DashboardState(), Recorder()
        state.add_provisional("02:26-02:31", "them", "pl", "half a sentence", True, at=10.0)
        for event in state.add_line("03:01-03:09", "them", "pl", "a later segment", True, at=20.0):
            bus.publish(event)

        assert state.provisional is None
        assert state.counters.prov_orphaned == 1 and state.counters.prov_superseded == 0

    def test_an_earlier_final_does_not_disturb_a_newer_provisional(self):
        state = dash.DashboardState()
        state.add_line("01:00-01:10", "them", "pl", "an old line", True, at=5.0)
        state.add_provisional("02:26-02:31", "them", "pl", "the open one", True, at=10.0)
        state.add_line("01:00-01:10", "them", "pl", "the same old line again", True, at=11.0)
        assert state.provisional is not None
        assert state.counters.prov_superseded == 0 and state.counters.prov_orphaned == 0

    def test_a_provisional_that_arrives_after_its_own_final_is_refused(self):
        """The recorder drops this before it costs a decode, but the consumer holds the same
        line on its own: two files, two tail threads, so arrival order across them is not
        something the recorder can guarantee."""
        state, bus = dash.DashboardState(), Recorder()
        state.add_line("02:26-02:34", "them", "pl", "the full question", True, at=20.0)
        for event in state.add_provisional("02:26-02:31", "them", "pl",
                                           "a stale prefix", True, at=21.0):
            bus.publish(event)
        assert state.provisional is None
        assert bus.of("provisional") == []
        assert state.counters.provisional == 0

    def test_an_empty_provisional_is_not_shown(self):
        state = dash.DashboardState()
        assert state.add_provisional("02:26-02:31", "them", "pl", "", True, at=10.0) == []
        assert state.provisional is None

    def test_a_newer_provisional_replaces_the_one_on_screen(self):
        state = dash.DashboardState()
        state.add_provisional("02:26-02:31", "them", "pl", "Czy może pan", True, at=10.0)
        state.add_provisional("02:26-02:36", "them", "pl", "Czy może pan stawić", True, at=15.0)
        assert state.provisional.text == "Czy może pan stawić"
        assert state.counters.provisional == 2

    def test_a_late_browser_is_sent_the_provisional_in_the_snapshot(self):
        state = dash.DashboardState()
        state.add_provisional("02:26-02:31", "them", "pl", "mid-sentence", True, at=10.0)
        assert state.snapshot()["provisional"]["text"] == "mid-sentence"

    def test_a_snapshot_with_nothing_open_says_so(self):
        assert dash.DashboardState().snapshot()["provisional"] is None


class TestProvisionalExpiry:
    def test_it_is_cleared_once_it_outlives_what_a_segment_can_be(self):
        """A segment cannot stay open past the ceiling, so a provisional older than that means
        the recorder stopped — and a stopped recorder must not leave text claiming to be live."""
        state = dash.DashboardState()
        state.add_provisional("02:26-02:31", "them", "pl", "mid-sentence", True, at=100.0)
        assert state.expire_provisional(now=130.0, ceiling=31.0) == []
        events = state.expire_provisional(now=132.0, ceiling=31.0)
        assert state.provisional is None
        assert state.counters.prov_expired == 1
        assert events[0]["provisional"] is None

    def test_expiry_on_an_empty_slot_is_a_no_op(self):
        state = dash.DashboardState()
        assert state.expire_provisional(now=1e6, ceiling=31.0) == []
        assert state.counters.prov_expired == 0

    def test_it_is_not_folded_into_tick_because_tick_runs_once_per_browser(self):
        """Structural, and it is a real bug if it regresses: `tick()` is called by each
        connected socket's own loop, so a mutation there would clear the slot for whichever
        browser ticked first and leave every other browser showing the line forever."""
        state = dash.DashboardState()
        state.add_provisional("02:26-02:31", "them", "pl", "mid-sentence", True, at=100.0)
        state.tick(now=1e6, ceiling=31.0)
        state.tick(now=1e6, ceiling=31.0)
        assert state.provisional is not None and state.counters.prov_expired == 0


class TestPartialTail:
    def test_the_partial_is_tailed_through_the_same_seam_as_the_transcript(self, partial):
        append(partial, "[02:26-02:31] them (pl): Czy może pan stawić")
        state, bus = dash.DashboardState(), Recorder()
        assert dash.tail_partial(partial, state, bus, stop_after_idle=0.4, from_start=True) == 1
        assert bus.of("provisional")[0]["provisional"]["language"] == "pl"

    def test_the_partial_header_never_reaches_the_ui(self, partial):
        state, bus = dash.DashboardState(), Recorder()
        dash.tail_partial(partial, state, bus, stop_after_idle=0.4, from_start=True)
        assert bus.of("provisional") == []

    def test_a_missing_partial_is_a_supported_run_not_an_error(self, tmp_path):
        """`PARTIAL_DECODE_ENABLED=0` leaves no `.partial` at all, and the dashboard must run."""
        state, bus = dash.DashboardState(), Recorder()
        assert dash.tail_partial(tmp_path / "nope.partial", state, bus, wait_seconds=0.0) == 0
        assert bus.events == []

    def test_a_provisional_replayed_from_the_top_cannot_outrun_the_transcript(self, partial):
        """Why `from_start` defaults to False here and True for the transcript: the transcript
        is the record and a late browser is owed all of it; a provisional is a claim about NOW."""
        append(partial, "[02:26-02:31] them (pl): an old prefix")
        state, bus = dash.DashboardState(), Recorder()
        assert dash.tail_partial(partial, state, bus, stop_after_idle=0.4) == 0
        assert state.provisional is None

    def test_end_to_end_nothing_provisional_survives_the_transcript(self, transcript, partial):
        """The #322 bar, on the real shape of both files and the real INTERLEAVING: two tails,
        two threads, a writer that alternates prefix -> final the way a call does. Batching the
        provisionals first would let the second overwrite the first in the slot and quietly
        turn a two-supersession run into a one-supersession one.
        """
        state, bus = dash.DashboardState(), Recorder()
        finals = threading.Thread(
            target=dash.tail_only, args=(transcript, state, bus),
            kwargs={"stop_after_idle": 1.5}, daemon=True)
        provisionals = threading.Thread(
            target=dash.tail_partial, args=(partial, state, bus),
            kwargs={"stop_after_idle": 1.5, "from_start": True}, daemon=True)
        finals.start()
        provisionals.start()

        for path, line in [
            (partial, "[02:26-02:31] them (pl): Czy może pan stawić"),
            (transcript, "[02:26-02:34] them (pl): Czy może pan stawić jakąś sytuację?"),
            (partial, "[02:51-02:56] them (pl): Interesuje mnie, jak podszedł Pan"),
            (transcript, "[02:51-03:01] them (pl): Interesuje mnie, jak podszedł Pan do fine-tuningu."),
        ]:
            append(path, line)
            time.sleep(0.4)      # > TRANSCRIPT_POLL_SECONDS, so each tail sees its own line
        finals.join(timeout=10)
        provisionals.join(timeout=10)

        assert state.provisional is None
        assert state.counters.lines == 2
        assert state.counters.provisional == 2
        assert state.counters.prov_superseded == 2 and state.counters.prov_orphaned == 0
        rendered = [line.text for line in state.lines]
        assert all("jakąś sytuację" in t or "fine-tuningu" in t for t in rendered)
        # Every provisional event that was published ends in a clear, and the last word on
        # screen is a transcript line.
        assert bus.of("provisional")[-1]["provisional"] is None


class TestProvisionalInTheUi:
    """The rendering rules the browser file must keep. Static assertions on the HTML — the
    same posture the rest of the UI tests take (no browser in the suite)."""

    UI = (PROJECT_ROOT / "scripts" / "dashboard_ui.html").read_text(encoding="utf-8")

    def test_the_provisional_is_rendered_as_call_content_never_as_html(self):
        block = self.UI.split("function renderProvisional")[1].split("\nfunction ")[0]
        # The ASSIGNMENT, not the word: the block carries a comment naming innerHTML as the
        # thing it refuses to use, and a test that cannot tell those apart is not a test.
        assert re.search(r"innerHTML\s*=", block) is None
        assert "bubble.textContent = p.text" in block

    def test_it_says_it_is_provisional_in_words_not_only_in_colour(self):
        block = self.UI.split("function renderProvisional")[1].split("\nfunction ")[0]
        assert 'textContent = "provisional"' in block
        assert "will be replaced by the transcript line" in block

    def test_the_slot_is_emptied_before_anything_is_drawn_into_it(self):
        """One slot, replaced wholesale: there is no path that leaves a superseded provisional
        on screen beside the final line for the same segment."""
        block = self.UI.split("function renderProvisional")[1].split("\nfunction ")[0]
        head = block.split("if (!p")[0]
        assert 'slot.textContent = ""' in head

    def test_transcript_lines_are_inserted_above_the_provisional_slot(self):
        block = self.UI.split("function renderLine")[1].split("\nfunction ")[0]
        assert 'body.insertBefore(turn, el("provisional-slot"))' in block
        assert "body.appendChild(turn)" not in block


# ==========================================================================
# Single-app control switches (#607): the inbound POST /control endpoint,
# the recorder-lifecycle controller, and the three switches it drives.
# ==========================================================================
from types import SimpleNamespace  # noqa: E402


def _app_args(**over):
    base = dict(backend=None, model=None, no_salience=True, suggestion_language=None,
                from_end=False, answer_speaker=None)
    base.update(over)
    return SimpleNamespace(**base)


def _controller(*, backend="local", cloud=False, partials_on=False):
    state = dash.DashboardState()
    bus = dash.EventBus()
    ctrl = dash.AppController(_app_args(backend=backend), state, bus,
                             bundle=object(), ceiling=31.0, partials_on=partials_on)
    ctrl.cloud_available = cloud
    return ctrl


def _control_client(controller=None):
    from fastapi.testclient import TestClient
    state = controller.state if controller else dash.DashboardState()
    bus = controller.bus if controller else dash.EventBus()
    info = dash.RunInfo(source="— (transcription off)", mode="live (app)", session="hr",
                        salience="on (D23)", ceiling_seconds=31.0, suggestions=True)
    return TestClient(dash.create_app(state, bus, info, 31.0, controller=controller))


class TestControlEndpoint:
    """The one inbound path. It exists only in --app mode; every switch paints the state the
    server confirms, and the cloud (egress) switch cannot be flipped into a promise the box
    cannot keep (SI1)."""

    def test_control_is_404_without_a_controller(self):
        # A plain dashboard (no --app) must not expose an inbound control surface at all.
        client = _control_client(controller=None)
        r = client.post("/control", json={"switch": "suggestions", "value": False})
        assert r.status_code == 404

    def test_suggestions_switch_flips_the_flag(self):
        ctrl = _controller()
        client = _control_client(ctrl)
        r = client.post("/control", json={"switch": "suggestions", "value": False})
        assert r.status_code == 200 and r.json()["suggestions"] is False
        assert ctrl.controls.snapshot()[0] is False
        r = client.post("/control", json={"switch": "suggestions", "value": True})
        assert r.json()["suggestions"] is True and ctrl.controls.snapshot()[0] is True

    def test_backend_cloud_is_refused_when_not_configured(self):
        ctrl = _controller(cloud=False)
        client = _control_client(ctrl)
        r = client.post("/control", json={"switch": "backend", "value": "cloud"})
        assert r.status_code == 400
        assert "not configured" in r.json()["error"]
        assert ctrl.controls.snapshot()[1] == "local"        # stayed local — no silent egress

    def test_backend_cloud_announces_the_egress_when_available(self, monkeypatch):
        monkeypatch.setattr(settings, "CLOUD_MODEL", "claude-test")
        ctrl = _controller(cloud=True)
        client = _control_client(ctrl)
        r = client.post("/control", json={"switch": "backend", "value": "cloud"})
        assert r.status_code == 200 and r.json()["backend"] == "cloud"
        assert "EGRESS" in r.json()["banner"]                # SI1: the switch is visible egress
        assert ctrl.controls.snapshot()[1] == "cloud"

    def test_transcription_switch_spawns_and_stops_the_recorder(self, monkeypatch):
        ctrl = _controller()
        # Do not launch a real recorder or load a model in a unit test: stub the lifecycle and
        # assert only the endpoint→controller state wiring.
        spawned = {"n": 0}
        stopped = {"n": 0}
        monkeypatch.setattr(ctrl, "_spawn_recorder",
                            lambda: Path("scripts/outputs/live_transcript_20260101_000000.txt"))
        monkeypatch.setattr(ctrl, "_start_workers", lambda p, s: spawned.__setitem__("n", spawned["n"] + 1))
        monkeypatch.setattr(ctrl, "_stop_recorder", lambda: stopped.__setitem__("n", stopped["n"] + 1))
        client = _control_client(ctrl)

        r = client.post("/control", json={"switch": "transcription", "value": True})
        assert r.status_code == 200 and r.json()["transcription"] is True
        assert ctrl.transcription_on is True and spawned["n"] == 1
        assert r.json()["source"] == "live_transcript_20260101_000000.txt"

        r = client.post("/control", json={"switch": "transcription", "value": False})
        assert r.json()["transcription"] is False
        assert ctrl.transcription_on is False and stopped["n"] == 1

    def test_unknown_switch_is_a_400(self):
        client = _control_client(_controller())
        r = client.post("/control", json={"switch": "nonsense", "value": 1})
        assert r.status_code == 400

    def test_hello_carries_the_controls_state_in_app_mode(self):
        ctrl = _controller(cloud=True)
        with _control_client(ctrl).websocket_connect("/ws") as socket:
            hello = json.loads(socket.receive_text())
        assert hello["controls"] == {
            "transcription": False, "suggestions": True, "backend": "local",
            "cloud_available": True, "source": None,
        }

    def test_hello_controls_is_null_without_app_mode(self):
        with _control_client(controller=None).websocket_connect("/ws") as socket:
            hello = json.loads(socket.receive_text())
        assert hello["controls"] is None
