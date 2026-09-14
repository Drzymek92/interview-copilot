"""Tests for the #428 meter-allowance instrument.

Deterministic and I/O-free apart from a tmp trace file: every function under test is pure
arithmetic over latency-trace rows. The cases below encode the distinctions the decision turns
on — a uniform slowdown vs a latency STEP, and a back-to-back gap vs one across a silence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import meter_budget as mb  # noqa: E402

CAP = 30.0


def _row(index: int, end: float, written_at: float, latency: float,
         dropped: bool = False, duration: float = 30.0) -> dict:
    return {
        "index": index, "start": end - duration, "end": end, "duration": duration,
        "speaker": "them", "continued": False, "dropped": dropped,
        "queued_at": written_at - latency, "written_at": written_at,
        "close_to_write": latency, "queue_wait": 0.0, "behind_partial": 0.0,
        "decode_seconds": latency, "language": "pl", "code_switch": False, "decode_passes": 1,
    }


def test_a_uniform_slowdown_does_not_reach_the_meter():
    """The point the whole decision rests on: latency cancels in the DIFFERENCE.

    Three back-to-back 30 s segments, each taking a full 2 s to decode — four times the shipped
    allowance — still arrive 30 s apart, because every line is late by the same amount.
    """
    rows = [_row(1, 30.0, 32.0, 2.0), _row(2, 60.0, 62.0, 2.0), _row(3, 90.0, 92.0, 2.0)]
    g = mb.classify_gaps(rows, CAP)
    assert [round(x["gap"], 6) for x in g] == [30.0, 30.0]
    assert all(x["healthy"] for x in g)
    assert mb.minimum_allowance(rows, CAP)[0] == 0.0


def test_a_latency_STEP_is_what_the_meter_feels():
    """A fast line followed by a slow one: only the step lands on the gap."""
    rows = [_row(1, 30.0, 30.2, 0.2), _row(2, 60.0, 61.4, 1.4)]
    g = mb.classify_gaps(rows, CAP)
    assert round(g[0]["gap"], 6) == 31.2
    assert round(g[0]["latency_step"], 6) == 1.2
    need, worst = mb.minimum_allowance(rows, CAP)
    assert round(need, 6) == 1.2 and worst["index"] == 2


def test_a_gap_across_a_silence_is_not_a_false_overdue():
    """The cap bounds a segment, not the pause before one — the real call's 43.2 s case."""
    rows = [_row(1, 30.0, 30.5, 0.5), _row(2, 100.0, 100.5, 0.5)]
    g = mb.classify_gaps(rows, CAP)
    assert g[0]["gap"] > 60.0
    assert g[0]["healthy"] is False
    # No allowance can or should fix it, so it never counts against one.
    assert all(f == 0 for _a, _c, f, _n in mb.budget_table(rows, CAP, [0.5, 1.0, 30.0]))


def test_a_segment_that_produced_no_line_is_not_an_arrival_but_lengthens_the_gap():
    rows = [_row(1, 30.0, 30.5, 0.5), _row(2, 45.0, 45.2, 0.2, dropped=True, duration=15.0),
            _row(3, 75.0, 75.6, 0.6)]
    assert [r["index"] for r in mb.arrivals(rows)] == [1, 3]
    g = mb.classify_gaps(rows, CAP)
    assert len(g) == 1 and g[0]["healthy"] is False  # spacing 45 s > the 30 s cap


def test_budget_table_counts_only_back_to_back_breaches():
    rows = [_row(1, 30.0, 30.2, 0.2), _row(2, 60.0, 61.4, 1.4), _row(3, 90.0, 90.5, 0.5)]
    table = {a: f for a, _c, f, _n in mb.budget_table(rows, CAP, [0.5, 1.0, 1.5, 2.0])}
    assert table[0.5] == 1 and table[1.0] == 1  # the 31.2 s gap breaches both
    assert table[1.5] == 0 and table[2.0] == 0


def test_minimum_allowance_is_zero_when_there_is_nothing_to_cover():
    assert mb.minimum_allowance([], CAP) == (0.0, None)


def test_pct_returns_a_value_that_actually_occurred():
    """Nearest-rank, no interpolation — every number the report quotes is a real measurement."""
    vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert mb.pct(vals, 100) == 0.5
    assert mb.pct(vals, 50) in vals
    assert mb.pct([], 95) != mb.pct([], 95)  # nan


def test_load_orders_by_arrival(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in
                           [_row(2, 60.0, 61.0, 1.0), _row(1, 30.0, 30.5, 0.5)]), encoding="utf-8")
    assert [r["index"] for r in mb.load(p)] == [1, 2]
