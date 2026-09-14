"""Tests for the #400 cap instrument (D28) — the GPU-free arithmetic only.

The decode arms (`codeswitch`, `transcribe`) are exercised by the reproducible runbook in
`scripts/segment_cap_probe.md`, not here; these pin the pure math the report is computed from.
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.replay_transcript import decode_seconds
from scripts.segment_cap_probe import divergence_between, synth_trace
from scripts.meter_budget import budget_table, classify_gaps


@dataclass
class FakeSeg:
    index: int
    start: float
    end: float
    text: str = ""


def test_synth_trace_uses_the_d25_decode_model() -> None:
    """close->write and arrival come straight from decode_seconds(duration), nothing else."""
    segs = [FakeSeg(0, 0.0, 30.0), FakeSeg(1, 30.0, 50.0)]
    rows = synth_trace(segs)
    assert [r["dropped"] for r in rows] == [False, False]
    assert rows[0]["close_to_write"] == round(decode_seconds(30.0), 4)
    assert rows[0]["written_at"] == round(30.0 + decode_seconds(30.0), 4)
    assert rows[1]["duration"] == 20.0
    assert rows[1]["written_at"] == round(50.0 + decode_seconds(20.0), 4)


def test_synth_trace_meter_sees_a_uniform_capped_run_as_healthy() -> None:
    """Back-to-back cap-length segments are spaced by the cap, so the meter's 1 s allowance holds."""
    cap = 20.0
    segs = [FakeSeg(i, i * cap, (i + 1) * cap) for i in range(6)]
    rows = synth_trace(segs)
    gaps = classify_gaps(rows, cap)
    assert gaps and all(g["healthy"] for g in gaps)  # all spaced exactly cap apart
    # ceiling = cap + allowance; zero false overdues at 1.0 s (the shipped allowance).
    at_1s = next(row for row in budget_table(rows, cap, [1.0]) if abs(row[0] - 1.0) < 1e-9)
    assert at_1s[2] == 0  # false overdues


def test_divergence_identical_is_zero_and_denominator_is_b() -> None:
    a = [FakeSeg(0, 0.0, 10.0, "alpha beta gamma delta")]
    b = [FakeSeg(0, 0.0, 10.0, "alpha beta gamma delta")]
    r = divergence_between(a, b, window=20.0)
    assert r["global_divergence"] == 0.0
    assert r["windowed_divergence"] == 0.0
    assert r["windowed_baseline_words"] == 4  # denominator is b (the baseline)


def test_divergence_counts_baseline_tokens_missing_from_the_new_arm() -> None:
    # baseline has 4 tokens; the new arm drops one -> 1/4 baseline tokens unmatched.
    a = [FakeSeg(0, 0.0, 10.0, "alpha beta gamma")]
    b = [FakeSeg(0, 0.0, 10.0, "alpha beta gamma delta")]
    r = divergence_between(a, b, window=20.0)
    assert r["windowed_baseline_words"] == 4
    assert abs(r["windowed_divergence"] - 0.25) < 1e-9
    assert abs(r["global_divergence"] - 0.25) < 1e-9
