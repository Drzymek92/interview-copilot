"""Tests for the straddling-segment code-switch handling (G8 d / #397).

Deterministic and GPU-free. The switch detector and the two candidate decoders are
exercised against a FAKE faster-whisper model whose `detect_language` and `transcribe`
are driven by a per-second language map, so a segment that straddles a switch can be
constructed exactly — including the two shapes measured on the real HR call:
`pl,pl,en,en,...` at [14:39-15:09] and `en,en,pl,pl,...` at [17:53-18:23].
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from scripts import stt  # noqa: E402

SR = settings.SAMPLE_RATE


# ---------------------------------------------------------------- fakes
@dataclass
class _FWSegment:
    """The subset of a faster-whisper segment `stt` reads."""

    start: float
    end: float
    text: str
    avg_logprob: float


class _FakeModel:
    """A faster-whisper stand-in driven by a per-second language map.

    `plan` is a list of (t0, t1, language) covering the buffer. `detect_language` returns
    the language covering most of the audio it is HANDED — which is what makes a
    sub-window probe see something the whole-buffer probe cannot. `scores` gives the
    avg_logprob a forced decode in each language earns, so candidate (a)'s arbitration is
    testable without a model.
    """

    def __init__(
        self,
        plan: list[tuple[float, float, str]],
        scores: dict[str, float],
        whole_buffer_vote: tuple[str, float] | None = None,
        prob: float = 0.99,
    ) -> None:
        self.plan = plan
        self.scores = scores
        self.whole_buffer_vote = whole_buffer_vote
        self.prob = prob
        self.detect_calls: list[float] = []
        self.transcribe_calls: list[tuple[float, str]] = []
        self._offset = 0.0  # set by the harness before each detect_language call

    def _dominant(self, t0: float, t1: float) -> str:
        by_lang: dict[str, float] = {}
        for a, b, lang in self.plan:
            overlap = max(0.0, min(b, t1) - max(a, t0))
            by_lang[lang] = by_lang.get(lang, 0.0) + overlap
        return max(by_lang, key=lambda k: by_lang[k])

    def detect_language(self, audio):  # noqa: ANN001
        seconds = len(audio) / SR
        self.detect_calls.append(seconds)
        # Every sample carries its absolute time (see `_buffer`), so a sub-window knows
        # where in the segment it came from.
        t0 = float(audio[0]) * 1000.0
        if self.whole_buffer_vote is not None and seconds > settings.STT_CODESWITCH_WINDOW_SECONDS + 0.01:
            lang, prob = self.whole_buffer_vote
            return lang, prob, [(lang, prob)]
        lang = self._dominant(t0, t0 + seconds)
        return lang, self.prob, [(lang, self.prob)]

    def transcribe(self, audio, beam_size=5, language="pl", vad_filter=True):  # noqa: ANN001, ARG002
        seconds = len(audio) / SR
        self.transcribe_calls.append((round(seconds, 2), language))
        score = self.scores.get(language, -1.0)
        seg = _FWSegment(start=0.0, end=seconds, text=f"<{language}:{seconds:.1f}s>", avg_logprob=score)
        return iter([seg]), object()


def _buffer(t0: float, seconds: float) -> np.ndarray:
    """A buffer where every sample encodes its own absolute time (scaled into [-1, 1]).

    Sliced sub-windows therefore carry their offset with them, which is what lets the fake
    model answer `detect_language` differently for a window than for the whole buffer —
    the asymmetry the whole defect is made of.
    """
    n = int(seconds * SR)
    return ((t0 + np.arange(n, dtype=np.float32) / SR) / 1000.0).astype(np.float32)


def _transcriber(model: _FakeModel) -> stt.Transcriber:
    """A Transcriber with the fake model injected — no faster-whisper import, no GPU."""
    t = stt.Transcriber.__new__(stt.Transcriber)
    t.model = model
    t.model_name, t.device, t.compute_type = "fake", "cpu", "int8"
    return t


@pytest.fixture(autouse=True)
def _codeswitch_defaults(monkeypatch):
    """Pin the knobs the tests reason about, so a settings edit cannot silently retune them."""
    monkeypatch.setattr(settings, "STT_DETECT_LANGUAGE", True)
    monkeypatch.setattr(settings, "STT_LANGUAGE", "pl")
    monkeypatch.setattr(settings, "STT_LANGUAGE_CANDIDATES", ("pl", "en"))
    monkeypatch.setattr(settings, "STT_LANGUAGE_MIN_PROB", 0.7)
    monkeypatch.setattr(settings, "STT_CODESWITCH_WINDOW_SECONDS", 6.0)
    monkeypatch.setattr(settings, "STT_CODESWITCH_HOP_SECONDS", 3.0)
    monkeypatch.setattr(settings, "STT_CODESWITCH_MIN_WINDOWS", 2)
    monkeypatch.setattr(settings, "STT_CODESWITCH_MIN_SECONDS", 10.0)
    monkeypatch.setattr(settings, "STT_CODESWITCH_MODE", "off")
    monkeypatch.setattr(settings, "STT_CODESWITCH_SCAN", "ends")


@pytest.fixture
def set_mode(monkeypatch):
    """Set STT_CODESWITCH_MODE for one test only — a bare assignment would leak."""

    def _set(mode: str) -> None:
        monkeypatch.setattr(settings, "STT_CODESWITCH_MODE", mode)

    return _set


# ---------------------------------------------------------------- pure helpers
def _win(t0: float, lang: str, prob: float = 0.99) -> stt.LanguageWindow:
    return stt.LanguageWindow(t0=t0, t1=t0 + 6.0, language=lang, probability=prob)


def test_language_runs_collapses_consecutive_votes():
    runs = stt.language_runs([_win(0, "pl"), _win(3, "pl"), _win(6, "en"), _win(9, "en")], 1)
    assert [(r.language, r.n_windows) for r in runs] == [("pl", 2), ("en", 2)]


def test_language_runs_drops_a_single_window_blip():
    """Four segments of the real call end on ONE spurious window; min_windows=2 ignores them."""
    windows = [_win(0, "pl"), _win(3, "pl"), _win(6, "pl"), _win(9, "en")]
    assert [r.language for r in stt.language_runs(windows, 2)] == ["pl"]
    assert [r.language for r in stt.language_runs(windows, 1)] == ["pl", "en"]


def test_spans_from_runs_covers_the_whole_buffer_and_cuts_at_the_midpoint():
    runs = stt.language_runs([_win(0, "pl"), _win(3, "pl")] + [_win(6 + 3 * i, "en") for i in range(7)], 2)
    spans = stt.spans_from_runs(runs, 30.0)
    assert [s.language for s in spans] == ["pl", "en"]
    assert spans[0].t0 == 0.0 and spans[-1].t1 == 30.0  # no audio is dropped by the split
    assert spans[0].t1 == spans[1].t0                   # and none is decoded twice
    assert 6.0 < spans[0].t1 < 12.0                     # the cut lands in the switch gap


def test_spans_from_runs_is_empty_without_runs():
    assert stt.spans_from_runs([], 30.0) == []


def test_weighted_avg_logprob_weights_by_duration_not_by_count():
    """A 0.3 s interjection must not outvote a 20 s sentence — the two decodes being
    compared do not even agree on how many sub-segments the same audio contains."""
    segs = [
        stt.TranscriptSegment(0.0, 20.0, "long", avg_logprob=-0.2),
        stt.TranscriptSegment(20.0, 20.3, "short", avg_logprob=-2.0),
    ]
    assert stt.weighted_avg_logprob(segs) == pytest.approx(-0.2266, abs=1e-3)
    assert stt.weighted_avg_logprob([]) == 0.0


# ---------------------------------------------------------------- planning
def test_off_mode_is_exactly_one_decode_in_the_whole_segment_vote(set_mode):
    """The pre-#397 path, unchanged: one detect, one decode, no windows probed."""
    model = _FakeModel([(0, 9, "pl"), (9, 30, "en")], {"pl": -0.66, "en": -0.34},
                       whole_buffer_vote=("pl", 0.58))
    set_mode("off")
    result = _transcriber(model).transcribe_array(_buffer(0.0, 30.0))
    assert model.transcribe_calls == [(30.0, "pl")]
    assert result.decode_passes == 1 and result.code_switch is False
    assert len(model.detect_calls) == 1  # no sub-window probing at all


def test_short_segments_are_never_probed(set_mode):
    model = _FakeModel([(0, 9, "pl")], {"pl": -0.2, "en": -0.9})
    set_mode("split")
    result = _transcriber(model).transcribe_array(_buffer(0.0, 9.0))
    assert result.code_switch is False
    assert len(model.detect_calls) == 1
    assert result.decode_passes == 1


def test_a_homogeneous_segment_is_not_flagged_and_costs_one_decode(set_mode):
    model = _FakeModel([(0, 30, "pl")], {"pl": -0.2, "en": -0.9})
    set_mode("split")
    result = _transcriber(model).transcribe_array(_buffer(0.0, 30.0))
    assert result.code_switch is False
    assert result.decode_passes == 1
    assert model.transcribe_calls == [(30.0, "pl")]


# ---------------------------------------------------------------- the two real shapes
def test_split_decodes_each_side_of_a_pl_to_en_switch_in_its_own_language(set_mode):
    """The [14:39-15:09] shape: 9 s Polish then 21 s English, whole-segment vote pl p=0.58."""
    model = _FakeModel([(0, 9, "pl"), (9, 30, "en")], {"pl": -0.66, "en": -0.34},
                       whole_buffer_vote=("pl", 0.58))
    set_mode("split")
    result = _transcriber(model).transcribe_array(_buffer(0.0, 30.0))
    assert result.code_switch is True
    assert result.languages == ("pl", "en")
    assert [lang for _s, lang in model.transcribe_calls] == ["pl", "en"]
    assert sum(s for s, _lang in model.transcribe_calls) == pytest.approx(30.0, abs=0.05)
    assert result.language == "en"  # the dominant span names the D19 line's (lang)


def test_split_decodes_each_side_of_an_en_to_pl_switch(set_mode):
    """The [17:53-18:23] shape: 9 s English then 21 s Polish, whole-segment vote pl p=0.99.

    This is the case STT_LANGUAGE_MIN_PROB cannot catch — the whole-segment vote is
    CONFIDENT and wrong about the opening — so the trigger has to be window disagreement.
    """
    model = _FakeModel([(0, 9, "en"), (9, 30, "pl")], {"pl": -0.47, "en": -0.32},
                       whole_buffer_vote=("pl", 0.99))
    set_mode("split")
    result = _transcriber(model).transcribe_array(_buffer(0.0, 30.0))
    assert result.code_switch is True
    assert result.languages == ("en", "pl")
    assert result.language == "pl"


def test_rescore_keeps_the_higher_scoring_language_and_decodes_twice(set_mode):
    model = _FakeModel([(0, 9, "pl"), (9, 30, "en")], {"pl": -0.66, "en": -0.34},
                       whole_buffer_vote=("pl", 0.58))
    set_mode("rescore")
    result = _transcriber(model).transcribe_array(_buffer(0.0, 30.0))
    assert result.decode_passes == 2
    assert [lang for _s, lang in model.transcribe_calls] == ["pl", "en"]
    assert all(s == pytest.approx(30.0, abs=0.05) for s, _lang in model.transcribe_calls)
    assert result.language == "en" and result.avg_logprob == pytest.approx(-0.34)


def test_rescore_keeps_polish_when_polish_scores_higher(set_mode):
    """The gate can fire on a segment that is really Polish; the arbiter must then keep pl.

    Four segments of the real call are exactly this (margins -0.11 to -1.09 nats), which
    is why a switch flag alone must never change the language.
    """
    model = _FakeModel([(0, 9, "pl"), (9, 30, "en")], {"pl": -0.18, "en": -1.27},
                       whole_buffer_vote=("pl", 0.99))
    set_mode("rescore")
    result = _transcriber(model).transcribe_array(_buffer(0.0, 30.0))
    assert result.code_switch is True
    assert result.language == "pl"


def test_a_failed_window_probe_never_ends_a_decode(set_mode):
    """A detector exception must cost the window, not the segment (and not the call)."""

    class _Flaky(_FakeModel):
        def detect_language(self, audio):  # noqa: ANN001
            if len(audio) / SR <= settings.STT_CODESWITCH_WINDOW_SECONDS + 0.01:
                raise RuntimeError("probe blew up")
            return super().detect_language(audio)

    model = _Flaky([(0, 30, "pl")], {"pl": -0.2, "en": -0.9}, whole_buffer_vote=("pl", 0.99))
    set_mode("split")
    result = _transcriber(model).transcribe_array(_buffer(0.0, 30.0))
    assert result.code_switch is False
    assert result.text  # a line still reaches the transcript
    assert model.transcribe_calls == [(30.0, "pl")]


def test_detection_off_disables_the_switch_path_entirely(set_mode, monkeypatch):
    """STT_DETECT_LANGUAGE=0 is the documented escape hatch to the proven forced-pl path."""
    monkeypatch.setattr(settings, "STT_DETECT_LANGUAGE", False)
    model = _FakeModel([(0, 9, "pl"), (9, 30, "en")], {"pl": -0.66, "en": -0.34})
    set_mode("split")
    result = _transcriber(model).transcribe_array(_buffer(0.0, 30.0))
    assert result.code_switch is False
    assert model.detect_calls == []
    assert model.transcribe_calls == [(30.0, "pl")]


def test_split_timestamps_are_offset_into_the_segment(set_mode):
    """A split part is decoded from 0.0; its timestamps must be shifted back into place."""
    model = _FakeModel([(0, 9, "pl"), (9, 30, "en")], {"pl": -0.66, "en": -0.34},
                       whole_buffer_vote=("pl", 0.58))
    set_mode("split")
    result = _transcriber(model).transcribe_array(_buffer(0.0, 30.0))
    assert result.segments[0].start == pytest.approx(0.0)
    assert result.segments[-1].end == pytest.approx(30.0, abs=0.1)
    assert result.segments[1].start > 6.0


# ---------------------------------------------------------------- the two-stage detector
def test_ends_scan_probes_only_two_windows_when_the_ends_agree(set_mode, monkeypatch):
    """The cost that matters: an ordinary segment must not pay for a full sliding scan.

    A full scan of a 30 s segment costs ten encoder passes (+1.363 s measured), enough to
    push a max-length line past the P5 meter's 31.0 s ceiling on 54% of the real call's
    segments. `ends` pays two.
    """
    monkeypatch.setattr(settings, "STT_CODESWITCH_SCAN", "ends")
    model = _FakeModel([(0, 30, "pl")], {"pl": -0.2, "en": -0.9}, whole_buffer_vote=("pl", 0.99))
    set_mode("split")
    result = _transcriber(model).transcribe_array(_buffer(0.0, 30.0))
    assert result.code_switch is False
    # one whole-buffer vote + exactly two end probes
    assert len(model.detect_calls) == 3
    assert model.transcribe_calls == [(30.0, "pl")]


def test_full_scan_probes_every_window(set_mode, monkeypatch):
    monkeypatch.setattr(settings, "STT_CODESWITCH_SCAN", "full")
    model = _FakeModel([(0, 30, "pl")], {"pl": -0.2, "en": -0.9}, whole_buffer_vote=("pl", 0.99))
    set_mode("split")
    _transcriber(model).transcribe_array(_buffer(0.0, 30.0))
    assert len(model.detect_calls) == 1 + 10  # 6 s windows on a 3 s hop across 30 s


def test_ends_scan_still_catches_both_measured_failure_shapes(set_mode, monkeypatch):
    monkeypatch.setattr(settings, "STT_CODESWITCH_SCAN", "ends")
    for plan, vote in (
        ([(0, 9, "pl"), (9, 30, "en")], ("pl", 0.58)),   # [14:39-15:09]
        ([(0, 9, "en"), (9, 30, "pl")], ("pl", 0.99)),   # [17:53-18:23]
    ):
        model = _FakeModel(plan, {"pl": -0.66, "en": -0.34}, whole_buffer_vote=vote)
        set_mode("split")
        assert _transcriber(model).transcribe_array(_buffer(0.0, 30.0)).code_switch is True


def test_ends_scan_misses_a_switch_that_returns_before_the_segment_ends(set_mode, monkeypatch):
    """The stated limit of the cheap stage, pinned so it cannot regress silently.

    pl -> en -> pl leaves both ends agreeing, so `ends` does not flag it and the segment is
    served exactly as it is today. `full` sees it. This is the price of keeping an ordinary
    line inside the meter's ceiling, and it is a documented trade, not an oversight.
    """
    plan = [(0, 8, "pl"), (8, 20, "en"), (20, 30, "pl")]
    monkeypatch.setattr(settings, "STT_CODESWITCH_SCAN", "ends")
    model = _FakeModel(plan, {"pl": -0.4, "en": -0.6}, whole_buffer_vote=("pl", 0.99))
    set_mode("split")
    assert _transcriber(model).transcribe_array(_buffer(0.0, 30.0)).code_switch is False

    monkeypatch.setattr(settings, "STT_CODESWITCH_SCAN", "full")
    model = _FakeModel(plan, {"pl": -0.4, "en": -0.6}, whole_buffer_vote=("pl", 0.99))
    assert _transcriber(model).transcribe_array(_buffer(0.0, 30.0)).code_switch is True


def test_an_unsure_end_is_not_evidence_of_a_switch(set_mode, monkeypatch):
    """Both ends must clear STT_LANGUAGE_MIN_PROB — an unsure probe decides nothing."""
    monkeypatch.setattr(settings, "STT_CODESWITCH_SCAN", "ends")
    model = _FakeModel([(0, 9, "pl"), (9, 30, "en")], {"pl": -0.66, "en": -0.34},
                       whole_buffer_vote=("pl", 0.58), prob=0.5)
    set_mode("split")
    assert _transcriber(model).transcribe_array(_buffer(0.0, 30.0)).code_switch is False


def test_a_switch_too_short_to_hold_a_window_run_is_invisible_to_EVERY_scan(set_mode, monkeypatch):
    """The detector's real reach, measured on a `--from-wav` re-segmentation of the same call.

    A minority language occupying only the segment's last ~2 s never produces the
    STT_CODESWITCH_MIN_WINDOWS consecutive confident windows a run needs, so NEITHER `ends`
    nor `full` sees it. The bound is the window run-length, not the cheap first stage —
    which is why widening the scan is not the answer to it.
    """
    plan = [(0, 28, "pl"), (28, 30, "en")]
    for scan in ("ends", "full"):
        monkeypatch.setattr(settings, "STT_CODESWITCH_SCAN", scan)
        model = _FakeModel(plan, {"pl": -0.4, "en": -0.9}, whole_buffer_vote=("pl", 0.99))
        set_mode("split")
        result = _transcriber(model).transcribe_array(_buffer(0.0, 30.0))
        assert result.code_switch is False, scan
