"""Tests for live_transcribe segmentation and speaker attribution (G9/#326).

Deterministic: the segmenter is a plain dataclass, so these run with no audio, no GPU
and no webrtcvad — we feed it the per-channel speech booleans directly, exactly what the
capture loop derives from the two ChannelGates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from scripts import live_transcribe as lt  # noqa: E402

FRAME_S = settings.VAD_FRAME_MS / 1000.0
SILENCE_FRAMES = int(settings.SEGMENT_MAX_SILENCE_SECONDS / FRAME_S) + 2
FRAME = b"\x00\x00" * int(settings.SAMPLE_RATE * FRAME_S)  # one silent frame's worth of bytes


def _drive(seg: lt.VadSegmenter, script: list[tuple[bool, bool, int]]):
    """Feed (remote_speech, mic_speech, n_frames) tuples; return the emitted segments."""
    emitted = []
    now = 0.0
    for remote, mic, n in script:
        for _ in range(n):
            now += FRAME_S
            out = seg.push(FRAME, remote, mic, now)
            if out is not None:
                emitted.append(out)
    now += FRAME_S
    tail = seg.flush(now)
    if tail is not None:
        emitted.append(tail)
    return emitted


def _speech_frames() -> int:
    """Frames of speech that comfortably clear SEGMENT_MIN_SECONDS."""
    return int((settings.SEGMENT_MIN_SECONDS + 1.0) / FRAME_S)


def test_interviewer_only_segment_is_tagged_them():
    seg = lt.VadSegmenter(frame_seconds=FRAME_S)
    out = _drive(seg, [(True, False, _speech_frames()), (False, False, SILENCE_FRAMES)])
    assert len(out) == 1
    assert out[0].speaker == "them"


def test_candidate_only_segment_is_tagged_you():
    seg = lt.VadSegmenter(frame_seconds=FRAME_S)
    out = _drive(seg, [(False, True, _speech_frames()), (False, False, SILENCE_FRAMES)])
    assert len(out) == 1
    assert out[0].speaker == "you"


def test_dominant_channel_wins_over_a_short_backchannel():
    """Interviewer asks a long question; candidate drops a 3-frame 'mhm' over it.
    The segment must still be attributed to the interviewer."""
    seg = lt.VadSegmenter(frame_seconds=FRAME_S)
    long_q = _speech_frames()
    out = _drive(seg, [
        (True, False, long_q // 2),
        (True, True, 3),            # brief overlap — candidate backchannel
        (True, False, long_q // 2),
        (False, False, SILENCE_FRAMES),
    ])
    assert len(out) == 1
    assert out[0].speaker == "them"


def test_monitor_only_run_never_mislabels_as_you():
    """With --no-mic, mic_speech is always False, so the tag can only ever be 'them'."""
    seg = lt.VadSegmenter(frame_seconds=FRAME_S)
    out = _drive(seg, [(True, False, _speech_frames()), (False, False, SILENCE_FRAMES)])
    assert all(s.speaker == "them" for s in out)


def test_write_line_puts_the_tag_on_transcript_but_not_the_plain_scorer_file(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "OUTPUT_DIR", tmp_path)  # keep the real scripts/outputs clean
    header = "# header\n"
    out = lt.open_outputs("20260101_000000", "run", record=False, channels=2, header=header)
    seg = lt.Segment(index=1, start=7.0, end=36.0, pcm=b"", speech_seconds=20.0, speaker="them")
    out.write_line(seg, "Dzień dobry", "09:00:00")
    out.close()
    transcript = out.transcript_path.read_text(encoding="utf-8")
    plain = out.plain_path.read_text(encoding="utf-8")
    assert "[00:07-00:36] them: Dzień dobry" in transcript
    assert plain.strip() == "Dzień dobry"          # scorer input stays tag-free
    assert "them:" not in plain


# ==========================================================================
# D25 / #399 — provisional lines for the STILL-OPEN segment
#
# The whole point of the decision is that a provisional word can never reach the
# transcript, so most of these tests assert about what is NOT written as much as what is.
# Deterministic like the rest of the file: no audio, no GPU, no Whisper.
# ==========================================================================
import queue  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from scripts import reasoning  # noqa: E402


@dataclass
class _Result:
    """The subset of stt.TranscriptionResult the worker touches."""

    text: str
    language: str = "pl"
    latency_seconds: float = 0.5


class _FakeTranscriber:
    """Returns queued texts in order; records what it was asked to decode."""

    def __init__(self, texts: list[str] | None = None, raises: bool = False) -> None:
        self.texts = list(texts or [])
        self.raises = raises
        self.calls: list[int] = []

    def transcribe_array(self, audio, sample_rate):  # noqa: ANN001, ARG002
        self.calls.append(len(audio))
        if self.raises:
            raise RuntimeError("decode blew up")
        return _Result(self.texts.pop(0) if self.texts else "")


def _loud_frame() -> bytes:
    """A frame well above SEGMENT_MIN_PEAK, so decode_segment does not drop it."""
    sample = int(0.5 * 32767).to_bytes(2, "little", signed=True)
    return sample * int(settings.SAMPLE_RATE * FRAME_S)


def _open_segment(frames: int = 40) -> lt.VadSegmenter:
    """A segmenter with one segment OPEN (speech started, no pause yet)."""
    seg = lt.VadSegmenter(frame_seconds=FRAME_S)
    now = 0.0
    for _ in range(frames):
        now += FRAME_S
        seg.push(_loud_frame(), True, False, now)
    return seg


# -- the snapshot ----------------------------------------------------------
def test_snapshot_is_none_when_nobody_is_speaking():
    seg = lt.VadSegmenter(frame_seconds=FRAME_S)
    assert seg.open_start() is None
    assert seg.snapshot(1.0) is None


def test_snapshot_of_an_open_segment_is_provisional_and_starts_where_the_final_will():
    seg = _open_segment()
    start = seg.open_start()
    snap = seg.snapshot(99.0)
    assert snap is not None
    assert snap.provisional is True
    assert snap.start == start
    assert snap.end == 99.0          # "as of this decode", not the segment's real end
    assert snap.pcm == b"".join(seg.frames)


def test_snapshot_does_not_disturb_the_segment_that_is_still_being_built():
    """D25 amends where lines are WRITTEN, not how segments are cut: the final segment must
    be byte-identical whether or not anyone snapshotted the open one."""
    script = [(True, False, _speech_frames()), (False, False, SILENCE_FRAMES)]
    clean = _drive(lt.VadSegmenter(frame_seconds=FRAME_S), script)

    watched = lt.VadSegmenter(frame_seconds=FRAME_S)
    now = 0.0
    emitted = []
    for remote, mic, n in script:
        for _ in range(n):
            now += FRAME_S
            out = watched.push(FRAME, remote, mic, now)
            watched.snapshot(now)               # snapshot on EVERY frame
            if out is not None:
                emitted.append(out)
    now += FRAME_S
    tail = watched.flush(now)
    if tail is not None:
        emitted.append(tail)

    assert len(emitted) == len(clean) == 1
    assert (emitted[0].start, emitted[0].end, emitted[0].pcm, emitted[0].speaker) == (
        clean[0].start, clean[0].end, clean[0].pcm, clean[0].speaker
    )


def test_snapshot_carries_the_speaker_vote_of_the_open_segment():
    seg = lt.VadSegmenter(frame_seconds=FRAME_S)
    now = 0.0
    for _ in range(40):
        now += FRAME_S
        seg.push(FRAME, False, True, now)       # candidate speaking
    snap = seg.snapshot(now)
    assert snap is not None and snap.speaker == "you"


def test_snapshot_is_withheld_below_the_minimum_speech_floor():
    """Too little voiced audio to be worth a decode — the same floor a final segment faces."""
    seg = lt.VadSegmenter(frame_seconds=FRAME_S)
    seg.push(FRAME, True, False, FRAME_S)       # exactly one speech frame
    assert seg.snapshot(FRAME_S) is None


def test_the_fixed_ab_harness_produces_no_provisional_lines():
    seg = lt.FixedSegmenter(frame_seconds=FRAME_S, window_seconds=8.0)
    seg.push(FRAME, True, False, FRAME_S)
    assert seg.open_start() is None
    assert seg.snapshot(1.0) is None


# -- the cadence -----------------------------------------------------------
def test_clock_fires_one_interval_after_the_segment_opened_then_every_interval():
    clock = lt.PartialClock(interval=5.0)
    assert clock.due(10.0, 12.0) is False       # 2 s into the segment
    assert clock.due(10.0, 15.0) is True        # exactly one interval in
    assert clock.due(10.0, 19.0) is False
    assert clock.due(10.0, 20.0) is True


def test_clock_resets_on_a_new_segment_so_a_long_pause_causes_no_catch_up_burst():
    clock = lt.PartialClock(interval=5.0)
    assert clock.due(10.0, 15.0) is True
    assert clock.due(None, 60.0) is False       # silence between segments
    assert clock.due(100.0, 101.0) is False     # new segment: the interval restarts
    assert clock.due(100.0, 105.0) is True


# -- the mailbox -----------------------------------------------------------
def test_mailbox_keeps_only_the_newest_snapshot_and_counts_what_it_displaced():
    box = lt.PartialMailbox()
    first = lt.Segment(index=1, start=0.0, end=5.0, pcm=b"", speech_seconds=5.0, provisional=True)
    second = lt.Segment(index=1, start=0.0, end=10.0, pcm=b"", speech_seconds=10.0, provisional=True)
    box.put(first)
    box.put(second)
    assert box.dropped == 1
    assert box.take() is second
    assert box.take() is None


# -- the .partial file -----------------------------------------------------
def _outputs(tmp_path, monkeypatch, partials=True):
    monkeypatch.setattr(lt, "OUTPUT_DIR", tmp_path)
    return lt.open_outputs("20260101_000000", "run", record=False, channels=1,
                           header="# header\n", partials=partials)


def test_write_partial_goes_only_to_the_partial_file(tmp_path, monkeypatch):
    out = _outputs(tmp_path, monkeypatch)
    snap = lt.Segment(index=3, start=7.0, end=17.0, pcm=b"", speech_seconds=10.0,
                      speaker="them", provisional=True)
    assert out.write_partial(snap, "Opowiedz mi o", language="pl") is True
    out.close()
    assert out.partial_path is not None
    assert "[00:07-00:17] them (pl): Opowiedz mi o" in out.partial_path.read_text(encoding="utf-8")
    assert "Opowiedz" not in out.transcript_path.read_text(encoding="utf-8")
    assert out.plain_path.read_text(encoding="utf-8") == ""      # scorer input untouched
    assert out.lines == 0 and out.partial_lines == 1


def test_no_partial_file_is_created_when_provisional_lines_are_off(tmp_path, monkeypatch):
    out = _outputs(tmp_path, monkeypatch, partials=False)
    snap = lt.Segment(index=1, start=0.0, end=5.0, pcm=b"", speech_seconds=5.0, provisional=True)
    assert out.write_partial(snap, "nie powinno tego byc") is False
    out.close()
    assert out.partial_path is None
    assert not list(tmp_path.glob("*.partial"))


def test_an_unchanged_re_decode_is_not_written_twice(tmp_path, monkeypatch):
    out = _outputs(tmp_path, monkeypatch)
    a = lt.Segment(index=1, start=2.0, end=7.0, pcm=b"", speech_seconds=5.0, provisional=True)
    b = lt.Segment(index=1, start=2.0, end=12.0, pcm=b"", speech_seconds=10.0, provisional=True)
    assert out.write_partial(a, "Dzien dobry") is True
    assert out.write_partial(b, "Dzien dobry") is False          # same words, longer prefix
    assert out.write_partial(b, "Dzien dobry, prosze") is True   # it grew
    out.close()
    assert out.partial_lines == 2


def test_the_same_words_in_a_new_segment_are_written_again(tmp_path, monkeypatch):
    """Two people can say 'tak' in a row; suppression is per-segment, not global."""
    out = _outputs(tmp_path, monkeypatch)
    first = lt.Segment(index=1, start=2.0, end=7.0, pcm=b"", speech_seconds=5.0, provisional=True)
    second = lt.Segment(index=2, start=40.0, end=45.0, pcm=b"", speech_seconds=5.0, provisional=True)
    assert out.write_partial(first, "tak") is True
    assert out.write_partial(second, "tak") is True
    out.close()
    assert out.partial_lines == 2


def test_an_empty_provisional_decode_writes_nothing(tmp_path, monkeypatch):
    out = _outputs(tmp_path, monkeypatch)
    snap = lt.Segment(index=1, start=0.0, end=5.0, pcm=b"", speech_seconds=5.0, provisional=True)
    assert out.write_partial(snap, "") is False
    out.close()
    assert out.partial_lines == 0


def test_a_provisional_is_refused_once_its_final_line_has_landed(tmp_path, monkeypatch):
    """MEASURED on the 2026-09-05 real-time self-test: a snapshot taken at 02:18 can be decoded
    AFTER the final for the same segment (finals have GPU priority), and it landed 0.39 s late
    carrying a wrong last word. A provisional that arrives after the text it was provisional
    about is noise."""
    out = _outputs(tmp_path, monkeypatch)
    final = lt.Segment(index=1, start=133.0, end=139.0, pcm=b"", speech_seconds=6.0, speaker="them")
    straggler = lt.Segment(index=1, start=133.0, end=138.0, pcm=b"", speech_seconds=5.0,
                           speaker="them", provisional=True)
    out.write_line(final, "doswiadczeniu w TELU", "09:00:00")
    assert out.superseded(straggler) is True
    assert out.write_partial(straggler, "doswiadczeniu w terenie") is False
    out.close()
    assert out.partial_lines == 0


def test_a_provisional_for_a_later_segment_still_passes_after_an_earlier_final(tmp_path, monkeypatch):
    out = _outputs(tmp_path, monkeypatch)
    out.write_line(lt.Segment(index=1, start=133.0, end=139.0, pcm=b"", speech_seconds=6.0),
                   "pierwszy", "09:00:00")
    later = lt.Segment(index=2, start=146.0, end=151.0, pcm=b"", speech_seconds=5.0,
                       provisional=True)
    assert out.superseded(later) is False
    assert out.write_partial(later, "drugi") is True
    out.close()


def test_a_superseded_snapshot_is_dropped_without_spending_a_decode(tmp_path, monkeypatch):
    out = _outputs(tmp_path, monkeypatch)
    out.write_line(lt.Segment(index=1, start=0.0, end=6.0, pcm=b"", speech_seconds=6.0),
                   "ostateczny", "09:00:00")
    fake = _FakeTranscriber(["nigdy"])
    stats = _stats()
    decode_provisional_args = (_real_segment(provisional=True, start=0.0, end=5.0),
                               fake, out, True, "", stats)
    lt.decode_provisional(*decode_provisional_args)
    out.close()
    assert fake.calls == []                       # the GPU was never asked
    assert stats["partial_superseded"] == 1 and stats["partial_decoded"] == 0


def test_the_partial_path_is_recorded_in_runs_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "RUNS_CSV", tmp_path / "runs.csv")
    out = _outputs(tmp_path, monkeypatch)
    out.close()
    lt.append_run_row("run", "start", "end", "ok", 12.0, out)
    assert "live_transcript_20260101_000000.partial" in (tmp_path / "runs.csv").read_text()


# -- the consumer seam (D25: "consumers tail it exactly as they tail the .txt") --
def test_a_partial_file_is_read_by_the_unmodified_transcript_tail(tmp_path, monkeypatch):
    """`reasoning.follow_transcript` + `parse_transcript_line` are D19's consumer seam. D25
    only holds if they read the .partial with NO change, header lines included."""
    out = _outputs(tmp_path, monkeypatch)
    snap = lt.Segment(index=1, start=7.0, end=17.0, pcm=b"", speech_seconds=10.0,
                      speaker="them", provisional=True)
    out.write_partial(snap, "Prosze opowiedziec o sobie", language="pl")
    out.close()

    lines = list(reasoning.follow_transcript(out.partial_path, stop_after_idle=0.0, poll=0.0))
    assert lines == [("00:07-00:17", "them", "pl", "Prosze opowiedziec o sobie")]


def test_the_provisional_and_its_final_line_share_a_start_stamp(tmp_path, monkeypatch):
    """The join key a consumer supersedes on. The end stamp differs (that is the point);
    the start does not, because the segment's start time is fixed at speech onset."""
    out = _outputs(tmp_path, monkeypatch)
    seg = _open_segment()
    snap = seg.snapshot(seg.open_start() + 5.0)
    final = lt.Segment(index=snap.index, start=snap.start, end=snap.start + 30.0, pcm=b"",
                       speech_seconds=30.0, speaker=snap.speaker)
    out.write_partial(snap, "prowizoryczny")
    out.write_line(final, "ostateczny", "09:00:00")
    out.close()

    prov = reasoning.parse_transcript_line(
        out.partial_path.read_text(encoding="utf-8").splitlines()[-1]
    )
    done = reasoning.parse_transcript_line(
        out.transcript_path.read_text(encoding="utf-8").splitlines()[-1]
    )
    assert prov[0].split("-")[0] == done[0].split("-")[0]
    assert prov[3] == "prowizoryczny" and done[3] == "ostateczny"


# -- the worker ------------------------------------------------------------
def _run_worker(work, transcriber, out, mailbox, stats):
    thread = threading.Thread(
        target=lt.transcribe_worker,
        args=(work, transcriber, out, True, stats, mailbox), daemon=True,
    )
    thread.start()
    return thread


def _stats() -> dict:
    # The production shape, not a copy of it: a test-local duplicate goes stale the moment the
    # worker writes a new key, and the worker's KeyError would be raised inside a daemon thread.
    return lt.new_stats()


def _real_segment(provisional=False, start=0.0, end=5.0) -> lt.Segment:
    pcm = _loud_frame() * 10
    return lt.Segment(index=1, start=start, end=end, pcm=pcm, speech_seconds=end - start,
                      speaker="them", provisional=provisional)


def test_the_worker_decodes_a_mailbox_snapshot_into_the_partial_file(tmp_path, monkeypatch):
    out = _outputs(tmp_path, monkeypatch)
    box = lt.PartialMailbox()
    box.put(_real_segment(provisional=True))
    work: queue.Queue = queue.Queue()
    stats = _stats()
    thread = _run_worker(work, _FakeTranscriber(["czesciowy"]), out, box, stats)
    deadline = time.monotonic() + 2.0
    while stats["partial_written"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    work.put(None)
    thread.join(timeout=2.0)
    out.close()
    assert stats["partial_written"] == 1 and stats["decoded"] == 0
    assert "czesciowy" in out.partial_path.read_text(encoding="utf-8")
    assert "czesciowy" not in out.transcript_path.read_text(encoding="utf-8")


def test_a_final_segment_is_decoded_before_a_waiting_provisional(tmp_path, monkeypatch):
    """Finals have strict priority — this is what keeps a provisional from pushing a final
    past the P5 meter's ceiling by queueing ahead of it."""
    out = _outputs(tmp_path, monkeypatch)
    box = lt.PartialMailbox()
    # A LATER segment, so the provisional is not (correctly) refused as superseded — this test
    # is about ordering, and supersession has its own tests.
    box.put(_real_segment(provisional=True, start=40.0, end=45.0))
    work: queue.Queue = queue.Queue()
    work.put(_real_segment())
    stats = _stats()
    fake = _FakeTranscriber(["ostateczny", "czesciowy"])   # first call wins the first text
    thread = _run_worker(work, fake, out, box, stats)
    deadline = time.monotonic() + 2.0
    while stats["partial_written"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    work.put(None)
    thread.join(timeout=2.0)
    out.close()
    assert "ostateczny" in out.transcript_path.read_text(encoding="utf-8")
    assert "czesciowy" in out.partial_path.read_text(encoding="utf-8")


def test_a_failed_provisional_decode_does_not_stop_the_final_lines(tmp_path, monkeypatch):
    """The recorder holds the irreplaceable artefact: nothing in the provisional path may
    take the transcript down with it."""
    out = _outputs(tmp_path, monkeypatch)
    box = lt.PartialMailbox()
    box.put(_real_segment(provisional=True))
    work: queue.Queue = queue.Queue()
    stats = _stats()
    thread = _run_worker(work, _FakeTranscriber(raises=True), out, box, stats)
    time.sleep(0.2)                                  # let the bad provisional be taken
    work.put(_real_segment())
    time.sleep(0.3)
    work.put(None)
    thread.join(timeout=2.0)
    out.close()
    assert stats["partial_written"] == 0
    assert out.lines == 0                            # the fake raises on the final too
    assert thread.is_alive() is False                # the worker survived


def test_with_no_mailbox_the_worker_path_is_unchanged(tmp_path, monkeypatch):
    """`mailbox=None` is the pre-D25 path: block on the queue, decode finals, nothing else."""
    out = _outputs(tmp_path, monkeypatch, partials=False)
    work: queue.Queue = queue.Queue()
    work.put(_real_segment())
    work.put(None)
    stats = _stats()
    lt.transcribe_worker(work, _FakeTranscriber(["ostateczny"]), out, True, stats, None)
    out.close()
    assert stats["decoded"] == 1 and stats["partial_decoded"] == 0
    assert "ostateczny" in out.transcript_path.read_text(encoding="utf-8")


def test_decode_segment_drops_silence_without_calling_the_model():
    quiet = lt.Segment(index=1, start=0.0, end=5.0, pcm=b"\x00\x00" * 1000,
                       speech_seconds=5.0, provisional=True)
    fake = _FakeTranscriber(["nigdy"])
    assert lt.decode_segment(quiet, fake) is None
    assert fake.calls == []


# -- what a provisional costs a closing line (D25's second number, #399) ---
def test_a_final_that_waits_behind_an_in_flight_provisional_is_measured(tmp_path, monkeypatch):
    """The one cost the final-first ordering cannot remove, so it is counted rather than
    argued about: a snapshot already ON the GPU when a segment closes holds that segment's
    final line for the rest of the decode. `final_behind_partial_max` is that wait."""
    out = _outputs(tmp_path, monkeypatch)
    box = lt.PartialMailbox()
    box.put(_real_segment(provisional=True, start=40.0, end=45.0))
    work: queue.Queue = queue.Queue()
    stats = _stats()

    class SlowOnProvisional(_FakeTranscriber):
        def transcribe_array(self, audio, sample_rate):     # noqa: ANN001
            if not self.calls:
                time.sleep(0.50)                            # the provisional, in flight
            return super().transcribe_array(audio, sample_rate)

    thread = _run_worker(work, SlowOnProvisional(["czesciowy", "ostateczny"]), out, box, stats)
    time.sleep(0.15)                       # past the queue poll: the provisional IS decoding
    final = _real_segment(start=50.0, end=55.0)
    final.queued_at = time.monotonic()     # enqueued WHILE the provisional is decoding
    work.put(final)
    deadline = time.monotonic() + 3.0
    while stats["decoded"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    work.put(None)
    thread.join(timeout=3.0)
    out.close()

    assert stats["decoded"] == 1
    # Ordering first: the provisional was decoded, the final came second and still won its file.
    assert "ostateczny" in out.transcript_path.read_text(encoding="utf-8")
    assert stats["final_waits"] == 1
    assert stats["final_behind_partial"] == 1
    assert 0.2 < stats["final_behind_partial_max"] < 1.5
    assert stats["final_behind_partial_max"] <= stats["final_wait_max"]


def test_a_final_that_waits_for_nothing_is_not_charged_to_the_provisional_path(tmp_path, monkeypatch):
    """The measurement must not flatter itself the other way either: an idle decoder means a
    zero attribution, not a small one."""
    out = _outputs(tmp_path, monkeypatch)
    work: queue.Queue = queue.Queue()
    stats = _stats()
    final = _real_segment(start=50.0, end=55.0)
    final.queued_at = time.monotonic()
    work.put(final)
    thread = _run_worker(work, _FakeTranscriber(["ostateczny"]), out, lt.PartialMailbox(), stats)
    deadline = time.monotonic() + 2.0
    while stats["decoded"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    work.put(None)
    thread.join(timeout=2.0)
    out.close()
    assert stats["final_behind_partial"] == 0
    assert stats["final_behind_partial_max"] == 0.0


def test_an_unstamped_segment_is_not_counted_as_a_zero_wait(tmp_path, monkeypatch):
    """`queued_at` is set by the capture loop. A segment built by a test or a future caller
    without it must be left out of the distribution rather than reported as instant."""
    out = _outputs(tmp_path, monkeypatch)
    work: queue.Queue = queue.Queue()
    work.put(_real_segment())              # queued_at defaults to 0.0
    stats = _stats()
    thread = _run_worker(work, _FakeTranscriber(["ostateczny"]), out, lt.PartialMailbox(), stats)
    deadline = time.monotonic() + 2.0
    while stats["decoded"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    work.put(None)
    thread.join(timeout=2.0)
    out.close()
    assert stats["decoded"] == 1 and stats["final_waits"] == 0


# ---------------------------------------------------------------- #428 latency trace
class _TraceResult:
    """The subset of stt.TranscriptionResult LatencyTrace reads."""

    def __init__(self, latency: float = 0.4, language: str = "pl") -> None:
        self.latency_seconds = latency
        self.language = language
        self.code_switch = False
        self.decode_passes = 1


def _traced_segment(index: int, start: float, end: float, queued_at: float) -> lt.Segment:
    return lt.Segment(index=index, start=start, end=end, pcm=b"", speech_seconds=end - start,
                      queued_at=queued_at)


def test_latency_trace_records_close_to_write_for_a_written_line(tmp_path):
    """The #428 measurement: wall time from a segment CLOSING to its line being on disk.

    It has to span the queue wait and the decode, not just the decode — the meter measures
    arrivals, and a line that sat in the queue is exactly as late as one that decoded slowly.
    """
    import json
    import time

    trace = lt.LatencyTrace(tmp_path / "t.jsonl")
    seg = _traced_segment(1, 0.0, 30.0, queued_at=time.monotonic() - 1.25)
    trace.record(seg, _TraceResult(latency=0.9), queue_wait=0.35, behind_partial=0.2, dropped=False)
    trace.close()

    row = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8").strip())
    assert row["dropped"] is False
    assert row["duration"] == 30.0
    assert row["close_to_write"] >= 1.25          # spans the wait, not only the decode
    assert row["close_to_write"] > row["decode_seconds"]
    assert row["queue_wait"] == 0.35 and row["behind_partial"] == 0.2


def test_latency_trace_records_segments_that_produced_no_line(tmp_path):
    """A dropped segment is not an arrival, but it lengthens the wait the meter feels — so it
    is in the trace, flagged, rather than silently absent."""
    import json
    import time

    trace = lt.LatencyTrace(tmp_path / "t.jsonl")
    trace.record(_traced_segment(2, 0.0, 5.0, time.monotonic()), None,
                 queue_wait=0.0, behind_partial=0.0, dropped=True)
    trace.close()
    row = json.loads((tmp_path / "t.jsonl").read_text(encoding="utf-8").strip())
    assert row["dropped"] is True and row["decode_seconds"] == 0.0


# ---------------------------------------------------------------- #458 capture drift
def test_no_stall_run_reports_near_zero_drift():
    """Wall and audio clocks advancing in lockstep — a healthy capture — must not be flagged."""
    mon = lt.CaptureDriftMonitor()
    wall = audio = 0.0
    frame = 0.02  # 20 ms frames, matching VAD_FRAME_MS
    for _ in range(500):
        wall += frame
        audio += frame
        mon.sample(wall, audio)
    assert mon.max_drift == pytest.approx(0.0, abs=1e-9)
    assert mon.total_drift == pytest.approx(0.0, abs=1e-9)


def test_an_injected_stall_sample_sets_max_drift_to_the_gap_size():
    """A single frame where wall jumps far ahead of audio (parec blocked) is the direct
    measurement of a capture stall — the divergence IS the stall's size."""
    mon = lt.CaptureDriftMonitor()
    mon.sample(0.02, 0.02)          # one healthy frame, drift ~0
    mon.sample(2.02, 0.04)          # parec blocked for 2s before the next frame arrived
    assert mon.max_drift == pytest.approx(1.98, abs=1e-9)   # (2.02 - 0.04) - (0.02 - 0.02)
    assert mon.total_drift == pytest.approx(1.98, abs=1e-9)


def test_drift_total_accumulates_across_independent_stalls_even_after_a_partial_recovery():
    """Two separate stalls with a partial recovery between them: `total_drift` sums the
    growth from BOTH stalls, while `max_drift` is only the single worst instantaneous
    reading — proving the two numbers the run summary needs are genuinely different
    quantities (total can exceed max once there is more than one stall)."""
    mon = lt.CaptureDriftMonitor()
    mon.sample(0.0, 0.0)
    mon.sample(1.0, 0.0)      # stall #1: drift grows 0 -> 1.0s
    mon.sample(1.3, 1.0)      # capture partly catches up: drift shrinks 1.0 -> 0.3s
    mon.sample(2.5, 1.3)      # stall #2: drift grows 0.3 -> 1.2s
    assert mon.max_drift == pytest.approx(1.2, abs=1e-9)
    assert mon.total_drift == pytest.approx(1.9, abs=1e-9)   # 1.0 (stall #1) + 0.9 (stall #2)
    assert mon.total_drift > mon.max_drift


def test_an_unpaced_replay_where_audio_races_ahead_reports_zero_drift_not_a_negative_number():
    """`--from-wav` without `--pace` intentionally lets audio time outrun wall time by ~45x
    (see the pacing comment in live_transcribe.py) — that is not a capture stall and must not
    show up as a large negative "drift"."""
    mon = lt.CaptureDriftMonitor()
    wall = 0.0
    audio = 0.0
    for _ in range(50):
        wall += 0.001   # wall barely moves
        audio += 0.5    # audio flies ahead, unthrottled
        mon.sample(wall, audio)
    assert mon.max_drift == pytest.approx(0.0, abs=1e-9)
    assert mon.total_drift == pytest.approx(0.0, abs=1e-9)


def test_line_29_stall_explained_by_the_drift_decomposition():
    """The one real breach of the meter's 31.0 s ceiling on the real call (D27/#428,
    `logs/live_transcribe_20260902.log`): line 29's arrival gap was 32.31s against a 30s
    segment spacing, with this line's decode at 0.70s vs the previous line's 0.38s. The
    latency-step model (D27) says the expected gap is `spacing + (decode(N) - decode(N-1))`;
    what is left over is a capture stall of the size CaptureDriftMonitor exists to catch. A
    live capture cannot be replayed, so this is EXPLAINED via the decomposition, not
    reproduced from raw frames — feeding the monitor the same wall/audio split it would have
    seen shows it lands on the documented +1.99s residual.
    """
    gap, spacing, decode_n, decode_prev = 32.31, 30.0, 0.70, 0.38
    expected_gap = spacing + (decode_n - decode_prev)
    residual = gap - expected_gap
    assert residual == pytest.approx(1.99, abs=0.01)

    mon = lt.CaptureDriftMonitor()
    mon.sample(0.0, 0.0)                      # previous line's arrival, clocks in lockstep
    mon.sample(gap, expected_gap)             # wall advanced the full gap; audio only the
                                               # amount the latency-step model expected
    assert mon.max_drift == pytest.approx(residual, abs=0.01)
    assert mon.total_drift == pytest.approx(residual, abs=0.01)
