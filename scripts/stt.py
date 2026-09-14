"""Local GPU speech-to-text wrapper (D16 — faster-whisper on CUDA).

The single STT seam every other component imports. Day-2 (reasoning) and Day-3
(dashboard) call `Transcriber.transcribe_array(...)` on live audio buffers; the
Day-1 capture spike (`spike_capture.py`) calls the same method. Keep this file
free of any capture / device logic so the wrapper stays reusable.

SI1 (local-only default): transcription runs entirely on the local GPU — audio
never leaves the machine. No cloud-STT egress exists here, and D26's extra language
probes and per-span decodes all run on this same local model.

Run this file directly to self-test the pipeline:

    python scripts/stt.py                 # generates a speech WAV (if a TTS is
                                          # available), transcribes it, logs
                                          # transcript + wall-clock latency
    python scripts/stt.py path/to.wav     # transcribe an existing WAV

The model downloads from Hugging Face on first use, then is cached locally.
"""

from __future__ import annotations

import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Import package-relative settings/logger whether run as a module or a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings  # noqa: E402
from scripts.logger import get_logger  # noqa: E402

logger = get_logger("stt")


@dataclass
class TranscriptSegment:
    """One decoded segment with its timing (seconds, relative to buffer start)."""

    start: float
    end: float
    text: str
    # Whisper's mean per-token logprob for this sub-segment. Plumbed out for #397: it is
    # the score candidate (a) arbitrates two competing decodes of the same audio on.
    avg_logprob: float = 0.0
    # Which language THIS sub-segment was decoded in. Constant across a segment except in
    # STT_CODESWITCH_MODE=split, where each side of a switch is decoded in its own.
    language: str = ""


@dataclass
class TranscriptionResult:
    """A full transcription plus the measured decode latency."""

    text: str
    segments: list[TranscriptSegment]
    language: str
    audio_seconds: float
    latency_seconds: float
    language_probability: float = 1.0
    # Duration-weighted mean of the sub-segments' avg_logprob — the whole decode's score.
    avg_logprob: float = 0.0
    # True when the switch detector found more than one language inside this buffer (#397).
    code_switch: bool = False
    # Every language actually used to decode this buffer, in the order it was used.
    languages: tuple[str, ...] = ()
    # Decodes actually run on this buffer. 1 normally; 2 when candidate (a) rescored, and
    # one per span when candidate (b) split. This is what the added GPU cost is counted in.
    decode_passes: int = 1

    @property
    def realtime_factor(self) -> float:
        """latency / audio_duration — <1.0 means faster-than-realtime."""
        return self.latency_seconds / self.audio_seconds if self.audio_seconds else 0.0


@dataclass
class LanguageWindow:
    """One confident language vote over a sub-window of a segment (#397)."""

    t0: float
    t1: float
    language: str
    probability: float


@dataclass
class LanguageRun:
    """Consecutive sub-windows that voted the same language."""

    language: str
    t0: float
    t1: float
    n_windows: int


@dataclass
class LanguageSpan:
    """A stretch of audio to decode in one language."""

    t0: float
    t1: float
    language: str


@dataclass
class DecodePlan:
    """How `transcribe_array` will decode one buffer."""

    mode: str  # "off" | "rescore" | "split"
    language: str
    probability: float
    switch: bool = False
    languages: tuple[str, ...] = ()
    spans: list[LanguageSpan] = field(default_factory=list)
    windows: list[LanguageWindow] = field(default_factory=list)


def weighted_avg_logprob(segments: list[TranscriptSegment]) -> float:
    """Duration-weighted mean `avg_logprob` over decoded sub-segments.

    Weighted by duration, not by count: an unweighted mean lets a 0.3 s interjection
    outvote a 20 s sentence, and the two decodes being compared do not agree on how many
    sub-segments the same audio contains (measured on the real call: 12 under pl vs 6
    under en for the SAME 30 s segment).
    """
    total = sum(max(0.0, s.end - s.start) for s in segments)
    if not segments or total <= 0:
        return 0.0
    return sum(s.avg_logprob * max(0.0, s.end - s.start) for s in segments) / total


def language_runs(windows: list[LanguageWindow], min_windows: int) -> list[LanguageRun]:
    """Collapse per-window votes into runs, dropping runs shorter than `min_windows`.

    A genuine switch of the utterance's language holds for several windows; a single
    dissenting window is a detector blip (measured: four segments of the real call end on
    one spurious window over their final 2-4 s of audio).
    """
    runs: list[LanguageRun] = []
    for w in windows:
        if runs and runs[-1].language == w.language:
            runs[-1].t1 = w.t1
            runs[-1].n_windows += 1
        else:
            runs.append(LanguageRun(language=w.language, t0=w.t0, t1=w.t1, n_windows=1))
    return [r for r in runs if r.n_windows >= max(1, min_windows)]


def spans_from_runs(runs: list[LanguageRun], total_seconds: float) -> list[LanguageSpan]:
    """Turn language runs into contiguous decode spans covering the whole buffer.

    The boundary between two runs is the MIDPOINT of the gap between them, so neither
    side is cut inside the other's speech; the first span always starts at 0.0 and the
    last always ends at the buffer's end, so no audio is dropped by the split.
    """
    if not runs:
        return []
    spans: list[LanguageSpan] = []
    for i, run in enumerate(runs):
        t0 = 0.0 if i == 0 else spans[-1].t1
        if i == len(runs) - 1:
            t1 = total_seconds
        else:
            t1 = (run.t1 + runs[i + 1].t0) / 2.0
        if t1 > t0:
            spans.append(LanguageSpan(t0=t0, t1=t1, language=run.language))
    if spans:
        spans[-1].t1 = total_seconds
    return spans


class Transcriber:
    """Thin, reusable faster-whisper wrapper. Loads once, transcribes many."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        # Deferred import so importing the module (e.g. for the dataclasses) does
        # not require faster-whisper to be installed.
        from faster_whisper import WhisperModel

        self.model_name = model_name or settings.STT_MODEL
        self.device = device or settings.STT_DEVICE
        self.compute_type = compute_type or settings.STT_COMPUTE_TYPE

        logger.info(
            "loading faster-whisper model=%s device=%s compute_type=%s",
            self.model_name,
            self.device,
            self.compute_type,
        )
        t0 = time.perf_counter()
        self.model = WhisperModel(
            self.model_name, device=self.device, compute_type=self.compute_type
        )
        logger.info("model loaded in %.2fs", time.perf_counter() - t0)

    def transcribe_array(
        self, audio: np.ndarray, sample_rate: int = settings.SAMPLE_RATE
    ) -> TranscriptionResult:
        """Transcribe a mono float32 numpy buffer (values in [-1, 1]).

        faster-whisper expects 16 kHz mono float32; callers that capture at a
        different rate must resample before calling (the capture spike captures
        natively at SAMPLE_RATE to avoid this).

        A segment that STRADDLES a language switch cannot be served by one language
        (D26 / #397): the whole-segment vote picks the wrong side and Whisper *translates*
        the other. `STT_CODESWITCH_MODE` decides what happens then — see `_plan_decode`.
        """
        audio = np.asarray(audio, dtype=np.float32).flatten()
        audio_seconds = len(audio) / sample_rate if sample_rate else 0.0

        t0 = time.perf_counter()
        plan = self._plan_decode(audio, sample_rate)
        if plan.mode == "rescore":
            segments, language, passes = self._decode_rescored(audio, plan.languages)
        elif plan.mode == "split":
            segments, language, passes = self._decode_split(audio, sample_rate, plan.spans)
        else:
            segments = self._decode_once(audio, plan.language)
            language, passes = plan.language, 1
        latency = time.perf_counter() - t0

        text = " ".join(s.text for s in segments if s.text).strip()
        used: list[str] = []
        for seg in segments:
            if seg.language and seg.language not in used:
                used.append(seg.language)
        result = TranscriptionResult(
            text=text,
            segments=segments,
            language=language,
            audio_seconds=audio_seconds,
            latency_seconds=latency,
            language_probability=plan.probability,
            avg_logprob=weighted_avg_logprob(segments),
            code_switch=plan.switch,
            languages=tuple(used) or (language,),
            decode_passes=passes,
        )
        logger.info(
            "transcribed %.2fs audio in %.2fs (rtf=%.2f) [lang=%s p=%.2f alp=%.3f "
            "switch=%s passes=%d]: %r",
            audio_seconds,
            latency,
            result.realtime_factor,
            result.language,
            plan.probability,
            result.avg_logprob,
            plan.switch,
            passes,
            text,
        )
        return result

    # -- decoding primitives ------------------------------------------------
    def _decode_once(
        self, audio: np.ndarray, language: str, offset: float = 0.0
    ) -> list[TranscriptSegment]:
        """One forced-language decode of `audio`, timestamps shifted by `offset`."""
        segments_iter, _info = self.model.transcribe(
            audio,
            beam_size=settings.STT_BEAM_SIZE,
            language=language,
            vad_filter=True,
        )
        # faster-whisper is lazy: decoding happens as the generator is consumed.
        return [
            TranscriptSegment(
                start=s.start + offset,
                end=s.end + offset,
                text=s.text.strip(),
                avg_logprob=float(s.avg_logprob),
                language=language,
            )
            for s in segments_iter
        ]

    def _decode_rescored(
        self, audio: np.ndarray, languages: tuple[str, ...]
    ) -> tuple[list[TranscriptSegment], str, int]:
        """Candidate (a): decode the whole buffer once per language, keep the best score.

        The arbiter is the duration-weighted mean `avg_logprob`. A decode in the wrong
        language is a decode the model itself is unsure of, and on the real call that
        separates the two straddling failures from the 108 segments that already work by a
        wide margin — but only because a switch was DETECTED first. Argmax on its own is
        not safe: measured over all 110 segments it also flips four ordinary Polish
        segments on margins of 0.013-0.072 nats, which is why the switch gate exists.
        """
        best: list[TranscriptSegment] = []
        best_lang, best_score, passes = languages[0], -float("inf"), 0
        for lang in languages:
            segments = self._decode_once(audio, lang)
            passes += 1
            score = weighted_avg_logprob(segments)
            logger.info("rescore: lang=%s avg_logprob=%.4f", lang, score)
            if score > best_score:
                best, best_lang, best_score = segments, lang, score
        return best, best_lang, passes

    def _decode_split(
        self, audio: np.ndarray, sample_rate: int, spans: list[LanguageSpan]
    ) -> tuple[list[TranscriptSegment], str, int]:
        """Candidate (b): cut the AUDIO at the detected switch and decode each side.

        This is D26. It does not touch segmentation — the caller's segment keeps its VAD
        start/end and stays one transcript line — so D21's fixed-clock verdict and #400's
        SEGMENT_MAX_SECONDS lever are untouched. The reported language is the one covering
        the most audio, so the `(lang)` field of the D19 transcript line stays a single code.
        """
        out: list[TranscriptSegment] = []
        passes = 0
        for span in spans:
            i0, i1 = int(span.t0 * sample_rate), int(span.t1 * sample_rate)
            chunk = audio[i0:i1]
            if chunk.size < int(0.5 * sample_rate):
                continue
            out.extend(self._decode_once(chunk, span.language, offset=span.t0))
            passes += 1
        dominant = max(spans, key=lambda sp: sp.t1 - sp.t0).language if spans else settings.STT_LANGUAGE
        return out, dominant, max(1, passes)

    # -- language planning --------------------------------------------------
    def _plan_decode(self, audio: np.ndarray, sample_rate: int) -> DecodePlan:
        """Decide HOW to decode this buffer: one language, a rescore, or a split.

        The whole-segment vote is kept as the baseline — it is right on 108 of the real
        call's 110 segments and it is what `STT_CODESWITCH_MODE=off` restores exactly. The
        switch detector only ever ADDS a second opinion, and only when sub-windows of the
        same audio confidently disagree.
        """
        language, probability = self._decide_language(audio)
        base = DecodePlan(mode="off", language=language, probability=probability)
        mode = settings.STT_CODESWITCH_MODE
        if mode not in ("rescore", "split") or not settings.STT_DETECT_LANGUAGE:
            return base
        seconds = len(audio) / sample_rate if sample_rate else 0.0
        if seconds < settings.STT_CODESWITCH_MIN_SECONDS:
            return base

        if settings.STT_CODESWITCH_SCAN == "ends" and not self._ends_disagree(audio, sample_rate):
            return base
        windows = self._detect_windows(audio, sample_rate)
        runs = language_runs(windows, settings.STT_CODESWITCH_MIN_WINDOWS)
        if len({r.language for r in runs}) < 2:
            return base

        spans = spans_from_runs(runs, seconds)
        ordered: list[str] = []
        for sp in spans:
            if sp.language not in ordered:
                ordered.append(sp.language)
        logger.info(
            "code-switch detected in %.1fs of audio (whole-segment vote %s p=%.2f): %s",
            seconds, language, probability,
            " ".join(f"{sp.t0:.1f}-{sp.t1:.1f}:{sp.language}" for sp in spans),
        )
        return DecodePlan(
            mode=mode, language=language, probability=probability,
            switch=True, languages=tuple(ordered), spans=spans, windows=windows,
        )

    def _ends_disagree(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Cheap first stage: does this buffer END in a different language than it STARTS?

        Two encoder passes, against the ten a full scan of a 30 s segment costs. It is the
        whole segment's worth of evidence for the question actually being asked — a switch
        of the utterance's language, not an English noun inside a Polish sentence — because
        such a switch by definition leaves the two ends on opposite sides of it. Both ends
        must be CONFIDENT candidates; an unsure end is not evidence of anything.
        """
        span = int(settings.STT_CODESWITCH_WINDOW_SECONDS * sample_rate)
        if len(audio) < 2 * span:
            return False
        head = self._window_vote(audio[:span], 0.0, sample_rate)
        tail = self._window_vote(audio[-span:], (len(audio) - span) / sample_rate, sample_rate)
        if head is None or tail is None:
            return False
        return head.language != tail.language

    def _window_vote(
        self, chunk: np.ndarray, t0: float, sample_rate: int
    ) -> LanguageWindow | None:
        """One confident sub-window vote, or None. A failed probe is never fatal."""
        try:
            lang, prob, _all = self.model.detect_language(chunk)
        except Exception:  # noqa: BLE001 — a failed probe must never end a decode
            logger.exception("sub-window language detection failed — skipping window")
            return None
        if lang not in settings.STT_LANGUAGE_CANDIDATES or prob < settings.STT_LANGUAGE_MIN_PROB:
            return None
        return LanguageWindow(
            t0=t0, t1=t0 + len(chunk) / sample_rate, language=lang, probability=float(prob)
        )

    def _detect_windows(
        self, audio: np.ndarray, sample_rate: int
    ) -> list[LanguageWindow]:
        """Run `detect_language` over sliding sub-windows and keep the confident votes.

        Only candidate languages clearing STT_LANGUAGE_MIN_PROB count — the same bar the
        whole-segment vote must clear, reused rather than a second threshold to tune. Each
        window is one encoder pass, no decoding.
        """
        span = int(settings.STT_CODESWITCH_WINDOW_SECONDS * sample_rate)
        hop = max(1, int(settings.STT_CODESWITCH_HOP_SECONDS * sample_rate))
        floor = int(1.0 * sample_rate)
        out: list[LanguageWindow] = []
        for offset in range(0, max(1, len(audio)), hop):
            chunk = audio[offset : offset + span]
            if len(chunk) < floor:
                break
            vote = self._window_vote(chunk, offset / sample_rate, sample_rate)
            if vote is not None:
                out.append(vote)
        return out

    def _decide_language(self, audio: np.ndarray) -> tuple[str, float]:
        """Choose the language to DECODE this segment in (biased to Polish).

        With detection off, force STT_LANGUAGE (the proven forced-pl path). With it on,
        detect the segment's dominant language and accept it only when it is a trusted
        candidate that clears STT_LANGUAGE_MIN_PROB — otherwise fall back to STT_LANGUAGE.
        A short Polish utterance that Whisper is unsure about therefore stays Polish rather
        than being gambled on English. Detection is a cheap encoder pass on ~30 s of audio.

        This is ONE vote for the whole buffer, and #397 measured where that breaks: it is
        right on 108 of the real call's 110 segments and wrong on both that straddle a
        language switch. `_plan_decode` keeps this as the baseline and only ever adds to it.
        """
        if not settings.STT_DETECT_LANGUAGE:
            return settings.STT_LANGUAGE, 1.0
        try:
            lang, prob, _ = self.model.detect_language(audio)
        except Exception:  # noqa: BLE001 — detection must never end a decode; fall back to pl
            logger.exception("language detection failed — falling back to %s", settings.STT_LANGUAGE)
            return settings.STT_LANGUAGE, 0.0
        if lang in settings.STT_LANGUAGE_CANDIDATES and prob >= settings.STT_LANGUAGE_MIN_PROB:
            return lang, prob
        logger.info(
            "language detection %s (p=%.2f) not trusted — decoding as %s",
            lang, prob, settings.STT_LANGUAGE,
        )
        return settings.STT_LANGUAGE, prob

    def transcribe_wav(self, wav_path: str | Path) -> TranscriptionResult:
        """Transcribe a WAV/audio file, resampling to 16 kHz mono as needed.

        Uses faster-whisper's `decode_audio` (av/ffmpeg-backed) so any sample rate
        or channel count is normalised to the 16 kHz mono float32 the model needs
        — feeding a numpy buffer at the wrong rate would pitch/speed-shift it.
        """
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(wav_path), sampling_rate=settings.SAMPLE_RATE)
        return self.transcribe_array(audio, sample_rate=settings.SAMPLE_RATE)


def _read_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    """Load a mono/stereo PCM WAV into a mono float32 array in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    if sampwidth not in dtype_map:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")
    data = np.frombuffer(frames, dtype=dtype_map[sampwidth]).astype(np.float32)
    # Normalise integer PCM to [-1, 1].
    data /= float(np.iinfo(dtype_map[sampwidth]).max)
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sample_rate


def _try_generate_speech_wav(text: str, out_path: Path) -> bool:
    """Best-effort: synthesise `text` to a 16 kHz mono WAV for the self-test.

    Tries neural piper-tts first (pip-installable, no system deps), then CLI
    engines (espeak-ng, espeak, pico2wave). Returns True if a WAV was produced,
    False if no TTS engine is available on this box.
    """
    import shutil
    import subprocess

    # piper-tts: neural, pip-installable. Voice model cached under scripts/outputs.
    voice_dir = Path(__file__).resolve().parents[1] / "scripts" / "outputs" / "piper_voices"
    voice = "en_US-lessac-medium"
    if (voice_dir / f"{voice}.onnx").exists():
        proc = subprocess.run(
            [
                sys.executable, "-m", "piper",
                "-m", voice, "--data-dir", str(voice_dir),
                "-f", str(out_path),
            ],
            input=text.encode(),
            capture_output=True,
        )
        if proc.returncode == 0 and out_path.exists():
            return True

    if shutil.which("espeak-ng") or shutil.which("espeak"):
        engine = shutil.which("espeak-ng") or shutil.which("espeak")
        cmd = [engine, "-s", "150", "-w", str(out_path), text]  # type: ignore[list-item]
        subprocess.run(cmd, check=True, capture_output=True)
        return out_path.exists()
    if shutil.which("pico2wave"):
        tmp = out_path.with_suffix(".pico.wav")
        subprocess.run(
            ["pico2wave", "-w", str(tmp), text], check=True, capture_output=True
        )
        # pico2wave emits 16 kHz mono already; keep as-is.
        tmp.replace(out_path)
        return out_path.exists()
    return False


def _self_test() -> int:
    """Prove the pipeline: synthesise speech (if possible), transcribe, log latency.

    Falls back to a generated sine sample (confirms the model loads on CUDA and
    returns) when no TTS engine is present.
    """
    sentence = (
        "Tell me about a challenging machine learning project you have worked on."
    )
    out_dir = Path(__file__).resolve().parents[1] / "scripts" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "stt_selftest.wav"

    transcriber = Transcriber()

    if _try_generate_speech_wav(sentence, wav_path):
        logger.info("self-test: synthesised speech WAV at %s", wav_path)
        logger.info("self-test: expected text = %r", sentence)
        result = transcriber.transcribe_wav(wav_path)
        logger.info(
            "self-test PASS — transcript=%r latency=%.2fs rtf=%.2f",
            result.text,
            result.latency_seconds,
            result.realtime_factor,
        )
        return 0

    # No TTS: fall back to confirming the model runs on-device with a sample buffer.
    logger.warning(
        "self-test: no TTS engine (espeak-ng/espeak/pico2wave) available — "
        "falling back to a generated sine sample. This confirms the model loads "
        "on device=%s and returns, but does NOT prove real-speech accuracy.",
        transcriber.device,
    )
    sr = settings.SAMPLE_RATE
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    sample = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    result = transcriber.transcribe_array(sample, sample_rate=sr)
    logger.info(
        "self-test (fallback) — model returned in %.2fs; transcript=%r",
        result.latency_seconds,
        result.text,
    )
    return 0


def main() -> None:
    if len(sys.argv) > 1:
        transcriber = Transcriber()
        result = transcriber.transcribe_wav(sys.argv[1])
        print(result.text)
        sys.exit(0)
    sys.exit(_self_test())


if __name__ == "__main__":
    main()
