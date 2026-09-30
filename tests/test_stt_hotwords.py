"""Tests for the #324 jargon/hotword-bias knob (`STT_HOTWORDS` / `Transcriber(hotwords=...)`).

Deterministic and GPU-free, against a FAKE faster-whisper model that records the exact kwargs
each call received. The two claims this knob makes are both checked here:
  1. Default OFF (`STT_HOTWORDS` empty) -> every decode call is BYTE-IDENTICAL to before this
     knob existed (no `hotwords` kwarg at all, not `hotwords=None`).
  2. When set, `hotwords` reaches every decode path (`_decode_once`, `_decode_rescored`,
     `_decode_split`) and NEVER reaches `detect_language` (language planning), because that
     model method has no such parameter to receive it.
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


@dataclass
class _FWSegment:
    start: float
    end: float
    text: str
    avg_logprob: float


class _FakeModel:
    """Records every `transcribe`/`detect_language` call's kwargs, decodes as one language."""

    def __init__(self, language: str = "pl", prob: float = 0.99) -> None:
        self.language = language
        self.prob = prob
        self.transcribe_calls: list[dict] = []
        self.detect_calls: list[dict] = []

    def transcribe(self, audio, **kwargs):  # noqa: ANN001
        self.transcribe_calls.append(kwargs)
        lang = kwargs.get("language", self.language)
        seconds = len(audio) / SR
        seg = _FWSegment(start=0.0, end=seconds, text=f"<{lang}>", avg_logprob=-0.1)
        return iter([seg]), object()

    def detect_language(self, audio, **kwargs):  # noqa: ANN001
        self.detect_calls.append(kwargs)
        return self.language, self.prob, [(self.language, self.prob)]


def _buffer(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _transcriber(model: _FakeModel, hotwords: str) -> stt.Transcriber:
    t = stt.Transcriber.__new__(stt.Transcriber)
    t.model = model
    t.model_name, t.device, t.compute_type = "fake", "cpu", "int8"
    t.hotwords = hotwords
    return t


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    """Pin the knobs these tests reason about (mirrors test_stt_codeswitch.py's fixture)."""
    monkeypatch.setattr(settings, "STT_DETECT_LANGUAGE", True)
    monkeypatch.setattr(settings, "STT_LANGUAGE", "pl")
    monkeypatch.setattr(settings, "STT_LANGUAGE_CANDIDATES", ("pl", "en"))
    monkeypatch.setattr(settings, "STT_LANGUAGE_MIN_PROB", 0.7)
    monkeypatch.setattr(settings, "STT_CODESWITCH_MODE", "off")
    monkeypatch.setattr(settings, "STT_HOTWORDS", ())


# --------------------------------------------------------------- construction / precedence
def test_settings_default_is_empty_tuple():
    """The knob's default is empty = today's behaviour (no env var set)."""
    assert settings.STT_HOTWORDS == ()


def test_stt_hotwords_parses_a_comma_env_list(monkeypatch):
    monkeypatch.setenv("STT_HOTWORDS", " BM25 , LLM ,Pydantic")
    import importlib

    from config import settings as settings_mod

    importlib.reload(settings_mod)
    try:
        assert settings_mod.STT_HOTWORDS == ("BM25", "LLM", "Pydantic")
    finally:
        monkeypatch.delenv("STT_HOTWORDS", raising=False)
        importlib.reload(settings_mod)


def test_transcriber_init_precedence(monkeypatch):
    """Exercise the real `__init__` branch (no model load — patch `WhisperModel`)."""
    monkeypatch.setattr(settings, "STT_HOTWORDS", ("BM25", "LLM"))

    class _StubWhisperModel:
        def __init__(self, *a, **kw):  # noqa: ANN002, ANN003
            pass

    monkeypatch.setitem(
        sys.modules, "faster_whisper", type(sys)("faster_whisper")
    )
    sys.modules["faster_whisper"].WhisperModel = _StubWhisperModel

    # hotwords=None -> settings default
    t1 = stt.Transcriber(hotwords=None)
    assert t1.hotwords == "BM25, LLM"

    # hotwords="" -> explicit off, even though settings has terms
    t2 = stt.Transcriber(hotwords="")
    assert t2.hotwords == ""

    # hotwords="RAG,Qdrant" -> CLI/session override wins outright
    t3 = stt.Transcriber(hotwords="RAG,Qdrant")
    assert t3.hotwords == "RAG,Qdrant"


# --------------------------------------------------------------- decode-path behaviour
def test_empty_knob_is_byte_identical_to_pre_324_kwargs():
    """No `hotwords` key at all when the knob is empty — not `hotwords=None`."""
    model = _FakeModel()
    result = _transcriber(model, "").transcribe_array(_buffer(5.0))
    assert model.transcribe_calls == [
        {"beam_size": settings.STT_BEAM_SIZE, "language": "pl", "vad_filter": True}
    ]
    assert "hotwords" not in model.transcribe_calls[0]
    assert result.text == "<pl>"


def test_set_knob_is_passed_on_the_single_decode_path():
    model = _FakeModel()
    _transcriber(model, "BM25, LLM, Pydantic").transcribe_array(_buffer(5.0))
    assert model.transcribe_calls[0]["hotwords"] == "BM25, LLM, Pydantic"


def test_hotwords_never_reaches_language_detection():
    """`detect_language`'s kwargs must never carry `hotwords` — it has no such parameter."""
    model = _FakeModel()
    _transcriber(model, "BM25, LLM").transcribe_array(_buffer(5.0))
    assert len(model.detect_calls) == 1
    assert "hotwords" not in model.detect_calls[0]


def test_set_knob_is_passed_on_every_call_in_rescore_mode(monkeypatch):
    monkeypatch.setattr(settings, "STT_CODESWITCH_MODE", "rescore")
    monkeypatch.setattr(settings, "STT_CODESWITCH_MIN_SECONDS", 1.0)

    class _SwitchModel(_FakeModel):
        def detect_language(self, audio, **kwargs):  # noqa: ANN001
            self.detect_calls.append(kwargs)
            seconds = len(audio) / SR
            # whole-buffer vote: pl, low-confidence, to trigger the switch path
            if seconds > settings.STT_CODESWITCH_WINDOW_SECONDS + 0.01:
                return "pl", 0.5, [("pl", 0.5)]
            # ends disagree: first window pl, last window en
            lang = "pl" if float(audio[0]) == 0.0 else "en"
            return lang, 0.99, [(lang, 0.99)]

    model = _SwitchModel()
    t = _transcriber(model, "BM25, LLM")
    audio = np.concatenate([np.zeros(int(9 * SR), dtype=np.float32),
                             np.ones(int(21 * SR), dtype=np.float32)])
    t.transcribe_array(audio)
    assert len(model.transcribe_calls) == 2  # one decode per candidate language (rescore)
    assert all(c.get("hotwords") == "BM25, LLM" for c in model.transcribe_calls)
    assert all("hotwords" not in c for c in model.detect_calls)


def test_set_knob_is_passed_on_every_call_in_split_mode(monkeypatch):
    monkeypatch.setattr(settings, "STT_CODESWITCH_MODE", "split")
    monkeypatch.setattr(settings, "STT_CODESWITCH_MIN_SECONDS", 1.0)

    class _SwitchModel(_FakeModel):
        def detect_language(self, audio, **kwargs):  # noqa: ANN001
            self.detect_calls.append(kwargs)
            seconds = len(audio) / SR
            if seconds > settings.STT_CODESWITCH_WINDOW_SECONDS + 0.01:
                return "pl", 0.5, [("pl", 0.5)]
            lang = "pl" if float(audio[0]) == 0.0 else "en"
            return lang, 0.99, [(lang, 0.99)]

    model = _SwitchModel()
    t = _transcriber(model, "BM25, LLM")
    audio = np.concatenate([np.zeros(int(9 * SR), dtype=np.float32),
                             np.ones(int(21 * SR), dtype=np.float32)])
    result = t.transcribe_array(audio)
    assert result.code_switch is True
    assert len(model.transcribe_calls) == 2  # one decode per span (split)
    assert all(c.get("hotwords") == "BM25, LLM" for c in model.transcribe_calls)
    assert all("hotwords" not in c for c in model.detect_calls)
