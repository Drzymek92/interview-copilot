"""Tests for the cross-platform capture backend (scripts/audio_backend.py).

These exercise the *portable* logic that decides frames, mixes/resamples audio, and selects a
backend per platform — everything that can be verified without a sound card. Real Windows WASAPI
loopback and macOS BlackHole capture are hardware paths that must be verified on those OSes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import audio_backend as ab  # noqa: E402


# ── to_mono_16k ──────────────────────────────────────────────────────────
def test_stereo_is_downmixed_to_mono():
    # interleaved L/R int16; average of each pair
    stereo = np.array([100, 300, -200, 0], dtype=np.int16)  # (100,300),(-200,0)
    mono = ab.to_mono_16k(stereo, src_rate=16000, channels=2)
    assert mono.tolist() == [200, -100]
    assert mono.dtype == np.int16


def test_passthrough_at_target_rate_is_identity_for_mono():
    x = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
    assert ab.to_mono_16k(x, src_rate=16000, channels=1).tolist() == x.tolist()


def test_downsample_48k_to_16k_thirds_the_length():
    x = np.zeros(4800, dtype=np.int16)  # 0.1 s @ 48 kHz
    out = ab.to_mono_16k(x, src_rate=48000, channels=1, target_rate=16000)
    assert out.size == 1600  # 0.1 s @ 16 kHz
    assert out.dtype == np.int16


def test_empty_input_returns_empty():
    assert ab.to_mono_16k(np.zeros(0, dtype=np.int16), 48000, 2).size == 0


def test_resample_clips_and_stays_int16():
    x = np.full(9600, 32767, dtype=np.int16)
    out = ab.to_mono_16k(x, src_rate=48000, channels=2, target_rate=16000)
    assert out.dtype == np.int16
    assert out.max() <= 32767 and out.min() >= -32768


# ── _FrameBuffer ─────────────────────────────────────────────────────────
def test_frame_buffer_hands_out_exact_frames_and_keeps_the_remainder():
    fb = ab._FrameBuffer(frame_bytes=4)
    assert fb.pop_frame() is None            # empty
    fb.push(b"\x01\x02")                       # 2 bytes — not a whole frame yet
    assert fb.pop_frame() is None
    fb.push(b"\x03\x04\x05")                   # now 5 bytes buffered
    assert fb.pop_frame() == b"\x01\x02\x03\x04"
    assert fb.pop_frame() is None              # 1 byte left over, held for next push
    fb.push(b"\x06\x07\x08")
    assert fb.pop_frame() == b"\x05\x06\x07\x08"


# ── select_backend ───────────────────────────────────────────────────────
def test_explicit_backend_names():
    assert ab.select_backend("parec").name == "parec"
    assert ab.select_backend("sounddevice").name == "sounddevice"


@pytest.mark.parametrize("platform,expected", [
    ("linux", "parec"),
    ("linux2", "parec"),
    ("win32", "sounddevice"),
    ("darwin", "sounddevice"),
])
def test_auto_selects_by_platform(monkeypatch, platform, expected):
    monkeypatch.setattr(ab.sys, "platform", platform)
    assert ab.select_backend("auto").name == expected


def test_unknown_backend_falls_back_to_auto(monkeypatch):
    monkeypatch.setattr(ab.sys, "platform", "win32")
    assert ab.select_backend("nonsense").name == "sounddevice"


# ── SoundDeviceBackend source resolution (stubbed PortAudio) ─────────────
class _FakeSD:
    """A minimal stand-in for the `sounddevice` module for the discovery logic."""

    def __init__(self, devices, default_in, default_out):
        self._devices = devices
        self.default = SimpleNamespace(device=[default_in, default_out])

    def query_devices(self, ref=None):
        if ref is None:
            return self._devices
        return self._devices[ref]


def _stub_sd(monkeypatch, fake):
    monkeypatch.setattr(ab, "_import_sounddevice", lambda: fake)


def _dev(name, ins, outs, rate=48000):
    return {"name": name, "max_input_channels": ins, "max_output_channels": outs,
            "default_samplerate": rate}


def test_macos_monitor_prefers_a_virtual_loopback_input(monkeypatch):
    monkeypatch.setattr(ab.sys, "platform", "darwin")
    fake = _FakeSD(
        devices=[_dev("MacBook Mic", 1, 0), _dev("BlackHole 2ch", 2, 2), _dev("Speakers", 0, 2)],
        default_in=0, default_out=2,
    )
    _stub_sd(monkeypatch, fake)
    be = ab.SoundDeviceBackend()
    assert be.default_monitor() == "BlackHole 2ch"       # the loopback, not the mic
    assert be.default_mic() == "MacBook Mic"


def test_windows_monitor_is_loopback_of_the_default_output(monkeypatch):
    monkeypatch.setattr(ab.sys, "platform", "win32")
    fake = _FakeSD(
        devices=[_dev("Microphone", 1, 0), _dev("Speakers", 0, 2)],
        default_in=0, default_out=1,
    )
    _stub_sd(monkeypatch, fake)
    be = ab.SoundDeviceBackend()
    assert be.default_monitor() == ab.LOOPBACK_PREFIX + "Speakers"
    assert be.default_mic() == "Microphone"


def test_default_mic_rejects_a_loopback_default_input(monkeypatch):
    monkeypatch.setattr(ab.sys, "platform", "darwin")
    fake = _FakeSD(devices=[_dev("BlackHole 2ch", 2, 2)], default_in=0, default_out=0)
    _stub_sd(monkeypatch, fake)
    assert ab.SoundDeviceBackend().default_mic() is None
