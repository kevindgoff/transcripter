"""Shared fixtures. Tests import the app as `voice_recorder_cli` straight from src/."""
import sys
import threading
import types
import urllib.request
import wave
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import voice_recorder_cli as v  # noqa: E402

TESTDATA = Path.home() / ".cache" / "transcripter" / "testdata"
_CLIP_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/{}"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
    config.addinivalue_line("markers", "integration: uses the real Whisper / diarization models")


# --------------------------------------------------------------------------
# audio helpers
# --------------------------------------------------------------------------

def sine(seconds, rate=16000, freq=440.0, amp=0.5):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def write_wav_file(path, samples, rate=16000, channels=1):
    """Independent WAV writer (stdlib `wave`) so tests don't trust the code under test."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
    return Path(path)


@pytest.fixture
def make_wav(tmp_path):
    def _make(name="in.wav", seconds=2.0, rate=16000, channels=1, freq=440.0, amp=0.5):
        mono = sine(seconds, rate, freq, amp)
        data = np.repeat(mono[:, None], channels, axis=1).reshape(-1) if channels > 1 else mono
        return write_wav_file(tmp_path / name, data, rate, channels)
    return _make


@pytest.fixture(scope="session")
def testdata():
    """Public two-speaker test clips; downloaded once, skipped if unavailable."""
    TESTDATA.mkdir(parents=True, exist_ok=True)

    def get(name):
        p = TESTDATA / name
        if not p.exists():
            try:
                urllib.request.urlretrieve(_CLIP_URL.format(name), p)
            except Exception as e:  # offline
                pytest.skip(f"test clip {name} unavailable: {e}")
        return p
    return get


@pytest.fixture(scope="session")
def models_available():
    if not (v.EMBEDDING_MODEL.exists() and v.SEGMENTATION_MODEL.exists()):
        pytest.skip("diarization models not downloaded (run once with --diarize)")


# --------------------------------------------------------------------------
# fake sounddevice
# --------------------------------------------------------------------------

def make_fake_sd(signal, native_rate=16000, reject_rates=(), block=1024, gate=False, status_every=0):
    """A stand-in `sounddevice` module that feeds `signal` (int16) to the callback
    from a thread. With gate=True it feeds the first half, sets `mod.half`, then
    waits for `mod.release` before feeding the rest."""
    mod = types.ModuleType("sounddevice")

    class PortAudioError(Exception):
        pass

    mod.PortAudioError = PortAudioError
    mod.half = threading.Event()
    mod.release = threading.Event()
    mod.rates_opened = []
    mod.query_devices = lambda kind=None: {"name": "Fake Mic", "default_samplerate": float(native_rate)}

    class InputStream:
        def __init__(self, samplerate, channels, dtype, callback):
            if samplerate in reject_rates:
                raise PortAudioError(f"rate {samplerate} rejected")
            mod.rates_opened.append(samplerate)
            self.cb, self.channels, self._t = callback, channels, None

        def start(self):
            def feed():
                n = len(signal)
                for i, pos in enumerate(range(0, n, block)):
                    if gate and pos >= n // 2 and not mod.half.is_set():
                        mod.half.set()
                        mod.release.wait(10)
                    chunk = signal[pos:pos + block].reshape(-1, 1)
                    status = "overflow" if status_every and i % status_every == 0 else None
                    self.cb(chunk, len(chunk), None, status)
                mod.half.set()

            self._t = threading.Thread(target=feed, daemon=True)
            self._t.start()

        def stop(self):
            mod.release.set()
            if self._t:
                self._t.join(30)

        def close(self):
            pass

    mod.InputStream = InputStream
    return mod


@pytest.fixture
def fake_sd(monkeypatch):
    def install(signal, **kw):
        mod = make_fake_sd(signal, **kw)
        monkeypatch.setitem(sys.modules, "sounddevice", mod)
        return mod
    return install


# --------------------------------------------------------------------------
# fake transcriber / diarizer / app
# --------------------------------------------------------------------------

class FakeTranscriber:
    """Returns canned segments; records how it was called."""

    def __init__(self, segments=None, error=None):
        self.segments = segments if segments is not None else [
            v.Segment(0.0, 2.0, "hello world", [v.Word(0.0, 1.0, " hello"), v.Word(1.0, 2.0, " world")])
        ]
        self.error = error
        self.calls = []

    def transcribe(self, samples, word_timestamps=False, progress=None, **kw):
        self.calls.append({"n": len(samples), "words": word_timestamps})
        if self.error:
            raise self.error
        for s in self.segments:
            if progress:
                progress(s.end)
        return self.segments


class FakeDiarizer:
    def __init__(self, turns=None, error=None):
        self.turns = turns
        self.error = error

    def diarize(self, samples, progress=None):
        if self.error:
            raise self.error
        if progress:
            progress(len(samples) / v.TARGET_RATE)
        return self.turns or []


@pytest.fixture
def make_app(tmp_path):
    def _make(transcriber=None, diarizer=None, save_audio=True, out=None, srt=False):
        app = v.VoiceRecorder(v.PlainUI())
        app._output_dir = out or tmp_path / "out"
        app._output_dir.mkdir(parents=True, exist_ok=True)
        app._save_audio = save_audio
        app._model_size = "tiny"
        app.transcriber = transcriber or FakeTranscriber()
        app.diarizer = diarizer
        if hasattr(app, "_srt"):
            app._srt = srt
        return app
    return _make
