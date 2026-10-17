"""Chunk planning for long recordings, and chunked transcription with a fake Whisper."""
from types import SimpleNamespace

import numpy as np
import pytest

import voice_recorder_cli as v

SR = v.TARGET_RATE


def noise(seconds, amp=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-amp, amp, int(seconds * SR)) * 32767).astype(np.int16)


def silence(x, start_s, end_s):
    x[int(start_s * SR):int(end_s * SR)] = 0


# ---- plan_chunks ---------------------------------------------------------

def test_short_audio_is_one_chunk():
    x = noise(30)
    assert v.plan_chunks(x) == [(0, len(x))]


def test_exactly_at_threshold_is_one_chunk():
    x = np.zeros(int(90 * SR), dtype=np.int16)
    assert v.plan_chunks(x, chunk_s=60, threshold_s=90) == [(0, len(x))]


def test_long_audio_is_cut_in_the_quiet_gaps():
    x = noise(220)
    gaps = [(57.0, 57.4), (115.0, 115.4), (173.0, 173.4)]  # each within 5 s before a cut target
    for a, b in gaps:
        silence(x, a, b)
    chunks = v.plan_chunks(x, chunk_s=60, search_s=5, threshold_s=90)
    assert len(chunks) == 4
    for (_, end), (a, b) in zip(chunks[:-1], gaps):
        assert a <= end / SR <= b, f"cut at {end / SR:.2f}s is not inside the {a}-{b}s gap"


def test_chunks_are_contiguous_and_cover_everything():
    x = noise(400, seed=1)
    chunks = v.plan_chunks(x, chunk_s=60, search_s=5, threshold_s=90)
    assert chunks[0][0] == 0 and chunks[-1][1] == len(x)
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
    assert all(b > a for a, b in chunks)


def test_no_chunk_is_much_longer_than_asked():
    x = noise(500, seed=2)
    for a, b in v.plan_chunks(x, chunk_s=60, search_s=5, threshold_s=90):
        assert (b - a) / SR <= 60 * 1.25 + 0.1


def test_cuts_stay_within_the_search_window_when_nothing_is_quiet():
    x = noise(300, seed=3)
    chunks = v.plan_chunks(x, chunk_s=60, search_s=5, threshold_s=90)
    for a, b in chunks[:-1]:
        assert 55 <= (b - a) / SR <= 60


def test_a_four_hour_recording_is_planned_quickly_and_sanely():
    x = np.zeros(4 * 3600 * SR, dtype=np.int16)  # 4 h of silence: 460 MB of int16, never copied
    chunks = v.plan_chunks(x)
    assert 22 <= len(chunks) <= 25
    assert chunks[-1][1] == len(x)


# ---- Transcriber chunking (fake Whisper) -----------------------------------

class FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append({"n": len(audio), "dtype": audio.dtype, "max": float(np.abs(audio).max()), **kw})
        words = None
        if kw.get("word_timestamps"):
            words = [SimpleNamespace(start=1.0, end=2.0, word=" he"), SimpleNamespace(start=2.0, end=3.0, word="llo")]

        def gen():
            yield SimpleNamespace(start=1.0, end=3.0, text=" hello", words=words)
        return gen(), SimpleNamespace()


def make_transcriber():
    t = v.Transcriber("tiny")
    t._model = FakeWhisper()
    return t


def test_chunk_timestamps_are_made_absolute():
    x = np.zeros(200 * SR, dtype=np.int16)
    t = make_transcriber()
    segs = t.transcribe(x, word_timestamps=True, chunk_s=60, threshold_s=100)
    plan = v.plan_chunks(x, SR, 60, threshold_s=100)
    assert len(segs) == len(plan) > 1
    for seg, (a, _) in zip(segs, plan):
        assert seg.start == pytest.approx(a / SR + 1.0)
        assert seg.words[0].start == pytest.approx(a / SR + 1.0)
        assert seg.words[1].end == pytest.approx(a / SR + 3.0)
    assert [s.start for s in segs] == sorted(s.start for s in segs)
    assert sum(c["n"] for c in t._model.calls) == len(x)  # every sample transcribed exactly once


def test_whisper_receives_normalised_float32():
    x = np.full(SR, 16384, dtype=np.int16)
    t = make_transcriber()
    t.transcribe(x)
    call = t._model.calls[0]
    assert call["dtype"] == np.float32 and call["max"] == pytest.approx(0.5)
    assert call["vad_filter"] is True and call["word_timestamps"] is False


def test_progress_is_monotonic_and_reaches_the_end():
    x = np.zeros(200 * SR, dtype=np.int16)
    seen = []
    make_transcriber().transcribe(x, progress=seen.append, chunk_s=60, threshold_s=100)
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(200.0)


def test_transcriber_errors_are_wrapped():
    t = v.Transcriber("tiny")
    with pytest.raises(RuntimeError, match="Model not loaded"):
        t.transcribe(np.zeros(100, dtype=np.int16))
    t._model = SimpleNamespace(transcribe=lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(RuntimeError, match="Transcription failed: boom"):
        t.transcribe(np.zeros(100, dtype=np.int16))


def test_invalid_model_size_is_rejected():
    with pytest.raises(ValueError):
        v.Transcriber("gigantic").load_model()
