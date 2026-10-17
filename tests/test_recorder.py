"""AudioRecorder against a fake sounddevice: streaming, levels, fallbacks, failures."""
import sys
import time
import wave

import numpy as np
import pytest

import voice_recorder_cli as v
from conftest import sine


def test_records_exactly_what_the_mic_delivered(fake_sd, tmp_path):
    sig = sine(3.0)
    fake_sd(sig)
    rec = v.AudioRecorder(tmp_path / "r.wav")
    rec.start()
    assert rec.device_name == "Fake Mic"
    seconds = rec.stop()
    assert seconds == pytest.approx(3.0)
    assert np.array_equal(v.read_wav_as_16k_mono(tmp_path / "r.wav"), sig)


def test_stop_is_idempotent(fake_sd, tmp_path):
    fake_sd(sine(1.0))
    rec = v.AudioRecorder(tmp_path / "r.wav")
    rec.start()
    a = rec.stop()
    assert rec.stop() == a


def test_live_levels_track_the_signal(fake_sd, tmp_path):
    sd = fake_sd(sine(2.0, amp=0.5), gate=True)
    rec = v.AudioRecorder(tmp_path / "r.wav")
    rec.start()
    assert sd.half.wait(5)
    time.sleep(0.2)  # let the writer thread drain
    snap = rec.snapshot()
    assert snap.peak == pytest.approx(0.5, abs=0.02)
    assert snap.rms == pytest.approx(0.5 / np.sqrt(2), abs=0.03)
    assert snap.silent_for < 0.5 and not snap.clipped
    assert len(snap.samples) == v.AudioRecorder.RING
    sd.release.set()
    rec.stop()


def test_silence_is_reported(fake_sd, tmp_path):
    sd = fake_sd(np.zeros(32000, dtype=np.int16), gate=True)
    rec = v.AudioRecorder(tmp_path / "r.wav")
    rec.start()
    assert sd.half.wait(5)
    time.sleep(0.4)
    snap = rec.snapshot()
    assert snap.peak == 0.0 and snap.silent_for >= 0.4
    sd.release.set()
    rec.stop()


def test_clipping_is_flagged(fake_sd, tmp_path):
    loud = np.where(np.arange(32000) % 2 == 0, 32767, -32768).astype(np.int16)
    sd = fake_sd(loud, gate=True)
    rec = v.AudioRecorder(tmp_path / "r.wav")
    rec.start()
    assert sd.half.wait(5)
    time.sleep(0.2)
    assert rec.snapshot().clipped
    sd.release.set()
    rec.stop()


def test_device_dropouts_are_counted_not_printed(fake_sd, tmp_path, capsys):
    fake_sd(sine(2.0), status_every=5)
    rec = v.AudioRecorder(tmp_path / "r.wav")
    rec.start()
    rec.stop()
    assert rec.snapshot().dropouts > 0
    assert capsys.readouterr().err == ""  # nothing scribbled over a live display


def test_falls_back_to_native_rate_and_ends_as_16k_mono(fake_sd, tmp_path):
    sig48 = sine(2.0, rate=48000, freq=440)
    sd = fake_sd(sig48, native_rate=48000, reject_rates=(16000,))
    rec = v.AudioRecorder(tmp_path / "r.wav")
    rec.start()
    rec.stop()
    assert sd.rates_opened == [48000]
    with wave.open(str(tmp_path / "r.wav")) as r:
        assert r.getframerate() == 16000
        assert abs(r.getnframes() - 32000) <= 3
    assert not list(tmp_path.glob("*.part"))


def test_no_usable_input_raises_cleanly(fake_sd, tmp_path):
    fake_sd(sine(1.0), native_rate=48000, reject_rates=(16000, 48000))
    rec = v.AudioRecorder(tmp_path / "r.wav")
    with pytest.raises(RuntimeError, match="Could not open input stream"):
        rec.start()
    assert not (tmp_path / "r.wav").exists()


def test_disk_failure_is_reported_and_does_not_hang(fake_sd, tmp_path, monkeypatch):
    fake_sd(sine(3.0), block=512)
    calls = []
    real = v.WavWriter.write

    def flaky(self, pcm):
        if len(calls) >= 3:
            raise OSError("No space left on device")
        calls.append(1)
        return real(self, pcm)

    monkeypatch.setattr(v.WavWriter, "write", flaky)
    notified = []
    rec = v.AudioRecorder(tmp_path / "r.wav", on_error=lambda: notified.append(True))
    rec.start()
    rec.stop()  # must return
    assert rec.error and "No space" in rec.error
    assert notified == [True]  # told once, not once per block


@pytest.mark.slow
def test_a_meeting_longer_than_an_hour_records_correctly(fake_sd, tmp_path):
    """65 minutes streamed to disk: right length, valid header, nothing held in RAM."""
    minutes = 65
    pattern = sine(1.0, freq=300)
    signal = np.tile(pattern, minutes * 60)
    fake_sd(signal, block=16000)
    rec = v.AudioRecorder(tmp_path / "long.wav")
    rec.start()
    seconds = rec.stop()
    assert seconds == pytest.approx(minutes * 60)
    with wave.open(str(tmp_path / "long.wav")) as r:
        assert r.getnframes() == len(signal)
        assert np.array_equal(np.frombuffer(r.readframes(16000), dtype=np.int16), pattern)
    assert (tmp_path / "long.wav").stat().st_size == 44 + len(signal) * 2
    assert rec._writer.data_bytes == len(signal) * 2
    # The recorder keeps only a tiny ring buffer, never the audio itself.
    assert rec._ring.nbytes < 10_000
    assert v.format_clock(seconds) == "01:05:00"
