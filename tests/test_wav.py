"""WAV streaming writer, WAV reading/resampling, and small helpers."""
import shutil
import subprocess
import wave

import numpy as np
import pytest

import voice_recorder_cli as v
from conftest import sine, write_wav_file


# ---- WavWriter -----------------------------------------------------------

def test_writer_produces_valid_wav(tmp_path):
    sig = sine(2.0)
    w = v.WavWriter(tmp_path / "a.wav")
    w.write(sig.tobytes())
    w.close()
    with wave.open(str(tmp_path / "a.wav")) as r:
        assert (r.getnchannels(), r.getsampwidth(), r.getframerate()) == (1, 2, 16000)
        assert r.getnframes() == len(sig)
        assert np.array_equal(np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16), sig)


def test_empty_recording_is_still_a_valid_wav(tmp_path):
    w = v.WavWriter(tmp_path / "e.wav")
    w.close()
    with wave.open(str(tmp_path / "e.wav")) as r:
        assert r.getnframes() == 0


def test_file_survives_a_crash_up_to_the_last_header_patch(tmp_path):
    """Simulates a power cut: data written after the last patch is lost, but the
    file is a valid, playable WAV holding everything up to that point."""
    w = v.WavWriter(tmp_path / "crash.wav")
    w.write(sine(3.0).tobytes())
    w.patch()                       # the periodic patch
    w.write(sine(1.0).tobytes())    # arrives after the patch ... then we "crash"
    # (no close)
    with wave.open(str(tmp_path / "crash.wav")) as r:
        assert r.getnframes() == 3 * 16000
        r.readframes(r.getnframes())  # fully readable, no exception
    w.close()
    with wave.open(str(tmp_path / "crash.wav")) as r:
        assert r.getnframes() == 4 * 16000


def test_header_is_patched_automatically_while_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(v.WavWriter, "PATCH_INTERVAL", 0.0)
    w = v.WavWriter(tmp_path / "auto.wav")
    w.write(sine(1.0).tobytes())
    with wave.open(str(tmp_path / "auto.wav")) as r:  # file still open for writing
        assert r.getnframes() == 16000
    w.close()


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffprobe not installed")
def test_ffprobe_agrees_on_duration(tmp_path):
    w = v.WavWriter(tmp_path / "f.wav")
    w.write(sine(5.0).tobytes())
    w.close()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(tmp_path / "f.wav")],
        capture_output=True, text=True, check=True,
    ).stdout
    assert float(out) == pytest.approx(5.0, abs=0.01)


# ---- read_wav_as_16k_mono ------------------------------------------------

def test_read_16k_mono_roundtrip(tmp_path):
    sig = sine(1.5)
    write_wav_file(tmp_path / "x.wav", sig)
    assert np.array_equal(v.read_wav_as_16k_mono(tmp_path / "x.wav"), sig)


def test_read_stereo_is_averaged_to_mono(tmp_path):
    left, right = np.full(1600, 1000, np.int16), np.full(1600, 3000, np.int16)
    stereo = np.stack([left, right], axis=1).reshape(-1)
    write_wav_file(tmp_path / "s.wav", stereo, channels=2)
    out = v.read_wav_as_16k_mono(tmp_path / "s.wav")
    assert len(out) == 1600 and set(out.tolist()) == {2000}


def test_read_48k_is_resampled_and_keeps_pitch(tmp_path):
    write_wav_file(tmp_path / "hi.wav", sine(2.0, rate=48000, freq=440.0), rate=48000)
    out = v.read_wav_as_16k_mono(tmp_path / "hi.wav")
    assert abs(len(out) - 32000) <= 3
    spectrum = np.abs(np.fft.rfft(out.astype(np.float64)))
    peak_hz = np.argmax(spectrum) * 16000 / len(out)
    assert peak_hz == pytest.approx(440.0, abs=2.0)


def test_long_file_spanning_several_read_blocks_stays_in_tune(tmp_path):
    """Regression: the resampler lost track of position across read-block boundaries
    (one block = 2**20 frames, ~22 s at 48 kHz), producing extra, misaligned samples."""
    secs = 50
    sig = sine(secs, rate=48000, freq=440.0)
    write_wav_file(tmp_path / "long48.wav", sig, rate=48000)
    out = v.read_wav_as_16k_mono(tmp_path / "long48.wav")
    one_shot = np.rint(v.LinearResampler(48000, 16000).process(sig.astype(np.float64))).astype(np.int16)
    assert len(out) == len(one_shot)                      # no extra / missing samples at block seams
    assert np.abs(out.astype(np.int32) - one_shot).max() <= 1  # and every sample is where it should be
    spectrum = np.abs(np.fft.rfft(out.astype(np.float64)))
    assert np.argmax(spectrum) * 16000 / len(out) == pytest.approx(440.0, abs=0.5)


def test_read_8k_is_upsampled(tmp_path):
    write_wav_file(tmp_path / "lo.wav", sine(1.0, rate=8000), rate=8000)
    assert abs(len(v.read_wav_as_16k_mono(tmp_path / "lo.wav")) - 16000) <= 3


def test_read_rejects_non_16bit_and_garbage(tmp_path):
    p = tmp_path / "8bit.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1), w.setsampwidth(1), w.setframerate(16000)
        w.writeframes(b"\x80" * 100)
    with pytest.raises(ValueError, match="16-bit"):
        v.read_wav_as_16k_mono(p)
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"this is not a wav file at all")
    with pytest.raises(ValueError, match="not a readable"):
        v.read_wav_as_16k_mono(junk)


# ---- LinearResampler -----------------------------------------------------

@pytest.mark.parametrize("src,dst", [(48000, 16000), (44100, 16000), (8000, 16000)])
def test_streaming_resampler_has_no_seams(src, dst):
    """Block-by-block output must equal one-shot output: no clicks at block edges."""
    x = sine(1.0, rate=src, freq=300).astype(np.float64)
    whole = v.LinearResampler(src, dst).process(x)
    rs, parts, rng, i = v.LinearResampler(src, dst), [], np.random.default_rng(0), 0
    while i < len(x):
        n = int(rng.integers(1, 5000))
        parts.append(rs.process(x[i:i + n]))
        i += n
    chunked = np.concatenate(parts)
    assert len(chunked) == len(whole)
    assert np.allclose(chunked, whole, atol=1e-6)


# ---- helpers -------------------------------------------------------------

@pytest.mark.parametrize("sec,expected", [
    (0, "00:00"), (59.9, "00:59"), (61, "01:01"), (3599, "59:59"),
    (3600, "01:00:00"), (3725, "01:02:05"), (36000, "10:00:00"), (-5, "00:00"),
])
def test_format_clock(sec, expected):
    assert v.format_clock(sec) == expected


def test_unique_stem_never_overwrites(tmp_path):
    assert v.unique_stem(tmp_path, "rec") == "rec"
    (tmp_path / "rec.md").write_text("x")
    assert v.unique_stem(tmp_path, "rec") == "rec_1"
    (tmp_path / "rec_1.wav").write_bytes(b"x")
    assert v.unique_stem(tmp_path, "rec") == "rec_2"


def test_outputs_share_one_base_name(tmp_path):
    o = v.Outputs(tmp_path, "2026-01-02_03-04-05")
    assert [p.name for p in (o.wav, o.md, o.srt)] == ["2026-01-02_03-04-05.wav", "2026-01-02_03-04-05.md", "2026-01-02_03-04-05.srt"]
    assert o.wav.parent == o.md.parent == o.srt.parent == tmp_path


def test_atomic_write_leaves_no_partial_files(tmp_path):
    v.write_text_atomic(tmp_path / "t.md", "héllo ✓")
    assert (tmp_path / "t.md").read_text(encoding="utf-8") == "héllo ✓"
    assert [p.name for p in tmp_path.iterdir()] == ["t.md"]
