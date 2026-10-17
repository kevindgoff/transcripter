"""Real Whisper + real diarization on public two-speaker clips (needs the models cached)."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import voice_recorder_cli as v

pytestmark = pytest.mark.integration
SCRIPT = Path(v.__file__)


@pytest.fixture(scope="module")
def tiny():
    t = v.Transcriber("tiny")
    try:
        t.load_model()
    except Exception as e:
        pytest.skip(f"tiny Whisper model unavailable: {e}")
    return t


def words_of(segments):
    return re.findall(r"[a-z']+", " ".join(s.text for s in segments).lower())


def test_cli_from_file_with_speakers_writes_labelled_transcript(testdata, models_available, tmp_path):
    clip = testdata("1-two-speakers-en.wav")
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--from-file", str(clip), "-o", str(out), "-m", "tiny", "--speakers", "2"],
        capture_output=True, text=True, timeout=240,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    md = (out / "1-two-speakers-en.md").read_text(encoding="utf-8")
    assert "**Speaker 1**" in md and "**Speaker 2**" in md and "speaker diarization" in md
    assert "pencil" in md.lower()
    assert [p.suffix for p in out.iterdir()] == [".md"]  # input wav is not copied


def test_chunked_transcription_matches_single_pass(testdata, tiny):
    """Cutting a recording into chunks must not lose or invent words, and must keep timestamps absolute."""
    x = v.read_wav_as_16k_mono(testdata("3-two-speakers-en.wav"))  # 55 s
    single = tiny.transcribe(x, word_timestamps=True)
    chunked = tiny.transcribe(x, word_timestamps=True, chunk_s=20, threshold_s=30)
    assert len(v.plan_chunks(x, v.TARGET_RATE, 20, threshold_s=30)) >= 2

    a, b = set(words_of(single)), set(words_of(chunked))
    assert len(a & b) / len(a | b) > 0.8
    starts = [s.start for s in chunked]
    assert starts == sorted(starts) and chunked[-1].end <= len(x) / v.TARGET_RATE + 1.0
    # word times after the first cut are absolute, not chunk-relative
    late = [w for s in chunked for w in s.words if w.start > 25]
    assert late and all(w.end <= len(x) / v.TARGET_RATE + 1.0 for w in late)


def test_diarization_progress_reaches_the_end(testdata, models_available):
    x = v.read_wav_as_16k_mono(testdata("1-two-speakers-en.wav"))
    d = v.Diarizer(num_speakers=2)
    seen = []
    turns = d.diarize(x, progress=seen.append)
    assert seen and seen == sorted(seen) and seen[-1] == pytest.approx(len(x) / v.TARGET_RATE)
    assert len({t.speaker for t in turns}) == 2


def test_wav_roundtrip_through_real_pipeline(testdata, tiny, tmp_path):
    """record-like WAV written by WavWriter is transcribed the same as the original clip."""
    x = v.read_wav_as_16k_mono(testdata("1-two-speakers-en.wav"))
    p = tmp_path / "copy.wav"
    v.write_wav(p, x)
    assert np.array_equal(v.read_wav_as_16k_mono(p), x)
    assert "pencil" in " ".join(words_of(tiny.transcribe(v.read_wav_as_16k_mono(p))))
