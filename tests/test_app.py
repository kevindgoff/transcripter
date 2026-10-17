"""End-to-end app flow with a fake mic and fake models: files, exit codes, hints."""
import os
import re
import signal
import sys
import tempfile
import threading
import time
import wave
from argparse import Namespace

import numpy as np
import pytest

import voice_recorder_cli as v
from conftest import FakeDiarizer, FakeTranscriber, sine, write_wav_file


def sigint_when_recording(app, timeout=10.0):
    """Send a real SIGINT once run_recording() has installed its handler
    (never earlier, or it would kill the test runner)."""
    original = signal.getsignal(signal.SIGINT)

    def fire():
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if app.recorder is not None and signal.getsignal(signal.SIGINT) is not original:
                time.sleep(0.2)
                os.kill(os.getpid(), signal.SIGINT)
                return
            time.sleep(0.02)

    threading.Thread(target=fire, daemon=True).start()


# ---- recording flow ---------------------------------------------------------

def test_ctrl_c_stops_recording_and_saves_wav_and_md_together(fake_sd, make_app, capsys):
    sig = sine(3.0)
    fake_sd(sig)
    app = make_app()
    sigint_when_recording(app)
    app.run_recording()

    out = app._output_dir
    wavs, mds = sorted(out.glob("*.wav")), sorted(out.glob("*.md"))
    assert len(wavs) == len(mds) == 1
    assert wavs[0].stem == mds[0].stem
    assert re.fullmatch(r"\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d", wavs[0].stem)
    assert np.array_equal(v.read_wav_as_16k_mono(wavs[0]), sig)

    md = mds[0].read_text(encoding="utf-8")
    assert "hello world" in md and "**Duration:** 00:03" in md
    assert f"**Audio:** [{wavs[0].name}]({wavs[0].name})" in md
    assert "Audio saved" in capsys.readouterr().out


def test_wav_is_on_disk_before_transcription_starts(fake_sd, make_app):
    """The whole point of saving first: a transcription crash must not lose the audio."""
    fake_sd(sine(2.0))
    seen = {}

    class Spy(FakeTranscriber):
        def transcribe(self, samples, **kw):
            seen["wavs"] = [p.name for p in app._output_dir.glob("*.wav")]
            return super().transcribe(samples, **kw)

    app = make_app(transcriber=Spy())
    sigint_when_recording(app)
    app.run_recording()
    assert len(seen["wavs"]) == 1


def test_no_audio_flag_keeps_no_wav_and_no_temp_files(fake_sd, make_app, tmp_path, monkeypatch):
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    fake_sd(sine(2.0))
    app = make_app(save_audio=False)
    sigint_when_recording(app)
    app.run_recording()
    assert list(app._output_dir.glob("*.wav")) == []
    assert len(list(app._output_dir.glob("*.md"))) == 1
    assert "**Audio:**" not in next(app._output_dir.glob("*.md")).read_text()
    assert list(tmp.iterdir()) == []


def test_too_short_recording_leaves_nothing_behind(fake_sd, make_app, capsys):
    fake_sd(sine(0.2))
    app = make_app()
    sigint_when_recording(app)
    app.run_recording()
    assert list(app._output_dir.iterdir()) == []
    assert "too short" in capsys.readouterr().out


def test_transcription_failure_keeps_the_audio_and_says_how_to_retry(fake_sd, make_app, capsys):
    fake_sd(sine(2.0))
    app = make_app(transcriber=FakeTranscriber(error=RuntimeError("Transcription failed: boom")))
    sigint_when_recording(app)
    with pytest.raises(SystemExit) as e:
        app.run_recording()
    assert e.value.code == 4
    wav = next(app._output_dir.glob("*.wav"))
    out = capsys.readouterr().out
    assert "boom" in out and f"--from-file {wav}" in out
    assert list(app._output_dir.glob("*.md")) == []


def test_aborting_with_a_second_ctrl_c_keeps_the_audio(fake_sd, make_app, capsys):
    fake_sd(sine(2.0))
    app = make_app(transcriber=FakeTranscriber(error=KeyboardInterrupt()))
    sigint_when_recording(app)
    with pytest.raises(SystemExit) as e:
        app.run_recording()
    assert e.value.code == 130
    assert len(list(app._output_dir.glob("*.wav"))) == 1
    assert "--from-file" in capsys.readouterr().out


def test_mic_failure_exits_1_without_leaving_a_wav(fake_sd, make_app):
    fake_sd(sine(1.0), native_rate=48000, reject_rates=(16000, 48000))
    app = make_app()
    with pytest.raises(SystemExit) as e:
        app.run_recording()
    assert e.value.code == 1
    assert list(app._output_dir.iterdir()) == []


def test_two_recordings_in_the_same_second_do_not_overwrite(fake_sd, make_app, monkeypatch):
    class FrozenDT(v.datetime):
        @classmethod
        def now(cls, tz=None):
            return v.datetime(2026, 1, 2, 3, 4, 5)

    monkeypatch.setattr(v, "datetime", FrozenDT)
    for _ in range(2):
        fake_sd(sine(1.0))
        app = make_app()
        sigint_when_recording(app)
        app.run_recording()
    names = sorted(p.name for p in app._output_dir.iterdir())
    assert names == ["2026-01-02_03-04-05.md", "2026-01-02_03-04-05.wav",
                     "2026-01-02_03-04-05_1.md", "2026-01-02_03-04-05_1.wav"]


# ---- --from-file ------------------------------------------------------------

def test_from_file_transcribes_any_wav_and_names_output_after_it(make_app, tmp_path):
    src = write_wav_file(tmp_path / "meeting.wav", np.repeat(sine(2.0, rate=48000), 2), rate=48000, channels=2)
    ft = FakeTranscriber()
    app = make_app(transcriber=ft, out=tmp_path / "notes")
    app.run_from_file(src)
    assert ft.calls[0]["n"] == pytest.approx(32000, abs=3)  # 16 kHz mono by the time it reaches Whisper
    assert [p.name for p in (tmp_path / "notes").iterdir()] == ["meeting.md"]
    assert f"**Audio:** [meeting.wav]({src})" in (tmp_path / "notes" / "meeting.md").read_text()


def test_from_file_next_to_its_output_links_by_name_and_never_overwrites(make_app, tmp_path):
    out = tmp_path / "same"
    out.mkdir()
    src = write_wav_file(out / "call.wav", sine(1.0))
    (out / "call.md").write_text("precious earlier transcript")
    app = make_app(out=out)
    app.run_from_file(src)
    assert (out / "call.md").read_text() == "precious earlier transcript"
    assert "**Audio:** [call.wav](call.wav)" in (out / "call_1.md").read_text()
    assert src.exists()


def test_from_file_errors_exit_1(make_app, tmp_path, capsys):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"nope")
    with pytest.raises(SystemExit) as e:
        make_app().run_from_file(junk)
    assert e.value.code == 1
    assert "not a readable" in capsys.readouterr().out


def test_from_file_too_short(make_app, tmp_path, capsys):
    app = make_app()
    app.run_from_file(write_wav_file(tmp_path / "tiny.wav", sine(0.1)))
    assert list(app._output_dir.iterdir()) == []


# ---- process() ----------------------------------------------------------------

def test_process_labels_speakers_and_shows_both_stages(make_app, capsys):
    segs = [v.Segment(0, 4, "hi there bye now", [v.Word(0, 1, " hi"), v.Word(1, 2, " there"),
                                                 v.Word(2, 3, " bye"), v.Word(3, 4, " now")])]
    app = make_app(transcriber=FakeTranscriber(segs), diarizer=FakeDiarizer([v.Turn(0, 2, 0), v.Turn(2, 4, 1)]))
    r = app.process(np.zeros(4 * 16000, dtype=np.int16))
    assert r.diarized and "**Speaker 1** (00:00): hi there" in r.body and "**Speaker 2** (00:02): bye now" in r.body
    assert app.transcriber.calls[0]["words"] is True  # word timing requested because we diarize
    out = capsys.readouterr().out
    assert "Transcribing" in out and "Identifying speakers" in out and "Found 2 speakers." in out


@pytest.mark.parametrize("diarizer", [FakeDiarizer(error=RuntimeError("model exploded")), FakeDiarizer(turns=[])])
def test_diarization_trouble_never_loses_the_transcript(make_app, diarizer, capsys):
    app = make_app(diarizer=diarizer)
    r = app.process(np.zeros(2 * 16000, dtype=np.int16))
    assert not r.diarized and r.body == "hello world"
    assert "Warning" in capsys.readouterr().out


def test_without_diarizer_word_timing_is_not_requested(make_app):
    app = make_app()
    app.process(np.zeros(16000, dtype=np.int16))
    assert app.transcriber.calls[0]["words"] is False


def test_silence_is_not_saved_as_an_empty_transcript(make_app, tmp_path, capsys):
    app = make_app(transcriber=FakeTranscriber(segments=[]))
    app.run_from_file(write_wav_file(tmp_path / "quiet.wav", np.zeros(32000, dtype=np.int16)))
    assert list(app._output_dir.glob("*.md")) == []
    assert "No speech detected" in capsys.readouterr().out


def test_plain_progress_prints_every_ten_percent_not_every_call(capsys):
    ui = v.PlainUI()
    with ui.progress("Working", 100.0) as p:
        for i in range(1, 101):
            p(float(i))
    lines = [l for l in capsys.readouterr().out.splitlines() if "%" in l]
    assert len(lines) == 10 and lines[-1].strip() == "Working: 100%"


# ---- CLI argument handling ------------------------------------------------------

def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["voice-record", *argv])
    v.main()


def test_speakers_below_one_is_rejected(monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, "--speakers", "0")
    assert e.value.code == 2 and "--speakers must be >= 1" in capsys.readouterr().err


def test_from_file_must_exist(monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        run_main(monkeypatch, "--from-file", "/definitely/not/here.wav")
    assert e.value.code == 2 and "no such file" in capsys.readouterr().err


def test_help_mentions_the_new_options(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, "--help")
    out = capsys.readouterr().out
    for flag in ("--no-audio", "--from-file", "--diarize", "--speakers"):
        assert flag in out
