"""Terminal UI: level/spectrum maths, rendered frames, progress bars, TTY detection."""
import io
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import voice_recorder_cli as v
from conftest import FakeTranscriber, sine, write_wav_file

rich_console = pytest.importorskip("rich.console")
from rich.console import Console  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")


def strip_ansi(s):
    return ANSI.sub("", s)


def capture_console(width=100):
    buf = io.StringIO()
    return Console(file=buf, force_terminal=True, width=width, color_system="truecolor", legacy_windows=False), buf


def tone(freq, n=2048, amp=0.5, rate=16000):
    t = np.arange(n) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---- level maths -------------------------------------------------------------------

@pytest.mark.parametrize("x,db", [(1.0, 0.0), (0.5, -6.02), (0.1, -20.0), (0.01, -40.0)])
def test_dbfs_known_values(x, db):
    assert v.to_dbfs(x) == pytest.approx(db, abs=0.01)


def test_dbfs_is_floored_not_minus_infinity():
    assert v.to_dbfs(0.0) == -60.0 and v.to_dbfs(1e-9) == -60.0 and v.to_dbfs(-1.0) == -60.0


def test_level_fraction_maps_the_meter_range():
    assert v.level_fraction(-60) == 0.0 and v.level_fraction(0) == 1.0 and v.level_fraction(-30) == 0.5
    assert v.level_fraction(-90) == 0.0 and v.level_fraction(6) == 1.0  # clamped both ways


# ---- spectrum ----------------------------------------------------------------------

def test_spectrum_of_silence_is_flat_zero():
    assert v.spectrum_bands(np.zeros(2048, np.float32)) == [0.0] * 16


def test_spectrum_needs_enough_samples():
    assert v.spectrum_bands(tone(1000, n=100)) == [0.0] * 16


def test_spectrum_puts_a_tone_in_the_right_band():
    edges = np.geomspace(100.0, 7000.0, 17)
    for freq in (200, 500, 1000, 2500, 5000):
        bands = v.spectrum_bands(tone(freq))
        expected = int(np.searchsorted(edges, freq, side="right") - 1)
        assert int(np.argmax(bands)) == expected, f"{freq} Hz landed in band {np.argmax(bands)}, expected {expected}"
        assert max(bands) > 0.9
        others = [b for i, b in enumerate(bands) if abs(i - expected) > 1]
        assert max(others) < 0.5


def test_spectrum_values_are_bounded_and_scale_with_loudness():
    loud, quiet = v.spectrum_bands(tone(1000, amp=0.9)), v.spectrum_bands(tone(1000, amp=0.001))
    assert all(0.0 <= b <= 1.0 for b in loud + quiet)
    assert max(loud) > max(quiet)
    noise = np.random.default_rng(0).uniform(-0.3, 0.3, 2048).astype(np.float32)
    assert sum(b > 0.3 for b in v.spectrum_bands(noise)) >= 12  # broadband: most bands lit


# ---- colours / drawing ---------------------------------------------------------------

def test_gradient_runs_green_to_red_and_is_valid_hex():
    assert v.gradient_color(0.0) == "#00d26a" and v.gradient_color(1.0) == "#ff3b30"
    assert v.gradient_color(-5) == v.gradient_color(0) and v.gradient_color(5) == v.gradient_color(1)
    reds = [int(v.gradient_color(i / 20)[1:3], 16) for i in range(21)]
    assert reds[0] < reds[12] <= 255 and reds[-1] == 255
    assert all(re.fullmatch(r"#[0-9a-f]{6}", v.gradient_color(i / 50)) for i in range(51))


def test_header_shows_clock_and_blinks():
    on, off = v.render_header(125, "Fake Mic", True), v.render_header(125, "Fake Mic", False)
    assert on.plain.startswith("● REC") and off.plain.startswith("○ REC")
    assert "02:05" in on.plain and "Fake Mic" in on.plain
    assert "01:02:05" in v.render_header(3725).plain


def test_meter_fills_in_proportion():
    for frac in (0.0, 0.25, 0.5, 1.0):
        t = v.render_meter(frac, -30.0, width=40)
        assert t.plain.count("█") == round(frac * 40) and t.plain.count("█") + t.plain.count("░") == 40
    assert "-30.0 dB" in v.render_meter(0.5, -30.0).plain
    styles = {str(s.style) for s in v.render_meter(1.0, 0.0).spans}
    assert "#00d26a" in " ".join(styles) or len(styles) > 5  # coloured per cell, not one flat colour


def test_spectrum_drawing_dimensions_and_extremes():
    empty = v.render_spectrum([0.0] * 16, rows=4).plain.split("\n")
    assert len(empty) == 4 and all(line.strip() == "" for line in empty)
    full = v.render_spectrum([1.0] * 16, rows=4).plain.split("\n")
    assert len(full) == 4 and all(line.count("█") == 32 for line in full)
    half = v.render_spectrum([0.5] * 4, rows=4).plain.split("\n")
    assert [l.count("█") for l in half] == [0, 0, 8, 8]  # lower two rows lit
    assert len({len(l) for l in full}) == 1


# ---- RecordingView (one live frame) ---------------------------------------------------

class FakeRec:
    device_name = "Fake Mic"

    def __init__(self, elapsed=5.0, samples=None, silent_for=0.0, dropouts=0, clipped=False):
        s = tone(1000) if samples is None else samples
        peak = float(np.abs(s).max())
        self._snap = v.LevelSnapshot(elapsed, float(np.sqrt((s ** 2).mean())), peak, s, silent_for, dropouts, clipped)

    def snapshot(self):
        return self._snap

    def elapsed(self):
        return self._snap.elapsed


def frame_text(rec, view=None):
    console, buf = capture_console()
    console.print((view or v.RecordingView(rec))())
    return strip_ansi(buf.getvalue())


def test_frame_shows_clock_meter_spectrum_and_hint():
    out = frame_text(FakeRec(elapsed=65.0))
    assert "REC" in out and "01:05" in out and "Fake Mic" in out
    assert "Level" in out and "dB" in out and "█" in out
    assert "Press Ctrl+C to stop" in out
    assert "No input signal" not in out and "Clipping" not in out


def test_clock_switches_to_hours_in_a_long_meeting():
    assert "01:23:20" in frame_text(FakeRec(elapsed=5000.0))


def test_silence_warning_appears_after_three_seconds():
    quiet = np.zeros(2048, np.float32)
    assert "No input signal" not in frame_text(FakeRec(samples=quiet, silent_for=2.0))
    out = frame_text(FakeRec(samples=quiet, silent_for=7.4))
    assert "No input signal for 7s" in out and "microphone" in out


def test_clipping_and_dropout_notices():
    assert "Clipping" in frame_text(FakeRec(clipped=True))
    out = frame_text(FakeRec(dropouts=3))
    assert "3 audio dropouts" in out and "1 audio dropout " in frame_text(FakeRec(dropouts=1)) + " "


def test_meter_decays_smoothly_instead_of_dropping_to_zero():
    view = v.RecordingView(FakeRec(samples=tone(1000, amp=0.9)))
    view()
    loud_level, loud_bands = view._level, list(view._bands)
    view.recorder = FakeRec(samples=np.zeros(2048, np.float32))
    view()
    assert 0 < view._level < loud_level and view._level == pytest.approx(loud_level * 0.9)
    assert all(0 < new < old for new, old in zip(view._bands, loud_bands) if old > 0.05)


def test_record_dot_blinks_with_time():
    a, b = frame_text(FakeRec(elapsed=0.1)), frame_text(FakeRec(elapsed=0.6))
    assert "● REC" in a and "○ REC" in b


# ---- progress bars ---------------------------------------------------------------------

def test_rich_progress_shows_label_percent_audio_clock_and_completes():
    console, buf = capture_console()
    ui = v.RichUI(console)
    with ui.progress("Transcribing", 120.0) as p:
        p(30.0)
        p(60.0)
        time.sleep(0.4)  # Rich redraws on a ~10 Hz timer
        mid = strip_ansi(buf.getvalue())
        p(1e9)  # beyond the end is clamped, not an error
    out = strip_ansi(buf.getvalue())
    assert "Transcribing" in out and "100%" in out
    assert "02:00 / 02:00" in out
    assert "50%" in mid  # the intermediate state really was drawn (60 s of 120 s)
    assert "01:00 / 02:00" in mid


def test_rich_progress_does_not_claim_completion_when_interrupted():
    console, buf = capture_console()
    with pytest.raises(KeyboardInterrupt):
        with v.RichUI(console).progress("Transcribing", 100.0) as p:
            p(40.0)
            raise KeyboardInterrupt()
    out = strip_ansi(buf.getvalue())
    assert "40%" in out and "100%" not in out


def test_rich_progress_handles_empty_audio():
    console, buf = capture_console()
    with v.RichUI(console).progress("Transcribing", 0.0) as p:
        p(0.0)
    assert "100%" in strip_ansi(buf.getvalue())


def test_rich_progress_reports_speed_once_work_has_taken_a_moment():
    console, buf = capture_console()
    with v.RichUI(console).progress("Transcribing", 100.0) as p:
        time.sleep(1.3)
        p(50.0)
        time.sleep(0.3)
    assert "x realtime" in strip_ansi(buf.getvalue())


# ---- live recording display ----------------------------------------------------------------

def test_rich_record_runs_until_stopped_then_prints_a_summary():
    console, buf = capture_console()
    stop = threading.Event()
    threading.Timer(0.6, stop.set).start()
    rec = FakeRec(elapsed=42.0)
    t0 = time.monotonic()
    v.RichUI(console).record(rec, stop)
    assert 0.5 < time.monotonic() - t0 < 5
    out = strip_ansi(buf.getvalue())
    assert "Recording stopped at 00:42" in out


def test_rich_ui_messages_are_not_interpreted_as_markup():
    console, buf = capture_console()
    ui = v.RichUI(console)
    ui.info("Saved to [bold]/tmp/x[/bold] [red]")
    ui.warn("odd [/] name")
    out = strip_ansi(buf.getvalue())
    assert "[bold]/tmp/x[/bold] [red]" in out and "Warning: odd [/] name" in out


# ---- choosing a UI -----------------------------------------------------------------------

def test_make_ui_choices(monkeypatch):
    assert isinstance(v.make_ui(no_ui=True), v.PlainUI) and not isinstance(v.make_ui(no_ui=True), v.RichUI)
    assert not isinstance(v.make_ui(), v.RichUI)        # pytest captures stdout: not a terminal
    console, _ = capture_console()
    assert isinstance(v.make_ui(console=console), v.RichUI)
    assert not isinstance(v.make_ui(no_ui=True, console=console), v.RichUI)
    monkeypatch.setenv("TERM", "dumb")
    assert not isinstance(v.make_ui(), v.RichUI)


def test_without_rich_installed_we_fall_back_to_plain(monkeypatch):
    monkeypatch.setitem(sys.modules, "rich.console", None)  # makes `import` raise ImportError
    assert type(v.make_ui(console=object())) is v.PlainUI


def test_help_documents_no_ui(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["voice-record", "--help"])
    with pytest.raises(SystemExit):
        v.main()
    assert "--no-ui" in capsys.readouterr().out


# ---- the app driving a RichUI -------------------------------------------------------------------

def test_app_with_rich_ui_shows_both_progress_bars_and_still_writes_files(make_app, tmp_path):
    from conftest import FakeDiarizer

    console, buf = capture_console()
    segs = [v.Segment(0, 2, "hi there", [v.Word(0, 1, " hi"), v.Word(1, 2, " there")])]
    app = make_app(transcriber=FakeTranscriber(segs), diarizer=FakeDiarizer([v.Turn(0, 2, 0)]), srt=True)
    app.ui = v.RichUI(console)
    app.run_from_file(write_wav_file(tmp_path / "t.wav", sine(3.0)))
    out = strip_ansi(buf.getvalue())
    assert "Transcribing" in out and "Identifying speakers" in out and out.count("100%") >= 2
    assert sorted(p.name for p in app._output_dir.iterdir()) == ["t.md", "t.srt"]


def test_live_recording_with_rich_ui_and_a_real_ctrl_c(fake_sd, make_app):
    from test_app import sigint_when_recording

    console, buf = capture_console()
    fake_sd(sine(3.0))
    app = make_app()
    app.ui = v.RichUI(console)
    sigint_when_recording(app)
    app.run_recording()
    out = strip_ansi(buf.getvalue())
    assert "Using microphone: Fake Mic" in out and "Recording stopped" in out and "Audio saved" in out
    assert sorted(p.suffix for p in app._output_dir.iterdir()) == [".md", ".wav"]


# ---- a real pseudo-terminal ------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(os.name != "posix", reason="needs a POSIX pty")
def test_cli_in_a_real_terminal_draws_progress_and_a_pipe_gets_plain_text(testdata, tmp_path):
    import pty
    import select

    clip = testdata("1-two-speakers-en.wav")
    cmd = [sys.executable, str(Path(v.__file__)), "--from-file", str(clip), "-m", "tiny", "--srt",
           "-o", str(tmp_path / "tty")]

    master, slave = pty.openpty()
    env = {**os.environ, "TERM": "xterm-256color", "COLUMNS": "100", "LINES": "30"}
    proc = subprocess.Popen(cmd, stdin=slave, stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)
    chunks, deadline = [], time.monotonic() + 180
    while time.monotonic() < deadline:
        if select.select([master], [], [], 0.5)[0]:
            try:
                data = os.read(master, 65536)
            except OSError:
                break
            if not data:
                break
            chunks.append(data)
        elif proc.poll() is not None:
            break
    proc.wait(30)
    os.close(master)
    tty_out = b"".join(chunks).decode("utf-8", "replace")
    assert proc.returncode == 0, tty_out
    assert "\x1b[" in tty_out                                       # colour / cursor control: Rich was used
    assert "\x1b[2J" in tty_out                                     # the screen was cleared...
    assert tty_out.index("\x1b[2J") < tty_out.index("Output directory")  # ...before anything was printed
    plain = strip_ansi(tty_out)
    assert "Transcribing" in plain and "100%" in plain and "Transcript saved" in plain
    # the caption preview: a bordered table, drawn after the transcript was saved
    assert "Captions - 1-two-speakers-en.srt" in plain and "╭" in plain and "Start" in plain
    assert plain.index("Transcript saved") < plain.index("Captions - 1-two-speakers-en.srt")

    piped = subprocess.run(cmd[:-1] + [str(tmp_path / "pipe")], capture_output=True, text=True, timeout=180)
    assert piped.returncode == 0
    assert "\x1b" not in piped.stdout                               # no escape codes into pipes / logs
    assert "Transcribing: 100%" in piped.stdout
    assert "Captions (" in piped.stdout and " --> " in piped.stdout  # compact preview for logs, not the table
    assert "╭" not in piped.stdout

    quiet = subprocess.run(cmd[:-1] + [str(tmp_path / "quiet"), "--no-preview"], capture_output=True, text=True, timeout=180)
    assert quiet.returncode == 0 and "Captions (" not in quiet.stdout
    assert (tmp_path / "quiet" / "1-two-speakers-en.srt").exists()  # --no-preview hides it; the file is still written
