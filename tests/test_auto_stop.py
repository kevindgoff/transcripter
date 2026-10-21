"""Silence-triggered recording stop and its cancel control."""
import io
import os
import sys
import threading

import numpy as np
import pytest

import voice_recorder_cli as v
from rich.console import Console


class FakeRecorder:
    device_name = "Fake Mic"

    def __init__(self, silent_for=0.0):
        self.silent_for = silent_for

    def snapshot(self):
        return v.LevelSnapshot(
            elapsed=25.0, rms=0.0, peak=0.0, samples=np.zeros(2048, dtype=np.float32),
            silent_for=self.silent_for, dropouts=0, clipped=False,
        )

    def elapsed(self):
        return 25.0


class FakeKeys:
    def __init__(self, presses=()):
        self.presses = list(presses)
        self.available = False
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1
        self.available = True

    def cancel_pressed(self):
        return self.presses.pop(0) if self.presses else False

    def stop(self):
        self.stops += 1
        self.available = False


# ---- deterministic timer state machine --------------------------------------------------------

def test_no_countdown_before_20_seconds_of_silence():
    c = v.SilenceStopCountdown()
    assert c.update(19.999, now=100.0).seconds_left is None
    status = c.update(20.0, now=100.0)
    assert status.seconds_left == 30 and not status.should_stop


def test_countdown_expires_30_seconds_after_it_starts():
    c = v.SilenceStopCountdown()
    assert c.update(20.0, now=100.0).seconds_left == 30
    assert c.update(49.0, now=129.0).seconds_left == 1
    status = c.update(50.0, now=130.0)
    assert status.should_stop and status.seconds_left == 0


def test_new_audio_clears_countdown_and_silence_must_reach_20_seconds_again():
    c = v.SilenceStopCountdown()
    assert c.update(20, now=100).seconds_left == 30
    assert c.update(0.1, now=101).seconds_left is None
    assert c.update(19.9, now=120).seconds_left is None
    assert c.update(20, now=121).seconds_left == 30


def test_cancel_stays_cancelled_until_audio_returns_then_rearms():
    c = v.SilenceStopCountdown()
    c.update(20, now=100)
    assert c.cancel().cancelled
    assert c.update(70, now=150).cancelled and not c.update(70, now=150).should_stop
    assert not c.update(0.2, now=151).cancelled
    assert c.update(20, now=171).seconds_left == 30


# ---- key polling state transitions ----------------------------------------------------------------

def test_countdown_starts_and_c_key_cancels_it():
    recorder, c, keys = FakeRecorder(20), v.SilenceStopCountdown(), FakeKeys([False, True])
    first = v.poll_silence_stop(recorder, c, keys, now=1.0)
    assert first.seconds_left == 30 and first.cancel_available
    second = v.poll_silence_stop(recorder, c, keys, now=2.0)
    assert second.cancelled and second.seconds_left is None and not second.cancel_available
    assert keys.starts == 2 and keys.stops >= 1


def test_windows_key_reader_accepts_c_and_skips_extended_keys():
    class FakeMsvcrt:
        def __init__(self):
            self.keys = [chr(0), "H", "C"]

        def kbhit(self):
            return bool(self.keys)

        def getwch(self):
            return self.keys.pop(0)

    reader = v.CancelKeyReader()
    reader._active = reader._windows = True
    reader._msvcrt = FakeMsvcrt()
    assert reader.cancel_pressed()


def test_key_reader_is_a_noop_without_interactive_stdin(monkeypatch):
    class NoTTY:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", NoTTY())
    reader = v.CancelKeyReader()
    reader.start()
    assert not reader.available and not reader.cancel_pressed()
    reader.stop()


@pytest.mark.skipif(os.name != "posix", reason="requires a POSIX pseudo-terminal")
def test_posix_key_reader_accepts_c_immediately_and_restores_terminal_mode(monkeypatch):
    import pty
    import termios

    master, slave = pty.openpty()
    stream = os.fdopen(os.dup(slave), "r", buffering=1)
    original = termios.tcgetattr(slave)
    monkeypatch.setattr(sys, "stdin", stream)
    reader = v.CancelKeyReader()
    try:
        reader.start()
        assert reader.available
        os.write(master, b"c")
        assert reader.cancel_pressed()
    finally:
        reader.stop()
        assert termios.tcgetattr(slave) == original
        stream.close()
        os.close(master)
        os.close(slave)


# ---- display / loop integration ---------------------------------------------------------------------

def test_recording_view_shows_countdown_and_keyboard_button():
    view = v.RecordingView(FakeRecorder(20))
    view.auto_stop_status = v.SilenceStopStatus(seconds_left=27, cancel_available=True)
    output = io.StringIO()
    Console(file=output, force_terminal=True, width=100, color_system=None).print(view())
    text = output.getvalue()
    assert "Auto-stop in 27s" in text
    assert "[C] Cancel countdown" in text


def test_rich_record_loop_stops_when_countdown_expires(monkeypatch):
    from rich.console import Console

    monkeypatch.setattr(v.SilenceStopCountdown, "SILENCE_AFTER_S", 0.02)
    monkeypatch.setattr(v.SilenceStopCountdown, "COUNTDOWN_S", 0.02)
    output = io.StringIO()
    ui = v.RichUI(Console(file=output, force_terminal=True, width=100, height=30, color_system=None))
    stop = threading.Event()
    ui.record(FakeRecorder(100.0), stop)
    assert stop.is_set() and "stopping recording" in output.getvalue().lower()


def test_plain_record_loop_stops_when_countdown_expires(monkeypatch, capsys):
    monkeypatch.setattr(v.SilenceStopCountdown, "SILENCE_AFTER_S", 0.02)
    monkeypatch.setattr(v.SilenceStopCountdown, "COUNTDOWN_S", 0.04)
    monkeypatch.setattr(v.PlainUI, "POLL_S", 0.01)
    stop = threading.Event()
    v.PlainUI().record(FakeRecorder(100.0), stop)
    assert stop.is_set()
    output = capsys.readouterr().out
    assert "Auto-stop in" in output and "stopping recording" in output.lower()


def test_plain_record_loop_leaves_recording_running_after_cancel(monkeypatch, capsys):
    monkeypatch.setattr(v.SilenceStopCountdown, "SILENCE_AFTER_S", 0.02)
    monkeypatch.setattr(v.SilenceStopCountdown, "COUNTDOWN_S", 0.04)
    monkeypatch.setattr(v.PlainUI, "POLL_S", 0.01)

    class CancelOnce(FakeKeys):
        def __init__(self):
            super().__init__(presses=[True])

    monkeypatch.setattr(v, "CancelKeyReader", CancelOnce)
    stop = threading.Event()
    ui = v.PlainUI()
    # The C key cancels the current silence episode; simulate a user pressing
    # Ctrl+C shortly afterwards to finish the recording normally.
    threading.Timer(0.12, stop.set).start()
    ui.record(FakeRecorder(100.0), stop)
    assert stop.is_set()
    assert "countdown cancelled" in capsys.readouterr().out.lower()
