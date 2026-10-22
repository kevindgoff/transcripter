"""Background (live) transcription: cut finder, governor, worker thread, recorder tap, end-to-end."""
import functools
import os
import signal
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import voice_recorder_cli as v
from conftest import FakeTranscriber, make_fake_sd, sine

SR = v.TARGET_RATE


def speech(seconds, seed=0, amp=0.2):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-amp, amp, int(seconds * SR)) * 32767).astype(np.int16)


def quiet(x, a, b):
    x[int(a * SR):int(b * SR)] = 0


class FakeWhisper:
    """Records every call; returns one segment per call whose text names the call."""

    def __init__(self, delay=0.0, fail_on=None):
        self.calls, self.delay, self.fail_on = [], delay, fail_on
        self.release = None  # optional Event: block each call until set

    def transcribe(self, audio, **kw):
        self.calls.append({"n": len(audio), **kw})
        if self.release is not None:
            self.release.wait(10)
        if self.fail_on == len(self.calls):
            raise ValueError("boom")
        time.sleep(self.delay)
        k = len(self.calls)
        words = [SimpleNamespace(start=1.0, end=2.0, word=f" w{k}")] if kw.get("word_timestamps") else None
        return iter([SimpleNamespace(start=1.0, end=2.0, text=f" chunk{k}", words=words)]), SimpleNamespace()


def make_live(model=None, **kw):
    t = v.Transcriber("tiny")
    t._model = model or FakeWhisper()
    kw.setdefault("min_chunk_s", 2.0)
    kw.setdefault("max_chunk_s", 5.0)
    return t, v.LiveTranscriber(t, **kw)


# ---- find_live_cut ---------------------------------------------------------

def test_too_early_to_cut():
    assert v.find_live_cut(speech(1.9), SR, 2.0, 5.0) is None


def test_waits_while_there_is_no_pause_and_the_cap_is_not_reached():
    assert v.find_live_cut(speech(4.0), SR, 2.0, 5.0) is None


def test_cuts_in_the_middle_of_the_first_pause_after_the_minimum():
    x = speech(8.0)
    quiet(x, 1.0, 1.6)   # before the minimum: ignored
    quiet(x, 3.0, 3.6)   # the one we want
    quiet(x, 4.0, 4.6)   # a later one: not chosen
    cut = v.find_live_cut(x, SR, 2.0, 5.0)
    assert 3.0 <= cut / SR <= 3.6
    assert abs(cut / SR - 3.3) < 0.15


def test_a_blip_shorter_than_the_pause_length_is_not_a_pause():
    x = speech(4.5)
    quiet(x, 3.0, 3.2)  # 0.2 s < 0.4 s
    assert v.find_live_cut(x, SR, 2.0, 5.0) is None


def test_hard_cap_cuts_at_the_quietest_frame_of_the_last_ten_seconds():
    x = speech(7.0, amp=0.3)
    x[int(4.0 * SR):int(4.05 * SR)] //= 20  # one dip, too short to be a pause
    cut = v.find_live_cut(x, SR, 2.0, 5.0)
    assert 4.0 <= cut / SR <= 4.1


def test_a_pause_after_the_cap_does_not_make_the_chunk_longer_than_the_cap():
    x = speech(9.0)
    quiet(x, 6.0, 7.0)
    cut = v.find_live_cut(x, SR, 2.0, 5.0)
    assert cut / SR <= 5.0


def test_noisy_room_pause_is_found_relative_to_speech_level():
    rng = np.random.default_rng(1)
    x = (rng.uniform(-0.3, 0.3, 8 * SR) * 32767).astype(np.int16)
    x[3 * SR:int(3.6 * SR)] = (rng.uniform(-0.02, 0.02, int(0.6 * SR)) * 32767).astype(np.int16)  # hum, not silence
    cut = v.find_live_cut(x, SR, 2.0, 5.0)
    assert 3.0 <= cut / SR <= 3.6


# ---- feed ------------------------------------------------------------------

def test_feed_resamples_to_16k():
    _, live = make_live()
    live.feed(sine(2.0, rate=8000), 8000)
    assert abs(live._total - 2 * SR) <= 2


def test_feed_ignores_audio_once_it_has_given_up():
    _, live = make_live()
    live.feed(speech(1.0), SR)
    live._abandon("test")
    live.feed(speech(1.0), SR)
    assert live._total == SR and live._parts == []


# ---- stepping the worker by hand (no threads) -------------------------------

def test_chunks_cover_everything_once_in_order_with_absolute_times():
    model = FakeWhisper()
    t, live = make_live(model, word_timestamps=True)
    x = speech(14.0)
    for p in (3.3, 6.3, 9.3):
        quiet(x, p - 0.3, p + 0.3)
    live.feed(x, SR)
    while live._step():
        pass
    res = live.finish()
    assert res.chunks == len(model.calls) >= 3
    assert sum(c["n"] for c in model.calls) == res.covered <= len(x)
    starts = [s.start for s in res.segments]
    assert starts == sorted(starts)
    offsets = [0.0]
    for c in model.calls[:-1]:
        offsets.append(offsets[-1] + c["n"] / SR)
    for seg, off in zip(res.segments, offsets):
        assert seg.start == pytest.approx(off + 1.0)
        assert seg.words[0].start == pytest.approx(off + 1.0)
    assert all(c["word_timestamps"] for c in model.calls)


def test_each_chunk_is_prompted_with_the_text_before_it():
    model = FakeWhisper()
    _, live = make_live(model)
    x = speech(14.0)
    for p in (3.3, 6.6, 9.9):
        quiet(x, p - 0.3, p + 0.3)
    live.feed(x, SR)
    while live._step():
        pass
    assert len(model.calls) >= 3
    assert "initial_prompt" not in model.calls[0]
    assert model.calls[1]["initial_prompt"] == "chunk1"
    assert model.calls[2]["initial_prompt"] == "chunk2"   # the previous chunk only, not everything so far


def test_leftover_audio_stays_uncovered():
    _, live = make_live()
    live.feed(speech(3.0), SR)  # past the minimum, no pause, under the cap
    assert live._step() is False
    assert live.finish().covered == 0


# ---- governor ---------------------------------------------------------------

def lagged(lag_s, **kw):
    """A live transcriber with `lag_s` seconds of audio waiting."""
    model = FakeWhisper()
    _, live = make_live(model, degrade_lag_s=20.0, defer_lag_s=40.0, recover_lag_s=8.0, **kw)
    live.feed(speech(lag_s), SR)
    return model, live


def test_normal_decoding_when_not_behind():
    model, live = lagged(6.0)
    live._step()
    assert model.calls[0]["beam_size"] == v.WHISPER_BEAM_SIZE


def test_falling_behind_switches_to_greedy_decoding():
    model, live = lagged(25.0)  # > 20 s degrade, < 40 s defer
    assert live._step() is True
    assert model.calls[0]["beam_size"] == 1
    assert live.finish().degraded is True


def test_it_recovers_the_normal_beam_when_caught_up():
    model, live = lagged(25.0)
    while live._step():
        pass                      # drains down to under the 8 s recover mark
    assert live._degraded is False
    live.feed(speech(6.0), SR)
    live._step()
    assert model.calls[-1]["beam_size"] == v.WHISPER_BEAM_SIZE


def test_far_behind_gives_up_and_releases_the_audio():
    model, live = lagged(45.0)
    assert live._step() is False
    assert model.calls == []
    assert live._parts == []
    res = live.finish()
    assert "behind" in res.stopped_reason and res.covered == 0


def test_dropped_audio_degrades_then_gives_up():
    drops = [0]
    model, live = lagged(6.0, dropouts=lambda: drops[0])
    drops[0] = 1
    assert live._govern() is True and live._degraded           # first strike: greedy
    live.feed(speech(0.1), SR)
    assert live._govern() is True                              # no new drops: carry on
    drops[0] = 2
    assert live._govern() is False                             # second strike: stop
    assert "dropped" in live.finish().stopped_reason


def test_no_recovery_after_a_dropout_even_when_caught_up():
    drops = [1]
    _, live = make_live(dropouts=lambda: drops[0])
    live._govern()
    assert live._degraded
    assert live._govern() is True and live._degraded           # lag is 0, but a strike is on record


def test_a_failing_model_gives_up_without_advancing_and_without_raising():
    model = FakeWhisper(fail_on=1)
    _, live = make_live(model)
    x = speech(8.0)
    quiet(x, 3.0, 3.6)
    live.feed(x, SR)
    live.TICK_S = 0.01
    live.start()
    deadline = time.monotonic() + 5
    while live._reason is None and time.monotonic() < deadline:
        time.sleep(0.02)
    res = live.finish()
    assert "failed" in res.stopped_reason and "boom" in res.stopped_reason
    assert res.covered == 0 and res.segments == []


# ---- the real thread --------------------------------------------------------

def test_worker_thread_transcribes_while_audio_arrives_and_finish_reports_coverage():
    model = FakeWhisper()
    _, live = make_live(model)
    live.TICK_S = 0.01
    live.start()
    x = speech(12.0)
    for p in (3.3, 6.3, 9.3):
        quiet(x, p - 0.3, p + 0.3)
    for i in range(0, len(x), 1024):
        live.feed(x[i:i + 1024], SR)
    deadline = time.monotonic() + 5
    while live._chunks < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    res = live.finish()
    assert res.chunks >= 3 and 0 < res.covered <= len(x)
    assert not live._thread.is_alive()


def test_finish_waits_for_the_chunk_in_progress():
    model = FakeWhisper()
    model.release = threading.Event()
    _, live = make_live(model)
    live.TICK_S = 0.01
    live.start()
    x = speech(6.0)
    quiet(x, 3.0, 3.6)
    live.feed(x, SR)
    deadline = time.monotonic() + 5
    while not live.busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert live.busy
    done = []
    t = threading.Thread(target=lambda: done.append(live.finish()))
    t.start()
    time.sleep(0.2)
    assert not done                    # still waiting on the chunk
    model.release.set()
    t.join(5)
    assert done and done[0].chunks == 1 and done[0].covered > 0


def test_finish_before_start_is_harmless():
    _, live = make_live()
    assert live.finish().covered == 0


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="per-thread nice is Linux-only")
def test_worker_runs_at_lower_priority_but_the_caller_does_not():
    seen = {}
    model = FakeWhisper()
    orig = model.transcribe

    def spy(audio, **kw):
        seen["nice"] = os.getpriority(os.PRIO_PROCESS, threading.get_native_id())
        return orig(audio, **kw)

    model.transcribe = spy
    _, live = make_live(model)
    before = os.getpriority(os.PRIO_PROCESS, threading.get_native_id())
    live.TICK_S = 0.01
    live.start()
    x = speech(6.0)
    quiet(x, 3.0, 3.6)
    live.feed(x, SR)
    deadline = time.monotonic() + 5
    while "nice" not in seen and time.monotonic() < deadline:
        time.sleep(0.01)
    live.finish()
    assert seen["nice"] == before + 10 or seen["nice"] == 19
    assert os.getpriority(os.PRIO_PROCESS, threading.get_native_id()) == before


# ---- recorder tap -----------------------------------------------------------

def run_recorder(fake_sd, tmp_path, sig, on_samples, **sd_kw):
    fake_sd(sig, **sd_kw)
    rec = v.AudioRecorder(tmp_path / "r.wav", on_samples=on_samples)
    rec.start()
    time.sleep(0.3)
    rec.stop()
    return rec


def test_tap_receives_exactly_what_is_written(fake_sd, tmp_path):
    sig, got = sine(2.0), []
    run_recorder(fake_sd, tmp_path, sig, lambda x, rate: got.append((x.copy(), rate)))
    assert all(r == SR for _, r in got)
    assert np.array_equal(np.concatenate([x for x, _ in got]), sig)
    assert np.array_equal(v.read_wav_as_16k_mono(tmp_path / "r.wav"), sig)


def test_tap_reports_the_device_rate_when_16k_is_refused(fake_sd, tmp_path):
    got = []
    run_recorder(fake_sd, tmp_path, sine(1.0, rate=8000), lambda x, rate: got.append(rate),
                 native_rate=8000, reject_rates=(16000,))
    assert set(got) == {8000}


def test_a_crashing_tap_is_dropped_and_recording_is_unaffected(fake_sd, tmp_path):
    sig, calls = sine(2.0), []

    def bad(x, rate):
        calls.append(1)
        raise RuntimeError("tap bug")

    rec = run_recorder(fake_sd, tmp_path, sig, bad)
    assert len(calls) == 1 and rec.error is None
    assert np.array_equal(v.read_wav_as_16k_mono(tmp_path / "r.wav"), sig)


def test_recorder_exposes_dropouts(fake_sd, tmp_path):
    rec = run_recorder(fake_sd, tmp_path, sine(1.0), None, status_every=5)
    assert rec.dropouts > 0 and rec.dropouts == rec.snapshot().dropouts


# ---- Transcriber additions --------------------------------------------------

def test_transcribe_with_offset_shifts_every_time():
    t = v.Transcriber("tiny")
    t._model = FakeWhisper()
    segs = t.transcribe(speech(2.0), word_timestamps=True, offset=100.0)
    assert segs[0].start == pytest.approx(101.0) and segs[0].words[0].end == pytest.approx(102.0)


def test_live_mode_limits_model_threads_and_default_does_not(monkeypatch):
    import types
    seen = []

    class M:
        def __init__(self, size, device, compute_type, **kw):
            seen.append(kw)

    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = M
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    v.Transcriber("tiny").load_model()
    v.Transcriber("tiny", cpu_threads=2).load_model()
    assert seen == [{}, {"cpu_threads": 2}]


def test_set_cpu_threads_reloads_only_when_it_changes(monkeypatch):
    t = v.Transcriber("tiny", cpu_threads=2)
    loads = []
    monkeypatch.setattr(t, "_load_cpu", lambda: loads.append(t.cpu_threads))
    t._model = object()
    t.set_cpu_threads(2)
    t.set_cpu_threads(0)
    t.set_cpu_threads(0)
    assert loads == [0]


def test_live_threads_is_about_half_the_cores(monkeypatch):
    for cores, want in ((1, 1), (2, 1), (4, 2), (8, 4), (64, 4)):
        monkeypatch.setattr(os, "cpu_count", lambda c=cores: c)
        assert v.live_cpu_threads() == want


# ---- app wiring -------------------------------------------------------------

def test_cli_flag_default_is_live_and_no_live_turns_it_off(monkeypatch):
    seen = []

    class App:
        def __init__(self, ui): pass
        def setup(self, args): seen.append(args.no_live)
        def run(self, args): pass

    monkeypatch.setattr(v, "VoiceRecorder", App)
    for argv, want in (([], False), (["--no-live"], True)):
        monkeypatch.setattr(sys, "argv", ["transcripter", "--no-ui"] + argv)
        v.main()
    assert seen == [False, True]


def setup_app(monkeypatch, tmp_path, **args):
    made = []

    class Quiet:
        def __init__(self, model_size, use_gpu=False, **kw):
            made.append(kw)

        def load_model(self): pass

    monkeypatch.setattr(v, "Transcriber", Quiet)
    app = v.VoiceRecorder(v.PlainUI())
    base = dict(output_dir=str(tmp_path), model="tiny", diarize=False, speakers=None, srt=False, no_audio=False)
    from argparse import Namespace
    app.setup(Namespace(**{**base, **args}))
    return app, made


def test_setup_enables_live_with_limited_threads(monkeypatch, tmp_path):
    app, made = setup_app(monkeypatch, tmp_path, no_live=False, from_file=None)
    assert app._live_enabled and made == [{"cpu_threads": v.live_cpu_threads()}]


def test_setup_no_live_and_from_file_keep_all_cores(monkeypatch, tmp_path):
    app, made = setup_app(monkeypatch, tmp_path, no_live=True, from_file=None)
    assert not app._live_enabled and made == [{}]
    app, made = setup_app(monkeypatch, tmp_path, no_live=False, from_file="x.wav")
    assert not app._live_enabled and made == [{}]


# ---- process(): only the uncovered tail is transcribed ----------------------

class TailSpy(FakeTranscriber):
    def __init__(self):
        super().__init__(segments=[v.Segment(0.0, 1.0, "tail", [])])
        self.offsets, self.thread_calls = [], []

    def transcribe(self, samples, word_timestamps=False, progress=None, offset=None, **kw):
        if offset is not None:          # only record when the caller actually passed one
            self.offsets.append(offset)
        return super().transcribe(samples, word_timestamps, progress)

    def set_cpu_threads(self, n):
        self.thread_calls.append(n)


def test_process_transcribes_only_what_live_did_not_cover(make_app):
    spy = TailSpy()
    app = make_app(transcriber=spy)
    live = v.LiveResult(segments=[v.Segment(0.0, 5.0, "early", [])], covered=10 * SR, chunks=1)
    res = app.process(speech(15.0), live)
    assert spy.calls[0]["n"] == 5 * SR and spy.offsets == [10.0]
    assert res.body.split() == ["early", "tail"] and res.duration == pytest.approx(15.0)
    assert spy.thread_calls == []        # small leftover: keep the quiet configuration


def test_process_skips_the_model_when_live_covered_everything(make_app):
    spy = TailSpy()
    app = make_app(transcriber=spy)
    live = v.LiveResult(segments=[v.Segment(0.0, 5.0, "all done", [])], covered=15 * SR)
    res = app.process(speech(15.1), live)
    assert spy.calls == [] and res.body == "all done"


def test_process_uses_all_cores_for_a_big_leftover(make_app):
    spy = TailSpy()
    app = make_app(transcriber=spy)
    app.process(speech(200.0), v.LiveResult(covered=0, stopped_reason="it fell 250 s behind the recording"))
    assert spy.thread_calls == [0] and spy.offsets == []


def test_without_live_process_is_unchanged(make_app):
    spy = TailSpy()
    app = make_app(transcriber=spy)
    app.process(speech(3.0))
    assert spy.calls[0]["n"] == 3 * SR and spy.offsets == [] and spy.thread_calls == []


# ---- end to end: fake mic + fake whisper through run_recording -------------

def sigint_when(app, condition, timeout=15.0):
    original = signal.getsignal(signal.SIGINT)

    def fire():
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if app.recorder is not None and signal.getsignal(signal.SIGINT) is not original and condition():
                os.kill(os.getpid(), signal.SIGINT)
                return
            time.sleep(0.02)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=fire, daemon=True).start()


def run_live_recording(monkeypatch, make_app, sig, wait_for_chunks, sd_kw=None, **live_kw):
    sd = make_fake_sd(sig, gate=True, **(sd_kw or {}))
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    model = FakeWhisper()
    t = v.Transcriber("tiny")
    t._model = model
    app = make_app(transcriber=t)
    app._live_enabled = True
    kw = dict(min_chunk_s=2.0, max_chunk_s=5.0, **live_kw)
    monkeypatch.setattr(v, "LiveTranscriber", functools.partial(v.LiveTranscriber, **kw))
    monkeypatch.setattr(v.LiveTranscriber.func, "TICK_S", 0.01)
    sigint_when(app, lambda: sd.half.is_set() and app._live is not None and app._live._chunks >= wait_for_chunks)
    app.run_recording()
    return app, model


def test_recording_is_mostly_transcribed_before_stop_and_nothing_is_lost_or_doubled(monkeypatch, make_app, capsys):
    sig = speech(24.0)
    for p in range(3, 24, 3):
        quiet(sig, p - 0.3, p + 0.3)
    app, model = run_live_recording(monkeypatch, make_app, sig, wait_for_chunks=2)

    assert len(app._live._segments) >= 2                  # chunks were done by the background worker
    assert sum(c["n"] for c in model.calls) == pytest.approx(len(sig), abs=2)   # each sample sent exactly once
    md = next(app._output_dir.glob("*.md")).read_text(encoding="utf-8")
    texts = [f"chunk{i + 1}" for i in range(len(model.calls))]
    assert all(t in md for t in texts)
    assert [md.index(t) for t in texts] == sorted(md.index(t) for t in texts)   # in recording order
    assert "was transcribed while recording" in capsys.readouterr().out
    wav = next(app._output_dir.glob("*.wav"))
    assert np.array_equal(v.read_wav_as_16k_mono(wav), sig)                      # the WAV is untouched


def test_when_live_gives_up_the_whole_recording_is_done_after_stop(monkeypatch, make_app, capsys):
    sig = speech(12.0)
    quiet(sig, 3.0, 3.6)
    # a degrade/defer limit of 1 s makes it abandon at the first check
    app, model = run_live_recording(monkeypatch, make_app, sig, wait_for_chunks=0,
                                    degrade_lag_s=0.5, defer_lag_s=1.0)
    out = capsys.readouterr().out
    assert "stopped early" in out
    assert sum(c["n"] for c in model.calls) == pytest.approx(len(sig), abs=2)
    assert next(app._output_dir.glob("*.md")).read_text(encoding="utf-8").count("chunk") >= 1


def test_audio_dropouts_reported_by_the_device_make_live_back_off(monkeypatch, make_app, capsys):
    """The recorder's dropout counter must actually reach the governor (two strikes -> give up)."""
    sig = speech(12.0)
    quiet(sig, 3.0, 3.6)
    app, model = run_live_recording(monkeypatch, make_app, sig, wait_for_chunks=0, sd_kw={"status_every": 1})
    assert app.recorder.dropouts > 0
    assert "dropped audio" in capsys.readouterr().out
    assert sum(c["n"] for c in model.calls) == pytest.approx(len(sig), abs=2)   # nothing lost: finished after stop


# ---- regression: audio arriving *while* a chunk is being transcribed --------

class ContinuityWhisper(FakeWhisper):
    """Checks every chunk is exactly the next slice of the source signal: no gap, no overlap."""

    def __init__(self, source, on_first_call=None):
        super().__init__()
        self.source, self.pos, self.on_first_call, self.mismatches = source, 0, on_first_call, []

    def transcribe(self, audio, **kw):
        got = np.rint(np.asarray(audio) * 32768.0).astype(np.int16)
        if not np.array_equal(got, self.source[self.pos:self.pos + len(got)]):
            self.mismatches.append(self.pos)
        self.pos += len(got)
        if self.on_first_call and not self.calls:
            self.on_first_call()   # more audio arrives while this chunk is "being transcribed"
        return super().transcribe(audio, **kw)


def test_audio_that_arrives_during_a_chunk_is_not_lost():
    sig = speech(20.0)
    for p in range(3, 20, 3):
        quiet(sig, p - 0.3, p + 0.3)
    half = len(sig) // 2
    _, live = make_live()

    def more_audio():
        for i in range(half, len(sig), 1024):
            live.feed(sig[i:i + 1024], SR)

    model = ContinuityWhisper(sig, on_first_call=more_audio)
    live._tr._model = model
    live.feed(sig[:half], SR)
    while live._step():
        pass
    res = live.finish()
    assert res.chunks >= 3
    assert model.mismatches == []                      # every chunk was the right slice of the recording
    assert res.covered == model.pos <= len(sig)
    assert live._total == len(sig)
    assert sum(len(p) for p in live._parts) == len(sig) - res.covered   # the rest is still waiting, intact
