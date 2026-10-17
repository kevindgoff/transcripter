"""SubRip (.srt) export: formatting, cue building, and strict validation of the output."""
import random
import re
import shutil
import subprocess
import sys

import pytest

import voice_recorder_cli as v
from conftest import FakeDiarizer, FakeTranscriber, sine, write_wav_file
from voice_recorder_cli import LabeledWord, Segment, Turn, Word

TS = r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"


def W(text, start, end, speaker=None):
    return LabeledWord(speaker, start, end, text if text.startswith(" ") else " " + text)


def to_ms(m):
    h, mi, s, ms = (int(x) for x in m)
    assert mi < 60 and s < 60, f"invalid timestamp component in {m}"
    return ((h * 60 + mi) * 60 + s) * 1000 + ms


def parse_srt(text, max_chars=42, max_lines=2):
    """A deliberately strict SubRip parser: any deviation from the format fails the test."""
    assert text == "" or text.endswith("\r\n\r\n"), "file must end with a blank line"
    assert "\n" not in text.replace("\r\n", ""), "every line break must be CRLF"
    cues, prev_end = [], 0
    for n, block in enumerate(text.split("\r\n\r\n")[:-1] if text else [], 1):
        lines = block.split("\r\n")
        assert lines[0] == str(n), f"cue index {lines[0]!r}, expected {n}"
        m = re.fullmatch(f"{TS} --> {TS}", lines[1])
        assert m, f"bad timing line: {lines[1]!r}"
        start, end = to_ms(m.groups()[:4]), to_ms(m.groups()[4:])
        assert end > start, f"cue {n}: end {end} <= start {start}"
        assert start >= prev_end, f"cue {n} overlaps the previous cue"
        body = lines[2:]
        assert 1 <= len(body) <= max_lines and all(body), f"cue {n}: bad text lines {body!r}"
        assert "-->" not in "\n".join(body)
        prev_end = end
        cues.append((start, end, body))
    return cues


# ---- timestamps ----------------------------------------------------------------

@pytest.mark.parametrize("ms,expected", [
    (0, "00:00:00,000"), (1, "00:00:00,001"), (999, "00:00:00,999"), (1000, "00:00:01,000"),
    (59_999, "00:00:59,999"), (60_000, "00:01:00,000"), (3_599_999, "00:59:59,999"),
    (3_600_000, "01:00:00,000"), (36_000_000, "10:00:00,000"), (-50, "00:00:00,000"),
])
def test_timestamp_format(ms, expected):
    assert v.format_srt_timestamp(ms) == expected


def test_rounding_up_to_a_new_minute_never_yields_sixty_seconds():
    """The classic SRT bug: 59.9996 s -> '00:00:60,000'. Integer-ms maths prevents it."""
    assert v.format_srt_timestamp(v.sec_to_ms(59.9996)) == "00:01:00,000"
    assert v.format_srt_timestamp(v.sec_to_ms(3599.9999)) == "01:00:00,000"
    for i in range(0, 200_000, 7):
        assert re.fullmatch(TS, v.format_srt_timestamp(v.sec_to_ms(i / 1000 + 0.0004)))
        to_ms(re.fullmatch(TS, v.format_srt_timestamp(v.sec_to_ms(i / 1000 + 0.0004))).groups())


# ---- line wrapping -------------------------------------------------------------

def test_wrap_short_text_stays_on_one_line():
    assert v.wrap_caption("Hello there.") == ["Hello there."]
    assert v.wrap_caption("  spaced   out  ") == ["spaced out"]
    assert v.wrap_caption("") == []


def test_wrap_balances_two_lines():
    text = "This sentence is long enough that it has to wrap onto a second line"
    lines = v.wrap_caption(text)
    assert len(lines) == 2 and " ".join(lines) == text and all(len(l) <= 42 for l in lines)
    assert abs(len(lines[0]) - len(lines[1])) <= 12  # balanced, not "42 chars then 3"


def test_wrap_refuses_what_cannot_fit():
    assert v.wrap_caption("word " * 30) is None            # needs 4 lines
    assert v.wrap_caption("x" * 50) is None                # one unbreakable word wider than a line
    assert v.wrap_caption("a b", max_chars=42, max_lines=1) == ["a b"]


# ---- cue building --------------------------------------------------------------

def test_basic_cue():
    cues = v.build_cues([W("Hello", 1.0, 1.4), W("world.", 1.4, 2.0)])
    assert len(cues) == 1
    assert (cues[0].start_ms, cues[0].end_ms, cues[0].lines) == (1000, 2000, ["Hello world."])


def test_empty_and_blank_words():
    assert v.build_cues([]) == []
    assert v.build_cues([W("  ", 0, 1), W("", 1, 2)]) == []


def test_a_long_pause_starts_a_new_cue():
    cues = v.build_cues([W("one", 0.0, 0.5), W("two", 0.5, 1.0), W("three", 3.0, 3.5)])
    assert [c.lines for c in cues] == [["one two"], ["three"]]


def test_sentence_end_breaks_only_when_the_cue_has_enough_text():
    short = v.build_cues([W("Yes.", 0.0, 0.4), W("I", 0.5, 0.6), W("agree", 0.6, 1.0)])
    assert [c.lines for c in short] == [["Yes. I agree"]]  # "Yes." alone would flash by
    words = [W(t, i * 0.4, i * 0.4 + 0.35) for i, t in enumerate("We should leave soon. Then we can eat".split())]
    long = v.build_cues(words)
    assert [c.lines for c in long] == [["We should leave soon."], ["Then we can eat"]]


def test_cue_never_exceeds_two_lines_of_42_or_seven_seconds():
    words = [W(f"word{i}", i * 0.5, i * 0.5 + 0.45) for i in range(120)]  # steady, no punctuation, no pauses
    cues = v.build_cues(words)
    assert len(cues) > 5
    for c in cues:
        assert len(c.lines) <= 2 and all(len(l) <= 42 for l in c.lines)
        assert c.end_ms - c.start_ms <= 7000


def test_every_word_is_kept_in_order():
    words = [W(f"w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(200)]
    cues = v.build_cues(words)
    assert " ".join(" ".join(c.lines) for c in cues).split() == [f"w{i}" for i in range(200)]


def test_speaker_prefix_only_where_a_turn_starts():
    words = ([W("Hi", 0.0, 0.3, 1), W("there", 0.3, 0.6, 1)]
             + [W("Hello", 0.7, 1.0, 2), W("friend", 1.0, 1.4, 2)]
             + [W("Bye", 1.5, 1.8, 1)])
    cues = v.build_cues(words)
    assert [c.lines for c in cues] == [["Speaker 1: Hi there"], ["Speaker 2: Hello friend"], ["Speaker 1: Bye"]]


def test_a_long_turn_is_labelled_once_not_on_every_cue():
    words = [W(f"w{i}", i * 0.4, i * 0.4 + 0.35, 1) for i in range(60)]
    cues = v.build_cues(words)
    assert len(cues) > 2
    assert cues[0].lines[0].startswith("Speaker 1: ")
    assert not any("Speaker" in line for c in cues[1:] for line in c.lines)


def test_prefix_counts_toward_the_line_limit():
    words = [W(f"abcdefgh{i}", i * 0.3, i * 0.3 + 0.25, 3) for i in range(12)]
    for c in v.build_cues(words):
        assert all(len(l) <= 42 for l in c.lines)


def test_unlabelled_words_get_no_prefix():
    cues = v.build_cues([W("plain", 0, 1), W("text", 1, 2)])
    assert "Speaker" not in cues[0].lines[0]


def test_short_cue_is_held_up_to_one_second_but_never_into_the_next_cue():
    roomy = v.build_cues([W("Hi", 0.0, 0.3), W("later", 5.0, 5.5)])
    assert (roomy[0].start_ms, roomy[0].end_ms) == (0, 1000)
    tight = v.build_cues([W("Hi", 0.0, 0.3), W("again", 1.8, 2.5)], gap_break_ms=1000)
    assert tight[0].end_ms == 1000 and tight[1].start_ms == 1800
    packed = v.build_cues([W("Hi", 0.0, 0.3, 1), W("yo", 0.5, 0.9, 2)])  # next cue starts at 0.5 s
    assert packed[0].end_ms == 500


def test_overlapping_word_times_do_not_produce_overlapping_cues():
    cues = v.build_cues([W("a", 0.0, 1.2, 1), W("b", 1.0, 2.0, 2), W("c", 1.9, 3.0, 1)])
    assert all(a.end_ms <= b.start_ms for a, b in zip(cues, cues[1:]))
    assert all(c.end_ms > c.start_ms for c in cues)


def test_zero_length_words_still_get_visible_cues():
    cues = v.build_cues([W("blip", 4.0, 4.0)])
    assert cues[0].end_ms - cues[0].start_ms >= v.SRT_MIN_SPAN_MS


def test_cues_are_clamped_to_the_audio_length():
    cues = v.build_cues([W("early", 1.0, 1.5), W("late", 9.9, 12.0), W("after", 15.0, 16.0)], max_end_ms=10_000)
    assert [c.lines for c in cues] == [["early"], ["late"]]
    assert cues[-1].end_ms == 10_000


def test_text_that_could_corrupt_the_file_is_neutralised():
    cues = v.build_cues([W("a --> b", 0, 1), W("\n", 1, 2)])
    assert "-->" not in cues[0].lines[0]
    parse_srt(v.render_srt(cues))


@pytest.mark.parametrize("seed", range(40))
def test_random_transcripts_always_produce_a_valid_srt(seed):
    """Property test: whatever Whisper hands us (gaps, overlaps, odd punctuation,
    speaker flips), the file must be valid, ordered and within the limits."""
    rng = random.Random(seed)
    t, words, speaker = 0.0, [], 1
    for _ in range(rng.randint(1, 400)):
        t += rng.choice([0.0, 0.0, 0.05, 0.3, 1.0, 2.5]) - rng.choice([0.0, 0.0, 0.0, 0.08])  # gaps and overlaps
        t = max(t, 0.0)
        dur = rng.uniform(0.0, 1.4)
        text = "".join(rng.choice("abcdefghijklmnopqrstuvwxyzéü") for _ in range(rng.randint(1, 14)))
        text += rng.choice(["", "", "", ",", ".", "?", "!", "…"])
        if rng.random() < 0.1:
            speaker = rng.choice([1, 2, 3])
        words.append(W(text, t, t + dur, speaker if seed % 2 else None))
        t += dur
    cues = v.build_cues(words, max_end_ms=v.sec_to_ms(t) + 500)
    parsed = parse_srt(v.render_srt(cues))
    assert len(parsed) == len(cues) and cues
    for start, end, body in parsed:
        assert end <= v.sec_to_ms(t) + 500
        assert all(len(l) <= 42 for l in body) or len(" ".join(body).split()) == 1
    got = [w for _, _, b in parsed for w in " ".join(b).split() if not (w.startswith("Speaker") or re.fullmatch(r"\d:", w))]
    assert got == [w.text.strip() for w in words]


# ---- rendering / files ---------------------------------------------------------

def test_render_matches_the_srt_layout_exactly():
    cues = [v.Cue(1000, 2500, ["Hello"]), v.Cue(3_661_005, 3_662_000, ["Two", "lines"])]
    assert v.render_srt(cues) == (
        "1\r\n00:00:01,000 --> 00:00:02,500\r\nHello\r\n\r\n"
        "2\r\n01:01:01,005 --> 01:01:02,000\r\nTwo\r\nlines\r\n\r\n"
    )
    assert v.render_srt([]) == ""


def test_file_is_utf8_without_bom_with_crlf_and_no_partial_files(tmp_path):
    p = tmp_path / "c.srt"
    v.write_srt(p, [v.Cue(0, 1500, ["Café — naïve ✓"]), v.Cue(2000, 3000, ["日本語"])])
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    assert parse_srt(raw.decode("utf-8"))[0][2] == ["Café — naïve ✓"]
    assert [x.name for x in tmp_path.iterdir()] == ["c.srt"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_ffmpeg_accepts_the_file_and_sees_every_cue(tmp_path):
    words = [W(f"word{i}", i * 0.6, i * 0.6 + 0.5, 1 + (i // 15) % 2) for i in range(90)]
    cues = v.build_cues(words)
    p = tmp_path / "c.srt"
    v.write_srt(p, cues)
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-f", "webvtt", "-"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stderr == ""
    assert r.stdout.count(" --> ") == len(cues)

    def vtt_ms(t):  # ffmpeg writes MM:SS.mmm below one hour, HH:MM:SS.mmm above
        parts = t.split(":")
        h, m, s = ([0] * (3 - len(parts)) + [float(x) for x in parts])
        return round(((h * 60 + m) * 60 + s) * 1000)

    seen = [(vtt_ms(a), vtt_ms(b)) for a, b in re.findall(r"([\d:]+\.\d{3}) --> ([\d:]+\.\d{3})", r.stdout)]
    assert seen == [(c.start_ms, c.end_ms) for c in cues]  # an independent parser reads exactly our timings


# ---- app wiring ------------------------------------------------------------------

WORDS = [Word(0.0, 0.5, " Hello"), Word(0.5, 1.0, " there."), Word(2.6, 3.1, " Goodbye"), Word(3.1, 3.6, " now.")]
SEG = [Segment(0.0, 3.6, "Hello there. Goodbye now.", WORDS)]  # 1.6 s pause between the two sentences


def test_srt_flag_writes_srt_beside_md_with_same_stem_and_links_it(make_app, tmp_path):
    ft = FakeTranscriber(SEG)
    app = make_app(transcriber=ft, srt=True)
    app.run_from_file(write_wav_file(tmp_path / "talk.wav", sine(5.0)))
    out = app._output_dir
    assert sorted(p.name for p in out.iterdir()) == ["talk.md", "talk.srt"]
    assert ft.calls[0]["words"] is True  # word timing is requested because captions need it
    cues = parse_srt((out / "talk.srt").read_bytes().decode("utf-8"))
    assert [c[2] for c in cues] == [["Hello there."], ["Goodbye now."]]
    assert "**Captions:** [talk.srt](talk.srt)" in (out / "talk.md").read_text()


def test_no_srt_flag_means_no_srt_and_no_word_timing(make_app, tmp_path):
    ft = FakeTranscriber(SEG)
    app = make_app(transcriber=ft, srt=False)
    app.run_from_file(write_wav_file(tmp_path / "talk.wav", sine(5.0)))
    assert list(app._output_dir.glob("*.srt")) == [] and ft.calls[0]["words"] is False
    assert "Captions" not in (app._output_dir / "talk.md").read_text()


def test_srt_with_diarization_labels_speakers(make_app, tmp_path):
    app = make_app(transcriber=FakeTranscriber(SEG), diarizer=FakeDiarizer([Turn(0, 1.8, 0), Turn(1.8, 5, 1)]), srt=True)
    app.run_from_file(write_wav_file(tmp_path / "talk.wav", sine(5.0)))
    cues = parse_srt((app._output_dir / "talk.srt").read_bytes().decode("utf-8"))
    assert [c[2] for c in cues] == [["Speaker 1: Hello there."], ["Speaker 2: Goodbye now."]]


def test_no_speech_means_no_srt(make_app, tmp_path, capsys):
    app = make_app(transcriber=FakeTranscriber(segments=[]), srt=True)
    app.run_from_file(write_wav_file(tmp_path / "quiet.wav", sine(2.0)))
    assert list(app._output_dir.iterdir()) == []
    assert "no speech to caption" in capsys.readouterr().out


def test_recording_with_srt_produces_wav_md_srt_with_one_stem(fake_sd, make_app):
    from test_app import sigint_when_recording
    fake_sd(sine(5.0))
    app = make_app(transcriber=FakeTranscriber(SEG), srt=True)
    sigint_when_recording(app)
    app.run_recording()
    files = sorted(p for p in app._output_dir.iterdir())
    assert sorted(p.suffix for p in files) == [".md", ".srt", ".wav"]
    assert len({p.stem for p in files}) == 1


def test_srt_cues_stay_inside_the_audio(make_app, tmp_path):
    late = [Segment(0, 9, "x", [Word(8.0, 20.0, " runs"), Word(21.0, 22.0, " over")])]
    app = make_app(transcriber=FakeTranscriber(late), srt=True)
    app.run_from_file(write_wav_file(tmp_path / "short.wav", sine(10.0)))
    cues = parse_srt((app._output_dir / "short.srt").read_bytes().decode("utf-8"))
    assert all(end <= 10_000 for _, end, _ in cues)


def test_srt_option_is_documented_in_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["voice-record", "--help"])
    with pytest.raises(SystemExit):
        v.main()
    assert "--srt" in capsys.readouterr().out
