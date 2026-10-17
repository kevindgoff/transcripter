"""Caption preview: shown after transcribing when a caption file was written; --no-preview turns it off."""
import io
import re
import sys
from argparse import Namespace
from pathlib import Path

import pytest

import voice_recorder_cli as v
from conftest import FakeTranscriber, FakeDiarizer, sine, write_wav_file
from test_srt import SEG

pytest.importorskip("rich")
from rich.console import Console  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def cue(n, text=None, start=None, lines=None):
    start = n * 2000 if start is None else start
    return v.Cue(start, start + 1500, lines or [text or f"caption number {n}"])


def render(cues, height=30, width=100, name="talk.srt", color=True):
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=width, height=height,
                      color_system="truecolor" if color else None)
    v.RichUI(console).show_captions(cues, Path(name))
    raw = buf.getvalue()
    return raw, ANSI.sub("", raw)


# ---- select_preview -----------------------------------------------------------------

def test_everything_that_fits_is_shown():
    cues = [cue(i) for i in range(5)]
    assert v.select_preview(cues, 10) == (cues, 0)


def test_budget_counts_screen_lines_not_cues():
    cues = [cue(i, lines=["line one", "line two"]) for i in range(10)]  # 2 lines each
    shown, hidden = v.select_preview(cues, 7)
    assert len(shown) == 3 and hidden == 7  # 3 x 2 = 6 lines; a 4th would make 8 > 7


def test_at_least_one_cue_is_always_shown():
    shown, hidden = v.select_preview([cue(0, lines=["a", "b"]), cue(1)], 0)
    assert len(shown) == 1 and hidden == 1


def test_empty_input():
    assert v.select_preview([], 10) == ([], 0)


# ---- the rich view ----------------------------------------------------------------------

def test_table_has_title_headers_and_the_exact_timestamps_from_the_file():
    cues = [cue(1, "Hello there."), cue(2, "Goodbye now.", start=3_661_005)]
    raw, text = render(cues)
    assert "Captions - talk.srt (2 cues)" in text
    for header in ("#", "Start", "End", "Caption"):
        assert header in text
    for c in cues:
        assert v.format_srt_timestamp(c.start_ms) in text and v.format_srt_timestamp(c.end_ms) in text
    assert "01:01:01,005" in text and "Hello there." in text and "Goodbye now." in text
    assert "\x1b[" in raw  # it is actually styled


def test_singular_title_for_one_cue():
    assert "(1 cue)" in render([cue(1)])[1]


def test_caption_text_is_not_parsed_as_markup():
    _, text = render([cue(1, "[red]not a style[/red] and [bold]this[/bold]")])
    assert "[red]not a style[/red] and [bold]this[/bold]" in text


def test_two_line_cues_keep_both_lines_in_one_row():
    _, text = render([cue(1, lines=["first line of it", "second line of it"])])
    assert "first line of it" in text and "second line of it" in text
    assert text.count("│ 1 ") + text.count("│  1 ") >= 1  # one numbered row, not two


def test_speaker_labels_are_coloured_per_speaker_and_text_is_intact():
    raw, text = render([cue(1, "Speaker 1: Hi there"), cue(2, "Speaker 2: Hello back"), cue(3, "Speaker 3: third"),
                        cue(4, "Speaker 7: loop")])
    assert "Speaker 1: Hi there" in text and "Speaker 2: Hello back" in text
    def fg_of(label):
        """The foreground colour code applied to `label` (ignoring bold and the row's background shading)."""
        chain = re.search(r"((?:\x1b\[[0-9;]*m)+)" + re.escape(label), raw).group(1)
        params = ";".join(re.findall(r"\x1b\[([0-9;]*)m", chain))
        fg = re.findall(r"(?:^|;)(38;2;\d+;\d+;\d+|38;5;\d+|3[0-7]|9[0-7])(?=;|$)", params)
        assert fg, f"no foreground colour found for {label!r}: {params!r}"
        return fg[-1]

    assert len({fg_of("Speaker 1: "), fg_of("Speaker 2: "), fg_of("Speaker 3: ")}) == 3  # distinct per speaker
    assert fg_of("Speaker 7: ") == fg_of("Speaker 1: ")  # a 6-colour palette wraps around, no crash


def test_long_files_are_cut_to_fit_the_screen_and_say_so():
    cues = [cue(i) for i in range(1, 201)]
    _, text = render(cues, height=30)
    assert "(200 cues)" in text
    shown = len(re.findall(r"caption number \d+", text))
    assert 8 <= shown <= 18 and "caption number 1 " in text + " "
    assert "caption number 200" not in text
    m = re.search(r"\.\.\. (\d+) more cues in talk\.srt", text)
    assert m and int(m.group(1)) == 200 - shown


def test_taller_terminals_show_more():
    cues = [cue(i) for i in range(1, 201)]
    short = len(re.findall(r"caption number", render(cues, height=24)[1]))
    tall = len(re.findall(r"caption number", render(cues, height=60)[1]))
    assert tall > short


def test_narrow_terminals_wrap_instead_of_crashing():
    _, text = render([cue(1, "word " * 30)], width=50)
    assert text.count("word") == 30


def test_no_colour_terminals_still_get_a_readable_table():
    raw, text = render([cue(1, "Plain text")], color=False)
    assert "Plain text" in text and "Start" in text


# ---- the plain view -------------------------------------------------------------------------

def test_plain_view_is_one_line_per_cue_without_escape_codes(capsys):
    v.PlainUI().show_captions([cue(1, "Hello."), cue(2, lines=["two", "lines"])], Path("talk.srt"))
    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert "Captions (2 cues) - talk.srt" in out
    assert "   1  00:00:02,000 --> 00:00:03,500  Hello." in out
    assert "   2  00:00:04,000 --> 00:00:05,500  two / lines" in out


def test_plain_view_is_capped_for_logs(capsys):
    v.PlainUI().show_captions([cue(i) for i in range(1, 301)], Path("long.srt"))
    out = capsys.readouterr().out
    assert out.count("caption number") == v.PlainUI.PLAIN_PREVIEW_LINES
    assert f"... {300 - v.PlainUI.PLAIN_PREVIEW_LINES} more cues in long.srt" in out


# ---- in the app ------------------------------------------------------------------------------

def run_file(make_app, tmp_path, srt=True, preview=True, diarizer=None, ui=None):
    app = make_app(transcriber=FakeTranscriber(SEG), diarizer=diarizer, srt=srt)
    app._preview = preview
    if ui is not None:
        app.ui = ui
    app.run_from_file(write_wav_file(tmp_path / "talk.wav", sine(5.0)))
    return app


def test_preview_is_on_by_default_and_comes_after_both_files_are_saved(make_app, tmp_path, capsys):
    app = v.VoiceRecorder(v.PlainUI())
    assert app._preview is True  # default
    run_file(make_app, tmp_path)
    out = capsys.readouterr().out
    assert "Captions (2 cues) - talk.srt" in out
    assert "Hello there." in out and "Goodbye now." in out
    assert out.index("Transcript saved") < out.index("Captions (2 cues)")  # the last thing on screen


def test_preview_can_be_turned_off(make_app, tmp_path, capsys):
    run_file(make_app, tmp_path, preview=False)
    out = capsys.readouterr().out
    assert "Captions (" not in out and "Hello there." not in out
    assert "Captions saved to" in out  # the file is still written and announced


def test_no_srt_means_no_preview_even_though_it_defaults_to_on(make_app, tmp_path, capsys):
    run_file(make_app, tmp_path, srt=False)
    assert "Captions" not in capsys.readouterr().out


def test_no_speech_means_nothing_to_preview(make_app, tmp_path, capsys):
    app = make_app(transcriber=FakeTranscriber(segments=[]), srt=True)
    app.run_from_file(write_wav_file(tmp_path / "quiet.wav", sine(2.0)))
    out = capsys.readouterr().out
    assert "Captions (" not in out and "no speech to caption" in out


def test_preview_with_rich_ui_and_speakers(make_app, tmp_path):
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=100, height=40)
    run_file(make_app, tmp_path, diarizer=FakeDiarizer([v.Turn(0, 1.8, 0), v.Turn(1.8, 5, 1)]), ui=v.RichUI(console))
    text = ANSI.sub("", buf.getvalue())
    assert "Captions - talk.srt (2 cues)" in text
    assert "Speaker 1: Hello there." in text and "Speaker 2: Goodbye now." in text


def test_preview_shows_what_is_actually_in_the_file(make_app, tmp_path):
    from test_srt import parse_srt

    buf = io.StringIO()
    app = run_file(make_app, tmp_path, ui=v.RichUI(Console(file=buf, force_terminal=True, width=120, height=40)))
    text = ANSI.sub("", buf.getvalue())
    for start, end, body in parse_srt((app._output_dir / "talk.srt").read_bytes().decode("utf-8")):
        assert v.format_srt_timestamp(start) in text and v.format_srt_timestamp(end) in text
        assert " ".join(body) in text.replace("\n", " ") or all(line in text for line in body)


# ---- command line -----------------------------------------------------------------------------------

def test_no_preview_flag_is_documented_and_reaches_the_app(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(v.VoiceRecorder, "setup", lambda self, args: seen.setdefault("args", args))
    monkeypatch.setattr(v.VoiceRecorder, "run", lambda self, args: None)
    for argv, expected in ((["--srt"], False), (["--srt", "--no-preview"], True)):
        seen.clear()
        monkeypatch.setattr(sys, "argv", ["transcripter", "--no-ui", *argv])
        v.main()
        assert seen["args"].no_preview is expected
    monkeypatch.setattr(sys, "argv", ["transcripter", "--help"])
    with pytest.raises(SystemExit):
        v.main()
    assert "--no-preview" in capsys.readouterr().out


def test_setup_turns_the_flag_into_app_state(monkeypatch, tmp_path):
    class Quiet:
        def __init__(self, model_size): pass
        def load_model(self): pass

    monkeypatch.setattr(v, "Transcriber", Quiet)
    for flag, expected in ((False, True), (True, False)):
        app = v.VoiceRecorder(v.PlainUI())
        app.setup(Namespace(output_dir=str(tmp_path), model="tiny", diarize=False, speakers=None,
                            srt=True, no_preview=flag, no_audio=False))
        assert app._preview is expected
