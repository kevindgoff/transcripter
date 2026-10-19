"""Explicit output paths, filename prefixes, and diarized Markdown layout."""
from pathlib import Path
import sys

import numpy as np
import pytest

import voice_recorder_cli as v
from conftest import FakeTranscriber, sine, write_wav_file


def test_default_paths_keep_existing_timestamp_layout(tmp_path):
    out = v.resolve_output_paths(tmp_path, "2026-01-02_03-04-05")
    assert out.wav == tmp_path / "2026-01-02_03-04-05.wav"
    assert out.md == tmp_path / "2026-01-02_03-04-05.md"
    assert out.srt == tmp_path / "2026-01-02_03-04-05.srt"


def test_only_md_path_derives_media_sibling(tmp_path):
    md = tmp_path / "notes" / "design-review.md"
    out = v.resolve_output_paths(tmp_path / "unused", "ignored", md_path=md)
    assert out.md == md
    assert out.wav == md.with_suffix(".wav")
    assert out.srt == md.with_suffix(".srt")


def test_only_media_path_derives_md_sibling(tmp_path):
    audio = tmp_path / "recordings" / "design-review.wav"
    out = v.resolve_output_paths(tmp_path / "unused", "ignored", audio_path=audio)
    assert out.wav == audio
    assert out.md == audio.with_suffix(".md")
    assert out.srt == audio.with_suffix(".srt")


def test_two_explicit_paths_are_independent(tmp_path):
    audio = tmp_path / "media" / "meeting.wav"
    md = tmp_path / "notes" / "summary.md"
    out = v.resolve_output_paths(tmp_path / "unused", "ignored", audio_path=audio, md_path=md)
    assert out.wav == audio and out.md == md
    assert out.srt == md.with_suffix(".srt")


def test_prefix_changes_basenames_but_not_directories(tmp_path):
    audio = tmp_path / "media" / "meeting.wav"
    md = tmp_path / "notes" / "summary.md"
    out = v.resolve_output_paths(tmp_path, "ignored", audio_path=audio, md_path=md, prefix="team-")
    assert out.wav == audio.with_name("team-meeting.wav")
    assert out.md == md.with_name("team-summary.md")
    assert out.srt == md.with_name("team-summary.srt")


def test_prefix_applies_to_default_timestamp_names(tmp_path):
    out = v.resolve_output_paths(tmp_path, "2026-01-02_03-04-05", prefix="standup-")
    assert out.wav.name == "standup-2026-01-02_03-04-05.wav"
    assert out.md.name == "standup-2026-01-02_03-04-05.md"


@pytest.mark.parametrize("prefix", ["../bad", "subdir/", "subdir\\"])
def test_prefix_cannot_smuggle_in_a_directory(prefix, tmp_path):
    with pytest.raises(ValueError, match="filename prefix"):
        v.resolve_output_paths(tmp_path, "meeting", prefix=prefix)


@pytest.mark.parametrize("provided,suffix", [("report.txt", ".md"), ("audio.mp3", ".wav")])
def test_explicit_path_rejects_wrong_extension(provided, suffix, tmp_path):
    bad = tmp_path / provided
    kwargs = {"md_path": bad} if suffix == ".md" else {"audio_path": bad}
    with pytest.raises(ValueError, match=suffix):
        v.resolve_output_paths(tmp_path, "meeting", **kwargs)


def test_extension_is_added_when_explicit_path_omits_it(tmp_path):
    out = v.resolve_output_paths(tmp_path, "ignored", md_path=tmp_path / "notes" / "review")
    assert out.md == tmp_path / "notes" / "review.md"
    assert out.wav == tmp_path / "notes" / "review.wav"


def test_existing_media_or_transcript_never_gets_overwritten(tmp_path):
    audio = tmp_path / "media" / "meeting.wav"
    md = tmp_path / "notes" / "summary.md"
    audio.parent.mkdir()
    md.parent.mkdir()
    md.write_text("precious")
    out = v.resolve_output_paths(tmp_path, "ignored", audio_path=audio, md_path=md)
    assert out.wav == audio.with_name("meeting_1.wav")
    assert out.md == md.with_name("summary_1.md")
    assert out.srt == md.with_name("summary_1.srt")
    assert md.read_text() == "precious"


def test_existing_srt_also_advances_both_names(tmp_path):
    md = tmp_path / "summary.md"
    md.with_suffix(".srt").write_text("precious captions")
    out = v.resolve_output_paths(tmp_path, "summary", md_path=md)
    assert out.md == tmp_path / "summary_1.md"
    assert out.wav == tmp_path / "summary_1.wav"


def test_existing_audio_also_advances_both_names(tmp_path):
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"precious audio")
    out = v.resolve_output_paths(tmp_path, "meeting")
    assert out.wav == tmp_path / "meeting_1.wav"
    assert out.md == tmp_path / "meeting_1.md"


def test_markdown_media_link_is_relative_and_url_encoded(tmp_path):
    md = tmp_path / "notes folder" / "summary.md"
    audio = tmp_path / "media folder" / "meeting 1.wav"
    assert v.markdown_link_target(audio, md) == "../media%20folder/meeting%201.wav"


def test_from_file_uses_md_path_and_prefix_and_links_to_original_media(make_app, tmp_path):
    (tmp_path / "input").mkdir()
    src = write_wav_file(tmp_path / "input" / "meeting.wav", sine(2.0))
    md_path = tmp_path / "notes" / "summary notes.md"
    app = make_app(transcriber=FakeTranscriber(), out=tmp_path / "unused", srt=True)
    app._md_path = md_path
    app._prefix = "team-"
    app.run_from_file(src)
    out_md = md_path.with_name("team-summary notes.md")
    out_srt = out_md.with_suffix(".srt")
    assert out_md.exists() and out_srt.exists()
    assert src.exists()  # --from-file input is never moved or rewritten
    text = out_md.read_text(encoding="utf-8")
    assert "**Audio:** [meeting.wav](../input/meeting.wav)" in text
    assert "**Captions:** [team-summary notes.srt](team-summary%20notes.srt)" in text


def test_cli_accepts_paths_and_prefix(monkeypatch):
    seen = {}
    monkeypatch.setattr(v.VoiceRecorder, "setup", lambda self, args: seen.setdefault("args", args))
    monkeypatch.setattr(v.VoiceRecorder, "run", lambda self, args: None)
    monkeypatch.setattr(sys, "argv", ["transcripter", "--no-ui", "--audio-path", "media.wav",
                                       "--md-path", "notes.md", "--prefix", "client-"])
    v.main()
    args = seen["args"]
    assert args.audio_path == "media.wav" and args.md_path == "notes.md" and args.prefix == "client-"


@pytest.mark.parametrize("flags,expected", [
    (["--no-audio", "--audio-path", "meeting.wav"], "--audio-path cannot be used with --no-audio"),
    (["--from-file", "meeting.wav", "--audio-path", "copy.wav"], "--audio-path cannot be used with --from-file"),
])
def test_incompatible_output_audio_flags_are_rejected(monkeypatch, capsys, flags, expected):
    monkeypatch.setattr(Path, "is_file", lambda self: True)  # pass input-file validation to exercise the conflict
    monkeypatch.setattr(sys, "argv", ["transcripter", "--no-ui", *flags])
    with pytest.raises(SystemExit) as err:
        v.main()
    assert err.value.code == 2
    assert expected in capsys.readouterr().err


def test_diarized_utterances_put_speaker_and_time_on_a_separate_markdown_line():
    result = v.format_utterances([v.Utterance(1, 12.0, "A point."), v.Utterance(2, 65.0, "Agreed.")])
    assert result == "**Speaker 1 (00:12)**\n\nA point.\n\n**Speaker 2 (01:05)**\n\nAgreed."


def test_undiarized_transcript_body_remains_plain(make_app):
    app = make_app(transcriber=FakeTranscriber(), diarizer=None)
    result = app.process(np.zeros(2 * 16000, dtype=np.int16))
    assert result.body == "hello world"
