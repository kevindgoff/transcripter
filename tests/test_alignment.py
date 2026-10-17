"""Speaker alignment: words -> speakers -> utterances."""
import voice_recorder_cli as v
from voice_recorder_cli import Segment, Turn, Word


def seg(*pairs, start=None, end=None):
    """seg(("hi", 0, 1), ("there", 1, 2)) -> Segment with word timing."""
    words = [Word(s, e, " " + t) for t, s, e in pairs]
    return Segment(start if start is not None else words[0].start, end if end is not None else words[-1].end,
                   " ".join(t for t, _, _ in pairs), words)


def test_speakers_are_numbered_by_first_appearance_not_by_diarizer_id():
    s = [seg(("a", 0, 1), ("b", 1, 2), ("c", 2, 3))]
    turns = [Turn(0, 1, 7), Turn(1, 2, 2), Turn(2, 3, 7)]
    assert [w.speaker for w in v.label_words(s, turns)] == [1, 2, 1]


def test_a_turn_change_in_the_middle_of_a_segment_splits_the_utterance():
    s = [seg(("how", 0.0, 0.5), ("are", 0.5, 1.0), ("you", 1.0, 1.5), ("fine", 1.6, 2.0), ("thanks", 2.0, 2.5))]
    turns = [Turn(0.0, 1.55, 0), Turn(1.55, 2.5, 1)]
    utts = v.assign_speakers(s, turns)
    assert [(u.speaker, u.text) for u in utts] == [(1, "how are you"), (2, "fine thanks")]
    assert utts[1].start == 1.6


def test_a_word_in_a_gap_goes_to_the_nearest_turn():
    s = [seg(("one", 0.0, 1.0), ("gap", 3.0, 3.2), ("two", 9.0, 10.0))]
    turns = [Turn(0, 1, 0), Turn(9, 10, 1)]
    assert [w.speaker for w in v.label_words(s, turns)] == [1, 1, 2]  # 3.1 is 2.1 s from turn 1, 5.9 s from turn 2
    s = [seg(("one", 0.0, 1.0), ("gap", 7.0, 7.2), ("two", 9.0, 10.0))]
    assert [w.speaker for w in v.label_words(s, turns)] == [1, 2, 2]  # 7.1 is nearer the second turn


def test_overlap_wins_when_a_word_spans_two_turns():
    s = [seg(("hmm", 1.0, 2.0))]
    turns = [Turn(0, 1.2, 0), Turn(1.2, 3, 1)]  # 0.2 s vs 0.8 s
    assert v.label_words(s, turns)[0].speaker == 1


def test_no_turns_means_unlabelled_words():
    s = [seg(("a", 0, 1), ("b", 1, 2))]
    assert [w.speaker for w in v.label_words(s, None)] == [None, None]
    assert [w.speaker for w in v.label_words(s, [])] == [None, None]
    assert [u.text for u in v.group_utterances(v.label_words(s, None))] == ["a b"]


def test_segment_without_word_timing_is_spread_across_its_duration():
    s = [Segment(0.0, 10.0, "aaaa bbbb cccc dddd", words=[])]
    words = v.segment_words(s[0])
    assert [w.text.strip() for w in words] == ["aaaa", "bbbb", "cccc", "dddd"]
    assert words[0].start == 0.0 and abs(words[-1].end - 10.0) < 1e-9
    turns = [Turn(0, 5, 0), Turn(5, 10, 1)]  # a change in the middle of a segment still splits it
    assert [(u.speaker, u.text) for u in v.assign_speakers(s, turns)] == [(1, "aaaa bbbb"), (2, "cccc dddd")]


def test_empty_input():
    assert v.label_words([], [Turn(0, 1, 0)]) == []
    assert v.group_utterances([]) == []
    assert v.format_utterances([]) == ""


def test_format_uses_hours_only_when_needed():
    out = v.format_utterances([v.Utterance(1, 59, "early"), v.Utterance(2, 3725, "late")])
    assert "**Speaker 1** (00:59): early" in out
    assert "**Speaker 2** (01:02:05): late" in out


def test_plain_text_joins_all_segments():
    segs = [Segment(0, 1, "one"), Segment(1, 2, ""), Segment(2, 3, "three")]
    assert v.plain_text(segs) == "one three"
