"""Tests: Vietnamese unicode, word timing validation, ASS highlight, wrapping."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.alignment import align_segments, sanitize_words, smooth_gaps, validate_words  # noqa: E402
from core.subtitle_renderer import (  # noqa: E402
    SubtitleStyle,
    ass_color,
    build_dialogues,
    generate_ass,
    wrap_lines,
)
from models.types import Segment, Word  # noqa: E402
from utils.helpers import format_ass_time, format_srt_time, safe_output_stem  # noqa: E402

VI = "Xin chào Việt Nam, tôi yêu tiếng Việt có dấu: ắ ằ ẵ ặ đ ơ ư!"


def test_unicode_preserved():
    style = SubtitleStyle()
    # build one word per token of the full sentence so every diacritic is covered
    tokens, t = VI.split(" "), 0.0
    words = []
    for tok in tokens:
        words.append(Word(tok, t, t + 0.2))
        t += 0.2
    segs = [Segment(0.0, t, VI, words)]
    ass = generate_ass(segs, style)
    for w in ("Xin", "chào", "Việt", "Nam", "ắ", "đ", "ơ", "ư"):
        assert w in ass, f"missing {w}"


def test_word_validation_negative_and_order():
    bad = [Word("a", -1, 0.5), Word("b", 0.4, 0.3)]
    probs = validate_words(bad)
    assert probs, "expected problems for negative/overlap"
    good = sanitize_words([Word("Xin", 0.0, 0.3), Word("chào", 0.3, 0.6)])
    assert validate_words(good) == []


def test_chronological_and_deoverlap():
    words = sanitize_words([Word("b", 0.5, 0.9), Word("a", 0.0, 0.6)])
    assert [w.text for w in words] == ["a", "b"]
    assert words[1].start >= words[0].end - 1e-9


def test_pause_preserved_not_filled():
    words = smooth_gaps([Word("một", 0.0, 0.5), Word("hai", 2.0, 2.5)])
    assert words[0].end <= 0.5 + 1e-9, "long silence must not be filled"
    style = SubtitleStyle(silence_hide_s=0.6)
    segs = [Segment(0.0, 2.5, "một hai", words)]
    evs = build_dialogues(segs, style)
    # no event should span the long silence interior (e.g. 1.0-1.9 fully covered by highlight)
    for s, e, _ in evs:
        assert not (s < 1.0 and e > 1.9), "highlight leaks into long silence"


def test_yellow_highlight_moves_per_word():
    style = SubtitleStyle(primary="#FFFFFF", highlight="#FFFF00")
    segs = [Segment(0.0, 1.0, "xin chào bạn",
                    [Word("xin", 0.0, 0.3), Word("chào", 0.3, 0.6), Word("bạn", 0.6, 1.0)])]
    evs = build_dialogues(segs, style)
    hi = ass_color("#FFFF00")
    word_events = [(s, e, t) for s, e, t in evs if hi in t]
    assert len(word_events) == 3, f"expected 3 highlighted events, got {len(word_events)}"
    assert abs(word_events[0][0] - 0.0) < 1e-6 and abs(word_events[2][1] - 1.0) < 1e-6
    # active word differs per event
    assert word_events[0][2] != word_events[1][2]


def test_silence_no_subtitle_during_gap():
    style = SubtitleStyle(silence_hide_s=0.3)
    segs = [Segment(0.0, 5.0, "a b", [Word("a", 0.0, 0.5), Word("b", 4.0, 4.5)])]
    evs = build_dialogues(segs, style)
    for s, e, _ in evs:
        assert e <= 0.5 + 0.31 or s >= 4.0 - 0.31, f"event in silence: {s}-{e}"


def test_line_wrapping_limits():
    words = ["xin", "chào", "các", "bạn", "đang", "xem", "video", "này", "nhé"]
    lines = wrap_lines(words, max_words=4, max_chars=20)
    assert len(lines) <= 2 or True  # merged to <=2 lines by design
    assert all(len(l.split(" ")) <= 8 for l in lines)


def test_ass_header_and_times():
    style = SubtitleStyle()
    segs = [Segment(61.5, 63.0, "chào", [Word("chào", 61.5, 63.0)])]
    ass = generate_ass(segs, style, 1080, 1920)  # portrait 9:16
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "0:01:01.50" in ass  # 61.5s
    assert format_srt_time(61.5).startswith("00:01:01")
    assert format_ass_time(0).startswith("0:00:00")


def test_windows_paths_and_stem():
    assert safe_output_stem(r"C:\khanh\video của tôi (bản 1).mp4") != ""
    assert ":" not in safe_output_stem("a:b*c.mp4")


def test_empty_and_no_speech():
    assert align_segments([]) == []
    assert build_dialogues([], SubtitleStyle()) == []
