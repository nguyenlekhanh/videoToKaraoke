"""Pure-Python ASS regression tests (always run, no ffmpeg needed)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.subtitle_renderer import (  # noqa: E402
    SubtitleStyle,
    generate_ass,
    validate_ass,
    write_ass,
)
from core.video_processor import _ass_filter, pick_sample_times  # noqa: E402
from models.types import Segment, Word  # noqa: E402


def _segs():
    return [Segment(1.0, 3.0, "xin chào bạn",
                    [Word("xin", 1.0, 1.6), Word("chào", 1.6, 2.3), Word("bạn", 2.3, 3.0)])]


def test_write_ass_has_no_bom(tmp_path=None):
    import tempfile
    p = Path(tempfile.mkdtemp()) / "t.ass"
    write_ass(p, _segs(), SubtitleStyle(), 640, 360)
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "BOM breaks some libass builds -> silent no-sub video"
    assert raw.decode("utf-8").startswith("[Script Info]")


def test_validate_ass_ok():
    ass = generate_ass(_segs(), SubtitleStyle(), 640, 360)
    assert validate_ass(ass, SubtitleStyle(), video_duration=10.0, width=640, height=360) == []


def test_dialogues_each_on_own_line():
    # regression: missing trailing newline collapsed all events into one line
    # -> libass showed nothing while ffmpeg exited 0
    from core.subtitle_renderer import build_dialogues
    style = SubtitleStyle()
    segs = _segs()
    ass = generate_ass(segs, style, 640, 360)
    expected = len(build_dialogues(segs, style))
    lines = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(lines) == expected and expected > 1
    assert ass.count("Dialogue:") == expected


def test_validate_ass_catches_collapsed_lines():
    style = SubtitleStyle()
    ass = generate_ass(_segs(), style, 640, 360)
    collapsed = ass.replace("\nDialogue:", "Dialogue:")  # simulate the old bug
    assert collapsed.count("Dialogue:") > 1
    probs = validate_ass(collapsed, style, video_duration=10.0)
    assert any("vỡ định dạng dòng" in p for p in probs)


def test_validate_ass_catches_missing_subtitles():
    style = SubtitleStyle()
    assert validate_ass("", style) != []
    assert any("Dialogue" in p for p in validate_ass("[Script Info]\n[Events]\n", style))
    # all events outside the video -> would render no visible text
    ass = generate_ass(_segs(), style, 640, 360)
    probs = validate_ass(ass, style, video_duration=0.5)
    assert any("ngoài video" in p or "trong thời lượng" in p for p in probs)


def test_validate_ass_catches_bad_style():
    ass = generate_ass(_segs(), SubtitleStyle(), 640, 360)
    tiny = SubtitleStyle(font_size=4)
    assert any("nhỏ" in p for p in validate_ass(ass, tiny, height=1080))
    huge_margin = SubtitleStyle(margin_v=900)
    assert any("MarginV" in p for p in validate_ass(ass, huge_margin, height=720))


def test_highlight_tag_present():
    style = SubtitleStyle(primary="#FFFFFF", highlight="#FFFF00")
    ass = generate_ass(_segs(), style, 640, 360)
    assert "&H0000FFFF" in ass  # yellow in ASS BGR order
    assert validate_ass(ass, style, video_duration=10.0) == []


def test_ass_filter_escaping():
    f = _ass_filter(Path(r"C:\vid eo\test;1\sub karaoke.ass"))
    assert f.startswith("subtitles='") and f.endswith("'")
    assert "\\;" in f, "unescaped ';' would split the filtergraph and drop subtitles"
    assert " " in f, "spaces must survive inside the quoted filter value"


def test_pick_sample_times_inside_duration():
    times = pick_sample_times(_segs(), total_dur=10.0, n=3)
    assert len(times) == 3
    assert all(1.0 <= t <= 3.0 for t in times)
    # clip mode: only times inside the clip
    assert pick_sample_times(_segs(), total_dur=1.2, n=3) == [0.6]
    assert pick_sample_times([], 10.0) == [5.0]
    assert pick_sample_times([], 0.0) == [1.0]
