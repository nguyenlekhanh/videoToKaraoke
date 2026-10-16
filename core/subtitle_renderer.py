"""ASS subtitle generation with word-by-word yellow flash highlight.

Design (reliable, easy to test):
- Each subtitle line-group becomes a set of ASS Dialogue events.
- For every word interval [w.start, w.end]: one event where ONLY that word
  is wrapped in yellow override tags, the rest stays white (style default).
- Silence gaps between words: one all-white event (or no event for long
  pauses > silence_hide_s so nothing shows during silence).
- Previously spoken words return to white (spec requirement), so this is a
  "flash" highlight, NOT progressive karaoke fill (\\k/\\kf would stay filled).

This avoids \\kf timing pitfalls with Vietnamese combining marks / shaping
and renders identically in libass (mpv/VLC/ffmpeg) and vsfilter.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from models.types import Segment
from utils.helpers import format_ass_time, format_srt_time

ASS_WHITE = "&H00FFFFFF"
ASS_YELLOW = "&H0000FFFF"
ASS_BLACK = "&H00000000"


@dataclass
class SubtitleStyle:
    font: str = "Arial"
    font_size: int = 22
    primary: str = ASS_WHITE      # default text: white
    highlight: str = ASS_YELLOW   # active word: yellow
    outline_colour: str = ASS_BLACK
    outline: int = 2
    shadow: int = 0
    bold: bool = True
    alignment: int = 2            # 2 = bottom-center
    margin_v: int = 40
    max_words_per_line: int = 8
    max_chars_per_line: int = 42
    silence_hide_s: float = 0.6   # gaps longer than this show nothing
    min_event_dur: float = 0.12


def ass_color(tag: str) -> str:
    tag = (tag or "").strip().upper().replace("#", "")
    if re.fullmatch(r"&H[0-9A-F]{8}", tag):
        return tag
    if re.fullmatch(r"[0-9A-F]{6}", tag):  # RRGGBB -> ASS &HAABBGGRR (opaque)
        r, g, b = tag[0:2], tag[2:4], tag[4:6]
        return f"&H00{b}{g}{r}"
    return ASS_WHITE


def wrap_lines(words: list[str], max_words: int, max_chars: int) -> list[str]:
    """Greedy wrap into <=2 lines respecting word and char limits."""
    lines: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and (len(cur) >= max_words or cur_len + add > max_chars):
            lines.append(" ".join(cur))
            cur, cur_len = [w], len(w)
        else:
            cur.append(w)
            cur_len += add
    if cur:
        lines.append(" ".join(cur))
    # keep at most 2 lines per event for readability; merge extras
    while len(lines) > 2:
        lines[-2] = lines[-2] + " " + lines[-1]
        lines.pop()
    return lines


def split_segments(segments: list[Segment], style: SubtitleStyle) -> list[Segment]:
    """Split long segments by punctuation / pause / length limits."""
    out: list[Segment] = []
    for seg in segments:
        words = list(seg.words or [])
        if not words:
            continue
        chunks: list[list] = []
        cur: list = []
        for i, w in enumerate(words):
            cur.append(w)
            ends_punct = w.text.endswith((".", ",", "?", "!", ";", ":", "…", "。", "，"))
            gap = (words[i + 1].start - w.end) if i + 1 < len(words) else 0
            if len(cur) >= style.max_words_per_line or ends_punct or gap >= 1.0:
                # also respect char limit
                if sum(len(x.text) for x in cur) >= style.max_chars_per_line or ends_punct or gap >= 1.0 or len(cur) >= style.max_words_per_line:
                    chunks.append(cur)
                    cur = []
        if cur:
            chunks.append(cur)
        for ch in chunks:
            text = " ".join(w.text for w in ch)
            out.append(Segment(start=ch[0].start, end=ch[-1].end, text=text, words=ch))
    return out


def _event_text(words: list[str], active_idx: int | None, style: SubtitleStyle) -> str:
    lines = wrap_lines(words, style.max_words_per_line, style.max_chars_per_line)
    # find which line holds the active word: map word index -> line
    if active_idx is None:
        return "\\N".join(escape_ass(l) for l in lines)
    # rebuild per-word with highlight by locating word position
    flat: list[str] = []
    for i, w in enumerate(words):
        e = escape_ass(w)
        if i == active_idx:
            flat.append(f"{{\\1c{ass_color(style.highlight)}}}{e}{{\\1c{ass_color(style.primary)}}}")
        else:
            flat.append(e)
    return "\\N".join(_rewrap(flat, words, lines))


def _rewrap(flat: list[str], words: list[str], lines: list[str]) -> list[str]:
    """Re-split highlighted words into the same line breaks as plain text."""
    out, idx = [], 0
    for line in lines:
        n = len(line.split(" "))
        out.append(" ".join(flat[idx:idx + n]))
        idx += n
    if idx < len(flat):
        out[-1] = out[-1] + " " + " ".join(flat[idx:])
    return out


def escape_ass(text: str) -> str:
    return text.replace("{", "（").replace("}", "）")


def build_dialogues(segments: list[Segment], style: SubtitleStyle) -> list[tuple[float, float, str]]:
    """Return (start, end, ass_text) events. Pure function -> easy to test."""
    events: list[tuple[float, float, str]] = []
    for seg in split_segments(segments, style):
        words = seg.words
        texts = [w.text for w in words]
        # lead-in silence: show all-white only if short
        if words[0].start > seg.start:
            gap = words[0].start - seg.start
            if gap <= style.silence_hide_s and gap >= style.min_event_dur:
                events.append((seg.start, words[0].start, _event_text(texts, None, style)))
        for i, w in enumerate(words):
            if w.end - w.start >= style.min_event_dur - 1e-9:
                events.append((w.start, w.end, _event_text(texts, i, style)))
            # gap after word
            nxt = words[i + 1].start if i + 1 < len(words) else seg.end
            gap = nxt - w.end
            if gap >= style.min_event_dur and gap <= style.silence_hide_s:
                events.append((w.end, nxt, _event_text(texts, None, style)))
        # ensure segment end coverage handled by loop above
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def generate_ass(segments: list[Segment], style: SubtitleStyle, width: int = 1280, height: int = 720) -> str:
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{style.font},{style.font_size},{ass_color(style.primary)},"
        f"{ass_color(style.primary)},{ass_color(style.outline_colour)},{ASS_BLACK},"
        f"{-1 if style.bold else 0},0,0,0,100,100,0,0,1,{style.outline},{style.shadow},"
        f"{style.alignment},20,20,{style.margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [header]
    for s, e, text in build_dialogues(segments, style):
        # NOTE: every Dialogue MUST end with a newline. Without it all events
        # collapse into one giant line: libass parses at most a single broken
        # event and the video renders with NO visible subtitles (ffmpeg still
        # exits 0). This was the actual "no subs" root cause.
        lines.append(f"Dialogue: 0,{format_ass_time(s)},{format_ass_time(e)},Default,,0,0,0,,{text}\n")
    return "".join(lines)


def write_ass(path: str | Path, segments: list[Segment], style: SubtitleStyle, width: int = 1280, height: int = 720) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: plain UTF-8, NO BOM. A BOM prefix (utf-8-sig) makes some
    # libass builds fail to recognise the "[Script Info]" header, so zero
    # events are parsed and the video renders with NO subtitles, silently.
    p.write_text(generate_ass(segments, style, width, height), encoding="utf-8")
    return p


def write_srt(path: str | Path, segments: list[Segment]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    idx = 1
    for seg in segments:
        words = [w.text for w in seg.words] or [seg.text]
        lines = wrap_lines(words, 10, 48)
        out.append(str(idx))
        out.append(f"{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}")
        out.extend(lines)
        out.append("")
        idx += 1
    p.write_text("\n".join(out).strip() + "\n", encoding="utf-8")
    return p


def write_words_json(path: str | Path, segments: list[Segment]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "words": [{"text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence} for w in s.words],
        }
        for s in segments
    ]
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def validate_ass(
    ass_text: str,
    style: SubtitleStyle,
    video_duration: float = 0.0,
    width: int = 1280,
    height: int = 720,
) -> list[str]:
    """Pre-render sanity check. Returns problems (empty = OK).

    Catches the "video renders but no subtitles" class of bugs BEFORE
    spending minutes on an encode: empty event list, timestamps outside the
    video, invisible styling (zero opacity, black-on-dark, off-frame text).
    """
    problems: list[str] = []
    if ass_text.startswith("\ufeff"):
        problems.append("File ASS có BOM — một số bản libass sẽ bỏ qua toàn bộ phụ đề.")
    if "[Script Info]" not in ass_text:
        problems.append("Thiếu header [Script Info].")
    if "[Events]" not in ass_text:
        problems.append("Thiếu section [Events].")
    # collapsed-lines guard: every event must sit on its OWN line, otherwise
    # libass sees one malformed event and renders nothing (the real bug seen
    # in 0828_karaoke.ass where all Dialogues shared a single line)
    total_marks = ass_text.count("Dialogue:")
    dialogues = [l for l in ass_text.splitlines() if l.startswith("Dialogue:")]
    if total_marks != len(dialogues):
        problems.append(
            f"File ASS vỡ định dạng dòng: {total_marks} event nhưng chỉ {len(dialogues)} dòng "
            "— libass sẽ không hiện chữ. Hãy tạo lại file ASS.")
        return problems
    if not dialogues:
        problems.append("Không có event Dialogue nào — video render ra sẽ không có chữ.")
        return problems
    inside = 0
    for d in dialogues:
        parts = d.split(",", 9)
        if len(parts) < 10:
            problems.append(f"Event lỗi định dạng: {d[:60]}")
            continue
        try:
            s = _ass_ts_to_s(parts[1])
            e = _ass_ts_to_s(parts[2])
        except ValueError:
            problems.append(f"Timestamp lỗi: {d[:60]}")
            continue
        if e <= s:
            problems.append(f"Event end <= start: {d[:80]}")
        if s < 0:
            problems.append(f"Event timestamp âm: {d[:80]}")
        if video_duration > 0 and s < video_duration:
            inside += 1
        if video_duration > 0 and s >= video_duration:
            problems.append(f"Event nằm ngoài video ({s:.1f}s >= {video_duration:.1f}s).")
    if video_duration > 0 and inside == 0:
        problems.append("Không có event nào nằm trong thời lượng video — sẽ không thấy chữ.")
    # styling visibility
    if style.font_size < max(10, int(height * 0.02)):
        problems.append(f"Cỡ chữ {style.font_size}px quá nhỏ so với chiều cao {height}px.")
    if style.font_size > height * 0.25:
        problems.append(f"Cỡ chữ {style.font_size}px quá lớn, tràn khung hình.")
    if style.margin_v >= height // 2:
        problems.append(f"MarginV {style.margin_v} quá lớn, chữ bị đẩy ra ngoài khung.")
    if style.alignment not in (1, 2, 3, 5, 6, 7, 8, 9):
        problems.append(f"Alignment {style.alignment} không hợp lệ.")
    hi = ass_color(style.highlight)
    if not any(hi in d for d in dialogues):
        problems.append("Không event nào chứa màu highlight vàng — highlight từng từ đã mất.")
    return problems


def _ass_ts_to_s(ts: str) -> float:
    ts = ts.strip()
    h, m, rest = ts.split(":")
    sec, cs = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(cs) / 100.0


FATAL_HINTS = ("Dialogue", "[Events]", "[Script Info]", "BOM", "ngoài video",
               "trong thời lượng", "end <= start", "Timestamp", "Alignment", "highlight")


def is_fatal(problem: str) -> bool:
    """Style/size warnings (tiny font on 4K...) must not block a render;
    structural problems (no events, out-of-range, no highlight) must."""
    return any(h in problem for h in FATAL_HINTS)
