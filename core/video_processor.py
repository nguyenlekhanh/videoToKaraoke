"""Video/audio processing: probe, extract audio, burn subtitles, validate.

Performance fixes (2026-10):
- All ffmpeg calls go through core.ffmpeg_runner (stderr drained on a thread,
  real `-progress` stats) — fixes the pipe-deadlock hang.
- Default x264 preset is `veryfast` (old code hardcoded `medium`), configurable.
- Audio is stream-copied (`-c:a copy`) when the source codec is MP4-compatible
  instead of always re-encoding to AAC.
"""
from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from core.ffmpeg_manager import find_tool
from core.ffmpeg_runner import FfmpegStats, run_ffmpeg, stats_line
from utils.helpers import safe_output_stem, unique_path


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    duration: float
    has_audio: bool
    fps: float = 30.0
    audio_codec: str = ""


PRESETS = ("ultrafast", "veryfast", "fast", "medium")

# audio codecs that MP4 accepts without re-encode
MP4_SAFE_AUDIO = {"aac", "mp3", "ac3", "alac"}

CRF_MAP = {"cao (crf 17)": "17", "medium (crf 20)": "20", "nhẹ (crf 24)": "24"}


def should_copy_audio(audio_codec: str) -> bool:
    return (audio_codec or "").lower() in MP4_SAFE_AUDIO


def validate_input(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Định dạng chưa hỗ trợ: {p.suffix}. Hãy dùng MP4/MKV/MOV/AVI.")
    if p.stat().st_size == 0:
        raise ValueError("File video rỗng (0 byte).")
    return p


SUPPORTED_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".ts", ".flv"}


def probe(path: str) -> VideoInfo:
    ffprobe = find_tool("ffprobe")
    if not ffprobe:
        raise RuntimeError("Không tìm thấy ffprobe. Hãy cài FFmpeg (xem README).")
    cmd = [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"Không đọc được metadata video: {(out.stderr or '')[:500]}")
    try:
        data = json.loads(out.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe trả về dữ liệu lỗi: {exc}") from exc
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise ValueError("File không chứa track video.")
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    width, height = int(v.get("width") or 0), int(v.get("height") or 0)
    duration = float(data.get("format", {}).get("duration") or 0)
    fps = 30.0
    try:
        num, den = (v.get("r_frame_rate") or "30/1").split("/")
        if float(den):
            fps = float(num) / float(den)
    except Exception:
        pass
    return VideoInfo(path=str(path), width=width or 1280, height=height or 720,
                     duration=duration, has_audio=a is not None, fps=fps,
                     audio_codec=(a.get("codec_name") or "" if a else ""))


def extract_audio(
    video: str,
    out_wav: str,
    cancel: threading.Event | None = None,
    on_progress: Callable[[float, str], None] | None = None,
    clip_seconds: float = 0.0,
) -> Path:
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Không tìm thấy FFmpeg. Hãy cài FFmpeg (xem README).")
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", str(video)]
    if clip_seconds and clip_seconds > 0:
        cmd += ["-t", f"{clip_seconds:.1f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-progress", "pipe:1", "-nostats", str(out)]

    def _cb(stats: FfmpegStats, frac: float):
        if on_progress:
            on_progress(frac, f"Trích audio: {stats_line(stats)}")

    total = clip_seconds if clip_seconds and clip_seconds > 0 else 0.0
    if total <= 0:
        try:
            total = probe(video).duration or 0.0
        except Exception:
            total = 0.0
    run_ffmpeg(cmd, total_duration=total, log_tag="extract",
               on_stats=_cb if on_progress else None, cancel=cancel)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("Trích xuất audio thất bại: không tạo được file WAV.")
    return out


def build_render_command(ffmpeg: str, video_in: str | Path, ass_path: str | Path,
                         out: str | Path, crf: str, preset: str,
                         audio_args: list[str], clip_seconds: float = 0.0) -> list[str]:
    """Exact burn-in command. Invariants (covered by tests):
    - the subtitles video filter is ALWAYS applied to the video stream
    - video is ALWAYS re-encoded (never `-c:v copy`: stream copy would
      bypass the filter and silently drop all subtitles)
    - only audio may be stream-copied, and only when MP4-compatible
    """
    cmd = [ffmpeg, "-y", "-i", str(video_in)]
    if clip_seconds and clip_seconds > 0:
        cmd += ["-t", f"{clip_seconds:.1f}"]
    cmd += [
        "-vf", _ass_filter(Path(ass_path)),
        "-c:v", "libx264", "-crf", crf, "-preset", preset, "-pix_fmt", "yuv420p",
        *audio_args, "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats",
        str(out),
    ]
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("subtitles="), "subtitle filter missing from render command"
    assert cmd[cmd.index("-c:v") + 1] != "copy", "video stream copy would bypass subtitles"
    return cmd


def _ass_filter(ass_path: Path) -> str:
    # subtitles filter value escaping on Windows: forward slashes, then
    # escape every char special to the filtergraph parser: \ ' : , ; [ ]
    # A ';' in the path would otherwise split the filtergraph and silently
    # drop the subtitles filter.
    s = str(ass_path.resolve()).replace("\\", "/")
    for ch in ("\\", "'", ":", ",", ";", "[", "]"):
        s = s.replace(ch, "\\" + ch)
    return f"subtitles='{s}'"


def assert_subtitles_filter() -> None:
    """Fail fast if this FFmpeg build cannot burn subtitles (no libass).

    Never silently output a video without subtitles.
    """
    from core.ffmpeg_manager import check_tools

    st = check_tools()
    if not st.ffmpeg:
        raise RuntimeError("Không tìm thấy FFmpeg. Hãy cài FFmpeg (xem README).")
    if not st.has_libass:
        raise RuntimeError(
            "Bản FFmpeg hiện tại KHÔNG có filter 'subtitles' (thiếu libass) "
            f"— render sẽ ra video không có chữ.\n{st.ffmpeg_version}\n"
            "Hãy cài bản full: winget install Gyan.FFmpeg (gói release-full).")


def render_burned(
    video_in: str,
    ass_path: str | Path,
    out_mp4: str | Path,
    quality: str = "medium (crf 20)",
    preset: str = "veryfast",
    on_progress: Callable[[float, str], None] | None = None,
    cancel: threading.Event | None = None,
    clip_seconds: float = 0.0,
    total_duration: float = 0.0,
    style: object | None = None,
) -> Path:
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Không tìm thấy FFmpeg.")
    assert_subtitles_filter()  # fail fast: never render a sub-less video silently
    # validate the ASS BEFORE the expensive encode (structure + real style)
    from core.subtitle_renderer import SubtitleStyle, validate_ass

    try:
        src_info = probe(video_in)
        src_dur = src_info.duration
    except Exception:
        src_info, src_dur = None, 0.0
    ass_text = Path(ass_path).read_text(encoding="utf-8-sig")
    real_style = style if style is not None else SubtitleStyle()
    from core.subtitle_renderer import is_fatal
    w = src_info.width if src_info else 1280
    h = src_info.height if src_info else 720
    problems = validate_ass(ass_text, real_style, video_duration=src_dur, width=w, height=h)
    fatal = [p for p in problems if is_fatal(p)]
    style_warnings = [p for p in problems if p not in fatal]
    if fatal:
        raise RuntimeError("File ASS không hợp lệ, dừng render để tránh video không chữ:\n"
                           + "\n".join(f"- {p}" for p in fatal[:8]))
    crf = CRF_MAP.get(quality, "20")
    if preset not in PRESETS:
        preset = "veryfast"
    out = unique_path(Path(out_mp4))
    out.parent.mkdir(parents=True, exist_ok=True)
    # copy audio when compatible (much faster than AAC re-encode)
    src_audio = src_info.audio_codec if src_info else ""
    audio_args = ["-c:a", "copy"] if should_copy_audio(src_audio) else ["-c:a", "aac", "-b:a", "160k"]
    cmd = build_render_command(ffmpeg, video_in, ass_path, out, crf, preset,
                               audio_args, clip_seconds)

    def _cb(stats: FfmpegStats, frac: float):
        if on_progress:
            on_progress(frac, f"Render: {stats_line(stats)}")

    _, render_log = run_ffmpeg(cmd, total_duration=total_duration or clip_seconds or 0.0,
               log_tag="render", on_stats=_cb if on_progress else None, cancel=cancel)
    if style_warnings:
        try:
            with open(render_log, "a", encoding="utf-8") as f:
                f.write("\n[ASS STYLE WARNINGS — render tiếp tục]\n"
                        + "\n".join(f"- {p}" for p in style_warnings) + "\n")
        except OSError:
            pass
    return validate_output(str(out), expect_min_s=0.5)


def validate_output(path: str, expect_min_s: float = 0.5) -> Path:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Không tạo được file output: {path}")
    if p.stat().st_size < 1024:
        raise RuntimeError("File output quá nhỏ, có thể render lỗi.")
    try:
        info = probe(str(p))
    except Exception as exc:
        raise RuntimeError(f"File output không đọc được: {exc}") from exc
    if info.duration < expect_min_s:
        raise RuntimeError("File output quá ngắn, có thể render lỗi.")
    return p


def default_outputs(video_in: str, out_dir: str | Path) -> dict[str, Path]:
    stem = safe_output_stem(video_in)
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return {
        "mp4": unique_path(d / f"{stem}_subtitled.mp4"),
        "srt": unique_path(d / f"{stem}_subtitled.srt"),
        "json": unique_path(d / f"{stem}_word_timestamps.json"),
        "ass": unique_path(d / f"{stem}_karaoke.ass"),
        "wav": d / f"{stem}_audio16k.wav",
    }


# ---------------------------------------------------------------------------
# Post-render subtitle visibility check (pure stdlib, no PIL/OpenCV).
#
# An FFmpeg exit code of 0 does NOT prove subtitles are visible. So after
# rendering we extract the same frame timestamps from the SOURCE and from the
# OUTPUT and count bright (near-white/yellow) pixels in the subtitle band
# (bottom-center). Burned white/yellow text always adds bright pixels there.
# Verdicts:
#   PASS         — output clearly gains bright pixels at a word timestamp
#   FAIL         — output band has ~zero bright pixels at all samples
#                  (subtitles almost certainly missing)
#   INCONCLUSIVE — bright scene, can't tell automatically; inspect the
#                  preview PNGs saved next to the output.
# ---------------------------------------------------------------------------

def extract_frame_rgb(video: str, t: float) -> tuple[int, int, bytes]:
    """Extract one frame as raw RGB24 bytes. Returns (w, h, data)."""
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Không tìm thấy FFmpeg.")
    info = probe(video)
    cmd = [ffmpeg, "-v", "error", "-ss", f"{max(0.0, t):.2f}", "-i", str(video),
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"Không trích được frame tại {t:.1f}s.")
    n = info.width * info.height * 3
    data = out.stdout[:n]
    if len(data) < n:
        raise RuntimeError(f"Frame tại {t:.1f}s thiếu dữ liệu ({len(data)}/{n} bytes).")
    return info.width, info.height, data


def bright_ratio_in_band(w: int, h: int, data: bytes, thresh: int = 180) -> float:
    """Fraction of bright pixels in the bottom-center subtitle band."""
    x0, x1 = int(w * 0.10), int(w * 0.90)
    y0, y1 = int(h * 0.62), int(h * 0.97)
    bright = 0
    total = 0
    for y in range(y0, y1, 2):  # stride 2 keeps this fast in pure Python
        row = y * w * 3
        for x in range(x0, x1, 2):
            o = row + x * 3
            if data[o] >= thresh and data[o + 1] >= thresh and data[o + 2] >= thresh:
                bright += 1
            total += 1
    return bright / total if total else 0.0


def check_visibility(source: str, output: str, sample_times: list[float],
                    min_gain: float = 0.0008) -> dict:
    """Compare source vs output frames. See module docstring for verdicts."""
    samples: list[dict] = []
    for t in sample_times:
        try:
            sw, sh, sdata = extract_frame_rgb(source, t)
            ow, oh, odata = extract_frame_rgb(output, t)
        except Exception as exc:
            samples.append({"t": t, "error": str(exc)[:120]})
            continue
        if (sw, sh) != (ow, oh):
            samples.append({"t": t, "error": "độ phân giải frame khác nhau"})
            continue
        rs = bright_ratio_in_band(sw, sh, sdata)
        ro = bright_ratio_in_band(ow, oh, odata)
        samples.append({"t": round(t, 2), "source_bright": round(rs, 5),
                        "output_bright": round(ro, 5), "gain": round(ro - rs, 5)})
    ok = [s for s in samples if "gain" in s]
    if not ok:
        verdict = "INCONCLUSIVE"
    elif any(s["gain"] >= min_gain for s in ok):
        verdict = "PASS"
    elif all(s["output_bright"] <= 0.0002 for s in ok):
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "samples": samples}
def band_color_counts(w: int, h: int, data: bytes) -> tuple[int, int, int]:
    """Count (yellow, white, total) pixels in the subtitle band.

    Yellow = active word (#FFFF00 minus compression wobble): strong red AND
    green with clearly weaker blue. White = any other word: all channels high.
    """
    x0, x1 = int(w * 0.10), int(w * 0.90)
    y0, y1 = int(h * 0.62), int(h * 0.97)
    yellow = white = total = 0
    for y in range(y0, y1, 2):  # stride 2 keeps this fast in pure Python
        row = y * w * 3
        for x in range(x0, x1, 2):
            o = row + x * 3
            r, g, b = data[o], data[o + 1], data[o + 2]
            total += 1
            if r >= 180 and g >= 180 and b >= 180:
                white += 1
            elif r >= 140 and g >= 140 and (r - b) >= 50 and (g - b) >= 50:
                yellow += 1
    return yellow, white, total


def check_karaoke_colors(source: str, output: str, sample_times: list[float],
                         min_pixels: int = 20) -> dict:
    """Prove the YELLOW highlight exists (not just 'some bright pixels').

    For each word-mid sample: the output band must gain yellow pixels the
    source band does not have. Verdicts:
      PASS         — yellow found where a word is being spoken
      FAIL         — white text visible but ZERO yellow anywhere: the highlight
                     tags are broken even though text renders
      INCONCLUSIVE — no text visible at all (visibility check owns that case)
                     or source band already yellowish
    """
    samples: list[dict] = []
    for t in sample_times:
        try:
            sw, sh, sdata = extract_frame_rgb(source, t)
            ow, oh, odata = extract_frame_rgb(output, t)
        except Exception as exc:
            samples.append({"t": t, "error": str(exc)[:120]})
            continue
        if (sw, sh) != (ow, oh):
            samples.append({"t": t, "error": "độ phân giải frame khác nhau"})
            continue
        sy, sw_, _ = band_color_counts(sw, sh, sdata)
        oy, ow_, _ = band_color_counts(ow, oh, odata)
        samples.append({"t": round(t, 2), "src_yellow": sy, "out_yellow": oy,
                        "src_white": sw_, "out_white": ow_})
    ok = [s for s in samples if "out_yellow" in s]
    if not ok:
        verdict = "INCONCLUSIVE"
    elif any(s["out_yellow"] >= min_pixels and s["out_yellow"] > s["src_yellow"] + min_pixels for s in ok):
        verdict = "PASS"
    elif any(s["out_white"] >= 200 for s in ok) and all(s["out_yellow"] == 0 for s in ok):
        verdict = "FAIL"  # text clearly renders, but nothing ever turns yellow
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "samples": samples}


def save_preview_frames(video: str, times: list[float], out_dir: str | Path,
                        stem: str = "preview") -> list[Path]:
    """Save PNG previews at given timestamps for human visual inspection."""
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Không tìm thấy FFmpeg.")
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i, t in enumerate(times):
        p = d / f"{stem}_{i:02d}_{t:.1f}s.png"
        cmd = [ffmpeg, "-y", "-v", "error", "-ss", f"{max(0.0, t):.2f}",
               "-i", str(video), "-frames:v", "1", str(p)]
        out = subprocess.run(cmd, capture_output=True, timeout=60)
        if out.returncode == 0 and p.exists():
            saved.append(p)
    return saved


def pick_sample_times(segments: list, total_dur: float, n: int = 3) -> list[float]:
    """Mid-word timestamps (where the yellow highlight must be visible)."""
    times: list[float] = []
    for s in segments:
        for w in getattr(s, "words", []) or []:
            t = (w.start + w.end) / 2
            if 0 <= t < (total_dur or float("inf")):
                times.append(t)
    if not times:
        return [total_dur / 2 if total_dur and total_dur > 0 else 1.0]
    step = max(1, len(times) // n)
    return sorted(times[::step][:n])
