"""End-to-end burn-in regression test with a synthetic video.

Builds a 5s black 640x360 clip (with AAC audio so the audio-copy path is
exercised), a known Vietnamese ASS file, burns it with the real project
pipeline (fast preset), then verifies:
  - output MP4 exists, readable, ~5s
  - pre-render ASS validation passes
  - frame-level visibility check verdict is PASS (white/yellow pixels appear
    in the subtitle band where the source frame had none)
  - preview PNGs are saved

Skipped automatically when ffmpeg/ffprobe (or libass) is unavailable.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ffmpeg_manager import check_tools  # noqa: E402
from core.subtitle_renderer import SubtitleStyle, validate_ass, write_ass  # noqa: E402
from core.video_processor import (  # noqa: E402
    band_color_counts,
    build_render_command,
    check_karaoke_colors,
    check_visibility,
    probe,
    render_burned,
    save_preview_frames,
)
from models.types import Segment, Word  # noqa: E402

ffmpeg = shutil.which("ffmpeg")
ffprobe = shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(not (ffmpeg and ffprobe), reason="ffmpeg/ffprobe not installed")


def _libass_available() -> bool:
    try:
        return bool(check_tools().has_libass)
    except Exception:
        return False

needs_libass = pytest.mark.skipif(not _libass_available(), reason="ffmpeg build lacks libass")


@needs_ffmpeg
@needs_libass
def test_synthetic_burnin_visibility():
    d = Path(tempfile.mkdtemp(prefix="subvn_burnin_"))
    src = d / "src.mp4"
    mk = subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i", "color=black:s=640x360:d=5:r=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)],
        capture_output=True, timeout=120)
    assert mk.returncode == 0, "cannot build synthetic video"
    assert abs(probe(str(src)).duration - 5.0) < 0.6

    segs = [Segment(1.0, 4.0, "xin chào Việt Nam",
                    [Word("xin", 1.0, 1.8), Word("chào", 1.8, 2.8),
                     Word("Việt", 2.8, 3.4), Word("Nam", 3.4, 4.0)])]
    style = SubtitleStyle(font="Arial", font_size=28, primary="#FFFFFF",
                          highlight="#FFFF00", margin_v=30)
    ass = d / "test.ass"
    write_ass(ass, segs, style, 640, 360)
    assert validate_ass(ass.read_text(encoding="utf-8"), style,
                        video_duration=5.0, width=640, height=360) == []

    out = render_burned(str(src), ass, d / "out.mp4", quality="nhẹ (crf 24)",
                        preset="ultrafast", total_duration=5.0, style=style)
    assert Path(out).exists()
    assert abs(probe(str(out)).duration - 5.0) < 0.6

    vis = check_visibility(str(src), str(out), [1.4, 2.3, 3.7])
    assert vis["verdict"] == "PASS", f"subtitles not visible: {vis}"

    # karaoke proof: yellow active word + white other words on output frames,
    # nothing on source frames
    kara = check_karaoke_colors(str(src), str(out), [1.4, 2.3, 3.7])
    assert kara["verdict"] == "PASS", f"yellow highlight not proven: {kara}"

    previews = save_preview_frames(str(out), [2.3], d, stem="preview")
    assert len(previews) == 1 and previews[0].exists()


@needs_ffmpeg
def test_render_command_burns_subtitles():
    """The command must filter video (never stream-copy it) and keep ASS alive."""
    cmd = build_render_command(ffmpeg, "in.mp4", "sub karaoke.ass", "out.mp4",
                               "20", "veryfast", ["-c:a", "copy"], 0.0)
    vf = cmd[cmd.index("-vf") + 1]
    assert "subtitles=" in vf and "sub karaoke.ass" in vf
    assert cmd[cmd.index("-c:v") + 1] == "libx264"  # encoded, not copied
    assert "copy" not in cmd[cmd.index("-c:v"):cmd.index("-c:v") + 2]


def test_band_color_counts_pure():
    # 10x10 all-yellow image -> yellow detected, white not confused
    yellow_px = bytes([255, 255, 0] * 100)
    y, w, n = band_color_counts(10, 10, yellow_px)
    assert y > 0 and w == 0 and n > 0
    white_px = bytes([255, 255, 255] * 100)
    y2, w2, _ = band_color_counts(10, 10, white_px)
    assert w2 > 0 and y2 == 0
    black_px = bytes([0, 0, 0] * 100)
    y3, w3, _ = band_color_counts(10, 10, black_px)
    assert y3 == 0 and w3 == 0
