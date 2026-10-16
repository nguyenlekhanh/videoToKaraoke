"""Minimal burn-in diagnostic: synthetic video + real karaoke ASS -> frame proof.

Usage (from project root, venv active):
    python tools/diagnose_burnin.py

Builds a 6s black 640x360 clip, writes a 4-word Vietnamese ASS with the real
renderer (white + yellow per-word events), burns it with the real pipeline
(ultrafast), then for each word-mid timestamp reports yellow/white pixel
counts in source vs output frames and saves preview PNGs to
diagnose_out/previews/ for eyeball verification.

Exit code 0 = yellow highlight proven on frames. Anything else = bug.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ffmpeg_manager import check_tools
from core.subtitle_renderer import SubtitleStyle, validate_ass, write_ass
from core.video_processor import (
    band_color_counts,
    build_render_command,
    check_karaoke_colors,
    extract_frame_rgb,
    probe,
    run_ffmpeg,
    save_preview_frames,
    validate_output,
)
from models.types import Segment, Word


def main() -> int:
    st = check_tools()
    print("ffmpeg:", st.ffmpeg_version or "MISSING")
    if not st.ffmpeg:
        print("FAIL: no ffmpeg"); return 1
    if not st.has_libass:
        print("FAIL: ffmpeg build lacks libass (subtitles filter)"); return 1

    d = Path(tempfile.mkdtemp(prefix="subvn_diag_"))
    out_dir = ROOT / "diagnose_out"
    out_dir.mkdir(exist_ok=True)
    src = d / "src.mp4"
    r = subprocess.run(
        [st.ffmpeg, "-y", "-f", "lavfi", "-i", "color=black:s=640x360:d=6:r=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)],
        capture_output=True, timeout=120)
    assert r.returncode == 0, "cannot build synthetic video"

    segs = [Segment(1.0, 5.0, "xin chào Việt Nam",
                    [Word("xin", 1.0, 2.0), Word("chào", 2.0, 3.0),
                     Word("Việt", 3.0, 4.0), Word("Nam", 4.0, 5.0)])]
    style = SubtitleStyle(font="Arial", font_size=30)
    ass = d / "diag.ass"
    write_ass(ass, segs, style, 640, 360)
    problems = validate_ass(ass.read_text(encoding="utf-8"), style, 6.0, 640, 360)
    print("ASS validation:", problems or "OK")
    assert not problems, problems

    out = out_dir / "diag_subtitled.mp4"
    if out.exists():
        out.unlink()
    cmd = build_render_command(st.ffmpeg, src, ass, out, "24", "ultrafast",
                               ["-c:a", "copy"], 0.0)
    print("[CMD]", " ".join(cmd))
    run_ffmpeg(cmd, total_duration=6.0, log_tag="diag")
    validate_output(str(out))

    times = [1.5, 2.5, 3.5, 4.5]  # mid-word: xin | chào | Việt | Nam
    for t in times:
        sw, sh, sdata = extract_frame_rgb(str(src), t)
        ow, oh, odata = extract_frame_rgb(str(out), t)
        sy, sw_, _ = band_color_counts(sw, sh, sdata)
        oy, ow_, _ = band_color_counts(ow, oh, odata)
        print(f"t={t:.1f}s  source(y={sy},w={sw_})  output(y={oy},w={ow_})")
    res = check_karaoke_colors(str(src), str(out), times)
    print("karaoke verdict:", res["verdict"])
    previews = save_preview_frames(str(out), times, out_dir / "previews", stem="diag")
    print("previews:", [str(p) for p in previews])
    if res["verdict"] != "PASS":
        print("FAIL: yellow highlight NOT proven on frames"); return 1
    print("OK: karaoke subtitles permanently burned in with yellow highlight")
    return 0


if __name__ == "__main__":
    sys.exit(main())
