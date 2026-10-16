"""Locate and validate FFmpeg / ffprobe on Windows."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class ToolStatus:
    ffmpeg: str | None
    ffprobe: str | None
    ffmpeg_version: str = ""
    has_libass: bool = False


def find_tool(name: str) -> str | None:
    return shutil.which(name)


def _version(exe: str | None) -> str:
    if not exe:
        return ""
    try:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=10)
        first = (out.stdout or "").splitlines()
        return first[0].strip() if first else ""
    except Exception:
        return ""


def _has_libass(ffmpeg: str | None) -> bool:
    if not ffmpeg:
        return False
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=15)
        text = (out.stdout or "") + (out.stderr or "")
        return "ass" in text.lower() and "subtitles" in text.lower()
    except Exception:
        return False


def check_tools() -> ToolStatus:
    ffmpeg = find_tool("ffmpeg")
    ffprobe = find_tool("ffprobe")
    return ToolStatus(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        ffmpeg_version=_version(ffmpeg),
        has_libass=_has_libass(ffmpeg),
    )


def friendly_install_help() -> str:
    return (
        "Không tìm thấy FFmpeg/ffprobe.\n"
        "Cách cài trên Windows:\n"
        "1) Tải bản full tại https://www.gyan.dev/ffmpeg/builds/ (ffmpeg-release-full.7z)\n"
        "2) Giải nén, thêm thư mục bin vào PATH, mở lại ứng dụng.\n"
        "Hoặc: winget install Gyan.FFmpeg"
    )
