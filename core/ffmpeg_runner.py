"""Safe FFmpeg subprocess runner.

Two bugs in the old code are fixed here:

1. PIPE DEADLOCK (the likely 2183s hang): ffmpeg logs to stderr. The old
   code passed stderr=PIPE but never read it while the process ran. Once the
   OS pipe buffer (~64KB) fills — e.g. libass spamming "Glyph ... not found"
   warnings per frame for a font without Vietnamese glyphs — ffmpeg blocks on
   write() forever and the GUI waits forever. This runner drains stderr on a
   dedicated thread straight into logs/ffmpeg_<ts>.log, so the pipe can never
   fill up.

2. FAKE PROGRESS: ffmpeg's human progress uses '\\r' (no newlines), so
   readline() on stderr blocks. Instead we pass `-progress pipe:1 -nostats`,
   which emits machine-readable `key=value` lines on STDOUT (newline
   terminated, safe for readline): frame, fps, out_time_ms, speed,
   total_size. Real numbers reach the GUI.
"""
from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from core.ffmpeg_manager import find_tool
from core.stages import log_dir


@dataclass
class FfmpegStats:
    frame: int = 0
    fps: float = 0.0
    out_time_s: float = 0.0
    speed: float = 0.0
    total_size: int = 0
    done: bool = False


def parse_progress_block(block: dict[str, str], stats: FfmpegStats) -> FfmpegStats:
    """Fold one `-progress` key=value block into stats. Pure -> unit tested."""
    try:
        if "frame" in block:
            stats.frame = int(float(block["frame"]))
        if "fps" in block:
            stats.fps = float(block["fps"] or 0)
        if "out_time_ms" in block:
            v = block["out_time_ms"].strip()
            stats.out_time_s = max(0.0, int(v) / 1_000_000) if v not in ("", "N/A") else stats.out_time_s
        if "speed" in block:
            v = (block["speed"] or "").strip().rstrip("x")
            stats.speed = float(v) if v not in ("", "N/A") else 0.0
        if "total_size" in block:
            stats.total_size = int(float(block["total_size"] or 0))
        if block.get("progress") == "end":
            stats.done = True
    except (ValueError, TypeError):
        pass
    return stats


def stats_line(s: FfmpegStats) -> str:
    size_mb = s.total_size / 1_000_000
    return (f"frame {s.frame} | {s.out_time_s:.1f}s | {s.fps:.1f} fps | "
            f"{s.speed:.2f}x | {size_mb:.1f} MB")


def run_ffmpeg(
    args: list[str],
    total_duration: float = 0.0,
    log_tag: str = "ffmpeg",
    on_stats: Callable[[FfmpegStats, float], None] | None = None,
    cancel: threading.Event | None = None,
    stall_timeout: float = 120.0,
    log_stderr: bool = True,
) -> tuple[int, Path]:
    """Run ffmpeg with real progress. Returns (returncode, stderr_log_path).

    - stderr is drained by a thread into a log file (never blocks ffmpeg).
    - stdout carries `-progress` blocks (caller must include -progress pipe:1
      -nostats in args, or pass on_stats=None for plain commands).
    - stall_timeout: abort if no progress block arrives for this long
      (prevents infinite silent hangs); 0 disables.
    """
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg and not (args and args[0]):
        raise RuntimeError("Không tìm thấy FFmpeg.")
    log_path = log_dir() / f"{log_tag}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    # record the EXACT command line first (append mode below) so any render
    # can be reproduced / audited byte-for-byte from the log
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as f:
            f.write("[CMD] " + " ".join(f'"{a}"' if " " in a else a for a in args) + "\n")
    except OSError:
        pass
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if log_stderr else subprocess.DEVNULL,
        text=True, bufsize=1,
    )
    # --- drain stderr on a thread (THE deadlock fix) ---
    def _drain():
        try:
            with open(log_path, "a", encoding="utf-8", errors="replace") as f:
                assert proc.stderr is not None
                for line in proc.stderr:
                    f.write(line)
        except Exception:
            pass

    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    stats = FfmpegStats()
    block: dict[str, str] = {}
    last_progress = time.time()
    rc = -1
    try:
        assert proc.stdout is not None
        # readline is safe here: -progress lines are newline-terminated.
        # If the command has no -progress output, stdout just EOFs at exit.
        while True:
            line = proc.stdout.readline()
            if line:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    block[k.strip()] = v.strip()
                    if k.strip() == "progress":
                        stats = parse_progress_block(block, stats)
                        block = {}
                        last_progress = time.time()
                        if on_stats:
                            frac = (stats.out_time_s / total_duration) if total_duration > 0 else 0.0
                            on_stats(stats, min(0.999, max(0.0, frac)))
                continue
            # no line: check exit / stall / cancel
            rc_poll = proc.poll()
            if rc_poll is not None:
                # consume remainder, then parse trailing block
                rest = proc.stdout.read() or ""
                for rl in rest.splitlines():
                    if "=" in rl:
                        k, v = rl.split("=", 1)
                        block[k.strip()] = v.strip()
                if block:
                    stats = parse_progress_block(block, stats)
                    if on_stats:
                        frac = (stats.out_time_s / total_duration) if total_duration > 0 else 0.0
                        on_stats(stats, min(0.999, max(0.0, frac)))
                rc = rc_poll
                break
            if cancel is not None and cancel.is_set():
                proc.kill()
                raise InterruptedError("Người dùng đã hủy thao tác FFmpeg.")
            if stall_timeout and (time.time() - last_progress) > stall_timeout:
                proc.kill()
                raise TimeoutError(
                    f"FFmpeg không tiến triển quá {stall_timeout:.0f}s "
                    f"(xem {log_path.name}). Có thể filter subtitles/libass bị kẹt.")
            time.sleep(0.1)
    finally:
        try:
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        t.join(timeout=10)
    if isinstance(rc, int) and rc != 0 or proc.returncode not in (0, None):
        code = proc.returncode if proc.returncode is not None else rc
        tail = ""
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-25:])
        except OSError:
            pass
        raise RuntimeError(f"FFmpeg thất bại (mã {code}). Log: {log_path.name}\n{tail[-800:]}")
    return (proc.returncode or 0), log_path
