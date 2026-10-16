"""Stage-level timing and file logging.

Every pipeline run appends to:
  logs/pipeline_<timestamp>.jsonl   (one JSON object per stage: name, elapsed_s, detail)
  logs/pipeline_<timestamp>.log     (human readable)
  logs/ffmpeg_<timestamp>.log       (raw ffmpeg stderr, via ffmpeg_runner)

This is what lets us tell "downloading model" apart from "rendering video"
instead of staring at a stuck progress bar.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

STAGES = (
    "probe",
    "extract_audio",
    "model_load",
    "transcribe",
    "align",
    "subtitle_gen",
    "render",
    "validate",
)

VI_LABELS = {
    "probe": "Đọc metadata video",
    "extract_audio": "Trích xuất audio",
    "model_load": "Tải mô hình Whisper",
    "transcribe": "Nhận dạng giọng nói",
    "align": "Căn chỉnh timestamps",
    "subtitle_gen": "Sinh phụ đề ASS/SRT/JSON",
    "render": "Render FFmpeg (burn-in)",
    "validate": "Kiểm tra file output",
}


def log_dir() -> Path:
    d = Path(__file__).resolve().parents[1] / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


class RunLog:
    """Per-run logger. Thread-safe enough for one worker thread + GUI reads."""

    def __init__(self, tag: str = "run"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base = log_dir() / f"{tag}_{stamp}"
        self.jsonl_path = self.base.with_suffix(".jsonl")
        self.text_path = self.base.with_suffix(".log")
        self.records: list[dict] = []
        self._write(f"=== pipeline log {stamp} ===")

    def _write(self, line: str) -> None:
        try:
            with open(self.text_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def event(self, msg: str) -> None:
        self._write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def stage(self, name: str, elapsed_s: float, detail: str = "") -> dict:
        rec = {"stage": name, "label": VI_LABELS.get(name, name),
               "elapsed_s": round(elapsed_s, 2), "detail": detail}
        self.records.append(rec)
        self.event(f"STAGE {name}: {elapsed_s:.1f}s {detail}")
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return rec

    def total(self) -> float:
        return sum(r["elapsed_s"] for r in self.records)

    def summary(self) -> str:
        lines = [f"- {r['label']}: {r['elapsed_s']:.1f}s {r['detail']}" for r in self.records]
        lines.append(f"TỔNG: {self.total():.1f}s")
        return "\n".join(lines)


@contextmanager
def timed(runlog: RunLog | None, name: str, detail: str = "") -> Iterator[None]:
    t0 = time.time()
    try:
        yield
    finally:
        if runlog is not None:
            runlog.stage(name, time.time() - t0, detail)
