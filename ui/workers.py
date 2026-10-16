"""Background workers (QThread) so the GUI stays responsive.

Each stage is timed with core.stages.RunLog and reported to the GUI —
no stage ever shows "done" before it finishes, and skipped stages
(fast-test cache reuse) are reported explicitly as SKIPPED.
"""
from __future__ import annotations

import tempfile
import threading
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from core.alignment import align_segments
from core.cache import cache_key, cache_path, load as load_cache, save as save_cache
from core.stages import RunLog, timed
from core.subtitle_renderer import SubtitleStyle, write_ass, write_srt, write_words_json
from core.transcriber import Transcriber
from core.video_processor import (
    check_karaoke_colors, check_visibility, default_outputs, extract_audio,
    pick_sample_times, probe, render_burned, save_preview_frames, validate_input,
)


class WorkerSignals(QObject):
    progress = Signal(float, str)   # 0..1, message
    stage = Signal(str, float, str)  # stage name, elapsed_s, status (done/skipped)
    ffmpeg = Signal(str)            # raw ffmpeg stats line
    finished = Signal(object)       # result payload
    error = Signal(str)             # user-friendly message


class TranscribeWorker(QObject):
    def __init__(self, video: str, workdir: str, model: str, device: str, use_vad: bool,
                 clip_seconds: float = 0.0, reuse_cache: bool = True):
        super().__init__()
        self.video = video
        self.workdir = workdir
        self.model = model
        self.device = device
        self.use_vad = use_vad
        self.clip_seconds = clip_seconds
        self.reuse_cache = reuse_cache
        self.signals = WorkerSignals()
        self._cancel = threading.Event()
        self._transcriber: Transcriber | None = None
        self.runlog = RunLog("transcribe")

    def cancel(self):
        self._cancel.set()
        if self._transcriber:
            self._transcriber.cancel()

    def _emit(self, frac: float, msg: str):
        self.signals.progress.emit(frac, msg)
        self.runlog.event(f"{frac * 100:.0f}% {msg}")

    def run(self):
        skipped: list[str] = []
        try:
            t_all = time.time()
            self._emit(0.02, "Đang kiểm tra video...")
            with timed(self.runlog, "probe"):
                validate_input(self.video)
                info = probe(self.video)
            self.signals.stage.emit("probe", self.runlog.records[-1]["elapsed_s"], "done")
            self._emit(0.08, f"Video {info.width}x{info.height}, {info.duration:.1f}s")
            if not info.has_audio:
                raise ValueError("Video không có track audio — không thể nhận dạng giọng nói.")
            if self._cancel.is_set():
                raise InterruptedError("Đã hủy.")

            # --- transcript cache: skip whisper entirely on cache hit ---
            key = cache_key(self.video, self.model, self.use_vad)
            cpath = cache_path(self.workdir, self.video)
            if self.reuse_cache and self.clip_seconds <= 0:
                t0 = time.time()
                cached = load_cache(cpath, key)
                if cached is not None:
                    el = time.time() - t0
                    self.runlog.stage("transcribe", el, "CACHE HIT — bỏ qua Whisper")
                    self.signals.stage.emit("transcribe", el, "skipped (cache)")
                    self.runlog.stage("align", 0.0, "dùng timestamps đã cache")
                    self.signals.stage.emit("align", 0.0, "skipped (cache)")
                    skipped = ["model_load", "transcribe", "align"]
                    self._emit(1.0, f"Dùng cache: {len(cached)} đoạn (không chạy Whisper).")
                    self.signals.finished.emit({"segments": cached, "info": info,
                                                "paths": default_outputs(self.video, self.workdir),
                                                "skipped": skipped,
                                                "log": str(self.runlog.text_path)})
                    return
                else:
                    self.runlog.event("cache MISS — chạy Whisper đầy đủ")

            self._emit(0.12, "Đang trích xuất audio (16kHz mono)...")
            outs = default_outputs(self.video, self.workdir)
            with timed(self.runlog, "extract_audio"):
                extract_audio(self.video, str(outs["wav"]), cancel=self._cancel,
                              on_progress=lambda f, m: self._emit(0.12 + 0.08 * f, m),
                              clip_seconds=self.clip_seconds)
            self.signals.stage.emit("extract_audio", self.runlog.records[-1]["elapsed_s"], "done")

            self._emit(0.22, "Đang tải mô hình Whisper (lần đầu cần tải ~1.5GB)...")
            self._transcriber = Transcriber(model_size=self.model, device=self.device)
            with timed(self.runlog, "model_load",
                       detail=f"model={self.model} device={self.device} cached={Transcriber.cached_models()}"):
                self._transcriber._load_model()
            self.signals.stage.emit("model_load", self.runlog.records[-1]["elapsed_s"], "done")

            audio_dur = self.clip_seconds if self.clip_seconds > 0 else info.duration

            def _cb(f: float, msg: str):
                self._emit(0.25 + 0.60 * f, msg)

            with timed(self.runlog, "transcribe"):
                segs = self._transcriber.transcribe(str(outs["wav"]), language="vi",
                                                    vad=self.use_vad, progress_cb=_cb,
                                                    audio_duration=audio_dur)
            self.signals.stage.emit("transcribe", self.runlog.records[-1]["elapsed_s"], "done")
            if self._cancel.is_set():
                raise InterruptedError("Đã hủy.")
            if not segs:
                raise ValueError("Không nhận dạng được lời nói nào (video có thể không có tiếng Việt hoặc chỉ có nhạc).")
            with timed(self.runlog, "align"):
                segs = align_segments(segs)
            self.signals.stage.emit("align", self.runlog.records[-1]["elapsed_s"], "done")
            if self.clip_seconds <= 0:
                save_cache(segs, cpath, key)
                self.runlog.event(f"đã lưu cache {cpath.name}")
            total = time.time() - t_all
            self._emit(1.0, f"Xong sau {total:.0f}s: {len(segs)} đoạn, "
                            f"{sum(len(s.words) for s in segs)} từ.")
            self.signals.finished.emit({"segments": segs, "info": info, "paths": outs,
                                        "skipped": skipped,
                                        "log": str(self.runlog.text_path)})
        except (InterruptedError, TimeoutError) as exc:
            self.signals.error.emit(str(exc))
        except Exception as exc:
            self.signals.error.emit(f"{exc}\nLog: {self.runlog.text_path}\n{traceback.format_exc(limit=5)}")


class RenderWorker(QObject):
    def __init__(self, video: str, segments: list, style: SubtitleStyle, out_dir: str,
                 width: int, height: int, quality: str, preset: str = "veryfast",
                 export_srt: bool = True, export_json: bool = True,
                 clip_seconds: float = 0.0):
        super().__init__()
        self.video = video
        self.segments = segments
        self.style = style
        self.out_dir = out_dir
        self.width = width
        self.height = height
        self.quality = quality
        self.preset = preset
        self.export_srt = export_srt
        self.export_json = export_json
        self.clip_seconds = clip_seconds
        self.signals = WorkerSignals()
        self._cancel = threading.Event()
        self.runlog = RunLog("render")

    def cancel(self):
        self._cancel.set()

    def _emit(self, frac: float, msg: str):
        self.signals.progress.emit(frac, msg)
        self.runlog.event(f"{frac * 100:.0f}% {msg}")

    def run(self):
        try:
            t_all = time.time()
            info = probe(self.video)
            total_dur = self.clip_seconds if self.clip_seconds > 0 else info.duration
            self._emit(0.02, "Đang tạo phụ đề ASS (highlight từng từ)...")
            outs = default_outputs(self.video, self.out_dir)
            t0 = time.time()
            write_ass(outs["ass"], self.segments, self.style, self.width, self.height)
            if self.export_srt:
                write_srt(outs["srt"], self.segments)
            if self.export_json:
                write_words_json(outs["json"], self.segments)
            el = time.time() - t0
            self.runlog.stage("subtitle_gen", el,
                              detail=f"{len(self.segments)} đoạn, {outs['ass'].name}")
            self.signals.stage.emit("subtitle_gen", el, "done")
            if self._cancel.is_set():
                raise InterruptedError("Đã hủy.")

            def _cb(f: float, msg: str):
                self._emit(0.05 + 0.90 * f, msg)
                self.signals.ffmpeg.emit(msg)

            with timed(self.runlog, "render",
                       detail=f"preset={self.preset} crf={self.quality}"):
                mp4 = render_burned(self.video, outs["ass"], outs["mp4"],
                                    quality=self.quality, preset=self.preset,
                                    on_progress=_cb, cancel=self._cancel,
                                    clip_seconds=self.clip_seconds,
                                    total_duration=total_dur,
                                    style=self.style)
            self.signals.stage.emit("render", self.runlog.records[-1]["elapsed_s"], "done")
            # --- post-render visibility validation (exit code 0 proves nothing) ---
            self._emit(0.96, "Đang kiểm tra chữ có hiện trên video không...")
            t0 = time.time()
            sample_times = pick_sample_times(self.segments, total_dur, n=3)
            visibility: dict = {"verdict": "INCONCLUSIVE", "samples": []}
            karaoke: dict = {"verdict": "INCONCLUSIVE", "samples": []}
            previews: list[str] = []
            try:
                visibility = check_visibility(self.video, str(mp4), sample_times)
                karaoke = check_karaoke_colors(self.video, str(mp4), sample_times)
                previews = [str(p) for p in save_preview_frames(
                    str(mp4), sample_times, Path(self.out_dir) / "previews",
                    stem=Path(str(mp4)).stem)]
            except Exception as exc:
                self.runlog.event(f"visibility check lỗi: {exc}")
            el = time.time() - t0
            self.runlog.stage("validate", el, detail=(
                f"visibility={visibility.get('verdict')} "
                f"karaoke_yellow={karaoke.get('verdict')}"))
            self.signals.stage.emit("validate", el, f"done ({visibility.get('verdict')})")
            if visibility.get("verdict") == "FAIL":
                raise RuntimeError(
                    "Render xong nhưng KHÔNG thấy chữ trên video "
                    f"(kiểm tra frame tại {sample_times}). "
                    f"Hãy mở ảnh preview trong {Path(self.out_dir) / 'previews'} để xem. "
                    "Nguyên nhân thường gặp: bản FFmpeg thiếu libass, file ASS lỗi, "
                    "hoặc font không có chữ Việt.")
            if karaoke.get("verdict") == "FAIL":
                raise RuntimeError(
                    "Chữ hiện nhưng KHÔNG có từ nào chuyển vàng — tag highlight "
                    "không có tác dụng. Hãy mở ảnh preview để kiểm tra, thử font "
                    "Arial/Segoe UI và render lại.")
            total = time.time() - t_all
            self._emit(1.0, f"Hoàn tất sau {total:.0f}s: {Path(mp4).name} "
                            f"[chữ: {visibility.get('verdict')}, vàng: {karaoke.get('verdict')}]")
            self.signals.finished.emit({"mp4": str(mp4), "srt": str(outs["srt"]),
                                        "json": str(outs["json"]), "ass": str(outs["ass"]),
                                        "visibility": visibility, "karaoke": karaoke,
                                        "previews": previews,
                                        "log": str(self.runlog.text_path)})
        except (InterruptedError, TimeoutError) as exc:
            self.signals.error.emit(str(exc))
        except Exception as exc:
            self.signals.error.emit(f"{exc}\nLog: {self.runlog.text_path}\n{traceback.format_exc(limit=5)}")


def default_workdir() -> str:
    d = Path(tempfile.gettempdir()) / "subvn_cache"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
