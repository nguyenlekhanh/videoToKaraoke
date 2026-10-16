"""Main window — Vietnamese UI for karaoke subtitle app."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QFileDialog, QFontComboBox,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QSplitter, QVBoxLayout,
    QWidget,
)

from core.alignment import sanitize_words
from core.ffmpeg_manager import check_tools, friendly_install_help
from core.subtitle_renderer import SubtitleStyle
from core.video_processor import probe
from models.types import Segment, Word
from ui.workers import RenderWorker, TranscribeWorker
from utils.helpers import format_srt_time


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tạo phụ đề Karaoke tiếng Việt")
        self.resize(1080, 760)
        self.video_path = ""
        self.video_info = None
        self.segments: list[Segment] = []
        self.thread: QThread | None = None
        self.worker = None
        self._build_ui()
        self._check_env()

    # -- UI ----------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QVBoxLayout(central)

        # video row
        file_box = QGroupBox("1. Chọn video")
        fl = QHBoxLayout(file_box)
        self.lbl_file = QLabel("Chưa chọn file")
        self.lbl_file.setTextInteractionFlags(Qt.TextSelectableByMouse)
        btn_browse = QPushButton("Chọn video...")
        btn_browse.clicked.connect(self.pick_video)
        fl.addWidget(self.lbl_file, 1)
        fl.addWidget(btn_browse)
        main.addWidget(file_box)

        self.lbl_meta = QLabel("Tên file / phân giải / thời lượng sẽ hiện ở đây.")
        main.addWidget(self.lbl_meta)

        # options row
        opt = QHBoxLayout()
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.cmb_model.setCurrentText("medium")
        self.cmb_device = QComboBox()
        self.cmb_device.addItems(["auto", "cpu", "cuda"])
        self.chk_vad = QCheckBox("Lọc khoảng lặng (VAD)")
        self.chk_vad.setChecked(True)
        self.cmb_quality = QComboBox()
        self.cmb_quality.addItems(["cao (crf 17)", "medium (crf 20)", "nhẹ (crf 24)"])
        self.cmb_quality.setCurrentIndex(1)
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(["ultrafast", "veryfast", "fast", "medium"])
        self.cmb_preset.setCurrentText("veryfast")
        self.cmb_preset.setToolTip("veryfast: render nhanh (khuyên dùng khi test). medium: chậm hơn, file nhỏ hơn.")
        opt.addWidget(QLabel("Model Whisper:"))
        opt.addWidget(self.cmb_model)
        opt.addWidget(QLabel("Thiết bị:"))
        opt.addWidget(self.cmb_device)
        opt.addWidget(self.chk_vad)
        opt.addWidget(QLabel("Chất lượng:"))
        opt.addWidget(self.cmb_quality)
        opt.addWidget(QLabel("Preset:"))
        opt.addWidget(self.cmb_preset)
        opt.addStretch(1)
        main.addLayout(opt)

        # fast-test row
        fast_box = QGroupBox("Chế độ test nhanh (lặp nhanh khi chỉnh style)")
        fl2 = QHBoxLayout(fast_box)
        self.chk_fast = QCheckBox("Chỉ xử lý đoạn đầu video")
        self.chk_fast.setToolTip("Bật để test style/render nhanh: chỉ trích + render N giây đầu, dùng cache transcript.")
        self.spin_clip = QSpinBox()
        self.spin_clip.setRange(5, 120)
        self.spin_clip.setValue(20)
        self.spin_clip.setSuffix(" giây")
        self.chk_cache = QCheckBox("Dùng lại transcript đã cache")
        self.chk_cache.setChecked(True)
        self.chk_cache.setToolTip("Nếu video + model không đổi, bỏ qua Whisper/alignment, dùng kết quả lần trước.")
        fl2.addWidget(self.chk_fast)
        fl2.addWidget(self.spin_clip)
        fl2.addWidget(self.chk_cache)
        fl2.addStretch(1)
        main.addWidget(fast_box)

        splitter = QSplitter(Qt.Horizontal)

        # transcript editor
        left = QGroupBox("2. Soát lại lời thoại (sửa trước khi render)")
        ll = QVBoxLayout(left)
        self.txt_transcript = QPlainTextEdit()
        self.txt_transcript.setPlaceholderText("Sau khi nhận dạng, lời thoại sẽ hiện ở đây để bạn sửa lỗi chính tả...")
        ll.addWidget(self.txt_transcript)
        row_e = QHBoxLayout()
        btn_apply = QPushButton("Áp dụng chỉnh sửa")
        btn_apply.clicked.connect(self.apply_edits)
        btn_retrans = QPushButton("🔁 Nhận dạng lại")
        btn_retrans.clicked.connect(self.start_transcribe)
        row_e.addWidget(btn_apply)
        row_e.addWidget(btn_retrans)
        ll.addLayout(row_e)
        splitter.addWidget(left)

        # style options
        right = QGroupBox("3. Tùy chỉnh phụ đề")
        form = QFormLayout(right)
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(self.font())
        self.spin_size = QSpinBox()
        self.spin_size.setRange(12, 72)
        self.spin_size.setValue(40)
        self.btn_primary = QPushButton("Trắng (mặc định)")
        self._primary = "#FFFFFF"
        self.btn_primary.clicked.connect(lambda: self._pick_color("primary"))
        self.btn_hi = QPushButton("Vàng (từ đang nói)")
        self._hi = "#FFFF00"
        self.btn_hi.clicked.connect(lambda: self._pick_color("hi"))
        self.spin_outline = QSpinBox()
        self.spin_outline.setRange(0, 6)
        self.spin_outline.setValue(2)
        self.spin_margin = QSpinBox()
        self.spin_margin.setRange(0, 300)
        self.spin_margin.setValue(80)
        self.spin_maxw = QSpinBox()
        self.spin_maxw.setRange(3, 20)
        self.spin_maxw.setValue(8)
        self.cmb_pos = QComboBox()
        self.cmb_pos.addItems(["Dưới giữa", "Trên giữa", "Dưới trái", "Dưới phải"])
        self.out_dir_edit = QLabel(str(Path.home() / "Videos" / "SubVN"))
        btn_out = QPushButton("Chọn thư mục...")
        btn_out.clicked.connect(self.pick_outdir)
        self.chk_srt = QCheckBox("Xuất thêm file .srt")
        self.chk_srt.setChecked(True)
        self.chk_json = QCheckBox("Xuất JSON timestamps")
        self.chk_json.setChecked(True)
        form.addRow("Font chữ:", self.font_combo)
        form.addRow("Cỡ chữ:", self.spin_size)
        form.addRow("Màu chữ thường:", self.btn_primary)
        form.addRow("Màu từ đang nói:", self.btn_hi)
        form.addRow("Viền đen:", self.spin_outline)
        form.addRow("Cách mép dưới (px):", self.spin_margin)
        form.addRow("Số từ tối đa/dòng:", self.spin_maxw)
        form.addRow("Vị trí:", self.cmb_pos)
        form.addRow("Thư mục xuất:", self.out_dir_edit)
        form.addRow("", btn_out)
        form.addRow("", self.chk_srt)
        form.addRow("", self.chk_json)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        main.addWidget(splitter, 1)

        # action row
        row = QHBoxLayout()
        self.btn_start = QPushButton("▶ Bắt đầu nhận dạng")
        self.btn_start.clicked.connect(self.start_transcribe)
        self.btn_render = QPushButton("🎬 Render video karaoke")
        self.btn_render.clicked.connect(self.start_render)
        self.btn_render.setEnabled(False)
        self.btn_cancel = QPushButton("Hủy")
        self.btn_cancel.clicked.connect(self.cancel_job)
        self.btn_cancel.setEnabled(False)
        self.btn_open_video = QPushButton("Mở video đã render")
        self.btn_open_video.clicked.connect(self.open_result)
        self.btn_open_video.setEnabled(False)
        self.btn_open_folder = QPushButton("Mở thư mục")
        self.btn_open_folder.clicked.connect(self.open_folder)
        self.btn_logs = QPushButton("Mở log")
        self.btn_logs.clicked.connect(self.open_logs)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_render)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_open_video)
        row.addWidget(self.btn_open_folder)
        row.addWidget(self.btn_logs)
        main.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        main.addWidget(self.progress)
        self.lbl_status = QLabel("Sẵn sàng.")
        main.addWidget(self.lbl_status)
        # stage timing panel + live ffmpeg stats
        self.lbl_stages = QLabel("Các giai đoạn sẽ hiện thời gian thực tế ở đây.")
        self.lbl_stages.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_stages.setStyleSheet("font-family: Consolas, monospace;")
        main.addWidget(self.lbl_stages)
        self.lbl_ffmpeg = QLabel("")
        self.lbl_ffmpeg.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_ffmpeg.setStyleSheet("font-family: Consolas, monospace; color: #0a6ebd;")
        main.addWidget(self.lbl_ffmpeg)
        self.result_path = ""
        self.stage_times: dict[str, tuple[float, str]] = {}

    # -- env ---------------------------------------------------------------
    def _check_env(self):
        st = check_tools()
        if not st.ffmpeg or not st.ffprobe:
            QMessageBox.warning(self, "Thiếu FFmpeg", friendly_install_help())
            self.lbl_status.setText("⚠ Chưa tìm thấy FFmpeg — xem hướng dẫn trong hộp thoại.")
        elif not st.has_libass:
            self.lbl_status.setText("Lưu ý: FFmpeg có thể thiếu filter libass — render karaoke có thể lỗi.")

    # -- helpers -----------------------------------------------------------
    def _pick_color(self, which: str):
        cur = QColor(self._primary if which == "primary" else self._hi)
        c = QColorDialog.getColor(cur, self, "Chọn màu")
        if c.isValid():
            if which == "primary":
                self._primary = c.name().upper()
                self.btn_primary.setText(self._primary)
            else:
                self._hi = c.name().upper()
                self.btn_hi.setText(self._hi)

    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn video", str(Path.home()),
                                              "Video (*.mp4 *.mkv *.avi *.mov *.m4v *.webm)")
        if not path:
            return
        self.video_path = path
        self.lbl_file.setText(path)
        try:
            self.video_info = probe(path)
            i = self.video_info
            mins = int(i.duration // 60)
            self.lbl_meta.setText(
                f"{Path(path).name} — {i.width}x{i.height} — {mins}:{i.duration % 60:04.1f} "
                f"— {'có audio' if i.has_audio else 'KHÔNG có audio'}")
            self.lbl_status.setText("Đã chọn video. Nhấn 'Bắt đầu nhận dạng'.")
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi video", str(exc))
            self.lbl_status.setText(f"Lỗi: {exc}")

    def pick_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục xuất", self.out_dir_edit.text())
        if d:
            self.out_dir_edit.setText(d)

    def make_style(self) -> SubtitleStyle:
        align_map = {"Dưới giữa": 2, "Trên giữa": 8, "Dưới trái": 1, "Dưới phải": 3}
        return SubtitleStyle(
            font=self.font_combo.currentFont().family(),
            font_size=self.spin_size.value(),
            primary=self._primary,
            highlight=self._hi,
            outline=self.spin_outline.value(),
            margin_v=self.spin_margin.value(),
            max_words_per_line=self.spin_maxw.value(),
            alignment=align_map.get(self.cmb_pos.currentText(), 2),
        )

    def set_busy(self, busy: bool, msg: str = ""):
        self.btn_cancel.setEnabled(busy)
        self.btn_start.setEnabled(not busy)
        self.btn_render.setEnabled(not busy and bool(self.segments))
        if msg:
            self.lbl_status.setText(msg)
        QApplication.processEvents()

    # -- transcript --------------------------------------------------------
    def show_transcript(self):
        lines = []
        for s in self.segments:
            lines.append(f"[{format_srt_time(s.start)} --> {format_srt_time(s.end)}] {s.text}")
        self.txt_transcript.setPlainText("\n".join(lines))

    def apply_edits(self):
        """Re-map edited text onto existing word timings proportionally per segment."""
        if not self.segments:
            return
        raw = [l.strip() for l in self.txt_transcript.toPlainText().splitlines() if l.strip()]
        if not raw:
            QMessageBox.warning(self, "Trống", "Khung soát lỗi đang trống.")
            return
        import re
        new_texts: list[str] = []
        for line in raw:
            m = re.search(r"\]\s*(.*)$", line)
            new_texts.append((m.group(1) if m else line).strip())
        # merge/split to segment count: join all then redistribute if counts differ
        if len(new_texts) != len(self.segments):
            QMessageBox.information(
                self, "Gợi ý",
                f"Số dòng ({len(new_texts)}) khác số đoạn ({len(self.segments)}). "
                "Ứng dụng sẽ giữ nguyên mốc thời gian đoạn và chia lại từ.")
            # flatten words across whole timeline proportionally
            all_words_text = " ".join(new_texts).split()
            flat_old = [w for s in self.segments for w in s.words]
            if not all_words_text or not flat_old:
                return
            t0, t1 = flat_old[0].start, flat_old[-1].end
            total_chars = sum(len(w) for w in all_words_text) or 1
            t = t0
            new_flat: list[Word] = []
            for wt in all_words_text:
                dur = max(0.12, (t1 - t0) * len(wt) / total_chars)
                new_flat.append(Word(text=wt, start=t, end=min(t1, t + dur)))
                t += dur
            # re-chunk by original segment boundaries count
            per = max(1, len(new_flat) // len(self.segments))
            for i, s in enumerate(self.segments):
                chunk = new_flat[i * per:(i + 1) * per] if i < len(self.segments) - 1 else new_flat[i * per:]
                if chunk:
                    s.words = sanitize_words(chunk)
                    s.start, s.end = chunk[0].start, chunk[-1].end
                    s.text = " ".join(w.text for w in chunk)
        else:
            for s, nt in zip(self.segments, new_texts):
                if not nt:
                    continue
                s.text = nt
                words_new = nt.split()
                old = s.words
                if len(words_new) == len(old):
                    for w, nw in zip(old, words_new):
                        w.text = nw
                elif old:
                    # redistribute within [old.start, old.end] by char length
                    t0, t1 = old[0].start, old[-1].end
                    total = sum(len(w) for w in words_new) or 1
                    t = t0
                    rebuilt = []
                    for wn in words_new:
                        dur = max(0.12, (t1 - t0) * len(wn) / total)
                        rebuilt.append(Word(text=wn, start=t, end=min(t1, t + dur)))
                        t += dur
                    s.words = sanitize_words(rebuilt)
        self.show_transcript()
        self.lbl_status.setText("Đã áp dụng chỉnh sửa (không cần chạy lại Whisper).")

    # -- jobs --------------------------------------------------------------
    def _run_worker(self, worker, on_done):
        from core.stages import VI_LABELS
        self.stage_times = {}
        self.lbl_stages.setText("Bắt đầu...")
        self.lbl_ffmpeg.setText("")
        self.thread = QThread()
        worker.moveToThread(self.thread)
        worker.signals.progress.connect(lambda f, m: (self.progress.setValue(int(f * 100)), self.lbl_status.setText(m)))
        worker.signals.error.connect(self._on_error)
        worker.signals.finished.connect(on_done)
        worker.signals.finished.connect(self._job_done)
        worker.signals.error.connect(self._job_done_err)
        worker.signals.ffmpeg.connect(lambda s: self.lbl_ffmpeg.setText("FFmpeg: " + s))

        def _on_stage(name: str, elapsed: float, status: str):
            self.stage_times[name] = (elapsed, status)
            lines = []
            for n, (e, st) in self.stage_times.items():
                mark = "⏭" if st.startswith("skipped") else "✔"
                lines.append(f"{mark} {VI_LABELS.get(n, n)}: {e:.1f}s ({st})")
            self.lbl_stages.setText("\n".join(lines))

        worker.signals.stage.connect(_on_stage)
        self.thread.started.connect(worker.run)
        self.worker = worker
        self.set_busy(True)
        self.thread.start()

    def start_transcribe(self):
        if not self.video_path:
            QMessageBox.warning(self, "Chưa chọn video", "Hãy chọn file video trước.")
            return
        from ui.workers import default_workdir
        clip = float(self.spin_clip.value()) if self.chk_fast.isChecked() else 0.0
        w = TranscribeWorker(self.video_path, default_workdir(), self.cmb_model.currentText(),
                             self.cmb_device.currentText(), self.chk_vad.isChecked(),
                             clip_seconds=clip, reuse_cache=self.chk_cache.isChecked())
        if clip > 0:
            self.lbl_status.setText(f"Test nhanh: chỉ xử lý {clip:.0f}s đầu video.")
        self._run_worker(w, self._on_transcribed)

    def _on_transcribed(self, payload):
        self.segments = payload["segments"]
        self.video_info = payload["info"]
        self.show_transcript()
        self.btn_render.setEnabled(True)
        n = sum(len(s.words) for s in self.segments)
        skip = payload.get("skipped") or []
        skip_msg = f"\nBỏ qua (cache): {', '.join(skip)}." if skip else ""
        QMessageBox.information(self, "Xong", f"Nhận dạng xong: {len(self.segments)} đoạn, {n} từ.{skip_msg}\n"
                                              f"Hãy soát lỗi rồi nhấn Render.\nLog: {payload.get('log', '')}")

    def start_render(self):
        if not self.segments:
            QMessageBox.warning(self, "Chưa có phụ đề", "Hãy nhận dạng trước khi render.")
            return
        if not self.video_info:
            self.video_info = probe(self.video_path)
        clip = float(self.spin_clip.value()) if self.chk_fast.isChecked() else 0.0
        w = RenderWorker(self.video_path, self.segments, self.make_style(), self.out_dir_edit.text(),
                         self.video_info.width, self.video_info.height, self.cmb_quality.currentText(),
                         preset=self.cmb_preset.currentText(),
                         export_srt=self.chk_srt.isChecked(), export_json=self.chk_json.isChecked(),
                         clip_seconds=clip)
        self._run_worker(w, self._on_rendered)

    def _on_rendered(self, payload):
        self.result_path = payload["mp4"]
        self.btn_open_video.setEnabled(True)
        vis = (payload.get("visibility") or {}).get("verdict", "?")
        kar = (payload.get("karaoke") or {}).get("verdict", "?")
        prev = payload.get("previews") or []
        prev_msg = f"\nẢnh kiểm tra: {prev[0]}" if prev else ""
        QMessageBox.information(self, "Hoàn tất",
                                f"Video đã lưu:\n{self.result_path}\n"
                                f"Chữ hiện trên video: {vis} | Highlight vàng: {kar}{prev_msg}\n"
                                f"Log: {payload.get('log', '')}")

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Lỗi", msg[:2000])

    def _job_done(self, *_):
        self.set_busy(False, "Sẵn sàng.")
        if self.thread:
            self.thread.quit()
            self.thread.wait(3000)

    def _job_done_err(self, *_):
        self.set_busy(False)
        if self.thread:
            self.thread.quit()
            self.thread.wait(3000)

    def cancel_job(self):
        if self.worker and hasattr(self.worker, "cancel"):
            self.worker.cancel()
            self.lbl_status.setText("Đang hủy...")

    def open_result(self):
        if self.result_path and Path(self.result_path).exists():
            try:
                import os
                os.startfile(self.result_path)  # type: ignore[attr-defined]
            except Exception:
                subprocess.Popen(["explorer", "/select,", self.result_path])

    def open_folder(self):
        d = self.out_dir_edit.text()
        Path(d).mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", d])

    def open_logs(self):
        from core.stages import log_dir
        d = str(log_dir())
        subprocess.Popen(["explorer", d])


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SubVN Karaoke")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
