"""Vietnamese transcription with faster-whisper + word timestamps.

Uses word_timestamps=True and language='vi'. VAD filter is optional and
guarded so it never crashes when the VAD model is unavailable.
"""
from __future__ import annotations

import threading
from typing import Callable

from models.types import Segment, Word

DEFAULT_MODEL = "medium"
VI_MODELS = ("tiny", "base", "small", "medium", "large-v3")

# Process-wide model cache: load once, reuse across runs / retries.
# Key = (model_size, device, compute). Never reload per segment/word.
_MODEL_CACHE: dict[tuple[str, str, str], object] = {}
_MODEL_LOCK = threading.Lock()


class Transcriber:
    def __init__(self, model_size: str = DEFAULT_MODEL, device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self._cancel = threading.Event()
        self._model = None

    def cancel(self) -> None:
        self._cancel.set()

    def reset(self) -> None:
        self._cancel.clear()

    # -- device resolution -------------------------------------------------
    def resolve_device(self) -> tuple[str, str]:
        """Return (device, compute_type). CPU fallback always works."""
        req = (self.device or "auto").lower()
        if req.startswith("cpu"):
            return "cpu", "int8"
        # "cuda"/"gpu"/"auto": prefer CUDA, _load_model falls back to CPU
        # on any CUDA/cuDNN error (no expensive probe-load here).
        return "cuda", "float16"

    def _load_model(self):
        device, compute = self.resolve_device()
        key = (self.model_size, device, compute)
        with _MODEL_LOCK:
            if key in _MODEL_CACHE:
                self._model = _MODEL_CACHE[key]
                return self._model
        from faster_whisper import WhisperModel

        try:
            model = WhisperModel(self.model_size, device=device, compute_type=compute)
        except Exception as exc:
            if device != "cpu":
                # CUDA/cuDNN problems -> CPU fallback with clear error context
                key = (self.model_size, "cpu", "int8")
                with _MODEL_LOCK:
                    if key in _MODEL_CACHE:
                        self._model = _MODEL_CACHE[key]
                        return self._model
                model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            else:
                raise RuntimeError(f"Không tải được mô hình Whisper ({self.model_size}): {exc}") from exc
        with _MODEL_LOCK:
            _MODEL_CACHE[key] = model
        self._model = model
        return self._model

    @staticmethod
    def cached_models() -> list[str]:
        with _MODEL_LOCK:
            return [f"{k[0]}:{k[1]}" for k in _MODEL_CACHE]

    # -- main API ----------------------------------------------------------
    def transcribe(
        self,
        audio_path: str,
        language: str = "vi",
        vad: bool = True,
        progress_cb: Callable[[float, str], None] | None = None,
        audio_duration: float = 0.0,
    ) -> list[Segment]:
        """Transcribe wav audio to segments with word timestamps."""
        self.reset()
        model = self._load_model()

        kwargs: dict = dict(
            language=language or "vi",
            task="transcribe",
            word_timestamps=True,
            vad_filter=bool(vad),
            condition_on_previous_text=False,
        )
        # beam size 5 is a good default for Vietnamese accuracy/speed
        segments_iter, _info = model.transcribe(audio_path, beam_size=5, **kwargs)

        out: list[Segment] = []
        for seg in segments_iter:
            if self._cancel.is_set():
                raise InterruptedError("Người dùng đã hủy quá trình nhận dạng.")
            words: list[Word] = []
            for w in getattr(seg, "words", None) or []:
                text = (w.word or "").strip()
                if not text:
                    continue
                try:
                    s, e = float(w.start), float(w.end)
                except (TypeError, ValueError):
                    continue
                if e < s:
                    s, e = e, s
                words.append(Word(text=text, start=s, end=e, confidence=float(getattr(w, "probability", 1.0) or 1.0)))
            text = (seg.text or "").strip()
            if not text and not words:
                continue
            if not words:
                # No word timing available -> keep segment-level timing only
                words = [Word(text=text, start=float(seg.start), end=float(seg.end))]
            out.append(Segment(start=float(seg.start), end=float(seg.end), text=text, words=words))
            if progress_cb:
                # REAL progress: segment end vs total audio duration
                frac = (float(seg.end) / audio_duration) if audio_duration > 0 else 0.0
                progress_cb(min(0.99, max(0.0, frac)), f"Đang nhận dạng... {text[:40]}")
        if progress_cb:
            progress_cb(1.0, f"Hoàn tất nhận dạng: {len(out)} đoạn.")
        return out
