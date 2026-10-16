"""Word-timestamp cleanup / light alignment for Vietnamese.

Evaluation note (documented, not assumed):
- Popular forced aligners (e.g. whisperx / wav2vec2-based) do NOT ship a
  reliable Vietnamese acoustic model. `wav2vec2-large-960h` is English-only;
  `whisperx` alignment models cover en/es/fr/de/it/pt/nl, not vi.
- Therefore this project does NOT depend on an external forced aligner.
  It uses faster-whisper word timestamps (which do support Vietnamese)
  plus deterministic cleanup: sorting, de-overlap, gap clamping, silence
  handling. This is honest about accuracy limits and stays local/free.

If a genuine Vietnamese aligner becomes available, plug it in via
`align_with_external()` without changing the GUI/renderer.
"""
from __future__ import annotations

from models.types import Segment, Word

MIN_WORD_DUR = 0.08
MAX_WORD_DUR = 3.0
MAX_GAP_FILL = 0.15  # extend word end into small gaps so highlight feels continuous


def sanitize_words(words: list[Word]) -> list[Word]:
    clean: list[Word] = []
    for w in words:
        text = (w.text or "").strip()
        if not text:
            continue
        try:
            s, e = float(w.start), float(w.end)
        except (TypeError, ValueError):
            continue
        if s < 0:
            s = 0.0
        if e < 0:
            e = 0.0
        if e < s:
            s, e = e, s
        if e - s < 0.01:
            e = s + 0.12  # whisper sometimes emits ~0 duration
        if e - s > MAX_WORD_DUR:
            e = s + MAX_WORD_DUR
        clean.append(Word(text=text, start=s, end=e, confidence=w.confidence))
    clean.sort(key=lambda w: (w.start, w.end))
    # de-overlap: never let a word start before previous word ends
    for i in range(1, len(clean)):
        if clean[i].start < clean[i - 1].end:
            clean[i].start = clean[i - 1].end
            if clean[i].end <= clean[i].start:
                clean[i].end = clean[i].start + MIN_WORD_DUR
    return clean


def smooth_gaps(words: list[Word]) -> list[Word]:
    """Stretch word end-times slightly into tiny pauses (<=MAX_GAP_FILL).

    Large silences are preserved so no subtitle shows highlight during silence.
    """
    words = sanitize_words(words)
    for i in range(len(words) - 1):
        gap = words[i + 1].start - words[i].end
        if 0 < gap <= MAX_GAP_FILL:
            words[i].end = words[i + 1].start
    return words


def align_segments(segments: list[Segment]) -> list[Segment]:
    """Clean every segment; drop empty/silent ones; fix ordering."""
    out: list[Segment] = []
    for seg in segments:
        words = smooth_gaps(list(seg.words or []))
        if not words:
            continue
        start = min(seg.start, words[0].start)
        end = max(seg.end, words[-1].end)
        text = seg.text.strip() or " ".join(w.text for w in words)
        out.append(Segment(start=max(0.0, start), end=max(start + 0.1, end), text=text, words=words))
    out.sort(key=lambda s: s.start)
    return out


def align_with_external(words: list[Word], *_args, **_kwargs) -> list[Word]:
    """Placeholder for a future Vietnamese forced aligner.

    Currently just runs the local cleanup so the pipeline keeps working.
    """
    return smooth_gaps(words)


def validate_words(words: list[Word]) -> list[str]:
    """Return human-readable problems (empty list = OK)."""
    problems: list[str] = []
    for i, w in enumerate(words):
        if w.start < 0 or w.end < 0:
            problems.append(f"Từ {i} ('{w.text}') có timestamp âm.")
        if w.end < w.start:
            problems.append(f"Từ {i} ('{w.text}') có end < start.")
        if i > 0 and w.start < words[i - 1].end - 1e-6:
            problems.append(f"Từ {i} ('{w.text}') chồng lấn từ trước.")
        if i > 0 and w.start < words[i - 1].start - 1e-6:
            problems.append(f"Từ {i} ('{w.text}') không theo thứ tự thời gian.")
    return problems
