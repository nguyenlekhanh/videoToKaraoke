"""Transcript cache: never rerun Whisper just because style changed.

Cache key = sha1(video_path, file size, mtime, model_size, vad flag).
Stored next to the output dir as <stem>_transcript_cache.json so a restart
of the app (or a styling-only iteration) reuses it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from models.types import Segment, Word


def cache_key(video: str, model_size: str, vad: bool) -> str:
    p = Path(video)
    h = hashlib.sha1()
    h.update(str(p.resolve()).encode("utf-8", "ignore"))
    try:
        st = p.stat()
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    except OSError:
        pass
    h.update(f"|{model_size}|{int(bool(vad))}".encode())
    return h.hexdigest()[:16]


def cache_path(out_dir: str | Path, video: str) -> Path:
    from utils.helpers import safe_output_stem

    return Path(out_dir) / f"{safe_output_stem(video)}_transcript_cache.json"


def save(segments: list[Segment], path: str | Path, key: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": key,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text,
             "words": [{"text": w.text, "start": w.start, "end": w.end,
                        "confidence": w.confidence} for w in s.words]}
            for s in segments
        ],
    }
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def load(path: str | Path, key: str) -> list[Segment] | None:
    """Return segments if cache exists AND key matches, else None."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("key") != key:
        return None
    out: list[Segment] = []
    for s in payload.get("segments", []):
        try:
            words = [Word(text=w["text"], start=float(w["start"]), end=float(w["end"]),
                          confidence=float(w.get("confidence", 1.0))) for w in s.get("words", [])]
        except (KeyError, TypeError, ValueError):
            continue
        if not words:
            continue
        out.append(Segment(start=float(s["start"]), end=float(s["end"]),
                           text=s.get("text", ""), words=words))
    return out or None
