"""Shared dataclasses for transcript / words / segments."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    text: str
    start: float
    end: float
    confidence: float = 1.0

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
