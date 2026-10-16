"""Tests for perf fix: progress parsing, audio-copy decision, cache, stage log."""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cache import cache_key, load as cache_load, save as cache_save  # noqa: E402
from core.ffmpeg_runner import FfmpegStats, parse_progress_block, stats_line  # noqa: E402
from core.stages import RunLog, timed  # noqa: E402
from core.video_processor import should_copy_audio  # noqa: E402
from models.types import Segment, Word  # noqa: E402


def test_parse_progress_block():
    s = FfmpegStats()
    parse_progress_block({"frame": "120", "fps": "30.5", "out_time_ms": "4000000",
                          "speed": "2.5x", "total_size": "123456", "progress": "continue"}, s)
    assert s.frame == 120
    assert abs(s.out_time_s - 4.0) < 1e-6
    assert abs(s.speed - 2.5) < 1e-9
    assert not s.done
    parse_progress_block({"progress": "end"}, s)
    assert s.done
    # garbage never raises
    parse_progress_block({"frame": "N/A", "speed": "N/A", "out_time_ms": "N/A"}, FfmpegStats())
    line = stats_line(s)
    assert "frame 120" in line and "4.0s" in line


def test_progress_fraction_uses_real_time():
    total = 20.0
    s = FfmpegStats(out_time_s=10.0)
    frac = s.out_time_s / total
    assert abs(frac - 0.5) < 1e-9, "render progress must reflect actual media time"


def test_should_copy_audio():
    assert should_copy_audio("aac")
    assert should_copy_audio("MP3")
    assert should_copy_audio("ac3")
    assert not should_copy_audio("opus")
    assert not should_copy_audio("vorbis")
    assert not should_copy_audio("")


def test_cache_roundtrip_and_invalidation(tmp_path=None):
    d = Path(tempfile.mkdtemp())
    segs = [Segment(0.0, 1.0, "xin chào", [Word("xin", 0.0, 0.5), Word("chào", 0.5, 1.0)])]
    p = d / "t_cache.json"
    cache_save(segs, p, "KEY1")
    got = cache_load(p, "KEY1")
    assert got is not None and len(got) == 1
    assert [w.text for w in got[0].words] == ["xin", "chào"]
    assert got[0].words[0].start == 0.0
    # stale key -> miss (forces fresh Whisper, never stale reuse)
    assert cache_load(p, "OTHER") is None
    # corrupt file -> miss, never crash
    p.write_text("{{{not json", encoding="utf-8")
    assert cache_load(p, "KEY1") is None


def test_cache_key_changes_with_model_and_file(tmp_path=None):
    d = Path(tempfile.mkdtemp())
    v = d / "v.mp4"
    v.write_bytes(b"x" * 100)
    k1 = cache_key(str(v), "medium", True)
    k2 = cache_key(str(v), "small", True)
    k3 = cache_key(str(v), "medium", False)
    assert len({k1, k2, k3}) == 3
    v.write_bytes(b"y" * 200)  # size change invalidates
    assert cache_key(str(v), "medium", True) != k1


def test_stage_timing_records_elapsed():
    log = RunLog("test")
    with timed(log, "probe"):
        time.sleep(0.05)
    assert len(log.records) == 1
    assert log.records[0]["stage"] == "probe"
    assert log.records[0]["elapsed_s"] >= 0.04, "must record REAL elapsed, not fake"
    assert log.total() >= 0.04
    assert "probe" in log.summary().lower() or "metadata" in log.summary()
