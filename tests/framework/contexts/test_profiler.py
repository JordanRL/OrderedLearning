"""Tests for framework/contexts/profiler.py — CheckpointProfiler."""

import time

from framework.contexts.profiler import CheckpointProfiler


class TestCheckpointProfilerDisabled:

    def test_section_noop_when_disabled(self):
        """section() records nothing when enabled=False."""
        profiler = CheckpointProfiler(enabled=False)
        with profiler.section("test"):
            pass
        assert len(profiler._timings) == 0


class TestCheckpointProfilerEnabled:

    def test_section_records_timing(self):
        """section() records timing when enabled=True."""
        profiler = CheckpointProfiler(enabled=True)
        with profiler.section("test_section"):
            time.sleep(0.001)
        assert "test_section" in profiler._timings
        assert len(profiler._timings["test_section"]) == 1
        assert profiler._timings["test_section"][0] > 0

    def test_multiple_sections_accumulate(self):
        """Multiple calls to the same section accumulate."""
        profiler = CheckpointProfiler(enabled=True)
        for _ in range(3):
            with profiler.section("repeated"):
                pass
        assert len(profiler._timings["repeated"]) == 3

    def test_report_returns_summary(self):
        """report() returns dict with count/total_ms/mean_ms."""
        profiler = CheckpointProfiler(enabled=True)
        with profiler.section("op"):
            pass
        summary = profiler.report()
        assert "op" in summary
        assert summary["op"]["count"] == 1
        assert summary["op"]["total_ms"] >= 0
        assert summary["op"]["mean_ms"] >= 0

    def test_reset_clears_timings(self):
        """reset() clears all accumulated timings."""
        profiler = CheckpointProfiler(enabled=True)
        with profiler.section("something"):
            pass
        profiler.reset()
        assert len(profiler._timings) == 0
