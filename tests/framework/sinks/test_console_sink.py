"""Tests for framework/sinks/console.py — ConsoleSink."""

import pytest
from framework.sinks.console import ConsoleSink
from framework.hooks.hook_point import HookPoint


class TestClassifyChange:

    def test_both_zero(self):
        assert ConsoleSink._classify_change(0.0, 0.0) == 'flat'

    def test_zero_to_positive(self):
        assert ConsoleSink._classify_change(0.0, 1.0) == 'up'

    def test_zero_to_negative(self):
        assert ConsoleSink._classify_change(0.0, -1.0) == 'down'

    def test_up(self):
        assert ConsoleSink._classify_change(1.0, 1.5) == 'up'

    def test_down(self):
        assert ConsoleSink._classify_change(1.0, 0.5) == 'down'

    def test_flat_within_threshold(self):
        """Changes < 1% are flat."""
        assert ConsoleSink._classify_change(100.0, 100.5) == 'flat'


class TestUpdateHistoryAndTrend:

    def test_first_value_no_trend(self):
        """Single value produces no trend."""
        sink = ConsoleSink()
        result = sink._update_history_and_trend("key", 1.0)
        assert result == ""

    def test_two_values_produce_trend(self):
        """Two values produce an up trend indicator."""
        sink = ConsoleSink()
        sink._update_history_and_trend("key", 1.0)
        result = sink._update_history_and_trend("key", 2.0)
        assert ConsoleSink._TREND_UP in result

    def test_three_values_produce_two_trends(self):
        """Three values produce two trend indicators."""
        sink = ConsoleSink()
        sink._update_history_and_trend("key", 1.0)
        sink._update_history_and_trend("key", 2.0)
        result = sink._update_history_and_trend("key", 3.0)
        # Should have 2 trend characters
        assert ConsoleSink._TREND_UP in result

    def test_history_capped_at_3(self):
        """History doesn't grow beyond 3 values."""
        sink = ConsoleSink()
        for i in range(10):
            sink._update_history_and_trend("key", float(i))
        assert len(sink._metric_history["key"]) == 3

    def test_list_values_averaged(self):
        """List values are reduced to mean."""
        sink = ConsoleSink()
        sink._update_history_and_trend("key", 1.0)
        result = sink._update_history_and_trend("key", [2.0, 4.0])  # mean=3.0
        assert result != ""

    def test_empty_list_returns_empty(self):
        """Empty list returns no trend."""
        sink = ConsoleSink()
        assert sink._update_history_and_trend("key", []) == ""

    def test_integer_values_skipped(self):
        """Integer values (indices/counts) produce no trend."""
        sink = ConsoleSink()
        sink._update_history_and_trend("key", 1)
        result = sink._update_history_and_trend("key", 2)
        assert result == ""

    def test_non_numeric_returns_empty(self):
        """Non-numeric values return empty trend."""
        sink = ConsoleSink()
        assert sink._update_history_and_trend("key", "text") == ""


class TestConsoleSinkEmit:

    def test_empty_metrics_no_op(self):
        """Empty metrics dict is a no-op."""
        sink = ConsoleSink()
        sink.emit({}, 0, HookPoint.POST_STEP)

    def test_normal_mode_buffers_non_snapshot(self):
        """Non-SNAPSHOT metrics are buffered in normal mode."""
        sink = ConsoleSink()
        sink.emit({"hook/metric": 1.0}, 0, HookPoint.POST_STEP)
        assert len(sink._buffered) == 1

    def test_normal_mode_flushes_on_snapshot(self):
        """SNAPSHOT flushes buffer and prints table."""
        sink = ConsoleSink()
        sink.emit({"hook/metric": 1.0}, 0, HookPoint.POST_STEP)
        sink.emit({"hook/metric2": 2.0}, 0, HookPoint.SNAPSHOT)
        assert len(sink._buffered) == 0

    def test_emit_to_main_filters_per_param(self):
        """Per-parameter metrics (3+ slashes) are excluded from console."""
        sink = ConsoleSink()
        metrics = {
            "hook/metric": 1.0,
            "hook/metric/param.weight": 2.0,  # per-param, should be hidden
        }
        sink.emit(metrics, 0, HookPoint.SNAPSHOT)


class TestConsoleSinkSetRunContext:

    def test_clears_state(self):
        """set_run_context clears all metric state."""
        sink = ConsoleSink()
        sink._metric_history["key"] = [1.0, 2.0]
        sink._current_values["key"] = 1.0
        sink._trend_cache["key"] = "up"
        sink.set_run_context(strategy="new")
        assert len(sink._metric_history) == 0
        assert len(sink._current_values) == 0
        assert len(sink._trend_cache) == 0
