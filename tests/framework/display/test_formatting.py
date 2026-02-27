"""Tests for framework/display/formatting.py — value formatters for display output."""

from framework.display.formatting import (
    format_prob,
    format_change,
    format_accuracy,
    format_loss,
)


class TestFormatProb:

    def test_scientific_notation(self):
        """Formats probability in scientific notation."""
        result = format_prob(0.00032)
        assert "3.2" in result or "3.20" in result
        assert "e" in result

    def test_with_baseline_shows_ratio(self):
        """With baseline, shows ratio suffix."""
        result = format_prob(0.002, baseline=0.001)
        assert "2.0x baseline" in result


class TestFormatChange:

    def test_improvement_shows_green(self):
        """Improvement (ratio >= 1) uses metric.improved markup."""
        result = format_change(1.0, 2.0, higher_is_better=True)
        assert "metric.improved" in result
        assert "2.0x" in result

    def test_degradation_shows_red(self):
        """Degradation (ratio < 1) uses metric.degraded markup."""
        result = format_change(2.0, 1.0, higher_is_better=True)
        assert "metric.degraded" in result

    def test_init_zero_returns_dash(self):
        """init=0 returns placeholder dash."""
        result = format_change(0, 1.0)
        assert "\u2014" in result

    def test_lower_is_better(self):
        """higher_is_better=False inverts the ratio (lower final = improvement)."""
        result = format_change(2.0, 1.0, higher_is_better=False)
        assert "metric.improved" in result
        assert "2.0x" in result


class TestFormatAccuracy:

    def test_excellent(self):
        """99+ uses excellent style."""
        result = format_accuracy(99.5)
        assert "accuracy.excellent" in result

    def test_good(self):
        """80-99 uses good style."""
        result = format_accuracy(85.0)
        assert "accuracy.good" in result

    def test_fair(self):
        """50-80 uses fair style."""
        result = format_accuracy(65.0)
        assert "accuracy.fair" in result

    def test_poor(self):
        """<50 uses poor style."""
        result = format_accuracy(30.0)
        assert "accuracy.poor" in result


class TestFormatLoss:

    def test_wraps_in_metric_value(self):
        """Loss formatted with metric.value markup."""
        result = format_loss(0.1234)
        assert "metric.value" in result
        assert "0.1234" in result
