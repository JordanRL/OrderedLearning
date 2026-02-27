"""Tests for framework/hooks/hook_point.py — HookPoint enum and StepSchedule."""

import pytest
from framework.hooks.hook_point import HookPoint, MetricInfo, StepSchedule, STEP_HOOK_POINTS


class TestHookPoint:

    def test_all_five_values(self):
        """HookPoint has exactly 5 members."""
        assert len(HookPoint) == 5

    def test_step_hook_points_frozenset(self):
        """STEP_HOOK_POINTS contains PRE_STEP and POST_STEP."""
        assert STEP_HOOK_POINTS == frozenset({HookPoint.PRE_STEP, HookPoint.POST_STEP})


class TestMetricInfo:

    def test_frozen(self):
        """MetricInfo is frozen (immutable)."""
        info = MetricInfo(name="loss", description="Training loss")
        with pytest.raises(AttributeError):
            info.name = "changed"


class TestStepSchedule:

    def test_continual_always_active(self):
        """Continual mode fires every step (after warmup)."""
        schedule = StepSchedule(mode='continual')
        assert schedule.is_active(0) is True
        assert schedule.is_active(1) is True
        assert schedule.is_active(999) is True

    def test_continual_with_warmup(self):
        """Continual mode skips steps before warmup."""
        schedule = StepSchedule(mode='continual', warmup=10)
        assert schedule.is_active(0) is False
        assert schedule.is_active(9) is False
        assert schedule.is_active(10) is True
        assert schedule.is_active(11) is True

    def test_stride_mode(self):
        """Stride mode fires every N steps (modulo-based)."""
        schedule = StepSchedule(mode='stride', stride=5)
        assert schedule.is_active(0) is True   # 0 % 5 == 0
        assert schedule.is_active(1) is False
        assert schedule.is_active(5) is True
        assert schedule.is_active(10) is True
        assert schedule.is_active(7) is False

    def test_stride_with_warmup(self):
        """Stride mode with warmup skips early steps even on stride boundary."""
        schedule = StepSchedule(mode='stride', stride=5, warmup=3)
        assert schedule.is_active(0) is False   # before warmup
        assert schedule.is_active(5) is True    # on stride boundary, past warmup

    def test_burst_mode(self):
        """Burst mode fires burst_length consecutive steps every stride steps."""
        schedule = StepSchedule(mode='burst', stride=10, burst_length=3)
        # step % 10 < 3 fires
        assert schedule.is_active(0) is True    # 0 % 10 = 0 < 3
        assert schedule.is_active(1) is True    # 1 % 10 = 1 < 3
        assert schedule.is_active(2) is True    # 2 % 10 = 2 < 3
        assert schedule.is_active(3) is False   # 3 % 10 = 3, not < 3
        assert schedule.is_active(10) is True   # 10 % 10 = 0 < 3
        assert schedule.is_active(13) is False  # 13 % 10 = 3, not < 3
