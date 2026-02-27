"""Tests for HookManager.fire() mechanics, sink dispatch, step buffering, and lifecycle."""

import pytest
import torch
from typing import Any
from unittest.mock import MagicMock

from framework.hooks.hook_point import HookPoint, StepSchedule
from framework.hooks.training_hook import TrainingHook
from framework.hooks.intervention_hook import InterventionHook
from framework.hooks.manager import HookManager
from framework.contexts.run_context import RunContext
from framework.capabilities import HookNeeds, HookRequirements


# ---- Mock hooks ----

class ObserverA(TrainingHook):
    name = "observer_a"
    hook_points = {HookPoint.POST_EPOCH, HookPoint.SNAPSHOT}

    def compute(self, ctx, **state):
        return {"metric_a": 1.0}


class ObserverB(TrainingHook):
    name = "observer_b"
    hook_points = {HookPoint.POST_EPOCH}

    def compute(self, ctx, **state):
        return {"metric_b": 2.0}


class EmptyObserver(TrainingHook):
    """Observer that returns empty dict."""
    name = "empty_obs"
    hook_points = {HookPoint.POST_EPOCH}

    def compute(self, ctx, **state):
        return {}


class StepObserver(TrainingHook):
    """Observer that fires at POST_STEP."""
    name = "step_obs"
    hook_points = {HookPoint.POST_STEP}

    def compute(self, ctx, **state):
        return {"step_metric": ctx.step or 0}


class RngConsumingObserver(TrainingHook):
    """Observer that consumes RNG during compute."""
    name = "rng_consumer"
    hook_points = {HookPoint.POST_EPOCH}

    def compute(self, ctx, **state):
        torch.randn(10)  # consume RNG
        return {"noise": 1.0}


class EpochGatedHook(TrainingHook):
    """Hook with epoch-gated loop_points: active only at epochs 5-10."""
    name = "epoch_gated"
    hook_points = {HookPoint.POST_EPOCH}
    loop_points = {
        'epoch': {HookPoint.POST_EPOCH: (5, 10)},
    }

    def compute(self, ctx, **state):
        return {"gated": 1.0}


class ResettableHook(TrainingHook):
    """Hook that tracks whether reset() was called."""
    name = "resettable"
    hook_points = {HookPoint.POST_EPOCH}

    def __init__(self):
        self.was_reset = False

    def compute(self, ctx, **state):
        return {"val": 1.0}

    def reset(self):
        self.was_reset = True


# ---- Fixtures ----

@pytest.fixture
def post_epoch_ctx():
    return RunContext(hook_point=HookPoint.POST_EPOCH, epoch=1)


@pytest.fixture
def post_step_ctx():
    return RunContext(hook_point=HookPoint.POST_STEP, epoch=0, step=5)


# ---- fire() observer mechanics ----

class TestFireObserverMechanics:

    def test_multiple_observers_merged(self, post_epoch_ctx):
        """Multiple observers at same point → all metrics merged."""
        hm = HookManager(hooks=[ObserverA(), ObserverB()], step_metrics_log=None)
        metrics = hm.fire(HookPoint.POST_EPOCH, post_epoch_ctx)
        assert "observer_a/metric_a" in metrics
        assert "observer_b/metric_b" in metrics

    def test_empty_observer_no_metrics(self, post_epoch_ctx):
        """Observer returning empty dict adds no metrics."""
        hm = HookManager(hooks=[EmptyObserver()], step_metrics_log=None)
        metrics = hm.fire(HookPoint.POST_EPOCH, post_epoch_ctx)
        assert len(metrics) == 0

    def test_post_step_metrics_buffered(self, post_step_ctx):
        """fire() at POST_STEP buffers metrics (not dispatched to sinks)."""
        mock_sink = MagicMock()
        hm = HookManager(
            hooks=[StepObserver()], sinks=[mock_sink], step_metrics_log=None,
        )
        hm.advance_step()  # step=0
        hm.fire(HookPoint.POST_STEP, post_step_ctx)
        # Sink should NOT have been called — step metrics are buffered
        mock_sink.emit.assert_not_called()
        # But buffer should have the metric
        assert "step_obs/step_metric" in hm._step_metrics_buffer

    def test_post_epoch_metrics_dispatched(self, post_epoch_ctx):
        """fire() at POST_EPOCH dispatches metrics immediately to sinks."""
        mock_sink = MagicMock()
        hm = HookManager(
            hooks=[ObserverA()], sinks=[mock_sink], step_metrics_log=None,
        )
        hm.fire(HookPoint.POST_EPOCH, post_epoch_ctx)
        mock_sink.emit.assert_called_once()
        call_args = mock_sink.emit.call_args
        assert "observer_a/metric_a" in call_args[0][0]


# ---- Step metrics buffering ----

class TestStepMetricsBuffering:

    def test_buffer_accumulates_as_lists(self):
        """Multiple POST_STEP fires accumulate per-key lists."""
        hm = HookManager(hooks=[StepObserver()], step_metrics_log=None)

        for step in range(3):
            hm.advance_step()
            ctx = RunContext(hook_point=HookPoint.POST_STEP, epoch=0, step=step)
            hm.fire(HookPoint.POST_STEP, ctx)

        buffer = hm._step_metrics_buffer
        assert "step_obs/step_metric" in buffer
        assert len(buffer["step_obs/step_metric"]) == 3

    def test_flush_dispatches_and_clears(self):
        """flush_step_metrics() dispatches buffer to sinks and clears it."""
        mock_sink = MagicMock()
        hm = HookManager(
            hooks=[StepObserver()], sinks=[mock_sink], step_metrics_log=None,
        )
        hm.advance_step()
        ctx = RunContext(hook_point=HookPoint.POST_STEP, epoch=0, step=0)
        hm.fire(HookPoint.POST_STEP, ctx)

        assert len(hm._step_metrics_buffer) > 0
        hm.flush_step_metrics(epoch=0)
        assert len(hm._step_metrics_buffer) == 0
        mock_sink.emit.assert_called_once()


# ---- Epoch gating ----

class TestEpochGating:

    def test_hook_active_within_range(self):
        """Epoch-gated hook fires within its epoch range."""
        hm = HookManager(
            hooks=[EpochGatedHook()], step_metrics_log=None, loop_type='epoch',
        )
        ctx = RunContext(hook_point=HookPoint.POST_EPOCH, epoch=7)
        metrics = hm.fire(HookPoint.POST_EPOCH, ctx)
        assert "epoch_gated/gated" in metrics

    def test_hook_inactive_outside_range(self):
        """Epoch-gated hook does not fire outside its epoch range."""
        hm = HookManager(
            hooks=[EpochGatedHook()], step_metrics_log=None, loop_type='epoch',
        )
        ctx = RunContext(hook_point=HookPoint.POST_EPOCH, epoch=15)
        metrics = hm.fire(HookPoint.POST_EPOCH, ctx)
        assert metrics == {}

    def test_none_epoch_is_conservative(self):
        """_is_hook_active with None epoch always returns True."""
        hm = HookManager(
            hooks=[EpochGatedHook()], step_metrics_log=None, loop_type='epoch',
        )
        ctx = RunContext(hook_point=HookPoint.POST_EPOCH, epoch=15)
        # has_hooks_at with epoch=None should return True (conservative)
        assert hm.has_hooks_at(HookPoint.POST_EPOCH, epoch=None) is True


# ---- RNG isolation ----

class TestRngIsolation:

    def test_rng_state_preserved_across_fire(self, post_epoch_ctx):
        """RNG state before fire() equals state after fire() (hook RNG is isolated)."""
        hm = HookManager(hooks=[RngConsumingObserver()], step_metrics_log=None)

        torch.manual_seed(42)
        rng_before = torch.random.get_rng_state().clone()
        expected = torch.randn(5).clone()

        # Reset to same state
        torch.random.set_rng_state(rng_before)

        # Fire (hook consumes RNG internally)
        hm.fire(HookPoint.POST_EPOCH, post_epoch_ctx)

        # After fire, RNG should be restored — next randn should match expected
        actual = torch.randn(5)
        assert torch.allclose(actual, expected)


# ---- Sink dispatch ----

class TestSinkDispatch:

    def test_set_run_context_propagates(self):
        """set_run_context() propagates to all hooks and sinks."""
        mock_sink = MagicMock()
        hook = ObserverA()
        hm = HookManager(hooks=[hook], sinks=[mock_sink], step_metrics_log=None)
        hm.set_run_context(strategy='test_strat')
        mock_sink.set_run_context.assert_called_once_with(strategy='test_strat')

    def test_flush_sinks_calls_flush(self):
        """flush_sinks() calls flush() on all registered sinks."""
        mock_sink = MagicMock()
        hm = HookManager(hooks=[], sinks=[mock_sink], step_metrics_log=None)
        hm.flush_sinks()
        mock_sink.flush.assert_called_once()

    def test_emit_metrics_dispatches_directly(self, post_epoch_ctx):
        """emit_metrics() dispatches directly to sinks."""
        mock_sink = MagicMock()
        hm = HookManager(hooks=[], sinks=[mock_sink], step_metrics_log=None)
        hm.emit_metrics({"custom/metric": 42.0}, step=5, hook_point=HookPoint.POST_EPOCH)
        mock_sink.emit.assert_called_once()


# ---- Lifecycle methods ----

class TestLifecycleMethods:

    def test_reset_all_clears_state(self):
        """reset_all() clears global_step, buffers, and calls hook.reset()."""
        hook = ResettableHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        hm.advance_step()
        hm._step_metrics_buffer = {"key": [1, 2]}
        hm._last_metrics = {HookPoint.POST_EPOCH: {"val": 1}}

        hm.reset_all()

        assert hm.global_step == -1
        assert hm._step_metrics_buffer == {}
        assert hm._last_metrics == {}
        assert hook.was_reset is True

    def test_get_last_metrics(self, post_epoch_ctx):
        """get_last_metrics() returns metrics from most recent fire()."""
        hm = HookManager(hooks=[ObserverA()], step_metrics_log=None)
        hm.fire(HookPoint.POST_EPOCH, post_epoch_ctx)
        last = hm.get_last_metrics(HookPoint.POST_EPOCH)
        assert "observer_a/metric_a" in last

    def test_get_last_metrics_empty_if_never_fired(self):
        """get_last_metrics() returns empty dict if never fired at that point."""
        hm = HookManager(hooks=[], step_metrics_log=None)
        assert hm.get_last_metrics(HookPoint.SNAPSHOT) == {}

    def test_get_hook_by_name(self):
        """get_hook() returns hook instance by name; None for unknown."""
        hook = ObserverA()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        assert hm.get_hook("observer_a") is hook
        assert hm.get_hook("nonexistent") is None
