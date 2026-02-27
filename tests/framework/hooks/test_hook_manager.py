"""Tests for framework/hooks/manager.py — HookManager orchestration."""

import pytest
from typing import Any

from framework.hooks.hook_point import HookPoint, StepSchedule
from framework.hooks.training_hook import TrainingHook
from framework.hooks.manager import HookManager
from framework.contexts.run_context import RunContext
from framework.capabilities import (
    TrainingCapabilities, TrainingParadigm,
    ModelCapability, GradientAvailability,
    HookNeeds, HookRequirements,
)


# ---- Minimal mock hooks ----

class DummyObserver(TrainingHook):
    """Minimal observer hook that returns a fixed metric."""
    name = "dummy_observer"
    hook_points = {HookPoint.POST_EPOCH, HookPoint.SNAPSHOT}

    def compute(self, ctx, **state):
        return {"value": 42.0}


class DummyStepHook(TrainingHook):
    """Observer that fires at POST_STEP with stride=2."""
    name = "dummy_step"
    hook_points = {HookPoint.POST_STEP}
    step_schedule = StepSchedule(mode='stride', stride=2)

    def compute(self, ctx, **state):
        return {"step_val": 1.0}


class GradNeedingHook(TrainingHook):
    """Hook that declares ACCUMULATED_GRADS need."""
    name = "grad_needer"
    hook_points = {HookPoint.SNAPSHOT}
    needs = HookNeeds.ACCUMULATED_GRADS

    def compute(self, ctx, **state):
        return {}


class BackpropOnlyHook(TrainingHook):
    """Hook with paradigm requirement: BACKPROP only."""
    name = "backprop_only"
    hook_points = {HookPoint.POST_EPOCH}
    requires = HookRequirements(paradigm=TrainingParadigm.BACKPROP)

    def compute(self, ctx, **state):
        return {"bp": 1.0}


# ---- Fixtures ----

@pytest.fixture
def dummy_ctx():
    return RunContext(hook_point=HookPoint.POST_EPOCH, epoch=1)


# ---- Tests ----

class TestHookManagerConstruction:

    def test_empty_manager(self):
        """HookManager with no hooks is valid."""
        hm = HookManager(hooks=[], step_metrics_log=None)
        assert hm.has_hooks_at(HookPoint.POST_EPOCH) is False

    def test_hooks_indexed_by_point(self):
        """Pre-instantiated hooks are indexed under their hook_points."""
        hm = HookManager(hooks=[DummyObserver()], step_metrics_log=None)
        assert hm.has_hooks_at(HookPoint.POST_EPOCH) is True
        assert hm.has_hooks_at(HookPoint.SNAPSHOT) is True
        assert hm.has_hooks_at(HookPoint.PRE_STEP) is False


class TestHookManagerFire:

    def test_fire_collects_namespaced_metrics(self, dummy_ctx):
        """fire() returns metrics namespaced as 'hook_name/metric_key'."""
        hm = HookManager(hooks=[DummyObserver()], step_metrics_log=None)
        metrics = hm.fire(HookPoint.POST_EPOCH, dummy_ctx)
        assert "dummy_observer/value" in metrics
        assert metrics["dummy_observer/value"] == 42.0

    def test_fire_empty_at_unregistered_point(self, dummy_ctx):
        """fire() at a point with no hooks returns empty dict."""
        hm = HookManager(hooks=[DummyObserver()], step_metrics_log=None)
        metrics = hm.fire(HookPoint.PRE_STEP, dummy_ctx)
        assert metrics == {}


class TestHookManagerCapabilities:

    def test_set_capabilities_filters_incompatible(self):
        """set_capabilities() excludes hooks whose requirements are not met."""
        hook = BackpropOnlyHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        assert hm.has_hooks_at(HookPoint.POST_EPOCH) is True

        # Set evolutionary capabilities — backprop-only hook should be excluded
        evo_caps = TrainingCapabilities(
            paradigm=TrainingParadigm.EVOLUTIONARY,
            model_capabilities=ModelCapability.PARAMETERS,
            gradient_availability=GradientAvailability.NONE,
        )
        hm.set_capabilities(evo_caps)
        assert hm.has_hooks_at(HookPoint.POST_EPOCH) is False

    def test_needs_grad_accumulation(self):
        """needs_grad_accumulation() returns True when any hook declares the need."""
        hm = HookManager(hooks=[GradNeedingHook()], step_metrics_log=None)
        assert hm.needs_grad_accumulation() is True

    def test_no_needs_grad_accumulation(self):
        """needs_grad_accumulation() returns False when no hook declares the need."""
        hm = HookManager(hooks=[DummyObserver()], step_metrics_log=None)
        assert hm.needs_grad_accumulation() is False


class TestHookManagerStepScheduling:

    def test_advance_step_increments(self):
        """advance_step() increments global_step from -1."""
        hm = HookManager(hooks=[], step_metrics_log=None)
        assert hm.global_step == -1
        hm.advance_step()
        assert hm.global_step == 0
        hm.advance_step()
        assert hm.global_step == 1

    def test_step_schedule_gates_active_hooks(self):
        """has_active_step_hooks respects the hook's step_schedule."""
        hook = DummyStepHook()  # stride=2, fires at step 0, 2, 4, ...
        hm = HookManager(hooks=[hook], step_metrics_log=None)

        hm.advance_step()  # step=0
        assert hm.has_active_step_hooks(HookPoint.POST_STEP) is True

        hm.advance_step()  # step=1
        assert hm.has_active_step_hooks(HookPoint.POST_STEP) is False

        hm.advance_step()  # step=2
        assert hm.has_active_step_hooks(HookPoint.POST_STEP) is True
