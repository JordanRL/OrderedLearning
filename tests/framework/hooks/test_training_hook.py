"""Tests for framework/hooks/training_hook.py — TrainingHook base class defaults."""

import torch

from framework.hooks.training_hook import TrainingHook
from framework.hooks.hook_point import HookPoint


class ConcreteHook(TrainingHook):
    """Minimal concrete hook for testing base class defaults."""
    name = "test_concrete"
    hook_points = {HookPoint.POST_STEP}

    def compute(self, ctx, **state):
        return {"value": 1.0}


class TestTrainingHookDefaults:

    def test_get_state_tensors_empty(self):
        """Default get_state_tensors() returns empty dict."""
        hook = ConcreteHook()
        assert hook.get_state_tensors() == {}

    def test_set_state_tensors_noop(self):
        """Default set_state_tensors() doesn't crash."""
        hook = ConcreteHook()
        hook.set_state_tensors({"x": torch.zeros(3)})

    def test_reset_noop(self):
        """Default reset() doesn't crash."""
        hook = ConcreteHook()
        hook.reset()

    def test_set_run_context_noop(self):
        """Default set_run_context() doesn't crash."""
        hook = ConcreteHook()
        hook.set_run_context(strategy="test", output_dir="/tmp")

    def test_set_reference_weights_stores(self):
        """set_reference_weights() stores the reference as self._ref."""
        hook = ConcreteHook()
        sentinel = object()
        hook.set_reference_weights(sentinel)
        assert hook._ref is sentinel

    def test_describe_metrics_empty(self):
        """Default describe_metrics() returns empty list."""
        hook = ConcreteHook()
        assert hook.describe_metrics() == []

    def test_print_metric_descriptions_no_error(self):
        """print_metric_descriptions() runs without error on hook with no metrics."""
        hook = ConcreteHook()
        hook.print_metric_descriptions()  # should not raise
