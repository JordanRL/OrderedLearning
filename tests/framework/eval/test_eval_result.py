"""Tests for framework/eval/eval_result.py — EvalResult merge semantics."""

from framework.eval.eval_result import EvalResult


class TestEvalResultMerge:

    def test_merge_both_none(self):
        """merge(None, None) returns None."""
        assert EvalResult.merge(None, None) is None

    def test_merge_first_none(self):
        """merge(None, b) returns b."""
        b = EvalResult(metrics={"loss": 0.5})
        assert EvalResult.merge(None, b) is b

    def test_merge_second_none(self):
        """merge(a, None) returns a."""
        a = EvalResult(metrics={"loss": 0.3})
        assert EvalResult.merge(a, None) is a

    def test_merge_metrics_combined(self):
        """Merged result has metrics from both, b overwrites a on conflict."""
        a = EvalResult(metrics={"loss": 0.5, "acc": 0.8})
        b = EvalResult(metrics={"loss": 0.3, "val_acc": 0.9})
        merged = EvalResult.merge(a, b)
        assert merged.metrics == {"loss": 0.3, "acc": 0.8, "val_acc": 0.9}

    def test_merge_should_stop_or(self):
        """should_stop is True if either result signals stop."""
        a = EvalResult(should_stop=False)
        b = EvalResult(should_stop=True)
        assert EvalResult.merge(a, b).should_stop is True

    def test_merge_display_data(self):
        """Display data dicts are merged, b overwrites on conflict."""
        a = EvalResult(display_data={"text": "hello"})
        b = EvalResult(display_data={"chart": [1, 2]})
        merged = EvalResult.merge(a, b)
        assert merged.display_data == {"text": "hello", "chart": [1, 2]}
