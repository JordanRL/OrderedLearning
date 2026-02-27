"""Evaluation result dataclass.

Standardized container for evaluation output from ExperimentRunner.evaluate().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Standardized evaluation result.

    Attributes:
        metrics: All numeric metrics (loss, acc, prob, etc.).
            Common keys the framework display utilities know how to format:
            - loss, train_acc, val_acc (classification experiments)
            - seq_prob, avg_target_prob (LM probability experiments)
            - perplexity, first_token_prob (LM experiments)
            Experiment-specific keys pass through to sinks but framework
            display skips what it doesn't recognize.
        should_stop: Early stopping signal (e.g., grokking achieved).
        display_data: Non-numeric data for display (gen_text, per-token
            breakdown, etc.). Not sent to metric sinks.
    """
    metrics: dict[str, float] = field(default_factory=dict)
    should_stop: bool = False
    display_data: dict[str, Any] | None = None

    @classmethod
    def merge(cls, a: EvalResult | None, b: EvalResult | None) -> EvalResult | None:
        """Combine two EvalResults into one.

        Metrics and display_data dicts are merged (b overwrites a on conflict).
        should_stop is True if either result signals stop.

        Returns None if both inputs are None.
        """
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a

        merged_metrics = {**a.metrics, **b.metrics}
        merged_display = None
        if a.display_data is not None or b.display_data is not None:
            merged_display = {**(a.display_data or {}), **(b.display_data or {})}

        return cls(
            metrics=merged_metrics,
            should_stop=a.should_stop or b.should_stop,
            display_data=merged_display,
        )
