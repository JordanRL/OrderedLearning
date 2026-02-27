"""Evaluation metrics context for hooks.

Generic (paradigm-agnostic) frozen dataclass that carries evaluation
metrics. The HookManager passes this to hooks as a state kwarg.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalMetrics:
    """Generic evaluation metrics dict."""
    metrics: dict[str, float] = field(default_factory=dict)
