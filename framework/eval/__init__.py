"""Evaluation infrastructure: result container and target evaluation.

Provides EvalResult for standardized evaluation output, and eval target
utilities for trigger/completion pair evaluation during training.
"""

from .eval_result import EvalResult
from .eval_targets import (
    EvalTarget, DEFAULT_TARGETS,
    prepare_eval_targets, evaluate_target, evaluate_all_targets,
    load_targets_from_file, build_eval_targets, build_sink_metrics,
)

__all__ = [
    'EvalResult',
    'EvalTarget', 'DEFAULT_TARGETS',
    'prepare_eval_targets', 'evaluate_target', 'evaluate_all_targets',
    'load_targets_from_file', 'build_eval_targets', 'build_sink_metrics',
]
