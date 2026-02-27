"""Framework display utilities for experiment output.

Standardized display functions called by the framework loops and
ExperimentRunner methods. All functions obtain the OLConsole singleton
internally and use semantic theme styles from console/themes.py.

This subpackage re-exports all public names so that existing callers
using ``from framework import display`` followed by
``display.function_name()`` continue to work unchanged.
"""

from .lifecycle import (
    display_experiment_banner,
    display_condition_header,
    display_training_start,
    display_resume_info,
)
from .progress import (
    TASK_TRAINING, TASK_EPOCH, TASK_BATCH, TASK_EVAL,
    training_progress_start, training_progress_update, training_progress_end,
    epoch_progress_start, epoch_progress_update, epoch_progress_end,
    batch_progress_start, batch_progress_update, batch_progress_end,
    eval_progress_start, eval_progress_update, eval_progress_end,
)
from .results import (
    display_eval_update,
    display_final_results,
    display_comparison_table,
    display_post_live_summary,
    display_phase_transition,
    display_grokking_achieved,
)
from .formatting import (
    format_prob,
    format_change,
    format_accuracy,
    format_loss,
)

__all__ = [
    # Lifecycle
    'display_experiment_banner', 'display_condition_header',
    'display_training_start', 'display_resume_info',
    # Progress
    'TASK_TRAINING', 'TASK_EPOCH', 'TASK_BATCH', 'TASK_EVAL',
    'training_progress_start', 'training_progress_update', 'training_progress_end',
    'epoch_progress_start', 'epoch_progress_update', 'epoch_progress_end',
    'batch_progress_start', 'batch_progress_update', 'batch_progress_end',
    'eval_progress_start', 'eval_progress_update', 'eval_progress_end',
    # Results
    'display_eval_update', 'display_final_results', 'display_comparison_table',
    'display_post_live_summary', 'display_phase_transition', 'display_grokking_achieved',
    # Formatting
    'format_prob', 'format_change', 'format_accuracy', 'format_loss',
]
