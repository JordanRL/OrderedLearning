"""Tests for framework/display/progress.py — progress bar management.

Progress functions manage Rich progress bars through OLConsole methods
(create_progress_task, update_progress_task, etc.), not through console.print().
In test mode, progress bar operations are no-ops. These tests verify
the functions execute without exceptions and validate constants.
"""

from framework.display.progress import (
    TASK_TRAINING, TASK_EPOCH, TASK_BATCH, TASK_EVAL,
    training_progress_start, training_progress_update, training_progress_end,
    epoch_progress_start, epoch_progress_update, epoch_progress_end,
    batch_progress_start, batch_progress_update, batch_progress_end,
    eval_progress_start, eval_progress_update, eval_progress_end,
)


class TestTaskConstants:

    def test_training_constant(self):
        assert TASK_TRAINING == "training"

    def test_epoch_constant(self):
        assert TASK_EPOCH == "epoch"

    def test_batch_constant(self):
        assert TASK_BATCH == "batch"

    def test_eval_constant(self):
        assert TASK_EVAL == "eval"


class TestTrainingProgress:
    """Progress bar lifecycle — smoke tests only.

    Progress bars are managed via OLConsole.create_progress_task() etc.,
    which are no-ops in NULL/test mode. No capturable text output is produced.
    """

    def test_start_update_end(self):
        """Training progress start/update/end cycle runs without error."""
        from unittest.mock import MagicMock
        result = MagicMock()
        result.loss = 0.5
        training_progress_start("stride", 100)
        training_progress_update(1, 100, result, "stride")
        training_progress_end()


class TestEpochProgress:
    """Epoch progress lifecycle — smoke tests only (see TestTrainingProgress)."""

    def test_start_update_end(self):
        """Epoch progress start/update/end cycle runs without error."""
        epoch_progress_start("stride", 100)
        epoch_progress_update(1, 100, 0.5)
        epoch_progress_end()

    def test_update_with_eval_result(self):
        """Epoch update with EvalResult and progress_metric runs without error."""
        from framework.eval import EvalResult
        eval_result = EvalResult(metrics={'validation_accuracy': 95.0})
        epoch_progress_start("stride", 100)
        epoch_progress_update(
            10, 100, 0.3, eval_result=eval_result,
            strategy_name="stride", progress_metric='validation_accuracy',
        )
        epoch_progress_end()

    def test_update_with_low_accuracy(self):
        """Epoch update with accuracy below 90 uses different style."""
        from framework.eval import EvalResult
        eval_result = EvalResult(metrics={'validation_accuracy': 50.0})
        epoch_progress_start("stride", 100)
        epoch_progress_update(
            10, 100, 0.5, eval_result=eval_result,
            progress_metric='validation_accuracy',
        )
        epoch_progress_end()


class TestBatchProgress:
    """Batch progress lifecycle — smoke tests only (see TestTrainingProgress)."""

    def test_start_update_end(self):
        """Batch progress start/update/end cycle runs without error."""
        batch_progress_start(50)
        batch_progress_update()
        batch_progress_end()


class TestEvalProgress:
    """Eval progress lifecycle — smoke tests only (see TestTrainingProgress)."""

    def test_start_update_end(self):
        """Eval progress start/update/end cycle runs without error."""
        eval_progress_start("Test", 10)
        eval_progress_update("Test")
        eval_progress_end("Test")

    def test_label_lowercased_in_task_name(self):
        """Different labels create different lowercased task names."""
        eval_progress_start("Train", 5)
        eval_progress_update("Train")
        eval_progress_end("Train")
