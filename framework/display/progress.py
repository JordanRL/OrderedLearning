"""Progress bar management functions.

Create, update, and close progress bars for training, epoch, batch,
and evaluation phases.
"""

from console import OLConsole


# Progress task name constants
TASK_TRAINING = "training"
TASK_EPOCH = "epoch"
TASK_BATCH = "batch"
TASK_EVAL = "eval"


def training_progress_start(strategy_name, total):
    """Create the main training progress bar (step-based)."""
    console = OLConsole()
    console.create_progress_task(
        TASK_TRAINING,
        f"[strategy]{strategy_name}[/strategy]",
        total=total,
    )


def training_progress_update(step, total_steps, result, strategy_name):
    """Update step-based progress bar with current step and loss."""
    console = OLConsole()
    if console.is_live:
        desc = f"[strategy]{strategy_name}[/strategy] [label]step {step}[/label]"
    else:
        desc = (f"[strategy]{strategy_name}[/strategy] "
                f"[label]loss=[/label][metric.value]{float(result.loss):.4f}[/metric.value]")
    console.update_progress_task(TASK_TRAINING, description=desc, advance=1)


def training_progress_end():
    """Close the step-based training progress bar."""
    OLConsole().progress_stop()


def epoch_progress_start(strategy_name, total_epochs):
    """Create epoch-level progress bar."""
    console = OLConsole()
    console.create_progress_task(
        TASK_EPOCH,
        f"[strategy]{strategy_name}[/strategy]",
        total=total_epochs,
        is_app_task=False,
    )


def epoch_progress_update(epoch, total_epochs, avg_loss, eval_result=None,
                          strategy_name=None, progress_metric=None):
    """Update epoch progress with loss and optional eval info.

    Args:
        progress_metric: Optional metric key to highlight in the progress bar
            (e.g., 'validation_accuracy'). When set and present in eval_result,
            displayed alongside loss.
    """
    console = OLConsole()
    if console.is_live:
        if strategy_name:
            desc = f"[strategy]{strategy_name}[/strategy] [label]epoch {epoch}[/label]"
        else:
            desc = f"[label]epoch {epoch}[/label]"
    else:
        parts = [f"[label]epoch {epoch}[/label]",
                 f"[label]loss=[/label][metric.value]{avg_loss:.4f}[/metric.value]"]
        if progress_metric and eval_result and progress_metric in eval_result.metrics:
            val = eval_result.metrics[progress_metric]
            style = "metric.improved" if val >= 90 else "metric.value"
            parts.append(f"[label]{progress_metric}=[/label][{style}]{val:.1f}%[/{style}]")
        desc = " ".join(parts)
    console.update_progress_task(TASK_EPOCH, description=desc, advance=1)


def epoch_progress_end():
    """Close epoch progress bar."""
    OLConsole().progress_stop()


def batch_progress_start(total_batches):
    """Create batch-level progress bar within an epoch."""
    OLConsole().create_progress_task(TASK_BATCH, "[label]Batches[/label]",
                                     total=total_batches)


def batch_progress_update():
    """Advance batch progress by 1."""
    OLConsole().update_progress_task(TASK_BATCH, advance=1)


def batch_progress_end():
    """Close batch progress bar."""
    OLConsole().remove_progress_task(TASK_BATCH)


def eval_progress_start(label, total_batches):
    """Create an evaluation progress bar."""
    task_name = f"{TASK_EVAL}_{label.lower()}"
    OLConsole().create_progress_task(
        task_name, f"[label]{label}[/label]", total=total_batches,
    )


def eval_progress_update(label):
    """Advance evaluation progress by 1."""
    task_name = f"{TASK_EVAL}_{label.lower()}"
    OLConsole().update_progress_task(task_name, advance=1)


def eval_progress_end(label):
    """Close evaluation progress bar."""
    task_name = f"{TASK_EVAL}_{label.lower()}"
    OLConsole().remove_progress_task(task_name)
