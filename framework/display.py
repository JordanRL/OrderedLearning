"""Framework display utilities for experiment output.

Standardized display functions called by the framework loops and
ExperimentRunner methods. All functions obtain the OLConsole singleton
internally and use semantic theme styles from console/themes.py.
"""

import numpy as np
from rich.align import Align
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

from console import OLConsole


# ── Experiment lifecycle display ──────────────────────────────────────

def display_experiment_banner(title, description=None,
                              targets=None, extra_lines=None):
    """Primary panel with experiment overview. Called once at start."""
    console = OLConsole()
    lines = []
    if description:
        lines.append(description)
    if targets:
        lines.append("")
        for t in targets:
            lines.append(f"  [target]{t}[/target]")
    if extra_lines:
        lines.append("")
        for line in extra_lines:
            lines.append(f"  [detail]{line}[/detail]")

    content = "\n".join(lines) if lines else ""
    console.print(Panel(
        content,
        title=f"[bold]{title}[/bold]",
        border_style="panel.primary",
    ))


def display_condition_header(strategy_name, settings=None):
    """Yellow rule + info table when starting a new strategy/condition."""
    console = OLConsole()
    console.print()
    console.print(Rule(f"[bold][strategy]{strategy_name}[/strategy][/bold]",
                       style="panel.attention"))
    if settings:
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column("Setting", style="label")
        table.add_column("Value", style="metric.value")
        for key, value in settings.items():
            table.add_row(key, str(value))
        console.print(Align.center(table))


def display_training_start(total, batch_size, extra_rows=None):
    """Green rule + settings table before training begins."""
    console = OLConsole()
    console.print(Rule("[bold][panel.success]Training[/panel.success][/bold]", style="panel.success"))
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Setting", style="label")
    table.add_column("Value", style="metric.value")
    table.add_row("Total", f"{total:,}")
    table.add_row("Batch size", str(batch_size))
    if extra_rows:
        for key, value in extra_rows.items():
            table.add_row(key, str(value))
    console.print(table)


# ── Progress display ─────────────────────────────────────────────────

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
                          strategy_name=None):
    """Update epoch progress with loss and optional eval info."""
    console = OLConsole()
    if console.is_live:
        if strategy_name:
            desc = f"[strategy]{strategy_name}[/strategy] [label]epoch {epoch}[/label]"
        else:
            desc = f"[label]epoch {epoch}[/label]"
    else:
        parts = [f"[label]epoch {epoch}[/label]",
                 f"[label]loss=[/label][metric.value]{avg_loss:.4f}[/metric.value]"]
        if eval_result and 'val_acc' in eval_result.metrics:
            acc = eval_result.metrics['val_acc']
            style = "metric.improved" if acc >= 90 else "metric.value"
            parts.append(f"[label]val_acc=[/label][{style}]{acc:.1f}%[/{style}]")
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


# ── Evaluation display ───────────────────────────────────────────────

def display_eval_update(step_or_epoch, eval_result,
                        init_eval=None, context_label=None,
                        counter_label="Step"):
    """Standard periodic evaluation table.

    Formats known metric keys with appropriate styles. Shows change
    from init_eval when provided.
    """
    console = OLConsole()
    metrics = eval_result.metrics
    title_parts = [f"[bold]{counter_label} {step_or_epoch:,}[/bold]"]
    if context_label:
        title_parts.append(f"[detail]| {context_label}[/detail]")

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="table.header",
        title=" ".join(title_parts),
        title_style="",
    )

    # Build columns and row based on what metrics are present
    row = []

    if 'loss' in metrics:
        table.add_column("Loss", justify="right")
        row.append(f"[metric.value]{metrics['loss']:.4f}[/metric.value]")

    if 'train_acc' in metrics:
        table.add_column("Train Acc", justify="right")
        row.append(format_accuracy(metrics['train_acc']))

    if 'val_acc' in metrics:
        table.add_column("Val Acc", justify="right")
        row.append(format_accuracy(metrics['val_acc']))

    if 'seq_prob' in metrics:
        table.add_column("Seq Prob", justify="right")
        val = metrics['seq_prob']
        if init_eval and 'seq_prob' in init_eval.metrics and init_eval.metrics['seq_prob'] > 0:
            change = val / init_eval.metrics['seq_prob']
            row.append(f"{val:.2e} [detail]({change:.1f}x)[/detail]")
        else:
            row.append(f"{val:.2e}")

    if 'avg_target_prob' in metrics:
        table.add_column("Target Prob", justify="right", style="value.count")
        val = metrics['avg_target_prob']
        if init_eval and 'avg_target_prob' in init_eval.metrics and init_eval.metrics['avg_target_prob'] > 0:
            change = val / init_eval.metrics['avg_target_prob']
            row.append(f"[target]{val:.2e}[/target] [detail]({change:.1f}x)[/detail]")
        else:
            row.append(f"[target]{val:.2e}[/target]")

    if 'avg_sim' in metrics:
        table.add_column("Similarity", justify="right")
        row.append(f"[metric.value]{metrics['avg_sim']:+.4f}[/metric.value]")

    if 'perplexity' in metrics:
        table.add_column("Perplexity", justify="right")
        row.append(f"[metric.value]{metrics['perplexity']:.2f}[/metric.value]")
    elif 'loss' in metrics:
        table.add_column("Perplexity", justify="right")
        row.append(f"[metric.value]{np.exp(metrics['loss']):.2f}[/metric.value]")

    # Caption: generated text if available
    if eval_result.display_data and 'gen_text' in eval_result.display_data:
        gen = eval_result.display_data['gen_text']
        table.caption = f"[caption]{gen[:70]}{'...' if len(gen) > 70 else ''}[/caption]"
        table.caption_style = ""

    if row:
        table.add_row(*row)
        console.print(table)


def display_final_results(name, init_eval, final_eval,
                          baseline_prob=None):
    """Green-bordered panel comparing initial vs final metrics."""
    console = OLConsole()
    table = Table(box=box.ROUNDED, show_header=True, header_style="table.header")
    table.add_column("Metric", style="label")
    table.add_column("Initial", justify="right", style="detail")
    table.add_column("Final", justify="right", style="value.count")
    table.add_column("Change", justify="right")

    init_m = init_eval.metrics if init_eval else {}
    final_m = final_eval.metrics if final_eval else {}

    no_init = "[placeholder]—[/placeholder]"

    for key in final_m:
        init_val = init_m.get(key)
        final_val = final_m[key]

        # Determine formatting based on metric type
        if 'acc' in key:
            table.add_row(
                key,
                format_accuracy(init_val) if init_val is not None else no_init,
                format_accuracy(final_val),
                format_change(init_val, final_val, higher_is_better=True) if init_val is not None else no_init,
            )
        elif 'prob' in key:
            table.add_row(
                key,
                f"{init_val:.2e}" if init_val is not None else no_init,
                f"{final_val:.2e}",
                format_change(init_val, final_val, higher_is_better=True) if init_val is not None else no_init,
            )
        elif 'loss' in key:
            table.add_row(
                key,
                f"{init_val:.4f}" if init_val is not None else no_init,
                f"{final_val:.4f}",
                format_change(init_val, final_val, higher_is_better=False) if init_val is not None else no_init,
            )
        else:
            table.add_row(
                key,
                f"{init_val:.4f}" if init_val is not None else no_init,
                f"{final_val:.4f}",
                format_change(init_val, final_val) if init_val is not None else no_init,
            )

    console.print(Panel(
        table,
        title=f"[bold]FINAL RESULTS: {name}[/bold]",
        border_style="panel.success",
    ))


def display_comparison_table(all_results, metric_keys=None):
    """Cross-strategy comparison table. Auto-detects best performer."""
    if not all_results:
        return

    console = OLConsole()

    # Auto-detect metric keys from first result
    if metric_keys is None:
        first = next(iter(all_results.values()))
        final_eval = first.get('final_eval', {})
        metric_keys = list(final_eval.keys()) if isinstance(final_eval, dict) else []

    table = Table(box=box.ROUNDED, show_header=True, header_style="table.header")
    table.add_column("Strategy", style="strategy")
    table.add_column("Steps/Epochs", justify="right", style="detail")
    for key in metric_keys:
        table.add_column(key, justify="right")

    # Find best values for highlighting
    best = {}
    for key in metric_keys:
        values = []
        for name, summary in all_results.items():
            final = summary.get('final_eval', {})
            if isinstance(final, dict) and key in final:
                values.append((name, final[key]))
        if values:
            higher_better = 'loss' not in key
            best[key] = max(values, key=lambda x: x[1] if higher_better else -x[1])[0]

    for name, summary in all_results.items():
        training = summary.get('training', {})
        total = (training.get('actual_steps') or training.get('actual_epochs')
                 or summary.get('total_steps', '?'))
        final = summary.get('final_eval', {})
        row = [name, f"{total:,}" if isinstance(total, int) else str(total)]
        for key in metric_keys:
            if isinstance(final, dict) and key in final:
                val = final[key]
                is_best = best.get(key) == name
                if 'acc' in key:
                    formatted = format_accuracy(val)
                elif 'prob' in key:
                    formatted = f"{val:.2e}"
                elif 'loss' in key:
                    formatted = f"{val:.4f}"
                else:
                    formatted = f"{val:.4f}"
                if is_best:
                    formatted = f"[metric.improved]{formatted}[/metric.improved]"
                row.append(formatted)
            else:
                row.append("-")
        table.add_row(*row)

    console.print()
    console.print(Panel(table, title="[bold]STRATEGY COMPARISON[/bold]",
                        border_style="panel.info"))


def display_post_live_summary(all_results):
    """Print experiment summary after live mode exits.

    Re-renders the final results and comparison table so they're
    visible on the normal console after the live display is torn down.
    """
    if not all_results:
        return

    console = OLConsole()
    console.print()
    console.print(Rule("[bold]Experiment Complete[/bold]", style="panel.success"))
    console.print()

    # Per-strategy final results
    for name, summary in all_results.items():
        init_m = summary.get('init_eval') or {}
        final_m = summary.get('final_eval') or {}
        if not final_m:
            continue

        table = Table(box=box.ROUNDED, show_header=True, header_style="table.header")
        table.add_column("Metric", style="label")
        table.add_column("Initial", justify="right", style="detail")
        table.add_column("Final", justify="right", style="value.count")
        table.add_column("Change", justify="right")

        no_init = "[placeholder]—[/placeholder]"

        for key in final_m:
            init_val = init_m.get(key) if init_m else None
            final_val = final_m[key]

            if 'acc' in key:
                table.add_row(
                    key,
                    format_accuracy(init_val) if init_val is not None else no_init,
                    format_accuracy(final_val),
                    format_change(init_val, final_val, higher_is_better=True) if init_val is not None else no_init,
                )
            elif 'loss' in key:
                table.add_row(
                    key,
                    f"{init_val:.4f}" if init_val is not None else no_init,
                    f"{final_val:.4f}",
                    format_change(init_val, final_val, higher_is_better=False) if init_val is not None else no_init,
                )
            else:
                table.add_row(
                    key,
                    f"{init_val:.4f}" if init_val is not None else no_init,
                    f"{final_val:.4f}",
                    format_change(init_val, final_val) if init_val is not None else no_init,
                )

        training = summary.get('training', {})
        total = (training.get('actual_steps') or training.get('actual_epochs')
                 or summary.get('total_steps', '?'))
        total_str = f"{total:,}" if isinstance(total, int) else str(total)
        console.print(Panel(
            table,
            title=f"[bold]{name}[/bold] [detail]({total_str} steps/epochs)[/detail]",
            border_style="panel.success",
        ))

    # Cross-strategy comparison (when multiple strategies)
    if len(all_results) > 1:
        display_comparison_table(all_results)


def display_phase_transition(old_phase, new_phase, metrics=None):
    """Yellow-bordered panel for curriculum phase advancement."""
    lines = [f"[phase]{old_phase}[/phase] → [phase]{new_phase}[/phase]"]
    if metrics:
        for key, val in metrics.items():
            if isinstance(val, float):
                lines.append(f"  [label]{key}:[/label] {val:.4f}")
    OLConsole().print(Panel(
        "\n".join(lines),
        title="[bold][panel.attention]Phase Transition[/panel.attention][/bold]",
        border_style="panel.attention",
    ))


def display_resume_info(step_or_epoch, checkpoint_path, completed_strategies,
                        counter_label="step"):
    """Display resume information when continuing from a checkpoint."""
    console = OLConsole()
    lines = [
        f"[label]Resuming from checkpoint at {counter_label} "
        f"{step_or_epoch:,}[/label]",
        f"[detail]{checkpoint_path}[/detail]",
    ]
    if completed_strategies:
        names = ', '.join(completed_strategies)
        lines.append(
            f"[label]Skipping completed strategies:[/label] [detail]{names}[/detail]"
        )
    console.print(Panel(
        "\n".join(lines),
        title="[bold][panel.info]Resume[/panel.info][/bold]",
        border_style="panel.info",
    ))


def display_grokking_achieved(epoch):
    """Success message when grokking target is reached."""
    OLConsole().print(Panel(
        f"[success]Target accuracy reached at epoch {epoch:,}[/success]",
        title="[bold][panel.success]Grokking Achieved[/panel.success][/bold]",
        border_style="panel.success",
    ))


# ── Metric formatters ────────────────────────────────────────────────

def format_prob(value, baseline=None):
    """Format a probability value: '3.2e-04 (1.6x baseline)'."""
    s = f"{value:.2e}"
    if baseline and baseline > 0:
        ratio = value / baseline
        s += f" ({ratio:.1f}x baseline)"
    return s


def format_change(init, final, higher_is_better=True):
    """Format a change: '2.1x' in green or '0.3x' in red."""
    if init == 0:
        return "[placeholder]—[/placeholder]"

    if higher_is_better:
        ratio = final / init
    else:
        ratio = init / final  # For loss: lower is better

    if ratio >= 1.0:
        return f"[metric.improved]{ratio:.1f}x[/metric.improved]"
    else:
        return f"[metric.degraded]{ratio:.1f}x[/metric.degraded]"


def format_accuracy(value):
    """Format accuracy with graduated color coding."""
    if value >= 99:
        return f"[accuracy.excellent]{value:.2f}%[/accuracy.excellent]"
    elif value >= 80:
        return f"[accuracy.good]{value:.2f}%[/accuracy.good]"
    elif value >= 50:
        return f"[accuracy.fair]{value:.2f}%[/accuracy.fair]"
    else:
        return f"[accuracy.poor]{value:.2f}%[/accuracy.poor]"


def format_loss(value):
    """Format loss value."""
    return f"[metric.value]{value:.4f}[/metric.value]"
