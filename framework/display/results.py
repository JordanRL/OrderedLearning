"""Evaluation and comparison display functions.

Periodic eval updates, final results panels, cross-strategy
comparison tables, and special event displays (phase transition,
grokking achieved).
"""

import numpy as np
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

from console import OLConsole

from .formatting import format_accuracy, format_change


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

    if 'training_accuracy' in metrics:
        table.add_column("Train Acc", justify="right")
        row.append(format_accuracy(metrics['training_accuracy']))

    if 'validation_accuracy' in metrics:
        table.add_column("Val Acc", justify="right")
        row.append(format_accuracy(metrics['validation_accuracy']))

    if 'sequence_probability' in metrics:
        table.add_column("Seq Prob", justify="right")
        val = metrics['sequence_probability']
        if init_eval and 'sequence_probability' in init_eval.metrics and init_eval.metrics['sequence_probability'] > 0:
            change = val / init_eval.metrics['sequence_probability']
            row.append(f"{val:.2e} [detail]({change:.1f}x)[/detail]")
        else:
            row.append(f"{val:.2e}")

    if 'average_target_probability' in metrics:
        table.add_column("Target Prob", justify="right", style="value.count")
        val = metrics['average_target_probability']
        if init_eval and 'average_target_probability' in init_eval.metrics and init_eval.metrics['average_target_probability'] > 0:
            change = val / init_eval.metrics['average_target_probability']
            row.append(f"[target]{val:.2e}[/target] [detail]({change:.1f}x)[/detail]")
        else:
            row.append(f"[target]{val:.2e}[/target]")

    if 'average_similarity' in metrics:
        table.add_column("Similarity", justify="right")
        row.append(f"[metric.value]{metrics['average_similarity']:+.4f}[/metric.value]")

    if 'perplexity' in metrics:
        table.add_column("Perplexity", justify="right")
        row.append(f"[metric.value]{metrics['perplexity']:.2f}[/metric.value]")
    elif 'loss' in metrics:
        table.add_column("Perplexity", justify="right")
        row.append(f"[metric.value]{np.exp(metrics['loss']):.2f}[/metric.value]")

    # Caption: generated text if available
    if eval_result.display_data and 'generated_text' in eval_result.display_data:
        gen = eval_result.display_data['generated_text']
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

    no_init = "[placeholder]\u2014[/placeholder]"

    for key in final_m:
        init_val = init_m.get(key)
        final_val = final_m[key]

        # Determine formatting based on metric type
        if 'accuracy' in key:
            table.add_row(
                key,
                format_accuracy(init_val) if init_val is not None else no_init,
                format_accuracy(final_val),
                format_change(init_val, final_val, higher_is_better=True) if init_val is not None else no_init,
            )
        elif 'probability' in key:
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
                if 'accuracy' in key:
                    formatted = format_accuracy(val)
                elif 'probability' in key:
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

        no_init = "[placeholder]\u2014[/placeholder]"

        for key in final_m:
            init_val = init_m.get(key) if init_m else None
            final_val = final_m[key]

            if 'accuracy' in key:
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
    lines = [f"[phase]{old_phase}[/phase] \u2192 [phase]{new_phase}[/phase]"]
    if metrics:
        for key, val in metrics.items():
            if isinstance(val, float):
                lines.append(f"  [label]{key}:[/label] {val:.4f}")
    OLConsole().print(Panel(
        "\n".join(lines),
        title="[bold][panel.attention]Phase Transition[/panel.attention][/bold]",
        border_style="panel.attention",
    ))


def display_grokking_achieved(epoch):
    """Success message when grokking target is reached."""
    OLConsole().print(Panel(
        f"[success]Target accuracy reached at epoch {epoch:,}[/success]",
        title="[bold][panel.success]Grokking Achieved[/panel.success][/bold]",
        border_style="panel.success",
    ))
