"""Convergence analysis tool.

Time-to-threshold analysis and normalized learning curves. Answers
"when does each strategy reach a given metric value?" and provides
shape-normalized curves for comparing convergence dynamics.

Supports multiple metrics with framework-level --layout and --group-by.

Usage:
    python analyze_experiment.py mod_arithmetic convergence \
        --metrics training_metrics/validation_accuracy --threshold 95.0

    python analyze_experiment.py mod_arithmetic convergence \
        --metrics training_metrics/loss --threshold 3.0 --direction below

    python analyze_experiment.py mod_arithmetic convergence \
        --metrics training_metrics/validation_accuracy training_metrics/loss \
        --layout grid --group-by strategy

    python analyze_experiment.py mod_arithmetic convergence \
        --metrics training_metrics/validation_accuracy --normalize
"""

from __future__ import annotations

import numpy as np

from rich.table import Table
from rich import box

from console import OLConsole
from ..base import AnalysisTool, AnalysisContext, ToolRegistry
from ..style import get_strategy_colors, get_metric_colors
from ..visualize import OLFigure, plot_time_series, ema_smooth


def _find_threshold_step(steps, values, threshold: float,
                         direction: str) -> int | None:
    """Find the first step where values cross the threshold.

    Args:
        steps: Array of step values.
        values: Array of metric values.
        threshold: Target value.
        direction: 'above' (value >= threshold) or 'below' (value <= threshold).

    Returns:
        Step number, or None if threshold is never reached.
    """
    if direction == 'above':
        mask = values >= threshold
    else:
        mask = values <= threshold

    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    return int(steps[indices[0]])


def _normalize_series(values):
    """Normalize values to [0, 1] range based on min/max."""
    values = np.asarray(values, dtype=float)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if vmax == vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def _auto_direction(metric: str) -> str:
    """Auto-detect threshold direction from metric name."""
    metric_lower = metric.lower()
    if any(k in metric_lower for k in ('loss', 'error', 'perplexity')):
        return 'below'
    return 'above'


@ToolRegistry.register
class ConvergenceTool(AnalysisTool):
    """Analyze convergence speed and normalized learning curves."""

    name = "convergence"
    description = "Time-to-threshold analysis and normalized learning curves"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            '--metrics', nargs='+', required=True,
            help='Metrics to analyze (hook_name/metric_name format)',
        )
        parser.add_argument(
            '--threshold', type=float, default=None,
            help='Target value for time-to-threshold analysis',
        )
        parser.add_argument(
            '--direction', choices=['above', 'below'], default=None,
            help='Threshold direction: above (value >= t) or below (value <= t). '
                 'Default: auto-detect from metric name',
        )
        parser.add_argument(
            '--normalize', action='store_true', default=False,
            help='Plot normalized [0,1] learning curves for shape comparison',
        )

    def describe_outputs(self) -> list[str]:
        return [
            'convergence.png — metric curves (overlay or grid)',
            'threshold.png — bar chart of steps to reach threshold',
            'normalized.png — normalized [0,1] learning curves',
        ]

    def run(self, context: AnalysisContext) -> None:
        console = OLConsole()
        args = context.args
        threshold = args.threshold
        normalize = args.normalize
        smooth = getattr(args, 'smooth', None)
        dpi = getattr(args, 'dpi', 300)

        # Validate metrics
        available_cols = set(context.data.columns) - {'step', 'strategy'}
        valid_metrics = []
        for m in args.metrics:
            if m in available_cols:
                valid_metrics.append(m)
            else:
                console.print(
                    f"[warning.content]Warning: metric '{m}' not found in data[/warning.content]"
                )

        if not valid_metrics:
            console.print("[error.content]No valid metrics found. Available metrics:[/error.content]")
            for col in sorted(available_cols):
                console.print(f"  [metric.value]{col}[/metric.value]")
            return

        strategies = context.strategies
        strat_colors = get_strategy_colors(strategies)
        layout = getattr(args, 'layout', 'overlay')
        group_by = getattr(args, 'group_by', 'strategy')
        use_grid_strategy = (layout == 'grid' and group_by == 'strategy')

        # Plot curves
        self._plot_curves(
            context, valid_metrics, strategies, strat_colors,
            threshold, smooth, dpi, use_grid_strategy,
        )

        # Threshold analysis (per metric)
        if threshold is not None:
            self._threshold_analysis(
                context, valid_metrics, strategies, strat_colors,
                threshold, args.direction, dpi,
            )

        # Normalized curves
        if normalize:
            self._plot_normalized(
                context, valid_metrics, strategies, strat_colors,
                smooth, dpi, use_grid_strategy,
            )

    def _plot_curves(self, context, metrics, strategies, strat_colors,
                     threshold, smooth, dpi, use_grid_strategy):
        """Plot metric curves with optional threshold line."""
        console = OLConsole()
        exp_title = context.experiment_name if context.args.experiment_title else None

        if use_grid_strategy:
            metric_labels = [context.resolver.label(m) for m in metrics]
            met_colors = get_metric_colors(metric_labels)
            fig = OLFigure(n_plots=len(strategies), title=exp_title,
                           share_x=False)

            for i, strat in enumerate(strategies):
                ax = fig.axes[i]
                strat_df = context.data[context.data['strategy'] == strat]
                for metric, label in zip(metrics, metric_labels):
                    subset = strat_df[['step', metric]].dropna(subset=[metric])
                    if subset.empty:
                        continue
                    plot_time_series(
                        ax, subset, x='step', y=[metric],
                        labels=[label], colors=[met_colors[label]],
                        smooth=smooth,
                    )
                if threshold is not None:
                    ax.axhline(y=threshold, color='#8A8F98', linestyle=':',
                               linewidth=1)
                ax.set_title(strat)
                ax.set_xlabel(context.x_label)
                ax.legend()
        else:
            fig = OLFigure(n_plots=len(metrics), title=exp_title)

            for i, metric in enumerate(metrics):
                ax = fig.axes[i]
                for strat in strategies:
                    strat_df = context.data[context.data['strategy'] == strat]
                    subset = strat_df[['step', metric]].dropna(subset=[metric])
                    if subset.empty:
                        continue
                    plot_time_series(
                        ax, subset, x='step', y=[metric],
                        labels=[strat], colors=[strat_colors[strat]],
                        smooth=smooth,
                    )
                if threshold is not None:
                    ax.axhline(y=threshold, color='#8A8F98', linestyle=':',
                               linewidth=1, label=f'threshold = {threshold}')
                ax.set_title(context.resolver.label(metric))
                ax.set_xlabel(context.x_label)
                ax.legend()

        path = fig.save(context.output_path('convergence', metrics), dpi=dpi)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")

    def _threshold_analysis(self, context, metrics, strategies, strat_colors,
                            threshold, direction_override, dpi):
        """Bar chart + table of steps-to-threshold for each metric."""
        console = OLConsole()

        for metric in metrics:
            direction = direction_override or _auto_direction(metric)
            metric_label = context.resolver.label(metric)

            # Compute steps-to-threshold per strategy
            results = {}
            for strat in strategies:
                strat_df = context.data[context.data['strategy'] == strat]
                subset = strat_df[['step', metric]].dropna(subset=[metric])
                if subset.empty:
                    results[strat] = None
                    continue
                results[strat] = _find_threshold_step(
                    subset['step'].values, subset[metric].values,
                    threshold, direction,
                )

            # Bar chart
            plot_strats = [s for s in strategies if results[s] is not None]
            if plot_strats:
                from ..visualize import plot_bar

                fig = OLFigure(
                    n_plots=1,
                    title=context.experiment_name if context.args.experiment_title else None,
                    share_x=False,
                )
                ax = fig.axes[0]
                labels = plot_strats
                heights = [results[s] for s in plot_strats]
                colors = [strat_colors[s] for s in plot_strats]
                plot_bar(ax, labels, heights, colors=colors)
                ax.set_title(f'{metric_label} {direction} {threshold}')
                ax.set_ylabel('Steps')

                path = fig.save(
                    context.output_path('threshold', [metric]), dpi=dpi,
                )
                console.print(f"[label]Saved:[/label] [path]{path}[/path]")

            # Rich table
            table = Table(
                title=f"Steps to {metric_label} {direction} {threshold}",
                box=box.SIMPLE,
                show_header=True,
                header_style="table.header",
            )
            table.add_column("Strategy", style="strategy")
            table.add_column("Steps", justify="right", style="metric.value")
            table.add_column("Status", style="detail")

            reached = {s: v for s, v in results.items() if v is not None}
            best_strat = min(reached, key=reached.get) if reached else None

            for strat in strategies:
                step = results[strat]
                if step is None:
                    table.add_row(strat, '—', 'not reached')
                else:
                    formatted = f'{step:,}'
                    if strat == best_strat:
                        formatted = f'[metric.improved]{formatted}[/metric.improved]'
                    table.add_row(strat, formatted, 'reached')

            console.print(table)

    def _plot_normalized(self, context, metrics, strategies, strat_colors,
                         smooth, dpi, use_grid_strategy):
        """Plot [0,1] normalized learning curves."""
        console = OLConsole()
        exp_title = context.experiment_name if context.args.experiment_title else None

        if use_grid_strategy:
            metric_labels = [context.resolver.label(m) for m in metrics]
            met_colors = get_metric_colors(metric_labels)
            fig = OLFigure(n_plots=len(strategies), title=exp_title,
                           share_x=False)

            for i, strat in enumerate(strategies):
                ax = fig.axes[i]
                strat_df = context.data[context.data['strategy'] == strat]
                for metric, label in zip(metrics, metric_labels):
                    subset = strat_df[['step', metric]].dropna(subset=[metric])
                    if subset.empty:
                        continue
                    steps = subset['step'].values
                    norm = _normalize_series(subset[metric].values)
                    color = met_colors[label]
                    if smooth and smooth > 0:
                        ax.plot(steps, norm, color=color, alpha=0.15,
                                linewidth=0.8)
                        ax.plot(steps, ema_smooth(norm, smooth), color=color,
                                label=label, linewidth=1.5)
                    else:
                        ax.plot(steps, norm, color=color, label=label,
                                linewidth=1.5)
                ax.set_title(strat)
                ax.set_xlabel(context.x_label)
                ax.set_ylabel('normalized')
                ax.set_ylim(-0.05, 1.05)
                ax.legend()
        else:
            fig = OLFigure(n_plots=len(metrics), title=exp_title)

            for i, metric in enumerate(metrics):
                ax = fig.axes[i]
                metric_label = context.resolver.label(metric)
                for strat in strategies:
                    strat_df = context.data[context.data['strategy'] == strat]
                    subset = strat_df[['step', metric]].dropna(subset=[metric])
                    if subset.empty:
                        continue
                    steps = subset['step'].values
                    norm = _normalize_series(subset[metric].values)
                    color = strat_colors[strat]
                    if smooth and smooth > 0:
                        ax.plot(steps, norm, color=color, alpha=0.15,
                                linewidth=0.8)
                        ax.plot(steps, ema_smooth(norm, smooth), color=color,
                                label=strat, linewidth=1.5)
                    else:
                        ax.plot(steps, norm, color=color, label=strat,
                                linewidth=1.5)
                ax.set_title(f'{metric_label} (normalized)')
                ax.set_xlabel(context.x_label)
                ax.set_ylabel('normalized')
                ax.set_ylim(-0.05, 1.05)
                ax.legend()

        path = fig.save(context.output_path('normalized', metrics), dpi=dpi)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")
