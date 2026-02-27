"""Cross-strategy comparison tool.

Bar charts comparing aggregate metric values (final, min, max, mean)
across strategies, with a Rich summary table.

Usage:
    python analyze_experiment.py mod_arithmetic compare \
        --metrics training_metrics/loss training_metrics/validation_accuracy \
        --stat final
"""

from __future__ import annotations

import numpy as np

from rich.table import Table
from rich import box

from console import OLConsole
from ..base import AnalysisTool, AnalysisContext, ToolRegistry
from ..style import get_strategy_colors
from ..visualize import OLFigure, plot_bar


_STAT_FUNCS = {
    'final': lambda s: s.iloc[-1],
    'min': lambda s: s.min(),
    'max': lambda s: s.max(),
    'mean': lambda s: s.mean(),
}


def _compute_stat(series, stat: str):
    """Compute aggregate statistic, returning NaN for empty series."""
    series = series.dropna()
    if series.empty:
        return np.nan
    return _STAT_FUNCS[stat](series)


@ToolRegistry.register
class CompareTool(AnalysisTool):
    """Compare aggregate metric values across strategies."""

    name = "compare"
    description = "Bar charts comparing final/min/max/mean metrics across strategies"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            '--metrics', nargs='+', required=True,
            help='Metric columns to compare (hook_name/metric_name format)',
        )
        parser.add_argument(
            '--stat', choices=['final', 'min', 'max', 'mean'], default='final',
            help='Aggregate statistic to compare (default: final)',
        )

    def describe_outputs(self) -> list[str]:
        return ['compare.png — bar chart of aggregate metrics per strategy']

    def run(self, context: AnalysisContext) -> None:
        console = OLConsole()
        args = context.args
        stat = args.stat
        dpi = getattr(args, 'dpi', 300)

        # Validate metrics
        available_cols = set(context.data.columns) - {'step', 'strategy'}
        valid_metrics = []
        for m in args.metrics:
            if m in available_cols:
                valid_metrics.append(m)
            else:
                console.print(f"[warning.content]Warning: metric '{m}' not found in data[/warning.content]")

        if not valid_metrics:
            console.print("[error.content]No valid metrics found. Available metrics:[/error.content]")
            for col in sorted(available_cols):
                console.print(f"  [metric.value]{col}[/metric.value]")
            return

        strategies = context.strategies
        strat_colors = get_strategy_colors(strategies)

        # Compute aggregate values: {metric: {strategy: value}}
        values = {}
        for metric in valid_metrics:
            values[metric] = {}
            for strat in strategies:
                series = context.data.loc[
                    context.data['strategy'] == strat, metric
                ]
                values[metric][strat] = _compute_stat(series, stat)

        # Bar chart: one subplot per metric
        fig = OLFigure(
            n_plots=len(valid_metrics),
            title=f'{context.experiment_name} — {stat}' if context.args.experiment_title else stat.capitalize(),
            share_x=False, share_y=False,
        )

        for i, metric in enumerate(valid_metrics):
            ax = fig.axes[i]
            labels = strategies
            heights = [values[metric][s] for s in strategies]
            colors = [strat_colors[s] for s in strategies]
            plot_bar(ax, labels, heights, colors=colors)
            ax.set_title(context.resolver.label(metric))
            ax.set_ylabel(stat)

        path = fig.save(context.output_path('compare', valid_metrics), dpi=dpi)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")

        # Summary table
        self._print_summary(context, valid_metrics, strategies, values, stat)

    def _print_summary(self, context, metrics, strategies, values, stat):
        """Print Rich comparison table."""
        console = OLConsole()
        table = Table(
            title=f"Strategy Comparison ({stat})",
            box=box.SIMPLE,
            show_header=True,
            header_style="table.header",
        )
        table.add_column("Strategy", style="strategy")
        for metric in metrics:
            table.add_column(context.resolver.label(metric), justify="right", style="metric.value")

        # Find best per metric for highlighting
        best = {}
        for metric in metrics:
            vals = {s: values[metric][s] for s in strategies
                    if not np.isnan(values[metric][s])}
            if vals:
                # Heuristic: metrics containing 'loss' or 'error' → lower is better
                metric_lower = metric.lower()
                if any(k in metric_lower for k in ('loss', 'error', 'perplexity')):
                    best[metric] = min(vals, key=vals.get)
                else:
                    best[metric] = max(vals, key=vals.get)

        for strat in strategies:
            row = [strat]
            for metric in metrics:
                v = values[metric][strat]
                formatted = f"{v:.6g}" if not np.isnan(v) else '—'
                if best.get(metric) == strat:
                    formatted = f"[metric.improved]{formatted}[/metric.improved]"
                row.append(formatted)
            table.add_row(*row)

        console.print(table)
