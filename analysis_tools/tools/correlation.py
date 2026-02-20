"""Correlation scatter plot tool.

Scatter plot of one metric against one or more others, colored by strategy,
with optional linear trend line.

Supports multiple y-metrics with framework-level --layout and --group-by.

Usage:
    python analyze_experiment.py mod_arithmetic correlation \
        --x training_metrics/loss --y training_metrics/val_acc \
        --trend-line

    python analyze_experiment.py mod_arithmetic correlation \
        --x training_metrics/loss \
        --y training_metrics/val_acc training_diagnostics/loss_volatility \
        --layout grid --group-by strategy
"""

from __future__ import annotations

import numpy as np

from console import OLConsole
from ..base import AnalysisTool, AnalysisContext, ToolRegistry
from ..style import get_strategy_colors, get_metric_colors
from ..visualize import OLFigure


@ToolRegistry.register
class CorrelationTool(AnalysisTool):
    """Scatter plot showing correlation between metrics."""

    name = "correlation"
    description = "Scatter plot of metric A vs metric B (or multiple), colored by strategy"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            '--x', required=True, dest='x_metric',
            help='X-axis metric (hook_name/metric_name format)',
        )
        parser.add_argument(
            '--y', nargs='+', required=True, dest='y_metrics',
            help='Y-axis metric(s) (hook_name/metric_name format)',
        )
        parser.add_argument(
            '--trend-line', action='store_true', default=False,
            dest='trend_line',
            help='Add linear regression trend line per strategy',
        )
        parser.add_argument(
            '--sample', type=int, default=None,
            help='Plot every Nth data point (default: all points)',
        )

    def describe_outputs(self) -> list[str]:
        return ['correlation.png — scatter plot(s) of x-metric vs y-metric(s)']

    def run(self, context: AnalysisContext) -> None:
        console = OLConsole()
        args = context.args
        x_metric = args.x_metric
        y_metrics = args.y_metrics
        dpi = getattr(args, 'dpi', 300)

        # Validate x metric
        available_cols = set(context.data.columns) - {'step', 'strategy'}
        if x_metric not in available_cols:
            console.print(
                f"[error.content]X metric '{x_metric}' not found in data[/error.content]"
            )
            console.print("[label]Available metrics:[/label]")
            for col in sorted(available_cols):
                console.print(f"  [metric.value]{col}[/metric.value]")
            return

        # Validate y metrics
        valid_y = []
        for m in y_metrics:
            if m in available_cols:
                valid_y.append(m)
            else:
                console.print(
                    f"[warning.content]Warning: metric '{m}' not found in data[/warning.content]"
                )

        if not valid_y:
            console.print(
                "[error.content]No valid Y metrics found. Available metrics:[/error.content]"
            )
            for col in sorted(available_cols):
                console.print(f"  [metric.value]{col}[/metric.value]")
            return

        strategies = context.strategies
        strat_colors = get_strategy_colors(strategies)
        x_label = context.resolver.label(x_metric)

        layout = getattr(args, 'layout', 'overlay')
        group_by = getattr(args, 'group_by', 'strategy')
        use_grid_strategy = (layout == 'grid' and group_by == 'strategy')

        if use_grid_strategy:
            self._plot_grid_by_strategy(
                context, x_metric, valid_y, strategies, strat_colors,
                x_label, dpi,
            )
        else:
            self._plot_by_metric(
                context, x_metric, valid_y, strategies, strat_colors,
                x_label, dpi,
            )

        # Print correlations
        self._print_correlations(context, x_metric, valid_y, strategies)

    def _plot_by_metric(self, context, x_metric, y_metrics, strategies,
                        strat_colors, x_label, dpi):
        """One subplot per y-metric, all strategies overlaid."""
        console = OLConsole()
        args = context.args
        exp_title = context.experiment_name if args.experiment_title else None

        fig = OLFigure(
            n_plots=len(y_metrics),
            title=exp_title,
            share_x=False, share_y=False,
        )

        for i, y_metric in enumerate(y_metrics):
            ax = fig.axes[i]
            y_label = context.resolver.label(y_metric)

            for strat in strategies:
                strat_df = context.data[context.data['strategy'] == strat]
                pair = strat_df[[x_metric, y_metric]].dropna()
                if pair.empty:
                    continue

                if args.sample and args.sample > 1:
                    pair = pair.iloc[::args.sample]

                x_vals = pair[x_metric].values
                y_vals = pair[y_metric].values
                color = strat_colors[strat]

                ax.scatter(
                    x_vals, y_vals, c=color, label=strat,
                    alpha=0.4, s=12, edgecolors='none',
                )

                if args.trend_line and len(x_vals) >= 2:
                    self._add_trend(ax, x_vals, y_vals, color)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f'{x_label} vs {y_label}')
            ax.legend()

        path = fig.save(context.output_path('scatter', [x_metric] + y_metrics), dpi=dpi)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")

    def _plot_grid_by_strategy(self, context, x_metric, y_metrics, strategies,
                                strat_colors, x_label, dpi):
        """One subplot per strategy, all y-metrics overlaid with metric colors."""
        console = OLConsole()
        args = context.args
        exp_title = context.experiment_name if args.experiment_title else None

        y_labels = [context.resolver.label(m) for m in y_metrics]
        met_colors = get_metric_colors(y_labels)

        fig = OLFigure(
            n_plots=len(strategies),
            title=exp_title,
            share_x=False, share_y=False,
        )

        for i, strat in enumerate(strategies):
            ax = fig.axes[i]
            strat_df = context.data[context.data['strategy'] == strat]

            for y_metric, y_label in zip(y_metrics, y_labels):
                pair = strat_df[[x_metric, y_metric]].dropna()
                if pair.empty:
                    continue

                if args.sample and args.sample > 1:
                    pair = pair.iloc[::args.sample]

                x_vals = pair[x_metric].values
                y_vals = pair[y_metric].values
                color = met_colors[y_label]

                ax.scatter(
                    x_vals, y_vals, c=color, label=y_label,
                    alpha=0.4, s=12, edgecolors='none',
                )

                if args.trend_line and len(x_vals) >= 2:
                    self._add_trend(ax, x_vals, y_vals, color)

            ax.set_xlabel(x_label)
            ax.set_title(strat)
            ax.legend()

        path = fig.save(context.output_path('scatter', [x_metric] + y_metrics), dpi=dpi)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")

    @staticmethod
    def _add_trend(ax, x_vals, y_vals, color):
        """Add a linear regression trend line to an axes."""
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        if mask.sum() < 2:
            return
        coeffs = np.polyfit(x_vals[mask], y_vals[mask], 1)
        x_range = np.array([x_vals[mask].min(), x_vals[mask].max()])
        ax.plot(
            x_range, np.polyval(coeffs, x_range),
            color=color, linewidth=1.5, linestyle='--',
        )

    def _print_correlations(self, context, x_metric, y_metrics, strategies):
        """Print Pearson correlation coefficients per strategy per y-metric."""
        from rich.table import Table
        from rich import box

        console = OLConsole()
        x_label = context.resolver.label(x_metric)

        table = Table(
            title="Pearson Correlation",
            box=box.SIMPLE,
            show_header=True,
            header_style="table.header",
        )
        table.add_column("Strategy", style="strategy")

        y_labels = [context.resolver.label(m) for m in y_metrics]
        for y_label in y_labels:
            table.add_column(
                f'{x_label} vs {y_label}\nr',
                justify="right", style="metric.value",
            )
            table.add_column('n', justify="right", style="detail")

        for strat in strategies:
            strat_df = context.data[context.data['strategy'] == strat]
            row = [strat]
            for y_metric in y_metrics:
                pair = strat_df[[x_metric, y_metric]].dropna()
                if len(pair) < 2:
                    row.extend(['—', str(len(pair))])
                    continue
                x_vals = pair[x_metric].values
                y_vals = pair[y_metric].values
                mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                if mask.sum() < 2:
                    row.extend(['—', str(int(mask.sum()))])
                    continue
                r = np.corrcoef(x_vals[mask], y_vals[mask])[0, 1]
                row.extend([f'{r:.4f}', str(int(mask.sum()))])
            table.add_row(*row)

        console.print(table)
