"""Metric plot analysis tool.

Plots time series of selected metrics across strategies with configurable
layout (overlay or grid) and grouping.

Usage:
    python analyze_experiment.py mod_arithmetic metric_plot \
        --metrics training_metrics/loss training_metrics/validation_accuracy \
        --layout overlay --smooth 0.9
"""

from __future__ import annotations

import numpy as np

from rich.table import Table
from rich import box

from console import OLConsole
from ..base import AnalysisTool, AnalysisContext, ToolRegistry
from ..style import get_strategy_colors, get_metric_colors
from ..visualize import OLFigure, plot_time_series


@ToolRegistry.register
class MetricPlotTool(AnalysisTool):
    """Plot time series of experiment metrics."""

    name = "metric_plot"
    description = "Plot time series of selected metrics across strategies"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            '--metrics', nargs='+', required=True,
            help='Metric columns to plot (hook_name/metric_name format)',
        )
        parser.add_argument(
            '--log-scale', action='store_true', default=False,
            dest='log_scale',
            help='Use log scale on y-axis',
        )
        parser.add_argument(
            '--share-y', action='store_true', default=False,
            dest='share_y',
            help='Use consistent y-axis scale across all subplots',
        )

    def describe_outputs(self) -> list[str]:
        return [
            'overlay.png — all strategies on one plot per metric',
            'grid_strategy.png — one subplot per strategy',
            'grid_metric.png — one subplot per metric',
        ]

    def run(self, context: AnalysisContext) -> None:
        console = OLConsole()
        args = context.args
        metrics = args.metrics
        smooth = getattr(args, 'smooth', None)
        dpi = getattr(args, 'dpi', 300)
        log_scale = getattr(args, 'log_scale', False)
        share_y = getattr(args, 'share_y', False)

        # Validate requested metrics exist
        available_cols = set(context.data.columns) - {'step', 'strategy'}
        valid_metrics = []
        for m in metrics:
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
        layout = getattr(args, 'layout', 'overlay')
        group_by = getattr(args, 'group_by', 'strategy')

        if layout == 'overlay':
            self._plot_overlay(context, valid_metrics, strategies, smooth,
                               log_scale, share_y, dpi)
        elif group_by == 'strategy':
            self._plot_grid_by_strategy(context, valid_metrics, strategies,
                                        smooth, log_scale, share_y, dpi)
        else:
            self._plot_grid_by_metric(context, valid_metrics, strategies,
                                      smooth, log_scale, share_y, dpi)

        # Print summary table
        self._print_summary(context, valid_metrics, strategies)

    def _plot_overlay(self, context, metrics, strategies, smooth, log_scale,
                      share_y, dpi):
        """One subplot per metric, all strategies overlaid with strategy colors."""
        console = OLConsole()
        strat_colors = get_strategy_colors(strategies)
        fig = OLFigure(n_plots=len(metrics), title=context.experiment_name if context.args.experiment_title else None,
                       share_y=share_y)

        for i, metric in enumerate(metrics):
            ax = fig.axes[i]
            for strat in strategies:
                strat_df = context.data[context.data['strategy'] == strat]
                if metric not in strat_df.columns:
                    continue
                subset = strat_df[['step', metric]].dropna(subset=[metric])
                if subset.empty:
                    continue
                plot_time_series(
                    ax, subset, x='step', y=[metric],
                    labels=[strat],
                    colors=[strat_colors[strat]],
                    smooth=smooth, log_scale=log_scale,
                )
            ax.set_title(context.resolver.label(metric))
            ax.set_xlabel(context.x_label)
            ax.legend()

        path = fig.save(context.output_path('overlay', metrics), dpi=dpi)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")

    def _plot_grid_by_strategy(self, context, metrics, strategies, smooth,
                               log_scale, share_y, dpi):
        """One subplot per strategy, multiple metrics with metric colors."""
        console = OLConsole()
        met_colors = get_metric_colors(metrics)
        fig = OLFigure(n_plots=len(strategies), title=context.experiment_name if context.args.experiment_title else None,
                       share_x=False, share_y=share_y)

        for i, strat in enumerate(strategies):
            ax = fig.axes[i]
            strat_df = context.data[context.data['strategy'] == strat]
            for metric in metrics:
                if metric not in strat_df.columns:
                    continue
                subset = strat_df[['step', metric]].dropna(subset=[metric])
                if subset.empty:
                    continue
                plot_time_series(
                    ax, subset, x='step', y=[metric],
                    labels=[context.resolver.label(metric)],
                    colors=[met_colors[metric]],
                    smooth=smooth, log_scale=log_scale,
                )
            ax.set_title(strat)
            ax.set_xlabel(context.x_label)
            ax.legend()

        path = fig.save(context.output_path('grid_strategy', metrics), dpi=dpi)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")

    def _plot_grid_by_metric(self, context, metrics, strategies, smooth,
                             log_scale, share_y, dpi):
        """One subplot per metric, all strategies with strategy colors."""
        console = OLConsole()
        strat_colors = get_strategy_colors(strategies)
        fig = OLFigure(n_plots=len(metrics), title=context.experiment_name if context.args.experiment_title else None,
                       share_y=share_y)

        for i, metric in enumerate(metrics):
            ax = fig.axes[i]
            for strat in strategies:
                strat_df = context.data[context.data['strategy'] == strat]
                if metric not in strat_df.columns:
                    continue
                subset = strat_df[['step', metric]].dropna(subset=[metric])
                if subset.empty:
                    continue
                plot_time_series(
                    ax, subset, x='step', y=[metric],
                    labels=[strat],
                    colors=[strat_colors[strat]],
                    smooth=smooth, log_scale=log_scale,
                )
            ax.set_title(context.resolver.label(metric))
            ax.set_xlabel(context.x_label)
            ax.legend()

        path = fig.save(context.output_path('grid_metric', metrics), dpi=dpi)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")

    def _print_summary(self, context, metrics, strategies):
        """Print a Rich summary table: per-strategy final/min/max for each metric."""
        console = OLConsole()
        table = Table(
            title="Metric Summary",
            box=box.SIMPLE,
            show_header=True,
            header_style="table.header",
        )
        table.add_column("Strategy", style="strategy")
        for metric in metrics:
            label = context.resolver.label(metric)
            table.add_column(f"{label}\nfinal", justify="right", style="metric.value")
            table.add_column(f"{label}\nmin", justify="right", style="detail")
            table.add_column(f"{label}\nmax", justify="right", style="detail")

        for strat in strategies:
            strat_df = context.data[context.data['strategy'] == strat]
            row = [strat]
            for metric in metrics:
                if metric in strat_df.columns:
                    series = strat_df[metric].dropna()
                    if not series.empty:
                        final = series.iloc[-1]
                        row.append(f"{final:.6g}")
                        row.append(f"{series.min():.6g}")
                        row.append(f"{series.max():.6g}")
                    else:
                        row.extend(['—', '—', '—'])
                else:
                    row.extend(['—', '—', '—'])
            table.add_row(*row)

        console.print(table)
