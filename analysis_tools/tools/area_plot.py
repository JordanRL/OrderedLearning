"""Stacked area plot analysis tool.

Visualizes distribution-valued metrics (e.g., fourier/freq_powers) as
stacked area charts showing how components accumulate over training.

Usage:
    python analyze_experiment.py mod_arithmetic area_plot \
        --metric fourier/freq_powers

    python analyze_experiment.py mod_arithmetic area_plot \
        --metric fourier/freq_powers --top-n 10 --sort-by power

    python analyze_experiment.py mod_arithmetic area_plot \
        --metric fourier/freq_powers --normalize
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from rich.table import Table
from rich import box

from console import OLConsole
from ..base import AnalysisTool, AnalysisContext, ToolRegistry


def _extract_components(data, step_col, metric_col):
    """Extract component time series from a dict-valued column.

    Args:
        data: DataFrame filtered to one strategy.
        step_col: Column name for x-axis (e.g. 'step').
        metric_col: Column name containing dict values.

    Returns:
        (steps, components) where steps is a 1D array and components is
        a dict mapping component key to 1D array of values (0 where absent).
        Returns (None, None) if no dict-valued rows exist.
    """
    rows = []
    for _, row in data.iterrows():
        val = row[metric_col]
        if isinstance(val, dict):
            rows.append((row[step_col], val))

    if not rows:
        return None, None

    # Collect all component keys that appear anywhere
    all_keys = set()
    for _, d in rows:
        all_keys.update(d.keys())

    steps = np.array([s for s, _ in rows])
    components = {}
    for key in all_keys:
        values = np.array([float(d.get(key, 0.0)) for _, d in rows])
        components[key] = values

    return steps, components


def _sort_components(components, sort_by):
    """Return sorted list of (key, values) pairs.

    Args:
        components: Dict mapping key to values array.
        sort_by: 'index' (numeric key order), 'power' (final value
                 descending), or 'acquired' (first appearance order).

    Returns:
        Sorted list of (key, values) tuples.
    """
    items = list(components.items())

    if sort_by == 'power':
        items.sort(key=lambda kv: kv[1][-1], reverse=True)
    elif sort_by == 'index':
        items.sort(key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])
    elif sort_by == 'acquired':
        def first_nonzero(values):
            nz = np.nonzero(values)[0]
            return nz[0] if len(nz) > 0 else len(values)
        items.sort(key=lambda kv: first_nonzero(kv[1]))

    return items


def _apply_top_n(sorted_items, top_n):
    """Limit to top N components, aggregating the rest as 'other'.

    Args:
        sorted_items: Pre-sorted list of (key, values) tuples.
        top_n: Max components to show individually, or None for all.

    Returns:
        (top_items, other) where other is ('other', summed_values) or None.
    """
    if top_n is None or top_n >= len(sorted_items):
        return sorted_items, None

    top = sorted_items[:top_n]
    rest = sorted_items[top_n:]

    other_values = np.sum([v for _, v in rest], axis=0)
    return top, ('other', other_values)


def _component_colors(n, has_other=False):
    """Generate visually distinct colors for stacked area bands.

    Uses tab20 for up to 20 components, falls back to cycling for more.
    The 'other' band (if present) is always neutral gray.

    Args:
        n: Total number of bands including 'other'.
        has_other: Whether the last band is the 'other' aggregate.

    Returns:
        List of RGBA tuples.
    """
    n_real = n - 1 if has_other else n
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20 / 20) for i in range(n_real)]
    if has_other:
        colors.append((0.33, 0.33, 0.33, 0.6))
    return colors


@ToolRegistry.register
class AreaPlotTool(AnalysisTool):
    """Stacked area plot for distribution-valued metrics."""

    name = "area_plot"
    description = "Stacked area charts for distribution-valued metrics (e.g., freq_powers)"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            '--metric', required=True,
            help='Dict-valued metric column (e.g., fourier/freq_powers)',
        )
        parser.add_argument(
            '--top-n', type=int, default=None,
            dest='top_n',
            help='Show only top N components (rest grouped as "other")',
        )
        parser.add_argument(
            '--sort-by', choices=['index', 'power', 'acquired'],
            default='power', dest='sort_by',
            help='Component ordering: index (numeric), power (final value), '
                 'acquired (first appearance)',
        )
        parser.add_argument(
            '--normalize', action='store_true', default=False,
            help='Normalize each timestep so components sum to 1.0',
        )
        parser.add_argument(
            '--share-y', action='store_true', default=False,
            dest='share_y',
            help='Use consistent y-axis scale across all subplots',
        )

    def describe_outputs(self) -> list[str]:
        return [
            'area_{metric}.png — stacked area chart (one subplot per strategy)',
        ]

    def run(self, context: AnalysisContext) -> None:
        console = OLConsole()
        args = context.args
        metric = args.metric
        dpi = getattr(args, 'dpi', 300)

        # Validate metric exists
        if metric not in context.data.columns:
            available_cols = set(context.data.columns) - {'step', 'strategy'}
            console.print(
                f"[error.content]Metric '{metric}' not found. Available:[/error.content]"
            )
            for col in sorted(available_cols):
                console.print(f"  [metric.value]{col}[/metric.value]")
            return

        # Verify it contains dict values
        non_null = context.data[metric].dropna()
        if non_null.empty or not isinstance(non_null.iloc[0], dict):
            console.print(
                f"[error.content]Metric '{metric}' does not contain dict values. "
                f"Area plots require distribution-valued metrics.[/error.content]"
            )
            return

        strategies = context.strategies
        share_y = getattr(args, 'share_y', False)
        self._plot_area(context, metric, strategies, args.sort_by,
                        args.top_n, args.normalize, share_y, dpi)
        self._print_summary(context, metric, strategies)

    def _plot_area(self, context, metric, strategies, sort_by, top_n,
                   normalize, share_y, dpi):
        """Render stacked area chart with one subplot per strategy."""
        console = OLConsole()
        exp_title = (context.experiment_name
                     if context.args.experiment_title else None)
        metric_label = context.resolver.label(metric)

        from ..visualize import OLFigure
        fig = OLFigure(
            n_plots=len(strategies), title=exp_title,
            share_x=False, share_y=share_y,
        )

        for i, strat in enumerate(strategies):
            ax = fig.axes[i]
            strat_df = context.data[context.data['strategy'] == strat]

            steps, components = _extract_components(strat_df, 'step', metric)
            if steps is None or not components:
                ax.set_title(strat)
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', color='#8A8F98')
                continue

            sorted_items = _sort_components(components, sort_by)
            top_items, other = _apply_top_n(sorted_items, top_n)

            # Build parallel arrays for stackplot
            labels = []
            arrays = []
            for key, values in top_items:
                labels.append(str(key))
                arrays.append(values)
            has_other = other is not None
            if has_other:
                labels.append(other[0])
                arrays.append(other[1])

            if normalize:
                stacked = np.array(arrays)
                totals = stacked.sum(axis=0)
                totals[totals == 0] = 1.0
                arrays = [a / totals for a in arrays]

            colors = _component_colors(len(arrays), has_other=has_other)
            ax.stackplot(steps, *arrays, labels=labels, colors=colors,
                         alpha=0.85)
            ax.set_title(strat)
            ax.set_xlabel(context.x_label)
            ax.set_ylabel('fraction' if normalize else metric_label)

            # Legend placement depends on component count
            n_entries = len(labels)
            if n_entries <= 10:
                ax.legend(fontsize=7, loc='upper left')
            else:
                ax.legend(fontsize=6, ncol=2, loc='upper left')

        slug = metric.replace('/', '_')
        path = fig.save(context.output_path(f'area_{slug}', [metric]),
                        dpi=dpi)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")

    def _print_summary(self, context, metric, strategies):
        """Print component summary table."""
        console = OLConsole()
        metric_label = context.resolver.label(metric)

        table = Table(
            title=f"{metric_label} Component Summary",
            box=box.SIMPLE,
            show_header=True,
            header_style="table.header",
        )
        table.add_column("Strategy", style="strategy")
        table.add_column("Components", justify="right", style="metric.value")
        table.add_column("Total Power\n(final)", justify="right",
                         style="metric.value")
        table.add_column("Top Component", style="detail")
        table.add_column("Top Power\n(final)", justify="right", style="detail")

        for strat in strategies:
            strat_df = context.data[context.data['strategy'] == strat]
            steps, components = _extract_components(strat_df, 'step', metric)

            if steps is None or not components:
                table.add_row(strat, '\u2014', '\u2014', '\u2014', '\u2014')
                continue

            n_components = len(components)
            total_final = sum(v[-1] for v in components.values())

            sorted_items = _sort_components(components, 'power')
            top_key, top_vals = sorted_items[0]

            table.add_row(
                strat,
                str(n_components),
                f'{total_final:.4g}',
                str(top_key),
                f'{top_vals[-1]:.4g}',
            )

        console.print(table)
