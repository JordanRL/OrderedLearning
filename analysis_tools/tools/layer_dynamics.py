"""Layer dynamics heatmap tool.

Visualizes per-parameter hook metrics over training as heatmaps:
parameters on the y-axis, steps on the x-axis, color intensity = metric
value. Shows which layers are active when during training.

Designed for per-parameter metrics exported by hooks (column names with
3+ slash-separated segments like ``norms/grad_norm/transformer.h.0.attn.weight``).

Usage:
    python analyze_experiment.py mod_arithmetic layer_dynamics \
        --metric norms/grad_norm

    python analyze_experiment.py presorted layer_dynamics \
        --metric norms/weight_norm --aggregate-layers --log-norm
"""

from __future__ import annotations

import re

import numpy as np

from rich.table import Table
from rich import box

from console import OLConsole
from ..base import AnalysisTool, AnalysisContext, ToolRegistry
from ..visualize import plot_heatmap

import matplotlib.pyplot as plt


def _extract_param_columns(all_columns: list[str], prefix: str) -> dict[str, str]:
    """Find columns matching a metric prefix and extract parameter names.

    Args:
        all_columns: All DataFrame column names.
        prefix: Metric prefix (e.g., 'norms/grad_norm').

    Returns:
        Dict of {column_name: param_name} for matching columns.
    """
    prefix_slash = prefix if prefix.endswith('/') else prefix + '/'
    result = {}
    for col in all_columns:
        if col.startswith(prefix_slash):
            param = col[len(prefix_slash):]
            if param:
                result[col] = param
    return result


def _layer_sort_key(param_name: str) -> tuple:
    """Sort key that groups by layer number, then by component name.

    Extracts numeric layer indices from patterns like 'h.0', 'h.12', etc.
    Non-layer params (embeddings, final norm) sort to the edges.
    """
    # Find layer number pattern like .h.N. or .layer.N.
    match = re.search(r'\.h\.(\d+)\.|\.layer\.(\d+)\.', param_name)
    if match:
        layer_num = int(match.group(1) or match.group(2))
        # Sort by: (1=middle, layer_num, component name)
        return (1, layer_num, param_name)
    # Embeddings and early params sort first
    if any(k in param_name for k in ('wte', 'wpe', 'embed')):
        return (0, 0, param_name)
    # Final norm / head sort last
    return (2, 0, param_name)


def _aggregate_by_layer(columns: dict[str, str], data, steps) -> tuple[list[str], np.ndarray]:
    """Average per-parameter values within each layer block.

    Groups parameters by their layer identifier (e.g., 'h.0', 'h.1',
    'embed', 'ln_f') and averages within each group.

    Returns:
        (layer_labels, aggregated_data) where aggregated_data is
        (n_layers, n_steps).
    """
    groups: dict[str, list[str]] = {}
    for col, param in columns.items():
        match = re.search(r'(h\.(\d+))', param)
        if match:
            group = match.group(1)
        elif any(k in param for k in ('wte', 'wpe', 'embed')):
            group = 'embed'
        elif 'ln_f' in param:
            group = 'ln_f'
        else:
            group = 'other'
        groups.setdefault(group, []).append(col)

    # Sort groups: embed first, then h.0, h.1, ..., then ln_f, other
    def group_sort_key(name):
        if name == 'embed':
            return (0, 0)
        match = re.match(r'h\.(\d+)', name)
        if match:
            return (1, int(match.group(1)))
        if name == 'ln_f':
            return (2, 0)
        return (3, 0)

    sorted_groups = sorted(groups.keys(), key=group_sort_key)

    labels = []
    rows = []
    for group in sorted_groups:
        cols = groups[group]
        # Average across all params in this group
        group_data = data[cols].values
        row = np.nanmean(group_data, axis=1)
        labels.append(group)
        rows.append(row)

    return labels, np.array(rows)


@ToolRegistry.register
class LayerDynamicsTool(AnalysisTool):
    """Heatmap of per-parameter metrics over training steps."""

    name = "layer_dynamics"
    description = "Heatmap showing per-layer/parameter metric evolution over training"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            '--metric', required=True,
            help='Metric prefix to visualize (e.g., norms/grad_norm). '
                 'Discovers all per-parameter columns with this prefix.',
        )
        parser.add_argument(
            '--aggregate-layers', action='store_true', default=False,
            dest='aggregate_layers',
            help='Average parameters within each layer block',
        )
        parser.add_argument(
            '--log-norm', action='store_true', default=False,
            dest='log_norm',
            help='Use logarithmic color normalization',
        )
        parser.add_argument(
            '--sort', choices=['name', 'variance'], default='name',
            help='Sort parameters by name (layer order) or variance (most variable on top)',
        )
        parser.add_argument(
            '--top', type=int, default=None,
            help='Show only the N parameters with highest variance',
        )

    def describe_outputs(self) -> list[str]:
        return ['heatmap_{strategy}.png — per-parameter metric heatmap']

    def run(self, context: AnalysisContext) -> None:
        console = OLConsole()
        args = context.args
        prefix = args.metric
        dpi = getattr(args, 'dpi', 300)

        # Find per-parameter columns matching prefix
        all_cols = list(context.data.columns)
        param_cols = _extract_param_columns(all_cols, prefix)

        if not param_cols:
            console.print(
                f"[error.content]No per-parameter columns found for prefix '{prefix}'[/error.content]"
            )
            # Show available prefixes (columns with 2+ slashes)
            prefixes = set()
            for col in all_cols:
                parts = col.split('/')
                if len(parts) >= 3:
                    prefixes.add('/'.join(parts[:2]))
            if prefixes:
                console.print("[label]Available metric prefixes:[/label]")
                for p in sorted(prefixes):
                    console.print(f"  [metric.value]{p}[/metric.value]")
            else:
                console.print(
                    "[detail]No per-parameter metrics found. "
                    "Re-run experiment with hooks that produce per-parameter "
                    "metrics (e.g., norms, parameter_delta).[/detail]"
                )
            return

        strategies = context.strategies
        console.print(
            f"[label]Found[/label] [value.count]{len(param_cols)}[/value.count] "
            f"[label]parameters for[/label] [metric.value]{prefix}[/metric.value]"
        )

        # One heatmap per strategy
        for strat in strategies:
            strat_df = context.data[context.data['strategy'] == strat].copy()
            if strat_df.empty:
                continue

            steps = strat_df['step'].values
            col_names = list(param_cols.keys())

            if args.aggregate_layers:
                labels, matrix = _aggregate_by_layer(param_cols, strat_df, steps)
            else:
                # Sort parameters
                if args.sort == 'variance':
                    variances = {
                        col: strat_df[col].var() for col in col_names
                        if col in strat_df.columns
                    }
                    sorted_cols = sorted(variances, key=variances.get, reverse=True)
                else:
                    sorted_cols = sorted(col_names, key=lambda c: _layer_sort_key(param_cols[c]))

                # Top N filter
                if args.top and args.top < len(sorted_cols):
                    if args.sort != 'variance':
                        # Sort by variance first to pick top, then re-sort
                        by_var = sorted(
                            sorted_cols,
                            key=lambda c: strat_df[c].var() if c in strat_df.columns else 0,
                            reverse=True,
                        )
                        sorted_cols = by_var[:args.top]
                        sorted_cols = sorted(sorted_cols, key=lambda c: _layer_sort_key(param_cols[c]))
                    else:
                        sorted_cols = sorted_cols[:args.top]

                labels = [param_cols[c] for c in sorted_cols]
                matrix = strat_df[sorted_cols].values.T  # (n_params, n_steps)

            if matrix.size == 0:
                continue

            # Build figure — taller for many parameters
            n_rows = len(labels)
            height = max(4, min(20, 0.3 * n_rows + 2))
            fig, ax = plt.subplots(figsize=(10, height))

            metric_label = context.resolver.label(prefix)
            title = f'{strat}: {metric_label}'
            if args.aggregate_layers:
                title += ' (layer avg)'

            plot_heatmap(
                ax, matrix,
                x_labels=[str(int(s)) for s in steps],
                y_labels=labels,
                log_norm=args.log_norm,
            )
            ax.set_title(title)
            ax.set_xlabel('step')

            # Save
            out_path = context.output_path(f'heatmap_{strat}', [prefix])
            fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            console.print(f"[label]Saved:[/label] [path]{out_path}[/path]")

        # Print summary: top parameters by variance across all strategies
        self._print_variance_summary(context, param_cols, prefix, strategies)

    def _print_variance_summary(self, context, param_cols, prefix, strategies):
        """Print the parameters with highest variance across training."""
        console = OLConsole()
        table = Table(
            title=f"Top Parameters by Variance ({prefix})",
            box=box.SIMPLE,
            show_header=True,
            header_style="table.header",
        )
        table.add_column("Parameter", style="trigger")
        for strat in strategies:
            table.add_column(strat, justify="right", style="metric.value")

        # Compute variance per (param, strategy)
        all_variances = {}
        for col, param in param_cols.items():
            for strat in strategies:
                series = context.data.loc[
                    context.data['strategy'] == strat, col
                ].dropna()
                all_variances.setdefault(param, {})[strat] = (
                    series.var() if not series.empty else 0.0
                )

        # Rank by max variance across strategies
        ranked = sorted(
            all_variances.keys(),
            key=lambda p: max(all_variances[p].values()),
            reverse=True,
        )

        for param in ranked[:15]:
            row = [param]
            for strat in strategies:
                v = all_variances[param].get(strat, 0.0)
                row.append(f'{v:.4g}')
            table.add_row(*row)

        if len(ranked) > 15:
            table.add_row(
                f'[detail]... {len(ranked) - 15} more[/detail]',
                *([''] * len(strategies)),
            )

        console.print(table)
