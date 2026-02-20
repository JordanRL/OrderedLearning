"""Table export tool for papers.

Generates LaTeX or Markdown tables of aggregate metrics per strategy,
with configurable precision and optional bolding of best values.

Usage:
    python analyze_experiment.py mod_arithmetic export_table \
        --metrics training_metrics/loss training_metrics/val_acc \
        --stat final --table-format latex --bold-best
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from console import OLConsole
from ..base import AnalysisTool, AnalysisContext, ToolRegistry


def _compute_stat(series, stat: str):
    """Compute aggregate statistic, returning NaN for empty series."""
    series = series.dropna()
    if series.empty:
        return np.nan
    funcs = {
        'final': lambda s: s.iloc[-1],
        'min': lambda s: s.min(),
        'max': lambda s: s.max(),
        'mean': lambda s: s.mean(),
    }
    return funcs[stat](series)


def _is_lower_better(metric: str) -> bool:
    """Heuristic: metrics with loss/error/perplexity → lower is better."""
    lower = metric.lower()
    return any(k in lower for k in ('loss', 'error', 'perplexity'))


@ToolRegistry.register
class ExportTableTool(AnalysisTool):
    """Export metric comparison tables in LaTeX or Markdown format."""

    name = "export_table"
    description = "Export LaTeX or Markdown tables of aggregate metrics for papers"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            '--metrics', nargs='+', required=True,
            help='Metric columns to include (hook_name/metric_name format)',
        )
        parser.add_argument(
            '--stat', choices=['final', 'min', 'max', 'mean'], default='final',
            help='Aggregate statistic to report (default: final)',
        )
        parser.add_argument(
            '--table-format', choices=['latex', 'markdown'], default='latex',
            dest='table_format',
            help='Output table format (default: latex)',
        )
        parser.add_argument(
            '--precision', type=int, default=4,
            help='Decimal places for values (default: 4)',
        )
        parser.add_argument(
            '--bold-best', action='store_true', default=False,
            dest='bold_best',
            help='Bold the best value per metric column',
        )

    def describe_outputs(self) -> list[str]:
        return [
            'table.tex — LaTeX table',
            'table.md — Markdown table',
        ]

    def run(self, context: AnalysisContext) -> None:
        console = OLConsole()
        args = context.args
        stat = args.stat
        precision = args.precision
        bold_best = args.bold_best
        table_format = args.table_format

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

        # Compute values: {metric: {strategy: value}}
        values = {}
        for metric in valid_metrics:
            values[metric] = {}
            for strat in strategies:
                series = context.data.loc[
                    context.data['strategy'] == strat, metric
                ]
                values[metric][strat] = _compute_stat(series, stat)

        # Find best per metric
        best = {}
        if bold_best:
            for metric in valid_metrics:
                vals = {s: values[metric][s] for s in strategies
                        if not np.isnan(values[metric][s])}
                if vals:
                    if _is_lower_better(metric):
                        best[metric] = min(vals, key=vals.get)
                    else:
                        best[metric] = max(vals, key=vals.get)

        # Generate table
        # Resolve display headers
        headers = [context.resolver.label(m) for m in valid_metrics]

        if table_format == 'latex':
            text = self._render_latex(
                valid_metrics, strategies, values, best, precision, stat,
                headers,
            )
            ext = 'tex'
        else:
            text = self._render_markdown(
                valid_metrics, strategies, values, best, precision, stat,
                headers,
            )
            ext = 'md'

        # Save to file
        out_path = context.output_path('table', valid_metrics, ext=ext)
        out_path.write_text(text)
        console.print(f"[label]Saved:[/label] [path]{out_path}[/path]")

        # Print to console
        console.print()
        console.print(text)

    def _render_latex(self, metrics, strategies, values, best, precision, stat,
                      headers):
        """Render a LaTeX tabular environment."""
        col_spec = 'l' + 'r' * len(metrics)
        lines = []
        lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
        lines.append('\\toprule')
        lines.append('Strategy & ' + ' & '.join(headers) + ' \\\\')
        lines.append('\\midrule')

        for strat in strategies:
            cells = [strat.replace('_', '\\_')]
            for metric in metrics:
                v = values[metric][strat]
                if np.isnan(v):
                    cells.append('---')
                else:
                    formatted = f'{v:.{precision}f}'
                    if best.get(metric) == strat:
                        formatted = f'\\textbf{{{formatted}}}'
                    cells.append(formatted)
            lines.append(' & '.join(cells) + ' \\\\')

        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')

        # Add caption comment
        lines.insert(0, f'% {stat} values across strategies')
        return '\n'.join(lines)

    def _render_markdown(self, metrics, strategies, values, best,
                         precision, stat, headers):
        """Render a Markdown table."""
        lines = []
        lines.append('| Strategy | ' + ' | '.join(headers) + ' |')
        lines.append('|' + '|'.join(['---'] + ['---:'] * len(metrics)) + '|')

        for strat in strategies:
            cells = [strat]
            for metric in metrics:
                v = values[metric][strat]
                if np.isnan(v):
                    cells.append('---')
                else:
                    formatted = f'{v:.{precision}f}'
                    if best.get(metric) == strat:
                        formatted = f'**{formatted}**'
                    cells.append(formatted)
            lines.append('| ' + ' | '.join(cells) + ' |')

        return '\n'.join(lines)
