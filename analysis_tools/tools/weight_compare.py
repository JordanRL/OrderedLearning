"""Weight comparison tool for structural analysis.

Loads final model weights from strategy output directories and compares
their internal structure: per-layer norms, singular value spectra,
and pairwise cosine similarity.

Usage:
    python analyze_experiment.py mod_arithmetic weight_compare

    python analyze_experiment.py presorted weight_compare \
        --analyses norms svd --top-layers 6
"""

from __future__ import annotations

import re
from pathlib import Path
from itertools import combinations

import numpy as np

from rich.table import Table
from rich import box

from console import OLConsole
from ..base import AnalysisTool, AnalysisContext, ToolRegistry
from ..style import get_strategy_colors
from ..visualize import OLFigure, plot_bar, plot_heatmap

import matplotlib.pyplot as plt


def _load_state_dict(strategy_dir: Path, strategy_name: str) -> dict | None:
    """Load model state_dict from a strategy's final weights file.

    Looks for {strategy_name}_final.pt in the strategy directory.
    Returns the state_dict, or None if not found.
    """
    import torch

    weight_path = strategy_dir / f'{strategy_name}_final.pt'
    if not weight_path.exists():
        # Also try any *_final.pt in case naming differs
        candidates = list(strategy_dir.glob('*_final.pt'))
        if candidates:
            weight_path = candidates[0]
        else:
            return None

    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    # Might be a raw state dict
    return checkpoint


def _layer_sort_key(name: str) -> tuple:
    """Sort key for parameter names by layer order."""
    match = re.search(r'\.h\.(\d+)\.|\.layer\.(\d+)\.', name)
    if match:
        num = int(match.group(1) or match.group(2))
        return (1, num, name)
    if any(k in name for k in ('wte', 'wpe', 'embed')):
        return (0, 0, name)
    return (2, 0, name)


def _short_name(param_name: str) -> str:
    """Shorten parameter name for display."""
    # Remove common prefixes
    name = param_name
    for prefix in ('transformer.', 'model.'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def _compute_svd_spectrum(tensor, max_rank: int = 50) -> np.ndarray:
    """Compute singular values of a weight matrix.

    For >2D tensors (e.g., conv layers), reshapes to 2D first.
    Returns up to max_rank singular values, normalized by the largest.
    """
    t = tensor.float()
    if t.dim() < 2:
        return np.array([])
    if t.dim() > 2:
        t = t.reshape(t.shape[0], -1)
    try:
        s = np.linalg.svd(t.numpy(), compute_uv=False)
    except np.linalg.LinAlgError:
        return np.array([])
    s = s[:max_rank]
    if s[0] > 0:
        s = s / s[0]  # Normalize by largest
    return s


@ToolRegistry.register
class WeightCompareTool(AnalysisTool):
    """Compare internal structure of final model weights across strategies."""

    name = "weight_compare"
    description = "Per-layer norm comparison, SVD spectra, and pairwise similarity of final weights"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            '--analyses', nargs='+',
            choices=['norms', 'svd', 'similarity'],
            default=['norms', 'svd', 'similarity'],
            help='Which analyses to run (default: all)',
        )
        parser.add_argument(
            '--top-layers', type=int, default=None,
            dest='top_layers',
            help='Only show the N layers with largest norm difference',
        )
        parser.add_argument(
            '--svd-rank', type=int, default=50,
            dest='svd_rank',
            help='Max singular values to compute per layer (default: 50)',
        )

    def describe_outputs(self) -> list[str]:
        return [
            'norms.png — per-layer weight norm comparison',
            'svd_{layer}.png — singular value spectra per layer',
            'similarity.png — pairwise cosine similarity matrix',
        ]

    def run(self, context: AnalysisContext) -> None:
        console = OLConsole()
        args = context.args
        dpi = getattr(args, 'dpi', 300)
        output_dir = Path(args.output_dir)

        # Load state dicts from strategy directories
        strategies = context.strategies
        state_dicts: dict[str, dict] = {}

        for strat in strategies:
            strat_dir = output_dir / context.experiment_name / strat
            sd = _load_state_dict(strat_dir, strat)
            if sd is not None:
                state_dicts[strat] = sd
                console.print(
                    f"[label]Loaded weights:[/label] [strategy]{strat}[/strategy] "
                    f"[detail]({len(sd)} parameters)[/detail]"
                )
            else:
                console.print(
                    f"[warning.content]No final weights found for "
                    f"strategy '{strat}' in {strat_dir}[/warning.content]"
                )

        if len(state_dicts) < 1:
            console.print(
                "[error.content]No model weights found. "
                "Ensure experiments were run and produced *_final.pt files.[/error.content]"
            )
            return

        loaded_strategies = list(state_dicts.keys())

        # Find shared parameter names (should be identical across strategies)
        shared_params = set(state_dicts[loaded_strategies[0]].keys())
        for strat in loaded_strategies[1:]:
            shared_params &= set(state_dicts[strat].keys())
        # Sort by layer order
        sorted_params = sorted(shared_params, key=_layer_sort_key)

        # Filter to weight matrices (>= 2D) for SVD, keep all for norms
        weight_params = [
            p for p in sorted_params
            if state_dicts[loaded_strategies[0]][p].dim() >= 2
        ]

        console.print(
            f"[label]Shared parameters:[/label] [value.count]{len(sorted_params)}[/value.count] "
            f"[detail]({len(weight_params)} weight matrices)[/detail]"
        )

        analyses = args.analyses

        if 'norms' in analyses:
            self._plot_norms(
                state_dicts, loaded_strategies, sorted_params,
                context, args, dpi,
            )

        if 'svd' in analyses and weight_params:
            self._plot_svd(
                state_dicts, loaded_strategies, weight_params,
                context, args, dpi,
            )

        if 'similarity' in analyses and len(loaded_strategies) >= 2:
            self._plot_similarity(
                state_dicts, loaded_strategies, sorted_params,
                context, dpi,
            )

    def _plot_norms(self, state_dicts, strategies, params, context, args,
                    dpi):
        """Grouped bar chart of per-layer L2 norms."""
        console = OLConsole()
        strat_colors = get_strategy_colors(strategies)

        # Compute norms
        norms = {}
        for strat in strategies:
            norms[strat] = {}
            for param in params:
                t = state_dicts[strat][param].float()
                norms[strat][param] = t.norm().item()

        # Optional: top N by max norm difference
        display_params = params
        if args.top_layers and args.top_layers < len(params):
            diffs = {}
            for param in params:
                vals = [norms[s][param] for s in strategies]
                diffs[param] = max(vals) - min(vals)
            display_params = sorted(diffs, key=diffs.get, reverse=True)[:args.top_layers]
            display_params = sorted(display_params, key=_layer_sort_key)

        short_labels = [_short_name(p) for p in display_params]

        # Grouped bar chart
        n_strats = len(strategies)
        n_params = len(display_params)
        x = np.arange(n_params)
        width = 0.8 / n_strats

        fig, ax = plt.subplots(figsize=(max(8, 0.5 * n_params), 5))
        for i, strat in enumerate(strategies):
            vals = [norms[strat][p] for p in display_params]
            offset = (i - n_strats / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=strat,
                   color=strat_colors[strat])

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=60, ha='right', fontsize=7)
        ax.set_ylabel('L2 Norm')
        ax.set_title(f'{context.experiment_name}: Weight Norms' if context.args.experiment_title else 'Weight Norms')
        ax.legend()

        path = context.output_path('norms')
        fig.savefig(str(path), dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")

    def _plot_svd(self, state_dicts, strategies, weight_params, context,
                  args, dpi):
        """Singular value spectra overlaid per layer."""
        console = OLConsole()
        strat_colors = get_strategy_colors(strategies)
        svd_rank = args.svd_rank

        # Select layers to plot — pick a representative subset if many
        params_to_plot = weight_params
        if args.top_layers and args.top_layers < len(weight_params):
            # Pick layers with most spectral divergence
            divergences = {}
            for param in weight_params:
                spectra = []
                for strat in strategies:
                    s = _compute_svd_spectrum(state_dicts[strat][param], svd_rank)
                    if len(s) > 0:
                        spectra.append(s)
                if len(spectra) >= 2:
                    # Pad to same length
                    max_len = max(len(s) for s in spectra)
                    padded = [np.pad(s, (0, max_len - len(s))) for s in spectra]
                    # Max pairwise L2 distance
                    max_dist = 0
                    for a, b in combinations(padded, 2):
                        max_dist = max(max_dist, np.linalg.norm(a - b))
                    divergences[param] = max_dist
                else:
                    divergences[param] = 0
            params_to_plot = sorted(divergences, key=divergences.get, reverse=True)
            params_to_plot = params_to_plot[:args.top_layers]

        n_plots = len(params_to_plot)
        if n_plots == 0:
            return

        fig = OLFigure(
            n_plots=n_plots,
            title=f'{context.experiment_name}: SVD Spectra' if context.args.experiment_title else 'SVD Spectra',
            share_x=False, share_y=False,
        )

        for i, param in enumerate(params_to_plot):
            ax = fig.axes[i]
            for strat in strategies:
                s = _compute_svd_spectrum(state_dicts[strat][param], svd_rank)
                if len(s) > 0:
                    ax.plot(range(len(s)), s, color=strat_colors[strat],
                            label=strat, linewidth=1.2)
            ax.set_title(_short_name(param), fontsize=9)
            ax.set_xlabel('rank')
            ax.set_ylabel('σ / σ₁')
            if i == 0:
                ax.legend(fontsize=7)

        path = fig.save(context.output_path('svd'), dpi=dpi)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")

    def _plot_similarity(self, state_dicts, strategies, params, context,
                         dpi):
        """Pairwise cosine similarity matrix between strategies."""
        console = OLConsole()

        n = len(strategies)
        sim_matrix = np.zeros((n, n))

        # Flatten all parameters into one vector per strategy
        flat_vecs = {}
        for strat in strategies:
            parts = []
            for param in params:
                parts.append(state_dicts[strat][param].float().flatten())
            import torch
            flat_vecs[strat] = torch.cat(parts)

        for i, s1 in enumerate(strategies):
            for j, s2 in enumerate(strategies):
                v1, v2 = flat_vecs[s1], flat_vecs[s2]
                cos = (v1 @ v2) / (v1.norm() * v2.norm() + 1e-10)
                sim_matrix[i, j] = cos.item()

        fig, ax = plt.subplots(figsize=(max(5, 0.8 * n + 2), max(4, 0.8 * n + 1)))
        im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='equal')

        ax.set_xticks(range(n))
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.set_yticks(range(n))
        ax.set_yticklabels(strategies)
        ax.set_title(f'{context.experiment_name}: Weight Cosine Similarity' if context.args.experiment_title else 'Weight Cosine Similarity')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                color = 'white' if abs(sim_matrix[i, j]) > 0.7 else 'black'
                ax.text(j, i, f'{sim_matrix[i, j]:.3f}',
                        ha='center', va='center', fontsize=9, color=color)

        path = context.output_path('similarity')
        fig.savefig(str(path), dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")

        # Print table
        table = Table(
            title="Pairwise Cosine Similarity",
            box=box.SIMPLE,
            show_header=True,
            header_style="table.header",
        )
        table.add_column("", style="strategy")
        for strat in strategies:
            table.add_column(strat, justify="right", style="metric.value")

        for i, s1 in enumerate(strategies):
            row = [s1]
            for j, s2 in enumerate(strategies):
                row.append(f'{sim_matrix[i, j]:.4f}')
            table.add_row(*row)

        console.print(table)
