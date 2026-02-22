"""Weight matrix heatmap visualization tool.

Loads final model weights and renders every 2D weight matrix as a
heatmap in forward-pass order. For tall matrices (embedding, decoder),
produces additional SVD projection and SVD-sorted views that reveal
learned structure.

Usage:
    python analyze_experiment.py mod_arithmetic weight_heatmap

    python analyze_experiment.py mod_arithmetic weight_heatmap \
        --svd-components 16 --strategies stride random
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from console import OLConsole
from ..base import AnalysisTool, AnalysisContext, ToolRegistry
from ..data_loader import load_state_dict
from ..param_labels import label_param, param_sort_key, _strip_compile_prefix
from ..visualize import OLFigure, plot_heatmap


def _is_tall_matrix(tensor, threshold_ratio: float = 10.0) -> bool:
    """Check if a weight matrix has extreme aspect ratio (e.g., embedding)."""
    if tensor.dim() < 2:
        return False
    return tensor.shape[0] > threshold_ratio * tensor.shape[1]


def _svd_project(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Project matrix onto top-k left singular vectors.

    Returns:
        (projected, sort_order) where projected is (rows, k) with rows
        sorted by first component, and sort_order is the permutation.
    """
    U, S, _ = np.linalg.svd(matrix, full_matrices=False)
    projected = U[:, :k] * S[:k]
    order = np.argsort(projected[:, 0])
    return projected[order], order


@ToolRegistry.register
class WeightHeatmapTool(AnalysisTool):
    """Visualize weight matrices as heatmaps with SVD views for tall matrices."""

    name = "weight_heatmap"
    description = "Heatmaps of all weight matrices, with SVD views for embeddings"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            '--svd-components', type=int, default=8,
            dest='svd_components',
            help='Number of SVD components for projection view (default: 8)',
        )

    def describe_outputs(self) -> list[str]:
        return [
            '{strategy}_weights.png — grid of weight matrix heatmaps',
            '{strategy}_{param}_full.png — native-resolution heatmap (tall matrices)',
            '{strategy}_{param}_sorted.png — SVD-sorted heatmap (tall matrices)',
            '{strategy}_{param}_svd.png — SVD projection (tall matrices)',
        ]

    def run(self, context: AnalysisContext) -> None:
        console = OLConsole()
        args = context.args
        dpi = getattr(args, 'dpi', 300)
        ext = getattr(args, 'format', 'png') or 'png'
        k = args.svd_components

        strategies = context.strategies
        output_dir_str = str(getattr(args, 'output_dir', 'output'))

        for strat in strategies:
            sd = load_state_dict(context.experiment_name, strat,
                                 output_dir=output_dir_str)
            if sd is None:
                console.print(
                    f"[warning.content]No final weights found for '{strat}'[/warning.content]"
                )
                continue

            console.print(
                f"[label]Loaded weights:[/label] [strategy]{strat}[/strategy] "
                f"[detail]({len(sd)} parameters)[/detail]"
            )

            # Separate 2D params into normal and tall
            params_2d = [
                (name, tensor) for name, tensor in sd.items()
                if tensor.dim() >= 2
            ]
            params_2d.sort(key=lambda x: param_sort_key(x[0]))

            normal_params = [(n, t) for n, t in params_2d if not _is_tall_matrix(t)]
            tall_params = [(n, t) for n, t in params_2d if _is_tall_matrix(t)]

            # Grid of normal-sized weight matrices
            if normal_params:
                self._plot_weight_grid(
                    normal_params, strat, context, dpi, ext,
                )

            # Individual views for tall matrices
            for name, tensor in tall_params:
                matrix = tensor.float().numpy()
                clean_name = _strip_compile_prefix(name)
                label = label_param(name)
                # Filename-safe slug
                slug = clean_name.replace('.', '_')

                self._plot_tall_full(
                    matrix, label, strat, slug, context, dpi, ext,
                )
                self._plot_tall_svd(
                    matrix, label, strat, slug, k, context, dpi, ext,
                )

    def _plot_weight_grid(self, params, strategy, context, dpi, ext):
        """Render grid of normal-sized weight matrix heatmaps."""
        console = OLConsole()
        exp_title = (context.experiment_name
                     if context.args.experiment_title else None)
        fig = OLFigure(n_plots=len(params), title=exp_title,
                       share_x=False, share_y=False)

        for i, (name, tensor) in enumerate(params):
            ax = fig.axes[i]
            data = tensor.float().numpy()
            plot_heatmap(ax, data)
            ax.set_title(label_param(name), fontsize=9)

        path = context.output_dir / f'{strategy}_weights.{ext}'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.save(path, dpi=dpi)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")

    def _plot_tall_full(self, matrix, label, strategy, slug, context, dpi, ext):
        """Render native-resolution heatmap of a tall weight matrix."""
        console = OLConsole()
        rows, cols = matrix.shape
        height = max(6, min(24, rows / 400))
        fig, ax = plt.subplots(figsize=(8, height))
        plot_heatmap(ax, matrix, colorbar=True)
        ax.set_title(f'{strategy}: {label}')
        ax.set_ylabel('Token index')
        ax.set_xlabel('Dimension')

        path = context.output_dir / f'{strategy}_{slug}_full.{ext}'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")

    def _plot_tall_svd(self, matrix, label, strategy, slug, k, context,
                       dpi, ext):
        """Render SVD-sorted full heatmap and SVD projection for a tall matrix."""
        console = OLConsole()
        rows, cols = matrix.shape
        k = min(k, min(rows, cols))

        try:
            projected, order = _svd_project(matrix, k)
        except np.linalg.LinAlgError:
            console.print(
                f"[warning.content]SVD failed for {label}, skipping SVD views[/warning.content]"
            )
            return

        height = max(6, min(24, rows / 400))

        # SVD-sorted full heatmap
        sorted_matrix = matrix[order]
        fig, ax = plt.subplots(figsize=(8, height))
        plot_heatmap(ax, sorted_matrix, colorbar=True)
        ax.set_title(f'{strategy}: {label} (SVD-sorted)')
        ax.set_ylabel('Token index (sorted by 1st SVD component)')
        ax.set_xlabel('Dimension')

        path = context.output_dir / f'{strategy}_{slug}_sorted.{ext}'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")

        # SVD projection (compact view)
        fig, ax = plt.subplots(figsize=(max(4, k * 0.6), height))
        plot_heatmap(
            ax, projected,
            x_labels=[f'SV {i}' for i in range(k)],
            colorbar=True,
        )
        ax.set_title(f'{strategy}: {label} (top-{k} SVD)')
        ax.set_ylabel('Token index (sorted by 1st component)')

        path = context.output_dir / f'{strategy}_{slug}_svd.{ext}'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        console.print(f"[label]Saved:[/label] [path]{path}[/path]")
