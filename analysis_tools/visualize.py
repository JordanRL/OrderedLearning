"""Visualization framework for analysis tools.

Provides OLFigure for subplot management, plot helper functions that
operate on individual Axes, and EMA smoothing.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .style import STRATEGY_PALETTE


def ema_smooth(values, weight: float):
    """Exponential moving average smoothing, NaN-aware.

    Args:
        values: Array-like of values to smooth.
        weight: EMA weight in (0, 1). Higher = more smoothing.

    Returns:
        numpy array of smoothed values, same length as input.
    """
    values = np.asarray(values, dtype=float)
    smoothed = np.empty_like(values)
    last = np.nan
    for i, v in enumerate(values):
        if np.isnan(v):
            smoothed[i] = last
        elif np.isnan(last):
            smoothed[i] = v
            last = v
        else:
            last = weight * last + (1 - weight) * v
            smoothed[i] = last
    return smoothed


def _grid_cols(n_plots: int) -> int:
    """Choose number of columns for a subplot grid.

    Prefers square-ish layouts: 1→1, 2→2, 3→3, 4→2, 5→3, 6→3, 7→3, ...
    """
    if n_plots <= 3:
        return n_plots
    if n_plots == 4:
        return 2
    return 3


class OLFigure:
    """Managed figure with automatic subplot grid layout.

    Creates a grid of subplots with up to 3 columns, hiding unused cells.

    Args:
        n_plots: Number of subplots needed.
        title: Optional suptitle for the figure.
        share_x: Share x-axis across subplots.
        share_y: Share y-axis across subplots.
    """

    def __init__(self, n_plots: int = 1, title: str | None = None,
                 share_x: bool = True, share_y: bool = False):
        cols = _grid_cols(n_plots)
        rows = math.ceil(n_plots / cols)
        self.fig, axes_grid = plt.subplots(
            rows, cols,
            figsize=(5 * cols, 4 * rows),
            sharex=share_x, sharey=share_y,
            squeeze=False,
        )
        # Flatten to list
        all_axes = axes_grid.flatten().tolist()
        # Visible axes (length = n_plots)
        self._axes = all_axes[:n_plots]
        # Hide unused cells
        for ax in all_axes[n_plots:]:
            ax.set_visible(False)
        if title:
            self.fig.suptitle(title, fontsize=14, fontweight='bold')

    @property
    def axes(self) -> list[Axes]:
        """Only visible axes (length = n_plots)."""
        return self._axes

    def save(self, path: str | Path, dpi: int = 300,
             format: str | None = None) -> Path:
        """Save the figure to disk.

        Args:
            path: Output file path.
            dpi: Resolution.
            format: File format (inferred from extension if None).

        Returns:
            Path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.tight_layout()
        self.fig.savefig(str(path), dpi=dpi, format=format, bbox_inches='tight')
        plt.close(self.fig)
        return path


def plot_time_series(ax: Axes, df, x: str = 'step', y: list[str] | str = None,
                     labels: list[str] | None = None,
                     colors: list[str] | None = None,
                     smooth: float | None = None,
                     log_scale: bool = False):
    """Plot one or more time series on a single axes.

    With smoothing: raw data drawn as faint background (alpha=0.15),
    smoothed EMA as solid line.

    Args:
        ax: Matplotlib axes to plot on.
        df: DataFrame with x column and y columns.
        x: Column name for x-axis.
        y: Column name(s) for y-axis. If None, plots all numeric columns except x.
        labels: Display labels for each y column.
        colors: Colors for each y column.
        smooth: EMA weight (0-1). None for no smoothing.
        log_scale: Use log scale on y-axis.
    """
    if y is None:
        y = [c for c in df.select_dtypes(include='number').columns if c != x]
    elif isinstance(y, str):
        y = [y]

    if labels is None:
        labels = y
    if colors is None:
        colors = STRATEGY_PALETTE[:len(y)]

    x_vals = df[x].values

    for i, col in enumerate(y):
        if col not in df.columns:
            continue
        vals = df[col].values
        color = colors[i % len(colors)]
        label = labels[i] if i < len(labels) else col

        if smooth and smooth > 0:
            # Raw data as faint background
            ax.plot(x_vals, vals, color=color, alpha=0.15, linewidth=0.8)
            # Smoothed as solid
            smoothed = ema_smooth(vals, smooth)
            ax.plot(x_vals, smoothed, color=color, label=label, linewidth=1.5)
        else:
            ax.plot(x_vals, vals, color=color, label=label, linewidth=1.5)

    if log_scale:
        ax.set_yscale('log')
    if len(y) > 1:
        ax.legend()


def plot_multi_axis(ax: Axes, df, x: str = 'step',
                    left_y: list[str] | str = None,
                    right_y: list[str] | str = None,
                    smooth: float | None = None):
    """Plot with dual y-axes via ax.twinx().

    Args:
        ax: Matplotlib axes (becomes left y-axis).
        df: DataFrame with data.
        x: Column name for x-axis.
        left_y: Column(s) for left y-axis.
        right_y: Column(s) for right y-axis.
        smooth: EMA weight for smoothing.
    """
    if isinstance(left_y, str):
        left_y = [left_y]
    if isinstance(right_y, str):
        right_y = [right_y]
    left_y = left_y or []
    right_y = right_y or []

    x_vals = df[x].values

    # Left axis
    for i, col in enumerate(left_y):
        if col not in df.columns:
            continue
        color = STRATEGY_PALETTE[i % len(STRATEGY_PALETTE)]
        vals = df[col].values
        if smooth and smooth > 0:
            ax.plot(x_vals, vals, color=color, alpha=0.15, linewidth=0.8)
            ax.plot(x_vals, ema_smooth(vals, smooth), color=color,
                    label=col, linewidth=1.5)
        else:
            ax.plot(x_vals, vals, color=color, label=col, linewidth=1.5)

    # Right axis
    if right_y:
        ax2 = ax.twinx()
        offset = len(left_y)
        for i, col in enumerate(right_y):
            if col not in df.columns:
                continue
            color = STRATEGY_PALETTE[(offset + i) % len(STRATEGY_PALETTE)]
            vals = df[col].values
            if smooth and smooth > 0:
                ax2.plot(x_vals, vals, color=color, alpha=0.15, linewidth=0.8)
                ax2.plot(x_vals, ema_smooth(vals, smooth), color=color,
                         label=col, linewidth=1.5, linestyle='--')
            else:
                ax2.plot(x_vals, vals, color=color, label=col,
                         linewidth=1.5, linestyle='--')
        ax2.legend(loc='upper left')

    ax.legend(loc='upper right')


def plot_bar(ax: Axes, labels: list[str], values: list[float],
             colors: list[str] | None = None):
    """Simple bar chart.

    Args:
        ax: Matplotlib axes.
        labels: Bar labels.
        values: Bar heights.
        colors: Bar colors.
    """
    if colors is None:
        colors = [STRATEGY_PALETTE[i % len(STRATEGY_PALETTE)]
                  for i in range(len(labels))]
    ax.bar(labels, values, color=colors)
    ax.tick_params(axis='x', rotation=45)


def plot_heatmap(ax: Axes, data: np.ndarray, x_labels=None, y_labels=None,
                 cmap: str = 'viridis', vmin=None, vmax=None,
                 colorbar: bool = True, log_norm: bool = False):
    """Render a 2D array as a heatmap.

    Args:
        ax: Matplotlib axes.
        data: 2D numpy array (rows × columns).
        x_labels: Labels for columns (plotted sparsely if many).
        y_labels: Labels for rows.
        cmap: Colormap name.
        colorbar: Whether to add a colorbar.
        log_norm: Use logarithmic color normalization.
    """
    from matplotlib.colors import LogNorm

    kwargs = {'cmap': cmap, 'aspect': 'auto'}
    if log_norm:
        # Clamp to positive for LogNorm
        safe_min = np.nanmin(data[data > 0]) if np.any(data > 0) else 1e-10
        kwargs['norm'] = LogNorm(vmin=vmin or safe_min, vmax=vmax)
    else:
        if vmin is not None:
            kwargs['vmin'] = vmin
        if vmax is not None:
            kwargs['vmax'] = vmax

    im = ax.imshow(data, **kwargs)

    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=7)
    if x_labels is not None:
        # Show sparse x ticks to avoid overlap
        n = len(x_labels)
        stride = max(1, n // 8)
        ticks = list(range(0, n, stride))
        ax.set_xticks(ticks)
        ax.set_xticklabels([x_labels[i] for i in ticks], fontsize=8)

    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return im
