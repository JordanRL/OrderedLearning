"""Color palettes and matplotlib style for analysis tools.

Two style modes:
- 'dark': matches OLDarkTheme console colors (default)
- 'paper': clean white background with Wong colorblind-friendly palette,
  suitable for dropping into publications

Each mode has its own strategy and metric palettes. The active mode is set
by apply_style() and consulted by get_strategy_colors / get_metric_colors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


# --- Dark palette (OLDarkTheme) ---

_DARK_BLUE = '#61AFEF'
_DARK_RICH_BLUE = '#4B6BFF'
_DARK_CYAN = '#56B6C2'
_DARK_GREEN = '#98C379'
_DARK_YELLOW = '#E5C07B'
_DARK_RED = '#E06C75'
_DARK_ORANGE = '#D19A66'
_DARK_LAVENDER = '#B87FD9'
_DARK_MAGENTA = '#BE50AE'
_DARK_PINK = '#FF69B4'

_DARK_STRATEGY_PALETTE = [
    _DARK_BLUE, _DARK_RED, _DARK_GREEN, _DARK_YELLOW, _DARK_MAGENTA,
    _DARK_CYAN, _DARK_ORANGE, _DARK_LAVENDER, _DARK_PINK, _DARK_RICH_BLUE,
]

_DARK_METRIC_PALETTE = [
    _DARK_CYAN, _DARK_ORANGE, _DARK_GREEN, _DARK_RED, _DARK_LAVENDER,
    _DARK_YELLOW, _DARK_BLUE, _DARK_MAGENTA, _DARK_PINK, _DARK_RICH_BLUE,
]


# --- Paper palette (Wong colorblind-friendly, Nature Methods 2011) ---

_PAPER_BLUE = '#0072B2'
_PAPER_VERMILLION = '#D55E00'
_PAPER_GREEN = '#009E73'
_PAPER_ORANGE = '#E69F00'
_PAPER_PURPLE = '#CC79A7'
_PAPER_SKY = '#56B4E9'
_PAPER_YELLOW = '#F0E442'
_PAPER_BLACK = '#000000'

_PAPER_STRATEGY_PALETTE = [
    _PAPER_BLUE, _PAPER_VERMILLION, _PAPER_GREEN, _PAPER_ORANGE,
    _PAPER_PURPLE, _PAPER_SKY, _PAPER_YELLOW, _PAPER_BLACK,
]

_PAPER_METRIC_PALETTE = [
    _PAPER_SKY, _PAPER_ORANGE, _PAPER_GREEN, _PAPER_VERMILLION,
    _PAPER_PURPLE, _PAPER_YELLOW, _PAPER_BLUE, _PAPER_BLACK,
]


# --- Public aliases (updated by apply_style) ---

STRATEGY_PALETTE: list[str] = list(_DARK_STRATEGY_PALETTE)
METRIC_PALETTE: list[str] = list(_DARK_METRIC_PALETTE)

# Current active style name
_active_style: str = 'dark'

AVAILABLE_STYLES = ('dark', 'paper')


def get_strategy_colors(names: list[str]) -> dict[str, str]:
    """Assign consistent colors to strategy names."""
    return {
        name: STRATEGY_PALETTE[i % len(STRATEGY_PALETTE)]
        for i, name in enumerate(names)
    }


def get_metric_colors(names: list[str]) -> dict[str, str]:
    """Assign consistent colors to metric names."""
    return {
        name: METRIC_PALETTE[i % len(METRIC_PALETTE)]
        for i, name in enumerate(names)
    }


def apply_style(style: str = 'dark'):
    """Load a matplotlib style and update the active palettes.

    Args:
        style: 'dark' for OLDarkTheme or 'paper' for publication-ready.
    """
    global _active_style

    style_files = {
        'dark': 'ol_analysis.mplstyle',
        'paper': 'ol_analysis_paper.mplstyle',
    }
    palettes = {
        'dark': (_DARK_STRATEGY_PALETTE, _DARK_METRIC_PALETTE),
        'paper': (_PAPER_STRATEGY_PALETTE, _PAPER_METRIC_PALETTE),
    }

    if style not in style_files:
        raise ValueError(
            f"Unknown style '{style}'. Available: {', '.join(AVAILABLE_STYLES)}"
        )

    style_path = Path(__file__).parent / style_files[style]
    if style_path.exists():
        plt.style.use(str(style_path))

    strat, met = palettes[style]
    STRATEGY_PALETTE[:] = strat
    METRIC_PALETTE[:] = met
    _active_style = style
