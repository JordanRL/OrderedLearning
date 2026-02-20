"""
Utility functions for gradient analysis.

Contains streaming utilities, flatten functions, and common constants.
"""

import gc
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Iterator, Tuple, Optional

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Try to import scipy
try:
    from scipy.ndimage import gaussian_filter1d
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    gaussian_filter1d = None
    scipy_stats = None

# Standard colors for each model type
MODEL_COLORS = {
    'stride': 'blue',
    'random': 'red',
    'target': 'green'
}


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

# Try to import rich for progress bars
try:
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn,
        MofNCompleteColumn
    )
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


class ProgressContext:
    """
    Context manager for Rich-based progress tracking.

    Provides a unified interface for progress bars that works with or without Rich.
    Supports nested progress (overall analyses + per-snapshot progress).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and RICH_AVAILABLE
        self.progress = None
        self._main_task = None
        self._sub_task = None

    def __enter__(self):
        if self.enabled:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
                expand=False,
            )
            self.progress.__enter__()
        return self

    def __exit__(self, *args):
        if self.progress:
            self.progress.__exit__(*args)

    def add_main_task(self, description: str, total: int):
        """Add the main (outer) progress task."""
        if self.enabled:
            self._main_task = self.progress.add_task(description, total=total)
        return self._main_task

    def add_sub_task(self, description: str, total: int):
        """Add a sub-task (inner) progress bar."""
        if self.enabled:
            self._sub_task = self.progress.add_task(f"  {description}", total=total)
        return self._sub_task

    def update_main(self, advance: int = 1, description: str = None):
        """Update the main task progress."""
        if self.enabled and self._main_task is not None:
            kwargs = {'advance': advance}
            if description:
                kwargs['description'] = description
            self.progress.update(self._main_task, **kwargs)

    def update_sub(self, advance: int = 1, description: str = None):
        """Update the sub-task progress."""
        if self.enabled and self._sub_task is not None:
            kwargs = {'advance': advance}
            if description:
                kwargs['description'] = f"  {description}"
            self.progress.update(self._sub_task, **kwargs)

    def remove_sub_task(self):
        """Remove the current sub-task (when done)."""
        if self.enabled and self._sub_task is not None:
            self.progress.remove_task(self._sub_task)
            self._sub_task = None

    def reset_sub_task(self, description: str, total: int):
        """Reset sub-task for a new iteration."""
        self.remove_sub_task()
        return self.add_sub_task(description, total)


# Global progress context (set during main execution)
_progress_ctx: Optional[ProgressContext] = None


def get_progress_context() -> Optional[ProgressContext]:
    """Get the current progress context."""
    return _progress_ctx


def set_progress_context(ctx: Optional[ProgressContext]):
    """Set the global progress context."""
    global _progress_ctx
    _progress_ctx = ctx


def log_progress(message: str):
    """Print progress message only if Rich progress bars are not active."""
    ctx = get_progress_context()
    if ctx is None or not ctx.enabled:
        print(message)


# =============================================================================
# STREAMING UTILITIES
# =============================================================================

def iter_snapshots(snapshot_dir: str, desc: str = "Processing snapshots") -> Iterator[Tuple[int, int, dict]]:
    """
    Iterate over snapshots, yielding one at a time.

    Yields:
        (index, total, snapshot_dict)

    Memory: Only one snapshot in memory at a time.
    Updates progress context if available.
    """
    snapshot_dir = Path(snapshot_dir)
    snapshot_files = sorted(snapshot_dir.glob('snapshot_*.pt'))
    total = len(snapshot_files)

    # Set up progress tracking
    ctx = get_progress_context()
    if ctx:
        ctx.reset_sub_task(desc, total)

    for i, path in enumerate(snapshot_files):
        snapshot = torch.load(path, map_location='cpu')
        yield i, total, snapshot

        # Update progress
        if ctx:
            ctx.update_sub(advance=1)

        del snapshot  # Explicit cleanup

    # Clean up sub-task
    if ctx:
        ctx.remove_sub_task()


def iter_snapshots_sync(snapshot_dirs: Dict[str, str], desc: str = "Processing snapshots") -> Iterator[Tuple[int, int, Dict[str, dict]]]:
    """
    Iterate over multiple snapshot directories in sync.

    Args:
        snapshot_dirs: Dict mapping model names to directory paths
                      e.g., {'stride': './traj_stride', 'random': './traj_random', ...}

    Yields:
        (index, total, {model_name: snapshot_dict})

    Memory: Only one snapshot per model at a time.
    Updates progress context if available.
    """
    dirs = {name: Path(d) for name, d in snapshot_dirs.items()}
    files = {name: sorted(d.glob('snapshot_*.pt')) for name, d in dirs.items()}

    # Verify counts
    lengths = {name: len(f) for name, f in files.items()}
    if len(set(lengths.values())) > 1:
        if not get_progress_context():  # Only print if not using progress bars
            print(f"Warning: Mismatched snapshot counts: {lengths}")

    total = min(lengths.values())

    # Set up progress tracking
    ctx = get_progress_context()
    if ctx:
        ctx.reset_sub_task(desc, total)

    for i in range(total):
        snapshots = {}
        for name, file_list in files.items():
            snapshots[name] = torch.load(file_list[i], map_location='cpu')

        yield i, total, snapshots

        # Update progress
        if ctx:
            ctx.update_sub(advance=1)

        # Explicit cleanup
        for s in snapshots.values():
            del s
        del snapshots

    # Clean up sub-task
    if ctx:
        ctx.remove_sub_task()


def iter_snapshots_async(snapshot_dirs: Dict[str, str], desc: str = "Processing snapshots") -> Iterator[Tuple[int, int, Dict[str, dict], List[str]]]:
    """
    Iterate over multiple snapshot directories asynchronously.

    Unlike iter_snapshots_sync, this continues iteration even when some models
    run out of snapshots. This is useful for comparing models that trained for
    different durations.

    Args:
        snapshot_dirs: Dict mapping model names to directory paths
                      e.g., {'stride': './traj_stride', 'random': './traj_random', ...}

    Yields:
        (index, total, {model_name: snapshot_dict}, available_model_names)

        The available_model_names list indicates which models have data at this index.
        Models are dropped once they run out of snapshots.

    Memory: Only one snapshot per model at a time.
    Updates progress context if available.
    """
    dirs = {name: Path(d) for name, d in snapshot_dirs.items()}
    files = {name: sorted(d.glob('snapshot_*.pt')) for name, d in dirs.items()}

    # Report counts
    lengths = {name: len(f) for name, f in files.items()}
    if len(set(lengths.values())) > 1:
        if not get_progress_context():
            print(f"    Asymmetric snapshot counts: {lengths}")

    # Total is the maximum count (we'll iterate through all)
    total = max(lengths.values()) if lengths else 0

    # Set up progress tracking
    ctx = get_progress_context()
    if ctx:
        ctx.reset_sub_task(desc, total)

    for i in range(total):
        snapshots = {}
        available_models = []

        for name, file_list in files.items():
            if i < len(file_list):
                snapshots[name] = torch.load(file_list[i], map_location='cpu')
                available_models.append(name)

        if snapshots:  # Only yield if we have at least one model
            yield i, total, snapshots, available_models

        # Update progress
        if ctx:
            ctx.update_sub(advance=1)

        # Explicit cleanup
        for s in snapshots.values():
            del s
        del snapshots

    # Clean up sub-task
    if ctx:
        ctx.remove_sub_task()


def load_final_params(snapshot_dir: str) -> dict:
    """Load just the final params from a snapshot directory."""
    path = Path(snapshot_dir) / 'final_params.pt'
    if not path.exists():
        raise FileNotFoundError(f"final_params.pt not found in {snapshot_dir}")
    return torch.load(path, map_location='cpu')


def flatten_grads(grads: Dict[str, torch.Tensor], exclude_bias: bool = True) -> torch.Tensor:
    """Flatten gradient dictionary to single vector."""
    grad_list = []
    for name, grad in sorted(grads.items()):  # Sort for consistency
        if exclude_bias and 'bias' in name:
            continue
        grad_list.append(grad.view(-1).float())
    return torch.cat(grad_list)


def flatten_params(params: Dict[str, torch.Tensor], exclude_bias: bool = True) -> torch.Tensor:
    """Flatten parameter dictionary to single vector."""
    param_list = []
    for name, param in sorted(params.items()):
        if exclude_bias and 'bias' in name:
            continue
        param_list.append(param.view(-1).float())
    return torch.cat(param_list)
