"""Data loading for analysis tools.

Discovers strategy directories under output/{experiment}/, loads JSONL or CSV
metric files into DataFrames, normalizes column names, merges rows at the
same (step, strategy), and concatenates all strategies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_experiment_data(
    experiment_name: str,
    output_dir: str = 'output',
    strategies: list[str] | None = None,
) -> pd.DataFrame:
    """Load and merge metric data from all strategies of an experiment.

    Args:
        experiment_name: Name of the experiment directory.
        output_dir: Base output directory (default: 'output').
        strategies: If provided, only load these strategies. Otherwise
            discovers all strategy directories.

    Returns:
        DataFrame with columns: step, strategy, + metric columns.
        Sorted by (strategy, step).

    Raises:
        FileNotFoundError: If no metric data is found.
    """
    experiment_path = Path(output_dir) / experiment_name

    if not experiment_path.is_dir():
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_path}\n"
            f"Run the experiment first, or check --output-dir."
        )

    # Discover strategy directories (exclude 'analysis')
    if strategies:
        strategy_dirs = [
            experiment_path / s for s in strategies
            if (experiment_path / s).is_dir()
        ]
    else:
        strategy_dirs = [
            d for d in sorted(experiment_path.iterdir())
            if d.is_dir() and d.name != 'analysis' and not d.name.startswith('.')
        ]

    if not strategy_dirs:
        raise FileNotFoundError(
            f"No strategy directories found in {experiment_path}/\n"
            f"Re-run the experiment with --hook-csv or --hook-jsonl to export metrics."
        )

    all_frames = []
    for strat_dir in strategy_dirs:
        strategy_name = strat_dir.name
        df = _load_strategy_data(strat_dir)
        if df is not None and not df.empty:
            df['strategy'] = strategy_name
            all_frames.append(df)

    if not all_frames:
        raise FileNotFoundError(
            f"No metric data found in {experiment_path}/\n"
            f"Re-run with --hook-csv or --hook-jsonl to export metrics."
        )

    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values(['strategy', 'step']).reset_index(drop=True)
    return result


def _load_strategy_data(strat_dir: Path) -> pd.DataFrame | None:
    """Load metric data from a single strategy directory.

    Prefers *.jsonl over *.csv. Normalizes column names and merges
    rows at the same step.
    """
    # Prefer JSONL
    jsonl_files = sorted(strat_dir.glob('*.jsonl'))
    csv_files = sorted(strat_dir.glob('*.csv'))

    if jsonl_files:
        df = _load_jsonl(jsonl_files[0])
    elif csv_files:
        df = _load_csv(csv_files[0])
    else:
        return None

    if df is None or df.empty:
        return None

    # Normalize: rename 'epoch' -> 'step' if 'step' column doesn't exist
    if 'step' not in df.columns and 'epoch' in df.columns:
        df = df.rename(columns={'epoch': 'step'})

    # Drop hook_point column (not useful for analysis)
    if 'hook_point' in df.columns:
        df = df.drop(columns=['hook_point'])

    # Collapse list-valued cells to their mean
    df = _collapse_lists(df)

    # Merge rows at same step: different hook_points may produce different
    # metrics at the same step. groupby().last() keeps the latest value
    # for each column at each step.
    if 'step' in df.columns:
        non_step_cols = [c for c in df.columns if c != 'step']
        df = df.groupby('step', as_index=False)[non_step_cols].last()

    return df


def _load_jsonl(path: Path) -> pd.DataFrame | None:
    """Load a JSONL file into a DataFrame."""
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return None
    return pd.DataFrame(records)


def _load_csv(path: Path) -> pd.DataFrame | None:
    """Load a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _collapse_lists(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse list-valued cells to their mean.

    JSONL files may contain list values (accumulated step metrics).
    Convert them to their scalar mean for analysis.
    """
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(_list_to_mean)
    return df


def _list_to_mean(value):
    """Convert a list of numbers to its mean, pass through scalars."""
    if isinstance(value, list):
        nums = [v for v in value if isinstance(v, (int, float))]
        if nums:
            return np.mean(nums)
        return np.nan
    return value


def load_experiment_config(
    experiment_name: str,
    output_dir: str = 'output',
) -> dict | None:
    """Load the first experiment_config.json found for the experiment.

    Searches strategy directories under output/{experiment}/ for
    experiment_config.json and returns the first one found.

    Returns:
        Config dict, or None if not found.
    """
    experiment_path = Path(output_dir) / experiment_name

    if not experiment_path.is_dir():
        return None

    for strat_dir in sorted(experiment_path.iterdir()):
        if not strat_dir.is_dir() or strat_dir.name == 'analysis':
            continue
        config_path = strat_dir / 'experiment_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)

    return None
