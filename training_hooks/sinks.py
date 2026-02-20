"""
Metric sinks that consume hook output.

Sinks receive scalar metrics from hooks and route them to different
destinations (console, CSV, JSONL, etc.). The HookManager dispatches to all
registered sinks after each hook firing.
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from console import OLConsole
from .base import HookPoint


def _flatten_for_csv(value: Any) -> str:
    """Flatten a non-scalar value to a CSV-safe string using semicolons.

    Dicts become ``key:value;key:value``, lists become ``val;val;val``,
    and scalars pass through as-is.
    """
    if isinstance(value, dict):
        return ';'.join(f'{k}:{v}' for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ';'.join(str(v) for v in value)
    return value


def _format_metric_value(value: Any, compact: bool = False) -> str:
    """Format a metric value for console display.

    Args:
        value: The metric value (float, int, list, etc.).
        compact: If True, use shorter format (no n= suffix for lists).
    """
    if isinstance(value, (list, tuple)):
        n = len(value)
        if n > 0 and isinstance(value[0], (int, float)):
            mean = sum(value) / n
            formatted = _format_number(mean)
            if not compact:
                formatted += f" (n={n})"
            return formatted
        return f"[{n} items]"
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


def _format_number(value: float) -> str:
    """Format a numeric value with appropriate precision."""
    if abs(value) < 0.001 or abs(value) > 10000:
        return f"{value:.4e}"
    return f"{value:.6f}"


class MetricSink(ABC):
    """Base class for metric output destinations."""

    @abstractmethod
    def emit(self, metrics: dict[str, Any], epoch: int, hook_point: HookPoint):
        """Receive metrics from a hook firing.

        Args:
            metrics: Namespaced metric dict (e.g., "norms/total_norm": 1.23).
            epoch: Current training epoch.
            hook_point: Which lifecycle point produced these metrics.
        """
        ...

    def set_run_context(self, **kwargs):
        """Signal a new run context (e.g., strategy change).

        Called by HookManager when the training loop starts a new logical
        run within the same experiment. Sinks that need per-run separation
        (like W&B) should override this. Default is a no-op.

        Args:
            **kwargs: Context key-value pairs (e.g., strategy='stride').
        """
        pass

    def flush(self):
        """Flush any buffered output. Called at end of training."""
        pass


class ConsoleSink(MetricSink):
    """Display hook metrics in a Rich table via OLConsole.

    Buffers non-SNAPSHOT metrics and displays them alongside SNAPSHOT
    metrics when a SNAPSHOT fires, so the console isn't spammed every epoch.

    In LIVE mode with live_metrics configured, routes selected metrics to
    a persistent side column that updates on every emit, organized by
    group with trend indicators.
    """

    # Trend indicator characters and their theme styles
    _TREND_UP = '▲'
    _TREND_DOWN = '▼'
    _TREND_FLAT = '━'
    _TREND_STYLES = {
        'up': 'metric.improved',
        'down': 'metric.degraded',
        'flat': 'detail',
    }
    # Relative change threshold below which a metric is considered flat
    _TREND_THRESHOLD = 0.01  # 1%

    def __init__(self, live_metrics=None):
        """
        Args:
            live_metrics: Optional grouped dict mapping group names to
                {display_label: metric_key} dicts. Example::

                    {'Basic': {'Loss': 'training_metrics/loss'},
                     'Gradients': {'Total Norm': 'norms/total_norm'}}

                When set and console is in LIVE mode, metrics are rendered
                in a persistent side column organized by group.
        """
        self._console = OLConsole()
        self._buffered: dict[str, Any] = {}
        self._live_metrics = live_metrics or {}
        self._col_idx = None  # Lazy-created column index for LIVE mode
        # LIVE mode persistent state
        self._current_values: dict[str, Any] = {}  # latest value per metric key
        self._metric_history: dict[str, list[float]] = {}  # last 3 values per key
        self._trend_cache: dict[str, str] = {}  # last rendered trend per key

    def emit(self, metrics: dict[str, Any], epoch: int, hook_point: HookPoint):
        if not metrics:
            return

        if self._console.is_live and self._live_metrics:
            # LIVE mode: accumulate values and update sidebar on every emit
            self._current_values.update(metrics)
            self._emit_to_column(updated_keys=set(metrics.keys()))
        else:
            # NORMAL mode: buffer until SNAPSHOT, then print full table
            if hook_point != HookPoint.SNAPSHOT:
                self._buffered.update(metrics)
                return
            combined = {**self._buffered, **metrics}
            self._buffered = {}
            self._emit_to_main(combined)

    def _emit_to_main(self, combined: dict[str, Any]):
        """Render full metrics table in the main panel (NORMAL mode).

        Per-parameter metrics (keys with 3+ segments like
        ``hook/metric/param.name``) are excluded from console display
        since they produce hundreds of rows that are only useful in
        logged data (CSV/JSONL/W&B) for graphing.
        """
        from rich.table import Table
        from rich import box

        # Filter: keep aggregate metrics (hook/metric), skip per-parameter (hook/metric/param)
        # Hook metrics always have exactly 1 slash; per-parameter have 2+
        display_metrics = {
            k: v for k, v in combined.items()
            if k.count('/') < 2
        }
        n_hidden = len(combined) - len(display_metrics)

        if not display_metrics:
            return

        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="table.header",
            title="Hook Metrics",
            title_style="detail",
            padding=(0, 1),
        )
        table.add_column("Hook", style="hook.name")
        table.add_column("Metric")
        table.add_column("Value", justify="right")

        for key in sorted(display_metrics.keys()):
            parts = key.split("/", 1)
            hook_name = parts[0] if len(parts) > 1 else ""
            metric_name = parts[1] if len(parts) > 1 else parts[0]
            formatted = _format_metric_value(display_metrics[key])
            table.add_row(hook_name, metric_name, formatted)

        self._console.print(table)

        if n_hidden > 0:
            self._console.print(
                f"  [detail]({n_hidden} per-parameter metrics hidden — "
                f"see CSV/JSONL logs for full data)[/detail]"
            )

    def _emit_to_column(self, updated_keys: set[str]):
        """Render grouped metrics in a persistent side column (LIVE mode)."""
        from rich.console import Group
        from rich.table import Table
        from rich.rule import Rule
        from rich import box

        # Lazy-create the column on first emit
        if self._col_idx is None:
            self._col_idx = self._console.add_column_to_main(width=50)
            if self._col_idx is None:
                return

        renderables = []
        for group_name, group_metrics in self._live_metrics.items():
            # Skip groups where no metric has data yet
            if not any(k in self._current_values for k in group_metrics.values()):
                continue

            renderables.append(
                Rule(f"[bold]{group_name}[/bold]", style="panel.info")
            )

            table = Table(
                box=box.SIMPLE,
                show_header=False,
                padding=(0, 1),
                expand=True,
            )
            table.add_column("Label", style="label", no_wrap=True)
            table.add_column("Value", justify="right", style="metric.value")
            table.add_column("", width=3)

            for label, metric_key in group_metrics.items():
                if metric_key not in self._current_values:
                    continue
                value = self._current_values[metric_key]
                formatted = _format_metric_value(value, compact=True)

                # Only update trend for freshly emitted metrics
                if metric_key in updated_keys:
                    trend = self._update_history_and_trend(metric_key, value)
                    self._trend_cache[metric_key] = trend
                else:
                    trend = self._trend_cache.get(metric_key, "")

                table.add_row(label, formatted, trend)

            renderables.append(table)

        if renderables:
            self._console.update_column_content(
                self._col_idx, Group(*renderables)
            )

    def _update_history_and_trend(self, key: str, value: Any) -> str:
        """Track metric history and return a styled trend indicator string.

        Compares the last 3 scalar values to produce up to 2 trend characters
        showing the direction of change between consecutive measurements.
        Skips trend for integer values (indices, counts, codes).
        """
        # Reduce lists to their mean
        if isinstance(value, (list, tuple)):
            if len(value) > 0 and isinstance(value[0], (int, float)):
                value = sum(value) / len(value)
            else:
                return ""
        # Skip non-numeric and integer values (indices, counts, codes)
        if not isinstance(value, (int, float)):
            return ""
        if isinstance(value, int):
            return ""

        history = self._metric_history.setdefault(key, [])
        history.append(float(value))
        if len(history) > 3:
            history[:] = history[-3:]

        if len(history) < 2:
            return ""

        # Build trend string from consecutive pairs
        parts = []
        for i in range(len(history) - 1):
            prev, curr = history[i], history[i + 1]
            direction = self._classify_change(prev, curr)
            char = {
                'up': self._TREND_UP,
                'down': self._TREND_DOWN,
                'flat': self._TREND_FLAT,
            }[direction]
            style = self._TREND_STYLES[direction]
            parts.append(f"[{style}]{char}[/{style}]")

        return "".join(parts)

    @classmethod
    def _classify_change(cls, prev: float, curr: float) -> str:
        """Classify the change between two values as 'up', 'down', or 'flat'."""
        if prev == 0:
            return 'flat' if curr == 0 else 'up' if curr > 0 else 'down'
        relative = (curr - prev) / abs(prev)
        if abs(relative) < cls._TREND_THRESHOLD:
            return 'flat'
        return 'up' if relative > 0 else 'down'

    def set_run_context(self, **kwargs):
        """Reset metric state on strategy change."""
        self._metric_history.clear()
        self._current_values.clear()
        self._trend_cache.clear()


class CSVSink(MetricSink):
    """Append hook metrics to a CSV file.

    Writes incrementally on each emit. If a new column appears (e.g., when
    SNAPSHOT hooks fire for the first time after POST_EPOCH hooks have been
    writing), the file is rewritten with the expanded header.

    Two modes:
    - Fixed path: ``CSVSink(filepath='path/to/file.csv')``
    - Auto path: ``CSVSink(output_dir='output', experiment_name='presorted')``
      Defers path resolution to ``set_run_context(strategy=...)``, producing
      ``{output_dir}/{experiment_name}/{strategy}/{strategy}.csv`` with
      collision avoidance via numeric suffixes.
    """

    def __init__(
        self,
        filepath: str | Path | None = None,
        output_dir: str | None = None,
        experiment_name: str | None = None,
    ):
        """
        Args:
            filepath: Fixed path to the output CSV file.
            output_dir: Base output directory for auto-path mode.
            experiment_name: Experiment name for auto-path mode.
        """
        if filepath is not None:
            self._filepath = Path(filepath)
            self._auto_mode = False
        elif output_dir is not None and experiment_name is not None:
            self._filepath = None
            self._output_dir = output_dir
            self._experiment_name = experiment_name
            self._auto_mode = True
        else:
            raise ValueError(
                "CSVSink requires either filepath or (output_dir + experiment_name)"
            )
        self._fieldnames: list[str] = []
        self._rows: list[dict] = []
        self._file = None
        self._writer = None

    def _open(self):
        """Open the CSV file and write the header."""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._filepath, 'w', newline='')
        self._writer = csv.DictWriter(
            self._file, fieldnames=self._fieldnames, restval='',
        )
        self._writer.writeheader()

    def _rewrite(self):
        """Rewrite the entire file with the current fieldnames and rows."""
        if self._file is not None:
            self._file.close()
        self._open()
        for row in self._rows:
            self._writer.writerow(self._flatten_row(row))
        self._file.flush()

    @staticmethod
    def _flatten_row(row: dict) -> dict:
        """Flatten non-scalar values so every cell is CSV-safe."""
        return {k: _flatten_for_csv(v) for k, v in row.items()}

    def set_run_context(self, **kwargs):
        """Resolve per-strategy CSV path in auto-path mode."""
        # Close any currently-open file and reset state for new strategy
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
        self._fieldnames = []
        self._rows = []

        if not self._auto_mode:
            return

        strategy = kwargs.get('strategy')
        if not strategy:
            return

        # Compute experiment dir: {output_dir}/{experiment_name}/{strategy}/
        experiment_dir = Path(self._output_dir) / self._experiment_name / strategy

        # Resolve filename with collision avoidance
        base_path = experiment_dir / f"{strategy}.csv"
        if not base_path.exists():
            self._filepath = base_path
        else:
            num = 1
            while True:
                candidate = experiment_dir / f"{strategy}_{num}.csv"
                if not candidate.exists():
                    self._filepath = candidate
                    break
                num += 1

    def emit(self, metrics: dict[str, Any], epoch: int, hook_point: HookPoint):
        if not metrics:
            return
        if self._filepath is None:
            return

        row = {"epoch": epoch, "hook_point": hook_point.name, **metrics}
        self._rows.append(row)

        # Check if any new columns appeared
        new_keys = [k for k in row if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
            # Rewrite the whole file with the expanded header
            self._rewrite()
        else:
            # Append incrementally
            if self._writer is None:
                self._open()
            self._writer.writerow(self._flatten_row(row))
            self._file.flush()

    def flush(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None


class JSONLSink(MetricSink):
    """Append hook metrics as JSON Lines (one JSON object per line).

    Handles nested/structured values natively — dicts, lists, and scalars
    are all written as proper JSON types. No column management needed.

    Two modes:
    - Fixed path: ``JSONLSink(filepath='path/to/file.jsonl')``
    - Auto path: ``JSONLSink(output_dir='output', experiment_name='presorted')``
      Defers path resolution to ``set_run_context(strategy=...)``, producing
      ``{output_dir}/{experiment_name}/{strategy}/{strategy}.jsonl`` with
      collision avoidance via numeric suffixes.
    """

    def __init__(
        self,
        filepath: str | Path | None = None,
        output_dir: str | None = None,
        experiment_name: str | None = None,
    ):
        """
        Args:
            filepath: Fixed path to the output .jsonl file.
            output_dir: Base output directory for auto-path mode.
            experiment_name: Experiment name for auto-path mode.
        """
        if filepath is not None:
            self._filepath = Path(filepath)
            self._auto_mode = False
        elif output_dir is not None and experiment_name is not None:
            self._filepath = None
            self._output_dir = output_dir
            self._experiment_name = experiment_name
            self._auto_mode = True
        else:
            raise ValueError(
                "JSONLSink requires either filepath or (output_dir + experiment_name)"
            )
        self._file = None

    def _ensure_open(self):
        if self._filepath is None:
            return
        if self._file is None:
            self._filepath.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._filepath, 'a', newline='')

    def set_run_context(self, **kwargs):
        """Resolve per-strategy JSONL path in auto-path mode."""
        # Close any currently-open file
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

        if not self._auto_mode:
            return

        strategy = kwargs.get('strategy')
        if not strategy:
            return

        # Compute experiment dir: {output_dir}/{experiment_name}/{strategy}/
        experiment_dir = Path(self._output_dir) / self._experiment_name / strategy

        # Resolve filename with collision avoidance
        base_path = experiment_dir / f"{strategy}.jsonl"
        if not base_path.exists():
            self._filepath = base_path
        else:
            num = 1
            while True:
                candidate = experiment_dir / f"{strategy}_{num}.jsonl"
                if not candidate.exists():
                    self._filepath = candidate
                    break
                num += 1

    def emit(self, metrics: dict[str, Any], epoch: int, hook_point: HookPoint):
        if not metrics:
            return

        self._ensure_open()
        if self._file is None:
            return
        record = {"epoch": epoch, "hook_point": hook_point.name, **metrics}
        self._file.write(json.dumps(record, default=_json_default) + '\n')
        self._file.flush()

    def flush(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


class WandbSink(MetricSink):
    """Log hook metrics to Weights & Biases.

    Creates one W&B run per strategy within a shared group, so runs from
    the same experiment are grouped together but each strategy is tracked
    independently with its own metric curves.

    Metrics are logged as ``hook_point/hook_name/metric_name`` with
    ``epoch`` as the x-axis step.

    Requires ``wandb`` to be installed.
    """

    def __init__(
        self,
        project: str | None = None,
        group: str | None = None,
        config: dict | None = None,
    ):
        """
        Args:
            project: W&B project name.
            group: Experiment group name. If None, auto-generated from
                   a timestamp so all strategy runs are grouped together.
            config: Experiment config dict logged with every run.
        """
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "WandbSink requires the 'wandb' package. "
                "Install it with: pip install wandb"
            )
        self._wandb = wandb
        self._project = project
        self._config = config or {}
        if group is None:
            import datetime
            group = f"experiment_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        self._group = group

    def set_run_context(self, **kwargs):
        # Finish previous run if any
        if self._wandb.run is not None:
            self._wandb.finish()

        run_config = {**self._config, **kwargs}
        # Use the strategy (or other context) as the run name
        run_name = kwargs.get('strategy', None)

        self._wandb.init(
            project=self._project,
            group=self._group,
            name=run_name,
            config=run_config,
            reinit=True,
            settings=self._wandb.Settings(console="off"),
        )

    def emit(self, metrics: dict[str, Any], epoch: int, hook_point: HookPoint):
        if not metrics:
            return
        if self._wandb.run is None:
            return

        logged = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, bool)) or hasattr(sub_value, 'item'):
                        logged[f'{key}/{sub_key}'] = self._to_scalar(sub_value)
            elif isinstance(value, (list, tuple)):
                # Accumulated step metrics — log as histogram + scalar summary
                scalars = [self._to_scalar(v) for v in value]
                if scalars and isinstance(scalars[0], (int, float)):
                    logged[key] = self._wandb.Histogram(scalars)
                    logged[f"{key}_mean"] = sum(scalars) / len(scalars)
                else:
                    logged[key] = scalars
            elif isinstance(value, (int, float, bool)) or hasattr(value, 'item'):
                logged[key] = self._to_scalar(value)
        if logged:
            self._wandb.log(logged, step=epoch)

    @staticmethod
    def _to_scalar(value: Any):
        """Coerce to a Python scalar for W&B logging."""
        if hasattr(value, 'item'):
            return value.item()
        return value

    def flush(self):
        if self._wandb.run is not None:
            self._wandb.finish()


def _json_default(obj):
    """JSON serializer fallback for types json.dumps doesn't handle natively."""
    if hasattr(obj, 'item'):
        # numpy/torch scalar
        return obj.item()
    if hasattr(obj, 'tolist'):
        # numpy/torch array
        return obj.tolist()
    return str(obj)
