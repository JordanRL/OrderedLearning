"""Console sink for Rich-based metric display."""

from __future__ import annotations

from typing import Any

from console import OLConsole
from ..hooks.hook_point import HookPoint
from .base import MetricSink, _format_metric_value


class ConsoleSink(MetricSink):
    """Display hook metrics in a Rich table via OLConsole.

    Buffers non-SNAPSHOT metrics and displays them alongside SNAPSHOT
    metrics when a SNAPSHOT fires, so the console isn't spammed every epoch.

    In LIVE mode with live_metrics configured, routes selected metrics to
    a persistent side column that updates on every emit, organized by
    group with trend indicators.
    """

    # Trend indicator characters and their theme styles
    _TREND_UP = '\u25b2'
    _TREND_DOWN = '\u25bc'
    _TREND_FLAT = '\u2501'
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
                f"  [detail]({n_hidden} per-parameter metrics hidden \u2014 "
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
