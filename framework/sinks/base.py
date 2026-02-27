"""Sink base classes and shared formatting helpers.

Defines the MetricSink ABC and the FilePathSink base for sinks that
write to auto-resolved file paths (CSV, JSONL).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..hooks.hook_point import HookPoint


# --- Formatting helpers used by multiple sinks ---

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


# --- Base classes ---

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


class FilePathSink(MetricSink):
    """Base for sinks that write to auto-resolved file paths.

    Handles two modes:
    - Fixed path: filepath provided directly
    - Auto path: deferred to set_run_context(strategy=...) with collision avoidance
    """

    _file_extension: str  # subclasses must set this

    def __init__(
        self,
        filepath: str | Path | None = None,
        output_dir: str | None = None,
        experiment_name: str | None = None,
    ):
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
                f"{self.__class__.__name__} requires either filepath or "
                "(output_dir + experiment_name)"
            )
        self._file = None

    def _resolve_path(self, strategy: str) -> Path | None:
        """Resolve a collision-free path for the given strategy.

        Returns the resolved Path, or None if not in auto mode.
        """
        if not self._auto_mode:
            return None
        experiment_dir = Path(self._output_dir) / self._experiment_name / strategy
        base_path = experiment_dir / f"{strategy}.{self._file_extension}"
        if not base_path.exists():
            return base_path
        num = 1
        while True:
            candidate = experiment_dir / f"{strategy}_{num}.{self._file_extension}"
            if not candidate.exists():
                return candidate
            num += 1

    def _close_file(self):
        """Flush and close the current file handle if open."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def set_run_context(self, **kwargs):
        """Resolve per-strategy path and reset subclass state."""
        self._close_file()
        strategy = kwargs.get('strategy')
        if strategy:
            resolved = self._resolve_path(strategy)
            if resolved:
                self._filepath = resolved
        self._on_new_context(**kwargs)

    def _on_new_context(self, **kwargs):
        """Override for subclass-specific reset on context change."""
        pass

    def flush(self):
        self._close_file()
