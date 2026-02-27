"""Profiler for hook and checkpoint operations."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import torch

from console import OLConsole


class CheckpointProfiler:
    """Profiler for hook and checkpoint operations.

    When enabled, times each section with CUDA synchronization to get
    accurate wall-clock measurements. Reports cumulative stats.

    Usage::

        profiler = CheckpointProfiler(enabled=True)
        with profiler.section("save_model"):
            ...
        profiler.report()  # prints timing summary
    """

    def __init__(self, enabled: bool = False, device: torch.device | None = None):
        self.enabled = enabled
        self._device = device
        self._timings: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        if self._device is not None and self._device.type == 'cuda':
            torch.cuda.synchronize(self._device)
        start = time.perf_counter()
        yield
        if self._device is not None and self._device.type == 'cuda':
            torch.cuda.synchronize(self._device)
        self._timings[name].append(time.perf_counter() - start)

    def report(self) -> dict[str, dict[str, float]]:
        """Return and print timing summary.

        Returns:
            Dict of section_name -> {count, total_ms, mean_ms}.
        """
        console = OLConsole()
        summary = {}
        for name, times in sorted(self._timings.items()):
            total = sum(times) * 1000
            count = len(times)
            mean = total / count
            summary[name] = {'count': count, 'total_ms': total, 'mean_ms': mean}
            console.print(
                f"  [label]{name}:[/label] {count}x, "
                f"total [metric.value]{total:.1f}ms[/metric.value], "
                f"mean [metric.value]{mean:.1f}ms[/metric.value]"
            )
        return summary

    def reset(self):
        self._timings.clear()
