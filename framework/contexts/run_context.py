"""RunContext: frozen, run-agnostic lifecycle context passed to all hooks.

Contains only information about WHERE we are in the run — nothing about
what's being trained or how. Paradigm-specific state (model, gradients,
metrics, batch data) is passed separately via **state kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..hooks.hook_point import HookPoint
from .profiler import CheckpointProfiler


@dataclass(frozen=True)
class RunContext:
    """Run-agnostic lifecycle position. The only thing hooks always receive.

    Paradigm-specific state (model, gradients, eval metrics, batch data)
    is passed to hooks as separate **state kwargs by the HookManager.
    """
    hook_point: HookPoint
    epoch: int
    step: int | None = None
    config: object | None = None
    profiler: CheckpointProfiler | None = None
