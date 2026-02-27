"""Batch/data state for hook contexts.

Generic (paradigm-agnostic) frozen dataclass that carries current batch
and data loader information. The HookManager passes this to hooks as
a state kwarg.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BatchState:
    """Current batch and data source access."""
    loader: Any = None
    batch_idx: int | None = None
    batch_data: Any = None
