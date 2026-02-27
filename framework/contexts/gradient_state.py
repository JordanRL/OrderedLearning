"""Gradient state objects for hook contexts.

Paradigm-specific frozen dataclasses that carry gradient information.
The HookManager passes these to hooks as state kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BackpropGradientState:
    """Gradient access for backprop training."""
    accumulated_grads: dict[str, torch.Tensor] | None = None
    prev_step_grads: dict[str, torch.Tensor] | None = None
    target_grad: dict[str, torch.Tensor] | None = None
