"""Model state objects for hook contexts.

Paradigm-specific frozen dataclasses that carry model access information.
The HookManager passes these to hooks as state kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BackpropModelState:
    """Model access for single-model backprop training."""
    model: nn.Module
    lr: float | None = None

    @property
    def device(self) -> torch.device:
        """Device the model lives on."""
        return next(self.model.parameters()).device
