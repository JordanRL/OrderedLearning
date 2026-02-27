"""Predictive coding state objects for hook contexts.

Carries PC model access and per-layer error/activation information
for the local learning paradigm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PredictiveCodingModelState:
    """Model access for predictive coding training.

    Carries the PC model plus settled activations and layer errors
    from the last training step, enabling hooks to inspect layer-level
    dynamics without re-running inference.
    """
    model: nn.Module
    lr: float | None = None
    settled_activations: list[torch.Tensor] | None = None
    layer_errors: list[torch.Tensor] | None = None
    free_energy: float | None = None

    @property
    def device(self) -> torch.device:
        """Device the model lives on."""
        return next(self.model.parameters()).device


@dataclass(frozen=True)
class PredictiveCodingGradientState:
    """Gradient access for predictive coding training.

    Per-layer local gradients computed by the Hebbian rule, not by autograd.
    """
    layer_weight_grads: list[torch.Tensor] | None = None
    layer_error_norms: list[float] | None = None
