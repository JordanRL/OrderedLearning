"""Meta-learning state objects for hook contexts.

Carries meta-model access and per-task adaptation information
for the NESTED paradigm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MetaLearningModelState:
    """Model access for meta-learning training.

    The model field is the meta-model (the model being meta-optimized).
    inner_lr is the inner-loop learning rate used for task adaptation.
    """
    model: nn.Module
    lr: float | None = None
    inner_lr: float | None = None

    @property
    def device(self) -> torch.device:
        """Device the meta-model lives on."""
        return next(self.model.parameters()).device


@dataclass(frozen=True)
class MetaLearningGradientState:
    """Gradient access for meta-learning training.

    Meta-gradients are the gradients of the meta-loss w.r.t. the
    meta-model parameters. Per-task inner_gradients are optional —
    strategies cache them if available.
    """
    meta_gradients: dict[str, torch.Tensor] | None = None
    inner_gradients: list[dict[str, torch.Tensor]] | None = None
    task_losses: list[float] | None = None
