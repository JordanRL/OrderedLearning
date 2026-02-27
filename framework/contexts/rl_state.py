"""Reinforcement learning state objects for hook contexts.

Carries actor-critic model access for RL paradigm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RLModelState:
    """Model access for actor-critic RL training."""
    actor: nn.Module
    critic: nn.Module
    lr: float | dict[str, float] | None = None

    @property
    def model(self) -> nn.Module:
        """Primary model (actor) for hooks that expect a single model."""
        return self.actor

    @property
    def device(self) -> torch.device:
        """Device the actor lives on."""
        return next(self.actor.parameters()).device


@dataclass(frozen=True)
class RLGradientState:
    """Gradient access for RL training."""
    policy_gradients: dict[str, torch.Tensor] | None = None
    value_gradients: dict[str, torch.Tensor] | None = None
